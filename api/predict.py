"""E8 앙상블 추론 엔진 — 싱글턴 ModelServer + 배치 지원

구조:
  Stage 1 (QC) : signal_qc.qc_stats — API 입력에서 사전 계산 또는 생략
  Stage 2 (P1) : LightGBM 5-fold 평균 (항상 실행)
  Stage 3 (DL) : EfficientNet-B0 5-fold 평균 × 2 모델 (스펙트로그램 있을 때)
  Stage 4 (E8) : 0.85 × DL_mean + 0.15 × P1_mean  (DL 없으면 P1 단독)
  Stage 5 (R1b): 저확신(p_max < P_LOW) 또는 근거 불일치 → flag=True
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
RECIPE_PATH = ROOT / "e8_recipe.json"

CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
N_FOLDS = 5
P_LOW = 0.47        # R1b 저확신 기준 (nb06, 고득표 최고 확률 하위 20%)
DL_WEIGHT = 0.85    # e8_recipe.json


# ── R1b 플래그 로직 ────────────────────────────────────────────────

def _r1b_flag(prob: np.ndarray, qeeg: np.ndarray | None = None) -> tuple[bool, str]:
    """(6,) 확률 → (flag, reason).

    판정 기준:
    1. 저확신: max_prob < P_LOW
    2. 편측 축 불일치: LPD/LRDA 판정인데 좌우 비대칭 지표가 낮음 (P1 피처 인덱스 기준)
    3. 율동 축 불일치: LRDA/GRDA 판정인데 델타 율동 지표가 낮음
    qeeg 없으면 저확신만 판정.
    """
    p_max = float(prob.max())
    pred_idx = int(prob.argmax())
    pred_cls = CLASSES[pred_idx]

    if p_max < P_LOW:
        return True, f"저확신 (max_prob={p_max:.3f} < {P_LOW})"

    if qeeg is None:
        return False, "없음"

    # P1 메타에서 인덱스 조회 (사전에 로드된 x_cols 기준)
    # abs_asym_total: 편측성, rel_delta: 율동성 대리 지표
    # 실제 임계값은 학습 fold에서 추출하나 여기서는 P_LOW 기반 경험치 사용
    try:
        meta = ModelServer.instance().p1_meta
        xcols = meta.get("x_cols", [])
        asym_idx = next((i for i, c in enumerate(xcols) if "asym_total" in c), None)
        delta_idx = next((i for i, c in enumerate(xcols) if "rel_delta" == c), None)

        if pred_cls in ("lpd", "lrda") and asym_idx is not None:
            if float(qeeg[asym_idx]) < 0.15:   # 편측 기대 클래스인데 비대칭 낮음
                return True, f"편측 근거 불일치 (abs_asym_total={qeeg[asym_idx]:.3f})"

        if pred_cls in ("lrda", "grda") and delta_idx is not None:
            if float(qeeg[delta_idx]) < 0.35:   # 율동 기대 클래스인데 델타 파워 낮음
                return True, f"율동 근거 불일치 (rel_delta={qeeg[delta_idx]:.3f})"
    except Exception as e:  # noqa: BLE001
        logger.debug("R1b 근거 불일치 검사 건너뜀: %s", e)

    return False, "없음"


# ── ModelServer ────────────────────────────────────────────────────

class ModelServer:
    _inst: "ModelServer | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self._dl_models: list | None = None   # [(E4_fold0, ...), (E5_fold0, ...)]
        self._p1_models: list | None = None   # [lgb.Booster × 5]
        self.p1_meta: dict = {}
        self._recipe: dict = {}
        self._dl_loaded = False
        self._p1_loaded = False
        self._load_p1()
        self._try_load_dl()

    @classmethod
    def instance(cls) -> "ModelServer":
        if cls._inst is None:
            with cls._lock:
                if cls._inst is None:
                    cls._inst = cls()
        return cls._inst

    # ── 로딩 ──────────────────────────────────────────────────────

    def _load_p1(self) -> None:
        try:
            import lightgbm as lgb  # noqa: PLC0415
        except ImportError:
            logger.warning("lightgbm 미설치 — P1 비활성화")
            return
        models = []
        for fold in range(N_FOLDS):
            pt = MODEL_DIR / "P1" / f"fold{fold}.txt"
            if not pt.exists():
                logger.warning("P1 fold%d 없음: %s", fold, pt)
                return
            models.append(lgb.Booster(model_file=str(pt)))
        meta_path = MODEL_DIR / "P1" / "p1_meta.json"
        if meta_path.exists():
            self.p1_meta = json.loads(meta_path.read_text())
        self._p1_models = models
        self._p1_loaded = True
        logger.info("P1 로드 완료 (%d folds)", N_FOLDS)

        if RECIPE_PATH.exists():
            self._recipe = json.loads(RECIPE_PATH.read_text())

    def _try_load_dl(self) -> None:
        try:
            import timm  # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError:
            logger.info("timm/torch 미설치 — DL 비활성화 (P1 단독 모드)")
            return
        try:
            from src.models.eeg_spec import HMSNet  # noqa: PLC0415
        except ImportError:
            logger.warning("src.models 임포트 실패 — DL 비활성화")
            return

        all_models = []
        for model_name in ("E4", "E5"):
            folds = []
            for fold in range(N_FOLDS):
                pt = MODEL_DIR / model_name / f"fold{fold}.pt"
                if not pt.exists():
                    logger.warning("%s fold%d 없음", model_name, fold)
                    return
                backbone = timm.create_model(
                    "efficientnet_b0", pretrained=False, num_classes=6, in_chans=1
                )
                net = HMSNet(backbone, input_mode="both")
                ckpt = torch.load(str(pt), map_location="cpu", weights_only=True)
                state = ckpt.get("model_state_dict", ckpt)
                net.backbone.load_state_dict(state, strict=True)
                net.eval()
                folds.append(net)
            all_models.append(folds)

        self._dl_models = all_models
        self._dl_loaded = True
        logger.info("DL (E4+E5) 로드 완료")

    # ── 추론 ──────────────────────────────────────────────────────

    def _infer_p1(self, qeeg: np.ndarray) -> np.ndarray:
        """(67,) → (6,) 확률"""
        assert self._p1_loaded and self._p1_models
        preds = []
        feat = qeeg.reshape(1, -1).astype(np.float32)
        for bst in self._p1_models:
            preds.append(bst.predict(feat)[0])   # (6,)
        return np.array(preds).mean(axis=0)

    def _infer_dl(self, spec_arr: np.ndarray) -> np.ndarray:
        """(400, 556) float32 → (6,) 확률"""
        assert self._dl_loaded and self._dl_models
        import torch  # noqa: PLC0415
        x = torch.from_numpy(spec_arr).unsqueeze(0).unsqueeze(0)  # (1,1,400,556)
        model_probs = []
        with torch.no_grad():
            for folds in self._dl_models:          # E4, E5
                fold_logits = []
                for net in folds:
                    # HMSNet.forward(spec=x) — 스펙트로그램 직접 입력
                    out = net.backbone(x)          # (1, 6)
                    fold_logits.append(torch.softmax(out, dim=1)[0].numpy())
                model_probs.append(np.array(fold_logits).mean(axis=0))
        return np.array(model_probs).mean(axis=0)  # E4+E5 평균 (6,)

    def predict(
        self,
        qeeg: np.ndarray,
        spec_b64: str | None = None,
    ) -> dict:
        """
        Returns:
            prob (np.ndarray, shape (6,)), flag (bool), reason (str), model_used (str)
        """
        t0 = time.perf_counter()

        if not self._p1_loaded:
            raise RuntimeError("P1 모델이 로드되지 않았습니다.")

        p1_prob = self._infer_p1(qeeg)

        model_used = "p1_only"
        final_prob = p1_prob

        if spec_b64 is not None and self._dl_loaded:
            try:
                raw = base64.b64decode(spec_b64)
                spec_arr = np.frombuffer(raw, dtype=np.float32).reshape(400, 556)
                dl_prob = self._infer_dl(spec_arr)
                final_prob = DL_WEIGHT * dl_prob + (1 - DL_WEIGHT) * p1_prob
                model_used = "e8_ensemble"
            except Exception as e:  # noqa: BLE001
                logger.warning("DL 추론 실패, P1 단독 사용: %s", e)

        # softmax 정규화 (LightGBM softmax 출력이므로 이미 합=1이지만 앙상블 후 재정규화)
        final_prob = final_prob / final_prob.sum()

        flag, reason = _r1b_flag(final_prob, qeeg)
        ms = (time.perf_counter() - t0) * 1000

        return {
            "prob": final_prob.tolist(),
            "flag": flag,
            "reason": reason,
            "model_used": model_used,
            "ms": ms,
        }

    @property
    def dl_loaded(self) -> bool:
        return self._dl_loaded

    @property
    def p1_loaded(self) -> bool:
        return self._p1_loaded

    @property
    def recipe(self) -> dict:
        return self._recipe
