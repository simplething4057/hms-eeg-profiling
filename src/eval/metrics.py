"""평가 지표 (코드북 5장)

- KL Divergence : 대회 지표. 전체 / 고득표 샘플을 함께 보고하고 모델 선택은 고득표 기준
- 축별 AUC : axis_mask = 1 샘플만 사용
- Seizure 민감도 @ 특이도 : 고득표 샘플, y_seizure_bin 기준
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-15


def kl_divergence(y_true, y_pred, eps=EPS):
    """행별 KL(y_true || y_pred)의 평균. y_pred는 확률 합이 1이 되도록 정규화한다."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(t > 0, t * np.log(t / p), 0.0)
    return float(term.sum(axis=1).mean())


def axis_auc(axis_true, axis_score, mask):
    """축 정답(확률)을 0.5 기준 이진화해 AUC 계산. 한 클래스만 있으면 NaN."""
    m = np.asarray(mask).astype(bool)
    y = (np.asarray(axis_true)[m] >= 0.5).astype(int)
    s = np.asarray(axis_score)[m]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def sensitivity_at_specificity(y_true, score, target_spec=0.90):
    """특이도 >= target_spec 조건에서 가능한 최대 민감도와 그 임계값."""
    fpr, tpr, thr = roc_curve(y_true, score)
    ok = fpr <= (1 - target_spec)
    if not ok.any():
        return float("nan"), float("nan")
    i = np.argmax(np.where(ok, tpr, -1))
    return float(tpr[i]), float(thr[i])


def predicted_axes(prob):
    """6클래스 예측 확률(seizure, lpd, gpd, lrda, grda, other 순) → 예측 편측·주기성 확률."""
    p = np.asarray(prob, dtype=np.float64)
    four = p[:, 1:5].sum(axis=1) + EPS
    lat = (p[:, 1] + p[:, 3]) / four
    pd_ = (p[:, 1] + p[:, 2]) / four
    return lat, pd_


def evaluate(df, prob, target_spec=0.90):
    """build_targets 결과 df와 예측 확률로 코드북 지표를 한 번에 계산한다."""
    from src.data.targets import PROB_COLS

    y = df[PROB_COLS].to_numpy()
    hv = df["high_vote"].to_numpy().astype(bool)
    lat, pd_ = predicted_axes(prob)
    sens, thr = sensitivity_at_specificity(
        df.loc[hv, "y_seizure_bin"], np.asarray(prob)[hv, 0], target_spec
    ) if hv.any() and df.loc[hv, "y_seizure_bin"].nunique() == 2 else (float("nan"), float("nan"))
    return {
        "kl_all": kl_divergence(y, prob),
        "kl_high": kl_divergence(y[hv], np.asarray(prob)[hv]) if hv.any() else float("nan"),
        "auc_lat": axis_auc(df["axis_lat"], lat, df["axis_mask"]),
        "auc_pd": axis_auc(df["axis_pd"], pd_, df["axis_mask"]),
        f"sens_at_spec{int(target_spec * 100)}": sens,
        "seizure_threshold": thr,
        "n_all": int(len(df)),
        "n_high": int(hv.sum()),
    }
