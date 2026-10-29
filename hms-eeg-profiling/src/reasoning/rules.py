"""R1 추론 로직 : 딥러닝 판정과 프로파일 근거의 일관성 검증

- 근거 지표는 H5에서 지지(Cohen's d ≥ 0.5)된 지표만 사용한다 (코드북 4-4, 20차 수정 100).
- 임계값은 학습 fold의 고득표 구간에서만 정한다 (코드북 5장 : 규칙 임계값 누수 방지).
- 주기성(PD) 근거는 H5에서 확정되지 않아 '근거 미확정'으로 처리한다.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# 판독 축 : 근거 지표와 임계값 학습에 쓰는 집단 (pos = 지표가 높게 기대되는 집단, neg = 비교 집단)
AXES = {
    "lateral": {"name": "편측성", "features": ["abs_asym_total", "abs_asym_delta"],
                "pos": ["lpd", "lrda"], "neg": ["gpd", "grda"]},
    "rhythmic": {"name": "율동성(델타)", "features": ["delta_peak_sharp_max", "rel_delta"],
                 "pos": ["lrda", "grda"], "neg": ["lpd", "gpd"]},
    "seizure": {"name": "시간 변화(진화)", "features": ["center_ctx_power_absmax", "center_ctx_centroid_absmax"],
                "pos": ["seizure"], "neg": ["lpd", "gpd", "lrda", "grda", "other"]},
}
# 판정 클래스별 주장 : (축, 기대 방향). 주기성은 근거 미확정이라 주장에서 제외 (리포트에만 표시)
CLAIMS = {
    "seizure": [("seizure", "high")],
    "lpd": [("lateral", "high")],
    "gpd": [("lateral", "low")],
    "lrda": [("lateral", "high"), ("rhythmic", "high")],
    "grda": [("lateral", "low"), ("rhythmic", "high")],
    "other": [],
}
PERIODIC = {"lpd", "gpd"}
CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
FEATURES = [f for ax in AXES.values() for f in ax["features"]]


def youden_threshold(pos, neg):
    """pos 집단이 높다고 볼 때 Youden J(민감도 + 특이도 − 1)가 최대인 임계값."""
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[np.asarray(pos, float), np.asarray(neg, float)]
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i]) if np.isfinite(thr[i]) else float(np.median(s))


def fit_thresholds(df):
    """df : 학습 fold의 고득표 구간 (expert_consensus 열 필요) → {지표: 임계값}."""
    cls = df["expert_consensus"].str.lower()
    out = {}
    for ax in AXES.values():
        pos, neg = df[cls.isin(ax["pos"])], df[cls.isin(ax["neg"])]
        for f in ax["features"]:
            out[f] = youden_threshold(pos[f], neg[f])
    return out


def check_claims(df, pred_class, thr, min_hits=1, drop=()):
    """구간별 판정(pred_class)의 주장마다 근거 지표 충족 수를 세고 일치 여부를 판정한다.

    반환 열 : claim_<축>(충족 지표 수, 해당 없음 = -1), status('일치'·'불일치'·'판정 대상 아님'), n_claims
    - 주장 1개의 근거 지표 중 min_hits개 이상 충족 → 그 주장은 지지 (기본 1개, 진단용으로 2개 가능)
    - drop : 근거에서 뺄 지표 (진단용, 예 : 임계값이 불안정한 지표)
    - 모든 주장이 지지 → 일치 / 하나라도 0개 → 불일치 / 주장 없음(Other) → 판정 대상 아님
    """
    pred_class = pd.Series(np.asarray(pred_class), index=df.index).str.lower()
    out = pd.DataFrame(index=df.index)
    for ax_key, ax in AXES.items():
        out[f"claim_{ax_key}"] = -1
    supported_all = pd.Series(True, index=df.index)
    n_claims = pd.Series(0, index=df.index)
    for c, claims in CLAIMS.items():
        m = pred_class == c
        if not m.any():
            continue
        for ax_key, direction in claims:
            feats = [f for f in AXES[ax_key]["features"] if f not in drop]
            hits = sum(((df.loc[m, f] > thr[f]) if direction == "high" else (df.loc[m, f] <= thr[f])).astype(int)
                       for f in feats)
            out.loc[m, f"claim_{ax_key}"] = hits
            supported_all.loc[m] &= hits >= min(min_hits, len(feats))
            n_claims.loc[m] += 1
    out["n_claims"] = n_claims
    out["status"] = np.where(n_claims == 0, "판정 대상 아님", np.where(supported_all, "일치", "불일치"))
    return out
