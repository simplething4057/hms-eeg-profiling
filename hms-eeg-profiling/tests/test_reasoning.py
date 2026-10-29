"""R1 : 임계값 학습, 판정-근거 일치 판정, 리포트 문구."""
import numpy as np
import pandas as pd

from src.reasoning.report import format_report, make_report
from src.reasoning.rules import FEATURES, check_claims, fit_thresholds, youden_threshold


def test_youden_separates_groups():
    t = youden_threshold(np.r_[5, 6, 7, 8], np.r_[1, 2, 3, 4])
    assert 4 <= t <= 5


def _toy():
    rng = np.random.default_rng(0)
    rows = []
    for cls, lat, rda, sz in [("lpd", 1, 0, 0), ("lrda", 1, 1, 0), ("gpd", 0, 0, 0), ("grda", 0, 1, 0),
                              ("seizure", 0, 0, 1), ("other", 0, 0, 0)]:
        for _ in range(50):
            rows.append({"expert_consensus": cls.upper() if cls != "seizure" else "Seizure",
                         "abs_asym_total": lat * 0.4 + rng.normal(0.1, 0.03), "abs_asym_delta": lat * 0.4 + rng.normal(0.1, 0.03),
                         "delta_peak_sharp_max": rda * 1.0 + rng.normal(2.5, 0.1), "rel_delta": rda * 0.1 + rng.normal(0.75, 0.02),
                         "center_ctx_power_absmax": sz * 0.3 + rng.normal(0.2, 0.05), "center_ctx_centroid_absmax": sz * 0.5 + rng.normal(0.5, 0.1),
                         "asym_total": lat * 0.4, "focal_region": 0})
    return pd.DataFrame(rows)


def test_claims_consistent_when_prediction_matches_truth():
    df = _toy()
    thr = fit_thresholds(df)
    assert set(thr) == set(FEATURES)
    res = check_claims(df, df.expert_consensus, thr)
    checkable = res.status != "판정 대상 아님"
    assert (res.loc[checkable, "status"] == "일치").mean() > 0.95
    assert (res.loc[df.expert_consensus == "OTHER", "status"] == "판정 대상 아님").all()


def test_claims_flag_wrong_lateral_prediction():
    """실제 전반(GPD) 구간을 편측(LPD)으로 판정하면 편측성 근거가 없어 불일치 플래그."""
    df = _toy()
    thr = fit_thresholds(df)
    gpd = df[df.expert_consensus == "GPD"]
    res = check_claims(gpd, ["lpd"] * len(gpd), thr)
    assert (res.status == "불일치").mean() > 0.9


def test_report_recheck_and_low_confidence_flag():
    df = _toy()
    thr = fit_thresholds(df)
    row = df[df.expert_consensus == "GPD"].iloc[0]                  # 실제 전반 구간
    sure = make_report(row, np.array([0.05, 0.7, 0.1, 0.05, 0.05, 0.05]), thr, p_low=0.47)   # LPD 70% 판정
    text = format_report(sure)
    for key in ["[소견]", "[판정]", "[근거]", "[신뢰도]"]:
        assert key in text
    assert "주기성 : 정량 근거 미확정" in text
    assert "[재확인] 편측·전반 판단 재확인" in text                   # 편측 근거 없음 → 재확인 문구
    assert sure["플래그"] == ""                                       # 확신 0.70 ≥ 0.47 → 플래그 없음
    unsure = make_report(row, np.array([0.05, 0.4, 0.3, 0.1, 0.1, 0.05]), thr, p_low=0.47)
    assert unsure["플래그"].startswith("전문의 우선 확인") and "낮음" in unsure["신뢰도"]


def test_seizure_axis_is_info_only():
    df = _toy()
    thr = fit_thresholds(df)
    row = df[df.expert_consensus == "OTHER"].iloc[0]                # 시간 변화 근거 없음
    r = make_report(row, np.array([0.8, 0.04, 0.04, 0.04, 0.04, 0.04]), thr, p_low=0.47)
    assert "[참고 : 판별력 미확인]" in format_report(r) and r["재확인"] == []


def test_stricter_rule_flags_more():
    df = _toy()
    thr = fit_thresholds(df)
    noisy = df.copy()
    noisy.loc[noisy.index[::3], "abs_asym_delta"] = 0.0                  # 편측 근거 1개만 남는 구간
    lpd = noisy[noisy.expert_consensus == "LPD"]
    loose = check_claims(lpd, ["lpd"] * len(lpd), thr, min_hits=1)
    strict = check_claims(lpd, ["lpd"] * len(lpd), thr, min_hits=2)
    assert (strict.status == "불일치").sum() > (loose.status == "불일치").sum()
    dropped = check_claims(lpd, ["lpd"] * len(lpd), thr, min_hits=2, drop=("abs_asym_delta",))
    assert (dropped.status == "불일치").sum() <= (strict.status == "불일치").sum()
