"""타깃 생성 (코드북 4-2)

- y_prob : 클래스별 득표 수 / 총 득표 수 (주 학습 타깃)
- axis_lat / axis_pd : 4개 패턴 클래스 득표로 계산한 편측·주기성 확률
- axis_mask : 4개 패턴 득표가 0이면 축 정답이 정의되지 않으므로 0 (보조 손실·축 AUC에서 제외)
- y_seizure_bin : Seizure 득표 비율 >= 기준 (평가는 고득표 샘플에서만)
"""
import numpy as np
import pandas as pd

CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTE_COLS = [f"{c}_vote" for c in CLASSES]
PROB_COLS = [f"p_{c}" for c in CLASSES]
PATTERN_VOTES = ["lpd_vote", "gpd_vote", "lrda_vote", "grda_vote"]


def build_targets(df, high_vote=10, seizure_pos=0.5):
    """train.csv 형식 DataFrame에 타깃 컬럼을 추가해 새 DataFrame으로 반환한다."""
    out = df.copy()
    out["n_votes"] = out[VOTE_COLS].sum(axis=1)
    if (out["n_votes"] <= 0).any():
        raise ValueError("득표 수가 0인 행이 있습니다. 원본 데이터를 확인하세요.")
    out[PROB_COLS] = out[VOTE_COLS].div(out["n_votes"], axis=0).to_numpy()
    out["high_vote"] = (out["n_votes"] >= high_vote).astype(int)

    four = out[PATTERN_VOTES].sum(axis=1)
    out["axis_mask"] = (four > 0).astype(int)
    denom = four.where(four > 0)  # 0이면 NaN으로 두어 정의되지 않음을 명시
    out["axis_lat"] = (out["lpd_vote"] + out["lrda_vote"]) / denom
    out["axis_pd"] = (out["lpd_vote"] + out["gpd_vote"]) / denom

    out["y_seizure_bin"] = (out["p_seizure"] >= seizure_pos).astype(int)
    return out


def aggregate_by_eeg(df):
    """평가용 eeg_id 단위 집계 : 같은 기록의 겹치는 구간이 평가에 중복 반영되지 않도록 투표를 합산한다."""
    agg = df.groupby("eeg_id").agg(
        patient_id=("patient_id", "first"),
        n_segments=("eeg_id", "size"),
        **{c: (c, "sum") for c in VOTE_COLS},
    ).reset_index()
    return build_targets(agg)


def select_representative(df):
    """eeg_id마다 학습·저장용 대표 구간 1개를 고른다.

    규칙 : 득표 수가 가장 많은 구간 → 동률이면 offset 중앙값에 가장 가까운 구간.
    타깃은 그 구간 자신의 투표를 사용한다 (같은 기록 안에서도 라벨이 바뀔 수 있어 신호와 라벨을 맞추기 위함).
    """
    d = df.copy()
    med = d.groupby("eeg_id")["eeg_label_offset_seconds"].transform("median")
    d["_dist"] = (d["eeg_label_offset_seconds"] - med).abs()
    d["_seg_count"] = d.groupby("eeg_id")["eeg_id"].transform("size")
    d = d.sort_values(["eeg_id", "n_votes", "_dist"], ascending=[True, False, True])
    rep = d.drop_duplicates("eeg_id", keep="first").rename(columns={"_seg_count": "n_segments"})
    return rep.drop(columns=["_dist"]).reset_index(drop=True)
