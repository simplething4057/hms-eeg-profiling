"""환자 단위 교차검증 분할

같은 환자가 학습·검증에 동시에 들어가면 성능이 낙관적으로 추정되므로 patient_id를 그룹으로 쓴다.
"""
import numpy as np
from sklearn.model_selection import GroupKFold


def make_folds(df, n_folds=5, seed=42, group_col="patient_id"):
    """df에 fold 컬럼(0 ~ n_folds-1)을 추가해 반환한다. 시드로 환자 순서를 섞어 재현 가능하게 만든다."""
    out = df.copy()
    patients = out[group_col].unique()
    rng = np.random.default_rng(seed)
    order = {p: i for i, p in enumerate(rng.permutation(patients))}
    groups = out[group_col].map(order)

    out["fold"] = -1
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(out, groups=groups)):
        out.iloc[va, out.columns.get_loc("fold")] = k
    return out


def check_no_leak(df, group_col="patient_id"):
    """한 환자가 두 개 이상의 fold에 걸치면 AssertionError."""
    spread = df.groupby(group_col)["fold"].nunique()
    assert (spread == 1).all(), f"fold 누수 환자 {int((spread > 1).sum())}명"
    return True
