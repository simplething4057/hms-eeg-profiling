import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.folds import check_no_leak, make_folds
from src.data.montage import CHANNEL_NAMES, double_banana, flip_lr
from src.data.targets import PROB_COLS, aggregate_by_eeg, build_targets
from src.eval.metrics import (axis_auc, evaluate, kl_divergence,
                              sensitivity_at_specificity)

ELECTRODES = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
              "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]


def toy_train():
    rows = [
        # eeg_id, patient, seizure, lpd, gpd, lrda, grda, other
        (1, 10, 3, 0, 0, 0, 0, 9),   # 패턴 득표 없음 → axis_mask 0
        (1, 10, 0, 8, 1, 2, 0, 1),   # LPD 우세
        (2, 11, 10, 1, 0, 0, 0, 1),  # Seizure 우세, 고득표
        (3, 12, 0, 0, 1, 0, 9, 2),   # GRDA 우세
        (4, 13, 1, 0, 6, 0, 0, 0),
        (5, 14, 0, 0, 0, 5, 0, 0),
    ]
    cols = ["eeg_id", "patient_id", "seizure_vote", "lpd_vote", "gpd_vote",
            "lrda_vote", "grda_vote", "other_vote"]
    return pd.DataFrame(rows, columns=cols)


def test_build_targets_definitions():
    t = build_targets(toy_train(), high_vote=10)
    assert np.allclose(t[PROB_COLS].sum(axis=1), 1)
    assert t.loc[0, "axis_mask"] == 0 and np.isnan(t.loc[0, "axis_lat"])
    assert t.loc[1, "axis_lat"] == pytest.approx(10 / 11)   # (LPD 8 + LRDA 2) / 11
    assert t.loc[1, "axis_pd"] == pytest.approx(9 / 11)     # (LPD 8 + GPD 1) / 11
    assert t.loc[2, "y_seizure_bin"] == 1 and t.loc[2, "high_vote"] == 1
    assert t.loc[4, "high_vote"] == 0


def test_aggregate_by_eeg_sums_votes():
    agg = aggregate_by_eeg(toy_train())
    row = agg[agg.eeg_id == 1].iloc[0]
    assert row["n_segments"] == 2 and row["n_votes"] == 24


def test_zero_votes_raise():
    df = toy_train()
    df.loc[0, ["seizure_vote", "other_vote"]] = 0
    with pytest.raises(ValueError):
        build_targets(df)


def test_montage_shape_and_flip():
    eeg = pd.DataFrame(np.random.default_rng(0).normal(size=(400, 20)), columns=ELECTRODES)
    x = double_banana(eeg)
    assert x.shape == (16, 400) and len(CHANNEL_NAMES) == 16
    assert np.allclose(x[0], eeg["Fp1"] - eeg["F7"])
    assert np.allclose(flip_lr(x)[0], x[4])          # LL 첫 채널 ↔ RL 첫 채널
    assert np.allclose(flip_lr(flip_lr(x)), x)       # 두 번 뒤집으면 원래대로


def test_folds_have_no_patient_leak():
    df = pd.DataFrame({"patient_id": np.repeat(np.arange(20), 5)})
    f = make_folds(df, n_folds=5, seed=1)
    assert set(f["fold"]) == set(range(5)) and check_no_leak(f)
    assert (make_folds(df, 5, 1)["fold"] == f["fold"]).all()  # 재현성


def test_kl_perfect_and_uniform():
    y = np.array([[0.7, 0, 0, 0, 0, 0.3], [0, 1, 0, 0, 0, 0]])
    assert kl_divergence(y, y) == pytest.approx(0, abs=1e-9)
    uni = np.full_like(y, 1 / 6)
    expected = np.mean([0.7 * np.log(0.7 * 6) + 0.3 * np.log(0.3 * 6), np.log(6)])
    assert kl_divergence(y, uni) == pytest.approx(expected)


def test_axis_auc_and_sensitivity():
    assert axis_auc([0.9, 0.8, 0.1, 0.2, np.nan], [0.9, 0.7, 0.2, 0.1, 0.5], [1, 1, 1, 1, 0]) == 1.0
    y = np.array([0] * 10 + [1] * 10)
    s = np.r_[np.linspace(0, 0.5, 10), np.linspace(0.6, 1, 10)]
    sens, _ = sensitivity_at_specificity(y, s, 0.9)
    assert sens == 1.0


def test_evaluate_runs_end_to_end():
    t = build_targets(toy_train(), high_vote=5)
    m = evaluate(t, t[PROB_COLS].to_numpy())
    assert m["kl_all"] == pytest.approx(0, abs=1e-9) and m["n_high"] == 6


def test_config_loads():
    cfg = load_config()
    assert cfg["labels"]["high_vote"] == 10 and cfg["n_folds"] == 5
