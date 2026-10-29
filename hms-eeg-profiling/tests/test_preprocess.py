import numpy as np
import pandas as pd
import pytest

from src.data.build import build_dataset, process_one
from src.data.filters import bandpass, interp_nan
from src.data.load import eeg_window, find_data_dir, spec_window
from src.data.targets import build_targets, select_representative
from src.quality.signal_qc import qc_stats

ELECTRODES = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
              "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
FREQS = np.round(np.linspace(0.59, 19.92, 100), 2)


@pytest.fixture
def fake_hms(tmp_path):
    """HMS 구조를 흉내 낸 소형 데이터 (competitions/ 하위에 둬서 실제 마운트 구조도 함께 검증)."""
    root = tmp_path / "input" / "competitions" / "hms"
    (root / "train_eegs").mkdir(parents=True)
    (root / "train_spectrograms").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for e in range(6):
        eid, sid = 100 + e, 900 + e
        n_sec = 80 if e != 5 else 55  # 마지막 기록은 offset+50초보다 짧게 → 패딩 검증
        t = np.arange(n_sec * 200) / 200
        eeg = pd.DataFrame(rng.normal(0, 20, (len(t), 20)).astype("float32"), columns=ELECTRODES)
        eeg["Fp1"] += 50 * np.sin(2 * np.pi * 2 * t)          # 2Hz 성분 (통과 대역)
        eeg["F7"] += 50 * np.sin(2 * np.pi * 50 * t)          # 50Hz 성분 (차단 대역)
        if e == 1:
            eeg.loc[100:400, "C3"] = np.nan
        if e == 2:
            eeg["O2"] = np.nan
        eeg.to_parquet(root / "train_eegs" / f"{eid}.parquet")
        tt = np.arange(1, 900, 2)
        spec = {"time": tt}
        for r in ["LL", "RL", "LP", "RP"]:
            for f in FREQS:
                spec[f"{r}_{f}"] = rng.random(len(tt)) * 100
        pd.DataFrame(spec).to_parquet(root / "train_spectrograms" / f"{sid}.parquet")
        for k, off in enumerate([0, 10, 20][: 1 + e % 3]):
            v = rng.multinomial(int(rng.integers(3, 15)), [.2, .15, .15, .1, .1, .3])
            rows.append(dict(eeg_id=eid, eeg_sub_id=k, eeg_label_offset_seconds=float(off if e != 5 else 10),
                             spectrogram_id=sid, spectrogram_sub_id=k, spectrogram_label_offset_seconds=float(off),
                             label_id=len(rows), patient_id=e % 4, expert_consensus="Other",
                             **{c: int(x) for c, x in zip(["seizure_vote", "lpd_vote", "gpd_vote",
                                                           "lrda_vote", "grda_vote", "other_vote"], v)}))
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    return tmp_path / "input", root


def test_find_data_dir_nested(fake_hms, monkeypatch):
    monkeypatch.delenv("HMS_DATA_DIR", raising=False)
    root_input, root = fake_hms
    assert find_data_dir(root_input) == root


def test_interp_and_bandpass():
    x = np.vstack([np.r_[1.0, np.nan, 3.0, 4.0], np.full(4, np.nan)]).astype(np.float32)
    y = interp_nan(x)
    assert y[0, 1] == pytest.approx(2.0) and (y[1] == 0).all()
    t = np.arange(10000) / 200
    sig = (np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 50 * t))[None, :]
    out = bandpass(sig)
    spec = np.abs(np.fft.rfft(out[0]))
    f = np.fft.rfftfreq(10000, 1 / 200)
    assert spec[np.argmin(abs(f - 2))] > 50 * spec[np.argmin(abs(f - 50))]  # 2Hz 유지, 50Hz 제거


def test_windows_pad_and_shape(fake_hms):
    _, root = fake_hms
    eeg = pd.read_parquet(root / "train_eegs" / "105.parquet")
    w = eeg_window(eeg, 10, sec=50)
    assert len(w) == 10000 and w.iloc[-1].isna().all()
    spec = pd.read_parquet(root / "train_spectrograms" / "900.parquet")
    s = spec_window(spec, 0)
    assert s.shape == (4, 100, 300) and np.isfinite(s).all()


def test_qc_flags():
    x = np.random.default_rng(0).normal(size=(16, 1000))
    assert qc_stats(x)["qc_flag"] == 0
    x[:5] = 0
    q = qc_stats(x)
    assert q["n_flat"] == 5 and q["qc_flag"] == 1


def test_select_representative_rule():
    df = pd.DataFrame({"eeg_id": [1, 1, 1, 2], "eeg_label_offset_seconds": [0, 10, 20, 0],
                       "n_votes": [3, 9, 9, 5]})
    rep = select_representative(df)
    assert len(rep) == 2
    assert rep.loc[rep.eeg_id == 1, "eeg_label_offset_seconds"].item() == 10  # 동률 9표 중 중앙값(10)에 가까운 구간
    assert rep.loc[rep.eeg_id == 1, "n_segments"].item() == 3


def test_build_dataset_end_to_end(fake_hms, tmp_path):
    _, root = fake_hms
    train = build_targets(pd.read_csv(root / "train.csv"))
    meta = select_representative(train)
    out = build_dataset(meta, root, tmp_path / "out", n_jobs=1, chunk=4)
    eeg = np.load(tmp_path / "out" / "eeg50.npy", mmap_mode="r")
    spec = np.load(tmp_path / "out" / "spec10.npy", mmap_mode="r")
    assert eeg.shape == (6, 16, 10000) and spec.shape == (6, 4, 100, 300)
    assert eeg.dtype == np.float16 and np.isfinite(eeg).all() and np.isfinite(spec).all()
    assert len(out) == 6 and {"nan_ratio", "n_flat", "qc_flag"} <= set(out.columns)
    assert out.loc[out.eeg_id == 102, "n_flat"].item() >= 1   # O2 전체 결측 → 평탄 채널
    assert out.loc[out.eeg_id == 105, "nan_ratio"].item() > 0  # 짧은 기록 패딩
