"""hms-processed 데이터셋 생성

저장 구조 (/kaggle/working/hms_processed/) : 파일 수 제한(약 500개)을 피하려고 단일 배열 파일로 저장
- eeg50.npy  : (N, 16, 10000) float16  더블 바나나 몽타주, 결측 보간 + 0.5~20Hz 필터
- spec10.npy : (N, 4, 100, 300) float16  대회 제공 10분 스펙트로그램, log1p 변환 (float16 범위 초과 방지)
- meta.csv   : 배열 행 순서와 같은 순서의 eeg_id, 타깃, fold, 품질 지표
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.filters import bandpass, interp_nan
from src.data.load import eeg_window, spec_window
from src.data.montage import PAIRS
from src.quality.signal_qc import qc_stats

AMP_CLIP = 3000.0  # float16 저장 전 극단값 제한 (µV)


def eeg_segment(eeg, offset_sec, fs=200, band=(0.5, 20.0)):
    """EEG parquet DataFrame → (50초 몽타주 float16 (16, 10000), qc dict). 학습(nb02)과 제출(nb05)이 같은 함수를 쓴다."""
    seg = eeg_window(eeg, offset_sec, fs=fs, sec=50)
    raw = np.stack([seg[a].to_numpy() - seg[b].to_numpy() for a, b in PAIRS]).astype(np.float32)
    qc = qc_stats(raw)
    x = bandpass(interp_nan(raw), fs=fs, low=band[0], high=band[1])
    return np.clip(x, -AMP_CLIP, AMP_CLIP).astype(np.float16), qc


def process_one(row, data_dir, fs=200, band=(0.5, 20.0)):
    """meta 한 행 → (eeg50, spec10, qc dict)."""
    data_dir = Path(data_dir)
    eeg = pd.read_parquet(data_dir / "train_eegs" / f"{int(row.eeg_id)}.parquet")
    x, qc = eeg_segment(eeg, row.eeg_label_offset_seconds, fs=fs, band=band)

    spec = pd.read_parquet(data_dir / "train_spectrograms" / f"{int(row.spectrogram_id)}.parquet")
    s = spec_window(spec, row.spectrogram_label_offset_seconds)
    s = np.log1p(np.clip(s, 0, None)).astype(np.float16)
    return x, s, qc


def build_dataset(meta, data_dir, out_dir, n_jobs=4, chunk=512, fs=200, band=(0.5, 20.0)):
    """meta 순서대로 배열을 만들어 out_dir에 저장하고 품질 지표를 붙인 meta를 반환한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(meta)
    eeg_mm = np.lib.format.open_memmap(out_dir / "eeg50.npy", mode="w+", dtype=np.float16, shape=(n, 16, 50 * fs))
    spec_mm = np.lib.format.open_memmap(out_dir / "spec10.npy", mode="w+", dtype=np.float16, shape=(n, 4, 100, 300))
    qcs, t0 = [], time.time()
    rows = list(meta.itertuples(index=False))
    for start in range(0, n, chunk):
        part = rows[start:start + chunk]
        res = Parallel(n_jobs=n_jobs)(delayed(process_one)(r, data_dir, fs, band) for r in part)
        for i, (x, s, qc) in enumerate(res):
            eeg_mm[start + i], spec_mm[start + i] = x, s
            qcs.append(qc)
        done = start + len(part)
        print(f"  {done:,}/{n:,} ({done / n:.0%}) | {time.time() - t0:,.0f}s", flush=True)  # 무응답 셀 중단 방지
    eeg_mm.flush(); spec_mm.flush()
    out = pd.concat([meta.reset_index(drop=True), pd.DataFrame(qcs)], axis=1)
    out.to_csv(out_dir / "meta.csv", index=False)
    return out
