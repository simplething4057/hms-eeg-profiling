"""데이터 경로 탐색 및 구간 절단

- Kaggle 대회 데이터는 /kaggle/input/competitions/<slug>/ 처럼 하위 폴더에 붙을 수 있어 위치를 탐색한다.
- EEG : offset(초)부터 50초 = 10,000 샘플 절단, 짧으면 NaN으로 채워 길이를 맞춘다.
- 스펙트로그램 : offset부터 600초(2초 간격 300칸) 절단, 부위 순서는 몽타주 체인과 같은 LL, RL, LP, RP.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

SPEC_REGIONS = ["LL", "RL", "LP", "RP"]


def find_data_dir(root="/kaggle/input"):
    """train.csv와 train_eegs/가 함께 있는 폴더를 반환한다. 환경변수 HMS_DATA_DIR가 있으면 우선한다."""
    if os.environ.get("HMS_DATA_DIR"):
        return Path(os.environ["HMS_DATA_DIR"])
    for pattern in ["*/train.csv", "*/*/train.csv", "*/*/*/train.csv"]:
        for csv in sorted(Path(root).glob(pattern)):
            if (csv.parent / "train_eegs").is_dir():
                return csv.parent
    raise FileNotFoundError(f"{root} 아래에서 대회 데이터를 찾지 못했습니다. Add Input 연결을 확인하세요.")


def find_input_dir(name, root="/kaggle/input"):
    """Kaggle Dataset 이름으로 입력 폴더를 찾는다 (예: hms-src, hms-processed). 없으면 None."""
    for pattern in [name, f"*/{name}", f"*/*/{name}"]:
        hits = sorted(Path(root).glob(pattern))
        if hits:
            return hits[0]
    return None


def eeg_window(eeg: pd.DataFrame, offset_sec: float, fs=200, sec=50) -> pd.DataFrame:
    """offset부터 sec초 구간. 기록이 짧으면 부족한 행을 NaN으로 채운다."""
    start, n = int(round(offset_sec * fs)), int(sec * fs)
    seg = eeg.iloc[start:start + n]
    if len(seg) < n:
        pad = pd.DataFrame(np.nan, index=range(n - len(seg)), columns=eeg.columns)
        seg = pd.concat([seg, pad], ignore_index=True)
    return seg.reset_index(drop=True)


def spec_window(spec: pd.DataFrame, offset_sec: float, sec=600, n_freq=100) -> np.ndarray:
    """(4, n_freq, 300) 배열. time 값이 offset 이상 offset+sec 미만인 행을 사용하고 300칸에 맞춰 자르거나 0으로 채운다."""
    n_time = sec // 2
    w = spec[(spec["time"] >= offset_sec) & (spec["time"] < offset_sec + sec)]
    out = np.zeros((len(SPEC_REGIONS), n_freq, n_time), dtype=np.float32)
    for i, r in enumerate(SPEC_REGIONS):
        cols = [c for c in spec.columns if c.startswith(r + "_")][:n_freq]
        v = w[cols].to_numpy(dtype=np.float32)[:n_time].T  # (freq, time)
        out[i, : v.shape[0], : v.shape[1]] = v
    return np.nan_to_num(out, nan=0.0)
