"""원시 EEG parquet → E8 추론 파이프라인

입력: parquet 파일 (전극 컬럼: Fp1, F7, T3, T5, O1, Fp2, F8, T4, T6, O2, F3, C3, P3, F4, C4, P4 등)
출력: (qeeg_67d, spec_b64, meta)
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import spectrogram as scipy_spec

from src.data.filters import bandpass, interp_nan
from src.data.montage import double_banana
from src.profile.qeeg import profile_features
from src.quality.signal_qc import qc_stats

FS = 200
SEC = 50
N_SAMPLES = FS * SEC  # 10000

_CHAIN_SLICES = [(0, 4), (4, 8), (8, 12), (12, 16)]  # LL, RL, LP, RP
_SPEC_FREQS = np.linspace(0.59, 19.92, 100)


def _make_spec4(eeg_50s: np.ndarray) -> np.ndarray:
    """(16, 10000) → (4, 100, 300) 간이 스펙트로그램 (temporal_features 입력용).
    10분 스펙트로그램이 없으므로 50초 신호를 주파수 분석 후 300프레임으로 보간.
    """
    out = np.zeros((4, 100, 300), dtype=np.float32)
    for i, (lo, hi) in enumerate(_CHAIN_SLICES):
        chain_sig = eeg_50s[lo:hi].mean(axis=0)
        f, _, Sxx = scipy_spec(chain_sig, fs=FS, nperseg=FS * 2, noverlap=FS, scaling="density")
        Sxx_log = np.log1p(Sxx)
        freq_interp = interp1d(f, Sxx_log, axis=0, bounds_error=False, fill_value=0.0)
        Sxx_100 = freq_interp(_SPEC_FREQS)  # (100, t_frames)
        t_src = np.arange(Sxx_100.shape[1])
        t_dst = np.linspace(0, max(t_src[-1], 0), 300)
        time_interp = interp1d(t_src, Sxx_100, axis=1, bounds_error=False, fill_value=0.0)
        out[i] = time_interp(t_dst).astype(np.float32)
    return out


def _make_dl_spec_b64(eeg_50s: np.ndarray) -> str | None:
    """(16, 10000) → (400, 556) float32 → base64.

    - 우측 256열: EEGSpectrogram(50초 EEG)
    - 좌측 300열: 위 결과를 300폭으로 리사이즈해 10분 스펙트로그램 자리 대체
    """
    try:
        import torch
        import torch.nn.functional as F

        from src.models.eeg_spec import EEGSpectrogram
    except ImportError:
        return None

    spec_module = EEGSpectrogram()
    spec_module.eval()

    x = torch.from_numpy(eeg_50s).unsqueeze(0)  # (1, 16, 10000)
    with torch.no_grad():
        eeg_img = spec_module(x)  # (1, 1, 400, 256)

    ten_min_part = F.interpolate(eeg_img, size=(400, 300), mode="bilinear", align_corners=False)
    combined = torch.cat([ten_min_part, eeg_img], dim=3)  # (1, 1, 400, 556)

    arr = combined[0, 0].numpy().astype(np.float32)  # (400, 556)
    return base64.b64encode(arr.tobytes()).decode()


def preprocess_eeg_file(data: bytes) -> tuple[np.ndarray, str | None, dict]:
    """parquet bytes → (qeeg_67d, spec_b64_or_None, meta).

    meta 키: shape, channels, duration_sec, qc, n_features
    """
    eeg_df = pd.read_parquet(io.BytesIO(data))

    # 1. QC (원신호 기준)
    raw = double_banana(eeg_df).astype(np.float32)
    qc = qc_stats(raw)

    # 2. 전처리: NaN 보간 + 대역통과
    eeg = interp_nan(raw)
    eeg = bandpass(eeg)

    # 3. 50초 창 (중앙 또는 처음부터)
    if eeg.shape[1] >= N_SAMPLES:
        start = (eeg.shape[1] - N_SAMPLES) // 2
        eeg_50s = eeg[:, start : start + N_SAMPLES]
    else:
        eeg_50s = np.pad(eeg, ((0, 0), (0, N_SAMPLES - eeg.shape[1])))

    # 4. qEEG 피처
    spec4 = _make_spec4(eeg_50s)
    feats = profile_features(eeg_50s, spec4)

    # 5. x_cols 순서로 67차원 벡터 조립
    from api.predict import ModelServer  # 지연 임포트 (순환 참조 방지)

    x_cols = ModelServer.instance().p1_meta.get("x_cols", [])
    qc_feats = {
        "nan_ratio": qc["nan_ratio"],
        "n_flat": float(qc["n_flat"]),
        "max_abs": qc["max_abs"],
    }
    all_feats = {**feats, **qc_feats}
    vec = np.array([all_feats.get(c, 0.0) for c in x_cols], dtype=np.float32)

    # 6. DL 스펙트로그램 (torch 탑재 시)
    spec_b64: str | None = None
    try:
        if ModelServer.instance().dl_loaded:
            spec_b64 = _make_dl_spec_b64(eeg_50s)
    except Exception:  # noqa: BLE001
        pass

    meta = {
        "shape": list(eeg_df.shape),
        "channels": list(eeg_df.columns),
        "duration_sec": round(eeg_df.shape[0] / FS, 1),
        "qc": qc,
        "n_features": int(len(vec)),
    }
    return vec, spec_b64, meta
