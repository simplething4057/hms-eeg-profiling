"""신호 전처리 : 결측 보간과 대역 통과 필터

- 결측 : 채널별 선형 보간, 채널 전체가 NaN이면 0 (해당 채널은 품질 플래그로 따로 기록)
- 필터 : 0.5~20Hz 4차 Butterworth, sosfiltfilt(위상 왜곡 없음). 판정 대상 패턴이 대부분 20Hz 이하
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def interp_nan(x: np.ndarray) -> np.ndarray:
    """(C, T) 배열의 NaN을 채널별로 선형 보간한다."""
    x = x.astype(np.float32, copy=True)
    t = np.arange(x.shape[1])
    for c in range(x.shape[0]):
        bad = np.isnan(x[c])
        if bad.all():
            x[c] = 0.0
        elif bad.any():
            x[c, bad] = np.interp(t[bad], t[~bad], x[c, ~bad])
    return x


def bandpass(x: np.ndarray, fs=200, low=0.5, high=20.0, order=4) -> np.ndarray:
    """(C, T) 배열에 대역 통과 필터 적용."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)
