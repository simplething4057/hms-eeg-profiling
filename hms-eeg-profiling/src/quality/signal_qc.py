"""1단계 신호 품질 판정 : 보간·필터 전 원신호 기준으로 계산한다.

- nan_ratio : 결측 비율
- n_flat : 표준편차가 거의 0인(평탄) 채널 수
- max_abs : 최대 절대 진폭 (과도 진폭 탐지)
"""
import warnings

import numpy as np

FLAT_STD = 1e-3
HIGH_AMP = 1000.0  # µV, 초과 시 과도 진폭으로 표시 (nb03에서 분포 확인 후 조정)


def qc_stats(x: np.ndarray) -> dict:
    """(C, T) 원신호 → 품질 지표 dict."""
    nan_ratio = float(np.isnan(x).mean())
    with warnings.catch_warnings():  # 전체 NaN 채널의 경고는 정상 상황이므로 숨김
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(x, axis=1)
    n_flat = int(np.sum(~(std > FLAT_STD)))  # NaN 채널도 평탄으로 계산
    max_abs = float(np.nanmax(np.abs(x))) if not np.isnan(x).all() else 0.0
    return {
        "nan_ratio": nan_ratio,
        "n_flat": n_flat,
        "max_abs": max_abs,
        "qc_flag": int(nan_ratio > 0.2 or n_flat >= 4 or max_abs > HIGH_AMP),
    }
