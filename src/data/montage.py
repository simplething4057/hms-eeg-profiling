"""더블 바나나(종방향 양극) 몽타주 16채널 생성

HMS 정답은 전문의가 더블 바나나 화면으로 판정한 결과이므로, 모델 입력도 같은 관점으로 맞춘다.
정중앙 전극(Fz, Cz, Pz)과 EKG는 사용하지 않는다.
"""
import numpy as np
import pandas as pd

CHAINS = {
    "LL": ["Fp1", "F7", "T3", "T5", "O1"],  # 좌측 외측
    "RL": ["Fp2", "F8", "T4", "T6", "O2"],  # 우측 외측
    "LP": ["Fp1", "F3", "C3", "P3", "O1"],  # 좌측 부정중
    "RP": ["Fp2", "F4", "C4", "P4", "O2"],  # 우측 부정중
}
PAIRS = [(a, b) for ch in CHAINS.values() for a, b in zip(ch[:-1], ch[1:])]
CHANNEL_NAMES = [f"{a}-{b}" for a, b in PAIRS]
# 좌우 반전 증강(H3)용 : 좌측 체인 ↔ 우측 체인 인덱스 매핑
FLIP_INDEX = [*range(4, 8), *range(0, 4), *range(12, 16), *range(8, 12)]


def double_banana(eeg: pd.DataFrame) -> np.ndarray:
    """전극별 원신호(행=시간) → (16, T) float32 배열."""
    missing = {e for pair in PAIRS for e in pair} - set(eeg.columns)
    if missing:
        raise KeyError(f"필요한 전극 컬럼 없음: {sorted(missing)}")
    return np.stack([eeg[a].to_numpy() - eeg[b].to_numpy() for a, b in PAIRS]).astype(np.float32)


def flip_lr(x: np.ndarray) -> np.ndarray:
    """(16, T) 몽타주 신호의 좌우 체인을 교체한다. 편측 라벨은 '어느 쪽이든 편측'이므로 유지된다."""
    return x[FLIP_INDEX]
