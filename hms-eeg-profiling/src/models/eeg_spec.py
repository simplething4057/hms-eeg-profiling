"""50초 EEG → 스펙트로그램 이미지 변환 (E3·E4 입력, GPU에서 배치 단위로 계산)

대회 제공 10분 스펙트로그램과 같은 배치(4개 체인 LL·RL·LP·RP를 세로로 이어 붙임, 위쪽이 낮은 주파수)로 만들어
E1·E2와 같은 백본·전처리 흐름을 유지한다. 학습 파라미터가 없으므로 저장 가중치는 백본만 포함된다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHAINS, PAIRS_PER_CHAIN = 4, 4          # 더블 바나나 16채널 = 체인 4개 x 인접 전극쌍 4개 (montage.CHAINS 순서)


class EEGSpectrogram(nn.Module):
    """(B, 16, T) 몽타주 신호 → (B, 1, 4 * n_freq, width) 표본별 표준화 로그 파워 이미지.

    - STFT : Hann 창 n_fft 표본(기본 2.56초, 주파수 간격 0.39Hz), hop 표본 간격
    - 체인 안의 전극쌍 4개 파워를 평균 (대회 스펙트로그램과 같은 방식)
    - fmin~fmax 구간만 남기고 (n_freq, width)로 보간 → log → 체인 4개를 세로로 연결 → 표본별 표준화
    """

    def __init__(self, fs=200, n_fft=512, hop=40, fmin=0.5, fmax=20.0, n_freq=100, width=256):
        super().__init__()
        self.n_fft, self.hop, self.n_freq, self.width = n_fft, hop, n_freq, width
        freqs = torch.fft.rfftfreq(n_fft, 1 / fs)
        idx = torch.nonzero((freqs >= fmin) & (freqs <= fmax)).flatten()
        self.f_lo, self.f_hi = int(idx[0]), int(idx[-1]) + 1
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):
        b, c, t = x.shape
        assert c == N_CHAINS * PAIRS_PER_CHAIN, f"16채널 몽타주가 필요합니다 (입력 {c}채널)"
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = torch.stft(x.float().reshape(b * c, t), self.n_fft, hop_length=self.hop, window=self.window,
                           center=True, return_complex=True)
            p = z.abs().pow(2)[:, self.f_lo:self.f_hi]                         # (B*16, F, frames)
            p = p.reshape(b, N_CHAINS, PAIRS_PER_CHAIN, *p.shape[1:]).mean(2)  # (B, 4, F, frames)
            p = F.interpolate(p, size=(self.n_freq, self.width), mode="bilinear", align_corners=False)
            img = torch.log(p + 1e-6).reshape(b, 1, N_CHAINS * self.n_freq, self.width)
            mu = img.mean(dim=(2, 3), keepdim=True)
            sd = img.std(dim=(2, 3), keepdim=True)
            return (img - mu) / (sd + 1e-6)


class HMSNet(nn.Module):
    """입력 방식별 분류기. input_mode : 'spec'(10분, E1·E2) / 'eeg'(50초, E3) / 'both'(가로 연결, E4).

    spec 입력은 Dataset에서 이미 표준화된 (B, 1, 400, 300) 이미지, eeg 입력은 (B, 16, 10000) 몽타주 신호.
    가중치 저장·로드는 backbone(timm 모델)만 대상으로 해 E1·E2 가중치 형식과 같게 유지한다.
    """

    def __init__(self, backbone, input_mode="spec", **spec_kwargs):
        super().__init__()
        assert input_mode in ("spec", "eeg", "both")
        self.input_mode = input_mode
        self.backbone = backbone
        self.front = EEGSpectrogram(**spec_kwargs) if input_mode != "spec" else None

    def image(self, spec=None, eeg=None):
        parts = []
        if self.input_mode in ("spec", "both"):
            parts.append(spec)
        if self.input_mode in ("eeg", "both"):
            parts.append(self.front(eeg))
        return torch.cat(parts, dim=3)                                          # 'both' : 400 x (300 + 256)

    def forward(self, spec=None, eeg=None):
        return self.backbone(self.image(spec, eeg))
