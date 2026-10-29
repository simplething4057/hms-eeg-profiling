"""2단계 뇌 기능 프로파일링 : 판독 관점 정량 지표 7종 (코드북 4-4)

입력
- eeg : (16, 10000) 더블 바나나 몽타주, 0.5~20Hz 필터 완료 (hms-processed eeg50)
- spec : (4, 100, 300) 10분 스펙트로그램 log1p (hms-processed spec10)

판정 대상은 50초 중 가운데 10초이므로, 파형 지표(1~5, 7)는 중앙 10초에서 계산하고
시간 변화(6)만 10분 스펙트로그램에서 계산한다.

채널 순서 (src.data.montage) : LL 0~3, RL 4~7, LP 8~11, RP 12~15
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

FS = 200
CENTER = slice(20 * FS, 30 * FS)  # 50초 중 판정 대상 중앙 10초
REGIONS = {"LL": slice(0, 4), "RL": slice(4, 8), "LP": slice(8, 12), "RP": slice(12, 16)}
LEFT, RIGHT = [0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 20)}
SPEC_FREQS = np.linspace(0.59, 19.92, 100)  # 대회 스펙트로그램 주파수 축 (근사)
EPS = 1e-8


def _band_power(f, psd, lo, hi):
    """주파수 칸 합 x 칸 폭 : 대역끼리 겹치거나 비는 칸이 없어 대역 합이 전체와 정확히 같다."""
    m = (f >= lo) & (f < hi)
    return psd[..., m].sum(axis=-1) * (f[1] - f[0])


def frequency_features(f, psd):
    """① 주파수 구성 : 영역별·전체 상대 파워 (델타·세타·알파·베타)."""
    total = _band_power(f, psd, 0.5, 20) + EPS
    out = {}
    for b, (lo, hi) in BANDS.items():
        rel = _band_power(f, psd, lo, hi) / total  # (16,)
        out[f"rel_{b}"] = float(rel.mean())
        for r, sl in REGIONS.items():
            out[f"rel_{b}_{r}"] = float(rel[sl].mean())
    return out


def laterality_features(f, psd):
    """② 편측성 : 대역별 좌우 비대칭 지수 (좌 - 우) / (좌 + 우), 부호와 절대값."""
    out = {}
    for b, (lo, hi) in list(BANDS.items()) + [("total", (0.5, 20))]:
        p = _band_power(f, psd, lo, hi)
        left, right = p[LEFT].sum(), p[RIGHT].sum()
        ai = (left - right) / (left + right + EPS)
        out[f"asym_{b}"], out[f"abs_asym_{b}"] = float(ai), float(abs(ai))
    # 체인 쌍 비교 : 외측(LL vs RL), 부정중(LP vs RP)
    p = _band_power(f, psd, 0.5, 20)
    for name, a, c in [("lat", "LL", "RL"), ("par", "LP", "RP")]:
        l, r = p[REGIONS[a]].sum(), p[REGIONS[c]].sum()
        out[f"abs_asym_chain_{name}"] = float(abs(l - r) / (l + r + EPS))
    return out


def focality_features(f, psd):
    """③ 국소성 : 델타 파워가 가장 큰 영역과 그 집중도 (최대 / 평균)."""
    p = _band_power(f, psd, 0.5, 4)
    reg = np.array([p[sl].mean() for sl in REGIONS.values()])
    return {"focal_region": int(reg.argmax()), "focal_ratio": float(reg.max() / (reg.mean() + EPS))}


def _envelope(x, fs=FS, smooth_sec=0.05, down=4):
    """뾰족파 반복을 잡기 위한 포락선 : |미분| 이동평균 → 50Hz로 축소."""
    d = np.abs(np.diff(x, axis=-1))
    k = max(int(smooth_sec * fs), 1)
    ker = np.ones(k) / k
    env = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), -1, d)
    return env[..., ::down], fs / down


def periodicity_features(x, lag_range=(0.3, 3.0)):
    """④ 주기성 : 포락선 자기상관의 최대 봉우리 (반복 강도)와 그 시차 (반복 간격, 초).
    PD는 짧은 방전이 0.5~2초 간격으로 반복되므로 강도가 높고 간격이 이 범위에 든다."""
    env, fs = _envelope(x)
    strength, interval = _autocorr_peak(env, fs, lag_range)
    best = int(strength.argmax())
    return {"period_strength_max": float(strength.max()), "period_strength_mean": float(strength.mean()),
            "period_strength_L": float(strength[LEFT].mean()), "period_strength_R": float(strength[RIGHT].mean()),
            "period_interval": float(interval[best])}


def _autocorr_peak(env, fs, lag_range):
    """채널별 포락선 자기상관의 lag_range(초) 안 최대 봉우리 값과 그 시차(초)."""
    env = env - env.mean(axis=-1, keepdims=True)
    n = env.shape[-1]
    spec = np.fft.rfft(env, n=2 * n, axis=-1)
    ac = np.fft.irfft(spec * np.conj(spec), axis=-1)[..., :n]
    ac = ac / (ac[..., :1] + EPS)
    lo, hi = int(lag_range[0] * fs), int(lag_range[1] * fs)
    seg = ac[..., lo:hi]
    return seg.max(axis=-1), (seg.argmax(axis=-1) + lo) / fs


_HF_SOS = butter(4, 4.0, btype="highpass", fs=FS, output="sos")


def discharge_features(x, lag_range=(0.3, 3.0)):
    """④-2 주기성 수정판 (10/21, H5 1차 결과 반영)
    - 기존 지표는 RDA의 연속 율동파(1~2Hz)도 같은 시차에서 자기상관 봉우리를 만들어 PD와 구분이 약했다.
    - 판독에서 PD를 가르는 특징은 '뾰족한 방전 + 그 사이 조용한 구간'이다. 뾰족한 방전은 4Hz 이상 성분을 갖지만
      순수 델타파는 거의 갖지 않으므로, 4~20Hz 성분의 포락선으로 반복성을 다시 계산한다.
    - 방전 돌출도 : 포락선의 상위 5% 값 / 중앙값. 방전 사이가 조용할수록 커진다."""
    hf = sosfiltfilt(_HF_SOS, x, axis=-1)
    env, fs = _envelope(hf)
    strength, interval = _autocorr_peak(env, fs, lag_range)
    prom = np.percentile(env, 95, axis=-1) / (np.median(env, axis=-1) + EPS)
    best = int(strength.argmax())
    return {"period_hf_strength_max": float(strength.max()), "period_hf_strength_mean": float(strength.mean()),
            "period_hf_strength_L": float(strength[LEFT].mean()), "period_hf_strength_R": float(strength[RIGHT].mean()),
            "period_hf_interval": float(interval[best]),
            "discharge_prom_max": float(prom.max()), "discharge_prom_mean": float(prom.mean())}


def rhythmicity_features(f, psd):
    """⑤ 율동성 : 델타 대역 스펙트럼 봉우리의 선명도 (최대 / 평균)와 봉우리 주파수.
    RDA는 일정한 주파수의 느린 파가 지속되므로 델타 대역에 뾰족한 봉우리가 생긴다."""
    m = (f >= 0.5) & (f < 4)
    pd_ = psd[:, m]
    sharp = pd_.max(axis=1) / (pd_.mean(axis=1) + EPS)  # (16,)
    best = int(sharp.argmax())
    return {"delta_peak_sharp_max": float(sharp.max()), "delta_peak_sharp_mean": float(sharp.mean()),
            "delta_peak_sharp_L": float(sharp[LEFT].mean()), "delta_peak_sharp_R": float(sharp[RIGHT].mean()),
            "delta_peak_freq": float(f[m][pd_[best].argmax()])}


def temporal_features(spec):
    """⑥ 시간 변화 : 10분 스펙트로그램의 전체 파워·주파수 중심의 추세 (분당 기울기)와 변동성.
    발작은 주파수·진폭·위치가 시간에 따라 변하는 '진화'가 특징이다."""
    s = np.asarray(spec, dtype=np.float32)               # (4, F, T) log1p
    power = s.mean(axis=1)                               # (4, T)
    w = np.expm1(np.clip(s, 0, 30))
    centroid = (w * SPEC_FREQS[: s.shape[1], None]).sum(1) / (w.sum(1) + EPS)  # (4, T)
    t = np.arange(s.shape[-1]) * 2 / 60.0                # 분
    mid = slice(s.shape[-1] // 2 - 30, s.shape[-1] // 2 + 30)  # 중앙 2분
    slope = lambda y, tt: np.polyfit(tt, y, 1)[0]
    out = {
        "power_slope_10m": float(np.mean([slope(p, t) for p in power])),
        "power_slope_2m": float(np.mean([slope(p[mid], t[mid]) for p in power])),
        "centroid_slope_2m": float(np.mean([slope(c[mid], t[mid]) for c in centroid])),
        "power_std_10m": float(power.std(axis=1).mean()),
        "centroid_std_2m": float(centroid[:, mid].std(axis=1).mean()),
    }
    # ⑥-2 수정판 (10/21) : 4부위 평균은 한쪽에서만 진화하는 발작을 희석하므로 부위 최대값을 추가하고,
    # 판정 구간(중앙 1분)이 앞뒤 2분과 얼마나 다른지(국소 변화량)를 추가한다.
    T = s.shape[-1]
    c = slice(T // 2 - 15, T // 2 + 15)                                   # 중앙 1분
    ctx = np.r_[T // 2 - 75: T // 2 - 15, T // 2 + 15: T // 2 + 75]      # 앞뒤 각 2분
    out.update({
        "power_std_10m_max": float(power.std(axis=1).max()),
        "centroid_std_2m_max": float(centroid[:, mid].std(axis=1).max()),
        "power_slope_2m_absmax": float(np.max([abs(slope(p[mid], t[mid])) for p in power])),
        "center_ctx_power_absmax": float(np.abs(power[:, c].mean(1) - power[:, ctx].mean(1)).max()),
        "center_ctx_centroid_absmax": float(np.abs(centroid[:, c].mean(1) - centroid[:, ctx].mean(1)).max()),
    })
    return out


def background_features(x, fs=FS, suppress_uv=10.0):
    """⑦ 배경 활동 : 전체 진폭 수준과 억제 구간 비율 (1초 창 RMS가 기준 미만인 비율)."""
    amp = np.std(x, axis=1)
    win = x[:, : (x.shape[1] // fs) * fs].reshape(x.shape[0], -1, fs)
    rms = np.sqrt((win ** 2).mean(axis=2)).mean(axis=0)   # 창별 전체 채널 평균 RMS
    return {"amp_median": float(np.median(amp)), "amp_max": float(amp.max()),
            "suppression_ratio": float((rms < suppress_uv).mean())}


def profile_features(eeg, spec):
    """한 구간의 프로파일 지표 dict. eeg (16, 10000) / spec (4, 100, 300)."""
    x = np.asarray(eeg, dtype=np.float32)[:, CENTER]
    f, psd = welch(x, fs=FS, nperseg=2 * FS, axis=-1)   # 0.5Hz 해상도
    feats = {}
    for fn in (frequency_features, laterality_features, focality_features, rhythmicity_features):
        feats.update(fn(f, psd))
    feats.update(periodicity_features(x))
    feats.update(discharge_features(x))
    feats.update(temporal_features(spec))
    feats.update(background_features(x))
    return feats
