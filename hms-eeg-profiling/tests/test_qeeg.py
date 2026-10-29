"""프로파일 지표가 정답을 아는 합성 신호에서 판독 관점과 같은 방향으로 움직이는지 검증."""
import numpy as np
import pytest

from src.data.filters import bandpass
from src.profile.qeeg import LEFT, profile_features

FS, N = 200, 10000
t = np.arange(N) / FS
rng = np.random.default_rng(0)
SPEC = np.full((4, 100, 300), 1.0, dtype=np.float32)


def noise(scale=5.0):
    return bandpass(rng.normal(0, scale, (16, N)))


def test_relative_power_sums_to_one():
    f = profile_features(noise(), SPEC)
    assert sum(f[f"rel_{b}"] for b in ["delta", "theta", "alpha", "beta"]) == pytest.approx(1, abs=1e-6)


def test_laterality_detects_left_delta():
    x = noise()
    x[LEFT] += 60 * np.sin(2 * np.pi * 1.5 * t)          # 좌측에만 델타파
    lat, sym = profile_features(x, SPEC), profile_features(noise(), SPEC)
    assert lat["asym_delta"] > 0.5 and lat["abs_asym_delta"] > sym["abs_asym_delta"] + 0.4


def test_focality_region():
    x = noise()
    x[8:12] += 60 * np.sin(2 * np.pi * 1.0 * t)          # LP 영역(인덱스 2)
    f = profile_features(x, SPEC)
    assert f["focal_region"] == 2 and f["focal_ratio"] > 2


def test_periodicity_detects_1s_discharges():
    x = noise()
    spike = np.zeros(N)
    spike[::FS] = 1.0                                     # 1초 간격 방전
    spike = np.convolve(spike, np.hanning(20) * 150, mode="same")
    pdx = profile_features(bandpass(np.tile(spike, (16, 1))) + x, SPEC)
    base = profile_features(noise(), SPEC)
    assert pdx["period_strength_max"] > base["period_strength_max"] + 0.3
    assert pdx["period_interval"] == pytest.approx(1.0, abs=0.1)


def test_rhythmicity_detects_2hz_delta():
    x = noise() + 50 * np.sin(2 * np.pi * 2.0 * t)
    rda, base = profile_features(x, SPEC), profile_features(noise(), SPEC)
    assert rda["delta_peak_sharp_max"] > 2 * base["delta_peak_sharp_max"]
    assert rda["delta_peak_freq"] == pytest.approx(2.0, abs=0.5)


def test_temporal_evolution_slope():
    rising = np.repeat(np.linspace(0.5, 3.0, 300)[None, None, :], 100, axis=1).repeat(4, axis=0).astype(np.float32)
    up, flat = profile_features(noise(), rising), profile_features(noise(), SPEC)
    assert up["power_slope_10m"] > 0.1 and abs(flat["power_slope_10m"]) < 1e-6


def test_background_suppression():
    quiet = profile_features(noise(0.5), SPEC)
    normal = profile_features(noise(60), SPEC)   # 필터 후 약 25µV (실제 EEG 수준)
    assert quiet["suppression_ratio"] == 1.0 and normal["suppression_ratio"] == 0.0
    assert normal["amp_median"] > quiet["amp_median"]


def test_all_features_finite():
    f = profile_features(np.zeros((16, N), np.float32), SPEC)  # 평탄 신호에서도 NaN·inf 없음
    assert all(np.isfinite(v) for v in f.values()) and len(f) > 50


def _pd_and_rda():
    """진폭을 맞춘 1초 간격 방전(PD)과 1.5Hz 율동파(RDA)."""
    spike = np.zeros(N); spike[::FS] = 1.0
    spike = bandpass(np.tile(np.convolve(spike, np.hanning(20) * 150, mode="same"), (16, 1)))
    sine = np.tile(np.sin(2 * np.pi * 1.5 * t), (16, 1))
    sine *= spike.std() / sine.std()
    return spike + noise(), sine + noise()


def test_old_periodicity_confuses_rda():
    """1차 H5에서 주기성이 약했던 원인 재현 : 기존 지표는 RDA에도 높게 나온다."""
    pdx, rda = (profile_features(x, SPEC) for x in _pd_and_rda())
    assert rda["period_strength_max"] > 0.5


def test_hf_periodicity_separates_pd_from_rda():
    pdx, rda = (profile_features(x, SPEC) for x in _pd_and_rda())
    assert pdx["period_hf_strength_max"] > rda["period_hf_strength_max"] + 0.2
    assert pdx["discharge_prom_max"] > 1.5 * rda["discharge_prom_max"]
    assert pdx["period_hf_interval"] == pytest.approx(1.0, abs=0.1)


def test_temporal_region_max_and_center_change():
    """한 부위(LL)에서만 판정 구간에 파워가 오르는 국소 변화 : 최대값·중앙 대비 지표가 평균 지표보다 민감해야 한다."""
    s = SPEC.copy()
    s[0, :, 135:165] += 2.0
    f = profile_features(noise(), s)
    assert f["power_std_10m_max"] > 2 * f["power_std_10m"]
    assert f["center_ctx_power_absmax"] > 1.5
