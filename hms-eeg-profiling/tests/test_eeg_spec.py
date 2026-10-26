"""EEG 스펙트로그램 front-end : 모양, 주파수 위치, 체인 위치, 입력 방식별 결합 검증."""
import numpy as np
import torch

from src.models.eeg_spec import EEGSpectrogram, HMSNet

FS, T = 200, 10000
t = torch.arange(T) / FS


def _peak_row(img_chain):
    return int(img_chain.mean(-1).argmax())


def test_shape_and_standardization():
    x = torch.randn(3, 16, T)
    img = EEGSpectrogram()(x)
    assert img.shape == (3, 1, 400, 256)
    assert torch.allclose(img.mean(dim=(2, 3)), torch.zeros(3, 1), atol=1e-4)
    assert torch.allclose(img.std(dim=(2, 3)), torch.ones(3, 1), atol=1e-3)


def test_frequency_axis_low_on_top():
    """2Hz 사인파는 10Hz 사인파보다 위쪽(작은 행 번호)에 봉우리가 생긴다 : 대회 스펙트로그램과 같은 방향."""
    m = EEGSpectrogram()
    lo = m((torch.sin(2 * np.pi * 2 * t) * 50).repeat(1, 16, 1) + 0.1 * torch.randn(1, 16, T))
    hi = m((torch.sin(2 * np.pi * 10 * t) * 50).repeat(1, 16, 1) + 0.1 * torch.randn(1, 16, T))
    r_lo, r_hi = _peak_row(lo[0, 0, :100]), _peak_row(hi[0, 0, :100])
    assert r_lo < r_hi
    assert abs(r_hi / 100 * 19.5 + 0.5 - 10) < 1.0                       # 행 번호 → 주파수 환산 약 10Hz


def test_chain_position():
    """RL 체인(채널 4~7)에만 델타파를 넣으면 두 번째 100행 블록이 가장 밝다."""
    x = 0.5 * torch.randn(1, 16, T)
    x[0, 4:8] += 80 * torch.sin(2 * np.pi * 1.5 * t)
    img = EEGSpectrogram()(x)[0, 0]
    block_power = [float(img[i * 100:(i + 1) * 100].mean()) for i in range(4)]
    assert int(np.argmax(block_power)) == 1


def test_hmsnet_modes_concat_width():
    import timm
    spec, eeg = torch.randn(2, 1, 400, 300), torch.randn(2, 16, T)
    for mode, w in [("spec", 300), ("eeg", 256), ("both", 556)]:
        net = HMSNet(timm.create_model("efficientnet_b0", pretrained=False, in_chans=1, num_classes=6), mode)
        assert net.image(spec, eeg).shape == (2, 1, 400, w)
    assert set(net.state_dict()) == {f"backbone.{k}" for k in net.backbone.state_dict()}  # front-end는 저장 대상 없음


def test_flip_consistent_between_eeg_and_spec():
    """E5 좌우 반전 : EEG 채널 반전(flip_lr) 후 만든 스펙트로그램 = 원래 스펙트로그램의 영역 반전(flip_spec)."""
    from src.data.montage import flip_lr, flip_spec
    x = 0.5 * torch.randn(1, 16, T)
    x[0, 0:4] += 80 * torch.sin(2 * np.pi * 1.5 * t)                     # LL 체인에만 델타파
    m = EEGSpectrogram()
    a = m(torch.from_numpy(flip_lr(x[0].numpy()).copy())[None])[0, 0].reshape(4, 100, 256).numpy()
    b = flip_spec(m(x)[0, 0].reshape(4, 100, 256).numpy())
    assert np.allclose(a, b, atol=1e-4)
    assert int(np.argmax(a.mean(axis=(1, 2)))) == 1                     # 델타파가 RL 영역으로 이동
