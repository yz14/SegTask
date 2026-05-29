"""Tests for anisotropic downsampling (per-axis stride schedule).

Covers:
- Downsample / Upsample accept per-axis stride tuples (iso default unchanged).
- _auto_anisotropic_strides nnU-Net-style schedule (thin z preserved).
- compute_downsample_strides: disabled→None, auto, explicit.
- End-to-end build_model: anisotropic keeps z resolution vs isotropic, and the
  full UNet reconstructs the input spatial size (encoder/decoder symmetry).
- Explicit downsample_strides path builds and runs.
- Compatibility guards + config validation.
"""

from __future__ import annotations

import pytest
import torch

from segtask_v1.models.blocks import Downsample, Upsample, _as_stride_tuple


# ---------------------------------------------------------------------------
# Stride helper
# ---------------------------------------------------------------------------
def test_as_stride_tuple():
    assert _as_stride_tuple(2, 3) == (2, 2, 2)
    assert _as_stride_tuple(2, 2) == (2, 2)
    assert _as_stride_tuple((1, 2, 2), 3) == (1, 2, 2)
    with pytest.raises(ValueError):
        _as_stride_tuple((2, 2), 3)       # wrong length
    with pytest.raises(ValueError):
        _as_stride_tuple((0, 2, 2), 3)    # value < 1


# ---------------------------------------------------------------------------
# Downsample / Upsample unit shapes
# ---------------------------------------------------------------------------
def test_downsample_isotropic_default_unchanged():
    ds = Downsample(8, 16, mode="conv", spatial_dims=3)  # default stride=2
    y = ds(torch.randn(1, 8, 8, 16, 16))
    assert y.shape == (1, 16, 4, 8, 8)


@pytest.mark.parametrize("mode", ["conv", "maxpool", "avgpool"])
def test_downsample_anisotropic_keeps_z(mode):
    ds = Downsample(8, 16, mode=mode, spatial_dims=3, stride=(1, 2, 2))
    y = ds(torch.randn(1, 8, 12, 16, 16))
    assert y.shape == (1, 16, 12, 8, 8)   # z preserved, H/W halved


@pytest.mark.parametrize("mode", ["blurpool", "pixelunshuffle"])
def test_downsample_anisotropic_rejected_for_special_modes(mode):
    with pytest.raises(ValueError):
        Downsample(8, 16, mode=mode, spatial_dims=3, stride=(1, 2, 2))


@pytest.mark.parametrize("mode", ["transpose", "trilinear", "nearest"])
def test_upsample_anisotropic_mirrors(mode):
    up = Upsample(16, 8, mode=mode, spatial_dims=3, stride=(1, 2, 2))
    y = up(torch.randn(1, 16, 12, 8, 8))
    assert y.shape == (1, 8, 12, 16, 16)  # z preserved, H/W doubled


@pytest.mark.parametrize("mode", ["pixelshuffle", "carafe", "dysample"])
def test_upsample_anisotropic_rejected_for_special_modes(mode):
    with pytest.raises(ValueError):
        Upsample(16, 8, mode=mode, spatial_dims=3, stride=(1, 2, 2))


def test_downsample_upsample_roundtrip_size():
    """A matched down→up pair returns the original spatial size per axis."""
    ds = Downsample(4, 8, mode="conv", spatial_dims=3, stride=(1, 2, 2))
    up = Upsample(8, 4, mode="transpose", spatial_dims=3, stride=(1, 2, 2))
    x = torch.randn(1, 4, 12, 32, 32)
    assert up(ds(x)).shape[2:] == x.shape[2:]


# ---------------------------------------------------------------------------
# Auto schedule
# ---------------------------------------------------------------------------
def test_auto_schedule_thin_z_3d():
    from segtask_v1.models.factory import _auto_anisotropic_strides
    # z=32, H/W=256, 4 downsamples (5-stage encoder).
    sched = _auto_anisotropic_strides([32, 256, 256], num_down=4)
    assert len(sched) == 4
    # z should be pooled at most once (32 stays ahead until H/W catch up).
    z_pools = sum(1 for s in sched if s[0] == 2)
    hw_pools = sum(1 for s in sched if s[1] == 2)
    assert hw_pools == 4               # H/W pooled every level
    assert z_pools <= 1                # z barely pooled
    # Final z size >= 16 (vs isotropic which would give 2).
    z = 32
    for s in sched:
        if s[0] == 2:
            z //= 2
    assert z >= 16


def test_auto_schedule_isotropic_when_balanced():
    from segtask_v1.models.factory import _auto_anisotropic_strides
    sched = _auto_anisotropic_strides([64, 64, 64], num_down=3)
    assert all(s == (2, 2, 2) for s in sched)


# ---------------------------------------------------------------------------
# compute_downsample_strides config plumbing
# ---------------------------------------------------------------------------
def _cfg_3d(**overrides):
    from segtask_v1.config import Config
    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [16, 32, 64, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.downsample_mode = "conv"
    cfg.model.upsample_mode = "trilinear"
    for k, v in overrides.items():
        setattr(cfg.model, k, v)
    cfg.sync()
    cfg.validate()
    return cfg


def test_compute_strides_disabled_returns_none():
    from segtask_v1.models.factory import compute_downsample_strides
    cfg = _cfg_3d()
    assert compute_downsample_strides(cfg, 3, len(cfg.model.encoder_channels)) is None


def test_compute_strides_auto():
    from segtask_v1.models.factory import compute_downsample_strides
    cfg = _cfg_3d(anisotropic_pooling=True)
    sched = compute_downsample_strides(cfg, 3, len(cfg.model.encoder_channels))
    assert sched is not None and len(sched) == 3
    # patch z=16, H/W=64 → z preserved relative to H/W.
    assert sum(s[1] == 2 for s in sched) >= sum(s[0] == 2 for s in sched)


def test_compute_strides_explicit_overrides_auto():
    from segtask_v1.models.factory import compute_downsample_strides
    cfg = _cfg_3d(anisotropic_pooling=True,
                  downsample_strides=[[1, 2, 2], [1, 2, 2], [2, 2, 2]])
    sched = compute_downsample_strides(cfg, 3, len(cfg.model.encoder_channels))
    assert sched == [(1, 2, 2), (1, 2, 2), (2, 2, 2)]


# ---------------------------------------------------------------------------
# End-to-end build_model
# ---------------------------------------------------------------------------
def test_build_model_isotropic_default_z_halves():
    from segtask_v1.models.factory import build_model
    cfg = _cfg_3d()  # 4 stages → 3 downsamples
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    feats = model.encoder(x)
    # isotropic: z 16 → 8 → 4 → 2 at bottleneck.
    assert feats[-1].shape[2] == 2
    with torch.no_grad():
        y = model(x)
    main = y[0] if isinstance(y, list) else y
    assert main.shape[-3:] == (16, 64, 64)


def test_build_model_anisotropic_preserves_z_and_reconstructs():
    from segtask_v1.models.factory import build_model
    cfg = _cfg_3d(anisotropic_pooling=True)
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    feats = model.encoder(x)
    # anisotropic: z should stay larger than the isotropic bottleneck (2).
    assert feats[-1].shape[2] > 2
    with torch.no_grad():
        y = model(x)
    main = y[0] if isinstance(y, list) else y
    assert main.shape[-3:] == (16, 64, 64)   # full reconstruction


def test_build_model_explicit_strides_runs():
    from segtask_v1.models.factory import build_model
    cfg = _cfg_3d(downsample_strides=[[1, 2, 2], [1, 2, 2], [2, 2, 2]])
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    feats = model.encoder(x)
    # z pooled once (last stage) → 16/2 = 8.
    assert feats[-1].shape[2] == 8
    with torch.no_grad():
        y = model(x)
    main = y[0] if isinstance(y, list) else y
    assert main.shape[-3:] == (16, 64, 64)


# ---------------------------------------------------------------------------
# Guards + validation
# ---------------------------------------------------------------------------
def test_guard_anisotropic_rejects_unetpp():
    from segtask_v1.models.factory import build_model
    cfg = _cfg_3d(anisotropic_pooling=True, decoder_type="unetpp",
                  upsample_mode="transpose")
    with pytest.raises(ValueError):
        build_model(cfg)


def test_guard_anisotropic_rejects_blurpool():
    from segtask_v1.models.factory import build_model
    cfg = _cfg_3d(downsample_strides=[[1, 2, 2], [1, 2, 2], [2, 2, 2]],
                  downsample_mode="blurpool")
    with pytest.raises(ValueError):
        build_model(cfg)


def test_validate_rejects_bad_downsample_strides_length():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.encoder_channels = [16, 32, 64, 64]
    cfg.model.downsample_strides = [[1, 2, 2]]  # need 3 entries
    cfg.sync()
    with pytest.raises(AssertionError):
        cfg.validate()


def test_validate_rejects_bad_stride_value():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.encoder_channels = [16, 32, 64, 64]
    cfg.model.downsample_strides = [[1, 2, 2], [1, 2, 2], [3, 2, 2]]  # 3 invalid
    cfg.sync()
    with pytest.raises(AssertionError):
        cfg.validate()
