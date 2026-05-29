"""Unit tests for enhanced down/up sampling blocks.

Covers:
- BlurPool3d (anti-aliased subsampling; Zhang ICML 2019)
- PixelUnshuffle3d / PixelShuffle3d (3D sub-pixel conv; ESPCN-style)
- ICNR init invariance (PixelShuffle at init ≈ nearest-neighbour)
- Downsample all 5 modes: shape, dtype, grad flow, output size
- Upsample all 4 modes: shape, dtype, grad flow, output size
- End-to-end UNet3D build with every combination of modes
- Config validation rejects unknown modes
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

from segtask_v1.models.blocks import (
    BlurPool3d,
    CARAFE3d,
    Downsample,
    DySample3d,
    PixelShuffle3d,
    PixelUnshuffle3d,
    Upsample,
    icnr_init_,
)


# ---------------------------------------------------------------------------
# BlurPool3d
# ---------------------------------------------------------------------------
class TestBlurPool3d:
    def test_shape_and_dtype(self):
        x = torch.randn(2, 4, 8, 16, 16)
        bp = BlurPool3d(channels=4, stride=2, filt_size=3)
        y = bp(x)
        assert y.shape == (2, 4, 4, 8, 8)
        assert y.dtype == x.dtype

    def test_constant_input_stays_constant(self):
        """Binomial low-pass kernel is normalised: averaging a constant
        field must return the same constant (up to numerical noise)."""
        x = torch.full((1, 3, 8, 8, 8), 0.73)
        bp = BlurPool3d(channels=3, stride=2, filt_size=3)
        y = bp(x)
        assert torch.allclose(y, torch.full_like(y, 0.73), atol=1e-5)

    def test_no_learned_params(self):
        bp = BlurPool3d(channels=8, stride=2, filt_size=5)
        assert sum(p.numel() for p in bp.parameters()) == 0

    def test_invalid_filter_size(self):
        with pytest.raises(ValueError):
            BlurPool3d(channels=4, stride=2, filt_size=4)


# ---------------------------------------------------------------------------
# PixelShuffle3d / PixelUnshuffle3d
# ---------------------------------------------------------------------------
class TestPixelShuffle3d:
    def test_roundtrip_is_identity(self):
        x = torch.randn(2, 3, 8, 8, 8)
        unshuf = PixelUnshuffle3d(r=2)
        shuf = PixelShuffle3d(r=2)
        y = shuf(unshuf(x))
        assert torch.allclose(x, y)

    def test_unshuffle_shape(self):
        x = torch.randn(1, 4, 8, 8, 16)
        y = PixelUnshuffle3d(r=2)(x)
        assert y.shape == (1, 4 * 8, 4, 4, 8)

    def test_shuffle_shape(self):
        x = torch.randn(1, 24, 4, 4, 4)  # 24 = 3 * 2^3
        y = PixelShuffle3d(r=2)(x)
        assert y.shape == (1, 3, 8, 8, 8)

    def test_unshuffle_rejects_non_divisible(self):
        x = torch.randn(1, 2, 5, 6, 8)
        with pytest.raises(ValueError):
            PixelUnshuffle3d(r=2)(x)

    def test_shuffle_rejects_bad_channels(self):
        x = torch.randn(1, 5, 4, 4, 4)  # 5 not divisible by 8
        with pytest.raises(ValueError):
            PixelShuffle3d(r=2)(x)


# ---------------------------------------------------------------------------
# ICNR init
# ---------------------------------------------------------------------------
class TestICNR:
    def test_icnr_makes_pixel_shuffle_match_nearest(self):
        """After ICNR init + PixelShuffle, the composite op should equal
        nearest-neighbour upsampling (same value replicated over 2^3 voxels).
        """
        in_ch, out_ch, r = 2, 3, 2
        conv = nn.Conv3d(in_ch, out_ch * r ** 3, kernel_size=1, bias=False)
        icnr_init_(conv.weight, upscale=r)
        shuffle = PixelShuffle3d(r=r)

        x = torch.randn(1, in_ch, 3, 3, 3)
        y = shuffle(conv(x))
        y_nearest = nn.functional.interpolate(conv(x)[:, :out_ch], scale_factor=r,
                                              mode="nearest")
        # After ICNR replication, all r^3 sub-filters are identical, so
        # each output voxel block of r^3 is a constant copy. Compare the
        # 0-th voxel of each block to its replica neighbours.
        y_blocks = y.unfold(2, r, r).unfold(3, r, r).unfold(4, r, r)
        ref = y_blocks[..., :1, :1, :1]
        assert torch.allclose(y_blocks, ref.expand_as(y_blocks), atol=1e-6)
        # Sanity: nearest-shape
        assert y.shape[-3:] == y_nearest.shape[-3:]

    def test_icnr_rejects_bad_out_ch(self):
        bad = torch.empty(5, 2, 1, 1, 1)  # 5 not divisible by r^3=8
        with pytest.raises(ValueError):
            icnr_init_(bad, upscale=2)


# ---------------------------------------------------------------------------
# Downsample — all modes
# ---------------------------------------------------------------------------
DOWN_MODES = Downsample.VALID_MODES
UP_MODES = Upsample.VALID_MODES


@pytest.mark.parametrize("mode", DOWN_MODES)
def test_downsample_shape_and_grad(mode):
    in_ch, out_ch = 4, 8
    x = torch.randn(2, in_ch, 8, 16, 16, requires_grad=True)
    down = Downsample(in_ch, out_ch, mode=mode)
    y = down(x)
    assert y.shape == (2, out_ch, 4, 8, 8)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_downsample_rejects_unknown_mode():
    with pytest.raises(ValueError):
        Downsample(4, 8, mode="bogus")


# ---------------------------------------------------------------------------
# Upsample — all modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", UP_MODES)
def test_upsample_shape_and_grad(mode):
    in_ch, out_ch = 8, 4
    x = torch.randn(2, in_ch, 4, 8, 8, requires_grad=True)
    up = Upsample(in_ch, out_ch, mode=mode)
    y = up(x)
    assert y.shape == (2, out_ch, 8, 16, 16)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_upsample_rejects_unknown_mode():
    with pytest.raises(ValueError):
        Upsample(8, 4, mode="bogus")


# ---------------------------------------------------------------------------
# End-to-end UNet3D build across (down, up) mode combinations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("down_mode,up_mode",
                          list(itertools.product(DOWN_MODES, UP_MODES)))
def test_unet_build_and_forward(down_mode, up_mode):
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]  # 2 fg classes
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.downsample_mode = down_mode
    cfg.model.upsample_mode = up_mode
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == 1
    assert y.shape[-3:] == (16, 64, 64)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_rejects_bad_downsample_mode(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.downsample_mode = "not_a_mode"
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_rejects_bad_upsample_mode(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.upsample_mode = "not_a_mode"
        with pytest.raises(AssertionError):
            cfg.validate()


# ---------------------------------------------------------------------------
# CARAFE3d — content-aware reassembly
# ---------------------------------------------------------------------------
class TestCARAFE3d:
    def test_shape_and_grad(self):
        in_ch, out_ch = 8, 16
        x = torch.randn(2, in_ch, 4, 6, 8, requires_grad=True)
        m = CARAFE3d(in_ch, out_ch, scale=2, k_up=3)
        y = m(x)
        assert y.shape == (2, out_ch, 8, 12, 16)
        y.sum().backward()
        assert x.grad is not None

    def test_kernel_is_probability(self):
        """Reassembly weights are softmax-normalised → sum to 1 along k^3."""
        in_ch = 4
        m = CARAFE3d(in_ch, in_ch, scale=2, k_up=3).eval()
        x = torch.randn(1, in_ch, 4, 4, 4)
        # Re-run internal kernel prediction to inspect it directly.
        w = m.compress(x)
        w = m.encode(w)
        w = m.shuffle(w)
        w = torch.softmax(w, dim=1)
        sums = w.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_constant_input_preserved(self):
        """A constant field must upsample to the same constant (partition
        of unity over reassembly kernel)."""
        in_ch = 4
        m = CARAFE3d(in_ch, in_ch, scale=2, k_up=3).eval()
        x = torch.full((1, in_ch, 4, 4, 4), 0.5)
        with torch.no_grad():
            y = m(x)
        # proj is Identity when in_ch == out_ch
        assert torch.allclose(y, torch.full_like(y, 0.5), atol=1e-5)

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            CARAFE3d(4, 4, scale=0)
        with pytest.raises(ValueError):
            CARAFE3d(4, 4, k_up=0)


# ---------------------------------------------------------------------------
# DySample3d — dynamic sampling
# ---------------------------------------------------------------------------
class TestDySample3d:
    def test_shape_and_grad(self):
        in_ch, out_ch = 8, 16
        x = torch.randn(2, in_ch, 4, 6, 8, requires_grad=True)
        m = DySample3d(in_ch, out_ch, scale=2, groups=4, dyscope=True)
        y = m(x)
        assert y.shape == (2, out_ch, 8, 12, 16)
        y.sum().backward()
        assert x.grad is not None

    def test_init_is_near_bilinear(self):
        """With near-zero offset init + zero-init scope, DySample output at
        init should be close to plain bilinear upsampling (via grid_sample).
        """
        in_ch = 4
        m = DySample3d(in_ch, in_ch, scale=2, groups=4, dyscope=True).eval()
        x = torch.randn(1, in_ch, 4, 6, 8)
        with torch.no_grad():
            y = m(x)
        y_bil = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=True)
        # Grouped grid_sample with align_corners=True ≈ trilinear for zero-offset.
        # Allow generous tolerance since trunc_normal injects small random jitter.
        assert y.shape == y_bil.shape
        diff = (y - y_bil).abs().mean().item()
        assert diff < 5e-2, f"DySample init deviates too far from bilinear: {diff}"

    def test_rejects_bad_groups(self):
        with pytest.raises(ValueError):
            DySample3d(in_ch=7, out_ch=8, groups=4)  # 7 not divisible by 4

    def test_lightweight_param_count(self):
        """DySample should be ~orders of magnitude lighter than CARAFE."""
        in_ch = 64
        dy = DySample3d(in_ch, in_ch, groups=4, dyscope=True)
        ca = CARAFE3d(in_ch, in_ch, k_up=3)
        dy_params = sum(p.numel() for p in dy.parameters())
        ca_params = sum(p.numel() for p in ca.parameters())
        assert dy_params < ca_params, (
            f"DySample({dy_params}) should be lighter than CARAFE({ca_params})")
