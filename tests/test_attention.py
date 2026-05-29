"""Unit tests for attention modules + attention-gated skip connections.

Covers:
- ECA3D, CBAM3D, CoordAttention3D, AttentionGate3D: shape + grad flow
- make_attention factory: dispatch + unknown-name rejection
- Legacy ``use_se`` still activates SE when ``attention_type='none'``
- Stage-level injection in ResNetStage and ConvNeXtStage
- Skip-attention-gated UNet end-to-end forward
- Config validation rejects unknown ``attention_type``
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from segtask_v1.models.blocks import (
    ATTENTION_TYPES,
    AttentionGate3D,
    CBAM3D,
    CoordAttention3D,
    ECA3D,
    SqueezeExcite3D,
    make_attention,
)


# ---------------------------------------------------------------------------
# Individual attention modules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls,channels", [
    (SqueezeExcite3D, 16),
    (ECA3D, 16),
    (CBAM3D, 16),
    (CoordAttention3D, 64),
])
def test_attention_shape_and_grad(cls, channels):
    x = torch.randn(2, channels, 4, 8, 8, requires_grad=True)
    attn = cls(channels)
    y = attn(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_eca_adaptive_kernel_is_odd():
    """The adaptively-chosen 1D conv kernel must always be odd."""
    for c in (8, 16, 32, 64, 128, 256, 512):
        m = ECA3D(c)
        assert m.conv.kernel_size[0] % 2 == 1


def test_cbam_rejects_even_spatial_kernel():
    with pytest.raises(ValueError):
        CBAM3D(channels=16, spatial_kernel=6)


def test_coord_attention_three_axes_independent():
    """CoordAttention produces three axis-wise maps that modulate x."""
    ca = CoordAttention3D(channels=16, reduction=4).eval()
    x = torch.randn(1, 16, 4, 6, 8)
    with torch.no_grad():
        y = ca(x)
    assert y.shape == x.shape
    # Attention is sigmoid → strictly positive gain, |y| <= |x|.
    assert (y.abs() <= x.abs() + 1e-5).all()


# ---------------------------------------------------------------------------
# make_attention factory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ATTENTION_TYPES)
def test_factory_dispatch(name):
    attn = make_attention(name, channels=16)
    if name == "none":
        assert isinstance(attn, nn.Identity)
    else:
        x = torch.randn(1, 16, 2, 4, 4)
        y = attn(x)
        assert y.shape == x.shape


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        make_attention("transformer", channels=16)


# ---------------------------------------------------------------------------
# AttentionGate3D (skip gating)
# ---------------------------------------------------------------------------
class TestAttentionGate:
    def test_same_resolution(self):
        x = torch.randn(2, 16, 4, 8, 8, requires_grad=True)
        g = torch.randn(2, 8, 4, 8, 8)
        gate = AttentionGate3D(x_ch=16, g_ch=8)
        out = gate(x, g)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad is not None

    def test_auto_resizes_gating(self):
        """Gating signal at coarser resolution is interpolated up to x."""
        x = torch.randn(1, 8, 4, 8, 8)
        g = torch.randn(1, 4, 2, 4, 4)  # half resolution
        gate = AttentionGate3D(x_ch=8, g_ch=4)
        assert gate(x, g).shape == x.shape

    def test_output_bounded_by_input(self):
        """Sigmoid gate ∈ [0, 1] → |out| ≤ |x|."""
        x = torch.randn(1, 8, 4, 4, 4)
        g = torch.randn(1, 4, 4, 4, 4)
        gate = AttentionGate3D(x_ch=8, g_ch=4).eval()
        with torch.no_grad():
            out = gate(x, g)
        assert (out.abs() <= x.abs() + 1e-5).all()


# ---------------------------------------------------------------------------
# Stage-level injection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attn", ATTENTION_TYPES)
def test_resnet_stage_with_attention(attn):
    from segtask_v1.models.resnet import ResNetStage
    stage = ResNetStage(4, 8, num_blocks=2, attention_type=attn)
    x = torch.randn(1, 4, 4, 8, 8, requires_grad=True)
    y = stage(x)
    assert y.shape == (1, 8, 4, 8, 8)
    y.sum().backward()


@pytest.mark.parametrize("attn", ATTENTION_TYPES)
def test_convnext_stage_with_attention(attn):
    from segtask_v1.models.convnext import ConvNeXtStage
    stage = ConvNeXtStage(4, 8, num_blocks=2, attention_type=attn)
    x = torch.randn(1, 4, 4, 8, 8, requires_grad=True)
    y = stage(x)
    assert y.shape == (1, 8, 4, 8, 8)
    y.sum().backward()


def test_legacy_use_se_still_works():
    """use_se=True with attention_type='none' must still enable SE."""
    from segtask_v1.models.resnet import ResNetBlock
    blk = ResNetBlock(8, 8, use_se=True, attention_type="none")
    assert isinstance(blk.attn, SqueezeExcite3D)


def test_explicit_attention_type_wins_over_use_se():
    from segtask_v1.models.resnet import ResNetBlock
    blk = ResNetBlock(8, 8, use_se=True, attention_type="eca")
    assert isinstance(blk.attn, ECA3D)


# ---------------------------------------------------------------------------
# End-to-end UNet3D with skip attention + various attention types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attn", ["none", "se", "eca", "cbam", "coord"])
def test_unet_forward_with_attention(attn):
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.attention_type = attn
    cfg.model.skip_attention = True
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-3:] == (16, 64, 64)


def test_unet_convnext_skip_attention():
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model

    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "convnext"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.attention_type = "eca"
    cfg.model.skip_attention = True
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-3:] == (16, 64, 64)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_rejects_bad_attention_type():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.attention_type = "not_a_thing"
    with pytest.raises(AssertionError):
        cfg.validate()
