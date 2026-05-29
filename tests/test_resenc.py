"""Unit tests for nnU-Net ResEncUNet-style deeper encoder blocks.

Covers:
- New residual block variants: PreActResNetBlock, BottleneckBlock
- ResNetStage dispatch over block_type="basic"/"preact"/"bottleneck"
- Per-stage block count lists (asymmetric encoder/decoder depths)
- ResEnc presets S/M/L/XL auto-populate per-stage counts
- End-to-end build_model with preset + block_type variants
- Parameter count: encoder scales with per-stage blocks; decoder stays light
- Config validation for new fields
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from segtask_v1.models.resnet import (
    BLOCK_TYPES,
    BottleneckBlock,
    PreActResNetBlock,
    ResNetBlock,
    ResNetStage,
    _make_block,
)


# ---------------------------------------------------------------------------
# Individual block variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [ResNetBlock, PreActResNetBlock, BottleneckBlock])
def test_block_shape_and_grad(cls):
    blk = cls(in_ch=8, out_ch=16)
    x = torch.randn(2, 8, 4, 8, 8, requires_grad=True)
    y = blk(x)
    assert y.shape == (2, 16, 4, 8, 8)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_preact_shortcut_is_unnormed():
    """Canonical pre-act design: shortcut operates on raw x (no norm)."""
    blk = PreActResNetBlock(in_ch=8, out_ch=16)
    # 1×1×1 projection without BN/IN — shortcut is a bare Conv3d.
    assert isinstance(blk.shortcut, nn.Conv3d)
    # Same-channel case: shortcut is Identity, never applies norm.
    blk2 = PreActResNetBlock(in_ch=16, out_ch=16)
    assert isinstance(blk2.shortcut, nn.Identity)


def test_bottleneck_reduces_params_vs_basic_at_wide_channels():
    """Bottleneck's 1x1 reduce → 3x3x3 (on ch/expansion) → 1x1 expand
    design trades depth for width: at wide channels it is much cheaper
    than two full 3x3x3 convs. (At narrow channels the two designs are
    comparable.)"""
    basic = ResNetBlock(256, 256)
    btl   = BottleneckBlock(256, 256, expansion=4)
    b_params = sum(p.numel() for p in basic.parameters())
    t_params = sum(p.numel() for p in btl.parameters())
    assert t_params < b_params, (
        f"Bottleneck ({t_params}) should be lighter than basic ({b_params}) "
        f"at wide channels")


def test_make_block_rejects_unknown():
    with pytest.raises(ValueError):
        _make_block("bogus", 8, 16)


# ---------------------------------------------------------------------------
# ResNetStage block_type dispatch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_type", BLOCK_TYPES)
def test_stage_block_type_shape(block_type):
    stage = ResNetStage(8, 16, num_blocks=3, block_type=block_type)
    x = torch.randn(1, 8, 4, 8, 8)
    assert stage(x).shape == (1, 16, 4, 8, 8)


def test_stage_uniform_block_type():
    """All blocks inside a stage are of the requested type."""
    stage = ResNetStage(8, 16, num_blocks=3, block_type="preact")
    for b in stage.blocks:
        assert isinstance(b, PreActResNetBlock)


def test_stage_rejects_zero_blocks():
    with pytest.raises(ValueError):
        ResNetStage(8, 16, num_blocks=0)


# ---------------------------------------------------------------------------
# ResEnc preset expansion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("preset,expected_first_three", [
    ("S",  [1, 2, 2]),
    ("M",  [1, 3, 4]),
    ("L",  [1, 3, 4]),
    ("XL", [1, 4, 6]),
])
def test_preset_populates_encoder_blocks(preset, expected_first_three):
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.encoder_channels = [32, 64, 128, 256, 320]
    cfg.model.resenc_preset = preset
    cfg.sync()
    assert cfg.model.encoder_blocks_per_stage[:3] == expected_first_three
    # Matches encoder depth.
    assert len(cfg.model.encoder_blocks_per_stage) == 5
    # Decoder: lightweight 1 / stage.
    assert cfg.model.decoder_blocks_per_stage == [1, 1, 1, 1]


def test_preset_extends_deeper_than_template():
    """If user has more encoder levels than preset template length, extend
    by repeating the deepest count."""
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.encoder_channels = [16, 32, 64, 128, 256, 320, 320, 320, 320]
    cfg.model.resenc_preset = "M"
    cfg.sync()
    # Template "M" = [1, 3, 4, 6, 6, 6], len=6; user n=9 → extend 3 entries.
    assert len(cfg.model.encoder_blocks_per_stage) == 9
    assert cfg.model.encoder_blocks_per_stage[-3:] == [6, 6, 6]
    assert cfg.model.encoder_blocks_per_stage[:6] == [1, 3, 4, 6, 6, 6]


def test_user_counts_override_preset():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.encoder_blocks_per_stage = [2, 2, 2]
    cfg.model.decoder_blocks_per_stage = [1, 1]
    cfg.model.resenc_preset = "M"
    cfg.sync()
    # User values preserved.
    assert cfg.model.encoder_blocks_per_stage == [2, 2, 2]
    assert cfg.model.decoder_blocks_per_stage == [1, 1]


# ---------------------------------------------------------------------------
# End-to-end build with preset / block_type
# ---------------------------------------------------------------------------
def _make_test_cfg(**overrides):
    from segtask_v1.config import Config
    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [8, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    for k, v in overrides.items():
        setattr(cfg.model, k, v)
    cfg.sync()
    cfg.validate()
    return cfg


@pytest.mark.parametrize("block_type", BLOCK_TYPES)
def test_build_model_with_block_type(block_type):
    from segtask_v1.models.factory import build_model
    cfg = _make_test_cfg(block_type=block_type)
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 8, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-3:] == (8, 32, 32)


@pytest.mark.parametrize("preset", ["S", "M", "L", "XL"])
def test_build_model_with_preset(preset):
    from segtask_v1.models.factory import build_model
    cfg = _make_test_cfg(resenc_preset=preset)
    model = build_model(cfg).train()
    x = torch.randn(1, cfg.model.in_channels, 8, 32, 32)
    outs = model(x)
    main = outs[0] if isinstance(outs, list) else outs
    assert main.shape[-3:] == (8, 32, 32)


def test_asymmetric_encoder_decoder_depth():
    """ResEnc hallmark: encoder much deeper than decoder."""
    from segtask_v1.models.factory import build_model
    cfg = _make_test_cfg(
        encoder_blocks_per_stage=[1, 4, 4],
        decoder_blocks_per_stage=[1, 1],
    )
    model = build_model(cfg)
    pc = model.param_count()
    # Encoder should carry ≥ 2× the parameters of the decoder (3 levels of
    # deep blocks vs 2 levels of single blocks).
    assert pc["encoder"] >= 2 * pc["decoder"], (
        f"Expected asymmetric encoder-heavy budget, got "
        f"enc={pc['encoder']}, dec={pc['decoder']}")


def test_preset_with_preact_and_unetpp():
    """ResEnc preset composes with pre-act blocks AND UNet++ decoder."""
    from segtask_v1.models.factory import build_model
    cfg = _make_test_cfg(
        resenc_preset="M",
        block_type="preact",
        decoder_type="unetpp",
    )
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 8, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-3:] == (8, 32, 32)


def test_convnext_respects_per_stage_counts():
    """ConvNeXt backbone also honours per-stage counts."""
    from segtask_v1.models.factory import build_model
    cfg = _make_test_cfg(
        backbone="convnext",
        encoder_blocks_per_stage=[1, 2, 3],
        decoder_blocks_per_stage=[1, 1],
    )
    model = build_model(cfg)
    # Encoder.stages is a ModuleList of ConvNeXtStage. Each stage's
    # internal `blocks` should have the expected count.
    n_blocks_per_stage = [len(st.blocks) for st in model.encoder.stages]
    assert n_blocks_per_stage == [1, 2, 3]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_rejects_bad_block_type(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.block_type = "bogus"
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_rejects_bad_preset(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.resenc_preset = "ludicrous"
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_rejects_mismatched_encoder_list(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.encoder_blocks_per_stage = [1, 2]  # wrong length
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_rejects_mismatched_decoder_list(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.decoder_blocks_per_stage = [1]  # should be 2
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_rejects_zero_blocks_in_list(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.encoder_blocks_per_stage = [1, 0, 2]
        with pytest.raises(AssertionError):
            cfg.validate()
