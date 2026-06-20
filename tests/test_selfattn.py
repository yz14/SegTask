"""Tests for the content-based self-attention (QKV / linear QKV) blocks.

Covers:
- SelfAttentionBlock: shape preservation + grad flow (softmax/linear, 2D/3D),
  head_dim head resolution, and zero-init => identity-at-init.
- factory wiring: encoder-only / decoder-only / both, per-stage masks.
- default-off (selfattn_enabled=False) is byte-identical to baseline.
- config.validate rejects malformed settings incl. softmax O(N^2) token guard.
"""

from __future__ import annotations

import pytest
import torch

from segtask_v1.config import Config, ConfigError, resolve_selfattn_stage
from segtask_v1.models.blocks import (
    SelfAttentionBlock, _LinearQKVAttention, _SoftmaxQKVAttention)
from segtask_v1.models.factory import build_model


def _count_attn(model):
    soft = sum(isinstance(m, _SoftmaxQKVAttention) for m in model.modules())
    lin = sum(isinstance(m, _LinearQKVAttention) for m in model.modules())
    return soft, lin


def _nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Block level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spatial_dims,shape", [(3, (4, 8, 8)), (2, (8, 8))])
@pytest.mark.parametrize("attn_type", ["softmax", "linear"])
def test_selfattn_block_shape_and_grad(spatial_dims, shape, attn_type):
    block = SelfAttentionBlock(
        16, attn_type=attn_type, num_heads=4, spatial_dims=spatial_dims,
        zero_init=False)
    x = torch.randn(2, 16, *shape, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("attn_type", ["softmax", "linear"])
def test_selfattn_zero_init_is_identity(attn_type):
    """zero-init proj => block output == input at initialization (residual)."""
    block = SelfAttentionBlock(16, attn_type=attn_type, num_heads=4,
                               spatial_dims=3, zero_init=True)
    x = torch.randn(2, 16, 4, 8, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_selfattn_head_dim_resolution():
    block = SelfAttentionBlock(64, attn_type="softmax", head_dim=32,
                               spatial_dims=3)
    assert block.num_heads == 2


def test_selfattn_block_rejects_bad_type_and_heads():
    with pytest.raises(ValueError):
        SelfAttentionBlock(16, attn_type="bogus")
    with pytest.raises(ValueError):
        SelfAttentionBlock(18, attn_type="softmax", num_heads=4)  # 18 % 4 != 0


# ---------------------------------------------------------------------------
# End-to-end build_model wiring
# ---------------------------------------------------------------------------
def _cfg(patch_mode="z_axis"):
    cfg = Config()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1, 2]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    return cfg


@pytest.mark.parametrize("attn_type,enc,dec", [
    ("softmax", [0, 1, 1], []),        # encoder deep stages
    ("softmax", [0, 0, 1], []),        # bottleneck only
    ("softmax", [], [1, 0]),           # decoder only (shallow level kept off)
    ("softmax", [0, 1, 1], [1, 0]),    # both
    ("linear",  [1, 1, 1], [1, 1]),    # linear everywhere (O(N), no guard)
])
def test_unet_forward_with_selfattn(attn_type, enc, dec):
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_type = attn_type
    cfg.model.selfattn_encoder_stages = enc
    cfg.model.selfattn_decoder_stages = dec
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-3:] == (16, 64, 64)


def test_unet_forward_with_selfattn_2_5d():
    cfg = _cfg("2_5d")
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_type = "softmax"
    cfg.model.selfattn_encoder_stages = [0, 1, 1]
    cfg.model.selfattn_decoder_stages = [1, 1]
    cfg.sync()
    cfg.validate()
    assert cfg.model.spatial_dims == 2
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-2:] == (64, 64)


def test_default_off_is_identical_to_baseline():
    """selfattn_enabled=False keeps params identical even if stages are set."""
    base = _cfg("z_axis"); base.sync(); base.validate()
    off = _cfg("z_axis")
    off.model.selfattn_enabled = False
    off.model.selfattn_encoder_stages = [1, 1, 1]  # must be ignored
    off.sync(); off.validate()
    assert _nparams(build_model(base)) == _nparams(build_model(off))


def test_enabled_adds_params():
    base = _cfg("z_axis"); base.sync(); base.validate()
    on = _cfg("z_axis")
    on.model.selfattn_enabled = True
    on.model.selfattn_encoder_stages = [0, 1, 1]
    on.sync(); on.validate()
    assert _nparams(build_model(on)) > _nparams(build_model(base))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def _bad_cfg(**model_kwargs):
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    for k, v in model_kwargs.items():
        setattr(cfg.model, k, v)
    cfg.sync()
    return cfg


@pytest.mark.parametrize("kwargs", [
    dict(selfattn_type="bogus", selfattn_encoder_stages=[0, 0, 1]),       # bad type
    dict(selfattn_num_heads=0, selfattn_encoder_stages=[0, 0, 1]),        # bad heads
    dict(selfattn_encoder_stages=[1, 1]),                                 # bad enc len
    dict(selfattn_decoder_stages=[1]),                                    # bad dec len
    dict(selfattn_num_heads=5, selfattn_encoder_stages=[0, 0, 1]),        # 64 % 5 != 0
    dict(selfattn_head_dim=24, selfattn_encoder_stages=[0, 0, 1]),        # 64 % 24 != 0
    dict(backbone="convnext", selfattn_encoder_stages=[0, 0, 1]),         # not resnet
    dict(decoder_type="unetpp", selfattn_decoder_stages=[1, 1]),          # dec+unetpp
    dict(selfattn_type="softmax", selfattn_encoder_stages=[1, 0, 0]),     # O(N^2) guard
])
def test_config_rejects_bad_selfattn(kwargs):
    cfg = _bad_cfg(**kwargs)
    with pytest.raises((ConfigError, ValueError)):
        cfg.validate()


def test_softmax_guard_allows_linear_on_shallow():
    """The O(N^2) guard applies to softmax only; linear is allowed shallow."""
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_type = "linear"
    cfg.model.selfattn_encoder_stages = [1, 0, 0]  # shallow, huge token count
    cfg.sync()
    cfg.validate()  # must not raise


def test_resolve_selfattn_stage():
    assert resolve_selfattn_stage(0, "softmax") is None
    assert resolve_selfattn_stage("none", "softmax") is None
    assert resolve_selfattn_stage(1, "linear") == "linear"      # 1 -> default
    assert resolve_selfattn_stage("default", "softmax") == "softmax"
    assert resolve_selfattn_stage("softmax", "linear") == "softmax"
    assert resolve_selfattn_stage("linear", "softmax") == "linear"
    with pytest.raises(ConfigError):
        resolve_selfattn_stage("bogus", "softmax")


def test_per_level_mixed_types_build():
    """Different layers use different attention types in one network."""
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    # stage1 (8192 tok) linear, stage2 bottleneck (1024 tok) softmax.
    cfg.model.selfattn_encoder_stages = [0, "linear", "softmax"]
    cfg.model.selfattn_decoder_stages = ["linear", 0]
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).eval()
    soft, lin = _count_attn(model)
    assert soft == 1 and lin == 2          # 1 softmax (enc), 2 linear (enc+dec)
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-3:] == (16, 64, 64)


def test_per_level_softmax_guard_only_hits_softmax_layers():
    """Shallow 'linear' is fine, but shallow 'softmax' is rejected."""
    ok = _cfg("z_axis")
    ok.model.selfattn_enabled = True
    ok.model.selfattn_encoder_stages = ["linear", 0, "softmax"]  # shallow=linear
    ok.sync()
    ok.validate()  # must not raise
    bad = _cfg("z_axis")
    bad.model.selfattn_enabled = True
    bad.model.selfattn_encoder_stages = ["softmax", 0, "softmax"]  # shallow=softmax
    bad.sync()
    with pytest.raises((ConfigError, ValueError)):
        bad.validate()


def test_legacy_0_1_still_works():
    """Old 0/1 masks keep working: 1 means the global selfattn_type."""
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_type = "linear"
    cfg.model.selfattn_encoder_stages = [1, 1, 1]   # all -> linear
    cfg.sync()
    cfg.validate()
    model = build_model(cfg)
    soft, lin = _count_attn(model)
    assert soft == 0 and lin == 3


def test_config_rejects_invalid_stage_string():
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_encoder_stages = [0, 0, "bogus"]
    cfg.sync()
    with pytest.raises((ConfigError, ValueError)):
        cfg.validate()


def test_config_accepts_disabled_with_garbage():
    """When disabled, malformed selfattn settings are not validated."""
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = False
    cfg.model.selfattn_type = "bogus"          # would be invalid if enabled
    cfg.model.selfattn_encoder_stages = [9, 9]
    cfg.sync()
    cfg.validate()  # must not raise
