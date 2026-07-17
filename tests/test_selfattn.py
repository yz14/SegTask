"""Tests for the content-based self-attention (QKV / linear QKV) blocks.

Covers:
- SelfAttentionBlock: shape preservation + grad flow (softmax/linear, 2D/3D),
  head_dim head resolution, and zero-init => identity-at-init.
- factory wiring: encoder-only / decoder-only / both, per-stage masks.
- default-off (selfattn_enabled=False) is byte-identical to baseline.
- config.validate rejects malformed settings incl. softmax O(N^2) token guard.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from taskcore.config.core import Config, ConfigError, resolve_selfattn_stage
from taskcore.models.blocks import (
    SelfAttentionBlock, _GridQKVAttention, _LinearQKVAttention,
    _SoftmaxQKVAttention, _WindowQKVAttention, _apply_rope_nd,
    _window_partition_tokens, _window_unpartition_tokens, _ROPE_ND_CACHE)
from taskcore.models.factory import build_model


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


def test_softmax_qkv_attention_matches_sdpa_reference():
    qkv = torch.randn(2, 96, 64)
    attn = _SoftmaxQKVAttention(num_heads=4)

    out = attn(qkv)

    h = attn.num_heads
    c = qkv.shape[1] // (3 * h)
    qkv_h = qkv.view(qkv.shape[0] * h, 3 * c, qkv.shape[2])
    q, k, v = qkv_h.split(c, dim=1)
    scale = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(float(c))))
    weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    ref = torch.einsum("bts,bcs->bct", weight, v)
    ref = ref.reshape(qkv.shape[0], h * c, qkv.shape[2])

    max_diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), max_diff


def test_rope_preserves_norm_and_relative_logits():
    q = torch.randn(2, 3, 24, 18)
    k = torch.randn(2, 3, 24, 18)
    shape = (2, 3, 4)
    q_rot, k_rot = _apply_rope_nd(q, k, shape)
    q_shift, k_shift = _apply_rope_nd(q, k, shape, position_offsets=(1, 2, 3))

    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-6, rtol=1e-6)

    ref_logits = torch.einsum("bhnd,bhmd->bhnm", q_rot, k_rot)
    shifted_logits = torch.einsum("bhnd,bhmd->bhnm", q_shift, k_shift)
    assert torch.allclose(ref_logits, shifted_logits, atol=1e-5, rtol=1e-5)


def test_apply_rope_nd_reuses_cache_for_repeated_calls():
    cache_before = len(_ROPE_ND_CACHE)
    q = torch.randn(1, 2, 60, 24)
    k = torch.randn(1, 2, 60, 24)
    shape = (2, 5, 6)
    offsets = (3, 1, 4)

    q1, k1 = _apply_rope_nd(q, k, shape, position_offsets=offsets)
    cache_mid = len(_ROPE_ND_CACHE)
    q2, k2 = _apply_rope_nd(q, k, shape, position_offsets=offsets)

    assert cache_mid > cache_before
    assert len(_ROPE_ND_CACHE) == cache_mid
    assert torch.allclose(q1, q2)
    assert torch.allclose(k1, k2)


def test_selfattn_rope_rejects_linear():
    with pytest.raises(ValueError, match="only supported with 'softmax'"):
        SelfAttentionBlock(16, attn_type="linear", use_rope=True)


@pytest.mark.parametrize("attn_type", ["softmax", "linear"])
def test_selfattn_zero_init_is_identity(attn_type):
    """zero-init proj => block output == input at initialization (residual)."""
    block = SelfAttentionBlock(16, attn_type=attn_type, num_heads=4,
                               spatial_dims=3, zero_init=True)
    x = torch.randn(2, 16, 4, 8, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


@pytest.mark.parametrize("spatial_dims,shape", [(2, (8, 8)), (3, (4, 8, 8))])
def test_selfattn_rope_ffn_shape_and_grad(spatial_dims, shape):
    block = SelfAttentionBlock(
        16, attn_type="softmax", num_heads=4, spatial_dims=spatial_dims,
        zero_init=False, use_rope=True, use_ffn=True)
    x = torch.randn(2, 16, *shape, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_selfattn_rope_ffn_zero_init_identity_and_state_dict():
    block = SelfAttentionBlock(
        16, attn_type="softmax", num_heads=4, spatial_dims=3,
        zero_init=True, use_rope=True, use_ffn=True)
    keys = set(block.state_dict().keys())
    assert keys == {
        "norm.weight", "norm.bias",
        "qkv.weight", "qkv.bias",
        "proj.weight", "proj.bias",
        "ffn_norm.weight", "ffn_norm.bias",
        "ffn_in.weight", "ffn_in.bias",
        "ffn_out.weight", "ffn_out.bias",
    }
    x = torch.randn(2, 16, 4, 8, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_selfattn_ffn_changes_when_perturbed():
    block = SelfAttentionBlock(
        16, attn_type="softmax", num_heads=4, spatial_dims=3,
        zero_init=True, use_ffn=True)
    x = torch.randn(2, 16, 4, 8, 8)
    y0 = block(x)
    assert torch.allclose(y0, x, atol=1e-6)
    with torch.no_grad():
        block.ffn_out.weight.normal_()
        block.ffn_out.bias.normal_()
    y1 = block(x)
    assert not torch.allclose(y1, x)


def test_selfattn_off_matches_reference_and_state_dict():
    block = SelfAttentionBlock(
        16, attn_type="softmax", num_heads=4, spatial_dims=3,
        zero_init=False, use_rope=False, use_ffn=False)
    assert set(block.state_dict().keys()) == {
        "norm.weight", "norm.bias",
        "qkv.weight", "qkv.bias",
        "proj.weight", "proj.bias",
    }
    x = torch.randn(2, 16, 4, 8, 8)
    out = block(x)

    spatial = x.shape[2:]
    h = rearrange(block.norm(x), "b c ... -> b c (...)")
    h = block.qkv(h)
    qkv_h = rearrange(h, "b (h c3) n -> b h c3 n", h=block.num_heads)
    q, k, v = qkv_h.chunk(3, dim=2)
    q = q.permute(0, 1, 3, 2)
    k = k.permute(0, 1, 3, 2)
    v = v.permute(0, 1, 3, 2)
    ref = F.scaled_dot_product_attention(q, k, v)
    ref = ref.permute(0, 1, 3, 2)
    ref = rearrange(ref, "b h c n -> b (h c) n")
    ref = block.proj(ref).unflatten(-1, spatial)
    ref = x + ref
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("attn_cls,attn_type,shape", [
    (_WindowQKVAttention, "window", (4, 8, 8)),
    (_GridQKVAttention, "grid", (4, 8, 8)),
    (_WindowQKVAttention, "window", (3, 7, 5)),
    (_GridQKVAttention, "grid", (3, 7, 5)),
])
def test_window_grid_attention_shape_and_grad(attn_cls, attn_type, shape):
    if attn_type == "window":
        attn = attn_cls(num_heads=4, window_size=4, spatial_dims=len(shape))
    else:
        attn = attn_cls(num_heads=4, grid_size=4, spatial_dims=len(shape))
    qkv = torch.randn(2, 96, math.prod(shape), requires_grad=True)
    y = attn(qkv, spatial_shape=shape)
    assert y.shape == (2, 32, math.prod(shape))
    y.sum().backward()
    assert qkv.grad is not None


def test_window_attention_is_local_and_grid_spans_residue_classes():
    shape = (8, 8)
    n_tok = math.prod(shape)
    qkv = torch.zeros(1, 96, n_tok)
    qkv[:, 64:, 0] = 1.0
    win = _WindowQKVAttention(num_heads=4, window_size=4, spatial_dims=2)
    grid = _GridQKVAttention(num_heads=4, grid_size=4, spatial_dims=2)
    out_win = win(qkv, spatial_shape=shape)
    out_grid = grid(qkv, spatial_shape=shape)
    far_token = 32  # (4,0): same grid residue as token 0, different 4x4 window
    assert torch.allclose(out_win[..., far_token], torch.zeros_like(out_win[..., far_token]), atol=1e-6)
    assert out_grid[..., far_token].abs().max().item() > 0


def test_window_rope_relative_logits_with_offsets():
    q = torch.randn(1, 4, 16, 8)
    k = torch.randn(1, 4, 16, 8)
    q0, k0 = _apply_rope_nd(q, k, (4, 4), position_offsets=(0, 0))
    q1, k1 = _apply_rope_nd(q, k, (4, 4), position_offsets=(4, 0))
    ref0 = torch.einsum("bhnd,bhmd->bhnm", q0, k0)
    ref1 = torch.einsum("bhnd,bhmd->bhnm", q1, k1)
    assert torch.allclose(ref0, ref1, atol=1e-5, rtol=1e-5)


def test_rope_grid_rejected_and_window_allowed():
    with pytest.raises(ValueError, match="not supported with 'grid'"):
        SelfAttentionBlock(16, attn_type="grid", use_rope=True)
    block = SelfAttentionBlock(
        16, attn_type="window", use_rope=True, window_size=4,
        spatial_dims=2, zero_init=False)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_window_padding_matches_manual_crop():
    block = SelfAttentionBlock(
        16, attn_type="window", window_size=4, spatial_dims=2,
        zero_init=False)
    x = torch.randn(1, 16, 10, 10)
    y = block(x)
    assert y.shape == x.shape
    h = rearrange(block.norm(x), "b c ... -> b c (...)")
    h = block.qkv(h)
    qkv_h = rearrange(h, "b (h c3) n -> b h c3 n", h=block.num_heads)
    q, k, v = qkv_h.chunk(3, dim=2)
    q = rearrange(q.permute(0, 1, 3, 2).unflatten(-2, x.shape[2:]),
                  "b h ... c -> b h c ...")
    k = rearrange(k.permute(0, 1, 3, 2).unflatten(-2, x.shape[2:]),
                  "b h ... c -> b h c ...")
    v = rearrange(v.permute(0, 1, 3, 2).unflatten(-2, x.shape[2:]),
                  "b h ... c -> b h c ...")
    q, mask, meta = _window_partition_tokens(q, x.shape[2:], 4)
    k, _, _ = _window_partition_tokens(k, x.shape[2:], 4)
    v, _, _ = _window_partition_tokens(v, x.shape[2:], 4)
    attn_mask = torch.zeros((mask.shape[0], 1, 1, mask.shape[1]),
                            device=q.device, dtype=q.dtype)
    attn_mask = attn_mask.masked_fill(~mask[:, None, None, :],
                                      torch.finfo(q.dtype).min)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    ref = _window_unpartition_tokens(ref, meta)
    ref = rearrange(ref, "b h c ... -> b (h c) ...").flatten(2)
    ref = block.proj(ref).unflatten(-1, x.shape[2:])
    ref = x + ref
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)


def test_selfattn_head_dim_resolution():
    block = SelfAttentionBlock(64, attn_type="softmax", head_dim=32,
                               spatial_dims=3)
    assert block.num_heads == 2


def test_selfattn_block_rejects_bad_type_and_heads():
    with pytest.raises(ValueError):
        SelfAttentionBlock(16, attn_type="bogus")
    with pytest.raises(ValueError):
        SelfAttentionBlock(18, attn_type="softmax", num_heads=4)  # 18 % 4 != 0


@pytest.mark.parametrize("attn_type", ["window", "grid"])
def test_unet_forward_with_window_grid_selfattn(attn_type):
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_type = attn_type
    cfg.model.selfattn_window_size = 4
    cfg.model.selfattn_grid_size = 4
    cfg.model.selfattn_encoder_stages = [0, 1, 1]
    cfg.model.selfattn_decoder_stages = []
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).eval()
    attns = [m for m in model.modules()
             if isinstance(m, (_WindowQKVAttention, _GridQKVAttention))]
    assert attns
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-3:] == (16, 64, 64)


def test_default_off_state_dict_unchanged_with_window_grid():
    base = _cfg("z_axis")
    base.sync()
    base.validate()
    off = _cfg("z_axis")
    off.model.selfattn_enabled = True
    off.model.selfattn_type = "window"
    off.model.selfattn_encoder_stages = []
    off.model.selfattn_decoder_stages = []
    off.sync()
    off.validate()
    assert set(build_model(base).state_dict().keys()) == set(
        build_model(off).state_dict().keys())


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


def test_unet_forward_with_selfattn_rope_ffn_threading():
    cfg = _cfg("z_axis")
    cfg.model.selfattn_enabled = True
    cfg.model.selfattn_type = "softmax"
    cfg.model.selfattn_rope = True
    cfg.model.selfattn_ffn = True
    cfg.model.selfattn_ffn_ratio = 2.0
    cfg.model.selfattn_encoder_stages = [0, 1, 1]
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).eval()
    blocks = [m for m in model.modules() if isinstance(m, SelfAttentionBlock)]
    assert blocks
    assert all(m.use_rope for m in blocks)
    assert all(m.use_ffn for m in blocks)
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-3:] == (16, 64, 64)


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
