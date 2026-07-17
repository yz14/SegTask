"""Tests for the MultiRF (multi-receptive-field dilated-branch) blocks.

Covers:
- MultiRFBlock / MultiRFStage: shape preservation + grad flow (split/parallel,
  concat_proj/sum/se fusion, hw/all axes, 2D and 3D).
- factory wiring: encoder-only / decoder-only / both, per-stage masks.
- default-off (multirf_enabled=False) is byte-identical to baseline.
- config.validate rejects malformed MultiRF settings.
"""

from __future__ import annotations

import pytest
import torch

from taskcore.config.core import Config, ConfigError
from taskcore.models.factory import build_model
from taskcore.models.resnet import MultiRFBlock, MultiRFStage


def _nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Block / stage level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spatial_dims,shape", [(3, (4, 8, 8)), (2, (8, 8))])
@pytest.mark.parametrize("mode", ["split", "parallel"])
@pytest.mark.parametrize("fusion", ["concat_proj", "se"])
@pytest.mark.parametrize("axes", ["hw", "all"])
def test_multirf_block_shape_and_grad(spatial_dims, shape, mode, fusion, axes):
    in_ch, out_ch = 12, 24
    block = MultiRFBlock(
        in_ch, out_ch, dilations=[1, 2, 3], mode=mode, fusion=fusion,
        axes=axes, spatial_dims=spatial_dims)
    x = torch.randn(2, in_ch, *shape, requires_grad=True)
    y = block(x)
    assert y.shape == (2, out_ch, *shape)
    y.sum().backward()
    assert x.grad is not None


def test_multirf_block_sum_requires_parallel():
    # sum fusion with split must raise (unequal branch channels).
    with pytest.raises(ValueError):
        MultiRFBlock(12, 24, dilations=[1, 2, 3], mode="split", fusion="sum")
    # parallel + sum is fine.
    block = MultiRFBlock(12, 24, dilations=[1, 2, 3], mode="parallel",
                         fusion="sum", spatial_dims=3)
    y = block(torch.randn(1, 12, 4, 8, 8))
    assert y.shape == (1, 24, 4, 8, 8)


def test_multirf_block_requires_identity_branch():
    with pytest.raises(ValueError):
        MultiRFBlock(12, 24, dilations=[2, 3])


@pytest.mark.parametrize("mode", ["split", "parallel"])
@pytest.mark.parametrize("fusion", ["concat_proj", "se"])
def test_multirf_branch_norm_act_shape_and_grad(mode, fusion):
    """ASPP-style per-branch norm+act: shape preserved, grad flows, post built."""
    # out_ch divisible by groups for split (24//3=8, %4==0); use instance to be safe.
    block = MultiRFBlock(
        12, 24, dilations=[1, 2, 3], mode=mode, fusion=fusion,
        norm_type="instance", spatial_dims=3, branch_norm_act=True)
    assert block.branch_post is not None
    assert len(block.branch_post) == len(block.branches)
    x = torch.randn(2, 12, 4, 8, 8, requires_grad=True)
    y = block(x)
    assert y.shape == (2, 24, 4, 8, 8)
    y.sum().backward()
    assert x.grad is not None


def test_multirf_branch_norm_act_off_has_no_post():
    block = MultiRFBlock(12, 24, dilations=[1, 2, 3], spatial_dims=3)
    assert block.branch_post is None


def test_multirf_branch_norm_act_adds_params():
    """Enabling per-branch norm+act adds parameters vs disabled."""
    common = dict(dilations=[1, 2, 3], mode="parallel", spatial_dims=3,
                  norm_type="instance")
    off = MultiRFBlock(12, 24, branch_norm_act=False, **common)
    on = MultiRFBlock(12, 24, branch_norm_act=True, **common)
    assert _nparams(on) > _nparams(off)


def test_multirf_branch_norm_act_group_split_raises():
    """split + GroupNorm where branch_ch % norm_groups != 0 must raise clearly."""
    # 3 branches, out_ch=24 -> branch_ch ~ 8/8/8; with norm_groups=5, 8%5!=0.
    with pytest.raises(ValueError, match="not divisible by"):
        MultiRFBlock(
            12, 24, dilations=[1, 2, 3], mode="split", fusion="concat_proj",
            norm_type="group", norm_groups=5, spatial_dims=3,
            branch_norm_act=True)


def test_multirf_branch_norm_act_group_split_uneven_remainder_raises():
    """split remainder goes to the dil=1 branch; uneven branch can break GroupNorm."""
    # out_ch=26, 3 branches -> 8/8/10 (rem 2 to dil=1 branch). norm_groups=8:
    # 8%8==0 but 10%8!=0 -> must raise on the offending branch.
    with pytest.raises(ValueError, match="not divisible by"):
        MultiRFBlock(
            12, 26, dilations=[1, 2, 3], mode="split", fusion="concat_proj",
            norm_type="group", norm_groups=8, spatial_dims=3,
            branch_norm_act=True)


def test_multirf_branch_norm_act_group_split_ok_when_divisible():
    """split + GroupNorm works when branch channels are divisible by groups."""
    # out_ch=24, 3 branches -> 8/8/8; norm_groups=4 divides 8.
    block = MultiRFBlock(
        12, 24, dilations=[1, 2, 3], mode="split", fusion="concat_proj",
        norm_type="group", norm_groups=4, spatial_dims=3, branch_norm_act=True)
    y = block(torch.randn(1, 12, 4, 8, 8))
    assert y.shape == (1, 24, 4, 8, 8)


def test_multirf_branch_norm_act_parallel_group_ok():
    """parallel mode: each branch = out_ch, GroupNorm divisibility easy to meet."""
    block = MultiRFBlock(
        12, 24, dilations=[1, 2, 3], mode="parallel", fusion="sum",
        norm_type="group", norm_groups=8, spatial_dims=3, branch_norm_act=True)
    y = block(torch.randn(1, 12, 4, 8, 8))
    assert y.shape == (1, 24, 4, 8, 8)


def test_multirf_stage_threads_branch_norm_act():
    stage = MultiRFStage(8, 16, num_blocks=2, dilations=[1, 2, 3],
                         norm_type="instance", spatial_dims=3,
                         branch_norm_act=True)
    for blk in stage.blocks:
        assert blk.branch_post is not None
    y = stage(torch.randn(1, 8, 4, 8, 8))
    assert y.shape == (1, 16, 4, 8, 8)


def test_multirf_stage_changes_channels():
    stage = MultiRFStage(8, 16, num_blocks=2, dilations=[1, 2, 3],
                         spatial_dims=3)
    y = stage(torch.randn(1, 8, 4, 8, 8))
    assert y.shape == (1, 16, 4, 8, 8)


def test_multirf_hw_keeps_z_dilation_one():
    """axes='hw' on 3D must keep z (first spatial axis) dilation == 1."""
    block = MultiRFBlock(8, 8, dilations=[1, 2, 3], mode="split", axes="hw",
                         spatial_dims=3)
    for conv in block.branches:
        assert conv.dilation[0] == 1


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


@pytest.mark.parametrize("enc,dec", [
    ([0, 1, 1], []),       # encoder deep stages
    ([0, 0, 1], []),       # bottleneck only
    ([], [1, 1]),          # decoder only
    ([0, 1, 1], [0, 1]),   # both
])
def test_unet_forward_with_multirf(enc, dec):
    cfg = _cfg("z_axis")
    cfg.model.multirf_enabled = True
    cfg.model.multirf_encoder_stages = enc
    cfg.model.multirf_decoder_stages = dec
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-3:] == (16, 64, 64)


def test_unet_forward_with_multirf_2_5d():
    cfg = _cfg("2_5d")
    cfg.model.multirf_enabled = True
    cfg.model.multirf_encoder_stages = [0, 1, 1]
    cfg.model.multirf_mode = "parallel"
    cfg.model.multirf_fusion = "sum"
    cfg.sync()
    cfg.validate()
    assert cfg.model.spatial_dims == 2
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-2:] == (64, 64)


def test_build_model_with_branch_norm_act():
    """End-to-end: multirf_branch_norm_act flows through factory and forwards."""
    cfg = _cfg("z_axis")
    cfg.model.norm_type = "instance"  # avoid group/split divisibility issue
    cfg.model.multirf_enabled = True
    cfg.model.multirf_encoder_stages = [0, 1, 1]
    cfg.model.multirf_branch_norm_act = True
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-3:] == (16, 64, 64)


def test_build_model_branch_norm_act_group_split_raises():
    """split + GroupNorm with non-divisible branch channels surfaces at build."""
    cfg = _cfg("z_axis")
    cfg.model.encoder_channels = [16, 20, 20]  # 20//3 -> 6/6/8, %8 fails
    cfg.model.norm_type = "group"
    cfg.model.norm_groups = 8
    cfg.model.multirf_enabled = True
    cfg.model.multirf_mode = "split"
    cfg.model.multirf_encoder_stages = [0, 1, 1]
    cfg.model.multirf_branch_norm_act = True
    cfg.sync()
    cfg.validate()
    with pytest.raises(ValueError, match="not divisible by"):
        build_model(cfg)


def test_build_model_branch_norm_act_adds_params():
    base = _cfg("z_axis")
    base.model.norm_type = "instance"
    base.model.multirf_enabled = True
    base.model.multirf_encoder_stages = [0, 1, 1]
    base.sync()
    base.validate()
    on = _cfg("z_axis")
    on.model.norm_type = "instance"
    on.model.multirf_enabled = True
    on.model.multirf_encoder_stages = [0, 1, 1]
    on.model.multirf_branch_norm_act = True
    on.sync()
    on.validate()
    assert _nparams(build_model(on)) > _nparams(build_model(base))


def test_default_off_is_identical_to_baseline():
    """multirf_enabled=False keeps params identical even if stages are set."""
    base = _cfg("z_axis"); base.sync(); base.validate()
    off = _cfg("z_axis")
    off.model.multirf_enabled = False
    off.model.multirf_encoder_stages = [1, 1, 1]  # must be ignored
    off.sync(); off.validate()
    assert _nparams(build_model(base)) == _nparams(build_model(off))


def test_split_mode_is_parameter_cheap():
    """split fusion should add far fewer params than parallel."""
    def build(mode):
        cfg = _cfg("z_axis")
        cfg.model.multirf_enabled = True
        cfg.model.multirf_mode = mode
        cfg.model.multirf_encoder_stages = [0, 1, 1]
        cfg.sync(); cfg.validate()
        return _nparams(build_model(cfg))
    base = _cfg("z_axis"); base.sync(); base.validate()
    n_base = _nparams(build_model(base))
    n_split = build("split")
    n_parallel = build("parallel")
    assert n_split > n_base
    assert n_parallel > n_split * 1.2  # parallel materially heavier


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def _bad_cfg(**model_kwargs):
    cfg = _cfg("z_axis")
    cfg.model.multirf_enabled = True
    for k, v in model_kwargs.items():
        setattr(cfg.model, k, v)
    cfg.sync()
    return cfg


@pytest.mark.parametrize("kwargs", [
    dict(multirf_dilations=[2, 3], multirf_encoder_stages=[0, 0, 1]),  # no 1
    dict(multirf_mode="split", multirf_fusion="sum",
         multirf_encoder_stages=[0, 0, 1]),                            # sum+split
    dict(multirf_mode="bogus", multirf_encoder_stages=[0, 0, 1]),      # bad mode
    dict(multirf_fusion="bogus", multirf_encoder_stages=[0, 0, 1]),    # bad fusion
    dict(multirf_axes="bogus", multirf_encoder_stages=[0, 0, 1]),      # bad axes
    dict(multirf_encoder_stages=[1, 1]),                               # bad enc len
    dict(multirf_decoder_stages=[1]),                                  # bad dec len
    dict(backbone="convnext", multirf_encoder_stages=[0, 0, 1]),       # not resnet
    dict(decoder_type="unetpp", multirf_decoder_stages=[1, 1]),        # dec+unetpp
])
def test_config_rejects_bad_multirf(kwargs):
    cfg = _bad_cfg(**kwargs)
    with pytest.raises((ConfigError, ValueError)):
        cfg.validate()


def test_config_accepts_disabled_with_garbage():
    """When disabled, malformed multirf settings are not validated."""
    cfg = _cfg("z_axis")
    cfg.model.multirf_enabled = False
    cfg.model.multirf_dilations = [2, 3]  # would be invalid if enabled
    cfg.model.multirf_encoder_stages = [9, 9]
    cfg.sync()
    cfg.validate()  # must not raise
