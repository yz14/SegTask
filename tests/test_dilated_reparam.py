"""Tests for MedNeXt dilated reparameterization.

覆盖：
- DilatedReparamBlock 的训练态 / deploy 态一致性与 kernel 展开；
- MedNeXtBlock / Stage / build_model 的端到端 reparameterize 路径；
- Config 验证与默认值；
- flag 关闭时结构保持不变。
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from segtask_v1.config import Config
from segtask_v1.models.factory import build_model
from segtask_v1.models.mednext import (
    DilatedReparamBlock,
    MedNeXtBlock,
    reparameterize_model,
)


def _make_cfg(
    patch_mode: str = "z_axis",
    deep_supervision: bool = True,
    dilated_reparam: bool = True,
) -> Config:
    cfg = Config()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.model.arch = "unet"
    cfg.model.backbone = "mednext"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.decoder_type = "unet"
    cfg.model.norm_type = "group"
    cfg.model.norm_groups = 8
    cfg.model.activation = "gelu"
    cfg.model.deep_supervision = deep_supervision
    cfg.model.mednext_kernel_size = 5
    cfg.model.mednext_expand_ratio = 4
    cfg.model.mednext_dilated_reparam = dilated_reparam
    if patch_mode == "2_5d":
        cfg.data.patch_size = [16, 64, 64]
    cfg.sync()
    cfg.validate()
    return cfg


def _flatten(out):
    if torch.is_tensor(out):
        return [out]
    if isinstance(out, dict):
        flat = []
        for v in out.values():
            flat.extend(_flatten(v))
        return flat
    flat = []
    for v in out:
        flat.extend(_flatten(v))
    return flat


def _scalar(out) -> torch.Tensor:
    return sum(t.float().pow(2).sum() for t in _flatten(out))


def _bn_identity_(bn: torch.nn.modules.batchnorm._BatchNorm) -> None:
    bn.weight.data.fill_(1.0)
    bn.bias.data.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0)


def _populate_bn_stats(module: torch.nn.Module, x: torch.Tensor, steps: int = 4) -> None:
    module.train()
    with torch.no_grad():
        for _ in range(steps):
            module(x)


def _scatter_expected(weight: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    spatial_dims = weight.ndim - 2
    k = weight.shape[-1]
    eff = (k - 1) * dilation + 1
    out = weight.new_zeros((weight.shape[0], weight.shape[1]) + (kernel_size,) * spatial_dims)
    start = (kernel_size - eff) // 2
    for idx in torch.cartesian_prod(*[torch.arange(k) for _ in range(spatial_dims)]):
        idx = tuple(int(i) for i in idx.tolist())
        target = tuple(start + i * dilation for i in idx)
        out[(slice(None), slice(None)) + target] = weight[(slice(None), slice(None)) + idx]
    return out


@pytest.mark.parametrize("spatial_dims,xshape", [
    (2, (2, 8, 32, 32)),
    (3, (2, 8, 8, 32, 32)),
])
def test_dilated_reparam_block_eval_deploy_allclose(spatial_dims, xshape):
    block = DilatedReparamBlock(8, 5, spatial_dims=spatial_dims)
    x = torch.randn(*xshape)
    _populate_bn_stats(block, x)
    block.eval()
    with torch.no_grad():
        y_before = block(x)
    params_before = sum(p.numel() for p in block.parameters())

    block.switch_to_deploy()

    with torch.no_grad():
        y_after = block(x)
    params_after = sum(p.numel() for p in block.parameters())

    assert torch.allclose(y_before, y_after, atol=1e-5)
    assert params_after < params_before
    assert isinstance(block.reparam, torch.nn.Module)
    assert block.reparam.bias is not None
    assert block.switch_to_deploy() is block
    with torch.no_grad():
        y_after_2 = block(x)
    assert torch.allclose(y_after, y_after_2, atol=0.0)


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_dilated_reparam_kernel_expansion_matches_scatter(spatial_dims):
    block = DilatedReparamBlock(
        4, 5,
        branch_kernel_sizes=[3],
        branch_dilations=[2],
        spatial_dims=spatial_dims)

    with torch.no_grad():
        block.lk.weight.zero_()
        _bn_identity_(block.lk_bn)
        branch = block.branches[0]
        branch[0].weight.copy_(
            torch.arange(branch[0].weight.numel(), dtype=branch[0].weight.dtype).view_as(branch[0].weight))
        _bn_identity_(branch[1])

    block.switch_to_deploy()
    weight = block.reparam.weight.detach()
    expected = _scatter_expected(branch[0].weight.detach(), 5, 2)

    assert torch.allclose(weight, expected)


def test_dilated_reparam_rejects_invalid_branch_specs():
    with pytest.raises(ValueError):
        DilatedReparamBlock(8, 4, spatial_dims=2)
    with pytest.raises(ValueError):
        DilatedReparamBlock(
            8, 5,
            branch_kernel_sizes=[5],
            branch_dilations=[2],
            spatial_dims=2)


@pytest.mark.parametrize("spatial_dims,xshape", [
    (2, (2, 8, 32, 32)),
    (3, (2, 8, 8, 32, 32)),
])
def test_mednext_block_dilated_reparam_eval_deploy_and_backward(spatial_dims, xshape):
    block = MedNeXtBlock(
        8, kernel_size=5, spatial_dims=spatial_dims, dilated_reparam=True)
    x = torch.randn(*xshape)
    _populate_bn_stats(block, x)
    block.eval()
    with torch.no_grad():
        y_before = block(x)
    block.dwconv.switch_to_deploy()
    with torch.no_grad():
        y_after = block(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)

    block.train()
    x_train = torch.randn(*xshape, requires_grad=True)
    loss = block(x_train).float().pow(2).mean()
    loss.backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert torch.isfinite(loss)
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("patch_mode,xshape", [
    ("z_axis", (1, 16, 16, 64, 64)),
    ("2_5d", (1, 16, 64, 64)),
])
def test_build_model_reparameterize_model_allclose(patch_mode, xshape):
    cfg = _make_cfg(patch_mode=patch_mode, dilated_reparam=True)
    model = build_model(cfg)
    xshape = (xshape[0], cfg.model.in_channels) + xshape[2:]
    x = torch.randn(*xshape)
    _populate_bn_stats(model, x)
    model.eval()
    with torch.no_grad():
        y_before = model(x)
    reparameterize_model(model)
    with torch.no_grad():
        y_after = model(x)

    y_before_main = y_before[0] if isinstance(y_before, list) else y_before
    y_after_main = y_after[0] if isinstance(y_after, list) else y_after
    assert y_before_main.shape == y_after_main.shape
    assert torch.allclose(y_before_main, y_after_main, atol=1e-5)


@pytest.mark.parametrize("patch_mode,xshape", [
    ("z_axis", (1, 16, 16, 64, 64)),
    ("2_5d", (1, 16, 64, 64)),
])
def test_build_model_dilated_reparam_train_backward_finite(patch_mode, xshape):
    cfg = _make_cfg(patch_mode=patch_mode, dilated_reparam=True)
    model = build_model(cfg).train()
    xshape = (xshape[0], cfg.model.in_channels) + xshape[2:]
    x = torch.randn(*xshape, requires_grad=True)
    loss = _scalar(model(x))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss)
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_config_default_and_validation_and_flag_off_structure():
    cfg = Config()
    assert cfg.model.mednext_dilated_reparam is False
    cfg.validate()

    cfg_bad = Config()
    cfg_bad.model.backbone = "mednext"
    cfg_bad.model.mednext_dilated_reparam = True
    cfg_bad.model.mednext_dilated_reparam_kernel_sizes = [5]
    cfg_bad.model.mednext_dilated_reparam_dilations = [2]
    with pytest.raises(AssertionError):
        cfg_bad.validate()

    cfg_off = _make_cfg(dilated_reparam=False)
    model = build_model(cfg_off)
    assert not any(isinstance(m, DilatedReparamBlock) for m in model.modules())
