"""A5 块级升级回归测试：drop-path / GRN / AttentionGate norm。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from segtask_v1.config import Config
from segtask_v1.models.blocks import (
    AttentionGate3D,
    DropPath,
    GlobalResponseNorm,
)
from segtask_v1.models.convnext import ConvNeXtBlock
from segtask_v1.models.factory import build_model
from segtask_v1.models.mednext import MedNeXtBlock
from segtask_v1.models.resnet import (
    BottleneckBlock,
    MultiRFBlock,
    PreActResNetBlock,
    R2Plus1DBlock,
    ResNetBlock,
)


def _build_cfg(backbone: str) -> Config:
    cfg = Config()
    cfg.data.patch_mode = "z_axis"
    cfg.data.patch_size = [8, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.model.backbone = backbone
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 2
    cfg.model.drop_path_rate = 0.3
    cfg.model.deep_supervision = False
    cfg.model.attention_type = "none"
    if backbone == "mednext":
        cfg.model.norm_type = "group"
        cfg.model.activation = "gelu"
    cfg.sync()
    cfg.validate()
    return cfg


def _exercise_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = block(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    return y


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (ResNetBlock, dict(in_ch=8, out_ch=8, spatial_dims=3)),
        (MultiRFBlock, dict(
            in_ch=8, out_ch=8, dilations=[1, 2], mode="split",
            fusion="concat_proj", spatial_dims=3)),
    ],
)
def test_resnet_drop_path_identity_and_forced_drop(cls, kwargs, monkeypatch):
    block_ref = cls(drop_path=0.0, **kwargs).train()
    block_dp = cls(drop_path=0.5, **kwargs).train()
    block_dp.load_state_dict(block_ref.state_dict(), strict=False)

    assert isinstance(block_ref.drop_path, nn.Identity)
    assert not isinstance(block_dp.drop_path, nn.Identity)

    calls = []

    def fake_forward(self, x):
        calls.append(x.shape)
        return torch.zeros_like(x)

    monkeypatch.setattr(DropPath, "forward", fake_forward)

    x_ref = torch.randn(2, 8, 4, 8, 8, requires_grad=True)
    y_ref = block_ref(x_ref)

    x_dp = x_ref.detach().clone().requires_grad_(True)
    y_dp = _exercise_block(block_dp, x_dp)

    assert calls, "expected DropPath.forward to be invoked"
    assert y_ref.shape == y_dp.shape == x_ref.shape
    assert not torch.allclose(y_ref, y_dp)


def test_mednext_drop_path_identity_and_forced_drop(monkeypatch):
    block_ref = MedNeXtBlock(8, drop_path=0.0, spatial_dims=3).train()
    block_dp = MedNeXtBlock(8, drop_path=0.5, spatial_dims=3).train()
    block_dp.load_state_dict(block_ref.state_dict(), strict=False)

    assert isinstance(block_ref.drop_path, nn.Identity)
    assert not isinstance(block_dp.drop_path, nn.Identity)

    calls = []

    def fake_forward(self, x):
        calls.append(x.shape)
        return torch.zeros_like(x)

    monkeypatch.setattr(DropPath, "forward", fake_forward)

    x_ref = torch.randn(2, 8, 4, 8, 8, requires_grad=True)
    y_ref = block_ref(x_ref)

    x_dp = x_ref.detach().clone().requires_grad_(True)
    y_dp = _exercise_block(block_dp, x_dp)

    assert calls, "expected DropPath.forward to be invoked"
    assert y_ref.shape == y_dp.shape == x_ref.shape
    assert not torch.allclose(y_ref, y_dp)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (ResNetBlock, dict(in_ch=8, out_ch=16, spatial_dims=3)),
        (PreActResNetBlock, dict(in_ch=8, out_ch=16, spatial_dims=3)),
        (BottleneckBlock, dict(in_ch=8, out_ch=16, spatial_dims=3)),
        (R2Plus1DBlock, dict(in_ch=8, out_ch=16, spatial_dims=3)),
        (MultiRFBlock, dict(
            in_ch=8, out_ch=16, dilations=[1, 2], mode="split",
            fusion="concat_proj", spatial_dims=3)),
    ],
)
def test_projection_blocks_keep_drop_path_enabled(cls, kwargs):
    block = cls(drop_path=0.25, **kwargs)
    assert not isinstance(block.drop_path, nn.Identity)


@pytest.mark.parametrize("backbone", ["resnet", "mednext"])
def test_factory_threads_drop_path_schedule(backbone):
    cfg = _build_cfg(backbone)
    model = build_model(cfg).train()
    x = torch.randn(1, cfg.model.in_channels, 8, 32, 32, requires_grad=True)
    y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    assert out.shape[-3:] == (8, 32, 32)
    out.mean().backward()
    assert x.grad is not None

    drop_paths = [m.drop_path for m in model.modules() if hasattr(m, "drop_path")]
    probs = [dp.drop_prob for dp in drop_paths if hasattr(dp, "drop_prob")]
    assert drop_paths, "expected at least one drop_path attribute"
    assert any(isinstance(dp, nn.Identity) for dp in drop_paths)
    assert probs, "expected at least one stochastic-depth module"
    assert max(probs) == pytest.approx(cfg.model.drop_path_rate)
    assert any(p > 0 for p in probs)


@pytest.mark.parametrize("spatial_dims,shape", [(2, (8, 8)), (3, (4, 8, 8))])
def test_grn_identity_and_changes_output(spatial_dims, shape):
    grn = GlobalResponseNorm(16, spatial_dims=spatial_dims)
    x = torch.randn(2, 16, *shape, requires_grad=True)
    y = grn(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    y.sum().backward()
    assert x.grad is not None

    with torch.no_grad():
        grn.gamma.fill_(0.25)
        grn.beta.fill_(0.5)
    y2 = grn(x.detach())
    assert y2.shape == x.shape
    assert not torch.allclose(y2, x.detach())


@pytest.mark.parametrize(
    "cls, spatial_dims",
    [
        (ConvNeXtBlock, 2),
        (ConvNeXtBlock, 3),
        (MedNeXtBlock, 2),
        (MedNeXtBlock, 3),
    ],
)
def test_grn_is_wired_into_blocks(cls, spatial_dims):
    block = cls(16, drop_path=0.0, spatial_dims=spatial_dims, use_grn=True)
    assert isinstance(block.grn, GlobalResponseNorm)


def test_attention_gate_norm_type_default_and_configurable():
    gate_default = AttentionGate3D(16, 8, spatial_dims=3)
    assert isinstance(gate_default.W_x[1], nn.BatchNorm3d)
    assert isinstance(gate_default.W_g[1], nn.BatchNorm3d)
    assert isinstance(gate_default.psi[1], nn.BatchNorm3d)

    gate = AttentionGate3D(
        16, 8, norm_type="group", norm_groups=4, spatial_dims=3)
    assert isinstance(gate.W_x[1], nn.GroupNorm)
    assert isinstance(gate.W_g[1], nn.GroupNorm)
    assert isinstance(gate.psi[1], nn.GroupNorm)

    x = torch.randn(2, 16, 4, 8, 8, requires_grad=True)
    g = torch.randn(2, 8, 4, 8, 8)
    out = gate(x, g)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize(
    "attn_gate_norm, expected",
    [
        ("batch", nn.BatchNorm3d),
        ("instance", nn.InstanceNorm3d),
        ("group", nn.GroupNorm),
    ],
)
def test_factory_threads_attention_gate_norm(attn_gate_norm, expected):
    cfg = _build_cfg("resnet")
    cfg.model.skip_attention = True
    cfg.model.attn_gate_norm = attn_gate_norm
    if attn_gate_norm == "group":
        cfg.model.norm_groups = 4
    cfg.sync()
    cfg.validate()

    model = build_model(cfg).eval()
    gates = [m for m in model.modules() if isinstance(m, AttentionGate3D)]
    assert gates, "expected at least one AttentionGate3D"
    for gate in gates:
        assert isinstance(gate.W_x[1], expected)
        assert isinstance(gate.W_g[1], expected)
        assert isinstance(gate.psi[1], expected)
