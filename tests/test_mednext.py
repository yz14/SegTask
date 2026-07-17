"""Tests for the MedNeXt backbone (档位 A: 残差倒瓶颈块 + 复用通用重采样).

覆盖：
- MedNeXtBlock：形状保持、残差、通道级 GroupNorm（num_groups==C）、深度卷积 groups==C、
  可配核大小（3/5）与扩张比 R、2D/3D、梯度流。
- MedNeXtAdaptBlock / MedNeXtStage：首块改通道、in==out 时投影为 Identity。
- build_model(backbone='mednext')：z_axis(3D) / 2_5d(2D) × decoder {unet,unetpp,unet3p}
  端到端前向尺寸正确、训练前向+反向梯度有限；与 anisotropic_pooling、attention_type 兼容。
- Config.validate：mednext_kernel_size / mednext_expand_ratio 白名单。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch
import torch.nn as nn

from taskcore.config.core import Config
from taskcore.models.factory import build_model
from taskcore.models.mednext import (
    MedNeXtAdaptBlock,
    MedNeXtBlock,
    MedNeXtStage,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_cfg(
    decoder_type: str = "unet",
    patch_mode: str = "z_axis",
    deep_supervision: bool = True,
    expand_ratio: int = 4,
    kernel_size: int = 3,
    attention_type: str = "none",
    anisotropic_pooling: bool = False,
) -> Config:
    cfg = Config()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.model.backbone = "mednext"
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.decoder_type = decoder_type
    cfg.model.norm_type = "group"       # stem/skip 投影；mednext 块内固定通道级 GN
    cfg.model.norm_groups = 8
    cfg.model.activation = "gelu"
    cfg.model.deep_supervision = deep_supervision
    cfg.model.attention_type = attention_type
    cfg.model.anisotropic_pooling = anisotropic_pooling
    cfg.model.mednext_expand_ratio = expand_ratio
    cfg.model.mednext_kernel_size = kernel_size
    if decoder_type == "unet3p":
        cfg.model.unet3p_cat_channels = 16
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


# ---------------------------------------------------------------------------
# block-level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_mednext_block_shape_grad_and_norm(spatial_dims):
    dim = 16
    blk = MedNeXtBlock(dim, expand_ratio=4, kernel_size=3, spatial_dims=spatial_dims)
    # 通道级 GroupNorm：每通道一组。
    assert isinstance(blk.norm, nn.GroupNorm)
    assert blk.norm.num_groups == dim and blk.norm.num_channels == dim
    # 深度卷积：groups==C、核 3。
    assert blk.dwconv.groups == dim
    assert tuple(blk.dwconv.kernel_size) == (3,) * spatial_dims
    # 扩张：pwconv1 输出 C*R。
    assert blk.pwconv1.out_channels == dim * 4

    shape = (2, dim) + (8,) * spatial_dims
    x = torch.randn(*shape, requires_grad=True)
    y = blk(x)
    assert y.shape == x.shape          # 残差块形状不变
    y.float().pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mednext_block_kernel5_padding():
    blk = MedNeXtBlock(8, kernel_size=5, spatial_dims=3)
    assert tuple(blk.dwconv.kernel_size) == (5, 5, 5)
    assert tuple(blk.dwconv.padding) == (2, 2, 2)
    x = torch.randn(1, 8, 8, 8, 8)
    assert blk(x).shape == x.shape     # 'same' 卷积，尺寸不变


def test_mednext_expand_ratio_propagates():
    blk = MedNeXtBlock(8, expand_ratio=3, kernel_size=3, spatial_dims=3)
    assert blk.pwconv1.out_channels == 24
    assert blk.pwconv2.in_channels == 24 and blk.pwconv2.out_channels == 8


def test_mednext_stage_channel_change_and_identity_proj():
    stage = MedNeXtStage(8, 16, num_blocks=2, spatial_dims=3)
    x = torch.randn(1, 8, 8, 8, 8)
    y = stage(x)
    assert y.shape == (1, 16, 8, 8, 8)
    # in==out 时 AdaptBlock 投影为 Identity。
    same = MedNeXtAdaptBlock(16, 16, spatial_dims=3)
    assert isinstance(same.proj, nn.Identity)
    diff = MedNeXtAdaptBlock(8, 16, spatial_dims=3)
    assert not isinstance(diff.proj, nn.Identity)


# ---------------------------------------------------------------------------
# end-to-end build_model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoder_type", ["unet", "unetpp", "unet3p"])
def test_mednext_unet_end_to_end_3d(decoder_type):
    cfg = _make_cfg(decoder_type=decoder_type, patch_mode="z_axis")
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    main = y[0] if isinstance(y, list) else y
    assert main.shape[2:] == (16, 64, 64)   # 全分辨率还原


def test_mednext_2_5d_end_to_end():
    cfg = _make_cfg(decoder_type="unet", patch_mode="2_5d")
    model = build_model(cfg).eval()
    # 2.5D 折叠：输入 (B, D, H, W)，D=patch_size[0]=in_channels。
    x = torch.randn(2, cfg.model.in_channels, 64, 64)
    with torch.no_grad():
        y = model(x)
    main = y[0] if isinstance(y, list) else y
    assert main.shape[2:] == (64, 64)


def test_mednext_train_backward_finite():
    cfg = _make_cfg(decoder_type="unet", patch_mode="z_axis", kernel_size=5)
    model = build_model(cfg).train()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    _scalar(model(x)).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_mednext_anisotropic_pooling_compatible():
    # MedNeXt 复用通用 Downsample → 与 anisotropic_pooling 兼容（区别于 ConvNeXt LN-first）。
    cfg = _make_cfg(decoder_type="unet", patch_mode="z_axis", anisotropic_pooling=True)
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    main = y[0] if isinstance(y, list) else y
    assert main.shape[2:] == (16, 64, 64)


def test_mednext_with_attention():
    cfg = _make_cfg(decoder_type="unet", patch_mode="z_axis", attention_type="se")
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        y = model(x)
    main = y[0] if isinstance(y, list) else y
    assert main.shape[2:] == (16, 64, 64)


def test_mednext_grad_checkpointing_combo():
    # MedNeXt + 梯度检查点：训练前向+反向梯度有限（两特性正交可叠加）。
    cfg = _make_cfg(decoder_type="unet", patch_mode="z_axis")
    cfg.model.grad_checkpointing = True
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).train()
    assert model.encoder.grad_checkpointing is True
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    _scalar(model(x)).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def test_config_accepts_mednext():
    cfg = Config()
    cfg.model.backbone = "mednext"
    cfg.model.mednext_kernel_size = 5
    cfg.model.mednext_expand_ratio = 3
    cfg.sync()
    cfg.validate()  # 不应抛错


def test_config_rejects_bad_mednext_kernel_size():
    cfg = Config()
    cfg.model.backbone = "mednext"
    cfg.model.mednext_kernel_size = 4   # 仅 3/5/7 合法
    with pytest.raises(AssertionError):
        cfg.validate()


def test_config_rejects_bad_mednext_expand_ratio():
    cfg = Config()
    cfg.model.backbone = "mednext"
    cfg.model.mednext_expand_ratio = 0
    with pytest.raises(AssertionError):
        cfg.validate()


def test_config_rejects_unknown_backbone():
    cfg = Config()
    cfg.model.backbone = "not_a_backbone"
    with pytest.raises(AssertionError):
        cfg.validate()


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
