"""Tests for MedNeXt UpKern remapping.

覆盖：
- k=3 → k=5 的 depthwise 卷积权重插值（2D/3D）；
- 常量核插值不变；
- build_model(backbone='mednext') 端到端预训练权重迁移；
- Config 默认值与校验。
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
import torch.nn as nn

from taskcore.config.core import Config
from taskcore.models.factory import build_model
from taskcore.models.mednext import MedNeXtStage, upkern_remap_state_dict
from taskcore.engine.checkpoint import (
    extract_model_state_dict,
    strip_common_prefixes,
)


def _make_cfg(kernel_size: int, patch_mode: str) -> Config:
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
    cfg.model.deep_supervision = False
    cfg.model.mednext_expand_ratio = 4
    cfg.model.mednext_kernel_size = kernel_size
    cfg.train.pretrain_upkern = True
    cfg.sync()
    cfg.validate()
    return cfg


def _single_tensor_name(model: torch.nn.Module) -> str:
    for name, tensor in model.state_dict().items():
        if name.endswith("dwconv.weight") and tensor.ndim in (4, 5):
            return name
    raise AssertionError("No MedNeXt depthwise weight found.")


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_upkern_remap_resizes_depthwise_weights(spatial_dims):
    src = MedNeXtStage(8, 8, num_blocks=1, kernel_size=3,
                       spatial_dims=spatial_dims)
    tgt = MedNeXtStage(8, 8, num_blocks=1, kernel_size=5,
                       spatial_dims=spatial_dims)

    src_sd = src.state_dict()
    remapped = upkern_remap_state_dict(src_sd, tgt)
    key = _single_tensor_name(tgt)

    assert key in remapped
    assert remapped[key].shape == tgt.state_dict()[key].shape
    result = tgt.load_state_dict(remapped, strict=False)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_upkern_constant_kernel_stays_constant(spatial_dims):
    src = MedNeXtStage(8, 8, num_blocks=1, kernel_size=3,
                       spatial_dims=spatial_dims)
    tgt = MedNeXtStage(8, 8, num_blocks=1, kernel_size=5,
                       spatial_dims=spatial_dims)

    key = _single_tensor_name(src)
    dict(src.named_parameters())[key].data.fill_(2.75)
    remapped = upkern_remap_state_dict(src.state_dict(), tgt)
    weight = remapped[key]

    assert torch.allclose(weight, torch.full_like(weight, 2.75))


def test_upkern_plain_checkpoint_remaps_into_reparam_target(caplog):
    src = MedNeXtStage(8, 8, num_blocks=1, kernel_size=3, spatial_dims=3)
    tgt = MedNeXtStage(
        8, 8, num_blocks=1, kernel_size=5, spatial_dims=3,
        dilated_reparam=True)

    key = next(k for k in tgt.state_dict() if k.endswith("dwconv.lk.weight"))
    initial = tgt.state_dict()[key].clone()

    caplog.set_level("WARNING")
    remapped = upkern_remap_state_dict(src.state_dict(), tgt)

    assert key in remapped
    assert remapped[key].shape == tgt.state_dict()[key].shape
    assert not torch.allclose(remapped[key], initial)
    assert any(
        "plain checkpoint -> reparameterized target" in rec.message
        for rec in caplog.records)
    assert any("target-init keys stay random" in rec.message
               for rec in caplog.records)


def test_upkern_skips_non_depthwise_resize_and_keeps_target_init(caplog):
    src = nn.Sequential(nn.Conv3d(4, 4, kernel_size=3, padding=1, bias=False))
    tgt = nn.Sequential(nn.Conv3d(4, 4, kernel_size=5, padding=2, bias=False))

    caplog.set_level("WARNING")
    remapped = upkern_remap_state_dict(src.state_dict(), tgt)

    assert "0.weight" not in remapped
    assert any("skipping non-depthwise tensor" in rec.message
               for rec in caplog.records)


@pytest.mark.parametrize("patch_mode,spatial_dims,input_shape", [
    ("z_axis", 3, None),
    ("2_5d", 2, None),
])
def test_upkern_build_model_roundtrip(patch_mode, spatial_dims, input_shape,
                                     tmp_path):
    src_cfg = _make_cfg(kernel_size=3, patch_mode=patch_mode)
    tgt_cfg = _make_cfg(kernel_size=5, patch_mode=patch_mode)
    if patch_mode == "z_axis":
        input_shape = (1, src_cfg.model.in_channels, 16, 64, 64)
    else:
        input_shape = (1, src_cfg.model.in_channels, 64, 64)

    src_model = build_model(src_cfg).eval()
    tgt_model = build_model(tgt_cfg).train()

    ckpt_path = tmp_path / "pretrain.pth"
    torch.save({"model_state_dict": src_model.state_dict()}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, _ = extract_model_state_dict(ckpt, prefer_ema=False)
    sd = strip_common_prefixes(sd)
    remapped = upkern_remap_state_dict(sd, tgt_model)
    result = tgt_model.load_state_dict(remapped, strict=False)

    assert result.missing_keys == []
    assert result.unexpected_keys == []

    x = torch.randn(*input_shape)
    y = tgt_model(x)
    main = y[0] if isinstance(y, list) else y
    loss = main.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in tgt_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_config_default_pretrain_upkern_and_validate():
    cfg = Config()
    assert cfg.train.pretrain_upkern is False
    cfg.validate()
