"""Regression tests for gradient checkpointing (model.grad_checkpointing).

Contract验证：
- 默认关闭（cfg.model.grad_checkpointing 默认 False；模块 attribute 同步关闭）。
- 训练前向 + 反向：开/关在数值上**严格一致**（主输出 + 全部参数梯度 allclose），
  跨 decoder_type {unet, unetpp, unet3p} × backbone {resnet, convnext}（drop_path=0）。
- eval / no_grad（验证）路径：开/关 bit-identical（checkpoint 不触发，零开销直通）。
- 2.5D（spatial_dims=2）+ ConvNeXt drop_path>0：开启 checkpoint 端到端可反传、梯度有限，
  覆盖 DropPath 在反向重算时的 RNG 复现路径（preserve_rng_state）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python tests/test_grad_checkpointing.py` 直接运行（repo root 入 sys.path）。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch

from segtask_v1.config import Config
from segtask_v1.models.factory import build_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_cfg(
    decoder_type: str = "unet",
    patch_mode: str = "z_axis",
    backbone: str = "resnet",
    deep_supervision: bool = True,
    drop_path: float = 0.0,
    grad_checkpointing: bool = False,
) -> Config:
    cfg = Config()
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [16, 64, 64]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.model.backbone = backbone
    cfg.model.encoder_channels = [16, 32, 64]
    cfg.model.blocks_per_level = 1
    cfg.model.decoder_type = decoder_type
    cfg.model.norm_type = "group"
    cfg.model.norm_groups = 8
    cfg.model.deep_supervision = deep_supervision
    cfg.model.drop_path_rate = drop_path
    cfg.model.grad_checkpointing = grad_checkpointing
    if decoder_type == "unet3p":
        cfg.model.unet3p_cat_channels = 16
    cfg.sync()
    cfg.validate()
    return cfg


def _build_pair(decoder_type: str, backbone: str):
    """同权重的两份模型：grad_checkpointing 关 / 开。"""
    cfg_off = _make_cfg(decoder_type, "z_axis", backbone, grad_checkpointing=False)
    cfg_on = _make_cfg(decoder_type, "z_axis", backbone, grad_checkpointing=True)
    torch.manual_seed(0)
    m_off = build_model(cfg_off).train()
    torch.manual_seed(0)
    m_on = build_model(cfg_on).train()
    m_on.load_state_dict(m_off.state_dict())  # 逐参数一致
    return cfg_off, m_off, m_on


def _flatten(out):
    """把 Tensor / list / dict 的模型输出摊平为 tensor 列表（顺序确定）。"""
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
# 默认关闭 + 正确 plumbing
# ---------------------------------------------------------------------------
def test_grad_checkpointing_defaults_off():
    assert Config().model.grad_checkpointing is False
    cfg = _make_cfg(grad_checkpointing=False)
    model = build_model(cfg)
    assert model.encoder.grad_checkpointing is False
    assert model.decoder.grad_checkpointing is False


@pytest.mark.parametrize("decoder_type", ["unet", "unetpp", "unet3p"])
def test_flag_plumbed_into_decoder(decoder_type):
    cfg = _make_cfg(decoder_type=decoder_type, grad_checkpointing=True)
    model = build_model(cfg)
    assert model.encoder.grad_checkpointing is True
    assert model.decoder.grad_checkpointing is True


# ---------------------------------------------------------------------------
# 训练前向 + 梯度：开/关严格一致
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone", ["resnet", "convnext"])
@pytest.mark.parametrize("decoder_type", ["unet", "unetpp", "unet3p"])
def test_grad_checkpoint_matches_baseline(decoder_type, backbone):
    cfg, m_off, m_on = _build_pair(decoder_type, backbone)
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)

    out_off = m_off(x)
    out_on = m_on(x)

    # 前向输出（含深监督各尺度）逐一一致
    flat_off, flat_on = _flatten(out_off), _flatten(out_on)
    assert len(flat_off) == len(flat_on) and len(flat_off) >= 1
    for a, b in zip(flat_off, flat_on):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)

    # 反向：全部参数梯度一致
    _scalar(out_off).backward()
    _scalar(out_on).backward()
    params_off = dict(m_off.named_parameters())
    params_on = dict(m_on.named_parameters())
    assert params_off.keys() == params_on.keys()
    n_checked = 0
    for name, p_off in params_off.items():
        p_on = params_on[name]
        if p_off.grad is None:
            assert p_on.grad is None, name
            continue
        assert p_on.grad is not None, name
        assert torch.allclose(p_off.grad, p_on.grad, atol=1e-4, rtol=1e-3), name
        n_checked += 1
    assert n_checked > 0


# ---------------------------------------------------------------------------
# eval / no_grad：开/关 bit-identical（checkpoint 不触发）
# ---------------------------------------------------------------------------
def test_grad_checkpoint_eval_identical():
    cfg, m_off, m_on = _build_pair("unet", "resnet")
    m_off.eval()
    m_on.eval()
    x = torch.randn(1, cfg.model.in_channels, 16, 64, 64)
    with torch.no_grad():
        a = m_off(x)
        b = m_on(x)
    a = a[0] if isinstance(a, list) else a
    b = b[0] if isinstance(b, list) else b
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 2.5D + ConvNeXt drop_path>0：开启 checkpoint 可反传、梯度有限
# ---------------------------------------------------------------------------
def test_grad_checkpoint_2_5d_droppath_runs():
    cfg = _make_cfg(
        decoder_type="unet",
        patch_mode="2_5d",
        backbone="convnext",
        deep_supervision=True,
        drop_path=0.2,
        grad_checkpointing=True,
    )
    torch.manual_seed(0)
    model = build_model(cfg).train()
    assert model.encoder.grad_checkpointing is True

    # 2.5D 折叠：输入 (B, D, H, W)，D=patch_size[0] 折入通道。
    x = torch.randn(2, cfg.model.in_channels, 64, 64)
    out = model(x)
    _scalar(out).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
