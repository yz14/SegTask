"""S4-B/S5-B/S5-E/S5-F 新增可选模式：默认保旧、opt-in 生效。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from taskcore.config.core import Config, ConfigError
from taskcore.models.blocks import (
    AttentionGate3D,
    LKA3D,
    MSCA3D,
    SelfAttentionBlock,
    Upsample,
    make_attention,
)
from taskcore.models.factory import build_model


def _tiny_cfg(**unet_kwargs) -> Config:
    cfg = Config()
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    for k, v in unet_kwargs.items():
        setattr(cfg.model.unet, k, v)
    cfg.sync()
    cfg.validate()
    return cfg


# ---------------- S5-B: attn_gate_norm='none' ----------------

def test_attn_gate_norm_none_uses_identity():
    gate = AttentionGate3D(x_ch=8, g_ch=8, norm_type="none")
    assert isinstance(gate.W_x[1], nn.Identity)
    assert isinstance(gate.W_g[1], nn.Identity)
    x = torch.randn(2, 8, 4, 6, 6)
    g = torch.randn(2, 8, 4, 6, 6)
    out = gate(x, g)
    assert out.shape == x.shape


def test_attn_gate_norm_none_accepted_by_config():
    cfg = _tiny_cfg(skip_attention=True, attn_gate_norm="none")
    assert cfg.model.unet.attn_gate_norm == "none"


# ---------------- S5-E: lka_aniso / msca_aniso ----------------

def test_lka_aniso_z_kernel_shapes():
    iso = LKA3D(8, spatial_dims=3)
    aniso = LKA3D(8, spatial_dims=3, aniso_z=True)
    assert iso.dw.kernel_size == (5, 5, 5)
    assert aniso.dw.kernel_size == (3, 5, 5)
    assert aniso.dw_dilated.kernel_size == (3, 7, 7)
    assert aniso.dw_dilated.dilation == (1, 3, 3)
    x = torch.randn(1, 8, 6, 12, 12)
    assert aniso(x).shape == x.shape


def test_msca_aniso_forward_shape():
    aniso = MSCA3D(8, spatial_dims=3, aniso_z=True)
    x = torch.randn(1, 8, 6, 12, 12)
    assert aniso(x).shape == x.shape


def test_factory_aniso_attention_names():
    m1 = make_attention("lka_aniso", 8, spatial_dims=3)
    m2 = make_attention("msca_aniso", 8, spatial_dims=3)
    assert isinstance(m1, LKA3D) and isinstance(m2, MSCA3D)
    cfg = _tiny_cfg(attention_type="lka_aniso")
    assert cfg.model.unet.attention_type == "lka_aniso"
    with pytest.raises(ConfigError):
        _tiny_cfg(attention_type="lka_bogus")


# ---------------- S5-F: upsample_interp_dtype ----------------

def test_upsample_interp_dtype_validation():
    Upsample(8, 8, mode="trilinear", interp_dtype="native")
    with pytest.raises(ValueError):
        Upsample(8, 8, mode="trilinear", interp_dtype="fp64")
    with pytest.raises(ConfigError):
        _tiny_cfg(upsample_interp_dtype="fp64")


def test_upsample_interp_dtype_fp32_equivalent():
    torch.manual_seed(0)
    up_legacy = Upsample(4, 4, mode="trilinear", interp_dtype="legacy")
    up_native = Upsample(4, 4, mode="trilinear", interp_dtype="native")
    up_native.load_state_dict(up_legacy.state_dict())
    x = torch.randn(1, 4, 4, 6, 6)
    assert torch.equal(up_legacy(x), up_native(x))


def test_upsample_interp_dtype_native_bf16_no_fp32_roundtrip():
    torch.manual_seed(0)
    up = Upsample(4, 4, mode="trilinear", interp_dtype="native").to(
        torch.bfloat16)
    x = torch.randn(1, 4, 4, 6, 6, dtype=torch.bfloat16)
    assert up(x).dtype == torch.bfloat16


def test_upsample_interp_dtype_plumbed_from_config():
    cfg = _tiny_cfg(upsample_mode="trilinear",
                    upsample_interp_dtype="native")
    model = build_model(cfg)
    ups = [m for m in model.modules() if isinstance(m, Upsample)]
    assert ups and all(not u.interp_fp32 for u in ups)
    cfg2 = _tiny_cfg(upsample_mode="trilinear")
    model2 = build_model(cfg2)
    ups2 = [m for m in model2.modules() if isinstance(m, Upsample)]
    assert ups2 and all(u.interp_fp32 for u in ups2)


# ---------------- S4-B: init_strategy 跳过自定义初始化 ----------------

def test_init_strategy_preserves_selfattn_zero_init():
    cfg = _tiny_cfg()
    cfg.model.init_strategy = "kaiming"
    cfg.model.unet.selfattn.enabled = True
    cfg.model.unet.selfattn.encoder_stages = [0, 0, 1]
    cfg.model.unet.selfattn.zero_init = True
    cfg.sync()
    cfg.validate()
    model = build_model(cfg)
    sa = [m for m in model.modules()
          if isinstance(m, SelfAttentionBlock)]
    assert sa
    for blk in sa:
        assert torch.all(blk.proj.weight == 0)
        assert torch.all(blk.proj.bias == 0)


def test_init_strategy_preserves_icnr_pixelshuffle():
    cfg = _tiny_cfg(upsample_mode="pixelshuffle")
    cfg.model.init_strategy = "kaiming"
    cfg.sync()
    cfg.validate()
    torch.manual_seed(0)
    model = build_model(cfg)
    ups = [m for m in model.modules()
           if isinstance(m, Upsample) and m.mode == "pixelshuffle"]
    assert ups
    for u in ups:
        w = u.expand.weight
        rd = 2 ** 3
        # ICNR：每 rd 个输出滤波器同源复制。
        w2 = w.reshape(w.shape[0] // rd, rd, *w.shape[1:])
        assert torch.equal(w2[:, 0], w2[:, 1])


def test_init_strategy_still_overrides_plain_convs():
    cfg = _tiny_cfg()
    cfg.model.init_strategy = "kaiming"
    cfg.sync()
    cfg.validate()
    torch.manual_seed(0)
    m_kaiming = build_model(cfg)
    cfg2 = _tiny_cfg()
    torch.manual_seed(0)
    m_legacy = build_model(cfg2)
    diff = any(
        not torch.equal(a, b) for (_, a), (_, b) in zip(
            m_kaiming.state_dict().items(),
            m_legacy.state_dict().items()))
    assert diff
