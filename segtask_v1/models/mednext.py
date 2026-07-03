"""MedNeXt blocks for 3D/2D UNet (Roy et al., MICCAI 2023, dim-agnostic 2D/3D).

档位 A（本文件）：实现 MedNeXt 的核心**残差倒瓶颈块**，复用框架既有的
``Downsample`` / ``Upsample`` 做重采样（``downsample_mode`` / ``upsample_mode`` 仍生效，
且与 ``anisotropic_pooling`` 兼容）。MedNeXt 原生的「重采样残差块（Up/Down block 把 stride
融入深度卷积 + 1×1 残差）」与 UpKern 大核权重迁移为后续档位 B。

Block（C 通道输入，参照论文 §2.1，3 层 mirror Transformer）:
  1. Depthwise Conv k³（groups=C）→ 通道级 GroupNorm（num_groups=C；小 batch 稳定，
     替代原 ConvNeXt 的 LayerNorm）。
  2. Expansion: 1×1 Conv（C → C·R）→ GELU。
  3. Compression: 1×1 Conv（C·R → C）。
  + 残差（in==out, stride=1）。
与 ConvNeXt 的差异：GroupNorm（非 LN）、核 3/5（非 7）、扩张比 R 可配（非固定 4）、无 LayerScale。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DropPath, GlobalResponseNorm, _CONV, make_attention


def _channelwise_groupnorm(num_channels: int) -> nn.GroupNorm:
    """通道级 GroupNorm（num_groups == num_channels）：MedNeXt 原作选型，
    等价逐通道按空间统计，小 batch 比 LayerNorm/BatchNorm 更稳。"""
    return nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)


def upkern_remap_state_dict(src_sd: dict, target_model: nn.Module) -> dict:
    """把小核 MedNeXt checkpoint 的深度卷积权重插值到目标大核。

    仅处理与目标参数同名、同 rank、同通道形状的 depthwise-conv-like 权重：
    ``(C, 1, k, k[, k])``。当仅空间核尺寸不一致时，按空间维做
    ``bilinear``/``trilinear`` 插值并保留其余张量不变；不在目标模型中的键
    直接丢弃，无法对齐的张量也保持目标模型初始化值。

    Parameters
    ----------
    src_sd:
        源 checkpoint 的 state_dict。
    target_model:
        目标 MedNeXt 模型，用于提供目标形状。
    """
    target_sd = target_model.state_dict()
    remapped = {}
    for key, src_tensor in src_sd.items():
        tgt_tensor = target_sd.get(key)
        if tgt_tensor is None or not torch.is_tensor(src_tensor):
            continue
        if not torch.is_tensor(tgt_tensor):
            continue
        if src_tensor.shape == tgt_tensor.shape:
            remapped[key] = src_tensor
            continue
        if (src_tensor.ndim not in (4, 5)
                or tgt_tensor.ndim != src_tensor.ndim
                or src_tensor.shape[:2] != tgt_tensor.shape[:2]):
            continue
        if src_tensor.shape[2:] == tgt_tensor.shape[2:]:
            remapped[key] = src_tensor
            continue
        mode = "bilinear" if src_tensor.ndim == 4 else "trilinear"
        spatial = tuple(int(s) for s in tgt_tensor.shape[2:])
        work = src_tensor.detach().to(dtype=torch.float32)
        work = work.reshape(work.shape[0] * work.shape[1], 1, *work.shape[2:])
        work = F.interpolate(work, size=spatial, mode=mode, align_corners=True)
        work = work.reshape(*tgt_tensor.shape).to(dtype=src_tensor.dtype)
        remapped[key] = work
    return remapped


class MedNeXtBlock(nn.Module):
    """MedNeXt 残差倒瓶颈块（stride=1, in==out）。

    dwconv(k) → 通道级 GroupNorm → pwconv↑(×R) → GELU → pwconv↓ → attn? → +residual。
    """

    def __init__(
        self,
        dim           : int,
        expand_ratio  : int = 4,
        kernel_size   : int = 3,
        drop_path     : float = 0.0,
        attention_type: str = "none",
        use_grn       : bool = False,
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        self.spatial_dims = d
        hidden  = int(dim * expand_ratio)
        padding = kernel_size // 2

        self.dwconv  = _CONV[d](
            dim, dim, kernel_size=kernel_size, padding=padding,
            groups=dim, bias=True)
        self.norm    = _channelwise_groupnorm(dim)
        self.pwconv1 = _CONV[d](dim, hidden, kernel_size=1, bias=True)
        self.act     = nn.GELU()
        self.grn     = GlobalResponseNorm(hidden, spatial_dims=d) if use_grn else nn.Identity()
        self.pwconv2 = _CONV[d](hidden, dim, kernel_size=1, bias=True)
        self.attn    = make_attention(attention_type, dim, spatial_dims=d)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.grn(out)
        out = self.pwconv2(out)
        out = self.attn(out)
        return res + self.drop_path(out)


class MedNeXtAdaptBlock(nn.Module):
    """通道适配版：in_ch != out_ch 时先 1×1 投影（+GroupNorm）再走标准 MedNeXt 块。

    本框架在「stage 首个 block」处升通道（stage 间下采样保持通道），故 stage 起始块需此适配
    （与 ConvNeXtAdaptBlock 同构）。投影后残差在 out_ch 维度内闭合。
    """

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        expand_ratio  : int = 4,
        kernel_size   : int = 3,
        drop_path     : float = 0.0,
        attention_type: str = "none",
        use_grn       : bool = False,
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        self.proj = (
            nn.Sequential(
                _CONV[d](in_ch, out_ch, 1, bias=False),
                _channelwise_groupnorm(out_ch))
            if in_ch != out_ch else nn.Identity())
        self.block = MedNeXtBlock(
            out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
            drop_path=drop_path, attention_type=attention_type,
            use_grn=use_grn, spatial_dims=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))


class MedNeXtStage(nn.Module):
    """单分辨率 N 个 MedNeXt 块（首块可改通道）。接口与 ConvNeXtStage/ResNetStage 一致。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        num_blocks    : int = 2,
        expand_ratio  : int = 4,
        kernel_size   : int = 3,
        drop_path_rates: list = None,
        attention_type: str = "none",
        use_grn       : bool = False,
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        blocks = [MedNeXtAdaptBlock(
            in_ch, out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
            drop_path=drop_path_rates[0], attention_type=attention_type,
            use_grn=use_grn, spatial_dims=d)]
        for i in range(1, num_blocks):
            dp = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            blocks.append(MedNeXtBlock(
                out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
                drop_path=dp, attention_type=attention_type,
                use_grn=use_grn, spatial_dims=d))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


__all__ = [
    "MedNeXtBlock",
    "MedNeXtAdaptBlock",
    "MedNeXtStage",
    "upkern_remap_state_dict",
]
