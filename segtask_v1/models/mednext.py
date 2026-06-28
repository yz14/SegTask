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

from .blocks import _CONV, make_attention


def _channelwise_groupnorm(num_channels: int) -> nn.GroupNorm:
    """通道级 GroupNorm（num_groups == num_channels）：MedNeXt 原作选型，
    等价逐通道按空间统计，小 batch 比 LayerNorm/BatchNorm 更稳。"""
    return nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)


class MedNeXtBlock(nn.Module):
    """MedNeXt 残差倒瓶颈块（stride=1, in==out）。

    dwconv(k) → 通道级 GroupNorm → pwconv↑(×R) → GELU → pwconv↓ → attn? → +residual。
    """

    def __init__(
        self,
        dim           : int,
        expand_ratio  : int = 4,
        kernel_size   : int = 3,
        attention_type: str = "none",
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
        self.pwconv2 = _CONV[d](hidden, dim, kernel_size=1, bias=True)
        self.attn    = make_attention(attention_type, dim, spatial_dims=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.attn(out)
        return res + out


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
        attention_type: str = "none",
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
            attention_type=attention_type, spatial_dims=d)

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
        attention_type: str = "none",
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        blocks = [MedNeXtAdaptBlock(
            in_ch, out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
            attention_type=attention_type, spatial_dims=d)]
        for _ in range(1, num_blocks):
            blocks.append(MedNeXtBlock(
                out_ch, expand_ratio=expand_ratio, kernel_size=kernel_size,
                attention_type=attention_type, spatial_dims=d))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
