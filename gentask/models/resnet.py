"""UNet stage 用 ResNet 块。block_type 示例：'basic' 轻量后置激活 、'r2plus1d' (1,3,3)+(3,1,1) 仅 3D。还有 'preact'/'bottleneck'。下采样由 blocks.Downsample 外部完成。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import (
    _CONV, _DROP, ConvNormAct, get_activation, get_norm, make_attention)


class ResNetBlock(nn.Module):
    """后置激活 ResNet 块（可选 attention）。use_se=True 且 attention_type=='none' 时提升为 'se'。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        use_se        : bool = False,
        se_reduction  : int = 16,
        attention_type: str = "none",
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        self.conv1 = _CONV[d](in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act1  = get_activation(activation)

        self.conv2 = _CONV[d](out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act2  = get_activation(activation)

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        if attention_type == "none" and use_se:
            attention_type = "se"  # 旧 use_se 向后兼容
        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)

        self.shortcut = (
            nn.Sequential(_CONV[d](in_ch, out_ch, 1, bias=False),
                          get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.attn(out)
        return self.act2(out + res)


class PreActResNetBlock(nn.Module):
    """预激活 ResNet 块 (He 2016)：norm-act-conv × 2 + 残差；适合深 encoder。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        use_se        : bool = False,
        se_reduction  : int = 16,
        attention_type: str = "none",
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        self.norm1 = get_norm(norm_type, in_ch, norm_groups, spatial_dims=d)
        self.act1  = get_activation(activation)
        self.conv1 = _CONV[d](in_ch, out_ch, 3, padding=1, bias=False)

        self.norm2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act2  = get_activation(activation)
        self.conv2 = _CONV[d](out_ch, out_ch, 3, padding=1, bias=False)

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        if attention_type == "none" and use_se:
            attention_type = "se"
        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)

        # shortcut 作用于原 x；通道不匹配时用 1×1 投影（标准 pre-act）。
        self.shortcut = (
            _CONV[d](in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.drop(out)
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.attn(out)
        return out + res


class BottleneckBlock(nn.Module):
    """ResNet-50 风 bottleneck：1×1 压 → 3×3 → 1×1 扩（expansion=4，适 ResEnc-XL）。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        expansion     : int = 4,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        use_se        : bool = False,
        se_reduction  : int = 16,
        attention_type: str = "none",
        spatial_dims  : int = 3):
        super().__init__()
        d = spatial_dims
        mid = max(out_ch // expansion, 1)  # 压缩

        self.conv1 = _CONV[d](in_ch, mid, 1, bias=False)
        self.norm1 = get_norm(norm_type, mid, norm_groups, spatial_dims=d)
        self.act1  = get_activation(activation)

        self.conv2 = _CONV[d](mid, mid, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, mid, norm_groups, spatial_dims=d)
        self.act2  = get_activation(activation)

        self.conv3 = _CONV[d](mid, out_ch, 1, bias=False)
        self.norm3 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act3  = get_activation(activation)

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        if attention_type == "none" and use_se:
            attention_type = "se"
        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)

        self.shortcut = (
            nn.Sequential(_CONV[d](in_ch, out_ch, 1, bias=False),
                          get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.drop(out)
        out = self.norm3(self.conv3(out))
        out = self.attn(out)
        return self.act3(out + res)


class R2Plus1DBlock(nn.Module):
    """R(2+1)D 残差块 (Tran 2018)，仅 3D。每个 3×3×3 拆为 (1,3,3) 空间 + norm + act + (3,1,1) 时间；中间非线性不可省，mid_ch=out_ch。"""

    def __init__(
        self,
        in_ch          : int,
        out_ch         : int,
        norm_type      : str = "instance",
        norm_groups    : int = 8,
        activation     : str = "leakyrelu",
        dropout        : float = 0.0,
        use_se         : bool = False,
        se_reduction   : int = 16,
        attention_type : str = "none",
        spatial_dims   : int = 3,
        temporal_kernel: int = 3):
        super().__init__()
        if spatial_dims != 3:
            # D 必须是真空间轴；2.5D 中 D 被折叠到通道。
            raise ValueError(
                "R2Plus1DBlock requires spatial_dims=3 (D must be a real "
                "spatial axis). For 2.5D mode (spatial_dims=2), use "
                "block_type='basic'/'preact'/'bottleneck' instead, or "
                "switch your config to a 3D patch_mode (z_axis / cubic / "
                "whole) where the depth axis is preserved.")
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError(
                f"temporal_kernel must be a positive odd integer, "
                f"got {temporal_kernel}")
        d = 3
        t_pad = temporal_kernel // 2

        # 第一组 (2+1)D：in_ch → out_ch。
        self.spatial1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm_s1  = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_s1   = get_activation(activation)

        self.temporal1 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(temporal_kernel, 1, 1), padding=(t_pad, 0, 0), bias=False)
        self.norm_t1   = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_t1    = get_activation(activation)

        # 第二组 (2+1)D：out_ch → out_ch。
        self.spatial2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm_s2  = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_s2   = get_activation(activation)

        self.temporal2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(temporal_kernel, 1, 1), padding=(t_pad, 0, 0), bias=False)
        self.norm_t2   = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_out   = get_activation(activation)  # 残差相加后再激活

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        if attention_type == "none" and use_se:
            attention_type = "se"
        self.attn = make_attention(attention_type, out_ch, spatial_dims=d, reduction=se_reduction)

        self.shortcut = (
            nn.Sequential(
                _CONV[d](in_ch, out_ch, 1, bias=False),
                get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"R2Plus1DBlock expects rank-5 input (B, C, D, H, W); "
                f"got shape={tuple(x.shape)}.")
        res = self.shortcut(x)
        out = self.act_s1(self.norm_s1(self.spatial1(x)))
        out = self.act_t1(self.norm_t1(self.temporal1(out)))
        out = self.drop(out)
        # 后置激活：残差前不加 act（对齐 ResNetBlock）。
        out = self.act_s2(self.norm_s2(self.spatial2(out)))
        out = self.norm_t2(self.temporal2(out))
        out = self.attn(out)
        return self.act_out(out + res)


_BLOCK_REGISTRY = {
    "basic"     : ResNetBlock,
    "preact"    : PreActResNetBlock,
    "bottleneck": BottleneckBlock,
    "r2plus1d"  : R2Plus1DBlock}

BLOCK_TYPES = tuple(_BLOCK_REGISTRY.keys())


def _make_block(block_type: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    if block_type not in _BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block_type: {block_type!r}. Valid: {BLOCK_TYPES}")
    return _BLOCK_REGISTRY[block_type](in_ch, out_ch, **kwargs)


class ResNetStage(nn.Module):
    """同分辨率下的 N 个残差块，首块可变通道。block_type：'basic'/'preact'/'bottleneck'/'r2plus1d'。"""

    def __init__(
        self,
        in_ch         : int,
        out_ch        : int,
        num_blocks    : int = 2,
        norm_type     : str = "instance",
        norm_groups   : int = 8,
        activation    : str = "leakyrelu",
        dropout       : float = 0.0,
        use_se        : bool = False,
        se_reduction  : int = 16,
        attention_type: str = "none",
        block_type    : str = "basic",
        spatial_dims  : int = 3,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        kwargs = dict(
            norm_type=norm_type, norm_groups=norm_groups, activation=activation,
            dropout=dropout, use_se=use_se, se_reduction=se_reduction,
            attention_type=attention_type, spatial_dims=spatial_dims)
        blocks = [_make_block(block_type, in_ch, out_ch, **kwargs)]
        for _ in range(1, num_blocks):
            blocks.append(_make_block(block_type, out_ch, out_ch, **kwargs))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
