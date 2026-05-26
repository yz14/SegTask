"""ResNet blocks for 3D/2D UNet stages. Block types (block_type):
  - 'basic'     : post-act, conv-norm-act-conv-norm; default, light
  - 'preact'    : norm-act-conv-... (He 2016); better for deep ResEnc-L/XL
  - 'bottleneck': 1x1 reduce / 3x3 / 1x1 expand (ResEnc-XL)
  - 'r2plus1d'  : (1,3,3) spatial + (3,1,1) temporal, 3D-only; cheap z-context
Downsampling is external (blocks.Downsample).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import (
    _CONV, _DROP, ConvNormAct, get_activation, get_norm, make_attention)


class ResNetBlock(nn.Module):
    """Post-act ResNet block (+ optional attention). attention_type: none|se|eca|cbam|coord.
    Legacy use_se=True is promoted to attention_type='se' when not set.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        attention_type: str = "none",
        spatial_dims: int = 3):
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
            attention_type = "se"  # legacy use_se back-compat
        self.attn = make_attention(attention_type, out_ch,
                                   spatial_dims=d, reduction=se_reduction)

        self.shortcut = (
            nn.Sequential(_CONV[d](in_ch, out_ch, 1, bias=False),
                          get_norm(norm_type, out_ch, norm_groups,
                                   spatial_dims=d))
            if in_ch != out_ch
            else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.attn(out)
        return self.act2(out + residual)


class PreActResNetBlock(nn.Module):
    """Pre-act ResNet block (He 2016): norm-act-conv x2 + residual; better for deep encoders."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        attention_type: str = "none",
        spatial_dims: int = 3,
    ):
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
        self.attn = make_attention(attention_type, out_ch,
                                   spatial_dims=d, reduction=se_reduction)

        # shortcut on raw x; channel-mismatch uses 1x1 projection (canonical pre-act)
        self.shortcut = (
            _CONV[d](in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.drop(out)
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.attn(out)
        return out + residual


class BottleneckBlock(nn.Module):
    """ResNet-50-style bottleneck: 1x1 reduce → 3x3 → 1x1 expand (expansion=4, for ResEnc-XL)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expansion: int = 4,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        attention_type: str = "none",
        spatial_dims: int = 3,
    ):
        super().__init__()
        d = spatial_dims
        mid = max(out_ch // expansion, 1)

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
        self.attn = make_attention(attention_type, out_ch,
                                   spatial_dims=d, reduction=se_reduction)

        self.shortcut = (
            nn.Sequential(_CONV[d](in_ch, out_ch, 1, bias=False),
                          get_norm(norm_type, out_ch, norm_groups,
                                   spatial_dims=d))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.drop(out)
        out = self.norm3(self.conv3(out))
        out = self.attn(out)
        return self.act3(out + residual)


class R2Plus1DBlock(nn.Module):
    """R(2+1)D residual block (Tran 2018), 3D-only.
    Each 3x3x3 → (1,3,3) spatial conv + norm + act + (3,1,1) temporal conv.
    Mid non-linearity is essential. mid_ch = out_ch (no bottleneck).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        attention_type: str = "none",
        spatial_dims: int = 3,
        temporal_kernel: int = 3,
    ):
        super().__init__()
        if spatial_dims != 3:
            # D must be a real axis; in 2.5D D is folded into channels.
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

        # First (2+1)D pair (in_ch → out_ch)
        self.spatial1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), bias=False)
        self.norm_s1 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_s1 = get_activation(activation)
        self.temporal1 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(temporal_kernel, 1, 1),
            padding=(t_pad, 0, 0), bias=False)
        self.norm_t1 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_t1 = get_activation(activation)

        # Second (2+1)D pair (out_ch → out_ch)
        self.spatial2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), bias=False)
        self.norm_s2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_s2 = get_activation(activation)
        self.temporal2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(temporal_kernel, 1, 1),
            padding=(t_pad, 0, 0), bias=False)
        self.norm_t2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_out = get_activation(activation)  # applied after residual add

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        if attention_type == "none" and use_se:
            attention_type = "se"
        self.attn = make_attention(
            attention_type, out_ch, spatial_dims=d, reduction=se_reduction)

        self.shortcut = (
            nn.Sequential(
                _CONV[d](in_ch, out_ch, 1, bias=False),
                get_norm(norm_type, out_ch, norm_groups, spatial_dims=d))
            if in_ch != out_ch
            else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"R2Plus1DBlock expects rank-5 input (B, C, D, H, W); "
                f"got shape={tuple(x.shape)}.")
        residual = self.shortcut(x)
        out = self.act_s1(self.norm_s1(self.spatial1(x)))
        out = self.act_t1(self.norm_t1(self.temporal1(out)))
        out = self.drop(out)
        # post-act: no act before residual add (matches ResNetBlock)
        out = self.act_s2(self.norm_s2(self.spatial2(out)))
        out = self.norm_t2(self.temporal2(out))
        out = self.attn(out)
        return self.act_out(out + residual)


_BLOCK_REGISTRY = {
    "basic": ResNetBlock,
    "preact": PreActResNetBlock,
    "bottleneck": BottleneckBlock,
    "r2plus1d": R2Plus1DBlock,
}

BLOCK_TYPES = tuple(_BLOCK_REGISTRY.keys())


def _make_block(block_type: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    if block_type not in _BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block_type: {block_type!r}. Valid: {BLOCK_TYPES}")
    return _BLOCK_REGISTRY[block_type](in_ch, out_ch, **kwargs)


class ResNetStage(nn.Module):
    """N residual blocks at one resolution. First block may change channels.
    block_type: 'basic' (default) | 'preact' | 'bottleneck' | 'r2plus1d'.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        attention_type: str = "none",
        block_type: str = "basic",
        spatial_dims: int = 3,
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
