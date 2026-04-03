"""Vision Transformer (ViT) encoder for UNet.

Uses patch embedding + transformer blocks at each encoder level.
The spatial resolution is reduced by patch merging between levels
(similar to Swin Transformer's approach for hierarchical features).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import get_conv, get_norm, get_activation


class PatchEmbed(nn.Module):
    """Convert spatial feature map to patch tokens via convolution."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 2,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
    ):
        super().__init__()
        Conv = get_conv(spatial_dims)
        self.proj = Conv(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.norm = get_norm(norm_type, embed_dim, spatial_dims, norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention operating on spatial feature maps.

    Input/output: (B, C, *spatial) — we flatten spatial dims internally,
    run attention, then reshape back.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial)
        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]
        N = 1
        for s in spatial_shape:
            N *= s

        # (B, C, N) -> (B, N, C)
        x_flat = x.reshape(B, C, N).permute(0, 2, 1)

        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))

        # (B, N, C) -> (B, C, *spatial)
        out = out.permute(0, 2, 1).reshape(B, C, *spatial_shape)
        return out


class TransformerBlock(nn.Module):
    """Transformer block operating on spatial feature maps.

    Pre-norm design: Norm -> Attention -> Residual -> Norm -> MLP -> Residual.
    Uses GroupNorm (works on spatial feature maps) instead of LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.norm1 = get_norm(norm_type, dim, spatial_dims, norm_groups)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=dropout, proj_drop=dropout,
        )
        self.norm2 = get_norm(norm_type, dim, spatial_dims, norm_groups)

        mlp_hidden = int(dim * mlp_ratio)
        Conv = get_conv(spatial_dims)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            Conv(mlp_hidden, dim, 1),
            nn.Dropout(dropout),
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask / keep_prob


class ViTEncoder(nn.Module):
    """Vision Transformer encoder with hierarchical feature maps.

    Architecture per level:
    1. Patch embedding (downsample spatial resolution by 2x, except level 0)
    2. N transformer blocks
    3. Output feature map at this resolution

    This produces multi-scale features suitable for UNet skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        blocks_per_level: List[int] = None,
        spatial_dims: int = 2,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        patch_size: int = 2,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]
        if blocks_per_level is None:
            blocks_per_level = [2] * len(channels)

        assert len(channels) == len(blocks_per_level)

        self.num_levels = len(channels)
        self.out_channels = channels

        # Stochastic depth schedule
        total_blocks = sum(blocks_per_level)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stem = nn.Sequential(
            get_conv(spatial_dims)(in_channels, channels[0], 3, padding=1, bias=False),
            get_norm(norm_type, channels[0], spatial_dims, norm_groups),
            get_activation(activation),
        )

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        block_idx = 0
        for i, (ch, n_blocks) in enumerate(zip(channels, blocks_per_level)):
            # Ensure num_heads divides channel dim
            n_heads = min(num_heads, ch)
            while ch % n_heads != 0 and n_heads > 1:
                n_heads -= 1

            blocks = []
            for j in range(n_blocks):
                blocks.append(
                    TransformerBlock(
                        dim=ch,
                        num_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        dropout=dropout,
                        drop_path=dpr[block_idx],
                        spatial_dims=spatial_dims,
                        norm_type=norm_type,
                        norm_groups=norm_groups,
                    )
                )
                block_idx += 1
            self.levels.append(nn.Sequential(*blocks))

            if i < len(channels) - 1:
                self.downsamples.append(
                    PatchEmbed(
                        ch, channels[i + 1],
                        patch_size=patch_size,
                        spatial_dims=spatial_dims,
                        norm_type=norm_type,
                        norm_groups=norm_groups,
                    )
                )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        features = []
        for i, level in enumerate(self.levels):
            x = level(x)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return features
