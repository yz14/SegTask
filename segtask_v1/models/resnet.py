"""ResNet building blocks for 3D UNet encoder and decoder.

ResNet block: two 3x3x3 convolutions with residual connection.
If in_ch != out_ch, a 1x1x1 projection is used for the shortcut.

Encoder level: N ResNet blocks (no downsampling inside — handled by Downsample).
Decoder level: N ResNet blocks after skip-connection fusion.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import ConvNormAct, SqueezeExcite3D, get_norm, get_activation


class ResNetBlock(nn.Module):
    """Single ResNet block: conv-norm-act-conv-norm + optional SE + residual."""

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
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = get_norm(norm_type, out_ch, norm_groups)
        self.act1  = get_activation(activation)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_ch, norm_groups)
        self.act2  = get_activation(activation)

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.se   = SqueezeExcite3D(out_ch, se_reduction) if use_se else nn.Identity()

        # Shortcut projection if channel mismatch
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                get_norm(norm_type, out_ch, norm_groups))
            if in_ch != out_ch
            else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return self.act2(out + residual)


class ResNetStage(nn.Module):
    """A stage of N ResNet blocks at a fixed resolution.

    First block may change channels (in_ch → out_ch).
    Subsequent blocks maintain out_ch.
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
    ):
        super().__init__()
        blocks = [ResNetBlock(in_ch, out_ch, norm_type, norm_groups, activation, dropout, use_se, se_reduction)]
        for _ in range(1, num_blocks):
            blocks.append(ResNetBlock(out_ch, out_ch, norm_type, norm_groups, activation, dropout, use_se, se_reduction))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
