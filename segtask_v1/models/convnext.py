"""ConvNeXt building blocks for 3D UNet encoder and decoder.

ConvNeXt block (adapted from "A ConvNet for the 2020s", Liu et al.):
  - Depthwise 7x7x7 conv
  - LayerNorm (channels-last)
  - Pointwise expansion (4x)
  - GELU activation
  - Pointwise projection back
  - Residual connection + optional drop path

Encoder level: N ConvNeXt blocks (no downsampling — handled by Downsample).
Decoder level: N ConvNeXt blocks after skip-connection fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import make_attention


class DropPath(nn.Module):
    """Stochastic depth (drop path) for residual blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Sample the Bernoulli mask in float32 regardless of x.dtype. Under
        # AMP (fp16/bf16) torch.bernoulli on the probability tensor has
        # inconsistent backend support across CUDA versions — generate in
        # fp32, then cast to x.dtype. The /keep division is fused into the
        # cast to preserve numerical stability.
        prob = torch.full(shape, keep, device=x.device, dtype=torch.float32)
        mask = torch.bernoulli(prob).to(dtype=x.dtype)
        return x * mask / keep


class LayerNorm3d(nn.Module):
    """Channel-first LayerNorm for 3D data: (B, C, D, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) → compute stats over C
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.weight.reshape(shape) + self.bias.reshape(shape)


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block (3D version).

    Architecture:
      depthwise 7x7x7 conv → LayerNorm → 1x1 expand (4x) → GELU → 1x1 project
      → optional attention (se/eca/cbam/coord) → residual.
    """

    def __init__(
        self,
        dim: int,
        expand_ratio: float = 4.0,
        drop_path: float = 0.0,
        attention_type: str = "none",
    ):
        super().__init__()
        hidden = int(dim * expand_ratio)

        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, dim, kernel_size=1, bias=True)
        self.attn = make_attention(attention_type, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.attn(out)
        return residual + self.drop_path(out)


class ConvNeXtAdaptBlock(nn.Module):
    """ConvNeXt block with channel adaptation.

    If in_ch != out_ch, applies a 1x1 projection before the ConvNeXt block.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: float = 4.0,
        drop_path: float = 0.0,
        attention_type: str = "none",
    ):
        super().__init__()
        self.proj = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                LayerNorm3d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.block = ConvNeXtBlock(out_ch, expand_ratio, drop_path,
                                   attention_type=attention_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))


class ConvNeXtStage(nn.Module):
    """A stage of N ConvNeXt blocks at a fixed resolution.

    First block may change channels. Subsequent blocks maintain out_ch.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int = 2,
        expand_ratio: float = 4.0,
        drop_path_rates: list = None,
        attention_type: str = "none",
    ):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        blocks = [ConvNeXtAdaptBlock(in_ch, out_ch, expand_ratio,
                                     drop_path_rates[0], attention_type)]
        for i in range(1, num_blocks):
            dp = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            blocks.append(ConvNeXtAdaptBlock(out_ch, out_ch, expand_ratio,
                                             dp, attention_type))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
