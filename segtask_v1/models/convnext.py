"""ConvNeXt building blocks for 3D UNet encoder and decoder.

ConvNeXt block (adapted from "A ConvNet for the 2020s", Liu et al. 2022):
  - Depthwise 7x7x7 conv
  - LayerNorm (channels-first, mathematically equivalent to the official
    channels-last LN since stats are over the C axis only)
  - Pointwise expansion (4x, 1x1 conv ≡ Linear in channels-last)
  - GELU activation
  - Pointwise projection back
  - LayerScale (learnable per-channel scale, init ``1e-6``) — critical for
    stable training of deep networks; matches official block. Set
    ``layer_scale_init_value <= 0`` to disable.
  - Residual connection + optional drop path

Encoder level: N ConvNeXt blocks (no downsampling — handled separately).
Decoder level: N ConvNeXt blocks after skip-connection fusion.

A paper-faithful ``ConvNeXtDownsample`` (LayerNorm → Conv stride 2) is
also provided for use between encoder stages when the backbone is ConvNeXt.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import _CONV, make_attention


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
    """Channel-first LayerNorm — dim-agnostic (works for 2D and 3D inputs).

    Computes per-spatial-position statistics across the channel axis.
    Class name kept for API stability; auto-detects ndim from input.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial) — stats over C dimension.
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
        spatial_dims: int = 3,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        d = spatial_dims
        self.spatial_dims = d
        hidden = int(dim * expand_ratio)

        self.dwconv = _CONV[d](dim, dim, kernel_size=7, padding=3,
                               groups=dim, bias=True)
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = _CONV[d](dim, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = _CONV[d](hidden, dim, kernel_size=1, bias=True)
        self.attn = make_attention(attention_type, dim, spatial_dims=d)
        # LayerScale (Touvron et al. "Going deeper with Image Transformers",
        # adopted by ConvNeXt). Initialised at 1e-6 so the block starts
        # near-identity — essential for stable training when combined with
        # stochastic depth. Disabled when ``layer_scale_init_value <= 0``.
        if layer_scale_init_value > 0.0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.register_parameter("gamma", None)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.attn(out)
        if self.gamma is not None:
            # Reshape gamma to (1, C, 1, ...) for broadcast on channel-first.
            shape = (1, -1) + (1,) * self.spatial_dims
            out = out * self.gamma.reshape(shape)
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
        spatial_dims: int = 3,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        d = spatial_dims
        self.proj = (
            nn.Sequential(
                _CONV[d](in_ch, out_ch, 1, bias=False),
                LayerNorm3d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.block = ConvNeXtBlock(
            out_ch, expand_ratio, drop_path,
            attention_type=attention_type,
            spatial_dims=d,
            layer_scale_init_value=layer_scale_init_value)

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
        spatial_dims: int = 3,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        d = spatial_dims
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        blocks = [ConvNeXtAdaptBlock(
            in_ch, out_ch, expand_ratio,
            drop_path_rates[0], attention_type,
            spatial_dims=d,
            layer_scale_init_value=layer_scale_init_value)]
        for i in range(1, num_blocks):
            dp = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            blocks.append(ConvNeXtAdaptBlock(
                out_ch, out_ch, expand_ratio,
                dp, attention_type,
                spatial_dims=d,
                layer_scale_init_value=layer_scale_init_value))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvNeXtDownsample(nn.Module):
    """Paper-faithful ConvNeXt inter-stage downsample: LayerNorm → Conv(s=2).

    Official ConvNeXt (Liu et al. 2022) places a *separate* downsample layer
    between every pair of stages, consisting of:

        LayerNorm(C_in)  →  Conv(k=2, s=2, C_in → C_out)

    Two differences from the generic :class:`Downsample` in ``blocks.py``:

    1. **Norm-first ordering**: LayerNorm is applied *before* the stride-2
       convolution (not after), regularising the input distribution to the
       downsampling conv. This is what the paper does and what every
       reference re-implementation (timm, mmcls, monai) follows.
    2. **LayerNorm specifically** (not BN/IN/GN), matching the channel
       statistics used inside each ConvNeXt block.

    Channel projection (``in_ch → out_ch``) is folded into the stride-2 conv
    — no extra 1x1 needed. When ``in_ch == out_ch`` (the layout this UNet
    uses, where channel growth happens inside the next stage's first block),
    the conv is square in channels but the spatial halving and the
    pre-conv LN remain paper-faithful.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_dims: int = 3,
    ):
        super().__init__()
        d = spatial_dims
        self.norm = LayerNorm3d(in_ch)
        self.conv = _CONV[d](in_ch, out_ch, kernel_size=2, stride=2,
                             bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))
