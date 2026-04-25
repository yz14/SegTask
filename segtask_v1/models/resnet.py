"""ResNet building blocks for 3D UNet encoder and decoder.

Three block variants are provided (selectable per-stage):

- ``ResNetBlock`` (``basic``, post-activation, default)
    Classic: conv → norm → act → conv → norm (+ attention) → residual → act.
    Lightweight and the best default for shallow-to-medium networks.

- ``PreActResNetBlock`` (``preact``)
    Pre-activation (He et al., "Identity Mappings in Deep Residual Networks",
    ECCV 2016): norm → act → conv → norm → act → conv (+ attention) → residual.
    Trains better at depth (16+ blocks per stage), recommended for
    nnU-Net ResEnc-L / XL configurations.

- ``BottleneckBlock`` (``bottleneck``)
    1×1×1 reduce → 3×3×3 → 1×1×1 expand with an inverted-residual style
    4× expansion (nnU-Net ResEnc XL), post-activation with pre-residual
    norm matching the original ResNet-50 design.

Encoder level: N blocks (asymmetric counts allowed via
``encoder_blocks_per_stage`` in config). Downsampling is external
(see blocks.Downsample).
Decoder level: M blocks (typically 1 in ResEnc), applied after skip fusion.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import (
    _CONV, _DROP, ConvNormAct, get_activation, get_norm, make_attention)


class ResNetBlock(nn.Module):
    """Single ResNet block: conv-norm-act-conv-norm + optional attention + residual.

    Attention variant is controlled by ``attention_type`` (none/se/eca/cbam/coord).
    The legacy ``use_se`` flag remains for backwards compatibility and is
    treated as ``attention_type='se'`` when attention_type is 'none'.
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

        # Back-compat: promote legacy use_se → attention_type="se" when the
        # caller did not set attention_type explicitly.
        if attention_type == "none" and use_se:
            attention_type = "se"
        self.attn = make_attention(attention_type, out_ch,
                                   spatial_dims=d, reduction=se_reduction)

        # Shortcut projection if channel mismatch
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
    """Pre-activation ResNet block (He et al., ECCV 2016).

    Order: norm → act → conv → norm → act → conv (+ attention) → residual.
    The raw ``x`` (no norm/act applied) forms the identity path, which
    empirically improves gradient flow for deep encoders (ResEnc L/XL).
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

        # Shortcut is applied on the ORIGINAL x (no normalisation).  If the
        # channel count changes we use a 1×1(×1) projection (still in the raw
        # identity path — this follows the canonical pre-act design).
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
    """Inverted-residual / ResNet-50-style bottleneck block (3D).

    1×1×1 reduce → 3×3×3 → 1×1×1 expand with ``expansion`` = 4 by default.
    Used in nnU-Net ResEnc-XL for exceptionally deep encoders where basic
    blocks become parameter-heavy.
    """

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


# ---------------------------------------------------------------------------
# Block-type dispatch registry
# ---------------------------------------------------------------------------
_BLOCK_REGISTRY = {
    "basic": ResNetBlock,
    "preact": PreActResNetBlock,
    "bottleneck": BottleneckBlock,
}

BLOCK_TYPES = tuple(_BLOCK_REGISTRY.keys())


def _make_block(block_type: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    if block_type not in _BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block_type: {block_type!r}. Valid: {BLOCK_TYPES}")
    return _BLOCK_REGISTRY[block_type](in_ch, out_ch, **kwargs)


class ResNetStage(nn.Module):
    """A stage of N residual blocks at a fixed resolution.

    First block may change channels (in_ch → out_ch).
    Subsequent blocks maintain out_ch.

    ``block_type`` selects the residual unit: "basic" (default),
    "preact", or "bottleneck" (see module docstring).
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
