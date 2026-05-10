"""ResNet building blocks for 3D UNet encoder and decoder.

Four block variants are provided (selectable per-stage):

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

- ``R2Plus1DBlock`` (``r2plus1d``, **3D-only**)
    Factorized (2+1)D residual block. Each logical 3×3×3 convolution is
    split into a spatial 2D conv (kernel 1×3×3, operating on H/W) followed
    by a temporal 1D conv (kernel 3×1×1, operating along the depth axis).
    Source: Tran et al., "A Closer Look at Spatiotemporal Convolutions for
    Action Recognition" (R(2+1)D, CVPR 2018) and Qiu et al. "P3D-ResNet"
    (ICCV 2017). Adopted in nnFormer / MedNeXt for medical volumes with
    thin-slab geometry, where the z axis carries genuine context but a
    full isotropic 3×3×3 kernel is overkill (and parameter-heavy).

    Why: lets a network with a (mostly) 2D inductive bias inject explicit
    inter-slice context at a fraction of full-3D-conv FLOPs. Total params
    per block ≈ ``24·C²`` vs. ``54·C²`` for full-3D basic (k=3, ignoring
    shortcut), and the spatial sub-conv stays initialisable from any 2D
    pretrained weight set.

    Restriction: requires ``spatial_dims == 3``. Using it in 2.5D mode
    (``spatial_dims == 2``, D folded into the channel axis) is rejected
    in ``__init__`` — the depth axis must be present as a real spatial
    dimension for the temporal sub-conv to be meaningful.

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
# R(2+1)D residual block — 3D-only.
#
# Each logical 3×3×3 conv is factorised into:
#   * spatial sub-conv : kernel (1, 3, 3), padding (0, 1, 1) — H/W only.
#   * temporal sub-conv: kernel (3, 1, 1), padding (1, 0, 0) — D only.
# Norm + activation are inserted between the two sub-convs (per the
# original R(2+1)D paper; this is the key ingredient that makes the
# factorisation strictly more expressive than a plain 3D conv with the
# same param budget — non-linearity in the middle decouples spatial vs.
# temporal feature spaces).
#
# We keep the *number of intermediate channels equal to ``out_ch``* (no
# rank-reducing bottleneck). The R(2+1)D paper proposes an intermediate
# width that exactly matches the param count of an isotropic 3D conv,
# but in our setting the isotropic baseline is ``ResNetBlock`` (already
# in the registry). Setting mid = out_ch keeps the block's internal
# behaviour easy to reason about while still being substantially cheaper
# (and stronger on z-context) than full 3D — see the docstring at the top
# of this file for the param accounting.
# ---------------------------------------------------------------------------
class R2Plus1DBlock(nn.Module):
    """Factorised (2+1)D residual block. ``spatial_dims=3`` only.

    Mirrors :class:`ResNetBlock`'s post-activation residual structure but
    replaces each 3×3×3 conv with a (1×3×3) spatial conv → norm → act →
    (3×1×1) temporal conv. The block is stride-1; downsampling stays
    external (handled by ``blocks.Downsample`` between stages, identical
    to all other block types).

    Args mirror :class:`ResNetBlock`. ``temporal_kernel`` defaults to 3.
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
            # The temporal sub-conv only makes sense when D is a real
            # spatial axis. In 2.5D (spatial_dims=2) D has been folded
            # into the channel axis, so a (3,1,1) kernel cannot reach
            # neighbouring slices. Fail fast with a precise diagnostic.
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

        # --- First (2+1)D pair (in_ch → out_ch via a mid layer of out_ch) ---
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

        # --- Second (2+1)D pair (out_ch → out_ch) ----------------------------
        self.spatial2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), bias=False)
        self.norm_s2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act_s2 = get_activation(activation)
        self.temporal2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(temporal_kernel, 1, 1),
            padding=(t_pad, 0, 0), bias=False)
        self.norm_t2 = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        # Final activation applied AFTER residual addition (post-act style).
        self.act_out = get_activation(activation)

        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

        # Attention on the post-conv pre-residual feature, matching ``ResNetBlock``.
        if attention_type == "none" and use_se:
            attention_type = "se"
        self.attn = make_attention(
            attention_type, out_ch, spatial_dims=d, reduction=se_reduction)

        # Channel-mismatch shortcut: 1×1×1 + norm (mirrors ``ResNetBlock``).
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
        # First (2+1)D pair: spatial → temporal, with mid-non-linearity.
        out = self.act_s1(self.norm_s1(self.spatial1(x)))
        out = self.act_t1(self.norm_t1(self.temporal1(out)))
        out = self.drop(out)
        # Second (2+1)D pair: matching post-act ResNet structure — no
        # activation between norm_t2 and the residual add (the final act
        # fires after addition, like ResNetBlock).
        out = self.act_s2(self.norm_s2(self.spatial2(out)))
        out = self.norm_t2(self.temporal2(out))
        out = self.attn(out)
        return self.act_out(out + residual)


# ---------------------------------------------------------------------------
# Block-type dispatch registry
# ---------------------------------------------------------------------------
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
