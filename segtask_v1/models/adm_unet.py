"""ADM U-Net segmentation backbone (Dhariwal & Nichol 2021).

Faithful re-impl of the enc/mid/dec blocks from ``guided_diffusion/unet.py``:
  - ``GroupNorm32`` (32 groups, fallback when not divisible) + SiLU.
  - ``ResBlock`` with the timestep-embedding path removed.
  - Multi-head ``AttentionBlock`` (QKVAttentionLegacy, zero-init ``proj_out``).
  - Stride-2 conv ``Downsample``; nearest+conv ``Upsample``.
  - Per-block skip topology identical to ADM (one skip per enc ResBlock
    + one per Downsample + one for the stem).

Removed (diffusion-only): ``time_embed`` / ``label_emb`` / ``use_scale_shift_norm`` /
``AttentionPool2d`` / fp16 utils / ``checkpoint`` / ``Encoder/SuperResModel``.

Multi-FOV stem: delegated to ``build_context_stem`` (``shared_stem`` |
``multi_stem_proj``). ``hierarchical`` is rejected up front in ``Config.validate``.

Intentional deviations from the paper:
  1. Stem goes through ``ConvNormAct`` (Conv3x3 + GN + SiLU) instead of the bare
     paper Conv2d, to compose with the multi-FOV stem helper.
  2. Output head is ``Conv1x1`` (:class:`SegmentationHead`) rather than the paper's
     ``GN → SiLU → Conv3x3 (zero-init)`` — same IO contract, smaller capacity.
  3. DS / aux heads are out-of-paper extensions following :class:`UNet3D` convention.
  4. ``resblock_updown`` not exposed; plain Down/Up always.
  5. Attention specified by level index (``adm_attention_levels``) instead of by
     downsample factor (paper's ``attention_resolutions``).

Optional extension (off by default; lucidrains-style ``denoising-diffusion-pytorch``):
  ``adm_linear_attention_levels`` adds ``Residual(PreNorm(LinearAttention))`` at the
  end of each listed level. Composes additively with ``adm_attention_levels``.

Stage-level dec feature capture: ``dec_features[i] = post-blocks, pre-upsample
feature of level (L-2-i)`` (length ``L-1``, ordered ``[low_res, ..., high_res]``);
mirrors :attr:`models.unet.Decoder.out_channels`. Deepest level is NOT captured.

Output contract mirrors :class:`models.unet.UNet3D`: eval/no-aux returns a tensor
(or list with DS); train + ``aux_seg_supervision`` returns ``{"main": ..., "aux": [...]}``.
2.5D folded: ``num_fg_classes = num_fg * D``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem import build_context_stem
from .unet import SegmentationHead, _build_aux_head

logger = logging.getLogger(__name__)


# ============================================================================
# Paper-faithful primitives (no-emb variants).
# ============================================================================


class _GroupNorm32(nn.GroupNorm):
    """ADM's ``GroupNorm32`` — fp32 forward then cast back."""

    def forward(self, x):  # type: ignore[override]
        return super().forward(x.float()).type(x.dtype)


def _norm(channels: int) -> nn.Module:
    """ADM's ``normalization`` — 32 groups by default, fallback when needed."""
    g = 32 if channels % 32 == 0 else math.gcd(channels, 32) or 1
    return _GroupNorm32(g, channels)


def _conv2d(*args, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(*args, **kwargs)


def _zero_(module: nn.Module) -> nn.Module:
    """ADM's ``zero_module``: zero-init all parameters in place."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class _Upsample(nn.Module):
    """ADM ``Upsample``: nearest 2× + optional conv."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = _conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # ``upsample_nearest2d`` lacks a bfloat16 / fp16 kernel on older
        # PyTorch builds (observed on torch < 2.1 with bf16 AMP).
        # Cast→interpolate→cast-back keeps the AMP outer-context happy
        # without forcing fp32 on the conv that follows.
        if x.dtype in (torch.bfloat16, torch.float16):
            orig_dtype = x.dtype
            x = F.interpolate(x.float(), scale_factor=2.0,
                              mode="nearest").to(orig_dtype)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    """ADM ``Downsample``: stride-2 conv (or avg pool when ``use_conv=False``)."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = _conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class _ResBlockNoEmb(nn.Module):
    """ADM ``ResBlock`` with the timestep-embedding path removed.

    Topology (matches ``guided_diffusion/unet.py:ResBlock`` minus the
    ``emb_layers`` / FiLM modulation):

        in_layers : norm → silu → conv3
        out_layers: norm → silu → dropout → conv3 (zero-init)
        skip      : Identity / conv1 / conv3 (matches paper's branching)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_conv_skip: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_layers = nn.Sequential(
            _norm(in_channels),
            nn.SiLU(),
            _conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            _norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_(_conv2d(out_channels, out_channels, 3, padding=1)),
        )
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        elif use_conv_skip:
            self.skip_connection = _conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = _conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class _QKVAttentionLegacy(nn.Module):
    """ADM ``QKVAttentionLegacy`` — split heads before split qkv."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class _LinearAttention(nn.Module):
    """O(N) linear attention (Shen 2021) via KᵀV trick: ``softmax(Q,feat) @ (softmax(K,spat) @ Vᵀ)ᵀ``.

    Cost ``O(D² N)`` instead of softmax attention's ``O(D N²)`` — enables attention on
    high-resolution maps without the N² memory blowup. Matches lucidrains' impl: 1×1
    QKV conv (no bias), Q/K softmax over feat/spatial, ``head_dim**-0.5`` scale on Q,
    final ``Conv1×1 + GroupNorm(1)`` projection.
    """

    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.scale = float(head_dim) ** -0.5
        self.num_heads = num_heads
        hidden = num_heads * head_dim
        self.to_qkv = nn.Conv2d(channels, hidden * 3, 1, bias=False)
        # ``GroupNorm(1, channels)`` ≡ per-sample LayerNorm over channels;
        # lucidrains uses this on the post-projection output (NOT zero-init).
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden, channels, 1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            t.reshape(b, self.num_heads, -1, h * w) for t in qkv
        )  # each [B, H, head_dim, N]
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        # context: [B, H, head_dim_v, head_dim_k] (square in head_dim).
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class _LinearAttentionBlock(nn.Module):
    """``Residual(PreNorm(LinearAttention))`` (lucidrains style).

    Pre-norm is ``GroupNorm(1)`` (per-sample LN over channels) to match
    LinearAttention's channel-softmax statistic. Composes additively with
    ADM's softmax ``_AttentionBlock``; placed once per level after all ResBlocks.
    """

    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = _LinearAttention(channels, num_heads=num_heads,
                                      head_dim=head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm(x)) + x


class _AttentionBlock(nn.Module):
    """ADM ``AttentionBlock`` — multi-head self-attention over flattened spatial."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
    ):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"channels {channels} not divisible by num_head_channels "
                f"{num_head_channels}")
            self.num_heads = channels // num_head_channels
        self.norm = _norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = _QKVAttentionLegacy(self.num_heads)
        self.proj_out = _zero_(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        h = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(h))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x.reshape(b, c, -1) + h).reshape(b, c, *spatial)


# ============================================================================
# ADM segmentation U-Net wrapper.
# ============================================================================


def _resolve_attention_levels(
    n_levels: int, attn_levels: Optional[Sequence[int]]
) -> List[int]:
    """Resolve attention level indices; default = deepest two levels (matches ADM's
    typical ``attention_resolutions=[16, 8]`` on a 256-input 5-level model).
    """
    if attn_levels is None:
        if n_levels >= 2:
            return [n_levels - 2, n_levels - 1]
        return [n_levels - 1]
    out = sorted({int(v) for v in attn_levels})
    for v in out:
        if v < 0 or v >= n_levels:
            raise ValueError(
                f"adm.attn_levels entry {v} out of range [0, {n_levels - 1}]")
    return out


class _ADMEncoder(nn.Module):
    """ADM encoder: stem + L levels of (nb × ResBlock + optional Attn) + Downsample.

    Returns ``{enc_features, enc_skips}``. ``enc_skips`` holds the full ADM stack
    (one per ResBlock + one per Downsample + one for the stem); decoder pops in reverse.
    ``enc_features`` is post-blocks-pre-downsample per level (diagnostics only).
    """

    def __init__(
        self,
        stem: nn.Module,
        encoder_channels: List[int],
        encoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        num_heads: int,
        num_head_channels: int,
        dropout: float,
        linear_attention_levels: Optional[List[int]] = None,
        linear_attention_num_heads: int = 4,
        linear_attention_head_dim: int = 32,
    ):
        super().__init__()
        self.stem = stem
        self.encoder_channels = list(encoder_channels)
        n_levels = len(encoder_channels)
        self.n_levels = n_levels
        self.attention_levels = list(attention_levels)
        self.linear_attention_levels = list(linear_attention_levels or [])
        self.encoder_blocks_per_stage = list(encoder_blocks_per_stage)

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        # Track skip-channel layout so the decoder can size cat-fusion in_ch.
        self.skip_channels: List[int] = [encoder_channels[0]]  # stem skip

        for level, ch_out in enumerate(encoder_channels):
            ch_in = encoder_channels[0] if level == 0 else encoder_channels[level - 1]
            blocks_this_level: List[nn.Module] = []
            n_blocks = encoder_blocks_per_stage[level]
            for i in range(n_blocks):
                blocks_this_level.append(
                    _ResBlockNoEmb(
                        ch_in if i == 0 else ch_out,
                        ch_out,
                        dropout=dropout,
                    )
                )
                if level in self.attention_levels:
                    blocks_this_level.append(
                        _AttentionBlock(
                            ch_out,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.skip_channels.append(ch_out)
            # One LinearAttention at end of level (after all ResBlocks + any softmax-attn);
            # rewrites the last pushed skip with the linear-attended feature.
            if level in self.linear_attention_levels:
                blocks_this_level.append(
                    _LinearAttentionBlock(
                        ch_out,
                        num_heads=linear_attention_num_heads,
                        head_dim=linear_attention_head_dim,
                    )
                )
            self.levels.append(nn.ModuleList(blocks_this_level))
            if level < n_levels - 1:
                self.downsamples.append(_Downsample(ch_out, use_conv=True))
                self.skip_channels.append(ch_out)
            else:
                self.downsamples.append(None)  # type: ignore[arg-type]

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, Any]:
        """Returns dict ``{enc_features, enc_skips}``."""
        x = self.stem(x)
        enc_skips: List[torch.Tensor] = [x]
        enc_features: List[torch.Tensor] = []
        for level, blocks in enumerate(self.levels):
            for blk in blocks:
                if isinstance(blk, _ResBlockNoEmb):
                    x = blk(x)
                    enc_skips.append(x)
                else:  # AttentionBlock
                    x = blk(x)
                    # ADM groups ResBlock + Attn in one TimestepEmbedSequential and
                    # pushes the sequential's output — i.e. attended feature replaces
                    # the previous skip rather than being appended.
                    enc_skips[-1] = x
            enc_features.append(x)
            if level < self.n_levels - 1:
                x = self.downsamples[level](x)
                enc_skips.append(x)
        return {
            "bottleneck": x,
            "enc_features": enc_features,
            "enc_skips": enc_skips,
        }


class _ADMMiddle(nn.Module):
    """ADM middle block: ResBlock → AttentionBlock → ResBlock."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_head_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.r1 = _ResBlockNoEmb(channels, channels, dropout=dropout)
        self.a = _AttentionBlock(
            channels, num_heads=num_heads, num_head_channels=num_head_channels)
        self.r2 = _ResBlockNoEmb(channels, channels, dropout=dropout)

    def forward(self, x):
        return self.r2(self.a(self.r1(x)))


class _ADMDecoder(nn.Module):
    """ADM decoder: per-level (nb+1) × (cat skip → ResBlock + opt Attn) + Upsample.

    Captures ``dec_features[i] = post-blocks, pre-upsample`` of level (L-2-i) for
    every level except the deepest. Length ``L-1``, ordered ``[low_res, ..., high_res]``.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        skip_channels: List[int],
        decoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        num_heads: int,
        num_head_channels: int,
        dropout: float,
        linear_attention_levels: Optional[List[int]] = None,
        linear_attention_num_heads: int = 4,
        linear_attention_head_dim: int = 32,
    ):
        super().__init__()
        self.encoder_channels = list(encoder_channels)
        n_levels = len(encoder_channels)
        self.n_levels = n_levels
        # Each level uses (decoder_blocks_per_stage[level] + 1) ResBlocks; length
        # mirrors the encoder counts (extra entry at deepest = bottleneck refinement).
        if len(decoder_blocks_per_stage) != n_levels:
            raise ValueError(
                f"decoder_blocks_per_stage length {len(decoder_blocks_per_stage)} "
                f"!= expected {n_levels}")
        self.decoder_blocks_per_stage = list(decoder_blocks_per_stage)

        # Build deepest → shallowest; pop skip_channels (in encoder push order) in reverse.
        skip_stack = list(skip_channels)
        self.levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # Start ch (entering level L-1) = bottleneck = encoder_channels[-1].
        ch = encoder_channels[-1]
        for ridx, level in enumerate(reversed(range(n_levels))):
            n_blocks = self.decoder_blocks_per_stage[level] + 1
            blocks_this_level: List[nn.Module] = []
            for i in range(n_blocks):
                ich = skip_stack.pop()
                blocks_this_level.append(
                    _ResBlockNoEmb(
                        ch + ich,
                        encoder_channels[level],
                        dropout=dropout,
                    )
                )
                ch = encoder_channels[level]
                if level in attention_levels:
                    blocks_this_level.append(
                        _AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
            # One LinearAttention at end of level (after cat-fusion ResBlocks + softmax-attn).
            # Decoder linear-attn does not touch the skip stack (pop-count = ResBlock count).
            if linear_attention_levels and level in linear_attention_levels:
                blocks_this_level.append(
                    _LinearAttentionBlock(
                        ch,
                        num_heads=linear_attention_num_heads,
                        head_dim=linear_attention_head_dim,
                    )
                )
            self.levels.append(nn.ModuleList(blocks_this_level))
            if level > 0:
                # Upsample preserves channel count (ADM convention).
                self.upsamples.append(_Upsample(ch, use_conv=True))
            else:
                self.upsamples.append(None)  # type: ignore[arg-type]
        if skip_stack:
            raise RuntimeError(
                f"ADM decoder skip-stack mismatch: {len(skip_stack)} "
                f"skip(s) left after build (expected 0).")

        # Decoder out_channels (low_res → high_res) for levels [L-2, ..., 0].
        # Used by the wrapping module to size DS / aux heads identically
        # to UNet3D's ``decoder.out_channels`` contract.
        self.out_channels: List[int] = [
            encoder_channels[level] for level in reversed(range(n_levels - 1))
        ]

    def forward(
        self, bottleneck: torch.Tensor, enc_skips: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Returns ``dec_features`` of length ``n_levels - 1``,
        ordered ``[low_res, ..., high_res]``.
        """
        x = bottleneck
        skip_stack = list(enc_skips)
        # NOTE: bottleneck is the *post-Downsample* feature pushed last
        # onto enc_skips. ADM pops it as the FIRST skip in level L-1's
        # first block. Replicate that behaviour exactly.
        dec_features: List[torch.Tensor] = []
        for ridx, level in enumerate(reversed(range(self.n_levels))):
            for blk in self.levels[ridx]:
                if isinstance(blk, _ResBlockNoEmb):
                    skip = skip_stack.pop()
                    if skip.shape[2:] != x.shape[2:]:
                        # Defensive — should never trigger when input
                        # spatial dims align with the encoder strides.
                        skip = F.interpolate(
                            skip, size=x.shape[2:],
                            mode="bilinear", align_corners=False)
                    x = blk(torch.cat([x, skip], dim=1))
                else:  # AttentionBlock
                    x = blk(x)
            # Capture every level *except* the deepest (level L-1) so
            # the dec_features length matches UNet3D's convention.
            if level < self.n_levels - 1:
                dec_features.append(x)
            if self.upsamples[ridx] is not None:
                x = self.upsamples[ridx](x)
        if skip_stack:
            raise RuntimeError(
                f"ADM decoder leftover skips at runtime: {len(skip_stack)}")
        return dec_features


class ADMSegModel(nn.Module):
    """ADM U-Net adapted as a segmentation model.

    See module docstring for the design rationale and the contract for
    ``forward()``'s return type (mirrors :class:`models.unet.UNet3D`).
    """

    def __init__(
        self,
        # Stem / multi-FOV.
        stem: nn.Module,
        stem_stride: int,
        context_n_views: int,
        context_fusion: str,
        # Encoder topology.
        encoder_channels: List[int],
        encoder_blocks_per_stage: List[int],
        decoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        num_heads: int,
        num_head_channels: int,
        dropout: float,
        # Output / heads.
        num_fg_classes: int,
        deep_supervision: bool,
        aux_seg_supervision: bool,
        aux_head_mode: str,
        # Optional lucidrains-style linear attention (composes with softmax attn).
        linear_attention_levels: Optional[List[int]] = None,
        linear_attention_num_heads: int = 4,
        linear_attention_head_dim: int = 32,
        aux_head_out_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.spatial_dims = 2
        self.stem_stride = int(stem_stride)
        self.context_n_views = int(context_n_views)
        self.context_fusion = context_fusion
        self.deep_supervision = bool(deep_supervision)
        self.num_fg_classes = int(num_fg_classes)

        self.encoder = _ADMEncoder(
            stem=stem,
            encoder_channels=encoder_channels,
            encoder_blocks_per_stage=encoder_blocks_per_stage,
            attention_levels=attention_levels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            linear_attention_levels=linear_attention_levels,
            linear_attention_num_heads=linear_attention_num_heads,
            linear_attention_head_dim=linear_attention_head_dim,
        )
        self.middle = _ADMMiddle(
            channels=encoder_channels[-1],
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
        )
        self.decoder = _ADMDecoder(
            encoder_channels=encoder_channels,
            skip_channels=self.encoder.skip_channels,
            decoder_blocks_per_stage=decoder_blocks_per_stage,
            attention_levels=attention_levels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            linear_attention_levels=linear_attention_levels,
            linear_attention_num_heads=linear_attention_num_heads,
            linear_attention_head_dim=linear_attention_head_dim,
        )

        # 1×1 logit conv (paper uses Conv3+GN+SiLU; we match UNet3D / DS contract).
        self.seg_head = SegmentationHead(
            self.decoder.out_channels[-1],
            num_fg_classes,
            spatial_dims=2,
        )

        # DS heads on lower-res decoder features (matches UNet3D).
        self.ds_heads = nn.ModuleList()
        if self.deep_supervision:
            for ch in reversed(self.decoder.out_channels[:-1]):
                self.ds_heads.append(
                    SegmentationHead(ch, num_fg_classes, spatial_dims=2)
                )

        # Aux seg supervision (Plan A only: all aux heads on highest-res dec feat).
        n_views = self.context_n_views
        self.aux_seg_supervision = bool(aux_seg_supervision and n_views > 1)
        n_aux_expected = max(n_views - 1, 0) if self.aux_seg_supervision else 0
        if aux_head_out_channels is None:
            aux_out = [num_fg_classes] * n_aux_expected
        else:
            if len(aux_head_out_channels) != n_aux_expected:
                raise ValueError(
                    f"aux_head_out_channels length "
                    f"{len(aux_head_out_channels)} != expected "
                    f"{n_aux_expected}")
            aux_out = [int(c) for c in aux_head_out_channels]
        self.aux_head_out_channels = aux_out
        self.aux_head_mode = aux_head_mode
        self.aux_heads = nn.ModuleList()
        self.aux_feat_indices: List[int] = []
        if self.aux_seg_supervision:
            in_ch = self.decoder.out_channels[-1]
            n_dec = len(self.decoder.out_channels)
            for k in range(1, n_views):
                self.aux_feat_indices.append(n_dec - 1)
                self.aux_heads.append(
                    _build_aux_head(
                        mode=aux_head_mode,
                        in_ch=in_ch,
                        num_classes=aux_out[k - 1],
                        spatial_dims=2,
                        # GN+SiLU matches ADM style (only used when mode='conv').
                        norm_type="group",
                        norm_groups=32 if in_ch % 32 == 0 else 8,
                        activation="swish",  # SiLU
                    )
                )

    # ----- Forward --------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        target_size = x.shape[2:]
        enc_out = self.encoder(x)
        bottleneck = self.middle(enc_out["bottleneck"])
        dec_features = self.decoder(bottleneck, enc_out["enc_skips"])

        # Main head (always at decoder's highest-res feature).
        main_out = self.seg_head(dec_features[-1])
        if main_out.shape[2:] != target_size:
            main_out = F.interpolate(
                main_out, size=target_size,
                mode="bilinear", align_corners=False)

        # Aux heads (training only; mirrors UNet3D contract).
        aux_outs: List[torch.Tensor] = []
        if self.aux_seg_supervision and self.training:
            for head, feat_idx in zip(self.aux_heads, self.aux_feat_indices):
                ao = head(dec_features[feat_idx])
                if ao.shape[2:] != target_size:
                    ao = F.interpolate(
                        ao, size=target_size,
                        mode="bilinear", align_corners=False)
                aux_outs.append(ao)

        # Main path with optional deep supervision.
        if self.deep_supervision and self.training:
            main_path: Union[torch.Tensor, List[torch.Tensor]] = [main_out]
            for i, head in enumerate(self.ds_heads):
                main_path.append(head(dec_features[-2 - i]))
        else:
            main_path = main_out

        if aux_outs:
            return {"main": main_path, "aux": aux_outs}
        return main_path

    def param_count(self) -> Dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        mid = sum(p.numel() for p in self.middle.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.seg_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "encoder": enc, "middle": mid, "decoder": dec,
            "seg_head": head, "total": total,
        }


# ============================================================================
# Public factory.
# ============================================================================


def build_adm_seg_model(cfg) -> ADMSegModel:
    """Build :class:`ADMSegModel` from :class:`segtask_v1.config.Config`.

    Reads the standard data/model fields plus ADM-specific:
      ``adm_attention_levels`` (level indices; default = [L-2, L-1]),
      ``adm_num_heads`` (default 4), ``adm_num_head_channels`` (default -1).
    ``model.dropout`` is reused as the ADM ResBlock dropout.
    """
    mc = cfg.model
    enc_channels = list(mc.encoder_channels)
    n_levels = len(enc_channels)
    num_fg = cfg.num_fg_classes

    # 2.5D-only for now (other modes need stem/in_ch/out_ch wiring).
    assert cfg.data.patch_mode == "2_5d", (
        "arch='adm' is currently only wired for patch_mode='2_5d'.")
    D = int(cfg.data.patch_size[0])
    out_classes = num_fg * D
    n_views = max(len(cfg.data.multi_res_scales), 1)

    # Per-stage block counts.
    enc_bps = list(mc.encoder_blocks_per_stage)
    if not enc_bps:
        enc_bps = [int(mc.blocks_per_level)] * n_levels
    if len(enc_bps) != n_levels:
        raise ValueError(
            f"encoder_blocks_per_stage length {len(enc_bps)} "
            f"!= len(encoder_channels) {n_levels}")
    # Skip-stack balance fixes dec_bps_full[k] = enc_bps[k] (decoder adds the +1
    # internally). User-supplied decoder_blocks_per_stage is ignored — warn if set.
    if mc.decoder_blocks_per_stage and (
            list(mc.decoder_blocks_per_stage) != enc_bps[:-1]):
        logger.warning(
            "model.arch='adm' ignores model.decoder_blocks_per_stage=%s; "
            "ADM's per-level decoder count is fixed by the encoder's "
            "skip-stack topology (nb+1 ResBlocks per level, paper-faithful). "
            "Using encoder_blocks_per_stage=%s instead.",
            list(mc.decoder_blocks_per_stage), enc_bps)
    dec_bps_full = list(enc_bps)

    attn_levels = _resolve_attention_levels(
        n_levels, getattr(mc, "adm_attention_levels", None))

    # Optional lucidrains-style linear attention; off by default.
    raw_lin = getattr(mc, "adm_linear_attention_levels", None) or []
    lin_attn_levels: List[int] = sorted({int(v) for v in raw_lin})
    for v in lin_attn_levels:
        if v < 0 or v >= n_levels:
            raise ValueError(
                f"adm_linear_attention_levels entry {v} out of range "
                f"[0, {n_levels - 1}]")
    lin_num_heads = int(getattr(mc, "adm_linear_attention_num_heads", 4))
    lin_head_dim = int(getattr(mc, "adm_linear_attention_head_dim", 32))

    # Only shared_stem / multi_stem_proj supported; hierarchical needs mid-encoder injection.
    if mc.context_fusion == "hierarchical":
        raise ValueError(
            "model.arch='adm' does not yet support context_fusion="
            "'hierarchical'. Use 'shared_stem' or 'multi_stem_proj' "
            "for ADM. (Hierarchical fusion will be added in a follow-up.)")

    # Native-depth ON: per-view variable D_k; OFF: uniform D.
    in_ch_per_view_list = None
    aux_head_out_channels = None
    if (bool(getattr(cfg.data, "aux_keep_native_d", False))
            and n_views > 1):
        depths = list(cfg.aux_view_depths)
        in_ch_per_view_list = depths
        aux_head_out_channels = [num_fg * d_k for d_k in depths[1:]]
        in_channels = sum(depths)
    else:
        in_channels = D * n_views
    in_ch_per_view = D  # uniform fallback

    stem, stem_stride = build_context_stem(
        mode=mc.stem_mode,
        fusion=mc.context_fusion,
        n_views=n_views,
        in_ch_per_view=in_ch_per_view,
        out_ch=enc_channels[0],
        # ADM-style stem: GroupNorm + SiLU.
        norm_type="group",
        norm_groups=32 if enc_channels[0] % 32 == 0 else 8,
        activation="swish",  # SiLU
        spatial_dims=2,
        stage_channels=enc_channels,
        in_ch_per_view_list=in_ch_per_view_list)

    aux_seg = bool(getattr(mc, "aux_seg_supervision", False)) and n_views > 1

    model = ADMSegModel(
        stem=stem,
        stem_stride=stem_stride,
        context_n_views=n_views,
        context_fusion=mc.context_fusion,
        encoder_channels=enc_channels,
        encoder_blocks_per_stage=enc_bps,
        decoder_blocks_per_stage=dec_bps_full,
        attention_levels=attn_levels,
        num_heads=int(getattr(mc, "adm_num_heads", 4)),
        num_head_channels=int(getattr(mc, "adm_num_head_channels", -1)),
        dropout=float(getattr(mc, "dropout", 0.0)),
        linear_attention_levels=lin_attn_levels,
        linear_attention_num_heads=lin_num_heads,
        linear_attention_head_dim=lin_head_dim,
        num_fg_classes=out_classes,
        deep_supervision=bool(mc.deep_supervision),
        aux_seg_supervision=aux_seg,
        aux_head_mode=str(getattr(mc, "aux_head_mode", "linear")),
        aux_head_out_channels=aux_head_out_channels,
    )

    pc = model.param_count()
    logger.info(
        "Built ADMSegModel: enc=%.2fM, mid=%.2fM, dec=%.2fM, total=%.2fM, "
        "channels=%s, enc_blocks=%s, dec_blocks=%s, attn_levels=%s, "
        "lin_attn_levels=%s (heads=%d, head_dim=%d), "
        "in_ch=%d (per_view=%s), out_classes=%d (fg=%d, D=%d), "
        "stem=%s(stride=%d, n_views=%d, fusion=%s), ds=%s, aux_seg=%s "
        "(n_aux=%d, mode=%s)",
        pc["encoder"] / 1e6, pc["middle"] / 1e6, pc["decoder"] / 1e6,
        pc["total"] / 1e6,
        enc_channels, enc_bps, dec_bps_full, attn_levels,
        lin_attn_levels, lin_num_heads, lin_head_dim,
        in_channels,
        in_ch_per_view_list if in_ch_per_view_list is not None
        else [in_ch_per_view] * n_views,
        out_classes, num_fg, D,
        mc.stem_mode, stem_stride, n_views, mc.context_fusion,
        bool(mc.deep_supervision), aux_seg,
        len(model.aux_heads), getattr(mc, "aux_head_mode", "linear"))

    return model
