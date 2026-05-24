"""EDM2 U-Net segmentation backbone (Karras et al., CVPR 2024).

Faithful reimplementation of the magnitude-preserving (MP) U-Net blocks
from ``edm2/training/networks_edm2.py``:

  * ``MPConv`` with **forced weight normalization** during training and
    magnitude-preserving scaling at every forward.
  * ``mp_silu`` (SiLU rescaled by 1/0.596 to preserve magnitude).
  * ``mp_sum`` / ``mp_cat`` (magnitude-preserving residual + skip-cat).
  * Pixel-norm at the start of each ``Block`` (encoder flavor).
  * ``Block`` with optional resampling (``up``/``down``) and optional
    multi-head self-attention. The diffusion noise/class-embedding path
    (``emb_linear``, ``emb_gain``, FiLM modulation) is **removed** —
    segmentation does not need timestep conditioning.
  * Encoder: per-level ``_conv`` (level 0) or ``_down`` Block (level>0)
    followed by ``num_blocks`` enc Blocks; one skip is pushed per block.
  * Decoder: deepest level uses ``_in0`` (attention) + ``_in1`` Blocks;
    other levels use ``_up`` Block (resample 'up'); each level then
    runs ``num_blocks + 1`` dec Blocks, every one ``mp_cat``ing the
    next skip onto its input.

Removed (vs. the original)
--------------------------
  * ``MPFourier`` noise embedding & class-label embedding (no diffusion).
  * The +1 "ones" channel concatenation at input (paper trick for
    diffusion conditioning); irrelevant to segmentation.
  * The ``emb_linear`` FiLM modulation inside each ``Block``.

Stem
----
The multi-FOV input is consumed by an MP-native multi-FOV stem:

  * ``shared_stem``     — single ``MPConv`` over ``n_views * D``
                          channels.
  * ``multi_stem_proj`` — per-view ``MPConv`` → ``mp_cat`` → ``MPConv``
                          1×1 fusion back to ``encoder_channels[0]``.
  * ``hierarchical``    — rejected (same as ADM, will be added later).

Output contract
---------------
Mirrors :class:`models.unet.UNet3D`:

  * eval / no aux → tensor (or list when DS is on at construction).
  * train + ``aux_seg_supervision`` → ``{"main": ..., "aux": [...]}``.

For 2.5D folded mode ``num_fg_classes = num_fg * D`` (main head); aux
head ``k`` emits ``num_fg * D_k`` channels with ``aux_keep_native_d``.

Deviations vs. the original EDM2 paper / repo (audited & accepted)
------------------------------------------------------------------
The MP primitives + ``Block`` internals + per-level encoder/decoder
topology are byte-faithful to ``training/networks_edm2.py``:
``MPConv`` with forced weight normalization & magnitude-preserving
scaling, ``mp_silu``, ``mp_sum``, ``mp_cat``, ``_normalize`` (pixel-norm),
``_resample`` (low-pass filter resample), encoder = ``_conv|_down`` +
``nb`` enc Blocks, decoder deepest = ``_in0(attn=True) + _in1`` /
others = ``_up`` + ``nb+1`` dec Blocks each ``mp_cat``-ing one skip,
``concat_balance`` / ``res_balance`` / ``attn_balance`` parameters,
learnable ``out_gain`` scalar.

Intentional deviations made for segmentation use:

  1. **Output head kernel.** Paper uses ``MPConv kernel=[3, 3]``;
     we use ``MPConv kernel=[1, 1]`` for parity with the other archs'
     1×1 seg head contract.
  2. **Multi-FOV stem.** Paper has a single ``MPConv 3×3`` stem
     (single-resolution input). For the 2.5D multi-FOV setting we
     compose per-view ``MPConv 3×3`` stems with ``mp_cat`` + an
     ``MPConv 1×1`` fusion. NOTE: an extra ``mp_silu`` is currently
     applied to each per-view stem output before ``mp_cat``. The
     paper's stem feeds directly into the next ``Block``'s pixel-norm
     + ``conv_res0`` without an intervening activation, so this is
     a (very small) divergence; left in to keep behaviour matching
     the smoke-tested baseline. Remove ``_mp_silu`` from
     ``_MPMultiStemProj.forward`` to recover paper-faithful flow.
  3. **DS / aux heads.** Outside the paper's scope (segmentation-
     only). Implemented as ``MPSegHead`` (MPConv 1×1 with its own
     ``out_gain``) per UNet3D conventions.

What's *removed* (also intentionally) — diffusion-only:
  - ``MPFourier`` noise embedding & ``emb_label`` MPConv.
  - The ``emb_linear`` FiLM modulation inside each ``Block`` (the
    ``mp_silu(y * (emb*gain + 1))`` step is replaced by ``mp_silu(y)``).
  - The ``cat([x, ones_like(x[:, :1])], dim=1)`` ones-channel trick
    at encoder input.
  - ``Precond`` wrapper (sigma preconditioning, EDM2 only).

DS / aux head capture rule mirrors ADM's: capture
``dec_features[i] = post-blocks of level (L-2-i)`` (i.e. *after* that
level's dec Blocks, which is post-resample for non-deepest levels);
length ``L - 1``, ordered ``[low_res, ..., high_res]``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem import HierarchicalStems  # only for the explicit reject branch

logger = logging.getLogger(__name__)


# ============================================================================
# Magnitude-preserving primitives (paper-faithful).
# ============================================================================


def _normalize(x: torch.Tensor, dim=None, eps: float = 1e-4) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=float(np.sqrt(norm.numel() / x.numel())))
    return x / norm.to(x.dtype)


def _resample(x: torch.Tensor, f=(1, 1), mode: str = "keep") -> torch.Tensor:
    if mode == "keep":
        return x
    f_arr = np.float32(f)
    assert f_arr.ndim == 1 and len(f_arr) % 2 == 0
    pad = (len(f_arr) - 1) // 2
    f_arr = f_arr / f_arr.sum()
    f2 = np.outer(f_arr, f_arr)[None, None, :, :]
    f_t = x.new_tensor(f2)
    c = x.shape[1]
    if mode == "down":
        return F.conv2d(
            x, f_t.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == "up"
    return F.conv_transpose2d(
        x, (f_t * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))


def _mp_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x) / 0.596


def _mp_sum(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return a.lerp(b, t) / float(np.sqrt((1 - t) ** 2 + t ** 2))


def _mp_cat(a: torch.Tensor, b: torch.Tensor, dim: int = 1, t: float = 0.5) -> torch.Tensor:
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = float(np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2)))
    wa = C / float(np.sqrt(Na)) * (1 - t)
    wb = C / float(np.sqrt(Nb)) * t
    return torch.cat([wa * a, wb * b], dim=dim)


class _MPConv(nn.Module):
    """Magnitude-preserving conv / fully-connected with forced weight norm.

    Faithful to ``edm2.training.networks_edm2.MPConv`` (kernel-empty
    list ⇒ FC; otherwise 2D conv with ``padding=k//2``). Independent
    of ``torch_utils.persistence`` (we drop persistence for runtime
    flexibility).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(_normalize(w))  # forced weight normalization
        w = _normalize(w)  # traditional weight normalization
        # Magnitude-preserving scaling: divide by sqrt(fan_in).
        fan_in = float(w[0].numel())
        w = w * (gain / float(np.sqrt(fan_in)))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4, f"unexpected weight ndim {w.ndim}"
        return F.conv2d(x, w, padding=(w.shape[-1] // 2,))


class _Block(nn.Module):
    """EDM2 ``Block`` with the diffusion-emb path removed.

    Topology (matches ``networks_edm2.Block`` minus the
    ``emb_linear`` / ``emb_gain`` FiLM):

        x = resample(x, mode)
        if flavor == 'enc' and conv_skip is not None: x = conv_skip(x)
        if flavor == 'enc': x = pixel_norm(x)
        y = conv_res0(mp_silu(x))
        y = mp_silu(y)               # ← was: mp_silu(y * (emb*gain + 1))
        y = dropout(y) if training
        y = conv_res1(y)
        if flavor == 'dec' and conv_skip is not None: x = conv_skip(x)
        x = mp_sum(x, y, t=res_balance)
        if attention:
            y = attn_qkv(x); ... ; y = attn_proj(y)
            x = mp_sum(x, y, t=attn_balance)
        x = clip(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        flavor: str = "enc",
        resample_mode: str = "keep",
        resample_filter=(1, 1),
        attention: bool = False,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        res_balance: float = 0.3,
        attn_balance: float = 0.3,
        clip_act: float = 256.0,
    ):
        super().__init__()
        assert flavor in ("enc", "dec")
        assert resample_mode in ("keep", "up", "down")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = tuple(resample_filter)
        self.resample_mode = resample_mode
        self.num_heads = (out_channels // channels_per_head) if attention else 0
        self.dropout = float(dropout)
        self.res_balance = float(res_balance)
        self.attn_balance = float(attn_balance)
        self.clip_act = clip_act

        self.conv_res0 = _MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels, kernel=[3, 3])
        self.conv_res1 = _MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = (
            _MPConv(in_channels, out_channels, kernel=[1, 1])
            if in_channels != out_channels else None)
        self.attn_qkv = (
            _MPConv(out_channels, out_channels * 3, kernel=[1, 1])
            if self.num_heads != 0 else None)
        self.attn_proj = (
            _MPConv(out_channels, out_channels, kernel=[1, 1])
            if self.num_heads != 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main branch.
        x = _resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = _normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(_mp_silu(x))
        y = _mp_silu(y)  # emb FiLM removed
        if self.training and self.dropout != 0:
            y = F.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = _mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = _normalize(y, dim=2).unbind(3)
            w_attn = torch.einsum(
                "nhcq,nhck->nhqk", q, k / float(np.sqrt(q.shape[2]))
            ).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w_attn, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = _mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-float(self.clip_act), float(self.clip_act))
        return x


# ============================================================================
# Multi-FOV MP stem.
# ============================================================================


class _MPMultiStemProj(nn.Module):
    """``n_views`` independent ``MPConv`` stems → ``mp_cat`` → ``MPConv 1×1`` fusion.

    All sub-stems use a stride-1 ``3×3`` ``MPConv`` (matches EDM2's
    encoder ``_conv`` entry at level 0). Output spatial dims are
    preserved (``stem_stride = 1``).
    """

    def __init__(
        self,
        n_views: int,
        in_ch_per_view_list: List[int],
        out_ch: int,
    ):
        super().__init__()
        assert len(in_ch_per_view_list) == n_views
        self.n_views = n_views
        self.in_ch_per_view_list = [int(c) for c in in_ch_per_view_list]
        self.stems = nn.ModuleList([
            _MPConv(c_v, out_ch, kernel=[3, 3])
            for c_v in self.in_ch_per_view_list
        ])
        # 1×1 fusion back to out_ch (cat is 2 ways at a time via mp_cat).
        # When n_views >= 3 we fold by repeated mp_cat then a single
        # MPConv 1×1 over the ``n_views * out_ch`` cat'd channels.
        self.proj = _MPConv(n_views * out_ch, out_ch, kernel=[1, 1])
        self.stem_stride = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = sum(self.in_ch_per_view_list)
        if x.shape[1] != expected:
            raise ValueError(
                f"_MPMultiStemProj expected {expected} input channels "
                f"(per_view={self.in_ch_per_view_list}); got {x.shape[1]}")
        chunks = torch.split(x, self.in_ch_per_view_list, dim=1)
        feats = [_mp_silu(stem(c)) for stem, c in zip(self.stems, chunks)]
        # Repeated mp_cat keeps magnitude approximately preserved across
        # the n-view fold (each pairwise mp_cat is t=0.5 by default).
        out = feats[0]
        for f in feats[1:]:
            out = _mp_cat(out, f, dim=1)
        return self.proj(out)


class _MPSharedStem(nn.Module):
    """Single ``MPConv 3×3`` over the full ``n_views*D`` input slab."""

    def __init__(self, in_channels: int, out_ch: int):
        super().__init__()
        self.conv = _MPConv(in_channels, out_ch, kernel=[3, 3])
        self.stem_stride = 1

    def forward(self, x):
        return self.conv(x)


def _build_edm2_stem(
    fusion: str,
    n_views: int,
    in_ch_per_view: int,
    out_ch: int,
    in_ch_per_view_list: Optional[List[int]] = None,
):
    """Dispatch to the right MP stem variant. Rejects 'hierarchical'."""
    if fusion == "hierarchical":
        raise ValueError(
            "model.arch='edm2' does not yet support context_fusion="
            "'hierarchical'. Use 'shared_stem' or 'multi_stem_proj'.")
    per_view = (
        list(in_ch_per_view_list) if in_ch_per_view_list is not None
        else [in_ch_per_view] * n_views)
    if n_views == 1 or fusion == "shared_stem":
        return _MPSharedStem(int(sum(per_view)), out_ch)
    if fusion == "multi_stem_proj":
        return _MPMultiStemProj(n_views, per_view, out_ch)
    raise ValueError(f"unknown context_fusion: {fusion!r}")


# ============================================================================
# Encoder / Decoder.
# ============================================================================


def _resolve_attn_levels(
    n_levels: int, attn_levels: Optional[Sequence[int]]
) -> List[int]:
    """Default = deepest level only (matches the typical EDM2 ``attn_resolutions``
    being a single small res like 16)."""
    if attn_levels is None:
        return [n_levels - 1]
    out = sorted({int(v) for v in attn_levels})
    for v in out:
        if v < 0 or v >= n_levels:
            raise ValueError(
                f"edm2.attn_levels entry {v} out of range [0, {n_levels - 1}]")
    return out


class _EDM2Encoder(nn.Module):
    """Per-level ``_conv`` (level 0) or ``_down`` (level>0) + ``num_blocks`` enc
    Blocks. Each block pushes a skip onto the stack.
    """

    def __init__(
        self,
        stem: nn.Module,
        encoder_channels: List[int],
        encoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        block_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.stem = stem
        self.encoder_channels = list(encoder_channels)
        n_levels = len(encoder_channels)
        self.n_levels = n_levels
        self.attention_levels = list(attention_levels)
        self.encoder_blocks_per_stage = list(encoder_blocks_per_stage)

        self.level_blocks = nn.ModuleList()
        # The level-entry op:
        #   level 0 = stem (already applied) → identity here.
        #   level k>0 = _down Block (preserves channels, halves res).
        self.level_entries = nn.ModuleList()
        # Track skip-channel layout for the decoder to size cat-fusion.
        self.skip_channels: List[int] = [encoder_channels[0]]  # stem feature

        for level in range(n_levels):
            ch = encoder_channels[level]
            if level == 0:
                self.level_entries.append(nn.Identity())
            else:
                # ``_down`` keeps in/out channels equal to the previous level's.
                prev_ch = encoder_channels[level - 1]
                self.level_entries.append(
                    _Block(prev_ch, prev_ch,
                           flavor="enc", resample_mode="down",
                           **block_kwargs))
                self.skip_channels.append(prev_ch)
            blocks: List[nn.Module] = []
            cin = encoder_channels[0] if level == 0 else encoder_channels[level - 1]
            n_blocks = encoder_blocks_per_stage[level]
            for idx in range(n_blocks):
                cout = ch
                blocks.append(
                    _Block(
                        cin, cout,
                        flavor="enc",
                        attention=(level in self.attention_levels),
                        **block_kwargs))
                cin = cout
                self.skip_channels.append(cout)
            self.level_blocks.append(nn.ModuleList(blocks))

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        x = self.stem(x)
        enc_skips: List[torch.Tensor] = [x]
        enc_features: List[torch.Tensor] = []
        for level in range(self.n_levels):
            entry = self.level_entries[level]
            if not isinstance(entry, nn.Identity):
                x = entry(x)
                enc_skips.append(x)
            for blk in self.level_blocks[level]:
                x = blk(x)
                enc_skips.append(x)
            enc_features.append(x)
        return {
            "bottleneck": x,
            "enc_features": enc_features,
            "enc_skips": enc_skips,
        }


class _EDM2Decoder(nn.Module):
    """Per-level entry (deepest=``_in0+_in1``, else ``_up``) + ``num_blocks+1``
    dec Blocks. Every dec block ``mp_cat``s the next skip onto its input.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        skip_channels: List[int],
        decoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        concat_balance: float,
        block_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.encoder_channels = list(encoder_channels)
        n_levels = len(encoder_channels)
        self.n_levels = n_levels
        self.concat_balance = float(concat_balance)
        if len(decoder_blocks_per_stage) != n_levels:
            raise ValueError(
                f"decoder_blocks_per_stage length {len(decoder_blocks_per_stage)} "
                f"!= expected {n_levels}")

        skip_stack = list(skip_channels)
        self.level_entries = nn.ModuleList()
        self.level_entries_kind: List[str] = []
        self.level_blocks = nn.ModuleList()
        ch = encoder_channels[-1]  # entering deepest level
        for ridx, level in enumerate(reversed(range(n_levels))):
            attention = (level in attention_levels)
            if level == n_levels - 1:
                # Deepest: _in0 (attention=True) + _in1
                self.level_entries.append(
                    _Block(ch, ch, flavor="dec", attention=True,
                           **block_kwargs))
                # _in1 chained inside the same entry (use a small Sequential).
                # Express both as a single ModuleList for clarity.
                self.level_entries_kind.append("in0_in1")
                self._in1 = _Block(ch, ch, flavor="dec", **block_kwargs)
            else:
                self.level_entries.append(
                    _Block(ch, ch, flavor="dec", resample_mode="up",
                           **block_kwargs))
                self.level_entries_kind.append("up")
            n_blocks = decoder_blocks_per_stage[level] + 1
            blocks: List[nn.Module] = []
            for idx in range(n_blocks):
                ich = skip_stack.pop()
                cin = ch + ich
                cout = encoder_channels[level]
                blocks.append(
                    _Block(cin, cout, flavor="dec",
                           attention=attention,
                           **block_kwargs))
                ch = cout
            self.level_blocks.append(nn.ModuleList(blocks))
        if skip_stack:
            raise RuntimeError(
                f"EDM2 decoder skip-stack mismatch: {len(skip_stack)} left")

        self.out_channels: List[int] = [
            encoder_channels[level] for level in reversed(range(n_levels - 1))
        ]

    def forward(
        self, bottleneck: torch.Tensor, enc_skips: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        x = bottleneck
        skip_stack = list(enc_skips)
        dec_features: List[torch.Tensor] = []
        for ridx, level in enumerate(reversed(range(self.n_levels))):
            entry = self.level_entries[ridx]
            kind = self.level_entries_kind[ridx]
            x = entry(x)
            if kind == "in0_in1":
                x = self._in1(x)
            for blk in self.level_blocks[ridx]:
                skip = skip_stack.pop()
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(
                        skip, size=x.shape[2:],
                        mode="bilinear", align_corners=False)
                x = _mp_cat(x, skip, dim=1, t=self.concat_balance)
                x = blk(x)
            if level < self.n_levels - 1:
                dec_features.append(x)
        if skip_stack:
            raise RuntimeError(
                f"EDM2 decoder leftover skips: {len(skip_stack)}")
        return dec_features


# ============================================================================
# MP-aware seg head.
# ============================================================================


class _MPSegHead(nn.Module):
    """1×1 MPConv classifier with a learnable ``out_gain`` scalar
    (matches EDM2's ``out_conv`` topology, but with kernel=[1,1] for our
    seg-head contract instead of paper's [3,3])."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = _MPConv(in_ch, num_classes, kernel=[1, 1])
        self.out_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``self.out_gain`` is a 0-dim trainable scalar; ``MPConv.forward``
        # multiplies the (already MP-scaled) weight by ``gain`` before the
        # conv. Passing the tensor directly preserves autograd.
        return self.conv(x, gain=self.out_gain)


# ============================================================================
# Top-level wrapper.
# ============================================================================


class EDM2SegModel(nn.Module):
    """EDM2 U-Net adapted as a segmentation model.

    See module docstring; output contract mirrors :class:`models.unet.UNet3D`.
    """

    def __init__(
        self,
        # Stem.
        stem: nn.Module,
        stem_stride: int,
        context_n_views: int,
        context_fusion: str,
        # Encoder topology.
        encoder_channels: List[int],
        encoder_blocks_per_stage: List[int],
        decoder_blocks_per_stage: List[int],
        attention_levels: List[int],
        # Block hyperparams.
        channels_per_head: int,
        dropout: float,
        res_balance: float,
        attn_balance: float,
        concat_balance: float,
        clip_act: float,
        # Output / heads.
        num_fg_classes: int,
        deep_supervision: bool,
        aux_seg_supervision: bool,
        aux_head_mode: str,  # 'linear' only for EDM2 (MP-style)
        aux_head_out_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.spatial_dims = 2
        self.stem_stride = int(stem_stride)
        self.context_n_views = int(context_n_views)
        self.context_fusion = context_fusion
        self.deep_supervision = bool(deep_supervision)
        self.num_fg_classes = int(num_fg_classes)

        block_kwargs = dict(
            channels_per_head=channels_per_head,
            dropout=dropout,
            res_balance=res_balance,
            attn_balance=attn_balance,
            clip_act=clip_act,
        )

        self.encoder = _EDM2Encoder(
            stem=stem,
            encoder_channels=encoder_channels,
            encoder_blocks_per_stage=encoder_blocks_per_stage,
            attention_levels=attention_levels,
            block_kwargs=block_kwargs,
        )
        self.decoder = _EDM2Decoder(
            encoder_channels=encoder_channels,
            skip_channels=self.encoder.skip_channels,
            decoder_blocks_per_stage=decoder_blocks_per_stage,
            attention_levels=attention_levels,
            concat_balance=concat_balance,
            block_kwargs=block_kwargs,
        )

        # Main + DS heads (MP-style).
        self.seg_head = _MPSegHead(
            self.decoder.out_channels[-1], num_fg_classes)
        self.ds_heads = nn.ModuleList()
        if self.deep_supervision:
            for ch in reversed(self.decoder.out_channels[:-1]):
                self.ds_heads.append(_MPSegHead(ch, num_fg_classes))

        # Aux seg supervision (Plan A only).
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
                self.aux_heads.append(_MPSegHead(in_ch, aux_out[k - 1]))

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        target_size = x.shape[2:]
        enc_out = self.encoder(x)
        dec_features = self.decoder(enc_out["bottleneck"], enc_out["enc_skips"])

        main_out = self.seg_head(dec_features[-1])
        if main_out.shape[2:] != target_size:
            main_out = F.interpolate(
                main_out, size=target_size,
                mode="bilinear", align_corners=False)

        aux_outs: List[torch.Tensor] = []
        if self.aux_seg_supervision and self.training:
            for head, feat_idx in zip(self.aux_heads, self.aux_feat_indices):
                ao = head(dec_features[feat_idx])
                if ao.shape[2:] != target_size:
                    ao = F.interpolate(
                        ao, size=target_size,
                        mode="bilinear", align_corners=False)
                aux_outs.append(ao)

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
        dec = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.seg_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "decoder": dec, "seg_head": head,
                "total": total}


# ============================================================================
# Public factory.
# ============================================================================


def build_edm2_seg_model(cfg) -> EDM2SegModel:
    """Build :class:`EDM2SegModel` from a :class:`segtask_v1.config.Config`.

    Reads the same whitelisted set of config fields as :func:`build_adm_seg_model`
    plus EDM2-specific hyper-parameters:

      * ``model.edm2_attention_levels`` (List[int]; default = [L-1])
      * ``model.edm2_channels_per_head`` (default 64)
      * ``model.edm2_res_balance`` (default 0.3)
      * ``model.edm2_attn_balance`` (default 0.3)
      * ``model.edm2_concat_balance`` (default 0.5)
      * ``model.edm2_clip_act`` (default 256.0)

    The shared ``model.dropout`` is reused as EDM2's ``dropout`` Block kwarg.
    """
    mc = cfg.model
    enc_channels = list(mc.encoder_channels)
    n_levels = len(enc_channels)
    num_fg = cfg.num_fg_classes

    assert cfg.data.patch_mode == "2_5d", (
        "arch='edm2' is currently only wired for patch_mode='2_5d'.")
    D = int(cfg.data.patch_size[0])
    out_classes = num_fg * D
    n_views = max(len(cfg.data.multi_res_scales), 1)

    enc_bps = list(mc.encoder_blocks_per_stage)
    if not enc_bps:
        enc_bps = [int(mc.blocks_per_level)] * n_levels
    if len(enc_bps) != n_levels:
        raise ValueError(
            f"encoder_blocks_per_stage length {len(enc_bps)} "
            f"!= len(encoder_channels) {n_levels}")
    # Same paper-mandated coupling as ADM: encoder pushes ``nb_k + 1``
    # skips per level, decoder pops ``nb_k + 1`` (the +1 is added inside
    # the decoder's per-level loop). User-supplied
    # ``decoder_blocks_per_stage`` is ignored — log a warning when set.
    if mc.decoder_blocks_per_stage and (
            list(mc.decoder_blocks_per_stage) != enc_bps[:-1]):
        logger.warning(
            "model.arch='edm2' ignores model.decoder_blocks_per_stage=%s; "
            "EDM2's per-level decoder count is fixed by the encoder's "
            "skip-stack topology (nb+1 dec Blocks per level, paper-faithful). "
            "Using encoder_blocks_per_stage=%s instead.",
            list(mc.decoder_blocks_per_stage), enc_bps)
    dec_bps_full = list(enc_bps)

    attn_levels = _resolve_attn_levels(
        n_levels, getattr(mc, "edm2_attention_levels", None))

    if mc.context_fusion == "hierarchical":
        raise ValueError(
            "model.arch='edm2' does not yet support context_fusion="
            "'hierarchical'. Use 'shared_stem' or 'multi_stem_proj'.")

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
    in_ch_per_view = D

    stem = _build_edm2_stem(
        fusion=mc.context_fusion,
        n_views=n_views,
        in_ch_per_view=in_ch_per_view,
        out_ch=enc_channels[0],
        in_ch_per_view_list=in_ch_per_view_list,
    )
    stem_stride = stem.stem_stride

    aux_seg = bool(getattr(mc, "aux_seg_supervision", False)) and n_views > 1

    model = EDM2SegModel(
        stem=stem,
        stem_stride=stem_stride,
        context_n_views=n_views,
        context_fusion=mc.context_fusion,
        encoder_channels=enc_channels,
        encoder_blocks_per_stage=enc_bps,
        decoder_blocks_per_stage=dec_bps_full,
        attention_levels=attn_levels,
        channels_per_head=int(getattr(mc, "edm2_channels_per_head", 64)),
        dropout=float(getattr(mc, "dropout", 0.0)),
        res_balance=float(getattr(mc, "edm2_res_balance", 0.3)),
        attn_balance=float(getattr(mc, "edm2_attn_balance", 0.3)),
        concat_balance=float(getattr(mc, "edm2_concat_balance", 0.5)),
        clip_act=float(getattr(mc, "edm2_clip_act", 256.0)),
        num_fg_classes=out_classes,
        deep_supervision=bool(mc.deep_supervision),
        aux_seg_supervision=aux_seg,
        aux_head_mode=str(getattr(mc, "aux_head_mode", "linear")),
        aux_head_out_channels=aux_head_out_channels,
    )

    pc = model.param_count()
    logger.info(
        "Built EDM2SegModel: enc=%.2fM, dec=%.2fM, total=%.2fM, "
        "channels=%s, enc_blocks=%s, dec_blocks=%s, attn_levels=%s, "
        "in_ch=%d (per_view=%s), out_classes=%d (fg=%d, D=%d), "
        "stem=%s(stride=%d, n_views=%d, fusion=%s), ds=%s, aux_seg=%s "
        "(n_aux=%d, mode=%s)",
        pc["encoder"] / 1e6, pc["decoder"] / 1e6, pc["total"] / 1e6,
        enc_channels, enc_bps, dec_bps_full, attn_levels,
        in_channels,
        in_ch_per_view_list if in_ch_per_view_list is not None
        else [in_ch_per_view] * n_views,
        out_classes, num_fg, D,
        getattr(mc, "stem_mode", "conv3"), stem_stride, n_views,
        mc.context_fusion,
        bool(mc.deep_supervision), aux_seg,
        len(model.aux_heads), getattr(mc, "aux_head_mode", "linear"))

    return model
