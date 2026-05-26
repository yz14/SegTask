"""Stem / patch-embed builders for 3D UNet.

Stem modes (control encoder input resolution):
  conv3 / conv7 / dual : stride-1 (preserve resolution)
  patch2 / patch4      : stride-N patchify (halves / quarters resolution; UNet wrapper
                         applies a final upsample to restore output resolution)

Multi-FOV context fusion (2.5D, n_views>1):
  shared_stem     : single stem over all (n_views * D) channels (cheapest)
  multi_stem_proj : per-view stems → concat → 1×1 fusion (recommended; FOV-specific filters)
  hierarchical    : view 0 drives main stem at native stride; aux view k uses a patchify
                    stem of stride main_stride * 2^k, injected at encoder stage k
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import _CONV, ConvNormAct, get_activation, get_norm


STEM_MODES = ("conv3", "conv7", "dual", "patch2", "patch4")


class DualConvStem(nn.Module):
    """Two stacked 3×3×3 conv-norm-act blocks (nnU-Net stem)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.block1 = ConvNormAct(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        self.block2 = ConvNormAct(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

    def forward(self, x):
        return self.block2(self.block1(x))


class PatchEmbedStem(nn.Module):
    """Patch-embedding stem: stride-N conv + norm + activation; reduces resolution by ``patch_size``."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        patch_size: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "gelu",
        spatial_dims: int = 3,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size
        self.conv = _CONV[spatial_dims](
            in_ch, out_ch,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups,
                             spatial_dims=spatial_dims)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def build_stem(
    mode: str,
    in_ch: int,
    out_ch: int,
    norm_type: str = "instance",
    norm_groups: int = 8,
    activation: str = "leakyrelu",
    spatial_dims: int = 3,
) -> Tuple[nn.Module, int]:
    """Construct a stem; returns ``(stem_module, stem_stride)`` (1 for conv3/7/dual, N for patchN)."""
    if mode not in STEM_MODES:
        raise ValueError(f"Unknown stem mode: {mode!r}. Valid: {STEM_MODES}")

    if mode == "conv3":
        stem = ConvNormAct(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        return stem, 1

    if mode == "conv7":
        stem = ConvNormAct(
            in_ch, out_ch, kernel_size=7, stride=1, padding=3,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        return stem, 1

    if mode == "dual":
        stem = DualConvStem(
            in_ch, out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        return stem, 1

    # patch-embed variants (default to GELU when caller leaves activation at the leakyrelu default).
    patch_size = 2 if mode == "patch2" else 4
    stem = PatchEmbedStem(
        in_ch, out_ch, patch_size=patch_size,
        norm_type=norm_type, norm_groups=norm_groups,
        activation="gelu" if activation == "leakyrelu" else activation,
        spatial_dims=spatial_dims)
    return stem, patch_size


# ---------------------------------------------------------------------------
# Multi-FOV context fusion stems (2.5D multi-z-FOV mode)
# ---------------------------------------------------------------------------
CONTEXT_FUSION_MODES = ("shared_stem", "multi_stem_proj", "hierarchical")


class MultiStemProj(nn.Module):
    """``n_views`` independent stems → channel-concat → 1×1 fusion to ``out_ch``.

    Each view (channel slab) gets its own stem so FOV-specific filters can be learned;
    the 1×1 fusion keeps the encoder channel contract identical to a single-stem build.
    """

    def __init__(
        self,
        mode: str,
        n_views: int,
        in_ch_per_view: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        spatial_dims: int = 3,
        in_ch_per_view_list: List[int] = None,
    ):
        """Channel layout: uniform ``in_ch_per_view`` (default) or per-view
        ``in_ch_per_view_list`` (length must equal ``n_views``; takes precedence).
        """
        super().__init__()
        if n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        self.n_views = n_views
        if in_ch_per_view_list is not None:
            if len(in_ch_per_view_list) != n_views:
                raise ValueError(
                    f"in_ch_per_view_list length ({len(in_ch_per_view_list)}) "
                    f"must equal n_views ({n_views})")
            self.in_ch_per_view_list: List[int] = [int(c) for c in in_ch_per_view_list]
        else:
            self.in_ch_per_view_list = [int(in_ch_per_view)] * n_views
        # Back-compat shim: first view's count (use in_ch_per_view_list for full info).
        self.in_ch_per_view = self.in_ch_per_view_list[0]

        stems: List[nn.Module] = []
        strides: List[int] = []
        for c_v in self.in_ch_per_view_list:
            s, stride = build_stem(
                mode, c_v, out_ch,
                norm_type=norm_type, norm_groups=norm_groups,
                activation=activation, spatial_dims=spatial_dims)
            stems.append(s)
            strides.append(stride)
        if len(set(strides)) != 1:
            # Sub-stems share mode → stride must agree; defensive guard.
            raise RuntimeError(
                f"MultiStemProj sub-stems disagree on stride: {strides}")
        self.stems = nn.ModuleList(stems)
        self.stem_stride = strides[0]

        # 1×1 fusion back to out_ch keeps downstream channel contract intact.
        self.proj = ConvNormAct(
            n_views * out_ch, out_ch,
            kernel_size=1, stride=1, padding=0,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, sum(in_ch_per_view_list), *spatial)``."""
        expected_c = sum(self.in_ch_per_view_list)
        if x.shape[1] != expected_c:
            raise ValueError(
                f"MultiStemProj expects {expected_c} input channels "
                f"(per-view={self.in_ch_per_view_list}); got {x.shape[1]}")
        # Per-view channel split (zero-copy views).
        chunks = torch.split(x, self.in_ch_per_view_list, dim=1)
        feats = [stem(c) for stem, c in zip(self.stems, chunks)]
        return self.proj(torch.cat(feats, dim=1))


class HierarchicalStems(nn.Module):
    """Per-FOV stems with stage-aligned strides; encoder fuses aux features per level.

    View 0 → ``main_stem`` at native stride ``s0``. View ``k`` (k≥1) → ``aux_stems[k-1]``
    (``PatchEmbedStem``, stride ``s0 * 2^k``) with output channels =
    ``stage_channels[k-1]`` so that the encoder's per-level cat-fusion is
    ``2 * stage_channels[k-1] → stage_channels[k-1]``.

    No combined ``forward``: encoder calls ``forward_main`` / ``forward_aux``
    separately to interleave aux features with stage-level downsampling.
    """

    def __init__(
        self,
        mode: str,
        n_views: int,
        in_ch_per_view: int,
        stage_channels: List[int],
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        spatial_dims: int = 3,
        aux_channels: List[int] = None,
        in_ch_per_view_list: List[int] = None,
    ):
        """Channel layout same as :class:`MultiStemProj`: uniform or per-view list."""
        super().__init__()
        if n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        n_aux = n_views - 1
        if n_aux > 0 and len(stage_channels) <= n_aux:
            raise ValueError(
                f"hierarchical fusion requires at least n_views={n_views} "
                f"encoder stages (one stage per aux injection level + the "
                f"main path); got {len(stage_channels)} stages.")
        self.n_views = n_views
        if in_ch_per_view_list is not None:
            if len(in_ch_per_view_list) != n_views:
                raise ValueError(
                    f"in_ch_per_view_list length ({len(in_ch_per_view_list)}) "
                    f"must equal n_views ({n_views})")
            self.in_ch_per_view_list: List[int] = [int(c) for c in in_ch_per_view_list]
        else:
            self.in_ch_per_view_list = [int(in_ch_per_view)] * n_views
        # Back-compat shim — first view's count.
        self.in_ch_per_view = self.in_ch_per_view_list[0]

        # Main stem (view 0): user's stem mode at native stride.
        self.main_stem, self.stem_stride = build_stem(
            mode, self.in_ch_per_view_list[0], stage_channels[0],
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

        # Aux stems: stride = main_stride * 2^k; out_ch defaults to stage_channels[k-1].
        if aux_channels is None:
            aux_channels = [stage_channels[k - 1] for k in range(1, n_views)]
        if len(aux_channels) != n_aux:
            raise ValueError(
                f"aux_channels length ({len(aux_channels)}) must equal "
                f"n_views - 1 ({n_aux})")

        self.aux_stems = nn.ModuleList()
        self.aux_levels: List[int] = []
        self.aux_strides: List[int] = []
        self.aux_out_channels: List[int] = list(aux_channels)
        for k in range(1, n_views):
            stride = self.stem_stride * (2 ** k)
            self.aux_levels.append(k)
            self.aux_strides.append(stride)
            self.aux_stems.append(
                PatchEmbedStem(
                    in_ch=self.in_ch_per_view_list[k],
                    out_ch=aux_channels[k - 1],
                    patch_size=stride,
                    norm_type=norm_type, norm_groups=norm_groups,
                    activation=("gelu" if activation == "leakyrelu"
                                else activation),
                    spatial_dims=spatial_dims))

    def split_views(self, x: torch.Tensor) -> List[torch.Tensor]:
        expected_c = sum(self.in_ch_per_view_list)
        if x.shape[1] != expected_c:
            raise ValueError(
                f"HierarchicalStems expects {expected_c} input channels "
                f"(per-view={self.in_ch_per_view_list}); got {x.shape[1]}")
        return list(torch.split(x, self.in_ch_per_view_list, dim=1))

    def forward_main(self, x_view0: torch.Tensor) -> torch.Tensor:
        return self.main_stem(x_view0)

    def forward_aux(
        self, chunks: List[torch.Tensor],
    ) -> "OrderedDict[int, torch.Tensor]":
        """Run each aux stem on its view chunk; returns ordered ``{level: aux_feature}``."""
        from collections import OrderedDict
        out: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        for k, stem in enumerate(self.aux_stems):
            level = self.aux_levels[k]
            out[level] = stem(chunks[k + 1])  # +1 to skip view 0
        return out


def build_context_stem(
    mode: str,
    fusion: str,
    n_views: int,
    in_ch_per_view: int,
    out_ch: int,
    norm_type: str = "instance",
    norm_groups: int = 8,
    activation: str = "leakyrelu",
    spatial_dims: int = 3,
    stage_channels: List[int] = None,
    in_ch_per_view_list: List[int] = None,
) -> Tuple[nn.Module, int]:
    """Dispatch the 2.5D multi-FOV context stem; returns ``(module, stem_stride)``.

    - ``n_views==1`` or ``shared_stem``: single stem over all channels (legacy path).
    - ``multi_stem_proj``: per-view stems + 1×1 fusion.
    - ``hierarchical``: needs ``stage_channels``; encoder cat-fuses per level.
    """
    if fusion not in CONTEXT_FUSION_MODES:
        raise ValueError(
            f"Unknown context_fusion: {fusion!r}. Valid: {CONTEXT_FUSION_MODES}")
    # Validate the per-view-list / uniform layouts agree on total channel count.
    if in_ch_per_view_list is not None and len(in_ch_per_view_list) != n_views:
        raise ValueError(
            f"in_ch_per_view_list length ({len(in_ch_per_view_list)}) "
            f"must equal n_views ({n_views})")
    if n_views == 1 or fusion == "shared_stem":
        # Total input channel count: per-view list when given, else uniform.
        total_in = (sum(in_ch_per_view_list)
                    if in_ch_per_view_list is not None
                    else n_views * in_ch_per_view)
        return build_stem(
            mode, total_in, out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
    if fusion == "multi_stem_proj":
        msp = MultiStemProj(
            mode=mode, n_views=n_views,
            in_ch_per_view=in_ch_per_view, out_ch=out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims,
            in_ch_per_view_list=in_ch_per_view_list)
        return msp, msp.stem_stride
    # hierarchical
    if stage_channels is None:
        raise ValueError(
            "hierarchical fusion requires stage_channels (encoder channel "
            "list) so aux stems can size their output channels to match "
            "each injection level. Pass stage_channels=encoder_channels.")
    if stage_channels[0] != out_ch:
        # Contract: out_ch must equal stage_channels[0] (main-stem output).
        raise ValueError(
            f"hierarchical fusion: out_ch ({out_ch}) must equal "
            f"stage_channels[0] ({stage_channels[0]}).")
    hier = HierarchicalStems(
        mode=mode, n_views=n_views,
        in_ch_per_view=in_ch_per_view,
        stage_channels=stage_channels,
        norm_type=norm_type, norm_groups=norm_groups,
        activation=activation, spatial_dims=spatial_dims,
        in_ch_per_view_list=in_ch_per_view_list)
    return hier, hier.stem_stride
