"""Stem / patch-embed builders for 3D UNet.

Plan C — hierarchical multi-FOV injection
-----------------------------------------
``"hierarchical"`` adds a third fusion strategy on top of ``shared_stem``
and ``multi_stem_proj``. View 0 (true 1× FOV) drives the **main stem** at
its native stride. Each auxiliary view ``k = 1..n_views-1`` is consumed
by an independent **patchify stem of stride ``main_stem_stride * 2^k``**,
producing a low-resolution feature map whose spatial size matches the
encoder feature map at the **entrance of stage ``k``** (i.e., right after
``Downsample`` ``k``). The encoder concatenates the aux feature with the
main feature at that injection point and runs a 1×1 ``ConvNormAct``
fusion to compress back to the stage's expected channel count — keeping
the rest of the encoder / decoder / skip-connection contract bit-exact.

Why hierarchical: coarse-FOV context naturally aligns with deeper, lower-
resolution semantic features (HRNet / PSPNet / nnFormer multi-scale
context aggregation). Plan A fuses everything at the input resolution
and lets the network compress wider FOVs by force; Plan C lets each FOV
land at the matching semantic level by construction.


Provides the initial feature-extraction layer applied to raw input volumes.
The stem determines the resolution at which the encoder operates:

- "conv3"  : classical 3×3×3 stride-1 conv (default; preserves resolution).
- "conv7"  : ConvNeXt-style large-kernel 7×7×7 stride-1 conv (larger RF).
- "dual"   : nnU-Net-style two stacked 3×3×3 stride-1 convs.
- "patch2" : 2×2×2 stride-2 patch embedding (halves resolution).
- "patch4" : 4×4×4 stride-4 patch embedding (Swin / ConvNeXt standard).

Patch-embed stems (`patchN`) reduce spatial resolution by N, meaning the
encoder produces features starting at (input / N).  The UNet wrapper
(see `UNet3D`) restores the original resolution with a final learned
upsample applied only to the main segmentation output.

Multi-FOV context fusion (2.5D mode): when ``data.multi_res_scales`` has
more than one z-FOV in 2.5D mode, the model input is laid out as
``(B, n_views * D, H, W)`` — view 0 is the 1× FOV (D real slices), views
1..K are wider z-FOVs each resampled back to D channels. Two fusion
strategies are exposed:

- "shared_stem"     : feed all ``n_views * D`` channels into ONE stem.
                      Cheapest, but the stem must learn a single filter
                      bank that works on physically heterogeneous channels
                      (real slices vs. resampled "virtual" slices).
- "multi_stem_proj" : ``n_views`` independent stems (each consumes D
                      channels) → cat → 1×1 ConvNormAct fusion back to
                      ``encoder_channels[0]``. Strictly more expressive
                      than shared_stem at negligible param cost (stems are
                      a small fraction of the network). RECOMMENDED.
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
    """Patch-embedding stem: stride-N conv + norm + activation.

    Resolution is reduced by a factor of ``patch_size`` along every spatial
    axis.  Inspired by Swin Transformer and ConvNeXt patchify stems.
    """

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
    """Construct a stem module.

    Returns:
        (stem_module, stem_stride): ``stem_stride`` is the spatial
        downsampling factor introduced by the stem (1 for stride-1 stems,
        2 or 4 for patch-embed stems).  Callers use this to decide whether
        a matching final-upsample is required downstream.
    """
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

    # patch-embed variants
    patch_size = 2 if mode == "patch2" else 4
    # Patch-embed stems typically use GELU; expose activation for flexibility.
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
    """``n_views`` independent stems → channel-concat → 1×1 fusion.

    Designed for the 2.5D multi-FOV setup where the input tensor is laid
    out as ``(B, n_views * in_ch_per_view, *spatial)`` with view ``i``
    occupying channel slab ``[i * C, (i+1) * C)``. Each view goes through
    its own stem (same ``mode`` / ``out_ch`` / hyper-params, independent
    weights) so the network can learn FOV-specific low-level filters
    instead of forcing a single shared filter bank to cover physically
    heterogeneous inputs (raw slices vs. resampled wide-FOV slices).

    The ``n_views`` stem outputs are concatenated on the channel axis,
    yielding ``(B, n_views * out_ch, *spatial')`` (``spatial'`` = spatial
    after each stem's stride). A 1×1 ``ConvNormAct`` projects them back
    to ``out_ch`` so the rest of the encoder is contract-identical to a
    single-stem build (no downstream channel-count surgery required).

    All sub-stems share the same ``stem_stride`` (they're built from the
    same ``mode``); ``MultiStemProj.stem_stride`` exposes that value so
    the wrapping ``Encoder`` / ``UNet3D`` can keep its existing logic.
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
    ):
        super().__init__()
        if n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        self.n_views = n_views
        self.in_ch_per_view = in_ch_per_view

        stems: List[nn.Module] = []
        strides: List[int] = []
        for _ in range(n_views):
            s, stride = build_stem(
                mode, in_ch_per_view, out_ch,
                norm_type=norm_type, norm_groups=norm_groups,
                activation=activation, spatial_dims=spatial_dims)
            stems.append(s)
            strides.append(stride)
        if len(set(strides)) != 1:
            # Defensive: all stems are built from the same mode so this
            # invariant should hold by construction.
            raise RuntimeError(
                f"MultiStemProj sub-stems disagree on stride: {strides}")
        self.stems = nn.ModuleList(stems)
        self.stem_stride = strides[0]

        # 1×1 fusion: cheap (1×1 conv on small spatial), keeps decoder /
        # downsample channel contract intact (output channels = out_ch).
        self.proj = ConvNormAct(
            n_views * out_ch, out_ch,
            kernel_size=1, stride=1, padding=0,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, n_views * in_ch_per_view, *spatial)."""
        expected_c = self.n_views * self.in_ch_per_view
        if x.shape[1] != expected_c:
            raise ValueError(
                f"MultiStemProj expects {expected_c} input channels "
                f"(n_views={self.n_views} * in_ch_per_view="
                f"{self.in_ch_per_view}); got {x.shape[1]}")
        # Channel-wise split into n_views chunks, each (B, in_ch_per_view, *).
        chunks = torch.split(x, self.in_ch_per_view, dim=1)
        feats = [stem(c) for stem, c in zip(self.stems, chunks)]
        return self.proj(torch.cat(feats, dim=1))


class HierarchicalStems(nn.Module):
    """Plan C: per-FOV stems with stage-aligned strides for hierarchical injection.

    Layout
    ------
    Input ``(B, n_views * in_ch_per_view, *spatial)``.
      - View 0 (``chunks[0]``) → ``main_stem`` with stride ``s0`` =
        ``build_stem(mode)`` native stride. Output spatial = ``spatial / s0``.
      - View ``k`` for ``k = 1..n_views-1`` → ``aux_stems[k-1]``:
        ``PatchEmbedStem`` with patch_size = ``s0 * 2^k``. Output spatial =
        ``spatial / (s0 * 2^k)`` — matches the encoder feature map at the
        entrance of stage ``k`` (after ``Downsample`` ``k``).

    The encoder is responsible for cat-fusing each aux feature into the
    main path at its registered injection level (``aux_levels[k-1] = k``)
    via a 1×1 ``ConvNormAct``. This module only owns the stems and does
    not perform fusion itself — keeping responsibilities separated and
    letting the encoder size the per-level fusion conv from
    ``aux_out_channels``.

    Aux stem channel choice
    -----------------------
    Output channels of aux stem ``k`` default to ``stage_channels[k - 1]``
    — the channel count at the injection point (post-downsample = pre-stage).
    This makes the cat fusion exactly ``2 * stage_channels[k - 1]`` →
    ``stage_channels[k - 1]``, a clean uniform pattern. Callers may pass
    ``aux_channels`` explicitly to override.

    Notes
    -----
    - With ``n_views == 1`` no aux stems are built; the module behaves
      like a thin wrapper around the main stem (the dispatcher routes
      this case to the single-stem path instead, but the constructor
      handles it for robustness).
    - ``forward`` is intentionally NOT implemented as a single op — the
      encoder calls ``forward_main`` and ``forward_aux`` separately so
      it can interleave aux features with stage-level downsampling.
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
    ):
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
        self.in_ch_per_view = in_ch_per_view

        # Main stem — uses the user's chosen stem mode at native stride.
        self.main_stem, self.stem_stride = build_stem(
            mode, in_ch_per_view, stage_channels[0],
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

        # Aux stems: stride = main_stem_stride * 2^k for k = 1..n_aux.
        # Output channels default to stage_channels[k - 1] (the channel
        # count at the injection point, post-Downsample-k, pre-stage-k).
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
                    in_ch=in_ch_per_view,
                    out_ch=aux_channels[k - 1],
                    patch_size=stride,
                    norm_type=norm_type, norm_groups=norm_groups,
                    activation=("gelu" if activation == "leakyrelu"
                                else activation),
                    spatial_dims=spatial_dims))

    def split_views(self, x: torch.Tensor) -> List[torch.Tensor]:
        expected_c = self.n_views * self.in_ch_per_view
        if x.shape[1] != expected_c:
            raise ValueError(
                f"HierarchicalStems expects {expected_c} input channels "
                f"(n_views={self.n_views} * in_ch_per_view="
                f"{self.in_ch_per_view}); got {x.shape[1]}")
        return list(torch.split(x, self.in_ch_per_view, dim=1))

    def forward_main(self, x_view0: torch.Tensor) -> torch.Tensor:
        return self.main_stem(x_view0)

    def forward_aux(
        self, chunks: List[torch.Tensor],
    ) -> "OrderedDict[int, torch.Tensor]":
        """Run each aux stem on its corresponding view chunk.

        Returns an ordered mapping ``level -> aux_feature``. The encoder
        looks up by level inside the stage loop.
        """
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
) -> Tuple[nn.Module, int]:
    """Build the context-fusion stem for 2.5D multi-FOV mode.

    Dispatch:
      - ``n_views == 1`` OR ``fusion == "shared_stem"``: standard
        ``build_stem`` over ``n_views * in_ch_per_view`` input channels.
        With ``n_views == 1`` this is bit-identical to the single-FOV
        legacy path (zero behaviour change).
      - ``fusion == "multi_stem_proj"``: ``MultiStemProj`` with one
        ``build_stem`` per view + 1×1 fusion back to ``out_ch``.
      - ``fusion == "hierarchical"``: ``HierarchicalStems`` — main stem
        for view 0 + per-aux-view patchify stems with stride-aligned
        outputs. Requires ``stage_channels`` so aux output channels can
        match the injection points. The encoder is responsible for the
        per-level cat-fusion 1×1 conv (see ``Encoder.__init__``).

    Returns ``(module, stem_stride)`` matching the ``build_stem`` ABI.
    """
    if fusion not in CONTEXT_FUSION_MODES:
        raise ValueError(
            f"Unknown context_fusion: {fusion!r}. Valid: {CONTEXT_FUSION_MODES}")
    if n_views == 1 or fusion == "shared_stem":
        return build_stem(
            mode, n_views * in_ch_per_view, out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
    if fusion == "multi_stem_proj":
        msp = MultiStemProj(
            mode=mode, n_views=n_views,
            in_ch_per_view=in_ch_per_view, out_ch=out_ch,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        return msp, msp.stem_stride
    # hierarchical
    if stage_channels is None:
        raise ValueError(
            "hierarchical fusion requires stage_channels (encoder channel "
            "list) so aux stems can size their output channels to match "
            "each injection level. Pass stage_channels=encoder_channels.")
    if stage_channels[0] != out_ch:
        # The dispatcher's contract is that out_ch == stage_channels[0]
        # (the main-stem output). Fail fast on misalignment.
        raise ValueError(
            f"hierarchical fusion: out_ch ({out_ch}) must equal "
            f"stage_channels[0] ({stage_channels[0]}).")
    hier = HierarchicalStems(
        mode=mode, n_views=n_views,
        in_ch_per_view=in_ch_per_view,
        stage_channels=stage_channels,
        norm_type=norm_type, norm_groups=norm_groups,
        activation=activation, spatial_dims=spatial_dims)
    return hier, hier.stem_stride
