"""Generic 3D UNet architecture.

Structure:
  Input → Stem → [Encoder levels with Downsample] → Bottleneck
                      ↓ (skip connections)
  [Decoder levels with Upsample + skip fusion] → Segmentation Head

The encoder and decoder are symmetric in channel count.
Backbone blocks (ResNet / ConvNeXt) are injected via the factory.

Supports:
- Deep supervision (multi-scale outputs during training)
- Configurable skip connection mode (concatenate or add)
- Per-class independent sigmoid output
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import (
    _CONV, INTERP_SMOOTH,
    AttentionGate3D, ConvNormAct, Downsample, Upsample, get_norm)
from .stem import HierarchicalStems, build_context_stem, build_stem


class Encoder(nn.Module):
    """UNet encoder: stem + N stages with downsampling between them.

    Produces multi-scale features [level_0, level_1, ..., bottleneck].
    level_0 is at the highest resolution.
    """

    def __init__(
        self,
        in_channels: int,
        stage_channels: List[int],
        stage_builder,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        downsample_mode: str = "conv",
        stem_mode: str = "conv3",
        spatial_dims: int = 3,
        context_n_views: int = 1,
        context_fusion: str = "shared_stem",
        in_ch_per_view_list: List[int] = None,
        downsample_builder: Optional[Callable[[int, int], nn.Module]] = None):
        super().__init__()
        self.spatial_dims = spatial_dims
        # Stem: project input to first channel count. The stem may introduce
        # a spatial stride (``patch2``/``patch4``) — ``stem_stride`` is
        # preserved as a property so the wrapping UNet can add a matching
        # final upsample when necessary.
        #
        # Multi-FOV context fusion (2.5D mode): when ``context_n_views > 1``
        # the input layout is ``(B, sum_k in_ch_per_view_list[k], *spatial)``.
        # Two channel layouts are supported by the underlying
        # ``build_context_stem``:
        #   - Uniform (default): ``in_ch_per_view = in_channels // context_n_views``
        #     — the legacy / OFF path with all views having equal D channels.
        #   - Per-view list (``in_ch_per_view_list`` provided): each view
        #     contributes ``in_ch_per_view_list[k]`` channels (native-depth
        #     ON path with ``D_k = round(D * s_k)``). Length must equal
        #     ``context_n_views`` and ``sum(...)`` must equal ``in_channels``.
        if context_n_views < 1:
            raise ValueError(
                f"context_n_views must be >= 1, got {context_n_views}")
        if in_ch_per_view_list is not None:
            if len(in_ch_per_view_list) != context_n_views:
                raise ValueError(
                    f"in_ch_per_view_list length "
                    f"({len(in_ch_per_view_list)}) must equal "
                    f"context_n_views ({context_n_views})")
            if sum(in_ch_per_view_list) != in_channels:
                raise ValueError(
                    f"sum(in_ch_per_view_list)={sum(in_ch_per_view_list)} "
                    f"must equal in_channels ({in_channels})")
            in_ch_per_view = int(in_ch_per_view_list[0])
        else:
            if in_channels % context_n_views != 0:
                raise ValueError(
                    f"in_channels ({in_channels}) must be divisible by "
                    f"context_n_views ({context_n_views})")
            in_ch_per_view = in_channels // context_n_views
        self.context_n_views = context_n_views
        # Persist on the encoder so UNet3D can mirror the stem topology
        # when building aux seg heads (Plan A vs. Plan C symmetric layout).
        self.context_fusion = context_fusion
        self.in_ch_per_view_list: List[int] = (
            list(in_ch_per_view_list) if in_ch_per_view_list is not None
            else [in_ch_per_view] * context_n_views)
        self.stem, self.stem_stride = build_context_stem(
            mode=stem_mode, fusion=context_fusion,
            n_views=context_n_views,
            in_ch_per_view=in_ch_per_view,
            out_ch=stage_channels[0],
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims,
            stage_channels=stage_channels,
            in_ch_per_view_list=in_ch_per_view_list)

        # Encoder stages and downsampling
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, ch in enumerate(stage_channels):
            in_ch = stage_channels[i - 1] if i > 0 else stage_channels[0]
            self.stages.append(stage_builder(in_ch, ch))
            if i > 0:
                # Inter-stage downsample. Layout convention: in_ch == out_ch
                # (channel growth is handled inside the next stage's first
                # block). ``downsample_builder``, when provided by the
                # factory, overrides the generic ``Downsample`` so backbone-
                # specific topologies (e.g. ConvNeXt's LN-first
                # ``LayerNorm → Conv(s=2)``) can be injected without
                # polluting the generic Downsample contract.
                ds_in = stage_channels[i - 1]
                ds_out = stage_channels[i - 1]
                if downsample_builder is not None:
                    self.downsamples.append(downsample_builder(ds_in, ds_out))
                else:
                    self.downsamples.append(
                        Downsample(
                            ds_in, ds_out,
                            norm_type=norm_type, norm_groups=norm_groups,
                            mode=downsample_mode, spatial_dims=spatial_dims))

        # ----- Plan C: per-injection-level cat-fusion 1×1 ConvNormAct ----
        # Built only when the stem is ``HierarchicalStems``. Each fuse
        # consumes ``cat(main_post_downsample, aux_feat)`` at level k
        # (channel count ``stage_channels[k-1] + aux_out_channels[k-1]``)
        # and projects back to ``stage_channels[k-1]`` so the downstream
        # stage block sees its expected input channel count — keeping
        # the encoder/decoder/skip contract bit-identical to the non-
        # hierarchical paths. ``ModuleDict`` is keyed by ``str(level)``.
        self.aux_fuse = nn.ModuleDict()
        if isinstance(self.stem, HierarchicalStems):
            hs = self.stem
            for idx, level in enumerate(hs.aux_levels):
                main_ch = stage_channels[level - 1]
                aux_ch = hs.aux_out_channels[idx]
                self.aux_fuse[str(level)] = ConvNormAct(
                    main_ch + aux_ch, main_ch,
                    kernel_size=1, stride=1, padding=0,
                    norm_type=norm_type, norm_groups=norm_groups,
                    activation=activation, spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns features from each level: [level_0, level_1, ..., level_N].

        Plan C (hierarchical fusion): when ``self.stem`` is
        ``HierarchicalStems``, the input is split per view; view 0 drives
        the main stem and aux views are stem-encoded into low-resolution
        features that get cat-fused into the main path immediately after
        the matching downsample step.
        """
        if isinstance(self.stem, HierarchicalStems):
            chunks = self.stem.split_views(x)
            x = self.stem.forward_main(chunks[0])
            aux_feats = self.stem.forward_aux(chunks)
        else:
            x = self.stem(x)
            aux_feats = {}

        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
                if i in aux_feats:
                    aux = aux_feats[i]
                    if aux.shape[2:] != x.shape[2:]:
                        # Defensive: aux stem strides are computed to
                        # match exactly. A mismatch implies a config /
                        # patch_size combination that misaligns spatial
                        # dims — fail fast with a precise diagnostic
                        # rather than silently breaking via interpolate.
                        raise RuntimeError(
                            f"Plan C aux feature spatial mismatch at "
                            f"level {i}: main={tuple(x.shape[2:])}, "
                            f"aux={tuple(aux.shape[2:])}. Check that "
                            f"input spatial dims are divisible by the "
                            f"aux stem stride.")
                    x = self.aux_fuse[str(i)](torch.cat([x, aux], dim=1))
            x = stage(x)
            features.append(x)
        return features


class DecoderLevel(nn.Module):
    """Single decoder level: upsample + (optional attention-gated) skip
    fusion + stage blocks.

    Args:
        skip_attention: if True, the skip feature is re-weighted by an
            ``AttentionGate3D`` driven by the upsampled decoder feature
            (Oktay et al., "Attention U-Net", MIDL 2018).
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        stage_builder,
        upsample_mode: str = "transpose",
        skip_mode: str = "cat",
        skip_attention: bool = False,
        spatial_dims: int = 3):
        super().__init__()
        self.skip_mode = skip_mode
        self.spatial_dims = spatial_dims
        self.upsample  = Upsample(in_ch, out_ch, mode=upsample_mode,
                                  spatial_dims=spatial_dims)

        if skip_mode == "cat":
            fused_ch = out_ch + skip_ch
        else:  # add
            # Project skip to match out_ch if needed
            self.skip_proj = (
                _CONV[spatial_dims](skip_ch, out_ch, 1, bias=False)
                if skip_ch != out_ch else nn.Identity())
            fused_ch = out_ch

        self.attn_gate = (
            AttentionGate3D(x_ch=skip_ch, g_ch=out_ch,
                            spatial_dims=spatial_dims)
            if skip_attention else None
        )
        self.stage = stage_builder(fused_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch (can happen due to odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = _match_size(x, skip.shape[2:], self.spatial_dims)

        if self.attn_gate is not None:
            # Gate skip using the upsampled decoder feature as the gating
            # signal (shapes now match after the resize above).
            skip = self.attn_gate(skip, x)

        if self.skip_mode == "cat":
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + self.skip_proj(skip)

        return self.stage(x)


class Decoder(nn.Module):
    """UNet decoder: N levels of upsample + skip fusion + blocks.

    Takes encoder features [level_0, ..., bottleneck] and produces
    decoder features [dec_low_res, ..., dec_high_res].
    """

    def __init__(
        self,
        encoder_channels: List[int],
        stage_builder,
        upsample_mode: str = "transpose",
        skip_mode: str = "cat",
        skip_attention: bool = False,
        spatial_dims: int = 3):
        super().__init__()
        self.levels = nn.ModuleList()
        self.spatial_dims = spatial_dims
        n = len(encoder_channels)

        # Decoder levels: from deepest to shallowest
        # Level i connects encoder[n-2-i] (skip) with previous decoder output
        for i in range(n - 1):
            in_ch = encoder_channels[n - 1 - i]  # from deeper level
            skip_ch = encoder_channels[n - 2 - i]  # skip connection
            out_ch = encoder_channels[n - 2 - i]   # symmetric output

            self.levels.append(
                DecoderLevel(in_ch, skip_ch, out_ch, stage_builder,
                             upsample_mode=upsample_mode,
                             skip_mode=skip_mode,
                             skip_attention=skip_attention,
                             spatial_dims=spatial_dims))

        # Output channels at each decoder level (low-res → high-res)
        self.out_channels = [encoder_channels[n - 2 - i] for i in range(n - 1)]

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode features.

        Args:
            encoder_features: [level_0, level_1, ..., bottleneck]

        Returns:
            Decoder features [dec_low_res, ..., dec_high_res]
        """
        x = encoder_features[-1]  # bottleneck
        outputs = []
        for i, level in enumerate(self.levels):
            skip_idx = len(encoder_features) - 2 - i
            x = level(x, encoder_features[skip_idx])
            outputs.append(x)
        return outputs


class SegmentationHead(nn.Module):
    """1×1(×1) convolution to produce per-class logits."""

    def __init__(self, in_ch: int, num_classes: int,
                 spatial_dims: int = 3):
        super().__init__()
        self.conv = _CONV[spatial_dims](in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvSegmentationHead(nn.Module):
    """3×3 ConvNormAct → 1×1 logits.

    Drop-in replacement for :class:`SegmentationHead` that adds a single
    ConvNormAct stage of capacity matching the decoder's per-level
    operators. Designed for Plan C aux heads which read a low-resolution
    decoder feature: the extra 3×3 conv lets the head re-aggregate
    spatial context before the linear classifier — closer to the
    "main head + 1 decoder block" capacity of the main path.

    Param overhead vs. the linear head is ``9 * in_ch^2`` (3D: 27) plus
    norm/bias — at typical decoder widths (32–256) this is well under
    1% of the total network.
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        spatial_dims: int = 3,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
    ):
        super().__init__()
        self.conv = ConvNormAct(
            in_ch, in_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)
        self.classifier = _CONV[spatial_dims](in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(x))


def _build_aux_head(
    mode: str,
    in_ch: int,
    num_classes: int,
    spatial_dims: int,
    norm_type: str = "instance",
    norm_groups: int = 8,
    activation: str = "leakyrelu",
) -> nn.Module:
    """Dispatch the aux seg head topology.

    ``mode`` is one of:
      - ``"linear"`` → :class:`SegmentationHead` (1×1).
      - ``"conv"``   → :class:`ConvSegmentationHead` (3×3 + 1×1).

    See ``ModelConfig.aux_head_mode`` for the rationale per fusion mode.
    """
    if mode == "linear":
        return SegmentationHead(in_ch, num_classes, spatial_dims=spatial_dims)
    if mode == "conv":
        return ConvSegmentationHead(
            in_ch, num_classes, spatial_dims=spatial_dims,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation)
    raise ValueError(
        f"Unknown aux_head_mode: {mode!r}. Valid: 'linear' | 'conv'.")


class UNet3D(nn.Module):
    """Generic 3D UNet with pluggable encoder/decoder stages.

    Args:
        encoder: Encoder module.
        decoder: Decoder module.
        num_fg_classes: Number of foreground classes (output channels).
        deep_supervision: Output at multiple decoder levels during training.
        aux_seg_supervision: Build per-aux-FOV seg heads symmetric to the
            encoder's stem fusion topology. Active only when
            ``encoder.context_n_views > 1`` (2.5D multi-FOV mode); silently
            ignored otherwise (no extra parameters built).

    Forward returns:
        - eval mode (``self.training=False``)::
              tensor (B, num_fg, *spatial)            — single scale.
              list  [main_out, 2nd_high, ..., lowest] — when DS heads
                                                          existed at construction
                                                          (kept for backward
                                                          compatibility).
        - training mode without aux supervision: same as above.
        - training mode with aux supervision active::
              dict { "main": tensor | list,                 — main path
                                                              (DS list when
                                                              deep_supervision=True,
                                                              tensor otherwise)
                     "aux":  [aux_view_1, aux_view_2, ...]} — one tensor per
                                                              aux view at the
                                                              SAME (H, W) as
                                                              main_out.
        Predictor / val paths only exercise eval mode → contract preserved.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder,
        num_fg_classes: int,
        deep_supervision: bool = False,
        spatial_dims: int = 3,
        aux_seg_supervision: bool = False,
        aux_head_mode: str = "linear",
        # Norm / activation are propagated to ``ConvSegmentationHead`` when
        # ``aux_head_mode == "conv"``; mirror the encoder defaults so the
        # aux head's norm/act stay homogeneous with the rest of the model.
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        aux_head_out_channels: List[int] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_fg_classes   = num_fg_classes
        self.deep_supervision = deep_supervision
        self.spatial_dims = spatial_dims

        # Main segmentation head (highest resolution decoder output). If the
        # encoder uses a patch-embed stem (stride > 1), the decoder's highest
        # resolution is input/stem_stride, so the main output must be
        # upsampled back to the original input resolution. DS heads remain
        # at their native decoder resolutions — DeepSupervisionLoss already
        # downsamples the target to match.
        self.stem_stride = getattr(encoder, "stem_stride", 1)
        self.seg_head = SegmentationHead(
            decoder.out_channels[-1], num_fg_classes,
            spatial_dims=spatial_dims)

        # Deep supervision heads (lower-resolution outputs).
        # decoder.out_channels is [low, ..., high]; we want DS outputs ordered
        # from 2nd-highest to lowest resolution so that forward() can return
        # [main_out, 2nd_high, 3rd_high, ..., lowest] — matching the
        # DeepSupervisionLoss convention weights[0]=highest-res.
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            # Reverse [low..2nd-high] → [2nd-high..low]
            for ch in reversed(decoder.out_channels[:-1]):
                self.ds_heads.append(SegmentationHead(
                    ch, num_fg_classes, spatial_dims=spatial_dims))

        # ----- Multi-FOV auxiliary segmentation heads (2.5D mode) ---------
        # Mirror the encoder's stem fusion topology so each aux supervision
        # signal lands at the matching depth:
        #
        #   shared_stem / multi_stem_proj (Plan A — early fusion at full res):
        #     all views funnel into ONE encoder pyramid, so aux heads sit
        #     in PARALLEL on the highest-resolution decoder feature
        #     (``dec_features[-1]``). Each head has its own 1×1 conv but
        #     shares the upstream representation; the loss forces the shared
        #     trunk to retain enough cross-view information that every view
        #     can be reconstructed by a per-view classifier. This matches
        #     standard multi-task heads on a shared backbone.
        #
        #   hierarchical (Plan C — per-FOV stems at stride-aligned strides):
        #     aux view k is injected at encoder stage k. The symmetric
        #     decoder feature at the same semantic depth is ``dec_features
        #     [-1-k]`` (decoder features run [low_res, ..., high_res], so
        #     index -1 is the highest-res mirroring stage 0, -1-k mirrors
        #     stage k). Aux head k reads ``dec_features[-1-k]``, applies a
        #     1×1 conv, then is interpolated back to (H, W) so the loss
        #     can be computed against view k's resampled D-slice label
        #     (which lives at full (H, W) per the dataset contract).
        #
        # When ``encoder.context_n_views == 1`` (single FOV) or
        # ``aux_seg_supervision == False`` the construction below is a
        # no-op and the path is bit-identical to the legacy build.
        n_views = int(getattr(encoder, "context_n_views", 1))
        fusion = str(getattr(encoder, "context_fusion", "shared_stem"))
        self.aux_seg_supervision = bool(aux_seg_supervision and n_views > 1)
        self.aux_n_views = n_views
        # ``self.aux_feat_indices[k-1]`` = decoder feature index that aux
        # head ``k`` (k=1..n_views-1) reads from. Persisted so forward()
        # stays a flat lookup (no Python branching per call).
        self.aux_feat_indices: List[int] = []
        self.aux_heads = nn.ModuleList()
        # Per-aux-head output channel count. With the legacy / OFF path
        # every aux head emits the SAME ``num_fg_classes`` count as the
        # main head (``num_fg * D`` in 2.5D). With the native-depth ON
        # path each aux head k emits ``num_fg * D_k`` channels — the
        # caller passes ``aux_head_out_channels = [num_fg*D_1, ...,
        # num_fg*D_{K-1}]`` (length n_views - 1).
        n_aux_expected = max(n_views - 1, 0) if self.aux_seg_supervision else 0
        if aux_head_out_channels is None:
            self.aux_head_out_channels: List[int] = (
                [num_fg_classes] * n_aux_expected)
        else:
            if len(aux_head_out_channels) != n_aux_expected:
                raise ValueError(
                    f"aux_head_out_channels length "
                    f"({len(aux_head_out_channels)}) must equal "
                    f"n_views - 1 ({n_aux_expected}).")
            self.aux_head_out_channels = [int(c) for c in aux_head_out_channels]
        # The head builder bakes in norm/activation only when the mode is
        # ``conv`` — for the linear case it's a 1×1 conv with no norm.
        def _head(in_ch: int, out_ch: int) -> nn.Module:
            return _build_aux_head(
                mode=aux_head_mode,
                in_ch=in_ch,
                num_classes=out_ch,
                spatial_dims=spatial_dims,
                norm_type=norm_type,
                norm_groups=norm_groups,
                activation=activation,
            )
        self.aux_head_mode = aux_head_mode
        if self.aux_seg_supervision:
            n_dec = len(decoder.out_channels)
            if fusion == "hierarchical":
                # dec_features ordering: [-1] = highest res (stage 0), [-2]
                # = next-deeper (stage 1), ... so aux view k → index -1-k.
                # Need n_dec >= n_views so aux view (n_views-1) gets a
                # valid feature; validated upstream in Config.validate().
                if n_views > n_dec:
                    raise ValueError(
                        f"aux_seg_supervision (hierarchical) requires "
                        f"len(decoder.out_channels) >= n_views; got "
                        f"n_dec={n_dec}, n_views={n_views}.")
                for k in range(1, n_views):
                    feat_idx = n_dec - 1 - k
                    self.aux_feat_indices.append(feat_idx)
                    self.aux_heads.append(
                        _head(decoder.out_channels[feat_idx],
                              self.aux_head_out_channels[k - 1]))
            else:
                # Plan A — all aux heads on the highest-res decoder feat.
                in_ch = decoder.out_channels[-1]
                for k in range(1, n_views):
                    self.aux_feat_indices.append(n_dec - 1)
                    self.aux_heads.append(
                        _head(in_ch, self.aux_head_out_channels[k - 1]))

    def forward(
        self, x: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        """Forward pass.

        Args:
            x: (B, in_channels, *spatial) input. For 2.5D multi-FOV the
               channel layout is (B, n_views * D, H, W).

        Returns:
            See class docstring "Forward returns" for the contract matrix.
        """
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)
        target_size = x.shape[2:]

        main_out = self.seg_head(dec_features[-1])
        if main_out.shape[2:] != target_size:
            # Restore main output to the original input resolution. Use
            # bilinear/trilinear up-sampling per spatial_dims — consistent
            # with SegFormer/nnFormer-style patch-embed decoders.
            main_out = F.interpolate(
                main_out, size=target_size,
                mode=INTERP_SMOOTH[self.spatial_dims], align_corners=False)

        # ----- Build the aux heads' outputs (training only) ---------------
        # Gating on ``self.training`` keeps eval / inference paths bit-
        # identical to the legacy contract (predictor.py never sees a dict).
        aux_outs: List[torch.Tensor] = []
        if self.aux_seg_supervision and self.training:
            for head, feat_idx in zip(self.aux_heads, self.aux_feat_indices):
                ao = head(dec_features[feat_idx])
                if ao.shape[2:] != target_size:
                    ao = F.interpolate(
                        ao, size=target_size,
                        mode=INTERP_SMOOTH[self.spatial_dims],
                        align_corners=False)
                aux_outs.append(ao)

        # ----- Assemble main path ----------------------------------------
        if self.deep_supervision and self.training:
            # dec_features = [low, ..., high]; dec_features[-1] is already
            # used as main_out. DS heads consume in decreasing resolution.
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
        return {"encoder": enc, "decoder": dec, "seg_head": head, "total": total}


def _match_size(x: torch.Tensor, target_size, spatial_dims: int = 3) -> torch.Tensor:
    """Resize x to match target spatial size (bilinear/trilinear by dim)."""
    return F.interpolate(
        x, size=target_size,
        mode=INTERP_SMOOTH[spatial_dims], align_corners=False)
