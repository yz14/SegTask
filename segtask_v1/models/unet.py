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

from typing import Dict, List, Union

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
        context_fusion: str = "shared_stem"):
        super().__init__()
        self.spatial_dims = spatial_dims
        # Stem: project input to first channel count. The stem may introduce
        # a spatial stride (``patch2``/``patch4``) — ``stem_stride`` is
        # preserved as a property so the wrapping UNet can add a matching
        # final upsample when necessary.
        #
        # Multi-FOV context fusion (2.5D mode): when ``context_n_views > 1``
        # the input layout is ``(B, n_views * in_ch_per_view, *spatial)``
        # with ``in_ch_per_view = in_channels // n_views``; the dispatcher
        # in ``build_context_stem`` selects either a single shared stem
        # ("shared_stem") or per-view independent stems + 1×1 fusion
        # ("multi_stem_proj"). For ``context_n_views == 1`` the behaviour
        # is bit-identical to the legacy single-stem path.
        if context_n_views < 1:
            raise ValueError(
                f"context_n_views must be >= 1, got {context_n_views}")
        if in_channels % context_n_views != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by "
                f"context_n_views ({context_n_views})")
        self.context_n_views = context_n_views
        self.context_fusion = context_fusion
        in_ch_per_view = in_channels // context_n_views
        self.stem, self.stem_stride = build_context_stem(
            mode=stem_mode, fusion=context_fusion,
            n_views=context_n_views,
            in_ch_per_view=in_ch_per_view,
            out_ch=stage_channels[0],
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims,
            stage_channels=stage_channels)

        # Encoder stages and downsampling
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, ch in enumerate(stage_channels):
            in_ch = stage_channels[i - 1] if i > 0 else stage_channels[0]
            self.stages.append(stage_builder(in_ch, ch))
            if i > 0:
                self.downsamples.append(
                    Downsample(
                        stage_channels[i - 1], stage_channels[i - 1],
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


class UNet3D(nn.Module):
    """Generic 3D UNet with pluggable encoder/decoder stages.

    Args:
        encoder: Encoder module.
        decoder: Decoder module.
        num_fg_classes: Number of foreground classes (output channels).
        deep_supervision: Output at multiple decoder levels during training.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder,
        num_fg_classes: int,
        deep_supervision: bool = False,
        spatial_dims: int = 3):
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

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: (B, 1, D, H, W) input.

        Returns:
            If deep_supervision=False or eval: (B, num_fg, D, H, W) logits.
            If deep_supervision=True and training: list of multi-scale logits
                ordered [main_out (highest-res), 2nd_high, 3rd_high, ..., lowest].
        """
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        main_out = self.seg_head(dec_features[-1])
        if self.stem_stride > 1:
            # Restore main output to the original input resolution. Use
            # bilinear/trilinear up-sampling per spatial_dims — consistent
            # with SegFormer/nnFormer-style patch-embed decoders.
            main_out = F.interpolate(
                main_out, size=x.shape[2:],
                mode=INTERP_SMOOTH[self.spatial_dims], align_corners=False)

        if not self.deep_supervision or not self.training:
            return main_out

        # dec_features = [low, ..., high]; dec_features[-1] is already used
        # as main_out. DS heads must consume features in decreasing resolution:
        # dec_features[-2] (2nd-highest), dec_features[-3], ..., dec_features[0] (lowest).
        outputs = [main_out]
        for i, head in enumerate(self.ds_heads):
            outputs.append(head(dec_features[-2 - i]))
        return outputs

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
