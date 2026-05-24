"""Model factory: build UNet3D from config.

Creates the appropriate encoder/decoder stages based on the backbone choice
(resnet or convnext), then assembles them into a UNet3D.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, List

import numpy as np

from ..config import Config
from .blocks import Downsample, Upsample
from .convnext import ConvNeXtDownsample, ConvNeXtStage
from .resnet import ResNetStage
from .unet import Encoder, Decoder, UNet3D
from .unet3p import UNet3PDecoder
from .unetpp import UNetPPDecoder

logger = logging.getLogger(__name__)


def _resolve_blocks_per_stage(
    explicit: List[int],
    n_stages: int,
    fallback: int,
) -> List[int]:
    """Pick per-stage block counts: explicit list wins; else broadcast fallback."""
    if explicit:
        if len(explicit) != n_stages:
            raise ValueError(
                f"Per-stage block list length {len(explicit)} "
                f"!= expected {n_stages}")
        return list(explicit)
    return [fallback] * n_stages


class _StatefulStageBuilder:
    """Builder that consumes a per-stage block-count list in call order.

    Each call returns a stage with ``num_blocks = counts[idx]`` and advances
    the internal index.  Used by ``Encoder`` / ``Decoder`` which call the
    builder once per level in deterministic order.
    """

    def __init__(self, factory_fn, counts: List[int]):
        self._fn = factory_fn
        self._counts = counts
        self._idx = 0

    def __call__(self, in_ch: int, out_ch: int):
        if self._idx >= len(self._counts):
            raise RuntimeError(
                f"StageBuilder exhausted after {self._idx} calls, "
                f"counts={self._counts}")
        n_blocks = self._counts[self._idx]
        self._idx += 1
        return self._fn(in_ch, out_ch, n_blocks)


def _make_resnet_stage_builder(cfg: Config, counts: List[int]) -> _StatefulStageBuilder:
    """Return a stateful builder for the given per-stage block counts."""
    mc = cfg.model
    spatial_dims = getattr(mc, "spatial_dims", 3)

    def factory(in_ch: int, out_ch: int, num_blocks: int) -> ResNetStage:
        return ResNetStage(
            in_ch, out_ch,
            num_blocks=num_blocks,
            norm_type=mc.norm_type,
            norm_groups=mc.norm_groups,
            activation=mc.activation,
            dropout=mc.dropout,
            use_se=mc.use_se,
            se_reduction=mc.se_reduction,
            attention_type=mc.attention_type,
            block_type=mc.block_type,
            spatial_dims=spatial_dims,
        )

    return _StatefulStageBuilder(factory, counts)


def _make_convnext_stage_builder(cfg: Config, counts: List[int]) -> _StatefulStageBuilder:
    """Return a callable(in_ch, out_ch) → ConvNeXtStage.

    ISSUE-N: ConvNeXt blocks hard-code LayerNorm + GELU internally (matching
    the original paper). `cfg.model.norm_type / activation / norm_groups`
    are therefore ignored by the ConvNeXt path. We warn explicitly when the
    user configured anything other than the defaults so the discrepancy is
    visible rather than silent. The stem and skip-projection paths (built
    in `Encoder` / `Decoder`) DO still honour `norm_type/activation`, so
    this warning is specifically about the intra-stage blocks.
    """
    mc = cfg.model
    spatial_dims = getattr(mc, "spatial_dims", 3)
    non_default = []
    if mc.norm_type != "instance":
        non_default.append(f"norm_type={mc.norm_type!r}")
    if mc.activation != "leakyrelu":
        non_default.append(f"activation={mc.activation!r}")
    if mc.use_se:
        non_default.append("use_se=True")
    if mc.dropout and mc.dropout > 0.0:
        non_default.append(f"dropout={mc.dropout}")
    if non_default:
        logger.warning(
            "Backbone=convnext: block-internal norm/activation are fixed to "
            "LayerNorm+GELU and the following settings are IGNORED inside "
            "ConvNeXt blocks: %s. (They still apply to the stem/decoder "
            "skip projections built in Encoder/Decoder.)",
            ", ".join(non_default))
    # Distribute drop-path rates linearly over the TOTAL number of blocks
    # this builder will instantiate.
    total_blocks = sum(counts)
    dp_rates = np.linspace(0, mc.drop_path_rate, max(total_blocks, 1)).tolist()
    rate_idx = [0]
    # LayerScale init value (Touvron et al.); <=0 disables. Default 1e-6
    # matches the official ConvNeXt block and is essential for stable
    # training of deep ConvNeXt-style networks.
    ls_init = float(getattr(mc, "convnext_layer_scale_init", 1e-6))

    def factory(in_ch: int, out_ch: int, num_blocks: int) -> ConvNeXtStage:
        start = rate_idx[0]
        end = start + num_blocks
        rates = dp_rates[start:end] if dp_rates else [0.0] * num_blocks
        rate_idx[0] = end
        return ConvNeXtStage(
            in_ch, out_ch,
            num_blocks=num_blocks,
            drop_path_rates=rates,
            attention_type=mc.attention_type,
            spatial_dims=spatial_dims,
            layer_scale_init_value=ls_init,
        )

    return _StatefulStageBuilder(factory, counts)


def _make_convnext_downsample_builder(
    cfg: Config) -> Callable[[int, int], ConvNeXtDownsample]:
    """Return a builder ``(in_ch, out_ch) -> ConvNeXtDownsample``.

    Used to inject the paper-faithful ``LayerNorm → Conv(s=2)`` topology
    into ``Encoder`` when ``backbone == "convnext"`` and
    ``convnext_downsample_lnfirst`` is enabled. See
    :class:`ConvNeXtDownsample` for the rationale.
    """
    spatial_dims = getattr(cfg.model, "spatial_dims", 3)

    def build(in_ch: int, out_ch: int) -> ConvNeXtDownsample:
        return ConvNeXtDownsample(in_ch, out_ch, spatial_dims=spatial_dims)

    return build


def build_model(cfg: Config):
    """Build a segmentation model from config.

    Dispatches on ``cfg.model.arch``:

      * ``"unet"`` (default) — :class:`models.unet.UNet3D` built via the
        ResNet/ConvNeXt backbone path below (legacy, bit-identical).
      * ``"adm"``  — :class:`models.adm_unet.ADMSegModel`. Paper-faithful
        ADM blocks (Dhariwal & Nichol, NeurIPS 2021) with timestep-emb
        path stripped. ``backbone`` / ``block_type`` / ``norm_type`` /
        ``activation`` / ``use_se`` / ``attention_type`` / ``dropout``
        configured at the YAML level are IGNORED for ADM (paper fixes
        GroupNorm32 + SiLU); ``dropout`` is reused as ``ResBlock``
        dropout. Reads ``encoder_channels``, ``encoder_blocks_per_stage``,
        ``decoder_blocks_per_stage``, ``stem_mode``, ``context_fusion``,
        ``deep_supervision``, ``aux_seg_supervision``, ``aux_head_mode``,
        plus ``adm_*`` extras.
      * ``"edm2"`` — :class:`models.edm2_unet.EDM2SegModel`. Paper-faithful
        magnitude-preserving blocks (Karras et al., CVPR 2024) with
        noise/class-emb path stripped. Same whitelist as ``adm`` plus
        ``edm2_*`` extras.

    The non-``unet`` archs route around all backbone-specific code below;
    they have their own self-contained build functions in their modules.
    """
    arch = str(getattr(cfg.model, "arch", "unet")).lower()
    if arch == "adm":
        from .adm_unet import build_adm_seg_model
        return build_adm_seg_model(cfg)
    if arch == "edm2":
        from .edm2_unet import build_edm2_seg_model
        return build_edm2_seg_model(cfg)
    if arch != "unet":
        raise ValueError(
            f"Unknown model.arch: {arch!r}. Valid: 'unet' | 'adm' | 'edm2'.")

    mc = cfg.model
    enc_channels = list(mc.encoder_channels)
    num_fg = cfg.num_fg_classes
    n_levels = len(enc_channels)
    spatial_dims = getattr(mc, "spatial_dims", 3)

    # Output channel count by mode:
    #   3D modes (z_axis / cubic / whole) — num_fg per resolution scale
    #     (C_res >= 1; default [1.0] gives num_fg).
    #   2.5D mode (folded) — num_fg per slice (D slices stacked as input
    #     channels); SliceChannelLoss splits the (num_fg * D)-channel
    #     output into per-fg-class D-slice binary masks.
    #   2.5D mode + lift_2_5d_to_3d — D preserved as a real spatial axis.
    #     The model is built as a true 3D UNet with ``out_classes=num_fg``
    #     (single-resolution) and the trainer routes the loss through
    #     MultiResolutionLoss(num_res=1). Bit-equivalent to the single-
    #     scale 3D mode shape contract.
    lift = bool(getattr(mc, "lift_2_5d_to_3d", False))
    if cfg.data.patch_mode == "2_5d" and not lift:
        # Folded 2.5D: output is num_fg * D for the MAIN head — independent
        # of how many context z-FOV views feed the stem. Multi-FOV affects
        # ONLY the stem's input-channel count (D * n_views legacy, or
        # sum(D_k) when aux_keep_native_d=True), not the main head output:
        # the main loss / metrics consume the 1× FOV's true geometry.
        num_res = 1
        D = int(cfg.data.patch_size[0])
        out_classes = num_fg * D
    elif cfg.data.patch_mode == "2_5d" and lift:
        # Lifted 2.5D: model is single-resolution 3D over (B, n_views, D, H, W).
        # Aux views still feed the stem as extra input channels (multi-FOV
        # context fusion remains active via ``context_fusion``); only the
        # main supervision target (view 0 = 1× FOV) drives the loss.
        num_res = 1
        out_classes = num_fg
    else:
        num_res = len(cfg.data.multi_res_scales)
        out_classes = num_fg * num_res

    # Number of multi-FOV context views fed to the stem (2.5D mode only).
    # In 3D modes the stem already consumes ``len(multi_res_scales)`` input
    # channels directly, with no per-view stem split — so n_views stays 1.
    # Lift mode reuses the 2.5D stem-fusion topology (each view = 1 input
    # channel) so multi_stem_proj / hierarchical context fusion still
    # apply; the only difference vs. folded 2.5D is that ``in_ch_per_view``
    # is 1 (a single channel per view) instead of D.
    if cfg.data.patch_mode == "2_5d":
        context_n_views = max(len(cfg.data.multi_res_scales), 1)
    else:
        context_n_views = 1

    # Per-view input channel layout (2.5D ON path only). For OFF path /
    # 3D modes we leave this as ``None`` so the encoder uses the legacy
    # uniform ``in_channels // n_views`` split (bit-identical behaviour).
    in_ch_per_view_list = None
    aux_head_out_channels = None
    if (cfg.data.patch_mode == "2_5d"
            and bool(getattr(cfg.data, "aux_keep_native_d", False))
            and context_n_views > 1):
        depths = list(cfg.aux_view_depths)
        # Stem consumes per-view native depths; sum equals model.in_channels.
        in_ch_per_view_list = depths
        # Aux head k emits ``num_fg * D_k`` channels (vs. main's num_fg*D_0).
        aux_head_out_channels = [num_fg * d_k for d_k in depths[1:]]

    # Resolve per-stage block counts.  Encoder has ``n_levels`` stages;
    # a classical UNet-style decoder has ``n_levels - 1`` stages.  For
    # UNet++/UNet3+ variants, the decoder builder is called a different
    # number of times — we therefore provide a "generous" count list that
    # repeats the first value enough times to never exhaust.
    enc_counts = _resolve_blocks_per_stage(
        mc.encoder_blocks_per_stage, n_levels, mc.blocks_per_level)

    if mc.decoder_type == "unet":
        expected_dec_calls = n_levels - 1
    elif mc.decoder_type == "unetpp":
        # UNet++ builds n*(n-1)/2 nested nodes via the stage_builder.
        expected_dec_calls = n_levels * (n_levels - 1) // 2
    else:  # unet3p — no stage_builder calls; counts are unused
        expected_dec_calls = 0

    if mc.decoder_blocks_per_stage and mc.decoder_type == "unet":
        dec_counts = _resolve_blocks_per_stage(
            mc.decoder_blocks_per_stage, expected_dec_calls, mc.blocks_per_level)
    elif mc.decoder_blocks_per_stage:
        # UNet++: broadcast first decoder count to every nested node.
        dec_counts = [mc.decoder_blocks_per_stage[0]] * max(expected_dec_calls, 1)
    else:
        dec_counts = [mc.blocks_per_level] * max(expected_dec_calls, 1)

    # Select backbone stage builder (separate instances for enc/dec — each
    # owns its own call counter).
    downsample_builder = None
    if mc.backbone == "resnet":
        enc_builder = _make_resnet_stage_builder(cfg, enc_counts)
        dec_builder = _make_resnet_stage_builder(cfg, dec_counts)
    elif mc.backbone == "convnext":
        enc_builder = _make_convnext_stage_builder(cfg, enc_counts)
        dec_builder = _make_convnext_stage_builder(cfg, dec_counts)
        # Paper-faithful LN-first inter-stage downsample. The toggle lets
        # users fall back to the generic Downsample for ablation.
        if bool(getattr(mc, "convnext_downsample_lnfirst", True)):
            downsample_builder = _make_convnext_downsample_builder(cfg)
    else:
        raise ValueError(f"Unknown backbone: {mc.backbone}")

    # Build encoder
    encoder = Encoder(
        in_channels=mc.in_channels,
        stage_channels=enc_channels,
        stage_builder=enc_builder,
        norm_type=mc.norm_type,
        norm_groups=mc.norm_groups,
        activation=mc.activation,
        downsample_mode=mc.downsample_mode,
        stem_mode=mc.stem_mode,
        spatial_dims=spatial_dims,
        context_n_views=context_n_views,
        context_fusion=getattr(mc, "context_fusion", "shared_stem"),
        in_ch_per_view_list=in_ch_per_view_list,
        downsample_builder=downsample_builder)

    # Build decoder — classical UNet / UNet++ / UNet3+.
    if mc.decoder_type == "unet3p":
        decoder = UNet3PDecoder(
            encoder_channels=enc_channels,
            cat_channels=mc.unet3p_cat_channels,
            norm_type=mc.norm_type,
            norm_groups=mc.norm_groups,
            activation=mc.activation,
            skip_attention=mc.skip_attention,
            spatial_dims=spatial_dims)
    elif mc.decoder_type == "unetpp":
        decoder = UNetPPDecoder(
            encoder_channels=enc_channels,
            stage_builder=dec_builder,
            upsample_mode=mc.upsample_mode,
            skip_attention=mc.skip_attention,
            spatial_dims=spatial_dims)
    else:
        decoder = Decoder(
            encoder_channels=enc_channels,
            stage_builder=dec_builder,
            upsample_mode=mc.upsample_mode,
            skip_mode=mc.skip_mode,
            skip_attention=mc.skip_attention,
            spatial_dims=spatial_dims)

    # Assemble UNet
    aux_seg_supervision = bool(getattr(mc, "aux_seg_supervision", False))
    # Aux supervision is only meaningful with multi-FOV views; UNet3D's
    # constructor disables it internally when n_views==1, but we mirror
    # the gate here so the log line tells the truth in single-FOV configs.
    aux_seg_supervision = aux_seg_supervision and context_n_views > 1
    # Aux head per-view output channel counts: present only for the 2.5D
    # native-depth ON path (each aux view has its own D_k); ``None`` for
    # OFF path so UNet3D defaults each aux head to ``num_fg_classes``
    # channels (== num_fg * D, identical to the main head — bit-identical
    # to the legacy build).
    aux_head_out_channels_arg = (
        aux_head_out_channels if (aux_seg_supervision and aux_head_out_channels)
        else None)
    model = UNet3D(
        encoder=encoder,
        decoder=decoder,
        num_fg_classes=out_classes,
        deep_supervision=mc.deep_supervision,
        spatial_dims=spatial_dims,
        aux_seg_supervision=aux_seg_supervision,
        aux_head_mode=getattr(mc, "aux_head_mode", "linear"),
        aux_head_out_channels=aux_head_out_channels_arg,
        norm_type=mc.norm_type,
        norm_groups=mc.norm_groups,
        activation=mc.activation)

    # Log model info
    pc = model.param_count()
    logger.info(
        "Built UNet3D [%s/%s, decoder=%s, preset=%s]: "
        "enc=%.2fM, dec=%.2fM, total=%.2fM, channels=%s, "
        "enc_blocks=%s, dec_blocks=%s, out_classes=%d (fg=%d, res=%d), "
        "stem=%s(stride=%d, n_views=%d, fusion=%s), "
        "down=%s, up=%s, skip=%s, attn=%s, skip_attn=%s, "
        "ds=%s, aux_seg=%s(n_aux_heads=%d, mode=%s)",
        mc.backbone, mc.block_type, mc.decoder_type, mc.resenc_preset,
        pc["encoder"] / 1e6, pc["decoder"] / 1e6, pc["total"] / 1e6,
        enc_channels,
        enc_counts, dec_counts,
        out_classes, num_fg,
        num_res if num_res > 0 else 1,
        mc.stem_mode, encoder.stem_stride,
        context_n_views, getattr(mc, "context_fusion", "shared_stem"),
        mc.downsample_mode, mc.upsample_mode, mc.skip_mode,
        mc.attention_type, mc.skip_attention,
        mc.deep_supervision, aux_seg_supervision, len(model.aux_heads),
        getattr(mc, "aux_head_mode", "linear"))

    return model
