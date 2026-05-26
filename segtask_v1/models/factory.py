"""Model factory: build UNet3D/ADM/EDM2 from config."""

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
    fallback: int) -> List[int]:
    """Pick per-stage block counts: explicit list wins; else broadcast fallback."""
    if explicit:
        if len(explicit) != n_stages:
            raise ValueError(
                f"Per-stage block list length {len(explicit)} "
                f"!= expected {n_stages}")
        return list(explicit)
    return [fallback] * n_stages


class _StatefulStageBuilder:
    """Per-call stage builder; reads num_blocks from counts[idx], advances idx."""

    def __init__(self, factory_fn, counts: List[int]):
        self._fn     = factory_fn
        self._counts = counts
        self._idx    = 0

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
            num_blocks     = num_blocks,
            norm_type      = mc.norm_type,
            norm_groups    = mc.norm_groups,
            activation     = mc.activation,
            dropout        = mc.dropout,
            use_se         = mc.use_se,
            se_reduction   = mc.se_reduction,
            attention_type = mc.attention_type,
            block_type     = mc.block_type,
            spatial_dims   = spatial_dims)

    return _StatefulStageBuilder(factory, counts)


def _make_convnext_stage_builder(cfg: Config, counts: List[int]) -> _StatefulStageBuilder:
    """ConvNeXt stage builder. Blocks hard-code LN+GELU; warns if user set other norm/act."""
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
    # linear drop-path rates over total blocks
    total_blocks = sum(counts)
    dp_rates     = np.linspace(0, mc.drop_path_rate, max(total_blocks, 1)).tolist()
    rate_idx     = [0]
    ls_init      = float(getattr(mc, "convnext_layer_scale_init", 1e-6))  # <=0 disables

    def factory(in_ch: int, out_ch: int, num_blocks: int) -> ConvNeXtStage:
        start = rate_idx[0]
        end   = start + num_blocks
        rates = dp_rates[start:end] if dp_rates else [0.0] * num_blocks
        rate_idx[0] = end
        return ConvNeXtStage(
            in_ch, out_ch,
            num_blocks             = num_blocks,
            drop_path_rates        = rates,
            attention_type         = mc.attention_type,
            spatial_dims           = spatial_dims,
            layer_scale_init_value = ls_init)

    return _StatefulStageBuilder(factory, counts)


def _make_convnext_downsample_builder(
    cfg: Config) -> Callable[[int, int], ConvNeXtDownsample]:
    """Builder for paper-faithful LN → Conv(s=2) inter-stage downsample."""
    spatial_dims = getattr(cfg.model, "spatial_dims", 3)

    def build(in_ch: int, out_ch: int) -> ConvNeXtDownsample:
        return ConvNeXtDownsample(in_ch, out_ch, spatial_dims=spatial_dims)

    return build


def build_model(cfg: Config):
    """Build seg model dispatched on cfg.model.arch: 'unet' (default) | 'adm' | 'edm2'.
    'adm'/'edm2' ignore most backbone/block knobs (paper-faithful GN+SiLU / MP blocks).
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

    mc           = cfg.model
    enc_channels = list(mc.encoder_channels)
    num_fg       = cfg.num_fg_classes
    n_levels     = len(enc_channels)
    spatial_dims = getattr(mc, "spatial_dims", 3)

    # out_classes by mode:
    #   3D (z_axis/cubic/whole): num_fg * num_res (one per multi_res scale)
    #   2.5D folded: num_fg * D  (SliceChannelLoss splits per-fg into D slices)
    #   2.5D lifted (lift_2_5d_to_3d): true 3D, single-res, num_fg
    lift = bool(getattr(mc, "lift_2_5d_to_3d", False))
    if cfg.data.patch_mode == "2_5d" and not lift:
        num_res = 1
        D = int(cfg.data.patch_size[0])
        out_classes = num_fg * D
    elif cfg.data.patch_mode == "2_5d" and lift:
        num_res = 1
        out_classes = num_fg
    else:
        num_res = len(cfg.data.multi_res_scales)
        out_classes = num_fg * num_res

    # multi-FOV context views into the stem (2.5D-only; 3D stem reads scales directly)
    if cfg.data.patch_mode == "2_5d":
        context_n_views = max(len(cfg.data.multi_res_scales), 1)
    else:
        context_n_views = 1

    # Per-view in-ch list (2.5D + aux_keep_native_d ON); else None → uniform split
    in_ch_per_view_list   = None
    aux_head_out_channels = None
    if (cfg.data.patch_mode == "2_5d"
            and bool(getattr(cfg.data, "aux_keep_native_d", False))
            and context_n_views > 1):
        depths = list(cfg.aux_view_depths)
        in_ch_per_view_list   = depths
        aux_head_out_channels = [num_fg * d_k for d_k in depths[1:]]  # aux head k: num_fg*D_k

    # decoder builder call count varies: unet=n-1, unetpp=n*(n-1)/2, unet3p=0
    enc_counts = _resolve_blocks_per_stage(
        mc.encoder_blocks_per_stage, n_levels, mc.blocks_per_level)

    if   mc.decoder_type == "unet":
        expected_dec_calls = n_levels - 1
    elif mc.decoder_type == "unetpp":
        expected_dec_calls = n_levels * (n_levels - 1) // 2
    else:  # unet3p: no stage_builder calls
        expected_dec_calls = 0

    if mc.decoder_blocks_per_stage and mc.decoder_type == "unet":
        dec_counts = _resolve_blocks_per_stage(
            mc.decoder_blocks_per_stage, expected_dec_calls, mc.blocks_per_level)
    elif mc.decoder_blocks_per_stage:
        # UNet++: broadcast first count to every nested node
        dec_counts = [mc.decoder_blocks_per_stage[0]] * max(expected_dec_calls, 1)
    else:
        dec_counts = [mc.blocks_per_level] * max(expected_dec_calls, 1)

    # separate enc/dec builders so call counters are independent
    downsample_builder = None
    if   mc.backbone == "resnet":
        enc_builder = _make_resnet_stage_builder(cfg, enc_counts)
        dec_builder = _make_resnet_stage_builder(cfg, dec_counts)
    elif mc.backbone == "convnext":
        enc_builder = _make_convnext_stage_builder(cfg, enc_counts)
        dec_builder = _make_convnext_stage_builder(cfg, dec_counts)
        # LN-first downsample; toggle to fall back to generic Downsample for ablation
        if bool(getattr(mc, "convnext_downsample_lnfirst", True)):
            downsample_builder = _make_convnext_downsample_builder(cfg)
    else:
        raise ValueError(f"Unknown backbone: {mc.backbone}")

    # Build encoder
    encoder = Encoder(
        in_channels         = mc.in_channels,
        stage_channels      = enc_channels,
        stage_builder       = enc_builder,
        norm_type           = mc.norm_type,
        norm_groups         = mc.norm_groups,
        activation          = mc.activation,
        downsample_mode     = mc.downsample_mode,
        stem_mode           = mc.stem_mode,
        spatial_dims        = spatial_dims,
        context_n_views     = context_n_views,
        context_fusion      = getattr(mc, "context_fusion", "shared_stem"),
        in_ch_per_view_list = in_ch_per_view_list,
        downsample_builder  = downsample_builder)

    # decoder: unet | unetpp | unet3p
    if   mc.decoder_type == "unet3p":
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
            encoder_channels = enc_channels,
            stage_builder    = dec_builder,
            upsample_mode    = mc.upsample_mode,
            skip_mode        = mc.skip_mode,
            skip_attention   = mc.skip_attention,
            spatial_dims     = spatial_dims)

    aux_seg_supervision = bool(getattr(mc, "aux_seg_supervision", False))
    # aux only meaningful with multi-FOV; mirror UNet3D's internal gate for accurate logging
    aux_seg_supervision = aux_seg_supervision and context_n_views > 1
    # per-view aux out channels only when 2.5D native-D ON; else None → default num_fg_classes
    aux_head_out_channels_arg = (
        aux_head_out_channels if (aux_seg_supervision and aux_head_out_channels) else None)
    model = UNet3D(
        encoder               = encoder,
        decoder               = decoder,
        num_fg_classes        = out_classes,
        deep_supervision      = mc.deep_supervision,
        spatial_dims          = spatial_dims,
        aux_seg_supervision   = aux_seg_supervision,
        aux_head_mode         = getattr(mc, "aux_head_mode", "linear"),
        aux_head_out_channels = aux_head_out_channels_arg,
        norm_type             = mc.norm_type,
        norm_groups           = mc.norm_groups,
        activation            = mc.activation)

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
