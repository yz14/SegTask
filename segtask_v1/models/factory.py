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
from .convnext import ConvNeXtStage
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
        )

    return _StatefulStageBuilder(factory, counts)


def build_model(cfg: Config) -> UNet3D:
    """Build UNet3D model from config.

    Args:
        cfg: Full configuration.

    Returns:
        UNet3D model ready for training.
    """
    mc = cfg.model
    enc_channels = list(mc.encoder_channels)
    num_fg = cfg.num_fg_classes
    n_levels = len(enc_channels)

    # Output = num_fg per resolution scale (C_res >= 1, default [1.0])
    num_res = len(cfg.data.multi_res_scales)
    out_classes = num_fg * num_res

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
    if mc.backbone == "resnet":
        enc_builder = _make_resnet_stage_builder(cfg, enc_counts)
        dec_builder = _make_resnet_stage_builder(cfg, dec_counts)
    elif mc.backbone == "convnext":
        enc_builder = _make_convnext_stage_builder(cfg, enc_counts)
        dec_builder = _make_convnext_stage_builder(cfg, dec_counts)
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
        stem_mode=mc.stem_mode)

    # Build decoder — classical UNet / UNet++ / UNet3+.
    if mc.decoder_type == "unet3p":
        decoder = UNet3PDecoder(
            encoder_channels=enc_channels,
            cat_channels=mc.unet3p_cat_channels,
            norm_type=mc.norm_type,
            norm_groups=mc.norm_groups,
            activation=mc.activation,
            skip_attention=mc.skip_attention)
    elif mc.decoder_type == "unetpp":
        decoder = UNetPPDecoder(
            encoder_channels=enc_channels,
            stage_builder=dec_builder,
            upsample_mode=mc.upsample_mode,
            skip_attention=mc.skip_attention)
    else:
        decoder = Decoder(
            encoder_channels=enc_channels,
            stage_builder=dec_builder,
            upsample_mode=mc.upsample_mode,
            skip_mode=mc.skip_mode,
            skip_attention=mc.skip_attention)

    # Assemble UNet
    model = UNet3D(
        encoder=encoder,
        decoder=decoder,
        num_fg_classes=out_classes,
        deep_supervision=mc.deep_supervision)

    # Log model info
    pc = model.param_count()
    logger.info(
        "Built UNet3D [%s/%s, decoder=%s, preset=%s]: "
        "enc=%.2fM, dec=%.2fM, total=%.2fM, channels=%s, "
        "enc_blocks=%s, dec_blocks=%s, out_classes=%d (fg=%d, res=%d), "
        "stem=%s(stride=%d), down=%s, up=%s, skip=%s, attn=%s, skip_attn=%s",
        mc.backbone, mc.block_type, mc.decoder_type, mc.resenc_preset,
        pc["encoder"] / 1e6, pc["decoder"] / 1e6, pc["total"] / 1e6,
        enc_channels,
        enc_counts, dec_counts,
        out_classes, num_fg,
        num_res if num_res > 0 else 1,
        mc.stem_mode, encoder.stem_stride,
        mc.downsample_mode, mc.upsample_mode, mc.skip_mode,
        mc.attention_type, mc.skip_attention)

    return model
