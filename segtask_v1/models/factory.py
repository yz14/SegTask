"""Model factory: build UNet3D from config.

Creates the appropriate encoder/decoder stages based on the backbone choice
(resnet or convnext), then assembles them into a UNet3D.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable

import numpy as np

from ..config import Config
from .blocks import Downsample, Upsample
from .convnext import ConvNeXtStage
from .resnet import ResNetStage
from .unet import Encoder, Decoder, UNet3D

logger = logging.getLogger(__name__)


def _make_resnet_stage_builder(cfg: Config) -> Callable:
    """Return a callable(in_ch, out_ch) → ResNetStage."""
    mc = cfg.model
    return partial(
        ResNetStage,  # 多层残差单元
        num_blocks=mc.blocks_per_level,
        norm_type=mc.norm_type,
        norm_groups=mc.norm_groups,
        activation=mc.activation,
        dropout=mc.dropout,
        use_se=mc.use_se,
        se_reduction=mc.se_reduction)


def _make_convnext_stage_builder(cfg: Config) -> Callable:
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
    n_levels = len(mc.encoder_channels)
    total_blocks = n_levels * mc.blocks_per_level * 2  # encoder + decoder
    # Linearly increasing drop path rates
    dp_rates = np.linspace(0, mc.drop_path_rate, total_blocks).tolist()

    # We'll distribute drop path rates across stages
    block_idx = [0]  # mutable counter

    def builder(in_ch: int, out_ch: int) -> ConvNeXtStage:
        start = block_idx[0]
        end   = start + mc.blocks_per_level
        rates = dp_rates[start:end]
        block_idx[0] = end
        return ConvNeXtStage(
            in_ch, out_ch,
            num_blocks=mc.blocks_per_level,
            drop_path_rates=rates)

    return builder


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

    # Output = num_fg per resolution scale (C_res >= 1, default [1.0])
    num_res = len(cfg.data.multi_res_scales)
    out_classes = num_fg * num_res

    # Select backbone stage builder
    if   mc.backbone == "resnet":
        enc_builder = _make_resnet_stage_builder(cfg)
        dec_builder = _make_resnet_stage_builder(cfg)
    elif mc.backbone == "convnext":
        enc_builder = _make_convnext_stage_builder(cfg)
        dec_builder = _make_convnext_stage_builder(cfg)
    else:
        raise ValueError(f"Unknown backbone: {mc.backbone}")

    # Build encoder
    encoder = Encoder(
        in_channels=mc.in_channels,
        stage_channels=enc_channels,
        stage_builder=enc_builder,
        norm_type=mc.norm_type,
        norm_groups=mc.norm_groups,
        activation=mc.activation)

    # Build decoder (symmetric channels)
    decoder = Decoder(
        encoder_channels=enc_channels,
        stage_builder=dec_builder,
        upsample_mode=mc.upsample_mode,
        skip_mode=mc.skip_mode)

    # Assemble UNet
    model = UNet3D(
        encoder=encoder,
        decoder=decoder,
        num_fg_classes=out_classes,
        deep_supervision=mc.deep_supervision)

    # Log model info
    pc = model.param_count()
    logger.info(
        "Built UNet3D [%s]: enc=%.2fM, dec=%.2fM, total=%.2fM, "
        "channels=%s, blocks=%d, out_classes=%d (fg=%d, res=%d)",
        mc.backbone,
        pc["encoder"] / 1e6, pc["decoder"] / 1e6, pc["total"] / 1e6,
        enc_channels, mc.blocks_per_level, out_classes, num_fg,
        num_res if num_res > 0 else 1)

    return model
