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
    """Return a callable(in_ch, out_ch) → ConvNeXtStage."""
    mc = cfg.model
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
        num_fg_classes=num_fg,
        deep_supervision=mc.deep_supervision)

    # Log model info
    pc = model.param_count()
    logger.info(
        "Built UNet3D [%s]: enc=%.2fM, dec=%.2fM, total=%.2fM, "
        "channels=%s, blocks=%d, fg_classes=%d",
        mc.backbone,
        pc["encoder"] / 1e6, pc["decoder"] / 1e6, pc["total"] / 1e6,
        enc_channels, mc.blocks_per_level, num_fg)

    return model
