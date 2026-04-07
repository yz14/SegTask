"""Model factory: builds complete UNet from config.

Handles encoder/decoder selection, channel configuration,
and dimensionality (2D/3D) setup.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Config, ModelConfig
from .encoders import build_encoder
from .decoders import build_decoder
from .unet import UNet

logger = logging.getLogger(__name__)


def build_model(cfg: Config) -> UNet:
    """Build a UNet model from configuration.

    Args:
        cfg: Full configuration.

    Returns:
        Configured UNet model.
    """
    mc = cfg.model
    num_classes = cfg.data.num_classes

    if num_classes < 2:
        raise ValueError(
            f"num_classes must be >= 2 (got {num_classes}). "
            "Set data.label_values or data.num_classes in config."
        )

    # Common kwargs for encoder/decoder blocks
    common_kwargs = dict(
        spatial_dims=mc.spatial_dims,
        norm_type=mc.norm_type,
        norm_groups=mc.norm_groups,
        activation=mc.activation,
        dropout=mc.dropout)

    # Build encoder
    encoder_kwargs = dict(
        in_channels=mc.in_channels,
        channels=mc.encoder_channels,
        blocks_per_level=mc.encoder_blocks_per_level,
        **common_kwargs)
    if mc.encoder_name == "vit":
        encoder_kwargs.update(
            num_heads=mc.vit_num_heads,
            mlp_ratio=mc.vit_mlp_ratio,
            qkv_bias=mc.vit_qkv_bias,
            drop_path_rate=mc.vit_drop_path_rate,
            patch_size=mc.vit_patch_size)
    encoder = build_encoder(mc.encoder_name, **encoder_kwargs)

    # Build decoder
    decoder_kwargs = dict(
        encoder_channels=mc.encoder_channels,
        decoder_channels=mc.decoder_channels,
        blocks_per_level=mc.decoder_blocks_per_level,
        upsample_mode=mc.upsample_mode,
        skip_mode=mc.skip_mode,
        **common_kwargs)
    if mc.decoder_name == "vit":
        decoder_kwargs.update(
            num_heads=mc.vit_num_heads,
            mlp_ratio=mc.vit_mlp_ratio,
            qkv_bias=mc.vit_qkv_bias,
            drop_path_rate=mc.vit_drop_path_rate)
    decoder = build_decoder(mc.decoder_name, **decoder_kwargs)

    # For 2.5D mode: model outputs predictions for ALL input slices
    # Output channels = num_classes * total_slices
    # e.g. 3 classes × 3 slices = 9 output channels
    total_slices = 1
    if cfg.data.mode == "2.5d":
        total_slices = 2 * cfg.data.num_slices_per_side + 1
    num_classes_out = num_classes * total_slices

    # Build UNet
    model = UNet(
        encoder=encoder,
        decoder=decoder,
        num_classes=num_classes_out,
        spatial_dims=mc.spatial_dims,
        deep_supervision=mc.deep_supervision)

    # Store metadata for trainer/predictor
    model.semantic_classes = num_classes
    model.total_slices = total_slices

    # Log model info
    param_count = model.get_param_count()
    logger.info(
        "Model built: encoder=%s, decoder=%s, dims=%dD, classes=%d, slices=%d (out_ch=%d)",
        mc.encoder_name, mc.decoder_name, mc.spatial_dims, num_classes,
        total_slices, num_classes_out,
    )
    logger.info(
        "Parameters: encoder=%dK, decoder=%dK, head=%dK, total=%.2fM",
        param_count["encoder"] // 1000,
        param_count["decoder"] // 1000,
        param_count["seg_head"] // 1000,
        param_count["total"] / 1e6)

    return model
