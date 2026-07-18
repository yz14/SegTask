"""根据 config 构建 gentask 的共享 2.5D/3D 图像到图像模型。"""

from __future__ import annotations

import logging
from typing import List


from ..config import Config
# 共享 stage/stride 构建件与 taskcore 合流（gen ModelConfig 继承核心段，
# multirf/selfattn/drop-path 等扩展默认关闭时逐位兼容历史行为）。
from taskcore.models.factory import (
    _make_convnext_downsample_builder,
    _make_convnext_stage_builder,
    _make_resnet_stage_builder,
    _resolve_blocks_per_stage,
    compute_downsample_strides,
)
from taskcore.models.topology import ModelTopology, build_topology
from taskcore.models.unet import Encoder, Decoder, UNet3D
from taskcore.models.unet3p import UNet3PDecoder
from taskcore.models.unetpp import UNetPPDecoder

logger = logging.getLogger(__name__)


def build_model(cfg: Config):
    """顶层模型工厂。

    生成任务（``cfg.is_generation``）分派到回归 / 扩散生成模型；否则按
    ``cfg.model.arch`` 构造分割 backbone（见 ``build_backbone``）。
    """
    if getattr(cfg, "is_generation", False):
        from .generation import build_generation_model
        return build_generation_model(cfg)
    return build_backbone(cfg)


def _resolve_decoder_counts(mc, n_levels: int) -> list[int]:
    """按 decoder 类型解析各 stage 的 block 数。"""
    if mc.decoder_type == "unet":
        expected_dec_calls = n_levels - 1
    elif mc.decoder_type == "unetpp":
        expected_dec_calls = n_levels * (n_levels - 1) // 2
    else:
        expected_dec_calls = 0

    if mc.decoder_blocks_per_stage and mc.decoder_type == "unet":
        return _resolve_blocks_per_stage(
            mc.decoder_blocks_per_stage, expected_dec_calls, mc.blocks_per_level)
    if mc.decoder_blocks_per_stage:
        return [mc.decoder_blocks_per_stage[0]] * max(expected_dec_calls, 1)
    return [mc.blocks_per_level] * max(expected_dec_calls, 1)


def _resolve_backbone_stage_builders(cfg: Config, enc_counts, dec_counts):
    """按 backbone 解析 encoder/decoder stage builder。"""
    mc = cfg.model
    # 旧 use_se 兼容（生成侧遗留字段）：attention_type='none' 时提升为 'se'，
    # 与块内语义一致；共享 builder 只读 attention_type。
    if mc.use_se and mc.attention_type == "none":
        mc.attention_type = "se"
    downsample_builder = None
    if mc.backbone == "resnet":
        enc_builder = _make_resnet_stage_builder(cfg, enc_counts)
        dec_builder = _make_resnet_stage_builder(cfg, dec_counts)
    elif mc.backbone == "convnext":
        enc_builder = _make_convnext_stage_builder(cfg, enc_counts)
        dec_builder = _make_convnext_stage_builder(cfg, dec_counts)
        if bool(mc.convnext_downsample_lnfirst):
            downsample_builder = _make_convnext_downsample_builder(cfg)
    else:
        raise ValueError(f"Unknown backbone: {mc.backbone}")
    return enc_builder, dec_builder, downsample_builder


def _validate_anisotropic_downsampling(cfg: Config, downsample_builder, ds_strides):
    """检查各向异性 stride 与 decoder / downsample 实现的兼容性。"""
    mc = cfg.model
    if ds_strides is None or not any(
            any(int(s) != 2 for s in stage) for stage in ds_strides):
        return
    if downsample_builder is not None:
        raise ValueError(
            "Anisotropic downsampling is not supported with ConvNeXt "
            "LN-first downsample. Set model.convnext_downsample_lnfirst="
            "false to use the generic Downsample, or disable "
            "anisotropic_pooling/downsample_strides.")
    if mc.decoder_type != "unet":
        raise ValueError(
            f"Anisotropic downsampling currently supports only "
            f"decoder_type='unet'; got {mc.decoder_type!r}. "
            f"(unetpp/unet3p decoders use isotropic ×2 up/down.)")
    if mc.downsample_mode not in _ANISO_DOWN_MODES:
        raise ValueError(
            f"Anisotropic downsampling requires downsample_mode in "
            f"{_ANISO_DOWN_MODES}; got {mc.downsample_mode!r}.")
    if mc.upsample_mode not in _ANISO_UP_MODES:
        raise ValueError(
            f"Anisotropic downsampling requires upsample_mode in "
            f"{_ANISO_UP_MODES}; got {mc.upsample_mode!r}.")
    logger.info("Anisotropic downsample strides (per level): %s", ds_strides)


def _build_unet_backbone(cfg: Config):
    """构建共享 UNet-style backbone。"""
    mc = cfg.model
    enc_channels = list(mc.encoder_channels)
    num_fg = cfg.num_fg_classes
    n_levels = len(enc_channels)

    topo = build_topology(cfg)
    spatial_dims = topo.spatial_dims
    out_classes = topo.out_classes
    num_stem_fusion_views = topo.num_stem_fusion_views
    in_ch_per_view_list = topo.in_ch_per_view_list
    aux_head_out_channels = topo.aux_head_out_channels
    cond_in_channels = topo.cond_in_channels

    enc_counts = _resolve_blocks_per_stage(
        mc.encoder_blocks_per_stage, n_levels, mc.blocks_per_level)
    dec_counts = _resolve_decoder_counts(mc, n_levels)
    enc_builder, dec_builder, downsample_builder = _resolve_backbone_stage_builders(
        cfg, enc_counts, dec_counts)

    ds_strides = compute_downsample_strides(cfg, spatial_dims, n_levels)
    _validate_anisotropic_downsampling(cfg, downsample_builder, ds_strides)

    encoder = Encoder(
        in_channels           = mc.in_channels,
        stage_channels        = enc_channels,
        stage_builder         = enc_builder,
        norm_type             = mc.norm_type,
        norm_groups           = mc.norm_groups,
        activation            = mc.activation,
        downsample_mode       = mc.downsample_mode,
        stem_mode             = mc.stem_mode,
        spatial_dims          = spatial_dims,
        num_stem_fusion_views = num_stem_fusion_views,
        stem_fusion_mode      = mc.stem_fusion_mode,
        in_ch_per_view_list   = in_ch_per_view_list,
        cond_in_channels      = cond_in_channels,
        downsample_builder    = downsample_builder,
        downsample_strides    = ds_strides,
        grad_checkpointing    = mc.grad_checkpointing,
        grad_ckpt_stages      = mc.grad_ckpt_encoder_stages)

    if mc.decoder_type == "unet3p":
        decoder = UNet3PDecoder(
            encoder_channels=enc_channels,
            cat_channels=mc.unet3p_cat_channels,
            norm_type=mc.norm_type,
            norm_groups=mc.norm_groups,
            activation=mc.activation,
            skip_attention=mc.skip_attention,
            spatial_dims=spatial_dims,
            grad_checkpointing=mc.grad_checkpointing)
    elif mc.decoder_type == "unetpp":
        decoder = UNetPPDecoder(
            encoder_channels=enc_channels,
            stage_builder=dec_builder,
            upsample_mode=mc.upsample_mode,
            skip_attention=mc.skip_attention,
            # 生成主线历史语义：门控上采样分支（分割主线为门控 skips）。
            attn_gate_target="upsample",
            spatial_dims=spatial_dims,
            grad_checkpointing=mc.grad_checkpointing)
    else:
        decoder = Decoder(
            encoder_channels   = enc_channels,
            stage_builder      = dec_builder,
            upsample_mode      = mc.upsample_mode,
            skip_mode          = mc.skip_mode,
            skip_attention     = mc.skip_attention,
            spatial_dims       = spatial_dims,
            downsample_strides = ds_strides,
            grad_checkpointing = mc.grad_checkpointing)

    aux_seg_supervision = topo.aux_seg_active
    aux_head_out_channels_arg = (
        aux_head_out_channels if (aux_seg_supervision and aux_head_out_channels) else None)
    model = UNet3D(
        encoder               = encoder,
        decoder               = decoder,
        out_channels          = out_classes,
        deep_supervision      = mc.deep_supervision,
        spatial_dims          = spatial_dims,
        aux_seg_supervision   = aux_seg_supervision,
        aux_head_mode         = mc.aux_head_mode,
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
        topo.num_res_groups if topo.num_res_groups > 0 else 1,
        mc.stem_mode, encoder.stem_stride,
        num_stem_fusion_views, mc.stem_fusion_mode,
        mc.downsample_mode, mc.upsample_mode, mc.skip_mode,
        mc.attention_type, mc.skip_attention,
        mc.deep_supervision, aux_seg_supervision, len(model.aux_heads),
        mc.aux_head_mode)

    return model


def _model_axis_scales(cfg: Config, spatial_dims: int) -> List[int]:
    """模型空间轴的逐轴超分倍率（与退化算子 axis_scales 一致）。"""
    per_axis = list(cfg.task.sr_scale_per_axis)
    if per_axis:
        return [int(s) for s in per_axis]
    return [int(cfg.task.sr_scale)] * spatial_dims


def _build_sisr_backbone(cfg: Config):
    """构建经典 SISR backbone（EDSR / RCAN，post-upsampling）。

    输入为真 LR 网格（配套 ``SuperResDegradation(keep_lr_size=True)``），
    上采头按逐轴倍率把特征放大回 HR 网格。
    """
    from .sisr import SISRNet

    mc = cfg.model
    topo = build_topology(cfg)
    factors = _model_axis_scales(cfg, topo.spatial_dims)
    model = SISRNet(
        in_channels  = mc.in_channels,
        out_channels = topo.out_classes,
        factors      = factors,
        arch         = str(mc.arch).lower(),
        channels     = mc.sisr_channels,
        num_blocks   = mc.sisr_num_blocks,
        num_groups   = mc.sisr_num_groups,
        res_scale    = mc.sisr_res_scale,
        activation   = mc.activation,
        se_reduction = mc.se_reduction,
        spatial_dims = topo.spatial_dims)
    logger.info(
        "Built SISRNet [%s]: total=%.2fM, channels=%d, blocks=%d, groups=%d, "
        "factors=%s, in_ch=%d, out_ch=%d, spatial_dims=%d",
        mc.arch, model.param_count()["total"] / 1e6, mc.sisr_channels,
        mc.sisr_num_blocks, mc.sisr_num_groups, factors,
        mc.in_channels, topo.out_classes, topo.spatial_dims)
    return model


def build_backbone(cfg: Config):
    """按 `model.arch` 构造共享 backbone：UNet、ADM、EDM2、EDSR/RCAN。"""
    arch = str(cfg.model.arch).lower()
    if cfg.model.grad_checkpointing and arch in ("edsr", "rcan"):
        # SISR 网络较浅，未接检查点；UNet/ADM/EDM2 均已逐块包装。
        logger.warning(
            "model.grad_checkpointing=True is not supported for arch=%r "
            "(only 'unet' | 'adm' | 'edm2'); ignored.", arch)
    if arch in ("edsr", "rcan"):
        return _build_sisr_backbone(cfg)
    if arch == "adm":
        from taskcore.models.adm_unet import build_adm_backbone
        return build_adm_backbone(cfg)
    if arch == "edm2":
        from taskcore.models.edm2_unet import build_edm2_backbone
        return build_edm2_backbone(cfg)
    if arch != "unet":
        raise ValueError(
            f"Unknown model.arch: {arch!r}. "
            f"Valid: 'unet' | 'adm' | 'edm2' | 'edsr' | 'rcan'.")
    return _build_unet_backbone(cfg)
