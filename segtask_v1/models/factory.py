"""根据 config 构建 UNet3D / ADM / EDM2 模型。"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, List

import numpy as np
import torch.nn as nn

from ..config import Config, resolve_selfattn_stage
from .blocks import Downsample, SelfAttentionBlock, Upsample
from .convnext import ConvNeXtDownsample, ConvNeXtStage
from .mednext import MedNeXtStage
from .resnet import MultiRFStage, ResNetStage
from .topology import ModelTopology, build_topology
from .unet import Encoder, Decoder, UNet3D
from .unet3p import UNet3PDecoder
from .unetpp import UNetPPDecoder

logger = logging.getLogger(__name__)


def _resolve_blocks_per_stage(
    explicit: List[int],
    n_stages: int,
    fallback: int) -> List[int]:
    """逐级 block 数：显式列表优先，否则广播 fallback。"""
    if explicit:
        if len(explicit) != n_stages:
            raise ValueError(
                f"Per-stage block list length {len(explicit)} "
                f"!= expected {n_stages}")
        return list(explicit)
    return [fallback] * n_stages


class _StatefulStageBuilder:
    """有状态 stage 构建器：逐次调用从 counts[idx] 读 num_blocks。"""

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


def _make_resnet_stage_builder(
    cfg: Config,
    counts: List[int],
    multirf_mask: List[bool] = None,
    selfattn_types: List = None) -> _StatefulStageBuilder:
    """返回按逐级 block 数构建 ResNet stage 的有状态函数。

    ``multirf_mask`` 非空时，逐 stage 决定该 stage 是否用 ``MultiRFStage``（多感受野
    空洞分支块）替代标准 ``ResNetStage``；长度须与 ``counts`` 对齐。``None``/全 False
    时行为与历史一致（全部 ``ResNetStage``）。

    ``selfattn_types`` 非空时，逐 stage 给出该 stage 末尾追加的 ``SelfAttentionBlock``
    类型（``'softmax'``/``'linear'``，``None`` 为该层不加）；与 ``multirf_mask`` 正交可叠加。
    """
    mc = cfg.model
    spatial_dims = mc.spatial_dims
    attention_type = mc.attention_type
    mask = list(multirf_mask) if multirf_mask else []
    sa_types = list(selfattn_types) if selfattn_types else []

    def _build_stage(in_ch: int, out_ch: int, num_blocks: int):
        use_mrf = bool(mask[factory_idx[0]]) if factory_idx[0] < len(mask) else False
        if use_mrf:
            return MultiRFStage(
                in_ch, out_ch,
                num_blocks     = num_blocks,
                dilations      = list(mc.multirf_dilations),
                mode           = mc.multirf_mode,
                fusion         = mc.multirf_fusion,
                axes           = mc.multirf_axes,
                norm_type      = mc.norm_type,
                norm_groups    = mc.norm_groups,
                activation     = mc.activation,
                dropout        = mc.dropout,
                se_reduction   = mc.se_reduction,
                attention_type = attention_type,
                spatial_dims   = spatial_dims,
                branch_norm_act = mc.multirf_branch_norm_act)
        return ResNetStage(
            in_ch, out_ch,
            num_blocks     = num_blocks,
            norm_type      = mc.norm_type,
            norm_groups    = mc.norm_groups,
            activation     = mc.activation,
            dropout        = mc.dropout,
            se_reduction   = mc.se_reduction,
            attention_type = attention_type,
            block_type     = mc.block_type,
            spatial_dims   = spatial_dims)

    def factory(in_ch: int, out_ch: int, num_blocks: int):
        # _StatefulStageBuilder 顺序调用，下方按"已构建数"取 mask。
        idx = factory_idx[0]
        stage = _build_stage(in_ch, out_ch, num_blocks)
        sa_type = sa_types[idx] if idx < len(sa_types) else None
        factory_idx[0] += 1
        if sa_type:
            return nn.Sequential(stage, SelfAttentionBlock(
                out_ch,
                attn_type    = sa_type,
                num_heads    = mc.selfattn_num_heads,
                head_dim     = mc.selfattn_head_dim,
                norm_groups  = mc.norm_groups,
                zero_init    = mc.selfattn_zero_init,
                spatial_dims = spatial_dims))
        return stage

    factory_idx = [0]
    return _StatefulStageBuilder(factory, counts)


def _make_convnext_stage_builder(cfg: Config, counts: List[int]) -> _StatefulStageBuilder:
    """ConvNeXt stage 构建器：块内硬编码 LN+GELU；用户设其他 norm/act 时警告。"""
    mc = cfg.model
    spatial_dims = mc.spatial_dims
    non_default = []
    if mc.norm_type != "instance":
        non_default.append(f"norm_type={mc.norm_type!r}")
    if mc.activation != "leakyrelu":
        non_default.append(f"activation={mc.activation!r}")
    if mc.dropout and mc.dropout > 0.0:
        non_default.append(f"dropout={mc.dropout}")
    if non_default:
        logger.warning(
            "Backbone=convnext: block-internal norm/activation are fixed to "
            "LayerNorm+GELU and the following settings are IGNORED inside "
            "ConvNeXt blocks: %s. (They still apply to the stem/decoder "
            "skip projections built in Encoder/Decoder.)",
            ", ".join(non_default))
    # 总 block 上线性增的 drop-path。
    total_blocks = sum(counts)
    dp_rates     = np.linspace(0, mc.drop_path_rate, max(total_blocks, 1)).tolist()
    rate_idx     = [0]
    ls_init      = float(mc.convnext_layer_scale_init)  # <=0 禁用

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
    """论文风 ConvNeXt 阶间下采样 LN→Conv(s=2) 构建器。"""
    spatial_dims = cfg.model.spatial_dims

    def build(in_ch: int, out_ch: int) -> ConvNeXtDownsample:
        return ConvNeXtDownsample(in_ch, out_ch, spatial_dims=spatial_dims)

    return build


def _make_mednext_stage_builder(cfg: Config, counts: List[int]) -> _StatefulStageBuilder:
    """MedNeXt stage 构建器：块内固定通道级 GroupNorm + GELU；用户设其他 norm/act/dropout 时警告。

    重采样复用通用 Downsample/Upsample（``downsample_mode`` / ``upsample_mode`` 决定），
    与 ConvNeXt 的 LN-first 阶间下采样不同——故 ``anisotropic_pooling`` 仍可用。
    """
    mc = cfg.model
    spatial_dims = mc.spatial_dims
    non_default = []
    if mc.norm_type != "group":
        non_default.append(f"norm_type={mc.norm_type}")
    if mc.activation != "gelu":
        non_default.append(f"activation={mc.activation}")
    if mc.dropout and mc.dropout > 0:
        non_default.append(f"dropout={mc.dropout}")
    if non_default:
        logger.warning(
            "Backbone=mednext: block-internal norm/activation are fixed to "
            "channel-wise GroupNorm+GELU and the following settings are "
            "IGNORED inside MedNeXt blocks: %s. (They still apply to the "
            "stem/decoder skip projections built in Encoder/Decoder.)",
            ", ".join(non_default))

    def factory(in_ch: int, out_ch: int, num_blocks: int) -> MedNeXtStage:
        return MedNeXtStage(
            in_ch, out_ch,
            num_blocks     = num_blocks,
            expand_ratio   = mc.mednext_expand_ratio,
            kernel_size    = mc.mednext_kernel_size,
            attention_type = mc.attention_type,
            spatial_dims   = spatial_dims)

    return _StatefulStageBuilder(factory, counts)


# 各向异性自动调度的最小特征边长（nnU-Net 默认 4）：降采样后某轴不小于此值才继续降。
_MIN_FEATURE_SIZE = 4

#: 各向异性下采样兼容的下/上采样模式（其余模式核结构要求各向同性 2）。
_ANISO_DOWN_MODES = ("conv", "maxpool", "avgpool")
_ANISO_UP_MODES   = ("transpose", "trilinear", "nearest")


def _stem_stride_of(stem_mode: str) -> int:
    """patchN stem 在进 encoder stage 前先各向同性降 N 倍；其余 stem stride=1。"""
    if stem_mode == "patch2":
        return 2
    if stem_mode == "patch4":
        return 4
    return 1


def _auto_anisotropic_strides(
    spatial_sizes: List[int],
    num_down     : int,
    min_size     : int = _MIN_FEATURE_SIZE) -> List[tuple]:
    """nnU-Net 式各向异性调度：逐级仅对"分辨率仍偏大"的轴降采样。

    某轴本级降采样（stride 2）的条件：(a) 当前尺寸为偶数；(b) 减半后仍 >= min_size；
    (c) 当前尺寸 > 本级最大轴尺寸的一半（即该轴分辨率落后不超过 2×）。这样各轴分辨率
    始终保持在彼此 2× 以内，避免薄 z 轴被过早压成 1。
    """
    sizes = [int(s) for s in spatial_sizes]
    nd = len(sizes)
    schedule: List[tuple] = []
    for _ in range(num_down):
        ref = max(sizes)
        stride = []
        for ax in range(nd):
            do_pool = (sizes[ax] % 2 == 0
                       and sizes[ax] // 2 >= min_size
                       and sizes[ax] * 2 > ref)  # sizes[ax] > ref/2
            if do_pool:
                stride.append(2)
                sizes[ax] //= 2
            else:
                stride.append(1)
        schedule.append(tuple(stride))
    return schedule


def compute_downsample_strides(cfg: Config, spatial_dims: int, n_levels: int):
    """决定逐级下采样 stride。

    优先级：显式 ``model.downsample_strides`` > ``model.anisotropic_pooling``
    自动推导 > None（各向同性，沿用历史行为）。返回 ``None`` 或长度 ``n_levels-1``
    的 per-axis stride 元组列表。
    """
    mc       = cfg.model
    num_down = n_levels - 1
    if num_down <= 0:
        return None

    explicit = list(mc.downsample_strides or [])
    if explicit:
        return [tuple(int(x) for x in s) for s in explicit]

    if not bool(mc.anisotropic_pooling):
        return None  # 各向同性默认：Downsample/Upsample 用 stride=2

    # 自动推导：基于 patch 的"模型空间轴"尺寸（2.5D 的 D 折进通道，不计）。
    patch         = [int(x) for x in cfg.data.patch_size]  # [D, H, W]
    spatial_sizes = patch[1:] if spatial_dims == 2 else patch
    stem_stride   = _stem_stride_of(mc.stem_mode)
    spatial_sizes = [max(1, s // stem_stride) for s in spatial_sizes]
    return _auto_anisotropic_strides(spatial_sizes, num_down)


def build_model(cfg: Config):
    """按 cfg.model.arch 分派：'unet' 默认 或 'adm' | 'edm2'
    （后者忽略大多数 backbone/block 选项，使用论文原保 GN+SiLU / MP）。"""
    arch = str(cfg.model.arch).lower()
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

    topo                  = build_topology(cfg)
    spatial_dims          = topo.spatial_dims
    out_classes           = topo.out_classes
    num_stem_fusion_views = topo.num_stem_fusion_views  # 2.5D才需要融合，3D是通道拼接
    in_ch_per_view_list   = topo.in_ch_per_view_list    # 2.5D才有每个视图的in_ch=depth
    aux_head_out_channels = topo.aux_head_out_channels  # 2.5D才有aux监督选项，3D是必须

    enc_counts = _resolve_blocks_per_stage(  # 确认enc各stage通道
        mc.encoder_blocks_per_stage, n_levels, mc.blocks_per_level)

    if   mc.decoder_type == "unet":
        expected_dec_calls = n_levels - 1
    elif mc.decoder_type == "unetpp":
        expected_dec_calls = n_levels * (n_levels - 1) // 2
    else:  # unet3p: no stage_builder calls
        expected_dec_calls = 0

    if   mc.decoder_blocks_per_stage and mc.decoder_type == "unet":
        dec_counts = _resolve_blocks_per_stage(  # 确认dec各stage通道
            mc.decoder_blocks_per_stage, expected_dec_calls, mc.blocks_per_level)
    elif mc.decoder_blocks_per_stage:  # UNet++：首项广播到所有嵌套节点
        dec_counts = [mc.decoder_blocks_per_stage[0]] * max(expected_dec_calls, 1)
    else:
        dec_counts = [mc.blocks_per_level] * max(expected_dec_calls, 1)

    # MultiRF 逐 stage mask（默认空 = 全关，逐位兼容历史）。
    # encoder mask 对齐 enc stage 顺序（浅→深，末位为 bottleneck）；
    # decoder mask 对齐 decoder level 顺序（深→浅，见 Decoder 构造），仅 unet decoder。
    enc_mrf_mask: List[bool] = []
    dec_mrf_mask: List[bool] = []
    if mc.multirf_enabled:
        enc_mrf_mask = [bool(int(v)) for v in mc.multirf_encoder_stages]
        if mc.decoder_type == "unet":  # TODO 只有unet才支持吗？
            dec_mrf_mask = [bool(int(v)) for v in mc.multirf_decoder_stages]

    # SelfAttention 逐 stage 类型（默认空 = 全关，逐位兼容历史）。对齐顺序同 MultiRF；
    # 每层解析为 'softmax'/'linear'/None，支持同一网络不同层用不同注意力。
    enc_sa_types: List = []
    dec_sa_types: List = []
    if mc.selfattn_enabled:
        enc_sa_types = [resolve_selfattn_stage(v, mc.selfattn_type)
                        for v in mc.selfattn_encoder_stages]
        if mc.decoder_type == "unet":
            dec_sa_types = [resolve_selfattn_stage(v, mc.selfattn_type)
                            for v in mc.selfattn_decoder_stages]

    # enc/dec 分别构建以使计数独立。
    downsample_builder = None
    if   mc.backbone == "resnet":
        enc_builder = _make_resnet_stage_builder(
            cfg, enc_counts, enc_mrf_mask, enc_sa_types)
        dec_builder = _make_resnet_stage_builder(
            cfg, dec_counts, dec_mrf_mask, dec_sa_types)
        if mc.multirf_enabled and (any(enc_mrf_mask) or any(dec_mrf_mask)):
            logger.info(
                "MultiRF ENABLED: dilations=%s, mode=%s, fusion=%s, axes=%s, "
                "enc_stages=%s, dec_stages=%s",
                list(mc.multirf_dilations), mc.multirf_mode, mc.multirf_fusion,
                mc.multirf_axes,
                [int(b) for b in enc_mrf_mask], [int(b) for b in dec_mrf_mask])
        if mc.selfattn_enabled and (any(enc_sa_types) or any(dec_sa_types)):
            logger.info(
                "SelfAttention ENABLED: default_type=%s, num_heads=%s, "
                "head_dim=%s, zero_init=%s, enc_types=%s, dec_types=%s",
                mc.selfattn_type, mc.selfattn_num_heads, mc.selfattn_head_dim,
                mc.selfattn_zero_init, enc_sa_types, dec_sa_types)
    elif mc.backbone == "convnext":
        enc_builder = _make_convnext_stage_builder(cfg, enc_counts)
        dec_builder = _make_convnext_stage_builder(cfg, dec_counts)
        # LN-first 下采样；置 False 回退通用 Downsample（消融实验）。
        if bool(mc.convnext_downsample_lnfirst):
            downsample_builder = _make_convnext_downsample_builder(cfg)
    elif mc.backbone == "mednext":
        # 档位 A：MedNeXt 残差倒瓶颈块 + 通用 Downsample/Upsample（无自定义 downsample_builder）。
        enc_builder = _make_mednext_stage_builder(cfg, enc_counts)
        dec_builder = _make_mednext_stage_builder(cfg, dec_counts)
    else:
        raise ValueError(f"Unknown backbone: {mc.backbone}")

    # 各向异性下采样 stride 调度（None = 各向同性，沿用历史行为）。
    ds_strides = compute_downsample_strides(cfg, spatial_dims, n_levels)
    if ds_strides is not None and any(
            any(int(s) != 2 for s in stage) for stage in ds_strides):
        # 仅对真正非各向同性的调度做兼容性校验；全 2 的调度等价于默认。
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

    # Build encoder
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
        downsample_builder    = downsample_builder,
        downsample_strides    = ds_strides,
        grad_checkpointing    = mc.grad_checkpointing)

    # decoder: unet | unetpp | unet3p
    if   mc.decoder_type == "unet3p":
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
            spatial_dims=spatial_dims,
            upsample_norm_act=mc.upsample_norm_act,
            norm_type=mc.norm_type,
            norm_groups=mc.norm_groups,
            activation=mc.activation,
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
            upsample_norm_act  = mc.upsample_norm_act,
            norm_type          = mc.norm_type,
            norm_groups        = mc.norm_groups,
            activation         = mc.activation,
            grad_checkpointing = mc.grad_checkpointing)

    # aux 门控统一由 topology 决定（已合并 ``aux_seg_supervision and n_views>1``）。
    aux_seg_supervision       = topo.aux_seg_active
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
        aux_topo_head         = mc.aux_topo_head,
        aux_topo_head_mode    = mc.aux_topo_head_mode,
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
        "ds=%s, aux_seg=%s(n_aux_heads=%d, mode=%s), grad_ckpt=%s",
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
        mc.aux_head_mode, mc.grad_checkpointing)

    return model
