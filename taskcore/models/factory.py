"""根据 config 构建 UNet3D / ADM / EDM2 模型。"""

from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np
import torch.nn as nn

from ..config.core import Config, ConfigError, resolve_selfattn_stage
from ..config.geometry import (
    auto_anisotropic_strides,
    compute_downsample_strides,
    decoder_stage_count,
    stem_stride_of,
)
from .blocks import DySample3d, SelfAttentionBlock, Upsample
from .convnext import ConvNeXtDownsample, ConvNeXtStage
from .mednext import MedNeXtStage
from .resnet import MultiRFStage, ResNetStage
from .topology import build_topology
from .unet import Encoder, Decoder, UNet3D
from .unet3p import UNet3PDecoder
from .unetpp import UNetPPDecoder

logger = logging.getLogger(__name__)


def _custom_init_param_ids(model: nn.Module) -> set:
    """收集带自定义初始化契约的参数 id：SelfAttentionBlock（zero-init 残差出口）、
    DySample3d（offset/scope 近零）、pixelshuffle Upsample 的 ICNR expand 卷积。
    非 legacy 策略不得覆盖这些初始化，否则破坏其设计意图。"""
    ids: set = set()
    for module in model.modules():
        if isinstance(module, (SelfAttentionBlock, DySample3d)):
            ids.update(id(p) for p in module.parameters())
        elif isinstance(module, Upsample) and module.mode == "pixelshuffle":
            ids.update(id(p) for p in module.expand.parameters())
    return ids


def _apply_init_strategy(model: nn.Module, strategy: str) -> nn.Module:
    """按显式策略覆盖模型初始化；legacy 保留各模块既有初始化。"""
    strategy = str(strategy).lower()
    if strategy == "legacy":
        return model
    protected = _custom_init_param_ids(model)
    for module in model.modules():
        if any(id(p) in protected
               for p in module.parameters(recurse=False)):
            continue
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                               nn.Linear)):
            if strategy == "kaiming":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                 nn.BatchNorm3d, nn.GroupNorm,
                                 nn.LayerNorm, nn.InstanceNorm1d,
                                 nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


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


def _decoder_call_count(decoder_type: str, n_levels: int) -> int:
    """返回 decoder 实际构造的 stage/node 数（委托 config.geometry 单一口径）。"""
    return decoder_stage_count(decoder_type, n_levels)


def _resolve_decoder_block_counts(mc, n_levels: int) -> List[int]:
    expected = _decoder_call_count(mc.unet.decoder_type, n_levels)
    values = list(mc.decoder_blocks_per_stage or [])
    if expected == 0 and len(values) > 1:
        # unet3p 等 decoder 不消费逐级 block 数；显式列表仅告警后忽略。
        logger.warning(
            "decoder_blocks_per_stage=%s is not consumed by "
            "decoder_type=%r; ignored.", values, mc.unet.decoder_type)
        values = []
    if not values or len(values) == 1:
        value = values[0] if values else mc.blocks_per_level
        return [int(value)] * max(expected, 1)
    return _resolve_blocks_per_stage(values, expected, mc.blocks_per_level)


def _make_drop_path_rates(counts: List[int], drop_path_rate: float) -> List[float]:
    """按总 block 深度线性增长的 stochastic depth 率。"""
    total_blocks = sum(counts)
    return np.linspace(0, drop_path_rate, max(total_blocks, 1)).tolist()


class _StatefulStageBuilder:
    """有状态 stage 构建器：逐次调用从 counts[idx] 读 num_blocks。

    stage 索引由本类单一计数器维护并传给 factory_fn，避免 factory 闭包
    另设计数器产生双计数器漂移。"""

    def __init__(self, factory_fn, counts: List[int]):
        self._fn     = factory_fn
        self._counts = counts
        self._idx    = 0

    def __call__(self, in_ch: int, out_ch: int):
        if self._idx >= len(self._counts):
            raise RuntimeError(
                f"StageBuilder exhausted after {self._idx} calls, "
                f"counts={self._counts}")
        idx = self._idx
        n_blocks = self._counts[idx]
        self._idx += 1
        return self._fn(in_ch, out_ch, n_blocks, idx)


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
    attention_type = mc.unet.attention_type
    dp_rates = _make_drop_path_rates(counts, mc.unet.drop_path_rate)
    mask = list(multirf_mask) if multirf_mask else []
    sa_types = list(selfattn_types) if selfattn_types else []

    def _build_stage(
        in_ch: int, out_ch: int, num_blocks: int, start: int, stage_idx: int):
        use_mrf = bool(mask[stage_idx]) if stage_idx < len(mask) else False
        if use_mrf:
            return MultiRFStage(
                in_ch, out_ch,
                num_blocks     = num_blocks,
                dilations      = list(mc.unet.multirf.dilations),
                mode           = mc.unet.multirf.mode,
                fusion         = mc.unet.multirf.fusion,
                axes           = mc.unet.multirf.axes,
                norm_type      = mc.unet.norm_type,
                norm_groups    = mc.unet.norm_groups,
                activation     = mc.unet.activation,
                dropout        = mc.dropout,
                se_reduction   = mc.unet.se_reduction,
                attention_type = attention_type,
                drop_path_rates = dp_rates[start:start + num_blocks],
                spatial_dims   = spatial_dims,
                branch_norm_act = mc.unet.multirf.branch_norm_act)
        return ResNetStage(
            in_ch, out_ch,
            num_blocks     = num_blocks,
            norm_type      = mc.unet.norm_type,
            norm_groups    = mc.unet.norm_groups,
            activation     = mc.unet.activation,
            dropout        = mc.dropout,
            se_reduction   = mc.unet.se_reduction,
            attention_type = attention_type,
            block_type     = mc.unet.block_type,
            spatial_dims   = spatial_dims,
            drop_path_rates = dp_rates[start:start + num_blocks])

    def factory(in_ch: int, out_ch: int, num_blocks: int, stage_idx: int):
        # stage_idx 由 _StatefulStageBuilder 的单一计数器传入。
        start = sum(counts[:stage_idx])
        stage = _build_stage(in_ch, out_ch, num_blocks, start, stage_idx)
        sa_type = sa_types[stage_idx] if stage_idx < len(sa_types) else None
        if sa_type:
            return nn.Sequential(stage, SelfAttentionBlock(
                out_ch,
                attn_type    = sa_type,
                num_heads    = mc.unet.selfattn.num_heads,
                head_dim     = mc.unet.selfattn.head_dim,
                norm_groups  = mc.unet.norm_groups,
                zero_init    = mc.unet.selfattn.zero_init,
                use_rope     = mc.unet.selfattn.rope,
                use_ffn      = mc.unet.selfattn.ffn,
                ffn_ratio    = mc.unet.selfattn.ffn_ratio,
                window_size  = mc.unet.selfattn.window_size,
                grid_size    = mc.unet.selfattn.grid_size,
                spatial_dims = spatial_dims))
        return stage

    return _StatefulStageBuilder(factory, counts)


def _make_convnext_stage_builder(cfg: Config, counts: List[int]) -> _StatefulStageBuilder:
    """ConvNeXt stage 构建器：块内硬编码 LN+GELU；用户设其他 norm/act 时警告。"""
    mc = cfg.model
    spatial_dims = mc.spatial_dims
    non_default = []
    if mc.unet.norm_type != "instance":
        non_default.append(f"norm_type={mc.unet.norm_type!r}")
    if mc.unet.activation != "leakyrelu":
        non_default.append(f"activation={mc.unet.activation!r}")
    if mc.dropout and mc.dropout > 0.0:
        non_default.append(f"dropout={mc.dropout}")
    if non_default:
        logger.warning(
            "Backbone=convnext: block-internal norm/activation are fixed to "
            "LayerNorm+GELU and the following settings are IGNORED inside "
            "ConvNeXt blocks: %s. (They still apply to the stem/decoder "
            "skip projections built in Encoder/Decoder.)",
            ", ".join(non_default))
    dp_rates = _make_drop_path_rates(counts, mc.unet.drop_path_rate)
    ls_init      = float(mc.unet.convnext_layer_scale_init)  # <=0 禁用

    def factory(
        in_ch: int, out_ch: int, num_blocks: int, stage_idx: int
        ) -> ConvNeXtStage:
        start = sum(counts[:stage_idx])
        end   = start + num_blocks
        rates = dp_rates[start:end] if dp_rates else [0.0] * num_blocks
        return ConvNeXtStage(
            in_ch, out_ch,
            num_blocks             = num_blocks,
            drop_path_rates        = rates,
            attention_type         = mc.unet.attention_type,
            use_grn                = mc.unet.grn_enabled,
            spatial_dims           = spatial_dims,
            layer_scale_init_value = ls_init,
            attn_reduction         = mc.unet.se_reduction)

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
    if mc.unet.norm_type != "group":
        non_default.append(f"norm_type={mc.unet.norm_type}")
    if mc.unet.activation != "gelu":
        non_default.append(f"activation={mc.unet.activation}")
    if mc.dropout and mc.dropout > 0:
        non_default.append(f"dropout={mc.dropout}")
    if non_default:
        logger.warning(
            "Backbone=mednext: block-internal norm/activation are fixed to "
            "channel-wise GroupNorm+GELU and the following settings are "
            "IGNORED inside MedNeXt blocks: %s. (They still apply to the "
            "stem/decoder skip projections built in Encoder/Decoder.)",
            ", ".join(non_default))
    dp_rates = _make_drop_path_rates(counts, mc.unet.drop_path_rate)

    def factory(
        in_ch: int, out_ch: int, num_blocks: int, stage_idx: int
        ) -> MedNeXtStage:
        start = sum(counts[:stage_idx])
        end = start + num_blocks
        rates = dp_rates[start:end] if dp_rates else [0.0] * num_blocks
        return MedNeXtStage(
            in_ch, out_ch,
            num_blocks     = num_blocks,
            expand_ratio   = mc.unet.mednext.expand_ratio,
            kernel_size    = mc.unet.mednext.kernel_size,
            drop_path_rates = rates,
            attention_type = mc.unet.attention_type,
            use_grn        = mc.unet.grn_enabled,
            spatial_dims   = spatial_dims,
            attn_reduction = mc.unet.se_reduction,
            dilated_reparam = mc.unet.mednext.dilated_reparam,
            dilated_reparam_branch_kernel_sizes = (
                mc.unet.mednext.dilated_reparam_kernel_sizes or None),
            dilated_reparam_branch_dilations = (
                mc.unet.mednext.dilated_reparam_dilations or None))

    return _StatefulStageBuilder(factory, counts)


# 各向异性下采样兼容的下/上采样模式（其余模式核结构要求各向同性 2）。
_ANISO_DOWN_MODES = ("conv", "maxpool", "avgpool")
_ANISO_UP_MODES   = ("transpose", "trilinear", "nearest")


# Backward-compatible private names used by existing tests and callers.
_stem_stride_of = stem_stride_of
_auto_anisotropic_strides = auto_anisotropic_strides


def _build_unet_encoder_decoder(
    cfg: Config,
    *,
    attn_gate_target: str = "skips",
):
    """构造 UNet 家族 ``(encoder, decoder)``，不含分割头 / deep-sup / aux。

    cls / det / ssl 骨干装配与 ``build_model`` 共用此路径，保证
    ``encoder.*`` / ``decoder.*`` 同名同形；任务头由各任务自行挂载。
    """
    if str(cfg.model.arch).lower() != "unet":
        raise ValueError(
            f"_build_unet_encoder_decoder requires model.arch=='unet'; "
            f"got {cfg.model.arch!r}.")
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

    if attn_gate_target != "skips" and mc.unet.decoder_type != "unetpp":
        raise ConfigError(
            f"attn_gate_target={attn_gate_target!r} is unsupported for "
            f"decoder_type={mc.unet.decoder_type!r}; only UNet++ consumes it.")
    dec_counts = _resolve_decoder_block_counts(mc, n_levels)

    # MultiRF 逐 stage mask（默认空 = 全关，逐位兼容历史）。
    # encoder mask 对齐 enc stage 顺序（浅→深，末位为 bottleneck）；
    # decoder mask 对齐 decoder level 顺序（深→浅，见 Decoder 构造），仅 unet decoder。
    enc_mrf_mask: List[bool] = []
    dec_mrf_mask: List[bool] = []
    if mc.unet.multirf.enabled:
        enc_mrf_mask = [bool(int(v)) for v in mc.unet.multirf.encoder_stages]
        # decoder 侧 MultiRF 仅 unet decoder 有 mask 通路；unetpp/unet3p 未接。
        if mc.unet.decoder_type == "unet":
            dec_mrf_mask = [bool(int(v)) for v in mc.unet.multirf.decoder_stages]

    # SelfAttention 逐 stage 类型（默认空 = 全关，逐位兼容历史）。对齐顺序同 MultiRF；
    # 每层解析为 'softmax'/'linear'/None，支持同一网络不同层用不同注意力。
    enc_sa_types: List = []
    dec_sa_types: List = []
    if mc.unet.selfattn.enabled:
        enc_sa_types = [resolve_selfattn_stage(v, mc.unet.selfattn.type)
                        for v in mc.unet.selfattn.encoder_stages]
        if mc.unet.decoder_type == "unet":
            dec_sa_types = [resolve_selfattn_stage(v, mc.unet.selfattn.type)
                            for v in mc.unet.selfattn.decoder_stages]

    # enc/dec 分别构建以使计数独立。
    downsample_builder = None
    if   mc.unet.backbone == "resnet":
        enc_builder = _make_resnet_stage_builder(
            cfg, enc_counts, enc_mrf_mask, enc_sa_types)
        dec_builder = _make_resnet_stage_builder(
            cfg, dec_counts, dec_mrf_mask, dec_sa_types)
        if mc.unet.multirf.enabled and (any(enc_mrf_mask) or any(dec_mrf_mask)):
            logger.info(
                "MultiRF ENABLED: dilations=%s, mode=%s, fusion=%s, axes=%s, "
                "enc_stages=%s, dec_stages=%s",
                list(mc.unet.multirf.dilations), mc.unet.multirf.mode, mc.unet.multirf.fusion,
                mc.unet.multirf.axes,
                [int(b) for b in enc_mrf_mask], [int(b) for b in dec_mrf_mask])
        if mc.unet.selfattn.enabled and (any(enc_sa_types) or any(dec_sa_types)):
            logger.info(
                "SelfAttention ENABLED: default_type=%s, num_heads=%s, "
                "head_dim=%s, zero_init=%s, enc_types=%s, dec_types=%s",
                mc.unet.selfattn.type, mc.unet.selfattn.num_heads, mc.unet.selfattn.head_dim,
                mc.unet.selfattn.zero_init, enc_sa_types, dec_sa_types)
    elif mc.unet.backbone == "convnext":
        enc_builder = _make_convnext_stage_builder(cfg, enc_counts)
        dec_builder = _make_convnext_stage_builder(cfg, dec_counts)
        # LN-first 下采样；置 False 回退通用 Downsample（消融实验）。
        if bool(mc.unet.convnext_downsample_lnfirst):
            downsample_builder = _make_convnext_downsample_builder(cfg)
    elif mc.unet.backbone == "mednext":
        # 档位 A：MedNeXt 残差倒瓶颈块 + 通用 Downsample/Upsample（无自定义 downsample_builder）。
        enc_builder = _make_mednext_stage_builder(cfg, enc_counts)
        dec_builder = _make_mednext_stage_builder(cfg, dec_counts)
    else:
        raise ValueError(f"Unknown backbone: {mc.unet.backbone}")

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
        if mc.unet.decoder_type != "unet":
            raise ValueError(
                f"Anisotropic downsampling currently supports only "
                f"decoder_type='unet'; got {mc.unet.decoder_type!r}. "
                f"(unetpp/unet3p decoders use isotropic ×2 up/down.)")
        if mc.stem_fusion_mode == "hierarchical" and num_stem_fusion_views > 1:
            # hierarchical aux-stem 注入尺寸假定各级各向同性 ×2 下采；
            # 构造期报错，避免 forward 期才因空间尺寸不匹配失败。
            raise ValueError(
                "Anisotropic downsampling is not supported with "
                "stem_fusion_mode='hierarchical': aux-view injection sizes "
                "assume isotropic x2 encoder downsampling. Use "
                "'shared_stem'/'multi_stem_proj', or disable "
                "anisotropic_pooling/downsample_strides.")
        if mc.unet.downsample_mode not in _ANISO_DOWN_MODES:
            raise ValueError(
                f"Anisotropic downsampling requires downsample_mode in "
                f"{_ANISO_DOWN_MODES}; got {mc.unet.downsample_mode!r}.")
        if mc.unet.upsample_mode not in _ANISO_UP_MODES:
            raise ValueError(
                f"Anisotropic downsampling requires upsample_mode in "
                f"{_ANISO_UP_MODES}; got {mc.unet.upsample_mode!r}.")
        logger.info("Anisotropic downsample strides (per level): %s", ds_strides)

    # Build encoder
    encoder = Encoder(
        in_channels           = topo.in_channels,
        stage_channels        = enc_channels,
        stage_builder         = enc_builder,
        norm_type             = mc.unet.norm_type,
        norm_groups           = mc.unet.norm_groups,
        activation            = mc.unet.activation,
        downsample_mode       = mc.unet.downsample_mode,
        stem_mode             = mc.stem_mode,
        spatial_dims          = spatial_dims,
        num_stem_fusion_views = num_stem_fusion_views,
        stem_fusion_mode      = mc.stem_fusion_mode,
        in_ch_per_view_list   = in_ch_per_view_list,
        cond_in_channels      = topo.cond_in_channels,
        downsample_builder    = downsample_builder,
        downsample_strides    = ds_strides,
        grad_checkpointing    = mc.grad_checkpointing,
        grad_ckpt_stages      = mc.unet.grad_ckpt_encoder_stages,
        grad_ckpt_stem_downsample = mc.grad_ckpt_stem_downsample)

    # attn_gate_norm='auto' 跟随全局 norm_type（避免小 batch 3D 下门控 BN 统计噪）。
    attn_gate_norm = (mc.unet.norm_type if mc.unet.attn_gate_norm == "auto"
                      else mc.unet.attn_gate_norm)

    # decoder: unet | unetpp | unet3p
    if   mc.unet.decoder_type == "unet3p":
        decoder = UNet3PDecoder(
            encoder_channels=enc_channels,
            cat_channels=mc.unet.unet3p_cat_channels,
            norm_type=mc.unet.norm_type,
            norm_groups=mc.unet.norm_groups,
            activation=mc.unet.activation,
            skip_attention=mc.unet.skip_attention,
            attn_gate_norm=attn_gate_norm,
            spatial_dims=spatial_dims,
            grad_checkpointing=mc.grad_checkpointing,
            grad_ckpt_decoder_branches=mc.grad_ckpt_decoder_branches)
    elif mc.unet.decoder_type == "unetpp":
        decoder = UNetPPDecoder(
            encoder_channels=enc_channels,
            stage_builder=dec_builder,
            upsample_mode=mc.unet.upsample_mode,
            skip_attention=mc.unet.skip_attention,
            attn_gate_norm=attn_gate_norm,
            attn_gate_target=attn_gate_target,
            spatial_dims=spatial_dims,
            upsample_norm_act=mc.unet.upsample_norm_act,
            norm_type=mc.unet.norm_type,
            norm_groups=mc.unet.norm_groups,
            activation=mc.unet.activation,
            grad_checkpointing=mc.grad_checkpointing,
            grad_ckpt_decoder_branches=mc.grad_ckpt_decoder_branches,
            upsample_interp_dtype=mc.unet.upsample_interp_dtype)
    else:
        decoder = Decoder(
            encoder_channels   = enc_channels,
            stage_builder      = dec_builder,
            upsample_mode      = mc.unet.upsample_mode,
            skip_mode          = mc.unet.skip_mode,
            skip_attention     = mc.unet.skip_attention,
            attn_gate_norm     = attn_gate_norm,
            spatial_dims       = spatial_dims,
            downsample_strides = ds_strides,
            upsample_norm_act  = mc.unet.upsample_norm_act,
            norm_type          = mc.unet.norm_type,
            norm_groups        = mc.unet.norm_groups,
            activation         = mc.unet.activation,
            grad_checkpointing = mc.grad_checkpointing,
            upsample_interp_dtype = mc.unet.upsample_interp_dtype)

    return encoder, decoder


def build_backbone(
    cfg: Config,
    *,
    with_decoder: bool = False,
    attn_gate_target: str = "skips",
):
    """骨干专用入口：只建 encoder（及可选 decoder），不建任务头。

    * ``with_decoder=False``（默认）→ 返回 ``encoder``（cls / DINO / BYOL…）
    * ``with_decoder=True`` → 返回 ``(encoder, decoder)``（det / SSL 重建）

    仅支持 ``model.arch=='unet'``；adm/edm2 请继续用 ``build_model``。
    Encoder ``forward`` 返回逐级特征列表；末级通道 =
    ``cfg.model.encoder_channels[-1]``；Decoder 暴露 ``out_channels``
    （深→浅）。
    """
    encoder, decoder = _build_unet_encoder_decoder(
        cfg, attn_gate_target=attn_gate_target)
    if with_decoder:
        return encoder, decoder
    return encoder


def build_model(cfg: Config, *, attn_gate_target: str = "skips"):
    """按 cfg.model.arch 分派：'unet' 默认 或 'adm' | 'edm2'
    （后者忽略大多数 backbone/block 选项，使用论文原保 GN+SiLU / MP）。"""
    arch = str(cfg.model.arch).lower()
    strategy = str(cfg.model.init_strategy).lower()
    if strategy != "legacy" and arch != "unet":
        raise ConfigError(
            "model.init_strategy=%r is only supported with "
            "model.arch='unet'; ADM/EDM2 use architecture-specific "
            "initialization contracts." % cfg.model.init_strategy)
    if arch == "adm":
        from .adm_unet import build_adm_seg_model
        return _apply_init_strategy(
            build_adm_seg_model(cfg), cfg.model.init_strategy)
    if arch == "edm2":
        from .edm2_unet import build_edm2_seg_model
        return _apply_init_strategy(
            build_edm2_seg_model(cfg), cfg.model.init_strategy)
    if arch != "unet":
        raise ValueError(
            f"Unknown model.arch: {arch!r}. Valid: 'unet' | 'adm' | 'edm2'.")

    encoder, decoder = _build_unet_encoder_decoder(
        cfg, attn_gate_target=attn_gate_target)
    mc = cfg.model
    topo = build_topology(cfg)
    spatial_dims = topo.spatial_dims
    out_classes = topo.out_classes
    num_fg = cfg.num_fg_classes
    enc_channels = list(mc.encoder_channels)
    n_levels = len(enc_channels)
    enc_counts = _resolve_blocks_per_stage(
        mc.encoder_blocks_per_stage, n_levels, mc.blocks_per_level)
    dec_counts = _resolve_decoder_block_counts(mc, n_levels)
    num_stem_fusion_views = topo.num_stem_fusion_views
    aux_head_out_channels = topo.aux_head_out_channels

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
        aux_topo_head         = mc.unet.aux_topo_head,
        aux_topo_head_mode    = mc.unet.aux_topo_head_mode,
        norm_type             = mc.unet.norm_type,
        norm_groups           = mc.unet.norm_groups,
        activation            = mc.unet.activation)

    pc = model.param_count()
    logger.info(
        "Built UNet3D [%s/%s, decoder=%s, preset=%s]: "
        "enc=%.2fM, dec=%.2fM, total=%.2fM, channels=%s, "
        "enc_blocks=%s, dec_blocks=%s, out_classes=%d (fg=%d, res=%d), "
        "stem=%s(stride=%d, n_views=%d, fusion=%s), "
        "down=%s, up=%s, skip=%s, attn=%s, skip_attn=%s, "
        "ds=%s, aux_seg=%s(n_aux_heads=%d, mode=%s), grad_ckpt=%s",
        mc.unet.backbone, mc.unet.block_type, mc.unet.decoder_type, mc.resenc_preset,
        pc["encoder"] / 1e6, pc["decoder"] / 1e6, pc["total"] / 1e6,
        enc_channels,
        enc_counts, dec_counts,
        out_classes, num_fg,
        topo.num_res_groups if topo.num_res_groups > 0 else 1,
        mc.stem_mode, encoder.stem_stride,
        num_stem_fusion_views, mc.stem_fusion_mode,
        mc.unet.downsample_mode, mc.unet.upsample_mode, mc.unet.skip_mode,
        mc.unet.attention_type, mc.unet.skip_attention,
        mc.deep_supervision, aux_seg_supervision, len(model.aux_heads),
        mc.aux_head_mode, mc.grad_checkpointing)

    return _apply_init_strategy(model, mc.init_strategy)
