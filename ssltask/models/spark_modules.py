"""方案① SparK-3D 专有模块：掩码-稠密等价编码前向 + 轻量层次解码器。

SparK 的 CNN 原生掩码图像建模。本实现遵守 ``TODO`` 决策 D4「掩码-稠密等价、无新依赖」：
**不引入 spconv/TorchSparse**，而用稠密卷积 + 逐尺度掩码门控来模拟子流形稀疏前向——

* **掩码-稠密等价**：被遮单元在输入处置零，整图进稠密 ``encoder``；每个 stage 输出后
  用对应尺度的掩码重新置空（门控），使被遮位点在各尺度保持为零、感受野不向其渗透
  （近似 SparK「每 stage 后用掩码重置」的稀疏剔除）。
* **满密度退化为稠密**：掩码全 0（全可见）时所有门控 = 恒等，:func:`spark_encode`
  严格等于普通 ``encoder.forward``——故「预训练稀疏前向、下游稠密前向」共享同一套
  ``encoder.*`` 权重，零转换（见单测 ``test_spark_encode_full_density_*``）。

解码端 :class:`SparkLightDecoder` 是一个**比 encoder 窄得多**的对称 UNet（宽度由
``spark_decoder_dim_div`` 控制，SSL.md：参数约 encoder 的 1/5–1/10，用完即弃）：先
densify（被遮位点填入逐尺度可学习 ``mask_embed``），再逐级三线性上采样 + 融合同尺度
encoder 横向特征，最终输出单通道重建。

交接契约：``encoder`` 取自 ``segtask_v1.models.factory.build_model(cfg).encoder``（逐
参数同名同形）；解码器命名为 ``spark_decoder.*``、嵌入为 ``spark_decoder.mask_embed.*``，
**绝不复用** ``decoder.*``/``seg_head.*`` 前缀 → 下游 ``train.pretrain``（strict=False）
仅命中 ``encoder.*``，其余作 unexpected 干净丢弃（MIM 不预训练解码器）。
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.models.blocks import (
    _CONV, INTERP_SMOOTH, ConvNormAct, Upsample)
from segtask_v1.models.factory import build_model

from ..data.masking import densify, downsample_mask_to

logger = logging.getLogger(__name__)


def spark_encode(encoder: nn.Module, x: torch.Tensor, mask_full: torch.Tensor
                 ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """掩码-稠密等价编码：被遮位点置零 + 逐尺度门控，模拟稀疏前向。

    参数
    ----
    encoder   : ``segtask_v1`` 的 :class:`Encoder`（非 hierarchical stem）。
    x         : (B, C, *spatial) 干净输入。
    mask_full : (B, 1, *spatial) {0,1} 全分辨率掩码（1=被遮）。

    返回
    ----
    ``(features, visibles)``：``features[i]`` 为第 i 级门控后的 encoder 特征
    （被遮位点为 0）；``visibles[i]`` 为该尺度可见掩码 (B,1,*scale)（1=可见），供
    解码端 densify 复用。``mask_full`` 全 0 时退化为普通稠密前向。
    """
    if len(getattr(encoder, "aux_fuse", {})) > 0:
        raise NotImplementedError(
            "spark_encode does not support hierarchical stems "
            "(encoder.aux_fuse is non-empty). Use a shared/conv stem for SparK.")

    vis_full = (1.0 - mask_full).to(x.dtype)
    x = x * vis_full                                  # 被遮输入置零（无 mask token）
    x = encoder.stem(x)
    # stem 输出门控（最高分辨率级；与 stage 0 同尺度）。
    vis = downsample_mask_to(vis_full, x.shape[2:])
    x = x * vis

    features: List[torch.Tensor] = []
    visibles: List[torch.Tensor] = []
    for i, stage in enumerate(encoder.stages):
        if i > 0:
            x = encoder.downsamples[i - 1](x)
        x = stage(x)
        vis = downsample_mask_to(vis_full, x.shape[2:])  # 该尺度可见掩码
        x = x * vis                                      # 重新置空被遮位点
        features.append(x)
        visibles.append(vis)
    return features, visibles


class SparkUpLevel(nn.Module):
    """轻量解码上采样级：上采样 → cat 同尺度 densified skip → ConvNormAct 精修。"""

    def __init__(
        self,
        in_ch        : int,
        skip_ch      : int,
        out_ch       : int,
        spatial_dims : int = 3,
        upsample_mode: str = "trilinear",
        upsample_stride=2,
        norm_type    : str = "instance",
        norm_groups  : int = 8,
        activation   : str = "leakyrelu"):
        super().__init__()
        self.upsample = Upsample(
            in_ch, out_ch, mode=upsample_mode, spatial_dims=spatial_dims,
            stride=upsample_stride, norm_act=False,
            norm_type=norm_type, norm_groups=norm_groups, activation=activation)
        self.fuse = ConvNormAct(
            out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError(
                f"SparkUpLevel size mismatch after upsample: "
                f"x={tuple(x.shape[2:])} vs skip={tuple(skip.shape[2:])}. "
                f"Check input spatial dims divisible by total encoder stride.")
        return self.fuse(torch.cat([x, skip], dim=1))


class SparkLightDecoder(nn.Module):
    """轻量层次解码器（用完即弃）：densify → 逐级上采样 + 横向融合 → 单通道重建。

    宽度由 ``dim_div`` 控制（各级通道 = ``max(encoder_channels[k]//dim_div, min_dim)``）。
    自带逐 encoder 尺度的可学习 ``mask_embed``（densify 时填补被遮位点）。
    """

    def __init__(
        self,
        encoder_channels  : List[int],
        out_channels      : int,
        spatial_dims      : int = 3,
        dim_div           : int = 4,
        min_dim           : int = 16,
        upsample_mode     : str = "trilinear",
        downsample_strides=None,
        norm_type         : str = "instance",
        norm_groups       : int = 8,
        activation        : str = "leakyrelu"):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        self.mode = INTERP_SMOOTH[self.spatial_dims]
        n = len(encoder_channels)

        def narrow(c: int) -> int:
            return max(int(c) // int(dim_div), int(min_dim))

        # 逐 encoder 尺度的 densify 嵌入（bottleneck + 各横向），下游丢弃。
        self.mask_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, int(c), *([1] * self.spatial_dims)))
            for c in encoder_channels])
        for p in self.mask_embed:
            nn.init.normal_(p, std=0.02)

        # 镜像 encoder 逐级 stride（解码 level i 还原对应下采样）。
        ds_strides = (list(downsample_strides)
                      if downsample_strides is not None else [2] * (n - 1))
        if len(ds_strides) != n - 1:
            raise ValueError(
                f"downsample_strides length {len(ds_strides)} != "
                f"len(encoder_channels)-1 ({n - 1}).")

        # bottleneck 投影到窄通道。
        bott = narrow(encoder_channels[-1])
        self.bottleneck_proj = ConvNormAct(
            encoder_channels[-1], bott, kernel_size=1, stride=1, padding=0,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=self.spatial_dims)

        self.levels = nn.ModuleList()
        prev = bott
        for i in range(n - 1):
            skip_ch = encoder_channels[n - 2 - i]
            out_ch = narrow(skip_ch)
            up_stride = ds_strides[n - 2 - i]
            self.levels.append(SparkUpLevel(
                prev, skip_ch, out_ch, spatial_dims=self.spatial_dims,
                upsample_mode=upsample_mode, upsample_stride=up_stride,
                norm_type=norm_type, norm_groups=norm_groups,
                activation=activation))
            prev = out_ch

        self.out_conv = _CONV[self.spatial_dims](prev, int(out_channels),
                                                 kernel_size=1)

    def forward(
        self,
        features    : List[torch.Tensor],
        visibles    : List[torch.Tensor],
        target_spatial) -> torch.Tensor:
        # densify 所有尺度（被遮位点填入对应 mask_embed）。
        feats = [densify(f, v, e) for f, v, e in
                 zip(features, visibles, self.mask_embed)]
        n = len(feats)
        x = self.bottleneck_proj(feats[-1])
        for i, level in enumerate(self.levels):
            x = level(x, feats[n - 2 - i])
        out = self.out_conv(x)
        target_spatial = tuple(int(s) for s in target_spatial)
        if out.shape[2:] != target_spatial:           # stem_stride>1 时补回输入分辨率
            out = F.interpolate(out, size=target_spatial, mode=self.mode,
                                align_corners=False)
        return out


class SSLSparkModel(nn.Module):
    """SparK 模型：复用 ``segtask`` encoder + :class:`SparkLightDecoder`。

    forward(x, mask_full): 干净输入 (B,C,*spatial) + 掩码 (B,1,*spatial) → 重建
    (B,C,*spatial)。掩码-稠密等价编码 + 层次解码（densify 后逐级上采样）。下游仅迁移
    ``encoder.*``（``spark_decoder.*`` 作 unexpected 丢弃）。
    """

    def __init__(
        self,
        encoder     : nn.Module,
        decoder     : SparkLightDecoder,
        in_channels : int,
        spatial_dims: int = 3):
        super().__init__()
        self.encoder = encoder
        self.spark_decoder = decoder
        self.spatial_dims = int(spatial_dims)
        self.out_channels = int(in_channels)

    def forward(self, x: torch.Tensor, mask_full: torch.Tensor) -> torch.Tensor:
        features, visibles = spark_encode(self.encoder, x, mask_full)
        out = self.spark_decoder(features, visibles, x.shape[2:])
        if out.shape[2:] != x.shape[2:]:
            raise RuntimeError(
                f"SSL SparK output size mismatch: got {tuple(out.shape[2:])}, "
                f"expected {tuple(x.shape[2:])}. Check encoder downsampling / "
                f"stem_stride vs input spatial dims.")
        return out

    def param_count(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.spark_decoder.parameters())
        return {"encoder": enc, "spark_decoder": dec, "total": enc + dec}


def build_spark_decoder(cfg, encoder: nn.Module, dim_div: int = 4,
                        min_dim: int = 16) -> SparkLightDecoder:
    """在给定 ``encoder`` 上构造轻量层次解码器（供 SparK① 与 ⑧ SparK+DINO 共用）。

    解码器参数由 ``cfg.model`` 与 encoder 的 ``downsample_strides`` 推导；不持有 encoder。
    """
    spatial_dims = int(cfg.model.spatial_dims)
    encoder_channels = [int(c) for c in cfg.model.encoder_channels]
    out_ch = int(cfg.model.in_channels)
    return SparkLightDecoder(
        encoder_channels  = encoder_channels,
        out_channels      = out_ch,
        spatial_dims      = spatial_dims,
        dim_div           = int(dim_div),
        min_dim           = int(min_dim),
        upsample_mode     = cfg.model.upsample_mode,
        downsample_strides= getattr(encoder, "downsample_strides", None),
        norm_type         = cfg.model.norm_type,
        norm_groups       = cfg.model.norm_groups,
        activation        = cfg.model.activation)


def build_ssl_spark_model(cfg, dim_div: int = 4, min_dim: int = 16
                          ) -> SSLSparkModel:
    """构造 SparK 模型：复用 ``build_model`` 的 encoder（保证下游同名同形）。

    仅取 encoder（解码器/分割头丢弃）→ SSL ckpt 经下游 ``train.pretrain``
    （strict=False）命中 ``encoder.*``，``spark_decoder.*``/``decoder.*``/``seg_head.*``
    保持随机或被丢弃（SparK 解码器用完即弃、不迁移）。
    """
    arch = str(cfg.model.arch).lower()
    if arch != "unet":
        raise ValueError(
            f"build_ssl_spark_model requires model.arch=='unet'; got {arch!r}.")
    seg_model = build_model(cfg)                 # 同一构建路径，确保 encoder 同名同形
    encoder = seg_model.encoder
    spatial_dims = int(cfg.model.spatial_dims)
    out_ch = int(cfg.model.in_channels)
    decoder = build_spark_decoder(cfg, encoder, dim_div=dim_div, min_dim=min_dim)
    model = SSLSparkModel(
        encoder=encoder, decoder=decoder, in_channels=out_ch,
        spatial_dims=spatial_dims)
    pc = model.param_count()
    logger.info(
        "Built SSLSparkModel: enc=%.2fM, spark_decoder=%.2fM (dim_div=%d, "
        "~1/%.1f of enc), total=%.2fM, out_channels=%d (=in_channels).",
        pc["encoder"] / 1e6, pc["spark_decoder"] / 1e6, dim_div,
        max(pc["encoder"] / max(pc["spark_decoder"], 1), 1e-9),
        pc["total"] / 1e6, model.out_channels)
    return model


__all__ = [
    "spark_encode", "SparkUpLevel", "SparkLightDecoder",
    "SSLSparkModel", "build_spark_decoder", "build_ssl_spark_model",
]
