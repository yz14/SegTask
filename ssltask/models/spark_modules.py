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

from taskcore.models.blocks import (
    _CONV, INTERP_SMOOTH, ConvNormAct, Upsample)
from taskcore.models.factory import build_model

from ..data.masking import densify, downsample_mask_to

logger = logging.getLogger(__name__)


class _SparkVisHolder:
    """spark 前向期间的当前全分辨率可见掩码（``None`` = 稠密前向）。

    由 :func:`spark_encode` 在每次掩码前向前设置、结束后清空；同一 encoder 的其它
    稠密前向（DINO 分支 / 在线探针 / 下游）看到 ``None``，走原生 InstanceNorm。
    """

    __slots__ = ("vis_full",)

    def __init__(self) -> None:
        self.vis_full = None


class _MaskedInstanceNormMixin:
    """可见位点统计的 InstanceNorm：修复掩码-稠密等价中的归一化污染。

    被遮位点置零后仍参与稠密 InstanceNorm 的 mean/var 会系统性拉偏统计
    （高掩码率下训练/下游分布不一致）。本类在 holder 提供掩码时仅在
    **可见位点**上算逐 (样本,通道) 的 mean/var，并在输出处门控（被遮位点
    置零，进一步逼近 SparK「每个块后重置」的稀疏副作用）；掩码全可见或
    holder 为空时与原生 InstanceNorm 数值一致（biased var + eps 入 sqrt）。

    子类继承自对应的 ``nn.InstanceNormNd`` → 参数/缓冲与 state_dict 键完全
    不变，不影响 ``encoder.*`` 导出与下游 strict=False 加载。
    """

    _spark_holder: "_SparkVisHolder | None" = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        holder = self._spark_holder
        vis_full = holder.vis_full if holder is not None else None
        if vis_full is None:
            return super().forward(x)                 # 稠密前向：原生路径
        dims = tuple(range(2, x.dim()))
        vis = downsample_mask_to(vis_full, x.shape[2:]).to(x.dtype)  # (B,1,*sp)
        cnt = vis.sum(dim=dims, keepdim=True).clamp_min(1.0)         # (B,1,1..)
        mean = (x * vis).sum(dim=dims, keepdim=True) / cnt           # (B,C,1..)
        var = ((x - mean).pow(2) * vis).sum(dim=dims, keepdim=True) / cnt
        y = (x - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            shape = (1, -1) + (1,) * (x.dim() - 2)
            y = y * self.weight.view(shape) + self.bias.view(shape)
        return y * vis


class MaskedInstanceNorm2d(_MaskedInstanceNormMixin, nn.InstanceNorm2d):
    pass


class MaskedInstanceNorm3d(_MaskedInstanceNormMixin, nn.InstanceNorm3d):
    pass


_MASKED_NORM_MAP = {
    nn.InstanceNorm2d: MaskedInstanceNorm2d,
    nn.InstanceNorm3d: MaskedInstanceNorm3d,
}


def enable_masked_instance_norm(encoder: nn.Module) -> int:
    """把 ``encoder`` 内所有 InstanceNorm 就地换为 Masked 版本，返回替换数。

    替换保持参数值与 state_dict 键不变（子类化）；同时在 encoder 上挂
    ``_spark_vis_holder`` 供 :func:`spark_encode` 逐前向传递可见掩码。重复
    调用幂等。若 encoder 不含 InstanceNorm（如 norm_type='group'/'batch'）
    则返回 0 并告警：这些归一化的统计仍会被置零位点污染。
    """
    holder = getattr(encoder, "_spark_vis_holder", None)
    if holder is None:
        holder = _SparkVisHolder()
        encoder._spark_vis_holder = holder
    n = 0
    for parent in list(encoder.modules()):
        for name, child in list(parent.named_children()):
            cls = _MASKED_NORM_MAP.get(type(child))
            if cls is None:
                if isinstance(child, _MaskedInstanceNormMixin):
                    child._spark_holder = holder      # 幂等：已转换只续 holder
                    n += 1
                continue
            if bool(child.track_running_stats):
                logger.warning(
                    "enable_masked_instance_norm: skipping %s with "
                    "track_running_stats=True (unsupported).", name)
                continue
            new = cls(child.num_features, eps=child.eps,
                      momentum=child.momentum, affine=child.affine,
                      track_running_stats=False)
            new.load_state_dict(child.state_dict())
            new._spark_holder = holder
            new.to(next(child.parameters(), torch.empty(0)).device
                   if child.affine else torch.device("cpu"))
            setattr(parent, name, new)
            n += 1
    if n == 0:
        logger.warning(
            "enable_masked_instance_norm: no InstanceNorm found in encoder; "
            "masked-dense normalization statistics remain polluted by zeroed "
            "positions (non-instance norm_type?).")
    return n


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
    holder = getattr(encoder, "_spark_vis_holder", None)
    if holder is not None and bool(mask_full.any()):  # 全可见时走原生路径（位级等价）
        holder.vis_full = vis_full                    # Masked InstanceNorm 统计只看可见位点
    try:
        x = x * vis_full                              # 被遮输入置零（无 mask token）
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
    finally:
        if holder is not None:
            holder.vis_full = None                    # 其它稠密前向不受影响


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


class SSLSparkSegDecModel(nn.Module):
    """SparK 变体：用**下游真解码器**（``decoder.*`` 同名同形）做重建。

    与 :class:`SSLSparkModel`（轻量解码器用完即弃）的对照：解码器与下游
    分割模型的 ``decoder`` 同构同名 → SSL ckpt 经 ``train.pretrain``
    （strict=False）同时命中 ``encoder.*`` 与 ``decoder.*``，下游只剩
    ``seg_head.*`` 随机初始化（decoder warm-start，对分割最友好）。
    ``recon_head.*`` / ``mask_embed.*`` 为 SSL 专用，下游作 unexpected 丢弃。
    """

    def __init__(
        self,
        encoder         : nn.Module,
        decoder         : nn.Module,
        encoder_channels: List[int],
        in_channels     : int,
        spatial_dims    : int = 3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder                       # 与下游同名：可迁移 warm-start
        self.spatial_dims = int(spatial_dims)
        self.out_channels = int(in_channels)
        self.mode = INTERP_SMOOTH[self.spatial_dims]
        # 逐 encoder 尺度的 densify 嵌入（同 SparkLightDecoder；下游丢弃）。
        self.mask_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, int(c), *([1] * self.spatial_dims)))
            for c in encoder_channels])
        for p in self.mask_embed:
            nn.init.normal_(p, std=0.02)
        # 重建头：真解码器最高分辨率输出 → in_channels（下游丢弃）。
        self.recon_head = _CONV[self.spatial_dims](
            int(decoder.out_channels[-1]), int(in_channels), kernel_size=1)

    def forward(self, x: torch.Tensor, mask_full: torch.Tensor) -> torch.Tensor:
        features, visibles = spark_encode(self.encoder, x, mask_full)
        feats = [densify(f, v, e) for f, v, e in
                 zip(features, visibles, self.mask_embed)]
        out = self.recon_head(self.decoder(feats)[-1])
        if out.shape[2:] != x.shape[2:]:             # stem_stride>1 时补回输入分辨率
            out = F.interpolate(out, size=x.shape[2:], mode=self.mode,
                                align_corners=False)
        return out

    def param_count(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
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


def build_ssl_spark_model(cfg, dim_div: int = 4, min_dim: int = 16,
                          masked_norm: bool = True,
                          decoder_mode: str = "light") -> nn.Module:
    """构造 SparK 模型：复用 ``build_model`` 的 encoder（保证下游同名同形）。

    ``decoder_mode='light'``（默认，SparK 原方）：轻量窄解码器用完即弃 →
    下游仅命中 ``encoder.*``，``spark_decoder.*`` 作 unexpected 丢弃。

    ``decoder_mode='seg'``：重建经过下游**真解码器**（同一 ``build_model``
    的 ``decoder``，同名同形）→ SSL ckpt 同时命中 ``encoder.*`` 与
    ``decoder.*``，下游仅 ``seg_head.*`` 随机（decoder warm-start）。
    """
    arch = str(cfg.model.arch).lower()
    if arch != "unet":
        raise ValueError(
            f"build_ssl_spark_model requires model.arch=='unet'; got {arch!r}.")
    mode = str(decoder_mode).lower()
    if mode not in ("light", "seg"):
        raise ValueError(
            f"decoder_mode must be 'light' or 'seg'; got {decoder_mode!r}.")
    seg_model = build_model(cfg)                 # 同一构建路径，确保 encoder 同名同形
    encoder = seg_model.encoder
    if masked_norm:
        n_conv = enable_masked_instance_norm(encoder)
        logger.info(
            "SparK masked InstanceNorm enabled: %d norm layer(s) converted.",
            n_conv)
    spatial_dims = int(cfg.model.spatial_dims)
    out_ch = int(cfg.model.in_channels)
    if mode == "seg":
        model: nn.Module = SSLSparkSegDecModel(
            encoder=encoder, decoder=seg_model.decoder,
            encoder_channels=[int(c) for c in cfg.model.encoder_channels],
            in_channels=out_ch, spatial_dims=spatial_dims)
        pc = model.param_count()
        logger.info(
            "Built SSLSparkSegDecModel (decoder_mode='seg'): enc=%.2fM, "
            "decoder=%.2fM (downstream-transferable), total=%.2fM.",
            pc["encoder"] / 1e6, pc["spark_decoder"] / 1e6, pc["total"] / 1e6)
        return model
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
    "SSLSparkModel", "SSLSparkSegDecModel",
    "build_spark_decoder", "build_ssl_spark_model",
    "MaskedInstanceNorm2d", "MaskedInstanceNorm3d",
    "enable_masked_instance_norm",
]
