"""自监督重建模型 + 构造器（复用 segtask_v1 骨干）。

``SSLReconModel`` = 与分割同构的 UNet 编/解码器（由 ``segtask_v1.models.factory.
build_model`` 构建后复用其 ``encoder`` / ``decoder`` 子模块）+ 一个**独立命名**的重建头
``recon_head``（out_channels = 模型输入通道数）。

关键设计：重建头名为 ``recon_head``（而非分割的 ``seg_head``），因此 SSL ckpt 与分割
模型之间不会发生 head 权重的 shape 冲突——下游分割/分类侧 ``train.pretrain``
（``strict=False``）会让 ``encoder.*`` / ``decoder.*`` 全部命中、``recon_head.*`` 作为
unexpected 被丢弃、``seg_head.*`` 作为 missing 保持随机初始化。这是 SSL→下游干净
交接的核心，由 ``build_model`` 复用保证逐参数同名同形。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.models.blocks import _CONV, INTERP_SMOOTH, ConvNormAct
from segtask_v1.models.factory import build_model
from segtask_v1.models.unet import SegmentationHead

logger = logging.getLogger(__name__)


class SSLReconModel(nn.Module):
    """编码器-解码器重建模型（Models Genesis / Frangi 先验回归共用）。

    forward(x): (B, C, *spatial) → (B, C, *spatial) 重建/回归。"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        out_channels: int,
        spatial_dims: int = 3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out_channels = int(out_channels)
        self.recon_head = SegmentationHead(
            decoder.out_channels[-1], self.out_channels,
            spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)
        out = self.recon_head(dec_features[-1])
        if out.shape[2:] != x.shape[2:]:
            raise RuntimeError(
                f"SSL recon output size mismatch: got {tuple(out.shape[2:])}, "
                f"expected {tuple(x.shape[2:])}. Check stem_stride / encoder "
                f"downsampling vs input spatial dims.")
        return out

    def param_count(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters())
        dec  = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.recon_head.parameters())
        return {"encoder": enc, "decoder": dec, "recon_head": head,
                "total": enc + dec + head}


def build_ssl_recon_model(cfg) -> SSLReconModel:
    """构造与分割同构的 SSL 重建模型。

    复用 ``segtask_v1.models.factory.build_model`` 保证编/解码器与下游逐参数同名
    同形 → SSL ckpt 可经下游已有的 ``train.pretrain`` 非严格加载干净衔接。
    分割头/深监督/aux 头在此被丢弃（仅取 encoder/decoder），不参与 SSL。
    """
    arch = str(cfg.model.arch).lower()
    if arch != "unet":
        raise ValueError(
            f"build_ssl_recon_model requires model.arch=='unet'; got {arch!r}.")
    seg_model = build_model(cfg)  # 同一构建路径，确保 enc/dec 同名同形
    model = SSLReconModel(
        encoder      = seg_model.encoder,
        decoder      = seg_model.decoder,
        out_channels = int(cfg.model.in_channels),
        spatial_dims = int(cfg.model.spatial_dims))
    pc = model.param_count()
    logger.info(
        "Built SSLReconModel: enc=%.2fM, dec=%.2fM, recon_head=%.2fM, "
        "total=%.2fM, out_channels=%d (=in_channels).",
        pc["encoder"] / 1e6, pc["decoder"] / 1e6,
        pc["recon_head"] / 1e6, pc["total"] / 1e6, model.out_channels)
    return model


# ---------------------------------------------------------------------------
# SimMIM② —— encoder + 轻量像素预测头（无解码器、无 skip）
# ---------------------------------------------------------------------------
class LightPixelHead(nn.Module):
    """SimMIM 轻量像素预测头：线性投影 + 上采样到输入分辨率（**无跨尺度 skip**）。

    bottleneck 特征 ``(B, in_ch, *grid)`` → 1×1 投影 → 平滑插值到输入 spatial →
    一层 3×3 ``ConvNormAct`` 局部精修 → 1×1 输出 ``out_ch``。刻意极轻（SSL.md 方案②：
    重负担落在 encoder，预测头只做最后映射），与 SparK 的层次化解码器形成对照变量。
    """

    def __init__(
        self,
        in_ch       : int,
        out_ch      : int,
        hidden      : int,
        spatial_dims: int = 3,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        activation  : str = "leakyrelu"):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        self.mode = INTERP_SMOOTH[self.spatial_dims]
        self.proj = _CONV[self.spatial_dims](in_ch, hidden, kernel_size=1)
        self.refine = ConvNormAct(
            hidden, hidden, kernel_size=3, stride=1, padding=1,
            norm_type=norm_type, norm_groups=norm_groups,
            activation=activation, spatial_dims=self.spatial_dims)
        self.out = _CONV[self.spatial_dims](hidden, out_ch, kernel_size=1)

    def forward(self, feat: torch.Tensor, target_spatial) -> torch.Tensor:
        x = self.proj(feat)
        x = F.interpolate(
            x, size=tuple(int(s) for s in target_spatial),
            mode=self.mode, align_corners=False)
        x = self.refine(x)
        return self.out(x)


class SSLMIMModel(nn.Module):
    """SimMIM 模型：复用 segtask 编码器 + ``LightPixelHead`` + 可学习 ``mask_token``。

    forward(x): 被遮输入 (B, C, *spatial) → 重建 (B, C, *spatial)。掩码施加（用
    ``mask_token`` 替换被遮单元）由方法在 ``compute_loss`` 内完成；本模型只前向。
    解码器/分割头在此被丢弃（MIM 不预训练解码器）。
    """

    def __init__(
        self,
        encoder     : nn.Module,
        head         : LightPixelHead,
        in_channels  : int,
        spatial_dims : int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.spatial_dims = int(spatial_dims)
        self.out_channels = int(in_channels)
        # 可学习 mask token（每输入通道一个标量，跨被遮单元广播）。下游不需要，
        # 经 strict=False 加载时作为 unexpected 被丢弃。
        self.mask_token = nn.Parameter(
            torch.zeros(1, self.out_channels, *([1] * self.spatial_dims)))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self.encoder(x)
        out = self.head(enc_features[-1], x.shape[2:])
        if out.shape[2:] != x.shape[2:]:
            raise RuntimeError(
                f"SSL MIM output size mismatch: got {tuple(out.shape[2:])}, "
                f"expected {tuple(x.shape[2:])}. Check encoder downsampling / "
                f"stem_stride vs input spatial dims.")
        return out

    def param_count(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"encoder": enc, "head": head,
                "mask_token": self.mask_token.numel(),
                "total": enc + head + self.mask_token.numel()}


def build_ssl_mim_model(cfg, head_dim: int = 0) -> SSLMIMModel:
    """构造 SimMIM 模型：复用 ``build_model`` 的 encoder（保证下游同名同形）。

    仅取 encoder（解码器/分割头丢弃）→ SSL ckpt 经下游 ``train.pretrain``（strict=False）
    命中 ``encoder.*``，``decoder.*``/``seg_head.*`` 保持随机（MIM 不迁移解码器）。
    ``head_dim<=0`` 时自动取 ``max(enc_last//2, 32)``。
    """
    arch = str(cfg.model.arch).lower()
    if arch != "unet":
        raise ValueError(
            f"build_ssl_mim_model requires model.arch=='unet'; got {arch!r}.")
    seg_model = build_model(cfg)  # 同一构建路径，确保 encoder 同名同形
    encoder = seg_model.encoder
    spatial_dims = int(cfg.model.spatial_dims)
    enc_last = int(cfg.model.encoder_channels[-1])
    out_ch = int(cfg.model.in_channels)
    hidden = int(head_dim) if int(head_dim) > 0 else max(enc_last // 2, 32)
    head = LightPixelHead(
        in_ch=enc_last, out_ch=out_ch, hidden=hidden,
        spatial_dims=spatial_dims,
        norm_type=cfg.model.norm_type, norm_groups=cfg.model.norm_groups,
        activation=cfg.model.activation)
    model = SSLMIMModel(
        encoder=encoder, head=head, in_channels=out_ch,
        spatial_dims=spatial_dims)
    pc = model.param_count()
    logger.info(
        "Built SSLMIMModel: enc=%.2fM, light_head=%.2fM (hidden=%d), "
        "total=%.2fM, out_channels=%d (=in_channels).",
        pc["encoder"] / 1e6, pc["head"] / 1e6, hidden,
        pc["total"] / 1e6, model.out_channels)
    return model


__all__ = [
    "SSLReconModel", "build_ssl_recon_model",
    "LightPixelHead", "SSLMIMModel", "build_ssl_mim_model",
]
