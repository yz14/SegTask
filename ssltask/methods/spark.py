"""方案① SparK-3D：掩码-稠密等价 + 层次解码器像素重建（image-only，无标注）。

以 16³ 单元 patch-wise 掩码（``spark_mask_ratio`` 默认 0.6）；编码端用掩码-稠密等价
（被遮位点置零 + 逐尺度门控）模拟稀疏前向；解码端为轻量层次 UNet（densify → 逐级上采样
+ 横向融合），输出单通道重建；**仅对被遮位点**计算损失（默认 L2/mse，与 SimMIM② 的 L1
形成对照），目标可选 per-unit 归一化（``spark_norm_pix``）。

与 SimMIM② 的对照变量：稀疏（门控） vs 稠密、层次解码器 vs 极简头。下游仅迁移 encoder
（解码器用完即弃 → ``train.pretrain`` strict=False 仅命中 ``encoder.*``）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..data.masking import make_unit_mask, masked_recon_loss, per_unit_normalize
from ..models.spark_modules import build_ssl_spark_model
from .base import SSLMethod


class SparKMethod(SSLMethod):
    name = "spark"
    trainer_augment = True   # 重建类：输入=目标同源，接受 trainer 级通用增强

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.mask_ratio = float(ssl.spark_mask_ratio)
        self.unit = int(ssl.spark_mask_unit)
        self.loss_name = str(ssl.recon_loss)
        self.norm_pix = bool(ssl.spark_norm_pix)

    def build_modules(self) -> nn.Module:
        return build_ssl_spark_model(
            self.cfg,
            dim_div=int(self.ssl.spark_decoder_dim_div),
            min_dim=int(self.ssl.spark_decoder_min_dim),
            masked_norm=bool(self.ssl.spark_masked_norm),
            decoder_mode=str(self.ssl.spark_decoder_mode))

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        clean = batch["image"]                       # (B, C, *spatial)
        spatial = clean.shape[2:]
        mask_full = make_unit_mask(
            clean.shape[0], spatial, self.unit, self.mask_ratio, clean.device)
        pred = self.module(clean, mask_full)
        target = (per_unit_normalize(clean, self.unit)
                  if self.norm_pix else clean)
        loss = masked_recon_loss(pred, target, mask_full, self.loss_name)
        return loss, {"recon_loss": loss.detach(),
                      "mask_ratio": self.mask_ratio}

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        from segtask_v1.trainer.checkpoint import unwrap_compile
        sd = unwrap_compile(self.module).state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}


__all__ = ["SparKMethod"]
