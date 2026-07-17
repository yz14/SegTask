"""方案② SimMIM-3D：稠密卷积 + mask token 掩码图像建模。

朴素稠密 MIM：以单元（默认 16³/16²）为粒度 patch-wise 掩码，被遮单元在**输入处**用
可学习 ``mask_token`` 占位，整图进稠密 encoder；接一个极轻的预测头（``LightPixelHead``，
无跨尺度 skip）映射回输入分辨率；**仅对被遮位点**计算重建损失（默认 L1）。

与方案① SparK 的对照变量：稠密 vs 稀疏编码、极简头 vs 层次解码器。下游仅迁移 encoder
（解码器在 MIM 中不预训练 → 下游 ``train.pretrain`` strict=False 加载时保持随机）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..data.masking import apply_mask_token, make_unit_mask, masked_recon_loss
from ..models.ssl_models import build_ssl_mim_model
from .base import SSLMethod


class SimMIMMethod(SSLMethod):
    name = "simmim"
    trainer_augment = True   # 重建类：输入=目标同源，接受 trainer 级通用增强

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.mask_ratio = float(ssl.mim_mask_ratio)
        self.unit = int(ssl.mim_mask_unit)
        self.loss_name = str(ssl.recon_loss)

    def build_modules(self) -> nn.Module:
        return build_ssl_mim_model(self.cfg, head_dim=int(self.ssl.mim_head_dim))

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        clean = batch["image"]                       # (B, C, *spatial)
        spatial = clean.shape[2:]
        mask_full = make_unit_mask(
            clean.shape[0], spatial, self.unit, self.mask_ratio, clean.device)
        x_masked = apply_mask_token(clean, mask_full, self.module.mask_token)
        pred = self.module(x_masked)
        loss = masked_recon_loss(pred, clean, mask_full, self.loss_name)
        return loss, {"recon_loss": loss.detach(),
                      "mask_ratio": self.mask_ratio}

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        from taskcore.engine.checkpoint import unwrap_compile
        sd = unwrap_compile(self.module).state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}


__all__ = ["SimMIMMethod"]
