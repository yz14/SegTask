"""方案③ Models Genesis：多变换破坏 → 重建原图。

input = 破坏图（Bézier 强度 / 局部打乱 / 内外补全），target = 干净原图；逐体素重建
损失。训练整套 enc+dec（下游分割可同时迁移编/解码器）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..data.corruptions import GenesisCorruptor
from ..models.ssl_models import build_ssl_recon_model
from .base import RECON_LOSS_FNS, SSLMethod


class GenesisMethod(SSLMethod):
    name = "genesis"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.corruptor = GenesisCorruptor(ssl, int(cfg.model.spatial_dims))
        self.recon_loss_fn = RECON_LOSS_FNS[ssl.recon_loss]

    def build_modules(self) -> nn.Module:
        return build_ssl_recon_model(self.cfg)

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
        clean = batch["image"]
        model_input = self.corruptor(clean)          # @no_grad 内部已 clone
        pred = self.module(model_input)
        loss = self.recon_loss_fn(pred.float(), clean.float())
        return loss, {"recon_loss": float(loss.detach())}

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        from segtask_v1.trainer.checkpoint import unwrap_compile
        sd = unwrap_compile(self.module).state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}


__all__ = ["GenesisMethod"]
