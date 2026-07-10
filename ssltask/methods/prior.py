"""经典几何先验自蒸馏：回归 Frangi vesselness（label-free）。

从原始 CT 直接算出"管状结构置信图"作为回归目标，把"什么是血管"的几何先验灌进
encoder/decoder。input = 干净图（可选 Genesis 破坏），target = 干净图的 vesselness。
与 ``genesis``（破坏→重建）正交，可分别预训练后各自衔接对比。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..data.corruptions import GenesisCorruptor
from ..data.vesselness import frangi_vesselness
from ..models.ssl_models import build_ssl_recon_model
from .base import RECON_LOSS_FNS, SSLMethod


class PriorMethod(SSLMethod):
    name = "prior"
    trainer_augment = True   # 重建类：输入=目标同源，接受 trainer 级通用增强

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.corruptor = GenesisCorruptor(ssl, self.spatial_dims)
        self.recon_loss_fn = RECON_LOSS_FNS[ssl.recon_loss]

    def build_modules(self) -> nn.Module:
        return build_ssl_recon_model(self.cfg)

    @torch.no_grad()
    def _make_io(self, clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.ssl
        target = frangi_vesselness(
            clean, scales=s.prior_scales, spatial_dims=self.spatial_dims,
            alpha=s.prior_alpha, beta=s.prior_beta,
            black_vessels=s.prior_black_vessels)
        model_input = self.corruptor(clean) if s.prior_corrupt_input else clean
        return model_input, target

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        clean = batch["image"]
        model_input, target = self._make_io(clean)
        pred = self.module(model_input)
        loss = self.recon_loss_fn(pred.float(), target.float())
        return loss, {"recon_loss": float(loss.detach())}

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        from segtask_v1.trainer.checkpoint import unwrap_compile
        sd = unwrap_compile(self.module).state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}


__all__ = ["PriorMethod"]
