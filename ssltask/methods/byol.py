"""方案 B1：BYOL-3D（online encoder/projector + predictor；EMA target）。"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.models.factory import build_model
from segtask_v1.trainer.checkpoint import unwrap_compile

from ..data.multicrop import MultiCropGenerator
from ..models.dino_modules import DINOHead
from .base import SSLMethod


class _ProjectedEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        z = feats[-1].mean(dim=tuple(range(2, feats[-1].ndim)))
        z = self.projector(z)
        return F.normalize(z, dim=-1)


class _BYOLModule(nn.Module):
    def __init__(self, online: _ProjectedEncoder, target: _ProjectedEncoder,
                 predictor: nn.Module):
        super().__init__()
        self.online = online
        self.target = target
        self.predictor = predictor
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    def train(self, mode: bool = True):
        """冻结的 EMA 目标网络始终保持 eval 模式。"""
        super().train(mode)
        self.target.eval()
        return self


class BYOLMethod(SSLMethod):
    name = "byol"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.proj_dim = int(ssl.byol_proj_dim)
        self.pred_hidden_dim = int(ssl.byol_pred_hidden_dim)
        self.momentum_base = float(ssl.byol_momentum_base)
        self.momentum_final = float(ssl.byol_momentum_final)
        self.dino_hidden_dim = int(ssl.dino_hidden_dim)
        self._step = 0
        self.total_steps = 1

        patch = [int(s) for s in cfg.data.patch_size]
        model_spatial = patch if self.spatial_dims == 3 else patch[1:]
        self.multicrop = MultiCropGenerator(
            spatial_dims=self.spatial_dims,
            global_size=model_spatial,
            local_size=model_spatial,
            n_global=2,
            n_local=0,
            global_scale=tuple(ssl.dino_global_scale),
            local_scale=tuple(ssl.dino_local_scale),
            flip_prob=float(ssl.dino_flip_prob),
            intensity_scale=float(ssl.dino_intensity_scale),
            intensity_shift=float(ssl.dino_intensity_shift),
        )

    def build_modules(self) -> nn.Module:
        proj_dim = int(self.ssl.byol_proj_dim)
        hidden_dim = int(self.ssl.dino_hidden_dim)
        pred_hidden_dim = int(self.ssl.byol_pred_hidden_dim)
        enc_online = build_model(self.cfg).encoder
        enc_target = build_model(self.cfg).encoder
        enc_target.load_state_dict(enc_online.state_dict())   # 目标初始 = 在线
        projector_online = DINOHead(
            in_dim=int(self.cfg.model.encoder_channels[-1]),
            out_dim=proj_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=proj_dim,
            n_layers=2,
            use_bn=False,
        )
        projector_target = DINOHead(
            in_dim=int(self.cfg.model.encoder_channels[-1]),
            out_dim=proj_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=proj_dim,
            n_layers=2,
            use_bn=False,
        )
        projector_target.load_state_dict(projector_online.state_dict())
        target_enc = _ProjectedEncoder(enc_target, projector_target)
        online = _ProjectedEncoder(enc_online, projector_online)
        predictor = DINOHead(
            in_dim=proj_dim,
            out_dim=proj_dim,
            hidden_dim=pred_hidden_dim,
            bottleneck_dim=proj_dim,
            n_layers=2,
            use_bn=False,
        )
        return _BYOLModule(online, target_enc, predictor)

    def configure_schedule(self, total_steps: int) -> None:
        self.total_steps = max(int(total_steps), 1)

    def _momentum(self) -> float:
        progress = min(self._step / self.total_steps, 1.0)
        return self.momentum_final - (self.momentum_final - self.momentum_base) * (
            math.cos(math.pi * progress) + 1.0) / 2.0

    def _pair_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float().detach(), dim=-1)
        return 2.0 - 2.0 * (pred * target).sum(dim=-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        crops = self.multicrop(batch["image"])["global"]
        online_proj = [self.module.online(c) for c in crops]
        online_pred = [F.normalize(self.module.predictor(z), dim=-1)
                       for z in online_proj]
        with torch.no_grad():
            target_proj = [self.module.target(c) for c in crops]
        total = batch["image"].new_zeros(())
        n_pairs = 0
        for i, p in enumerate(online_pred):
            for j, z in enumerate(target_proj):
                if i == j:
                    continue
                total = total + self._pair_loss(p, z).mean()
                n_pairs += 1
        loss = total / max(n_pairs, 1)
        return loss, {"byol_loss": float(loss.detach()), "ema_momentum": self._momentum()}

    def on_resume(self, global_step: int) -> None:
        self._step = int(global_step)

    def on_after_step(self, global_step: int, stepped: bool = True) -> None:
        self._step = int(global_step)
        if not stepped:                       # 跳步：不推进 EMA 目标网络
            return
        m = self._momentum()
        with torch.no_grad():
            for ps, pt in zip(self.module.online.parameters(), self.module.target.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
            for bs, bt in zip(self.module.online.buffers(), self.module.target.buffers()):
                bt.copy_(bs)

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        online = unwrap_compile(self.module).online
        return {f"encoder.{k}": v.detach().cpu().clone()
                for k, v in online.encoder.state_dict().items()}


__all__ = ["BYOLMethod"]
