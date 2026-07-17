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
        # 2.5D（spatial_dims==2）下是否在折叠前的体上按 3D Frangi 计算目标：穿面
        # 血管敏感 + 层间归一一致（见 SSLConfig.prior_target_3d）。3D cubic 恒为 3D。
        self.target_3d = (bool(getattr(ssl, "prior_target_3d", True))
                          and self.spatial_dims == 2)
        # 物理体素间距（可选）：给定时 prior_scales 按物理尺度(mm)解释，Frangi
        # 各向异性计算；空=体素单位（旧行为）。target_3d 时长度为 3 (sz,sy,sx)。
        self.prior_spacing = (
            [float(s) for s in ssl.prior_spacing] if ssl.prior_spacing
            else None)

    def build_modules(self) -> nn.Module:
        return build_ssl_recon_model(self.cfg)

    @torch.no_grad()
    def _make_io(self, clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.ssl
        if self.target_3d and clean.dim() == 4:
            # 2.5D：clean 已被 trainer 折成 (B, D, H, W)（D=切片数=通道）。把深度
            # 还原成空间轴 (B,1,D,H,W)，按 3D Frangi 整卷计算（穿面敏感、整卷归一），
            # 再折回 (B,D,H,W) 与模型输出对齐。
            b, d, h, w = clean.shape
            vol = clean.reshape(b, 1, d, h, w)
            tgt = frangi_vesselness(
                vol, scales=s.prior_scales, spatial_dims=3,
                alpha=s.prior_alpha, beta=s.prior_beta,
                black_vessels=s.prior_black_vessels, spacing=self.prior_spacing)
            target = tgt.reshape(b, d, h, w)
        else:
            target = frangi_vesselness(
                clean, scales=s.prior_scales, spatial_dims=self.spatial_dims,
                alpha=s.prior_alpha, beta=s.prior_beta,
                black_vessels=s.prior_black_vessels, spacing=self.prior_spacing)
        model_input = self.corruptor(clean) if s.prior_corrupt_input else clean
        return model_input, target

    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        clean = batch["image"]
        model_input, target = self._make_io(clean)
        pred = self.module(model_input)
        loss = self.recon_loss_fn(pred.float(), target.float())
        # 返回**未同步**的 device 标量（0-dim tensor）作日志值：由 SSLTrainer 在
        # 日志/边界处批量 .tolist() 一次性取回，避免每 micro-step 一次 D2H 同步。
        return loss, {"recon_loss": loss.detach()}

    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        from taskcore.engine.checkpoint import unwrap_compile
        sd = unwrap_compile(self.module).state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}


__all__ = ["PriorMethod"]
