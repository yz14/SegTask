"""方案⑦ JEPA-3D：隐空间掩码预测（image-only，无标注）。

联合嵌入预测（I-JEPA 的 CNN 适配）：同一骨干同时充当**上下文编码器**（可训练）与
**目标编码器**（EMA、冻结），外加一个轻量**预测器**。流程：

1. 采样目标块掩码（单元网格，``1``=被遮目标块）。
2. 上下文编码器看 ``apply_mask_token`` 遮挡后的输入（CNN 不能丢 token，故用 mask-token
   稠密输入等价"遮去目标块的上下文"）→ 上下文特征图。
3. 预测器据上下文特征预测被遮位点的特征。
4. 目标编码器看**完整**输入 → 目标特征图（stop-grad）。
5. 损失 = 仅被遮特征位点的 L2：``L = mean‖predictor(ctx) − sg(target)‖²``（可叠加 VICReg
   方差/协方差正则抗坍缩）。

防坍缩：目标侧 EMA（不接收梯度）+ 非对称预测器（必备），可选 VICReg（默认关闭）。
隔离轴=**隐空间目标 vs 像素目标**（对照①/②）。下游交接与 ④ 同构：导出 **目标**
encoder（EMA，更稳）为 ``encoder.*``；预测器/mask-token 经 strict=False 加载被丢弃。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.masking import apply_mask_token, downsample_mask_to, make_unit_mask, masked_recon_loss
from ..models.jepa_modules import build_jepa_encoder, build_jepa_predictor
from .base import SSLMethod


class _JEPAModule(nn.Module):
    """承载 context_encoder / target_encoder（冻结）/ predictor / mask_token 的容器。"""

    def __init__(self, context_encoder: nn.Module, target_encoder: nn.Module,
                 predictor: nn.Module, in_channels: int, spatial_dims: int):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = predictor
        # 可学习 mask token（每输入通道一个标量，跨被遮单元广播）。
        self.mask_token = nn.Parameter(
            torch.zeros(1, int(in_channels), *([1] * int(spatial_dims))))
        nn.init.normal_(self.mask_token, std=0.02)


class JEPAMethod(SSLMethod):
    name = "jepa"

    def __init__(self, cfg, ssl, device: torch.device):
        super().__init__(cfg, ssl, device)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.mask_unit = int(ssl.jepa_mask_unit)
        self.mask_ratio = float(ssl.jepa_mask_ratio)
        self.level = int(ssl.jepa_feature_level)
        self.momentum_base = float(ssl.jepa_momentum_base)
        self.momentum_final = float(ssl.jepa_momentum_final)
        self.var_weight = float(ssl.jepa_var_weight)
        self.cov_weight = float(ssl.jepa_cov_weight)
        self._step = 0
        self.total_steps = 1

    # ---- modules ----------------------------------------------------------
    def build_modules(self) -> nn.Module:
        cfg, ssl = self.cfg, self.ssl
        context = build_jepa_encoder(cfg)
        target = build_jepa_encoder(cfg)
        target.load_state_dict(context.state_dict())     # 目标初始 = 上下文
        channels = int(cfg.model.encoder_channels[int(ssl.jepa_feature_level)])
        predictor = build_jepa_predictor(
            cfg, channels=channels, hidden=int(ssl.jepa_predictor_hidden),
            depth=int(ssl.jepa_predictor_depth))
        return _JEPAModule(context, target, predictor,
                           int(cfg.model.in_channels),
                           int(cfg.model.spatial_dims))

    # ---- schedules --------------------------------------------------------
    def configure_schedule(self, total_steps: int) -> None:
        self.total_steps = max(int(total_steps), 1)

    def _momentum(self) -> float:
        """目标编码器 EMA 动量：cosine 从 base 升到 final。"""
        progress = min(self._step / self.total_steps, 1.0)
        return self.momentum_final - (self.momentum_final - self.momentum_base) * (
            math.cos(math.pi * progress) + 1.0) / 2.0

    # ---- loss -------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        image = batch["image"]
        b, spatial = image.shape[0], image.shape[2:]
        # 目标块掩码（1=被遮目标块）。
        mask_full = make_unit_mask(
            b, spatial, self.mask_unit, self.mask_ratio, image.device)
        # 上下文编码器看遮挡输入；目标编码器看完整输入（stop-grad）。
        ctx_in = apply_mask_token(image, mask_full, self.module.mask_token)
        ctx_feat = self.module.context_encoder(ctx_in)[self.level]
        with torch.no_grad():
            tgt_feat = self.module.target_encoder(image)[self.level]
        pred = self.module.predictor(ctx_feat)

        feat_mask = downsample_mask_to(mask_full, pred.shape[2:])
        loss = masked_recon_loss(pred, tgt_feat.detach(), feat_mask, "mse")
        logs = {"jepa_loss": float(loss.detach()),
                "mask_ratio": self.mask_ratio,
                "ema_momentum": self._momentum()}

        if self.var_weight > 0.0 or self.cov_weight > 0.0:
            var_loss, cov_loss = self._vicreg(pred, feat_mask)
            loss = loss + self.var_weight * var_loss + self.cov_weight * cov_loss
            logs["vicreg_var"] = float(var_loss.detach())
            logs["vicreg_cov"] = float(cov_loss.detach())
        return loss, logs

    def _vicreg(self, feat: torch.Tensor, feat_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """VICReg 方差(hinge std>=1) + 协方差(去相关) 正则，仅在被遮特征位点上。"""
        c = feat.shape[1]
        x = feat.float().flatten(2).permute(0, 2, 1).reshape(-1, c)   # (B*N, C)
        sel = feat_mask.flatten(2).permute(0, 2, 1).reshape(-1) > 0.5
        z = x[sel]                                                    # (M, C)
        if z.shape[0] < 2:
            zero = feat.new_zeros(())
            return zero, zero
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (z.shape[0] - 1)                          # (C, C)
        off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = off_diag_sq / c
        return var_loss, cov_loss

    # ---- EMA target encoder update ---------------------------------------
    def on_after_step(self, global_step: int) -> None:
        self._step = int(global_step)
        m = self._momentum()
        with torch.no_grad():
            for pc, pt in zip(self.module.context_encoder.parameters(),
                              self.module.target_encoder.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)
            for bc, bt in zip(self.module.context_encoder.buffers(),
                              self.module.target_encoder.buffers()):
                bt.copy_(bc)

    # ---- export -----------------------------------------------------------
    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """导出 **目标** encoder（EMA，表征更稳），键命名为 ``encoder.*``。"""
        from segtask_v1.trainer.checkpoint import unwrap_compile
        target = unwrap_compile(self.module).target_encoder
        return {f"encoder.{k}": v.detach().cpu().clone()
                for k, v in target.state_dict().items()}


__all__ = ["JEPAMethod"]
