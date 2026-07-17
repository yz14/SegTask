"""分类损失：BCE / CE / Focal（多标签 sigmoid 或 单标签 softmax）。

统一接口 ``loss_fn(logits, target) -> scalar``：

* 多标签（bce/focal）：logits 与 target 同形——volume (B, K) 或 slice
  (B, K, D)，target ∈ [0, 1]（mixup 后为软标签）。
* 单标签（ce）：logits (B, K)，target (B,) long 或 (B, K) 软标签
  （mixup 产生）。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from taskcore.config.core import Config as SegConfig

from ..config import ClsConfig, resolve_num_classes


def _class_weight_tensor(weights: List[float]) -> Optional[torch.Tensor]:
    return torch.tensor(weights, dtype=torch.float32) if weights else None


class MultiLabelBCELoss(nn.Module):
    """sigmoid BCE（可选逐类 pos_weight / label smoothing）。"""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(label_smoothing)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def _smooth(self, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0:
            return target
        return target * (1 - self.smoothing) + 0.5 * self.smoothing

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight
        if pw is not None and logits.ndim == 3:      # (B, K, D)：K 轴广播
            pw = pw.view(1, -1, 1)
        return F.binary_cross_entropy_with_logits(
            logits, self._smooth(target.float()), pos_weight=pw)


class MultiLabelFocalLoss(nn.Module):
    """sigmoid focal loss（Lin et al., ICCV 2017）；软标签兼容。"""

    def __init__(self, gamma: float = 2.0, alpha: float = -1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce * (1 - p_t).clamp_min(1e-6) ** self.gamma
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        return loss.mean()


class SingleLabelCELoss(nn.Module):
    """softmax CE；接受 (B,) 硬标签或 (B, K) 软标签（mixup）。"""

    def __init__(self, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(label_smoothing)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:               # 软标签
            logp = F.log_softmax(logits, dim=1)
            t = target.float()
            if self.smoothing > 0:
                k = logits.shape[1]
                t = t * (1 - self.smoothing) + self.smoothing / k
            loss = -(t * logp)
            if self.weight is not None:
                loss = loss * self.weight.view(1, -1)
            return loss.sum(dim=1).mean()
        return F.cross_entropy(logits, target.long(), weight=self.weight,
                               label_smoothing=self.smoothing)


def build_cls_loss(cfg: SegConfig, cls: ClsConfig) -> nn.Module:
    """按 ``(cfg, cls)`` 构建损失（工厂，与 segtask ``build_loss`` 同风格）。"""
    k = resolve_num_classes(cls, cfg)
    w = _class_weight_tensor(list(cls.class_weights))
    if w is not None and w.numel() != k:
        raise ValueError(
            f"class_weights length {w.numel()} != num_classes {k}.")
    if cls.loss_type == "bce":
        return MultiLabelBCELoss(pos_weight=w,
                                 label_smoothing=cls.label_smoothing)
    if cls.loss_type == "focal":
        return MultiLabelFocalLoss(gamma=cls.focal_gamma,
                                   alpha=cls.focal_alpha)
    if cls.loss_type == "ce":
        return SingleLabelCELoss(weight=w,
                                 label_smoothing=cls.label_smoothing)
    raise ValueError(f"Unknown cls.loss_type: {cls.loss_type!r}")


__all__ = [
    "MultiLabelBCELoss", "MultiLabelFocalLoss", "SingleLabelCELoss",
    "build_cls_loss",
]
