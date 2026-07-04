"""检测损失：sigmoid Focal / GIoU / L1 / 匈牙利集合匹配（2D/3D 同构）。

损失统一 fp32 计算（承接 segtask AMP 口径：head 输出可能是 fp16，调用方
在 autocast 外调用本模块）。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..ops import generalized_box_iou

__all__ = [
    "sigmoid_focal_loss", "box_reg_loss", "hungarian_match",
]


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "sum") -> torch.Tensor:
    """RetinaNet focal（targets ∈ {0,1} 同形状）。"""
    logits = logits.float()
    targets = targets.float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def box_reg_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                 kind: str = "giou", reduction: str = "sum") -> torch.Tensor:
    """已解码框的回归损失。``kind ∈ {giou, l1, smooth_l1}``。"""
    pred_boxes = pred_boxes.float()
    gt_boxes = gt_boxes.float()
    if pred_boxes.numel() == 0:
        return pred_boxes.sum()
    if kind == "giou":
        giou = generalized_box_iou(pred_boxes, gt_boxes).diagonal()
        loss = 1.0 - giou
    elif kind == "l1":
        loss = (pred_boxes - gt_boxes).abs().sum(dim=-1)
    elif kind == "smooth_l1":
        loss = F.smooth_l1_loss(pred_boxes, gt_boxes,
                                reduction="none").sum(dim=-1)
    else:
        raise ValueError(f"unknown reg loss kind: {kind!r}")
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


@torch.no_grad()
def hungarian_match(
    pred_logits: torch.Tensor,     # (Q, K) 每 query 类 logits（sigmoid 口径）
    pred_boxes : torch.Tensor,     # (Q, 2d) 已解码框（patch 坐标）
    gt_boxes   : torch.Tensor,     # (G, 2d)
    gt_labels  : torch.Tensor,     # (G,)
    cls_weight : float = 2.0,
    l1_weight  : float = 5.0,
    giou_weight: float = 2.0,
    norm_size  : torch.Tensor = None,   # (d,) L1 归一化尺寸（patch 大小）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DETR 匈牙利匹配：返回 (query 索引, gt 索引)。"""
    G = gt_boxes.shape[0]
    device = pred_logits.device
    if G == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    prob = pred_logits.float().sigmoid()             # (Q, K)
    cost_cls = -prob[:, gt_labels]                   # (Q, G)
    pb, gb = pred_boxes.float(), gt_boxes.float()
    if norm_size is not None:
        s = norm_size.to(pb).repeat(2)
        pb, gb = pb / s, gb / s
    cost_l1 = torch.cdist(pb, gb, p=1)               # (Q, G)
    cost_giou = -generalized_box_iou(pred_boxes.float(), gt_boxes.float())
    cost = (cls_weight * cost_cls + l1_weight * cost_l1
            + giou_weight * cost_giou)
    q_idx, g_idx = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(q_idx, dtype=torch.long, device=device),
            torch.as_tensor(g_idx, dtype=torch.long, device=device))
