"""检测评估：COCO 式 mAP + FROC（医学标准，Plan §3.6）。

匹配口径统一为 IoU >= ``iou_thresh``（2D/3D 通用）。FROC 由 predictor 在
拼接后的 3D 框上调用（2.5D 与 3D 分支同一读数口径，Plan §7-5）。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .ops import box_iou

__all__ = ["match_detections", "average_precision", "detection_map", "froc"]


def match_detections(pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                     gt_boxes: torch.Tensor, iou_thresh: float
                     ) -> torch.Tensor:
    """按分数降序贪心匹配 gt（一 gt 至多配一检出）。→ (N,) bool TP 标记。"""
    n = pred_boxes.shape[0]
    tp = torch.zeros(n, dtype=torch.bool)
    if n == 0 or gt_boxes.numel() == 0:
        return tp
    order = pred_scores.argsort(descending=True)
    iou = box_iou(pred_boxes, gt_boxes)              # (N, G)
    taken = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for i in order.tolist():
        best, g = iou[i].max(dim=0)
        if best >= iou_thresh and not taken[g]:
            taken[g] = True
            tp[i] = True
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray,
                      num_gt: int) -> float:
    """全 recall 区间插值 AP（COCO 连续口径）。"""
    if num_gt == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float64)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / num_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # 单调化 precision 包络后积分。
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    r = np.concatenate([[0.0], recall, [recall[-1]]])
    p = np.concatenate([[precision[0] if precision.size else 0.0],
                        precision, [0.0]])
    return float(np.sum((r[1:-1] - r[:-2]) * p[1:-1]))


def detection_map(
    preds: List[Dict[str, torch.Tensor]],
    gts  : List[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    iou_thresh : float,
) -> Dict[str, float]:
    """样本集 mAP：逐类累计 TP/score → AP → 宏平均（无 gt 类跳过）。"""
    aps = []
    per_class: Dict[str, float] = {}
    for k in range(num_classes):
        scores_all, tp_all, num_gt = [], [], 0
        for pred, (gb, gl) in zip(preds, gts):
            sel = pred["labels"] == k
            pb = pred["boxes"][sel].cpu()
            ps = pred["scores"][sel].cpu()
            gsel = gl == k
            g = gb[gsel].cpu()
            num_gt += int(gsel.sum())
            tp = match_detections(pb, ps, g, iou_thresh)
            scores_all.append(ps.numpy())
            tp_all.append(tp.numpy())
        ap = average_precision(np.concatenate(scores_all),
                               np.concatenate(tp_all), num_gt)
        per_class[f"ap_c{k}"] = ap
        if not np.isnan(ap):
            aps.append(ap)
    return {"map": float(np.mean(aps)) if aps else 0.0, **per_class}


def froc(
    preds: List[Dict[str, torch.Tensor]],
    gts  : List[Tuple[torch.Tensor, torch.Tensor]],
    fp_per_vol: Sequence[float],
    iou_thresh: float,
) -> Dict[str, float]:
    """FROC：给定每卷假阳个数阈值下的灵敏度，取均值（类无关口径）。"""
    n_vol = max(len(preds), 1)
    scores_all, tp_all, num_gt = [], [], 0
    for pred, (gb, _gl) in zip(preds, gts):
        pb, ps = pred["boxes"].cpu(), pred["scores"].cpu()
        tp = match_detections(pb, ps, gb.cpu(), iou_thresh)
        scores_all.append(ps.numpy())
        tp_all.append(tp.numpy())
        num_gt += int(gb.shape[0])
    scores = np.concatenate(scores_all)
    tp = np.concatenate(tp_all)
    out: Dict[str, float] = {}
    if num_gt == 0:
        out["froc"] = float("nan")
        return out
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float64)
    fp_cum = np.cumsum(1.0 - tp)
    tp_cum = np.cumsum(tp)
    sens_list = []
    for f in fp_per_vol:
        limit = f * n_vol
        idx = np.searchsorted(fp_cum, limit, side="right") - 1
        sens = float(tp_cum[idx] / num_gt) if idx >= 0 else 0.0
        out[f"sens@{f:g}fp"] = sens
        sens_list.append(sens)
    out["froc"] = float(np.mean(sens_list))
    return out
