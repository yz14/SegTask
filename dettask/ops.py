"""dettask 通用框算子（纯 PyTorch，2D/3D 同构）。

框格式（体素坐标，半开区间 [lo, hi)）：

* 2D —— ``(N, 4)  = [y1, x1, y2, x2]``；
* 3D —— ``(N, 6)  = [z1, y1, x1, z2, y2, x2]``。

即 ``(N, 2*dim)``，前 dim 列为下界、后 dim 列为上界。所有算子以
``dim = boxes.shape[-1] // 2`` 自适应 2D/3D，避免维护两份实现
（依赖克制：不引入 torchvision，NMS / ROIAlign 自实现）。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "box_area", "box_iou", "generalized_box_iou", "clip_boxes",
    "nms", "batched_nms", "roi_align", "box_center_size", "center_size_to_box",
]


def _split(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = boxes.shape[-1] // 2
    return boxes[..., :dim], boxes[..., dim:]


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """(N, 2d) → (N,) 面积/体积（半开区间，负边长截为 0）。"""
    lo, hi = _split(boxes)
    return (hi - lo).clamp(min=0).prod(dim=-1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, 2d) × (M, 2d) → (N, M) IoU。"""
    dim = a.shape[-1] // 2
    lo = torch.max(a[:, None, :dim], b[None, :, :dim])
    hi = torch.min(a[:, None, dim:], b[None, :, dim:])
    inter = (hi - lo).clamp(min=0).prod(dim=-1)          # (N, M)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / union.clamp(min=1e-7)


def generalized_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, 2d) × (M, 2d) → (N, M) GIoU ∈ [-1, 1]。"""
    dim = a.shape[-1] // 2
    iou = box_iou(a, b)
    lo = torch.min(a[:, None, :dim], b[None, :, :dim])
    hi = torch.max(a[:, None, dim:], b[None, :, dim:])
    hull = (hi - lo).clamp(min=0).prod(dim=-1)
    inter_lo = torch.max(a[:, None, :dim], b[None, :, :dim])
    inter_hi = torch.min(a[:, None, dim:], b[None, :, dim:])
    inter = (inter_hi - inter_lo).clamp(min=0).prod(dim=-1)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return iou - (hull - union) / hull.clamp(min=1e-7)


def clip_boxes(boxes: torch.Tensor, size) -> torch.Tensor:
    """把框裁进 ``[0, size)``；``size`` 为 (d,) 空间尺寸（与坐标同序）。"""
    dim = boxes.shape[-1] // 2
    sz = torch.as_tensor(size, dtype=boxes.dtype, device=boxes.device)
    lo = boxes[..., :dim].clamp(min=0)
    hi = torch.min(boxes[..., dim:], sz)
    return torch.cat([torch.min(lo, hi), hi], dim=-1)


def box_center_size(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lo, hi = _split(boxes)
    return (lo + hi) * 0.5, (hi - lo)


def center_size_to_box(center: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    half = size * 0.5
    return torch.cat([center - half, center + half], dim=-1)


def nms(boxes: torch.Tensor, scores: torch.Tensor,
        iou_thresh: float) -> torch.Tensor:
    """贪心 NMS（2D/3D 通用），返回保留索引（按分数降序）。"""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = box_iou(boxes[i:i + 1], boxes[rest]).squeeze(0)
        order = rest[iou <= iou_thresh]
    return torch.stack(keep) if keep else torch.empty(
        0, dtype=torch.long, device=boxes.device)


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor,
                labels: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """逐类 NMS：给每类框加大偏移使类间不相交，再统一 NMS。"""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    dim = boxes.shape[-1] // 2
    max_coord = boxes.max()
    offsets = labels.to(boxes.dtype) * (max_coord + 1)
    shifted = boxes + offsets[:, None].repeat(1, 2 * dim)
    return nms(shifted, scores, iou_thresh)


def roi_align(features: torch.Tensor, boxes: torch.Tensor,
              batch_idx: torch.Tensor, output_size,
              stride) -> torch.Tensor:
    """grid_sample 版 ROIAlign（2D/3D 通用，可反传）。

    Args:
        features: (B, C, *S) 特征图。
        boxes: (N, 2d) 输入图坐标系框。
        batch_idx: (N,) 每框所属 batch 索引。
        output_size: (d,) ROI 输出网格尺寸。
        stride: (d,) 特征图相对输入的逐轴步长。
    Returns:
        (N, C, *output_size)
    """
    dim = boxes.shape[-1] // 2
    n = boxes.shape[0]
    C = features.shape[1]
    out_sz = [int(s) for s in output_size]
    if n == 0:
        return features.new_zeros((0, C, *out_sz))
    stride_t = torch.as_tensor(stride, dtype=boxes.dtype, device=boxes.device)
    feat_sz = torch.as_tensor(features.shape[2:], dtype=boxes.dtype,
                              device=boxes.device)
    lo = boxes[:, :dim] / stride_t                      # 特征图坐标
    hi = boxes[:, dim:] / stride_t
    # 每 ROI 均匀采样网格中心（align_corners=False 语义）。
    grids = []
    for ax in range(dim):
        steps = out_sz[ax]
        frac = (torch.arange(steps, dtype=boxes.dtype, device=boxes.device)
                + 0.5) / steps                          # (steps,)
        coord = lo[:, ax, None] + frac[None, :] * (hi[:, ax] - lo[:, ax])[:, None]
        # 体素中心对齐：feature grid_sample 归一化到 [-1, 1]。
        coord = coord / feat_sz[ax] * 2 - 1
        grids.append(coord)                             # (N, steps)
    if dim == 2:
        gy = grids[0][:, :, None].expand(n, out_sz[0], out_sz[1])
        gx = grids[1][:, None, :].expand(n, out_sz[0], out_sz[1])
        grid = torch.stack([gx, gy], dim=-1)            # (N, oh, ow, 2) x,y 序
    else:
        gz = grids[0][:, :, None, None].expand(n, *out_sz)
        gy = grids[1][:, None, :, None].expand(n, *out_sz)
        gx = grids[2][:, None, None, :].expand(n, *out_sz)
        grid = torch.stack([gx, gy, gz], dim=-1)        # (N, od, oh, ow, 3)
    feats = features[batch_idx.long()]                  # (N, C, *S)
    return F.grid_sample(feats, grid, mode="bilinear", align_corners=False)
