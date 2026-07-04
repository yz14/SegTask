"""检测 target 工具：anchor 生成 / 正负分配 / 框编解码 / 增强几何联动。

全部以 ``spatial_dims``（框列数 // 2）参数化，2D 与 3D 同一实现（Plan §3.4）。
坐标序与体素轴一致：2D = (y, x)，3D = (z, y, x)；框为半开区间
``[lo..., hi...]``。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch

from .ops import box_center_size, box_iou, center_size_to_box

__all__ = [
    "generate_anchors", "grid_points", "encode_boxes", "decode_boxes",
    "assign_max_iou", "assign_atss", "flip_boxes", "crop_boxes",
    "slice_boxes_to_2d",
]


# ---------------------------------------------------------------------------
# anchor / point 生成
# ---------------------------------------------------------------------------
def _cell_anchors(base_size: float, ratios: Sequence[float],
                  scales: Sequence[float], z_scales: Sequence[float],
                  dim: int, device, dtype=torch.float32) -> torch.Tensor:
    """单位置 anchor 尺寸组合 → (A, 2*dim) 以原点为中心的框。"""
    sizes = []
    for r in ratios:            # r = h/w
        for s in scales:
            h = base_size * s * math.sqrt(r)
            w = base_size * s / math.sqrt(r)
            if dim == 2:
                sizes.append([h, w])
            else:
                for zs in z_scales:
                    sizes.append([base_size * s * zs, h, w])
    t = torch.tensor(sizes, dtype=dtype, device=device)     # (A, dim)
    return torch.cat([-t / 2, t / 2], dim=-1)               # (A, 2*dim)


def grid_points(feat_shape: Sequence[int], stride: Sequence[float],
                device, dtype=torch.float32) -> torch.Tensor:
    """特征图每个位置的输入坐标系中心点 → (P, dim)。"""
    coords = [
        (torch.arange(int(n), dtype=dtype, device=device) + 0.5) * float(s)
        for n, s in zip(feat_shape, stride)]
    mesh = torch.meshgrid(*coords, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)


def generate_anchors(feat_shape: Sequence[int], stride: Sequence[float],
                     base_size: float, ratios: Sequence[float],
                     scales: Sequence[float], z_scales: Sequence[float],
                     device) -> torch.Tensor:
    """单 FPN 层 anchor → (P*A, 2*dim)，与 head 输出 reshape 序一致
    （位置优先、组合次之）。"""
    dim = len(feat_shape)
    cell = _cell_anchors(base_size, ratios, scales, z_scales, dim, device)
    pts = grid_points(feat_shape, stride, device)            # (P, dim)
    boxes = pts[:, None, :].repeat(1, cell.shape[0], 2) + cell[None]
    return boxes.reshape(-1, 2 * dim)


# ---------------------------------------------------------------------------
# 框编解码（Δcenter/anchor_size + log size 比，faster-rcnn 口径）
# ---------------------------------------------------------------------------
def encode_boxes(gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    gc, gs = box_center_size(gt)
    ac, asz = box_center_size(anchors)
    asz = asz.clamp(min=1e-4)
    return torch.cat([(gc - ac) / asz, torch.log(gs.clamp(min=1e-4) / asz)],
                     dim=-1)


def decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    dim = anchors.shape[-1] // 2
    ac, asz = box_center_size(anchors)
    asz = asz.clamp(min=1e-4)
    center = deltas[..., :dim] * asz + ac
    size = torch.exp(deltas[..., dim:].clamp(max=8.0)) * asz
    return center_size_to_box(center, size)


# ---------------------------------------------------------------------------
# 正负分配
# ---------------------------------------------------------------------------
def assign_max_iou(anchors: torch.Tensor, gt: torch.Tensor,
                   pos_iou: float, neg_iou: float,
                   ensure_gt: bool = True) -> torch.Tensor:
    """max-IoU 分配：返回 (P,) —— 匹配 gt 索引；-1 = 负样本；-2 = ignore。

    ``ensure_gt``：每个 gt 的最大 IoU anchor 强制为正（低 IoU 小目标兜底）。
    """
    P = anchors.shape[0]
    matches = torch.full((P,), -1, dtype=torch.long, device=anchors.device)
    if gt.numel() == 0:
        return matches
    iou = box_iou(anchors, gt)                       # (P, G)
    best_iou, best_gt = iou.max(dim=1)
    matches[best_iou >= pos_iou] = best_gt[best_iou >= pos_iou]
    ignore = (best_iou >= neg_iou) & (best_iou < pos_iou)
    matches[ignore] = -2
    if ensure_gt:
        gt_best_anchor = iou.argmax(dim=0)           # (G,)
        matches[gt_best_anchor] = torch.arange(
            gt.shape[0], device=anchors.device)
    return matches


def assign_atss(anchors: torch.Tensor, gt: torch.Tensor,
                num_level_anchors: Sequence[int],
                topk: int = 9) -> torch.Tensor:
    """ATSS：每层取中心最近 topk 候选，阈值 = 候选 IoU 均值 + 标准差；
    正样本须中心落在 gt 内。返回 (P,) 匹配 gt 索引，-1 = 负。"""
    P = anchors.shape[0]
    matches = torch.full((P,), -1, dtype=torch.long, device=anchors.device)
    if gt.numel() == 0:
        return matches
    dim = anchors.shape[-1] // 2
    iou = box_iou(anchors, gt)                       # (P, G)
    ac, _ = box_center_size(anchors)
    gc, _ = box_center_size(gt)
    dist = (ac[:, None, :] - gc[None, :, :]).pow(2).sum(-1).sqrt()  # (P, G)

    cand_idx: List[torch.Tensor] = []
    start = 0
    for n in num_level_anchors:
        d = dist[start:start + n]
        k = min(topk, n)
        idx = d.topk(k, dim=0, largest=False).indices + start   # (k, G)
        cand_idx.append(idx)
        start += n
    cand = torch.cat(cand_idx, dim=0)                # (K, G)
    G = gt.shape[0]
    cand_iou = iou.gather(0, cand)                   # (K, G)
    thr = cand_iou.mean(dim=0) + cand_iou.std(dim=0, unbiased=False)  # (G,)

    # 中心在 gt 内。
    inside = ((ac[:, None, :] >= gt[None, :, :dim])
              & (ac[:, None, :] <= gt[None, :, dim:])).all(dim=-1)   # (P, G)

    is_pos = torch.zeros_like(iou, dtype=torch.bool)
    for g in range(G):
        ok = cand[:, g][(cand_iou[:, g] >= thr[g])]
        is_pos[ok, g] = True
    is_pos &= inside
    # 一 anchor 多 gt 时取 IoU 最大者。
    iou_masked = torch.where(is_pos, iou, torch.full_like(iou, -1.0))
    best_iou, best_gt = iou_masked.max(dim=1)
    matches[best_iou > 0] = best_gt[best_iou > 0]
    return matches


# ---------------------------------------------------------------------------
# 增强几何联动（Plan §7-4：crop / flip 必须同步作用于框）
# ---------------------------------------------------------------------------
def flip_boxes(boxes: torch.Tensor, axis: int, size: Sequence[int]
               ) -> torch.Tensor:
    """沿 ``axis`` 翻转（半开区间：new_lo = S - hi, new_hi = S - lo）。"""
    dim = boxes.shape[-1] // 2
    out = boxes.clone()
    s = float(size[axis])
    out[..., axis] = s - boxes[..., dim + axis]
    out[..., dim + axis] = s - boxes[..., axis]
    return out


def crop_boxes(boxes: torch.Tensor, labels: torch.Tensor,
               offset: Sequence[int], crop_size: Sequence[int],
               min_visibility: float = 0.25
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """裁剪联动：平移到 crop 坐标 → 裁到 crop 内 → 过滤可见比例过低的框。"""
    dim = boxes.shape[-1] // 2
    off = torch.as_tensor(list(offset) * 2, dtype=boxes.dtype,
                          device=boxes.device)
    shifted = boxes - off
    sz = torch.as_tensor(crop_size, dtype=boxes.dtype, device=boxes.device)
    lo = shifted[..., :dim].clamp(min=0)
    hi = torch.min(shifted[..., dim:],
                   sz.expand_as(shifted[..., dim:]))
    clipped = torch.cat([torch.min(lo, hi), hi], dim=-1)
    orig_area = ((boxes[..., dim:] - boxes[..., :dim]).clamp(min=0)
                 .prod(dim=-1))
    new_area = (hi - lo).clamp(min=0).prod(dim=-1)
    keep = new_area >= min_visibility * orig_area.clamp(min=1e-7)
    keep &= new_area > 0
    return clipped[keep], labels[keep]


def slice_boxes_to_2d(boxes3d: torch.Tensor, labels: torch.Tensor,
                      z_lo: int, z_hi: int, min_overlap: float = 0.25
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D 框 → slab [z_lo, z_hi) 内的 2D 框（Plan：3D 真值切片派生 2D）。

    保留 z 向与 slab 交叠比例 >= ``min_overlap`` 的框，输出其 yx 范围。
    """
    if boxes3d.numel() == 0:
        return boxes3d.new_zeros((0, 4)), labels[:0]
    inter = (torch.min(boxes3d[:, 3], torch.tensor(float(z_hi)))
             - torch.max(boxes3d[:, 0], torch.tensor(float(z_lo)))).clamp(min=0)
    depth = (boxes3d[:, 3] - boxes3d[:, 0]).clamp(min=1e-7)
    keep = inter / depth >= min_overlap
    b = boxes3d[keep]
    return torch.stack([b[:, 1], b[:, 2], b[:, 4], b[:, 5]], dim=-1), \
        labels[keep]
