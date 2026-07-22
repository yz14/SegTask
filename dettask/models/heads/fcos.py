"""FCOS 头（anchor-free 逐点 + centerness，Plan §3.5-3）。

逐点回归 distance-to-boundary（2d 个距离），层间按回归距离范围分工；
dims 参数化 2D/3D 同一实现。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from taskcore.models.blocks import get_conv

from ...config import DetConfig
from ...losses.det_loss import box_reg_loss, sigmoid_focal_loss
from ...ops import batched_nms, clip_boxes
from ...targets import grid_points

__all__ = ["FCOSHead"]


def _tower(conv, ch: int, n: int = 4) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(n):
        layers += [conv(ch, ch, kernel_size=3, padding=1),
                   nn.GroupNorm(8, ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class FCOSHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, det: DetConfig,
                 spatial_dims: int):
        super().__init__()
        self.dim = int(spatial_dims)
        self.K = int(num_classes)
        self.det = det
        conv = get_conv(self.dim)
        self.cls_tower = _tower(conv, in_channels)
        self.reg_tower = _tower(conv, in_channels)
        self.cls_logits = conv(in_channels, self.K, kernel_size=3, padding=1)
        self.bbox_pred = conv(in_channels, 2 * self.dim, kernel_size=3,
                              padding=1)
        self.centerness = conv(in_channels, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.cls_logits.bias, -math.log(99.0))

    # ------------------------------------------------------------------
    def _per_level(self, feats: List[torch.Tensor], img_size: Sequence[int]
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, List[Tuple[float, float]]]:
        """→ (cls (B,P,K), reg 距离 (B,P,2d), ctr (B,P), points (P,d),
        每点回归范围)。"""
        cls_all, reg_all, ctr_all, pts_all, ranges = [], [], [], [], []
        n_lvl = len(feats)
        for lvl, f in enumerate(feats):
            fs = list(f.shape[2:])
            stride = [i / s for i, s in zip(img_size, fs)]
            s_ref = max(stride[-2:])
            B = f.shape[0]
            c = self.cls_logits(self.cls_tower(f))
            rt = self.reg_tower(f)
            r = F.relu(self.bbox_pred(rt)) * s_ref     # 距离，尺度随层步长
            t = self.centerness(rt)
            cls_all.append(c.flatten(2).transpose(1, 2))
            reg_all.append(r.flatten(2).transpose(1, 2))
            ctr_all.append(t.flatten(2).transpose(1, 2).squeeze(-1))
            pts = grid_points(fs, stride, f.device)
            pts_all.append(pts)
            # 层分工范围（feats 为 low-res → high-res，即 stride 递减）。
            lo = 0.0 if lvl == n_lvl - 1 else 2.0 * s_ref
            hi = math.inf if lvl == 0 else 8.0 * s_ref
            ranges.extend([(lo, hi)] * pts.shape[0])
        rng = torch.tensor(ranges, device=feats[0].device)      # (P, 2)
        return (torch.cat(cls_all, 1), torch.cat(reg_all, 1),
                torch.cat(ctr_all, 1), torch.cat(pts_all, 0), rng)

    @staticmethod
    def _decode(points: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
        d = points.shape[-1]
        return torch.cat([points - dists[..., :d], points + dists[..., d:]],
                         dim=-1)

    # ------------------------------------------------------------------
    def compute_loss(self, feats: List[torch.Tensor],
                     gt_boxes: List[torch.Tensor],
                     gt_labels: List[torch.Tensor],
                     img_size: Sequence[int]) -> Dict[str, torch.Tensor]:
        cls_l, reg_d, ctr_l, pts, rng = self._per_level(feats, img_size)
        det = self.det
        d = self.dim
        total_cls = cls_l.new_zeros(())
        total_reg = cls_l.new_zeros(())
        total_ctr = cls_l.new_zeros(())
        num_pos = 0
        for b in range(cls_l.shape[0]):
            gb, gl = gt_boxes[b].to(pts), gt_labels[b]
            tgt_cls = torch.zeros_like(cls_l[b])
            if gb.numel() > 0:
                # 点在框内 + 最大回归距离落在该层范围 → 候选；多框取最小体积。
                lo = pts[:, None, :] - gb[None, :, :d]           # (P, G, d)
                hi = gb[None, :, d:] - pts[:, None, :]
                dist = torch.cat([lo, hi], dim=-1)               # (P, G, 2d)
                inside = dist.min(dim=-1).values > 0
                dmax = dist.max(dim=-1).values                   # (P, G)
                in_rng = (dmax >= rng[:, None, 0]) & (dmax <= rng[:, None, 1])
                cand = inside & in_rng
                area = (gb[:, d:] - gb[:, :d]).clamp(min=0).prod(-1)  # (G,)
                area_m = torch.where(cand, area[None].expand_as(cand),
                                     torch.full_like(dmax, math.inf))
                best_area, best_g = area_m.min(dim=1)
                pos = torch.isfinite(best_area)
                if pos.any():
                    g = best_g[pos]
                    tgt_cls[pos, gl[g]] = 1.0
                    dpos = dist[pos, g].float()                  # (Np, 2d)
                    # centerness：逐轴 min/max 比的几何平均
                    # （2D 退化为标准 FCOS 的 sqrt(lr·tb)）。
                    lo_d, hi_d = dpos[:, :d], dpos[:, d:]
                    ctr_tgt = ((torch.min(lo_d, hi_d)
                                / torch.max(lo_d, hi_d).clamp(min=1e-6))
                               .prod(-1).clamp(min=0).pow(1.0 / d))
                    pred_box = self._decode(pts[pos], reg_d[b][pos].float())
                    total_reg = total_reg + box_reg_loss(
                        pred_box, gb[g], det.reg_loss if det.reg_loss !=
                        "smooth_l1" else "giou")
                    total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                        ctr_l[b][pos].float(), ctr_tgt, reduction="sum")
                    num_pos += int(pos.sum())
            total_cls = total_cls + sigmoid_focal_loss(
                cls_l[b], tgt_cls, det.focal_alpha, det.focal_gamma)
        norm = max(num_pos, 1)
        return {"cls": total_cls / norm,
                "reg": det.reg_weight * total_reg / norm,
                "ctr": total_ctr / norm}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats: List[torch.Tensor], img_size: Sequence[int]
                ) -> List[Dict[str, torch.Tensor]]:
        cls_l, reg_d, ctr_l, pts, _ = self._per_level(feats, img_size)
        det = self.det
        out = []
        for b in range(cls_l.shape[0]):
            scores = (cls_l[b].float().sigmoid()
                      * ctr_l[b].float().sigmoid()[:, None]).reshape(-1)
            k = min(det.max_dets * 20, scores.numel())
            top, idx = scores.topk(k)
            keep0 = top >= det.score_thresh
            idx = idx[keep0]
            pi, ki = idx // self.K, idx % self.K
            boxes = self._decode(pts[pi], reg_d[b][pi].float())
            boxes = clip_boxes(boxes, img_size)
            keep = batched_nms(boxes, top[keep0], ki, det.nms_iou)
            keep = keep[:det.max_dets]
            out.append({"boxes": boxes[keep], "scores": top[keep0][keep],
                        "labels": ki[keep]})
        return out
