"""RetinaNet 头（一阶段 anchor + Focal，医学 3D 检测公认基线，Plan §3.5-1）。

纯卷积 tower，dims 参数化 2D/3D 同一实现；挂在共享 Encoder+Decoder 金字塔上
即 nnDetection 的 Retina U-Net 形态。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from segtask_v1.models.blocks import _CONV

from ...config import DetConfig
from ...losses.det_loss import box_reg_loss, sigmoid_focal_loss
from ...ops import batched_nms, clip_boxes
from ...targets import assign_atss, assign_max_iou, decode_boxes, \
    encode_boxes, generate_anchors

__all__ = ["RetinaHead"]


def _tower(conv, ch: int, n: int = 4) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(n):
        layers += [conv(ch, ch, kernel_size=3, padding=1),
                   nn.GroupNorm(8, ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class RetinaHead(nn.Module):
    """逐层共享的 cls / reg tower + anchor 生成 / 分配 / 编解码。"""

    def __init__(self, in_channels: int, num_classes: int, det: DetConfig,
                 spatial_dims: int):
        super().__init__()
        self.dim = int(spatial_dims)
        self.K = int(num_classes)
        self.det = det
        conv = _CONV[self.dim]
        ratios = list(det.anchor_ratios)
        scales = list(det.anchor_scales)
        z_scales = list(det.anchor_z_scales) if self.dim == 3 else [1.0]
        self.num_anchors = (len(ratios) * len(scales)
                            * (len(z_scales) if self.dim == 3 else 1))
        self.cls_tower = _tower(conv, in_channels)
        self.reg_tower = _tower(conv, in_channels)
        self.cls_logits = conv(in_channels, self.num_anchors * self.K,
                               kernel_size=3, padding=1)
        self.bbox_pred = conv(in_channels, self.num_anchors * 2 * self.dim,
                              kernel_size=3, padding=1)
        # 先验偏置：初始正类概率 ~0.01，稳定 focal 早期训练。
        nn.init.constant_(self.cls_logits.bias, -math.log(99.0))

    # ------------------------------------------------------------------
    def _flatten(self, t: torch.Tensor, per_anchor: int) -> torch.Tensor:
        """(B, A*c, *S) → (B, P*A, c)，位置优先、anchor 次之（与
        generate_anchors 的 reshape 序一致）。"""
        B = t.shape[0]
        t = t.reshape(B, self.num_anchors, per_anchor, *t.shape[2:])
        t = t.flatten(3).permute(0, 3, 1, 2)          # (B, P, A, c)
        return t.reshape(B, -1, per_anchor)

    def _anchors(self, feats: List[torch.Tensor], img_size: Sequence[int]
                 ) -> Tuple[torch.Tensor, List[int]]:
        anchors, counts = [], []
        for lvl, f in enumerate(feats):
            fs = list(f.shape[2:])
            stride = [i / s for i, s in zip(img_size, fs)]
            if self.det.anchor_sizes:
                base = float(self.det.anchor_sizes[lvl])
            else:
                base = 4.0 * max(stride[-2:])
            a = generate_anchors(fs, stride, base, self.det.anchor_ratios,
                                 self.det.anchor_scales,
                                 self.det.anchor_z_scales, f.device)
            anchors.append(a)
            counts.append(a.shape[0])
        return torch.cat(anchors), counts

    def forward(self, feats: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_all, reg_all = [], []
        for f in feats:
            cls_all.append(self._flatten(self.cls_logits(self.cls_tower(f)),
                                         self.K))
            reg_all.append(self._flatten(self.bbox_pred(self.reg_tower(f)),
                                         2 * self.dim))
        return torch.cat(cls_all, dim=1), torch.cat(reg_all, dim=1)

    # ------------------------------------------------------------------
    def compute_loss(self, feats: List[torch.Tensor],
                     gt_boxes: List[torch.Tensor],
                     gt_labels: List[torch.Tensor],
                     img_size: Sequence[int]) -> Dict[str, torch.Tensor]:
        cls_logits, reg_deltas = self.forward(feats)
        anchors, counts = self._anchors(feats, img_size)
        det = self.det
        total_cls = cls_logits.new_zeros(())
        total_reg = cls_logits.new_zeros(())
        num_pos = 0
        for b in range(cls_logits.shape[0]):
            gb, gl = gt_boxes[b].to(anchors), gt_labels[b]
            if det.assigner == "atss":
                m = assign_atss(anchors, gb, counts, det.atss_topk)
            else:
                m = assign_max_iou(anchors, gb, det.pos_iou, det.neg_iou)
            pos = m >= 0
            valid = m >= -1                       # -2 = ignore
            tgt = torch.zeros_like(cls_logits[b])
            if pos.any():
                tgt[pos, gl[m[pos]]] = 1.0
            total_cls = total_cls + sigmoid_focal_loss(
                cls_logits[b][valid], tgt[valid],
                det.focal_alpha, det.focal_gamma)
            if pos.any():
                pred = decode_boxes(reg_deltas[b][pos].float(), anchors[pos])
                if det.reg_loss == "giou":
                    total_reg = total_reg + box_reg_loss(pred, gb[m[pos]],
                                                         "giou")
                else:
                    d = encode_boxes(gb[m[pos]], anchors[pos])
                    total_reg = total_reg + box_reg_loss(
                        reg_deltas[b][pos].float(), d, det.reg_loss)
                num_pos += int(pos.sum())
        norm = max(num_pos, 1)
        return {"cls": total_cls / norm,
                "reg": det.reg_weight * total_reg / norm}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats: List[torch.Tensor], img_size: Sequence[int]
                ) -> List[Dict[str, torch.Tensor]]:
        cls_logits, reg_deltas = self.forward(feats)
        anchors, _ = self._anchors(feats, img_size)
        det = self.det
        out = []
        for b in range(cls_logits.shape[0]):
            scores = cls_logits[b].float().sigmoid()       # (P, K)
            flat = scores.reshape(-1)
            k = min(det.max_dets * 20, flat.numel())
            top, idx = flat.topk(k)
            keep0 = top >= det.score_thresh
            idx = idx[keep0]
            pi, ki = idx // self.K, idx % self.K
            boxes = decode_boxes(reg_deltas[b][pi].float(), anchors[pi])
            boxes = clip_boxes(boxes, img_size)
            keep = batched_nms(boxes, top[keep0], ki, det.nms_iou)
            keep = keep[:det.max_dets]
            out.append({"boxes": boxes[keep], "scores": top[keep0][keep],
                        "labels": ki[keep]})
        return out
