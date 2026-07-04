"""Faster R-CNN 头（两阶段经典，Plan §3.5-2）。

RPN（anchor + objectness）→ proposal（解码 + NMS）→ ROIAlign
（grid_sample 自实现，2D/3D 同一路径，见 ``dettask.ops.roi_align``）→
两层 FC → 类别 softmax（K+1 含背景）+ 类无关框回归。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segtask_v1.models.blocks import _CONV

from ...config import DetConfig
from ...losses.det_loss import box_reg_loss
from ...ops import batched_nms, clip_boxes, nms, roi_align
from ...targets import assign_max_iou, decode_boxes, encode_boxes, \
    generate_anchors

__all__ = ["FasterRCNNHead"]


class FasterRCNNHead(nn.Module):
    """RPN + ROI 头。ROIAlign 取金字塔最高分辨率层（医学小目标优先）。"""

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
        # RPN。
        self.rpn_conv = nn.Sequential(
            conv(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.rpn_obj = conv(in_channels, self.num_anchors, kernel_size=1)
        self.rpn_reg = conv(in_channels, self.num_anchors * 2 * self.dim,
                            kernel_size=1)
        # ROI 头。
        roi_feat = det.fpn_channels * det.roi_output_size ** self.dim
        self.roi_fc = nn.Sequential(
            nn.Flatten(1), nn.Linear(roi_feat, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.roi_cls = nn.Linear(256, self.K + 1)   # +1 = 背景
        self.roi_reg = nn.Linear(256, 2 * self.dim)  # 类无关

    # ------------------------------------------------------------------
    def _flatten(self, t: torch.Tensor, per_anchor: int) -> torch.Tensor:
        B = t.shape[0]
        t = t.reshape(B, self.num_anchors, per_anchor, *t.shape[2:])
        return t.flatten(3).permute(0, 3, 1, 2).reshape(B, -1, per_anchor)

    def _rpn_forward(self, feats: List[torch.Tensor],
                     img_size: Sequence[int]
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj_all, reg_all, anchors = [], [], []
        for lvl, f in enumerate(feats):
            fs = list(f.shape[2:])
            stride = [i / s for i, s in zip(img_size, fs)]
            base = (float(self.det.anchor_sizes[lvl])
                    if self.det.anchor_sizes else 4.0 * max(stride[-2:]))
            h = self.rpn_conv(f)
            obj_all.append(self._flatten(self.rpn_obj(h), 1).squeeze(-1))
            reg_all.append(self._flatten(self.rpn_reg(h), 2 * self.dim))
            anchors.append(generate_anchors(
                fs, stride, base, self.det.anchor_ratios,
                self.det.anchor_scales, self.det.anchor_z_scales, f.device))
        return (torch.cat(obj_all, 1), torch.cat(reg_all, 1),
                torch.cat(anchors, 0))

    def _proposals(self, obj: torch.Tensor, reg: torch.Tensor,
                   anchors: torch.Tensor, img_size: Sequence[int]
                   ) -> List[torch.Tensor]:
        det = self.det
        props = []
        for b in range(obj.shape[0]):
            scores = obj[b].float().sigmoid()
            k = min(det.rpn_pre_nms_topk, scores.numel())
            top, idx = scores.topk(k)
            boxes = decode_boxes(reg[b][idx].float(), anchors[idx])
            boxes = clip_boxes(boxes, img_size)
            keep = nms(boxes, top, det.rpn_nms_iou)[:det.rpn_post_nms_topk]
            props.append(boxes[keep].detach())
        return props

    def _roi_forward(self, feat: torch.Tensor, rois: List[torch.Tensor],
                     img_size: Sequence[int]
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = torch.cat(rois, 0)
        batch_idx = torch.cat([
            torch.full((r.shape[0],), i, dtype=torch.long, device=feat.device)
            for i, r in enumerate(rois)])
        stride = [i / s for i, s in zip(img_size, feat.shape[2:])]
        pooled = roi_align(feat, boxes, batch_idx,
                           [self.det.roi_output_size] * self.dim, stride)
        h = self.roi_fc(pooled)
        return self.roi_cls(h), self.roi_reg(h), batch_idx

    # ------------------------------------------------------------------
    def compute_loss(self, feats: List[torch.Tensor],
                     gt_boxes: List[torch.Tensor],
                     gt_labels: List[torch.Tensor],
                     img_size: Sequence[int]) -> Dict[str, torch.Tensor]:
        det = self.det
        obj, reg, anchors = self._rpn_forward(feats, img_size)
        B = obj.shape[0]
        # ---- RPN 损失（采样平衡）。
        rpn_cls = obj.new_zeros(())
        rpn_reg_l = obj.new_zeros(())
        n_samp = 0
        for b in range(B):
            gb = gt_boxes[b].to(anchors)
            m = assign_max_iou(anchors, gb, det.rpn_pos_iou, det.rpn_neg_iou)
            pos_idx = (m >= 0).nonzero(as_tuple=True)[0]
            neg_idx = (m == -1).nonzero(as_tuple=True)[0]
            n_pos = min(pos_idx.numel(), det.rpn_batch_per_img // 2)
            n_neg = min(neg_idx.numel(), det.rpn_batch_per_img - n_pos)
            pos_idx = pos_idx[torch.randperm(pos_idx.numel(),
                                             device=obj.device)[:n_pos]]
            neg_idx = neg_idx[torch.randperm(neg_idx.numel(),
                                             device=obj.device)[:n_neg]]
            samp = torch.cat([pos_idx, neg_idx])
            tgt = torch.zeros(samp.shape[0], device=obj.device)
            tgt[:n_pos] = 1.0
            rpn_cls = rpn_cls + F.binary_cross_entropy_with_logits(
                obj[b][samp].float(), tgt, reduction="sum")
            if n_pos > 0:
                d = encode_boxes(gb[m[pos_idx]], anchors[pos_idx])
                rpn_reg_l = rpn_reg_l + box_reg_loss(
                    reg[b][pos_idx].float(), d, "smooth_l1")
            n_samp += samp.shape[0]

        # ---- proposal + gt 混合训练 ROI 头。
        props = self._proposals(obj, reg, anchors, img_size)
        rois, roi_tgt_cls, roi_tgt_box = [], [], []
        for b in range(B):
            gb, gl = gt_boxes[b].to(anchors), gt_labels[b]
            cand = torch.cat([props[b], gb], 0) if gb.numel() else props[b]
            m = assign_max_iou(cand, gb, det.rpn_pos_iou, det.rpn_pos_iou,
                               ensure_gt=False)
            pos_idx = (m >= 0).nonzero(as_tuple=True)[0]
            neg_idx = (m < 0).nonzero(as_tuple=True)[0]
            n_pos = min(pos_idx.numel(),
                        int(det.roi_batch_per_img * det.roi_pos_fraction))
            n_neg = min(neg_idx.numel(), det.roi_batch_per_img - n_pos)
            pos_idx = pos_idx[torch.randperm(pos_idx.numel(),
                                             device=obj.device)[:n_pos]]
            neg_idx = neg_idx[torch.randperm(neg_idx.numel(),
                                             device=obj.device)[:n_neg]]
            samp = torch.cat([pos_idx, neg_idx])
            rois.append(cand[samp])
            cls_t = torch.full((samp.shape[0],), self.K, dtype=torch.long,
                               device=obj.device)      # 背景 = K
            if n_pos > 0:
                cls_t[:n_pos] = gl[m[pos_idx]]
            roi_tgt_cls.append(cls_t)
            box_t = torch.zeros(samp.shape[0], 2 * self.dim,
                                device=obj.device)
            if n_pos > 0:
                box_t[:n_pos] = encode_boxes(gb[m[pos_idx]], cand[pos_idx])
            roi_tgt_box.append(box_t)

        cls_logits, box_deltas, _ = self._roi_forward(feats[-1], rois,
                                                      img_size)
        tgt_cls = torch.cat(roi_tgt_cls)
        tgt_box = torch.cat(roi_tgt_box)
        roi_cls_l = F.cross_entropy(cls_logits.float(), tgt_cls,
                                    reduction="mean")
        fg = tgt_cls < self.K
        roi_reg_l = box_reg_loss(box_deltas[fg].float(), tgt_box[fg],
                                 "smooth_l1") / max(int(fg.sum()), 1)
        norm = max(n_samp, 1)
        return {"rpn_cls": rpn_cls / norm,
                "rpn_reg": rpn_reg_l / norm,
                "roi_cls": roi_cls_l,
                "roi_reg": det.reg_weight * roi_reg_l}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats: List[torch.Tensor], img_size: Sequence[int]
                ) -> List[Dict[str, torch.Tensor]]:
        det = self.det
        obj, reg, anchors = self._rpn_forward(feats, img_size)
        props = self._proposals(obj, reg, anchors, img_size)
        cls_logits, box_deltas, batch_idx = self._roi_forward(
            feats[-1], props, img_size)
        probs = cls_logits.float().softmax(-1)[:, :self.K]     # 去背景
        out = []
        for b in range(obj.shape[0]):
            sel = batch_idx == b
            if not sel.any():
                empty = anchors.new_zeros((0, 2 * self.dim))
                out.append({"boxes": empty,
                            "scores": anchors.new_zeros((0,)),
                            "labels": anchors.new_zeros((0,),
                                                        dtype=torch.long)})
                continue
            p = probs[sel]
            boxes = decode_boxes(box_deltas[sel].float(), props[b])
            boxes = clip_boxes(boxes, img_size)
            scores, labels = p.max(dim=-1)
            keep0 = scores >= det.score_thresh
            keep = batched_nms(boxes[keep0], scores[keep0], labels[keep0],
                               det.nms_iou)[:det.max_dets]
            out.append({"boxes": boxes[keep0][keep],
                        "scores": scores[keep0][keep],
                        "labels": labels[keep0][keep]})
        return out
