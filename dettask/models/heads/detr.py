"""Deformable-DETR 风格头（Transformer 集合预测，免 NMS，Plan §3.5-4）。

依赖克制（Plan §7-7）：可变形注意力用纯 PyTorch ``grid_sample`` 实现
（query 预测采样偏移 + 注意力权重，在单尺度特征图上采样），2D/3D 同一实现，
不引入 CUDA 扩展。查询含可学习参考点，逐层框细化；匈牙利匹配 +
（focal-BCE + L1 + GIoU）集合损失。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import DetConfig
from ...losses.det_loss import hungarian_match, sigmoid_focal_loss
from ...ops import center_size_to_box, clip_boxes, generalized_box_iou

__all__ = ["DETRHead"]


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class DeformableCrossAttention(nn.Module):
    """单尺度可变形交叉注意力：每 query 每 head 采 ``num_points`` 个偏移点。

    grid_sample 坐标序：2D = (x, y)；3D = (x, y, z)——与体素轴 (y, x) /
    (z, y, x) 相反，采样前翻转。
    """

    def __init__(self, dim: int, hidden: int, heads: int, points: int):
        super().__init__()
        self.dim, self.heads, self.points = dim, heads, points
        self.head_dim = hidden // heads
        self.offset = nn.Linear(hidden, heads * points * dim)
        self.weight = nn.Linear(hidden, heads * points)
        self.value_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.offset.weight)
        # 初始偏移：单位球均匀方向，稳定训练早期。
        with torch.no_grad():
            g = torch.randn(heads * points, dim)
            g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self.offset.bias.copy_(g.reshape(-1) * 0.05)
        nn.init.zeros_(self.weight.bias)

    def forward(self, query: torch.Tensor, feat_v: torch.Tensor,
                ref: torch.Tensor) -> torch.Tensor:
        """query (B,Q,H*Dh)，feat_v (B,H*Dh,*S)，ref (B,Q,dim) ∈ [0,1]。"""
        B, Q, _ = query.shape
        H, P, d = self.heads, self.points, self.dim
        off = self.offset(query).reshape(B, Q, H, P, d)
        w = self.weight(query).reshape(B, Q, H, P).softmax(dim=-1)
        loc = (ref[:, :, None, None, :] + off).clamp(0, 1) * 2 - 1  # [-1,1]
        loc = loc.flip(-1)                                # 体素序 → xy(z) 序
        v = self.value_proj(feat_v.flatten(2).transpose(1, 2))
        v = v.transpose(1, 2).reshape(B * H, self.head_dim, *feat_v.shape[2:])
        grid = loc.permute(0, 2, 1, 3, 4).reshape(B * H, Q, P, d)
        if d == 3:
            grid = grid.unsqueeze(2)                      # (BH, Q, 1, P, 3)
            samp = F.grid_sample(v, grid, mode="bilinear",
                                 align_corners=False)     # (BH, Dh, Q, 1, P)
            samp = samp.squeeze(3)
        else:
            samp = F.grid_sample(v, grid, mode="bilinear",
                                 align_corners=False)     # (BH, Dh, Q, P)
        samp = samp.reshape(B, H, self.head_dim, Q, P)
        out = (samp * w.permute(0, 2, 1, 3)[:, :, None]).sum(-1)  # (B,H,Dh,Q)
        out = out.permute(0, 3, 1, 2).reshape(B, Q, H * self.head_dim)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, hidden: int, heads: int, points: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, heads,
                                               batch_first=True)
        self.cross_attn = DeformableCrossAttention(dim, hidden, heads, points)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(),
                                 nn.Linear(hidden * 4, hidden))
        self.n1 = nn.LayerNorm(hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.n3 = nn.LayerNorm(hidden)

    def forward(self, q: torch.Tensor, feat_v: torch.Tensor,
                ref: torch.Tensor) -> torch.Tensor:
        a, _ = self.self_attn(q, q, q)
        q = self.n1(q + a)
        q = self.n2(q + self.cross_attn(q, feat_v, ref))
        return self.n3(q + self.ffn(q))


class DETRHead(nn.Module):
    """作用在金字塔最低分辨率层；框以归一化 center/size 参数化并逐层细化。"""

    def __init__(self, in_channels: int, num_classes: int, det: DetConfig,
                 spatial_dims: int):
        super().__init__()
        self.dim = int(spatial_dims)
        self.K = int(num_classes)
        self.det = det
        hidden = det.detr_hidden_dim
        self.input_proj = nn.Conv3d if self.dim == 3 else nn.Conv2d
        self.input_proj = self.input_proj(in_channels, hidden, kernel_size=1)
        self.query_embed = nn.Embedding(det.num_queries, hidden)
        self.ref_embed = nn.Embedding(det.num_queries, self.dim)
        nn.init.uniform_(self.ref_embed.weight, 0.05, 0.95)
        self.layers = nn.ModuleList(
            DecoderLayer(self.dim, hidden, det.detr_num_heads,
                         det.detr_num_points)
            for _ in range(det.detr_dec_layers))
        self.cls_head = nn.Linear(hidden, self.K)
        self.box_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * self.dim))   # Δcenter + log-size（归一化）
        nn.init.constant_(self.cls_head.bias, -math.log(99.0))

    # ------------------------------------------------------------------
    def forward(self, feats: List[torch.Tensor], img_size: Sequence[int]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ (cls_logits (B,Q,K), boxes (B,Q,2d) patch 坐标)。"""
        feat = self.input_proj(feats[0])            # 最低分辨率层
        B = feat.shape[0]
        q = self.query_embed.weight[None].expand(B, -1, -1)
        ref = self.ref_embed.weight[None].expand(B, -1, -1)  # (B,Q,d) ∈ (0,1)
        size = torch.as_tensor(img_size, dtype=feat.dtype, device=feat.device)
        for layer in self.layers:
            q = layer(q, feat, ref)
        delta = self.box_head(q)                    # (B,Q,2d)
        center = (_inverse_sigmoid(ref) + delta[..., :self.dim]).sigmoid()
        scale = (delta[..., self.dim:] - 2.0).sigmoid()   # 初始偏小尺寸
        boxes = center_size_to_box(center * size, scale * size)
        return self.cls_head(q), boxes

    # ------------------------------------------------------------------
    def compute_loss(self, feats: List[torch.Tensor],
                     gt_boxes: List[torch.Tensor],
                     gt_labels: List[torch.Tensor],
                     img_size: Sequence[int]) -> Dict[str, torch.Tensor]:
        cls_l, boxes = self.forward(feats, img_size)
        det = self.det
        size = torch.as_tensor(img_size, dtype=boxes.dtype,
                               device=boxes.device)
        total_cls = cls_l.new_zeros(())
        total_l1 = cls_l.new_zeros(())
        total_giou = cls_l.new_zeros(())
        num_pos = 0
        for b in range(cls_l.shape[0]):
            gb, gl = gt_boxes[b].to(boxes), gt_labels[b]
            qi, gi = hungarian_match(
                cls_l[b], boxes[b], gb, gl, det.detr_cls_weight,
                det.detr_l1_weight, det.detr_giou_weight, norm_size=size)
            tgt = torch.zeros_like(cls_l[b])
            if qi.numel() > 0:
                tgt[qi, gl[gi]] = 1.0
                pb = boxes[b][qi].float() / size.repeat(2)
                tb = gb[gi].float() / size.repeat(2)
                total_l1 = total_l1 + (pb - tb).abs().sum()
                giou = generalized_box_iou(boxes[b][qi].float(),
                                           gb[gi].float()).diagonal()
                total_giou = total_giou + (1.0 - giou).sum()
                num_pos += int(qi.numel())
            total_cls = total_cls + sigmoid_focal_loss(
                cls_l[b], tgt, det.focal_alpha, det.focal_gamma)
        norm = max(num_pos, 1)
        return {"cls": det.detr_cls_weight * total_cls / norm,
                "l1": det.detr_l1_weight * total_l1 / norm,
                "giou": det.detr_giou_weight * total_giou / norm}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats: List[torch.Tensor], img_size: Sequence[int]
                ) -> List[Dict[str, torch.Tensor]]:
        cls_l, boxes = self.forward(feats, img_size)
        det = self.det
        out = []
        for b in range(cls_l.shape[0]):
            scores = cls_l[b].float().sigmoid().reshape(-1)   # (Q*K,)
            k = min(det.max_dets, scores.numel())
            top, idx = scores.topk(k)
            keep = top >= det.score_thresh
            idx = idx[keep]
            qi, ki = idx // self.K, idx % self.K
            bx = clip_boxes(boxes[b][qi].float(), img_size)
            out.append({"boxes": bx, "scores": top[keep], "labels": ki})
        return out
