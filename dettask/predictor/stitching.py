"""2.5D 跨层拼接：逐 slab 2D 框 → 3D 框（Plan §3.3 stitching）。

相邻 slab 的同类 2D 框按 yx IoU >= ``link_iou`` 贪心链接成链，链的 z 范围
取所覆盖 slab 的并集、yx 范围取加权（按分数）并集、分数取链内最大值；
z 跨度不足 ``min_span`` 个 slab 的链丢弃（Plan §7-5：小病灶召回权衡点）。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from ..ops import box_iou

__all__ = ["stitch_slab_detections"]


def stitch_slab_detections(
    slab_dets: List[Dict[str, torch.Tensor]],
    slab_z    : List[List[float]],       # 每 slab 的 [z_lo, z_hi)（卷坐标）
    link_iou  : float = 0.3,
    min_span  : int = 2,
) -> Dict[str, torch.Tensor]:
    """→ ``{'boxes': (N, 6), 'scores': (N,), 'labels': (N,)}`` 3D 检出。"""
    chains: List[Dict] = []          # {yx_box, score, label, z_lo, z_hi,
    #                                   last_slab, last_box}
    for si, (dets, (z_lo, z_hi)) in enumerate(zip(slab_dets, slab_z)):
        boxes = dets["boxes"]
        for j in range(boxes.shape[0]):
            b, s, l = boxes[j], float(dets["scores"][j]), int(dets["labels"][j])
            best, best_iou = None, link_iou
            for c in chains:
                if c["label"] != l or c["last_slab"] != si - 1:
                    continue
                iou = float(box_iou(b[None], c["last_box"][None])[0, 0])
                if iou >= best_iou:
                    best, best_iou = c, iou
            if best is None:
                chains.append({"yx_box": b.clone(), "score": s, "label": l,
                               "z_lo": z_lo, "z_hi": z_hi, "n_slab": 1,
                               "last_slab": si, "last_box": b})
            else:
                best["yx_box"] = torch.stack(
                    [torch.min(best["yx_box"][:2], b[:2]),
                     torch.max(best["yx_box"][2:], b[2:])]).reshape(-1)
                best["score"] = max(best["score"], s)
                best["z_hi"] = z_hi
                best["n_slab"] += 1
                best["last_slab"] = si
                best["last_box"] = b

    kept = [c for c in chains if c["n_slab"] >= min_span]
    if not kept:
        z = torch.zeros
        return {"boxes": z((0, 6)), "scores": z((0,)),
                "labels": z((0,), dtype=torch.long)}
    boxes3d = torch.stack([
        torch.tensor([c["z_lo"], c["yx_box"][0], c["yx_box"][1],
                      c["z_hi"], c["yx_box"][2], c["yx_box"][3]])
        for c in kept])
    return {
        "boxes": boxes3d,
        "scores": torch.tensor([c["score"] for c in kept]),
        "labels": torch.tensor([c["label"] for c in kept],
                               dtype=torch.long),
    }
