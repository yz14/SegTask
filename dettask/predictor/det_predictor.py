"""检测推理器：整卷 → patch/slab 滑窗 → 3D 检出 + FROC 评估。

* 3D —— 三轴滑窗（步长 = patch 的 1/2 重叠），窗内检出平移回卷坐标，
  跨窗 3D NMS 去重；
* 2.5D —— 沿 z 逐 slab（步长 = slab 深度 / 2），每 slab 2D 检出 →
  ``stitch_slab_detections`` 跨层拼接 3D 框。

FROC 统一在 3D 框上评估（两几何同一读数口径，Plan §7-5）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from segtask_v1.config import Config as SegConfig
from segtask_v1.data.dataset import preprocess_image

from ..config import DetConfig
from ..data.det_dataset import load_volume_boxes
from ..metrics import froc
from ..ops import batched_nms
from .stitching import stitch_slab_detections

logger = logging.getLogger(__name__)


def _grid_offsets(dim: int, patch: int, stride: int) -> List[int]:
    """滑窗左边界序列（末窗贴边，完整覆盖）。"""
    if dim <= patch:
        return [0]
    offs = list(range(0, dim - patch, max(stride, 1)))
    offs.append(dim - patch)
    return sorted(set(offs))


class DetPredictor:
    """整卷检测推理；``predict_volume`` 返回 3D 框检出。"""

    def __init__(self, model: torch.nn.Module, cfg: SegConfig,
                 det: DetConfig, device: torch.device):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.det = det
        self.device = device
        self.patch = tuple(int(s) for s in cfg.data.patch_size)
        self.spatial_dims = int(cfg.model.spatial_dims)

    def _load_volume(self, npz_path: str) -> np.ndarray:
        dc = self.cfg.data
        with np.load(npz_path, allow_pickle=True) as f:
            return preprocess_image(
                f["image"], dc.intensity_min, dc.intensity_max, dc.normalize,
                dc.global_mean, dc.global_std, inplace=False)

    def _extract(self, vol: np.ndarray, off: Sequence[int]) -> np.ndarray:
        sl = tuple(slice(o, o + p) for o, p in zip(off, self.patch))
        out = vol[sl]
        pads = [(0, p - s) for p, s in zip(self.patch, out.shape)]
        if any(b for _, b in pads):
            out = np.pad(out, pads, mode="edge")
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_3d(self, vol: np.ndarray) -> Dict[str, torch.Tensor]:
        offs = [_grid_offsets(d, p, p // 2)
                for d, p in zip(vol.shape, self.patch)]
        origins = [(oz, oy, ox) for oz in offs[0]
                   for oy in offs[1] for ox in offs[2]]
        bs = max(int(self.det.infer_batch_size), 1)
        boxes_all, scores_all, labels_all = [], [], []
        for i in range(0, len(origins), bs):
            chunk = origins[i:i + bs]
            x = torch.stack([
                torch.from_numpy(self._extract(vol, o)
                                 .astype(np.float32, copy=False))
                for o in chunk])[:, None].to(self.device)
            for o, dets in zip(chunk, self.model(x)):
                shift = torch.tensor(
                    list(o) * 2, dtype=torch.float32,
                    device=dets["boxes"].device)
                boxes_all.append(dets["boxes"] + shift)
                scores_all.append(dets["scores"])
                labels_all.append(dets["labels"])
        boxes = torch.cat(boxes_all).cpu()
        scores = torch.cat(scores_all).cpu()
        labels = torch.cat(labels_all).cpu()
        keep = batched_nms(boxes, scores, labels, self.det.nms_iou)
        return {"boxes": boxes[keep], "scores": scores[keep],
                "labels": labels[keep]}

    @torch.no_grad()
    def _predict_2_5d(self, vol: np.ndarray) -> Dict[str, torch.Tensor]:
        d = self.patch[0]
        # 步长 1 slab 会指数增加算量；取 d//2 重叠保证跨层链接连续性。
        z_offs = _grid_offsets(vol.shape[0], d, max(d // 2, 1))
        bs = max(int(self.det.infer_batch_size), 1)
        slab_dets, slab_z = [], []
        for i in range(0, len(z_offs), bs):
            chunk = z_offs[i:i + bs]
            x = torch.stack([
                torch.from_numpy(self._extract(vol, (oz, 0, 0))
                                 .astype(np.float32, copy=False))
                for oz in chunk]).to(self.device)
            for oz, dets in zip(chunk, self.model(x)):
                slab_dets.append({k: v.cpu() for k, v in dets.items()})
                slab_z.append([float(oz),
                               float(min(oz + d, vol.shape[0]))])
        return stitch_slab_detections(
            slab_dets, slab_z, self.det.stitch_link_iou,
            self.det.stitch_min_span)

    @torch.no_grad()
    def predict_volume(self, npz_path: str) -> Dict[str, torch.Tensor]:
        """→ ``{'boxes': (N,6) 3D 卷坐标, 'scores', 'labels'}``。"""
        vol = self._load_volume(npz_path)
        if self.spatial_dims == 2:
            return self._predict_2_5d(vol)
        return self._predict_3d(vol)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_dir(self, npz_paths: Sequence[str], out_dir: str,
                    evaluate: bool = True) -> Dict[str, float]:
        """批量推理；检出写 ``{pid}_dets.npz`` + ``detections.csv``；
        有真值时计算体级 FROC。"""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        rows = ["pid,z1,y1,x1,z2,y2,x2,score,label"]
        preds: List[Dict[str, torch.Tensor]] = []
        gts: List[Tuple[torch.Tensor, torch.Tensor]] = []
        have_gt = True
        fg_values = [float(v) for v in (self.cfg.data.label_values[1:]
                                        if len(self.cfg.data.label_values) > 1
                                        else [1.0])]
        for path in npz_paths:
            pid = Path(path).stem
            res = self.predict_volume(path)
            preds.append(res)
            np.savez_compressed(
                out_path / f"{pid}_dets.npz",
                boxes=res["boxes"].numpy(), scores=res["scores"].numpy(),
                labels=res["labels"].numpy())
            for b, s, l in zip(res["boxes"], res["scores"], res["labels"]):
                rows.append(pid + "," + ",".join(f"{v:.2f}" for v in b)
                            + f",{s:.4f},{int(l)}")
            logger.info("Predicted %s: %d detection(s)", pid,
                        res["boxes"].shape[0])
            if evaluate and have_gt:
                try:
                    _, gb, gl = load_volume_boxes(
                        path, fg_values, self.det.boxes_from_mask,
                        self.det.min_box_voxels)
                    gts.append((torch.from_numpy(gb), torch.from_numpy(gl)))
                except KeyError:
                    have_gt = False
        (out_path / "detections.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
        metrics: Dict[str, float] = {}
        if evaluate and have_gt and gts:
            metrics = froc(preds, gts, self.det.froc_fp_per_vol,
                           self.det.eval_iou_thresh)
            logger.info("Volume-level FROC: %s",
                        {k: round(v, 4) for k, v in metrics.items()})
        return metrics


__all__ = ["DetPredictor"]
