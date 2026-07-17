"""检测推理器：整卷 → patch/slab 滑窗 → 3D 检出 + FROC 评估。

patch 抽取几何与训练（``DetPatchDataset``）逐位一致，按 ``patch_mode``：

* ``cubic``  —— 三轴滑窗（步长 = patch 的 1/2 重叠），窗内检出平移回卷坐标，
  跨窗 3D NMS 去重；
* ``z_axis`` —— z 轴滑窗 + H/W 面内 resize，检出面内缩放回原尺寸、z 平移回
  卷坐标，跨窗 3D NMS；
* ``whole``  —— 全卷 resize 到 patch，检出三轴缩放回卷坐标；
* ``2_5d``   —— 沿 z 逐 slab（步长 = slab 深度 / 2）+ 面内 resize，每 slab
  2D 检出缩放回原尺寸 → ``stitch_slab_detections`` 跨层拼接 3D 框（容忍
  ``det.stitch_max_gap`` 漏检）→ 最终 3D NMS。

推理 AMP 口径同训练（``train.use_amp`` / ``train.amp_dtype``，仅 CUDA）；
``det.tta_flips`` 开启 flip TTA（3D 三轴 7 组合、2.5D 仅 H/W 3 组合，框经
``flip_boxes`` 回翻后并入 NMS/拼接池）。

FROC 统一在 3D 框上评估（两几何同一读数口径，Plan §7-5）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from taskcore.config.core import Config as SegConfig
from taskcore.data.dataset import load_npz_image, resize_3d
from taskcore.engine.base_predictor import BasePredictor

from ..config import DetConfig
from ..data.det_dataset import load_boxes
from ..metrics import froc
from ..ops import batched_nms
from ..targets import flip_boxes, scale_boxes
from .stitching import stitch_slab_detections

logger = logging.getLogger(__name__)


def _grid_offsets(dim: int, patch: int, stride: int) -> List[int]:
    """滑窗左边界序列（末窗贴边，完整覆盖）。"""
    if dim <= patch:
        return [0]
    offs = list(range(0, dim - patch, max(stride, 1)))
    offs.append(dim - patch)
    return sorted(set(offs))


class DetPredictor(BasePredictor):
    """整卷检测推理；``predict_volume`` 返回 3D 框检出。"""

    def __init__(self, model: torch.nn.Module, cfg: SegConfig,
                 det: DetConfig, device: torch.device):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.det = det
        self.device = device
        self.patch = tuple(int(s) for s in cfg.data.patch_size)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.mode = str(cfg.data.patch_mode).lower()

        tc = cfg.train
        self._setup_infer_amp(bool(tc.use_amp))

        # flip TTA 组合（空组合 = 原图；轴为 patch 空间轴）。
        if det.tta_flips:
            axes = (1, 2) if self.spatial_dims == 2 else (0, 1, 2)
            self._tta_combos: List[Tuple[int, ...]] = self.flip_tta_combos(
                axes, include_identity=True)
        else:
            self._tta_combos = [()]

    def _load_volume(self, npz_path: str) -> np.ndarray:
        # 读取走 seg 的 memmap 零拷贝快路径（压缩 npz 自动回退，逐位一致）。
        dc = self.cfg.data
        return load_npz_image(
            npz_path, dc.intensity_min, dc.intensity_max, dc.normalize,
            dc.global_mean, dc.global_std)

    def _extract(self, vol: np.ndarray, off: Sequence[int]) -> np.ndarray:
        sl = tuple(slice(o, o + p) for o, p in zip(off, self.patch))
        out = vol[sl]
        pads = [(0, p - s) for p, s in zip(self.patch, out.shape)]
        if any(b for _, b in pads):
            out = np.pad(out, pads, mode="edge")
        return out

    # ------------------------------------------------------------------
    def _forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """批量前向（AMP + flip TTA）→ 每样本检出 dict（patch 坐标）。

        3D 输入 (B,1,D,H,W)，框 (N,6)；2.5D 输入 (B,D,H,W)，框 (N,4)。
        TTA：翻转输入前向，框沿相同轴回翻，与原图检出拼接（去重交给
        调用方的 NMS / 拼接）。
        """
        box_size = (self.patch if x.ndim == 5
                    else (self.patch[1], self.patch[2]))
        merged: List[Dict[str, List[torch.Tensor]]] = [
            {"boxes": [], "scores": [], "labels": []}
            for _ in range(x.shape[0])]
        for combo in self._tta_combos:
            if combo:
                # 3D (B,1,D,H,W)：空间轴 a → 张量维 a+2；
                # 2.5D 折叠 (B,D,H,W)：空间轴 (1,2)=(H,W) → 张量维 a+1。
                dims = ([a + 2 for a in combo] if x.ndim == 5
                        else [a + 1 for a in combo])
                xin = torch.flip(x, dims=dims)
            else:
                xin = x
            with self._autocast():
                dets = self.model(xin)
            for i, d in enumerate(dets):
                b = d["boxes"].float()
                if x.ndim == 5:
                    for a in combo:
                        b = flip_boxes(b, a, box_size)
                else:
                    for a in combo:
                        b = flip_boxes(b, a - 1, box_size)
                merged[i]["boxes"].append(b)
                merged[i]["scores"].append(d["scores"].float())
                merged[i]["labels"].append(d["labels"])
        return [{k: torch.cat(v) for k, v in m.items()} for m in merged]

    @staticmethod
    def _clamp_boxes(boxes: torch.Tensor,
                     shape: Sequence[int]) -> torch.Tensor:
        """框夹取到卷范围 [0, shape]（半开区间）。"""
        dim = boxes.shape[-1] // 2
        sz = torch.as_tensor(shape, dtype=boxes.dtype, device=boxes.device)
        hi = torch.min(boxes[..., dim:].clamp(min=0),
                       sz.expand_as(boxes[..., dim:]))
        lo = torch.min(boxes[..., :dim].clamp(min=0), hi)
        return torch.cat([lo, hi], dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_cubic(self, vol: np.ndarray) -> Dict[str, torch.Tensor]:
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
            for o, dets in zip(chunk, self._forward(x)):
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
        boxes = self._clamp_boxes(boxes[keep], vol.shape)
        return {"boxes": boxes, "scores": scores[keep],
                "labels": labels[keep]}

    @torch.no_grad()
    def _predict_whole(self, vol: np.ndarray) -> Dict[str, torch.Tensor]:
        pD, pH, pW = self.patch
        D, H, W = vol.shape
        x = torch.from_numpy(
            resize_3d(vol, pD, pH, pW).astype(np.float32, copy=False)
        )[None, None].to(self.device)
        dets = self._forward(x)[0]
        boxes = scale_boxes(dets["boxes"].cpu(),
                            (D / pD, H / pH, W / pW))
        scores, labels = dets["scores"].cpu(), dets["labels"].cpu()
        keep = batched_nms(boxes, scores, labels, self.det.nms_iou)
        boxes = self._clamp_boxes(boxes[keep], vol.shape)
        return {"boxes": boxes, "scores": scores[keep],
                "labels": labels[keep]}

    @torch.no_grad()
    def _predict_z_axis(self, vol: np.ndarray) -> Dict[str, torch.Tensor]:
        pD, pH, pW = self.patch
        D, H, W = vol.shape
        z_offs = _grid_offsets(D, pD, max(pD // 2, 1))
        bs = max(int(self.det.infer_batch_size), 1)
        boxes_all, scores_all, labels_all = [], [], []
        for i in range(0, len(z_offs), bs):
            chunk = z_offs[i:i + bs]
            x = torch.stack([
                torch.from_numpy(
                    resize_3d(self._extract(vol, (oz, 0, 0)), pD, pH, pW)
                    .astype(np.float32, copy=False))
                for oz in chunk])[:, None].to(self.device)
            for oz, dets in zip(chunk, self._forward(x)):
                b = scale_boxes(dets["boxes"].cpu(),
                                (1.0, H / pH, W / pW))
                b[:, 0] += float(oz)
                b[:, 3] += float(oz)
                boxes_all.append(b)
                scores_all.append(dets["scores"].cpu())
                labels_all.append(dets["labels"].cpu())
        boxes = torch.cat(boxes_all)
        scores = torch.cat(scores_all)
        labels = torch.cat(labels_all)
        keep = batched_nms(boxes, scores, labels, self.det.nms_iou)
        boxes = self._clamp_boxes(boxes[keep], vol.shape)
        return {"boxes": boxes, "scores": scores[keep],
                "labels": labels[keep]}

    @torch.no_grad()
    def _predict_2_5d(self, vol: np.ndarray) -> Dict[str, torch.Tensor]:
        pD, pH, pW = self.patch
        D, H, W = vol.shape
        # 步长 1 slab 会指数增加算量；取 d//2 重叠保证跨层链接连续性。
        z_offs = _grid_offsets(D, pD, max(pD // 2, 1))
        bs = max(int(self.det.infer_batch_size), 1)
        slab_dets, slab_z = [], []
        for i in range(0, len(z_offs), bs):
            chunk = z_offs[i:i + bs]
            x = torch.stack([
                torch.from_numpy(
                    resize_3d(self._extract(vol, (oz, 0, 0)), pD, pH, pW)
                    .astype(np.float32, copy=False))
                for oz in chunk]).to(self.device)
            for oz, dets in zip(chunk, self._forward(x)):
                b = scale_boxes(dets["boxes"].cpu(), (H / pH, W / pW))
                slab_dets.append({"boxes": b,
                                  "scores": dets["scores"].cpu(),
                                  "labels": dets["labels"].cpu()})
                slab_z.append([float(oz), float(min(oz + pD, D))])
        res = stitch_slab_detections(
            slab_dets, slab_z, self.det.stitch_link_iou,
            self.det.stitch_min_span, self.det.stitch_max_gap)
        # 拼接后最终 3D NMS：重叠滑窗 / gap 断链残留的同灶多链去重。
        keep = batched_nms(res["boxes"], res["scores"], res["labels"],
                           self.det.nms_iou)
        boxes = self._clamp_boxes(res["boxes"][keep], vol.shape)
        return {"boxes": boxes, "scores": res["scores"][keep],
                "labels": res["labels"][keep]}

    @torch.no_grad()
    def predict_volume(self, npz_path: str) -> Dict[str, torch.Tensor]:
        """→ ``{'boxes': (N,6) 3D 卷坐标, 'scores', 'labels'}``。"""
        vol = self._load_volume(npz_path)
        if self.mode == "2_5d":
            return self._predict_2_5d(vol)
        if self.mode == "whole":
            return self._predict_whole(vol)
        if self.mode == "z_axis":
            return self._predict_z_axis(vol)
        return self._predict_cubic(vol)

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
                    # 只读框真值（image 已在 predict_volume 读过）。
                    gb, gl = load_boxes(
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
