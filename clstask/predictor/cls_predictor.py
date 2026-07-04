"""分类推理器：整卷 → patch 网格 → 前向 → MIL 聚合 → 卷级/逐 slice 概率。

* patch 网格：z 轴（及 3D cubic 下 H/W）均匀铺格覆盖整卷，数量上限
  ``cls.eval_patches_per_volume``（与验证抽样口径一致）。
* volume 粒度：patch 概率经 ``cls.agg_mode``（mean/max/lse/topk）聚合为卷级
  概率（MIL：卷中任一处阳性即卷阳性 → max/topk/lse 更贴合，mean 更稳）。
* slice 粒度：patch 的逐 slice 概率按 patch 在卷内的绝对 z 回填，重叠切片
  取 max；输出每卷 (K, Z) 概率矩阵。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from segtask_v1.config import Config as SegConfig
from segtask_v1.data.dataset import preprocess_image

from ..config import ClsConfig, resolve_num_classes
from ..data.cls_dataset import _extract_cubic_patch

logger = logging.getLogger(__name__)


def aggregate_probs(probs: torch.Tensor, mode: str, topk: int = 3,
                    lse_r: float = 4.0) -> torch.Tensor:
    """patch 概率 (N, K) → 卷级 (K,)。"""
    if mode == "mean":
        return probs.mean(dim=0)
    if mode == "max":
        return probs.amax(dim=0)
    if mode == "topk":
        k = min(int(topk), probs.shape[0])
        return probs.topk(k, dim=0).values.mean(dim=0)
    if mode == "lse":
        r = float(lse_r)
        return torch.logsumexp(r * probs, dim=0).div(r) - \
            torch.log(torch.tensor(float(probs.shape[0]))).div(r)
    raise ValueError(f"Unknown agg_mode: {mode!r}")


def _grid_1d(dim: int, patch: int, n: int) -> List[int]:
    """沿单轴取 n 个 patch 中心（均匀覆盖，含端点内缩 patch/2）。"""
    if dim <= patch:
        return [dim // 2]
    lo, hi = patch // 2, dim - (patch - patch // 2)
    if n <= 1:
        return [(lo + hi) // 2]
    return sorted({int(round(lo + (hi - lo) * i / (n - 1)))
                   for i in range(n)})


def grid_centers(shape: Sequence[int], patch: Sequence[int],
                 max_patches: int, spatial_dims: int) -> List[Tuple[int, ...]]:
    """patch 中心网格。2.5D / z_axis 沿 z 铺格；3D cubic 三轴铺格。"""
    if spatial_dims == 2:
        zs = _grid_1d(shape[0], patch[0], max_patches)
        return [(z, shape[1] // 2, shape[2] // 2) for z in zs]
    per_axis = max(int(round(max_patches ** (1 / 3))), 1)
    axes = [_grid_1d(d, p, per_axis) for d, p in zip(shape, patch)]
    centers = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]
    if len(centers) > max_patches:
        idx = np.linspace(0, len(centers) - 1, max_patches).round().astype(int)
        centers = [centers[i] for i in idx]
    return centers


class ClsPredictor:
    """整卷分类推理；``predict_volume`` 返回卷级（及可选逐 slice）概率。"""

    def __init__(self, model: torch.nn.Module, cfg: SegConfig, cls: ClsConfig,
                 device: torch.device):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.cls = cls
        self.device = device
        self.patch = tuple(int(s) for s in cfg.data.patch_size)
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.num_classes = resolve_num_classes(cls, cfg)

    def _load_volume(self, npz_path: str) -> np.ndarray:
        dc = self.cfg.data
        with np.load(npz_path, allow_pickle=True) as f:
            return preprocess_image(
                f["image"], dc.intensity_min, dc.intensity_max, dc.normalize,
                dc.global_mean, dc.global_std, inplace=False)

    @torch.no_grad()
    def predict_volume(self, npz_path: str) -> Dict[str, np.ndarray]:
        """→ ``{'volume_probs': (K,)[, 'slice_probs': (K, Z)]}``。"""
        vol = self._load_volume(npz_path)
        centers = grid_centers(vol.shape, self.patch,
                               self.cls.eval_patches_per_volume,
                               self.spatial_dims)
        batch = []
        for c in centers:
            p = _extract_cubic_patch(vol, c, self.patch)
            t = torch.from_numpy(p.astype(np.float32, copy=False))
            if self.spatial_dims == 3:
                t = t.unsqueeze(0)
            batch.append(t)
        x = torch.stack(batch).to(self.device)
        logits = self.model(x)
        single = not self.cls.multi_label
        probs = (torch.softmax(logits, dim=1) if single
                 else torch.sigmoid(logits))

        out: Dict[str, np.ndarray] = {}
        if self.cls.label_granularity == "volume":
            out["volume_probs"] = aggregate_probs(
                probs.cpu(), self.cls.agg_mode, self.cls.agg_topk,
                self.cls.agg_lse_r).numpy()
        else:
            # (N, K, D)：按 patch 绝对 z 回填，重叠取 max；卷级 = slice 概率
            # 再走同一 MIL 聚合。
            z_dim = vol.shape[0]
            slice_probs = np.zeros((self.num_classes, z_dim), dtype=np.float32)
            d = self.patch[0]
            probs_np = probs.cpu().numpy()
            for (cz, _, _), p in zip(centers, probs_np):
                lo = cz - d // 2
                for j in range(d):
                    z = min(max(lo + j, 0), z_dim - 1)
                    slice_probs[:, z] = np.maximum(slice_probs[:, z], p[:, j])
            out["slice_probs"] = slice_probs
            out["volume_probs"] = aggregate_probs(
                torch.from_numpy(slice_probs.T), self.cls.agg_mode,
                self.cls.agg_topk, self.cls.agg_lse_r).numpy()
        return out

    @torch.no_grad()
    def predict_dir(self, npz_paths: Sequence[str],
                    out_dir: str) -> Dict[str, np.ndarray]:
        """批量推理；卷级概率写 ``predictions.csv``，逐 slice 概率各存 npz。"""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results: Dict[str, np.ndarray] = {}
        rows = ["pid," + ",".join(f"prob_{k}" for k in
                                  range(self.num_classes))]
        for path in npz_paths:
            pid = Path(path).stem
            res = self.predict_volume(path)
            results[pid] = res["volume_probs"]
            rows.append(pid + "," + ",".join(
                f"{v:.6f}" for v in res["volume_probs"]))
            if "slice_probs" in res:
                np.savez_compressed(out_path / f"{pid}_slice_probs.npz",
                                    slice_probs=res["slice_probs"])
            logger.info("Predicted %s: %s", pid,
                        np.round(res["volume_probs"], 4))
        (out_path / "predictions.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
        return results


__all__ = ["ClsPredictor", "aggregate_probs", "grid_centers"]
