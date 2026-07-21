"""分类推理器：整卷 → patch 网格 → 前向 → MIL 聚合 → 卷级/逐 slice 概率。

* patch 抽取口径与训练（``ClsPatchDataset``）逐位一致：
  - ``whole``            —— 全卷 resize，单样本前向；
  - ``z_axis`` / ``2_5d`` —— z 轴铺格（edge-padded 抽取）+ H/W 面内 resize；
  - ``cubic``            —— 按各轴长 ceil(dim/patch) 铺格（半窗内缩贴边覆盖），
    总数上限 ``cls.eval_patches_per_volume``。
* volume 粒度：patch 概率经 ``cls.agg_mode``（mean/max/lse/topk）聚合为卷级
  概率（MIL：卷中任一处阳性即卷阳性 → max/topk/lse 更贴合，mean 更稳）。
* slice 粒度：patch 的逐 slice 概率按 patch 在卷内的绝对 z 回填，重叠切片
  取 max；输出每卷 (K, Z) 概率矩阵。
* ``cls.tta_flips=True`` 时启用翻转 TTA（口径同 segtask predictor：3D 7 种
  轴组合翻转，2.5D 仅翻 H/W——深度折进通道后翻通道会破坏 conv 权重语义），
  各变体概率取平均；slice 粒度下 z 翻转的输出沿 D 轴翻回。
* ``cfg.train.use_amp=True`` 且 CUDA 设备时前向走 autocast（口径同训练；
  概率在 fp32 下聚合）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from taskcore.config.core import Config as SegConfig
from taskcore.data.dataset import load_npz_image
from taskcore.data.patch_extract import extract_patch_by_mode
from taskcore.engine.base_predictor import BasePredictor

from ..config import ClsConfig, resolve_num_classes

logger = logging.getLogger(__name__)

#: 3D 翻转 TTA 轴组合（输入 (B,1,D,H,W)：2=z, 3=y, 4=x），同 segtask 3D。
_FLIP_SPECS_3D: Tuple[Tuple[int, ...], ...] = (
    (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4))
#: 2.5D 翻转 TTA 轴组合（输入 (B,D,H,W)：2=H, 3=W）；不翻深度通道。
_FLIP_SPECS_2_5D: Tuple[Tuple[int, ...], ...] = ((2,), (3,), (2, 3))


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
                 max_patches: int, patch_mode: str) -> List[Tuple[int, ...]]:
    """patch 中心网格（几何随 ``patch_mode``，与训练抽取口径一致）。

    * ``whole``           —— 单中心（全卷 resize，无铺格语义）；
    * ``z_axis``/``2_5d`` —— 仅沿 z 铺格（H/W 面内 resize，无需面内铺格），
      数量 = min(ceil(D/pD), max_patches)（含半窗重叠的贴边覆盖）；
    * ``cubic``           —— 各轴 ceil(dim/patch) 铺格；总数超过
      ``max_patches`` 时等距下采样。
    """
    if patch_mode == "whole":
        return [(shape[0] // 2, shape[1] // 2, shape[2] // 2)]
    if patch_mode in ("z_axis", "2_5d"):
        n_z = min(max(-(-shape[0] // patch[0]), 1), max(int(max_patches), 1))
        return [(z, shape[1] // 2, shape[2] // 2)
                for z in _grid_1d(shape[0], patch[0], n_z)]
    # cubic：按各轴长度成比例铺格（而非三轴等数），保证长轴不欠覆盖。
    ns = [max(-(-d // p), 1) for d, p in zip(shape, patch)]
    axes = [_grid_1d(d, p, n) for d, p, n in zip(shape, patch, ns)]
    centers = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]
    if len(centers) > max_patches:
        idx = np.linspace(0, len(centers) - 1, max_patches).round().astype(int)
        centers = [centers[i] for i in idx]
    return centers


class ClsPredictor(BasePredictor):
    """整卷分类推理；``predict_volume`` 返回卷级（及可选逐 slice）概率。"""

    def __init__(self, model: torch.nn.Module, cfg: SegConfig, cls: ClsConfig,
                 device: torch.device):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.cls = cls
        self.device = device
        self.patch = tuple(int(s) for s in cfg.data.patch_size)
        self.patch_mode = str(cfg.data.patch_mode).lower()
        self.spatial_dims = int(cfg.model.spatial_dims)
        self.num_classes = resolve_num_classes(cls, cfg)
        # AMP 前向（口径同训练）：CUDA + use_amp 时 autocast。
        self._setup_infer_amp(bool(cfg.train.use_amp))
        self._flip_specs: Tuple[Tuple[int, ...], ...] = ()
        if bool(getattr(cls, "tta_flips", False)):
            self._flip_specs = (_FLIP_SPECS_3D if self.spatial_dims == 3
                                else _FLIP_SPECS_2_5D)

    def _load_volume(self, npz_path: str) -> np.ndarray:
        dc = self.cfg.data
        return load_npz_image(
            npz_path, dc.intensity_min, dc.intensity_max, dc.normalize,
            dc.global_mean, dc.global_std)

    def _extract(self, vol: np.ndarray,
                 center: Tuple[int, ...]) -> np.ndarray:
        """按 patch_mode 抽严格 (pD,pH,pW) patch（与训练数据集同一几何）。"""
        return extract_patch_by_mode(
            vol, self.patch_mode, center, self.patch, is_label=False)



    def _post(self, logits: torch.Tensor) -> torch.Tensor:
        single = not self.cls.multi_label
        return (torch.softmax(logits.float(), dim=1) if single
                else torch.sigmoid(logits.float()))

    def _forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        """micro-batch 前向 → 概率；启用 TTA 时各翻转变体概率取平均。

        slice 粒度（输出 (B, K, D)）下，3D 输入含 z 翻转（dim=2）的变体沿
        输出 D 轴翻回后再平均；volume 粒度输出无空间轴，无需回翻。
        """
        with self._autocast():
            prob = self._post(self.model(x))
            if not self._flip_specs:
                return prob
            total = prob
            for dims in self._flip_specs:
                p = self._post(self.model(torch.flip(x, dims)))
                if p.ndim == 3 and self.spatial_dims == 3 and 2 in dims:
                    p = torch.flip(p, (2,))
                total = total + p
            return total / float(1 + len(self._flip_specs))

    @torch.no_grad()
    def predict_volume(self, npz_path: str) -> Dict[str, np.ndarray]:
        """→ ``{'volume_probs': (K,)[, 'slice_probs': (K, Z)]}``。"""
        vol = self._load_volume(npz_path)
        max_patches = int(self.cls.eval_patches_per_volume)
        # slice 粒度：z 铺格不受上限截断（否则厚卷未覆盖 z 的逐 slice 概率
        # 恒 0，等效静默假阴性）；volume 粒度维持上限控制推理成本。
        if (self.cls.label_granularity == "slice"
                and self.patch_mode in ("z_axis", "2_5d")):
            max_patches = max(-(-vol.shape[0] // self.patch[0]), max_patches)
        centers = grid_centers(vol.shape, self.patch, max_patches,
                               self.patch_mode)
        batch = []
        for c in centers:
            p = self._extract(vol, c)
            t = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))
            if self.spatial_dims == 3:
                t = t.unsqueeze(0)
            batch.append(t)
        # micro-batch 前向，避免大卷一次性堆叠全部 patch 致 OOM。
        bs = max(int(self.cls.infer_batch_size), 1)
        probs = torch.cat([
            self._forward_probs(torch.stack(batch[i:i + bs]).to(self.device))
            for i in range(0, len(batch), bs)]).cpu()

        out: Dict[str, np.ndarray] = {}
        if self.cls.label_granularity == "volume":
            out["volume_probs"] = aggregate_probs(
                probs, self.cls.agg_mode, self.cls.agg_topk,
                self.cls.agg_lse_r).numpy()
        else:
            # (N, K, D)：按 patch 绝对 z 回填，重叠取 max；卷级 = slice 概率
            # 再走同一 MIL 聚合。
            z_dim = vol.shape[0]
            slice_probs = np.zeros((self.num_classes, z_dim), dtype=np.float32)
            d = self.patch[0]
            probs_np = probs.numpy()
            if self.patch_mode == "whole":
                # 全卷 resize：patch 第 j 个 slice ↔ 原卷 z 按比例最近邻映射。
                src = np.minimum((np.arange(z_dim) * d) // max(z_dim, 1),
                                 d - 1)
                slice_probs = probs_np[0][:, src]
            else:
                for (cz, _, _), p in zip(centers, probs_np):
                    lo = cz - d // 2
                    for j in range(d):
                        z = min(max(lo + j, 0), z_dim - 1)
                        slice_probs[:, z] = np.maximum(slice_probs[:, z],
                                                       p[:, j])
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
