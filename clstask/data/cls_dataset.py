"""分类 patch 数据集：npz 卷 → (image patch, 分类 target)。

复用 segtask 的 npz IO / 预处理（``preprocess_image``），patch 抽取口径与
ssltask ``LabeledPatchDataset`` 一致（cubic 随机中心、越界 edge 复制）。

标签派生（见 ``clstask.config.ClsConfig``）：

* ``label_source='mask'`` —— 由分割 mask 弱标签派生：
  - volume 粒度 → 每前景类"patch 内是否出现" → target (K,)；
  - slice  粒度 → 每前景类"每 z 切片是否出现"  → target (K, D)。
* ``label_source='table'`` —— 显式标签表（pid → 标签），volume 粒度；
  - 单标签：target 标量 long（softmax CE）；
  - 多标签：target (K,) float 多热。

输出布局随 ``spatial_dims`` 切换（与 SSL 预训练折叠口径一致）：

* ``spatial_dims==3`` —— ``image (1, D, H, W)``；
* ``spatial_dims==2`` —— 2.5D 折叠：``image (D, H, W)``（C=D 折进通道）。
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from segtask_v1.data.dataset import preprocess_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 标签表
# ---------------------------------------------------------------------------
def load_label_table(path: str, num_classes: int,
                     multi_label: bool) -> Dict[str, np.ndarray]:
    """读 csv/json 标签表 → ``{pid: target ndarray}``。

    * 单标签：target 为 0-dim int64（类别索引 ∈ [0, K)）。
    * 多标签：target 为 (K,) float32 多热。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"cls.label_table not found: {path!r}")
    raw: Dict[str, object] = {}
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            raw = dict(json.load(f))
    else:  # csv
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or header[0].strip().lower() != "pid":
                raise ValueError(
                    f"label_table csv must start with a 'pid' header column; "
                    f"got {header}.")
            n_cols = len(header) - 1
            for row in reader:
                if not row:
                    continue
                pid = row[0].strip()
                vals = [v.strip() for v in row[1:]]
                raw[pid] = vals[0] if n_cols == 1 and not multi_label else vals

    table: Dict[str, np.ndarray] = {}
    for pid, v in raw.items():
        if multi_label:
            arr = np.asarray(
                [float(x) for x in (v if isinstance(v, (list, tuple)) else [v])],
                dtype=np.float32)
            if arr.shape != (num_classes,):
                raise ValueError(
                    f"label_table entry {pid!r} has shape {arr.shape}; "
                    f"expected ({num_classes},) multi-hot.")
            table[pid] = arr
        else:
            iv = int(v if not isinstance(v, (list, tuple)) else v[0])
            if not 0 <= iv < num_classes:
                raise ValueError(
                    f"label_table entry {pid!r} has class {iv}; must be in "
                    f"[0, {num_classes}).")
            table[pid] = np.asarray(iv, dtype=np.int64)
    if not table:
        raise ValueError(f"label_table {path!r} is empty.")
    return table


def match_table_to_paths(npz_paths: Sequence[str],
                         table: Dict[str, np.ndarray],
                         npz_suffix: str = ".npz") -> List[np.ndarray]:
    """按 npz 基名（去后缀）匹配标签表；缺失即报错（宁缺毋滥）。"""
    targets: List[np.ndarray] = []
    missing: List[str] = []
    for path in npz_paths:
        name = Path(path).name
        pid = name[:-len(npz_suffix)] if name.endswith(npz_suffix) else name
        if pid in table:
            targets.append(table[pid])
        else:
            missing.append(pid)
    if missing:
        raise KeyError(
            f"{len(missing)} volume(s) missing from cls.label_table "
            f"(first few: {missing[:5]}). Provide labels for all volumes or "
            f"exclude them via data.exclude_list.")
    return targets


# ---------------------------------------------------------------------------
# patch 抽取（与 ssltask 探针口径一致）
# ---------------------------------------------------------------------------
def _extract_cubic_patch(vol: np.ndarray, center: Tuple[int, int, int],
                         patch: Tuple[int, int, int]) -> np.ndarray:
    """以 center 为中心抽 patch；越界 edge 复制填充。"""
    slices, pads = [], []
    for dim, c, p in zip(vol.shape, center, patch):
        lo = c - p // 2
        hi = lo + p
        pad_lo = max(-lo, 0)
        pad_hi = max(hi - dim, 0)
        slices.append(slice(max(lo, 0), min(hi, dim)))
        pads.append((pad_lo, pad_hi))
    out = vol[tuple(slices)]
    if any(a or b for a, b in pads):
        out = np.pad(out, pads, mode="edge")
    return out


class ClsPatchDataset(Dataset):
    """分类 patch 数据集（几何无关：3D 或 2.5D 折叠由 ``spatial_dims`` 决定）。

    每 epoch 样本数 = ``len(npz_paths) * samples_per_volume``。训练时随机中心
    抽 patch（可按 ``fg_oversample_ratio`` 概率以前景 voxel 为中心，缓解类不
    平衡）；验证时中心由样本索引确定性派生（epoch 间可复现）。
    """

    def __init__(
        self,
        npz_paths           : Sequence[str],
        patch_size          : Sequence[int],
        num_classes         : int,
        label_granularity   : str = "volume",
        label_source        : str = "mask",
        table_targets       : Optional[Sequence[np.ndarray]] = None,
        fg_values           : Optional[Sequence[float]] = None,
        intensity_min       : float = -1024.0,
        intensity_max       : float = 3071.0,
        normalize           : str = "minmax",
        global_mean         : float = 0.0,
        global_std          : float = 1.0,
        samples_per_volume  : int = 8,
        spatial_dims        : int = 3,
        is_train            : bool = True,
        fg_oversample_ratio : float = 0.0,
        seed                : int = 42):
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError("ClsPatchDataset got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)
        if len(self.patch) != 3:
            raise ValueError(f"patch_size must be [D, H, W]; got {patch_size}.")
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3; got {spatial_dims}.")
        if label_granularity not in ("volume", "slice"):
            raise ValueError(f"bad label_granularity: {label_granularity!r}")
        if label_source not in ("mask", "table"):
            raise ValueError(f"bad label_source: {label_source!r}")
        if label_source == "table":
            if table_targets is None or len(table_targets) != len(self.paths):
                raise ValueError(
                    "label_source='table' requires table_targets aligned with "
                    "npz_paths.")
        self.spatial_dims = int(spatial_dims)
        self.fold_2_5d = self.spatial_dims == 2
        self.num_classes = int(num_classes)
        self.granularity = str(label_granularity)
        self.source = str(label_source)
        self.table_targets = list(table_targets) if table_targets else None
        self.fg_values = ([float(v) for v in fg_values]
                          if fg_values else [1.0])
        if self.source == "mask" and len(self.fg_values) != self.num_classes:
            raise ValueError(
                f"mask label source: len(fg_values)={len(self.fg_values)} "
                f"must equal num_classes={self.num_classes}.")
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.normalize = str(normalize)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)
        self.spv = max(int(samples_per_volume), 1)
        self.is_train = bool(is_train)
        self.fg_ratio = float(fg_oversample_ratio)
        self.seed = int(seed)
        self._needs_label = self.source == "mask" or self.fg_ratio > 0
        logger.info(
            "ClsPatchDataset(%s): %d volumes x %d samples, patch=%s, "
            "spatial_dims=%d (%s), granularity=%s, source=%s, K=%d",
            "train" if is_train else "val", len(self.paths), self.spv,
            self.patch, self.spatial_dims,
            "2.5D folded" if self.fold_2_5d else "3D",
            self.granularity, self.source, self.num_classes)

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    # ------------------------------------------------------------------
    def _load(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with np.load(path, allow_pickle=True) as f:
            if "image" not in f.files:
                raise KeyError(f"npz {path!r} has no 'image' key "
                               f"(keys={list(f.files)}).")
            img = preprocess_image(
                f["image"], self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std,
                inplace=False)
            lbl = None
            if self._needs_label:
                if "label" not in f.files:
                    raise KeyError(
                        f"npz {path!r} has no 'label' key required by "
                        f"label_source='mask' / fg oversampling "
                        f"(keys={list(f.files)}).")
                lbl = np.asarray(f["label"])
        if img.ndim != 3:
            raise ValueError(f"expected 3D volume (D,H,W); got {img.shape} "
                             f"in {path!r}.")
        if lbl is not None and lbl.shape != img.shape:
            raise ValueError(f"image/label shape mismatch in {path!r}: "
                             f"{img.shape} vs {lbl.shape}.")
        return img, lbl

    def _sample_center(self, rng: np.random.Generator, shape: Tuple[int, ...],
                       lbl: Optional[np.ndarray]) -> Tuple[int, int, int]:
        if (self.is_train and lbl is not None and self.fg_ratio > 0
                and rng.random() < self.fg_ratio):
            # 前景中心采样：在 stride 网格上找前景 voxel（控制 argwhere 成本）。
            stride = 4
            sub = lbl[::stride, ::stride, ::stride]
            coords = np.argwhere(sub > 0)
            if coords.shape[0] > 0:
                c = coords[int(rng.integers(coords.shape[0]))] * stride
                return tuple(int(x) for x in c)
        center = []
        for dim, p in zip(shape, self.patch):
            if dim <= p:
                center.append(dim // 2)
            else:
                lo = int(rng.integers(0, dim - p + 1))
                center.append(lo + p // 2)
        return tuple(center)

    def _target_from_mask(self, lbl_patch: np.ndarray) -> torch.Tensor:
        """mask 弱标签：volume → (K,)；slice → (K, D)。"""
        if self.granularity == "volume":
            t = np.asarray(
                [float((lbl_patch == v).any()) for v in self.fg_values],
                dtype=np.float32)
        else:
            t = np.stack(
                [(lbl_patch == v).any(axis=(1, 2)).astype(np.float32)
                 for v in self.fg_values], axis=0)  # (K, D)
        return torch.from_numpy(t)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx % len(self.paths)
        img, lbl = self._load(self.paths[vol_idx])
        if self.is_train:
            rng = np.random.default_rng()
        else:
            # 验证：中心由 (seed, idx) 确定性派生，epoch 间可复现。
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        center = self._sample_center(rng, img.shape, lbl)
        img_patch = _extract_cubic_patch(img, center, self.patch)
        img_t = torch.from_numpy(img_patch.astype(np.float32, copy=False))
        if not self.fold_2_5d:
            img_t = img_t.unsqueeze(0)             # (1, D, H, W)
        # 2.5D：(D, H, W) —— 深度折进通道。

        if self.source == "mask":
            lbl_patch = _extract_cubic_patch(lbl, center, self.patch)
            target = self._target_from_mask(lbl_patch)
        else:
            target = torch.from_numpy(np.asarray(self.table_targets[vol_idx]))
        return {"image": img_t, "target": target}


__all__ = [
    "ClsPatchDataset", "load_label_table", "match_table_to_paths",
]
