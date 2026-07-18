"""分类 patch 数据集：npz 卷 → (image patch, 分类 target)。

复用 segtask 的 npz IO / 预处理（``preprocess_image`` / ``resize_3d`` /
``extract_z_patch_padded``），patch 抽取口径与 segtask 各 patch_mode 逐位一致
（保证 SSL / 分割预训练 encoder 看到的输入分布一致）：

* ``patch_mode='cubic'``  —— 3 轴随机中心 cube，越界 edge 复制（同
  ``SegDataset3DCubic``；中心夹匯到安全范围）；
* ``patch_mode='z_axis'`` / ``'2_5d'`` —— z 轴滑窗（edge-padded 保物理 z-FOV）
  + H/W 面内 resize 到 patch_size（同 ``SegDataset3D``）；
* ``patch_mode='whole'``  —— 全卷 resize 到 patch_size（同
  ``SegDataset3DWhole``；不采中心，``fg_oversample_ratio`` 忽略）。

前景过采样直接读 make_data 预计算的 npz 索引（``fg_coords``/``fg_slices``，
含 ``*_cls`` 键时先均匀选类再选点/切片，类均衡，口径同 segtask dataset）；
旧 npz 缺索引键时逐卷惰性回退为全分辨率 argwhere（一次计算、逐 worker 缓存）。

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

验证采样：默认中心由 (seed, idx) 确定性派生（epoch 间可复现）；
``val_grid_coverage=True`` 时改为确定性网格覆盖（z 模式沿 z 等距 bin 中心，
cubic 用 Halton(2,3,5) 低差异序列铺满安全中心域，口径同 segtask）——与推理
的网格铺点一致，选模指标更贴近部署表现。

GPU 增强模式（``gpu_augment=True``，仅训练）：输出始终为未折叠的 3D
``image (1, D, H, W)``；mask 源额外输出 ``label (1, D, H, W)``且不在此派生
target（由 trainer 在 GPU 增强后派生，再按 spatial_dims 折叠），保证空间
变换后 image/target 一致。
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

from taskcore.data.dataset import (
    VolumeCache,
    _group_fg_coords_by_class,
    _group_fg_slices_by_class,
    _halton,
    _open_npz,
    derive_volume_targets,
    extract_z_patch_padded,
    load_npz_image,
    load_npz_label,
    load_npz_label_counts,
    resize_3d,
)

logger = logging.getLogger(__name__)

#: fg 索引惰性回退（旧 npz 无 fg_coords/fg_slices 键）时每卷坐标数上限，
#: 与 make_data 的逐类 cap 同量级，防大器官卷索引占用过多内存。
_FG_FALLBACK_CAP = 50_000


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
    pids: Dict[str, str] = {}
    for path in npz_paths:
        name = Path(path).name
        pid = name[:-len(npz_suffix)] if name.endswith(npz_suffix) else name
        # 递归目录下同名文件会静默共享同一 pid 标签，显式报错。
        if pid in pids and pids[pid] != str(path):
            raise ValueError(
                f"duplicate pid {pid!r} from {pids[pid]!r} and {path!r}; "
                f"npz basenames must be unique for table matching.")
        pids[pid] = str(path)
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
# patch 抽取（与 segtask 口径一致）
# ---------------------------------------------------------------------------
def _extract_cubic_patch(vol: np.ndarray, center: Tuple[int, int, int],
                         patch: Tuple[int, int, int]) -> np.ndarray:
    """以 center 为中心抽 patch；越界 edge 复制填充（同 segtask cubic）。"""
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


def _safe_center_range(
    shape: Tuple[int, ...], patch: Tuple[int, ...],
) -> List[Tuple[int, int]]:
    """逐轴返中心点 (lo, hi) 半开区间（供 randint/clip），使 patch 尽量在界内；
    轴 size < patch 时退为体积中心（接受边界填充）。口径同
    ``SegDataset3DCubic._safe_center_range``。"""
    out = []
    for size, p in zip(shape, patch):
        half = p // 2
        lo, hi = half, size - (p - half)
        if hi <= lo:
            mid = size // 2
            out.append((mid, mid + 1))
        else:
            out.append((lo, hi))
    return out


class ClsPatchDataset(Dataset):
    """分类 patch 数据集（patch 抽取口径由 ``patch_mode`` 决定，3D 或 2.5D
    折叠由 ``spatial_dims`` 决定）。

    每 epoch 样本数 = ``len(npz_paths) * samples_per_volume``。训练时随机中心
    抽 patch（可按 ``fg_oversample_ratio`` 概率以前景为中心，缓解类不平衡）；
    验证时中心由样本索引确定性派生，``val_grid_coverage=True`` 时用确定性
    网格覆盖（与推理铺点同口径）。
    """

    def __init__(
        self,
        npz_paths           : Sequence[str],
        patch_size          : Sequence[int],
        num_classes         : int,
        patch_mode          : str = "cubic",
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
        seed                : int = 42,
        gpu_augment         : bool = False,
        val_grid_coverage   : bool = False,
        cache_enabled       : bool = False,
        cache_max_volumes   : int = 0):
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError("ClsPatchDataset got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)
        if len(self.patch) != 3:
            raise ValueError(f"patch_size must be [D, H, W]; got {patch_size}.")
        self.mode = str(patch_mode).lower()
        if self.mode not in ("whole", "z_axis", "cubic", "2_5d"):
            raise ValueError(f"bad patch_mode: {patch_mode!r}")
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
        self.fg_ratio = (float(fg_oversample_ratio)
                         if self.mode != "whole" else 0.0)
        self.seed = int(seed)
        self.gpu_augment = bool(gpu_augment) and self.is_train
        self.val_grid_coverage = bool(val_grid_coverage) and not self.is_train
        self._needs_label = self.source == "mask"
        # 逐 worker LRU 缓存（复用 segtask VolumeCache：pickle 到 worker 时清空），
        # 缓预处理后的卷，避免 samples_per_volume>1 时同卷重复 IO/预处理。
        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)
        # 逐 worker 训练采样 RNG（惰性创建，口径同 segtask dataset._rng）。
        self._rng_cache: Optional[np.random.Generator] = None
        self._rng_wid  : Optional[int] = None

        # 前景索引（仅训练 + fg_ratio>0）：读 make_data 预计算的 npz 键；
        # 缺键的旧 npz 标 None，__getitem__ 惰性回退全分辨率计算。
        self._fg_index: List[Optional[Tuple[List[np.ndarray], bool]]] = []
        self._fg_fallback: Dict[int, List[np.ndarray]] = {}
        if self.is_train and self.fg_ratio > 0:
            self._build_fg_index()
            if any(e is None for e in self._fg_index):
                # 回退路径要扫 label 卷。
                self._needs_label = True

        logger.info(
            "ClsPatchDataset(%s): %d volumes x %d samples, mode=%s, patch=%s, "
            "spatial_dims=%d (%s), granularity=%s, source=%s, K=%d%s",
            "train" if is_train else "val", len(self.paths), self.spv,
            self.mode, self.patch, self.spatial_dims,
            "2.5D folded" if self.fold_2_5d else "3D",
            self.granularity, self.source, self.num_classes,
            ", val_grid_coverage" if self.val_grid_coverage else "")

    # ------------------------------------------------------------------
    # 前景索引
    # ------------------------------------------------------------------
    def _build_fg_index(self) -> None:
        """从 npz 读 make_data 预计算的 fg 索引（z 模式用 fg_slices，cubic 用
        fg_coords；含 ``*_cls`` 键时逐类分组，先选类再选点，类均衡）。"""
        want_coords = self.mode == "cubic"
        key = "fg_coords" if want_coords else "fg_slices"
        n_missing = 0
        for path in self.paths:
            with _open_npz(path) as f:
                if key not in f.files:
                    self._fg_index.append(None)
                    n_missing += 1
                    continue
                if want_coords:
                    coords = np.asarray(f["fg_coords"], dtype=np.int32)
                    per_cls = _group_fg_coords_by_class(f, coords)
                    groups = per_cls if per_cls else (
                        [coords] if len(coords) else [])
                else:
                    zs = np.asarray(f["fg_slices"], dtype=np.int32)
                    per_cls = _group_fg_slices_by_class(f)
                    groups = per_cls if per_cls else ([zs] if len(zs) else [])
            self._fg_index.append((groups, per_cls is not None))
        if n_missing:
            logger.warning(
                "%d/%d npz package(s) lack pre-computed '%s'; falling back to "
                "on-the-fly full-resolution foreground scan (cached per "
                "volume per worker). Regenerate npz with a recent make_data "
                "to avoid this.", n_missing, len(self.paths), key)

    def _fg_groups(self, vol_idx: int,
                   lbl: Optional[np.ndarray]) -> List[np.ndarray]:
        """返回该卷的逐类前景索引组（cubic → (N,3) 坐标；z 模式 → (M,) z）。
        npz 无预计算键时由 label 全分辨率计算一次并缓存。"""
        entry = self._fg_index[vol_idx]
        if entry is not None:
            return entry[0]
        cached = self._fg_fallback.get(vol_idx)
        if cached is not None:
            return cached
        groups: List[np.ndarray] = []
        if lbl is not None:
            rng = np.random.default_rng((self.seed, vol_idx))
            for v in self.fg_values:
                mask = lbl == v
                if self.mode == "cubic":
                    g = np.argwhere(mask).astype(np.int32)
                    if len(g) > _FG_FALLBACK_CAP:
                        g = g[rng.choice(len(g), _FG_FALLBACK_CAP,
                                         replace=False)]
                else:
                    g = np.flatnonzero(
                        mask.any(axis=(1, 2))).astype(np.int32)
                if len(g):
                    groups.append(g)
        self._fg_fallback[vol_idx] = groups
        return groups

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    def _rng(self) -> np.random.Generator:
        """逐 worker 训练 RNG（以 PyTorch 逐 worker 基种子创建，可复现且
        跨 worker 不重复；主进程用 ``torch.initial_seed()``）。"""
        info = torch.utils.data.get_worker_info()
        wid = -1 if info is None else info.id
        if self._rng_cache is None or self._rng_wid != wid:
            seed = torch.initial_seed() if info is None else info.seed
            self._rng_cache = np.random.default_rng(seed % (2 ** 63))
            self._rng_wid = wid
        return self._rng_cache

    # ------------------------------------------------------------------
    def _load(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = self._img_cache.get(path)
        lbl = self._lbl_cache.get(path) if self._needs_label else None
        if img is not None and (lbl is not None or not self._needs_label):
            return img, lbl
        # 键存在性先行校验（只解析 zip 目录，不解码数据），报错信息明确。
        with _open_npz(path) as f:
            keys = list(f.files)
        if "image" not in keys:
            raise KeyError(f"npz {path!r} has no 'image' key (keys={keys}).")
        if self._needs_label and "label" not in keys:
            raise KeyError(
                f"npz {path!r} has no 'label' key required by "
                f"label_source='mask' / fg oversampling fallback "
                f"(keys={keys}).")
        # 读取走 seg 的 memmap 零拷贝快路径（页缓存跨 worker 共享；压缩
        # npz 自动回退 zipfile 路径，返回值逐位一致）。
        img = load_npz_image(
            path, self.intensity_min, self.intensity_max, self.normalize,
            self.global_mean, self.global_std)
        lbl = np.asarray(load_npz_label(path)) if self._needs_label else None
        if img.ndim != 3:
            raise ValueError(f"expected 3D volume (D,H,W); got {img.shape} "
                             f"in {path!r}.")
        if lbl is not None and lbl.shape != img.shape:
            raise ValueError(f"image/label shape mismatch in {path!r}: "
                             f"{img.shape} vs {lbl.shape}.")
        self._img_cache.put(path, img)
        if lbl is not None:
            self._lbl_cache.put(path, lbl)
        return img, lbl

    # ------------------------------------------------------------------
    # 中心采样
    # ------------------------------------------------------------------
    def _sample_z(self, rng: np.random.Generator, D_vol: int, vol_idx: int,
                  lbl: Optional[np.ndarray],
                  cov_j: Optional[int]) -> int:
        """z 模式中心 z：val 覆盖 → 等距 bin 中心；训练 fg 命中 → 先均匀选类
        再选该类前景切片；否则均匀采样（口径同 ``SegDataset3D._sample_z``）。"""
        if cov_j is not None:
            return min(int((cov_j + 0.5) * D_vol / self.spv), D_vol - 1)
        if (self.is_train and self.fg_ratio > 0
                and rng.random() < self.fg_ratio):
            groups = self._fg_groups(vol_idx, lbl)
            if groups:
                zs = groups[int(rng.integers(len(groups)))]
                return int(rng.choice(zs))
        return int(rng.integers(0, D_vol))

    def _sample_center(self, rng: np.random.Generator,
                       shape: Tuple[int, ...], vol_idx: int,
                       lbl: Optional[np.ndarray],
                       cov_j: Optional[int]) -> Tuple[int, int, int]:
        """cubic 模式中心 (d,h,w)：val 覆盖 → Halton(2,3,5) 铺满安全中心域；
        训练 fg 命中 → 先均匀选类再选点并夹匯到安全范围；否则安全域内均匀
        采样（口径同 ``SegDataset3DCubic._sample_center``）。"""
        ranges = _safe_center_range(shape, self.patch)
        if cov_j is not None:
            fracs = [_halton(cov_j + 1, b) for b in (2, 3, 5)]
            return tuple(
                lo + min(int(f * (hi - lo)), hi - lo - 1)
                for f, (lo, hi) in zip(fracs, ranges))
        if (self.is_train and self.fg_ratio > 0
                and rng.random() < self.fg_ratio):
            groups = self._fg_groups(vol_idx, lbl)
            if groups:
                coords = groups[int(rng.integers(len(groups)))]
                c = coords[int(rng.integers(len(coords)))]
                return tuple(
                    int(np.clip(int(x), lo, hi - 1))
                    for x, (lo, hi) in zip(c, ranges))
        return tuple(int(rng.integers(lo, hi)) for lo, hi in ranges)

    # ------------------------------------------------------------------
    # patch 抽取（image / label 同一几何）
    # ------------------------------------------------------------------
    def _extract(self, vol: np.ndarray, center: Tuple[int, int, int],
                 is_label: bool) -> np.ndarray:
        """按 patch_mode 抽取严格 (pD,pH,pW) patch（label 用 nearest resize）。"""
        pD, pH, pW = self.patch
        if self.mode == "whole":
            return resize_3d(vol, pD, pH, pW, is_label=is_label)
        if self.mode == "cubic":
            return _extract_cubic_patch(vol, center, self.patch)
        # z_axis / 2_5d：z 轴 edge-padded 抽取 + 面内 resize。
        p = extract_z_patch_padded(vol, center[0], pD)
        return resize_3d(p, pD, pH, pW, is_label=is_label)

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
        # 同卷样本索引连续（idx // spv），顺序遍历时逐 worker LRU 缓存可命中。
        vol_idx = idx // self.spv
        img, lbl = self._load(self.paths[vol_idx])
        if self.is_train:
            rng = self._rng()
        else:
            # 验证：中心由 (seed, idx) 确定性派生，epoch 间可复现。
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        # 网格覆盖：卷内第 j 个样本（j = idx % spv）取确定性网格位置。
        cov_j = (idx % self.spv) if self.val_grid_coverage else None
        if self.mode == "z_axis" or self.mode == "2_5d":
            z = self._sample_z(rng, img.shape[0], vol_idx, lbl, cov_j)
            center = (z, 0, 0)
        elif self.mode == "cubic":
            center = self._sample_center(rng, img.shape, vol_idx, lbl, cov_j)
        else:  # whole
            center = (0, 0, 0)
        img_patch = self._extract(img, center, is_label=False)
        img_t = torch.from_numpy(
            np.ascontiguousarray(img_patch, dtype=np.float32))
        vol_t = torch.tensor(vol_idx, dtype=torch.long)

        if self.gpu_augment:
            # GPU 增强模式：输出未折叠 3D；target 由 trainer 在增强后派生。
            out = {"image": img_t.unsqueeze(0), "vol_idx": vol_t}
            if self.source == "mask":
                lbl_patch = self._extract(lbl, center, is_label=True)
                out["label"] = torch.from_numpy(np.ascontiguousarray(
                    lbl_patch, dtype=np.float32)).unsqueeze(0)
            else:
                out["target"] = torch.from_numpy(
                    np.asarray(self.table_targets[vol_idx]))
            return out

        if not self.fold_2_5d:
            img_t = img_t.unsqueeze(0)             # (1, D, H, W)
        # 2.5D：(D, H, W) —— 深度折进通道。

        if self.source == "mask":
            lbl_patch = self._extract(lbl, center, is_label=True)
            target = self._target_from_mask(lbl_patch)
        else:
            target = torch.from_numpy(np.asarray(self.table_targets[vol_idx]))
        return {"image": img_t, "target": target, "vol_idx": vol_t}


# derive_volume_targets 已上提 taskcore.data.dataset；此处保留旧路径 re-export。


__all__ = [
    "ClsPatchDataset", "load_label_table", "match_table_to_paths",
    "derive_volume_targets",
]
