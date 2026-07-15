"""Image-only（无标注）npz patch 数据集 + dataloader 构造。

SSL 的价值来源是**大规模无标注**语料，故数据通路与分割的"标签耦合"管线解耦：
本数据集只读 npz 的 ``image`` 键（``make_ssl_data`` 产出的 image-only npz，或任何含
``image`` 的既有 npz 皆可），**不读** label / fg_coords / fg_slices。抽取几何与
segtask ``SegDataset3D``（``patch_mode=2_5d/z_axis``）逐字一致：仅沿 z 抽片
（越界 edge-pad），面内 H/W **整片 resize** 到 (pH,pW)（不裁窗），返回
``{"image": (1, eD, pH, pW)}``。

底层 IO / 预处理 / 抽取（``_open_npz`` / ``preprocess_image`` /
``extract_z_patch_padded`` / ``resize_3d``）直接复用 ``segtask_v1.data.dataset``，
不另造轮子。
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from segtask_v1.trainer.dist_utils import (
    get_rank, get_world_size, is_dist_avail_and_initialized)

from segtask_v1.data.dataset import (
    VolumeCache,
    _extract_cubic_patch,
    _open_npz,
    extract_z_patch_padded,
    preprocess_image,
    resize_3d,
)

logger = logging.getLogger(__name__)


def _is_zaxis_mode(patch_mode: str) -> bool:
    """与 segtask specs 同口径：2_5d/z_axis → z 抽片 + H/W 整片 resize；
    cubic → 三轴立方体裁窗；其余（whole 等）SSL 不支持。"""
    pm = str(patch_mode).lower()
    if pm in ("2_5d", "z_axis"):
        return True
    if pm == "cubic":
        return False
    raise ValueError(
        f"SSL dataset supports patch_mode in {{'2_5d','z_axis','cubic'}}; "
        f"got {patch_mode!r}.")


def discover_image_npz(npz_dir: str, npz_suffix: str = ".npz") -> List[str]:
    """递归发现 ``npz_dir`` 下所有以 ``npz_suffix`` 结尾的 npz；按路径排序。"""
    if not npz_dir or not os.path.isdir(npz_dir):
        raise FileNotFoundError(
            f"SSL npz_dir not found or empty: {npz_dir!r}. Point data.npz_dir "
            f"to a directory of image npz packages (image-only or labelled).")
    paths = sorted(
        glob.glob(os.path.join(npz_dir, "**", f"*{npz_suffix}"), recursive=True))
    if not paths:
        raise RuntimeError(
            f"No '*{npz_suffix}' packages found under {npz_dir!r}.")
    return paths


def read_npz_spacing(path: str) -> Optional[Tuple[float, float, float]]:
    """从 npz meta 读有效体素间距 (sz, sy, sx) mm（numpy 轴序 (D,H,W)）。

    make_data 写入的 meta 含 ``orig_spacing``，若 ``spacing_normalized`` 则实际
    体素间距为 ``target_spacing``。无 meta / 字段缺失 / 非法值返 None（调用方
    退化为体素单位）。"""
    try:
        with _open_npz(path) as f:
            if "meta" not in f.files:
                return None
            meta = f["meta"].item()
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    sp = (meta.get("target_spacing") if meta.get("spacing_normalized")
          else meta.get("orig_spacing"))
    if sp is None:
        return None
    try:
        vals = [float(s) for s in sp]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or any((not np.isfinite(v)) or v <= 0.0 for v in vals):
        return None
    return (vals[0], vals[1], vals[2])


def _rand_center(dim: int, p: int,
                 rng: Optional[random.Random] = None) -> int:
    """在 [0, dim) 取一个 center，使大小 p 的 patch 尽量落在体内（dim<=p 时取中点）。"""
    half = p // 2
    if dim <= p:
        return dim // 2
    r = rng if rng is not None else random
    lo = r.randint(0, dim - p)   # patch 起点 ∈ [0, dim-p]，保证不越界
    return lo + half


def _clamp_center(c: int, dim: int, p: int) -> int:
    """把任意体素坐标 clamp 成合法 patch center（patch 尽量落在体内）。"""
    half = p // 2
    if dim <= p:
        return dim // 2
    return min(max(int(c), half), dim - p + half)


class ImageOnlyPatchDataset(Dataset):
    """从含 ``image`` 键的 npz 随机抽 2.5D/z 轴 patch（image-only）。

    每个 epoch 的样本数 = ``len(paths) * samples_per_volume``；``__getitem__`` 内随机
    选体 + 随机 z 中心抽 z-cube（edge-pad），H/W 整片 resize 到 (pH,pW)，与
    segtask ``SegDataset3D``（``patch_mode=2_5d/z_axis``）同口径。

    输出**统一为 3D** ``{"image": (1, pD, pH, pW)} fp32``（含 2.5D）：2.5D 的
    "深度 D 折进通道"改由 trainer 在**数据增强之后、送模型之前**统一折叠（与
    segtask 的 ``squeeze_2_5d`` 送模型前口径一致），从而 3D ``GPUAugmentor`` 也
    能作用于 2.5D 样本；不再在 dataset 层提前折叠。
    """

    def __init__(
        self,
        npz_paths        : Sequence[str],
        patch_size       : Sequence[int],
        intensity_min    : float,
        intensity_max    : float,
        normalize        : str = "minmax",
        samples_per_volume: int = 1,
        global_mean      : float = 0.0,
        global_std       : float = 1.0,
        spatial_dims     : int = 3,
        patch_mode       : str = "cubic",
        aug_oversample_ratio: float = 1.0,
        cache_enabled    : bool = False,
        cache_max_volumes: int = 0):
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError("ImageOnlyPatchDataset got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)  # (pD, pH, pW)
        if len(self.patch) != 3:
            raise ValueError(
                f"patch_size must be 3D (D,H,W) for SSL image-only dataset; "
                f"got {patch_size}.")
        # 仅 z 轴过采样：多抽 round(pD*ratio) 片，供增强后由 trainer 沿 z 中心裁回
        # pD（与 segtask aug_oversample_ratio 口径一致，规避 flip/affine 边界伪影）。
        # 无标注、纯几何余量，不涉及前景。ratio==1.0 时 extract==patch（无余量）。
        self.oversample = float(aug_oversample_ratio)
        if self.oversample < 1.0:
            raise ValueError(
                f"aug_oversample_ratio must be >= 1.0; got {self.oversample}.")
        pD, pH, pW = self.patch
        self.extract_size = (int(round(pD * self.oversample)), pH, pW)
        self.zaxis = _is_zaxis_mode(patch_mode)
        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"spatial_dims must be 2 (2.5D folded) or 3; "
                f"got {self.spatial_dims}.")
        self.fold_2_5d = self.spatial_dims == 2
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.normalize = str(normalize)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)
        self.spv = max(int(samples_per_volume), 1)
        # 逐 worker LRU 缓存（复用 segtask VolumeCache：pickle 到 worker 时清空），
        # 避免 samples_per_volume>1 时每个 patch 都重新解压全卷。
        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        logger.info(
            "ImageOnlyPatchDataset: %d volumes x %d samples = %d, patch=%s, "
            "spatial_dims=%d (%s)",
            len(self.paths), self.spv, len(self), self.patch,
            self.spatial_dims,
            "2.5D (fold deferred to trainer)" if self.fold_2_5d else "3D")

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    def _load_volume(self, path: str) -> np.ndarray:
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        with _open_npz(path) as f:
            if "image" not in f.files:
                raise KeyError(
                    f"npz {path!r} has no 'image' key (keys={list(f.files)}).")
            img_int16 = f["image"]
            img = preprocess_image(
                img_int16, self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std,
                inplace=False)
        self._img_cache.put(path, img)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx % len(self.paths)]
        vol = self._load_volume(path)              # (D, H, W) fp32
        if vol.ndim != 3:
            raise ValueError(
                f"SSL expects 3D image volume (D,H,W); got {vol.shape} in {path!r}.")
        _, pH, pW = self.patch
        eD = self.extract_size[0]
        if self.zaxis:
            # 与 segtask SegDataset3D(2_5d/z_axis) 抽取口径一致：仅沿 z 抽 eD 片
            # （eD>=pD 含过采样余量，越界 edge-pad），面内 H/W 整片 resize 到
            # (pH,pW)（不裁窗，与下游 encoder 看到的空间尺度一致）。
            z = _rand_center(vol.shape[0], eD)
            patch = extract_z_patch_padded(vol, z, eD)            # (eD, H, W)
            patch = resize_3d(patch, eD, pH, pW, is_label=False)  # (eD, pH, pW)
        else:
            # cubic：三轴裁窗（越界 edge-pad）；z 轴含过采样余量。
            center = tuple(_rand_center(d, p)
                           for d, p in zip(vol.shape, self.extract_size))
            patch = _extract_cubic_patch(vol, center, self.extract_size)
        t = torch.from_numpy(patch.astype(np.float32, copy=False))
        # 统一返回 3D (1,eD,pH,pW)；trainer 在增强后沿 z 中心裁回 pD，再（2.5D）
        # 把深度折进通道（见 ssl_trainer._center_crop_z / _fold_batch），与 segtask 一致。
        t = t.unsqueeze(0)                                            # (1,eD,pH,pW)
        return {"image": t}


class LabeledPatchDataset(Dataset):
    """从含 ``image`` + ``label`` 键的 npz 抽**配对** patch，供 §0.5 在线探针。

    与 :class:`ImageOnlyPatchDataset` 共用 IO/预处理与抽取几何（z 抽片 edge-pad +
    H/W 整片 resize，同 segtask ``SegDataset3D``），额外读 ``label`` 并以 *同一 z
    中心* 抽取对齐的 label patch（resize 用最近邻 ``is_label=True`` 保整数取值）。
    label 为原始取值，前景二值化在探针侧按 ``label_values`` 完成。仅用于轻量评测，
    不进 SSL 训练主路径。

    输出布局随 ``spatial_dims`` 切换（与 :class:`ImageOnlyPatchDataset` 折叠口径一致）：

    输出**统一为 3D** ``{"image": (1, pD, pH, pW), "label": (1, pD, pH, pW)}``（含
    2.5D）：2.5D 折叠由消费方（探针 :class:`ssltask.eval.probe.SegProbe`）在送模型
    前完成，探针侧按 ``b (c d) h w`` 口径逐 (类,切片) 二值化。
    """

    def __init__(
        self,
        npz_paths         : Sequence[str],
        patch_size        : Sequence[int],
        intensity_min     : float,
        intensity_max     : float,
        normalize         : str = "minmax",
        samples_per_volume: int = 1,
        global_mean       : float = 0.0,
        global_std        : float = 1.0,
        spatial_dims      : int = 3,
        patch_mode        : str = "cubic",
        cls_label_key     : str = "",
        cache_enabled     : bool = False,
        cache_max_volumes : int = 0,
        deterministic     : bool = False,
        fg_aware          : bool = False,
        seed              : int = 0):
        self.paths = list(npz_paths)
        if not self.paths:
            raise ValueError("LabeledPatchDataset got empty npz_paths.")
        self.patch = tuple(int(s) for s in patch_size)
        if len(self.patch) != 3:
            raise ValueError(
                f"patch_size must be 3D (D,H,W) for the seg probe dataset; "
                f"got {patch_size}.")
        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"spatial_dims must be 2 (2.5D folded) or 3; "
                f"got {self.spatial_dims}.")
        self.fold_2_5d = self.spatial_dims == 2
        self.zaxis = _is_zaxis_mode(patch_mode)
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.normalize = str(normalize)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)
        self.spv = max(int(samples_per_volume), 1)
        self.cls_label_key = str(cls_label_key)
        # deterministic：逐 idx 固定 RNG（跨 epoch/跨进程可重现，供验证集）；
        # fg_aware：优先以 npz 预烘 fg_coords 中前景体素的 z 坐标为 z 中心（clamp
        # 到体内；H/W 整片 resize，无需面内定位），避免随机 patch 大量空前景导致
        # 评测噪声；无 fg_coords/空前景退化为均匀随机 z 中心。
        self.deterministic = bool(deterministic)
        self.fg_aware = bool(fg_aware)
        self.seed = int(seed)
        self._fg_cache: Dict[str, Optional[np.ndarray]] = {}
        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._cls_cache: Dict[str, Optional[np.ndarray]] = {}
        # 逐卷体素间距 (sz,sy,sx) mm，供探针算 spacing-aware HD95；无 meta 退 1mm。
        self._spacing_cache: Dict[str, Tuple[float, float, float]] = {}
        logger.info(
            "LabeledPatchDataset (probe): %d volumes x %d samples = %d, patch=%s, "
            "spatial_dims=%d (%s)",
            len(self.paths), self.spv, len(self), self.patch,
            self.spatial_dims,
            "2.5D (fold deferred to probe)" if self.fold_2_5d else "3D")

    def __len__(self) -> int:
        return len(self.paths) * self.spv

    def _load(self, path: str):
        img = self._img_cache.get(path)
        lbl = self._lbl_cache.get(path)
        if img is not None and lbl is not None and path in self._cls_cache:
            return img, lbl, self._cls_cache[path]
        with _open_npz(path) as f:
            if "image" not in f.files or "label" not in f.files:
                raise KeyError(
                    f"probe npz {path!r} must have both 'image' and 'label' "
                    f"keys (keys={list(f.files)}).")
            img = preprocess_image(
                f["image"], self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std,
                inplace=False)
            lbl = np.asarray(f["label"])
            cls_label = None
            if self.cls_label_key:
                if self.cls_label_key not in f.files:
                    raise KeyError(
                        f"probe npz {path!r} has no {self.cls_label_key!r} "
                        f"key (keys={list(f.files)}).")
                cls_label = np.asarray(f[self.cls_label_key])
        if img.shape != lbl.shape:
            raise ValueError(
                f"image/label shape mismatch in {path!r}: "
                f"{img.shape} vs {lbl.shape}.")
        self._img_cache.put(path, img)
        self._lbl_cache.put(path, lbl)
        if self._img_cache.get(path) is not None:
            self._cls_cache[path] = cls_label
        return img, lbl, cls_label

    def _load_fg_coords(self, path: str) -> Optional[np.ndarray]:
        if path in self._fg_cache:
            return self._fg_cache[path]
        coords: Optional[np.ndarray] = None
        try:
            with _open_npz(path) as f:
                if "fg_coords" in f.files:
                    arr = np.asarray(f["fg_coords"])
                    if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
                        coords = arr.astype(np.int64, copy=False)
        except Exception:
            coords = None
        self._fg_cache[path] = coords
        return coords

    def _rng(self, idx: int) -> Optional[random.Random]:
        if not self.deterministic:
            return None
        return random.Random((self.seed * 1000003 + idx) * 2654435761 % (2**63))

    def _pick_z(self, idx: int, path: str, D: int) -> int:
        pD = self.patch[0]
        rng = self._rng(idx)
        if self.fg_aware:
            coords = self._load_fg_coords(path)
            if coords is not None:
                r = rng if rng is not None else random
                z = int(coords[r.randrange(len(coords))][0])
                return _clamp_center(z, D, pD)
        return _rand_center(D, pD, rng)

    def _pick_center(self, idx: int, path: str, shape) -> tuple:
        rng = self._rng(idx)
        if self.fg_aware:
            coords = self._load_fg_coords(path)
            if coords is not None:
                r = rng if rng is not None else random
                c = coords[r.randrange(len(coords))]
                return tuple(_clamp_center(int(ci), d, p)
                             for ci, d, p in zip(c, shape, self.patch))
        return tuple(_rand_center(d, p, rng)
                     for d, p in zip(shape, self.patch))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx % len(self.paths)]
        img, lbl, cls_label = self._load(path)            # (D,H,W) fp32 / raw
        if img.ndim != 3:
            raise ValueError(
                f"probe expects 3D volume (D,H,W); got {img.shape} in {path!r}.")
        pD, pH, pW = self.patch
        if self.zaxis:
            # 与 SegDataset3D(2_5d/z_axis) 一致：z 抽 pD 片（edge-pad），H/W 整片
            # resize；label 用最近邻（is_label=True）保持整数取值。
            z = self._pick_z(idx, path, img.shape[0])
            img_patch = resize_3d(
                extract_z_patch_padded(img, z, pD), pD, pH, pW, is_label=False)
            lbl_patch = resize_3d(
                extract_z_patch_padded(lbl, z, pD), pD, pH, pW, is_label=True)
        else:
            center = self._pick_center(idx, path, img.shape)
            img_patch = _extract_cubic_patch(img, center, self.patch)
            lbl_patch = _extract_cubic_patch(lbl, center, self.patch)
        img_t = torch.from_numpy(img_patch.astype(np.float32, copy=False))
        lbl_t = torch.from_numpy(lbl_patch.astype(np.float32, copy=False))
        # 统一返回 3D (1,pD,pH,pW)；2.5D 折叠由探针在送模型前完成。
        img_t = img_t.unsqueeze(0)                                 # (1,pD,pH,pW)
        lbl_t = lbl_t.unsqueeze(0)
        spacing = self._spacing_cache.get(path)
        if spacing is None:
            spacing = read_npz_spacing(path) or (1.0, 1.0, 1.0)
            self._spacing_cache[path] = spacing
        # 2.5D：(pD,pH,pW) = (C=D, H, W)，深度折进通道。
        out = {"image": img_t, "label": lbl_t,
               "spacing": torch.tensor(spacing, dtype=torch.float64)}
        if cls_label is not None:
            out["cls_label"] = torch.from_numpy(np.asarray(cls_label))
        return out


def build_ssl_dataloader(cfg) -> DataLoader:
    """按 ``cfg.data`` 构造 image-only 训练 dataloader（无 val：见 §0.5 在线探针）。

    依据 ``cfg.model.spatial_dims`` 自动选 3D / 2.5D 折叠输出。2.5D（折叠 D 进通道）
    仅支持单 FOV：要求 ``in_channels == patch_size[0]``（即 ``multi_res_scales==[1.0]``）；
    多 FOV 2.5D 需要数据增强级的多分辨率裁剪（segtask ``split_views_native_d``），
    暂不在 image-only SSL 通路内支持。
    """
    dc = cfg.data
    spatial_dims = int(cfg.model.spatial_dims)
    if spatial_dims == 2:
        D = int(dc.patch_size[0])
        in_ch = int(cfg.model.in_channels)
        if in_ch != D:
            raise ValueError(
                f"2.5D SSL pretraining supports single-FOV only: model.in_channels "
                f"({in_ch}) must equal patch_size[0] (D={D}), i.e. "
                f"data.multi_res_scales==[1.0]. Multi-FOV 2.5D needs augmentation-"
                f"level multi-resolution cropping, not available in image-only SSL.")
    paths = discover_image_npz(dc.npz_dir, dc.npz_suffix)
    ds = ImageOnlyPatchDataset(
        npz_paths         = paths,
        patch_size        = dc.patch_size,
        intensity_min     = dc.intensity_min,
        intensity_max     = dc.intensity_max,
        normalize         = dc.normalize,
        samples_per_volume= dc.samples_per_volume,
        global_mean       = dc.global_mean,
        global_std        = dc.global_std,
        spatial_dims      = spatial_dims,
        patch_mode        = dc.patch_mode,
        aug_oversample_ratio = dc.aug_oversample_ratio,
        cache_enabled     = dc.cache_mode == "memory",
        cache_max_volumes = dc.cache_max_volumes)
    num_workers = int(dc.num_workers)
    kwargs: Dict[str, object] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(dc.persistent_workers)
        kwargs["prefetch_factor"] = int(dc.prefetch_factor)
    # DDP：各 rank 分片采样（trainer 每 epoch ``set_epoch`` 重洗）。
    sampler = None
    shuffle = True
    if is_dist_avail_and_initialized() and get_world_size() > 1:
        sampler = DistributedSampler(
            ds, num_replicas=get_world_size(), rank=get_rank(),
            shuffle=True, drop_last=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size      = int(dc.batch_size),
        shuffle         = shuffle,
        sampler         = sampler,
        num_workers     = num_workers,
        pin_memory      = bool(dc.pin_memory),
        drop_last       = True,
        **kwargs)


__all__ = [
    "ImageOnlyPatchDataset", "LabeledPatchDataset",
    "build_ssl_dataloader", "discover_image_npz", "read_npz_spacing",
]
