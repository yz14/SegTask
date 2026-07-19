"""Dataset classes and patch extractors for gentask.data.dataset."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from taskcore.data.dataset import (
    BBox, VolumeCache, _open_npz, compute_bbox_from_volume,
    compute_region_weight_map,
    load_nifti, load_nifti_cropped, load_nifti_with_spacing,
    load_npz_fg_coords, load_npz_fg_slices, load_npz_image,
    load_npz_image_raw,
    load_npz_label, load_npz_label_for_split, load_npz_region_weight,
    load_region_weight_volume, npz_has_rw, preprocess_image, resize_3d,
)

from .io import load_npz_cond

logger = logging.getLogger(__name__)

# 验证集确定性采样的固定基种子（与样本序号组合派生逐样本 RNG）。
_VAL_SAMPLING_SEED = 0x5EED_2024


def _halton(i: int, base: int) -> float:
    """Halton 低差异序列第 i 项（i>=1），返回 [0,1) 内均匀覆盖的确定性分数；
    不同素数 base 给出准独立维度（val_grid_coverage 的 3D 中心铺点用 2/3/5）。"""
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class VolumeNpzDatasetBase(Dataset):
    """共用 npz I/O + 缓存基类。子类负责索引构建、采样与 __getitem__。

    抽出三类（z 轴 / cubic / whole）重复的 image/label/region-weight 读取、LRU 缓存、
    强度归一化与 region_weights 形参。子类通过 super().__init__(...) 注入公用配置后，
    只补充自己的 patch 抽取/采样/索引逻辑。
    """

    def __init__(
        self,
        image_paths         : List[str],
        label_paths         : List[str],
        label_values        : List[int],
        npz_paths           : List[str],
        patch_size          : Tuple[int, int, int],
        aug_oversample_ratio: float,
        intensity_min       : float,
        intensity_max       : float,
        normalize           : str,
        global_mean         : float,
        global_std          : float,
        samples_per_volume  : int,
        is_train            : bool,
        cache_enabled       : bool,
        cache_max_volumes   : int,
        cache_int16         : bool,
        region_weights      : Optional[List[float]],
        cond_normalize      : str,
        cond_intensity_min  : float,
        cond_intensity_max  : float,
        cond_global_mean    : float,
        cond_global_std     : float,
        val_grid_coverage   : bool = False):
        super().__init__()
        assert len(image_paths) == len(label_paths)
        assert npz_paths is not None and len(npz_paths) == len(image_paths), (
            f"{type(self).__name__} requires npz_paths (training is npz-only).")
        assert aug_oversample_ratio >= 1.0, (
            f"aug_oversample_ratio must be >= 1.0, got {aug_oversample_ratio}")

        self.image_paths        = image_paths
        self.label_paths        = label_paths
        self.label_values       = label_values
        self.patch_size         = tuple(patch_size)
        self.oversample         = float(aug_oversample_ratio)
        self.intensity_min      = intensity_min
        self.intensity_max      = intensity_max
        self.normalize          = normalize
        self.global_mean        = global_mean
        self.global_std         = global_std
        self.samples_per_volume = samples_per_volume
        self.is_train           = is_train
        self.region_weights     = region_weights
        self.cond_normalize     = cond_normalize
        self.cond_intensity_min = cond_intensity_min
        self.cond_intensity_max = cond_intensity_max
        self.cond_global_mean   = cond_global_mean
        self.cond_global_std    = cond_global_std
        self.val_grid_coverage  = bool(val_grid_coverage)

        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._rw_cache  = VolumeCache(cache_enabled, cache_max_volumes)
        self._cond_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._cache_int16 = bool(cache_int16)

        # NPZ 预计算包（make_data 产出）提供 bbox / fg 索引 / 可选 rw。
        self._npz_paths       : List[str]       = list(npz_paths)
        self._npz_has_rw_cache: Dict[int, bool] = {}

        # 逐 worker 采样 RNG（惰性创建，见 _rng）。
        self._rng_cache: Optional[np.random.Generator] = None
        self._rng_wid  : Optional[int] = None
        # 当前 __getitem__ 的样本序号（供验证态确定性采样派生 RNG）。
        self._sample_idx: int = 0

    def _rng(self) -> np.random.Generator:
        """逐 worker 采样 RNG。

        DataLoader fork 后 numpy 全局 RNG 状态在各 worker 间复制，直接用
        ``np.random.*`` 会导致跨 worker 重复采样。这里以 PyTorch 逐 worker
        基种子（主进程则用 ``torch.initial_seed()``）惰性创建独立的
        ``np.random.Generator``。
        """
        info = torch.utils.data.get_worker_info()
        wid = -1 if info is None else info.id
        if self._rng_cache is None or self._rng_wid != wid:
            seed = torch.initial_seed() if info is None else info.seed
            self._rng_cache = np.random.default_rng(seed % (2 ** 63))
            self._rng_wid = wid
        return self._rng_cache

    def _sample_rng(self) -> np.random.Generator:
        """patch 采样 RNG。

        训练用逐 worker 流式 RNG（每 epoch 不同，保采样多样性）；验证用
        当前样本序号派生的确定性 RNG，使每个 epoch 评估同一组 patch，
        save_best / early-stopping / plateau 不被采样噪声驱动。"""
        if self.is_train:
            return self._rng()
        return np.random.default_rng((_VAL_SAMPLING_SEED, self._sample_idx))

    def _val_coverage_pos(self) -> Optional[Tuple[int, int]]:
        """val 确定性网格覆盖（val_grid_coverage=True）：返回当前样本在卷内的
        序号 j 与每卷样本数 S；未启用或训练态返回 None（回退随机位置）。"""
        if self.is_train or not self.val_grid_coverage:
            return None
        j = self._sample_idx // len(self.image_paths)
        return j, max(int(self.samples_per_volume), 1)

    # ------------------------------------------------------------------
    # 共用 npz 读取（子类可直接复用，缓存按 path 共享于同一 worker）
    # ------------------------------------------------------------------
    def _load_image(self, vol_idx: int) -> np.ndarray:
        """加载+预处理 image（npz，带缓存）。

        cache_int16：缓存原始 int16 卷（RAM 减半），每次取用重跑
        preprocess_image（产出与 fp32 缓存逐位一致）。"""
        path = self._npz_paths[vol_idx]
        if self._cache_int16:
            raw = self._img_cache.get(path)
            if raw is None:
                raw = load_npz_image_raw(path)
                self._img_cache.put(path, raw)
            return preprocess_image(
                raw, self.intensity_min, self.intensity_max,
                self.normalize, self.global_mean, self.global_std,
                inplace=False)
        cached = self._img_cache.get(path)
        if cached is not None:
            return cached
        img = load_npz_image(
            path, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)
        self._img_cache.put(path, img)
        return img

    def _load_label(self, vol_idx: int) -> np.ndarray:
        """加载原始 int16 label（npz，带缓存）。"""
        path   = self._npz_paths[vol_idx]
        cached = self._lbl_cache.get(path)
        if cached is not None:
            return cached
        lbl = load_npz_label(path)
        self._lbl_cache.put(path, lbl)
        return lbl

    def _has_region_weight_file(self, vol_idx: int) -> bool:
        cached = self._npz_has_rw_cache.get(vol_idx)
        if cached is None:
            cached = npz_has_rw(self._npz_paths[vol_idx])
            self._npz_has_rw_cache[vol_idx] = cached
        return cached

    def _load_region_weight(self, vol_idx: int) -> Optional[np.ndarray]:
        """加载区域权重（npz；+1 偏移由 make_data 加过）；无 rw 返 None。"""
        path   = self._npz_paths[vol_idx]
        cached = self._rw_cache.get(path)
        if cached is not None:
            return cached
        rw = load_npz_region_weight(path)
        if rw is not None:
            self._rw_cache.put(path, rw)
        return rw

    def _load_cond(self, vol_idx: int) -> Optional[np.ndarray]:
        """加载条件体（npz）；无 cond 返 None。"""
        path   = self._npz_paths[vol_idx]
        cached = self._cond_cache.get(path)
        if cached is not None:
            return cached
        cond = load_npz_cond(path)
        if cond is None:
            return None
        if cond.ndim == 3:
            cond = cond[np.newaxis]
        if cond.ndim != 4:
            raise ValueError(
                f"Expected cond volume to have shape (C,D,H,W) or (D,H,W); got {cond.shape}")
        normed = np.empty_like(cond, dtype=np.float32)
        for i, ch in enumerate(cond):
            normed[i] = preprocess_image(
                ch, self.cond_intensity_min, self.cond_intensity_max,
                self.cond_normalize, self.cond_global_mean, self.cond_global_std)
        self._cond_cache.put(path, normed)
        return normed

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_volume


# ---------------------------------------------------------------------------
# 3D Volume Dataset (z-axis sliding window)
# ---------------------------------------------------------------------------
class Volume3D(VolumeNpzDatasetBase):
    """3D z 轴滑窗 dataset。z 轴滑动抖中心 z，折取 round(eD*s) 切片，
    仅 z 过采样；H/W 全分辨率 resize 到 patch_size。与 predictor.sliding.sliding_window_z 一致。

    多分辨率 multi_res_scales=[1.0] 为单分辨率；s>1 强制 edge-replicate 以保留物理 z-FOV。
    输出 shape：image/label/weight_map = (C_res, eD, pH, pW)。
    """

    def __init__(
        self,
        image_paths         : List[str],
        label_paths         : List[str],
        label_values        : List[int],
        patch_size          : Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio: float = 1.0,
        multi_res_scales    : Optional[List[float]] = None,
        intensity_min       : float = -1024.0,
        intensity_max       : float = 3071.0,
        normalize           : str = "minmax",
        global_mean         : float = 0.0,
        global_std          : float = 1.0,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume  : int = 8,
        is_train            : bool = True,
        cache_enabled       : bool = True,
        cache_max_volumes   : int = 0,
        cache_int16         : bool = False,
        region_weights      : Optional[List[float]] = None,
        cond_normalize      : str = "minmax",
        cond_intensity_min  : float = -1024.0,
        cond_intensity_max  : float = 1024.0,
        cond_global_mean    : float = 0.0,
        cond_global_std     : float = 1.0,
        z_boundary_mode     : str = "stretch",
        npz_paths           : Optional[List[str]] = None,
        val_grid_coverage   : bool = False):
        super().__init__(
            image_paths          = image_paths,
            label_paths          = label_paths,
            label_values         = label_values,
            npz_paths            = npz_paths,
            patch_size           = patch_size,
            aug_oversample_ratio = aug_oversample_ratio,
            intensity_min        = intensity_min,
            intensity_max        = intensity_max,
            normalize            = normalize,
            global_mean          = global_mean,
            global_std           = global_std,
            samples_per_volume   = samples_per_volume,
            is_train             = is_train,
            cache_enabled        = cache_enabled,
            cache_max_volumes    = cache_max_volumes,
            cache_int16          = cache_int16,
            region_weights       = region_weights,
            cond_normalize       = cond_normalize,
            cond_intensity_min   = cond_intensity_min,
            cond_intensity_max   = cond_intensity_max,
            cond_global_mean     = cond_global_mean,
            cond_global_std      = cond_global_std,
            val_grid_coverage    = val_grid_coverage)
        if z_boundary_mode not in ("stretch", "edge_pad"):
            raise ValueError(
                f"z_boundary_mode must be 'stretch' or 'edge_pad', "
                f"got {z_boundary_mode!r}")

        # 仅 z 轴过采样（供增强后中心裁减）；H/W 一次 resize 到 patch_size
        pD, pH, pW = self.patch_size
        self.extract_size = (int(round(pD * self.oversample)), pH, pW)
        # 多分辨率 z FOV：multi_res_scales=[1.0] 单分辨率；len>1 时 view 0 必为 1.0
        self.multi_res_scales = list(multi_res_scales) if multi_res_scales else [1.0]
        assert all(s >= 1.0 for s in self.multi_res_scales), (
            f"All multi_res_scales must be >= 1.0, got {self.multi_res_scales}")
        assert self.multi_res_scales[0] == 1.0, (
            "multi_res_scales[0] must be 1.0 (canonical view); got "
            f"{self.multi_res_scales}")
        self._max_scale = float(max(self.multi_res_scales))
        self.fg_ratio = foreground_oversample_ratio

        # 边界处理：max_scale==1.0 可选 stretch 或 edge_pad；
        # max_scale>1.0 必须 edge_pad，否则跨 scale 物理 z-FOV 不一致。
        self.z_boundary_mode = z_boundary_mode
        if self._max_scale > 1.0 and self.z_boundary_mode != "edge_pad":
            raise ValueError(
                f"multi-res (max_scale={self._max_scale}) requires "
                f"z_boundary_mode='edge_pad'; got {self.z_boundary_mode!r}.")

        # 逐卷前景索引（从 npz 读）驱动 _sample_z 过采样。
        self._vol_fg_slices : List[np.ndarray] = []
        self._vol_all_slices: List[int] = []
        self._build_index()

    def _build_index(self) -> None:
        """NPZ 模式 fg-slice 索引：make_data 预计算，此处仅读取。"""
        logger.info(
            "Loading pre-computed fg indices from %d npz packages...",
            len(self._npz_paths))
        total_fg     = 0
        total_slices = 0
        for path in self._npz_paths:
            f  = _open_npz(path)
            fg = np.asarray(f["fg_slices"], dtype=np.int32)
            D  = int(f["image"].shape[0])
            self._vol_fg_slices.append(fg)
            self._vol_all_slices.append(D)
            total_fg += len(fg)
            total_slices += D
        logger.info(
            "NPZ index built: %d volumes, %d/%d foreground slices",
            len(self._npz_paths), total_fg, total_slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """总是发单 max-FOV z-cube (1, eD_max, eH, eW)，eD_max=round(eD*max_scale)。多分辨率交由
        trainer 中心裁拆视图；2.5D / z_axis 在数据集侧抽取逻辑完全一致。单分辨率时
        max_scale=1.0，eD_max==eD。"""
        self._sample_idx = idx
        vol_idx  = idx % len(self.image_paths)
        img, lbl = self._load_image(vol_idx), self._load_label(vol_idx)
        D_vol    = img.shape[0]
        # extract_size = (eD,pH,pW)；仅 z 过采样，oversample=1 时 eD==pD，trainer 跳裁。
        eD, eH, eW = self.extract_size

        z = self._sample_z(vol_idx, D_vol)

        # 样本区域权重文件 > 静态 region_weights 映射。
        rw_vol = (self._load_region_weight(vol_idx)
                  if self._has_region_weight_file(vol_idx) else None)
        cond_vol = self._load_cond(vol_idx)
        return self._getitem_max_fov(img, lbl, rw_vol, cond_vol, z, eD, eH, eW)

    def _getitem_max_fov(
        self,
        img    : np.ndarray,
        lbl    : np.ndarray,
        rw_vol : Optional[np.ndarray],
        cond_vol: Optional[np.ndarray],
        z      : int,
        eD     : int,
        eH     : int,
        eW     : int) -> Dict[str, torch.Tensor]:
        """抽单 max-FOV z-cube (1, eD_max, eH, eW)：edge-padded z 保证严格 eD_max，面内 resize 到 (eH,eW)。
        eD_max==eD 时（单分辨率）表现为普通 patch；s>1 时为多 FOV 超尺寸 cube，trainer 拆视图。"""
        eD_max = int(round(eD * self._max_scale))
        # edge-padded：跨 z 边界保物理 FOV，不走 stretch resize。
        img_s, lbl_s = self._extract_z_patch_padded(img, lbl, z, eD_max)
        rw_s = (self._extract_z_single(rw_vol, z, eD_max, use_padded=True)
                if rw_vol is not None else None)
        cond_s = (_channelwise_3d(
            cond_vol,
            lambda ch: extract_z_patch_padded(ch, z, eD_max))
            if cond_vol is not None else None)

        # 面内 resize 到 (eH,eW)；D 轴保持 eD_max（不重采样）。
        img_s = resize_3d(img_s, eD_max, eH, eW, is_label=False)
        lbl_s = resize_3d(lbl_s, eD_max, eH, eW, is_label=True)
        if cond_s is not None:
            cond_s = resize_3d(cond_s, eD_max, eH, eW, is_label=False)
        result = {
            # 领头 "1" = 压叠 C_res 轴，与旧输出布局一致。
            "image": torch.from_numpy(img_s[None].astype(np.float32, copy=False)),
            "label": torch.from_numpy(np.ascontiguousarray(lbl_s[None]))}
        if rw_s is not None:
            # rw 是分级序权重（离散值），必须 nearest 避免产生伪连续值；resize_3d(is_label=True) = order=0。
            rw_s = resize_3d(rw_s, eD_max, eH, eW, is_label=True)
            result["weight_map"] = torch.from_numpy(
                rw_s[None].astype(np.float32, copy=False))
        elif self.region_weights:
            rw_s = compute_region_weight_map(
                lbl_s, self.label_values, self.region_weights)
            result["weight_map"] = torch.from_numpy(
                rw_s.astype(np.float32, copy=False))
        if cond_s is not None:
            result["cond"] = torch.from_numpy(
                np.ascontiguousarray(cond_s.astype(np.float32, copy=False)))
        return result

    def _sample_z(self, vol_idx: int, D_vol: int) -> int:
        """采样中心 z：训练以 fg_ratio 概率从前景切片采样，否则均匀采样；
        验证用逐样本确定性 RNG 均匀采样（见 _sample_rng）。"""
        cov = self._val_coverage_pos()
        if cov is not None:
            # 网格覆盖：卷内第 j 个样本取 z 轴等距位置（bin 中心）。
            j, S = cov
            return min(int((j + 0.5) * D_vol / S), D_vol - 1)
        fg_slices = self._vol_fg_slices[vol_idx]
        rng = self._sample_rng()
        if (self.is_train and self.fg_ratio > 0
            and len(fg_slices) > 0
            and rng.random() < self.fg_ratio):
            return int(rng.choice(fg_slices))
        return int(rng.integers(0, D_vol))

    def _extract_z_patch_padded(
        self, img: np.ndarray, lbl: np.ndarray, z_center: int, D_patch: int
        ) -> Tuple[np.ndarray, np.ndarray]:
        """image+label 同步 edge-padded 抽取（语义见模块级 extract_z_patch_padded）。"""
        return (
            extract_z_patch_padded(img, z_center, D_patch),
            extract_z_patch_padded(lbl, z_center, D_patch))

    def _extract_z_single(
        self, vol: np.ndarray, z_center: int, D_patch: int, use_padded: bool
        ) -> np.ndarray:
        """单卷 z-patch 抽取（与 image+label 对齐），供区域权重体积复用。"""
        if use_padded:
            return extract_z_patch_padded(vol, z_center, D_patch)
        D_vol   = vol.shape[0]
        half    = D_patch // 2
        d_start = max(0, z_center - half)
        d_end   = min(D_vol, d_start + D_patch)
        return vol[d_start:d_end].copy()


# ---------------------------------------------------------------------------
# Module-level z-axis patch extractor (shared with Predictor)
# ---------------------------------------------------------------------------
def extract_z_patch_padded(
    vol: np.ndarray, z_center: int, D_patch: int) -> np.ndarray:
    """以 z_center 为中心从 vol 抽 D_patch 切片；越界部分 mode='edge' 复制边界。
    保留物理 z-FOV（输出始终 D_patch）；H/W 不动；label 下复制边界近邻值安全。"""
    D_vol      = vol.shape[0]
    half       = D_patch // 2
    lo         = z_center - half
    hi         = lo + D_patch
    src_lo     = max(lo, 0)
    src_hi     = min(hi, D_vol)
    pad_before = max(-lo, 0)
    pad_after  = max(hi - D_vol, 0)

    patch = vol[src_lo:src_hi]
    if pad_before > 0 or pad_after > 0:
        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (vol.ndim - 1)
        patch = np.pad(patch, pad_width, mode="edge")
    return patch.copy()


def _channelwise_3d(
    vol: Optional[np.ndarray], fn) -> Optional[np.ndarray]:
    """对 (C,D,H,W) 逐通道应用 3D 体操作；3D 输入会先补通道维。"""
    if vol is None:
        return None
    if vol.ndim == 3:
        vol = vol[np.newaxis]
    if vol.ndim != 4:
        raise ValueError(f"Expected 3D or 4D volume, got {vol.ndim}D")
    out = [fn(ch) for ch in vol]
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# 3D Cubic Patch Dataset
# ---------------------------------------------------------------------------
def _extract_cubic_patch(
    vol: np.ndarray, center: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
    """以 (d,h,w) 为中心抽出严格 (pD,pH,pW) cube；越界部分 mode='edge' 复制边界。"""
    D, H, W    = vol.shape
    pD, pH, pW = size
    cd, ch, cw = center

    # 逐轴计算起止与填充
    starts, ends, pad_before, pad_after = [], [], [], []
    for c, p, s in [(cd, pD, D), (ch, pH, H), (cw, pW, W)]:
        half = p // 2
        lo = c - half
        hi = lo + p
        # 夹匯到边界并计算 padding
        src_lo = max(lo, 0)
        src_hi = min(hi, s)
        starts.append(src_lo)
        ends.append(src_hi)
        pad_before.append(max(-lo, 0))
        pad_after.append(max(hi - s, 0))

    patch = vol[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    # 越界时 mode='edge' 填充以保证准确 size；避免下游 resize_3d 各向异性拉伸，
    # 也与 predictor.sliding.sliding_window_cubic 默认 pad 一致。
    if any(pb > 0 or pa > 0 for pb, pa in zip(pad_before, pad_after)):
        patch = np.pad(
            patch,
            list(zip(pad_before, pad_after)),
            mode="edge")

    return patch


class Volume3DCubic(VolumeNpzDatasetBase):
    """3D cubic patch dataset：以 (d,h,w) 为中心抽 3D cube。支持增强过采样与
    多分辨率（同中心多 scale resize 后拼通道）。输出与 Volume3D 一致：(C_res, eD, eH, eW)，
    label 以原始整数传到损失处二值化。"""

    def __init__(
        self,
        image_paths                : List[str],
        label_paths                : List[str],
        label_values               : List[int],
        patch_size                 : Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio       : float = 1.0,
        multi_res_scales           : Optional[List[float]] = None,
        intensity_min              : float = -1024.0,
        intensity_max              : float = 3071.0,
        normalize                  : str = "minmax",
        global_mean                : float = 0.0,
        global_std                 : float = 1.0,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume         : int = 8,
        is_train                   : bool = True,
        cache_enabled              : bool = True,
        cache_max_volumes          : int = 0,
        cache_int16                : bool = False,
        region_weights             : Optional[List[float]] = None,
        cond_normalize             : str = "minmax",
        cond_intensity_min         : float = -1024.0,
        cond_intensity_max         : float = 1024.0,
        cond_global_mean           : float = 0.0,
        cond_global_std            : float = 1.0,
        npz_paths                  : Optional[List[str]] = None,
        val_grid_coverage          : bool = False):
        super().__init__(
            image_paths          = image_paths,
            label_paths          = label_paths,
            label_values         = label_values,
            npz_paths            = npz_paths,
            patch_size           = patch_size,
            aug_oversample_ratio = aug_oversample_ratio,
            intensity_min        = intensity_min,
            intensity_max        = intensity_max,
            normalize            = normalize,
            global_mean          = global_mean,
            global_std           = global_std,
            samples_per_volume   = samples_per_volume,
            is_train             = is_train,
            cache_enabled        = cache_enabled,
            cache_max_volumes    = cache_max_volumes,
            cache_int16          = cache_int16,
            region_weights       = region_weights,
            cond_normalize       = cond_normalize,
            cond_intensity_min   = cond_intensity_min,
            cond_intensity_max   = cond_intensity_max,
            cond_global_mean     = cond_global_mean,
            cond_global_std      = cond_global_std,
            val_grid_coverage    = val_grid_coverage)

        self.extract_size = tuple(  # 有效抽取尺寸（增强过采样余量）
            int(round(p * self.oversample)) for p in self.patch_size)
        self.multi_res_scales = list(multi_res_scales) if multi_res_scales else [1.0]
        assert all(s >= 1.0 for s in self.multi_res_scales), (
            f"All multi_res_scales must be >= 1.0, got {self.multi_res_scales}")
        assert self.multi_res_scales[0] == 1.0, (
            "multi_res_scales[0] must be 1.0 (canonical view); got "
            f"{self.multi_res_scales}")
        # 最大 scale 决定 cube 抽取尺寸
        self._max_scale = float(max(self.multi_res_scales))
        self.fg_ratio   = foreground_oversample_ratio

        # 逐卷 3D fg 坐标索引（make_data 预抽：seed=42, cap=50000）驱动 _sample_center 过采样。
        self._vol_shapes   : List[Tuple[int, int, int]] = []
        self._vol_fg_coords: List[np.ndarray]           = []
        self._build_index()

    def _build_index(self) -> None:
        """NPZ 模式 fg-coord 索引：make_data 预计算，此处仅读取。"""
        logger.info(
            "Loading pre-computed fg coords from %d npz packages...",
            len(self._npz_paths))
        total_fg = 0
        for path in self._npz_paths:
            f      = _open_npz(path)
            coords = np.asarray(f["fg_coords"], dtype=np.int32)
            shape  = tuple(int(s) for s in f["image"].shape)
            self._vol_shapes.append(shape)
            self._vol_fg_coords.append(coords)
            total_fg += len(coords)
        logger.info(
            "NPZ cubic index: %d volumes, %d fg voxels sampled",
            len(self._npz_paths), total_fg)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """总是发单 max-FOV cube (1, eD_max, eH_max, eW_max)，size = round(extract_size*max_scale)。
        多分辨率交由 trainer 中心裁拆视图。单分辨率时 max_scale=1.0，cube == extract_size。"""
        self._sample_idx = idx
        vol_idx    = idx % len(self.image_paths)
        img, lbl   = self._load_image(vol_idx), self._load_label(vol_idx)
        D, H, W    = img.shape
        center     = self._sample_center(vol_idx, D, H, W)
        eD, eH, eW = self.extract_size

        rw_vol = (self._load_region_weight(vol_idx)
                  if self._has_region_weight_file(vol_idx) else None)
        cond_vol = self._load_cond(vol_idx)
        return self._getitem_max_fov(center, img, lbl, rw_vol, cond_vol, eD, eH, eW)

    def _getitem_max_fov(
        self,
        center: Tuple[int, int, int],
        img   : np.ndarray,
        lbl   : np.ndarray,
        rw_vol: Optional[np.ndarray],
        cond_vol: Optional[np.ndarray],
        eD    : int,
        eH    : int,
        eW    : int) -> Dict[str, torch.Tensor]:
        """抽单 max-FOV cube；越轴体积过小时由 _extract_cubic_patch edge-pad 保证严格尺寸。"""
        eD_max   = int(round(eD * self._max_scale))
        eH_max   = int(round(eH * self._max_scale))
        eW_max   = int(round(eW * self._max_scale))
        size_max = (eD_max, eH_max, eW_max)

        img_s = _extract_cubic_patch(img, center, size_max)
        lbl_s = _extract_cubic_patch(lbl, center, size_max)
        rw_s  = (_extract_cubic_patch(rw_vol, center, size_max)
                if rw_vol is not None else None)
        cond_s = (_channelwise_3d(
            cond_vol, lambda ch: _extract_cubic_patch(ch, center, size_max))
            if cond_vol is not None else None)

        result = {
            # 领头 "1" = 压叠 C_res 轴；trainer 逐视图裁+resize。
            "image": torch.from_numpy(img_s[None].astype(np.float32, copy=False)),
            "label": torch.from_numpy(np.ascontiguousarray(lbl_s[None]))}
        if rw_s is not None:
            # rw 离散权重，_extract_cubic_patch 已是按位裁剪 + edge-pad，无重采样，无需 nearest。
            result["weight_map"] = torch.from_numpy(
                rw_s[None].astype(np.float32, copy=False))
        elif self.region_weights:
            rw_s = compute_region_weight_map(
                lbl_s, self.label_values, self.region_weights)
            result["weight_map"] = torch.from_numpy(
                rw_s.astype(np.float32, copy=False))
        if cond_s is not None:
            cond_s = resize_3d(cond_s, eD_max, eH_max, eW_max, is_label=False)
            result["cond"] = torch.from_numpy(
                np.ascontiguousarray(cond_s.astype(np.float32, copy=False)))
        return result

    def _safe_center_range(
        self, D: int, H: int, W: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """逐轴返中心点 (lo,hi) 区间（hi 独立上界，供 randint/clip）：使最大 scale cube 在界内。
        轴 size < patch 时退为体积中心，接受边界填充（与旧行为一致）。。"""
        eD, eH, eW = self.extract_size
        sD         = int(round(eD * self._max_scale))
        sH         = int(round(eH * self._max_scale))
        sW         = int(round(eW * self._max_scale))

        def _axis(size: int, patch: int) -> Tuple[int, int]:
            half = patch // 2
            lo   = half
            # _extract_cubic_patch 取 [c-patch//2, c-patch//2+patch)
            hi = size - (patch - half)
            if hi <= lo:
                # 该轴体积太小：中心采样，接受填充。
                mid = size // 2
                return mid, mid + 1
            return lo, hi

        return _axis(D, sD), _axis(H, sH), _axis(W, sW)

    def _sample_center(self, vol_idx: int, D: int, H: int, W: int) -> Tuple[int, int, int]:
        """采样中心 (d,h,w) 并夹匯至 _safe_center_range，以免 max-FOV cube 越界
        导致>50% 体素来自边界复制（偏移训练分布）。验证用逐样本确定性
        RNG（见 _sample_rng）。"""
        (dlo, dhi), (hlo, hhi), (wlo, whi) = self._safe_center_range(D, H, W)
        cov = self._val_coverage_pos()
        if cov is not None:
            # 网格覆盖：卷内第 j 个样本用 Halton(2,3,5) 低差异序列均匀铺满
            # 安全中心域（任意 S 均可，无需三轴因子分解）。
            j, _ = cov
            fd, fh, fw = (_halton(j + 1, b) for b in (2, 3, 5))
            return (dlo + min(int(fd * (dhi - dlo)), dhi - dlo - 1),
                    hlo + min(int(fh * (hhi - hlo)), hhi - hlo - 1),
                    wlo + min(int(fw * (whi - wlo)), whi - wlo - 1))
        fg_coords = self._vol_fg_coords[vol_idx]
        rng = self._sample_rng()
        if (self.is_train and self.fg_ratio > 0
                and len(fg_coords) > 0
                and rng.random() < self.fg_ratio):
            idx = int(rng.integers(len(fg_coords)))
            d, h, w = fg_coords[idx]
            # np.clip 上界含于，需 -1。
            d = int(np.clip(int(d), dlo, dhi - 1))
            h = int(np.clip(int(h), hlo, hhi - 1))
            w = int(np.clip(int(w), wlo, whi - 1))
            return (d, h, w)
        return (int(rng.integers(dlo, dhi)),
                int(rng.integers(hlo, hhi)),
                int(rng.integers(wlo, whi)))


# ---------------------------------------------------------------------------
# 3D Whole-Volume Dataset (no sliding window, no sub-cropping)
# ---------------------------------------------------------------------------
class Volume3DWhole(VolumeNpzDatasetBase):
    """整体卷 dataset：全卷 resize 到 extract_size = round(patch_size*oversample)，
    不切块/不采中心。trainer 增强后中心裁为 patch_size。。

    samples_per_volume 控每 epoch 增强变体数；foreground_oversample_ratio 忽略；
    multi_res_scales 必为 [1.0]（整体 resize 上多分辨率无物理意义，Config 验证）。
    输出：image/label/weight_map = (1, eD, eH, eW)。。"""

    def __init__(
        self,
        image_paths         : List[str],
        label_paths         : List[str],
        label_values        : List[int],
        patch_size          : Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio: float = 1.0,
        intensity_min       : float = -1024.0,
        intensity_max       : float = 3071.0,
        normalize           : str = "minmax",
        global_mean         : float = 0.0,
        global_std          : float = 1.0,
        samples_per_volume  : int = 1,
        is_train            : bool = True,
        cache_enabled       : bool = True,
        cache_max_volumes   : int = 0,
        cache_int16         : bool = False,
        region_weights      : Optional[List[float]] = None,
        cond_normalize      : str = "minmax",
        cond_intensity_min  : float = -1024.0,
        cond_intensity_max  : float = 1024.0,
        cond_global_mean    : float = 0.0,
        cond_global_std     : float = 1.0,
        npz_paths           : Optional[List[str]] = None):
        super().__init__(
            image_paths          = image_paths,
            label_paths          = label_paths,
            label_values         = label_values,
            npz_paths            = npz_paths,
            patch_size           = patch_size,
            aug_oversample_ratio = aug_oversample_ratio,
            intensity_min        = intensity_min,
            intensity_max        = intensity_max,
            normalize            = normalize,
            global_mean          = global_mean,
            global_std           = global_std,
            samples_per_volume   = samples_per_volume,
            is_train             = is_train,
            cache_enabled        = cache_enabled,
            cache_max_volumes    = cache_max_volumes,
            cache_int16          = cache_int16,
            region_weights       = region_weights,
            cond_normalize       = cond_normalize,
            cond_intensity_min   = cond_intensity_min,
            cond_intensity_max   = cond_intensity_max,
            cond_global_mean     = cond_global_mean,
            cond_global_std      = cond_global_std)
        # 3 轴同步过采样：与 cubic 一致，给增强（旋转/弹性）留中心裁余量。
        self.extract_size = tuple(
            int(round(p * self.oversample)) for p in self.patch_size)

        logger.info(
            "Whole-volume dataset: %d volumes, extract_size=%s, "
            "samples_per_volume=%d [npz mode]",
            len(self.image_paths), self.extract_size, self.samples_per_volume)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx    = idx % len(self.image_paths)
        img, lbl   = self._load_image(vol_idx), self._load_label(vol_idx)
        eD, eH, eW = self.extract_size

        # 全卷单次 3D zoom。
        img_r = resize_3d(img, eD, eH, eW, is_label=False)
        lbl_r = resize_3d(lbl, eD, eH, eW, is_label=True)
        cond_r = self._load_cond(vol_idx)
        if cond_r is not None:
            cond_r = resize_3d(cond_r, eD, eH, eW, is_label=False)

        result = {
            "image": torch.from_numpy(img_r[np.newaxis]).float(),
            "label": torch.from_numpy(np.ascontiguousarray(lbl_r[np.newaxis]))}

        # 区域权重优先级：样本文件 > 静态映射。
        if self._has_region_weight_file(vol_idx):
            rw_vol = self._load_region_weight(vol_idx)
            # rw 是分级权重（离散），必须 nearest 避免产生伪连续值；与 z_axis/cubic
            # 路径一致（resize_3d(is_label=True) = order=0）。
            rw_vol = resize_3d(rw_vol, eD, eH, eW, is_label=True)
            result["weight_map"] = torch.from_numpy(rw_vol[np.newaxis]).float()
        elif self.region_weights:
            rw_vol = compute_region_weight_map(
                lbl_r, self.label_values, self.region_weights)
            result["weight_map"] = torch.from_numpy(rw_vol).float()
        if cond_r is not None:
            result["cond"] = torch.from_numpy(
                np.ascontiguousarray(cond_r.astype(np.float32, copy=False)))
        return result
