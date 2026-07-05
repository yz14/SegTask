"""3D 分割 dataset。

三种 patch 模式：SegDataset3D（z_axis 滑块）、SegDataset3DCubic（cubic 3轴滑块）、
SegDataset3DWhole（整体 resize）。输出逐前景类二值通道：
label_values=[0,1,2] → 2 个前景通道。
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# 验证集确定性采样的固定基种子（与样本序号组合派生逐样本 RNG）。
_VAL_SAMPLING_SEED = 0x5EED_2024


# ---------------------------------------------------------------------------
# Volume I/O
# ---------------------------------------------------------------------------
# NIfTI 读取重试：网盘/虚拟路径偶尔报 nifti_image_load failed，重试可恢复。
# 环境变量：SEGTASK_NIFTI_READ_RETRIES（默认 4）、SEGTASK_NIFTI_READ_BACKOFF_S（默认 0.5）。
_NIFTI_READ_RETRIES = max(1, int(os.environ.get("SEGTASK_NIFTI_READ_RETRIES", "4")))
_NIFTI_READ_BACKOFF_S = max(0.0, float(os.environ.get("SEGTASK_NIFTI_READ_BACKOFF_S", "0.5")))


# RuntimeError 消息中含有以下子串表示 OOM，不重试，直接折为 MemoryError。
_ALLOC_ERROR_MARKERS = (
    "bad allocation",
    "failed to allocate memory",
    "std::bad_alloc",
    "cannot allocate memory",
)


def _is_alloc_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _ALLOC_ERROR_MARKERS)


def _sitk_read_with_retry(read_callable, path: str) -> "sitk.Image":
    """有限重试调用 sitk 读取闭包；仅重试 I/O 瞬态故障，OOM 直接报 MemoryError。"""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _NIFTI_READ_RETRIES + 1):
        try:
            return read_callable()
        except RuntimeError as exc:  # SimpleITK 包装底层错误
            if _is_alloc_error(exc):
                # Host OOM — 直接折为 MemoryError，不重试。
                raise MemoryError(
                    f"NIfTI read aborted (host OOM) for {path}: {exc}") from exc
            last_exc = exc
            if attempt >= _NIFTI_READ_RETRIES:
                break
            wait = _NIFTI_READ_BACKOFF_S * (2 ** (attempt - 1))
            logger.warning(
                "NIfTI read failed (attempt %d/%d) for %s: %s — retrying in %.2fs",
                attempt, _NIFTI_READ_RETRIES, path, exc, wait)
            if wait > 0:
                time.sleep(wait)
    raise RuntimeError(
        f"NIfTI read permanently failed after {_NIFTI_READ_RETRIES} attempts "
        f"for {path}: {last_exc}") from last_exc


def load_nifti(path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """NIfTI → (D,H,W) ndarray（SimpleITK）。浮点请求时 sitk 原生代 scl_slope/inter，节省中间 fp64 临时 buffer。"""
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        # 直接解码到请求精度；sitk 同时应用 scl_slope/inter。
        sitk_pixel = (sitk.sitkFloat32 if np_dtype == np.float32
                      else sitk.sitkFloat64)
        read_args = (str(path), sitk_pixel)
    else:
        # 读原生 dtype，不提升，后续手动封装转型。
        read_args = (str(path),)

    img = _sitk_read_with_retry(lambda: sitk.ReadImage(*read_args), path)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X) = (D, H, W)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    return arr


def load_nifti_with_spacing(
    path: str, dtype: np.dtype = np.float32,
) -> "Tuple[np.ndarray, float]":
    """NIfTI → (volume, z_spacing_mm)。仅推理 z-interleave 需要。缺 meta 时 z_spacing 退 1.0。"""
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        sitk_pixel = (sitk.sitkFloat32 if np_dtype == np.float32
                      else sitk.sitkFloat64)
        read_args = (str(path), sitk_pixel)
    else:
        read_args = (str(path),)
    img = _sitk_read_with_retry(lambda: sitk.ReadImage(*read_args), path)
    arr = sitk.GetArrayFromImage(img)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    spacing = img.GetSpacing()  # (sx, sy, sz)
    z_spacing = float(spacing[2]) if len(spacing) >= 3 else 1.0
    if not np.isfinite(z_spacing) or z_spacing <= 0.0:
        z_spacing = 1.0
    return arr, z_spacing


def read_nifti_spacing(path: str) -> "Tuple[float, float, float]":
    """仅读 NIfTI 头（不解码像素）返 (sz, sy, sx) —— numpy 轴序 (D,H,W) 的 mm spacing。
    sitk GetSpacing 为 (sx, sy, sz)，此处倒转对齐 numpy。非有限/非正退 1.0。"""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    sp = reader.GetSpacing()  # (sx, sy, sz)
    sx = float(sp[0]) if len(sp) >= 1 else 1.0
    sy = float(sp[1]) if len(sp) >= 2 else 1.0
    sz = float(sp[2]) if len(sp) >= 3 else 1.0
    out = []
    for s in (sz, sy, sx):
        out.append(s if (np.isfinite(s) and s > 0.0) else 1.0)
    return (out[0], out[1], out[2])


def resample_to_spacing(
    vol: np.ndarray,
    src_spacing: "Tuple[float, float, float]",
    target_spacing: "Tuple[float, float, float]",
    is_label: bool = False) -> np.ndarray:
    """(D,H,W) 体积从 src_spacing 重采样到 target_spacing（均为 numpy 轴序 (D,H,W) mm）。
    新尺寸 = round(shape * src/target)（每轴 >=1）；image order=1 线性、label order=0 近邻。"""
    D, H, W = vol.shape
    new = []
    for n, s, t in zip((D, H, W), src_spacing, target_spacing):
        new.append(max(1, int(round(n * float(s) / float(t)))))
    return resize_3d(vol, new[0], new[1], new[2], is_label=is_label)


def load_nifti_cropped(
    path: str,
    bbox: "Optional[BBox]" = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """流式 bbox-裁剪读取 NIfTI：利用 sitk SetExtractIndex/Size 仅返 ROI，
    平均峰值与裁后体积同量级（可能比全卷读小 ~14×）。返 (D',H',W') C-contig owned ndarray。。"""
    np_dtype = np.dtype(dtype)
    floating = np.issubdtype(np_dtype, np.floating)
    sitk_pixel = (
        sitk.sitkFloat32 if (floating and np_dtype == np.float32)
        else sitk.sitkFloat64 if floating
        else None)

    def _read() -> "sitk.Image":
        # 闭包内重建 reader，避免重试复用部分失败的 IORegion 状态。
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        # 头信息只读（~352 字节）以获 GetSize() 防御性夹匯 bbox。
        reader.ReadImageInformation()
        if bbox is not None:
            # sitk 为 (X,Y,Z) 顺序，与 numpy/本 BBox 相反，需倒转。
            full_w, full_h, full_d = reader.GetSize()
            (d0, d1), (h0, h1), (w0, w1) = bbox
            # bbox 夹匯：在 Execute() 出 cryptic ITK 越界错之前处理。
            d0c = max(0, min(d0, full_d))
            d1c = max(d0c, min(d1, full_d))
            h0c = max(0, min(h0, full_h))
            h1c = max(h0c, min(h1, full_h))
            w0c = max(0, min(w0, full_w))
            w1c = max(w0c, min(w1, full_w))
            if d1c > d0c and h1c > h0c and w1c > w0c:
                reader.SetExtractIndex([w0c, h0c, d0c])
                reader.SetExtractSize([w1c - w0c, h1c - h0c, d1c - d0c])
            # 空 bbox 退为全卷读；保留明确分支供下游错误信息体现。
        if sitk_pixel is not None:
            # 强制 fp32/fp64 输出；sitk 在转型中应用 scl_slope/inter。
            reader.SetOutputPixelType(sitk_pixel)
        return reader.Execute()

    img = _sitk_read_with_retry(_read, path)
    # GetArrayViewFromImage 与 sitk buffer 共内存；del img 前必须 copy 到 owned。
    view = sitk.GetArrayViewFromImage(img)  # (Z, Y, X) = (D', H', W')
    arr = np.array(view, copy=True, order="C")
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    del view
    del img
    return arr


# ---------------------------------------------------------------------------
# NPZ pre-computed package I/O
# ---------------------------------------------------------------------------
# make_data 输出的 <pid>.npz（ZIP_STORED，无 gzip），含：
#   image int16 (D',H',W') HU bbox-裁剪
#   label int16 (D',H',W') 原始标签 bbox-裁剪
#   rw    float32 (D',H',W') +1 偏移后的区域权重（可选）
#   fg_slices int32 (M,) 全前景 z 切片并集
#   fg_coords int32 (N,3) 逐类 argwhere 拼接（seed=42、每类 cap=50000）
#   fg_coords_cls   int16 (N,) 与 fg_coords 对齐的类值（make_data≥1.1）
#   fg_slices_cls_z int32 (K,) 逐类 z 切片拼接（make_data≥1.1）
#   fg_slices_cls   int16 (K,) 与之对齐的类值（make_data≥1.1）
#   meta  object 0-d dict
# numpy 对 .npz 忽略 mmap_mode，逐 worker 为 owned ndarray；OS page cache 跨 worker 共享。


def _open_npz(path: str) -> "np.lib.npyio.NpzFile":
    """打开 npz（仅解析 zip 目录）。allow_pickle=True 供 meta dict 反序列。"""
    return np.load(path, allow_pickle=True)


def load_npz_image(
    path: str,
    intensity_min: float,
    intensity_max: float,
    normalize: str,
    global_mean: float = 0.0,
    global_std: float = 1.0) -> np.ndarray:
    """读 npz image（int16 HU）后运行 preprocess_image → owned fp32。"""
    with _open_npz(path) as f:
        img_int16 = f["image"]
        return preprocess_image(
            img_int16, intensity_min, intensity_max,
            normalize, global_mean, global_std,
            inplace=False)


def load_npz_label(path: str) -> np.ndarray:
    """返 npz 中 owned int16 label ndarray。"""
    with _open_npz(path) as f:
        return f["label"]


def load_npz_region_weight(path: str) -> Optional[np.ndarray]:
    """返 owned fp32 区域权重（+1 偏移已由 make_data 加过）；无 rw 返 None。"""
    with _open_npz(path) as f:
        if "rw" not in f.files:
            return None
        rw = f["rw"]
    if rw.dtype != np.float32:
        rw = rw.astype(np.float32, copy=False)
    return rw


def npz_has_rw(path: str) -> bool:
    """仅查 npz 是否含 rw 键（不解码数据）。"""
    with _open_npz(path) as f:
        return "rw" in f.files


def load_npz_fg_slices(path: str) -> np.ndarray:
    """返 npz 内预计算的逐 z 前景切片索引（裁剪后坐标系）。"""
    with _open_npz(path) as f:
        return np.asarray(f["fg_slices"], dtype=np.int32)


def load_npz_fg_coords(path: str) -> np.ndarray:
    """返 npz 内预计算的 (N,3) 前景 voxel 坐标（裁剪后坐标系）。"""
    with _open_npz(path) as f:
        return np.asarray(f["fg_coords"], dtype=np.int32)


def _group_fg_coords_by_class(
    f: "np.lib.npyio.NpzFile", coords: np.ndarray) -> Optional[List[np.ndarray]]:
    """按 fg_coords_cls 将 coords 分组为逐类非空坐标数组列表；
    无该键（make_data<1.1 的旧 npz）返 None。"""
    if "fg_coords_cls" not in f.files:
        return None
    cls = np.asarray(f["fg_coords_cls"])
    groups = [coords[cls == v] for v in np.unique(cls)]
    return [g for g in groups if len(g)] or None


def _group_fg_slices_by_class(
    f: "np.lib.npyio.NpzFile") -> Optional[List[np.ndarray]]:
    """按 fg_slices_cls 将 fg_slices_cls_z 分组为逐类非空 z 索引列表；
    无该键（make_data<1.1 的旧 npz）返 None。"""
    if "fg_slices_cls" not in f.files or "fg_slices_cls_z" not in f.files:
        return None
    cls = np.asarray(f["fg_slices_cls"])
    zs  = np.asarray(f["fg_slices_cls_z"], dtype=np.int32)
    groups = [zs[cls == v] for v in np.unique(cls)]
    return [g for g in groups if len(g)] or None


def load_npz_label_for_split(path: str) -> np.ndarray:
    """owned int16 label copy，供 loader.py 预扫描使用。强制 copy 以免父进程持有 mmap 句柄。"""
    with _open_npz(path) as f:
        return np.array(f["label"])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(
    volume: np.ndarray,
    intensity_min: float,
    intensity_max: float,
    normalize: str,
    global_mean: float = 0.0,
    global_std: float = 1.0,
    inplace: bool = False) -> np.ndarray:
    """强度窗 + 归一化 → fp32。单次分配 + in-place clip/normalize，避免中间临时 buffer。

    inplace=True 且输入本是 fp32 时复用 buffer（调用方拥有该数组才可启用）。。"""
    vol = np.asarray(volume, dtype=np.float32)
    if vol is volume and not inplace:
        # 输入本为 fp32 且未明示同意 in-place：拷贝避免污染上游。
        vol = volume.copy()
    np.clip(vol, intensity_min, intensity_max, out=vol)

    if normalize == "minmax":
        denom = float(intensity_max - intensity_min)
        if denom > 0:
            vol -= float(intensity_min)
            vol /= denom
        else:
            vol.fill(0.0)
    elif normalize == "zscore":
        if global_std > 0:
            vol -= float(global_mean)
            vol /= float(global_std)
        else:
            vol.fill(0.0)
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return vol


def compute_region_weight_map(
    volume: np.ndarray, label_values: List[int],
    region_weights: List[float]) -> np.ndarray:
    """由整数 label 与逐值权重生成 (1,D,H,W) fp32 区域权重图；未命中标签赋 1.0。"""
    vol  = np.round(volume).astype(np.int16)  # int16已经足够
    wmap = np.ones_like(vol, dtype=np.float32)
    for lv, w in zip(label_values, region_weights):
        wmap[vol == lv] = w
    return wmap[np.newaxis]  # (1, D, H, W)


def load_region_weight_volume(
    path: str, bbox: "Optional[BBox]" = None) -> np.ndarray:
    """读样本区域权重 NIfTI 并 +1 偏移（背景变 1.0，标注 w 变 w+1）。
    传 bbox 时 “裁后 +1”，避免全卷 fp32 临时 buffer。。"""
    rw = load_nifti_cropped(path, bbox=bbox, dtype=np.float32)
    rw += 1.0
    return rw


def preprocess_label(volume: np.ndarray, label_values: List[int]) -> np.ndarray:
    """整数 label → 逐前景类二值堆叠 (num_fg,D,H,W) fp32；label_values 首位为背景。"""
    vol = np.round(volume).astype(np.int32)
    fg_values = label_values[1:]
    # 向量化比较：(C,1,1,1) == (D,H,W) → (C,D,H,W)。
    lv = np.array(fg_values, dtype=np.int32).reshape(-1, *([1] * vol.ndim))
    return (vol[np.newaxis] == lv).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------
def resize_3d(arr: np.ndarray, target_d: int, target_h: int, target_w: int, is_label: bool = False) -> np.ndarray:
    """(D,H,W) 或 (C,D,H,W) resize：图像 order=1 线性，label order=0 近邻。"""
    if arr.ndim == 3:
        D, H, W = arr.shape
        if D == target_d and H == target_h and W == target_w:
            return arr
        factors = [target_d / D, target_h / H, target_w / W]
    elif arr.ndim == 4:
        _, D, H, W = arr.shape
        if D == target_d and H == target_h and W == target_w:
            return arr
        factors = [1.0, target_d / D, target_h / H, target_w / W]
    else:
        raise ValueError(f"Expected 3D or 4D array, got {arr.ndim}D")
    order = 0 if is_label else 1
    # zoom 已返输入 dtype；copy=False 避免冷拷贝。
    return zoom(arr, factors, order=order).astype(arr.dtype, copy=False)


# ---------------------------------------------------------------------------
# Bounding-box helpers (optional ROI cropping of image / label volumes)
# ---------------------------------------------------------------------------
# bbox 约定：((d0,d1),(h0,h1),(w0,w1))，半开区间。
BBox = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def compute_bbox_from_volume(vol: np.ndarray) -> Optional[BBox]:
    """计算 (D,H,W) ROI 掩膜非零区域的轴齐包围盒；掩膜全空返 None。使用 np.any 逐轴崩装，比 argwhere 快且低内存。"""
    if vol.ndim != 3:
        raise ValueError(f"BBox volume must be 3D (D,H,W), got {vol.ndim}D")
    mask = np.round(vol).astype(np.int16) != 0
    if not mask.any():
        return None
    d_any = np.any(mask, axis=(1, 2))
    h_any = np.any(mask, axis=(0, 2))
    w_any = np.any(mask, axis=(0, 1))

    def _span(flat: np.ndarray) -> Tuple[int, int]:
        idx = np.where(flat)[0]
        return int(idx[0]), int(idx[-1]) + 1  # half-open

    return (_span(d_any), _span(h_any), _span(w_any))


# ---------------------------------------------------------------------------
# Volume cache
# ---------------------------------------------------------------------------
class VolumeCache:
    """内存卷 LRU 缓存。max_volumes=0 不限容量；enabled=False 禁用。"""

    def __init__(self, enabled: bool = False, max_volumes: int = 0):
        self._enabled = enabled
        self._max = max(int(max_volumes), 0)
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: str) -> Optional[np.ndarray]:
        if not self._enabled:
            return None
        data = self._store.get(path)
        if data is not None:
            # Mark as most-recently-used.
            self._store.move_to_end(path)
        return data

    def put(self, path: str, data: np.ndarray) -> None:
        if not self._enabled:
            return
        if path in self._store:
            self._store.move_to_end(path)
            self._store[path] = data
            return
        self._store[path] = data
        if self._max > 0:
            while len(self._store) > self._max:
                # popitem(last=False) pops the LEAST-recently-used entry.
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)

    # Pickling：传到 DataLoader worker 时丢弃缓存内容（Windows spawn 下防管道超限，
    # 并且 worker 间不共享内存，传输是纯开销）。
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_store"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not isinstance(self._store, OrderedDict):
            self._store = OrderedDict()


# ---------------------------------------------------------------------------
# Common npz-backed Dataset base
# ---------------------------------------------------------------------------
class SegDatasetNpzBase(Dataset):
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
        region_weights      : Optional[List[float]]):
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

        self._img_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._lbl_cache = VolumeCache(cache_enabled, cache_max_volumes)
        self._rw_cache  = VolumeCache(cache_enabled, cache_max_volumes)

        # NPZ 预计算包（make_data 产出）提供 bbox / fg 索引 / 可选 rw。
        self._npz_paths       : List[str]       = list(npz_paths)
        self._npz_has_rw_cache: Dict[int, bool] = {}

        # 逐 worker 采样 RNG（惰性创建，见 _rng）。
        self._rng_cache: Optional[np.random.Generator] = None
        self._rng_wid  : Optional[int] = None
        # 当前样本序号（__getitem__ 入口处写入，驱动验证确定性采样）。
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
        save_best / early-stopping / plateau 不被采样噪声驱动。
        """
        if self.is_train:
            return self._rng()
        return np.random.default_rng((_VAL_SAMPLING_SEED, self._sample_idx))

    # ------------------------------------------------------------------
    # 共用 npz 读取（子类可直接复用，缓存按 path 共享于同一 worker）
    # ------------------------------------------------------------------
    def _load_image(self, vol_idx: int) -> np.ndarray:
        """加载+预处理 image（npz，带缓存）。"""
        path   = self._npz_paths[vol_idx]
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

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_volume


# ---------------------------------------------------------------------------
# 3D Segmentation Dataset (z-axis sliding window)
# ---------------------------------------------------------------------------
class SegDataset3D(SegDatasetNpzBase):
    """3D z 轴滑窗 dataset。z 轴滑动抖中心 z，折取 round(eD*s) 切片，
    仅 z 过采样；H/W 全分辨率 resize 到 patch_size。与 predictor.sliding.sliding_window_z 一致。

    多分辨率 multi_res_scales=[1.0] 为单分辨率；s>1 强制 edge-replicate 以保留物理 z-FOV。
    输出 shape：image/label/weight_map = (C_res, eD, pH, pW)。
    """

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
        region_weights             : Optional[List[float]] = None,
        z_boundary_mode            : str = "stretch",
        npz_paths                  : Optional[List[str]] = None):

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
            region_weights       = region_weights)
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

        # 逐卷前景索引（从 npz 读）驱动 _sample_z 过采样。含 *_cls 键时额外
        # 构建逐类切片列表，前景采样先选类再选切片（类均衡）。
        self._vol_fg_slices : List[np.ndarray] = []
        self._vol_fg_slices_by_cls: List[Optional[List[np.ndarray]]] = []
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
            self._vol_fg_slices_by_cls.append(_group_fg_slices_by_class(f))
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
        return self._getitem_max_fov(img, lbl, rw_vol, z, eD, eH, eW)

    def _getitem_max_fov(
        self,
        img    : np.ndarray,
        lbl    : np.ndarray,
        rw_vol : Optional[np.ndarray],
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

        # 面内 resize 到 (eH,eW)；D 轴保持 eD_max（不重采样）。
        img_s = resize_3d(img_s, eD_max, eH, eW, is_label=False)
        lbl_s = resize_3d(lbl_s, eD_max, eH, eW, is_label=True)
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
        return result

    def _sample_z(self, vol_idx: int, D_vol: int) -> int:
        """采样中心 z：训练以 fg_ratio 概率从前景切片采样（npz 含逐类索引时
        先均匀选类再选该类切片，避免稀有类被大器官挤压），否则均匀采样；
        验证用逐样本确定性 RNG 均匀采样（见 _sample_rng）。"""
        fg_slices = self._vol_fg_slices[vol_idx]
        rng = self._sample_rng()
        if (self.is_train and self.fg_ratio > 0
            and len(fg_slices) > 0
            and rng.random() < self.fg_ratio):
            per_cls = self._vol_fg_slices_by_cls[vol_idx]
            if per_cls:
                cls_slices = per_cls[int(rng.integers(len(per_cls)))]
                return int(rng.choice(cls_slices))
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


class SegDataset3DCubic(SegDatasetNpzBase):
    """3D cubic patch dataset：以 (d,h,w) 为中心抽 3D cube。支持增强过采样与
    多分辨率（同中心多 scale resize 后拼通道）。输出与 SegDataset3D 一致：(C_res, eD, eH, eW)，
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
        region_weights             : Optional[List[float]] = None,
        npz_paths                  : Optional[List[str]] = None):
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
            region_weights       = region_weights)
        
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

        # 逐卷 3D fg 坐标索引（make_data 预抽：seed=42, 每类 cap=50000）驱动
        # _sample_center 过采样；含 fg_coords_cls 时逐类分组，先选类再选点。
        self._vol_shapes   : List[Tuple[int, int, int]] = []
        self._vol_fg_coords: List[np.ndarray]           = []
        self._vol_fg_coords_by_cls: List[Optional[List[np.ndarray]]] = []
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
            self._vol_fg_coords_by_cls.append(
                _group_fg_coords_by_class(f, coords))
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
        return self._getitem_max_fov(center, img, lbl, rw_vol, eD, eH, eW)

    def _getitem_max_fov(
        self,
        center: Tuple[int, int, int],
        img   : np.ndarray,
        lbl   : np.ndarray,
        rw_vol: Optional[np.ndarray],
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

        result = {
            # 领头 "1" = 压叠 C_res 轴；trainer 逐视图裁+resize。
            # ascontiguousarray：无越界填充时 _extract_cubic_patch 返回的是缓存卷
            # 的视图，这里复制断开别名，避免下游 in-place 操作污染 LRU 缓存；
            # 同时保证内存连续，利于 pin_memory。
            "image": torch.from_numpy(
                np.ascontiguousarray(img_s[None], dtype=np.float32)),
            "label": torch.from_numpy(np.ascontiguousarray(lbl_s[None]))}
        if rw_s is not None:
            # rw 离散权重，_extract_cubic_patch 已是按位裁剪 + edge-pad，无重采样，无需 nearest。
            result["weight_map"] = torch.from_numpy(
                np.ascontiguousarray(rw_s[None], dtype=np.float32))
        elif self.region_weights:
            rw_s = compute_region_weight_map(
                lbl_s, self.label_values, self.region_weights)
            result["weight_map"] = torch.from_numpy(
                rw_s.astype(np.float32, copy=False))
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
        fg_coords = self._vol_fg_coords[vol_idx]
        rng = self._sample_rng()
        if (self.is_train and self.fg_ratio > 0
                and len(fg_coords) > 0
                and rng.random() < self.fg_ratio):
            # 含逐类索引时先均匀选类再选点（类均衡）；旧 npz 退回合并采样。
            per_cls = self._vol_fg_coords_by_cls[vol_idx]
            if per_cls:
                fg_coords = per_cls[int(rng.integers(len(per_cls)))]
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
class SegDataset3DWhole(SegDatasetNpzBase):
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
        region_weights      : Optional[List[float]] = None,
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
            region_weights       = region_weights)
        # 3 轴同步过采样：与 cubic 一致，给增强（旋转/弹性）留中心裁余量。
        self.extract_size = tuple(
            int(round(p * self.oversample)) for p in self.patch_size)

        logger.info(
            "Whole-volume dataset: %d volumes, extract_size=%s, "
            "samples_per_volume=%d [npz mode]",
            len(self.image_paths), self.extract_size, self.samples_per_volume)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._sample_idx = idx
        vol_idx    = idx % len(self.image_paths)
        img, lbl   = self._load_image(vol_idx), self._load_label(vol_idx)
        eD, eH, eW = self.extract_size

        # 全卷单次 3D zoom。
        img_r = resize_3d(img, eD, eH, eW, is_label=False)
        lbl_r = resize_3d(lbl, eD, eH, eW, is_label=True)

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
        return result