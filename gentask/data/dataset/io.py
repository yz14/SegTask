"""Shared dataset helpers for 3D/2.5D volume I/O and patching.

Provides NIfTI loading, intensity preprocessing, and patch-based dataset
classes used by gentask's super-resolution path and the shared legacy segmentation configs.
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


# ---------------------------------------------------------------------------
# Volume I/O
# ---------------------------------------------------------------------------
# NIfTI 读取重试：网盘/虚拟路径偊尔报 nifti_image_load failed，重试可恢复。
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


def read_nifti_spacing_zyx(path: str) -> Tuple[float, float, float]:
    """仅读 NIfTI 头信息返 (sz, sy, sx) spacing（mm，与体轴 (D,H,W) 同序）。
    缺失/非法分量退 1.0。"""
    def _read() -> "sitk.Image":
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return reader
    reader = _sitk_read_with_retry(_read, path)
    sp = reader.GetSpacing()  # (sx, sy, sz)
    out = []
    for i in (2, 1, 0):
        v = float(sp[i]) if len(sp) > i else 1.0
        out.append(v if np.isfinite(v) and v > 0.0 else 1.0)
    return (out[0], out[1], out[2])


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
#   cond  float32/int16 (C,D',H',W') 条件体（可选）
#   fg_slices int32 (M,)
#   fg_coords int32 (N,3) seed=42、cap=50000
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


def load_npz_cond(path: str) -> Optional[np.ndarray]:
    """返 owned 条件体 ndarray；无 cond 返 None。"""
    with _open_npz(path) as f:
        if "cond" not in f.files:
            return None
        cond = f["cond"]
    return np.array(cond, copy=True)


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


def load_npz_spacing(path: str) -> Optional[Tuple[float, float, float]]:
    """返 npz meta 中烘焙的 (sz, sy, sx) spacing（mm）；旧包无该字段返 None。"""
    with _open_npz(path) as f:
        if "meta" not in f.files:
            return None
        meta = f["meta"].item()
    sp = meta.get("spacing_zyx") if isinstance(meta, dict) else None
    if not sp or len(sp) != 3:
        return None
    return (float(sp[0]), float(sp[1]), float(sp[2]))


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


def denormalize_image(
    volume: np.ndarray,
    intensity_min: float,
    intensity_max: float,
    normalize: str,
    global_mean: float = 0.0,
    global_std: float = 1.0) -> np.ndarray:
    """``preprocess_image`` 的逆变换：归一化域 → 原强度（HU）。

    minmax: ``x*(max-min)+min``；zscore: ``x*std+mean``。返回 fp32 新数组
    （不修改输入）。用于推理写出时恢复物理标定。
    """
    vol = np.asarray(volume, dtype=np.float32).copy()
    if normalize == "minmax":
        vol *= float(intensity_max - intensity_min)
        vol += float(intensity_min)
    elif normalize == "zscore":
        vol *= float(global_std)
        vol += float(global_mean)
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
