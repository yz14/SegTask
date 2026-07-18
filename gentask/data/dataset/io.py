"""Volume I/O 与预处理：共享实现已与 ``taskcore.data.dataset`` 合流。

本模块仅保留生成任务专有的原语（条件体、spacing 头读取、反归一化），其余
NIfTI/npz 读取、强度预处理、区域权重、resize、bbox 均 re-export taskcore
实现（含 memmap 快路、n±1 形状校正、region_weights "+1" 语义等后续修正）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

from taskcore.data.dataset import (  # noqa: F401  (re-export，保持旧 import 路径可用)
    BBox,
    _is_alloc_error,
    _open_npz,
    _sitk_read_with_retry,
    compute_bbox_from_volume,
    compute_region_weight_map,
    load_nifti,
    load_nifti_cropped,
    load_nifti_with_spacing,
    load_npz_fg_coords,
    load_npz_fg_slices,
    load_npz_image,
    load_npz_label,
    load_npz_label_for_split,
    load_npz_region_weight,
    load_region_weight_volume,
    npz_has_rw,
    preprocess_image,
    resize_3d,
)


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


def load_npz_cond(path: str) -> Optional[np.ndarray]:
    """返 owned 条件体 ndarray；无 cond 返 None。"""
    with _open_npz(path) as f:
        if "cond" not in f.files:
            return None
        cond = f["cond"]
    return np.array(cond, copy=True)


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
