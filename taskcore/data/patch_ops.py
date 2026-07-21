"""Patch 抽取纯函数（P2c 叶子层）。

四 ``patch_mode`` 口径中 **cubic / z / whole** 的立方体与中心域计算在此
单点维护；seg / cls / det / gen / ssl / predictor 共用，避免五份复制漂移。

* ``extract_cubic_patch`` —— 以 center 为中心抽出严格 ``size`` cube，越界
  ``edge`` 填充；无 padding 时 ``copy`` 断开 LRU 缓存别名。
* ``extract_cubic_patch_with_origin`` —— 同上，并返回逐轴逻辑左边界 ``lo``
  （可为负，供检测 ``crop_boxes`` 联动）。
* ``safe_center_range`` —— 逐轴合法中心半开区间 ``(lo, hi)``。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

Center3 = Tuple[int, int, int]
Size3 = Tuple[int, int, int]
AxisRange = Tuple[int, int]
CenterRanges = Tuple[AxisRange, AxisRange, AxisRange]


def extract_cubic_patch(
    vol: np.ndarray,
    center: Center3,
    size: Size3,
) -> np.ndarray:
    """以 ``center`` 为中心抽出严格 ``size`` cube；越界 ``edge`` 填充。"""
    D, H, W = vol.shape
    pD, pH, pW = size
    cd, ch, cw = center

    starts, ends, pad_before, pad_after = [], [], [], []
    for c, p, s in ((cd, pD, D), (ch, pH, H), (cw, pW, W)):
        half = p // 2
        lo = c - half
        hi = lo + p
        src_lo = max(lo, 0)
        src_hi = min(hi, s)
        starts.append(src_lo)
        ends.append(src_hi)
        pad_before.append(max(-lo, 0))
        pad_after.append(max(hi - s, 0))

    patch = vol[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    if any(pb > 0 or pa > 0 for pb, pa in zip(pad_before, pad_after)):
        patch = np.pad(
            patch,
            list(zip(pad_before, pad_after)),
            mode="edge")
    else:
        patch = patch.copy()

    return patch


def extract_cubic_patch_with_origin(
    vol: np.ndarray,
    center: Center3,
    patch: Size3,
) -> Tuple[np.ndarray, Center3]:
    """``extract_cubic_patch`` + 逐轴逻辑左边界 ``lo``（检测框联动用）。"""
    slices, pads, los = [], [], []
    for dim, c, p in zip(vol.shape, center, patch):
        lo = c - p // 2
        hi = lo + p
        pads.append((max(-lo, 0), max(hi - dim, 0)))
        slices.append(slice(max(lo, 0), min(hi, dim)))
        los.append(lo)
    out = vol[tuple(slices)]
    if any(a or b for a, b in pads):
        out = np.pad(out, pads, mode="edge")
    else:
        out = out.copy()
    return out, (los[0], los[1], los[2])


def _axis_center_range(size: int, patch: int) -> AxisRange:
    """单轴合法中心半开区间 ``(lo, hi)``。"""
    half = patch // 2
    lo = half
    hi = size - (patch - half)
    if hi <= lo:
        mid = size // 2
        return mid, mid + 1
    return lo, hi


def safe_center_range(
    shape: Sequence[int],
    patch: Sequence[int],
) -> Union[CenterRanges, List[AxisRange]]:
    """逐轴返中心点 ``(lo, hi)`` 半开区间；``shape``/``patch`` 长度须一致。"""
    if len(shape) != len(patch):
        raise ValueError(
            f"shape and patch must have same length; got {len(shape)} vs "
            f"{len(patch)}.")
    return tuple(_axis_center_range(int(s), int(p)) for s, p in zip(shape, patch))


__all__ = [
    "extract_cubic_patch",
    "extract_cubic_patch_with_origin",
    "safe_center_range",
]
