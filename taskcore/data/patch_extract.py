"""按 ``patch_mode`` 抽取严格 ``patch_size`` 体素块（P2c 叶子层）。

cls / det / ssl 探针 / predictor 与 seg 训练口径一致：

* ``whole``   —— 全卷 resize
* ``cubic``   —— 以 center 为中心 cube（edge-pad）
* ``z_axis`` / ``2_5d`` —— z 轴 edge-pad 抽片 + 面内 resize
"""

from __future__ import annotations

from typing import Callable, FrozenSet

import numpy as np

from .dataset import extract_z_patch_padded, resize_3d
from .patch_ops import Center3, Size3, extract_cubic_patch

VALID_PATCH_MODES: FrozenSet[str] = frozenset(
    {"whole", "z_axis", "cubic", "2_5d"})


def normalize_patch_mode(mode: str) -> str:
    """校验并规范化 patch_mode 字符串。"""
    m = str(mode).lower()
    if m not in VALID_PATCH_MODES:
        raise ValueError(
            f"bad patch_mode: {mode!r}; expected one of "
            f"{sorted(VALID_PATCH_MODES)}.")
    return m


def extract_patch_by_mode(
    vol: np.ndarray,
    mode: str,
    center: Center3,
    patch_size: Size3,
    *,
    is_label: bool = False,
) -> np.ndarray:
    """按 ``patch_mode`` 从 ``vol`` 抽取严格 ``patch_size`` patch。"""
    m = normalize_patch_mode(mode)
    pD, pH, pW = (int(x) for x in patch_size)
    if m == "whole":
        return resize_3d(vol, pD, pH, pW, is_label=is_label)
    if m == "cubic":
        return extract_cubic_patch(vol, center, (pD, pH, pW))
    # z_axis / 2_5d：仅 center[0]（z）参与；H/W 整片 resize。
    slab = extract_z_patch_padded(vol, int(center[0]), pD)
    return resize_3d(slab, pD, pH, pW, is_label=is_label)


def resolve_patch_center(
    mode: str,
    *,
    sample_z: Callable[[], int],
    sample_center: Callable[[], Center3],
) -> Center3:
    """按 patch_mode 解析采样中心（whole 忽略回调）。"""
    m = normalize_patch_mode(mode)
    if m in ("z_axis", "2_5d"):
        return (int(sample_z()), 0, 0)
    if m == "cubic":
        c = sample_center()
        return (int(c[0]), int(c[1]), int(c[2]))
    return (0, 0, 0)


__all__ = [
    "VALID_PATCH_MODES",
    "extract_patch_by_mode",
    "normalize_patch_mode",
    "resolve_patch_center",
]
