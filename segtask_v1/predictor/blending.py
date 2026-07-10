"""Predictor 包几何/概率 helpers（R6）。

R6 抽自 ``predictor.py``：

* ``compute_1d_positions`` —— 一维滑窗 (start, end) 列表
* ``build_1d_weight`` —— 对称 1D blending 窗（gaussian / 均匀）
* ``build_3d_weight`` —— 可分离 3D blending 权重（三轴外积）
* ``prob_to_label`` —— 概率体 → 整数 label map（NaN-safe，自动选最小 dtype）

零外部 SegTask 依赖；可独立单元测试。
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sliding-window 1D position calculation
# ---------------------------------------------------------------------------
def compute_1d_positions(
    length: int, patch: int, stride: int,
) -> List[Tuple[int, int]]:
    """逐轴返 ``(start, end)`` 窗口列表。

    * ``length <= patch`` 时单窗 ``(0, length)``
    * 否则按 ``stride`` 滑动；尾窗反推使长度恢复为 ``patch``（全覆盖）
    """
    if length <= patch:
        return [(0, length)]
    positions: List[Tuple[int, int]] = []
    pos = 0
    while pos + patch <= length:
        positions.append((pos, pos + patch))
        pos += stride
    if positions[-1][1] < length:
        positions.append((length - patch, length))
    return positions


# ---------------------------------------------------------------------------
# Blending weights
# ---------------------------------------------------------------------------
def build_1d_weight(n: int, mode: str = "gaussian") -> np.ndarray:
    """对称 1D blending 窗（长 ``n``），fp32。

    * ``gaussian``: 高斯核，σ = n/4，中心权重最大
    * 其他: 均匀 1.0
    """
    if mode == "gaussian" and n > 1:
        center = (n - 1) / 2.0
        sigma = max(n / 4.0, 1e-6)
        z = np.arange(n, dtype=np.float32)
        return np.exp(-0.5 * ((z - center) / sigma) ** 2).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def build_3d_weight(pD: int, pH: int, pW: int, mode: str) -> np.ndarray:
    """可分离 3D blending 权重（三轴独立 1D 外积），fp32。"""
    if mode == "gaussian":
        wd = build_1d_weight(pD, "gaussian")
        wh = build_1d_weight(pH, "gaussian")
        ww = build_1d_weight(pW, "gaussian")
        return (wd[:, None, None] * wh[None, :, None]
                * ww[None, None, :]).astype(np.float32)
    return np.ones((pD, pH, pW), dtype=np.float32)


# ---------------------------------------------------------------------------
# Probability volume → integer label map
# ---------------------------------------------------------------------------
def prob_to_label(
    prob_volume: np.ndarray,
    *,
    label_values: Sequence[int],
    num_fg: int,
    threshold: Union[float, Sequence[float]],
) -> np.ndarray:
    """概率体 ``(num_fg, D, H, W)`` → 整数 label map ``(D, H, W)``。

    * 逐体素：``max fg 概率 > threshold`` 取对应 ``label_values[1:][argmax]``，否则 ``label_values[0]``
    * ``threshold`` 可为标量（全类共享）或逐前景类序列（长度 = num_fg，与
      ``label_values[1:]`` 一一对应）；逐类时每个体素按其 argmax 类的阈值判背景
    * NaN 体素强制为背景并 ``logger.error``（典型成因：fp16 LayerNorm 溢出 → "全前景"假象）
    * 输出 dtype 选能装下所有 ``label_values`` 的最小有符号整型
    """
    bg_val = label_values[0]
    fg_values = np.array(label_values[1:], dtype=np.int64)
    assert len(fg_values) == num_fg

    nan_mask = np.isnan(prob_volume).any(axis=0)  # (D, H, W)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        total = int(nan_mask.size)
        logger.error(
            "prob_to_label: %d/%d voxels (%.2f%%) contain NaN "
            "probabilities — forcing to background. Root cause is "
            "almost always fp16 forward overflow; rerun inference "
            "with '--precision bf16' (or 'fp32').",
            n_nan, total, 100.0 * n_nan / max(1, total))
        # NaN → -inf 使 argmax/max 忽略；nan_mask 后面强制为 bg。
        prob_volume = np.where(np.isnan(prob_volume),
                               np.float32(-np.inf), prob_volume)

    max_prob = prob_volume.max(axis=0)            # (D, H, W)
    max_class = prob_volume.argmax(axis=0)        # (D, H, W)
    label_map = fg_values[max_class]
    thr = np.asarray(threshold, dtype=np.float32)
    # 与验证侧（utils.dice_batch_stats 等的 ``prob > threshold``）同一契约：
    # 严格大于阈值才取前景，``prob == threshold`` 判背景。
    if thr.ndim == 0:
        below = max_prob <= float(thr)
    else:
        if thr.shape != (num_fg,):
            raise ValueError(
                f"prob_to_label: per-class threshold length {thr.shape[0]} "
                f"!= num_fg {num_fg}.")
        below = max_prob <= thr[max_class]
    label_map[below] = bg_val
    if n_nan > 0:
        label_map[nan_mask] = bg_val

    # 选能装下所有 label 的最小有符号整型。
    max_abs = int(max(abs(int(v)) for v in label_values))
    if max_abs <= np.iinfo(np.int8).max:
        out_dtype = np.int8
    elif max_abs <= np.iinfo(np.int16).max:
        out_dtype = np.int16
    else:
        out_dtype = np.int32
    return label_map.astype(out_dtype)


__all__ = [
    "compute_1d_positions",
    "build_1d_weight",
    "build_3d_weight",
    "prob_to_label",
]
