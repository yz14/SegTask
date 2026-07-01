"""离线评测与在线探针共用的指标工具。"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _rank_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """手写二分类 ROC-AUC（Mann-Whitney rank 形式，支持 ties）。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.5
    pos = y_true > 0.5
    n_pos = int(pos.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0
        i = j
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    sum_pos = float(full_ranks[pos].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_f1(y_true: np.ndarray, y_score: np.ndarray,
               threshold: float = 0.5) -> float:
    """手写二分类 F1（阈值 0.5）。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1) > 0.5
    y_pred = np.asarray(y_score, dtype=np.float64).reshape(-1) >= threshold
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:
        return 0.0
    return float((2.0 * tp) / denom)


def macro_cls_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """逐类 ROC-AUC / F1，然后取 macro 平均。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return {"cls_auc": 0.0, "cls_f1": 0.0}
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_score = y_score[:, None]
    aucs = [_rank_auc(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]
    f1s = [_binary_f1(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]
    return {
        "cls_auc": float(np.mean(aucs)) if aucs else 0.0,
        "cls_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _as_bool_mask(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return arr > 0
    return arr > 0.5


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """mask_a 到 mask_b 的边界距离集合。"""
    if not mask_a.any() or not mask_b.any():
        return np.asarray([], dtype=np.float64)
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    if mask_a.ndim < 2:
        raise ValueError(f"hd95 expects spatial masks, got ndim={mask_a.ndim}.")
    struct = np.ones([3] * mask_a.ndim, dtype=bool)
    surf_a = np.logical_xor(mask_a, binary_erosion(mask_a, structure=struct,
                                                   border_value=0))
    surf_b = np.logical_xor(mask_b, binary_erosion(mask_b, structure=struct,
                                                   border_value=0))
    if not surf_a.any() or not surf_b.any():
        return np.asarray([], dtype=np.float64)
    dt_b = distance_transform_edt(~mask_b)
    dt_a = distance_transform_edt(~mask_a)
    return np.concatenate([dt_b[surf_a], dt_a[surf_b]]).astype(np.float64, copy=False)


def hd95(pred, target) -> float:
    """95% 分位对称 Hausdorff 距离。

    约定：
    - 两 mask 相同且非空：返回 0。
    - 仅一侧为空：返回 ``nan``，由上层在 macro 平均中跳过。
    - 两侧都为空：返回 0。

    支持 2D/3D binary mask，或形状为 ``(B, C, *spatial)`` 的 logits/mask 张量；
    前两维会被折叠为独立样本逐个计算，再对 finite 值取均值。
    """
    pred_arr = np.asarray(pred)
    tgt_arr = np.asarray(target)
    if pred_arr.shape != tgt_arr.shape:
        raise ValueError(f"hd95 shape mismatch: {pred_arr.shape} vs {tgt_arr.shape}")
    if pred_arr.ndim > 3:
        pred_arr = pred_arr.reshape(-1, *pred_arr.shape[2:])
        tgt_arr = tgt_arr.reshape(-1, *tgt_arr.shape[2:])
        vals = [hd95(p, t) for p, t in zip(pred_arr, tgt_arr)]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0
    pred_mask = _as_bool_mask(pred_arr)
    tgt_mask = _as_bool_mask(tgt_arr)
    if np.array_equal(pred_mask, tgt_mask):
        return 0.0
    if not pred_mask.any() and not tgt_mask.any():
        return 0.0
    if not pred_mask.any() or not tgt_mask.any():
        return float("nan")
    dists = _surface_distances(pred_mask, tgt_mask)
    if dists.size == 0:
        return 0.0 if np.array_equal(pred_mask, tgt_mask) else float("nan")
    return float(np.percentile(dists, 95))


__all__ = [
    "macro_cls_metrics",
    "_binary_f1",
    "_rank_auc",
    "hd95",
]
