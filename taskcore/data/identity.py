"""数据产物身份与数据集指纹（1-3 / 2-3）。

- ``compute_data_identifier``：由打包口径参数派生稳定短标识（dsid-<hash>）。
  预处理口径（spacing 归一化、target_spacing、label_values、fg_subsample、
  bbox/rw 有无）任一变化 → 标识变化，杜绝不同口径产物静默混用同一目录。
- ``compute_case_intensity_stats``：逐病例前景强度统计（nnU-Net 式采样）。
- ``aggregate_dataset_fingerprint``：跨病例汇聚数据集级前景强度指纹，
  供 ``data.normalize='ct_fingerprint'``（裁剪到 p0.5/p99.5 + 全集 z-score）消费。
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

# 逐病例前景强度采样上限（nnU-Net 同量级；指纹分位数估计足够）。
FG_SAMPLE_CAP = 10_000


def compute_data_identifier(
    *,
    spacing_normalization: bool,
    target_spacing: Optional[Sequence[float]],
    label_values: Sequence[int],
    fg_subsample: int,
    has_bbox: bool,
    has_rw: bool,
) -> str:
    """打包口径的稳定短标识 ``dsid-<sha1[:12]>``。

    仅纳入影响 npz 内容语义的口径参数；与样本集合/顺序无关。
    target_spacing 量化到 1e-4 mm，避免浮点噪声抖动标识。"""
    payload = {
        "spacing_normalization": bool(spacing_normalization),
        "target_spacing": (
            [round(float(s), 4) for s in target_spacing]
            if target_spacing is not None else None),
        "label_values": [int(v) for v in label_values],
        "fg_subsample": int(fg_subsample),
        "has_bbox": bool(has_bbox),
        "has_rw": bool(has_rw),
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
    return f"dsid-{digest}"


def compute_case_intensity_stats(
    image: np.ndarray,
    label: np.ndarray,
    max_samples: int = FG_SAMPLE_CAP,
    seed: int = 42,
) -> Dict[str, object]:
    """逐病例前景（label>0）强度统计 + 采样值。

    返回 dict：mean/std/median/p00_5/p99_5/min/max/n_fg_voxels 标量 +
    ``fg_sample``（float32 采样数组，cap 到 ``max_samples``，seed 确定）。
    无前景时统计为 None、fg_sample 为空数组。"""
    fg_mask = np.asarray(label) > 0
    n_fg = int(fg_mask.sum())
    if n_fg == 0:
        return {
            "mean": None, "std": None, "median": None,
            "p00_5": None, "p99_5": None, "min": None, "max": None,
            "n_fg_voxels": 0,
            "fg_sample": np.zeros((0,), dtype=np.float32),
        }
    vals = np.asarray(image)[fg_mask].astype(np.float32, copy=False)
    if vals.size > max_samples:
        rng = np.random.RandomState(seed)
        vals_sampled = vals[rng.choice(vals.size, max_samples, replace=False)]
    else:
        vals_sampled = vals
    vals64 = vals.astype(np.float64, copy=False)
    p = np.percentile(vals64, (0.5, 50.0, 99.5))
    return {
        "mean": float(vals64.mean()),
        "std": float(vals64.std()),
        "median": float(p[1]),
        "p00_5": float(p[0]),
        "p99_5": float(p[2]),
        "min": float(vals64.min()),
        "max": float(vals64.max()),
        "n_fg_voxels": n_fg,
        "fg_sample": np.ascontiguousarray(
            vals_sampled, dtype=np.float32),
    }


def aggregate_dataset_fingerprint(
    fg_samples: List[np.ndarray],
    n_cases: int,
) -> Optional[Dict[str, object]]:
    """把逐病例前景采样池化为数据集级强度指纹。

    返回 None 表示全数据集无前景采样（无法构建指纹）。"""
    pooled_parts = [
        np.asarray(s, dtype=np.float64).reshape(-1)
        for s in fg_samples if s is not None and np.asarray(s).size]
    if not pooled_parts:
        return None
    pooled = np.concatenate(pooled_parts)
    p = np.percentile(pooled, (0.5, 50.0, 99.5))
    return {
        "fg_mean": float(pooled.mean()),
        "fg_std": float(pooled.std()),
        "fg_median": float(p[1]),
        "fg_p00_5": float(p[0]),
        "fg_p99_5": float(p[2]),
        "fg_min": float(pooled.min()),
        "fg_max": float(pooled.max()),
        "n_samples": int(pooled.size),
        "n_cases": int(n_cases),
    }


def fingerprint_normalization_params(
    fingerprint: Dict[str, object],
) -> Dict[str, float]:
    """由数据集指纹解析 ``normalize='ct_fingerprint'`` 的归一化参数。

    契约：clip 到 [fg_p00_5, fg_p99_5] 后 (x - fg_mean) / fg_std
    （nnU-Net CTNormalization 同款）。"""
    required = ("fg_p00_5", "fg_p99_5", "fg_mean", "fg_std")
    missing = [k for k in required
               if fingerprint.get(k) is None]
    if missing:
        raise ValueError(
            f"dataset_fingerprint is missing key(s) {missing}; re-run "
            "make_data (>= 1.9) with --overwrite to bake intensity "
            "statistics.")
    std = float(fingerprint["fg_std"])
    if not (np.isfinite(std) and std > 0.0):
        raise ValueError(
            f"dataset_fingerprint fg_std={std!r} is not a positive finite "
            "number; the foreground intensity distribution is degenerate.")
    return {
        "intensity_min": float(fingerprint["fg_p00_5"]),
        "intensity_max": float(fingerprint["fg_p99_5"]),
        "global_mean": float(fingerprint["fg_mean"]),
        "global_std": std,
    }
