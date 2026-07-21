"""Patch 中心采样辅助（P2c 叶子层）。

* ``WorkerNumpyRng`` —— DataLoader 逐 worker 独立 ``numpy.random.Generator``
* ``z_grid_center`` / ``halton_center`` —— val 网格覆盖（seg/cls/det 共用）
* ``halton`` —— 低差异序列（val cubic 铺点）
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

Center3 = Tuple[int, int, int]
AxisRange = Tuple[int, int]

# 验证集确定性采样的固定基种子（与样本序号组合派生逐样本 RNG）。
VAL_SAMPLING_SEED = 0x5EED_2024


def halton(i: int, base: int) -> float:
    """Halton 低差异序列第 i 项（i>=1），返回 [0,1) 内均匀覆盖的确定性分数。"""
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class WorkerNumpyRng:
    """惰性持有逐 worker 的 ``np.random.Generator``。

    DataLoader fork 后 numpy 全局 RNG 在各 worker 间复制；以 PyTorch 逐
    worker 基种子（主进程用 ``torch.initial_seed()``）创建独立流。
    """

    __slots__ = ("_cache", "_wid")

    def __init__(self) -> None:
        self._cache: Optional[np.random.Generator] = None
        self._wid: Optional[int] = None

    def get(self) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        wid = -1 if info is None else info.id
        if self._cache is None or self._wid != wid:
            seed = torch.initial_seed() if info is None else info.seed
            self._cache = np.random.default_rng(seed % (2 ** 63))
            self._wid = wid
        return self._cache

    def reset(self) -> None:
        """pickle 到 worker 后可选清空（``VolumeCache`` 同口径）。"""
        self._cache = None
        self._wid = None


def val_sample_rng(
    is_train: bool,
    worker_rng: WorkerNumpyRng,
    sample_idx: int,
    val_seed: int = VAL_SAMPLING_SEED,
) -> np.random.Generator:
    """训练 → 逐 worker 流式 RNG；验证 → 样本序号确定性 RNG。"""
    if is_train:
        return worker_rng.get()
    return np.random.default_rng((val_seed, sample_idx))


def deterministic_idx_rng(seed: int, idx: int) -> np.random.Generator:
    """cls/det 验证：中心由 ``(seed, idx)`` 确定性派生。"""
    return np.random.default_rng(int(seed) * 1_000_003 + int(idx))


def val_coverage_j_interleaved(sample_idx: int, n_volumes: int) -> int:
    """卷交错索引（seg/det）：卷内序号 ``j = idx // n_volumes``。"""
    return sample_idx // max(n_volumes, 1)


def val_coverage_j_blocked(sample_idx: int, samples_per_volume: int) -> int:
    """同卷连续索引（cls）：卷内序号 ``j = idx % spv``。"""
    return sample_idx % max(samples_per_volume, 1)


def z_grid_center(j: int, samples_per_volume: int, D_vol: int) -> int:
    """z 轴 val 网格：第 j 个样本取 bin 中心 z。"""
    spv = max(int(samples_per_volume), 1)
    return min(int((j + 0.5) * D_vol / spv), D_vol - 1)


def halton_center(
    j: int,
    ranges: Sequence[AxisRange],
    bases: Sequence[int] = (2, 3, 5),
) -> Center3:
    """cubic val 网格：Halton 低差异序列铺满安全中心域。"""
    fracs = [halton(j + 1, b) for b in bases[: len(ranges)]]
    return tuple(
        lo + min(int(f * (hi - lo)), hi - lo - 1)
        for f, (lo, hi) in zip(fracs, ranges)
    )


def uniform_center(
    rng: np.random.Generator,
    ranges: Sequence[AxisRange],
) -> Center3:
    """安全域内均匀随机中心。"""
    return tuple(int(rng.integers(lo, hi)) for lo, hi in ranges)


def clip_center_to_ranges(
    center: Sequence[int],
    ranges: Sequence[AxisRange],
) -> Center3:
    """将坐标夹匯到各轴 ``(lo, hi-1)``。"""
    return tuple(
        int(np.clip(int(c), lo, hi - 1))
        for c, (lo, hi) in zip(center, ranges)
    )


__all__ = [
    "VAL_SAMPLING_SEED",
    "WorkerNumpyRng",
    "clip_center_to_ranges",
    "deterministic_idx_rng",
    "halton",
    "halton_center",
    "uniform_center",
    "val_coverage_j_blocked",
    "val_coverage_j_interleaved",
    "val_sample_rng",
    "z_grid_center",
]
