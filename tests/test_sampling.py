"""P2c：sampling 单点维护 —— worker RNG 与 val 网格采样。"""

from __future__ import annotations

import numpy as np

from taskcore.data.dataset import _halton
from taskcore.data.sampling import (
    VAL_SAMPLING_SEED,
    WorkerNumpyRng,
    halton,
    halton_center,
    val_coverage_j_blocked,
    val_coverage_j_interleaved,
    val_sample_rng,
    z_grid_center,
)


def test_halton_alias_matches_dataset():
    assert halton(5, 3) == _halton(5, 3)


def test_z_grid_center_matches_legacy_formula():
    D, spv, j = 100, 8, 3
    assert z_grid_center(j, spv, D) == min(int((j + 0.5) * D / spv), D - 1)


def test_halton_center_matches_manual():
    ranges = ((4, 20), (8, 40), (8, 40))
    j = 2
    manual = tuple(
        lo + min(int(_halton(j + 1, b) * (hi - lo)), hi - lo - 1)
        for b, (lo, hi) in zip((2, 3, 5), ranges))
    assert halton_center(j, ranges) == manual


def test_val_coverage_indexing_schemes():
    n, spv = 5, 4
    # seg/det interleaved: idx=7 → vol 2, j=1
    assert val_coverage_j_interleaved(7, n) == 1
    # cls blocked: idx=7 → vol 1, j=3
    assert val_coverage_j_blocked(7, spv) == 3


def test_val_sample_rng_deterministic_on_val():
    wr = WorkerNumpyRng()
    r1 = val_sample_rng(False, wr, 42)
    r2 = val_sample_rng(False, wr, 42)
    assert r1.integers(0, 1000) == r2.integers(0, 1000)
    r3 = val_sample_rng(False, wr, 43)
    assert r1.integers(0, 1000) != r3.integers(0, 1000)


def test_worker_numpy_rng_stable_within_worker():
    wr = WorkerNumpyRng()
    a = wr.get().integers(0, 1_000_000)
    b = wr.get().integers(0, 1_000_000)
    assert a != b  # stream advances
