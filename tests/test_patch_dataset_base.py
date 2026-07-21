"""P2c：NpzPatchDatasetBase 索引与 RNG 契约。"""

from __future__ import annotations

import numpy as np
import pytest

from taskcore.data.patch_dataset_base import IndexScheme, NpzPatchDatasetBase


class _Stub(NpzPatchDatasetBase):
    def __getitem__(self, idx):
        raise NotImplementedError


def _mk(index_scheme: str, n_vols=3, spv=4) -> _Stub:
    paths = [f"/d/v{i}.npz" for i in range(n_vols)]
    return _Stub(
        paths, (8, 16, 16), "cubic", spv, 3, True, 0, False,
        -1024.0, 3071.0, "minmax", 0.0, 1.0, 0.0, False, 0,
        index_scheme=index_scheme,
    )


def test_vol_idx_blocked():
    ds = _mk(IndexScheme.BLOCKED)
    assert ds._vol_idx(0) == 0
    assert ds._vol_idx(3) == 0
    assert ds._vol_idx(4) == 1


def test_vol_idx_interleaved():
    ds = _mk(IndexScheme.INTERLEAVED)
    assert ds._vol_idx(0) == 0
    assert ds._vol_idx(1) == 1
    assert ds._vol_idx(3) == 0


def test_item_rng_and_cov_val_grid():
    ds = _mk(IndexScheme.BLOCKED, n_vols=2, spv=3)
    ds.is_train = False
    ds.val_grid_coverage = True
    rng1, cov1 = ds._item_rng_and_cov(4)
    rng2, cov2 = ds._item_rng_and_cov(4)
    assert cov1 == 1  # idx=4, spv=3 → blocked j=1
    assert rng1.integers(0, 999) == rng2.integers(0, 999)


def test_normalize_patch_mode_on_init():
    with pytest.raises(ValueError, match="bad patch_mode"):
        _Stub(["/d/a.npz"], (8, 16, 16), "bad", 1, 3, True, 0, False,
              -1024.0, 3071.0, "minmax", 0.0, 1.0, 0.0, False, 0)
