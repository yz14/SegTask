"""P2c：patch_extract —— 四 patch_mode 抽取口径。"""

from __future__ import annotations

import numpy as np
import pytest

from taskcore.data.dataset import extract_z_patch_padded, resize_3d
from taskcore.data.patch_extract import (
    extract_patch_by_mode,
    normalize_patch_mode,
)
from taskcore.data.patch_ops import extract_cubic_patch
from taskcore.data.sampling import deterministic_idx_rng


def test_normalize_patch_mode_rejects_unknown():
    with pytest.raises(ValueError, match="bad patch_mode"):
        normalize_patch_mode("sliding")


def test_extract_whole_resize():
    vol = np.random.randn(10, 20, 30).astype(np.float32)
    out = extract_patch_by_mode(vol, "whole", (0, 0, 0), (8, 16, 16))
    assert out.shape == (8, 16, 16)


def test_extract_cubic_matches_direct():
    vol = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
    center = (3, 4, 4)
    patch = (4, 4, 4)
    assert np.array_equal(
        extract_patch_by_mode(vol, "cubic", center, patch),
        extract_cubic_patch(vol, center, patch),
    )


def test_extract_z_axis_matches_legacy():
    vol = np.random.randn(12, 24, 24).astype(np.float32)
    z, patch = 5, (6, 16, 16)
    legacy = resize_3d(extract_z_patch_padded(vol, z, patch[0]), *patch)
    out = extract_patch_by_mode(vol, "z_axis", (z, 0, 0), patch)
    assert np.allclose(legacy, out)


def test_extract_label_uses_nearest():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[5, :, :] = 3.0
    z = 5
    out = extract_patch_by_mode(
        vol, "z_axis", (z, 0, 0), (4, 8, 8), is_label=True)
    assert out.shape == (4, 8, 8)
    assert np.all((out == 0.0) | (out == 3.0))


def test_deterministic_idx_rng_stable():
    r1 = deterministic_idx_rng(42, 7)
    r2 = deterministic_idx_rng(42, 7)
    assert r1.integers(0, 999) == r2.integers(0, 999)
