"""P2c：patch_ops 单点维护 —— cls/det 与 taskcore 等价性。"""

from __future__ import annotations

import numpy as np

from taskcore.data.dataset import _extract_cubic_patch
from taskcore.data.patch_ops import (
    extract_cubic_patch,
    extract_cubic_patch_with_origin,
    safe_center_range,
)


def test_extract_cubic_patch_alias_matches_patch_ops():
    vol = np.arange(10 * 12 * 14, dtype=np.float32).reshape(10, 12, 14)
    center = (3, 6, 7)
    size = (8, 8, 8)
    assert np.array_equal(
        _extract_cubic_patch(vol, center, size),
        extract_cubic_patch(vol, center, size),
    )


def test_extract_cubic_patch_no_pad_copies_off_cache_view():
    """无 padding 时必须 copy，断开与源卷别名（cls 旧实现曾缺此步）。"""
    vol = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    center = (2, 2, 2)
    patch = (4, 4, 4)
    out = extract_cubic_patch(vol, center, patch)
    out[0, 0, 0] = -1.0
    assert vol[0, 0, 0] != -1.0


def test_extract_cubic_patch_with_origin_lo_can_be_negative():
    vol = np.ones((10, 10, 10), dtype=np.float32)
    patch, lo = extract_cubic_patch_with_origin(vol, (1, 1, 1), (6, 6, 6))
    assert patch.shape == (6, 6, 6)
    assert lo == (-2, -2, -2)


def test_safe_center_range_small_axis_degenerates_to_center():
    ranges = safe_center_range((5, 20, 20), (32, 8, 8))
    assert ranges[0] == (2, 3)  # D too small → mid
    assert ranges[1] == (4, 16)   # hi = 20 - (8 - 4)
    assert ranges[2] == (4, 16)


def test_cls_det_share_safe_center_range_with_seg_cubic():
    """cls/det 直接 patch_size；seg cubic 用 max_scale 后的 extract_size。"""
    shape = (48, 96, 96)
    patch = (32, 64, 64)
    direct = safe_center_range(shape, patch)
    # seg 侧 sD,sH,sW == patch when max_scale=1
    from taskcore.data.dataset import SegDataset3DCubic

    class _Stub(SegDataset3DCubic):
        def __init__(self):
            self.extract_size = patch
            self._max_scale = 1.0

    seg_ranges = _Stub()._safe_center_range(*shape)
    assert direct == seg_ranges
