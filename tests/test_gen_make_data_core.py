"""R3: gen make_data 委托 core — 契约回归。"""

from __future__ import annotations

import inspect

import numpy as np

from taskcore.data import make_data as core_md
from gentask.data import make_data as gen_md


def test_gen_prepare_one_is_core_prepare_one():
    """gen 不再维护独立 prepare_one 实现。"""
    assert gen_md.prepare_one is core_md.prepare_one


def test_core_prepare_one_accepts_cond_paths():
    sig = inspect.signature(core_md.prepare_one)
    assert "cond_paths" in sig.parameters
    assert "spacing_normalization" in sig.parameters
    assert "target_spacing" in sig.parameters


def test_compute_fg_indices_per_class():
    """逐类 fg 索引：两类前景各自有 coords，并带 cls 对齐数组。"""
    label = np.zeros((4, 8, 8), dtype=np.int16)
    label[1, 2:4, 2:4] = 1
    label[2, 4:6, 4:6] = 2
    (fg_slices, fg_coords, fg_coords_cls,
     fg_slices_cls_z, fg_slices_cls) = core_md._compute_fg_indices(
        label, [0, 1, 2], fg_subsample=50_000)
    assert fg_slices.size >= 2
    assert set(fg_coords_cls.tolist()) == {1, 2}
    assert len(fg_coords) == len(fg_coords_cls)
    assert set(fg_slices_cls.tolist()) == {1, 2}
    assert len(fg_slices_cls_z) == len(fg_slices_cls)


def test_npz_meta_skip_requires_fg_per_class_keys():
    ok, reason = core_md._npz_meta_allows_skip(
        {"label_counts": {0: 1}, "image_shape": [1, 2, 3]},
        spacing_normalization=False,
        target_spacing=None)
    assert not ok and "fg_per_class" in reason

    # 缺 intensity_stats（make_data<1.9 陈旧包）不得静默 skip（2-3）。
    ok, reason = core_md._npz_meta_allows_skip(
        {"label_counts": {0: 1}, "image_shape": [1, 2, 3],
         "fg_per_class": True, "spacing_normalized": False},
        spacing_normalization=False,
        target_spacing=None)
    assert not ok and "intensity_stats" in reason

    ok, reason = core_md._npz_meta_allows_skip(
        {"label_counts": {0: 1}, "image_shape": [1, 2, 3],
         "fg_per_class": True, "spacing_normalized": False,
         "intensity_stats": {"mean": 0.0}},
        spacing_normalization=False,
        target_spacing=None)
    assert ok, reason


def test_load_npz_spacing_falls_back_to_orig_spacing(tmp_path):
    from gentask.data.dataset.io import load_npz_spacing

    path = tmp_path / "s.npz"
    meta = np.array(
        {"orig_spacing": [1.0, 0.5, 0.5], "pid": "x"}, dtype=object)
    np.savez(path, meta=meta)
    sp = load_npz_spacing(str(path))
    assert sp == (1.0, 0.5, 0.5)
