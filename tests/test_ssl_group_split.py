"""S3: 组级(患者级) train/val 划分 + 确定性/前景感知验证 patch。"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ssltask.eval.split import group_key, group_split
from ssltask.data.ssl_dataset import LabeledPatchDataset


# ---------------------------------------------------------------------------
# group_key / group_split
# ---------------------------------------------------------------------------
def test_group_key_default_is_stem():
    assert group_key("/a/b/P001_series0.npz") == "P001_series0"
    assert group_key("P002.npz") == "P002"


def test_group_key_regex_first_capture():
    rx = r"(P\d+)"
    assert group_key("/x/P001_a.npz", rx) == "P001"
    assert group_key("/x/P001_b.npz", rx) == "P001"


def test_group_split_no_leakage_across_groups():
    # 3 patients x 2 series each.
    paths = [f"/d/P{p:03d}_{s}.npz" for p in range(3) for s in range(2)]
    rx = r"(P\d+)"
    train, val = group_split(paths, val_ratio=0.34, seed=0, group_regex=rx)
    train_g = {group_key(p, rx) for p in train}
    val_g = {group_key(p, rx) for p in val}
    assert train_g and val_g
    assert train_g.isdisjoint(val_g)          # 无同患者跨侧泄漏
    # 同患者的两个序列始终同侧
    for p in range(3):
        both = {f"/d/P{p:03d}_0.npz", f"/d/P{p:03d}_1.npz"}
        assert both <= set(train) or both <= set(val)


def test_group_split_deterministic_with_seed():
    paths = [f"/d/P{p:03d}.npz" for p in range(10)]
    a = group_split(paths, 0.3, seed=7)
    b = group_split(paths, 0.3, seed=7)
    assert a == b


def test_group_split_single_group_raises_by_default():
    paths = ["/d/P001_a.npz", "/d/P001_b.npz"]
    with pytest.raises(ValueError):
        group_split(paths, 0.3, seed=0, group_regex=r"(P\d+)")


def test_group_split_single_group_allowed_reuses():
    paths = ["/d/P001_a.npz", "/d/P001_b.npz"]
    train, val = group_split(paths, 0.3, seed=0, group_regex=r"(P\d+)",
                             allow_single_group=True)
    assert set(train) == set(val) == set(paths)


def test_group_split_unmatched_regex_falls_back_to_stem(caplog):
    paths = ["/d/P001_a.npz", "/d/unrelated.npz", "/d/P002_a.npz"]
    with caplog.at_level("WARNING"):
        train, val = group_split(
            paths, val_ratio=0.34, seed=0, group_regex=r"(P\d+)")
    assert train and val
    assert set(train).isdisjoint(set(val))
    assert "/d/unrelated.npz" in set(train) | set(val)
    assert "falling back to filename stem" in caplog.text


def test_group_split_train_always_nonempty():
    # 2 groups, tiny val_ratio still yields >=1 val and >=1 train group.
    paths = ["/d/A.npz", "/d/B.npz"]
    train, val = group_split(paths, 0.01, seed=1)
    assert train and val
    assert set(train).isdisjoint(set(val))


# ---------------------------------------------------------------------------
# deterministic / fg-aware validation patches
# ---------------------------------------------------------------------------
def _write_fg_npz(path, shape=(20, 40, 40)):
    img = (np.random.rand(*shape) * 400 - 200).astype(np.int16)
    lbl = np.zeros(shape, dtype=np.int16)
    lbl[10, 20, 20] = 1                       # single fg voxel
    # fg_coords 与 make_data 口径一致：(N, 3) int，(z,y,x)
    fg = np.array([[10, 20, 20]], dtype=np.int64)
    np.savez(path, image=img, label=lbl, fg_coords=fg)


def test_deterministic_dataset_repeatable(tmp_path):
    p = tmp_path / "P001.npz"
    _write_fg_npz(p)
    ds = LabeledPatchDataset(
        [str(p)], patch_size=[8, 16, 16],
        intensity_min=-1024.0, intensity_max=1024.0, normalize="minmax",
        samples_per_volume=4, deterministic=True, seed=3)
    a = [ds[i]["image"].clone() for i in range(4)]
    b = [ds[i]["image"].clone() for i in range(4)]
    for x, y in zip(a, b):
        assert np.array_equal(x.numpy(), y.numpy())   # 跨遍历可重现


def test_fg_aware_patch_contains_foreground(tmp_path):
    p = tmp_path / "P001.npz"
    _write_fg_npz(p)
    ds = LabeledPatchDataset(
        [str(p)], patch_size=[8, 16, 16],
        intensity_min=-1024.0, intensity_max=1024.0, normalize="minmax",
        samples_per_volume=6, deterministic=True, fg_aware=True, seed=1)
    # fg-aware 应保证每个验证 patch 都覆盖那唯一前景体素
    assert all(ds[i]["label"].sum().item() >= 1 for i in range(6))
