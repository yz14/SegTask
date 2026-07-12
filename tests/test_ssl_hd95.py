"""S2 回归：spacing-aware 双向 HD95 + 空集显式处理/计数。

参考值来自手工可验的几何构型（与 MedPy `hd95` 口径一致：双向 surface 距离
合并后取 95 分位）。
"""

from __future__ import annotations

import numpy as np
import pytest

from ssltask.data.ssl_dataset import read_npz_spacing
from ssltask.eval.metrics import hd95, hd95_batch


def _cube(shape, sl):
    m = np.zeros(shape, dtype=np.uint8)
    m[sl] = 1
    return m


# ---------------------------------------------------------------------------
# 基础/等价性
# ---------------------------------------------------------------------------
def test_identical_and_empty_conventions():
    a = _cube((16, 16), (slice(4, 8), slice(4, 8)))
    empty = np.zeros((16, 16), dtype=np.uint8)
    assert hd95(a, a) == 0.0
    assert hd95(empty, empty) == 0.0
    assert np.isnan(hd95(a, empty))
    assert np.isnan(hd95(empty, a))


def test_unit_spacing_matches_default():
    a = _cube((20, 20, 20), (slice(4, 9), slice(4, 9), slice(4, 9)))
    b = _cube((20, 20, 20), (slice(6, 11), slice(4, 9), slice(4, 9)))
    assert hd95(a, b, spacing=(1.0, 1.0, 1.0)) == pytest.approx(hd95(a, b))


# ---------------------------------------------------------------------------
# spacing 感知（各向异性）
# ---------------------------------------------------------------------------
def test_anisotropic_spacing_scales_axis0_shift():
    """沿轴 0 平移 2 voxel 的两个平板：HD95 = 2 * spacing[0]。"""
    a = _cube((12, 8, 8), (slice(2, 4), slice(0, 8), slice(0, 8)))
    b = _cube((12, 8, 8), (slice(4, 6), slice(0, 8), slice(0, 8)))
    assert hd95(a, b) == pytest.approx(2.0)
    assert hd95(a, b, spacing=(2.5, 1.0, 1.0)) == pytest.approx(5.0)
    # 平移轴以外的 spacing 不影响该构型的距离
    assert hd95(a, b, spacing=(2.5, 7.0, 3.0)) == pytest.approx(5.0)


def test_spacing_validation():
    a = _cube((8, 8), (slice(2, 4), slice(2, 4)))
    b = _cube((8, 8), (slice(3, 5), slice(2, 4)))
    with pytest.raises(ValueError):
        hd95(a, b, spacing=(1.0,))          # 长度不匹配
    with pytest.raises(ValueError):
        hd95(a, b, spacing=(1.0, -1.0))     # 非正
    with pytest.raises(ValueError):
        hd95(a, b, spacing=(1.0, float("nan")))


# ---------------------------------------------------------------------------
# 批量 + 空集计数
# ---------------------------------------------------------------------------
def test_hd95_batch_counts_empty_cases():
    a = _cube((10, 10), (slice(2, 5), slice(2, 5)))
    b = _cube((10, 10), (slice(3, 6), slice(2, 5)))
    empty = np.zeros((10, 10), dtype=np.uint8)
    pred = np.stack([a, empty, empty, a])[:, None]     # (4,1,10,10)
    tgt = np.stack([b, a, empty, empty])[:, None]
    res = hd95_batch(pred, tgt)
    assert res["n_cases"] == 4
    assert res["n_finite"] == 2            # 样本0（正常）+ 样本2（双空=0）
    assert res["n_both_empty"] == 1
    assert res["n_pred_empty_only"] == 1
    assert res["n_target_empty_only"] == 1
    # 均值只含 finite：(hd95(a,b) + 0) / 2
    assert res["hd95"] == pytest.approx(hd95(a, b) / 2.0)


def test_hd95_batch_all_one_side_empty_returns_nan():
    a = _cube((10, 10), (slice(2, 5), slice(2, 5)))
    empty = np.zeros((10, 10), dtype=np.uint8)
    res = hd95_batch(a[None, None], empty[None, None])
    assert np.isnan(res["hd95"])           # 不隐式返 0 美化
    assert res["n_finite"] == 0
    assert res["n_pred_empty_only"] == 0
    assert res["n_target_empty_only"] == 1


def test_hd95_batch_per_sample_spacing():
    a = _cube((12, 8, 8), (slice(2, 4), slice(0, 8), slice(0, 8)))
    b = _cube((12, 8, 8), (slice(4, 6), slice(0, 8), slice(0, 8)))
    pred = np.stack([a, a])[:, None]
    tgt = np.stack([b, b])[:, None]
    sp = np.asarray([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    res = hd95_batch(pred, tgt, spacing=sp)
    # 样本0: 2mm；样本1: 6mm → 均值 4mm
    assert res["hd95"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        hd95_batch(pred, tgt, spacing=np.ones((3, 3)))  # B 不匹配


# ---------------------------------------------------------------------------
# npz meta spacing 读取
# ---------------------------------------------------------------------------
def _write_npz(path, meta):
    payload = {"image": np.zeros((4, 4, 4), dtype=np.int16)}
    if meta is not None:
        payload["meta"] = np.array(meta, dtype=object)
    np.savez(path, **payload)


def test_read_npz_spacing(tmp_path):
    p1 = str(tmp_path / "a.npz")
    _write_npz(p1, {"spacing_normalized": True,
                    "orig_spacing": [5.0, 0.7, 0.7],
                    "target_spacing": [1.0, 1.0, 1.0]})
    assert read_npz_spacing(p1) == (1.0, 1.0, 1.0)

    p2 = str(tmp_path / "b.npz")
    _write_npz(p2, {"spacing_normalized": False,
                    "orig_spacing": [5.0, 0.7, 0.7],
                    "target_spacing": None})
    assert read_npz_spacing(p2) == (5.0, 0.7, 0.7)

    p3 = str(tmp_path / "c.npz")
    _write_npz(p3, None)                       # 无 meta
    assert read_npz_spacing(p3) is None

    p4 = str(tmp_path / "d.npz")
    _write_npz(p4, {"spacing_normalized": False, "orig_spacing": None})
    assert read_npz_spacing(p4) is None
