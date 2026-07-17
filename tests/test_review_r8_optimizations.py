# -*- coding: utf-8 -*-
"""R8 优化回归测试：AdaBN 估计期窗口抽样 + npz memmap 零拷贝快路径。

覆盖：
* predict.adabn_sample_ratio 默认值 / 校验 / 抽样判据语义；
* 抽样只作用于 AdaBN 估计期（真实预测路径恒全窗）；
* _open_npy_member_mmap 对未压缩 npz 生效、压缩 npz 回退 None；
* load_npz_image / load_npz_label / load_npz_region_weight 在 memmap 快
  路径与 zipfile 回退路径下逐位一致，且返回 owned 可写数组。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taskcore.config.core import Config
from taskcore.data.dataset import (
    _open_npy_member_mmap,
    load_npz_image,
    load_npz_label,
    load_npz_region_weight,
)
from segtask_v1.predictor.predictor import Predictor


# ---------------------------------------------------------------------------
# AdaBN 估计期窗口抽样
# ---------------------------------------------------------------------------
class _Stub:
    """仅承载 _adabn_keep_window 所需属性的桩对象。"""

    def __init__(self, estimating: bool, ratio: float):
        self._adabn_estimating = estimating
        self.adabn_sample_ratio = ratio


def _keep(stub, idx):
    return Predictor._adabn_keep_window(stub, idx)


def test_adabn_keep_window_noop_outside_estimation():
    stub = _Stub(estimating=False, ratio=0.25)
    assert all(_keep(stub, i) for i in range(32))


def test_adabn_keep_window_full_ratio_keeps_all():
    stub = _Stub(estimating=True, ratio=1.0)
    assert all(_keep(stub, i) for i in range(32))


def test_adabn_keep_window_subsamples_deterministically():
    stub = _Stub(estimating=True, ratio=0.25)
    kept = [i for i in range(32) if _keep(stub, i)]
    assert kept == list(range(0, 32, 4))
    # 首窗恒保留（保证至少一窗驱动 BN 更新）。
    assert _keep(stub, 0)


def test_adabn_sample_ratio_default_and_validation():
    cfg = Config()
    assert cfg.predict.adabn_sample_ratio == 1.0

    cfg = Config()
    cfg.predict.adabn_enabled = True
    cfg.predict.adabn_sample_ratio = 0.25
    cfg.validate()  # 合法值不报错

    for bad in (0.0, -0.5, 1.5):
        cfg = Config()
        cfg.predict.adabn_enabled = True
        cfg.predict.adabn_sample_ratio = bad
        with pytest.raises(ValueError):
            cfg.validate()


def test_adabn_sample_ratio_ignored_when_disabled():
    # adabn_enabled=False 时不校验该字段（与现有 adabn 校验分支一致）。
    cfg = Config()
    cfg.predict.adabn_sample_ratio = 1.5
    cfg.validate()


# ---------------------------------------------------------------------------
# npz memmap 零拷贝快路径
# ---------------------------------------------------------------------------
@pytest.fixture()
def npz_pair(tmp_path):
    """同一份数据的未压缩 / 压缩 npz 各一份。"""
    rng = np.random.default_rng(0)
    image = rng.integers(-1024, 2048, size=(9, 17, 13)).astype(np.int16)
    label = rng.integers(0, 3, size=(9, 17, 13)).astype(np.int16)
    rw = rng.random((9, 17, 13)).astype(np.float32) + 1.0
    stored = tmp_path / "stored.npz"
    deflated = tmp_path / "deflated.npz"
    np.savez(stored, image=image, label=label, rw=rw)
    np.savez_compressed(deflated, image=image, label=label, rw=rw)
    return stored, deflated, image, label, rw


def test_member_mmap_stored_vs_deflated(npz_pair):
    stored, deflated, image, label, _ = npz_pair
    mm = _open_npy_member_mmap(str(stored), "image")
    assert mm is not None
    assert mm.dtype == np.int16 and mm.shape == image.shape
    assert np.array_equal(np.asarray(mm), image)
    # 压缩成员无法 memmap → 回退 None。
    assert _open_npy_member_mmap(str(deflated), "image") is None
    # 不存在的成员 / 文件同样安全返 None。
    assert _open_npy_member_mmap(str(stored), "nope") is None
    assert _open_npy_member_mmap(str(stored) + ".missing", "image") is None


def test_load_npz_label_identical_and_writable(npz_pair):
    stored, deflated, _, label, _ = npz_pair
    out_fast = load_npz_label(str(stored))
    out_slow = load_npz_label(str(deflated))
    assert np.array_equal(out_fast, label)
    assert np.array_equal(out_slow, label)
    assert out_fast.dtype == np.int16
    # owned 可写（下游可能就地改写）。
    assert out_fast.flags.writeable and out_fast.flags.owndata
    out_fast[0, 0, 0] = 7  # 不应抛（read-only memmap 泄漏会在此失败）


def test_load_npz_image_identical(npz_pair):
    stored, deflated, _, _, _ = npz_pair
    kw = dict(intensity_min=-200.0, intensity_max=400.0,
              normalize="minmax")
    out_fast = load_npz_image(str(stored), **kw)
    out_slow = load_npz_image(str(deflated), **kw)
    assert out_fast.dtype == np.float32
    assert np.array_equal(out_fast, out_slow)
    assert out_fast.flags.writeable


def test_load_npz_region_weight_identical(npz_pair):
    stored, deflated, _, _, rw = npz_pair
    out_fast = load_npz_region_weight(str(stored))
    out_slow = load_npz_region_weight(str(deflated))
    assert np.array_equal(out_fast, rw)
    assert np.array_equal(out_slow, rw)
    assert out_fast.flags.writeable and out_fast.flags.owndata
