"""Unit tests for ``segtask_v1.data.specs``.

CPU-only, 不读盘、不构造真 dataset（用 monkeypatch 拦截 dataset class）。覆盖：

1. ``build_data_spec`` 选对子类（4 种 patch_mode 全覆盖）
2. ``DatasetCommonCfg.from_cfg`` 字段 round-trip
3. ``DatasetSpec.make_split`` 转发的 kwargs 等价于"重构前 ``loader.py`` 直接传"
4. train/val 两次调用的 split-dependent kwargs（aug_oversample / fg_ratio /
   samples_per_volume）按 is_train 切换
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from taskcore.config.core import Config
from segtask_v1.data import specs as specs_mod
from taskcore.data.specs import (
    CubicSpec, DatasetCommonCfg, SplitPaths, WholeSpec, ZCubeSpec,
    build_data_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_cfg(**overrides):
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.batch_size = 1
    cfg.data.aug_oversample_ratio = 1.5
    cfg.data.foreground_oversample_ratio = 0.4
    cfg.data.samples_per_volume = 8
    cfg.data.cache_mode = "memory"
    cfg.data.cache_max_volumes = 4
    cfg.data.intensity_min = -1024.0
    cfg.data.intensity_max = 3071.0
    cfg.data.normalize = "minmax"
    cfg.data.global_mean = 0.0
    cfg.data.global_std = 1.0
    cfg.loss.region_weights = [1.0, 2.0, 3.0]
    for k, v in overrides.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return cfg


def _paths(n=3):
    return SplitPaths(
        image_paths=[f"/fake/img_{i}.npz" for i in range(n)],
        label_paths=[f"/fake/lbl_{i}.npz" for i in range(n)],
        npz_paths=[f"/fake/{i}.npz" for i in range(n)],
    )


# ===========================================================================
# Factory dispatch
# ===========================================================================
class TestFactoryDispatch:
    @pytest.mark.parametrize("pm,expected", [
        ("z_axis", ZCubeSpec),
        ("2_5d", ZCubeSpec),
        ("whole", WholeSpec),
        ("cubic", CubicSpec),
    ])
    def test_dispatch(self, pm, expected):
        cfg = _base_cfg()
        cfg.data.patch_mode = pm
        # whole 模式下 multi_res_scales 必须 [1.0]；其他保留默认。
        if pm == "whole":
            cfg.data.multi_res_scales = [1.0]
        spec = build_data_spec(cfg)
        assert isinstance(spec, expected)
        assert spec.cfg is cfg

    def test_unknown_patch_mode(self):
        cfg = _base_cfg()
        cfg.data.patch_mode = "bogus"
        with pytest.raises(ValueError, match="Unknown patch_mode"):
            build_data_spec(cfg)


# ===========================================================================
# Common config snapshot
# ===========================================================================
class TestDatasetCommonCfg:
    def test_from_cfg_roundtrip(self):
        cfg = _base_cfg()
        common = DatasetCommonCfg.from_cfg(cfg)
        assert common.label_values == [0, 1, 2]
        assert common.patch_size == (4, 16, 16)
        assert common.intensity_min == -1024.0
        assert common.intensity_max == 3071.0
        assert common.normalize == "minmax"
        assert common.cache_enabled is True       # cache_mode='memory'
        assert common.cache_max_volumes == 4
        assert common.region_weights == [1.0, 2.0, 3.0]

    def test_cache_disabled(self):
        cfg = _base_cfg()
        cfg.data.cache_mode = "none"
        common = DatasetCommonCfg.from_cfg(cfg)
        assert common.cache_enabled is False

    def test_no_region_weights(self):
        cfg = _base_cfg()
        cfg.loss.region_weights = []
        common = DatasetCommonCfg.from_cfg(cfg)
        assert common.region_weights is None

    def test_to_kwargs_keys_match_dataset_init(self):
        """``to_kwargs`` 必须覆盖 ``SegDatasetNpzBase.__init__`` 的 11 个公共参数。"""
        common = DatasetCommonCfg.from_cfg(_base_cfg())
        expected = {
            "label_values", "patch_size",
            "intensity_min", "intensity_max", "normalize",
            "global_mean", "global_std",
            "cache_enabled", "cache_max_volumes", "region_weights",
        }
        assert set(common.to_kwargs().keys()) == expected


# ===========================================================================
# Spec → dataset construction (via mock)
# ===========================================================================
class TestSpecMakeSplit:
    """拦截 dataset 类构造器，断言 spec 转发的 kwargs 与历史等价。"""

    @pytest.fixture
    def captured(self, monkeypatch):
        """Replace 3 dataset classes with MagicMock; return captured kwargs."""
        kw_box: dict = {}

        def _factory(name):
            def _ctor(**kwargs):
                kw_box["cls"] = name
                kw_box["kwargs"] = kwargs
                return MagicMock(name=f"{name}_instance")
            return _ctor

        # dataset 类经各 spec 的 dataset_cls 类属性解析（子项目可覆盖）。
        monkeypatch.setattr(ZCubeSpec, "dataset_cls", _factory("SegDataset3D"))
        monkeypatch.setattr(CubicSpec, "dataset_cls", _factory("SegDataset3DCubic"))
        monkeypatch.setattr(WholeSpec, "dataset_cls", _factory("SegDataset3DWhole"))
        return kw_box

    def test_zcube_train(self, captured):
        cfg = _base_cfg()
        cfg.data.patch_mode = "z_axis"
        cfg.data.multi_res_scales = [1.0, 2.0]
        cfg.data.z_boundary_mode = "edge_pad"
        spec = build_data_spec(cfg)
        common = DatasetCommonCfg.from_cfg(cfg)
        spec.make_split(_paths(), is_train=True, common=common)

        assert captured["cls"] == "SegDataset3D"
        kw = captured["kwargs"]
        # split-dependent
        assert kw["is_train"] is True
        assert kw["aug_oversample_ratio"] == 1.5
        assert kw["foreground_oversample_ratio"] == 0.4
        assert kw["samples_per_volume"] == 8
        # mode-specific
        assert kw["multi_res_scales"] == [1.0, 2.0]
        assert kw["z_boundary_mode"] == "edge_pad"
        # common
        assert kw["patch_size"] == (4, 16, 16)
        assert kw["cache_enabled"] is True
        # paths
        assert len(kw["npz_paths"]) == 3

    def test_zcube_val_overrides(self, captured):
        cfg = _base_cfg()
        cfg.data.patch_mode = "2_5d"
        spec = build_data_spec(cfg)
        common = DatasetCommonCfg.from_cfg(cfg)
        spec.make_split(_paths(), is_train=False, common=common)

        kw = captured["kwargs"]
        assert kw["is_train"] is False
        assert kw["aug_oversample_ratio"] == 1.0   # val 始终 1.0
        assert kw["foreground_oversample_ratio"] == 0.0  # val 不 fg-oversample
        assert kw["samples_per_volume"] == 4       # 8 // 2

    def test_whole_no_multi_res_no_z_boundary(self, captured):
        cfg = _base_cfg()
        cfg.data.patch_mode = "whole"
        cfg.data.multi_res_scales = [1.0]
        spec = build_data_spec(cfg)
        common = DatasetCommonCfg.from_cfg(cfg)
        spec.make_split(_paths(), is_train=True, common=common)

        assert captured["cls"] == "SegDataset3DWhole"
        kw = captured["kwargs"]
        # whole 不传 multi_res_scales / fg_oversample / z_boundary_mode
        assert "multi_res_scales" not in kw
        assert "foreground_oversample_ratio" not in kw
        assert "z_boundary_mode" not in kw
        assert kw["is_train"] is True
        assert kw["aug_oversample_ratio"] == 1.5

    def test_cubic_train_passes_multi_res(self, captured):
        cfg = _base_cfg()
        cfg.data.patch_mode = "cubic"
        cfg.data.multi_res_scales = [1.0, 1.5]
        spec = build_data_spec(cfg)
        common = DatasetCommonCfg.from_cfg(cfg)
        spec.make_split(_paths(), is_train=True, common=common)

        assert captured["cls"] == "SegDataset3DCubic"
        kw = captured["kwargs"]
        assert kw["multi_res_scales"] == [1.0, 1.5]
        assert kw["foreground_oversample_ratio"] == 0.4
        # cubic 不传 z_boundary_mode（与 whole 一致）
        assert "z_boundary_mode" not in kw


# ===========================================================================
# Aug oversample never < 1.0
# ===========================================================================
class TestSplitDependentKwargs:
    def test_aug_oversample_floor(self):
        cfg = _base_cfg()
        cfg.data.aug_oversample_ratio = 0.5  # 用户设错（<1）
        cfg.data.patch_mode = "whole"
        cfg.data.multi_res_scales = [1.0]
        spec = build_data_spec(cfg)
        # train 强制下限 1.0，val 仍 1.0
        assert spec._aug_oversample(is_train=True) == 1.0
        assert spec._aug_oversample(is_train=False) == 1.0

    def test_samples_per_volume_val_min_1(self):
        cfg = _base_cfg()
        cfg.data.samples_per_volume = 1
        cfg.data.patch_mode = "whole"
        cfg.data.multi_res_scales = [1.0]
        spec = build_data_spec(cfg)
        assert spec._samples_per_volume(is_train=False) == 1   # max(1//2, 1)
