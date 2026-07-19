"""Dataset selection strategy：复用 ``taskcore.data.specs`` 的策略骨架。

gentask 仅扩展两点：
* ``DatasetCommonCfg`` 追加条件体（cond_*）公共构造参数，且
  ``region_weights`` 取自 ``cfg.data``（生成任务的静态区域权重挂在 data 段）；
* 三个具体 spec 通过 ``dataset_cls`` 类属性接到 gentask 的
  ``Volume3D`` / ``Volume3DCubic`` / ``Volume3DWhole`` 上。

选择逻辑（patch_mode → spec）、train/val 动态参数（aug_oversample /
samples_per_volume / fg_ratio）全部继承 taskcore 实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from taskcore.data.specs import (  # noqa: F401  (re-export，保持旧 import 路径可用)
    CubicSpec as _CoreCubicSpec,
    DatasetCommonCfg as _CoreDatasetCommonCfg,
    DatasetSpec,
    SplitPaths,
    WholeSpec as _CoreWholeSpec,
    ZCubeSpec as _CoreZCubeSpec,
)

from ..config import Config
from .dataset import Volume3D, Volume3DCubic, Volume3DWhole


@dataclass(frozen=True)
class DatasetCommonCfg(_CoreDatasetCommonCfg):
    """公共构造参数（taskcore 12 字段 + gentask cond 扩展）。"""

    cond_normalize: str = "minmax"
    cond_intensity_min: float = -1024.0
    cond_intensity_max: float = 1024.0
    cond_global_mean: float = 0.0
    cond_global_std: float = 1.0

    @classmethod
    def from_cfg(cls, cfg: Config) -> "DatasetCommonCfg":
        dc = cfg.data
        return cls(
            label_values      = list(dc.label_values),
            patch_size        = tuple(int(x) for x in dc.patch_size),
            intensity_min     = float(dc.intensity_min),
            intensity_max     = float(dc.intensity_max),
            normalize         = str(dc.normalize),
            global_mean       = float(dc.global_mean),
            global_std        = float(dc.global_std),
            cache_enabled     = (str(dc.cache_mode) == "memory"),
            cache_max_volumes = int(dc.cache_max_volumes),
            cache_int16       = (str(dc.cache_dtype) == "int16"),
            region_weights    = (list(dc.region_weights)
                                 if dc.region_weights else None),
            cond_normalize    = str(dc.cond_normalize),
            cond_intensity_min = float(dc.cond_intensity_min),
            cond_intensity_max = float(dc.cond_intensity_max),
            cond_global_mean  = float(dc.cond_global_mean),
            cond_global_std   = float(dc.cond_global_std))


class WholeSpec(_CoreWholeSpec):
    dataset_cls = Volume3DWhole


class ZCubeSpec(_CoreZCubeSpec):
    dataset_cls = Volume3D


class CubicSpec(_CoreCubicSpec):
    dataset_cls = Volume3DCubic


def build_data_spec(cfg: Config) -> DatasetSpec:
    """``cfg.data.patch_mode`` → ``DatasetSpec``。新增 patch_mode 仅需在此追加一行。"""
    pm = str(cfg.data.patch_mode).lower()
    if pm in ("2_5d", "z_axis"):
        return ZCubeSpec(cfg)
    if pm == "whole":
        return WholeSpec(cfg)
    if pm == "cubic":
        return CubicSpec(cfg)
    raise ValueError(
        f"Unknown patch_mode: {pm!r}. Valid: 'z_axis' | '2_5d' | 'whole' | 'cubic'.")


__all__ = [
    "DatasetCommonCfg",
    "SplitPaths",
    "DatasetSpec",
    "WholeSpec",
    "ZCubeSpec",
    "CubicSpec",
    "build_data_spec"]
