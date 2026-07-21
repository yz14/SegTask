"""Dataset classes and patch extractors for gentask.data.dataset."""

from __future__ import annotations

from typing import List, Optional, Tuple

from taskcore.data.dataset import (
    SegDataset3D,
    SegDataset3DCubic,
    SegDataset3DWhole,
    SegDatasetNpzBase,
)

from .cond_mixin import CondVolumeMixin


class VolumeNpzDatasetBase(CondVolumeMixin, SegDatasetNpzBase):
    """生成任务 npz 基类：``SegDatasetNpzBase`` + cond 加载 mixin。"""

    def __init__(
        self,
        image_paths         : List[str],
        label_paths         : List[str],
        label_values        : List[int],
        npz_paths           : List[str],
        patch_size          : Tuple[int, int, int],
        aug_oversample_ratio: float,
        intensity_min       : float,
        intensity_max       : float,
        normalize           : str,
        global_mean         : float,
        global_std          : float,
        samples_per_volume  : int,
        is_train            : bool,
        cache_enabled       : bool,
        cache_max_volumes   : int,
        cache_int16         : bool,
        region_weights      : Optional[List[float]],
        cond_normalize      : str,
        cond_intensity_min  : float,
        cond_intensity_max  : float,
        cond_global_mean    : float,
        cond_global_std     : float,
        val_grid_coverage   : bool = False):
        super().__init__(
            image_paths          = image_paths,
            label_paths          = label_paths,
            label_values         = label_values,
            npz_paths            = npz_paths,
            patch_size           = patch_size,
            aug_oversample_ratio = aug_oversample_ratio,
            intensity_min        = intensity_min,
            intensity_max        = intensity_max,
            normalize            = normalize,
            global_mean          = global_mean,
            global_std           = global_std,
            samples_per_volume   = samples_per_volume,
            is_train             = is_train,
            cache_enabled        = cache_enabled,
            cache_max_volumes    = cache_max_volumes,
            cache_int16          = cache_int16,
            region_weights       = region_weights,
            val_grid_coverage    = val_grid_coverage,
        )
        self._init_cond_fields(
            cond_normalize     = cond_normalize,
            cond_intensity_min = cond_intensity_min,
            cond_intensity_max = cond_intensity_max,
            cond_global_mean   = cond_global_mean,
            cond_global_std    = cond_global_std,
            cache_enabled      = cache_enabled,
            cache_max_volumes  = cache_max_volumes,
        )


def _init_cond_from_dataset(
    ds: CondVolumeMixin,
    *,
    cond_normalize      : str,
    cond_intensity_min  : float,
    cond_intensity_max  : float,
    cond_global_mean    : float,
    cond_global_std     : float,
    cache_enabled       : bool,
    cache_max_volumes   : int,
) -> None:
    ds._init_cond_fields(
        cond_normalize     = cond_normalize,
        cond_intensity_min = cond_intensity_min,
        cond_intensity_max = cond_intensity_max,
        cond_global_mean   = cond_global_mean,
        cond_global_std    = cond_global_std,
        cache_enabled      = cache_enabled,
        cache_max_volumes  = cache_max_volumes,
    )


class Volume3D(CondVolumeMixin, SegDataset3D):
    """3D z 轴滑窗 dataset（继承 seg 实现 + cond hook）。"""

    def __init__(
        self,
        image_paths                : List[str],
        label_paths                : List[str],
        label_values               : List[int],
        patch_size                 : Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio       : float = 1.0,
        multi_res_scales           : Optional[List[float]] = None,
        intensity_min              : float = -1024.0,
        intensity_max              : float = 3071.0,
        normalize                  : str = "minmax",
        global_mean                : float = 0.0,
        global_std                 : float = 1.0,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume         : int = 8,
        is_train                   : bool = True,
        cache_enabled              : bool = True,
        cache_max_volumes          : int = 0,
        cache_int16                : bool = False,
        region_weights             : Optional[List[float]] = None,
        cond_normalize             : str = "minmax",
        cond_intensity_min         : float = -1024.0,
        cond_intensity_max         : float = 1024.0,
        cond_global_mean           : float = 0.0,
        cond_global_std            : float = 1.0,
        z_boundary_mode            : str = "stretch",
        npz_paths                  : Optional[List[str]] = None,
        val_grid_coverage          : bool = False):
        super().__init__(
            image_paths                = image_paths,
            label_paths                = label_paths,
            label_values               = label_values,
            patch_size                 = patch_size,
            aug_oversample_ratio       = aug_oversample_ratio,
            multi_res_scales           = multi_res_scales,
            intensity_min              = intensity_min,
            intensity_max              = intensity_max,
            normalize                  = normalize,
            global_mean                = global_mean,
            global_std                 = global_std,
            foreground_oversample_ratio= foreground_oversample_ratio,
            samples_per_volume         = samples_per_volume,
            is_train                   = is_train,
            cache_enabled              = cache_enabled,
            cache_max_volumes          = cache_max_volumes,
            cache_int16                = cache_int16,
            region_weights             = region_weights,
            z_boundary_mode            = z_boundary_mode,
            npz_paths                  = npz_paths,
            val_grid_coverage          = val_grid_coverage,
        )
        _init_cond_from_dataset(
            self,
            cond_normalize     = cond_normalize,
            cond_intensity_min = cond_intensity_min,
            cond_intensity_max = cond_intensity_max,
            cond_global_mean   = cond_global_mean,
            cond_global_std    = cond_global_std,
            cache_enabled      = cache_enabled,
            cache_max_volumes  = cache_max_volumes,
        )


class Volume3DCubic(CondVolumeMixin, SegDataset3DCubic):
    """3D cubic patch dataset（继承 seg 实现 + cond hook）。"""

    def __init__(
        self,
        image_paths                : List[str],
        label_paths                : List[str],
        label_values               : List[int],
        patch_size                 : Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio       : float = 1.0,
        multi_res_scales           : Optional[List[float]] = None,
        intensity_min              : float = -1024.0,
        intensity_max              : float = 3071.0,
        normalize                  : str = "minmax",
        global_mean                : float = 0.0,
        global_std                 : float = 1.0,
        foreground_oversample_ratio: float = 0.5,
        samples_per_volume         : int = 8,
        is_train                   : bool = True,
        cache_enabled              : bool = True,
        cache_max_volumes          : int = 0,
        cache_int16                : bool = False,
        region_weights             : Optional[List[float]] = None,
        cond_normalize             : str = "minmax",
        cond_intensity_min         : float = -1024.0,
        cond_intensity_max         : float = 1024.0,
        cond_global_mean           : float = 0.0,
        cond_global_std            : float = 1.0,
        npz_paths                  : Optional[List[str]] = None,
        val_grid_coverage          : bool = False):
        super().__init__(
            image_paths                = image_paths,
            label_paths                = label_paths,
            label_values               = label_values,
            patch_size                 = patch_size,
            aug_oversample_ratio       = aug_oversample_ratio,
            multi_res_scales           = multi_res_scales,
            intensity_min              = intensity_min,
            intensity_max              = intensity_max,
            normalize                  = normalize,
            global_mean                = global_mean,
            global_std                 = global_std,
            foreground_oversample_ratio= foreground_oversample_ratio,
            samples_per_volume         = samples_per_volume,
            is_train                   = is_train,
            cache_enabled              = cache_enabled,
            cache_max_volumes          = cache_max_volumes,
            cache_int16                = cache_int16,
            region_weights             = region_weights,
            npz_paths                  = npz_paths,
            val_grid_coverage          = val_grid_coverage,
        )
        _init_cond_from_dataset(
            self,
            cond_normalize     = cond_normalize,
            cond_intensity_min = cond_intensity_min,
            cond_intensity_max = cond_intensity_max,
            cond_global_mean   = cond_global_mean,
            cond_global_std    = cond_global_std,
            cache_enabled      = cache_enabled,
            cache_max_volumes  = cache_max_volumes,
        )


class Volume3DWhole(CondVolumeMixin, SegDataset3DWhole):
    """整体卷 dataset（继承 seg 实现 + cond hook）。"""

    def __init__(
        self,
        image_paths         : List[str],
        label_paths         : List[str],
        label_values        : List[int],
        patch_size          : Tuple[int, int, int] = (64, 128, 128),
        aug_oversample_ratio: float = 1.0,
        intensity_min       : float = -1024.0,
        intensity_max       : float = 3071.0,
        normalize           : str = "minmax",
        global_mean         : float = 0.0,
        global_std          : float = 1.0,
        samples_per_volume  : int = 1,
        is_train            : bool = True,
        cache_enabled       : bool = True,
        cache_max_volumes   : int = 0,
        cache_int16         : bool = False,
        region_weights      : Optional[List[float]] = None,
        cond_normalize      : str = "minmax",
        cond_intensity_min  : float = -1024.0,
        cond_intensity_max  : float = 1024.0,
        cond_global_mean    : float = 0.0,
        cond_global_std     : float = 1.0,
        npz_paths           : Optional[List[str]] = None):
        super().__init__(
            image_paths          = image_paths,
            label_paths          = label_paths,
            label_values         = label_values,
            patch_size           = patch_size,
            aug_oversample_ratio = aug_oversample_ratio,
            intensity_min        = intensity_min,
            intensity_max        = intensity_max,
            normalize            = normalize,
            global_mean          = global_mean,
            global_std           = global_std,
            samples_per_volume   = samples_per_volume,
            is_train             = is_train,
            cache_enabled        = cache_enabled,
            cache_max_volumes    = cache_max_volumes,
            cache_int16          = cache_int16,
            region_weights       = region_weights,
            npz_paths            = npz_paths,
        )
        _init_cond_from_dataset(
            self,
            cond_normalize     = cond_normalize,
            cond_intensity_min = cond_intensity_min,
            cond_intensity_max = cond_intensity_max,
            cond_global_mean   = cond_global_mean,
            cond_global_std    = cond_global_std,
            cache_enabled      = cache_enabled,
            cache_max_volumes  = cache_max_volumes,
        )
