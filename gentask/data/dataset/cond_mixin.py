"""生成任务 cond 体加载 mixin（P2c hook 实现）。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from taskcore.data.dataset import (
    VolumeCache,
    extract_z_patch_padded,
    preprocess_image,
    resize_3d,
)
from taskcore.data.patch_ops import extract_cubic_patch

from .io import load_npz_cond


def _channelwise_3d(
    vol: Optional[np.ndarray],
    fn,
) -> Optional[np.ndarray]:
    """对 (C,D,H,W) 逐通道应用 3D 体操作；3D 输入会先补通道维。"""
    if vol is None:
        return None
    if vol.ndim == 3:
        vol = vol[np.newaxis]
    if vol.ndim != 4:
        raise ValueError(f"Expected 3D or 4D volume, got {vol.ndim}D")
    out = [fn(ch) for ch in vol]
    return np.stack(out, axis=0)


class CondVolumeMixin:
    """在 seg patch dataset 的 hook 上追加 cond 张量。"""

    def _init_cond_fields(
        self,
        *,
        cond_normalize: str,
        cond_intensity_min: float,
        cond_intensity_max: float,
        cond_global_mean: float,
        cond_global_std: float,
        cache_enabled: bool,
        cache_max_volumes: int,
    ) -> None:
        self.cond_normalize = cond_normalize
        self.cond_intensity_min = cond_intensity_min
        self.cond_intensity_max = cond_intensity_max
        self.cond_global_mean = cond_global_mean
        self.cond_global_std = cond_global_std
        self._cond_cache = VolumeCache(cache_enabled, cache_max_volumes)

    def _load_cond(self, vol_idx: int) -> Optional[np.ndarray]:
        """加载条件体（npz）；无 cond 返 None。"""
        path = self._npz_paths[vol_idx]
        cached = self._cond_cache.get(path)
        if cached is not None:
            return cached
        cond = load_npz_cond(path)
        if cond is None:
            return None
        if cond.ndim == 3:
            cond = cond[np.newaxis]
        if cond.ndim != 4:
            raise ValueError(
                f"Expected cond volume to have shape (C,D,H,W) or (D,H,W); "
                f"got {cond.shape}")
        normed = np.empty_like(cond, dtype=np.float32)
        for i, ch in enumerate(cond):
            normed[i] = preprocess_image(
                ch,
                self.cond_intensity_min,
                self.cond_intensity_max,
                self.cond_normalize,
                self.cond_global_mean,
                self.cond_global_std,
            )
        self._cond_cache.put(path, normed)
        return normed

    def _pack_extra_sample_tensors(
        self,
        result: Dict[str, torch.Tensor],
        *,
        vol_idx: int,
        mode: str,
        **loc,
    ) -> None:
        cond_vol = self._load_cond(vol_idx)
        if cond_vol is None:
            return
        if mode == "z_axis":
            self._pack_cond_z(result, cond_vol, loc)
        elif mode == "cubic":
            self._pack_cond_cubic(result, cond_vol, loc)
        elif mode == "whole":
            self._pack_cond_whole(result, cond_vol, loc)

    def _pack_cond_z(
        self,
        result: Dict[str, torch.Tensor],
        cond_vol: np.ndarray,
        loc: dict,
    ) -> None:
        z = int(loc["z"])
        eD_max = int(loc["eD_max"])
        eH = int(loc["eH"])
        eW = int(loc["eW"])
        cond_s = _channelwise_3d(
            cond_vol, lambda ch: extract_z_patch_padded(ch, z, eD_max))
        assert cond_s is not None
        cond_s = resize_3d(cond_s, eD_max, eH, eW, is_label=False)
        result["cond"] = torch.from_numpy(
            np.ascontiguousarray(cond_s.astype(np.float32, copy=False)))

    def _pack_cond_cubic(
        self,
        result: Dict[str, torch.Tensor],
        cond_vol: np.ndarray,
        loc: dict,
    ) -> None:
        center: Tuple[int, int, int] = loc["center"]
        eD_max = int(loc["eD_max"])
        eH_max = int(loc["eH_max"])
        eW_max = int(loc["eW_max"])
        size_max = (eD_max, eH_max, eW_max)
        cond_s = _channelwise_3d(
            cond_vol, lambda ch: extract_cubic_patch(ch, center, size_max))
        assert cond_s is not None
        cond_s = resize_3d(cond_s, eD_max, eH_max, eW_max, is_label=False)
        result["cond"] = torch.from_numpy(
            np.ascontiguousarray(cond_s.astype(np.float32, copy=False)))

    def _pack_cond_whole(
        self,
        result: Dict[str, torch.Tensor],
        cond_vol: np.ndarray,
        loc: dict,
    ) -> None:
        eD = int(loc["eD"])
        eH = int(loc["eH"])
        eW = int(loc["eW"])
        cond_r = resize_3d(cond_vol, eD, eH, eW, is_label=False)
        result["cond"] = torch.from_numpy(
            np.ascontiguousarray(cond_r.astype(np.float32, copy=False)))


__all__ = ["CondVolumeMixin"]
