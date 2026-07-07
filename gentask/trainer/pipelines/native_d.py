"""2.5D 原生深度多视图管线（keep_native_view_depth）。

max-FOV cube ``(B, 1, Dm, H, W)`` → 逐视图中心裁 z-slab（原生深度 D_k，
不 resize）→ 按通道拼接为 ``(B, ΣD_k, H, W)``（模型 ``in_ch_per_view_list``
布局）。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from ..views import center_crop, split_views_native_d
from .base import GenViewPipeline


class NativeDPipeline(GenViewPipeline):
    """2.5D keep_native_view_depth 管线。"""

    def __init__(self, patch_size, scales, per_view_depths: Sequence[int]):
        super().__init__(patch_size, scales)
        self.depths = [int(d) for d in per_view_depths]
        pD, pH, pW = self.patch_size
        self.cube_size = (max(self.depths), pH, pW)
        self.main_depth = self.depths[0]

    def prepare_batch(
        self,
        image: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # rank-4 预打包（合成测试 batch）原样透传。
        if image.ndim == 4:
            return image, weight_map, cond

        cube = center_crop(image, self.cube_size)
        image = split_views_native_d(cube, self.depths)
        pD, pH, pW = self.patch_size
        main_size = (self.main_depth, pH, pW)
        if weight_map is not None and weight_map.ndim == 5:
            weight_map = center_crop(weight_map, main_size)
        if cond is not None and cond.ndim == 5:
            cond = center_crop(cond, main_size)
        return image, weight_map, cond


__all__ = ["NativeDPipeline"]
