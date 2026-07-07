"""多视图堆叠管线：max-FOV cube → 逐视图裁剪 + resize → 通道堆叠。

覆盖 z_axis / cubic 3D 多分辨率，以及 2.5D 统一深度（uniform）与
lift_2_5d_to_3d 变体——四者仅视图裁剪尺寸不同，打包布局一致：
输出 ``(B, n_views, pD, pH, pW)``（2.5D 非 lift 由模型 ``_pack_2_5d``
折叠为 ``(B, n_views*D, H, W)``；lift 保持 rank-5 走真 3D）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..views import center_crop, split_views_stacked, view_sizes_cubic, view_sizes_z
from .base import GenViewPipeline


class StackedMultiResPipeline(GenViewPipeline):
    """多分辨率视图堆叠管线。

    ``cubic_fov=True``（patch_mode='cubic'）时三轴同步按 scale 放大；否则
    （z_axis / 2.5D）仅 z 轴放大、面内恒 patch 尺寸。
    """

    def __init__(self, patch_size, scales, cubic_fov: bool):
        super().__init__(patch_size, scales)
        self.cubic_fov = bool(cubic_fov)
        sizes = (view_sizes_cubic if self.cubic_fov else view_sizes_z)(
            self.patch_size, self.scales)
        self.view_sizes = sizes
        # 过采样余量裁剪目标 = 最大视图 FOV（各轴逐视图最大值）。
        self.cube_size = tuple(max(s[ax] for s in sizes) for ax in range(3))

    def prepare_batch(
        self,
        image: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 预打包输入（rank-4，或 rank-5 已按视图拆好）原样透传。
        if image.ndim == 4 or (image.ndim == 5 and image.shape[1] != 1):
            return image, weight_map, cond

        cube = center_crop(image, self.cube_size)
        image = split_views_stacked(cube, self.view_sizes, self.patch_size)
        # 主视图（view 0，scale=1.0）几何 = patch_size 中心裁剪，无 resize；
        # weight_map / cond 只监督 / 条件化主视图。
        if weight_map is not None and weight_map.ndim == 5:
            weight_map = center_crop(weight_map, self.patch_size)
        if cond is not None and cond.ndim == 5:
            cond = center_crop(cond, self.patch_size)
        return image, weight_map, cond


__all__ = ["StackedMultiResPipeline"]
