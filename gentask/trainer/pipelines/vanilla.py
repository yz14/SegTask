"""单视图管线：仅移除增强过采样余量（中心裁剪回 patch_size）。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..views import center_crop
from .base import GenViewPipeline


class VanillaPipeline(GenViewPipeline):
    """单视图（n_views == 1）whole / z_axis / cubic / 2.5D 通用管线。

    dataset 发 ``(B, 1, eD, eH, eW)``（e* = round(patch*oversample)，未过采样
    的轴与 patch 等尺寸）；此处中心裁剪回 patch_size。已等尺寸时全 no-op，
    rank-4 预打包输入（合成测试 batch）同样按最后三维裁剪。
    """

    def prepare_batch(
        self,
        image: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        image = center_crop(image, self.patch_size)
        if weight_map is not None:
            weight_map = center_crop(weight_map, self.patch_size)
        if cond is not None:
            cond = center_crop(cond, self.patch_size)
        return image, weight_map, cond


__all__ = ["VanillaPipeline"]
