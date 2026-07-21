"""GPU 3D 生成数据增强：薄封装 ``taskcore.data.augment``。

与分割入口差异仅在伴随张量契约：
* 无 label；``cond`` 与 image 同 warp（bilinear）
* ``weight_map`` **不**做越界覆写（``oob_fill=None``，保留 border）
* 可选 rank-4 输入自动升维后还原
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from taskcore.data.augment import Companion, GPUAugmentor as _CoreAugmentor

from ..config import AugConfig


class GPUAugmentor(_CoreAugmentor):
    """生成变体：``__call__(image, weight_map, cond)``。"""

    def __init__(self, cfg: AugConfig, max_scale: float = 1.0,
                 seed: Optional[int] = None,
                 inplace: Optional[bool] = None):
        super().__init__(
            cfg, max_scale=max_scale, label_fill=0.0,
            seed=seed, inplace=inplace)

    def __call__(
        self,
        image: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """对 batch 应用增强；返回 (image, weight_map, cond)。"""
        if not self.enabled:
            return image, weight_map, cond

        squeeze_back = image.ndim == 4
        if squeeze_back:
            image = image.unsqueeze(1)
            if weight_map is not None and weight_map.ndim == 4:
                weight_map = weight_map.unsqueeze(1)
            if cond is not None and cond.ndim == 4:
                cond = cond.unsqueeze(1)

        comps: List[Companion] = []
        wmap_i: Optional[int] = None
        cond_i: Optional[int] = None
        if weight_map is not None:
            wmap_i = len(comps)
            comps.append(Companion(
                weight_map, mode=self.wmap_interp_mode, oob_fill=None))
        if cond is not None:
            cond_i = len(comps)
            comps.append(Companion(cond, mode="bilinear", oob_fill=None))

        image, comps = self.apply(image, comps)

        weight_map = comps[wmap_i].tensor if wmap_i is not None else None
        cond = comps[cond_i].tensor if cond_i is not None else None

        if squeeze_back:
            image = image.squeeze(1)
            if weight_map is not None and weight_map.shape[1] == 1:
                weight_map = weight_map.squeeze(1)
            if cond is not None and cond.shape[1] == 1:
                cond = cond.squeeze(1)
        return image, weight_map, cond


__all__ = ["GPUAugmentor", "Companion"]
