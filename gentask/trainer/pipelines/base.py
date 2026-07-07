"""生成任务多视图消费管线抽象基类。

dataset（SuperRes / Volume3D 系）总是发单条 max-FOV 过采样 cube
``(B, 1, eD, eH, eW)``；GPU 增强在完整 cube 上执行（保留裁剪余量），随后由
管线完成：过采样余量中心裁剪 → 逐视图 FOV 拆分 / 打包成模型输入布局。

管线只做几何（裁剪 / resize / 通道打包），不做强度处理；输入若已是模型
布局（例如测试直接合成的预打包 batch），``prepare_batch`` 原样透传。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class GenViewPipeline(ABC):
    """训练 / 验证 batch 的几何准备管线。

    Attributes:
        patch_size: 模型 patch 尺寸 (pD, pH, pW)。
        scales    : 多分辨率 scale 列表（view 0 恒 1.0）。
        n_views   : 视图数（= len(scales)）。
    """

    def __init__(self, patch_size, scales):
        self.patch_size = tuple(int(p) for p in patch_size)
        self.scales = [float(s) for s in (scales or [1.0])]
        self.n_views = len(self.scales)

    @abstractmethod
    def prepare_batch(
        self,
        image: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """把增强后的 max-FOV cube 变换为模型输入布局。

        返回 (image, weight_map, cond)；weight_map / cond 均对齐到主视图
        （view 0）几何。已是模型布局的输入原样透传。
        """

    def metric_views(
        self,
        rec: torch.Tensor,
        hr: torch.Tensor,
        lr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把验证指标的 (rec, hr, lr) 对齐到复原输出的通道布局。

        多视图时模型只复原主视图（rec 通道数 < hr）；hr / lr 打包到与 rec
        同 rank 后裁取领头 rec.shape[1] 通道（主视图恒在最前）。
        """
        def _align(x: torch.Tensor) -> torch.Tensor:
            if rec.ndim == 4 and x.ndim == 5:
                x = x.flatten(1, 2)
            if x.shape[1] > rec.shape[1]:
                x = x[:, : rec.shape[1]]
            return x

        return rec, _align(hr), _align(lr)


__all__ = ["GenViewPipeline"]
