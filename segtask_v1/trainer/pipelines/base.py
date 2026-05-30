"""``ViewPipeline`` 策略对象基类。

每个子类对应 TODO #4 列出的一种训练模式。Pipeline 拥有：

* ``criterion``       —— 主损失（必要时已被 ``DeepSupervisionLoss`` 包装）
* ``inner_loss``      —— 内部 ``MultiResolutionLoss`` / ``SliceChannelLoss``，供 metrics ``split_for_metrics`` 共用
* ``aux_inner_loss``  —— 共享 aux 内损（folded / lift+aux）；否则 ``None``
* ``aux_inner_losses``—— 逐视图 aux 内损（仅 native_d 异深度路径）；否则 ``None``
* ``aux_weights``     —— ``list[float]``，长度 = ``n_aux_views``
* ``target_patch_size``——增强后中心裁回的目标尺寸
* 若适用：``mr_native_sizes``（3D 懒多分辨率）/ ``per_view_depths``（2.5D 异深 aux）

Pipeline 不持模型 / 优化器 / scaler / EMA —— 这些归 ``Trainer``。

接口：
    * ``prepare_batch(image, label, wmap)`` →  ``(model_input, SupervisionPack)``
    * ``prepare_val_batch(image, label)``   →  ``(model_input, label_main_for_metrics)``
    * ``compute_loss(pred, sup, breakdown=None)`` → ``Tensor``（主+aux 聚合）
    * ``split_for_metrics(pred, label_main)`` → ``(pred_1x, label_1x)``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class SupervisionPack:
    """单 step 监督信号容器。

    各 pipeline 仅填写自己需要的字段；``compute_loss`` 内部按需取用，调用方
    （Trainer._train_epoch）一目了然，不需要预知任何模式分支。
    """

    label_main: torch.Tensor
    wmap_main: Optional[torch.Tensor] = None
    # native_d 路径：逐视图 D_k 不同 → 必须以 list 携带（不可堆叠）
    aux_labels: Optional[List[torch.Tensor]] = None
    aux_wmaps: Optional[List[Optional[torch.Tensor]]] = field(default=None)
    # folded-aux / lift-aux 路径：逐视图 label 形状一致，rank-5 整存
    label_all_views: Optional[torch.Tensor] = None
    wmap_all_views: Optional[torch.Tensor] = None


class ViewPipeline(ABC):
    """Strategy base：训练模式专用的视图重塑 + 损失聚合。"""

    # 子类必须设置（在 __init__ 中）：
    criterion: torch.nn.Module
    inner_loss: torch.nn.Module
    aux_inner_loss: Optional[torch.nn.Module] = None
    aux_inner_losses: Optional[List[torch.nn.Module]] = None
    aux_weights: List[float]
    target_patch_size: Tuple[int, int, int]

    # 命名口径统一（消除 TODO 痛点 b 的"数量混用"）：
    n_views: int           # 数据/模型几何视图数 = len(cfg.data.multi_res_scales)
    n_aux_views: int       # = max(n_views - 1, 0) 当启用 aux；否则 0
    num_res_groups: int    # 损失内部"通道分组数"；2.5D folded=1, lift=1, 3D=n_views
    slab_depth: int = 0    # 仅 2.5D：D = patch_size[0]

    # 可选：仅特定 pipeline 用
    mr_native_sizes: List[Tuple[int, int, int]] = []
    per_view_depths: List[int] = []

    @abstractmethod
    def prepare_batch(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        wmap: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, SupervisionPack]:
        """把 dataset+augment 给的 batch 重塑为 ``(model_input, SupervisionPack)``。"""

    @abstractmethod
    def prepare_val_batch(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """validation 路径（无增强、无 wmap、无 aux）：返回 ``(model_input, label_main_for_metrics)``。"""

    @abstractmethod
    def compute_loss(
        self,
        pred,
        sup: SupervisionPack,
        breakdown: Optional[dict] = None,
    ) -> torch.Tensor:
        """主+aux 聚合损失。``breakdown`` 不为 ``None`` 时填 ``L_main / L_aux_k / w_aux_k / L_total`` 标量。"""

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    @staticmethod
    def extract_main_pred(pred):
        """提取主路输出：``dict→main → list[0]``；``list→[0]``；tensor 原返。"""
        if isinstance(pred, dict):
            pred = pred["main"]
        if isinstance(pred, list):
            pred = pred[0]
        return pred

    def split_for_metrics(self, pred, label_main):
        """与模式无关的 metrics reshape；委托给 ``inner_loss.split_for_metrics``。"""
        return self.inner_loss.split_for_metrics(pred, label_main)


__all__ = ["ViewPipeline", "SupervisionPack"]
