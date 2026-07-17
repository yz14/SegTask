"""中心线 / 距离场辅助头损失（拓扑感知多任务监督）。

辅助头与主分割头同形（``(B, C, *spatial)`` logits），监督目标由二值 label 即时派生
（``@torch.no_grad``）：

* ``centerline`` —— 复用 clDice 同款可微 soft-skeleton（Shit+ CVPR2021）作为软骨架目标，
  以 soft-Dice / BCE 监督。鼓励网络显式感知血管中心线、保连通/拓扑。
* ``distance``   —— 形态学迭代腐蚀得到"到边界的距离 / 血管半径"场（纯 torch，GPU，无新依赖），
  归一化到 ``[0, 1]`` 后以 SmoothL1 / MSE 回归监督。鼓励网络区分粗主干与细末梢。

目标变换按通道在空间维独立进行：2.5D 折叠 ``(B, num_fg*D, H, W)`` 自动得到逐切片 2D 拓扑，
3D ``(B, num_fg, D, H, W)`` 得到 3D 拓扑——与 2D/3D 模型口径天然一致，与具体 pipeline 无关。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from taskcore.config.core import LossConfig, ModelConfig
from .losses import _soft_erode, _soft_skeletonize


@torch.no_grad()
def soft_skeleton_target(label: torch.Tensor, n_iter: int) -> torch.Tensor:
    """二值 label → soft 骨架目标 ``[0, 1]``，同形。spatial_ndim 由 rank 推导。"""
    spatial_ndim = label.ndim - 2
    binary = (label > 0.5).to(label.dtype)
    return _soft_skeletonize(binary, n_iter, spatial_ndim)


@torch.no_grad()
def morph_distance_target(label: torch.Tensor, max_iter: int) -> torch.Tensor:
    """二值 label → 归一化到边界距离场 ``[0, 1]``（迭代腐蚀的 chebyshev 距离）。

    对前景体素，统计其在连续腐蚀下存活的步数 = 到边界的距离（体素，截断于 max_iter），
    再除以 max_iter 归一化。边界体素 ≈ 0，越深（越粗）越接近 1。
    """
    spatial_ndim = label.ndim - 2
    work = (label > 0.5).to(label.dtype)
    dist = torch.zeros_like(work)
    for _ in range(max_iter):
        work = _soft_erode(work, spatial_ndim)
        dist = dist + work
    return dist / float(max_iter)


def _resolve_loss_name(target: str, loss: str) -> str:
    """``auto`` → centerline 用 soft-dice，distance 用 smooth_l1。"""
    if loss != "auto":
        return loss
    return "dice" if target == "centerline" else "smooth_l1"


class AuxTopoLoss(nn.Module):
    """中心线/距离场辅助头损失。

    forward(pred, target_binary, weight_map=None)：
      * ``pred``          —— 辅助头 logits ``(B, C, *spatial)``。
      * ``target_binary`` —— 与 pred 同形的二值前景图（由 ``main_loss_fn.binarize_full``
        从整数 label 派生）；内部即时派生软骨架 / 距离场目标。
      * ``weight_map``    —— 接受但忽略（拓扑/距离目标上逐体素区域权重无一致语义，与 clDice 一致）。
    """

    def __init__(
        self,
        target: str = "centerline",
        loss: str = "auto",
        iter_: int = 3,
        smooth: float = 1e-5,
    ):
        super().__init__()
        if target not in ("centerline", "distance"):
            raise ValueError(
                f"target must be 'centerline' | 'distance', got {target!r}")
        if iter_ < 1:
            raise ValueError(f"iter_ must be >= 1, got {iter_}")
        self.target = target
        self.iter = int(iter_)
        self.smooth = float(smooth)
        self.loss_name = _resolve_loss_name(target, loss)
        # centerline 软骨架与 distance 半径场都是 [0,1] 标量场，回归/重叠损失统一在 sigmoid(pred) 上算。
        self._is_regression = self.loss_name in ("smooth_l1", "mse")

    @torch.no_grad()
    def _build_target(self, label: torch.Tensor) -> torch.Tensor:
        if self.target == "centerline":
            return soft_skeleton_target(label, self.iter)
        return morph_distance_target(label, self.iter)

    def _soft_dice(self, prob: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        p = prob.reshape(prob.shape[0], prob.shape[1], -1)
        t = tgt.reshape(tgt.shape[0], tgt.shape[1], -1)
        inter = (p * t).sum(dim=-1)
        denom = p.sum(dim=-1) + t.sum(dim=-1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target_binary: torch.Tensor,
        weight_map: torch.Tensor = None,
    ) -> torch.Tensor:
        del weight_map  # 有意忽略（见类 docstring）。
        if pred.shape != target_binary.shape:
            raise ValueError(
                f"AuxTopoLoss: pred/target shape mismatch "
                f"{tuple(pred.shape)} vs {tuple(target_binary.shape)}.")
        spatial_ndim = pred.ndim - 2
        if spatial_ndim not in (2, 3):
            raise ValueError(
                f"AuxTopoLoss expects 2D or 3D spatial input; got "
                f"pred.ndim={pred.ndim}.")
        tgt = self._build_target(target_binary.detach().to(pred.dtype))

        if self.loss_name == "bce":
            return F.binary_cross_entropy_with_logits(pred, tgt)

        prob = torch.sigmoid(pred)
        if self.loss_name == "dice":
            return self._soft_dice(prob, tgt)
        if self.loss_name == "smooth_l1":
            return F.smooth_l1_loss(prob, tgt)
        if self.loss_name == "mse":
            return F.mse_loss(prob, tgt)
        raise ValueError(f"Unknown aux_topo loss: {self.loss_name!r}")


def build_aux_topo_loss(
    model_cfg: ModelConfig, loss_cfg: LossConfig) -> AuxTopoLoss:
    """由 config 构建 ``AuxTopoLoss``。"""
    return AuxTopoLoss(
        target = model_cfg.aux_topo_target,
        loss   = loss_cfg.aux_topo_loss,
        iter_  = loss_cfg.aux_topo_iter,
    )


__all__ = [
    "AuxTopoLoss",
    "build_aux_topo_loss",
    "soft_skeleton_target",
    "morph_distance_target",
]
