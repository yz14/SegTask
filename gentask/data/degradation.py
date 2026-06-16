"""生成任务的退化算子（GPU）。

当前仅实现 **超分（super-resolution）** 退化：把干净的高分图（HR）下采样到
``1/sr_scale`` 再上采样回原尺寸，得到与 HR 同尺寸的低分图（LR），作为模型输入；
HR 本身作为重建目标（pre-upsampling SISR，类 SRCNN / VDSR 设定，输入输出同尺寸，
可直接复用编解码同尺寸的 U-Net）。可选在 LR 上叠加高斯噪声模拟采集噪声。

退化在 trainer 增强（augment + 中心裁）之后、于 GPU 上对一个 batch 施加，
故 ``degrade`` 接收 ``(B, C, *spatial)`` 的张量：

* 3D（``spatial_dims==3``）：tensor 形如 ``(B, 1, D, H, W)``，三个空间轴同时下采样。
* 2.5D（``spatial_dims==2``）：tensor 形如 ``(B, D, H, W)``（D 折叠进通道轴），
  仅在 ``(H, W)`` 两个空间轴下采样（逐切片 2D 超分），与 2.5D 把 D 视作通道一致。

设计为策略对象，未来可在此扩展 denoise / inpaint 等其它退化而不改调用方。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ..config import TaskConfig

# F.interpolate 的可选插值模式（按空间维度区分）。
_SCALABLE_MODES = {
    2: {"trilinear": "bilinear", "area": "area", "nearest": "nearest"},
    3: {"trilinear": "trilinear", "area": "area", "nearest": "nearest"},
}


def _interp_mode(sr_kernel: str, spatial_dims: int) -> str:
    """把配置里的 ``sr_kernel`` 映射到 ``F.interpolate`` 的 mode（按空间维度）。"""
    table = _SCALABLE_MODES[spatial_dims]
    if sr_kernel not in table:
        raise ValueError(
            f"Unknown sr_kernel {sr_kernel!r}; valid: {sorted(table)}.")
    return table[sr_kernel]


class SuperResDegradation:
    """超分退化：HR → LR（同尺寸，已上采样回原大小）。

    ``scale`` 必须能整除各空间轴尺寸；否则下采样后上采样会产生 ±1 像素偏差，
    这里通过显式记录原尺寸并在上采样时还原避免。
    """

    def __init__(
        self,
        scale: int,
        spatial_dims: int,
        kernel: str = "area",
        noise_std: float = 0.0):
        if scale < 1:
            raise ValueError(f"sr_scale must be >= 1; got {scale}.")
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3; got {spatial_dims}.")
        self.scale = int(scale)
        self.spatial_dims = int(spatial_dims)
        self.mode = _interp_mode(kernel, spatial_dims)
        self.noise_std = float(noise_std)
        # area / nearest 不支持 align_corners；线性族传 False 保持几何一致。
        self._align = None if self.mode in ("area", "nearest") else False

    def degrade(self, hr: torch.Tensor) -> torch.Tensor:
        """从干净 HR 生成同尺寸 LR。``hr`` 形如 ``(B, C, *spatial)``。"""
        if self.scale == 1 and self.noise_std == 0.0:
            return hr.clone()

        spatial = hr.shape[-self.spatial_dims:]
        if self.scale > 1:
            low = [max(int(round(s / self.scale)), 1) for s in spatial]
            down = F.interpolate(
                hr, size=tuple(low), mode=self.mode, align_corners=self._align)
            lr = F.interpolate(
                down, size=tuple(int(s) for s in spatial),
                mode=self.mode, align_corners=self._align)
        else:
            lr = hr.clone()

        if self.noise_std > 0.0:
            lr = lr + torch.randn_like(lr) * self.noise_std
        return lr


def build_degradation(cfg_task: TaskConfig, spatial_dims: int) -> SuperResDegradation:
    """按 ``task`` 配置构造退化算子。当前仅 'superres'。"""
    deg = str(cfg_task.degradation).lower()
    if deg != "superres":
        raise ValueError(f"Unsupported degradation {deg!r}; only 'superres'.")
    return SuperResDegradation(
        scale=int(cfg_task.sr_scale),
        spatial_dims=spatial_dims,
        kernel=str(cfg_task.sr_kernel).lower(),
        noise_std=float(cfg_task.sr_noise_std))


def make_pair(
    degradation: SuperResDegradation, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """便捷封装：返回 ``(lr_input, hr_target)``。"""
    return degradation.degrade(hr), hr


__all__ = ["SuperResDegradation", "build_degradation", "make_pair"]
