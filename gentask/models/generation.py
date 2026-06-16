"""生成（超分）模型封装：回归复原 与 条件扩散。

对外暴露统一接口，便于 trainer / predictor 不感知具体算法：

* ``forward(hr)``  —— 训练用。内部用退化算子从干净图 ``hr`` 造低分条件图 ``lr``：
    * 回归：返回 ``{"pred": HR̂, "target": hr}``；
    * 扩散：返回 ``{"pred", "target", "weight"}``（交 ``DiffusionLoss``）。
* ``restore(lr)`` —— 推理用，由低分图复原高分图。
* ``degrade(hr)`` —— 暴露退化算子，便于验证阶段在线造 (lr, hr) 对。

仅超分（``task.degradation=='superres'``）。2.5D 折叠 D 到通道；退化只作用空间轴。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..data.degradation import build_degradation
from .diffusion import build_diffusion
from .factory import build_backbone


class RegressionModel(nn.Module):
    """前馈回归复原（DnCNN / SRCNN / U-Net regression）。

    复用分割工厂构造的图到图网络（输出通道已按 ``out_channels`` 配置），一次前向把
    低分图映射回高分图；``residual=True`` 时学习残差 ``HR−LR``（DnCNN / VDSR）。
    """

    def __init__(self, net: nn.Module, degradation, residual: bool = False):
        super().__init__()
        self.net = net
        self.degradation = degradation
        self.residual = bool(residual)

    def degrade(self, hr: torch.Tensor) -> torch.Tensor:
        return self.degradation.degrade(hr)

    def restore(self, lr: torch.Tensor) -> torch.Tensor:
        out = self.net(lr)
        if isinstance(out, (list, tuple)):
            out = out[0]
        elif isinstance(out, dict):
            out = out["main"][0] if isinstance(out["main"], list) else out["main"]
        if self.residual:
            if out.shape[1] != lr.shape[1]:
                raise ValueError(
                    "residual=True requires net out channels == lr channels "
                    f"({out.shape[1]} != {lr.shape[1]}); set task.residual=False "
                    "or match out_channels to input depth.")
            out = out + lr
        return out

    def forward(self, hr: torch.Tensor) -> Dict[str, torch.Tensor]:
        lr = self.degrade(hr)
        return {"pred": self.restore(lr), "target": hr}


class DiffusionModel(nn.Module):
    """条件扩散复原（DDPM / EDM），以低分图为条件迭代去噪。"""

    def __init__(self, diffusion: nn.Module, degradation):
        super().__init__()
        self.diffusion = diffusion
        self.degradation = degradation

    def degrade(self, hr: torch.Tensor) -> torch.Tensor:
        return self.degradation.degrade(hr)

    @torch.no_grad()
    def restore(self, lr: torch.Tensor) -> torch.Tensor:
        return self.diffusion.sample(cond=lr)

    def forward(self, hr: torch.Tensor) -> Dict[str, torch.Tensor]:
        lr = self.degrade(hr)
        return self.diffusion.train_outputs(hr, cond=lr)


def _slab_depth(cfg) -> int:
    """2.5D 折叠深度 D（= patch_size[0]）；3D 返回 1（不折叠）。"""
    if cfg.data.patch_mode == "2_5d":
        return int(cfg.data.patch_size[0])
    return 1


def build_generation_model(cfg) -> nn.Module:
    """按 ``cfg.task.algorithm`` 构造回归 / 扩散生成模型。"""
    task = cfg.task
    if str(task.degradation).lower() != "superres":
        raise ValueError(
            f"task.degradation must be 'superres'; got {task.degradation!r}")

    spatial_dims = 2 if cfg.data.patch_mode == "2_5d" else 3
    degradation = build_degradation(task, spatial_dims=spatial_dims)
    algo = str(task.algorithm).lower()

    if algo == "regression":
        net = build_backbone(cfg)  # 复用图到图 backbone（输出通道按 out_channels）
        return RegressionModel(net, degradation, residual=bool(task.residual))

    if algo == "diffusion":
        arch = str(cfg.model.arch).lower()
        D = _slab_depth(cfg)
        target_ch = int(task.out_channels) * D       # 高分（噪声）图折叠通道
        cond_ch = int(cfg.model.in_channels)         # 低分条件图通道
        in_ch = target_ch + cond_ch
        if arch == "adm":
            from .adm_unet import build_adm_diffusion_unet
            net = build_adm_diffusion_unet(cfg, in_channels=in_ch, out_channels=target_ch)
        elif arch == "edm2":
            from .edm2_unet import build_edm2_diffusion_unet
            net = build_edm2_diffusion_unet(cfg, in_channels=in_ch, out_channels=target_ch)
        else:
            raise ValueError(
                f"task.algorithm='diffusion' requires model.arch in "
                f"{{'adm','edm2'}} (paper-faithful σ/timestep conditioning); "
                f"got arch={arch!r}.")
        diffusion = build_diffusion(cfg, net)
        return DiffusionModel(diffusion, degradation)

    raise ValueError(
        f"task.algorithm must be 'regression' | 'diffusion'; got {algo!r}")


__all__ = ["RegressionModel", "DiffusionModel", "build_generation_model"]
