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

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.degradation import build_degradation
from .diffusion import build_diffusion
from .factory import build_backbone
from .topology import build_topology


class RegressionModel(nn.Module):
    """前馈回归复原（DnCNN / SRCNN / U-Net regression）。

    复用分割工厂构造的图到图网络（输出通道已按 ``out_channels`` 配置），一次前向把
    低分图映射回高分图；``residual=True`` 时学习残差 ``HR−LR``（DnCNN / VDSR）。

    深监督（``model.deep_supervision=True``）：训练时 backbone 输出多尺度头
    ``[full, /2, ...]``，``forward`` 经 ``ds_preds`` 全部返回，由训练器对每个尺度
    与下采样后的 HR 算重建损失并加权聚合；推理（``restore``）只取全分辨率头。
    """

    def __init__(
        self,
        net: nn.Module,
        degradation,
        residual: bool = False,
        spatial_dims: int = 2,
        view_depths: Optional[Sequence[int]] = None,
        aux_views_active: bool = False):
        super().__init__()
        self.net = net
        self.degradation = degradation
        self.residual = bool(residual)
        self.spatial_dims = int(spatial_dims)
        self.view_depths = tuple(int(v) for v in view_depths) if view_depths else ()
        self.aux_views_active = bool(aux_views_active)

    def _is_multi_view_2_5d(self) -> bool:
        return self.spatial_dims == 2 and len(self.view_depths) > 1

    def _pack_2_5d(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """单视图 2.5D 把 (B,C,D,H,W) 折为 (B,C*D,H,W)；4D 输入保持不变。"""
        if x is None or self.spatial_dims != 2:
            return x
        if x.ndim == 5:
            return x.flatten(1, 2)
        return x

    def _concat_cond(
        self, lr: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return lr
        lr_p = self._pack_2_5d(lr)
        cond_p = self._pack_2_5d(cond)
        if lr_p.shape[0] != cond_p.shape[0] or lr_p.shape[2:] != cond_p.shape[2:]:
            raise ValueError(
                f"cond shape {tuple(cond.shape)} cannot be aligned with "
                f"lr shape {tuple(lr.shape)}.")
        return torch.cat([lr_p, cond_p], dim=1)

    def _view_splits(self, x: torch.Tensor) -> List[torch.Tensor]:
        if not self._is_multi_view_2_5d():
            return [x]
        try:
            return list(torch.split(x, list(self.view_depths), dim=1))
        except RuntimeError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"packed 2.5D tensor with shape {tuple(x.shape)} does not "
                f"match view_depths={self.view_depths}.") from exc

    def _main_view(self, x: torch.Tensor) -> torch.Tensor:
        return self._view_splits(x)[0]

    def degrade(self, hr: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2 and hr.ndim == 5:
            hr = hr.flatten(1, 2)
        return self.degradation.degrade(hr)

    def _add_residual(self, out: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """残差基线：全分辨率头加 ``lr``；下采样头加 ``lr`` 缩放到该头尺寸。"""
        if out.shape[-self.spatial_dims:] != base.shape[-self.spatial_dims:]:
            base = F.interpolate(
                base, size=tuple(out.shape[-self.spatial_dims:]), mode="area")
        if out.shape[1] != base.shape[1]:
            raise ValueError(
                "residual=True requires net out channels == base channels "
                f"({out.shape[1]} != {base.shape[1]}); set task.residual=False "
                "or match out_channels to input depth.")
        return out + base

    def _heads(
        self, lr: torch.Tensor, base_lr: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """返回主路径多尺度头与可选 aux 头；head[0] 为全分辨率。"""
        out = self.net(lr)
        aux = []
        if isinstance(out, dict):
            aux = list(out.get("aux", []))
            out = out["main"]
        heads = list(out) if isinstance(out, (list, tuple)) else [out]
        base = self._main_view(base_lr if base_lr is not None else lr)
        if self.residual:
            heads = [self._add_residual(h, base) for h in heads]
        # aux 头直接重建各自 view，不叠 residual；residual 只作用于主 view 0。
        return heads, aux

    def restore(self, lr: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        lr = self._pack_2_5d(lr)
        base_lr = lr
        lr = self._concat_cond(lr, cond)
        return self._heads(lr, base_lr=base_lr)[0][0]

    def _target_views(self, hr: torch.Tensor) -> List[torch.Tensor]:
        if not self._is_multi_view_2_5d():
            return [self._pack_2_5d(hr)]
        return self._view_splits(hr)

    def forward(self, hr: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hr = self._pack_2_5d(hr)
        cond = self._pack_2_5d(cond) if cond is not None else None
        lr = self.degrade(hr)
        base_lr = lr
        lr = self._concat_cond(lr, cond)
        heads, aux = self._heads(lr, base_lr=base_lr)
        target_views = self._target_views(hr)
        out = {"pred": heads[0], "ds_preds": heads, "target": target_views[0]}
        if self._is_multi_view_2_5d() and self.aux_views_active and aux:
            if len(aux) != len(target_views) - 1:
                raise ValueError(
                    f"aux head count {len(aux)} does not match auxiliary "
                    f"view count {len(target_views) - 1}.")
            out["aux_preds"] = aux
            out["aux_targets"] = target_views[1:]
        return out


class DiffusionModel(nn.Module):
    """条件扩散复原（DDPM / EDM），以低分图为条件迭代去噪。"""

    def __init__(self, diffusion: nn.Module, degradation, spatial_dims: int):
        super().__init__()
        self.diffusion = diffusion
        self.degradation = degradation
        self.spatial_dims = int(spatial_dims)

    def degrade(self, hr: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2 and hr.ndim == 5:
            hr = hr.flatten(1, 2)
        return self.degradation.degrade(hr)

    def _pack_2_5d(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or self.spatial_dims != 2:
            return x
        if x.ndim == 5:
            return x.flatten(1, 2)
        return x

    def _concat_cond(
        self, lr: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return lr
        if lr.shape[0] != cond.shape[0] or lr.shape[2:] != cond.shape[2:]:
            raise ValueError(
                f"cond shape {tuple(cond.shape)} cannot be aligned with "
                f"lr shape {tuple(lr.shape)}.")
        return torch.cat([lr, cond], dim=1)

    @torch.no_grad()
    def restore(self, lr: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        lr = self._pack_2_5d(lr)
        cond = self._pack_2_5d(cond)
        cond_full = self._concat_cond(lr, cond)
        return self.diffusion.sample(cond=cond_full, target_channels=lr.shape[1])

    def forward(self, hr: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hr = self._pack_2_5d(hr)
        cond = self._pack_2_5d(cond)
        lr = self.degrade(hr)
        cond_full = self._concat_cond(lr, cond)
        return self.diffusion.train_outputs(hr, cond=cond_full)


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

    topology = build_topology(cfg)
    # 单一真相源：由 topology 派生（2.5D+lift_2_5d_to_3d 时为 3，退化/打包均按
    # 真 3D 空间轴处理），不在此处重复推导。
    spatial_dims = int(topology.spatial_dims)
    degradation = build_degradation(task, spatial_dims=spatial_dims)
    algo = str(task.algorithm).lower()

    if algo == "regression":
        if topology.patch_mode == "2_5d" and topology.n_views > 1 and not topology.lift_2_5d_to_3d:
            if topology.keep_native_view_depth and topology.per_view_depths:
                view_depths = list(topology.per_view_depths)
            else:
                view_depths = [topology.slab_depth] * topology.n_views
        else:
            view_depths = None
        net = build_backbone(cfg)  # 复用图到图 backbone（输出通道按 out_channels）
        return RegressionModel(net, degradation, residual=bool(task.residual),
                               spatial_dims=spatial_dims,
                               view_depths=view_depths,
                               aux_views_active=(topology.aux_seg_active
                                                 and topology.patch_mode == "2_5d"
                                                 and not topology.lift_2_5d_to_3d))

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
        return DiffusionModel(diffusion, degradation, spatial_dims=spatial_dims)

    raise ValueError(
        f"task.algorithm must be 'regression' | 'diffusion'; got {algo!r}")


__all__ = ["RegressionModel", "DiffusionModel", "build_generation_model"]
