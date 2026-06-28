"""掩码图像建模（MIM）共享工具（2D/3D，GPU 友好，无外部依赖）。

供掩码类自监督方法复用：SimMIM②（mask-token 稠密前向）、SparK①（稀疏/掩码-稠密）、
iBOT⑥（掩码特征预测）、JEPA⑦（隐空间掩码）。统一约定：

* **单元网格掩码**：在与骨干总步长对齐（或可配）的"单元"上采样二值掩码，``1`` 表示
  *被遮*、``0`` 表示*可见*；同一单元内所有体素同遮，保证逐尺度下采样后掩码空间一致
  （SSL.md 方案①"多尺度传播"、方案②"patch-wise 掩码"）。
* **逐样本固定比例**：每个样本独立按 ``mask_ratio`` 随机抽取被遮单元（最近邻上采样到
  目标分辨率），与 batch 内其它样本解耦。

所有函数对 2D/3D 通用（由张量维度自动推断 spatial_dims）。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

#: 仅在被遮位点计算的重建/回归损失（reduction='none' 版，与 SSLConfig.recon_loss 对齐）。
_ELEMWISE_LOSS_FNS = {
    "l1": lambda p, t: F.l1_loss(p, t, reduction="none"),
    "smooth_l1": lambda p, t: F.smooth_l1_loss(p, t, reduction="none"),
    "mse": lambda p, t: F.mse_loss(p, t, reduction="none"),
}


def _as_unit_tuple(unit, spatial_dims: int) -> Tuple[int, ...]:
    """把标量/序列单元尺寸广播为 per-axis 元组。"""
    if isinstance(unit, (int, float)):
        return tuple(int(unit) for _ in range(spatial_dims))
    u = tuple(int(v) for v in unit)
    if len(u) != spatial_dims:
        raise ValueError(
            f"mask unit length {len(u)} != spatial_dims {spatial_dims}.")
    return u


def compute_grid_shape(spatial: Sequence[int], unit) -> Tuple[int, ...]:
    """由 spatial 尺寸与单元尺寸算单元网格形状（向上取整，保证每轴 >= 1）。"""
    spatial = tuple(int(s) for s in spatial)
    u = _as_unit_tuple(unit, len(spatial))
    return tuple(max(1, math.ceil(s / max(uu, 1))) for s, uu in zip(spatial, u))


def sample_unit_mask(
    batch_size: int,
    grid_shape: Sequence[int],
    mask_ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """逐样本随机单元掩码。返回 ``(B, 1, *grid_shape)`` 的 {0,1} 张量（1=被遮）。

    每个样本独立遮 ``round(mask_ratio * num_units)`` 个单元（下取整后至少 1、至多
    全部减一，保证既非全可见也非全被遮，避免退化损失/前向）。
    """
    grid_shape = tuple(int(g) for g in grid_shape)
    num_units = 1
    for g in grid_shape:
        num_units *= g
    if num_units < 1:
        raise ValueError(f"empty grid_shape={grid_shape}.")
    num_mask = int(round(float(mask_ratio) * num_units))
    num_mask = min(max(num_mask, 1), max(num_units - 1, 1))

    noise = torch.rand(batch_size, num_units, device=device, generator=generator)
    ids_shuffle = noise.argsort(dim=1)
    mask = torch.zeros(batch_size, num_units, device=device)
    mask.scatter_(1, ids_shuffle[:, :num_mask], 1.0)
    return mask.view(batch_size, 1, *grid_shape)


def upsample_mask_to(mask_grid: torch.Tensor,
                     target_spatial: Sequence[int]) -> torch.Tensor:
    """把单元网格掩码最近邻上采样到 ``target_spatial``。返回 ``(B, 1, *target)`` {0,1}。"""
    target_spatial = tuple(int(s) for s in target_spatial)
    if mask_grid.shape[2:] == target_spatial:
        return mask_grid
    # 最近邻保持二值；mode='nearest' 对 2D/3D 通用。
    return F.interpolate(mask_grid, size=target_spatial, mode="nearest")


def make_unit_mask(
    batch_size: int,
    spatial: Sequence[int],
    unit,
    mask_ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """便捷封装：采样单元掩码并上采样到 ``spatial``。返回 ``(B, 1, *spatial)`` {0,1}。"""
    grid = compute_grid_shape(spatial, unit)
    mask_grid = sample_unit_mask(batch_size, grid, mask_ratio, device, generator)
    return upsample_mask_to(mask_grid, spatial)


def apply_mask_token(x: torch.Tensor, mask_full: torch.Tensor,
                     mask_token: torch.Tensor) -> torch.Tensor:
    """在被遮位点用可学习 ``mask_token`` 替换输入（SimMIM 稠密前向）。

    ``x``: (B, C, *spatial)；``mask_full``: (B, 1, *spatial) {0,1}；``mask_token``:
    可广播到 (1, C, *1) 的可学习向量。返回同形被遮输入（不原地修改 ``x``）。
    """
    m = mask_full.to(x.dtype)
    return x * (1.0 - m) + mask_token.to(x.dtype) * m


def downsample_mask_to(mask_full: torch.Tensor,
                       target_spatial: Sequence[int]) -> torch.Tensor:
    """把全分辨率掩码最近邻重采样到 ``target_spatial``（2D/3D 通用）。

    与 :func:`upsample_mask_to` 同为 ``mode='nearest'``，但语义上用于*下采样*到各
    encoder 尺度（SparK 逐尺度门控），保持二值不被插值污染。返回 ``(B, 1, *target)``。
    """
    target_spatial = tuple(int(s) for s in target_spatial)
    if mask_full.shape[2:] == target_spatial:
        return mask_full
    return F.interpolate(mask_full, size=target_spatial, mode="nearest")


def _avg_pool_nd(x: torch.Tensor, unit: Sequence[int]) -> torch.Tensor:
    """对 (B,C,*spatial) 做 kernel=stride=unit 的均值池化（2D/3D，ceil + 不计 pad）。"""
    dims = x.dim() - 2
    k = tuple(int(u) for u in unit)
    if dims == 3:
        return F.avg_pool3d(x, kernel_size=k, stride=k, ceil_mode=True,
                            count_include_pad=False)
    if dims == 2:
        return F.avg_pool2d(x, kernel_size=k, stride=k, ceil_mode=True,
                            count_include_pad=False)
    raise ValueError(f"_avg_pool_nd supports 2D/3D only; got {dims}D.")


def per_unit_normalize(x: torch.Tensor, unit, eps: float = 1e-6
                       ) -> torch.Tensor:
    """对每个单元内体素做归一化（减单元均值、除单元标准差），逐 (样本,通道,单元)。

    SSL.md 方案①：归一化重建目标以稳定优化、削弱 HU 绝对值偏置（沿用 MAE 经验）。
    用 kernel=stride=unit 的均值池化求单元均值与平方均值，再最近邻上采样回原分辨率
    广播。``x``: (B, C, *spatial) → 同形归一化张量（2D/3D 通用）。
    """
    spatial = x.shape[2:]
    u = _as_unit_tuple(unit, len(spatial))
    mean = _avg_pool_nd(x, u)                       # (B,C,*grid)
    mean_sq = _avg_pool_nd(x * x, u)
    var = (mean_sq - mean * mean).clamp_min(0.0)
    mean = upsample_mask_to(mean, spatial)          # 最近邻广播回原分辨率
    std = upsample_mask_to(var, spatial).add_(eps * eps).sqrt_()
    return (x - mean) / std


def densify(feat: torch.Tensor, visible: torch.Tensor,
            mask_embed: torch.Tensor) -> torch.Tensor:
    """SparK densify：被遮位点用可学习 ``mask_embed`` 填充，可见位点保留特征。

    ``feat``: (B, C, *spatial)；``visible``: (B, 1, *spatial) {0,1}（1=可见）；
    ``mask_embed``: 可广播到 (1, C, *1) 的可学习向量。返回同形稠密特征。
    """
    v = visible.to(feat.dtype)
    return feat * v + mask_embed.to(feat.dtype) * (1.0 - v)


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor,
                      mask_full: torch.Tensor, loss_name: str = "l1"
                      ) -> torch.Tensor:
    """仅在被遮位点计算重建损失（按被遮元素数 × 通道数归一）。

    ``pred``/``target``: (B, C, *spatial)；``mask_full``: (B, 1, *spatial) {0,1}。
    """
    fn = _ELEMWISE_LOSS_FNS.get(loss_name)
    if fn is None:
        raise ValueError(
            f"Unknown loss_name {loss_name!r}; valid: {sorted(_ELEMWISE_LOSS_FNS)}.")
    per_elem = fn(pred.float(), target.float())          # (B, C, *spatial)
    m = mask_full.to(per_elem.dtype)                     # (B, 1, *spatial)
    denom = m.sum() * per_elem.shape[1]                  # 被遮体素数 × 通道数
    return (per_elem * m).sum() / denom.clamp_min(1.0)


__all__ = [
    "compute_grid_shape",
    "sample_unit_mask",
    "upsample_mask_to",
    "make_unit_mask",
    "apply_mask_token",
    "downsample_mask_to",
    "per_unit_normalize",
    "densify",
    "masked_recon_loss",
]
