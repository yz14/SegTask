"""生成（图像复原）任务的损失与指标。

包含两部分：

1. **回归复原损失** —— Charbonnier / L1 / MSE 主损失，可选叠加 (1-SSIM) 与梯度
   （边缘）L1。由 ``ReconstructionLoss`` 统一封装，``build_recon_loss(cfg)`` 构造。
2. **扩散损失** —— ``DiffusionLoss``：对 ``DiffusionTrainWrapper`` 输出的
   ``{pred, target, weight}`` 做逐元素加权 MSE（EDM/DDPM 的 σ 加权由 wrapper 给出）。

以及训练/验证通用的图像质量指标 ``psnr`` / ``ssim``（均在数据归一化后的尺度上计算）。

约定：张量形如 ``(B, C, *spatial)``。2.5D 下 C=D（切片折叠为通道），SSIM/梯度按
逐通道 + 空间窗计算，与"D 视作通道"的设定一致。
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 逐像素回归损失
# ---------------------------------------------------------------------------
def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier（平滑 L1）：sqrt((x-y)^2 + eps^2) 的均值。"""
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


_PIXEL_LOSSES = {
    "l1": lambda p, t, eps: F.l1_loss(p, t),
    "mse": lambda p, t, eps: F.mse_loss(p, t),
    "charbonnier": lambda p, t, eps: charbonnier(p, t, eps),
}


# ---------------------------------------------------------------------------
# 梯度（边缘）损失
# ---------------------------------------------------------------------------
def _finite_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿空间维 ``dim`` 的一阶前向差分（保持其余形状）。"""
    return x.narrow(dim, 1, x.size(dim) - 1) - x.narrow(dim, 0, x.size(dim) - 1)


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """对各空间轴的一阶差分做 L1，鼓励边缘对齐。"""
    total = pred.new_zeros(())
    first_spatial = pred.ndim - spatial_dims
    for d in range(first_spatial, pred.ndim):
        total = total + F.l1_loss(_finite_diff(pred, d), _finite_diff(target, d))
    return total / spatial_dims


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    return g / g.sum()


def _separable_blur(x: torch.Tensor, win1d: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """对 ``(B, C, *spatial)`` 做可分离高斯模糊（逐通道，groups=C）。"""
    conv = {2: F.conv2d, 3: F.conv3d}[spatial_dims]
    C = x.shape[1]
    k = win1d.numel()
    pad = k // 2
    out = x
    for axis in range(spatial_dims):
        shape = [1, 1] + [1] * spatial_dims
        shape[2 + axis] = k
        kernel = win1d.view(*shape).repeat(C, 1, *([1] * spatial_dims))
        padding = [0] * spatial_dims
        padding[axis] = pad
        out = conv(out, kernel, padding=tuple(padding), groups=C)
    return out


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    spatial_dims: int,
    window_size: int = 7,
    sigma: float = 1.5,
    data_range: float = 1.0) -> torch.Tensor:
    """高斯窗 SSIM（标量均值）。``data_range`` 为像素动态范围（minmax 归一化后≈1）。"""
    win = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    mu_p = _separable_blur(pred, win, spatial_dims)
    mu_t = _separable_blur(target, win, spatial_dims)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sigma_p2 = _separable_blur(pred * pred, win, spatial_dims) - mu_p2
    sigma_t2 = _separable_blur(target * target, win, spatial_dims) - mu_t2
    sigma_pt = _separable_blur(pred * target, win, spatial_dims) - mu_pt
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2))
    return ssim_map.mean()


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """峰值信噪比（dB），整 batch 均方误差版。"""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


# ---------------------------------------------------------------------------
# 组合回归损失
# ---------------------------------------------------------------------------
class ReconstructionLoss(nn.Module):
    """回归复原总损失：pixel + ssim_weight*(1-SSIM) + grad_weight*grad_L1。"""

    def __init__(
        self,
        spatial_dims: int,
        pixel_loss: str = "charbonnier",
        charbonnier_eps: float = 1e-3,
        ssim_weight: float = 0.0,
        ssim_window: int = 7,
        grad_weight: float = 0.0,
        data_range: float = 1.0):
        super().__init__()
        pl = str(pixel_loss).lower()
        if pl not in _PIXEL_LOSSES:
            raise ValueError(
                f"Unknown pixel loss {pl!r}; valid: {sorted(_PIXEL_LOSSES)}.")
        self.spatial_dims = int(spatial_dims)
        self.pixel_loss = pl
        self.charbonnier_eps = float(charbonnier_eps)
        self.ssim_weight = float(ssim_weight)
        self.ssim_window = int(ssim_window)
        self.grad_weight = float(grad_weight)
        self.data_range = float(data_range)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        breakdown: Dict[str, float] = None) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        pix = _PIXEL_LOSSES[self.pixel_loss](pred, target, self.charbonnier_eps)
        total = pix
        if breakdown is not None:
            breakdown["L_pixel"] = float(pix.detach().item())
        if self.ssim_weight > 0.0:
            s = ssim(pred, target, self.spatial_dims,
                     window_size=self.ssim_window, data_range=self.data_range)
            ssim_term = self.ssim_weight * (1.0 - s)
            total = total + ssim_term
            if breakdown is not None:
                breakdown["L_ssim"] = float(ssim_term.detach().item())
        if self.grad_weight > 0.0:
            g = gradient_l1(pred, target, self.spatial_dims)
            grad_term = self.grad_weight * g
            total = total + grad_term
            if breakdown is not None:
                breakdown["L_grad"] = float(grad_term.detach().item())
        return total


# ---------------------------------------------------------------------------
# 扩散损失
# ---------------------------------------------------------------------------
class DiffusionLoss(nn.Module):
    """对 ``DiffusionTrainWrapper`` 输出做逐样本加权 MSE。

    wrapper 返回 ``{"pred", "target", "weight"}``：``weight`` 为按 batch 的 σ 加权
    （EDM ``(σ²+σ_data²)/(σ·σ_data)²`` 或 DDPM 的 1）。本损失做
    ``mean(weight * (pred-target)²)``，weight 广播到逐元素。
    """

    def forward(self, out: Dict[str, torch.Tensor], breakdown: Dict[str, float] = None) -> torch.Tensor:
        pred = out["pred"].float()
        target = out["target"].float()
        weight = out["weight"].float()
        # weight: (B,) → (B, 1, 1, ...) 广播。
        w = weight.view(weight.shape[0], *([1] * (pred.ndim - 1)))
        loss = (w * (pred - target) ** 2).mean()
        if breakdown is not None:
            breakdown["L_diffusion"] = float(loss.detach().item())
        return loss


def build_recon_loss(cfg) -> ReconstructionLoss:
    """按 ``cfg`` 构造回归复原损失（读取 task.* 与 model.spatial_dims）。"""
    t = cfg.task
    data_range = 1.0 if cfg.data.normalize == "minmax" else 2.0
    return ReconstructionLoss(
        spatial_dims=int(cfg.model.spatial_dims),
        pixel_loss=str(t.recon_loss).lower(),
        charbonnier_eps=float(t.charbonnier_eps),
        ssim_weight=float(t.ssim_weight),
        ssim_window=int(t.ssim_window),
        grad_weight=float(t.grad_weight),
        data_range=data_range)


__all__ = [
    "ReconstructionLoss",
    "DiffusionLoss",
    "build_recon_loss",
    "charbonnier",
    "gradient_l1",
    "ssim",
    "psnr",
]
