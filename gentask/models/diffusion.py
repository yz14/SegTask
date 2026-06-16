"""条件扩散框架（超分）：EDM（Karras 2022）与 DDPM（Ho 2020），均以低分图为条件。

两套范式共享同一 backbone 接口 ``net(x_cat, c_noise)``：

* ``x_cat = cat([x_in, cond], dim=1)`` —— 噪声图（经各自预条件缩放）与条件图在通道维拼接；
* ``c_noise`` —— 形如 ``(B,)`` 的噪声水平标量（EDM 取 ``0.25·ln σ``，DDPM 取时间步），
  backbone 内部正弦 / Fourier 嵌入后贯穿各 ResBlock 的条件。

训练侧统一产出 ``{"pred", "target", "weight"}`` 交由 ``losses.recon.DiffusionLoss``
做逐样本加权 MSE；推理侧 ``sample(cond)`` 迭代去噪出 HR。仅空间 2D（2.5D 折叠 D 到通道）。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


def _expand(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """把 ``(B,)`` 广播到 ``(B,1,...,1)`` 以便与 ``(B,C,H,W)`` 逐元素运算。"""
    return t.reshape(t.shape[0], *([1] * (ndim - 1)))


# ============================================================================
# EDM（Karras 2022）：去噪预条件 + 对数正态 σ 采样 + Heun 二阶采样
# ============================================================================


class EDMDiffusion(nn.Module):
    """EDM 条件扩散。``net`` 须实现 ``net(x_cat, c_noise) -> F_θ``。"""

    def __init__(
        self,
        net: nn.Module,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sampler: str = "edm_heun",
        sample_steps: int = 18):
        super().__init__()
        self.net = net
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.sampler = str(sampler)
        self.sample_steps = int(sample_steps)

    # ----- 预条件 D_θ(x; σ, cond) ----------------------------------------
    def denoise(
        self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """EDM 预条件去噪：``D = c_skip·x + c_out·F(c_in·x ⊕ cond, c_noise)``。"""
        sd2 = self.sigma_data ** 2
        s = _expand(sigma, x.ndim)
        c_skip = sd2 / (s ** 2 + sd2)
        c_out = s * self.sigma_data / (s ** 2 + sd2).sqrt()
        c_in = 1.0 / (s ** 2 + sd2).sqrt()
        c_noise = 0.25 * torch.log(sigma)
        x_cat = torch.cat([c_in * x, cond], dim=1)
        f = self.net(x_cat, c_noise)
        return c_skip * x + c_out * f

    # ----- 训练 ----------------------------------------------------------
    def train_outputs(
        self, hr: torch.Tensor, cond: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """采样 σ、加噪、预条件去噪，返回逐样本加权 MSE 所需的三元组。"""
        b = hr.shape[0]
        rnd = torch.randn(b, device=hr.device, dtype=hr.dtype)
        sigma = torch.exp(self.p_mean + self.p_std * rnd)
        noise = torch.randn_like(hr) * _expand(sigma, hr.ndim)
        d = self.denoise(hr + noise, sigma, cond)
        # EDM 损失权重 λ(σ) = (σ²+σ_data²)/(σ·σ_data)²。
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return {"pred": d, "target": hr, "weight": weight}

    # ----- 采样 ----------------------------------------------------------
    def _sigma_schedule(self, device, dtype) -> torch.Tensor:
        n = self.sample_steps
        ramp = torch.linspace(0, 1, n, device=device, dtype=torch.float64)
        min_inv = self.sigma_min ** (1 / self.rho)
        max_inv = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** self.rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])  # 末尾接 0
        return sigmas.to(dtype)

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """从噪声出发、以 ``cond`` 为条件迭代去噪，返回复原 HR。"""
        sigmas = self._sigma_schedule(cond.device, cond.dtype)
        x = torch.randn(
            cond.shape, device=cond.device, dtype=cond.dtype,
            generator=generator) * sigmas[0]
        deterministic_euler = self.sampler == "ddim"
        for i in range(self.sample_steps):
            s = sigmas[i]
            s_next = sigmas[i + 1]
            sig = s.expand(cond.shape[0])
            d_cur = (x - self.denoise(x, sig, cond)) / s
            x_next = x + (s_next - s) * d_cur
            # Heun 二阶校正（除最后一步或纯 Euler 外）。
            if s_next > 0 and not deterministic_euler:
                sig_n = s_next.expand(cond.shape[0])
                d_next = (x_next - self.denoise(x_next, sig_n, cond)) / s_next
                x_next = x + (s_next - s) * 0.5 * (d_cur + d_next)
            x = x_next
        return x


# ============================================================================
# DDPM（Ho 2020）：ε-预测，方差保持；祖先 / DDIM 采样
# ============================================================================


def _make_betas(schedule: str, n: int) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, n, dtype=torch.float64)
    if schedule == "cosine":
        # Nichol & Dhariwal 2021 余弦 ᾱ schedule。
        steps = torch.arange(n + 1, dtype=torch.float64) / n
        f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        acp = f / f[0]
        betas = 1 - acp[1:] / acp[:-1]
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"unknown beta_schedule {schedule!r}")


class DDPMDiffusion(nn.Module):
    """DDPM ε-预测条件扩散。``net(x_cat, c_noise) -> ε̂``，``c_noise`` 取时间步。"""

    def __init__(
        self,
        net: nn.Module,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        sampler: str = "ddpm",
        sample_steps: int = 18,
        ddim_eta: float = 0.0):
        super().__init__()
        self.net = net
        self.num_train_timesteps = int(num_train_timesteps)
        self.sampler = str(sampler)
        self.sample_steps = int(sample_steps)
        self.ddim_eta = float(ddim_eta)
        betas = _make_betas(beta_schedule, self.num_train_timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", acp.float())

    def train_outputs(
        self, hr: torch.Tensor, cond: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """采样时间步、按 ᾱ 加噪，目标为所加噪声 ε（权重 1）。"""
        b = hr.shape[0]
        t = torch.randint(
            0, self.num_train_timesteps, (b,), device=hr.device)
        acp = self.alphas_cumprod[t]
        a = _expand(acp.sqrt(), hr.ndim)
        sig = _expand((1 - acp).sqrt(), hr.ndim)
        noise = torch.randn_like(hr)
        x_t = a * hr + sig * noise
        x_cat = torch.cat([x_t, cond], dim=1)
        eps_hat = self.net(x_cat, t.float())
        weight = torch.ones(b, device=hr.device, dtype=hr.dtype)
        return {"pred": eps_hat, "target": noise, "weight": weight}

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """以 ``cond`` 为条件，从 x_T~N(0,I) 反向去噪。支持祖先 / DDIM。"""
        device, dtype = cond.device, cond.dtype
        # 子序列时间步（均匀间隔）。
        steps = torch.linspace(
            self.num_train_timesteps - 1, 0, self.sample_steps,
            device=device).round().long()
        x = torch.randn(cond.shape, device=device, dtype=dtype, generator=generator)
        ddim = self.sampler == "ddim"
        for i, t in enumerate(steps):
            acp_t = self.alphas_cumprod[t]
            eps = self.net(
                torch.cat([x, cond], dim=1),
                t.float().expand(cond.shape[0]))
            x0 = (x - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()
            x0 = x0.clamp(-1.5, 1.5)
            t_next = steps[i + 1] if i + 1 < len(steps) else torch.tensor(-1)
            acp_next = (
                self.alphas_cumprod[t_next] if t_next >= 0
                else torch.tensor(1.0, device=device))
            if ddim:
                sigma_t = self.ddim_eta * (
                    ((1 - acp_next) / (1 - acp_t)).sqrt()
                    * (1 - acp_t / acp_next).sqrt())
                dir_xt = (1 - acp_next - sigma_t ** 2).clamp(min=0).sqrt() * eps
                x = acp_next.sqrt() * x0 + dir_xt
                if t_next >= 0 and sigma_t > 0:
                    x = x + sigma_t * torch.randn_like(x)
            else:  # 祖先采样
                if t_next >= 0:
                    beta_t = 1 - acp_t / acp_next
                    mean = (
                        acp_next.sqrt() * beta_t / (1 - acp_t) * x0
                        + (acp_t / acp_next).sqrt() * (1 - acp_next) / (1 - acp_t) * x)
                    var = beta_t * (1 - acp_next) / (1 - acp_t)
                    x = mean + var.clamp(min=0).sqrt() * torch.randn_like(x)
                else:
                    x = x0
        return x


def build_diffusion(cfg, net: nn.Module) -> nn.Module:
    """按 ``cfg.task.parameterization`` 构造 EDM / DDPM 扩散封装。"""
    t = cfg.task
    param = str(t.parameterization).lower()
    if param == "edm":
        return EDMDiffusion(
            net,
            sigma_data=t.sigma_data, p_mean=t.p_mean, p_std=t.p_std,
            sigma_min=t.sigma_min, sigma_max=t.sigma_max, rho=t.rho,
            sampler=t.sampler, sample_steps=t.sample_steps)
    if param == "ddpm_eps":
        return DDPMDiffusion(
            net,
            num_train_timesteps=t.num_train_timesteps,
            beta_schedule=t.beta_schedule,
            sampler=t.sampler, sample_steps=t.sample_steps,
            ddim_eta=t.ddim_eta)
    raise ValueError(
        f"task.parameterization must be 'edm' | 'ddpm_eps'; got {param!r}")


__all__ = [
    "EDMDiffusion", "DDPMDiffusion", "build_diffusion",
]
