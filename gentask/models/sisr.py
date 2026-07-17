"""经典 SISR backbone：EDSR / RCAN + post-upsampling 上采头（H7+H8）。

与 pre-upsampling（U-Net 回归，输入=已插回 HR 网格的 LR）不同，本模块采用
post-upsampling 设定：网络在 **真 LR 网格** 上做特征提取（省算力），最后由
PixelShuffle+ICNR 上采头一次性放大到 HR 网格。支持 **各向异性倍率**
（``sr_scale_per_axis``，如 z-SISR 的 ``[s,1,1]`` 或 2.5D 面内的 ``[s,s]``）。

* EDSR（Lim 2017）：去 BN 的残差块 ``conv-act-conv``，块输出乘 ``res_scale``；
* RCAN（Zhang 2018）：RCAB（残差块 + 通道注意力）× 组内块数，再套残差组
  （组尾 conv + 组级跳连），最外全局跳连。

配套 ``SuperResDegradation(keep_lr_size=True)``：训练时 degrade 直接产出真
LR 尺寸的输入（不上采回 HR），net 输出即 HR patch。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from taskcore.models.blocks import SqueezeExcite3D, get_activation, get_conv


class AnisoPixelShuffle(nn.Module):
    """各向异性 PixelShuffle：逐轴因子 ``factors=(r_1..r_d)``。

    输入 ``(B, C*prod(r), *S)`` → 输出 ``(B, C, *[s_i*r_i])``。各轴因子可不同
    （含 1），子像素相位排布与标准 PixelShuffle 逐轴一致。
    """

    def __init__(self, factors: Sequence[int]):
        super().__init__()
        self.factors = tuple(int(r) for r in factors)
        if any(r < 1 for r in self.factors):
            raise ValueError(f"factors must be >= 1; got {self.factors}")
        self.prod = 1
        for r in self.factors:
            self.prod *= r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = len(self.factors)
        if x.ndim != d + 2:
            raise ValueError(
                f"AnisoPixelShuffle(d={d}) expects rank-{d + 2}; got {x.ndim}")
        b, ctot = x.shape[:2]
        if ctot % self.prod != 0:
            raise ValueError(
                f"channels {ctot} not divisible by prod(factors)={self.prod}")
        c = ctot // self.prod
        sizes = list(x.shape[2:])
        # (B, C, r_1..r_d, s_1..s_d) → (B, C, s_1, r_1, ..., s_d, r_d)
        x = x.view(b, c, *self.factors, *sizes)
        perm = [0, 1]
        for i in range(d):
            perm += [2 + d + i, 2 + i]
        x = x.permute(*perm).contiguous()
        return x.view(b, c, *[s * r for s, r in zip(sizes, self.factors)])


def aniso_icnr_init_(weight: torch.Tensor, prod: int) -> None:
    """各向异性 ICNR：同一子像素组共享滤波器 → 初始等价最近邻上采样。"""
    if weight.shape[0] % prod != 0:
        raise ValueError("ICNR: out_ch must be divisible by prod(factors)")
    sub = torch.empty(weight.shape[0] // prod, *weight.shape[1:],
                      device=weight.device, dtype=weight.dtype)
    nn.init.kaiming_normal_(sub)
    weight.data.copy_(sub.repeat_interleave(prod, dim=0))


class UpsampleHead(nn.Module):
    """post-upsampling 头：conv(C→C*prod(r)) + AnisoPixelShuffle + ICNR。

    倍率全 1 时为恒等（同尺寸复原退化，如 denoise-only）。
    """

    def __init__(self, channels: int, factors: Sequence[int],
                 spatial_dims: int = 3):
        super().__init__()
        self.factors = tuple(int(r) for r in factors)
        if all(r == 1 for r in self.factors):
            self.body = nn.Identity()
            return
        conv = get_conv(spatial_dims)
        shuffle = AnisoPixelShuffle(self.factors)
        expand = conv(channels, channels * shuffle.prod, 3, padding=1)
        aniso_icnr_init_(expand.weight, shuffle.prod)
        nn.init.zeros_(expand.bias)
        self.body = nn.Sequential(expand, shuffle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EDSRBlock(nn.Module):
    """EDSR 残差块：conv-act-conv（无 norm），输出乘 ``res_scale`` 后加跳连。"""

    def __init__(self, channels: int, activation: str = "relu",
                 res_scale: float = 1.0, spatial_dims: int = 3):
        super().__init__()
        conv = get_conv(spatial_dims)
        self.body = nn.Sequential(
            conv(channels, channels, 3, padding=1),
            get_activation(activation),
            conv(channels, channels, 3, padding=1))
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class RCAB(nn.Module):
    """RCAN 残差通道注意力块：conv-act-conv + CA（SE 式）+ 跳连。"""

    def __init__(self, channels: int, activation: str = "relu",
                 reduction: int = 16, spatial_dims: int = 3):
        super().__init__()
        conv = get_conv(spatial_dims)
        self.body = nn.Sequential(
            conv(channels, channels, 3, padding=1),
            get_activation(activation),
            conv(channels, channels, 3, padding=1),
            SqueezeExcite3D(channels, reduction=reduction,
                            spatial_dims=spatial_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """RCAN 残差组：RCAB × n + 组尾 conv + 组级跳连。"""

    def __init__(self, channels: int, num_blocks: int, activation: str = "relu",
                 reduction: int = 16, spatial_dims: int = 3):
        super().__init__()
        conv = get_conv(spatial_dims)
        self.body = nn.Sequential(
            *[RCAB(channels, activation, reduction, spatial_dims)
              for _ in range(num_blocks)],
            conv(channels, channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SISRNet(nn.Module):
    """EDSR / RCAN 风格 SISR 网络：head conv → body（全局跳连）→ 上采头 → tail。

    输入 ``(B, in_ch, *LR)``，输出 ``(B, out_ch, *[LR_i*r_i])``。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factors: Sequence[int],
        arch: str = "edsr",
        channels: int = 64,
        num_blocks: int = 16,
        num_groups: int = 10,
        res_scale: float = 1.0,
        activation: str = "relu",
        se_reduction: int = 16,
        spatial_dims: int = 3):
        super().__init__()
        conv = get_conv(spatial_dims)
        arch = str(arch).lower()
        self.head = conv(in_channels, channels, 3, padding=1)
        if arch == "edsr":
            blocks = [EDSRBlock(channels, activation, res_scale, spatial_dims)
                      for _ in range(num_blocks)]
        elif arch == "rcan":
            blocks = [ResidualGroup(channels, num_blocks, activation,
                                    se_reduction, spatial_dims)
                      for _ in range(num_groups)]
        else:
            raise ValueError(f"Unknown SISR arch: {arch!r}")
        self.body = nn.Sequential(
            *blocks, conv(channels, channels, 3, padding=1))
        self.upsample = UpsampleHead(channels, factors, spatial_dims)
        self.tail = conv(channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.head(x)
        f = f + self.body(f)  # 全局跳连（EDSR/RCAN 长跳）
        return self.tail(self.upsample(f))

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": 0, "decoder": 0, "total": total}


__all__ = ["SISRNet", "AnisoPixelShuffle", "UpsampleHead",
           "EDSRBlock", "RCAB", "ResidualGroup"]
