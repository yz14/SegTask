"""encoder/decoder 通用 2D/3D 基础块。所有模块接受 spatial_dims（默认3）；*3D 名称仅为 API 兼容。包含工厂、ConvNormAct、attention (SE/ECA/CBAM/Coord/Gate)、BlurPool、PixelShuffle、CARAFE/DySample、3D 上/下采样。"""

from __future__ import annotations

from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# spatial_dims 分派表。
_CONV     = {2: nn.Conv2d,            3: nn.Conv3d}
_CONV_T   = {2: nn.ConvTranspose2d,   3: nn.ConvTranspose3d}
_BN       = {2: nn.BatchNorm2d,       3: nn.BatchNorm3d}
_IN       = {2: nn.InstanceNorm2d,    3: nn.InstanceNorm3d}
_DROP     = {2: nn.Dropout2d,         3: nn.Dropout3d}
_MAXPOOL  = {2: nn.MaxPool2d,         3: nn.MaxPool3d}
_AVGPOOL  = {2: nn.AvgPool2d,         3: nn.AvgPool3d}
_AAVGPOOL = {2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}
_AMAXPOOL = {2: nn.AdaptiveMaxPool2d, 3: nn.AdaptiveMaxPool3d}

#: F.interpolate 的平滑插值模式。
INTERP_SMOOTH = {2: "bilinear", 3: "trilinear"}


def _check_dims(spatial_dims: int) -> int:
    if spatial_dims not in (2, 3):
        raise ValueError(
            f"spatial_dims must be 2 or 3, got {spatial_dims!r}")
    return spatial_dims


def get_conv3d() -> Type[nn.Module]:
    """向后兼容别名：始终返回 nn.Conv3d。"""
    return nn.Conv3d


def get_conv(spatial_dims: int = 3) -> Type[nn.Module]:
    """返回对应维度的 Conv2d/Conv3d 类。"""
    return _CONV[_check_dims(spatial_dims)]


def get_norm(
    norm_type   : str,
    num_channels: int,
    num_groups  : int = 8,
    spatial_dims: int = 3) -> nn.Module:
    """nD norm：'batch' | 'instance' | 'group'（与维度无关）。"""
    d = _check_dims(spatial_dims)
    if   norm_type == "batch":
        return _BN[d](num_channels)
    elif norm_type == "instance":
        return _IN[d](num_channels, affine=True)
    elif norm_type == "group":
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm: {norm_type}")


def get_activation(name: str) -> nn.Module:
    """创建激活层：'relu' | 'leakyrelu' | 'gelu' | 'swish'。"""
    if   name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "swish":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {name}")


class ConvNormAct(nn.Module):
    """nD Conv + Norm + Activation（默认 3D）。"""

    def __init__(
        self,
        in_ch       : int,
        out_ch      : int,
        kernel_size : int = 3,
        stride      : int = 1,
        padding     : int = 1,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        activation  : str = "leakyrelu",
        dropout     : float = 0.0,
        spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.conv = _CONV[d](in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act  = get_activation(activation)
        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class SqueezeExcite3D(nn.Module):
    """nD SE (Hu 2018)：GAP → FC 压 → ReLU → FC 扩 → Sigmoid → 通道加权。"""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            _AAVGPOOL[d](1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C) -> (B, C, 1, 1[, 1])
        pat = 'b c -> b c' + ' 1' * self.spatial_dims
        scale = rearrange(self.fc(x), pat)
        return x * scale


class ECA3D(nn.Module):
    """nD ECA (Wang 2020)：用通道 1D 卷积替代 SE 的两 FC；自适应奇数核 k=|log2(C)/γ+b/γ|。"""

    def __init__(self, channels: int, k_size: int = 0, gamma: int = 2, b: int = 1,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        if k_size <= 0:
            # 自适应核尺寸，强制奇数。
            import math
            k = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
            k_size = k if k % 2 else k + 1
        self.avg = _AAVGPOOL[d](1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=k_size // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, 1[, 1]) -> (B, 1, C) -> conv1d -> (B, 1, C) -> (B, C, 1, 1[, 1])
        y = self.avg(x).flatten(1).unsqueeze(1)
        y = self.sig(self.conv(y)).squeeze(1)
        pat = 'b c -> b c' + ' 1' * self.spatial_dims
        return x * rearrange(y, pat)


# CBAM (Woo 2018)：通道 attention (MLP 于 GAP+GMP) → 空间 attention (在 avg+max cat 上的 k×k 卷积)。
class _CBAMChannelAttn(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.avg = _AAVGPOOL[d](1)
        self.max = _AMAXPOOL[d](1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # avg/max pool 后空间轴均为 1，flatten(1) 与 .view(B,C) 等价且不是 reshape。
        avg = self.mlp(self.avg(x).flatten(1))
        mx  = self.mlp(self.max(x).flatten(1))
        pat = 'b c -> b c' + ' 1' * self.spatial_dims
        w   = rearrange(torch.sigmoid(avg + mx), pat)
        return x * w


class _CBAMSpatialAttn(nn.Module):
    def __init__(self, kernel_size: int = 7, spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        if kernel_size % 2 == 0:
            raise ValueError("CBAM spatial kernel must be odd.")
        self.conv = _CONV[d](2, 1, kernel_size=kernel_size,
                             padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM3D(nn.Module):
    """nD CBAM（通道→空间）。"""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7, spatial_dims: int = 3):
        super().__init__()
        self.channel = _CBAMChannelAttn(channels, reduction,
                                        spatial_dims=spatial_dims)
        self.spatial = _CBAMSpatialAttn(spatial_kernel,
                                        spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# Coordinate Attention (Hou 2021)：逐轴 pool → 共享 MLP → 逐轴加权；适合细长结构（血管/气道/脉柱）。
class CoordAttention3D(nn.Module):
    """nD Coord Attention（3D 中逐 D/H/W 轴 pool）。"""

    def __init__(self, channels: int, reduction: int = 32,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        mid = max(channels // reduction, 8)

        # 每个 pool 保留一个轴，其余压为 1。
        self.pools = nn.ModuleList([
            _AAVGPOOL[d](self._axis_pool_size(d, axis))
            for axis in range(d)
        ])

        # 共享 bottleneck 卷积，作用于列拼后的 rank-(d+2) 张量（首空间轴为拼接轴）。
        self.conv1 = _CONV[d](channels, mid, kernel_size=1, bias=False)
        self.norm1 = _BN[d](mid)
        self.act = nn.Hardswish(inplace=True)

        # 逐轴输出卷积：mid→channels。
        self.axis_convs = nn.ModuleList([
            _CONV[d](mid, channels, kernel_size=1, bias=False)
            for _ in range(d)
        ])

    @staticmethod
    def _axis_pool_size(spatial_dims: int, keep_axis: int) -> Tuple:
        """保留 keep_axis (None)，其余轴压为 1。"""
        return tuple(None if i == keep_axis else 1 for i in range(spatial_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.spatial_dims
        B, C = x.shape[:2]
        sizes = list(x.shape[2:])

        # 逐轴 pool → 把保留轴移到首空间位 → cat 供共享 1×1。
        descriptors = []
        for axis in range(d):
            p = self.pools[axis](x)  # 保留该轴，其余为 1
            # p 空间 shape 除 axis 轴 均 1；flatten(2) 集成 (B,C,size) 后补 (d-1) 个单位轴。
            tail_pat = 'b c s -> b c s' + ' 1' * (d - 1)
            descriptors.append(rearrange(p.flatten(2), tail_pat))
        y = torch.cat(descriptors, dim=2)
        y = self.act(self.norm1(self.conv1(y)))
        y_axes = torch.split(y, sizes, dim=2)

        out = x
        for axis in range(d):
            # 从 (B,C,size,1,...,1) 转为 (B,C,1,...,size@axis,...,1)。
            a = torch.sigmoid(self.axis_convs[axis](y_axes[axis])).flatten(2)
            tail = ['1'] * d
            tail[axis] = 's'
            tail_pat = 'b c s -> b c ' + ' '.join(tail)
            out = out * rearrange(a, tail_pat)
        return out


def make_attention(name: str, channels: int,
                   spatial_dims: int = 3, **kwargs) -> nn.Module:
    """Attention 工厂：'none'/'se'/'eca'/'cbam'/'coord'。"""
    name = (name or "none").lower()
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SqueezeExcite3D(channels, reduction=kwargs.get("reduction", 16),
                               spatial_dims=spatial_dims)
    if name == "eca":
        return ECA3D(channels, spatial_dims=spatial_dims)
    if name == "cbam":
        return CBAM3D(channels, reduction=kwargs.get("reduction", 16),
                      spatial_dims=spatial_dims)
    if name == "coord":
        return CoordAttention3D(channels, reduction=kwargs.get("reduction", 32),
                                spatial_dims=spatial_dims)
    raise ValueError(
        f"Unknown attention type: {name!r}. "
        f"Valid: none|se|eca|cbam|coord")


ATTENTION_TYPES = ("none", "se", "eca", "cbam", "coord")


class AttentionGate3D(nn.Module):
    """UNet skip 加性 attention gate (Oktay 2018)：1×1→ReLU→1×1→sigmoid。inter 默认 x_ch//2。"""

    def __init__(self, x_ch: int, g_ch: int, inter: int = 0,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        if inter <= 0:
            inter = max(x_ch // 2, 1)
        self.W_x = nn.Sequential(
            _CONV[d](x_ch, inter, kernel_size=1, bias=False),
            _BN[d](inter),
        )
        self.W_g = nn.Sequential(
            _CONV[d](g_ch, inter, kernel_size=1, bias=False),
            _BN[d](inter),
        )
        self.psi = nn.Sequential(
            _CONV[d](inter, 1, kernel_size=1, bias=False),
            _BN[d](1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """用解码信号 g 门控 skip x；必要时将 g 重采样到 x 尺寸。"""
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(
                g, size=x.shape[2:],
                mode=INTERP_SMOOTH[self.spatial_dims],
                align_corners=False)
        return x * self.psi(self.relu(self.W_x(x) + self.W_g(g)))


class BlurPool3d(nn.Module):
    """抗混叠下采样 (Zhang 2019)：二项式低通 [1,2,1] (filt=3) 或 [1,4,6,4,1] (filt=5)。"""

    _BINOMIAL: dict = {
        2: (1., 1.),
        3: (1., 2., 1.),
        5: (1., 4., 6., 4., 1.),
    }

    def __init__(self, channels: int, stride: int = 2, filt_size: int = 3,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        if filt_size not in self._BINOMIAL:
            raise ValueError(f"Unsupported BlurPool filt_size: {filt_size}")
        self.channels = channels
        self.stride = stride
        self.pad = filt_size // 2

        a = torch.tensor(self._BINOMIAL[filt_size], dtype=torch.float32)
        # nD 可分离核：逐次外积。
        kernel = a
        for _ in range(d - 1):
            kernel = kernel.unsqueeze(-1) * a
        kernel = kernel / kernel.sum()
        # 补 (out_ch=channels, in_ch_per_group=1) 领轴。
        kernel = kernel[None, None].expand(
            channels, 1, *kernel.shape).contiguous()
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # replicate padding 保留边界统计。
        if self.pad:
            x = F.pad(x, [self.pad] * (2 * self.spatial_dims), mode="replicate")
        if self.spatial_dims == 3:
            return F.conv3d(
                x, self.kernel,
                stride=self.stride, padding=0, groups=self.channels)
        return F.conv2d(
            x, self.kernel,
            stride=self.stride, padding=0, groups=self.channels)


# Sub-pixel 操作（ESPCN 风）：nD reshape+permute，无损无参。
class PixelUnshuffle3d(nn.Module):
    """nD space-to-depth：(B,C,r*s0,r*s1,...) → (B,C*r^d,s0,s1,...)。"""

    def __init__(self, r: int = 2, spatial_dims: int = 3):
        super().__init__()
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        self.r = r
        self.spatial_dims = _check_dims(spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.spatial_dims
        r = self.r
        if x.ndim != d + 2:
            raise ValueError(
                f"PixelUnshuffle3d(spatial_dims={d}) expects rank-{d+2} input, "
                f"got {x.ndim}")
        B, C = x.shape[:2]
        spatial = list(x.shape[2:])
        for s in spatial:
            if s % r:
                raise ValueError(
                    f"PixelUnshuffle3d(r={r}) needs spatial dims divisible "
                    f"by r, got {tuple(spatial)}")
        # 每个空间轴拆为 (size/r, r) 交错，再将全部 r-axes 捏入通道。
        if d == 2:
            return rearrange(
                x, 'b c (h r1) (w r2) -> b (c r1 r2) h w', r1=r, r2=r)
        return rearrange(
            x, 'b c (h r1) (w r2) (z r3) -> b (c r1 r2 r3) h w z',
            r1=r, r2=r, r3=r)


class PixelShuffle3d(nn.Module):
    """nD depth-to-space；PixelUnshuffle3d 的逆。"""

    def __init__(self, r: int = 2, spatial_dims: int = 3):
        super().__init__()
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        self.r = r
        self.spatial_dims = _check_dims(spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.spatial_dims
        r = self.r
        if x.ndim != d + 2:
            raise ValueError(
                f"PixelShuffle3d(spatial_dims={d}) expects rank-{d+2} input, "
                f"got {x.ndim}")
        B, Crd = x.shape[:2]
        spatial = list(x.shape[2:])
        rd = r ** d
        if Crd % rd:
            raise ValueError(
                f"PixelShuffle3d(r={r}, spatial_dims={d}) needs channels "
                f"divisible by r^d={rd}, got C={Crd}")
        C = Crd // rd
        # 从 (B, C*r^d, *spatial) 拆出 r-axes 并与原空间轴交错。
        if d == 2:
            return rearrange(
                x, 'b (c r1 r2) h w -> b c (h r1) (w r2)', r1=r, r2=r)
        return rearrange(
            x, 'b (c r1 r2 r3) h w z -> b c (h r1) (w r2) (z r3)',
            r1=r, r2=r, r3=r)


def icnr_init_(weight: torch.Tensor, upscale: int,
               spatial_dims: int = 3,
               init: Type[nn.Module] = None) -> None:
    """ICNR init (Aitken 2017)：conv+PixelShuffle 初始近似最近邻上采样。"""
    d = _check_dims(spatial_dims)
    rd = upscale ** d
    out_total = weight.shape[0]
    if out_total % rd != 0:
        raise ValueError("ICNR: out_ch must be divisible by r^d")
    out_ch = out_total // rd
    sub = torch.empty(out_ch, *weight.shape[1:], device=weight.device,
                      dtype=weight.dtype)
    nn.init.kaiming_normal_(sub)
    # 每个滤波器复制 r^d 次，子像素同源 → NN 上采样。
    weight.data.copy_(sub.repeat_interleave(rd, dim=0))


class Downsample(nn.Module):
    """×2 下采样 + in_ch→out_ch 投影 + norm。模式：'conv' 带步长 、'maxpool'/'avgpool'/'blurpool' (+1×1)、'pixelunshuffle'(s2d+1×1)。"""

    VALID_MODES = ("conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle")

    def __init__(
        self,
        in_ch       : int,
        out_ch      : int,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        mode        : str = "conv",
        spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown downsample mode: {mode}. "
                f"Valid: {self.VALID_MODES}")
        self.mode         = mode
        self.spatial_dims = d

        if   mode == "conv":
            self.op = _CONV[d](in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        elif mode == "maxpool":
            self.op = nn.Sequential(
                _MAXPOOL[d](kernel_size=2, stride=2),
                _CONV[d](in_ch, out_ch, kernel_size=1, bias=False))
        elif mode == "avgpool":
            self.op = nn.Sequential(
                _AVGPOOL[d](kernel_size=2, stride=2),
                _CONV[d](in_ch, out_ch, kernel_size=1, bias=False))
        elif mode == "blurpool":
            self.op = nn.Sequential(
                BlurPool3d(in_ch, stride=2, filt_size=3, spatial_dims=d),
                _CONV[d](in_ch, out_ch, kernel_size=1, bias=False))
        else:  # pixelunshuffle：r=2，通道×2^d。
            channel_mult = 2 ** d
            self.op = nn.Sequential(
                PixelUnshuffle3d(r=2, spatial_dims=d),
                _CONV[d](in_ch * channel_mult, out_ch, kernel_size=1, bias=False))

        self.norm = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.op(x))


class CARAFE3d(nn.Module):
    """3D CARAFE (Wang ICCV 2019)：预测逐体素重装配核，局部性优于 pixelshuffle/trilinear。默认 scale=2,k_up=3（3D 中 k_up=5 过重）。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        scale: int = 2,
        k_up: int = 3,
        k_enc: int = 3,
        c_mid: int = 64,
    ):
        super().__init__()
        if scale < 1 or k_up < 1 or k_enc < 1:
            raise ValueError("CARAFE3d: scale/k_up/k_enc must be >= 1")
        self.scale = scale
        self.k_up = k_up
        self.pad = k_up // 2

        # Bottleneck → 内容编码器（预测 scale^3 · k_up^3 核 logits/voxel）。
        self.compress = nn.Conv3d(in_ch, c_mid, kernel_size=1)
        self.encode = nn.Conv3d(
            c_mid, (scale ** 3) * (k_up ** 3),
            kernel_size=k_enc, padding=k_enc // 2)
        self.shuffle = PixelShuffle3d(r=scale)
        self.proj = (nn.Conv3d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        s = self.scale
        k = self.k_up

        # 1) 高分辨率上预测重装配核 + softmax。
        w = self.compress(x)
        w = self.encode(w)                         # (B, s^3·k^3, D, H, W)
        w = self.shuffle(w)                        # (B, k^3, sD, sH, sW)
        w = F.softmax(w, dim=1)

        # 2) 取 k^3 邻域 patch。
        x_pad = F.pad(x, [self.pad] * 6, mode="replicate")
        x_unf = (x_pad
                 .unfold(2, k, 1).unfold(3, k, 1).unfold(4, k, 1)
                 .contiguous())
        # unfold 输出末尾为 3 个 k 轴；将其捏入 (C,k^3) 通道。
        x_unf = rearrange(
            x_unf, 'b c d h w k1 k2 k3 -> b (c k1 k2 k3) d h w')

        # 3) 最近邻上采 patch，4) 沿 k^3 轴加权求和。
        x_up = F.interpolate(x_unf, scale_factor=s, mode="nearest")
        x_up = rearrange(
            x_up, 'b (c kk) d h w -> b c kk d h w', c=C)
        out = (x_up * w.unsqueeze(1)).sum(dim=2)
        return self.proj(out)


class DySample3d(nn.Module):
    """3D DySample (Liu ICCV 2023)：预测 grid_sample 偏移，轻于 CARAFE；偏移近 0 初始 ≈ 双线性上采样。默认 scale=2,groups=4,dyscope=True。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        dyscope: bool = True,
    ):
        super().__init__()
        if in_ch % groups != 0:
            raise ValueError(
                f"DySample3d: in_ch({in_ch}) must be divisible by groups({groups})")
        self.scale = scale
        self.groups = groups
        self.dyscope = dyscope

        off_ch = 3 * groups * (scale ** 3)  # 3 坐标 × groups × s^3
        self.offset = nn.Conv3d(in_ch, off_ch, kernel_size=1)
        nn.init.trunc_normal_(self.offset.weight, std=1e-3)
        nn.init.zeros_(self.offset.bias)

        if dyscope:
            self.scope = nn.Conv3d(in_ch, off_ch, kernel_size=1)
            nn.init.zeros_(self.scope.weight)
            nn.init.zeros_(self.scope.bias)

        self.shuffle = PixelShuffle3d(r=scale)
        self.proj = (nn.Conv3d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    @staticmethod
    def _normalised_grid(D: int, H: int, W: int,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """grid_sample 的基础网格 (x,y,z)，范围 [-1,1]。"""
        zs = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1)    # (D, H, W, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        s = self.scale
        g = self.groups
        Du, Hu, Wu = D * s, H * s, W * s

        # 1) 低分辨率预测偏移，shuffle 到高分辨率。
        off = self.offset(x)
        if self.dyscope:
            # DySample-S：可学 scope gate，范围 [0, 0.5]。
            off = off * self.scope(x).sigmoid() * 0.5
        off = self.shuffle(off)
        # (B, g*3, Du, Hu, Wu) → (B, g, Du, Hu, Wu, 3)。
        off = rearrange(
            off, 'b (g c) d h w -> b g d h w c', g=g, c=3)

        # 2) 偏移归一化（像素→grid 坐标；末轴顺序 x,y,z = W,H,D）。
        norm = torch.tensor(
            [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1), 2.0 / max(D - 1, 1)],
            device=x.device, dtype=x.dtype)
        off = off * norm

        # 3) 基础网格+偏移，4) 分组 grid_sample（合并 (B,g) 为 batch）。
        base = self._normalised_grid(Du, Hu, Wu, x.device, x.dtype)
        coord = base.unsqueeze(0).unsqueeze(0) + off
        x_g = rearrange(
            x, 'b (g c) d h w -> (b g) c d h w', g=g)
        coord = rearrange(coord, 'b g d h w c -> (b g) d h w c')
        out = F.grid_sample(
            x_g, coord, mode="bilinear",
            padding_mode="border", align_corners=True)
        out = rearrange(
            out, '(b g) c d h w -> b (g c) d h w', g=g)
        return self.proj(out)


class Upsample(nn.Module):
    """×2 上采样 + in_ch→out_ch 投影。模式：'transpose' 、'trilinear'/'nearest'(插值+3×3精修)、'pixelshuffle'(子像素+ICNR)、'carafe'/'dysample' 仅 3D。"""

    VALID_MODES = ("transpose", "trilinear", "nearest", "pixelshuffle",
                   "carafe", "dysample")
    _MODES_3D_ONLY = ("carafe", "dysample")

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mode: str = "transpose",
        spatial_dims: int = 3,
    ):
        super().__init__()
        d = _check_dims(spatial_dims)
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown upsample mode: {mode}. Valid: {self.VALID_MODES}")
        if d == 2 and mode in self._MODES_3D_ONLY:
            raise ValueError(
                f"Upsample mode {mode!r} is only supported for spatial_dims=3."
                f" For 2D, use one of: transpose | trilinear | nearest |"
                f" pixelshuffle.")
        self.mode = mode
        self.spatial_dims = d

        if mode == "transpose":
            self.up = _CONV_T[d](in_ch, out_ch, kernel_size=2, stride=2)
        elif mode in ("trilinear", "nearest"):
            self.up = _CONV[d](in_ch, out_ch, kernel_size=3, padding=1,
                               bias=False)
        elif mode == "pixelshuffle":
            channel_mult = 2 ** d
            self.expand = _CONV[d](in_ch, out_ch * channel_mult,
                                   kernel_size=1, bias=False)
            self.shuffle = PixelShuffle3d(r=2, spatial_dims=d)
            icnr_init_(self.expand.weight, upscale=2, spatial_dims=d)
        elif mode == "carafe":
            self.up = CARAFE3d(in_ch, out_ch, scale=2, k_up=3, k_enc=3, c_mid=64)
        else:  # dysample
            groups = _choose_groups(in_ch, preferred=4)
            self.up = DySample3d(in_ch, out_ch, scale=2,
                                 groups=groups, dyscope=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.up(x)
        if self.mode == "trilinear":
            x = F.interpolate(
                x, scale_factor=2,
                mode=INTERP_SMOOTH[self.spatial_dims],
                align_corners=False)
            return self.up(x)
        if self.mode == "nearest":
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            return self.up(x)
        if self.mode == "pixelshuffle":
            return self.shuffle(self.expand(x))
        # carafe / dysample（__init__ 已限 3D）。
        return self.up(x)


def _choose_groups(in_ch: int, preferred: int = 4) -> int:
    """不超过 preferred 的 in_ch 最大因子。"""
    for g in range(min(preferred, in_ch), 0, -1):
        if in_ch % g == 0:
            return g
    return 1
