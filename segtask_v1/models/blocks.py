"""Common 2D/3D building blocks for encoder/decoder.

All modules accept ``spatial_dims`` (default 3); ``*3D`` names are kept for
API stability but support 2D via ``spatial_dims=2``.

Provides:
  Factories (conv/norm/act/pool/drop), ConvNormAct,
  Attention: SE, ECA, CBAM, CoordAttention, AttentionGate; ``make_attention`` factory,
  BlurPool, PixelShuffle/Unshuffle, CARAFE (3D-only), DySample (3D-only),
  Downsample (conv/maxpool/avgpool/blurpool/pixelunshuffle),
  Upsample (transpose/trilinear/nearest/pixelshuffle/carafe/dysample).
"""

from __future__ import annotations

from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer factories — spatial_dims dispatch tables
# ---------------------------------------------------------------------------
_CONV     = {2: nn.Conv2d,            3: nn.Conv3d}
_CONV_T   = {2: nn.ConvTranspose2d,   3: nn.ConvTranspose3d}
_BN       = {2: nn.BatchNorm2d,       3: nn.BatchNorm3d}
_IN       = {2: nn.InstanceNorm2d,    3: nn.InstanceNorm3d}
_DROP     = {2: nn.Dropout2d,         3: nn.Dropout3d}
_MAXPOOL  = {2: nn.MaxPool2d,         3: nn.MaxPool3d}
_AVGPOOL  = {2: nn.AvgPool2d,         3: nn.AvgPool3d}
_AAVGPOOL = {2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}
_AMAXPOOL = {2: nn.AdaptiveMaxPool2d, 3: nn.AdaptiveMaxPool3d}

#: smooth (linear) interpolation mode for ``F.interpolate`` per spatial_dims.
INTERP_SMOOTH = {2: "bilinear", 3: "trilinear"}


def _check_dims(spatial_dims: int) -> int:
    if spatial_dims not in (2, 3):
        raise ValueError(
            f"spatial_dims must be 2 or 3, got {spatial_dims!r}")
    return spatial_dims


def get_conv3d() -> Type[nn.Module]:
    """Back-compat alias — returns ``nn.Conv3d`` unconditionally."""
    return nn.Conv3d


def get_conv(spatial_dims: int = 3) -> Type[nn.Module]:
    """Return the conv class (`Conv2d`/`Conv3d`) for the given dim."""
    return _CONV[_check_dims(spatial_dims)]


def get_norm(
    norm_type   : str,
    num_channels: int,
    num_groups  : int = 8,
    spatial_dims: int = 3) -> nn.Module:
    """nD norm: ``batch`` / ``instance`` (per spatial_dims) | ``group`` (dim-agnostic)."""
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
    """Create an activation layer."""
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


# ---------------------------------------------------------------------------
# Conv + Norm + Activation
# ---------------------------------------------------------------------------
class ConvNormAct(nn.Module):
    """nD convolution + normalization + activation (default 3D)."""

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
        self.conv = _CONV[d](
            in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)
        self.act  = get_activation(activation)
        self.drop = _DROP[d](dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (channel attention)
# ---------------------------------------------------------------------------
class SqueezeExcite3D(nn.Module):
    """nD Squeeze-and-Excitation (Hu 2018): GAP → FC reduce → ReLU → FC expand → Sigmoid → scale."""

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
        scale = self.fc(x).view(x.shape[0], x.shape[1],
                                *([1] * self.spatial_dims))
        return x * scale


# ECA-Net (Wang 2020): 1D conv over channels in place of SE's two FC layers;
# adaptive kernel k = |log2(C)/gamma + b/gamma|_odd.
class ECA3D(nn.Module):
    """nD Efficient Channel Attention (Wang 2020)."""

    def __init__(self, channels: int, k_size: int = 0, gamma: int = 2, b: int = 1,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        if k_size <= 0:
            # Adaptive kernel size; force odd.
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
        return x * y.view(y.size(0), y.size(1), *([1] * self.spatial_dims))


# CBAM (Woo 2018): channel attention (MLP over GAP+GMP) → spatial attention
# (k×k conv over channel-wise avg+max concat).
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
        B, C = x.shape[:2]
        avg = self.mlp(self.avg(x).view(B, C))
        mx  = self.mlp(self.max(x).view(B, C))
        w = torch.sigmoid(avg + mx).view(B, C, *([1] * self.spatial_dims))
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
    """nD Convolutional Block Attention Module (channel → spatial)."""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7, spatial_dims: int = 3):
        super().__init__()
        self.channel = _CBAMChannelAttn(channels, reduction,
                                        spatial_dims=spatial_dims)
        self.spatial = _CBAMSpatialAttn(spatial_kernel,
                                        spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# Coordinate Attention (Hou 2021): per-axis pools → shared MLP → axis-wise scale.
# Effective on elongated structures (vessels / airways / spine).
class CoordAttention3D(nn.Module):
    """nD Coordinate Attention (Hou 2021); generalised to per-axis pools (D/H/W in 3D)."""

    def __init__(self, channels: int, reduction: int = 32,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        mid = max(channels // reduction, 8)

        # Per-axis pool: each pool keeps one spatial axis full and collapses
        # the others to size 1. The output_size tuple has length == d.
        self.pools = nn.ModuleList([
            _AAVGPOOL[d](self._axis_pool_size(d, axis))
            for axis in range(d)
        ])

        # Shared bottleneck conv (operates on a column-stacked rank-(d+2)
        # tensor where the FIRST spatial axis is the concatenation axis).
        self.conv1 = _CONV[d](channels, mid, kernel_size=1, bias=False)
        self.norm1 = _BN[d](mid)
        self.act = nn.Hardswish(inplace=True)

        # Per-axis output conv: maps mid -> channels, applied on the slice
        # corresponding to that axis.
        self.axis_convs = nn.ModuleList([
            _CONV[d](mid, channels, kernel_size=1, bias=False)
            for _ in range(d)
        ])

    @staticmethod
    def _axis_pool_size(spatial_dims: int, keep_axis: int) -> Tuple:
        """Pool tuple keeping ``keep_axis`` full (None) and others size 1."""
        return tuple(None if i == keep_axis else 1 for i in range(spatial_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.spatial_dims
        B, C = x.shape[:2]
        sizes = list(x.shape[2:])

        # Per-axis pool → reshape kept axis to first spatial position → cat for shared 1×1.
        descriptors = []
        for axis in range(d):
            p = self.pools[axis](x)  # keeps axis full, others 1
            # Move axis to the first spatial position.
            new_shape = [B, C, sizes[axis]] + [1] * (d - 1)
            descriptors.append(p.reshape(new_shape))
        y = torch.cat(descriptors, dim=2)
        y = self.act(self.norm1(self.conv1(y)))
        y_axes = torch.split(y, sizes, dim=2)

        out = x
        for axis in range(d):
            # Broadcast back to (B, C, 1, ..., size_axis, ..., 1).
            broadcast_shape = [B, C] + [1] * d
            broadcast_shape[2 + axis] = sizes[axis]
            a = torch.sigmoid(self.axis_convs[axis](y_axes[axis])).reshape(
                broadcast_shape)
            out = out * a
        return out


# ---------------------------------------------------------------------------
# Unified channel-attention factory.
# ---------------------------------------------------------------------------
def make_attention(name: str, channels: int,
                   spatial_dims: int = 3, **kwargs) -> nn.Module:
    """Attention factory: ``none``|``se``|``eca``|``cbam``|``coord``."""
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


# Attention Gate (Oktay 2018): gates encoder skip ``x`` by decoder signal ``g``
# via additive 1×1 → ReLU → 1×1 → sigmoid mask. Standard for medical UNets.
class AttentionGate3D(nn.Module):
    """Additive attention gate for UNet skips (Oktay 2018); ``inter`` defaults to ``x_ch // 2``."""

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
        """Gate skip ``x`` by decoder signal ``g``; resizes ``g`` to ``x`` if shapes differ."""
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(
                g, size=x.shape[2:],
                mode=INTERP_SMOOTH[self.spatial_dims],
                align_corners=False)
        return x * self.psi(self.relu(self.W_x(x) + self.W_g(g)))


# Anti-aliased downsample (Zhang 2019): fixed binomial low-pass before stride.
class BlurPool3d(nn.Module):
    """nD anti-aliased blur+stride; binomial kernels [1,2,1] (filt=3) or [1,4,6,4,1] (filt=5)."""

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
        # nD separable kernel via outer products.
        kernel = a
        for _ in range(d - 1):
            kernel = kernel.unsqueeze(-1) * a  # iterative outer product
        kernel = kernel / kernel.sum()
        # Add (out_ch=channels, in_ch_per_group=1) leading dims.
        kernel = kernel[None, None].expand(
            channels, 1, *kernel.shape).contiguous()
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate padding to preserve boundary stats.
        if self.pad:
            x = F.pad(x, [self.pad] * (2 * self.spatial_dims), mode="replicate")
        if self.spatial_dims == 3:
            return F.conv3d(
                x, self.kernel,
                stride=self.stride, padding=0, groups=self.channels)
        return F.conv2d(
            x, self.kernel,
            stride=self.stride, padding=0, groups=self.channels)


# Sub-pixel ops (ESPCN-style): nD reshape+permute, lossless and parameter-free.
class PixelUnshuffle3d(nn.Module):
    """nD space-to-depth: ``(B, C, r*s0, r*s1, ...) → (B, C*r^d, s0, s1, ...)``."""

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
        # View each spatial axis as (size/r, r), interleaved.
        view_shape = [B, C]
        for s in spatial:
            view_shape.extend([s // r, r])
        y = x.view(view_shape)
        # Permute B, C, all r-axes (odd), all size/r-axes (even). 3D: (0,1,3,5,7,2,4,6).
        r_axes = [2 + 2 * i + 1 for i in range(d)]
        s_axes = [2 + 2 * i     for i in range(d)]
        y = y.permute(0, 1, *r_axes, *s_axes).contiguous()
        return y.view(B, C * (r ** d), *(s // r for s in spatial))


class PixelShuffle3d(nn.Module):
    """nD depth-to-space; inverse of :class:`PixelUnshuffle3d`."""

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
        # View: (B, C, r,...,r, s0,...,sd) — d r-axes then d size-axes.
        view_shape = [B, C] + [r] * d + spatial
        y = x.view(view_shape)
        # Interleave (size_axis, r_axis) per spatial dim. 3D ref: (0,1,5,2,6,3,7,4).
        perm = [0, 1]
        for i in range(d):
            perm.append(2 + d + i)   # size axis i
            perm.append(2 + i)       # r axis i
        y = y.permute(perm).contiguous()
        return y.view(B, C, *(s * r for s in spatial))


def icnr_init_(weight: torch.Tensor, upscale: int,
               spatial_dims: int = 3,
               init: Type[nn.Module] = None) -> None:
    """ICNR init (Aitken 2017): conv→PixelShuffle starts ≈ nearest-neighbour upsample."""
    d = _check_dims(spatial_dims)
    rd = upscale ** d
    out_total = weight.shape[0]
    if out_total % rd != 0:
        raise ValueError("ICNR: out_ch must be divisible by r^d")
    out_ch = out_total // rd
    sub = torch.empty(out_ch, *weight.shape[1:], device=weight.device,
                      dtype=weight.dtype)
    nn.init.kaiming_normal_(sub)
    # Replicate each filter r^d times so sub-pixel siblings start identical → NN upsample.
    weight.data.copy_(sub.repeat_interleave(rd, dim=0))


# ---------------------------------------------------------------------------
# Downsampling (multi-mode, factor 2)
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    """Factor-2 downsample + ``in_ch→out_ch`` projection + norm.

    Modes: ``conv`` (strided), ``maxpool`` / ``avgpool`` / ``blurpool`` (+ 1×1),
    ``pixelunshuffle`` (space-to-depth + 1×1).
    """

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
        else:  # pixelunshuffle
            channel_mult = 2 ** d  # r=2; channels grow by 2^spatial_dims
            self.op = nn.Sequential(
                PixelUnshuffle3d(r=2, spatial_dims=d),
                _CONV[d](in_ch * channel_mult, out_ch, kernel_size=1, bias=False))

        self.norm = get_norm(norm_type, out_ch, norm_groups, spatial_dims=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.op(x))


# CARAFE (Wang ICCV 2019): predicts a spatially-varying reassembly kernel; better
# locality than pixelshuffle/trilinear. ``k_up=3`` default in 3D (5 is too heavy).
class CARAFE3d(nn.Module):
    """3D Content-Aware ReAssembly of FEatures (default scale=2, k_up=3)."""

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

        # Bottleneck → content encoder (predicts scale^3 · k_up^3 kernel logits/voxel).
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

        # 1) Predict + softmax reassembly kernel at upsampled resolution.
        w = self.compress(x)
        w = self.encode(w)                         # (B, s^3·k^3, D, H, W)
        w = self.shuffle(w)                        # (B, k^3, sD, sH, sW)
        w = F.softmax(w, dim=1)

        # 2) Extract k^3 neighbourhood patches.
        x_pad = F.pad(x, [self.pad] * 6, mode="replicate")
        x_unf = (x_pad
                 .unfold(2, k, 1).unfold(3, k, 1).unfold(4, k, 1)
                 .contiguous()
                 .view(B, C, D, H, W, k ** 3)
                 .permute(0, 1, 5, 2, 3, 4)
                 .reshape(B, C * k ** 3, D, H, W))

        # 3) Nearest-upsample patches (sub-voxels inherit parent neighbourhood),
        # then 4) weighted sum along the k^3 axis.
        x_up = F.interpolate(x_unf, scale_factor=s, mode="nearest")
        x_up = x_up.view(B, C, k ** 3, D * s, H * s, W * s)
        out = (x_up * w.unsqueeze(1)).sum(dim=2)
        return self.proj(out)


# DySample (Liu ICCV 2023): predicts grid_sample offsets; far lighter than CARAFE.
# Near-zero offset init → starts ≈ plain bilinear upsample.
class DySample3d(nn.Module):
    """3D dynamic-sampling upsampler (default scale=2, groups=4, dyscope=True)."""

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

        off_ch = 3 * groups * (scale ** 3)  # 3 coords × groups × s^3
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
        """Base grid in grid_sample (x, y, z) convention, range [-1, 1]."""
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

        # 1) Predict offsets at low-res, shuffle to high-res.
        off = self.offset(x)
        if self.dyscope:
            # DySample-S: learnable scope gate in [0, 0.5].
            off = off * self.scope(x).sigmoid() * 0.5
        off = self.shuffle(off)
        off = off.view(B, g, 3, Du, Hu, Wu)
        off = off.permute(0, 1, 3, 4, 5, 2)

        # 2) Normalise offsets (pixels → grid coord; last-dim order x,y,z = W,H,D).
        norm = torch.tensor(
            [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1), 2.0 / max(D - 1, 1)],
            device=x.device, dtype=x.dtype)
        off = off * norm

        # 3) Base grid + offset, 4) grouped grid_sample (collapse (B, g) into batch).
        base = self._normalised_grid(Du, Hu, Wu, x.device, x.dtype)
        coord = base.unsqueeze(0).unsqueeze(0) + off
        x_g = x.view(B, g, C // g, D, H, W).reshape(B * g, C // g, D, H, W)
        coord = coord.reshape(B * g, Du, Hu, Wu, 3)
        out = F.grid_sample(
            x_g, coord, mode="bilinear",
            padding_mode="border", align_corners=True)
        out = out.view(B, C, Du, Hu, Wu)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Upsampling (multi-mode, factor 2)
# ---------------------------------------------------------------------------
class Upsample(nn.Module):
    """Factor-2 upsample + ``in_ch→out_ch`` projection.

    Modes: ``transpose`` (ConvTranspose), ``trilinear`` / ``nearest`` (interp + 3×3 refine),
    ``pixelshuffle`` (sub-pixel + ICNR), ``carafe`` / ``dysample`` (3D-only).
    """

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
        # carafe / dysample (validated 3D-only in __init__).
        return self.up(x)


def _choose_groups(in_ch: int, preferred: int = 4) -> int:
    """Largest divisor of ``in_ch`` not exceeding ``preferred``."""
    for g in range(min(preferred, in_ch), 0, -1):
        if in_ch % g == 0:
            return g
    return 1
