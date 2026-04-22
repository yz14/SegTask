"""Common building blocks shared across encoder/decoder implementations.

Provides:
- Layer factories (conv, norm, activation)
- ConvNormAct: single conv + norm + activation
- SqueezeExcite3D: channel attention (SE block, Hu 2018)
- ECA3D: Efficient Channel Attention (Wang et al., CVPR 2020).
- CBAM3D: Convolutional Block Attention Module (Woo et al., ECCV 2018):
  channel + spatial attention in sequence.
- CoordAttention3D: Coordinate Attention (Hou et al., CVPR 2021), extended
  from 2D H/W pooling to 3D D/H/W axis-wise pooling.
- AttentionGate3D: skip-connection attention for UNet (Oktay et al.,
  MIDL 2018 — "Attention U-Net"), the de-facto standard for 3D medical
  segmentation.
- make_attention(name, channels): unified factory returning a channel
  attention block (Identity/SE/ECA/CBAM/CoordAttention).
- BlurPool3D: anti-aliased low-pass filter for shift-invariant strided ops
  (Zhang, "Making Convolutional Networks Shift-Invariant Again", ICML 2019).
- PixelShuffle3D / PixelUnshuffle3D: lossless space<->depth rearrange,
  3D extension of ESPCN sub-pixel convolution (Shi et al., CVPR 2016).
- CARAFE3d: content-aware reassembly upsampler (Wang et al., ICCV 2019),
  3D port with memory-aware default kernel size.
- DySample3d: dynamic-sampling upsampler (Liu et al., ICCV 2023), 3D port.
- Downsample: multi-mode factor-2 3D downsampling (conv / maxpool / avgpool /
  blurpool / pixelunshuffle).
- Upsample: multi-mode factor-2 3D upsampling (transpose / trilinear /
  nearest / pixelshuffle / carafe / dysample).
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer factories
# ---------------------------------------------------------------------------
def get_conv3d() -> Type[nn.Module]:
    return nn.Conv3d


def get_norm(
    norm_type: str,
    num_channels: int,
    num_groups: int = 8,
) -> nn.Module:
    """Create a 3D normalization layer."""
    if norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    elif norm_type == "group":
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm: {norm_type}")


def get_activation(name: str) -> nn.Module:
    """Create an activation layer."""
    if name == "relu":
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
    """3D convolution + normalization + activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = get_norm(norm_type, out_ch, norm_groups)
        self.act = get_activation(activation)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (channel attention)
# ---------------------------------------------------------------------------
class SqueezeExcite3D(nn.Module):
    """3D Squeeze-and-Excitation block (Hu et al., 2018).

    Global average pool → FC reduce → ReLU → FC expand → Sigmoid → scale.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ---------------------------------------------------------------------------
# Efficient Channel Attention (ECA)
# Reference: Wang et al., "ECA-Net: Efficient Channel Attention for Deep
# Convolutional Neural Networks", CVPR 2020.
# Replaces SE's two FC layers with a single 1D convolution over channels.
# Kernel size k is adaptively chosen from C: k = |log2(C)/gamma + b/gamma|_odd.
# ---------------------------------------------------------------------------
class ECA3D(nn.Module):
    """3D Efficient Channel Attention."""

    def __init__(self, channels: int, k_size: int = 0, gamma: int = 2, b: int = 1):
        super().__init__()
        if k_size <= 0:
            # Adaptive kernel size; force odd.
            import math
            k = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
            k_size = k if k % 2 else k + 1
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=k_size // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, 1, 1) -> (B, 1, C) -> conv1d -> (B, 1, C) -> (B, C, 1, 1, 1)
        y = self.avg(x).flatten(1).unsqueeze(1)
        y = self.sig(self.conv(y)).squeeze(1)
        return x * y.view(y.size(0), y.size(1), 1, 1, 1)


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).
# Two sequential sub-modules: channel attention (MLP over GAP+GMP) then
# spatial attention (7×7×7 conv over channel-wise avg+max concat).
# ---------------------------------------------------------------------------
class _CBAMChannelAttn(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.max = nn.AdaptiveMaxPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        avg = self.mlp(self.avg(x).view(B, C))
        mx  = self.mlp(self.max(x).view(B, C))
        w = torch.sigmoid(avg + mx).view(B, C, 1, 1, 1)
        return x * w


class _CBAMSpatialAttn(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("CBAM spatial kernel must be odd.")
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module (channel → spatial)."""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel = _CBAMChannelAttn(channels, reduction)
        self.spatial = _CBAMSpatialAttn(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ---------------------------------------------------------------------------
# Coordinate Attention (Hou et al., CVPR 2021) — 3D extension.
# Encodes spatial position by pooling along each axis separately, then
# applies shared MLP to concatenated descriptors. Captures long-range
# dependencies along one axis at a time — very effective for elongated
# anatomical structures (vessels, airways, spine, …).
# ---------------------------------------------------------------------------
class CoordAttention3D(nn.Module):
    """3D Coordinate Attention.

    Produces three axis-wise attention maps (along D, H, W) fused back
    multiplicatively onto the input feature map.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(channels // reduction, 8)
        # Pool along each axis to a 1D profile (other two dims collapsed).
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        self.conv1 = nn.Conv3d(channels, mid, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(mid)
        self.act = nn.Hardswish(inplace=True)

        self.conv_d = nn.Conv3d(mid, channels, kernel_size=1, bias=False)
        self.conv_h = nn.Conv3d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv3d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # Axis-wise pooled descriptors (each shape has two of the spatial
        # dims collapsed to 1).
        pd = self.pool_d(x)  # (B, C, D, 1, 1)
        ph = self.pool_h(x)  # (B, C, 1, H, 1)
        pw = self.pool_w(x)  # (B, C, 1, 1, W)
        # Flatten spatial axis into the D dim of a common rank-5 tensor so
        # a shared conv1×1 can be applied.
        y = torch.cat([
            pd.view(B, C, D, 1, 1),
            ph.view(B, C, H, 1, 1),
            pw.view(B, C, W, 1, 1),
        ], dim=2)
        y = self.act(self.norm1(self.conv1(y)))
        y_d, y_h, y_w = torch.split(y, [D, H, W], dim=2)

        a_d = torch.sigmoid(self.conv_d(y_d)).view(B, C, D, 1, 1)
        a_h = torch.sigmoid(self.conv_h(y_h)).view(B, C, 1, H, 1)
        a_w = torch.sigmoid(self.conv_w(y_w)).view(B, C, 1, 1, W)
        return x * a_d * a_h * a_w


# ---------------------------------------------------------------------------
# Unified channel-attention factory.
# ---------------------------------------------------------------------------
def make_attention(name: str, channels: int, **kwargs) -> nn.Module:
    """Return a channel/spatial attention block by name.

    Names: "none" | "se" | "eca" | "cbam" | "coord".
    Unknown name → ValueError.
    """
    name = (name or "none").lower()
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SqueezeExcite3D(channels, reduction=kwargs.get("reduction", 16))
    if name == "eca":
        return ECA3D(channels)
    if name == "cbam":
        return CBAM3D(channels, reduction=kwargs.get("reduction", 16))
    if name == "coord":
        return CoordAttention3D(channels, reduction=kwargs.get("reduction", 32))
    raise ValueError(
        f"Unknown attention type: {name!r}. "
        f"Valid: none|se|eca|cbam|coord")


ATTENTION_TYPES = ("none", "se", "eca", "cbam", "coord")


# ---------------------------------------------------------------------------
# Attention Gate for skip connections (Oktay et al., MIDL 2018).
# Gates the encoder skip feature by a gating signal from the coarser
# decoder path. Has become the standard attention mechanism for 3D medical
# UNets (Attention U-Net).
#
#   x  (skip)  -- W_x 1x1 -- + -- ReLU -- W_psi 1x1 -- sigmoid -- * -- out
#   g  (gate)  -- W_g 1x1 --/
# ---------------------------------------------------------------------------
class AttentionGate3D(nn.Module):
    """Additive attention gate for UNet skip connections.

    Args:
        x_ch:  channels of the skip (encoder) feature.
        g_ch:  channels of the gating (decoder) feature.
        inter: bottleneck channel count (default = x_ch // 2, min 1).
    """

    def __init__(self, x_ch: int, g_ch: int, inter: int = 0):
        super().__init__()
        if inter <= 0:
            inter = max(x_ch // 2, 1)
        self.W_x = nn.Sequential(
            nn.Conv3d(x_ch, inter, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter),
        )
        self.W_g = nn.Sequential(
            nn.Conv3d(g_ch, inter, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Gate skip feature ``x`` using decoder signal ``g``.

        If the spatial sizes differ, ``g`` is trilinearly resized to match
        ``x``. The returned tensor has the same shape as ``x``.
        """
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear",
                              align_corners=False)
        return x * self.psi(self.relu(self.W_x(x) + self.W_g(g)))


# ---------------------------------------------------------------------------
# Anti-aliased downsampling: BlurPool3D
# Reference: Zhang, "Making Convolutional Networks Shift-Invariant Again",
# ICML 2019. Applying a low-pass (binomial) filter before subsampling
# dramatically improves shift-invariance compared to naive strided ops.
# ---------------------------------------------------------------------------
class BlurPool3d(nn.Module):
    """3D anti-aliased blur + stride downsample.

    A fixed (non-learned), depthwise separable binomial low-pass filter is
    applied prior to subsampling. filt_size=3 uses the classic [1,2,1]
    binomial kernel (outer-product in 3D), filt_size=5 uses [1,4,6,4,1].
    """

    _BINOMIAL: dict = {
        2: (1., 1.),
        3: (1., 2., 1.),
        5: (1., 4., 6., 4., 1.),
    }

    def __init__(self, channels: int, stride: int = 2, filt_size: int = 3):
        super().__init__()
        if filt_size not in self._BINOMIAL:
            raise ValueError(f"Unsupported BlurPool filt_size: {filt_size}")
        self.channels = channels
        self.stride = stride
        self.pad = filt_size // 2

        a = torch.tensor(self._BINOMIAL[filt_size], dtype=torch.float32)
        # 3D separable kernel via outer products.
        kernel = a[:, None, None] * a[None, :, None] * a[None, None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None].expand(channels, 1, *kernel.shape).contiguous()
        # Non-learned filter, registered as buffer so it moves with .to(device).
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reflection padding preserves boundary statistics (zero-padding
        # would bias constant regions towards 0 near the border).
        if self.pad:
            x = F.pad(x, [self.pad] * 6, mode="replicate")
        return F.conv3d(
            x, self.kernel,
            stride=self.stride, padding=0, groups=self.channels,
        )


# ---------------------------------------------------------------------------
# PixelShuffle / PixelUnshuffle in 3D (sub-pixel conv, ESPCN-style).
# PyTorch ships nn.PixelShuffle for 2D only; we implement the 3D analogue
# via a single reshape+permute. Both ops are lossless and parameter-free.
# ---------------------------------------------------------------------------
class PixelUnshuffle3d(nn.Module):
    """Space-to-depth: (B, C, rD, rH, rW) -> (B, C*r^3, D, H, W)."""

    def __init__(self, r: int = 2):
        super().__init__()
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        r = self.r
        if D % r or H % r or W % r:
            raise ValueError(
                f"PixelUnshuffle3d(r={r}) needs spatial dims divisible by r, "
                f"got (D,H,W)=({D},{H},{W})")
        x = x.view(B, C, D // r, r, H // r, r, W // r, r)
        # Bring sub-pixel axes (r,r,r) next to channel axis.
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return x.view(B, C * (r ** 3), D // r, H // r, W // r)


class PixelShuffle3d(nn.Module):
    """Depth-to-space: (B, C*r^3, D, H, W) -> (B, C, rD, rH, rW)."""

    def __init__(self, r: int = 2):
        super().__init__()
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cr3, D, H, W = x.shape
        r = self.r
        if Cr3 % (r ** 3):
            raise ValueError(
                f"PixelShuffle3d(r={r}) needs channels divisible by r^3={r**3}, "
                f"got C={Cr3}")
        C = Cr3 // (r ** 3)
        x = x.view(B, C, r, r, r, D, H, W)
        # Interleave sub-pixel axes back into spatial dims.
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(B, C, D * r, H * r, W * r)


def icnr_init_(weight: torch.Tensor, upscale: int, init: Type[nn.Module] = None) -> None:
    """ICNR initialisation for sub-pixel convolution weights.

    Reference: Aitken et al., "Checkerboard artifact free sub-pixel
    convolution" (2017). The conv-then-shuffle composition is initialised
    so that its effect is approximately nearest-neighbour upsampling,
    eliminating checkerboard artefacts at the start of training.

    Args:
        weight: conv weight tensor shape (out_ch*r^3, in_ch, k, k, k).
        upscale: r (upscale factor).
    """
    r3 = upscale ** 3
    out_total = weight.shape[0]
    if out_total % r3 != 0:
        raise ValueError("ICNR: out_ch must be divisible by r^3")
    out_ch = out_total // r3
    sub = torch.empty(out_ch, *weight.shape[1:], device=weight.device,
                      dtype=weight.dtype)
    nn.init.kaiming_normal_(sub)
    # Replicate each output filter r^3 times so all sub-pixel siblings start
    # identical → PixelShuffle output equals nearest-neighbour upsampling.
    weight.data.copy_(sub.repeat_interleave(r3, dim=0))


# ---------------------------------------------------------------------------
# Downsampling (multi-mode, factor 2)
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    """Downsample spatial dims by factor 2 and project channels in_ch→out_ch.

    Modes:
      - "conv"          : strided 2×2×2 convolution (baseline).
      - "maxpool"       : MaxPool3d(2) + 1×1×1 conv for channel projection.
      - "avgpool"       : AvgPool3d(2) + 1×1×1 conv. Smoother than max.
      - "blurpool"      : binomial blur + stride 2 + 1×1×1 conv
                          (anti-aliased; Zhang 2019).
      - "pixelunshuffle": lossless space-to-depth (r=2) + 1×1×1 conv.
                          Information-preserving, similar spirit to Swin
                          patch-merging.
    All modes are followed by a normalization layer.
    """

    VALID_MODES = ("conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle")

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "instance",
        norm_groups: int = 8,
        mode: str = "conv",
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown downsample mode: {mode}. "
                f"Valid: {self.VALID_MODES}")
        self.mode = mode

        if mode == "conv":
            self.op = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        elif mode == "maxpool":
            self.op = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            )
        elif mode == "avgpool":
            self.op = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            )
        elif mode == "blurpool":
            self.op = nn.Sequential(
                BlurPool3d(in_ch, stride=2, filt_size=3),
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            )
        else:  # pixelunshuffle
            self.op = nn.Sequential(
                PixelUnshuffle3d(r=2),
                nn.Conv3d(in_ch * 8, out_ch, kernel_size=1, bias=False),
            )

        self.norm = get_norm(norm_type, out_ch, norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.op(x))


# ---------------------------------------------------------------------------
# Content-aware upsampling: CARAFE (3D)
# Reference: Wang et al., "CARAFE: Content-Aware ReAssembly of FEatures",
# ICCV 2019.  Predicts a spatially-varying reassembly kernel at every output
# location, then upsamples each channel as a weighted sum over a k_up×k_up×k_up
# neighbourhood in the low-res feature map.  Strong locality awareness,
# significantly better than pixelshuffle/trilinear on dense prediction tasks.
#
# 3D notes
# --------
# For a k_up × k_up × k_up kernel the reassembly requires a (C, k_up^3)
# intermediate tensor per spatial location, so memory scales with C·k_up^3.
# The original 2D default (k_up=5) is too heavy in 3D; we default to k_up=3
# (≈1/5 the memory) which keeps quality comparable on medical volumes.
# ---------------------------------------------------------------------------
class CARAFE3d(nn.Module):
    """3D Content-Aware ReAssembly of FEatures (scale=2).

    Args:
        in_ch:      input channel count.
        out_ch:     output channel count (1×1×1 projection after reassembly).
        scale:      upsample factor (only 2 tested here; any s ≥ 1 supported).
        k_up:       reassembly kernel size (cube). Default 3 (memory friendly).
        k_enc:      kernel size of the content-encoder conv. Default 3.
        c_mid:      bottleneck channels for kernel prediction. Default 64.
    """

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

        # Channel compressor (cheap bottleneck).
        self.compress = nn.Conv3d(in_ch, c_mid, kernel_size=1)
        # Content encoder: predicts scale^3 · k_up^3 kernel logits per voxel.
        self.encode = nn.Conv3d(
            c_mid, (scale ** 3) * (k_up ** 3),
            kernel_size=k_enc, padding=k_enc // 2)
        self.shuffle = PixelShuffle3d(r=scale)
        # Final channel projection.
        self.proj = (nn.Conv3d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        s = self.scale
        k = self.k_up

        # 1) Predict normalised reassembly kernel at upsampled resolution.
        w = self.compress(x)
        w = self.encode(w)                         # (B, s^3·k^3, D, H, W)
        w = self.shuffle(w)                        # (B, k^3, sD, sH, sW)
        w = F.softmax(w, dim=1)

        # 2) Extract k^3 neighbourhood patches from the low-res feature map.
        x_pad = F.pad(x, [self.pad] * 6, mode="replicate")
        x_unf = (x_pad
                 .unfold(2, k, 1).unfold(3, k, 1).unfold(4, k, 1)
                 .contiguous()
                 .view(B, C, D, H, W, k ** 3)
                 .permute(0, 1, 5, 2, 3, 4)        # (B, C, k^3, D, H, W)
                 .reshape(B, C * k ** 3, D, H, W))

        # 3) Upsample patches with nearest — each sub-voxel inherits its
        #    parent voxel's k^3 neighbourhood, so the predicted kernel picks
        #    a different linear combination at every sub-voxel.
        x_up = F.interpolate(x_unf, scale_factor=s, mode="nearest")
        x_up = x_up.view(B, C, k ** 3, D * s, H * s, W * s)

        # 4) Weighted sum along the k^3 axis.
        out = (x_up * w.unsqueeze(1)).sum(dim=2)   # (B, C, sD, sH, sW)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Content-aware upsampling: DySample (3D)
# Reference: Liu et al., "Learning to Upsample by Learning to Sample",
# ICCV 2023.  Predicts sampling offsets and uses grid_sample — extremely
# light-weight (1–2 orders of magnitude fewer params than CARAFE) and in
# 2D matches or outperforms CARAFE/DyConv on dense prediction.  Initial
# offsets are near-zero so training starts from plain upsampling.
# ---------------------------------------------------------------------------
class DySample3d(nn.Module):
    """3D dynamic-sampling upsampler (scale=2).

    Args:
        in_ch:   input channel count.
        out_ch:  output channel count.
        scale:   upsample factor (default 2).
        groups:  number of sampling groups. Higher = finer-grained sampling
                 but more params. in_ch must be divisible by groups.
        dyscope: enable DySample-S (dynamic scope gate) from the paper.
    """

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

        off_ch = 3 * groups * (scale ** 3)  # 3 coords × groups × s^3 sub-voxels
        self.offset = nn.Conv3d(in_ch, off_ch, kernel_size=1)
        # Near-zero offset init so DySample starts ≈ bilinear upsampling.
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
        """Build base grid in grid_sample's (x, y, z) convention, range [-1,1]."""
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

        # 1) Predict offsets at low-res, then shuffle to high-res.
        off = self.offset(x)                                   # (B, 3·g·s^3, D, H, W)
        if self.dyscope:
            # DySample-S: learnable scope gate in [0, 0.5].
            off = off * self.scope(x).sigmoid() * 0.5
        off = self.shuffle(off)                                # (B, 3·g, Du, Hu, Wu)
        off = off.view(B, g, 3, Du, Hu, Wu)                    # (B, g, 3, Du, Hu, Wu)
        off = off.permute(0, 1, 3, 4, 5, 2)                    # (B, g, Du, Hu, Wu, 3)

        # 2) Normalise offsets (pixels → grid_sample coord).  grid_sample
        #    uses last-dim order (x, y, z) corresponding to (W, H, D).
        norm = torch.tensor(
            [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1), 2.0 / max(D - 1, 1)],
            device=x.device, dtype=x.dtype)
        off = off * norm

        # 3) Base grid at upsampled resolution + predicted offset.
        base = self._normalised_grid(Du, Hu, Wu, x.device, x.dtype)  # (Du,Hu,Wu,3)
        coord = base.unsqueeze(0).unsqueeze(0) + off                # (B,g,Du,Hu,Wu,3)

        # 4) Grouped grid_sample.  Collapse (B, g) into batch dim.
        x_g = x.view(B, g, C // g, D, H, W).reshape(B * g, C // g, D, H, W)
        coord = coord.reshape(B * g, Du, Hu, Wu, 3)
        out = F.grid_sample(
            x_g, coord, mode="bilinear",
            padding_mode="border", align_corners=True,
        )                                                      # (B·g, C/g, Du, Hu, Wu)
        out = out.view(B, C, Du, Hu, Wu)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Upsampling (multi-mode, factor 2)
# ---------------------------------------------------------------------------
class Upsample(nn.Module):
    """Upsample spatial dims by factor 2 and project channels in_ch→out_ch.

    Modes:
      - "transpose"   : ConvTranspose3d(2,2). Classic but can cause
                        checkerboard artefacts (Odena et al. 2016).
      - "trilinear"   : trilinear interp + 3×3×3 refinement conv. Smooth.
      - "nearest"     : nearest interp + 3×3×3 conv. Odena-style
                        anti-checkerboard; widely used in modern UNets.
      - "pixelshuffle": sub-pixel conv (3D ESPCN). 1×1×1 conv expands
                        channels to out_ch * 2^3, then PixelShuffle3d(r=2)
                        rearranges into higher resolution. Initialised
                        with ICNR to prevent checkerboard at init.
      - "carafe"      : 3D CARAFE — content-aware reassembly kernel
                        (Wang ICCV 2019). k_up=3 by default for 3D.
      - "dysample"    : 3D DySample (Liu ICCV 2023) — lightweight dynamic
                        sampling via learned offsets + grid_sample.
    """

    VALID_MODES = ("transpose", "trilinear", "nearest", "pixelshuffle",
                   "carafe", "dysample")

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mode: str = "transpose",
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown upsample mode: {mode}. Valid: {self.VALID_MODES}")
        self.mode = mode

        if mode == "transpose":
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        elif mode in ("trilinear", "nearest"):
            # Parameter-free interp → 3×3×3 refinement.
            self.up = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        elif mode == "pixelshuffle":
            self.expand = nn.Conv3d(in_ch, out_ch * 8, kernel_size=1, bias=False)
            self.shuffle = PixelShuffle3d(r=2)
            icnr_init_(self.expand.weight, upscale=2)
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
            x = F.interpolate(x, scale_factor=2, mode="trilinear",
                              align_corners=False)
            return self.up(x)
        if self.mode == "nearest":
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            return self.up(x)
        if self.mode == "pixelshuffle":
            return self.shuffle(self.expand(x))
        # carafe / dysample
        return self.up(x)


def _choose_groups(in_ch: int, preferred: int = 4) -> int:
    """Largest divisor of in_ch not exceeding `preferred`."""
    for g in range(min(preferred, in_ch), 0, -1):
        if in_ch % g == 0:
            return g
    return 1
