"""encoder/decoder 通用 2D/3D 基础块。所有模块接受 spatial_dims（默认3）；*3D 名称仅为 API 兼容。包含工厂、ConvNormAct、attention (SE/ECA/CBAM/Coord/Gate)、BlurPool、PixelShuffle、CARAFE/DySample、3D 上/下采样。"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from itertools import product
from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from einops import rearrange

logger = logging.getLogger(__name__)


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
# RoPE cos/sin 有界 LRU 缓存：滑窗推理/多分辨率会产生多种形状 key，
# 设上限防长期运行时条目单调增长。
_ROPE_ND_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
_ROPE_ND_CACHE_MAX = 128

# GroupNorm 组数回退告警去重：同一 (channels, num_groups) 组合只报一次。
_GN_FALLBACK_WARNED: set = set()


def _check_dims(spatial_dims: int) -> int:
    if spatial_dims not in (2, 3):
        raise ValueError(
            f"spatial_dims must be 2 or 3, got {spatial_dims!r}")
    return spatial_dims


def checkpoint_if(enabled: bool, fn, *args):
    """可选梯度检查点：用算力换激活显存。

    ``enabled`` 且当前处于需要梯度的前向（训练且 ``torch.is_grad_enabled()``）时，用
    ``torch.utils.checkpoint`` 包裹 ``fn(*args)``——前向不保存其内部激活，反向时重算一次。
    其余情况（eval / ``torch.no_grad()`` 验证 / 关闭）直接 ``fn(*args)``，**零开销且数值与
    未开启严格一致**。

    - ``use_reentrant=False``：PyTorch 推荐的非重入实现，正确处理「输入不需梯度但子模块参数
      需梯度」（如首个 encoder stage）与多输入（如 ``DecoderLevel(x, skip)``）情形。
    - ``preserve_rng_state=True``：重算时复现 DropPath/dropout 的随机掩码，使梯度无偏。
    """
    if enabled and torch.is_grad_enabled():
        return _torch_checkpoint(
            fn, *args, use_reentrant=False, preserve_rng_state=True)
    return fn(*args)


class DropPath(nn.Module):
    """残差随机深度：训练时按样本丢弃 residual 分支，eval 直通。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # 先 fp32 采样，避免 AMP 下 bernoulli 后端差异。
        prob = torch.full(shape, keep, device=x.device, dtype=torch.float32)
        mask = torch.bernoulli(prob).to(dtype=x.dtype)
        return x * mask / keep


class GlobalResponseNorm(nn.Module):
    """ConvNeXt-V2 GRN：按通道做全局响应归一化，gamma/beta 零初始化。"""

    def __init__(self, channels: int, spatial_dims: int = 3, eps: float = 1e-6):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, x.ndim))
        gx = torch.sqrt(torch.sum(x * x, dim=dims, keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        pat = "c -> 1 c" + " 1" * self.spatial_dims
        gamma = rearrange(self.gamma, pat)
        beta = rearrange(self.beta, pat)
        return x + gamma * (x * nx) + beta


def _as_stride_tuple(stride, spatial_dims: int) -> Tuple[int, ...]:
    """把 int / 序列规整为长度 spatial_dims 的 per-axis stride 元组。

    int → 各轴同 stride（各向同性）；序列 → 直接使用（各向异性）。
    用于支持薄 z 轴的各向异性下采样（如 (1,2,2) 只降 H/W、保 z 分辨率）。
    """
    d = _check_dims(spatial_dims)
    if isinstance(stride, int):
        return (stride,) * d
    s = tuple(int(x) for x in stride)
    if len(s) != d:
        raise ValueError(
            f"stride tuple length {len(s)} must equal spatial_dims={d}; "
            f"got {stride!r}")
    if any(x < 1 for x in s):
        raise ValueError(f"stride values must be >= 1; got {stride!r}")
    return s


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
        requested = num_groups
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        if num_groups != requested and (num_channels, requested) not in _GN_FALLBACK_WARNED:
            _GN_FALLBACK_WARNED.add((num_channels, requested))
            logger.warning(
                "GroupNorm fallback: channels=%d not divisible by num_groups=%d; "
                "using %d group(s) instead (1 group \u2248 LayerNorm).",
                num_channels, requested, num_groups)
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
                 spatial_dims: int = 3, norm_type: str = "group",
                 norm_groups: int = 8):
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
        # 小 batch 3D 下 BatchNorm 统计很噪；归一化类型可配，默认 group。
        self.conv1 = _CONV[d](channels, mid, kernel_size=1, bias=False)
        self.norm1 = get_norm(norm_type, mid, norm_groups, spatial_dims=d)
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


# Large Kernel Attention (Guo 2022, VAN)：大核卷积分解为 DW 小核 + DW 膨胀大核 +
# 1×1，产生空间-通道联合注意力图后逐元素加权；成本远低于自注意力。
class LKA3D(nn.Module):
    """nD Large Kernel Attention：DW k1 → DW k2(dilation) → 1×1 → 逐元素门控。

    默认 k1=5、k2=7、dilation=3，等效感受野≈ 21³（VAN 原始配方）。padding
    对称补齐，任意空间尺寸（含深层小特征图）均合法。注意力图不过 sigmoid
    （与 VAN 一致，保留负相关抑制能力）。"""

    def __init__(self, channels: int, spatial_dims: int = 3,
                 kernel_size: int = 5, dilated_kernel_size: int = 7,
                 dilation: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        conv = _CONV[d]
        self.dw = conv(channels, channels, kernel_size,
                       padding=kernel_size // 2, groups=channels)
        self.dw_dilated = conv(
            channels, channels, dilated_kernel_size,
            padding=(dilated_kernel_size // 2) * dilation,
            dilation=dilation, groups=channels)
        self.pw = conv(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.pw(self.dw_dilated(self.dw(x)))


# Multi-Scale Convolutional Attention (Guo 2022, SegNeXt)：DW 小核局部聚合 +
# 多尺度逐轴条形 DW 核分支 + 1×1 融合 → 逐元素加权。条形核逐轴独立，对
# 各向异性医学体数据（大层厚）及细长结构尤其合适。
class MSCA3D(nn.Module):
    """nD Multi-Scale Conv Attention（SegNeXt 的 nD 推广）。

    每个尺度分支对各空间轴依次做长度 k 的条形 DW 卷积（2D 即 1×k + k×1，
    3D 再多一轴）；分支和 + 局部项后经 1×1 混合通道得注意力图。默认
    scales=(7, 11, 21) 为 SegNeXt 原始配方；padding 对称补齐，任意空间
    尺寸合法。"""

    def __init__(self, channels: int, spatial_dims: int = 3,
                 local_kernel_size: int = 5,
                 scales: Tuple[int, ...] = (7, 11, 21)):
        super().__init__()
        d = _check_dims(spatial_dims)
        conv = _CONV[d]
        self.local = conv(channels, channels, local_kernel_size,
                          padding=local_kernel_size // 2, groups=channels)
        # 每尺度一个分支：逐轴条形 DW 核串接（轴 axis 上长 k，其余轴 1）。
        self.branches = nn.ModuleList()
        for k in scales:
            strips = []
            for axis in range(d):
                size = tuple(k if i == axis else 1 for i in range(d))
                pad  = tuple(k // 2 if i == axis else 0 for i in range(d))
                strips.append(conv(channels, channels, size,
                                   padding=pad, groups=channels))
            self.branches.append(nn.Sequential(*strips))
        self.pw = conv(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.local(x)
        attn = a + sum(branch(a) for branch in self.branches)
        return x * self.pw(attn)


def make_attention(name: str, channels: int,
                   spatial_dims: int = 3, **kwargs) -> nn.Module:
    """Attention 工厂：'none'/'se'/'eca'/'cbam'/'coord'/'lka'/'msca'。

    kwargs：``reduction``；``norm_type``/``norm_groups`` 仅 'coord' 使用
    （其余类型无归一化层）。"""
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
                                spatial_dims=spatial_dims,
                                norm_type=kwargs.get("norm_type", "group"),
                                norm_groups=kwargs.get("norm_groups", 8))
    if name == "lka":
        return LKA3D(channels, spatial_dims=spatial_dims)
    if name == "msca":
        return MSCA3D(channels, spatial_dims=spatial_dims)
    raise ValueError(
        f"Unknown attention type: {name!r}. "
        f"Valid: none|se|eca|cbam|coord|lka|msca")


ATTENTION_TYPES = ("none", "se", "eca", "cbam", "coord", "lka", "msca")


# ---------------------------------------------------------------------------
# Content-based self-attention (QKV / linear QKV)，2.5D/3D 通用。
# 将空间轴拍平为 token 序列后用 Conv1d 做 QKV，故与 spatial_dims 无关。
# softmax：标准多头自注意力 O(N²)，全保真，放最深/瓶颈层；
# linear ：Shen 2021 O(N) 线性注意力（KᵀV 技巧），放次深层。
# 输出投影可 zero-init → 训练初始为恒等残差，几乎不扰动已调好的基线。
# ---------------------------------------------------------------------------
SELFATTN_TYPES = ("softmax", "linear", "window", "grid")


def _resolve_attn_heads(channels: int, num_heads: int, head_dim: int) -> int:
    """head_dim != -1 时按 channels//head_dim 推导头数，否则用 num_heads。"""
    if head_dim is not None and head_dim != -1:
        if channels % head_dim != 0:
            raise ValueError(
                f"SelfAttention: channels ({channels}) not divisible by "
                f"head_dim ({head_dim}).")
        return channels // head_dim
    if num_heads < 1:
        raise ValueError(f"SelfAttention: num_heads must be >= 1, got {num_heads}.")
    if channels % num_heads != 0:
        raise ValueError(
            f"SelfAttention: channels ({channels}) not divisible by "
            f"num_heads ({num_heads}).")
    return num_heads


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1)


def _rope_cache_key(
    spatial_shape: Sequence[int],
    rot_dim: int,
    position_offsets: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    axis: int,
) -> tuple:
    return (
        tuple(int(s) for s in spatial_shape),
        int(rot_dim),
        tuple(int(o) for o in position_offsets),
        device.type,
        device.index,
        dtype,
        int(axis),
    )


def _rope_axis_cos_sin(
    pos: torch.Tensor,
    spatial_shape: Sequence[int],
    rot_dim: int,
    position_offsets: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _rope_cache_key(
        spatial_shape, rot_dim, position_offsets, device, dtype, axis)
    cached = _ROPE_ND_CACHE.get(key)
    if cached is not None:
        _ROPE_ND_CACHE.move_to_end(key)
        return cached
    inv_freq = 1.0 / (
        10000 ** (torch.arange(
            0, rot_dim, 2, device=device, dtype=torch.float32) / rot_dim))
    angles = pos.to(dtype=torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = angles.cos().to(dtype=dtype)
    sin = angles.sin().to(dtype=dtype)
    _ROPE_ND_CACHE[key] = (cos, sin)
    while len(_ROPE_ND_CACHE) > _ROPE_ND_CACHE_MAX:
        _ROPE_ND_CACHE.popitem(last=False)
    return cos, sin


def _apply_rope_nd(
    q: torch.Tensor,
    k: torch.Tensor,
    spatial_shape: Sequence[int],
    position_offsets: Sequence[int] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按空间坐标对 q/k 施加 nD RoPE；多余通道保留不转。"""
    if not spatial_shape:
        return q, k
    num_axes = len(spatial_shape)
    head_dim = q.shape[-1]
    rot_dim = (head_dim // (2 * num_axes)) * 2
    if rot_dim < 2:
        return q, k
    if position_offsets:
        if len(position_offsets) != num_axes:
            raise ValueError(
                f"RoPE position_offsets length ({len(position_offsets)}) "
                f"must match spatial axes ({num_axes}).")
        offsets = [int(v) for v in position_offsets]
    else:
        offsets = [0] * num_axes

    mesh = torch.meshgrid(*[
        torch.arange(s, device=q.device) + off
        for s, off in zip(spatial_shape, offsets)
    ], indexing="ij")
    flat_coords = [m.reshape(-1) for m in mesh]
    tokens = flat_coords[0].numel()
    rot_pairs = rot_dim // 2
    q_out = q.clone()
    k_out = k.clone()
    for axis, pos in enumerate(flat_coords):
        start = axis * rot_dim
        end = start + rot_dim
        cos, sin = _rope_axis_cos_sin(
            pos, spatial_shape, rot_dim, offsets, q.device, q.dtype, axis)
        cos = cos.reshape(1, 1, tokens, rot_pairs, 1)
        sin = sin.reshape(1, 1, tokens, rot_pairs, 1)

        q_blk = q_out[..., start:end].reshape(*q_out.shape[:-1], rot_pairs, 2)
        k_blk = k_out[..., start:end].reshape(*k_out.shape[:-1], rot_pairs, 2)
        q_out[..., start:end] = (
            q_blk * cos + _rope_rotate_half(q_blk) * sin).flatten(-2)
        k_out[..., start:end] = (
            k_blk * cos + _rope_rotate_half(k_blk) * sin).flatten(-2)
    return q_out, k_out


def _normalize_spatial_sizes(
    size: int | Sequence[int],
    spatial_dims: int,
) -> Tuple[int, ...]:
    if isinstance(size, int):
        out = (int(size),) * spatial_dims
    else:
        out = tuple(int(v) for v in size)
    if len(out) != spatial_dims:
        raise ValueError(
            f"Expected {spatial_dims} spatial sizes, got {list(out)}.")
    if any(v < 1 for v in out):
        raise ValueError(f"Spatial sizes must be >= 1; got {list(out)}.")
    return out


def _pad_spatial_to_multiple(
    x: torch.Tensor,
    spatial_shape: Sequence[int],
    block_sizes: Sequence[int],
) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    padded = tuple(
        int(math.ceil(s / b)) * b
        for s, b in zip(spatial_shape, block_sizes))
    if tuple(spatial_shape) == padded:
        return x, padded, tuple(spatial_shape)
    pad = []
    for cur, target in zip(reversed(spatial_shape), reversed(padded)):
        pad.extend([0, int(target - cur)])
    return F.pad(x, pad), padded, tuple(spatial_shape)


def _window_partition_tokens(
    x: torch.Tensor,
    spatial_shape: Sequence[int],
    window_size: int | Sequence[int],
):
    """把 (B,H,C,*spatial) 切成窗口组 (B*num_win,H,tokens,C)。"""
    d = len(spatial_shape)
    ws = _normalize_spatial_sizes(window_size, d)
    x, padded, orig = _pad_spatial_to_multiple(x, spatial_shape, ws)
    if d == 2:
        n1, n2 = padded[0] // ws[0], padded[1] // ws[1]
        x = rearrange(
            x, "b h c (n1 w1) (n2 w2) -> (n1 n2 b) h (w1 w2) c",
            n1=n1, n2=n2, w1=ws[0], w2=ws[1])
        offsets = [
            (i * ws[0], j * ws[1])
            for i, j in product(range(n1), range(n2))
        ]
        mask = torch.ones((x.shape[0] // (n1 * n2),) + tuple(orig),
                          device=x.device, dtype=torch.bool)
        mask = F.pad(mask, (0, padded[1] - orig[1], 0, padded[0] - orig[0]))
        mask = rearrange(
            mask, "b (n1 w1) (n2 w2) -> (n1 n2 b) (w1 w2)",
            n1=n1, n2=n2, w1=ws[0], w2=ws[1])
    else:
        n1, n2, n3 = (
            padded[0] // ws[0], padded[1] // ws[1], padded[2] // ws[2])
        x = rearrange(
            x,
            "b h c (n1 w1) (n2 w2) (n3 w3) -> (n1 n2 n3 b) h (w1 w2 w3) c",
            n1=n1, n2=n2, n3=n3, w1=ws[0], w2=ws[1], w3=ws[2])
        offsets = [
            (i * ws[0], j * ws[1], k * ws[2])
            for i, j, k in product(range(n1), range(n2), range(n3))
        ]
        mask = torch.ones((x.shape[0] // (n1 * n2 * n3),) + tuple(orig),
                          device=x.device, dtype=torch.bool)
        mask = F.pad(
            mask,
            (0, padded[2] - orig[2], 0, padded[1] - orig[1],
             0, padded[0] - orig[0]))
        mask = rearrange(
            mask,
            "b (n1 w1) (n2 w2) (n3 w3) -> (n1 n2 n3 b) (w1 w2 w3)",
            n1=n1, n2=n2, n3=n3, w1=ws[0], w2=ws[1], w3=ws[2])
    meta = {
        "mode": "window",
        "batch": x.shape[0] // len(offsets),
        "group_sizes": tuple(p // w for p, w in zip(padded, ws)),
        "token_sizes": ws,
        "orig_shape": tuple(orig),
        "padded_shape": padded,
        "offsets": offsets,
        "spatial_dims": d,
    }
    return x, mask, meta


def _window_unpartition_tokens(x: torch.Tensor, meta: dict) -> torch.Tensor:
    d = meta["spatial_dims"]
    b = meta["batch"]
    group_sizes = meta["group_sizes"]
    token_sizes = meta["token_sizes"]
    if d == 2:
        x = rearrange(
            x, "(n1 n2 b) h (w1 w2) c -> b h c (n1 w1) (n2 w2)",
            b=b, n1=group_sizes[0], n2=group_sizes[1],
            w1=token_sizes[0], w2=token_sizes[1])
    else:
        x = rearrange(
            x,
            "(n1 n2 n3 b) h (w1 w2 w3) c -> b h c (n1 w1) (n2 w2) (n3 w3)",
            b=b, n1=group_sizes[0], n2=group_sizes[1], n3=group_sizes[2],
            w1=token_sizes[0], w2=token_sizes[1], w3=token_sizes[2])
    orig = meta["orig_shape"]
    if d == 2:
        return x[..., :orig[0], :orig[1]]
    return x[..., :orig[0], :orig[1], :orig[2]]


def _grid_partition_tokens(
    x: torch.Tensor,
    spatial_shape: Sequence[int],
    grid_size: int | Sequence[int],
):
    """把 (B,H,C,*spatial) 切成网格组 (B*num_grid,H,tokens,C)。"""
    d = len(spatial_shape)
    gs = _normalize_spatial_sizes(grid_size, d)
    strides = tuple(max(1, int(math.ceil(s / g))) for s, g in zip(spatial_shape, gs))
    x, padded, orig = _pad_spatial_to_multiple(x, spatial_shape, strides)
    if d == 2:
        g1, g2 = padded[0] // strides[0], padded[1] // strides[1]
        x = rearrange(
            x, "b h c (g1 s1) (g2 s2) -> (s1 s2 b) h (g1 g2) c",
            g1=g1, g2=g2, s1=strides[0], s2=strides[1])
        mask = torch.ones((x.shape[0] // (strides[0] * strides[1]),) + tuple(orig),
                          device=x.device, dtype=torch.bool)
        mask = F.pad(mask, (0, padded[1] - orig[1], 0, padded[0] - orig[0]))
        mask = rearrange(
            mask, "b (g1 s1) (g2 s2) -> (s1 s2 b) (g1 g2)",
            g1=g1, g2=g2, s1=strides[0], s2=strides[1])
    else:
        g1, g2, g3 = (
            padded[0] // strides[0], padded[1] // strides[1], padded[2] // strides[2])
        x = rearrange(
            x,
            "b h c (g1 s1) (g2 s2) (g3 s3) -> (s1 s2 s3 b) h (g1 g2 g3) c",
            g1=g1, g2=g2, g3=g3,
            s1=strides[0], s2=strides[1], s3=strides[2])
        mask = torch.ones((x.shape[0] // math.prod(strides),) + tuple(orig),
                          device=x.device, dtype=torch.bool)
        mask = F.pad(
            mask,
            (0, padded[2] - orig[2], 0, padded[1] - orig[1],
             0, padded[0] - orig[0]))
        mask = rearrange(
            mask,
            "b (g1 s1) (g2 s2) (g3 s3) -> (s1 s2 s3 b) (g1 g2 g3)",
            g1=g1, g2=g2, g3=g3,
            s1=strides[0], s2=strides[1], s3=strides[2])
    meta = {
        "mode": "grid",
        "batch": x.shape[0] // math.prod(strides),
        "group_sizes": strides,
        "token_sizes": tuple(p // s for p, s in zip(padded, strides)),
        "orig_shape": tuple(orig),
        "padded_shape": padded,
        "offsets": None,
        "spatial_dims": d,
    }
    return x, mask, meta


def _grid_unpartition_tokens(x: torch.Tensor, meta: dict) -> torch.Tensor:
    d = meta["spatial_dims"]
    b = meta["batch"]
    group_sizes = meta["group_sizes"]
    token_sizes = meta["token_sizes"]
    if d == 2:
        x = rearrange(
            x, "(s1 s2 b) h (g1 g2) c -> b h c (g1 s1) (g2 s2)",
            b=b, s1=group_sizes[0], s2=group_sizes[1],
            g1=token_sizes[0], g2=token_sizes[1])
    else:
        x = rearrange(
            x,
            "(s1 s2 s3 b) h (g1 g2 g3) c -> b h c (g1 s1) (g2 s2) (g3 s3)",
            b=b, s1=group_sizes[0], s2=group_sizes[1], s3=group_sizes[2],
            g1=token_sizes[0], g2=token_sizes[1], g3=token_sizes[2])
    orig = meta["orig_shape"]
    if d == 2:
        return x[..., :orig[0], :orig[1]]
    return x[..., :orig[0], :orig[1], :orig[2]]


class _SoftmaxQKVAttention(nn.Module):
    """标准多头 softmax 自注意力，输入 qkv=(B, 3*C, N)，输出 (B, C, N)。

    采用 SDPA，计算仍是 O(N²)，但可走更省显存的 fused backend；可选 RoPE。"""

    def __init__(self, num_heads: int, use_rope: bool = False,
                 spatial_dims: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.use_rope = bool(use_rope)
        self.spatial_dims = _check_dims(spatial_dims)

    def forward(self, qkv: torch.Tensor,
                spatial_shape: Sequence[int] = ()) -> torch.Tensor:
        qkv_h = rearrange(qkv, "b (h c3) n -> b h c3 n", h=self.num_heads)
        q, k, v = qkv_h.chunk(3, dim=2)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        if self.use_rope:
            if not spatial_shape:
                raise ValueError("RoPE requires spatial_shape in forward().")
            q, k = _apply_rope_nd(q, k, spatial_shape)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.permute(0, 1, 3, 2)
        return rearrange(a, "b h c n -> b (h c) n")


class _WindowQKVAttention(nn.Module):
    """局部窗口注意力：每个窗口独立做 SDPA，2D/3D 通用。"""

    def __init__(self, num_heads: int, window_size: int | Sequence[int],
                 use_rope: bool = False, spatial_dims: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_rope = bool(use_rope)
        self.spatial_dims = _check_dims(spatial_dims)

    def forward(self, qkv: torch.Tensor,
                spatial_shape: Sequence[int] = ()) -> torch.Tensor:
        qkv_h = rearrange(qkv, "b (h c3) n -> b h c3 n", h=self.num_heads)
        q, k, v = qkv_h.chunk(3, dim=2)
        q = rearrange(q.permute(0, 1, 3, 2).unflatten(-2, spatial_shape),
                      "b h ... c -> b h c ...")
        k = rearrange(k.permute(0, 1, 3, 2).unflatten(-2, spatial_shape),
                      "b h ... c -> b h c ...")
        v = rearrange(v.permute(0, 1, 3, 2).unflatten(-2, spatial_shape),
                      "b h ... c -> b h c ...")
        q, mask, meta = _window_partition_tokens(q, spatial_shape, self.window_size)
        k, _, _ = _window_partition_tokens(k, spatial_shape, self.window_size)
        v, _, _ = _window_partition_tokens(v, spatial_shape, self.window_size)
        attn_mask = torch.zeros(
            (mask.shape[0], 1, 1, mask.shape[1]),
            device=q.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~mask[:, None, None, :],
                                          torch.finfo(q.dtype).min)
        if self.use_rope:
            if not meta["offsets"]:
                raise ValueError("RoPE requires spatial offsets.")
            q, k = _apply_rope_nd(q, k, meta["token_sizes"])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = _window_unpartition_tokens(out, meta)
        return rearrange(out, "b h c ... -> b (h c) ...").flatten(2)


class _GridQKVAttention(nn.Module):
    """稀疏网格注意力：按 residue group 做 SDPA。"""

    def __init__(self, num_heads: int, grid_size: int | Sequence[int],
                 spatial_dims: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.spatial_dims = _check_dims(spatial_dims)

    def forward(self, qkv: torch.Tensor,
                spatial_shape: Sequence[int] = ()) -> torch.Tensor:
        qkv_h = rearrange(qkv, "b (h c3) n -> b h c3 n", h=self.num_heads)
        q, k, v = qkv_h.chunk(3, dim=2)
        q = rearrange(q.permute(0, 1, 3, 2).unflatten(-2, spatial_shape),
                      "b h ... c -> b h c ...")
        k = rearrange(k.permute(0, 1, 3, 2).unflatten(-2, spatial_shape),
                      "b h ... c -> b h c ...")
        v = rearrange(v.permute(0, 1, 3, 2).unflatten(-2, spatial_shape),
                      "b h ... c -> b h c ...")
        q, mask, meta = _grid_partition_tokens(q, spatial_shape, self.grid_size)
        k, _, _ = _grid_partition_tokens(k, spatial_shape, self.grid_size)
        v, _, _ = _grid_partition_tokens(v, spatial_shape, self.grid_size)
        attn_mask = torch.zeros(
            (mask.shape[0], 1, 1, mask.shape[1]),
            device=q.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~mask[:, None, None, :],
                                          torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = _grid_unpartition_tokens(out, meta)
        return rearrange(out, "b h c ... -> b (h c) ...").flatten(2)


class _LinearQKVAttention(nn.Module):
    """O(N) 线性注意力（Shen 2021），输入 qkv=(B, 3*C, N)，输出 (B, C, N)。

    Q 在 head_dim 维 softmax、K 在 token 维 softmax，先算 context=KᵀV 再乘 Q，
    复杂度 O(N·d²) 而非 O(N²·d)。
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor,
                spatial_shape: Sequence[int] = ()) -> torch.Tensor:
        h = self.num_heads
        ch = qkv.shape[1] // (3 * h)
        qkv_h = rearrange(qkv, "b (h c) l -> b h c l", h=h)
        q, k, v = qkv_h.split(ch, dim=2)
        q = q.softmax(dim=2)               # 在 head_dim 维
        k = k.softmax(dim=-1)              # 在 token 维
        q = q * (float(ch) ** -0.5)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        return rearrange(out, "b h c l -> b (h c) l")


class SelfAttentionBlock(nn.Module):
    """内容寻址自注意力残差块（2.5D/3D 通用）。

    结构：x → GroupNorm(PreNorm) → flatten → Conv1d-QKV →
    {softmax|linear|window|grid} attn → Conv1d-proj(可 zero-init) →
    unflatten → + x（残差）。
    可选 RoPE（仅 softmax）与 GEGLU FFN（zero-init 输出投影）。
    """

    def __init__(
        self,
        channels    : int,
        attn_type   : str = "softmax",
        num_heads   : int = 4,
        head_dim    : int = -1,
        norm_groups : int = 32,
        zero_init   : bool = True,
        use_rope    : bool = False,
        use_ffn     : bool = False,
        ffn_ratio   : float = 4.0,
        window_size : int | Sequence[int] = 7,
        grid_size   : int | Sequence[int] = 7,
        spatial_dims: int = 3):
        super().__init__()
        _check_dims(spatial_dims)
        if attn_type not in SELFATTN_TYPES:
            raise ValueError(
                f"Unknown self-attention type: {attn_type!r}; "
                f"expected one of {SELFATTN_TYPES}.")
        if use_rope and attn_type == "linear":
            raise ValueError(
                "SelfAttentionBlock(use_rope=True) is only supported with "
                "'softmax' attention; linear attention factorization does not "
                "support position-coupled q/k rotation.")
        if use_rope and attn_type == "grid":
            raise ValueError(
                "SelfAttentionBlock(use_rope=True) is not supported with "
                "'grid' attention; the strided grouping is incompatible with "
                "a single global rotary coordinate system.")
        self.spatial_dims = spatial_dims
        self.use_rope = bool(use_rope)
        self.num_heads = _resolve_attn_heads(channels, num_heads, head_dim)
        self.window_size = window_size
        self.grid_size = grid_size
        g = norm_groups
        while channels % g != 0 and g > 1:
            g //= 2
        self.norm = nn.GroupNorm(g, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        if attn_type == "softmax":
            self.attn = _SoftmaxQKVAttention(
                self.num_heads, use_rope=use_rope, spatial_dims=spatial_dims)
        elif attn_type == "linear":
            self.attn = _LinearQKVAttention(self.num_heads)
        elif attn_type == "window":
            self.attn = _WindowQKVAttention(
                self.num_heads, window_size=window_size,
                use_rope=use_rope, spatial_dims=spatial_dims)
        else:
            self.attn = _GridQKVAttention(
                self.num_heads, grid_size=grid_size,
                spatial_dims=spatial_dims)
        self.proj = nn.Conv1d(channels, channels, 1)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            hidden = max(int(channels * float(ffn_ratio)), 1)
            self.ffn_norm = nn.GroupNorm(g, channels)
            self.ffn_in = nn.Conv1d(channels, hidden * 2, 1)
            self.ffn_out = nn.Conv1d(hidden, channels, 1)
            nn.init.zeros_(self.ffn_out.weight)
            nn.init.zeros_(self.ffn_out.bias)
        else:
            self.ffn_norm = None
            self.ffn_in = None
            self.ffn_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[2:]
        h = rearrange(self.norm(x), "b c ... -> b c (...)")
        h = self.qkv(h)
        h = self.attn(h, spatial_shape=spatial)
        h = self.proj(h)
        h = h.unflatten(-1, spatial)
        x = x + h
        if self.use_ffn:
            f = rearrange(self.ffn_norm(x), "b c ... -> b c (...)")
            f = self.ffn_in(f)
            a, b = f.chunk(2, dim=1)
            f = a * F.gelu(b)
            f = self.ffn_out(f)
            x = x + f.unflatten(-1, spatial)
        return x


class AttentionGate3D(nn.Module):
    """UNet skip 加性 attention gate (Oktay 2018)：1×1→ReLU→1×1→sigmoid；norm_type 可配。"""

    def __init__(self, x_ch: int, g_ch: int, inter: int = 0,
                 norm_type: str = "batch", norm_groups: int = 8,
                 spatial_dims: int = 3):
        super().__init__()
        d = _check_dims(spatial_dims)
        self.spatial_dims = d
        if inter <= 0:
            inter = max(x_ch // 2, 1)
        self.W_x = nn.Sequential(
            _CONV[d](x_ch, inter, kernel_size=1, bias=False),
            get_norm(norm_type, inter, norm_groups, spatial_dims=d),
        )
        self.W_g = nn.Sequential(
            _CONV[d](g_ch, inter, kernel_size=1, bias=False),
            get_norm(norm_type, inter, norm_groups, spatial_dims=d),
        )
        self.psi = nn.Sequential(
            _CONV[d](inter, 1, kernel_size=1, bias=False),
            get_norm(norm_type, 1, norm_groups, spatial_dims=d),
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
        rd = r ** d
        if Crd % rd:
            raise ValueError(
                f"PixelShuffle3d(r={r}, spatial_dims={d}) needs channels "
                f"divisible by r^d={rd}, got C={Crd}")
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
    """下采样 + in_ch→out_ch 投影 + norm。模式：'conv' 带步长 、'maxpool'/'avgpool'/'blurpool' (+1×1)、'pixelunshuffle'(s2d+1×1)。

    ``stride`` 支持 int（各向同性，默认 2）或 per-axis 元组（各向异性，如 (1,2,2)
    只降 H/W、保 z 分辨率）。各向异性仅 'conv'/'maxpool'/'avgpool' 支持；
    'blurpool'/'pixelunshuffle' 因核结构限制只支持各向同性 stride 2。
    """

    VALID_MODES = ("conv", "maxpool", "avgpool", "blurpool", "pixelunshuffle")

    def __init__(
        self,
        in_ch       : int,
        out_ch      : int,
        norm_type   : str = "instance",
        norm_groups : int = 8,
        mode        : str = "conv",
        spatial_dims: int = 3,
        stride      = 2):
        super().__init__()
        d = _check_dims(spatial_dims)
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown downsample mode: {mode}. "
                f"Valid: {self.VALID_MODES}")
        st = _as_stride_tuple(stride, d)
        self.mode         = mode
        self.spatial_dims = d
        self.stride       = st
        isotropic2        = all(s == 2 for s in st)

        if   mode == "conv":
            # kernel_size == stride：非重叠 tile（per-axis stride=1 → 该轴 1×kernel，保分辨率）。
            self.op = _CONV[d](in_ch, out_ch, kernel_size=st, stride=st, bias=False)
        elif mode == "maxpool":
            self.op = nn.Sequential(
                _MAXPOOL[d](kernel_size=st, stride=st),
                _CONV[d](in_ch, out_ch, kernel_size=1, bias=False))
        elif mode == "avgpool":
            self.op = nn.Sequential(
                _AVGPOOL[d](kernel_size=st, stride=st),
                _CONV[d](in_ch, out_ch, kernel_size=1, bias=False))
        elif mode == "blurpool":
            if not isotropic2:
                raise ValueError(
                    f"downsample_mode='blurpool' only supports isotropic "
                    f"stride 2; got {st}. Use 'conv'/'maxpool'/'avgpool' for "
                    f"anisotropic downsampling.")
            self.op = nn.Sequential(
                BlurPool3d(in_ch, stride=2, filt_size=3, spatial_dims=d),
                _CONV[d](in_ch, out_ch, kernel_size=1, bias=False))
        else:  # pixelunshuffle：r=2，通道×2^d。
            if not isotropic2:
                raise ValueError(
                    f"downsample_mode='pixelunshuffle' only supports "
                    f"isotropic stride 2; got {st}. Use 'conv'/'maxpool'/"
                    f"'avgpool' for anisotropic downsampling.")
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
    """上采样 + in_ch→out_ch 投影。模式：'transpose' 、'trilinear'/'nearest'(插值+3×3精修)、'pixelshuffle'(子像素+ICNR)、'carafe'/'dysample' 仅 3D。

    ``stride`` 支持 int（各向同性，默认 2）或 per-axis 元组（各向异性，须与对应
    encoder Downsample 的 stride 镜像一致）。各向异性仅 'transpose'/'trilinear'/
    'nearest' 支持；'pixelshuffle'/'carafe'/'dysample' 只支持各向同性 stride 2。
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
        stride = 2,
        norm_act: bool = False,
        norm_type: str = "instance",
        norm_groups: int = 8,
        activation: str = "leakyrelu",
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
        st = _as_stride_tuple(stride, d)
        self.mode = mode
        self.spatial_dims = d
        self.stride = st
        isotropic2 = all(s == 2 for s in st)

        # 可选：插值上采样精修 conv 之后再接 norm+act，使插值分支成为真正的
        # 非线性特征变换（否则 interpolate→conv 两层连续线性，直到下游 stage 才有
        # 非线性）。仅对插值模式 'trilinear'/'nearest' 生效；其余模式忽略该选项。
        self.post = nn.Identity()
        if norm_act and mode in ("trilinear", "nearest"):
            self.post = nn.Sequential(
                get_norm(norm_type, out_ch, norm_groups, spatial_dims=d),
                get_activation(activation))

        if mode == "transpose":
            self.up = _CONV_T[d](in_ch, out_ch, kernel_size=st, stride=st)
        elif mode in ("trilinear", "nearest"):
            self.up = _CONV[d](in_ch, out_ch, kernel_size=3, padding=1,
                               bias=False)
        elif mode == "pixelshuffle":
            if not isotropic2:
                raise ValueError(
                    f"upsample_mode='pixelshuffle' only supports isotropic "
                    f"stride 2; got {st}. Use 'transpose'/'trilinear'/"
                    f"'nearest' for anisotropic upsampling.")
            channel_mult = 2 ** d
            self.expand = _CONV[d](in_ch, out_ch * channel_mult,
                                   kernel_size=1, bias=False)
            self.shuffle = PixelShuffle3d(r=2, spatial_dims=d)
            icnr_init_(self.expand.weight, upscale=2, spatial_dims=d)
        elif mode == "carafe":
            if not isotropic2:
                raise ValueError(
                    f"upsample_mode='carafe' only supports isotropic stride 2; "
                    f"got {st}. Use 'transpose'/'trilinear'/'nearest' for "
                    f"anisotropic upsampling.")
            self.up = CARAFE3d(in_ch, out_ch, scale=2, k_up=3, k_enc=3, c_mid=64)
        else:  # dysample
            if not isotropic2:
                raise ValueError(
                    f"upsample_mode='dysample' only supports isotropic stride 2; "
                    f"got {st}. Use 'transpose'/'trilinear'/'nearest' for "
                    f"anisotropic upsampling.")
            groups = _choose_groups(in_ch, preferred=4)
            self.up = DySample3d(in_ch, out_ch, scale=2,
                                 groups=groups, dyscope=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.up(x)
        if self.mode == "trilinear":
            orig_dtype = x.dtype
            if orig_dtype in (torch.bfloat16, torch.float16):
                x = x.float()
            x = F.interpolate(
                x, scale_factor=self.stride,
                mode=INTERP_SMOOTH[self.spatial_dims],
                align_corners=False)
            if x.dtype != orig_dtype:
                x = x.to(orig_dtype)
            return self.post(self.up(x))
        if self.mode == "nearest":
            orig_dtype = x.dtype
            if orig_dtype in (torch.bfloat16, torch.float16):
                x = x.float()
            x = F.interpolate(x, scale_factor=self.stride, mode="nearest")
            if x.dtype != orig_dtype:
                x = x.to(orig_dtype)
            return self.post(self.up(x))
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
