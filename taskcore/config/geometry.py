"""配置与模型共用的空间 stride 推导工具。"""

from __future__ import annotations

from typing import Any, List

# 各向异性自动调度的最小特征边长（nnU-Net 默认 4）：降采样后某轴
# 不小于此值才继续降。
_MIN_FEATURE_SIZE = 4


def stem_stride_of(stem_mode: str) -> int:
    """返回 patch stem 的各向同性 stride。"""
    return {"patch2": 2, "patch4": 4}.get(str(stem_mode), 1)


def decoder_stage_count(decoder_type: Any, n_levels: int) -> int:
    """decoder 实际构造的 stage/node 数（decoder_blocks_per_stage 的唯一口径）。

    * ``unet``   — 逐级解码 ``n_levels - 1``；
    * ``unetpp`` — 三角嵌套节点 ``n(n-1)/2``；
    * 其它（``unet3p`` 等）— 0：decoder 不消费逐级 block 数。
    """
    dtype = str(decoder_type).lower()
    n = int(n_levels)
    if dtype == "unet":
        return max(n - 1, 0)
    if dtype == "unetpp":
        return n * (n - 1) // 2
    return 0


def auto_anisotropic_strides(
    spatial_sizes: List[int],
    num_down: int,
    min_size: int = _MIN_FEATURE_SIZE,
) -> List[tuple]:
    """按当前空间尺寸推导逐级各向异性 stride。

    每一级、每个轴独立判断三项条件：
    (a) 当前尺寸为偶数；(b) 减半后仍不小于 ``min_size``；
    (c) 当前轴尺寸相对本级最大轴尺寸足够大（``size * 2 > ref``）。
    条件按轴分别计算；任一轴停止减半不会阻止其它轴继续下采样。
    """
    sizes = [int(s) for s in spatial_sizes]
    schedule: List[tuple] = []
    for _ in range(num_down):
        ref = max(sizes)
        stride = []
        for ax in range(len(sizes)):
            do_pool = (
                sizes[ax] % 2 == 0
                and sizes[ax] // 2 >= min_size
                and sizes[ax] * 2 > ref
            )
            stride.append(2 if do_pool else 1)
            if do_pool:
                sizes[ax] //= 2
        schedule.append(tuple(stride))
    return schedule


def compute_downsample_strides(cfg: Any, spatial_dims: int, n_levels: int):
    """决定逐级下采样 stride，返回 ``None`` 或 per-axis stride 列表。

    显式 ``downsample_strides`` 优先；否则仅在开启
    ``anisotropic_pooling`` 时按当前 patch 推导；关闭时返回 ``None``，
    由 Encoder 沿用历史各向同性 ×2。显式配置的优先级高于自动调度。
    """
    num_down = int(n_levels) - 1
    if num_down <= 0:
        return None
    model = cfg.model
    unet = model.unet
    explicit = list(unet.downsample_strides or [])
    if explicit:
        return [tuple(int(x) for x in s) for s in explicit]
    if not bool(unet.anisotropic_pooling):
        return None
    patch = [int(x) for x in cfg.data.patch_size]
    spatial_sizes = patch[1:] if int(spatial_dims) == 2 else patch
    s0 = stem_stride_of(model.stem_mode)
    spatial_sizes = [max(1, s // s0) for s in spatial_sizes]
    return auto_anisotropic_strides(spatial_sizes, num_down)


def effective_patch_divisors(cfg: Any, spatial_dims: int, n_levels: int):
    """返回每个有效空间轴应满足的总 stride divisor。"""
    s0 = stem_stride_of(cfg.model.stem_mode)
    strides = compute_downsample_strides(cfg, spatial_dims, n_levels)
    if strides is None:
        strides = [(2,) * int(spatial_dims)] * max(int(n_levels) - 1, 0)
    totals = [s0] * int(spatial_dims)
    for stride in strides:
        for i, value in enumerate(stride):
            totals[i] *= int(value)
    return totals
