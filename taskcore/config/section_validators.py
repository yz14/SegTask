"""跨任务共享的配置段校验（R7）：供 core Config 与 gen Config 共用。

设计：函数接受 duck-typed ``cfg``（需具备 ``.data`` / ``.model`` / ``.augment``
等属性），由 core 方法体迁出；gen 通过调用本模块或直接委托
``CoreConfig._validate_*`` 获得同一套约束，避免平行 fork 漂移。
"""

from __future__ import annotations

import logging
from typing import Any

from .core import _require
from .geometry import effective_patch_divisors

logger = logging.getLogger(__name__)


def validate_encoder_decoder_stage_lengths(cfg: Any) -> None:
    """逐级 block 数 / downsample_strides 长度与 encoder 深度对齐（unet 拓扑）。"""
    n_levels = len(cfg.model.encoder_channels)
    ebps = cfg.model.encoder_blocks_per_stage
    dbps = cfg.model.decoder_blocks_per_stage
    if ebps:
        _require(
            len(ebps) == n_levels,
            f"encoder_blocks_per_stage must have {n_levels} entries "
            f"(= len(encoder_channels)); got {len(ebps)}")
        _require(
            all(b >= 1 for b in ebps),
            "encoder_blocks_per_stage entries must all be >= 1")
    if dbps:
        decoder_type = str(cfg.model.unet.decoder_type).lower()
        expected = (
            n_levels - 1 if decoder_type == "unet"
            else n_levels * (n_levels - 1) // 2
            if decoder_type == "unetpp" else n_levels - 1)
        _require(
            len(dbps) in (1, expected),
            f"decoder_blocks_per_stage for decoder_type={decoder_type!r} "
            f"must have {expected} entries (or 1 entry for explicit broadcast); "
            f"got {len(dbps)}")
        _require(
            all(b >= 1 for b in dbps),
            "decoder_blocks_per_stage entries must all be >= 1")
    # downsample_strides 仅 unet 段有意义；缺省属性则跳过。
    unet = getattr(cfg.model, "unet", None)
    if unet is None:
        return
    sds = getattr(unet, "downsample_strides", None) or []
    if sds:
        sd_dim = int(cfg.model.spatial_dims)
        _require(
            len(sds) == n_levels - 1,
            f"downsample_strides must have {n_levels - 1} entries "
            f"(= len(encoder_channels) - 1); got {len(sds)}")
        for s in sds:
            _require(
                len(s) == sd_dim,
                f"each downsample_strides entry must have "
                f"spatial_dims={sd_dim} values; got {list(s)}")
            _require(
                all(int(v) in (1, 2) for v in s),
                f"downsample_strides values must be 1 or 2; got {list(s)}")


def validate_patch_geometry(cfg: Any) -> None:
    """校验 patch 能完整经过 stem 与 encoder 下采样。"""
    mode = str(getattr(cfg.data, "patch_mode", "cubic"))
    spatial_dims = int(getattr(cfg.model, "spatial_dims", 3))
    n_levels = len(cfg.model.encoder_channels)
    divisors = effective_patch_divisors(cfg, spatial_dims, n_levels)
    patch = [int(x) for x in cfg.data.patch_size]
    axes = patch[1:] if spatial_dims == 2 else patch
    bad = [
        (idx, size, divisor)
        for idx, (size, divisor) in enumerate(zip(axes, divisors))
        if size % divisor
    ]
    if bad:
        detail = ", ".join(
            f"axis {idx} size {size} (needs divisible by {divisor}; "
            f"nearest legal >= value is "
            f"{((size + divisor - 1) // divisor) * divisor})"
            for idx, size, divisor in bad)
        _require(
            False,
            f"patch_size {patch} is incompatible with {mode!r} encoder geometry: "
            f"{detail}. Choose patch dimensions divisible by the reported "
            "stem×downsample stride, or reduce encoder depth.")


def validate_stem_modes(cfg: Any) -> None:
    """stem / stem_fusion / aux_head / spatial_dims 公共枚举。"""
    _require(
        cfg.model.spatial_dims in (2, 3),
        f"Invalid spatial_dims: {cfg.model.spatial_dims} (must be 2 or 3)")
    _require(
        cfg.model.stem_mode in (
            "conv3", "conv7", "dual", "patch2", "patch4",
        ),
        f"Invalid stem_mode: {cfg.model.stem_mode}")
    _require(
        cfg.model.stem_fusion_mode in (
            "shared_stem", "multi_stem_proj", "hierarchical",
        ),
        f"Invalid stem_fusion_mode: {cfg.model.stem_fusion_mode!r}")
    _require(
        cfg.model.aux_head_mode in ("linear", "conv"),
        f"Invalid aux_head_mode: {cfg.model.aux_head_mode!r}")


__all__ = [
    "validate_encoder_decoder_stage_lengths",
    "validate_stem_modes",
]
