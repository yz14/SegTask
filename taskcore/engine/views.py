"""2.5D 折叠原语与折叠契约（全任务共用）。

折叠契约
--------
2.5D 任务（``model.spatial_dims == 2`` 而 patch 为 rank-5 体数据）统一遵循：

1. dataset 总是发 rank-5 ``(B, C, D, H, W)``（C 为视图/模态维，纯 2.5D 时 C=1）；
2. GPU 增强（``GPUAugmentor``）在 rank-5 完整 cube 上执行，使 3D 几何/强度
   增强可无差别作用于 2.5D 样本；
3. 增强后先中心裁掉过采样余量（``aug_oversample_ratio``），必要时逐视图拆分，
   最后**送模型前**才折叠 D 轴进通道：``(B, C, D, H, W) → (B, C*D, H, W)``；
4. 检测任务例外：box 几何必须与图像体素坐标保持对齐，不做折叠。

训练侧（seg pipelines / ssl trainer）与推理侧（predictor 窗口 forward）都使用
本模块的同一批原语，保证口径逐位一致。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange


def fold_depth_to_channels(x: torch.Tensor) -> torch.Tensor:
    """核心折叠原语：rank-5 ``(B, C, D, H, W) → (B, C*D, H, W)``。"""
    if x.ndim != 5:
        raise ValueError(
            f"fold_depth_to_channels expects rank-5 (B, C, D, H, W); "
            f"got shape={tuple(x.shape)}")
    return rearrange(x, 'b c d h w -> b (c d) h w').contiguous()


def squeeze_2_5d(
    image: torch.Tensor,
    label: torch.Tensor,
    wmap: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(B,C_res,D,H,W) → (B,C_res*D,H,W)`` for image；label/wmap 仅取 view 0。"""
    if image.ndim != 5:
        raise ValueError(
            f"2.5D _squeeze expects rank-5 image (B, C_res, D, H, W); "
            f"got shape={tuple(image.shape)}")
    if label.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"image / label batch+C_res mismatch: image="
            f"{tuple(image.shape)}, label={tuple(label.shape)}")
    image = fold_depth_to_channels(image)
    label = label[:, 0]
    if wmap is not None:
        wmap = wmap[:, 0]
    return image, label, wmap


def squeeze_2_5d_keep_views(
    image: torch.Tensor,
    label: torch.Tensor,
    wmap: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``squeeze_2_5d`` 的 aux 变体：image 折叠，label/wmap 保 rank-5 供逐视图索引。"""
    if image.ndim != 5:
        raise ValueError(
            f"2.5D _squeeze_keep_views expects rank-5 image "
            f"(B, C_res, D, H, W); got shape={tuple(image.shape)}")
    if label.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"image / label batch+C_res mismatch: image="
            f"{tuple(image.shape)}, label={tuple(label.shape)}")
    image_2d = fold_depth_to_channels(image)
    return image_2d, label, wmap


__all__ = ["fold_depth_to_channels", "squeeze_2_5d", "squeeze_2_5d_keep_views"]
