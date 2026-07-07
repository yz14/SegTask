"""多视图消费侧共用几何原语（生成任务）。

dataset 总是发单条 max-FOV 过采样 cube（rank-5 ``(B, 1, eD, eH, eW)``），由
trainer 侧在 GPU 增强后完成：过采样余量中心裁剪 → 逐视图 FOV 中心裁剪 →
resize 回 patch_size → 打包成模型输入布局。本模块提供这些纯几何原语，
供 ``trainer.pipelines`` 各管线组合调用。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def center_crop(x: torch.Tensor, target: Sequence[int]) -> torch.Tensor:
    """对最后 ``len(target)`` 个维度做中心裁剪；已等尺寸的轴为 no-op。

    要求各轴 ``x.size >= target``（dataset 侧保证过采样余量只增不减）。
    """
    nd = len(target)
    sizes = x.shape[-nd:]
    slices: List[slice] = [slice(None)] * (x.ndim - nd)
    for cur, tgt in zip(sizes, target):
        tgt = int(tgt)
        if cur < tgt:
            raise ValueError(
                f"center_crop target {tuple(target)} exceeds input spatial "
                f"shape {tuple(sizes)}.")
        lo = (cur - tgt) // 2
        slices.append(slice(lo, lo + tgt))
    return x[tuple(slices)]


def resize_volume(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """rank-5 ``(B, C, D, H, W)`` trilinear resize；等尺寸时 no-op。"""
    if tuple(x.shape[-3:]) == tuple(size):
        return x
    return F.interpolate(
        x, size=tuple(int(s) for s in size), mode="trilinear",
        align_corners=False)


def view_sizes_z(
    patch_size: Sequence[int], scales: Sequence[float]) -> List[Tuple[int, int, int]]:
    """z 轴多 FOV：仅 z 轴按 scale 放大，H/W 恒为 patch 面内尺寸。"""
    pD, pH, pW = (int(p) for p in patch_size)
    return [(int(round(pD * float(s))), pH, pW) for s in scales]


def view_sizes_cubic(
    patch_size: Sequence[int], scales: Sequence[float]) -> List[Tuple[int, int, int]]:
    """cubic 多 FOV：三轴同步按 scale 放大。"""
    pD, pH, pW = (int(p) for p in patch_size)
    return [(int(round(pD * float(s))),
             int(round(pH * float(s))),
             int(round(pW * float(s)))) for s in scales]


def split_views_stacked(
    cube: torch.Tensor,
    view_sizes: Sequence[Tuple[int, int, int]],
    out_size: Sequence[int]) -> torch.Tensor:
    """从 max-FOV cube ``(B, 1, Dm, Hm, Wm)`` 拆多视图并 resize 到统一 patch 尺寸。

    view k：中心裁剪 ``view_sizes[k]`` → trilinear resize 到 ``out_size``；
    输出 ``(B, n_views, pD, pH, pW)``（视图堆在通道轴）。view 0 应为 scale=1.0
    的原生 FOV（无 resize）。
    """
    views = [
        resize_volume(center_crop(cube, size), out_size)
        for size in view_sizes]
    return torch.cat(views, dim=1)


def split_views_native_d(
    cube: torch.Tensor, depths: Sequence[int]) -> torch.Tensor:
    """2.5D 原生深度多视图：从 ``(B, 1, Dm, H, W)`` 逐视图中心裁 z-slab（深度
    D_k，不 resize），压掉领头通道后按通道拼接为 ``(B, ΣD_k, H, W)``。
    """
    slabs = [center_crop(cube, (int(d),) + tuple(cube.shape[-2:]))[:, 0]
             for d in depths]
    return torch.cat(slabs, dim=1)


__all__ = [
    "center_crop", "resize_volume", "view_sizes_z", "view_sizes_cubic",
    "split_views_stacked", "split_views_native_d",
]
