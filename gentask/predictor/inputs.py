"""Predictor 包窗口建造子模块（R6）。

R6 抽自 ``predictor.py``：6 个 window/batch builders 全部改为**模块级纯函数**，
显式接收 patch 几何与 mode 派生量，不再依赖 ``self`` 隐式状态。``Predictor`` 类侧
保留同名 ``_build_*`` thin shim 方法（许多单元测试通过类直调消费这些方法），方法
内部委托至此处。

3 个 z 轴 GPU builders + 1 个 CPU multi-res builder + 2 个 cubic batch builders：

* ``build_z_window_single_res_gpu``       —— OFF/single-res GPU；返 ``(1, pD, pH, pW)``
* ``build_z_window_native_multi_res_gpu`` —— 3D z_axis ON；返 ``(C_res, pD, pH, pW)``
* ``build_z_window_native_d_gpu``         —— 2.5D ON；返 ``(sum(D_k), pH, pW)``
* ``build_z_window_cpu_multi_res``        —— 多分辨率 CPU 退化路径；返 ``(C_res, pD, pH, pW)`` ndarray
* ``build_cubic_batch_native_multi_res``  —— cubic ON 批；返 ``(B, C_res, pD, pH, pW)``
* ``build_cubic_batch_cpu_multi_res``     —— cubic OFF 批；返 ``(B, C_res, pD, pH, pW)``
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import _extract_cubic_patch, extract_z_patch_padded, resize_3d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _edge_pad_z(slab: torch.Tensor, pad_before: int,
                pad_after: int) -> torch.Tensor:
    """对 ``slab[D, H, W]`` 沿 D 轴 edge-replicate（``expand`` 零拷贝 + 单次 cat）。"""
    if pad_before == 0 and pad_after == 0:
        return slab
    chunks: List[torch.Tensor] = []
    if pad_before > 0:
        chunks.append(slab[0:1].expand(pad_before, -1, -1))
    chunks.append(slab)
    if pad_after > 0:
        chunks.append(slab[-1:].expand(pad_after, -1, -1))
    return torch.cat(chunks, dim=0)


def _edge_pad_axis(t: torch.Tensor, axis: int, pad_before: int,
                   pad_after: int) -> torch.Tensor:
    """沿任意 ``axis`` 复制边界填充（cubic ON 三轴 padding 共用）。"""
    if pad_before == 0 and pad_after == 0:
        return t
    chunks: List[torch.Tensor] = []
    base_shape = list(t.shape)
    if pad_before > 0:
        first = t.narrow(axis, 0, 1)
        shape = list(base_shape)
        shape[axis] = pad_before
        chunks.append(first.expand(shape))
    chunks.append(t)
    if pad_after > 0:
        last = t.narrow(axis, t.shape[axis] - 1, 1)
        shape = list(base_shape)
        shape[axis] = pad_after
        chunks.append(last.expand(shape))
    return torch.cat(chunks, dim=axis)


def _extract_z_slab_resized(vol_t: torch.Tensor, z_center: int, eD: int,
                            pH: int, pW: int) -> torch.Tensor:
    """共享子例程：以 ``z_center`` 为中心抽 ``eD`` 深 slab、edge-replicate、面内 resize 到 ``(pH, pW)``。

    供 native_multi_res 与 native_d 共享。返 ``(eD, pH, pW)`` fp32。
    """
    D_vol = vol_t.shape[0]
    zlo = z_center - eD // 2
    zhi = zlo + eD
    zlo_in = max(zlo, 0)
    zhi_in = min(zhi, D_vol)
    slab = vol_t[zlo_in:zhi_in]
    slab = _edge_pad_z(slab, max(0, -zlo), max(0, zhi - D_vol))
    if slab.shape[0] != eD:
        raise RuntimeError(
            f"_extract_z_slab_resized: expected depth {eD}, got "
            f"{slab.shape[0]} (z_center={z_center}, D_vol={D_vol}).")

    H_orig, W_orig = slab.shape[1], slab.shape[2]
    slab = slab.unsqueeze(0).unsqueeze(0).float()  # (1,1,eD,H,W)
    if H_orig != pH or W_orig != pW:
        slab = F.interpolate(
            slab, size=(eD, pH, pW),
            mode="trilinear", align_corners=False)
    return slab[0, 0]                              # (eD, pH, pW)


# ---------------------------------------------------------------------------
# Z-axis window builders (GPU)
# ---------------------------------------------------------------------------
def build_z_window_single_res_gpu(
    vol_t: torch.Tensor, z0: int, z1: int,
    *, pD: int, pH: int, pW: int, z_boundary_mode: str,
) -> torch.Tensor:
    """单分辨率 GPU 窗口建造。

    * ``stretch``: 取 ``vol[z0:z1]`` 后三线性 resize 到 ``(pD, pH, pW)``
    * ``edge_pad``: ``ad < pD`` 时对称复制填充到 ``pD`` 后再 resize（与训练 multi-res 一致）

    返 ``(1, pD, pH, pW)``。
    """
    patch = vol_t[z0:z1]
    ad, H, W = patch.shape

    if z_boundary_mode == "edge_pad" and ad < pD:
        pad_before = (pD - ad) // 2
        pad_after = pD - ad - pad_before
        patch = _edge_pad_z(patch, pad_before, pad_after)
        ad = pD

    patch = patch.unsqueeze(0).unsqueeze(0).float()  # (1,1,ad,H,W)
    if (ad != pD) or (H != pH) or (W != pW):
        patch = F.interpolate(
            patch, size=(pD, pH, pW),
            mode="trilinear", align_corners=False)
    return patch.squeeze(0)                          # (1, pD, pH, pW)


def build_z_window_native_multi_res_gpu(
    vol_t: torch.Tensor, z0: int, z1: int,
    *, pD: int, pH: int, pW: int,
    target_shape: Tuple[int, int, int],
    native_sizes: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """3D ``z_axis`` ON 模式窗口建造：抽单 max-FOV cube → 面内 resize 到 ``(pH, pW)`` →
    逐视图中心裁 ``D_k`` 后 D 轴 trilinear 回 ``pD`` → 拼 ``C_res``。

    ``target_shape = (eD_max, pH, pW)``；``native_sizes[k] = (D_k, H_k=pH, W_k=pW)``。
    返 ``(C_res, pD, pH, pW)``。
    """
    eD_max = target_shape[0]
    z_center = (z0 + z1) // 2
    slab = _extract_z_slab_resized(vol_t, z_center, eD_max, pH, pW)
    slab = slab.unsqueeze(0).unsqueeze(0)            # (1, 1, eD_max, pH, pW)

    view_chunks: List[torch.Tensor] = []
    for D_k, _, _ in native_sizes:
        d0 = (eD_max - D_k) // 2
        crop = slab[:, :, d0:d0 + D_k, :, :]         # (1, 1, D_k, pH, pW)
        if D_k != pD:
            crop = F.interpolate(
                crop, size=(pD, pH, pW),
                mode="trilinear", align_corners=False)
        view_chunks.append(crop[0])                  # (1, pD, pH, pW)
    return torch.cat(view_chunks, dim=0).contiguous()  # (C_res, pD, pH, pW)


def build_z_window_native_d_gpu(
    vol_t: torch.Tensor, z0: int, z1: int,
    *, pH: int, pW: int, eD_max: int, view_depths: List[int],
) -> torch.Tensor:
    """2.5D ``keep_native_view_depth=True`` 模式窗口建造：抽 ``eD_max`` max-FOV slab、
    面内 resize 到 ``(pH, pW)``、逐视图中心抽 ``D_k`` 切片后拼通道。

    返 ``(sum(D_k), pH, pW)``。
    """
    z_center = (z0 + z1) // 2
    slab = _extract_z_slab_resized(vol_t, z_center, eD_max, pH, pW)
    # (eD_max, pH, pW)

    view_chunks: List[torch.Tensor] = []
    for D_k in view_depths:
        d0 = (eD_max - D_k) // 2
        view_chunks.append(slab[d0:d0 + D_k])         # (D_k, pH, pW)
    return torch.cat(view_chunks, dim=0).contiguous()


# ---------------------------------------------------------------------------
# Z-axis CPU multi-res fallback
# ---------------------------------------------------------------------------
def build_z_window_cpu_multi_res(
    vol: np.ndarray, z0: int, z1: int,
    *, pD: int, pH: int, pW: int,
    multi_res_scales: List[float], z_boundary_mode: str,
) -> np.ndarray:
    """多分辨率 z 窗口堆（CPU 退化路径）。

    * ``scale > 1``: 总走 edge-padded 抽 ``round(pD * scale)`` 切片
    * ``scale == 1``: 按 ``z_boundary_mode`` 选择 ``stretch`` / ``edge_pad``

    返 ``(C_res, pD, pH, pW)`` fp32。
    """
    z_center = (z0 + z1) // 2
    channels: List[np.ndarray] = []
    for scale in multi_res_scales:
        if scale == 1.0:
            if z_boundary_mode == "edge_pad":
                patch = extract_z_patch_padded(vol, z_center, pD)
            else:
                # 旧尾窗行为：取实际切片，后面 resize。
                patch = vol[z0:z1]
        else:
            D_s = int(round(pD * scale))
            patch = extract_z_patch_padded(vol, z_center, D_s)
        patch = resize_3d(patch, pD, pH, pW, is_label=False)
        channels.append(patch)
    return np.stack(channels, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cubic batch builders
# ---------------------------------------------------------------------------
def build_cubic_batch_native_multi_res(
    centers: List[Tuple[int, int, int]],
    vol_t: torch.Tensor,
    *, pD: int, pH: int, pW: int,
    target_shape: Tuple[int, int, int],
    native_sizes: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """3D ``cubic`` ON 模式批建造：逐中心抽单 max-FOV cube → 逐视图中心裁
    ``(D_k, H_k, W_k)`` 后 trilinear 回 ``(pD, pH, pW)`` → 拼 ``C_res``。

    返 ``(B, C_res, pD, pH, pW)``。
    """
    tD, tH, tW = target_shape
    D_vol, H_vol, W_vol = vol_t.shape

    cubes: List[torch.Tensor] = []
    for (cd, ch, cw) in centers:
        d_lo = cd - tD // 2; d_hi = d_lo + tD
        h_lo = ch - tH // 2; h_hi = h_lo + tH
        w_lo = cw - tW // 2; w_hi = w_lo + tW

        d_lo_in, d_hi_in = max(d_lo, 0), min(d_hi, D_vol)
        h_lo_in, h_hi_in = max(h_lo, 0), min(h_hi, H_vol)
        w_lo_in, w_hi_in = max(w_lo, 0), min(w_hi, W_vol)
        slab = vol_t[d_lo_in:d_hi_in, h_lo_in:h_hi_in, w_lo_in:w_hi_in]
        slab = _edge_pad_axis(slab, 0, max(0, -d_lo), max(0, d_hi - D_vol))
        slab = _edge_pad_axis(slab, 1, max(0, -h_lo), max(0, h_hi - H_vol))
        slab = _edge_pad_axis(slab, 2, max(0, -w_lo), max(0, w_hi - W_vol))
        if slab.shape != (tD, tH, tW):
            raise RuntimeError(
                f"build_cubic_batch_native_multi_res: slab shape "
                f"{tuple(slab.shape)} != target {target_shape}")

        cube = slab.unsqueeze(0).unsqueeze(0).float()   # (1, 1, tD, tH, tW)
        view_chunks: List[torch.Tensor] = []
        for (D_k, H_k, W_k) in native_sizes:
            d0 = (tD - D_k) // 2
            h0 = (tH - H_k) // 2
            w0 = (tW - W_k) // 2
            crop = cube[:, :, d0:d0 + D_k, h0:h0 + H_k, w0:w0 + W_k]
            if (D_k, H_k, W_k) != (pD, pH, pW):
                crop = F.interpolate(
                    crop, size=(pD, pH, pW),
                    mode="trilinear", align_corners=False)
            view_chunks.append(crop[0])                 # (1, pD, pH, pW)
        cubes.append(torch.cat(view_chunks, dim=0))     # (C_res, pD, pH, pW)

    return torch.stack(cubes, dim=0).contiguous()       # (B, C_res, pD, pH, pW)


def build_cubic_batch_cpu_multi_res(
    patches: List[np.ndarray],
    centers: List[Tuple[int, int, int]],
    vol: np.ndarray,
    *, pD: int, pH: int, pW: int,
    multi_res_scales: List[float],
    device: torch.device,
) -> torch.Tensor:
    """``cubic`` 模式 CPU 批建造（OFF 路径）。

    ``scale = 1`` 复用已抽取的 ``patch_1x``；``scale != 1`` 重抽 ``round(p * s)`` 后
    resize 回 ``(pD, pH, pW)``。返 ``(B, C_res, pD, pH, pW)`` 已上 GPU。
    """
    batch_list: List[np.ndarray] = []
    for patch_1x, center in zip(patches, centers):
        channels: List[np.ndarray] = []
        for scale in multi_res_scales:
            if scale == 1.0:
                channels.append(patch_1x)
                continue
            sD = int(round(pD * scale))
            sH = int(round(pH * scale))
            sW = int(round(pW * scale))
            patch_s = _extract_cubic_patch(vol, center, (sD, sH, sW))
            patch_s = resize_3d(patch_s, pD, pH, pW, is_label=False)
            channels.append(patch_s)
        batch_list.append(np.stack(channels, axis=0))
    batch = np.stack(batch_list, axis=0)               # (B, C_res, D, H, W)
    return torch.from_numpy(batch).float().to(device, non_blocking=True)


__all__ = [
    "build_z_window_single_res_gpu",
    "build_z_window_native_multi_res_gpu",
    "build_z_window_native_d_gpu",
    "build_z_window_cpu_multi_res",
    "build_cubic_batch_native_multi_res",
    "build_cubic_batch_cpu_multi_res",
]
