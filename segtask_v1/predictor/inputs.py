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

from taskcore.data.dataset import extract_cubic_patch, extract_z_patch_padded, resize_3d

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


def _extract_z_range_padded(
    vol: np.ndarray, z_start: int, depth: int) -> np.ndarray:
    """从绝对起点 ``z_start`` 抽 ``depth`` 切片；越界部分 mode='edge' 复制边界。

    ``extract_z_patch_padded`` 的起点锚定版（后者按窗心锚定）。"""
    D_vol = vol.shape[0]
    lo, hi = z_start, z_start + depth
    src_lo, src_hi = max(lo, 0), min(hi, D_vol)
    pad_before, pad_after = max(-lo, 0), max(hi - D_vol, 0)
    patch = vol[src_lo:src_hi]
    if pad_before or pad_after:
        patch = np.pad(
            patch, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    return patch


def content_pad_before(patch_len: int, actual_len: int) -> int:
    """短窗内容在 patch 轴上的居中填充前置量（单一真相源）。

    builder 与 blend 必须共用同一记账：内容 ``[a0, a0+actual)`` 落在 patch
    坐标 ``[pad_before, pad_before+actual)``，blend 从同一偏移处 trim 回贴。
    """
    return (patch_len - actual_len) // 2 if actual_len < patch_len else 0


def _gaussian_blur3d(x: torch.Tensor,
                     sigmas: Tuple[float, float, float]) -> torch.Tensor:
    """分离式 3D 高斯模糊（GPU）：镜像 ``scipy.ndimage.gaussian_filter``
    （truncate=4.0，mode='nearest' → replicate padding）。``x`` 形如
    ``(N, 1, D, H, W)``。"""
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        radius = int(4.0 * sigma + 0.5)
        if radius < 1:
            continue
        coords = torch.arange(
            -radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        pad = [0, 0, 0, 0, 0, 0]  # F.pad 顺序 (W_l, W_r, H_l, H_r, D_l, D_r)
        pad_idx = (2 - axis) * 2
        pad[pad_idx] = pad[pad_idx + 1] = radius
        x = F.pad(x, pad, mode="replicate")
        x = F.conv3d(x, kernel.view(shape))
    return x


def resize_trilinear(x: torch.Tensor, size: Tuple[int, int, int],
                     *, antialias: bool = False) -> torch.Tensor:
    """GPU trilinear resize，镜像训练侧 ``resize_3d``（scipy zoom）语义：

    * 采样网格：zoom ``grid_mode=False`` ≡ ``align_corners=True``；
    * 抗混叠：下采样轴按 ``sigma = (1/f - 1) * 0.5`` 预高斯模糊
      （同 ``resize_3d(anti_alias=True)``）。

    ``x`` 形如 ``(N, C, D, H, W)``（antialias 路径要求 ``C == 1``）。"""
    in_shape = tuple(x.shape[2:])
    if in_shape == tuple(size):
        return x
    if antialias:
        sigmas = tuple(
            max((i / o - 1.0) * 0.5, 0.0)
            for i, o in zip(in_shape, size))
        if any(s > 0 for s in sigmas):
            x = _gaussian_blur3d(x, sigmas)
    return F.interpolate(
        x, size=tuple(size), mode="trilinear", align_corners=True)


def _extract_z_slab_resized(vol_t: torch.Tensor, z_start: int, eD: int,
                            pH: int, pW: int,
                            antialias: bool = False) -> torch.Tensor:
    """共享子例程：从绝对起点 ``z_start`` 抽 ``eD`` 深 slab、edge-replicate、面内 resize 到 ``(pH, pW)``。

    供 native_multi_res 与 native_d 共享。返 ``(eD, pH, pW)`` fp32。
    """
    D_vol = vol_t.shape[0]
    zlo = z_start
    zhi = zlo + eD
    zlo_in = max(zlo, 0)
    zhi_in = min(zhi, D_vol)
    slab = vol_t[zlo_in:zhi_in]
    slab = _edge_pad_z(slab, max(0, -zlo), max(0, zhi - D_vol))
    if slab.shape[0] != eD:
        raise RuntimeError(
            f"_extract_z_slab_resized: expected depth {eD}, got "
            f"{slab.shape[0]} (z_start={z_start}, D_vol={D_vol}).")

    H_orig, W_orig = slab.shape[1], slab.shape[2]
    slab = slab.unsqueeze(0).unsqueeze(0).float()  # (1,1,eD,H,W)
    if H_orig != pH or W_orig != pW:
        # 镜像训练侧 dataset 面内 resize（resize_3d）语义。
        slab = resize_trilinear(
            slab, (eD, pH, pW), antialias=antialias)
    return slab[0, 0]                              # (eD, pH, pW)


# ---------------------------------------------------------------------------
# Z-axis window builders (GPU)
# ---------------------------------------------------------------------------
def build_z_window_single_res_gpu(
    vol_t: torch.Tensor, z0: int, z1: int,
    *, pD: int, pH: int, pW: int, z_boundary_mode: str,
    antialias: bool = False,
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
        # 镜像训练侧 resize_3d 语义（align_corners=True + 可选抗混叠）。
        patch = resize_trilinear(
            patch, (pD, pH, pW), antialias=antialias)
    return patch.squeeze(0)                          # (1, pD, pH, pW)


def build_z_window_native_multi_res_gpu(
    vol_t: torch.Tensor, z0: int, z1: int,
    *, pD: int, pH: int, pW: int,
    target_shape: Tuple[int, int, int],
    native_sizes: List[Tuple[int, int, int]],
    antialias: bool = False,
) -> torch.Tensor:
    """3D ``z_axis`` ON 模式窗口建造：抽单 max-FOV cube → 面内 resize 到 ``(pH, pW)`` →
    逐视图中心裁 ``D_k`` 后 D 轴 trilinear 回 ``pD`` → 拼 ``C_res``。

    ``target_shape = (eD_max, pH, pW)``；``native_sizes[k] = (D_k, H_k=pH, W_k=pW)``。
    返 ``(C_res, pD, pH, pW)``。

    锚点约定：窗口内容 ``[z0, z1)`` 在 view-1x（深 ``pD``）坐标中落在
    ``content_pad_before(pD, ad)`` 偏移处，与 blend 端的 trim 记账一致
    （短窗不再用窗心居中语义，避免 ad 与 pD 奇偶不同时错位 1 体素）。
    """
    eD_max = target_shape[0]
    ad = z1 - z0
    zlo = z0 - content_pad_before(pD, ad) - (eD_max - pD) // 2
    slab = _extract_z_slab_resized(
        vol_t, zlo, eD_max, pH, pW, antialias=antialias)
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
    antialias: bool = False,
) -> torch.Tensor:
    """2.5D ``keep_native_view_depth=True`` 模式窗口建造：抽 ``eD_max`` max-FOV slab、
    面内 resize 到 ``(pH, pW)``、逐视图中心抽 ``D_k`` 切片后拼通道。

    返 ``(sum(D_k), pH, pW)``。

    锚点约定同 ``build_z_window_native_multi_res_gpu``：内容在首视图
    （深 ``view_depths[0]``）中落在 ``content_pad_before`` 偏移处。
    """
    D0 = view_depths[0]
    ad = z1 - z0
    zlo = z0 - content_pad_before(D0, ad) - (eD_max - D0) // 2
    slab = _extract_z_slab_resized(
        vol_t, zlo, eD_max, pH, pW, antialias=antialias)
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
    antialias: bool = False,
) -> np.ndarray:
    """多分辨率 z 窗口堆（CPU 退化路径）。

    * ``scale > 1``: 总走 edge-padded 抽 ``round(pD * scale)`` 切片
    * ``scale == 1``: 按 ``z_boundary_mode`` 选择 ``stretch`` / ``edge_pad``

    返 ``(C_res, pD, pH, pW)`` fp32。
    """
    ad = z1 - z0
    pb = content_pad_before(pD, ad)
    channels: List[np.ndarray] = []
    for scale in multi_res_scales:
        if scale == 1.0:
            if z_boundary_mode == "edge_pad":
                # 锚点与 blend 记账一致：内容落在 pb 偏移处。
                patch = _extract_z_range_padded(vol, z0 - pb, pD)
            else:
                # 旧尾窗行为：取实际切片，后面 resize。
                patch = vol[z0:z1]
        else:
            D_s = int(round(pD * scale))
            patch = _extract_z_range_padded(
                vol, z0 - pb - (D_s - pD) // 2, D_s)
        patch = resize_3d(patch, pD, pH, pW, is_label=False,
                          anti_alias=antialias)
        channels.append(patch)
    return np.stack(channels, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cubic batch builders
# ---------------------------------------------------------------------------
def build_cubic_batch_native_multi_res(
    windows: List[Tuple[int, int, int, int, int, int]],
    vol_t: torch.Tensor,
    *, pD: int, pH: int, pW: int,
    target_shape: Tuple[int, int, int],
    native_sizes: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """3D ``cubic`` ON 模式批建造：逐窗抽单 max-FOV cube → 逐视图中心裁
    ``(D_k, H_k, W_k)`` 后 trilinear 回 ``(pD, pH, pW)`` → 拼 ``C_res``。

    ``windows[i] = (d0, h0, w0, ad, ah, aw)``：窗口原点与各轴实际长度。
    返 ``(B, C_res, pD, pH, pW)``。

    锚点约定：窗口内容在 view-1x 各轴坐标中落在 ``content_pad_before``
    偏移处，与 blend 端 coords 里的 pb_d/pb_h/pb_w trim 记账一致
    （短窗不再用窗心居中语义，避免奇偶不同时错位 1 体素）。
    """
    tD, tH, tW = target_shape
    D_vol, H_vol, W_vol = vol_t.shape

    cubes: List[torch.Tensor] = []
    for (d0, h0, w0, ad, ah, aw) in windows:
        d_lo = d0 - content_pad_before(pD, ad) - (tD - pD) // 2
        h_lo = h0 - content_pad_before(pH, ah) - (tH - pH) // 2
        w_lo = w0 - content_pad_before(pW, aw) - (tW - pW) // 2
        d_hi = d_lo + tD
        h_hi = h_lo + tH
        w_hi = w_lo + tW

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
    antialias: bool = False,
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
            patch_s = extract_cubic_patch(vol, center, (sD, sH, sW))
            patch_s = resize_3d(patch_s, pD, pH, pW, is_label=False,
                                anti_alias=antialias)
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
