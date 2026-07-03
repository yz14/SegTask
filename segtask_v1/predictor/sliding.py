"""Predictor 包 sliding-window 主循环（R6）。

R6 抽自 ``predictor.py``：4 种 ``predict_volume`` 调度路径的主体循环 / blending 累加。

* ``whole_volume_forward``      —— 整卷 resize 单 forward
* ``sliding_window_z``          —— z 轴滑窗（含 2.5D folded / 2.5D native_d / 3D
                                   keep_native / 单分辨率 / 多分辨率 5 种 builder 分派）
* ``sliding_window_z_interleaved``  —— z-interleave wrapper（按 z spacing 拆 k 个互斥子流）
* ``sliding_window_cubic``      —— 3 轴 cubic 滑窗（含 keep_native / CPU 双 builder 分派）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from ..data.dataset import resize_3d
from . import blending as _blending
from . import forwards as _forwards
from . import inputs as _inputs

if TYPE_CHECKING:  # pragma: no cover
    from .predictor import Predictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whole-volume (single forward, no sliding)
# ---------------------------------------------------------------------------
def whole_volume_forward(p: "Predictor", vol: np.ndarray) -> np.ndarray:
    """全卷 resize 到 ``(pD, pH, pW)`` 单次 forward，后 resize 概率回原尺寸。"""
    D_orig, H_orig, W_orig = vol.shape
    pD, pH, pW = p.patch_D, p.patch_H, p.patch_W

    if p.log_progress:
        logger.info(
            "Whole-volume inference: orig=(%d,%d,%d) → model=(%d,%d,%d)",
            D_orig, H_orig, W_orig, pD, pH, pW)

    vol_resized = resize_3d(vol, pD, pH, pW, is_label=False)
    batch = torch.from_numpy(vol_resized[np.newaxis, np.newaxis]) \
        .float().to(p.device, non_blocking=True)
    probs = _forwards.forward_batch_numpy(p, batch)   # (1, num_fg, pD, pH, pW)
    prob_small = probs[0]                             # (num_fg, pD, pH, pW)

    # resize_3d 原生支持领头通道轴（ndim==4）。
    return resize_3d(prob_small, D_orig, H_orig, W_orig, is_label=False)


# ---------------------------------------------------------------------------
# Z-axis sliding window
# ---------------------------------------------------------------------------
def sliding_window_z(p: "Predictor", vol: np.ndarray) -> np.ndarray:
    """z 轴滑窗推理。H/W 总 resize 到模型输入尺寸；z 轴按 stride 滑动并 blend。

    builder 分派（与重构前等价）：

    * ``keep_native_view_depth=True``                                → 2.5D ON, rank-3 (sum(D_k), pH, pW)
    * ``keep_native_multi_res=True`` & ``patch_mode='z_axis'``  → 3D ON, rank-4 (C_res, pD, pH, pW)
    * 单分辨率 ``scales == [1.0]``                              → 单分辨率 GPU
    * 否则                                                     → 多分辨率 CPU 退化 + 一次性上 GPU
    """
    D_orig, H_orig, W_orig = vol.shape
    pD, pH, pW = p.patch_D, p.patch_H, p.patch_W

    stride = max(1, int(pD * (1 - p.overlap)))
    z_positions = _blending.compute_1d_positions(D_orig, pD, stride)

    if p.log_progress:
        logger.info(
            "Z-axis sliding window: D_patch=%d, stride=%d, num_windows=%d, "
            "scales=%s, blend=%s",
            pD, stride, len(z_positions), p.multi_res_scales, p.blend_mode)

    # GPU 常驻：体积在 GPU，F.interpolate 替代 scipy.ndimage.zoom；仅最后 blend
    # 后的概率体返 host。累加器 dtype/落点由 predict.acc_dtype /
    # accumulate_on_cpu 控制（大卷 × 多类的显存逃生门）。
    vol_t = torch.from_numpy(vol).to(p.device, non_blocking=True)
    z_weight_t = torch.from_numpy(_blending.build_1d_weight(pD)).to(
        device=p.acc_device, dtype=p.acc_dtype)       # (pD,)

    acc_pred = torch.zeros(
        (p.num_fg, D_orig, H_orig, W_orig),
        dtype=p.acc_dtype, device=p.acc_device)
    acc_weight = torch.zeros(
        (1, D_orig, 1, 1), dtype=p.acc_dtype, device=p.acc_device)

    # 多分辨率 z 轴实践上少见（2.5D 强制 [1.0]）；单分辨率走 GPU 抽取路径。
    single_res = (len(p.multi_res_scales) == 1
                  and p.multi_res_scales[0] == 1.0)
    # 3D ON 路径：GPU builder 返 (C_res,pD,pH,pW)，与旧多分辨率 z_axis 布局一致。
    keep_native_3d = bool(p.keep_native_multi_res
                          and p.patch_mode == "z_axis")

    window_inputs: List[torch.Tensor] = []
    patch_metas: List[Tuple[int, int, int]] = []   # (z0, z1, actual_d)

    n_windows = len(z_positions)
    for idx, (z0, z1) in enumerate(z_positions):
        actual_d = z1 - z0
        if p.keep_native_view_depth:
            # 2.5D ON: rank-3 (sum(D_k), pH, pW)。
            window_inputs.append(
                _inputs.build_z_window_native_d_gpu(
                    vol_t, z0, z1,
                    pH=pH, pW=pW,
                    eD_max=p._eD_max, view_depths=p.per_view_depths))
        elif keep_native_3d:
            # 3D ON: rank-4 (C_res, pD, pH, pW)。
            window_inputs.append(
                _inputs.build_z_window_native_multi_res_gpu(
                    vol_t, z0, z1,
                    pD=pD, pH=pH, pW=pW,
                    target_shape=p._mr_target_shape,
                    native_sizes=p._mr_native_sizes))
        elif single_res:
            window_inputs.append(
                _inputs.build_z_window_single_res_gpu(
                    vol_t, z0, z1,
                    pD=pD, pH=pH, pW=pW,
                    z_boundary_mode=p.z_boundary_mode))
        else:
            # 多分辨率退化：CPU builder 后一次上 GPU。
            wi_np = _inputs.build_z_window_cpu_multi_res(
                vol, z0, z1,
                pD=pD, pH=pH, pW=pW,
                multi_res_scales=p.multi_res_scales,
                z_boundary_mode=p.z_boundary_mode)
            window_inputs.append(
                torch.from_numpy(wi_np).to(p.device, non_blocking=True))
        patch_metas.append((z0, z1, actual_d))

        is_last = idx == n_windows - 1
        if len(window_inputs) >= p.batch_size or is_last:
            # (B, C_res, pD, pH, pW)
            batch = torch.stack(window_inputs, dim=0).float()
            # (B, num_fg, pD, pH, pW) on GPU
            probs = _forwards.forward_batch_gpu(p, batch)

            # 按 actual_d 分组 → 合并为一次 F.interpolate。常见场景 ad==pD，仅一次上采样。
            groups: Dict[int, List[int]] = {}
            for i, (_, _, ad) in enumerate(patch_metas):
                groups.setdefault(ad, []).append(i)

            for ad, idxs in groups.items():
                sub = probs[idxs]                     # (b, num_fg, pD, pH, pW)

                # 倒 resize 回原几何：edge_pad+ad<pD 时仅取中心 ad 切片不插值 z（H/W 可 resize）；
                # 其余走一次性 trilinear resize 到 (ad, H_orig, W_orig)。
                if p.z_boundary_mode == "edge_pad" and ad < pD:
                    pad_before = (pD - ad) // 2
                    sub = sub[:, :, pad_before:pad_before + ad, :, :]
                    if (H_orig != pH) or (W_orig != pW):
                        sub = F.interpolate(
                            sub, size=(ad, H_orig, W_orig),
                            mode="trilinear", align_corners=False)
                elif (ad != pD) or (H_orig != pH) or (W_orig != pW):
                    sub = F.interpolate(
                        sub, size=(ad, H_orig, W_orig),
                        mode="trilinear", align_corners=False)
                # 逐 ad 对称 blending 权重（与累加器同 dtype/device）。
                if ad == pD:
                    w = z_weight_t
                else:
                    w = torch.from_numpy(
                        _blending.build_1d_weight(ad)).to(
                            device=p.acc_device, dtype=p.acc_dtype)
                w_4d = rearrange(w, 'c -> 1 c 1 1')

                # 一次性转到累加器 dtype/device（CPU 逃生门时为 GPU→CPU 拷贝）。
                sub = sub.to(device=p.acc_device, dtype=p.acc_dtype)
                for j, i in enumerate(idxs):
                    zs, ze, _ = patch_metas[i]
                    # in-place fused mul-add。
                    acc_pred[:, zs:ze, :, :].addcmul_(
                        sub[j], w_4d, value=1.0)
                    acc_weight[:, zs:ze, :, :].add_(w_4d)

            window_inputs.clear()
            patch_metas.clear()

            if p.log_progress and (
                    (idx + 1) % max(1, 10 * p.batch_size) == 0 or is_last):
                logger.info("  z-window %d/%d", idx + 1, n_windows)

    return _finalize_accumulators(acc_pred, acc_weight)


# ---------------------------------------------------------------------------
# Z-axis sliding window — interleaved wrapper
# ---------------------------------------------------------------------------
def choose_interleave_factor(p: "Predictor", z_spacing: float) -> int:
    """根据物理 z spacing (mm) 选 ``k``：首个 ``thresholds[j] >= z_spacing`` 返 ``factors[j]``，否则 fallback。"""
    thresholds = p.z_interleave_thresholds
    factors = p.z_interleave_factors
    for t, f in zip(thresholds, factors):
        if z_spacing <= float(t):
            return max(1, int(f))
    return max(1, int(factors[-1]))


def sliding_window_z_interleaved(p: "Predictor", vol: np.ndarray,
                                 z_spacing: float) -> np.ndarray:
    """2.5D z-交错推理：按 stride-``k`` 拆为 ``k`` 个互斥子体独立推理后以
    ``out[:, i::k] = stream_i`` 缝回。``k <= 1`` 时退化为标准 ``sliding_window_z``。
    覆盖全划分，缝接无需跨流加权。
    """
    k = choose_interleave_factor(p, z_spacing)
    if k <= 1:
        if p.log_progress:
            logger.info(
                "z-interleave: z_spacing=%.4f mm → k=1 (no split); "
                "falling through to standard 2.5D z-sliding window.",
                z_spacing)
        return sliding_window_z(p, vol)

    D, H, W = vol.shape
    if p.log_progress:
        logger.info(
            "z-interleave: z_spacing=%.4f mm → k=%d. Splitting volume "
            "(D=%d) into %d disjoint stride-%d sub-streams; per-stream "
            "depths=%s.",
            z_spacing, k, D, k, k,
            [int(np.ceil((D - i) / k)) for i in range(k)])

    out = np.zeros((p.num_fg, D, H, W), dtype=np.float32)
    for i in range(k):
        # vol[i::k] 为 view；copy 以免下游 in-place 错误。
        sub_vol = np.ascontiguousarray(vol[i::k])
        sub_D = sub_vol.shape[0]
        if p.log_progress:
            logger.info(
                "  z-interleave stream %d/%d: indices=%d::%d, sub_D=%d",
                i + 1, k, i, k, sub_D)
        sub_prob = sliding_window_z(p, sub_vol)
        # 防御：sliding_window_z 须保证输出深度 == 输入深度。
        if sub_prob.shape != (p.num_fg, sub_D, H, W):
            raise RuntimeError(
                f"z-interleave stream {i}: expected sub-prob shape "
                f"({p.num_fg}, {sub_D}, {H}, {W}), got "
                f"{tuple(sub_prob.shape)}")
        out[:, i::k, :, :] = sub_prob
    return out


# ---------------------------------------------------------------------------
# Cubic 3-axis sliding window
# ---------------------------------------------------------------------------
def sliding_window_cubic(p: "Predictor", vol: np.ndarray) -> np.ndarray:
    """3 轴 ``cubic`` 滑窗推理（带 overlap blending）。

    builder 分派：

    * ``keep_native_multi_res=True``  → ``build_cubic_batch_native_multi_res``（GPU 全流）
    * 否则                            → ``build_cubic_batch_cpu_multi_res``（CPU patch 列表 → batch）
    """
    D_orig, H_orig, W_orig = vol.shape
    pD, pH, pW = p.patch_D, p.patch_H, p.patch_W

    stride_d = max(1, int(pD * (1 - p.overlap)))
    stride_h = max(1, int(pH * (1 - p.overlap)))
    stride_w = max(1, int(pW * (1 - p.overlap)))
    pos_d = _blending.compute_1d_positions(D_orig, pD, stride_d)
    pos_h = _blending.compute_1d_positions(H_orig, pH, stride_h)
    pos_w = _blending.compute_1d_positions(W_orig, pW, stride_w)

    total_windows = len(pos_d) * len(pos_h) * len(pos_w)
    if p.log_progress:
        logger.info(
            "Cubic sliding window: patch=(%d,%d,%d), strides=(%d,%d,%d), "
            "windows=%d×%d×%d=%d, blend=%s",
            pD, pH, pW, stride_d, stride_h, stride_w,
            len(pos_d), len(pos_h), len(pos_w), total_windows, p.blend_mode)

    # 累加器 dtype/落点由 predict.acc_dtype / accumulate_on_cpu 控制（与 z 路径
    # 一致；默认 fp32 常驻 device，仅最后一次返 host）。
    weight_3d = torch.from_numpy(
        _blending.build_3d_weight(pD, pH, pW, p.blend_mode)).to(
            device=p.acc_device, dtype=p.acc_dtype)

    acc_pred = torch.zeros(
        (p.num_fg, D_orig, H_orig, W_orig),
        dtype=p.acc_dtype, device=p.acc_device)
    acc_weight = torch.zeros(
        (1, D_orig, H_orig, W_orig), dtype=p.acc_dtype, device=p.acc_device)

    # 3D cubic ON 路径：体积一次上 GPU，builder 全程 on-device（逐视图一次 F.interpolate，
    # 零 scipy.ndimage.zoom）。OFF 路径保旧 CPU pipeline。
    keep_native_3d = bool(p.keep_native_multi_res
                          and p.patch_mode == "cubic")
    vol_t: Optional[torch.Tensor] = (
        torch.from_numpy(vol).float().to(p.device, non_blocking=True)
        if keep_native_3d else None)

    patches: List[np.ndarray] = []
    coords: List[Tuple[int, int, int, int, int, int, int, int, int]] = []
    centers: List[Tuple[int, int, int]] = []
    processed = 0

    def _flush() -> None:
        nonlocal processed
        if not patches:
            return
        if keep_native_3d:
            batch = _inputs.build_cubic_batch_native_multi_res(
                centers, vol_t,
                pD=pD, pH=pH, pW=pW,
                target_shape=p._mr_target_shape,
                native_sizes=p._mr_native_sizes)
        else:
            batch = _inputs.build_cubic_batch_cpu_multi_res(
                patches, centers, vol,
                pD=pD, pH=pH, pW=pW,
                multi_res_scales=p.multi_res_scales,
                device=p.device)
        probs = _forwards.forward_batch_gpu(p, batch)   # (B, num_fg, pD, pH, pW) on GPU
        # 一次性转到累加器 dtype/device（CPU 逃生门时为 GPU→CPU 拷贝）。
        probs = probs.to(device=p.acc_device, dtype=p.acc_dtype)
        for pred, (d0, d1, h0, h1, w0, w1, ad, ah, aw) in zip(probs, coords):
            # Trim prediction to actual (non-padded) size in each axis
            pred_trim = pred[:, :ad, :ah, :aw]
            w_trim = weight_3d[:ad, :ah, :aw].unsqueeze(0)   # (1, ad, ah, aw)
            acc_pred[:, d0:d0 + ad, h0:h0 + ah, w0:w0 + aw].addcmul_(
                pred_trim, w_trim, value=1.0)
            acc_weight[:, d0:d0 + ad, h0:h0 + ah, w0:w0 + aw].add_(w_trim)
        processed += len(patches)
        if p.log_progress and (
                processed % max(1, 10 * p.batch_size) == 0
                or processed == total_windows):
            logger.info("  cubic window %d/%d", processed, total_windows)
        patches.clear()
        coords.clear()
        centers.clear()

    for d0, d1 in pos_d:
        for h0, h1 in pos_h:
            for w0, w1 in pos_w:
                patch = vol[d0:d1, h0:h1, w0:w1]
                ad, ah, aw = patch.shape

                # 填短尾窗口到 (pD,pH,pW)。默认 mode='edge' 复制边界（归一化后 0 不是空气）。
                if ad < pD or ah < pH or aw < pW:
                    pad_width = ((0, pD - ad), (0, pH - ah), (0, pW - aw))
                    if p.pad_value is None:
                        patch = np.pad(patch, pad_width, mode="edge")
                    else:
                        patch = np.pad(
                            patch, pad_width, mode="constant",
                            constant_values=p.pad_value)

                patches.append(patch)
                coords.append((d0, d1, h0, h1, w0, w1, ad, ah, aw))
                centers.append(
                    ((d0 + d1) // 2, (h0 + h1) // 2, (w0 + w1) // 2))

                if len(patches) >= p.batch_size:
                    _flush()

    _flush()

    return _finalize_accumulators(acc_pred, acc_weight)


def _finalize_accumulators(acc_pred: torch.Tensor,
                           acc_weight: torch.Tensor) -> np.ndarray:
    """加权累加器 → fp32 概率体 (numpy)。

    clamp 下界按 dtype 选：fp16 下 1e-8 会下溢为 0 失去保护，改用其最小
    正规数量级；除法在累加器 dtype 上就地完成（不额外峰值显存），返回前转 fp32。
    """
    eps = 6.1e-5 if acc_pred.dtype == torch.float16 else 1e-8
    acc_weight.clamp_(min=eps)
    acc_pred /= acc_weight
    return acc_pred.float().cpu().numpy()


__all__ = [
    "whole_volume_forward",
    "sliding_window_z",
    "sliding_window_z_interleaved",
    "sliding_window_cubic",
    "choose_interleave_factor",
]
