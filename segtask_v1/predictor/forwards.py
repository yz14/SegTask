"""Predictor 包 forward + TTA + diag 子模块（R6）。

R6 抽自 ``predictor.py``：3 种 forward（3D / 2.5D folded / 2.5D lift）+ 2 种 TTA（3D
轴 flip 7 种组合 / 2.5D 仅 H/W flip 3 种）+ 1 个首 batch 诊断日志器，全部改为
**模块级函数**，显式接收 ``predictor`` 引用以读取模型 / dtype / num_fg / 阈值等
状态。行为与重构前严格等价。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from einops import rearrange
from torch.amp import autocast

if TYPE_CHECKING:  # pragma: no cover
    from .predictor import Predictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2.5D input reshape  (folded layout: rank-5 → rank-4)
# ---------------------------------------------------------------------------
def reshape_2_5d_input(p: "Predictor", x: torch.Tensor) -> torch.Tensor:
    """折 2.5D 输入的 ``C_res`` 轴 → ``(B, C_res*D, H, W)``，与 ``Trainer._squeeze_2_5d`` 一致。"""
    if x.ndim != 5:
        raise ValueError(
            "2.5D inference expects rank-5 input "
            f"(B, C_res, D, H, W); got x.shape={tuple(x.shape)}")
    _, _, D, _, _ = x.shape
    if D != p.patch_D:
        raise ValueError(
            f"2.5D input D-axis ({D}) != patch_D ({p.patch_D}). "
            "Window builder produced an unexpected slice count.")
    return rearrange(x, 'b c d h w -> b (c d) h w').contiguous()


# ---------------------------------------------------------------------------
# TTA — 3D (7 flip combos) and 2.5D (3 H/W combos)
# ---------------------------------------------------------------------------
# (flip_x_dims, flip_prob_dims) 规格表：x 与 prob 的轴布局可能不同，故分开记录。
# 3D：x=(B,C,D,H,W)、prob=(B,num_fg,D,H,W) 同布局，flip 轴一致（2=D,3=H,4=W）。
_FLIP_SPECS_3D = (
    ([2], [2]), ([3], [3]), ([4], [4]),
    ([2, 3], [2, 3]), ([2, 4], [2, 4]), ([3, 4], [3, 4]),
    ([2, 3, 4], [2, 3, 4]),
)
# 2.5D：x_2d=(B,C_res*D,H,W) 仅 H/W 可 flip（2=H,3=W）；prob=(B,num_fg,D,H,W)（3=H,4=W）。
# D 是输入通道轴，flip 会反转物理切片顺序 → 分布偏移，故不翻 D。
_FLIP_SPECS_2_5D = (
    ([2], [3]),        # H
    ([3], [4]),        # W
    ([2, 3], [3, 4]),  # H + W
)


def _tta_chunk_size(p: "Predictor") -> int:
    """单次前向拼接的 flip 变体数：``tta_batch_size`` (None→batch_size)，下限 1。

    AdaBN per_volume 估计期 (``p._adabn_estimating``) 强制返回 1（串行）：估计期 BN
    处于 train+累积平均模式，把多个变体拼成大 batch 会改变 BN 见到的 batch 统计构成与
    running stats 累积，破坏与逐变体串行实现的一致性。真实 eval 预测不受影响。
    """
    if p._adabn_estimating:
        return 1
    cs = p.tta_batch_size or p.batch_size
    return max(1, int(cs))


def _flip_tta_batched(p: "Predictor", x: torch.Tensor,
                      base_prob: torch.Tensor, flip_specs, post_fn):
    """通用 flip-TTA：``base_prob`` 为原图概率，``flip_specs`` 中各 flip 变体按
    ``_tta_chunk_size`` 分块——每块沿 batch 轴 ``torch.cat`` 成 ``(B*g, ...)`` 一次
    前向、``post_fn`` 转概率、逐变体反 flip 后累加，最后除以 ``1+len(flip_specs)``。

    与逐变体串行实现严格等价（eval 下 BN 用 running stats、变体间无 batch 耦合；仅前向
    顺序/批大小改变）；累加顺序也保持原图→变体序，浮点结果同序。
    """
    total = base_prob.clone()
    count = float(1 + len(flip_specs))
    B = x.shape[0]
    chunk = _tta_chunk_size(p)
    for start in range(0, len(flip_specs), chunk):
        specs = flip_specs[start:start + chunk]
        x_cat = torch.cat([torch.flip(x, fx) for fx, _ in specs], dim=0)
        pred = p.model(x_cat.to(p.model_dtype))
        if isinstance(pred, list):
            pred = pred[0]
        prob_cat = post_fn(pred)                  # (B*g, num_fg, ...)
        for j, (_, fprob) in enumerate(specs):
            prob_j = torch.flip(prob_cat[j * B:(j + 1) * B], fprob)
            total = total + prob_j
    return total / count


def tta_flip_ensemble(p: "Predictor", x: torch.Tensor,
                      base_prob: torch.Tensor) -> torch.Tensor:
    """3D TTA：原始 + 7 种轴 flip 组合取均；变体按 ``tta_batch_size`` 批量化前向。"""
    def _post(pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(pred.float())[:, :p.num_fg]

    return _flip_tta_batched(p, x, base_prob, _FLIP_SPECS_3D, _post)


def tta_flip_ensemble_2_5d(p: "Predictor", x_2d: torch.Tensor,
                           base_prob: torch.Tensor) -> torch.Tensor:
    """2.5D TTA：仅 H/W flip；变体按 ``tta_batch_size`` 批量化前向。"""
    D = p.patch_D

    def _post(pred: torch.Tensor) -> torch.Tensor:
        # (B*g, num_fg*D, H, W) → (B*g, num_fg, D, H, W)
        pred_5d = rearrange(
            pred, 'b (c d) h w -> b c d h w', c=p.num_fg, d=D)
        return torch.sigmoid(pred_5d.float())

    return _flip_tta_batched(p, x_2d, base_prob, _FLIP_SPECS_2_5D, _post)


# ---------------------------------------------------------------------------
# Diagnostics — log per-volume first-batch input/logits/prob stats
# ---------------------------------------------------------------------------
@torch.no_grad()
def diag_log_first_batch(p: "Predictor", tag: str,
                         x: torch.Tensor, logits: torch.Tensor,
                         prob: torch.Tensor) -> None:
    """逐卷一次性诊断：input / logits / sigmoid 的 stats + 阈上比例 + NaN 计数。

    区分"模型本身饱和"（logits ≫ 5, sigmoid ≈ 1, frac ≈ 1）与"blend / 后处理坍塌"。
    """
    if p._diag_first_batch_logged:
        return
    p._diag_first_batch_logged = True

    def _q3(t: torch.Tensor, qs):
        """采样式分位数：torch.quantile 在 CUDA 上对 >~1.6e7 元素会拒；
        用整数 stride 切片避免 float32 linspace 越界。"""
        flat = t.detach().float().flatten()
        n = flat.numel()
        cap = 1_000_000
        if n > cap:
            stride = max(1, n // cap)
            flat = flat[::stride]
        qs_t = torch.tensor(qs, device=flat.device, dtype=flat.dtype)
        return torch.quantile(flat, qs_t).cpu().tolist()

    try:
        xs = x.detach().float()
        ls = logits.detach().float()
        ps = prob.detach().float()
        xq = _q3(xs, [0.01, 0.5, 0.99])
        lq = _q3(ls, [0.01, 0.5, 0.99])
        pq = _q3(ps, [0.5, 0.9, 0.99])
        n_nan_logits = int(torch.isnan(ls).sum().item())
        n_nan_prob = int(torch.isnan(ps).sum().item())
        frac_thr = float((ps >= p.threshold_min).float().mean().item())
        logger.info(
            "[diag/forward %s] input: shape=%s, min=%.4f, max=%.4f, "
            "mean=%.4f, q1=%.4f, q50=%.4f, q99=%.4f",
            tag, tuple(xs.shape),
            float(xs.min()), float(xs.max()), float(xs.mean()),
            float(xq[0]), float(xq[1]), float(xq[2]))
        logger.info(
            "[diag/forward %s] logits: shape=%s, min=%.4f, max=%.4f, "
            "mean=%.4f, q1=%.4f, q50=%.4f, q99=%.4f, n_nan=%d",
            tag, tuple(ls.shape),
            float(ls.min()), float(ls.max()), float(ls.mean()),
            float(lq[0]), float(lq[1]), float(lq[2]), n_nan_logits)
        logger.info(
            "[diag/forward %s] sigmoid: shape=%s, min=%.4f, max=%.4f, "
            "mean=%.4f, q50=%.4f, q90=%.4f, q99=%.4f, "
            "frac(prob>=%.2f)=%.4f, n_nan=%d",
            tag, tuple(ps.shape),
            float(ps.min()), float(ps.max()), float(ps.mean()),
            float(pq[0]), float(pq[1]), float(pq[2]),
            p.threshold_min, frac_thr, n_nan_prob)
        if n_nan_logits > 0 or n_nan_prob > 0:
            logger.error(
                "[diag/forward %s] NaN detected (logits=%d, prob=%d). "
                "This is the root cause of the 'all-foreground' "
                "predictions — re-run with '--precision bf16'.",
                tag, n_nan_logits, n_nan_prob)
    except Exception as _e:
        logger.warning("[diag/forward %s] stat failed: %s", tag, _e)


# ---------------------------------------------------------------------------
# Forward dispatch — return GPU tensor (for sliding-window blending paths)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_batch_gpu(p: "Predictor", x: torch.Tensor) -> torch.Tensor:
    """Mode-aware forward；返 ``(B, num_fg, D, H, W)`` 概率张量（GPU 常驻，含 TTA）。

    分派优先级（与重构前 ``forward_batch_gpu`` 等价）：

    * ``patch_mode == '2_5d'`` & ``lift_2_5d_to_3d``    → rank-5 直送 3D 模型，3D TTA
    * ``patch_mode == '2_5d'`` & 否则                   → 折 ``C_res*D`` 走 2D 模型，2.5D TTA
    * 其他                                              → 3D 模型，3D TTA
    """
    if p.patch_mode == "2_5d":
        # Plan A lift：rank-5 (B,n_views,pD,pH,pW) 直送三维 UNet，输出 (B,num_fg,pD,pH,pW)。
        # TTA 复用 3D ensemble（D 是真空间轴）。
        if p.lift_2_5d_to_3d:
            if x.ndim != 5:
                raise ValueError(
                    "lift_2_5d_to_3d=True expects rank-5 input "
                    f"(B, n_views, D, H, W); got x.shape={tuple(x.shape)}")
            with autocast(device_type="cuda", enabled=p.use_amp,
                          dtype=p.amp_dtype):
                pred = p.model(x.to(p.model_dtype))
                if isinstance(pred, list):
                    pred = pred[0]
                if pred.shape[1] < p.num_fg:
                    raise ValueError(
                        f"Lift-mode model output has {pred.shape[1]} "
                        f"channels at dim 1; expected at least "
                        f"num_fg={p.num_fg}.")
                prob = torch.sigmoid(pred.float())[:, :p.num_fg]
                diag_log_first_batch(p, "2.5D lift", x, pred[:, :p.num_fg], prob)
                if p.tta_flip:
                    prob = tta_flip_ensemble(p, x, prob)
            return prob

        # 两种输入：OFF rank-5 (B,C_res,pD,pH,pW) 需折 C_res*D；
        # ON rank-4 (B,sum(D_k),H,W) 已在入参布局，直接透传。
        if x.ndim == 5:
            x_2d = reshape_2_5d_input(p, x)         # (B, C_res*D, H, W)
        elif x.ndim == 4:
            x_2d = x
        else:
            raise ValueError(
                f"2.5D forward expects rank-4 or rank-5 input; "
                f"got x.shape={tuple(x.shape)}")
        D = p.patch_D
        with autocast(device_type="cuda", enabled=p.use_amp,
                      dtype=p.amp_dtype):
            pred = p.model(x_2d.to(p.model_dtype))
            if isinstance(pred, list):
                pred = pred[0]
            expected_c = p.num_fg * D
            if pred.shape[1] != expected_c:
                raise ValueError(
                    f"2.5D model output channels {pred.shape[1]} != "
                    f"num_fg*D = {p.num_fg}*{D} = {expected_c}")
            pred_5d = rearrange(
                pred, 'b (c d) h w -> b c d h w', c=p.num_fg, d=D)
            prob = torch.sigmoid(pred_5d.float())
            diag_log_first_batch(p, "2.5D folded", x_2d, pred_5d, prob)
            if p.tta_flip:
                prob = tta_flip_ensemble_2_5d(p, x_2d, prob)
        return prob

    # 3D
    with autocast(device_type="cuda", enabled=p.use_amp,
                  dtype=p.amp_dtype):
        pred = p.model(x.to(p.model_dtype))
        if isinstance(pred, list):
            pred = pred[0]
        assert pred.shape[1] >= p.num_fg, (
            f"Model output has {pred.shape[1]} channels; "
            f"expected at least num_fg={p.num_fg} at 1x resolution.")
        prob = torch.sigmoid(pred.float())[:, :p.num_fg]
        diag_log_first_batch(p, "3D", x, pred[:, :p.num_fg], prob)
        if p.tta_flip:
            prob = tta_flip_ensemble(p, x, prob)
    return prob


# ---------------------------------------------------------------------------
# Forward — return numpy (for whole-volume + cubic CPU-blending paths)
# ---------------------------------------------------------------------------
def forward_batch_numpy(p: "Predictor", x: torch.Tensor) -> np.ndarray:
    """numpy-返版 forward：``(B, num_fg, D, H, W)`` fp32。等价于
    ``forward_batch_gpu(...).cpu().numpy()``，保留这个独立入口仅为类侧 shim 命名一致。

    注意：与 ``forward_batch_gpu`` 的细微差别——重构前的 numpy 返回版 forward
    *不* 调 diag logger（仅 GPU 路径调）。本函数复刻该行为，禁用 diag。
    """
    if p.patch_mode == "2_5d":
        return forward_batch_2_5d_numpy(p, x)

    with autocast(device_type="cuda", enabled=p.use_amp, dtype=p.amp_dtype):
        pred = p.model(x.to(p.model_dtype))
        if isinstance(pred, list):
            pred = pred[0]
        assert pred.shape[1] >= p.num_fg, (
            f"Model output has {pred.shape[1]} channels; "
            f"expected at least num_fg={p.num_fg} at 1x resolution.")
        prob = torch.sigmoid(pred.float())[:, :p.num_fg]
        if p.tta_flip:
            prob = tta_flip_ensemble(p, x, prob)
    return prob.float().cpu().numpy()


def forward_batch_2_5d_numpy(p: "Predictor", x: torch.Tensor) -> np.ndarray:
    """2.5D forward (numpy 返)；折 C_res 走 2D 模型 / lift 走 3D 模型。"""
    if p.lift_2_5d_to_3d:
        if x.ndim != 5:
            raise ValueError(
                "lift_2_5d_to_3d=True expects rank-5 input "
                f"(B, n_views, D, H, W); got x.shape={tuple(x.shape)}")
        with autocast(device_type="cuda", enabled=p.use_amp,
                      dtype=p.amp_dtype):
            pred = p.model(x.to(p.model_dtype))
            if isinstance(pred, list):
                pred = pred[0]
            if pred.shape[1] < p.num_fg:
                raise ValueError(
                    f"Lift-mode model output has {pred.shape[1]} "
                    f"channels at dim 1; expected at least "
                    f"num_fg={p.num_fg}.")
            prob = torch.sigmoid(pred.float())[:, :p.num_fg]
            if p.tta_flip:
                prob = tta_flip_ensemble(p, x, prob)
        return prob.float().cpu().numpy()

    x_2d = reshape_2_5d_input(p, x)                # (B, C_res*D, H, W)
    D = p.patch_D
    with autocast(device_type="cuda", enabled=p.use_amp, dtype=p.amp_dtype):
        pred = p.model(x_2d.to(p.model_dtype))
        if isinstance(pred, list):
            pred = pred[0]
        expected_c = p.num_fg * D
        if pred.shape[1] != expected_c:
            raise ValueError(
                f"2.5D model output channels {pred.shape[1]} != "
                f"num_fg*D = {p.num_fg}*{D} = {expected_c}")
        pred_5d = rearrange(
            pred, 'b (c d) h w -> b c d h w', c=p.num_fg, d=D)
        prob = torch.sigmoid(pred_5d.float())
        if p.tta_flip:
            prob = tta_flip_ensemble_2_5d(p, x_2d, prob)
    return prob.float().cpu().numpy()


__all__ = [
    "reshape_2_5d_input",
    "tta_flip_ensemble",
    "tta_flip_ensemble_2_5d",
    "diag_log_first_batch",
    "forward_batch_gpu",
    "forward_batch_numpy",
    "forward_batch_2_5d_numpy",
]
