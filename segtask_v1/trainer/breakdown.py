"""训练期 per-step 损失分量收集与渲染。

- ``collect_multi_res_breakdown``：把 ``MultiResolutionLoss`` 的 per-res 诊断
  抽到 ``breakdown`` dict，主路键 ``L_res_{r}``、aux 路（lift 时）键
  ``L_aux_res_{r}``。被 ``DeepSupervisionLoss`` 多次调用时，
  ``MultiResolutionLoss.pop_per_res_diag`` 已对 DS 尺度取均值。
- ``format_breakdown``：渲染 epoch / step 末尾的 ``" | L_main=... L_aux_k=..."`` 串。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..losses.losses import DeepSupervisionLoss, MultiResolutionLoss


def collect_multi_res_breakdown(
    criterion,
    aux_loss_fn,
    breakdown: Dict[str, float],
) -> None:
    """从主/aux 损失里抽 per-res 诊断到 ``breakdown``；非 MR 安静跳过。"""
    # 主路：仅当确实被 DS 包过才取内层；MultiResolutionLoss 自身也有
    # .base_loss（指向内层 base 损失），盲用 getattr 会多解一层导致无 DS
    # 时诊断丢失且 _per_res_history 永不清空。
    main_inner = (criterion.base_loss
                  if isinstance(criterion, DeepSupervisionLoss)
                  else criterion)
    if isinstance(main_inner, MultiResolutionLoss):
        diag = main_inner.pop_per_res_diag()
        if diag is not None:
            for r, v in enumerate(diag):
                if math.isfinite(v):
                    breakdown[f"L_res_{r}"] = v

    # aux 路：仅当存在且为 MR（即 lift_2_5d_to_3d 路径）时收集。
    if isinstance(aux_loss_fn, MultiResolutionLoss):
        diag = aux_loss_fn.pop_per_res_diag()
        if diag is not None:
            for r, v in enumerate(diag):
                if math.isfinite(v):
                    breakdown[f"L_aux_res_{r}"] = v
    # native_d 的 aux_loss_fns（list of SliceChannelLoss，无 MR）：跳过。


def format_breakdown(breakdown: Optional[Dict[str, float]]) -> str:
    """渲染 ``" | L_main=... L_aux_k=...(w=...)"``；空 breakdown 返 ``""``。"""
    if not breakdown:
        return ""
    parts: List[str] = []
    # L_main 优先，随后按 k 升序输出 L_aux_k。
    if "L_main" in breakdown:
        parts.append(f"L_main={breakdown['L_main']:.4f}")
    # 仅取真正的 view-aux 键 L_aux_{k}（排除 L_aux_res_*）。
    aux_keys = sorted(
        (k for k in breakdown
         if k.startswith("L_aux_") and not k.startswith("L_aux_res_")
         and k.split("_")[-1].isdigit()),
        key=lambda k: int(k.split("_")[-1]))
    for k in aux_keys:
        view_k = k.split("_")[-1]
        w_key = f"w_aux_{view_k}"
        if w_key in breakdown:
            parts.append(
                f"{k}={breakdown[k]:.4f}(w={breakdown[w_key]:.3g})")
        else:
            parts.append(f"{k}={breakdown[k]:.4f}")
    # 多分辨率诊断键：L_res_r / L_aux_res_r 按 r 升序输出。
    for prefix in ("L_res_", "L_aux_res_"):
        res_keys = sorted(
            (k for k in breakdown
             if k.startswith(prefix) and k[len(prefix):].isdigit()),
            key=lambda k, p=prefix: int(k[len(p):]))
        for k in res_keys:
            parts.append(f"{k}={breakdown[k]:.4f}")
    # 拓扑辅助头（中心线/距离场）损失。
    if "L_topo" in breakdown:
        if "w_topo" in breakdown:
            parts.append(
                f"L_topo={breakdown['L_topo']:.4f}"
                f"(w={breakdown['w_topo']:.3g})")
        else:
            parts.append(f"L_topo={breakdown['L_topo']:.4f}")
    return " | " + " ".join(parts)


__all__ = ["collect_multi_res_breakdown", "format_breakdown"]
