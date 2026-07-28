"""监督头损失均衡器（1-6）：静态归一化权重 与 GradNorm 自适应权重。

监督头 = main + 各多 FOV aux 视图 + （可选）topo 辅助头。深监督各级仍在
main 头内部（``DeepSupervisionLoss`` 自带归一化），不拆为独立头。

* ``StaticBalancer``：固定权重线性组合；``normalize=True`` 时把全部头权重
  缩放到 Σw=1，消除"头越多总损失量纲越大"的隐患（学习率/梯度裁剪阈值
  在不同头数配置间可比）。
* ``GradNormBalancer``：GradNorm（Chen+ ICML 2018）——头权重可学习，
  按各头相对训练速率 r_i 与梯度范数 G_i 自适应再平衡：
  ``L_grad = Σ_i |w_i·g_i − Ḡ·r_i^α|``，只更新 w，随后重归一化保持 Σw 不变。
  g_i 由 ``autograd.grad(L_i, shared_params)`` 得到（detach 标量），故
  L_grad 对 w 解析可导，无需二阶图；DDP 下 g_i/L_i 先 all-reduce 均值，
  各 rank 得到相同的 w 更新，无需广播。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from taskcore.engine.dist_utils import (
    all_reduce_sum_,
    get_world_size,
    is_dist_avail_and_initialized,
)

logger = logging.getLogger(__name__)


class SupervisionBalancer:
    """基类：``combine(heads)`` 把 ``[(name, loss)]`` 线性组合为总损失。"""

    def weight(self, name: str) -> float:
        raise NotImplementedError

    def combine(
        self,
        heads: Sequence[Tuple[str, torch.Tensor]],
        breakdown: Optional[dict] = None,
    ) -> torch.Tensor:
        total = None
        for name, l in heads:
            w = self.weight(name)
            term = w * l
            total = term if total is None else total + term
            if breakdown is not None:
                breakdown[f"L_{name}"] = float(l.detach().item())
                breakdown[f"w_{name}"] = float(w)
        if total is None:
            raise ValueError("combine() requires at least one head loss")
        if breakdown is not None:
            breakdown["L_total"] = float(total.detach().item())
        return total

    # GradNorm 钩子：静态均衡器为 no-op。
    def wants_update(self) -> bool:
        return False

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class StaticBalancer(SupervisionBalancer):
    """固定权重；``normalize=True`` 时按配置头全集把权重缩放到 Σw=1。"""

    def __init__(self, weights: Dict[str, float], normalize: bool):
        if not weights:
            raise ValueError("StaticBalancer requires at least one head")
        for k, v in weights.items():
            if not (float(v) >= 0.0):
                raise ValueError(f"head weight must be >= 0; got {k}={v}")
        self.normalize = bool(normalize)
        total = sum(float(v) for v in weights.values())
        if self.normalize:
            if total <= 0:
                raise ValueError(
                    f"cannot normalize all-zero head weights: {weights}")
            self._w = {k: float(v) / total for k, v in weights.items()}
        else:
            self._w = {k: float(v) for k, v in weights.items()}

    def weight(self, name: str) -> float:
        if name not in self._w:
            raise KeyError(
                f"unknown supervision head '{name}'; configured heads: "
                f"{sorted(self._w)}")
        return self._w[name]


class GradNormBalancer(SupervisionBalancer, nn.Module):
    """GradNorm 自适应头权重。

    ``combine`` 用当前 w（detach 标量）组合任务损失，并暂存本 micro-batch 的
    逐头 loss 张量（含图）；trainer 在 backward **之前**、按
    ``update_every`` 个优化步的节奏调用 ``update(shared_params, amp_scale)``：

    1. ``g_i = ||autograd.grad(L_i·s, shared)||/s``（retain_graph，主 backward 不受影响）
    2. DDP：g_i、L_i all-reduce 求均值（各 rank 同步更新）
    3. 首次更新记 ``L_i(0)``；``r_i = (L_i/L_i(0)) / mean_j(L_j/L_j(0))``
    4. ``L_grad(w) = Σ|w_i·g_i − Ḡ·r_i^α|``（Ḡ、target 均 detach）→ Adam 更新 w
    5. w 夹正并重归一化保持 Σw = 初始 Σw
    """

    def __init__(
        self,
        weights: Dict[str, float],
        alpha: float = 1.5,
        lr: float = 0.025,
        update_every: int = 25,
        normalize: bool = True,
    ):
        nn.Module.__init__(self)
        if len(weights) < 2:
            raise ValueError(
                "GradNorm needs >= 2 supervision heads; got "
                f"{sorted(weights)}. Enable aux/topo heads or disable "
                "loss.gradnorm_enabled.")
        self.names: List[str] = list(weights.keys())
        init = torch.tensor([float(weights[n]) for n in self.names],
                            dtype=torch.float64)
        if bool(normalize):
            if float(init.sum()) <= 0:
                raise ValueError(
                    f"cannot normalize all-zero head weights: {weights}")
            init = init / init.sum()
        if (init <= 0).any():
            raise ValueError(
                "GradNorm initial head weights must be > 0 (a zero weight "
                f"kills its gradient signal permanently); got {weights}.")
        self.w = nn.Parameter(init.clone())
        self._w_sum0 = float(init.sum())
        self.alpha = float(alpha)
        self.update_every = int(update_every)
        self._opt = torch.optim.Adam([self.w], lr=float(lr))
        self._l0: Optional[torch.Tensor] = None  # (n,) float64
        self._boundary_clock = 0
        # 仅在 trainer 先 arm_stash() 的 micro-batch 才暂存逐头 loss 张量（含图）；
        # 无条件暂存会把上一步的激活图多留一整步，白占显存。
        self._stash: Optional[List[Tuple[str, torch.Tensor]]] = None
        self._stash_armed = False

    def weight(self, name: str) -> float:
        try:
            i = self.names.index(name)
        except ValueError:
            raise KeyError(
                f"unknown supervision head '{name}'; configured heads: "
                f"{self.names}") from None
        return float(self.w.detach()[i])

    def combine(self, heads, breakdown=None):
        present = [n for n, _ in heads]
        if present != self.names:
            raise RuntimeError(
                f"GradNorm head set mismatch: expected {self.names}, got "
                f"{present}. Head structure must be identical every step.")
        if self._stash_armed:
            self._stash = list(heads)
            self._stash_armed = False
        return super().combine(heads, breakdown)

    def tick_boundary(self) -> bool:
        """优化步边界打点；返回本步是否应执行 GradNorm 更新。"""
        due = (self._boundary_clock % self.update_every) == 0
        self._boundary_clock += 1
        return due

    def arm_stash(self) -> None:
        """声明下一次 combine() 需暂存逐头 loss（随后 update() 消费）。"""
        self._stash_armed = True

    def wants_update(self) -> bool:  # 供 trainer 判断是否走 GradNorm 路径
        return True

    @staticmethod
    def _mean_across_ranks(vec: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not is_dist_avail_and_initialized():
            return vec
        t = vec.to(device=device, dtype=torch.float32)
        all_reduce_sum_(t)
        return (t / get_world_size()).to(dtype=vec.dtype, device=vec.device)

    def update(
        self,
        shared_params: List[torch.Tensor],
        device: torch.device,
        amp_scale: float = 1.0,
    ) -> None:
        """一次 GradNorm 权重更新。须在主 backward 之前调用（图未释放）。"""
        heads = self._stash
        self._stash = None
        if heads is None:
            raise RuntimeError(
                "GradNormBalancer.update called without a preceding "
                "combine() in the same micro-batch.")
        losses = [l for _, l in heads]
        if not all(torch.isfinite(l.detach()) for l in losses):
            logger.warning("GradNorm: non-finite head loss; skipping update.")
            return
        s = float(amp_scale) if amp_scale and amp_scale > 0 else 1.0
        g = []
        for l in losses:
            grads = torch.autograd.grad(
                l * s, shared_params, retain_graph=True, allow_unused=True)
            sq = None
            for gr in grads:
                if gr is None:
                    continue
                v = gr.detach().float().pow(2).sum()
                sq = v if sq is None else sq + v
            g.append(0.0 if sq is None else float(sq.sqrt()) / s)
        g_vec = torch.tensor(g, dtype=torch.float64)
        l_vec = torch.tensor([float(l.detach()) for l in losses],
                             dtype=torch.float64)
        g_vec = self._mean_across_ranks(g_vec, device)
        l_vec = self._mean_across_ranks(l_vec, device)
        if not (torch.isfinite(g_vec).all() and torch.isfinite(l_vec).all()):
            logger.warning(
                "GradNorm: non-finite reduced stats; skipping update.")
            return
        if self._l0 is None:
            self._l0 = l_vec.clamp(min=1e-8).clone()
        ratio = (l_vec / self._l0).clamp(min=1e-8)
        r = ratio / ratio.mean()
        G = self.w * g_vec  # 对 w 解析可导（g_vec 常数）
        target = (G.detach().mean() * r.pow(self.alpha)).detach()
        l_grad = (G - target).abs().sum()
        self._opt.zero_grad(set_to_none=True)
        l_grad.backward()
        self._opt.step()
        with torch.no_grad():
            self.w.clamp_(min=1e-4 * self._w_sum0 / self.w.numel())
            self.w.mul_(self._w_sum0 / self.w.sum())

    def state_dict(self) -> dict:  # type: ignore[override]
        return {
            "names": list(self.names),
            "w": self.w.detach().clone(),
            "l0": None if self._l0 is None else self._l0.clone(),
            "boundary_clock": self._boundary_clock,
            "opt": self._opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:  # type: ignore[override]
        if list(state.get("names", [])) != self.names:
            logger.warning(
                "GradNorm resume: head set changed (%s -> %s); starting "
                "weights fresh.", state.get("names"), self.names)
            return
        with torch.no_grad():
            self.w.copy_(state["w"].to(self.w.dtype))
        l0 = state.get("l0")
        self._l0 = None if l0 is None else l0.clone().to(torch.float64)
        self._boundary_clock = int(state.get("boundary_clock", 0))
        try:
            self._opt.load_state_dict(state["opt"])
        except (KeyError, ValueError) as e:
            logger.warning("GradNorm resume: optimizer state skipped (%s).", e)


def build_balancer(
    cfg,
    aux_weights: Sequence[float],
    topo_weight: Optional[float],
) -> SupervisionBalancer:
    """由配置组装均衡器。头全集：main(=1.0) + aux_k + （启用时）topo。"""
    weights: Dict[str, float] = {"main": 1.0}
    for k, w in enumerate(aux_weights):
        weights[f"aux_{k + 1}"] = float(w)
    if topo_weight is not None:
        weights["topo"] = float(topo_weight)
    lc = cfg.loss
    if bool(lc.gradnorm_enabled):
        bal = GradNormBalancer(
            weights,
            alpha=float(lc.gradnorm_alpha),
            lr=float(lc.gradnorm_lr),
            update_every=int(lc.gradnorm_update_every),
            normalize=bool(lc.normalize_supervision_weights))
        logger.info(
            "Loss balancer: GradNorm (heads=%s, init_w=%s, alpha=%.2f, "
            "lr=%.4g, update_every=%d)", bal.names,
            [round(float(x), 4) for x in bal.w.detach().tolist()],
            bal.alpha, lc.gradnorm_lr, bal.update_every)
        return bal
    bal = StaticBalancer(
        weights, normalize=bool(lc.normalize_supervision_weights))
    logger.info(
        "Loss balancer: static (normalize=%s, weights=%s)",
        bal.normalize, {k: round(v, 4) for k, v in bal._w.items()})
    return bal


__all__ = ["SupervisionBalancer", "StaticBalancer", "GradNormBalancer",
           "build_balancer"]
