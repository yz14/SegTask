"""Optimizer / Scheduler 工厂与 ``WarmupScheduler`` 包装。

从 ``Trainer`` 中拆出，纯函数 + 单独的 wrapper 类，与训练循环无耦合。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taskcore.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------
def _param_groups(model: nn.Module, weight_decay: float) -> list:
    """参数分组：ndim<=1（norm affine/bias 等向量参数）免 weight decay，
    ndim>=2（conv/linear 权重）正常衰减。对向量参数做衰减会把归一化尺度/
    偏置无理由地拉向 0，是 AdamW 惯例上应避免的。"""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _zero_redundancy_enabled(cfg: Config) -> bool:
    """ZeRO-1 分片是否实际可用：要求显式开启且处于多进程 DDP 环境。"""
    if not cfg.train.zero_redundancy_optimizer:
        return False
    import torch.distributed as dist
    if (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1):
        return True
    logger.warning(
        "train.zero_redundancy_optimizer=True 但当前非多卡 DDP 环境；"
        "无分片收益，回退普通优化器。")
    return False


def _build_zero_optimizer(groups: list, optim_cls, **defaults):
    """用 ZeroRedundancyOptimizer 包装 ``optim_cls``，保留参数分组语义。

    ZeRO 构造器只接受单个参数列表，其余分组经 ``add_param_group`` 追加
    （各 rank 调用顺序一致，分片确定性由 ZeRO 保证）。优化器状态均分到
    world_size 张卡，每卡省 state_bytes×(1−1/N)；step 后各 rank broadcast
    自己分片的参数，数值与普通 DDP+同优化器严格等价。
    """
    from torch.distributed.optim import ZeroRedundancyOptimizer

    non_empty = [g for g in groups if g["params"]]
    if not non_empty:
        raise ValueError("No trainable parameters for ZeroRedundancyOptimizer.")
    first, *rest = non_empty
    first_kwargs = {k: v for k, v in first.items() if k != "params"}
    opt = ZeroRedundancyOptimizer(
        first["params"], optimizer_class=optim_cls,
        **{**defaults, **first_kwargs})
    for g in rest:
        opt.add_param_group(dict(g))
    logger.info(
        "ZeroRedundancyOptimizer enabled: %s state sharded across ranks.",
        optim_cls.__name__)
    return opt


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    tc = cfg.train
    groups = _param_groups(model, tc.weight_decay)
    use_zero = _zero_redundancy_enabled(cfg)
    if   tc.optimizer == "adamw":
        first = next((p for p in model.parameters()), None)
        on_cuda = first is not None and first.is_cuda
        use_fused = tc.adamw_fused and torch.cuda.is_available()
        kwargs = {"lr": tc.lr, "fused": (use_fused and on_cuda)}
        if use_zero:
            return _build_zero_optimizer(groups, torch.optim.AdamW, **kwargs)
        return torch.optim.AdamW(groups, **kwargs)
    elif tc.optimizer == "adam":
        if use_zero:
            return _build_zero_optimizer(groups, torch.optim.Adam, lr=tc.lr)
        return torch.optim.Adam(groups, lr=tc.lr)
    elif tc.optimizer == "sgd":
        kwargs = {"lr": tc.lr, "momentum": tc.momentum,
                  "nesterov": tc.nesterov}
        if use_zero:
            return _build_zero_optimizer(groups, torch.optim.SGD, **kwargs)
        return torch.optim.SGD(groups, **kwargs)
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    steps_per_epoch: int,
    post_warmup_steps: int,
):
    """构造 warmup 之后的 base scheduler；horizon 按 ``post_warmup_steps`` 对齐。"""
    tc = cfg.train
    horizon = max(post_warmup_steps, 1)

    if tc.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(1 - step / horizon, 0.0) ** tc.poly_power)
    elif tc.scheduler == "step":
        milestones = list(range(
            tc.step_size * steps_per_epoch, horizon,
            tc.step_size * steps_per_epoch))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=tc.step_gamma)
    elif tc.scheduler == "plateau":
        # mode 跟随 save_best_mode。
        plateau_mode = tc.save_best_mode if tc.save_best_mode in ("max", "min") else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=plateau_mode, patience=tc.plateau_patience,
            factor=tc.plateau_factor)
    elif tc.scheduler == "cosine_warm_restarts":
        T_0 = tc.cosine_restart_period * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(T_0, 1), T_mult=tc.cosine_restart_mult,
            eta_min=tc.cosine_min_lr)
    elif tc.scheduler == "one_cycle":
        # warmup_epochs 映射为 OneCycleLR 的 pct_start；外层 warmup 由 Trainer 关掉。
        total_steps = tc.epochs * steps_per_epoch
        # pct_start 直接按 warmup_epochs/epochs 配比；下限保证 warmup 段
        # 至少 1 个 step（OneCycleLR 要求两段均非空），上限留出退火段。
        pct_start = tc.warmup_epochs / max(tc.epochs, 1)
        pct_start = min(max(pct_start, 2.0 / max(total_steps, 4)), 0.9)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tc.lr, total_steps=total_steps,
            pct_start=pct_start)
    raise ValueError(f"Unknown scheduler: {tc.scheduler}")


# ---------------------------------------------------------------------------
# Warmup wrapper
# ---------------------------------------------------------------------------
class WarmupScheduler:
    """线性 warmup → base scheduler。Plateau 逐 epoch step，其余逐 step。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        warmup_steps: int,
        warmup_lr: float,
        base_lr: float,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_step = 0
        self._is_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        if warmup_steps > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            alpha = self.current_step / max(self.warmup_steps, 1)
            lr = self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        elif self.scheduler is not None and not self._is_plateau:
            self.scheduler.step()

    def step_epoch(self, metric: Optional[float] = None) -> None:
        if (self._is_plateau
                and self.scheduler is not None
                and self.current_step > self.warmup_steps
                and metric is not None):
            self.scheduler.step(metric)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict:
        # 一并存 warmup 参数，便于 load 时检出配置漂移。
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "warmup_lr": self.warmup_lr,
            "base_lr": self.base_lr,
            "base_scheduler": (self.scheduler.state_dict()
                               if self.scheduler is not None else None),
        }

    def load_state_dict(self, state: Dict) -> None:
        ckpt_warmup_steps = state.get("warmup_steps")
        ckpt_warmup_lr = state.get("warmup_lr")
        ckpt_base_lr = state.get("base_lr")
        # warmup 参数漂移会改变 schedule 形状；不致命但告警。
        mismatches = []
        if (ckpt_warmup_steps is not None
                and int(ckpt_warmup_steps) != int(self.warmup_steps)):
            mismatches.append(
                f"warmup_steps: ckpt={ckpt_warmup_steps} vs cfg={self.warmup_steps}")
        if ckpt_warmup_lr is not None and float(ckpt_warmup_lr) != float(self.warmup_lr):
            mismatches.append(
                f"warmup_lr: ckpt={ckpt_warmup_lr} vs cfg={self.warmup_lr}")
        if ckpt_base_lr is not None and float(ckpt_base_lr) != float(self.base_lr):
            mismatches.append(
                f"base_lr: ckpt={ckpt_base_lr} vs cfg={self.base_lr}")
        if mismatches:
            logger.warning(
                "Warmup config drift on resume (%s); current_step restored "
                "but schedule shape differs.", "; ".join(mismatches))

        self.current_step = int(state.get("current_step", 0))
        base_state = state.get("base_scheduler", None)
        if base_state is not None and self.scheduler is not None:
            base_state = self._reconcile_one_cycle_horizon(base_state)
            self.scheduler.load_state_dict(base_state)

    def _reconcile_one_cycle_horizon(self, base_state: Dict) -> Dict:
        """OneCycleLR 的 total_steps 在构建时定死；resume 时若 epochs/累积/数据量
        变化导致 horizon 漂移，直接恢复旧状态会在超出旧 total_steps 时抛
        "Tried to step ... times"。这里检出漂移：保留新构建的 horizon/相位边界，
        仅把已走步数按比例折算进新 horizon 并告警。无漂移时原样返回。"""
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            return base_state
        ckpt_total = base_state.get("total_steps")
        cur_total = int(self.scheduler.total_steps)
        if ckpt_total is None or int(ckpt_total) == cur_total:
            return base_state
        old_last = int(base_state.get("last_epoch", 0))
        new_last = min(
            int(round(old_last / max(int(ckpt_total), 1) * cur_total)),
            cur_total - 1)
        logger.warning(
            "OneCycleLR horizon drift on resume: ckpt total_steps=%s vs "
            "current=%d (epochs/grad_accum/dataset size changed?). Keeping "
            "the freshly built schedule and fast-forwarding last_epoch "
            "%d -> %d proportionally.",
            ckpt_total, cur_total, old_last, new_last)
        new_state = self.scheduler.state_dict()
        new_state["last_epoch"] = new_last
        new_state["_step_count"] = new_last + 1
        return new_state


__all__ = ["build_optimizer", "build_scheduler", "WarmupScheduler"]
