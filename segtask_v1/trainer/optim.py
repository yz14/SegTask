"""Optimizer / Scheduler 工厂与 ``WarmupScheduler`` 包装。

从 ``Trainer`` 中拆出，纯函数 + 单独的 wrapper 类，与训练循环无耦合。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import Config

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


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    tc = cfg.train
    groups = _param_groups(model, tc.weight_decay)
    if   tc.optimizer == "adamw":
        first = next((p for p in model.parameters()), None)
        on_cuda = first is not None and first.is_cuda
        use_fused = tc.adamw_fused and torch.cuda.is_available()
        return torch.optim.AdamW(
            groups, lr=tc.lr, fused=(use_fused and on_cuda))
    elif tc.optimizer == "adam":
        return torch.optim.Adam(groups, lr=tc.lr)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(
            groups, lr=tc.lr,
            momentum=tc.momentum, nesterov=tc.nesterov)
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
            self.scheduler.load_state_dict(base_state)


__all__ = ["build_optimizer", "build_scheduler", "WarmupScheduler"]
