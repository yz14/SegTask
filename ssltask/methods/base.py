"""SSL 方法统一接口。

每个自监督方法把"如何在共享骨干上构造模块 + 如何由一个 image batch 算出损失 +
如何导出可迁移骨干权重"封装为一个 :class:`SSLMethod` 子类，从而让训练循环
（``ssltask.trainer.SSLTrainer``）对方法**完全无感**：循环只负责 optimizer /
scheduler / AMP / EMA / ckpt / 日志，方法负责目标本身。新增方法 = 新增一个子类并
在 ``ssltask.methods`` 注册表登记，零改训练循环。

约定的不变量（保证 SSL→下游交接）：``export_backbone_state_dict`` 返回的 state_dict
的键必须与 ``segtask_v1.models.factory.build_model`` 逐参数同名（``encoder.*`` /
``decoder.*`` / 方法附加头），下游 ``train.pretrain``（strict=False）即可命中 enc(+dec)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from taskcore.engine.dist_utils import (
    get_rank, get_world_size, is_dist_avail_and_initialized)

#: 重建/回归损失函数表（与 ``SSLConfig.recon_loss`` 取值对应）。
RECON_LOSS_FNS = {
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
    "mse": F.mse_loss,
}


class _GatherCatLayer(torch.autograd.Function):
    """带梯度的 all-gather + 沿 batch 维 cat（VICReg 官方 ``FullGatherLayer`` 语义）。

    前向：收集各 rank 的 ``x``（逐 rank 等形，由 DistributedSampler drop_last
    保证）拼成全局 batch；反向：all-reduce 全局梯度后取回本 rank 切片，使跨 rank
    统计量（variance/covariance）的梯度正确回流到本地样本。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        world = get_world_size()
        outs = [torch.zeros_like(x) for _ in range(world)]
        dist.all_gather(outs, x.contiguous())
        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        n = grad.shape[0] // get_world_size()
        r = get_rank()
        return grad[r * n:(r + 1) * n]


def gather_cat_with_grad(x: torch.Tensor) -> torch.Tensor:
    """DDP 下把本 rank 嵌入 ``x: (N, D)`` 拼成全局 ``(world·N, D)``（保留梯度）；
    非分布式时原样返回。供 VICReg 族的 variance/covariance 在**全局** batch 上
    计算（官方 VICReg 做法），避免小 per-GPU batch 下正则统计噪声大/偷弱。要求
    各 rank 的 ``x`` 等形。"""
    if not (is_dist_avail_and_initialized() and get_world_size() > 1):
        return x
    return _GatherCatLayer.apply(x)


class SSLMethod(ABC):
    """自监督方法基类。

    生命周期：``__init__`` 内调用 :meth:`build_modules` 构造可训练 ``nn.Module``
    （含共享骨干 + 方法附加模块）并搬到 ``device``，存于 ``self.module``。训练循环对
    ``self.module`` 做优化 / AMP / EMA / 保存。
    """

    #: 注册键（与 ``SSLConfig.method`` / ``ssltask.config.METHODS`` 对齐）。
    name: str = ""

    #: 是否接受 trainer 级通用增强（segtask ``GPUAugmentor``，由 ``cfg.augment``
    #: 控制）。重建类方法（破坏/掩码→重建，输入与目标同源）置 True——增强后的
    #: 图即新的自洽样本；视图类方法（dino/byol/moco/jepa）自带多视图增广管道，
    #: 置 False 以免与其视图变换叠加改变方法语义。
    trainer_augment: bool = False

    def __init__(self, cfg, ssl, device: torch.device):
        self.cfg = cfg
        self.ssl = ssl
        self.device = device
        self.module: nn.Module = self.build_modules().to(device)

    # ---- 子类必须实现 ------------------------------------------------------
    @abstractmethod
    def build_modules(self) -> nn.Module:
        """构造并返回可训练 ``nn.Module``（训练循环将优化其 ``parameters()``）。"""

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """由一个 batch（``{'image': (B,C,*spatial)}`` 等）算出 ``(loss, logs)``。

        本方法在训练循环的 AMP autocast 上下文内被调用；损失内部应在 fp32 计算
        以避免汇总误差。``logs`` 为标量字典，仅用于日志/监控；其值可为 Python
        ``float``，**也可为未同步的 0-dim ``torch.Tensor``**（device 标量）——
        ``SSLTrainer`` 会在日志/累积边界处对本组所有 device 标量批量 ``.tolist()``
        一次性取回，从而免去每个 micro-step 的 host/device 同步（首选后者）。
        """

    @abstractmethod
    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """返回与 ``build_model`` 同名的 CPU state_dict，供下游 ``train.pretrain``。"""

    # ---- 可选 hook ---------------------------------------------------------
    def configure_schedule(self, total_steps: int) -> None:
        """训练开始前由 ``SSLTrainer`` 调用一次，告知总优化步数（= epochs ×
        steps_per_epoch ÷ grad_accum）。供需要全程进度的方法预计算 cosine 调度
        （如自蒸馏的 EMA 教师动量、teacher 温度 warmup）。默认 no-op。"""

    def on_before_optimizer_step(self) -> None:
        """每次 ``optimizer.step`` 前调用（梯度已就绪、可能已 unscale/clip）。
        供方法取消特定参数的梯度（如 DINO 冻结投影头末层的稳定化期）。默认 no-op。"""

    def on_after_step(self, global_step: int, stepped: bool = True) -> None:
        """每个优化步边界后调用（EMA 教师更新 / 温度·动量调度）。默认 no-op。

        ``stepped=False`` 表示本边界的优化步被跳过（非有限 loss 丢弃梯度、或
        fp16 GradScaler 因 inf/NaN 梯度内部跳步）：子类应照常推进调度计数
        （与 scheduler 时钟对齐），但**不得**施加 EMA / center / queue 等状态
        更新，并丢弃本 accum 组内缓存的待处理状态。"""

    def on_resume(self, global_step: int) -> None:
        """resume 加载后由 ``SSLTrainer`` 调用一次，恢复步进度（温度/动量调度的
        当前步）。仅恢复计数，不应产生 EMA 更新等副作用。默认 no-op。"""

    # ---- 便捷透传 ----------------------------------------------------------
    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def parameters(self):
        return self.module.parameters()


__all__ = ["SSLMethod", "RECON_LOSS_FNS", "gather_cat_with_grad"]
