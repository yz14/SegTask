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
import torch.nn as nn
import torch.nn.functional as F

#: 重建/回归损失函数表（与 ``SSLConfig.recon_loss`` 取值对应）。
RECON_LOSS_FNS = {
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
    "mse": F.mse_loss,
}


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
        以避免汇总误差。``logs`` 为标量字典，仅用于日志/监控。
        """

    @abstractmethod
    def export_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """返回与 ``build_model`` 同名的 CPU state_dict，供下游 ``train.pretrain``。"""

    # ---- 可选 hook ---------------------------------------------------------
    def configure_schedule(self, total_steps: int) -> None:
        """训练开始前由 ``SSLTrainer`` 调用一次，告知总优化步数（= epochs ×
        steps_per_epoch ÷ grad_accum）。供需要全程进度的方法预计算 cosine 调度
        （如自蒸馏的 EMA 教师动量、teacher 温度 warmup）。默认 no-op。"""

    def on_after_step(self, global_step: int) -> None:
        """每个优化步边界后调用（EMA 教师更新 / 温度·动量调度）。默认 no-op。"""

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


__all__ = ["SSLMethod", "RECON_LOSS_FNS"]
