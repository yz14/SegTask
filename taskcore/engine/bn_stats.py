"""BatchNorm running-stats 重估公共件（AdaBN / SWA BN 重校准共用）。

两类使用场景：推理期 AdaBN（目标域重估，见 segtask_v1.predictor.adabn）与 SWA
收尾的 BN 重校准（平均权重下激活分布改变，见 BaseTrainer._finalize_swa）。

动机：本框架不做物理 spacing 重采样，跨数据集推理时 z 层厚 / 背景填充值等域漂移
会直接改变 BN 输入的特征分布，而 BN 的 ``running_mean/var`` 是在源域（训练集）上
统计的。AdaBN 在推理前/推理时用目标域前向（无标签、不回传、不重训）重估这些统计，
对 BN 域敏感的假阳常有立竿见影的抑制效果。

两种使用方式（由调用方组织）：

* **global**：用少量目标域整卷预热一次 BN 统计，全程复用（见 ``run_inference``）。
* **per_volume**（transductive BN）：每卷推理前用该卷自身 patch 重估，再冻结预测
  （见 ``Predictor.predict_volume``）。

实现要点：仅把 BN 子模块切到 ``train()`` 并令 ``momentum=None``（累积平均，配合
``reset_running_stats`` 得到纯目标域统计），模型其余部分保持 ``eval()``；估计完成后
恢复 BN 的 training/momentum，running stats 即保留新值供后续 eval 前向使用。BN 的
running-stats 更新不走 autograd，故全程可在 ``torch.no_grad()`` 下进行。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Iterator, List

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

logger = logging.getLogger(__name__)


def collect_bn_modules(model: nn.Module) -> List[_BatchNorm]:
    """收集模型中所有“带可更新 running stats”的 BatchNorm 子模块。

    仅纳入 ``track_running_stats=True`` 且 buffer 非 ``None`` 的层（InstanceNorm /
    GroupNorm 不在此列，对它们 AdaBN 无意义）。
    """
    mods = [
        m for m in model.modules()
        if isinstance(m, _BatchNorm)
        and m.track_running_stats
        and m.running_mean is not None
    ]
    return mods


def reset_running_stats(bn_modules: List[_BatchNorm]) -> None:
    """将各 BN 的 ``running_mean→0 / running_var→1 / num_batches_tracked→0``。

    配合 ``momentum=None`` 的累积平均，使后续前向累计出的 running stats 等于目标域
    所有 batch 的真实均值/方差（不掺入源域先验）。
    """
    for m in bn_modules:
        m.reset_running_stats()


@contextmanager
def bn_estimation_mode(bn_modules: List[_BatchNorm]) -> Iterator[None]:
    """临时把 BN 切到统计累积模式：``train()`` + ``momentum=None``（累积平均）。

    退出时恢复每个 BN 原有的 ``training`` 标志与 ``momentum``；running stats 保留为
    累积所得的新值。模型整体的 train/eval 状态不受影响（仅触及 BN 子模块）。
    """
    prev_training = [m.training for m in bn_modules]
    prev_momentum = [m.momentum for m in bn_modules]
    for m in bn_modules:
        m.train()
        m.momentum = None  # 累积移动平均：avg_factor = 1/num_batches_tracked
    try:
        yield
    finally:
        for m, training, momentum in zip(
                bn_modules, prev_training, prev_momentum):
            m.train(training)
            m.momentum = momentum


@torch.no_grad()
def estimate_bn_stats(
    bn_modules: List[_BatchNorm],
    run_forward: Callable[[], None],
) -> None:
    """重估 BN running stats：先 ``reset`` 再在累积模式下跑 ``run_forward``。

    ``run_forward`` 由调用方提供，应在目标域数据上触发若干次模型前向（输入布局须与
    真实推理一致，以保证 BN 看到的分布同构）。函数返回后 BN 已恢复 eval、running
    stats 持有目标域估计值。
    """
    if not bn_modules:
        logger.warning(
            "[AdaBN] estimate_bn_stats called with no BatchNorm modules — "
            "no-op (model likely uses instance/group norm).")
        return
    reset_running_stats(bn_modules)
    with bn_estimation_mode(bn_modules):
        run_forward()


__all__ = [
    "collect_bn_modules",
    "reset_running_stats",
    "bn_estimation_mode",
    "estimate_bn_stats",
]
