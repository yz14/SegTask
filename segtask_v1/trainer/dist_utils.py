"""分布式（DDP）小工具：rank / world_size 查询、main-process 判定、可加张量
all-reduce(SUM)。

设计目标：**单卡 / 非分布式路径零行为变化**。所有查询在未初始化 process group
时退化为 ``rank=0, world_size=1, is_main=True``，``all_reduce_sum_`` 直接返回。
因此 ``Trainer`` / 验证器只需无条件调用这些 helper，无需到处写 ``if dist:`` 分支。
"""

from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist

__all__ = [
    "is_dist_avail_and_initialized",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "all_reduce_sum_",
    "all_reduce_flag_any",
    "all_reduce_bn_running_stats_",
    "shard_for_rank",
]


def is_dist_avail_and_initialized() -> bool:
    """torch.distributed 已编译可用且 process group 已初始化。"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """当前进程的全局 rank；非分布式时为 0。"""
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """参与进程数；非分布式时为 1。"""
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """是否为 rank0（落盘 / 日志 / 渲染等副作用只应在此进程发生）。"""
    return get_rank() == 0


def barrier() -> None:
    """同步屏障；非分布式时 no-op。"""
    if is_dist_avail_and_initialized() and get_world_size() > 1:
        dist.barrier()


@torch.no_grad()
def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """对 ``tensor`` 做就地 all-reduce(SUM) 并返回它；非分布式时原样返回。

    仅适用于"跨样本可加"的量（混淆量分子分母、计数等）：各 rank 处理不相交的
    样本子集后逐元素求和，与单进程在全集上累加在数学上严格相等。
    """
    if is_dist_avail_and_initialized() and get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@torch.no_grad()
def all_reduce_flag_any(flag: bool, device: torch.device) -> bool:
    """跨 rank 对布尔标志做 any 归并（all-reduce MAX）；非分布式时原样返回。

    用于必须全 rank 一致的控制流决策（如跳过优化步）：任一 rank 为 True
    则全体为 True，避免各 rank 依据本地信息做出不同决策而破坏 DDP 副本一致性。
    注意：分布式下是集体通信，所有 rank 必须在相同步调对齐调用。
    """
    if not (is_dist_avail_and_initialized() and get_world_size() > 1):
        return flag
    t = torch.tensor(1.0 if flag else 0.0, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


@torch.no_grad()
def all_reduce_bn_running_stats_(bn_modules: List) -> None:
    """跨 rank 聚合 BatchNorm running stats（就地）；非分布式时 no-op。

    面向 ``estimate_bn_stats`` 的累积平均语义（momentum=None：running_mean/var
    为各 batch 统计的等权均值）：以各 rank 的 ``num_batches_tracked`` 为权重做
    加权平均，与单进程在所有 rank 的 batch 全集上累积**严格相等**（等 batch
    大小下）。``num_batches_tracked`` 归并为全局总和。各 rank 数据 shard 不同
    时，未聚合的 stats 只代表本 rank shard —— 这正是本函数要修复的偏差。
    """
    if not (is_dist_avail_and_initialized() and get_world_size() > 1):
        return
    for m in bn_modules:
        n = m.num_batches_tracked
        n_f = n.to(torch.float64)
        mean_w = m.running_mean.to(torch.float64) * n_f
        var_w = m.running_var.to(torch.float64) * n_f
        n_total = n_f.clone()
        all_reduce_sum_(mean_w)
        all_reduce_sum_(var_w)
        all_reduce_sum_(n_total)
        if float(n_total.item()) > 0:
            m.running_mean.copy_(
                (mean_w / n_total).to(m.running_mean.dtype))
            m.running_var.copy_(
                (var_w / n_total).to(m.running_var.dtype))
        all_reduce_sum_(n)


def shard_for_rank(items: List) -> List:
    """把一个可索引序列按 ``items[rank::world_size]`` 切给当前 rank。

    用于把验证整卷列表不相交地分到各进程；非分布式时返回整列。
    """
    ws = get_world_size()
    if ws <= 1:
        return list(items)
    r = get_rank()
    return list(items)[r::ws]
