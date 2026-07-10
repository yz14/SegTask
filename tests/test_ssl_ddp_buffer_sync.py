"""ssltask 手动 DDP 下 buffer 同步的多进程测试（CPU + gloo）。

手动 DDP（SSLTrainer 梯度 all-reduce + 初始一次性 buffer 广播）不同步训练中
更新的 buffer；本文件验证方法侧的补偿逻辑：

* MoCo：入队前 ``_concat_all_gather`` 跨 rank 收集 key → 各 rank 的
  queue/queue_ptr 保持逐步一致，且负样本扩充到全局 batch；
* DINO/iBOT：``_global_batch_mean`` 把本地 batch 均值归约为全局均值 →
  center buffer 在各副本间保持一致。

用 gloo 后端在 CPU 上 spawn 多进程，无需 GPU / 真实数据。
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from ssltask.methods.dino import _global_batch_mean
from ssltask.methods.moco import MoCoMethod, _concat_all_gather

PROJ_DIM = 8
QUEUE_SIZE = 32


# ---- 非分布式：两个 helper 均为恒等 ------------------------------------------
def test_concat_all_gather_identity_when_not_distributed():
    t = torch.randn(4, PROJ_DIM)
    assert _concat_all_gather(t) is t


def test_global_batch_mean_identity_when_not_distributed():
    c = torch.randn(1, 16)
    assert _global_batch_mean(c) is c


# ---- 多进程：gather / 全局均值 / MoCo 队列一致性 -----------------------------
class _NS:
    """轻量属性容器（模拟 method.module 上的 queue buffer）。"""


def _rank_keys(rank: int, n: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + rank)
    return F.normalize(torch.randn(n, PROJ_DIM, generator=g), dim=-1)


def _ddp_worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # 1) _concat_all_gather：rank 序拼接、各 rank 结果相同。
        gathered = _concat_all_gather(_rank_keys(rank))
        expected = torch.cat([_rank_keys(r) for r in range(world_size)], dim=0)
        assert torch.allclose(gathered, expected, atol=1e-6), (
            f"rank{rank}: gather mismatch")

        # 2) _global_batch_mean：== 全 rank 局部均值的均值，各 rank 一致。
        local_mean = torch.full((1, 3), float(rank + 1))
        global_mean = _global_batch_mean(local_mean.clone())
        exp_mean = torch.full(
            (1, 3), sum(range(1, world_size + 1)) / world_size)
        assert torch.allclose(global_mean, exp_mean, atol=1e-6), (
            f"rank{rank}: global mean mismatch")

        # 3) MoCo 入队后 queue/queue_ptr 跨 rank 逐元素一致。
        holder = _NS()
        holder.module = _NS()
        g = torch.Generator().manual_seed(7)          # 各 rank 相同的初始队列
        holder.module.queue = F.normalize(
            torch.randn(PROJ_DIM, QUEUE_SIZE, generator=g), dim=0)
        holder.module.queue_ptr = torch.zeros(1, dtype=torch.long)
        for step in range(3):
            MoCoMethod._dequeue_and_enqueue(holder, _rank_keys(rank) + step)
        qs = [torch.empty_like(holder.module.queue) for _ in range(world_size)]
        dist.all_gather(qs, holder.module.queue)
        ptrs = [torch.empty_like(holder.module.queue_ptr)
                for _ in range(world_size)]
        dist.all_gather(ptrs, holder.module.queue_ptr)
        for r in range(world_size):
            assert torch.allclose(qs[r], qs[0], atol=1e-6), (
                f"rank{rank}: queue diverged on rank {r}")
            assert bool(torch.equal(ptrs[r], ptrs[0])), (
                f"rank{rank}: queue_ptr diverged on rank {r}")
        # 入队条数 = world_size × 本地 key 数 × 步数（全局 batch 入队）。
        assert int(ptrs[0].item()) == (world_size * 4 * 3) % QUEUE_SIZE
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch.distributed gloo backend unavailable")
@pytest.mark.parametrize("world_size", [2, 3])
def test_ssl_buffers_stay_in_sync_across_ranks(world_size):
    mp.spawn(
        _ddp_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True)
