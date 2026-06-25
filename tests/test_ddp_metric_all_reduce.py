"""DDP 验证 all-reduce 数值等价性测试（CPU + gloo，多进程）。

验证 ``MetricAccumulator.all_reduce`` 的核心承诺：各 rank 在**不相交**样本子集上
累加可加混淆量、再 all-reduce(SUM) 后由 ``compute()`` 闭式导出的全部指标，与
**单进程在全集上累加**严格相等（仅浮点求和次序导致的极小误差）。这是"DDP 整卷
验证切分 + all-reduce 与单卡选模质量一致"的数值基础。

用 gloo 后端在 CPU 上 spawn 多进程，无需 GPU / 真实数据。
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from segtask_v1.trainer.validation import MetricAccumulator

NUM_FG = 2
SHAPE = (6, 10, 10)
CRITERION = "balanced"   # 触发 surface-dice 分子/分母累加路径
SD_TOL = 2
SD_W = 0.5


def _make_samples(n: int, seed: int = 1234):
    """确定性合成 n 个 (logits, target) 样本；各进程重建后按 rank 取子集。"""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n):
        logits = torch.randn(1, NUM_FG, *SHAPE, generator=g)
        target = (torch.rand(1, NUM_FG, *SHAPE, generator=g) > 0.5).float()
        samples.append((logits, target))
    return samples


def _single_process_metrics(n: int):
    acc = MetricAccumulator(CRITERION, SD_TOL, SD_W)
    for logits, target in _make_samples(n):
        acc.update(logits, target)
    return acc.compute(log_prefix="single", log=False)


def _ddp_worker(rank: int, world_size: int, n: int, port: int, out_path: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        samples = _make_samples(n)
        acc = MetricAccumulator(CRITERION, SD_TOL, SD_W)
        # 各 rank 取不相交子集（与 VolumeValEvaluator 的 npz_paths[rank::ws] 同构）。
        for logits, target in samples[rank::world_size]:
            acc.update(logits, target)
        acc.all_reduce(NUM_FG, torch.device("cpu"))
        metrics = acc.compute(log_prefix=f"rank{rank}", log=False)
        if rank == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f)
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
@pytest.mark.parametrize("world_size,n_samples", [(2, 8), (3, 7), (3, 2)])
def test_all_reduce_matches_single_process(world_size, n_samples, tmp_path):
    """N 进程切分 + all-reduce 的指标 == 单进程全集累加（含空 shard 情形）。

    (3, 2)：world_size > n_samples，rank2 分到空子集，验证零参与求和不破坏数值。
    """
    expected = _single_process_metrics(n_samples)

    out_path = str(tmp_path / "ddp_metrics.json")
    port = _free_port()
    mp.spawn(
        _ddp_worker,
        args=(world_size, n_samples, port, out_path),
        nprocs=world_size,
        join=True)

    with open(out_path, "r", encoding="utf-8") as f:
        got = json.load(f)

    assert set(got.keys()) == set(expected.keys())
    for k, exp_v in expected.items():
        g_v = got[k]
        if isinstance(exp_v, float) and (exp_v != exp_v):  # NaN（如 val_loss）
            assert g_v != g_v, f"{k}: expected NaN, got {g_v}"
            continue
        assert g_v == pytest.approx(exp_v, rel=1e-5, abs=1e-6), (
            f"metric {k!r}: ddp={g_v} vs single={exp_v}")
