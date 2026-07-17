"""Resume / checkpoint 辅助函数回归：RNG roundtrip、optimizer 设备对齐、async CPU 快照。"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from taskcore.engine.checkpoint import (
    pack_rng_state_for_save,
    relocate_optimizer_state,
    restore_rng_state,
    state_to_cpu,
)


def _sample_rng_triplet():
    return (
        torch.rand(4).tolist(),
        np.random.rand(4).tolist(),
        [random.random() for _ in range(4)],
    )


def test_rng_pack_restore_roundtrip():
    torch.manual_seed(11)
    np.random.seed(12)
    random.seed(13)

    rng = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    packed = pack_rng_state_for_save(rng)
    assert isinstance(packed["torch_cpu"], (bytes, bytearray))
    ref = _sample_rng_triplet()

    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)

    restore_rng_state(packed)
    assert _sample_rng_triplet() == ref


def test_rng_state_to_cpu_async_path_roundtrip():
    torch.manual_seed(21)
    np.random.seed(22)
    random.seed(23)

    rng = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    cpu_snapshot = state_to_cpu({"rng_state": rng})["rng_state"]
    assert isinstance(cpu_snapshot["torch_cpu"], (bytes, bytearray))
    ref = _sample_rng_triplet()

    torch.manual_seed(0)
    restore_rng_state(cpu_snapshot)
    assert _sample_rng_triplet() == ref


def test_rng_restore_legacy_tensor_format():
    torch.manual_seed(31)
    legacy = {
        "torch_cpu": torch.get_rng_state().to(dtype=torch.uint8).contiguous(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    ref_t = torch.rand(3).tolist()

    torch.manual_seed(0)
    restore_rng_state(legacy)
    assert torch.rand(3).tolist() == ref_t


def test_relocate_optimizer_state_unifies_devices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 2, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    loss = model(torch.randn(2, 4, device=device)).sum()
    loss.backward()
    opt.step()

    # 模拟 async checkpoint：state 经 CPU 落盘再 map 回 GPU 时的混合设备。
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor):
                    st[k] = v.detach().cpu()
            if "exp_avg" in st:
                st["exp_avg"] = st["exp_avg"].to(device)

    devices = {
        v.device
        for st in opt.state.values()
        for v in st.values()
        if isinstance(v, torch.Tensor)
    }
    assert len(devices) > 1, "test setup must create mixed devices"

    n = relocate_optimizer_state(opt)
    assert n > 0
    for group in opt.param_groups:
        for p in group["params"]:
            for v in opt.state[p].values():
                if isinstance(v, torch.Tensor):
                    assert v.device == p.device

    loss = model(torch.randn(2, 4, device=device)).sum()
    loss.backward()
    opt.step()  # fused AdamW must not raise device mismatch


def test_relocate_optimizer_state_zero_wrapper():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 2, device=device)
    inner = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    loss = model(torch.randn(2, 4, device=device)).sum()
    loss.backward()
    inner.step()

    class _FakeZeroWrapper:
        optim = inner

    zro = _FakeZeroWrapper()
    for p in inner.param_groups[0]["params"]:
        st = inner.state[p]
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor):
                st[k] = v.detach().cpu()
        if "exp_avg" in st:
            st["exp_avg"] = st["exp_avg"].to(device)

    relocate_optimizer_state(zro)  # type: ignore[arg-type]
    for p in inner.param_groups[0]["params"]:
        for v in inner.state[p].values():
            if isinstance(v, torch.Tensor):
                assert v.device == p.device

    loss = model(torch.randn(2, 4, device=device)).sum()
    loss.backward()
    inner.step()


def test_metric_accumulator_logs_na_for_missing_loss(caplog):
    import logging

    from segtask_v1.trainer.validation import MetricAccumulator

    caplog.set_level(logging.INFO)
    acc = MetricAccumulator(
        criterion="dice",
        surface_dice_tolerance=1,
        surface_dice_weight=0.5,
    )
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.zeros(1, 1, 2, 2)
    acc.update(pred, target, loss_value=None)
    acc.compute(log_prefix="Val[full-3D]", log=True)
    assert any("loss=N/A" in rec.message for rec in caplog.records)


def test_load_checkpoint_source_uses_restore_helper():
    import inspect

    from segtask_v1.trainer import Trainer

    src = inspect.getsource(Trainer._load_checkpoint)
    assert "restore_rng_state" in src
    assert "relocate_optimizer_state" in src
