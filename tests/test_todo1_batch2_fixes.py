"""批 2 回归测试：数学等价的显存、吞吐与观测改动。"""

from __future__ import annotations

from pathlib import Path

import torch


def test_state_to_cpu_does_not_alias_cpu_source():
    from taskcore.engine.checkpoint import state_to_cpu

    source = torch.ones(3)
    saved = state_to_cpu(source)
    source.zero_()
    assert torch.equal(saved, torch.ones(3))
    assert saved.data_ptr() != source.data_ptr()


def test_ema_apply_restore_preserves_values_on_cpu():
    from taskcore.utils.common import ModelEMA

    model = torch.nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.9)
    online = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.apply_shadow(model)
    ema.restore(model)
    for key, value in model.state_dict().items():
        assert torch.equal(value, online[key])


def test_ema_cpu_offload_does_not_request_pinned_memory_without_cuda():
    from taskcore.utils.common import ModelEMA

    model = torch.nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.9, offload_device="cpu")
    ema.apply_shadow(model)
    assert all(not value.is_pinned() for value in ema._backup.values())
    ema.restore(model)


def test_best_checkpoint_async_snapshots_live_model_without_ema():
    from taskcore.engine.base_trainer import BaseTrainer

    class Saver:
        def submit(self, state, path, on_done):
            self.state = state

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer._is_main = True
    trainer.model = torch.nn.Linear(2, 2)
    trainer.ema = None
    trainer.best_metric = 1.0
    trainer.best_epoch = 0
    trainer.best_key = "score"
    trainer.cfg = object()
    trainer.output_dir = Path(".")
    trainer._ckpt_saver = Saver()
    trainer._ckpt_task_label = "model"
    trainer._save_best(0, {"score": 1.0})
    saved = trainer._ckpt_saver.state["model_state_dict"]
    assert all(t.device.type == "cpu" for t in saved.values())
    original = {k: v.clone() for k, v in saved.items()}
    for value in trainer.model.parameters():
        value.data.zero_()
    assert all(torch.equal(saved[k], value) for k, value in original.items())


def test_memory_estimate_exposes_ema_backup_budget():
    from taskcore.engine.memory import estimate_train_memory

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = type("EMA", (), {
        "shadow": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "_backup": {},
    })()
    result = estimate_train_memory(model, optimizer, ema)
    assert result["ema_mib"] == 0.0
    assert result["ema_backup_mib"] == 0.0
    assert "persistent_mib" in result


def test_ssl_gradient_sync_uses_flattened_buckets():
    import ssltask.trainer.ssl_trainer as trainer_mod

    class FakeDist:
        class ReduceOp:
            SUM = object()

        def __init__(self):
            self.calls = 0

        def all_reduce(self, value, op):
            self.calls += 1
            value.mul_(2)

    method = torch.nn.Module()
    method.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    method.b = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    method.a.grad = torch.tensor([2.0, 4.0])
    method.b.grad = torch.tensor([6.0, 8.0], dtype=torch.float64)
    trainer = object.__new__(trainer_mod.SSLTrainer)
    trainer.method = method
    trainer._is_dist = True
    trainer._world_size = 2
    fake = FakeDist()
    old_dist = trainer_mod.dist
    trainer_mod.dist = fake
    try:
        trainer._sync_grads()
    finally:
        trainer_mod.dist = old_dist
    assert fake.calls == 2
    assert torch.equal(method.a.grad, torch.tensor([2.0, 4.0]))
    assert torch.equal(method.b.grad, torch.tensor([6.0, 8.0],
                                                     dtype=torch.float64))


def test_optimizer_observation_exposes_planned_actual_and_scheduler():
    from taskcore.engine.base_trainer import BaseTrainer

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer._planned_optimizer_steps = 12
    trainer._actual_optimizer_steps = 7
    trainer.scheduler = type("Scheduler", (), {"current_step": 7})()
    assert trainer.optimizer_step_observation() == {
        "planned_optimizer_steps": 12,
        "actual_optimizer_steps": 7,
        "scheduler_steps": 7,
    }


def test_grid_dropout_uses_output_clone_instead_of_full_hole_mask():
    from taskcore.data.augment import _bernoulli_mask, _grid_dropout_companions

    image = torch.arange(2 * 1 * 4 * 5 * 6, dtype=torch.float32).reshape(
        2, 1, 4, 5, 6)
    before = image.clone()
    gen_new = torch.Generator().manual_seed(7)
    out, _ = _grid_dropout_companions(
        image, 0.5, 0.3, 2, gen_cpu=gen_new, gen_dev=gen_new)

    gen_old = torch.Generator().manual_seed(7)
    selected = _bernoulli_mask(2, 0.5, gen_old)
    frac = (0.3 / 2) ** (1.0 / 3.0)
    hd, hh, hw = (min(4, max(1, int(4 * frac))),
                  min(5, max(1, int(5 * frac))),
                  min(6, max(1, int(6 * frac))))
    d0 = torch.randint(0, 4 - hd + 1, (2, 2), generator=gen_old)
    h0 = torch.randint(0, 5 - hh + 1, (2, 2), generator=gen_old)
    w0 = torch.randint(0, 6 - hw + 1, (2, 2), generator=gen_old)
    mask = torch.ones(2, 1, 4, 5, 6)
    for k in range(2):
        for b in range(2):
            if selected[b]:
                mask[b, :, d0[b, k]:d0[b, k] + hd,
                     h0[b, k]:h0[b, k] + hh,
                     w0[b, k]:w0[b, k] + hw] = 0
    assert torch.equal(out, image * mask)
    assert torch.equal(image, before)
    assert torch.equal(out[~selected], image[~selected])


def test_foreground_index_is_included_in_cache_budget_log():
    source = Path("taskcore/data/loader.py").read_text(encoding="utf-8")
    assert "Foreground index footprint" in source
    assert "index_bytes" in source


def test_lr_multiplier_path_keeps_zero_redundancy_option(tmp_path):
    import torch.distributed as dist
    from taskcore.config.core import Config
    import taskcore.engine.optim as optim_mod

    if not dist.is_available():
        return
    rendezvous = f"file://{tmp_path / 'rdzv'}"
    dist.init_process_group("gloo", rank=0, world_size=1,
                            init_method=rendezvous)
    try:
        cfg = Config()
        cfg.train.zero_redundancy_optimizer = True
        cfg.train.optimizer = "adamw"
        cfg.train.lr = 0.01
        cfg.train.weight_decay = 0.1
        cfg.sync()
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(2, 2)
        model.head = torch.nn.Linear(2, 1)
        original_enabled = optim_mod._zero_redundancy_enabled
        optim_mod._zero_redundancy_enabled = lambda _: True
        try:
            optimizer = optim_mod.build_optimizer_with_lr_mult(
                model, cfg, encoder_lr_mult=0.25)
        finally:
            optim_mod._zero_redundancy_enabled = original_enabled
        assert type(optimizer).__name__ == "ZeroRedundancyOptimizer"
        groups = optimizer.param_groups
        assert {round(float(g["lr"]), 6) for g in groups} == {0.0025, 0.01}
        assert {float(g["weight_decay"]) for g in groups} == {0.0, 0.1}
    finally:
        dist.destroy_process_group()


def test_edm2_sdpa_matches_fallback_and_enforces_token_guard(monkeypatch):
    import taskcore.models.edm2_unet as edm2

    block = edm2._Block(4, 4, attention=True, channels_per_head=2).eval()
    x = torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        sdpa_out = block(x)

    def unavailable(*args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(edm2.F, "scaled_dot_product_attention", unavailable)
    with torch.no_grad():
        fallback_out = block(x)
    assert torch.allclose(sdpa_out, fallback_out, rtol=1e-5, atol=1e-6)

    oversized = torch.randn(1, 4, 1, 32769)
    with torch.no_grad():
        try:
            block(oversized)
        except ValueError as exc:
            assert "token count" in str(exc)
        else:
            raise AssertionError(
                "expected EDM2 token guard to reject oversized input")
