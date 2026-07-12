"""S1 回归：SSLTrainer 的 scheduler/warmup/方法 schedule 统一 optimizer-step 时钟。

覆盖：
- grad_accum_steps=1 时行为与旧口径一致（micro==opt step）。
- grad_accum_steps>1 时 scheduler horizon 按 ceil(len(loader)/accum) 构建，
  一个训练周期结束后 warmup+cosine 恰好走完（LR 收敛到 cosine_min_lr 附近）。
- 尾批不足 accum 的组只推进一次 scheduler/global_step。
- 方法 configure_schedule 收到的总步数与 trainer 时钟一致。
- resume 后 scheduler current_step 连续。
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from segtask_v1.config import Config as SegConfig
from ssltask.config import SSLConfig, validate_ssl
from ssltask.methods import build_method
from ssltask.trainer import SSLTrainer


class _ImgDataset(Dataset):
    def __init__(self, n, ch, shape):
        self.x = [torch.rand(ch, *shape) for _ in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i]}


def _cfg():
    cfg = SegConfig()
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.sync()
    cfg.validate()
    return cfg


def _make_trainer(tmp_path, *, n_samples, batch_size, accum, epochs,
                  warmup_epochs=0, scheduler="cosine", method="genesis"):
    cfg = _cfg()
    cfg.train.epochs = epochs
    cfg.train.warmup_epochs = warmup_epochs
    cfg.train.scheduler = scheduler
    cfg.train.use_ema = False
    cfg.train.use_amp = False
    cfg.train.grad_accum_steps = accum
    cfg.train.output_dir = str(tmp_path)
    cfg.sync()
    cfg.validate()

    ssl = SSLConfig(method=method)
    validate_ssl(ssl, cfg)
    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(n_samples, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=batch_size)
    return SSLTrainer(m, cfg, ssl, loader, device), cfg


def test_accum1_clock_matches_micro_steps(tmp_path):
    """accum=1：opt step == micro step，horizon 与旧口径一致。"""
    trainer, cfg = _make_trainer(
        tmp_path, n_samples=6, batch_size=2, accum=1, epochs=2)
    micro = len(trainer.train_loader)                    # 3
    assert trainer._opt_steps_per_epoch == micro
    assert trainer._total_opt_steps == cfg.train.epochs * micro
    trainer.fit()
    assert trainer.scheduler.current_step == trainer._total_opt_steps
    assert trainer._global_step == trainer._total_opt_steps


@pytest.mark.parametrize("n_samples,batch_size,accum", [
    (8, 2, 2),    # 4 micro / epoch, 整除
    (10, 2, 4),   # 5 micro / epoch, 尾批 1 个 micro 的 partial 组
])
def test_accum_gt1_scheduler_completes(tmp_path, n_samples, batch_size, accum):
    """accum>1：horizon=ceil(micro/accum)*epochs；训练结束 scheduler 恰好走满，
    LR 收敛到 cosine 末端（而非只走了 1/accum 的曲线）。"""
    epochs = 3
    trainer, cfg = _make_trainer(
        tmp_path, n_samples=n_samples, batch_size=batch_size,
        accum=accum, epochs=epochs)
    micro = len(trainer.train_loader)
    expect_per_epoch = math.ceil(micro / accum)
    assert trainer._opt_steps_per_epoch == expect_per_epoch
    assert trainer._total_opt_steps == epochs * expect_per_epoch

    trainer.fit()
    # 尾批 partial 组只推进一次：总推进数 == 边界数 == total_opt_steps。
    assert trainer.scheduler.current_step == trainer._total_opt_steps
    assert trainer._global_step == trainer._total_opt_steps
    # cosine 按 optimizer-step horizon 构建 → 结束时 LR 达到曲线末端。
    final_lr = trainer.scheduler.get_lr()
    assert final_lr == pytest.approx(cfg.train.cosine_min_lr, abs=1e-8)


def test_warmup_in_opt_steps(tmp_path):
    """warmup_epochs 换算成 optimizer-step：warmup_steps=warmup_epochs*ceil(micro/accum)。"""
    trainer, cfg = _make_trainer(
        tmp_path, n_samples=8, batch_size=2, accum=2, epochs=4,
        warmup_epochs=1)
    assert trainer.scheduler.warmup_steps == trainer._opt_steps_per_epoch
    trainer.fit()
    # warmup 结束后 cosine 段 = (epochs-warmup)*opt_steps_per_epoch，正好走满。
    assert trainer.scheduler.current_step == trainer._total_opt_steps
    assert trainer.scheduler.get_lr() == pytest.approx(
        cfg.train.cosine_min_lr, abs=1e-8)


def test_method_schedule_uses_same_clock(tmp_path):
    """方法（如 DINO EMA/温度 cosine）的总步数与 trainer optimizer-step 时钟一致。"""
    trainer, _ = _make_trainer(
        tmp_path, n_samples=8, batch_size=2, accum=4, epochs=2, method="dino")
    assert trainer.method.total_steps == trainer._total_opt_steps


def test_resume_scheduler_continuity(tmp_path):
    """resume 后 scheduler.current_step 与 global_step 连续，不重复也不跳步。"""
    trainer, cfg = _make_trainer(
        tmp_path, n_samples=8, batch_size=2, accum=2, epochs=1)
    trainer.fit()
    step_after_e1 = trainer.scheduler.current_step
    assert step_after_e1 == trainer._opt_steps_per_epoch

    # 用 resume ckpt 续训到 2 epochs。
    cfg.train.epochs = 2
    cfg.train.resume = str(tmp_path / "ssl_resume.pt")
    cfg.sync()
    cfg.validate()
    ssl = SSLConfig(method="genesis")
    validate_ssl(ssl, cfg)
    device = torch.device("cpu")
    m = build_method(cfg, ssl, device)
    ds = _ImgDataset(8, cfg.model.in_channels, (16, 32, 32))
    loader = DataLoader(ds, batch_size=2)
    trainer2 = SSLTrainer(m, cfg, ssl, loader, device)
    assert trainer2.scheduler.current_step == step_after_e1
    assert trainer2._global_step == step_after_e1
    trainer2.fit()
    assert trainer2.scheduler.current_step == trainer2._total_opt_steps
    assert trainer2._global_step == trainer2._total_opt_steps
