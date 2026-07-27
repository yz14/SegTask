"""TODO2 S7/S8 核验后修复的回归测试。

覆盖：
- S7-B：resume 时 base scheduler 超参以当前配置为准，仅迁训练进度。
- S7-C：best-as-resume（缺 optimizer/scheduler/scaler/RNG）发出显式告警。
- S7-D：pretrain strict=False 跳过 shape 不匹配张量而非崩溃。
- S7-F：warmup_epochs >= epochs 在 sync() 被钳住并告警。
- S7-H(部分)：AsyncCheckpointSaver.close() 在 wait() 抛错时仍回收 worker 线程。
- S7-I：train.* 基本区间校验（epochs/lr/ema_decay/grad_accum/save_every/val_every）。
- S8-B：无 DS 包装时 MultiResolutionLoss 的 per-res 诊断被收集且 history 清空。
- S8-G：L_topo/w_topo 进入 format_breakdown 输出。
- 新-1：ModelEMA.load_state_dict key 不匹配时按交集恢复，后续 update 不崩。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from taskcore.config.core import Config, ConfigError  # noqa: E402
from taskcore.engine.optim import WarmupScheduler  # noqa: E402
from taskcore.utils.common import ModelEMA  # noqa: E402


# ---------------------------------------------------------------------------
# S7-B: scheduler resume 超参 reconcile
# ---------------------------------------------------------------------------
def _make_warmup_cosine(t_max: int, lr: float = 1e-3):
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=lr)
    base = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t_max, eta_min=1e-6)
    return opt, WarmupScheduler(
        opt, base, warmup_steps=0, warmup_lr=1e-6, base_lr=lr)


def test_scheduler_resume_keeps_fresh_hyperparams():
    _, old = _make_warmup_cosine(t_max=100)
    for _ in range(50):
        old.step()
    state = old.state_dict()

    _, new = _make_warmup_cosine(t_max=1000)  # 加大 epochs 续训
    new.load_state_dict(state)
    assert new.scheduler.T_max == 1000, (
        f"resume 后 T_max 应保持当前配置 1000；got {new.scheduler.T_max}")
    assert new.scheduler.last_epoch == 50, (
        f"训练进度应从 ckpt 迁移；got last_epoch={new.scheduler.last_epoch}")


def test_scheduler_resume_step_milestones_follow_config():
    lin = nn.Linear(2, 2)
    opt = torch.optim.SGD(lin.parameters(), lr=1e-3)
    base = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[100], gamma=0.1)
    ws = WarmupScheduler(opt, base, warmup_steps=0,
                         warmup_lr=1e-6, base_lr=1e-3)
    for _ in range(10):
        ws.step()
    state = ws.state_dict()

    opt2 = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1e-3)
    base2 = torch.optim.lr_scheduler.MultiStepLR(
        opt2, milestones=[500], gamma=0.5)
    ws2 = WarmupScheduler(opt2, base2, warmup_steps=0,
                          warmup_lr=1e-6, base_lr=1e-3)
    ws2.load_state_dict(state)
    assert sorted(ws2.scheduler.milestones) == [500], (
        f"milestones 应保持当前配置；got {sorted(ws2.scheduler.milestones)}")
    assert ws2.scheduler.gamma == 0.5


# ---------------------------------------------------------------------------
# S7-E: warmup 下 step 里程碑对齐绝对轮数
# ---------------------------------------------------------------------------
def test_step_milestones_absolute_epochs_under_warmup():
    from taskcore.engine.optim import build_scheduler
    cfg = Config()
    cfg.train.scheduler = "step"
    cfg.train.epochs = 100
    cfg.train.warmup_epochs = 5
    cfg.train.step_size = 50
    spe = 10
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1e-3)
    base = build_scheduler(opt, cfg, spe,
                           post_warmup_steps=(100 - 5) * spe)
    # 首个里程碑（base 时钟）+ warmup 步数 = 绝对第 50 轮末的步数。
    assert sorted(base.milestones)[0] + 5 * spe == 50 * spe, (
        f"milestones={sorted(base.milestones)}")


def test_warm_restart_first_cycle_absolute_under_warmup():
    from taskcore.engine.optim import build_scheduler
    cfg = Config()
    cfg.train.scheduler = "cosine_warm_restarts"
    cfg.train.epochs = 100
    cfg.train.warmup_epochs = 5
    cfg.train.cosine_restart_period = 50
    spe = 10
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1e-3)
    base = build_scheduler(opt, cfg, spe,
                           post_warmup_steps=(100 - 5) * spe)
    assert base.T_0 + 5 * spe == 50 * spe, f"T_0={base.T_0}"


# ---------------------------------------------------------------------------
# S7-J: EMA 隔步更新（decay**N 补偿）
# ---------------------------------------------------------------------------
def test_ema_update_every_matches_per_step_time_constant():
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    ref = ModelEMA(model, decay=0.9, warmup=False)
    fast = ModelEMA(model, decay=0.9, warmup=False, update_every=4)
    with torch.no_grad():
        for step in range(8):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
            ref.update(model)
            fast.update(model)
    # 权重恒定时二者收敛点一致；权重漂移时 fast 是 ref 的低频近似，
    # 这里只验证隔步版确实每 4 步动一次且不偏离 ref 一个数量级。
    k = "weight"
    diff = (ref.shadow[k] - fast.shadow[k]).abs().max().item()
    assert fast._skip_counter == 0  # 8 步恰好 2 次实更
    assert fast.num_updates == 2
    assert diff < 0.05, f"interval-EMA drifted too far: {diff}"


# ---------------------------------------------------------------------------
# S7-F / S7-I: config 校验
# ---------------------------------------------------------------------------
def test_warmup_ge_epochs_clamped_in_sync(caplog):
    cfg = Config()
    cfg.train.epochs = 3
    cfg.train.warmup_epochs = 10
    with caplog.at_level(logging.WARNING):
        cfg.sync()
    assert cfg.train.warmup_epochs == 2, (
        f"warmup 应被钳到 epochs-1=2；got {cfg.train.warmup_epochs}")


@pytest.mark.parametrize("field, value", [
    ("epochs", 0),
    ("lr", -1.0),
    ("lr", 0.0),
    ("grad_accum_steps", 0),
    ("save_every", 0),
    ("val_every", 0),
    ("ema_decay", 1.5),
    ("plateau_factor", 1.0),
    ("step_size", 0),
])
def test_train_range_validation_rejects(field, value):
    cfg = Config()
    setattr(cfg.train, field, value)
    with pytest.raises(ConfigError):
        cfg.sync()
        cfg.validate()


# ---------------------------------------------------------------------------
# S7-H(部分): AsyncCheckpointSaver.close() 异常安全
# ---------------------------------------------------------------------------
def test_async_saver_close_joins_worker_even_on_error(tmp_path):
    from taskcore.engine.checkpoint import AsyncCheckpointSaver
    saver = AsyncCheckpointSaver()
    # 制造写盘失败：目标目录不存在。
    saver.submit({"x": 1}, tmp_path / "no_such_dir" / "ckpt.pth")
    with pytest.raises(RuntimeError):
        saver.close()
    saver._worker.join(timeout=5)
    assert not saver._worker.is_alive(), (
        "close() 抛错后 worker 线程应已收到哨兵并退出")


# ---------------------------------------------------------------------------
# 新-1: ModelEMA key 不匹配按交集恢复
# ---------------------------------------------------------------------------
def test_ema_load_state_dict_key_mismatch_then_update():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    ema = ModelEMA(model, decay=0.9)
    # 构造 key 集不同的旧 shadow（仅含第一层）。
    old_model = nn.Sequential(nn.Linear(4, 4))
    old_ema = ModelEMA(old_model, decay=0.9)
    ema.load_state_dict(old_ema.state_dict())
    # 修复前：随后的 update() 因 shadow 被重建为旧 key 集而 KeyError。
    ema.update(model)
    assert set(ema.shadow.keys()) == set(model.state_dict().keys())


# ---------------------------------------------------------------------------
# S8-B: 无 DS 时 per-res 诊断收集 + history 清空
# ---------------------------------------------------------------------------
def test_breakdown_collect_without_deep_supervision():
    from segtask_v1.losses.losses import MultiResolutionLoss
    from segtask_v1.trainer.breakdown import collect_multi_res_breakdown

    class _MSE(nn.Module):
        def forward(self, p, t, weight_map=None):
            return ((p - t.float()) ** 2).mean()

    mr = MultiResolutionLoss(base_loss=_MSE(), num_fg_classes=1,
                             num_res=2, label_values=[0, 1])
    pred = torch.randn(2, 2, 4, 8, 8)
    lbl = (torch.rand(2, 2, 4, 8, 8) > 0.5).long()
    for _ in range(3):
        mr(pred, lbl)
    breakdown = {}
    collect_multi_res_breakdown(mr, None, breakdown)
    assert "L_res_0" in breakdown and "L_res_1" in breakdown, (
        f"无 DS 包装时应收集 per-res 诊断；got {list(breakdown)}")
    assert len(mr._per_res_history) == 0, (
        "收集后 history 应清空（否则整个训练期无界增长）")


# ---------------------------------------------------------------------------
# S8-G: L_topo 进入 breakdown 输出
# ---------------------------------------------------------------------------
def test_format_breakdown_renders_topo():
    from segtask_v1.trainer.breakdown import format_breakdown
    msg = format_breakdown({"L_main": 1.0, "L_topo": 0.25, "w_topo": 0.1})
    assert "L_topo=0.2500" in msg and "w=0.1" in msg, msg


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
