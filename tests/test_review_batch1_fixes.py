"""Regression tests for review batch-1 fixes.

覆盖：
- P1-02 SoftCLDiceLoss 完美预测损失应趋近 0（最终调和均值分母不加 smooth）
- P1-03 scheduler/EMA 仅在 optimizer 真正更新后推进（fp16 跳步 / 非有限 guard）
- P1-06 prob_to_label 阈值语义与验证侧一致（prob == threshold → 背景）
- P2-08 checkpoint 原子写（tmp + os.replace，失败保留旧文件）
- P3-01 Grid Dropout 起点上界含 D-hd（randint 上界 +1）

Run with:
    python -m pytest tests/test_review_batch1_fixes.py -v
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# P1-02: clDice
# ---------------------------------------------------------------------------
def test_soft_cldice_perfect_prediction_loss_is_near_zero():
    from segtask_v1.losses.losses import SoftCLDiceLoss

    torch.manual_seed(0)
    target = torch.zeros(1, 1, 24, 24, 24)
    target[0, 0, 4:20, 10:14, 10:14] = 1.0  # 管状前景
    logits = torch.where(target > 0.5, 30.0, -30.0)  # sigmoid ≈ 1/0

    loss = SoftCLDiceLoss(iter_=3, smooth=1.0)(logits, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-3), (
        "perfect prediction must give ~0 clDice loss "
        f"(got {loss.item():.4f}; 旧实现分母多加 smooth 时恒为 ~1/3)")


def test_soft_cldice_loss_stays_finite_with_zero_smooth():
    from segtask_v1.losses.losses import SoftCLDiceLoss

    target = torch.zeros(1, 1, 16, 16, 16)
    target[0, 0, 2:14, 7:9, 7:9] = 1.0
    logits = torch.where(target > 0.5, 30.0, -30.0)
    loss = SoftCLDiceLoss(iter_=2, smooth=0.0)(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# P1-03: scheduler/EMA 仅在 optimizer 真正更新后推进
# ---------------------------------------------------------------------------
def test_scheduler_and_ema_gated_on_effective_optimizer_step():
    from taskcore.engine.base_trainer import BaseTrainer

    # 门控语义收敛到 BaseTrainer._optimizer_step_boundary（五任务共用）。
    src = inspect.getsource(BaseTrainer._optimizer_step_boundary)

    # 默认路径：非有限 skip 分支在 always_step_scheduler 之外不得推 scheduler。
    # （ssl 经 always_step_scheduler=True 显式开边界时钟，不在此断言范围。）
    # skip 分支以 `if skip_optim_step:` 起，成功路径以 `if before_step` 起。
    skip_branch = src.split("if skip_optim_step:")[1].split(
        "if before_step is not None:")[0]
    # skip 分支内仅允许 always_step_scheduler 门控下的 scheduler.step
    assert "if always_step_scheduler:" in skip_branch
    # 去掉该门控块后，不应再有无条件 scheduler.step
    skip_wo_ssl = skip_branch.split("if always_step_scheduler:")[0]
    assert "scheduler.step()" not in skip_wo_ssl
    assert "ema.update" not in skip_branch
    assert "skipped_nonfinite=True" in skip_branch

    # GradScaler 成功路径：scheduler.step 与 ema.update 均在 not scaler_skipped 门控内。
    m = re.search(
        r"if not scaler_skipped:\s*\n"
        r"\s*self\.scheduler\.step\(\)\s*\n"
        r"\s*if self\.ema is not None:\s*\n"
        r"\s*self\.ema\.update", src)
    assert m is not None, "scheduler/EMA must be gated on 'not scaler_skipped'"


def test_gen_trainer_logs_unscaled_loss():
    """gen 记录的 loss 必须是 _step_loss 原始值（loss_scaled 仅供 backward）。

    回归：pending 里若再乘 eff_accum，accum_steps>1 时 loss_meter/history
    会被放大 accum 倍。
    """
    from gentask.trainer.gen_trainer import GenerationTrainer

    src = inspect.getsource(GenerationTrainer._train_epoch)
    assert "pending.append((step, loss.detach(), hr.shape[0]))" in src
    assert "* eff_accum" not in src.split("backward()")[1]


# ---------------------------------------------------------------------------
# P1-06: 部署阈值语义与验证侧（prob > threshold 取前景）一致
# ---------------------------------------------------------------------------
def test_prob_to_label_threshold_equality_is_background():
    from segtask_v1.predictor.blending import prob_to_label

    prob = np.zeros((1, 1, 1, 3), dtype=np.float32)
    prob[0, 0, 0] = [0.49, 0.50, 0.51]
    out = prob_to_label(
        prob, label_values=[0, 1], num_fg=1, threshold=0.5)
    # 与验证侧 (sigmoid(pred) > threshold) 相同：== threshold 判背景。
    assert out[0, 0].tolist() == [0, 0, 1]


def test_prob_to_label_per_class_threshold_equality_is_background():
    from segtask_v1.predictor.blending import prob_to_label

    prob = np.zeros((2, 1, 1, 2), dtype=np.float32)
    prob[:, 0, 0, 0] = [0.60, 0.10]   # argmax=class0, p=0.60 == thr0
    prob[:, 0, 0, 1] = [0.10, 0.45]   # argmax=class1, p=0.45 >  thr1=0.4
    out = prob_to_label(
        prob, label_values=[0, 1, 2], num_fg=2, threshold=[0.60, 0.40])
    assert out[0, 0].tolist() == [0, 2]


# ---------------------------------------------------------------------------
# P2-08: checkpoint 原子写
# ---------------------------------------------------------------------------
def test_atomic_torch_save_writes_loadable_file_and_leaves_no_tmp():
    from taskcore.engine.checkpoint import atomic_torch_save

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "best_model.pth"
        atomic_torch_save({"x": torch.arange(4)}, path)
        assert path.is_file()
        assert not (Path(td) / "best_model.pth.tmp").exists()
        loaded = torch.load(path, weights_only=False)
        assert torch.equal(loaded["x"], torch.arange(4))


def test_atomic_torch_save_failure_preserves_previous_checkpoint():
    from taskcore.engine.checkpoint import atomic_torch_save

    class _Unpicklable:
        def __reduce__(self):
            raise RuntimeError("simulated mid-write failure")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "best_model.pth"
        atomic_torch_save({"epoch": 1}, path)
        with pytest.raises(RuntimeError):
            atomic_torch_save({"bad": _Unpicklable()}, path)
        # 旧文件完好、无残留 tmp。
        assert torch.load(path, weights_only=False)["epoch"] == 1
        assert not (Path(td) / "best_model.pth.tmp").exists()


def test_async_checkpoint_saver_uses_atomic_write():
    from taskcore.engine.checkpoint import AsyncCheckpointSaver

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "checkpoint_epoch_1.pth"
        saver = AsyncCheckpointSaver()
        try:
            saver.submit({"epoch": 1}, path)
            saver.wait()
        finally:
            saver.close()
        assert torch.load(path, weights_only=False)["epoch"] == 1
        assert not (Path(td) / "checkpoint_epoch_1.pth.tmp").exists()


# ---------------------------------------------------------------------------
# P3-01: Grid Dropout 起点上界
# ---------------------------------------------------------------------------
def test_grid_dropout_hole_start_reaches_last_valid_position():
    from taskcore.data import augment as aug

    D = H = W = 8
    image = torch.ones(1, 1, D, H, W)
    label = torch.zeros(1, 1, D, H, W)

    # ratio/num_holes 使 hd=hh=hw=4：合法起点 0..4，旧实现 randint 上界为 4
    # 永远采不到 4。多次采样后洞应能覆盖到最末端体素 (7,7,7)。
    torch.manual_seed(0)
    tail_hit = False
    for _ in range(200):
        out, _, _ = aug._grid_dropout(
            image.clone(), label, prob=1.0, ratio=0.125, num_holes=1)
        if out[0, 0, D - 1, H - 1, W - 1] == 0:
            tail_hit = True
            break
    assert tail_hit, "hole start D-hd must be reachable (randint upper bound +1)"

    # hd == 轴长（frac >= 1）时起点只能是 0，且不得越界。
    out, _, _ = aug._grid_dropout(
        image.clone(), label, prob=1.0, ratio=8.0, num_holes=1)
    assert (out == 0).all()
