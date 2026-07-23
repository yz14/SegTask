"""R4: seg/gen best checkpoint 统一走 BaseTrainer._save_best。"""

from __future__ import annotations

import inspect

from taskcore.engine.base_trainer import BaseTrainer
from gentask.trainer.gen_trainer import GenerationTrainer
from segtask_v1.trainer.trainer import Trainer as SegTrainer


def test_gen_does_not_override_save_best():
    """gen 删除自定义 _save_best，直接继承 BaseTrainer。"""
    assert " _save_best" not in inspect.getsource(GenerationTrainer)
    assert GenerationTrainer._save_best is BaseTrainer._save_best


def test_seg_best_path_calls_base_save_best():
    """seg fit 选模落盘使用 BaseTrainer._save_best（非自管 best 写盘）。"""
    src = inspect.getsource(SegTrainer.fit)
    assert "self._save_best(" in src
    assert "is_best=True" not in src or "_save_checkpoint(epoch, is_best=True)" not in src


def test_seg_save_checkpoint_forwards_legacy_is_best():
    """_save_checkpoint(is_best=True) 防御性转发到 _save_best。"""
    src = inspect.getsource(SegTrainer._save_checkpoint)
    assert "self._save_best(" in src
    assert "best_model.pth" not in src  # best 路径不再本方法写盘
