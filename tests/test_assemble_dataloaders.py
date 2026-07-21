"""P2f：assemble_train_val_loaders / ensure_train_batch_capacity 回归。"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from taskcore.config.core import Config
from taskcore.data.loader import (
    assemble_train_val_loaders,
    ensure_train_batch_capacity,
)


class _Toy(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": torch.zeros(1), "i": i}


def _cfg(batch_size: int = 2, num_workers: int = 0) -> Config:
    cfg = Config()
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.batch_size = batch_size
    cfg.data.num_workers = num_workers
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.sync()
    return cfg


def test_ensure_train_batch_capacity_raises():
    with pytest.raises(ValueError, match="zero batches"):
        ensure_train_batch_capacity(_Toy(1), batch_size=4)


def test_assemble_single_process_drop_last():
    cfg = _cfg(batch_size=2)
    train_loader, val_loader = assemble_train_val_loaders(
        _Toy(5), _Toy(3), cfg, log_prefix="test")
    assert train_loader.drop_last is True
    assert val_loader.drop_last is False
    assert len(train_loader) == 2  # floor(5/2) with drop_last
    batch = next(iter(train_loader))
    assert batch["x"].shape[0] == 2


def test_assemble_rejects_undersized_train():
    cfg = _cfg(batch_size=8)
    with pytest.raises(ValueError, match="zero batches"):
        assemble_train_val_loaders(_Toy(3), _Toy(3), cfg)


def test_assemble_train_drop_last_false_keeps_tail():
    """cls/det 单进程契约：drop_last=False 保留尾批，小数据集不拦截。"""
    cfg = _cfg(batch_size=2)
    train_loader, _ = assemble_train_val_loaders(
        _Toy(5), _Toy(3), cfg, train_drop_last=False)
    assert train_loader.drop_last is False
    assert len(train_loader) == 3  # ceil(5/2)，尾批保留

    cfg8 = _cfg(batch_size=8)
    train_loader, _ = assemble_train_val_loaders(
        _Toy(3), _Toy(3), cfg8, train_drop_last=False)
    assert len(train_loader) == 1  # 不足一个 batch 也应产出


def test_assemble_capacity_checked_under_ddp_semantics():
    """DDP 恒 drop_last=True：即便 train_drop_last=False 也须拦零批次。

    不真起进程组，仅验证 world_size>1 时装配前的容量检查生效
    （DistributedSampler 允许未初始化 dist 时显式传 num_replicas/rank）。
    """
    cfg = _cfg(batch_size=8)
    with pytest.raises(ValueError, match="zero batches"):
        assemble_train_val_loaders(
            _Toy(3), _Toy(3), cfg, rank=0, world_size=2,
            train_drop_last=False)
