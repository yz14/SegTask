"""Unit tests for ``taskcore.data.mixed_sampler`` + 双批混合配置校验。

CPU-only、不读盘。覆盖：

1. ``resolve_per_batch_counts`` 把整数权重比解析为每 batch 计数（含非法输入）
2. ``MixedBatchSampler`` 每 batch 金/粗计数正确、粗标每 epoch 恰好覆盖一遍、
   ``__len__`` 正确、金标准循环过采样、跨 epoch 顺序可复现且各异
3. ``SourceTaggedDataset`` 透传字段并注入 ``source`` 标量
4. ``Config._validate_data`` 仅在 ``npz_dir_secondary`` 非空时校验 ``mix_ratio``
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from taskcore.config.core import Config, ConfigError
from taskcore.data.mixed_sampler import (
    SOURCE_PRIMARY,
    SOURCE_SECONDARY,
    MixedBatchSampler,
    SourceTaggedDataset,
    resolve_per_batch_counts,
)


# ---------------------------------------------------------------------------
# resolve_per_batch_counts
# ---------------------------------------------------------------------------
def test_resolve_counts_balanced():
    assert resolve_per_batch_counts([1, 1], 4) == (2, 2)


def test_resolve_counts_weighted():
    assert resolve_per_batch_counts([1, 3], 4) == (1, 3)
    assert resolve_per_batch_counts([1, 3], 8) == (2, 6)


@pytest.mark.parametrize("ratio,bs", [([1, 1], 3), ([2, 3], 4)])
def test_resolve_counts_indivisible(ratio, bs):
    with pytest.raises(ValueError):
        resolve_per_batch_counts(ratio, bs)


@pytest.mark.parametrize("ratio", [[0, 1], [1, 0], [1, 2, 3]])
def test_resolve_counts_bad_ratio(ratio):
    with pytest.raises(ValueError):
        resolve_per_batch_counts(ratio, 4)


# ---------------------------------------------------------------------------
# MixedBatchSampler
# ---------------------------------------------------------------------------
def test_sampler_batch_composition_and_coverage():
    n_primary, n_secondary = 7, 20
    gold_pb, coarse_pb = 1, 3            # batch_size=4, mix_ratio=[1,3]
    s = MixedBatchSampler(n_primary, n_secondary, gold_pb, coarse_pb, seed=0)

    expected_batches = n_secondary // coarse_pb   # 6
    assert len(s) == expected_batches

    batches = list(iter(s))
    assert len(batches) == expected_batches

    seen_secondary = []
    for batch in batches:
        assert len(batch) == gold_pb + coarse_pb
        golds   = [i for i in batch if i < n_primary]
        coarses = [i for i in batch if i >= n_primary]
        assert len(golds) == gold_pb
        assert len(coarses) == coarse_pb
        # 全局索引落在各自区间内。
        assert all(0 <= i < n_primary for i in golds)
        assert all(n_primary <= i < n_primary + n_secondary for i in coarses)
        seen_secondary.extend(i - n_primary for i in coarses)

    # 粗标本 epoch 恰好覆盖一遍（floor(20/3)*3 = 18 个唯一索引，无重复）。
    assert len(seen_secondary) == expected_batches * coarse_pb
    assert len(set(seen_secondary)) == len(seen_secondary)


def test_sampler_gold_oversampled_when_minority():
    # 金标准很少：仍应每 batch 出现，且通过循环重采样填满。
    s = MixedBatchSampler(
        n_primary=2, n_secondary=10, gold_per_batch=1, coarse_per_batch=1,
        seed=1)
    batches = list(iter(s))
    assert len(batches) == 10
    gold_hits = sum(1 for b in batches for i in b if i < 2)
    assert gold_hits == 10   # 每 batch 恰 1 个金标准


def test_sampler_reproducible_and_epoch_varies():
    def first_batch(seed_epoch_runs):
        s = MixedBatchSampler(5, 12, 1, 2, seed=42)
        out = []
        for _ in range(seed_epoch_runs):
            out.append(sorted(next(iter(s))))
        return out

    # 同一 sampler 连续两个 epoch 顺序应不同（epoch 计数推进 RNG）。
    s = MixedBatchSampler(5, 12, 1, 2, seed=42)
    ep0 = [sorted(b) for b in iter(s)]
    ep1 = [sorted(b) for b in iter(s)]
    assert ep0 != ep1

    # 相同 seed 重新构造 → 第 0 个 epoch 完全可复现。
    s2 = MixedBatchSampler(5, 12, 1, 2, seed=42)
    ep0_again = [sorted(b) for b in iter(s2)]
    assert ep0 == ep0_again


@pytest.mark.parametrize("kwargs", [
    dict(n_primary=0, n_secondary=10, gold_per_batch=1, coarse_per_batch=1),
    dict(n_primary=10, n_secondary=2, gold_per_batch=1, coarse_per_batch=3),
    dict(n_primary=10, n_secondary=10, gold_per_batch=0, coarse_per_batch=1),
])
def test_sampler_degenerate_raises(kwargs):
    with pytest.raises(ValueError):
        MixedBatchSampler(**kwargs)


# ---------------------------------------------------------------------------
# SourceTaggedDataset
# ---------------------------------------------------------------------------
class _DictDataset(Dataset):
    def __init__(self, n, marker):
        self.n = n
        self.marker = marker

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"image": torch.full((2,), float(idx)), "marker": self.marker}


def test_source_tagged_injects_source_and_passes_through():
    base = _DictDataset(3, marker="gold")
    tagged = SourceTaggedDataset(base, SOURCE_PRIMARY)
    assert len(tagged) == 3
    sample = tagged[1]
    assert sample["marker"] == "gold"
    assert torch.equal(sample["image"], torch.full((2,), 1.0))
    assert int(sample["source"]) == SOURCE_PRIMARY
    assert sample["source"].dtype == torch.long

    tagged2 = SourceTaggedDataset(_DictDataset(2, "coarse"), SOURCE_SECONDARY)
    assert int(tagged2[0]["source"]) == SOURCE_SECONDARY


def test_source_tagged_forwards_attributes():
    base = _DictDataset(3, marker="gold")
    base.some_attr = [1, 2, 3]
    tagged = SourceTaggedDataset(base, SOURCE_PRIMARY)
    assert tagged.some_attr == [1, 2, 3]


def test_source_tagged_pickle_roundtrip():
    import pickle
    tagged = SourceTaggedDataset(_DictDataset(3, "gold"), SOURCE_PRIMARY)
    restored = pickle.loads(pickle.dumps(tagged))
    assert len(restored) == 3
    assert int(restored[1]["source"]) == SOURCE_PRIMARY
    assert restored[1]["marker"] == "gold"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def _cfg_with_secondary(ratio, batch_size, secondary="some/dir"):
    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.npz_dir = "primary/dir"
    cfg.data.npz_dir_secondary = secondary
    cfg.data.mix_ratio = ratio
    cfg.data.batch_size = batch_size
    return cfg


def test_validate_mixed_ok():
    cfg = _cfg_with_secondary([1, 3], 4)
    cfg.validate()   # should not raise


def test_validate_mixed_indivisible():
    cfg = _cfg_with_secondary([1, 1], 3)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_mixed_bad_ratio():
    cfg = _cfg_with_secondary([0, 1], 4)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_no_secondary_skips_ratio_check():
    # 副源为空时，即便 mix_ratio 非法也不应触发校验（退化为单批行为）。
    cfg = _cfg_with_secondary([0, 5], 3, secondary="")
    cfg.validate()   # should not raise
