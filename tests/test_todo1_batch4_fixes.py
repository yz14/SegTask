from __future__ import annotations

import logging

import numpy as np

from taskcore.config.core import Config
from taskcore.data.loader import (
    finalize_from_data,
    group_aware_train_val_split,
)
from taskcore.data.make_data import _META_SPACING_ATOL
from taskcore.data.mixed_sampler import MixedBatchSampler


def test_group_aware_entry_prefers_groups_and_preserves_empty_regex():
    paths = ["P1_a.npz", "P1_b.npz", "P2_a.npz", "P3_a.npz"]
    train, val = group_aware_train_val_split(
        paths, 0.34, 7, group_id_regex=r"^(P\d+)_")
    assert {p.split("_")[0] for p in paths if p in train}.isdisjoint(
        {p.split("_")[0] for p in paths if p in val})

    expected = group_aware_train_val_split(paths, 0.34, 7)
    from taskcore.data.loader import train_val_split
    assert expected == train_val_split(len(paths), 0.34, 7)


def test_finalize_from_data_is_explicit_and_syncs_derived_fields():
    cfg = Config()
    result, counts = finalize_from_data(
        cfg, [0, 2, 7], per_sample_counts=[{0: 3}])
    assert result is cfg
    assert counts == [{0: 3}]
    assert cfg.data.label_values == [0, 2, 7]
    assert cfg.data.num_classes == 3


def test_actual_spacing_formula_matches_rounded_resample_shape():
    source = np.asarray([5.0, 2.0, 1.3])
    target = np.asarray([1.5, 1.1, 0.9])
    before = np.asarray([24, 17, 19])
    after = np.maximum(1, np.rint(before * source / target).astype(int))
    achieved = source * before / after
    assert not np.allclose(achieved, target, rtol=0.0, atol=_META_SPACING_ATOL)
    assert np.allclose(achieved * after / before, source)


def test_mixed_sampler_warns_when_gold_is_under_covered(caplog):
    with caplog.at_level(logging.WARNING):
        sampler = MixedBatchSampler(
            n_primary=100, n_secondary=4,
            gold_per_batch=1, coarse_per_batch=2)
    assert len(sampler) == 2
    assert "under-covered" in caplog.text
    assert "100.0%" not in caplog.text
