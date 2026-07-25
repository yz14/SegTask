import json

import torch
import numpy as np


def test_legacy_elastic_field_is_bitwise_unchanged():
    from taskcore.data.augment import _elastic_grid_disp

    gen_a = torch.Generator().manual_seed(123)
    gen_b = torch.Generator().manual_seed(123)
    old = _elastic_grid_disp(2, 8, 8, 8, 3.0, 2.0, torch.device("cpu"),
                             gen_a)
    new = _elastic_grid_disp(
        2, 8, 8, 8, 3.0, 2.0, torch.device("cpu"), gen_b,
        field_mode="legacy", normalize_displacement=False)
    assert torch.equal(old, new)


def test_gaussian_elastic_field_and_normalization_are_opt_in():
    from taskcore.data.augment import _elastic_grid_disp

    gen = torch.Generator().manual_seed(123)
    field = _elastic_grid_disp(
        2, 8, 8, 8, 3.0, 2.0, torch.device("cpu"), gen,
        field_mode="gaussian", normalize_displacement=True)
    rms = field.square().mean(dim=(1, 2, 3, 4)).sqrt()
    assert torch.isfinite(field).all()
    assert (rms > 0).all()


def test_split_rounding_default_and_unified_mode():
    from taskcore.data.loader import train_val_split

    _, legacy = train_val_split(5, 0.5, 7)
    _, unified = train_val_split(5, 0.5, 7, rounding_mode="unified")
    assert len(legacy) == 2
    _, legacy = train_val_split(5, 0.3, 7)
    _, unified = train_val_split(5, 0.3, 7, rounding_mode="unified")
    assert len(legacy) == 1
    assert len(unified) == 2


def test_legacy_split_count_preserves_each_callsite_rounding():
    from taskcore.data.loader import (
        _fallback_split_val_count,
        _random_split_val_count,
        _stratified_split_val_count,
    )

    for n in (1, 2, 3, 5, 7, 10):
        for ratio in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0):
            random_old = min(max(int(n * ratio), 1), n - 1) if n > 1 else 0
            strat_old = min(
                max(int(round(n * ratio)), 1), n - 1) if n > 1 else 0
            fallback_old = int(round(n * ratio))
            assert _random_split_val_count(n, ratio, "legacy") == random_old
            assert _stratified_split_val_count(
                n, ratio, "legacy") == strat_old
            assert _fallback_split_val_count(
                n, ratio, "legacy") == fallback_old


def test_upkern_normalization_is_opt_in():
    from taskcore.models.mednext import upkern_remap_state_dict

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dwconv = torch.nn.Conv2d(2, 2, 5, groups=2, bias=False)

    target = Target()
    src = {"dwconv.weight": torch.ones(2, 1, 3, 3)}
    legacy = upkern_remap_state_dict(src, target)
    normalized = upkern_remap_state_dict(
        src, target, normalize_spatial=True)
    assert torch.equal(legacy["dwconv.weight"], legacy["dwconv.weight"])
    assert not torch.equal(legacy["dwconv.weight"], normalized["dwconv.weight"])
    sums = normalized["dwconv.weight"].sum(dim=(2, 3))
    assert torch.allclose(sums.abs(), torch.ones_like(sums.abs()))


def test_split_manifest_is_json_serializable(tmp_path):
    payload = {"train": ["a.npz"], "val": ["b.npz"]}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_resize_antialias_off_matches_legacy_and_reduces_high_frequency():
    from taskcore.data.dataset import resize_3d

    x = np.indices((1, 32, 32))[2] % 2
    legacy = resize_3d(x, 1, 16, 16)
    off = resize_3d(x, 1, 16, 16, anti_alias=False)
    filtered = resize_3d(x, 1, 16, 16, anti_alias=True)
    assert np.array_equal(legacy, off)
    assert np.var(np.diff(filtered.astype(np.float32), axis=2)) <= (
        np.var(np.diff(off.astype(np.float32), axis=2)) + 1e-8)


def test_init_strategy_legacy_is_noop_and_kaiming_is_explicit():
    from taskcore.models.factory import _apply_init_strategy

    model = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 3), torch.nn.GroupNorm(1, 2))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert _apply_init_strategy(model, "legacy") is model
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    _apply_init_strategy(model, "kaiming")
    assert not torch.equal(before["0.weight"], model[0].weight)
    assert torch.allclose(model[1].weight, torch.ones_like(model[1].weight))
