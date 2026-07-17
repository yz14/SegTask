"""审查报告第二批修复的回归测试。

覆盖：
1. MixedBatchSampler DDP rank 切分：各 rank 等长、batch 不相交、并集等于
   全局序列前缀；set_epoch 对齐/重洗；单卡行为向后兼容。
2. make_data manifest 持久化 target_spacing + Predictor 从 manifest 回读。
3. make_data 物理几何（spacing/origin/direction）一致性 fail-fast。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taskcore.data.mixed_sampler import MixedBatchSampler  # noqa: E402


# ---------------------------------------------------------------------------
# 1. MixedBatchSampler DDP sharding
# ---------------------------------------------------------------------------

def _epoch_batches(sampler: MixedBatchSampler, epoch: int) -> list:
    sampler.set_epoch(epoch)
    return list(sampler)


def test_mixed_sampler_single_rank_matches_legacy_length_and_quota():
    s = MixedBatchSampler(5, 12, 1, 2, seed=42)
    batches = list(s)
    assert len(s) == 12 // 2
    assert len(batches) == len(s)
    for b in batches:
        prim = [i for i in b if i < 5]
        sec = [i for i in b if i >= 5]
        assert len(prim) == 1 and len(sec) == 2


def test_mixed_sampler_ddp_ranks_are_disjoint_equal_length_and_cover_global():
    n_primary, n_secondary = 8, 24
    gold_pb, coarse_pb = 1, 2
    world_size = 3
    global_s = MixedBatchSampler(
        n_primary, n_secondary, gold_pb, coarse_pb, seed=7)
    global_batches = _epoch_batches(global_s, epoch=0)

    per_rank = []
    for rank in range(world_size):
        s = MixedBatchSampler(
            n_primary, n_secondary, gold_pb, coarse_pb, seed=7,
            rank=rank, world_size=world_size)
        batches = _epoch_batches(s, epoch=0)
        assert len(batches) == len(s)
        assert len(s) == len(global_batches) // world_size
        per_rank.append(batches)

    # 各 rank 拿到的正是全局序列的 strided 切片（同 seed+epoch 全局排列一致）。
    for rank, batches in enumerate(per_rank):
        for i, b in enumerate(batches):
            assert b == global_batches[rank + i * world_size]

    # batch 级不相交（比较 batch 内容元组）。
    seen = set()
    for batches in per_rank:
        for b in batches:
            key = tuple(b)
            assert key not in seen
            seen.add(key)


def test_mixed_sampler_ddp_secondary_coverage_no_duplicates_within_epoch():
    """一个 epoch 内所有 rank 的粗标样本互不重复（粗标不跨 rank 重发）。"""
    world_size = 2
    n_primary, n_secondary = 4, 10
    all_sec = []
    for rank in range(world_size):
        s = MixedBatchSampler(
            n_primary, n_secondary, 1, 1, seed=3,
            rank=rank, world_size=world_size)
        for b in _epoch_batches(s, epoch=5):
            all_sec.extend(i for i in b if i >= n_primary)
    assert len(all_sec) == len(set(all_sec))


def test_mixed_sampler_set_epoch_reshuffles_and_aligns():
    kwargs = dict(n_primary=6, n_secondary=12, gold_per_batch=1,
                  coarse_per_batch=2, seed=11)
    a = MixedBatchSampler(**kwargs)
    b = MixedBatchSampler(**kwargs)
    assert _epoch_batches(a, 0) == _epoch_batches(b, 0)
    assert _epoch_batches(a, 1) == _epoch_batches(b, 1)
    assert _epoch_batches(a, 0) != _epoch_batches(a, 1)


def test_mixed_sampler_rejects_fewer_global_batches_than_ranks():
    with pytest.raises(ValueError):
        MixedBatchSampler(4, 2, 1, 2, seed=0, rank=0, world_size=2)


def test_trainer_recognizes_batch_sampler_set_epoch():
    """trainer 的采样器识别按 set_epoch 协议鸭子识别 batch_sampler。"""
    import inspect

    from taskcore.engine.base_trainer import BaseTrainer

    # 识别逻辑已下沉公共层：BaseTrainer._setup_train_sampler。
    src = inspect.getsource(BaseTrainer._setup_train_sampler)
    assert "batch_sampler" in src and "set_epoch" in src


# ---------------------------------------------------------------------------
# 2. manifest target_spacing 持久化 + Predictor 回读
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, array: np.ndarray, spacing=None,
                 origin=None, direction=None) -> None:
    img = sitk.GetImageFromArray(array)
    if spacing is not None:
        img.SetSpacing(spacing)      # (sx, sy, sz)
    if origin is not None:
        img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, str(path))


def _make_pair_dirs(root: Path, n: int = 2, label_geometry=None):
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx in range(n):
        image = rng.integers(-100, 100, size=(6, 12, 12), dtype=np.int16)
        label = np.zeros((6, 12, 12), dtype=np.int16)
        label[2:4, 4:8, 4:8] = 1
        stem = f"case_{idx:02d}"
        _write_nifti(image_dir / f"{stem}.nii.gz", image,
                     spacing=(1.0, 1.0, 2.0))
        geo = label_geometry or {}
        _write_nifti(label_dir / f"{stem}.nii.gz", label,
                     spacing=geo.get("spacing", (1.0, 1.0, 2.0)),
                     origin=geo.get("origin"),
                     direction=geo.get("direction"))
    return image_dir, label_dir


def _make_cfg(image_dir: Path, label_dir: Path, npz_dir: Path):
    from taskcore.config.core import Config
    cfg = Config()
    cfg.data.image_dir = str(image_dir)
    cfg.data.label_dir = str(label_dir)
    cfg.data.npz_dir = str(npz_dir)
    cfg.data.image_suffix = ".nii.gz"
    cfg.data.label_suffix = ".nii.gz"
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    return cfg


def test_manifest_records_resolved_target_spacing():
    from taskcore.data.make_data import prepare_dataset
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image_dir, label_dir = _make_pair_dirs(root)
        npz_dir = root / "npz"
        cfg = _make_cfg(image_dir, label_dir, npz_dir)
        cfg.data.spacing_normalization = True
        cfg.data.target_spacing = None   # 自动中位数
        counters = prepare_dataset(cfg, str(npz_dir), workers=0)
        assert counters["failed"] == 0
        with open(npz_dir / "_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["spacing_normalization"] is True
        # spacing (sx,sy,sz)=(1,1,2) → numpy (D,H,W)=(2,1,1) 的中位数。
        assert manifest["target_spacing"] == [2.0, 1.0, 1.0]


def test_manifest_target_spacing_none_when_normalization_off():
    from taskcore.data.make_data import prepare_dataset
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image_dir, label_dir = _make_pair_dirs(root)
        npz_dir = root / "npz"
        cfg = _make_cfg(image_dir, label_dir, npz_dir)
        counters = prepare_dataset(cfg, str(npz_dir), workers=0)
        assert counters["failed"] == 0
        with open(npz_dir / "_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["spacing_normalization"] is False
        assert manifest["target_spacing"] is None


def test_predictor_manifest_target_spacing_helper():
    from segtask_v1.predictor.predictor import _manifest_target_spacing
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # 目录/文件缺失 → None。
        assert _manifest_target_spacing("") is None
        assert _manifest_target_spacing(str(root)) is None
        # 合法字段 → 回读。
        (root / "_manifest.json").write_text(
            json.dumps({"target_spacing": [2.0, 1.0, 1.0]}), encoding="utf-8")
        assert _manifest_target_spacing(str(root)) == [2.0, 1.0, 1.0]
        # 非法字段 → None。
        (root / "_manifest.json").write_text(
            json.dumps({"target_spacing": [0.0, 1.0]}), encoding="utf-8")
        assert _manifest_target_spacing(str(root)) is None


# ---------------------------------------------------------------------------
# 3. 物理几何一致性校验
# ---------------------------------------------------------------------------

def test_geometry_check_passes_for_coregistered_pair():
    from taskcore.data.make_data import _check_physical_geometry
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image_dir, label_dir = _make_pair_dirs(root, n=1)
        img = str(next(image_dir.glob("*.nii.gz")))
        lbl = str(next(label_dir.glob("*.nii.gz")))
        _check_physical_geometry("p0", img, [("label", lbl), ("bbox", None)])


@pytest.mark.parametrize("geometry, field", [
    ({"spacing": (1.0, 1.0, 2.5)}, "spacing"),
    ({"origin": (5.0, 0.0, 0.0)}, "origin"),
    ({"direction": (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
     "direction"),
])
def test_geometry_check_rejects_mismatch(geometry, field):
    from taskcore.data.make_data import _check_physical_geometry
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image_dir, label_dir = _make_pair_dirs(
            root, n=1, label_geometry=geometry)
        img = str(next(image_dir.glob("*.nii.gz")))
        lbl = str(next(label_dir.glob("*.nii.gz")))
        with pytest.raises(ValueError, match=field):
            _check_physical_geometry("p0", img, [("label", lbl)])


def test_prepare_one_fails_fast_on_geometry_mismatch():
    from taskcore.data.make_data import prepare_one
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image_dir, label_dir = _make_pair_dirs(
            root, n=1, label_geometry={"origin": (7.0, 0.0, 0.0)})
        img = str(next(image_dir.glob("*.nii.gz")))
        lbl = str(next(label_dir.glob("*.nii.gz")))
        with pytest.raises(ValueError, match="physical"):
            prepare_one(
                pid="p0", image_path=img, label_path=lbl,
                bbox_path=None, rw_path=None,
                out_path=str(root / "p0.npz"), label_values=[0, 1])
