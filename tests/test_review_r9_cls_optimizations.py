"""clstask 审查落地项回归测试（R9）。

覆盖：
  1. 验证集每卷 patch 数 = cls.eval_patches_per_volume（与推理铺格上限同源）；
  2. ClsPatchDataset._load 走 seg memmap 快路径后与旧 zipfile 路径逐位一致
     （未压缩 + 压缩 npz 双路径）；
  3. derive_volume_targets：meta.label_counts 快路径与整卷 any() 回退一致；
  4. 卷级 MIL 真值与 patch 抽样解耦：抽样 patch 全为阴性时，含前景卷的
     vol target 仍为 1；
  5. 训练/验证在 prefetch_to_gpu=true（CPU 下 no-op）+ 验证 autocast 路径下
     可端到端跑通。

Run: pytest tests/test_review_r9_cls_optimizations.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_npz(path: Path, shape=(16, 48, 48), fg: bool = True,
              compressed: bool = False, with_counts: bool = False) -> None:
    rng = np.random.default_rng(hash(path.name) % (2 ** 31))
    img = (rng.standard_normal(shape) * 200.0).astype(np.int16)
    lbl = np.zeros(shape, dtype=np.uint8)
    n_fg = 0
    if fg:
        lbl[4:10, 8:24, 8:24] = 1
        n_fg = int((lbl == 1).sum())
    kw = dict(image=img, label=lbl)
    if with_counts:
        kw["meta"] = np.asarray(
            {"label_counts": {0: int(lbl.size - n_fg), 1: n_fg}},
            dtype=object)
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **kw)


@pytest.fixture()
def npz_dir(tmp_path: Path) -> Path:
    d = tmp_path / "npz"
    d.mkdir()
    for i in range(6):
        _make_npz(d / f"vol_{i:03d}.npz", fg=(i % 2 == 0))
    return d


# ---------------------------------------------------------------------------
# 1. 验证集每卷 patch 数 = eval_patches_per_volume
# ---------------------------------------------------------------------------
def test_val_loader_uses_eval_patches_per_volume(npz_dir: Path):
    from clstask.config import ClsConfig
    from clstask.data.loader import build_cls_dataloaders
    from segtask_v1.config import Config

    cfg = Config()
    cfg.data.npz_dir = str(npz_dir)
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [8, 32, 32]
    cfg.data.num_workers = 0
    cfg.data.samples_per_volume = 4
    cfg.data.val_ratio = 0.34
    cfg.sync()
    cls = ClsConfig(eval_patches_per_volume=5)
    train_loader, val_loader = build_cls_dataloaders(cfg, cls)
    assert train_loader.dataset.spv == 4
    assert val_loader.dataset.spv == 5, (
        "val samples_per_volume must equal cls.eval_patches_per_volume")


# ---------------------------------------------------------------------------
# 2. _load memmap 快路径 vs 旧 zipfile 路径逐位一致
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("compressed", [False, True])
def test_dataset_load_matches_legacy(tmp_path: Path, compressed: bool):
    from clstask.data.cls_dataset import ClsPatchDataset
    from segtask_v1.data.dataset import preprocess_image

    p = tmp_path / "vol.npz"
    _make_npz(p, fg=True, compressed=compressed)
    ds = ClsPatchDataset([str(p)], [8, 32, 32], num_classes=1,
                         label_source="mask", spatial_dims=3,
                         samples_per_volume=1, is_train=False)
    img, lbl = ds._load(str(p))
    with np.load(p, allow_pickle=True) as f:
        ref_img = preprocess_image(
            f["image"], ds.intensity_min, ds.intensity_max, ds.normalize,
            ds.global_mean, ds.global_std, inplace=False)
        ref_lbl = np.asarray(f["label"])
    np.testing.assert_array_equal(img, ref_img)
    np.testing.assert_array_equal(lbl, ref_lbl)
    assert lbl.flags.writeable


def test_dataset_load_missing_keys_fail_fast(tmp_path: Path):
    from clstask.data.cls_dataset import ClsPatchDataset

    p = tmp_path / "noimg.npz"
    np.savez(p, other=np.zeros((2, 2)))
    ds = ClsPatchDataset([str(p)], [8, 32, 32], num_classes=1,
                         label_source="mask", spatial_dims=3,
                         samples_per_volume=1, is_train=False)
    with pytest.raises(KeyError, match="image"):
        ds._load(str(p))


# ---------------------------------------------------------------------------
# 3. derive_volume_targets：meta 快路径与整卷回退一致
# ---------------------------------------------------------------------------
def test_derive_volume_targets(tmp_path: Path):
    from clstask.data.cls_dataset import derive_volume_targets

    p_fg = tmp_path / "fg.npz"
    p_bg = tmp_path / "bg.npz"
    p_fg_meta = tmp_path / "fg_meta.npz"
    _make_npz(p_fg, fg=True)
    _make_npz(p_bg, fg=False)
    _make_npz(p_fg_meta, fg=True, with_counts=True)
    t = derive_volume_targets(
        [str(p_fg), str(p_bg), str(p_fg_meta)], [1.0])
    assert t.shape == (3, 1)
    assert t.tolist() == [[1.0], [0.0], [1.0]]


# ---------------------------------------------------------------------------
# 4+5. 卷级真值解耦 + 端到端（prefetch 开关 / 验证 autocast 路径）
# ---------------------------------------------------------------------------
def _tiny_trainer(npz_dir: Path, out_dir: Path):
    from clstask.config import ClsConfig, validate_cls
    from clstask.data.loader import build_cls_dataloaders
    from clstask.models.factory import build_classifier
    from clstask.trainer.cls_trainer import ClsTrainer
    from segtask_v1.config import Config

    cfg = Config()
    cfg.data.npz_dir = str(npz_dir)
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [8, 32, 32]
    cfg.data.num_workers = 0
    cfg.data.batch_size = 2
    cfg.data.samples_per_volume = 2
    cfg.data.val_ratio = 0.34
    cfg.data.label_values = [0, 1]
    cfg.augment.enabled = False
    cfg.model.encoder_channels = [8, 16]
    cfg.model.encoder_blocks_per_stage = [1, 1]
    cfg.train.epochs = 1
    cfg.train.use_amp = False
    cfg.train.use_ema = False
    cfg.train.prefetch_to_gpu = True    # CPU 下 no-op，验证开关不破坏训练
    cfg.train.output_dir = str(out_dir)
    cfg.sync()
    cfg.validate()
    cls = ClsConfig(eval_patches_per_volume=2)
    validate_cls(cls, cfg)
    train_loader, val_loader = build_cls_dataloaders(cfg, cls)
    model = build_classifier(cfg, cls)
    return ClsTrainer(model, cfg, cls, train_loader, val_loader,
                      torch.device("cpu"))


def test_volume_targets_decoupled_from_patch_sampling(
        npz_dir: Path, tmp_path: Path):
    trainer = _tiny_trainer(npz_dir, tmp_path / "out")
    n_val = len(trainer.val_loader.dataset.paths)
    # 抽样 patch 全阴性（target 全 0），但整卷含前景的卷 vol target 仍为 1。
    n = n_val * 2
    probs = torch.rand(n, 1)
    targets = torch.zeros(n, 1)
    vols = torch.arange(n) // 2
    trainer._volume_metrics(probs, targets, vols)
    vt = trainer._val_vol_targets
    assert vt is not None and vt.shape == (n_val, 1)
    from clstask.data.cls_dataset import derive_volume_targets
    expect = derive_volume_targets(
        trainer.val_loader.dataset.paths, trainer.fg_values)
    assert torch.equal(vt, expect)
    assert vt.sum() > 0, "val split should contain at least one fg volume"


def test_end_to_end_fit(npz_dir: Path, tmp_path: Path):
    trainer = _tiny_trainer(npz_dir, tmp_path / "out")
    metrics = trainer.fit()
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["val_vol_auc"] <= 1.0
