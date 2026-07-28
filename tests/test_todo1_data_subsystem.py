"""TODO2 数据子系统批次回归（1-3 / 1-4 / 2-3）。

覆盖：
- 稳定哈希划分对样本增删的稳定性、seed/ratio 变化、group 隔离、兑底；
- per-case 归一化数值（zscore_volume / ct_fingerprint）与 cache/non-cache 一致；
- data_identifier 稳定性与对预处理参数的敏感性；
- 数据集指纹解析归一化参数；
- make_data 逐病例强度统计 + 数据集指纹落盘 + skip 幂等要求。
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from taskcore.config import Config

from taskcore.data.identity import (
    aggregate_dataset_fingerprint,
    compute_case_intensity_stats,
    compute_data_identifier,
    fingerprint_normalization_params,
)
from taskcore.data.loader import (
    _hash_fraction,
    grouped_hash_train_val_split,
    hash_train_val_split,
)
from taskcore.data.dataset import preprocess_image


# ---------------------------------------------------------------------------
# 1-4 稳定哈希划分
# ---------------------------------------------------------------------------
def test_hash_split_deterministic():
    keys = [f"case_{i:03d}" for i in range(40)]
    a = hash_train_val_split(keys, 0.2, 42)
    b = hash_train_val_split(keys, 0.2, 42)
    assert a == b


def test_hash_split_stable_under_add_remove():
    """增删样本不改变其余样本的 train/val 归属（根治指标污染）。"""
    keys = [f"case_{i:03d}" for i in range(50)]
    tr0, va0 = hash_train_val_split(keys, 0.25, 7)
    val_names0 = {keys[i] for i in va0}

    # 删掉若干 + 新增若干后，原有样本归属不变。
    keys2 = keys[5:] + [f"new_{i}" for i in range(8)]
    tr2, va2 = hash_train_val_split(keys2, 0.25, 7)
    val_names2 = {keys2[i] for i in va2}
    common = set(keys) & set(keys2)
    for name in common:
        assert (name in val_names0) == (name in val_names2), name


def test_hash_split_seed_and_ratio_change():
    keys = [f"c{i}" for i in range(60)]
    tr_a, va_a = hash_train_val_split(keys, 0.2, 1)
    tr_b, va_b = hash_train_val_split(keys, 0.2, 2)
    assert va_a != va_b  # seed 变则划分变
    _, va_hi = hash_train_val_split(keys, 0.5, 1)
    assert len(va_hi) > len(va_a)  # ratio 越大 val 越多


def test_hash_split_no_overlap_and_cover():
    keys = [f"c{i}" for i in range(30)]
    tr, va = hash_train_val_split(keys, 0.3, 99)
    assert set(tr).isdisjoint(va)
    assert sorted(tr + va) == list(range(30))


def test_hash_split_fallback_val_nonempty():
    """val_ratio 极小仍确定性保证 val 非空（n>1）。"""
    keys = [f"c{i}" for i in range(5)]
    tr, va = hash_train_val_split(keys, 1e-9, 3)
    assert len(va) == 1 and len(tr) == 4


def test_hash_split_fallback_train_nonempty():
    keys = [f"c{i}" for i in range(5)]
    tr, va = hash_train_val_split(keys, 1.0 - 1e-9, 3)
    assert len(tr) == 1 and len(va) == 4


def test_grouped_hash_split_isolation_and_stability():
    """同 group 整体同侧；增删其他组不改变既有组归属。"""
    paths = [f"P{p:02d}_T{t}.npz" for p in range(20) for t in range(2)]
    regex = r"^(P\d+)"
    tr, va = grouped_hash_train_val_split(paths, regex, 0.3, 5)
    # 每个 group 完整落在同一侧。
    def gid(i): return re.match(regex, paths[i]).group(1)
    train_g = {gid(i) for i in tr}
    val_g = {gid(i) for i in va}
    assert train_g.isdisjoint(val_g)
    # 删掉一个 group 后其余组归属不变。
    keep = [p for p in paths if not p.startswith("P00_")]
    tr2, va2 = grouped_hash_train_val_split(keep, regex, 0.3, 5)
    val_g2 = {re.match(regex, keep[i]).group(1) for i in va2}
    for g in (val_g | train_g) - {"P00"}:
        assert (g in val_g) == (g in val_g2), g


def test_hash_fraction_range():
    for i in range(100):
        f = _hash_fraction(f"k{i}", 42)
        assert 0.0 <= f < 1.0


# ---------------------------------------------------------------------------
# 2-3 per-case 归一化
# ---------------------------------------------------------------------------
def _vol():
    rng = np.random.RandomState(0)
    return rng.uniform(-200, 300, size=(4, 8, 8)).astype(np.float32)


def test_zscore_volume_normalization():
    vol = _vol()
    out = preprocess_image(vol, -1024, 1024, "zscore_volume")
    assert abs(float(out.mean())) < 1e-4
    assert abs(float(out.std()) - 1.0) < 1e-4


def test_zscore_volume_constant_volume_zeroed():
    vol = np.full((2, 4, 4), 5.0, dtype=np.float32)
    out = preprocess_image(vol, -1024, 1024, "zscore_volume")
    assert np.allclose(out, 0.0)


def test_ct_fingerprint_matches_zscore_algebra():
    vol = _vol()
    a = preprocess_image(vol, -100, 200, "ct_fingerprint", 50.0, 30.0)
    b = preprocess_image(vol, -100, 200, "zscore", 50.0, 30.0)
    assert np.allclose(a, b)


def test_cache_int16_parity_zscore_volume():
    """cache_dtype=int16 重算路径与 fp32 直算数值一致（口径不变）。"""
    vol = _vol().astype(np.int16).astype(np.float32)
    fp32 = preprocess_image(vol, -1024, 1024, "zscore_volume", inplace=False)
    # 模拟 int16 缓存后重跑：从原始体素再算一次。
    recomputed = preprocess_image(
        vol.astype(np.int16).astype(np.float32),
        -1024, 1024, "zscore_volume", inplace=False)
    assert np.allclose(fp32, recomputed)


# ---------------------------------------------------------------------------
# per-case 强度统计 + 数据集指纹
# ---------------------------------------------------------------------------
def test_case_intensity_stats_foreground_only():
    img = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    lab = np.zeros((3, 3, 3), dtype=np.int16)
    lab[0, 0, 0] = 1  # value 0
    lab[2, 2, 2] = 1  # value 26
    stats = compute_case_intensity_stats(img, lab)
    assert stats["n_fg_voxels"] == 2
    assert stats["min"] == 0.0 and stats["max"] == 26.0
    assert stats["mean"] == 13.0


def test_case_intensity_stats_empty_label():
    img = np.ones((2, 2, 2), dtype=np.float32)
    lab = np.zeros((2, 2, 2), dtype=np.int16)
    stats = compute_case_intensity_stats(img, lab)
    assert stats["n_fg_voxels"] == 0
    assert stats["mean"] is None
    assert stats["fg_sample"].size == 0


def test_dataset_fingerprint_aggregate_and_params():
    rng = np.random.RandomState(1)
    samples = [rng.normal(100.0, 20.0, size=500).astype(np.float32)
               for _ in range(4)]
    fp = aggregate_dataset_fingerprint(samples, n_cases=4)
    assert fp is not None and fp["n_cases"] == 4
    params = fingerprint_normalization_params(fp)
    assert params["intensity_min"] < params["intensity_max"]
    assert params["global_std"] > 0


def test_dataset_fingerprint_all_empty_none():
    assert aggregate_dataset_fingerprint([], n_cases=0) is None
    assert aggregate_dataset_fingerprint(
        [np.zeros((0,), np.float32)], n_cases=2) is None


def test_fingerprint_params_missing_key_raises():
    with pytest.raises(ValueError, match="missing key"):
        fingerprint_normalization_params({"fg_mean": 1.0})


def test_fingerprint_params_degenerate_std_raises():
    fp = {"fg_p00_5": 0.0, "fg_p99_5": 1.0, "fg_mean": 0.5, "fg_std": 0.0}
    with pytest.raises(ValueError, match="fg_std"):
        fingerprint_normalization_params(fp)


# ---------------------------------------------------------------------------
# 1-3 data_identifier
# ---------------------------------------------------------------------------
def _ident(**kw):
    base = dict(spacing_normalization=True, target_spacing=[1.0, 1.0, 1.0],
                label_values=[0, 1], fg_subsample=50000,
                has_bbox=False, has_rw=False)
    base.update(kw)
    return compute_data_identifier(**base)


def test_data_identifier_stable():
    assert _ident() == _ident()
    assert _ident().startswith("dsid-")


def test_data_identifier_sensitive_to_params():
    base = _ident()
    assert _ident(target_spacing=[1.0, 1.0, 2.0]) != base
    assert _ident(label_values=[0, 1, 2]) != base
    assert _ident(spacing_normalization=False) != base
    assert _ident(fg_subsample=10000) != base
    assert _ident(has_rw=True) != base
    assert _ident(has_bbox=True) != base


def test_data_identifier_spacing_float_noise_stable():
    """target_spacing 微小浮点噪声不抖动标识（量化到 1e-4）。"""
    a = _ident(target_spacing=[1.0, 1.0, 1.0])
    b = _ident(target_spacing=[1.00000001, 1.0, 1.0])
    assert a == b


# ---------------------------------------------------------------------------
# 端到端：make_data → manifest → build_dataloaders 消费
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(out_dir: Path, n_volumes: int = 4,
                            shape=(12, 32, 32), seed: int = 0):
    nib = pytest.importorskip("nibabel")
    rng = np.random.RandomState(seed)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    for i in range(n_volumes):
        img = (rng.randn(*shape) * 50.0 + 40.0).astype(np.float32)
        nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), affine),
                 str(img_dir / f"vol_{i:02d}.nii.gz"))
        lbl = np.zeros(shape, dtype=np.int16)
        lbl[4:8, 8:20, 8:20] = 1
        nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), affine),
                 str(lbl_dir / f"vol_{i:02d}.nii.gz"))
    return str(img_dir), str(lbl_dir)


def _e2e_cfg(td: Path) -> Config:
    img_dir, lbl_dir = _make_synthetic_dataset(td)
    cfg = Config()
    cfg.data.image_dir = img_dir
    cfg.data.label_dir = lbl_dir
    cfg.data.npz_dir = str(td / "npz")
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [8, 16, 16]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.multi_res_scales = [1.0]
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.data.samples_per_volume = 1
    cfg.data.val_ratio = 0.25
    cfg.augment.enabled = False
    cfg.sync()
    cfg.validate()
    return cfg


def test_make_data_manifest_has_fingerprint_and_identifier():
    from taskcore.data.make_data import prepare_dataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _e2e_cfg(td)
        prepare_dataset(cfg, out_dir=cfg.data.npz_dir, workers=0)

        manifest = json.loads(
            (Path(cfg.data.npz_dir) / "_manifest.json").read_text(
                encoding="utf-8"))
        assert str(manifest["data_identifier"]).startswith("dsid-")
        fp = manifest["dataset_fingerprint"]
        assert fp["n_cases"] == 4 and fp["n_samples"] > 0
        assert fp["fg_std"] > 0
        # 每个 npz meta 带 per-case 统计。
        npz = sorted(Path(cfg.data.npz_dir).glob("*.npz"))[0]
        with np.load(npz, allow_pickle=True) as zf:
            meta = zf["meta"].item()
        assert meta["intensity_stats"]["n_fg_voxels"] > 0
        assert len(meta["fg_intensity_sample"]) > 0

        # skip 幂等重跑：指纹覆盖 skip 病例、内容一致。
        prepare_dataset(cfg, out_dir=cfg.data.npz_dir, workers=0)
        manifest2 = json.loads(
            (Path(cfg.data.npz_dir) / "_manifest.json").read_text(
                encoding="utf-8"))
        assert manifest2["data_identifier"] == manifest["data_identifier"]
        fp2 = manifest2["dataset_fingerprint"]
        assert fp2["n_cases"] == 4
        assert abs(fp2["fg_mean"] - fp["fg_mean"]) < 1e-6


def test_loader_ct_fingerprint_resolution_and_split_manifest():
    from taskcore.data.loader import build_dataloaders
    from taskcore.data.make_data import prepare_dataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _e2e_cfg(td)
        prepare_dataset(cfg, out_dir=cfg.data.npz_dir, workers=0)

        cfg.data.normalize = "ct_fingerprint"
        build_dataloaders(cfg)
        # 归一化参数已由指纹回写。
        assert cfg.data.global_std not in (0.0, 1.0)
        assert cfg.data.intensity_min < cfg.data.intensity_max
        assert cfg.data.data_identifier.startswith("dsid-")
        # 默认 split manifest 落在 npz_dir 下且记录 split_method。
        sm = json.loads(
            (Path(cfg.data.npz_dir) / "_split_manifest.json").read_text(
                encoding="utf-8"))
        assert sm["split_method"] == "hash"
        assert sm["train"] and sm["val"]
        assert set(sm["train"]).isdisjoint(sm["val"])


def test_loader_rejects_mismatched_package():
    from taskcore.data.loader import build_dataloaders
    from taskcore.data.make_data import prepare_dataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _e2e_cfg(td)
        prepare_dataset(cfg, out_dir=cfg.data.npz_dir, workers=0)

        # 篡改 manifest 模拟"另一口径烘的包"。
        p = Path(cfg.data.npz_dir) / "_manifest.json"
        manifest = json.loads(p.read_text(encoding="utf-8"))
        manifest["label_values"] = [0, 1, 2]
        p.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="label_values"):
            build_dataloaders(cfg)


def test_loader_ct_fingerprint_requires_manifest():
    from taskcore.data.loader import build_dataloaders
    from taskcore.data.make_data import prepare_dataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _e2e_cfg(td)
        prepare_dataset(cfg, out_dir=cfg.data.npz_dir, workers=0)
        (Path(cfg.data.npz_dir) / "_manifest.json").unlink()

        cfg.data.normalize = "ct_fingerprint"
        with pytest.raises(ValueError, match="ct_fingerprint"):
            build_dataloaders(cfg)


# ---------------------------------------------------------------------------
# 推理侧 ct_fingerprint 参数采纳（checkpoint → 推理 cfg）
# ---------------------------------------------------------------------------
def test_predictor_adopts_fingerprint_normalization():
    from segtask_v1.predictor.io import _adopt_fingerprint_normalization

    train_cfg = Config()
    train_cfg.data.normalize = "ct_fingerprint"
    train_cfg.data.intensity_min = -57.0
    train_cfg.data.intensity_max = 303.0
    train_cfg.data.global_mean = 99.5
    train_cfg.data.global_std = 41.2
    ckpt = {"config": train_cfg}

    infer_cfg = Config()
    infer_cfg.data.normalize = "ct_fingerprint"
    _adopt_fingerprint_normalization(ckpt, infer_cfg, "ckpt.pth")
    assert infer_cfg.data.intensity_min == -57.0
    assert infer_cfg.data.intensity_max == 303.0
    assert infer_cfg.data.global_mean == 99.5
    assert infer_cfg.data.global_std == 41.2


def test_predictor_fingerprint_requires_ckpt_config():
    from segtask_v1.predictor.io import _adopt_fingerprint_normalization

    infer_cfg = Config()
    infer_cfg.data.normalize = "ct_fingerprint"
    with pytest.raises(RuntimeError, match="ct_fingerprint"):
        _adopt_fingerprint_normalization({}, infer_cfg, "ckpt.pth")


def test_predictor_no_adoption_when_modes_differ():
    from segtask_v1.predictor.io import _adopt_fingerprint_normalization

    train_cfg = Config()
    train_cfg.data.normalize = "minmax"
    infer_cfg = Config()
    infer_cfg.data.normalize = "ct_fingerprint"
    before = float(infer_cfg.data.global_mean)
    _adopt_fingerprint_normalization(
        {"config": train_cfg}, infer_cfg, "ckpt.pth")
    assert infer_cfg.data.global_mean == before  # 不采纳，交给镜像比对报错
