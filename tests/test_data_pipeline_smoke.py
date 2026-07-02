"""Synthetic end-to-end smoke for the gentask data pipeline.

Exercises the real NIfTI -> make_data -> npz -> build_dataloaders path on tiny
volumes so later data-layer work has a safety net.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gentask.config import Config  # noqa: E402
from gentask.data.loader import build_dataloaders  # noqa: E402
from gentask.data.make_data import prepare_dataset  # noqa: E402


def _write_nifti(path: Path, array: np.ndarray) -> None:
    img = sitk.GetImageFromArray(array)
    sitk.WriteImage(img, str(path))


def _make_synthetic_nifti_pair_dirs(root: Path, n_volumes: int = 4) -> tuple[Path, Path]:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1234)
    for idx in range(n_volumes):
        image = rng.integers(-120, 120, size=(8, 16, 16), dtype=np.int16)
        label = np.zeros((8, 16, 16), dtype=np.int16)
        label[2:5, 4:8, 5:9] = 1
        if idx % 2 == 1:
            label[1:3, 9:13, 2:6] = 1

        stem = f"case_{idx:02d}"
        _write_nifti(image_dir / f"{stem}.nii.gz", image)
        _write_nifti(label_dir / f"{stem}.nii.gz", label)

    return image_dir, label_dir


def _base_cfg(patch_mode: str, image_dir: Path, label_dir: Path, npz_dir: Path) -> Config:
    cfg = Config()
    cfg.data.image_dir = str(image_dir)
    cfg.data.label_dir = str(label_dir)
    cfg.data.npz_dir = str(npz_dir)
    cfg.data.npz_auto_build = False
    cfg.data.image_suffix = ".nii.gz"
    cfg.data.label_suffix = ".nii.gz"
    cfg.data.patch_mode = patch_mode
    cfg.data.patch_size = [4, 8, 8]
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.val_ratio = 0.25
    cfg.data.stratified_split = False
    cfg.data.samples_per_volume = 1
    cfg.data.foreground_oversample_ratio = 1.0
    cfg.data.cache_mode = "none"
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2

    cfg.model.encoder_channels = [16, 32, 48]
    cfg.task.type = "generation"
    cfg.task.algorithm = "regression"
    cfg.task.degradation = "superres"
    cfg.task.out_channels = 1
    cfg.task.sr_scale = 2
    cfg.sync()
    cfg.validate()
    return cfg


def _exercise_pipeline(patch_mode: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image_dir, label_dir = _make_synthetic_nifti_pair_dirs(root)
        npz_dir = root / f"{patch_mode}_npz"
        cfg = _base_cfg(patch_mode, image_dir, label_dir, npz_dir)

        counters = prepare_dataset(
            cfg=cfg,
            out_dir=str(npz_dir),
            workers=0,
            overwrite=False,
            limit=0)
        assert counters["written"] == 4, counters
        assert counters["failed"] == 0, counters
        assert len(list(npz_dir.glob("*.npz"))) == 4

        train_loader, val_loader = build_dataloaders(cfg)
        assert len(train_loader) > 0, "train loader must be non-empty"
        assert len(val_loader) > 0, "val loader must be non-empty"

        batch = next(iter(train_loader))
        assert set(batch) >= {"image", "label"}
        assert batch["image"].dtype.is_floating_point
        assert batch["image"].ndim == 5
        assert batch["label"].shape == batch["image"].shape
        assert batch["image"].shape[0] == 2
        assert batch["image"].shape[1] == 1
        assert tuple(batch["image"].shape[2:]) == tuple(cfg.data.patch_size)
        return {
            "counters": counters,
            "batch_shape": tuple(batch["image"].shape),
            "model_spatial_dims": cfg.model.spatial_dims,
            "model_in_channels": cfg.model.in_channels,
        }


def test_data_pipeline_z_axis_npz_roundtrip():
    res = _exercise_pipeline("z_axis")
    assert res["model_spatial_dims"] == 3
    assert res["model_in_channels"] == 1
    assert res["batch_shape"] == (2, 1, 4, 8, 8)


def test_data_pipeline_2_5d_npz_roundtrip():
    res = _exercise_pipeline("2_5d")
    assert res["model_spatial_dims"] == 2
    assert res["model_in_channels"] == 4
    assert res["batch_shape"] == (2, 1, 4, 8, 8)


def main() -> int:
    tests = [
        test_data_pipeline_z_axis_npz_roundtrip,
        test_data_pipeline_2_5d_npz_roundtrip,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"[ok] {t.__name__}")
        except Exception as exc:  # pragma: no cover
            failures += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        return 1
    print("All data pipeline smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
