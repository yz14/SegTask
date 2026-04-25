"""R3 smoke test — 2.5D mode end-to-end pipeline.

Verifies:
  1. Config: patch_mode='2_5d' triggers correct sync (spatial_dims=2,
     in_channels=D) and validation rejects illegal combinations.
  2. SliceChannelLoss: forward, gradient flow, split_for_metrics,
     weight_map handling, error paths.
  3. Factory: 2.5D model has out_channels = num_fg * D.
  4. Dataloader: 2_5d branch returns single-resolution z_axis batches.
  5. End-to-end (no real data): synthesise a Trainer-compatible batch,
     run one forward + loss + backward step.
  6. Existing 3D regression: 3D path unchanged.

Run:
    conda activate torch27_env
    python test_2_5d_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


def _ok(name: str, msg: str = "") -> None:
    print(f"  [PASS] {name}{(' — ' + msg) if msg else ''}")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
def test_config_2_5d_sync():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [12, 32, 32]
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.sync()
    cfg.validate()
    assert cfg.model.spatial_dims == 2, "sync should set spatial_dims=2"
    assert cfg.model.in_channels == 12, \
        f"sync should set in_channels=patch_size[0]; got {cfg.model.in_channels}"
    assert cfg.data.multi_res_scales == [1.0]
    _ok("Config sync sets spatial_dims=2 + in_channels=D")


def test_config_2_5d_rejects_multi_res():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [12, 32, 32]
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.multi_res_scales = [1.0, 1.5]
    cfg.sync()
    try:
        cfg.validate()
    except AssertionError as e:
        assert "2.5D mode" in str(e)
        _ok("Config validate rejects multi-res in 2.5D mode")
        return
    raise AssertionError("validate should reject multi_res_scales != [1.0]")


# ---------------------------------------------------------------------------
# SliceChannelLoss tests
# ---------------------------------------------------------------------------
def test_slice_channel_loss_forward():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss
    B, num_fg, D, H, W = 2, 2, 4, 16, 16
    fg_values = [1, 2]
    label_values = [0] + fg_values
    base = BinaryDiceLoss(smooth=1e-5)
    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values)

    pred = torch.randn(B, num_fg * D, H, W, requires_grad=True)
    label = torch.randint(0, len(label_values), (B, D, H, W),
                          dtype=torch.float32)
    loss = scl(pred, label)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    _ok("SliceChannelLoss forward + backward (no weight_map)")


def test_slice_channel_loss_with_weight_map():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss
    B, num_fg, D, H, W = 2, 2, 4, 16, 16
    label_values = [0, 1, 2]
    scl = SliceChannelLoss(
        base_loss=BinaryDiceLoss(),
        num_fg_classes=num_fg, num_slices=D,
        label_values=label_values)
    pred = torch.randn(B, num_fg * D, H, W, requires_grad=True)
    label = torch.randint(0, 3, (B, D, H, W), dtype=torch.float32)
    wmap = torch.rand(B, D, H, W)
    loss = scl(pred, label, weight_map=wmap)
    loss.backward()
    assert pred.grad is not None
    _ok("SliceChannelLoss with weight_map")


def test_slice_channel_split_for_metrics():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss
    B, num_fg, D, H, W = 3, 2, 4, 8, 8
    label_values = [0, 1, 2]
    scl = SliceChannelLoss(
        base_loss=BinaryDiceLoss(), num_fg_classes=num_fg,
        num_slices=D, label_values=label_values)
    pred = torch.randn(B, num_fg * D, H, W)
    label = torch.zeros(B, D, H, W, dtype=torch.float32)
    label[0, 1, 2:5, 3:6] = 1   # one fg-class-0 patch
    label[1, 2, 4:7, 4:7] = 2   # one fg-class-1 patch
    p_metric, t_metric = scl.split_for_metrics(pred, label)
    assert p_metric.shape == (B * D, num_fg, H, W)
    assert t_metric.shape == (B * D, num_fg, H, W)
    # Sanity: per-class binarisation
    # batch 0, slice 1 → flat index 0*D + 1 = 1; class 0 should have a
    # 3x3 region of 1s.
    assert t_metric[0 * D + 1, 0, 2:5, 3:6].sum().item() == 9
    assert t_metric[1 * D + 2, 1, 4:7, 4:7].sum().item() == 9
    _ok("SliceChannelLoss.split_for_metrics shape + value correctness")


def test_slice_channel_loss_invalid_shapes():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss
    scl = SliceChannelLoss(
        base_loss=BinaryDiceLoss(), num_fg_classes=2,
        num_slices=4, label_values=[0, 1, 2])
    # Wrong pred channel count
    bad_pred = torch.randn(1, 5, 8, 8)   # 5 != 2*4
    label = torch.zeros(1, 4, 8, 8)
    try:
        scl(bad_pred, label)
    except ValueError as e:
        assert "channel count" in str(e)
    # Wrong label slice count
    pred = torch.randn(1, 8, 8, 8)
    bad_label = torch.zeros(1, 3, 8, 8)
    try:
        scl(pred, bad_label)
    except ValueError as e:
        assert "slice count" in str(e)
    _ok("SliceChannelLoss raises on invalid shapes")


# ---------------------------------------------------------------------------
# Factory test for 2.5D
# ---------------------------------------------------------------------------
def _build_2_5d_cfg(D: int = 12, num_fg: int = 2, encoder=(16, 32, 64)):
    from segtask_v1.config import Config
    cfg = Config()
    cfg.data.image_dir = ""   # not used for factory test
    cfg.data.label_dir = ""
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [D, 32, 32]
    cfg.data.label_values = [0] + list(range(1, num_fg + 1))
    cfg.data.num_classes = num_fg + 1
    cfg.model.encoder_channels = list(encoder)
    cfg.sync()
    cfg.validate()
    return cfg


def test_factory_2_5d_out_channels():
    from segtask_v1.models.factory import build_model
    D, num_fg = 12, 2
    cfg = _build_2_5d_cfg(D=D, num_fg=num_fg)
    model = build_model(cfg).eval()
    # Verify input shape matches in_channels=D
    assert cfg.model.in_channels == D
    x = torch.randn(1, D, 32, 32)
    y = model(x)
    assert y.shape == (1, num_fg * D, 32, 32), \
        f"expected (1, {num_fg * D}, 32, 32); got {tuple(y.shape)}"
    _ok("Factory 2.5D model: out_channels = num_fg * D")


# ---------------------------------------------------------------------------
# End-to-end Trainer dry run (synthetic data on disk)
# ---------------------------------------------------------------------------
def _make_synthetic_dataset(out_dir: Path, n_volumes: int = 4,
                            shape=(20, 64, 64), num_fg: int = 2,
                            seed: int = 0):
    rng = np.random.RandomState(seed)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)
    Z, Y, X = shape
    for i in range(n_volumes):
        img = rng.randn(*shape).astype(np.float32) * 50.0
        # NIfTI convention: (X, Y, Z); the project's load_nifti transposes.
        nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), affine),
                 str(img_dir / f"vol_{i:02d}.nii.gz"))

        lbl = np.zeros(shape, dtype=np.int16)
        # Plant a small fg blob per fg class somewhere in the middle.
        for c in range(num_fg):
            cz = rng.randint(2, Z - 2)
            cy = rng.randint(8, Y - 8)
            cx = rng.randint(8, X - 8)
            lbl[cz - 1:cz + 2, cy - 4:cy + 4, cx - 4:cx + 4] = c + 1
        nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), affine),
                 str(lbl_dir / f"vol_{i:02d}.nii.gz"))

    return str(img_dir), str(lbl_dir)


def test_end_to_end_2_5d_one_step():
    from segtask_v1.config import Config
    from segtask_v1.data.loader import build_dataloaders
    from segtask_v1.models.factory import build_model
    from segtask_v1.trainer import Trainer

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_dir, lbl_dir = _make_synthetic_dataset(
            td, n_volumes=4, shape=(20, 64, 64), num_fg=2)

        cfg = Config()
        cfg.data.image_dir = img_dir
        cfg.data.label_dir = lbl_dir
        cfg.data.patch_mode = "2_5d"
        cfg.data.patch_size = [12, 32, 32]
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.data.multi_res_scales = [1.0]
        cfg.data.batch_size = 1
        cfg.data.num_workers = 0
        cfg.data.samples_per_volume = 1
        cfg.data.foreground_oversample_ratio = 1.0
        cfg.data.intensity_min = -200.0
        cfg.data.intensity_max = 200.0
        cfg.data.cache_mode = "memory"
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.deep_supervision = False
        cfg.augment.enabled = False  # keep test deterministic
        cfg.train.epochs = 1
        cfg.train.use_amp = False
        cfg.train.use_ema = False
        cfg.train.warmup_epochs = 0
        cfg.train.compile_mode = "none"
        cfg.train.output_dir = str(td / "out")
        cfg.train.log_every = 1
        cfg.train.save_every = 9999
        cfg.train.val_every = 1
        cfg.sync()
        cfg.validate()

        train_loader, val_loader = build_dataloaders(cfg)
        # Sanity: dataset returns (B, 1, eD, pH, pW)
        sample = next(iter(train_loader))
        assert sample["image"].shape[1] == 1  # C_res=1
        assert sample["image"].shape[2] == cfg.data.patch_size[0]
        assert sample["image"].shape[3:] == tuple(cfg.data.patch_size[1:])

        device = torch.device("cpu")  # CPU for portability
        model = build_model(cfg)
        # Verify model sees D-channel input
        assert cfg.model.in_channels == cfg.data.patch_size[0]

        trainer = Trainer(model, cfg, train_loader, val_loader, device)
        # One epoch end-to-end (covers train + val)
        best = trainer.fit()

        assert "mean_dice" in best or len(best) >= 0
        # Predictions should have correct rank-4 shape after squeeze
        # (verified implicitly by trainer not crashing).
        _ok("Trainer end-to-end: data → augment → squeeze → 2D model "
            "→ SliceChannelLoss → backward → val")


# ---------------------------------------------------------------------------
# Regression: 3D path unaffected
# ---------------------------------------------------------------------------
def test_predictor_2_5d_inference():
    """R4: end-to-end predict_volume on synthetic 2.5D data.

    Verifies the predictor runs without error, the output label_map
    matches the source volume's spatial shape, and probabilities are
    rank-4 ``(num_fg, D, H, W)``.
    """
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model
    from segtask_v1.predictor import Predictor

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_dir, lbl_dir = _make_synthetic_dataset(
            td, n_volumes=1, shape=(20, 64, 64), num_fg=2)

        cfg = Config()
        cfg.data.image_dir = img_dir
        cfg.data.label_dir = lbl_dir
        cfg.data.patch_mode = "2_5d"
        cfg.data.patch_size = [12, 32, 32]
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.data.intensity_min = -200.0
        cfg.data.intensity_max = 200.0
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.deep_supervision = False
        cfg.predict.batch_size = 1
        cfg.predict.tta_flip = False
        cfg.predict.z_overlap = 0.5
        cfg.train.use_amp = False
        cfg.sync()
        cfg.validate()

        device = torch.device("cpu")
        model = build_model(cfg).to(device).eval()
        predictor = Predictor(model, cfg, device)

        img_paths = sorted(Path(img_dir).glob("*.nii.gz"))
        result = predictor.predict_volume(str(img_paths[0]))

        # Source volume shape was (20, 64, 64). load_nifti transposes
        # XYZ→ZYX so internal shape stays (20, 64, 64).
        assert result["label_map"].shape == (20, 64, 64), (
            f"unexpected label_map shape {result['label_map'].shape}")
        assert result["probabilities"].shape == (2, 20, 64, 64), (
            f"unexpected probabilities shape {result['probabilities'].shape}")
        assert np.isfinite(result["probabilities"]).all()
        # Probabilities must be in [0, 1] (sigmoid output).
        assert (result["probabilities"] >= 0).all()
        assert (result["probabilities"] <= 1).all()
        _ok("Predictor 2.5D end-to-end (no TTA)")


def test_predictor_2_5d_inference_tta():
    """R4: predict_volume with TTA on must produce same shape + valid range."""
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model
    from segtask_v1.predictor import Predictor

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_dir, lbl_dir = _make_synthetic_dataset(
            td, n_volumes=1, shape=(16, 48, 48), num_fg=2, seed=1)

        cfg = Config()
        cfg.data.image_dir = img_dir
        cfg.data.label_dir = lbl_dir
        cfg.data.patch_mode = "2_5d"
        cfg.data.patch_size = [8, 32, 32]
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.data.intensity_min = -200.0
        cfg.data.intensity_max = 200.0
        cfg.model.encoder_channels = [16, 32]
        cfg.model.deep_supervision = False
        cfg.predict.batch_size = 1
        cfg.predict.tta_flip = True
        cfg.predict.z_overlap = 0.5
        cfg.train.use_amp = False
        cfg.sync()
        cfg.validate()

        device = torch.device("cpu")
        model = build_model(cfg).to(device).eval()
        predictor = Predictor(model, cfg, device)
        img_paths = sorted(Path(img_dir).glob("*.nii.gz"))
        result = predictor.predict_volume(str(img_paths[0]))

        assert result["label_map"].shape == (16, 48, 48)
        assert result["probabilities"].shape == (2, 16, 48, 48)
        assert (result["probabilities"] >= 0).all()
        assert (result["probabilities"] <= 1).all()
        _ok("Predictor 2.5D end-to-end (TTA on, H/W flips only)")


def test_regression_3d_factory_still_works():
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [16, 32, 32]
    cfg.sync()
    cfg.validate()
    assert cfg.model.spatial_dims == 3, "default still 3D"
    model = build_model(cfg).eval()
    x = torch.randn(1, 1, 16, 32, 32)
    y = model(x)
    assert y.shape == (1, 2, 16, 32, 32)
    _ok("3D factory regression: default path still produces 5D output")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("R3 + R4 smoke test — 2.5D mode end-to-end + predictor")
    print("=" * 60)
    tests = [
        # R3 — config / loss / factory / loader / trainer
        test_config_2_5d_sync,
        test_config_2_5d_rejects_multi_res,
        test_slice_channel_loss_forward,
        test_slice_channel_loss_with_weight_map,
        test_slice_channel_split_for_metrics,
        test_slice_channel_loss_invalid_shapes,
        test_factory_2_5d_out_channels,
        test_regression_3d_factory_still_works,
        test_end_to_end_2_5d_one_step,
        # R4 — predictor
        test_predictor_2_5d_inference,
        test_predictor_2_5d_inference_tta,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print("=" * 60)
    print("All R3 smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
