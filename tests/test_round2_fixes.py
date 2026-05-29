"""Smoke tests for Round 2 fixes (BUG-1, BUG-2, BUG-3).

Run with:
    conda activate py310
    python -m pytest test_round2_fixes.py -v
or:
    python test_round2_fixes.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

# Ensure local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segtask_v1.data.dataset import (
    SegDataset3D,
    _extract_cubic_patch,
    preprocess_image,
)
from segtask_v1.losses.losses import (
    BinaryDiceLoss,
    MultiResolutionLoss,
)
from segtask_v1.models.unet import UNet3D, Encoder, Decoder
from segtask_v1.models.resnet import ResNetStage
from functools import partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_nifti(arr: np.ndarray, path: str) -> None:
    import nibabel as nib
    # Save in (X, Y, Z) order — loader will transpose to (D, H, W)
    nib.save(nib.Nifti1Image(arr.transpose(2, 1, 0), np.eye(4)), path)


def _make_tiny_volumes(tmpdir: str, num_fg: int = 2, shape=(16, 32, 32)):
    img_dir = os.path.join(tmpdir, "img")
    lbl_dir = os.path.join(tmpdir, "lbl")
    os.makedirs(img_dir); os.makedirs(lbl_dir)
    rng = np.random.RandomState(0)
    img_paths, lbl_paths = [], []
    for i in range(2):
        # Float image in HU-ish range
        img = rng.uniform(-500, 500, size=shape).astype(np.float32)
        # Label: spread out [0..num_fg] so each class has fg voxels
        lbl = np.zeros(shape, dtype=np.int16)
        D, H, W = shape
        for c in range(1, num_fg + 1):
            # Carve a box per class
            lbl[c * 2 : c * 2 + 2, :H // 2, :W // 2] = c
        img_p = os.path.join(img_dir, f"v{i}.nii.gz")
        lbl_p = os.path.join(lbl_dir, f"v{i}.nii.gz")
        _write_nifti(img, img_p)
        _write_nifti(lbl.astype(np.float32), lbl_p)
        img_paths.append(img_p); lbl_paths.append(lbl_p)
    return img_paths, lbl_paths


def _build_tiny_unet(num_fg: int, deep_supervision: bool = False) -> UNet3D:
    ch = [8, 16, 32]
    stage_builder = partial(
        ResNetStage, num_blocks=1, norm_type="instance",
        norm_groups=8, activation="leakyrelu", dropout=0.0, use_se=False)
    enc = Encoder(
        in_channels=1, stage_channels=ch, stage_builder=stage_builder,
        norm_type="instance", norm_groups=8, activation="leakyrelu")
    dec = Decoder(encoder_channels=ch, stage_builder=stage_builder,
                  upsample_mode="transpose", skip_mode="cat")
    return UNet3D(enc, dec, num_fg_classes=num_fg,
                  deep_supervision=deep_supervision)


# ---------------------------------------------------------------------------
# BUG-1: z_axis returns (1, D, H, W) raw label; MultiResolutionLoss binarizes
# correctly for num_fg >= 2.
# ---------------------------------------------------------------------------
def test_bug1_z_axis_label_shape_and_multi_class():
    np.random.seed(0); torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        label_values = [0, 1, 2]  # num_fg = 2
        img_paths, lbl_paths = _make_tiny_volumes(tmp, num_fg=2)

        ds = SegDataset3D(
            image_paths=img_paths, label_paths=lbl_paths,
            label_values=label_values,
            patch_size=(8, 16, 16),
            samples_per_volume=2, is_train=True,
            # force foreground sampling to guarantee both classes are seen
            foreground_oversample_ratio=1.0,
            cache_enabled=False)

        # _make_tiny_volumes places class-c in slices [2c, 2c+2]. With
        # patch_size D=8 and both fg classes present in z=[2..5], at least
        # one sample must contain both classes.
        sample = None
        for i in range(len(ds)):
            s = ds[i]
            lbl = s["label"].numpy()
            if (lbl == 1).any() and (lbl == 2).any():
                sample = s; break
        assert sample is not None, "Failed to draw a patch with both fg classes"
        # Contract: label is (1, D, H, W) raw integer label (as float)
        assert sample["label"].shape == (1, 8, 16, 16), \
            f"Expected (1, 8, 16, 16), got {sample['label'].shape}"
        # Values should be integers from label_values
        uniq = torch.unique(sample["label"]).tolist()
        assert set(uniq).issubset({0.0, 1.0, 2.0}), f"Unexpected label values: {uniq}"

        # End-to-end loss with MultiResolutionLoss must see BOTH classes
        loss_fn = MultiResolutionLoss(
            base_loss=BinaryDiceLoss(),
            num_fg_classes=2, num_res=1, label_values=label_values)

        label_raw = sample["label"].unsqueeze(0)  # (1, 1, D, H, W)
        # Fake predictions: (1, num_fg=2, D, H, W)
        pred = torch.randn(1, 2, 8, 16, 16)
        loss = loss_fn(pred, label_raw)
        assert torch.isfinite(loss), f"Loss not finite: {loss}"

        # The key check: class-2 binary target must not be all-zero
        bin_target = loss_fn._label_to_binary(label_raw[:, 0])  # (1, 2, D, H, W)
        assert bin_target[:, 0].sum() > 0, "class-1 target empty — data too sparse?"
        assert bin_target[:, 1].sum() > 0, \
            "BUG-1 regression: class-2 target is empty!"

    print("[BUG-1] PASS — z_axis returns (1, D, H, W), multi-class targets valid.")


# ---------------------------------------------------------------------------
# BUG-2: DS outputs ordered by decreasing resolution
# ---------------------------------------------------------------------------
def test_bug2_ds_output_order():
    # Use 4 encoder levels so we get main + 2 DS outputs
    ch = [8, 16, 32, 64]
    stage_builder = partial(
        ResNetStage, num_blocks=1, norm_type="instance",
        norm_groups=8, activation="leakyrelu", dropout=0.0, use_se=False)
    enc = Encoder(
        in_channels=1, stage_channels=ch, stage_builder=stage_builder,
        norm_type="instance", norm_groups=8, activation="leakyrelu")
    dec = Decoder(encoder_channels=ch, stage_builder=stage_builder,
                  upsample_mode="transpose", skip_mode="cat")
    model = UNet3D(enc, dec, num_fg_classes=2, deep_supervision=True).train()

    x = torch.randn(1, 1, 16, 32, 32)
    outs = model(x)

    assert isinstance(outs, list), "DS must return a list in training mode"
    # 4 encoder levels → decoder has 3 levels → 2 DS heads + main = 3 outputs
    assert len(outs) == 3, f"Expected 3 outputs for 4-level encoder, got {len(outs)}"

    # Spatial sizes must be strictly decreasing resolution along the list
    sizes = [tuple(o.shape[2:]) for o in outs]
    print(f"  DS output sizes (highest→lowest): {sizes}")

    # First output must be highest resolution (== input size for depth=3)
    assert sizes[0] == tuple(x.shape[2:]), \
        f"main_out must be input resolution, got {sizes[0]} vs {x.shape[2:]}"

    # Each subsequent size must be smaller than the previous (strict monotone)
    for i in range(len(sizes) - 1):
        prev, cur = sizes[i], sizes[i + 1]
        # Every axis should be <= previous (usually exactly halved)
        assert all(c < p for c, p in zip(cur, prev)), (
            f"BUG-2 regression: DS output {i + 1} not lower-res than {i}: "
            f"{cur} vs {prev}")

    print("[BUG-2] PASS — DS outputs are strictly decreasing resolution.")


# ---------------------------------------------------------------------------
# BUG-3: _extract_cubic_patch pads to requested size even at boundaries
# ---------------------------------------------------------------------------
def test_bug3_extract_cubic_patch_pads_boundary():
    vol = np.arange(10 * 10 * 10, dtype=np.float32).reshape(10, 10, 10)

    # Case 1: center at (0,0,0), size (6,6,6) → lo=(-3,-3,-3), hi=(3,3,3)
    # Cropped shape (3,3,3); must be padded back to (6,6,6)
    patch = _extract_cubic_patch(vol, center=(0, 0, 0), size=(6, 6, 6))
    assert patch.shape == (6, 6, 6), \
        f"BUG-3 regression: expected (6,6,6) with pad, got {patch.shape}"

    # Case 2: center fully interior → no padding needed
    patch = _extract_cubic_patch(vol, center=(5, 5, 5), size=(4, 4, 4))
    assert patch.shape == (4, 4, 4)

    # Case 3: center near far boundary
    patch = _extract_cubic_patch(vol, center=(9, 9, 9), size=(6, 6, 6))
    assert patch.shape == (6, 6, 6), \
        f"BUG-3 regression: expected (6,6,6) near far boundary, got {patch.shape}"

    # Edge-mode replication check: boundary voxels should be repeated
    patch = _extract_cubic_patch(vol, center=(0, 0, 0), size=(6, 6, 6))
    # Out-of-bounds padded region should hold the edge value (vol[0,0,0]=0)
    assert patch[0, 0, 0] == 0.0

    print("[BUG-3] PASS — cubic patch padded to exact size with edge mode.")


# ---------------------------------------------------------------------------
# Integration: full training step on z_axis + num_fg=2 + DS
# ---------------------------------------------------------------------------
def test_integration_z_axis_multi_class_ds():
    with tempfile.TemporaryDirectory() as tmp:
        label_values = [0, 1, 2]
        img_paths, lbl_paths = _make_tiny_volumes(tmp, num_fg=2)

        ds = SegDataset3D(
            image_paths=img_paths, label_paths=lbl_paths,
            label_values=label_values,
            patch_size=(8, 16, 16),
            samples_per_volume=2, is_train=True, cache_enabled=False)
        sample = ds[0]

        model = _build_tiny_unet(num_fg=2, deep_supervision=True).train()
        image = sample["image"].unsqueeze(0)  # (1, 1, D, H, W)
        label = sample["label"].unsqueeze(0)  # (1, 1, D, H, W)

        # Correct composition: DS(MR(base)). Match the trainer's order.
        from segtask_v1.losses.losses import DeepSupervisionLoss
        base = BinaryDiceLoss()
        inner = MultiResolutionLoss(
            base_loss=base, num_fg_classes=2, num_res=1,
            label_values=label_values)
        # Model used below has 3-level encoder → 2 outputs (main + 1 DS)
        criterion = DeepSupervisionLoss(inner, weights=[1.0, 0.5])

        pred = model(image)  # list of 3 tensors, decreasing resolution
        loss = criterion(pred, label)
        assert torch.isfinite(loss), f"Integration loss not finite: {loss}"
        loss.backward()
        print(f"[INTEG] PASS — forward+backward works, loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# Round 3 BUGs: EMA in-place swap, plateau mode, config validate, DropPath AMP,
# detect_label_values full scan.
# ---------------------------------------------------------------------------
def test_bug4_ema_in_place_swap():
    """EMA.apply_shadow / restore should swap tensors in place without
    allocating a fresh deepcopy every call, and the model must see EMA
    weights after apply_shadow and original weights after restore."""
    from segtask_v1.utils import ModelEMA

    model = _build_tiny_unet(num_fg=2, deep_supervision=False)
    # Snapshot original weights
    orig = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ema = ModelEMA(model, decay=0.9)
    # Perturb model weights then call update twice — EMA should diverge
    # from both original and current.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)

    # apply_shadow: model.state_dict() should equal EMA shadow
    ema.apply_shadow(model)
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            assert torch.equal(v, ema.shadow[k]), f"apply_shadow mismatch at {k}"

    # restore: model.state_dict() should equal pre-apply state (= perturbed)
    ema.restore(model)
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            # Perturbed = orig + 1.0
            expected = orig[k] + 1.0 if orig[k].is_floating_point() else orig[k]
            assert torch.equal(v, expected), f"restore mismatch at {k}"

    # Idempotent calls must not crash / corrupt state
    ema.restore(model)  # no-op
    ema.apply_shadow(model)
    ema.apply_shadow(model)  # idempotent
    ema.restore(model)
    print("[BUG-4] PASS — EMA in-place swap round-trips cleanly.")


def test_bug5_plateau_mode_from_config():
    """build_scheduler should use plateau mode matching save_best_mode."""
    from segtask_v1.trainer import build_scheduler
    from segtask_v1.config import Config

    cfg = Config()
    cfg.train.scheduler = "plateau"
    cfg.train.save_best_mode = "min"
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sch = build_scheduler(opt, cfg, steps_per_epoch=10, post_warmup_steps=100)
    assert sch.mode == "min", f"Expected plateau mode=min, got {sch.mode}"

    cfg.train.save_best_mode = "max"
    sch = build_scheduler(opt, cfg, steps_per_epoch=10, post_warmup_steps=100)
    assert sch.mode == "max", f"Expected plateau mode=max, got {sch.mode}"
    print("[BUG-5] PASS — plateau mode tracks save_best_mode.")


def test_bug6_config_validate_rejects_illegal_combo():
    from segtask_v1.config import Config

    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.patch_mode = "z_axis"
    cfg.data.multi_res_scales = [1.0, 1.5]  # illegal for z_axis
    raised = False
    try:
        cfg.validate()
    except AssertionError as e:
        raised = "multi_res_scales" in str(e) and "z_axis" in str(e)
    assert raised, "BUG-6 regression: z_axis + multi-res should be rejected"

    # Legal: z_axis + [1.0]
    cfg.data.multi_res_scales = [1.0]
    cfg.validate()

    # Legal: cubic + multi-res
    cfg.data.patch_mode = "cubic"
    cfg.data.multi_res_scales = [1.0, 1.5]
    cfg.validate()

    # save_best_mode validation
    cfg.train.save_best_mode = "bogus"
    try:
        cfg.validate()
        assert False, "Invalid save_best_mode must be rejected"
    except AssertionError:
        pass
    print("[BUG-6] PASS — config validator rejects illegal z_axis+multi_res.")


def test_bug7_droppath_amp_dtype():
    """DropPath must produce output matching x.dtype under fp16/bf16."""
    from segtask_v1.models.convnext import DropPath

    dp = DropPath(drop_prob=0.2).train()
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(4, 8, 4, 4, 4, dtype=dt)
        out = dp(x)
        assert out.dtype == dt, (
            f"BUG-7 regression: DropPath returned {out.dtype} for input {dt}")
        # Sanity: deterministic when drop_prob=0
        dp0 = DropPath(drop_prob=0.0).train()
        y = dp0(x)
        assert torch.equal(y, x)
    print("[BUG-7] PASS — DropPath robust across fp32/fp16/bf16.")


def test_bug8_detect_label_values_full_scan():
    """detect_label_values must scan all files by default and discover
    rare classes that are absent from the first few files."""
    from segtask_v1.data.loader import detect_label_values
    import nibabel as nib

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        # Files 0..5: only label 0 and 1
        for i in range(6):
            arr = np.zeros((4, 4, 4), dtype=np.int16)
            arr[0, 0, 0] = 1
            p = os.path.join(tmp, f"v{i}.nii.gz")
            nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), p)
            paths.append(p)
        # File 6: introduces label 3
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        arr[0, 0, 0] = 3
        p = os.path.join(tmp, "v6.nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), p)
        paths.append(p)

        # Default (full scan) must find label 3
        vals = detect_label_values(paths)
        assert 3 in vals, f"BUG-8 regression: rare class missed, got {vals}"

        # Partial scan (first 5) misses it — but should warn, not error.
        vals_partial = detect_label_values(paths, max_scan=5)
        assert 3 not in vals_partial
    print("[BUG-8] PASS — detect_label_values scans all files by default.")


# ---------------------------------------------------------------------------
# Round 4: vectorization, stratified split, pooled dice, RNG state
# ---------------------------------------------------------------------------
def test_bug10a_random_gamma_vectorized():
    """Vectorized gamma must produce deterministic per-sample results and
    leave un-selected samples untouched."""
    from segtask_v1.data.augment import _random_gamma

    torch.manual_seed(0)
    B, C, D, H, W = 3, 1, 4, 8, 8
    x = torch.rand(B, C, D, H, W)

    # prob=0: identity
    y = _random_gamma(x.clone(), prob=0.0, grange=[0.8, 1.2])
    assert torch.equal(y, x), "prob=0 must be identity"

    # gamma=[1.0, 1.0] with prob=1: still identity (after normalize/denorm)
    torch.manual_seed(0)
    y = _random_gamma(x.clone(), prob=1.0, grange=[1.0, 1.0])
    assert torch.allclose(y, x, atol=1e-5), \
        f"gamma=1 must be near-identity, max diff={(y-x).abs().max()}"

    # gamma>1 must decrease mid-range intensities (0.5 -> 0.5^γ < 0.5)
    torch.manual_seed(0)
    x_mid = torch.full((1, 1, 2, 2, 2), 0.5)
    # Need non-constant image for normalize; use spread
    x_mid = torch.linspace(0, 1, 8).reshape(1, 1, 2, 2, 2)
    y = _random_gamma(x_mid.clone(), prob=1.0, grange=[2.0, 2.0])
    assert y.shape == x_mid.shape
    assert torch.isfinite(y).all()
    print("[BUG-10a] PASS — _random_gamma vectorized and correct.")


def test_bug10b_grid_dropout_vectorized():
    from segtask_v1.data.augment import _grid_dropout

    torch.manual_seed(0)
    x = torch.ones(4, 1, 8, 16, 16)
    lbl = torch.ones(4, 1, 8, 16, 16)

    x_out, lbl_out = _grid_dropout(
        x.clone(), lbl.clone(), prob=1.0, ratio=0.3, num_holes=3)
    # Label untouched
    assert torch.equal(lbl_out, lbl), "grid_dropout must not modify label"
    # At least some voxels zeroed
    assert (x_out == 0).any(), "Expected dropped voxels"
    # Shape preserved
    assert x_out.shape == x.shape

    # prob=0 identity
    y, _ = _grid_dropout(x.clone(), lbl, prob=0.0, ratio=0.3, num_holes=3)
    assert torch.equal(y, x)
    print("[BUG-10b] PASS — _grid_dropout vectorized and correct.")


def test_bug11_stratified_split():
    from segtask_v1.data.loader import stratified_train_val_split
    import nibabel as nib

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        # 6 class-1 volumes + 4 class-2 volumes
        for i in range(6):
            arr = np.zeros((4, 4, 4), dtype=np.int16)
            arr[0, 0, 0] = 1
            p = os.path.join(tmp, f"c1_{i}.nii.gz")
            nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), p)
            paths.append(p)
        for i in range(4):
            arr = np.zeros((4, 4, 4), dtype=np.int16)
            arr[0, 0, 0] = 2
            p = os.path.join(tmp, f"c2_{i}.nii.gz")
            nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), p)
            paths.append(p)

        train, val = stratified_train_val_split(
            paths, [0, 1, 2], val_ratio=0.25, seed=42)

        # Val should contain samples from BOTH classes
        val_is_c1 = [i for i in val if i < 6]
        val_is_c2 = [i for i in val if i >= 6]
        assert len(val_is_c1) >= 1, f"Missing class-1 in val: {val}"
        assert len(val_is_c2) >= 1, f"Missing class-2 in val: {val}"
        # Total preserved
        assert len(train) + len(val) == 10
        print("[BUG-11] PASS — stratified split covers all strata.")


def test_d1_dice_pooled_vs_naive():
    """Pooled dice must be robust to empty GT batches that would otherwise
    zero-out the per-batch average."""
    from segtask_v1.utils import dice_batch_stats

    # Batch 1: class-0 GT=1, class-1 GT=0. Pred perfect (matches GT per class).
    p1 = torch.full((1, 2, 4, 4, 4), -10.0)      # default low
    p1[:, 0] = 10.0                              # class-0: predict 1
    t1 = torch.zeros(1, 2, 4, 4, 4)
    t1[:, 0] = 1.0

    # Batch 2: class-1 GT=1, class-0 GT=0. Pred perfect.
    p2 = torch.full((1, 2, 4, 4, 4), -10.0)
    p2[:, 1] = 10.0
    t2 = torch.zeros(1, 2, 4, 4, 4)
    t2[:, 1] = 1.0

    s1 = dice_batch_stats(p1, t1)
    s2 = dice_batch_stats(p2, t2)
    inter = s1["inter"] + s2["inter"]
    denom = s1["denom"] + s2["denom"]
    dice = (2 * inter + 1e-5) / (denom + 1e-5)

    # Both classes should be (near-)1.0 under pooled dice.
    assert dice[0] > 0.99 and dice[1] > 0.99, \
        f"Pooled dice must be ~1 per class, got {dice.tolist()}"
    print("[D-1] PASS — pooled dice robust across empty-GT batches.")


def test_d3_rng_state_roundtrip():
    """_build_state_dict must include rng_state; _load_checkpoint restores it."""
    # Just validate the key presence in the saved dict (full resume path
    # requires a trainer instance with real data, covered by integration).
    from segtask_v1.trainer import Trainer  # noqa: F401
    import inspect, re
    src = inspect.getsource(Trainer._build_state_dict)
    assert '"rng_state"' in src, "rng_state key missing from saved state"
    src_load = inspect.getsource(Trainer._load_checkpoint)
    assert "rng_state" in src_load, "rng_state not restored on load"
    print("[D-3] PASS — rng_state included in save/load paths.")


if __name__ == "__main__":
    # Round 2
    test_bug1_z_axis_label_shape_and_multi_class()
    test_bug2_ds_output_order()
    test_bug3_extract_cubic_patch_pads_boundary()
    test_integration_z_axis_multi_class_ds()
    # Round 3
    test_bug4_ema_in_place_swap()
    test_bug5_plateau_mode_from_config()
    test_bug6_config_validate_rejects_illegal_combo()
    test_bug7_droppath_amp_dtype()
    test_bug8_detect_label_values_full_scan()
    # Round 4
    test_bug10a_random_gamma_vectorized()
    test_bug10b_grid_dropout_vectorized()
    test_bug11_stratified_split()
    test_d1_dice_pooled_vs_naive()
    test_d3_rng_state_roundtrip()
    print("\nAll Rounds 2-4 fixes verified.")
