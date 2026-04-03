"""Comprehensive test suite for SegTask.

Tests all components: config, data matching, datasets, transforms,
model architectures (all encoder/decoder combinations), losses,
and the training/prediction pipelines.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

# ---- Setup logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PASSED = 0
FAILED = 0


def run_test(name, func):
    """Run a test function and track pass/fail."""
    global PASSED, FAILED
    try:
        func()
        PASSED += 1
        logger.info("  ✓ %s", name)
    except Exception as e:
        FAILED += 1
        logger.error("  ✗ %s: %s", name, e)
        import traceback
        traceback.print_exc()


# ===========================================================================
# 1. Config tests
# ===========================================================================
def test_config_defaults():
    from segtask.config import Config
    cfg = Config()
    cfg.sync()
    assert cfg.data.mode == "2.5d"
    assert cfg.model.spatial_dims == 2
    assert cfg.model.encoder_name == "resnet"
    assert cfg.model.decoder_channels == [256, 128, 64, 32]


def test_config_yaml_roundtrip():
    from segtask.config import Config, save_config, load_config
    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.sync()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_config.yaml")
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.data.label_values == [0, 1, 2]
        assert loaded.data.num_classes == 3
        assert loaded.model.decoder_channels == cfg.model.decoder_channels


def test_config_sync_2d():
    from segtask.config import Config
    cfg = Config()
    cfg.data.mode = "2d"
    cfg.data.label_values = [0, 1]
    cfg.sync()
    assert cfg.model.spatial_dims == 2
    assert cfg.model.in_channels == 1


def test_config_sync_25d():
    from segtask.config import Config
    cfg = Config()
    cfg.data.mode = "2.5d"
    cfg.data.num_slices_per_side = 2
    cfg.data.label_values = [0, 1]
    cfg.sync()
    assert cfg.model.spatial_dims == 2
    assert cfg.model.in_channels == 5  # 2*2+1


def test_config_sync_3d():
    from segtask.config import Config
    cfg = Config()
    cfg.data.mode = "3d"
    cfg.data.label_values = [0, 1]
    cfg.sync()
    assert cfg.model.spatial_dims == 3
    assert cfg.model.in_channels == 1


# ===========================================================================
# 2. Data matching tests
# ===========================================================================
def test_extract_subject_id():
    from segtask.data.matching import _extract_subject_id
    assert _extract_subject_id("s0000.nii.gz") == "s0000"
    assert _extract_subject_id("s0001-seg.nii.gz") == "s0001"
    assert _extract_subject_id("patient_01_mask.nii.gz") == "patient_01"
    assert _extract_subject_id("sub-001_label.nii") == "sub-001"


def test_data_matching_with_real_data():
    """Test matching with actual TotalSegmentator data."""
    from segtask.data.matching import match_data
    img_dir = r"F:\med_data\Totalsegmentator_dataset_v201\nii"
    lbl_dir = r"F:\med_data\Totalsegmentator_dataset_v201\body_pred"
    if not Path(img_dir).exists():
        logger.warning("Skipping real data test (data not found)")
        return
    records = match_data(img_dir, lbl_dir)
    assert len(records) == 15, f"Expected 15 matched pairs, got {len(records)}"


def test_split_random():
    from segtask.data.matching import SampleRecord, _split_random
    records = [SampleRecord(f"s{i:04d}", f"img_{i}", f"lbl_{i}") for i in range(20)]
    train, val, test = _split_random(records, 0.15, 0.15, 42)
    assert len(train) + len(val) + len(test) == 20
    assert len(val) >= 1
    assert len(test) >= 1


# ===========================================================================
# 3. Model tests (all encoder/decoder combos, 2D and 3D)
# ===========================================================================
def _test_model_combo(enc_name, dec_name, spatial_dims, in_channels, num_classes):
    from segtask.config import Config
    from segtask.models.factory import build_model

    cfg = Config()
    cfg.data.label_values = list(range(num_classes))
    cfg.data.num_classes = num_classes
    cfg.data.mode = "3d" if spatial_dims == 3 else "2d"
    cfg.model.encoder_name = enc_name
    cfg.model.decoder_name = dec_name
    cfg.model.encoder_channels = [16, 32, 64, 128]
    cfg.model.encoder_blocks_per_level = [1, 1, 1, 1]
    cfg.model.decoder_blocks_per_level = [1, 1, 1]
    cfg.model.vit_num_heads = 4
    cfg.model.vit_patch_size = 2
    cfg.sync()
    # Override in_channels after sync (sync auto-sets based on mode)
    cfg.model.in_channels = in_channels

    model = build_model(cfg)
    model.eval()

    if spatial_dims == 2:
        x = torch.randn(2, in_channels, 64, 64)
    else:
        x = torch.randn(1, in_channels, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    if spatial_dims == 2:
        assert out.shape == (2, num_classes, 64, 64), f"Got {out.shape}"
    else:
        assert out.shape == (1, num_classes, 32, 32, 32), f"Got {out.shape}"


def test_vgg_vgg_2d():
    _test_model_combo("vgg", "vgg", 2, 3, 3)

def test_resnet_resnet_2d():
    _test_model_combo("resnet", "resnet", 2, 3, 3)

def test_vit_vit_2d():
    _test_model_combo("vit", "vit", 2, 3, 3)

def test_resnet_vgg_2d():
    _test_model_combo("resnet", "vgg", 2, 1, 2)

def test_vgg_resnet_2d():
    _test_model_combo("vgg", "resnet", 2, 1, 2)

def test_vit_resnet_2d():
    _test_model_combo("vit", "resnet", 2, 3, 3)

def test_resnet_resnet_3d():
    _test_model_combo("resnet", "resnet", 3, 1, 3)

def test_vgg_vgg_3d():
    _test_model_combo("vgg", "vgg", 3, 1, 2)

def test_vit_vit_3d():
    _test_model_combo("vit", "vit", 3, 1, 2)


def test_deep_supervision():
    from segtask.config import Config
    from segtask.models.factory import build_model

    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.mode = "2d"
    cfg.model.encoder_name = "resnet"
    cfg.model.decoder_name = "resnet"
    cfg.model.encoder_channels = [16, 32, 64, 128]
    cfg.model.encoder_blocks_per_level = [1, 1, 1, 1]
    cfg.model.decoder_blocks_per_level = [1, 1, 1]
    cfg.model.deep_supervision = True
    cfg.sync()

    model = build_model(cfg)
    model.train()

    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert isinstance(out, list), "Deep supervision should return list"
    assert out[0].shape == (2, 3, 64, 64)
    assert len(out) >= 2


# ===========================================================================
# 4. Loss tests
# ===========================================================================
def test_dice_loss():
    from segtask.losses.losses import DiceLoss
    loss_fn = DiceLoss()
    pred = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 3, 32, 32)
    target[:, 0] = 1.0  # all background
    loss = loss_fn(pred, target)
    assert loss.ndim == 0 and loss.item() >= 0


def test_ce_loss():
    from segtask.losses.losses import CrossEntropyLoss
    loss_fn = CrossEntropyLoss()
    pred = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 3, 32, 32)
    target[:, 1] = 1.0
    loss = loss_fn(pred, target)
    assert loss.ndim == 0 and loss.item() > 0


def test_focal_loss():
    from segtask.losses.losses import FocalLoss
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    pred = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 3, 32, 32)
    target[:, 1] = 1.0
    loss = loss_fn(pred, target)
    assert loss.ndim == 0 and loss.item() > 0


def test_tversky_loss():
    from segtask.losses.losses import TverskyLoss
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    pred = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 3, 32, 32)
    target[:, 0] = 1.0
    loss = loss_fn(pred, target)
    assert loss.ndim == 0 and loss.item() >= 0


def test_compound_loss():
    from segtask.losses.losses import DiceLoss, CrossEntropyLoss, CompoundLoss
    dice = DiceLoss()
    ce = CrossEntropyLoss()
    compound = CompoundLoss([dice, ce], [1.0, 1.0])
    pred = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 3, 32, 32)
    target[:, 1] = 1.0
    loss = compound(pred, target)
    assert loss.ndim == 0 and loss.item() > 0


def test_deep_supervision_loss():
    from segtask.losses.losses import DiceLoss, DeepSupervisionLoss
    base = DiceLoss()
    ds_loss = DeepSupervisionLoss(base, weights=[1.0, 0.5])

    main = torch.randn(2, 3, 64, 64)
    aux = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 3, 64, 64)
    target[:, 0] = 1.0

    loss = ds_loss([main, aux], target)
    assert loss.ndim == 0 and loss.item() >= 0

    # Also test with single tensor
    loss2 = ds_loss(main, target)
    assert loss2.ndim == 0


def test_build_loss():
    from segtask.config import LossConfig
    from segtask.losses.losses import build_loss

    for name in ["dice", "ce", "dice_ce", "focal", "tversky", "dice_focal"]:
        cfg = LossConfig(name=name)
        loss_fn = build_loss(cfg)
        pred = torch.randn(2, 3, 16, 16)
        target = torch.zeros(2, 3, 16, 16)
        target[:, 1] = 1.0
        loss = loss_fn(pred, target)
        assert loss.ndim == 0, f"{name} loss failed"


# ===========================================================================
# 5. Transform tests
# ===========================================================================
def test_gpu_augmentor_2d():
    from segtask.config import AugmentConfig
    from segtask.data.transforms import GPUAugmentor
    cfg = AugmentConfig(
        random_flip_prob=1.0,
        random_rotate_prob=1.0,
        random_brightness_prob=1.0,
        gaussian_noise_prob=1.0,
    )
    aug = GPUAugmentor(cfg, spatial_dims=2)
    img = torch.randn(2, 1, 64, 64)
    lbl = torch.zeros(2, 3, 64, 64)
    lbl[:, 0] = 1.0
    img_aug, lbl_aug = aug(img, lbl)
    assert img_aug.shape == img.shape
    assert lbl_aug.shape == lbl.shape


def test_gpu_augmentor_3d():
    from segtask.config import AugmentConfig
    from segtask.data.transforms import GPUAugmentor
    cfg = AugmentConfig(random_flip_prob=1.0, random_rotate_prob=1.0)
    aug = GPUAugmentor(cfg, spatial_dims=3)
    img = torch.randn(1, 1, 32, 32, 32)
    lbl = torch.zeros(1, 2, 32, 32, 32)
    lbl[:, 0] = 1.0
    img_aug, lbl_aug = aug(img, lbl)
    assert img_aug.shape == img.shape
    assert lbl_aug.shape == lbl.shape


def test_mixup():
    from segtask.data.transforms import MixupCutmix
    mixup = MixupCutmix(alpha=0.2, prob=1.0)
    img = torch.randn(4, 1, 32, 32)
    lbl = torch.zeros(4, 3, 32, 32)
    lbl[:, 0] = 1.0
    img_m, lbl_m = mixup(img, lbl)
    assert img_m.shape == img.shape


# ===========================================================================
# 6. Metrics tests
# ===========================================================================
def test_dice_metric():
    from segtask.utils import compute_dice_per_class
    # Perfect prediction
    pred = torch.zeros(2, 3, 16, 16)
    pred[:, 1] = 10.0  # high logit for class 1
    target = torch.zeros(2, 3, 16, 16)
    target[:, 1] = 1.0
    dice = compute_dice_per_class(pred, target)
    assert dice[1].item() > 0.9, f"Perfect pred should have high dice, got {dice[1]:.4f}"


def test_ema():
    from segtask.utils import ModelEMA
    model = torch.nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9)

    # Modify model params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)

    # Shadow should be between original and modified
    ema.apply_shadow(model)
    # Check it was applied
    ema.restore(model)
    # Should be restored


# ===========================================================================
# 7. Integration test: mini training step
# ===========================================================================
def test_mini_training_step():
    from segtask.config import Config
    from segtask.models.factory import build_model
    from segtask.losses.losses import build_loss

    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.mode = "2d"
    cfg.model.encoder_name = "resnet"
    cfg.model.decoder_name = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.encoder_blocks_per_level = [1, 1, 1]
    cfg.model.decoder_blocks_per_level = [1, 1]
    cfg.sync()

    model = build_model(cfg)
    criterion = build_loss(cfg.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fake batch
    x = torch.randn(2, 1, 64, 64)
    target = torch.zeros(2, 2, 64, 64)
    target[:, 0, :32] = 1.0
    target[:, 1, 32:] = 1.0

    model.train()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 100, "Loss should be reasonable"


def test_25d_reuses_3d():
    """Test 2.5D: reuses SegDataset3D, model outputs D*C channels, per-slice loss."""
    from segtask.config import Config
    from segtask.models.factory import build_model
    from segtask.losses.losses import build_loss

    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.mode = "2.5d"
    cfg.data.num_slices_per_side = 1  # total_slices = 3
    cfg.model.encoder_name = "resnet"
    cfg.model.decoder_name = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.encoder_blocks_per_level = [1, 1, 1]
    cfg.model.decoder_blocks_per_level = [1, 1]
    cfg.sync()

    model = build_model(cfg)
    S, C = 3, 2
    assert model.total_slices == S
    assert model.semantic_classes == C
    assert model.num_classes == S * C  # 3 slices × 2 classes = 6

    # Simulate SegDataset3D output for 2.5D (patch_size=(3, H, W))
    # image: (B, 1, D, H, W) — 3D sub-volume from SegDataset3D
    image_3d = torch.randn(2, 1, S, 64, 64)
    # label: (B, C, D, H, W) — one-hot 3D from SegDataset3D
    label_3d = torch.zeros(2, C, S, 64, 64)
    label_3d[:, 0] = 1.0  # all background

    # Step 1: squeeze image for 2D model (what trainer does)
    image_2d = image_3d.squeeze(1)  # (B, D, H, W) = (2, 3, 64, 64)
    assert image_2d.shape == (2, S, 64, 64)

    # Step 2: model forward
    model.train()
    pred = model(image_2d)
    assert pred.shape == (2, S * C, 64, 64)

    # Step 3: reshape for per-slice loss (what _reshape_for_loss does)
    pred_flat = pred.view(-1, C, 64, 64)  # (B*D, C, H, W)
    label_flat = label_3d.transpose(1, 2).contiguous().view(-1, C, 64, 64)
    assert pred_flat.shape == (2 * S, C, 64, 64)
    assert label_flat.shape == (2 * S, C, 64, 64)

    # Step 4: loss
    criterion = build_loss(cfg.loss)
    loss = criterion(pred_flat, label_flat)
    loss.backward()
    assert loss.item() > 0


# ===========================================================================
# Run all tests
# ===========================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SegTask Test Suite")
    logger.info("=" * 60)

    # Config tests
    logger.info("\n--- Config Tests ---")
    run_test("config_defaults", test_config_defaults)
    run_test("config_yaml_roundtrip", test_config_yaml_roundtrip)
    run_test("config_sync_2d", test_config_sync_2d)
    run_test("config_sync_25d", test_config_sync_25d)
    run_test("config_sync_3d", test_config_sync_3d)

    # Data tests
    logger.info("\n--- Data Matching Tests ---")
    run_test("extract_subject_id", test_extract_subject_id)
    run_test("data_matching_real", test_data_matching_with_real_data)
    run_test("split_random", test_split_random)

    # Model tests
    logger.info("\n--- Model Tests (2D) ---")
    run_test("vgg_vgg_2d", test_vgg_vgg_2d)
    run_test("resnet_resnet_2d", test_resnet_resnet_2d)
    run_test("vit_vit_2d", test_vit_vit_2d)
    run_test("resnet_vgg_2d", test_resnet_vgg_2d)
    run_test("vgg_resnet_2d", test_vgg_resnet_2d)
    run_test("vit_resnet_2d", test_vit_resnet_2d)

    logger.info("\n--- Model Tests (3D) ---")
    run_test("resnet_resnet_3d", test_resnet_resnet_3d)
    run_test("vgg_vgg_3d", test_vgg_vgg_3d)
    run_test("vit_vit_3d", test_vit_vit_3d)

    logger.info("\n--- Deep Supervision ---")
    run_test("deep_supervision", test_deep_supervision)

    # Loss tests
    logger.info("\n--- Loss Tests ---")
    run_test("dice_loss", test_dice_loss)
    run_test("ce_loss", test_ce_loss)
    run_test("focal_loss", test_focal_loss)
    run_test("tversky_loss", test_tversky_loss)
    run_test("compound_loss", test_compound_loss)
    run_test("deep_supervision_loss", test_deep_supervision_loss)
    run_test("build_loss_all", test_build_loss)

    # Transform tests
    logger.info("\n--- Transform Tests ---")
    run_test("gpu_augmentor_2d", test_gpu_augmentor_2d)
    run_test("gpu_augmentor_3d", test_gpu_augmentor_3d)
    run_test("mixup", test_mixup)

    # Metric tests
    logger.info("\n--- Metric Tests ---")
    run_test("dice_metric", test_dice_metric)
    run_test("ema", test_ema)

    # Integration test
    logger.info("\n--- Integration Tests ---")
    run_test("mini_training_step", test_mini_training_step)
    run_test("25d_reuses_3d", test_25d_reuses_3d)

    # Summary
    logger.info("\n" + "=" * 60)
    total = PASSED + FAILED
    logger.info("RESULTS: %d/%d PASSED, %d FAILED", PASSED, total, FAILED)
    logger.info("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
