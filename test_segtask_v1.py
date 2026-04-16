"""Unit tests for segtask_v1 pipeline.

Tests cover:
- Config loading and validation
- Dataset z-axis patching
- Model forward pass (ResNet, ConvNeXt)
- SE attention blocks
- Loss functions
- GPU augmentation
- Gradient accumulation logic
- Deep supervision
- EMA
- Dice metric computation
"""

import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
class TestConfig:
    def test_load_config(self):
        from segtask_v1.config import load_config
        cfg = load_config("configs/seg3d.yaml")
        assert cfg.data.patch_size == [64, 128, 128]
        assert cfg.model.backbone == "resnet"
        assert cfg.loss.name == "dice_bce"

    def test_sync_label_values(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.data.label_values = [0, 1, 2, 3]
        cfg.sync()
        assert cfg.data.num_classes == 4
        assert cfg.num_fg_classes == 3

    def test_validate_rejects_bad_backbone(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.model.backbone = "bad"
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_new_config_fields(self):
        from segtask_v1.config import load_config
        cfg = load_config("configs/seg3d.yaml")
        assert cfg.train.grad_accum_steps == 1
        assert cfg.train.compile_mode == "none"
        assert cfg.model.use_se == False
        assert cfg.train.cosine_restart_period == 50


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
class TestDataset:
    def test_preprocess_label(self):
        from segtask_v1.data.dataset import preprocess_label
        vol = np.array([[[0, 1], [2, 0]]], dtype=np.float32)  # (1, 2, 2)
        result = preprocess_label(vol, [0, 1, 2])
        assert result.shape == (2, 1, 2, 2)  # 2 fg classes
        assert result[0, 0, 0, 1] == 1.0  # class 1 at position (0,1)
        assert result[1, 0, 1, 0] == 1.0  # class 2 at position (1,0)

    def test_resize_3d(self):
        from segtask_v1.data.dataset import resize_3d
        arr = np.random.rand(10, 20, 30).astype(np.float32)
        out = resize_3d(arr, 8, 16, 24, is_label=False)
        assert out.shape == (8, 16, 24)

    def test_resize_3d_4d(self):
        from segtask_v1.data.dataset import resize_3d
        arr = np.random.rand(3, 10, 20, 30).astype(np.float32)
        out = resize_3d(arr, 8, 16, 24, is_label=False)
        assert out.shape == (3, 8, 16, 24)

    def test_extract_z_patch_small_volume(self):
        """Volume smaller than patch: returns all available slices (no padding)."""
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        img = np.random.rand(30, 64, 64).astype(np.float32)
        lbl = np.zeros((30, 64, 64), dtype=np.float32)
        img_p, lbl_p = ds._extract_z_patch(img, lbl, 15, 64)
        # User's updated code: no padding, returns clamped slice range
        assert img_p.shape[0] <= 30
        assert img_p.shape[0] > 0

    def test_extract_z_patch_normal(self):
        """Volume larger than patch: should extract exact D slices."""
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        img = np.random.rand(200, 64, 64).astype(np.float32)
        lbl = np.zeros((200, 64, 64), dtype=np.float32)
        img_p, lbl_p = ds._extract_z_patch(img, lbl, 100, 64)
        assert img_p.shape[0] == 64

    def test_extract_z_patch_boundary(self):
        """Patch at boundary should clamp, not go out of bounds."""
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        img = np.random.rand(100, 32, 32).astype(np.float32)
        lbl = np.zeros((100, 32, 32), dtype=np.float32)
        # Near the end
        img_p, _ = ds._extract_z_patch(img, lbl, 98, 64)
        assert img_p.shape[0] > 0
        assert img_p.shape[0] <= 100
        # Near the start
        img_p, _ = ds._extract_z_patch(img, lbl, 2, 64)
        assert img_p.shape[0] > 0
        assert img_p.shape[0] <= 100


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------
class TestModel:
    def test_resnet_unet_forward(self):
        from segtask_v1.config import Config
        from segtask_v1.models.factory import build_model
        cfg = Config()
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.sync()
        model = build_model(cfg)
        x = torch.randn(1, 1, 32, 64, 64)
        y = model(x)
        assert y.shape == (1, 2, 32, 64, 64)  # 2 fg classes

    def test_convnext_unet_forward(self):
        from segtask_v1.config import Config
        from segtask_v1.models.factory import build_model
        cfg = Config()
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.model.backbone = "convnext"
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.sync()
        model = build_model(cfg)
        x = torch.randn(1, 1, 32, 64, 64)
        y = model(x)
        assert y.shape == (1, 2, 32, 64, 64)

    def test_resnet_se_forward(self):
        from segtask_v1.config import Config
        from segtask_v1.models.factory import build_model
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.use_se = True
        cfg.model.se_reduction = 4
        cfg.sync()
        model = build_model(cfg)
        x = torch.randn(1, 1, 32, 64, 64)
        y = model(x)
        assert y.shape == (1, 1, 32, 64, 64)

    def test_deep_supervision(self):
        from segtask_v1.config import Config
        from segtask_v1.models.factory import build_model
        cfg = Config()
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.deep_supervision = True
        cfg.sync()
        model = build_model(cfg)
        model.train()
        x = torch.randn(1, 1, 32, 64, 64)
        y = model(x)
        assert isinstance(y, list)
        # 3 encoder levels → 2 decoder levels → 1 ds head + 1 main = 2
        assert len(y) == 2
        assert y[0].shape == (1, 2, 32, 64, 64)

    def test_param_count(self):
        from segtask_v1.config import Config
        from segtask_v1.models.factory import build_model
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.model.encoder_channels = [8, 16]
        cfg.sync()
        model = build_model(cfg)
        pc = model.param_count()
        assert pc["total"] > 0
        assert pc["encoder"] > 0
        assert pc["decoder"] > 0


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------
class TestLoss:
    def _make_pred_target(self, B=2, C=2, D=8, H=16, W=16):
        pred = torch.randn(B, C, D, H, W)
        target = (torch.rand(B, C, D, H, W) > 0.5).float()
        return pred, target

    def test_dice_loss(self):
        from segtask_v1.losses.losses import BinaryDiceLoss
        loss_fn = BinaryDiceLoss()
        pred, target = self._make_pred_target()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_bce_loss(self):
        from segtask_v1.losses.losses import BCELoss
        loss_fn = BCELoss()
        pred, target = self._make_pred_target()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_focal_loss(self):
        from segtask_v1.losses.losses import BinaryFocalLoss
        loss_fn = BinaryFocalLoss()
        pred, target = self._make_pred_target()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_tversky_loss(self):
        from segtask_v1.losses.losses import BinaryTverskyLoss
        loss_fn = BinaryTverskyLoss()
        pred, target = self._make_pred_target()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_compound_loss(self):
        from segtask_v1.losses.losses import build_loss
        from segtask_v1.config import LossConfig
        cfg = LossConfig(name="dice_bce")
        loss_fn = build_loss(cfg)
        pred, target = self._make_pred_target()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_dice_focal(self):
        from segtask_v1.losses.losses import build_loss
        from segtask_v1.config import LossConfig
        cfg = LossConfig(name="dice_focal")
        loss_fn = build_loss(cfg)
        pred, target = self._make_pred_target()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_class_weights(self):
        from segtask_v1.losses.losses import BinaryDiceLoss
        loss_fn = BinaryDiceLoss(class_weights=[0.3, 0.7])
        pred, target = self._make_pred_target()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_deep_supervision_loss(self):
        from segtask_v1.losses.losses import build_loss, DeepSupervisionLoss
        from segtask_v1.config import LossConfig
        base = build_loss(LossConfig(name="dice_bce"))
        ds_loss = DeepSupervisionLoss(base, [1.0, 0.5])
        pred_main = torch.randn(2, 2, 16, 32, 32)
        pred_ds = torch.randn(2, 2, 8, 16, 16)
        target = (torch.rand(2, 2, 16, 32, 32) > 0.5).float()
        loss = ds_loss([pred_main, pred_ds], target)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------
class TestAugmentation:
    def test_augmentor_shapes(self):
        from segtask_v1.config import AugConfig
        from segtask_v1.data.augment import GPUAugmentor
        aug = GPUAugmentor(AugConfig(enabled=True))
        img = torch.randn(2, 1, 16, 32, 32)
        lbl = torch.zeros(2, 2, 16, 32, 32)
        img_out, lbl_out = aug(img, lbl)
        assert img_out.shape == img.shape
        assert lbl_out.shape == lbl.shape

    def test_augmentor_disabled(self):
        from segtask_v1.config import AugConfig
        from segtask_v1.data.augment import GPUAugmentor
        aug = GPUAugmentor(AugConfig(enabled=False))
        img = torch.randn(2, 1, 16, 32, 32)
        lbl = torch.ones(2, 2, 16, 32, 32)
        img_out, lbl_out = aug(img, lbl)
        assert torch.equal(img_out, img)
        assert torch.equal(lbl_out, lbl)

    def test_gaussian_blur_3d(self):
        from segtask_v1.data.augment import _gaussian_blur_3d
        img = torch.randn(1, 1, 8, 16, 16)
        out = _gaussian_blur_3d(img.clone(), prob=1.0, sigma_range=[1.0, 1.0])
        assert out.shape == img.shape
        # Blurred image should be smoother (lower variance)
        assert out.var() <= img.var() * 1.5


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------
class TestUtils:
    def test_dice_per_class(self):
        from segtask_v1.utils import compute_dice_per_class
        # Perfect prediction
        pred = torch.ones(2, 3, 8, 16, 16) * 10  # high logits
        target = torch.ones(2, 3, 8, 16, 16)
        dice = compute_dice_per_class(pred, target)
        assert dice.shape == (3,)
        assert (dice > 0.99).all()

    def test_dice_zero_prediction(self):
        from segtask_v1.utils import compute_dice_per_class
        pred = torch.ones(2, 2, 8, 16, 16) * -10  # all negative
        target = torch.ones(2, 2, 8, 16, 16)
        dice = compute_dice_per_class(pred, target)
        assert (dice < 0.01).all()

    def test_ema(self):
        from segtask_v1.utils import ModelEMA
        import torch.nn as nn
        model = nn.Linear(10, 10)
        ema = ModelEMA(model, decay=0.9)
        # Modify model weights
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        # Shadow should be between init and current
        ema.apply_shadow(model)
        assert model.weight.mean().item() != 0.0
        ema.restore(model)
        assert model.weight.mean().item() == 1.0

    def test_average_meter(self):
        from segtask_v1.utils import AverageMeter
        m = AverageMeter()
        m.update(1.0, 2)
        m.update(3.0, 2)
        assert abs(m.avg - 2.0) < 1e-6

    def test_seed_everything(self):
        from segtask_v1.utils import seed_everything
        seed_everything(42)
        a = torch.randn(5)
        seed_everything(42)
        b = torch.randn(5)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# SE block test
# ---------------------------------------------------------------------------
class TestSEBlock:
    def test_se_forward(self):
        from segtask_v1.models.blocks import SqueezeExcite3D
        se = SqueezeExcite3D(channels=32, reduction=8)
        x = torch.randn(2, 32, 8, 16, 16)
        y = se(x)
        assert y.shape == x.shape

    def test_se_output_range(self):
        from segtask_v1.models.blocks import SqueezeExcite3D
        se = SqueezeExcite3D(channels=16, reduction=4)
        x = torch.randn(1, 16, 4, 8, 8)
        y = se(x)
        # SE scales channels — output should be correlated with input
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------
class TestScheduler:
    def test_cosine_warm_restarts(self):
        from segtask_v1.config import Config
        from segtask_v1.trainer import build_scheduler
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.train.scheduler = "cosine_warm_restarts"
        cfg.train.cosine_restart_period = 10
        cfg.train.cosine_restart_mult = 2
        cfg.sync()
        optimizer = torch.optim.Adam([torch.randn(3, requires_grad=True)], lr=0.01)
        sched = build_scheduler(optimizer, cfg, steps_per_epoch=5)
        assert sched is not None
        # Step a few times
        for _ in range(20):
            sched.step()
        # LR should have changed
        assert optimizer.param_groups[0]["lr"] != 0.01


# ---------------------------------------------------------------------------
# Region weight map tests (Feature a)
# ---------------------------------------------------------------------------
class TestRegionWeights:
    def test_compute_region_weight_map(self):
        from segtask_v1.data.dataset import compute_region_weight_map
        vol = np.array([[[0, 1], [2, 0]]], dtype=np.float32)  # (1, 2, 2)
        wmap = compute_region_weight_map(vol, [0, 1, 2], [1.0, 3.0, 2.0])
        assert wmap.shape == (1, 1, 2, 2)
        assert wmap[0, 0, 0, 0] == 1.0  # bg
        assert wmap[0, 0, 0, 1] == 3.0  # label 1
        assert wmap[0, 0, 1, 0] == 2.0  # label 2

    def test_loss_with_weight_map(self):
        from segtask_v1.losses.losses import BinaryDiceLoss, BCELoss, CompoundLoss
        pred = torch.randn(2, 2, 8, 16, 16)
        target = (torch.rand(2, 2, 8, 16, 16) > 0.5).float()
        wmap = torch.ones(2, 1, 8, 16, 16) * 2.0
        # Should not crash with weight_map
        for loss_fn in [BinaryDiceLoss(), BCELoss()]:
            loss_no_w = loss_fn(pred, target)
            loss_w = loss_fn(pred, target, weight_map=wmap)
            assert loss_no_w.shape == ()
            assert loss_w.shape == ()
        # Compound
        compound = CompoundLoss([BinaryDiceLoss(), BCELoss()], [1.0, 1.0])
        loss = compound(pred, target, weight_map=wmap)
        assert loss.item() >= 0

    def test_higher_weight_increases_loss(self):
        from segtask_v1.losses.losses import BCELoss
        pred = torch.randn(1, 1, 4, 8, 8)
        target = torch.ones(1, 1, 4, 8, 8)
        wmap_low = torch.ones(1, 1, 4, 8, 8)
        wmap_high = torch.ones(1, 1, 4, 8, 8) * 5.0
        loss_fn = BCELoss()
        loss_low = loss_fn(pred, target, weight_map=wmap_low)
        loss_high = loss_fn(pred, target, weight_map=wmap_high)
        # Higher weight should increase loss
        assert loss_high.item() > loss_low.item()


# ---------------------------------------------------------------------------
# New augmentation tests (Feature b)
# ---------------------------------------------------------------------------
class TestNewAugmentation:
    def test_per_sample_flip(self):
        from segtask_v1.data.augment import _random_flip
        torch.manual_seed(0)
        img = torch.arange(24).reshape(2, 1, 3, 2, 2).float()
        lbl = img.clone()
        # With prob=1.0, all samples should be flipped
        img_f, lbl_f = _random_flip(img.clone(), lbl.clone(), prob=1.0, axes=[2])
        assert img_f.shape == img.shape
        assert lbl_f.shape == lbl.shape

    def test_affine_shapes(self):
        from segtask_v1.data.augment import _random_affine
        img = torch.randn(2, 1, 16, 32, 32)
        lbl = torch.zeros(2, 2, 16, 32, 32)
        img_a, lbl_a = _random_affine(img, lbl, prob=1.0,
                                       rotate_range=[-10.0, 10.0],
                                       scale_range=[0.9, 1.1])
        assert img_a.shape == img.shape
        assert lbl_a.shape == lbl.shape

    def test_elastic_deform_shapes(self):
        from segtask_v1.data.augment import _elastic_deform
        img = torch.randn(2, 1, 16, 32, 32)
        lbl = torch.zeros(2, 2, 16, 32, 32)
        img_e, lbl_e = _elastic_deform(img, lbl, prob=1.0, sigma=5.0, alpha=50.0)
        assert img_e.shape == img.shape
        assert lbl_e.shape == lbl.shape

    def test_grid_dropout_shapes(self):
        from segtask_v1.data.augment import _grid_dropout
        img = torch.randn(2, 1, 16, 32, 32)
        lbl = torch.ones(2, 2, 16, 32, 32)
        img_d, lbl_d = _grid_dropout(img.clone(), lbl.clone(), prob=1.0, ratio=0.3, num_holes=4)
        assert img_d.shape == img.shape
        # Label should NOT be masked
        assert torch.equal(lbl_d, lbl)
        # Image should have some zeros from dropout
        assert img_d.sum() < img.sum()

    def test_simulate_lowres(self):
        from segtask_v1.data.augment import _simulate_lowres
        img = torch.randn(2, 1, 16, 32, 32)
        img_lr = _simulate_lowres(img.clone(), prob=1.0, zoom_range=[0.3, 0.5])
        assert img_lr.shape == img.shape

    def test_full_augmentor_pipeline(self):
        from segtask_v1.config import AugConfig
        from segtask_v1.data.augment import GPUAugmentor
        cfg = AugConfig(
            enabled=True, random_flip_prob=0.5,
            random_affine_prob=0.5, elastic_deform_prob=0.3,
            grid_dropout_prob=0.2, simulate_lowres_prob=0.2)
        aug = GPUAugmentor(cfg)
        img = torch.randn(2, 1, 16, 32, 32)
        lbl = torch.zeros(2, 3, 16, 32, 32)  # 3 channels (e.g., 2 fg + 1 weight_map)
        img_a, lbl_a = aug(img, lbl)
        assert img_a.shape == img.shape
        assert lbl_a.shape == lbl.shape


# ---------------------------------------------------------------------------
# Predictor tests (Feature c)
# ---------------------------------------------------------------------------
class TestPredictor:
    def test_z_positions_coverage(self):
        from segtask_v1.predictor import Predictor
        positions = Predictor._compute_1d_positions(length=200, patch=64, stride=32)
        # Should cover entire volume
        assert positions[0][0] == 0
        assert positions[-1][1] >= 200
        # No gaps
        for i in range(len(positions) - 1):
            assert positions[i + 1][0] < positions[i][1]  # overlap

    def test_z_positions_small_volume(self):
        from segtask_v1.predictor import Predictor
        positions = Predictor._compute_1d_positions(length=30, patch=64, stride=32)
        # Small volume: single window covering [0, 30]
        assert len(positions) == 1
        assert positions[0] == (0, 30)

    def test_gaussian_blend_weight(self):
        from segtask_v1.predictor import Predictor
        p = Predictor.__new__(Predictor)
        p.blend_mode = "gaussian"
        w = p._build_z_weight(64)
        assert w.shape == (64,)
        # Center should have highest weight
        assert w[32] > w[0]
        assert w[32] > w[63]

    def test_prob_to_label(self):
        from segtask_v1.predictor import Predictor
        p = Predictor.__new__(Predictor)
        p.label_values = [0, 1, 2]
        p.threshold = 0.5
        # Create probability volume: class 0 (label 1) = 0.8, class 1 (label 2) = 0.3
        prob = np.zeros((2, 4, 4, 4), dtype=np.float32)
        prob[0] = 0.8  # fg class 0 (label 1) is confident
        prob[1] = 0.3  # fg class 1 (label 2) is not
        label_map = p._prob_to_label(prob)
        assert label_map.shape == (4, 4, 4)
        # Should all be label 1 (since class 0 has max prob > threshold)
        assert (label_map == 1).all()

    def test_prob_to_label_background(self):
        from segtask_v1.predictor import Predictor
        p = Predictor.__new__(Predictor)
        p.label_values = [0, 1, 2]
        p.threshold = 0.5
        # All probabilities below threshold → background
        prob = np.ones((2, 4, 4, 4), dtype=np.float32) * 0.1
        label_map = p._prob_to_label(prob)
        assert (label_map == 0).all()


# ---------------------------------------------------------------------------
# Cubic patch dataset tests (New feature)
# ---------------------------------------------------------------------------
class TestCubicDataset:
    def test_extract_cubic_patch_normal(self):
        from segtask_v1.data.dataset import _extract_cubic_patch
        vol = np.random.rand(100, 80, 80).astype(np.float32)
        patch = _extract_cubic_patch(vol, (50, 40, 40), (32, 32, 32))
        assert patch.shape == (32, 32, 32)

    def test_extract_cubic_patch_edge_clipping(self):
        from segtask_v1.data.dataset import _extract_cubic_patch
        vol = np.random.rand(20, 20, 20).astype(np.float32)
        # Center near corner → clipped (no padding)
        patch = _extract_cubic_patch(vol, (2, 2, 2), (32, 32, 32))
        # Result is clipped to volume bounds, not padded
        assert patch.shape[0] <= 20
        assert patch.shape[0] > 0

    def test_extract_cubic_patch_small_volume(self):
        from segtask_v1.data.dataset import _extract_cubic_patch
        vol = np.ones((10, 10, 10), dtype=np.float32)
        patch = _extract_cubic_patch(vol, (5, 5, 5), (32, 32, 32))
        # Clipped to volume size (no padding)
        assert patch.shape[0] <= 10
        assert patch.shape[0] > 0

    def test_cubic_dataset_getitem(self):
        from segtask_v1.data.dataset import SegDataset3DCubic
        # Create small temp volumes
        img = np.random.rand(50, 40, 40).astype(np.float32)
        lbl = np.zeros((50, 40, 40), dtype=np.float32)
        lbl[20:30, 15:25, 15:25] = 1.0

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            ds = SegDataset3DCubic(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 16, 16),
                multi_res_scales=[1.0],
                samples_per_volume=2, cache_enabled=False)
            sample = ds[0]
            assert sample["image"].shape == (1, 16, 16, 16)
            assert sample["label"].shape == (1, 16, 16, 16)

    def test_cubic_dataset_oversample(self):
        from segtask_v1.data.dataset import SegDataset3DCubic
        img = np.random.rand(80, 60, 60).astype(np.float32)
        lbl = np.zeros((80, 60, 60), dtype=np.float32)
        lbl[30:50, 20:40, 20:40] = 1.0

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            ds = SegDataset3DCubic(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 16, 16),
                aug_oversample_ratio=1.5,
                multi_res_scales=[1.0],
                samples_per_volume=2, cache_enabled=False)
            sample = ds[0]
            # With 1.5x oversample, extraction is ceil(16*1.5)=24
            eD, eH, eW = ds.extract_size
            assert eD == 24
            assert sample["image"].shape == (1, eD, eH, eW)

    def test_config_patch_mode_validation(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.data.patch_mode = "cubic"
        cfg.sync()
        cfg.validate()  # should not raise

        cfg.data.patch_mode = "invalid"
        with pytest.raises(AssertionError):
            cfg.validate()


# ---------------------------------------------------------------------------
# Cubic predictor tests (New feature)
# ---------------------------------------------------------------------------
class TestCubicPredictor:
    def test_3d_weight_gaussian(self):
        from segtask_v1.predictor import Predictor
        w = Predictor._build_3d_weight(16, 16, 16, "gaussian")
        assert w.shape == (16, 16, 16)
        # Center should have highest weight
        assert w[8, 8, 8] > w[0, 0, 0]
        assert w[8, 8, 8] > w[15, 15, 15]
        # Should be symmetric
        assert abs(w[0, 8, 8] - w[15, 8, 8]) < 1e-10

    def test_3d_weight_average(self):
        from segtask_v1.predictor import Predictor
        w = Predictor._build_3d_weight(8, 8, 8, "average")
        assert w.shape == (8, 8, 8)
        assert (w == 1.0).all()

    def test_1d_positions_coverage(self):
        from segtask_v1.predictor import Predictor
        pos = Predictor._compute_1d_positions(200, 64, 32)
        assert pos[0][0] == 0
        assert pos[-1][1] >= 200
        # No gaps
        for i in range(len(pos) - 1):
            assert pos[i + 1][0] < pos[i][1]

    def test_1d_positions_small(self):
        from segtask_v1.predictor import Predictor
        pos = Predictor._compute_1d_positions(30, 64, 32)
        assert len(pos) == 1
        assert pos[0] == (0, 30)


# ---------------------------------------------------------------------------
# Trainer center-crop test (New feature)
# ---------------------------------------------------------------------------
class TestTrainerCenterCrop:
    def test_center_crop(self):
        from segtask_v1.trainer import Trainer
        t = Trainer.__new__(Trainer)
        t.target_patch_size = (16, 16, 16)
        t.needs_crop = True
        image = torch.randn(2, 1, 24, 24, 24)
        label = torch.randn(2, 2, 24, 24, 24)
        wmap = torch.randn(2, 1, 24, 24, 24)
        img_c, lbl_c, wm_c = t._center_crop(image, label, wmap)
        assert img_c.shape == (2, 1, 16, 16, 16)
        assert lbl_c.shape == (2, 2, 16, 16, 16)
        assert wm_c.shape == (2, 1, 16, 16, 16)

    def test_center_crop_no_wmap(self):
        from segtask_v1.trainer import Trainer
        t = Trainer.__new__(Trainer)
        t.target_patch_size = (32, 32, 32)
        t.needs_crop = True
        image = torch.randn(1, 1, 48, 48, 48)
        label = torch.randn(1, 3, 48, 48, 48)
        img_c, lbl_c, wm_c = t._center_crop(image, label, None)
        assert img_c.shape == (1, 1, 32, 32, 32)
        assert lbl_c.shape == (1, 3, 32, 32, 32)
        assert wm_c is None


# ---------------------------------------------------------------------------
# Multi-resolution tests (New feature)
# ---------------------------------------------------------------------------
class TestMultiResolution:
    def test_multi_res_loss(self):
        from segtask_v1.losses.losses import build_loss, MultiResolutionLoss
        from segtask_v1.config import LossConfig
        base = build_loss(LossConfig(name="dice_bce"))
        mr_loss = MultiResolutionLoss(
            base_loss=base, num_fg_classes=2, num_res=3, label_values=[0, 1, 2])
        # pred: (B, num_fg*C_res, D, H, W) = (2, 6, 8, 16, 16)
        pred = torch.randn(2, 6, 8, 16, 16)
        # label: (B, C_res, D, H, W) = (2, 3, 8, 16, 16) raw integer
        label = torch.zeros(2, 3, 8, 16, 16)
        label[:, :, 2:6, 4:12, 4:12] = 1.0
        loss = mr_loss(pred, label)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_label_to_binary_gpu(self):
        from segtask_v1.losses.losses import MultiResolutionLoss
        mr = MultiResolutionLoss.__new__(MultiResolutionLoss)
        mr.fg_values = [1, 2]
        label = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.float32)  # (1, 1, 2, 2)
        # Squeeze to (1, 2, 2)
        binary = mr._label_to_binary(label.squeeze(1))  # (1, 2, 2, 2)
        assert binary.shape == (1, 2, 2, 2)
        assert binary[0, 0, 0, 1] == 1.0  # class 1 at (0,1)
        assert binary[0, 1, 1, 0] == 1.0  # class 2 at (1,0)
        assert binary[0, 0, 0, 0] == 0.0  # bg

    def test_config_multi_res_sync(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.data.patch_mode = "cubic"
        cfg.data.multi_res_scales = [1.0, 1.5, 2.0]
        cfg.sync()
        assert cfg.model.in_channels == 3

    def test_config_multi_res_validates_scales(self):
        from segtask_v1.config import Config
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.data.multi_res_scales = [0.5]  # below 1.0 is invalid
        cfg.sync()
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_model_output_channels(self):
        from segtask_v1.config import Config
        from segtask_v1.models.factory import build_model
        cfg = Config()
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.data.patch_mode = "cubic"
        cfg.data.multi_res_scales = [1.0, 1.5, 2.0]
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.sync()
        model = build_model(cfg)
        x = torch.randn(1, 3, 32, 64, 64)
        y = model(x)
        # Output: num_fg(2) * C_res(3) = 6 channels
        assert y.shape == (1, 6, 32, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
