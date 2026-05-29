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
        # `build_scheduler` takes `post_warmup_steps` explicitly since the
        # warmup wrapper was introduced. Pass a reasonable horizon so the
        # base scheduler has enough steps to actually advance its LR.
        sched = build_scheduler(
            optimizer, cfg, steps_per_epoch=5, post_warmup_steps=100)
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

    def test_weight_map_emphasizes_region(self):
        """`_weighted_voxel_mean` normalises by `sum(w)` (see losses.py:
        "loss magnitude is invariant to the total weight"), so a UNIFORM
        scale on the weight map must leave the loss unchanged. A
        NON-UNIFORM map that puts more weight on higher-error voxels must
        increase the reported loss.
        """
        from segtask_v1.losses.losses import BCELoss
        torch.manual_seed(0)
        # Two-voxel pseudo-volume: voxel A has low BCE, voxel B has high.
        # pred logits: A very confident (low loss), B very wrong (high loss).
        pred = torch.tensor([[[[[10.0, -10.0]]]]])         # (1,1,1,1,2)
        target = torch.tensor([[[[[1.0, 1.0]]]]])           # both positive
        loss_fn = BCELoss()

        # Uniform map (any scale) → identical to no-weight baseline.
        wmap_uniform_1x = torch.ones_like(pred)
        wmap_uniform_5x = torch.ones_like(pred) * 5.0
        loss_base = loss_fn(pred, target).item()
        loss_u1 = loss_fn(pred, target, weight_map=wmap_uniform_1x).item()
        loss_u5 = loss_fn(pred, target, weight_map=wmap_uniform_5x).item()
        assert abs(loss_u1 - loss_base) < 1e-6
        assert abs(loss_u5 - loss_base) < 1e-6

        # Non-uniform map emphasising the HIGH-ERROR voxel → loss increases.
        wmap_emph_high = torch.tensor([[[[[1.0, 5.0]]]]])
        loss_emph = loss_fn(pred, target, weight_map=wmap_emph_high).item()
        assert loss_emph > loss_base

        # Non-uniform map emphasising the LOW-ERROR voxel → loss decreases.
        wmap_emph_low = torch.tensor([[[[[5.0, 1.0]]]]])
        loss_deemph = loss_fn(pred, target, weight_map=wmap_emph_low).item()
        assert loss_deemph < loss_base


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
        # `_build_z_weight` was renamed to `_build_1d_weight` (static) when
        # weight construction was unified across z_axis and cubic modes.
        from segtask_v1.predictor import Predictor
        w = Predictor._build_1d_weight(64, mode="gaussian")
        assert w.shape == (64,)
        # Center should have highest weight
        assert w[32] > w[0]
        assert w[32] > w[63]

    def test_prob_to_label(self):
        from segtask_v1.predictor import Predictor
        p = Predictor.__new__(Predictor)
        p.label_values = [0, 1, 2]
        p.num_fg = 2  # required by `_prob_to_label` (added to Predictor state)
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
        p.num_fg = 2
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

    def test_extract_cubic_patch_edge_padding(self):
        """`_extract_cubic_patch` now pads to the EXACT requested size using
        edge-replication (see dataset.py docstring — avoids the anisotropic
        stretch artefact that clip-then-resize produced at volume corners).
        """
        from segtask_v1.data.dataset import _extract_cubic_patch
        vol = np.random.rand(20, 20, 20).astype(np.float32)
        # Center near corner → padded with edge-replicated voxels.
        patch = _extract_cubic_patch(vol, (2, 2, 2), (32, 32, 32))
        assert patch.shape == (32, 32, 32)

    def test_extract_cubic_patch_small_volume(self):
        """Volume smaller than the requested patch on every axis: output
        must still be exactly `size`, edge-replicated outside bounds.
        Because the input is a constant-1 volume, the padded output must
        be all-ones as well (a strict correctness check on the pad mode).
        """
        from segtask_v1.data.dataset import _extract_cubic_patch
        vol = np.ones((10, 10, 10), dtype=np.float32)
        patch = _extract_cubic_patch(vol, (5, 5, 5), (32, 32, 32))
        assert patch.shape == (32, 32, 32)
        assert np.all(patch == 1.0)

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

    def test_z_axis_dataset_extract_size(self):
        """z_axis mode: extract_size must be (pD*oversample, pH, pW).

        Only the depth axis may be oversampled — H, W already land at
        patch_size in-plane since the window slides along z only.
        """
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        ds.patch_size = (16, 128, 128)
        ds.oversample = 1.5
        # Replay the __init__ branch we care about.
        pD, pH, pW = ds.patch_size
        ds.extract_size = (int(round(pD * ds.oversample)), pH, pW)
        assert ds.extract_size == (24, 128, 128), (
            "z_axis oversample must only inflate z, got "
            f"{ds.extract_size}")

    def test_z_axis_dataset_getitem_shape(self):
        """z_axis sample shape: (1, eD, pH, pW) with full-res H,W collapsed."""
        from segtask_v1.data.dataset import SegDataset3D
        # Anisotropic volume — H,W far larger than patch H,W to prove
        # in-plane full-size extraction then resize in one step.
        img = np.random.rand(60, 256, 256).astype(np.float32)
        lbl = np.zeros((60, 256, 256), dtype=np.float32)
        lbl[20:40, 100:150, 100:150] = 1.0

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            # oversample=1.25 → eD=20 from pD=16.
            ds = SegDataset3D(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 64, 64),
                aug_oversample_ratio=1.25,
                samples_per_volume=2, cache_enabled=False)
            eD, eH, eW = ds.extract_size
            assert (eD, eH, eW) == (20, 64, 64)
            sample = ds[0]
            assert sample["image"].shape == (1, 20, 64, 64)
            assert sample["label"].shape == (1, 20, 64, 64)

    def test_z_axis_dataset_no_oversample(self):
        """oversample=1.0 → extract_size == patch_size exactly on all axes."""
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        ds.patch_size = (32, 96, 96)
        ds.oversample = 1.0
        pD, pH, pW = ds.patch_size
        ds.extract_size = (int(round(pD * ds.oversample)), pH, pW)
        assert ds.extract_size == (32, 96, 96)

    def test_z_axis_multires_shape_default_is_backward_compatible(self):
        """Default multi_res_scales=[1.0] must produce the legacy 1-channel shape."""
        from segtask_v1.data.dataset import SegDataset3D
        img = np.random.rand(60, 128, 128).astype(np.float32)
        lbl = np.zeros((60, 128, 128), dtype=np.float32)
        lbl[20:40, 40:80, 40:80] = 1.0

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            ds = SegDataset3D(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 64, 64),
                samples_per_volume=1, cache_enabled=False)
            sample = ds[0]
            # C_res == 1 (default [1.0]) — identical to pre-multires shape.
            assert sample["image"].shape == (1, 16, 64, 64)
            assert sample["label"].shape == (1, 16, 64, 64)

    def test_z_axis_multires_shape_three_scales(self):
        """multi_res_scales=[1.0, 1.5, 2.0] → image/label (3, eD, pH, pW)."""
        from segtask_v1.data.dataset import SegDataset3D
        img = np.random.rand(80, 128, 128).astype(np.float32)
        lbl = np.zeros((80, 128, 128), dtype=np.float32)
        lbl[30:50, 40:80, 40:80] = 1.0

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            ds = SegDataset3D(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 64, 64),
                aug_oversample_ratio=1.0,
                multi_res_scales=[1.0, 1.5, 2.0],
                samples_per_volume=1, cache_enabled=False)
            sample = ds[0]
            assert sample["image"].shape == (3, 16, 64, 64)
            assert sample["label"].shape == (3, 16, 64, 64)

    def test_z_axis_multires_edge_padded_preserves_exact_depth(self):
        """Scale > 1 extraction at volume boundary must return EXACTLY D_patch
        slices (edge-padded), preserving physical z-FOV vs. clamp-and-stretch.
        """
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        # Small volume so scale=2 around center exceeds bounds.
        img = np.arange(20 * 4 * 4, dtype=np.float32).reshape(20, 4, 4)
        lbl = np.zeros((20, 4, 4), dtype=np.float32)

        # z_center=1 with D_patch=16 → lo=-7, hi=9 → needs 7 slices of
        # edge-padding BEFORE the volume start.
        img_p, lbl_p = ds._extract_z_patch_padded(img, lbl, 1, 16)
        assert img_p.shape == (16, 4, 4)
        assert lbl_p.shape == (16, 4, 4)
        # First 7 slices must all equal img[0] (edge-replicated).
        for i in range(7):
            np.testing.assert_array_equal(img_p[i], img[0])
        # Slice 7 should map to img[0], slice 8 to img[1], …
        np.testing.assert_array_equal(img_p[7], img[0])
        np.testing.assert_array_equal(img_p[8], img[1])

        # z_center near top: z=18, D_patch=16 → lo=10, hi=26 → 6 slices
        # of edge-padding AFTER the volume.
        img_p2, _ = ds._extract_z_patch_padded(img, lbl, 18, 16)
        assert img_p2.shape == (16, 4, 4)
        for i in range(6):
            np.testing.assert_array_equal(img_p2[-1 - i], img[-1])

    def test_z_axis_multires_scale1_matches_legacy(self):
        """multi_res_scales=[1.0] must yield IDENTICAL image as the legacy
        single-res path (confirms channel-0 bit-compatibility).
        """
        from segtask_v1.data.dataset import SegDataset3D
        np.random.seed(0)
        img = np.random.rand(50, 64, 64).astype(np.float32)
        lbl = np.zeros((50, 64, 64), dtype=np.float32)
        lbl[15:35, 20:40, 20:40] = 1.0

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            kwargs = dict(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 32, 32),
                foreground_oversample_ratio=0.0,  # deterministic z sample
                samples_per_volume=1, cache_enabled=False, is_train=False)

            # Seed RNG before each call so `_sample_z` picks the same z.
            np.random.seed(123)
            ds_legacy = SegDataset3D(multi_res_scales=[1.0], **kwargs)
            legacy = ds_legacy[0]["image"].numpy()

            np.random.seed(123)
            ds_multi = SegDataset3D(
                multi_res_scales=[1.0, 1.5], **kwargs)
            multi = ds_multi[0]["image"].numpy()

            # Channel 0 of the multi-res output must match the legacy single-res.
            np.testing.assert_allclose(multi[0], legacy[0], rtol=0, atol=0)

    def test_predictor_build_z_window_input_single_res(self):
        """Single-res (`[1.0]`): (1, pD, pH, pW), legacy resize path."""
        from segtask_v1.predictor import Predictor
        p = Predictor.__new__(Predictor)
        p.patch_D, p.patch_H, p.patch_W = 8, 16, 16
        p.multi_res_scales = [1.0]
        vol = np.arange(40 * 24 * 24, dtype=np.float32).reshape(40, 24, 24)
        # Normal window
        out = p._build_z_window_input(vol, 10, 18)
        assert out.shape == (1, 8, 16, 16)
        # Tail window (short)
        out_tail = p._build_z_window_input(vol, 36, 40)
        assert out_tail.shape == (1, 8, 16, 16)

    def test_predictor_build_z_window_input_multi_res(self):
        """Multi-res: (C_res, pD, pH, pW); scale>1 uses edge-padded z extraction
        centered on window center, preserving physical z-FOV at boundaries."""
        from segtask_v1.predictor import Predictor
        p = Predictor.__new__(Predictor)
        p.patch_D, p.patch_H, p.patch_W = 8, 16, 16
        p.multi_res_scales = [1.0, 1.5, 2.0]
        vol = np.random.rand(40, 24, 24).astype(np.float32)

        # Window fully inside volume
        out = p._build_z_window_input(vol, 10, 18)
        assert out.shape == (3, 8, 16, 16)

        # Window at top boundary — scale=2 needs D_s=16 slices centered at
        # z_center=(0+8)//2=4, so lo=-4, hi=12 → 4 slices of edge pad.
        out_top = p._build_z_window_input(vol, 0, 8)
        assert out_top.shape == (3, 8, 16, 16)

        # Window at bottom boundary — verify no exceptions.
        out_bot = p._build_z_window_input(vol, 32, 40)
        assert out_bot.shape == (3, 8, 16, 16)

    def test_predictor_z_window_scale1_matches_legacy(self):
        """Channel 0 of multi-res output must be pixel-identical to the
        scale-1.0-only path for an interior window (where boundary padding
        doesn't enter the picture)."""
        from segtask_v1.predictor import Predictor
        np.random.seed(7)
        vol = np.random.rand(40, 24, 24).astype(np.float32)

        p_single = Predictor.__new__(Predictor)
        p_single.patch_D, p_single.patch_H, p_single.patch_W = 8, 16, 16
        p_single.multi_res_scales = [1.0]

        p_multi = Predictor.__new__(Predictor)
        p_multi.patch_D, p_multi.patch_H, p_multi.patch_W = 8, 16, 16
        p_multi.multi_res_scales = [1.0, 1.5, 2.0]

        out_single = p_single._build_z_window_input(vol, 12, 20)
        out_multi  = p_multi._build_z_window_input(vol, 12, 20)
        np.testing.assert_allclose(out_multi[0], out_single[0], rtol=0, atol=0)

    def test_whole_dataset_shape(self):
        """Whole-volume mode: each sample is the full volume resized to
        extract_size (no cropping, no sliding). Output shape matches the
        single-channel z_axis contract for interoperability.
        """
        from segtask_v1.data.dataset import SegDataset3DWhole
        img = np.random.rand(70, 90, 90).astype(np.float32)
        lbl = np.zeros((70, 90, 90), dtype=np.float32)
        lbl[20:50, 30:60, 30:60] = 1.0

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            ds = SegDataset3DWhole(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 32, 32),
                aug_oversample_ratio=1.0,
                samples_per_volume=3, cache_enabled=False)
            # __len__ = num_volumes * samples_per_volume
            assert len(ds) == 3
            sample = ds[0]
            assert sample["image"].shape == (1, 16, 32, 32)
            assert sample["label"].shape == (1, 16, 32, 32)

    def test_whole_dataset_oversample(self):
        """Oversample inflates all 3 axes so the trainer can center-crop."""
        from segtask_v1.data.dataset import SegDataset3DWhole
        img = np.random.rand(30, 40, 40).astype(np.float32)
        lbl = np.zeros((30, 40, 40), dtype=np.float32)

        import tempfile, os, nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "test.nii.gz")
            lbl_path = os.path.join(td, "test_lbl.nii.gz")
            nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), np.eye(4)), lbl_path)

            ds = SegDataset3DWhole(
                image_paths=[img_path], label_paths=[lbl_path],
                label_values=[0, 1], patch_size=(16, 16, 16),
                aug_oversample_ratio=1.5,
                samples_per_volume=1, cache_enabled=False)
            assert ds.extract_size == (24, 24, 24)
            sample = ds[0]
            assert sample["image"].shape == (1, 24, 24, 24)

    def test_whole_mode_config_rejects_multires(self):
        """Whole-volume mode must reject multi_res_scales beyond [1.0]."""
        from segtask_v1.config import Config
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.data.patch_mode = "whole"
        cfg.data.multi_res_scales = [1.0, 1.5]
        cfg.sync()
        with pytest.raises(AssertionError, match="whole-volume"):
            cfg.validate()

    def test_whole_mode_config_single_res_passes(self):
        """Whole-volume + default [1.0] validates cleanly."""
        from segtask_v1.config import Config
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.data.patch_mode = "whole"
        cfg.sync()
        cfg.validate()  # must not raise
        assert cfg.model.in_channels == 1

    def test_config_z_axis_multires_allowed(self):
        """z_axis + multi_res_scales>1 should now validate and auto-sync
        model.in_channels = len(scales)."""
        from segtask_v1.config import Config
        cfg = Config()
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.data.patch_mode = "z_axis"
        cfg.data.multi_res_scales = [1.0, 1.5, 2.0]
        cfg.sync()
        cfg.validate()  # must not raise
        assert cfg.model.in_channels == 3

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
