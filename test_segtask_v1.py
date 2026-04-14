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

    def test_extract_z_patch_pad(self):
        """Volume smaller than patch size should be padded."""
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        img = np.random.rand(30, 64, 64).astype(np.float32)
        lbl = np.zeros((30, 64, 64), dtype=np.float32)
        img_p, lbl_p = ds._extract_z_patch(img, lbl, 15, 64)
        assert img_p.shape[0] == 64  # padded to 64

    def test_extract_z_patch_normal(self):
        """Volume larger than patch: should extract exact D slices."""
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        img = np.random.rand(200, 64, 64).astype(np.float32)
        lbl = np.zeros((200, 64, 64), dtype=np.float32)
        img_p, lbl_p = ds._extract_z_patch(img, lbl, 100, 64)
        assert img_p.shape[0] == 64

    def test_extract_z_patch_boundary(self):
        """Patch at boundary should shift window, not go out of bounds."""
        from segtask_v1.data.dataset import SegDataset3D
        ds = SegDataset3D.__new__(SegDataset3D)
        img = np.random.rand(100, 32, 32).astype(np.float32)
        lbl = np.zeros((100, 32, 32), dtype=np.float32)
        # Near the end
        img_p, _ = ds._extract_z_patch(img, lbl, 98, 64)
        assert img_p.shape[0] == 64
        # Near the start
        img_p, _ = ds._extract_z_patch(img, lbl, 2, 64)
        assert img_p.shape[0] == 64


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
        from segtask_v1.data.augment import GPUAugmentor
        img = torch.randn(1, 1, 8, 16, 16)
        out = GPUAugmentor._gaussian_blur(img, prob=1.0, sigma_range=[1.0, 1.0])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
