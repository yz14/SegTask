"""Tests for newly added high-quality segmentation losses.

Covered:
  - GeneralizedDiceLoss        (Sudre et al., DLMIA 2017)
  - BinaryFocalTverskyLoss     (Abraham & Khan, ISBI 2019)
  - LovaszHingeLoss            (Berman et al., CVPR 2018)
  - SoftCLDiceLoss             (Shit et al., CVPR 2021)

Contract checks:
  - Shape / dtype / non-negative scalar output
  - Gradient flows to logits
  - class_weights affects the loss
  - weight_map is consumed without errors (where supported)
  - Factory and compound builders wire through build_loss
  - Validator whitelist updated
  - DeepSupervisionLoss + MultiResolutionLoss still wrap new losses
  - Behavioural sanity: each loss strictly prefers a good pred to a bad pred

Run:  conda activate torch27_env  &&  pytest -xvs test_new_losses.py
"""
from __future__ import annotations

import math
import pytest
import torch

from segtask_v1.losses.losses import (
    BinaryDiceLoss, BCELoss, BinaryFocalLoss, BinaryTverskyLoss,
    GeneralizedDiceLoss, BinaryFocalTverskyLoss, LovaszHingeLoss,
    SoftCLDiceLoss,
    CompoundLoss, DeepSupervisionLoss, MultiResolutionLoss,
    _lovasz_grad_batched, _soft_skeletonize,
    build_loss,
)
from segtask_v1.config import LossConfig, Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_pred_target(
    B: int = 2, C: int = 2, D: int = 6, H: int = 16, W: int = 16,
    seed: int = 0, dtype=torch.float32):
    torch.manual_seed(seed)
    pred = torch.randn(B, C, D, H, W, dtype=dtype, requires_grad=False)
    target = (torch.rand(B, C, D, H, W) > 0.6).to(dtype)
    return pred, target


def _make_structured_pred_target(
    B: int = 1, C: int = 1, D: int = 8, H: int = 16, W: int = 16):
    """Create a tube-like foreground with a matching / corrupted prediction.

    Returns (pred_good_logits, pred_bad_logits, target) where pred_good is
    aligned with target and pred_bad is random.
    """
    torch.manual_seed(0)
    target = torch.zeros(B, C, D, H, W)
    # A thin tubular structure along z-axis in the centre.
    target[..., H // 2 - 1:H // 2 + 1, W // 2 - 1:W // 2 + 1] = 1.0
    # Good logits: large positive where target=1, large negative where =0.
    pred_good = (target * 2.0 - 1.0) * 5.0            # ±5 logits
    pred_bad = torch.randn(B, C, D, H, W) * 0.1       # near-zero noise
    return pred_good, pred_bad, target


# ---------------------------------------------------------------------------
# Generalized Dice Loss
# ---------------------------------------------------------------------------
class TestGeneralizedDice:
    def test_basic_forward(self):
        pred, target = _make_pred_target()
        loss_fn = GeneralizedDiceLoss()
        loss = loss_fn(pred.requires_grad_(), target)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()
        assert 0.0 <= loss.item() <= 2.0   # Dice-like range

    def test_gradient_flow(self):
        pred, target = _make_pred_target()
        pred.requires_grad_()
        loss_fn = GeneralizedDiceLoss()
        loss_fn(pred, target).backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum().item() > 0

    def test_weight_types(self):
        pred, target = _make_pred_target()
        for wt in ("square", "simple", "uniform"):
            loss = GeneralizedDiceLoss(weight_type=wt)(pred, target)
            assert torch.isfinite(loss).all()
            assert loss.item() >= 0

    def test_invalid_weight_type(self):
        with pytest.raises(ValueError):
            GeneralizedDiceLoss(weight_type="bogus")

    def test_weight_map(self):
        pred, target = _make_pred_target()
        wmap = torch.ones(pred.shape[0], 1, *pred.shape[2:]) * 2.0
        loss_fn = GeneralizedDiceLoss()
        loss_no = loss_fn(pred, target).item()
        loss_w = loss_fn(pred, target, weight_map=wmap).item()
        # Uniform weight_map is approximately invariant under GDL (exact in the
        # limit smooth → 0 and w_max → ∞; fp32 summation order causes ~0.3%
        # drift in practice). We assert loose invariance + finite output.
        assert math.isfinite(loss_w)
        assert abs(loss_no - loss_w) / max(abs(loss_no), 1e-6) < 0.01

    def test_weight_map_nonuniform_changes_loss(self):
        # A non-uniform weight_map should demonstrably change the loss —
        # consuming the parameter is not a no-op.
        pred, target = _make_pred_target()
        loss_fn = GeneralizedDiceLoss()
        B, _, D, H, W = pred.shape
        wmap = torch.ones(B, 1, D, H, W)
        wmap[:, :, :, :, : W // 2] = 5.0    # emphasise left half
        loss_no = loss_fn(pred, target).item()
        loss_w = loss_fn(pred, target, weight_map=wmap).item()
        assert not math.isclose(loss_no, loss_w, abs_tol=1e-5)

    def test_class_weights_affects_loss(self):
        pred, target = _make_pred_target()
        loss_a = GeneralizedDiceLoss(class_weights=[1.0, 1.0])(pred, target).item()
        loss_b = GeneralizedDiceLoss(class_weights=[0.1, 1.0])(pred, target).item()
        assert not math.isclose(loss_a, loss_b, abs_tol=1e-6)

    def test_per_sample_mode(self):
        pred, target = _make_pred_target()
        loss = GeneralizedDiceLoss(batch_dice=False)(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    def test_prefers_good_pred(self):
        p_good, p_bad, target = _make_structured_pred_target()
        loss_fn = GeneralizedDiceLoss()
        assert loss_fn(p_good, target).item() < loss_fn(p_bad, target).item()


# ---------------------------------------------------------------------------
# Focal Tversky
# ---------------------------------------------------------------------------
class TestFocalTversky:
    def test_basic_forward(self):
        pred, target = _make_pred_target()
        loss = BinaryFocalTverskyLoss()(pred.requires_grad_(), target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gamma_one_matches_tversky(self):
        pred, target = _make_pred_target()
        ft = BinaryFocalTverskyLoss(gamma=1.0, alpha=0.3, beta=0.7)(pred, target).item()
        t = BinaryTverskyLoss(alpha=0.3, beta=0.7)(pred, target).item()
        assert math.isclose(ft, t, abs_tol=1e-5), (ft, t)

    def test_gamma_gt_one_focuses_hard(self):
        # With gamma > 1, each class contribution (1 - TI)^gamma is smaller
        # than (1 - TI) since (1 - TI) ≤ 1. So FTL ≤ Tversky in magnitude.
        pred, target = _make_pred_target()
        ft = BinaryFocalTverskyLoss(gamma=2.0)(pred, target).item()
        t = BinaryTverskyLoss()(pred, target).item()
        assert ft <= t + 1e-6

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            BinaryFocalTverskyLoss(gamma=0.0)

    def test_weight_map(self):
        pred, target = _make_pred_target()
        wmap = torch.ones(pred.shape[0], 1, *pred.shape[2:])
        loss = BinaryFocalTverskyLoss()(pred, target, weight_map=wmap)
        assert torch.isfinite(loss).all()

    def test_gradient_flow(self):
        pred, target = _make_pred_target()
        pred.requires_grad_()
        BinaryFocalTverskyLoss()(pred, target).backward()
        assert pred.grad is not None and pred.grad.abs().sum().item() > 0

    def test_prefers_good_pred(self):
        p_good, p_bad, target = _make_structured_pred_target()
        loss_fn = BinaryFocalTverskyLoss()
        assert loss_fn(p_good, target).item() < loss_fn(p_bad, target).item()


# ---------------------------------------------------------------------------
# Lovász-Hinge
# ---------------------------------------------------------------------------
class TestLovasz:
    def test_lovasz_grad_properties(self):
        # Sorted gt: should produce a valid gradient vector.
        gt = torch.tensor([1.0, 1.0, 0.0, 0.0])
        grad = _lovasz_grad_batched(gt)
        assert grad.shape == gt.shape
        assert torch.isfinite(grad).all()
        # Sum equals jaccard at the last entry (by construction).
        # jaccard_last = 1 - (Σt - Σt) / (Σt + (L-Σt)) = 1 - 0/L = 1.
        # After the diff-rebuild, sum(grad) == jaccard_last.
        assert math.isclose(grad.sum().item(), 1.0, abs_tol=1e-5)

    def test_lovasz_grad_batched_shape(self):
        gt = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        grad = _lovasz_grad_batched(gt)
        assert grad.shape == gt.shape

    def test_basic_forward_per_sample(self):
        pred, target = _make_pred_target()
        pred.requires_grad_()
        loss = LovaszHingeLoss(per_sample=True)(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()
        loss.backward()
        assert pred.grad is not None and pred.grad.abs().sum().item() > 0

    def test_basic_forward_batch(self):
        pred, target = _make_pred_target()
        loss = LovaszHingeLoss(per_sample=False)(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    def test_class_weights(self):
        pred, target = _make_pred_target()
        loss = LovaszHingeLoss(class_weights=[0.5, 1.5])(pred, target)
        assert torch.isfinite(loss).all()

    def test_weight_map(self):
        pred, target = _make_pred_target()
        wmap = torch.ones(pred.shape[0], 1, *pred.shape[2:]) * 2.0
        loss = LovaszHingeLoss()(pred, target, weight_map=wmap)
        assert torch.isfinite(loss).all()

    def test_prefers_good_pred(self):
        p_good, p_bad, target = _make_structured_pred_target()
        loss_fn = LovaszHingeLoss()
        assert loss_fn(p_good, target).item() < loss_fn(p_bad, target).item()

    def test_perfect_prediction_near_zero(self):
        # With very confident correct logits, loss should be near zero.
        torch.manual_seed(0)
        target = (torch.rand(1, 1, 4, 8, 8) > 0.5).float()
        pred = (target * 2 - 1) * 20.0   # ±20 logits
        loss = LovaszHingeLoss()(pred, target).item()
        assert loss < 1e-3


# ---------------------------------------------------------------------------
# Soft clDice
# ---------------------------------------------------------------------------
class TestCLDice:
    def test_soft_skeletonize_3d(self):
        img = torch.zeros(1, 1, 6, 12, 12)
        img[..., 4:8, 4:8] = 1.0
        skel = _soft_skeletonize(img, n_iter=3, spatial_ndim=3)
        assert skel.shape == img.shape
        assert skel.max().item() <= 1.0 + 1e-5
        assert skel.min().item() >= -1e-5
        # Skeleton is strictly thinner (i.e. sums less) than the source mask.
        assert skel.sum().item() < img.sum().item()

    def test_soft_skeletonize_2d(self):
        img = torch.zeros(1, 1, 16, 16)
        img[..., 6:10, 6:10] = 1.0
        skel = _soft_skeletonize(img, n_iter=3, spatial_ndim=2)
        assert skel.shape == img.shape
        assert skel.sum().item() < img.sum().item()

    def test_basic_forward_3d(self):
        pred, target = _make_pred_target()
        pred.requires_grad_()
        loss = SoftCLDiceLoss()(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()
        loss.backward()
        assert pred.grad is not None and pred.grad.abs().sum().item() > 0

    def test_basic_forward_2d(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 2, 32, 32, requires_grad=True)
        target = (torch.rand(2, 2, 32, 32) > 0.5).float()
        loss = SoftCLDiceLoss(iter_=3)(pred, target)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert pred.grad is not None

    def test_invalid_iter(self):
        with pytest.raises(ValueError):
            SoftCLDiceLoss(iter_=0)

    def test_invalid_spatial_ndim(self):
        # 1D spatial: (B, C, L) → spatial_ndim=1, unsupported.
        with pytest.raises(ValueError):
            SoftCLDiceLoss()(torch.randn(1, 1, 8), torch.zeros(1, 1, 8))

    def test_weight_map_ignored(self):
        pred, target = _make_pred_target()
        wmap = torch.ones(pred.shape[0], 1, *pred.shape[2:]) * 7.0
        loss_no = SoftCLDiceLoss()(pred, target).item()
        loss_w = SoftCLDiceLoss()(pred, target, weight_map=wmap).item()
        assert math.isclose(loss_no, loss_w, abs_tol=1e-6)

    def test_prefers_good_pred(self):
        p_good, p_bad, target = _make_structured_pred_target()
        loss_fn = SoftCLDiceLoss(iter_=3)
        assert loss_fn(p_good, target).item() < loss_fn(p_bad, target).item()


# ---------------------------------------------------------------------------
# Factory / build_loss integration
# ---------------------------------------------------------------------------
class TestFactory:
    @pytest.mark.parametrize(
        "name",
        ["gdl", "focal_tversky", "lovasz", "cldice"])
    def test_build_single(self, name):
        cfg = LossConfig(name=name, class_weights=[1.0, 1.0])
        fn = build_loss(cfg)
        pred, target = _make_pred_target()
        loss = fn(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    @pytest.mark.parametrize(
        "name",
        ["dice_cldice", "dice_focal_tversky", "dice_lovasz",
         "bce_lovasz", "gdl_bce", "gdl_focal", "focal_plus_tversky"])
    def test_build_compound(self, name):
        cfg = LossConfig(name=name, class_weights=[1.0, 1.0])
        fn = build_loss(cfg)
        assert isinstance(fn, CompoundLoss)
        pred, target = _make_pred_target()
        loss = fn(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            build_loss(LossConfig(name="no_such_loss"))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    @pytest.mark.parametrize(
        "name",
        ["gdl", "focal_tversky", "lovasz", "cldice",
         "dice_cldice", "dice_focal_tversky", "dice_lovasz",
         "bce_lovasz", "gdl_bce", "gdl_focal", "focal_plus_tversky"])
    def test_new_names_accepted(self, name):
        cfg = Config()
        cfg.loss.name = name
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.sync()
        cfg.validate()  # must not raise

    def test_invalid_gdl_weight_type_raises(self):
        cfg = Config()
        cfg.loss.gdl_weight_type = "bogus"
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.sync()
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_invalid_focal_tversky_gamma_raises(self):
        cfg = Config()
        cfg.loss.focal_tversky_gamma = 0.0
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.sync()
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_invalid_cldice_iter_raises(self):
        cfg = Config()
        cfg.loss.cldice_iter = 0
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.sync()
        with pytest.raises(AssertionError):
            cfg.validate()


# ---------------------------------------------------------------------------
# Integration with DS / MR wrappers (the trainer-level composition)
# ---------------------------------------------------------------------------
class TestDSMRIntegration:
    def test_ds_wraps_new_loss(self):
        base = build_loss(LossConfig(name="gdl", class_weights=[1.0, 1.0]))
        ds = DeepSupervisionLoss(base, [1.0, 0.5])
        pred_full = torch.randn(2, 2, 8, 16, 16, requires_grad=True)
        pred_half = torch.randn(2, 2, 4, 8, 8, requires_grad=True)
        target = (torch.rand(2, 2, 8, 16, 16) > 0.5).float()
        loss = ds([pred_full, pred_half], target)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert pred_full.grad is not None and pred_half.grad is not None

    def test_mr_wraps_new_loss(self):
        base = build_loss(LossConfig(
            name="dice_cldice", class_weights=[1.0, 1.0]))
        mr = MultiResolutionLoss(
            base_loss=base, num_fg_classes=2, num_res=3,
            label_values=[0, 1, 2])
        # (B, num_fg*C_res, D, H, W) = (1, 6, 6, 16, 16)
        pred = torch.randn(1, 6, 6, 16, 16, requires_grad=True)
        label = torch.zeros(1, 3, 6, 16, 16)
        label[:, :, 2:4, 4:12, 4:12] = 1.0
        loss = mr(pred, label)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert pred.grad is not None

    def test_ds_over_mr_composition(self):
        """The real trainer stack: DS(MR(base))."""
        base = build_loss(LossConfig(
            name="dice_focal_tversky", class_weights=[1.0, 1.0]))
        mr = MultiResolutionLoss(
            base_loss=base, num_fg_classes=2, num_res=2,
            label_values=[0, 1, 2])
        ds = DeepSupervisionLoss(mr, [1.0, 0.5])
        pred_full = torch.randn(1, 4, 6, 16, 16, requires_grad=True)  # 2 res * 2 fg
        pred_half = torch.randn(1, 4, 3, 8, 8, requires_grad=True)
        label = torch.zeros(1, 2, 6, 16, 16)
        label[:, :, 2:4, 4:12, 4:12] = 1.0
        loss = ds([pred_full, pred_half], label)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert pred_full.grad is not None and pred_half.grad is not None


# ---------------------------------------------------------------------------
# Cross-loss sanity: all new losses prefer good-over-bad predictions.
# ---------------------------------------------------------------------------
class TestBehaviouralSanity:
    @pytest.mark.parametrize(
        "fn",
        [
            GeneralizedDiceLoss(),
            BinaryFocalTverskyLoss(),
            LovaszHingeLoss(),
            SoftCLDiceLoss(iter_=3),
        ])
    def test_good_better_than_bad(self, fn):
        p_good, p_bad, target = _make_structured_pred_target()
        good_val = fn(p_good, target).item()
        bad_val = fn(p_bad, target).item()
        assert good_val < bad_val, (type(fn).__name__, good_val, bad_val)
