"""Tests for ``SliceChannelLoss.class_weights`` propagation (TODO-3 fix).

Bug being fixed
---------------
``SliceChannelLoss`` iterates per foreground class and feeds ``base_loss``
ONE channel at a time (``pred_c``, ``target_c`` both with C=1). In that
single-channel regime the base loss's internal ``class_weights`` reduces
to a no-op (the cw factor cancels in numerator and denominator of the
normalised weighted mean, or in the simple class-mean reduction). So a
user-supplied ``cfg.loss.class_weights`` would silently fail to take
effect.

Fix
---
``SliceChannelLoss._aggregate_per_class`` now reads ``base_loss
.class_weights`` at the wrapper level and combines per-class losses as
``\u03a3 cw[c] L_c / \u03a3 cw[c]``. ``cw == None`` collapses to the legacy mean.

What the tests verify
---------------------
  1. Regression: ``cw == None`` produces the legacy ``mean`` aggregator
     bit-for-bit (per_slice + per_volume).
  2. Regression: ``cw == [1.0, 1.0]`` is bit-equivalent to the legacy
     mean (this is the user's current shipped config; the fix MUST NOT
     change behaviour for them).
  3. Fix: ``cw == [1.0, 2.0]`` produces ``(L_0 + 2 L_1) / 3`` and is
     numerically distinct from the unweighted mean.
  4. Fix works for ``per_slice`` AND ``per_volume`` reductions.
  5. Fix works WITH a non-trivial weight_map (the rw-style spatial
     weighting that motivated TODO-2 / TODO-3).
  6. Fix is consistent across base-loss families: Dice, BCE, Focal,
     Tversky (covers both ``_weighted_mean_over_classes`` users and
     ``_weighted_voxel_mean`` users).
  7. Construction-time guard: cw length mismatch raises ``ValueError``.
  8. Gradients flow through with cw != [1, 1].

Run:
    D:/miniconda/envs/torch27_env/python.exe test_slice_channel_loss_class_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def _ok(name: str, msg: str = "") -> None:
    print(f"  [PASS] {name}{(' -- ' + msg) if msg else ''}")


def _make_inputs(B: int, num_fg: int, D: int, H: int, W: int,
                 label_values: List[int], seed: int = 0):
    torch.manual_seed(seed)
    pred = torch.randn(B, num_fg * D, H, W)
    label = torch.randint(
        0, len(label_values), (B, D, H, W), dtype=torch.float32)
    return pred, label


# ---------------------------------------------------------------------------
# 1. Regression: cw == None bit-equivalent to the legacy mean
# ---------------------------------------------------------------------------
def test_no_cw_matches_legacy_mean_per_slice():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 2, 2, 4, 12, 12
    pred, label = _make_inputs(B, num_fg, D, H, W, [0, 1, 2], seed=0)
    base = BinaryDiceLoss(smooth=1e-5)  # class_weights=None
    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg, num_slices=D,
        label_values=[0, 1, 2], reduction="per_slice")

    # Manual reference: simple mean of per-class single-channel base_loss.
    ref = pred.new_zeros(())
    pred_5d = pred.reshape(B, num_fg, D, H, W)
    pred_flat = pred_5d.permute(0, 2, 1, 3, 4).reshape(B * D, num_fg, H, W)
    fg = torch.tensor([1, 2], dtype=torch.float32)
    target_flat = (label.reshape(B * D, H, W).unsqueeze(1)
                   == fg.reshape(1, -1, 1, 1)).float()
    for c in range(num_fg):
        ref = ref + base(pred_flat[:, c:c + 1], target_flat[:, c:c + 1])
    ref = ref / num_fg

    got = scl(pred, label)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)
    _ok("cw=None per_slice == legacy mean reference")


def test_no_cw_matches_legacy_mean_per_volume():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 2, 2, 4, 12, 12
    pred, label = _make_inputs(B, num_fg, D, H, W, [0, 1, 2], seed=1)
    base = BinaryDiceLoss(smooth=1e-5)
    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg, num_slices=D,
        label_values=[0, 1, 2], reduction="per_volume")

    pred_5d = pred.reshape(B, num_fg, D, H, W)
    fg = torch.tensor([1, 2], dtype=torch.float32)
    target_5d = (label.unsqueeze(1) == fg.reshape(1, -1, 1, 1, 1)).float()
    ref = pred.new_zeros(())
    for c in range(num_fg):
        ref = ref + base(pred_5d[:, c:c + 1], target_5d[:, c:c + 1])
    ref = ref / num_fg

    got = scl(pred, label)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)
    _ok("cw=None per_volume == legacy mean reference")


# ---------------------------------------------------------------------------
# 2. Regression: cw == [1.0, 1.0] bit-equivalent to cw == None
# ---------------------------------------------------------------------------
def test_uniform_cw_bit_equivalent_to_no_cw():
    """The user's current ``loss.class_weights: [1.0, 1.0]`` MUST produce
    the same scalar as ``None``. This is the safety guarantee for the
    silently-shipped fix.
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 2, 2, 4, 12, 12
    pred, label = _make_inputs(B, num_fg, D, H, W, [0, 1, 2], seed=2)

    for reduction in ("per_slice", "per_volume"):
        base_none = BinaryDiceLoss(smooth=1e-5, class_weights=None)
        scl_none = SliceChannelLoss(
            base_loss=base_none, num_fg_classes=num_fg, num_slices=D,
            label_values=[0, 1, 2], reduction=reduction)

        base_uni = BinaryDiceLoss(smooth=1e-5, class_weights=[1.0, 1.0])
        scl_uni = SliceChannelLoss(
            base_loss=base_uni, num_fg_classes=num_fg, num_slices=D,
            label_values=[0, 1, 2], reduction=reduction)

        a = scl_none(pred, label)
        b = scl_uni(pred, label)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    _ok("cw=[1, 1] bit-equivalent to cw=None (per_slice + per_volume)")


# ---------------------------------------------------------------------------
# 3. Fix: cw=[1, 2] produces (L_0 + 2 L_1) / 3, distinct from mean
# ---------------------------------------------------------------------------
def _per_class_terms(scl, base_for_terms, pred, label, reduction):
    """Compute per-class L_c the same way the wrapper does so we can
    construct the analytical expected weighted mean."""
    B = pred.shape[0]
    num_fg = scl.num_fg
    D = scl.num_slices
    H, W = pred.shape[-2:]
    fg = torch.tensor([1, 2][:num_fg], dtype=torch.float32)
    if reduction == "per_volume":
        pred_5d = pred.reshape(B, num_fg, D, H, W)
        target_5d = (label.unsqueeze(1) == fg.reshape(
            1, -1, 1, 1, 1)).float()
        terms = []
        for c in range(num_fg):
            terms.append(base_for_terms(
                pred_5d[:, c:c + 1], target_5d[:, c:c + 1]))
        return terms
    # per_slice
    pred_5d = pred.reshape(B, num_fg, D, H, W)
    pred_flat = pred_5d.permute(0, 2, 1, 3, 4).reshape(
        B * D, num_fg, H, W)
    target_flat = (label.reshape(B * D, H, W).unsqueeze(1)
                   == fg.reshape(1, -1, 1, 1)).float()
    terms = []
    for c in range(num_fg):
        terms.append(base_for_terms(
            pred_flat[:, c:c + 1], target_flat[:, c:c + 1]))
    return terms


def test_nonuniform_cw_applies_weighted_mean_per_slice():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 2, 2, 4, 12, 12
    pred, label = _make_inputs(B, num_fg, D, H, W, [0, 1, 2], seed=3)
    cw = [1.0, 2.0]

    # Build a SECOND base for the analytical reference whose
    # class_weights is None (so the wrapper's cw is provably the
    # ONLY source of class weighting in the analytical comparison).
    base_for_ref = BinaryDiceLoss(smooth=1e-5, class_weights=None)
    base_ours = BinaryDiceLoss(smooth=1e-5, class_weights=cw)
    scl = SliceChannelLoss(
        base_loss=base_ours, num_fg_classes=num_fg, num_slices=D,
        label_values=[0, 1, 2], reduction="per_slice")

    terms = _per_class_terms(scl, base_for_ref, pred, label, "per_slice")
    expected = (cw[0] * terms[0] + cw[1] * terms[1]) / sum(cw)
    got = scl(pred, label)

    # Sanity: actual is NOT the unweighted mean (otherwise the fix
    # wouldn't be doing anything).
    unweighted = (terms[0] + terms[1]) / 2
    assert not torch.isclose(got, unweighted, atol=1e-6), (
        f"got matches unweighted mean ({got.item():.6f}) -- cw=[1,2] "
        f"silently no-op!")
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
    _ok(f"cw=[1, 2] per_slice == (L0 + 2 L1) / 3   "
        f"(got={got.item():.4f}, mean={unweighted.item():.4f})")


def test_nonuniform_cw_applies_weighted_mean_per_volume():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 2, 2, 4, 12, 12
    pred, label = _make_inputs(B, num_fg, D, H, W, [0, 1, 2], seed=4)
    cw = [1.0, 3.0]

    base_for_ref = BinaryDiceLoss(smooth=1e-5, class_weights=None)
    base_ours = BinaryDiceLoss(smooth=1e-5, class_weights=cw)
    scl = SliceChannelLoss(
        base_loss=base_ours, num_fg_classes=num_fg, num_slices=D,
        label_values=[0, 1, 2], reduction="per_volume")

    terms = _per_class_terms(scl, base_for_ref, pred, label, "per_volume")
    expected = (cw[0] * terms[0] + cw[1] * terms[1]) / sum(cw)
    got = scl(pred, label)

    unweighted = (terms[0] + terms[1]) / 2
    assert not torch.isclose(got, unweighted, atol=1e-6), (
        "got matches unweighted mean -- cw=[1,3] silently no-op!")
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
    _ok(f"cw=[1, 3] per_volume == (L0 + 3 L1) / 4   "
        f"(got={got.item():.4f}, mean={unweighted.item():.4f})")


# ---------------------------------------------------------------------------
# 4. Fix works WITH a non-trivial weight_map (rw-style spatial weighting)
# ---------------------------------------------------------------------------
def test_nonuniform_cw_with_weight_map():
    """rw-style weight_map (high spatial weights at boundaries) and
    non-uniform class_weights interact: wrapper must still produce
    the analytical ``(cw[0] L_0 + cw[1] L_1) / sum(cw)``.
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    torch.manual_seed(5)
    B, num_fg, D, H, W = 2, 2, 4, 12, 12
    pred = torch.randn(B, num_fg * D, H, W)
    label = torch.randint(0, 3, (B, D, H, W), dtype=torch.float32)
    # rw-style weight_map: bg=1, varied {3, 5, 8, 10, 20} like the actual
    # lung_prep npz inspection found.
    rw_choices = torch.tensor([1, 3, 5, 8, 10, 20], dtype=torch.float32)
    wmap = rw_choices[torch.randint(0, len(rw_choices), (B, D, H, W))]

    cw = [1.0, 2.5]
    base_for_ref = BinaryDiceLoss(smooth=1e-5, class_weights=None)
    base_ours = BinaryDiceLoss(smooth=1e-5, class_weights=cw)
    scl = SliceChannelLoss(
        base_loss=base_ours, num_fg_classes=num_fg, num_slices=D,
        label_values=[0, 1, 2], reduction="per_volume")

    # Reference: per-class L_c with the SAME weight_map but cw=None.
    fg = torch.tensor([1, 2], dtype=torch.float32)
    pred_5d = pred.reshape(B, num_fg, D, H, W)
    target_5d = (label.unsqueeze(1) == fg.reshape(1, -1, 1, 1, 1)).float()
    wmap_5d = wmap.unsqueeze(1)
    terms = []
    for c in range(num_fg):
        terms.append(base_for_ref(
            pred_5d[:, c:c + 1], target_5d[:, c:c + 1],
            weight_map=wmap_5d))
    expected = (cw[0] * terms[0] + cw[1] * terms[1]) / sum(cw)
    got = scl(pred, label, weight_map=wmap)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
    _ok("cw=[1, 2.5] + rw-style weight_map matches analytical formula")


# ---------------------------------------------------------------------------
# 5. Fix is consistent across base-loss families
# ---------------------------------------------------------------------------
def test_nonuniform_cw_across_loss_families():
    """The fix is base_loss-agnostic. Verify the wrapper-level weighting
    produces the analytical ``\u03a3 cw[c] L_c / \u03a3 cw[c]`` for the four
    base losses you actually use in production:

      Dice / Tversky use ``_weighted_mean_over_classes`` internally.
      BCE / Focal use ``_weighted_voxel_mean`` internally.

    Both code paths reduce to a no-op on single-channel input but the
    wrapper's per-class weighting is independent of either.
    """
    from segtask_v1.losses.losses import (
        SliceChannelLoss, BinaryDiceLoss, BCELoss, BinaryFocalLoss,
        BinaryTverskyLoss)

    B, num_fg, D, H, W = 2, 2, 4, 8, 8
    pred, label = _make_inputs(B, num_fg, D, H, W, [0, 1, 2], seed=6)
    cw = [1.0, 4.0]

    base_factories = {
        "BinaryDice": (lambda c: BinaryDiceLoss(class_weights=c, smooth=1e-5)),
        "BCE":        (lambda c: BCELoss(class_weights=c)),
        "Focal":      (lambda c: BinaryFocalLoss(
            alpha=0.5, gamma=2.0, class_weights=c)),
        "Tversky":    (lambda c: BinaryTverskyLoss(
            alpha=0.3, beta=0.7, class_weights=c, smooth=1e-5)),
    }
    for name, factory in base_factories.items():
        base_ref = factory(None)
        base_ours = factory(cw)
        scl = SliceChannelLoss(
            base_loss=base_ours, num_fg_classes=num_fg, num_slices=D,
            label_values=[0, 1, 2], reduction="per_slice")
        terms = _per_class_terms(scl, base_ref, pred, label, "per_slice")
        expected = (cw[0] * terms[0] + cw[1] * terms[1]) / sum(cw)
        got = scl(pred, label)
        torch.testing.assert_close(
            got, expected, atol=1e-6, rtol=1e-6,
            msg=f"family={name}: got {got.item()} vs expected "
                f"{expected.item()}")
    _ok(f"cw weighting consistent across {len(base_factories)} loss families")


# ---------------------------------------------------------------------------
# 6. Construction-time length mismatch guard
# ---------------------------------------------------------------------------
def test_cw_length_mismatch_raises():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    base = BinaryDiceLoss(smooth=1e-5, class_weights=[1.0, 2.0, 3.0])
    try:
        SliceChannelLoss(
            base_loss=base, num_fg_classes=2, num_slices=4,
            label_values=[0, 1, 2], reduction="per_slice")
    except ValueError as e:
        assert "class_weights" in str(e) and "num_fg_classes" in str(e)
        _ok("cw length mismatch raises ValueError early")
        return
    raise AssertionError("expected ValueError on cw length mismatch")


# ---------------------------------------------------------------------------
# 7. Gradient flow with cw != [1, 1]
# ---------------------------------------------------------------------------
def test_gradient_flow_with_nonuniform_cw():
    from segtask_v1.losses.losses import (
        SliceChannelLoss, build_loss)
    from taskcore.config.core import LossConfig

    torch.manual_seed(7)
    B, num_fg, D, H, W = 2, 2, 4, 8, 8
    cw = [1.0, 3.0]

    cfg = LossConfig()
    cfg.name = "dice_focal"
    cfg.class_weights = list(cw)
    cfg.compound_weights = [1.0, 1.0]
    cfg.dice_smooth = 1e-5
    cfg.focal_alpha = 0.5
    cfg.focal_gamma = 2.0
    base = build_loss(cfg)
    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg, num_slices=D,
        label_values=[0, 1, 2], reduction="per_volume")

    pred = torch.randn(
        B, num_fg * D, H, W, requires_grad=True)
    label = torch.randint(0, 3, (B, D, H, W), dtype=torch.float32)
    loss = scl(pred, label)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), "non-finite grad"

    # Diagnostic: per-class slab gradient magnitudes. Channel-major
    # layout under SliceChannelLoss: class 0 (lung) occupies channels
    # 0..D-1; class 1 (bone) occupies channels D..2D-1.
    g_lung = pred.grad[:, 0:D].abs().mean().item()
    g_bone = pred.grad[:, D:2 * D].abs().mean().item()
    ratio = g_bone / max(g_lung, 1e-9)
    # Note: the analytical wrapper-level coefficient ratio is exactly
    # cw[1] / cw[0] = 3. The empirical |grad| ratio also depends on the
    # per-class loss derivatives ``d L_c / d pred_c`` which include
    # Dice's volume-normalisation and Focal's ``(1 - pt)^gamma``
    # suppression near pt = 0.5 (random init); so the empirical ratio
    # need NOT equal 3 here. We only assert finiteness; the ratio is
    # printed for diagnostic.
    print(f"      grad |dL|_lung={g_lung:.6f}  |dL|_bone={g_bone:.6f}  "
          f"empirical ratio={ratio:.2f}x  (analytical coef ratio = 3.0x)")
    _ok("gradient flows finite under cw=[1, 3] dice_focal compound")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("SliceChannelLoss class_weights propagation tests (TODO-3 fix)")
    print("=" * 70)
    tests = [
        test_no_cw_matches_legacy_mean_per_slice,
        test_no_cw_matches_legacy_mean_per_volume,
        test_uniform_cw_bit_equivalent_to_no_cw,
        test_nonuniform_cw_applies_weighted_mean_per_slice,
        test_nonuniform_cw_applies_weighted_mean_per_volume,
        test_nonuniform_cw_with_weight_map,
        test_nonuniform_cw_across_loss_families,
        test_cw_length_mismatch_raises,
        test_gradient_flow_with_nonuniform_cw,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print("=" * 70)
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
