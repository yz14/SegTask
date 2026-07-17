"""Tests for ``LossConfig.slice_loss_reduction`` toggle (Improvement #1).

Verifies the new ``per_slice`` (default, backward compatible) vs.
``per_volume`` reduction path inside ``SliceChannelLoss``:

  1. Default reduction is ``per_slice`` and remains bit-identical to the
     pre-toggle implementation (regression safety).
  2. ``per_volume`` matches the closed-form 3D Dice formula on a hand
     constructed example.
  3. ``per_volume`` correctly fixes the empty-slice "white-give" failure
     mode: a window with mostly-empty slices and one wrong prediction
     gets a *higher* Dice loss under ``per_volume`` than under
     ``per_slice`` (which is diluted by ≈0 contributions from the
     empty slices).
  4. BCE / Focal are reduction-invariant (per-voxel mean of the same
     voxels), so the two reductions yield numerically identical loss
     for these.
  5. Pooled metric primitives (``dice_batch_stats``) are
     reduction-invariant — pooled Dice across the whole val set is
     unchanged so existing val curves stay comparable.
  6. ``split_for_metrics`` returns the expected rank for each mode.
  7. Gradient flows through both modes for compound losses.
  8. Config validation rejects invalid reduction strings.

Run:
    conda activate torch27_env
    python test_slice_loss_reduction.py
"""
from __future__ import annotations

import math
import sys

import torch


def _ok(name: str, msg: str = "") -> None:
    print(f"  [PASS] {name}{(' — ' + msg) if msg else ''}")


# ---------------------------------------------------------------------------
# 1. Default reduction is "per_slice" + backward compatibility
# ---------------------------------------------------------------------------
def test_default_reduction_is_per_slice():
    from taskcore.config.core import LossConfig
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    cfg = LossConfig()
    assert cfg.slice_loss_reduction == "per_slice", (
        f"default must be 'per_slice'; got {cfg.slice_loss_reduction!r}")

    scl = SliceChannelLoss(
        base_loss=BinaryDiceLoss(),
        num_fg_classes=2, num_slices=4, label_values=[0, 1, 2])
    assert scl.reduction == "per_slice"
    _ok("Default LossConfig.slice_loss_reduction == 'per_slice'")


def test_per_slice_matches_legacy_split_path():
    """``per_slice`` must produce the same scalar as manually flattening
    pred/target to ``(B*D, 1, H, W)`` and calling base_loss directly —
    the contract callers relied on before the toggle existed.
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    torch.manual_seed(0)
    B, num_fg, D, H, W = 2, 2, 4, 12, 12
    label_values = [0, 1, 2]
    base = BinaryDiceLoss(smooth=1e-5)
    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_slice")

    pred = torch.randn(B, num_fg * D, H, W)
    label = torch.randint(0, len(label_values), (B, D, H, W),
                          dtype=torch.float32)

    # Legacy reference path: replicate the old in-line logic.
    pred_5d = pred.reshape(B, num_fg, D, H, W)
    pred_flat = pred_5d.permute(0, 2, 1, 3, 4).reshape(B * D, num_fg, H, W)
    fg = torch.tensor(label_values[1:], dtype=torch.float32)
    target_flat = (label.reshape(B * D, H, W).unsqueeze(1)
                   == fg.reshape(1, -1, 1, 1)).float()
    ref = pred.new_zeros(())
    for c in range(num_fg):
        ref = ref + base(pred_flat[:, c:c + 1], target_flat[:, c:c + 1])
    ref = ref / num_fg

    new = scl(pred, label)
    torch.testing.assert_close(new, ref, atol=1e-6, rtol=1e-6)
    _ok("per_slice numerically matches legacy reference path")


# ---------------------------------------------------------------------------
# 2. per_volume matches the closed-form 3D Dice
# ---------------------------------------------------------------------------
def test_per_volume_matches_closed_form_3d_dice():
    """Hand-constructed deterministic case where the volumetric Dice has
    a known closed-form value. Verifies the per_volume path actually
    sums over (D, H, W) rather than (H, W) only.
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 1, 1, 4, 4, 4
    label_values = [0, 1]
    smooth = 0.0   # exact arithmetic, no smoothing needed

    base = BinaryDiceLoss(smooth=smooth)
    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    # Build pred logits that sigmoid → exactly 0.0 (very negative) except
    # in a chosen 2x2 region of slice 1 where logit = +large → sigmoid 1.
    LARGE = 20.0   # sigmoid(20) ≈ 1.0 with 9-decimals precision
    NEG = -20.0
    pred = torch.full((B, num_fg * D, H, W), NEG)
    pred[0, 1, 0:2, 0:2] = LARGE  # channel index 1 = (fg_class=0, slice=1)

    # Label: GT foreground (label=1) in slice 1 covers (0:2, 0:3) — i.e.
    # 2x3=6 voxels, of which the prediction overlaps the 2x2=4 in
    # (0:2, 0:2). FP=0 (model only predicts inside GT), FN=2.
    label = torch.zeros(B, D, H, W)
    label[0, 1, 0:2, 0:3] = 1

    # Closed-form 3D Dice:
    #   |P ∩ T| = 4
    #   |P| + |T| = 4 + 6 = 10
    #   Dice = 2*4/10 = 0.8 → loss = 0.2
    expected_loss = 0.2
    got = scl(pred, label).item()
    assert math.isclose(got, expected_loss, abs_tol=1e-4), (
        f"per_volume Dice loss {got:.6f} != expected {expected_loss}")
    _ok(f"per_volume Dice = closed-form 3D Dice (loss={got:.4f})")


# ---------------------------------------------------------------------------
# 3. Empty-slice failure mode IS fixed by per_volume
# ---------------------------------------------------------------------------
def test_per_volume_fixes_empty_slice_dilution():
    """Construct a window where:
      - slice 0 has a non-trivial wrong prediction (high FP, no FN-cover)
      - slices 1..D-1 are entirely empty (label) AND model predicts empty.

    Under ``per_slice``: slice 0 contributes a real Dice loss, but
    slices 1..D-1 each contribute Dice ≈ (0+smooth)/(0+smooth) ≈ 1 →
    loss ≈ 0. The mean across slices dilutes the slice-0 signal by
    a factor of D.

    Under ``per_volume``: the (D, H, W) sum aggregates the slice-0
    intersection / denominator with the empty-slice zeros (which add
    nothing). The loss reflects the slice-0 error directly, NOT diluted
    by D-1 trivial slices.

    We assert: per_volume_loss > per_slice_loss by a large margin (the
    expected dilution ratio is ~D for sparsely-foreground windows).
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 1, 1, 8, 16, 16
    label_values = [0, 1]
    smooth = 1e-5

    base = BinaryDiceLoss(smooth=smooth)
    scl_slice = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_slice")
    scl_vol = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    # NEG=-50 → sigmoid ≈ 1.9e-22 (essentially fp32 zero). With H*W=256
    # voxels per empty slice and ``smooth=1e-5`` this guarantees the
    # empty-slice contribution is dominated by smooth/smooth ≈ 1
    # (i.e. loss ≈ 0), reproducing the "white-give" failure mode that
    # this test exists to demonstrate. Higher NEG (e.g. -20) leaks
    # sigmoid mass into the empty-slice denominator and biases the
    # per_slice loss upward by a few percent.
    LARGE = 50.0
    NEG = -50.0
    pred = torch.full((B, num_fg * D, H, W), NEG)
    # Channel layout for num_fg=1: channel k = (class 0, slice k).
    # Confidently-wrong prediction in slice 0: predict a 8x8 region as fg.
    pred[0, 0, 4:12, 4:12] = LARGE

    # GT: slice 0 has a SMALLER 4x4 fg region; rest empty.
    label = torch.zeros(B, D, H, W)
    label[0, 0, 4:8, 4:8] = 1

    loss_slice = scl_slice(pred, label).item()
    loss_vol = scl_vol(pred, label).item()

    # per_volume should be larger because per_slice gets averaged with
    # D-1 ≈ 0-loss empty slices.
    assert loss_vol > loss_slice + 0.1, (
        f"expected per_volume loss to dominate empty-slice-diluted "
        f"per_slice loss; got per_slice={loss_slice:.4f} vs "
        f"per_volume={loss_vol:.4f}")

    # Quantitative check: per_slice ≈ slice-0 Dice loss / D (since the
    # other D-1 slices contribute ~0 each).
    # Slice 0: |P∩T|=16, |P|=64, |T|=16 → Dice=2*16/80=0.4, loss=0.6
    # per_slice ≈ 0.6/D = 0.075   (with smooth, 7 zero-zero slices give ≈0)
    expected_slice_loss = 0.6 / D
    assert math.isclose(loss_slice, expected_slice_loss, abs_tol=0.01), (
        f"per_slice empty-dilution check: expected ≈{expected_slice_loss:.4f}, "
        f"got {loss_slice:.4f}")
    # per_volume: |P∩T|=16, |P|=64, |T|=16 → same 3D Dice=0.4, loss=0.6
    expected_vol_loss = 0.6
    assert math.isclose(loss_vol, expected_vol_loss, abs_tol=0.01), (
        f"per_volume should equal pure 3D Dice loss: expected "
        f"{expected_vol_loss:.4f}, got {loss_vol:.4f}")
    _ok(f"per_volume fixes empty-slice dilution: "
        f"per_slice={loss_slice:.4f}, per_volume={loss_vol:.4f} "
        f"(ratio≈{loss_vol/max(loss_slice, 1e-9):.1f}x)")


# ---------------------------------------------------------------------------
# 4. BCE / Focal: reduction-invariant
# ---------------------------------------------------------------------------
def test_bce_invariant_under_reduction():
    """BCE is per-voxel mean — independent of how voxels are partitioned
    along the batch axis. Both reduction modes must yield the same scalar.
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BCELoss

    torch.manual_seed(1)
    B, num_fg, D, H, W = 2, 2, 4, 8, 8
    label_values = [0, 1, 2]
    base = BCELoss()

    scl_slice = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_slice")
    scl_vol = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    pred = torch.randn(B, num_fg * D, H, W)
    label = torch.randint(0, 3, (B, D, H, W), dtype=torch.float32)

    a = scl_slice(pred, label)
    b = scl_vol(pred, label)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    _ok("BCE: per_slice == per_volume (reduction-invariant)")


def test_focal_invariant_under_reduction():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryFocalLoss

    torch.manual_seed(2)
    B, num_fg, D, H, W = 2, 1, 6, 8, 8
    label_values = [0, 1]
    base = BinaryFocalLoss(alpha=0.25, gamma=2.0)

    scl_slice = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_slice")
    scl_vol = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    pred = torch.randn(B, num_fg * D, H, W)
    label = (torch.rand(B, D, H, W) > 0.7).float()

    a = scl_slice(pred, label)
    b = scl_vol(pred, label)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    _ok("Focal: per_slice == per_volume (reduction-invariant)")


# ---------------------------------------------------------------------------
# 5. Pooled Dice metric primitives are reduction-invariant
# ---------------------------------------------------------------------------
def test_pooled_dice_metric_invariant_under_reduction():
    """``dice_batch_stats`` sums (inter, denom) over every voxel; the
    shape (per_slice 4D vs per_volume 5D) must not change those sums.
    Otherwise pooled validation Dice would shift between the two modes,
    making val curves incomparable across runs.
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss
    from taskcore.utils.common import dice_batch_stats

    torch.manual_seed(3)
    B, num_fg, D, H, W = 3, 2, 5, 16, 16
    label_values = [0, 1, 2]
    base = BinaryDiceLoss()

    scl_s = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_slice")
    scl_v = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    pred = torch.randn(B, num_fg * D, H, W)
    label = torch.randint(0, 3, (B, D, H, W), dtype=torch.float32)

    p_s, t_s = scl_s.split_for_metrics(pred, label)
    p_v, t_v = scl_v.split_for_metrics(pred, label)
    assert p_s.shape == (B * D, num_fg, H, W)
    assert p_v.shape == (B, num_fg, D, H, W)

    s_s = dice_batch_stats(p_s, t_s)
    s_v = dice_batch_stats(p_v, t_v)

    torch.testing.assert_close(s_s["inter"], s_v["inter"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s_s["denom"], s_v["denom"], atol=1e-5, rtol=1e-5)
    _ok("Pooled (inter, denom) identical across reductions → "
        "global pooled Dice unchanged")


# ---------------------------------------------------------------------------
# 6. split_for_metrics shape contract per mode
# ---------------------------------------------------------------------------
def test_split_for_metrics_shape_per_mode():
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    B, num_fg, D, H, W = 2, 2, 4, 8, 8
    label_values = [0, 1, 2]
    base = BinaryDiceLoss()

    scl_s = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_slice")
    scl_v = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    pred = torch.randn(B, num_fg * D, H, W)
    label = torch.randint(0, 3, (B, D, H, W), dtype=torch.float32)

    ps, ts = scl_s.split_for_metrics(pred, label)
    pv, tv = scl_v.split_for_metrics(pred, label)
    assert ps.shape == (B * D, num_fg, H, W)
    assert ts.shape == (B * D, num_fg, H, W)
    assert pv.shape == (B, num_fg, D, H, W)
    assert tv.shape == (B, num_fg, D, H, W)
    _ok("split_for_metrics returns rank-4 (per_slice) / rank-5 (per_volume)")


# ---------------------------------------------------------------------------
# 7. Gradient flow through compound loss + DeepSupervision in per_volume
# ---------------------------------------------------------------------------
def test_per_volume_gradient_flow_compound_dice_bce():
    """Compound dice_bce + DeepSupervision must produce finite gradients
    in per_volume mode (the path that actually changes Dice aggregation).
    """
    from segtask_v1.losses.losses import (
        SliceChannelLoss, DeepSupervisionLoss, build_loss)
    from taskcore.config.core import LossConfig

    torch.manual_seed(4)
    B, num_fg, D, H, W = 2, 1, 6, 16, 16
    label_values = [0, 1]

    cfg = LossConfig()
    cfg.name = "dice_bce"
    cfg.compound_weights = [1.0, 1.0]
    cfg.dice_smooth = 1e-5
    base = build_loss(cfg)

    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    # Mock deep-supervision multi-scale prediction list (2 scales).
    pred_full = torch.randn(B, num_fg * D, H, W, requires_grad=True)
    pred_half = torch.randn(B, num_fg * D, H // 2, W // 2, requires_grad=True)
    label = (torch.rand(B, D, H, W) > 0.7).float()

    ds = DeepSupervisionLoss(scl, weights=[1.0, 0.5])
    loss = ds([pred_full, pred_half], label)
    assert torch.isfinite(loss), f"DS+per_volume loss not finite: {loss}"
    loss.backward()
    assert pred_full.grad is not None and torch.isfinite(pred_full.grad).all()
    assert pred_half.grad is not None and torch.isfinite(pred_half.grad).all()
    _ok("per_volume + DeepSupervision: forward/backward/grads finite")


# ---------------------------------------------------------------------------
# 8. Config validation rejects invalid reduction
# ---------------------------------------------------------------------------
def test_config_validate_rejects_invalid_reduction():
    from taskcore.config.core import Config
    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [12, 32, 32]
    cfg.loss.slice_loss_reduction = "bogus"
    cfg.sync()
    try:
        cfg.validate()
    except AssertionError as e:
        assert "slice_loss_reduction" in str(e)
        _ok("Config.validate rejects invalid slice_loss_reduction")
        return
    raise AssertionError("validate should have rejected 'bogus' reduction")


# ---------------------------------------------------------------------------
# 9. Weight map is correctly broadcast in per_volume
# ---------------------------------------------------------------------------
def test_per_volume_weight_map_broadcasting():
    """Weight map shape (B, D, H, W) must be reshaped to (B, 1, D, H, W)
    in per_volume mode and applied as a per-voxel summation weight.
    Verify by:
      - All-zero wmap → loss is zero (no voxel contributes).
      - All-one wmap → loss equals the un-weighted loss exactly.
    """
    from segtask_v1.losses.losses import SliceChannelLoss, BinaryDiceLoss

    torch.manual_seed(5)
    B, num_fg, D, H, W = 1, 1, 4, 8, 8
    label_values = [0, 1]
    base = BinaryDiceLoss(smooth=1e-5)
    scl = SliceChannelLoss(
        base_loss=base, num_fg_classes=num_fg,
        num_slices=D, label_values=label_values, reduction="per_volume")

    pred = torch.randn(B, num_fg * D, H, W)
    label = (torch.rand(B, D, H, W) > 0.5).float()

    # Identity wmap → loss == unweighted loss
    wm_one = torch.ones(B, D, H, W)
    a = scl(pred, label).item()
    b = scl(pred, label, weight_map=wm_one).item()
    assert math.isclose(a, b, abs_tol=1e-5), (
        f"all-one wmap should not change loss: {a} vs {b}")

    # Zero wmap → numerator+denominator both 0; with smooth, dice=1, loss=0
    wm_zero = torch.zeros(B, D, H, W)
    c = scl(pred, label, weight_map=wm_zero).item()
    assert math.isclose(c, 0.0, abs_tol=1e-3), (
        f"all-zero wmap should give Dice loss ≈ 0 (smooth/smooth=1); got {c}")
    _ok(f"per_volume weight_map broadcasting: "
        f"unweighted={a:.4f}, all-one={b:.4f}, all-zero={c:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    print("Improvement #1 tests — slice_loss_reduction toggle")
    print("=" * 60)
    tests = [
        test_default_reduction_is_per_slice,
        test_per_slice_matches_legacy_split_path,
        test_per_volume_matches_closed_form_3d_dice,
        test_per_volume_fixes_empty_slice_dilution,
        test_bce_invariant_under_reduction,
        test_focal_invariant_under_reduction,
        test_pooled_dice_metric_invariant_under_reduction,
        test_split_for_metrics_shape_per_mode,
        test_per_volume_gradient_flow_compound_dice_bce,
        test_config_validate_rejects_invalid_reduction,
        test_per_volume_weight_map_broadcasting,
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
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
