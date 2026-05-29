"""Regression tests for BUG-1: ``_random_affine`` must use
``padding_mode="border"`` (not ``"zeros"``) so the weight_map's
``+1`` shift contract (bg=1.0, fg=w+1) is preserved in oversample
slack regions instead of being silently zeroed-out.

Run:
    D:\\miniconda\\envs\\torch27_env\\python.exe -m pytest test_augment_padding_bug1.py -v
"""

from __future__ import annotations

import torch

from segtask_v1.data.augment import _random_affine, _elastic_deform


def _make_inputs(B=2, C_lbl=2, D=8, H=16, W=16, device="cpu"):
    image = torch.randn(B, 1, D, H, W, device=device)
    label = torch.zeros(B, C_lbl, D, H, W, device=device)
    # weight_map carries the dataset's +1 shift: bg=1.0 everywhere by default
    weight_map = torch.ones(B, 1, D, H, W, device=device)
    return image, label, weight_map


# ---------------------------------------------------------------------------
# BUG-1 invariants
# ---------------------------------------------------------------------------
def test_affine_weight_map_no_zero_padding_when_zoomed_out():
    """With strong zoom-out (scale > 1 in the affine matrix means image
    shrinks in screen space), the previous ``padding_mode="zeros"`` left
    a border of weight=0 (loss-ignored). After the fix the entire
    weight_map must stay >= 1 because bg=1 is replicated outward.
    """
    torch.manual_seed(0)
    image, label, weight = _make_inputs()
    # prob=1 forces every sample to take the rotated/scaled branch.
    # Aggressive scale so out-of-bounds sampling is guaranteed.
    _, _, w_out = _random_affine(
        image, label, prob=1.0,
        rotate_range=[20.0, 25.0],
        scale_range=[1.4, 1.5],
        weight_map=weight,
    )
    assert w_out is not None
    # bg=1 invariant: with border padding, replicated voxels retain 1.0.
    # No voxel may drop below 1 (would imply zero-padding leakage).
    assert torch.all(w_out >= 1.0 - 1e-6), (
        f"weight_map min={w_out.min().item()} dropped below 1 — "
        f"padding_mode='zeros' regression?"
    )


def test_affine_weight_map_preserves_fg_weight_at_center():
    """Regions deep inside the volume (untouched by border replication)
    must keep their original fg weight, regardless of padding_mode.
    Sanity guard against a future refactor breaking the inner geometry.
    """
    torch.manual_seed(1)
    image, label, weight = _make_inputs(B=1, D=8, H=16, W=16)
    weight[..., 4, 8, 8] = 10.0  # center voxel: fg weight (e.g., w=9 + 1)
    _, _, w_out = _random_affine(
        image, label, prob=1.0,
        rotate_range=[-1.0, 1.0],   # tiny rotation — center stays put
        scale_range=[1.0, 1.0],
        weight_map=weight,
    )
    # Center pixel should still be close to 10 (bilinear ≈ identity here).
    assert w_out[0, 0, 4, 8, 8].item() > 5.0, (
        "Center fg weight degraded too much under near-identity affine — "
        "interpolation pivot may be wrong."
    )


def test_affine_image_label_padding_consistency_with_elastic():
    """``_random_affine`` and ``_elastic_deform`` must agree on the
    out-of-bounds policy. Both should now use border replication so a
    pipeline that runs both transforms produces a single class of
    boundary artefact.
    """
    torch.manual_seed(2)
    image, label, weight = _make_inputs()
    # An aggressive elastic deformation pushes samples out of bounds.
    img_e, _, w_e = _elastic_deform(
        image.clone(), label.clone(), prob=1.0, sigma=4.0, alpha=20.0,
        weight_map=weight.clone(),
    )
    # Border replication keeps the weight_map >= 1 everywhere too.
    assert torch.all(w_e >= 1.0 - 1e-6)

    img_a, _, w_a = _random_affine(
        image.clone(), label.clone(), prob=1.0,
        rotate_range=[15.0, 20.0],
        scale_range=[1.3, 1.4],
        weight_map=weight.clone(),
    )
    assert torch.all(w_a >= 1.0 - 1e-6), (
        "_random_affine still leaks zero-padding into weight_map; "
        "BUG-1 fix incomplete."
    )


def test_affine_no_weight_map_does_not_crash():
    """Backwards compat: when weight_map is None the third return must
    also be None and shapes must round-trip.
    """
    torch.manual_seed(3)
    image, label, _ = _make_inputs()
    img, lbl, w = _random_affine(
        image, label, prob=1.0,
        rotate_range=[-10.0, 10.0],
        scale_range=[0.9, 1.1],
        weight_map=None,
    )
    assert w is None
    assert img.shape == image.shape and lbl.shape == label.shape


if __name__ == "__main__":
    test_affine_weight_map_no_zero_padding_when_zoomed_out()
    test_affine_weight_map_preserves_fg_weight_at_center()
    test_affine_image_label_padding_consistency_with_elastic()
    test_affine_no_weight_map_does_not_crash()
    print("All BUG-1 regression tests passed.")
