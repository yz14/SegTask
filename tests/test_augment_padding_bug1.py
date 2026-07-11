"""Regression tests for BUG-1: affine/elastic resampling must use
``padding_mode="border"`` (not ``"zeros"``) so the weight_map's
``+1`` shift contract (bg=1.0, fg=w+1) is preserved in oversample
slack regions instead of being silently zeroed-out.

历史上 ``_random_affine`` / ``_elastic_deform`` 是两个独立函数；现已融合为
``_random_affine_elastic``（单次 grid_sample），不变量不变：三路
grid_sample 均 border padding。

Run:
    python -m pytest tests/test_augment_padding_bug1.py -v
"""

from __future__ import annotations

import torch

from segtask_v1.data.augment import _random_affine_elastic


def _make_inputs(B=2, C_lbl=2, D=8, H=16, W=16, device="cpu"):
    image = torch.randn(B, 1, D, H, W, device=device)
    label = torch.zeros(B, C_lbl, D, H, W, device=device)
    # weight_map carries the dataset's +1 shift: bg=1.0 everywhere by default
    weight_map = torch.ones(B, 1, D, H, W, device=device)
    return image, label, weight_map


def _affine_only(image, label, weight_map, rotate_range, scale_range):
    return _random_affine_elastic(
        image, label,
        affine_prob=1.0, rotate_range=rotate_range, scale_range=scale_range,
        elastic_prob=0.0, sigma=5.0, alpha=0.0,
        weight_map=weight_map)


# ---------------------------------------------------------------------------
# BUG-1 invariants
# ---------------------------------------------------------------------------
def test_affine_weight_map_no_zero_padding_when_zoomed_out():
    """With strong zoom-out (scale > 1 in the affine matrix means image
    shrinks in screen space), zero padding would leave a border of
    weight=0 (loss-ignored). With border padding the entire weight_map
    must stay >= 1 because bg=1 is replicated outward.
    """
    torch.manual_seed(0)
    image, label, weight = _make_inputs()
    _, _, w_out = _affine_only(
        image, label, weight,
        rotate_range=[20.0, 25.0], scale_range=[1.4, 1.5])
    assert w_out is not None
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
    _, _, w_out = _affine_only(
        image, label, weight,
        rotate_range=[-1.0, 1.0],   # tiny rotation — center stays put
        scale_range=[1.0, 1.0])
    assert w_out[0, 0, 4, 8, 8].item() > 5.0, (
        "Center fg weight degraded too much under near-identity affine — "
        "interpolation pivot may be wrong."
    )


def test_elastic_weight_map_no_zero_padding():
    """Aggressive elastic deformation pushes samples out of bounds; border
    replication keeps the weight_map >= 1 everywhere too.
    """
    torch.manual_seed(2)
    image, label, weight = _make_inputs()
    _, _, w_e = _random_affine_elastic(
        image.clone(), label.clone(),
        affine_prob=0.0, rotate_range=[0.0, 0.0], scale_range=[1.0, 1.0],
        elastic_prob=1.0, sigma=4.0, alpha=20.0,
        weight_map=weight.clone())
    assert torch.all(w_e >= 1.0 - 1e-6)

    _, _, w_a = _affine_only(
        image.clone(), label.clone(), weight.clone(),
        rotate_range=[15.0, 20.0], scale_range=[1.3, 1.4])
    assert torch.all(w_a >= 1.0 - 1e-6), (
        "affine path still leaks zero-padding into weight_map; "
        "BUG-1 fix incomplete."
    )


def test_affine_no_weight_map_does_not_crash():
    """Backwards compat: when weight_map is None the third return must
    also be None and shapes must round-trip.
    """
    torch.manual_seed(3)
    image, label, _ = _make_inputs()
    img, lbl, w = _affine_only(
        image, label, None,
        rotate_range=[-10.0, 10.0], scale_range=[0.9, 1.1])
    assert w is None
    assert img.shape == (2, 1, 8, 16, 16) and lbl.shape == (2, 2, 8, 16, 16)
