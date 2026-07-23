"""Smoke / unit tests for the ``data.keep_native_view_depth`` (R1 + R2) feature.

Scope (intentionally model-free — R3 will add stem/aux-head support for
varying per-view depths, after which the end-to-end forward path can be
exercised). This module verifies:

1. Config layer:
   - ``sync()`` derives ``in_channels = sum_k round(D * s_k)`` when ON.
   - ``sync()`` auto-upgrades ``z_boundary_mode`` to ``"edge_pad"``.
   - ``per_view_depths`` property has the right shape (D_0 == D).
   - ``validate()`` rejects misuse outside 2.5D and without aux supervision.
   - OFF path is bit-identical to legacy (in_channels = D * n_views).

2. Dataset layer (``SegDataset3D``):
   - ON mode emits a SINGLE max-FOV cube of shape
     ``(1, eD_max, eH, eW)`` for image/label/(weight_map).
   - Center-cropping ``D`` slices from that cube reproduces the legacy
     view-0 ``edge_pad`` extraction VOXEL-FOR-VOXEL (geometric
     equivalence of the simplification).

3. Trainer split utility (``Trainer._split_views_native_d``):
   - Output shapes match the per-view contract.
   - View 0 occupies the LEADING ``D`` channels of ``image_2d``
     (per the channel-layout documented in the trainer).
   - Aux labels' depths equal ``round(D * s_k)`` exactly.

Run:
    conda activate torch27_env
    python test_keep_native_view_depth.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from taskcore.config.core import (  # noqa: E402
    Config, DataConfig, ModelConfig, LossConfig, TrainConfig, AugConfig,
    PredictConfig,
)
from taskcore.config.seg_bundle import SegBundle, merge_seg_bundle  # noqa: E402
from taskcore.config.seg_task import SegTaskConfig  # noqa: E402
from taskcore.data.dataset import SegDataset3D  # noqa: E402
from segtask_v1.losses.losses import SliceChannelLoss, build_loss  # noqa: E402
from taskcore.models.factory import build_model  # noqa: E402
from segtask_v1.trainer import views  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================
def _make_cfg(
    *,
    multi_res_scales=(1.0, 1.5, 2.0),
    keep_native_view_depth: bool = True,
    aux_seg_supervision: bool = True,
    z_boundary_mode: str = "stretch",  # exercise auto-upgrade by default
    patch_size=(8, 32, 32),
    patch_mode: str = "2_5d",
    stem_fusion_mode: str = "multi_stem_proj",
    aux_head_mode: str = "linear",
    encoder_channels=(16, 32, 64, 128),
):
    cfg = merge_seg_bundle(
        Config(
            data=DataConfig(
                image_dir="dummy", label_dir="dummy",
                label_values=[0, 1, 2], num_classes=3,
                patch_size=list(patch_size),
                patch_mode=patch_mode,
                multi_res_scales=list(multi_res_scales),
                keep_native_view_depth=keep_native_view_depth,
                z_boundary_mode=z_boundary_mode,
            ),
            augment=AugConfig(enabled=False),
            model=ModelConfig(
                encoder_channels=list(encoder_channels),
                blocks_per_level=1,
                stem_fusion_mode=stem_fusion_mode,
                aux_seg_supervision=aux_seg_supervision,
                aux_head_mode=aux_head_mode,
                stem_mode="conv3",
                decoder_type="unet",
            ),
            train=TrainConfig(epochs=1, output_dir=str(ROOT / "outputs" / "tmp")),
        ),
        SegTaskConfig(
            loss=LossConfig(name="dice_bce", slice_loss_reduction="per_volume"),
            predict=PredictConfig(),
        ),
    )
    cfg.sync()
    return cfg


class _SyntheticSegDataset(SegDataset3D):
    """``SegDataset3D`` subclass that bypasses NIfTI I/O.

    The image/label are pre-populated into the volume cache by hijacking
    ``_load_image`` / ``_load_label``. This lets us drive the dataset
    pipeline (``__getitem__`` / ``_getitem_native_d``) entirely from
    in-memory numpy arrays without touching disk.
    """

    def __init__(self, img: np.ndarray, lbl: np.ndarray, **kw):
        self._fake_img = img
        self._fake_lbl = lbl
        super().__init__(
            image_paths=["__synth__.nii.gz"],
            label_paths=["__synth__.nii.gz"],
            npz_paths=["__synth__.npz"],
            **kw,
        )

    def _build_index(self) -> None:
        # Synthetic in-memory volume: no npz package to read.
        self._vol_fg_slices = [np.zeros(0, dtype=np.int32)]
        self._vol_fg_slices_by_cls = [None]
        self._vol_all_slices = [int(self._fake_img.shape[0])]

    def _has_region_weight_file(self, vol_idx):
        return False

    def _load_image(self, vol_idx: int) -> np.ndarray:
        return self._fake_img

    def _load_label(self, vol_idx: int) -> np.ndarray:
        return self._fake_lbl


def _fake_volume(D=40, H=32, W=32, seed: int = 0):
    """Build a deterministic image+label pair with a recognisable z-pattern.

    The image stores ``z`` in every voxel of slice ``z`` so the per-view
    center-crop can be checked by simple equality of slice indices.
    """
    rng = np.random.default_rng(seed)
    img = np.broadcast_to(
        np.arange(D, dtype=np.float32)[:, None, None],
        (D, H, W),
    ).copy()
    # Add a pinch of noise so intensity normalisation doesn't blow up.
    img = img + rng.standard_normal((D, H, W), dtype=np.float32) * 0.001
    lbl = np.zeros((D, H, W), dtype=np.float32)
    # Label slice z with class (z % 3) so each fg class is sampled.
    for z in range(D):
        lbl[z] = float(z % 3)
    return img, lbl


# ===========================================================================
# Test cases
# ===========================================================================
def test_config_sync_on_mode():
    """ON mode: in_channels = sum(round(D*s)); z_boundary_mode auto-upgraded."""
    cfg = _make_cfg(
        multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_view_depth=True,
        aux_seg_supervision=True,
        z_boundary_mode="stretch",
        patch_size=(8, 32, 32),
    )
    cfg.validate()
    expected_depths = [8, 12, 16]
    assert cfg.per_view_depths == expected_depths, (
        f"per_view_depths={cfg.per_view_depths}, want {expected_depths}")
    assert cfg.model.in_channels == sum(expected_depths) == 36, (
        f"in_channels={cfg.model.in_channels}")
    assert cfg.data.z_boundary_mode == "edge_pad", (
        "sync() should auto-upgrade z_boundary_mode")


def test_config_sync_off_mode_unchanged():
    """OFF mode: legacy in_channels = D * n_views; stretch is deprecated and
    always auto-upgraded to edge_pad by sync() regardless of the flag."""
    cfg = _make_cfg(
        multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_view_depth=False,
        aux_seg_supervision=True,
        z_boundary_mode="stretch",
        patch_size=(8, 32, 32),
    )
    cfg.validate()
    assert cfg.model.in_channels == 8 * 3, (
        f"OFF mode in_channels should be D*n_views=24; got {cfg.model.in_channels}")
    assert cfg.data.z_boundary_mode == "edge_pad", (
        "'stretch' is deprecated; sync() must auto-upgrade to 'edge_pad'")


def test_config_validate_rejects_on_outside_2_5d():
    """ON mode outside 2.5D must be rejected."""
    cfg = _make_cfg(
        multi_res_scales=[1.0, 1.5],
        keep_native_view_depth=True,
        aux_seg_supervision=False,
        patch_mode="z_axis",
    )
    try:
        cfg.validate()
    except AssertionError as e:
        assert "patch_mode" in str(e), f"unexpected message: {e}"
        return
    raise AssertionError("validate() should reject ON mode in z_axis")


def test_config_validate_requires_aux_supervision():
    """ON mode without aux_seg_supervision must be rejected."""
    cfg = _make_cfg(
        multi_res_scales=[1.0, 1.5],
        keep_native_view_depth=True,
        aux_seg_supervision=False,  # mis-config
        patch_size=(8, 32, 32),
    )
    try:
        cfg.validate()
    except AssertionError as e:
        assert "aux_seg_supervision" in str(e), f"unexpected message: {e}"
        return
    raise AssertionError("validate() should reject ON without aux supervision")


def test_dataset_native_d_shape_and_geometry():
    """ON dataset emits (1, eD_max, eH, eW) and view-0 == legacy edge_pad path.

    Geometric equivalence: center-cropping ``D`` slices from the
    max-FOV cube must reproduce the legacy view-0 ``edge_pad`` extraction
    voxel-for-voxel (since slice spacing == 1 and z-center is shared).
    """
    D, H, W = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]  # max=2.0 → eD_max = 16
    img_vol, lbl_vol = _fake_volume(D=40, H=H, W=W)

    common = dict(
        label_values=[0, 1, 2],
        patch_size=(D, H, W),
        aug_oversample_ratio=1.0,
        intensity_min=-1.0, intensity_max=39.0,  # keeps img normalisation cheap
        normalize="minmax",
        foreground_oversample_ratio=0.0,
        samples_per_volume=1,
        is_train=False,  # disables stochastic z; centers fall back to randint
        cache_enabled=True,
        cache_max_volumes=1,
    )

    # Force a deterministic z by stubbing _sample_z via numpy seeding +
    # subclassing.
    class _FixedZ(_SyntheticSegDataset):
        def _sample_z(self, vol_idx, D_vol):
            return D_vol // 2  # = 20 for our 40-deep volume

    # Dataset always emits the single max-FOV cube (the per-view split now
    # lives entirely in the trainer; there is no legacy OFF dataset path).
    ds_on = _FixedZ(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        z_boundary_mode="edge_pad",
        **common,
    )
    out_on = ds_on[0]
    eD_max = int(round(D * 2.0))  # = 16
    assert tuple(out_on["image"].shape) == (1, eD_max, H, W), (
        f"ON image shape {tuple(out_on['image'].shape)}")
    assert tuple(out_on["label"].shape) == (1, eD_max, H, W), (
        f"ON label shape {tuple(out_on['label'].shape)}")

    # View-0 reference: an independent edge_pad extraction of exactly D
    # slices around the same z-center, resized in-plane to (H, W). This is
    # the geometric ground truth the trainer's view-0 center-crop must equal.
    from taskcore.data.dataset import resize_3d
    z = 40 // 2
    img_pre = ds_on._load_image(0)   # already preprocessed (normalised)
    lbl_pre = ds_on._load_label(0)
    ref_img, ref_lbl = ds_on._extract_z_patch_padded(img_pre, lbl_pre, z, D)
    ref_img = resize_3d(ref_img, D, H, W, is_label=False)
    ref_lbl = resize_3d(ref_lbl, D, H, W, is_label=True)

    # View-0 equivalence (centered D slices of the ON cube vs. reference).
    on_img = out_on["image"][0].numpy()  # (eD_max, H, W)
    on_lbl = out_on["label"][0].numpy()
    d0 = (eD_max - D) // 2
    on_view0_img = on_img[d0:d0 + D]
    on_view0_lbl = on_lbl[d0:d0 + D]

    if not np.allclose(on_view0_img, ref_img, atol=1e-5):
        diff = float(np.abs(on_view0_img - ref_img).max())
        raise AssertionError(
            f"View-0 image mismatch (max abs diff={diff:.6g}); the ON cube's "
            f"centered-D crop must equal an independent edge_pad view-0 "
            f"extraction voxel-for-voxel.")
    if not np.array_equal(on_view0_lbl, ref_lbl):
        raise AssertionError("View-0 label mismatch vs. reference extraction")


def test_dataset_native_d_aux_view_geometry():
    """View 1 (1.5x) center-crop from ON cube == OFF view-1 raw extraction
    BEFORE the OFF-path's z-resize back to D.

    Since the OFF path resizes view 1 from round(eD*1.5) to eD slices, we
    cannot compare voxel-for-voxel against ``out_off['image'][1]``; instead
    we verify that center-cropping ``D_1`` slices from the ON cube equals
    the *unresampled* slab produced by ``_extract_z_patch_padded`` directly.
    """
    D, H, W = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    img_vol, lbl_vol = _fake_volume(D=40, H=H, W=W)
    common = dict(
        label_values=[0, 1, 2],
        patch_size=(D, H, W),
        aug_oversample_ratio=1.0,
        intensity_min=-1.0, intensity_max=39.0,
        normalize="minmax",
        foreground_oversample_ratio=0.0,
        samples_per_volume=1,
        is_train=False,
        cache_enabled=True, cache_max_volumes=1,
    )

    class _FixedZ(_SyntheticSegDataset):
        def _sample_z(self, vol_idx, D_vol):
            return D_vol // 2

    ds_on = _FixedZ(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        z_boundary_mode="edge_pad",
        **common,
    )
    out = ds_on[0]
    eD_max = 16
    on_img = out["image"][0].numpy()

    # View 1: centered D_1 = 12 slices around the cube center.
    D1 = int(round(D * 1.5))  # 12
    d0 = (eD_max - D1) // 2
    view1_from_cube = on_img[d0:d0 + D1]

    # Reference: take the same z-center and call the dataset's helper
    # directly to extract round(eD * 1.5) slices from the source volume,
    # then preprocess and resize H/W. This is the geometric ground truth
    # for "view 1 native depth".
    z = 40 // 2
    img_pre = ds_on._load_image(0)  # already preprocessed (normalised)
    img_v1, _ = ds_on._extract_z_patch_padded(
        img_pre, ds_on._load_label(0), z, D1)
    # H, W resize matches the dataset path.
    from taskcore.data.dataset import resize_3d
    img_v1 = resize_3d(img_v1, D1, H, W, is_label=False)

    if not np.allclose(view1_from_cube, img_v1, atol=1e-5):
        diff = float(np.abs(view1_from_cube - img_v1).max())
        raise AssertionError(
            f"View-1 native-depth mismatch (max abs diff={diff:.6g}). "
            f"Center-cropping {D1} slices from the max-FOV cube must equal "
            "an independent edge_pad extraction of the same physical slab.")


def test_trainer_split_views_native_d():
    """Trainer split utility shapes & layout (model-free; uses a stub Trainer)."""
    # Build a stub holding only the attributes the split helper reads.
    class _Stub:
        keep_native_view_depth = True
        per_view_depths = [8, 12, 16]   # D, round(1.5D), round(2D)
        target_patch_size = (16, 32, 32)

    stub = _Stub()
    # Call the pure view-split function directly (no Trainer instance needed).
    stub._split_views_native_d = (
        lambda image, label, wmap: views.split_views_native_d(
            image, label, wmap,
            per_view_depths=stub.per_view_depths,
            target_patch_size=stub.target_patch_size))
    B = 2
    eD_max = 16
    H, W = 32, 32

    # Build a tagged image where axis-2 carries the slice index, so
    # post-split we can verify which slices each view picked up.
    image = (torch.arange(eD_max, dtype=torch.float32)
             .reshape(1, 1, eD_max, 1, 1)
             .expand(B, 1, eD_max, H, W).contiguous())
    label = torch.zeros(B, 1, eD_max, H, W)
    wmap = torch.ones(B, 1, eD_max, H, W)

    image_2d, label_main, wmap_main, aux_labels, aux_wmaps = (
        stub._split_views_native_d(image, label, wmap))

    # Channel layout: D + round(1.5D) + round(2D) = 8 + 12 + 16 = 36.
    assert tuple(image_2d.shape) == (B, 36, H, W), (
        f"image_2d shape {tuple(image_2d.shape)}, want {(B, 36, H, W)}")
    # View 0 (channels 0..7) must be the centered D slices.
    d0_view0 = (eD_max - 8) // 2  # = 4
    expect_view0 = torch.arange(d0_view0, d0_view0 + 8, dtype=torch.float32)
    got_view0 = image_2d[0, :8, 0, 0]
    if not torch.equal(got_view0, expect_view0):
        raise AssertionError(
            f"view-0 channels expected {expect_view0.tolist()}, "
            f"got {got_view0.tolist()}")
    # View 1 (channels 8..19) must be the centered 12 slices.
    d0_view1 = (eD_max - 12) // 2  # = 2
    expect_view1 = torch.arange(d0_view1, d0_view1 + 12, dtype=torch.float32)
    got_view1 = image_2d[0, 8:20, 0, 0]
    if not torch.equal(got_view1, expect_view1):
        raise AssertionError(
            f"view-1 channels expected {expect_view1.tolist()}, "
            f"got {got_view1.tolist()}")
    # View 2 (channels 20..35) must be ALL 16 slices (cube center).
    expect_view2 = torch.arange(0, 16, dtype=torch.float32)
    got_view2 = image_2d[0, 20:36, 0, 0]
    if not torch.equal(got_view2, expect_view2):
        raise AssertionError(
            f"view-2 channels expected {expect_view2.tolist()}, "
            f"got {got_view2.tolist()}")

    # Label / wmap_main shapes
    assert tuple(label_main.shape) == (B, 8, H, W), (
        f"label_main shape {tuple(label_main.shape)}")
    assert tuple(wmap_main.shape) == (B, 8, H, W)
    assert len(aux_labels) == 2 and len(aux_wmaps) == 2
    assert tuple(aux_labels[0].shape) == (B, 12, H, W)
    assert tuple(aux_labels[1].shape) == (B, 16, H, W)
    assert tuple(aux_wmaps[0].shape) == (B, 12, H, W)


def _check_model_native_d(label, *, stem_fusion_mode: str, aux_head_mode: str = "linear",
                          encoder_channels=(16, 32, 64, 128)):
    """End-to-end model build + train-mode forward + backward in ON mode.

    Verifies:
      * ``model.in_channels == sum(D_k)`` (factory plumbing).
      * Encoder stem accepts ``in_ch_per_view_list`` for variable-D views.
      * UNet3D produces ``{"main": (B, num_fg*D, H, W),
                            "aux":  [(B, num_fg*D_1, H, W),
                                     (B, num_fg*D_2, H, W), ...]}``.
      * Eval mode emits a single-tensor main output (predictor contract).
      * Per-view ``SliceChannelLoss(num_slices=D_k)`` accepts each aux
        head and gradient flows back into both encoder and aux heads.
    """
    cfg = _make_cfg(
        multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_view_depth=True,
        aux_seg_supervision=True,
        patch_size=(8, 32, 32),
        stem_fusion_mode=stem_fusion_mode,
        aux_head_mode=aux_head_mode,
        encoder_channels=encoder_channels,
    )
    cfg.validate()
    expected_depths = [8, 12, 16]
    assert cfg.per_view_depths == expected_depths
    assert cfg.model.in_channels == sum(expected_depths) == 36

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    # Sanity-check the per-aux-head out_channels.
    num_fg = cfg.num_fg_classes
    expected_aux_out = [num_fg * d for d in expected_depths[1:]]
    assert list(model.aux_head_out_channels) == expected_aux_out, (
        f"[{label}] aux_head_out_channels={model.aux_head_out_channels}, "
        f"want {expected_aux_out}")

    # ---- Train-mode forward ---------------------------------------------
    B = 2
    D, H, W = cfg.data.patch_size
    image = torch.randn(B, cfg.model.in_channels, H, W, device=device)
    model.train()
    out = model(image)
    assert isinstance(out, dict), f"[{label}] expected dict, got {type(out).__name__}"
    main = out["main"]
    aux = out["aux"]
    main_t = main[0] if isinstance(main, list) else main
    expected_main = (B, num_fg * D, H, W)
    assert tuple(main_t.shape) == expected_main, (
        f"[{label}] main shape {tuple(main_t.shape)}, want {expected_main}")
    assert len(aux) == 2, f"[{label}] expected 2 aux outputs"
    for k, ao in enumerate(aux, start=1):
        D_k = expected_depths[k]
        want = (B, num_fg * D_k, H, W)
        assert tuple(ao.shape) == want, (
            f"[{label}] aux head {k} shape {tuple(ao.shape)}, want {want}")

    # ---- Eval-mode (single-tensor main path; legacy predictor contract) -
    model.eval()
    with torch.no_grad():
        out_eval = model(image)
    assert not isinstance(out_eval, dict), (
        f"[{label}] eval must NOT return dict")
    eval_t = out_eval[0] if isinstance(out_eval, list) else out_eval
    assert tuple(eval_t.shape) == expected_main

    # ---- Backward through main + per-view aux SliceChannelLoss --------
    model.train()
    out = model(image)
    base_loss = build_loss(cfg.loss)
    sc_main = SliceChannelLoss(
        base_loss=base_loss, num_fg_classes=num_fg, num_slices=D,
        label_values=cfg.data.label_values,
        reduction=cfg.loss.slice_loss_reduction)
    aux_losses = [
        SliceChannelLoss(
            base_loss=base_loss, num_fg_classes=num_fg, num_slices=D_k,
            label_values=cfg.data.label_values,
            reduction=cfg.loss.slice_loss_reduction)
        for D_k in expected_depths[1:]
    ]
    # Build per-view labels with at least one fg voxel.
    fg_vals = cfg.data.label_values[1:]
    label_main = torch.randint(0, len(cfg.data.label_values),
                                (B, D, H, W), device=device, dtype=torch.long)
    label_main[:, 0, 0, 0] = fg_vals[0]
    aux_labels = []
    for D_k in expected_depths[1:]:
        lk = torch.randint(0, len(cfg.data.label_values),
                            (B, D_k, H, W), device=device, dtype=torch.long)
        lk[:, 0, 0, 0] = fg_vals[0]
        aux_labels.append(lk)

    main_pred = out["main"][0] if isinstance(out["main"], list) else out["main"]
    loss = sc_main(main_pred.float(), label_main)
    for ap, lk, sc_k in zip(out["aux"], aux_labels, aux_losses):
        loss = loss + 0.5 * sc_k(ap.float(), lk)
    loss.backward()

    # Each aux head must have non-zero gradient on at least one parameter.
    for k, head in enumerate(model.aux_heads, start=1):
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads, f"[{label}] aux head {k} got no gradients"
        assert any(g.abs().sum().item() > 0 for g in grads), (
            f"[{label}] aux head {k} gradients are all zero")
    # Encoder must also receive gradient.
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum().item() > 0 for g in enc_grads), (
        f"[{label}] encoder received no non-zero gradient")


def test_model_native_d_multi_stem_proj():
    _check_model_native_d("Plan A (multi_stem_proj)",
                          stem_fusion_mode="multi_stem_proj")


def test_model_native_d_shared_stem():
    _check_model_native_d("Plan A (shared_stem)",
                          stem_fusion_mode="shared_stem")


def test_model_native_d_hierarchical():
    # hierarchical needs n_views <= n_stages and patch H/W divisible by
    # 2^(n_views-1). With n_views=3, stem_stride=1: deepest stride=4.
    # encoder_channels must have at least 4 stages (n_views < n_levels).
    _check_model_native_d("Plan C (hierarchical)",
                          stem_fusion_mode="hierarchical",
                          encoder_channels=(16, 32, 64, 128, 256))


def test_model_native_d_aux_head_conv():
    _check_model_native_d("Plan A (multi_stem_proj, aux_head_mode=conv)",
                          stem_fusion_mode="multi_stem_proj",
                          aux_head_mode="conv")


def test_predictor_native_d_end_to_end():
    """Predictor sliding-window inference in ON mode.

    Builds a tiny synthetic volume, runs ``predict_volume`` in 2.5D + native
    depth mode, and verifies:
      * The GPU window builder ``predictor.inputs.build_z_window_native_d_gpu``
        returns the correct ``(sum(D_k), pH, pW)`` shape per window.
      * The forward path correctly bypasses ``predictor.forwards.reshape_2_5d_input`` and
        feeds the rank-4 ``(B, in_channels, H, W)`` tensor straight to the
        model.
      * The aggregated probability map has shape ``(num_fg, D_orig, H, W)``
        (the standard predictor contract — main head consumes the centered
        D slices regardless of the multi-FOV input layout).
    """
    import shutil
    import tempfile
    import SimpleITK as sitk
    from segtask_v1.predictor import Predictor

    cfg = _make_cfg(
        multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_view_depth=True,
        aux_seg_supervision=True,
        patch_size=(8, 32, 32),
        stem_fusion_mode="multi_stem_proj",
        encoder_channels=(16, 32, 64, 128),
    )
    cfg.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.eval()

    # Sanity-check the per-window builder before invoking the full pipeline.
    predictor = Predictor(model, cfg, device)
    expected_in = sum(cfg.per_view_depths)
    assert predictor.keep_native_view_depth
    assert predictor.per_view_depths == cfg.per_view_depths
    # Direct builder call — must produce rank-3 (in_channels, pH, pW).
    D_vol = 40
    vol_t = torch.randn(D_vol, 32, 32, device=device)
    from segtask_v1.predictor.inputs import build_z_window_native_d_gpu
    win = build_z_window_native_d_gpu(
        vol_t, 16, 24, pH=predictor.patch_H, pW=predictor.patch_W,
        eD_max=predictor._eD_max, view_depths=predictor.per_view_depths)
    assert tuple(win.shape) == (expected_in, 32, 32), (
        f"native-d window shape {tuple(win.shape)}, want "
        f"{(expected_in, 32, 32)}")

    # Build a tiny NIfTI on disk, run predict_volume.
    tmpdir = Path(tempfile.mkdtemp(prefix="aux_native_d_pred_"))
    try:
        rng = np.random.default_rng(0)
        vol_np = rng.standard_normal((D_vol, 32, 32), dtype=np.float32) * 100.0
        img = sitk.GetImageFromArray(vol_np)
        img_path = tmpdir / "vol.nii.gz"
        sitk.WriteImage(img, str(img_path))
        out = predictor.predict_volume(str(img_path), output_dir=str(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    probs = out["probabilities"]
    label_map = out["label_map"]
    num_fg = cfg.num_fg_classes
    assert probs.shape == (num_fg, D_vol, 32, 32), (
        f"probabilities shape {probs.shape}, want {(num_fg, D_vol, 32, 32)}")
    assert label_map.shape == (D_vol, 32, 32), (
        f"label_map shape {label_map.shape}, want {(D_vol, 32, 32)}")
    # Probabilities are sigmoid outputs ∈ [0, 1].
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_predictor_native_d_tta_flip():
    """TTA flip path must work in ON mode (rank-4 x_2d, H/W only)."""
    import shutil
    import tempfile
    import SimpleITK as sitk
    from segtask_v1.predictor import Predictor

    cfg = _make_cfg(
        multi_res_scales=[1.0, 1.5],
        keep_native_view_depth=True,
        aux_seg_supervision=True,
        patch_size=(8, 32, 32),
        stem_fusion_mode="multi_stem_proj",
        encoder_channels=(16, 32, 64, 128),
    )
    cfg.predict.tta_flip = True
    cfg.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.eval()
    predictor = Predictor(model, cfg, device)

    tmpdir = Path(tempfile.mkdtemp(prefix="aux_native_d_tta_"))
    try:
        rng = np.random.default_rng(1)
        vol_np = rng.standard_normal((24, 32, 32), dtype=np.float32) * 50.0
        img = sitk.GetImageFromArray(vol_np)
        img_path = tmpdir / "vol.nii.gz"
        sitk.WriteImage(img, str(img_path))
        out = predictor.predict_volume(str(img_path), output_dir=str(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    probs = out["probabilities"]
    assert probs.shape == (cfg.num_fg_classes, 24, 32, 32)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


# ===========================================================================
# Driver
# ===========================================================================
def main():
    cases = [
        ("config / sync ON",        test_config_sync_on_mode),
        ("config / sync OFF",       test_config_sync_off_mode_unchanged),
        ("config / reject !2_5d",   test_config_validate_rejects_on_outside_2_5d),
        ("config / require auxsup", test_config_validate_requires_aux_supervision),
        ("dataset / native shape",  test_dataset_native_d_shape_and_geometry),
        ("dataset / aux geometry",  test_dataset_native_d_aux_view_geometry),
        ("trainer / split views",   test_trainer_split_views_native_d),
        ("model / Plan A multi_stem_proj", test_model_native_d_multi_stem_proj),
        ("model / Plan A shared_stem",     test_model_native_d_shared_stem),
        ("model / Plan C hierarchical",    test_model_native_d_hierarchical),
        ("model / aux_head_mode=conv",     test_model_native_d_aux_head_conv),
        ("predictor / native_d e2e",       test_predictor_native_d_end_to_end),
        ("predictor / native_d TTA flip",  test_predictor_native_d_tta_flip),
    ]
    n_pass, n_fail = 0, 0
    for label, fn in cases:
        try:
            fn()
            print(f"  OK  {label}")
            n_pass += 1
        except Exception:
            print(f"  FAIL {label}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n[smoke] passed={n_pass}, failed={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
