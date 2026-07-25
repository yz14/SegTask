"""Tests for ``data.keep_native_multi_res`` (3D z_axis / cubic lazy-extraction).

Scope (R1): dataset + config layer only. The trainer-side per-view
crop+resize step lands in R2; this file therefore validates only the
data emission contract.

Coverage
--------
1. Config layer:
   - ``sync()`` auto-upgrades ``z_boundary_mode`` to ``edge_pad`` for
     ``patch_mode='z_axis'`` when ON.
   - ``validate()`` rejects ON outside ``{'z_axis', 'cubic'}``.
   - ``validate()`` rejects ON with single-scale / non-1.0 view-0.
   - ``validate()`` rejects mutually-exclusive combo with
     ``keep_native_view_depth``.
   - OFF mode is bit-identical to legacy (in_channels = n_views).

2. Dataset layer (z_axis -- ``SegDataset3D``):
   - ON emits ``(1, eD_max, eH, eW)`` for image / label / weight_map.
   - Center-cropping ``D`` slices from the ON cube reproduces the
     OFF-path view-0 voxel-for-voxel (``edge_pad`` boundary).
   - Center-cropping ``D_k`` slices from the ON cube for an aux view
     equals the OFF-path's pre-resize slab from
     ``extract_z_patch_padded`` (geometric ground truth, modulo the
     OFF-path's z-resize that the lazy path defers).

3. Dataset layer (cubic -- ``SegDataset3DCubic``):
   - ON emits ``(1, eD_max, eH_max, eW_max)`` for image / label /
     weight_map.
   - Center-cropping ``round(extract_size * s_k)`` from the ON cube
     equals the OFF-path's pre-resize cube from
     ``_extract_cubic_patch`` for every view k.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from taskcore.config.core import (  # noqa: E402
    Config, DataConfig, ModelConfig, LossConfig, TrainConfig, AugConfig,
    PredictConfig,
)
from taskcore.config.seg_bundle import merge_seg_bundle  # noqa: E402
from taskcore.config.seg_task import SegTaskConfig  # noqa: E402
from taskcore.data.dataset import (  # noqa: E402
    SegDataset3D, SegDataset3DCubic, _extract_cubic_patch,
    extract_z_patch_padded, preprocess_image, resize_3d,
)


# ===========================================================================
# Helpers
# ===========================================================================
def _make_cfg(
    *,
    patch_mode: str = "z_axis",
    multi_res_scales=(1.0, 1.5, 2.0),
    keep_native_multi_res: bool = True,
    z_boundary_mode: str = "stretch",  # exercise auto-upgrade by default
    keep_native_view_depth: bool = False,
    patch_size=(8, 32, 32),
):
    cfg = merge_seg_bundle(
        Config(
            data=DataConfig(
                image_dir="dummy", label_dir="dummy",
                label_values=[0, 1, 2], num_classes=3,
                patch_size=list(patch_size),
                patch_mode=patch_mode,
                multi_res_scales=list(multi_res_scales),
                keep_native_multi_res=keep_native_multi_res,
                keep_native_view_depth=keep_native_view_depth,
                z_boundary_mode=z_boundary_mode,
            ),
            augment=AugConfig(enabled=False),
            model=ModelConfig(
                encoder_channels=[16, 32, 64, 128],
                blocks_per_level=1,
                stem_mode="conv3",
                decoder_type="unet",
            ),
            train=TrainConfig(epochs=1, output_dir=str(ROOT / "outputs" / "tmp")),
        ),
        SegTaskConfig(
            loss=LossConfig(name="dice_bce"),
            predict=PredictConfig(),
        ),
    )
    cfg.sync()
    return cfg


class _SyntheticSegDataset3D(SegDataset3D):
    """``SegDataset3D`` with NIfTI I/O bypassed for unit tests."""

    def __init__(self, img: np.ndarray, lbl: np.ndarray, **kw):
        self._fake_img_pre = img      # PRE-normalisation raw
        self._fake_lbl_raw = lbl
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
        self._vol_all_slices = [int(self._fake_img_pre.shape[0])]

    def _has_region_weight_file(self, vol_idx):
        return False

    def _load_image(self, vol_idx: int) -> np.ndarray:
        # Mirror the production path: preprocess (intensity normalisation)
        # is done inside ``_load_image``, so the cube the lazy path emits
        # is already in [0, 1]. Tests need access to the SAME normalised
        # array for direct comparison.
        return preprocess_image(
            self._fake_img_pre,
            self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)

    def _load_label(self, vol_idx: int) -> np.ndarray:
        return self._fake_lbl_raw


class _SyntheticSegDataset3DCubic(SegDataset3DCubic):
    """``SegDataset3DCubic`` with NIfTI I/O bypassed for unit tests."""

    def __init__(self, img: np.ndarray, lbl: np.ndarray, **kw):
        self._fake_img_pre = img
        self._fake_lbl_raw = lbl
        super().__init__(
            image_paths=["__synth__.nii.gz"],
            label_paths=["__synth__.nii.gz"],
            npz_paths=["__synth__.npz"],
            **kw,
        )

    def _build_index(self) -> None:
        # Synthetic in-memory volume: no npz package to read.
        self._vol_shapes = [tuple(self._fake_img_pre.shape)]
        self._vol_fg_coords = [np.zeros((0, 3), dtype=np.int32)]
        self._vol_fg_coords_by_cls = [None]

    def _has_region_weight_file(self, vol_idx):
        return False

    def _load_image(self, vol_idx: int) -> np.ndarray:
        return preprocess_image(
            self._fake_img_pre,
            self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)

    def _load_label(self, vol_idx: int) -> np.ndarray:
        return self._fake_lbl_raw


def _fake_volume(D=40, H=32, W=32, seed: int = 0):
    """Deterministic image+label with a recognisable z-pattern."""
    rng = np.random.default_rng(seed)
    img = np.broadcast_to(
        np.arange(D, dtype=np.float32)[:, None, None], (D, H, W),
    ).copy()
    img = img + rng.standard_normal((D, H, W), dtype=np.float32) * 0.001
    lbl = np.zeros((D, H, W), dtype=np.float32)
    for z in range(D):
        lbl[z] = float(z % 3)
    return img, lbl


# ===========================================================================
# Config layer
# ===========================================================================
def test_config_sync_z_axis_on_auto_upgrades_z_boundary():
    cfg = _make_cfg(
        patch_mode="z_axis", multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_multi_res=True, z_boundary_mode="stretch",
    )
    cfg.validate()
    assert cfg.data.z_boundary_mode == "edge_pad", (
        "sync() should auto-upgrade z_boundary_mode for z_axis ON path")
    # in_channels follows multi_res_scales count for 3D modes regardless
    # of the lazy flag (model contract is unchanged).
    assert cfg.model.in_channels == 3, (
        f"in_channels should equal len(multi_res_scales)=3; got "
        f"{cfg.model.in_channels}")


def test_config_sync_cubic_on_does_not_touch_z_boundary():
    """'stretch' is deprecated; sync() always auto-upgrades to 'edge_pad'
    (training-side extraction is edge-pad-only), cubic mode included."""
    cfg = _make_cfg(
        patch_mode="cubic", multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_multi_res=True, z_boundary_mode="stretch",
    )
    cfg.validate()
    assert cfg.data.z_boundary_mode == "edge_pad", (
        "'stretch' is deprecated; sync() must auto-upgrade to 'edge_pad'")
    assert cfg.model.in_channels == 3


def test_config_off_mode_unchanged():
    cfg = _make_cfg(
        patch_mode="z_axis", multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_multi_res=False, z_boundary_mode="stretch",
    )
    cfg.validate()
    assert cfg.data.z_boundary_mode == "edge_pad", (
        "'stretch' is deprecated; sync() must auto-upgrade to 'edge_pad'")
    assert cfg.model.in_channels == 3


def test_config_validate_rejects_on_in_2_5d():
    cfg = _make_cfg(
        patch_mode="2_5d", multi_res_scales=[1.0, 1.5],
        keep_native_multi_res=True,
    )
    try:
        cfg.validate()
    except AssertionError as e:
        assert "patch_mode" in str(e), f"unexpected message: {e}"
        return
    raise AssertionError("validate() should reject ON in 2_5d")


def test_config_validate_rejects_single_scale():
    cfg = _make_cfg(
        patch_mode="z_axis", multi_res_scales=[1.0],
        keep_native_multi_res=True,
    )
    try:
        cfg.validate()
    except AssertionError as e:
        assert "multi_res_scales" in str(e), f"unexpected message: {e}"
        return
    raise AssertionError("validate() should reject single-scale ON")


def test_config_validate_rejects_non_canonical_view0():
    cfg = _make_cfg(
        patch_mode="cubic", multi_res_scales=[1.5, 2.0],
        keep_native_multi_res=True,
    )
    try:
        cfg.validate()
    except AssertionError as e:
        assert "1.0" in str(e), f"unexpected message: {e}"
        return
    raise AssertionError("validate() should reject non-1.0 view-0")


def test_config_validate_rejects_mutex_with_keep_native_view_depth():
    """Two flags target different patch modes; setting both must be rejected
    even if the patch_mode side would silently make one of them inactive.
    """
    cfg = _make_cfg(
        patch_mode="z_axis", multi_res_scales=[1.0, 1.5],
        keep_native_multi_res=True, keep_native_view_depth=True,
    )
    try:
        cfg.validate()
    except AssertionError as e:
        # Order-dependent: keep_native_view_depth's own check fires first
        # (rejects non-2_5d). Either message is acceptable as long as
        # validate() rejects the combo.
        msg = str(e)
        assert ("mutually exclusive" in msg
                or "keep_native_view_depth" in msg
                or "patch_mode" in msg), f"unexpected message: {e}"
        return
    raise AssertionError("validate() should reject the mutex combo")


# ===========================================================================
# Dataset layer — z_axis
# ===========================================================================
def _z_axis_common_kwargs(D, H, W):
    return dict(
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


def test_z_axis_on_shape_and_view0_equivalence():
    D, H, W = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]  # max=2 → eD_max=16
    img_vol, lbl_vol = _fake_volume(D=40, H=H, W=W)
    common = _z_axis_common_kwargs(D, H, W)

    class _FixedZ(_SyntheticSegDataset3D):
        def _sample_z(self, vol_idx, D_vol, *, sample_idx=None):
            return D_vol // 2  # = 20

    ds_on = _FixedZ(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        z_boundary_mode="edge_pad",
        **common,
    )
    out_on = ds_on[0]
    eD_max = int(round(D * 2.0))  # 16
    assert tuple(out_on["image"].shape) == (1, eD_max, H, W), (
        f"ON image shape {tuple(out_on['image'].shape)}")
    assert tuple(out_on["label"].shape) == (1, eD_max, H, W), (
        f"ON label shape {tuple(out_on['label'].shape)}")

    # View-0 reference: an independent edge_pad extraction of exactly D
    # slices around the same z-center (the legacy OFF-path view-0). The
    # centered D slices of the max-FOV cube must equal it voxel-for-voxel.
    z = 40 // 2
    img_pre = ds_on._load_image(0)  # already normalised
    lbl_raw = ds_on._load_label(0)
    off_view0_img = resize_3d(
        extract_z_patch_padded(img_pre, z, D), D, H, W, is_label=False)
    off_view0_lbl = resize_3d(
        extract_z_patch_padded(lbl_raw, z, D), D, H, W, is_label=True)

    on_img = out_on["image"][0].numpy()
    on_lbl = out_on["label"][0].numpy()
    d0 = (eD_max - D) // 2
    on_view0_img = on_img[d0:d0 + D]
    on_view0_lbl = on_lbl[d0:d0 + D]

    if not np.allclose(on_view0_img, off_view0_img, atol=1e-5):
        diff = float(np.abs(on_view0_img - off_view0_img).max())
        raise AssertionError(
            f"z_axis view-0 image mismatch (max abs diff={diff:.6g})")
    if not np.array_equal(on_view0_lbl, off_view0_lbl):
        raise AssertionError("z_axis view-0 label mismatch")


def test_z_axis_on_aux_view_geometric_ground_truth():
    """View k center-crop from ON cube == ``extract_z_patch_padded`` on
    the source volume (the geometric ground truth that the OFF-path then
    z-resizes to D and the lazy path defers).
    """
    D, H, W = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    img_vol, lbl_vol = _fake_volume(D=40, H=H, W=W)
    common = _z_axis_common_kwargs(D, H, W)

    class _FixedZ(_SyntheticSegDataset3D):
        def _sample_z(self, vol_idx, D_vol, *, sample_idx=None):
            return D_vol // 2

    ds_on = _FixedZ(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        z_boundary_mode="edge_pad",
        **common,
    )
    out = ds_on[0]
    on_img = out["image"][0].numpy()
    on_lbl = out["label"][0].numpy()
    eD_max = 16

    z = 40 // 2
    img_pre = ds_on._load_image(0)  # already normalised
    lbl_raw = ds_on._load_label(0)

    for s in multi_res_scales:
        D_k = int(round(D * s))
        d0 = (eD_max - D_k) // 2
        view_k_img = on_img[d0:d0 + D_k]
        view_k_lbl = on_lbl[d0:d0 + D_k]

        gt_img = extract_z_patch_padded(img_pre, z, D_k)
        gt_lbl = extract_z_patch_padded(lbl_raw, z, D_k)
        # In-plane: source is already 16x16 == (eH, eW), so resize is a no-op.
        # (resize_3d takes a fast-path when shapes match.)
        gt_img = resize_3d(gt_img, D_k, H, W, is_label=False)
        gt_lbl = resize_3d(gt_lbl, D_k, H, W, is_label=True)

        if not np.allclose(view_k_img, gt_img, atol=1e-5):
            diff = float(np.abs(view_k_img - gt_img).max())
            raise AssertionError(
                f"z_axis view-{s} image mismatch (max abs diff={diff:.6g})")
        if not np.array_equal(view_k_lbl, gt_lbl):
            raise AssertionError(f"z_axis view-{s} label mismatch")


def test_z_axis_on_weight_map_shape_and_geometry():
    """Region-weight map (static mapping) emits at max-FOV depth."""
    D, H, W = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    img_vol, lbl_vol = _fake_volume(D=40, H=H, W=W)
    common = _z_axis_common_kwargs(D, H, W)

    class _FixedZ(_SyntheticSegDataset3D):
        def _sample_z(self, vol_idx, D_vol, *, sample_idx=None):
            return D_vol // 2

    ds_on = _FixedZ(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        z_boundary_mode="edge_pad",
        region_weights=[1.0, 4.0, 2.0],
        **common,
    )
    out = ds_on[0]
    eD_max = 16
    assert "weight_map" in out, "static region_weights should emit a weight_map"
    assert tuple(out["weight_map"].shape) == (1, eD_max, H, W), (
        f"weight_map shape {tuple(out['weight_map'].shape)}")
    # Values follow the label-driven mapping: final weight = configured + 1
    # (same semantics as the per-sample weight-file path).
    lbl_int = out["label"][0].numpy().round().astype(np.int32)
    wmap = out["weight_map"][0].numpy()
    expected = np.ones_like(wmap)
    expected[lbl_int == 0] = 1.0 + 1.0
    expected[lbl_int == 1] = 4.0 + 1.0
    expected[lbl_int == 2] = 2.0 + 1.0
    assert np.allclose(wmap, expected), "weight_map values mismatch"


# ===========================================================================
# Dataset layer — cubic
# ===========================================================================
def _cubic_common_kwargs(patch_size):
    return dict(
        label_values=[0, 1, 2],
        patch_size=patch_size,
        aug_oversample_ratio=1.0,
        intensity_min=-1.0, intensity_max=39.0,
        normalize="minmax",
        foreground_oversample_ratio=0.0,
        samples_per_volume=1,
        is_train=False,
        cache_enabled=True, cache_max_volumes=1,
    )


def test_cubic_on_shape_and_per_view_geometry():
    """ON cubic emits ``(1, eD_max, eH_max, eW_max)``; per-view crops
    voxel-equal the legacy per-view ``_extract_cubic_patch`` output.
    """
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    # Volume must be wide enough that the max cube fits in-bounds (so
    # _safe_center_range gives a real range, not the degenerate fallback).
    Dv, Hv, Wv = 40, 48, 48
    img_vol, lbl_vol = _fake_volume(D=Dv, H=Hv, W=Wv, seed=1)
    common = _cubic_common_kwargs((pD, pH, pW))

    fixed_center = (Dv // 2, Hv // 2, Wv // 2)

    class _FixedCenter(_SyntheticSegDataset3DCubic):
        def _sample_center(self, vol_idx, D, H, W, *, sample_idx=None):
            return fixed_center

    ds_on = _FixedCenter(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        **common,
    )
    out = ds_on[0]
    eD_max = int(round(pD * 2.0))
    eH_max = int(round(pH * 2.0))
    eW_max = int(round(pW * 2.0))
    assert tuple(out["image"].shape) == (1, eD_max, eH_max, eW_max), (
        f"ON image shape {tuple(out['image'].shape)}")
    assert tuple(out["label"].shape) == (1, eD_max, eH_max, eW_max)

    on_img = out["image"][0].numpy()
    on_lbl = out["label"][0].numpy()

    # For every view k: center-crop the ON cube to native size and
    # compare to a fresh ``_extract_cubic_patch`` from the source.
    img_pre = ds_on._load_image(0)
    lbl_raw = ds_on._load_label(0)
    for s in multi_res_scales:
        sD = int(round(pD * s))
        sH = int(round(pH * s))
        sW = int(round(pW * s))
        d0 = (eD_max - sD) // 2
        h0 = (eH_max - sH) // 2
        w0 = (eW_max - sW) // 2
        view_img = on_img[d0:d0 + sD, h0:h0 + sH, w0:w0 + sW]
        view_lbl = on_lbl[d0:d0 + sD, h0:h0 + sH, w0:w0 + sW]

        gt_img = _extract_cubic_patch(img_pre, fixed_center, (sD, sH, sW))
        gt_lbl = _extract_cubic_patch(lbl_raw, fixed_center, (sD, sH, sW))

        if not np.allclose(view_img, gt_img, atol=1e-5):
            diff = float(np.abs(view_img - gt_img).max())
            raise AssertionError(
                f"cubic view-{s} image mismatch (max abs diff={diff:.6g})")
        if not np.array_equal(view_lbl, gt_lbl):
            raise AssertionError(f"cubic view-{s} label mismatch")


def test_cubic_off_path_unchanged():
    """Dataset always emits the single max-FOV cube (per-view split lives
    in the trainer); with scales [1.0, 1.5] the cube is round(p * 1.5)."""
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5]
    Dv, Hv, Wv = 40, 32, 32
    img_vol, lbl_vol = _fake_volume(D=Dv, H=Hv, W=Wv, seed=2)
    common = _cubic_common_kwargs((pD, pH, pW))

    fixed_center = (Dv // 2, Hv // 2, Wv // 2)

    class _FixedCenter(_SyntheticSegDataset3DCubic):
        def _sample_center(self, vol_idx, D, H, W, *, sample_idx=None):
            return fixed_center

    ds = _FixedCenter(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        **common,
    )
    out = ds[0]
    eD, eH, eW = (int(round(p * 1.5)) for p in (pD, pH, pW))
    assert tuple(out["image"].shape) == (1, eD, eH, eW)
    assert tuple(out["label"].shape) == (1, eD, eH, eW)


# ===========================================================================
# CLI smoke run
# ===========================================================================
if __name__ == "__main__":
    tests = [
        test_config_sync_z_axis_on_auto_upgrades_z_boundary,
        test_config_sync_cubic_on_does_not_touch_z_boundary,
        test_config_off_mode_unchanged,
        test_config_validate_rejects_on_in_2_5d,
        test_config_validate_rejects_single_scale,
        test_config_validate_rejects_non_canonical_view0,
        test_config_validate_rejects_mutex_with_keep_native_view_depth,
        test_z_axis_on_shape_and_view0_equivalence,
        test_z_axis_on_aux_view_geometric_ground_truth,
        test_z_axis_on_weight_map_shape_and_geometry,
        test_cubic_on_shape_and_per_view_geometry,
        test_cubic_off_path_unchanged,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {exc}")
        else:
            n_pass += 1
            print(f"  ok    {t.__name__}")
    print(f"\n{n_pass}/{len(tests)} passed")
    if n_pass != len(tests):
        sys.exit(1)
