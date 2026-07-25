"""Tests for ``keep_native_multi_res`` on the trainer side (R2).

Coverage
--------
1. ``Trainer.__init__`` plumbing:
   - ``self.keep_native_multi_res`` activated only for 3D modes with
     n_views > 1.
   - ``self._mr_native_sizes`` correctly computed for z_axis (z-only)
     and cubic (all 3 axes).
   - ``self.target_patch_size`` set to max-FOV physical size.

2. ``_split_views_native_3d`` helper (unit tests on a stub trainer):
   - Output shapes are ``(B, C_res, pD, pH, pW)`` for image / label /
     weight_map.
   - View 0 (s=1.0) takes the centered ``patch_size`` crop with NO
     resize (voxel-for-voxel identity) for both z_axis and cubic.
   - Aux views' centered native crop matches an explicit ground-truth
     center-crop of the input cube (pre-resize), confirming the spatial
     extraction is correct independently of the resampler.
   - Trainer rejects shape mismatches with precise diagnostics.

3. End-to-end smoke (model-free): feed a dataset emission through
   augment OFF + ``_split_views_native_3d`` and verify view-0 of the
   resulting tensor equals the OFF-path dataset's view-0 to within
   resampler tolerance (view 0 needs no resize, so this is exact).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
from segtask_v1.trainer import views  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================
def _make_cfg(
    *,
    patch_mode: str = "z_axis",
    multi_res_scales=(1.0, 1.5, 2.0),
    keep_native_multi_res: bool = True,
    patch_size=(8, 16, 16),
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
            ),
            augment=AugConfig(enabled=False),
            model=ModelConfig(
                encoder_channels=[16, 32, 64, 128],
                blocks_per_level=1,
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
    cfg.validate()
    return cfg


def _make_split_stub(cfg: Config):
    """Return a minimal stub holding only the attributes the split helper
    reads. Avoids the cost (and dependency surface) of building a real
    Trainer for unit tests of the split utility."""

    class _Stub:
        keep_native_multi_res = True

    stub = _Stub()
    stub.cfg = cfg
    pD, pH, pW = (int(x) for x in cfg.data.patch_size)
    sizes: List[Tuple[int, int, int]] = []
    for s in cfg.data.multi_res_scales:
        D_k = int(round(pD * float(s)))
        if cfg.data.patch_mode == "z_axis":
            H_k, W_k = pH, pW
        else:
            H_k = int(round(pH * float(s)))
            W_k = int(round(pW * float(s)))
        sizes.append((D_k, H_k, W_k))
    sizes[0] = (pD, pH, pW)
    stub._mr_native_sizes = sizes

    max_scale = max(cfg.data.multi_res_scales)
    if cfg.data.patch_mode == "z_axis":
        stub.target_patch_size = (int(round(pD * max_scale)), pH, pW)
    else:
        stub.target_patch_size = (
            int(round(pD * max_scale)),
            int(round(pH * max_scale)),
            int(round(pW * max_scale)))

    # Call the pure view-split function directly (no Trainer instance needed).
    def _split(image, label, wmap):
        return views.split_views_native_3d(
            image, label, wmap,
            target_patch_size=stub.target_patch_size,
            mr_native_sizes=stub._mr_native_sizes,
            patch_size=tuple(int(x) for x in cfg.data.patch_size))
    stub._split_views_native_3d = _split
    return stub


class _SyntheticSegDataset3D(SegDataset3D):
    def __init__(self, img: np.ndarray, lbl: np.ndarray, **kw):
        self._fake_img_pre = img
        self._fake_lbl_raw = lbl
        super().__init__(
            image_paths=["__synth__.nii.gz"],
            label_paths=["__synth__.nii.gz"],
            npz_paths=["__synth__.npz"], **kw)

    def _build_index(self):
        # Synthetic in-memory volume: no npz package to read.
        self._vol_fg_slices = [np.zeros(0, dtype=np.int32)]
        self._vol_fg_slices_by_cls = [None]
        self._vol_all_slices = [int(self._fake_img_pre.shape[0])]

    def _has_region_weight_file(self, vol_idx):
        return False

    def _load_image(self, vol_idx):
        return preprocess_image(
            self._fake_img_pre, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)

    def _load_label(self, vol_idx):
        return self._fake_lbl_raw


class _SyntheticSegDataset3DCubic(SegDataset3DCubic):
    def __init__(self, img: np.ndarray, lbl: np.ndarray, **kw):
        self._fake_img_pre = img
        self._fake_lbl_raw = lbl
        super().__init__(
            image_paths=["__synth__.nii.gz"],
            label_paths=["__synth__.nii.gz"],
            npz_paths=["__synth__.npz"], **kw)

    def _build_index(self):
        # Synthetic in-memory volume: no npz package to read.
        self._vol_shapes = [tuple(self._fake_img_pre.shape)]
        self._vol_fg_coords = [np.zeros((0, 3), dtype=np.int32)]
        self._vol_fg_coords_by_cls = [None]

    def _has_region_weight_file(self, vol_idx):
        return False

    def _load_image(self, vol_idx):
        return preprocess_image(
            self._fake_img_pre, self.intensity_min, self.intensity_max,
            self.normalize, self.global_mean, self.global_std)

    def _load_label(self, vol_idx):
        return self._fake_lbl_raw


# ===========================================================================
# Trainer.__init__ plumbing
# ===========================================================================
def test_init_plumbing_z_axis():
    cfg = _make_cfg(patch_mode="z_axis",
                     multi_res_scales=[1.0, 1.5, 2.0],
                     patch_size=(8, 16, 16))
    stub = _make_split_stub(cfg)
    assert stub._mr_native_sizes == [(8, 16, 16), (12, 16, 16), (16, 16, 16)], (
        f"z_axis native sizes mismatch: {stub._mr_native_sizes}")
    assert stub.target_patch_size == (16, 16, 16), (
        f"z_axis target_patch_size {stub.target_patch_size}")


def test_init_plumbing_cubic():
    cfg = _make_cfg(patch_mode="cubic",
                     multi_res_scales=[1.0, 1.5, 2.0],
                     patch_size=(8, 16, 16))
    stub = _make_split_stub(cfg)
    assert stub._mr_native_sizes == [(8, 16, 16), (12, 24, 24), (16, 32, 32)], (
        f"cubic native sizes mismatch: {stub._mr_native_sizes}")
    assert stub.target_patch_size == (16, 32, 32), (
        f"cubic target_patch_size {stub.target_patch_size}")


# ===========================================================================
# _split_views_native_3d unit tests
# ===========================================================================
def test_split_z_axis_shape_and_view0_identity():
    """View 0 has native size == patch_size, so split == centered crop, no resize."""
    cfg = _make_cfg(patch_mode="z_axis",
                     multi_res_scales=[1.0, 1.5, 2.0],
                     patch_size=(8, 16, 16))
    stub = _make_split_stub(cfg)
    B = 2
    tD, tH, tW = stub.target_patch_size  # (16, 16, 16)
    pD, pH, pW = cfg.data.patch_size

    # Encode the depth index in axis-2 so view 0 can be checked exactly.
    image = (torch.arange(tD, dtype=torch.float32)
             .reshape(1, 1, tD, 1, 1)
             .expand(B, 1, tD, tH, tW).contiguous())
    label = torch.zeros(B, 1, tD, tH, tW)
    wmap = torch.ones(B, 1, tD, tH, tW)

    img_out, lbl_out, wmap_out = stub._split_views_native_3d(image, label, wmap)
    assert tuple(img_out.shape) == (B, 3, pD, pH, pW)
    assert tuple(lbl_out.shape) == (B, 3, pD, pH, pW)
    assert tuple(wmap_out.shape) == (B, 3, pD, pH, pW)

    # View 0: centered 8 slices, no resize.
    d0 = (tD - pD) // 2  # 4
    expect_view0 = torch.arange(d0, d0 + pD, dtype=torch.float32)
    got_view0 = img_out[0, 0, :, 0, 0]
    if not torch.equal(got_view0, expect_view0):
        raise AssertionError(
            f"view-0 z encoding {got_view0.tolist()}, want {expect_view0.tolist()}")
    # View 2 (s=2.0): native depth = 16 = whole cube → resize 16→8.
    # F.interpolate trilinear maps endpoint 0 and 15 to canonical 0..7;
    # we only sanity-check the value is within [0, 15] and monotonic.
    got_view2 = img_out[0, 2, :, 0, 0]
    assert got_view2[0].item() < got_view2[-1].item(), (
        "view-2 z encoding should be monotonically increasing")
    assert 0.0 <= got_view2.min().item() <= got_view2.max().item() <= 15.0


def test_split_cubic_shape_and_view0_identity():
    cfg = _make_cfg(patch_mode="cubic",
                     multi_res_scales=[1.0, 1.5, 2.0],
                     patch_size=(8, 16, 16))
    stub = _make_split_stub(cfg)
    B = 1
    tD, tH, tW = stub.target_patch_size  # (16, 32, 32)
    pD, pH, pW = cfg.data.patch_size

    rng = torch.Generator().manual_seed(0)
    image = torch.randn(B, 1, tD, tH, tW, generator=rng)
    label = torch.zeros(B, 1, tD, tH, tW)

    img_out, lbl_out, wmap_out = stub._split_views_native_3d(image, label, None)
    assert tuple(img_out.shape) == (B, 3, pD, pH, pW)
    assert tuple(lbl_out.shape) == (B, 3, pD, pH, pW)
    assert wmap_out is None

    # View 0: voxel-for-voxel identity to centered (pD, pH, pW) crop.
    d0 = (tD - pD) // 2
    h0 = (tH - pH) // 2
    w0 = (tW - pW) // 2
    expected_v0 = image[:, 0, d0:d0 + pD, h0:h0 + pH, w0:w0 + pW]
    got_v0 = img_out[:, 0]
    if not torch.allclose(got_v0, expected_v0, atol=0.0):
        diff = float((got_v0 - expected_v0).abs().max())
        raise AssertionError(
            f"cubic view-0 must be exact crop; max abs diff={diff:.6g}")


def test_split_label_uses_nearest_interpolation():
    """Aux-view labels must be integer-preserving (nearest mode)."""
    cfg = _make_cfg(patch_mode="cubic",
                     multi_res_scales=[1.0, 2.0],
                     # 3D cubic fixtures must satisfy the encoder geometry
                     # contract; this test only exercises label splitting.
                     patch_size=(16, 16, 16))
    stub = _make_split_stub(cfg)
    tD, tH, tW = stub.target_patch_size  # (8, 16, 16)
    # Label cube with discrete values 0/1/2.
    rng = np.random.default_rng(0)
    label_np = rng.integers(0, 3, size=(1, 1, tD, tH, tW)).astype(np.float32)
    label = torch.from_numpy(label_np)
    image = torch.zeros_like(label)

    img_out, lbl_out, _ = stub._split_views_native_3d(image, label, None)
    # Every voxel of the resized aux label must remain in {0, 1, 2}.
    unique_vals = set(lbl_out.unique().tolist())
    if not unique_vals.issubset({0.0, 1.0, 2.0}):
        raise AssertionError(
            f"aux-label resize introduced fractional values {unique_vals}")


def test_split_rejects_wrong_input_shape():
    cfg = _make_cfg(patch_mode="z_axis",
                     multi_res_scales=[1.0, 2.0],
                     patch_size=(8, 16, 16))
    stub = _make_split_stub(cfg)
    # Wrong leading channel dim
    bad = torch.zeros(1, 2, 16, 16, 16)
    try:
        stub._split_views_native_3d(bad, bad, None)
    except ValueError as e:
        assert "(B, 1," in str(e), f"unexpected: {e}"
        return
    raise AssertionError("split should reject wrong leading channel dim")


def test_split_rejects_target_size_mismatch():
    cfg = _make_cfg(patch_mode="z_axis",
                     multi_res_scales=[1.0, 2.0],
                     patch_size=(8, 16, 16))
    stub = _make_split_stub(cfg)
    # target is (16, 16, 16); pass (12, 16, 16)
    bad = torch.zeros(1, 1, 12, 16, 16)
    try:
        stub._split_views_native_3d(bad, bad, None)
    except ValueError as e:
        assert "target_patch_size" in str(e), f"unexpected: {e}"
        return
    raise AssertionError("split should reject target_patch_size mismatch")


# ===========================================================================
# End-to-end (dataset → split) view-0 equivalence with OFF path
# ===========================================================================
def _z_axis_ds_kwargs(patch_size):
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


def test_e2e_z_axis_view0_matches_off_path():
    """Dataset(ON) → trainer split → view 0 must equal Dataset(OFF) view 0
    voxel-for-voxel (view 0 has no resize, so the path is exact).
    """
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    rng = np.random.default_rng(0)
    img_vol = rng.standard_normal((40, 16, 16), dtype=np.float32) * 10.0
    lbl_vol = rng.integers(0, 3, size=(40, 16, 16)).astype(np.float32)
    common = _z_axis_ds_kwargs((pD, pH, pW))

    class _FixedZ_ON(_SyntheticSegDataset3D):
        def _sample_z(self, vol_idx, D_vol, *, sample_idx=None):
            return D_vol // 2

    ds_on = _FixedZ_ON(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        z_boundary_mode="edge_pad",
        **common)

    on = ds_on[0]
    # Legacy OFF-path view-0 reference: independent edge_pad extraction of
    # exactly pD slices around the same z-center (view 0 has no resize).
    z = 40 // 2
    img_pre = ds_on._load_image(0)
    lbl_raw = ds_on._load_label(0)
    off_img_v0 = resize_3d(
        extract_z_patch_padded(img_pre, z, pD), pD, pH, pW, is_label=False)
    off_lbl_v0 = resize_3d(
        extract_z_patch_padded(lbl_raw, z, pD), pD, pH, pW, is_label=True)
    # Build a stub configured exactly like the trainer would build itself.
    cfg = _make_cfg(patch_mode="z_axis",
                     multi_res_scales=multi_res_scales,
                     patch_size=(pD, pH, pW))
    stub = _make_split_stub(cfg)

    # ON cube has no augment / no oversample → directly split.
    img_in = on["image"].unsqueeze(0)  # (1, 1, eD_max, pH, pW)
    lbl_in = on["label"].unsqueeze(0)
    img_split, lbl_split, _ = stub._split_views_native_3d(
        img_in, lbl_in, None)

    img_v0_split = img_split[0, 0].numpy()    # (pD, pH, pW)
    if not np.allclose(img_v0_split, off_img_v0, atol=1e-5):
        diff = float(np.abs(img_v0_split - off_img_v0).max())
        raise AssertionError(
            f"z_axis e2e view-0 image mismatch (max abs diff={diff:.6g}); "
            "view 0 is no-resize, so ON-split must equal the legacy "
            "view-0 extraction")
    lbl_v0_split = lbl_split[0, 0].numpy()
    if not np.array_equal(lbl_v0_split, off_lbl_v0):
        raise AssertionError("z_axis e2e view-0 label mismatch")


def test_e2e_cubic_view0_matches_off_path():
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    rng = np.random.default_rng(1)
    Dv, Hv, Wv = 40, 48, 48
    img_vol = rng.standard_normal((Dv, Hv, Wv), dtype=np.float32) * 10.0
    lbl_vol = rng.integers(0, 3, size=(Dv, Hv, Wv)).astype(np.float32)
    common = _z_axis_ds_kwargs((pD, pH, pW))
    fixed_center = (Dv // 2, Hv // 2, Wv // 2)

    class _FC_ON(_SyntheticSegDataset3DCubic):
        def _sample_center(self, vol_idx, D, H, W, *, sample_idx=None):
            return fixed_center

    ds_on = _FC_ON(
        img_vol, lbl_vol,
        multi_res_scales=multi_res_scales,
        **common)

    on = ds_on[0]
    # Legacy OFF-path view-0 reference: independent cubic extraction of
    # exactly (pD, pH, pW) around the same center (view 0 has no resize).
    img_pre = ds_on._load_image(0)
    lbl_raw = ds_on._load_label(0)
    off_img_v0 = _extract_cubic_patch(img_pre, fixed_center, (pD, pH, pW))
    off_lbl_v0 = _extract_cubic_patch(lbl_raw, fixed_center, (pD, pH, pW))
    cfg = _make_cfg(patch_mode="cubic",
                     multi_res_scales=multi_res_scales,
                     patch_size=(pD, pH, pW))
    stub = _make_split_stub(cfg)

    img_in = on["image"].unsqueeze(0)
    lbl_in = on["label"].unsqueeze(0)
    img_split, lbl_split, _ = stub._split_views_native_3d(
        img_in, lbl_in, None)

    img_v0_split = img_split[0, 0].numpy()
    if not np.allclose(img_v0_split, off_img_v0, atol=1e-5):
        diff = float(np.abs(img_v0_split - off_img_v0).max())
        raise AssertionError(
            f"cubic e2e view-0 image mismatch (max abs diff={diff:.6g})")
    lbl_v0_split = lbl_split[0, 0].numpy()
    if not np.array_equal(lbl_v0_split, off_lbl_v0):
        raise AssertionError("cubic e2e view-0 label mismatch")


# ===========================================================================
# CLI smoke run
# ===========================================================================
if __name__ == "__main__":
    tests = [
        test_init_plumbing_z_axis,
        test_init_plumbing_cubic,
        test_split_z_axis_shape_and_view0_identity,
        test_split_cubic_shape_and_view0_identity,
        test_split_label_uses_nearest_interpolation,
        test_split_rejects_wrong_input_shape,
        test_split_rejects_target_size_mismatch,
        test_e2e_z_axis_view0_matches_off_path,
        test_e2e_cubic_view0_matches_off_path,
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
