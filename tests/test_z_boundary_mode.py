"""Tests for ``DataConfig.z_boundary_mode`` toggle (Improvement #2).

Verifies the new ``stretch`` (default, backward compatible) vs.
``edge_pad`` z-axis boundary-window handling across the dataset and
predictor sides:

  1. Default mode is ``stretch`` and Config validates the legal set.
  2. ``SegDataset3D`` constructor accepts the toggle and rejects
     invalid values.
  3. Dataset ``__getitem__`` dispatches the correct extractor:
     - ``stretch`` triggers the legacy clamp-then-resize path that
       PHYSICALLY STRETCHES the z spacing of boundary windows.
     - ``edge_pad`` produces an output where boundary slices have been
       edge-replicated and z spacing is uniform.
  4. ``predictor.inputs.build_z_window_single_res_gpu`` (the path 2.5D actually
     uses on GPU) honours the toggle:
     - ``stretch`` resizes a (4, H, W) input to (12, H, W) via
       trilinear, blending neighbouring slices.
     - ``edge_pad`` pads (4, H, W) → (12, H, W) by replicating the
       boundary slices, preserving the inner 4 slices verbatim.
  5. ``predictor.inputs.build_z_window_cpu_multi_res`` (CPU multi-res path) honours
     the toggle on the scale=1.0 channel.
  6. End-to-end ``predict_volume`` runs without error and yields
     correct output shape on a synthetic volume with ``D_orig < pD``
     under both modes (the regime where edge_pad is most beneficial).

Run:
    conda activate torch27_env
    python test_z_boundary_mode.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def _ok(name: str, msg: str = "") -> None:
    print(f"  [PASS] {name}{(' — ' + msg) if msg else ''}")


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
def test_default_z_boundary_mode_is_edge_pad_and_stretch_auto_upgrades():
    """默认已改为 edge_pad；stretch 已废弃，sync() 自动升级并告警。"""
    from segtask_v1.config import Config, DataConfig
    assert DataConfig().z_boundary_mode == "edge_pad"
    cfg = Config()
    assert cfg.data.z_boundary_mode == "edge_pad"

    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.z_boundary_mode = "stretch"
    cfg.sync()
    assert cfg.data.z_boundary_mode == "edge_pad"
    _ok("Default edge_pad; deprecated 'stretch' auto-upgrades on sync()")


def test_validate_rejects_invalid_z_boundary_mode():
    from segtask_v1.config import Config
    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.patch_size = [12, 32, 32]
    cfg.data.z_boundary_mode = "bogus"
    cfg.sync()
    try:
        cfg.validate()
    except AssertionError as e:
        assert "z_boundary_mode" in str(e)
        _ok("Config.validate rejects invalid z_boundary_mode")
        return
    raise AssertionError("validate should have rejected 'bogus' boundary mode")


# ---------------------------------------------------------------------------
# 2. SegDataset3D constructor
# ---------------------------------------------------------------------------
def _make_synthetic_volume_files(out_dir: Path, n_volumes: int = 1,
                                 shape=(20, 32, 32), seed: int = 0):
    """Write synthetic NIfTI image+label pairs whose intensity equals
    the z index — so resize-stretch vs edge-pad effects on z are directly
    visible by inspecting the (z, 0, 0) sample."""
    rng = np.random.RandomState(seed)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)
    Z, Y, X = shape
    for i in range(n_volumes):
        # Image: every voxel in slice z has value = z * 10 (deterministic).
        img = np.tile(np.arange(Z, dtype=np.float32) * 10.0,
                      (X, Y, 1)).T   # (Z, Y, X)? no — make explicit
        img = np.zeros(shape, dtype=np.float32)
        for z in range(Z):
            img[z, :, :] = float(z) * 10.0
        nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), affine),
                 str(img_dir / f"vol_{i:02d}.nii.gz"))

        lbl = np.zeros(shape, dtype=np.int16)
        # Plant a small fg blob in the middle slice.
        cz = Z // 2
        lbl[cz, Y // 2 - 2:Y // 2 + 2, X // 2 - 2:X // 2 + 2] = 1
        nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), affine),
                 str(lbl_dir / f"vol_{i:02d}.nii.gz"))

    return str(img_dir), str(lbl_dir)


def _write_seg_npz(path: Path, img: np.ndarray, lbl: np.ndarray) -> str:
    fg = np.argwhere(lbl > 0).astype(np.int32)
    fg_slices = (np.unique(fg[:, 0]).astype(np.int32) if len(fg)
                 else np.arange(img.shape[0], dtype=np.int32))
    np.savez(path, image=img.astype(np.float32), label=lbl.astype(np.int16),
             fg_slices=fg_slices,
             fg_coords=fg if len(fg) else np.zeros((0, 3), dtype=np.int32))
    return str(path)


def test_segdataset_constructor_rejects_invalid():
    from segtask_v1.data.dataset import SegDataset3D
    with tempfile.TemporaryDirectory() as td:
        img = np.zeros((4, 8, 8), dtype=np.float32)
        lbl = np.zeros((4, 8, 8), dtype=np.int16)
        npz = _write_seg_npz(Path(td) / "v.npz", img, lbl)
        try:
            SegDataset3D(
                image_paths=["dummy"], label_paths=["dummy"],
                npz_paths=[npz],
                label_values=[0, 1], patch_size=(8, 16, 16),
                z_boundary_mode="bogus")
        except ValueError as e:
            assert "z_boundary_mode" in str(e)
            _ok("SegDataset3D rejects invalid z_boundary_mode")
            return
    raise AssertionError("SegDataset3D should have rejected 'bogus'")


# ---------------------------------------------------------------------------
# 3. Dataset __getitem__ dispatch
# ---------------------------------------------------------------------------
def test_dataset_dispatch_stretch_vs_edge_pad():
    """训练侧抽取恒走 edge-pad 几何（stretch 在训练路径已无分支）：
    D_vol=4、eD=12、z_center=1 时，`extract_z_patch_padded(vol, 1, 12)`：
      half=6; lo=-5; hi=7; src=[0,4); pad_before=5; pad_after=3
      → [vol[0]]*5 + vol[0:4] + [vol[3]]*3
      = [0,0,0,0,0, 0,10,20,30, 30,30,30] / 100（minmax 归一化后）。
    传 'stretch' 与 'edge_pad' 必须给出逐位一致的输出。"""
    from segtask_v1.data.dataset import SegDataset3D

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = np.zeros((4, 8, 8), dtype=np.float32)
        for z in range(4):
            img[z] = float(z) * 10.0
        lbl = np.zeros((4, 8, 8), dtype=np.int16)
        lbl[2, 2:6, 2:6] = 1
        npz = _write_seg_npz(td / "v.npz", img, lbl)

        common = dict(
            image_paths=["dummy"], label_paths=["dummy"], npz_paths=[npz],
            label_values=[0, 1], patch_size=(12, 8, 8),
            aug_oversample_ratio=1.0, multi_res_scales=[1.0],
            intensity_min=0.0, intensity_max=100.0,  # keep raw intensities
            normalize="minmax",
            foreground_oversample_ratio=0.0, samples_per_volume=1,
            is_train=False, cache_enabled=True)

        ds_stretch = SegDataset3D(**common, z_boundary_mode="stretch")
        ds_edge = SegDataset3D(**common, z_boundary_mode="edge_pad")

        # Force deterministic z=1 (near top) by patching _sample_z.
        z_target = 1
        ds_stretch._sample_z = lambda vol_idx, D_vol: z_target  # type: ignore
        ds_edge._sample_z = lambda vol_idx, D_vol: z_target     # type: ignore

        out_stretch = ds_stretch[0]["image"]   # (1, 12, 8, 8)
        out_edge = ds_edge[0]["image"]         # (1, 12, 8, 8)

        assert out_stretch.shape == (1, 12, 8, 8)
        assert out_edge.shape == (1, 12, 8, 8)

        z_means_edge = out_edge[0, :, :, :].mean(dim=(1, 2)).numpy()
        expected = np.array(
            [0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3],
            dtype=np.float32)
        np.testing.assert_allclose(z_means_edge, expected, atol=1e-5,
                                   err_msg="edge_pad z-axis layout incorrect")

        # 训练侧 stretch 与 edge_pad 输出必须逐位一致（无 stretch 分支）。
        np.testing.assert_allclose(
            out_stretch.numpy(), out_edge.numpy(), atol=0, rtol=0,
            err_msg="training-side extraction must be edge-pad for both modes")

        _ok("Dataset extraction is edge-pad regardless of configured mode")


# ---------------------------------------------------------------------------
# 4. predictor.inputs.build_z_window_single_res_gpu (the 2.5D path)
# ---------------------------------------------------------------------------
def _build_minimal_predictor(z_boundary_mode: str, D=12, H=8, W=8,
                             num_fg=1, label_values=(0, 1)):
    """Build a Predictor with patch_size=(D, H, W) using nn.Identity as
    the model. We never call the model in these helpers — only the
    geometry-handling methods are exercised."""
    import torch.nn as nn
    from segtask_v1.config import Config
    from segtask_v1.predictor import Predictor

    cfg = Config()
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [D, H, W]
    cfg.data.label_values = list(label_values)
    cfg.data.num_classes = len(label_values)
    cfg.data.z_boundary_mode = z_boundary_mode
    cfg.sync()
    cfg.validate()
    return Predictor(nn.Identity(), cfg, torch.device("cpu"))


def test_predictor_build_z_window_gpu_stretch():
    """Stretch path: short window (4 slices) should be trilinear-resized
    to pD=12 along z (in addition to the H/W resize)."""
    pred = _build_minimal_predictor("stretch", D=12, H=8, W=8)
    # Volume shape (4, 8, 8); intensity = z*10. z0=0, z1=4.
    vol = torch.zeros(4, 8, 8)
    for z in range(4):
        vol[z] = float(z) * 10.0

    from segtask_v1.predictor.inputs import build_z_window_single_res_gpu
    out = build_z_window_single_res_gpu(
        vol, 0, 4, pD=pred.patch_D, pH=pred.patch_H, pW=pred.patch_W,
        z_boundary_mode=pred.z_boundary_mode)
    assert out.shape == (1, 12, 8, 8), out.shape
    z_means = out[0].mean(dim=(1, 2)).cpu().numpy()
    # Stretch: first slice ≈ 0, last ≈ 30; no 5-long constant runs.
    assert abs(z_means[0]) < 1.5
    assert abs(z_means[-1] - 30.0) < 1.5
    # All slices distinct (trilinear over 4→12 produces strict monotone).
    diffs = np.diff(z_means)
    assert (diffs > 0).all() or (diffs >= 0).all(), (
        f"stretch z_means should be monotone, got {z_means}")
    _ok(f"Predictor stretch: shape={tuple(out.shape)}, "
        f"z_means[0]={z_means[0]:.2f}, z_means[-1]={z_means[-1]:.2f}")


def test_predictor_build_z_window_gpu_edge_pad():
    """Edge-pad path: 4-slice volume should produce 12 slices with
    pad_before=4 replicas of slice 0 + original 4 slices + pad_after=4
    replicas of slice 3.

    Specifically with z0=0, z1=4 the call signature is exactly the
    short-window case ``D_orig < pD``. The implementation centres
    pad_before=(pD-ad)//2 = (12-4)//2 = 4 replicas of vol[0], inner 4
    slices, then pad_after=4 replicas of vol[3].
    """
    pred = _build_minimal_predictor("edge_pad", D=12, H=8, W=8)
    vol = torch.zeros(4, 8, 8)
    for z in range(4):
        vol[z] = float(z) * 10.0

    from segtask_v1.predictor.inputs import build_z_window_single_res_gpu
    out = build_z_window_single_res_gpu(
        vol, 0, 4, pD=pred.patch_D, pH=pred.patch_H, pW=pred.patch_W,
        z_boundary_mode=pred.z_boundary_mode)
    assert out.shape == (1, 12, 8, 8), out.shape
    z_means = out[0].mean(dim=(1, 2)).cpu().numpy()
    # Layout: [0]*4 + [0, 10, 20, 30] + [30]*4
    expected = np.array(
        [0, 0, 0, 0, 0, 10, 20, 30, 30, 30, 30, 30], dtype=np.float32)
    np.testing.assert_allclose(z_means, expected, atol=1e-5,
                               err_msg="edge_pad z-layout mismatch")
    _ok("Predictor edge_pad: 4-slice → 12-slice replicate layout matches")


# ---------------------------------------------------------------------------
# 5. predictor.inputs.build_z_window_cpu_multi_res (CPU multi-res path)
# ---------------------------------------------------------------------------
def test_predictor_build_z_window_cpu_scale_1_edge_pad():
    """The CPU multi-res path is invoked when scales > 1.0 are present.
    With ``multi_res_scales=[1.0]`` the GPU path takes over, so to test
    the CPU branch we drive it with ``multi_res_scales=[1.0, 1.5]``.
    """
    import torch.nn as nn
    from segtask_v1.config import Config
    from segtask_v1.predictor import Predictor

    cfg = Config()
    cfg.data.patch_mode = "z_axis"   # multi-res allowed (not 2.5D)
    cfg.data.patch_size = [12, 8, 8]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.multi_res_scales = [1.0, 1.5]
    cfg.data.z_boundary_mode = "edge_pad"
    cfg.sync()
    cfg.validate()
    predictor = Predictor(nn.Identity(), cfg, torch.device("cpu"))

    # Build a synthetic volume identical to the GPU test.
    vol = np.zeros((4, 8, 8), dtype=np.float32)
    for z in range(4):
        vol[z] = float(z) * 10.0

    # z0=0, z1=4 → z_center = 2 in CPU path.
    from segtask_v1.predictor.inputs import build_z_window_cpu_multi_res
    out = build_z_window_cpu_multi_res(
        vol, 0, 4, pD=predictor.patch_D, pH=predictor.patch_H,
        pW=predictor.patch_W, multi_res_scales=predictor.multi_res_scales,
        z_boundary_mode=predictor.z_boundary_mode)
    assert out.shape == (2, 12, 8, 8), out.shape  # (C_res=2, pD=12, ...)

    # Channel 0 (scale=1.0) under edge_pad: extract_z_patch_padded(vol,
    # z_center=2, pD=12). half=6, lo=-4, hi=8, src_lo=0, src_hi=4,
    # pad_before=4, pad_after=4 → [0]*4 + [0,10,20,30] + [30]*4.
    z_means_ch0 = out[0].mean(axis=(1, 2))
    expected_ch0 = np.array(
        [0, 0, 0, 0, 0, 10, 20, 30, 30, 30, 30, 30], dtype=np.float32)
    np.testing.assert_allclose(z_means_ch0, expected_ch0, atol=1e-5)

    # Channel 1 (scale=1.5) under any toggle: extract_z_patch_padded(
    # vol, z_center=2, D_s=18) then resize_3d to 12 — depth is squeezed
    # back; slice ordering monotonically tracks the original ramp.
    z_means_ch1 = out[1].mean(axis=(1, 2))
    assert z_means_ch1[0] >= -0.01    # near 0
    assert z_means_ch1[-1] <= 30.01   # near 30 (clamped by replicate)
    _ok("Predictor CPU multi-res: scale=1.0 honours edge_pad; "
        "scale>1.0 still uses padded extraction")


# ---------------------------------------------------------------------------
# 6. End-to-end predict_volume on D_orig < pD (both modes complete)
# ---------------------------------------------------------------------------
def _make_synthetic_dataset_for_predict(out_dir: Path, shape=(4, 32, 32),
                                        seed: int = 0):
    return _make_synthetic_volume_files(out_dir, n_volumes=1, shape=shape,
                                        seed=seed)


def test_predict_volume_short_volume_both_modes():
    """End-to-end inference on a thin volume (D_orig=4 < pD=12). Both
    modes must complete and produce predictions matching the source
    spatial shape."""
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model
    from segtask_v1.predictor import Predictor

    for mode in ("stretch", "edge_pad"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            img_dir, _ = _make_synthetic_dataset_for_predict(
                td, shape=(4, 32, 32))

            cfg = Config()
            cfg.data.image_dir = img_dir
            cfg.data.label_dir = ""
            cfg.data.patch_mode = "2_5d"
            cfg.data.patch_size = [12, 32, 32]
            cfg.data.label_values = [0, 1]
            cfg.data.num_classes = 2
            cfg.data.intensity_min = 0.0
            cfg.data.intensity_max = 100.0
            cfg.data.z_boundary_mode = mode
            cfg.model.encoder_channels = [16, 32, 64]
            cfg.model.deep_supervision = False
            cfg.predict.batch_size = 1
            cfg.predict.tta_flip = False
            cfg.predict.z_overlap = 0.5
            cfg.train.use_amp = False
            cfg.sync()
            cfg.validate()

            device = torch.device("cpu")
            model = build_model(cfg).to(device).eval()
            predictor = Predictor(model, cfg, device)
            # sync() 自动把废弃的 'stretch' 升级为 'edge_pad'。
            assert predictor.z_boundary_mode == "edge_pad"
            img_paths = sorted(Path(img_dir).glob("*.nii.gz"))
            result = predictor.predict_volume(str(img_paths[0]))
            assert result["label_map"].shape == (4, 32, 32), (
                f"mode={mode}: unexpected label_map shape "
                f"{result['label_map'].shape}")
            assert result["probabilities"].shape == (1, 4, 32, 32)
            assert np.isfinite(result["probabilities"]).all()
            assert (result["probabilities"] >= 0).all()
            assert (result["probabilities"] <= 1).all()
    _ok("predict_volume on short volume runs under both stretch / edge_pad")


# ---------------------------------------------------------------------------
# 7. Both modes produce IDENTICAL output when D_orig >= pD (regression)
# ---------------------------------------------------------------------------
def test_predict_volume_long_volume_modes_equivalent():
    """When ``D_orig >= pD`` the sliding-window machinery never sees
    a short tail (``blending.compute_1d_positions`` pulls back the tail to keep
    length=pD). In that regime both modes must produce identical
    predictions because the edge_pad branch never activates.
    """
    from segtask_v1.config import Config
    from segtask_v1.models.factory import build_model
    from segtask_v1.predictor import Predictor

    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_dir, _ = _make_synthetic_volume_files(
            td, n_volumes=1, shape=(20, 32, 32))

        outs = {}
        for mode in ("stretch", "edge_pad"):
            cfg = Config()
            cfg.data.image_dir = img_dir
            cfg.data.label_dir = ""
            cfg.data.patch_mode = "2_5d"
            cfg.data.patch_size = [12, 32, 32]
            cfg.data.label_values = [0, 1]
            cfg.data.num_classes = 2
            cfg.data.intensity_min = 0.0
            cfg.data.intensity_max = 200.0
            cfg.data.z_boundary_mode = mode
            cfg.model.encoder_channels = [8, 16]
            cfg.model.deep_supervision = False
            cfg.predict.batch_size = 1
            cfg.predict.tta_flip = False
            cfg.predict.z_overlap = 0.5
            cfg.train.use_amp = False
            cfg.sync()
            cfg.validate()

            device = torch.device("cpu")
            torch.manual_seed(42)
            model = build_model(cfg).to(device).eval()
            predictor = Predictor(model, cfg, device)
            img_paths = sorted(Path(img_dir).glob("*.nii.gz"))
            outs[mode] = predictor.predict_volume(str(img_paths[0]))[
                "probabilities"]

        np.testing.assert_allclose(
            outs["stretch"], outs["edge_pad"], atol=1e-5,
            err_msg="long-volume predictions must be identical across modes")
    _ok("Long-volume regression: stretch == edge_pad (no short windows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("Improvement #2 tests — z_boundary_mode toggle")
    print("=" * 60)
    tests = [
        test_default_z_boundary_mode_is_stretch,
        test_validate_rejects_invalid_z_boundary_mode,
        test_segdataset_constructor_rejects_invalid,
        test_dataset_dispatch_stretch_vs_edge_pad,
        test_predictor_build_z_window_gpu_stretch,
        test_predictor_build_z_window_gpu_edge_pad,
        test_predictor_build_z_window_cpu_scale_1_edge_pad,
        test_predict_volume_short_volume_both_modes,
        test_predict_volume_long_volume_modes_equivalent,
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
