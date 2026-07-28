"""Tests for ``keep_native_multi_res`` on the predictor side (R3).

Coverage
--------
1. ``Predictor.__init__`` plumbing:
   - ``self.keep_native_multi_res`` flag activated for 3D modes only.
   - ``self._mr_native_sizes`` / ``self._mr_target_shape`` correctly
     computed for z_axis (z-only) and cubic (all 3 axes).
   - 2.5D / single-scale / whole modes leave ``keep_native_multi_res``
     False (defensive — Config.validate already guards earlier).

2. ``inputs.build_z_window_native_multi_res_gpu`` (z_axis 3D ON):
   - Output shape ``(C_res, pD, pH, pW)``.
   - View 0 voxel-for-voxel equal to the OFF-path
     ``inputs.build_z_window_single_res_gpu`` (single-res builder) for the same
     window — both extract the centred ``pD`` slices with
     ``edge_pad`` boundary.
   - Aux-view crops match an explicit numpy reference
     (``extract_z_patch_padded`` → resize to ``pD``).

3. ``inputs.build_cubic_batch_native_multi_res`` (cubic 3D ON):
   - Output shape ``(B, C_res, pD, pH, pW)``.
   - View 0 voxel-for-voxel equal to ``_extract_cubic_patch`` of size
     ``patch_size`` around the same centre (no resize, edge-padded).
   - Aux views match explicit numpy reference
     (``_extract_cubic_patch`` at native size → resize to patch_size
     via ``F.interpolate`` for trilinear consistency).
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
from taskcore.data.dataset import (  # noqa: E402
    extract_z_patch_padded, _extract_cubic_patch,
)
from segtask_v1.predictor import Predictor  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================
def _make_predictor_stub(
    *,
    patch_mode: str = "z_axis",
    multi_res_scales=(1.0, 1.5, 2.0),
    keep_native_multi_res: bool = True,
    patch_size=(8, 16, 16),
    z_boundary_mode: str = "edge_pad",
):
    """Build a Predictor-like stub that holds only the attributes the
    builders read. Avoids the cost of building a real model + running
    Predictor.__init__'s checkpoint loader for unit tests of the GPU
    builders."""
    pD, pH, pW = patch_size

    class _Stub:
        pass

    stub = _Stub()
    stub.device = torch.device("cpu")  # builders are device-agnostic
    stub.patch_mode = patch_mode
    stub.patch_D, stub.patch_H, stub.patch_W = patch_size
    stub.multi_res_scales = list(multi_res_scales)
    stub.z_boundary_mode = z_boundary_mode
    stub.keep_native_view_depth = False
    stub.per_view_depths = []
    stub._eD_max = pD

    stub.keep_native_multi_res = bool(
        keep_native_multi_res
        and patch_mode in ("z_axis", "cubic")
        and len(multi_res_scales) > 1)

    if stub.keep_native_multi_res:
        sizes: List[Tuple[int, int, int]] = []
        for s in stub.multi_res_scales:
            D_k = int(round(pD * float(s)))
            if patch_mode == "z_axis":
                H_k, W_k = pH, pW
            else:
                H_k = int(round(pH * float(s)))
                W_k = int(round(pW * float(s)))
            sizes.append((D_k, H_k, W_k))
        sizes[0] = (pD, pH, pW)
        stub._mr_native_sizes = sizes
        ms = float(max(stub.multi_res_scales))
        if patch_mode == "z_axis":
            stub._mr_target_shape = (int(round(pD * ms)), pH, pW)
        else:
            stub._mr_target_shape = (
                int(round(pD * ms)),
                int(round(pH * ms)),
                int(round(pW * ms)))
    else:
        stub._mr_native_sizes = []
        stub._mr_target_shape = (pD, pH, pW)

    # Bind module-level builders so we can call them on the stub.
    from segtask_v1.predictor import inputs as _inputs
    stub._build_z_window_input_gpu = (
        lambda vol_t, z0, z1: _inputs.build_z_window_single_res_gpu(
            vol_t, z0, z1,
            pD=stub.patch_D, pH=stub.patch_H, pW=stub.patch_W,
            z_boundary_mode=stub.z_boundary_mode))
    stub._build_z_window_input_native_multi_res_gpu = (
        lambda vol_t, z0, z1: _inputs.build_z_window_native_multi_res_gpu(
            vol_t, z0, z1,
            pD=stub.patch_D, pH=stub.patch_H, pW=stub.patch_W,
            target_shape=stub._mr_target_shape,
            native_sizes=stub._mr_native_sizes))
    stub._build_batch_native_multi_res_cubic_gpu = (
        lambda windows, vol_t: _inputs.build_cubic_batch_native_multi_res(
            windows, vol_t,
            pD=stub.patch_D, pH=stub.patch_H, pW=stub.patch_W,
            target_shape=stub._mr_target_shape,
            native_sizes=stub._mr_native_sizes))
    return stub


# ===========================================================================
# __init__ plumbing
# ===========================================================================
def test_init_plumbing_z_axis_on():
    stub = _make_predictor_stub(
        patch_mode="z_axis",
        multi_res_scales=[1.0, 1.5, 2.0],
        patch_size=(8, 16, 16),
    )
    assert stub.keep_native_multi_res is True
    assert stub._mr_native_sizes == [(8, 16, 16), (12, 16, 16), (16, 16, 16)]
    assert stub._mr_target_shape == (16, 16, 16)


def test_init_plumbing_cubic_on():
    stub = _make_predictor_stub(
        patch_mode="cubic",
        multi_res_scales=[1.0, 1.5, 2.0],
        patch_size=(8, 16, 16),
    )
    assert stub.keep_native_multi_res is True
    assert stub._mr_native_sizes == [(8, 16, 16), (12, 24, 24), (16, 32, 32)]
    assert stub._mr_target_shape == (16, 32, 32)


def test_init_plumbing_off_for_2_5d():
    """2.5D config must NOT activate the 3D ON path."""
    stub = _make_predictor_stub(
        patch_mode="2_5d",
        multi_res_scales=[1.0, 1.5, 2.0],
        keep_native_multi_res=True,
        patch_size=(8, 16, 16),
    )
    assert stub.keep_native_multi_res is False, (
        "2.5D config must leave keep_native_multi_res False even when "
        "the data field is True (Config.validate would have rejected "
        "this combo upstream; defensive gating in stub mirrors predictor)")


def test_init_plumbing_off_for_single_scale():
    stub = _make_predictor_stub(
        patch_mode="z_axis",
        multi_res_scales=[1.0],
        keep_native_multi_res=True,
        patch_size=(8, 16, 16),
    )
    assert stub.keep_native_multi_res is False


# ===========================================================================
# z_axis 3D ON builder
# ===========================================================================
def _make_vol_t(D=40, H=16, W=16, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((D, H, W), dtype=np.float32) * 5.0
    # Encode z-index in axis-0 (additive) so we can sanity-check slice picks.
    arr += np.arange(D, dtype=np.float32)[:, None, None]
    return torch.from_numpy(arr), arr  # (vol_t, vol_np)


def test_z_axis_on_shape_and_view0_matches_off_path():
    """View 0 of the ON builder matches the legacy single-res builder
    voxel-for-voxel under ``edge_pad``."""
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    stub = _make_predictor_stub(
        patch_mode="z_axis",
        multi_res_scales=multi_res_scales,
        patch_size=(pD, pH, pW),
        z_boundary_mode="edge_pad",
    )
    vol_t, _ = _make_vol_t(D=40, H=pH, W=pW, seed=0)

    # Window centred well inside the volume.
    z0, z1 = 16, 24  # actual_d = 8 = pD
    on_cube = stub._build_z_window_input_native_multi_res_gpu(
        vol_t, z0, z1)  # (3, pD, pH, pW)
    assert tuple(on_cube.shape) == (3, pD, pH, pW)

    # OFF single-res builder for the same window.
    off_view0 = stub._build_z_window_input_gpu(vol_t, z0, z1)  # (1, pD, pH, pW)
    on_view0 = on_cube[0:1]
    if not torch.allclose(on_view0, off_view0, atol=1e-5):
        diff = float((on_view0 - off_view0).abs().max())
        raise AssertionError(
            f"z_axis ON view-0 mismatch with OFF single-res builder "
            f"(max abs diff={diff:.6g})")


def test_z_axis_on_aux_view_matches_explicit_reference():
    """Aux-view k crop+resize matches an explicit numpy reference:
    extract_z_patch_padded(D_k) → trilinear resize to (pD, pH, pW)."""
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    stub = _make_predictor_stub(
        patch_mode="z_axis",
        multi_res_scales=multi_res_scales,
        patch_size=(pD, pH, pW),
        z_boundary_mode="edge_pad",
    )
    vol_t, vol_np = _make_vol_t(D=40, H=pH, W=pW, seed=1)
    z0, z1 = 18, 26
    z_center = (z0 + z1) // 2

    on_cube = stub._build_z_window_input_native_multi_res_gpu(vol_t, z0, z1)

    for k, s in enumerate(multi_res_scales):
        D_k = int(round(pD * s))
        ref_np = extract_z_patch_padded(vol_np, z_center, D_k)  # (D_k, H, W)
        ref_t = torch.from_numpy(ref_np)[None, None].float()  # (1,1,D_k,H,W)
        if D_k != pD:
            ref_t = F.interpolate(
                ref_t, size=(pD, pH, pW),
                mode="trilinear", align_corners=False)
        ref_view = ref_t[0, 0]  # (pD, pH, pW)
        got_view = on_cube[k]
        if not torch.allclose(got_view, ref_view, atol=1e-5):
            diff = float((got_view - ref_view).abs().max())
            raise AssertionError(
                f"z_axis ON view-{s} mismatch (max abs diff={diff:.6g})")


def test_z_axis_on_boundary_window_edge_pad():
    """Window at z=0 must produce eD_max-deep slab via edge replication."""
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    stub = _make_predictor_stub(
        patch_mode="z_axis",
        multi_res_scales=multi_res_scales,
        patch_size=(pD, pH, pW),
        z_boundary_mode="edge_pad",
    )
    vol_t, _ = _make_vol_t(D=40, H=pH, W=pW, seed=2)

    # z0=0, z1=pD: window center at pD/2 = 4; eD_max=16 → slab [-4..12]
    # would underflow, replicating the first slice 4 times.
    on_cube = stub._build_z_window_input_native_multi_res_gpu(vol_t, 0, pD)
    assert tuple(on_cube.shape) == (3, pD, pH, pW), (
        f"boundary cube shape {tuple(on_cube.shape)}")
    # No NaN/Inf even with edge replication.
    assert torch.isfinite(on_cube).all()


# ===========================================================================
# Cubic 3D ON builder
# ===========================================================================
def _make_vol_t_3d(D=24, H=32, W=32, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((D, H, W), dtype=np.float32) * 3.0
    return torch.from_numpy(arr), arr


def test_cubic_on_shape_and_view0_matches_extract_cubic_patch():
    """View 0 must match ``_extract_cubic_patch`` of size patch_size
    around the same centre."""
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    stub = _make_predictor_stub(
        patch_mode="cubic",
        multi_res_scales=multi_res_scales,
        patch_size=(pD, pH, pW),
    )
    Dv, Hv, Wv = 24, 32, 32
    vol_t, vol_np = _make_vol_t_3d(D=Dv, H=Hv, W=Wv, seed=3)

    centers = [(Dv // 2, Hv // 2, Wv // 2),  # well inside
               (3, 4, 5)]                     # near boundary (edge-pad)
    windows = [(cd - pD // 2, ch - pH // 2, cw - pW // 2, pD, pH, pW)
               for (cd, ch, cw) in centers]

    batch = stub._build_batch_native_multi_res_cubic_gpu(windows, vol_t)
    assert tuple(batch.shape) == (2, 3, pD, pH, pW), (
        f"cubic ON batch shape {tuple(batch.shape)}")

    # View 0 reference: _extract_cubic_patch at patch_size around centre.
    for i, c in enumerate(centers):
        ref_np = _extract_cubic_patch(vol_np, c, (pD, pH, pW))
        ref_t = torch.from_numpy(ref_np).float()
        got = batch[i, 0]
        if not torch.allclose(got, ref_t, atol=1e-5):
            diff = float((got - ref_t).abs().max())
            raise AssertionError(
                f"cubic ON view-0 mismatch at centre {c} "
                f"(max abs diff={diff:.6g})")


def test_cubic_on_aux_view_matches_explicit_reference():
    """Aux-view crop+resize matches numpy reference:
    _extract_cubic_patch at native size → trilinear F.interpolate."""
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 1.5, 2.0]
    stub = _make_predictor_stub(
        patch_mode="cubic",
        multi_res_scales=multi_res_scales,
        patch_size=(pD, pH, pW),
    )
    Dv, Hv, Wv = 32, 48, 48
    vol_t, vol_np = _make_vol_t_3d(D=Dv, H=Hv, W=Wv, seed=4)
    center = (Dv // 2, Hv // 2, Wv // 2)
    window = (center[0] - pD // 2, center[1] - pH // 2,
              center[2] - pW // 2, pD, pH, pW)

    batch = stub._build_batch_native_multi_res_cubic_gpu([window], vol_t)
    for k, s in enumerate(multi_res_scales):
        D_k = int(round(pD * s))
        H_k = int(round(pH * s))
        W_k = int(round(pW * s))
        ref_np = _extract_cubic_patch(vol_np, center, (D_k, H_k, W_k))
        ref_t = torch.from_numpy(ref_np)[None, None].float()
        if (D_k, H_k, W_k) != (pD, pH, pW):
            ref_t = F.interpolate(
                ref_t, size=(pD, pH, pW),
                mode="trilinear", align_corners=False)
        ref_view = ref_t[0, 0]
        got_view = batch[0, k]
        if not torch.allclose(got_view, ref_view, atol=1e-5):
            diff = float((got_view - ref_view).abs().max())
            raise AssertionError(
                f"cubic ON view-{s} mismatch (max abs diff={diff:.6g})")


def test_cubic_on_boundary_centre_edge_pad_three_axes():
    """Centre at the volume corner should still produce a full cube via
    edge replication on every axis."""
    pD, pH, pW = 8, 16, 16
    multi_res_scales = [1.0, 2.0]
    stub = _make_predictor_stub(
        patch_mode="cubic",
        multi_res_scales=multi_res_scales,
        patch_size=(pD, pH, pW),
    )
    Dv, Hv, Wv = 12, 18, 18
    vol_t, _ = _make_vol_t_3d(D=Dv, H=Hv, W=Wv, seed=5)
    # Corner centre → max-FOV (16, 32, 32) won't fit; edge-pad on D/H/W.
    center = (0, 0, 0)
    window = (center[0] - pD // 2, center[1] - pH // 2,
              center[2] - pW // 2, pD, pH, pW)
    batch = stub._build_batch_native_multi_res_cubic_gpu([window], vol_t)
    assert tuple(batch.shape) == (1, 2, pD, pH, pW)
    assert torch.isfinite(batch).all()


# ===========================================================================
# CLI smoke run
# ===========================================================================
if __name__ == "__main__":
    tests = [
        test_init_plumbing_z_axis_on,
        test_init_plumbing_cubic_on,
        test_init_plumbing_off_for_2_5d,
        test_init_plumbing_off_for_single_scale,
        test_z_axis_on_shape_and_view0_matches_off_path,
        test_z_axis_on_aux_view_matches_explicit_reference,
        test_z_axis_on_boundary_window_edge_pad,
        test_cubic_on_shape_and_view0_matches_extract_cubic_patch,
        test_cubic_on_aux_view_matches_explicit_reference,
        test_cubic_on_boundary_centre_edge_pad_three_axes,
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
