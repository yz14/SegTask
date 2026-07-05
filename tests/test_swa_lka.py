"""Tests for SWA weight averaging (B1) and LKA / MSCA attention (C2).

Covers:
- LKA3D / MSCA3D: shape + grad flow (2D/3D), tiny-spatial legality,
  factory dispatch, config/manifest acceptance.
- ModelSWA: exact equal-average math, apply/restore round trip,
  state_dict round trip.
- Trainer end-to-end with ``swa_enabled=True`` on a BatchNorm model:
  swa_model.pth is produced, loads, and BN stats were re-estimated.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from segtask_v1.config import Config  # noqa: E402
from segtask_v1.models.blocks import (  # noqa: E402
    ATTENTION_TYPES, LKA3D, MSCA3D, make_attention,
)
from segtask_v1.utils import ModelSWA  # noqa: E402


# ---------------------------------------------------------------------------
# C2: LKA / MSCA attention modules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [LKA3D, MSCA3D])
@pytest.mark.parametrize("spatial_dims,shape", [
    (3, (2, 16, 4, 8, 8)),
    (2, (2, 16, 8, 8)),
])
def test_large_kernel_attention_shape_and_grad(cls, spatial_dims, shape):
    x = torch.randn(*shape, requires_grad=True)
    m = cls(16, spatial_dims=spatial_dims)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("name", ["lka", "msca"])
def test_large_kernel_attention_tiny_spatial(name):
    """Deep-stage feature maps smaller than the kernels must stay legal
    (symmetric k//2 padding guarantees output size == input size)."""
    m = make_attention(name, 8, spatial_dims=3)
    x = torch.randn(1, 8, 1, 2, 2)
    assert m(x).shape == x.shape


def test_factory_registers_lka_msca():
    assert "lka" in ATTENTION_TYPES and "msca" in ATTENTION_TYPES
    assert isinstance(make_attention("lka", 16), LKA3D)
    assert isinstance(make_attention("msca", 16), MSCA3D)


def test_msca_branch_count_matches_scales_and_dims():
    m3 = MSCA3D(8, spatial_dims=3, scales=(7, 11))
    assert len(m3.branches) == 2
    assert all(len(b) == 3 for b in m3.branches)  # one strip per axis
    m2 = MSCA3D(8, spatial_dims=2, scales=(7,))
    assert len(m2.branches) == 1 and len(m2.branches[0]) == 2


def test_config_accepts_lka_msca():
    cfg = Config()
    for name in ("lka", "msca"):
        cfg.model.attention_type = name
        cfg.validate()


# ---------------------------------------------------------------------------
# B1: ModelSWA unit tests
# ---------------------------------------------------------------------------
def _tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(1, 4, 3, padding=1), nn.BatchNorm3d(4), nn.Conv3d(4, 1, 1))


def test_model_swa_equal_average_math():
    m = _tiny_model()
    swa = ModelSWA(m)
    snaps = []
    for _ in range(3):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        snaps.append({k: v.clone() for k, v in m.state_dict().items()})
        swa.update(m)
    assert swa.n_averaged == 3
    for k, v in m.state_dict().items():
        if v.is_floating_point():
            expect = sum(s[k].to(torch.float32) for s in snaps) / 3
            assert torch.allclose(swa.shadow[k], expect, atol=1e-6), k
        else:
            assert torch.equal(swa.shadow[k], snaps[-1][k]), k


def test_model_swa_apply_restore_roundtrip():
    m = _tiny_model()
    swa = ModelSWA(m)
    swa.update(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    live = {k: v.clone() for k, v in m.state_dict().items()}
    swa.apply_shadow(m)
    for k, v in m.state_dict().items():
        assert torch.allclose(v.float(), swa.shadow[k].float(), atol=1e-6), k
    swa.apply_shadow(m)  # idempotent while swapped
    swa.restore(m)
    for k, v in m.state_dict().items():
        assert torch.equal(v, live[k]), k
    swa.restore(m)  # idempotent while restored


def test_model_swa_state_dict_roundtrip():
    m = _tiny_model()
    swa = ModelSWA(m)
    swa.update(m)
    swa.update(m)
    state = swa.state_dict()
    swa2 = ModelSWA(m)
    swa2.load_state_dict(state)
    assert swa2.n_averaged == 2
    for k, v in state["shadow"].items():
        assert torch.equal(swa2.shadow[k], v)


# ---------------------------------------------------------------------------
# B1: Trainer end-to-end with SWA (+ BN re-estimation path)
# ---------------------------------------------------------------------------
def _make_synthetic_dataset(out_dir: Path, n_volumes: int = 4,
                            shape=(20, 64, 64), num_fg: int = 2,
                            seed: int = 0):
    nib = pytest.importorskip("nibabel")
    rng = np.random.RandomState(seed)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    Z, Y, X = shape
    for i in range(n_volumes):
        img = rng.randn(*shape).astype(np.float32) * 50.0
        nib.save(nib.Nifti1Image(img.transpose(2, 1, 0), affine),
                 str(img_dir / f"vol_{i:02d}.nii.gz"))
        lbl = np.zeros(shape, dtype=np.int16)
        for c in range(num_fg):
            cz = rng.randint(2, Z - 2)
            cy = rng.randint(8, Y - 8)
            cx = rng.randint(8, X - 8)
            lbl[cz - 1:cz + 2, cy - 4:cy + 4, cx - 4:cx + 4] = c + 1
        nib.save(nib.Nifti1Image(lbl.transpose(2, 1, 0), affine),
                 str(lbl_dir / f"vol_{i:02d}.nii.gz"))
    return str(img_dir), str(lbl_dir)


def test_trainer_swa_end_to_end():
    from segtask_v1.data.loader import build_dataloaders
    from segtask_v1.data.make_data import prepare_dataset
    from segtask_v1.models.factory import build_model
    from segtask_v1.trainer import Trainer
    from torch.nn.modules.batchnorm import _BatchNorm

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_dir, lbl_dir = _make_synthetic_dataset(td)

        cfg = Config()
        cfg.data.image_dir = img_dir
        cfg.data.label_dir = lbl_dir
        cfg.data.npz_dir = str(td / "npz")
        cfg.data.patch_mode = "2_5d"
        cfg.data.patch_size = [12, 32, 32]
        cfg.data.label_values = [0, 1, 2]
        cfg.data.num_classes = 3
        cfg.data.multi_res_scales = [1.0]
        cfg.data.batch_size = 2
        cfg.data.num_workers = 0
        cfg.data.samples_per_volume = 1
        cfg.data.foreground_oversample_ratio = 1.0
        cfg.data.intensity_min = -200.0
        cfg.data.intensity_max = 200.0
        cfg.data.cache_mode = "memory"
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.norm_type = "batch"  # exercise BN re-estimation path
        cfg.model.deep_supervision = False
        cfg.augment.enabled = False
        cfg.train.epochs = 2
        cfg.train.use_amp = False
        cfg.train.use_ema = False
        cfg.train.warmup_epochs = 0
        cfg.train.compile_mode = "none"
        cfg.train.output_dir = str(td / "out")
        cfg.train.log_every = 1
        cfg.train.save_every = 9999
        cfg.train.val_every = 1
        cfg.train.swa_enabled = True
        cfg.train.swa_start_ratio = 0.5  # start epoch = floor(0.5*2) = 1
        cfg.train.swa_bn_update_steps = 2
        cfg.sync()
        cfg.validate()

        prepare_dataset(cfg, out_dir=cfg.data.npz_dir, workers=0)
        train_loader, val_loader = build_dataloaders(cfg)
        model = build_model(cfg)
        trainer = Trainer(model, cfg, train_loader, val_loader,
                          torch.device("cpu"))
        trainer.fit()

        assert trainer.swa is not None and trainer.swa.n_averaged == 1

        swa_path = td / "out" / "swa_model.pth"
        assert swa_path.exists(), "swa_model.pth was not saved"
        ckpt = torch.load(swa_path, map_location="cpu", weights_only=False)
        assert ckpt["swa_n_averaged"] == 1
        sd = ckpt["model_state_dict"]
        model2 = build_model(cfg)
        model2.load_state_dict(sd)

        # BN re-estimation happened: num_batches_tracked was reset to 0
        # and equals the number of forward batches actually run (1..2).
        bn = [m for m in model2.modules() if isinstance(m, _BatchNorm)]
        assert bn, "expected BatchNorm modules with norm_type='batch'"
        for m in bn:
            assert 1 <= int(m.num_batches_tracked) <= 2

        # SWA state rides along in training checkpoints for resume.
        state = trainer._build_state_dict(ema_as_primary=False)
        assert "swa_state_dict" in state
        assert state["swa_state_dict"]["n_averaged"] == 1
