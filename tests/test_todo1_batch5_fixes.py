"""批 5 回归：checkpoint、采样开关、初始化与架构几何。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_framework_checkpoint_config_dataclass_is_accepted(tmp_path: Path):
    from taskcore.config.core import Config
    from taskcore.engine.base_trainer import BaseTrainer

    cfg = Config()
    cfg.sync()
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "framework.pth"
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.cfg = cfg
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.ema = None
    trainer._load_pretrain_weights(
        str(path), strict=True, load_ema=False)

    assert ckpt["config"].__class__ is Config


def test_checkpoint_geometry_dict_and_missing_fields():
    from taskcore.engine.base_trainer import _checkpoint_geometry

    assert _checkpoint_geometry({
        "data": {"patch_mode": "z_axis"},
        "model": {"spatial_dims": 3, "in_channels": 1},
    }) == ("z_axis", 3, 1)
    assert _checkpoint_geometry({"data": {"patch_mode": "z_axis"}}) is None


def test_fractional_integer_override_is_rejected():
    from taskcore.config.core import ConfigError
    from taskcore.config.task_io import coerce_override_value

    assert coerce_override_value(1, "3.0") == 3
    with pytest.raises(ConfigError):
        coerce_override_value(1, "3.7")


def test_split_manifest_rank0_atomic_writer(tmp_path: Path):
    from taskcore.data.loader import _write_split_manifest

    path = tmp_path / "nested" / "split.json"
    _write_split_manifest(
        path, seed=42, val_ratio=0.2, rounding_mode="legacy",
        train=["a.npz"], val=["b.npz"], rank=1)
    assert not path.exists()
    _write_split_manifest(
        path, seed=42, val_ratio=0.2, rounding_mode="legacy",
        train=["a.npz"], val=["b.npz"], rank=0)
    assert json.loads(path.read_text(encoding="utf-8"))["val"] == ["b.npz"]
    assert not list(path.parent.glob("split.json.*.tmp"))


def test_legacy_z_sampling_matches_old_helpers():
    from taskcore.data.dataset import SegDataset3D
    from taskcore.data.sampling import safe_z_grid_center, z_grid_center

    assert z_grid_center(1, 4, 20) == 7
    assert safe_z_grid_center(1, 4, 20, 8) != z_grid_center(1, 4, 20)
    ds = SegDataset3D.__new__(SegDataset3D)
    ds.is_train = False
    ds.val_grid_coverage = True
    ds._sample_idx = 1
    ds.image_paths = ["x"]
    ds.samples_per_volume = 4
    ds.extract_size = (8, 8, 8)
    ds.z_sampling_mode = "legacy"
    assert ds._sample_z(0, 20) == z_grid_center(1, 4, 20)
    ds.z_sampling_mode = "safe"
    assert ds._sample_z(0, 20) == safe_z_grid_center(1, 4, 20, 8)


def test_z_sampling_mode_is_shared_by_cls_and_det():
    from clstask.data.cls_dataset import ClsPatchDataset
    from dettask.data.det_dataset import DetPatchDataset

    cls = ClsPatchDataset(
        ["x"], [8, 8, 8], 1, patch_mode="z_axis",
        fg_values=[1], z_sampling_mode="legacy")
    det = DetPatchDataset(
        ["x"], [8, 8, 8], patch_mode="z_axis",
        z_sampling_mode="legacy")
    rng = np.random.default_rng(0)
    assert cls._sample_z(rng, 20, 0, None, None) == 17
    assert det._sample_z(np.random.default_rng(0), 20,
                         np.zeros((0, 6), np.float32), None) == 17


def _assert_same_rng_result(call, reference):
    rng = np.random.default_rng(1234)
    ref_rng = np.random.default_rng(1234)
    assert call(rng) == reference(ref_rng)
    assert rng.integers(0, 100000) == ref_rng.integers(0, 100000)


def test_legacy_z_sampling_matches_batch1_reference_for_all_tasks():
    from clstask.data.cls_dataset import ClsPatchDataset
    from dettask.data.det_dataset import DetPatchDataset
    from taskcore.data.dataset import SegDataset3D
    from taskcore.data.sampling import safe_z_grid_center, z_grid_center

    seg = SegDataset3D.__new__(SegDataset3D)
    seg.is_train = True
    seg.val_grid_coverage = False
    seg.fg_ratio = 1.0
    seg._vol_fg_slices = [np.asarray([19], dtype=np.int32)]
    seg._vol_fg_slices_by_cls = [None]
    seg.z_sampling_mode = "legacy"
    seg.extract_size = (8, 8, 8)

    def seg_reference(rng):
        rng.random()
        return int(rng.choice(np.asarray([19], dtype=np.int32)))

    _assert_same_rng_result(
        lambda rng: (
            setattr(seg, "_sample_rng", lambda sample_idx: rng) or
            seg._sample_z(0, 20, sample_idx=0)),
        seg_reference)

    cls = ClsPatchDataset(
        ["x"], [8, 8, 8], 1, patch_mode="z_axis",
        fg_values=[1], z_sampling_mode="legacy")
    cls.is_train = True
    cls.fg_ratio = 1.0
    cls._fg_groups = lambda vol_idx, lbl: [
        np.asarray([19], dtype=np.int32)]

    def cls_reference(rng):
        rng.random()
        rng.integers(1)
        return int(rng.choice(np.asarray([19], dtype=np.int32)))

    _assert_same_rng_result(
        lambda rng: cls._sample_z(rng, 20, 0, None, None),
        cls_reference)

    det = DetPatchDataset(
        ["x"], [8, 8, 8], patch_mode="z_axis",
        z_sampling_mode="legacy")
    det.is_train = True
    det.fg_ratio = 1.0
    boxes = np.asarray([[19, 0, 0, 20, 1, 1]], dtype=np.float32)

    def det_reference(rng):
        rng.random()
        rng.integers(1)
        return int(np.clip(round((boxes[0, 0] + boxes[0, 3]) / 2), 0, 19))

    _assert_same_rng_result(
        lambda rng: det._sample_z(rng, 20, boxes, None),
        det_reference)

    seg.fg_ratio = 0.0
    _assert_same_rng_result(
        lambda rng: (
            setattr(seg, "_sample_rng", lambda sample_idx: rng) or
            seg._sample_z(0, 20, sample_idx=0)),
        lambda rng: int(rng.integers(0, 20)))
    cls.fg_ratio = 0.0
    _assert_same_rng_result(
        lambda rng: cls._sample_z(rng, 20, 0, None, None),
        lambda rng: int(rng.integers(0, 20)))
    det.fg_ratio = 0.0
    _assert_same_rng_result(
        lambda rng: det._sample_z(rng, 20, boxes, None),
        lambda rng: (rng.random(), int(rng.integers(0, 20)))[1])

    # The safe branches retain the current bounded sampling behavior.
    seg.z_sampling_mode = "safe"
    seg.fg_ratio = 1.0
    _assert_same_rng_result(
        lambda rng: (
            setattr(seg, "_sample_rng", lambda sample_idx: rng) or
            seg._sample_z(0, 20, sample_idx=0)),
        lambda rng: (
            rng.random(),
            int(np.clip(rng.choice(np.asarray([19], dtype=np.int32)), 4, 17)))[1])
    cls.z_sampling_mode = "safe"
    cls.fg_ratio = 1.0
    _assert_same_rng_result(
        lambda rng: cls._sample_z(rng, 20, 0, None, None),
        lambda rng: (
            rng.random(),
            rng.integers(1),
            int(np.clip(rng.choice(np.asarray([19], dtype=np.int32)), 4, 17)))[2])
    det.z_sampling_mode = "safe"
    det.fg_ratio = 1.0
    _assert_same_rng_result(
        lambda rng: det._sample_z(rng, 20, boxes, None),
        lambda rng: (
            rng.random(),
            rng.integers(1),
            int(np.clip(round((boxes[0, 0] + boxes[0, 3]) / 2), 4, 16)))[2])

    # Validation branches consume no RNG and retain the old z-grid exactly.
    seg.z_sampling_mode = "legacy"
    cls.z_sampling_mode = "legacy"
    det.z_sampling_mode = "legacy"
    seg.fg_ratio = cls.fg_ratio = det.fg_ratio = 0.0
    seg.is_train = cls.is_train = det.is_train = False
    assert cls._sample_z(np.random.default_rng(1), 20, 0, None, 1) == \
        z_grid_center(1, cls.spv, 20)
    assert det._sample_z(np.random.default_rng(1), 20, boxes, 1) == \
        z_grid_center(1, det.spv, 20)
    seg.is_train = False
    seg.val_grid_coverage = True
    seg._sample_idx = 1
    seg.image_paths = ["x"]
    seg.samples_per_volume = 4
    seg.extract_size = (8, 8, 8)
    assert seg._sample_z(0, 20) == z_grid_center(1, 4, 20)

    cls.z_sampling_mode = "safe"
    det.z_sampling_mode = "safe"
    seg.z_sampling_mode = "safe"
    assert cls._sample_z(np.random.default_rng(1), 20, 0, None, 1) == \
        safe_z_grid_center(1, cls.spv, 20, cls.patch[0])
    assert det._sample_z(np.random.default_rng(1), 20, boxes, 1) == \
        safe_z_grid_center(1, det.spv, 20, det.patch[0])
    assert seg._sample_z(0, 20) == safe_z_grid_center(1, 4, 20, 8)


def test_det_legacy_foreground_clips_box_center():
    from dettask.data.det_dataset import DetPatchDataset

    det = DetPatchDataset(
        ["x"], [8, 8, 8], patch_mode="z_axis",
        z_sampling_mode="legacy")
    det.is_train = True
    det.fg_ratio = 1.0
    upper_box = np.asarray([[0, 0, 0, 40, 1, 1]], dtype=np.float32)
    lower_box = np.asarray([[-40, 0, 0, -1, 1, 1]], dtype=np.float32)
    assert det._sample_z(
        np.random.default_rng(0), 20, upper_box, None) == 19
    assert det._sample_z(
        np.random.default_rng(0), 20, lower_box, None) == 0


@pytest.mark.parametrize("arch", ["adm", "edm2"])
def test_nonlegacy_init_strategy_rejected_for_specialized_arch(arch):
    from taskcore.config.core import Config, ConfigError

    cfg = Config()
    cfg.model.arch = arch
    cfg.model.init_strategy = "kaiming"
    cfg.data.patch_mode = "2_5d"
    cfg.sync()
    with pytest.raises(ConfigError, match="architecture-specific"):
        cfg.validate()


@pytest.mark.parametrize("arch", ["adm", "edm2"])
def test_legacy_specialized_init_is_not_overwritten(monkeypatch, arch):
    from taskcore.config.core import Config
    from taskcore.models import factory

    cfg = Config()
    cfg.model.arch = arch
    cfg.model.init_strategy = "legacy"
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.zero_()
    module_name = ("taskcore.models.adm_unet" if arch == "adm"
                   else "taskcore.models.edm2_unet")
    module = __import__(module_name, fromlist=["builder"])
    monkeypatch.setattr(
        module,
        "build_adm_seg_model" if arch == "adm"
        else "build_edm2_seg_model",
        lambda cfg: model)
    built = factory.build_model(cfg)
    assert built[0].weight.equal(torch.zeros_like(built[0].weight))


def test_unet3p_accepts_nondivisible_encoder_geometry():
    from taskcore.config.core import Config

    cfg = Config()
    cfg.data.patch_size = [15, 32, 32]
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.unet.decoder_type = "unet3p"
    cfg.sync()
    cfg.validate()


def test_classic_unet_keeps_strict_geometry_check():
    from taskcore.config.core import Config, ConfigError

    cfg = Config()
    cfg.data.patch_size = [15, 32, 32]
    cfg.sync()
    with pytest.raises(ConfigError, match="nearest legal"):
        cfg.validate()


def test_adm_geometry_does_not_call_unet_divisor_helper(monkeypatch):
    from taskcore.config.core import Config
    import taskcore.config.section_validators as validators

    cfg = Config()
    cfg.model.arch = "adm"
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [3, 16, 16]
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.sync()
    monkeypatch.setattr(
        validators, "effective_patch_divisors",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    cfg.validate()
