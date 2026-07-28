"""R2 审查热修回归：SegBundle pickle、旧 override、validate skip、make_data skip。"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest
import torch
import yaml

from taskcore.config.core import Config, ConfigError, load_config as core_load
from taskcore.config.seg_bundle import SegBundle, make_test_config
from taskcore.config.seg_task import SegTaskConfig
from taskcore.data.make_data import _npz_meta_allows_skip
from segtask_v1.seg_config import apply_overrides, load_config as seg_load


def test_seg_bundle_pickle_roundtrip():
    cfg = make_test_config()
    cfg.loss.name = "dice"
    cfg.train.epochs = 3
    blob = pickle.dumps(cfg)
    out = pickle.loads(blob)
    assert isinstance(out, SegBundle)
    assert out.loss.name == "dice"
    assert out.train.epochs == 3
    assert out.data.patch_mode == cfg.data.patch_mode


def test_seg_bundle_deepcopy_and_torch_load(tmp_path: Path):
    cfg = make_test_config()
    cfg.predict.threshold = 0.4
    c2 = copy.deepcopy(cfg)
    assert c2.predict.threshold == 0.4
    assert c2 is not cfg and c2.core is not cfg.core

    path = tmp_path / "ckpt.pth"
    torch.save({"config": cfg, "epoch": 1}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(loaded["config"], SegBundle)
    assert loaded["config"].predict.threshold == 0.4


def test_legacy_loss_predict_override_routed_to_seg():
    cfg = make_test_config()
    apply_overrides(cfg, [
        "loss.name=lovasz",
        "predict.threshold=0.3",
        "train.epochs=7",
    ])
    assert cfg.loss.name == "lovasz"
    assert cfg.predict.threshold == pytest.approx(0.3)
    assert cfg.train.epochs == 7
    # 写入的是 seg 段，不是 core dataclass 字段
    from dataclasses import fields as dc_fields
    assert "loss" not in {f.name for f in dc_fields(cfg.core)}


def test_seg_prefixed_override_still_works():
    cfg = make_test_config()
    apply_overrides(cfg, ["seg.loss.name=gdl"])
    assert cfg.loss.name == "gdl"


def test_validate_skip_loss_only_skips_loss():
    cfg = make_test_config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.patch_size = [8, 16, 16]
    cfg.data.patch_mode = "z_axis"
    cfg.data.multi_res_scales = [1.0]
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.loss.name = "not_a_seg_loss"
    cfg.sync()
    # 只 skip loss：不应因 invalid loss 失败
    cfg.validate(skip={"loss"})
    with pytest.raises(ConfigError, match="Invalid loss"):
        cfg.validate()


def test_core_load_config_warns_when_discarding_seg(tmp_path: Path, caplog):
    path = tmp_path / "with_seg.yaml"
    path.write_text(yaml.dump({
        "data": {
            "label_values": [0, 1], "num_classes": 2,
            "patch_size": [8, 16, 16], "patch_mode": "z_axis",
            "multi_res_scales": [1.0],
        },
        "model": {
            "backbone": "resnet", "encoder_channels": [8, 16, 32],
            "blocks_per_level": 1, "stem_mode": "conv3",
        },
        "train": {"save_best_criterion": "dice"},
        "seg": {"loss": {"name": "lovasz"}},
    }), encoding="utf-8")
    with caplog.at_level("WARNING"):
        cfg = core_load(path)
    assert isinstance(cfg, Config)
    assert any("discards top-level 'seg'" in r.message for r in caplog.records)
    # 分割入口仍能读到 lovasz
    bundle = seg_load(path)
    assert bundle.loss.name == "lovasz"


def test_npz_meta_skip_checks_label_values_and_fg_subsample():
    meta = {
        "label_counts": {0: 1},
        "image_shape": [1, 2, 3],
        "fg_per_class": True,
        "spacing_normalized": False,
        "label_values": [0, 1, 2],
        "fg_subsample": 1000,
        "intensity_stats": {"mean": 0.0},  # make_data>=1.9 skip 必备键
    }
    ok, _ = _npz_meta_allows_skip(
        meta, spacing_normalization=False, target_spacing=None,
        label_values=[0, 1, 2], fg_subsample=1000)
    assert ok
    ok, reason = _npz_meta_allows_skip(
        meta, spacing_normalization=False, target_spacing=None,
        label_values=[0, 1], fg_subsample=1000)
    assert not ok and "label_values" in reason
    ok, reason = _npz_meta_allows_skip(
        meta, spacing_normalization=False, target_spacing=None,
        label_values=[0, 1, 2], fg_subsample=500)
    assert not ok and "fg_subsample" in reason


def test_gen_subclass_alias_mro_lookup():
    """gen DataConfig 子类应命中 core 父类 _FIELD_ALIASES。"""
    from taskcore.config.core import dataclass_from_dict, ConfigError
    from gentask.config.dataclasses import DataConfig as GenDataConfig

    with pytest.raises(ConfigError, match="keep_native_view_depth"):
        dataclass_from_dict(
            GenDataConfig,
            {"aux_keep_native_d": True, "label_values": [0, 1]})
