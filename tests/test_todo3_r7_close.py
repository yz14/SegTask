"""TODO 3 R7 关闭回归：validation 委托、hoist fail-fast、specs.loss 收窄。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from taskcore.config.core import Config as CoreConfig, ConfigError
from taskcore.config.seg_bundle import make_test_config
from taskcore.config.seg_task import hoist_legacy_seg_sections
from taskcore.data.specs import DatasetCommonCfg
from gentask.config.validation import Config as GenConfig


def test_gen_data_validate_delegates_core_checks():
    """gen 委托 core 后应获得 core 独有约束（如 mix_ratio）。"""
    cfg = GenConfig()
    cfg.data.npz_dir_secondary = "/tmp/secondary"
    cfg.data.mix_ratio = [1, 1]
    cfg.data.batch_size = 3  # 不可被 sum(mix_ratio)=2 整除
    cfg.sync()
    with pytest.raises(ConfigError, match="mix_ratio"):
        cfg.validate()


def test_gen_augment_validate_delegates_translate_warning(caplog):
    cfg = GenConfig()
    cfg.augment.enabled = True
    cfg.augment.random_translate_range = [-0.5, 0.5]
    cfg.data.aug_oversample_ratio = 1.0
    cfg.sync()
    with caplog.at_level("WARNING"):
        cfg.validate()
    assert any("aug_oversample_ratio" in r.message for r in caplog.records)


def test_hoist_legacy_rejects_duplicate_loss():
    raw = {
        "loss": {"name": "dice"},
        "seg": {"loss": {"name": "ce"}},
        "data": {},
    }
    with pytest.raises(ConfigError, match="top-level 'loss'"):
        hoist_legacy_seg_sections(raw)


def test_hoist_legacy_rejects_duplicate_predict():
    raw = {
        "predict": {"threshold": 0.5},
        "seg": {"predict": {"threshold": 0.3}},
    }
    with pytest.raises(ConfigError, match="top-level 'predict'"):
        hoist_legacy_seg_sections(raw)


def test_hoist_legacy_moves_top_level_when_seg_empty():
    raw = {"loss": {"name": "dice"}, "predict": {"threshold": 0.4}}
    hoist_legacy_seg_sections(raw)
    assert "loss" not in raw and "predict" not in raw
    assert raw["seg"]["loss"]["name"] == "dice"
    assert raw["seg"]["predict"]["threshold"] == 0.4


def test_dataset_common_cfg_from_seg_bundle():
    cfg = make_test_config()
    cfg.loss.region_weights = [1.0, 2.0]
    common = DatasetCommonCfg.from_cfg(cfg)
    assert common.region_weights == [1.0, 2.0]


def test_dataset_common_cfg_from_object_without_loss():
    """无 ``.loss`` 的 duck cfg 不 AttributeError，region_weights=None。"""
    core = CoreConfig()
    duck = SimpleNamespace(data=core.data)  # 故意不挂 loss（绕过 conftest 猴补）
    common = DatasetCommonCfg.from_cfg(duck)
    assert common.region_weights is None


def test_gen_2_5d_skips_seg_channel_layout():
    """cond 使 in_channels=2*D，不得被 seg 的 D*n_views 约束误杀。"""
    cfg = GenConfig()
    cfg.data.patch_mode = "2_5d"
    cfg.data.patch_size = [4, 16, 16]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.cond_dirs = ["dummy"]
    cfg.task.type = "generation"
    cfg.task.out_channels = 1
    cfg.sync()
    assert cfg.model.in_channels == 8  # 4 image + 4 cond
    cfg.validate()  # 不应因 in_channels!=4 失败


def test_shared_stage_length_validator():
    from taskcore.config.section_validators import (
        validate_encoder_decoder_stage_lengths,
    )

    cfg = CoreConfig()
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.encoder_blocks_per_stage = [1, 1]  # 长度错
    with pytest.raises(ConfigError, match="encoder_blocks_per_stage"):
        validate_encoder_decoder_stage_lengths(cfg)
