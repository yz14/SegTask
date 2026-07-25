"""批 1 回归测试：几何、采样、配置加载、迁移与 override。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import yaml


def test_patch_geometry_rejects_non_divisible_and_suggests_value():
    from taskcore.config.core import Config, ConfigError

    cfg = Config()
    cfg.data.patch_size = [15, 32, 32]
    cfg.sync()
    with pytest.raises(ConfigError, match="nearest legal >= value is 16"):
        cfg.validate()


def test_decoder_mismatch_paths_are_hard_errors():
    for path in (
        "taskcore/models/unetpp.py",
        "taskcore/models/adm_unet.py",
        "taskcore/models/edm2_unet.py",
    ):
        text = Path(path).read_text(encoding="utf-8")
        assert "F.interpolate" not in text.split("size mismatch", 1)[-1][:500]
    from taskcore.models.unet3p import UNet3PDecoder

    assert "固有" in UNet3PDecoder._resize_to.__doc__


def test_legacy_seg_yaml_registry_matches_direct_loader():
    from segtask_v1.seg_config import load_config as direct_load
    from taskcore.config.registry import load_task_config
    import segtask_v1.seg_config  # registration side effect

    raw = {
        "data": {"patch_size": [16, 32, 32], "label_values": [0, 1],
                 "num_classes": 2},
        "loss": {"name": "dice"},
        "predict": {"threshold": 0.5},
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "legacy.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        direct = direct_load(path)
        core, task = load_task_config(path, "seg")
    assert direct.seg.loss.name == task.loss.name
    assert direct.seg.predict.threshold == task.predict.threshold
    assert core.data.patch_size == direct.data.patch_size


def test_pretrain_zero_match_errors():
    from taskcore.models.pretrain import load_pretrained_modules

    module = torch.nn.Linear(2, 2)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "empty.pt"
        torch.save({"state_dict": {"wrong.weight": torch.ones(2, 2)}}, path)
        with pytest.raises(RuntimeError, match="0 tensors"):
            load_pretrained_modules(
                {"encoder": module}, str(path), zero_match_error="0 tensors")


def test_pretrain_factory_helpers_share_common_loader():
    from clstask.models.factory import load_pretrained_encoder
    from dettask.models.factory import load_pretrained_backbone

    assert load_pretrained_encoder.__module__ == "clstask.models.factory"
    assert load_pretrained_backbone.__module__ == "dettask.models.factory"
    assert "load_pretrained_modules" in Path(
        "clstask/models/factory.py").read_text()
    assert "load_pretrained_modules" in Path(
        "dettask/models/factory.py").read_text()


def test_attn_gate_target_is_not_silent_for_classic_unet():
    from taskcore.config.core import ConfigError
    from taskcore.models.factory import build_model

    cfg = __import__("taskcore.config.core", fromlist=["Config"]).Config()
    cfg.sync()
    with pytest.raises(ConfigError, match="only UNet\\+\\+"):
        build_model(cfg, attn_gate_target="upsample")


def test_unetpp_decoder_block_length_uses_nested_nodes():
    from taskcore.config.core import Config, ConfigError

    cfg = Config()
    cfg.model.unet.decoder_type = "unetpp"
    cfg.model.decoder_blocks_per_stage = [2, 2, 2]
    cfg.sync()
    with pytest.raises(ConfigError, match="10 entries"):
        cfg.validate()


def test_annotation_driven_optional_list_override():
    from taskcore.config.task_io import set_dotted_attr
    from taskcore.config.seg_task import SegTaskConfig

    task = SegTaskConfig()
    set_dotted_attr(task, "predict.threshold", "[0.3,0.6]")
    assert task.predict.threshold == [0.3, 0.6]


def test_safe_z_sampling_uses_centered_thin_volume_and_warns_once(caplog):
    from taskcore.data.sampling import safe_z_center_range, safe_z_grid_center

    with caplog.at_level("WARNING"):
        assert safe_z_center_range(4, 16) == (2, 3)
        assert safe_z_grid_center(0, 4, 4, 16) == 2
    assert "centered edge-padded" in caplog.text
