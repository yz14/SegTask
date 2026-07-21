"""P2a B1：taskcore.config.task_io 泛型 I/O / override 回归。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from taskcore.config.core import Config
from taskcore.config.task_io import (
    apply_dotted_overrides,
    coerce_override_value,
    load_core_and_task_config,
    save_core_and_task_config,
)
from taskcore.config.registry import (
    TaskSectionSpec,
    clear_task_registry,
    load_task_config,
    register_task_section,
    registered_task_sections,
)


def test_coerce_optional_none_uses_yaml():
    assert coerce_override_value(None, "[1.0, 1.0, 1.0]") == [1.0, 1.0, 1.0]
    assert coerce_override_value(None, "true") is True
    assert coerce_override_value(None, "3") == 3


def test_coerce_typed_fields():
    assert coerce_override_value(True, "false") is False
    assert coerce_override_value(1, "7") == 7
    assert coerce_override_value(1.5, "2.5") == 2.5
    assert coerce_override_value([1, 2], "[3, 4]") == [3, 4]
    assert coerce_override_value("x", "y") == "y"


def test_apply_dotted_overrides_routes_task_section():
    cfg = Config()
    cfg.train.epochs = 1

    @dataclass
    class _Task:
        lr_mult: float = 1.0
        name: str = "a"

    task = _Task()
    apply_dotted_overrides(
        cfg,
        ["train.epochs=5", "cls.lr_mult=0.25", "cls.name=demo"],
        sections={"cls": task},
    )
    assert cfg.train.epochs == 5
    assert task.lr_mult == 0.25
    assert task.name == "demo"


def test_load_save_roundtrip_core_and_task(tmp_path: Path):
    from clstask.config import ClsConfig, validate_cls

    src = tmp_path / "in.yaml"
    # 最小可 sync/validate 的几何；cls 用 mask 弱标签默认。
    blob = {
        "data": {
            "patch_mode": "cubic",
            "patch_size": [16, 32, 32],
            "multi_res_scales": [1.0],
            "label_values": [0, 1],
            "num_classes": 2,
        },
        "model": {
            "backbone": "resnet",
            "encoder_channels": [8, 16, 32],
            "blocks_per_level": 1,
            "stem_mode": "conv3",
        },
        "cls": {
            "backbone": "encoder",
            "label_source": "mask",
            "label_granularity": "volume",
        },
    }
    src.write_text(yaml.dump(blob), encoding="utf-8")

    cfg, cls = load_core_and_task_config(
        src, section="cls", task_cls=ClsConfig, validate_task=validate_cls)
    assert isinstance(cfg, Config)
    assert cls.label_source == "mask"

    out = tmp_path / "out.yaml"
    save_core_and_task_config(cfg, cls, out, section="cls")
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "cls" in raw
    assert raw["cls"]["label_source"] == "mask"


def test_clstask_apply_overrides_optional_spacing(tmp_path: Path):
    from clstask.config import apply_overrides, load_config

    src = tmp_path / "cls.yaml"
    src.write_text(
        yaml.dump({
            "data": {
                "patch_mode": "cubic",
                "patch_size": [16, 32, 32],
                "multi_res_scales": [1.0],
                "label_values": [0, 1],
                "num_classes": 2,
            },
            "model": {
                "backbone": "resnet",
                "encoder_channels": [8, 16, 32],
                "blocks_per_level": 1,
                "stem_mode": "conv3",
            },
            "cls": {
                "backbone": "encoder",
                "label_source": "mask",
                "label_granularity": "volume",
            },
        }),
        encoding="utf-8",
    )
    cfg, cls = load_config(src)
    assert cfg.data.target_spacing is None
    apply_overrides(cfg, cls, ["data.target_spacing=[1.0,1.0,1.0]"])
    assert cfg.data.target_spacing == [1.0, 1.0, 1.0]


def test_task_registry_lists_composite_tasks():
    import clstask.config  # noqa: F401
    import dettask.config  # noqa: F401
    import ssltask.config  # noqa: F401

    assert registered_task_sections() == ("cls", "det", "ssl")


def test_load_task_config_via_registry(tmp_path: Path):
    import clstask.config  # noqa: F401

    src = tmp_path / "cls.yaml"
    src.write_text(
        yaml.dump({
            "data": {
                "patch_mode": "cubic",
                "patch_size": [16, 32, 32],
                "multi_res_scales": [1.0],
                "label_values": [0, 1],
                "num_classes": 2,
            },
            "model": {
                "backbone": "resnet",
                "encoder_channels": [8, 16, 32],
                "blocks_per_level": 1,
                "stem_mode": "conv3",
            },
            "cls": {
                "backbone": "encoder",
                "label_source": "mask",
                "label_granularity": "volume",
            },
        }),
        encoding="utf-8",
    )
    cfg, cls = load_task_config(src, "cls")
    assert isinstance(cfg, Config)
    assert cls.label_granularity == "volume"


def test_register_task_section_rejects_duplicate():
    from taskcore.config import registry as reg_mod

    saved = dict(reg_mod._REGISTRY)
    clear_task_registry()
    try:

        @dataclass
        class _TaskA:
            x: int = 1

        def _validate_a(_t, _c):
            pass

        register_task_section(TaskSectionSpec(
            name="demo", task_cls=_TaskA, validate_task=_validate_a))
        with pytest.raises(ValueError, match="already registered"):
            register_task_section(TaskSectionSpec(
                name="demo", task_cls=_TaskA, validate_task=_validate_a))
    finally:
        reg_mod._REGISTRY.clear()
        reg_mod._REGISTRY.update(saved)


def test_composite_task_skips_seg_loss_predict_validate():
    import clstask.config  # noqa: F401
    from clstask.config import validate_cls

    from taskcore.config.core import Config, ConfigError

    cfg = Config()
    cfg.data.patch_mode = "cubic"
    cfg.data.patch_size = [16, 32, 32]
    cfg.data.multi_res_scales = [1.0]
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.model.backbone = "resnet"
    cfg.model.encoder_channels = [8, 16, 32]
    cfg.model.blocks_per_level = 1
    cfg.model.stem_mode = "conv3"
    cfg.loss.name = "not_a_seg_loss"
    cfg.sync()

    with pytest.raises(ConfigError, match="Invalid loss"):
        cfg.validate()

    cfg.validate(skip={"loss", "predict"})
    validate_cls(clstask.config.ClsConfig(), cfg)
