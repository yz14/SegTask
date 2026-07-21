"""D2-1 迁移契约测试：在 ModelConfig 拆嵌套动工前锁定现状。

四类锚点（对应调研的验收矩阵 A/B/D 类）：
* 映射完备性 —— model_migration 归属表与 ModelConfig 字段集恰好互斥全覆盖；
* YAML 基线 —— configs/ 全部配置可加载，adm/edm2 代表配置的关键字段值锁定；
* override 基线 —— 旧扁平点路径（model.backbone / model.adm_num_heads）可写；
* 序列化基线 —— save→load 语义等价；Config 整体 pickle 可 roundtrip
  （checkpoint 内嵌 config 的 torch.load 依赖类路径可还原）。

D2-2+ 引入嵌套结构后，本文件所有断言必须继续成立（兼容层的验收标准）。
"""

from __future__ import annotations

import io
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# ---------------------------------------------------------------------------
# 映射完备性
# ---------------------------------------------------------------------------
def test_migration_map_covers_modelconfig_exactly():
    """归属映射与嵌套 schema 恰好互相覆盖：无遗漏、无多余、路径全可解析。"""
    from taskcore.config.core import ModelConfig
    from taskcore.config.model_migration import (
        ADM_FIELD_MAP, COMMON_FIELDS, EDM2_FIELD_MAP, FLAT_TO_NESTED,
        UNET_FIELD_MAP, flat_to_nested_path,
    )

    # 顶层 = 公共字段 + 三个嵌套段，别无其他。
    top = {f.name for f in dataclass_fields(ModelConfig)}
    assert top == set(COMMON_FIELDS) | {"unet", "adm", "edm2"}, (
        f"unexpected top-level fields: {sorted(top)}")

    # 四组互斥。
    groups = [set(COMMON_FIELDS), set(UNET_FIELD_MAP),
              set(ADM_FIELD_MAP), set(EDM2_FIELD_MAP)]
    total = set().union(*groups)
    assert sum(len(g) for g in groups) == len(total)

    # 每条映射路径在嵌套 schema 上真实存在（逐段走 dataclass 字段）。
    def _resolve(root_cls, path: str):
        cls = root_cls
        parts = path.split(".")
        for p in parts[:-1]:
            f = {x.name: x for x in dataclass_fields(cls)}[p]
            cls = f.default_factory  # 嵌套段约定 default_factory=子类。
        assert parts[-1] in {x.name for x in dataclass_fields(cls)}, path

    mc_nested = {f.name: f.default_factory
                 for f in dataclass_fields(ModelConfig)
                 if f.name in ("unet", "adm", "edm2")}
    for flat, path in FLAT_TO_NESTED.items():
        section, _, rest = path.partition(".")
        _resolve(mc_nested[section], rest)
        assert flat_to_nested_path(flat) == path
    for name in COMMON_FIELDS:
        assert flat_to_nested_path(name) == name
    # 嵌套段叶子总数 == 映射条目数（嵌套侧无映射之外的多余叶子）。
    def _count_leaves(cls) -> int:
        n = 0
        for f in dataclass_fields(cls):
            sub = f.default_factory
            if isinstance(sub, type) and hasattr(sub, "__dataclass_fields__"):
                n += _count_leaves(sub)
            else:
                n += 1
        return n

    assert sum(_count_leaves(c) for c in mc_nested.values()) \
        == len(FLAT_TO_NESTED)
    with pytest.raises(KeyError):
        flat_to_nested_path("no_such_field")


def test_gentask_modelconfig_extension_is_only_sisr():
    """gentask 子类扩展面锁定：仅嵌套 ``sisr`` 一段（叶子经兼容层转发）。"""
    from gentask.config.dataclasses import ModelConfig as GenModelConfig, SISRConfig
    from taskcore.config.core import ModelConfig
    from taskcore.config.model_migration import SISR_FIELD_MAP

    core = {f.name for f in dataclass_fields(ModelConfig)}
    gen = {f.name for f in dataclass_fields(GenModelConfig)}
    assert gen - core == {"sisr"}
    assert {f.name for f in dataclass_fields(SISRConfig)} == {
        "channels", "num_blocks", "num_groups", "res_scale"}
    assert set(SISR_FIELD_MAP) == {
        "sisr_channels", "sisr_num_blocks", "sisr_num_groups", "sisr_res_scale"}
    assert all(v.startswith("sisr.") for v in SISR_FIELD_MAP.values())


# ---------------------------------------------------------------------------
# YAML 加载基线
# ---------------------------------------------------------------------------
def _load_any(path: Path):
    """按任务归属选 loader；返回核心 Config（组合任务丢弃任务段）。"""
    name = path.name
    if name.startswith(("cls", "det")):
        mod = __import__(
            f"{'clstask' if name.startswith('cls') else 'dettask'}.config",
            fromlist=["load_config"])
        return mod.load_config(str(path))[0]
    if name.startswith("ssltask"):
        from ssltask.config import load_config
        return load_config(str(path))[0]
    if name.startswith("gensr"):
        from gentask.config import load_config
        cfg = load_config(str(path))
        return cfg[0] if isinstance(cfg, tuple) else cfg
    from taskcore.config import load_config
    return load_config(str(path))


@pytest.mark.parametrize(
    "yaml_path",
    sorted(CONFIG_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_all_repo_configs_load(yaml_path: Path):
    cfg = _load_any(yaml_path)
    assert cfg.model.arch in ("unet", "adm", "edm2")


def test_adm_edm2_config_field_anchors():
    """arch 专属字段的现值锚点：迁移兼容层必须还原出完全相同的值。"""
    from taskcore.config import load_config

    adm = load_config(str(CONFIG_DIR / "seg2_5d_adm.yaml"))
    assert adm.model.arch == "adm"
    assert adm.model.adm_attention_levels == [3, 4]
    assert adm.model.adm_num_heads == 4
    assert adm.model.adm_num_head_channels == -1

    edm2 = load_config(str(CONFIG_DIR / "seg2_5d_edm2.yaml"))
    assert edm2.model.arch == "edm2"
    assert edm2.model.edm2_channels_per_head > 0
    assert 0.0 < edm2.model.edm2_res_balance < 1.0
    assert 0.0 < edm2.model.edm2_concat_balance < 1.0


# ---------------------------------------------------------------------------
# override 基线（旧扁平路径必须持续可用）
# ---------------------------------------------------------------------------
def test_flat_model_overrides_keep_working():
    from taskcore.config.core import Config
    from taskcore.config.task_io import apply_dotted_overrides

    cfg = Config()
    apply_dotted_overrides(cfg, [
        "model.backbone=convnext",
        "model.adm_num_heads=8",
        "model.edm2_channels_per_head=32",
        "model.encoder_channels=[8, 16, 32]",
        "model.multirf_enabled=true",
    ])
    assert cfg.model.backbone == "convnext"
    assert cfg.model.adm_num_heads == 8
    assert cfg.model.edm2_channels_per_head == 32
    assert cfg.model.encoder_channels == [8, 16, 32]
    assert cfg.model.multirf_enabled is True


# ---------------------------------------------------------------------------
# 序列化基线
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_semantics(tmp_path: Path):
    from dataclasses import asdict

    from taskcore.config.core import load_config, save_config

    src = CONFIG_DIR / "seg2_5d_adm.yaml"
    cfg = load_config(str(src))
    out = tmp_path / "roundtrip.yaml"
    save_config(cfg, out)
    cfg2 = load_config(str(out))
    assert asdict(cfg.model) == asdict(cfg2.model)
    # 派生只读量经 sync 恢复一致。
    assert cfg2.model.spatial_dims == cfg.model.spatial_dims
    assert cfg2.model.in_channels == cfg.model.in_channels


def test_config_pickle_roundtrip_via_torch():
    """checkpoint 内嵌 Config 的 torch.load 依赖类路径可还原（D 类）。"""
    import torch

    from taskcore.config.core import Config

    cfg = Config()
    cfg.model.backbone = "convnext"
    cfg.model.adm_num_heads = 8
    cfg.sync()

    buf = io.BytesIO()
    torch.save({"config": cfg, "epoch": 3}, buf)
    buf.seek(0)
    restored = torch.load(buf, weights_only=False)
    rc = restored["config"]
    assert type(rc).__module__ == "taskcore.config.core"
    assert rc.model.backbone == "convnext"
    assert rc.model.adm_num_heads == 8
    assert rc.model.spatial_dims == cfg.model.spatial_dims


def test_asdict_model_section_is_nested():
    """D2-2 起：asdict(model) 为「公共扁平 + unet/adm/edm2 嵌套」形状。"""
    from dataclasses import asdict

    from taskcore.config.core import Config
    from taskcore.config.model_migration import COMMON_FIELDS

    blob = asdict(Config().model)
    assert set(blob) == set(COMMON_FIELDS) | {"unet", "adm", "edm2"}
    assert isinstance(blob["unet"], dict)
    assert isinstance(blob["unet"]["mednext"], dict)
    assert isinstance(blob["adm"]["linear_attention"], dict)
    assert blob["edm2"]["channels_per_head"] == 64


# ---------------------------------------------------------------------------
# D2-2：嵌套写法与兼容层
# ---------------------------------------------------------------------------
def test_new_nested_yaml_equals_legacy_flat_yaml(tmp_path: Path):
    """同一配置的旧扁平写法与新嵌套写法必须得到逐字段相同的 Config。"""
    from dataclasses import asdict

    from taskcore.config.core import load_config

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "data:\n"
        "  image_dir: img\n"
        "  label_dir: lbl\n"
        "  label_values: [0, 1]\n"
        "  num_classes: 2\n"
        "model:\n"
        "  backbone: convnext\n"
        "  attention_type: none\n"
        "  mednext_expand_ratio: 3\n"
        "  selfattn_window_size: 5\n",
        encoding="utf-8")
    nested = tmp_path / "nested.yaml"
    nested.write_text(
        "data:\n"
        "  image_dir: img\n"
        "  label_dir: lbl\n"
        "  label_values: [0, 1]\n"
        "  num_classes: 2\n"
        "model:\n"
        "  unet:\n"
        "    backbone: convnext\n"
        "    attention_type: none\n"
        "    mednext:\n"
        "      expand_ratio: 3\n"
        "    selfattn:\n"
        "      window_size: 5\n",
        encoding="utf-8")
    a = load_config(str(legacy))
    b = load_config(str(nested))
    assert asdict(a.model) == asdict(b.model)
    assert a.model.unet.backbone == "convnext"
    assert a.model.unet.mednext.expand_ratio == 3


def test_flat_and_nested_conflict_is_rejected(tmp_path: Path):
    from taskcore.config.core import ConfigError, load_config

    bad = tmp_path / "conflict.yaml"
    bad.write_text(
        "model:\n"
        "  backbone: convnext\n"
        "  unet:\n"
        "    backbone: resnet\n",
        encoding="utf-8")
    with pytest.raises(ConfigError, match="both legacy flat key"):
        load_config(str(bad))


def test_flat_python_attribute_compat_read_write():
    """旧 Python 访问 cfg.model.backbone 等与嵌套路径读写等价。"""
    from taskcore.config.core import Config

    cfg = Config()
    # 读：默认值经转发一致。
    assert cfg.model.backbone == cfg.model.unet.backbone == "resnet"
    assert cfg.model.adm_num_heads == cfg.model.adm.num_heads == 4
    # 写扁平 → 嵌套可见。
    cfg.model.backbone = "mednext"
    cfg.model.mednext_kernel_size = 5
    cfg.model.edm2_res_balance = 0.4
    assert cfg.model.unet.backbone == "mednext"
    assert cfg.model.unet.mednext.kernel_size == 5
    assert cfg.model.edm2.res_balance == 0.4
    # 写嵌套 → 扁平可见。
    cfg.model.unet.selfattn.num_heads = 8
    assert cfg.model.selfattn_num_heads == 8


def test_flat_kwargs_constructor_compat():
    """ModelConfig(backbone=..., adm_num_heads=...) 旧构造仍可用。"""
    from taskcore.config.core import ModelConfig

    mc = ModelConfig(
        encoder_channels=[8, 16], decoder_type="unetpp",
        adm_num_heads=8, multirf_enabled=True)
    assert mc.encoder_channels == [8, 16]
    assert mc.unet.decoder_type == "unetpp"
    assert mc.adm.num_heads == 8
    assert mc.unet.multirf.enabled is True


def test_nested_dotted_overrides():
    """新点路径 override（model.unet.backbone 等）可写。"""
    from taskcore.config.core import Config
    from taskcore.config.task_io import apply_dotted_overrides

    cfg = Config()
    apply_dotted_overrides(cfg, [
        "model.unet.backbone=convnext",
        "model.unet.mednext.kernel_size=5",
        "model.adm.num_heads=8",
        "model.edm2.channels_per_head=32",
    ])
    assert cfg.model.unet.backbone == "convnext"
    assert cfg.model.unet.mednext.kernel_size == 5
    assert cfg.model.adm.num_heads == 8
    assert cfg.model.edm2.channels_per_head == 32


def test_legacy_flat_pickle_state_migrates():
    """老 checkpoint pickle（扁平 ModelConfig __dict__）经 __setstate__ 迁移。"""
    from taskcore.config.core import ModelConfig

    legacy_state = {
        "arch": "adm",
        "backbone": "convnext",
        "encoder_channels": [16, 32, 64],
        "adm_num_heads": 8,
        "edm2_clip_act": 128.0,
        "mednext_kernel_size": 5,
        "_spatial_dims": 2,
        "_in_channels": 7,
    }
    mc = ModelConfig.__new__(ModelConfig)
    mc.__setstate__(dict(legacy_state))
    assert mc.arch == "adm"
    assert mc.unet.backbone == "convnext"
    assert mc.adm.num_heads == 8
    assert mc.edm2.clip_act == 128.0
    assert mc.unet.mednext.kernel_size == 5
    assert mc.spatial_dims == 2 and mc.in_channels == 7
    # 未出现在老状态里的字段用默认补齐。
    assert mc.unet.selfattn.enabled is False


def test_gentask_sisr_nested_and_flat_compat():
    """gentask：sisr 嵌套段为真相源；旧扁平 YAML 键/属性/构造均兼容。"""
    from dataclasses import asdict

    from gentask.config.dataclasses import ModelConfig as GenModelConfig
    from gentask.config.io import _dataclass_from_dict

    mc = GenModelConfig(backbone="convnext", sisr_channels=32)
    assert mc.unet.backbone == "convnext"
    assert mc.sisr.channels == 32
    assert mc.sisr_channels == 32

    flat = _dataclass_from_dict(GenModelConfig, {
        "arch": "edsr",
        "sisr_channels": 48,
        "sisr_num_blocks": 8,
    })
    nested = _dataclass_from_dict(GenModelConfig, {
        "arch": "edsr",
        "sisr": {"channels": 48, "num_blocks": 8},
    })
    assert asdict(flat.sisr) == asdict(nested.sisr)
    assert flat.sisr.channels == 48
    assert flat.sisr_channels == 48

    import pytest
    from gentask.config.dataclasses import ConfigError
    with pytest.raises(ConfigError, match="both legacy flat key"):
        _dataclass_from_dict(GenModelConfig, {
            "sisr_channels": 32,
            "sisr": {"channels": 64},
        })


def _iter_flat_model_reads(pkg_root: Path):
    """扫描包内生产代码：yield (file, lineno, attr) —— 对 model 配置接收者
    （``mc`` / ``*.model``）读取已迁移扁平字段的 AST 读点。

    嵌套读法（``mc.unet.backbone`` 等，receiver 是 unet/adm/edm2 子段）不算；
    兼容层自身（model_migration.py）豁免。gentask 包额外守 sisr_*。
    """
    import ast

    from taskcore.config.model_migration import FLAT_TO_NESTED, SISR_FIELD_MAP

    flat = set(FLAT_TO_NESTED)
    if pkg_root.name == "gentask":
        flat |= set(SISR_FIELD_MAP)
    for py in sorted(pkg_root.rglob("*.py")):
        if py.name == "model_migration.py" or "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr in flat):
                continue
            recv = node.value
            is_model_recv = (
                (isinstance(recv, ast.Name) and recv.id == "mc")
                or (isinstance(recv, ast.Attribute) and recv.attr == "model"))
            if is_model_recv:
                yield py, node.lineno, node.attr


@pytest.mark.parametrize(
    "pkg",
    ["taskcore", "segtask_v1", "clstask", "dettask", "ssltask", "gentask"],
)
def test_production_code_reads_only_nested_paths(pkg: str):
    """AST 守门（D2-3/D2-4）：生产代码不得继续读旧扁平 model 字段。

    兼容层（转发 property / YAML 路由）只服务外部旧接口；内部读点一律走
    嵌套路径，防止两套读法长期并存造成语义漂移。
    """
    root = Path(__file__).resolve().parent.parent
    offenders = [
        f"{py.relative_to(root)}:{ln} reads flat '{attr}'"
        for py, ln, attr in _iter_flat_model_reads(root / pkg)
    ]
    assert not offenders, "\n".join(offenders)


def test_launcher_schema_still_presents_flat_model_fields():
    """launcher 表单沿用扁平字段名视图；default/tooltip 来自嵌套子段。"""
    from segtask_v1.launcher.schema import build_field_schema

    schema = build_field_schema()
    model = schema["model"]
    for name in ("arch", "backbone", "mednext_expand_ratio",
                 "adm_num_heads", "edm2_channels_per_head",
                 "selfattn_enabled"):
        assert name in model, name
    assert "unet" not in model and "adm" not in model
    assert model["backbone"]["default"] == "resnet"
    assert model["mednext_expand_ratio"]["default"] == 4
    assert model["backbone"]["tooltip"]  # 注释来自 UNetConfig 源码。
