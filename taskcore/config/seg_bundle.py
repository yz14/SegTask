"""分割运行期统一配置视图：core + seg 任务段合并委托。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Set

from .core import Config
from .seg_task import SegTaskConfig


class SegBundle:
    """训练/推理期 ``cfg`` 对象：core 字段 + ``loss``/``predict`` 来自 seg 段。

    对外保持 ``cfg.loss`` / ``cfg.predict`` / ``cfg.data`` 等同址访问习惯；
    ``sync()`` / ``validate()`` 编排 core 与 seg 两侧逻辑。
    """

    __slots__ = ("_core", "_seg")

    def __init__(self, core: Config, seg: Optional[SegTaskConfig] = None) -> None:
        self._core = core
        self._seg = seg if seg is not None else SegTaskConfig()

    @property
    def core(self) -> Config:
        return self._core

    @property
    def seg(self) -> SegTaskConfig:
        return self._seg

    @property
    def loss(self) -> Any:
        return self._seg.loss

    @loss.setter
    def loss(self, value: Any) -> None:
        self._seg.loss = value

    @property
    def predict(self) -> Any:
        return self._seg.predict

    @predict.setter
    def predict(self, value: Any) -> None:
        self._seg.predict = value

    def sync(self) -> None:
        self._core.sync()

    def validate(self, *, skip: Optional[Set[str]] = None) -> None:
        skip = skip or set()
        self._core.validate(skip=skip | {"loss", "predict"})
        # 按段拆分：仅当 loss/predict 均未 skip 时才跑整段 seg 校验；
        # 若只 skip 其一，仍校验未 skip 的段（_validate_loss / _validate_predict）。
        run_loss = "loss" not in skip
        run_predict = "predict" not in skip
        if run_loss or run_predict:
            if run_loss and run_predict:
                self._seg.validate(self._core)
            elif run_loss:
                self._seg._validate_loss(self._core)
                self._seg._validate_cross(self._core)
            else:
                self._seg._validate_predict(self._core)

    def __getattr__(self, name: str) -> Any:
        # 与 SourceTaggedDataset 同构：unpickle / deepcopy 时 slots 未填，
        # 须对 _core/_seg 与 dunder 抛 AttributeError，避免自递归 RecursionError。
        if name in ("_core", "_seg") or (
                name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        try:
            core = object.__getattribute__(self, "_core")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(core, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_core", "_seg"):
            object.__setattr__(self, name, value)
        elif name in ("loss", "predict"):
            setattr(self._seg, name, value)
        else:
            setattr(self._core, name, value)

    def asdict(self) -> dict:
        """合并 dict（seg 段写入 ``seg`` 键）。"""
        blob = asdict(self._core)
        blob["seg"] = asdict(self._seg)
        return blob


def merge_seg_bundle(core: Config, seg: Optional[SegTaskConfig] = None) -> SegBundle:
    """便捷工厂。"""
    return SegBundle(core, seg)


def make_test_config() -> SegBundle:
    """测试/脚本便捷工厂：core 默认 + seg 默认 loss/predict。"""
    bundle = SegBundle(Config())
    bundle.sync()
    return bundle


__all__ = ["SegBundle", "merge_seg_bundle", "make_test_config"]
