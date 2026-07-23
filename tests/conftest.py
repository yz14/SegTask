"""Pytest bootstrap (R2): bare ``Config()`` 懒加载 seg loss/predict，便于单测构造。

仅作用于 pytest 进程。生产分割路径必须用 :class:`~taskcore.config.seg_bundle.SegBundle`
（或 ``segtask_v1.seg_config.load_config``）；勿依赖本文件的 ``Config.loss`` 猴补。
"""

from __future__ import annotations

import taskcore.config.core as _core
from taskcore.config.seg_task import SegTaskConfig

_BRIDGE = "_r2_seg_bridge"


def _get_bridge(cfg: _core.Config) -> SegTaskConfig:
    bridge = cfg.__dict__.get(_BRIDGE)
    if bridge is None:
        bridge = SegTaskConfig()
        object.__setattr__(cfg, _BRIDGE, bridge)
    return bridge


def _config_loss(self: _core.Config):
    return _get_bridge(self).loss


def _config_loss_setter(self, value) -> None:
    _get_bridge(self).loss = value


def _config_predict(self: _core.Config):
    return _get_bridge(self).predict


def _config_predict_setter(self, value) -> None:
    _get_bridge(self).predict = value


# 生产分割路径用 SegBundle；此处仅恢复单测 ``Config()`` + ``cfg.loss`` 习惯。
_core.Config.loss = property(_config_loss, _config_loss_setter)
_core.Config.predict = property(_config_predict, _config_predict_setter)

_orig_config_validate = _core.Config.validate


def _config_validate(self, *, skip=None) -> None:
    skip = skip or set()
    _orig_config_validate(self, skip=skip)
    bridge = self.__dict__.get(_BRIDGE)
    if bridge is not None and not {"loss", "predict"}.issubset(skip):
        from taskcore.config.seg_task import validate_seg_task

        validate_seg_task(bridge, self)


_core.Config.validate = _config_validate
