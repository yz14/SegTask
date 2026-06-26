"""Segtask 训练/推理可视化启动器子包。

设计目标：用一份本地服务 + 两个单页 HTML（2.5D / 3D），把 ``segtask_v1`` 的 YAML
配置渲染成"只含该模式下可调且生效参数"的表单，带 ``?`` 悬浮说明、实时校验、YAML
预览，并一键发起 ``python -m segtask_v1.train/predict`` 且回传实时日志。

非侵入：不修改既有 ``train.py`` / ``predict.py`` / ``config.py``，全部信息（字段、
默认值、注释 tooltip、跨字段校验）从 ``config.py`` 单一真相源自动抽取或复用。

入口：``python -m segtask_v1.launcher``。
"""

from __future__ import annotations
