"""dettask —— 医学影像目标检测项目（3D / 2.5D 双几何）。

与 clstask 同构：复用 ``taskcore`` 的配置载体 / 数据 IO / 训练基建 /
Encoder+Decoder 金字塔（Retina U-Net 思路），叠加检测专属的 bbox 数据层、
anchor/分配/编解码、四模板检测头（RetinaNet / FCOS / Faster R-CNN /
Deformable-DETR）、mAP/FROC 评估与 2.5D 跨层拼接推理。SSL 预训练
``encoder.*``（重建式含 ``decoder.*``）权重可经 ``det.pretrained_ckpt``
无缝迁移。
"""

from .config import DetConfig, load_config, save_config

__all__ = ["DetConfig", "load_config", "save_config"]
