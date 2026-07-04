"""clstask —— 医学影像分类项目（3D / 2.5D 双几何）。

设计与 ssltask 同构：复用 ``segtask_v1`` 的配置载体 / 数据 IO / 训练基建 /
编码器（ResNet/ConvNeXt），叠加分类专属的数据标签派生、backbone（DenseNet/
ViT）、分类头、损失、指标与 MIL 聚合推理。SSL 预训练 ``encoder.*`` 权重可经
``cls.pretrained_ckpt`` 无缝迁移。
"""

from .config import ClsConfig, load_config, save_config

__all__ = ["ClsConfig", "load_config", "save_config"]
