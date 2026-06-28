"""ssltask —— 自监督预训练（SSL）专题任务包。

定位：**上游骨干预训练**。固定 ``segtask_v1`` 的 CNN 编/解码器为唯一骨干真相源
（直接 import ``segtask_v1.models.factory.build_model``，不另起一份），用多种自监督
目标（掩码建模 / 还原 / 自蒸馏 / 隐空间预测 / 对比 …，见 ``SSL.md``）预训练，产出与
``build_model`` **逐参数同名同形**的 ``encoder.*``(+``decoder.*``) 权重；下游分割/分类
经各自现成的 ``train.pretrain`` 非严格加载干净衔接，零侵入。

入口：``python -m ssltask.pretrain --config configs/ssltask_genesis.yaml``。
"""

__all__ = []
