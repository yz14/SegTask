## 总结性描述

ssltask 是与 `segtask_v1` 共骨干的无标注自监督预训练工程：`pretrain.py` 负责方法无关的预训练，`evaluate.py` 负责离线 few-shot 评测，`config.py` 在复用 `segtask_v1.config.Config` 的基础上叠加 `SSLConfig`。方法库覆盖 Genesis / Prior / SimMIM / SparK / DINO / DINO+Gram / iBOT / JEPA / BYOL / MoCo / SparK+DINO；数据侧提供 image-only npz、multi-crop、masking、Frangi vesselness、在线线性 / 分割 probe；评测侧提供在线 probe 与 P6 离线对比 harness。

## 详细说明

### 总览与模块树
```text
ssltask/
├── ssltask/__init__.py  # 包入口 / 注册表
├── ssltask/__main__.py  # python -m ssltask 入口
├── ssltask/config.py  # SegConfig 复用 + SSLConfig
├── ssltask/evaluate.py  # 离线 few-shot 评测 CLI
├── ssltask/pretrain.py  # 自监督预训练 CLI
├── ssltask/data/__init__.py  # 包入口 / 注册表
├── ssltask/data/corruptions.py  # Genesis 式破坏变换
├── ssltask/data/masking.py  # mask / 掩码建模共享工具
├── ssltask/data/multicrop.py  # 多裁剪视图生成器
├── ssltask/data/ssl_dataset.py  # image-only / labeled npz 数据集
├── ssltask/data/vesselness.py  # Frangi vesselness 目标
├── ssltask/eval/__init__.py  # 包入口 / 注册表
├── ssltask/eval/cls_probe.py  # 在线分类探针
├── ssltask/eval/metrics.py  # AUC / F1 / HD95 指标
├── ssltask/eval/pipeline.py  # 离线评测与对比 pipeline
├── ssltask/eval/probe.py  # 在线分割探针
├── ssltask/methods/__init__.py  # 包入口 / 注册表
├── ssltask/methods/base.py  # SSLMethod 抽象接口
├── ssltask/methods/byol.py  # BYOL-3D
├── ssltask/methods/dino.py  # DINO-3D
├── ssltask/methods/dino_gram.py  # DINO + Gram anchoring
├── ssltask/methods/genesis.py  # Models Genesis 式重建
├── ssltask/methods/ibot.py  # iBOT / DINOv2 混合
├── ssltask/methods/jepa.py  # JEPA 隐空间预测
├── ssltask/methods/moco.py  # MoCo-3D
├── ssltask/methods/prior.py  # Frangi vesselness 回归
├── ssltask/methods/simmim.py  # SimMIM-3D
├── ssltask/methods/spark.py  # SparK-3D
├── ssltask/methods/sparkdino.py  # SparK + DINO 联合目标
├── ssltask/models/__init__.py  # 包入口 / 注册表
├── ssltask/models/dino_modules.py  # DINOHead / DINONet
├── ssltask/models/ibot_modules.py  # iBOT 密集头工具
├── ssltask/models/jepa_modules.py  # JEPA encoder / predictor builder
├── ssltask/models/spark_modules.py  # SparK 稀疏编码与轻量解码器
├── ssltask/models/ssl_models.py  # 重建 / MIM 模型与 head builder
├── ssltask/trainer/__init__.py  # 包入口 / 注册表
└── ssltask/trainer/ssl_trainer.py  # 方法无关的 SSL 训练循环
```
### 方法家族速览

- `genesis.py`：Models Genesis 式多变换破坏 → 重建。
- `prior.py`：回归 Frangi vesselness 经典几何先验。
- `simmim.py`：SimMIM 风格 mask token 稠密掩码建模。
- `spark.py`：SparK 风格稀疏掩码建模 + 轻量层次解码器。
- `dino.py`：DINO 多裁剪 + EMA 教师自蒸馏。
- `dino_gram.py`：DINO + Gram anchoring。
- `ibot.py`：DINO 全局蒸馏 + iBOT 掩码密集特征预测。
- `jepa.py`：JEPA 隐空间掩码预测。
- `byol.py` / `moco.py`：实例判别式自蒸馏 / 对比学习基线。
- `sparkdino.py`：SparK 像素重建 + DINO 全局蒸馏的组合目标。

### 配置 / 数据 / 评测

- 入口：`python -m ssltask.pretrain --config configs/ssltask_*.yaml`、`python -m ssltask.evaluate --config ...`
- 配置：`ssltask/config.py` 读取 `ssl:` 段并复用 `segtask_v1` 的 `data/model/train` 三段；所有方法共享同一骨干几何。
- 数据：`ssltask/data/ssl_dataset.py` 负责 image-only / labeled npz；`multicrop.py` 提供 DINO/BYOL/MoCo 的视图生成；`masking.py` 统一掩码工具；`corruptions.py` 与 `vesselness.py` 支持 Genesis / Prior。
- 评测：`ssltask/eval/probe.py`、`cls_probe.py` 与 `pipeline.py` 提供在线线性探针与离线 few-shot 对比。
- 训练：`ssltask/trainer/ssl_trainer.py` 只负责优化器、调度器、AMP、EMA、checkpoint，具体损失由 method 实现。
- 示例配置：`configs/ssltask_genesis.yaml`、`configs/ssltask_prior.yaml`、`configs/ssltask_simmim.yaml`、`configs/ssltask_spark.yaml`、`configs/ssltask_dino.yaml`、`configs/ssltask_dino_gram.yaml`、`configs/ssltask_ibot.yaml`、`configs/ssltask_jepa.yaml`、`configs/ssltask_byol.yaml`、`configs/ssltask_moco.yaml`、`configs/ssltask_sparkdino.yaml`。
