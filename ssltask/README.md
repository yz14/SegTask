# ssltask — 与分割主线共骨干的自监督预训练

ssltask 是一套与 `segtask_v1` 共骨干的自监督预训练工程：它负责在无标注或少标注数据上做表示学习，并把学到的 encoder 权重迁移回分割、分类与检测任务。方法库覆盖重建式、对比式、自蒸馏式与掩码建模式多种 SSL 路线，评测侧则提供在线 probe 与离线 few-shot 评测。

## 模块树

```text
ssltask/
├── README.md  # 本文件
├── __init__.py  # 包入口与注册表
├── __main__.py  # python -m ssltask 入口
├── config.py  # 复用分割配置并叠加 SSLConfig
├── pretrain.py  # 自监督预训练 CLI
├── evaluate.py  # 离线 few-shot 评测 CLI
├── data/  # 数据子系统
│   ├── __init__.py  # 数据包入口
│   ├── corruptions.py  # Genesis 式破坏变换
│   ├── masking.py  # 掩码建模共享工具
│   ├── multicrop.py  # 多裁剪视图生成器
│   ├── ssl_dataset.py  # image-only / labeled npz 数据集
│   └── vesselness.py  # Frangi vesselness 目标
├── eval/  # 评测子系统
│   ├── __init__.py  # 评测包入口
│   ├── cls_probe.py  # 在线分类探针
│   ├── metrics.py  # AUC / F1 / HD95 指标
│   ├── pipeline.py  # 离线评测 pipeline
│   └── probe.py  # 在线分割探针
├── methods/  # SSL 方法注册与实现
│   ├── __init__.py  # 方法注册表 / build_method
│   ├── base.py  # SSLMethod 抽象接口
│   ├── byol.py  # BYOL
│   ├── dino.py  # DINO
│   ├── dino_gram.py  # DINO + Gram anchoring
│   ├── genesis.py  # Models Genesis 式重建
│   ├── ibot.py  # iBOT / DINOv2 混合
│   ├── jepa.py  # JEPA 隐空间预测
│   ├── moco.py  # MoCo
│   ├── prior.py  # Frangi vesselness 回归
│   ├── simmim.py  # SimMIM
│   ├── spark.py  # SparK
│   └── sparkdino.py  # SparK + DINO 联合目标
├── models/  # SSL 专有模型模块
│   ├── __init__.py  # 模型包入口
│   ├── dino_modules.py  # DINOHead / DINONet
│   ├── ibot_modules.py  # iBOT 密集头工具
│   ├── jepa_modules.py  # JEPA encoder / predictor builder
│   ├── spark_modules.py  # SparK 稀疏编码与轻量解码器
│   └── ssl_models.py  # 重建 / MIM 模型与 head builder
└── trainer/  # SSL 训练循环
    ├── __init__.py  # 训练包入口
    └── ssl_trainer.py  # 方法无关的 SSL 训练循环
```

## 关键概念

- **共骨干迁移**：ssltask 的主要价值是产出可迁移 encoder；分割、分类、检测都可以直接复用这份权重。
- **方法注册表**：不同 SSL 方法通过统一的注册入口装配，训练循环尽量保持方法无关。
- **数据视图**：image-only npz、多裁剪、masking、corruption、vesselness 都是围绕视图构造展开的。
- **评测探针**：在线 probe 与离线 few-shot 评测并存，既看表示可分性，也看下游迁移效果。
- **方法家族**：重建、对比、自蒸馏、掩码建模和先验回归几类方法都在同一仓库内对齐实现。
- **配置入口**：所有方法都从 `configs/ssltask_*.yaml` 进入，和主线分割配置保持一致的数据与几何约束。

## 用法

```bash
# 预训练
python -m ssltask.pretrain --config configs/ssltask_genesis.yaml
python -m ssltask.pretrain --config configs/ssltask_dino.yaml

# 评测
python -m ssltask.evaluate --config configs/ssltask_prior.yaml

# 其他示例配置
# configs/ssltask_simmim.yaml / configs/ssltask_spark.yaml / configs/ssltask_ibot.yaml
```
