# clstask — 3D / 2.5D 医学影像分类

clstask 复用 `taskcore` 公共基建（配置、几何拓扑、预处理、优化器、AMP、EMA；经 `segtask_v1` 的 shim 路径引用），提供医学影像分类训练与推理闭环。它支持 3D 与 2.5D 两种几何，能够直接接入 ssltask 预训练的 encoder 权重，也能用自己的分类骨干单独训练。

> 端到端训练/推理流程见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

## 模块树

```text
clstask/
├── README.md  # 本文件
├── __init__.py  # 包入口与对外导出
├── __main__.py  # python -m clstask 入口
├── config.py  # 分类配置、YAML I/O、校验
├── train.py  # 训练 CLI
├── predict.py  # 推理 CLI
├── data/  # 数据发现与样本构建
│   ├── __init__.py  # 数据包入口
│   ├── cls_dataset.py  # 分类样本、标签派生与读取逻辑
│   └── loader.py  # 配对、切分与 dataloader 工厂
├── docs/  # 方案与模型综述
│   ├── classification_models_survey.md  # 分类骨干与设计备忘
│   └── WORKFLOW.md  # 端到端训练/推理流程
├── losses/  # 分类损失
│   ├── __init__.py  # 损失包入口
│   └── cls_loss.py  # CE / BCE / focal 等分类损失
├── metrics.py  # AUC / F1 / ACC 等指标
├── models/  # 分类模型与骨干
│   ├── __init__.py  # 模型包入口
│   ├── classifier.py  # 分类模型封装
│   ├── densenet.py  # DenseNet 骨干
│   ├── factory.py  # backbone 装配与迁移入口
│   └── vit.py  # ViT 分类骨干
├── predictor/  # 推理器
│   ├── __init__.py  # 推理包入口
│   └── cls_predictor.py  # 整例 / 整卷分类推理与聚合（继承 taskcore.engine.BasePredictor）
└── trainer/  # 训练循环
    ├── __init__.py  # 训练包入口
    ├── cls_trainer.py  # 训练、验证与选模（继承 taskcore.engine.BaseTrainer）
    └── mixup.py  # mixup / cutmix 增强
```

## 关键概念

- **双几何**：`patch_mode` 沿用分割仓库的语义。3D 使用 `(B, 1, D, H, W)`，2.5D 把 slab 深度折进通道，使用 `(B, D, H, W)` 形式。
- **四个 backbone 模板**：`resnet`、`convnext`、`densenet`、`vit` 四条路线覆盖常见分类实验，其中前两者可直接复用 SSL 预训练 encoder。
- **标签契约**：支持 `mask` 派生弱标签与 `table` 显式标签表两种来源；`volume` 和 `slice` 粒度的输出形状不同，配置需要和数据源对齐。
- **损失与增强**：支持 BCE、focal、CE、label smoothing、class weights，以及仅在卷级分类上启用的 mixup / cutmix。
- **SSL 迁移**：`pretrained_ckpt` 只加载 encoder 相关权重，几何或 backbone 不一致时直接报错，不做静默降级。
- **推理聚合**：推理时先做 patch 级预测（抽取几何与训练一致，可选 autocast 与 `cls.tta_flips` 翻转 TTA），再按几何与 `agg_mode` 聚合成卷级结果；slice 粒度还会保留逐层输出。
- **训练工程**：每 epoch 原子写 latest_model.pth，`train.resume` 完整续训，history.json 逐 epoch 落盘，`train.early_stopping` 早停；warmup 段保持差分学习率倍率。

## 用法

```bash
# 训练
python -m clstask.train --config configs/cls3d_cubic.yaml
python -m clstask.train --config configs/cls2_5d.yaml

# 推理
python -m clstask.predict --config configs/cls3d_cubic.yaml \
  --ckpt outputs/cls3d_cubic/best_model.pth \
  --npz-dir /path/to/npz --out-dir predictions/cls

# 冒烟测试
python tests/test_clstask_smoke.py
```
