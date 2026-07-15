# dettask — 3D / 2.5D 医学影像目标检测

dettask 复用 `segtask_v1` 的配置、几何拓扑、预处理、优化器、AMP 和 EMA 基建，提供医学影像目标检测训练、评估与推理闭环。它把 2D / 3D 框算子统一到同一套检测几何上，支持 RetinaNet、FCOS、Faster R-CNN 和 DETR 四种检测头，并可迁移 ssltask 的 encoder 权重。

> 端到端训练/推理流程见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

## 模块树

```text
dettask/
├── README.md  # 本文件
├── __init__.py  # 包入口与对外导出
├── __main__.py  # python -m dettask 入口
├── config.py  # 检测配置、YAML I/O、校验
├── train.py  # 训练 CLI
├── predict.py  # 推理 CLI
├── ops.py  # IoU / GIoU / NMS / ROIAlign 等检测算子
├── targets.py  # anchor、分配、编解码与增强联动
├── metrics.py  # mAP / FROC 等指标
├── data/  # 检测数据集与 loader
│   ├── __init__.py  # 数据包入口
│   ├── det_dataset.py  # 检测样本、框读取与抽样
│   └── loader.py  # 配对发现、切分与 dataloader 工厂
├── docs/  # 检测头与方案综述
│   ├── detection_models_survey.md  # 检测骨干与设计备忘
│   └── WORKFLOW.md  # 端到端训练/推理流程
├── losses/  # 检测损失
│   ├── __init__.py  # 损失包入口
│   └── det_loss.py  # 检测损失实现
├── models/  # 检测模型与头部
│   ├── __init__.py  # 模型包入口
│   ├── detector.py  # 检测主模型封装
│   ├── factory.py  # backbone + head 装配入口
│   ├── fpn.py  # FPNAdapter
│   └── heads/  # 具体检测头
│       ├── __init__.py  # 头部注册
│       ├── detr.py  # DETR
│       ├── fcos.py  # FCOS
│       ├── frcnn.py  # Faster R-CNN
│       └── retina.py  # RetinaNet
├── predictor/  # 推理器
│   ├── __init__.py  # 推理包入口
│   ├── det_predictor.py  # 滑窗 / slab 推理
│   └── stitching.py  # 2.5D 跨 slab 链接成 3D 框
└── trainer/  # 训练循环
    ├── __init__.py  # 训练包入口
    └── det_trainer.py  # 检测训练与验证
```

## 关键概念

- **双几何**：3D 框与 2.5D slab 框共享同一套参数化和配对逻辑；2.5D 的 2D 框由 3D 真值框自动派生；patch 抽取几何按 `patch_mode` 与分割同语义（whole 全卷 resize、z_axis/2_5d 面内 resize、cubic 三轴裁剪），框随裁剪/缩放/翻转全程联动。
- **四个检测头模板**：RetinaNet、FCOS、Faster R-CNN、DETR 四头复用同一套骨干与金字塔适配器。
- **真值来源**：框真值只保留一份，既可以来自标注文件，也可以从分割 mask 连通域派生。
- **共享算子**：IoU、GIoU、NMS、ROIAlign、编解码都按框维度参数化，不依赖额外检测第三方扩展。
- **训练与评估**：训练侧主要看 patch 级 mAP，体级推理用 FROC 做汇总，更贴合医学小目标场景。
- **推理拼接**：2.5D 推理时先做 slab 级检测，再跨层 stitching 成 3D 框（可容忍 `det.stitch_max_gap` 个漏检 slab），拼接后做最终 3D NMS；推理可选 autocast 与 `det.tta_flips` 翻转 TTA。
- **训练工程**：每 epoch 原子写 latest_model.pth，`train.resume` 完整续训，history.json 逐 epoch 落盘，`train.early_stopping` 早停；非有限 loss/梯度丢弃 accum 组；warmup 段保持差分学习率倍率。

## 用法

```bash
# 训练
python -m dettask.train --config configs/det2_5d.yaml
python -m dettask.train --config configs/det3d.yaml

# 推理
python -m dettask.predict --config configs/det2_5d.yaml \
  --ckpt outputs/det2_5d/best_model.pth --npz-dir data/npz \
  --out-dir outputs/det2_5d/preds

# 冒烟测试
python tests/test_dettask_smoke.py
```
