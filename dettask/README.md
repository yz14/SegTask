# dettask — 3D / 2.5D 医学影像目标检测

复用 segtask_v1 基建（配置 / 几何拓扑 / 预处理 / Encoder+Decoder / 优化器 /
AMP / EMA）的检测工程。共享 Encoder+Decoder 金字塔即 Retina U-Net 形态，
SSL（ssltask）预训练 `encoder.*`（重建式含 `decoder.*`）权重可直接迁移。

## 双几何

与分割/分类同一套 `patch_mode` 语义：

| 几何 | patch_mode | 框格式 | 算子路径 |
|------|-----------|--------|---------|
| 3D   | `whole` / `z_axis` / `cubic` | `[z1,y1,x1,z2,y2,x2]` | 3D 卷积头 + 自实现 3D NMS / ROIAlign |
| 2.5D | `2_5d`（slab 折叠） | slab 内 `[y1,x1,y2,x2]` | 2D 头；推理跨 slab IoU 链接拼 3D 框 |

框真值只存 3D 一份（npz `boxes` 键 `(N,7)=[z1,y1,x1,z2,y2,x2,cls]`，或由
分割 mask 连通域派生 `det.boxes_from_mask`）；2.5D 的 2D 框由 3D 框对 slab
切片自动派生（`targets.slice_boxes_to_2d`）。所有框算子（IoU/GIoU/NMS/
ROIAlign/编解码/增强联动）以 `dim = 框列数 // 2` 参数化，2D/3D 同一实现，
零第三方检测依赖（不引 torchvision / CUDA 扩展）。

## 四个检测头模板（`det.arch`）

| 模板 | 范式 | 关键机制 |
|------|------|---------|
| `retinanet` | 一阶段 anchor | Focal + GIoU，anchor 尺寸/比例/z-scale 可配，max-IoU / ATSS 分配 |
| `fcos` | anchor-free 逐点 | distance-to-boundary + centerness，层间按回归距离范围分工 |
| `faster_rcnn` | 两阶段 | RPN → proposal NMS → grid_sample 版 ROIAlign（2D/3D）→ K+1 softmax |
| `detr` | Transformer 集合预测 | grid_sample 版可变形交叉注意力 + 可学习参考点 + 匈牙利匹配，免 NMS |

四头共享 `FPNAdapter`（decoder 金字塔 1×1 通道对齐 + 3×3 平滑，
`det.fpn_levels` 选层）；DETR 取金字塔最低分辨率层。

## 数据 / 增强联动

`DetPatchDataset` 抽 patch 时框同步联动（`targets.crop_boxes`：平移 + 裁剪 +
可见比例过滤）；`fg_oversample_ratio` 概率以某 gt 框为中心抽 patch 保证正
样本供给。`targets.flip_boxes` 提供翻转联动（半开区间口径，单测覆盖往返
一致性）。

## 训练 / 评估

* `DetTrainer`：复用 segtask optim/warmup/AMP/EMA；损失 fp32；
  encoder 差分学习率 `det.encoder_lr_mult`。
* patch 级验证 mAP@`det.eval_iou_thresh`（医学小目标默认 0.1）；
  `det.save_best_metric ∈ {map, loss}` 选模。
* 体级 FROC（`DetPredictor.predict_dir`）：统一在 3D 框上评估
  （2.5D 拼接后 / 3D 滑窗 NMS 后同一口径），
  `sens@{0.125..8}fp` + 均值 `froc`。

## 推理

* 3D：三轴滑窗（1/2 重叠）→ 窗内检出平移回卷坐标 → 跨窗逐类 3D NMS。
* 2.5D：沿 z 逐 slab（1/2 重叠）→ 每 slab 2D 检出 →
  `stitch_slab_detections` 按 `det.stitch_link_iou` 跨层链接、
  `det.stitch_min_span` 过滤 → 3D 框。

## SSL / 分割权重迁移

```yaml
det:
  pretrained_ckpt: outputs/ssl_2_5d/ssl_best.pt   # encoder.* (+decoder.*)
  encoder_lr_mult: 0.1
  # freeze_encoder: true
```

strict=False + 命中数打日志；0 命中直接报错（几何 patch_mode/spatial_dims/
in_channels 与预训练不一致时不静默）。

## 用法

```bash
# 训练（2.5D RetinaNet / 3D 各一份 reference 配置）
python -m dettask.train --config configs/det2_5d.yaml
python -m dettask.train --config configs/det3d.yaml --override det.arch=fcos

# 整卷推理 + FROC
python -m dettask.predict --config configs/det2_5d.yaml \
    --ckpt outputs/det2_5d/best_model.pth --npz-dir data/npz \
    --out-dir outputs/det2_5d/preds

# 冒烟测试（单测 + 四头×双几何 + 训练/迁移/推理端到端）
python tests/test_dettask_smoke.py
```

## 目录

```
dettask/
├── config.py            # DetConfig + validate_det + YAML I/O/overrides
├── ops.py               # IoU/GIoU/NMS/batched NMS/ROIAlign（2D/3D 同构）
├── targets.py           # anchor 生成/分配(max-IoU,ATSS)/编解码/增强联动/切片派生
├── metrics.py           # mAP + FROC
├── data/                # DetPatchDataset（boxes/mask 双源）+ loader
├── models/
│   ├── fpn.py           # Decoder 金字塔 → FPNAdapter
│   ├── heads/           # retina / fcos / frcnn / detr
│   ├── detector.py      # DetectorModel（encoder/decoder/fpn/det_head）
│   └── factory.py       # build_detector + load_pretrained_backbone
├── losses/det_loss.py   # focal / 框回归 / 匈牙利匹配
├── trainer/det_trainer.py
├── predictor/           # det_predictor（滑窗/slab）+ stitching（2.5D→3D）
└── docs/detection_models_survey.md
```
