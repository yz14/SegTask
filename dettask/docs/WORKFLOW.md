# dettask 检测流程总览

> 入口：`python -m dettask.train --config configs/det3d.yaml | det2_5d.yaml`；
> 推理：`python -m dettask.predict --ckpt best_model.pth`。
> 复用 `segtask_v1` 的配置/几何拓扑/预处理/优化器/AMP/EMA 基建，检测专属设置集中在 YAML 顶层 `det:` 段；
> 框算子（IoU/GIoU/NMS/ROIAlign/编解码）全部按框维度参数化、2D/3D 同一实现，不依赖 torchvision/CUDA 扩展。

---

## 0. 共享主干

```
配置加载（segtask Config + DetConfig）→ npz 发现 → train/val 划分
 → Dataset 抽 patch + 框真值联动（变长框 det_collate 保持 list）
 → DetectorModel：encoder → decoder 金字塔 → FPNAdapter → 检测头（4 模板）
 → 头内 fp32 损失 dict → 求和 → backward → optimizer/scheduler → EMA
 → 验证（patch 级 mAP 选模）
 → 推理：整卷滑窗/slab → NMS/拼接 → 3D 检出 → FROC
```

### 几何（`patch_mode`，与分割/SSL/分类同语义）

| 几何 | patch_mode | 模型输入 | 框 |
|---|---|---|---|
| 3D | `whole` / `z_axis` / `cubic` | (B, 1, D, H, W) | 3D 框 (N, 6)=[z1,y1,x1,z2,y2,x2]，半开区间 |
| 2.5D | `2_5d` | (B, D, H, W)，D 折进通道 | slab 内 2D 框 (N, 4)=[y1,x1,y2,x2]；推理端跨 slab 拼回 3D |

### 框真值契约（只存 3D 一份）

- npz `boxes` 键 (N, 7)=[z1,y1,x1,z2,y2,x2,cls] 优先；
- 否则由分割 mask 连通域派生（`det.boxes_from_mask`，逐前景类，`min_box_voxels` 滤噪点，逐卷缓存）；
- patch 裁剪时框同步联动（平移 → 裁到 patch 内 → 过滤可见比例 < min_visibility 的框）；
- 2.5D 折叠时由 3D 框对 slab 切片自动派生 2D 框。

---

## 1. 训练流程

```
【Dataset，CPU worker → image (B,1,D,H,W) 或 2.5D (B,D,H,W)，boxes/labels 变长 list】
npz 读取（image + boxes 键 / mask 派生框）
 → 预处理（img 归一化，与分割同参）
 → 采样偏移（训练随机；按 fg_oversample_ratio 概率以某 gt 框中心为锚，保证正样本供给；
   验证由 (seed, idx) 确定性派生）
 → 裁 patch + 框裁剪联动（crop_boxes）→ 2.5D 折叠 + 3D 框切片派生 2D 框

【Trainer，GPU】
前向：encoder → decoder 金字塔 → FPN 1×1 对齐 + 3×3 平滑（fpn_levels 选层）→ 检测头
 → 头内分配/编解码 + fp32 损失 dict（focal / GIoU 数值敏感，autocast 外）→ 求和
 → backward → 梯度裁剪 → optimizer / warmup+scheduler → EMA
 → 每 epoch 验证（EMA shadow：patch 级 predict → mAP@eval_iou_thresh）
 → 按 det.save_best_metric（map | loss）选模存 best_model.pth
```

### 四检测头模板（`det.arch`，共享 encoder+decoder 金字塔）

```
【retinanet】一阶段 anchor 基线（Retina U-Net 形态）
逐层 anchor 生成（sizes×ratios×scales，3D 加 z_scale）
 → 分配（max-IoU pos/neg 阈值 或 ATSS）→ 编码回归 + sigmoid Focal 分类

【fcos】anchor-free 逐点
特征图每点回归 distance-to-boundary（2·dim 距离）+ centerness
 → 层间按回归距离范围分工 → Focal + GIoU + centerness BCE

【faster_rcnn】两阶段
RPN（anchor + objectness）→ proposal（解码 + NMS，pre/post topk）
 → ROIAlign（grid_sample 自实现，取金字塔最高分辨率层，小目标优先）
 → 两层 FC → softmax(K+1 含背景) + 类无关框回归（RPN/ROI 采样配比可配）

【detr】Deformable-DETR 集合预测（免 NMS）
金字塔最低分辨率层 → 可学习 query + 参考点 → 纯 grid_sample 可变形交叉注意力
 → 逐解码层框细化（inverse-sigmoid 累加）→ 匈牙利匹配
 → focal-BCE + L1 + GIoU 集合损失（各解码层独立匹配求和，aux loss）
```

---

## 2. 通用训练技巧（复用 segtask `train.*`）

| 技巧 | 说明 |
|---|---|
| 混合精度 AMP | `use_amp` / `amp_dtype`（auto/bf16/fp16 + GradScaler）；检测损失全 fp32 |
| EMA | `use_ema`；验证与 best 保存均用 EMA shadow |
| warmup + scheduler | `warmup_epochs` / `scheduler`，按 step 推进 |
| 梯度裁剪 | `grad_clip_norm` |
| gt 框中心过采样 | 训练集 `max(data.foreground_oversample_ratio, 0.5)` 概率以 gt 框中心为锚（保证正样本供给；验证关闭）；验证每卷 patch 数 = samples_per_volume 的一半 |
| SSL/分割迁移 | `det.pretrained_ckpt`：命中 `encoder.*`（重建式 SSL 亦命中 `decoder.*`），strict=False + 命中统计，0 命中报错（几何不一致不静默） |
| 微调策略 | `det.encoder_lr_mult` 差分学习率（复用 clstask 分组实现）；`det.freeze_encoder` |
| 选模 | `det.save_best_metric`：map / loss（mAP@`eval_iou_thresh`，默认 0.1 医学小目标口径） |

---

## 3. 推理（整卷 → 3D 检出）

```
【3D】
三轴滑窗（步长 = patch 的 1/2，末窗贴边完整覆盖）
 → 窗内检出（score_thresh 过滤 → 逐类 NMS → max_dets 上限）
 → 平移回卷坐标 → 跨窗 3D 逐类 NMS 去重 → 3D 检出

【2.5D】
沿 z 逐 slab（步长 = slab 深度 / 2）→ 每 slab 2D 检出
 → 跨层拼接（相邻 slab 同类 2D 框按 yx IoU ≥ stitch_link_iou 贪心链接成链；
   z 范围 = 链覆盖 slab 并集，yx = 链内 min/max，分数取链内最大；
   z 跨度 < stitch_min_span 个 slab 的链丢弃）→ 3D 框
```

micro-batch 前向（`infer_batch_size`）防大卷 OOM；DETR 免 NMS（集合预测直接输出）。

---

## 4. 评估指标

- 训练验证：patch 级 COCO 式 mAP（逐类累计 TP/score → 全 recall 区间插值 AP → 宏平均，无 gt 类跳过）；
- 体级：FROC——给定每卷假阳个数阈值（`froc_fp_per_vol`）下的灵敏度均值（类无关口径）；
- FROC 统一在拼接后的 3D 框上评估：2.5D 与 3D 两几何同一读数口径；
- 匹配统一 IoU ≥ `eval_iou_thresh`（按分数降序贪心，一 gt 至多配一检出）。

---

## 5. 一致性契约

- 参数命名 `encoder.* / decoder.* / fpn.* / det_head.*`：encoder/decoder 与分割/SSL 同名同形，预训练权重 strict=False 直接迁移；
- 迁移时 patch_mode / spatial_dims / in_channels 必须与预训练一致（`validate_det` 交叉校验）；
- 框格式统一 (N, 2·dim) 体素坐标半开区间，坐标序与体素轴一致（2D=(y,x)，3D=(z,y,x)）；所有算子按 `dim = 列数 // 2` 自适应 2D/3D；
- 2.5D 的 2D 框永远由 3D 真值派生（训练裁剪联动、推理拼接还原），保证两几何在 3D 框上可比。
