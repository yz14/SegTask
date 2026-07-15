# clstask 分类流程总览

> 入口：`python -m clstask.train --config configs/cls3d_cubic.yaml | cls2_5d.yaml`；
> 推理：`python -m clstask.predict --ckpt best_model.pth`。
> 复用 `segtask_v1` 的配置/几何拓扑/预处理/优化器/AMP/EMA 基建，分类专属设置集中在 YAML 顶层 `cls:` 段；
> 可直接接入 ssltask 预训练 encoder（`cls.pretrained_ckpt`），几何不一致直接报错、不静默降级。

---

## 0. 共享主干

```
配置加载（segtask Config + ClsConfig）→ npz 发现 → train/val 划分
 → Dataset 抽 patch + 标签派生 → mixup/cutmix（可选）
 → encoder（4 模板）+ 池化 + cls_head → BCE/CE/Focal（fp32）
 → backward → optimizer/scheduler → EMA → 验证（AUC/F1/acc 选模）
 → 推理：整卷铺格 → patch 前向 → MIL 聚合 → 卷级/逐 slice 概率
```

### 几何（`patch_mode`，与分割/SSL 同语义）

| 几何 | patch_mode | 模型输入 |
|---|---|---|
| 3D | `whole` / `z_axis` / `cubic` | (B, 1, D, H, W)，spatial_dims=3 |
| 2.5D | `2_5d` | (B, D, H, W)，D 折进通道（Dataset 侧即折叠）；仅单 FOV（`multi_res_scales=[1.0]`），不支持 `lift_2_5d_to_3d` |

### 标签契约（`cls.label_source` × `cls.label_granularity`）

| 来源 × 粒度 | target | 说明 |
|---|---|---|
| mask × volume | (K,) 多热 | 由分割 mask 派生弱标签：每前景类"patch 内是否出现"；K=num_fg_classes |
| mask × slice | (K, D) | 每前景类"每 z 切片是否出现" |
| table × volume | (K,) 多热 或 标量 long | 显式标签表（csv/json，pid→标签，按 npz 基名匹配、缺失即报错）；单标签走 softmax CE |

---

## 1. 训练流程

```
【Dataset，CPU worker → (B, 1, D, H, W) 或 2.5D (B, D, H, W)】
npz 读取（image；mask 标签源 / 前景过采样时再读 label）
 → 预处理（img 归一化，与分割同参）
 → 采样中心（训练随机；可按 data.foreground_oversample_ratio 概率以前景 voxel 为中心；
   验证中心由 (seed, idx) 确定性派生，epoch 间可复现）
 → 三轴裁 patch（越界 edge 复制；无 H/W 整面 resize、无 GPU 增强管道）
 → 标签派生（mask any() / table 继承卷标签）→ 2.5D 折叠（D→通道）

【Trainer，GPU】
mixup / cutmix（可选，仅 volume 粒度；标签软化，CutMix λ 按实际裁剪体积回算）
 → 前向：encoder → 池化 → cls_head → logits
 → 损失（autocast 外 fp32 + logit clamp）→ backward → 梯度裁剪
 → optimizer / warmup+scheduler → EMA
 → 每 epoch 验证（EMA shadow 权重）→ 按 cls.save_best_metric 选模存 best_model.pth
```

### 模型（encoder 挂 `self.encoder`、头挂 `self.cls_head`，与 seg/SSL 命名一致）

| backbone | 说明 |
|---|---|
| `encoder`（resnet/convnext） | 复用 `segtask_v1.models.factory.build_model(cfg).encoder`，与 SSL 同一构建路径 → `pretrained_ckpt` strict=False 直接命中 `encoder.*` |
| `densenet` | DenseNet-BC（2D/3D），逐 stage 特征，instance norm + leakyrelu |
| `vit` | 标准 ViT（patch-embed → Pre-LN block ×N），单尺度特征图，mean-token 池化 |

分类头：volume 粒度全空间池化 → MLP → (B,K)；slice 粒度 → (B,K,D)
（2.5D：头输出 K×D 再 reshape；3D：池化 H/W 保 z → 逐深度共享 MLP → z 线性插值回 D）。

### 损失（`cls.loss`）

| 损失 | 适用 | 说明 |
|---|---|---|
| bce | 多标签 | sigmoid BCE，可选逐类 pos_weight / label smoothing |
| focal | 多标签 | sigmoid focal，软标签兼容 |
| ce | 单标签（table×volume） | softmax CE，接受硬标签或 mixup 软标签 |

---

## 2. 通用训练技巧（复用 segtask `train.*`）

| 技巧 | 说明 |
|---|---|
| 混合精度 AMP | `use_amp` / `amp_dtype`（auto/bf16/fp16 + GradScaler）；损失 fp32 + logit clamp |
| EMA | `use_ema`；验证与 best 保存均用 EMA shadow |
| warmup + scheduler | `warmup_epochs` / `scheduler`，按 step 推进 |
| 梯度裁剪 | `grad_clip_norm` |
| mixup / cutmix | `cls.mixup_alpha` / `cls.cutmix_alpha`（>0 启用；同时启用每 batch 二选一；仅 volume 粒度） |
| 前景过采样 | `data.foreground_oversample_ratio`（仅训练集；缓解类不平衡，需 npz 有 label） |
| SSL/分割迁移 | `cls.pretrained_ckpt`：只取 `encoder.*`（strict=False，打印命中/缺失统计）；仅 `backbone='encoder'` |
| 微调策略 | `cls.encoder_lr_mult` encoder 差分学习率；`cls.freeze_encoder=true` 只训头（linear probe） |
| 选模 | `cls.save_best_metric`：auc / f1 / acc / loss |

---

## 3. 验证与指标

- 训练集按 `data.val_ratio` 划分；验证每卷取确定性 patch（samples_per_volume 的一半，无前景过采样），收集全量 logits/targets 统一计算；
- 多标签：逐类 AUC（Mann-Whitney U rank 法，含并列校正）/ F1 / acc 宏平均；某类全正/全负跳过不计入；
- 单标签：one-vs-rest 宏 AUC + argmax acc / 宏 F1；
- slice 粒度把 (N, K, D) 摊平为 (N·D, K) 后同口径。

---

## 4. 推理（整卷 → 卷级/逐 slice 概率）

```
整卷读取 → 预处理
 → patch 中心铺格（2.5d/z_axis 沿 z 均匀铺格，H/W 大于 patch 时面内也铺；
   cubic 三轴铺格；上限 eval_patches_per_volume）
 → micro-batch 前向（infer_batch_size 防 OOM）→ patch 概率
 → volume 粒度：MIL 聚合 agg_mode（mean / max / lse / topk）→ 卷级 (K,)
 → slice 粒度：逐 slice 概率按 patch 绝对 z 回填，重叠切片取 max → 每卷 (K, Z)
```

MIL 直觉：卷中任一处阳性即卷阳性 → max/topk/lse 更贴合，mean 更稳。

---

## 5. 一致性契约

- 几何逐位一致：`patch_mode` × `build_topology` → `spatial_dims` / `in_channels` 与分割/SSL 同一派生路径，保证 encoder 权重无缝迁移；
- 预训练迁移时 patch_mode / spatial_dims / in_channels 必须与预训练一致（`validate_cls` 交叉校验，不一致直接报错）；
- 2.5D 仅单 FOV，与 image-only SSL 预训练口径一致；多 FOV 折叠为后续扩展点；
- mask 标签源要求 npz 含 `label` 键且 `len(fg_values)==num_classes`；table 源 pid 匹配缺失即报错（宁缺毋滥）。
