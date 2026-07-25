# clstask 分类流程总览

> 入口：`python -m clstask.train --config configs/cls3d_cubic.yaml | cls2_5d.yaml`；
> 推理：`python -m clstask.predict --ckpt best_model.pth`。
> 复用 `taskcore` 的配置/几何拓扑/预处理/优化器/AMP/EMA 基建，分类专属设置集中在 YAML 顶层 `cls:` 段；
> 可直接接入 ssltask 预训练 encoder（`cls.pretrained_ckpt`），几何不一致直接报错、不静默降级。

---

## 0. 共享主干

```
配置加载（taskcore Config + ClsConfig）→ npz 发现 → train/val 划分（`data.group_id_regex` 非空时优先组级隔离，否则保持 `cls.stratify_split` 分层/随机语义）
 → Dataset 抽 patch + 标签派生 → GPU 增强（augment.enabled，可选）
 → mixup/cutmix（可选）
 → encoder（`cls.backbone`：encoder / densenet / vit）+ 池化 + cls_head → BCE/CE/Focal（fp32）
 → backward（非有限 loss/梯度跳组）→ optimizer/scheduler → EMA
 → 验证（patch 级 AUC/F1/acc + 卷级 MIL vol_auc/vol_f1/vol_acc 选模）
 → 推理：整卷铺格 → patch 前向 → MIL 聚合 → 卷级/逐 slice 概率
```

### 几何（`patch_mode`，与分割/SSL 同语义）

| 几何 | patch_mode | 模型输入 |
|---|---|---|
| 3D | `whole` / `z_axis` / `cubic` | (B, 1, D, H, W)，spatial_dims=3 |
| 2.5D | `2_5d` | (B, D, H, W)，D 折进通道（GPU 增强开启时增强后折叠，关闭时 Dataset 侧折叠，见 §5 契约）；仅单 FOV（`multi_res_scales=[1.0]`），不支持 `lift_2_5d_to_3d` |

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
npz 读取（image；mask 标签源 / 前景过采样时再读 label；
  data.cache_mode="memory" 时逐 worker LRU 缓存预处理后的卷，
  同卷连续索引下 samples_per_volume 次采样只读 1 次盘）
 → 预处理（img 归一化，与分割同参）
 → 采样中心（训练随机；可按 data.foreground_oversample_ratio 概率以前景 voxel 为中心，
   优先读 npz 预计算 fg_coords/fg_slices（含 *_cls 键时先选类再选点，类均衡），
   旧 npz 惰性回退一次计算并缓存；验证中心由 (seed, idx) 确定性派生，
   data.val_grid_coverage=true 时改网格铺点（z 等距 bin / Halton(2,3,5)，与推理同口径））
 → 按 patch_mode 抽 patch（与分割同语义：whole 全卷 resize；z_axis/2_5d z 轴抽取
   （越界 edge pad）+ 面内 resize 到 patch H/W；cubic 安全中心域三轴裁剪）
 → 标签派生（mask any() / table 继承卷标签）→ 2.5D 折叠（D→通道）
（augment.enabled=true 时：训练集输出未折叠 (1,D,H,W) image[+label]，
  由 trainer 在 GPU 上用 taskcore GPUAugmentor 联合增强后派生 target 再折叠）

【Trainer，GPU】
mixup / cutmix（可选，仅 volume 粒度；标签软化，CutMix λ 按实际裁剪体积回算）
 → 前向：encoder → 池化 → cls_head → logits
 → 损失（autocast 外 fp32 + logit clamp）→ backward → 梯度裁剪
 → 非有限守护（bf16/fp32：loss/梯度非有限丢弃本 accum 组；fp16 由 GradScaler 跳步）
 → optimizer / warmup+scheduler（按优化步计时）→ EMA（warmup + 可选 CPU offload）
 → 每 epoch 验证（EMA shadow 权重；patch 级 + 卷级 MIL 指标）
 → 按 cls.save_best_metric 选模存 best_model.pth（启用 EMA 时 model_state_dict
   为 EMA 权重，在线权重另存 model_online_state_dict）
```

### 模型（encoder 挂 `self.encoder`、头挂 `self.cls_head`，与 seg/SSL 命名一致）

| backbone | 说明 |
|---|---|
| `encoder`（resnet/convnext） | `taskcore.models.factory.build_backbone(cfg)`，与 SSL 同一构建路径 → `cls.pretrained_ckpt` strict=False 直接命中 `encoder.*` |
| `densenet` | DenseNet-BC（2D/3D），逐 stage 特征，instance norm + leakyrelu |
| `vit` | 标准 ViT（patch-embed → Pre-LN block ×N），单尺度特征图，mean-token 池化 |

分类头：volume 粒度全空间池化 → MLP → (B,K)；slice 粒度 → (B,K,D)
（2.5D：头输出 K×D 再 reshape；3D：池化 H/W 保 z → 逐深度共享 MLP → z 线性插值回 D）。

### 损失（`cls.loss_type`）

| 损失 | 适用 | 说明 |
|---|---|---|
| bce | 多标签 | sigmoid BCE，可选逐类 pos_weight / label smoothing |
| focal | 多标签 | sigmoid focal，软标签兼容 |
| ce | 单标签（table×volume） | softmax CE，接受硬标签或 mixup 软标签 |

---

## 2. 通用训练技巧（复用 `train.*`）

| 技巧 | 说明 |
|---|---|
| 混合精度 AMP | `train.use_amp` / `train.amp_dtype`（auto/bf16/fp16 + GradScaler）；损失 fp32 + logit clamp |
| EMA | `train.use_ema` / `train.ema_warmup` / `train.ema_device`；验证与 best 保存均用 EMA shadow（best 的 model_state_dict 即 EMA） |
| GPU 增强 | `augment.enabled`（复用 taskcore GPUAugmentor，image+label 联合增强后派生 target） |
| warmup + scheduler | `train.warmup_epochs` / `train.scheduler`，按优化步推进（accum 尾组按真实尾长归一）；warmup 段保持 encoder/head 差分 lr 倍率（`GroupWarmupScheduler`）；plateau 方向与 `cls.save_best_metric` 不一致时显式报错 |
| fused AdamW | CUDA 上自动启用 fused 实现（含差分 lr 分组分支） |
| 续训 / 落盘 | 每 epoch 原子写 latest_model.pth（模型+optimizer/scheduler/scaler/EMA）；`train.resume` 完整恢复；history.json 逐 epoch 落盘 |
| 早停 | `train.early_stopping`：连续 N 个 epoch 无提升即止 |
| 梯度裁剪 | `grad_clip_norm` |
| GPU 预取 | `train.prefetch_to_gpu`（复用 seg CudaPrefetcher：独立 copy stream 提前一个 batch 上卡，需 `data.pin_memory`） |
| mixup / cutmix | `cls.mixup_alpha` / `cls.cutmix_alpha`（>0 启用；同时启用每 batch 二选一；仅 volume 粒度） |
| 前景过采样 | `data.foreground_oversample_ratio`（仅训练集；复用 npz 预计算 fg 索引，类均衡采样） |
| 梯度检查点 | `model.grad_checkpointing`：反向重算激活、算力换显存；四模板全支持（encoder 系逐 stage（可配 `grad_ckpt_encoder_stages` 掩码）、DenseNet 逐 DenseBlock、ViT 逐 transformer Block）；eval/no_grad 零开销，数值与关闭时严格一致 |
| 扩展检查点范围 | `model.grad_ckpt_stem_downsample` / `model.grad_ckpt_decoder_branches` | 默认关闭，分别扩展 stem/downsample 与 decoder 分支覆盖 |
| 公共可选策略 | `data.resize_antialias`、`model.init_strategy` | 默认 false/`legacy`；分别控制 CPU 下采样预滤波和最终模型初始化策略 |
| DDP 多卡 | `train.gpus` 配≥2 张卡即启用（mp.spawn 每卡一进程，与 seg 同模式）：训练集 DistributedSampler（逐 epoch set_epoch 重洗）、验证集按 batch 块不相交分片；验证指标先跨卡聚齐全集 logits/targets 再算 AUC/F1（不可分解指标不做卡间平均）；checkpoint/history/monitor 仅 rank0 落盘；单卡/CPU 路径零变化 |
| 验证网格覆盖 | `data.val_grid_coverage`：验证 patch 改确定性网格铺点，与推理同口径 |
| 推理 TTA / AMP | `cls.tta_flips`：翻转 TTA（3D 7 组合；2.5D 仅 H/W；slice 粒度 z 翻转输出回翻）；推理 autocast 口径同训练 |
| SSL/分割迁移 | `cls.pretrained_ckpt`：只取 `encoder.*`（strict=False，打印命中/缺失统计）；仅 `backbone='encoder'` |
| 微调策略 | `cls.encoder_lr_mult` encoder 差分学习率；`cls.freeze_encoder=true` 只训头（linear probe） |
| 选模 | `cls.save_best_metric`：auc / f1 / acc / loss（patch 级）或 vol_auc / vol_f1 / vol_acc（卷级 MIL，与推理 agg_mode 同口径） |

---

## 3. 验证与指标

- 训练集按 `data.val_ratio` 划分（默认 `cls.stratify_split` 按标签分层：table 源用显式标签、mask 源用整卷多热真值分层，小类两侧均有代表；关闭回退纯随机）；验证每卷取确定性 patch（数量 = `cls.eval_patches_per_volume`，与推理铺格上限同源，选模与部署同口径；无前景过采样），前向 autocast 口径同训练/推理，收集全量 logits/targets 统一计算；
- 多标签：逐类 AUC（Mann-Whitney U rank 法，含并列校正）/ F1 / acc 宏平均；某类全正/全负跳过不计入；
- 单标签：one-vs-rest 宏 AUC + argmax acc / 宏 F1；
- slice 粒度把 (N, K, D) 摊平为 (N·D, K) 后同口径；
- 卷级 MIL 指标（vol_auc / vol_f1 / vol_acc）：按卷分组，用与推理同口径的 `aggregate_probs`（agg_mode/topk/lse_r）聚合 patch 概率；卷级 target：mask 源用整卷 label 派生的精确多热真值（优先读 meta.label_counts，旧 npz 回退整卷 any()；与 patch 抽样解耦），table 源为卷内常量。

---

## 4. 推理（整卷 → 卷级/逐 slice 概率）

```
整卷读取 → 预处理
 → patch 中心铺格（抽取几何与训练一致：2.5d/z_axis 沿 z 铺格 + 面内 resize；
   whole 全卷 resize；cubic 按各轴长 ceil(dim/patch) 分配铺格；
   上限 eval_patches_per_volume；slice 粒度 + z 铺格不受上限截断，
   保证厚卷全 z 覆盖）
 → micro-batch 前向（infer_batch_size 防 OOM；autocast 口径同训练；
   cls.tta_flips 翻转 TTA 均值）→ patch 概率
 → volume 粒度：MIL 聚合 agg_mode（mean / max / lse / topk）→ 卷级 (K,)
 → slice 粒度：逐 slice 概率按 patch 绝对 z 回填（whole 按比例回填），
   重叠切片取 max → 每卷 (K, Z)
```

MIL 直觉：卷中任一处阳性即卷阳性 → max/topk/lse 更贴合，mean 更稳。

---

## 5. 一致性契约

- 几何逐位一致：`patch_mode` × `build_topology` → `spatial_dims` / `in_channels` 与分割/SSL 同一派生路径，保证 encoder 权重无缝迁移；
- 预训练迁移时 patch_mode / spatial_dims / in_channels 必须与预训练一致（`validate_cls` 交叉校验，不一致直接报错）；
- 2.5D 仅单 FOV，与 image-only SSL 预训练口径一致；多 FOV 折叠为后续扩展点；
- 2.5D 折叠时机契约（全仓统一）：折叠发生在**数据增强之后、送模型之前**——GPU 增强路径 dataset 发未折叠 (1,D,H,W)，trainer 增强后派生 target 再折叠；关闭增强时 dataset 侧直接折叠（该路径无 3D 空间增强，两者等价）；
- mask 标签源要求 npz 含 `label` 键且 `len(fg_values)==num_classes`；table 源 pid 匹配缺失即报错（宁缺毋滥）。
