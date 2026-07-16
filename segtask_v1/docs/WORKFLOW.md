# SegTask 分割流程总览（segtask_v1）

## 0. 共享主干

四方案共用一条主干，差异只在两处：**① Dataset 怎么抽样本；② Pipeline 怎么把样本变成模型输入**。

```
配置加载/校验 → 样本发现 + train/val 划分（可分层/按组）→ npz 自动烘焙(可选)
 → Dataset 抽样本（差异点①）
 → GPU 同步 3D 增强 → 中心裁剪（去 oversample 余量）
 → Pipeline 视图拆分/折叠（差异点②）
 → 模型 forward（AMP）→ 损失（fp32）→ backward → optimizer/scheduler → EMA
 → 验证（medium=随机patch / high=整卷滑窗）→ 选模/checkpoint/早停
```

### 四方案一页对比

| | whole | cubic | zaxis | 2.5d |
|---|---|---|---|---|
| 样本 | 整卷 resize | 3 轴 cube | z 向厚片 | z 向厚片（同 zaxis） |
| 采样中心 | 无 | 前景 3D 坐标 | 前景 z 切片 | 前景 z 切片 |
| oversample 方向 | 三轴 | 三轴 | 仅 z | 仅 z |
| 多分辨率 | 强制 [1.0] | 支持 | 支持 | 支持 |
| 模型维度 | 3D | 3D | 3D | 2D（D 折叠进通道） |
| 推理 | 整卷单前向 | 3 轴滑窗 | z 轴滑窗 | z 轴滑窗 + 可选 z-interleave |
| 适用 | 小目标定位/低精度快扫 | 各向同性大卷 | 各向异性 CT（z 厚层） | 各向异性 CT，省显存/大 H,W |

---

## 1. whole（`patch_mode: "whole"`）

```
【Dataset，CPU worker → (B, 1, eD, eH, eW)】
npz 读取（image/label/可选 rw，worker 级 LRU 缓存）
 → 预处理（img 归一化，lbl 保整数）
 → 整卷 resize 到 extract_size = round(patch_size × oversample)
   （不采中心、前景采样无效；samples_per_volume = 每 epoch 增强变体数）

【Trainer，GPU → (B, 1, pD, pH, pW)】
GPU 同步 3D 增强（空间变换 img/lbl/wmap 同步；强度变换仅 img）
 → 中心裁剪（去 oversample 余量）

【Pipeline：Vanilla3DPipeline → (B, 1, pD, pH, pW)】
直通不重塑（multi_res 强制 [1.0]）

【模型 + 损失】
3D 模型 forward（AMP）
 → MultiResolutionLoss（fp32；可选 deep supervision / 拓扑辅助损失）
 → backward → 梯度累积/裁剪 → optimizer/scheduler → EMA
```

推理：整卷 resize 到 (pD,pH,pW) → 单次前向 → 概率 resize 回原尺寸 → 阈值二值化 → 写 NIfTI。

---

## 2. cubic（`patch_mode: "cubic"`）

```
【Dataset，CPU worker → (B, 1, eD_max, eH_max, eW_max)】
npz 读取（image/label/可选 rw + 前景 3D 坐标索引 fg_coords，worker 级 LRU 缓存）
 → 预处理（img 归一化，lbl 保整数）
 → 采样中心点（前景偏置 + 类均衡；val 确定性，可选 Halton 网格覆盖）
 → 抽 max-FOV cube（三轴均含 oversample 与 multi-res 余量，越界 edge-pad）

【Trainer，GPU → (B, 1, pD×ms, pH×ms, pW×ms)】ms=max_scale
GPU 同步 3D 增强（空间变换 img/lbl/wmap 同步；强度变换仅 img）
 → 中心裁剪（去 oversample 余量）

【Pipeline：Vanilla3DPipeline → (B, 1, pD, pH, pW)】单分辨率默认路径
直通不重塑

【Pipeline：Patch3DNativeMultiResPipeline → (B, n_views, pD, pH, pW)】keep_native_multi_res=true
视图拆分（各 scale 同中心裁出原生尺寸 → 各自 resize 回 patch_size → 堆到通道）

【模型 + 损失】
3D 模型 forward（AMP）
 → MultiResolutionLoss（view 0 为主监督；fp32；可选 deep supervision / 拓扑辅助损失）
 → backward → 梯度累积/裁剪 → optimizer/scheduler → EMA
```

推理：3 轴滑窗（stride = patch×(1-overlap)，z 轴用 `z_overlap`、H/W 轴可用 `hw_overlap` 单设；短尾窗居中 edge-pad）→ 前向 → gaussian/average blend 加权累加 → 阈值二值化 → 还原 → 写 NIfTI。

---

## 3. zaxis（`patch_mode: "z_axis"`）

```
【Dataset，CPU worker → (B, 1, eD_max, H, W)】
npz 读取（image/label/可选 rw + 前景 z 切片索引 fg_slices，worker 级 LRU 缓存）
 → 预处理（img 归一化，lbl 保整数）
 → 采样中心 z（前景偏置 + 类均衡；val 确定性，可选均匀 z 网格覆盖）
 → 抽 max-FOV z-cube（z 含 oversample 与 multi-res 余量，越界 edge-pad；
   H/W 保持全尺寸整面 resize 到 (H, W)，不裁剪）

【Trainer，GPU → (B, 1, pD×ms, H, W)】ms=max_scale
GPU 同步 3D 增强（空间变换 img/lbl/wmap 同步；强度变换仅 img）
 → 中心裁剪（仅 z 向，去 oversample 余量）

【Pipeline：Vanilla3DPipeline → (B, 1, pD, H, W)】单分辨率默认路径
直通不重塑

【Pipeline：Patch3DNativeMultiResPipeline → (B, n_views, pD, H, W)】keep_native_multi_res=true
视图拆分（各 scale 同中心裁原生 D_k → z resize 回 pD → 堆到通道）

【模型 + 损失】
3D 模型 forward（AMP）
 → MultiResolutionLoss（view 0 为主监督；fp32；可选 deep supervision / 拓扑辅助损失）
 → backward → 梯度累积/裁剪 → optimizer/scheduler → EMA
```

推理：z 轴滑窗（H/W 整面 resize，z 按 stride 滑动）→ 前向 → 概率倒 resize 回原几何 + blend → 阈值二值化 → 还原 → 写 NIfTI。

---

## 4. 2.5d（`patch_mode: "2_5d"`）

```
【Dataset，CPU worker → (B, 1, eD_max, H, W)】
npz 读取（image/label/可选 rw + 前景索引，worker 级 LRU 缓存）
 → 预处理（img 归一化，lbl 保整数）
 → 采样中心 z（前景偏置 + 类均衡；val 确定性）
 → 抽 max-FOV z-cube（z 含 oversample 与 multi-res 余量，越界 edge-pad；
   H/W 保持全尺寸整面 resize 到 (H, W)，不裁剪）

【Trainer，GPU → (B, 1, D×max_scale, H, W)】
GPU 同步 3D 增强（空间变换 img/lbl/wmap 同步；强度变换仅 img）
 → 中心裁剪（去 oversample 余量）

【Pipeline：Slab2_5DPipeline → (B, n_views·D, H, W)】默认折叠路径
视图拆分（各 scale 同中心裁 + z resize 回 D → (B,n_views,D,H,W)；单分辨率透传）
 → 折叠 2D（(B,n_views,D,H,W) → (B,n_views·D,H,W)；lbl 取 view 0）

【Pipeline：Slab2_5DNativeDPipeline → (B, ΣD_k, H, W)】keep_native_view_depth=true
视图拆分（各 scale 同中心裁，保留原生 D_k 不 resize）
 → 折叠 2D（逐视图沿通道拼接 → (B,ΣD_k,H,W)；主 lbl 取 view 0，
   aux lbl 逐视图保原生 D_k 供 aux head 监督）

【Pipeline：Lift2_5DPipeline → (B, C, D, H, W)】lift_2_5d_to_3d=true（与上互斥）
视图拆分（同默认路径）
 → 不折叠（厚片保 3D 布局喂真 3D 模型，输出真 3D (B,num_fg,D,H,W)）

【模型 + 损失】
2D（lift 时 3D）模型 forward（AMP）
 → SliceChannelLoss（逐切片对齐，fp32；可选 deep supervision / aux 监督 / 拓扑辅助损失）
 → backward → 梯度累积/裁剪 → optimizer/scheduler → EMA
```

推理：

```
checkpoint 加载 → NIfTI 读取
 → 可选 bbox 裁剪 / spacing 重采样 → 与训练一致的归一化
 → 可选 AdaBN
 → z 轴滑窗（窗口构建与训练逐一镜像）→ 2D 前向 → 可选 flip TTA
 → 概率回原几何 + blend（gaussian/average）
 → 可选 z-interleave（按 z 间距拆 k 个交错子体独立推理再缝回，加宽 z 感受野）
 → 阈值二值化 → 还原 bbox/spacing → 写 NIfTI
```

对比 zaxis：数据抽取、增强、裁剪、视图拆分、滑窗几何全部相同；唯一差异：zaxis 不折叠、用 3D 模型，2.5d 折叠用 2D 模型 + 可选 z-interleave。

---

## 5. 通用训练技巧

### 精度 / 速度 / 显存

| 技巧 | 配置键 | 说明 |
|---|---|---|
| 混合精度 AMP | `train.use_amp` / `amp_dtype` | auto=Ampere+ 选 bf16 否则 fp16；fp16 自动带 GradScaler；**损失恒 fp32 计算** |
| torch.compile | `train.compile_mode` | default / reduce-overhead / max-autotune |
| channels_last | `train.channels_last` | 内存排布优化，数值等价 |
| GPU 预取 | `train.prefetch_to_gpu` | 独立 copy stream 提前一个 batch 上卡，H2D 与计算重叠（需 `data.pin_memory`） |
| 梯度检查点 | `model.grad_checkpointing` (+`grad_ckpt_encoder_stages`) | 反向重算激活，算力换显存 |
| CUDA 碎片缓解 | `train.cuda_expandable_segments` | expandable segments allocator |
| fused AdamW | `train.adamw_fused` | 单 kernel 更新全部参数 |
| EMA offload | `train.ema_device: cpu` | shadow 常驻 CPU，省 1× 参数显存 |

### 优化与正则

| 技巧 | 配置键 | 说明 |
|---|---|---|
| 优化器 | `train.optimizer` | adam / adamw / sgd(+nesterov) |
| 调度器 + warmup | `train.scheduler` / `warmup_epochs` | cosine / warm_restarts / poly / step / plateau / one_cycle |
| 梯度累积 | `train.grad_accum_steps` | 等效 batch = batch_size × 此值 |
| 梯度裁剪 | `train.grad_clip_norm` | 全局范数裁剪；非有限 loss/grad 自动跳步保护 |
| EMA | `train.use_ema` / `ema_decay` / `ema_warmup` | 权重指数滑动平均，验证/选模用 shadow |
| SWA | `train.swa_enabled` / `swa_start_ratio` | 尾段等权平均 + BN 重估，另存 swa_model.pth |
| 数据增强 | `augment.*` | GPU 端同步 3D：flip/仿射/弹性/grid-dropout（img/lbl/wmap 同步）+ 亮度/对比度/gamma/噪声/模糊/低清模拟（仅 img） |
| drop path / dropout | `model.drop_path_rate` / `dropout` | 结构正则 |

### 监督增强

| 技巧 | 配置键 | 说明 |
|---|---|---|
| 深监督 | `model.deep_supervision` + `loss.deep_supervision_weights` | 多个 decoder 级同时出预测加权求损 |
| aux 多 FOV 监督 | `model.aux_seg_supervision` + `loss.aux_supervision_weights` | 仅 2.5d 多视图；辅助视图各配 aux head 单独监督，权重默认 0.5^k；验证只看 view-0 |
| 拓扑辅助头 | `model.aux_topo_head` + `loss.aux_topo_*` | centerline/distance 辅助目标，仅训练期前向，零推理开销 |
| region weight | `loss.region_weights` 或 `data.region_weight_dir` | 空间加权损失（全局按 label 值 / 逐样本权重图） |
| 类均衡采样 | npz 逐类索引（make_data≥1.1 自动） | 前景采样先抽类再抽位置，防大类淹没小类 |
| 双源混采 | `data.npz_dir_secondary` + `mix_ratio` | 金标准/粗标按比例混 batch（val 仅金标准） |

### 分布式 / 工程

| 技巧 | 配置键 | 说明 |
|---|---|---|
| DDP 多卡 | `train.gpus` | no_sync 免非边界通信、静态图、bucket-view 省梯度显存 |
| ZeRO-1 | `train.zero_redundancy_optimizer` | 优化器状态分片 |
| checkpoint | `train.save_every` / `save_keep_last` / `save_async` | 周期+best+异步写盘；resume 位精确恢复（含 RNG） |
| 预训练迁移 | `train.pretrain` (+`pretrain_strict/load_ema/upkern`) | 仅加载权重初始化 |
| 早停 | `train.early_stopping` | 连续 N 次验证无提升即停 |
| 验证口径 | `train.val_metric_mode` | medium=随机 patch（快）/ high=整卷滑窗（与部署同口径） |
| 选模标准 | `train.save_best_criterion` (+`save_best_preset`) | loss / dice / iou / mcc / min_dice / dice+surface_dice / balanced |
| 监控 | `monitor.*` | jsonl + HTML 仪表盘 + 梯度/权重健康指标 |

---

## 6. 通用推理技巧

| 技巧 | 配置键 | 说明 |
|---|---|---|
| overlap blend | `predict.z_overlap` / `blend_mode` | gaussian（中心高）/ average；cubic 可用 `hw_overlap` 单设 H/W 轴（null=三轴同 z） |
| flip TTA | `predict.tta_flip` | 翻转变体预测取平均（2.5d 翻 H/W 4×） |
| AdaBN | `predict.adabn_enabled` / `adabn_mode` | 目标域重估 BN 统计（global 预热 / per_volume）；`adabn_sample_ratio`<1 抽样窗口估计降额外成本 |
| z-interleave | `predict.z_interleave_*` | 仅 2.5d；按物理 z 间距拆 k 个交错子体推理再缝回 |
| 阈值 | `predict.threshold` | 标量或逐前景类列表 |
| 显存逃生门 | `predict.acc_dtype/vol_dtype: fp16`、`accumulate_on_cpu` | 累加器/整卷半精度、累加放 CPU |
| 跳空窗 | `predict.skip_empty_windows` | 纯背景窗不前向（低强度启发式，zscore 需调阈值） |
| 提速 | `predict.cudnn_benchmark` / `use_inference_mode` / `channels_last` | 数值等价纯提速 |
| 概率输出 | `predict.save_probabilities` | 二值 mask 外另存概率图 |

**训练-推理一致性契约**（滑窗输入构建与 Dataset 抽取逐一镜像）：
`patch_size` / `patch_mode` / `multi_res_scales` / `keep_native_view_depth` /
`keep_native_multi_res` / `z_boundary_mode` / 归一化参数（intensity 窗、normalize、global_mean/std）/
`spacing_normalization` 必须与训练一致，否则几何/分布错位。

---

## 附：pipeline 选择速查（trainer/pipelines/factory.py）

| patch_mode | 旗标 | Pipeline |
|---|---|---|
| whole / z_axis / cubic | 默认 | Vanilla3DPipeline |
| z_axis / cubic | keep_native_multi_res | Patch3DNativeMultiResPipeline |
| 2_5d | 默认 | Slab2_5DPipeline |
| 2_5d | aux_seg_supervision | Slab2_5DAuxPipeline |
| 2_5d | aux + keep_native_view_depth | Slab2_5DNativeDPipeline |
| 2_5d | lift_2_5d_to_3d | Lift2_5DPipeline（+aux → Lift2_5DAuxPipeline） |
