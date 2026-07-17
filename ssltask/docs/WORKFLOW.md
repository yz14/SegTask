# ssltask 自监督预训练流程总览

> 入口：`python -m ssltask.pretrain --config configs/ssltask_<method>.yaml`；
> 产出 `ssl_best.pt`，下游用 `train.pretrain=<路径>` 非严格加载衔接（enc(+dec) 命中，任务头随机）。
> 训练循环对方法完全无感：数据/优化/AMP/EMA/ckpt 通用，"破坏→损失"全部封装在 `SSLMethod` 插件内。

---

## 0. 共享主干

```
配置加载（复用 segtask 配置 + SSLConfig 叠加）→ npz 发现（image-only，无 train/val 划分）
 → Dataset 抽 patch（无标签，随机中心）
 → GPU 同步 3D 增强（仅重建类方法）→ z 中心裁剪 → 2.5D 折叠
 → method.compute_loss（差异点：破坏/掩码/多视图构造 + 目标）
 → backward → 梯度累积/裁剪 → optimizer/scheduler → EMA
 → 在线探针（表征质量选模）→ checkpoint（导出可迁移 encoder）
```

### 方法一页对比（`ssl.method`）

| 方法 | 范式 | 目标 | 预训练解码器 | trainer 增强 |
|---|---|---|---|---|
| genesis | 还原 | 多变换破坏→重建干净原图（全体素） | 是（整 UNet） | 是 |
| prior | 先验回归 | 回归 Frangi vesselness 置信图 | 是 | 是 |
| simmim | 像素掩码 | mask token 稠密前向，仅被遮位点 L1 | 否（极轻头） | 是 |
| spark | 像素掩码 | 掩码-稠密等价 + 层次解码器，仅被遮位点 L2 | 否（用完即弃） | 是 |
| dino | 自蒸馏 | 多裁剪 EMA 师生分布一致 | 否 | 否（自带多视图） |
| dino_gram | 自蒸馏 | DINO + 密集特征 Gram anchoring | 否 | 否 |
| ibot | 蒸馏+隐空间掩码 | DINO + 被遮位点密集特征预测 | 否 | 否 |
| sparkdino | 掩码+蒸馏 | L_SparK + μ·L_DINO 双分支共享 encoder | 否 | 否 |
| jepa | 隐空间掩码 | 预测被遮位点的 EMA 目标特征（L2+VICReg 正则） | 否 | 否 |
| byol | 对比（无负样本） | 在线网预测 EMA 目标网投影 | 否 | 否 |
| moco | 对比 | 动量队列 InfoNCE | 否 | 否 |
| vicregl | 稠密对应 | 全局 VIC 三项 + 位置匹配稠密 VIC（孪生） | 否 | 否 |

---

## 1. 数据与训练循环（全方法共用）

```
【Dataset，CPU worker → (B, 1, eD, pH, pW)】
npz 读取（仅读 image 键，image-only；不读 label/前景索引；worker 级 LRU 缓存）
 → 预处理（img 归一化，与下游 segtask 同参）
 → 随机采样中心（2_5d/z_axis：随机 z；cubic：三轴随机；whole 不支持）
 → 抽 patch（z 含 oversample 余量，越界 edge-pad；
   2_5d/z_axis：H/W 保持全尺寸整面 resize 到 (pH, pW)，不裁窗，与 segtask 同口径）

【Trainer，GPU】
GPU 同步 3D 增强（仅 trainer_augment=True 的重建类；增强后图即新的自洽重建样本）
 → z 中心裁剪（去 oversample 余量 eD→pD）
 → 2.5D 折叠（spatial_dims=2 时 (B,1,D,H,W)→(B,D,H,W)，D 进通道；仅单 FOV：
   要求 in_channels==pD、multi_res_scales==[1.0]）

【方法插件：method.compute_loss（AMP 内调用，损失内部 fp32）】
破坏 / 掩码 / 多视图构造（见 §2）
 → 前向（student / context / enc+dec，按方法）
 → 方法目标损失
 → backward → 梯度累积/裁剪 → optimizer/scheduler → EMA
 → on_after_step（方法内 teacher EMA / 温度·动量调度，仅真实优化步推进状态）
```

无验证集：选模靠训练 loss 或在线探针（§4）。

---

## 2. 各方法核心流程

### 还原 / 像素掩码类（输入=破坏图，目标=像素）

```
genesis   干净图 → Genesis 破坏（Bézier 强度变换 / 局部打乱 / 内补全 / 外补全，按概率组合）
           → 整套 enc+dec 重建干净图 → 全体素 recon_loss（l1/smooth_l1/mse）
prior     干净图（可选 Genesis 破坏输入）→ enc+dec 回归 Frangi vesselness 目标
           （多尺度、prior_spacing 非空时按物理 mm 解释，各向异性感知）→ 全体素 recon_loss
simmim    单元网格掩码（mim_mask_ratio）→ 被遮单元换可学习 mask_token → 稠密 encoder
           → 极轻像素头（无 skip）→ 仅被遮位点 L1
spark     单元掩码（spark_mask_ratio=0.6，unit=16）→ 掩码-稠密等价（置零 + 逐尺度门控）
           → densify（mask_embed 填空位）→ 轻量层次解码器（逐级上采样 + 横向融合）
           → 仅被遮位点 L2（可选 per-unit 归一化目标 spark_norm_pix）
```

### 自蒸馏 / 混合类（EMA 教师，教师恒 eval）

```
dino      multi-crop（global×2 + local×6，scale 按体积占比；翻转 + 强度增广）
           → 学生看全部裁剪 / EMA 教师只看 global → 全局池化 + 投影头 → softmax
           → 教师 center（EMA 中心化）+ sharpen（低温）→ 学生×教师配对交叉熵
           → 教师动量 cosine 0.996→1.0；前段冻结学生投影头末层
dino_gram 上者 + Gram anchoring：进度 ≥ start_frac 后，学生 global 裁剪的密集特征
           Gram 矩阵逼近"早期 EMA 教师快照"；快照首次生效时锚定、
           每 dino_gram_refresh_steps 步刷新；L = L_DINO + λ·L_gram
ibot      上者（DINO 全局项）+ iBOT 掩码密集项：新 global 裁剪按 ibot_mask_ratio 掩码
           （mask-token 稠密输入）→ 被遮位点学生密集特征 vs 教师（看完整图）
           在原型上的逐位点交叉熵（独立/共享头）；L = L_DINO + λ·L_iBOT
sparkdino 双分支共享学生 encoder：DINO 分支（同 dino）+ SparK 分支在原始整图上重建（同 spark）
           L = L_SparK + μ·L_DINO
```

### 对比 / 稠密对应 / 隐空间类

```
byol      两增广视图 → 在线网（投影+预测头）预测 EMA 目标网投影 → 余弦损失（无负样本）
moco      两视图 → query encoder vs EMA key encoder + 动量负样本队列 → InfoNCE
vicregl   成对高重叠裁剪（带坐标元数据）→ 全局 VIC 三项（invariance/variance/covariance）
           + overlap-aware 位置匹配位点的稠密 VIC（孪生共享权重，无 EMA/负样本）
jepa      EMA 目标编码器编码完整图 → 上下文编码器看遮后输入（单元掩码）
           → 轻量预测器预测被遮位点特征 → L2（目标侧 stop-grad）+ VICReg 方差/协方差防坍缩
```

---

## 3. 通用训练技巧（复用 segtask `train.*` 配置）

| 技巧 | 说明 |
|---|---|
| 混合精度 AMP | `use_amp` / `amp_dtype`（auto/bf16/fp16，fp16 带 GradScaler）；方法损失内部 fp32 |
| 优化步时钟 | scheduler / warmup / global_step / 方法内调度均按**真实 optimizer.step 边界**推进（非 micro-batch）；尾批按实际累积长度归一 |
| 跳步一致性 | loss 非有限或 GradScaler 内部跳步：调度时钟照常推进，但 EMA / center / queue / Gram 快照一律冻结；DDP 下跳步决策 all-reduce(any) 统一 |
| EMA | trainer 级 `use_ema`（正交于方法内 teacher）；导出权重 EMA 优先 |
| teacher 恒 eval | 所有方法内冻结 EMA 分支（DINO/iBOT teacher、BYOL target、MoCo key、JEPA target）覆写 `train()` 保持 eval |
| 梯度累积 / 裁剪 | `grad_accum_steps` / `grad_clip_norm`；非有限梯度跳步保护 |
| DDP 多卡 | mp.spawn + 初始参数广播 + accum 边界手动梯度均值 all-reduce（方法直调子模块，不套 DDP wrapper）；epoch loss 按样本数加权 all-reduce |
| ZeRO-1 | 优化器状态分片；保存前集合式 consolidate 到 rank0 |
| torch.compile / channels_last | `compile_mode` / `channels_last`（2.5D 折叠后转 4D 排布） |
| 梯度检查点 | `model.grad_checkpointing` (+`grad_ckpt_encoder_stages`)：encoder/decoder 经公共 factory 构建即生效；SSL 特有 wrapper（投影头/predictor 等）激活占用小，刻意不包检查点 |
| checkpoint | 原子写（临时文件 + os.replace）+ sha256 状态指纹；`save_async` 后台线程写盘；仅 rank0 落盘 |
| resume | 全状态：method（含 teacher/queue/center buffer）+ optimizer/scheduler/scaler/EMA + RNG；指纹校验、方法名校验；rank>0 重新分流 RNG |
| 监控 | 复用 segtask monitor：jsonl + HTML 仪表盘 + 梯度/权重健康指标（失败隔离不阻断训练） |

---

## 4. 评测与选模

```
在线分割探针（ssl.probe_enabled，每 probe_every 个 epoch，仅 rank0）
 → 冻结（或低 lr 微调）当前 encoder → 固定种子重置多尺度 1×1 线性头 → 训练固定步数
 → 标注 npz 上报 probe Dice / HD95（spacing-aware，空掩码显式计数）
 → probe_select_best=true 时以 probe Dice 选 best（否则按训练 loss）
```

- 探针数据组级（患者级）train/val 划分：同组绝不跨集；验证 patch 确定性 + 前景感知。
- 在线分类探针（cls probe）：encoder + 全局池化 + MLP，多标签"每类是否出现"。
- 离线 P6 评测（`python -m ssltask.evaluate`）：嵌套 few-shot 子集（大 shots 包含小 shots）
  + frozen / finetune 双读数 + from-scratch（B2）同路径基线，横向对比多份 SSL 权重。

---

## 5. 下游迁移

```
ssl_best.pt（EMA 优先导出）/ ssl_last.pt / ssl_resume.pt（全状态续训）
 → model_state_dict 键与 segtask build_model 逐参数同名（encoder.* / decoder.*）
 → 下游 train.pretrain=<路径>（strict=False）：enc(+dec) 命中、任务头随机
```

**一致性契约**：patch_size / patch_mode / 归一化参数须与下游任务一致（探针 encoder
`strict=True` 校验同名同形）；2.5D 预训练仅支持单 FOV（`in_channels==patch_size[0]`、
`multi_res_scales==[1.0]`），多 FOV 需增强级多分辨率裁剪、不在 image-only 通路内；
patch_mode 支持 `2_5d`/`z_axis`/`cubic`——**有意不支持 `whole`**（整体 resize 抹掉分辨率
信息，与 SSL 学局部结构的目标冲突，且预训练分布与下游 patch 训练分布不一致）；
2.5D 折叠时机契约（全仓统一）：dataset 恒发未折叠 3D，折叠由 trainer 在**数据增强之后、
送模型之前**统一完成（同 seg `squeeze_2_5d` 口径），3D GPUAugmentor 因此也作用于 2.5D 样本。
