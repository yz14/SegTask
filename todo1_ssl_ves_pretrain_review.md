# TODO #1 · `ssltask_ves_pretrain.yaml`（prior / 2.5D 单 FOV）聚焦审查报告

> 依据 TODO 第 1 项：以 `configs/ssltask_ves_pretrain.yaml`（`method=prior`，Frangi
> vesselness 回归，2.5D 单 FOV）为锚，对 SSL 全流程做**配置特定**审查，并逐域对照
> `segtask_v1` 的 2.5D 单 FOV，验证两个命题：
> 1. **"1:1 一致"** —— 该 yaml 下 SSL 全流程原则上应与 seg 2.5D 单 FOV 完全一致；
> 2. **"技巧可借鉴"** —— seg 的训练技巧应可迁移到 SSL。
>
> 本轮**未修改任何代码/文档**；所有结论基于对当前源码的逐文件精读（含 seg↔ssl 逐行对照）。
> 标注 `[高]/[中]/[低]` 为影响优先级。

---

## 0. 结论速览

**"1:1 一致"命题：成立（在合理约束下）。** 几何派生、IO/预处理/抽取、encoder/decoder
装配、2.5D 折叠、权重迁移契约在本 yaml 下与 seg 2.5D 单 FOV **逐项等价或字面共享代码**。
少数分歧（采样策略、无 val loader、独立 trainer 实现）均是 SSL 语义/工程的**必要差异**，
非缺陷。

**"技巧可借鉴"命题：大部分已落地。** 优化步时钟、非有限 loss 守护并跳步、梯度累积尾组、
异步 checkpoint、全状态 resume、EMA/ZeRO/channels_last/compile、监控面板、DDP 梯度
all-reduce 均已在 SSL trainer 实现，与 seg 等价。

**⚠ 重要：既有 `ssltask_investigation_report.md` 已显著过期。** 其标为 P0 的 4.1（bf16
非有限守护缺失）、以及 1.1（无卷缓存）、4.3（阻塞保存）、4.4（无 resume）、4.7（无监控）
**在当前代码中均已修复/实现**。本报告以现状为准，不再复述这些已解决项。

**唯一值得优先处理的实质项：** `[中]` 数据侧**内容偏置采样缺失** —— image-only 数据集
均匀随机抽 z 中心，CT 空气占比高会产出大量"近空" patch，其 vesselness 目标近乎全零，
对 prior 是弱/退化监督。这是本 yaml 下最有价值的优化点（详见 §1.2）。

---

## 1. 数据读取（Domain 1）

**文件**：`ssltask/data/ssl_dataset.py` ↔ `segtask_v1/data/dataset.py`

### 1:1 对照

| 环节 | seg 2.5D 单 FOV | ssl（本 yaml） | 判定 |
|---|---|---|---|
| npz 打开 | `_open_npz` | `_open_npz`（**同一函数**） | 字面一致 |
| 预处理 | `preprocess_image`（clip+minmax） | `preprocess_image`（**同一函数**） | 字面一致 |
| z 抽片 | `extract_z_patch_padded`（edge-pad） | `extract_z_patch_padded`（**同一函数**） | 字面一致 |
| 面内缩放 | `resize_3d`（H/W→pH,pW，order=1） | `resize_3d`（**同一函数**） | 字面一致 |
| 卷缓存 | `VolumeCache` LRU | `VolumeCache` LRU（**同一类**，`ssl_dataset.py:176`） | 一致 |
| z 过采样 | `aug_oversample_ratio` z 余量 | 同口径（yaml=1.0，无余量） | 一致 |

抽取几何（z 抽片 edge-pad + 面内整片 resize）由**共享函数**保证与 seg 逐字一致
（`ssl_dataset.py:211-217`）。`build_ssl_dataloader` 对 2.5D 强制 `in_channels==patch_size[0]`
单 FOV 校验（`:433`），错误信息清晰。DDP 用 `DistributedSampler` 分片 + 每 epoch
`set_epoch`，与 seg 语义一致。

### 合理分歧（非缺陷）

- **采样策略**：seg `_sample_z` 支持前景偏置（`fg_ratio`、逐类均衡）+ 验证确定性网格覆盖
  （`dataset.py:862-881`）；ssl `_rand_center` 均匀随机（`ssl_dataset.py:97-105`）。**SSL 无
  标签，无法做前景采样** —— 此分歧是语义必然，不是 bug。

### 问题与优化

**[中] 1.1 缺内容偏置采样（本 yaml 下影响最大的实质项）。**
`ImageOnlyPatchDataset.__getitem__` 均匀随机取 z 中心（`:215`）。胸部 CT 大量 z 层为
空气/床板，均匀采样将产出大比例"近空" slab；其 Frangi vesselness 目标近乎全零，prior
回归到常数即可"拟合"，是**弱监督甚至捷径**。
- 建议（opt-in、零依赖）：对抽到的 patch 做廉价的**内容拒绝采样**（强度方差 / 非背景
  体素占比低于阈值则重采样，最多重试 K 次）。这与 seg 报告 §1 建议、nnU-Net 前景/背景
  采样比同族，对 prior/genesis 的有效监督密度提升明显。

**[低] 1.2 语料 spacing 混杂假设未强约束。** 数据集假定各 npz 已按同一 spacing 预处理；
若混杂，patch 物理尺度不一致会影响 Frangi 尺度语义。属数据准备约定，建议文档显式声明
（`prior_spacing` 见 §3.2）。

---

## 2. 模型构建（Domain 2）

**文件**：`ssltask/models/ssl_models.py`、`ssltask/methods/prior.py` ↔ `segtask_v1/models/factory.py`、`topology.py`

### 1:1 对照（核心，命题成立的关键）

- **几何派生同源**：`in_channels` / `spatial_dims` 由 seg `build_topology` 统一派生
  （`topology.py:103-112`）。本 yaml：`patch_mode=2_5d`、`multi_res_scales=[1.0]`、
  `patch_size[0]=12` → `spatial_dims=2`、`in_channels=D×n_views=12×1=12`。ssltask 复用
  seg `Config`（`config.py:24-29,671-673`），派生链**完全同源**，无第二处重算。
- **encoder/decoder 字面同构**：`build_ssl_recon_model` 直接调用**同一** `build_model(cfg)`
  再取 `.encoder`/`.decoder`（`ssl_models.py:84-89`）。故 yaml 中所有高级项
  （`multirf_*`、`selfattn_*`、`attention_type=se`、`stem_mode=dual`、`block_type=basic`、
  `norm_type=batch` 等）**全部被 SSL 继承**，与下游逐参数同名同形。
- **头隔离正确**：重建头命名 `recon_head`（`SegmentationHead`，out=in_channels=12），
  与下游 `seg_head` 不撞名。前向 `encoder→decoder→recon_head→（stem_stride>1 时）
  `_resize_logits``（`ssl_models.py:52-63`），镜像 seg UNet 主头路径；DS/aux 头在 SSL
  被丢弃（yaml 里 `deep_supervision/aux_seg_supervision=true` 仅为保持几何一致，SSL 不用）。
- **尺寸校验**：输出与输入 spatial 不符时显式 `RuntimeError`（`:58-62`），不静默 resize。

### 问题与优化

**[低] 2.1 `arch` 硬约束 `unet`。** `validate_ssl` 与 `build_ssl_recon_model` 均要求
`arch=='unet'`（`config.py:568-570`、`ssl_models.py:81`）。本 yaml 满足；若未来下游换
backbone（如 mednext）则 SSL 需解耦。属能力边界，非本 yaml 问题。

**[低] 2.2 迁移未附 backbone 指纹。** 若下游 config 的 `encoder_channels/stem/norm` 与
预训练不一致，`strict=False` 会静默丢 key。建议 ckpt 内存一份 backbone 关键超参，下游
加载时比对并对命中率过低告警（见 §5.2）。

---

## 3. 增强 / 处理 + vesselness 目标（Domain 3）

**文件**：`ssltask/trainer/ssl_trainer.py`（增强调用）、`ssltask/data/vesselness.py`、`corruptions.py` ↔ `segtask_v1/data/augment.py`

### 1:1 对照

- **通用增强复用 seg**：trainer 用 **同一** `GPUAugmentor(cfg.augment, max_scale=1.0)`
  （`ssl_trainer.py:146`），本 yaml 只开保守空间增强（flip 0.2 / affine 0.3、禁平移、
  强度扰动全关）。prior 的 `clean=batch["image"]` 取的是**增强后**图，target=frangi(clean)、
  input=clean 同源（`prior.py:51-52`）→ 空间增强对输入与目标一致施加，语义自洽。
- **2.5D 折叠 1:1**：trainer `_fold_batch` 用 `reshape(b, c*d, h, w)`（c=1）→ `(b,12,h,w)`
  （`ssl_trainer.py:383-384`），与 seg `views.squeeze_2_5d` 的 `rearrange('b c d h w ->
  b (c d) h w')`（c=1）**数学一致**（`views.py:250`）。增强在 3D 体上做、折叠推迟到增强后
  送模型前，与 seg 送模型前口径一致。

### vesselness 目标审查（`vesselness.py`，纯 torch 多尺度 Frangi）

- **实现正确且用心**：可分离高斯（`_separable_gaussian`）+ 中心差分 Hessian
  （`torch.gradient`，spacing-aware）+ **闭式特征值**（`_eigvalsh_2x2/3x3`，`:69-94`）；
  刻意避开 `torch.linalg.eigvalsh`（CUDA cuSOLVER 对百万级 2×2/3×3 小矩阵会
  `CUSOLVER_STATUS_INVALID_VALUE` 且无 bf16 kernel）—— 这是**正确且必要**的选择。
- **2.5D 语义正确**：`spatial_dims=2`、按通道 depthwise 处理，即对 12 个 z 切片各算独立
  2D vesselness，输出 `(B,12,H,W)` 与 recon 头输出同形。
- **γ 归一化 `×σ²`**（`:196`）、**逐样本逐通道 [0,1] 归一**（`_amax_spatial`，`:201-203`）、
  `black_vessels` 亮/暗符号处理（`:119,133,143`）、`nan_to_num` 前处理（`:187`）均正确。
- **各向异性支持**：`spacing` 非空时 `scales` 解释为物理尺度、逐轴体素 sigma
  `sigma/spacing[axis]`（`:192`），对薄层/各向异性 CT 更准。

### 问题与优化

**[低] 3.1 `prior_spacing` 推荐但本 yaml 未启用。** yaml 注释建议填 `[sz,sy,sx]`（`:203`）
但当前留空 → Frangi 走体素单位各向同性。若数据为各向异性 CT，管径尺度物理不一致，
建议填入数据 spacing（`validate_ssl` 已校验长度==spatial_dims=2）。属配置建议。

**[低] 3.2 `aug_oversample_ratio=1.0` 与 `affine_prob=0.3` 并存。** 无 z 余量下 affine
旋转/缩放在 H/W 面会引入 edge-replicate 边界。对 prior 影响小（目标 post-aug 一致计算），
但若切到 genesis（重建原图）边界会成为目标。注意：seg 的 oversample 仅补 z 余量、不解
H/W affine 边界，故此非 seg 分歧，仅提示可评估调低 affine 或接受该边界。

---

## 4. 训练全流程（Domain 4，重点）

**文件**：`ssltask/trainer/ssl_trainer.py` ↔ `segtask_v1/trainer/trainer.py`

### seg 训练技巧的落地对照（"可借鉴"命题核验）

| seg 训练技巧 | SSL trainer 现状 | 判定 |
|---|---|---|
| 优化步时钟（warmup/total 按 opt-step） | `:86-104` 按 `ceil(len/accum)` | ✅ 等价 |
| one_cycle+warmup 冲突报错 | `:94-96` | ✅ 等价 |
| 梯度累积尾组归一 | `_effective_accum`（`:637-645`） | ✅ 等价 |
| **非有限 loss 守护 + 跳步** | `:723-749`（bf16 zero_grad 跳步、scheduler 照走、EMA 不推进；DDP `all_reduce_flag_any` 统一） | ✅ **已修复**（旧报告 P0） |
| fp16 GradScaler 跳步识别 | scale 前后对比（`:766-771`） | ✅ 等价 |
| 异步 checkpoint | `AsyncCheckpointSaver`（`:158,415-419`） | ✅ 已实现 |
| 原子写 + 指纹 | `_atomic_save`+`_state_fingerprint`（`:401-435`） | ✅ 优于旧 |
| 全状态 resume | method/opt/sched/scaler/EMA/RNG（`:456-540`），ZeRO consolidate | ✅ 已实现 |
| EMA / ZeRO / channels_last / compile | `:131,120-127,212-225` | ✅ 等价 |
| 监控面板 + 健康指标 | `MetricsLogger`、grad_norm/nonfinite/clip_frac（`:228-346`） | ✅ 已实现 |
| DDP epoch loss 加权 all-reduce | `_reduce_meter_avg`（`:663-674`） | ✅ 等价 |

### 合理分歧（非缺陷）

- **独立 trainer 实现**：SSL 不复用 seg `Trainer`/`pipelines`，而是重写循环（因 dino 系
  方法直调子模块、无单一 forward 入口，无法套 DDP wrapper，改用手动梯度 all-reduce，
  `:647-661`）。这是**方法无关框架的必要设计**。**代价（维护性）**：seg trainer 未来的
  改进需手动镜像到 SSL trainer（本次已见旧报告若干项被逐一补齐，说明此镜像成本真实存在）。
- **无 val loader**：SSL 以**在线分割探针**（`SegProbe`）驱动 best 选择，绕开"SSL 代理
  loss 与表征质量不单调"陷阱（`:557-598`）。探针 val 现已 `deterministic=True,
  fg_aware=True`（`probe.py:76`），失败被 try/except 兜住、末尾 `_best_saved` 兜底，鲁棒。

### 问题与优化

**[低] 4.1 每 micro-step 一次 `loss.item()`（残留同步点）。**
`:722` 每 micro-step `loss.item()` 用于非有限判定与 loss_meter —— 正是 seg 已削掉的那类
GPU↔CPU 同步点。seg 在**边界**用 grad 范数有限性判定、loss 延迟取回。SSL 可改为 GPU 张量
累加 + 边界处基于 grad 的有限性判定，消除逐步同步。收益随 batch 小、step 多而积累，属
低风险提速项。

**[低] 4.2 非有限守护是 loss-based，非 grad-based。**
bf16 路径以 `group_has_nonfinite`（loss 有限性，`:723-724`）决定跳步；seg 在边界直接检查
**梯度**有限性。若出现 loss 有限但反传梯度非有限的罕见情形，SSL bf16 路径不拦截。对本
yaml（prior=L1、数值稳）可忽略；若将来开对比/自蒸馏方法值得加 grad 有限性校验。

**[低] 4.3 recon 方法可选用 DDP wrapper 提速。** prior/genesis 是单一 `self.module(x)`
前向，理论上可套 `DistributedDataParallel` 获得反传-通信重叠（当前手动 all-reduce 无重叠）。
仅对 recon 族适用、需与 dino 族分流，属可选提速（收益视卡数/带宽）。

**[低] 4.4 探针 val 为 fg 感知随机位置，非网格覆盖。** 已比旧实现（纯随机）显著降噪；
若要进一步降 best 选择方差，可引入 seg 式 z 网格覆盖。低优先。

---

## 5. 推理 / 权重迁移到下游（Domain 5）

**文件**：`ssltask/methods/prior.export_backbone_state_dict`、`ssl_trainer._export_state_dict/_save` ↔ `segtask_v1/trainer/checkpoint.py`

### 1:1 / 契约对照

- **导出**：`_export_state_dict` **EMA 优先**（`apply_shadow`→导出→`restore`，`:390-399`），
  键与 `build_model` 同名；prior 导出 encoder+decoder+recon_head 全量、`.detach().cpu().clone()`
  （`prior.py:57-60`）。ckpt 含 `ssl_method` 标签 + 内容指纹。
- **下游加载兼容性（旧报告 §5.1 疑点 → 确认 OK）**：seg `extract_model_state_dict` 显式
  识别 `model_state_dict` 包装（`checkpoint.py:252-253`），并 `strip_common_prefixes` 剥
  `module.`/`_orig_mod.`。故 `train.pretrain=ssl_best.pt` 经 `strict=False`：`encoder.*`
  (+`decoder.*`) 命中、`recon_head.*` 作 unexpected 丢弃、`seg_head.*` 作 missing 保持随机。
  **契约闭环成立，不会静默命中 0 key。**

### 问题与优化

**[低] 5.1 建议下游打印命中/丢弃 key 统计**（若尚未），便于发现 backbone 配置漂移
导致的大面积静默丢权重。

**[低] 5.2 导出附 backbone 超参指纹**（见 §2.2），下游比对命中率过低时告警。

---

## 6. 可借鉴 / 优化清单（按收益/成本/风险排序，全部 opt-in、不破坏现状）

| 优先 | 项 | 关联域 | 说明 | 依赖 |
|---|---|---|---|---|
| **[中] P1** | 内容偏置拒绝采样 | 1.1 | 避免近空 patch 使 vesselness 目标退化；对 prior 有效监督密度提升最大 | 无 |
| **[低] P2** | 削 micro-step `loss.item()` 同步点 | 4.1 | GPU 累加 + 边界 grad 有限性判定，移植 seg 做法 | 无 |
| **[低] P2** | `prior_spacing` 填数据 spacing | 3.1 | 各向异性 CT 下 Frangi 物理尺度一致 | 无（校验已存在） |
| **[低] P3** | grad-based 非有限守护 | 4.2 | 补 loss 有限/grad 非有限的罕见情形（换方法时更相关） | 无 |
| **[低] P3** | recon 族改用 DDP wrapper | 4.3 | 反传-通信重叠提速；需与 dino 族分流 | 无 |
| **[低] P3** | 探针 val z 网格覆盖 | 4.4 | 进一步降 best 选择方差 | 无 |
| **[低] P3** | 下游加载命中率告警 + backbone 指纹 | 2.2/5 | 防配置漂移静默丢权重 | 无 |

> 说明：本 yaml 下 SSL 与 seg 2.5D 单 FOV 的**正确性与几何一致性已成立**，工程健壮性
> （非有限守护/异步保存/resume/监控）也已补齐至与 seg 对齐。真正剩余的实质项集中在
> **数据侧内容偏置采样（P1）**，其余为低风险微优化。

---

## 7. 结论

- **正确性**：五域整体正确。几何 1:1、enc/dec 字面同构、2.5D 折叠一致、Frangi 目标
  实现正确、迁移契约闭环、非有限守护已覆盖 bf16。**未发现影响正确性的缺陷。**
- **"1:1 一致"命题**：**成立**。分歧仅为 SSL 语义/工程必要项（无标签→均匀采样、无 val
  loader→在线探针、独立 trainer 实现）。
- **"技巧可借鉴"命题**：**大部分已落地**。seg 的优化步时钟、非有限跳步、异步保存、
  resume、监控、DDP 汇总均已等价实现。
- **优先建议**：仅一项中优先——**数据侧内容偏置采样（§1.1）**；其余为低风险微优化。
- **附带发现**：既有 `ssltask_investigation_report.md` 对本 yaml 相关项已**显著过期**
  （其 P0/多数 P1 均已修复），建议后续以本报告为准或据此更新旧报告。

**下一步（待你确认再进入编码阶段）**：若要落地，建议先做 §6 的 P1（内容偏置采样，
opt-in、零依赖、对 prior 收益最高）。确认后我再出该项的分步实施计划与回归测试设计。
