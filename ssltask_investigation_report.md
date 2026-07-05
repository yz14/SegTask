# SSL 预训练模块（ssltask）现状调研报告 — Phase 1（只调研，不动代码）

> 依据 TODO 第 2 项：按「数据读取 / 模型构建 / 数据增强·处理 / 训练全流程(含 val) / 推理全流程」五部分，
> 审查代码·算法·设计·架构的正确性、合理性、优化空间，以及可借鉴/适配/新增的高质量内容。
> 本轮**不修改任何代码/文档**。所有结论均基于对 main 分支 ssltask 源码的逐文件精读。

---

## 0. 总体架构评价

ssltask 是构建在 `segtask_v1` 之上的**方法无关（method-agnostic）SSL 预训练模块**，覆盖 11 种方法：
`genesis / prior / simmim / spark / dino / dino_gram / ibot / jepa / byol / moco / sparkdino`。

架构分层非常清晰，是本模块最大的优点：

- **data 层**只负责「读 patch」与「构造视图/破坏」（`ssl_dataset` / `corruptions` / `masking` / `multicrop` / `vesselness`），与目标无关；
- **method 层**（`methods/*.py`，统一继承 `SSLMethod` 抽象基类）只负责「用一个 batch 算 loss」与「导出骨干权重」；
- **model 层**（`models/*.py`）复用 `segtask_v1.models.factory.build_model` 的 encoder/decoder，保证与下游**逐参数同名同形**；
- **trainer 层**（`ssl_trainer.py`）是唯一的、方法无关的训练循环，复用 segtask 的 optimizer/scheduler/WarmupScheduler/AMP/EMA；
- **eval 层**（`eval/probe.py`）用在线分割线性探针驱动 best 选择，绕开「SSL 代理损失与下游表征质量不单调」的经典陷阱。

这套「破坏/视图在 data、目标在 method、骨干复用 factory、训练循环唯一」的正交分工，**质量高于多数开源 SSL 代码库**，SSL→下游的交接（`recon_head`/`seg_head` 命名隔离 + `strict=False`）设计得很干净。下面逐域给出发现的问题与优化点，按优先级标注。

---

## 1. 数据读取（Domain 1）

**文件**：`ssltask/data/ssl_dataset.py`

### 现状（正确/合理之处）
- `ImageOnlyPatchDataset` 只读 npz 的 `image` 键，不碰 label/fg_coords，语义上与「无标注大语料」目标一致，解耦干净。
- IO/预处理直接复用 `segtask_v1.data.dataset` 的 `_open_npz` / `preprocess_image` / `_extract_cubic_patch`，不重复造轮子。
- `_extract_cubic_patch` 越界 edge-pad，`_rand_center` 保证 patch 起点不越界；2D/2.5D 折叠口径与 segtask `squeeze_2_5d` 一致。
- `LabeledPatchDataset` 与 image-only 共享抽样逻辑，仅探针使用，不进主训练路径。
- `build_ssl_dataloader` 对 2.5D 做了 `in_channels == patch_size[0]` 的单-FOV 约束校验，错误信息清晰。

### 问题与优化空间

**[高] 1.1 每次 `__getitem__` 都全量读盘 + 全量预处理，无任何缓存。**
`_load_volume`(ssl_dataset.py:110-119) 每次都 `_open_npz` 读整卷 + `preprocess_image` 整卷归一化，然后只抠一个 patch。
- SSL 语料通常是「少量大卷 × 高 `samples_per_volume`」，同一卷会被反复整卷读+整卷预处理，浪费严重（npz 还带解压）。
- `segtask_v1.data.dataset` 里**已有** `cache_max_volumes` 的 LRU 卷缓存机制，但 SSL 数据集**没有复用**。
- 建议（Phase 2）：给 `ImageOnlyPatchDataset` 加一层 worker 内 LRU 卷缓存（缓存**预处理后**的 fp32 卷，按 `cache_max_volumes` 上限），可直接移植 segtask 的实现思路。预计对 IO-bound 的 SSL 预训练吞吐提升明显。

**[中] 1.2 均匀随机采样，未做「内容/前景」偏置。**
`_rand_center` 在整卷内均匀取中心。医学 CT 大量体素是空气/背景，均匀采样会产出大比例「近空」patch。
- 对重建/掩码类方法（genesis/simmim/spark/jepa）而言，空 patch 提供的自监督信号很弱，甚至可能让模型学到「输出常数」的捷径。
- 建议：加一个廉价的 **content-based rejection sampling**（如 patch 的强度方差 / 非背景体素占比低于阈值则重采样，最多重试 K 次）。这是 Models Genesis / nnU-Net 预处理里的常见做法，opt-in、零依赖。

**[低] 1.3 无 patch 级 label-free 质量过滤 / 无多卷分辨率归一。** 目前假设所有 npz 已按同一 spacing 预处理；若语料 spacing 混杂，SSL patch 的物理尺度不一致。属数据准备约定，可在文档中显式声明。

**[低] 1.4 `discover_image_npz` 每次训练启动都递归 glob 全目录并排序。** 大语料下可缓存清单；非瓶颈。

---

## 2. 模型构建（Domain 2）

**文件**：`ssltask/models/ssl_models.py`（recon/MIM）、`dino_modules.py`、`spark_modules.py`、`ibot_modules.py`、`jepa_modules.py`；`methods/*.py`

### 现状（正确/合理之处）
- `SSLReconModel`（genesis/prior）复用 factory 的 encoder+decoder，重建头**独立命名** `recon_head`（用 `SegmentationHead`），与下游 `seg_head` 不撞名——这是 SSL→下游 `strict=False` 干净交接的核心，设计正确。
- `SSLMIMModel`（simmim）只取 encoder + 轻量 `LightPixelHead`（1×1 proj → 插值 → 3×3 refine → 1×1），刻意与 SparK 层次化解码器形成「预测头轻重」对照变量，思路清晰。
- DINO 族用 `_DINOModule`（student/teacher/center buffer）容器，teacher 冻结 `requires_grad_(False)`；iBOT 在其上扩 dual head + mask-token + ibot_center，继承结构合理。
- 各方法 `build_modules()` 后 teacher/target 一律 `load_state_dict(student.state_dict())` 初始化对齐，正确。
- 前向都带 `out.shape[2:] != x.shape[2:]` 的显式尺寸校验并抛清晰 RuntimeError（ssl_models.py:52-56, 160-164），避免下采样/ stem_stride 不匹配时静默错。

### 问题与优化空间

**[中] 2.1 仅支持 `arch=='unet'`。** `build_ssl_recon_model` / `build_ssl_mim_model` 都硬性 `require arch=='unet'`。segtask_v1 若支持其它 backbone（如 MedNeXt 重参数化变体），SSL 无法预训练它们。属能力边界，Phase 2 若要扩展需在此解耦。

**[中] 2.2 导出粒度不统一（设计如此，但需在下游侧留意）。**
- genesis/prior 的 `SSLReconModel` 导出 **encoder+decoder+recon_head 全量**（genesis.py:39-42 直接 `module.state_dict()`），下游 seg 可迁移 decoder；
- DINO/iBOT 导出 **teacher.encoder** only（dino.py:164-170）；simmim/spark/jepa/byol/moco 导出 encoder（±decoder）。
这套差异是合理的（对比/自蒸馏方法本就没训练可迁移的 decoder），但意味着**下游能否复用 decoder 依方法而异**。建议在文档/config 里显式列一张「方法 → 可迁移子模块」对照表，避免用户误以为所有 SSL ckpt 都能迁移 decoder。

**[低] 2.3 `recon_head`/`mask_token`/`center` 等 SSL 专属参数依赖下游 `strict=False` 静默丢弃。** 目前正确（downstream 作为 unexpected key 丢弃），但一旦下游误用 `strict=True` 会炸。建议下游加载 SSL 权重时打印命中/丢弃 key 统计（可能 segtask 已做，需在 Phase 2 核实 `checkpoint.load` 的日志粒度）。

---

## 3. 数据增强 / 处理（Domain 3）

**文件**：`corruptions.py`（Genesis 4 类破坏）、`masking.py`（MIM 掩码工具）、`multicrop.py`（多裁剪）、`vesselness.py`（Frangi 先验目标）

### 现状（正确/合理之处）
- **masking.py** 质量很高：`sample_unit_mask` 用 `argsort(noise)` 的 MAE 式无偏采样，`num_mask` 夹到 `[1, num_units-1]` 避免全遮/全见退化；`per_unit_normalize` 用 `avg_pool(kernel=stride=unit)` 求单元均值/方差再最近邻广播，等价 MAE 的 norm-pix 目标；`masked_recon_loss` 只在被遮位点、按「被遮体素×通道」归一，口径正确。2D/3D 由张量维度自动推断，通用性好。
- **vesselness.py** 是亮点：纯 torch 实现多尺度 Frangi（可分离高斯 + 中心差分 Hessian + `eigvalsh` + γ 归一化 `×σ²`），label-free 地把「管状结构几何先验」灌进 encoder，`black_vessels` 亮/暗血管符号处理正确。这是很有针对性的领域创新。
- **corruptions.py** 忠实实现 Models Genesis 四类破坏（Bézier 非线性强度 / 局部像素打乱 / 内补 / 外补），Bézier 曲线用排序后 1D 插值保单调，`@no_grad` + 不原地改输入。
- **multicrop.py** 标准 DINO random-resized-crop + 轻量翻转/强度增广，`@no_grad`，同类裁剪尺寸固定便于 batch 内对齐。

### 问题与优化空间

**[中] 3.1 corruptions / multicrop 都是「Python for b in range(B)」逐样本循环。**
`GenesisCorruptor.__call__`(corruptions.py:146-157) 与 `MultiCropGenerator._make_crops`(multicrop.py:105-114) 都对 batch 内每个样本、每个 crop 单独跑一遍（含 `.tolist()` / `float(rng)` 等隐式同步）。
- B 很小（2~4）时可接受；但 DINO 的 n_local=6、n_global=2 意味着每步 8×B 次独立 interpolate。
- 建议（Phase 2，低风险）：multicrop 的同类裁剪可尝试「同一 batch 共享一次 grid_sample」的向量化，或至少把 per-sample 随机盒子堆叠后一次 `grid_sample`。收益视 batch/crop 数而定。

**[中] 3.2 重建/掩码方法的增强偏弱。** genesis 之外，simmim/spark/jepa 的输入几乎只有「掩码」这一种扰动，没有强度抖动/高斯噪声/模糊等。
- SOTA MIM（如 SimMIM/MAE 医学改版）通常仍叠加轻量强度增广提升不变性。
- 建议：把 `multicrop._augment` 里的强度 scale/shift（甚至加 gamma / 高斯噪声）抽成一个共享的轻量强度增广，供掩码方法可选启用。opt-in。

**[低] 3.3 `_paint` 外补全用「样本值域内均匀噪声」填充。** 对已归一化到 [0,1]/z-score 的输入，均匀噪声的分布与真实解剖背景差异较大；Models Genesis 原文亦如此，可接受，属可调项。

**[低] 3.4 multicrop 强度增广是**全局**scale/shift**，未做逐通道/局部。** 对 2.5D（通道=切片）可能希望逐切片扰动。属细化项。

---

## 4. 训练全流程（含 val / 在线探针）（Domain 4）

**文件**：`ssltask/trainer/ssl_trainer.py`、`ssltask/eval/probe.py`、`ssltask/pretrain.py`

### 现状（正确/合理之处）
- 复用 segtask 的 `build_optimizer/build_scheduler/WarmupScheduler/ModelEMA/GradScaler`，`one_cycle + warmup>0` 冲突显式报错，AMP dtype `auto` 解析、fp16 才启用 scaler——这些都与 segtask 一致且正确。
- **grad accumulation 边界处理正确**：`is_boundary` 判定含末步补齐，只有边界步才 `unscale_→clip→step→update→zero_grad→scheduler.step→ema.update→on_after_step`。
- `configure_schedule(total_opt_steps)`（按 `ceil(steps/accum)` 计算）让 DINO/JEPA/BYOL/MoCo 预计算 teacher 温度/EMA 动量的 cosine 调度，方法无关地下发总步数，设计优雅。
- **在线探针（§0.5）是最大亮点**：`SegProbe` 冻结/微调 encoder、只训多尺度 1×1 线性头固定步数，回报前景 Dice/HD95 作为**可比的表征质量信号**，并用它（而非 SSL 代理 loss）驱动 best 选择。探针评估用 `_save_rng_state/_restore_rng_state` 隔离随机态、固定 seed 重置头，保证跨 epoch 可比。探针失败被 `try/except` 兜住「绝不打断预训练」，且末尾有 `_best_saved` 兜底存 best——鲁棒性考虑周到。
- DINO teacher EMA、center EMA、teacher 温度 warmup 全部实现正确（dino.py:97-161），config 还会在「DINO/JEPA + trainer 级 use_ema」冗余时打 warning。

### 问题与优化空间

**[高] 4.1 非有限 loss 的防护在 bf16 路径上失效——可能污染权重。**
训练步顺序是：`compute_loss → scaler.scale(loss).backward()`（line 220）**先做了反传和 optimizer.step**（line 228），**之后**才在 line 238 检查 `math.isfinite(step_loss)`——而这个检查**只用于是否更新 loss_meter，并不阻止已经发生的 step**。
- fp16 路径：`GradScaler` 会在检测到 inf/nan 梯度时自动跳过 `step` + 缩放 scaler，尚有保护。
- **bf16 / 无 scaler 路径**：一旦 `compute_loss` 出 NaN/Inf（如对比方法数值不稳、探针无关），非有限梯度会**照常 `optimizer.step` 写进权重**，只在事后打一条 warning。
- 对照：segtask trainer（PR #1）已对跳步/非有限做了处理；**SSL trainer 是独立实现，未继承这一保护**。
- 建议（Phase 2）：边界步在 `step` 前检查 loss/grad 有限性，非有限则 `zero_grad` 跳过该 boundary（bf16 路径尤其需要）。

**[中] 4.2 每个 micro-step 都 `loss.item()`，制造 GPU↔CPU 同步点。**
line 237 `step_loss = loss.item()` 每步一次同步——正是 PR #1 在 segtask 里削掉的那类同步点，但 SSL trainer 没享受到。
- 建议：GPU 上张量累加 loss，仅在 `log_every` / epoch 末一次性取回。

**[中] 4.3 checkpoint 是阻塞 `torch.save`。**
`_save`(line 118-129) 同步写盘。PR #1 给 segtask 加了异步保存，SSL trainer 未复用。SSL 骨干可能不小、`save_every` 频繁时阻塞主循环。建议复用 segtask 的异步保存工具。

**[中] 4.4 无「预训练 resume」能力。**
`_save` 只存 `model_state_dict`（导出的骨干）+ epoch/best，**不存 optimizer/scheduler/scaler/EMA/RNG 状态**。SSL 预训练常跑很久，一旦中断无法断点续训，只能从头。建议加一个「完整训练态」checkpoint（可与「可迁移骨干导出」分离）以支持 resume。

**[中] 4.5 探针 val 是随机位置 patch，方差偏大。**
`build_probe_loaders` 的 val_loader 用 `LabeledPatchDataset`（`_rand_center` 随机位置），且 `probe_samples_per_volume//2`。作为「廉价可比信号」尚可（固定 seed + 每次重置头），但同一卷每次 val 抠的位置随机，Dice 有噪声。若探针 Dice 是 best 选择的唯一依据，建议 val 改「固定网格覆盖」降噪（与我在 segtask PR #1 对验证 patch 做的网格覆盖同理）。

**[低] 4.6 探针每次 `evaluate` 都从头训练线性头 `probe_iters` 步。** 设计使然（保证可比），但每 `probe_every` 个 epoch 都付出这份开销。可接受，属成本-可比性权衡。

**[低] 4.7 无 TensorBoard/结构化指标输出，仅 logger。** 长预训练缺少曲线可视化。非功能性缺陷。

---

## 5. 推理 / 权重迁移到下游（Domain 5）

**文件**：`methods/*.export_backbone_state_dict`、`ssl_trainer._export_state_dict/_save`、`pretrain.py`

### 现状（正确/合理之处）
- 迁移路径干净：`_export_state_dict`(ssl_trainer.py:108-116) **EMA 优先**（有 EMA 时先 `apply_shadow` 导出再 `restore`），键与 `build_model` 同名；下游只需 `train.pretrain=<ssl_best.pt>`，经已有 `strict=False` 加载：`encoder.*`(+`decoder.*`) 命中、`recon_head/mask_token/center` 作为 unexpected 丢弃、`seg_head` 作为 missing 保持随机。这是本模块的核心价值，实现正确。
- 导出一律 `.detach().cpu().clone()`，避免持有 GPU 引用/视图，正确。
- ckpt 内含 `ssl_method` 标签，便于下游溯源。

### 问题与优化空间

**[中] 5.1 下游加载器需能吃 `{"model_state_dict": ...}` 包装。**
`_save` 存的是 `{"epoch",  "model_state_dict", "ssl_method", ...}`。下游 `train.pretrain` 的加载逻辑必须知道从 `model_state_dict` 取权重（而非把整个 dict 当 state_dict）。**需在 Phase 2 核实** `segtask_v1` 的 pretrain 加载路径是否兼容这层包装（probe 侧 `_load_encoder` 用 `strip_common_prefixes`，说明前缀处理已有基础）。若不兼容会静默命中 0 key。

**[低] 5.2 导出未附带 backbone 配置指纹。** 若下游 config 的 `encoder_channels/stem/norm` 与预训练不一致，`strict=False` 会静默丢弃大量 key 而不报警。建议在 ckpt 里存一份 backbone 关键超参，下游加载时比对并对「命中率过低」告警。

---

## 6. 可借鉴 / 适配 / 新增的 SOTA 内容（供 Phase 2 选型）

按「收益/成本/风险」排序，全部 opt-in、不破坏现状：

| 优先 | 项 | 说明 | 依赖 |
|---|---|---|---|
| **P0** | 4.1 非有限 loss 防护（bf16 路径） | 直接影响训练正确性，低成本 | 无 |
| **P0** | 1.1 卷 LRU 缓存 | 移植 segtask 已有机制，显著提吞吐 | 无 |
| **P1** | 4.2/4.3 削同步点 + 异步保存 | 移植 PR #1 已验证的做法到 SSL trainer | 无 |
| **P1** | 1.2 内容偏置采样 | 医学 CT 避免空 patch，提升信号 | 无 |
| **P1** | 4.4 预训练 resume | 长训练必备 | 无 |
| **P2** | 3.2 掩码方法叠加轻量强度增广 | 提升 MIM 不变性 | 无 |
| **P2** | 4.5 探针 val 网格覆盖 | 降 best 选择噪声 | 无 |
| **P3** | 新方法：MAE(纯 encoder,drop token) / VICRegL / DINOv2 register tokens | 补齐 SOTA 谱系 | 无/小 |
| **P3** | 3.1 multicrop/corruption 向量化 | 吞吐优化 | 无 |

> 说明：模块**已覆盖**当前主流 SSL 谱系（重建 Genesis、领域先验 Frangi、MIM SimMIM/SparK、自蒸馏 DINO/iBOT、隐空间 JEPA、对比 BYOL/MoCo、DINOv3 Gram、混合 SparK+DINO），再「堆方法」边际收益有限。真正的短板集中在**工程健壮性与吞吐**（Domain 1/4），而非算法覆盖度。

---

## 7. 结论

- **正确性**：五域整体正确。唯一影响正确性的实质问题是 **4.1（bf16 路径非有限 loss 不阻止 optimizer.step）**，建议 Phase 2 优先修。
- **合理性/架构**：分层与正交分工质量很高，SSL→下游交接干净，在线探针设计是超出多数开源实现的亮点。
- **优化空间**：主要在**数据 IO 缓存（1.1）**、**训练同步点/异步保存/resume（4.2-4.4）**——其中数条正是 segtask 已在 PR #1 验证、但 SSL trainer 因独立实现而未继承的优化。
- **可新增**：算法谱系已很全，建议 Phase 2 聚焦「工程健壮性 + 吞吐」而非新方法；若要新增，MAE/VICRegL/DINOv2 register tokens 是低风险候选。

**下一步**：请确认希望我在 Phase 2 落地哪些项（建议至少 P0 两项：4.1 非有限 loss 防护 + 1.1 卷缓存）。确认后我再出分步实施计划并按项分 PR。
