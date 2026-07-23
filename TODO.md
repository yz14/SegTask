# 规则与要求

## 一、任务启动(两阶段,强制)
新任务必须**先调研规划、后编码执行**,二者分属不同轮次,不得在同一轮合并完成。

- **第一轮(调研规划,禁止写实现代码)**:阅读相关现有代码与上下文,检索业界做法;明确目标、范围边界、约束与难点;产出一份**可拆分、各步骤可独立执行**的计划,每步含目标、预期产出、验收标准及依赖关系。本轮只输出结论与计划,等我确认后再动手。
- **第二轮起(执行)**:计划确认后才开始编码,严格按步推进,每轮专注一步并保证质量。计划需调整时先说明原因,不擅自偏离。

## 二、质量第一
- 充分思考分析后再动手,宁可慢不可糙。
- **单轮质量优先于单轮完整性**:宁可这轮少做做透,不贪多做糙。

## 三、范围克制
- 不擅自修改任务范围外的代码、不做无关重构。确需引入新的第三方依赖时,先说明用途和必要性再使用(避免为小功能引入重型库或重复造轮子)。

## 四、代码质量
- 模块化设计,职责分离,不把所有代码堆在一个文件。
- 复用优先,避免重复代码。
- 涉及具体库/API 时以最新官方文档为准,不凭记忆臆断版本与用法。

## 五、调试支持
- 需要时加入日志辅助定位:正式日志走 logging 并保留;临时调试代码统一加 `# DEBUG` 标记,便于稳定后清理。

## 六、完成与自查
- 完成后自查:是否真正达成目标;改动是否破坏了原本正常的功能(尤其与改动处有依赖关系的地方);边界情况是否处理。并说明如何验证。

## 七、沟通规范
- **开始前**:复述你理解的目标与将遵守的规则。
- **进行中**:如需拆分,明确告知本轮要完成什么。
- **完成后**:总结本轮成果与后续计划。


本地测试环境为: **D:\miniconda\envs\torch27_env\python.exe**。  


内容（**注意**：需查看各自对应readme/design/workflow文档理解设计）：  
segtask_v1是2.5D/3D分割项目（项目起源，最完善）。  
ssltask是自监督学习项目（基于segtask_v1改造），是对分割，生成，分类，检测进行预训练。  
clstask是分类项目（基于segtask_v1改造）。  
dettask是检测项目（基于segtask_v1改造）。  
gentask是生成/超分项目（基于segtask_v1改造）。  


# TODO  
1 公共框架代码深度审查（需结合所有 readme/design/workflow 一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：

审查主要内容为代码、算法、设计、架构、工程等等：
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。现在是2026年7月，不局限医学图像领域，可能自然图像的分类/分割/检测/生成等、NLP、LLM、VLM等有更好、更先进的想法。

进展：

### Step A · `taskcore/config/` 配置层审查（2026-07-22）

审查范围：`core.py`(2593 行)/`registry.py`/`task_io.py`/`model_migration.py`/`__init__.py`；交叉核对 `gentask/config/io.py`、`clstask|dettask|ssltask/config.py` 的接入方式。

**总评**：配置层是全仓最成熟的一层。dataclass+YAML+集中式 `validate()` 设计清晰、注释详尽、错误信息可读、迁移契约有测试（`test_d2_migration_contract.py` / `test_task_config_io.py`）覆盖。`_require`→`ConfigError(AssertionError, ValueError)` 的设计巧妙解决了 `python -O` 剥除 `assert` 的隐患，同时向后兼容既有捕获 `AssertionError` 的调用方。未发现阻断性正确性 bug。以下为分级结论。

**（1）设计亮点（正确、值得保持）**
- `ConfigError` 双继承 + `_require`：`core.py:18-31`，配置层已全面弃用裸 `assert`。
- 几何派生量单一真相源：`spatial_dims`/`in_channels` 由 `sync()`→`build_topology` 写入私有 backing、只读 property 暴露（`core.py:605-619, 1336-1341`），并硬拒绝在 YAML 里手设（`_DEPRECATED_DERIVED_KEYS`, `core.py:2446-2455`），杜绝"设了却被静默重写"。`save_best_metric/mode` 同法由 `save_best_criterion` 派生（`core.py:967-978`）。
- selfattn 的 softmax O(N²) token 护栏：`_est_stage_tokens` + cap（`core.py:1645-1672, 1742-1761`），建块前就给清晰 OOM 预警，工程性好。
- D2 迁移契约集中在 `model_migration.py`：扁平↔嵌套双向映射 + 新旧同设 fail-fast（`route_legacy_model_dict`, `model_migration.py:214-257`），无静默优先级。
- 跨段一致性预警到位：`prefetch_to_gpu` 需 `pin_memory`（`core.py:2264-2269`）、`zscore` 下增强绝对幅值提示（`core.py:1906-1926`）、平移边缘复制带超出 oversample 余量提示（`core.py:1859-1875`）等。

**（2）一致性欠账 / 可优化（非阻断，建议 TODO 3 收敛）**
- **[中] gen fork 重复 `_dataclass_from_dict`**：`gentask/config/io.py:69-142` 与 `core.py:2482-2553` 近乎逐字重复（含 `_FIELD_ALIASES`/`_DEPRECATED_DERIVED_KEYS`/`_REMOVED_KEYS`/`load_config`/`save_config`）。**且两份已经出现行为分叉**：core 的 `_dataclass_from_dict` 不用 `_SUB_CONFIGS` 兜底（该常量仅被 `segtask_v1/launcher/schema.py:30` 外部消费），gen 版加了 `if sub_cls is None and k in _SUB_CONFIGS` 兜底（`io.py:114-115`）。这正是"双轨 fork"的实际维护成本：core 若新增一条 `_REMOVED_KEYS`，gen 不会自动获得。建议把 core `_dataclass_from_dict` 泛化出 `extra_flat_to_nested` 与 `sub_configs` 两个参数后供 gen 直接复用（可消 ~100 行重复，属低风险高价值重构）。
- **[中] P2a 债：`loss`/`predict` 段仍以 seg 形常驻 core `Config`**：组合式任务（cls/det/ssl）靠 `skip_core_validators=("loss","predict")`（`registry.py:30, 84-88`）跳过校验绕开，但这两段仍随 core 序列化/存在于对象上，属"存在但对该任务无意义"的状态。README 已承认为 P2a 后续。建议方向：把 seg 专属的 loss/predict 下沉为 seg 的任务段（与 cls/det/ssl 对称），core 只保留真正五任务共享的字段。
- **[低] `coerce_override_value` 类型分支不一致**：~~`old is None` 走 `yaml.safe_load`、`list` 走 `json.loads`~~ → **✅ 已修（2026-07-22）**：list 统一 `yaml.safe_load`。
- **[低] `apply_dotted_overrides` 静默忽略非法项**：~~无 `=` 的 override 被 `continue` 直接吞掉~~ → **✅ 已修（2026-07-22）**：非空且无 `=` 项发 warning。
- **[低] `coerce_override_value` bool 静默 False**：~~非真值串一律判 False~~ → **✅ 已修（2026-07-22）**：非法 bool 串抛 `ValueError`。

**（3）2026 SOTA 对标（借思路，不建议迁移）**
- 当前手写 override（`coerce_override_value`/`set_dotted_attr`）本质是 OmegaConf dotlist 的迷你实现；集中式 `validate()` 是 pydantic v2 校验的手写版。考虑到"范围克制"、现有系统成熟且测试完备、迁移会改变错误语义与依赖重量，**不建议迁移到 Hydra/OmegaConf 或 pydantic**。仅记录：若未来需要配置组合(config groups)/多组网格实验(multirun sweep)，OmegaConf structured configs + `from_dotlist` 是业界标准迁移目标；届时可先只替换 override 层（低风险切入点）。

**结论**：配置层质量高，无需在 TODO 1 内改动。真正值得做的两项（gen fork 去重、loss/predict 下沉）本质是 TODO 3「抽离通用框架」的一部分，留待 TODO 3 统一处理；本轮仅记录证据与方向。下一步 Step B（`models/` 模型层）。

### Step B · `taskcore/models/` 模型层审查（2026-07-22）

审查范围：`topology.py`/`factory.py`/`blocks.py`(1502 行)/`stem.py`/`unet.py`/`unetpp.py`/`unet3p.py`/`resnet.py`/`convnext.py`/`mednext.py`/`adm_unet.py`/`edm2_unet.py`/`arch_compat.py`。

**总评**：models 层是全仓工程密度最高的一层，质量极高，**无阻断性 bug**。模块化彻底（block→stage→encoder/decoder→factory 分层清晰），2D/3D 维度无关（`_CONV[d]` 分派），AMP/显存/torch.compile 细节处理到位。架构覆盖 2023-2024 SOTA：ConvNeXt-V2(GRN)、MedNeXt+UniRepLKNet 可重参数化空洞大核、VAN-LKA、SegNeXt-MSCA、CARAFE、DySample、BlurPool、RoPE、window/grid(MaxViT 式) 与线性注意力、ADM、EDM2(magnitude-preserving)。

**（1）设计亮点（正确、值得保持）**
- 几何派生单一真相源 `build_topology`（`topology.py`）：`patch_mode × 5 mode flags → 全部通道/输出几何`，config.sync 与 factory 都读它，消除"多处各算一遍"的漂移隐患。
- `factory._StatefulStageBuilder`（`factory.py:44-63`）：用单一计数器传 `stage_idx`，杜绝 factory 闭包另设计数器导致的 drop_path/mask 错位。
- 各向异性下采样的**构建期 fail-fast 兼容性护栏**（`factory.py:418-449`）：与 ConvNeXt LN-first 下采样/非 unet decoder/hierarchical stem/非兼容 up-down 模式冲突时构建期即报错，而非 forward 期尺寸错。
- `blocks.py` 工程细节：AMP 安全（GRN/LayerNorm3d/ADM GroupNorm32 均 fp32 归约、fp16/bf16 插值先转 fp32）；`checkpoint_if` 用 `use_reentrant=False + preserve_rng_state=True`（正确处理 DropPath 复现与 eval 零开销）；RoPE cos/sin 有界 LRU 且 `torch.compiler.is_compiling()` 时旁路缓存避免 graph break；GroupNorm 组数不整除自动折半回退并去重告警；SelfAttentionBlock 走 `F.scaled_dot_product_attention`（可用 flash/mem-efficient 后端）+ 输出投影 zero-init（启用即近似恒等残差）。
- MedNeXt `DilatedReparamBlock`（`mednext.py`）：训练态多空洞分支 + BN、推理态 `switch_to_deploy` 折叠为单一大核 depthwise（零额外推理开销），并配 `upkern_remap_state_dict` 小核→大核插值迁移。
- ADM/EDM2 论文忠实：ADM `_GroupNorm32` fp32、`_zero_` 输出零初始化；EDM2 `_MPConv` 强制 weight-norm、`_mp_silu/_mp_sum/_mp_cat` 保幅原语、FiLM 条件——三 arch 的 `forward` 合同与 `UNet3D` 完全一致（`Tensor` / `[main, ds…]` / `{main, aux, topo}`），使 trainer/loss 侧无需感知 arch。
- `arch_compat.warn_ignored_model_fields`：arch='adm'/'edm2' 下被静默忽略且非默认的通用 UNet 旋钮统一汇总告警，提升可发现性。

**（2）可优化 / 一致性（非阻断）**
- **[低·可加速] 扩散侧注意力未走 SDPA**：ADM `_QKVAttentionLegacy`（`adm_unet.py:196-215`）与 EDM2 `_Block` 自注意力（`edm2_unet.py:173-189`）用显式 einsum + softmax(fp32) 的 O(N²) 实现，而 `blocks.SelfAttentionBlock` 已用 `F.scaled_dot_product_attention`。扩散注意力多在 bottleneck 小分辨率，收益有限，但改用 SDPA 可拿 flash/mem-efficient 后端的显存与速度收益（数值近似等价）。属可选优化。
- **[低·一致性] ADM `_LinearAttentionBlock` 的 `to_out` 非 zero-init**（`adm_unet.py:228-232`，注释明示 "non zero-init"）：启用 `adm_linear_attention_levels` 会在 init 即扰动论文忠实基线；而 `_AttentionBlock`/`SelfAttentionBlock` 的输出投影都 zero-init。建议给 linear-attn 也加 zero-init 选项，与"启用附加模块即近似恒等残差"的全局范式对齐。
- **[需 GPU 验证·非 bug] `_MPConv` 训练期 in-place 参数变异**：`edm2_unet.py:80-82`。**✅ GPU 已验证（2026-07-22）**：RTX 3080 Ti + bf16 eager 3 步前反向无异常；`torch.compile` 因 Windows 无 Triton 与 `BaseTrainer` 一致回退 eager，非代码欠账（见下方 GPU 验证节）。
- **[跨层契约·待 Step C 核对] cond 通道打包顺序**：`Encoder.forward`（`unet.py:186-188`）与 ADM/EDM2 encoder 均按"输入张量末 `cond_in_channels` 个通道 = 条件图"做 `torch.split`。这是与数据层的隐式契约，需在 Step C 核对 gen dataset 是否确实把 cond 拼在末尾（不一致会静默学错）。

**（3）2026 SOTA 对标（已联网核实）**
- **nnU-Net Revisited（Isensee et al., 2024, arXiv:2404.09556）** 严格基准结论：3D 医学分割 SOTA 配方 = ①CNN-based U-Net（含 **ResNet 与 ConvNeXt 变体**）②nnU-Net 框架 ③按现代硬件缩放模型；大量 Transformer/Mamba 的"超越"在修正基线/数据/算力口径后不成立。本仓 models 层的 **ResEnc UNet + ConvNeXt + MedNeXt** 正是该配方核心 → **架构选型已对齐当前最强基线，无追新必要**。
- **唯一明显缺席的现代族：Mamba/SSM**（U-Mamba / SegMamba / UlikeMamba / MambaClinix 等）。据 2024-2025 多篇严格基准（arXiv:2503.01306、2503.19308），Mamba 相对 nnU-Net **至多相当、参数更少但训练显著更慢**，差距常被高估；且 selective-scan CUDA kernel 在 Windows 构建有摩擦。→ 结论：作为"**可选、待评估**"的 backbone 候选记录在案，**不属当前欠账，不建议本轮/近期投入**。若要做，最稳妥形态是 MambaClinix 式"浅层卷积 + 深层 residual Mamba 块"的混合 stage，可挂到现有 `stage_builder` 机制上（与 ConvNeXt/MedNeXt 同构接入）。
- **扩散侧**：ADM(2021)+EDM2(2024) 已覆盖主线。若生成任务后续要提采样质量，flow matching / rectified flow 属**训练目标层**改动（不改 backbone，现有 ADM/EDM2 backbone 可直接复用），留待生成项目（TODO 2 关联）评估。

**结论**：models 层无需在 TODO 1 内改动，且架构先进性已对齐 2026 最强实证基线。可选微优化（扩散注意力 SDPA 化、linear-attn zero-init 一致性）价值中低，可纳入 TODO 3 或按需处理；Mamba 混合 backbone 属可选研究项。下一步 Step C（`data/` 数据层）。

### Step C · `taskcore/data/` 数据层审查（2026-07-22）

审查范围：`dataset.py`(1342 行)/`loader.py`(1112 行)/`augment.py`(637 行)/`sampling.py`/`specs.py`/`patch_ops.py`/`patch_extract.py`/`patch_dataset_base.py`/`mixed_sampler.py`/`make_data.py`；交叉核对 gentask/data 与 gen cond 契约。无阻断性 bug；只记问题/优化点。

- **[中·已修] `SourceTaggedDataset.__getattr__` 抛 KeyError 而非 AttributeError**（`mixed_sampler.py`）：**✅ 已修（2026-07-22）**；dunder / `base` 未初始化时显式 `AttributeError`。**✅ GPU 已验证**：`num_workers=2` DataLoader 在 Windows spawn 下正常取 batch。
- **[中·已修] make_data 幂等 skip 无版本/参数校验**（`make_data.py`）：**✅ 已修（2026-07-22）**；skip 前读 meta 比对 `spacing_normalized`/`target_spacing`/必需键，漂移则告警并重生成。
- **[低] fg 抑采与 fg_slices 口径不完全对齐**（`make_data._compute_fg_indices`）：`fg_coords` 逐类 cap 到 fg_subsample（大器官被抑采），`fg_slices`/`fg_slices_cls_z` 不抑采；所有卷共用固定 seed=42。影响极小（cap 内均匀随机），仅记录以防未来把 fg_coords 当"全量前景"使用。
- **[低] `IndexScheme.BLOCKED` 的 LRU 收益仅在顺序遍历成立**（`patch_dataset_base.py:26-33`）：训练 DataLoader shuffle 后 blocked 布局对缓存命中无效（收益只剩验证/非 shuffle 路径）。若 cls 训练侧 LRU miss 成为瓶颈，可考虑"按 worker 分卷 + 卷内连续"的采样器；当前非欠账。
- **[低·契约备忘] `_axis_center_range` 在 patch>size 时退化为单点 (mid, mid+1)**（`patch_ops.py:81-89`）：配合 edge-pad 行为正确，但"合法中心区间"语义退化为强制中心，调用方若用区间宽度推采样多样性会得到 1。
- **（闭环）Step B 遗留的 gen cond 通道顺序契约已核对一致**：gen 侧 `torch.cat([lr, cond], dim=1)`（`gentask/models/generation.py:84,206`、`diffusion.py:68`）都把 cond 拼在**末**通道，与 taskcore Encoder 按"末 cond_in_channels 通道=条件图" split 的约定一致，无静默错位。
- 既有已知项不再重复：augment 合流（伴随张量 spec 化）、data 侧残留 `assert`（python -O 失效）、gen spacing 归一化/逐类前景索引未接——均已在 TODO 3 进展区记录。

### Step D · `taskcore/engine/` 训练/推理层审查（2026-07-22）

审查范围：`base_trainer.py`(1030 行)/`base_predictor.py`/`optim.py`/`checkpoint.py`/`amp.py`/`prefetch.py`/`bn_stats.py`/`dist_utils.py`/`launch.py`/`memory.py`/`views.py`。无阻断性 bug；只记问题/优化点。

- **[中·性能] bf16/fp32 非有限守护每个优化步强制 host 同步**：无 scaler 时 `_boundary_grad_norm` 无条件走 `_global_grad_norm()`（`base_trainer.py:297-298`），其末尾 `.item()`（`base_trainer.py:477-478`）+ `all_reduce_flag_any` 的 `t.item()`（`dist_utils.py:85`）= 每边界至少两次 GPU→CPU 同步。fp16 有 `grad_norm_lazy_sync` 旁路，bf16 没有对应开关——而 bf16 恰是 `amp_dtype=auto` 在 Ampere+ 的默认。可选优化：给 bf16 提供延迟/抽样检查模式（每 N 步查一次，或 GradScaler 式异步 found_inf 张量驱动跳步）；注意跳步决策必须全 rank 一致，异步化需谨慎设计。属可优化非 bug。
- **[低] plateau 调度依赖子类记得调 `step_epoch(metric)`**（`optim.py:232-237`）：`WarmupScheduler.step` 对 plateau 是 no-op，某任务 trainer 若忘调 `step_epoch`，plateau 静默不生效且无告警。建议 WarmupScheduler 记录调用情况，scheduler 为 plateau 且多个 epoch 未见 `step_epoch` 时告警一次。
- **[低] one_cycle + fp16 scaler 跳步时钟欠步**：默认"仅真正更新才推 scheduler"（`base_trainer.py:408-413`）对 OneCycleLR 意味着早期 scaler 跳步使实际步数 < total_steps，退火尾部轻微截断。量级极小（跳步通常只在最初几十步），仅记录；SSL 的 `always_step_scheduler` 无此问题。
- **[低] `_restore_train_state` 无条件加载 scaler_state**（`base_trainer.py:619-623`）：resume 时若 amp_dtype 从 fp16 改 bf16（scaler enabled→disabled），加载 enabled 状态的行为随 torch 版本（忽略或告警）。建议显式检测 dtype 漂移并告警（与 WarmupScheduler 漂移告警同风格）。
- **[低] `AsyncCheckpointSaver.close` 异常路径**（`checkpoint.py:218-222`）：close→wait 若抛（最后一次写盘失败），`put(None)`/`join` 不执行，后台 daemon 线程残留；且 `_error` 只保留最后一个异常。影响小，记录即可。
- **[低] `CudaPrefetcher` 只搬 batch 顶层张量**（`prefetch.py:51-55`）：嵌套容器（如 det 的 list[Tensor] boxes）原样透传留在 CPU，不预取（行为正确但收益缺失）。属可选增强。
- **[低] `memory.estimate_train_memory` 的固有近似**（`memory.py:38-50`）：optimizer 状态恒按 fp32×2 估，未涵盖 8-bit optimizer 等未来选项；grad_bytes 未计 `gradient_as_bucket_view=False` 时的 bucket 双份。静态估计可接受，记录即可。

### Step E · `taskcore/metrics.py` + `monitor/` + `utils/` 审查与全局串联（2026-07-22）

审查范围：`metrics.py`(301 行)/`utils/common.py`/`utils/logging_utils.py`/`monitor/`(history/charts/dashboard/assets)。无阻断性 bug；只记问题/优化点。

- **[中·性能] spacing 感知 NSD 全程 CPU scipy 双重 EDT**（`metrics.py:192-257`）：逐样本逐类两次 `distance_transform_edt` + 全量 D2H 拷贝，大体积多类验证下会成为验证墙钟瓶颈。可选优化：① 只在 pred∪gt bbox（外扩 τ）内算 EDT；② 仅在 best/终验时启用 mm 版、日常 epoch 用 voxel-Chebyshev 版（现有 GPU 实现）；③ τ 小时用逐轴步进的 GPU max-pool 近似各向异性欧氏球。
- **[低·已修] `surface_dice_batch_stats` 用裸 assert 校验 rank**（`metrics.py`）：**✅ 已修（2026-07-22）** → `ValueError`。
- **[低] `compute_dice_per_class` batch 全空类返 0**（`metrics.py:40-42`）：该 0 进 mean 会拉低均值（nnU-Net 惯例是空类不计入）。调用方多用 pooled 版（`dice_batch_stats`）不受影响，仅提醒勿把该函数用于含"常空类"的选模路径。
- **[低] MetricsLogger 每 epoch 全量重写 jsonl+summary 且各 fsync 一次**（`history.py:341-353`）：数千 epoch 时 O(n) 重写；体量小无实际问题，若未来接 step 级记录需改追加+定期压实。
- **[低] `seed_everything` 顺带改全局 TF32/benchmark**（`utils/common.py:300-311`）：deterministic=False 分支强制开 benchmark/TF32，属"设种子"函数的隐藏副作用。行为合理，建议 TODO 3 抽框架时拆成显式 `configure_backends()`。
- **[低·微优化] ModelEMA CPU offload 每步整流 `synchronize()`**（`common.py:120-121`）：整流同步会等该流全部工作完成，比 `torch.cuda.Event` 只等 D2H 拷贝粗。收益取决于 update 时点流上负载，属可选微优化。

**全局串联核查（只列风险项）**：
- 五任务对 `_optimizer_step_boundary`/ack 协议、checkpoint 布局（best=EMA 为主、SSL 例外有文档）、`extract_model_state_dict` 契约一致；SSL 的 `always_step_scheduler` 分叉有运行时护栏（`_check_boundary_scheduler_clock`），无漂移风险。
- DDP 验证一致性依赖两条路径：可加量走 `all_reduce_sum_`/`all_reduce_meters_`，不可分解指标走 `all_gather_objects`——均要求各 rank 调用次数一致。`ValBatchShardSampler` 各 rank batch 数可不等长，但 reduce 只在 epoch 末发生一次，核对无死锁风险。**注意点**：未来若在 val batch 循环内加集体通信，须记住分片不等长会挂死。
- 已知项不再重复：TTA 键名不一致、gen config fork、SWA/AdaBN/UpKern/双源混采仅 seg、loss/predict 段常驻 core——均已在 TODO 记录。

**TODO 1 总结论（2026-07-22，审查 + 小修 + GPU 验证已闭环）**：

- **审查**：A–E 五层 + 全局串联已完成，无剩余盲区。
- **已修（本轮）**：① `SourceTaggedDataset.__getattr__` ② `make_data` skip meta 校验 ③ `task_io` override 三项（bool/list/warning）④ `metrics.surface_dice` assert→`ValueError`。
- **GPU 验证（本地 RTX 3080 Ti Laptop 17.2 GB，`tools/gpu_todo1_verify.py`）**：

| 项 | 结果 | 备注 |
|---|---|---|
| `SourceTaggedDataset` + `num_workers=2` | ✅ | Windows spawn worker 正常 |
| EDM2 bf16 eager（含 `_MPConv` in-place） | ✅ | 3 步前反向无异常 |
| EDM2/ADM + `torch.compile` | ⚠️ 跳过 | 环境无 Triton；`BaseTrainer` 已回退 eager |
| 扩散 backbone（ADM/EDM2）GPU 前反向 | ✅ | 小模型 peak ~35–40 MB |
| 接近生产 EDM2（3-view、128²、seg 通道宽度） | ✅ | peak ~693 MB |
| 显存摸底（EDM2 seg 通道宽，仅模型前反向） | ✅ | bs=4, patch 256² → ~2.1 GB；`seg2_5d_edm2.yaml` 配置在 16G 卡上模型侧余量充足 |

- **仍开放、不归 TODO 1（建议 TODO 3 或按需）**：bf16 非有限守护同步开销优化；NSD bbox 裁剪加速；gen fork 去重；loss/predict 下沉；扩散 SDPA；ADM linear-attn zero-init；plateau 未调 `step_epoch` 告警；data/model 侧残留 `assert` 等。
- **记录项、无需行动**：Mamba backbone、Hydra/pydantic 迁移、fg_slices 口径、IndexScheme.BLOCKED LRU、one_cycle 时钟欠步等。

**→ TODO 1 可结项。下一步建议 TODO 2（分割项目审查）。**


2 分割项目代码审查（需结合对应 readme/design/workflow 一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：

分割项目 = 公共框架层 `taskcore` + 任务层 `segtask_v1`，审查按此两级展开。代码大致分 5 部分，数据读取、模型构建、数据增强/处理、训练全流程（含 val）、推理全流程，先独立深度审查，再串联起来全局分析。每部分先审公共层、再审任务层。

审查主要内容为代码、算法、设计、架构、工程等等：
是否正确、合理；是否有优化空间；是否有训练加速/GPU优化空间；是否有更好的高质量内容（算法/模块/设计/架构/损失等等）可以借鉴、适配或新增。现在是2026年7月，不局限医学图像领域，可能自然图像的分类/分割/检测/生成等、NLP、LLM、VLM等有更好、更先进的想法。

进展：  



3 重构调研：由于cls/det/gen/ssl都是基于seg构建的，而且在设计上能和seg保持一致的都和seg保持一致了（可能还有不一致我未发现），能复用技巧也基本上都复用了（可能会有没有复用的我未发现）。现在我想将公用的内容抽离出来，形成一个通用的框架，然后在各个子项目中复用，如果有的模块实在做不到通用，那就例如把通用的当父类，具体的子项目当子类，继承父类的通用部分，然后重写具体的子项目部分。仍然还是大致以数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程5部分来。先认真的彻底分析和理解现有cls/det/gen/ssl/seg项目代码（需结合对应readme/design/workflow一起理解），再仔细的调研公认高质量项目的架构设计等等（不要局限医疗，可能自然图像，NLP，LLM，VLM有更好的项目）。  

进展：

### 重构现状总览（2026-07-23 关闭）

**TODO 3 已高质量关闭。** `taskcore` 为五任务公共框架层（见 `taskcore/README.md`），设计取向是**不吞并任务主流程**，只收敛逐字重复的工程件。R1–R7 + 审查热修已落地；性能/产品可选项归档至独立 backlog，不再阻塞本项。

**五任务共性已全员接入**：AMP/bf16、channels_last、fused AdamW+wd 分组、梯度累积/裁剪、非有限守护、EMA(+CPU offload)、warmup 调度、resume fail-fast、原子 checkpoint+history、DDP、梯度检查点、`CudaPrefetcher`（含 ssl）、`build_topology` 几何单一真相源、`GPUAugmentor`+`Companion` 增强核心。

---

### 五模块 × 五任务接入矩阵（关闭时快照）

| 模块 | seg | cls | det | gen | ssl |
|---|---|---|---|---|---|
| **Config** | `SegBundle`（core + `seg:`） | registry + `ClsConfig` | registry + `DetConfig` | 子类扩展 + **io/validation 委托 core** | registry + `SSLConfig` |
| **Data** | core 全套 + 双源混采 | core patch 基类 + `cls_dataset` | core patch 基类 + `det_dataset` | **委托 core `prepare_one`** + cond | core `ImageOnlyPatchDataset` |
| **Augment** | core `GPUAugmentor` | core | core | core 薄封装（cond 契约） | core / multicrop |
| **Models** | core `build_model` | core backbone + 头 | backbone + FPN + 头 | core + SISR/扩散 | core blocks + 插件 |
| **Engine** | `BaseTrainer` + `_save_best` | 同左 | 同左 | 同左 | 同左 + backbone 导出 |
| **Predictor** | 自有（滑窗/AdaBN 等） | `BasePredictor` | + stitching | `BasePredictor` | 无（probe） |

---

### 已收敛、无需再动

- 模型 backbone 全家桶均在 `taskcore/models/`
- cls/det/ssl registry 配置 I/O；`_COMPOSITE_SKIP_CORE=()`（R2）
- 增强核心 `Companion` 在 core；gen 薄封装是契约胶水
- checkpoint 统一 `BaseTrainer._save_best` + `extract_model_state_dict`
- shim 仅保留 4 个声明入口（`segtask_v1/config.py`、`data/make_data.py`、`monitor`×2）
- **R7**：gen `data`/`augment` 全委托、`2_5d` 委托几何段（`check_channel_layout=False`，因 gen in_channels 含 cond）；stem/stage 共享 `section_validators.py`；model arch allowlist 故意分叉（gen: edsr/rcan；core: mednext/…）

---

### ~~真实欠账~~ → 已关闭清单（原 P0–P2）

**P0（已完成）**

1. ~~gen config fork~~ → R1 io de-fork + **R7 validation 委托**（model 任务侧保留为接受的 allowlist 分叉）
2. ~~loss/predict 常驻 core~~ → R2 `SegTaskConfig` / `SegBundle`
3. ~~gen make_data fork~~ → R3 委托 core `prepare_one`

**P1–P2（已完成或归档）**

4. ~~checkpoint 统一~~ → R4
5. ~~augment GPU 等价~~ → R5
6. ~~shim 清理~~ → R6（余 4）
7. ~~data 路径 `assert`~~ → R7：`dataset.py`/`loader.py` 改异常；**模型/prefetch 内断言**保留为内部不变量（见 backlog）
8. TTA 键名 / one_cycle 欠步 / memory 估计近似 → **维持记录，不改**

---

### 独立 backlog（移出 TODO 3；性能/健壮性可选项）

| 项 | 说明 |
|---|---|
| bf16 非有限守护同步 | 性能优化 |
| NSD EDT / bbox 加速 | 性能优化 |
| 扩散 SDPA | 可选加速 |
| ADM linear-attn zero-init | 一致性微优化 |
| plateau 未调 `step_epoch` 告警 | UX |
| scaler dtype 漂移告警 | UX |
| `AsyncCheckpointSaver.close` | 生命周期 |
| `seed_everything` 拆 `configure_backends` | API 洁癖 |
| adm/edm2/prefetch 内部 `assert` | 接受为构造期不变量（非配置路径） |

---

### 产品决策项（关闭时拍板：**全部维持现状**）

| 项 | 决议 |
|---|---|
| gen 训练/推理分辨率不等价 | **维持现状** |
| gen spacing 归一化 + 逐类 fg 索引 | **维持现状**（gen 不接） |
| 双源混采推广到 cls/det | **维持现状**（仅 seg） |
| AdaBN 上提 taskcore | **维持现状**（仅 seg 推理） |
| UpKern pretrain 重映射上提 | **维持现状**（仅 seg） |
| z-interleaved 滑窗搬到 gen | **维持现状**（不搬） |
| SSL SWA + early stopping | **维持现状**（有 EMA teacher） |
| Mamba backbone | **维持现状**（不追） |

---

### 各任务层「应保留」的专有代码（不是欠账）

- **seg**：`pipelines/`、滑窗/blend/AdaBN、topo loss、`losses/`
- **det**：FPN、四检测头、`targets.py`、`stitching.py`
- **cls**：MIL、mixup/cutmix、分类头
- **gen**：degradation/diffusion/generation、SISR；**model 校验 allowlist**
- **ssl**：方法插件、`probe`、backbone 导出

---

### 实施路线图 — 全部完成

| 轮次 | 内容 | 状态 |
|---|---|---|
| **R1** | gen config io de-fork | ✅ |
| **R2** | loss/predict → `seg:` | ✅ + 审查热修 |
| **R3** | gen make_data → core | ✅ |
| **R4** | checkpoint → `_save_best` | ✅ |
| **R5** | augment GPU 固定 seed | ✅ |
| **R6** | shim 清理（余 4） | ✅ |
| **R7** | validation 委托 + hoist fail-fast + specs/assert | ✅ |

**→ R7 已完成（2026-07-23）**：gen validation `data`/`augment` 全委托 `CoreConfig`；`2_5d` 委托几何段且 `check_channel_layout=False`（gen `in_channels` 含 cond，不可套用 seg 的 `D*n_views`）；`section_validators` 共享 stem/stage；`hoist_legacy_seg_sections` 新旧并存 fail-fast；`DatasetCommonCfg.from_cfg` 用 `Any`+`getattr(loss)`；data 路径 assert→异常。回归：`tests/test_todo3_r7_close.py`。

**→ R2 审查热修 / R6–R1**：见历史条目（2026-07-22/23）。

#### **TODO 3 关闭声明（2026-07-23）**

抽离通用框架主题下的欠账已收敛；性能项与产品项已归档标注。后续工作请开独立 TODO，勿再往本项追加。

4 模型流可视化需要有层次化，结构化，美化，可以清晰看到计算流的走向，可以清晰理解模型架构，可以清晰的溯源。总之：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判、讨厌"自动布局默认输出"式的无设计感结果。以下是一些例子：  

- 聚焦模式到stem, stage这个层级为止：
点击模块A，进入聚焦模式，模块群B和A有联系，模块群C和A没有联系，所以模块群C谈出，模块群B突显。我希望到stem，stage这个级别的模块能进入聚焦，再进一步的子模块例如stem，stage里面的子模块则不进入聚焦。  

- 连线走线需要清晰、不重叠、不交叉、美观、可以溯源：
需要清晰的看到不同模块的关系，并能溯源输入输出等等

- 位置清晰，层次清晰，严格遵守各自的位置关系：
例如输入后可能同时结果多个stem，那么这几个stem就是位置并列的；例如如果有deep supervision，且在dec level 0后有ds head 2, dec level 1后有ds head 1等等，那么ds head 2位置就应该和dec level 1并列，因为它们就是dec level 0的下一个计算。

- 其它的我暂时没有想到，请你根据我的喜好推荐，注意，原则是：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判。

进展：
