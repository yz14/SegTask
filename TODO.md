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

Step A — 配置层与真相源

目标：审 core.py 的公共 dataclass 继承体系、各任务段叠加机制、sync/validate/build_topology 契约、跨任务默认值与 fail-fast 覆盖面。
产出：分层清单（正确性/优化/可借鉴），含行级引用。验收：结论可复核。依赖：无。

Step A 审查报告 — 配置层与真相源
审查范围
公共层 @/d:/codes/work-projects/SegTask/taskcore/config/core.py（2368 行）+ @/d:/codes/work-projects/SegTask/taskcore/models/topology.py（真相源派生），并对照五任务的配置扩展方式：config.py（shim）、config.py、config.py、config.py（组合式）与 gentask/config/{dataclasses,validation,io}.py（子类式）。

总体评价
配置层工程成熟度很高，是全框架"契约闭合"的中枢：_dataclass_from_dict 对未知键/废弃别名/已派生只读键三类全部 fail-fast 并给迁移提示（core.py:2266-2305）；几何派生量收敛到 build_topology 单点、以只读 property 暴露、YAML 不可写（core.py:535-543、topology.py:75-166）；validate() 按 section 拆分、覆盖面极广（model/selfattn/multirf/augment/loss/data/2_5d/train/predict/monitor 共 11 个子校验器）；ConfigError 同继承 AssertionError, ValueError 兼顾历史 assert 语义与 python -O（core.py:16-23）；num_classes 自动探测后 loader 重跑 cfg.sync() 修正 topology（data/loader.py:780-781、gentask/data/loader.py:135-136），无陈旧派生 bug。未发现破坏性正确性 bug。以下按严重度列出可改进项，重点是框架级（跨五任务）结构问题，与 TODO2 的 seg 语境不重复。

A. 正确性 / 稳健性
A1（中，框架级最大结构欠账）配置子系统存在两套并行扩展策略，gen 完整 fork 了 core

组合式（cls/det/ssl）：原样复用 taskcore.config.core.Config 作为 data/model/train 载体，仅新增任务段 dataclass 从 YAML 顶层 cls:/det:/ssl: 解析（clstask/config.py:29-33,48,295；dettask/config.py:28-32,214；ssltask/config.py:24-28,733）。
子类式（gen）：gentask/config/ 是一整套平行实现——dataclasses.py 逐个子类化核心段（DataConfig(_core.DataConfig) 等，gentask/config/dataclasses.py:31,52,74,85）、validation.py（35KB，几乎重写了 Config + 全部 _validate_* + 生成专属 _validate_task/_validate_sisr_arch，gentask/config/validation.py:20,113,248）、io.py（重写 load_config+_dataclass_from_dict，gentask/config/io.py:99）。
这意味着 core.py 的 2368 行校验逻辑在 gen 侧有一份大规模复制；任何一处校验修 bug 都要同步两处。这是配置层唯一的系统性架构欠账，直接对应 TODO3 的重构入口。价值高（阻塞五任务统一）、成本中（需设计统一的"核心段 + 任务段"注册机制）。

A2（中）"公共" Config.validate() 内嵌 seg 专属语义，非 seg 任务被迫满足 validate()（core.py:1310-1319）无条件跑 _validate_loss（seg 损失名白名单，core.py:1773-1776）、_validate_predict（阈值/AdaBN/z-interleave，core.py:2092）、_validate_2_5d（aux_seg_supervision，core.py:1915）。cls/det/ssl 复用该 Config 却不消费 cfg.loss/cfg.predict，只因默认值恰是合法 seg 值才通过。这是"共享层其实是 seg 的 Config 被别人复用"的漏抽象——loss/predict 段本应属任务段而非公共段。gen 正是因此不得不 fork（A1）。

A3（低，真相源例外的 config 侧回声）sync() 局部重算 n_views 与 keep_native 决策 sync() 在 core.py:1223 重算 n_views=max(len(multi_res_scales),1)，并在 :1224-1239 重复 keep_native_view_depth/keep_native_multi_res 的判定逻辑做 z_boundary_mode 副作用，与 topology.py:96-105 的同名决策重复。注释（core.py:1221-1222）自认这是"data 侧副作用、非 topology 范畴"，当前值等价、无 bug；但它是 TODO2-X1「真相源遵守度例外」的 config 侧回声——新增 patch_mode 时此处易漏改。建议改为读 build_topology(self) 的结果再做副作用。

A4（低，CLI override 缺口）_coerce 对 None 默认值字段回退为字符串 _coerce（clstask/config.py:322-331，det/ssl 同构）按 old 的类型分派；当字段默认是 None（data.target_spacing: Optional[List[float]]、model.random_rotate_range_per_axis、train.val_empty_cache: Optional[bool]）时所有 isinstance 均不命中，直接返回原始字符串。故这些 Optional 字段无法经 --override 正确设值（会被写成 str，下游按 list/bool 消费时类型错配）。低频但属真实缺口，建议对 None 走 yaml.safe_load/json.loads 兜底。

B. 优化空间 / 工程质量（DRY / 可维护性）
B1（中）YAML I/O 样板在四任务逐字复制 _coerce/_set_dotted/apply_overrides/load_config/save_config 在 cls/det/ssl 三处近乎逐字重复（clstask/config.py:322,334,345；dettask/config.py:239,251,262；ssltask/config.py:758,772,783），gen 另有一份。建议 taskcore 提供泛型 load_task_config(path, TaskCfgCls, section, validate_fn) 与 apply_task_overrides(cfg, task_cfg, overrides, section)，五任务各减 ~60 行样板。低成本、纯收敛。

B2（低）build_topology 用 getattr 反向读取任务段字段 topology.py:90-93 以 getattr(dc, "cond_dirs", ()) 读取一个公共 DataConfig 并不声明、仅存在于 gen fork 的字段，并内嵌"生成任务 cond"注释与 cond_in_channels 派生。公共真相源反向依赖任务子类字段，是双向的漏抽象（呼应 A2/TODO2-X2）。统一后应让公共 DataConfig 正式声明 cond_dirs（seg 恒空）或由任务段显式注入 topology 输入，去掉 getattr 探测。

B3（低，记录，非缺陷）派生只读量与序列化的隔离已到位 _spatial_dims/_in_channels 在 __post_init__ 作实例属性而非 dataclass field（core.py:531-533），save_config 的 asdict(cfg) 只序列化声明字段，不会把派生量泄进 YAML；property（spatial_dims/in_channels/save_best_metric/save_best_mode）同理。round-trip 安全，无需改动，仅记录该设计正确。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中）向 Hydra/OmegaConf 式「结构化配置 + 组合」靠拢，替代手写 _dataclass_from_dict 当前是自研 dataclass↔YAML + 手写别名/废弃/未知键校验（core.py:2232-2305）。2026 社区（HF/timm/mmengine/nnU-Net v2 plans）普遍用 OmegaConf structured configs 或 Pydantic v2：原生支持嵌套 override（data.target_spacing=[1,1,1] 直解、解决 A4）、类型强校验、defaults 组合（把"核心段 + 任务段"做成可组合 group，直接解决 A1/A2 的两套策略）。属较大重构，列为 TODO3 的候选底座，不建议轻动。

C2（中）任务段注册表（registry）统一 A1/A2 即便不引入外部库，也可在 taskcore 定义 TaskConfig 协议 + 注册表：公共 Config 只含真正五任务共用段（data/model/train/aug），把 loss/predict 下沉为"seg 任务段"，gen/cls/det/ssl 各注册自己的段与 validate 钩子，load_task_config 统一装配。既消 gen fork（A1），又消 seg 语义泄漏（A2），且不改 YAML 用户接口。这是 C1 的轻量版，成本中、价值高。

C3（低）Pydantic v2 校验器替代散落的 _require validate() 是 11 个手写子校验器、数百个 _require。Pydantic v2 的 field/model validator 可把"枚举白名单/区间/长度/跨字段一致性"声明化，报错定位更准，并天然支持 JSON schema 导出（利于 launcher 表单自动生成，DESIGN.md:186-188 已有 launcher 表单需求）。属工程升级，优先级低。

Step B — 引擎抽象层

目标：审 engine（base_trainer/base_predictor/amp/optim/checkpoint/dist_utils/launch/prefetch/bn_stats/memory/views）作为五任务共用基类的模板方法设计、复用度、可扩展性；对照各任务 trainer/predictor 子类化实际接线。
产出/验收：同上。依赖：A（config 契约）。

Step B 审查报告 — 引擎抽象层
审查范围
公共层 @/d:/codes/work-projects/SegTask/taskcore/engine/ 全 11 个模块：base_trainer.py（692 行）、base_predictor.py（65 行）、amp.py、optim.py、checkpoint.py、dist_utils.py、launch.py、memory.py、prefetch.py、bn_stats.py、views.py；对照五任务子类接线：segtask_v1/trainer/trainer.py、clstask/trainer/cls_trainer.py、dettask/trainer/det_trainer.py、gentask/trainer/gen_trainer.py、ssltask/trainer/ssl_trainer.py，以及五个 *Predictor 子类。

总体评价
引擎层采用"模板方法的显式装配变体"（base_trainer.py:3-14）：基类**刻意不吞并训练/推理主循环**，只把构造期与运行期的逐字重复工程件收敛为 protected helper，子类在 __init__ 里按原顺序显式调用。构造期复用度很高——channels_last / optim+sched / amp / ema / compile / ddp / swa / monitor / resume / pretrain 全部下沉基类（base_trainer.py:71-166、472-528、_restore_train_state:266-304、_load_pretrain_weights:312-360、_finalize_swa:415-467）。纯工具模块工程成熟度亦高：原子写盘 + 后台异步 saver（checkpoint.py:26-43、172-221）、位精确 RNG resume（checkpoint.py:326-334、88-104）、ZeRO-1 分片 + fused AdamW + ndim<=1 免 decay 分组（optim.py:25-101）、OneCycleLR horizon 漂移重算（optim.py:280-304）、expandable_segments/孤儿兜底/信号处理（launch.py）、EMA CPU offload（base_trainer.py:133-139）、CUDA H2D 预取（prefetch.py）、AdaBN/SWA BN 重估（bn_stats.py）、DDP 可加量归并（dist_utils.py）。BasePredictor 三个 helper（_setup_infer_amp/_autocast/flip_tta_combos）被五任务 predictor 一致复用（predictor 子类均 import 并调用）。以下按严重度列出，重点是框架级（跨五任务）结构问题，与 TODO2 seg 语境不重复。

A. 正确性 / 稳健性
A1（中，引擎层最大结构欠账，且含一处真实缺口）训练主循环在五任务重复实现，且梯度累积/非有限跳步/DDP 同步的处理不一致
基类只统一了 __init__ 装配，运行期的 `_train_epoch` 每任务各写一份（trainer.py:398-666、cls_trainer.py:256-343、det 同构、gen_trainer.py:248-356、ssl_trainer.py:580-793），而"累积边界 → unscale/clip → scaler.step + 跳步检测 → scheduler/EMA 推进 → 健康指标累计"这段 ~150 行结构几乎逐字重复。重复本身是 DRY 问题（见 B1），但其中**跳步与 DDP 同步的语义在五任务并不一致**，最严重处是一个真实缺口：
- seg（参考实现）：非边界步 `fwd_model.no_sync()` 免 all-reduce（trainer.py:481-483）、**恒反传**、bf16/fp32 跳步用 `all_reduce_flag_any` 统一各 rank 决策（trainer.py:566-567）。
- ssl：方法插件无单一 forward 入口，改"初始广播 + 边界手动梯度 all-reduce 均值"，跳步同样用 `all_reduce_flag_any`（ssl_trainer.py:67-75、718-720）。
- cls/det：DDP 包装（_setup_ddp）但**不使用 no_sync**，且**恒反传**（cls_trainer.py:270 无条件 backward）。因恒反传，非有限性经逐 micro-step 的 all-reduce 传播到所有 rank，跳步判定 `grad_nonfinite` 天然一致——**结论安全**，但代价是累积下每 micro-step 都 all-reduce 一次梯度（带宽浪费，见 B2）。
- gen：**条件反传**——非有限 loss 时不 backward（gen_trainer.py:285-297），却仍走 DDP 包装（_setup_ddp，gen_trainer.py:101）、无 no_sync、无 all_reduce_flag_any，跳步用**本地** `group_bad`（gen_trainer.py:299-302）。多卡下这会产生两个问题：(a) 某 rank 因非有限 loss 少调一次 backward，与其它 rank 的 all-reduce 集体计数不匹配 → 可能 NCCL 挂起；(b) 即便不挂，本地 group_bad 使某 rank 跳步、别的 rank 更新 → 副本参数发散。属低频路径（需 DDP + 非有限 loss），但是一处**潜伏的正确性/挂起缺口**。
这是 Step B 的核心结论：主循环是引擎层最后一块未统一的工程面，且未统一已经导致 gen 侧的语义偏差。建议在 BaseTrainer 增设 `_optimizer_step_boundary(...)` 模板（含 no_sync 上下文、all_reduce_flag_any 跳步、scaler.step/skip 检测、scheduler/EMA 推进、健康指标），五任务共用，一并消除 gen 缺口。价值高、成本中。

A2（中）"统一"的 _setup_optim_sched 实际只服务 seg/gen/cls，det/ssl 绕开并重写了 horizon 数学
det 需要 encoder 差分学习率 + 保留倍率的 warmup，故弃用 _setup_optim_sched，改 `build_optimizer_with_lr_mult` + `GroupWarmupScheduler`，并把 steps_per_epoch/warmup_steps/total_steps 的推导**内联重写**一遍（det_trainer.py:87-108）。ssl 也因"方法模块 + 自定义时钟"内联重写同一段（ssl_trainer.py:86-104）。于是"每 epoch 优化步 = ceil(micro/accum)、one_cycle 归零外层 warmup"这套 horizon 契约被复述三处，任一处口径漂移不会被另两处发现。建议把 _setup_optim_sched 参数化（可注入 optimizer 与 scheduler 包装类），吸收 det/ssl 两个变体。

A3（低）one_cycle + warmup_epochs>0 的处理三任务不一致
同一个非法/冗余配置组合：_setup_optim_sched 静默把 warmup 归零（base_trainer.py:110），det 亦静默归零（det_trainer.py:102），而 ssl 直接 `raise ValueError`（ssl_trainer.py:94-96）。行为都不致命，但"静默 vs fail-fast"跨任务不统一；统一到基类后应择一（建议 fail-fast，与 config 层 A 风格一致）。

A4（低）gen 训练循环每 micro-step `float(loss.item())` 强制 D2H 同步
gen_trainer.py:284 每步 `loss.item()`，打断 CUDA 流水；seg/cls/ssl 已改用 `pending` 缓存 + 单次 stack().tolist() 的懒同步（cls_trainer.py:234-254、trainer.py:426-450、ssl_trainer.py:606-648）。属正确但未对齐的性能欠账（也是 B3）。

B. 优化空间 / 工程质量（DRY / 可维护性）
B1（中）主循环 ~150 行在五任务近乎重复
如 A1 所述，pending flush / unscale+clip / scaler.step+skip 检测 / scheduler+EMA 推进 / 健康指标累计逐份复制。一个 `_optimizer_step_boundary` + `_train_loss_readback`（懒同步）模板即可收敛，并顺带消除 A1 的 gen 缺口与 A4 的 gen 同步。低成本、纯收敛、且修 bug。

B2（低）cls/det/gen 累积下无 no_sync → 每 micro-step 冗余 all-reduce
仅 seg 用 `no_sync`（trainer.py:481-483）。cls/det/gen 在 grad_accum_steps>1 时每个 micro-step 都触发一次梯度 all-reduce，通信量放大约 accum 倍。统一主循环后顺手接入 no_sync 即可（纯带宽优化）。

B3（低）checkpoint 保存侧仍每任务重复
`_restore_train_state`/`_load_pretrain_weights` 已上提基类（base_trainer.py:266-360），但 `_build_state_dict`/`_save_checkpoint` 仍留在各任务 Trainer 上——且 checkpoint.py:3-6 明确说明这是为让 `inspect.getsource(Trainer._build_state_dict)` 的 token 测试通过而**刻意**保留。这形成"加载已统一、保存未统一"的不对称，且测试与实现位置耦合。建议把保存侧也做成基类模板（容器 schema 由子类以 dict 片段注入），并把 token 测试改为针对基类。

B4（低，记录）memory 估计对非 Adam/SGD/Lion 优化器按 2× 保守；BasePredictor 极薄
memory.py:39-49 对未知优化器取 optim_mult=2，属安全默认，仅记录。BasePredictor 只有 3 个 helper（65 行），复用面窄但定位清晰，无需扩张。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中）以 HF Accelerate / Lightning Fabric 承接"AMP + 梯度累积 + DDP 同步"后端
2026 主流做法（HF/timm/mmengine）不再手写 no_sync/scaler/跳步/设备摆放，而是用 `accelerator.accumulate(model)` + `accelerator.backward(loss)` + `accelerator.clip_grad_norm_`，由框架统一处理累积边界、no_sync、GradScaler、DDP/FSDP 与 ZeRO。这恰好一次性覆盖 A1（gen 缺口）、A2、B1、B2。属较大重构，列为 TODO3 引擎底座候选，不建议轻动，但价值最高。
C2（中）FSDP2 补齐大模型分片，超越当前 ZeRO-1
现仅 ZeRO-1（只分片优化器状态，optim.py:54-77、memory.py:28-50）。gen 的 ADM/EDM2 扩散骨干（TODO3 已提）参数量大，FSDP2（分片 params+grads+optim，支持 `torch.distributed.checkpoint` 分片异步存盘）能显著降每卡显存、放大可训模型。可作为 gen 大模型分支的可选后端。
C3（低）torch.distributed.checkpoint（DCP）用于大模型分片 checkpoint
当前 AsyncCheckpointSaver 是单线程全量 state 深拷 + 后台写（checkpoint.py:172-221），对现规模足够；模型进一步放大后，DCP 的分片并行存盘更省内存与时间。优先级低，随 C2 一并考虑。
C4（低）prefetch 泛化为多层/嵌套结构预取
CudaPrefetcher 仅移动 dict 顶层的 Tensor 值（prefetch.py:51-55）；det 的 target 是 list[dict]（框），不会被预取上卡。若 det 接入预取需泛化为递归搬运（呼应 TODO3 "ssl 未接 CudaPrefetcher"外，det 的嵌套 batch 也非预取友好）。属工程小改，优先级低。

Step C — 数据框架层（通用性视角）

目标：审 data（dataset/loader/make_data/augment/specs/mixed_sampler）作为五任务通用数据底座的抽象边界、四 patch_mode 统一口径、2.5D 折叠契约、seg/gen 增强分叉。
产出/验收：同上。依赖：A。

Step C 审查报告 — 数据框架层（通用性视角）
审查范围
公共层 taskcore/data 全 6 模块：dataset.py（1364 行）、loader.py（980 行）、make_data.py（613 行）、augment.py（558 行）、specs.py（259 行）、mixed_sampler.py（216 行）；对照五任务的数据层扩展：segtask_v1/data/dataset.py（shim，逐字 re-export taskcore，dataset.py:1-8）、gentask/data/{dataset/core.py(≈900 行),augment.py(535 行),loader.py(288 行),make_data.py,specs.py}、clstask/data/{cls_dataset.py,loader.py}、dettask/data/{det_dataset.py,loader.py}、ssltask/data/ssl_dataset.py。

总体评价
data 层的"叶子层"（纯函数 I/O + 预处理 + cache）工程成熟度极高，是全框架性能与正确性的基石：make_data 逐样本 npz 预烘包 + 物理几何一致性 fail-fast（make_data.py:117-139）、原子写 + manifest 谱系（make_data.py:267-286,488-511）、逐类 fg 索引 + 精确 label_counts 快路（make_data.py:58-107,239-241）；dataset 侧未压缩 npz 的零拷贝 memmap 快路（dataset.py:253-292）让页缓存跨 DataLoader worker 共享、逐 worker LRU 卷缓存 + pickle 清空（dataset.py:602-646）、逐 worker/逐样本确定性采样 RNG（dataset.py:717-742）、NIfTI 读重试与 OOM 折叠（dataset.py:66-88）、四 patch_mode 的 max-FOV 抽取 + edge-pad 保物理 FOV（dataset.py:1018-1076）、GPU 增强的全零同步设计（CPU Bernoulli 掩码 + 异步搬设备，augment.py:20-24,90-96）、affine+elastic 融合单次 grid_sample（augment.py:239-345）、越界区 label/wmap 语义中性填充（augment.py:329-343）、双源配额混采 + DDP strided 切分（mixed_sampler.py:190-215）、val DDP 按 batch 块切分（loader.py:40-70）。这些叶子件被四任务一致 import 复用（cls_dataset.py:55-67、det_dataset.py:43-51、ssl_dataset.py:30-37、gen core.py:12-20），是本层做得最好的部分。未发现破坏性正确性 bug。

但从"通用数据底座"视角，data 层暴露出全框架最重的结构欠账：类级抽象（SegDatasetNpzBase + 三个 patch dataset）实际上只有 seg 真正复用；gen 完整 fork、cls/det/ssl 各自以独立 Dataset 重写。即"共享的只是叶子纯函数，四 patch_mode 口径 + 2.5D 折叠契约的类级结构被复制/重写了 5 份"。以下按严重度列出框架级结构问题，与 TODO2 seg 语境不重复。

A. 正确性 / 稳健性
A1（中，data 层最大结构欠账）patch dataset 的类级抽象只对 seg 成立，四 patch_mode 口径 + 2.5D 折叠契约在五处各自表达
SegDatasetNpzBase 注释自陈"抽出三类（z 轴/cubic/whole）重复的读取/缓存/采样"（dataset.py:652-658），但这份抽象的真实复用面只有 seg（shim 逐字 re-export）：
- gen：整套 fork——gentask/data/dataset/core.py 重新定义 VolumeNpzDatasetBase(Dataset)（core.py:41）、Volume3D（core.py:235）、Volume3DCubic（core.py:520）、Volume3DWhole，以及模块级 extract_z_patch_padded（core.py:450）、_extract_cubic_patch（core.py:486）、_halton，仅 import taskcore 的叶子 I/O helper（core.py:12-20）。与 seg 的唯一差异是全程多带一个 cond 伴随体（core.py:381-384,646-648）。
- cls：ClsPatchDataset(Dataset)（cls_dataset.py:198）独立实现，并逐一重写"口径同 segtask"的 _extract_cubic_patch（cls_dataset.py:163）、_safe_center_range（cls_dataset.py:180）、_rng（cls_dataset.py:367）、_sample_z（cls_dataset.py:414）、_sample_center（cls_dataset.py:429）、_load+cache（cls_dataset.py:379）、_build_fg_index（cls_dataset.py:307）。
- det：同构（det_dataset.py 独立 Dataset，仅 import 叶子 helper，det_dataset.py:43-51）。
- ssl：ImageOnlyPatchDataset 独立实现，甚至另写一套中心逻辑 _rand_center/_clamp_center（ssl_dataset.py:97-113），并用 Python random.Random 而非 numpy Generator，与 seg/cls 的 RNG 口径都不同。
后果：四 patch_mode 的抽取口径（whole/z_axis/2_5d/cubic）与 2.5D 折叠契约同时活在 5 个实现里，靠"口径同 segtask"注释 + 测试维持一致。任一处口径漂移不会被另四处发现。价值高（阻塞五任务统一）、成本中，直接对应 TODO3 的 data 层重构入口，与 config A1、engine A1 是同一病根在 data 层的第三次显现。

A2（低，真实不一致）越界 edge-pad cube 提取器有四份实现，且别名安全策略不统一
同一"以中心抽严格 size cube、越界 edge 复制"的 _extract_cubic_patch：taskcore（dataset.py:1041-1076）末尾对"无 padding"分支**无条件 .copy()**，显式断开与 LRU 缓存卷的别名（dataset.py:1072-1074）；ssl 直接 import 该 taskcore 版（ssl_dataset.py:32）；gen fork 私有一份（core.py:486）；而 cls 定义**私有** _extract_cubic_patch（cls_dataset.py:163-177），无 padding 分支返回的是缓存卷的**视图**（cls_dataset.py:174 `out = vol[tuple(slices)]`，无 copy），随后 `np.ascontiguousarray(img_patch, dtype=np.float32)`（cls_dataset.py:499）对已连续的 fp32 视图不产生拷贝——即返回张量与 worker LRU 缓存卷共享内存。当前 collate 会 stack 出新张量、CPU 侧无原地写，故实践大概率安全；但这是与 taskcore/gen"防御性 copy"契约的一处潜伏别名差异，建议统一到单一实现（呼应 A1）。低（需在 cache_mode=memory + 特定下游路径下验证）。

A3（低）样本→卷索引方案 seg 与 cls 相反，缓存局部性/验证铺点口径不一致
seg/gen：vol_idx = idx % n_vols（dataset.py:924、core.py:351），val 覆盖序号 j = idx // n_vols（dataset.py:749）——连续 idx 走不同卷。cls：vol_idx = idx // spv（cls_dataset.py:482），cov_j = idx % spv——连续 idx 走同一卷（cls 注释明说利于逐 worker LRU 命中，cls_dataset.py:481）。两者都正确，但缓存命中率与（shuffle=False 的验证态）batch 组成不同；统一后 LRU 行为与验证铺点才跨任务一致。低。

A4（低，真实不一致）单进程训练 loader 的 drop_last 跨任务不一致
seg（loader.py:898）、gen（gen loader.py:218）单进程训练 loader 用 drop_last=True，且 gen 额外显式拦截 len<batch_size 空转（gen loader.py:171-176）；而 cls 单进程训练 loader 用 drop_last=False（clstask/data/loader.py:155），无 len<batch_size 拦截——末批可能 batch=1，BN/批统计不稳。行为都不致命，但"丢末批 vs 保末批 + 是否拦截空转"三任务不齐；统一到基类装配器后应择一。低。

B. 优化空间 / 工程质量（DRY / 可维护性）
B1（中）build_dataloaders 在四处近乎整体重复
taskcore.build_dataloaders（loader.py:696-980）、gen build_dataloaders（gentask/data/loader.py:58-288 整份 fork）、build_cls_dataloaders、det、ssl 各一份。其中 DDP sampler 装配、num_workers 逐卡平摊（scaled_num_workers，已上提但调用重复）、loader_kwargs 组装、ValBatchShardSampler 接线逐份复制。建议 taskcore 提供泛型 build_task_dataloaders(cfg, make_train_ds, make_val_ds, *, mixed_cfg=None)，把"划分→spec→DataLoader/DDP 装配"收敛为一处，五任务各减 ~60–120 行。纯收敛。

B2（中）VolumeCache 缓存足迹估计块整段复制到 gen；cls/det/ssl 反而缺失
loader.py:924-978（~55 行诊断，估算逐 worker × 逐卷 RAM 并给 cache_max_volumes 建议）被 gen loader.py:236-286 逐字复制；cls/det/ssl 无此诊断（属缺失而非重复）。收敛为 taskcore.log_volume_cache_estimate(ds, cfg, world_size) 后全员受益且 gen 去重。

B3（低）region_weights 来源两任务不一致
seg 的 DatasetCommonCfg 从 cfg.loss.region_weights 取（specs.py:73），gen 从 cfg.data.region_weights 取（gentask/data/specs.py:55）。这是 config A2「loss 段其实是 seg 语义、应下沉任务段」的 data 侧回声；统一 config 分层后此处也应收敛到单一来源。低。

B4（低，记录，非缺陷）叶子层复用确实到位
extract_z_patch_padded / resize_3d / preprocess_image / load_npz_* / VolumeCache / _halton / _group_fg_*_by_class 被四任务一致 import（cls_dataset.py:55-67、det_dataset.py:43-51、ssl_dataset.py:30-37、gen core.py:12-20）；memmap 零拷贝快路（dataset.py:253-292）+ OS 页缓存跨 worker 共享是全框架 I/O 性能基石。round-trip 与别名策略在 taskcore 侧正确，仅记录，无需改动。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中，价值最高，A1/B1 的落地）把 SegDatasetNpzBase 提为真正的模板基类 + patch-extraction 策略注册表
公共基类持有 I/O / LRU cache / 逐 worker RNG / 四 patch_mode 中心采样 / max-FOV 抽取 / 2.5D 折叠契约；任务子类仅通过 hook 注入"样本组装差异"——seg 出 label（逐类二值堆叠）、gen 出 cond（bilinear companion）、cls 出 target（mask 弱标签/表）、det 出 boxes（连通域/npz）、ssl 出 masking/multicrop 视图。这样既消 gen 的整套 fork，又消 cls/det/ssl 三份口径重写，patch_mode 口径回到单一真相源。这是 TODO3 data 层入口，价值最高、成本中。

C2（中，data 层最后一个实现分叉）以 MONAI/TorchIO 式"伴随张量（companion）"抽象统一空间联动增强
现状 augment.py（558 行）与 gentask/data/augment.py（535 行）中，_build_rotation_matrices / _elastic_grid_disp / _random_affine_elastic 主体 / 全部强度增强（brightness/contrast/gamma/noise/blur/lowres）逐字相同，唯一差异是伴随张量不同（seg: label+weight_map；gen: cond+weight_map）。2026 主流（MONAI 1.4 MetaTensor dictionary transforms、TorchIO Subject）正是把 label/weight/cond 声明为"随 image 同 warp 的 companion 张量"，一份 spatial transform 消化全部，插值模式按张量语义（image/cond=bilinear、label=nearest、wmap=可配）声明。TODO3 已把此列为"伴随张量 spec 化"最后一个分叉；落地需 GPU 做固定 seed 等价性验证 + 短程 sanity check。价值高。

C3（低）把 MixedBatchSampler 泛化为通用 QuotaBatchSampler
现"整数配额混采 + DDP strided 不相交切分 + RNG 消费对齐"（mixed_sampler.py:190-215）仅 seg 双源可用。抽象为与来源数无关的 QuotaBatchSampler（N 源、每源每 batch 配额、DDP 对齐）后，cls/det 的类均衡/难例混采可直接复用（是否需要属产品决策，TODO3 已列）。低。

C4（低-中）统一 make_data，让 gen 直接获得 spacing 归一化 + 逐类 fg 索引
gentask/data/make_data.py 是 taskcore make_data 的旧 fork，落后于 taskcore≥1.3/1.6 的 spacing_normalization（make_data.py:216-231,343-364）与逐类 fg 索引/label_counts 快路（make_data.py:58-107,239-241,257-258）。把 make_data 统一（cond 打包作为 hook 注入 payload），gen 可直接获得中位 spacing 各向同性重采样与类均衡采样。TODO3 已列为待你决策项（需新增 gen 配置、重打包数据）。低-中。

Step D — 模型框架层（通用性视角）

目标：审 models（factory/topology/blocks 库/各骨干）作为通用装配层跨 seg/cls/det/gen/ssl 的复用与缝隙（呼应 TODO2 X1/X2 真相源欠账，但从框架泛化角度）。
产出/验收：同上。依赖：A。

Step D 审查报告 — 模型框架层（通用性视角）
审查范围
公共层 taskcore/models 全 13 模块：blocks.py（1502 行，通用块库）、stem.py（338 行，context-fusion stem）、topology.py（170 行，真相源）、unet.py（611 行，Encoder/Decoder/DecoderLevel/UNet3D/头）、unet3p.py、unetpp.py、resnet.py、convnext.py、mednext.py、adm_unet.py、edm2_unet.py、factory.py（557 行，装配总入口）、__init__.py；对照五任务模型层接线：segtask_v1/models/*（12 个逐字 shim，re-export taskcore）、gentask/models/*（10 个 shim + fork 的 factory.py + 新增 generation.py/diffusion.py/sisr.py）、clstask/models/{factory,classifier,densenet,vit}.py、dettask/models/{factory,detector,fpn,heads/*}.py、ssltask/models/{ssl_models,dino/ibot/jepa/spark/vicregl_modules}.py + methods/*。

总体评价
模型层是全框架抽象做得最好的一层，与 config/data/engine 的「两套并行实现」形成鲜明对比：整套块库 + backbone + UNet 家族 + stem + 真相源（blocks/stem/topology/unet/unet3p/unetpp/resnet/convnext/mednext/adm_unet/edm2_unet）全部单点收敛在 taskcore，seg 与 gen 均以逐字 shim（sys.modules 别名）re-export（grep 确认 seg 12 个、gen 10 个模型文件均为 `[shim]`），未见任何实现复制。块库工程成熟度很高：spatial_dims 分派表统一 2D/3D（blocks.py:21-29）、7 种通道/空间注意力 + 工厂（SE/ECA/CBAM/Coord/LKA/MSCA，blocks.py:436-467）、4 种内容自注意力（softmax/linear/window/grid 均走 SDPA，blocks.py:795-916）、nD RoPE 带 torch.compile 感知的有界 LRU（blocks.py:521-550）、抗混叠 BlurPool / 子像素 PixelShuffle+ICNR / CARAFE / DySample 上采样族（blocks.py:1052-1493）、各向异性 per-axis stride 贯穿 Downsample/Upsample（blocks.py:109-125,1204-1244）、DropPath 先 fp32 采样避免 AMP 后端差异（blocks.py:79-82）、GRN/CoordAttn 的 fp32 统计累加（blocks.py:97-102）、逐 stage 梯度检查点 use_reentrant=False+preserve_rng_state（blocks.py:49-64）。真相源 topology.build_topology 单点派生全部几何量（topology.py:75-166，Step A 已详审）。跨任务复用范式一致且正确：cls/det/ssl 都通过 taskcore.build_model 复用 encoder(/decoder)，靠「encoder./decoder. 同名同形」保证 SSL/分割 checkpoint 经 strict=False 干净迁移（classifier.py:3-5、detector.py:3-4、ssl_models.py:7-11），命中率打日志、0 命中 fail-fast（cls factory.py:104-122、det factory.py:59-63）。未发现破坏性正确性 bug。以下按严重度列出框架级结构问题，与 TODO2 seg 语境不重复。

A. 正确性 / 稳健性
A1（中，真实缺口，模型层唯一确定性 bug）gen fork 的 factory 引用未导入的模块级常量，各向异性下采样路径必崩
gentask/models/factory.py:12-18 只从 taskcore.factory import 了 `_make_*_stage_builder`/`_resolve_blocks_per_stage`/`compute_downsample_strides`，但其自带的 `_validate_anisotropic_downsampling`（gen factory.py:77-102）在 :94 与 :98 引用 `_ANISO_DOWN_MODES` / `_ANISO_UP_MODES`——这两个名字只定义在 taskcore.factory.py:248-249，gen 侧既未 import 也未定义。该校验函数在 `ds_strides` 含非 2 stride 时才被触发（gen factory.py:80-82 提前 return 各向同性情形），即用户设 `model.anisotropic_pooling=true` 或显式 `downsample_strides` 时，进入 :94 立即抛 `NameError: name '_ANISO_DOWN_MODES' is not defined`，而非预期的可读 ValueError。grep 确认 gen 全仓无此二常量定义。结论：gen 的 UNet backbone 各向异性下采样路径完全不可用（一进校验即 NameError）。属真实缺口（低频但确定崩溃）。修复只需 import 该二常量，或（更好）见 C2 让 gen 复用 taskcore.build_model 不再 fork。

A2（中，模型层最大结构欠账）gen 的 `_build_unet_backbone` 是 taskcore.build_model UNet 路径的整段 fork，且已漂移出多处特性缺失
块库虽全共享，但「装配逻辑」factory.py 是 gen 唯一 fork 的模型文件（gentask/models/factory.py:105-216 vs taskcore/models/factory.py:331-556）。fork 导致 gen 的 UNet 相对 seg 主线缺失以下能力，且不会被 seg 的测试覆盖发现：
- 无 mednext backbone：gen 只认 resnet/convnext，其余 `raise ValueError`（gen factory.py:64-73），而 taskcore 支持 mednext（taskcore factory.py:409-412）。
- 无 MultiRF / SelfAttention 逐 stage 注入：gen 调 `_make_resnet_stage_builder(cfg, enc_counts)`（gen factory.py:65-66）不传 multirf_mask/selfattn_types 两个 mask 参数，而 taskcore 逐 stage 装配空洞多分支块与自注意力块（taskcore factory.py:364-402）。gen 的 UNet 因此无法使用这两族增强。
- 无 aux_topo_head：gen 构造 UNet3D 时未传 `aux_topo_head`/`aux_topo_head_mode`（gen factory.py:182-193 vs taskcore factory.py:529-530），中心线/距离场辅助头对 gen 不可用。
- decoder norm/attn 契约漂移：gen 向 Decoder/UNetPPDecoder/UNet3PDecoder 均省略 `attn_gate_norm`/`norm_type`/`norm_groups`/`activation`/`upsample_norm_act`（gen factory.py:148-177），全部退回类默认（`attn_gate_norm="batch"`、`norm_type="instance"`）。后果有二：(a) 一旦 gen 用户设 `skip_attention=true`，注意力门控恒用 BatchNorm——这正是 seg 主线用 `attn_gate_norm="auto"→跟随 norm_type`（taskcore factory.py:471-473）刻意规避的小 batch 3D 噪声 BN 问题，gen 无此保护；(b) gen 的 unetpp/unet3p 融合卷积恒 instance norm，即便用户把 `model.norm_type` 设为 group/batch 也不生效，与 encoder（经 stage_builder 读 cfg.norm_type）不一致。
这是模型层与 config A1 / data A1 / engine A1 同源病根的第四次显现，但显著更轻——只 fork 了一个 factory.py，块库无复制。价值高（阻塞五任务模型装配统一 + 消 A1 崩溃 + 补齐特性），成本中。

A3（低）「build 全模型再丢弃头」是共用范式的既定代价，但也是一处隐式契约耦合
cls（build_seg_model(cfg).encoder，cls factory.py:46）、det（取 encoder+decoder，det factory.py:67-68）、ssl（build_model 后取 encoder[+decoder]，ssl_models.py:84,192）都先构造完整 UNet3D（含 seg_head、可选 ds_heads、aux_heads），再丢弃只留骨干。这是为「逐参数同名同形以便 pretrain 迁移」付的价，语义正确；但代价是：(a) 构造期白建并即刻 GC 掉解码头（cls 甚至连整个 decoder 都不用却被迫构建）；(b) 这些下游任务被迫让 seg 的 `deep_supervision`/`aux_seg_supervision`/`out_classes` 配置保持自洽，即便它们从不消费。当前默认值下无 bug，但 taskcore 缺一个「只出骨干、不出任务头」的公共入口（见 C1）。低。

B. 优化空间 / 工程质量（DRY / 可维护性）
B1（中）gen factory 复述了 taskcore 的三段装配数学，任一处口径漂移不会被另一处发现
除 A2 的整段 UNet fork 外，gen 还逐份重写了 `_resolve_decoder_counts`（gen factory.py:39-53 = taskcore factory.py:346-359 的 unet/unetpp/unet3p 分派）与 `_validate_anisotropic_downsampling`（gen factory.py:77-102 = taskcore factory.py:418-449），以及 backbone→stage_builder 分派（gen factory.py:56-74 = taskcore factory.py:384-414 的子集）。这些都是「装配契约」的复述，是 A1 崩溃的直接土壤。收敛方向见 C2：让 taskcore.build_model 参数化出「注入 cond/额外头」的 hook，gen 直接复用而非 fork。

B2（低）注意力/自注意力工厂已很全，但选择键分散在两处枚举 + factory 参数里
`ATTENTION_TYPES`（blocks.py:467，通道/空间注意力）与 `SELFATTN_TYPES`（blocks.py:477，内容自注意力）是两套正交枚举，分别经 `make_attention` 与 `SelfAttentionBlock` 装配，再由 factory 的 `_make_resnet_stage_builder` 分别接线（taskcore factory.py:82,124-138）。功能正确、覆盖极广，仅记录：新增注意力类型需同时改块库枚举 + factory 接线 + config 校验白名单三处，属可接受的分散度，暂无需动。

B3（低，记录，非缺陷）块库的 nD 泛化与数值稳健策略已到位
spatial_dims 分派表（blocks.py:21-29）、per-axis stride 规整（blocks.py:109-125）、GroupNorm 组数不整除自动回退 + 去重告警（blocks.py:150-159）、DropPath/GRN/CoordAttn 的 fp32 统计（blocks.py:79-82,97-102）、RoPE 缓存在 torch.compile 下绕过 dict LRU 免 graph break（blocks.py:530-532）、window/grid 注意力的 padding+mask 边界处理（blocks.py:640-695）均正确且被四任务一致复用。round-trip 与 2D/3D 等价性由块库单点保证，仅记录，无需改动。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中，价值最高，A3 的落地）taskcore 增设「骨干专用」公共入口 + timm 式 feature_info 契约
2026 主流骨干库（timm 1.x 的 `features_only=True` + `feature_info`、mmengine 的 registry、transformers 的 backbone API）都把「只出多尺度特征、不含任务头」作为一等公民。当前 cls/det/ssl 全靠「build 全模型再丢弃头」拿骨干（A3）。建议 taskcore 提供 `build_backbone(cfg) -> (encoder, decoder|None)` 与显式的 `encoder.out_channels_list` / `decoder.out_channels` 特征契约（Encoder 已返回逐级特征列表，unet.py:179-213；只差一个不建任务头的入口）。收益：cls 不再白建 decoder+heads、det/ssl 装配意图显式化、pretrain 同名同形契约由「同一 backbone 入口」而非「同一全模型入口」保证。成本低、纯收敛。

C2（中，A1/A2/B1 的统一）把 gen 的 fork 收敛为「taskcore.build_model + 生成 hook」
gen fork factory 的真实增量只有两点：cond 伴随通道（已由 topology.cond_in_channels + Encoder.cond_stem 在公共层支持，unet.py:107-124）与「生成主线用 attn_gate_target='upsample'」（unetpp）。建议给 taskcore.build_model 增一个可选 `decoder_overrides`/`attn_gate_target` 参数（默认 seg 语义），gen 传生成语义即可，彻底删掉 gentask/models/factory.py 的 UNet 分支——一举消除 A1 的 NameError、A2 的四项特性缺失与 norm 漂移、B1 的三段复述。这是模型层对应 TODO3 的最小重构入口，价值高、成本低（远小于 config/data/engine 的统一）。

C3（低）window/grid 注意力可迁移到 torch flex_attention / 可变长后端
当前 window/grid 注意力用手工 partition + additive mask 走 SDPA（blocks.py:823-891），正确但 mask 物化为 (B*win,1,1,N) 张量、pad 到 block 整数倍有额外算力。PyTorch 2.5+ 的 `torch.nn.attention.flex_attention` 支持以 score_mod/block_mask 声明窗口/网格稀疏，避免物化 mask 且可编译为融合核。属性能升级，需评估 3D + 各向异性窗口的表达力，优先级低。

C4（低）扩散骨干可纳入更现代的时间/条件注入与 flow-matching 接口
adm_unet/edm2_unet 已是论文忠实实现（GN+SiLU、σ/timestep 条件），2026 生成社区（SD3/Flux 的 MM-DiT、rectified flow / flow-matching、consistency/蒸馏采样）在超分/复原上亦有采用。gen 的 diffusion 封装（generation.py:174-221）接口已抽象（train_outputs/sample），预留了替换 backbone 的空间。属 TODO3 gen 大模型分支的算法候选，本轮仅记录，不建议轻动。

Step E — 监控与通用工具

目标：审 monitor（jsonl+HTML/rank0 守卫/失败隔离）与 utils（common/logging_utils：seed/计量/EMA/SWA）。
产出/验收：同上。依赖：无。

Step E 审查报告 — 监控与通用工具
审查范围
公共层 taskcore/monitor 全 6 模块：history.py（371 行，jsonl+summary 数据层）、dashboard.py（124 行，HTML 渲染入口 + 原子写盘）、charts.py（501 行，MetricsHistory→渲染就绪 payload）、assets.py（617 行，零依赖 CSS + SVG 折线 JS）、__main__.py（112 行，离线重渲染/多 run 对比 CLI）、__init__.py；以及 taskcore/utils 全 2 模块：common.py（591 行，AverageMeter/ModelEMA/ModelSWA/Timer/seed_everything + seg 指标数学）、logging_utils.py（182 行，控制台按模块类别+级别上色、文件纯文本）。对照五任务接线：base_trainer 的 _setup_monitor/_monitor_log_epoch/_monitor_render/_monitor_finalize（base_trainer.py:533-653）、五任务 fit() 各自调用（trainer.py:348、cls_trainer.py:601、det_trainer.py:484、gen_trainer.py:646、ssl_trainer.py:517）；五任务 train.py/pretrain.py 统一调 taskcore.setup_logging（segtask_v1/train.py:32,124；det/gen/ssl 同构）；EMA/SWA/seed 经 taskcore.utils.common 复用（测试 test_round2_fixes.py:265,348、test_swa_lka.py:30 等）。

总体评价
监控与工具是全框架**编排抽象做得最好的层之一**，与 config/data/engine 的「两套并行实现」形成对比：monitor 单点收敛于 taskcore.monitor，五任务经 base_trainer 的 _setup_monitor/_monitor_log_epoch/_monitor_render 一致接入，无任何 fork——rank0 守卫（is_main，base_trainer.py:555-556）+ 全程异常隔离（init/log/render/finalize 四处 try/except，base_trainer.py:586,622,642,651，「监测失败绝不阻断训练」）+ 原子写盘（dashboard.py:80-91、history.py:335-347 均 tmp+fsync+os.replace）+ jsonl 每 epoch 全量重写以规避续训重复行（history.py:16,343-347）+ present-only 渲染（charts 老 run 缺键整组不出现）+ 跨自动重载的 sessionStorage 状态持久化（assets.py:160-178）+ payload 反 </script> 注入（dashboard.py:28-29）。数据层数值卫生严谨：_finite_scalars/_finite_or_none 丢弃 NaN/Inf（history.py:37-60）、jsonl 解析逐行容错（history.py:156-160）。utils 侧 ModelEMA/ModelSWA 工程成熟度高：EMA 按 (device,dtype) 分组 foreach 热路径 + CPU offload 经 pinned staging 异步 D2H 单次流同步（common.py:78-130）、SWA fp32 CPU 累积等权均值（common.py:200-224）、二者原地 apply/restore + key 不匹配从零重建 + 明确文档化「BN running stats 被平滑且无收尾重校准」的取舍（common.py:47-49,197-198）；seed_everything 分 deterministic/性能两档设 TF32/matmul precision（common.py:571-590）；logging_utils 副本字段着色不污染原 record、文件恒纯文本、NO_COLOR/FORCE_COLOR/TTY 三态探测（logging_utils.py:71-129,169-175）。未发现破坏性正确性 bug。以下按严重度列出框架级结构问题，与 TODO2 seg 语境不重复。

A. 正确性 / 稳健性
A1（中，监控层最大结构欠账）监控「编排」已全统一，但「图表内容」层 charts.py 是 seg 语义硬编码，非 seg 四任务的验证曲线完全不渲染
监控管道（落盘/渲染/rank0/隔离/原子写）在 base_trainer 单点统一、五任务零 fork——这是本层做得最好的部分。但 charts.py 的**指标名清单**全是 seg 专属：_OVERVIEW_METRICS=mean_dice/mean_iou/…（charts.py:38-41）、_UNIT_SCALE_PREFIXES=dice_class_/iou_class_/…（charts.py:45-50）、_OTHER_SCALE_PREFIXES=mcc_class_/vol_sim_class_（charts.py:51-54）。而其余四任务的验证键完全不命中：cls=auc/f1/acc/vol_*（cls_trainer.py:414-415,579）、det=map（det_trainer.py:343-345,463）、gen=psnr/ssim/psnr_lr（gen_trainer.py:415-416）。后果有二：
- **验证指标面板全空**：build_single_payload 的 Validation 组（概览均值图 charts.py:216-235、逐类合并/单图 charts.py:240-277）对 cls/det/gen 均无一命中 → 整个 Validation 区块不出现；best 卡片 means/matrix 亦空（_best_card 走同一 _OVERVIEW_METRICS/_MATRIX_PREFIXES，charts.py:339,350），核心指标（AUC/mAP/PSNR）只落入次要文本列表 rest（charts.py:382）。
- **连验证损失曲线也不渲染**：charts 找 val_base_loss/val_loss（charts.py:192-193），而只有 seg 的 val 字典用 val_base_loss（validation.py:240）；cls/det/gen 的 val 字典键是裸 "loss"（cls_trainer.py:414、det_trainer.py:345、gen base_m 走 psnr_lr 无 val loss）→ 非 seg 无验证损失曲线。
即非 seg 仪表盘实际仅剩 train loss + LR + GPU + Model health 四类通用面板（这些恰好都是任务无关的，正常渲染）。它 present-only 优雅降级、不崩溃，但对 cls/det/gen/ssl **严重欠服务**——这是「共享的只是监控编排、内容层其实是 seg 的 charts」漏抽象，与 config A2「公共 Config 内嵌 seg 语义」同源，在 Step E 的第五次显现。价值高（让四任务获得一等公民验证曲线）、成本中（纯 taskcore 内收敛，见 C1）。

A2（低，真实跨任务不一致）logging_utils 顶包硬编码 "segtask_v1"，模块类别上色只对 seg 生效
_TOP_PACKAGE = "segtask_v1"（logging_utils.py:56），_module_category 仅当 logger 名首段等于该常量时取第 2 段作类别（logging_utils.py:65-68）。五任务都调 taskcore.setup_logging（segtask_v1/train.py:32、dettask/train.py:29、gentask/train.py:29、ssltask/pretrain.py:21），但 cls/det/gen/ssl 的 logger 名形如 clstask.trainer.*，首段 "clstask"≠"segtask_v1" → 返回 "clstask" 本身 → 不在 _MODULE_COLORS（logging_utils.py:34-44）→ 全部退回 _DEFAULT_MODULE_COLOR（白）。即「按 data/models/trainer/… 分类上色」只对 seg 控制台生效，其余四任务模块名恒白（级别上色仍正常、文件纯文本不受影响）。低。建议：去掉固定顶包，改按 logger 名「去掉首段后的下一段」判类，或以已知子包名集合（data/models/trainer/predictor/…）匹配任意顶包。

B. 优化空间 / 工程质量（DRY / 可维护性）
B1（低，记录，正面样板）监控编排是全框架抽象范本，无欠账
与 config/data/engine 的两套实现相反，MetricsLogger/dashboard/charts/history/CLI 单点收敛，五任务经 base_trainer._setup_monitor/_monitor_log_epoch/_monitor_render/_monitor_finalize 一致复用（base_trainer.py:533-653），rank0 守卫 + 四处异常隔离 + 原子写盘齐备，grep 确认无任务侧 monitor fork。仅记录，无需改动——可作为 TODO3「公共父类 + 显式装配」的正面参照。

B2（低）charts 单/多 run 两条路径各枚举一遍 seg 概览指标
build_single_payload（charts.py:216）与 build_compare_payload（charts.py:424-426、_compare_table charts.py:468）分别复述 _OVERVIEW_METRICS 选取逻辑；A1 泛化为「任务指标 spec」后两处一并收敛。低。

B3（低）common.py 混装了「真正通用」与「seg 专属指标数学」
AverageMeter/ModelEMA/ModelSWA/Timer/seed_everything 是五任务通用件；但 compute_dice_per_class/dice_batch_stats/derive_overlap_metrics/surface_dice_batch_stats/_nsd_stats_spacing_aware/harmonic_mean_metrics（common.py:284-568）是 seg 指标数学，仅 seg + 测试 import（cls/det/gen 各有自己的 metrics）。同 MetricsLogger.save_best_metric 默认 "mean_dice"（history.py:235）亦偏 seg（调用方均显式覆盖，无 bug）。建议把 seg 指标数学下沉到 taskcore.metrics（或 seg 任务层），common.py 只留任务无关工具，边界更清晰。低。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中，价值最高，A1 的落地）任务声明「指标展示 spec」，charts 按 spec 组装
让每任务提供一份轻量 spec（overview 键列表 / 逐类前缀+尺度 / 选模键 / val-loss 键名），charts 据此生成 Validation 面板与 best 卡片，去掉 charts.py:38-64 的 seg 硬编码。轻量即可让 cls(AUC/F1/acc)、det(mAP)、gen(PSNR/SSIM) 获得一等公民验证曲线与 best 磁贴/矩阵，同时消 A1、B2。价值高、成本低（纯 taskcore 内收敛，不改零依赖 HTML 与用户接口）。2026 主流训练可视化（W&B/TensorBoard/MLflow/aim）皆按 metric namespace 自动分组成图，可借其「命名空间分组」思路，但此处自研零依赖仪表盘已够，只需补一层 spec。

C2（低）seed_everything 补齐 2026 完整可复现开关
现 deterministic 分支仅设 cudnn.deterministic + 关 TF32（common.py:578-584），未启用 torch.use_deterministic_algorithms(True) 与 CUBLAS_WORKSPACE_CONFIG=:4096:8；2026 PyTorch 完整 bit 级复现建议二者齐备（且无确定性实现的 op 会 fail-fast 提示，利于定位不可复现源）。另 os.environ["PYTHONHASHSEED"] 在解释器启动后设置仅影响子进程（common.py:577）。属可选增强，仅严格复现需求时需要，优先级低。

C3（低）EMA/SWA 权重评估前自动触发 BN 重估
ModelEMA/ModelSWA 已文档化「BN running stats 被平滑/平均、无收尾重校准」的取舍（common.py:47-49,197-198），引擎侧已有 bn_stats.py 重估能力。可在 apply_shadow 评估路径挂一个可选钩子：默认 instance/group norm 时 no-op，引入 BN backbone（cls 的 densenet/resnet-BN）时自动重估，消除该已知隐患。属工程增强，优先级低。

C4（低）监控可选增补通用训练健康信号
现 Model health 面板（grad/weight norm、clip frac、nonfinite steps、amp scale、update ratio，charts.py:82-163）已相当完整。2026 可选增补吞吐（samples/s、step time）、data-loading 占比、per-layer 范数/直方图等任务无关信号——与 A1 的 spec 化正交、任务通用，随 C1 一并考虑。仅列增量。

Step F — 框架级横向综合

目标：五任务「通用技巧 × 任务」接入矩阵复核、抽象边界与 shim 策略评估、可扩展性总评、2026 跨域先进做法适配优先级排序（价值×成本）。
产出：全局问题总表 + 优化建议 + 借鉴清单。依赖：A-E。

Step F 审查报告 — 框架级横向综合
审查范围
综合 Step A–E 的分层结论（config/engine/data/models/monitor+utils），横向复核五任务（seg 主线 + cls/det/gen/ssl）在「通用层复用 vs 任务层扩展」上的一致性、抽象边界与 shim 策略、可扩展性，并对 2026 跨域先进做法按价值×成本排序。本轮不动任何代码/文档；本节所有跨层结论均可回溯至 A–E 的行级引用（此处不重复罗列，仅在关键处标注）。与 TODO2 的 seg 语境不重复。

总体评价（一句话根因）
全框架有且只有一条系统性病根，在五层中出现五次：**「公共层」实际上是 seg 的实现被别人复用；凡 seg 语义与通用语义冲突处，要么被迫复用（cls/det/ssl 靠默认值恰好合法而通过）、要么整段 fork（gen）**。模型层与监控/工具层抽象最好（仅 gen 一个 factory fork + charts 内容层硬编码），config/data/engine 三层是重灾区（gen 大规模 fork + 主循环/数据类/装配数学在五处复述）。除此之外未见破坏性正确性 bug；已定位的确定性缺口仅 2 处（gen factory NameError、gen DDP+非有限 loss 挂起/发散），其余均为 DRY/一致性/欠服务类。

一、全局问题总表（按根因聚类，非按层）

病根 R —「seg 实现被当公共层复用 / gen fork」的五次显现（同一件事）
| 编号 | 层 | 显现形式 | 严重度 | 确定性缺口? | TODO3 入口 |
| --- | --- | --- | --- | --- | --- |
| R-config | config | gen 整套 fork（dataclasses/validation 35KB/io），2368 行校验逻辑双份；loss/predict 段本属任务段却在公共 Config（A1/A2） | 中 | 否 | data/model 之外的第一入口 |
| R-engine | engine | `_train_epoch` ~150 行在五任务复述，且 gen 条件反传+本地 group_bad+无 all_reduce_flag_any（engine A1） | 中 | **是**（gen DDP 挂起/参数发散） | `_optimizer_step_boundary` 模板 |
| R-data | data | `SegDatasetNpzBase`+三 patch dataset 抽象只对 seg 成立；四 patch_mode 口径 + 2.5D 折叠契约活在 5 份实现（data A1） | 中 | 否 | 模板基类 + patch 策略注册表 |
| R-model | models | gen 唯一 fork `factory.py` 的 UNet 分支，漂移出 4 项特性缺失 + norm 契约漂移（model A2） | 中 | **是**（gen 各向异性下采样 NameError，model A1） | `build_model` + 生成 hook |
| R-monitor | monitor | 编排全统一、但 `charts.py` 指标名清单 seg 硬编码，cls/det/gen/ssl 验证面板全空（monitor A1） | 中 | 否 | 任务「指标展示 spec」 |

单点确定性缺口（需优先修，独立于大重构）
| 编号 | 位置 | 问题 | 严重度 | 最小修复 |
| --- | --- | --- | --- | --- |
| G1 | gen factory.py:94,98 | 引用未导入的 `_ANISO_DOWN_MODES/_ANISO_UP_MODES`，各向异性下采样一进校验即 NameError（model A1） | 中（低频必崩） | import 该二常量，或收敛到 `build_model`（C2） |
| G2 | gen_trainer.py:285-302 | DDP + 非有限 loss 下条件反传 + 本地 group_bad + 无 all_reduce_flag_any → NCCL 挂起 / 副本发散（engine A1） | 中（低频） | 统一主循环 `_optimizer_step_boundary` 时一并消除 |

跨任务一致性欠账（不致命，统一装配器后择一即可）
| 编号 | 层 | 不一致点 | 严重度 |
| --- | --- | --- | --- |
| I1 | engine | one_cycle+warmup>0：seg/det 静默归零 vs ssl fail-fast（engine A3） | 低 |
| I2 | engine | 累积下 no_sync：仅 seg 用，cls/det/gen 每 micro-step 冗余 all-reduce（engine B2） | 低 |
| I3 | engine | gen 每步 `loss.item()` D2H 同步 vs 其余懒同步（engine A4/B3） | 低 |
| I4 | data | 单进程 train loader drop_last：seg/gen True vs cls False（data A4） | 低 |
| I5 | data | 样本→卷索引 seg/gen（idx%n_vols）vs cls（idx//spv）相反，LRU/验证铺点口径不一（data A3） | 低 |
| I6 | data | cube 提取别名安全：taskcore/gen 防御性 copy vs cls 返回缓存视图（data A2） | 低 |
| I7 | config | Optional 字段（target_spacing/val_empty_cache 等）经 --override 无法正确设值（config A4） | 低 |
| I8 | monitor | logging 顶包硬编码 "segtask_v1"，模块类别上色只对 seg 生效（monitor A2） | 低 |
| I9 | 通用 | TTA 键名 seg=`predict.tta_flip` vs cls/det/gen=`tta_flips`（改动破坏已有配置，不值得动） | 低 |

DRY 收敛项（纯工程收益，无行为变更）
| 编号 | 层 | 重复面 | 收敛方向 |
| --- | --- | --- | --- |
| D1 | config | `_coerce/_set_dotted/apply_overrides/load/save` 在 cls/det/ssl+gen 逐字重复（config B1） | `load_task_config`/`apply_task_overrides` 泛型 |
| D2 | data | `build_dataloaders` 四处整体重复（data B1）；VolumeCache 估算块复制到 gen、cls/det/ssl 缺失（data B2） | `build_task_dataloaders` + `log_volume_cache_estimate` |
| D3 | engine | 主循环 ~150 行五份（engine B1）；保存侧 `_build_state_dict/_save_checkpoint` 未上提（engine B3） | 主循环模板 + 保存侧基类模板 |
| D4 | models | gen 复述 `_resolve_decoder_counts`/`_validate_anisotropic`/装配数学（model B1） | 随 C2 收敛 |
| D5 | utils | common.py 混装通用件与 seg 指标数学（monitor B3） | seg 指标下沉 `taskcore.metrics` |

二、五任务「通用技巧 × 任务」接入矩阵复核
全员接入（已复核，不再列细节）：AMP/bf16、torch.compile、channels_last、expandable_segments、fused AdamW+ndim<=1 免 decay 分组、梯度累积/裁剪、非有限守护、EMA(+CPU offload)、warmup 调度、resume fail-fast、原子 checkpoint+history、DDP、逐 stage 梯度检查点、label_counts 快路、原子写盘监控编排。

「仅部分接入」矩阵（✓=接入，—=未接入，n/a=语义不适用）：
| 技巧 | seg | cls | det | gen | ssl | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| CudaPrefetcher | ✓ | ✓ | ✓ | ✓ | **—** | **唯一真欠账**：ssl multicrop 批量大、H2D 重，接入收益明确、改动小（核验：ssl_trainer 无 prefetch import） |
| SWA + early stopping | ✓ | ✓ | ✓ | ✓ | — | ssl 固定 schedule+已有 EMA teacher，语义存疑，维持现状 |
| 双源混采（mix_ratio） | ✓ | — | — | — | — | cls/det/gen 理论可用粗标混训，属产品决策（有无数据源） |
| AdaBN（推理 BN 域自适应） | ✓ | — | — | n/a | n/a | cls/det 域偏移时适用，可上提 taskcore；gen 意义不大 |
| UpKern pretrain 重映射 | ✓ | — | — | — | — | gen UNet 理论可吃，需要时挂 pretrain hook |
| z-interleaved 滑窗推理 | ✓ | n/a | n/a | n/a | n/a | gen SR z 是退化轴，不建议硬搬 |
| no_sync（累积免 all-reduce） | ✓ | — | — | — | ✓(手动) | 见 I2，统一主循环后全员受益 |
| 验证曲线仪表盘 | ✓ | 空 | 空 | 空 | 空 | monitor A1：非 seg 验证面板全空，需 spec 化（欠服务） |

结论：真正的技巧接入欠账只有 **CudaPrefetcher→ssl** 一项（改动小、收益明确、无争议），其余「仅 seg」项要么是产品决策（双源/AdaBN/UpKern），要么语义不适用（z-interleaved）。但「监控验证曲线」对四任务是**欠服务**（不是技巧缺失，是内容层未泛化），价值高于多数接入项。

三、抽象边界与 shim 策略评估
- **shim 策略（正面）**：seg/gen 旧路径以 `sys.modules[__name__] = _impl` 逐字别名 re-export（核验 `segtask_v1/trainer/prefetch.py`、`gentask/trainer/prefetch.py`），零行为差异、向后兼容旧 import 路径与旧 pickle。seg 12 个模型文件 / gen 10 个模型文件均为纯 shim（model D 已 grep 确认），是「实现单点收敛 + 路径兼容」的正确做法，**建议保留**（TODO3 已列：确认外部无脚本/旧 pickle 依赖后再删，目前留着无害）。
- **抽象边界（问题）**：边界画错的是**语义分层**而非路径。当前「公共 = seg 的全部」，正确边界应是「公共 = 五任务真正共用段（data/model/train/aug + 引擎装配 + 数据叶子件 + 监控编排 + 通用 utils）」，把 **loss/predict（seg 任务段）、seg 指标数学、charts 指标清单** 下沉任务层。config A2 / data B3 / model B2（`build_topology` 用 getattr 反读 gen 的 `cond_dirs`）/ monitor A1&B3 全是「边界含 seg 语义」的同一问题在各层的回声。
- **抽象做得对的层（可作 TODO3 正面参照）**：models 块库（blocks/backbone/UNet 家族全单点 + 逐字 shim，零复制）、monitor 编排（rank0 守卫 + 四处异常隔离 + 原子写盘，五任务零 fork）、data 叶子层（memmap 零拷贝/VolumeCache/RNG/patch 抽取被四任务一致 import）、utils 的 EMA/SWA/seed。TODO3 的目标应是把 config/data/engine 提升到这三者的抽象水平。

四、可扩展性总评
- **加新任务的成本现状**：需 (a) fork 或复用一个 Config 并绕开不消费的 seg 段（config），(b) 重写 `_train_epoch`（engine），(c) 独立写 Dataset + 复述 patch_mode 口径（data），(d) 复述 build_dataloaders（data），(e) 若模型偏离 seg 则 fork factory（gen 已如此），(f) 验证曲线不会自动出现（monitor）。即「共享叶子件、复述所有编排/契约」——扩展成本集中在**编排与契约的复述**，而非算法本身。
- **可扩展性瓶颈排序**：engine 主循环 > data patch/loader 装配 > config 分层 > monitor 内容层 > model factory。前两者是「每加一个任务就再复述一遍且可能漂移」的高摩擦点。
- **正向**：块库/叶子件/监控编排的单点收敛使「新算法模块、新骨干、新指标面板项」的接入成本很低（改一处全员受益）。框架的「显式装配变体模板方法」（基类只收敛构造期、不吞主循环）本身是可辩护的设计取向，问题在于**运行期编排至今未提供任何可选模板**，导致 gen 的语义偏差（G2）无处收敛。

五、2026 跨域先进做法适配优先级排序（价值×成本）
说明：价值=对五任务统一/正确性/服务面的贡献；成本=改动面与风险。列为 TODO3 候选底座的均**不建议本轮轻动**。

优先级 1（高价值 / 低成本，建议最先落地）
- **P1a 任务「指标展示 spec」**（monitor C1）：每任务给一份轻量 spec（overview 键/逐类前缀+尺度/选模键/val-loss 键），charts 据此组装。一举让 cls(AUC/F1)、det(mAP)、gen(PSNR/SSIM) 获得一等公民验证曲线+best 磁贴，消 monitor A1/B2。纯 taskcore 内收敛，不改零依赖 HTML 与用户接口。**风险最低、服务面提升最大**。
- **P1b `build_backbone` 骨干专用入口 + timm 式 feature_info**（model C1）：cls/det/ssl 不再「build 全模型丢弃头」，pretrain 同名同形契约由「同一 backbone 入口」保证。成本低、纯收敛。
- **P1c 修 G1（gen NameError）**：import 二常量的一行修复，或直接并入 C2。
- **P1d CudaPrefetcher→ssl**：唯一技巧欠账，改动小。

优先级 2（高价值 / 中成本，TODO3 主体重构）
- **P2a 任务段注册表 registry**（config C2）：公共 Config 只留五任务共用段，loss/predict 下沉 seg 任务段，gen/cls/det/ssl 各注册自己的段与 validate 钩子，`load_task_config` 统一装配。消 gen fork（R-config）+ seg 语义泄漏（A2），且不改 YAML 用户接口。C1 的轻量版。
- **P2b `_optimizer_step_boundary` 主循环模板**（engine A1/B1）：含 no_sync 上下文 + all_reduce_flag_any 跳步 + scaler.step/skip + scheduler/EMA 推进 + 懒同步健康指标。一举消 R-engine + G2 缺口 + I2/I3。
- **P2c patch dataset 模板基类 + patch 策略注册表**（data C1）：公共基类持 I/O/LRU/RNG/四 patch_mode 中心采样/2.5D 折叠；子类仅 hook 注入样本组装差异（seg label / gen cond / cls target / det boxes / ssl 视图）。消 R-data + I5/I6。
- **P2d `build_model` + 生成 hook**（model C2）：给 `build_model` 加 `decoder_overrides`/`attn_gate_target`，删 gen factory UNet 分支。一举消 G1 + R-model 四项特性缺失 + norm 漂移 + D4。模型层成本最低的统一（远小于 config/data/engine）。
- **P2e 伴随张量（companion）spec 化增强**（data C2）：MONAI MetaTensor / TorchIO Subject 式，把 label/weight/cond 声明为随 image 同 warp 的 companion，一份 spatial transform 消化全部。消 data 层最后一个实现分叉；需 GPU 固定 seed 等价性验证 + 短程 sanity。
- **P2f `build_task_dataloaders` 泛型装配器**（data B1/B2 + I4）：收敛 D2、择一 drop_last。

优先级 3（中价值 / 高成本，列为底座候选，不轻动）
- **P3a OmegaConf structured configs / Pydantic v2**（config C1/C3）：原生嵌套 override（解 I7）、类型强校验、defaults 组合（把核心段+任务段做成可组合 group，天然解 R-config/A2）、JSON schema 导出（利于 launcher 表单）。TODO3 config 底座候选。
- **P3b HF Accelerate / Lightning Fabric 承接 AMP+累积+DDP**（engine C1）：`accelerator.accumulate/backward/clip_grad_norm_` 一次覆盖 R-engine/G2/I2。价值最高但重构面最大。
- **P3c FSDP2 + torch.distributed.checkpoint（DCP）**（engine C2/C3）：gen ADM/EDM2 大骨干分片，降每卡显存、分片异步存盘。gen 大模型分支可选后端。

优先级 4（低 / 增量，随主线捎带或仅记录）
- flex_attention 稀疏窗口/网格注意力（model C3）、flow-matching/MM-DiT 扩散骨干（model C4）、seed_everything 补齐 bit 级复现开关（monitor C2）、EMA/SWA 评估前自动 BN 重估（monitor C3）、吞吐/step-time 等训练健康信号（monitor C4）、QuotaBatchSampler 泛化（data C3）、gen 统一 make_data 获得 spacing 归一化+逐类 fg 索引（data C4，需你决策重打包）、prefetch 递归搬运支持 det 嵌套 batch（engine C4）。

六、优化建议（落地顺序）
1. **先清 P1（低风险高收益）**：P1c 一行修 G1；P1d ssl 接 prefetch；P1a 指标 spec（四任务立即获得验证曲线）；P1b build_backbone。这批不需大重构、风险低、可先交付。
2. **再做 P2 五层统一（TODO3 主体）**，建议顺序：**config registry（P2a）→ engine 主循环模板（P2b，同时清 G2）→ data 模板基类+装配器（P2c/P2f）→ model build_model hook（P2d）→ augment companion（P2e，需 GPU 等价性验证）**。config 先行是因为 loss/predict 下沉是其余各层去 seg 语义（data B3/model B2/monitor B3）的前提。
3. **一致性收敛（I1–I8）随对应层统一时择一**（建议 fail-fast 风格，与 config 层一致），不单独立项。
4. **P3 底座候选**在 P2 完成、接口稳定后再评估；不建议在手写实现未统一前引入外部框架。
5. **验证要求**：任一层统一后须在 Windows GPU 环境跑全量测试；主循环统一（P2b）需 DDP 多卡 + 注入非有限 loss 的回归用例覆盖 G2；augment companion（P2e）需固定 seed 等价性 + 短程训练 sanity。

七、借鉴清单（2026 跨域，按可落地度）
- **timm 1.x** `features_only=True` + `feature_info` → P1b 骨干契约。
- **W&B/TensorBoard/MLflow/aim** metric namespace 自动分组 → P1a 指标 spec 的分组思路（自研零依赖仪表盘已够，只补 spec 层）。
- **MONAI 1.4 MetaTensor dict transforms / TorchIO Subject** → P2e companion 张量增强。
- **HF/timm/mmengine registry + OmegaConf structured / Pydantic v2** → P2a/P3a 任务段注册与结构化配置。
- **HF Accelerate / Lightning Fabric** → P3b 训练后端（累积/DDP/FSDP 统一）。
- **PyTorch FSDP2 + DCP** → P3c gen 大模型分片。
- **SD3/Flux MM-DiT、rectified flow / flow-matching、consistency 蒸馏** → gen 扩散骨干算法候选（model C4）。
- **PyTorch 2.5+ flex_attention** → 窗口/网格注意力融合核（model C3）。

自查
- 目标达成：产出全局问题总表（病根 R×5 + 单点缺口 G1/G2 + 一致性 I1–I9 + DRY D1–D5）、接入矩阵复核（唯一真欠账=ssl prefetch）、抽象边界与 shim 策略评估、可扩展性总评、2026 适配优先级（P1–P4，价值×成本）+ 优化落地顺序 + 借鉴清单。
- 依赖 A–E 均已完成并被逐条引用；承重断言（shim/接入矩阵/病根跨层复现）已代码侧核验，无冲突。
- 与 TODO2 seg 语境无重复；本轮未改任何代码/项目文档，仅在 TODO.md 追加本审查报告。
- 复核方式：所有行级引用可在对应文件核对；接入矩阵可 grep `CudaPrefetcher`/`prefetch` 逐任务复核；病根可对照 A–E 各自 A1 结论复核。



2 分割项目代码审查（需结合对应 readme/design/workflow 一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：

分割项目 = 公共框架层 `taskcore` + 任务层 `segtask_v1`，审查按此两级展开。代码大致分 5 部分，数据读取、模型构建、数据增强/处理、训练全流程（含 val）、推理全流程，先独立深度审查，再串联起来全局分析。每部分先审公共层、再审任务层。

审查主要内容为代码、算法、设计、架构、工程等等：
是否正确、合理；是否有优化空间；是否有训练加速/GPU优化空间；是否有更好的高质量内容（算法/模块/设计/架构/损失等等）可以借鉴、适配或新增。现在是2026年7月，不局限医学图像领域，可能自然图像的分类/分割/检测/生成等、NLP、LLM、VLM等有更好、更先进的想法。

进展：  



3 重构调研：由于cls/det/gen/ssl都是基于seg构建的，而且在设计上能和seg保持一致的都和seg保持一致了（可能还有不一致我未发现），能复用技巧也基本上都复用了（可能会有没有复用的我未发现）。现在我想将公用的内容抽离出来，形成一个通用的框架，然后在各个子项目中复用，如果有的模块实在做不到通用，那就例如把通用的当父类，具体的子项目当子类，继承父类的通用部分，然后重写具体的子项目部分。仍然还是大致以数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程5部分来。先认真的彻底分析和理解现有cls/det/gen/ssl/seg项目代码（需结合对应readme/design/workflow一起理解），再仔细的调研公认高质量项目的架构设计等等（不要局限医疗，可能自然图像，NLP，LLM，VLM有更好的项目）。  

进展：
代码侧的确定性欠账已经清完，剩下的都是此前列过、需要你决策或 GPU 配合的项，按价值排序：

augment 合流（设计文档已交付）：把 taskcore 增强管线泛化为"伴随张量 spec 化"，五任务共用一份实现，gen 的 cond 作为 bilinear companion 接入。这是最后一个真正的实现分叉；需要 GPU 做固定 seed 等价性验证和短程训练 sanity check。
gen spacing 归一化 + 逐类前景索引：taskcore make_data 已有、gen 未接；属功能增强（需新增 gen 配置字段、重新打包数据、dataset 端同步消费），取决于你的生成任务是否需要各向同性重采样与类均衡采样。
gen 训练/推理频谱不等价（WORKFLOW 里你标注的"已知待定"）：whole/z_axis/2_5d 训练侧 resize 到 patch、推理侧原生分辨率；候选方案是训练侧改原生分辨率裁剪，但涉及已训模型兼容，需要你拍板方向。
可选清理：确认外部无脚本/旧 pickle 依赖后删各任务 shim（目前留着无害）。
小项：ssl 未接 CudaPrefetcher、双源混采/AdaBN 仅 seg——是否接入属产品决策，不是欠账。
另外建议你在 Windows GPU 环境权威复核一遍全量测试，并对新开的 ADM/EDM2 检查点跑一小段扩散训练确认显存收益。要继续的话，1 是我建议的下一轮。

扫完接入矩阵。全员接入的（AMP/bf16、torch.compile、channels_last、expandable segments、fused AdamW+wd 分组、梯度累积/裁剪、非有限守护、EMA(+CPU offload)、warmup 调度、resume fail-fast、原子 checkpoint+history、DDP、梯度检查点(本轮补齐)、label_counts 快路）不再列。真正"只有部分项目接入"的还剩这些：

建议接入（欠账性质）

CudaPrefetcher：仅 ssl 缺（seg/cls/det/gen 都有）。SSL multicrop 批量大、H2D 拷贝重，接入收益明确，改动小——这是唯一我认为算欠账的项。
有意不接 / 需要你决策的 2. SWA + early stopping：ssl 缺。SSL 预训练通常固定 schedule、已有 EMA teacher，SWA/早停语义存疑，倾向维持现状。 3. 双源混采（npz_dir_secondary/mix_ratio）：仅 seg。cls/det/gen 理论上也可用粗标数据混训，但是否需要取决于你有没有对应数据源，属产品决策。 4. AdaBN：仅 seg（推理期 BN 统计域自适应）。cls/det 推理遇域偏移时同样适用，可上提 taskcore；gen 意义不大。 5. UpKern pretrain 重映射：仅 seg。gen 的 UNet 理论上也能吃（不同 kernel 尺寸预训练迁移），需要时可挂到已有的 pretrain hook 上。 6. z-interleaved 滑窗推理：仅 seg（按 z spacing 拆互斥子流）。gen 的 SR 任务语义不同（z 本身是退化轴），不建议硬搬。

微小不一致：TTA 配置键名 seg 是 predict.tta_flip、cls/det/gen 是 tta_flips，功能都有，仅命名不统一——改了会破坏已有配置文件，不值得。


4 模型流可视化需要有层次化，结构化，美化，可以清晰看到计算流的走向，可以清晰理解模型架构，可以清晰的溯源。总之：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判、讨厌"自动布局默认输出"式的无设计感结果。以下是一些例子：  

- 聚焦模式到stem, stage这个层级为止：
点击模块A，进入聚焦模式，模块群B和A有联系，模块群C和A没有联系，所以模块群C谈出，模块群B突显。我希望到stem，stage这个级别的模块能进入聚焦，再进一步的子模块例如stem，stage里面的子模块则不进入聚焦。  

- 连线走线需要清晰、不重叠、不交叉、美观、可以溯源：
需要清晰的看到不同模块的关系，并能溯源输入输出等等

- 位置清晰，层次清晰，严格遵守各自的位置关系：
例如输入后可能同时结果多个stem，那么这几个stem就是位置并列的；例如如果有deep supervision，且在dec level 0后有ds head 2, dec level 1后有ds head 1等等，那么ds head 2位置就应该和dec level 1并列，因为它们就是dec level 0的下一个计算。

- 其它的我暂时没有想到，请你根据我的喜好推荐，注意，原则是：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判。

进展：
