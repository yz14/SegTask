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
Step C — 数据框架层（通用性视角）

目标：审 data（dataset/loader/make_data/augment/specs/mixed_sampler）作为五任务通用数据底座的抽象边界、四 patch_mode 统一口径、2.5D 折叠契约、seg/gen 增强分叉。
产出/验收：同上。依赖：A。
Step D — 模型框架层（通用性视角）

目标：审 models（factory/topology/blocks 库/各骨干）作为通用装配层跨 seg/cls/det/gen/ssl 的复用与缝隙（呼应 TODO2 X1/X2 真相源欠账，但从框架泛化角度）。
产出/验收：同上。依赖：A。
Step E — 监控与通用工具

目标：审 monitor（jsonl+HTML/rank0 守卫/失败隔离）与 utils（common/logging_utils：seed/计量/EMA/SWA）。
产出/验收：同上。依赖：无。
Step F — 框架级横向综合

目标：五任务「通用技巧 × 任务」接入矩阵复核、抽象边界与 shim 策略评估、可扩展性总评、2026 跨域先进做法适配优先级排序（价值×成本）。
产出：全局问题总表 + 优化建议 + 借鉴清单。依赖：A-E。



2 分割项目代码审查（需结合对应 readme/design/workflow 一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：

分割项目 = 公共框架层 `taskcore` + 任务层 `segtask_v1`，审查按此两级展开。代码大致分 5 部分，数据读取、模型构建、数据增强/处理、训练全流程（含 val）、推理全流程，先独立深度审查，再串联起来全局分析。每部分先审公共层、再审任务层。

审查主要内容为代码、算法、设计、架构、工程等等：
是否正确、合理；是否有优化空间；是否有训练加速/GPU优化空间；是否有更好的高质量内容（算法/模块/设计/架构/损失等等）可以借鉴、适配或新增。现在是2026年7月，不局限医学图像领域，可能自然图像的分类/分割/检测/生成等、NLP、LLM、VLM等有更好、更先进的想法。

进展：  

Step 1 — 数据读取 目标：审 loader/dataset/make_data/specs/mixed_sampler 的样本发现/划分、pid 契约、npz mmap、四 patch_mode 抽取几何、缓存、类均衡/双源混采。产出：分层问题清单（正确性/优化/GPU/可借鉴）。验收：结论有代码行级引用、可复核。依赖：无。

Step 1 审查报告 — 数据读取
总体评价
数据读取层工程成熟度很高：npz 预烘焙 + 零拷贝 memmap + OS page-cache 跨 worker 共享、逐 worker LRU、逐 worker RNG、val 确定性采样、类均衡逐类前景索引、患者级/分层划分、DDP 三种采样器、物理几何 fail-fast 校验、原子写。未发现破坏性正确性 bug，以下按严重度列出可改进项。

A. 正确性 / 稳健性
A1（低）识别性冗余 except — derive_volume_targets @/Users/.../taskcore/data/dataset.py:430 except KeyError 实际不可达：load_npz_label_counts 对缺失 meta 已返回 None（dataset.py:410-411），不抛 KeyError。死代码，无害但误导。【已修复：删除 try/except，直接调用并注释语义】

A2（低）whole 模式 resize 恒等时的缓存别名 @/Users/.../taskcore/data/dataset.py:1307-1311 resize_3d 在形状已匹配时直接返回入参（dataset.py:540-541），此时 torch.from_numpy(img_r[None]).float() 与 LRU 缓存的 fp32 卷共享内存。当前仅靠 default_collate 的 torch.stack 分配新内存兜底；whole 模式 extract_size=round(patch*oversample) 几乎永不恒等，故实际安全。但相较 cubic/z 路径，此处防御性偏弱，属一致性瑕疵。【核验修正：cubic/z 路径真正断开别名的机制是 extractor 内部的无条件 .copy()（extract_z_patch_padded 末尾 dataset.py:1008；_extract_cubic_patch 无 padding 分支 :1044-1047），而非 ascontiguousarray（后者对已连续数组不复制，仅保连续性）；dataset.py 旧注释与实现矛盾（见 N2）。】【已修复：whole 模式恒等 resize 后对 img/lbl/rw 加 `is` 判断 + copy 断开缓存别名；非恒等路径 zoom 本就新分配，无额外拷贝开销】

A3（中，跨步）DDP 验证采样器不等长 → 需确认验证无逐 batch 集合通信 @/Users/.../taskcore/data/loader.py:40-70 ValBatchShardSampler 各 rank batch 数可不等长（block strided 切分），注释假设"指标经 all-reduce 一次汇总"。若 validation.py 存在逐 batch 的 all_reduce/barrier，会因 rank 间 batch 数不齐而死锁。MixedBatchSampler/DistributedSampler(drop_last) 等长无此风险，唯 val 分片不等长。留作 Step 4 重点核对项。

B. 优化空间（性能 / 内存）
B1（中）LRU 缓存存 fp32 预处理卷，内存翻倍 @/Users/.../taskcore/data/dataset.py:749-753 _img_cache 缓存的是 preprocess_image 后的 fp32（4B/voxel），是原始 int16（2B）的 2×。loader.py:924-973 已有 RAM 估计与 OOM 告警，但可提供"缓存 int16 原始卷、取用时再 preprocess"选项，以 CPU 换 RAM（大数据集/多 worker 更划算）。属可配置增强，非缺陷。

B2（中）CPU 端 scipy.ndimage.zoom 面内/整卷 resize 是潜在吞吐瓶颈 @/Users/.../taskcore/data/dataset.py:554（resize_3d），z_axis 面内 resize dataset.py:928、whole 整卷 dataset.py:1307 每样本每 epoch 在 worker 内做一次 zoom。设计上数据管线纯 CPU（GPU 增强在 trainer 侧），此为合理取舍，但大 H/W 时是主要 CPU 成本。可评估：H/W 面内 resize 改逐切片更快的后端，或将 resize 后移到 GPU（需打破当前 worker 契约，权衡较大）。列为观察项。

B3（低）--compress 牺牲 page-cache 共享 @/Users/.../taskcore/data/dataset.py:264-266 压缩 npz（ZIP_DEFLATED）走不了零拷贝 memmap 快路，回退 zipfile 流式整卷解包，且 worker 间不共享。已在文档标注取舍。见 C2 的现代替代方案。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中）nnU-Net 各向异性 spacing 指纹 @/Users/.../taskcore/data/make_data.py:343-364 _resolve_target_spacing 取逐轴中位数。nnU-Net v2 对高各向异性轴改用 spacing 的高百分位（如第 10 百分位）而非中位数，避免沿低分辨率轴过度上采样导致内存/伪影。CT z 厚层场景常见，值得适配。

C2（中）blosc2 分块压缩替代 npz — 呼应 B3 nnU-Net v2 已从 npz 迁移到 blosc2 压缩 npy：兼得压缩率与近 mmap 的随机块访问（跨进程共享解压缓存）。可同时解决"压缩 vs page-cache 共享"的二选一。属存储层升级，改动较大，列为候选。

C3（低）多标签分层划分 @/Users/.../taskcore/data/loader.py:463-475 _volume_primary_class 以"最大体素前景类"做单标签分层，多器官共存时偏粗。可引入 iterative multilabel stratification（基于 label_counts 已有的逐类计数，零额外 I/O）提升小类在 train/val 两侧的代表性。

Step 2 — 模型构建 目标：审 factory/topology/unet/blocks/stem 及各骨干的装配正确性、拓扑真相源一致性、深监督/aux/条件通道接线、数值/显存点。产出：同上分层清单。依赖：Step 1 的几何契约理解。

Step 2 审查报告 — 模型构建（公共层 taskcore.models；segtask_v1.models/* 全为 shim 转发，无独立实现）

总体评价
模型层工程成熟度高、2D/3D dim-agnostic（_CONV/_BN/_IN 等分派表，*3D 命名仅 API 兼容）。ModelTopology 作为几何/通道派生单一真相源，主 UNet 路径完全遵守（build_model 全量读 topo，不再自算）；Config.sync 也经 build_topology 写 in_channels/spatial_dims 只读 property。block/attention/上下采样库丰富（ResNet 系 basic/preact/bottleneck/r2plus1d、ConvNeXt、MedNeXt+DilatedReparam/UpKern、SE/ECA/CBAM/Coord/LKA/MSCA、softmax/linear/window/grid 自注意力+RoPE、BlurPool/PixelShuffle/CARAFE/DySample、各向异性下采样调度带兼容性 fail-fast、逐 stage 梯度检查点）。深监督/多 FOV aux/topo aux/cond 通道布线清晰，DecoderLevel/Encoder 均有尺寸 fail-fast。未发现破坏性正确性 bug；以下按严重度列出可改进项（已排除 config.validate 已拦截的误配路径）。

A. 正确性 / 稳健性
A1（低-中，跨真相源）ADM/EDM2 build 未消费 topology，本地重算几何 —— @/Users/.../taskcore/models/adm_unet.py:766-768,814-820,859 与 @/Users/.../taskcore/models/edm2_unet.py:661-663,704-710,728 均调用了 build_topology(cfg) 却仅取 topo.cond_in_channels，其余 out_classes(=num_fg*D)、n_views、aux_seg、in_ch_per_view_list、aux_head_out_channels 全部就地重推。这正是 topology.py 文档（@/Users/.../taskcore/models/topology.py:3-16）声明 R5 要消除的「多处各算一遍」模式。当前因 config.validate 已硬拒 lift+adm/edm2（core.py:1930-1934）、拒 hierarchical（adm_unet.py:805、edm2_unet.py:697），重算值恰与 topo 等价、无现存 bug；但新增 patch_mode/几何时此处最易漏改。建议对齐主 UNet（@/Users/.../taskcore/models/factory.py:334-339）改读 topo.out_classes / topo.aux_seg_active / topo.num_stem_fusion_views / topo.in_ch_per_view_list / topo.aux_head_out_channels。属可维护性欠账。【已修复：ADM/EDM2 build 改读 topo.slab_depth / out_classes / n_views / in_ch_per_view_list / aux_head_out_channels / aux_seg_active，仅保留 2_5d 硬拒与 hierarchical 拒绝等受限契约检查；日志用 recon in_channels = topo.in_channels - cond_in_channels】

A2（低，潜在契约缺口）factory 未把 cond 通道透传给通用 Encoder —— @/Users/.../taskcore/models/factory.py:450-466 构建 Encoder 时未传 cond_in_channels，而 topo.in_channels 已把 cond_in_channels 计入（topology.py:93,115,122,127）。分割当前无 cond_dirs 字段（topo.cond_in_channels=0）故无碍；但若日后给 seg 接 cond，Encoder 会把末尾 cond 通道当作 recon 主输入、不建 cond stem/cond_fuse（Encoder 已具备该能力，unet.py:107-124），静默错配。ADM/EDM2 反而显式传了 cond。建议 factory 显式传 cond_in_channels=topo.cond_in_channels，或加断言 topo.cond_in_channels==0。【已修复：factory 构建 Encoder 时显式透传 cond_in_channels=topo.cond_in_channels；topo.in_ch_per_view_list 只计 recon 通道，与 Encoder 内 recon_in_channels 校验自洽；seg 当前 cond=0 行为不变】

A3（低，一致性瑕疵）ConvNeXt/MedNeXt 块内 make_attention 未透传 norm/reduction 配置 —— @/Users/.../taskcore/models/convnext.py:58 与 @/Users/.../taskcore/models/mednext.py:424 调 make_attention(type, dim, spatial_dims=d)：①未带 reduction，故 attention_type='se'/'cbam'/'coord' 时 model.se_reduction（core.py:371）被静默忽略、恒用默认 16/16/32（核验新发现 N1）；②未带 norm_type/norm_groups，'coord' 时 CoordAttention3D 内部归一化恒用默认 group/8（blocks.py:450-454）。ResNet 系（resnet.py:42-43 等）已正确透传。【已修复（reduction 部分）：ConvNeXt/MedNeXt 块链路新增 attn_reduction 参数，factory 透传 mc.se_reduction；norm 有意不透传——两骨干块内 norm 本就固定（LN / 通道级 GroupNorm，factory 已有“IGNORED inside blocks”告警），coord 内部归一化保持默认 group/8 与该设计一致，已在调用处注释】

B. 优化空间（性能 / 显存 / GPU）
B1（中，数值稳健）手写 LayerNorm/GRN 统计在 AMP 低精度下计算 —— @/Users/.../taskcore/models/convnext.py:27-29（LayerNorm3d 的 mean/var）与 @/Users/.../taskcore/models/blocks.py:97-99（GlobalResponseNorm 的 sqrt(Σx²)）跟随输入 dtype，autocast 下即 fp16/bf16，大特征图上 L2/均值累加数值不如 fp32 稳。项目他处（unet.py:26-32 _resize_logits、blocks.py:1458-1466 Upsample 插值）已有「fp16→fp32→回原 dtype」范式，建议这两处对齐上采到 fp32 再回写。低成本稳健性提升。

B2（中，torch.compile 友好性）RoPE 路径不利于图捕获 —— @/Users/.../taskcore/models/blocks.py:35-36,527-542 用全局可变 LRU dict 缓存 cos/sin，@/Users/.../taskcore/models/blocks.py:568-591 每次 forward 现算 meshgrid，并 q.clone()/k.clone() 后就地切片写回。全局 dict 变更 + 就地写会触发 torch.compile graph break / 重编译，clone 亦增峰值显存。selfattn_rope 打开时训练可能拿不到 compile 收益。建议：坐标/inv_freq 预注册为按形状 lazy 的 buffer、改用非就地 rotate 组装。仅影响开 RoPE 的深层小特征图，范围可控。

B3（低，显存）CARAFE3d 邻域展开显存放大 —— @/Users/.../taskcore/models/blocks.py:1272-1285 三次 unfold 把 k³ 邻域捏入通道，中间张量约 C·k³×。已默认 k_up=3 且文档注明「3D 中 k_up=5 过重」，列为观察项；大分辨率上采用时优先 DySample（blocks.py:1289 更轻）。

B4（低）patch-embed/下采样投影多为 1×1 conv 接 norm（blocks.py:1200-1225），channels_last 内存格式收益依赖 trainer 侧设置（本层未强制，属正确分层）；无需改动，仅记录。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中）主 seg_head 单层 1×1（unet.py:462-464）符合 nnU-Net 标准；可提供可配置的更强 head（如 SegNeXt Hamburger / 轻量 MLP head）作为消融选项（DS 头已用 3×3+1×1 的 ConvSegmentationHead，unet.py:469-476）。优先级低。
C2（中）上采样已含 CARAFE/DySample；可补 2024-2025 的 feature-aligned 上采样（SAPA / FADE 系）作消融，对细管/边界对齐更好，接口可复用现有 Upsample.VALID_MODES 分派（blocks.py:1380-1381）。
C3（低）自注意力已覆盖 window/grid(MaxViT 式)+linear+RoPE；可补显式 flash/SDPA 后端选择或 3D Neighborhood Attention(NATTEN) 作为局部注意的高效实现，减小 O(N²) 深层成本。
C4（低）MedNeXt 已含 UpKern 重映射与 DilatedReparam（mednext.py:145-383）；后续可落地档位 B「重采样残差块」（文件头 TODO 已列）与 UniRepLKNet 的 SE-before-reparam。

验证方式
本轮为纯静态审查（不改代码/文档），结论均带行级引用、可复核；并对照 config.validate 契约确认 A1/A-lift 等误配已被拦截。建议在 Windows GPU 环境跑既有模型单测复核：tests/test_model_flow.py、test_model_topology.py、test_a5_blocks.py、test_attention.py、test_selfattn.py、test_mednext.py、test_multirf.py、test_anisotropic_downsample.py、test_adm_edm2_seg_smoke.py、test_grad_checkpointing.py、test_aux_seg_supervision.py、test_lift_aux_ds.py、test_keep_native_view_depth.py。若采纳 A1/A2/B1，改动局限于 adm_unet.py/edm2_unet.py 的 build 函数与 factory.py/convnext.py/mednext.py/blocks.py 少量行，风险低且被上述单测覆盖。

Step 3 — 数据增强/处理 目标：审 augment.py GPU 同步空间/强度变换、img/lbl/wmap 同步性、oversample 余量与折叠时机契约、归一化/spacing 重采样。产出：同上。依赖：Step 1。

Step 3 审查报告 — 数据增强/处理
目标复述：审 augment.py 的 GPU 同步空间/强度变换、img/lbl/wmap 同步性、oversample 余量与折叠时机契约、归一化/spacing 重采样。遵守规则：本轮不动任何代码/文档，纯静态审查，结论均带行级引用、可复核。

审查范围：公共层 taskcore.data.augment（seg 经 @/d:/codes/work-projects/SegTask/segtask_v1/data/augment.py:1-8 shim 转发，任务层无独立实现）；调用侧 trainer.py + views.center_crop；归一化 preprocess_image 与 spacing 重采样（offline，make_data）。

总体评价
增强层设计成熟：affine+elastic 融合为单次 grid_sample（G(x)=Θ(x+d)=Θx+M·d，@/d:/codes/work-projects/SegTask/taskcore/data/augment.py:270-278），省一轮重采样且免双重插值模糊；选样掩码与标量参数全部 CPU 采样（_bernoulli_mask，@/d:/codes/work-projects/SegTask/taskcore/data/augment.py:20-22）以规避 device→host 隐式同步；越界填充语义正确（label→label_fill、wmap→1.0，@/d:/codes/work-projects/SegTask/taskcore/data/augment.py:287-294）；img/lbl/wmap 空间同步（flip + affine/elastic 三者同 warp，grid-dropout/强度仅 image）契约清晰；oversample 折叠时机正确（增强在过采样 cube 上、之后再 center_crop 回切，@/d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:452-456）；且增强整体运行在 autocast 之外（@/d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:453 vs :477），grid_sample/interpolate 在 fp32 下计算，无低精度插值隐患。未发现破坏性正确性 bug，以下按严重度列出可改进项。

A. 正确性 / 稳健性
A1（低）_gaussian_blur_3d 无 sigma>0 守卫 — @/d:/codes/work-projects/SegTask/taskcore/data/augment.py:443 中 x[None,:]/sigmas[:,None]，sigmas 由 uniform_(sigma_range[0], sigma_range[1])（:438-439）采样。【核验修正：触发条件比原述苛刻——下界配 0 时 uniform_ 采到精确 0 的概率极低（接近 0 的 sigma 数值上仍安全，核退化为归一化 delta）；确定性危险的是 sigma_range=[0,0] 或负值配置。】_validate_augment（@/d:/codes/work-projects/SegTask/taskcore/config/core.py:1738-1771）仅校验 wmap_interp_mode/per_axis/translate_range 长度，未见对 gaussian_blur_sigma、elastic_deform_sigma、gaussian_noise_std 正性的硬校验。默认 [0.5,1.5] 安全，属误配防护缺口。建议在 _validate_augment 补 sigma/std/zoom 正性与区间校验。【已修复：_validate_augment 新增 gaussian_blur_sigma（0<lo<=hi）、elastic_deform_sigma（>0）、elastic_deform_alpha（>=0）、gaussian_noise_std（>=0）、simulate_lowres_zoom（0<lo<=hi<=1）、random_gamma_range（0<lo<=hi）硬校验；prob=0 禁用项不受影响】

A2（低，一致性）brightness/contrast 硬编码 rank-5 形状，与同文件 dim-agnostic 风格不一致 — @/d:/codes/work-projects/SegTask/taskcore/data/augment.py:358（torch.empty(B,1,1,1,1)）与 :374-376 假定 image.ndim==5，而 _random_gamma/_gaussian_noise/_random_contrast 的 reduce 用动态 range(2, image.ndim)（如 :372、:391）。seg 增强入口恒为 5D（2.5D 深度折叠推迟到增强之后，见 @/d:/codes/work-projects/SegTask/ssltask/trainer/ssl_trainer.py:123-125），故无现存 bug；但 gen fork 已用 unsqueeze/squeeze 兜 4D（@/d:/codes/work-projects/SegTask/gentask/data/augment.py:60-66,116-121），公共层若未来复用到 2D/4D 入口会静默错配。属一致性欠账。

A3（中，确定性）增强 RNG 跨 CPU/CUDA 两套全局生成器，"固定 seed 等价性"复现脆弱 — Bernoulli 掩码与逐样本标量（角度/scale/gamma/brightness）走 CPU 全局 torch.rand/uniform_（@/d:/codes/work-projects/SegTask/taskcore/data/augment.py:22,248,398），而弹性位移 torch.randn(...,device)（:181）与 grid-dropout torch.randint(...,device)（:322-324）走 CUDA 全局生成器。TODO 进展里标注的"固定 seed 等价性验证"（@/d:/codes/work-projects/SegTask/TODO.md:138）需同时锁定两套 RNG，且不与训练循环其它 torch.rand 交错，复现依赖全局状态、较脆。建议 GPUAugmentor 持有独立 torch.Generator（CPU+CUDA 各一）并在所有采样点显式传入，隔离增强随机流——这正是等价性验证落地的前置。

A4（低，取舍非 bug）纯 affine（无 elastic）样本旋转角点：image 走 border 复制、label 置背景，二者语义不一致 — @/d:/codes/work-projects/SegTask/taskcore/data/augment.py:283-288 image 用 padding_mode='border'，而越界点 label 被强制 label_fill（背景）。这是刻意取舍（避免边缘外推伪前景，对齐 nnU-Net 常数填充），但会给模型"有复制纹理却标背景"的监督点。当前靠 oversample 余量把角点裁掉（center_crop）缓解；z_axis 模式 H/W 无余量（@/d:/codes/work-projects/SegTask/taskcore/config/core.py:235-236）时角点会留在 patch 内。记为观察项。

B. 优化空间（性能 / 显存 / GPU）
B1（中，显存）默认 inplace=False 在过采样 cube 上多克隆一份 img/lbl/wmap — @/d:/codes/work-projects/SegTask/taskcore/data/augment.py:52-56 入口 clone。而训练循环的输入本就是 H2D 私有拷贝（batch["image"].to(device, non_blocking=True)，@/d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:444-448，增强后不再以原值复用），恰好满足 AugConfig.inplace 契约（@/d:/codes/work-projects/SegTask/taskcore/config/core.py:201-205）。aug_oversample_ratio>1 时这份克隆按过采样体积放大。建议 seg/ssl/gen trainer 侧显式开 inplace=True（或在文档明确推荐），省一份过采样 cube 的瞬时显存；保持 config 默认 False 作防御。【已修复（seg）：GPUAugmentor 新增 inplace 覆写参，seg trainer 传 inplace=True；ssl/gen 未动，待各自调用方确认所有权后同样接入】

B2（低，显存）elastic 位移场在过采样全尺寸上生成 fp32 (n,3,D,H,W) — @/d:/codes/work-projects/SegTask/taskcore/data/augment.py:181-182 粗网格 randn 后 interpolate 到过采样 (D,H,W)，峰值随选中样本数线性增长。空间增强必须在裁剪前完成（否则边缘伪影入 patch），故无法把这步后移到 center_crop 之后——取舍合理，列为观察；可评估 disp 融入 grid 后（:277）提前 del 以压峰值。

B3（低，GPU）_grid_dropout 逐洞 Python 循环 + 高级索引 scatter — @/d:/codes/work-projects/SegTask/taskcore/data/augment.py:330-340 对 num_holes 做 Python 循环，每洞广播出 (B,hd,hh,hw) 索引张量写 hole_mask。默认 grid_dropout_prob=0（@/d:/codes/work-projects/SegTask/taskcore/config/core.py:247）未启用，低优先；启用时可改为一次性向量化构造掩码。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中，跨步 TODO3）增强实现的 seg/gen 双分叉应收敛为"伴随张量 spec 化" — taskcore.data.augment 与 @/d:/codes/work-projects/SegTask/gentask/data/augment.py:1-122 是两份近重复实现：gen 去掉 label、把 cond 作为 bilinear companion 同 warp（@/d:/codes/work-projects/SegTask/gentask/data/augment.py:282-285），seg 则保留 label（nearest）+ OOB 语义。这正是 TODO 进展标注的"最后一个真正的实现分叉"（@/d:/codes/work-projects/SegTask/TODO.md:138）。建议把 (image, label, wmap) 泛化为 primary + companions[spec]（每 companion 携 interp_mode/oob_fill/受强度增强 标志）：一份实现覆盖 seg 的 label/wmap 与 gen 的 cond，也让 seg 未来接 cond/多标签更干净。属 TODO3 重构范畴，此处标注。

C2（中）强度增强对 zscore 归一化不自适应 — brightness/noise 幅值为绝对量、隐含 image≈[0,1]（@/d:/codes/work-projects/SegTask/taskcore/config/core.py:252-253）。而 normalize 可为 zscore（@/d:/codes/work-projects/SegTask/taskcore/data/dataset.py:489-492），此时同幅值扰动量纲不符、偏弱。建议幅值按归一化模式自适应（zscore 时以 σ 为单位），或在 config.validate 当 normalize=='zscore' 且沿用默认绝对幅值时告警。低成本稳健性提升。

C3（低）spacing 校正只在 offline 做，aspect_correct 仅纠正 voxel-count 各向异性 — random_affine_aspect_correct 在 voxel 计数坐标里旋转（R←A⁻¹RA, A=diag(W,H,D)，@/d:/codes/work-projects/SegTask/taskcore/data/augment.py:160-162），文档明确"不代替真实 spacing 校正"（:133-134）。当 make_data 未做各向同性重采样（spacing_normalized=False，@/d:/codes/work-projects/SegTask/taskcore/data/dataset.py:447-448）时，物理各向异性下的旋转仍混入剪切。呼应 Step 1 C1（nnU-Net v2 高各向异性轴用高百分位 spacing）。建议：文档提示"开 affine 旋转时优先启用 spacing 归一化"，或让 aspect 用物理 spacing 而非纯 voxel 计数。

C4（低）现代空间增强消融位 — 当前 elastic 为粗网格 randn 上采位移场；可选补 MONAI RandGridDistortion/thin-plate-spline 作 elastic 消融；_simulate_lowres 各轴同 zoom（@/d:/codes/work-projects/SegTask/taskcore/data/augment.py:482），可补 nnU-Net v2 的逐轴各向异性 lowres（对 CT 厚层更贴合退化）。样本间 MixUp/CutMix 当前无（grid-dropout 仅单样本 cutout），医学分割争议大，作为 cls/det 可选项可上提 taskcore。低优先。

Step 4 — 训练全流程（含 val） 目标：审 base_trainer+seg trainer/validation/pipelines、AMP/EMA/SWA/调度/累积裁剪/DDP/ZeRO/checkpoint/prefetch/grad-ckpt、损失库（Dice/Focal/Tversky/GDL/clDice/topo）与多分辨率/深监督分解、val 双口径。产出：同上，含训练加速专项。依赖：Step 1-3。

Step 4 审查报告 — 训练全流程（含 val）
目标复述：审 base_trainer + seg trainer/validation/pipelines、AMP/EMA/SWA/调度/累积裁剪/DDP/ZeRO/checkpoint/prefetch/grad-ckpt、损失库（Dice/Focal/Tversky/GDL/clDice/topo）与多分辨率/深监督分解、val 双口径。含训练加速专项。遵守规则：本轮不动任何代码/文档，结论均带行级引用、可复核。

审查范围：公共层 taskcore.engine（base_trainer/optim/amp/prefetch/checkpoint/bn_stats/dist_utils）；任务层 segtask_v1.trainer（trainer/validation/breakdown/pipelines/*）与 segtask_v1.losses（losses/topo_aux）。

总体评价
训练层工程成熟度极高，多数"魔鬼在细节"的正确性点都处理到位：模板方法式显式装配（_setup_* 由子类按序调用，见 @d:/codes/work-projects/SegTask/taskcore/engine/base_trainer.py:71-179）；调度器按"优化步"而非"尝试步"推进、one_cycle 关闭外层 warmup 避免双叠加（base_trainer.py:90-113）；梯度累积尾批用真实尾长作分母（base_trainer.py:199-207）；fp16 走 GradScaler 跳步、bf16/fp32 用 all-reduce(any) 统一的非有限守护，且 scheduler/EMA 仅在 optimizer 真更新后推进（@d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:509-601）；损失强制 fp32 + logit clamp±50 防 fp16 溢出（@d:/codes/work-projects/SegTask/taskcore/engine/amp.py:86-96）；未缩放 loss 延迟到日志步单次 stack+tolist 取回、免每 micro-step 一次 D2H（trainer.py:411-435）；val 混淆量 float64 累加 + 单次 all-reduce、medium/high 同结构（@d:/codes/work-projects/SegTask/segtask_v1/trainer/validation.py:136-213）；async 原子 checkpoint、ZeRO consolidate 早于 rank 早退（trainer.py:841-873）。未发现破坏性正确性 bug。以下按严重度列出可改进项，并先解决 Step 1 留待本步的跨步项。

A. 正确性 / 稳健性
A0（跨步项，Step 1-A3 结论：安全，非 bug） — DDP 验证不等长采样器无死锁风险。Step 1 担忧 ValBatchShardSampler 各 rank batch 数不等长会与逐 batch 集合通信死锁。核对验证路径：PatchValEvaluator 逐 rank 全量迭代本地 loader，仅在末尾一次 acc.all_reduce（@d:/codes/work-projects/SegTask/segtask_v1/trainer/validation.py:389-412）；VolumeValEvaluator 用 shard_for_rank 切整卷列表后同样只末尾一次 all-reduce（validation.py:483-538）。验证全程无逐 batch collective/barrier，故不等长不会死锁。Step 1-A3 可关闭。

A1【核验后撤销：结论不成立】 — 原述“fp16+GradScaler 路径下 nonfinite_steps 健康计数会漏计”有误：pending 列表在非日志步并不清空，每个 micro-step 的 loss 都会 append（trainer.py:496）；_flush_pending 在下一个日志步（:499）与 epoch 末（:654）处理全部积压条目并逐条判 isfinite 累加 nonfinite_steps（:421-426），该计数在 epoch 末上报（:665）前已完整。fp16 下计数仅延迟到日志步/epoch 末，不会漏计；无需修改。

A2（低，可维护性） — seg 重实现了 _finalize_swa/_swa_recalibrate_bn（trainer.py:708-791），与 BaseTrainer 的回调式通用版本（base_trainer.py:397-467）并行存在。base 版本被其它任务复用（非全局死代码），但 seg 未走 base 的 validate_fn/bn_forward_fn/run_forward 注入口径，两份逻辑几乎等价却各自维护——SWA 收尾若日后修一处易漏另一处。建议 seg 改为提供 bn_forward_fn/validate_fn 闭包复用 base 版本。【已修复：删除 seg 的 _finalize_swa/_swa_recalibrate_bn 重实现，改为传 validate_fn=evaluator.evaluate 与 _swa_bn_forward 闭包调 base 版（与 cls/det/ssl/gen 同构）；DDP BN 聚合由 base 版 _swa_recalibrate_bn 内的 all_reduce_bn_running_stats_ 保留】

A3（低，冗余同步） — 日志步存在重复 D2H。compute_loss 在 breakdown 分支写 breakdown["L_total"]=loss.detach().item()（如 @d:/codes/work-projects/SegTask/segtask_v1/trainer/pipelines/slab25d.py:121），而同一日志步 _flush_pending 又对同一 loss 做 stack().tolist()（trainer.py:418）。两次同步都只在日志步发生、开销可忽略，但可合并（L_total 由 pending 值回填）。属一致性瑕疵。

B. 优化空间（性能 / 显存 / GPU）
B1（低-中，热路径 H2D） — 损失内每次 forward 重建 fg_values 张量，触发逐步微小 H2D。MultiResolutionLoss._label_to_binary（@d:/codes/work-projects/SegTask/segtask_v1/losses/losses.py:784）、SliceChannelLoss._label_to_binary/_label_to_binary_5d（losses.py:868、losses.py:942）均 torch.tensor(self.fg_values, device=label.device, ...)。这在每个训练步（主路 + 每个 aux 视图）执行，从 Python list 构造并落 device，是可能的隐式同步点。建议构造时 register_buffer("fg_values", ..., persistent=False) 一次性驻留，forward 直接复用。低成本、纯提速。【已修复：改为首次 forward 惰性构建设备张量后缓存复用（_fg_tensor）；未用 register_buffer，因 criterion 不随 model .to(device)，buffer 会永驻 CPU 仍逐步 H2D】

B2（中，训练加速专项） — val 成本可占 epoch 相当比例，且无"低频高保真"档。trainer 已逐 epoch 打印 val 占比（trainer.py:321-329），说明作者已察觉。val_metric_mode 为全局单选：medium 每 epoch 逐 batch loss.item()（validation.py:404，每 val batch 一次 D2H）、high 每 epoch 对全部 val 卷重载 npz + 滑窗（validation.py:483-495，Predictor 已缓存但整卷每 epoch 重读重预处理）。建议加"medium 每 epoch + high 每 N epoch/末段"的混合验证调度，显著降 high 模式 val 墙钟；high 模式若 RAM 允许可缓存预处理整卷。属训练总时长优化。【已修复（调度部分）：新增 train.val_high_interval（默认 1=既有行为），>1 时 HybridValEvaluator 每 N 次验证跑一次 high（及末 epoch 必跑）、其余轮次跑 medium 监控；选模/早停/plateau 只看 high 轮次（selects_model 门控，避免口径混用污染 best）；高轮次按 epoch 推导，resume 相位不变。整卷 RAM 缓存未做（内存权衡需实测）】

B3（低，训练加速已到位，记录） — DDP 梯度重叠、no_sync 非边界步免 all-reduce（trainer.py:466-468）、fused AdamW（@d:/codes/work-projects/SegTask/taskcore/engine/optim.py:87-91）、ZeRO-1（optim.py:54-101）、TF32/cudnn.benchmark 经 seed_everything 默认开（taskcore/utils/common.py:586-589）、channels_last、CudaPrefetcher、grad-ckpt 均已接入。batch-pooled 损失（batch_dice/GDL/Tversky）在 accum/DDP 下统计窗口收缩已显式告警（base_trainer.py:506-511）——为已知取舍，非缺陷。无需改动，仅记录加速面已充分。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中）blob loss / instance-aware 损失 — 本项目主打细管/栓子等大量小连通前景，当前损失库为逐类语义 Dice/Tversky/GDL/clDice（losses.py），缺 instance-level 项。blob loss（Kofler+ 2022/2023）对多个小病灶逐实例加权，缓解"大病灶主导 Dice、小病灶被淹没"。可作为可配置复合项接入现有 _COMPOUND_BUILDERS（losses.py:705-715）。

C2（中）Top-k / OHEM 复合项 — 稀疏前景下 CE/BCE 梯度被海量易背景稀释；nnU-Net 可选 Top-k CE、OHEM 聚焦难体素。当前 BCELoss/BinaryFocalLoss 走全体素归一化均值（losses.py:169-204）。可加 topk/ohem 复合项（仅取误差最大的 k% 体素求均），config 级新增、与现有工厂契合。

C3（中）现代优化器作可选档 — 现支持 adamw/adam/sgd（optim.py:80-102）。可补 Schedule-Free AdamW（Defazio 2024，弱化 LR schedule 依赖）或 Muon/SOAP（2024-2025，收敛更快）作 opt-in，接入 build_optimizer 分派即可，不改训练循环。

C4（低）损失权重退火 — topo/clDice/aux 权重当前为常数（slab25d.py:167、@d:/codes/work-projects/SegTask/segtask_v1/trainer/pipelines/base.py:139）。可借鉴多任务训练常用的"warmup/anneal 辅助权重"（早期低、后期升），减轻早期拓扑/中心线目标噪声干扰主分割收敛。低优先。

C5（低）CompoundLoss 组件级并行/复用 — CompoundLoss 顺序累加各分量（losses.py:274-282），dice_bce 会各自 sigmoid/展平一遍。可在复合层预算一次 sigmoid/flatten 复用给多个重叠型分量，省重复算子。收益小，仅深层大特征图明显。

Step 5 — 推理全流程 目标：审 base_predictor+seg predictor/sliding/forwards/inputs/blending/adabn/io、四模式滑窗几何与训练镜像一致性、TTA/blend/z-interleave/AdaBN、显存逃生门、反归一化/写出。产出：同上，含训推一致性专项。依赖：Step 1-4。

Step 5 审查报告 — 推理全流程
目标复述：审 taskcore.engine.base_predictor + seg predictor/sliding/forwards/inputs/blending/io（adabn.py 为 shim 转发 taskcore.engine.bn_stats），四模式滑窗几何与训练镜像一致性、TTA/blend/z-interleave/AdaBN、显存逃生门、反归一化/写出，含训推一致性专项。遵守规则：本轮纯静态审查，结论均带行级引用、可复核。

审查范围：公共层 @/d:/codes/work-projects/SegTask/taskcore/engine/base_predictor.py + bn_stats.py；任务层 segtask_v1/predictor/{predictor,sliding,forwards,inputs,blending,io}.py；训推镜像对照 dataset.py（_getitem_max_fov/extract_z_patch_padded/_extract_cubic_patch）与 segtask_v1/trainer/validation.py 的 VolumeValEvaluator。

总体评价
推理层工程成熟度极高，训推镜像一致性扎实：四模式的推理窗口几何与训练侧一一对应——whole 走整卷 resize（@/d:/codes/work-projects/SegTask/segtask_v1/predictor/sliding.py:46 vs 训练 dataset.py:1307）、cubic/z_axis 走 max-FOV 中心抽取 + edge-replicate 短尾填充（inputs.py:_extract_z_slab_resized/_edge_pad_axis vs dataset.py:_extract_cubic_patch/extract_z_patch_padded）、2.5D 折叠 C_res*D 与 Trainer._squeeze_2_5d 同布局（forwards.py:44-55）、z_boundary_mode 从 data 配置同步以保边界几何一致（predictor.py:199）。TTA 批量化与逐变体串行严格等价且累加同序（forwards.py:90-113）、2.5D 只翻 H/W 不翻 D 通道轴（forwards.py:70-74）、概率强制 sigmoid(pred.float()) 上采 fp32（forwards.py:120,237,281）、NaN 体素强制背景并 logger.error 溯源（blending.py:100-112）、prob_to_label 阈值语义（严格 > 取前景）与验证侧 prob > threshold 契约一致（blending.py:114-133 vs validation.py:511）。显存逃生门齐全：acc_dtype/accumulate_on_cpu/vol_dtype fp16 常驻 + 按窗升 fp32、skip_empty_windows、tta_batch_size、channels_last、use_inference_mode、val_empty_cache（predictor.py:92-117、trainer.py:689-703）。ckpt 加载稳健（compile 前缀剥离/EMA 拆包/形状预校验/半数参数硬拒随机权重，io.py:25-158）。AdaBN global/per_volume 双模式 + 估计期强制 TTA 串行以保 BN 统计构成一致（forwards.py:77-87、predictor.py:316-319）。config.validate 对 threshold/overlap/acc_dtype/vol_dtype/z_interleave（含 thresholds 升序、factors 长度）/AdaBN 均有硬校验（core.py:2093-2181）。未发现破坏性正确性 bug，以下按严重度列出可改进项。

A. 正确性 / 稳健性
A1（低，误配防护缺口）predict.blend_mode 无枚举校验，拼写错误静默退化为均匀权重 — build_1d_weight（@/d:/codes/work-projects/SegTask/segtask_v1/predictor/blending.py:55）与 build_3d_weight（blending.py:65-71）均以 mode == "gaussian" 为唯一特判，其余一切字符串走 else 返回全 1（均匀 average）。core.py:2093-2181 的 predict 校验覆盖了 threshold/overlap/dtype/z_interleave/adabn，唯独没有 _require(blend_mode in {"gaussian","average"})。用户若把 blend_mode 误写为 "guassian"/"gauss"，会静默丢失高斯重叠融合、退化为均匀平均（拼缝处更易出现阶梯伪影），且无任何告警。建议在 _validate_predict 补一行枚举校验。属误配防护缺口，非现存 bug。【已修复：_validate_predict 新增 blend_mode ∈ {gaussian, average} 枚举校验】

A2（低，训推一致性覆盖缺口）val（high）路径从不覆盖 spacing 镜像重采样代码 — VolumeValEvaluator 直调 predict_preprocessed_array（@/d:/codes/work-projects/SegTask/segtask_v1/trainer/validation.py:493），吃的是 make_data 已按 target_spacing 烘焙的 npz 卷；而 CLI 部署 predict_volume 才走 resample_to_spacing 前处理 + 概率回采（@/d:/codes/work-projects/SegTask/segtask_v1/predictor/predictor.py:432-486）。二者对已烘焙数据自洽（label 同 npz 分辨率），但意味着训练期 high 选模指标永不经历部署侧 resample 的两次重采样数值误差（原生→target→原生插值损失），选出的 best 模型与真实部署之间存在一处未被验证的细微口径差。开 spacing_normalization 时值得留意。观察项。

A3（低，诊断护孔复位职责单点）_diag_first_batch_logged 仅由 predict_volume 复位 — 初值 True（predictor.py:314），仅 predict_volume 顶部置 False 触发一次 forward 诊断（predictor.py:473）。VolumeValEvaluator 直调 predict_preprocessed_array 绕过该复位（validation.py:493），故 high 验证永不发 forward 诊断——这是有意（避免每 epoch × 81 卷刷屏），但复位职责耦合在单一 CLI 入口，未来若新增直调 predict_preprocessed_array 的入口需自行处理护孔，属可维护性欠账。

B. 优化空间 / GPU
B1（中，GPU 快路不对称）cubic 单分辨率无 GPU 常驻 builder，退回 CPU numpy 抽取 + H2D — z 路径对单分辨率有专门的 GPU 常驻快路 build_z_window_single_res_gpu（vol 一次上 GPU、F.interpolate 取窗，@/d:/codes/work-projects/SegTask/segtask_v1/predictor/inputs.py:100-125，由 sliding.py:148-153 的 single_res 分支调度）；而 cubic 路径仅在 keep_native_multi_res=True（多 FOV）时走 GPU builder（sliding.py:342-351），单分辨率 cubic（最常见 cubic 配置，keep_native_multi_res=False）落到 build_cubic_batch_cpu_multi_res（逐 patch vol[...] numpy 切片 + np.pad + 每 batch 一次 torch.from_numpy(...).to(device)，inputs.py:262-290，sliding.py:369-374）。大卷 cubic 滑窗（窗口数 = D×H×W 三维乘积，远多于 z 路径）时这是主要 CPU + H2D 成本。建议补一个 cubic 单分辨率 GPU builder（类比 build_z_window_single_res_gpu，vol_t 常驻 + _edge_pad_axis + 逐窗 slice），与 z 路径对齐。收益随 cubic 窗口数放大，改动局限于 inputs.py/sliding.py 的单分辨率分支。

B2（低，GPU）z-interleave 每子流重复上传整卷 vol_t — sliding_window_z_interleaved 对 k 个子流各调一次 sliding_window_z(p, sub_vol)（@/d:/codes/work-projects/SegTask/segtask_v1/predictor/sliding.py:287），每次进入 sliding_window_z 都 torch.from_numpy(vol).to(device) 重新上传（sliding.py:86-89）。子流合计体素 = 整卷，故总上传量 ≈ 1× 整卷（非 k×，因各子流不相交），开销可控；但 k 次独立 H2D + k 次累加器分配有固定开销。低优先，仅高 k（如 k=3）+ 小卷时可见。

B3（低，已到位，记录）_blend_z_batch 按 actual_d 分组合并 F.interpolate（sliding.py:196-221）、addcmul_ 原地融合（sliding.py:236）、fp16 累加器先降精度再插值省瞬态（sliding.py:203-204）、_finalize_accumulators 先回 host 再升 fp32 免 GPU 端 fp32 副本（sliding.py:485-487） —— blending 热路径的显存/同步优化已充分。非 pD 尾窗的 build_1d_weight CPU 重建 + .to(device)（sliding.py:226-228）仅在 D_orig < pD 单窗场景发生，可忽略。无需改动，仅记录。

C. 可借鉴 / 新增（2026 视角，跨领域）
C1（中）高斯重叠权重可配置 σ + 边缘地板 — build_1d_weight 固定 σ = n/4（@/d:/codes/work-projects/SegTask/segtask_v1/predictor/blending.py:57）。nnU-Net 的 sliding-window Gaussian importance 用 σ_scale=0.125（σ = patch/8，更强中心权重）并对权重设 max*0.5 之类地板，避免窗口最边缘权重趋零导致的覆盖空洞/拼缝。当前 n/4 较宽（中心强调弱），且边缘相对权重 ≈ exp(-2)≈0.135 非零故无空洞，但值得把 sigma_scale + weight_floor 配置化作消融，对细管/边界拼缝对齐可能更好。低成本。

C2（中）缺 test-time 后处理钩子（连通域 / 小物体过滤） — 推理输出为纯 threshold + argmax（blending.py:prob_to_label），无形态学/连通域后处理接口。本项目主打细管/栓子等大量小连通前景，暴力 largest-CC 会误删真前景，故不宜默认开；但可提供可选、逐类可配的 largest-CC 保留 / 最小体积过滤钩子（对大器官类开、对细管类关），接在 prob_to_label 之后。属功能增强，优先级中低。

C3（低）TTA 增强项 — 当前仅 flip + 概率空间算术平均（forwards.py:99-113）。可选补：(a) logit 空间平均（几何平均，对饱和 sigmoid 更稳）作消融；(b) 旋转 90°/缩放 TTA（对各向同性重采样后数据）。flash/SDPA 后端已在模型层（Step 2 C3），此处不重复。低优先。

C4（低）滑窗调度可选 mirroring padding — 边界目前 edge-replicate（inputs.py:_edge_pad_z/_edge_pad_axis），与训练一致、已足够；nnU-Net 用镜像 padding，理论上对边界结构对称性略好，可作消融位，收益边际。

Step 6 — 串联与横向综合 目标：跨 5 部分的一致性（config/topology 真相源、训推契约）、架构级建议与 2026 先进做法适配优先级排序。产出：全局审查报告（问题总表 + 优化建议 + 借鉴清单，按价值/成本排序），并说明验证方式。依赖：Step 1-5。

Step 6 审查报告 — 串联与横向综合
目标复述与规则
综合 Step 1-5 的分部结论，做跨 5 部分的一致性核对（config/topology 真相源、训推契约）、架构级建议与 2026 先进做法的适配优先级排序，产出全局问题总表 + 优化建议 + 借鉴清单（按价值/成本排序），并说明验证方式。遵守规则：本轮不动任何代码/文档。

一、全局架构评价
项目是一个成熟的两级架构：公共框架层 taskcore + 任务层 segtask_v1（任务层多为 shim 转发，实现集中在公共层）。三条主动脉贯穿全流程且契约闭合：

几何/通道真相源：build_topology(cfg) 单点派生 in_channels/out_classes/spatial_dims/num_res_groups/per_view_depths/…，Config.sync 写只读 backing 字段，factory.build_model 与 pipelines.factory 只读不算（@/d:/codes/work-projects/SegTask/taskcore/models/topology.py:1-16）。主 UNet 路径完全遵守。
训推镜像：四 patch_mode 的训练抽取几何与推理滑窗窗口一一对应（whole resize / cubic·z max-FOV 抽取 + edge-replicate / 2.5D 折叠布局），且 z_boundary_mode='stretch' 被强制升级为 edge_pad 以消除训推几何漂移（@/d:/codes/work-projects/SegTask/taskcore/config/core.py:1209-1219）。
数值/同步纪律：增强在 autocast 外走 fp32、损失强制 fp32+logit clamp、val 混淆量 float64 + 单次 all-reduce、CPU 侧采样规避隐式 D2H。
未发现破坏性正确性 bug。以下横向综合按主题收敛 Step 1-5 的分散结论，并补充跨部分新洞见。

二、跨部分一致性核对（Step 6 新增，非分步复述）
X1（中）真相源"遵守度"存在两处例外，方向相反，均属可维护性欠账
模型侧未消费 topo：ADM/EDM2 build 调 build_topology 却只取 cond_in_channels，其余几何就地重算（Step 2-A1，@/d:/codes/work-projects/SegTask/taskcore/models/adm_unet.py:766-859、@/d:/codes/work-projects/SegTask/taskcore/models/edm2_unet.py:661-728）。
config 侧绕过 topo 各算一遍：Config.sync 仍局部重算 n_views 做 z_boundary_mode 副作用判定（@/d:/codes/work-projects/SegTask/taskcore/config/core.py:1223），虽是 data 侧副作用（合理分层），但与 topology 的 keep_native_view_depth/keep_native_multi_res 决策逻辑重复（@/d:/codes/work-projects/SegTask/taskcore/models/topology.py:96-105）。
横向结论：真相源模式落地约 90%，剩余例外都因 config.validate 硬拒非法组合而"重算值恰好等价"，当前无 bug；风险是新增 patch_mode/几何时"多处漏改"。这是全项目唯一系统性的架构欠账，价值高、成本低。
X2（中）cond 通道透传契约在 seg 主路存在静默错配缺口
factory 构建通用 Encoder 时未透传 cond_in_channels（Step 2-A2，@/d:/codes/work-projects/SegTask/taskcore/models/factory.py:450-466），而 topo.in_channels 已把 cond 计入。seg 当前无 cond_dirs（topo.cond_in_channels=0）故无碍，但 topology 已为 gen fork 预留 cond 逻辑（@/d:/codes/work-projects/SegTask/taskcore/models/topology.py:90-93），一旦 seg 接 cond 会静默把 cond 通道当 recon 主输入。这是"框架为多任务预留、seg 主路未闭合"的接缝，与 TODO3 重构（cls/det/gen/ssl 复用）直接相关。

X3（低）训推一致性有一处永不被 val 覆盖的口径差
val(high) 直调 predict_preprocessed_array 吃已烘焙 npz（Step 5-A2，@/d:/codes/work-projects/SegTask/segtask_v1/trainer/validation.py:493），而部署 predict_volume 走 resample_to_spacing 前处理+概率回采（@/d:/codes/work-projects/SegTask/segtask_v1/predictor/predictor.py:432-486）。训练期选出的 best 模型指标从不经历部署侧两次重采样误差。开 spacing_normalization 时，选模口径与部署口径存在未被验证的细微差。呼应 Step 1-C1/Step 3-C3 的 spacing 主题。

X4（低）确定性契约横跨两套全局 RNG
增强的 Bernoulli/标量走 CPU 全局 RNG、弹性位移/grid-dropout 走 CUDA 全局 RNG（Step 3-A3，@/d:/codes/work-projects/SegTask/taskcore/data/augment.py:22,181,322）。而数据层已有逐 worker 独立 RNG 的良好范式（Step 1）。增强层是全项目唯一依赖全局 RNG 状态的随机源，"固定 seed 等价性验证"（TODO3 的前置）因此脆弱。价值中（阻塞 TODO3 的 augment 合流验证）。【已修复：GPUAugmentor 新增 seed 参数，非 None 时创建专属 CPU/设备 Generator 并透传到全部采样点（Bernoulli/标量 uniform_/弹性 randn/噪声 randn/grid-dropout randint），与全局 RNG 完全解耦；None 时沿用全局 RNG 保持历史行为。seg trainer 以 train.seed 逐 rank 分流构造；其余任务 trainer 未传 seed，行为不变。】

X5（低）误配防护在 predict/augment 两处有枚举/正性校验缺口
config.validate 对训练/数据/topology 覆盖严密，但：blend_mode 无枚举校验（拼错静默退化为均匀平均，Step 5-A1）、augment 的 sigma/std/zoom 无正性/区间校验（[0,0]/负值配置必炸、下界 0 小概率 NaN，Step 3-A1）。同类"fail-fast 覆盖面"缺口，宜合并一轮补齐。【均已修复：_validate_predict 补 blend_mode 枚举；_validate_augment 补 sigma/std/zoom/gamma 区间校验】

三、问题总表（正确性/稳健性，按价值×成本排序）
#	主题	严重度	价值/成本	位置	来源
P1	ADM/EDM2 build 不读 topo，就地重算几何【已修复】	中	高价值·低成本	adm_unet.py:766-859 edm2_unet.py:661-728	Step2-A1 / X1
P2	factory 未透传 cond 到 Encoder（seg 接 cond 会静默错配）【已修复】	低-中	中价值·低成本	factory.py:450-466	Step2-A2 / X2
P3	增强 RNG 跨两套全局生成器，等价性验证脆弱【已修复：GPUAugmentor 独立 Generator，seg trainer 逐 rank 分流】	中	中价值·中成本	augment.py:22,181,322	Step3-A3 / X4
P4	blend_mode 无枚举校验，拼错静默退化【已修复】	低	中价值·极低成本	blending.py:55-71	Step5-A1 / X5
P5	augment sigma/std/zoom 无正性/区间校验【已修复】	低	中价值·极低成本	core.py:1738-1771	Step3-A1 / X5
P6	ConvNeXt/MedNeXt 块内 attention 未透传 norm/reduction 配置（含核验新发现 N1：se_reduction 被静默忽略）【已修复 reduction 部分；norm 有意保持块内固定 norm 设计】	低	低价值·低成本	convnext.py:58 mednext.py:424	Step2-A3 / N1
P7	seg 重实现 SWA/BN 重校准，与 base 双份维护【已修复：合流回 base 回调版】	低	低价值·中成本	trainer.py:708-791	Step4-A2
P8	识别性冗余 except / whole 缓存别名一致性瑕疵（含核验新发现 N2：dataset.py:1169-1172 旧注释称“无越界填充时返回缓存卷视图”，与 extractor 无条件 .copy() 的实现矛盾）【已修复：删死 except、改正注释、whole 恒等 resize 补 copy 断别名】	低	低价值·低成本	dataset.py:430,1169-1172,1307-1311	Step1-A1/A2 / N2
已撤销：P9（fp16 路径 nonfinite 健康计数漏计）经核验不成立——pending 非日志步不清空，_flush_pending 在日志步/epoch 末处理全部积压条目，计数仅延迟不漏计（见 Step4-A1）。
已关闭：Step 1-A3（DDP val 不等长死锁）经 Step 4-A0 核对——验证全程无逐 batch collective，无死锁风险。

四、优化建议（性能/显存/GPU，按价值×成本排序）
#	主题	价值/成本	位置	来源
O1	cubic 单分辨率补 GPU 常驻 builder（对齐 z 路径），消 CPU 抽取+H2D	高价值·中成本（收益随 cubic 窗口数放大）	inputs.py:262-290 sliding.py:369-374	Step5-B1
O2	val 混合调度（medium 每 epoch + high 每 N epoch），降 high 墙钟【已修复：val_high_interval + HybridValEvaluator；整卷 RAM 缓存未做】	高价值·低成本	validation.py:483-495	Step4-B2
O3	损失 fg_values 改设备张量驻留，消热路径逐步 H2D【已修复：首次 forward 惰性构建后缓存复用（非 register_buffer：criterion 不随 model .to(device)，惰性缓存更稳妥）】	中价值·低成本	losses.py:784,868,942	Step4-B1
O4	trainer 侧显式开 inplace=True，省过采样 cube 一份瞬时显存【已修复：GPUAugmentor 新增 inplace 覆写参，seg trainer 传 inplace=True（H2D 私有拷贝满足契约）；config 默认 False 保留防御】	中价值·低成本	augment.py:52-56	Step3-B1
O5	手写 LayerNorm/GRN 统计 fp32 累加（对齐既有 fp16→fp32 范式）	中价值·低成本	convnext.py:27-29 blocks.py:97-99	Step2-B1
O6	LRU 缓存可选存 int16 原始卷（以 CPU 换 RAM，大数据集划算）	中价值·中成本	dataset.py:749-753	Step1-B1
O7	RoPE 路径 torch.compile 友好化（buffer 化坐标、非就地 rotate）	中价值·中成本（仅开 RoPE 时）	blocks.py:35-36,568-591	Step2-B2
O8	CPU zoom resize 后端评估（大 H/W 面内 resize 逐切片）	低价值·中成本（观察项）	dataset.py:554,928,1307	Step1-B2
五、借鉴清单（2026 视角，按价值×成本排序）
第一梯队（高价值，与本项目"细管/栓子小前景"主打强相关）

B-loss / instance-aware 损失（Step4-C1）：blob loss 逐实例加权，缓解"大病灶主导 Dice、小病灶被淹没"，接入现有 _COMPOUND_BUILDERS（@/d:/codes/work-projects/SegTask/segtask_v1/losses/losses.py:705-715）。
Top-k CE / OHEM 复合项（Step4-C2）：稀疏前景下聚焦难体素，config 级新增、与工厂契合。
nnU-Net v2 高各向异性轴 spacing 高百分位（Step1-C1）：CT 厚层场景避免低分辨率轴过度上采样，改动局限 make_data._resolve_target_spacing。
第二梯队（中价值，消融/工程升级）

augment "伴随张量 spec 化"合流（Step3-C1）：把 (image,label,wmap) 泛化为 primary + companions[spec]，五任务共用一份实现——这是 TODO3 的核心重构入口，需先落地 X4（独立 Generator）做等价性验证。
blosc2 分块压缩替代 npz（Step1-C2）：兼得压缩率与近 mmap 随机块访问，解决"压缩 vs page-cache 共享"二选一；改动较大列为候选。
现代优化器 opt-in（Step4-C3）：Schedule-Free AdamW / Muon / SOAP，接入 build_optimizer 分派即可。
上采样/注意力消融位：feature-aligned 上采样 SAPA/FADE（Step2-C2）、NATTEN/显式 SDPA 后端（Step2-C3）、高斯权重 σ_scale+weight_floor 配置化（Step5-C1）。
强度增强对 zscore 自适应（Step3-C2）：幅值按归一化模式自适应（zscore 时以 σ 为单位）。
第三梯队（低价值，收益边际）

多标签迭代分层划分（Step1-C3）、损失权重退火（Step4-C4）、更强 seg head（Step2-C1）、test-time 连通域后处理钩子（Step5-C2）、logit 空间 TTA（Step5-C3）、镜像 padding 消融（Step5-C4）。
六、推荐落地路线（价值/成本联合最优）
一轮"fail-fast 补齐 + 真相源对齐"（P1/P2/P4/P5，纯低成本、被现有单测覆盖）——消掉唯一系统性架构欠账 X1/X2 与两处校验缺口 X5。
一轮"推理/验证加速"（O1/O2/O3）——最高价值的墙钟收益，改动局限单分辨率分支/val 调度/损失 buffer。
augment 独立 Generator（X4/P3）→ 等价性验证 → 伴随张量合流（Step3-C1）——作为 TODO3 augment 合流的前置链。
损失创新梯队（blob/Top-k）作为算法消融按需接入。


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
