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

O1 GPU 常驻 builder：现在训练时，每个小块（patch）都是 CPU 在硬盘数据里现切现缩放，再搬给 GPU。O1 是把整卷数据直接常驻在 GPU 显存里，切块、缩放都让 GPU 来干——CPU 不再是瓶颈，训练可能明显提速。但要大改数据流水线，而且吃显存，所以建议你先在自己的机器上测一下"现在到底卡不卡在 CPU 上"，值得才动手。

O8 CPU zoom 后端：每次取样本都要用 scipy 做一次缩放（zoom），这一步可能挺慢。O8 就是先测测它到底占多少时间，慢的话换个更快的缩放库。纯观察项，先测再说。

借鉴清单：

blob loss：让损失函数按"一个个连通病灶"算分，防止大病灶淹没小病灶。
Top-k CE/OHEM：算损失时只挑"错得最狠的那部分体素"重点学，前景稀少时更有效。
spacing 百分位：重采样目标层厚不用中位数、改用百分位，避免厚层 CT 被过度上采样。
blosc2 存储：换更现代的压缩格式，文件小、读得还快。
后处理钩子：预测完可选做"只留最大连通块/过滤小碎块"之类的清理。
TODO 3 大重构：seg 和 gen 现在各有一份几乎一样的增强代码（复制粘贴的分叉），要合并成一份通用实现；配置文件结构也要拆分整理。工程收益大但改动面大，适合单独一大轮。

1. GPU 复核（最简单，先做这个） 在你 Windows 机器上打开命令行：



cd D:\codes\work-projects\SegTask
D:\miniconda\envs\torch27_env\python.exe -m pytest tests -q
等它跑完，全绿（passed）就说明改动在你的 GPU 环境也没问题。

2. O1 值不值得做——看训练是不是"GPU 在等 CPU" 最省事的办法：开一次正常训练，同时开任务管理器（性能页）看 GPU 利用率。

GPU 大部分时间跑在 90%+ → CPU 喂数据够快，O1 收益不大，不用做；
GPU 经常掉到很低、一抽一抽的 → 说明 GPU 在等 CPU 切块，O1 值得做。 辅助判断：训练日志里如果打印了 data time / 数据加载耗时，占比高（比如 >20%）也说明卡在数据侧。
3. O8 值不值得做——测 zoom 这一步有多慢 其实和上面是同一个问题的细化：如果第 2 步显示卡在 CPU，我可以下一轮写个小脚本发你，按你的真实 patch 尺寸测 scipy zoom 单次耗时，几秒钟出结果；如果第 2 步显示 GPU 一直吃满，O8 也不用测了。


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
