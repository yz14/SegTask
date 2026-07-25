# 规则与要求

## 一、任务启动(两阶段,强制)
新任务必须**先调研规划、后编码执行**,二者分属不同轮次,不得在同一轮合并完成。

- **第一轮(调研规划,禁止写实现代码)**:阅读相关现有代码、文档与上下文,检索业界做法;明确目标、范围边界、约束与难点;产出一份**可拆分、各步骤可独立执行**的计划,每步含目标、预期产出、验收标准及依赖关系。本轮只输出结论与计划,等我确认后再动手。
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


步骤 0：范围界定、体量摸底、契约清单、计划产出。
仓库结构：taskcore 为五任务（seg/cls/det/gen/ssl）公共层，分 config / data / models / engine / monitor / utils / metrics。
代码体量：taskcore 约 16k 行 / 56 个 py 文件，其中重量级文件为 core.py(2168)、blocks.py(1314)、dataset.py(1192)、loader.py(988)、base_trainer.py(938)、adm_unet.py(930)、edm2_unet.py(805)、make_data.py(652)。
文档面：仓库级 README.md、SSL.md，README.md，以及 5 个子项目 README + WORKFLOW.md（seg 另有 DESIGN.md），共 18 个 md。
已识别的跨层契约（后续审查的重点校验对象）：patch_mode 四模式（z_axis/2_5d/cubic/whole）五任务同语义、2.5D 折叠时机契约（默认送模型前折叠，det/cls 有例外）、checkpoint 槽位约定（extract_model_state_dict + EMA 语义）、DDP 指标可加性约定、[shim] 旧路径兼容层。

步骤 1 — 文档全面通读与"设计意图基线"：读完 18 个 md，抽出全部显式契约与设计取向，形成"文档声明 vs 代码实现"核对清单。依赖：无。产出：契约核对表（后续每轮拿它对代码）。
A. 契约核对表（27 条，按主题分组）
A1. 几何与数据口径
#	契约声明（出处）	待核实点（代码落点）	计划轮次
C1	patch_mode 四模式五任务同语义，且 seg/cls/det/gen 抽取口径逐位一致 @d:\codes\work-projects\SegTask\README.md:37	dataset.py 的三模式类 vs patch_dataset_base.py/patch_extract.py/patch_ops.py（cls/det 共用基类）是否真为同一抽取实现，还是两套并行代码	步骤 4
C2	z_axis/2_5d：H/W 整面 resize 不裁窗；cubic：安全中心域三轴裁剪 + edge-pad @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:90-95	边界 pad 模式、resize 插值算子（label 必须最近邻/保整数）、越界坐标夹取	步骤 4
C3	2.5D 折叠时机：dataset 发未折叠 3D，送模型前折叠；det 是唯一例外（dataset 内折叠，联动 slice_boxes_to_2d） @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:165 @d:\codes\work-projects\SegTask\dettask\docs\WORKFLOW.md:152	与 cls 的"关增强时 dataset 折叠"是否真等价（见 D1）；squeeze_2_5d 是否单点实现	步骤 4
C4	dataset 只发单分辨率 max-FOV cube，多分辨率裁剪/resize 推迟到 GPU 侧，避免二次插值 @d:\codes\work-projects\SegTask\segtask_v1\docs\DESIGN.md:51	oversample 余量计算是否覆盖 max_scale × aug_oversample_ratio；GPU 侧中心裁剪是否与 dataset 抽取同中心	步骤 4
C5	GPU 增强：空间变换 img/lbl/wmap/cond 同步，强度变换仅 img；det 的仿射/弹性自动关闭 @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:194 @d:\codes\work-projects\SegTask\dettask\docs\WORKFLOW.md:102	augment.py Companion 张量语义、label 的插值模式、det 关闭分支是否真的零副作用	步骤 4
C6	类均衡采样：npz 逐类 fg 索引（*_cls 键），先抽类再抽位置；旧 npz 惰性回退 @d:\codes\work-projects\SegTask\clstask\docs\WORKFLOW.md:48-50	sampling.py / make_data.py 的索引生成与消费是否同构；回退路径正确性与性能	步骤 3
C7	验证确定性：val 中心由 (seed, idx) 派生；val_grid_coverage 走 z 等距 bin / Halton(2,3,5)，与推理同口径 @d:\codes\work-projects\SegTask\clstask\docs\WORKFLOW.md:50	Halton 实现正确性、DDP 分片下确定性是否仍成立	步骤 3
C8	pid 强配对契约：image/label/bbox/rw/npz/exclude 同 pid，缺失即报错 @d:\codes\work-projects\SegTask\segtask_v1\docs\DESIGN.md:103-112	loader.py 的发现/配对/exclude 逻辑是否有静默跳过分支	步骤 3
C9	make_data 同口径：spacing 归一化 + fg 索引 + meta skip + 几何校验，gen 委托 prepare_one @d:\codes\work-projects\SegTask\taskcore\README.md:54	check_physical_geometry 的严格度、spacing 逆变换信息是否完整写入 meta	步骤 3
C10	双源混采：npz_dir_secondary + mix_ratio，val 仅金标准 @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:206	mixed_sampler.py 在 DDP + drop_last 下的比例保真与 epoch 长度定义	步骤 3
A2. 模型与拓扑
#	契约声明	待核实点	轮次
C11	build_topology 是几何派生单一真相源（in_channels / num_classes / spatial_dims / n_views / per-view depth / aux 激活） @d:\codes\work-projects\SegTask\segtask_v1\docs\DESIGN.md:89-97	全仓是否仍有旁路重算（grep spatial_dims = / in_channels = 的独立推导）	步骤 5
C12	参数命名 encoder.* / decoder.* / fpn.* / det_head.* / cls_head.* 跨任务同名同形，SSL 权重 strict=False 直接命中 @d:\codes\work-projects\SegTask\dettask\docs\WORKFLOW.md:148	factory.build_backbone 与 ssl export_backbone_state_dict 的键名一致性	步骤 5
C13	梯度检查点：encoder 逐 stage（grad_ckpt_encoder_stages 掩码）/ decoder 逐 level / ADM·EDM2 逐块；eval 零开销、数值与关闭时严格一致 @d:\codes\work-projects\SegTask\gentask\docs\WORKFLOW.md:102	与 use_reentrant 语义、RNG（dropout/drop_path）在重算时的一致性	步骤 5
C14	稀疏—稠密等价（SparK 路线）：满密度输入下退化为普通卷积，预训练/下游共享权重 @d:\codes\work-projects\SegTask\SSL.md:45	taskcore blocks.py 的门控/归一化是否满足该等价（norm 只在活动位点统计）	步骤 5
C15	依赖克制：NMS/ROIAlign/可变形注意力/AUC 全部纯 PyTorch 自实现，不引 torchvision/mmdet/sklearn/monai @d:\codes\work-projects\SegTask\dettask\docs\detection_models_survey.md:74	是否存在隐性第三方依赖；自实现算子的数值/性能代价	步骤 5、7
A3. 训练工程
#	契约声明	待核实点	轮次
C16	损失恒 fp32（autocast 外）+ logit clamp；AMP auto = Ampere+ 选 bf16 @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:175	amp.py 的 dtype 决策与 LOGIT_CLAMP 使用面；各任务是否真的都在 autocast 外算损失	步骤 6
C17	优化步时钟：scheduler/warmup/global_step/方法内调度按真实 optimizer.step 边界推进，尾批按实际累积长度归一 @d:\codes\work-projects\SegTask\ssltask\docs\WORKFLOW.md:121	base_trainer.py + optim.py 的 step 计数与 accum 尾组归一实现	步骤 6
C18	非有限守护：loss/梯度非有限丢弃整个 accum 组，不污染权重/EMA；fp16 由 GradScaler 跳步；DDP 下跳步决策 all-reduce(any) 统一 @d:\codes\work-projects\SegTask\ssltask\docs\WORKFLOW.md:122	跳步时 scheduler 是否照常推进（文档称照常）、EMA/queue/center 冻结是否完备	步骤 6
C19	EMA：验证与 best 均用 shadow；ema_device: cpu offload；warmup @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:182,192	shadow 与 buffer（BN running stats）的处理、offload 的同步开销	步骤 6
C20	checkpoint：原子写（temp + os.replace）+ 状态指纹；resume 位精确（含 RNG）；DDP 仅 rank0；可选异步写 @d:\codes\work-projects\SegTask\ssltask\README.md:100	checkpoint.py 的指纹算法、异步写与训练结束的竞态、RNG 恢复完整性（含 dataloader worker）	步骤 6
C21	选模槽位：best 的 model_state_dict = EMA 权重，在线权重另存 model_online_state_dict；统一经 extract_model_state_dict 读取 @d:\codes\work-projects\SegTask\taskcore\README.md:60-66	ssl 无独立 ema_state_dict 的分支、gen 历史 ckpt 兼容路径	步骤 6
C22	seg 选模 loss 口径定案为 val_base_loss（不含深监督/aux/正则，跨配置可比） @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:218	_save_best 的 criterion 分派与 plateau 调度方向一致性校验	步骤 6
C23	DDP：no_sync 免非边界通信、静态图、bucket-view；ssl 不套 DDP wrapper，手动 accum 边界梯度均值 all-reduce @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:212 @d:\codes\work-projects\SegTask\ssltask\docs\WORKFLOW.md:126	BaseTrainer 如何同时承载两条 DDP 路径；静态图与梯度检查点/未用参数的兼容	步骤 6
C24	指标 DDP：可加混淆量 + all-reduce(SUM)；batch 内池化比值损失（batch_dice）在 DDP 下为近似；不可分解指标（AUC/mAP）先聚齐全集再算 @d:\codes\work-projects\SegTask\taskcore\README.md:88	metrics.py 的可加性实现与各任务 metrics 是否重复造轮	步骤 7
A4. 配置与迁移
#	契约声明	待核实点	轮次
C25	配置接入两套并存：seg/cls/det/ssl 走 register_task_section，gen 走 dataclass 子类化 + 顶层 Config 组合委托 @d:\codes\work-projects\SegTask\taskcore\README.md:45-54	两套机制的能力差与维护成本；能否收敛为一套	步骤 2、8
C26	校验错误语义：配置一律 ConfigError，data 路径用 ValueError/FileNotFoundError，模型构造期 assert 仅内部不变量 @d:\codes\work-projects\SegTask\taskcore\README.md:85	实际抛错类型是否守约；assert 在 -O 下失效的风险点	步骤 2、5
C27	迁移交叉校验：patch_mode/spatial_dims/in_channels 与预训练不一致直接报错、不静默降级；0 命中报错 @d:\codes\work-projects\SegTask\clstask\docs\WORKFLOW.md:146 @d:\codes\work-projects\SegTask\dettask\docs\WORKFLOW.md:106	seg 侧 train.pretrain 是否有同等强度校验（文档只说 strict=False，未提交叉校验）	步骤 2、6
B. 纯文档层面已发现的疑点（未看代码，均待核实）
D1 · "唯一例外"表述互斥：seg WORKFLOW 称 2.5D 折叠时机的"唯一例外是 det" @d:\codes\work-projects\SegTask\segtask_v1\docs\WORKFLOW.md:165，但 cls WORKFLOW 明确存在第二条例外——关闭增强时在 dataset 侧折叠 @d:\codes\work-projects\SegTask\clstask\docs\WORKFLOW.md:148；顶层 README 的表述才是完整的 @d:\codes\work-projects\SegTask\README.md:38。文档不一致（cls 声称两路径"等价"，需代码验证等价性是否严格成立）。
D2 · gen 训练/推理频谱不等价（文档自认待定）：训练侧 whole/z_axis/2_5d 整卷或面内 resize 到 patch 尺寸，推理侧在原生分辨率滑窗 @d:\codes\work-projects\SegTask\gentask\docs\WORKFLOW.md:154。需核实这是否只影响 gen：seg/cls/det 的推理文档都写了"面内 resize + 反向缩放"，若属实则 gen 是独苗；若 taskcore 侧抽取代码共用，则风险可能外溢。
D3 · BasePredictor 有意薄 → 滑窗/blend 四处重复：taskcore 只放 AMP + flip TTA @d:\codes\work-projects\SegTask\taskcore\README.md:47，而 seg/gen/det/cls 各自实现滑窗、gaussian/average blend、micro-batch、坐标还原。四份高度同构逻辑分散是架构层最可疑的一处（重复代码 + 口径漂移风险，且与 C1"逐位一致"目标冲突）。
D4 · 指标实现分散：公共 metrics.py 之外，cls/det/ssl 各有 metrics.py（AUC/F1、mAP/FROC、AUC/F1/HD95）。ssl eval/metrics.py 与 cls metrics.py 的 AUC/F1 很可能重复实现，需核实是否可收敛。
D5 · core.py 2168 行单文件：承载 data/augment/model/train/vis/monitor 全部段 + 校验 + 迁移上下文，与"模块化、职责分离"取向张力明显；同时它是五任务共同的改动热点（每加一个任务字段都要动它）。
D6 · shim 债务面：seg/gen 包内大量 [shim] re-export，文档反复声明"新代码勿依赖" @d:\codes\work-projects\SegTask\taskcore\README.md:86。需量化：还有多少内部调用点走 shim、是否存在 shim → taskcore → shim 的环。
D7 · 私有名转正的兼容层：8 组"公开名 + 旧名别名" @d:\codes\work-projects\SegTask\taskcore\README.md:72-81，属可清理债务，需核实旧名是否仍有内部引用。
D8 · SSL.md 规格 vs 实现差异面：SSL.md 的统一骨干规格（128³、五级、总步长 16、32→320 通道、InstanceNorm/LeakyReLU）@d:\codes\work-projects\SegTask\SSL.md:15 是受控实验的前提；需核实 taskcore factory 默认配置能否精确复现该规格，否则"只变自监督目标一个变量"的实验控制不成立。
D9 · SparK 稀疏卷积：SSL.md 明确要求 spconv/TorchSparse 稀疏前向 @d:\codes\work-projects\SegTask\SSL.md:35，而 ssltask WORKFLOW 描述的是"置零 + 逐尺度门控"的稠密模拟 @d:\codes\work-projects\SegTask\ssltask\docs\WORKFLOW.md:80。二者在感受野是否向被遮区渗透上并不等价——这是方案一/二对照实验的核心变量，需重点核实。
D10 · 文档未覆盖面：monitor、launcher、visualization、tools/、tests/（83 项）在文档中只有一句话级描述；img_process/ 完全未提。审查时需从代码侧补齐认知。
C. 由文档推断的高风险靶点（后续轮次优先钻透）
抽取口径"逐位一致"是否名副其实（C1+C2+D2）：seg 的 dataset.py 与 cls/det 的 patch_dataset_base.py 双实现 → 步骤 4 头号任务。
2.5D 折叠时机三条路径的严格等价性（C3+D1）。
累积 × AMP × DDP × EMA × scheduler 五者交互（C17+C18+C23）：这是训练框架最容易出隐性 bug 的交叉区。
滑窗/blend 四处重复实现的口径漂移（D3）。
build_topology 单一真相源是否被旁路（C11）。
SSL 受控实验前提是否成立（D8+D9）。

步骤 2 — config 层：core.py、registry.py、task_io.py、seg_task.py、seg_bundle.py、section_validators.py、model_migration.py。重点：配置模型的可扩展性、校验完备性与错误语义、任务段注册机制、2168 行单文件的职责划分、与业界配置体系（Hydra/OmegaConf、pydantic v2、结构化配置 + 版本迁移）的对比取舍。依赖：步骤 1。
一、结论摘要
总体质量高于常见科研配置层：派生量（in_channels/spatial_dims/save_best_metric）已收敛为只读 property + build_topology 单一真相源；未知键 / 旧别名 / 新旧同设一律 fail-fast；错误文案普遍带修复建议。C25/C26 大体守约。
主要问题不在"写得对不对"，而在"同一逻辑存在两份"：gen 走 dataclass 子类化后复制了 sync / _apply_resenc_preset / _validate_train / arch allowlist，已产生 3 处可证实的语义漂移（见 F1/F4/F9）。
一个机制是空操作：Config.validate(skip=...) 的 skip 从未被消费（F2）。
架构层：2464 行单文件 + seg 的 4 条加载入口 + 两套任务接入机制，是后续每加一个任务/字段的固定摩擦源。
二、契约核对（步骤 1 挂账项）
契约	结论	依据
C25 两套接入并存	属实。seg/cls/det/ssl 经 register_task_section，gen 为子类化 + 顶层 Config 组合委托	@d:\codes\work-projects\SegTask\taskcore\config\registry.py:47-105、@d:\codes\work-projects\SegTask\gentask\config\validation.py:19-31
C26 错误语义	基本守约：config 全走 ConfigError（_require），无裸 assert；ConfigError(AssertionError, ValueError) 使 -O 下不失效。两处漏点：override 强转抛裸 ValueError、未知点路径抛 AttributeError	@d:\codes\work-projects\SegTask\taskcore\config\core.py:18-31、@d:\codes\work-projects\SegTask\taskcore\config\task_io.py:44-67
C27 迁移交叉校验	config 层无此校验（seg 侧只有 pretrain_strict / pretrain_upkern 提示）。cls/det 文档所述的 patch_mode/spatial_dims/in_channels 交叉校验若存在，只可能在 engine 侧 → 顺延步骤 6 核实	@d:\codes\work-projects\SegTask\taskcore\config\core.py:944-965,2172-2176
D5 core.py 单文件	确认：data/aug/model(+3 嵌套 arch 段+3 模块段)/loss/train/predict/vis/monitor 全部段定义 + 全部校验器 + 迁移表 + YAML I/O + legacy pickle 兼容同处一文件	@d:\codes\work-projects\SegTask\taskcore\config\core.py:62-2464
D7 私有名转正别名	config 层的 _nested_dataclass_type 已无任何内部引用，属可清理死别名	@d:\codes\work-projects\SegTask\taskcore\config\core.py:2290
三、缺陷清单
P0 —— 正确性 / 隐性行为差异
F1 · gen 缺 stretch → edge_pad 自动升级（跨项目漂移，可导致训推几何不一致） core 在 sync() 里把废弃的 z_boundary_mode='stretch' 强制升级并告警，理由是训练侧恒走 edge-pad、只有推理侧生效 → 薄卷训推几何 desync：



core.py:1305-1312
if self.data.z_boundary_mode == "stretch":
    logger.warning(
        "data.z_boundary_mode='stretch' is deprecated: training-side "
        "extraction always uses edge-pad geometry, so stretch would "
        "only take effect at inference and desync train/infer "
        "geometry for volumes thinner than the patch depth. "
        "Auto-upgraded to 'edge_pad'.")
    self.data.z_boundary_mode = "edge_pad"
gen 的 sync() 是这段逻辑的复制品，唯独漏了这个分支（@d:\codes\work-projects\SegTask\gentask\config\validation.py:33-72）；而 gen _validate_data 委托的 core 实现仍把 'stretch' 列为合法（@d:\codes\work-projects\SegTask\taskcore\config\core.py:1877-1880），Dataset 也照单全收（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:872-875）。结论：gentask 配置写 stretch 会一路通过，core 想防的问题在 gen 上完全没防。

F2 · Config.validate(skip=...) 是空操作 入参接了、skip = skip or set() 也写了，随后 6 个校验器无条件全跑，skip 再未出现：



core.py:1409-1415
skip = skip or set()
self._validate_model()
self._validate_augment()
self._validate_data()
self._validate_2_5d()
self._validate_train()
self._validate_monitor()
上游 TaskSectionSpec.skip_core_validators → validate_core_config → 此处（@d:\codes\work-projects\SegTask\taskcore\config\registry.py:85-88）以及 SegBundle.validate 传的 skip | {"loss","predict"}（@d:\codes\work-projects\SegTask\taskcore\config\seg_bundle.py:52-54）全部落空。当前 _COMPOSITE_SKIP_CORE = ()，故暂无 live bug，但这是典型的"看着生效、实则静默无效"结构：任何人日后填入段名都不会被执行，且不会报错。（SegBundle 自己那段按段拆分的 seg 校验是真实生效的，不受影响。）

F3 · 老 checkpoint pickle 迁移未补派生 backing 字段 spatial_dims/in_channels 现为只读 property，backing 由 __post_init__ 建立（@d:\codes\work-projects\SegTask\taskcore\config\core.py:607-619）。而 legacy pickle 分支绕过 __init__，只补 nested_sections 里的嵌套段，不补 _spatial_dims/_in_channels：



model_migration.py:295-297
for f in type(self).__dataclass_fields__.values():  # type: ignore[attr-defined]
    if f.name not in self.__dict__ and f.name in nested_sections:
        self.__dict__[f.name] = f.default_factory()  # type: ignore[misc]
老 state 里的扁平 spatial_dims 键会被塞进 __dict__，但 property 是数据描述符、实例字典不生效 → 读 mc.spatial_dims 抛 AttributeError。危害受限（文档称 ckpt 内嵌 config 不被消费），但属于"兼容层自身不兼容"的隐雷。

F4 · resenc_preset 大小写口径 core/gen 分叉 core 大小写不敏感、与 _apply_resenc_preset 的 .lower() 查表对齐（@d:\codes\work-projects\SegTask\taskcore\config\core.py:1509-1512）；gen 仍是严格大写白名单（@d:\codes\work-projects\SegTask\gentask\config\validation.py:365-367）。同一份 YAML 写 resenc_preset: m，seg 能跑、gen 报错。

F5 · 几何整除性校验缺位 → 迟到失败 配置期只在两条窄路径上检查整除性：lift 的 D % 2**(n_levels-1)（@d:\codes\work-projects\SegTask\taskcore\config\core.py:2025-2033）与 hierarchical 融合的 H/W（@d:\codes\work-projects\SegTask\taskcore\config\core.py:2096-2099）。常规 3D / 2.5D 路径下 patch_size 与总下采样步长（stem_stride × Π downsample_strides）的整除性无校验，非法组合要等到首个 forward 才炸：



unet.py:270-274
if x.shape[2:] != skip.shape[2:]:  # 上采样后必须与 skip 同尺寸
    raise RuntimeError(
        f"DecoderLevel size mismatch after upsample: "
        f"x={tuple(x.shape[2:])} vs skip={tuple(skip.shape[2:])}. "
        f"Check input spatial dims are divisible by total encoder stride.")
配置层已有 _est_stage_tokens 这套逐级 stride 推导能力（@d:\codes\work-projects\SegTask\taskcore\config\core.py:1580-1607），把它复用为整除性校验几乎零成本，却能把"跑完数据装配再崩"提前到秒级。

P1 —— 工程与可维护性
F6 · override 强转不能做标量→列表，且错误类型不统一 coerce_override_value 按旧值类型决定转换（@d:\codes\work-projects\SegTask\taskcore\config\task_io.py:46-56）：predict.threshold 默认 0.5（Union[float, List[float]]），--override seg.predict.threshold=[0.3,0.6] 会走 float("[0.3,0.6]") 直接崩；只有默认值为 None 的 Optional 字段（如 target_spacing）才走 YAML 解析。根因是"用运行时值推类型"而非"用声明类型推类型"。另 set_dotted_attr 对未知路径抛 AttributeError、bool/list 转换失败抛裸 ValueError，与 C26 的 ConfigError 口径不齐。

F7 · seg 有 4 条加载入口，且行为不一致 seg_config.load_config（→SegBundle，做 hoist）、load_config_parts（→ 元组，做 hoist）、registry.load_task_config("seg")（→ 元组，不做 hoist）、core.load_config（丢弃 seg 段只 warning，@d:\codes\work-projects\SegTask\taskcore\config\core.py:2405-2410）。其中第三条是注册表通路却缺 hoist_legacy_seg_sections（@d:\codes\work-projects\SegTask\taskcore\config\task_io.py:120-123）：同一份旧式顶层 loss:/predict: YAML，走 seg_config 正常、走 registry 直接 Unknown config key 'loss'。

F8 · sync → validate 是隐式顺序契约，且每个入口手抄四行仪式 validate 大量读派生量（如 @d:\codes\work-projects\SegTask\taskcore\config\core.py:1504），未 sync 先 validate 会拿默认 3D/1ch 去校验并给出误导性报错，无任何守卫（如 _synced 标志）。同时 8 个 CLI 各自重复 apply_overrides → sync → validate → validate_task（@d:\codes\work-projects\SegTask\clstask\train.py:98-101、@d:\codes\work-projects\SegTask\dettask\train.py:99-102、@d:\codes\work-projects\SegTask\ssltask\pretrain.py:87-90、@d:\codes\work-projects\SegTask\segtask_v1\train.py:130-132 等）——目前都写对了，但这属于"靠自觉"的一致性。

F9 · gen 的校验覆盖是 core 的真子集 gen 的 ModelConfig 继承 core，因而拥有 selfattn.* / multirf.* / mednext.* / grad_ckpt_encoder_stages 全部字段，但 gen _validate_model 完全不校验这几组（@d:\codes\work-projects\SegTask\gentask\config\validation.py:289-368），core 则有近 170 行专门校验（@d:\codes\work-projects\SegTask\taskcore\config\core.py:1558-1775）。若 models/factory 是共享的（步骤 5 核实），gen 可构出未经护栏（含 softmax O(N²) token 上限）的模块。

F10 · 无错误聚合、无 schema 版本、迁移表只增不减 _require 首错即抛，用户改配置只能一轮一个错；三张迁移表（_FIELD_ALIASES / _DEPRECATED_DERIVED_KEYS / _REMOVED_KEYS，@d:\codes\work-projects\SegTask\taskcore\config\core.py:2246-2273）+ FLAT_TO_NESTED（60 项）无版本号、无弃用窗口、无清理判据。另 core.__getattr__ 的告警文案写的是 segtask_v1.config.%s，而模块实为 taskcore.config.core（@d:\codes\work-projects\SegTask\taskcore\config\core.py:2459）——误导。

值得肯定（不建议改动）
派生量只读化 + _DEPRECATED_DERIVED_KEYS 硬拒绝，杜绝"设了却被 sync 静默重写"，方向正确。
route_legacy_model_dict 对"新旧同设"一律 fail-fast、不做静默优先级（@d:\codes\work-projects\SegTask\taskcore\config\model_migration.py:251-254），与 hoist_legacy_seg_sections 范式统一。
SegBundle.__getattr__ 对 _core/_seg 与 dunder 显式抛 AttributeError 防 unpickle 自递归（@d:\codes\work-projects\SegTask\taskcore\config\seg_bundle.py:68-78），是踩过坑的写法。
softmax 注意力的 token 数护栏（@d:\codes\work-projects\SegTask\taskcore\config\core.py:1677-1696）：把"跑到一半 OOM"提前成配置错误，正是 F5 应当照抄的范式。
四、架构评估
1) 2464 行单文件（D5 确认）：职责为「段定义 × 8 + 嵌套子段 × 6 + 校验器 × 8 + 预设表 × 2 + 迁移表 × 3 + YAML I/O + legacy pickle 钩子」。它同时是五任务的公共改动热点。建议的切分（仅建议）：sections/{data,augment,model,train,predict_loss,vis_monitor}.py + validators/{model,data,train,geometry}.py + io.py + legacy.py，core.py 退化为组装与再导出（__init__ 已有 from .core import *，切分对外部 import 面零影响）。

2) 两套接入机制的能力差（C25）：registry 路径只能加顶层任务段，不能给 core 段加字段；gen 因需要 data.cond_* / model.sisr 只能子类化，代价是复制了 sync/preset/train 校验。收敛可行路径：TaskSectionSpec.core_cls 本已支持自定义 core 类型（@d:\codes\work-projects\SegTask\taskcore\config\registry.py:43，目前无人使用）→ 让 gen 以 core_cls=GenConfig 走注册表，同时把 sync 的公共体（num_classes 推断 / z_boundary 升级 / topology 写回 / resenc preset）提成 section_validators 同级的共享函数，gen 只覆写差异。这一步能同时消灭 F1/F4/F9 的漂移根因。

3) 扩展一个新任务的成本（步骤 8 前置观察）：新增段 ≈ 1 个 dataclass + 1 个 validate_* + 1 次注册 + 4 行 CLI 仪式，成本已经很低；真正的成本在给 core 加字段（要改 core.py 段定义 + 校验器 + 可能的迁移表 + gen 的副本）。

五、2026 年视角的业界对标
方案	能解决本层什么	代价/适配性
pydantic v2（Rust core）	F6（声明类型驱动强转）、F10（一次性聚合全部错误 + 导出 JSON Schema 供 YAML 编辑器补全）、按 arch 的 Discriminated Union 天然表达 unet/adm/edm2/sisr 段互斥	引入重依赖；dataclass → BaseModel 迁移面大。可折中：只用 pydantic.dataclasses 装饰现有 dataclass，字段定义不动
draccus / tyro / jsonargparse	直接替掉自研 apply_dotted_overrides：类型来自注解、支持嵌套与 list、错误统一。最小侵入、性价比最高	轻量依赖；CLI 参数面需一次性对齐
Hydra / OmegaConf structured configs	配置组合（defaults: 复用五任务公共 base）、${} 插值（消除 sync 里的部分手工派生）、sweep/multirun	Hydra 接管入口，与自建 DDP spawn/launcher 的交互需实测；对本项目"单文件 YAML + override"习惯是较大改变
nnU-Net v2 plans.json + dataset fingerprint	与 resenc_preset/save_best_preset 同源思想的自然延伸：由数据指纹自动派生 patch_size / 逐级 stride / 深度，根治 F5 那类人工整除性配置	需 make_data 侧指纹（已有 spacing 指纹雏形，@d:\codes\work-projects\SegTask\taskcore\config\core.py:143-145），步骤 3 联动
配置溯源（asdict + schema_version + git sha 写入 ckpt）	根除 _LegacySSLConfig 这类"为反序列化保留占位类"的债务（@d:\codes\work-projects\SegTask\taskcore\config\core.py:2429-2447），并让 F10 的迁移表有版本锚点	需 checkpoint 层配合，步骤 6 联动
fvcore/detectron2 LazyConfig（Python 即配置）	表达条件依赖强	可 diff/可审计性差，与本项目"YAML 单一真相 + 校验"取向冲突，不建议
六、建议改进路线（本轮不实施）
P0（修正确性，均为小改动）：F1 gen 补 stretch 升级（或直接把 'stretch' 从合法枚举中移除，让废弃彻底化）；F2 让 skip 真正生效或删除该参数；F4 gen 对齐大小写口径；F3 legacy __setstate__ 补 backing 字段。
P1（防迟到失败 / 统一口径）：F5 复用 _est_stage_tokens 的 stride 推导加整除性校验；F7 把 hoist 收进 load_core_and_task_config 并砍掉冗余入口；F6 override 改由字段注解驱动 + 错误统一为 ConfigError；F8 提供 finalize(cfg, task_cfg) 单函数取代四行仪式。
P2（架构演进）：core.py 按段/校验器拆包；gen 改走 core_cls 注册表路径并共享 sync 公共体（消灭 F9 差集）；引入 schema_version + 迁移表弃用窗口，清理死别名（_nested_dataclass_type）。
P3（前沿）：pydantic v2 或 draccus 二选一试点（建议先 draccus，仅替换 override/CLI 层）；数据指纹自动派生几何（nnU-Net 路线）。
七、如何验证（供后续实施轮使用）
现有测试已覆盖迁移契约与 I/O：test_d2_migration_contract.py、test_task_config_io.py、test_r2_review_hotfixes.py。建议补的回归点：

gen YAML 设 z_boundary_mode: stretch → sync() 后应为 edge_pad（F1）；
validate(skip={"train"}) 应真正跳过（或该参数不存在）（F2）；
patch_size=[60,128,128] + 5 级 → 应在 validate() 报错而非 forward 期 RuntimeError（F5）；
--override seg.predict.threshold=[0.3,0.6] 应成功（F6）；
旧式顶层 loss: YAML 经 registry.load_task_config("seg") 应与 seg_config.load_config 等价（F7）。
命令（本机环境）：D:\miniconda\envs\torch27_env\python.exe -m pytest tests/test_d2_migration_contract.py tests/test_task_config_io.py tests/test_r2_review_hotfixes.py -q

步骤 3 — data 层 A（离线与装配）：make_data.py、specs.py、loader.py、mixed_sampler.py、sampling.py。重点：预处理正确性（spacing/几何/fg 采样）、划分与配对逻辑、DataLoader 装配与 IO 吞吐、混采语义、DDP 下采样一致性。
一、结论摘要
正确性基线不错：几何校验（spacing/origin/direction 三元组只读头 fail-fast）、逐类 fg 索引的生成↔消费确实同构、val 确定性采样在 DDP 下成立、MixedBatchSampler 的 DDP 切分与 RNG 重放逻辑是对的。这三处是最容易写错的地方，都写对了。
主要风险集中在"离线烘焙"而非"在线装配"：npz 的幂等 skip 判据不完整、替换非原子、DDP 下自动构建无 rank 守卫且 tmp 文件名固定 —— 三者叠加可以产出"能读出 meta、但内容损坏且此后永远被 skip"的数据包。这是本层唯一的数据污染型风险。
契约层面确认一处明确违约：C8 要求"强配对缺失即报错"，但 discover_samples 对 image↔label 缺配对是 warning + 静默丢弃。
架构层面：装配逻辑存在四份（seg 的 build_dataloaders、公共 assemble_train_val_loaders、gentask 的复制、各自的 auto-build 段），与步骤 1 记的 D3（BasePredictor 薄→滑窗四处重复）是同型债务，且已经产生了能力漂移（零批次守卫只在其中一份里）。
二、契约核销（步骤 1 挂账 C6–C10）
契约	结论	依据
C6 npz 逐类 fg 索引，先抽类再抽位置；旧 npz 惰性回退	属实且同构。生成侧逐类 argwhere + 每类独立 cap + *_cls 对齐数组；消费侧按 *_cls 分组、rng.integers(len(per_cls)) 选类后选点；缺键返 None 退回合并采样，两条模式（z / cubic）都有	生成 @d:\codes\work-projects\SegTask\taskcore\data\make_data.py:131-180；消费 @d:\codes\work-projects\SegTask\taskcore\data\dataset.py:397-417、@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:992-1009、@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:1234-1247
C7 val 中心由 (seed, idx) 派生；z 等距 bin / Halton(2,3,5)；DDP 下仍确定	成立。中心只依赖 dataset 全局 idx，ValBatchShardSampler 只改"谁算"不改"算什么"；Halton 径向反演实现正确、夹取无越界。但存在两套派生公式（见 G14）与铺点质量问题（G13）	@d:\codes\work-projects\SegTask\taskcore\data\sampling.py:60-103、@d:\codes\work-projects\SegTask\taskcore\data\loader.py:40-70、@d:\codes\work-projects\SegTask\taskcore\data\patch_dataset_base.py:106-126
C8 pid 强配对，缺失即报错	部分违约。bbox/rw 走 _match_per_sample_paths 是 fail-fast（正确）；image↔label 是 warning + 丢弃	@d:\codes\work-projects\SegTask\taskcore\data\loader.py:195-215 vs @d:\codes\work-projects\SegTask\taskcore\data\loader.py:255-260；文档 @d:\codes\work-projects\SegTask\segtask_v1\docs\DESIGN.md:112
C9 make_data 同口径：spacing 归一化 + fg 索引 + meta skip + 几何校验	流程齐全，严格度有缺口。几何校验只读头、三量分别设容差，写得好；但 skip 判据漏 rw/bbox 来源（G3），meta 缺逆变换必需信息（G6）	@d:\codes\work-projects\SegTask\taskcore\data\make_data.py:190-213、@d:\codes\work-projects\SegTask\taskcore\data\make_data.py:67-115、@d:\codes\work-projects\SegTask\taskcore\data\make_data.py:357-386
C10 双源混采 mix_ratio，val 仅金标准	属实。每 batch 配额精确、val 只取主源；DDP 下各 rank 同 seed 生成同一全局序列后 strided 切分、等长，比例保真；epoch 长度 = n_secondary // coarse_per_batch // world_size（coarse-bound）。副作用见 G15/G16	@d:\codes\work-projects\SegTask\taskcore\data\mixed_sampler.py:71-91、@d:\codes\work-projects\SegTask\taskcore\data\mixed_sampler.py:197-222、@d:\codes\work-projects\SegTask\taskcore\data\loader.py:1023-1061
三、缺陷清单
P0 —— 数据污染 / 静默错误
G1 · DDP 下 npz 自动构建竞态，可产出损坏包 build_dataloaders 在每个 rank 上执行，auto-build 分支没有 rank0 守卫也没有 barrier：



loader.py:650-656
logger.info(
    "data.npz_dir=%s is empty/missing — auto-building via "
    "make_data.prepare_dataset (workers=%d). One-time cost; ",
    npz_dir, max(dc.num_workers, 1))
from .make_data import prepare_dataset
counters = prepare_dataset(
    cfg, npz_dir, workers=max(dc.num_workers, 1), overwrite=False)
调用点无任何同步：@d:\codes\work-projects\SegTask\segtask_v1\train.py:59-60。而临时文件名是确定性的 <pid>.npz.tmp：



make_data.py:390-410
tmp_path = out_p.with_name(out_p.name + ".tmp")
save_fn  = np.savez_compressed if compress else np.savez
...
with open(tmp_path, "wb") as fh:
    save_fn(fh, **payload)
# Windows：rename 前目标不能存在。
if out_p.exists():
    out_p.unlink()
tmp_path.rename(out_p)
N 个 rank × 各自 workers 个进程会向同一个 tmp 路径交错写入同一 pid，然后各自 unlink + rename。最坏结果是 zip 目录区可读、meta 可解析但 image/label 数据段错乱的包 —— 而 skip 判据只看 meta 键（G3），此后每次重跑都会 skipped，错误被永久固化。gentask 是同源复制品，同样无守卫：@d:\codes\work-projects\SegTask\gentask\data\loader.py:81-95。 修复方向：auto-build 收敛为 rank0 执行 + dist.barrier()；tmp 名加 os.getpid()/uuid 后缀；失败路径清理 tmp。

G2 · 替换非原子，与 C20 的"原子写"范式不一致 同上代码：unlink() 与 rename() 之间崩溃 → 目标文件消失且 tmp 也已改名失败，样本丢失。os.replace(tmp, out) 在 Windows 上走 MoveFileEx(REPLACE_EXISTING)，本身就是原子覆盖，无需先 unlink。checkpoint 层已用的正是 os.replace（步骤 1 C20），此处属于范式未统一。

G3 · 幂等 skip 判据缺 rw / bbox 维度 判据只覆盖 spacing_normalized / target_spacing / label_values / fg_subsample：



make_data.py:78-115
for key in _REQUIRED_SKIP_META_KEYS:
    if key not in meta:
        return False, f"missing meta key {key!r} (stale package)"
cond 有专门补丁（@d:\codes\work-projects\SegTask\taskcore\data\make_data.py:255-256），恰好说明 rw/bbox 是遗漏而非有意：用户事后配置 data.region_weight_dir 或 data.bbox_dir 再重跑 make_data，全部样本 skipped，训练静默地在"无 rw / 未裁剪"的旧包上进行，且日志一片正常。同理 src_image 的 mtime/大小不比对，源 NIfTI 就地修订后旧包永久复用。

G4 · C8 违约：image↔label 缺配对静默丢弃



loader.py:207-215
if missing_bases:
    ...
    logger.warning(
        "discover_samples: %d/%d image bases have no matching label "
        "under %s for any of %s; dropping them. Missing bases: %s%s",
label 目录后缀写错、或标注只完成了一半时，训练会以"少了 60% 样本"的状态正常启动。同一文件里 bbox/rw 的匹配是 raise FileNotFoundError（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:255-260），口径自相矛盾。建议改为默认报错 + 显式 data.allow_unpaired: true 才降级为警告。

G5 · seg 主路径绕过零批次守卫 守卫函数存在并被 cls/det/gen 经 assemble_train_val_loaders 使用：



loader.py:722-729
def ensure_train_batch_capacity(train_ds, batch_size: int) -> None:
    """``drop_last=True`` 下训练集不足一个 batch 会静默零批次，显式拦截。"""
但 seg 自己的 build_dataloaders 三条分支全部硬编码 drop_last=True 且从不调用它（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:1062-1087）。小样本冒烟 / 大 batch 调参时会得到"训练 loss 恒为初值、epoch 秒过"的无提示空转。

P1 —— 口径与工程质量
G6 · 逆变换信息不足，npz 不自足 resample_to_spacing 的输出尺寸取整，因此实际达成 spacing ≠ 名义 target：



dataset.py:168-172
D, H, W = vol.shape
new = []
for n, s, t in zip((D, H, W), src_spacing, target_spacing):
    new.append(max(1, int(round(n * float(s) / float(t)))))
return resize_3d(vol, new[0], new[1], new[2], is_label=is_label)
meta 记的却是名义 target（@d:\codes\work-projects\SegTask\taskcore\data\make_data.py:379-383）。薄 z 轴（如 D=24）时相对误差可达百分级，推理侧按名义 spacing 反算物理尺寸会系统性偏移。另外 meta 记了 bbox 与 orig_spacing，但没有 origin / direction，bbox=None 时也没有 resample 前的原始 shape（image_shape 是 resample 后的）→ 从 npz 单独回写原始物理空间的 NIfTI 不可行，必须回读源文件。建议补 achieved_spacing、pre_resample_shape、origin、direction。

G7 · 重采样无抗混叠 resize_3d 一律 zoom(order=1/0)，无 prefilter、无面积平均（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:562-566）。下采样（0.6mm → 1.5mm 这类常见 CT 归一化）会引入混叠高频，等于给模型喂了带伪影的低分辨率图。nnU-Net 对图像用 order=3 且对各向异性轴单独处理。此项与步骤 4 的面内 resize 同源，建议合并到步骤 4 一并定案。

G8 · 类均衡仅限卷内，不是全局类均衡 选类是"在该卷出现的类里均匀选"（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:1237-1240），而卷的选择是 idx % n_vols 的严格均匀（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:1151）。因此"只出现在 3% 卷里的稀有类"在整体上仍被稀释约 30×，逐类 cap 只解决了同卷内大器官淹没小结构的问题。meta.label_counts 已经现成，做卷级按类加权采样几乎零成本。

G9 · 患者级隔离只有 seg 有 group_id_regex 仅在 taskcore loader 中被消费（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:974-982）；cls 的划分只有"分层 or 随机"两档（@d:\codes\work-projects\SegTask\clstask\data\loader.py:73-82），det/ssl/gen 同理。同一患者多卷进入 train/val 两侧会让 cls/det 的验证指标系统性乐观。这是跨任务的契约缺口，不是 seg 的实现问题。

G10 · 四套划分的取整口径不一致 train_val_split 用 int(n*ratio) 下取整（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:410）、grouped_train_val_split 对组数下取整（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:449）、stratified_train_val_split 层内 round（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:532）、stratified_split_by_key 也 round（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:616）。同一 val_ratio=0.2 在不同开关下 val 规模不同，跨配置的验证指标不严格可比。

G11 · 装配层四份实现 build_dataloaders（seg 专用，含 auto-build + 划分 + DDP 三分支）与 assemble_train_val_loaders（cls/det/gen 用）功能重叠但不互相调用；gentask 又复制了一整份含 auto-build 的 build_dataloaders（@d:\codes\work-projects\SegTask\gentask\data\loader.py:58-99）。G5 的守卫漂移正是这种重复的直接产物。可收敛为：resolve_sources() + split() + assemble() 三段，seg 只多一个 mixed 分支。

G12 · 启动期 O(N) 扫描 + fg 索引内存未计入预算 每个 rank 都要：扫全部 npz 的 meta 探测 label_values（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:954-967）、再打开全部 npz 读 fg_coords 建索引（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:1128-1145）。fg 索引常驻主进程后随 dataset pickle 复制到每个 worker：每卷每类上限 50000×3×int32 ≈ 586 KiB，1000 卷 2 类 ≈ 1.1 GiB × (1 + num_workers) 份。而缓存估算只统计 image/label/rw（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:838-849），系统性低估。建议：fg 索引改为惰性按需读（LRU），或以 _index.npz 单文件汇总一次读入并用 np.memmap 共享。

G13 · Halton 铺点跨卷不去相关 halton_center(j, ...) 的 j 由 idx // n_volumes 派生（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:742），所有卷共用同一串 j → 所有卷在归一化坐标系里取的是同一组相对位置。低差异序列的优势在卷内，卷间反而完全相关；且 base=5 维在 j<5 时是线性递增的，spv 小时覆盖质量接近"斜线扫"。建议 per-volume 加偏移或 Owen scrambling（几行纯 numpy）。

G14 · 两套 val 确定性派生公式



sampling.py:67-74
    if is_train:
        return worker_rng.get()
    return np.random.default_rng((val_seed, sample_idx))
 
 
def deterministic_idx_rng(seed: int, idx: int) -> np.random.Generator:
    """cls/det 验证：中心由 ``(seed, idx)`` 确定性派生。"""
    return np.random.default_rng(int(seed) * 1_000_003 + int(idx))
seg 用 SeedSequence 元组（推荐做法），cls/det 用手工乘法 hash（seed*1000003+idx 在 seed 相邻时会产生高度相关的流）。两者都满足 C7 字面，但没有理由不统一到前者。另外 cls 复用 dc.split_seed 同时作为划分种子和采样种子（@d:\codes\work-projects\SegTask\clstask\data\loader.py:119），语义混用。

G15 · MixedBatchSampler 的全局重放开销随卡数线性增长 每个 rank 生成全部 _num_batches_global 个 batch（含 rng.shuffle）再丢弃 (world_size-1)/world_size（@d:\codes\work-projects\SegTask\taskcore\data\mixed_sampler.py:207-222）。逻辑正确（RNG 消费必须对齐），但可以改为"先 strided 切 sec_perm、gold 用 counter-based RNG 按 rank 直接定位"，避免主进程每 epoch 的无效工作。

G16 · gold≫coarse 的反向配置无告警 epoch 长度恒由 coarse 决定，gold 循环消费。若主源远大于副源，日志仍写 Gold is cycled/oversampled ~%.2fx（@d:\codes\work-projects\SegTask\taskcore\data\mixed_sampler.py:167-179），实际比值 <1 意味着"每 epoch 只见到一部分金标准"，文案误导且无校验。

G17 · data 层对 config 做就地写回 dc.label_values / dc.num_classes 在 loader 内被赋值并 cfg.sync()（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:954-969）。这与步骤 2 的 F8（sync→validate 隐式顺序契约）叠加：配置的最终形态取决于 data 层是否被调用过；resolved_config.yaml 恰好在其后保存才捕获到探测值，属于顺序巧合而非设计。建议提为显式的 finalize_from_data(cfg, paths)。

四、2026 年视角的可借鉴项
方向	可解决什么	适配性
数据指纹 / plans（nnU-Net v2）	_resolve_target_spacing（@d:\codes\work-projects\SegTask\taskcore\data\make_data.py:467-488）已是 median spacing 指纹的雏形，扩成 dataset_fingerprint.json（spacing 分布、shape 分布、CT 强度 0.5/99.5 分位、类频）即可派生 patch_size/stride/归一化参数，直接根治步骤 2 的 F5（整除性迟到失败）与本层 G8	纯自研、零新依赖，收益/成本比最高
分块存储（Zarr v3 + sharding / tensorstore）	现在是"整卷 npz → 整卷解码 → LRU 缓存"，随机 patch 抽取的真实读放大等于 卷体积 / patch 体积。分块存储可按 patch 邻域直读，同时消灭 G12 的 fg 索引常驻内存	需重写 dataset 读路径，属 P2/P3；但对大数据集是量级差异
WebDataset / MosaicML Streaming（MDS）	顺序分片 + 网络盘/多机友好，解决 savez 依赖 OS page cache 在共享存储上完全失效的问题	与"随机 patch 中心"范式冲突较大，仅在切换为"离线预抽 patch"时才合适
blosc2 / zstd 分块压缩	替掉二选一的 savez(不压缩，但大) vs savez_compressed(zlib，慢且不共享 page cache)。zstd-1 通常 3–5× 压缩且解压带宽 >1 GB/s	需引入 blosc2 或 zarr，属轻量依赖，需按第三条规则先说明必要性
每 batch 强制前景配额（nnU-Net oversample_foreground_percent）	现在是逐样本伯努利（rng.random() < fg_ratio），batch 内前景样本数是二项分布，方差大 → 梯度噪声大。改为"batch 内前 k 个样本强制 fg"	需在 sampler 层实现，与 MixedBatchSampler 可复用同一套 batch 级配额机制
组内分层划分（StratifiedGroupKFold 语义）	现在 group 与 stratified 是二选一且 group 直接覆盖 stratified（@d:\codes\work-projects\SegTask\taskcore\data\loader.py:974-982）。贪心版组内分层 ~30 行纯 Python，可同时满足患者隔离与类平衡	零依赖，配合 G9 一并做
划分产物落盘（splits_final.json）	现在划分由 (paths, ratio, seed) 隐式决定，数据目录一旦增删样本，历史实验的 val 集就无声改变，实验不可比	极小改动、极高收益，建议与 G10 一起定案
五、建议改进路线（本轮不实施）
P0：G1 auto-build 收敛为 rank0 + barrier、tmp 名唯一化；G2 改 os.replace；G3 skip 判据补 has_rw / src_bbox / 源文件指纹；G4 image↔label 缺配对默认报错；G5 build_dataloaders 接入 ensure_train_batch_capacity。五项均为小改动。
P1：G6 meta 补 achieved_spacing/origin/direction/pre_resample_shape；G10 统一取整口径并落盘 splits；G9 把 group_id_regex 提到公共划分入口供五任务共用；G14 统一到 SeedSequence 元组；G11 三段式收敛装配层（顺带消灭 gentask 副本）。
P2：G12 fg 索引惰性化/共享内存；G8 卷级类加权采样；G13 Halton per-volume scramble；G7 抗混叠（与步骤 4 合并定案）。
P3：数据指纹 → plans 自动派生几何（联动步骤 2 F5）；分块存储试点。
六、如何验证（供后续实施轮）
现有覆盖：test_mixed_sampler.py、test_sampling.py、test_assemble_dataloaders.py、test_review_batch4_fixes.py、test_data_pipeline_smoke.py。建议补的回归点：

先建含 rw 的 npz，再以 region_weight_dir 重跑 make_data → 应重生成而非 skipped（G3）；
两个进程并发 prepare_dataset 同一 out_dir → 产物应逐个可完整解码（G1/G2）；
label 目录后缀故意写错 → discover_samples 应抛错而非 warning（G4）；
len(train_ds) < batch_size 时 build_dataloaders 应报错而非返回 0 批次 loader（G5）；
resample_to_spacing 对 D=24、s=5.0→t=1.5 的体积，meta 中记录的 spacing 应等于实际达成值（G6）。


步骤 4 — data 层 B（在线读取与增强）：dataset.py、patch_dataset_base.py、patch_ops.py、patch_extract.py、augment.py。重点：四种 patch_mode 抽取口径逐位一致性、2.5D 折叠时机、GPU 增强正确性与伴随张量语义、随机性/可复现性、性能热点。
一、结论摘要
叶子层是全仓库最干净的一处分层：extract_cubic_patch / extract_z_patch_padded / resize_3d 三个原语单点维护，seg/cls/det/gen/ssl/predictor 六方共用，没有复制。步骤 1 担心的"两套并行抽取代码"不成立于算子层。
但成立于组装层：seg 走 SegDataset3D/Cubic/Whole + specs.py 三策略，cls/det 走 NpzPatchDatasetBase + extract_patch_by_mode，两套各自维护"过采样余量 / 安全中心域 / 验证 RNG"。C1 的"逐位一致"应精确表述为抽取算子一致、采样策略与余量不一致，且已产生 1 处可致数据污染的守卫漂移（H1）。
增强层质量高于预期：仿射与弹性融合为单次 grid_sample 且合成公式数学正确；随机性全程 CPU 采样避免 device→host 同步；Companion 伴随张量语义清晰；gentask 是 74 行薄封装而非复制。C5 全项守约。
本层最大的工程债不是正确性而是"面内 resize"：z_axis/2_5d/whole 三模式每样本对 (eD,512,512) fp32 跑一次 scipy.zoom(order=1)，既是 dataloader 头号热点，又无抗混叠（承接步骤 3 的 G7）。
C3/D1 定案：文档不一致属实，但三条折叠路径的代码等价性严格成立，无需修代码，只需改文档。
二、契约核销（步骤 1 挂账 C1–C5、D1、D2）
契约	结论	依据
C1 四模式五任务同语义、抽取逐位一致	部分属实。算子层一致；组装层三处分叉：①过采样余量只有 seg 有；②cubic 安全中心域 seg 按 max-FOV 尺寸算、cls/det 按 patch_size 算；③val RNG 两套公式	@d:\codes\work-projects\SegTask\taskcore\data\patch_extract.py:33-50 vs @d:\codes\work-projects\SegTask\taskcore\data\dataset.py:1213-1221、@d:\codes\work-projects\SegTask\clstask\data\cls_dataset.py:360
C2 z/2.5D 整面 resize、cubic 三轴 edge-pad、越界夹取	属实。edge-pad 单点、label 恒 order=0、zoom(mode="nearest") 且带形状防御性校正、中心夹取有 safe_center_range 退化分支。一处自相矛盾见 H2	@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:548-573、@d:\codes\work-projects\SegTask\taskcore\data\patch_ops.py:81-89
C3 / D1 2.5D 折叠时机、det 唯一例外	文档错、代码对。cls 两条路径产出布局逐位一致（dataset 侧 (D,H,W)→collate (B,D,H,W)；trainer 侧 (B,1,D,H,W)→fold→(B,D,H,W)）；折叠原语单点。ssl 同 cls。结论：只需改 seg WORKFLOW 的"唯一例外"表述	@d:\codes\work-projects\SegTask\clstask\data\cls_dataset.py:409-423、@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:203-205、@d:\codes\work-projects\SegTask\taskcore\engine\views.py:26-32
C4 单分辨率 max-FOV cube，缩放推迟 GPU	属实且实现正确。augment → center_crop → split_views（逐视图中心裁 + 单次 interpolate），全程无二次插值	@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:470-477、@d:\codes\work-projects\SegTask\segtask_v1\trainer\views.py:85-95
C5 空间变换同步伴随张量、强度仅 image、det 自动关闭	属实。det 用 dataclasses.replace 清零四个概率，各函数 prob<=0 / 掩码全 False 早退，零副作用（仅多一次 zeros_like 分配）。两处浪费见 H-补注	@d:\codes\work-projects\SegTask\dettask\trainer\det_trainer.py:51-63、@d:\codes\work-projects\SegTask\taskcore\data\augment.py:287-377
D2 gen 训推频谱不等价是否外溢	不外溢到本层。四模式的训练几何在 dataset 侧五任务共用，gen 的训推差异源于 predictor 侧滑窗（属 D3 域），本层无风险	@d:\codes\work-projects\SegTask\gentask\data\specs.py 仅覆盖 dataset_cls
D7 补注 私有名转正债务	本层新增一组：augment.py 的 _random_flip / _random_affine_elastic / _grid_dropout 三个旧签名包装已被 *_companions 版完全取代，疑似仅测试在用	@d:\codes\work-projects\SegTask\taskcore\data\augment.py:201-214,380-409,464-479
另核实：z_boundary_mode='stretch' 在 dataset 侧确为死配置——构造期校验后 _getitem_max_fov 无条件走 padded 抽取，测试已把这一点固化为断言，与步骤 2 的 F1 描述吻合（@d:\codes\work-projects\SegTask\tests\test_z_boundary_mode.py:198-201）。

三、缺陷清单
P0 —— 正确性 / 静默失效
H1 · whole 模式在 cls/det/gen 侧存在 LRU 缓存别名，守卫只存在于 seg 一份

resize_3d 在形状已匹配时直接返回入参（不拷贝）：



dataset.py:550-553
if arr.ndim == 3:
    D, H, W = arr.shape
    if D == target_d and H == target_h and W == target_w:
        return arr
seg 对此有显式守卫：



dataset.py:1322-1325
if img_r is img:
    img_r = img_r.copy()
if lbl_r is lbl:
    lbl_r = lbl_r.copy()
而公共叶子 extract_patch_by_mode 的 whole 分支没有（@d:\codes\work-projects\SegTask\taskcore\data\patch_extract.py:44-45），cubic/z 两个分支则由 extractor 内部无条件 copy 兜住。cls 后续的 np.ascontiguousarray(img_patch, dtype=np.float32) 对已连续的 fp32 数组是 no-op，于是 torch.from_numpy 直接共享 worker LRU 缓存内存（@d:\codes\work-projects\SegTask\clstask\data\cls_dataset.py:404-406）。当前靠 default_collate 必然分配新存储兜住，但 cls 的 GPU 增强是 inplace=True 的就地写（@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:130-134）——一旦有人绕过 collate 或加就地预处理，缓存被污染且后续所有样本静默错误。 → 修复：whole 分支末尾无条件 copy()，与另两个分支口径统一。一行，本层性价比最高的改动。

H2 · z_axis/2_5d 的 z 中心不受安全域约束，与 cubic 的取向自相矛盾

cubic 明确防"过半体素来自边界复制"：



dataset.py:1223-1226
def _sample_center(self, vol_idx: int, D: int, H: int, W: int) -> Tuple[int, int, int]:
    """采样中心 (d,h,w) 并夹匯至 _safe_center_range，以免 max-FOV cube 越界
    导致>50% 体素来自边界复制（偏移训练分布）。验证用逐样本确定性
    RNG（见 _sample_rng）。"""
z 路径完全不设防：return int(rng.integers(0, D_vol))（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:1010），前景切片采样也不夹取（同文件 1007-1009）。验证侧 z_grid_center 铺到 (0.5/S)*D，首尾 bin 同样严重 padded（@d:\codes\work-projects\SegTask\taskcore\data\sampling.py:87-90）。cls/det 的 z 路径复制了同一行为。薄卷（D_vol < pD，CT 常见）时整批样本都是"真实内容 + 大段复制"，训练与验证分布双双偏移，且无任何日志。

H3 · intensity_clamp 会把 grid_dropout 的洞抬回去，dropout 静默失效

clamp 基准在任何增强前取，dropout 把洞置 0，末尾 clamp 又把 0 抬回 clamp_lo：



augment.py:160-161
if c.intensity_clamp:
    image = torch.maximum(torch.minimum(image, clamp_hi), clamp_lo)
当 patch 不含空气（minmax 下 clamp_lo>0；软组织窗、zscore 归一化下都常见）时，dropout 效果被系统性削弱乃至完全抵消。默认 grid_dropout_prob=0.0（@d:\codes\work-projects\SegTask\taskcore\config\core.py:254）故当前非 live，但属"填了参数看着生效、实则无效"结构——与步骤 2 的 F2 同型。 → 修复：dropout 移到 clamp 之后，或用 clamp_lo 而非 0 作为填充值。

H4 · test_z_boundary_mode.py 的 main() 引用了已改名的函数，按文件头文档的运行方式直接 NameError

文件头写明 python test_z_boundary_mode.py（@d:\codes\work-projects\SegTask\tests\test_z_boundary_mode.py:28-29），而 main() 的 tests 列表里是 test_default_z_boundary_mode_is_stretch（同文件 461 行），实际定义名为 test_default_z_boundary_mode_is_edge_pad_and_stretch_auto_upgrades（同文件 49 行）。pytest 逐函数收集不受影响，所以一直没被发现。

P1 —— 性能 / 口径 / 可维护性
H5 · 面内 resize 是本层第一热点，且无抗混叠（承接 G7）

z_axis 路径每个样本：先 extract_z_patch_padded 拷出 (eD, 512, 512) fp32 slab（eD=64 时约 67 MB），再 zoom(order=1) 到 (eD,128,128)（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:959-965）。有效计算量是最终 patch 的 16 倍，且 4× 下采样用双线性等同抽样，混叠严重（nnU-Net v2 对 image 用 order=3 并对各向异性轴单独处理）。cubic 模式无此问题（先裁后不 resize）——这也解释了为什么同一份配置换 patch_mode 后吞吐差异巨大。 → 三条路可选：①make_data 侧按目标面内分辨率预烘焙（最彻底，顺带消灭 G12 的读放大）；②dataset 只发 slab，面内 resize 挪到 GPU 用 F.interpolate(antialias=True)（与 C4"缩放推迟到 GPU"的既有取向完全一致，纯 torch 零新依赖）；③至少给下采样加高斯预滤波。

H6 · 弹性形变不是 B-spline、也不是高斯平滑场，sigma/alpha 语义均失真

_elastic_grid_disp 是"粗网格 randn + trilinear 上采"（@d:\codes\work-projects\SegTask\taskcore\data\augment.py:272-284），得到的是 C0 连续、控制点处梯度不连续的分片线性场；elastic_deform_sigma 实为"下采倍数"而非高斯 sigma。三处措辞互不匹配：配置注释写"位移平滑度"、docstring 写"B-spline 随机位移场"、代码是 D/sigma 的网格尺寸。且 randn 上采后方差衰减，实际位移远小于标称 alpha（配置注释已承认"近似标称"）→ alpha 不可跨 sigma 比较，调参不可复现。

H7 · 强度增强函数的维度假设不统一

_random_brightness / _random_contrast 硬编码 5D（torch.ones(B,1,1,1,1)，@d:\codes\work-projects\SegTask\taskcore\data\augment.py:493,511），而 _random_gamma 与 clamp 用 image.ndim 动态（同文件 530,541）。det 靠 unsqueeze(1) 兜住（@d:\codes\work-projects\SegTask\dettask\trainer\det_trainer.py:164），gen 靠 squeeze_back 兜住（@d:\codes\work-projects\SegTask\gentask\data\augment.py:40-46）——都是调用方替被调方补维。任何新调用方直传 rank-4 即广播报错。

H8 · _grid_dropout_companions 的实现代价与整个 patch 同量级，且破坏 in-place 约定

每次分配 (B,1,D,H,W) 掩码 + num_holes 次五维高级索引写入，末尾 image * effective 返回新张量（@d:\codes\work-projects\SegTask\taskcore\data\augment.py:443-461）——inplace=True 时也会多出一份 batch 显存。用 masked_fill_ 或直接对切片区间赋值可零额外分配。

H9 · 同一语义两种写法：seg 用 rng.choice(fg_slices)（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:1008-1009），cls 在同一文件内混用 rng.choice（349 行）与 rng.integers（367 行）。choice 比 arr[rng.integers(len(arr))] 慢一个量级，且两者消耗的随机数不同 → 即使统一 seed 也不可比。

H10 · 两套 val 确定性派生在本层的具体后果（承接 G14）：val_grid_coverage=True 时 seg（interleaved j）与 cls（blocked j）覆盖同一 bin 集合、等价；关闭时 seg 用 SeedSequence((seed,idx))、cls/det 用 seed*1000003+idx（@d:\codes\work-projects\SegTask\taskcore\data\sampling.py:60-74），同一份数据在 seg 与 cls 下的 val patch 集合完全不同 → SSL 受控实验的下游评估跨任务不可比（D8 的隐性前提再破一处）。

H11 · seg dataset 把参数藏进实例属性：__getitem__ 先写 self._sample_idx = idx / self._current_vol_idx，再由 _sample_rng()、_val_coverage_pos()、_pack_extra_sample_tensors 隐式读取（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:931-933,1150-1152）。单 worker 顺序取样安全，但 cls/det 基类已收敛为显式传参 _item_rng_and_cov(idx)（@d:\codes\work-projects\SegTask\taskcore\data\patch_dataset_base.py:106-126）——seg 侧是尚未收敛的旧写法。

H12 · 三个独立 VolumeCache 共用同一个容量数字：img/lbl/rw 各持一份 max_volumes 相同的 LRU（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:710-712），实际驻留是 3×N 卷，而配置只有一个数字；且三者淘汰不同步，会出现 image 命中而 label 未命中的抖动。这使步骤 3 的 G12（缓存预算低估）再放大一次。

补注（C5 的两处浪费，不算缺陷但可省）：cls 的 table 源用 torch.zeros_like(img) 造假 label 走完整空间 warp（@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:200-201）——直接调 augmentor.apply(img) 即可省掉一次全尺寸 grid_sample；gen 的 cond 参与空间 warp 但不参与强度增强，若 cond 是同源低剂量/另一时相体，条件—目标的强度关系会被打散，建议提供同步开关。

四、值得肯定（不建议改动）
三层切分（patch_ops 原语 / patch_extract 按模式派发 / sampling 采样策略）是本仓库分层最干净的一处，五任务 + predictor 共用零复制。
仿射与弹性融合为单次 grid_sample，合成公式 G(x)=Θ(x+d)=affine_grid+M·d 数学正确（@d:\codes\work-projects\SegTask\taskcore\data\augment.py:354-362）——比 MONAI 默认的两次重采样质量更好且成本更低。
随机性工程：Bernoulli 掩码与逐样本标量一律 CPU 采样再异步搬卡，明确规避 device→host 同步；逐 rank seed + 7919*(rank+1) 分流，四个 trainer 口径完全一致。
preprocess_image 的 owned/只读/inplace 三态处理与 _open_npy_member_mmap 的零拷贝快路径，都显式承诺并保证了"与慢路径逐位一致"（@d:\codes\work-projects\SegTask\taskcore\data\dataset.py:255-294,486-491）。
gentask 的 augment 是 74 行薄封装而非复制——与 config 层 F1/F4/F9、data-A 层 G11 的复制病形成鲜明对比，这个模式应作为其他层收敛的范本。
五、架构评估
**"叶子共享、组装分叉"**是本层的准确画像。收敛路径：让 SegDataset* 也继承 NpzPatchDatasetBase，把"oversample 余量 + max_scale"作为基类可选参数（cls/det 传 1.0），采样策略全部收进 sampling.py。一次消除 H1/H2/H10/H11 四处漂移；代价是 seg dataset 签名变动、测试面较大，属 P2。
CPU/GPU 职责线画得对，唯一没遵守的是 H5。把面内 resize 挪上卡后，这条线才真正一致，并顺带获得 antialias 与吞吐——这是本层投入产出比最高的架构级改动。
augment.py 638 行不属于 core.py 那种堆积（一个类 + 12 个纯函数，职责单一）。真正可清的是三个已被 Companion 版取代的旧签名包装（D7 型死代码）。
六、2026 年视角的可借鉴项
方向	解决什么	适配性
F.interpolate(antialias=True) / nnU-Net v2 各向异性重采样	H5-① 混叠	纯 torch，零新依赖
GPU 侧数据管线（DALI / MONAI GPUTransform 思路）	H5-② 吞吐	已有 GPUAugmentor 基础设施，只需把 resize 挪进来
RandAugment / TrivialAugment 式策略层	超参从 ~20 个 prob 降到 2 个；2024–2026 医学分割上多次验证不劣于手调	可与现有逐变换 prob 并存，P3 试点
block-wise masking（MAE / SparK）	取代 grid_dropout，并与 ssltask 已有遮挡机制共用一套而非各写一份	与步骤 5 的 C14 联动
3D Copy-Paste / CarveMix / ObjectAug	稀有类收益远大于再调 affine 幅度；与 G8 卷级类加权互补	可直接复用现成 fg_coords 索引，零依赖
SeedSequence.spawn 全链路种子体系	统一 H10 的两套派生公式	极小改动
七、建议改进路线（本轮不实施）
P0（均 ≤5 行）：H1 whole 分支补 copy；H2 z 中心加安全域或至少加边
（接上）P0：H1 whole 分支补 copy()；H2 z 中心加安全域（或加"边界复制占比"告警）；H3 dropout 与 clamp 顺序定案；H4 修 main() 死引用。

P1：H5 面内 resize 上 GPU + antialias（与 G7 合并定案）；H6 弹性场改高斯平滑并把 alpha 归一为真实体素位移；H7 强度函数统一走 image.ndim；H8 dropout 就地化；H9 统一 rng.integers。

P2：SegDataset* 并入 NpzPatchDatasetBase、采样策略收敛到 sampling.py（消灭 H10/H11）；三个 VolumeCache 合一并计入预算（H12，联动 G12）；清理 augment.py 三个死包装（D7）。

P3：RandAugment 式策略层；3D Copy-Paste；预烘焙 patch / DALI 试点。

八、如何验证（供后续实施轮）
现有覆盖：test_z_boundary_mode.py、test_segtask_v1.py（多分辨率 max-FOV shape）、test_data_pipeline_smoke.py、test_review_batch4_fixes.py。建议补的回归点：

同一 npz + 同一 center，SegDataset3DCubic（oversample=1、max_scale=1）与 ClsPatchDataset 的 image patch 应逐位相等 —— C1 目前没有任何跨任务比对测试，这是最该补的一条；
whole 模式取样后对返回张量 in-place 写入，worker LRU 缓存不应被改变（H1）；
D_vol=8, pD=32 时边界复制切片占比应有上界（H2）；
grid_dropout_prob=1.0 + intensity_clamp=True + 全正强度 patch → 输出应存在等于 0 的洞（H3）；
固定 seed 下 GPUAugmentor 两次 apply 逐位一致，且 rank0/rank1 不一致（增强 RNG 分流契约目前只有注释、无测试）。
命令：D:\miniconda\envs\torch27_env\python.exe -m pytest tests/test_z_boundary_mode.py tests/test_data_pipeline_smoke.py tests/test_segtask_v1.py -q

步骤 5 — models 层：topology.py、factory.py、stem.py、blocks.py、unet*.py、resnet.py、convnext.py、mednext.py、adm_unet.py、edm2_unet.py、arch_compat.py。重点：几何派生单一真相源是否真的唯一、构造期不变量、归一化/激活/初始化选择、显存与 channels_last/编译友好性，以及 2026 年可借鉴的架构要素。
一、结论摘要
C11（build_topology 单一真相源）成立，且比文档描述更彻底：全仓无一处旁路重算——Config.sync @d:\codes\work-projects\SegTask\taskcore\config\core.py:1334-1339、unet/adm/edm2 三条 build 路径 @d:\codes\work-projects\SegTask\taskcore\models\factory.py:336,560 @d:\codes\work-projects\SegTask\taskcore\models\adm_unet.py:762 @d:\codes\work-projects\SegTask\taskcore\models\edm2_unet.py:657、gen 的 SISR/扩散/pipeline/predictor 全部读 topology。这是目前审到的最干净的一条跨层契约。
本层的系统性病灶不是"算错"，而是"同一语义、五处实现、口径各异"：五个 decoder（unet / unetpp / unet3p / adm / edm2）对"上采样后与 skip 尺寸不符"给出了 1 处硬报错 + 3 处静默 interpolate + 1 处一次性 warning 五种行为（I1）；梯度检查点的粒度也是 逐 stage / 逐 level / 只包融合卷积 / 逐块 / 完全绕过 五档（I2、I3）。
发现 1 处会静默改变数值语义的正确性问题：BatchNorm × 梯度检查点 的 running stats 双更新（I4），在 norm_type='batch' 或 mednext.dilated_reparam=True 下必然触发，直接违反 C13 的"数值与关闭时严格一致"。
C14/D9 定案：SSL.md 要求的稀疏前向与实现的"稠密+逐 stage 门控"不等价，且差距比文档承认的更大——门控只发生在 stage 边界，stage 内部的每一层卷积仍把被遮区当作真值 0 参与计算（I5）。同时 spark_encode 复制了 Encoder.forward 却漏掉了梯度检查点（I6）。
C15（依赖克制）在本层完全守约：blocks/resnet/convnext/mednext/unet* 只依赖 torch + numpy + einops，无 torchvision/monai/mmcv；CARAFE/DySample/BlurPool/ICNR/UniRepLKNet 重参数化全部自实现且实现正确。
最大的未兑现性能红利：本层已经把 SDPA、channels_last 兼容、torch.compile 友好（RoPE 显式绕 dict 缓存 @d:\codes\work-projects\SegTask\taskcore\models\blocks.py:530-533）都做对了，却在 ADM/EDM2 的注意力里退回手写 einsum+softmax（I7），并且全网没有任何权重初始化策略（I8）。
二、契约核销（步骤 1 挂账项）
契约	结论	依据
C11 build_topology 是几何派生单一真相源	完全属实。spatial_dims/in_channels 已只读化，grep 全仓无旁路推导；三 arch + 五任务 + predictor + pipeline 全部读 ModelTopology	@d:\codes\work-projects\SegTask\taskcore\models\topology.py:75-166、@d:\codes\work-projects\SegTask\taskcore\models\factory.py:336-341
C12 encoder.*/decoder.* 跨任务同名同形，SSL strict=False 直接命中	属实但有名字冲突隐患。build_backbone 与 build_model 共用 _build_unet_encoder_decoder，键名逐参数一致；但 ADM/EDM2 也叫 encoder.*/decoder.* 而结构完全不同 → unet SSL 权重误加载进 adm 会命中同名不同形，靠 load_state_dict 的 size-mismatch 硬报错兜住（可接受），但 seg 侧缺 0 命中校验（I9）	@d:\codes\work-projects\SegTask\taskcore\models\factory.py:520-540 vs @d:\codes\work-projects\SegTask\taskcore\models\adm_unet.py:607-642
C13 梯度检查点：逐 stage / 逐 level / ADM·EDM2 逐块；eval 零开销、数值严格一致	粒度属实，"数值严格一致"不成立。use_reentrant=False + preserve_rng_state=True 写法正确（DropPath 可复现），但 BN running stats 会被重算路径二次更新（I4）；unet3p/unetpp 的 checkpoint 只包了融合卷积、漏掉主要激活来源（I3）；stem 与 Downsample 从不包裹（I2）	@d:\codes\work-projects\SegTask\taskcore\models\blocks.py:49-64、@d:\codes\work-projects\SegTask\taskcore\models\unet.py:215、@d:\codes\work-projects\SegTask\taskcore\models\unet3p.py:115-119
C14 稀疏—稠密等价（SparK）	满密度退化严格成立（有单测逐位断言）；稀疏侧不等价：门控只在 stage 边界，且只有 InstanceNorm 有 masked 版本，norm_type='group'/'batch'（mednext/convnext 路线）仅告警不修复（I5）	@d:\codes\work-projects\SegTask\ssltask\models\spark_modules.py:144-190,101-141、单测 @d:\codes\work-projects\SegTask\tests\test_ssltask.py:1465-1481
C15 NMS/ROIAlign/AUC 等纯 PyTorch 自实现、不引重型库	本层完全守约，零隐性第三方依赖	import 面：torch / numpy / einops
C26 模型构造期 assert 仅内部不变量	基本守约，两处越界：assert channels % num_head_channels == 0 是用户配置驱动的条件（-O 下失效将退化为静默错误的头数）	@d:\codes\work-projects\SegTask\taskcore\models\adm_unet.py:277-279、@d:\codes\work-projects\SegTask\taskcore\models\edm2_unet.py:113-114,211
D8 SSL.md 统一骨干规格可否精确复现	可以，但默认值不等于规格：五级/总步长 16（stem_mode='conv3' stride=1 + 4×2）/ InstanceNorm+LeakyReLU 都是默认；通道默认 [32,64,128,256,512] 而 SSL.md 写 320 → 受控实验必须在 YAML 里显式钉死通道，否则"只变自监督目标一个变量"不成立	@d:\codes\work-projects\SegTask\taskcore\config\core.py:556-557,392-395,573
D9 SparK 稀疏 vs 稠密模拟	确认不等价，且差距被低估（见 I5）。这是方案一/二对照实验的核心变量，结论层面需在论文/报告中如实标注为"masked-dense 近似"而非 SparK	@d:\codes\work-projects\SegTask\ssltask\models\spark_modules.py:178-186
三、缺陷清单
P0 —— 正确性 / 静默语义改变
I1 · 五个 decoder 对"尺寸不整除"的处理有四种不同语义（口径分裂 + 静默降级）

经典 Decoder 硬报错（这是步骤 2 F5 依赖的那道最后防线）：



unet.py:270-274
if x.shape[2:] != skip.shape[2:]:  # 上采样后必须与 skip 同尺寸
    raise RuntimeError(
        f"DecoderLevel size mismatch after upsample: "
        f"x={tuple(x.shape[2:])} vs skip={tuple(skip.shape[2:])}. "
        f"Check input spatial dims are divisible by total encoder stride.")
而 UNet++ 只 warn 一次然后 interpolate 兜住 @d:\codes\work-projects\SegTask\taskcore\models\unetpp.py:105-116；ADM @d:\codes\work-projects\SegTask\taskcore\models\adm_unet.py:547-549 与 EDM2 @d:\codes\work-projects\SegTask\taskcore\models\edm2_unet.py:461-464 连 warn 都没有，直接 F.interpolate 静默续跑；UNet3+ 则用 adaptive_pool/interpolate 把"任意尺寸"当作正常工况 @d:\codes\work-projects\SegTask\taskcore\models\unet3p.py:81-91。后果：同一份非法 patch_size，decoder_type='unet' 秒级报错、换成 unetpp/adm/edm2 则训练全程带着一层隐性重采样跑完，且训推几何不再等价（与 D2 同型风险）。定案建议：把整除性校验上提到配置层（步骤 2 F5），五个 decoder 统一为硬报错。

I2 · 梯度检查点漏掉了激活占用最大的两处：stem 与 Downsample

Encoder.forward 只包裹 stage @d:\codes\work-projects\SegTask\taskcore\models\unet.py:215，而 self.stem(x) @d:\codes\work-projects\SegTask\taskcore\models\unet.py:194 与 self.downsamples[i-1](x) @d:\codes\work-projects\SegTask\taskcore\models\unet.py:203 恒不包裹。stem 输出是全网分辨率最高的特征图（(B, C0, D, H, W)，128³×32ch fp16 ≈ 128 MB/样本），conv7/dual stem 更是两层。配置注释自称"stem/上下采样/头不包裹（开销小）"@d:\codes\work-projects\SegTask\taskcore\config\core.py:594，与实际显存分布相反。

I3 · unet3p/unetpp 的 checkpoint 只包了融合卷积，省下的显存很少



unet3p.py:115-119
    branches.append(self.branches[i][j](src))
 
fused = torch.cat(branches, dim=1)
decoder_nodes[i] = checkpoint_if(
    self.grad_checkpointing, self.fusions[i], fused)
每个节点有 n 条分支卷积（n=5 时 5 条）+ 1 条融合卷积，被包住的只有第 6 条；torch.cat 产生的 n*cat_channels 大张量反而必须保留。UNet++ 同理（@d:\codes\work-projects\SegTask\taskcore\models\unetpp.py:104-126 的 upsamples[key] 与 gates[key] 在检查点之外）。用户开了 grad_checkpointing=True 会看到"显存几乎没降、速度却降了"。

I4 · BatchNorm × 梯度检查点：running stats 每步被更新两次（数值语义改变）

checkpoint_if 在反向时重跑前向 @d:\codes\work-projects\SegTask\taskcore\models\blocks.py:61-63，preserve_rng_state 只还原 RNG、不还原 BN 的 running buffer。因此任何被包裹的 BN 会以 momentum 连续作用两次，running_mean/var 相对不开检查点时系统性偏移；这些 buffer 又直接决定 eval/推理输出，也会被 EMA 平滑（C19）。触发面有两条且都不需要用户主动"选 BN"：

norm_type='batch' @d:\codes\work-projects\SegTask\taskcore\models\blocks.py:145-146；
mednext.dilated_reparam=True —— 该块内部强制使用 BN（fold 需要），与用户选的 norm 无关 @d:\codes\work-projects\SegTask\taskcore\models\mednext.py:181,189。
这是本层唯一一处"开一个纯显存开关会改变模型数值"的地方，直接违反 C13 与配置注释 @d:\codes\work-projects\SegTask\taskcore\config\core.py:595-596。修法很小：包裹前把 BN 切到 track_running_stats 冻结，或在重算路径上禁用 buffer 更新（PyTorch 官方做法是自定义 context_fn/BatchNorm 包装）。

I5 · C14/D9 定案：门控粒度是 stage 级，不是 conv 级 —— 稀疏等价名不副实



spark_modules.py:178-186
for i, stage in enumerate(encoder.stages):
    if i > 0:
        x = encoder.downsamples[i - 1](x)
    x = stage(x)
    vis = downsample_mask_to(vis_full, x.shape[2:])  # 该尺度可见掩码
    x = x * vis                                      # 重新置空被遮位点
一个 stage 默认含 2 个残差块 = 4~6 层卷积。子流形稀疏卷积保证每一层的可见输出都不含被遮位点的贡献；这里只在 stage 出口把被遮位点抹零，stage 内部各层的可见位点已经吸收了"被遮区=0"这一伪信号。掩码率 0.6、感受野 3³ 的情况下，第 2 层起绝大多数可见位点都被污染。单测 test_spark_encode_gates_masked_positions_to_zero 验证的是"被遮位点为零"，而稀疏等价的关键是"可见位点的值等于稀疏卷积的值"——这一条没有测试、也不成立。 配套问题：masked 归一化只覆盖 InstanceNorm @d:\codes\work-projects\SegTask\ssltask\models\spark_modules.py:95-98，换 norm_type='group'（mednext 默认）或 convnext 的 LayerNorm3d 时只发一条 warning @d:\codes\work-projects\SegTask\ssltask\models\spark_modules.py:136-140，统计污染照旧。结论：SSL 方案一在报告中应表述为 "masked-dense 近似"，若要真做方案一/二的受控对照，必须把门控下沉到块内（或引入 spconv）。

I6 · spark_encode 是 Encoder.forward 的复制品，漏掉梯度检查点与 cond 分支

同一段循环，Encoder.forward 走 checkpoint_if(self._stage_ckpt[i], stage, x)，spark_encode 直接 stage(x)。后果：SSL 预训练时 model.grad_checkpointing / grad_ckpt_encoder_stages 静默失效，显存与下游不可比（而 SSL 恰恰是最吃显存的一步）。同时 cond_in_channels>0 的分支也未复制（当前 ssl 无 cond，属潜在雷）。这是 G11/D3 那类"复制装配层"的病在 models 层的第三例。

P1 —— 工程质量 / 口径 / 性能
I7 · ADM/EDM2 的注意力是手写 einsum+softmax，O(N²) 显存被显式物化，且无 token 数护栏



edm2_unet.py:180-183
w_attn = torch.einsum(
    "nhcq,nhck->nhqk", q, k / float(np.sqrt(q.shape[2]))
).softmax(dim=3)
taskcore 自己的 SelfAttentionBlock 早已用 SDPA @d:\codes\work-projects\SegTask\taskcore\models\blocks.py:795-820（可走 flash/mem-efficient 后端），且配置层有 softmax token 上限护栏 @d:\codes\work-projects\SegTask\taskcore\config\core.py:1677-1696——这两项 ADM/EDM2 都没有。2.5D 下 128×128 = 16384 token，注意力矩阵单头 fp16 ≈ 512 MB，用户在 adm.attention_levels 里多写一个浅层就是必 OOM，且是运行期而非配置期失败。改成 F.scaled_dot_product_attention 是数学等价的直接替换。

I8 · 全网没有权重初始化策略（只靠 PyTorch 默认）

grep 全 models 包，nn.init 只出现在四处：SelfAttention 的 zero-init proj/ffn、ADM 的 zero_module、DySample 的 offset、ICNR。ResNet/ConvNeXt/MedNeXt/UNet 的所有 conv 都用 PyTorch 默认 kaiming_uniform_(a=√5)——这是众所周知的"为兼容旧 API 而保留的次优默认"，nnU-Net 明确用 He normal + negative_slope 对齐 LeakyReLU。另外残差分支末层 norm 的 gamma 零初始化（ResNet "zero-init residual"、几乎所有现代配方的标配）也没有，深 encoder（ResEnc-L/XL 预设）初期训练不稳定的成本是隐性的。ConvNeXt 侧靠 LayerScale=1e-6 部分补偿，ResNet 侧完全没有。这是本层投入最小、收益最确定的一处改动（一个 _init_weights + model.apply）。

I9 · seg 侧 pretrain 缺"0 命中报错"，与 cls/det 口径相反（C27 在模型交接面的落点）

det 的加载器明确写着"0 命中报错（几何不一致不静默）"@d:\codes\work-projects\SegTask\dettask\models\factory.py:39-41；而公共 BaseTrainer._load_pretrain_weights 只对 missing/unexpected 各发一条 warning 后照常训练 @d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:668-686。把一个 cls ckpt 喂给 seg（或前缀写错），日志里只有两行 warning，训练从随机初始开始且指标"看起来正常"。建议把 det 的命中统计上提到 BaseTrainer 作为公共策略。

I10 · attn_gate_target 对 decoder_type='unet' 是静默 no-op

build_backbone(cfg, attn_gate_target=...) 一路传到 _build_unet_encoder_decoder，但只有 UNetPPDecoder 接收该参数 @d:\codes\work-projects\SegTask\taskcore\models\factory.py:494；经典 Decoder 的构造根本没有这个入参 @d:\codes\work-projects\SegTask\taskcore\models\factory.py:502-515。gen 侧传 'upsample' 期望改变门控方向，在默认 decoder 下静默失效（门控方向仍是 skips）。与步骤 2 的 F2、步骤 4 的 H3 同型："参数接了、没人消费"。

I11 · decoder_blocks_per_stage 在 UNet++ 下被静默截断为首元素广播



factory.py:356-357
elif mc.decoder_blocks_per_stage:  # UNet++：首项广播到所有嵌套节点
    dec_counts = [mc.decoder_blocks_per_stage[0]] * max(expected_dec_calls, 1)
用户写 [2,3,4]，实际得到 [2,2,2,...]，无任何日志。UNet 路径下同样的列表会因长度不符而 ValueError @d:\codes\work-projects\SegTask\taskcore\models\factory.py:30-33——同一字段在两个 decoder 下一个报错一个静默，口径不一。

I12 · drop_path 的深度线性 ramp 在 encoder / decoder 各跑一遍

_make_drop_path_rates 在 enc/dec 两个 builder 里分别调用 @d:\codes\work-projects\SegTask\taskcore\models\factory.py:83（经 386-389 各构建一次），于是 encoder 从 0 爬到 drop_path_rate、decoder 又从 0 爬一遍。标准做法（ConvNeXt/Swin/nnU-Net ResEnc）是按网络总深度单调递增。结果是 decoder 深层（低分辨率、最该正则）反而 drop_path≈0，浅层却继承了 encoder 末端的高值语义。另：np.linspace(0, r, 1) == [0.0]，即 blocks_per_level=1 且只有单 stage 的极端配置下 drop_path 恒为 0（静默）。

I13 · MedNeXt UpKern 插值未做幅度归一，且 align_corners 与官方相反

upkern_remap_state_dict 用 trilinear + align_corners=True 把 k=3 核插到 k=5 @d:\codes\work-projects\SegTask\taskcore\models\mednext.py:325。docstring 已诚实标注 align_corners 与官方 False 不同；但更实质的是核元素和随之膨胀约 (5/3)³ ≈ 4.6 倍（插值保幅值不保和），depthwise conv 的输出尺度会整体放大，随后的 GroupNorm 虽能吸收一阶尺度、但残差支路与 pwconv 的相对配比已变。官方实现同样不归一，属"照抄了论文的缺陷"，建议至少提供 sum 归一开关并做一次消融。

I14 · 其他小口径不一致（各 ≤ 3 行）

ConvNeXt 的 expand_ratio 在配置里没有出口（factory 不传，恒 4.0），而 MedNeXt 有 mednext.expand_ratio @d:\codes\work-projects\SegTask\taskcore\models\factory.py:171-179 vs 225-239。
Upsample 的 nearest 分支也把 bf16/fp16 强制升 fp32 再插值 @d:\codes\work-projects\SegTask\taskcore\models\blocks.py:1482-1489——最近邻是纯 gather，无精度问题，这里白白多一份全尺寸 fp32 拷贝（decoder 最高分辨率处）。
ConvTranspose 默认 bias=True @d:\codes\work-projects\SegTask\taskcore\models\blocks.py:1436，而全网其他 conv 一律 bias=False（后接 norm）。
GroupNorm 组数不整除时静默折半并 warn @d:\codes\work-projects\SegTask\taskcore\models\blocks.py:150-159，而 MultiRF 的同一情形是显式报错并写了长篇理由 @d:\codes\work-projects\SegTask\taskcore\models\resnet.py:355-366。两种态度都合理，但应二选一。
_build_unet_encoder_decoder 里 num_fg 赋值后从未使用 @d:\codes\work-projects\SegTask\taskcore\models\factory.py:333；build_model 又把 block 计数解析逻辑整段复制一遍（仅为日志）@d:\codes\work-projects\SegTask\taskcore\models\factory.py:564-580 vs 343-359，是未来漂移点。
reparam_deploy（MedNeXt 推理期折叠）只有 seg predictor 接了 @d:\codes\work-projects\SegTask\segtask_v1\predictor\io.py:160-163，cls/det/gen/ssl 全无 → 同一骨干在别的任务上推理白白多付分支开销。属 D3 家族。
spark_encode 里 bool(mask_full.any()) 是每步一次 device→host 同步 @d:\codes\work-projects\SegTask\ssltask\models\spark_modules.py:167，与步骤 4 认可的"增强层全程规避 D2H 同步"取向相悖。
四、值得肯定（不建议改动）
ModelTopology 是全仓最好的一处抽象：frozen dataclass、一次算齐、决策树集中在 30 行内、新增 patch_mode 只改一处，并且真的没有旁路。步骤 2/3/4 反复出现的"复制病"在这里被彻底根治，应作为其他层收敛的范本（与 gentask augment 的 74 行薄封装并列）。
checkpoint_if 的三条注释把坑写全了（use_reentrant=False 为何必需、preserve_rng_state 为何必需、eval 零开销）@d:\codes\work-projects\SegTask\taskcore\models\blocks.py:49-64——除了 BN 那条（I4），这是踩过坑的写法。
AMP 下的 fp32 统计护栏成体系：GlobalResponseNorm @blocks.py:97-102、LayerNorm3d @convnext.py:26-32、resize_logits @unet.py:27-33 三处独立实现同一范式且都注明"同 adm_unet fp32 范式"。
各向异性下采样的构造期护栏是全仓最好的失败前置：compute_downsample_strides 的 nnU-Net 式调度（三条件：偶数 / 减半后 ≥4 / 该轴不落后 2×）@factory.py:261-287，配合 5 条兼容性 raise（blurpool/pixelshuffle/unetpp/hierarchical/mode 白名单）@factory.py:418-448——正是步骤 2 F5 应当照抄的范式。
_StatefulStageBuilder 的单计数器设计@factory.py:44-63：显式解决"factory 闭包另设计数器 → 双计数器漂移"，并带 exhausted 报错。
arch_compat.warn_ignored_model_fields@arch_compat.py:57-84：用"与全新实例逐字段 diff"来发现被静默忽略的旋钮，是可复用的通用手法（建议推广到 gen 的 F9 差集检测）。
重参数化实现（DilatedReparamBlock）数学正确：conv-BN fold、dilated kernel 展开、幂等 switch_to_deploy、del 释放训练态分支，均无误。
五、架构评估
分层是对的，边界画在了正确的位置：blocks（原语，1502 行但全是独立小类，非 core.py 式堆积）→ resnet/convnext/mednext（block 家族）→ unet/unetpp/unet3p（拓扑）→ factory（装配）→ topology（几何真相源）。build_backbone / build_model 共用同一条装配路径，是 C12 得以成立的结构性原因。这一层不需要重构，只需要补齐口径。
真正的架构债是"decoder 家族缺少共同基类"：五个 decoder 各自实现"上采样→对齐→融合"，于是 I1（尺寸语义）、I3（检查点粒度）、I10（门控参数）三处漂移全部长在同一根上。收敛路径成本很低：抽一个 align_or_raise(x, skip, policy) 工具 + 一个 DecoderBase.checkpoint_node() 约定，五个 decoder 各改 3~5 行，不动权重键名（对 C12 零影响）。
grad_checkpointing 目前是"一个布尔 + 一个 encoder 掩码"，表达力不足：它需要表达的是"包哪些模块"，而现状是 encoder 有掩码、decoder 只有全局开关、stem/downsample 无接口、unet3p/unetpp 只包了一小块、SparK 路径完全绕过。建议统一为"逐模块可选的 checkpoint policy"（PyTorch 2.x 的 apply_activation_checkpointing + 一个 should_ckpt(module) -> bool 谓词），一次消灭 I2/I3/I6。
arch 的三分（unet / adm / edm2）代价已经显现：ADM/EDM2 各自复制了 stem 装配、aux/DS 头装配、skip 对齐、attention 实现，arch_compat 的 40 条"被忽略字段"清单就是这份代价的账单。短期不建议合并（论文忠实是明确设计取向），但注意力实现（I7）与 skip 对齐（I1）应该共享——这两处与"论文忠实"无关。
六、2026 年视角的可借鉴项
方向	解决本层什么	适配性
F.scaled_dot_product_attention 全覆盖（含 ADM/EDM2）	I7：O(N²) 物化 → flash/mem-efficient 后端；同时自动获得 bf16 与 nested-tensor 支持	纯 torch、数学等价、零风险，本层最高性价比
_init_weights 策略层（He-normal + zero-init residual gamma + trunc_normal for ConvNeXt/MedNeXt）	I8	约 30 行、零依赖；建议同时把 ADM 的 zero_module 范式推广到 UNet 的 DS/aux 头（新头零初始 = 不扰动主路）
PyTorch 2.x apply_activation_checkpointing + selective checkpointing（torch.utils.checkpoint 的 context_fn / SAC）	I2/I3/I4/I6：按模块谓词统一策略，SAC 还能"只重算便宜算子、保留 matmul 输出"，通常比全量重算快 20–40%	原生 API，与现有 checkpoint_if 可共存渐进迁移
真稀疏路线：spconv / MinkowskiEngine（或块内门控）	I5/D9：让 SSL 方案一名副其实	引入重依赖，与"依赖克制"冲突 → 折中方案是把门控下沉到 block 内（每个 conv 后乘掩码），成本 ~10 行、无新依赖，等价性显著改善
nnU-Net ResEnc 2024 复盘结论（"ResEnc-L + 正确的几何 > 花哨架构"）	本层已实现 ResEnc 预设、MedNeXt、UniRepLKNet 重参数化，覆盖面足够；结论是不必再加 backbone，应把预算投到 I8 的初始化与步骤 3/4 的数据几何	零成本的"不做"决策
Primus / 3D 医学纯 Transformer（2025）与 Mamba-3D（VM-UNet 系）	当前 selfattn 是"CNN + 少量注意力块"，缺一条纯序列建模基线	P3 试点；SelfAttentionBlock 的 window/grid 分区代码可直接复用
Mask2Former / query-based 分割头	现在只有逐体素 1×1 头；query 头对"少数大器官 + 稀有小结构"的类不平衡天然更稳，且与 det 的 head 可共享	P3；与 taskcore 的 decoder.out_channels 契约兼容
LoRA / DoRA / adapter 式迁移	C12 现在是"全量 strict=False 迁移"，冻结只有 freeze_encoder 二值开关（det/cls）；LoRA 可在小数据下游上显著优于全量微调	需在 factory 加一层可选包装；P2
EDM2 的 post-hoc EMA（Karras 2024 §3）	EDM2 骨干已引入，但配套的"训练后合成任意 EMA 长度"没有；这是 EDM2 论文最实用的工程贡献之一	与步骤 6 的 EMA 联动
muP / 宽度缩放律	resenc_preset 换档时学习率靠手调；muP 可让 S→M→L→XL 共享一组超参	P3，需与 optim 层联动
fp8 / torch.compile + max-autotune 的 3D conv 实测	框架已具备 compile 与 channels_last_3d 接口，但没有任何基线数据说明它们在本仓 3D conv 上是正收益还是负收益	建议在步骤 6 补一张实测表，而不是继续留"默认关、需 benchmark"的注释
七、建议改进路线（本轮不实施）
P0（修正确性，均为小改动）：I4 BN×检查点双更新（包裹前冻结 running stats）；I5 SparK 门控下沉到 block 内 + 非 InstanceNorm 时改为报错而非 warning；I6 spark_encode 复用 Encoder.forward 的检查点路径；I1 五个 decoder 统一为硬报错（配合步骤 2 F5 的配置期整除性校验）。
P1（口径统一 / 免费性能）：I7 ADM/EDM2 换 SDPA；I8 加初始化策略层；I9 0 命中报错 上提到 BaseTrainer；I2/I3 检查点覆盖 stem/downsample/分支卷积；I10 attn_gate_target 接入经典 Decoder 或删参；I11/I12/I14 的静默降级与口径不一。
P2（架构演进）：抽 DecoderBase（对齐策略 + 检查点节点约定）；grad_checkpointing 升级为模块谓词策略；把 reparam_deploy 提到公共 predictor 层（与 D3 一并做）；build_model 与 _build_unet_encoder_decoder 的重复计数逻辑合一。
P3（前沿）：SAC 选择性重算实测；LoRA/DoRA 迁移通路；query-based 头 / Mamba-3D 基线；muP 缩放。
八、如何验证（供后续实施轮）
现有覆盖：test_selfattn.py、test_swa_lka.py、test_upkern.py、test_ssltask.py（SparK 等价性）、test_segtask_v1.py（多分辨率 shape）。建议补的回归点：

norm_type='batch' + grad_checkpointing=True/False 各跑 5 步，BN running_mean 必须逐位相等（I4，本层最该补的一条）；
满密度下 spark_encode 与 dense 前向逐位相等（已有）之外，补：掩码率 0.6 时"可见位点值 == 用真稀疏参考实现的值"——预期当前失败，作为 I5 的红灯基线；
model.grad_checkpointing=True 下 SSL SparK 的峰值显存应低于关闭时（I6，现状预期无差异）；
patch_size=[60,128,128] + 5 级，对 decoder_type ∈ {unet, unetpp, unet3p} 与 arch ∈ {adm, edm2} 五者都应报错（I1，现状只有 unet 报）；
decoder_blocks_per_stage=[2,3,4] + decoder_type='unetpp' 应报错或至少 warn（I11）；
同一 drop_path_rate 下，encoder 末块与 decoder 首块的 drop_prob 应满足全局单调（I12）；
build_backbone(cfg, attn_gate_target='upsample') + decoder_type='unet' + skip_attention=True 应报错或生效，不得静默忽略（I10）。

步骤 6 — engine 层：base_trainer.py、optim.py、amp.py、checkpoint.py、prefetch.py、dist_utils.py、launch.py、memory.py、bn_stats.py、views.py、base_predictor.py。重点：AMP/EMA/SWA/DDP 交互正确性、grad accumulation 与调度器语义、断点续训状态完整性、非有限守护、训练加速空间（torch.compile、fused optim、bf16、ZeRO、overlap、dataloader 瓶颈）。
一、结论摘要
这一层的正确性设计密度是全仓最高的：OptimStepResult 的 acknowledge 协议 + _check_boundary_scheduler_clock 把"scheduler 与 optimizer 时钟漂移"变成即时 RuntimeError（@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:433-459）；非有限守护按 fp16/bf16 分两条语义且 DDP 下用 all_reduce_flag_any 统一决策；pending 延迟 D2H 把每 micro-step 一次 .item() 降到日志步/边界一次，并显式分析了它与非有限守护的耦合。C17/C18 在五任务上是真守约。

本层的系统性病灶与前五层相反，是"能力倒挂"：任务层副本比公共层强。ZeRO consolidate（只有 seg/ssl 有）、状态指纹（只有 ssl 有）、resume 后 rank RNG 重分流（只有 seg/ssl 有）、预训练 0 命中报错（只有 cls/det factory 有）—— 四项都停在任务层，公共 BaseTrainer 反而是能力最弱的那份。公共化时按"逐字重复"抽取、而非按"能力并集"抽取，是根因。

发现 1 处会在特定配置下直接崩训练/挂死的问题：BaseTrainer._save_latest 无 ZeRO consolidate 却先按 rank 早退（J1）。

C16 有一处实质违约：SSL 的损失整体在 autocast 内计算，且方法内的 .float() 对 matmul 无效（autocast 会把 fp32 输入的 matmul 重新降精度）——dino_gram 的 Gram 矩阵、jepa、moco 的 InfoNCE 全部受影响，moco 甚至没有 .float()（J2）。

C27 在本层结案：不成立。ckpt 里明明存了 config，全仓却无一处读它做 patch_mode/spatial_dims/in_channels 交叉校验；cls/det 的"0 命中报错"是唯一的近似替代，且公共 pretrain 路径没有（J11）。步骤 2 顺延到本轮的挂账项到此闭合。

二、契约核销（步骤 1 挂账 C16–C23 + C27）
契约	结论	依据
C16 损失恒 fp32（autocast 外）+ logit clamp；AMP auto = Ampere+ 选 bf16	部分违约。auto 解析正确；seg/cls 是标准范式（compute_loss_fp32 / 显式 autocast(enabled=False)+clamp）；gen 在 autocast 外算但靠 loss 内部 .float()；det 在 autocast 内算、靠各 head 逐张量 .float()；ssl 全程在 autocast 内（J2）	@d:\codes\work-projects\SegTask\taskcore\engine\amp.py:76-97、@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:173-177 vs @d:\codes\work-projects\SegTask\dettask\trainer\det_trainer.py:219-223、@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:668-670
C17 优化步时钟；尾批按实际累积长度归一	属实。steps_per_epoch=ceil(len/accum)、one_cycle 映射 pct_start 不叠加外层 warmup、_effective_accum 尾组取真实尾长；五任务口径一致	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:142-165,251-259、@d:\codes\work-projects\SegTask\taskcore\engine\optim.py:182-191
C18 非有限守护 + DDP all-reduce(any) 统一跳步	属实且实现优雅。fp16 交 GradScaler、bf16/fp32 走 all_reduce_flag_any；跳步时 EMA 不推进、always_step_scheduler 为 ssl 单开一条时钟；护栏见 _check_boundary_scheduler_clock	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:360-431、@d:\codes\work-projects\SegTask\taskcore\engine\dist_utils.py:73-85
C19 EMA：验证/best 用 shadow、cpu offload、warmup	属实，但 BN 语义有洞。offload 用 pinned staging + 单次流同步，数学等价；apply_shadow/restore 异常安全。缺口：浮点 buffer（BN running stats）也按 decay 平滑且无收尾重估，而 SWA 侧有（J13）	@d:\codes\work-projects\SegTask\taskcore\utils\common.py:105-154、@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:736-752
C20 原子写 + 状态指纹 + 位精确 resume + rank0 + 异步	部分属实。原子写（os.replace）全覆盖、异步写有错误回传；指纹只有 ssl 有；"位精确"实为"epoch 边界 + 主进程 RNG 精确"，dataloader worker RNG / epoch 中途不可恢复（J12）；ZeRO 下公共 latest 保存会崩（J1）	@d:\codes\work-projects\SegTask\taskcore\engine\checkpoint.py:27-44,173-222、@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:329-341
C21 best 槽位 = EMA 权重，在线权重另存	属实。_save_best 三态清晰、extract_model_state_dict 统一读取、_restore_train_state 优先取 model_online_state_dict	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:527-571,611-616、@d:\codes\work-projects\SegTask\taskcore\engine\checkpoint.py:230-265
C22 seg 选模用 val_base_loss	属实。_CRITERION_TO_METRIC["loss"]=("val_base_loss","min")，验证侧用裸 base_loss 经 compute_loss_fp32 单独算；plateau 方向由 save_best_mode 派生、与 criterion 同源	@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:408-411、@d:\codes\work-projects\SegTask\taskcore\engine\optim.py:171-176
C23 DDP no_sync / 静态图 / bucket-view；ssl 手动 all-reduce	属实但缺护栏与实测。_ddp_no_sync 把 forward 也包进去（正确）；ssl 手动均值 all-reduce 数学等价但逐张量、无分桶无重叠（J10）；static_graph × no_sync × 梯度检查点三者叠加零测试（J9）	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:261-272,811-856、@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:549-563
C27 迁移交叉校验（步骤 2 顺延）	不成立。engine 无任何交叉校验；_load_pretrain_weights 连 0 命中都只 warning，与 cls/det factory 的 raise 口径相反；ckpt 内已有 config 却无人消费（J11）	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:668-686 vs @d:\codes\work-projects\SegTask\clstask\models\factory.py:113-117、@d:\codes\work-projects\SegTask\dettask\models\factory.py:59-63
三、缺陷清单
P0 —— 正确性 / 崩溃 / 静默语义改变
J1 · _save_latest 缺 ZeRO consolidate，且先按 rank 早退 → 多卡 + ZeRO 必崩


base_trainer.py:577-583
if not self._is_main:   # DDP：落盘仅 rank0
    return
bare = unwrap_compile(self.model)
state = {
    ...
    "optimizer_state_dict": self.optimizer.state_dict(),
ZeroRedundancyOptimizer.state_dict() 要求先全 rank 集合式 consolidate_state_dict(to=0)，否则 rank0 抛 "Optimizer state has not been consolidated"（若其它 rank 已进入下一次集合通信，则表现为挂死）。seg 与 ssl 的自留保存函数都写了这道守卫并明确注释"必须在 rank 早退之前调用"（@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:740-746、@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:365-372），而公共版没有。触发路径真实存在：det 在 encoder_lr_mult==1 时走 build_optimizer（@d:\codes\work-projects\SegTask\dettask\trainer\det_trainer.py:83-87），train.zero_redundancy_optimizer=True + 多卡即命中，且 _save_latest 是每 epoch 调用的（@d:\codes\work-projects\SegTask\dettask\trainer\det_trainer.py:404、@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:494）。修法：把 consolidate 提到 _save_latest 顶部、早退之前。三行。

J2 · SSL 损失在 autocast 内计算，方法内 .float() 被 autocast 击穿（C16 违约）


ssl_trainer.py:668-670
with autocast(device_type="cuda", enabled=self.use_amp,
              dtype=self.amp_dtype):
    loss, logs = self.method.compute_loss(batch)
与其余四任务不同，ssl 从未退出 autocast。后果分两层：

.float() 无效于 matmul：autocast 是按 op 决策的，输入是 fp32 也会把 matmul/bmm/linear 降回 bf16/fp16。dino_gram._gram_matrix 先 F.normalize 再 x @ x.transpose(1,2)（@d:\codes\work-projects\SegTask\ssltask\methods\dino_gram.py:98-120）、jepa 的 feat.float() 后接矩阵运算（@d:\codes\work-projects\SegTask\ssltask\methods\jepa.py:135-138）都属此类——写了 .float()、以为拿到 fp32，实际仍是低精度 matmul。Gram/协方差这类 O(N²) 求和正是 fp16 最容易累积误差的形态。
有的方法连 .float() 都没有：moco 的 logits 拼接、除温度 0.07、F.cross_entropy（@d:\codes\work-projects\SegTask\ssltask\methods\moco.py:176-179）；vicregl 的 variance/covariance 项只对 g1/g2 做了 float，协方差矩阵仍在 autocast 下算。
这是"靠每个方法作者自觉"的架构：新增一个方法就多一次漏 .float() 的机会。修法与其它任务对齐即可——把 compute_loss 调用移出 autocast，只把 encoder 前向留在里面（方法插件需暴露 forward_features / 或在 compute_loss 首行 with autocast(enabled=False)）。

J3 · _boundary_grad_norm 忽略传入的 parameters，恒对 self.model 算范数


base_trainer.py:297-298
elif not self._scaler_active:
    grad_norm_val = self._global_grad_norm()
而 _global_grad_norm 硬编码 self.model.parameters()（@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:471），完全无视 _optimizer_step_boundary(parameters=...) 这个入参（ssl 传的是 self.method.parameters()）。当前无 live bug——SSLMethod.parameters() 直接返回 self.module.parameters() 且 SSLTrainer.self.model = method.module（@d:\codes\work-projects\SegTask\ssltask\methods\base.py:142-143、@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:57），二者恒相等。但这条范数在 bf16 路径下是跳步判据的唯一来源：任何方法一旦引入 module 之外的可训练参数（可学习温度 / prompt / 独立 head），其非有限梯度将静默逃过守护，且梯度裁剪已经作用于它们（clip 用的是传入的 parameters）——两条路径对同一集合的定义不一致，是典型的埋雷。一行改动：_global_grad_norm(parameters)。

J4 · cls/det/gen 的 resume 不做 rank RNG 重分流
seg 与 ssl 在 resume 后显式调用 reseed_rank_rng（@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:788-791、@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:438-441），因为 ckpt 里的 rng_state 只是 rank0 的快照，而所有 rank 都从同一个文件恢复。cls 的 _try_resume 只调 _restore_train_state 就结束（@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:434-452），det/gen 同理。后果：resume 之后全部 rank 共用 rank0 的 torch/numpy/python 随机流 —— DistributedSampler 的索引切分仍不同，所以不会退化成"各卡训同一批数据"，但 dataloader worker 的 base_seed、以及一切走全局 RNG 的随机决策（fg/bg 抽取的伯努利、dropout 的 CPU 侧掩码）在各 rank 间完全相关，等效多样性下降且只在 resume 后发生，极难察觉。更好的做法：把 reseed 收进 _restore_train_state 的末尾（它已经知道 _rank 和返回的 start_epoch），一次修三个任务。

P1 —— 显存 / 吞吐 / 口径
J5 · state_to_cpu 先在 GPU 上 clone 再搬 CPU，异步保存瞬时翻倍显存


checkpoint.py:162-163
if isinstance(obj, torch.Tensor):
    return obj.detach().clone().cpu()
.clone() 在原设备分配，.cpu() 再拷一份 —— 保存瞬间 GPU 上多出一整份 model + optimizer state（Adam 是 2×参数量，latest ckpt 全都带上）。正确写法是 obj.detach().to("cpu", copy=True)：单次 D2H、零额外显存。这条与 train.save_async 的宣传语（"主循环不再被写盘阻塞"）叠加，实际效果是"用一次显存尖峰换写盘异步"，在显存吃紧场景（正是开 ema_device=cpu、梯度检查点的那批用户）容易在 checkpoint 时刻 OOM。另：_save_best 已经手工 detach().cpu().clone() 过一遍（@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:535-540），再进 state_to_cpu 又克隆一次，纯浪费。

J6 · "异步 checkpoint"只异步了写盘，深拷仍在主线程
submit 前必须 state_to_cpu（设计如此，注释也写明），于是每 epoch 一次 O(3×参数量) 的同步深拷贝仍卡在训练主循环里，异步只省掉了 torch.save 的序列化+落盘。2026 的对应答案是 torch.distributed.checkpoint.async_save 的 pinned staging（D2H 到常驻 pinned 缓冲 + 后台序列化），或退一步：latest 只存 optimizer 的 fp32 引用而不深拷（写盘期间冻结训练一小段）。至少应在文档里如实标注"异步不含深拷"。

J7 · 跳步导致 scheduler 走不完 horizon，且无任何观测
默认口径下 scaler skip / 非有限跳步都不推进 scheduler（正确的设计选择），代价是实际优化步数 < total_steps = epochs × ceil(len/accum)。cosine 到不了 eta_min、poly 末端 lr 不为 0、one_cycle 停在退火中途。目前只有 resume 时的 OneCycleLR 有漂移兜底（@d:\codes\work-projects\SegTask\taskcore\engine\optim.py:280-304），常态漂移零监测。健康指标里有 nonfinite_steps 却没有 opt_steps_actual / opt_steps_planned 与 scheduler.current_step —— 三个数进 monitor 几乎零成本（_collect_health_metrics 已经拿到 opt_steps）。

J8 · CudaPrefetcher 在 yield 之前取下一个 batch，dataloader 延迟没被隐藏


prefetch.py:83-91
try:
    next_cpu = next(it)          # ← 阻塞等 dataloader
except StopIteration:
    yield batch
    return
with torch.cuda.stream(stream):
    next_gpu = self._to_device(next_cpu)
yield batch                      # ← 消费者此时才开始 enqueue 计算
next(it) 是同步阻塞的（等 worker 返回 + collate），而它发生在当前 batch 的计算内核被 enqueue 之前 → 这段 CPU 等待与 GPU 计算完全不重叠。预取因此只隐藏了 H2D 拷贝，没有隐藏"dataloader 慢"这个更常见的瓶颈——而步骤 3/4 已经证明本仓 dataloader 恰恰是热点（G12 启动扫描、H5 面内 resize）。把取数与拷贝挪到 yield 之后即可（yield batch → 再 next(it) → 再发起拷贝），行为等价、代码更短。流同步与 record_stream 的处理本身是正确的（这一点比多数开源实现都严谨）。另：prefetcher 只搬顶层 dict 里的 Tensor，det 的 boxes/labels 是 list-of-tensor，走 _to_device 另一条路 —— 不算错，但 det 的预取覆盖面比其它任务小，性能对比时需知道。

J9 · static_graph × no_sync × 梯度检查点：三者叠加零测试、零护栏
_setup_ddp 同时开放 ddp_static_graph / ddp_find_unused_parameters / gradient_as_bucket_view，只对前两者同开发了一条 warning（@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:836-840）。但真正微妙的是 static_graph 与梯度累积 no_sync 的交互：static_graph 在首次迭代记录反传/通信模式并在后续复用，而 accum 的非边界迭代根本不触发 reducer；再叠加梯度检查点（重算改变反传的执行顺序），三者的组合语义 PyTorch 官方也只给了"注意事项"级别的说明。本仓把三个开关都暴露给用户、默认值散落在 config，却没有一个组合被测试覆盖（tests 里无 DDP 训练用例）。这是 C23 中风险最高、可观测性最差的一格；建议至少在配置层拦截 static_graph=True 且 grad_accum_steps>1 且 grad_checkpointing=True 或补一个 2-rank gloo 小测。

J10 · SSL 手动梯度同步逐张量 all-reduce，无分桶无重叠


ssl_trainer.py:561-563
torch._foreach_div_(grads, float(self._world_size))
for g in grads:
    dist.all_reduce(g, op=dist.ReduceOp.SUM)
数学正确（先除后加，fp16 scale 不受影响），但一个 UNet encoder+decoder 有数百个参数张量 → 数百次小 all-reduce，每次都吃一遍延迟，且发生在边界的"串行窗口"里（前向后向已结束，无可重叠计算）。DDP wrapper 的核心优势（25MB bucket + 反传重叠）在此全丢。低成本改法：_flatten_dense_tensors 打平成一两个大 buffer 做单次 all-reduce 再 unflatten（~10 行），或用 dist.all_reduce_coalesced。注释里"代价是无反传-通信重叠（可接受）"低估了实际代价——丢的不止是重叠，还有分桶。

J11 · 公共 pretrain 无 0 命中报错、无 ckpt-config 交叉校验（C27 结案）
_load_pretrain_weights 对 missing/unexpected 各发一条 warning 后照常训练（@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:677-686）。把一个 cls ckpt 喂给 seg、或前缀写错，日志里只有两行 warning，训练从随机初始开始、指标"看起来正常"（这与步骤 5 的 I9 是同一处，本轮确认它在 engine 层的落点）。更可惜的是：_save_best/_save_latest 都把 "config": self.cfg 写进了 ckpt（base_trainer.py:552,590），做 C27 要求的 patch_mode/spatial_dims/in_channels 交叉校验所需的信息已经在文件里，只是没人读。建议 _load_pretrain_weights 统一：①统计命中数，0 命中 raise；②若 ckpt 带 config，比对三元组不一致即 raise（附"如确需跨几何迁移请设 pretrain_allow_geometry_mismatch"）。这一步同时把 cls/det factory 里两份重复的 0 命中逻辑收编。

J12 · resume 的"位精确"名不副实
_restore_train_state 恢复 model/EMA/optimizer/scheduler/scaler/SWA/best/RNG，覆盖面确实是五任务里最全的，但三处缺口应在文档中如实标注：①dataloader worker RNG 不可恢复（worker 进程在 epoch 开始时按 base_seed 派生，ckpt 里没有、也无处注入）；②只能 epoch 边界恢复，epoch 中途崩溃会整轮重跑（大 epoch 场景代价显著）；③不校验 ckpt config 与当前 config 的兼容性——步骤 2 的 F3（老 pickle 绕过 __init__ 导致 property 抛 AttributeError）正是在这条路径上被触发的。C20 的表述宜下调为"epoch 边界精确 + 主进程 RNG 精确"。

J13 · EMA × BatchNorm：best 权重的 running stats 是"被平滑过的"，且无重估
ModelEMA 对全部浮点 buffer 一视同仁地按 decay 平滑，docstring 自己承认了这一点并建议"若引入 BN backbone，以 EMA 权重评估前重估 BN 统计"（@d:\codes\work-projects\SegTask\taskcore\utils\common.py:45-47）——但没有任何代码这么做：SWA 收尾有 _swa_recalibrate_bn，EMA 侧一条都没有，而 EMA 才是 best/部署槽位（C21）。结合步骤 5 的 I4 已证明"BN 触发面有两条且都不需要用户主动选 BN"（norm_type='batch'、mednext.dilated_reparam=True），这不是纯理论风险。最小修法：_save_best / EMA 验证前若 collect_bn_modules() 非空则告警（一行），完整修法是复用 _swa_recalibrate_bn 的机制给 EMA 加同款开关。

J14 · EMA offload 的隐藏常数：每次验证 2× 参数量的 PCIe 往返
ema_device="cpu" 的宣传是"省 1× 参数量 GPU 显存，数学等价"，未提的是：apply_shadow 逐张量 GPU→CPU（存 backup）+ CPU→GPU（换入 shadow），restore 再反向一遍（@d:\codes\work-projects\SegTask\taskcore\utils\common.py:130-154），且这些 copy 没有 pinned staging、没有 non_blocking（与 update 热路径的精心优化形成对比）。每次验证 4 次全模型跨设备拷贝；_save_best 里还要再来

步骤 6 — engine 层审查结果
范围：taskcore/engine/ 全 11 文件通读 + 五任务 trainer 的 AMP/累积/边界/EMA/ckpt/resume 段落。未改动任何代码/文档。

一、结论摘要
本层正确性设计密度全仓最高：OptimStepResult ack 协议 + _check_boundary_scheduler_clock 把时钟漂移变成即时 RuntimeError（@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:433-459）；非有限守护 fp16/bf16 两条语义 + all_reduce_flag_any；pending 延迟 D2H。C17/C18 真守约。
系统性病灶与前五层相反：能力倒挂——ZeRO consolidate、状态指纹、resume 后 rank reseed、0 命中报错，四项都只在某一两个任务层副本里，公共 BaseTrainer 反而最弱。公共化按"逐字重复"抽取、未按"能力并集"抽取。
1 处特定配置必崩（J1）；C16 在 ssl 上实质违约（J2）；C27 结案：不成立（J11）。
二、契约核销
契约	结论	依据
C16 损失 fp32	部分违约。seg/cls 标准范式；gen 靠 loss 内 .float()；det 在 autocast 内算；ssl 全程 autocast 内	@d:\codes\work-projects\SegTask\taskcore\engine\amp.py:76-97、@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:668-670
C17 优化步时钟	属实	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:142-165,251-259
C18 非有限守护	属实且优雅	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:360-431
C19 EMA	属实，BN buffer 有洞（J13）	@d:\codes\work-projects\SegTask\taskcore\utils\common.py:105-154
C20 ckpt	部分。原子写全覆盖；指纹只有 ssl；"位精确"实为"epoch 边界+主进程 RNG"	@d:\codes\work-projects\SegTask\taskcore\engine\checkpoint.py:27-44
C21 best 槽位	属实	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:527-571
C22 val_base_loss	属实	@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:408-411
C23 DDP	属实但缺护栏/实测（J9、J10）	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:811-856
C27 交叉校验	不成立（J11）	@d:\codes\work-projects\SegTask\taskcore\engine\base_trainer.py:668-686
三、缺陷清单
P0
J1 · _save_latest 缺 ZeRO consolidate 且先按 rank 早退 → 多卡+ZeRO 必崩 base_trainer.py:577-583 先 if not self._is_main: return 再调 optimizer.state_dict()。ZeRO 要求全 rank 先 consolidate_state_dict(to=0)。seg（@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:740-746）与 ssl（@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:365-372）都有守卫，公共版没有。触发真实：det 在 encoder_lr_mult==1 时走 build_optimizer，_save_latest 每 epoch 调用（@d:\codes\work-projects\SegTask\dettask\trainer\det_trainer.py:404、@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:494）。三行可修。

J2 · SSL 损失在 autocast 内，方法里的 .float() 被击穿 autocast 按 op 决策，fp32 输入的 matmul 仍降精度：dino_gram 的 Gram（@d:\codes\work-projects\SegTask\ssltask\methods\dino_gram.py:98-120）、jepa（@d:\codes\work-projects\SegTask\ssltask\methods\jepa.py:135-138）"写了 float 实际没生效"；moco 的 InfoNCE 连 .float() 都没有（@d:\codes\work-projects\SegTask\ssltask\methods\moco.py:176-179）。Gram/协方差正是 O(N²) 求和最怕低精度的形态。架构问题：靠每个方法作者自觉。

J3 · _boundary_grad_norm 忽略传入 parameters _global_grad_norm 硬编码 self.model.parameters()（base_trainer.py:298,471），无视入参。当前恰好等价（SSLMethod.parameters() 即 module.parameters()），但该范数是 bf16 路径跳步的唯一判据，而 clip 用的是入参集合——两条路径定义不一致。方法一旦引入 module 外可训练参数即静默漏检。一行可修。

J4 · cls/det/gen resume 后不做 rank RNG 重分流 seg/ssl 有 reseed_rank_rng，cls（@d:\codes\work-projects\SegTask\clstask\trainer\cls_trainer.py:434-452）/det/gen 无 → resume 后全 rank 共用 rank0 随机流。建议收进 _restore_train_state 末尾，一次修三个任务。

P1
J5 · state_to_cpu 先 GPU clone 再 .cpu()（@d:\codes\work-projects\SegTask\taskcore\engine\checkpoint.py:162-163）→ 保存瞬间 GPU 多出一整份 model+optimizer。应 .detach().to("cpu", copy=True)。且 _save_best 已手工 clone 过，再 state_to_cpu 又克隆一次。

J6 · "异步 ckpt"只异步写盘，深拷仍在主线程（O(3×参数量)/epoch）。对标 torch.distributed.checkpoint.async_save 的 pinned staging。

J7 · 跳步致 scheduler 走不完 horizon 且零观测：cosine 到不了 eta_min、poly 末端非 0。_collect_health_metrics 已有 opt_steps，补 planned/actual + scheduler.current_step 几乎零成本。

J8 · CudaPrefetcher 在 yield 前 next(it)（@d:\codes\work-projects\SegTask\taskcore\engine\prefetch.py:83-91）→ dataloader 阻塞发生在当前 batch 计算 enqueue 之前，只隐藏了 H2D、没隐藏 dataloader 延迟（而步骤 3/4 已证明 dataloader 才是热点）。把取数挪到 yield 之后即可。流同步/record_stream 本身写得比多数开源实现严谨。

J9 · static_graph × no_sync × 梯度检查点零测试零护栏：三开关全暴露，只对 static_graph×find_unused 有一条 warning（base_trainer.py:836-840）。tests 无任何 DDP 训练用例。C23 里风险最高的一格。

J10 · SSL 手动同步逐张量 all-reduce（@d:\codes\work-projects\SegTask\ssltask\trainer\ssl_trainer.py:561-563）：数百次小通信、无分桶、发生在无可重叠的串行窗口。_flatten_dense_tensors 打平约 10 行。

J11 · 公共 pretrain 无 0 命中报错、不读 ckpt 里的 config：config 已写入 ckpt（base_trainer.py:552,590），做 C27 交叉校验的信息就在文件里却无人消费；cls/det factory 的两份 0 命中逻辑应上提收编。

J12 · resume "位精确"名不副实：worker RNG 不可恢复、只能 epoch 边界恢复、不校验 ckpt config（步骤 2 F3 正是在此触发）。

J13 · EMA × BN：EMA 平滑所有浮点 buffer 且无收尾重估（SWA 有、EMA 没有），而 EMA 才是 best/部署槽位。结合步骤 5 I4（BN 有两条非自愿触发路径），非理论风险。

J14 · EMA offload 隐藏常数：apply_shadow/restore 逐张量跨设备 copy、无 pinned、无 non_blocking（与 update 热路径的精心优化反差明显），每次验证 4 次全模型往返。

J15 · build_optimizer_with_lr_mult 不支持 ZeRO（@d:\codes\form...\taskcore\engine\optim.py:131-142），且 encoder_lr_mult==1 与 ≠1 的 param_groups 数不同（2 vs 4）→ 两条路径的 optimizer state 不可互换 resume。

J16 · launch.install_term_handlers 用 os._exit（@d:\codes\work-projects\SegTask\taskcore\engine\launch.py:71-83）：绕过 finally/atexit → 异步 ckpt 队列丢失、.tmp 残留。J17 · find_free_port TOCTOU（同文件 39-45）：bind→close→NCCL 再 bind，单机并发训练偶发 rendezvous 失败。J18 · memory.estimate_train_memory 不计 EMA backup（验证换入时的 1× 参数量）。

四、值得肯定
OptimStepResult.acknowledge + _check_boundary_scheduler_clock：把最难发现的时钟漂移变成即时异常，全仓最好的防御式设计，建议推广到 EMA/method 时钟。
relocate_optimizer_state + _iter_leaf_optimizers（checkpoint.py:108-153）：踩过 fused Adam×ZeRO×map_location 的坑才写得出。
all_reduce_bn_running_stats_ 以 num_batches_tracked 加权（dist_utils.py:88-114），与单进程累积平均严格相等，比常见的直接平均正确。
pending 延迟 D2H 机制（四任务同源）与非有限守护的耦合分析。
launch 的 PR_SET_PDEATHSIG 竞态自杀分支；bn_stats 的 momentum=None 累积平均 + 上下文恢复。
五、架构评估
BaseTrainer 边界画对了，但五份 _train_epoch 仍高度同构：pending/边界/健康计数/prefetch 四段在 seg/cls/det/gen/ssl 逐字重复。这是 D3（predictor 薄）的训练侧对应物。收敛方向：基类提供 micro-step 骨架模板，任务只实现 forward + loss。
checkpoint 布局三套（seg keep-last-k、cls/det best+latest、ssl 指纹双文件），能力互不相交 → 建议统一 CheckpointManager（槽位+保留策略+指纹+ZeRO consolidate+异步）。J1/J11/J12 都消灭在这一步里。
能力倒挂是本层唯一的架构性结论，与 config/data/models 三层"公共层强、副本弱"正好相反，修法明确：按能力并集重抽。
六、2026 对标
torch.distributed.checkpoint.async_save + pinned staging → J5/J6。
Selective Activation Checkpointing / apply_activation_checkpointing → 与步骤 5 I2/I3 联动。
EDM2 post-hoc EMA（Karras 2024）：训练后合成任意 EMA 长度，与 J13/J14 同一处改动。
schedule-free AdamW（2024）：无需 total_steps，根治 J7 的 horizon 漂移，对"改 epochs 后 resume"的科研流特别友好，P3。
FSDP2 结论建议是不做：3D UNet 瓶颈在激活不在参数，ZeRO-1 已够。
torch.compile + CUDA Graphs（reduce-overhead）需实测（步骤 5 已提缺基线）。
七、建议路线（本轮不实施）
P0：J1 consolidate 提到早退前；J2 ssl 损失移出 autocast；J3 _global_grad_norm(parameters)；J4 reseed 收进 _restore_train_state。四项均 ≤5 行。
P1：J5 .to("cpu", copy=True)；J8 prefetcher 取数后移；J11 0 命中报错 + ckpt-config 交叉校验上提；J7 时钟指标进 monitor；J10 梯度打平分桶；J13 EMA×BN 告警。
P2：CheckpointManager 统一三套布局；micro-step 骨架模板收敛五份训练循环；J9 补 2-rank DDP 组合测试或配置层拦截。
P3：async_save/SAC/post-hoc EMA/schedule-free 试点。
八、如何验证
现有覆盖：test_checkpoint_resume.py、test_review_r7_optimizations.py、test_todo_p_regressions.py、test_ddp_metric_all_reduce.py。建议补：

ZeRO + 2-rank gloo，调 _save_latest 应成功落盘（J1，本层最该补的一条）；
ssl 在 bf16 下 dino_gram 的 Gram 矩阵 dtype 应为 fp32（J2）；
构造 module 外可训练参数，其非有限梯度应触发跳步（J3）；
resume 后两 rank 的 torch.randn(1) 应不相等（J4）；
state_to_cpu 期间 max_memory_allocated 增量应 ≈0（J5）；
人为跳步 N 次后 scheduler.current_step 应等于 opt_steps_actual（J7）。
命令：D:\miniconda\envs\torch27_env\python.exe -m pytest tests/test_checkpoint_resume.py tests/test_review_r7_optimizations.py tests/test_todo_p_regressions.py -q

步骤 7 — metrics / utils / monitor：指标可加性与 DDP 归约正确性、边界情形（空前景/单类）、日志与随机种子、监控面板的成本与信息密度。
步骤 8 — 全局串联与跨层契约审查：把步骤 2–7 的结论与步骤 1 的契约表对齐，检查五任务共享层的抽象边界是否合理（是否过度共享 / 共享不足）、shim 债务、扩展一个新任务的成本。
步骤 9 — 业界对标与升级路线：按 2026 年视角给出可借鉴、可适配、可新增的清单（跨自然图像/NLP/LLM/VLM），并输出按优先级与投入产出排序的改进路线图（P0 修复 → P1 加速 → P2 架构演进 → P3 前沿实验），标明与现有设计的兼容性。




──────────────────────────────────────────────
核验记录（2026-07-25，对步骤 0–6 全部进展逐条比对当前代码；静态比对，未运行测试）：

总体结论：审查记录整体定位准确，但与当前代码已脱节——约 15 个 P0/P1 项在记录写成后已被修复但仍挂账；其余多数发现仍成立；另发现 3 条新问题与若干记录偏差。

一、已修复、应核销的项（当前代码已含修复，勿重复开工）
- config：F1（gen sync 已有 stretch→edge_pad 自动升级，@gentask/config/validation.py）；F2（core validate 已按 section 拆分、skip 真实生效，@taskcore/config/core.py validate()）；F3（_ensure_model_geometry_backing 已补齐 _spatial_dims/_in_channels，@taskcore/config/model_migration.py）；F4（gen resenc_preset 已 .lower()，@gentask/config/validation.py:377-380）。
- data-A：G1（auto-build 已 rank0 构建 + broadcast 成败标志 + barrier，@taskcore/data/loader.py:643-718，set_device 先于 init_process_group，实现正确）；G2（tmp 带 pid + os.replace 原子写，@taskcore/data/make_data.py:401-427）；G3（skip 校验含 has_rw/src_bbox 等 meta 完整性检查）；G4（配对缺失聚合报 FileNotFoundError + allow_unpaired 开关，@taskcore/data/loader.py:165-219）；G5（ensure_train_batch_capacity 已存在并在两处调用，@taskcore/data/loader.py:769,810,1065）。
- data-B：H1（patch_extract.py 已统一四模式抽取原语，whole 路径含防缓存污染 copy）；H3（grid_dropout 已移到 intensity_clamp 之后，@taskcore/data/augment.py:159-163）；H4（test_z_boundary_mode.py main() 引用已修正，:461）。
- models：I4（checkpoint_if 已带 _freeze_bn_running_stats context_fn，@taskcore/models/blocks.py:52-105）；I6（spark_encode 已按 _stage_ckpt 走 checkpoint_if，@ssltask/models/spark_modules.py:176-188）。
- engine：J1（_save_latest 的 ZeRO consolidate 已提到 rank 早退之前，@taskcore/engine/base_trainer.py:580-584）；J2（ssl compute_loss 已移出 autocast，@ssltask/trainer/ssl_trainer.py:667-671）；J3（_global_grad_norm 已支持自定义参数集合并与 clip 同集合）；J4（resume 已在 _restore_train_state 内 reseed_rank_rng，:655，四任务共用）；J14 子项（EMA update 已加 pinned staging，@taskcore/utils/common.py:89-121）。

二、复核后仍成立的项（维持原挂账）
- config：F5（整除性校验仍未上提配置层，core 仅 lift_2_5d_to_3d 与 hierarchical stem 有整除检查；unet/unetpp/adm/edm2 常规路径仍靠模型层兜底）；F6（coerce_override_value：int/float 抛 ValueError 非 ConfigError，标量→list 联合类型不支持，@taskcore/config/task_io.py:30）；F7（registry 通用路径缺 hoist_legacy_seg_sections，见新增 N1）；F10（core.__getattr__ 告警文案仍写 segtask_v1.config.%s，@taskcore/config/core.py 末尾）。
- data-A：G6、G7、G9（group_id_regex 仍仅 taskcore loader 支持）、G10、G12、G13、G14/H10（SeedSequence vs seed*1000003+idx 双口径，@taskcore/data/sampling.py:72-74）、G15、G17（loader 回写 dc.label_values 再 cfg.sync()，@taskcore/data/loader.py:1016-1018）。
- data-B：H2（训练 z 采样兜底仍 rng.integers(0, D_vol) 无 safe range，@taskcore/data/dataset.py:1010，cubic 有 clamp 而 z 无）、H5、H6、H7（@taskcore/data/augment.py:514 硬编码 5D）、H8、H9、H11、H12。
- models：I1（adm/edm2 尺寸不匹配仍静默 interpolate 无 warn，@taskcore/models/adm_unet.py:545-549、@taskcore/models/edm2_unet.py:461-465）、I2（stem/downsamples 未纳入检查点，@taskcore/models/unet.py:194,203）、I3、I5（SparK 门控仍 stage 边界级；bool(mask_full.any()) D2H 同步仍在 :167）、I7、I8、I10、I11（decoder_blocks_per_stage 仅取 [0] 广播，@taskcore/models/factory.py:357,578）、I13。
- engine：J5（state_to_cpu 无条件 clone().cpu()；_save_best EMA 路径二次 clone，@taskcore/engine/base_trainer.py:538-566）、J7、J8（见偏差 P2）、J10（ssl 逐张量 all_reduce 未桶化，@ssltask/trainer/ssl_trainer.py:560-562）、J11、J13（EMA 无 BN 重校准；SWA 有 _swa_recalibrate_bn）、J14 余项（apply_shadow/restore 仍逐张量同步 copy 无 pinned/non_blocking，@taskcore/utils/common.py:142-153）、J15、J16、J17、J18（见偏差 P4/新增 N2）。
- 文档：D1 仍成立（seg WORKFLOW:165"唯一例外"与 cls WORKFLOW:148 / 顶层 README:38 不一致），"文档错、代码对"定案维持。D2/D5/D7/D8/D9 定案抽查与现状一致。

三、记录本身的偏差（需修正）
- P1 体量数据全面过期：taskcore 现为 56 py / 20,473 行；core.py 2472（记录内部还有 2168 与 2464 两个数字自相矛盾）、blocks.py 1541、dataset.py 1351、loader.py 1172、base_trainer.py 1045、adm_unet.py 1050、edm2_unet.py 923、make_data.py 758。多处 @path:line 引用已漂移。
- P2 J8"这段 CPU 等待与 GPU 计算完全不重叠"表述过强：GPU kernel 异步 enqueue 下，next(it) 阻塞可与上一 batch 的 GPU 残余计算重叠；但 fetch 在 yield 之前、无法与当前 batch 计算重叠的结论方向正确。
- P3 F9 的 mednext 子项失效：gen _validate_model 的 backbone 白名单只允许 resnet/convnext，mednext 路径已被挡住；selfattn/multirf 无校验的子项仍成立。
- P4 J18 部分缓解：offload 模式下 EMA backup 现已落 CPU（@taskcore/utils/common.py:137-141）；非 offload 情形仍占 1× GPU 显存且 memory.py 不计。
- P5 J15 引用路径笔误 d:\codes\form...（应为 work-projects）。

四、新增发现
- N1（低，F7 精确化）：seg 配置两条加载路径对 legacy 顶层 loss:/predict: 段容忍度不一致——taskcore/config/registry.py 的 load_task_config(path,"seg") 走 load_core_and_task_config，不做 hoist_legacy_seg_sections；segtask_v1/seg_config.py 的 _load_raw 有 hoist。同一份旧 YAML 走前者会因 unknown key 报错、走后者正常。建议在 TaskSectionSpec 加可选 preprocess_raw 钩子统一。
- N2（低）：ModelEMA.apply_shadow 非 offload 时 _backup 在 GPU 上占 1× 参数显存且 memory.py 不计入（J18 的补充精确化）。
- N3（信息）：_save_best 仅 rank0 早退且不保存 optimizer，无 ZeRO consolidate 需求——J1 只改 _save_latest 的修复范围正确、无遗漏。

五、核验后的建议
1) 仍成立项按原优先级推进，建议首批：F5+I1（整除性校验上提配置层、adm/edm2 加 warn/报错）、H2（z 采样 safe range）、F7/N1（registry hoist 统一）、J10（梯度桶化）。
2) 涉及运行时行为的项（DDP 挂死、prefetch 重叠度、EMA 换入耗时）建议在 torch27_env 以针对性小实验验证后再定案。
3) 步骤 7–9 尚未开始，按原计划继续。
──────────────────────────────────────────────


2 分割项目代码审查（需结合对应 readme/design/workflow 一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：

分割项目 = 公共框架层 `taskcore` + 任务层 `segtask_v1`，审查按此两级展开。代码大致分 5 部分，数据读取、模型构建、数据增强/处理、训练全流程（含 val）、推理全流程，先独立深度审查，再串联起来全局分析。每部分先审公共层、再审任务层。

审查主要内容为代码、算法、设计、架构、工程等等：
是否正确、合理；是否有优化空间；是否有训练加速/GPU优化空间；是否有更好的高质量内容（算法/模块/设计/架构/损失等等）可以借鉴、适配或新增。现在是2026年7月，不局限医学图像领域，可能自然图像的分类/分割/检测/生成等、NLP、LLM、VLM等有更好、更先进的想法。

进展：  



3 


4 模型流可视化需要：有层次化、结构化、清晰、可溯源、美感。可以清晰：看到计算流的走向、理解模型架构，溯源。以下是一些例子：  

- 聚焦模式到stem, stage这个层级为止：
点击模块A，进入聚焦模式，模块群B和A有联系，模块群C和A没有联系，所以模块群C谈出，模块群B突显。我希望聚焦到stem, stage这个级别的模块即可，再进一步的子模块例如stem，stage里面的子模块则不进入聚焦。  

- 连线走线需要清晰、不重叠、不交叉、美观、可以溯源：
需要清晰的看到不同模块的关系，并能溯源输入输出等等

- 位置清晰，层次清晰，严格遵守各自的位置关系：
例如输入后可能同时结果多个stem，那么这几个stem就是位置并列的；例如如果有deep supervision，且在dec level 0后有ds head 2, dec level 1后有ds head 1等等，那么ds head 2位置就应该和dec level 1并列，因为它们就是dec level 0的下一个计算。

- 其它的我暂时没有想到，请你根据我的喜好推荐。总之：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判、讨厌"自动布局默认输出"式的无设计感结果。

进展：

──────────────────────────────────────────────────────────────
批 4 进展记录（架构收敛与文档，2026-07-26；实现后核验）

一、已修复
- G9：组级划分提升为 taskcore 公共入口，五任务按“group_id_regex 非空 → grouped，其次 stratified，再次 random”选择；空 regex 保持原分支。
- G17：新增显式 `finalize_from_data`，数据探测结果统一经该入口进入配置，loader 不再直接回写 `dc.label_values` / `dc.num_classes`。
- G6：npz metadata 增加 `achieved_spacing`、`pre_resample_shape`、`origin`、`direction`，保留历史 spacing 键；旧 npz 读取继续走原 fallback。
- G16：mixed sampler 增加 coarse-bound epoch、gold coverage、gold under-covered/cycled 的诊断；默认 batch 生成和随机流未改。
- H11：seg 采样 RNG 与验证 coverage 位置改为显式传递 sample index，避免依赖样本实例临时字段。
- H12：缓存估算日志明确按 image/label/region-weight 三个独立 cache 统计；默认 eviction 行为未改。
- D1：README、taskcore README、各任务 WORKFLOW/DESIGN 补充公共划分、批 1–3 开关、几何校验、pretrain 几何校验和 checkpoint 开关说明。

二、判定不成立或已失效
- G15：现有 global replay 的默认 batch 顺序与随机流契约无法在本轮以低风险方式证明逐位等价，因此未做 replay 优化，仅保留 G16 诊断。
- I14：未发现可以在不触及 checkpoint key、migration alias 或 pickle 兼容的前提下安全删除的清理项，因此未做激进清理。

三、仍推迟
- J8：真正的预取重叠需要后台取数线程或迭代器重构，可能改变 worker/RNG 时序。
- J16：现有终止与 cleanup 契约已能工作，本轮没有足够低风险的独立改进。
- I5：SparK 门控下沉会改变 block 内中间特征与归一化统计，需独立实验开关和数值评估。
- ADM 非标准 attention：legacy Q/K 分别 softmax，不满足标准 SDPA 等价契约。
- G13/G14/H9：RNG 派生与 Halton 去相关涉及跨五任务随机流统一，不能在本轮局部改动后声称默认逐位兼容。
- G7/H5 的 GPU 化部分：CPU 抗混叠已完成；in-plane resize GPU 化仍需 CPU/GPU 插值等价性和设备路径设计，避免改变默认数值。

──────────────────────────────────────────────────────────────
TODO 1 修复核验（2026-07-25，对批 1–4 修复逐条比对新旧代码 + 本机全量测试）

## 测试结果
- 4 个新增测试文件 32 项全部通过。
- 全仓测试：1528 passed / 6 skipped / 11 failed；11 个失败全部在 tests/test_model_flow.py，且在修复前的旧代码上以完全相同方式失败（本机缺 torchlens 可选依赖），与本次修复无关。

## 一、确认修复正确且质量良好
- F7/N1：TaskSectionSpec 新增 preprocess_raw 钩子，seg 注册传入 hoist_legacy_seg_sections（seg_config.py:112），registry 与直连两条加载路径对 legacy 顶层 loss/predict 行为一致；有等价性测试。实现正确（hoist 原地改 raw，钩子返回 None 也兜住了）。
- F5+I1：新建 taskcore/config/geometry.py 作为 stride 推导单一事实源，factory.py 改为引用同一实现（旧私有名保留别名，兼容既有测试）；validate_patch_geometry 在配置期硬报错并给出"nearest legal"建议值；adm/edm2/unetpp 运行期静默 interpolate 兜底改为 RuntimeError。方向正确、实现与模型层逐位一致。
- F6：override 按字段声明类型（get_type_hints）转型，支持 Optional[List[float]] 等联合类型；未知路径/转型失败统一抛 ConfigError。质量好。
- F10：__getattr__ 告警文案已改为 taskcore.config.core.%s。
- H2：seg/cls/det 三处 z 采样统一走 safe_z_center_range/safe_z_grid_center。边界数学核对正确：half-open 区间 [D_patch//2, D_vol-(D_patch-half)+1) 与 extract_z_patch_padded 的 lo=z-half 约定精确匹配；薄卷回退中心+单次告警；fg 切片 clip 后 patch 仍覆盖该切片。
- H8：grid_dropout 从"全尺寸 hole_mask 乘法"改为 clone+索引置零；RNG 消耗序列逐位不变（d0/h0/w0 仍按全 batch 采样），有逐位等价测试。
- J10：ssl 梯度同步按 (device,dtype) 桶化 flatten→单次 all_reduce→unflatten，正确。
- J5/J14/J18/N2：state_to_cpu 去掉 GPU 侧 clone（CUDA 直接 .to("cpu")）；_save_best 非 EMA 路径异步保存前才做 CPU 快照、EMA 路径不再二次 clone；EMA backup 走 pinned+non_blocking+stream sync（同步时机正确，empty_like 的 pin_memory 参数在 torch 2.x 合法，已实测）；memory.py 预算计入 ema_backup（未分配时按模型 CUDA 参数估计）。
- G6：npz meta 增 achieved_spacing/pre_resample_shape/origin/direction，公式核对正确（src×前/后形状比），旧键保留。
- G9：group_aware_train_val_split 公共入口，cls/det/gen loader 均接入，空 regex 分支逐位保留（有测试）。
- G16/G17：mixed sampler 覆盖率诊断只加日志；finalize_from_data 统一数据探测回写入口（gen/taskcore loader 均改）。
- EDM2 attention：SDPA 与 legacy einsum 数学等价（scale=1/sqrt(head_dim) 一致），CPU/旧后端回退保留，token 上限守卫合理；ADM 非标准 attention 正确地未动。
- ZeRO：adam/sgd/adamw 及 lr_mult 路径全部支持 ZeroRedundancyOptimizer（原来漏分支）。
- gentask attn_gate_target 按 decoder_type 选择 + 经典 UNet 传非 skips 时 factory 硬报错（I10）。
- 各"默认 legacy 开关"（resize_antialias、elastic_field_mode、split_rounding_mode、init_strategy、pretrain_upkern_normalize）默认值确实逐位保持旧行为，且都有 legacy 等价测试。
- 批 4 中"判定不成立/推迟"的项（G15、I14、J8、J16、I5、ADM attention、G13/G14/H9）理由核对成立，未强行改动是对的。

## 二、发现的问题（按严重度）
1. **P0（必修）pretrain 几何校验对本框架自产 checkpoint 必崩**：base_trainer.py:721-737 `ckpt_cfg = ckpt.get("config")` 后直接 `ckpt_cfg.get("data", {})`。但本框架 _save_best/_save_latest 存的是 `"config": self.cfg`（Config dataclass，无 .get）。已在本机复现：AttributeError: 'Config' object has no attribute 'get'。即任何 train.pretrain_ckpt 指向 best/latest checkpoint 的迁移训练都会在几何校验处崩溃（而不是执行校验）。需按 isinstance(dict) / dataclass 两种形态分别取值；当前 4 个新测试文件均未覆盖该路径。
2. **P1 upkern normalize 语义可疑**：mednext.py normalize_spatial 把插值后核除以自身空间和（|sum| 归一到 1），而非"保持源核空间和不变"（UpKern 常规做法是 rescale 使插值核 sum == 源核 sum）。源核 sum≠1 时开启该开关会整体改变响应幅度。默认关不影响现状，但该选项按现语义开启大概率有害。附带测试缺陷：test_upkern_normalization_is_opt_in 中 `torch.equal(legacy, legacy)` 是恒真式，应比较 legacy 与旧函数输出。
3. **P1 init_strategy 非 legacy 时是"地毯式覆盖"**：_apply_init_strategy 会覆盖所有 Conv/Linear，包括 ADM/EDM2 精心设计的零初始化输出投影、EDM2 magnitude-preserving 权重、attention gate 等；对 adm/edm2 开启 kaiming/trunc_normal 会破坏其初始化契约。建议限制 arch=='unet' 或排除零初始化层，至少在文档标注。
4. **P2 validate_patch_geometry 的适用边界**：(a) 对 arch=adm/edm2 也读取 unet.anisotropic_pooling/stem_mode（这两个 arch 并不消费），配置了 anisotropic_pooling 的 adm 会用错误的 divisor 校验；(b) unet3p decoder 本身支持任意尺寸全尺度重采样，现在也被一刀切拒绝；(c) 这是破坏性契约变更——原本能跑（靠 interpolate 兜底）的 YAML 现在配置期报错，configs/test_e2e.yaml 已被迫从 [12,256,256] 改为 [16,256,256]，用户存量配置需同样迁移。建议在 changelog/文档明示。
5. **P2 H2 是无开关的默认行为变更**：z 采样域收窄+val 覆盖 bin 变化会改变训练分布与验证指标口径（同一模型两版代码 val 指标不可比）。与本轮其它改动"默认 legacy"的一致性原则相悖；作为 bug 修复可接受，但建议在文档标注跨版本指标不可比。
6. **P3 split_manifest 并发写**：build_dataloaders 在所有 DDP rank 上执行，manifest 同名文件被多 rank 同时 write_text（非原子）。内容相同通常无害，建议 rank0-only 或原子写。
7. **P3 override int 语义微变**：`data.foo=3.7`（声明 int）旧代码 int("3.7") 抛错，现 yaml 解析成 3.7 后 int() 静默截断为 3。
8. **P4 风格**：unetpp.py:109-113 skips 分支多缩进 8 空格（合法但不一致）。

## 三、结论
批 1–4 修复整体质量高：几何单一事实源、逐位兼容开关、等价性测试的做法都很规范；32 项新测试有效覆盖了大多数修复的真实失效模式。必须处理的只有问题 1（pretrain 几何校验 AttributeError）；问题 2/3 建议在开启前修正语义/加防护；4/5 需要文档与迁移说明。
──────────────────────────────────────────────────────────────

进展：

批 5 实施进展（2026-07-26；已完成 review 并写回用户本地）

一、已实现
- P0：pretrain 几何元数据同时支持 dict 与 Config/dataclass；缺字段时跳过比较。新增自产 checkpoint 回归测试。
- P3：整数 override 接受 `3.0`、拒绝 `3.7`；split manifest 改为 rank0-only、同目录临时文件加 `os.replace` 原子写；修正 UNet++ 缩进。
- UpKern：可选 normalize 改为保持源核空间和；批 3 恒真断言改为独立插值 reference。
- init_strategy：ADM/EDM2 的非 legacy 策略在配置与 factory 入口显式拒绝，保留经典 UNet 的 kaiming/trunc_normal。
- geometry：ADM/EDM2 使用各自实际 stem/downsample 几何；UNet3P 放宽整除要求但保留 encoder 尺寸不足硬错误；经典 UNet 仍严格整除。
- H2：新增 `data.z_sampling_mode`，默认 `safe`（保持当前行为）；`legacy` 复刻批 1 之前 seg/cls/det 的全域 z 中心与旧验证 z-grid。文档已说明跨版本验证指标不可直接比较。

二、测试
- 新增 `tests/test_todo1_batch5_fixes.py`，覆盖上述修复及三任务 z sampling 开关；gentask 兼容入口保留。
- 定向回归：56 passed（批 1/3、配置 IO、cls/det smoke）；批 5 新测试 13 passed；
- 全量回归：1541 passed, 11 failed, 6 skipped, 145 warnings；11 个失败仍为既有 `tests/test_model_flow.py` / torchlens 缺失相关失败，未新增失败。
- py_compile：通过。

三、状态
- 镜像 TODO 已先同步用户最新版本，本节追加在用户审查记录之后。
- `/home/ubuntu/seg/batch5.diff` 已由用户 review 通过；批 5 实现与 TODO 追加已写回用户本地，未 commit/push，未修改 `.git`。

──────────────────────────────────────────────────────────────
批 5 修复复核（2026-07-25，逐文件比对 + 本机全量测试）

## 测试
- 批 1/3/5 定向测试 30 项全部通过；全仓 1541 passed / 6 skipped / 11 failed——11 个失败仍为 test_model_flow.py / 本机缺 torchlens 的既有失败，与批 5 无关，数字与用户记录一致。

## 逐条确认（对应上轮 8 个问题）
1. P0 pretrain 几何校验：已修复且正确。_checkpoint_config_value/_checkpoint_geometry 同时支持 Mapping 与 dataclass（getattr 走 Config.model 只读 property 也正常）；ckpt 非 Mapping 时跳过；缺字段返回 None 不参与比较。新增自产 dataclass checkpoint 回归测试直接调 _load_pretrain_weights，覆盖了真实失效路径。✔
2. upkern normalize：语义已改为"保持源核空间和"（source_sum/denom 缩放，denom<1e-8 回退 1），float32 中间精度处理正确；批 3 恒真断言改为独立 F.interpolate reference 并断言 sums==source_sums。✔
3. init_strategy：配置层（_validate_model）与 factory 双重拒绝 adm/edm2 + 非 legacy，经典 UNet 保留；有参数化测试并验证 legacy 下 adm/edm2 零初始化不被覆盖。✔
4. 几何校验边界：adm 用 stem×2^(n-1)（adm 确实消费 stem_mode，已核对 build_context_stem）、edm2 用 2^(n-1)（不吃 stem/anisotropic）、unet3p 放宽为"逐级特征尺寸不归零"检查、经典 UNet 维持严格整除；有 monkeypatch 测试证明 adm 路径不再误用 unet divisor 推导。✔
5. H2 行为开关：新增 data.z_sampling_mode（safe|legacy），贯通 seg/cls/det/gen 四任务（patch_dataset_base、specs、各 loader 均传参），legacy 路径复刻批 1 前的全域采样与旧 z_grid_center；枚举在 config 层与 dataset 层双重校验；测试覆盖三任务。✔（默认 safe = 保持批 1 后行为，属明示决策）
6. split manifest：rank0-only + 同目录 pid 临时文件 + os.replace 原子替换，异常时清理 tmp；有 rank!=0 不落盘 + 无残留 tmp 的测试。✔
7. int override：接受 3.0、拒绝 3.7 与 bool（yaml "true" 解析为 bool 被显式拒绝），抛 ConfigError。✔
8. UNet++ 缩进已修正。✔

## 遗留小项（不阻塞，记录备查）
- P4：det 的 legacy z 采样 fg 分支丢了原实现的 np.clip(z, 0, D_vol-1)——原批 1 前代码对 box 中心有全域 clip，现 legacy 直接返回 round((b0+b3)/2)。仅当 box 顶到卷 z 上界且坐标为排他上界时 z 可能等于 D_vol（越界 1），下游 edge-pad 抽取不会崩，但与"复刻旧行为"有一处极小偏差。
- P4：z_sampling_mode 默认 safe 意味着从批 1 前版本直接升级的用户默认仍会经历采样分布/验证口径变化；文档已声明，无需改代码。

## 结论
上轮 8 个问题全部得到正确、高质量的处理；P0 修复经真实路径回归测试验证；其余修复的开关语义、原子性与测试覆盖均到位。仅剩上述两条 P4 级备查项，可不处理。
──────────────────────────────────────────────────────────────



批 6 进展记录（2026-07-25；det legacy clip、三任务逐位回归与文档口径）

一、已修复
- H2/P4：det 的 `z_sampling_mode=legacy` 前景框中心恢复批 1 之前的 `np.clip(z, 0, D_vol - 1)` 语义；仅影响显式 legacy，safe 路径和默认随机流不变。
- H2：补充 seg/cls/det 三任务 ordinary、foreground、validation 三条 legacy 分支的固定 seed reference 对照，并检查调用后的 RNG 消耗顺序一致；同时覆盖 safe 分支与当前实现的结果对照。
- B：文档明确 `model.init_strategy` 对 ADM/EDM2 的 non-legacy 拒绝仅属于 taskcore 通用分割 `Config` / `factory.build_model` 路径，不泛化到仓库所有 ADM/EDM2 构建入口。

二、测试
- 批 6 定向测试：24 passed。
- 全量测试：1543 passed / 6 skipped / 11 failed；相对批 5 的 1541 passed / 6 skipped / 11 failed，新增通过项来自批 6 测试，无新增失败。
- 11 个失败仍为 `tests/test_model_flow.py` 在本机缺少 `torchlens` 的既有失败。

三、兼容性说明
- det legacy clip 修复不改变默认 `data.z_sampling_mode=safe` 的数值、采样分布或随机流。
- geometry validator 与 init_strategy 实现保持不变。
