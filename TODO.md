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




2 分割项目代码审查（需结合对应 readme/design/workflow 一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：

分割项目 = 公共框架层 `taskcore` + 任务层 `segtask_v1`，审查按此两级展开。代码大致分 5 部分，数据读取、模型构建、数据增强/处理、训练全流程（含 val）、推理全流程，先独立深度审查，再串联起来全局分析。每部分先审公共层、再审任务层。

审查主要内容为代码、算法、设计、架构、工程等等：
是否正确、合理；是否有优化空间；是否有训练加速/GPU优化空间；是否有更好的高质量内容（算法/模块/设计/架构/损失等等）可以借鉴、适配或新增。现在是2026年7月，不局限医学图像领域，可能自然图像的分类/分割/检测/生成等、NLP、LLM、VLM等有更好、更先进的想法。

进展：  

代码量与分布（已统计，排除 __pycache__）：taskcore + segtask_v1 约 30k 行 Python / 105 个文件。热点文件：

领域	主要文件（行数）
配置真相源	taskcore/config/core.py (2523)、seg_task.py (225)、seg_bundle.py (107)、task_io.py (211)、registry.py (120)、model_migration.py (343)、geometry.py (71)、seg_config.py (119)
数据读取	taskcore/data/dataset.py (1403)、loader.py (1311)、make_data.py (767)、specs.py (280)、mixed_sampler.py (201)、sampling.py (158)、patch_dataset_base.py / patch_ops.py / patch_extract.py
增强	taskcore/data/augment.py (632)
模型	models/blocks.py (1541)、adm_unet.py (996)、edm2_unet.py (938)、unet.py (620)、factory.py (600)、mednext.py (553)、resnet.py (505)、convnext.py (173)、unet3p.py / unetpp.py / stem.py (337)、topology.py (169)、arch_compat.py
训练	engine/base_trainer.py (1125)、optim.py (331)、checkpoint.py (356)、amp.py / prefetch.py / bn_stats.py / dist_utils.py / launch.py / memory.py / views.py；segtask_v1/trainer/trainer.py (758)、validation.py (618)、pipelines/*（slab25d 367 / lift25d 207 / patch3d 121 / vanilla3d 70 / base 150 / factory 81）、views.py (232)、breakdown.py
损失/指标	losses.py (1081)、topo_aux.py (146)、metrics.py (303)
推理	engine/base_predictor.py (64)、segtask_v1/predictor/predictor.py (596)、sliding.py (496)、forwards.py (365)、inputs.py (276)、io.py (257)、blending.py (153)
周边	monitor/*、launcher/*、visualization/*（TODO 4 已单列，本次仅在与主流程耦合处顺带看）
已读文档并抽出的核心契约（后续审查将逐条对代码验真）：

几何单一真相源：taskcore.models.topology.build_topology；配置层 SegBundle（core + seg: 段，旧顶层 loss/predict 经 hoist 兼容）。
四种 patch_mode（whole/cubic/z_axis/2_5d）+ 「dataset 只出单分辨率最大 FOV cube，多视图/多分辨率推迟到 trainer」原则。
2.5D 折叠时机契约：dataset 恒发未折叠 3D → GPU 3D 增强 → 裁余量/视图拆分 → 送模型前折叠（det 例外）。
训练-推理镜像契约：patch_size/patch_mode/multi_res_scales/keep_native_view_depth/keep_native_multi_res/z_boundary_mode/归一化/spacing_normalization 必须一致。
legacy 兼容开关族：elastic_field_mode、split_rounding_mode、resize_antialias、init_strategy、z_sampling_mode 等默认保旧行为（潜在"默认值即次优"的审查点）。
checkpoint 槽位约定：_save_best 存 EMA 为 primary、model_online_state_dict/ema_state_dict 并存，统一经 extract_model_state_dict 读取。
现有回归测试 67+ 个 位于 tests/（pipelines、topology、keep_native_*、checkpoint_resume、augment、losses 等），可作为"审查结论是否与既有行为冲突"的对照物。
范围边界与难点：

TODO 1（公共框架整体审查）尚未开工且与本任务在 taskcore 上重叠——本轮按 TODO 2 口径处理：只审分割链路实际经过的 taskcore 代码，cls/det/gen/ssl 专属分支仅在"公共抽象是否被其他任务扭曲"这一角度点到为止，不展开。
难点：core.py/blocks.py/base_trainer.py/losses.py 均为千行级，需分块精读；跨层契约（config→topology→dataset→pipeline→model→loss→predictor）必须端到端对齐才能判断正确性，因此必须留一轮做全局串联。
2026 前沿对标需要联网检索（分割 SOTA、训练加速、数据管线、损失/评测），会在对应步骤内执行并标注来源与适配代价。
建议的执行计划（每步一轮，可独立执行）
统一产出格式（每步）：① 事实梳理（该模块真实做了什么，带 文件:行号 引用）→ ② 正确性问题（Bug / 契约违背 / 边界未处理，按 严重/中等/轻微 分级）→ ③ 合理性与设计评价 → ④ 优化空间（含 GPU/吞吐/显存，标注预估收益与风险）→ ⑤ 2026 可借鉴项（方案、来源、适配代价、优先级）→ ⑥ 与既有测试/契约的冲突检查。 统一验收标准：结论均可溯源到具体代码行；不含"凭记忆臆断"的 API/版本断言（涉及库用法时查最新官方文档）；不提出破坏既有契约而未说明代价的建议；不产生任何代码/文档改动。

步	内容	主要对象	依赖
S1	配置与几何真相源（前置基座，非 5 部分之一但被全部依赖）	taskcore/config/*、models/topology.py、seg_config.py	无
S1 审查报告：配置与几何真相源
本轮范围：taskcore/config/*（core.py / geometry.py / section_validators.py / seg_task.py / seg_bundle.py / registry.py / task_io.py / model_migration.py）、topology.py、seg_config.py，以及它们在训练/推理入口的实际调用点。 本轮遵守：不改任何代码/文档；结论均给 文件:行号；关键判断用本地环境（D:\miniconda\envs\torch27_env\python.exe）实测复现；不臆断库用法（外部对标已附来源）。

① 事实梳理：这一层真实做了什么
加载链路（分割）：@d:/codes/work-projects/SegTask/segtask_v1/seg_config.py:47-58 → hoist_legacy_seg_sections（旧顶层 loss/predict 迁入 seg:，新旧同设即报错，@d:/codes/work-projects/SegTask/taskcore/config/seg_task.py:197-218）→ dataclass_from_dict（旧扁平 model 键路由 + 别名/派生键/未知键硬拒绝，@d:/codes/work-projects/SegTask/taskcore/config/core.py:2374-2448）→ SegBundle.sync() → SegBundle.validate()。

四个"单一真相源"：

几何/通道派生：build_topology(cfg) 一次算齐 in_channels / spatial_dims / out_classes / per_view_depths / num_stem_fusion_views / aux_head_out_channels（@d:/codes/work-projects/SegTask/taskcore/models/topology.py:75-166），sync() 只写两个私有 backing 字段，对外只读 property（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1358-1363、626-640）。
下采样 stride：显式 downsample_strides > anisotropic_pooling 自动调度 > 历史各向同性 ×2（@d:/codes/work-projects/SegTask/taskcore/config/geometry.py:47-81）。
选模标准：save_best_criterion → (metric, mode) 单表派生（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1079-1089、991-1002）。
旧扁平 model 接口：FLAT_TO_NESTED 一张表同时驱动 YAML 路由、转发 property、老 ckpt __setstate__（@d:/codes/work-projects/SegTask/taskcore/config/model_migration.py:150-154、349-373）。
校验编排：Config.validate() 六个段校验器 + 可 skip（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1432-1453）；SegBundle.validate() 把 loss/predict 从 core 摘出交给 seg 段（@d:/codes/work-projects/SegTask/taskcore/config/seg_bundle.py:52-66）。数据探测回写走 finalize_from_data 并重跑 sync()（@d:/codes/work-projects/SegTask/taskcore/data/loader.py:418-436），调用点在建模型之前（@d:/codes/work-projects/SegTask/segtask_v1/train.py:59-63），顺序正确。

② 正确性问题
严重
[S1-A] train.reparam_deploy 字段归属错位，导致独立推理 CLI 必然崩溃 字段定义在 TrainConfig（@d:/codes/work-projects/SegTask/taskcore/config/core.py:987-989），消费点却读 cfg.model.reparam_deploy（@d:/codes/work-projects/SegTask/segtask_v1/predictor/io.py:160）。ModelConfig 无该字段、也不在 FLAT_TO_NESTED 转发表内。实测：



Config().model 上 hasattr('reparam_deploy') = False
即 run_inference 在 load_state_dict 之后、.to(device) 之前 无条件抛 AttributeError，python -m segtask_v1.predict 全路径不可用（与 backbone 无关）。为何未被测试发现：@d:/codes/work-projects/SegTask/tests/test_dilated_reparam.py:255 用 cfg.model.reparam_deploy = True 先行 setattr，恰好把缺失字段补上了，掩盖了默认路径。 根因属配置层：字段归属（train vs model）与消费方不一致，且没有"配置字段必须有消费者/消费者路径必须存在"的契约测试。

中等
[S1-B] save_best_preset 在 sync() 中反复覆盖用户 override，静默丢失 _apply_save_best_preset 无条件覆盖三个字段（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1370-1394），而入口在 override 之后又调一次 cfg.sync()（@d:/codes/work-projects/SegTask/segtask_v1/train.py:129-132）。实测：



yaml 里 save_best_preset=vessel → criterion=balanced
--override train.save_best_criterion=iou 后再 sync → criterion 仍是 balanced
用户以为在换选模标准，实际整轮训练选模口径没变，只有一条 INFO 日志。

[S1-C] sync() 非幂等：resenc_preset 展开后无法再跟随 encoder_channels 变化 _apply_resenc_preset 只在 *_blocks_per_stage 为空时填（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1396-1430）。实测：



preset=M → [1,3,4,6,6]；再 override encoder_channels 为 6 级 → 二次 sync 不重填
→ REJECTED: encoder_blocks_per_stage must have 6 entries; got 5
结果是"配了 preset 就不能用 --override 改深度"，报错信息还指向一个用户从没写过的字段。B 与 C 同源：sync() 把派生（可重复执行）与一次性预设展开（不可重复执行）混在一个方法里。

[S1-D] lift 分支的 D 整除检查是第二个几何真相源，且与 geometry.py 冲突 @d:/codes/work-projects/SegTask/taskcore/config/core.py:2085-2093 硬编码 D % 2**(n_levels-1)，既不看 downsample_strides，也不看 stem stride。实测（lift + downsample_strides=[[1,2,2],[1,2,2],[2,2,2],[2,2,2]]，z 实际只降 4 倍，D=8 合法）：



REJECTED: lift_2_5d_to_3d=True with 5 encoder stages requires patch_size[0] (D=8) divisible by 16
误拒合法配置。同时它忽略 stem_mode=patch2/patch4（漏检部分幸被 validate_patch_geometry 兜住，@d:/codes/work-projects/SegTask/taskcore/config/section_validators.py:101）。正解是复用 effective_patch_divisors。

[S1-E] _validate_augment 里混进了 data / model 段的校验 data.split_rounding_mode、data.z_sampling_mode、model.init_strategy 三项枚举校验写在 augment 校验器内（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1873-1885）。validate(skip={"augment"}) 会连带跳过它们——与 validate() 文档承诺的"按 section 拆分"契约不符（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1432-1438）。

[S1-F] skip 未知段名静默忽略 @d:/codes/work-projects/SegTask/taskcore/config/core.py:1448-1450 只做 name not in skip，拼错（如 "2.5d"、"Model"）不报错，静默变成"全量校验"或"漏跳"。文档把静默忽略当特性（给组合式任务传 loss/predict），但代价是所有拼写错误都不可见；更稳的做法是白名单 + 显式允许集。

[S1-G] unet3p 提前 return，跳过 divisor 校验 @d:/codes/work-projects/SegTask/taskcore/config/section_validators.py:80-94 对 decoder_type=='unet3p' 只检查"每级尺寸 ≥ stride"就 return，不做整除检查。UNet3+ 的全尺度 skip 需要各级尺寸严格对齐，非整除时问题会推迟到运行期 shape mismatch。此条标注为待验真：需 S4/S5 核对 unet3p.py 内部是否用插值自适应对齐；若自适应，则本条降为"注释未说明豁免理由"。

轻微
问题	位置	说明
ModelTopology.num_fg_classes 在 lift 路径算错	topology.py:62-69 + 130	lift 下 slab_depth 仍= D，num_fg//1//D → 0 → max(...,1)=1。当前无生产消费者（各处用 cfg.num_fg_classes），属埋雷
patch_size 无正性校验	core.py:1925-1927	只校验长度 3；[0,128,128] 能通过（0 % divisor == 0），错误延到 dataset
num_classes 与 label_values 不一致不报错	core.py:1323-1324	只在 num_classes==0 时推断，两者同设且矛盾时静默采信 num_classes
amp_dtype / compile_mode 配置期无枚举校验	core.py:834-839	amp_dtype 拖到 _setup_amp 才 ValueError（base_trainer.py:201-204）；compile_mode 全无校验。二者都发生在数据扫描/npz 构建之后，浪费启动时间
z_boundary_mode 白名单仍含已废弃的 "stretch"	core.py:1937-1940	依赖 sync() 升级；手工构造后只 validate() 的路径能带着 stretch 通过
Optional 字段无法 override 回 None	task_io.py:80-81	--override predict.hw_overlap=null → float(None) → ConfigError
推理侧无训推镜像契约校验	predict.py:62-66、io.py:131-143	只有 state_dict 形状兜底；patch_mode / normalize / spacing_normalization / multi_res_scales 不一致时静默产出错误结果——这是 S10/S11 的核心风险，根因在配置层没有落盘指纹
seg 未走注册表加载，三份近重复加载逻辑	seg_config.py:47-74（load_task_config 导入后未用，第 20 行）、core.py:2456-2474、task_io.py:169-195	registry 里注册的 preprocess_raw 在 seg 实际路径上形同虚设（seg 自己又调了一遍 hoist），未来改 registry 不会同步到 seg
死代码 / 导出不全	registry.py:32（_COMPOSITE_SKIP_CORE=()）、section_validators.py:140-143（__all__ 漏 validate_patch_geometry，但它被 core 按名 import）	
③ 合理性与设计评价
做得好的（应保留为契约）：

派生量只读化 + 旧键硬拒绝：in_channels/spatial_dims/save_best_metric/mode 全部 property 化，旧写法在加载期给出迁移指引式报错（core.py:2312-2333、2420-2441），彻底消灭"设了却被静默重写"，是本层最有价值的设计。
build_topology 单入口：12 种 mode 组合有等价性回归（tests/test_model_topology.py:55-80），新增 patch_mode 只改一处决策树。
model_migration 双向映射：一张表驱动 YAML 路由 / property / pickle 迁移，且新旧同设 fail-fast（model_migration.py:254-257），比"静默优先级"稳妥得多。
结构性问题：

sync() 职责过载——它同时承担 ①纯派生（幂等）②语义自动升级（幂等）③一次性预设展开（非幂等）。S1-B/S1-C 都是这个混合的直接后果。建议拆为 derive()（每次都跑）/ normalize()（每次都跑）/ expand_presets()（加载期只跑一次，或改为"preset 只填未显式设置的字段"）。
几何真相源仍有 3 份 stem-stride 副本：geometry.stem_stride_of:12-14、core._est_stage_tokens:1634-1636、core._validate_2_5d:2149-2153。加一个 stem 模式就要改三处——这正是 R5 引入 build_topology 想消灭的那类问题，只是这次漏在了 stride 侧。
校验按段划分与实际字段归属脱节（S1-E），且跨段耦合校验散落在 _validate_model / _validate_2_5d / seg_task._validate_cross 三处，"哪些约束在什么时候跑"需要读三个文件才能拼出来。
SegBundle 是 duck-typed 门面（seg_bundle.py:68-86），但下游签名普遍写 cfg: Config（如 io.py:96），静态检查与 isinstance 全部失真；__getattr__ 还要手工防递归。收益（cfg.loss 同址访问）与代价（类型系统失效）值得在 S11 重估。
2.5D / 3D 的 keep_native_* 双胞胎门控写了三遍：sync（core.py:1338-1356）、build_topology（topology.py:97-103）、_validate_data（core.py:1981-2013）。三处条件必须同步修改，是新增 mode 时最易漏的地方。
cross-task 泄漏：build_topology 里为 gen 任务的 cond_dirs 做 getattr 兜底（topology.py:90-93）。按 TODO 2 口径只做记录：公共抽象已被非分割任务轻度扭曲，属 TODO 1 的处理对象。

S2	数据读取（一）公共层	data/specs.py、loader.py、dataset.py（含 npz IO / LRU / 前景索引 / 采样中心）	S1
审查报告：数据读取（一）公共层
本轮遵守：只审不改（未产生任何代码/文档改动）；结论均给 文件:行号；关键判断用 D:\miniconda\envs\torch27_env\python.exe 实测复现；不臆断库用法（未联网核实的对标项已显式标注）。

本轮范围：specs.py、dataset.py、以及其叶子依赖 sampling.py / patch_ops.py；loader.py 的发现/划分/装配主链路。make_data.py、mixed_sampler.py、patch_dataset_base.py 与任务层 segtask_v1/data/* 按计划留给 S3；augment 留给 S6。

① 事实梳理
装配链路（seg 训练）：build_dataloaders (@d:/codes/work-projects/SegTask/taskcore/data/loader.py:1058-1311) → _resolve_npz_paths（扫描 + 可选 rank0 内联 auto-build + exclude 过滤，:744-835）→ detect_label_values（优先读 meta.label_counts 免解码，:357-415）→ finalize_from_data（唯一写配置入口，:418-436）→ 划分（group > stratified > random，:1150-1176）→ DatasetCommonCfg.from_cfg + _split_paths_from（:1188-1190）→ build_data_spec（唯一 patch_mode 分支点，@d:/codes/work-projects/SegTask/taskcore/data/specs.py:260-270）→ spec.make_split → DataLoader/Sampler 装配（:1206-1295）。

四个真实契约：

npz-only：SegDatasetNpzBase.__init__ 强制 npz_paths 与 image_paths 等长（@d:/codes/work-projects/SegTask/taskcore/data/dataset.py:696-699）；_split_paths_from 让 image/label/npz 三者同源（loader.py:838-846），image/label 路径退化为"计数 + 缓存键"别名。
单分辨率最大 FOV：三个 dataset 恒发单 cube，多视图推迟到 trainer（dataset.py:948-951、1189-1191）。
split-dependent 参数只在 spec 内切换：_aug_oversample / _samples_per_volume（val 减半）/ _fg_ratio（val=0）（specs.py:130-143）。
验证确定性：训练用逐 worker 流式 RNG，验证用 (VAL_SAMPLING_SEED, sample_idx) 派生（sampling.py:63-72、dataset.py:738-746），val_grid_coverage 时改走 z-bin / Halton 铺点（dataset.py:748-756、sampling.py:111-132）。
I/O 层：未压缩 npz 走 _open_npy_member_mmap 零拷贝 memmap 快路径（dataset.py:257-296），preprocess_image 单次分配 + in-place（:478-511），load_nifti* 有限重试且 OOM 折为 MemoryError（:67-89）。三份独立 LRU（img/lbl/rw）在 pickle 时清空（:612-656）。

② 正确性问题
严重
[S2-A] 异构 npz（部分含 rw）会让 default_collate 直接 KeyError weight_map 是逐卷条件性写入的：有 rw 文件 → 写；无 rw 且 region_weights 为空 → 不写（dataset.py:994-1004、1233-1241、1382-1394）。同一 batch 内混入"有 rw / 无 rw"两类样本时 collate 崩溃。实测：



KeyError 'weight_map'   # torch.utils.data._utils.collate.default_collate
单一 npz_dir 内部因 match_region_weight_paths 强制 1:1（loader.py:341-349）通常同质，但双源混采必然踩中：npz_dir（金标，带 rw）与 npz_dir_secondary（粗标，离线单独构建、通常无 rw）经 ConcatDataset 混进同一 batch（loader.py:1215-1245）。根因是"样本 schema 随卷可变"，不是 collate 的问题；正解是 dataset 层把 weight_map 恒定输出（无 rw 时填全 1），或在 build_dataloaders 里对两源 rw 存在性做一致性 fail-fast。

[S2-B] 划分不可复现：manifest 只写不读，数据集增删即静默换 val 集 train_val_split 对位置索引做 RandomState(seed).permutation(n)（loader.py:501-512），而 primary_paths 是排序后的目录扫描结果。新增/删除一个 npz 会同时改变 n 与后续文件的位置。实测（20 个样本，插入 1 个 p04b.npz）：



val before ['p00', 'p01', 'p15', 'p17']
val after  ['p00', 'p01', 'p14', 'p16']
leaked (was val, now train): ['p15', 'p17']
而 split_manifest_path 只有写入路径（loader.py:1177-1185、468-498），全仓无任何读取点（grep 仅命中定义、写入与两个写入侧单测）。后果：① 补数据后 resume/续训，原 val 样本进入训练集，历史 best 指标口径失效；② 跨运行指标不可比。正解是 manifest 可回读并按文件名哈希（而非位置）划分。

中等
[S2-C] safe_center_range（cubic）与 safe_z_center_range（z）上界差 1，cubic 永远采不到每轴最后一个体素 实测（vol=100、patch=64）：



cubic ((32, 68), ...)   # max center 67 → 覆盖 [35, 99)，索引 99 永不入 patch
z     (32, 69)          # max center 68 → 覆盖 [36, 100)，边界可达
_axis_center_range 的 hi = size - (patch - half) 是半开上界（patch_ops.py:81-89），而 safe_z_center_range 多 +1（sampling.py:107-108）。数学上 z 侧才是正确的（c = D-(p-half) 恰好贴边、无 padding）。影响：cubic/whole-cube 模式下每轴最外层 1 个体素永远不参与训练与 val-grid 覆盖；同时这是"同一语义两份实现"的漂移点，cls/det 也共用 patch_ops。注意此条与既有测试冲突，见 ⑥。

[S2-D] z 路径多分辨率的安全中心域按 eD 而非 eD_max 计算 _sample_z 用 D_patch = self.extract_size[0]（dataset.py:1025），但实际抽取深度是 eD_max = round(eD * max_scale)（:978）；cubic 侧则显式乘了 _max_scale（:1260-1264）。scales=[1.0, 2.0] 时最粗视图最多可有 ~50% 深度来自 edge 复制，且分布随卷厚度变化。行内注释（:1023-1024）表明这是"保持采样域跨视图不变"的有意选择，但它与 cubic 的口径不一致、也未在 DESIGN 中作为契约声明；至少应统一并写明代价。

[S2-E] log_volume_cache_estimate 为读一个 shape 解压整卷，且泄漏文件句柄 _f["image"].shape（loader.py:994-995）会把整卷 int16 从 zip 解出——本模块自己的文档明确警告过这一点（dataset.py:299-303 的 _read_npz_image_shape：「仅为读形状时应避免」）。且 _f = open_npz(...) 未用 with，句柄直到 GC 才释放。纯诊断代码为此付出一次整卷解码的启动开销，正解是复用 _read_npz_image_shape。同一函数内日志自相矛盾：total_gb 已把 index_bytes 乘上了 workers（:1016-1017），随后却打印「shared per dataset process; not multiplied」（:1035-1038）——按 fork/spawn 实际语义，乘是对的、日志是错的。

[S2-F] region_weights 长度无校验，zip 静默截断/错位 compute_region_weight_map 直接 zip(label_values, region_weights)（dataset.py:524-525）。配置语义是"按 label_values 全长（含 bg）"（core.py:668-671 及各 YAML 注释），但 taskcore/config 内没有任何长度校验（grep 仅命中字段定义）。长度写短 → 尾部类静默失权；写成"仅前景"长度 → 权重整体错位一格（bg 拿到 fg1 的权重）。这与 S1-「num_classes 与 label_values 矛盾不报错」同属一类缺口。

[S2-G] cache_mode / cache_dtype / normalize 的字符串比较即"枚举"，拼错静默降级 cache_enabled = (str(dc.cache_mode) == "memory")、cache_int16 = (str(dc.cache_dtype) == "int16")（specs.py:79-81）：写成 "Memory" → 缓存静默关闭，只表现为变慢。normalize 未知值则拖到 worker 内每样本抛 ValueError（dataset.py:509-510）——发生在 npz 扫描/构建之后，浪费启动时间。同 S1-「amp_dtype/compile_mode 配置期无枚举校验」。

[S2-H] world_size > val_batch 数 时部分 rank 拿到 0 个 val batch ValBatchShardSampler._blocks = range(rank, n_batches, world_size)（loader.py:57-58）无下界保护。指标 all-reduce 本身能容忍（各 rank 计数不等长是设计内的），但验证阶段任何逐 batch 的集合通信（SyncBN / DDP forward）都会因步数不等而挂死。属需在 S9/S10 端到端核对的风险点，根因在本层缺守卫。 （正面结论：块划分与 DataLoader 的 batch 边界是对齐的——块按升序产出、只有全局最后一块可能不足 batch_size 且必落在其所有者的末尾，因此不会出现跨块拼批。这点写得正确。）

轻微
问题	位置	说明
Config 是未定义名	specs.py:119、260	无 import、无 TYPE_CHECKING；靠 from __future__ import annotations 在运行期不炸，但类型检查全失效
resize_antialias 绕道传参	specs.py:58/85-89/145-150	先作为 dataclass 字段、to_kwargs() 又 pop 掉、再用 inspect.signature 探测补回；三个 dataset 全都有该形参，探测恒真，是纯复杂度
_filter_by_exclude 返回值半废弃	loader.py:831-834	kept 变量未使用，随后又用 keep_idx 重算一遍
取整逻辑 4 份且互不一致	loader.py:439-465、547、738	grouped_train_val_split 与 stratified_split_by_key 完全无视 split_rounding_mode
_current_vol_idx 未在 __init__ 声明	dataset.py:954、1194 vs 704-732	只在 __getitem__ 里首次赋值；直接调用 _getitem_max_fov 的测试/子类路径会 AttributeError
_sample_idx 实例态	dataset.py:732/952/1192	所有调用点都显式传 sample_idx，该字段是可被并发写坏的冗余状态
三份 LRU 各自持 cache_max_volumes	dataset.py:720-722	配 N 实际最多常驻 3N 个卷；且容量按"卷数"而非字节，卷尺寸差异大时预算失真
whole 模式 label 未做 antialias 讨论	dataset.py:1370-1371	label order=0 正确，但 image 的 resize_antialias 默认 false（legacy），整卷 4× 下采样会明显混叠
逐类均衡是"卷内"而非"全局"	dataset.py:1036-1038、1283-1285	rng.integers(len(per_cls)) 在该卷出现的类里均匀选；某类只在少数卷出现时，全局仍然欠采
③ 合理性与设计评价
做得好的（建议固化为契约）：

_open_npy_member_mmap 零拷贝快路径（dataset.py:257-296）：正确识别 ZIP_STORED + 解析 npy 头 + object dtype 拒绝 + 异常兜底回退，语义与 zipfile 路径逐位一致；页缓存跨 worker 共享，是本层收益最高的工程设计。
验证采样确定性化（sampling.py:63-72）：把 save_best / early-stop / plateau 从采样噪声里摘出来，是很多同类框架都没做对的事。
DatasetSpec 策略化（specs.py）：patch_mode 分支收敛到一处，且 split 差异封在 spec 内，loader.py 不再感知 train/val 差异。
ValBatchShardSampler：与"逐 rank 全量迭代 + 跳批"严格同构，但 CPU 开销不随卡数翻倍；无 padding 无重复，指标可与单进程严格相等。
结构性问题：

build_dataloaders 未复用同文件的 assemble_train_val_loaders：loader.py:889-970 已把"workers 平摊 + loader kwargs + 零批次拦截 + sampler 选择"收敛成公共函数，但 seg 主链路在 :1206-1295 又把同样的逻辑内联写了一遍（loader_kwargs 手搓、DistributedSampler/ValBatchShardSampler 手搓）。两份实现已经出现分化（内联版没有 collate_fn / train_drop_last 参数），是下一次改 DDP 行为时的漏改点。
前景索引在内存里存了两份：_vol_fg_coords 保存全量 coords，_vol_fg_coords_by_cls 用布尔掩码切片再存一份（dataset.py:1181-1183；实测 coords[cls==v] 不共享内存）。fg 上限 50000/类/卷时，1000 卷 × 3 类 ≈ 1.8 GB → 实际 3.6 GB，且每个 worker 一份。按类排序后存视图（或存 offset 边界）即可归零这份冗余。
启动期同一个 npz 被打开 2–4 次：detect_label_values（meta）→ 可能的 load_npz_label_counts（loader.py:1163，meta 第二次）→ train split _build_index（dataset.py:928-946）→ val split _build_index。每次都是一次 zip 目录解析。合并为"一次 meta 扫描 + 缓存"是纯收益。
Windows spawn 下索引会被 pickle 进每个 worker：VolumeCache 专门写了 __getstate__ 丢弃缓存（dataset.py:646-656），但 GB 级的 _vol_fg_coords* 没有同等待遇；persistent_workers=False 时每 epoch 重传一次。
"安全中心域"两份实现（S2-C）与 "z 多分辨率域"两套口径（S2-D）：与 S1 的「几何真相源仍有 3 份 stem-stride 副本」是同型问题——patch_ops 已被立为单点，z 轴却另起炉灶。
样本 schema 不稳定（S2-A 根因）：weight_map 的有无由数据决定而非配置决定，违反"dataset 输出结构对 batch 内所有样本一致"这一隐含契约。
npz 之外的路径已死但代码仍在：load_nifti_cropped / load_region_weight_volume / compute_bbox_from_volume / resample_to_spacing（dataset.py:163-230、529-535、591-606）在训练链路上已被 npz-only 契约旁路，实际消费者在 make_data / predictor。放在 dataset.py 里会让读者误以为训练期还有 NIfTI 路径，建议 S11 一并评估是否下沉到 volume_io.py。
④ 优化空间（含 GPU/吞吐/显存）
按「收益 / 风险」排序：

#	优化	预估收益	风险
1	fg 索引按类排序后存视图，取消 _by_cls 拷贝	索引 RAM 减半（大数据集 GB 级/worker），spawn pickle 量同步减半	低；采样语义不变
2	启动期一次 meta 扫描共享给 detect_label_values / 分层划分 / 两个 split 的 _build_index	启动 zip 打开次数 4→1；千卷量级可省数十秒到分钟	低
3	log_volume_cache_estimate 改用 _read_npz_image_shape + with	省一次整卷解码 + 消除句柄泄漏	极低
4	三份 LRU 合并为单缓存、按字节计费（cache_max_bytes）	内存预算可预测（当前配 N 实占 3N 卷）；避免 OOM 靠告警兜底	中；cache_max_volumes 需保留兼容
5	cache_int16 路径每次取用都重跑 preprocess_image（dataset.py:768-776），可改为在 GPU 上做窗宽/归一化	CPU worker 从 O(卷) 浮点运算降为 memcpy；与 augment.py 的 GPU 3D 增强天然同址	中；需与 S6 的 GPU 增强入口统一，且要保证与 CPU 路径数值一致
6	面内 resize_3d 用 scipy.zoom（单线程 CPU）；大 FOV cubic 下这是 worker 主要热点	改为 GPU 上 F.interpolate（随 batch 一起做）可显著降 worker 占用	中高；会改变数值（zoom order=1 vs grid_sample），需回归比对
7	pin_memory 已开，但未设 pin_memory_device；DDP 下可省一次隐式设备查询	小	极低
8	_build_index 串行读 npz meta，可用线程池（纯 I/O + zip 目录解析，GIL 友好）	启动期线性→并行	低
⑤ 2026 可借鉴项
说明：本轮未联网核实以下方案的最新版本与 API 细节（连接不稳定）。这些条目是方向性建议，若进入落地阶段，我会先查各自最新官方文档再给具体接口，不凭记忆写实现。

方案	借鉴点	适配代价	优先级
nnU-Net v2 的前景过采样策略	当前是逐样本伯努利（fg_ratio 概率，dataset.py:1033），batch 内前景样本数方差大、小 batch 时可能整批无前景。nnU-Net 的做法是按 batch 内位置强制后 1/3 样本为前景样本，方差为 0	需在 sampler 层（而非 dataset 层）表达"batch 内配额"，与 MixedBatchSampler 有交互	高（直接影响小 batch 3D 训练稳定性）
nnU-Net v2 的 blosc2 分块存储（近两年替代 npz 的方向）	支持分块随机读：只解压 patch 所在的 chunk，而非整卷。当前 memmap 快路径已能零拷贝，但 --compress 产物会退回整卷解包，且缓存必须按整卷计费	需改 make_data 落盘格式 + 新增 reader；_open_npy_member_mmap 可保留为旧格式路径	中（大卷 + 内存受限场景收益大）；需核实当前稳定版是否已默认
按内容哈希划分 train/val（sha1(basename) % 100 < val_ratio*100）	直接消除 S2-B：增删样本不影响既有成员归属，manifest 退化为审计产物	低；但会一次性改变所有历史划分，需作为 split_mode 新枚举、默认保旧	高
per-case 归一化（nnU-Net 的 MRI 方案）	当前只有全局 zscore（global_mean/std）与 minmax（dataset.py:496-508），CT 够用，MRI/多中心数据缺 per-case（含 nonzero-mask）归一化	低（加一个 normalize 枚举值）；但必须同步进训推镜像契约（S1-「推理侧无镜像校验」）	中
MONAI SmartCacheDataset 式的缓存置换	当前 LRU 按访问置换，在"每卷采 S 个 patch、卷序被 shuffle 打散"的访问模式下命中率差；SmartCache 用"每 epoch 替换固定比例"匹配这种模式	中；或更简单——用卷块化 sampler（同卷的 S 个样本尽量同 epoch 邻近）把命中率拉起来，改动只在 sampler	中（cache_mode=memory 且卷数 >> 缓存容量时收益明显）
DataLoader(in_order=False)（PyTorch 新增的乱序返回）	卷尺寸差异大时 worker 之间耗时不均，当前严格按序返回会被最慢 worker 拖住	低；但训练确定性会改变，验证侧必须保持 in_order=True	低-中；需核实本仓 torch 版本是否支持
WebDataset / 顺序分片读	网盘/机械盘上随机小文件读是瓶颈（本仓已为此加了 NIfTI 读重试，dataset.py:47-89，说明确有不稳定存储场景）	高（改数据组织 + 采样语义）	低（仅当确认 I/O 为瓶颈）
⑥ 与既有测试/契约的冲突检查
S2-C 与既有测试直接冲突：@d:/codes/work-projects/SegTask/tests/test_patch_ops.py:42-46 断言 safe_center_range((5,20,20),(32,8,8))[1] == (4,16)（即 hi = 20-(8-4)），把当前的半开上界固化成了契约；:49-63 还断言 cls/det/seg-cubic 共用同一实现。修正上界需同步改这两处断言，并接受 cubic 采样分布发生可观测变化（cls/det 的 patch 分布也会变）——因此这不是一个"顺手修"的 bug，建议在 S12 里作为独立批次评估，或先只统一文档口径。
S2-D 与 tests/test_keep_native_multi_res.py:454-460 相邻：该测试特意把体积放大到"最大 cube 能完整落入"以避开退化分支，说明现有测试刻意回避了多分辨率 + 边界的组合，本条无现成对照物，需 S8 端到端验真后再定性。
S2-B 修正会改变所有历史划分，与 README.md:39-45 / DESIGN.md:25-31 声明的"legacy 默认保旧行为"原则冲突 → 必须以新枚举 + 默认 legacy 落地。
S2-A、S2-E、S2-F、S2-G 无既有测试覆盖，修正不破坏任何现有断言（S2-A 需新增"异构 rw 源"回归用例）。
tests/test_todo1_batch5_fixes.py:54-65 只覆盖 manifest 的写入（rank 过滤 + 原子替换），不覆盖回读——与 S2-B 的判断一致，非冲突。
tests/test_todo1_batch1_fixes.py:111-117 固化了 safe_z_center_range 的薄卷退化行为，与本轮结论不冲突。

S3	数据读取（二）预打包与混采 + 任务层	make_data.py、mixed_sampler.py、sampling.py、patch_*、segtask_v1/data/*	S2
审查报告：数据读取（二）预打包与混采 + 任务层
本轮遵守：只审不改（未产生任何代码/文档改动）；结论均给 文件:行号；关键判断用 D:\miniconda\envs\torch27_env\python.exe 实测复现（脚本写在系统临时目录，未落工作区）；外部对标已附来源。

本轮范围：make_data.py、mixed_sampler.py、sampling.py、patch_ops.py、patch_extract.py、patch_dataset_base.py、segtask_v1/data/*，以及它们在 loader.py 的装配调用点（_resolve_npz_paths 内联 auto-build、混采装配段）。loader.py 的发现/划分主链路属 S2，不重复。

① 事实梳理
预打包链路：CLI python -m taskcore.data.make_data --config --out [--workers/--fg-subsample/--compress/--overwrite/--limit]（@d:/codes/work-projects/SegTask/taskcore/data/make_data.py:712-763）；@d:/codes/work-projects/SegTask/segtask_v1/data/make_data.py:13 是 sys.modules 替换式 shim。 prepare_dataset（:521-667）= _build_sample_table（discover→exclude→match bbox/rw，:452-483）→ _resolve_label_values（:486-494）→ _resolve_target_spacing（nnU-Net 式头信息中位数指纹，:497-518）→ 逐样本 prepare_one → _failures.txt + _manifest.json。 prepare_one（:234-449）九步：skip 幂等判定 → 物理几何 fail-fast（:198-220）→ bbox → 流式裁剪读 image/label → rw（+1 偏移、int16/fp32 自适应，:299-321）→ cond（gen）→ 可选 spacing 重采样（:336-361）→ 逐类 fg 索引 + label_counts（:363-372）→ meta → tmp(带 pid)+os.replace 原子写（:410-439）。 skip 契约 _npz_meta_allows_skip（:67-123）：必需键 label_counts/image_shape/fg_per_class + 比对 spacing_normalized/target_spacing/label_values/fg_subsample/has_rw/src_bbox。内联 auto-build 由 rank0 独占 + barrier（@d:/codes/work-projects/SegTask/taskcore/data/loader.py:776-790）。

混采链路：loader.py:1113-1253 —— secondary 不划分、整批入训练；SourceTaggedDataset 打 source 标；ConcatDataset([primary|secondary])；MixedBatchSampler 作 batch_sampler；val 恒取金标。sampler 契约：coarse 每 epoch 顺序消费一遍、gold 循环过采样，epoch 长度 n_secondary // coarse_per_batch // world_size（@d:/codes/work-projects/SegTask/taskcore/data/mixed_sampler.py:157-160）；DDP 下各 rank 同 seed+epoch 生成同一全局序列再按 b % world_size 取切片（:225-239）。set_epoch 经 @d:/codes/work-projects/SegTask/taskcore/engine/base_trainer.py:953-962 鸭子识别、@d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:420-421 每 epoch 调用 —— 该链路正确闭合。

叶子层三件套：sampling.py（worker RNG / val 确定性 RNG / z-grid / Halton / 中心钳制）、patch_ops.py（cube 抽取 + 安全中心域）、patch_extract.py（VALID_PATCH_MODES 单点 + extract_patch_by_mode + resolve_patch_center）。patch_dataset_base.py 是 cls/det 专用模板基类，seg 走 SegDatasetNpzBase。

任务层事实：segtask_v1/data/ 只有空 __init__.py 与 shim —— 分割项目的数据读取任务层是空的，已 100% 下沉 taskcore。计划里 S3 的"任务层"部分在 seg 上无实体。

② 正确性问题
严重
[S3-A] skip 判据单向（只查"缺"不查"多"），关掉 region_weight_dir/bbox_dir 后重跑必然静默复用陈旧包 expect_rw/expect_bbox 只在"要求有而 on-disk 无"时拒绝（:119-122），反向不查。实测：



on-disk has_rw=True, src_bbox='F:/roi/p0.nii.gz'；请求 expect_rw=False, expect_bbox=False
→ (True, '')          # 允许 skip
两个后果：① 训练实际吃的是按旧 bbox 裁剪、且带 rw 的体积，与当前配置不符，image_shape 也不是未裁剪尺寸；② 若中途改配置（打了一半 → 去掉 region_weight_dir → 继续打），同一个 npz_dir 内部就会混合"有 rw / 无 rw"两类包 —— 这直接触发 S2-A 的 default_collate KeyError: 'weight_map'。即 S2-A 不是双源混采专属，单源同样能踩，根因在这里的 skip 契约，而不在 collate。

[S3-B] _failures.txt 与 data.exclude_list 不兼容（模块 docstring 与日志的承诺为假） 写出格式为 pid\terror（:633-635），而 _load_exclude_pids 整行 strip() 后只剥 .nii/.nii.gz（loader.py:84-93）。实测：



parsed exclude pids -> {'p0007\tValueError: image shape (10,) != label shape (11,)'}
用户照 :636-639 的日志提示把 _failures.txt 配成 exclude_list，实际零排除，训练会在同一批坏样本上再次失败。文件头部注释（:1）、汇总日志、CLI 帮助三处都在宣称兼容。

中等
[S3-C] 混采无跨源 pid 重叠检查 → 病例级数据泄漏 secondary 整批入训练（loader.py:1215-1226），primary 才走 train/val 划分（:1149-1176）。若同一病例既有金标又有粗标——这正是"金少粗多"的典型来源（同一批数据先粗标、抽一部分精标）——该病例的粗标版进训练、金标版可能进 val，val 指标虚高。group_id_regex 的防泄漏（:1150-1158）只作用于 primary 内部。全链路无任何跨源检查。

[S3-D] fg_subsample 不在必需键，存量包改 --fg-subsample 静默无效 _REQUIRED_SKIP_META_KEYS（:45-49）含 fg_per_class 但不含 fg_subsample，比对处又是 if fg_subsample is not None and "fg_subsample" in meta（:111）。实测 on-disk 缺该键、请求 1000 → (True, '')。影响面是 make_data<1.8 的存量包 —— 而这正是最需要重建的一批。

[S3-E] 内联 auto-build 借用 dc.num_workers 当打包并发度，是 host OOM 高风险路径 loader.py:785 传 workers=max(dc.num_workers, 1)。CLI 侧特意警告"每 worker 峰值 ≈ 一个裁剪样本的 RAM，按主机内存调"（:723-726），但 DataLoader 的 num_workers 常配 8–16，且 bbox_dir 未配时每个 make_data worker 的峰值是整卷 NIfTI而非裁剪后。缺独立旋钮。

[S3-F] 自动 target_spacing 依赖当前样本集合，--limit 冒烟会污染目录口径 中位数在 samples 上取（:497-518），--limit/exclude 变化即改变结果。skip 会因 target_spacing mismatch 触发重建（这点是对的），但中间态目录里两种 spacing 并存；没有任何"目录内 spacing 必须一致"的启动期校验，build_dataloaders 也不查。

[S3-G] extract_patch_by_mode 不透传 anti_alias，cls/det/predictor 口径恒无抗混叠 patch_extract.py:48/54 调 resize_3d 未传该形参（默认 False，dataset.py:552）。seg 有 data.resize_antialias 配置走 dataset 自己的 resize；共用这条口径的 predictor 则永远关闭。z_axis/whole 面内大比例下采样时训推频域不一致 —— 属训推镜像契约的一个未登记缺口（S10 需复核 predictor 是否确实经此路径）。

[S3-H] pid 无唯一性校验，重名样本互相覆盖 _stem（:129-136）+ out_path = out_p / f"{pid}.npz"（:553）。image_suffix 支持候选列表（loader._normalize_suffixes），a.nii 与 a.nii.gz 并存即产出同一 pid；tmp 名带 os.getpid() 不冲突，但 os.replace 相互覆盖，静默丢样本且 counters 显示两条 written。

轻微
问题	位置	说明
DDP 下"粗标整轮恰好覆盖一遍"的 docstring 契约不成立	mixed_sampler.py:102 vs :157-160	实测 ws=3 / 50 coarse / cpb=2：global 25 batch → 每 rank 8、总 24，covered 48/50。尾部丢弃是设计内的（drop_last 同构），但文档没说
timings/sizes 把 skipped 的 0 计入	:589/605 → :616-623	"mean compute"/"mean per sample size" 被 skip 稀释成无意义值
counters[status] 用返回值当 key	:677 vs :617-623	新增 status 不会出现在硬编码的汇总日志里
meta 的 origin/bbox 与落盘卷坐标系不一致	:382、:396	origin 取原图头值、未按 bbox 起点平移；bbox 记的是原生 spacing 坐标系而卷可能已重采样。目前全仓无消费者（predictor 从源图直接复制几何，predictor.py:567），属埋雷型审计元数据
spacing_zyx 用 requested 而非 achieved	:400-402、dataset.py:461-462	小卷 round 后偏差 ~0.5%，会传到推理 z-interleave 因子选择
halton_center 轴数 >3 时静默截断	sampling.py:122-132	实测 4 个 range → 返回 (5, 3, 2)（3 元组）。bases[:len(ranges)] + zip 双重截断
_THIN_Z_WARNED key 恒为 "thin_volume"	sampling.py:100-104	只告警一次且不带 pid，无法定位是哪些卷薄
采样复现性隐式依赖 persistent_workers	sampling.py:48-55	True 时同一 worker 跨 epoch 不重建 RNG（继续推进），False 时每 epoch 由新 info.seed 重建 → 同 seed 不同序列，未文档化
函数体内 import	patch_dataset_base.py:116-120	同一模块的 WorkerNumpyRng 已在顶部 import，无循环依赖，纯不一致
cube 抽取两份近重复实现	patch_ops.py:25-78	前者可由后者一行导出；tests/test_patch_ops.py:15-22 专门断言二者一致，说明漂移风险已被感知
resolve_per_batch_counts 是第二份校验真相源	mixed_sampler.py:76 自述与 Config._validate_data 一致	同 S1「几何真相源 3 份」型
SourceTaggedDataset 就地改 base 返回的 dict	mixed_sampler.py:55	依赖"底层每次新建 dict"这一未声明契约；将来若做样本级缓存即静默污染
注释乱码"夹匯"	dataset.py:201、sampling.py:147	应为"钳制/夹取"
③ 合理性与设计评价
做得好的（建议固化为契约）

原子写（:410-439）：tmp 名带 os.getpid() 避免多 rank/多 worker 交错，os.replace 而非 unlink→rename（注释明确写了 Windows MoveFileEx 语义）。配合 rank0-only auto-build + barrier，DDP 下不会写坏包。这是本层最扎实的工程点。
物理几何 fail-fast（:198-220）：用只读头的零成本，堵住"shape 相等 ≠ 共坐标系"这一业界高频静默错误，且级别定在与 shape 校验同级，是对的。
逐类 fg cap（:167-179，seed=42）：稀有小结构不被大器官淹没，且打包可复现。
label_counts 落 meta（:370-372）：S2 里"启动期免全量 label 解码"的能力就来自这里。
MixedBatchSampler 的 DDP 方案：所有 rank 生成同一全局序列后取模切片、RNG 消费严格对齐（:225-239），等长、无 padding、无重复。实测 3 rank 各 8 batch、24 个 batch 两两不重叠。比"各 rank 独立 sampler"稳。
gold 覆盖率三分支日志（:167-197）：把"金标其实在欠采样而非过采样"这一反直觉情形显式化，很多同类实现会漏。
结构性问题

公共层/任务层边界是按文件位置划的，不是按依赖方向。 segtask_v1/data/ 为空是好事（无重复实现），但代价是 taskcore.data 里堆了大量单任务代码：cond_paths → 仅 gen、NpzPatchDatasetBase → 仅 cls/det、MixedBatchSampler → 仅 seg。
两套并行的 patch dataset 基础设施。 NpzPatchDatasetBase（cls/det）与 SegDatasetNpzBase（seg）各自实现 LRU、worker RNG、val 网格、z_sampling_mode 校验、__len__。三
S3 审查报告（续）：③–⑥
①②已在上文给出。以下续完，均为只审不改。

③ 合理性与设计评价（续）
结构性问题（接前 1、2 条）

make_data 的幂等判据是"字段 allowlist 比对"而非"输入指纹"（:80-122）：每加一个打包参数要同步改三处（字段、比对分支、_TOOL_VERSION），且天然只能查"想到的字段"—— S3-A/S3-D 就是漏项。
混采只做了一半：source 标签全仓无消费者（grep SOURCE_SECONDARY/SourceTaggedDataset 仅命中 mixed_sampler.py、loader.py、tests/）。注释自称"为 loss 对粗标降权预留接口"（mixed_sampler.py:37-38），现状是粗标与金标等权进 loss。在"金少粗多 + epoch 长度由粗标决定"下，模型主要在粗标上收敛，而 save_best 用金标 val —— 这是会实际压低指标的缺口，不是纯 TODO。
epoch 语义被粗标绑架（:157）：加粗标数据 → epoch 变长 → 相同 max_epochs 下总步数、LR 调度、early-stop patience、EMA 半衰期全部隐式改变。粗标量应只决定"每 batch 配比"，不应同时决定"epoch 多长"。
cache 估计只看 primary（loader.py:1212-1213）：混采下常驻卷数是两源之和，内存告警低估约一倍。
cross-task 泄漏（记录不展开，属 TODO 1）：prepare_one 的 cond 分支（:247/323-334）只服务 gentask，而 prepare_dataset._kwargs（:567-580）根本不传 cond —— 公共实现里有一条本 CLI 永远走不到的分支。
④ 优化空间
#	优化	预估收益	风险
1	skip 改为输入指纹（影响产物的全部参数 + 源文件 mtime/size 哈希）落 meta	根治 S3-A/S3-D 及未来漏项	中；旧包一律不匹配→首次全量重建，需 --accept-legacy-meta 逃生门
2	合并 _resolve_label_values(:493 全量 NIfTI 解码) 与 prepare_one 的第二次读	未配 label_values 时启动 I/O 减半	低
3	打包期设 sitk 全局线程数=1	进程池 × sitk 内部线程的超订消除（重采样阶段明显）	极低
4	_compute_fg_indices 超 cap 时改 flatnonzero+choice+unravel_index，不先物化 argwhere	大器官峰值内存从 O(全部前景) 降到 O(cap)，数量级	低；需固定 RNG 消费顺序保可复现
5	prepare_dataset 改 chunked submit + 增量落 _failures.txt	万级任务下可中断续跑，future 开销降低	低
6	n_train_vols/cache 估计纳入 secondary	内存告警不再低估一半	极低
7	extract_cubic_patch 由 ..._with_origin 导出	消除两份实现	极低（已有等价性测试）
⑤ 2026 可借鉴项
方案	借鉴点	适配代价	优先级
nnU-Net blosc2 .b2nd（nnunetv2/training/dataloading/nnunet_dataset.py，comp_blosc2_params 按 patch_size 反推 chunk/block 命中 L3/L1，set_nthreads(1) 防超订）	只解压 patch 所在 chunk。重要修正 S2 的建议：nnU-Net 在 Windows 上显式禁用 mmap（os.name == "nt"，issue #2723），而本仓是 Windows 主力环境、当前最大收益点恰是 _open_npy_member_mmap 的零拷贝页缓存共享 → Windows 上会退化为整块解压且无跨 worker 共享	高	中偏低（由 S2 的"中"下调）；仅作 Linux 训练机可选后端，须先实测
产物目录名带参数标识（nnU-Net data_identifier：改预处理必须换标识，explanation_plans_files.md）	比指纹更简单的落地形态，天然杜绝"同目录混口径"（S3-A/S3-F 共同根因）	低	高
多数据集/部分标注训练（MultiTalent arXiv:2303.14444；DoDNet CVPR2021 / TPAMI2023；2024-25 mutual-learning、GAMOS）	补齐结构性问题 4。三路线：(a) 把 source 接进 loss 做常数降权；(b) 按 source 分离 Norm 统计或加 dataset-embedding；(c) MultiTalent 式独立输出通道 + 缺席类不计损失（同时解决金/粗类定义冲突）	(a) 低 / (b)(c) 高	(a) 高
batch 内前景配额（nnU-Net oversample_foreground_percent 按 batch 内位置强制）	S2 已提，S3 补落点：本仓已有 MixedBatchSampler 这层 batch_sampler 抽象，配额语义应并入同一个 QuotaBatchSampler（同时表达金/粗配比与前景配比），而非再造机制	中	高
划分写进 _manifest.json	manifest 已有"Predictor 回读 target_spacing"的现成契约（:641-665），把 split 纳入同一机制即可解 S2-B	低	高
打包期数据集指纹（nnU-Net fingerprint_extractor）	已在算 median spacing（:497-518），可顺带落 intensity 分位数、per-case 前景强度统计，为 per-case 归一化与窗宽自动化供数	低（同一遍扫描内）	中
⑥ 与既有测试/契约的冲突检查
S3-A 修正无冲突：tests/test_r2_review_hotfixes.py:116-130、tests/test_gen_make_data_core.py:40-48、tests/test_review_batch2_fixes.py:265-303 只覆盖"缺键/参数变化→不 skip"与"完全一致→skip"，无"on-disk 多出 rw/bbox"用例。但会让存量"带 rw 而配置已关 rw"的包重建一次，需 CHANGELOG 明示。
S3-D：加入必需键不破坏断言，但会让 make_data<1.8 存量包全部重建；按"legacy 默认保旧"原则宜先降为 WARNING，下版本再强制。
S3-B：两种修法——改写为"注释行 + 纯 pid"，或让 _load_exclude_pids 取首列（tab 前）。后者更优且不破坏现有 exclude_list（无 tab 时行为不变）。两侧均无测试覆盖。
S3-C：纯新增 fail-fast/warning；test_mixed_sampler.py 全在 sampler 层，不涉 loader 装配，无冲突。
S3-G：tests/test_patch_extract.py:38-43 断言 extract_patch_by_mode(z_axis) 与 resize_3d(extract_z_patch_padded(...)) 逐值相同 —— 默认仍 False 则不冲突；一旦跟随 data.resize_antialias，该测试需带同一开关。
S3-H：无测试覆盖；需先确认多后缀路径有无生产用法，建议先 WARNING 再升级为报错。
与 S1/S2 的一致性：S3-A 是 S2-A（KeyError: 'weight_map'）的上游根因之一，两者应合并为一个批次（dataset 恒定输出 weight_map + make_data skip 双向判据）；S3-D/S3-F 与 S1-「amp_dtype/compile_mode 配置期无枚举校验」、S2-G 同属"启动期缺前置校验、代价推迟到长耗时之后"这一族。

S4	模型构建（一）装配层	models/factory.py、topology.py、stem.py、unet.py/unetpp.py/unet3p.py、arch_compat.py、pretrain.py	S1
① 事实梳理：这一层真实做了什么
装配主链路（seg）：segtask_v1/train.py:63 → build_model(cfg)（@d:/codes/work-projects/SegTask/taskcore/models/factory.py:523-600）→ 按 model.arch 三分派（unet / adm / edm2，:533-543）→ _build_unet_encoder_decoder（:303-497）→ Encoder/Decoder|UNetPPDecoder|UNet3PDecoder → UNet3D（@d:/codes/work-projects/SegTask/taskcore/models/unet.py:417-613）→ _apply_init_strategy（:29-51）→ 一行汇总日志（:580-598）。推理侧同一入口（@d:/codes/work-projects/SegTask/segtask_v1/predictor/io.py:114-125）。

五个装配期真相源：

几何/通道：全部读 build_topology(cfg)，装配层不自算（factory.py:322-327、548-558）。四种 patch_mode 的决策树集中在 @d:/codes/work-projects/SegTask/taskcore/models/topology.py:108-148。
逐级 block 数：_resolve_blocks_per_stage（enc，:54-65）/ _resolve_decoder_block_counts（dec，:78-84），后者按 _decoder_call_count 决定"decoder 到底要建几个 stage"（unet=n-1、unetpp=n(n-1)/2、其他=0，:68-75）。
stage 索引单计数器：_StatefulStageBuilder（:93-112）——把 stage_idx 从闭包里收回到一处，drop_path 切片与 multirf/selfattn 掩码都用它索引，杜绝双计数器漂移。
各向异性 stride：compute_downsample_strides（@d:/codes/work-projects/SegTask/taskcore/config/geometry.py:47-68）+ 装配期兼容矩阵 fail-fast（factory.py:395-426：禁 ConvNeXt LN-first 下采、禁非 unet decoder、禁 hierarchical stem、限定 down/up mode）。
stem 拓扑：build_context_stem 三分支（@d:/codes/work-projects/SegTask/taskcore/models/stem.py:278-337）；hierarchical 的 aux stride 契约 s0·2^k、out_ch=stage_channels[k-1]（:243-255），由 Encoder.aux_fuse 逐级 1×1 cat 融合（unet.py:172-183、209-219）。
尺寸契约三档：Decoder/UNet++ 上采样后与 skip 必须严格相等，否则显式 RuntimeError（unet.py:275-279、unetpp.py:103-108）；UNet3+ 分支自适应重采样（adaptive_max_pool / interpolate，unet3p.py:83-94）；UNet3D 主头仅在 stem_stride>1 时插值补回输入分辨率，其余不匹配即报错（unet.py:566-574）。

其他事实：build_backbone（:500-520）是 cls/det/ssl 共用的骨干入口，与 build_model 同源；arch_compat.warn_ignored_model_fields 确有生产消费者（adm_unet.py:758、edm2_unet.py:667 等 4 处）；pretrain.py 不在 seg 链路上（seg 走 BaseTrainer._load_pretrain_weights + _pretrain_transform_state_dict，@d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:802-831），只被 clstask/dettask 消费。

② 正确性问题
严重
[S4-A] 各向异性 stride 调度随 data.patch_size 变化，而在 maxpool/avgpool + trilinear/nearest 下权重形状完全不变 → 训推几何漂移静默通过
compute_downsample_strides 在 anisotropic_pooling=True 时按 当前 cfg.data.patch_size 现算 schedule（geometry.py:62-68），而 predictor 的滑窗尺寸也直接取 cfg.data.patch_size（@d:/codes/work-projects/SegTask/segtask_v1/predictor/predictor.py:188）。问题在于：Downsample(maxpool/avgpool) 是"池化 + 1×1 conv"（@d:/codes/work-projects/SegTask/taskcore/models/blocks.py:1253-1260），Upsample(trilinear/nearest) 是"interpolate + 3×3 conv"（:1477-1479）——stride 只进池化/插值，不进任何权重形状。而这两组恰恰就是各向异性唯一被允许的模式（factory.py:294-295）。实测（同一 ckpt，推理侧把 patch 从 [16,64,64] 改成 [8,128,128]）：



modes=(maxpool,trilinear)
  train strides: [(1,2,2), (1,2,2), (2,2,2)]
  infer strides: [(1,2,2), (1,2,2), (1,2,2)]
  strict load_state_dict: SUCCEEDED (mismatch invisible)
  same-input output maxdiff: 0.3866
modes=(conv,transpose)
  strict load_state_dict FAILED: size mismatch for encoder.downsamples.2.op.weight
即 io.py:131-143 的形状预校验（S1 已指出它是唯一的兜底）在这条路径上结构性失效：网络实际下采样倍率变了、感受野变了、输出数值变了，却没有任何报错。conv/transpose 因 kernel_size==stride（blocks.py:1252、1476）侥幸能被形状查出来——这说明"能不能发现"取决于用户选了哪个 up/down mode，而不是取决于契约。根因在装配层没有落盘架构指纹，与 S1-「推理侧无训推镜像契约校验」同源，但这里给出了它最危险的具体形态。

[S4-B] init_strategy != 'legacy' 无差别覆盖，抹掉所有零初始化 / ICNR 初始化契约
_apply_init_strategy 在模型建完后遍历 modules()，对每一个 Conv/Linear 重新 kaiming_normal_ 或 trunc_normal_（factory.py:34-42）。被它覆盖的既有契约至少有三处：SelfAttentionBlock.proj / ffn_out 的零初始化（残差分支初始恒等，blocks.py:1022-1024、1031-1032）、DySample3d.offset/scope（近零偏移 ≈ 双线性，:1364-1370）、icnr_init_ 的子像素同源复制（:1490）。实测：



init_strategy=legacy       selfattn.proj.weight absmax=0.00000
init_strategy=trunc_normal selfattn.proj.weight absmax=0.06565
init_strategy=kaiming      selfattn.proj.weight absmax=0.67313
init_strategy=legacy       ICNR sub-filter replication preserved=True
init_strategy=trunc_normal ICNR sub-filter replication preserved=False
后果：开 selfattn + 非 legacy 初始化时，注意力残差分支在第 0 步就有 O(0.7) 量级输出（kaiming），训练早期不稳定，且没有任何提示。附带两个次级问题：kaiming_normal_(nonlinearity="relu") 与默认激活 leakyrelu 不匹配（增益偏小）；分割主头 1×1 也被 kaiming 重初始化（nnU-Net 对输出头保留默认小方差初始化）。缓解因素：init_strategy 默认 legacy（S1 列为 legacy 开关族），且 resume/pretrain 会覆盖权重——仅影响 opt-in + from-scratch，故未被任何测试发现（现有测试只用 Conv2d+GroupNorm 验证 legacy no-op，@d:/codes/work-projects/SegTask/tests/test_todo1_batch3_fixes.py:106-115）。

[S4-C] resenc_preset + decoder_type='unet3p' 必然崩溃：同一"decoder 节点数"有三份真相源且其中一份漏了 unet3p
_apply_resenc_preset 只对 unetpp 做了三角形节点特判，其余一律填 [1]*(n_levels-1)（@d:/codes/work-projects/SegTask/taskcore/config/core.py:1424-1430）；validate_encoder_decoder_stage_lengths 认为 unet3p 期望 n_levels-1（@d:/codes/work-projects/SegTask/taskcore/config/section_validators.py:38-46）；而 factory 的 _decoder_call_count 对 unet3p 返回 0（factory.py:75）。实测：



unet    dec_bps=[1,1,1,1] | validate OK | build OK
unetpp  dec_bps=[1]       | validate OK | build OK
unet3p  dec_bps=[1,1,1,1] | validate OK | build ValueError: Per-stage block list length 4 != expected 0
用户从未写过 decoder_blocks_per_stage，只配了 resenc_preset=M + decoder_type=unet3p，就得到一条指向该字段的报错——与 S1-C（preset 展开后无法改深度）是同一病：预设展开、配置校验、工厂三处各自编码同一几何常量。且崩溃点在 build_model，即 npz 扫描 / dataloader 装配之后，属 S1/S2/S3 反复出现的"启动期缺前置校验、代价推迟到长耗时之后"族。

中等
[S4-D] unet3p 分支绕过 divisor 校验：不崩，但同一 patch 在不同 decoder 下几何契约不一致
S1-G 待验真项现已定性：UNet3+ 的分支重采样确为自适应（adaptive_max_pool3d / interpolate，unet3p.py:87-94），因此非整除不会 shape mismatch。实测 3 级 encoder + patch_size=[16,30,30]：



decoder=unet3p : validate PASSED，前向输出 (1,1,16,30,30) == 输入
decoder=unet   : validate REJECTED: axis 1 size 30 needs divisible by 4 ...
所以 S1-G 应从"运行期 shape mismatch 风险"降级为：① 注释未说明豁免理由；② 同一份 patch_size 在 unet 下被拒、在 unet3p 下静默接受，且此时 encoder 走 floor 下采（30→15→7），adaptive_max_pool 用非均匀核回到 15，各级 skip 与 label 网格存在亚体素级不对齐。建议要么统一按 divisor 拒绝，要么在文档里把"unet3p 容忍非整除但存在重采样近似"写成显式契约。

[S4-E] unet3p decoder 静默吞掉整个 backbone 旋钮族，且 ConvNeXt/MedNeXt 的忽略告警重复两次
UNet3+ decoder 内部恒为朴素 3×3 ConvNormAct（unet3p.py:48-52、68、79），完全不消费 backbone / block_type / attention_type / se_reduction / drop_path_rate / decoder_blocks_per_stage。也就是说 backbone=mednext + decoder_type=unet3p 实际得到的是"MedNeXt encoder + 朴素 decoder"，无任何提示——对照 arch_compat 给 adm/edm2 做的忽略清单告警（arch_compat.py:57-84），这里是一个可发现性缺口。附带：factory 无条件构建 dec_builder（:380-389）即使 unet3p 根本不用，导致 _make_convnext_stage_builder 的"块内 norm/act 被忽略"告警打印两遍，实测 warnings emitted: 2。

[S4-F] decoder 的 stochastic depth 独立重启 0→max，且最大丢弃率落在紧邻 seg head 的最高分辨率层
_make_drop_path_rates 对 enc / dec 各跑一次 linspace(0, dpr, n)（factory.py:87-90，两次调用于 :132、:266、:211）。而 Decoder 的 level 构造顺序是深→浅（unet.py:330-347），_StatefulStageBuilder 的 idx 随之递增 → 最大 rate 落在最后一个（最高分辨率）decoder level。实测 drop_path_rate=0.2、blocks_per_level=2、3 级：



encoder: [0.04, 0.08, 0.12, 0.16, 0.2]     # 首块 0.0 被建成 Identity
decoder: [0.0667, 0.1333, 0.2]             # 0.2 = 紧邻 seg_head 的那一级
ConvNeXt / nnU-Net ResEnc 的惯例是全网一条单调 schedule、最深处最大；此处不仅重启，方向还相当于"越靠近输出丢得越狠"。这不是 bug（数值可训），但属"默认值即次优"，且与 encoder 的语义不自洽。

[S4-G] 3D 多分辨率下 model.aux_seg_supervision=True 被静默忽略
aux_seg_active 只在 is_2_5d and n_views>1 时为真（topology.py:104-105）。实测 patch_mode=cubic + multi_res_scales=[1.0,2.0] + aux_seg_supervision=True：in_channels=2, aux_heads=0, model.aux_seg_supervision=False，无 warning。对照 selfattn_enabled / multirf_enabled 在"全 no-op"时都有明确告警（core.py:1740-1744、1813-1818），此处口径不一致；用户会以为多分辨率辅助监督已生效。

[S4-H] unet3p 的 skip_attention 存在自门控分支，且 gate 数量是 O(n²)
每个节点 i 对全部 n 个分支各建一个 AttentionGate3D（unet3p.py:69-75），共 n(n-1) 个 gate；其中 j == i 的分支满足 src is encoder_features[i]、gate_signal is encoder_features[i]（:104、:110、:116-119），即 x * σ(f(x, x)) 的自门控——与 Oktay 2018"用更粗的解码信号门控 skip"的语义不符。且 gate 施加在重采样后、branch conv 之前的全分支宽度上，是本 decoder 显存的主要放大项之一。

[S4-I] UNet++ 的深监督取对角线而非论文的 X[0,j]，且 decoder_blocks_per_stage 的节点顺序不可用
UNetPPDecoder 返回对角线 X[i, n-1-i]（unetpp.py:124），因而 DS 头监督的是"浅列 + 低分辨率"节点；UNet++ 原文的深监督是 X[0,1..n-1]（全部全分辨率），这也是其推理期剪枝能力的前提——当前实现把剪枝语义丢掉了，同时保留了全部 n(n-1)/2 个节点的计算与显存。另外 decoder_blocks_per_stage 长度须为 n(n-1)/2，但元素与节点的对应是构造顺序（i 外 j 内，:59-75），既未文档化也无法从配置侧表达意图；实测 decoder 参数 unetpp 2.94M vs unet 2.13M vs unet3p 8.57M（4 级 [32,64,128,256]）。

轻微
问题	位置	说明
unet3p 的 fused_channels 未经配置暴露	unet3p.py:41 vs factory.py:455-465	恒为 cat_channels × n；4 级默认即 256 通道 @ 全分辨率（实测 decoder 8.57M 参数），DS 头也挂在 256ch 上，显存不可调
build_topology 每次 build_model 跑两遍	factory.py:322 与 :548	顺带 _resolve_blocks_per_stage / _resolve_decoder_block_counts 也各算两遍（:329/554、:336/556）；纯冗余，无正确性影响
_build_unet_encoder_decoder 内三个死变量	factory.py:319、324、327	num_fg / out_classes / aux_head_out_channels 取出后未使用（真正使用在 build_model 内重取）
hierarchical stem 不受 grad_ckpt_stem_downsample 保护	unet.py:191-198	非 hierarchical 走 checkpoint_if，hierarchical 分支直接调用；而 hierarchical 恰是最高分辨率、最多 stem的路径，显存最需要它
grad_ckpt_decoder_branches 对 decoder_type='unet' 无效	factory.py:482-495 vs :465/:480	只透传给 unetpp/unet3p；unet decoder 配了不报也不生效
"n_views 与层数"的阈值有三份	stem.py:208-212（n_levels > n_views-1）、unet.py:529-533（n_dec >= n_views）、core.py（n_views < n_levels，实测报错文案）	实测配置层先拦（aux_seg_supervision + hierarchical requires n_views < n_levels），后两处成为不可达冗余，但三者阈值并不严格等价
PatchEmbedStem 静默把 leakyrelu 换成 GELU	stem.py:109-115、:253-254	与 build_model 日志里打印的 activation 不一致，用户无从知晓 stem 用的是 GELU
aux_head_out_channels 长度报错文案误导	unet.py:508-512	aux 关闭时 n_aux_expected=0，报错却写 "must equal n_views - 1"
_resolve_blocks_per_stage 报错不带字段名	factory.py:61-63	即 S4-C 里那条 Per-stage block list length 4 != expected 0，用户无法判断是 encoder 还是 decoder 侧
兼容别名已无生产消费者	factory.py:299-300（_stem_stride_of / _auto_anisotropic_strides）	只服务存量测试
UNet3D(num_fg_classes=...) 弃用形参仍在	unet.py:440-455	全仓无生产调用点，只剩 deprecation 分支
param_count 不含 DS / aux / topo 头分项	unet.py:615-620	日志里 total - enc - dec - seg_head 的差额无处归因
ModelTopology.num_fg_classes 在 lift 路径算错	topology.py:62-69	S1 已记；S4 复核确认装配层无任何消费者（factory 一律传 out_classes），维持"埋雷"定性
arch_compat 忽略清单未覆盖新增嵌套段	arch_compat.py:28-54	grad_ckpt_decoder_branches、multirf.* / selfattn.* 的非 gate 子字段未列（gate 本身已被 validate 拒绝，故不致误导，但清单口径需随字段增长维护）
③ 合理性与设计评价
做得好的（建议固化为契约）

build_topology 单入口真正被遵守：装配层没有一处自行推导 in_channels/out_classes/spatial_dims（factory.py:322-327），这是 R5 的核心收益，S4 逐行核对通过。
_StatefulStageBuilder 的单计数器（factory.py:93-112）：drop_path 切片、multirf 掩码、selfattn 类型三套逐 stage 参数共用同一 idx，杜绝了"三份计数器"这一典型漂移源；且越界即 RuntimeError，不会静默少建。
各向异性兼容矩阵在构造期 fail-fast（factory.py:395-425）：把"blurpool/pixelunshuffle 只支持各向同性""hierarchical aux 假定 ×2""unetpp/unet3p 只支持各向同性"这些隐含假设全部前移到装配期报错，实测 blurpool+aniso → ValueError。这是本层质量最高的一段。
checkpoint_if 的数值等价保证（blocks.py:81-104）：use_reentrant=False + preserve_rng_state=True + 重算路径冻结 BN running buffer（_freeze_bn_running_stats，:51-78）。"开检查点与不开数值严格一致"这条在同类框架里经常被做错，此处做对了并写了理由。
build_backbone 与 build_model 同源（factory.py:500-520）：cls/det/ssl 与 seg 的 encoder.* / decoder.* 同名同形，是 ssl→seg 权重迁移能 strict=False 直接对上的结构基础（tests/test_ssltask.py:425-430 已固化）。
结构性问题

"谁负责几何校验"分散在四层：配置校验器（validate_patch_geometry）→ 工厂兼容矩阵（factory.py:395-425）→ 模块构造期长度校验（unet.py:137-140、321-324；stem.py:142-146、208-212）→ forward 期 shape 断言（unet.py:275-279、211-218；unetpp.py:103-108）。四层各有价值，但没有一层是权威的，S4-C/S4-D 都是层间口径不一致的产物。建议把"decoder 节点数""每轴总 divisor""aux 层数下限"三个量做成 topology 侧的单一派生函数，四层一律引用。
忽略字段的可发现性只对 adm/edm2 做了机制化：arch_compat 是好设计，但 unet 内部的四处忽略（unet3p decoder 吞 backbone 族、3D 下 aux_seg 失效、convnext/mednext 块内 norm/act、unet decoder 不接 grad_ckpt_decoder_branches）各自 ad-hoc 或干脆无声。建议把 warn_ignored_model_fields 泛化成"装配期忽略清单"通用机制，由各 builder 声明自己消费了哪些字段，工厂做差集告警。
初始化策略是"事后遍历"，与模块自带初始化契约天然冲突（S4-B 根因）：只要模块层有零初始化/结构化初始化，事后遍历就一定会破坏它，且随着 S5 引入更多现代块（LayerScale、zero-init residual）冲突面只会扩大。业界做法是模块级声明（timm 的 init_weights / MMEngine 的 init_cfg，或给参数打 _no_reinit 标记后在遍历里跳过）。
patchN stem 的分辨率恢复靠对 logits 做 trilinear 上采（unet.py:566-569）：patch4 时主头输出是 4× 插值的产物，边界精度受限；nnU-Net / Primus 的做法是让 decoder 一路上采回原分辨率再出 logits。当前设计等于把"能否用大 stride tokenizer"这个选项的上限压死了。
装配层没有架构指纹（S4-A 根因）：模型结构实际由 {arch, encoder_channels, blocks, stem_mode, decoder_type, downsample_strides(可能由 patch_size 现算), spatial_dims, out_classes} 共同决定，但落盘的只有权重 + 一份可被手工编辑的 YAML。既然已经有 resolved_config.yaml（train.py:76）与 checkpoint，把这组量算成指纹存进 ckpt、推理期比对，是本层能给 S10/S11 提供的最直接保障。
三种 decoder 的"接口相同、语义分层不同"未被写成契约：三者都暴露 out_channels（深→浅）供 UNet3D 挂 DS/aux 头，但 unet 是"镜像 encoder 分辨率"、unetpp 是"对角线"、unet3p 是"统一 fused 宽度"。DS 头与 aux 头对 decoder.out_channels[-1-k] 的语义假设（unet.py:534-545）只在 unet 上严格成立。
④ 优化空间（含 GPU / 吞吐 / 显存）
#	优化	预估收益	风险
1	hierarchical stem 纳入 grad_ckpt_stem_downsample（unet.py:191-198）	2.5D 多 FOV 下最高分辨率 stem 的激活可省，是该路径显存峰值所在	低；checkpoint_if 已保证数值一致
2	unet3p 暴露 fused_channels 并允许"只在低分辨率做全尺度融合"	当前 4 级默认 256ch@全分辨率、decoder 参数 8.57M（实测）；可降数倍激活显存	中；改默认会变权重形状，需作为新字段默认保旧
3	unet3p 的 gate 移到 branch conv 之后（cat_channels 宽度）并去掉 j==i 自门控	gate 显存/算力从 O(src_ch) 降到 O(64)，去掉 n 个无意义模块	中；改变权重布局与数值
4	UNet++ 支持"只算对角线所需节点 / 推理期列剪枝"	推理显存与时延显著下降（论文本就以此为卖点）	中；训练需保持全节点，推理路径要单独测
5	build_topology / block counts 在 build_model 内复用一次结果	纯清理，启动期微秒级	极低
6	主头分辨率恢复改为 decoder 末级增加一次可学上采（替代 logits 插值）	patch2/patch4 stem 的边界质量；打开"大 stride tokenizer"的设计空间	中高；改变参数量与 ckpt 兼容性
7	drop_path 改为全网单调一条 schedule（enc+dec 连续）	与 ConvNeXt/ResEnc 惯例对齐，去掉"输出端丢最狠"	低；数值会变，需作为开关默认保旧
8	UNet3D.forward 的 self.training 分支（DS/aux/topo）会让 torch.compile 产生两张图	现状可接受；若 S7 发现 recompile 抖动，可拆成 forward_train / forward_eval	低
⑤ 2026 可借鉴项
方案	借鉴点	适配代价	优先级
nnU-Net Revisited + ResEnc 预设（arXiv:2404.09556；documentation/resenc_presets.md，官方将 ResEnc-L 定为新默认）	结论支持本仓路线（CNN U-Net + 规模化仍是 SOTA）。可借鉴的是预设按 VRAM 分档（M/L/XL）而非只给 block 数模板：当前 resenc_preset 只填 *_blocks_per_stage（core.py:1406-1430），不联动 encoder_channels / patch_size / batch_size，用户拿到的"M"与 nnU-Net 的 M 并不等价	中（需引入显存预算估算）	高
Primus / PrimusV2（arXiv:2503.01835；nnU-Net documentation/primus.md：PrimusV2 与 ResEnc-L / MedNeXt 打平，且证明多数"Transformer 分割网"去掉 Transformer 后性能几乎不掉）	本仓已具备零件：patch stem（stem.py:47-73）、3D 轴向 RoPE + window/grid 自注意力（blocks.py:975-1035）。缺的是组合入口：selfattn 被限定 backbone='resnet'（core.py:1658-1661）、stem 最大 patch4、没有 LayerScale/SwiGLU 的统一开关、也没有"轻解码器"档位	中（多为装配层配置打通，非新算子）	中高（作为消融档位，先小规模验证）
模块级初始化契约（timm / MMEngine init_cfg 惯例）	直接解 S4-B：由模块声明"我已自行初始化"，工厂遍历时跳过，而非事后无差别覆盖	低	高
nnU-Net plans/data_identifier 式指纹（同 S3-⑤ 已提的目录标识思想）	解 S4-A：把架构决定量算成指纹落进 ckpt，推理期比对不一致即 fail-fast；这是唯一能覆盖"stride 变了但形状没变"的手段	低（纯新增）	最高
UNet++ 原始深监督 + 剪枝推理（Zhou 2020）	解 S4-I：全分辨率 X[0,j] 深监督 + 推理期只算前 j 列，速度/显存可按精度需求档位化	中	中
全网单调 stochastic depth（ConvNeXt / ResEnc 实践）	解 S4-E	低	中
（说明：以上均为方向性对标，已核对来源页面；若进入落地阶段，涉及具体 API 时会再查各自最新官方文档，不凭记忆写实现。）

⑥ 与既有测试 / 契约的冲突检查
S4-A（架构指纹）：纯新增校验。test_anisotropic_downsample.py 只在同一 cfg 内构建+前向，不涉及跨 cfg 加载，无冲突；但新增的推理期 fail-fast 会让"训练后手工改 patch_size 再推理"的既有用法报错，需 CHANGELOG 明示并提供 --allow-geometry-drift 逃生门（可复用已有的 pretrain_allow_geometry_mismatch 命名风格，core.py:985）。
S4-B（初始化契约）：tests/test_todo1_batch3_fixes.py:106-115 只断言"legacy 是 no-op""kaiming 改 Conv 权重、norm 权重置 1"，用的是裸 Conv2d+GroupNorm；加入"跳过已声明自初始化的模块"不破坏该断言。tests/test_todo1_batch5_fixes.py:255-288 只覆盖 adm/edm2 的 ConfigError 与 legacy 路径，同样无冲突。需新增回归：selfattn + init_strategy=kaiming 后 proj.weight 仍为零。
S4-C（unet3p 的 decoder 节点数）：两种修法——① 让 _apply_resenc_preset 与 validate_encoder_decoder_stage_lengths 统一引用 factory 的 _decoder_call_count（unet3p→0/1）；② 让 factory 对 unet3p 忽略该字段并告警。tests/test_resenc.py:112 断言 decoder_blocks_per_stage == [1,1,1,1]，但该用例用的是默认 decoder_type='unet'，两种修法都不触发；tests/test_todo1_batch1_fixes.py:93-99 固化的是 unetpp 的 "10 entries" 报错，亦不受影响。修法 ① 更彻底（消灭第三份真相源）。
S4-D（unet3p divisor）：test_stem_and_unet3p.py 全部用整除 patch，无用例固化"非整除被接受"这一行为；若改为统一拒绝，无断言冲突，但会拒掉历史上能跑的配置 → 按"legacy 默认保旧"原则宜先 WARNING。
S4-E（drop_path schedule）：会改变数值，tests/test_resenc.py:186-192 只断言参数量、不断言 drop rate，无断言冲突；仍建议以新开关落地、默认保旧。
S4-F / S4-G / S4-H / S4-I：均无既有测试覆盖。S4-F 只加 warning，零风险；S4-H/S4-I 会改权重布局，属"独立批次评估"级别，不适合顺手改。
与 S1–S3 的一致性：S4-A 是 S1-「推理侧无训推镜像契约校验」的最危险实例化（形状兜底在此结构性失效），二者应合并为一个批次（架构指纹落盘 + 推理期比对）；S4-C 与 S1-B/S1-C 同属 sync() 预设展开与下游期望脱节；S4-C/S4-B 又与 S2-G、S3-D 同属"启动期缺前置校验、代价推迟到长耗时之后"这一族。

S5	模型构建（二）算子与骨干	models/blocks.py(1541)、mednext.py、resnet.py、convnext.py；adm_unet.py/edm2_unet.py 仅看分割入口	S4

模型构建（二）算子与骨干
本轮范围：blocks.py(1541)、resnet.py(505)、convnext.py(173)、mednext.py(553)，以及 adm_unet.py/edm2_unet.py 的分割入口（build_adm_seg_model / build_edm2_seg_model）。装配逻辑属 S4，不重复；unet*.py 只在算子契约验真处引用。

① 事实梳理
算子层的真实分工（自下而上）：

nD 分派基座：_CONV/_BN/_IN/_DROP/_MAXPOOL/_AVGPOOL/_AAVGPOOL/_AMAXPOOL 八张 2/3D 表 + _check_dims（@d:/codes/work-projects/SegTask/taskcore/models/blocks.py:23-48）。所有块都以 spatial_dims 参数化，2.5D（=2）与 3D 共用一套实现，*3D 类名仅为 API 兼容。
工程件：checkpoint_if（:81-104）+ _freeze_bn_running_stats（:51-78）；DropPath（:107-122，fp32 采样后转 dtype）。
块内注意力（通道/空间门控）：make_attention 单入口（:476-507），七种 none|se|eca|cbam|coord|lka|msca。由 ResNetBlock/PreAct/Bottleneck/R2Plus1D/MultiRF/ConvNeXtBlock/MedNeXtBlock 在 pwconv2/conv2 之后、残差相加之前统一调用。
token 级自注意力：SelfAttentionBlock（:959-1053）= PreNorm(GN) → Conv1d-QKV → {softmax|linear|window|grid} → Conv1d-proj(zero-init) → 残差，可选 nD-RoPE（:593-645，有界 LRU 缓存 + torch.compiler.is_compiling() 旁路）与 GEGLU FFN。由 factory 逐 stage 追加（factory.py:175-187），仅 backbone='resnet'。
skip 门控：AttentionGate3D（:1056-1089），三个 decoder（unet.py:264-269、unetpp.py:78-84、unet3p.py:70-75）共用，norm_type 由 factory.py:449-451 的 attn_gate_norm=='auto' → mc.unet.norm_type 解析（默认 instance）。
重采样：Downsample（:1219-1284，5 模式）/ Upsample（:1422-1533，6 模式），per-axis stride 支持与各向异性 fail-fast 均下沉到算子构造期（:1262-1279、:1481-1506）。
骨干块三族：resnet.py 的 basic/preact/bottleneck/r2plus1d + MultiRFBlock（多膨胀并行分支）；convnext.py 的 dwconv7+LN+GELU+LayerScale(+GRN)；mednext.py 的 dwconv-k + 通道级 GroupNorm + 倒瓶颈(+GRN)，并额外提供 DilatedReparamBlock（UniRepLKNet 式训练多分支、推理折叠）与 upkern_remap_state_dict。
ADM/EDM2 分割入口：二者均硬性要求 patch_mode=='2_5d'（adm_unet.py:766-769、edm2_unet.py:675-678）、拒绝 hierarchical stem（:810-814 / :716-719）、忽略 decoder_blocks_per_stage 并告警（:784-791 / :692-699）、几何量一律读 build_topology。二者都以 num_fg_classes=out_classes 传参（adm_unet.py:880、edm2_unet.py:761）——口径一致，形参名误导（传的是含背景的 out_classes）。
② 正确性问题
严重
[S5-A] upsample_mode='carafe' 在生产 patch 尺寸下必然 OOM——中间张量是输入的 216 倍，且无任何守卫

CARAFE3d.forward（:1315-1340）先 unfold×3 + .contiguous() 物化 (B, C·k³, D,H,W)，再对它整体 F.interpolate(scale_factor=2)（:1336），得到 (B, C·27, 2D,2H,2W) = 216× 输入元素数。实测（CPU 前向 + CUDA 峰值，输入 (1,64,8,32,32)）：



input numel=0.52M ; x_up intermediate numel=113.2M (216x input)
carafe 1194 ms vs trilinear 36 ms  (x33.4)      # CPU
carafe peak=964.1 MiB  |  trilinear peak=68.9 MiB   # CUDA, no_grad
推理态即 14× 峰值；训练态该张量还要留给反向。按分割 decoder 最高分辨率级推算（C=32、上采到 32×256×256、bf16）：32×27×2.1M×2B ≈ 3.6 GB / 样本 / 层。也就是说这个配置项存在但不可用，而配置层（core.py 的 upsample_mode 白名单）与装配层（factory.py:395-425 的兼容矩阵）都不区分"生产级/实验级"，用户只会看到一个 OOM。正解：改写为"低分辨率加权 + shuffle"形式（不物化高分辨率 k³ 张量），或至少加显存估算 fail-fast 并在文档标注实验性。

[S5-B] AttentionGate3D 的 psi 归一化让门控对输入幅度完全不敏感，默认路径（auto→instance）把弱信号放大到满量程

psi = Conv(inter→1) → get_norm(norm_type, 1) → Sigmoid（:1075-1079）。norm_type 默认经 factory.py:449-451 取全局 norm_type（默认 instance）→ InstanceNorm3d(1, affine=True)；取 group 时 get_norm 静默回退到 1 组（同样是逐样本逐图归一化）。二者都会把单通道门控图强制为零均值单位方差，门控只取决于图的"形状"而与幅度无关。实测（同一模块，输入幅度差 1000 倍）：



norm=batch     input_scale=1.0    gate min=0.2911 max=0.6802 std=0.0669
norm=batch     input_scale=0.001  gate min=0.4997 max=0.5002 std=0.0001
norm=instance  input_scale=1.0    gate min=0.0279 max=0.9608 std=0.2030
norm=instance  input_scale=0.001  gate min=0.0327 max=0.9742 std=0.2037   ← 幅度降 1000× 而门控不变
norm=group     input_scale=0.001  gate min=0.0346 max=0.9229 std=0.2052
后果：一个整体响应微弱（本应被大体量抑制）的 skip，和一个强响应 skip 得到统计上完全相同的门控分布；深层小特征图上噪声被放大到 [0.03, 0.97] 全量程。Oktay 2018 原文用 BatchNorm（跨 batch+空间统计，保留单样本间的幅度差异），这也是实测里唯一在弱信号下退化为 ≈0.5 恒等直通的分支。影响面为全部三种 decoder 的 skip_attention=True 路径（unet3p 的 skip_attention 还是 O(n²) 个 gate，见 S4-H）。缓解因素：skip_attention 默认 False。

中等
[S5-C] 特征图缩到 1³ 时归一化层直接 ValueError，配置层不拦

MedNeXt 的通道级 GroupNorm(C, C)（mednext.py:31-34）与 get_norm('instance')（默认 backbone 的块内 norm）在单元素空间上都会抛异常，且 F.group_norm 的检查与 train/eval 无关（实测已 .eval()）：



spatial=(4,4,4)  norm_out absmax=3.211179          | instancenorm OK
spatial=(1,2,2)  norm_out absmax=1.731346          | instancenorm OK
spatial=(1,1,1)  ValueError: Expected more than 1 value per channel ...
                 instancenorm -> ValueError: Expected more than 1 spatial element ...
validate_patch_geometry 只检查整除性（section_validators.py:101），patch_size=[32,32,32] + 6 级 encoder（bottleneck 1³）是合法配置，崩溃点却在第一个 batch 的前向。属 S1/S2/S3/S4 反复出现的「启动期缺前置校验」族，建议在几何校验里加"最深级每轴 ≥ 2"。

[S5-D] window/grid 注意力恒物化 attn_mask，即使没有 padding

_WindowQKVAttention.forward:887-891（grid 同 :924-928）无条件构造 attn_mask 并传入 SDPA。实测尺寸整除时 mask 全 True、是纯 no-op，却仍然传入：



spatial=(4,14,14) window=7 -> padded=(7,14,14) mask.all()=False   # 需要 mask
spatial=(7,14,14) window=7 -> padded=(7,14,14) mask.all()=True    # 不需要，仍然传
SDPA 64×4×343×32 bf16: no mask 0.305 ms | w/ mask 0.435 ms   (+43%)
本机 Windows 轮子未编译 flash（实测 SDPBackend.FLASH_ATTENTION 恒 "No available kernel"），所以上面 43% 是 mem-efficient 后端内的开销；在 Linux flash 可用的机器上，任意浮点 attn_mask 会直接把 flash 后端排除，代价更大。正解：padded == orig 时传 attn_mask=None。附带同源浪费：q/k/v 各调用一次 _window_partition_tokens（:884-886、:921-923），mask 与 meta 被算三遍丢弃两遍；offsets 的 product() 列表（:694-697、:711-714）每次前向重建，而 RoPE 分支只用它做非空断言（:892-895）——窗口内 RoPE 本就是相对的，offsets 在数学上不需要。

[S5-E] 大核/条形核注意力全是各向同性硬编码，薄 z 轴上大部分抽头落在 padding 里

LKA3D 默认 k1=5, k2=7, dilation=3（:421-432），MSCA3D 默认 scales=(7,11,21) 逐轴条形核（:450-467），都不看 spacing/patch_size 的各向异性。实测：



LKA dw_dilated: k=(7,7,7) dil=(3,3,3) pad=(9,9,9) -> z taps at [-9,-6,-3,0,3,6,9]
D=8: taps inside volume for a center voxel: 3/7
params: MSCA=4304  ResNetBlock=13888   |  fwd: MSCA 14 ms vs ResNetBlock 7 ms
即在 z_axis/薄 slab 上，LKA 的 z 向膨胀分支 7 个抽头只有 3 个能取到真实体素，其余是 replicate/zero padding；MSCA 的 21 长条形核在 z 上几乎全是 padding。同时 MSCA 参数只占普通残差块的 31%，前向却是它的 2 倍（9 条深度卷积 + local + 1×1，全在 stage 分辨率上，且每个 block 都挂一份）。既有测试 tests/test_swa_lka.py:51-56 只断言"小空间尺寸合法"（不崩），未涉及有效性。建议：核长按轴可配，或按 spacing_normalization 自动推导 z 向核长。

[S5-F] 插值上采样强制 fp32 往返，torch 2.7 已原生支持 bf16

Upsample.forward:1511-1529 对 trilinear/nearest 在 bf16/fp16 下先 .float() 再插值再转回。实测本环境：



in torch.bfloat16 -> out torch.bfloat16; fp32 intermediate 1.05 MB vs bf16 0.52 MB
native bf16 trilinear dtype: torch.bfloat16      # 原生支持，无需上采
代价发生在 decoder 最高分辨率级——正是激活峰值处，多一份 2 倍大小的 fp32 临时张量 + 两次 cast 的带宽。注释在 adm_unet.py:63 里说明了历史原因（"旧 PyTorch (<2.1) 上 upsample_nearest2d 缺 bf16/fp16 kernel"），该前提在本仓 torch 2.7 上已不成立。

[S5-G] Downsample('conv') 的 kernel_size == stride（非重叠 tile），且池化模式下 stride 完全不进权重形状

:1252 显式令 kernel_size=st；maxpool/avgpool 则是 Pool(st) → 1×1 conv（:1253-1260）。实测：



stride=2       -> conv kernel=(2,2,2)
stride=(1,2,2) -> conv kernel=(1,2,2)
两个后果：① 与 nnU-Net/ResEnc 的 3×3 stride-2 相比，tile 之间无重叠、下采样处感受野被压到最小，且 stride=1 的轴上核长为 1（该轴在下采样层完全没有空间混合）；② maxpool/avgpool + 1×1 时 stride 不进任何权重形状——这正是 S4-A（推理期改 patch_size 导致 stride schedule 漂移却能 strict-load 成功）的算子级根因，本轮从算子侧确认：形状兜底在这条路径上结构性失效，只有 conv/transpose（kernel==stride）能被形状查出来。

[S5-H] 多膨胀分支有两套实现，只有一套能在推理期折叠

MultiRFBlock（resnet.py:277-424）与 DilatedReparamBlock（mednext.py:145-254）是同一思想的两份代码。后者实现了正确的推理期折叠（实测折叠等价性 maxdiff=2.6e-06 / rel=6.7e-07，且 switch_to_deploy 幂等），前者在默认 branch_norm_act=False 下整条分支路径（多分支 conv → concat → 1×1 fuse）是纯线性（代码注释 :345-347 已承认），因此本可折叠为单个等效卷积却一直按 N 分支计算——训练与推理都付 N 倍代价。二者的分支校验、膨胀-padding 推导、通道分配也各写一遍。

轻微
问题	位置	说明
BlurPool 的 filt_size=2 输出尺寸 +1	blocks.py:1095-1099、:1110	pad = 2//2 = 1 与偶数核不匹配。实测 filt_size=2 -> (1,4,9,9,9)（应为 8³）。当前 Downsample 硬编码 filt_size=3 故不可达，但 _BINOMIAL 表把它当成合法档位暴露着
各 Stage 对空 drop_path_rates 列表 IndexError	convnext.py:143、resnet.py:497、mednext.py:522	实测 ConvNeXtStage(..., drop_path_rates=[]) -> IndexError: list index out of range；后续 block 有 i < len(...) 保护，唯独首块 [0] 无
ECA3D 在函数体内 import math	blocks.py:279	模块顶部已 import math（:7）；纯不一致
GroupNorm 组数回退有三种口径	get_norm:189-199（静默折半+一次性告警）/ SelfAttentionBlock:1003-1006（静默折半，无告警）/ MultiRF:355-366（显式报错）	同一语义三份实现三种行为
_LinearQKVAttention 多乘 head_dim**-0.5	blocks.py:953	Shen 2021 的 ρ(Q)(ρ(K)ᵀV) 无此缩放；输出被额外缩小 1/√d（zero-init proj 可吸收，但与 docstring 声称的"Shen 2021"不符）
DySample3d 偏移单位与 align_corners 偏离原作	blocks.py:1403-1416、:1376-1384	归一化用低分辨率尺寸 2/(W-1)（原作用高分辨率 2/W），且 align_corners=True，与框架其余插值一律 align_corners=False 不一致
CARAFE3d 硬编码 3D	blocks.py:1307-1327	nn.Conv3d / F.pad([pad]*6) / unfold×3，破坏本文件的 nD 契约（Upsample:1455-1459 已 fail-fast，故不致误用）
Upsample('transpose') 带 bias	blocks.py:1476	框架其余 conv 一律 bias=False + 后接 norm；此处 ConvTranspose 默认 bias=True 且无 norm
icnr_init_ 的 init 形参未使用	blocks.py:1204	死参数（内部恒用 kaiming_normal_）
get_conv3d() 兼容别名无生产消费者	blocks.py:168-170	死代码
SE/CBAM/Coord 的 mid 下限不一致（4/4/8）	:252、:301、:360	无理由的三个魔数
PreActResNetBlock 的投影捷径作用于原始 x	resnet.py:96-98	He 2016 v2 的投影捷径作用于预激活后的张量；注释自称"标准 pre-act"，实际不是
BottleneckBlock.expansion 语义反转	resnet.py:127	mid = out_ch // expansion 是"内部压缩比"，而 ResNet 的 expansion 是"输出扩张比"；同名反义
MedNeXt expand_ratio 全网恒定	factory.py:277	原论文按 stage 变化（如 2/3/4 金字塔）
upkern 用 align_corners=True	mednext.py:329	与 MedNeXt 官方默认 False 不同（docstring 已声明，属已知偏差）
reparameterize_model 无日志/计数	mednext.py:257-263	对任何带 switch_to_deploy 的模块生效，折叠了几个块用户无从得知
checkpoint_if 的 BN 冻结只在 fn 是 nn.Module 时生效	blocks.py:95-104	全仓唯一的非 Module 调用点是 _ADMMiddle（adm_unet.py:443 传 bound method）；ADM 全用 GroupNorm 故当前无实际影响，但这是一条隐式约定
ADM/EDM2 的 num_fg_classes 形参名误导	adm_unet.py:880、edm2_unet.py:761	两处都传 out_classes（含背景/多分辨率组），口径一致但名字反义
③ 合理性与设计评价
做得好的（建议固化为契约）

checkpoint_if 的 BN 双更新修复（:51-104）：context_fn 在重算前快照、退出时恢复 running buffers，语义上恰好抵消第二次前向的 momentum；配合 use_reentrant=False + preserve_rng_state=True，"开检查点与不开数值严格一致"这条在同类框架里经常被做错。S5 复核算子侧：mednext 的 DilatedReparamBlock 内部 BN 也落在 fn.modules() 覆盖范围内（stage 才是被 checkpoint 的 Module）。
AMP 数值细节的系统性处理：DropPath 先 fp32 采样再转 dtype（:119-121，规避 bernoulli 后端差异）、GlobalResponseNorm 与 LayerNorm3d 的统计量 fp32 累加（:137-142、convnext.py:26-32）。这类"只在 fp16 大空间求和时才暴露"的坑被提前堵住了。
nD 单实现：八张分派表 + _check_dims 让 2.5D（spatial_dims=2）与 3D 共用全部块，唯一显式拒绝的是 R2Plus1DBlock（resnet.py:181-188，且报错文案给出了替代方案）与 carafe/dysample（:1455-1459）。这是 2.5D/3D 双模态框架能维持单份代码的基础。
RoPE 实现的两个非平凡决策（:570-590、:623-645）：torch.compiler.is_compiling() 时旁路 Python dict LRU（避免 graph break），以及"逐轴旋转块收集后一次 cat"而非 clone+切片写回（对 compile 友好且省一张全量拷贝）。
各向异性 fail-fast 下沉到算子构造期（:1262-1279、:1481-1506）：blurpool/pixelunshuffle/carafe/dysample 在拿到非各向同性 stride 时直接报错并给出可用替代，而不是静默产出错误几何。
DilatedReparamBlock 的折叠正确（实测 rel 误差 6.7e-7）：_fold_conv_bn + _expand_dilated_kernel 的稀疏展开、幂等 switch_to_deploy、以及 upkern_remap_state_dict 里对 plain↔reparam 前缀不匹配的显式告警（:353-372），完成度明显高于其余实验性算子。
MultiRF 的两个防御：强制存在 dilation=1 守门支路（resnet.py:309-311，抗网格效应）、branch_norm_act 下对 GroupNorm 不整除显式报错而非自适配（:355-366，并在报错里列出四种修法）。
结构性问题

归一化选型没有单一真相源，且"不整除"有三种行为：get_norm 静默折半+一次性告警、SelfAttentionBlock 静默折半无告警、MultiRF 显式报错；此外 MedNeXt 固定通道级 GN、ConvNeXt 固定 LN、AttentionGate 对单通道图做 norm（S5-B 的根因）。六处口径分散，新增 backbone 必然再加一处。
注意力有三套互不相通的抽象：make_attention（块内通道/空间门控，配置来自 se_reduction）、SelfAttentionBlock（stage 尾 token 注意力，配置来自 selfattn.*）、AttentionGate3D（skip 门控，配置来自 attn_gate_norm）。三者可同时开启且互不感知，配置层也没有"总注意力预算/显存"视图；lka/msca 的核参数甚至无法从配置到达（make_attention:498-501 不透传 kwargs）。
算子成熟度差异巨大却同级暴露：transpose/trilinear/nearest 是生产级，pixelshuffle 次之，carafe（S5-A：不可用）与 dysample（偏移语义偏离原作）是实验级；upsample_mode 白名单一视同仁，没有实验性标注、没有显存守卫、日志里也看不出差别。同样的问题在 attention_type 上（lka/msca vs se/eca）。
各向异性只贯彻到了"重采样算子"，没贯彻到"感受野算子"：Downsample/Upsample 支持 per-axis stride，但 LKA(k=5/7,dil=3)、MSCA(7,11,21)、MedNeXt kernel_size、ConvNeXt dwconv7、selfattn window_size/grid_size 全是各向同性标量（_normalize_spatial_sizes:648-661 支持序列，但配置层只给 int）。对层厚 5mm 的 CT，这是系统性错配（S5-E 是它最可测量的形态）。
同一思想两份实现（S5-H）：MultiRF 与 DilatedReparamBlock；以及 Downsample('conv') 与 ConvNeXtDownsample（convnext.py:163-173，硬编码 k=2/s=2/bias=True、norm 在前、不跟随 norm_type）。
私有名跨模块依赖：resnet.py:10-12、convnext.py:13、mednext.py:26 都从 blocks import _CONV/_BN/_DROP 等下划线名。blocks.py 1541 行里混装了工具函数 / 通道注意力 / token 注意力 / 重采样 / 初始化五类内容，"论文模块"（LKA/MSCA/CARAFE/DySample）更适合独立文件，否则这三个骨干文件会一直依赖它的私有约定。
④ 优化空间（含 GPU / 吞吐 / 显存）
#	优化	预估收益	风险
1	padded == orig 时不传 attn_mask（S5-D）；q/k/v 合并为一次 partition，去掉未使用的 offsets 列表	实测 SDPA +43% 时延可去；flash 可用的机器上收益更大；partition 拷贝 3→1	极低；数值不变（mask 全 True 时是 no-op）
2	去掉 trilinear/nearest 的 fp32 往返（S5-F），torch≥2.1 原生支持	decoder 最高分辨率处少一份 2× 大小的 fp32 临时张量 + 两次 cast	低；bf16 插值与 fp32 插值有末位差异，需回归比对
3	插值上采样改为「先 1×1 降通道 → 插值 → 3×3 精修」而非「插值 → 3×3(in→out)」（:1477-1479）	精修卷积在高分辨率上跑，当前 FLOPs ≈ 8×；重排后主成本降到低分辨率	中；改权重布局，需作为新开关默认保旧
4	CARAFE3d 重写为低分辨率加权形式 / 或加显存 fail-fast（S5-A）	从"必 OOM"变为可用；实测峰值 964→约 70 MiB 量级	中；数值等价需逐值验证
5	MultiRF 增加推理期折叠（复用 DilatedReparamBlock 的 _fold_conv_bn/_expand_dilated_kernel），并合并两套膨胀分支实现（S5-H）	推理少 N-1 条分支；同时消灭一份重复实现	中；3D 下折叠后的稠密核可能比稀疏分支更贵，需按 max(dilation) 判断是否折叠
6	DropPath 改 x.new_empty(shape).bernoulli_(keep)（:118-122）	每个残差块省一次 fp32 全量 torch.full + 一次 dtype cast；深网络里块数以百计	低；需保持 fp32 采样语义以维持 AMP 一致性
7	RoPE 缓存扩展到 flat_coords（当前只缓存 cos/sin，:616-621 的 meshgrid 每次前向重建）	省一次 meshgrid + reshape；token 数大时非平凡	极低
8	深度卷积骨干（MedNeXt/ConvNeXt）走 channels_last_3d	torch 2.7 的 dw-conv 在 channels_last 下 kernel 更优	中；需与 S7 的 AMP/torch.compile 协同，且要全网统一否则反复转置
9	MSCA/LKA 的核长按轴可配（S5-E）	薄 z 上去掉 4/7 的无效抽头计算；实测 MSCA 前向是普通残差块的 2×	低（纯新增字段）；改默认会变数值
⑤ 2026 可借鉴项
方案	借鉴点	适配代价	优先级
PyTorch flex_attention（2.5+ torch.nn.attention.flex_attention，本环境 2.7.1 已具备）	用 mask_mod/score_mod 表达窗口/网格，不物化 mask、不做 partition/unpartition 的 rearrange，且能编译成融合 kernel。直接解掉 S5-D 与 _window/_grid_partition_tokens 的全部拷贝；本仓的窗口/网格语义正好是它的典型用例	中（需核实其 3D flatten 语义与本仓 spatial_shape 的对应，落地前查最新官方文档）	高
QK-Norm（Dehghani 2023, ViT-22B；已成为 2024+ 大模型标配）	SelfAttentionBlock 目前只有 PreNorm + zero-init proj，缺 QK-Norm；3D token 数大、softmax logits 易发散，这是 5 行改动换训练稳定性	低	高
UniRepLKNet / RepLKNet 的结构重参数化（本仓 mednext.py 已落地一半）	把 DilatedReparamBlock 提升为公共 primitive，MultiRF、ConvNeXt dwconv7、MedNeXt 共用一套「训练多分支 / 推理单核」机制（解 S5-H）	中	高
各向异性大核 / 逐轴 spacing 感知核（SegNeXt 条形核思想 + nnU-Net 的 spacing 驱动几何）	MSCA 已有逐轴条形核的骨架（:458-467），只差"轴长随 spacing 变化"；对 CT 大层厚是直接收益（解 S5-E）	低	高
MedNeXt 原论文的 per-stage expand_ratio + UpKern 分阶段迁移	当前 expand_ratio 全网恒定（factory.py:277）；论文用金字塔式 R，且 UpKern 是"小核训练→大核微调"的两阶段流程，本仓只实现了权重重映射，没有流程	低	中
SSM / Mamba 类线性全局建模（SegMamba、U-Mamba、nnMamba，2024）	作为 _LinearQKVAttention 的替代：同为 O(N)，但在 3D 长序列上的表现明显更好；本仓已有"stage 尾插入全局块"的插槽（factory.py:175-187），接入面很小	中高（新依赖 / 自实现扫描算子，需评估必要性）	中（先做消融档位）
上采样算子选型的近年结论（DySample ICCV2023 vs CARAFE vs FADE/SAPA）	共识是 DySample 性价比最高、CARAFE 在 3D 下代价不可接受。建议把 carafe 标为实验性，并把 dysample 的偏移归一化对齐官方实现	低	中
LayerScale / zero-init residual 的统一声明机制（timm init_weights / MMEngine init_cfg）	与 S4-B 同一条：ConvNeXt 有 LayerScale、MedNeXt 没有；SelfAttentionBlock/DySample/ICNR 有零初始化契约却会被 init_strategy 抹掉。模块级声明可一并解决	低	高（与 S4-B 合并落地）
⑥ 与既有测试 / 契约的冲突检查
S5-A（CARAFE）：test_blocks_sampling.py 只做形状/梯度冒烟，无显存断言 → 改写实现需新增数值等价用例；加 fail-fast 会拒掉当前能"小尺寸跑通"的测试配置，需把守卫阈值设在实际显存估算上而非固定尺寸。
S5-B（AttentionGate 归一化）：tests/test_a5_blocks.py:213-228 固化的是 attn_gate_norm ∈ {batch, instance, group} 到具体 norm 类的映射，不覆盖 auto 分支；把 auto 的默认目标改为 batch（或改 psi 为无 norm + bias）不破坏该断言，但会改变所有 skip_attention=True 历史配置的数值 → 按"legacy 默认保旧"原则宜作为新枚举值落地并在 CHANGELOG 明示。
S5-D（attn_mask）：tests/test_selfattn.py:245-273 的参考实现逐行复制了"恒建 mask 并传入 SDPA"这一写法。改为"无 padding 时不传 mask"后数值应严格相同（mask 全零 bias），allclose 断言不会失败，但该测试的参考路径会与实现分歧 → 建议同步更新参考实现，否则它不再是有效对照物。
S5-E（各向异性核）：tests/test_swa_lka.py:36-70 断言形状、梯度、小空间合法性与分支数（len(b) == 3 逐轴条形核）。新增逐轴核长字段属纯新增，默认保旧则无冲突；test_msca_branch_count_matches_scales_and_dims 依赖"每分支恰好 spatial_dims 条 strip"，改结构时需同步。
S5-F（fp32 上采）：无测试断言中间 dtype；输出 dtype 契约不变。仅需一次数值回归比对。
S5-G（Downsample kernel）：test_anisotropic_downsample.py 在同一 cfg 内构建+前向，不跨 cfg 加载 → 与 S4-A 的架构指纹方案同一批次处理最合适；改 kernel_size 会改权重形状，属独立批次。
S5-H（MultiRF 折叠）：test_dilated_reparam.py 已固化 mednext 侧的 deploy 等价性（本轮实测 rel 6.7e-7 通过）；给 MultiRF 加折叠属纯新增推理路径，需照搬同款等价性用例。注意 test_dilated_reparam.py:255 仍是 S1-A 里那条"预先 setattr cfg.model.reparam_deploy 掩盖字段缺失"的用例——S5 复核确认该问题在配置层，与算子层无关。
S5-C（1³ 特征图）：无既有用例覆盖；修正为几何校验期 fail-fast 属纯新增，会拒掉理论上可写但一定崩溃的配置，无断言冲突。
轻微项：BlurPool filt_size=2、空 drop_path_rates、icnr_init_ 死参数、get_conv3d 均无测试覆盖，修正不破坏任何现有断言。
与 S1–S4 的一致性：S5-G 从算子侧确认了 S4-A 的根因（池化+1×1 路径下 stride 不进权重形状，形状兜底结构性失效），二者应合并为同一批次（架构指纹落盘 + 推理期比对）；S5-B/S5-E/S5-F 与 S4-F（drop_path schedule 方向）同属"默认值即次优"族；S5-C 与 S1-「amp_dtype/compile_mode 配置期无枚举校验」、S2-G、S3-D、S4-C 同属"启动期缺前置校验、代价推迟到长耗时之后"族——这已是连续五轮出现的最高频结构性问题，建议 S12 单列为一个落地批次。

S6	数据增强/处理	data/augment.py（GPU 同步 3D 增强、Companion）+ dataset 内归一化/resize/pad 预处理路径 + augment.* 配置语义	S2

审查报告：数据增强 / 处理
本轮遵守：只审不改（未产生任何代码/文档改动）；结论均给 文件:行号；关键判断用 D:\miniconda\envs\torch27_env\python.exe 实测复现（脚本写在系统临时目录 %TEMP%\s6_probe，未落工作区）；不臆断库用法。

本轮范围：augment.py（GPU 同步 3D 增强 + Companion）、dataset 内的归一化 / resize / edge-pad 预处理路径（dataset.py 的 preprocess_image / preprocess_label / resize_3d / extract_z_patch_padded / 三个 _getitem_max_fov）、augment.* 与 data.aug_oversample_ratio 的配置语义（core.py::AugConfig / _validate_augment）、以及增强在训练链路上的实际调用点与前后契约（trainer.py:461-479、views.center_crop）。视图拆分/折叠本身属 S8，本轮只在"增强的输入输出契约"处引用。

① 事实梳理
调用链（seg 训练）：dataset.__getitem__ 发单 max-FOV cube（未折叠 rank-5、单通道）→ trainer._train_epoch H2D（@d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:462-468）→ self.augmentor(image, label, wmap)（:471）→ views.center_crop 去 oversample 余量（:472-474）→ pipeline.prepare_batch 拆视图/折叠（:477）→ AMP forward。验证侧无增强、无裁剪（validation.py:397-400），因为 val 的 aug_oversample_ratio 被 spec 强制为 1.0（specs.py:130-134）。WORKFLOW.md:165 的"dataset 恒发未折叠 3D → GPU 3D 增强 → 裁余量/视图拆分 → 送模型前折叠"契约，在 seg 主链路上逐行核对通过。

增强器构造：GPUAugmentor(cfg.augment, max_scale=max(scales), label_fill=label_values[0], seed=train.seed+7919*(rank+1), inplace=True)（trainer.py:140-149）。三个非默认入参各有明确用途：max_scale 只用于缩小 elastic_deform_alpha（augment.py:123）、label_fill 作 label 越界填充、inplace=True 跳过入口 clone。

管线顺序（augment.py:113-166）：记录 clamp 基准 → flip → affine⊕elastic（融合为单次 grid_sample）→ brightness → contrast → gamma → noise → blur → lowres → intensity_clamp → grid_dropout。注释明确解释了两个非平凡的排序决策（clamp 基准取增强前、dropout 必须在 clamp 之后）。

四个真实契约：

Companion 同步：空间变换由同一份 grid 消化，label nearest+oob_fill=bg、wmap 按 wmap_interp_mode+oob_fill=1.0、gen 的 cond/wmap oob_fill=None 保 border（:399-405、:175-183）。
仿射⊕弹性单次重采样：同时选中的样本合成 G(x)=Θ(x+d(x))=affine_grid + M·d（:381-392），只插值一遍。
CPU 采样选样掩码：_bernoulli_mask 在 CPU 上采样，声称"后续 any/sum/nonzero 均零同步"（:27-31、模块 docstring:9-11）。
强度变换只作用 image；grid_dropout 也只挖 image，label/wmap 原样（:453）。
预处理侧：preprocess_image 单次分配 + in-place clip/normalize（dataset.py:478-511）；preprocess_label 向量化 one-hot（:538-544）；resize_3d 图像 order=1 / label order=0 + mode='nearest' 防边界注 0 + 形状防御性校正（:550-581）；extract_z_patch_padded 越界 mode='edge' 复制保物理 FOV（:1076-1093）。aug_oversample_ratio 的落点分两种：z_axis/2_5d 只放大 z（:898-900），cubic/whole 三轴同比（:1149-1150、:1350-1351）。

② 正确性问题
严重
[S6-A] 2.5D / z_axis 下"出面旋转 + aspect_correct"把两成以上的监督标签抹成背景
random_rotate_range 默认对三轴共用（core.py:241、random_rotate_range_per_axis=None），random_affine_aspect_correct 默认 True（core.py:250，实现在 augment.py:258-260：R←diag(1/a)·R·diag(a)，a=(W,H,D)）。aspect 校正把旋转搬到 voxel-count 各向同性坐标里做——这对立方 patch 是正确的，但对 D≪H 的薄 slab 意味着：一个面内偏移 u 的体素在出面旋转 θ 后 z 位移 u·sinθ，而 slab 只有 D 深。同时 z_axis/2_5d 的 oversample 只在 z 有余量，面内没有任何余量可裁（core.py:254-255 已承认这一固有限制，但只提到平移）。越界体素的 label 被 oob_fill=背景 覆写（augment.py:403-404），image 却是 border 复制的"看起来合理"的内容 → 系统性假阴性监督。

实测（B=2，全前景 label，affine_prob=1.0；第二组是中心裁剪之后、即真正进损失的比例）：



裁剪前（B,1,36,256,256）label 被抹成背景的比例
  aspect_correct=True , rot=±15 三轴 : 0.2896
  aspect_correct=False, rot=±15 三轴 : 0.1083
  aspect_correct=True , rot=±30 三轴 : 0.4983
  aspect_correct=True , 仅面内(绕 D) : 0.0266
  各向同性 cube 128³, aspect=True, ±15 : 0.1087
 
裁剪后（损失真正看到的）
  seg2_5d.yaml [12,256,256] r=1.5 s=2.0 rot±15 → eD36→crop24 : 0.2216
  seg3d.yaml   [16,128,128] r=1.5 s=2.0 rot±30 → eD48→crop32 : 0.1670
  cubic 三轴余量 [64,128,128] r=1.5 rot±15（三轴都裁）      : 0.0000
三点结论：① 这是生产默认配置（configs/seg2_5d.yaml:78 rot ±15、configs/seg3d.yaml:80 rot ±30、两者 aspect_correct: true）下的行为，仿射触发概率 0.3 时，平均每个 batch 约 5–7% 的体素带着错误的"背景"标签进损失，且空间上高度集中在面内边缘（模型会学到"靠边=背景"）；② aspect_correct=True 把 10.8% 放大到 29%，即这个"消除剪切"的改良项在薄 slab 上反而是放大器；③ cubic 模式完全没有这个问题（三轴都有余量，裁剪后 0.0%）——所以问题的根因不是仿射本身，而是"无面内余量的模式 + 各向同性角度范围"这一组合无人校验。_validate_augment 只对 random_translate_range 做了余量对比警告（core.py:1837-1853），对旋转引入的边缘带完全不检查。正解：出面角度上界按 arcsin(D_eff / max(H,W)) 自动收敛（或强制 rotate_range_per_axis），并把 oob 体素通过 wmap=0 排除出损失（见 ④-1、⑤）。

[S6-B] whole 模式下 aug_oversample_ratio>1 造成训练/验证 FOV 与体素尺度双重错位
SegDataset3DWhole 把整卷 resize 到 extract_size = round(patch_size × oversample)（dataset.py:1350-1351、1367-1371），trainer 增强后中心裁回 patch_size。cubic/z_axis 的 oversample 是"多抽一点原始体素、裁回来"（分辨率不变），whole 的 oversample 是"整卷放大 r 倍再裁中心"——分辨率和 FOV 同时变了。而 val 的 oversample 被强制 1.0（specs.py:130-134），整卷 resize 到 patch_size、不裁。实测：



volume=(80,300,300) patch=(64,128,128) r=1.5
  TRAIN: 整卷→(96,192,192) → 增强 → 中心裁→(64,128,128)
         = 只看每轴中间 67%，体素尺度比 val 细 1.50×
  VAL  : 整卷→(64,128,128) = 看 100%，1.00×
  实测源坐标覆盖跨度: train=199 vs val=299 (比值 0.67)
后果：① 训练分布与验证/推理分布在尺度上系统性错位（同一解剖结构在训练时大 1.5 倍）；② 整卷模式常用于"看全局"，而 r>1 恰好把边缘 33% 永久排除出训练；③ 与 S2-「whole 模式 label 未做 antialias 讨论」叠加，whole 路径是三种模式里 legacy 假设最多的一条。WORKFLOW.md:36-43 把这套流程当正常流程描述，没有任何提示；Config.validate 也不拦（whole 只强制 multi_res_scales=[1.0]）。正解：whole 模式下 oversample 应改为"resize 到 patch_size 后 pad 出余量"或直接禁用（fail-fast + 提示改用 cubic）。

中等
[S6-C] 增强私有 RNG 不进 checkpoint → 违反"resume 位精确恢复（含 RNG）"契约
GPUAugmentor 刻意用私有 _gen_cpu / _gen_dev 与全局 RNG 解耦（augment.py:65-92，注释称这是"固定 seed 等价性验证的前置"），而 snapshot_rng_state 只快照 torch CPU/CUDA + numpy + python 四路全局状态（checkpoint.py:332-340）。Trainer 每次构造都用同一个 seed = train.seed + 7919*(rank+1)（trainer.py:146，不含 epoch/step）。实测（同 seed 新建实例，连续三步的 brightness 偏移）：



run1: [[-0.08487,-0.06067], [-0.04449,0.07146], [-0.01867,-0.05364]]
run2 (fresh instance, same seed): 完全相同 -> identical: True
即每次 resume 都从头重放同一条增强序列：一个训练到 60 epoch 后中断三次的任务，会在三段续训里反复看到 epoch 0 的那批增强参数，等效增强多样性被压缩。segtask_v1/docs/WORKFLOW.md:215 明确承诺"resume 位精确恢复（含 RNG）"——该承诺在增强这一路上不成立。正解：把两个 Generator 的 get_state() 纳入 checkpoint（或把 seed 派生为 f(seed, rank, epoch)）。

[S6-D] "零 device→host 同步"契约只兑现了一半：10 个算子里 6 个触发同步
模块 docstring:9-11 声称掩码与逐样本标量在 CPU 采样"避免对 CUDA RNG 结果的隐式 device→host 同步打断流水"。但采样完还要把索引/参数搬上卡，而空间与噪声类算子用的是不带 non_blocking 的 .to(image.device)（:199、:344、:590、:613、:481），强度类算子则正确地传了 non_blocking=True（:528、:547、:571）。另外 _grid_dropout_companions 把 CPU 掩码搬上卡后在设备上做 nonzero（:481-482），这是真正的 D2H 同步（同文件其它算子都是在 CPU 上 nonzero 再搬）。实测（torch.cuda.set_sync_debug_mode("error")，逐算子）：



flip            : SYNC       brightness      : no sync
affine          : SYNC       contrast        : no sync
elastic         : SYNC       gamma           : no sync
noise           : SYNC       intensity_clamp : no sync
blur            : SYNC
lowres          : SYNC
grid_dropout    : SYNC
对照关系一一对应（传 non_blocking=True 的三个 + 纯设备端的 clamp 全部干净），说明这是可低成本消除的实现疏漏，不是原理限制。开 prefetch_to_gpu 时这些同步会直接抵消预取带来的重叠收益。

[S6-E] elastic_field_mode='gaussian' 是"叠加平滑"而非"改用高斯场"，幅度被二次衰减；控制网格下限使薄 z 的实际平滑尺度 ≠ sigma
_elastic_grid_disp（augment.py:272-309）无论哪种 mode 都先做"粗网格 randn → trilinear 上采"，gaussian 只是在其后再串三次可分离高斯卷积（:284-298）。配置注释（core.py:264-266）写的是"legacy 保持粗网格 randn 上采样；gaussian 使用高斯核平滑位移场"，读起来是二选一。实测（D=36,H=W=256, sigma=5, alpha=7，换算回体素）：



legacy    rms voxel disp (W,H,D) = [3.982, 4.000, 3.979]  max=28.61
gaussian  rms voxel disp (W,H,D) = [1.623, 1.648, 1.621]  max= 8.39
同一个 alpha 在两种 mode 下实际位移差 2.45 倍——切换 mode 等于同时改了幅度，用户无从预期（elastic_normalize_displacement 能把幅度拉回，但它默认 False 且是另一个开关）。附带同源问题：控制网格 max(round(D/sigma), 4) 的下限 4 让薄 z 的平滑尺度脱离 sigma：



D=36 sigma=5 -> 控制网格 z=7 -> 实际平滑长度 5.14 vox（≈ 请求值）
D=12 sigma=5 -> 控制网格 z=4 -> 实际平滑长度 3.00 vox
D=8  sigma=5 -> 控制网格 z=4 -> 实际平滑长度 2.00 vox
即 2.5D 生产配置（D=12）上 z 向弹性形变比配置声明的粗糙 40%，且 sigma 在 z 上事实上失效。这与 S5-E「大核/条形核全各向同性」、S1-「几何真相源副本」同属"各向异性只贯彻到了部分模块"。

[S6-F] augment.* 几乎没有区间/枚举校验，非法值静默生效或静默失效
_validate_augment（core.py:1820-1921）覆盖了 wmap_interp_mode / rotate_range_per_axis 形状 / translate_range 长度 / blur sigma / elastic sigma·alpha·mode / noise std / lowres zoom / gamma range，但所有概率字段、flip 轴、scale range、brightness/contrast range 的顺序、grid_dropout 的 ratio 与 holes 全部无校验。实测 8/8 条非法配置被 sync()+validate() 接受：



ACCEPTED : random_flip_axes=[0,1]（batch / 通道轴！）
ACCEPTED : random_flip_prob=1.7
ACCEPTED : random_affine_prob=-3
ACCEPTED : random_scale_range=[0.0,0.0]（退化，网格塌成一点）
ACCEPTED : random_scale_range=[1.2,0.8]（lo>hi，uniform_ 语义未定义）
ACCEPTED : random_rotate_range=[10.0]（长度 1，运行期 IndexError 才暴露）
ACCEPTED : random_brightness_range=[0.5,-0.5]（lo>hi）
ACCEPTED : grid_dropout_ratio=5.0, holes=0
其中两条有可观测的破坏性后果（实测）：grid_dropout_ratio=5.0 → 整幅图像 100% 置零而 label 不动，模型被要求在纯零输入上分割；grid_dropout_holes=0 → range(0) 空循环，静默 no-op（用户以为开了强正则）。random_flip_axes 无白名单：对 seg 主链路 C=1 时轴 1 无害，但 gen/ssl/cls 的多通道输入上轴 0/1 语义完全错误。这与 S1-「amp_dtype/compile_mode 配置期无枚举校验」、S2-G、S3-D、S4-C、S5-C 是连续第六轮出现的同一族问题。

[S6-G] max_scale / oversample 的幅度校正只做了 elastic alpha 一处
增强作用在 max-FOV cube 上（尺寸 = patch × oversample × max_scale），但配置里的几何幅度是用户按"主视图 patch"直觉写的。代码只对 elastic_deform_alpha 除了 max_scale（augment.py:123，且没有除 oversample）。其余全部未校正：

random_translate_range 是 cube 归一化坐标：seg2_5d 生产配置（pD=12, r=1.5, s=2.0 → eD_max=36）下 0.1 的平移 = 1.8 体素 = 主视图深度的 0.30，即实际幅度是配置值的 3 倍；
_validate_augment:1844-1853 的"边缘复制带 vs 中心裁剪余量"警告公式 margin=(r-1)/(2r) 同时忽略了 max_scale 和"z_axis 只有 z 有余量"——对 2.5D 它在面内给出的余量估计恒为虚假的正值；
grid_dropout 的洞尺寸按 cube 边长比例算（augment.py:465-468），裁剪后占最终 patch 的比例被放大 r·s 倍；
elastic_deform_sigma 同理（见 S6-E）。
[S6-H] normalize='zscore' 配默认 global_mean/std = 直接把裁剪后的 HU 送进网络，无任何校验
preprocess_image 的 zscore 分支用配置里的 global_mean=0.0 / global_std=1.0（core.py:144-145、dataset.py:503-506），全仓没有任何从数据自动估计这两个量的代码（finalize_from_data 只回写 label_values/num_classes，loader.py:418-436），配置层也不校验"选了 zscore 却没给统计量"。实测：



输入 HU [-3000,-500,0,500,3000] , normalize=zscore, mean=0/std=1
输出 [-1024.0, -500.0, 0.0, 500.0, 1024.0]   # 只做了 clip，归一化是恒等
Config.validate(): ACCEPTED（无 error，也无提示）
有意思的是同一个校验器已经为 zscore 写了一条"增强幅值量纲不匹配"的警告（core.py:1904-1921，实测会打印），却没检查更根本的"std=1 意味着根本没归一化"。后果是网络输入量级 1000×、第一层激活爆炸，症状表现为"zscore 训不动"，用户很难定位到配置。这与 S2-⑤「per-case 归一化缺失」是同一条线上的两级缺口（先补 dataset-level 统计，再谈 per-case）。

轻微
问题	位置	说明
inplace=True 的收益远小于宣称	augment.py:104-107 vs :477、:528、:547、:576	实测：grid_dropout 恒 image.clone()、brightness 返回 image+shift 新张量（out.data_ptr()!=img.data_ptr() 均为 True）。inplace 只省了入口那一份，管线内部仍有多份全量分配
flip 无 prob<=0 早退	:187-203	其余算子都有 if prob <= 0: return；flip 每轴都要采一次 CPU 掩码 + 一次同步 H2D（见 S6-D），random_flip_prob=0 时纯浪费
oob 判据略欠采	:394	align_corners=False 下最外半个体素的 `
blur 核长由 sigma 上界决定	:618	实测 sigma=[0.5,5.0] → ks=31 对所有被选样本（含 sigma=0.5 的）；注释称"归一化后等价"数值上成立，但算力按最坏情况付
simulate_lowres 三轴同 zoom	:641-668	实测 D=12, zoom=0.5 → z 深度 6。nnU-Net 的 SimulateLowResolution 对各向异性数据有 ignore_axes；此处在薄 slab 上模拟的是"层厚翻倍"而非"面内低清"
grid_dropout 挖 image 不挖 label	:453、:492	语义是 Cutout；但对分割而言等于要求模型在"信息被删除"的洞里给出正确前景，与 oob_fill=背景 的取向自相矛盾（一个填背景、一个要求照常预测）。默认 prob=0，属设计取向问题
三个"旧签名包装"已无生产消费者	:206-219、:410-443、:496-511	grep 确认 _random_flip / _random_affine_elastic / _grid_dropout 只被 tests 引用（6 个测试文件）；与 S3「同一思想两份实现」同型，且它们与 companion 版的默认 oob_fill 不一致（包装版 flip/dropout 传 None，_random_affine_elastic 传 0.0/1.0）
Companion.oob_fill 对 wmap 恒填 1.0	:179	与 make_data 的"+1 偏移，1.0=中性"口径一致（make_data.py:299-321），但该耦合没有写成契约；gen 路径又用 None（test_companion_augment.py:96-116 固化），三种任务三种语义散在调用点
intensity_clamp 每 batch 两次全量 reduce	:115-118	默认 True，amin/amax 各扫一遍；实测该项本身不引入同步，但对大 cube 是可省的带宽（可与 gamma 的 min/max 复用，gamma 也在算同样的量，:563-564）
resize_3d 在 label 上恒 order=0，image 侧 antialias 默认关	dataset.py:566-570、specs.py:58	S2 已记；S6 复核确认 z_axis 面内 resize（:985-989）与 whole 全卷 resize（:1367-1371）都走这条路径，是"CPU worker 上的 scipy.zoom + GPU 上的 grid_sample"两次插值串联（见 ④-5）
注释乱码"动态阐"	:572、losses.py:78	应为"动态展开/构造"；与 S3 的"夹匯"同批
③ 合理性与设计评价
做得好的（建议固化为契约）

仿射⊕弹性融合为单次 grid_sample（:381-392）：同时选中的样本只插值一遍，G=Θ(x+d) 的合成用 theta 的线性部分左乘位移，数学上正确且省一次全量重采样。多数同类框架（含早期 MONAI）在这里做两次串行 warp。
Companion 抽象（:34-44、:399-405）：把"谁跟着一起变、用什么插值、越界填什么"变成数据声明而不是 if 分支，seg/gen/ssl/cls 四个任务共用同一份 warp 代码而语义各自正确，是本层最有价值的设计。
CPU 侧采样选样掩码的思路正确（:27-31）：any/sum/nonzero 全在 CPU 上完成，避免了"CUDA 上采样 → 取回判断"的经典同步陷阱——只是搬运环节没贯彻到底（S6-D）。
增强随机流与全局 RNG 解耦 + 逐 rank 分流（:65-92、trainer.py:146）：DDP 下各 rank 增强不同、同 seed 可复现（tests/test_augment_gpu_r5.py:45-57 已固化 bit-identical）。这是很多框架直接用全局 RNG 而在 DDP 下退化为"各卡同一批增强"的地方。
两个排序决策有据且写了理由：clamp 基准取增强前（避免 border 复制/dropout 污染基准，:113-118）、grid_dropout 必须在 clamp 之后（否则洞被 clamp 抬回 clamp_lo、dropout 静默失效，:161-162）。后者是只有踩过才会知道的坑。
gamma 只在空间轴 reduce、通道独立（:561-566）：为多分辨率/多模态通道留了正确语义，虽然 seg 在增强时恒为 C=1。
resize_3d 的两个防御（dataset.py:571-580）：mode='nearest' 防止 zoom 在边界注 0、输出形状对目标做防御性校正，堵住了 scipy zoom 的两个已知行为。
结构性问题

"几何幅度"没有单一真相源（S6-A/S6-G 的共同根因）。决定"一次增强会把内容推出边界多远"的量分散在四处：augment.* 的角度/平移/弹性幅度、data.aug_oversample_ratio、multi_res_scales 的 max_scale、以及 patch 的各向异性比 D:H:W。代码里只有 elastic_alpha /= max_scale 一条校正，校验器里只有 translate 一条（还算错了）。应当有一个 augment_geometry_budget(cfg) 派生函数，统一算出"各轴可用余量 / 各轴最大允许角度与平移"，供校验器与运行期共用——这与 S1 建议的 build_topology 单入口、S4 建议的"三个几何量做成 topology 侧派生函数"是同一手法。
越界（oob）语义选错了工具。当前是"label 填背景"（:403-404），等于把"我不知道这里是什么"编码成"这里是背景"。而框架已经有一个天然的忽略机制：weight_map 在 Dice 里是求和权重、在 BCE/Focal 里是归一化加权均值的权重（losses.py:124-130、:67-82），填 0 即精确排除。把 oob 掩码乘进 wmap（而不是改 label）是 5 行改动、语义严格、且与 nnU-Net 的 ignore-label 思路一致。当前实现之所以选了填背景，是因为 tests/test_review_batch4_fixes.py:117-131 把它固化成了契约——那批修复解决的是"越界处 border 复制出假前景"这个更糟的旧行为，方向对但停在了半路。
增强的成本模型不透明。整条管线在训练循环主流上同步执行（trainer.py:471），且作用于未裁剪、未拆视图的 max-FOV cube（体积 = 最终 patch 的 r³ 或 r·s³ 倍）。实测（B=2, (1,36,256,256), 输入 18 MiB）：


seg2_5d 默认(aff .3/flip .2)  10.94 ms   峰值 +306.0 MiB  (17× 输入)
affine only p=1.0              7.31 ms   峰值 +162.0 MiB  ( 9× 输入)
elastic only p=1.0             9.05 ms   峰值 +306.0 MiB
blur only p=1.0                2.16 ms   峰值 + 76.0 MiB
lowres only p=1.0              0.84 ms   峰值 + 55.5 MiB
峰值的主要来源是 grid（(n,D,H,W,3) fp32 = 输入的 3 倍）+ oob 掩码 + image[idx] 高级索引的 gather/scatter 各一份拷贝。日志里对此零披露，用户在 OOM 时不会想到是增强而不是模型。
"legacy 包装 + companion 实现"双份（轻微表最后第三条）与 S3-「cube 抽取两份近重复」、S5-H「多膨胀分支两套实现」同型：包装层已无生产消费者，却仍在维护且默认值已与主实现漂移。
augment 与 dataset 预处理的职责切分是历史形成的，不是设计的。归一化/面内 resize 在 CPU worker（scipy，单线程）、空间/强度增强在 GPU；结果是 z_axis 路径上同一份数据被插值两次（scipy zoom order=1 → grid_sample bilinear），既多一次数值损失也多一次算力。nnU-Net 的做法是把"抽取+缩放+仿射"合成一次重采样。本仓已经把 affine⊕elastic 合并了，只剩这一处没合。
配置注释质量远高于配置校验质量。AugConfig 的注释把 scale 的反向语义（core.py:242-244）、brightness/noise 隐含 [0,1] 量纲（:276-277）、z_axis 无面内余量（:254-255）、per-axis 旋转的 CT 惯例（:245-246）都写清楚了——然后默认值全都不遵守这些注释（默认三轴同角、默认 zscore 下用 minmax 量纲的幅值）。注释在这里承担了本该由默认值和校验器承担的职责。
④ 优化空间（含 GPU / 吞吐 / 显存）
#	优化	预估收益	风险
1	oob 掩码乘进 weight_map（而非改 label），并对无 wmap 的样本恒定输出全 1 wmap	直接消除 S6-A 的假阴性监督；顺带解掉 S2-A 的 KeyError: 'weight_map'（样本 schema 恒定）	中；改变损失分母，需与 S9 的 _weighted_voxel_mean 口径联合验证
2	6 个算子的索引/参数搬运统一加 non_blocking=True；grid_dropout 改在 CPU 上 nonzero	实测 6/10 个算子的隐式同步归零；开 prefetch_to_gpu 时预取收益才真正兑现	极低；数值完全不变
3	出面角度上界按几何自动收敛（asin(D_eff/max(H,W))），并让 _validate_augment 用统一的余量预算函数（含 max_scale 与"仅 z 有余量"）	解 S6-A / S6-G 的配置侧；把"默认值即次优"变成"默认值即几何合法"	中；会改变既有训练分布 → 需按"legacy 默认保旧"作新开关落地
4	仿射/弹性改为全 batch 一次 grid_sample（未选中样本用单位 theta），去掉 image[idx] 的 gather/scatter	省两份被选子集的拷贝；实测 affine 峰值 162 MiB 中约 1/3 属此	低；prob 很小时反而更贵，需按 n/B 阈值切换
5	z_axis 的面内 resize 从 CPU scipy 移到 GPU，与 affine 合成同一次 grid_sample	消除双重插值 + worker 主热点（S2-⑥ 已列为 #6）；数值更准	中高；改变数值，需回归比对（与 S2-优化 6 是同一件事，应同批做）
6	intensity_clamp 与 gamma 复用同一次 amin/amax	每 batch 省两次全量 reduce	极低
7	grid 与 oob 用 bf16/半精度存储（grid_sample 支持），或按需分块	峰值 306 MiB → 约 200 MiB 量级；大 patch 下更明显	中；grid_sample 在低精度下的坐标精度需实测（256 体素轴上 bf16 尾数不足，可能要 fp16 或分块 fp32）
8	blur 核长按逐样本 sigma 分组（而非统一取上界）	sigma=[0.5,5] 时小 sigma 样本从 ks=31 降到 ks=5，算力降一个量级	低；分组会引入若干次小 conv，需按组数权衡
9	flip 加 prob<=0 早退；三个 legacy 包装下沉到 tests 或删除	纯清理；顺带消除 3 次无谓的同步 H2D	极低
⑤ 2026 可借鉴项
方案	借鉴点	适配代价	优先级
nnU-Net v2 / batchgeneratorsv2 的 SpatialTransform（p_rotation / p_scaling 独立概率 + 逐轴旋转角 + 从更大的源区域一次重采样到目标 patch）	三点直击本轮问题：① 旋转角本来就是逐轴配置，不存在"三轴同角撞薄 slab"；② 抽取+缩放+仿射合成一次重采样（解 ④-5 的双重插值）；③ 越界用 padding + ignore 语义而非"填背景"（解 S6-A）	中（本仓已有 companion/单次 warp 的骨架，主要是把"抽取"也并进 warp）	最高
ignore-label / 有效性掩码作为一等公民（nnU-Net 的 ignore_label、MONAI 的 RandAffine + mask 传播）	本仓已具备 weight_map 这一通道（losses.py:67-82 确认它在 Dice/BCE/Focal 上都是精确的求和权重），把 oob 写进 wmap 即可，无需新机制	低	高
MONAI 1.x 的 Compose(lazy=True) 延迟重采样（多个空间变换先合成一个仿射/DDF，最后只 resample 一次）	本仓在 affine⊕elastic 上已经手工做到了；MONAI 的思路是把 flip / crop / resize 也一并纳入同一次重采样。本仓的 flip（torch.flip）+ center_crop（切片）+ 视图 resize（F.interpolate）目前是三次独立的内存搬运	中	中
torch.nn.functional.grid_sample 之外：torchvision.transforms.v2 / Kornia 的 batched 3D 仿射（本环境 torch 2.7.1 已具备 SDPA/flex 等新算子，但 3D 增强仍以自实现为主流）	主要借鉴其"变换参数逐样本 batched 化"的接口设计；本仓已是 batched，无需引入新依赖。结论：不建议引入第三方增强库（会破坏 Companion 抽象与零同步设计）	—	低（记录为"已评估、不采纳"）
per-case / 数据集指纹驱动的归一化（nnU-Net fingerprint_extractor 的前景强度分位数 + per-case zscore）	解 S6-H：make_data 已经在扫 median spacing（S3-⑤ 已提），顺带落 intensity 统计即可自动填 global_mean/std，并为 normalize=per_case_zscore 新枚举供数	低（同一遍扫描内）	高
BigAug / MedNeXt 的"强增强用于跨中心泛化"消融结论（2024–25 多中心分割共识：几何增强收益 > 强度增强，但强度增强对跨设备泛化不可省）	本仓强度增强齐全但幅度默认按 minmax 量纲写死（S6-H 关联），缺"按 σ 自动换算"的一层；可加 intensity_scale='auto' 让幅值随 normalize 模式自动折算	低	中
CopyPaste / CarveMix / 前景导向的 mix 类增强（医学分割 2022–25 的稳定增益方向，尤其小结构）	本仓完全没有样本间混合类增强；MixedBatchSampler 那层已有"batch 内配比"的抽象位置，CarveMix 需要 label 连通域信息，可由 make_data 的逐类 fg 索引供数（已有）	中高	中（血管/小结构任务上值得做消融）
各向异性感知的增强参数自动推导（nnU-Net 按 spacing 决定是否做 3D 旋转 / 是否 ignore_axes）	与 S5-E（各向异性核）、S6-A/E 是同一条主线：patch 的 D:H:W 与 spacing 应当驱动角度上界、弹性 sigma、lowres 轴选择三处	低（都是纯派生）	高
（说明：上述为方向性对标，若进入落地阶段涉及具体 API 时会再查各自最新官方文档，不凭记忆写实现。torch 2.7.1 / cuda 可用 为本机实测环境。）

⑥ 与既有测试 / 契约的冲突检查
S6-A（oob 语义）与既有测试直接冲突：tests/test_review_batch4_fixes.py:117-131 把"越界 label 必须是背景 / 自定义 label_fill"固化成了断言。改为"oob 写进 wmap、label 保 border 复制"会让这三个用例失效 → 必须作为独立批次评估：要么保留 oob_fill 同时新增 oob-wmap（两者并存、旧断言不破），要么以新开关切换并同步改断言。仅收紧角度上界（④-3）不触碰这些断言，可先行落地。
S6-C（RNG 入 checkpoint）：tests/test_augment_gpu_r5.py:45-57 只断言"同 seed 两个实例逐位一致"，tests/test_round2_fixes.py:887 (test_d3_rng_state_roundtrip) 覆盖的是全局 RNG 快照。把私有 Generator 状态加入 checkpoint 属纯新增，两者都不受影响；需新增"resume 后增强序列继续而非重放"的回归。
S6-D（non_blocking / CPU nonzero）：数值完全不变，test_companion_augment.py 的全部 legacy↔companion 等价性断言与 test_augment_gpu_r5.py 的 bit-identical 断言都不会失败。tests/test_todo1_batch2_fixes.py:134-144 断言 _grid_dropout_companions 用"输出 clone"而非全量掩码——把设备端 nonzero 改到 CPU 不触碰该断言。
S6-E（elastic gaussian 语义）：全仓无用例断言两种 field_mode 的幅度关系（grep 仅命中配置校验与字段定义）。若把 gaussian 改为"替换而非叠加"，会改变所有 elastic_field_mode=gaussian 历史配置的数值 → 按"legacy 默认保旧"原则，宜先只修文档口径 + 加一行"幅度会额外衰减约 2.4×"的说明，实现修正作为新枚举值。
S6-F（配置校验补全）：纯新增 fail-fast。风险点是 tests/test_review_batch1_fixes.py:196-199 特意用 ratio=8.0 断言"hd == 轴长时起点只能是 0 且不越界"——如果给 grid_dropout_ratio 加上界校验（≤1），该用例会被拒。两种解法：校验只在 Config 层做（函数层仍容忍），或同步放宽该用例。tests/test_segtask_v1.py:530-537 的 _grid_dropout 形状断言不受影响。
S6-B（whole 模式）：test_data_specs.py 覆盖的是 spec 选择与 split-dependent 参数切换，不涉及 whole 的 FOV 语义；无断言冲突。但改动会改变所有 whole + r>1 历史配置的训练分布 → 建议先加启动期 WARNING（"whole 模式的 oversample 会同时改变 FOV 与体素尺度，与 val 不一致"），下一版再改语义。
S6-G / S6-H：无既有测试覆盖，均为纯新增校验/警告。S6-H 若从 WARNING 升为 ERROR，会拒掉"显式想用未归一化 HU"的边缘用法，宜先 WARNING。
与 S1–S5 的一致性：
S6-F/S6-H 与 S1-「amp_dtype/compile_mode 配置期无枚举校验」、S2-G、S3-D、S4-C、S5-C 同属**"启动期缺前置校验、代价推迟到长耗时之后"族——这已是连续六轮**出现的最高频结构性问题，S12 应把它单列为一个跨模块落地批次（一次性补齐所有段的枚举/区间校验 + 一条"每个配置字段必须有校验器与消费者"的元测试）。
S6-A/S6-E/S6-G 与 S5-E（各向异性大核）、S1-「几何真相源 3 份 stem-stride 副本」同属**"各向异性只贯彻了一半 / 几何量多份真相源"**族，建议与 ④-3 的 augment_geometry_budget 一起并入"几何单一真相源"批次。
S6-C 与 S1-「推理侧无训推镜像契约校验」、S4-A（架构指纹）同属**"落盘状态不完整导致契约在跨运行边界失效"**族，可与架构指纹同批落地（都是往 ckpt 里补元数据）。
S6-A 的正解（oob → wmap=0）依赖 S2-A 的修法（dataset 恒定输出 wmap），两者必须同批，否则会在"部分样本有 wmap"的数据上触发 default_collate 的 KeyError。

S7	训练（一）引擎与工程件	engine/base_trainer.py、optim.py、amp.py、checkpoint.py、prefetch.py、dist_utils.py、memory.py、launch.py	S1
S8	训练（二）任务层编排	segtask_v1/trainer/trainer.py、pipelines/*、views.py、breakdown.py（含 2.5D 折叠时机契约验真）	S7、S4
S9	训练（三）损失 / 指标 / 验证与选模	losses.py、topo_aux.py、metrics.py、trainer/validation.py、_save_best 口径	S8
S10	推理全流程	engine/base_predictor.py、bn_stats.py；predictor/{predictor,sliding,forwards,inputs,io,blending}.py（滑窗/blend/TTA/AdaBN/z-interleave/写出）	S2、S6、S8
S11	全局串联	训练↔推理镜像契约端到端核对、配置→topology→数据→模型→损失→推理的一致性、性能瓶颈全局画像（数据/显存/通信/编译）、legacy 默认值清单再评估	S1–S10
S12	总结与优先级排序	汇总全部问题 + 2026 借鉴清单，按「正确性风险 / 收益-成本比 / 改造范围」排序，给出建议的落地批次（仅建议，不改代码）	S11
说明：S5、S7、S9 是最重的三块（各 2k–3k 行），如单轮内容过多我会主动拆成 a/b 两轮而非降低深度；计划需要调整时先说明原因。



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
