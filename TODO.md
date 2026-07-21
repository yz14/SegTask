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

审查计划（7 步，逐轮执行，四类并重：①正确性/缺陷 ②设计/架构 ③工程/性能/GPU ④2026 SOTA 借鉴）：
- S0 文档-代码契约基线 + 不一致清单（本节）
- S1 config（core/task_io/registry）
- S2 data（dataset/loader/specs/make_data/augment/sampling/patch_*/mixed_sampler）
- S3 models（topology/factory/unet 家族/骨干/stem/blocks/ADM/EDM2）
- S4 engine（base_trainer/predictor/amp/optim/checkpoint/dist/launch/prefetch/memory/bn_stats/views）
- S5 metrics/monitor/utils
- S6 全局串联 + SOTA roadmap

taskcore 全景：6 子系统 / 49 文件 / ~1.9 万行。config(~2.7k, core.py 2457) · data(~4.6k, dataset 1334/loader 1112) · models(~6.6k, blocks 1501/adm 1045/edm2 918) · engine(~2.9k, base_trainer 951) · monitor(~1.8k) · metrics+utils(~0.8k)。

=== S0：文档-代码契约基线 + 不一致清单 ===

【A. 已确立的核心契约（跨五任务，文档一致、由代码单点收敛）】
- C1 patch_mode 四模式统一口径：z_axis / 2_5d / cubic / whole；seg/cls/det/gen 全支持且抽取逐位一致，ssl 有意不支持 whole。收敛点：taskcore/data/specs.py（唯一按 patch_mode 分支处）+ patch_extract.py。
- C2 2.5D 折叠时机契约：dataset 恒发未折叠 3D（max-FOV × aug_oversample_ratio 余量）→ GPU 3D 增强 → 裁余量/拆视图 → 送模型前折叠（engine/views.py squeeze_2_5d）；唯一例外 det（折叠需联动 slice_boxes_to_2d，在 dataset 层完成）。
- C3 几何单一真相源：models/topology.py build_topology 派生 in_channels/out_classes/spatial_dims/n_views/aux 拓扑；config.sync、factory、各 predictor 都从此读取。
- C4 npz 数据契约：make_data/1.6；image(int16 HU)/label/可选 rw/fg_slices/fg_coords(+*_cls 类均衡)/meta(label_counts, spacing, bbox)。默认 ZIP_STORED 支持 mmap 快路径。
- C5 权重迁移契约：ssl 产出 encoder.*(+decoder.*)，下游 strict=False 命中；迁移要求 patch_mode/spatial_dims/in_channels 一致，不一致直接报错（validate_cls/validate_det）。
- C6 采样确定性：train 用 worker RNG + 前景过采样；val 用 (seed, idx) 确定性 RNG（VAL_SAMPLING_SEED），可选网格覆盖（z 等距 / Halton），与推理铺格同口径。
- C7 优化步时钟：scheduler/warmup/global_step/方法内调度均按真实 optimizer.step 边界推进；AMP 跳步时调度时钟推进但 EMA/center/queue/Gram 状态冻结（DDP all-reduce(any) 统一）。
- C8 训练-推理一致性：patch_size/patch_mode/multi_res/keep_native_*/z_boundary_mode/归一化/spacing_normalization 必须两侧一致。

【B. 已确认的不一致 / 文档过期（含证据）】
- I1 [Minor·文档过期] taskcore/__init__.py:5-7 docstring 称 engine「后续步骤将加入 BaseTrainer/BasePredictor」、config/data/models「按重构计划分步迁入」，但三者均已完成迁入（engine/base_trainer.py 等已存在）；且未列 monitor 层（顶层 README.md:24-31 已列 6 层含 monitor）。属陈旧描述，S 后续可提修订。
- I2 [Minor·跨项目命名] TTA flip 配置键不统一：seg=predict.tta_flip（taskcore/config/core.py:1026），cls/det/gen=tta_flips（clstask/config.py:144、dettask/config.py:119、gentask/config/dataclasses.py:132）；文档亦镜像此分裂（seg WORKFLOW §6 vs 其余 WORKFLOW）。功能一致，改名会破坏既有 YAML，价值低（与用户 TODO3 记录一致）。
- I3 [观察·非缺陷] losses 未进公共框架：taskcore 无 losses 层，各任务自带（segtask_v1/losses、gentask/losses/recon.py、clstask/losses、dettask/losses）。README.md:3 把 taskcore 概括为「config/data/models/训练推理工程件/通用工具」（5 项），line 24 又称「分六层」（含 monitor）——两处措辞口径略有出入。损失分散是 TODO3 重构的候选抽离点，非当前缺陷。
- I4 [已知待定·gen] gen WORKFLOW §5 自述：whole/z_axis/2_5d 训练侧 resize 到 patch、推理侧原生分辨率，两侧频谱不等价，修改涉及已训模型兼容未实施。属自声明 TODO（与用户 TODO3 记录一致），非文档-代码矛盾。

【C. 需在 S1–S5 深审中定点核验的强声明（本轮不展开，留作后续步骤检查项）】
- V1(S2) C2 折叠时机在五任务 trainer/dataset 的实际落点是否与文档逐一吻合（尤其 det 例外路径 slice_boxes_to_2d、cls 关增强时 dataset 侧折叠的等价性）。
- V2(S2) C6 采样确定性：val RNG 与推理网格铺点是否真正同口径（sampling.py vs 各 predictor 铺格）。
- V3(S1/S3) C3 topology 派生是否被所有下游真正复用、无重复推导（config.sync / factory / predictor / cls/det 迁移校验）。
- V4(S4) C7 优化步时钟 + AMP 跳步一致性在 base_trainer 的实现是否与五任务文档声明完全一致（grad-accum 尾组归一、DDP 跳步 all-reduce）。
- V5(S2) C1 四模式抽取「逐位一致」是否经代码验证（patch_extract vs seg dataset 三类），这是 encoder 权重跨任务迁移分布一致的地基。
- V6(S4) checkpoint 原子写 + resume 位精确恢复（含 RNG/optimizer relocate）是否覆盖五任务差异化的 ckpt 命名布局。

S0 小结：文档体系整体高度自洽，核心契约（C1–C8）在文档层交叉印证且指向明确的代码收敛点；仅发现 2 处轻微不一致（I1 文档过期、I2 命名分裂）与 2 处自声明待办（I3/I4），均非正确性缺陷。已为 S1–S5 建立 6 项定点核验清单（V1–V6）。下一轮进入 S1（config 子系统深审）。

=== S1：config 子系统深审（core.py 2457 / task_io.py 139 / registry.py 123）===

总体评价：config 是全仓质量最高的部分之一——派生量只读 property（单一真相源）、strict YAML 加载（拒未知/废弃键 + 迁移提示）、按 section 拆分的 validate、软/硬校验分层、海量精确的字段注释。以下问题按严重度排列。

① 正确性 / 缺陷
- S1-1 [Major] train.* 校验被错置进 _validate_predict，导致 cls/det/ssl 永不校验。core.py:2217-2229 的 train.ema_device / train.swa_start_ratio / train.zero_redundancy_optimizer 三项校验写在 _validate_predict() 内（def @2177）；而组合式任务经 registry._COMPOSITE_SKIP_CORE=("loss","predict")（registry.py:30）在 validate(skip={"loss","predict"}) 时**跳过整个 _validate_predict**（core.py:1343-1344）。这三个字段由 BaseTrainer 对所有任务生效（ema_device 全任务、swa/zero 多卡），却对 5 任务里的 3 个（cls/det/ssl）完全不校验：非法 ema_device='gpu'、swa_start_ratio=1.5、单卡开 zero 等在这些任务下静默通过。根因是"skip 粒度按校验器方法、字段归属按配置段"两者错配。修法：把这三项移入 _validate_train（对所有任务恒执行）。
- S1-2 [Minor] resenc_preset 大小写不一致。sync 侧 _apply_resenc_preset 用 .lower() 做模板查表（core.py:1301，可吃 "m"/"M"），validate 侧 _require(resenc_preset in ("none","S","M","L","XL")) 大小写敏感（core.py:1463-1465）。结果：小写 "m" 会先在 sync 里展开 encoder/decoder_blocks_per_stage（产生副作用），随后被 validate 硬拒。虽最终报错、无危害，但两处口径应统一（validate 改为大小写不敏感，或 sync 不 lower）。
- S1-3 [Minor→Major] _validate_data 的枚举/区间校验不全。已校验 patch_size 长度、patch_mode、z_boundary_mode、cache_dtype、multi_res、keep_native_*、target_spacing、aug_oversample≥1；但**未**校验多个被消费字段：data.normalize（应 ∈{minmax,zscore}，非法值下游 preprocess 可能静默走某分支——影响最大）、cache_mode（∈{none,memory}）、val_ratio（∈(0,1)）、foreground_oversample_ratio（∈[0,1]）、samples_per_volume（≥1）、batch_size（≥1）、num_workers（≥0）。建议至少补 normalize 枚举校验（fail-fast，防训推归一化口径静默错配，直接违反 C8）。
- S1-4 [Minor] 长度依赖 num_fg 的字段延迟到运行期才校验。loss.class_weights / loss.region_weights（core.py:569/574，注释称"长度=num_fg_classes"）与 predict.threshold 列表均未在 config 期校验长度；理由是 label_values 可能待数据扫描后确定（合理）。但当 label_values 已显式给出（num_classes 已知）时仍可提前校验，现未做——留作低价值增强。

② 设计 / 架构
- S1-5 [设计观察→已落地] ModelConfig 已按 arch 拆嵌套（见 D2）：公共字段 + `unet`/`adm`/`edm2`（及模块子段）；旧扁平接口保留兼容层。原 flat 设计的 YAML/override 短路径收益由兼容层承接。
- S1-6 [设计·正面] 派生只读量模式值得推广。spatial_dims/in_channels（540-548）、save_best_metric/mode（892-902）用 property 从 build_topology / _CRITERION_TO_METRIC 单点派生，并在 _DEPRECATED_DERIVED_KEYS（2336-2345）硬拒旧写接口——彻底消除"设了却被 sync 静默重写"。这是 config 层最佳实践，其它子系统（如 gen fork）应对齐。
- S1-7 [设计] validate 的 skip 机制与字段归属错配（S1-1 的根因层面）。skip 以"校验器方法名"为单位，但 train.* 字段跨越 predict 校验器；应确立"每个校验器只校验同名 section 字段"的不变式，避免跨段泄漏再被 skip 波及。

③ 工程 / 性能
- S1-8 [Nit] Config.per_view_depths（2295-2303）每次访问都 import + 重跑 build_topology；sync 已算过一次 topo。config 非热路径，影响可忽略；若下游在循环里读它可考虑缓存。总体 config 无 GPU/性能热点。

④ 2026 可借鉴
- S1-9 [借鉴·低优先] _dataclass_from_dict（2355-2394）手写了 strict 加载 + 别名/废弃/移除三类迁移提示，本质是 pydantic v2 / cattrs 的子集。鉴于全仓"依赖克制"原则（survey 明示），不建议引入重型依赖；但 S1-5 的按 arch 分组可借鉴 Hydra structured config / config groups 思路（结构表达条件有效性），作为纯设计参考。
- S1-10 [借鉴·正面确认] 选模预设 _SAVE_BEST_PRESETS（915-971）按解剖结构映射 (criterion, sd_tol, sd_w) 的做法，与 nnU-Net 式"指纹→配方"思路一致，属高质量设计，无需改动。

S1 小结：config 子系统整体优秀。1 个 Major（S1-1 train 校验对 cls/det/ssl 失效，建议修）、1 个可升级为 Major 的校验缺口（S1-3 normalize 等枚举未校验，直接关联训推一致性契约 C8）、若干 Minor 与设计观察。对应 S0 的 V3（topology 复用）：sync 确以 build_topology 单点派生 in_channels/spatial_dims 并经 property 暴露，config 侧复用正确，待 S3 核验 factory/predictor 是否同样只读该源。下一轮进入 S2（data 子系统深审）。

=== S2：data 子系统深审（dataset 1334 / loader 1112 / make_data 613 / augment 638 / specs 259 / mixed_sampler 216 / patch_dataset_base 148 / sampling 138 / patch_ops 109 / patch_extract 75）===

总体评价：data 是全仓工程成熟度最高的子系统。别名安全、memmap 零拷贝快路径、bbox 流式裁剪、确定性验证采样、类均衡前景索引、DDP worker 平摊、伴随张量增强——细节扎实。发现按严重度排列。

① 正确性 / 缺陷
- S2-1 [Major·限 --override 分支] make_data.py:594 `from ..train import apply_overrides` 指向不存在的 `taskcore.train`（taskcore 无 train.py；apply_overrides 未定义于 taskcore，仅 task_io 有 apply_dotted_overrides）。`python -m taskcore.data.make_data --config ... --override ...` 会 ModuleNotFoundError 崩溃；不带 --override 正常。修法：改 `from ..config.task_io import apply_dotted_overrides` 并调用 `apply_dotted_overrides(cfg, args.override)`（该函数签名兼容 seg/gen 单段），随后 sync/validate。已用 Glob+Grep 双重确认 taskcore.train 不存在。

② 设计 / 架构
- S2-2 [正面·关键] C1/V5「四模式逐位一致」由共享纯函数落实：seg dataset 三类与 cls/det/ssl/predictor 用的 patch_extract 都调用同一 extract_z_patch_padded（dataset.py:1022）/ extract_cubic_patch（patch_ops.py:25）/ resize_3d（dataset.py:544）。单分辨率+无过采样时抽取几何逐位一致——V5 在 data 层成立，是跨任务 encoder 权重迁移分布一致（C5）的地基。specs.build_data_spec 为 data 侧唯一 patch_mode 分支点，收敛干净。
- S2-3 [正面] 别名安全全链路一致：抽取路径末尾无条件 copy，whole 的 resize 恒等返回也显式 copy（dataset.py:1304-1307/1319-1320），杜绝 worker LRU 缓存卷被下游 in-place 污染；VolumeCache.__getstate__ 丢缓存避免 Windows spawn 管道超限。
- S2-4 [Minor·默认漂移] SegDataset3D 直接构造默认 intensity_max=3071.0 / z_boundary_mode="stretch"（dataset.py:826/837），与 Config 默认（1024.0 / edge_pad）不一致。经 specs 走 Config 时被覆盖无运行期影响，但直接实例化/写测试易踩坑。建议对齐默认或注释「仅经 Config 路径」。

③ 工程 / 性能 / GPU
- S2-5 [Major·性能/GPU] in-plane resize 走 CPU scipy.ndimage.zoom（resize_3d）：z_axis/2_5d 每次 __getitem__ 对全 H/W 一次 zoom、whole 对整卷 zoom，是 DataLoader worker 的 CPU 热点（仓内 tools/bench_zoom_o8.py 佐证团队已在意）。与「dataset 只发 max-FOV、几何延后到 GPU」契约不完全对齐（z 多分辨率已延后，面内 resize 仍在 CPU）。优化方向：面内 resize 也延后到 GPU（trainer F.interpolate，label nearest），或换更快 CPU resize；属训练吞吐优化，需 GPU 环境实测（对齐用户 TODO3 的 GPU 复核项）。
- S2-6 [正面] IO/内存工程到位：未压缩 npz 零拷贝 memmap 快路径（_open_npy_member_mmap，跨 worker 共享 page cache）、bbox 流式裁剪读取、NIfTI 读重试 + host-OOM 折 MemoryError、免解码读 image_shape/label_counts（meta 快路，启动期不解码 label）、cache_int16 半内存、DDP worker 平摊 + 缓存足迹估计告警。
- S2-7 [正面] 混采 DDP 正确性（对应 C 类关注）：MixedBatchSampler 对每个全局 batch 都消费 RNG（shuffle+gold_stream）后再按 rank 过滤（mixed_sampler.py:200-215），各 rank 同 seed+epoch 得一致全局序列、不相交切分、等长——DDP 对齐无误。

④ 2026 可借鉴
- S2-8 [借鉴·低优先] make_data target_spacing 取逐轴中位数（make_data.py:360）；nnU-Net v2 对强各向异性数据在最低分辨率轴改用较激进百分位（防薄层过拉伸）。可作为 spacing_normalization 增强选项（功能增强，非缺陷）。
- S2-9 [借鉴·确认] GPUAugmentor 已高水准（伴随张量 spec 化、affine+elastic 融合单次 grid_sample、CPU Bernoulli 免 CUDA 同步、各向异性 aspect 校正、intensity_clamp 记录增强前范围）。可选扩展：CT mask-aware CutMix / RandGaussianSharpen；当前已覆盖主流强增强。此模块正是用户 TODO3「augment 合流」的目标底座，设计已支持伴随张量泛化。

S2 小结：data 子系统整体优秀，仅 1 个 Major 正确性缺陷（S2-1 make_data --override 坏导入，易修）+ 1 个 Major 性能项（S2-5 CPU zoom 热点，需 GPU 实测）+ 若干正面确认与 Minor。核验结果：V5（四模式逐位一致）成立于共享纯函数层；V2（val 采样与推理同口径）在 sampling.py 共享原语层成立，最终与 predictor 的对齐待 seg 项目审查（TODO2）核验；C2 折叠时机在 data 层「dataset 恒发未折叠 3D」已确认（det 例外在 dettask 层）。下一轮进入 S3（models 子系统深审）。

=== S3：models 子系统深审（blocks 1501 / adm 1045 / edm2 918 / factory 624 / unet 611 / mednext 540 / resnet 505 / stem 337 / topology 169 / convnext 173 / unetpp 128 / unet3p 121）===

覆盖：本轮全文精读 10/13 文件（topology/factory/unet/stem/resnet/blocks/convnext/unetpp/unet3p/adm_unet，含最大的 blocks.py 与 adm_unet.py）；edm2_unet 与 mednext 以 S 规划期结构图 + 接口一致性核验（未逐行，留作可选补读）。

总体评价：models 是全仓设计最现代、最克制的子系统之一。无发现正确性缺陷；防御式编程到位（形状不匹配 RuntimeError、ADM skip-stack 平衡断言、通道整除校验）。

① 正确性：本轮未发现 models 层的正确性 bug。抽取/装配/forward 契约均严谨。

② 设计 / 架构
- S3-1 [正面·V3 确认] topology 单一真相源被全 arch 复用：factory `_build_unet_encoder_decoder`/`build_model`（factory.py:336/560）、adm `build_adm_seg_model`（adm_unet.py:759）、edm2 均读 build_topology，不再自行推导 in_channels/out_classes/spatial_dims。R5 重构目标达成——V3 成立（S1 config 侧 + S3 model 侧双向确认）。
- S3-2 [设计观察·维护成本点] 双代码路径分叉（通用 UNet vs 论文忠实 ADM/EDM2）。ADM/EDM2 刻意忽略大量 model.* 通用旋钮（backbone/block_type/down·upsample_mode/attention_type/skip_mode/decoder_type/anisotropic/multirf/selfattn），改用论文固定结构（GN32+SiLU / MP）。是有意取舍（保真扩散配方），但：(a) adm/edm2 下用户设的多数 model 字段被**静默**忽略（仅 decoder_blocks_per_stage 有 warning，adm_unet.py:780）；(b) 两套 forward 契约需各自与 UNet3D 对齐维护。建议：adm/edm2 装配时对"被忽略的非默认 model 字段"统一 warning（提升可发现性，零功能风险）。
- S3-3 [设计·正面] 扩散孪生共享 enc/mid/dec，`_make_resblock` 按 emb_channels 开关 AdaGN（adm_unet.py:188）——seg 头与 diffusion 头共用骨干、参数命名对齐、权重可迁移；forward 契约（Tensor / [main,DS...] / {main,aux,topo} dict）与 UNet3D 严格一致（S3 直接比对 unet.py:597 与 adm_unet.py:730）。高质量。
- S3-4 [设计观察·能力边界] ADM/EDM2 仅 2.5D、不支持 hierarchical stem、decoder 块数被 skip-stack 拓扑锁定；均显式 raise/warning、无静默失败。属已知边界（文档 gap 已记），非缺陷。

③ 工程 / 性能 / GPU
- S3-5 [正面] 梯度检查点全 arch 覆盖（UNet 逐 stage/level、ADM/EDM2 逐块），checkpoint_if 用 use_reentrant=False + preserve_rng_state（DropPath 可复现）、eval 零开销数值一致（blocks.py:49）。AMP 数值安全极严谨：GRN / LayerNorm3d / GroupNorm32 fp32 统计、插值前 fp16→fp32、softmax-attn fp32、SDPA fused backend、attn/proj/out zero-init 残差。现代且稳。
- S3-6 [Minor·DRY] factory.build_model 为日志重算 enc_counts/dec_counts/expected_dec_calls（factory.py:566-580），与 _build_unet_encoder_decoder 内部逻辑重复；无功能影响，可让 encoder/decoder 暴露计数或抽共享。

④ 2026 可借鉴（本子系统 SOTA 讨论重点）
- S3-7 [算子库·已很全] blocks.py 已覆盖 2022-2023 主流：BlurPool/PixelShuffle/CARAFE/DySample 上采样、SE/ECA/CBAM/Coord/LKA(VAN)/MSCA(SegNeXt) 注意力、softmax/linear/window/grid 自注意力 + RoPE + GEGLU-FFN、ConvNeXt-V2 GRN。已是先进算子库，无需大改。
- S3-8 [能力空白·分割侧无 3D ViT/混合骨干] seg encoder 仅 CNN（resnet/convnext/mednext）；cls 有 ViT 但 seg/SSL encoder 无。若要吃 MAE/DINO 类 ViT SSL 权重或上 SwinUNETR/UNETR 类 3D transformer 混合骨干，seg 侧缺 ViT/hybrid encoder 是能力空白（SSL.md 亦以 CNN 骨干为固定项）。属新增范式的战略选择，供 TODO3/后续决策。
- S3-9 [生成侧范式] 当前 ADM/EDM2(扩散) + 回归。2026 可评估 **flow matching / rectified flow**（SD3/Flux，训练更简单、少步采样）与 **DiT/U-ViT** 扩散骨干备选（gen survey 已列 DiT）；EDM2 已是 2024 SOTA 扩散配方，建议 GPU 环境跑短程扩散确认显存/收益（与用户 TODO3 note 一致）。
- S3-10 [探索性] Mamba/SSM 类（U-Mamba/VMamba）在 3D 长序列医学分割 2024-2025 有进展，可作 selfattn 之外的线性复杂度全局建模备选（探索性，非必需）。

S3 小结：models 无正确性缺陷，设计现代克制、V3 全 arch 确认、扩散孪生与梯度检查点/AMP 工程一流。主要为设计观察（S3-2 双路径静默忽略字段建议补 warning）与 SOTA 战略项（S3-8 分割侧无 3D ViT/混合骨干、S3-9 生成 flow-matching/DiT）。edm2_unet/mednext 建议按需补逐行读。下一轮进入 S4（engine 子系统深审）。

=== S4：engine 子系统深审（base_trainer 951 / checkpoint 351 / optim 340 / dist_utils 155 / launch 144 / bn_stats 109 / amp 108 / prefetch 92 / views 91 / memory 73 / base_predictor 65）===

覆盖：全部 11 文件逐行精读。

总体评价：engine 是全仓工程最硬核、最成熟的子系统。**无正确性缺陷**；DDP/AMP/checkpoint/资源治理的边界情况处理极其到位，且对"数学等价性"的界限诚实标注。

① 正确性：未发现 bug。V4/V6 两项 S0 核验点在此确认（见下）。

② 设计 / 架构
- S4-1 [正面·V4 确认] 优化步时钟 + AMP 跳步一致性正确：`_optimizer_step_boundary`（base_trainer.py:281）——fp16 由 GradScaler 内部跳步、bf16/fp32 用 `all_reduce_flag_any`（dist_utils.py:73）跨 rank 一致跳步；scheduler 默认仅真正更新后推进、ssl `always_step_scheduler` 边界恒推进；EMA 仅真正更新后推进；`_effective_accum`（:221）尾批取真实尾长作分母。与五任务 WORKFLOW 声明的"优化步时钟/跳步冻结状态"完全吻合。
- S4-2 [正面·V6 确认] checkpoint 原子写 + 位精确 resume：`atomic_torch_save`（tmp+os.replace，失败清理，checkpoint.py:26）；`snapshot_rng_state`/`restore_rng_state`（torch CPU+CUDA+numpy+python，async 走 bytes 打包避免 ByteTensor 降级，:326/:88）；`relocate_optimizer_state`（resume 后 fused/ZeRO 状态搬回参数设备，:126）；`_restore_train_state`（model/EMA/optim/sched/scaler/SWA/best/RNG 全量，base_trainer.py:525）；AsyncCheckpointSaver 错误经 wait() 重抛不静默丢盘。best/latest 双布局 + 任务 `_ckpt_extra_state` 差异化——V6 成立。
- S4-3 [设计观察·维护成本点] "显式装配"模板：BaseTrainer 不吞并训练循环，各任务 `_train_epoch` 自行按序调 helpers。step-boundary/跳步/accum 逻辑虽集中于 helper，但**是否被各任务正确调用**取决于任务层——V4 的跨任务一致性最终需在 TODO2（seg）及各任务审查时逐一核对 helper 调用点。属与 S3-2 同性质的"灵活性 vs 强制性"取舍，非缺陷。
- S4-4 [Minor·文档过期] checkpoint.py:3-6 docstring 仍称"主流程方法 _build_state_dict/_save_checkpoint/_load_checkpoint/_load_pretrain 保留在 Trainer 类上，便于 inspect.getsource 测试校验 token"——但公共保存/恢复现已下沉到 BaseTrainer（_save_best/_save_latest/_restore_train_state/_load_pretrain_weights）。描述陈旧，建议更新。

③ 工程 / 性能 / GPU（本子系统亮点密集）
- S4-5 [正面] 数值/同步优化到位：健康范数用 `torch._foreach_norm` 批量 + 末尾单次 `.item()`（避免逐参数 D2H 打断流水，base_trainer.py:385）；`grad_norm_lazy_sync` 在 fp16+无健康监测时跳过 D2H；EMA CPU offload（省 1× 参数显存）；ZeRO-1 分片 + 保存前 consolidate；channels_last / torch.compile（Triton 缺失回退 eager）；梯度检查点已在 models 层覆盖。
- S4-6 [正面] 资源/健壮性治理：`maybe_enable_expandable_segments`（碎片治理，首次分配前注入、不覆盖用户设置）；`PR_SET_PDEATHSIG` 孤儿进程兜底 + SIGTERM/SIGINT 处理 + NCCL watchdog 超时（launch.py，防 DDP 卡死占卡）；`find_free_port` 防端口冲突。
- S4-7 [正面] CudaPrefetcher 流语义正确（copy stream + wait_stream + record_stream，防跨流显存过早回收）；AdaBN/SWA-BN 用 momentum=None 累积平均 + DDP 加权聚合（bn_stats.py + all_reduce_bn_running_stats_）与单进程全集严格相等。
- S4-8 [正面·诚实标注] DDP 数学等价边界明确写入 `_setup_ddp` 日志与 TrainConfig 注释：batch 池化比值损失（batch_dice/Tversky/GDL）在 grad-accum×多卡下统计窗口收缩为单 micro-batch，属近似而非严格等价。这是真实的数值局限，已对用户透明，非缺陷。
- S4-9 [Minor] scheduler resume horizon 漂移：仅 OneCycleLR 有 `_reconcile_one_cycle_horizon`（optim.py:280）；cosine/poly 在 resume 时若 epochs/数据量/accum 变化导致 horizon 变，用新 T_max + 恢复的 last_epoch 可能给出轻微偏移的 LR（正常同配置 resume 无影响）。可选：对 cosine/poly 也做类似折算或告警。

④ 2026 可借鉴
- S4-10 [借鉴·低优先] 当前大模型并行为 DDP + ZeRO-1（优化器状态分片）。若未来上超大 3D 模型，可评估 **FSDP / ZeRO-2/3**（梯度+参数分片）与 **选择性激活重计算（SAC）**；现有 grad_checkpointing 已是 full checkpointing，SAC 可在显存/算力间取更优点。属规模化时的选项，当前无需。engine 现代性已足够，此项优先级低。

S4 小结：engine 无正确性缺陷、工程一流。V4（优化步时钟+AMP跳步+DDP一致）与 V6（原子写+位精确resume）双双确认成立。主要产出为正面确认 + 2 个 Minor（S4-4 文档过期、S4-9 非 OneCycle 的 resume horizon 漂移）+ 1 个跨任务维护观察（S4-3，留待 TODO2 核对 helper 调用点）+ 1 个规模化 SOTA 备选（S4-10 FSDP/SAC）。下一轮进入 S5（metrics + monitor + utils）。

=== S5：metrics + utils + monitor 深审（monitor assets 617/charts 509/history 377/dashboard 124/__main__ 112 · utils/common 313/logging 193 · metrics 302）===

覆盖：精读 metrics.py、utils/common.py、utils/logging_utils.py、monitor/history.py（数据层）；monitor 渲染层（charts/dashboard/assets/__main__）为 fail-isolated HTML，以结构图覆盖、未逐行（不影响训练正确性）。

总体评价：metrics 数学正确且全部可 all-reduce；EMA/SWA 实现细腻；monitor 崩溃安全、失败隔离。

① 正确性
- S5-1 [Nit] metrics.py:268 `surface_dice_batch_stats` 签名用 `Optional[...]`，但文件顶部仅 `from typing import Dict, List, Union`（:10）未 import Optional。靠 `from __future__ import annotations`（:8，PEP 563 惰性注解）兜住，运行期不报错；但 `typing.get_type_hints()` 等内省会 NameError。修法：imports 补 Optional（一行）。

② 设计 / 架构
- S5-2 [正面·关联 C/V] 指标 pooled 设计天然可 all-reduce：dice_batch_stats/surface_dice_batch_stats 返回 inter/pred_sum/target_sum/voxels/sd_num/sd_denom/n_with_gt 等"跨样本可加"量，derive_overlap_metrics 闭式导出 dice/iou/recall/precision/vol_sim/mcc——各 rank 分片求和后与单进程全集严格相等（配 dist_utils.all_reduce_sum_）。voxels 用 float64 累加防大体素 fp32 精度上限。这是 S4 DDP 验证聚合正确性的数学地基。
- S5-3 [正面] monitor 数据/渲染分层解耦、全程 fail-isolated（BaseTrainer 已 try/except 包裹调用）；jsonl 每 epoch 原子全量重写（tmp+fsync+os.replace，续训/崩溃不产生重复行/半行）+ 按 epoch 去重；EpochRecord 只留有限标量（_finite_scalars 丢 NaN/Inf）。metrics/utils 分层清晰（metrics 数学、common 工具、re-export 保旧导入路径）。

③ 工程 / 性能 / GPU
- S5-4 [Major·性能，opt-in] NSD spacing-aware（_nsd_stats_spacing_aware，metrics.py:192）用 scipy EDT 在 B×C Python 双层循环逐样本 CPU 计算 + GPU→CPU 拷贝：大 val 集 × 多类 × high 整卷验证下开销显著（每卷每类两次 distance_transform_edt）。仅 surface_dice_tolerance_mm>0 启用（voxel-Chebyshev 路径为 GPU maxpool，快）。已文档标注 CPU/scipy。优化方向：GPU 近似 EDT / 限 NSD 评估卷数 / 仅对 best 候选算 NSD。需 GPU 实测权衡（对齐用户 TODO3 GPU 复核）。
- S5-5 [正面] surface_dice voxel 路径：τ≥2 用可分离 maxpool（k^d→d·k，与全核严格等价），τ=1 单核；erosion/dilation zero-pad 边界口径一致。EMA update foreach 批量 + 按 (device,dtype) 分组 + CPU offload pinned staging 单次流同步；SWA CPU fp32 增量平均（avg += (w-avg)/n）。seed_everything 的 TF32/deterministic 分档合理。

④ 2026 可借鉴
- S5-6 [借鉴·低优先] 指标已覆盖 dice/iou/recall/precision/vol_sim/mcc/surface-dice/spacing-aware NSD + 调和均值综合选模，超出多数框架。ssl eval 另有 spacing-aware HD95；seg 主选模未纳入 HD95（边界质量已由 NSD 覆盖，价值重叠）。当前无实质缺口。

S5 小结：metrics 数学正确且 DDP 可归约（印证 S4 验证聚合地基）、EMA/SWA 细腻、monitor 崩溃安全失败隔离。仅 1 个 Nit（S5-1 Optional 漏 import，PEP563 兜住）+ 1 个 opt-in 性能项（S5-4 NSD CPU EDT，需 GPU 实测）。至此 S1–S5 五子系统深审完成。下一轮进入 S6（全局串联 + SOTA roadmap 收尾）。

=== S6：全局串联 + 最终结论 + SOTA roadmap ===

【总体判断】taskcore 是一套设计现代、工程成熟、克制自洽的公共框架。契约以代码单点收敛（specs / topology / views / metrics），AMP/DDP/checkpoint/采样确定性等边界情况处理到位，且对"数学等价性边界"诚实标注。全面深审仅发现 1 个确定的功能缺陷（S2-1）+ 1 个校验覆盖缺口（S1-1/S1-3）+ 2 个需 GPU 实测的性能项（S2-5/S5-4），其余为 Minor/文档/设计观察。整体质量高于绝大多数科研代码库。

【V1–V6 定点核验结论】
- V3（topology 单一真相源被复用）：✅ 确认（S1 config 侧 + S3 factory/adm/edm2 侧双向）。
- V4（优化步时钟 + AMP 跳步 + DDP 一致）：✅ engine helper 层确认（S4）；各任务 helper 调用点的最终一致性留 TODO2 核对。
- V5（四 patch_mode 逐位一致）：✅ 确认（S2，seg dataset 与 cls/det/ssl/predictor 共享同一批纯函数）。
- V6（checkpoint 原子写 + 位精确 resume）：✅ 确认（S4）。
- V1（2.5D 折叠时机）：◑ data 层"dataset 恒发未折叠 3D"已确认；trainer/predictor 折叠落点与 det 例外（slice_boxes_to_2d）在任务层，留 TODO2。
- V2（val 采样与推理同口径）：◑ sampling.py 共享原语层确认；与各 predictor 铺格的最终对齐留 TODO2。

【五任务通用性评估】
- 已通用（taskcore 承载）：config 公共段 + 校验、四 patch_mode 数据读取与抽取、GPUAugmentor（伴随张量 spec 化）、UNet 家族 + ADM/EDM2 骨干 + topology、BaseTrainer/BasePredictor + AMP/optim/checkpoint/DDP/prefetch/bn_stats/views、metrics、monitor。抽象层次合理，cls/det/gen/ssl 复用充分。
- 未进公共层（各任务自持，属 TODO3 抽离候选，非缺陷）：losses（seg/gen/cls/det 各一份）、各任务 trainer 主循环、predictor 主流程。其中 losses 与"folding/几何"解耦度高，是 TODO3 最可行的下一个抽离点。
- 已知能力边界（文档已记）：ADM/EDM2 仅 2.5D、不支持 hierarchical stem；ssl 不支持 whole、未接 CudaPrefetcher（用户 TODO3 已记）；分割侧无 3D ViT/混合骨干。

【最终发现清单（按优先级）】
建议修（低风险、明确收益）：
- P1 S2-1 [Major] make_data.py:594 `from ..train import apply_overrides` → 改 `taskcore.config.task_io.apply_dotted_overrides`（--override 分支崩溃，一处修）。
- P1 S1-1 [Major] 把 train.ema_device/swa_start_ratio/zero_redundancy_optimizer 校验从 _validate_predict 移入 _validate_train（现对 cls/det/ssl 完全不校验）。
- P2 S1-3 [Minor→Major] _validate_data 补 normalize∈{minmax,zscore} 等枚举校验（直接护住训推一致性契约 C8）。
- P3 Minor 批量：S1-2 resenc_preset 大小写口径统一、S2-4 SegDataset 默认对齐 Config、S3-6 factory 日志重算 DRY、S4-4 checkpoint docstring 更新、S4-9 cosine/poly resume horizon 折算、S5-1 metrics 补 Optional import、I1 taskcore/__init__ docstring 更新、I2 tta_flip 命名（改动破坏既有 YAML，倾向不改）。

需 GPU 环境实测再决策（性能）：
- G1 S2-5 面内 resize 的 CPU scipy.zoom 热点是否延后到 GPU（训练吞吐）。
- G2 S5-4 NSD spacing-aware 的 CPU EDT 开销（大 val×多类×high）。
- G3 EDM2/ADM 短程扩散跑通确认显存/收益（用户 TODO3 已列）。

需你决策（设计/产品，非缺陷）：
- D1 S3-2 ✅ 已落地：新增 taskcore/models/arch_compat.py::warn_ignored_model_fields——对「被忽略且非默认值」的通用 UNet 旋钮（backbone/norm/act/上下采样/skip/attention/convnext·mednext 组等 25 项）+ 对侧扩散家族专属组（adm 下的 edm2_*、反之）发一条汇总 warning；4 个装配入口（build_adm_seg_model / build_edm2_seg_model / build_adm_diffusion_unet / build_edm2_diffusion_unet）统一接入。已被 validate 拒绝的（aux_topo_head/lift/selfattn/multirf）、已有专项 warning 的（decoder_blocks_per_stage、edm2 stem/aux_head）、经 sync 间接生效的（resenc_preset）与被 gate 控制的子字段不重复告警。验证：现有 adm/edm2/gen-adm YAML 零误报；新增回归 test_adm_edm2_warn_on_ignored_non_default_model_fields；adm·edm2 smoke + gen smoke + grad_ckpt 共 69 passed。
- D2 S1-5 ✅ 已落地：ModelConfig 按 arch 拆嵌套（公共字段 + `unet`/`adm`/`edm2`；unet 内再嵌 `mednext`/`multirf`/`selfattn`；adm 内嵌 `linear_attention`；gentask 另嵌 `sisr`）。兼容层保留旧扁平 YAML/CLI override/Python 属性/checkpoint pickle；`save_config` 与仓内 YAML 统一输出新嵌套格式。生产代码读点已迁嵌套路径，AST 守门防回流。验证：`test_d2_migration_contract` + 全量 pytest 通过。
- D3 S4-3 ✅ 采用「保留显式装配 + 机器护栏」，不让 BaseTrainer 吞并各任务训练循环。
  · 第一阶段：删除 SSL 重复的 `_effective_accum`；五任务 AST 守门（必须调用 `_optimizer_step_boundary`、读取 `skipped_nonfinite`、禁止复制累积算法）。
  · 第二阶段：`OptimStepResult` 标志互斥断言 + `acknowledge()` 门闩（下次进入边界前必须 ack，强制「看见结果」但不强制 continue）；边界内校验 scheduler 推进次数——默认路径仅 `stepped` 时 +1，`always_step_scheduler=True`（SSL）每次边界恰好 +1。五任务调用点均已 ack。验证：test_todo_p_regressions + review_batch1 + cls/det smoke + ssltask + swa_lka 共 295 passed。

【2026 SOTA borrow roadmap（按价值×可行性排序）】
1. [战略·高价值] 分割/SSL 侧 3D ViT 或混合骨干（SwinUNETR/UNETR/hybrid）——补齐吃 MAE/DINO 类 ViT-SSL 权重的能力空白（当前 SSL.md 固定 CNN 骨干、seg encoder 无 ViT）。影响 seg+ssl+cls 三任务，是最有战略意义的新增。
2. [高可行] augment 合流（用户 TODO3 已设计）——GPUAugmentor 的伴随张量 spec 化已是现成底座（S2-9），五任务共用一份、gen cond 作 bilinear companion 接入。需 GPU 固定 seed 等价性验证。
3. [中·生成] 生成侧引入 flow matching / rectified flow（SD3/Flux 范式，训练简单少步采样）与 DiT/U-ViT 扩散骨干备选；EDM2 已是 2024 SOTA 配方，先跑通确认。
4. [中·数据] make_data target_spacing 各向异性百分位（nnU-Net v2）作为 spacing_normalization 增强选项。
5. [探索] Mamba/SSM（U-Mamba/VMamba）作 selfattn 之外的线性复杂度全局建模；规模化时 FSDP/ZeRO-2-3/SAC。

【TODO 1 结论】公共框架审查完成，结论：框架健康、可放心作为五任务基石继续演进；建议先落地 P1（2 处明确缺陷，低风险）与 P2（normalize 校验），P3 择机批量清理；G1/G2/G3 待 Windows GPU 环境实测；SOTA roadmap 第 1、2 项建议作为后续重点。衔接：V1/V2/V4 的任务层落点核验 → TODO2（分割项目审查）；losses 抽离与 augment 合流 → TODO3（重构）。TODO 1 收尾。

【实施进展】（计划 E1→E5：P1 两缺陷 → P2 校验 → P3 Minor → 回归自查）
- E1 ✅ S2-1 已修：make_data.py --override 分支改为 `from ..config.task_io import apply_dotted_overrides`（sections=None 单段路由，签名兼容），保留 sync/validate。验证：torch27_env 实跑 import+override+sync/validate 通过；tests/test_task_config_io.py + test_data_specs.py 24 passed。
- E2 ✅ S1-1 已修：core.py 中 train.ema_device/swa_start_ratio/zero_redundancy_optimizer 三项校验从 _validate_predict 移入 _validate_train 末尾（逻辑逐字保留；zero 检查复用局部 gpus）。验证：skip={'loss','predict'} 路径现可拦下非法 ema_device='gpu'、swa_start_ratio=1.5，合法默认与单卡 zero warning 行为不变；新增回归 tests/test_todo_p_regressions.py::test_train_validators_run_when_predict_skipped；回归 test_task_config_io+test_swa_lka+test_resenc+test_clstask_smoke+test_dettask_smoke 共 72 passed。
- E3 ✅ S1-3 已修：_validate_data 补 7 项校验——normalize∈{minmax,zscore}、cache_mode∈{none,memory}、val_ratio∈(0,1)（split 侧会把 0 钳成至少 1 个 val 样本，语义无效故拒绝）、foreground_oversample_ratio∈[0,1]、samples_per_volume≥1、batch_size≥1、num_workers≥0。验证：8 组非法值全 fail-fast、合法边界值放行；configs/ 全部 38 个 YAML 以各自任务 loader 重载全通过（零误伤）；新增回归 test_validate_data_rejects_invalid_enums_and_ranges；相关套件 55 passed。
- E4 ✅ P3 必做批量完成（S3-6 factory DRY / S4-9 cosine·poly horizon 本轮 defer，仍属选做）：
  · S5-1 metrics.py 补 Optional import（typing.get_type_hints 可解析 surface_dice_batch_stats）。
  · I1 taskcore/__init__.py docstring 改为已落地的六层描述（含 BaseTrainer/BasePredictor/monitor）。
  · S4-4 checkpoint.py docstring 改为「公共保存/恢复在 BaseTrainer，任务 Trainer 保留薄封装供 inspect」。
  · S1-2 resenc_preset validate 改为 .lower() 枚举，与 sync 查表口径一致（"m"/"M" 均可）。
  · S2-4 SegDataset3D/Cubic/Whole 的 intensity_max 默认 3071→1024、SegDataset3D.z_boundary_mode 默认 stretch→edge_pad，对齐 Config.data；经 specs 路径仍以 Config 为准。
  · 新增回归 test_resenc_preset_accepts_lowercase / test_segdataset_defaults_align_with_config。
  · 回归 test_todo_p_regressions+test_resenc+test_z_boundary_mode+test_data_specs+test_patch_dataset_base 共 72 passed。
- E5 ✅ 自查收尾：
  · 相关回归套件 12 文件 154 passed（config/data/dataset/resenc/z_boundary/swa/cls·det smoke/round2 checkpoint 等）。
  · 源码扫描：无残留 `from ..train import`、train 校验已在 _validate_train、Dataset 默认已对齐 Config。
  · E1 override 路径 + configs/ 38 YAML 重载抽查通过。
  · **P1+P2+P3(必做) 全部落地**；defer 项：S3-6 factory DRY、S4-9 cosine/poly resume horizon、G1–G3 GPU 实测、I2 tta 改名。
- D3 第一阶段 ✅：SSL 删除逐字复制的 `_effective_accum`，统一继承 BaseTrainer；新增 `test_all_task_trainers_use_shared_optimizer_boundary_contract`，以 AST 检查五任务均调用公共 optimizer boundary、处理 skipped_nonfinite，且不覆盖共享累积算法。验证：test_todo_p_regressions 11 passed；test_ssltask 244 passed。
- D3 第二阶段 ✅：OptimStepResult 标志互斥 + acknowledge 门闩 + 边界内 scheduler 时钟校验（默认 vs always_step_scheduler 两套语义）；五任务 `_train_epoch` 均调用 `result.acknowledge()`；AST 守门同步要求 acknowledge。验证相关套件 295 passed。
- D1 ✅ 见上「需你决策」节。
- D2 ✅ 见上「需你决策」节；分步：D2-1 迁移契约 → D2-2 嵌套 schema+兼容层 → D2-3 taskcore 读点+AST 守门 → D2-4 五任务读点+gentask `model.sisr` → D2-5 仓内 YAML 嵌套化 → D2-6 全量回归与 TODO 更新。

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
