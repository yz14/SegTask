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
1 生成项目代码审查（需结合对应readme/design/workflow一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程、推理全流程。可先独立深度审查，再串联全流程分析。项目是基于segtask_v1改造，需要对比segtask_v1的代码进行分析。原则上和seg设计能保持一致的尽量保持一致，seg的所有技巧能适用的都要用。  

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  


进展：



2 分割项目代码审查（需结合对应 readme/design/workflow 一起理解）：需认真、仔细、严谨的理解、分析、思考和调研。为保证高质量完成，本轮不动任何代码/文档：

分割项目 = 公共框架层 `taskcore` + 任务层 `segtask_v1`，审查按此两级展开。代码大致分 5 部分，数据读取、模型构建、数据增强/处理、训练全流程（含 val）、推理全流程，先独立深度审查，再串联起来全局分析。每部分先审公共层、再审任务层。

审查主要内容为代码、算法、设计、架构、工程等等：
是否正确、合理；是否有优化空间；是否有训练加速/GPU优化空间；是否有更好的高质量内容（算法/模块/设计/架构/损失等等）可以借鉴、适配或新增。现在是2026年7月，不局限医学图像领域，可能自然图像的分类/分割/检测/生成等、NLP、LLM、VLM等有更好、更先进的想法。

进展：  


3 重构调研：由于cls/det/gen/ssl都是基于seg构建的，而且在设计上能和seg保持一致的都和seg保持一致了（可能还有不一致我未发现），能复用技巧也基本上都复用了（可能会有没有复用的我未发现）。现在我想将公用的内容抽离出来，形成一个通用的框架，然后在各个子项目中复用，如果有的模块实在做不到通用，那就例如把通用的当父类，具体的子项目当子类，继承父类的通用部分，然后重写具体的子项目部分。仍然还是大致以数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程5部分来。先认真的彻底分析和理解现有cls/det/gen/ssl/seg项目代码（需结合对应readme/design/workflow一起理解），再仔细的调研公认高质量项目的架构设计等等（不要局限医疗，可能自然图像，NLP，LLM，VLM有更好的项目），最后再出一个最终的方案（需要用简单易懂的大白话解释清楚）。  

进展：已完成调研规划与实施。新建顶层公共包 `taskcore/`（config / data / models / engine / utils 五层），公共配置、数据层、公共模型（含 gen 侧漂移合并，逐位对拍一致）、训练/推理工程件全部下沉；五任务训练器接入共用基类 `BaseTrainer`，seg/gen/cls/det 推理器接入 `BasePredictor`（任务主循环保留在各自子类）。总览见根 README「公共包 taskcore」一节。  

已全部下沉共享（五任务通用）：seed/EMA(warmup+offload)/SWA 工具、AMP(auto dtype)/GradScaler、优化器/调度器/warmup、checkpoint I/O（原子写、RNG 快照、前缀剥离、compile 解包、异步保存器）、分布式辅助、显存统计、CUDA 预取、GPUAugmentor、数据发现/npz 预打包、公共骨干与 topology、BaseTrainer/BasePredictor 工程件（channels_last、compile、best-tracking、梯度/权重范数与 update-ratio 健康度、accum 尾组处理、EMA 换入换出、推理 AMP/TTA 组合）。

第二轮补齐（本轮）：  
- 公共层新增：`taskcore/monitor`（仪表盘 + history 落盘 + 离线渲染 CLI，原 seg 独有，`segtask_v1.monitor` 保留为 shim）；BaseTrainer 新增 `_setup_ddp`（DDP 装配，self.model 保持裸/已 compile，前向走 fwd_model）、`_setup_train_sampler`（set_epoch 鸭子识别 sampler/batch_sampler）、`_setup_monitor/_monitor_log_epoch/_monitor_render/_monitor_finalize`（rank0 守卫、失败隔离）；`_setup_channels_last/_maybe_compile` 支持传入外部模块（供 SSL 的 method.module）。  
- 全部接上：cls/det 接 channels_last + compile（compile-safe EMA 走 unwrap_compile）；cls/det/gen 接 DDP 装配 + 训练采样器 set_epoch + monitor 仪表盘（gen Config 新增 monitor 节）+ async saver rank0 守卫；ssl 改用共享 channels_last/compile/monitor/采样器识别 helper（保留手动梯度 all-reduce 语义）；seg 的 SWA/DDP/monitor/采样器识别改走 BaseTrainer 共用实现。  
- 旧 import 全部更新：内部代码（含 tests）经 shim 的 import 全部切换 taskcore 直连（约 140 文件）；shim 仅保留给外部旧脚本与 legacy checkpoint pickle 兼容（tests/test_ssltask.py 的 shim 兼容用例保持旧路径）。  
- 修复：MixedSampler `super().__init__(data_source=None)` 新版 torch 兼容（改无参）；taskcore 内残留 segtask_v1 文案/类型注解清理；factory 的 MultiRF decoder TODO 改为明确能力边界说明。  
- 回归：全量 pytest 1341 过 / 12 失败 / 3 跳过；12 个失败（11 个 test_model_flow 可视化 + 1 个 save_best_criterion 映射）在重构前基线上逐项复现，均为既有问题（test_model_flow 与 todo4 相关，建议先查）。

第三轮补齐（本轮）：  
- 选模口径定案：`save_best_criterion="loss"` 统一映射 `val_base_loss`（只看主任务损失，不含深监督/aux/正则附加项，口径稳定跨配置可比）；测试期望已同步（此前唯一非 model_flow 既有失败已消除）。  
- 多卡启动公共化：新建 `taskcore/engine/launch.py`（空闲端口、孤儿进程兜底 PR_SET_PDEATHSIG、SIGTERM/SIGINT 处理、allocator 碎片治理、`init_ddp_worker`/`finalize_ddp_worker`），原 seg 独有启动工程下沉；seg train / ssl pretrain 改用公用件。  
- cls/det/gen 多进程入口：三个 train 入口接入与 seg 同模式的 `mp.spawn` DDP 启动（YAML 配 `train.gpus` 即启用；单卡/CPU 路径零变化）；单卡时支持 `gpus: [k]` 指定物理卡。  
- 数据切分：cls/det/gen 的 dataloader 工厂新增 `rank/world_size`，多卡时训练集用 `DistributedSampler`（set_epoch 已由 `_setup_train_sampler` 接好）、验证集用 `ValBatchShardSampler` 按 batch 块不相交切分，num_workers 按卡数平摊（同 seg）。  
- 验证指标全局归约：dist_utils 新增 `all_gather_objects`/`all_reduce_meters_`；cls（logits/targets/vols 聚齐后算全集 AUC/F1 等不可分解指标）、det（预测/真值聚齐后算全集 mAP）、gen（PSNR/SSIM meter 加权 all-reduce；整卷验证逐 rank 切分后汇总）；选模/早停决策各 rank 天然一致。  
- 落盘 rank0 守卫：cls/det/gen 的 best/latest checkpoint、history、resolved config 仅 rank0 写。  
- 验证：全量 pytest 1342 过 / 11 失败（均为既有 test_model_flow）/ 3 跳过；另经 2 进程 gloo smoke 验证 all_gather/all_reduce/分片聚合与单进程全集等价。多 GPU 真机验证待有卡环境进行。
- 梯度检查点补齐：gen 的 UNet 系 factory 透传 `grad_checkpointing`/`grad_ckpt_encoder_stages`（此前公共层支持但 gen 未接线）；cls 的 DenseNet（逐 DenseBlock）与 ViT（逐 transformer Block）经 `checkpoint_if` 接入；gen 非 UNet 架构（adm/edm2/edsr/rcan）开检查点时 warning 提示忽略；ssl 走公共 factory 已天然生效无需改。ckpt on/off 前向+梯度逐位一致已验证。

仍留在任务层（语义差异，未强行合并，已定案保持独立）：滑窗推理全家桶（seg 与 gen 几何语义差异大）；SSL 的手动梯度 all-reduce（多 forward 入口无法套 DDP 单入口假设）。


4 模型流可视化需要有层次化，结构化，美化，可以清晰看到计算流的走向，可以清晰理解模型架构，可以清晰的溯源。总之：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判、讨厌"自动布局默认输出"式的无设计感结果。以下是一些例子：  

- 聚焦模式到stem, stage这个层级为止：
点击模块A，进入聚焦模式，模块群B和A有联系，模块群C和A没有联系，所以模块群C谈出，模块群B突显。我希望到stem，stage这个级别的模块能进入聚焦，再进一步的子模块例如stem，stage里面的子模块则不进入聚焦。  

- 连线走线需要清晰、不重叠、不交叉、美观、可以溯源：
需要清晰的看到不同模块的关系，并能溯源输入输出等等

- 位置清晰，层次清晰，严格遵守各自的位置关系：
例如输入后可能同时结果多个stem，那么这几个stem就是位置并列的；例如如果有deep supervision，且在dec level 0后有ds head 2, dec level 1后有ds head 1等等，那么ds head 2位置就应该和dec level 1并列，因为它们就是dec level 0的下一个计算。

- 其它的我暂时没有想到，请你根据我的喜好推荐，注意，原则是：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判。

进展：
