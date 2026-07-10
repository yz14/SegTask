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


内容（需查看各自对应README.md理解设计）：  
segtask_v1是2.5D/3D分割项目（项目起源，目前最完善）。  
ssltask是自监督学习项目（基于segtask_v1改造），主要是对分割，生成，分类，检测的backbone（部分/全部）进行预训练。  
clstask是分类项目（基于segtask_v1改造）。  
dettask是检测项目（基于segtask_v1改造）。  
gentask是生成/超分项目（基于segtask_v1改造）。


# TODO  
1 SSL项目代码审查：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程、推理全流程。

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  
在针对下游分割项目的预训练模块/方向上是否有更好的高质量内容可以借鉴、适配或新增。  

进展：
ssltask 架构清晰：复用 segtask_v1 的 Config / build_model / optim / AMP，方法层用 SSLMethod 插件 + SSLTrainer 通用循环，11 种 SSL 路线（重建 / 对比 / 自蒸馏 / MIM / 先验）实现完整，测试覆盖率高。面向分割迁移的核心契约——encoder.*（及部分方法的 decoder.*）与下游 train.pretrain strict=False 衔接——设计正确且有多项单测保护。

相对母项目 segtask_v1，主要短板在：训练基建未完全继承（无 DDP / 续训 / 非有限守卫）、数据 IO 偏朴素（无缓存、worker RNG 未隔离）、配置校验未覆盖若干运行时组合。下面按五块展开。

跨层主线问题（最高优先级）
优先级	问题	位置	说明
P0
SSLReconModel 未处理 stem_stride>1 输出分辨率
ssl_models.py:48-56 vs unet.py:510-512
分割 UNet 主头已有 _resize_logits 上采；Genesis/Prior 重建头直接 RuntimeError。patch2/patch4 stem 下 genesis/prior 必崩。SparK 解码器有 interpolate 兜底（spark_modules.py:196-198），SimMIM 头也会插值，仅 enc+dec 重建路径受影响。
P0
非有限 loss 仍执行 optimizer.step()
ssl_trainer.py:237-243,228-229
日志写 "skipping" 但只跳过 meter，不跳过优化步，NaN/Inf 会污染权重。母项目有 skip_optim_step + DDP all_reduce。
P1
SparK 与 hierarchical stem 组合无配置拦截
spark_modules.py:58-61
运行期 NotImplementedError；validate_ssl 未校验 stem_mode × method。
P1
无 checkpoint 续训 / RNG 持久化
ssl_trainer.py:118-129
ckpt 仅含 model_state_dict + best 指标；长训中断无法严格复现。母项目 checkpoint 含完整 RNG/optim/sched。
P1
数据 IO：每 sample 全卷解码、无 LRU
ssl_dataset.py:110-119,121-128
冷路径每 patch 开 npz + 解压整卷；多 worker 下 IO 成为瓶颈（母项目有 VolumeCache + meta 快路）。
P1
Worker RNG 未隔离
ssl_dataset.py:47-53,127
_rand_center 用全局 random；fork 后多 worker 可能重复采样位置（母项目 dataset.py:592-606 已用 per-worker Generator）。
P2
训练基建相对母项目退化
ssl_trainer.py 全文
无 DDP、torch.compile（config 有 compile_mode 但未接线）、梯度累积尾批无 _effective_accum 缩放、无 async checkpoint。
P2
export_backbone_state_dict 语义不一致
各 method
DINO/JEPA/BYOL/MoCo 只导出 encoder.*；Genesis/Prior 导出 enc+dec+recon_head；SimMIM/SparK 导出含 head.*/spark_decoder.*/mask_token。功能上 strict=False 可用，但 ckpt 体积与命名易误导。
P2
MoCo 默认 queue 过小
config.py:139
默认 moco_queue_size=32；标准 MoCo 用 65536 级队列，小 batch 医学影像下负样本严重不足。
P2
预训练几乎无通用空间增强
configs/ssltask_*.yaml + ssl_dataset.py
示例均 augment.enabled: false；仅方法内增广（Genesis 破坏 / DINO multicrop）。对分割迁移，缺少旋转/弹性等可能限制表征泛化。
1) 数据读取
[正确性]
均匀随机 patch + edge-pad 抽取，逻辑正确

ImageOnlyPatchDataset 通过 _extract_cubic_patch（ssl_dataset.py:128）复用 segtask 几何原语；越界 edge 复制与分割一致。
2.5D 折叠口径 (C=D, H, W) 与下游一致（ssl_dataset.py:64-67,130-133）；build_ssl_dataloader 显式拒绝多 FOV 2.5D（ssl_dataset.py:248-256）。
探针数据集 image/label 同中心对齐

LabeledPatchDataset 共用 center 抽 patch（ssl_dataset.py:223-225），探针评测合理。
继承 segtask cubic 缓存别名风险（低概率）

无 padding 时 _extract_cubic_patch 可能返回缓存卷视图（母项目 dataset.py:922-932 已记录）；SSL 默认无 in-place 增强，风险低于分割 aug.inplace=True 路径。
[设计]
有意与分割标签管线解耦

image-only 不读 label/fg_coords，符合 SSL 大规模无标注场景（ssl_dataset.py:3-6）。
缺少 segtask 级数据治理

无 VolumeCache / 合并 npz 读取 / per-worker RNG / 前景过采样 / val 确定性采样。
对 LUNA 级 cohort + samples_per_volume=8 + num_workers=16，IO 与 CPU 预处理会成为首瓶颈。
build_ssl_dataloader 无 DDP 感知

单卡设计；多卡需从母项目移植 DistributedSampler + worker 平摊策略。
[优化]
合并 _load_volume 为 bundle 读取 + LRU（参考母项目 _load_npz_bundle 建议）。
可选前景偏置采样（即使无 label，也可用 intensity/gradient 启发式），提升小结构分割迁移。
启动期 lazy index（母项目 _build_index N 次开 npz 问题同类）。
[做得好]
复用 preprocess_image / _open_npz，零重复实现。
2.5D 单 FOV 守卫清晰，早失败优于 silent 错配。
discover_image_npz 递归发现 + 排序，行为确定。
[借鉴]（面向分割预训练）
来源	建议
nnU-Net
预计算前景位置 + patch 采样偏置（可仅用 image 梯度/边缘代理）
MONAI CacheDataset
patch 级或 volume 级跨 worker 共享缓存
segtask_v1 自身
直接移植 VolumeCache + per-worker RNG + DDP loader 策略
2) 模型构建
[正确性]
共骨干契约：设计正确，测试充分

全部 SSL 模型经 build_model(cfg) 取 encoder（ssl_models.py:78, dino_modules.py:110 等），保证与下游逐参数同名同形。
tests/test_ssltask.py 覆盖 genesis/simmim/spark/dino/jepa/ibot 等 handoff。
P0：Genesis/Prior 重建路径 stem_stride 缺口（见跨层表）

SparK 掩码-稠密等价：实现质量高

spark_encode 全可见时严格等于稠密 forward（单测 test_spark_encode_full_density_*）。
hierarchical stem 显式拒绝（spark_modules.py:58-61），但应在 config 层提前拦。
SimMIM 轻量头：正确

LightPixelHead 插值到输入分辨率（ssl_models.py:123-129），无 skip 的设计与论文一致。
DINO 投影头：忠实

weight_norm + 固定 g（dino_modules.py:60-64）；GAP 在 bottleneck（dino_modules.py:90-92）。
方法限制：arch=='unet' only

validate_ssl 硬约束（config.py:459-461）；ADM/EDM2 等不可用——对 SSL 合理，但应在 README 明确。
[设计]
方法族导出策略分化（有意但需文档化）

方法	导出内容	下游分割收益
genesis, prior
encoder + decoder + recon_head
enc+dec 均可 warm-start
simmim, spark
encoder + 方法头（head/spark_decoder/mask_token）
仅 encoder 命中
dino, jepa, ibot, byol, moco, sparkdino
仅 encoder（teacher/query）
仅 encoder 命中
BYOL/MoCo 各 build 两次 build_model

byol.py:77-78, moco.py:81-82：online/target 各一套 encoder，内存双倍；可 deepcopy 一次初始化。
线性注意力 / 自注意力等高级 backbone 特性

SSL 完全继承 segtask encoder 能力（MultiRF、self-attn 等），无额外限制——好。
[优化]
export_backbone_state_dict 统一过滤为「下游所需最小集」（encoder ± decoder），减小 ckpt、避免歧义。
config 增加 method × stem_mode × patch_mode 兼容性矩阵校验。
SimMIM/SparK 的 mask_unit 自动对齐 encoder 总 stride（当前手动配置，易错）。
[做得好]
SSLMethod 抽象干净（methods/base.py），新增方法零改 trainer。
SparK 解码器宽度可控（spark_decoder_dim_div），参数量约 encoder 1/5–1/10。
Frangi vesselness 纯 torch、2D/3D 通用（vesselness.py），domain-specific 创新。
[借鉴]（分割预训练方向）
方法	医学 3D 分割适配建议
SparK / SimMIM
首选 encoder-only MIM；mask_unit 对齐 nnU-Net patch 步长
Genesis
管状/对比剂场景仍有效；可与 prior 组合做 multi-task
Prior (Frangi)
血管/管腔分割 niche 强；可扩展 Hessian/Hessian+Learned filter
DINO + iBOT (⑥)
全局+密集特征，对分割 probe 最对齐；已有实现
JEPA (⑦)
隐空间预测，显存友好；VICReg 默认关，稀疏结构可开
SparkDino (⑧)
像素+全局双监督，工程上最均衡的多任务组合
3) 数据增强 / 处理
[正确性]
Genesis 破坏：实现正确

Bézier 强度 / 局部 shuffle / 内外补全（corruptions.py）；per-sample、不原地修改输入（单测覆盖）。
MIM 掩码工具：设计严谨

单元网格掩码 + 固定比例 + 禁止全遮/全见（masking.py:49-73）。
masked_recon_loss 仅在被遮位点归一（masking.py:165-179）。
SparK per_unit_normalize 与 MAE 一致（masking.py:136-151）。
MultiCrop：DINO 族共享，正确

random-resized-crop + flip + 强度扰动（multicrop.py）；2D/3D 通用。
Prior Frangi：可微、多尺度

分离高斯 + Hessian 特征值（vesselness.py）；2.5D 逐通道 2D 处理合理。
无通用 segtask 空间增强链路

SSL 训练不经过 GPUAugmentor；config 示例 augment.enabled: false。
对 DINO multicrop 内的轻量 flip/强度扰动足够其目标；对 Genesis/SparK/SimMIM 像素重建，缺少旋转/弹性可能限制分割迁移（尤其小结构方向不变性）。
[设计]
增广职责分层清晰

数据集：干净 patch；方法层：任务特定变换（破坏/掩码/多裁剪）。比全堆在 dataset 更易组合。
RNG 混用

Genesis/MultiCrop 用 Python random + GPU 默认 RNG；严格复现依赖 seed_everything 但 worker 级不可控。
Prior 目标在 GPU 实时算

每 step 算 Frangi（prior.py:36-39），大 patch 有算力开销；可离线烘焙 vesselness 到 npz（类似 make_data 哲学）。
[优化]
可选接入 segtask GPUAugmentor（仅 image 分支），配置开关 ssl.spatial_augment。
Genesis corruptor 可向量化（当前 per-sample Python 循环，corruptions.py:146-157）。
MultiCrop _make_crops 的 per-sample 循环（multicrop.py:110-113）可 batch 化。
[借鉴]
实践	说明
nnU-Net 式增强
旋转/缩放/弹性 + label-safe zeros padding（若未来做 pseudo-label SSL）
Models Genesis 原论文
已实现四类破坏；可加 slice shuffling（3D 特有）
MAE/SparK
norm_pix 已有；可加 random block aspect ratio
医学 SSL 综述
Anatomical constraint flip（左右不可翻的器官）
4) 训练 / 验证（评测）
[正确性]
P0：非有限 loss 不跳步（ssl_trainer.py:237-243）——见跨层表。

梯度累积尾批缩放缺失

loss / accum 固定除法（ssl_trainer.py:218-219），尾组 micro-batch 不足 accum 时梯度欠缩放（母项目 _effective_accum 已修）。
EMA 与方法内 teacher EMA 叠加

DINO/JEPA 已有内部 EMA teacher；validate_ssl 仅 warning（config.py:321-325,354-358），不强制 use_ema=false。trainer EMA 会平滑含 teacher 在内的 method.module，语义混乱。
在线探针：设计优秀

固定 seed 重置头 + 固定 iters（probe.py:231-257），跨 epoch 可比。
多尺度 1×1 线性头（probe.py:78-99）比单层 GAP 更贴分割。
2.5D 输出 num_fg×D 通道（probe.py:115-116），与分割 topology 对齐。
探针失败不中断训练（ssl_trainer.py:169-171），稳健。
探针 Dice 阈值

用 compute_dice_per_class 默认 0.5（probe.py:220）；若下游 predict.threshold≠0.5，probe 选模与部署指标可能分裂（母项目同类问题）。
best 选择逻辑

probe 启用时按 probe_dice（ssl_trainer.py:177-184）；EMA 权重导出（ssl_trainer.py:110-115）。与母项目 seg trainer（best 存 EMA）一致且正确。
探针全程失败时 fallback 存 final（ssl_trainer.py:189-193），合理。
无验证集 / 无 SSL val loss

设计选择：无标注语料无 val；靠 online probe 或 train loss。文档已说明。
[设计]
Trainer 薄、Method 厚

优化器/调度/AMP/EMA 复用 segtask（ssl_trainer.py:44-57），避免重复。
Checkpoint 极简

无 optim/sched/RNG/method 内部 state（DINO center、MoCo queue 等）；无法真正 resume。
无 DDP / ZeRO / compile

大规模预训练扩展性受限；config 字段存在但未用。
[优化]
从母项目移植：non-finite guard、_effective_accum、DDP、checkpoint resume、可选 torch.compile。
MoCo queue 改为可配置默认 4096+；按 batch 自动建议。
探针 probe_iters 可自适应（早停）以减 epoch 开销。
保存 method 内部状态（DINO center、MoCo queue_ptr）以便 resume。
[做得好]
configure_schedule / on_after_step hook 支持 DINO/JEPA/BYOL/MoCo 动量调度。
离线评测 pipeline 完整（eval/pipeline.py）：nested shots、seg+cls、linear/finetune、CSV/JSON 输出。
198 项单测覆盖 config/forward/backward/handoff/trainer smoke——质量标杆。
[借鉴]
实践	说明
nnU-Net 选模
probe Dice 优于 SSL loss——已实现，建议在默认 config 开启
DINOv2
iBOT + KoLeo（可选邻居排斥）；Gram 已有 dino_gram
线性 probe 规范
报告 k-shot 曲线（eval pipeline 已有）+ 多 seed 均值
5) 推理 / 采样（评测与权重导出）
ssltask 无分割式推理；「推理全流程」= 权重导出 + 下游加载 + 探针/离线评测。

[正确性]
下游加载路径正确

segtask_v1.trainer.Trainer._load_pretrain（trainer.py:1274-1304）strip prefix + strict=False + 日志 missing/unexpected。
示例 config 已文档化衔接（configs/ssltask_genesis.yaml:4-5）。
导出权重与验证模型一致

_export_state_dict EMA 优先（ssl_trainer.py:108-116）；DINO 等方法 export teacher encoder——与 eval 语义一致。
离线 eval prefer_ema=True

pipeline.py:75 与 trainer 保存逻辑对齐。
SparkDINO / iBOT 导出继承 DINO

仅 teacher encoder.*；SparK 解码器 / iBOT 头丢弃——符合「仅迁移 encoder」约定。
[设计]
无独立 SSL inference CLI

合理：预训练产出 ckpt，分割/检测走各自 predict.py。
eval 不跑完整 UNet 微调

pipeline.py:4-5 明确 finetune 仅 encoder+线性头；与「轻量探针」目的一致，但不等于最终分割性能上界。
[借鉴]
增加 encoder 特征可视化 / CKA 工具，辅助方法对比。
支持 partial load（仅 stem / 仅 stage0-2）用于渐进微调实验。
面向下游分割预训练的方法推荐
场景	推荐方法	理由
通用器官分割
SparK 或 SimMIM + probe 选模
encoder 表征强；SparK 稀疏门控与下游稠密 forward 等价已验证
需要 decoder warm-start
Genesis
唯一默认导出 enc+dec 的重建路线
血管 / 管腔
Prior (Frangi)
几何先验与任务对齐；可与 Genesis 分阶段
分割 + 分类双任务
iBOT 或 SparkDino
密集+全局监督；probe 已覆盖 seg+cls
少标注 probe 选模
任意 + probe_enabled=true
避免按 SSL proxy loss 误选



2 分割项目代码审查：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程。

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  

进展：



3 


4 模型流可视化需要有层次化，结构化，美化，可以清晰看到计算流的走向，可以清晰理解模型架构，可以清晰的溯源。总之：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判、讨厌"自动布局默认输出"式的无设计感结果。以下是一些例子：  

- 聚焦模式到stem, stage这个层级为止：
点击模块A，进入聚焦模式，模块群B和A有联系，模块群C和A没有联系，所以模块群C谈出，模块群B突显。我希望到stem，stage这个级别的模块能进入聚焦，再进一步的子模块例如stem，stage里面的子模块则不进入聚焦。  

- 连线走线需要清晰、不重叠、不交叉、美观、可以溯源：
需要清晰的看到不同模块的关系，并能溯源输入输出等等

- 位置清晰，层次清晰，严格遵守各自的位置关系：
例如输入后可能同时结果多个stem，那么这几个stem就是位置并列的；例如如果有deep supervision，且在dec level 0后有ds head 2, dec level 1后有ds head 1等等，那么ds head 2位置就应该和dec level 1并列，因为它们就是dec level 0的下一个计算。

- 其它的我暂时没有想到，请你根据我的喜好推荐，注意，原则是：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判。

审查：
