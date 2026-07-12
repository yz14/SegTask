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


内容（**注意**：需查看各自对应README.md理解设计）：  
segtask_v1是2.5D/3D分割项目（项目起源，目前最完善）。  
ssltask是自监督学习项目（基于segtask_v1改造），主要是对分割，生成，分类，检测的backbone（部分/全部）进行预训练。  
clstask是分类项目（基于segtask_v1改造）。  
dettask是检测项目（基于segtask_v1改造）。  
gentask是生成/超分项目（基于segtask_v1改造）。


# TODO  
1 SSL项目代码审查（需结合对应readme一起分析）：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程、推理全流程，可先独立深度审查，再串联起来全局分析。  

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  

同时分析在针对下游分割项目的预训练模块/方向上是否有更好的高质量内容可以借鉴、适配或新增来提升分割性能。  

进展：
总体判断
工程骨架质量较高：SSLMethod 统一接口、共享下游同构 backbone、encoder/decoder 可迁移、配置校验、LRU volume cache、完整断点恢复、手动 DDP buffer 同步、在线分割/分类 probe 均已形成闭环；现有约 160 个测试覆盖面也不错。base.pyssl_dataset.py:74-147ssl_trainer.py:370-439

但当前存在几项会直接影响训练日程、评测可信度和算法名称/论文一致性的问题。在修复前，不建议依据现有结果给 12 种方法排名或决定下游最优预训练方案。

P0：Phase 2 应先修复
梯度累积下 LR / warmup 日程计算错误。 Scheduler 用 len(loader) * epochs 作为总步数，但只在 accumulation boundary 调用 scheduler.step()；如 grad_accum_steps=4，整个训练只走完约 1/4 的 cosine schedule，warmup 实际拉长约 4 倍。DINO 等内部 EMA 日程使用的却是 optimizer-step 数，两套时钟不一致。ssl_trainer.py:78-91ssl_trainer.py:558-652
修法：统一定义 optimizer_steps_per_epoch=ceil(len(loader)/accum)，scheduler、warmup、EMA、checkpoint global step 全部基于 optimizer step。
验收：尾部不足 accum 的组也只记一次；训练结束 scheduler 恰好到达最终点；resume 后曲线连续。
HD95 不是标准 surface HD95。 当前 EDT 对整块前景 mask 求距离，而不是对预测/真值边界求双向 surface distance；一侧为空返回 NaN，汇总时又直接跳过，可能系统性美化漏检结果；同时未使用 voxel spacing，输出不是毫米。metrics.py:79-131
修法：提取二值表面，使用 spacing-aware 双向表面距离；明确规定空集情形，另报告 empty-case 计数。
验收：与 MedPy/MONAI 已知样例一致；各向异性 spacing、单边空、双边空均有测试。
dino_gram 当前不等价于 DINOv3 Gram anchoring。 默认每个 optimizer step 都把 Gram teacher 覆盖为当前 EMA teacher，而且从训练开始就刷新；到 start_frac 启用 Gram 时，“早期高质量快照”已不存在。官方 DINOv3 是独立 refinement stage，从先前 checkpoint 加载 anchor，并限制首次刷新、刷新频率和最大次数；官方还使用无扰动、更高分辨率的 Gram-teacher crop。当前实现却重新随机生成普通 global crops。dino_gram.py:101-141config.py:128-137
两种合规选择：A. 按官方 staged recipe 重构并保留 dino_gram 名称；B. 保留现算法但改名为 temporal Gram consistency，避免错误归因。
VICRegL 局部匹配会给无重叠视图强行制造正样本。 当前无论两个 crop 是否相交，都取最近 k 个位置；且仅做单向几何匹配，没有官方 VICRegL 的几何匹配 + 特征最近邻双路径。torch.var 默认 unbiased，在 batch size=1 时全局 variance 项可产生 NaN。vicregl.py:19-39vicregl.py:90-115
修法：显式计算 crop 交集，只在有效 overlap 内双向匹配；补 feature-based matching；variance 用 unbiased=False 或强制有效样本数≥2。
另有完整性问题：12 个方法中只有 VICRegL 缺少示例 YAML。
在线 probe 目前不足以作为可信的“最佳表征”选择器。 train/val 均从体积中均匀随机裁 patch；血管等稀疏目标会产生大量空 patch，而 Dice 汇总跳过空样本。划分按文件而非 patient/group，无法防止同患者多文件泄漏；HD95 又受上述问题影响。probe.py:41-79probe.py:253-270
修法：固定 patient-level split；验证集采用确定性前景/困难背景 patch 或完整滑窗；同时报告 Dice、HD95(mm)、NSD、empty recall；在线选择与最终独立测试分离。
P1：重要正确性与方法学问题
AMP/跳步状态不完整。 bf16/fp32 只检查 loss 是否有限，未检查 finite loss 产生的非有限梯度；fp16 GradScaler 若跳过 optimizer step，trainer 仍推进 scheduler、global step 和 teacher/EMA。所有非成功更新都应保持统一状态不推进。ssl_trainer.py:600-651
梯度累积并不等价于大 batch。 DINO/iBOT center、MoCo queue 等在每个 micro-batch 的 compute_loss 内立即更新，后续 micro-batch 看到已变化的目标状态；应在 accumulation group 完成后统一提交状态，或明确记录这是近似训练。
JEPA 实际是“mask-token latent regression”，不是 I-JEPA。 上下文 encoder 看到了 mask token，使用单一随机 unit mask；缺少 I-JEPA 的可见 context block、多 target block、位置条件 predictor。默认 anti-collapse 权重还是 0。建议重命名或按论文重构。jepa.py:91-133
DINO 论文一致性不足。 缺少 last-layer freeze；CNN/BatchNorm 场景下 teacher 仍处于 train mode；中心更新和梯度累积语义需重新定义。
MoCo/BYOL 正样本语义偏弱。 两个 independently cropped 3D views 可能几乎不重叠，却被强制做全局正对；医学小目标场景容易学习身体/扫描协议而非局部病灶。应采用 overlap-aware views 或坐标条件局部目标。
多裁剪比例按每个轴独立采样。 3D 下体积比例可被立方放大，容易出现极细长/极小 crop，与论文中面积/体积比例定义不同；还忽略物理 spacing。multicrop.py:105-127
Frangi prior 需 spacing-aware 并防止噪声放大。 sigma 当前是 voxel 单位，patch 内按最大值归一化会把无血管 patch 的微弱噪声放大到 1；逐体素 eigvalsh 也可能成为主要训练瓶颈。建议将其作为辅助项而非单独主目标，并先 profile/校准。
DDP 可扩展性。 手动逐参数 all-reduce 缺少 bucket overlap；epoch loss 只记录 rank0 本地值；rank0 在线 probe 会额外常驻一套 encoder，并使其他 rank 等待，存在显存不对称和 collective timeout 风险。
checkpoint 原子性与配置指纹不足。 当前直接覆盖 .pt；建议临时文件写完后原子替换，并保存数据清单 hash、git/config fingerprint、world size 和有效 batch 语义。
P2：性能与工程优化
MultiCrop、PairedCrop、Genesis corruption 均有 Python batch × crop 循环，可批量生成 affine grid 后 grid_sample，减少 GPU launch 和 host 同步。multicrop.py:105-203
训练热路径中大量 float(tensor) / .item() 导致 CUDA 同步；日志应延迟或按间隔聚合。
npz 是压缩格式；现有 LRU cache 已解决重复解压的主要问题，但大规模数据建议 benchmark mmap-friendly NPY/Zarr/HDF5 与 worker/cache 组合，而不是直接假定更换格式更快。
增加固定 compute budget、吞吐、峰值显存、IO wait 和每方法有效 views 数统计，否则不同 SSL loss 的横向比较不公平。
12 种方法的定位建议
优先基线：SimMIM、SparK。 与当前 CNN/UNet 下游架构最匹配；SparK 的 seg 模式还能预训练 decoder。需注意当前 SparK 是 dense zero-mask 近似，不具备官方 sparse-conv 的计算节省。
最值得修正后保留：VICRegL、iBOT、SparkDINO。 它们直接包含局部/密集目标，更符合分割迁移；但匹配、crop 和 loss balancing 必须先修。
辅助基线：Genesis、Prior、DINO、MoCo、BYOL。 可用于消融，但不应默认视为最佳分割方案。
需先纠正名称/算法：DINOGram、JEPA。 当前实现不能直接宣称复现对应官方方法。
Prior 的合理定位： Frangi/血管先验适合做小权重辅助任务，避免把经典滤波器的上限和伪响应固化为 encoder 的唯一目标。
面向下游分割的推荐路线
不建议继续无差别堆方法。更高收益的主线是：

同构 CNN hierarchical MIM：以修正后的 SparK/SimMIM 为强基线，多尺度重建 encoder features，并比较 encoder-only 与 encoder+decoder 迁移。
局部空间一致性：修正 VICRegL，使用真实 crop 坐标、物理 spacing 和 overlap mask；这比纯全局实例对比更贴近稀疏结构分割。
全局解剖上下文：若 CT body region 相对稳定，可引入 VoCo 式 contextual position prediction；位置应使用归一化身体坐标而非绝对 voxel index。
组合目标而非更多孤立方法：hierarchical MIM + local correspondence + 小权重 anatomical prior；先做独立消融，再考虑 SparkDINO 式联合。
数据规模与评测优先于方法新颖度。 CVPR 2025 的 3D MAE 工作强调大数据、下游同构 Residual Encoder U-Net 与严格多数据集评测，通常比不断换 pretext task 更重要。



2 分割项目代码审查（需结合对应readme一起理解）：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程，可先独立深度审查，再串联起来全局分析。  

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

进展：
