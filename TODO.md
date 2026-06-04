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


测试环境为: **D:\miniconda\envs\torch27_env\python.exe**。  


这是我写的2.5D/3D分割项目代码D:\codes\work-projects\SegTask\README.md，训练入口在D:\codes\work-projects\SegTask\segtask_v1\train.py。这里有3个3D方案，z轴滑块（只在z轴滑动切块，x,y为全尺寸）；cubic滑块（在x,y,z轴滑动中心切块）；whole（直接输入整个图像）。一个2.5D方案，它和z轴滑块的单分辨率/感受野方案非常的相似，区别是：a 在train的时候，当数据增强结束后，将3D数据B,1,D,H,W变为B,D,H,W作为2D输入,D张切片代表D个通道；b 模型采用2D模型。计算损失也和现有框架一致，模型输出为B,num_fgxD,H,W然后拆分为num_fg个B,D,H,W单标签预测，各自计算单标签损失。这里有一份小数据集作为测试：F:\med_data\Totalsegmentator_dataset_v201\small_data\nii，F:\med_data\Totalsegmentator_dataset_v201\small_data\mask，
F:\med_data\Totalsegmentator_dataset_v201\small_data\bbox，
F:\med_data\Totalsegmentator_dataset_v201\small_data\region_weihgt。  


# TODO  
1 2.5D用这个配置训练D:\codes\work-projects\SegTask\configs\seg2_5d.yaml，我已经开了多分辨率输入，但是训练出来的模型和单分辨率没有什么提升，尤其是z轴的空间信息感觉很差劲，感觉模型是一张切片一张切片的在单独处理，而无法联系输入的所有的切片，所以这个2.5D感觉根本没有起到2.5D的作用，似乎更像是单张切片的纯2D效果。请检查代码，是模型实现的有问题吗，还是哪里有问题呢？同时你有什么改进建议给出吗，例如针对这个配置，哪些参数可以做什么调整来改进效果。

2 为了探索上面1的问题，我用D:\codes\work-projects\SegTask\configs\seg3d.yaml进行了训练，发现它们的badcase一模一样，也就是说用3D的z-axis也无法解决1中的bad case。请你检查3D的全流程代码是否有问题，为什么换了模型，bad case一模一样而且现象也一模一样。以上的训练都是用了npz_dir: "F:/BaiduNetdiskDownload/lung_prep"这里的数据，同时分割骨头和肺，区域权重对肺的边缘加了很大的权重9，对肺外的一圈组织加了权重7，对骨头加了权重4，对hu和肺相近的地方加了权重14。

3 最后2的数据基础上，彻底检查一遍损失函数，是否都正确实现了，对应我这样的数据是否可行，是否有问题。

4 有些代码太大，逻辑绕口，让人读起来费劲，维护困难，例如config.py文件，如果某个参数会被自动重写或者更新，那么它就不应该暴露接口出来，例如save_best_metric。前全面检查代码其它地方是否有类似的问题。我需要的是让人读起来代码来不那么费劲，绕圈，甚至多个文件反复查看确认才能理解代码。

5 在推理阶段，如果开启z_interleave_enabled，adabn_enabled，tta_flip后，如何在不显著增加显存，不损害模型效果的情况下加速推理呢，我粗略的分析是：
一、当前瓶颈量化
我完整阅读了 predictor/ 包下的 predictor.py、forwards.py、sliding.py、adabn.py、io.py、inputs.py、blending.py 以及 config.py 的相关配置。三者叠加的耗时效应如下：

特性	当前实现	额外前向倍数
tta_flip	串行循环：原始输入后逐个 flip 再逐个 model() 前向，3D 7 种 flip、2.5D 3 种 flip	8× (3D) / 4× (2.5D)
z_interleave_enabled	串行处理 k 个子流：每子流独立走完整 sliding_window_z（含各自的 batching、interpolate、blend）	k × 整卷推理（默认 z≤1.0 时 k=3）
adabn_enabled (per_volume)	整卷跑两遍：先完整 predict_preprocessed_array 估计 BN，再真实预测	2× 整卷推理
最坏叠加：k=3 × AdaBN=2 × 3D TTA=8 = 48 倍原始单前向时间。这是本次优化的核心痛点。

二、优化方案与业界做法
方案 1：TTA 串行 → Batch 化（最核心、最安全）
当前 TTA 在 forwards.py:50-90 中是对每个 flip 变体串行调用 p.model(...)。工业界标准做法（nnUNet、MONAI 等）是将多个翻转版本 concat 成一个更大的 batch 统一前向，再拆分反 flip 后平均。

显存控制：不能一次性堆叠全部 8 个变体。引入 tta_batch_size 参数分块处理。例如 tta_batch_size=2 时，3D TTA 从 8 次前向降至 4 次，激活显存仅增加 2 倍。
效果：逐像素严格等价，因为只是改变了计算顺序。
方案 2：AdaBN per_volume → 稀疏采样（最减负）
当前 per_volume 模式在 predictor.py:301-311 中，是对整卷跑完整 predict_preprocessed_array 来估计 BN。但 BN running stats 的收敛只需要足够样本量，不需要所有 patch。

优化：估计阶段只采样最多 adabn_max_patches（如 64）个代表性 patch（均匀或随机选 z 位置/cubic 中心）做 forward，跳过 blend、interpolate、保存等后续步骤。
效果：估计时间从 ~1× 整卷推理降至 ~0.05–0.1×，且 BN 统计量已足够稳定。
方案 3：z_interleave 串行子流 → 统一 batch（减少循环开销）
当前 sliding_window_z_interleaved 在 sliding.py:207-248 中循环 for i in range(k): sub_prob = sliding_window_z(p, sub_vol)。k 个子流之间没有任何数据依赖，只是 blend 目标位置不同。

优化：不再串行调用 k 次 sliding_window_z，而是：
统一计算所有子流的 z_positions；
统一收集所有窗口输入（标记所属子流 id 和 z 位置）；
按原有 batch_size 统一 batch forward；
各子流分别 blend 到各自的 acc_pred/acc_weight；
最后缝回原体积。
效果：forward 总次数不变，但消除了 k 次重复的 Python 循环、kernel launch 和 CPU-GPU 同步，GPU 利用率更高。
三、执行计划（可拆分、可独立执行）
步骤 1：TTA Batch 化
目标文件：forwards.py + config.py
具体修改：
在 PredictConfig 新增 tta_batch_size: Optional[int] = None（None 时退化为 batch_size）。
重写 tta_flip_ensemble 和 tta_flip_ensemble_2_5d：将原始输入和所有 flip 变体按 tta_batch_size 分块，每块 torch.cat 成 (B*N, ...) 统一送入 p.model()，输出拆分、反 flip、累加后平均。
预期产出：3D TTA 前向次数从 8 次降至 ceil(8/tta_batch_size) 次；2.5D 从 4 次降至 ceil(4/tta_batch_size) 次。
验收标准：
同一张图开启 TTA 后，输出概率体与原始实现逐像素 max abs diff < 1e-5。
单测覆盖 3D 和 2.5D 两种 TTA 路径。
显存承诺：激活显存增加不超过 tta_batch_size 倍（默认 batch_size=2，即最多 2 倍）。
步骤 2：AdaBN per_volume 稀疏采样
目标文件：predictor.py + config.py
具体修改：
在 PredictConfig 新增 adabn_per_volume_max_patches: int = 64（0 表示全卷）。
在 predict_volume 的 per_volume AdaBN 估计分支中，不再调用完整 predict_preprocessed_array，而是：
对 sliding_window_z：从 z_positions 中均匀/步长采样最多 max_patches 个，构建窗口、forward、直接丢弃输出（仅驱动 BN 统计）。
对 cubic：从所有 cubic 窗口中均匀采样最多 max_patches 个。
对 whole：仍走全卷（本身只一次 forward，无优化空间）。
预期产出：per_volume 估计时间从 ~1× 整卷推理降至 ~0.05–0.1×。
验收标准：
稀疏采样后的 BN running_mean 与全卷估计的相对误差 < 1%。
最终预测结果（Dice）与全卷估计的差异 < 0.1%。
max_patches=0 时行为与现有实现逐像素等价。
步骤 3：z_interleave 子流统一 batch
目标文件：sliding.py + config.py（如有必要）
具体修改：
重写 sliding_window_z_interleaved：
为 k 个子流分别计算 z_positions；
统一收集所有子流的窗口输入，维护 (sub_stream_id, z0, z1, actual_d) 元数据列表；
按 batch_size 统一 batch forward（调用 forward_batch_gpu）；
每个子流维护独立的 acc_pred_sub[i] / acc_weight_sub[i]，interpolate 后按元数据写回；
所有子流完成后，out[:, i::k] = acc_pred_sub[i] / acc_weight_sub[i]。
预期产出：z-interleave 开启时，kernel launch 和循环同步开销从 k 倍压缩到 1 倍。
验收标准：
同一张图 z_interleave 开启时，输出概率体与原始实现逐像素 max abs diff < 1e-5。
k=3 时推理速度提升 > 1.3×。
四、依赖关系与执行顺序
步骤 1 与步骤 2 互相独立，无代码冲突，可分属不同轮次或并行提交。
步骤 3 涉及 sliding.py 较大面积重构，与步骤 1/2 无直接依赖，但建议作为单独轮次放在最后，以降低 review 难度。

修订后的执行计划
我建议按真实收益重排优先级,每步独立、可单独验收:

步骤 A(原 step 1):TTA 批量化 — 通用最高价值

文件:forwards.py + config.py
新增 tta_batch_size: Optional[int];tta_flip_ensemble / tta_flip_ensemble_2_5d 把原图+flip 变体按 tta_batch_size 分块 torch.cat 统一前向、拆分、反 flip、平均。
验收:纯 eval 下与原实现 max abs diff < 1e-5(3D+2.5D 两路);显存增幅 ≤ tta_batch_size×;AdaBN 估计期保持串行(规避上述陷阱)。
步骤 B(原 step 3):z_interleave 统一 batch — 当前配置主要收益点

文件:sliding.py
统一收集 k 个子流的所有窗口(带 (sub_id, z0, z1, actual_d) 元数据),按 batch_size 统一前向,各子流独立 blend 后 out[:, i::k] 缝回。
验收:与原实现 max abs diff < 1e-5;凑满 batch 带来的提速(我会实测,不预设 1.3×)。
步骤 C(原 step 2):AdaBN per_volume 稀疏采样 — 仅对 per_volume 有效

文件:predictor.py + config.py
新增 adabn_per_volume_max_patches;估计期从 z_positions/cubic 中心均匀采样 ≤N 个窗口仅驱动 BN,跳过 blend/保存。
验收:running_mean 相对误差 < 1%;最终 Dice 差异 < 0.1%;=0 时退化为现实现。
依赖:三步互相独立。步骤 A 与 B 对你当前配置(tta off / adabn global / k=4)收益最大的是 B;A、C 是为开启对应开关时准备的通用能力。

需要你确认
由于真实瓶颈与原分析有出入,我想先和你对齐范围再动手:

Q: 基于上述修正后的分析,本轮编码你希望我先做哪一步?(三步均独立,可分轮高质量完成)
先做 A,再议 B/C
Thought for 17s
You chose A — TTA 批量化. This is the execution round; I'll implement step A with equivalence regression tests and the AdaBN-estimation trap avoidance. Let me first confirm how YAML populates PredictConfig so the new optional field plumbs correctly.

Searcheddef from_dict|def from_yaml|PredictConfig\(|_from_dict|fields\(|asdict|def load in work-projects/SegTask/segtask_v1/config.py (7)
Thought for 1s
Loader is field-name based with defaults — adding an optional field is safe. Let me set up the task plan and implement.

Created Todo List5 tasks
config.py
+8
Thought for 2s
The config field may have already existed (earlier search showed it). Let me verify there's no duplicate.

Searchedtta_batch_size in work-projects/SegTask/segtask_v1/config.py (3)
Thought for 3s
Only one definition — good. The earlier search snippet was anticipatory. Let me verify whether predictor.py/forwards.py already reference tta_batch_size or _adabn_estimating to avoid conflicts.

Searchedtta_batch_size|_adabn_estimating|_tta_chunk in work-projects/SegTask/segtask_v1/predictor (0)
No existing references. Now updating predictor.py — read tta_batch_size, add the _adabn_estimating flag, and wrap the per-volume AdaBN estimation.

predictor.py
+2
Thought for 1s
Now add the _adabn_estimating flag initialization and wrap the per-volume AdaBN estimation.

predictor.py
+5
Now wrap the per-volume AdaBN estimation block with the flag: