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

5 我在服务器上用D:\codes\work-projects\SegTask\configs\segtest0.yaml训练body分割，训练数据集在本地也有：F:\Totalsegmentator_dataset_v201\nii，然后训练完成后在F:\airway_segment_with_img\imgs这个数据集推理，我发现推理的数据集效果一般般，假阳很多。我检查了一遍代码，感觉发现不了什么问题，你可以检查确认一遍。如果代码没有问题，那么是不是数据发布的问题，如果有较大可能是数据问题，你可否写一个脚本来给我允许，分析两个数据集究竟有什么差异（首先可以确定的是spacing肯定有差异的，但是我都resize到同样的尺寸了）。
进度：我写了脚本分析脚本: @d:\codes\work-projects\SegTask\scripts\analyze_dataset_shift.py

全量运行命令:



powershell
D:\miniconda\envs\torch27_env\python.exe scripts/analyze_dataset_shift.py `
    --train-dir F:/Totalsegmentator_dataset_v201/nii `
    --infer-dir F:/airway_segment_with_img/imgs `
    --out-dir   scripts/dataset_shift_report `
    --workers 12

结果跑出来在：
Wrote scripts\dataset_shift_report\per_file_stats.csv (1930 rows)
2026-06-04 10:53:57,240 INFO Wrote scripts\dataset_shift_report\summary_stats.csv (36 rows)
2026-06-04 10:53:58,866 INFO Wrote scripts\dataset_shift_report\geometry_comparison.png
2026-06-04 10:53:59,358 INFO Wrote scripts\dataset_shift_report\intensity_comparison.png
2026-06-04 10:53:59,359 INFO ========================================================================
2026-06-04 10:53:59,359 INFO VERDICT — median comparison (train(totalseg) vs infer(airway))
2026-06-04 10:53:59,359 INFO ------------------------------------------------------------------------
2026-06-04 10:53:59,361 INFO   slice spacing Z                            train=    1.500 mm     infer=    1.250  (infer/train=0.83x)
2026-06-04 10:53:59,361 INFO   in-plane spacing X                         train=    1.500 mm     infer=    0.705  (infer/train=0.47x)
2026-06-04 10:53:59,362 INFO   in-plane FOV X                             train=  354.000 mm     infer=  363.000  (infer/train=1.03x)
2026-06-04 10:53:59,363 INFO   effective in-plane spacing after resize    train=    1.354 mm/vox infer=    1.418  (infer/train=1.05x)
2026-06-04 10:53:59,364 INFO   physical z-thickness of patch_D slab       train=   18.000 mm     infer=   15.000  (infer/train=0.83x)
2026-06-04 10:53:59,372 INFO   per-volume median intensity                train= -848.000 HU     infer= -896.000  (infer/train=1.06x)
2026-06-04 10:53:59,372 INFO   frac voxels below window                   train=    0.004        infer=    0.000  (infer/train=0.00x)
2026-06-04 10:53:59,373 INFO ========================================================================
2026-06-04 10:53:59,373 INFO 解读：eff_spacing_h / slab_z_mm 的 infer/train 比值越偏离 1.0，说明进模型前的解剖体素尺度差异越大——这是 z_axis/2.5D 无 spacing 重采样导致跨域假阳的直接量化。
2026-06-04 10:53:59,373 INFO Done. Report written to D:\codes\work-projects\SegTask\scripts\dataset_shift_report

目标与规则
目标:确认 segtest0.yaml(body 分割)跨数据集推理假阳偏多是代码问题还是数据问题,并解读已跑出的 analyze_dataset_shift.py 结果。
遵守规则:本轮只做调研与分析,不改实现代码;给出结论与改进计划,等你确认后再动手(规则一/二/三)。
一、代码核查结论:推理/预处理流程本身没有 bug
我核对了训练 dataset 与推理 predictor 的全链路,关键事实:

全程不做物理 spacing 重采样。preprocess_image 只做强度窗 + minmax 归一化(@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:247-279);resize_3d 是按体素索引比例缩放,不是按 mm(@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:314-330)。
推理侧 predict_volume 同样只 load_nifti + preprocess_image,无重采样(@d:\codes\work-projects\SegTask\segtask_v1\predictor\predictor.py:261-263);z_axis 走 _sliding_window_z,按体素取 patch_D 张切片。
配置里 z_interleave_enabled: false(@d:\codes\work-projects\SegTask\configs\segtest0.yaml:181),所以即使有 z-spacing 也不会做 z 交错对齐。
结论:代码逻辑自洽,训练/推理预处理完全一致,不是实现 bug。 但这套"无 spacing 重采样"的设计意味着模型学到的是训练集的体素级解剖尺度,跨域推理对 spacing/强度分布天然敏感——这正是假阳的根源所在。所以是数据域漂移问题,而漂移之所以能直接伤害模型,是因为流程不做尺度归一。

二、报告结果解读(逐维度)
1. 面内尺度:已对齐,不是问题
指标	train(totalseg)	infer(airway)	infer/train
面内 spacing X (中位)	1.5 mm	0.705 mm	0.47x
面内 FOV X (中位)	354 mm	363 mm	1.03x
resize→256 后有效 spacing	1.354 mm/vox	1.418 mm/vox	1.05x
原始 spacing 差 2 倍,但物理 FOV 基本一致,resize 到 256 后有效像素尺度只差 5%。面内被 resize 自动归一了,不是假阳主因。

2. Z 轴尺度:真实漂移,主要几何因素
指标	train	infer	infer/train
slice spacing Z (中位)	1.5 mm	1.25 mm	0.83x
patch_D=12 slab 物理厚度 (中位)	18 mm	15 mm	0.83x
模型在 ~18mm 的 z 上下文里训练,推理时一个 patch 只覆盖 ~15mm,z 解剖语境被压缩 ~17%。
更关键的是方差:infer 的 slab_z_mm q05=8.4mm(train 紧贴 18mm),21.2%(150/707)的 infer 卷 slab<12mm。z 上下文严重不一致 → 模型在薄层卷上看到的结构比例失真,易产生假阳。
3. 强度:整体直方图相近,但一大批 infer 卷的强度被污染
窗内平均直方图 train/infer 几乎重合(HU 标定大体正常),但原始统计暴露严重离群:

15.0%(106/707)的 infer 卷 raw_max>5000(最高到 int16 上限 32767)。
49.9%(353/707)raw_min<-2000(到 -32768)。
42.6%(301/707)的 infer 卷有 >10% 体素低于 -1024 窗下限(train 仅 5.7%);直方图上 infer 在 frac≈0.21 处有一个 train 完全没有的尖峰。
这说明 airway 数据里很大一部分卷的背景/填充值与 TotalSeg 不同(重建视野外大片空气被填成远低于 -1024 的值,或个别卷根本不是标准 HU)。窗裁剪到 [-1024,1024] 后这些区域全饱和成 0,在体内造成与训练分布不同的大面积"纯黑"边缘——模型没见过这种分布,正是假阳高发区。

三、根因排序
主因 A — Z 轴尺度/层厚漂移:slab 18→15mm 且方差大,21% 卷 slab<12mm。
主因 B — 强度/背景填充域漂移:~43% infer 卷有大比例窗下饱和,部分卷强度非标准 HU(±32768)。
底层设计:管线不做物理 spacing 重采样,放大了上述两类漂移的影响。
面内尺度已被排除。

四、改进计划(待你确认后执行)
我建议分步,每步独立可验证:

步骤 1(无需重训,先验证):开启 z-interleave 对齐 z 尺度。segtest0.yaml 已有 z_interleave_thresholds/factors,把 z_interleave_enabled 设为 true,在 airway 上重跑推理对比假阳是否下降。预期产出:同一批 case 的假阳量对比;验收:假阳明显减少则确认 Z 是主因之一。
步骤 2(数据侧定位):扩展现有分析脚本,列出"问题 infer 卷"清单(raw_max>5000 / frac_below>0.1),抽查几例确认是背景填充还是非 HU,决定是否需要重标定/清洗。验收:能区分"可修复填充值" vs "真·分布外"。
步骤 3(训练侧增强,需重训):针对 Z 漂移加 z-spacing/层厚增强(模拟薄层),针对强度加更鲁棒的背景处理;或评估是否引入物理 spacing 重采样到固定 mm/vox。验收:在 airway 上 Dice/假阳改善且不损 TotalSeg 内验证。


我采取的措施：
predict:
  z_interleave_enabled: true
  z_interleave_thresholds: [1.1, 1.6, 2.1, 2.6]
  z_interleave_factors: [5, 4, 3, 2, 1]
像这样改了后，假阳确实减少了（减少70%），但是仍然有30%假阳。之后我开启了TTA，假阳只剩10%不到。patch_mode: "2_5d"，所以走的是 @d:\codes\work-projects\SegTask\segtask_v1\predictor\forwards.py:67-90 的 tta_flip_ensemble_2_5d，而不是 3D 那套 7 组合。
具体动作：
只翻转 H / W 两个面内轴，做 3 种组合：H、W、H+W（@forwards.py:74-78）。

我后续还可能：
测试时 AdaBN（自适应 BN）：推理前用目标域几卷重新估计 BatchNorm 的 running stats（model.train() 跑几个 forward 不回传，或重算 BN buffer）。针对 BatchNorm 域敏感，无需标签、不重训，常有立竿见影效果。

请你评估这个AdaBN方案是否可行，如果可以，则可以拟定实施计划（在推理配置加开关是否启用）