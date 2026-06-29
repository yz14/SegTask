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


segtask_v1是2.5D/3D分割项目代码D:\codes\work-projects\SegTask\README.md；训练入口在D:\codes\work-projects\SegTask\segtask_v1\train.py。
这里有3个3D方案，z轴滑块（只在z轴滑动切块，x,y为全尺寸）；cubic滑块（在x,y,z轴滑动中心切块）；whole（直接输入整个图像）和1个2.5D方案，它和z轴滑块的单分辨率/感受野方案非常的相似，区别是：a 在train的时候，当数据增强结束后，将3D数据B,1,D,H,W变为B,D,H,W作为2D输入,D张切片代表D个通道；b 模型采用2D模型。
计算损失统一为模型输出为B,num_fgxD,H,W然后拆分为num_fg个B,D,H,W单标签预测，各自计算单标签损失。
这里有一份小数据集作为测试：F:\med_data\Totalsegmentator_dataset_v201\small_data\nii，F:\med_data\Totalsegmentator_dataset_v201\small_data\mask，
F:\med_data\Totalsegmentator_dataset_v201\small_data\bbox，
F:\med_data\Totalsegmentator_dataset_v201\small_data\region_weihgt。  
数据流：只接受npz输入，所有多分辨率方案必须都只取max FOV后，待数据增强结束后通过中心截取制作成多分辨率。  


gentask是基于segtask_v1是适配的超分项目。模型只能是2D，自然图像公认的经典、关键、高质量超分算法（SISR/VFI）。数据流只接受npz输入（必须先制作好）。


# TODO  
1 修改和优化gentask里面的代码，这里的代码主要用于医学影像超分，先针对CT超分进行修改、适配和优化，主要沿用分割的2.5D方案：  
image_dir: 低分辨率输入可能缺失，label_dir: 高分辨率GT，bbox_dir : 可能有，region_weight_dir : 可能有，npz_dir: 制作好的路径  

a 厚层生成薄层  
GT是薄层，对于的厚层可能缺失，需要针对性的用数据增强算法从薄层模拟出厚层（一张厚层切片是层厚范围内体素的平均(部分容积效应)）  

输入可以是对厚层的z轴滑块，例如取连续两张切片作为起始和终点，然后模型往中间插切片2x,4x，类似这样，模型对应的可能需要采用VFI；输入还可以是在厚层的x或者y进行滑块，由于只有z轴是低分辨率，所以模型对应的可能采用SISR只对z轴2x,4x，类似这样超分。  

输入可以输入更多FOV来帮助模型，例如薄层是1,2,3,4,5,6,7切片，厚层是1,3,5,7，对z轴滑块的基本输入是，例如3,5(1x FOV)->3,4,5，多FOV就是3,5(1x)+1,3,5,7(2x)->3,4,5；如果是在x轴取切片，那么输入就和分割的2.5D方案非常类似，例如C(x轴),H(y),w(z)->C,H,W，在x轴上取更多的切片，例如Cx1.5，Cx2个切片作为多个FOV  

数据读取部分几乎可以沿用分割任务中2.5D方案，例如保留oversample这样数据增强后边界的异常可以通用中心剪裁去掉；只在数据增强结束后才做成2D数据等等；我理解的就是把输入图像换成低分辨率，标签换成高分辨率，区域权重可能是重要的器官等等；只对oversample的max FOV处理，然后中心剪裁出多FOV等等。    

模型对应的用自然图像公认的经典、关键、高质量超分算法（SISR/VFI）；安装分割的代码风格来写(虽然不能统一UNet框架，但是构造模型的过程的block, stage, encoder, decoder这些可以参考)。  

输出对应的就是插帧/超分。  

需要针对性的适配各种技巧（例如在分割中有很多的技巧，例如EMA，deep supervision, 多个输出头来辅助监督等等）

b 面内超分（x,y是低分辨率，D,h,w -> D,H,W）
这个更加直观，也更加容易在分割的2.5D方案上适配(类似a中的在x轴取切片，然后超分z轴，只不过现在是在z轴取切片然后需要同时超分x,y轴)，和自然图像任务最接近。


2 我打算做SSL专题，见ssl.md这个计划文档。我目前已经实现了分割任务segtask_v1和生成任务gentask的大致框架。请你分析，我这个SSL是分别加入到分割和生成任务中，还是单独做一个ssltask任务，然后给分割和分类来通用（我倾向后者）。请你分析，理解我的功能和代码，然后设计SSL的架构。我已完成部分，见下文。


计划阶段（P1–P6）完成度
阶段	内容	状态
P1	脚手架 + 骨干契约 + image-only 数据 + 通用 trainer + 注册表 + genesis/prior 迁入	完成
P2	SimMIM② + 掩码工具 + MIM 头 + 在线探针	完成
P3	DINO④（多裁剪+EMA 教师+center/sharpen）	方法完成；对比基线 B1 未做
P4	SparK①（掩码-稠密等价 + 层次解码器）	完成
P5	JEPA⑦、iBOT⑥、DINO+Gram⑤、SparK+DINO⑧	完成
P6	评测管线：离线线性探针 + 少样本曲线 + B2 from-scratch + 分类(D3)	未做
方法注册表（@/d:/codes/work-projects/SegTask/ssltask/methods/__init__.py:24）
9 个方法均已注册、配置校验（config.py:35 的 METHODS）、各带 YAML（configs/ssltask_*.yaml 共 9 份）并有测试：

genesis③ / prior：破坏→重建 / Frangi vesselness 回归
simmim②：mask-token 稠密 MIM + 极简头
spark①：掩码-稠密等价 + 轻量层次解码器
dino④：多裁剪 + EMA 教师 + center/sharpen + 温度·动量 cosine 调度
dino_gram⑤：DINO + Gram anchoring（含 Gram 教师快照刷新）
jepa⑦：隐空间掩码预测 + EMA 目标编码器 + 可选 VICReg
ibot⑥：DINO 全局 + iBOT 掩码密集特征（共享/独立原型头）
sparkdino⑧：共享 encoder 双分支（SparK 重建 + DINO 蒸馏）
测试覆盖（test_ssltask.py，1759 行）
每个方法均覆盖 4 类测试：loss+backward（梯度到学生、教师冻结）、encoder-only 交接（strict=False 仅命中 encoder.*）、2.5D 形状、单 epoch CPU smoke；外加掩码/densify/multicrop/vesselness/probe 等工具单测。各方法的 handoff 契约（导出键与 build_model 同名）均有断言保证。

关键缺口（未完成项）
D3 分类下游缺失 — 现有 @/d:/codes/work-projects/SegTask/ssltask/eval/probe.py 仅是分割线性探针，且显式 3D-only（probe.py:105 对 2.5D 抛错）。SSL.md §0.4 要求的「encoder + 全局池化 + MLP 头 + 冻结/微调」分类探针与简单分类 trainer 完全没有。
P6 离线评测管线缺失 — 无「标注量 vs 性能」少样本曲线（10/30/50/100）、无线性探针 vs 全参微调两种读数、无 Dice/HD95 + AUC/F1 汇总落盘。当前只有训练中的在线探针（§0.5）。
B1 基线（BYOL/MoCo）缺失 — multicrop.py 已为其预留共享设施，但无方法实现/注册。
B2 from-scratch 未形式化 — 概念上等同于「不加 train.pretrain 直训」，但没有纳入统一对照管线产出对照读数。