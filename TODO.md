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


我的计划和进展：  

P1 脚手架 + 骨干契约 + image-only 数据 + 通用 trainer + 方法注册表 + 迁入③

产出：ssltask 包、make_ssl_data.py+SSLDataset、SSLMethod ABC、genesis 方法。
验收：genesis 预训练 smoke 通过；ckpt 仅含 encoder.*/decoder.*/recon_head.*；seg train.pretrain 干净加载（移植 handoff 测试）。
依赖：D1/D2/D4。
P2 SimMIM② + 掩码工具 + MIM 解码/预测头 + 在线探针

验收：smoke 通过；探针曲线落盘；handoff 测试通过。依赖：P1。
P3 DINO④（多裁剪+EMA 教师+center/sharpen）+ 对比基线 B1

验收：多裁剪 collate 正确；教师动量/温度调度生效；防坍缩有效；handoff 通过。依赖：P1。
P4 SparK①（掩码-稠密等价）+ 轻量层次解码器

验收：掩码在各尺度一致、信息不泄漏（单测）；满密度退化为稠密；handoff 通过。依赖：P2。
P5 隐空间/混合：JEPA⑦、iBOT⑥、DINO+Gram⑤、SparK+DINO⑧

验收：各方法 smoke + 防坍缩信号 + handoff。依赖：P2/P3/P4。
P6 评测管线：离线线性探针 + 少样本曲线 + B2 from-scratch（+ 视 D3 接分类）

验收：在小数据上产出“标注量 vs 性能”曲线与 B2 对照。依赖：P1（+ D3）。


现有 segtask_v1 已埋了一套可用的 genesis/prior SSL 与经测试的权重交接契约，但数据是标签耦合的、且无分类任务，无法支撑 SSL.md 的无标注大语料对比。我推荐独立 ssltask 并 import segtask_v1 骨干工厂（唯一骨干真相源、复杂度隔离、零侵入交接），用 SSLMethod 注册表承载 8 方法 + 2 基线，并补 image-only 数据通路与 §0.4/0.5 评测探针。


D2 立刻迁入：P1 会把 segtask_v1 的 SSL 全量迁入 ssltask 并从 seg 主路径下线。受影响项需一并处理：pretrain.py、ssl_trainer.py、ssl.py、ssl_transforms.py、vesselness.py、config.py 的 SSLConfig+_validate_ssl 接线、ssl_genesis.yaml/ssl_prior.yaml、以及 test_ssl.py（迁移并改 import）。迁后 segtask_v1 仅保留 train.pretrain 加载契约（不变）。
D3 分类现在就做：P6 增设最小分类下游（encoder + 全局池化 + MLP 头 + 简单 trainer + 冻结/微调探针），使 §0.4 评测覆盖 seg+cls；分类骨干仍由同一 build_model 工厂搭，沿用同一 train.pretrain 交接。


现状理解（含两处关键缺口：数据标签耦合、无分类任务）、核心架构决策（独立 ssltask + import segtask_v1 骨干工厂，唯一骨干真相源、复杂度隔离、零侵入交接）、SSL 架构设计（SSLMethod 接口+注册表、image-only 数据通路、各方法 CNN 落点、配置体系、§0.4/0.5 评测探针）、以及 6 阶段可独立执行计划。

已锁定决策：D1 独立 ssltask ✔ ；D2 genesis/prior 立刻迁入并下线 ✔ ；D3 本专题补最小分类下游 ✔ ；D4 SparK 掩码-稠密、无新依赖 ✔ ；D5 方法顺序 ②①③④→⑤⑦→⑥⑧→B1B2 ✔ 。

调研支撑：SparK 官方可 pretrain 任意 CNN、无需 spconv（掩码-稠密等价）；JEPA-on-CNN 有 CNN-JEPA/Seg-JEPA 先例——两处最高风险点已被现有实践背书，符合规则四“以最新官方做法为准”。


请你分析，理解和掌握后，按照你认为最优的计划推进。如果需要借用segtask_v1里面代码的可以直接拷贝py文件后修改调整，不要自己重新造轮子，除非你有更好的重构方案。如果你有更好的方案则可以对以上计划做优化和调整。


剩余首批仅差 ① SparK-3D（掩码-稠密等价 + 层次解码器，技术风险最高项），之后进入 ⑤ DINO+Gram / ⑦ JEPA（可复用本轮的 EMA 教师/多裁剪/投影头）。