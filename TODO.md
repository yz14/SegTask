# 核心原则

## 质量第一
- 宁可多花时间，也要保证代码质量
- 充分思考、分析后再动手实现
- 不要为了快速完成而牺牲代码质量

## 分步完成
- 如果当前对话无法完成所有功能，主动拆分为多轮对话
- 每轮只专注完成一个清晰的目标
- 不贪多，确保每一步都高质量完成

## 充分调研
- 如有需要，充分、彻底地搜索和调研
- 分析和掌握现有的高质量功能实现和算法
- 借鉴业界最佳实践，不要闭门造车

## 调试支持
- 如有需要，可以加入 debug/logging 函数辅助开发
- 通过日志输出帮助定位和解决问题
- 调试代码可在功能稳定后标注或移除

## 代码质量  
- 注意代码尽可能模块化设计，职责尽可能的分离，不要把所有代码写在一个文件里，不方便后续理解和维护  
- 注意代码的复用性，不要写重复的代码  

## 沟通规范
- **开始前**：说明你理解的任务目标和将遵守的规则
- **进行中**：如需拆分，明确告知本轮将完成什么
- **完成后**：总结本轮成果，说明后续计划（如有）  

测试环境为: **conda activate torch27_env**  

这是我写的2.5D/3D分割代码，训练入口在D:\codes\work-projects\SegTask\segtask_v1\train.py。这里有3个3D方案，z轴滑块（只在z轴滑动切块，x,y为全尺寸）；cubic滑块（在x,y,z轴滑动中心切块）；whole（直接输入整个图像）。一个2.5D方案，它和z轴滑块的单分辨率/感受野方案非常的相似，区别是：a 在train的时候，当数据增强结束后，将3D数据B,1,D,H,W变为B,D,H,W作为2D输入,D张切片代表D个通道；b 模型采用2D模型。计算损失也和现有框架一致，模型输出为B,num_fgxD,H,W然后拆分为num_fg个B,D,H,W单标签预测，各自计算单标签损失。这里有一份小数据集作为测试：F:\med_data\Totalsegmentator_dataset_v201\small_data\nii，F:\med_data\Totalsegmentator_dataset_v201\small_data\mask，
F:\med_data\Totalsegmentator_dataset_v201\small_data\bbox，
F:\med_data\Totalsegmentator_dataset_v201\small_data\region_weihgt。  


# TODO  
1. 目前的2.5D训练(D:\codes\work-projects\SegTask\configs\seg2_5d.yaml)肺分割感觉上下文信息能力有点弱。我想针对2.5D也加入多级感受野的输入，我初步设想是这样的，假设当前模型输入为B,D,H,W，那么增加B,1.5D,H,W和B,2D,H,W输入，给原始的B,D,H,W输入提供更多的信息。一种做法是对三个输入分别用3个stem处理，然后拼接，再卷积，再下一个stage。另一个做法是将B,1.5D,H,W和B,2D,H,W重采样到B,D,H,W然后用相同的stem，然后拼接，然后卷积，再后续的stage。还有一种做法是将B,1.5D,H,W和B,2D,H,W分别重采样到B,D,H/2,W/2和B,D,H/4,W/4，然后用相同的stem，然后B,D,H/2,W/2经过stem的特征拼接到B,D,H,W第一个stage下采样后的特征上，B,D,H/4,W/4经过stem的特征拼接到B,D,H,W第二个stage下采样后的特征上。这些做法可能需要对encoder的input_channel做适当修改（自动化判断和修改）。请你先评估这几个方案的可行性和可靠性，然后理解我这么设计的意义，最后给出一个最佳解决方案。

方案 A：三个独立 stem → 拼接 → 卷积 → 后续 stage
可行性：高。等价于"each-FOV 独立 patch embedding，然后 early fusion"。改造点在 Encoder.__init__：把单 stem 替换为 n_views 个 stem + 一个 1×1 / 3×3 融合卷积。in_channels 自动由 n_views * D 推导，需在 Config.sync() 增加新字段（如 data.context_z_scales）。
可靠性：好。每个 FOV 拥有独立的低层滤波器，可学到尺度专属纹理（细节切片 vs. 粗糙整体走向），文献上属于 Multi-Stream/Multi-Path CNN（参考 DeepMedic、Kamnitsas 2017）的成熟做法。
代价：参数量 ~3×stem；早期通道翻 3 倍后 1×1 压回 base，几乎不影响显存与 FLOPs（stem 在整网占比极小）。
不足：所有视野在最高分辨率上融合，网络无法"按层级"使用粗 FOV（粗 FOV 信息会在前几层就被压扁，下采样后可能稀释）。

方案 C：分层注入（1.5D → stage1 后特征，2D → stage2 后特征），低分辨率分支
可行性：中。需要为每个辅助 FOV 配一个独立 stem（输出空间 H/2,W/2 或 H/4,W/4 已与目标 stage 对齐），然后改 Encoder.forward：在第 i 个下采样后的特征上 cat（再 1×1 压回原通道），后续 stage 的 in_channels 也要相应调整。
可靠性：理论上最优。符合"粗 FOV 上下文应注入到深层、低分辨率特征"的视觉/医学影像层级假设——这是 HRNet、PSPNet、HoVer-Net、SegFormer、nnFormer 多尺度上下文聚合的核心思想。粗 FOV 特征在深层 stage 与原 1× FOV 已下采样的语义特征空间分辨率匹配、感受野互补，融合更"理所当然"。
代价：实现复杂度最高；需要为每个下采样 stage 维护"哪个外部分支 cat 进来"的注册表，以及自动调整下游 stage in_channels（你已经预见到了"自动化判断和修改"，这正是 C 方案的关键工程量）。Decoder/skip 通道不需要改（skip 来自融合后特征，对外 API 不变）。
不足：辅助分支的 stem 还是要独立学一个 (D, H/2, W/2) → C1 的映射；如果你以后想增减 FOV，配置 ↔ 模型同步逻辑要稳健。