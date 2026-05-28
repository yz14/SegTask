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


测试环境为: **D:\miniconda\envs\torch27_env\python.exe**。  


这是我写的2.5D/3D分割项目代码D:\codes\work-projects\SegTask\README.md，训练入口在D:\codes\work-projects\SegTask\segtask_v1\train.py。这里有3个3D方案，z轴滑块（只在z轴滑动切块，x,y为全尺寸）；cubic滑块（在x,y,z轴滑动中心切块）；whole（直接输入整个图像）。一个2.5D方案，它和z轴滑块的单分辨率/感受野方案非常的相似，区别是：a 在train的时候，当数据增强结束后，将3D数据B,1,D,H,W变为B,D,H,W作为2D输入,D张切片代表D个通道；b 模型采用2D模型。计算损失也和现有框架一致，模型输出为B,num_fgxD,H,W然后拆分为num_fg个B,D,H,W单标签预测，各自计算单标签损失。这里有一份小数据集作为测试：F:\med_data\Totalsegmentator_dataset_v201\small_data\nii，F:\med_data\Totalsegmentator_dataset_v201\small_data\mask，
F:\med_data\Totalsegmentator_dataset_v201\small_data\bbox，
F:\med_data\Totalsegmentator_dataset_v201\small_data\region_weihgt。  


# TODO  
1 2.5D用这个配置训练D:\codes\work-projects\SegTask\configs\seg2_5d.yaml，我已经开了多分辨率输入，但是训练出来的模型和单分辨率没有什么提升，尤其是z轴的空间信息感觉很差劲，感觉模型是一张切片一张切片的在单独处理，而无法联系输入的所有的切片，所以这个2.5D感觉根本没有起到2.5D的作用，似乎更像是单张切片的纯2D效果。请检查代码，是模型实现的有问题吗，还是哪里有问题呢？同时你有什么改进建议给出吗，例如针对这个配置，哪些参数可以做什么调整来改进效果。

2 为了探索上面1的问题，我用D:\codes\work-projects\SegTask\configs\seg3d.yaml进行了训练，发现它们的badcase一模一样，也就是说用3D的z-axis也无法解决1中的bad case。请你检查3D的全流程代码是否有问题，为什么换了模型，bad case一模一样而且现象也一模一样。以上的训练都是用了npz_dir: "F:/BaiduNetdiskDownload/lung_prep"这里的数据，同时分割骨头和肺，区域权重对肺的边缘加了很大的权重9，对肺外的一圈组织加了权重7，对骨头加了权重4，对hu和肺相近的地方加了权重14。

3 最后2的数据基础上，彻底检查一遍损失函数，是否都正确实现了，对应我这样的数据是否可行，是否有问题。

4 在val挑选模型的时候，目前就是dice, loss, surface dice，有时候对于大目标，dice很高不一定能反应问题，所以我增加了surface dice，还有没有其它可靠，反应多维度的信息的指标可以来挑选模型，同时不要引入太多的时间消耗，所有的指标必须只能在gpu算