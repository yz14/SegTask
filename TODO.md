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


4 trainer写的让人读起来非常的费劲，费劲的理由主要是，a 里面的判断条件非常的多，容易让人记混，b 里面关于多少个视图的参数非常容易让人混乱，例如在2.5D时，多分辨率时有几个分辨率，参数里面就是该数字，而当3D时，全部都设成1，这读起来非常容易让人混乱，c 里面代码太多了，没有功能模块化处理。我目前可以想到的是，先将trainer模块化重构，然后再写一个通用的trainer.py，然后在分别写trainer2_5d.py和trainer3d.py去继承trainer.py和调用公用的模块，这样应该可以少很多的判断条件，使得代码清晰。如果你有更好的方案，则不必严格遵守我的方案。以下是我的分析：  
1. 状态空间维度太多，且彼此正交叠加
Trainer.__init__ 在 @d:/codes/work-projects/SegTask/segtask_v1/trainer.py:251-402 里同时维护 6 个布尔/枚举开关，它们两两组合就能产生 ~20 条有效分支：

patch_mode: 2_5d / z_axis / cubic / whole（4 种几何）
is_2_5d: 上面 2_5d 的派生快捷
lift_2_5d_to_3d: 2.5D 子分支，强制 num_res=1
aux_keep_native_d: 2.5D 子分支，aux 视图保原生 D_k
keep_native_multi_res: 仅 3D（z_axis/cubic），懒抽取
aux_seg_supervision: 仅 2.5D + n_views>1
deep_supervision: 是否再包一层 DeepSupervisionLoss
后果：例如训练步 _train_epoch @d:/codes/work-projects/SegTask/segtask_v1/trainer.py:806-832 里出现了 4 路嵌套 if 分发输入 + 3 路损失分发 @d:/codes/work-projects/SegTask/segtask_v1/trainer.py:842-854，再叠上 aux=on/off、lift=on/off、native_d=on/off 的笛卡尔积。代码本身没错，但读者必须同时记住 6 维状态。

2. "视图/分辨率/切片"语义被三个词混用
你在 TODO 里点到了痛点 b。代码中实际共存四套含义不同的"数量"：

变量	含义	2.5D 单分辨率	2.5D 多分辨率	3D 单分辨率	3D 多分辨率
len(multi_res_scales)	数据侧视图数	1	n	1	n
num_res (loss)	损失复合的输出通道分组数	1 (slice)	1 (slice)	1	n
n_aux = n_views-1	aux 监督数	0	n-1	0	0
num_slices = D	切片通道数	D	D	–	–
C_res (channel)	模型输入分辨率轴	1	n	1	n
@d:/codes/work-projects/SegTask/segtask_v1/trainer.py:300 干脆注释 num_res = 1   # 仅日志用 —— 这一行就是"参数被强行复用导致语义滑动"的活证据。

3. 单类承担了太多职责（God Class）
Trainer 1659 行同时干了 9 件事：

A. 工厂: build_optimizer / build_scheduler / WarmupScheduler (@d:/codes/work-projects/SegTask/segtask_v1/trainer.py:83-227)
B. AMP/Scaler 适配: GradScaler shim、_resolve_auto_amp_dtype、_compute_loss_fp32（精度切换）
C. EMA + compile + 编译探测
D. 内存核算: _estimate_train_memory
E. 损失组装: 4 种 inner loss 选择 + DS 外包 + aux inner 选择
F. 数据视图重塑: _split_views_native_3d / _split_views_native_d / _squeeze_2_5d / _squeeze_2_5d_keep_views / _center_crop
G. 训练循环 + 梯度累积 + breakdown 收集
H. 验证循环 + pooled dice + surface dice
I. Checkpoint I/O + pretrain 加载
A、B、C、D、I 都与 2.5D/3D 无关，却挤在同一个文件里增加了"信号噪声比"。

4. 控制流局部正确但全局难审计
_train_epoch 的输入重塑、_compute_loss_* 的损失分派、_extract_main_pred 的输出解包各自独立判断同一组 flags。任何新增模式都要在 3-5 处保持同步。
你自己留的 # TODO 不太懂 @d:/codes/work-projects/SegTask/segtask_v1/trainer.py:834 也是这个症状：尾批梯度累积折算与上文的视图重塑耦合在同一个 for-loop 里。
二、重构蓝图
核心思想：用"策略对象 (Strategy / Pipeline)" 替代 flags
把"如何在一次 step 中把 dataset 给的 batch 变成 (model_input, supervision_pack)、再把 model_output 变成 loss + metrics"封装成一个 ViewPipeline 接口，按训练模式实例化一个具体子类。Trainer 本身不再判断模式。

text
ViewPipeline (ABC)
├── prepare_batch(batch) → ModelInputs, SupervisionPack
├── compute_loss(pred, sup) → loss, breakdown
├── extract_main_for_metrics(pred, sup) → (pred_1x, label_1x)
└── make_inner_loss(cfg) / make_aux_inner_loss(cfg)   # 注册期一次
具体子类（一一对应你的方案，但更细化）：

Whole3DPipeline — patch_mode=whole（单分辨率，无 aux，无重塑）
Patch3DSinglePipeline — z_axis/cubic，单分辨率（即 keep_native_multi_res=False）
Patch3DNativeMultiResPipeline — z_axis/cubic，keep_native_multi_res=True（含 _split_views_native_3d）
Slab2_5DPipeline — 2.5D 单分辨率/折叠（含 _squeeze_2_5d）
Slab2_5DAuxPipeline — 2.5D + aux_seg_supervision 且 aux_keep_native_d=False（含 _squeeze_2_5d_keep_views + _compute_loss_aux_fp32）
Slab2_5DNativeDPipeline — 2.5D + aux_keep_native_d=True（含 _split_views_native_d + _compute_loss_aux_native_d_fp32）
Lift2_5DPipeline — 2.5D + lift_2_5d_to_3d（含 [:, :1] 切片）
Lift2_5DAuxPipeline — 2.5D + lift + aux
各 pipeline 自己持有 inner_loss / aux_inner_loss(es) / aux_weights / target_patch_size 等"自己模式专属的"成员，Trainer 只持有一个 self.pipeline。

效果：_train_epoch 主体压到 ~30 行：augment → pipeline.prepare_batch → forward → pipeline.compute_loss → backward → step → metrics，再也看不到 if self.is_2_5d 之类的判断。


命名口径统一（解决 TODO 痛点 b）
在重构期把所有"几"统一为以下命名，禁止再复用：

n_views — 数据/模型几何视图数 = len(cfg.data.multi_res_scales)
n_aux_views = n_views - 1
num_res_groups — 损失里的"通道分组数"（2.5D=1，3D=n_views，lift=1）
slab_depth — 2.5D 的 D
aux_view_depths[k] — 仅 aux_keep_native_d 用
凡是当前把 1 个量"借给"另一个语义的写法（如 @d:/codes/work-projects/SegTask/segtask_v1/trainer.py:300）一律替换。

文件拆分建议
下一轮动手时建议拆成（位置仍在 segtask_v1/）：

trainer/
├── __init__.py              # 暴露 build_trainer(cfg, model, loaders) 工厂
├── trainer.py               # Trainer 主体 (~300 行)：fit / _train_epoch / _validate
├── optim.py                 # build_optimizer / build_scheduler / WarmupScheduler
├── amp.py                   # GradScaler shim / _resolve_auto_amp_dtype / _compute_loss_fp32
├── memory.py                # _estimate_train_memory + peak 日志
├── checkpoint.py            # _build_state_dict / save / load / pretrain / 前缀剥离
├── breakdown.py             # _collect_multi_res_breakdown / _format_breakdown
└── pipelines/
    ├── base.py              # ViewPipeline ABC + ModelInputs / SupervisionPack dataclass
    ├── factory.py           # 根据 cfg 选 pipeline（唯一的 if 集中地）
    ├── whole3d.py
    ├── patch3d.py           # 单 + native_multi_res 两子类
    ├── slab25d.py           # 折叠 + aux + aux_native_d 三子类
    └── lift25d.py           # lift + lift_aux
factory.py 是整个 codebase 中唯一允许大段 if/elif 的地方，且它只 ~20 行。其他文件里看不到模式判断。

数据结构：用 dataclass 给 SupervisionPack 上类型
text
@dataclass
class SupervisionPack:
    label_main:      Tensor              # 主监督
    wmap_main:       Tensor | None
    aux_labels:      list[Tensor] | None  # 仅 aux 路径
    aux_wmaps:       list[Tensor | None] | None
    label_all_views: Tensor | None        # 仅 lift+aux / squeeze_keep_views
    wmap_all_views:  Tensor | None
compute_loss(pred, sup) 内部按需取字段，调用方一目了然。当前那种"在 _train_epoch 里同时定义 label_all_views / aux_view_labels / aux_view_wmaps 三组 Optional 变量再传给某个分支" @d:/codes/work-projects/SegTask/segtask_v1/trainer.py:800-832 就消失了。