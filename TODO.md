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


2 多卡训练3D：
 python -m segtask_v1.train --config configs/segtest1.yaml
[2026-06-25 18:31:21] INFO __mp_main__: DDP launched: world_size=4 on physical GPUs [1, 3, 4, 5] (MASTER_PORT=52701).
[2026-06-25 18:31:21] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-25 18:31:21] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-25 18:31:21] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-25 18:31:21] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-25 18:31:32] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-25 18:31:32] INFO segtask_v1.data.specs: Using CUBIC patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], max_scale=2.00) — SINGLE max-FOV cube extraction; trainer crops+resizes per view before the 3D forward.
[2026-06-25 18:31:32] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 88 npz packages...
[2026-06-25 18:31:36] INFO segtask_v1.data.dataset: NPZ cubic index: 88 volumes, 4400000 fg voxels sampled
[2026-06-25 18:31:36] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 22 npz packages...
[2026-06-25 18:31:37] INFO segtask_v1.data.dataset: NPZ cubic index: 22 volumes, 1100000 fg voxels sampled
[2026-06-25 18:31:37] INFO segtask_v1.data.loader: DDP DistributedSampler: rank=0/4, ~176 samples/rank (train).
[2026-06-25 18:31:37] INFO segtask_v1.data.loader: DataLoader: batch_size=1, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-25 18:31:37] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=48, num_workers=16 => up to ~174.79 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-25 18:31:37] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=all, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-25 18:31:37] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=65.60M, dec=59.78M, total=134.67M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=3 (fg=1, res=3), stem=dual(stride=1, n_views=1, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=True, ds=True, aux_seg=False(n_aux_heads=0, mode=conv)
[2026-06-25 18:31:38] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Patch3DNativeMultiResPipeline (patch_mode=cubic, n_views=3)
[2026-06-25 18:31:38] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Patch3DNativeMultiResPipeline (patch_mode=cubic, n_views=3)
[2026-06-25 18:31:38] WARNING segtask_v1.visualization.model_flow: model_flow: 数据流追踪失败，退化为纯结构图: Module.register_forward_pre_hook() got an unexpected keyword argument 'with_kwargs'
[2026-06-25 18:31:38] INFO segtask_v1.visualization: Pipeline visualization HTML written: outputs/ves_multirf3d/visualization/pipeline_vis.html
[2026-06-25 18:31:38] INFO __mp_main__: Pipeline visualization written to: outputs/ves_multirf3d/visualization/pipeline_vis.html
[2026-06-25 18:31:39] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Patch3DNativeMultiResPipeline (patch_mode=cubic, n_views=3)
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda:1).
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: DDP enabled: rank=0/4, device=cuda:1, find_unused_parameters=True. Training grads all-reduce per backward (math-equivalent to single-GPU under grad-accum).
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Validation metric mode: high (evaluator=VolumeValEvaluator)
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Training monitor enabled → metrics: outputs/ves_multirf3d/monitor | dashboard: outputs/ves_multirf3d/training_monitor.html
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Training: 1000 epochs, device=cuda:1
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Model params: 134.67M
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=513.7 + grad=513.7 + optim(AdamW,2x)=1027.5 + ema=513.7 = 2568.7 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=1575.1 MiB, reserved=2122.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Train batches: 176, Val batches: 88
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: AMP=False (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Grad accum=8, Effective batch=8
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: Pipeline=Patch3DNativeMultiResPipeline | n_views=3, n_aux_views=0, num_res_groups=3, slab_depth=0 | fg_classes=1, Loss=dice_focal
[2026-06-25 18:31:39] INFO segtask_v1.trainer.trainer: ============================================================
[W reducer.cpp:1298] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1298] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1298] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1298] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[2026-06-25 18:32:36] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 21927.9 MiB (forward + backward + optimizer.step + EMA update; accum=8 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
WARNING:segtask_v1.predictor.predictor:Unknown amp_dtype='auto', falling back to bfloat16.
[2026-06-25 18:36:39] INFO segtask_v1.predictor.predictor: Predictor keep_native_multi_res=True (cubic): per-view native sizes=[(96, 112, 112), (144, 168, 168), (192, 224, 224)], max-FOV target=(192, 224, 224), n_views=3.
[2026-06-25 18:36:39] WARNING segtask_v1.predictor.predictor: Unknown amp_dtype='auto', falling back to bfloat16.
WARNING:segtask_v1.predictor.predictor:Unknown amp_dtype='auto', falling back to bfloat16.
WARNING:segtask_v1.predictor.predictor:Unknown amp_dtype='auto', falling back to bfloat16.
[2026-06-25 19:16:29] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0563, per_class=['0.0563'], iou=0.0290, recall=0.8434, precision=0.0291, vol_sim=0.0668, mcc=0.0544, min_class_dice=0.0563, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0300, per_class_sd=['0.0300'], combined(w=0.50)=0.0432, balanced=0.0457
[2026-06-25 19:16:58] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf3d/best_model.pth
[2026-06-25 19:16:58] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0457 at epoch 1
[2026-06-25 19:16:58] INFO segtask_v1.trainer.trainer: Epoch 1/1000 | LR=3.48e-06 | loss=0.5639 | val_dice=0.0563 | best=0.0457 (ep1) | 00:45:19 | L_res_0=0.4650 L_res_1=0.5791 L_res_2=0.6593
[2026-06-25 19:16:58] INFO segtask_v1.trainer.trainer:   Phase time (epoch 1): train=00:04:59 | val=00:39:50 | val=88.9% of (train+val)
[2026-06-25 19:16:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 21927.9 MiB
[2026-06-25 20:02:36] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0569, per_class=['0.0569'], iou=0.0293, recall=0.9476, precision=0.0293, vol_sim=0.0600, mcc=0.0684, min_class_dice=0.0569, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0210, per_class_sd=['0.0210'], combined(w=0.50)=0.0389, balanced=0.0395
[2026-06-25 20:02:36] INFO segtask_v1.trainer.trainer: Epoch 2/1000 | LR=5.95e-06 | loss=0.4968 | val_dice=0.0569 | best=0.0457 (ep1) | 01:30:57 | L_res_0=0.4143 L_res_1=0.5130 L_res_2=0.5898
[2026-06-25 20:02:36] INFO segtask_v1.trainer.trainer:   Phase time (epoch 2): train=00:05:56 | val=00:39:41 | val=87.0% of (train+val)
[2026-06-25 20:02:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 15264.7 MiB
[2026-06-26 08:56:27] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0544, per_class=['0.0544'], iou=0.0280, recall=0.9889, precision=0.0280, vol_sim=0.0550, mcc=0.0656, min_class_dice=0.0544, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0137, per_class_sd=['0.0137'], combined(w=0.50)=0.0341, balanced=0.0311
[2026-06-26 08:56:27] INFO segtask_v1.trainer.trainer: Epoch 3/1000 | LR=8.42e-06 | loss=0.4441 | val_dice=0.0544 | best=0.0457 (ep1) | 14:24:47 | L_res_0=0.3747 L_res_1=0.4611 L_res_2=0.5494
[2026-06-26 08:56:27] INFO segtask_v1.trainer.trainer:   Phase time (epoch 3): train=10:38:43 | val=02:15:07 | val=17.5% of (train+val)
[2026-06-26 08:56:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 15265.1 Mi

Message from syslogd@imedway at Jun 26 08:58:12 ...
 kernel:[2048273.043637] watchdog: BUG: soft lockup - CPU#0 stuck for 23s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 08:59:43 ...
 kernel:[2048365.974103] watchdog: BUG: soft lockup - CPU#1 stuck for 24s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:00:09 ...
 kernel:[2048392.650811] watchdog: BUG: soft lockup - CPU#1 stuck for 49s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:00:45 ...
 kernel:[2048428.743766] watchdog: BUG: soft lockup - CPU#1 stuck for 82s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:01:13 ...
 kernel:[2048456.728510] watchdog: BUG: soft lockup - CPU#1 stuck for 108s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:01:40 ...
 kernel:[2048484.437247] watchdog: BUG: soft lockup - CPU#1 stuck for 134s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:02:22 ...
 kernel:[2048524.398308] watchdog: BUG: soft lockup - CPU#0 stuck for 23s! [python:3745298]

Message from syslogd@imedway at Jun 26 09:03:41 ...
 kernel:[2048605.416454] watchdog: BUG: soft lockup - CPU#24 stuck for 22s! [python:3745299]

Message from syslogd@imedway at Jun 26 09:06:49 ...
 kernel:[2048792.957428] watchdog: BUG: soft lockup - CPU#8 stuck for 23s! [python:3744821]

2.5D训练：
 python -m segtask_v1.train --config configs/segtest0.yaml
[2026-06-25 18:49:34] INFO __mp_main__: DDP launched: world_size=2 on physical GPUs [6, 7] (MASTER_PORT=58155).
[2026-06-25 18:49:34] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-25 18:49:34] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-25 18:49:34] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-25 18:49:34] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-25 18:49:46] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-25 18:49:46] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-25 18:49:46] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-25 18:49:49] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-25 18:49:49] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-25 18:49:50] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-25 18:49:50] INFO segtask_v1.data.loader: DDP DistributedSampler: rank=0/2, ~352 samples/rank (train).
[2026-06-25 18:49:50] INFO segtask_v1.data.loader: DataLoader: batch_size=4, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-25 18:49:50] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-25 18:49:50] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-25 18:49:50] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.28M, total=47.04M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-25 18:49:50] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-25 18:49:50] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-25 18:49:50] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-25 18:49:50] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-25 18:49:50] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-25 18:49:50] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-25 18:49:50] WARNING segtask_v1.visualization.model_flow: model_flow: 数据流追踪失败，退化为纯结构图: Module.register_forward_pre_hook() got an unexpected keyword argument 'with_kwargs'
[2026-06-25 18:49:50] INFO segtask_v1.visualization: Pipeline visualization HTML written: outputs/ves_multirf2d/visualization/pipeline_vis.html
[2026-06-25 18:49:50] INFO __mp_main__: Pipeline visualization written to: outputs/ves_multirf2d/visualization/pipeline_vis.html
[2026-06-25 18:49:52] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-25 18:49:52] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-25 18:49:52] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda:6).
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: DDP enabled: rank=0/2, device=cuda:6, find_unused_parameters=True. Training grads all-reduce per backward (math-equivalent to single-GPU under grad-accum).
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Validation metric mode: high (evaluator=VolumeValEvaluator)
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Training monitor enabled → metrics: outputs/ves_multirf2d/monitor | dashboard: outputs/ves_multirf2d/training_monitor.html
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Training: 1000 epochs, device=cuda:6
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Model params: 47.04M
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.4 + grad=179.4 + optim(AdamW,2x)=358.9 + ema=179.6 = 897.3 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=546.0 MiB, reserved=738.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Train batches: 88, Val batches: 22
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Grad accum=2, Effective batch=8
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_focal
[2026-06-25 18:49:52] INFO segtask_v1.trainer.trainer: ============================================================
[W reducer.cpp:1298] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1298] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[2026-06-25 18:50:22] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 6729.5 MiB (forward + backward + optimizer.step + EMA update; accum=2 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-25 18:51:35] INFO segtask_v1.predictor.predictor: Predictor keep_native_view_depth=True: per-view depths=[12, 18, 24], max-FOV cube depth=24, in_channels=54.
[2026-06-25 18:51:35] WARNING segtask_v1.predictor.predictor: Unknown amp_dtype='auto', falling back to bfloat16.
WARNING:segtask_v1.predictor.predictor:Unknown amp_dtype='auto', falling back to bfloat16.
[2026-06-25 18:52:14] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0440, per_class=['0.0440'], iou=0.0225, recall=0.5000, precision=0.0230, vol_sim=0.0881, mcc=-0.0030, min_class_dice=0.0440, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0804, per_class_sd=['0.0804'], combined(w=0.50)=0.0622, balanced=0.0490
[2026-06-25 18:52:15] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf2d/best_model.pth
[2026-06-25 18:52:15] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0490 at epoch 1
[2026-06-25 18:52:15] INFO segtask_v1.trainer.trainer: Epoch 1/1000 | LR=1.09e-05 | loss=1.6334 | val_dice=0.0440 | best=0.0490 (ep1) | 00:02:23 | L_main=0.8200 L_aux_1=0.8173(w=0.5) L_aux_2=0.8095(w=0.5)
[2026-06-25 18:52:15] INFO segtask_v1.trainer.trainer:   Phase time (epoch 1): train=00:01:43 | val=00:00:39 | val=27.4% of (train+val)
[2026-06-25 18:52:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 7098.6 MiB
[2026-06-25 18:53:24] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0441, per_class=['0.0441'], iou=0.0226, recall=0.5000, precision=0.0231, vol_sim=0.0882, mcc=-0.0027, min_class_dice=0.0441, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0803, per_class_sd=['0.0803'], combined(w=0.50)=0.0622, balanced=0.0491
[2026-06-25 18:53:28] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf2d/best_model.pth
[2026-06-25 18:53:28] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0491 at epoch 2
[2026-06-25 18:53:28] INFO segtask_v1.trainer.trainer: Epoch 2/1000 | LR=2.08e-05 | loss=1.5535 | val_dice=0.0441 | best=0.0491 (ep2) | 00:03:35 | L_main=0.7579 L_aux_1=0.7971(w=0.5) L_aux_2=0.7940(w=0.5)
[2026-06-25 18:53:28] INFO segtask_v1.trainer.trainer:   Phase time (epoch 2): train=00:00:29 | val=00:00:38 | val=57.0% of (train+val)
[2026-06-25 18:53:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 7279.9 MiB
[2026-06-25 18:54:37] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0442, per_class=['0.0442'], iou=0.0226, recall=0.5000, precision=0.0231, vol_sim=0.0884, mcc=-0.0025, min_class_dice=0.0442, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0805, per_class_sd=['0.0805'], combined(w=0.50)=0.0623, balanced=0.0492
[2026-06-25 18:54:41] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf2d/best_model.pth
[2026-06-25 18:54:41] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0492 at epoch 3
[2026-06-25 18:54:41] INFO segtask_v1.trainer.trainer: Epoch 3/1000 | LR=3.07e-05 | loss=1.4045 | val_dice=0.0442 | best=0.0492 (ep3) | 00:04:49 | L_main=0.6686 L_aux_1=0.7336(w=0.5) L_aux_2=0.7381(w=0.5)
[2026-06-25 18:54:41] INFO segtask_v1.trainer.trainer:   Phase time (epoch 3): train=00:00:30 | val=00:00:38 | val=56.0% of (train+val)
[2026-06-25 18:54:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 7279.9 MiB
[2026-06-25 18:55:51] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0442, per_class=['0.0442'], iou=0.0226, recall=0.5000, precision=0.0231, vol_sim=0.0884, mcc=-0.0025, min_class_dice=0.0442, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0804, per_class_sd=['0.0804'], combined(w=0.50)=0.0623, balanced=0.0492
[2026-06-25 18:55:51] INFO segtask_v1.trainer.trainer: Epoch 4/1000 | LR=4.06e-05 | loss=1.2989 | val_dice=0.0442 | best=0.0492 (ep3) | 00:05:59 | L_main=0.6065 L_aux_1=0.6855(w=0.5) L_aux_2=0.6992(w=0.5)
[2026-06-25 18:55:51] INFO segtask_v1.trainer.trainer:   Phase time (epoch 4): train=00:00:30 | val=00:00:39 | val=55.9% of (train+val)
[2026-06-25 18:55:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 4): 7279.9 MiB
[2026-06-25 18:56:59] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0442, per_class=['0.0442'], iou=0.0226, recall=0.4303, precision=0.0233, vol_sim=0.1027, mcc=-0.0011, min_class_dice=0.0442, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0798, per_class_sd=['0.0798'], combined(w=0.50)=0.0620, balanced=0.0491
[2026-06-25 18:56:59] INFO segtask_v1.trainer.trainer: Epoch 5/1000 | LR=5.05e-05 | loss=1.2867 | val_dice=0.0442 | best=0.0492 (ep3) | 00:07:07 | L_main=0.5954 L_aux_1=0.6854(w=0.5) L_aux_2=0.6972(w=0.5)
[2026-06-25 18:56:59] INFO segtask_v1.trainer.trainer:   Phase time (epoch 5): train=00:00:30 | val=00:00:37 | val=55.7% of (train+val)
[2026-06-25 18:56:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 5): 7279.9 MiB
[2026-06-25 18:58:06] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0445, per_class=['0.0445'], iou=0.0228, recall=0.4998, precision=0.0233, vol_sim=0.0891, mcc=-0.0011, min_class_dice=0.0445, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0742, per_class_sd=['0.0742'], combined(w=0.50)=0.0594, balanced=0.0489
[2026-06-25 18:58:06] INFO segtask_v1.trainer.trainer: Epoch 6/1000 | LR=6.04e-05 | loss=1.2044 | val_dice=0.0445 | best=0.0492 (ep3) | 00:08:13 | L_main=0.5451 L_aux_1=0.6524(w=0.5) L_aux_2=0.6662(w=0.5)
[2026-06-25 18:58:06] INFO segtask_v1.trainer.trainer:   Phase time (epoch 6): train=00:00:29 | val=00:00:37 | val=56.6% of (train+val)
[2026-06-25 18:58:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 6): 7279.9 MiB
[2026-06-25 18:59:12] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0427, per_class=['0.0427'], iou=0.0218, recall=0.3337, precision=0.0228, vol_sim=0.1280, mcc=-0.0032, min_class_dice=0.0427, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0814, per_class_sd=['0.0814'], combined(w=0.50)=0.0621, balanced=0.0479
[2026-06-25 18:59:12] INFO segtask_v1.trainer.trainer: Epoch 7/1000 | LR=7.03e-05 | loss=1.1525 | val_dice=0.0427 | best=0.0492 (ep3) | 00:09:20 | L_main=0.5122 L_aux_1=0.6346(w=0.5) L_aux_2=0.6460(w=0.5)
[2026-06-25 18:59:12] INFO segtask_v1.trainer.trainer:   Phase time (epoch 7): train=00:00:28 | val=00:00:37 | val=56.6% of (train+val)
[2026-06-25 18:59:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 7): 7279.9 MiB
[2026-06-25 19:00:20] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0435, per_class=['0.0435'], iou=0.0222, recall=0.3337, precision=0.0232, vol_sim=0.1302, mcc=-0.0011, min_class_dice=0.0435, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0822, per_class_sd=['0.0822'], combined(w=0.50)=0.0628, balanced=0.0487
[2026-06-25 19:00:20] INFO segtask_v1.trainer.trainer: Epoch 8/1000 | LR=8.02e-05 | loss=1.0701 | val_dice=0.0435 | best=0.0492 (ep3) | 00:10:28 | L_main=0.4610 L_aux_1=0.6041(w=0.5) L_aux_2=0.6141(w=0.5)
[2026-06-25 19:00:20] INFO segtask_v1.trainer.trainer:   Phase time (epoch 8): train=00:00:29 | val=00:00:37 | val=56.2% of (train+val)
[2026-06-25 19:00:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 8): 7279.9 MiB
[2026-06-25 19:01:27] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0440, per_class=['0.0440'], iou=0.0225, recall=0.3337, precision=0.0236, vol_sim=0.1320, mcc=0.0004, min_class_dice=0.0440, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0833, per_class_sd=['0.0833'], combined(w=0.50)=0.0636, balanced=0.0493
[2026-06-25 19:01:31] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf2d/best_model.pth
[2026-06-25 19:01:31] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0493 at epoch 9
[2026-06-25 19:01:31] INFO segtask_v1.trainer.trainer: Epoch 9/1000 | LR=9.01e-05 | loss=1.0213 | val_dice=0.0440 | best=0.0493 (ep9) | 00:11:38 | L_main=0.4296 L_aux_1=0.5878(w=0.5) L_aux_2=0.5957(w=0.5)
[2026-06-25 19:01:31] INFO segtask_v1.trainer.trainer:   Phase time (epoch 9): train=00:00:29 | val=00:00:37 | val=56.5% of (train+val)
[2026-06-25 19:01:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 9): 7279.9 MiB
[2026-06-25 19:02:38] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0337, per_class=['0.0337'], iou=0.0171, recall=0.1770, precision=0.0186, vol_sim=0.1905, mcc=-0.0172, min_class_dice=0.0337, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0804, per_class_sd=['0.0804'], combined(w=0.50)=0.0571, balanced=0.0391
[2026-06-25 19:02:38] INFO segtask_v1.trainer.trainer: Epoch 10/1000 | LR=1.00e-04 | loss=1.0138 | val_dice=0.0337 | best=0.0493 (ep9) | 00:12:46 | L_main=0.4217 L_aux_1=0.5899(w=0.5) L_aux_2=0.5942(w=0.5)
[2026-06-25 19:02:38] INFO segtask_v1.trainer.trainer:   Phase time (epoch 10): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 19:02:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 10): 7279.9 MiB
[2026-06-25 19:03:46] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0399, per_class=['0.0399'], iou=0.0203, recall=0.1668, precision=0.0226, vol_sim=0.2389, mcc=-0.0026, min_class_dice=0.0399, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0956, per_class_sd=['0.0956'], combined(w=0.50)=0.0677, balanced=0.0461
[2026-06-25 19:03:46] INFO segtask_v1.trainer.trainer: Epoch 11/1000 | LR=1.00e-04 | loss=0.9799 | val_dice=0.0399 | best=0.0493 (ep9) | 00:13:53 | L_main=0.3977 L_aux_1=0.5807(w=0.5) L_aux_2=0.5838(w=0.5)
[2026-06-25 19:03:46] INFO segtask_v1.trainer.trainer:   Phase time (epoch 11): train=00:00:29 | val=00:00:37 | val=56.5% of (train+val)
[2026-06-25 19:03:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 11): 7279.9 MiB
[2026-06-25 19:04:53] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0408, per_class=['0.0408'], iou=0.0208, recall=0.1668, precision=0.0232, vol_sim=0.2444, mcc=-0.0008, min_class_dice=0.0408, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0979, per_class_sd=['0.0979'], combined(w=0.50)=0.0693, balanced=0.0472
[2026-06-25 19:04:53] INFO segtask_v1.trainer.trainer: Epoch 12/1000 | LR=1.00e-04 | loss=0.8959 | val_dice=0.0408 | best=0.0493 (ep9) | 00:15:01 | L_main=0.3507 L_aux_1=0.5437(w=0.5) L_aux_2=0.5468(w=0.5)
[2026-06-25 19:04:53] INFO segtask_v1.trainer.trainer:   Phase time (epoch 12): train=00:00:29 | val=00:00:37 | val=56.2% of (train+val)
[2026-06-25 19:04:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 12): 7279.9 MiB
[2026-06-25 19:06:00] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0433, per_class=['0.0433'], iou=0.0221, recall=0.1142, precision=0.0267, vol_sim=0.3791, mcc=0.0071, min_class_dice=0.0433, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0563, per_class_sd=['0.0563'], combined(w=0.50)=0.0498, balanced=0.0455
[2026-06-25 19:06:00] INFO segtask_v1.trainer.trainer: Epoch 13/1000 | LR=1.00e-04 | loss=0.8851 | val_dice=0.0433 | best=0.0493 (ep9) | 00:16:08 | L_main=0.3387 L_aux_1=0.5461(w=0.5) L_aux_2=0.5465(w=0.5)
[2026-06-25 19:06:00] INFO segtask_v1.trainer.trainer:   Phase time (epoch 13): train=00:00:29 | val=00:00:37 | val=56.5% of (train+val)
[2026-06-25 19:06:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 13): 7279.9 MiB
[2026-06-25 19:07:07] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0000, vol_sim=0.5368, mcc=-0.0145, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:07:07] INFO segtask_v1.trainer.trainer: Epoch 14/1000 | LR=1.00e-04 | loss=0.8152 | val_dice=0.0000 | best=0.0493 (ep9) | 00:17:15 | L_main=0.3028 L_aux_1=0.5121(w=0.5) L_aux_2=0.5127(w=0.5)
[2026-06-25 19:07:07] INFO segtask_v1.trainer.trainer:   Phase time (epoch 14): train=00:00:29 | val=00:00:37 | val=56.1% of (train+val)
[2026-06-25 19:07:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 14): 7279.9 MiB
[2026-06-25 19:08:15] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0000, vol_sim=0.4561, mcc=-0.0130, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:08:15] INFO segtask_v1.trainer.trainer: Epoch 15/1000 | LR=1.00e-04 | loss=0.7674 | val_dice=0.0000 | best=0.0493 (ep9) | 00:18:23 | L_main=0.2769 L_aux_1=0.4899(w=0.5) L_aux_2=0.4912(w=0.5)
[2026-06-25 19:08:15] INFO segtask_v1.trainer.trainer:   Phase time (epoch 15): train=00:00:29 | val=00:00:37 | val=56.1% of (train+val)
[2026-06-25 19:08:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 15): 7279.9 MiB
[2026-06-25 19:09:22] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0000, vol_sim=0.2525, mcc=-0.0090, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:09:22] INFO segtask_v1.trainer.trainer: Epoch 16/1000 | LR=1.00e-04 | loss=0.7298 | val_dice=0.0000 | best=0.0493 (ep9) | 00:19:30 | L_main=0.2646 L_aux_1=0.4638(w=0.5) L_aux_2=0.4667(w=0.5)
[2026-06-25 19:09:22] INFO segtask_v1.trainer.trainer:   Phase time (epoch 16): train=00:00:29 | val=00:00:37 | val=56.2% of (train+val)
[2026-06-25 19:09:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 16): 7279.9 MiB
[2026-06-25 19:10:30] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0000, vol_sim=0.0041, mcc=-0.0011, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:10:30] INFO segtask_v1.trainer.trainer: Epoch 17/1000 | LR=1.00e-04 | loss=0.7050 | val_dice=0.0000 | best=0.0493 (ep9) | 00:20:38 | L_main=0.2527 L_aux_1=0.4515(w=0.5) L_aux_2=0.4531(w=0.5)
[2026-06-25 19:10:30] INFO segtask_v1.trainer.trainer:   Phase time (epoch 17): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:10:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 17): 7279.9 MiB
[2026-06-25 19:11:38] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0000, vol_sim=0.0031, mcc=-0.0009, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:11:38] INFO segtask_v1.trainer.trainer: Epoch 18/1000 | LR=1.00e-04 | loss=0.7340 | val_dice=0.0000 | best=0.0493 (ep9) | 00:21:45 | L_main=0.2671 L_aux_1=0.4663(w=0.5) L_aux_2=0.4675(w=0.5)
[2026-06-25 19:11:38] INFO segtask_v1.trainer.trainer:   Phase time (epoch 18): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:11:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 18): 7279.9 MiB
[2026-06-25 19:12:45] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0000, vol_sim=0.0010, mcc=-0.0005, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:12:45] INFO segtask_v1.trainer.trainer: Epoch 19/1000 | LR=1.00e-04 | loss=0.6542 | val_dice=0.0000 | best=0.0493 (ep9) | 00:22:52 | L_main=0.2306 L_aux_1=0.4213(w=0.5) L_aux_2=0.4258(w=0.5)
[2026-06-25 19:12:45] INFO segtask_v1.trainer.trainer:   Phase time (epoch 19): train=00:00:29 | val=00:00:37 | val=56.5% of (train+val)
[2026-06-25 19:12:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 19): 7279.9 MiB
[2026-06-25 19:13:52] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:13:52] INFO segtask_v1.trainer.trainer: Epoch 20/1000 | LR=1.00e-04 | loss=0.6219 | val_dice=0.0000 | best=0.0493 (ep9) | 00:23:59 | L_main=0.2205 L_aux_1=0.3989(w=0.5) L_aux_2=0.4040(w=0.5)
[2026-06-25 19:13:52] INFO segtask_v1.trainer.trainer:   Phase time (epoch 20): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 19:13:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 20): 7279.9 MiB
[2026-06-25 19:15:00] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:15:00] INFO segtask_v1.trainer.trainer: Epoch 21/1000 | LR=1.00e-04 | loss=0.6204 | val_dice=0.0000 | best=0.0493 (ep9) | 00:25:08 | L_main=0.2196 L_aux_1=0.3980(w=0.5) L_aux_2=0.4035(w=0.5)
[2026-06-25 19:15:00] INFO segtask_v1.trainer.trainer:   Phase time (epoch 21): train=00:00:29 | val=00:00:37 | val=56.2% of (train+val)
[2026-06-25 19:15:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 21): 7279.9 MiB
[2026-06-25 19:16:07] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:16:07] INFO segtask_v1.trainer.trainer: Epoch 22/1000 | LR=1.00e-04 | loss=0.5879 | val_dice=0.0000 | best=0.0493 (ep9) | 00:26:14 | L_main=0.2157 L_aux_1=0.3680(w=0.5) L_aux_2=0.3765(w=0.5)
[2026-06-25 19:16:07] INFO segtask_v1.trainer.trainer:   Phase time (epoch 22): train=00:00:28 | val=00:00:37 | val=56.8% of (train+val)
[2026-06-25 19:16:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 22): 7279.9 MiB
[2026-06-25 19:19:44] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:19:44] INFO segtask_v1.trainer.trainer: Epoch 23/1000 | LR=1.00e-04 | loss=0.5816 | val_dice=0.0000 | best=0.0493 (ep9) | 00:29:51 | L_main=0.2109 L_aux_1=0.3654(w=0.5) L_aux_2=0.3760(w=0.5)
[2026-06-25 19:19:44] INFO segtask_v1.trainer.trainer:   Phase time (epoch 23): train=00:00:29 | val=00:03:07 | val=86.3% of (train+val)
[2026-06-25 19:19:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 23): 7279.9 MiB
[2026-06-25 19:22:03] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:22:03] INFO segtask_v1.trainer.trainer: Epoch 24/1000 | LR=1.00e-04 | loss=0.5502 | val_dice=0.0000 | best=0.0493 (ep9) | 00:32:10 | L_main=0.1994 L_aux_1=0.3456(w=0.5) L_aux_2=0.3560(w=0.5)
[2026-06-25 19:22:03] INFO segtask_v1.trainer.trainer:   Phase time (epoch 24): train=00:01:39 | val=00:00:39 | val=28.3% of (train+val)
[2026-06-25 19:22:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 24): 7279.9 MiB
[2026-06-25 19:23:12] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:23:12] INFO segtask_v1.trainer.trainer: Epoch 25/1000 | LR=1.00e-04 | loss=0.5615 | val_dice=0.0000 | best=0.0493 (ep9) | 00:33:20 | L_main=0.2038 L_aux_1=0.3537(w=0.5) L_aux_2=0.3617(w=0.5)
[2026-06-25 19:23:12] INFO segtask_v1.trainer.trainer:   Phase time (epoch 25): train=00:00:30 | val=00:00:39 | val=56.6% of (train+val)
[2026-06-25 19:23:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 25): 7279.9 MiB
[2026-06-25 19:24:29] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:24:29] INFO segtask_v1.trainer.trainer: Epoch 26/1000 | LR=1.00e-04 | loss=0.5376 | val_dice=0.0000 | best=0.0493 (ep9) | 00:34:37 | L_main=0.1992 L_aux_1=0.3332(w=0.5) L_aux_2=0.3438(w=0.5)
[2026-06-25 19:24:29] INFO segtask_v1.trainer.trainer:   Phase time (epoch 26): train=00:00:39 | val=00:00:38 | val=49.3% of (train+val)
[2026-06-25 19:24:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 26): 7279.9 MiB
[2026-06-25 19:25:39] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:25:39] INFO segtask_v1.trainer.trainer: Epoch 27/1000 | LR=1.00e-04 | loss=0.4787 | val_dice=0.0000 | best=0.0493 (ep9) | 00:35:47 | L_main=0.1766 L_aux_1=0.2968(w=0.5) L_aux_2=0.3073(w=0.5)
[2026-06-25 19:25:39] INFO segtask_v1.trainer.trainer:   Phase time (epoch 27): train=00:00:31 | val=00:00:38 | val=55.5% of (train+val)
[2026-06-25 19:25:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 27): 7279.9 MiB
[2026-06-25 19:26:50] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:26:50] INFO segtask_v1.trainer.trainer: Epoch 28/1000 | LR=1.00e-04 | loss=0.5064 | val_dice=0.0000 | best=0.0493 (ep9) | 00:36:57 | L_main=0.1962 L_aux_1=0.3032(w=0.5) L_aux_2=0.3172(w=0.5)
[2026-06-25 19:26:50] INFO segtask_v1.trainer.trainer:   Phase time (epoch 28): train=00:00:31 | val=00:00:38 | val=54.7% of (train+val)
[2026-06-25 19:26:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 28): 7279.9 MiB
[2026-06-25 19:27:58] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:27:58] INFO segtask_v1.trainer.trainer: Epoch 29/1000 | LR=1.00e-04 | loss=0.4539 | val_dice=0.0000 | best=0.0493 (ep9) | 00:38:06 | L_main=0.1725 L_aux_1=0.2740(w=0.5) L_aux_2=0.2888(w=0.5)
[2026-06-25 19:27:58] INFO segtask_v1.trainer.trainer:   Phase time (epoch 29): train=00:00:29 | val=00:00:38 | val=56.2% of (train+val)
[2026-06-25 19:27:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 29): 7279.9 MiB
[2026-06-25 19:29:06] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:29:06] INFO segtask_v1.trainer.trainer: Epoch 30/1000 | LR=1.00e-04 | loss=0.4707 | val_dice=0.0000 | best=0.0493 (ep9) | 00:39:13 | L_main=0.1817 L_aux_1=0.2817(w=0.5) L_aux_2=0.2963(w=0.5)
[2026-06-25 19:29:06] INFO segtask_v1.trainer.trainer:   Phase time (epoch 30): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:29:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 30): 7279.9 MiB
[2026-06-25 19:30:14] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:30:14] INFO segtask_v1.trainer.trainer: Epoch 31/1000 | LR=1.00e-04 | loss=0.4443 | val_dice=0.0000 | best=0.0493 (ep9) | 00:40:22 | L_main=0.1698 L_aux_1=0.2664(w=0.5) L_aux_2=0.2826(w=0.5)
[2026-06-25 19:30:14] INFO segtask_v1.trainer.trainer:   Phase time (epoch 31): train=00:00:29 | val=00:00:37 | val=56.1% of (train+val)
[2026-06-25 19:30:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 31): 7279.9 MiB
[2026-06-25 19:31:22] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:31:22] INFO segtask_v1.trainer.trainer: Epoch 32/1000 | LR=1.00e-04 | loss=0.4259 | val_dice=0.0000 | best=0.0493 (ep9) | 00:41:30 | L_main=0.1631 L_aux_1=0.2537(w=0.5) L_aux_2=0.2720(w=0.5)
[2026-06-25 19:31:22] INFO segtask_v1.trainer.trainer:   Phase time (epoch 32): train=00:00:30 | val=00:00:37 | val=55.7% of (train+val)
[2026-06-25 19:31:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 32): 7279.9 MiB
[2026-06-25 19:32:29] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:32:29] INFO segtask_v1.trainer.trainer: Epoch 33/1000 | LR=1.00e-04 | loss=0.4520 | val_dice=0.0000 | best=0.0493 (ep9) | 00:42:37 | L_main=0.1731 L_aux_1=0.2706(w=0.5) L_aux_2=0.2871(w=0.5)
[2026-06-25 19:32:29] INFO segtask_v1.trainer.trainer:   Phase time (epoch 33): train=00:00:29 | val=00:00:37 | val=56.2% of (train+val)
[2026-06-25 19:32:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 33): 7279.9 MiB
[2026-06-25 19:33:37] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:33:37] INFO segtask_v1.trainer.trainer: Epoch 34/1000 | LR=1.00e-04 | loss=0.4628 | val_dice=0.0000 | best=0.0493 (ep9) | 00:43:45 | L_main=0.1886 L_aux_1=0.2655(w=0.5) L_aux_2=0.2829(w=0.5)
[2026-06-25 19:33:37] INFO segtask_v1.trainer.trainer:   Phase time (epoch 34): train=00:00:29 | val=00:00:37 | val=55.9% of (train+val)
[2026-06-25 19:33:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 34): 7279.9 MiB
[2026-06-25 19:34:45] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:34:45] INFO segtask_v1.trainer.trainer: Epoch 35/1000 | LR=1.00e-04 | loss=0.4370 | val_dice=0.0000 | best=0.0493 (ep9) | 00:44:53 | L_main=0.1769 L_aux_1=0.2498(w=0.5) L_aux_2=0.2705(w=0.5)
[2026-06-25 19:34:45] INFO segtask_v1.trainer.trainer:   Phase time (epoch 35): train=00:00:29 | val=00:00:37 | val=55.9% of (train+val)
[2026-06-25 19:34:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 35): 7279.9 MiB
[2026-06-25 19:35:52] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:35:52] INFO segtask_v1.trainer.trainer: Epoch 36/1000 | LR=1.00e-04 | loss=0.4183 | val_dice=0.0000 | best=0.0493 (ep9) | 00:46:00 | L_main=0.1691 L_aux_1=0.2393(w=0.5) L_aux_2=0.2591(w=0.5)
[2026-06-25 19:35:52] INFO segtask_v1.trainer.trainer:   Phase time (epoch 36): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:35:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 36): 7279.9 MiB
[2026-06-25 19:37:00] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:37:00] INFO segtask_v1.trainer.trainer: Epoch 37/1000 | LR=1.00e-04 | loss=0.4223 | val_dice=0.0000 | best=0.0493 (ep9) | 00:47:08 | L_main=0.1721 L_aux_1=0.2400(w=0.5) L_aux_2=0.2604(w=0.5)
[2026-06-25 19:37:00] INFO segtask_v1.trainer.trainer:   Phase time (epoch 37): train=00:00:29 | val=00:00:37 | val=55.9% of (train+val)
[2026-06-25 19:37:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 37): 7279.9 MiB
[2026-06-25 19:38:07] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:38:07] INFO segtask_v1.trainer.trainer: Epoch 38/1000 | LR=1.00e-04 | loss=0.4221 | val_dice=0.0000 | best=0.0493 (ep9) | 00:48:15 | L_main=0.1762 L_aux_1=0.2360(w=0.5) L_aux_2=0.2558(w=0.5)
[2026-06-25 19:38:07] INFO segtask_v1.trainer.trainer:   Phase time (epoch 38): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 19:38:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 38): 7279.9 MiB
[2026-06-25 19:39:15] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:39:15] INFO segtask_v1.trainer.trainer: Epoch 39/1000 | LR=9.99e-05 | loss=0.4280 | val_dice=0.0000 | best=0.0493 (ep9) | 00:49:23 | L_main=0.1795 L_aux_1=0.2375(w=0.5) L_aux_2=0.2594(w=0.5)
[2026-06-25 19:39:15] INFO segtask_v1.trainer.trainer:   Phase time (epoch 39): train=00:00:29 | val=00:00:37 | val=56.1% of (train+val)
[2026-06-25 19:39:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 39): 7279.9 MiB
[2026-06-25 19:40:22] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:40:22] INFO segtask_v1.trainer.trainer: Epoch 40/1000 | LR=9.99e-05 | loss=0.3997 | val_dice=0.0000 | best=0.0493 (ep9) | 00:50:30 | L_main=0.1667 L_aux_1=0.2228(w=0.5) L_aux_2=0.2433(w=0.5)
[2026-06-25 19:40:22] INFO segtask_v1.trainer.trainer:   Phase time (epoch 40): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:40:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 40): 7279.9 MiB
[2026-06-25 19:41:31] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:41:31] INFO segtask_v1.trainer.trainer: Epoch 41/1000 | LR=9.99e-05 | loss=0.3803 | val_dice=0.0000 | best=0.0493 (ep9) | 00:51:39 | L_main=0.1583 L_aux_1=0.2117(w=0.5) L_aux_2=0.2322(w=0.5)
[2026-06-25 19:41:31] INFO segtask_v1.trainer.trainer:   Phase time (epoch 41): train=00:00:29 | val=00:00:37 | val=55.8% of (train+val)
[2026-06-25 19:41:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 41): 7279.9 MiB
[2026-06-25 19:42:39] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:42:39] INFO segtask_v1.trainer.trainer: Epoch 42/1000 | LR=9.99e-05 | loss=0.4041 | val_dice=0.0000 | best=0.0493 (ep9) | 00:52:46 | L_main=0.1709 L_aux_1=0.2231(w=0.5) L_aux_2=0.2433(w=0.5)
[2026-06-25 19:42:39] INFO segtask_v1.trainer.trainer:   Phase time (epoch 42): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:42:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 42): 7279.9 MiB
[2026-06-25 19:43:47] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:43:47] INFO segtask_v1.trainer.trainer: Epoch 43/1000 | LR=9.99e-05 | loss=0.3854 | val_dice=0.0000 | best=0.0493 (ep9) | 00:53:55 | L_main=0.1632 L_aux_1=0.2108(w=0.5) L_aux_2=0.2336(w=0.5)
[2026-06-25 19:43:47] INFO segtask_v1.trainer.trainer:   Phase time (epoch 43): train=00:00:30 | val=00:00:38 | val=56.1% of (train+val)
[2026-06-25 19:43:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 43): 7279.9 MiB
[2026-06-25 19:44:54] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:44:54] INFO segtask_v1.trainer.trainer: Epoch 44/1000 | LR=9.99e-05 | loss=0.3817 | val_dice=0.0000 | best=0.0493 (ep9) | 00:55:02 | L_main=0.1618 L_aux_1=0.2100(w=0.5) L_aux_2=0.2298(w=0.5)
[2026-06-25 19:44:54] INFO segtask_v1.trainer.trainer:   Phase time (epoch 44): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 19:44:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 44): 7279.9 MiB
[2026-06-25 19:46:01] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:46:01] INFO segtask_v1.trainer.trainer: Epoch 45/1000 | LR=9.99e-05 | loss=0.3756 | val_dice=0.0000 | best=0.0493 (ep9) | 00:56:09 | L_main=0.1578 L_aux_1=0.2066(w=0.5) L_aux_2=0.2289(w=0.5)
[2026-06-25 19:46:01] INFO segtask_v1.trainer.trainer:   Phase time (epoch 45): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 19:46:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 45): 7279.9 MiB
[2026-06-25 19:47:09] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:47:09] INFO segtask_v1.trainer.trainer: Epoch 46/1000 | LR=9.99e-05 | loss=0.3766 | val_dice=0.0000 | best=0.0493 (ep9) | 00:57:17 | L_main=0.1604 L_aux_1=0.2053(w=0.5) L_aux_2=0.2272(w=0.5)
[2026-06-25 19:47:09] INFO segtask_v1.trainer.trainer:   Phase time (epoch 46): train=00:00:29 | val=00:00:37 | val=56.1% of (train+val)
[2026-06-25 19:47:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 46): 7279.9 MiB
[2026-06-25 19:48:16] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:48:16] INFO segtask_v1.trainer.trainer: Epoch 47/1000 | LR=9.99e-05 | loss=0.3727 | val_dice=0.0000 | best=0.0493 (ep9) | 00:58:23 | L_main=0.1619 L_aux_1=0.2000(w=0.5) L_aux_2=0.2217(w=0.5)
[2026-06-25 19:48:16] INFO segtask_v1.trainer.trainer:   Phase time (epoch 47): train=00:00:28 | val=00:00:37 | val=56.6% of (train+val)
[2026-06-25 19:48:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 47): 7279.9 MiB
[2026-06-25 19:49:23] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:49:23] INFO segtask_v1.trainer.trainer: Epoch 48/1000 | LR=9.99e-05 | loss=0.3789 | val_dice=0.0000 | best=0.0493 (ep9) | 00:59:31 | L_main=0.1646 L_aux_1=0.2031(w=0.5) L_aux_2=0.2256(w=0.5)
[2026-06-25 19:49:23] INFO segtask_v1.trainer.trainer:   Phase time (epoch 48): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:49:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 48): 7279.9 MiB
[2026-06-25 19:50:31] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:50:31] INFO segtask_v1.trainer.trainer: Epoch 49/1000 | LR=9.99e-05 | loss=0.3382 | val_dice=0.0000 | best=0.0493 (ep9) | 01:00:39 | L_main=0.1445 L_aux_1=0.1839(w=0.5) L_aux_2=0.2035(w=0.5)
[2026-06-25 19:50:31] INFO segtask_v1.trainer.trainer:   Phase time (epoch 49): train=00:00:29 | val=00:00:37 | val=55.9% of (train+val)
[2026-06-25 19:50:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 49): 7279.9 MiB
[2026-06-25 19:51:38] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:51:38] INFO segtask_v1.trainer.trainer: Epoch 50/1000 | LR=9.99e-05 | loss=0.3291 | val_dice=0.0000 | best=0.0493 (ep9) | 01:01:46 | L_main=0.1412 L_aux_1=0.1774(w=0.5) L_aux_2=0.1983(w=0.5)
[2026-06-25 19:51:38] INFO segtask_v1.trainer.trainer:   Phase time (epoch 50): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 19:51:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 50): 7279.9 MiB
[2026-06-25 19:52:46] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:52:46] INFO segtask_v1.trainer.trainer: Epoch 51/1000 | LR=9.99e-05 | loss=0.3278 | val_dice=0.0000 | best=0.0493 (ep9) | 01:02:54 | L_main=0.1416 L_aux_1=0.1756(w=0.5) L_aux_2=0.1967(w=0.5)
[2026-06-25 19:52:46] INFO segtask_v1.trainer.trainer:   Phase time (epoch 51): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:52:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 51): 7279.9 MiB
[2026-06-25 19:53:54] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:53:54] INFO segtask_v1.trainer.trainer: Epoch 52/1000 | LR=9.99e-05 | loss=0.3350 | val_dice=0.0000 | best=0.0493 (ep9) | 01:04:02 | L_main=0.1460 L_aux_1=0.1778(w=0.5) L_aux_2=0.2003(w=0.5)
[2026-06-25 19:53:54] INFO segtask_v1.trainer.trainer:   Phase time (epoch 52): train=00:00:29 | val=00:00:37 | val=56.0% of (train+val)
[2026-06-25 19:53:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 52): 7279.9 MiB
[2026-06-25 19:55:01] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:55:01] INFO segtask_v1.trainer.trainer: Epoch 53/1000 | LR=9.99e-05 | loss=0.3250 | val_dice=0.0000 | best=0.0493 (ep9) | 01:05:09 | L_main=0.1412 L_aux_1=0.1732(w=0.5) L_aux_2=0.1943(w=0.5)
[2026-06-25 19:55:01] INFO segtask_v1.trainer.trainer:   Phase time (epoch 53): train=00:00:29 | val=00:00:37 | val=56.2% of (train+val)
[2026-06-25 19:55:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 53): 7279.9 MiB
[2026-06-25 19:56:09] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:56:09] INFO segtask_v1.trainer.trainer: Epoch 54/1000 | LR=9.99e-05 | loss=0.3440 | val_dice=0.0000 | best=0.0493 (ep9) | 01:06:17 | L_main=0.1506 L_aux_1=0.1837(w=0.5) L_aux_2=0.2032(w=0.5)
[2026-06-25 19:56:09] INFO segtask_v1.trainer.trainer:   Phase time (epoch 54): train=00:00:29 | val=00:00:37 | val=55.9% of (train+val)
[2026-06-25 19:56:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 54): 7279.9 MiB
[2026-06-25 19:57:16] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:57:16] INFO segtask_v1.trainer.trainer: Epoch 55/1000 | LR=9.99e-05 | loss=0.3593 | val_dice=0.0000 | best=0.0493 (ep9) | 01:07:24 | L_main=0.1596 L_aux_1=0.1899(w=0.5) L_aux_2=0.2095(w=0.5)
[2026-06-25 19:57:16] INFO segtask_v1.trainer.trainer:   Phase time (epoch 55): train=00:00:29 | val=00:00:37 | val=56.1% of (train+val)
[2026-06-25 19:57:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 55): 7279.9 MiB
[2026-06-25 19:58:24] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:58:24] INFO segtask_v1.trainer.trainer: Epoch 56/1000 | LR=9.99e-05 | loss=0.3322 | val_dice=0.0000 | best=0.0493 (ep9) | 01:08:32 | L_main=0.1459 L_aux_1=0.1762(w=0.5) L_aux_2=0.1965(w=0.5)
[2026-06-25 19:58:24] INFO segtask_v1.trainer.trainer:   Phase time (epoch 56): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 19:58:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 56): 7279.9 MiB
[2026-06-25 19:59:31] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 19:59:31] INFO segtask_v1.trainer.trainer: Epoch 57/1000 | LR=9.99e-05 | loss=0.3175 | val_dice=0.0000 | best=0.0493 (ep9) | 01:09:39 | L_main=0.1426 L_aux_1=0.1655(w=0.5) L_aux_2=0.1843(w=0.5)
[2026-06-25 19:59:31] INFO segtask_v1.trainer.trainer:   Phase time (epoch 57): train=00:00:29 | val=00:00:37 | val=56.4% of (train+val)
[2026-06-25 19:59:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 57): 7279.9 MiB
[2026-06-25 20:00:38] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0001, per_class=['0.0001'], iou=0.0000, recall=0.0000, precision=0.9865, vol_sim=0.0001, mcc=0.0065, min_class_dice=0.0001, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-25 20:00:38] INFO segtask_v1.trainer.trainer: Epoch 58/1000 | LR=9.99e-05 | loss=0.3250 | val_dice=0.0001 | best=0.0493 (ep9) | 01:10:46 | L_main=0.1437 L_aux_1=0.1713(w=0.5) L_aux_2=0.1913(w=0.5)
[2026-06-25 20:00:38] INFO segtask_v1.trainer.trainer:   Phase time (epoch 58): train=00:00:29 | val=00:00:37 | val=56.3% of (train+val)
[2026-06-25 20:00:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 58): 7279.9 MiB
[2026-06-25 20:01:45] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0091, per_class=['0.0091'], iou=0.0046, recall=0.0046, precision=0.9995, vol_sim=0.0091, mcc=0.0669, min_class_dice=0.0091, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0013, per_class_sd=['0.0013'], combined(w=0.50)=0.0052, balanced=0.0037
[2026-06-25 20:01:45] INFO segtask_v1.trainer.trainer: Epoch 59/1000 | LR=9.99e-05 | loss=0.3200 | val_dice=0.0091 | best=0.0493 (ep9) | 01:11:53 | L_main=0.1427 L_aux_1=0.1671(w=0.5) L_aux_2=0.1876(w=0.5)
[2026-06-25 20:01:45] INFO segtask_v1.trainer.trainer:   Phase time (epoch 59): train=00:00:29 | val=00:00:37 | val=56.6% of (train+val)
[2026-06-25 20:01:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 59): 7279.9 MiB
[2026-06-26 06:00:50] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.0466, per_class=['0.0466'], iou=0.0239, recall=0.0239, precision=0.9940, vol_sim=0.0469, mcc=0.1522, min_class_dice=0.0466, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0090, per_class_sd=['0.0090'], combined(w=0.50)=0.0278, balanced=0.0228
[2026-06-26 06:00:51] INFO segtask_v1.trainer.trainer: Epoch 60/1000 | LR=9.98e-05 | loss=0.3506 | val_dice=0.0466 | best=0.0493 (ep9) | 11:10:58 | L_main=0.1585 L_aux_1=0.1828(w=0.5) L_aux_2=0.2014(w=0.5)
[2026-06-26 06:00:51] INFO segtask_v1.trainer.trainer:   Phase time (epoch 60): train=00:00:29 | val=09:58:35 | val=99.9% of (train+val)
[2026-06-26 06:00:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 60): 7279.9 MiB
[2026-06-26 08:10:11] INFO segtask_v1.trainer.validation:   Val[full-3D]: loss=nan, pooled_mean_dice=0.1011, per_class=['0.1011'], iou=0.0532, recall=0.0533, precision=0.9888, vol_sim=0.1022, mcc=0.2269, min_class_dice=0.1011, coverage=[22]/22 samples, pooled_mean_surface_dice@2px=0.0251, per_class_sd=['0.0251'], combined(w=0.50)=0.0631, balanced=0.0570
[2026-06-26 08:10:16] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf2d/best_model.pth
[2026-06-26 08:10:16] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0570 at epoch 61
[2026-06-26 08:10:16] INFO segtask_v1.trainer.trainer: Epoch 61/1000 | LR=9.98e-05 | loss=0.3057 | val_dice=0.1011 | best=0.0570 (ep61) | 13:20:24 | L_main=0.1375 L_aux_1=0.1588(w=0.5) L_aux_2=0.1776(w=0.5)
[2026-06-26 08:10:16] INFO segtask_v1.trainer.trainer:   Phase time (epoch 61): train=01:36:18 | val=00:32:59 | val=25.5% of (train+val)
[2026-06-26 08:10:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 61): 7279.9 MiB

Message from syslogd@imedway at Jun 26 08:58:12 ...
 kernel:[2048273.043637] watchdog: BUG: soft lockup - CPU#0 stuck for 23s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 08:59:43 ...
 kernel:[2048365.974103] watchdog: BUG: soft lockup - CPU#1 stuck for 24s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:00:09 ...
 kernel:[2048392.650811] watchdog: BUG: soft lockup - CPU#1 stuck for 49s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:00:45 ...
 kernel:[2048428.743766] watchdog: BUG: soft lockup - CPU#1 stuck for 82s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:01:13 ...
 kernel:[2048456.728510] watchdog: BUG: soft lockup - CPU#1 stuck for 108s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:01:40 ...
 kernel:[2048484.437247] watchdog: BUG: soft lockup - CPU#1 stuck for 134s! [cuda-EvtHandlr:3745250]

Message from syslogd@imedway at Jun 26 09:02:22 ...
 kernel:[2048524.398308] watchdog: BUG: soft lockup - CPU#0 stuck for 23s! [python:3745298]

Message from syslogd@imedway at Jun 26 09:03:41 ...
 kernel:[2048605.416454] watchdog: BUG: soft lockup - CPU#24 stuck for 22s! [python:3745299]

Message from syslogd@imedway at Jun 26 09:06:49 ...
 kernel:[2048792.957428] watchdog: BUG: soft lockup - CPU#8 stuck for 23s! [python:3744821]


3 开启多卡训练后，数据流，模型流，预测流生成失败，见上文信息。我理解的是就出单卡的数据流，模型流，预测流可视化就行了。请你分析，给一个最优的解决方案。