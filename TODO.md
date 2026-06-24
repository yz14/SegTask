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


2 以分割任务为例，对全流程增加可视化分析工具。例如用某个yaml开始训练模型后，如果yaml里面开启了可视化，则输出保存html展示：  
数据流：数据从npz读取到输入模型前的流程图，具体细节我设计不好，但是我可以给我期望的模型流来供你参考，看看我希望是什么样。  
模型流：包括从模型构建到训练全流程，我期望最终展示的是，输入数据是什么（一个框代表一个数据，里面是尺寸信息，如果是多分辨率，那么一个分辨率一个框），按数据经过模型的顺序来展示数据在模型中的处理过程和整个模型架构（箭头表示流向，一个框代表一个卷积/激活/归一化等等，里面只展示最关键信息，如果需要看到详细信息则需要双击框，由于一个stage可能由多个卷积/激活/归一化等等组成，那么再用一个大框来表示一个stage）。最终模型输出的结果指向一个损失（也用一个框来标识，并只展示关键信息，双击后展示详细参数信息）
预测流：类似模型流，但是不用展示模型细节了，将模型用一个框来表示，并只展示关键信息，双击后展示详细参数信息。
目前先把可视化做出了，供我分析使用，例如我看看数据流是否符合yaml，是否有优化空间。同时，分析模型架构是否符合yaml，是否可以针对不同任务做改进等等。html设计需要清晰，简洁，美观，精美，不要繁琐/一大堆看的眼花缭乱等等。  

3 仅仅将损失改为dice_focal后的训练：
2.5D:
[2026-06-23 16:52:14] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-23 16:52:14] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-23 16:52:14] INFO __main__: Device: cuda
[2026-06-23 16:52:15] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-23 16:52:15] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-23 16:52:15] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-23 16:52:15] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-23 16:52:30] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-23 16:52:30] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-23 16:52:30] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-23 16:53:05] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-23 16:53:05] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-23 16:53:17] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-23 16:53:17] INFO segtask_v1.data.loader: DataLoader: batch_size=4, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-23 16:53:17] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-23 16:53:17] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-23 16:53:18] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.28M, total=47.08M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-23 16:53:20] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-23 16:53:20] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-23 16:53:20] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-23 16:53:20] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=centerline, loss=dice, iter=3, weight=0.300)
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: Model params: 47.08M
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.6 + grad=179.6 + optim(AdamW,2x)=359.2 + ema=179.7 = 898.1 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=364.0 MiB, reserved=374.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: Train batches: 176, Val batches: 22
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=4
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_focal
[2026-06-23 16:53:20] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-23 16:53:32] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 6636.5 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-23 16:54:41] INFO segtask_v1.trainer.validation:   Val: loss=1.0380, pooled_mean_dice=0.0468, per_class=['0.0468'], iou=0.0240, recall=0.4166, precision=0.0248, vol_sim=0.1123, mcc=-0.0002, min_class_dice=0.0468, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.0885, per_class_sd=['0.0885'], combined(w=0.50)=0.0677, balanced=0.0524
[2026-06-23 16:54:43] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 16:54:43] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0524 at epoch 1
[2026-06-23 16:54:43] INFO segtask_v1.trainer.trainer: Epoch 1/400 | LR=2.01e-04 | loss=1.5260 | val_dice=0.0468 | best=0.0524 (ep1) | 00:01:22 | L_main=0.5851 L_aux_1=0.6377(w=0.5) L_aux_2=0.6498(w=0.5)
[2026-06-23 16:54:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 7035.7 MiB
[2026-06-23 16:55:41] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-23 16:55:41] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-23 16:55:41] INFO __main__: Device: cuda
[2026-06-23 16:55:41] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-23 16:55:41] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-23 16:55:41] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-23 16:55:41] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-23 16:55:56] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-23 16:55:56] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-23 16:55:56] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-23 16:56:31] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-23 16:56:31] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-23 16:56:41] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-23 16:56:41] INFO segtask_v1.data.loader: DataLoader: batch_size=8, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-23 16:56:41] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-23 16:56:41] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-23 16:56:42] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.28M, total=47.08M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-23 16:56:43] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-23 16:56:43] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-23 16:56:43] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-23 16:56:43] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=centerline, loss=dice, iter=3, weight=0.300)
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: Model params: 47.08M
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.6 + grad=179.6 + optim(AdamW,2x)=359.2 + ema=179.7 = 898.1 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=364.0 MiB, reserved=374.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: Train batches: 88, Val batches: 11
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=8
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_focal
[2026-06-23 16:56:43] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-23 16:57:02] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 12806.2 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-23 16:58:11] INFO segtask_v1.trainer.validation:   Val: loss=1.0370, pooled_mean_dice=0.0523, per_class=['0.0523'], iou=0.0269, recall=0.6660, precision=0.0272, vol_sim=0.0786, mcc=-0.0002, min_class_dice=0.0523, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.0847, per_class_sd=['0.0847'], combined(w=0.50)=0.0685, balanced=0.0570
[2026-06-23 16:58:14] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 16:58:14] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0570 at epoch 1
[2026-06-23 16:58:14] INFO segtask_v1.trainer.trainer: Epoch 1/400 | LR=2.01e-04 | loss=1.5523 | val_dice=0.0523 | best=0.0570 (ep1) | 00:01:31 | L_main=0.6012 L_aux_1=0.6463(w=0.5) L_aux_2=0.6609(w=0.5)
[2026-06-23 16:58:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 13240.3 MiB
[2026-06-23 16:59:26] INFO segtask_v1.trainer.validation:   Val: loss=1.0392, pooled_mean_dice=0.0421, per_class=['0.0421'], iou=0.0215, recall=0.4070, precision=0.0222, vol_sim=0.1035, mcc=0.0006, min_class_dice=0.0421, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0812, per_class_sd=['0.0812'], combined(w=0.50)=0.0617, balanced=0.0473
[2026-06-23 16:59:26] INFO segtask_v1.trainer.trainer: Epoch 2/400 | LR=4.01e-04 | loss=1.2123 | val_dice=0.0421 | best=0.0570 (ep1) | 00:02:43 | L_main=0.4040 L_aux_1=0.5074(w=0.5) L_aux_2=0.5174(w=0.5)
[2026-06-23 16:59:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 13421.7 MiB
[2026-06-23 17:00:32] INFO segtask_v1.trainer.validation:   Val: loss=1.0254, pooled_mean_dice=0.0361, per_class=['0.0361'], iou=0.0184, recall=0.0831, precision=0.0230, vol_sim=0.4342, mcc=-0.0004, min_class_dice=0.0361, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.0915, per_class_sd=['0.0915'], combined(w=0.50)=0.0638, balanced=0.0421
[2026-06-23 17:00:32] INFO segtask_v1.trainer.trainer: Epoch 3/400 | LR=6.00e-04 | loss=1.0040 | val_dice=0.0361 | best=0.0570 (ep1) | 00:03:49 | L_main=0.3001 L_aux_1=0.4064(w=0.5) L_aux_2=0.4248(w=0.5)
[2026-06-23 17:00:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 13421.7 MiB
[2026-06-23 17:01:37] INFO segtask_v1.trainer.validation:   Val: loss=1.0097, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0000, vol_sim=0.0168, mcc=-0.0023, min_class_dice=0.0000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:01:37] INFO segtask_v1.trainer.trainer: Epoch 4/400 | LR=8.00e-04 | loss=0.8333 | val_dice=0.0000 | best=0.0570 (ep1) | 00:04:54 | L_main=0.2620 L_aux_1=0.3075(w=0.5) L_aux_2=0.3421(w=0.5)
[2026-06-23 17:01:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 4): 13421.7 MiB
[2026-06-23 17:02:42] INFO segtask_v1.trainer.validation:   Val: loss=1.0023, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:02:42] INFO segtask_v1.trainer.trainer: Epoch 5/400 | LR=1.00e-03 | loss=0.7211 | val_dice=0.0000 | best=0.0570 (ep1) | 00:05:59 | L_main=0.2305 L_aux_1=0.2492(w=0.5) L_aux_2=0.2878(w=0.5)
[2026-06-23 17:02:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 5): 13421.7 MiB
[2026-06-23 17:03:48] INFO segtask_v1.trainer.validation:   Val: loss=0.9861, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:03:48] INFO segtask_v1.trainer.trainer: Epoch 6/400 | LR=1.00e-03 | loss=0.7236 | val_dice=0.0000 | best=0.0570 (ep1) | 00:07:05 | L_main=0.2397 L_aux_1=0.2456(w=0.5) L_aux_2=0.2789(w=0.5)
[2026-06-23 17:03:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 6): 13421.7 MiB
[2026-06-23 17:04:54] INFO segtask_v1.trainer.validation:   Val: loss=0.9929, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:04:54] INFO segtask_v1.trainer.trainer: Epoch 7/400 | LR=1.00e-03 | loss=0.6527 | val_dice=0.0000 | best=0.0570 (ep1) | 00:08:11 | L_main=0.2104 L_aux_1=0.2142(w=0.5) L_aux_2=0.2484(w=0.5)
[2026-06-23 17:04:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 7): 13421.7 MiB
[2026-06-23 17:06:00] INFO segtask_v1.trainer.validation:   Val: loss=0.9760, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:06:00] INFO segtask_v1.trainer.trainer: Epoch 8/400 | LR=1.00e-03 | loss=0.6098 | val_dice=0.0000 | best=0.0570 (ep1) | 00:09:17 | L_main=0.1922 L_aux_1=0.1944(w=0.5) L_aux_2=0.2256(w=0.5)
[2026-06-23 17:06:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 8): 13421.7 MiB
[2026-06-23 17:07:06] INFO segtask_v1.trainer.validation:   Val: loss=0.9762, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:07:06] INFO segtask_v1.trainer.trainer: Epoch 9/400 | LR=1.00e-03 | loss=0.5865 | val_dice=0.0000 | best=0.0570 (ep1) | 00:10:23 | L_main=0.1800 L_aux_1=0.1860(w=0.5) L_aux_2=0.2186(w=0.5)
[2026-06-23 17:07:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 9): 13421.7 MiB
[2026-06-23 17:08:14] INFO segtask_v1.trainer.validation:   Val: loss=0.9766, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:08:14] INFO segtask_v1.trainer.trainer: Epoch 10/400 | LR=1.00e-03 | loss=0.5585 | val_dice=0.0000 | best=0.0570 (ep1) | 00:11:30 | L_main=0.1712 L_aux_1=0.1738(w=0.5) L_aux_2=0.2040(w=0.5)
[2026-06-23 17:08:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 10): 13421.7 MiB
[2026-06-23 17:09:23] INFO segtask_v1.trainer.validation:   Val: loss=0.9792, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:09:23] INFO segtask_v1.trainer.trainer: Epoch 11/400 | LR=9.99e-04 | loss=0.5552 | val_dice=0.0000 | best=0.0570 (ep1) | 00:12:40 | L_main=0.1685 L_aux_1=0.1717(w=0.5) L_aux_2=0.1980(w=0.5)
[2026-06-23 17:09:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 11): 13421.7 MiB
[2026-06-23 17:10:29] INFO segtask_v1.trainer.validation:   Val: loss=0.9828, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:10:29] INFO segtask_v1.trainer.trainer: Epoch 12/400 | LR=9.99e-04 | loss=0.5497 | val_dice=0.0000 | best=0.0570 (ep1) | 00:13:46 | L_main=0.1670 L_aux_1=0.1691(w=0.5) L_aux_2=0.1960(w=0.5)
[2026-06-23 17:10:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 12): 13421.7 MiB
[2026-06-23 17:11:37] INFO segtask_v1.trainer.validation:   Val: loss=0.9844, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:11:37] INFO segtask_v1.trainer.trainer: Epoch 13/400 | LR=9.99e-04 | loss=0.5633 | val_dice=0.0000 | best=0.0570 (ep1) | 00:14:54 | L_main=0.1742 L_aux_1=0.1760(w=0.5) L_aux_2=0.2026(w=0.5)
[2026-06-23 17:11:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 13): 13421.7 MiB
[2026-06-23 17:12:57] INFO segtask_v1.trainer.validation:   Val: loss=0.9905, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:12:57] INFO segtask_v1.trainer.trainer: Epoch 14/400 | LR=9.99e-04 | loss=0.5237 | val_dice=0.0000 | best=0.0570 (ep1) | 00:16:14 | L_main=0.1545 L_aux_1=0.1571(w=0.5) L_aux_2=0.1821(w=0.5)
[2026-06-23 17:12:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 14): 13421.7 MiB
[2026-06-23 17:14:31] INFO segtask_v1.trainer.validation:   Val: loss=0.9945, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:14:31] INFO segtask_v1.trainer.trainer: Epoch 15/400 | LR=9.98e-04 | loss=0.5171 | val_dice=0.0000 | best=0.0570 (ep1) | 00:17:48 | L_main=0.1537 L_aux_1=0.1558(w=0.5) L_aux_2=0.1790(w=0.5)
[2026-06-23 17:14:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 15): 13421.7 MiB
[2026-06-23 17:16:27] INFO segtask_v1.trainer.validation:   Val: loss=1.0023, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:16:27] INFO segtask_v1.trainer.trainer: Epoch 16/400 | LR=9.98e-04 | loss=0.4995 | val_dice=0.0000 | best=0.0570 (ep1) | 00:19:44 | L_main=0.1470 L_aux_1=0.1495(w=0.5) L_aux_2=0.1730(w=0.5)
[2026-06-23 17:16:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 16): 13421.7 MiB
[2026-06-23 17:18:31] INFO segtask_v1.trainer.validation:   Val: loss=1.0068, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-23 17:18:31] INFO segtask_v1.trainer.trainer: Epoch 17/400 | LR=9.98e-04 | loss=0.5276 | val_dice=0.0000 | best=0.0570 (ep1) | 00:21:48 | L_main=0.1593 L_aux_1=0.1599(w=0.5) L_aux_2=0.1843(w=0.5)
[2026-06-23 17:18:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 17): 13421.7 MiB
[2026-06-23 17:20:23] INFO segtask_v1.trainer.validation:   Val: loss=1.0114, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0035, min_class_dice=0.0000, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.0005, per_class_sd=['0.0005'], combined(w=0.50)=0.0003, balanced=0.0001
[2026-06-23 17:20:23] INFO segtask_v1.trainer.trainer: Epoch 18/400 | LR=9.97e-04 | loss=0.4971 | val_dice=0.0000 | best=0.0570 (ep1) | 00:23:40 | L_main=0.1441 L_aux_1=0.1470(w=0.5) L_aux_2=0.1699(w=0.5)
[2026-06-23 17:20:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 18): 13421.7 MiB
[2026-06-23 17:21:53] INFO segtask_v1.trainer.validation:   Val: loss=1.0158, pooled_mean_dice=0.0005, per_class=['0.0005'], iou=0.0002, recall=0.0002, precision=0.9962, vol_sim=0.0005, mcc=0.0148, min_class_dice=0.0005, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0057, per_class_sd=['0.0057'], combined(w=0.50)=0.0031, balanced=0.0006
[2026-06-23 17:21:53] INFO segtask_v1.trainer.trainer: Epoch 19/400 | LR=9.97e-04 | loss=0.4740 | val_dice=0.0005 | best=0.0570 (ep1) | 00:25:09 | L_main=0.1348 L_aux_1=0.1375(w=0.5) L_aux_2=0.1587(w=0.5)
[2026-06-23 17:21:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 19): 13421.7 MiB
[2026-06-23 17:23:12] INFO segtask_v1.trainer.validation:   Val: loss=1.0075, pooled_mean_dice=0.0018, per_class=['0.0018'], iou=0.0009, recall=0.0009, precision=0.9842, vol_sim=0.0019, mcc=0.0298, min_class_dice=0.0018, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.0212, per_class_sd=['0.0212'], combined(w=0.50)=0.0115, balanced=0.0024
[2026-06-23 17:23:12] INFO segtask_v1.trainer.trainer: Epoch 20/400 | LR=9.96e-04 | loss=0.4998 | val_dice=0.0018 | best=0.0570 (ep1) | 00:26:29 | L_main=0.1445 L_aux_1=0.1466(w=0.5) L_aux_2=0.1696(w=0.5)
[2026-06-23 17:23:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 20): 13421.7 MiB
[2026-06-23 17:24:24] INFO segtask_v1.trainer.validation:   Val: loss=0.9960, pooled_mean_dice=0.0054, per_class=['0.0054'], iou=0.0027, recall=0.0027, precision=0.9926, vol_sim=0.0054, mcc=0.0511, min_class_dice=0.0054, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0502, per_class_sd=['0.0502'], combined(w=0.50)=0.0278, balanced=0.0069
[2026-06-23 17:24:24] INFO segtask_v1.trainer.trainer: Epoch 21/400 | LR=9.96e-04 | loss=0.4745 | val_dice=0.0054 | best=0.0570 (ep1) | 00:27:41 | L_main=0.1347 L_aux_1=0.1376(w=0.5) L_aux_2=0.1586(w=0.5)
[2026-06-23 17:24:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 21): 13421.7 MiB
[2026-06-23 17:25:33] INFO segtask_v1.trainer.validation:   Val: loss=0.9913, pooled_mean_dice=0.0136, per_class=['0.0136'], iou=0.0068, recall=0.0068, precision=0.9888, vol_sim=0.0137, mcc=0.0812, min_class_dice=0.0136, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.1000, per_class_sd=['0.1000'], combined(w=0.50)=0.0568, balanced=0.0172
[2026-06-23 17:25:33] INFO segtask_v1.trainer.trainer: Epoch 22/400 | LR=9.95e-04 | loss=0.4903 | val_dice=0.0136 | best=0.0570 (ep1) | 00:28:50 | L_main=0.1414 L_aux_1=0.1438(w=0.5) L_aux_2=0.1674(w=0.5)
[2026-06-23 17:25:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 22): 13421.7 MiB
[2026-06-23 17:26:46] INFO segtask_v1.trainer.validation:   Val: loss=0.9674, pooled_mean_dice=0.0277, per_class=['0.0277'], iou=0.0141, recall=0.0141, precision=0.9896, vol_sim=0.0280, mcc=0.1168, min_class_dice=0.0277, coverage=[67]/88 samples, pooled_mean_surface_dice@2px=0.1682, per_class_sd=['0.1682'], combined(w=0.50)=0.0980, balanced=0.0348
[2026-06-23 17:26:46] INFO segtask_v1.trainer.trainer: Epoch 23/400 | LR=9.95e-04 | loss=0.4695 | val_dice=0.0277 | best=0.0570 (ep1) | 00:30:03 | L_main=0.1323 L_aux_1=0.1342(w=0.5) L_aux_2=0.1551(w=0.5)
[2026-06-23 17:26:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 23): 13421.7 MiB
[2026-06-23 17:27:53] INFO segtask_v1.trainer.validation:   Val: loss=0.9570, pooled_mean_dice=0.0543, per_class=['0.0543'], iou=0.0279, recall=0.0279, precision=0.9766, vol_sim=0.0556, mcc=0.1632, min_class_dice=0.0543, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.2599, per_class_sd=['0.2599'], combined(w=0.50)=0.1571, balanced=0.0669
[2026-06-23 17:27:58] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:27:58] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0669 at epoch 24
[2026-06-23 17:27:58] INFO segtask_v1.trainer.trainer: Epoch 24/400 | LR=9.94e-04 | loss=0.4550 | val_dice=0.0543 | best=0.0669 (ep24) | 00:31:15 | L_main=0.1248 L_aux_1=0.1282(w=0.5) L_aux_2=0.1486(w=0.5)
[2026-06-23 17:27:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 24): 13421.7 MiB
[2026-06-23 17:29:07] INFO segtask_v1.trainer.validation:   Val: loss=0.9404, pooled_mean_dice=0.0763, per_class=['0.0763'], iou=0.0397, recall=0.0397, precision=0.9746, vol_sim=0.0783, mcc=0.1942, min_class_dice=0.0763, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.3192, per_class_sd=['0.3192'], combined(w=0.50)=0.1977, balanced=0.0928
[2026-06-23 17:29:12] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:29:12] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0928 at epoch 25
[2026-06-23 17:29:12] INFO segtask_v1.trainer.trainer: Epoch 25/400 | LR=9.94e-04 | loss=0.4539 | val_dice=0.0763 | best=0.0928 (ep25) | 00:32:29 | L_main=0.1279 L_aux_1=0.1304(w=0.5) L_aux_2=0.1498(w=0.5)
[2026-06-23 17:29:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 25): 13421.7 MiB
[2026-06-23 17:30:40] INFO segtask_v1.trainer.validation:   Val: loss=0.9023, pooled_mean_dice=0.0996, per_class=['0.0996'], iou=0.0524, recall=0.0525, precision=0.9660, vol_sim=0.1032, mcc=0.2230, min_class_dice=0.0996, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.3780, per_class_sd=['0.3780'], combined(w=0.50)=0.2388, balanced=0.1198
[2026-06-23 17:30:44] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:30:44] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.1198 at epoch 26
[2026-06-23 17:30:44] INFO segtask_v1.trainer.trainer: Epoch 26/400 | LR=9.93e-04 | loss=0.4926 | val_dice=0.0996 | best=0.1198 (ep26) | 00:34:01 | L_main=0.1421 L_aux_1=0.1441(w=0.5) L_aux_2=0.1657(w=0.5)
[2026-06-23 17:30:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 26): 13421.7 MiB
[2026-06-23 17:32:11] INFO segtask_v1.trainer.validation:   Val: loss=0.8648, pooled_mean_dice=0.1524, per_class=['0.1524'], iou=0.0825, recall=0.0828, precision=0.9610, vol_sim=0.1586, mcc=0.2790, min_class_dice=0.1524, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.4626, per_class_sd=['0.4626'], combined(w=0.50)=0.3075, balanced=0.1785
[2026-06-23 17:32:15] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:32:15] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.1785 at epoch 27
[2026-06-23 17:32:15] INFO segtask_v1.trainer.trainer: Epoch 27/400 | LR=9.92e-04 | loss=0.4541 | val_dice=0.1524 | best=0.1785 (ep27) | 00:35:32 | L_main=0.1259 L_aux_1=0.1276(w=0.5) L_aux_2=0.1483(w=0.5)
[2026-06-23 17:32:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 27): 13421.7 MiB
[2026-06-23 17:33:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8784, pooled_mean_dice=0.1406, per_class=['0.1406'], iou=0.0756, recall=0.0760, precision=0.9439, vol_sim=0.1490, mcc=0.2642, min_class_dice=0.1406, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.4343, per_class_sd=['0.4343'], combined(w=0.50)=0.2874, balanced=0.1652
[2026-06-23 17:33:45] INFO segtask_v1.trainer.trainer: Epoch 28/400 | LR=9.92e-04 | loss=0.4961 | val_dice=0.1406 | best=0.1785 (ep27) | 00:37:01 | L_main=0.1461 L_aux_1=0.1465(w=0.5) L_aux_2=0.1666(w=0.5)
[2026-06-23 17:33:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 28): 13421.7 MiB
[2026-06-23 17:35:08] INFO segtask_v1.trainer.validation:   Val: loss=0.8474, pooled_mean_dice=0.1615, per_class=['0.1615'], iou=0.0878, recall=0.0883, precision=0.9413, vol_sim=0.1715, mcc=0.2845, min_class_dice=0.1615, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.4745, per_class_sd=['0.4745'], combined(w=0.50)=0.3180, balanced=0.1883
[2026-06-23 17:35:13] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:35:13] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.1883 at epoch 29
[2026-06-23 17:35:13] INFO segtask_v1.trainer.trainer: Epoch 29/400 | LR=9.91e-04 | loss=0.4639 | val_dice=0.1615 | best=0.1883 (ep29) | 00:38:30 | L_main=0.1309 L_aux_1=0.1329(w=0.5) L_aux_2=0.1527(w=0.5)
[2026-06-23 17:35:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 29): 13421.7 MiB
[2026-06-23 17:36:31] INFO segtask_v1.trainer.validation:   Val: loss=0.7949, pooled_mean_dice=0.1934, per_class=['0.1934'], iou=0.1070, recall=0.1079, precision=0.9291, vol_sim=0.2081, mcc=0.3130, min_class_dice=0.1934, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.5144, per_class_sd=['0.5144'], combined(w=0.50)=0.3539, balanced=0.2225
[2026-06-23 17:36:36] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:36:36] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2225 at epoch 30
[2026-06-23 17:36:36] INFO segtask_v1.trainer.trainer: Epoch 30/400 | LR=9.90e-04 | loss=0.4490 | val_dice=0.1934 | best=0.2225 (ep30) | 00:39:53 | L_main=0.1199 L_aux_1=0.1240(w=0.5) L_aux_2=0.1445(w=0.5)
[2026-06-23 17:36:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 30): 13421.7 MiB
[2026-06-23 17:37:54] INFO segtask_v1.trainer.validation:   Val: loss=0.7792, pooled_mean_dice=0.2340, per_class=['0.2340'], iou=0.1325, recall=0.1342, precision=0.9117, vol_sim=0.2567, mcc=0.3449, min_class_dice=0.2340, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.5692, per_class_sd=['0.5692'], combined(w=0.50)=0.4016, balanced=0.2655
[2026-06-23 17:37:59] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:37:59] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2655 at epoch 31
[2026-06-23 17:37:59] INFO segtask_v1.trainer.trainer: Epoch 31/400 | LR=9.89e-04 | loss=0.4450 | val_dice=0.2340 | best=0.2655 (ep31) | 00:41:15 | L_main=0.1210 L_aux_1=0.1246(w=0.5) L_aux_2=0.1444(w=0.5)
[2026-06-23 17:37:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 31): 13421.7 MiB
[2026-06-23 17:39:13] INFO segtask_v1.trainer.validation:   Val: loss=0.7480, pooled_mean_dice=0.2408, per_class=['0.2408'], iou=0.1369, recall=0.1389, precision=0.9029, vol_sim=0.2667, mcc=0.3487, min_class_dice=0.2408, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.5606, per_class_sd=['0.5606'], combined(w=0.50)=0.4007, balanced=0.2717
[2026-06-23 17:39:17] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:39:17] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2717 at epoch 32
[2026-06-23 17:39:17] INFO segtask_v1.trainer.trainer: Epoch 32/400 | LR=9.89e-04 | loss=0.4247 | val_dice=0.2408 | best=0.2717 (ep32) | 00:42:34 | L_main=0.1112 L_aux_1=0.1149(w=0.5) L_aux_2=0.1337(w=0.5)
[2026-06-23 17:39:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 32): 13421.7 MiB
[2026-06-23 17:40:36] INFO segtask_v1.trainer.validation:   Val: loss=0.6669, pooled_mean_dice=0.3187, per_class=['0.3187'], iou=0.1896, recall=0.1940, precision=0.8920, vol_sim=0.3573, mcc=0.4118, min_class_dice=0.3187, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.6578, per_class_sd=['0.6578'], combined(w=0.50)=0.4883, balanced=0.3525
[2026-06-23 17:40:41] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:40:41] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.3525 at epoch 33
[2026-06-23 17:40:41] INFO segtask_v1.trainer.trainer: Epoch 33/400 | LR=9.88e-04 | loss=0.4355 | val_dice=0.3187 | best=0.3525 (ep33) | 00:43:58 | L_main=0.1178 L_aux_1=0.1210(w=0.5) L_aux_2=0.1394(w=0.5)
[2026-06-23 17:40:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 33): 13421.7 MiB
[2026-06-23 17:42:03] INFO segtask_v1.trainer.validation:   Val: loss=0.6580, pooled_mean_dice=0.2981, per_class=['0.2981'], iou=0.1752, recall=0.1805, precision=0.8558, vol_sim=0.3483, mcc=0.3872, min_class_dice=0.2981, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.6395, per_class_sd=['0.6395'], combined(w=0.50)=0.4688, balanced=0.3315
[2026-06-23 17:42:03] INFO segtask_v1.trainer.trainer: Epoch 34/400 | LR=9.87e-04 | loss=0.4306 | val_dice=0.2981 | best=0.3525 (ep33) | 00:45:20 | L_main=0.1151 L_aux_1=0.1190(w=0.5) L_aux_2=0.1375(w=0.5)
[2026-06-23 17:42:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 34): 13421.7 MiB
[2026-06-23 17:43:21] INFO segtask_v1.trainer.validation:   Val: loss=0.5760, pooled_mean_dice=0.4028, per_class=['0.4028'], iou=0.2522, recall=0.2652, precision=0.8368, vol_sim=0.4814, mcc=0.4654, min_class_dice=0.4028, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.7257, per_class_sd=['0.7257'], combined(w=0.50)=0.5642, balanced=0.4352
[2026-06-23 17:43:26] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:43:26] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.4352 at epoch 35
[2026-06-23 17:43:26] INFO segtask_v1.trainer.trainer: Epoch 35/400 | LR=9.86e-04 | loss=0.4179 | val_dice=0.4028 | best=0.4352 (ep35) | 00:46:42 | L_main=0.1092 L_aux_1=0.1121(w=0.5) L_aux_2=0.1300(w=0.5)
[2026-06-23 17:43:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 35): 13421.7 MiB
[2026-06-23 17:44:39] INFO segtask_v1.trainer.validation:   Val: loss=0.5506, pooled_mean_dice=0.4286, per_class=['0.4286'], iou=0.2727, recall=0.2871, precision=0.8451, vol_sim=0.5071, mcc=0.4861, min_class_dice=0.4286, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7320, per_class_sd=['0.7320'], combined(w=0.50)=0.5803, balanced=0.4591
[2026-06-23 17:44:43] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:44:43] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.4591 at epoch 36
[2026-06-23 17:44:43] INFO segtask_v1.trainer.trainer: Epoch 36/400 | LR=9.85e-04 | loss=0.4204 | val_dice=0.4286 | best=0.4591 (ep36) | 00:48:00 | L_main=0.1120 L_aux_1=0.1147(w=0.5) L_aux_2=0.1317(w=0.5)
[2026-06-23 17:44:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 36): 13421.7 MiB
[2026-06-23 17:46:06] INFO segtask_v1.trainer.validation:   Val: loss=0.4857, pooled_mean_dice=0.5019, per_class=['0.5019'], iou=0.3350, recall=0.3603, precision=0.8272, vol_sim=0.6068, mcc=0.5397, min_class_dice=0.5019, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.7778, per_class_sd=['0.7778'], combined(w=0.50)=0.6399, balanced=0.5290
[2026-06-23 17:46:11] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:46:11] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.5290 at epoch 37
[2026-06-23 17:46:11] INFO segtask_v1.trainer.trainer: Epoch 37/400 | LR=9.84e-04 | loss=0.4038 | val_dice=0.5019 | best=0.5290 (ep37) | 00:49:28 | L_main=0.1034 L_aux_1=0.1065(w=0.5) L_aux_2=0.1232(w=0.5)
[2026-06-23 17:46:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 37): 13421.7 MiB
[2026-06-23 17:47:40] INFO segtask_v1.trainer.validation:   Val: loss=0.4723, pooled_mean_dice=0.5451, per_class=['0.5451'], iou=0.3746, recall=0.4055, precision=0.8311, vol_sim=0.6559, mcc=0.5732, min_class_dice=0.5451, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.7900, per_class_sd=['0.7900'], combined(w=0.50)=0.6675, balanced=0.5681
[2026-06-23 17:47:45] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:47:45] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.5681 at epoch 38
[2026-06-23 17:47:45] INFO segtask_v1.trainer.trainer: Epoch 38/400 | LR=9.83e-04 | loss=0.4199 | val_dice=0.5451 | best=0.5681 (ep38) | 00:51:02 | L_main=0.1107 L_aux_1=0.1136(w=0.5) L_aux_2=0.1314(w=0.5)
[2026-06-23 17:47:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 38): 13421.7 MiB
[2026-06-23 17:49:12] INFO segtask_v1.trainer.validation:   Val: loss=0.4162, pooled_mean_dice=0.6427, per_class=['0.6427'], iou=0.4736, recall=0.5282, precision=0.8208, vol_sim=0.7831, mcc=0.6522, min_class_dice=0.6427, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8478, per_class_sd=['0.8478'], combined(w=0.50)=0.7453, balanced=0.6604
[2026-06-23 17:49:16] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:49:16] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6604 at epoch 39
[2026-06-23 17:49:16] INFO segtask_v1.trainer.trainer: Epoch 39/400 | LR=9.82e-04 | loss=0.4037 | val_dice=0.6427 | best=0.6604 (ep39) | 00:52:33 | L_main=0.1040 L_aux_1=0.1063(w=0.5) L_aux_2=0.1230(w=0.5)
[2026-06-23 17:49:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 39): 13421.7 MiB
[2026-06-23 17:50:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3798, pooled_mean_dice=0.7047, per_class=['0.7047'], iou=0.5440, recall=0.6306, precision=0.7984, vol_sim=0.8826, mcc=0.7039, min_class_dice=0.7047, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8788, per_class_sd=['0.8788'], combined(w=0.50)=0.7917, balanced=0.7183
[2026-06-23 17:50:45] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:50:45] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7183 at epoch 40
[2026-06-23 17:50:45] INFO segtask_v1.trainer.trainer: Epoch 40/400 | LR=9.81e-04 | loss=0.4406 | val_dice=0.7047 | best=0.7183 (ep40) | 00:54:02 | L_main=0.1263 L_aux_1=0.1261(w=0.5) L_aux_2=0.1443(w=0.5)
[2026-06-23 17:50:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 40): 13421.7 MiB
[2026-06-23 17:52:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3946, pooled_mean_dice=0.7051, per_class=['0.7051'], iou=0.5445, recall=0.6178, precision=0.8211, vol_sim=0.8587, mcc=0.7060, min_class_dice=0.7051, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8641, per_class_sd=['0.8641'], combined(w=0.50)=0.7846, balanced=0.7163
[2026-06-23 17:52:01] INFO segtask_v1.trainer.trainer: Epoch 41/400 | LR=9.80e-04 | loss=0.4097 | val_dice=0.7051 | best=0.7183 (ep40) | 00:55:18 | L_main=0.1136 L_aux_1=0.1151(w=0.5) L_aux_2=0.1336(w=0.5)
[2026-06-23 17:52:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 41): 13421.7 MiB
[2026-06-23 17:53:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3700, pooled_mean_dice=0.7315, per_class=['0.7315'], iou=0.5766, recall=0.6830, precision=0.7874, vol_sim=0.9290, mcc=0.7282, min_class_dice=0.7315, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8986, per_class_sd=['0.8986'], combined(w=0.50)=0.8151, balanced=0.7447
[2026-06-23 17:53:17] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:53:17] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7447 at epoch 42
[2026-06-23 17:53:17] INFO segtask_v1.trainer.trainer: Epoch 42/400 | LR=9.79e-04 | loss=0.4116 | val_dice=0.7315 | best=0.7447 (ep42) | 00:56:34 | L_main=0.1162 L_aux_1=0.1176(w=0.5) L_aux_2=0.1343(w=0.5)
[2026-06-23 17:53:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 42): 13421.7 MiB
[2026-06-23 17:54:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3776, pooled_mean_dice=0.7089, per_class=['0.7089'], iou=0.5491, recall=0.6697, precision=0.7531, vol_sim=0.9414, mcc=0.7051, min_class_dice=0.7089, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8882, per_class_sd=['0.8882'], combined(w=0.50)=0.7985, balanced=0.7232
[2026-06-23 17:54:32] INFO segtask_v1.trainer.trainer: Epoch 43/400 | LR=9.77e-04 | loss=0.3844 | val_dice=0.7089 | best=0.7447 (ep42) | 00:57:49 | L_main=0.1041 L_aux_1=0.1062(w=0.5) L_aux_2=0.1225(w=0.5)
[2026-06-23 17:54:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 43): 13421.7 MiB
[2026-06-23 17:55:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3323, pooled_mean_dice=0.7652, per_class=['0.7652'], iou=0.6197, recall=0.7527, precision=0.7781, vol_sim=0.9834, mcc=0.7591, min_class_dice=0.7652, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9039, per_class_sd=['0.9039'], combined(w=0.50)=0.8345, balanced=0.7746
[2026-06-23 17:55:54] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:55:54] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7746 at epoch 44
[2026-06-23 17:55:54] INFO segtask_v1.trainer.trainer: Epoch 44/400 | LR=9.76e-04 | loss=0.3765 | val_dice=0.7652 | best=0.7746 (ep44) | 00:59:10 | L_main=0.0999 L_aux_1=0.1031(w=0.5) L_aux_2=0.1192(w=0.5)
[2026-06-23 17:55:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 44): 13421.7 MiB
[2026-06-23 17:57:12] INFO segtask_v1.trainer.validation:   Val: loss=0.3169, pooled_mean_dice=0.7715, per_class=['0.7715'], iou=0.6279, recall=0.7828, precision=0.7604, vol_sim=0.9855, mcc=0.7652, min_class_dice=0.7715, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9070, per_class_sd=['0.9070'], combined(w=0.50)=0.8392, balanced=0.7806
[2026-06-23 17:57:17] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:57:17] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7806 at epoch 45
[2026-06-23 17:57:17] INFO segtask_v1.trainer.trainer: Epoch 45/400 | LR=9.75e-04 | loss=0.3811 | val_dice=0.7715 | best=0.7806 (ep45) | 01:00:34 | L_main=0.1030 L_aux_1=0.1058(w=0.5) L_aux_2=0.1216(w=0.5)
[2026-06-23 17:57:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 45): 13421.7 MiB
[2026-06-23 17:58:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3157, pooled_mean_dice=0.7753, per_class=['0.7753'], iou=0.6330, recall=0.8157, precision=0.7387, vol_sim=0.9505, mcc=0.7708, min_class_dice=0.7753, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9107, per_class_sd=['0.9107'], combined(w=0.50)=0.8430, balanced=0.7848
[2026-06-23 17:58:36] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 17:58:36] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7848 at epoch 46
[2026-06-23 17:58:36] INFO segtask_v1.trainer.trainer: Epoch 46/400 | LR=9.74e-04 | loss=0.3884 | val_dice=0.7753 | best=0.7848 (ep46) | 01:01:53 | L_main=0.1051 L_aux_1=0.1085(w=0.5) L_aux_2=0.1244(w=0.5)
[2026-06-23 17:58:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 46): 13421.7 MiB
[2026-06-23 17:59:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2695, pooled_mean_dice=0.8180, per_class=['0.8180'], iou=0.6920, recall=0.8771, precision=0.7663, vol_sim=0.9326, mcc=0.8151, min_class_dice=0.8180, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9294, per_class_sd=['0.9294'], combined(w=0.50)=0.8737, balanced=0.8255
[2026-06-23 18:00:01] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 18:00:01] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.8255 at epoch 47
[2026-06-23 18:00:01] INFO segtask_v1.trainer.trainer: Epoch 47/400 | LR=9.72e-04 | loss=0.3803 | val_dice=0.8180 | best=0.8255 (ep47) | 01:03:17 | L_main=0.1026 L_aux_1=0.1049(w=0.5) L_aux_2=0.1207(w=0.5)
[2026-06-23 18:00:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 47): 13421.7 MiB
[2026-06-23 18:01:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2743, pooled_mean_dice=0.8079, per_class=['0.8079'], iou=0.6776, recall=0.8738, precision=0.7512, vol_sim=0.9246, mcc=0.8047, min_class_dice=0.8079, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9244, per_class_sd=['0.9244'], combined(w=0.50)=0.8661, balanced=0.8157
[2026-06-23 18:01:16] INFO segtask_v1.trainer.trainer: Epoch 48/400 | LR=9.71e-04 | loss=0.3990 | val_dice=0.8079 | best=0.8255 (ep47) | 01:04:33 | L_main=0.1079 L_aux_1=0.1119(w=0.5) L_aux_2=0.1293(w=0.5)
[2026-06-23 18:01:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 48): 13421.7 MiB
[2026-06-23 18:02:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2626, pooled_mean_dice=0.8087, per_class=['0.8087'], iou=0.6788, recall=0.9064, precision=0.7300, vol_sim=0.8922, mcc=0.8085, min_class_dice=0.8087, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9242, per_class_sd=['0.9242'], combined(w=0.50)=0.8664, balanced=0.8167
[2026-06-23 18:02:32] INFO segtask_v1.trainer.trainer: Epoch 49/400 | LR=9.70e-04 | loss=0.3895 | val_dice=0.8087 | best=0.8255 (ep47) | 01:05:49 | L_main=0.1064 L_aux_1=0.1098(w=0.5) L_aux_2=0.1277(w=0.5)
[2026-06-23 18:02:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 49): 13421.7 MiB
[2026-06-23 18:03:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2712, pooled_mean_dice=0.8111, per_class=['0.8111'], iou=0.6823, recall=0.9278, precision=0.7205, vol_sim=0.8743, mcc=0.8129, min_class_dice=0.8111, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9203, per_class_sd=['0.9203'], combined(w=0.50)=0.8657, balanced=0.8183
[2026-06-23 18:03:50] INFO segtask_v1.trainer.trainer: Epoch 50/400 | LR=9.68e-04 | loss=0.3766 | val_dice=0.8111 | best=0.8255 (ep47) | 01:07:07 | L_main=0.1009 L_aux_1=0.1042(w=0.5) L_aux_2=0.1208(w=0.5)
[2026-06-23 18:03:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 50): 13421.7 MiB
[2026-06-23 18:05:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2639, pooled_mean_dice=0.8100, per_class=['0.8100'], iou=0.6806, recall=0.9462, precision=0.7080, vol_sim=0.8560, mcc=0.8140, min_class_dice=0.8100, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9199, per_class_sd=['0.9199'], combined(w=0.50)=0.8649, balanced=0.8174
[2026-06-23 18:05:04] INFO segtask_v1.trainer.trainer: Epoch 51/400 | LR=9.67e-04 | loss=0.3802 | val_dice=0.8100 | best=0.8255 (ep47) | 01:08:21 | L_main=0.1022 L_aux_1=0.1062(w=0.5) L_aux_2=0.1228(w=0.5)
[2026-06-23 18:05:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 51): 13421.7 MiB
[2026-06-23 18:06:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2811, pooled_mean_dice=0.8005, per_class=['0.8005'], iou=0.6673, recall=0.9477, precision=0.6928, vol_sim=0.8446, mcc=0.8061, min_class_dice=0.8005, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9162, per_class_sd=['0.9162'], combined(w=0.50)=0.8583, balanced=0.8086
[2026-06-23 18:06:21] INFO segtask_v1.trainer.trainer: Epoch 52/400 | LR=9.66e-04 | loss=0.3704 | val_dice=0.8005 | best=0.8255 (ep47) | 01:09:38 | L_main=0.0981 L_aux_1=0.1011(w=0.5) L_aux_2=0.1170(w=0.5)
[2026-06-23 18:06:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 52): 13421.7 MiB
[2026-06-23 18:07:34] INFO segtask_v1.trainer.validation:   Val: loss=0.2812, pooled_mean_dice=0.7914, per_class=['0.7914'], iou=0.6549, recall=0.9443, precision=0.6812, vol_sim=0.8381, mcc=0.7965, min_class_dice=0.7914, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9158, per_class_sd=['0.9158'], combined(w=0.50)=0.8536, balanced=0.8007
[2026-06-23 18:07:34] INFO segtask_v1.trainer.trainer: Epoch 53/400 | LR=9.64e-04 | loss=0.3824 | val_dice=0.7914 | best=0.8255 (ep47) | 01:10:51 | L_main=0.1029 L_aux_1=0.1063(w=0.5) L_aux_2=0.1216(w=0.5)
[2026-06-23 18:07:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 53): 13421.7 MiB
[2026-06-23 18:09:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2764, pooled_mean_dice=0.7895, per_class=['0.7895'], iou=0.6523, recall=0.9437, precision=0.6787, vol_sim=0.8366, mcc=0.7950, min_class_dice=0.7895, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9027, per_class_sd=['0.9027'], combined(w=0.50)=0.8461, balanced=0.7966
[2026-06-23 18:09:00] INFO segtask_v1.trainer.trainer: Epoch 54/400 | LR=9.63e-04 | loss=0.3992 | val_dice=0.7895 | best=0.8255 (ep47) | 01:12:17 | L_main=0.1103 L_aux_1=0.1136(w=0.5) L_aux_2=0.1298(w=0.5)
[2026-06-23 18:09:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 54): 13421.7 MiB
[2026-06-23 18:10:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2570, pooled_mean_dice=0.8000, per_class=['0.8000'], iou=0.6666, recall=0.9368, precision=0.6980, vol_sim=0.8540, mcc=0.8034, min_class_dice=0.8000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9091, per_class_sd=['0.9091'], combined(w=0.50)=0.8545, balanced=0.8066
[2026-06-23 18:10:06] INFO segtask_v1.trainer.trainer: Epoch 55/400 | LR=9.61e-04 | loss=0.3642 | val_dice=0.8000 | best=0.8255 (ep47) | 01:13:23 | L_main=0.0961 L_aux_1=0.0984(w=0.5) L_aux_2=0.1129(w=0.5)
[2026-06-23 18:10:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 55): 13421.7 MiB
[2026-06-23 18:11:13] INFO segtask_v1.trainer.validation:   Val: loss=0.2745, pooled_mean_dice=0.7883, per_class=['0.7883'], iou=0.6506, recall=0.9485, precision=0.6744, vol_sim=0.8311, mcc=0.7954, min_class_dice=0.7883, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9115, per_class_sd=['0.9115'], combined(w=0.50)=0.8499, balanced=0.7974
[2026-06-23 18:11:13] INFO segtask_v1.trainer.trainer: Epoch 56/400 | LR=9.59e-04 | loss=0.3517 | val_dice=0.7883 | best=0.8255 (ep47) | 01:14:30 | L_main=0.0895 L_aux_1=0.0935(w=0.5) L_aux_2=0.1080(w=0.5)
[2026-06-23 18:11:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 56): 13421.7 MiB
[2026-06-23 18:12:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2675, pooled_mean_dice=0.7869, per_class=['0.7869'], iou=0.6487, recall=0.9513, precision=0.6710, vol_sim=0.8272, mcc=0.7938, min_class_dice=0.7869, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9049, per_class_sd=['0.9049'], combined(w=0.50)=0.8459, balanced=0.7949
[2026-06-23 18:12:32] INFO segtask_v1.trainer.trainer: Epoch 57/400 | LR=9.58e-04 | loss=0.3613 | val_dice=0.7869 | best=0.8255 (ep47) | 01:15:49 | L_main=0.0950 L_aux_1=0.0983(w=0.5) L_aux_2=0.1124(w=0.5)
[2026-06-23 18:12:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 57): 13421.7 MiB
[2026-06-23 18:13:54] INFO segtask_v1.trainer.validation:   Val: loss=0.2690, pooled_mean_dice=0.7889, per_class=['0.7889'], iou=0.6513, recall=0.9728, precision=0.6634, vol_sim=0.8110, mcc=0.7981, min_class_dice=0.7889, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.9063, per_class_sd=['0.9063'], combined(w=0.50)=0.8476, balanced=0.7970
[2026-06-23 18:13:54] INFO segtask_v1.trainer.trainer: Epoch 58/400 | LR=9.56e-04 | loss=0.3580 | val_dice=0.7889 | best=0.8255 (ep47) | 01:17:11 | L_main=0.0940 L_aux_1=0.0968(w=0.5) L_aux_2=0.1101(w=0.5)
[2026-06-23 18:13:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 58): 13421.7 MiB
[2026-06-23 18:15:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2890, pooled_mean_dice=0.7854, per_class=['0.7854'], iou=0.6466, recall=0.9500, precision=0.6693, vol_sim=0.8267, mcc=0.7922, min_class_dice=0.7854, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9036, per_class_sd=['0.9036'], combined(w=0.50)=0.8445, balanced=0.7933
[2026-06-23 18:15:04] INFO segtask_v1.trainer.trainer: Epoch 59/400 | LR=9.55e-04 | loss=0.3586 | val_dice=0.7854 | best=0.8255 (ep47) | 01:18:21 | L_main=0.0913 L_aux_1=0.0944(w=0.5) L_aux_2=0.1085(w=0.5)
[2026-06-23 18:15:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 59): 13421.7 MiB
[2026-06-23 18:16:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2667, pooled_mean_dice=0.7909, per_class=['0.7909'], iou=0.6541, recall=0.9631, precision=0.6709, vol_sim=0.8212, mcc=0.7995, min_class_dice=0.7909, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9151, per_class_sd=['0.9151'], combined(w=0.50)=0.8530, balanced=0.8004
[2026-06-23 18:16:26] INFO segtask_v1.trainer.trainer: Epoch 60/400 | LR=9.53e-04 | loss=0.4044 | val_dice=0.7909 | best=0.8255 (ep47) | 01:19:43 | L_main=0.1134 L_aux_1=0.1158(w=0.5) L_aux_2=0.1322(w=0.5)
[2026-06-23 18:16:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 60): 13421.7 MiB
[2026-06-23 18:17:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2737, pooled_mean_dice=0.7927, per_class=['0.7927'], iou=0.6566, recall=0.9738, precision=0.6684, vol_sim=0.8140, mcc=0.8004, min_class_dice=0.7927, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9015, per_class_sd=['0.9015'], combined(w=0.50)=0.8471, balanced=0.7993
[2026-06-23 18:17:48] INFO segtask_v1.trainer.trainer: Epoch 61/400 | LR=9.51e-04 | loss=0.3578 | val_dice=0.7927 | best=0.8255 (ep47) | 01:21:05 | L_main=0.0915 L_aux_1=0.0945(w=0.5) L_aux_2=0.1090(w=0.5)
[2026-06-23 18:17:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 61): 13421.7 MiB
[2026-06-23 18:19:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2908, pooled_mean_dice=0.7715, per_class=['0.7715'], iou=0.6280, recall=0.9704, precision=0.6402, vol_sim=0.7950, mcc=0.7821, min_class_dice=0.7715, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8923, per_class_sd=['0.8923'], combined(w=0.50)=0.8319, balanced=0.7795
[2026-06-23 18:19:01] INFO segtask_v1.trainer.trainer: Epoch 62/400 | LR=9.50e-04 | loss=0.3700 | val_dice=0.7715 | best=0.8255 (ep47) | 01:22:18 | L_main=0.0962 L_aux_1=0.0996(w=0.5) L_aux_2=0.1143(w=0.5)
[2026-06-23 18:19:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 62): 13421.7 MiB
[2026-06-23 18:20:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2621, pooled_mean_dice=0.7916, per_class=['0.7916'], iou=0.6550, recall=0.9739, precision=0.6668, vol_sim=0.8128, mcc=0.7998, min_class_dice=0.7916, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.8996, per_class_sd=['0.8996'], combined(w=0.50)=0.8456, balanced=0.7980
[2026-06-23 18:20:20] INFO segtask_v1.trainer.trainer: Epoch 63/400 | LR=9.48e-04 | loss=0.3673 | val_dice=0.7916 | best=0.8255 (ep47) | 01:23:37 | L_main=0.0975 L_aux_1=0.1009(w=0.5) L_aux_2=0.1158(w=0.5)
[2026-06-23 18:20:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 63): 13421.7 MiB
[2026-06-23 18:21:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2702, pooled_mean_dice=0.7868, per_class=['0.7868'], iou=0.6485, recall=0.9783, precision=0.6579, vol_sim=0.8042, mcc=0.7966, min_class_dice=0.7868, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9020, per_class_sd=['0.9020'], combined(w=0.50)=0.8444, balanced=0.7945
[2026-06-23 18:21:31] INFO segtask_v1.trainer.trainer: Epoch 64/400 | LR=9.46e-04 | loss=0.3565 | val_dice=0.7868 | best=0.8255 (ep47) | 01:24:48 | L_main=0.0928 L_aux_1=0.0967(w=0.5) L_aux_2=0.1112(w=0.5)
[2026-06-23 18:21:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 64): 13421.7 MiB
[2026-06-23 18:22:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2736, pooled_mean_dice=0.7791, per_class=['0.7791'], iou=0.6381, recall=0.9680, precision=0.6519, vol_sim=0.8049, mcc=0.7882, min_class_dice=0.7791, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9000, per_class_sd=['0.9000'], combined(w=0.50)=0.8396, balanced=0.7874
[2026-06-23 18:22:46] INFO segtask_v1.trainer.trainer: Epoch 65/400 | LR=9.44e-04 | loss=0.3540 | val_dice=0.7791 | best=0.8255 (ep47) | 01:26:03 | L_main=0.0929 L_aux_1=0.0958(w=0.5) L_aux_2=0.1100(w=0.5)
[2026-06-23 18:22:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 65): 13421.7 MiB
[2026-06-23 18:24:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2761, pooled_mean_dice=0.7664, per_class=['0.7664'], iou=0.6212, recall=0.9747, precision=0.6314, vol_sim=0.7863, mcc=0.7792, min_class_dice=0.7664, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8960, per_class_sd=['0.8960'], combined(w=0.50)=0.8312, balanced=0.7760
[2026-06-23 18:24:06] INFO segtask_v1.trainer.trainer: Epoch 66/400 | LR=9.42e-04 | loss=0.3904 | val_dice=0.7664 | best=0.8255 (ep47) | 01:27:22 | L_main=0.1082 L_aux_1=0.1109(w=0.5) L_aux_2=0.1251(w=0.5)
[2026-06-23 18:24:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 66): 13421.7 MiB
[2026-06-23 18:25:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2615, pooled_mean_dice=0.7865, per_class=['0.7865'], iou=0.6482, recall=0.9853, precision=0.6545, vol_sim=0.7983, mcc=0.7970, min_class_dice=0.7865, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8927, per_class_sd=['0.8927'], combined(w=0.50)=0.8396, balanced=0.7925
[2026-06-23 18:25:16] INFO segtask_v1.trainer.trainer: Epoch 67/400 | LR=9.40e-04 | loss=0.3822 | val_dice=0.7865 | best=0.8255 (ep47) | 01:28:33 | L_main=0.1051 L_aux_1=0.1079(w=0.5) L_aux_2=0.1234(w=0.5)
[2026-06-23 18:25:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 67): 13421.7 MiB
[2026-06-23 18:26:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2682, pooled_mean_dice=0.7852, per_class=['0.7852'], iou=0.6464, recall=0.9725, precision=0.6584, vol_sim=0.8074, mcc=0.7952, min_class_dice=0.7852, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9024, per_class_sd=['0.9024'], combined(w=0.50)=0.8438, balanced=0.7932
[2026-06-23 18:26:30] INFO segtask_v1.trainer.trainer: Epoch 68/400 | LR=9.39e-04 | loss=0.3509 | val_dice=0.7852 | best=0.8255 (ep47) | 01:29:47 | L_main=0.0894 L_aux_1=0.0937(w=0.5) L_aux_2=0.1074(w=0.5)
[2026-06-23 18:26:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 68): 13421.7 MiB
[2026-06-23 18:27:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2784, pooled_mean_dice=0.7762, per_class=['0.7762'], iou=0.6343, recall=0.9807, precision=0.6423, vol_sim=0.7915, mcc=0.7877, min_class_dice=0.7762, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8931, per_class_sd=['0.8931'], combined(w=0.50)=0.8347, balanced=0.7838
[2026-06-23 18:27:43] INFO segtask_v1.trainer.trainer: Epoch 69/400 | LR=9.37e-04 | loss=0.3355 | val_dice=0.7762 | best=0.8255 (ep47) | 01:31:00 | L_main=0.0840 L_aux_1=0.0873(w=0.5) L_aux_2=0.0999(w=0.5)
[2026-06-23 18:27:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 69): 13421.7 MiB
[2026-06-23 18:28:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2721, pooled_mean_dice=0.7904, per_class=['0.7904'], iou=0.6534, recall=0.9801, precision=0.6622, vol_sim=0.8065, mcc=0.8001, min_class_dice=0.7904, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9003, per_class_sd=['0.9003'], combined(w=0.50)=0.8454, balanced=0.7972
[2026-06-23 18:28:56] INFO segtask_v1.trainer.trainer: Epoch 70/400 | LR=9.35e-04 | loss=0.3500 | val_dice=0.7904 | best=0.8255 (ep47) | 01:32:13 | L_main=0.0889 L_aux_1=0.0928(w=0.5) L_aux_2=0.1062(w=0.5)
[2026-06-23 18:28:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 70): 13421.7 MiB
[2026-06-23 18:30:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2687, pooled_mean_dice=0.7794, per_class=['0.7794'], iou=0.6385, recall=0.9806, precision=0.6467, vol_sim=0.7948, mcc=0.7906, min_class_dice=0.7794, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8992, per_class_sd=['0.8992'], combined(w=0.50)=0.8393, balanced=0.7877
[2026-06-23 18:30:16] INFO segtask_v1.trainer.trainer: Epoch 71/400 | LR=9.33e-04 | loss=0.3676 | val_dice=0.7794 | best=0.8255 (ep47) | 01:33:33 | L_main=0.0994 L_aux_1=0.1012(w=0.5) L_aux_2=0.1146(w=0.5)
[2026-06-23 18:30:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 71): 13421.7 MiB
[2026-06-23 18:31:38] INFO segtask_v1.trainer.validation:   Val: loss=0.2814, pooled_mean_dice=0.7725, per_class=['0.7725'], iou=0.6294, recall=0.9758, precision=0.6394, vol_sim=0.7917, mcc=0.7841, min_class_dice=0.7725, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8939, per_class_sd=['0.8939'], combined(w=0.50)=0.8332, balanced=0.7808
[2026-06-23 18:31:38] INFO segtask_v1.trainer.trainer: Epoch 72/400 | LR=9.31e-04 | loss=0.3767 | val_dice=0.7725 | best=0.8255 (ep47) | 01:34:55 | L_main=0.0997 L_aux_1=0.1032(w=0.5) L_aux_2=0.1170(w=0.5)
[2026-06-23 18:31:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 72): 13421.7 MiB
[2026-06-23 18:32:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2686, pooled_mean_dice=0.7814, per_class=['0.7814'], iou=0.6412, recall=0.9807, precision=0.6494, vol_sim=0.7968, mcc=0.7928, min_class_dice=0.7814, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8996, per_class_sd=['0.8996'], combined(w=0.50)=0.8405, balanced=0.7895
[2026-06-23 18:32:53] INFO segtask_v1.trainer.trainer: Epoch 73/400 | LR=9.29e-04 | loss=0.3442 | val_dice=0.7814 | best=0.8255 (ep47) | 01:36:10 | L_main=0.0844 L_aux_1=0.0890(w=0.5) L_aux_2=0.1030(w=0.5)
[2026-06-23 18:32:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 73): 13421.7 MiB
[2026-06-23 18:34:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2774, pooled_mean_dice=0.7831, per_class=['0.7831'], iou=0.6435, recall=0.9715, precision=0.6559, vol_sim=0.8060, mcc=0.7923, min_class_dice=0.7831, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8939, per_class_sd=['0.8939'], combined(w=0.50)=0.8385, balanced=0.7897
[2026-06-23 18:34:08] INFO segtask_v1.trainer.trainer: Epoch 74/400 | LR=9.27e-04 | loss=0.3248 | val_dice=0.7831 | best=0.8255 (ep47) | 01:37:24 | L_main=0.0792 L_aux_1=0.0822(w=0.5) L_aux_2=0.0945(w=0.5)
[2026-06-23 18:34:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 74): 13421.7 MiB
[2026-06-23 18:35:21] INFO segtask_v1.trainer.validation:   Val: loss=0.3002, pooled_mean_dice=0.7612, per_class=['0.7612'], iou=0.6145, recall=0.9806, precision=0.6221, vol_sim=0.7763, mcc=0.7751, min_class_dice=0.7612, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8901, per_class_sd=['0.8901'], combined(w=0.50)=0.8257, balanced=0.7705
[2026-06-23 18:35:21] INFO segtask_v1.trainer.trainer: Epoch 75/400 | LR=9.25e-04 | loss=0.3517 | val_dice=0.7612 | best=0.8255 (ep47) | 01:38:37 | L_main=0.0887 L_aux_1=0.0922(w=0.5) L_aux_2=0.1060(w=0.5)
[2026-06-23 18:35:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 75): 13421.7 MiB
[2026-06-23 18:36:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2640, pooled_mean_dice=0.7792, per_class=['0.7792'], iou=0.6382, recall=0.9822, precision=0.6457, vol_sim=0.7933, mcc=0.7890, min_class_dice=0.7792, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8863, per_class_sd=['0.8863'], combined(w=0.50)=0.8327, balanced=0.7849
[2026-06-23 18:36:30] INFO segtask_v1.trainer.trainer: Epoch 76/400 | LR=9.22e-04 | loss=0.3462 | val_dice=0.7792 | best=0.8255 (ep47) | 01:39:47 | L_main=0.0878 L_aux_1=0.0919(w=0.5) L_aux_2=0.1054(w=0.5)
[2026-06-23 18:36:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 76): 13421.7 MiB
[2026-06-23 18:37:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2973, pooled_mean_dice=0.7555, per_class=['0.7555'], iou=0.6071, recall=0.9804, precision=0.6146, vol_sim=0.7707, mcc=0.7699, min_class_dice=0.7555, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8807, per_class_sd=['0.8807'], combined(w=0.50)=0.8181, balanced=0.7639
[2026-06-23 18:37:47] INFO segtask_v1.trainer.trainer: Epoch 77/400 | LR=9.20e-04 | loss=0.3292 | val_dice=0.7555 | best=0.8255 (ep47) | 01:41:04 | L_main=0.0804 L_aux_1=0.0838(w=0.5) L_aux_2=0.0962(w=0.5)
[2026-06-23 18:37:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 77): 13421.7 MiB
[2026-06-23 18:38:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2874, pooled_mean_dice=0.7620, per_class=['0.7620'], iou=0.6155, recall=0.9753, precision=0.6252, vol_sim=0.7812, mcc=0.7747, min_class_dice=0.7620, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8903, per_class_sd=['0.8903'], combined(w=0.50)=0.8261, balanced=0.7711
[2026-06-23 18:38:58] INFO segtask_v1.trainer.trainer: Epoch 78/400 | LR=9.18e-04 | loss=0.3485 | val_dice=0.7620 | best=0.8255 (ep47) | 01:42:15 | L_main=0.0882 L_aux_1=0.0917(w=0.5) L_aux_2=0.1052(w=0.5)
[2026-06-23 18:38:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 78): 13421.7 MiB
[2026-06-23 18:40:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2638, pooled_mean_dice=0.7797, per_class=['0.7797'], iou=0.6389, recall=0.9837, precision=0.6457, vol_sim=0.7926, mcc=0.7903, min_class_dice=0.7797, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8848, per_class_sd=['0.8848'], combined(w=0.50)=0.8323, balanced=0.7851
[2026-06-23 18:40:14] INFO segtask_v1.trainer.trainer: Epoch 79/400 | LR=9.16e-04 | loss=0.3402 | val_dice=0.7797 | best=0.8255 (ep47) | 01:43:31 | L_main=0.0853 L_aux_1=0.0891(w=0.5) L_aux_2=0.1025(w=0.5)
[2026-06-23 18:40:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 79): 13421.7 MiB
[2026-06-23 18:41:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2980, pooled_mean_dice=0.7572, per_class=['0.7572'], iou=0.6092, recall=0.9836, precision=0.6155, vol_sim=0.7698, mcc=0.7714, min_class_dice=0.7572, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8840, per_class_sd=['0.8840'], combined(w=0.50)=0.8206, balanced=0.7659
[2026-06-23 18:41:22] INFO segtask_v1.trainer.trainer: Epoch 80/400 | LR=9.14e-04 | loss=0.3319 | val_dice=0.7572 | best=0.8255 (ep47) | 01:44:39 | L_main=0.0814 L_aux_1=0.0856(w=0.5) L_aux_2=0.0987(w=0.5)
[2026-06-23 18:41:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 80): 13421.7 MiB
[2026-06-23 18:42:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2944, pooled_mean_dice=0.7617, per_class=['0.7617'], iou=0.6152, recall=0.9803, precision=0.6229, vol_sim=0.7771, mcc=0.7760, min_class_dice=0.7617, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8864, per_class_sd=['0.8864'], combined(w=0.50)=0.8241, balanced=0.7703
[2026-06-23 18:42:33] INFO segtask_v1.trainer.trainer: Epoch 81/400 | LR=9.11e-04 | loss=0.3197 | val_dice=0.7617 | best=0.8255 (ep47) | 01:45:50 | L_main=0.0780 L_aux_1=0.0817(w=0.5) L_aux_2=0.0947(w=0.5)
[2026-06-23 18:42:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 81): 13421.7 MiB
[2026-06-23 18:43:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2754, pooled_mean_dice=0.7725, per_class=['0.7725'], iou=0.6294, recall=0.9837, precision=0.6360, vol_sim=0.7854, mcc=0.7843, min_class_dice=0.7725, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8876, per_class_sd=['0.8876'], combined(w=0.50)=0.8301, balanced=0.7796
[2026-06-23 18:43:50] INFO segtask_v1.trainer.trainer: Epoch 82/400 | LR=9.09e-04 | loss=0.3387 | val_dice=0.7725 | best=0.8255 (ep47) | 01:47:07 | L_main=0.0857 L_aux_1=0.0887(w=0.5) L_aux_2=0.1018(w=0.5)
[2026-06-23 18:43:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 82): 13421.7 MiB
[2026-06-23 18:44:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2986, pooled_mean_dice=0.7559, per_class=['0.7559'], iou=0.6076, recall=0.9846, precision=0.6135, vol_sim=0.7678, mcc=0.7719, min_class_dice=0.7559, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8932, per_class_sd=['0.8932'], combined(w=0.50)=0.8246, balanced=0.7668
[2026-06-23 18:44:59] INFO segtask_v1.trainer.trainer: Epoch 83/400 | LR=9.07e-04 | loss=0.3378 | val_dice=0.7559 | best=0.8255 (ep47) | 01:48:16 | L_main=0.0833 L_aux_1=0.0877(w=0.5) L_aux_2=0.1011(w=0.5)
[2026-06-23 18:44:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 83): 13421.7 MiB
[2026-06-23 18:46:13] INFO segtask_v1.trainer.validation:   Val: loss=0.2981, pooled_mean_dice=0.7464, per_class=['0.7464'], iou=0.5954, recall=0.9848, precision=0.6009, vol_sim=0.7579, mcc=0.7637, min_class_dice=0.7464, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8859, per_class_sd=['0.8859'], combined(w=0.50)=0.8161, balanced=0.7573
[2026-06-23 18:46:13] INFO segtask_v1.trainer.trainer: Epoch 84/400 | LR=9.05e-04 | loss=0.3616 | val_dice=0.7464 | best=0.8255 (ep47) | 01:49:30 | L_main=0.0915 L_aux_1=0.0958(w=0.5) L_aux_2=0.1097(w=0.5)
[2026-06-23 18:46:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 84): 13421.7 MiB
[2026-06-23 18:47:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2889, pooled_mean_dice=0.7675, per_class=['0.7675'], iou=0.6227, recall=0.9845, precision=0.6289, vol_sim=0.7796, mcc=0.7811, min_class_dice=0.7675, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8891, per_class_sd=['0.8891'], combined(w=0.50)=0.8283, balanced=0.7757
[2026-06-23 18:47:21] INFO segtask_v1.trainer.trainer: Epoch 85/400 | LR=9.02e-04 | loss=0.3591 | val_dice=0.7675 | best=0.8255 (ep47) | 01:50:38 | L_main=0.0948 L_aux_1=0.0970(w=0.5) L_aux_2=0.1101(w=0.5)
[2026-06-23 18:47:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 85): 13421.7 MiB
[2026-06-23 18:48:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3063, pooled_mean_dice=0.7484, per_class=['0.7484'], iou=0.5979, recall=0.9853, precision=0.6033, vol_sim=0.7595, mcc=0.7652, min_class_dice=0.7484, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8782, per_class_sd=['0.8782'], combined(w=0.50)=0.8133, balanced=0.7575
[2026-06-23 18:48:31] INFO segtask_v1.trainer.trainer: Epoch 86/400 | LR=9.00e-04 | loss=0.3401 | val_dice=0.7484 | best=0.8255 (ep47) | 01:51:48 | L_main=0.0856 L_aux_1=0.0889(w=0.5) L_aux_2=0.1017(w=0.5)
[2026-06-23 18:48:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 86): 13421.7 MiB
[2026-06-23 18:49:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2955, pooled_mean_dice=0.7518, per_class=['0.7518'], iou=0.6023, recall=0.9819, precision=0.6091, vol_sim=0.7657, mcc=0.7673, min_class_dice=0.7518, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8865, per_class_sd=['0.8865'], combined(w=0.50)=0.8192, balanced=0.7619
[2026-06-23 18:49:49] INFO segtask_v1.trainer.trainer: Epoch 87/400 | LR=8.97e-04 | loss=0.3317 | val_dice=0.7518 | best=0.8255 (ep47) | 01:53:06 | L_main=0.0818 L_aux_1=0.0851(w=0.5) L_aux_2=0.0973(w=0.5)
[2026-06-23 18:49:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 87): 13421.7 MiB
[2026-06-23 18:51:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2695, pooled_mean_dice=0.7803, per_class=['0.7803'], iou=0.6397, recall=0.9845, precision=0.6463, vol_sim=0.7926, mcc=0.7927, min_class_dice=0.7803, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.8894, per_class_sd=['0.8894'], combined(w=0.50)=0.8349, balanced=0.7867
[2026-06-23 18:51:01] INFO segtask_v1.trainer.trainer: Epoch 88/400 | LR=8.95e-04 | loss=0.3209 | val_dice=0.7803 | best=0.8255 (ep47) | 01:54:18 | L_main=0.0757 L_aux_1=0.0800(w=0.5) L_aux_2=0.0917(w=0.5)
[2026-06-23 18:51:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 88): 13421.7 MiB
[2026-06-23 18:52:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2758, pooled_mean_dice=0.7653, per_class=['0.7653'], iou=0.6198, recall=0.9900, precision=0.6237, vol_sim=0.7730, mcc=0.7795, min_class_dice=0.7653, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8845, per_class_sd=['0.8845'], combined(w=0.50)=0.8249, balanced=0.7730
[2026-06-23 18:52:12] INFO segtask_v1.trainer.trainer: Epoch 89/400 | LR=8.93e-04 | loss=0.3208 | val_dice=0.7653 | best=0.8255 (ep47) | 01:55:28 | L_main=0.0775 L_aux_1=0.0811(w=0.5) L_aux_2=0.0934(w=0.5)
[2026-06-23 18:52:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 89): 13421.7 MiB
[2026-06-23 18:53:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2921, pooled_mean_dice=0.7668, per_class=['0.7668'], iou=0.6218, recall=0.9891, precision=0.6261, vol_sim=0.7753, mcc=0.7805, min_class_dice=0.7668, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8864, per_class_sd=['0.8864'], combined(w=0.50)=0.8266, balanced=0.7747
[2026-06-23 18:53:20] INFO segtask_v1.trainer.trainer: Epoch 90/400 | LR=8.90e-04 | loss=0.3341 | val_dice=0.7668 | best=0.8255 (ep47) | 01:56:36 | L_main=0.0830 L_aux_1=0.0864(w=0.5) L_aux_2=0.0984(w=0.5)
[2026-06-23 18:53:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 90): 13421.7 MiB
[2026-06-23 18:54:35] INFO segtask_v1.trainer.validation:   Val: loss=0.3182, pooled_mean_dice=0.7385, per_class=['0.7385'], iou=0.5855, recall=0.9863, precision=0.5903, vol_sim=0.7488, mcc=0.7583, min_class_dice=0.7385, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8955, per_class_sd=['0.8955'], combined(w=0.50)=0.8170, balanced=0.7524
[2026-06-23 18:54:35] INFO segtask_v1.trainer.trainer: Epoch 91/400 | LR=8.88e-04 | loss=0.3453 | val_dice=0.7385 | best=0.8255 (ep47) | 01:57:52 | L_main=0.0901 L_aux_1=0.0933(w=0.5) L_aux_2=0.1059(w=0.5)
[2026-06-23 18:54:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 91): 13421.7 MiB
[2026-06-23 18:55:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2606, pooled_mean_dice=0.7842, per_class=['0.7842'], iou=0.6450, recall=0.9887, precision=0.6498, vol_sim=0.7932, mcc=0.7940, min_class_dice=0.7842, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8883, per_class_sd=['0.8883'], combined(w=0.50)=0.8363, balanced=0.7896
[2026-06-23 18:55:47] INFO segtask_v1.trainer.trainer: Epoch 92/400 | LR=8.85e-04 | loss=0.3395 | val_dice=0.7842 | best=0.8255 (ep47) | 01:59:04 | L_main=0.0842 L_aux_1=0.0873(w=0.5) L_aux_2=0.1009(w=0.5)
[2026-06-23 18:55:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 92): 13421.7 MiB
[2026-06-23 18:56:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2568, pooled_mean_dice=0.7837, per_class=['0.7837'], iou=0.6443, recall=0.9894, precision=0.6488, vol_sim=0.7921, mcc=0.7956, min_class_dice=0.7837, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9034, per_class_sd=['0.9034'], combined(w=0.50)=0.8435, balanced=0.7923
[2026-06-23 18:56:57] INFO segtask_v1.trainer.trainer: Epoch 93/400 | LR=8.83e-04 | loss=0.3182 | val_dice=0.7837 | best=0.8255 (ep47) | 02:00:13 | L_main=0.0765 L_aux_1=0.0798(w=0.5) L_aux_2=0.0919(w=0.5)
[2026-06-23 18:56:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 93): 13421.7 MiB
[2026-06-23 18:58:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2675, pooled_mean_dice=0.7785, per_class=['0.7785'], iou=0.6374, recall=0.9884, precision=0.6422, vol_sim=0.7876, mcc=0.7909, min_class_dice=0.7785, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8892, per_class_sd=['0.8892'], combined(w=0.50)=0.8338, balanced=0.7851
[2026-06-23 18:58:17] INFO segtask_v1.trainer.trainer: Epoch 94/400 | LR=8.80e-04 | loss=0.3343 | val_dice=0.7785 | best=0.8255 (ep47) | 02:01:34 | L_main=0.0825 L_aux_1=0.0866(w=0.5) L_aux_2=0.1001(w=0.5)
[2026-06-23 18:58:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 94): 13421.7 MiB
[2026-06-23 18:59:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2677, pooled_mean_dice=0.7736, per_class=['0.7736'], iou=0.6308, recall=0.9884, precision=0.6355, vol_sim=0.7827, mcc=0.7853, min_class_dice=0.7736, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8840, per_class_sd=['0.8840'], combined(w=0.50)=0.8288, balanced=0.7798
[2026-06-23 18:59:35] INFO segtask_v1.trainer.trainer: Epoch 95/400 | LR=8.77e-04 | loss=0.3151 | val_dice=0.7736 | best=0.8255 (ep47) | 02:02:52 | L_main=0.0742 L_aux_1=0.0783(w=0.5) L_aux_2=0.0902(w=0.5)
[2026-06-23 18:59:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 95): 13421.7 MiB
[2026-06-23 19:00:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2724, pooled_mean_dice=0.7667, per_class=['0.7667'], iou=0.6217, recall=0.9878, precision=0.6265, vol_sim=0.7762, mcc=0.7807, min_class_dice=0.7667, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8908, per_class_sd=['0.8908'], combined(w=0.50)=0.8288, balanced=0.7754
[2026-06-23 19:00:45] INFO segtask_v1.trainer.trainer: Epoch 96/400 | LR=8.75e-04 | loss=0.3320 | val_dice=0.7667 | best=0.8255 (ep47) | 02:04:01 | L_main=0.0812 L_aux_1=0.0849(w=0.5) L_aux_2=0.0977(w=0.5)
[2026-06-23 19:00:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 96): 13421.7 MiB
[2026-06-23 19:01:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2904, pooled_mean_dice=0.7524, per_class=['0.7524'], iou=0.6030, recall=0.9878, precision=0.6076, vol_sim=0.7617, mcc=0.7698, min_class_dice=0.7524, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.8831, per_class_sd=['0.8831'], combined(w=0.50)=0.8177, balanced=0.7619
[2026-06-23 19:01:55] INFO segtask_v1.trainer.trainer: Epoch 97/400 | LR=8.72e-04 | loss=0.3319 | val_dice=0.7524 | best=0.8255 (ep47) | 02:05:12 | L_main=0.0826 L_aux_1=0.0863(w=0.5) L_aux_2=0.0993(w=0.5)
[2026-06-23 19:01:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 97): 13421.7 MiB
[2026-06-23 19:03:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2549, pooled_mean_dice=0.7817, per_class=['0.7817'], iou=0.6416, recall=0.9879, precision=0.6467, vol_sim=0.7913, mcc=0.7925, min_class_dice=0.7817, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8924, per_class_sd=['0.8924'], combined(w=0.50)=0.8370, balanced=0.7883
[2026-06-23 19:03:04] INFO segtask_v1.trainer.trainer: Epoch 98/400 | LR=8.69e-04 | loss=0.3291 | val_dice=0.7817 | best=0.8255 (ep47) | 02:06:21 | L_main=0.0814 L_aux_1=0.0852(w=0.5) L_aux_2=0.0976(w=0.5)
[2026-06-23 19:03:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 98): 13421.7 MiB
[2026-06-23 19:04:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2912, pooled_mean_dice=0.7511, per_class=['0.7511'], iou=0.6015, recall=0.9865, precision=0.6065, vol_sim=0.7614, mcc=0.7675, min_class_dice=0.7511, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8888, per_class_sd=['0.8888'], combined(w=0.50)=0.8200, balanced=0.7618
[2026-06-23 19:04:18] INFO segtask_v1.trainer.trainer: Epoch 99/400 | LR=8.67e-04 | loss=0.3144 | val_dice=0.7511 | best=0.8255 (ep47) | 02:07:35 | L_main=0.0746 L_aux_1=0.0787(w=0.5) L_aux_2=0.0904(w=0.5)
[2026-06-23 19:04:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 99): 13421.7 MiB
[2026-06-23 19:05:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2680, pooled_mean_dice=0.7703, per_class=['0.7703'], iou=0.6265, recall=0.9861, precision=0.6321, vol_sim=0.7812, mcc=0.7838, min_class_dice=0.7703, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8876, per_class_sd=['0.8876'], combined(w=0.50)=0.8290, balanced=0.7779
[2026-06-23 19:05:35] INFO segtask_v1.trainer.trainer: Epoch 100/400 | LR=8.64e-04 | loss=0.3057 | val_dice=0.7703 | best=0.8255 (ep47) | 02:08:52 | L_main=0.0724 L_aux_1=0.0754(w=0.5) L_aux_2=0.0864(w=0.5)
[2026-06-23 19:05:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 100): 13421.7 MiB
[2026-06-23 19:06:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2634, pooled_mean_dice=0.7790, per_class=['0.7790'], iou=0.6380, recall=0.9887, precision=0.6427, vol_sim=0.7879, mcc=0.7907, min_class_dice=0.7790, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8935, per_class_sd=['0.8935'], combined(w=0.50)=0.8363, balanced=0.7863
[2026-06-23 19:06:53] INFO segtask_v1.trainer.trainer: Epoch 101/400 | LR=8.61e-04 | loss=0.3363 | val_dice=0.7790 | best=0.8255 (ep47) | 02:10:10 | L_main=0.0846 L_aux_1=0.0881(w=0.5) L_aux_2=0.1002(w=0.5)
[2026-06-23 19:06:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 101): 13421.7 MiB
[2026-06-23 19:08:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2805, pooled_mean_dice=0.7749, per_class=['0.7749'], iou=0.6326, recall=0.9895, precision=0.6368, vol_sim=0.7831, mcc=0.7874, min_class_dice=0.7749, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8874, per_class_sd=['0.8874'], combined(w=0.50)=0.8312, balanced=0.7817
[2026-06-23 19:08:12] INFO segtask_v1.trainer.trainer: Epoch 102/400 | LR=8.59e-04 | loss=0.3108 | val_dice=0.7749 | best=0.8255 (ep47) | 02:11:29 | L_main=0.0729 L_aux_1=0.0767(w=0.5) L_aux_2=0.0883(w=0.5)
[2026-06-23 19:08:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 102): 13421.7 MiB
[2026-06-23 19:09:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3046, pooled_mean_dice=0.7494, per_class=['0.7494'], iou=0.5992, recall=0.9867, precision=0.6041, vol_sim=0.7595, mcc=0.7665, min_class_dice=0.7494, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8829, per_class_sd=['0.8829'], combined(w=0.50)=0.8161, balanced=0.7593
[2026-06-23 19:09:25] INFO segtask_v1.trainer.trainer: Epoch 103/400 | LR=8.56e-04 | loss=0.3234 | val_dice=0.7494 | best=0.8255 (ep47) | 02:12:42 | L_main=0.0783 L_aux_1=0.0814(w=0.5) L_aux_2=0.0927(w=0.5)
[2026-06-23 19:09:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 103): 13421.7 MiB
[2026-06-23 19:10:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2949, pooled_mean_dice=0.7424, per_class=['0.7424'], iou=0.5903, recall=0.9896, precision=0.5940, vol_sim=0.7502, mcc=0.7613, min_class_dice=0.7424, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8741, per_class_sd=['0.8741'], combined(w=0.50)=0.8082, balanced=0.7518
[2026-06-23 19:10:41] INFO segtask_v1.trainer.trainer: Epoch 104/400 | LR=8.53e-04 | loss=0.3374 | val_dice=0.7424 | best=0.8255 (ep47) | 02:13:58 | L_main=0.0852 L_aux_1=0.0897(w=0.5) L_aux_2=0.1027(w=0.5)
[2026-06-23 19:10:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 104): 13421.7 MiB
[2026-06-23 19:11:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2911, pooled_mean_dice=0.7624, per_class=['0.7624'], iou=0.6161, recall=0.9900, precision=0.6200, vol_sim=0.7702, mcc=0.7779, min_class_dice=0.7624, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8825, per_class_sd=['0.8825'], combined(w=0.50)=0.8225, balanced=0.7703
[2026-06-23 19:11:58] INFO segtask_v1.trainer.trainer: Epoch 105/400 | LR=8.50e-04 | loss=0.3292 | val_dice=0.7624 | best=0.8255 (ep47) | 02:15:15 | L_main=0.0804 L_aux_1=0.0845(w=0.5) L_aux_2=0.0968(w=0.5)
[2026-06-23 19:11:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 105): 13421.7 MiB
[2026-06-23 19:13:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2626, pooled_mean_dice=0.7778, per_class=['0.7778'], iou=0.6364, recall=0.9889, precision=0.6410, vol_sim=0.7866, mcc=0.7909, min_class_dice=0.7778, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8923, per_class_sd=['0.8923'], combined(w=0.50)=0.8351, balanced=0.7852
[2026-06-23 19:13:18] INFO segtask_v1.trainer.trainer: Epoch 106/400 | LR=8.47e-04 | loss=0.3212 | val_dice=0.7778 | best=0.8255 (ep47) | 02:16:35 | L_main=0.0779 L_aux_1=0.0814(w=0.5) L_aux_2=0.0929(w=0.5)
[2026-06-23 19:13:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 106): 13421.7 MiB
[2026-06-23 19:14:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2660, pooled_mean_dice=0.7752, per_class=['0.7752'], iou=0.6329, recall=0.9879, precision=0.6379, vol_sim=0.7847, mcc=0.7873, min_class_dice=0.7752, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8885, per_class_sd=['0.8885'], combined(w=0.50)=0.8319, balanced=0.7821
[2026-06-23 19:14:32] INFO segtask_v1.trainer.trainer: Epoch 107/400 | LR=8.44e-04 | loss=0.3437 | val_dice=0.7752 | best=0.8255 (ep47) | 02:17:49 | L_main=0.0849 L_aux_1=0.0897(w=0.5) L_aux_2=0.1027(w=0.5)
[2026-06-23 19:14:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 107): 13421.7 MiB
[2026-06-23 19:15:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2829, pooled_mean_dice=0.7682, per_class=['0.7682'], iou=0.6237, recall=0.9861, precision=0.6292, vol_sim=0.7790, mcc=0.7826, min_class_dice=0.7682, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8930, per_class_sd=['0.8930'], combined(w=0.50)=0.8306, balanced=0.7772
[2026-06-23 19:15:46] INFO segtask_v1.trainer.trainer: Epoch 108/400 | LR=8.42e-04 | loss=0.3262 | val_dice=0.7682 | best=0.8255 (ep47) | 02:19:02 | L_main=0.0790 L_aux_1=0.0832(w=0.5) L_aux_2=0.0960(w=0.5)
[2026-06-23 19:15:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 108): 13421.7 MiB
[2026-06-23 19:17:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2868, pooled_mean_dice=0.7726, per_class=['0.7726'], iou=0.6295, recall=0.9891, precision=0.6339, vol_sim=0.7811, mcc=0.7864, min_class_dice=0.7726, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8930, per_class_sd=['0.8930'], combined(w=0.50)=0.8328, balanced=0.7809
[2026-06-23 19:17:00] INFO segtask_v1.trainer.trainer: Epoch 109/400 | LR=8.39e-04 | loss=0.3230 | val_dice=0.7726 | best=0.8255 (ep47) | 02:20:17 | L_main=0.0804 L_aux_1=0.0836(w=0.5) L_aux_2=0.0949(w=0.5)
[2026-06-23 19:17:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 109): 13421.7 MiB
[2026-06-23 19:18:11] INFO segtask_v1.trainer.validation:   Val: loss=0.2947, pooled_mean_dice=0.7532, per_class=['0.7532'], iou=0.6041, recall=0.9876, precision=0.6087, vol_sim=0.7627, mcc=0.7698, min_class_dice=0.7532, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8804, per_class_sd=['0.8804'], combined(w=0.50)=0.8168, balanced=0.7621
[2026-06-23 19:18:11] INFO segtask_v1.trainer.trainer: Epoch 110/400 | LR=8.36e-04 | loss=0.3074 | val_dice=0.7532 | best=0.8255 (ep47) | 02:21:27 | L_main=0.0728 L_aux_1=0.0764(w=0.5) L_aux_2=0.0875(w=0.5)
[2026-06-23 19:18:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 110): 13421.7 MiB
[2026-06-23 19:19:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2898, pooled_mean_dice=0.7596, per_class=['0.7596'], iou=0.6124, recall=0.9859, precision=0.6178, vol_sim=0.7705, mcc=0.7749, min_class_dice=0.7596, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8880, per_class_sd=['0.8880'], combined(w=0.50)=0.8238, balanced=0.7689
[2026-06-23 19:19:32] INFO segtask_v1.trainer.trainer: Epoch 111/400 | LR=8.33e-04 | loss=0.3457 | val_dice=0.7596 | best=0.8255 (ep47) | 02:22:49 | L_main=0.0876 L_aux_1=0.0922(w=0.5) L_aux_2=0.1050(w=0.5)
[2026-06-23 19:19:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 111): 13421.7 MiB
[2026-06-23 19:20:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2856, pooled_mean_dice=0.7633, per_class=['0.7633'], iou=0.6172, recall=0.9875, precision=0.6221, vol_sim=0.7730, mcc=0.7788, min_class_dice=0.7633, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8882, per_class_sd=['0.8882'], combined(w=0.50)=0.8257, balanced=0.7721
[2026-06-23 19:20:47] INFO segtask_v1.trainer.trainer: Epoch 112/400 | LR=8.30e-04 | loss=0.3127 | val_dice=0.7633 | best=0.8255 (ep47) | 02:24:03 | L_main=0.0750 L_aux_1=0.0789(w=0.5) L_aux_2=0.0903(w=0.5)
[2026-06-23 19:20:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 112): 13421.7 MiB
[2026-06-23 19:22:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2655, pooled_mean_dice=0.7733, per_class=['0.7733'], iou=0.6304, recall=0.9859, precision=0.6362, vol_sim=0.7844, mcc=0.7872, min_class_dice=0.7733, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8958, per_class_sd=['0.8958'], combined(w=0.50)=0.8346, balanced=0.7821
[2026-06-23 19:22:04] INFO segtask_v1.trainer.trainer: Epoch 113/400 | LR=8.27e-04 | loss=0.3142 | val_dice=0.7733 | best=0.8255 (ep47) | 02:25:21 | L_main=0.0754 L_aux_1=0.0789(w=0.5) L_aux_2=0.0901(w=0.5)
[2026-06-23 19:22:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 113): 13421.7 MiB
[2026-06-23 19:23:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2815, pooled_mean_dice=0.7624, per_class=['0.7624'], iou=0.6161, recall=0.9892, precision=0.6202, vol_sim=0.7707, mcc=0.7764, min_class_dice=0.7624, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8896, per_class_sd=['0.8896'], combined(w=0.50)=0.8260, balanced=0.7715
[2026-06-23 19:23:17] INFO segtask_v1.trainer.trainer: Epoch 114/400 | LR=8.24e-04 | loss=0.3078 | val_dice=0.7624 | best=0.8255 (ep47) | 02:26:33 | L_main=0.0730 L_aux_1=0.0771(w=0.5) L_aux_2=0.0888(w=0.5)
[2026-06-23 19:23:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 114): 13421.7 MiB
[2026-06-23 19:24:29] INFO segtask_v1.trainer.validation:   Val: loss=0.2597, pooled_mean_dice=0.7767, per_class=['0.7767'], iou=0.6349, recall=0.9893, precision=0.6393, vol_sim=0.7851, mcc=0.7893, min_class_dice=0.7767, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8996, per_class_sd=['0.8996'], combined(w=0.50)=0.8382, balanced=0.7856
[2026-06-23 19:24:29] INFO segtask_v1.trainer.trainer: Epoch 115/400 | LR=8.21e-04 | loss=0.3125 | val_dice=0.7767 | best=0.8255 (ep47) | 02:27:46 | L_main=0.0743 L_aux_1=0.0779(w=0.5) L_aux_2=0.0887(w=0.5)
[2026-06-23 19:24:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 115): 13421.7 MiB
[2026-06-23 19:25:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2973, pooled_mean_dice=0.7532, per_class=['0.7532'], iou=0.6041, recall=0.9869, precision=0.6090, vol_sim=0.7632, mcc=0.7704, min_class_dice=0.7532, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8906, per_class_sd=['0.8906'], combined(w=0.50)=0.8219, balanced=0.7640
[2026-06-23 19:25:48] INFO segtask_v1.trainer.trainer: Epoch 116/400 | LR=8.18e-04 | loss=0.3049 | val_dice=0.7532 | best=0.8255 (ep47) | 02:29:05 | L_main=0.0723 L_aux_1=0.0756(w=0.5) L_aux_2=0.0865(w=0.5)
[2026-06-23 19:25:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 116): 13421.7 MiB
[2026-06-23 19:27:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2786, pooled_mean_dice=0.7644, per_class=['0.7644'], iou=0.6186, recall=0.9875, precision=0.6235, vol_sim=0.7740, mcc=0.7785, min_class_dice=0.7644, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8888, per_class_sd=['0.8888'], combined(w=0.50)=0.8266, balanced=0.7731
[2026-06-23 19:27:09] INFO segtask_v1.trainer.trainer: Epoch 117/400 | LR=8.15e-04 | loss=0.3190 | val_dice=0.7644 | best=0.8255 (ep47) | 02:30:25 | L_main=0.0768 L_aux_1=0.0812(w=0.5) L_aux_2=0.0923(w=0.5)
[2026-06-23 19:27:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 117): 13421.7 MiB
[2026-06-23 19:28:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2571, pooled_mean_dice=0.7855, per_class=['0.7855'], iou=0.6467, recall=0.9878, precision=0.6519, vol_sim=0.7952, mcc=0.7964, min_class_dice=0.7855, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8997, per_class_sd=['0.8997'], combined(w=0.50)=0.8426, balanced=0.7930
[2026-06-23 19:28:30] INFO segtask_v1.trainer.trainer: Epoch 118/400 | LR=8.11e-04 | loss=0.3123 | val_dice=0.7855 | best=0.8255 (ep47) | 02:31:47 | L_main=0.0752 L_aux_1=0.0786(w=0.5) L_aux_2=0.0894(w=0.5)
[2026-06-23 19:28:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 118): 13421.7 MiB
[2026-06-23 19:29:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2611, pooled_mean_dice=0.7800, per_class=['0.7800'], iou=0.6394, recall=0.9862, precision=0.6451, vol_sim=0.7909, mcc=0.7923, min_class_dice=0.7800, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9044, per_class_sd=['0.9044'], combined(w=0.50)=0.8422, balanced=0.7893
[2026-06-23 19:29:47] INFO segtask_v1.trainer.trainer: Epoch 119/400 | LR=8.08e-04 | loss=0.3151 | val_dice=0.7800 | best=0.8255 (ep47) | 02:33:04 | L_main=0.0754 L_aux_1=0.0788(w=0.5) L_aux_2=0.0899(w=0.5)
[2026-06-23 19:29:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 119): 13421.7 MiB
[2026-06-23 19:31:05] INFO segtask_v1.trainer.validation:   Val: loss=0.2860, pooled_mean_dice=0.7595, per_class=['0.7595'], iou=0.6122, recall=0.9886, precision=0.6165, vol_sim=0.7682, mcc=0.7752, min_class_dice=0.7595, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9016, per_class_sd=['0.9016'], combined(w=0.50)=0.8305, balanced=0.7713
[2026-06-23 19:31:05] INFO segtask_v1.trainer.trainer: Epoch 120/400 | LR=8.05e-04 | loss=0.3218 | val_dice=0.7595 | best=0.8255 (ep47) | 02:34:22 | L_main=0.0788 L_aux_1=0.0818(w=0.5) L_aux_2=0.0937(w=0.5)
[2026-06-23 19:31:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 120): 13421.7 MiB
[2026-06-23 19:32:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2591, pooled_mean_dice=0.7812, per_class=['0.7812'], iou=0.6409, recall=0.9873, precision=0.6462, vol_sim=0.7912, mcc=0.7917, min_class_dice=0.7812, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9025, per_class_sd=['0.9025'], combined(w=0.50)=0.8419, balanced=0.7898
[2026-06-23 19:32:27] INFO segtask_v1.trainer.trainer: Epoch 121/400 | LR=8.02e-04 | loss=0.3297 | val_dice=0.7812 | best=0.8255 (ep47) | 02:35:44 | L_main=0.0819 L_aux_1=0.0858(w=0.5) L_aux_2=0.0978(w=0.5)
[2026-06-23 19:32:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 121): 13421.7 MiB
[2026-06-23 19:33:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2509, pooled_mean_dice=0.7852, per_class=['0.7852'], iou=0.6464, recall=0.9884, precision=0.6514, vol_sim=0.7945, mcc=0.7961, min_class_dice=0.7852, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9001, per_class_sd=['0.9001'], combined(w=0.50)=0.8427, balanced=0.7929
[2026-06-23 19:33:45] INFO segtask_v1.trainer.trainer: Epoch 122/400 | LR=7.99e-04 | loss=0.3181 | val_dice=0.7852 | best=0.8255 (ep47) | 02:37:02 | L_main=0.0752 L_aux_1=0.0796(w=0.5) L_aux_2=0.0912(w=0.5)
[2026-06-23 19:33:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 122): 13421.7 MiB
[2026-06-23 19:35:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2676, pooled_mean_dice=0.7715, per_class=['0.7715'], iou=0.6279, recall=0.9884, precision=0.6326, vol_sim=0.7805, mcc=0.7841, min_class_dice=0.7715, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8931, per_class_sd=['0.8931'], combined(w=0.50)=0.8323, balanced=0.7798
[2026-06-23 19:35:03] INFO segtask_v1.trainer.trainer: Epoch 123/400 | LR=7.96e-04 | loss=0.3192 | val_dice=0.7715 | best=0.8255 (ep47) | 02:38:20 | L_main=0.0778 L_aux_1=0.0824(w=0.5) L_aux_2=0.0949(w=0.5)
[2026-06-23 19:35:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 123): 13421.7 MiB
[2026-06-23 19:36:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2948, pooled_mean_dice=0.7494, per_class=['0.7494'], iou=0.5993, recall=0.9876, precision=0.6038, vol_sim=0.7588, mcc=0.7677, min_class_dice=0.7494, coverage=[67]/88 samples, pooled_mean_surface_dice@2px=0.8870, per_class_sd=['0.8870'], combined(w=0.50)=0.8182, balanced=0.7602
[2026-06-23 19:36:16] INFO segtask_v1.trainer.trainer: Epoch 124/400 | LR=7.92e-04 | loss=0.3113 | val_dice=0.7494 | best=0.8255 (ep47) | 02:39:32 | L_main=0.0730 L_aux_1=0.0767(w=0.5) L_aux_2=0.0881(w=0.5)
[2026-06-23 19:36:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 124): 13421.7 MiB
[2026-06-23 19:37:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2692, pooled_mean_dice=0.7729, per_class=['0.7729'], iou=0.6298, recall=0.9895, precision=0.6341, vol_sim=0.7811, mcc=0.7873, min_class_dice=0.7729, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9015, per_class_sd=['0.9015'], combined(w=0.50)=0.8372, balanced=0.7828
[2026-06-23 19:37:28] INFO segtask_v1.trainer.trainer: Epoch 125/400 | LR=7.89e-04 | loss=0.3110 | val_dice=0.7729 | best=0.8255 (ep47) | 02:40:45 | L_main=0.0730 L_aux_1=0.0767(w=0.5) L_aux_2=0.0876(w=0.5)
[2026-06-23 19:37:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 125): 13421.7 MiB
[2026-06-23 19:38:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2656, pooled_mean_dice=0.7779, per_class=['0.7779'], iou=0.6365, recall=0.9864, precision=0.6421, vol_sim=0.7886, mcc=0.7909, min_class_dice=0.7779, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9050, per_class_sd=['0.9050'], combined(w=0.50)=0.8414, balanced=0.7877
[2026-06-23 19:38:44] INFO segtask_v1.trainer.trainer: Epoch 126/400 | LR=7.86e-04 | loss=0.3196 | val_dice=0.7779 | best=0.8255 (ep47) | 02:42:01 | L_main=0.0771 L_aux_1=0.0800(w=0.5) L_aux_2=0.0911(w=0.5)
[2026-06-23 19:38:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 126): 13421.7 MiB
[2026-06-23 19:40:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2614, pooled_mean_dice=0.7744, per_class=['0.7744'], iou=0.6318, recall=0.9895, precision=0.6361, vol_sim=0.7826, mcc=0.7876, min_class_dice=0.7744, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8958, per_class_sd=['0.8958'], combined(w=0.50)=0.8351, balanced=0.7829
[2026-06-23 19:40:03] INFO segtask_v1.trainer.trainer: Epoch 127/400 | LR=7.83e-04 | loss=0.3026 | val_dice=0.7744 | best=0.8255 (ep47) | 02:43:20 | L_main=0.0696 L_aux_1=0.0732(w=0.5) L_aux_2=0.0836(w=0.5)
[2026-06-23 19:40:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 127): 13421.7 MiB
[2026-06-23 19:41:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2538, pooled_mean_dice=0.7847, per_class=['0.7847'], iou=0.6456, recall=0.9897, precision=0.6500, vol_sim=0.7928, mcc=0.7955, min_class_dice=0.7847, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9019, per_class_sd=['0.9019'], combined(w=0.50)=0.8433, balanced=0.7927
[2026-06-23 19:41:18] INFO segtask_v1.trainer.trainer: Epoch 128/400 | LR=7.79e-04 | loss=0.3120 | val_dice=0.7847 | best=0.8255 (ep47) | 02:44:35 | L_main=0.0730 L_aux_1=0.0766(w=0.5) L_aux_2=0.0875(w=0.5)
[2026-06-23 19:41:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 128): 13421.7 MiB
[2026-06-23 19:42:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2778, pooled_mean_dice=0.7595, per_class=['0.7595'], iou=0.6122, recall=0.9888, precision=0.6165, vol_sim=0.7681, mcc=0.7759, min_class_dice=0.7595, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8938, per_class_sd=['0.8938'], combined(w=0.50)=0.8267, balanced=0.7700
[2026-06-23 19:42:33] INFO segtask_v1.trainer.trainer: Epoch 129/400 | LR=7.76e-04 | loss=0.3138 | val_dice=0.7595 | best=0.8255 (ep47) | 02:45:50 | L_main=0.0739 L_aux_1=0.0776(w=0.5) L_aux_2=0.0886(w=0.5)
[2026-06-23 19:42:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 129): 13421.7 MiB
[2026-06-23 19:43:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2699, pooled_mean_dice=0.7761, per_class=['0.7761'], iou=0.6341, recall=0.9880, precision=0.6390, vol_sim=0.7855, mcc=0.7894, min_class_dice=0.7761, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9070, per_class_sd=['0.9070'], combined(w=0.50)=0.8416, balanced=0.7865
[2026-06-23 19:43:48] INFO segtask_v1.trainer.trainer: Epoch 130/400 | LR=7.73e-04 | loss=0.3173 | val_dice=0.7761 | best=0.8255 (ep47) | 02:47:04 | L_main=0.0755 L_aux_1=0.0793(w=0.5) L_aux_2=0.0910(w=0.5)
[2026-06-23 19:43:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 130): 13421.7 MiB
[2026-06-23 19:45:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2678, pooled_mean_dice=0.7776, per_class=['0.7776'], iou=0.6361, recall=0.9897, precision=0.6403, vol_sim=0.7856, mcc=0.7908, min_class_dice=0.7776, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9016, per_class_sd=['0.9016'], combined(w=0.50)=0.8396, balanced=0.7868
[2026-06-23 19:45:07] INFO segtask_v1.trainer.trainer: Epoch 131/400 | LR=7.69e-04 | loss=0.2969 | val_dice=0.7776 | best=0.8255 (ep47) | 02:48:24 | L_main=0.0695 L_aux_1=0.0731(w=0.5) L_aux_2=0.0831(w=0.5)
[2026-06-23 19:45:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 131): 13421.7 MiB
[2026-06-23 19:46:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2783, pooled_mean_dice=0.7693, per_class=['0.7693'], iou=0.6251, recall=0.9880, precision=0.6298, vol_sim=0.7786, mcc=0.7843, min_class_dice=0.7693, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9088, per_class_sd=['0.9088'], combined(w=0.50)=0.8390, balanced=0.7811
[2026-06-23 19:46:19] INFO segtask_v1.trainer.trainer: Epoch 132/400 | LR=7.66e-04 | loss=0.3019 | val_dice=0.7693 | best=0.8255 (ep47) | 02:49:36 | L_main=0.0711 L_aux_1=0.0750(w=0.5) L_aux_2=0.0858(w=0.5)
[2026-06-23 19:46:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 132): 13421.7 MiB
[2026-06-23 19:47:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2318, pooled_mean_dice=0.8039, per_class=['0.8039'], iou=0.6721, recall=0.9870, precision=0.6781, vol_sim=0.8145, mcc=0.8124, min_class_dice=0.8039, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9084, per_class_sd=['0.9084'], combined(w=0.50)=0.8562, balanced=0.8104
[2026-06-23 19:47:35] INFO segtask_v1.trainer.trainer: Epoch 133/400 | LR=7.63e-04 | loss=0.3039 | val_dice=0.8039 | best=0.8255 (ep47) | 02:50:51 | L_main=0.0711 L_aux_1=0.0746(w=0.5) L_aux_2=0.0855(w=0.5)
[2026-06-23 19:47:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 133): 13421.7 MiB
[2026-06-23 19:48:54] INFO segtask_v1.trainer.validation:   Val: loss=0.2724, pooled_mean_dice=0.7631, per_class=['0.7631'], iou=0.6169, recall=0.9891, precision=0.6211, vol_sim=0.7715, mcc=0.7777, min_class_dice=0.7631, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.8948, per_class_sd=['0.8948'], combined(w=0.50)=0.8289, balanced=0.7731
[2026-06-23 19:48:54] INFO segtask_v1.trainer.trainer: Epoch 134/400 | LR=7.59e-04 | loss=0.3127 | val_dice=0.7631 | best=0.8255 (ep47) | 02:52:11 | L_main=0.0740 L_aux_1=0.0777(w=0.5) L_aux_2=0.0893(w=0.5)
[2026-06-23 19:48:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 134): 13421.7 MiB
[2026-06-23 19:50:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2773, pooled_mean_dice=0.7686, per_class=['0.7686'], iou=0.6242, recall=0.9877, precision=0.6291, vol_sim=0.7782, mcc=0.7835, min_class_dice=0.7686, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9019, per_class_sd=['0.9019'], combined(w=0.50)=0.8352, balanced=0.7792
[2026-06-23 19:50:10] INFO segtask_v1.trainer.trainer: Epoch 135/400 | LR=7.56e-04 | loss=0.3145 | val_dice=0.7686 | best=0.8255 (ep47) | 02:53:27 | L_main=0.0749 L_aux_1=0.0793(w=0.5) L_aux_2=0.0911(w=0.5)
[2026-06-23 19:50:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 135): 13421.7 MiB
[2026-06-23 19:51:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2566, pooled_mean_dice=0.7865, per_class=['0.7865'], iou=0.6481, recall=0.9882, precision=0.6532, vol_sim=0.7959, mcc=0.7982, min_class_dice=0.7865, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9043, per_class_sd=['0.9043'], combined(w=0.50)=0.8454, balanced=0.7949
[2026-06-23 19:51:21] INFO segtask_v1.trainer.trainer: Epoch 136/400 | LR=7.53e-04 | loss=0.2956 | val_dice=0.7865 | best=0.8255 (ep47) | 02:54:38 | L_main=0.0665 L_aux_1=0.0702(w=0.5) L_aux_2=0.0806(w=0.5)
[2026-06-23 19:51:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 136): 13421.7 MiB
[2026-06-23 19:52:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2487, pooled_mean_dice=0.7877, per_class=['0.7877'], iou=0.6498, recall=0.9875, precision=0.6551, vol_sim=0.7977, mcc=0.7998, min_class_dice=0.7877, coverage=[66]/88 samples, pooled_mean_surface_dice@2px=0.9123, per_class_sd=['0.9123'], combined(w=0.50)=0.8500, balanced=0.7975
[2026-06-23 19:52:37] INFO segtask_v1.trainer.trainer: Epoch 137/400 | LR=7.49e-04 | loss=0.2971 | val_dice=0.7877 | best=0.8255 (ep47) | 02:55:54 | L_main=0.0702 L_aux_1=0.0732(w=0.5) L_aux_2=0.0831(w=0.5)
[2026-06-23 19:52:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 137): 13421.7 MiB
[2026-06-23 19:53:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2474, pooled_mean_dice=0.7903, per_class=['0.7903'], iou=0.6533, recall=0.9881, precision=0.6585, vol_sim=0.7999, mcc=0.8010, min_class_dice=0.7903, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9041, per_class_sd=['0.9041'], combined(w=0.50)=0.8472, balanced=0.7980
[2026-06-23 19:53:45] INFO segtask_v1.trainer.trainer: Epoch 138/400 | LR=7.46e-04 | loss=0.3027 | val_dice=0.7903 | best=0.8255 (ep47) | 02:57:01 | L_main=0.0691 L_aux_1=0.0737(w=0.5) L_aux_2=0.0843(w=0.5)
[2026-06-23 19:53:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 138): 13421.7 MiB
[2026-06-23 19:55:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2438, pooled_mean_dice=0.7907, per_class=['0.7907'], iou=0.6539, recall=0.9869, precision=0.6596, vol_sim=0.8012, mcc=0.8011, min_class_dice=0.7907, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9110, per_class_sd=['0.9110'], combined(w=0.50)=0.8508, balanced=0.7997
[2026-06-23 19:55:02] INFO segtask_v1.trainer.trainer: Epoch 139/400 | LR=7.42e-04 | loss=0.2974 | val_dice=0.7907 | best=0.8255 (ep47) | 02:58:18 | L_main=0.0676 L_aux_1=0.0714(w=0.5) L_aux_2=0.0819(w=0.5)
[2026-06-23 19:55:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 139): 13421.7 MiB
[2026-06-23 19:56:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2438, pooled_mean_dice=0.7885, per_class=['0.7885'], iou=0.6509, recall=0.9860, precision=0.6569, vol_sim=0.7997, mcc=0.7989, min_class_dice=0.7885, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9099, per_class_sd=['0.9099'], combined(w=0.50)=0.8492, balanced=0.7975
[2026-06-23 19:56:21] INFO segtask_v1.trainer.trainer: Epoch 140/400 | LR=7.39e-04 | loss=0.3021 | val_dice=0.7885 | best=0.8255 (ep47) | 02:59:38 | L_main=0.0703 L_aux_1=0.0736(w=0.5) L_aux_2=0.0835(w=0.5)
[2026-06-23 19:56:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 140): 13421.7 MiB
[2026-06-23 19:57:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2829, pooled_mean_dice=0.7562, per_class=['0.7562'], iou=0.6080, recall=0.9822, precision=0.6148, vol_sim=0.7699, mcc=0.7723, min_class_dice=0.7562, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8954, per_class_sd=['0.8954'], combined(w=0.50)=0.8258, balanced=0.7674
[2026-06-23 19:57:33] INFO segtask_v1.trainer.trainer: Epoch 141/400 | LR=7.35e-04 | loss=0.3131 | val_dice=0.7562 | best=0.8255 (ep47) | 03:00:50 | L_main=0.0745 L_aux_1=0.0787(w=0.5) L_aux_2=0.0898(w=0.5)
[2026-06-23 19:57:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 141): 13421.7 MiB
[2026-06-23 19:58:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2896, pooled_mean_dice=0.7666, per_class=['0.7666'], iou=0.6216, recall=0.9862, precision=0.6270, vol_sim=0.7774, mcc=0.7817, min_class_dice=0.7666, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9058, per_class_sd=['0.9058'], combined(w=0.50)=0.8362, balanced=0.7783
[2026-06-23 19:58:41] INFO segtask_v1.trainer.trainer: Epoch 142/400 | LR=7.32e-04 | loss=0.3026 | val_dice=0.7666 | best=0.8255 (ep47) | 03:01:58 | L_main=0.0699 L_aux_1=0.0736(w=0.5) L_aux_2=0.0848(w=0.5)
[2026-06-23 19:58:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 142): 13421.7 MiB
[2026-06-23 19:59:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2423, pooled_mean_dice=0.7958, per_class=['0.7958'], iou=0.6609, recall=0.9889, precision=0.6659, vol_sim=0.8048, mcc=0.8058, min_class_dice=0.7958, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9144, per_class_sd=['0.9144'], combined(w=0.50)=0.8551, balanced=0.8047
[2026-06-23 19:59:49] INFO segtask_v1.trainer.trainer: Epoch 143/400 | LR=7.28e-04 | loss=0.3011 | val_dice=0.7958 | best=0.8255 (ep47) | 03:03:06 | L_main=0.0681 L_aux_1=0.0724(w=0.5) L_aux_2=0.0835(w=0.5)
[2026-06-23 19:59:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 143): 13421.7 MiB
[2026-06-23 20:00:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2691, pooled_mean_dice=0.7859, per_class=['0.7859'], iou=0.6473, recall=0.9887, precision=0.6521, vol_sim=0.7949, mcc=0.7979, min_class_dice=0.7859, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9033, per_class_sd=['0.9033'], combined(w=0.50)=0.8446, balanced=0.7942
[2026-06-23 20:00:57] INFO segtask_v1.trainer.trainer: Epoch 144/400 | LR=7.25e-04 | loss=0.2937 | val_dice=0.7859 | best=0.8255 (ep47) | 03:04:14 | L_main=0.0663 L_aux_1=0.0701(w=0.5) L_aux_2=0.0804(w=0.5)
[2026-06-23 20:00:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 144): 13421.7 MiB
[2026-06-23 20:02:05] INFO segtask_v1.trainer.validation:   Val: loss=0.2553, pooled_mean_dice=0.7829, per_class=['0.7829'], iou=0.6433, recall=0.9858, precision=0.6493, vol_sim=0.7942, mcc=0.7945, min_class_dice=0.7829, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9076, per_class_sd=['0.9076'], combined(w=0.50)=0.8453, balanced=0.7924
[2026-06-23 20:02:05] INFO segtask_v1.trainer.trainer: Epoch 145/400 | LR=7.21e-04 | loss=0.2959 | val_dice=0.7829 | best=0.8255 (ep47) | 03:05:22 | L_main=0.0663 L_aux_1=0.0696(w=0.5) L_aux_2=0.0794(w=0.5)
[2026-06-23 20:02:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 145): 13421.7 MiB
[2026-06-23 20:03:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2689, pooled_mean_dice=0.7745, per_class=['0.7745'], iou=0.6319, recall=0.9899, precision=0.6360, vol_sim=0.7823, mcc=0.7880, min_class_dice=0.7745, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9100, per_class_sd=['0.9100'], combined(w=0.50)=0.8422, balanced=0.7857
[2026-06-23 20:03:22] INFO segtask_v1.trainer.trainer: Epoch 146/400 | LR=7.17e-04 | loss=0.3155 | val_dice=0.7745 | best=0.8255 (ep47) | 03:06:39 | L_main=0.0760 L_aux_1=0.0794(w=0.5) L_aux_2=0.0897(w=0.5)
[2026-06-23 20:03:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 146): 13421.7 MiB
[2026-06-23 20:04:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2881, pooled_mean_dice=0.7677, per_class=['0.7677'], iou=0.6230, recall=0.9893, precision=0.6272, vol_sim=0.7760, mcc=0.7828, min_class_dice=0.7677, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9034, per_class_sd=['0.9034'], combined(w=0.50)=0.8355, balanced=0.7787
[2026-06-23 20:04:30] INFO segtask_v1.trainer.trainer: Epoch 147/400 | LR=7.14e-04 | loss=0.3084 | val_dice=0.7677 | best=0.8255 (ep47) | 03:07:47 | L_main=0.0722 L_aux_1=0.0763(w=0.5) L_aux_2=0.0861(w=0.5)
[2026-06-23 20:04:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 147): 13421.7 MiB
[2026-06-23 20:05:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2494, pooled_mean_dice=0.7906, per_class=['0.7906'], iou=0.6537, recall=0.9897, precision=0.6582, vol_sim=0.7988, mcc=0.8007, min_class_dice=0.7906, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9115, per_class_sd=['0.9115'], combined(w=0.50)=0.8510, balanced=0.7996
[2026-06-23 20:05:37] INFO segtask_v1.trainer.trainer: Epoch 148/400 | LR=7.10e-04 | loss=0.3111 | val_dice=0.7906 | best=0.8255 (ep47) | 03:08:54 | L_main=0.0736 L_aux_1=0.0779(w=0.5) L_aux_2=0.0886(w=0.5)
[2026-06-23 20:05:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 148): 13421.7 MiB
[2026-06-23 20:06:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2764, pooled_mean_dice=0.7658, per_class=['0.7658'], iou=0.6205, recall=0.9888, precision=0.6249, vol_sim=0.7745, mcc=0.7810, min_class_dice=0.7658, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8967, per_class_sd=['0.8967'], combined(w=0.50)=0.8313, balanced=0.7759
[2026-06-23 20:06:44] INFO segtask_v1.trainer.trainer: Epoch 149/400 | LR=7.07e-04 | loss=0.3149 | val_dice=0.7658 | best=0.8255 (ep47) | 03:10:01 | L_main=0.0744 L_aux_1=0.0782(w=0.5) L_aux_2=0.0895(w=0.5)
[2026-06-23 20:06:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 149): 13421.7 MiB
[2026-06-23 20:07:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2634, pooled_mean_dice=0.7768, per_class=['0.7768'], iou=0.6351, recall=0.9884, precision=0.6399, vol_sim=0.7859, mcc=0.7905, min_class_dice=0.7768, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9013, per_class_sd=['0.9013'], combined(w=0.50)=0.8391, balanced=0.7861
[2026-06-23 20:07:52] INFO segtask_v1.trainer.trainer: Epoch 150/400 | LR=7.03e-04 | loss=0.3076 | val_dice=0.7768 | best=0.8255 (ep47) | 03:11:09 | L_main=0.0721 L_aux_1=0.0759(w=0.5) L_aux_2=0.0866(w=0.5)
[2026-06-23 20:07:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 150): 13421.7 MiB
[2026-06-23 20:09:11] INFO segtask_v1.trainer.validation:   Val: loss=0.2652, pooled_mean_dice=0.7789, per_class=['0.7789'], iou=0.6378, recall=0.9905, precision=0.6418, vol_sim=0.7863, mcc=0.7925, min_class_dice=0.7789, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9041, per_class_sd=['0.9041'], combined(w=0.50)=0.8415, balanced=0.7884
[2026-06-23 20:09:11] INFO segtask_v1.trainer.trainer: Epoch 151/400 | LR=6.99e-04 | loss=0.2936 | val_dice=0.7789 | best=0.8255 (ep47) | 03:12:28 | L_main=0.0664 L_aux_1=0.0700(w=0.5) L_aux_2=0.0799(w=0.5)
[2026-06-23 20:09:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 151): 13421.7 MiB
[2026-06-23 20:10:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2584, pooled_mean_dice=0.7913, per_class=['0.7913'], iou=0.6547, recall=0.9902, precision=0.6590, vol_sim=0.7992, mcc=0.8021, min_class_dice=0.7913, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9103, per_class_sd=['0.9103'], combined(w=0.50)=0.8508, balanced=0.8001
[2026-06-23 20:10:22] INFO segtask_v1.trainer.trainer: Epoch 152/400 | LR=6.96e-04 | loss=0.2950 | val_dice=0.7913 | best=0.8255 (ep47) | 03:13:39 | L_main=0.0677 L_aux_1=0.0716(w=0.5) L_aux_2=0.0817(w=0.5)
[2026-06-23 20:10:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 152): 13421.7 MiB
[2026-06-23 20:11:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2698, pooled_mean_dice=0.7697, per_class=['0.7697'], iou=0.6256, recall=0.9883, precision=0.6302, vol_sim=0.7788, mcc=0.7836, min_class_dice=0.7697, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9030, per_class_sd=['0.9030'], combined(w=0.50)=0.8364, balanced=0.7803
[2026-06-23 20:11:35] INFO segtask_v1.trainer.trainer: Epoch 153/400 | LR=6.92e-04 | loss=0.2929 | val_dice=0.7697 | best=0.8255 (ep47) | 03:14:52 | L_main=0.0668 L_aux_1=0.0701(w=0.5) L_aux_2=0.0795(w=0.5)
[2026-06-23 20:11:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 153): 13421.7 MiB
[2026-06-23 20:12:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2604, pooled_mean_dice=0.7783, per_class=['0.7783'], iou=0.6371, recall=0.9873, precision=0.6423, vol_sim=0.7883, mcc=0.7913, min_class_dice=0.7783, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8986, per_class_sd=['0.8986'], combined(w=0.50)=0.8385, balanced=0.7868
[2026-06-23 20:12:43] INFO segtask_v1.trainer.trainer: Epoch 154/400 | LR=6.88e-04 | loss=0.3011 | val_dice=0.7783 | best=0.8255 (ep47) | 03:15:59 | L_main=0.0684 L_aux_1=0.0720(w=0.5) L_aux_2=0.0828(w=0.5)
[2026-06-23 20:12:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 154): 13421.7 MiB
[2026-06-23 20:13:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2393, pooled_mean_dice=0.7969, per_class=['0.7969'], iou=0.6624, recall=0.9883, precision=0.6677, vol_sim=0.8064, mcc=0.8072, min_class_dice=0.7969, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9180, per_class_sd=['0.9180'], combined(w=0.50)=0.8575, balanced=0.8064
[2026-06-23 20:13:51] INFO segtask_v1.trainer.trainer: Epoch 155/400 | LR=6.85e-04 | loss=0.2976 | val_dice=0.7969 | best=0.8255 (ep47) | 03:17:08 | L_main=0.0686 L_aux_1=0.0720(w=0.5) L_aux_2=0.0825(w=0.5)
[2026-06-23 20:13:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 155): 13421.7 MiB
[2026-06-23 20:14:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2538, pooled_mean_dice=0.7827, per_class=['0.7827'], iou=0.6430, recall=0.9913, precision=0.6467, vol_sim=0.7896, mcc=0.7954, min_class_dice=0.7827, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9106, per_class_sd=['0.9106'], combined(w=0.50)=0.8467, balanced=0.7929
[2026-06-23 20:14:59] INFO segtask_v1.trainer.trainer: Epoch 156/400 | LR=6.81e-04 | loss=0.2915 | val_dice=0.7827 | best=0.8255 (ep47) | 03:18:16 | L_main=0.0649 L_aux_1=0.0684(w=0.5) L_aux_2=0.0786(w=0.5)
[2026-06-23 20:14:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 156): 13421.7 MiB
[2026-06-23 20:16:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2403, pooled_mean_dice=0.7981, per_class=['0.7981'], iou=0.6640, recall=0.9892, precision=0.6688, vol_sim=0.8067, mcc=0.8082, min_class_dice=0.7981, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9193, per_class_sd=['0.9193'], combined(w=0.50)=0.8587, balanced=0.8076
[2026-06-23 20:16:14] INFO segtask_v1.trainer.trainer: Epoch 157/400 | LR=6.77e-04 | loss=0.2880 | val_dice=0.7981 | best=0.8255 (ep47) | 03:19:31 | L_main=0.0662 L_aux_1=0.0699(w=0.5) L_aux_2=0.0803(w=0.5)
[2026-06-23 20:16:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 157): 13421.7 MiB
[2026-06-23 20:17:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2587, pooled_mean_dice=0.7729, per_class=['0.7729'], iou=0.6299, recall=0.9898, precision=0.6340, vol_sim=0.7809, mcc=0.7871, min_class_dice=0.7729, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8986, per_class_sd=['0.8986'], combined(w=0.50)=0.8358, balanced=0.7823
[2026-06-23 20:17:22] INFO segtask_v1.trainer.trainer: Epoch 158/400 | LR=6.74e-04 | loss=0.2903 | val_dice=0.7729 | best=0.8255 (ep47) | 03:20:39 | L_main=0.0660 L_aux_1=0.0696(w=0.5) L_aux_2=0.0798(w=0.5)
[2026-06-23 20:17:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 158): 13421.7 MiB
[2026-06-23 20:18:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2479, pooled_mean_dice=0.7881, per_class=['0.7881'], iou=0.6503, recall=0.9862, precision=0.6563, vol_sim=0.7991, mcc=0.7995, min_class_dice=0.7881, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9167, per_class_sd=['0.9167'], combined(w=0.50)=0.8524, balanced=0.7986
[2026-06-23 20:18:35] INFO segtask_v1.trainer.trainer: Epoch 159/400 | LR=6.70e-04 | loss=0.2861 | val_dice=0.7881 | best=0.8255 (ep47) | 03:21:52 | L_main=0.0654 L_aux_1=0.0689(w=0.5) L_aux_2=0.0781(w=0.5)
[2026-06-23 20:18:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 159): 13421.7 MiB
[2026-06-23 20:19:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2492, pooled_mean_dice=0.7969, per_class=['0.7969'], iou=0.6623, recall=0.9875, precision=0.6679, vol_sim=0.8069, mcc=0.8068, min_class_dice=0.7969, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9180, per_class_sd=['0.9180'], combined(w=0.50)=0.8574, balanced=0.8063
[2026-06-23 20:19:46] INFO segtask_v1.trainer.trainer: Epoch 160/400 | LR=6.66e-04 | loss=0.2931 | val_dice=0.7969 | best=0.8255 (ep47) | 03:23:02 | L_main=0.0664 L_aux_1=0.0691(w=0.5) L_aux_2=0.0783(w=0.5)
[2026-06-23 20:19:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 160): 13421.7 MiB
[2026-06-23 20:21:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2411, pooled_mean_dice=0.7923, per_class=['0.7923'], iou=0.6560, recall=0.9887, precision=0.6609, vol_sim=0.8013, mcc=0.8036, min_class_dice=0.7923, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9119, per_class_sd=['0.9119'], combined(w=0.50)=0.8521, balanced=0.8013
[2026-06-23 20:21:01] INFO segtask_v1.trainer.trainer: Epoch 161/400 | LR=6.62e-04 | loss=0.2962 | val_dice=0.7923 | best=0.8255 (ep47) | 03:24:17 | L_main=0.0673 L_aux_1=0.0706(w=0.5) L_aux_2=0.0800(w=0.5)
[2026-06-23 20:21:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 161): 13421.7 MiB
[2026-06-23 20:22:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2534, pooled_mean_dice=0.7916, per_class=['0.7916'], iou=0.6551, recall=0.9878, precision=0.6604, vol_sim=0.8014, mcc=0.8023, min_class_dice=0.7916, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9140, per_class_sd=['0.9140'], combined(w=0.50)=0.8528, balanced=0.8010
[2026-06-23 20:22:20] INFO segtask_v1.trainer.trainer: Epoch 162/400 | LR=6.59e-04 | loss=0.2986 | val_dice=0.7916 | best=0.8255 (ep47) | 03:25:37 | L_main=0.0686 L_aux_1=0.0725(w=0.5) L_aux_2=0.0824(w=0.5)
[2026-06-23 20:22:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 162): 13421.7 MiB
[2026-06-23 20:23:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2526, pooled_mean_dice=0.7807, per_class=['0.7807'], iou=0.6403, recall=0.9880, precision=0.6453, vol_sim=0.7901, mcc=0.7933, min_class_dice=0.7807, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9047, per_class_sd=['0.9047'], combined(w=0.50)=0.8427, balanced=0.7900
[2026-06-23 20:23:36] INFO segtask_v1.trainer.trainer: Epoch 163/400 | LR=6.55e-04 | loss=0.2989 | val_dice=0.7807 | best=0.8255 (ep47) | 03:26:53 | L_main=0.0701 L_aux_1=0.0736(w=0.5) L_aux_2=0.0837(w=0.5)
[2026-06-23 20:23:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 163): 13421.7 MiB
[2026-06-23 20:24:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2563, pooled_mean_dice=0.7819, per_class=['0.7819'], iou=0.6419, recall=0.9883, precision=0.6468, vol_sim=0.7912, mcc=0.7934, min_class_dice=0.7819, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9065, per_class_sd=['0.9065'], combined(w=0.50)=0.8442, balanced=0.7913
[2026-06-23 20:24:55] INFO segtask_v1.trainer.trainer: Epoch 164/400 | LR=6.51e-04 | loss=0.2971 | val_dice=0.7819 | best=0.8255 (ep47) | 03:28:12 | L_main=0.0686 L_aux_1=0.0722(w=0.5) L_aux_2=0.0830(w=0.5)
[2026-06-23 20:24:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 164): 13421.7 MiB
[2026-06-23 20:26:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2286, pooled_mean_dice=0.8006, per_class=['0.8006'], iou=0.6676, recall=0.9888, precision=0.6727, vol_sim=0.8097, mcc=0.8102, min_class_dice=0.8006, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9187, per_class_sd=['0.9187'], combined(w=0.50)=0.8597, balanced=0.8097
[2026-06-23 20:26:06] INFO segtask_v1.trainer.trainer: Epoch 165/400 | LR=6.47e-04 | loss=0.2828 | val_dice=0.8006 | best=0.8255 (ep47) | 03:29:23 | L_main=0.0644 L_aux_1=0.0679(w=0.5) L_aux_2=0.0774(w=0.5)
[2026-06-23 20:26:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 165): 13421.7 MiB
[2026-06-23 20:27:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2569, pooled_mean_dice=0.7784, per_class=['0.7784'], iou=0.6372, recall=0.9868, precision=0.6427, vol_sim=0.7888, mcc=0.7907, min_class_dice=0.7784, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9089, per_class_sd=['0.9089'], combined(w=0.50)=0.8437, balanced=0.7888
[2026-06-23 20:27:27] INFO segtask_v1.trainer.trainer: Epoch 166/400 | LR=6.43e-04 | loss=0.2804 | val_dice=0.7784 | best=0.8255 (ep47) | 03:30:43 | L_main=0.0624 L_aux_1=0.0656(w=0.5) L_aux_2=0.0749(w=0.5)
[2026-06-23 20:27:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 166): 13421.7 MiB
[2026-06-23 20:28:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2747, pooled_mean_dice=0.7810, per_class=['0.7810'], iou=0.6406, recall=0.9887, precision=0.6454, vol_sim=0.7899, mcc=0.7938, min_class_dice=0.7810, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9159, per_class_sd=['0.9159'], combined(w=0.50)=0.8484, balanced=0.7924
[2026-06-23 20:28:41] INFO segtask_v1.trainer.trainer: Epoch 167/400 | LR=6.40e-04 | loss=0.2880 | val_dice=0.7810 | best=0.8255 (ep47) | 03:31:58 | L_main=0.0650 L_aux_1=0.0684(w=0.5) L_aux_2=0.0775(w=0.5)
[2026-06-23 20:28:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 167): 13421.7 MiB
[2026-06-23 20:30:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2694, pooled_mean_dice=0.7820, per_class=['0.7820'], iou=0.6420, recall=0.9884, precision=0.6469, vol_sim=0.7911, mcc=0.7943, min_class_dice=0.7820, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9096, per_class_sd=['0.9096'], combined(w=0.50)=0.8458, balanced=0.7920
[2026-06-23 20:30:02] INFO segtask_v1.trainer.trainer: Epoch 168/400 | LR=6.36e-04 | loss=0.2900 | val_dice=0.7820 | best=0.8255 (ep47) | 03:33:18 | L_main=0.0663 L_aux_1=0.0694(w=0.5) L_aux_2=0.0787(w=0.5)
[2026-06-23 20:30:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 168): 13421.7 MiB
[2026-06-23 20:31:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2485, pooled_mean_dice=0.7859, per_class=['0.7859'], iou=0.6474, recall=0.9900, precision=0.6516, vol_sim=0.7939, mcc=0.7980, min_class_dice=0.7859, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9098, per_class_sd=['0.9098'], combined(w=0.50)=0.8478, balanced=0.7954
[2026-06-23 20:31:17] INFO segtask_v1.trainer.trainer: Epoch 169/400 | LR=6.32e-04 | loss=0.3084 | val_dice=0.7859 | best=0.8255 (ep47) | 03:34:34 | L_main=0.0722 L_aux_1=0.0757(w=0.5) L_aux_2=0.0868(w=0.5)
[2026-06-23 20:31:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 169): 13421.7 MiB
[2026-06-23 20:32:34] INFO segtask_v1.trainer.validation:   Val: loss=0.2575, pooled_mean_dice=0.7819, per_class=['0.7819'], iou=0.6420, recall=0.9878, precision=0.6471, vol_sim=0.7916, mcc=0.7932, min_class_dice=0.7819, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9101, per_class_sd=['0.9101'], combined(w=0.50)=0.8460, balanced=0.7920
[2026-06-23 20:32:34] INFO segtask_v1.trainer.trainer: Epoch 170/400 | LR=6.28e-04 | loss=0.3082 | val_dice=0.7819 | best=0.8255 (ep47) | 03:35:51 | L_main=0.0719 L_aux_1=0.0754(w=0.5) L_aux_2=0.0859(w=0.5)
[2026-06-23 20:32:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 170): 13421.7 MiB
[2026-06-23 20:33:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2506, pooled_mean_dice=0.7909, per_class=['0.7909'], iou=0.6541, recall=0.9883, precision=0.6592, vol_sim=0.8002, mcc=0.8019, min_class_dice=0.7909, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9088, per_class_sd=['0.9088'], combined(w=0.50)=0.8499, balanced=0.7995
[2026-06-23 20:33:47] INFO segtask_v1.trainer.trainer: Epoch 171/400 | LR=6.24e-04 | loss=0.2848 | val_dice=0.7909 | best=0.8255 (ep47) | 03:37:04 | L_main=0.0642 L_aux_1=0.0672(w=0.5) L_aux_2=0.0767(w=0.5)
[2026-06-23 20:33:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 171): 13421.7 MiB
[2026-06-23 20:35:05] INFO segtask_v1.trainer.validation:   Val: loss=0.2825, pooled_mean_dice=0.7680, per_class=['0.7680'], iou=0.6233, recall=0.9882, precision=0.6280, vol_sim=0.7771, mcc=0.7815, min_class_dice=0.7680, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.8979, per_class_sd=['0.8979'], combined(w=0.50)=0.8329, balanced=0.7778
[2026-06-23 20:35:05] INFO segtask_v1.trainer.trainer: Epoch 172/400 | LR=6.20e-04 | loss=0.2943 | val_dice=0.7680 | best=0.8255 (ep47) | 03:38:22 | L_main=0.0675 L_aux_1=0.0710(w=0.5) L_aux_2=0.0810(w=0.5)
[2026-06-23 20:35:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 172): 13421.7 MiB
[2026-06-23 20:36:25] INFO segtask_v1.trainer.validation:   Val: loss=0.2626, pooled_mean_dice=0.7906, per_class=['0.7906'], iou=0.6538, recall=0.9909, precision=0.6577, vol_sim=0.7979, mcc=0.8017, min_class_dice=0.7906, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9075, per_class_sd=['0.9075'], combined(w=0.50)=0.8491, balanced=0.7990
[2026-06-23 20:36:25] INFO segtask_v1.trainer.trainer: Epoch 173/400 | LR=6.17e-04 | loss=0.2867 | val_dice=0.7906 | best=0.8255 (ep47) | 03:39:42 | L_main=0.0661 L_aux_1=0.0693(w=0.5) L_aux_2=0.0788(w=0.5)
[2026-06-23 20:36:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 173): 13421.7 MiB
[2026-06-23 20:37:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2332, pooled_mean_dice=0.7989, per_class=['0.7989'], iou=0.6651, recall=0.9883, precision=0.6704, vol_sim=0.8084, mcc=0.8074, min_class_dice=0.7989, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9093, per_class_sd=['0.9093'], combined(w=0.50)=0.8541, balanced=0.8062
[2026-06-23 20:37:47] INFO segtask_v1.trainer.trainer: Epoch 174/400 | LR=6.13e-04 | loss=0.2888 | val_dice=0.7989 | best=0.8255 (ep47) | 03:41:04 | L_main=0.0649 L_aux_1=0.0686(w=0.5) L_aux_2=0.0781(w=0.5)
[2026-06-23 20:37:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 174): 13421.7 MiB
[2026-06-23 20:39:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2514, pooled_mean_dice=0.7981, per_class=['0.7981'], iou=0.6640, recall=0.9881, precision=0.6693, vol_sim=0.8077, mcc=0.8079, min_class_dice=0.7981, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9197, per_class_sd=['0.9197'], combined(w=0.50)=0.8589, balanced=0.8076
[2026-06-23 20:39:02] INFO segtask_v1.trainer.trainer: Epoch 175/400 | LR=6.09e-04 | loss=0.2857 | val_dice=0.7981 | best=0.8255 (ep47) | 03:42:19 | L_main=0.0640 L_aux_1=0.0677(w=0.5) L_aux_2=0.0767(w=0.5)
[2026-06-23 20:39:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 175): 13421.7 MiB
[2026-06-23 20:40:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2505, pooled_mean_dice=0.7845, per_class=['0.7845'], iou=0.6454, recall=0.9875, precision=0.6507, vol_sim=0.7944, mcc=0.7957, min_class_dice=0.7845, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9132, per_class_sd=['0.9132'], combined(w=0.50)=0.8488, balanced=0.7947
[2026-06-23 20:40:16] INFO segtask_v1.trainer.trainer: Epoch 176/400 | LR=6.05e-04 | loss=0.2851 | val_dice=0.7845 | best=0.8255 (ep47) | 03:43:33 | L_main=0.0643 L_aux_1=0.0679(w=0.5) L_aux_2=0.0774(w=0.5)
[2026-06-23 20:40:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 176): 13421.7 MiB
[2026-06-23 20:41:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2426, pooled_mean_dice=0.8027, per_class=['0.8027'], iou=0.6704, recall=0.9880, precision=0.6759, vol_sim=0.8125, mcc=0.8129, min_class_dice=0.8027, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9223, per_class_sd=['0.9223'], combined(w=0.50)=0.8625, balanced=0.8122
[2026-06-23 20:41:37] INFO segtask_v1.trainer.trainer: Epoch 177/400 | LR=6.01e-04 | loss=0.2787 | val_dice=0.8027 | best=0.8255 (ep47) | 03:44:53 | L_main=0.0603 L_aux_1=0.0633(w=0.5) L_aux_2=0.0718(w=0.5)
[2026-06-23 20:41:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 177): 13421.7 MiB
[2026-06-23 20:42:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2854, pooled_mean_dice=0.7594, per_class=['0.7594'], iou=0.6121, recall=0.9838, precision=0.6183, vol_sim=0.7719, mcc=0.7750, min_class_dice=0.7594, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9051, per_class_sd=['0.9051'], combined(w=0.50)=0.8323, balanced=0.7719
[2026-06-23 20:42:50] INFO segtask_v1.trainer.trainer: Epoch 178/400 | LR=5.97e-04 | loss=0.2876 | val_dice=0.7594 | best=0.8255 (ep47) | 03:46:07 | L_main=0.0645 L_aux_1=0.0679(w=0.5) L_aux_2=0.0774(w=0.5)
[2026-06-23 20:42:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 178): 13421.7 MiB
[2026-06-23 20:44:05] INFO segtask_v1.trainer.validation:   Val: loss=0.2258, pooled_mean_dice=0.8101, per_class=['0.8101'], iou=0.6808, recall=0.9882, precision=0.6863, vol_sim=0.8197, mcc=0.8182, min_class_dice=0.8101, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9237, per_class_sd=['0.9237'], combined(w=0.50)=0.8669, balanced=0.8187
[2026-06-23 20:44:05] INFO segtask_v1.trainer.trainer: Epoch 179/400 | LR=5.93e-04 | loss=0.2897 | val_dice=0.8101 | best=0.8255 (ep47) | 03:47:22 | L_main=0.0650 L_aux_1=0.0681(w=0.5) L_aux_2=0.0777(w=0.5)
[2026-06-23 20:44:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 179): 13421.7 MiB
[2026-06-23 20:45:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2636, pooled_mean_dice=0.7838, per_class=['0.7838'], iou=0.6445, recall=0.9876, precision=0.6497, vol_sim=0.7936, mcc=0.7959, min_class_dice=0.7838, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9110, per_class_sd=['0.9110'], combined(w=0.50)=0.8474, balanced=0.7939
[2026-06-23 20:45:18] INFO segtask_v1.trainer.trainer: Epoch 180/400 | LR=5.89e-04 | loss=0.2842 | val_dice=0.7838 | best=0.8255 (ep47) | 03:48:35 | L_main=0.0639 L_aux_1=0.0675(w=0.5) L_aux_2=0.0770(w=0.5)
[2026-06-23 20:45:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 180): 13421.7 MiB
[2026-06-23 20:46:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2775, pooled_mean_dice=0.7674, per_class=['0.7674'], iou=0.6226, recall=0.9886, precision=0.6272, vol_sim=0.7763, mcc=0.7827, min_class_dice=0.7674, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9060, per_class_sd=['0.9060'], combined(w=0.50)=0.8367, balanced=0.7790
[2026-06-23 20:46:37] INFO segtask_v1.trainer.trainer: Epoch 181/400 | LR=5.85e-04 | loss=0.2746 | val_dice=0.7674 | best=0.8255 (ep47) | 03:49:53 | L_main=0.0590 L_aux_1=0.0620(w=0.5) L_aux_2=0.0707(w=0.5)
[2026-06-23 20:46:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 181): 13421.7 MiB
[2026-06-23 20:47:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2414, pooled_mean_dice=0.7988, per_class=['0.7988'], iou=0.6651, recall=0.9887, precision=0.6702, vol_sim=0.8080, mcc=0.8084, min_class_dice=0.7988, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9168, per_class_sd=['0.9168'], combined(w=0.50)=0.8578, balanced=0.8077
[2026-06-23 20:47:49] INFO segtask_v1.trainer.trainer: Epoch 182/400 | LR=5.82e-04 | loss=0.2969 | val_dice=0.7988 | best=0.8255 (ep47) | 03:51:06 | L_main=0.0686 L_aux_1=0.0719(w=0.5) L_aux_2=0.0815(w=0.5)
[2026-06-23 20:47:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 182): 13421.7 MiB
[2026-06-23 20:49:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2568, pooled_mean_dice=0.7822, per_class=['0.7822'], iou=0.6423, recall=0.9868, precision=0.6479, vol_sim=0.7927, mcc=0.7943, min_class_dice=0.7822, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9081, per_class_sd=['0.9081'], combined(w=0.50)=0.8451, balanced=0.7919
[2026-06-23 20:49:02] INFO segtask_v1.trainer.trainer: Epoch 183/400 | LR=5.78e-04 | loss=0.3042 | val_dice=0.7822 | best=0.8255 (ep47) | 03:52:19 | L_main=0.0711 L_aux_1=0.0743(w=0.5) L_aux_2=0.0841(w=0.5)
[2026-06-23 20:49:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 183): 13421.7 MiB
[2026-06-23 20:50:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2402, pooled_mean_dice=0.7969, per_class=['0.7969'], iou=0.6624, recall=0.9853, precision=0.6691, vol_sim=0.8089, mcc=0.8066, min_class_dice=0.7969, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9174, per_class_sd=['0.9174'], combined(w=0.50)=0.8572, balanced=0.8062
[2026-06-23 20:50:15] INFO segtask_v1.trainer.trainer: Epoch 184/400 | LR=5.74e-04 | loss=0.2949 | val_dice=0.7969 | best=0.8255 (ep47) | 03:53:32 | L_main=0.0675 L_aux_1=0.0707(w=0.5) L_aux_2=0.0809(w=0.5)
[2026-06-23 20:50:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 184): 13421.7 MiB
[2026-06-23 20:51:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2535, pooled_mean_dice=0.7886, per_class=['0.7886'], iou=0.6510, recall=0.9881, precision=0.6562, vol_sim=0.7981, mcc=0.8006, min_class_dice=0.7886, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9151, per_class_sd=['0.9151'], combined(w=0.50)=0.8519, balanced=0.7988
[2026-06-23 20:51:33] INFO segtask_v1.trainer.trainer: Epoch 185/400 | LR=5.70e-04 | loss=0.2821 | val_dice=0.7886 | best=0.8255 (ep47) | 03:54:50 | L_main=0.0631 L_aux_1=0.0661(w=0.5) L_aux_2=0.0752(w=0.5)
[2026-06-23 20:51:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 185): 13421.7 MiB
[2026-06-23 20:52:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2532, pooled_mean_dice=0.7814, per_class=['0.7814'], iou=0.6412, recall=0.9864, precision=0.6469, vol_sim=0.7921, mcc=0.7935, min_class_dice=0.7814, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9162, per_class_sd=['0.9162'], combined(w=0.50)=0.8488, balanced=0.7927
[2026-06-23 20:52:51] INFO segtask_v1.trainer.trainer: Epoch 186/400 | LR=5.66e-04 | loss=0.2804 | val_dice=0.7814 | best=0.8255 (ep47) | 03:56:08 | L_main=0.0626 L_aux_1=0.0657(w=0.5) L_aux_2=0.0748(w=0.5)
[2026-06-23 20:52:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 186): 13421.7 MiB
[2026-06-23 20:54:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2408, pooled_mean_dice=0.7916, per_class=['0.7916'], iou=0.6551, recall=0.9896, precision=0.6596, vol_sim=0.7999, mcc=0.8022, min_class_dice=0.7916, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9147, per_class_sd=['0.9147'], combined(w=0.50)=0.8532, balanced=0.8012
[2026-06-23 20:54:00] INFO segtask_v1.trainer.trainer: Epoch 187/400 | LR=5.62e-04 | loss=0.2875 | val_dice=0.7916 | best=0.8255 (ep47) | 03:57:17 | L_main=0.0654 L_aux_1=0.0691(w=0.5) L_aux_2=0.0784(w=0.5)
[2026-06-23 20:54:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 187): 13421.7 MiB
[2026-06-23 20:55:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2584, pooled_mean_dice=0.7886, per_class=['0.7886'], iou=0.6509, recall=0.9888, precision=0.6558, vol_sim=0.7975, mcc=0.7992, min_class_dice=0.7886, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9136, per_class_sd=['0.9136'], combined(w=0.50)=0.8511, balanced=0.7983
[2026-06-23 20:55:12] INFO segtask_v1.trainer.trainer: Epoch 188/400 | LR=5.58e-04 | loss=0.2880 | val_dice=0.7886 | best=0.8255 (ep47) | 03:58:29 | L_main=0.0637 L_aux_1=0.0671(w=0.5) L_aux_2=0.0771(w=0.5)
[2026-06-23 20:55:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 188): 13421.7 MiB
[2026-06-23 20:56:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2438, pooled_mean_dice=0.7880, per_class=['0.7880'], iou=0.6501, recall=0.9905, precision=0.6542, vol_sim=0.7955, mcc=0.7997, min_class_dice=0.7880, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9183, per_class_sd=['0.9183'], combined(w=0.50)=0.8531, balanced=0.7988
[2026-06-23 20:56:20] INFO segtask_v1.trainer.trainer: Epoch 189/400 | LR=5.54e-04 | loss=0.2982 | val_dice=0.7880 | best=0.8255 (ep47) | 03:59:37 | L_main=0.0690 L_aux_1=0.0722(w=0.5) L_aux_2=0.0822(w=0.5)
[2026-06-23 20:56:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 189): 13421.7 MiB
[2026-06-23 20:57:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2406, pooled_mean_dice=0.7998, per_class=['0.7998'], iou=0.6664, recall=0.9897, precision=0.6710, vol_sim=0.8081, mcc=0.8089, min_class_dice=0.7998, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9208, per_class_sd=['0.9208'], combined(w=0.50)=0.8603, balanced=0.8093
[2026-06-23 20:57:30] INFO segtask_v1.trainer.trainer: Epoch 190/400 | LR=5.50e-04 | loss=0.2862 | val_dice=0.7998 | best=0.8255 (ep47) | 04:00:47 | L_main=0.0649 L_aux_1=0.0682(w=0.5) L_aux_2=0.0777(w=0.5)
[2026-06-23 20:57:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 190): 13421.7 MiB
[2026-06-23 20:58:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2727, pooled_mean_dice=0.7723, per_class=['0.7723'], iou=0.6291, recall=0.9875, precision=0.6341, vol_sim=0.7821, mcc=0.7865, min_class_dice=0.7723, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9067, per_class_sd=['0.9067'], combined(w=0.50)=0.8395, balanced=0.7833
[2026-06-23 20:58:44] INFO segtask_v1.trainer.trainer: Epoch 191/400 | LR=5.46e-04 | loss=0.2825 | val_dice=0.7723 | best=0.8255 (ep47) | 04:02:01 | L_main=0.0623 L_aux_1=0.0655(w=0.5) L_aux_2=0.0751(w=0.5)
[2026-06-23 20:58:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 191): 13421.7 MiB
[2026-06-23 20:59:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2553, pooled_mean_dice=0.7860, per_class=['0.7860'], iou=0.6474, recall=0.9898, precision=0.6518, vol_sim=0.7941, mcc=0.7982, min_class_dice=0.7860, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9181, per_class_sd=['0.9181'], combined(w=0.50)=0.8521, balanced=0.7971
[2026-06-23 20:59:55] INFO segtask_v1.trainer.trainer: Epoch 192/400 | LR=5.42e-04 | loss=0.2811 | val_dice=0.7860 | best=0.8255 (ep47) | 04:03:11 | L_main=0.0619 L_aux_1=0.0654(w=0.5) L_aux_2=0.0752(w=0.5)
[2026-06-23 20:59:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 192): 13421.7 MiB
[2026-06-23 21:01:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2393, pooled_mean_dice=0.7997, per_class=['0.7997'], iou=0.6663, recall=0.9889, precision=0.6713, vol_sim=0.8087, mcc=0.8094, min_class_dice=0.7997, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9203, per_class_sd=['0.9203'], combined(w=0.50)=0.8600, balanced=0.8092
[2026-06-23 21:01:01] INFO segtask_v1.trainer.trainer: Epoch 193/400 | LR=5.38e-04 | loss=0.2871 | val_dice=0.7997 | best=0.8255 (ep47) | 04:04:18 | L_main=0.0637 L_aux_1=0.0669(w=0.5) L_aux_2=0.0758(w=0.5)
[2026-06-23 21:01:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 193): 13421.7 MiB
[2026-06-23 21:02:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2261, pooled_mean_dice=0.8108, per_class=['0.8108'], iou=0.6818, recall=0.9910, precision=0.6860, vol_sim=0.8182, mcc=0.8191, min_class_dice=0.8108, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9212, per_class_sd=['0.9212'], combined(w=0.50)=0.8660, balanced=0.8188
[2026-06-23 21:02:09] INFO segtask_v1.trainer.trainer: Epoch 194/400 | LR=5.34e-04 | loss=0.2793 | val_dice=0.8108 | best=0.8255 (ep47) | 04:05:26 | L_main=0.0613 L_aux_1=0.0647(w=0.5) L_aux_2=0.0735(w=0.5)
[2026-06-23 21:02:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 194): 13421.7 MiB
[2026-06-23 21:03:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2568, pooled_mean_dice=0.8014, per_class=['0.8014'], iou=0.6687, recall=0.9885, precision=0.6739, vol_sim=0.8108, mcc=0.8114, min_class_dice=0.8014, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9144, per_class_sd=['0.9144'], combined(w=0.50)=0.8579, balanced=0.8096
[2026-06-23 21:03:22] INFO segtask_v1.trainer.trainer: Epoch 195/400 | LR=5.30e-04 | loss=0.2796 | val_dice=0.8014 | best=0.8255 (ep47) | 04:06:39 | L_main=0.0606 L_aux_1=0.0638(w=0.5) L_aux_2=0.0725(w=0.5)
[2026-06-23 21:03:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 195): 13421.7 MiB
[2026-06-23 21:04:38] INFO segtask_v1.trainer.validation:   Val: loss=0.2511, pooled_mean_dice=0.7860, per_class=['0.7860'], iou=0.6475, recall=0.9896, precision=0.6519, vol_sim=0.7943, mcc=0.7973, min_class_dice=0.7860, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9098, per_class_sd=['0.9098'], combined(w=0.50)=0.8479, balanced=0.7955
[2026-06-23 21:04:38] INFO segtask_v1.trainer.trainer: Epoch 196/400 | LR=5.26e-04 | loss=0.2914 | val_dice=0.7860 | best=0.8255 (ep47) | 04:07:54 | L_main=0.0679 L_aux_1=0.0710(w=0.5) L_aux_2=0.0805(w=0.5)
[2026-06-23 21:04:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 196): 13421.7 MiB
[2026-06-23 21:05:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2501, pooled_mean_dice=0.7865, per_class=['0.7865'], iou=0.6481, recall=0.9886, precision=0.6530, vol_sim=0.7955, mcc=0.7979, min_class_dice=0.7865, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9180, per_class_sd=['0.9180'], combined(w=0.50)=0.8523, balanced=0.7974
[2026-06-23 21:05:51] INFO segtask_v1.trainer.trainer: Epoch 197/400 | LR=5.22e-04 | loss=0.2948 | val_dice=0.7865 | best=0.8255 (ep47) | 04:09:08 | L_main=0.0680 L_aux_1=0.0706(w=0.5) L_aux_2=0.0802(w=0.5)
[2026-06-23 21:05:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 197): 13421.7 MiB
[2026-06-23 21:07:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2516, pooled_mean_dice=0.7849, per_class=['0.7849'], iou=0.6460, recall=0.9896, precision=0.6504, vol_sim=0.7932, mcc=0.7963, min_class_dice=0.7849, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9119, per_class_sd=['0.9119'], combined(w=0.50)=0.8484, balanced=0.7949
[2026-06-23 21:07:04] INFO segtask_v1.trainer.trainer: Epoch 198/400 | LR=5.18e-04 | loss=0.2817 | val_dice=0.7849 | best=0.8255 (ep47) | 04:10:21 | L_main=0.0618 L_aux_1=0.0647(w=0.5) L_aux_2=0.0736(w=0.5)
[2026-06-23 21:07:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 198): 13421.7 MiB
[2026-06-23 21:08:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2634, pooled_mean_dice=0.7950, per_class=['0.7950'], iou=0.6598, recall=0.9907, precision=0.6639, vol_sim=0.8025, mcc=0.8057, min_class_dice=0.7950, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9221, per_class_sd=['0.9221'], combined(w=0.50)=0.8586, balanced=0.8056
[2026-06-23 21:08:16] INFO segtask_v1.trainer.trainer: Epoch 199/400 | LR=5.14e-04 | loss=0.2779 | val_dice=0.7950 | best=0.8255 (ep47) | 04:11:33 | L_main=0.0605 L_aux_1=0.0636(w=0.5) L_aux_2=0.0729(w=0.5)
[2026-06-23 21:08:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 199): 13421.7 MiB
[2026-06-23 21:09:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2392, pooled_mean_dice=0.7884, per_class=['0.7884'], iou=0.6507, recall=0.9896, precision=0.6552, vol_sim=0.7967, mcc=0.8006, min_class_dice=0.7884, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9171, per_class_sd=['0.9171'], combined(w=0.50)=0.8527, balanced=0.7990
[2026-06-23 21:09:30] INFO segtask_v1.trainer.trainer: Epoch 200/400 | LR=5.10e-04 | loss=0.2888 | val_dice=0.7884 | best=0.8255 (ep47) | 04:12:47 | L_main=0.0641 L_aux_1=0.0673(w=0.5) L_aux_2=0.0763(w=0.5)
[2026-06-23 21:09:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 200): 13421.7 MiB
[2026-06-23 21:10:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2654, pooled_mean_dice=0.7785, per_class=['0.7785'], iou=0.6373, recall=0.9869, precision=0.6428, vol_sim=0.7888, mcc=0.7923, min_class_dice=0.7785, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9236, per_class_sd=['0.9236'], combined(w=0.50)=0.8510, balanced=0.7917
[2026-06-23 21:10:51] INFO segtask_v1.trainer.trainer: Epoch 201/400 | LR=5.06e-04 | loss=0.2877 | val_dice=0.7785 | best=0.8255 (ep47) | 04:14:08 | L_main=0.0644 L_aux_1=0.0679(w=0.5) L_aux_2=0.0774(w=0.5)
[2026-06-23 21:10:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 201): 13421.7 MiB
[2026-06-23 21:12:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2478, pooled_mean_dice=0.7985, per_class=['0.7985'], iou=0.6646, recall=0.9899, precision=0.6692, vol_sim=0.8067, mcc=0.8088, min_class_dice=0.7985, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9209, per_class_sd=['0.9209'], combined(w=0.50)=0.8597, balanced=0.8083
[2026-06-23 21:12:08] INFO segtask_v1.trainer.trainer: Epoch 202/400 | LR=5.02e-04 | loss=0.2906 | val_dice=0.7985 | best=0.8255 (ep47) | 04:15:25 | L_main=0.0659 L_aux_1=0.0692(w=0.5) L_aux_2=0.0787(w=0.5)
[2026-06-23 21:12:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 202): 13421.7 MiB
[2026-06-23 21:13:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2510, pooled_mean_dice=0.8007, per_class=['0.8007'], iou=0.6676, recall=0.9883, precision=0.6729, vol_sim=0.8101, mcc=0.8110, min_class_dice=0.8007, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9224, per_class_sd=['0.9224'], combined(w=0.50)=0.8616, balanced=0.8105
[2026-06-23 21:13:28] INFO segtask_v1.trainer.trainer: Epoch 203/400 | LR=4.99e-04 | loss=0.2804 | val_dice=0.8007 | best=0.8255 (ep47) | 04:16:44 | L_main=0.0620 L_aux_1=0.0653(w=0.5) L_aux_2=0.0746(w=0.5)
[2026-06-23 21:13:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 203): 13421.7 MiB
[2026-06-23 21:14:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2440, pooled_mean_dice=0.7895, per_class=['0.7895'], iou=0.6523, recall=0.9895, precision=0.6568, vol_sim=0.7980, mcc=0.8008, min_class_dice=0.7895, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9199, per_class_sd=['0.9199'], combined(w=0.50)=0.8547, balanced=0.8004
[2026-06-23 21:14:45] INFO segtask_v1.trainer.trainer: Epoch 204/400 | LR=4.95e-04 | loss=0.2848 | val_dice=0.7895 | best=0.8255 (ep47) | 04:18:02 | L_main=0.0641 L_aux_1=0.0669(w=0.5) L_aux_2=0.0764(w=0.5)
[2026-06-23 21:14:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 204): 13421.7 MiB
[2026-06-23 21:16:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2326, pooled_mean_dice=0.8095, per_class=['0.8095'], iou=0.6800, recall=0.9871, precision=0.6861, vol_sim=0.8201, mcc=0.8186, min_class_dice=0.8095, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9274, per_class_sd=['0.9274'], combined(w=0.50)=0.8685, balanced=0.8190
[2026-06-23 21:16:00] INFO segtask_v1.trainer.trainer: Epoch 205/400 | LR=4.91e-04 | loss=0.2996 | val_dice=0.8095 | best=0.8255 (ep47) | 04:19:16 | L_main=0.0685 L_aux_1=0.0718(w=0.5) L_aux_2=0.0815(w=0.5)
[2026-06-23 21:16:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 205): 13421.7 MiB
[2026-06-23 21:17:11] INFO segtask_v1.trainer.validation:   Val: loss=0.2342, pooled_mean_dice=0.8036, per_class=['0.8036'], iou=0.6717, recall=0.9885, precision=0.6770, vol_sim=0.8129, mcc=0.8132, min_class_dice=0.8036, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9181, per_class_sd=['0.9181'], combined(w=0.50)=0.8608, balanced=0.8121
[2026-06-23 21:17:11] INFO segtask_v1.trainer.trainer: Epoch 206/400 | LR=4.87e-04 | loss=0.2833 | val_dice=0.8036 | best=0.8255 (ep47) | 04:20:28 | L_main=0.0624 L_aux_1=0.0656(w=0.5) L_aux_2=0.0749(w=0.5)
[2026-06-23 21:17:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 206): 13421.7 MiB
[2026-06-23 21:18:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2488, pooled_mean_dice=0.7935, per_class=['0.7935'], iou=0.6576, recall=0.9876, precision=0.6631, vol_sim=0.8034, mcc=0.8042, min_class_dice=0.7935, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9194, per_class_sd=['0.9194'], combined(w=0.50)=0.8564, balanced=0.8037
[2026-06-23 21:18:26] INFO segtask_v1.trainer.trainer: Epoch 207/400 | LR=4.83e-04 | loss=0.2864 | val_dice=0.7935 | best=0.8255 (ep47) | 04:21:43 | L_main=0.0635 L_aux_1=0.0671(w=0.5) L_aux_2=0.0763(w=0.5)
[2026-06-23 21:18:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 207): 13421.7 MiB
[2026-06-23 21:19:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2096, pooled_mean_dice=0.8221, per_class=['0.8221'], iou=0.6979, recall=0.9885, precision=0.7036, vol_sim=0.8316, mcc=0.8288, min_class_dice=0.8221, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9282, per_class_sd=['0.9282'], combined(w=0.50)=0.8751, balanced=0.8299
[2026-06-23 21:19:41] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 21:19:41] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.8299 at epoch 208
[2026-06-23 21:19:41] INFO segtask_v1.trainer.trainer: Epoch 208/400 | LR=4.79e-04 | loss=0.2775 | val_dice=0.8221 | best=0.8299 (ep208) | 04:22:57 | L_main=0.0600 L_aux_1=0.0630(w=0.5) L_aux_2=0.0715(w=0.5)
[2026-06-23 21:19:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 208): 13421.7 MiB
[2026-06-23 21:20:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2473, pooled_mean_dice=0.7974, per_class=['0.7974'], iou=0.6631, recall=0.9862, precision=0.6693, vol_sim=0.8086, mcc=0.8086, min_class_dice=0.7974, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.9246, per_class_sd=['0.9246'], combined(w=0.50)=0.8610, balanced=0.8082
[2026-06-23 21:20:56] INFO segtask_v1.trainer.trainer: Epoch 209/400 | LR=4.75e-04 | loss=0.2795 | val_dice=0.7974 | best=0.8299 (ep208) | 04:24:13 | L_main=0.0620 L_aux_1=0.0653(w=0.5) L_aux_2=0.0743(w=0.5)
[2026-06-23 21:20:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 209): 13421.7 MiB
[2026-06-23 21:22:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2406, pooled_mean_dice=0.7983, per_class=['0.7983'], iou=0.6643, recall=0.9884, precision=0.6695, vol_sim=0.8076, mcc=0.8089, min_class_dice=0.7983, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.9302, per_class_sd=['0.9302'], combined(w=0.50)=0.8643, balanced=0.8100
[2026-06-23 21:22:12] INFO segtask_v1.trainer.trainer: Epoch 210/400 | LR=4.71e-04 | loss=0.2792 | val_dice=0.7983 | best=0.8299 (ep208) | 04:25:29 | L_main=0.0615 L_aux_1=0.0643(w=0.5) L_aux_2=0.0730(w=0.5)
[2026-06-23 21:22:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 210): 13421.7 MiB
[2026-06-23 21:23:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2437, pooled_mean_dice=0.7909, per_class=['0.7909'], iou=0.6542, recall=0.9870, precision=0.6599, vol_sim=0.8014, mcc=0.8022, min_class_dice=0.7909, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9224, per_class_sd=['0.9224'], combined(w=0.50)=0.8567, balanced=0.8021
[2026-06-23 21:23:28] INFO segtask_v1.trainer.trainer: Epoch 211/400 | LR=4.67e-04 | loss=0.2824 | val_dice=0.7909 | best=0.8299 (ep208) | 04:26:45 | L_main=0.0618 L_aux_1=0.0652(w=0.5) L_aux_2=0.0740(w=0.5)
[2026-06-23 21:23:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 211): 13421.7 MiB
[2026-06-23 21:24:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2148, pooled_mean_dice=0.8272, per_class=['0.8272'], iou=0.7054, recall=0.9862, precision=0.7124, vol_sim=0.8388, mcc=0.8332, min_class_dice=0.8272, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9366, per_class_sd=['0.9366'], combined(w=0.50)=0.8819, balanced=0.8359
[2026-06-23 21:24:46] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 21:24:46] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.8359 at epoch 212
[2026-06-23 21:24:46] INFO segtask_v1.trainer.trainer: Epoch 212/400 | LR=4.63e-04 | loss=0.2749 | val_dice=0.8272 | best=0.8359 (ep212) | 04:28:03 | L_main=0.0603 L_aux_1=0.0635(w=0.5) L_aux_2=0.0723(w=0.5)
[2026-06-23 21:24:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 212): 13421.7 MiB
[2026-06-23 21:26:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2456, pooled_mean_dice=0.8059, per_class=['0.8059'], iou=0.6750, recall=0.9874, precision=0.6808, vol_sim=0.8162, mcc=0.8145, min_class_dice=0.8059, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.9273, per_class_sd=['0.9273'], combined(w=0.50)=0.8666, balanced=0.8158
[2026-06-23 21:26:03] INFO segtask_v1.trainer.trainer: Epoch 213/400 | LR=4.59e-04 | loss=0.2808 | val_dice=0.8059 | best=0.8359 (ep212) | 04:29:20 | L_main=0.0615 L_aux_1=0.0650(w=0.5) L_aux_2=0.0743(w=0.5)
[2026-06-23 21:26:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 213): 13421.7 MiB
[2026-06-23 21:27:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2168, pooled_mean_dice=0.8115, per_class=['0.8115'], iou=0.6828, recall=0.9883, precision=0.6884, vol_sim=0.8211, mcc=0.8192, min_class_dice=0.8115, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9266, per_class_sd=['0.9266'], combined(w=0.50)=0.8691, balanced=0.8205
[2026-06-23 21:27:19] INFO segtask_v1.trainer.trainer: Epoch 214/400 | LR=4.55e-04 | loss=0.2770 | val_dice=0.8115 | best=0.8359 (ep212) | 04:30:36 | L_main=0.0613 L_aux_1=0.0644(w=0.5) L_aux_2=0.0736(w=0.5)
[2026-06-23 21:27:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 214): 13421.7 MiB
[2026-06-23 21:28:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2393, pooled_mean_dice=0.7888, per_class=['0.7888'], iou=0.6512, recall=0.9851, precision=0.6577, vol_sim=0.8007, mcc=0.8005, min_class_dice=0.7888, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9197, per_class_sd=['0.9197'], combined(w=0.50)=0.8542, balanced=0.7998
[2026-06-23 21:28:28] INFO segtask_v1.trainer.trainer: Epoch 215/400 | LR=4.51e-04 | loss=0.2719 | val_dice=0.7888 | best=0.8359 (ep212) | 04:31:45 | L_main=0.0579 L_aux_1=0.0608(w=0.5) L_aux_2=0.0692(w=0.5)
[2026-06-23 21:28:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 215): 13421.7 MiB
[2026-06-23 21:29:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2409, pooled_mean_dice=0.8023, per_class=['0.8023'], iou=0.6699, recall=0.9890, precision=0.6750, vol_sim=0.8113, mcc=0.8122, min_class_dice=0.8023, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9194, per_class_sd=['0.9194'], combined(w=0.50)=0.8609, balanced=0.8113
[2026-06-23 21:29:43] INFO segtask_v1.trainer.trainer: Epoch 216/400 | LR=4.47e-04 | loss=0.2760 | val_dice=0.8023 | best=0.8359 (ep212) | 04:33:00 | L_main=0.0613 L_aux_1=0.0647(w=0.5) L_aux_2=0.0735(w=0.5)
[2026-06-23 21:29:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 216): 13421.7 MiB
[2026-06-23 21:31:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2359, pooled_mean_dice=0.7973, per_class=['0.7973'], iou=0.6629, recall=0.9889, precision=0.6679, vol_sim=0.8063, mcc=0.8078, min_class_dice=0.7973, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9214, per_class_sd=['0.9214'], combined(w=0.50)=0.8593, balanced=0.8074
[2026-06-23 21:31:02] INFO segtask_v1.trainer.trainer: Epoch 217/400 | LR=4.43e-04 | loss=0.2768 | val_dice=0.7973 | best=0.8359 (ep212) | 04:34:19 | L_main=0.0611 L_aux_1=0.0638(w=0.5) L_aux_2=0.0723(w=0.5)
[2026-06-23 21:31:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 217): 13421.7 MiB
[2026-06-23 21:32:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2230, pooled_mean_dice=0.8099, per_class=['0.8099'], iou=0.6805, recall=0.9860, precision=0.6872, vol_sim=0.8214, mcc=0.8176, min_class_dice=0.8099, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9205, per_class_sd=['0.9205'], combined(w=0.50)=0.8652, balanced=0.8179
[2026-06-23 21:32:17] INFO segtask_v1.trainer.trainer: Epoch 218/400 | LR=4.39e-04 | loss=0.2740 | val_dice=0.8099 | best=0.8359 (ep212) | 04:35:33 | L_main=0.0586 L_aux_1=0.0614(w=0.5) L_aux_2=0.0699(w=0.5)
[2026-06-23 21:32:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 218): 13421.7 MiB
[2026-06-23 21:33:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2377, pooled_mean_dice=0.7987, per_class=['0.7987'], iou=0.6649, recall=0.9858, precision=0.6714, vol_sim=0.8103, mcc=0.8085, min_class_dice=0.7987, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9261, per_class_sd=['0.9261'], combined(w=0.50)=0.8624, balanced=0.8095
[2026-06-23 21:33:36] INFO segtask_v1.trainer.trainer: Epoch 219/400 | LR=4.35e-04 | loss=0.2709 | val_dice=0.7987 | best=0.8359 (ep212) | 04:36:53 | L_main=0.0588 L_aux_1=0.0614(w=0.5) L_aux_2=0.0695(w=0.5)
[2026-06-23 21:33:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 219): 13421.7 MiB
[2026-06-23 21:34:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2239, pooled_mean_dice=0.8055, per_class=['0.8055'], iou=0.6743, recall=0.9887, precision=0.6796, vol_sim=0.8147, mcc=0.8148, min_class_dice=0.8055, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.9255, per_class_sd=['0.9255'], combined(w=0.50)=0.8655, balanced=0.8152
[2026-06-23 21:34:55] INFO segtask_v1.trainer.trainer: Epoch 220/400 | LR=4.31e-04 | loss=0.2822 | val_dice=0.8055 | best=0.8359 (ep212) | 04:38:12 | L_main=0.0649 L_aux_1=0.0676(w=0.5) L_aux_2=0.0759(w=0.5)
[2026-06-23 21:34:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 220): 13421.7 MiB
[2026-06-23 21:36:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2187, pooled_mean_dice=0.8157, per_class=['0.8157'], iou=0.6888, recall=0.9879, precision=0.6947, vol_sim=0.8258, mcc=0.8230, min_class_dice=0.8157, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9270, per_class_sd=['0.9270'], combined(w=0.50)=0.8714, balanced=0.8242
[2026-06-23 21:36:07] INFO segtask_v1.trainer.trainer: Epoch 221/400 | LR=4.27e-04 | loss=0.2870 | val_dice=0.8157 | best=0.8359 (ep212) | 04:39:24 | L_main=0.0647 L_aux_1=0.0679(w=0.5) L_aux_2=0.0775(w=0.5)
[2026-06-23 21:36:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 221): 13421.7 MiB
[2026-06-23 21:37:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2388, pooled_mean_dice=0.7986, per_class=['0.7986'], iou=0.6648, recall=0.9876, precision=0.6703, vol_sim=0.8086, mcc=0.8086, min_class_dice=0.7986, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9208, per_class_sd=['0.9208'], combined(w=0.50)=0.8597, balanced=0.8084
[2026-06-23 21:37:21] INFO segtask_v1.trainer.trainer: Epoch 222/400 | LR=4.23e-04 | loss=0.2703 | val_dice=0.7986 | best=0.8359 (ep212) | 04:40:38 | L_main=0.0579 L_aux_1=0.0607(w=0.5) L_aux_2=0.0686(w=0.5)
[2026-06-23 21:37:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 222): 13421.7 MiB
[2026-06-23 21:38:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2342, pooled_mean_dice=0.8069, per_class=['0.8069'], iou=0.6763, recall=0.9888, precision=0.6815, vol_sim=0.8160, mcc=0.8152, min_class_dice=0.8069, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9276, per_class_sd=['0.9276'], combined(w=0.50)=0.8672, balanced=0.8167
[2026-06-23 21:38:36] INFO segtask_v1.trainer.trainer: Epoch 223/400 | LR=4.19e-04 | loss=0.2770 | val_dice=0.8069 | best=0.8359 (ep212) | 04:41:53 | L_main=0.0603 L_aux_1=0.0633(w=0.5) L_aux_2=0.0718(w=0.5)
[2026-06-23 21:38:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 223): 13421.7 MiB
[2026-06-23 21:39:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2369, pooled_mean_dice=0.8036, per_class=['0.8036'], iou=0.6716, recall=0.9860, precision=0.6781, vol_sim=0.8150, mcc=0.8135, min_class_dice=0.8036, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.9287, per_class_sd=['0.9287'], combined(w=0.50)=0.8662, balanced=0.8142
[2026-06-23 21:39:46] INFO segtask_v1.trainer.trainer: Epoch 224/400 | LR=4.16e-04 | loss=0.2778 | val_dice=0.8036 | best=0.8359 (ep212) | 04:43:03 | L_main=0.0613 L_aux_1=0.0641(w=0.5) L_aux_2=0.0722(w=0.5)
[2026-06-23 21:39:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 224): 13421.7 MiB
[2026-06-23 21:41:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2474, pooled_mean_dice=0.8020, per_class=['0.8020'], iou=0.6694, recall=0.9876, precision=0.6751, vol_sim=0.8121, mcc=0.8120, min_class_dice=0.8020, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9260, per_class_sd=['0.9260'], combined(w=0.50)=0.8640, balanced=0.8123
[2026-06-23 21:41:01] INFO segtask_v1.trainer.trainer: Epoch 225/400 | LR=4.12e-04 | loss=0.2818 | val_dice=0.8020 | best=0.8359 (ep212) | 04:44:18 | L_main=0.0631 L_aux_1=0.0659(w=0.5) L_aux_2=0.0741(w=0.5)
[2026-06-23 21:41:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 225): 13421.7 MiB
[2026-06-23 21:42:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2199, pooled_mean_dice=0.8103, per_class=['0.8103'], iou=0.6811, recall=0.9859, precision=0.6878, vol_sim=0.8219, mcc=0.8187, min_class_dice=0.8103, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9278, per_class_sd=['0.9278'], combined(w=0.50)=0.8690, balanced=0.8197
[2026-06-23 21:42:19] INFO segtask_v1.trainer.trainer: Epoch 226/400 | LR=4.08e-04 | loss=0.2832 | val_dice=0.8103 | best=0.8359 (ep212) | 04:45:36 | L_main=0.0631 L_aux_1=0.0659(w=0.5) L_aux_2=0.0747(w=0.5)
[2026-06-23 21:42:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 226): 13421.7 MiB
[2026-06-23 21:43:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2419, pooled_mean_dice=0.7960, per_class=['0.7960'], iou=0.6611, recall=0.9860, precision=0.6674, vol_sim=0.8073, mcc=0.8057, min_class_dice=0.7960, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9274, per_class_sd=['0.9274'], combined(w=0.50)=0.8617, balanced=0.8073
[2026-06-23 21:43:32] INFO segtask_v1.trainer.trainer: Epoch 227/400 | LR=4.04e-04 | loss=0.2820 | val_dice=0.7960 | best=0.8359 (ep212) | 04:46:49 | L_main=0.0616 L_aux_1=0.0648(w=0.5) L_aux_2=0.0734(w=0.5)
[2026-06-23 21:43:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 227): 13421.7 MiB
[2026-06-23 21:44:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2473, pooled_mean_dice=0.7967, per_class=['0.7967'], iou=0.6621, recall=0.9843, precision=0.6692, vol_sim=0.8094, mcc=0.8073, min_class_dice=0.7967, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9288, per_class_sd=['0.9288'], combined(w=0.50)=0.8628, balanced=0.8083
[2026-06-23 21:44:47] INFO segtask_v1.trainer.trainer: Epoch 228/400 | LR=4.00e-04 | loss=0.2870 | val_dice=0.7967 | best=0.8359 (ep212) | 04:48:04 | L_main=0.0646 L_aux_1=0.0680(w=0.5) L_aux_2=0.0775(w=0.5)
[2026-06-23 21:44:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 228): 13421.7 MiB
[2026-06-23 21:46:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2078, pooled_mean_dice=0.8200, per_class=['0.8200'], iou=0.6950, recall=0.9887, precision=0.7005, vol_sim=0.8294, mcc=0.8272, min_class_dice=0.8200, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9389, per_class_sd=['0.9389'], combined(w=0.50)=0.8794, balanced=0.8302
[2026-06-23 21:46:00] INFO segtask_v1.trainer.trainer: Epoch 229/400 | LR=3.96e-04 | loss=0.2798 | val_dice=0.8200 | best=0.8359 (ep212) | 04:49:17 | L_main=0.0607 L_aux_1=0.0636(w=0.5) L_aux_2=0.0722(w=0.5)
[2026-06-23 21:46:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 229): 13421.7 MiB
[2026-06-23 21:47:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2424, pooled_mean_dice=0.7932, per_class=['0.7932'], iou=0.6572, recall=0.9854, precision=0.6637, vol_sim=0.8049, mcc=0.8040, min_class_dice=0.7932, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9258, per_class_sd=['0.9258'], combined(w=0.50)=0.8595, balanced=0.8047
[2026-06-23 21:47:09] INFO segtask_v1.trainer.trainer: Epoch 230/400 | LR=3.92e-04 | loss=0.2778 | val_dice=0.7932 | best=0.8359 (ep212) | 04:50:26 | L_main=0.0626 L_aux_1=0.0654(w=0.5) L_aux_2=0.0738(w=0.5)
[2026-06-23 21:47:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 230): 13421.7 MiB
[2026-06-23 21:48:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2284, pooled_mean_dice=0.8004, per_class=['0.8004'], iou=0.6673, recall=0.9878, precision=0.6728, vol_sim=0.8103, mcc=0.8107, min_class_dice=0.8004, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9300, per_class_sd=['0.9300'], combined(w=0.50)=0.8652, balanced=0.8117
[2026-06-23 21:48:33] INFO segtask_v1.trainer.trainer: Epoch 231/400 | LR=3.88e-04 | loss=0.2810 | val_dice=0.8004 | best=0.8359 (ep212) | 04:51:50 | L_main=0.0620 L_aux_1=0.0654(w=0.5) L_aux_2=0.0746(w=0.5)
[2026-06-23 21:48:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 231): 13421.7 MiB
[2026-06-23 21:49:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2172, pooled_mean_dice=0.8151, per_class=['0.8151'], iou=0.6880, recall=0.9896, precision=0.6930, vol_sim=0.8237, mcc=0.8230, min_class_dice=0.8151, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9259, per_class_sd=['0.9259'], combined(w=0.50)=0.8705, balanced=0.8235
[2026-06-23 21:49:42] INFO segtask_v1.trainer.trainer: Epoch 232/400 | LR=3.84e-04 | loss=0.2760 | val_dice=0.8151 | best=0.8359 (ep212) | 04:52:59 | L_main=0.0600 L_aux_1=0.0627(w=0.5) L_aux_2=0.0712(w=0.5)
[2026-06-23 21:49:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 232): 13421.7 MiB
[2026-06-23 21:50:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2208, pooled_mean_dice=0.8117, per_class=['0.8117'], iou=0.6831, recall=0.9879, precision=0.6888, vol_sim=0.8216, mcc=0.8200, min_class_dice=0.8117, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9291, per_class_sd=['0.9291'], combined(w=0.50)=0.8704, balanced=0.8212
[2026-06-23 21:50:55] INFO segtask_v1.trainer.trainer: Epoch 233/400 | LR=3.81e-04 | loss=0.2717 | val_dice=0.8117 | best=0.8359 (ep212) | 04:54:12 | L_main=0.0591 L_aux_1=0.0618(w=0.5) L_aux_2=0.0698(w=0.5)
[2026-06-23 21:50:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 233): 13421.7 MiB
[2026-06-23 21:52:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2341, pooled_mean_dice=0.7983, per_class=['0.7983'], iou=0.6643, recall=0.9894, precision=0.6691, vol_sim=0.8069, mcc=0.8085, min_class_dice=0.7983, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9257, per_class_sd=['0.9257'], combined(w=0.50)=0.8620, balanced=0.8090
[2026-06-23 21:52:10] INFO segtask_v1.trainer.trainer: Epoch 234/400 | LR=3.77e-04 | loss=0.2720 | val_dice=0.7983 | best=0.8359 (ep212) | 04:55:27 | L_main=0.0602 L_aux_1=0.0624(w=0.5) L_aux_2=0.0700(w=0.5)
[2026-06-23 21:52:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 234): 13421.7 MiB
[2026-06-23 21:53:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2512, pooled_mean_dice=0.7975, per_class=['0.7975'], iou=0.6631, recall=0.9893, precision=0.6679, vol_sim=0.8060, mcc=0.8078, min_class_dice=0.7975, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9285, per_class_sd=['0.9285'], combined(w=0.50)=0.8630, balanced=0.8089
[2026-06-23 21:53:22] INFO segtask_v1.trainer.trainer: Epoch 235/400 | LR=3.73e-04 | loss=0.2706 | val_dice=0.7975 | best=0.8359 (ep212) | 04:56:39 | L_main=0.0590 L_aux_1=0.0619(w=0.5) L_aux_2=0.0703(w=0.5)
[2026-06-23 21:53:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 235): 13421.7 MiB
[2026-06-23 21:54:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2607, pooled_mean_dice=0.7939, per_class=['0.7939'], iou=0.6582, recall=0.9859, precision=0.6645, vol_sim=0.8053, mcc=0.8047, min_class_dice=0.7939, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9233, per_class_sd=['0.9233'], combined(w=0.50)=0.8586, balanced=0.8048
[2026-06-23 21:54:41] INFO segtask_v1.trainer.trainer: Epoch 236/400 | LR=3.69e-04 | loss=0.2735 | val_dice=0.7939 | best=0.8359 (ep212) | 04:57:58 | L_main=0.0604 L_aux_1=0.0629(w=0.5) L_aux_2=0.0714(w=0.5)
[2026-06-23 21:54:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 236): 13421.7 MiB
[2026-06-23 21:56:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2198, pooled_mean_dice=0.8116, per_class=['0.8116'], iou=0.6830, recall=0.9867, precision=0.6893, vol_sim=0.8225, mcc=0.8198, min_class_dice=0.8116, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9287, per_class_sd=['0.9287'], combined(w=0.50)=0.8702, balanced=0.8210
[2026-06-23 21:56:00] INFO segtask_v1.trainer.trainer: Epoch 237/400 | LR=3.65e-04 | loss=0.2631 | val_dice=0.8116 | best=0.8359 (ep212) | 04:59:17 | L_main=0.0564 L_aux_1=0.0591(w=0.5) L_aux_2=0.0669(w=0.5)
[2026-06-23 21:56:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 237): 13421.7 MiB
[2026-06-23 21:57:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2336, pooled_mean_dice=0.7938, per_class=['0.7938'], iou=0.6581, recall=0.9871, precision=0.6637, vol_sim=0.8041, mcc=0.8054, min_class_dice=0.7938, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9245, per_class_sd=['0.9245'], combined(w=0.50)=0.8591, balanced=0.8050
[2026-06-23 21:57:18] INFO segtask_v1.trainer.trainer: Epoch 238/400 | LR=3.61e-04 | loss=0.2807 | val_dice=0.7938 | best=0.8359 (ep212) | 05:00:35 | L_main=0.0622 L_aux_1=0.0651(w=0.5) L_aux_2=0.0739(w=0.5)
[2026-06-23 21:57:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 238): 13421.7 MiB
[2026-06-23 21:58:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2335, pooled_mean_dice=0.8028, per_class=['0.8028'], iou=0.6706, recall=0.9890, precision=0.6757, vol_sim=0.8118, mcc=0.8125, min_class_dice=0.8028, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9259, per_class_sd=['0.9259'], combined(w=0.50)=0.8644, balanced=0.8130
[2026-06-23 21:58:32] INFO segtask_v1.trainer.trainer: Epoch 239/400 | LR=3.58e-04 | loss=0.2655 | val_dice=0.8028 | best=0.8359 (ep212) | 05:01:48 | L_main=0.0563 L_aux_1=0.0590(w=0.5) L_aux_2=0.0670(w=0.5)
[2026-06-23 21:58:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 239): 13421.7 MiB
[2026-06-23 21:59:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2132, pooled_mean_dice=0.8166, per_class=['0.8166'], iou=0.6901, recall=0.9891, precision=0.6954, vol_sim=0.8256, mcc=0.8237, min_class_dice=0.8166, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9240, per_class_sd=['0.9240'], combined(w=0.50)=0.8703, balanced=0.8243
[2026-06-23 21:59:47] INFO segtask_v1.trainer.trainer: Epoch 240/400 | LR=3.54e-04 | loss=0.2666 | val_dice=0.8166 | best=0.8359 (ep212) | 05:03:04 | L_main=0.0577 L_aux_1=0.0604(w=0.5) L_aux_2=0.0683(w=0.5)
[2026-06-23 21:59:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 240): 13421.7 MiB
[2026-06-23 22:01:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2387, pooled_mean_dice=0.7954, per_class=['0.7954'], iou=0.6603, recall=0.9876, precision=0.6659, vol_sim=0.8054, mcc=0.8050, min_class_dice=0.7954, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9239, per_class_sd=['0.9239'], combined(w=0.50)=0.8597, balanced=0.8062
[2026-06-23 22:01:03] INFO segtask_v1.trainer.trainer: Epoch 241/400 | LR=3.50e-04 | loss=0.2767 | val_dice=0.7954 | best=0.8359 (ep212) | 05:04:20 | L_main=0.0611 L_aux_1=0.0641(w=0.5) L_aux_2=0.0729(w=0.5)
[2026-06-23 22:01:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 241): 13421.7 MiB
[2026-06-23 22:02:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2458, pooled_mean_dice=0.8014, per_class=['0.8014'], iou=0.6685, recall=0.9865, precision=0.6747, vol_sim=0.8123, mcc=0.8116, min_class_dice=0.8014, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9315, per_class_sd=['0.9315'], combined(w=0.50)=0.8664, balanced=0.8128
[2026-06-23 22:02:19] INFO segtask_v1.trainer.trainer: Epoch 242/400 | LR=3.46e-04 | loss=0.2706 | val_dice=0.8014 | best=0.8359 (ep212) | 05:05:36 | L_main=0.0586 L_aux_1=0.0612(w=0.5) L_aux_2=0.0690(w=0.5)
[2026-06-23 22:02:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 242): 13421.7 MiB
[2026-06-23 22:03:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2389, pooled_mean_dice=0.8037, per_class=['0.8037'], iou=0.6718, recall=0.9857, precision=0.6784, vol_sim=0.8153, mcc=0.8131, min_class_dice=0.8037, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9301, per_class_sd=['0.9301'], combined(w=0.50)=0.8669, balanced=0.8145
[2026-06-23 22:03:28] INFO segtask_v1.trainer.trainer: Epoch 243/400 | LR=3.42e-04 | loss=0.2771 | val_dice=0.8037 | best=0.8359 (ep212) | 05:06:45 | L_main=0.0621 L_aux_1=0.0645(w=0.5) L_aux_2=0.0726(w=0.5)
[2026-06-23 22:03:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 243): 13421.7 MiB
[2026-06-23 22:04:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2426, pooled_mean_dice=0.7950, per_class=['0.7950'], iou=0.6597, recall=0.9858, precision=0.6660, vol_sim=0.8064, mcc=0.8056, min_class_dice=0.7950, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9312, per_class_sd=['0.9312'], combined(w=0.50)=0.8631, balanced=0.8072
[2026-06-23 22:04:50] INFO segtask_v1.trainer.trainer: Epoch 244/400 | LR=3.39e-04 | loss=0.2711 | val_dice=0.7950 | best=0.8359 (ep212) | 05:08:07 | L_main=0.0583 L_aux_1=0.0612(w=0.5) L_aux_2=0.0693(w=0.5)
[2026-06-23 22:04:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 244): 13421.7 MiB
[2026-06-23 22:06:13] INFO segtask_v1.trainer.validation:   Val: loss=0.2534, pooled_mean_dice=0.7951, per_class=['0.7951'], iou=0.6599, recall=0.9878, precision=0.6653, vol_sim=0.8049, mcc=0.8058, min_class_dice=0.7951, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9264, per_class_sd=['0.9264'], combined(w=0.50)=0.8608, balanced=0.8065
[2026-06-23 22:06:13] INFO segtask_v1.trainer.trainer: Epoch 245/400 | LR=3.35e-04 | loss=0.2671 | val_dice=0.7951 | best=0.8359 (ep212) | 05:09:29 | L_main=0.0581 L_aux_1=0.0610(w=0.5) L_aux_2=0.0689(w=0.5)
[2026-06-23 22:06:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 245): 13421.7 MiB
[2026-06-23 22:07:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2535, pooled_mean_dice=0.8070, per_class=['0.8070'], iou=0.6764, recall=0.9869, precision=0.6825, vol_sim=0.8177, mcc=0.8157, min_class_dice=0.8070, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9332, per_class_sd=['0.9332'], combined(w=0.50)=0.8701, balanced=0.8179
[2026-06-23 22:07:21] INFO segtask_v1.trainer.trainer: Epoch 246/400 | LR=3.31e-04 | loss=0.2660 | val_dice=0.8070 | best=0.8359 (ep212) | 05:10:38 | L_main=0.0568 L_aux_1=0.0595(w=0.5) L_aux_2=0.0672(w=0.5)
[2026-06-23 22:07:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 246): 13421.7 MiB
[2026-06-23 22:08:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2193, pooled_mean_dice=0.8095, per_class=['0.8095'], iou=0.6799, recall=0.9893, precision=0.6849, vol_sim=0.8182, mcc=0.8188, min_class_dice=0.8095, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9284, per_class_sd=['0.9284'], combined(w=0.50)=0.8689, balanced=0.8192
[2026-06-23 22:08:41] INFO segtask_v1.trainer.trainer: Epoch 247/400 | LR=3.27e-04 | loss=0.2647 | val_dice=0.8095 | best=0.8359 (ep212) | 05:11:58 | L_main=0.0568 L_aux_1=0.0596(w=0.5) L_aux_2=0.0673(w=0.5)
[2026-06-23 22:08:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 247): 13421.7 MiB
[2026-06-23 22:09:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2172, pooled_mean_dice=0.8129, per_class=['0.8129'], iou=0.6848, recall=0.9838, precision=0.6926, vol_sim=0.8263, mcc=0.8213, min_class_dice=0.8129, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9284, per_class_sd=['0.9284'], combined(w=0.50)=0.8707, balanced=0.8221
[2026-06-23 22:09:57] INFO segtask_v1.trainer.trainer: Epoch 248/400 | LR=3.24e-04 | loss=0.2614 | val_dice=0.8129 | best=0.8359 (ep212) | 05:13:14 | L_main=0.0565 L_aux_1=0.0590(w=0.5) L_aux_2=0.0665(w=0.5)
[2026-06-23 22:09:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 248): 13421.7 MiB
[2026-06-23 22:11:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2200, pooled_mean_dice=0.8105, per_class=['0.8105'], iou=0.6814, recall=0.9874, precision=0.6874, vol_sim=0.8209, mcc=0.8185, min_class_dice=0.8105, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9283, per_class_sd=['0.9283'], combined(w=0.50)=0.8694, balanced=0.8200
[2026-06-23 22:11:15] INFO segtask_v1.trainer.trainer: Epoch 249/400 | LR=3.20e-04 | loss=0.2675 | val_dice=0.8105 | best=0.8359 (ep212) | 05:14:31 | L_main=0.0584 L_aux_1=0.0611(w=0.5) L_aux_2=0.0697(w=0.5)
[2026-06-23 22:11:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 249): 13421.7 MiB
[2026-06-23 22:12:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2502, pooled_mean_dice=0.7910, per_class=['0.7910'], iou=0.6543, recall=0.9847, precision=0.6610, vol_sim=0.8033, mcc=0.8024, min_class_dice=0.7910, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9334, per_class_sd=['0.9334'], combined(w=0.50)=0.8622, balanced=0.8043
[2026-06-23 22:12:36] INFO segtask_v1.trainer.trainer: Epoch 250/400 | LR=3.16e-04 | loss=0.2661 | val_dice=0.7910 | best=0.8359 (ep212) | 05:15:53 | L_main=0.0560 L_aux_1=0.0588(w=0.5) L_aux_2=0.0665(w=0.5)
[2026-06-23 22:12:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 250): 13421.7 MiB
[2026-06-23 22:13:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2209, pooled_mean_dice=0.8111, per_class=['0.8111'], iou=0.6823, recall=0.9844, precision=0.6897, vol_sim=0.8240, mcc=0.8196, min_class_dice=0.8111, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9340, per_class_sd=['0.9340'], combined(w=0.50)=0.8726, balanced=0.8216
[2026-06-23 22:13:52] INFO segtask_v1.trainer.trainer: Epoch 251/400 | LR=3.13e-04 | loss=0.2772 | val_dice=0.8111 | best=0.8359 (ep212) | 05:17:09 | L_main=0.0602 L_aux_1=0.0634(w=0.5) L_aux_2=0.0722(w=0.5)
[2026-06-23 22:13:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 251): 13421.7 MiB
[2026-06-23 22:15:10] INFO segtask_v1.trainer.validation:   Val: loss=0.1996, pooled_mean_dice=0.8260, per_class=['0.8260'], iou=0.7036, recall=0.9869, precision=0.7102, vol_sim=0.8370, mcc=0.8326, min_class_dice=0.8260, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9366, per_class_sd=['0.9366'], combined(w=0.50)=0.8813, balanced=0.8349
[2026-06-23 22:15:10] INFO segtask_v1.trainer.trainer: Epoch 252/400 | LR=3.09e-04 | loss=0.2738 | val_dice=0.8260 | best=0.8359 (ep212) | 05:18:27 | L_main=0.0593 L_aux_1=0.0627(w=0.5) L_aux_2=0.0713(w=0.5)
[2026-06-23 22:15:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 252): 13421.7 MiB
[2026-06-23 22:16:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2316, pooled_mean_dice=0.8029, per_class=['0.8029'], iou=0.6707, recall=0.9886, precision=0.6760, vol_sim=0.8122, mcc=0.8131, min_class_dice=0.8029, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9280, per_class_sd=['0.9280'], combined(w=0.50)=0.8654, balanced=0.8135
[2026-06-23 22:16:33] INFO segtask_v1.trainer.trainer: Epoch 253/400 | LR=3.05e-04 | loss=0.2829 | val_dice=0.8029 | best=0.8359 (ep212) | 05:19:50 | L_main=0.0627 L_aux_1=0.0659(w=0.5) L_aux_2=0.0750(w=0.5)
[2026-06-23 22:16:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 253): 13421.7 MiB
[2026-06-23 22:17:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2339, pooled_mean_dice=0.8069, per_class=['0.8069'], iou=0.6764, recall=0.9868, precision=0.6825, vol_sim=0.8177, mcc=0.8166, min_class_dice=0.8069, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9286, per_class_sd=['0.9286'], combined(w=0.50)=0.8678, balanced=0.8171
[2026-06-23 22:17:43] INFO segtask_v1.trainer.trainer: Epoch 254/400 | LR=3.02e-04 | loss=0.2732 | val_dice=0.8069 | best=0.8359 (ep212) | 05:21:00 | L_main=0.0599 L_aux_1=0.0630(w=0.5) L_aux_2=0.0715(w=0.5)
[2026-06-23 22:17:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 254): 13421.7 MiB
[2026-06-23 22:18:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2371, pooled_mean_dice=0.8014, per_class=['0.8014'], iou=0.6686, recall=0.9853, precision=0.6754, vol_sim=0.8134, mcc=0.8109, min_class_dice=0.8014, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9326, per_class_sd=['0.9326'], combined(w=0.50)=0.8670, balanced=0.8130
[2026-06-23 22:18:59] INFO segtask_v1.trainer.trainer: Epoch 255/400 | LR=2.98e-04 | loss=0.2700 | val_dice=0.8014 | best=0.8359 (ep212) | 05:22:16 | L_main=0.0577 L_aux_1=0.0603(w=0.5) L_aux_2=0.0683(w=0.5)
[2026-06-23 22:18:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 255): 13421.7 MiB
[2026-06-23 22:20:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2191, pooled_mean_dice=0.8109, per_class=['0.8109'], iou=0.6819, recall=0.9889, precision=0.6872, vol_sim=0.8200, mcc=0.8195, min_class_dice=0.8109, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9344, per_class_sd=['0.9344'], combined(w=0.50)=0.8726, balanced=0.8215
[2026-06-23 22:20:17] INFO segtask_v1.trainer.trainer: Epoch 256/400 | LR=2.94e-04 | loss=0.2690 | val_dice=0.8109 | best=0.8359 (ep212) | 05:23:34 | L_main=0.0583 L_aux_1=0.0610(w=0.5) L_aux_2=0.0693(w=0.5)
[2026-06-23 22:20:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 256): 13421.7 MiB
[2026-06-23 22:21:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2472, pooled_mean_dice=0.8030, per_class=['0.8030'], iou=0.6709, recall=0.9866, precision=0.6771, vol_sim=0.8139, mcc=0.8126, min_class_dice=0.8030, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9277, per_class_sd=['0.9277'], combined(w=0.50)=0.8654, balanced=0.8135
[2026-06-23 22:21:35] INFO segtask_v1.trainer.trainer: Epoch 257/400 | LR=2.91e-04 | loss=0.2723 | val_dice=0.8030 | best=0.8359 (ep212) | 05:24:52 | L_main=0.0589 L_aux_1=0.0615(w=0.5) L_aux_2=0.0695(w=0.5)
[2026-06-23 22:21:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 257): 13421.7 MiB
[2026-06-23 22:22:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2333, pooled_mean_dice=0.8099, per_class=['0.8099'], iou=0.6805, recall=0.9835, precision=0.6883, vol_sim=0.8235, mcc=0.8186, min_class_dice=0.8099, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9323, per_class_sd=['0.9323'], combined(w=0.50)=0.8711, balanced=0.8202
[2026-06-23 22:22:45] INFO segtask_v1.trainer.trainer: Epoch 258/400 | LR=2.87e-04 | loss=0.2684 | val_dice=0.8099 | best=0.8359 (ep212) | 05:26:02 | L_main=0.0578 L_aux_1=0.0605(w=0.5) L_aux_2=0.0688(w=0.5)
[2026-06-23 22:22:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 258): 13421.7 MiB
[2026-06-23 22:24:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2366, pooled_mean_dice=0.7973, per_class=['0.7973'], iou=0.6629, recall=0.9865, precision=0.6690, vol_sim=0.8082, mcc=0.8072, min_class_dice=0.7973, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9239, per_class_sd=['0.9239'], combined(w=0.50)=0.8606, balanced=0.8078
[2026-06-23 22:24:00] INFO segtask_v1.trainer.trainer: Epoch 259/400 | LR=2.84e-04 | loss=0.2755 | val_dice=0.7973 | best=0.8359 (ep212) | 05:27:17 | L_main=0.0588 L_aux_1=0.0616(w=0.5) L_aux_2=0.0700(w=0.5)
[2026-06-23 22:24:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 259): 13421.7 MiB
[2026-06-23 22:25:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2466, pooled_mean_dice=0.7898, per_class=['0.7898'], iou=0.6526, recall=0.9853, precision=0.6591, vol_sim=0.8016, mcc=0.8021, min_class_dice=0.7898, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9347, per_class_sd=['0.9347'], combined(w=0.50)=0.8623, balanced=0.8035
[2026-06-23 22:25:09] INFO segtask_v1.trainer.trainer: Epoch 260/400 | LR=2.80e-04 | loss=0.2733 | val_dice=0.7898 | best=0.8359 (ep212) | 05:28:26 | L_main=0.0599 L_aux_1=0.0631(w=0.5) L_aux_2=0.0710(w=0.5)
[2026-06-23 22:25:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 260): 13421.7 MiB
[2026-06-23 22:26:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2206, pooled_mean_dice=0.8136, per_class=['0.8136'], iou=0.6857, recall=0.9895, precision=0.6907, vol_sim=0.8222, mcc=0.8224, min_class_dice=0.8136, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9363, per_class_sd=['0.9363'], combined(w=0.50)=0.8749, balanced=0.8243
[2026-06-23 22:26:26] INFO segtask_v1.trainer.trainer: Epoch 261/400 | LR=2.76e-04 | loss=0.2696 | val_dice=0.8136 | best=0.8359 (ep212) | 05:29:42 | L_main=0.0585 L_aux_1=0.0612(w=0.5) L_aux_2=0.0687(w=0.5)
[2026-06-23 22:26:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 261): 13421.7 MiB
[2026-06-23 22:27:40] INFO segtask_v1.trainer.validation:   Val: loss=0.2358, pooled_mean_dice=0.8105, per_class=['0.8105'], iou=0.6814, recall=0.9890, precision=0.6865, vol_sim=0.8195, mcc=0.8192, min_class_dice=0.8105, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9351, per_class_sd=['0.9351'], combined(w=0.50)=0.8728, balanced=0.8213
[2026-06-23 22:27:40] INFO segtask_v1.trainer.trainer: Epoch 262/400 | LR=2.73e-04 | loss=0.2846 | val_dice=0.8105 | best=0.8359 (ep212) | 05:30:57 | L_main=0.0649 L_aux_1=0.0677(w=0.5) L_aux_2=0.0765(w=0.5)
[2026-06-23 22:27:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 262): 13421.7 MiB
[2026-06-23 22:28:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2469, pooled_mean_dice=0.7987, per_class=['0.7987'], iou=0.6649, recall=0.9855, precision=0.6715, vol_sim=0.8105, mcc=0.8087, min_class_dice=0.7987, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9316, per_class_sd=['0.9316'], combined(w=0.50)=0.8652, balanced=0.8105
[2026-06-23 22:28:59] INFO segtask_v1.trainer.trainer: Epoch 263/400 | LR=2.69e-04 | loss=0.2778 | val_dice=0.7987 | best=0.8359 (ep212) | 05:32:16 | L_main=0.0611 L_aux_1=0.0635(w=0.5) L_aux_2=0.0716(w=0.5)
[2026-06-23 22:28:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 263): 13421.7 MiB
[2026-06-23 22:30:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2169, pooled_mean_dice=0.8180, per_class=['0.8180'], iou=0.6921, recall=0.9876, precision=0.6982, vol_sim=0.8283, mcc=0.8251, min_class_dice=0.8180, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9329, per_class_sd=['0.9329'], combined(w=0.50)=0.8755, balanced=0.8273
[2026-06-23 22:30:17] INFO segtask_v1.trainer.trainer: Epoch 264/400 | LR=2.66e-04 | loss=0.2680 | val_dice=0.8180 | best=0.8359 (ep212) | 05:33:34 | L_main=0.0575 L_aux_1=0.0600(w=0.5) L_aux_2=0.0678(w=0.5)
[2026-06-23 22:30:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 264): 13421.7 MiB
[2026-06-23 22:31:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2095, pooled_mean_dice=0.8240, per_class=['0.8240'], iou=0.7007, recall=0.9865, precision=0.7075, vol_sim=0.8353, mcc=0.8308, min_class_dice=0.8240, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9346, per_class_sd=['0.9346'], combined(w=0.50)=0.8793, balanced=0.8328
[2026-06-23 22:31:30] INFO segtask_v1.trainer.trainer: Epoch 265/400 | LR=2.62e-04 | loss=0.2651 | val_dice=0.8240 | best=0.8359 (ep212) | 05:34:47 | L_main=0.0565 L_aux_1=0.0589(w=0.5) L_aux_2=0.0666(w=0.5)
[2026-06-23 22:31:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 265): 13421.7 MiB
[2026-06-23 22:32:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2120, pooled_mean_dice=0.8212, per_class=['0.8212'], iou=0.6966, recall=0.9850, precision=0.7041, vol_sim=0.8337, mcc=0.8278, min_class_dice=0.8212, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9330, per_class_sd=['0.9330'], combined(w=0.50)=0.8771, balanced=0.8300
[2026-06-23 22:32:44] INFO segtask_v1.trainer.trainer: Epoch 266/400 | LR=2.59e-04 | loss=0.2685 | val_dice=0.8212 | best=0.8359 (ep212) | 05:36:01 | L_main=0.0578 L_aux_1=0.0604(w=0.5) L_aux_2=0.0686(w=0.5)
[2026-06-23 22:32:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 266): 13421.7 MiB
[2026-06-23 22:34:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2267, pooled_mean_dice=0.8100, per_class=['0.8100'], iou=0.6807, recall=0.9853, precision=0.6877, vol_sim=0.8222, mcc=0.8176, min_class_dice=0.8100, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9337, per_class_sd=['0.9337'], combined(w=0.50)=0.8719, balanced=0.8206
[2026-06-23 22:34:03] INFO segtask_v1.trainer.trainer: Epoch 267/400 | LR=2.55e-04 | loss=0.2578 | val_dice=0.8100 | best=0.8359 (ep212) | 05:37:20 | L_main=0.0542 L_aux_1=0.0566(w=0.5) L_aux_2=0.0640(w=0.5)
[2026-06-23 22:34:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 267): 13421.7 MiB
[2026-06-23 22:35:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2291, pooled_mean_dice=0.8025, per_class=['0.8025'], iou=0.6701, recall=0.9867, precision=0.6762, vol_sim=0.8132, mcc=0.8124, min_class_dice=0.8025, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9344, per_class_sd=['0.9344'], combined(w=0.50)=0.8684, balanced=0.8143
[2026-06-23 22:35:19] INFO segtask_v1.trainer.trainer: Epoch 268/400 | LR=2.52e-04 | loss=0.2635 | val_dice=0.8025 | best=0.8359 (ep212) | 05:38:36 | L_main=0.0566 L_aux_1=0.0591(w=0.5) L_aux_2=0.0664(w=0.5)
[2026-06-23 22:35:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 268): 13421.7 MiB
[2026-06-23 22:36:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2303, pooled_mean_dice=0.8024, per_class=['0.8024'], iou=0.6700, recall=0.9862, precision=0.6763, vol_sim=0.8136, mcc=0.8122, min_class_dice=0.8024, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9324, per_class_sd=['0.9324'], combined(w=0.50)=0.8674, balanced=0.8138
[2026-06-23 22:36:36] INFO segtask_v1.trainer.trainer: Epoch 269/400 | LR=2.48e-04 | loss=0.2689 | val_dice=0.8024 | best=0.8359 (ep212) | 05:39:53 | L_main=0.0591 L_aux_1=0.0616(w=0.5) L_aux_2=0.0695(w=0.5)
[2026-06-23 22:36:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 269): 13421.7 MiB
[2026-06-23 22:37:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2295, pooled_mean_dice=0.8102, per_class=['0.8102'], iou=0.6810, recall=0.9876, precision=0.6868, vol_sim=0.8204, mcc=0.8188, min_class_dice=0.8102, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9321, per_class_sd=['0.9321'], combined(w=0.50)=0.8712, balanced=0.8205
[2026-06-23 22:37:46] INFO segtask_v1.trainer.trainer: Epoch 270/400 | LR=2.45e-04 | loss=0.2700 | val_dice=0.8102 | best=0.8359 (ep212) | 05:41:02 | L_main=0.0575 L_aux_1=0.0600(w=0.5) L_aux_2=0.0678(w=0.5)
[2026-06-23 22:37:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 270): 13421.7 MiB
[2026-06-23 22:39:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2190, pooled_mean_dice=0.8143, per_class=['0.8143'], iou=0.6867, recall=0.9869, precision=0.6930, vol_sim=0.8250, mcc=0.8222, min_class_dice=0.8143, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9373, per_class_sd=['0.9373'], combined(w=0.50)=0.8758, balanced=0.8250
[2026-06-23 22:39:03] INFO segtask_v1.trainer.trainer: Epoch 271/400 | LR=2.42e-04 | loss=0.2617 | val_dice=0.8143 | best=0.8359 (ep212) | 05:42:20 | L_main=0.0547 L_aux_1=0.0574(w=0.5) L_aux_2=0.0650(w=0.5)
[2026-06-23 22:39:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 271): 13421.7 MiB
[2026-06-23 22:40:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2133, pooled_mean_dice=0.8180, per_class=['0.8180'], iou=0.6921, recall=0.9865, precision=0.6987, vol_sim=0.8292, mcc=0.8248, min_class_dice=0.8180, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9377, per_class_sd=['0.9377'], combined(w=0.50)=0.8778, balanced=0.8282
[2026-06-23 22:40:16] INFO segtask_v1.trainer.trainer: Epoch 272/400 | LR=2.38e-04 | loss=0.2665 | val_dice=0.8180 | best=0.8359 (ep212) | 05:43:33 | L_main=0.0569 L_aux_1=0.0598(w=0.5) L_aux_2=0.0679(w=0.5)
[2026-06-23 22:40:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 272): 13421.7 MiB
[2026-06-23 22:41:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2162, pooled_mean_dice=0.8127, per_class=['0.8127'], iou=0.6844, recall=0.9868, precision=0.6907, vol_sim=0.8235, mcc=0.8204, min_class_dice=0.8127, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9304, per_class_sd=['0.9304'], combined(w=0.50)=0.8715, balanced=0.8222
[2026-06-23 22:41:31] INFO segtask_v1.trainer.trainer: Epoch 273/400 | LR=2.35e-04 | loss=0.2661 | val_dice=0.8127 | best=0.8359 (ep212) | 05:44:48 | L_main=0.0572 L_aux_1=0.0600(w=0.5) L_aux_2=0.0680(w=0.5)
[2026-06-23 22:41:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 273): 13421.7 MiB
[2026-06-23 22:42:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2270, pooled_mean_dice=0.8205, per_class=['0.8205'], iou=0.6957, recall=0.9889, precision=0.7011, vol_sim=0.8297, mcc=0.8280, min_class_dice=0.8205, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9381, per_class_sd=['0.9381'], combined(w=0.50)=0.8793, balanced=0.8305
[2026-06-23 22:42:49] INFO segtask_v1.trainer.trainer: Epoch 274/400 | LR=2.32e-04 | loss=0.2675 | val_dice=0.8205 | best=0.8359 (ep212) | 05:46:06 | L_main=0.0581 L_aux_1=0.0608(w=0.5) L_aux_2=0.0686(w=0.5)
[2026-06-23 22:42:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 274): 13421.7 MiB
[2026-06-23 22:44:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2107, pooled_mean_dice=0.8191, per_class=['0.8191'], iou=0.6937, recall=0.9853, precision=0.7009, vol_sim=0.8314, mcc=0.8264, min_class_dice=0.8191, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9417, per_class_sd=['0.9417'], combined(w=0.50)=0.8804, balanced=0.8300
[2026-06-23 22:44:08] INFO segtask_v1.trainer.trainer: Epoch 275/400 | LR=2.28e-04 | loss=0.2603 | val_dice=0.8191 | best=0.8359 (ep212) | 05:47:25 | L_main=0.0557 L_aux_1=0.0582(w=0.5) L_aux_2=0.0656(w=0.5)
[2026-06-23 22:44:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 275): 13421.7 MiB
[2026-06-23 22:45:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2132, pooled_mean_dice=0.8146, per_class=['0.8146'], iou=0.6872, recall=0.9843, precision=0.6948, vol_sim=0.8276, mcc=0.8225, min_class_dice=0.8146, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9365, per_class_sd=['0.9365'], combined(w=0.50)=0.8756, balanced=0.8251
[2026-06-23 22:45:22] INFO segtask_v1.trainer.trainer: Epoch 276/400 | LR=2.25e-04 | loss=0.2702 | val_dice=0.8146 | best=0.8359 (ep212) | 05:48:39 | L_main=0.0587 L_aux_1=0.0614(w=0.5) L_aux_2=0.0697(w=0.5)
[2026-06-23 22:45:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 276): 13421.7 MiB
[2026-06-23 22:46:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2276, pooled_mean_dice=0.8057, per_class=['0.8057'], iou=0.6746, recall=0.9866, precision=0.6808, vol_sim=0.8166, mcc=0.8144, min_class_dice=0.8057, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9324, per_class_sd=['0.9324'], combined(w=0.50)=0.8690, balanced=0.8166
[2026-06-23 22:46:37] INFO segtask_v1.trainer.trainer: Epoch 277/400 | LR=2.22e-04 | loss=0.2668 | val_dice=0.8057 | best=0.8359 (ep212) | 05:49:54 | L_main=0.0572 L_aux_1=0.0599(w=0.5) L_aux_2=0.0674(w=0.5)
[2026-06-23 22:46:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 277): 13421.7 MiB
[2026-06-23 22:47:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2256, pooled_mean_dice=0.8222, per_class=['0.8222'], iou=0.6981, recall=0.9868, precision=0.7047, vol_sim=0.8332, mcc=0.8290, min_class_dice=0.8222, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9422, per_class_sd=['0.9422'], combined(w=0.50)=0.8822, balanced=0.8327
[2026-06-23 22:47:51] INFO segtask_v1.trainer.trainer: Epoch 278/400 | LR=2.18e-04 | loss=0.2673 | val_dice=0.8222 | best=0.8359 (ep212) | 05:51:08 | L_main=0.0561 L_aux_1=0.0588(w=0.5) L_aux_2=0.0660(w=0.5)
[2026-06-23 22:47:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 278): 13421.7 MiB
[2026-06-23 22:49:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2310, pooled_mean_dice=0.8144, per_class=['0.8144'], iou=0.6869, recall=0.9877, precision=0.6928, vol_sim=0.8245, mcc=0.8224, min_class_dice=0.8144, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9336, per_class_sd=['0.9336'], combined(w=0.50)=0.8740, balanced=0.8243
[2026-06-23 22:49:06] INFO segtask_v1.trainer.trainer: Epoch 279/400 | LR=2.15e-04 | loss=0.2644 | val_dice=0.8144 | best=0.8359 (ep212) | 05:52:23 | L_main=0.0558 L_aux_1=0.0584(w=0.5) L_aux_2=0.0663(w=0.5)
[2026-06-23 22:49:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 279): 13421.7 MiB
[2026-06-23 22:50:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2145, pooled_mean_dice=0.8118, per_class=['0.8118'], iou=0.6832, recall=0.9880, precision=0.6889, vol_sim=0.8216, mcc=0.8202, min_class_dice=0.8118, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9338, per_class_sd=['0.9338'], combined(w=0.50)=0.8728, balanced=0.8222
[2026-06-23 22:50:15] INFO segtask_v1.trainer.trainer: Epoch 280/400 | LR=2.12e-04 | loss=0.2637 | val_dice=0.8118 | best=0.8359 (ep212) | 05:53:31 | L_main=0.0556 L_aux_1=0.0587(w=0.5) L_aux_2=0.0668(w=0.5)
[2026-06-23 22:50:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 280): 13421.7 MiB
[2026-06-23 22:51:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2083, pooled_mean_dice=0.8235, per_class=['0.8235'], iou=0.7000, recall=0.9885, precision=0.7057, vol_sim=0.8331, mcc=0.8305, min_class_dice=0.8235, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9395, per_class_sd=['0.9395'], combined(w=0.50)=0.8815, balanced=0.8333
[2026-06-23 22:51:33] INFO segtask_v1.trainer.trainer: Epoch 281/400 | LR=2.09e-04 | loss=0.2683 | val_dice=0.8235 | best=0.8359 (ep212) | 05:54:49 | L_main=0.0583 L_aux_1=0.0607(w=0.5) L_aux_2=0.0682(w=0.5)
[2026-06-23 22:51:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 281): 13421.7 MiB
[2026-06-23 22:52:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2276, pooled_mean_dice=0.8165, per_class=['0.8165'], iou=0.6899, recall=0.9839, precision=0.6978, vol_sim=0.8298, mcc=0.8247, min_class_dice=0.8165, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.9385, per_class_sd=['0.9385'], combined(w=0.50)=0.8775, balanced=0.8272
[2026-06-23 22:52:48] INFO segtask_v1.trainer.trainer: Epoch 282/400 | LR=2.05e-04 | loss=0.2652 | val_dice=0.8165 | best=0.8359 (ep212) | 05:56:05 | L_main=0.0564 L_aux_1=0.0589(w=0.5) L_aux_2=0.0668(w=0.5)
[2026-06-23 22:52:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 282): 13421.7 MiB
[2026-06-23 22:54:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2050, pooled_mean_dice=0.8215, per_class=['0.8215'], iou=0.6971, recall=0.9867, precision=0.7037, vol_sim=0.8326, mcc=0.8285, min_class_dice=0.8215, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9407, per_class_sd=['0.9407'], combined(w=0.50)=0.8811, balanced=0.8318
[2026-06-23 22:54:06] INFO segtask_v1.trainer.trainer: Epoch 283/400 | LR=2.02e-04 | loss=0.2705 | val_dice=0.8215 | best=0.8359 (ep212) | 05:57:23 | L_main=0.0575 L_aux_1=0.0600(w=0.5) L_aux_2=0.0681(w=0.5)
[2026-06-23 22:54:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 283): 13421.7 MiB
[2026-06-23 22:55:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2161, pooled_mean_dice=0.8181, per_class=['0.8181'], iou=0.6922, recall=0.9841, precision=0.7000, vol_sim=0.8313, mcc=0.8249, min_class_dice=0.8181, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9381, per_class_sd=['0.9381'], combined(w=0.50)=0.8781, balanced=0.8284
[2026-06-23 22:55:23] INFO segtask_v1.trainer.trainer: Epoch 284/400 | LR=1.99e-04 | loss=0.2585 | val_dice=0.8181 | best=0.8359 (ep212) | 05:58:40 | L_main=0.0535 L_aux_1=0.0561(w=0.5) L_aux_2=0.0632(w=0.5)
[2026-06-23 22:55:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 284): 13421.7 MiB
[2026-06-23 22:56:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2060, pooled_mean_dice=0.8261, per_class=['0.8261'], iou=0.7037, recall=0.9833, precision=0.7122, vol_sim=0.8401, mcc=0.8326, min_class_dice=0.8261, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9360, per_class_sd=['0.9360'], combined(w=0.50)=0.8811, balanced=0.8349
[2026-06-23 22:56:42] INFO segtask_v1.trainer.trainer: Epoch 285/400 | LR=1.96e-04 | loss=0.2600 | val_dice=0.8261 | best=0.8359 (ep212) | 05:59:59 | L_main=0.0544 L_aux_1=0.0573(w=0.5) L_aux_2=0.0652(w=0.5)
[2026-06-23 22:56:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 285): 13421.7 MiB
[2026-06-23 22:58:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2079, pooled_mean_dice=0.8220, per_class=['0.8220'], iou=0.6978, recall=0.9870, precision=0.7043, vol_sim=0.8329, mcc=0.8289, min_class_dice=0.8220, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9375, per_class_sd=['0.9375'], combined(w=0.50)=0.8797, balanced=0.8316
[2026-06-23 22:58:00] INFO segtask_v1.trainer.trainer: Epoch 286/400 | LR=1.93e-04 | loss=0.2685 | val_dice=0.8220 | best=0.8359 (ep212) | 06:01:17 | L_main=0.0578 L_aux_1=0.0600(w=0.5) L_aux_2=0.0681(w=0.5)
[2026-06-23 22:58:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 286): 13421.7 MiB
[2026-06-23 22:59:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2082, pooled_mean_dice=0.8190, per_class=['0.8190'], iou=0.6935, recall=0.9824, precision=0.7022, vol_sim=0.8337, mcc=0.8267, min_class_dice=0.8190, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9313, per_class_sd=['0.9313'], combined(w=0.50)=0.8752, balanced=0.8279
[2026-06-23 22:59:17] INFO segtask_v1.trainer.trainer: Epoch 287/400 | LR=1.90e-04 | loss=0.2621 | val_dice=0.8190 | best=0.8359 (ep212) | 06:02:33 | L_main=0.0557 L_aux_1=0.0581(w=0.5) L_aux_2=0.0656(w=0.5)
[2026-06-23 22:59:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 287): 13421.7 MiB
[2026-06-23 23:00:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2173, pooled_mean_dice=0.8137, per_class=['0.8137'], iou=0.6859, recall=0.9849, precision=0.6932, vol_sim=0.8262, mcc=0.8223, min_class_dice=0.8137, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9372, per_class_sd=['0.9372'], combined(w=0.50)=0.8755, balanced=0.8245
[2026-06-23 23:00:27] INFO segtask_v1.trainer.trainer: Epoch 288/400 | LR=1.86e-04 | loss=0.2626 | val_dice=0.8137 | best=0.8359 (ep212) | 06:03:44 | L_main=0.0562 L_aux_1=0.0586(w=0.5) L_aux_2=0.0662(w=0.5)
[2026-06-23 23:00:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 288): 13421.7 MiB
[2026-06-23 23:01:38] INFO segtask_v1.trainer.validation:   Val: loss=0.2449, pooled_mean_dice=0.7958, per_class=['0.7958'], iou=0.6609, recall=0.9844, precision=0.6679, vol_sim=0.8085, mcc=0.8062, min_class_dice=0.7958, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9346, per_class_sd=['0.9346'], combined(w=0.50)=0.8652, balanced=0.8086
[2026-06-23 23:01:38] INFO segtask_v1.trainer.trainer: Epoch 289/400 | LR=1.83e-04 | loss=0.2672 | val_dice=0.7958 | best=0.8359 (ep212) | 06:04:55 | L_main=0.0571 L_aux_1=0.0600(w=0.5) L_aux_2=0.0684(w=0.5)
[2026-06-23 23:01:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 289): 13421.7 MiB
[2026-06-23 23:02:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2321, pooled_mean_dice=0.8108, per_class=['0.8108'], iou=0.6818, recall=0.9825, precision=0.6902, vol_sim=0.8253, mcc=0.8189, min_class_dice=0.8108, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9369, per_class_sd=['0.9369'], combined(w=0.50)=0.8739, balanced=0.8219
[2026-06-23 23:02:59] INFO segtask_v1.trainer.trainer: Epoch 290/400 | LR=1.80e-04 | loss=0.2691 | val_dice=0.8108 | best=0.8359 (ep212) | 06:06:16 | L_main=0.0584 L_aux_1=0.0610(w=0.5) L_aux_2=0.0694(w=0.5)
[2026-06-23 23:02:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 290): 13421.7 MiB
[2026-06-23 23:04:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2395, pooled_mean_dice=0.8074, per_class=['0.8074'], iou=0.6770, recall=0.9857, precision=0.6837, vol_sim=0.8191, mcc=0.8160, min_class_dice=0.8074, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9343, per_class_sd=['0.9343'], combined(w=0.50)=0.8708, balanced=0.8184
[2026-06-23 23:04:15] INFO segtask_v1.trainer.trainer: Epoch 291/400 | LR=1.77e-04 | loss=0.2640 | val_dice=0.8074 | best=0.8359 (ep212) | 06:07:31 | L_main=0.0575 L_aux_1=0.0601(w=0.5) L_aux_2=0.0681(w=0.5)
[2026-06-23 23:04:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 291): 13421.7 MiB
[2026-06-23 23:05:29] INFO segtask_v1.trainer.validation:   Val: loss=0.2026, pooled_mean_dice=0.8260, per_class=['0.8260'], iou=0.7035, recall=0.9834, precision=0.7120, vol_sim=0.8399, mcc=0.8312, min_class_dice=0.8260, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9307, per_class_sd=['0.9307'], combined(w=0.50)=0.8783, balanced=0.8336
[2026-06-23 23:05:29] INFO segtask_v1.trainer.trainer: Epoch 292/400 | LR=1.74e-04 | loss=0.2639 | val_dice=0.8260 | best=0.8359 (ep212) | 06:08:45 | L_main=0.0569 L_aux_1=0.0593(w=0.5) L_aux_2=0.0671(w=0.5)
[2026-06-23 23:05:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 292): 13421.7 MiB
[2026-06-23 23:06:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2300, pooled_mean_dice=0.8140, per_class=['0.8140'], iou=0.6864, recall=0.9841, precision=0.6941, vol_sim=0.8272, mcc=0.8219, min_class_dice=0.8140, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9333, per_class_sd=['0.9333'], combined(w=0.50)=0.8737, balanced=0.8240
[2026-06-23 23:06:43] INFO segtask_v1.trainer.trainer: Epoch 293/400 | LR=1.71e-04 | loss=0.2645 | val_dice=0.8140 | best=0.8359 (ep212) | 06:10:00 | L_main=0.0556 L_aux_1=0.0579(w=0.5) L_aux_2=0.0652(w=0.5)
[2026-06-23 23:06:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 293): 13421.7 MiB
[2026-06-23 23:07:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2353, pooled_mean_dice=0.8111, per_class=['0.8111'], iou=0.6822, recall=0.9858, precision=0.6890, vol_sim=0.8228, mcc=0.8204, min_class_dice=0.8111, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9370, per_class_sd=['0.9370'], combined(w=0.50)=0.8741, balanced=0.8223
[2026-06-23 23:07:56] INFO segtask_v1.trainer.trainer: Epoch 294/400 | LR=1.68e-04 | loss=0.2630 | val_dice=0.8111 | best=0.8359 (ep212) | 06:11:12 | L_main=0.0560 L_aux_1=0.0585(w=0.5) L_aux_2=0.0663(w=0.5)
[2026-06-23 23:07:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 294): 13421.7 MiB
[2026-06-23 23:09:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2237, pooled_mean_dice=0.8105, per_class=['0.8105'], iou=0.6813, recall=0.9867, precision=0.6877, vol_sim=0.8214, mcc=0.8197, min_class_dice=0.8105, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9406, per_class_sd=['0.9406'], combined(w=0.50)=0.8755, balanced=0.8224
[2026-06-23 23:09:09] INFO segtask_v1.trainer.trainer: Epoch 295/400 | LR=1.65e-04 | loss=0.2614 | val_dice=0.8105 | best=0.8359 (ep212) | 06:12:26 | L_main=0.0548 L_aux_1=0.0573(w=0.5) L_aux_2=0.0647(w=0.5)
[2026-06-23 23:09:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 295): 13421.7 MiB
[2026-06-23 23:10:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2329, pooled_mean_dice=0.8027, per_class=['0.8027'], iou=0.6705, recall=0.9841, precision=0.6778, vol_sim=0.8157, mcc=0.8118, min_class_dice=0.8027, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9313, per_class_sd=['0.9313'], combined(w=0.50)=0.8670, balanced=0.8139
[2026-06-23 23:10:20] INFO segtask_v1.trainer.trainer: Epoch 296/400 | LR=1.62e-04 | loss=0.2664 | val_dice=0.8027 | best=0.8359 (ep212) | 06:13:37 | L_main=0.0574 L_aux_1=0.0605(w=0.5) L_aux_2=0.0687(w=0.5)
[2026-06-23 23:10:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 296): 13421.7 MiB
[2026-06-23 23:11:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2250, pooled_mean_dice=0.8223, per_class=['0.8223'], iou=0.6983, recall=0.9868, precision=0.7048, vol_sim=0.8333, mcc=0.8289, min_class_dice=0.8223, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9405, per_class_sd=['0.9405'], combined(w=0.50)=0.8814, balanced=0.8325
[2026-06-23 23:11:35] INFO segtask_v1.trainer.trainer: Epoch 297/400 | LR=1.59e-04 | loss=0.2678 | val_dice=0.8223 | best=0.8359 (ep212) | 06:14:52 | L_main=0.0569 L_aux_1=0.0594(w=0.5) L_aux_2=0.0670(w=0.5)
[2026-06-23 23:11:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 297): 13421.7 MiB
[2026-06-23 23:12:54] INFO segtask_v1.trainer.validation:   Val: loss=0.1926, pooled_mean_dice=0.8361, per_class=['0.8361'], iou=0.7183, recall=0.9830, precision=0.7273, vol_sim=0.8505, mcc=0.8408, min_class_dice=0.8361, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9444, per_class_sd=['0.9444'], combined(w=0.50)=0.8902, balanced=0.8450
[2026-06-23 23:12:58] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-23 23:12:58] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.8450 at epoch 298
[2026-06-23 23:12:58] INFO segtask_v1.trainer.trainer: Epoch 298/400 | LR=1.57e-04 | loss=0.2619 | val_dice=0.8361 | best=0.8450 (ep298) | 06:16:15 | L_main=0.0550 L_aux_1=0.0575(w=0.5) L_aux_2=0.0651(w=0.5)
[2026-06-23 23:12:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 298): 13421.7 MiB
[2026-06-23 23:14:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2190, pooled_mean_dice=0.8123, per_class=['0.8123'], iou=0.6839, recall=0.9857, precision=0.6908, vol_sim=0.8241, mcc=0.8201, min_class_dice=0.8123, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9336, per_class_sd=['0.9336'], combined(w=0.50)=0.8729, balanced=0.8225
[2026-06-23 23:14:15] INFO segtask_v1.trainer.trainer: Epoch 299/400 | LR=1.54e-04 | loss=0.2584 | val_dice=0.8123 | best=0.8450 (ep298) | 06:17:32 | L_main=0.0544 L_aux_1=0.0566(w=0.5) L_aux_2=0.0639(w=0.5)
[2026-06-23 23:14:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 299): 13421.7 MiB
[2026-06-23 23:15:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2404, pooled_mean_dice=0.8089, per_class=['0.8089'], iou=0.6791, recall=0.9864, precision=0.6855, vol_sim=0.8200, mcc=0.8181, min_class_dice=0.8089, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9372, per_class_sd=['0.9372'], combined(w=0.50)=0.8730, balanced=0.8204
[2026-06-23 23:15:28] INFO segtask_v1.trainer.trainer: Epoch 300/400 | LR=1.51e-04 | loss=0.2636 | val_dice=0.8089 | best=0.8450 (ep298) | 06:18:45 | L_main=0.0558 L_aux_1=0.0584(w=0.5) L_aux_2=0.0662(w=0.5)
[2026-06-23 23:15:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 300): 13421.7 MiB
[2026-06-23 23:16:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2506, pooled_mean_dice=0.7994, per_class=['0.7994'], iou=0.6659, recall=0.9838, precision=0.6732, vol_sim=0.8126, mcc=0.8085, min_class_dice=0.7994, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9272, per_class_sd=['0.9272'], combined(w=0.50)=0.8633, balanced=0.8102
[2026-06-23 23:16:48] INFO segtask_v1.trainer.trainer: Epoch 301/400 | LR=1.48e-04 | loss=0.2640 | val_dice=0.7994 | best=0.8450 (ep298) | 06:20:05 | L_main=0.0555 L_aux_1=0.0579(w=0.5) L_aux_2=0.0658(w=0.5)
[2026-06-23 23:16:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 301): 13421.7 MiB
[2026-06-23 23:18:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2454, pooled_mean_dice=0.8115, per_class=['0.8115'], iou=0.6828, recall=0.9851, precision=0.6899, vol_sim=0.8238, mcc=0.8196, min_class_dice=0.8115, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9357, per_class_sd=['0.9357'], combined(w=0.50)=0.8736, balanced=0.8223
[2026-06-23 23:18:08] INFO segtask_v1.trainer.trainer: Epoch 302/400 | LR=1.45e-04 | loss=0.2675 | val_dice=0.8115 | best=0.8450 (ep298) | 06:21:25 | L_main=0.0574 L_aux_1=0.0600(w=0.5) L_aux_2=0.0679(w=0.5)
[2026-06-23 23:18:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 302): 13421.7 MiB
[2026-06-23 23:19:24] INFO segtask_v1.trainer.validation:   Val: loss=0.2135, pooled_mean_dice=0.8155, per_class=['0.8155'], iou=0.6884, recall=0.9839, precision=0.6963, vol_sim=0.8288, mcc=0.8228, min_class_dice=0.8155, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9346, per_class_sd=['0.9346'], combined(w=0.50)=0.8750, balanced=0.8254
[2026-06-23 23:19:24] INFO segtask_v1.trainer.trainer: Epoch 303/400 | LR=1.42e-04 | loss=0.2652 | val_dice=0.8155 | best=0.8450 (ep298) | 06:22:41 | L_main=0.0573 L_aux_1=0.0599(w=0.5) L_aux_2=0.0679(w=0.5)
[2026-06-23 23:19:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 303): 13421.7 MiB
[2026-06-23 23:20:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2241, pooled_mean_dice=0.8056, per_class=['0.8056'], iou=0.6745, recall=0.9872, precision=0.6804, vol_sim=0.8161, mcc=0.8150, min_class_dice=0.8056, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9306, per_class_sd=['0.9306'], combined(w=0.50)=0.8681, balanced=0.8163
[2026-06-23 23:20:33] INFO segtask_v1.trainer.trainer: Epoch 304/400 | LR=1.40e-04 | loss=0.2638 | val_dice=0.8056 | best=0.8450 (ep298) | 06:23:50 | L_main=0.0558 L_aux_1=0.0583(w=0.5) L_aux_2=0.0664(w=0.5)
[2026-06-23 23:20:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 304): 13421.7 MiB
[2026-06-23 23:21:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2344, pooled_mean_dice=0.8041, per_class=['0.8041'], iou=0.6724, recall=0.9871, precision=0.6784, vol_sim=0.8147, mcc=0.8146, min_class_dice=0.8041, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9331, per_class_sd=['0.9331'], combined(w=0.50)=0.8686, balanced=0.8156
[2026-06-23 23:21:47] INFO segtask_v1.trainer.trainer: Epoch 305/400 | LR=1.37e-04 | loss=0.2663 | val_dice=0.8041 | best=0.8450 (ep298) | 06:25:04 | L_main=0.0566 L_aux_1=0.0594(w=0.5) L_aux_2=0.0675(w=0.5)
[2026-06-23 23:21:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 305): 13421.7 MiB
[2026-06-23 23:23:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2116, pooled_mean_dice=0.8163, per_class=['0.8163'], iou=0.6896, recall=0.9847, precision=0.6971, vol_sim=0.8290, mcc=0.8242, min_class_dice=0.8163, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9380, per_class_sd=['0.9380'], combined(w=0.50)=0.8771, balanced=0.8269
[2026-06-23 23:23:08] INFO segtask_v1.trainer.trainer: Epoch 306/400 | LR=1.34e-04 | loss=0.2636 | val_dice=0.8163 | best=0.8450 (ep298) | 06:26:25 | L_main=0.0564 L_aux_1=0.0593(w=0.5) L_aux_2=0.0673(w=0.5)
[2026-06-23 23:23:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 306): 13421.7 MiB
[2026-06-23 23:24:24] INFO segtask_v1.trainer.validation:   Val: loss=0.2347, pooled_mean_dice=0.8001, per_class=['0.8001'], iou=0.6668, recall=0.9830, precision=0.6746, vol_sim=0.8139, mcc=0.8107, min_class_dice=0.8001, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9435, per_class_sd=['0.9435'], combined(w=0.50)=0.8718, balanced=0.8140
[2026-06-23 23:24:24] INFO segtask_v1.trainer.trainer: Epoch 307/400 | LR=1.32e-04 | loss=0.2661 | val_dice=0.8001 | best=0.8450 (ep298) | 06:27:41 | L_main=0.0569 L_aux_1=0.0596(w=0.5) L_aux_2=0.0673(w=0.5)
[2026-06-23 23:24:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 307): 13421.7 MiB
[2026-06-23 23:25:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2321, pooled_mean_dice=0.7972, per_class=['0.7972'], iou=0.6627, recall=0.9892, precision=0.6676, vol_sim=0.8059, mcc=0.8086, min_class_dice=0.7972, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9317, per_class_sd=['0.9317'], combined(w=0.50)=0.8644, balanced=0.8093
[2026-06-23 23:25:44] INFO segtask_v1.trainer.trainer: Epoch 308/400 | LR=1.29e-04 | loss=0.2651 | val_dice=0.7972 | best=0.8450 (ep298) | 06:29:01 | L_main=0.0557 L_aux_1=0.0583(w=0.5) L_aux_2=0.0660(w=0.5)
[2026-06-23 23:25:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 308): 13421.7 MiB
[2026-06-23 23:27:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2145, pooled_mean_dice=0.8256, per_class=['0.8256'], iou=0.7030, recall=0.9885, precision=0.7088, vol_sim=0.8353, mcc=0.8328, min_class_dice=0.8256, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9416, per_class_sd=['0.9416'], combined(w=0.50)=0.8836, balanced=0.8356
[2026-06-23 23:27:04] INFO segtask_v1.trainer.trainer: Epoch 309/400 | LR=1.26e-04 | loss=0.2561 | val_dice=0.8256 | best=0.8450 (ep298) | 06:30:21 | L_main=0.0540 L_aux_1=0.0562(w=0.5) L_aux_2=0.0631(w=0.5)
[2026-06-23 23:27:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 309): 13421.7 MiB
[2026-06-23 23:28:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2163, pooled_mean_dice=0.8164, per_class=['0.8164'], iou=0.6898, recall=0.9865, precision=0.6963, vol_sim=0.8276, mcc=0.8233, min_class_dice=0.8164, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9346, per_class_sd=['0.9346'], combined(w=0.50)=0.8755, balanced=0.8262
[2026-06-23 23:28:23] INFO segtask_v1.trainer.trainer: Epoch 310/400 | LR=1.24e-04 | loss=0.2618 | val_dice=0.8164 | best=0.8450 (ep298) | 06:31:40 | L_main=0.0551 L_aux_1=0.0578(w=0.5) L_aux_2=0.0654(w=0.5)
[2026-06-23 23:28:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 310): 13421.7 MiB
[2026-06-23 23:29:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2220, pooled_mean_dice=0.8075, per_class=['0.8075'], iou=0.6771, recall=0.9877, precision=0.6828, vol_sim=0.8175, mcc=0.8167, min_class_dice=0.8075, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9309, per_class_sd=['0.9309'], combined(w=0.50)=0.8692, balanced=0.8179
[2026-06-23 23:29:41] INFO segtask_v1.trainer.trainer: Epoch 311/400 | LR=1.21e-04 | loss=0.2728 | val_dice=0.8075 | best=0.8450 (ep298) | 06:32:58 | L_main=0.0590 L_aux_1=0.0615(w=0.5) L_aux_2=0.0697(w=0.5)
[2026-06-23 23:29:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 311): 13421.7 MiB
[2026-06-23 23:30:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2390, pooled_mean_dice=0.8181, per_class=['0.8181'], iou=0.6922, recall=0.9814, precision=0.7014, vol_sim=0.8336, mcc=0.8256, min_class_dice=0.8181, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.9400, per_class_sd=['0.9400'], combined(w=0.50)=0.8790, balanced=0.8288
[2026-06-23 23:30:56] INFO segtask_v1.trainer.trainer: Epoch 312/400 | LR=1.18e-04 | loss=0.2597 | val_dice=0.8181 | best=0.8450 (ep298) | 06:34:13 | L_main=0.0553 L_aux_1=0.0576(w=0.5) L_aux_2=0.0648(w=0.5)
[2026-06-23 23:30:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 312): 13421.7 MiB
[2026-06-23 23:32:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2141, pooled_mean_dice=0.8185, per_class=['0.8185'], iou=0.6927, recall=0.9850, precision=0.7001, vol_sim=0.8309, mcc=0.8252, min_class_dice=0.8185, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9410, per_class_sd=['0.9410'], combined(w=0.50)=0.8798, balanced=0.8292
[2026-06-23 23:32:12] INFO segtask_v1.trainer.trainer: Epoch 313/400 | LR=1.16e-04 | loss=0.2572 | val_dice=0.8185 | best=0.8450 (ep298) | 06:35:28 | L_main=0.0536 L_aux_1=0.0560(w=0.5) L_aux_2=0.0635(w=0.5)
[2026-06-23 23:32:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 313): 13421.7 MiB
[2026-06-23 23:33:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2248, pooled_mean_dice=0.8201, per_class=['0.8201'], iou=0.6951, recall=0.9854, precision=0.7024, vol_sim=0.8323, mcc=0.8270, min_class_dice=0.8201, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9331, per_class_sd=['0.9331'], combined(w=0.50)=0.8766, balanced=0.8291
[2026-06-23 23:33:28] INFO segtask_v1.trainer.trainer: Epoch 314/400 | LR=1.13e-04 | loss=0.2629 | val_dice=0.8201 | best=0.8450 (ep298) | 06:36:45 | L_main=0.0555 L_aux_1=0.0581(w=0.5) L_aux_2=0.0658(w=0.5)
[2026-06-23 23:33:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 314): 13421.7 MiB
[2026-06-23 23:34:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2291, pooled_mean_dice=0.8075, per_class=['0.8075'], iou=0.6771, recall=0.9828, precision=0.6852, vol_sim=0.8216, mcc=0.8166, min_class_dice=0.8075, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9429, per_class_sd=['0.9429'], combined(w=0.50)=0.8752, balanced=0.8202
[2026-06-23 23:34:43] INFO segtask_v1.trainer.trainer: Epoch 315/400 | LR=1.11e-04 | loss=0.2575 | val_dice=0.8075 | best=0.8450 (ep298) | 06:38:00 | L_main=0.0544 L_aux_1=0.0569(w=0.5) L_aux_2=0.0641(w=0.5)
[2026-06-23 23:34:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 315): 13421.7 MiB
[2026-06-23 23:35:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2191, pooled_mean_dice=0.8134, per_class=['0.8134'], iou=0.6855, recall=0.9853, precision=0.6926, vol_sim=0.8255, mcc=0.8216, min_class_dice=0.8134, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9351, per_class_sd=['0.9351'], combined(w=0.50)=0.8742, balanced=0.8238
[2026-06-23 23:35:58] INFO segtask_v1.trainer.trainer: Epoch 316/400 | LR=1.08e-04 | loss=0.2584 | val_dice=0.8134 | best=0.8450 (ep298) | 06:39:14 | L_main=0.0548 L_aux_1=0.0574(w=0.5) L_aux_2=0.0653(w=0.5)
[2026-06-23 23:35:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 316): 13421.7 MiB
[2026-06-23 23:37:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2324, pooled_mean_dice=0.8047, per_class=['0.8047'], iou=0.6732, recall=0.9820, precision=0.6817, vol_sim=0.8195, mcc=0.8132, min_class_dice=0.8047, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9349, per_class_sd=['0.9349'], combined(w=0.50)=0.8698, balanced=0.8162
[2026-06-23 23:37:15] INFO segtask_v1.trainer.trainer: Epoch 317/400 | LR=1.06e-04 | loss=0.2595 | val_dice=0.8047 | best=0.8450 (ep298) | 06:40:31 | L_main=0.0549 L_aux_1=0.0575(w=0.5) L_aux_2=0.0650(w=0.5)
[2026-06-23 23:37:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 317): 13421.7 MiB
[2026-06-23 23:38:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2103, pooled_mean_dice=0.8259, per_class=['0.8259'], iou=0.7034, recall=0.9873, precision=0.7098, vol_sim=0.8365, mcc=0.8323, min_class_dice=0.8259, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9421, per_class_sd=['0.9421'], combined(w=0.50)=0.8840, balanced=0.8359
[2026-06-23 23:38:23] INFO segtask_v1.trainer.trainer: Epoch 318/400 | LR=1.04e-04 | loss=0.2607 | val_dice=0.8259 | best=0.8450 (ep298) | 06:41:40 | L_main=0.0544 L_aux_1=0.0569(w=0.5) L_aux_2=0.0644(w=0.5)
[2026-06-23 23:38:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 318): 13421.7 MiB
[2026-06-23 23:39:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2115, pooled_mean_dice=0.8184, per_class=['0.8184'], iou=0.6927, recall=0.9828, precision=0.7012, vol_sim=0.8327, mcc=0.8257, min_class_dice=0.8184, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9360, per_class_sd=['0.9360'], combined(w=0.50)=0.8772, balanced=0.8283
[2026-06-23 23:39:41] INFO segtask_v1.trainer.trainer: Epoch 319/400 | LR=1.01e-04 | loss=0.2579 | val_dice=0.8184 | best=0.8450 (ep298) | 06:42:58 | L_main=0.0547 L_aux_1=0.0573(w=0.5) L_aux_2=0.0650(w=0.5)
[2026-06-23 23:39:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 319): 13421.7 MiB
[2026-06-23 23:40:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2178, pooled_mean_dice=0.8249, per_class=['0.8249'], iou=0.7019, recall=0.9879, precision=0.7080, vol_sim=0.8350, mcc=0.8308, min_class_dice=0.8249, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9415, per_class_sd=['0.9415'], combined(w=0.50)=0.8832, balanced=0.8348
[2026-06-23 23:40:56] INFO segtask_v1.trainer.trainer: Epoch 320/400 | LR=9.87e-05 | loss=0.2570 | val_dice=0.8249 | best=0.8450 (ep298) | 06:44:12 | L_main=0.0544 L_aux_1=0.0568(w=0.5) L_aux_2=0.0640(w=0.5)
[2026-06-23 23:40:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 320): 13421.7 MiB
[2026-06-23 23:42:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2167, pooled_mean_dice=0.8216, per_class=['0.8216'], iou=0.6973, recall=0.9849, precision=0.7048, vol_sim=0.8343, mcc=0.8284, min_class_dice=0.8216, coverage=[83]/88 samples, pooled_mean_surface_dice@2px=0.9401, per_class_sd=['0.9401'], combined(w=0.50)=0.8808, balanced=0.8318
[2026-06-23 23:42:08] INFO segtask_v1.trainer.trainer: Epoch 321/400 | LR=9.64e-05 | loss=0.2595 | val_dice=0.8216 | best=0.8450 (ep298) | 06:45:25 | L_main=0.0550 L_aux_1=0.0570(w=0.5) L_aux_2=0.0638(w=0.5)
[2026-06-23 23:42:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 321): 13421.7 MiB
[2026-06-23 23:43:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2123, pooled_mean_dice=0.8199, per_class=['0.8199'], iou=0.6948, recall=0.9850, precision=0.7022, vol_sim=0.8324, mcc=0.8263, min_class_dice=0.8199, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.9416, per_class_sd=['0.9416'], combined(w=0.50)=0.8807, balanced=0.8306
[2026-06-23 23:43:22] INFO segtask_v1.trainer.trainer: Epoch 322/400 | LR=9.41e-05 | loss=0.2617 | val_dice=0.8199 | best=0.8450 (ep298) | 06:46:39 | L_main=0.0554 L_aux_1=0.0579(w=0.5) L_aux_2=0.0655(w=0.5)
[2026-06-23 23:43:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 322): 13421.7 MiB
[2026-06-23 23:44:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2029, pooled_mean_dice=0.8298, per_class=['0.8298'], iou=0.7091, recall=0.9848, precision=0.7169, vol_sim=0.8426, mcc=0.8355, min_class_dice=0.8298, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9440, per_class_sd=['0.9440'], combined(w=0.50)=0.8869, balanced=0.8396
[2026-06-23 23:44:42] INFO segtask_v1.trainer.trainer: Epoch 323/400 | LR=9.18e-05 | loss=0.2572 | val_dice=0.8298 | best=0.8450 (ep298) | 06:47:59 | L_main=0.0539 L_aux_1=0.0561(w=0.5) L_aux_2=0.0635(w=0.5)
[2026-06-23 23:44:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 323): 13421.7 MiB
[2026-06-23 23:45:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2209, pooled_mean_dice=0.8100, per_class=['0.8100'], iou=0.6807, recall=0.9862, precision=0.6873, vol_sim=0.8214, mcc=0.8190, min_class_dice=0.8100, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9378, per_class_sd=['0.9378'], combined(w=0.50)=0.8739, balanced=0.8215
[2026-06-23 23:45:57] INFO segtask_v1.trainer.trainer: Epoch 324/400 | LR=8.95e-05 | loss=0.2586 | val_dice=0.8100 | best=0.8450 (ep298) | 06:49:14 | L_main=0.0550 L_aux_1=0.0572(w=0.5) L_aux_2=0.0646(w=0.5)
[2026-06-23 23:45:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 324): 13421.7 MiB
[2026-06-23 23:47:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2213, pooled_mean_dice=0.8114, per_class=['0.8114'], iou=0.6827, recall=0.9880, precision=0.6884, vol_sim=0.8213, mcc=0.8201, min_class_dice=0.8114, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9392, per_class_sd=['0.9392'], combined(w=0.50)=0.8753, balanced=0.8229
[2026-06-23 23:47:10] INFO segtask_v1.trainer.trainer: Epoch 325/400 | LR=8.73e-05 | loss=0.2584 | val_dice=0.8114 | best=0.8450 (ep298) | 06:50:27 | L_main=0.0540 L_aux_1=0.0563(w=0.5) L_aux_2=0.0636(w=0.5)
[2026-06-23 23:47:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 325): 13421.7 MiB
[2026-06-23 23:48:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2125, pooled_mean_dice=0.8178, per_class=['0.8178'], iou=0.6917, recall=0.9846, precision=0.6993, vol_sim=0.8306, mcc=0.8249, min_class_dice=0.8178, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9396, per_class_sd=['0.9396'], combined(w=0.50)=0.8787, balanced=0.8284
[2026-06-23 23:48:21] INFO segtask_v1.trainer.trainer: Epoch 326/400 | LR=8.50e-05 | loss=0.2561 | val_dice=0.8178 | best=0.8450 (ep298) | 06:51:38 | L_main=0.0523 L_aux_1=0.0543(w=0.5) L_aux_2=0.0610(w=0.5)
[2026-06-23 23:48:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 326): 13421.7 MiB
[2026-06-23 23:49:39] INFO segtask_v1.trainer.validation:   Val: loss=0.2183, pooled_mean_dice=0.8200, per_class=['0.8200'], iou=0.6949, recall=0.9855, precision=0.7021, vol_sim=0.8321, mcc=0.8270, min_class_dice=0.8200, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9434, per_class_sd=['0.9434'], combined(w=0.50)=0.8817, balanced=0.8310
[2026-06-23 23:49:39] INFO segtask_v1.trainer.trainer: Epoch 327/400 | LR=8.29e-05 | loss=0.2533 | val_dice=0.8200 | best=0.8450 (ep298) | 06:52:56 | L_main=0.0523 L_aux_1=0.0546(w=0.5) L_aux_2=0.0617(w=0.5)
[2026-06-23 23:49:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 327): 13421.7 MiB
[2026-06-23 23:50:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2175, pooled_mean_dice=0.8151, per_class=['0.8151'], iou=0.6879, recall=0.9848, precision=0.6953, vol_sim=0.8277, mcc=0.8231, min_class_dice=0.8151, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9402, per_class_sd=['0.9402'], combined(w=0.50)=0.8777, balanced=0.8263
[2026-06-23 23:50:58] INFO segtask_v1.trainer.trainer: Epoch 328/400 | LR=8.07e-05 | loss=0.2582 | val_dice=0.8151 | best=0.8450 (ep298) | 06:54:15 | L_main=0.0539 L_aux_1=0.0564(w=0.5) L_aux_2=0.0640(w=0.5)
[2026-06-23 23:50:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 328): 13421.7 MiB
[2026-06-23 23:52:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2240, pooled_mean_dice=0.8172, per_class=['0.8172'], iou=0.6909, recall=0.9832, precision=0.6992, vol_sim=0.8312, mcc=0.8244, min_class_dice=0.8172, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9375, per_class_sd=['0.9375'], combined(w=0.50)=0.8773, balanced=0.8275
[2026-06-23 23:52:15] INFO segtask_v1.trainer.trainer: Epoch 329/400 | LR=7.85e-05 | loss=0.2627 | val_dice=0.8172 | best=0.8450 (ep298) | 06:55:31 | L_main=0.0561 L_aux_1=0.0586(w=0.5) L_aux_2=0.0663(w=0.5)
[2026-06-23 23:52:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 329): 13421.7 MiB
[2026-06-23 23:53:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2008, pooled_mean_dice=0.8261, per_class=['0.8261'], iou=0.7037, recall=0.9884, precision=0.7096, vol_sim=0.8358, mcc=0.8321, min_class_dice=0.8261, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9390, per_class_sd=['0.9390'], combined(w=0.50)=0.8825, balanced=0.8354
[2026-06-23 23:53:32] INFO segtask_v1.trainer.trainer: Epoch 330/400 | LR=7.64e-05 | loss=0.2548 | val_dice=0.8261 | best=0.8450 (ep298) | 06:56:49 | L_main=0.0534 L_aux_1=0.0559(w=0.5) L_aux_2=0.0633(w=0.5)
[2026-06-23 23:53:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 330): 13421.7 MiB
[2026-06-23 23:54:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2188, pooled_mean_dice=0.8237, per_class=['0.8237'], iou=0.7002, recall=0.9881, precision=0.7062, vol_sim=0.8336, mcc=0.8313, min_class_dice=0.8237, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9387, per_class_sd=['0.9387'], combined(w=0.50)=0.8812, balanced=0.8334
[2026-06-23 23:54:44] INFO segtask_v1.trainer.trainer: Epoch 331/400 | LR=7.43e-05 | loss=0.2600 | val_dice=0.8237 | best=0.8450 (ep298) | 06:58:01 | L_main=0.0543 L_aux_1=0.0566(w=0.5) L_aux_2=0.0641(w=0.5)
[2026-06-23 23:54:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 331): 13421.7 MiB
[2026-06-23 23:55:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2194, pooled_mean_dice=0.8189, per_class=['0.8189'], iou=0.6933, recall=0.9880, precision=0.6992, vol_sim=0.8288, mcc=0.8268, min_class_dice=0.8189, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9365, per_class_sd=['0.9365'], combined(w=0.50)=0.8777, balanced=0.8288
[2026-06-23 23:55:52] INFO segtask_v1.trainer.trainer: Epoch 332/400 | LR=7.23e-05 | loss=0.2518 | val_dice=0.8189 | best=0.8450 (ep298) | 06:59:09 | L_main=0.0529 L_aux_1=0.0552(w=0.5) L_aux_2=0.0623(w=0.5)
[2026-06-23 23:55:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 332): 13421.7 MiB
[2026-06-23 23:57:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2032, pooled_mean_dice=0.8291, per_class=['0.8291'], iou=0.7080, recall=0.9855, precision=0.7155, vol_sim=0.8413, mcc=0.8351, min_class_dice=0.8291, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9430, per_class_sd=['0.9430'], combined(w=0.50)=0.8860, balanced=0.8388
[2026-06-23 23:57:06] INFO segtask_v1.trainer.trainer: Epoch 333/400 | LR=7.03e-05 | loss=0.2655 | val_dice=0.8291 | best=0.8450 (ep298) | 07:00:22 | L_main=0.0568 L_aux_1=0.0593(w=0.5) L_aux_2=0.0674(w=0.5)
[2026-06-23 23:57:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 333): 13421.7 MiB
[2026-06-23 23:58:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2114, pooled_mean_dice=0.8195, per_class=['0.8195'], iou=0.6942, recall=0.9869, precision=0.7007, vol_sim=0.8304, mcc=0.8261, min_class_dice=0.8195, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9385, per_class_sd=['0.9385'], combined(w=0.50)=0.8790, balanced=0.8296
[2026-06-23 23:58:22] INFO segtask_v1.trainer.trainer: Epoch 334/400 | LR=6.83e-05 | loss=0.2546 | val_dice=0.8195 | best=0.8450 (ep298) | 07:01:39 | L_main=0.0519 L_aux_1=0.0540(w=0.5) L_aux_2=0.0608(w=0.5)
[2026-06-23 23:58:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 334): 13421.7 MiB
[2026-06-23 23:59:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2192, pooled_mean_dice=0.8172, per_class=['0.8172'], iou=0.6909, recall=0.9858, precision=0.6978, vol_sim=0.8290, mcc=0.8245, min_class_dice=0.8172, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9400, per_class_sd=['0.9400'], combined(w=0.50)=0.8786, balanced=0.8280
[2026-06-23 23:59:36] INFO segtask_v1.trainer.trainer: Epoch 335/400 | LR=6.63e-05 | loss=0.2573 | val_dice=0.8172 | best=0.8450 (ep298) | 07:02:53 | L_main=0.0535 L_aux_1=0.0556(w=0.5) L_aux_2=0.0626(w=0.5)
[2026-06-23 23:59:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 335): 13421.7 MiB
[2026-06-24 00:00:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2421, pooled_mean_dice=0.7923, per_class=['0.7923'], iou=0.6560, recall=0.9869, precision=0.6618, vol_sim=0.8028, mcc=0.8039, min_class_dice=0.7923, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9339, per_class_sd=['0.9339'], combined(w=0.50)=0.8631, balanced=0.8055
[2026-06-24 00:00:55] INFO segtask_v1.trainer.trainer: Epoch 336/400 | LR=6.43e-05 | loss=0.2596 | val_dice=0.7923 | best=0.8450 (ep298) | 07:04:12 | L_main=0.0548 L_aux_1=0.0571(w=0.5) L_aux_2=0.0643(w=0.5)
[2026-06-24 00:00:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 336): 13421.7 MiB
[2026-06-24 00:02:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2179, pooled_mean_dice=0.8205, per_class=['0.8205'], iou=0.6956, recall=0.9871, precision=0.7019, vol_sim=0.8312, mcc=0.8274, min_class_dice=0.8205, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9391, per_class_sd=['0.9391'], combined(w=0.50)=0.8798, balanced=0.8306
[2026-06-24 00:02:02] INFO segtask_v1.trainer.trainer: Epoch 337/400 | LR=6.24e-05 | loss=0.2530 | val_dice=0.8205 | best=0.8450 (ep298) | 07:05:19 | L_main=0.0527 L_aux_1=0.0549(w=0.5) L_aux_2=0.0621(w=0.5)
[2026-06-24 00:02:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 337): 13421.7 MiB
[2026-06-24 00:03:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2193, pooled_mean_dice=0.8178, per_class=['0.8178'], iou=0.6918, recall=0.9838, precision=0.6998, vol_sim=0.8313, mcc=0.8258, min_class_dice=0.8178, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9412, per_class_sd=['0.9412'], combined(w=0.50)=0.8795, balanced=0.8288
[2026-06-24 00:03:21] INFO segtask_v1.trainer.trainer: Epoch 338/400 | LR=6.05e-05 | loss=0.2511 | val_dice=0.8178 | best=0.8450 (ep298) | 07:06:37 | L_main=0.0518 L_aux_1=0.0541(w=0.5) L_aux_2=0.0612(w=0.5)
[2026-06-24 00:03:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 338): 13421.7 MiB
[2026-06-24 00:04:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2031, pooled_mean_dice=0.8264, per_class=['0.8264'], iou=0.7042, recall=0.9852, precision=0.7117, vol_sim=0.8388, mcc=0.8331, min_class_dice=0.8264, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9405, per_class_sd=['0.9405'], combined(w=0.50)=0.8835, balanced=0.8361
[2026-06-24 00:04:33] INFO segtask_v1.trainer.trainer: Epoch 339/400 | LR=5.86e-05 | loss=0.2549 | val_dice=0.8264 | best=0.8450 (ep298) | 07:07:50 | L_main=0.0529 L_aux_1=0.0547(w=0.5) L_aux_2=0.0615(w=0.5)
[2026-06-24 00:04:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 339): 13421.7 MiB
[2026-06-24 00:05:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2229, pooled_mean_dice=0.8079, per_class=['0.8079'], iou=0.6776, recall=0.9858, precision=0.6843, vol_sim=0.8195, mcc=0.8166, min_class_dice=0.8079, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9362, per_class_sd=['0.9362'], combined(w=0.50)=0.8720, balanced=0.8192
[2026-06-24 00:05:48] INFO segtask_v1.trainer.trainer: Epoch 340/400 | LR=5.68e-05 | loss=0.2607 | val_dice=0.8079 | best=0.8450 (ep298) | 07:09:05 | L_main=0.0548 L_aux_1=0.0571(w=0.5) L_aux_2=0.0645(w=0.5)
[2026-06-24 00:05:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 340): 13421.7 MiB
[2026-06-24 00:07:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2236, pooled_mean_dice=0.8111, per_class=['0.8111'], iou=0.6822, recall=0.9854, precision=0.6891, vol_sim=0.8231, mcc=0.8200, min_class_dice=0.8111, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9372, per_class_sd=['0.9372'], combined(w=0.50)=0.8741, balanced=0.8223
[2026-06-24 00:07:06] INFO segtask_v1.trainer.trainer: Epoch 341/400 | LR=5.50e-05 | loss=0.2613 | val_dice=0.8111 | best=0.8450 (ep298) | 07:10:22 | L_main=0.0554 L_aux_1=0.0576(w=0.5) L_aux_2=0.0652(w=0.5)
[2026-06-24 00:07:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 341): 13421.7 MiB
[2026-06-24 00:08:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2155, pooled_mean_dice=0.8172, per_class=['0.8172'], iou=0.6909, recall=0.9856, precision=0.6979, vol_sim=0.8292, mcc=0.8250, min_class_dice=0.8172, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9404, per_class_sd=['0.9404'], combined(w=0.50)=0.8788, balanced=0.8281
[2026-06-24 00:08:20] INFO segtask_v1.trainer.trainer: Epoch 342/400 | LR=5.32e-05 | loss=0.2607 | val_dice=0.8172 | best=0.8450 (ep298) | 07:11:37 | L_main=0.0545 L_aux_1=0.0570(w=0.5) L_aux_2=0.0645(w=0.5)
[2026-06-24 00:08:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 342): 13421.7 MiB
[2026-06-24 00:09:38] INFO segtask_v1.trainer.validation:   Val: loss=0.2153, pooled_mean_dice=0.8182, per_class=['0.8182'], iou=0.6923, recall=0.9854, precision=0.6995, vol_sim=0.8303, mcc=0.8259, min_class_dice=0.8182, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9404, per_class_sd=['0.9404'], combined(w=0.50)=0.8793, balanced=0.8289
[2026-06-24 00:09:38] INFO segtask_v1.trainer.trainer: Epoch 343/400 | LR=5.15e-05 | loss=0.2534 | val_dice=0.8182 | best=0.8450 (ep298) | 07:12:55 | L_main=0.0528 L_aux_1=0.0552(w=0.5) L_aux_2=0.0625(w=0.5)
[2026-06-24 00:09:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 343): 13421.7 MiB
[2026-06-24 00:10:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2131, pooled_mean_dice=0.8206, per_class=['0.8206'], iou=0.6957, recall=0.9864, precision=0.7025, vol_sim=0.8319, mcc=0.8265, min_class_dice=0.8206, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9359, per_class_sd=['0.9359'], combined(w=0.50)=0.8782, balanced=0.8300
[2026-06-24 00:10:46] INFO segtask_v1.trainer.trainer: Epoch 344/400 | LR=4.97e-05 | loss=0.2597 | val_dice=0.8206 | best=0.8450 (ep298) | 07:14:03 | L_main=0.0549 L_aux_1=0.0567(w=0.5) L_aux_2=0.0642(w=0.5)
[2026-06-24 00:10:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 344): 13421.7 MiB
[2026-06-24 00:11:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2701, pooled_mean_dice=0.7859, per_class=['0.7859'], iou=0.6474, recall=0.9843, precision=0.6541, vol_sim=0.7985, mcc=0.7992, min_class_dice=0.7859, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9362, per_class_sd=['0.9362'], combined(w=0.50)=0.8611, balanced=0.8005
[2026-06-24 00:11:56] INFO segtask_v1.trainer.trainer: Epoch 345/400 | LR=4.80e-05 | loss=0.2525 | val_dice=0.7859 | best=0.8450 (ep298) | 07:15:13 | L_main=0.0527 L_aux_1=0.0546(w=0.5) L_aux_2=0.0616(w=0.5)
[2026-06-24 00:11:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 345): 13421.7 MiB
[2026-06-24 00:13:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2279, pooled_mean_dice=0.8204, per_class=['0.8204'], iou=0.6955, recall=0.9869, precision=0.7019, vol_sim=0.8312, mcc=0.8278, min_class_dice=0.8204, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9396, per_class_sd=['0.9396'], combined(w=0.50)=0.8800, balanced=0.8307
[2026-06-24 00:13:10] INFO segtask_v1.trainer.trainer: Epoch 346/400 | LR=4.64e-05 | loss=0.2600 | val_dice=0.8204 | best=0.8450 (ep298) | 07:16:27 | L_main=0.0546 L_aux_1=0.0570(w=0.5) L_aux_2=0.0647(w=0.5)
[2026-06-24 00:13:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 346): 13421.7 MiB
[2026-06-24 00:14:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2130, pooled_mean_dice=0.8193, per_class=['0.8193'], iou=0.6939, recall=0.9852, precision=0.7012, vol_sim=0.8316, mcc=0.8269, min_class_dice=0.8193, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9366, per_class_sd=['0.9366'], combined(w=0.50)=0.8779, balanced=0.8292
[2026-06-24 00:14:22] INFO segtask_v1.trainer.trainer: Epoch 347/400 | LR=4.47e-05 | loss=0.2579 | val_dice=0.8193 | best=0.8450 (ep298) | 07:17:39 | L_main=0.0533 L_aux_1=0.0556(w=0.5) L_aux_2=0.0629(w=0.5)
[2026-06-24 00:14:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 347): 13421.7 MiB
[2026-06-24 00:15:40] INFO segtask_v1.trainer.validation:   Val: loss=0.1997, pooled_mean_dice=0.8302, per_class=['0.8302'], iou=0.7097, recall=0.9847, precision=0.7176, vol_sim=0.8431, mcc=0.8356, min_class_dice=0.8302, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9428, per_class_sd=['0.9428'], combined(w=0.50)=0.8865, balanced=0.8397
[2026-06-24 00:15:40] INFO segtask_v1.trainer.trainer: Epoch 348/400 | LR=4.31e-05 | loss=0.2580 | val_dice=0.8302 | best=0.8450 (ep298) | 07:18:57 | L_main=0.0548 L_aux_1=0.0573(w=0.5) L_aux_2=0.0650(w=0.5)
[2026-06-24 00:15:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 348): 13421.7 MiB
[2026-06-24 00:16:54] INFO segtask_v1.trainer.validation:   Val: loss=0.2095, pooled_mean_dice=0.8221, per_class=['0.8221'], iou=0.6980, recall=0.9850, precision=0.7055, vol_sim=0.8347, mcc=0.8295, min_class_dice=0.8221, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9450, per_class_sd=['0.9450'], combined(w=0.50)=0.8836, balanced=0.8333
[2026-06-24 00:16:54] INFO segtask_v1.trainer.trainer: Epoch 349/400 | LR=4.15e-05 | loss=0.2583 | val_dice=0.8221 | best=0.8450 (ep298) | 07:20:11 | L_main=0.0531 L_aux_1=0.0555(w=0.5) L_aux_2=0.0629(w=0.5)
[2026-06-24 00:16:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 349): 13421.7 MiB
[2026-06-24 00:18:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2085, pooled_mean_dice=0.8292, per_class=['0.8292'], iou=0.7082, recall=0.9866, precision=0.7151, vol_sim=0.8405, mcc=0.8355, min_class_dice=0.8292, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9432, per_class_sd=['0.9432'], combined(w=0.50)=0.8862, balanced=0.8390
[2026-06-24 00:18:08] INFO segtask_v1.trainer.trainer: Epoch 350/400 | LR=4.00e-05 | loss=0.2531 | val_dice=0.8292 | best=0.8450 (ep298) | 07:21:25 | L_main=0.0532 L_aux_1=0.0554(w=0.5) L_aux_2=0.0624(w=0.5)
[2026-06-24 00:18:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 350): 13421.7 MiB
[2026-06-24 00:19:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2116, pooled_mean_dice=0.8210, per_class=['0.8210'], iou=0.6964, recall=0.9829, precision=0.7049, vol_sim=0.8353, mcc=0.8279, min_class_dice=0.8210, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9443, per_class_sd=['0.9443'], combined(w=0.50)=0.8826, balanced=0.8321
[2026-06-24 00:19:22] INFO segtask_v1.trainer.trainer: Epoch 351/400 | LR=3.85e-05 | loss=0.2529 | val_dice=0.8210 | best=0.8450 (ep298) | 07:22:39 | L_main=0.0521 L_aux_1=0.0541(w=0.5) L_aux_2=0.0607(w=0.5)
[2026-06-24 00:19:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 351): 13421.7 MiB
[2026-06-24 00:20:39] INFO segtask_v1.trainer.validation:   Val: loss=0.1992, pooled_mean_dice=0.8298, per_class=['0.8298'], iou=0.7092, recall=0.9886, precision=0.7150, vol_sim=0.8394, mcc=0.8354, min_class_dice=0.8298, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9352, per_class_sd=['0.9352'], combined(w=0.50)=0.8825, balanced=0.8379
[2026-06-24 00:20:39] INFO segtask_v1.trainer.trainer: Epoch 352/400 | LR=3.70e-05 | loss=0.2616 | val_dice=0.8298 | best=0.8450 (ep298) | 07:23:55 | L_main=0.0547 L_aux_1=0.0568(w=0.5) L_aux_2=0.0643(w=0.5)
[2026-06-24 00:20:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 352): 13421.7 MiB
[2026-06-24 00:21:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2152, pooled_mean_dice=0.8217, per_class=['0.8217'], iou=0.6974, recall=0.9852, precision=0.7047, vol_sim=0.8340, mcc=0.8284, min_class_dice=0.8217, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.9385, per_class_sd=['0.9385'], combined(w=0.50)=0.8801, balanced=0.8316
[2026-06-24 00:21:58] INFO segtask_v1.trainer.trainer: Epoch 353/400 | LR=3.55e-05 | loss=0.2539 | val_dice=0.8217 | best=0.8450 (ep298) | 07:25:15 | L_main=0.0529 L_aux_1=0.0551(w=0.5) L_aux_2=0.0624(w=0.5)
[2026-06-24 00:21:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 353): 13421.7 MiB
[2026-06-24 00:23:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2304, pooled_mean_dice=0.8066, per_class=['0.8066'], iou=0.6759, recall=0.9826, precision=0.6841, vol_sim=0.8209, mcc=0.8153, min_class_dice=0.8066, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9417, per_class_sd=['0.9417'], combined(w=0.50)=0.8742, balanced=0.8192
[2026-06-24 00:23:08] INFO segtask_v1.trainer.trainer: Epoch 354/400 | LR=3.41e-05 | loss=0.2558 | val_dice=0.8066 | best=0.8450 (ep298) | 07:26:25 | L_main=0.0539 L_aux_1=0.0561(w=0.5) L_aux_2=0.0630(w=0.5)
[2026-06-24 00:23:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 354): 13421.7 MiB
[2026-06-24 00:24:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2118, pooled_mean_dice=0.8259, per_class=['0.8259'], iou=0.7035, recall=0.9829, precision=0.7122, vol_sim=0.8403, mcc=0.8320, min_class_dice=0.8259, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9462, per_class_sd=['0.9462'], combined(w=0.50)=0.8861, balanced=0.8367
[2026-06-24 00:24:15] INFO segtask_v1.trainer.trainer: Epoch 355/400 | LR=3.27e-05 | loss=0.2547 | val_dice=0.8259 | best=0.8450 (ep298) | 07:27:32 | L_main=0.0539 L_aux_1=0.0560(w=0.5) L_aux_2=0.0631(w=0.5)
[2026-06-24 00:24:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 355): 13421.7 MiB
[2026-06-24 00:25:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2084, pooled_mean_dice=0.8224, per_class=['0.8224'], iou=0.6983, recall=0.9850, precision=0.7059, vol_sim=0.8349, mcc=0.8294, min_class_dice=0.8224, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9381, per_class_sd=['0.9381'], combined(w=0.50)=0.8803, balanced=0.8321
[2026-06-24 00:25:30] INFO segtask_v1.trainer.trainer: Epoch 356/400 | LR=3.13e-05 | loss=0.2648 | val_dice=0.8224 | best=0.8450 (ep298) | 07:28:46 | L_main=0.0562 L_aux_1=0.0586(w=0.5) L_aux_2=0.0661(w=0.5)
[2026-06-24 00:25:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 356): 13421.7 MiB
[2026-06-24 00:26:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2324, pooled_mean_dice=0.8109, per_class=['0.8109'], iou=0.6820, recall=0.9847, precision=0.6893, vol_sim=0.8235, mcc=0.8185, min_class_dice=0.8109, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9346, per_class_sd=['0.9346'], combined(w=0.50)=0.8728, balanced=0.8215
[2026-06-24 00:26:44] INFO segtask_v1.trainer.trainer: Epoch 357/400 | LR=2.99e-05 | loss=0.2530 | val_dice=0.8109 | best=0.8450 (ep298) | 07:30:01 | L_main=0.0532 L_aux_1=0.0553(w=0.5) L_aux_2=0.0627(w=0.5)
[2026-06-24 00:26:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 357): 13421.7 MiB
[2026-06-24 00:28:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2318, pooled_mean_dice=0.8063, per_class=['0.8063'], iou=0.6754, recall=0.9892, precision=0.6805, vol_sim=0.8151, mcc=0.8166, min_class_dice=0.8063, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9364, per_class_sd=['0.9364'], combined(w=0.50)=0.8714, balanced=0.8181
[2026-06-24 00:28:00] INFO segtask_v1.trainer.trainer: Epoch 358/400 | LR=2.86e-05 | loss=0.2599 | val_dice=0.8063 | best=0.8450 (ep298) | 07:31:17 | L_main=0.0551 L_aux_1=0.0573(w=0.5) L_aux_2=0.0646(w=0.5)
[2026-06-24 00:28:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 358): 13421.7 MiB
[2026-06-24 00:29:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2283, pooled_mean_dice=0.8235, per_class=['0.8235'], iou=0.6999, recall=0.9853, precision=0.7073, vol_sim=0.8358, mcc=0.8302, min_class_dice=0.8235, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9428, per_class_sd=['0.9428'], combined(w=0.50)=0.8831, balanced=0.8340
[2026-06-24 00:29:08] INFO segtask_v1.trainer.trainer: Epoch 359/400 | LR=2.73e-05 | loss=0.2523 | val_dice=0.8235 | best=0.8450 (ep298) | 07:32:25 | L_main=0.0520 L_aux_1=0.0542(w=0.5) L_aux_2=0.0611(w=0.5)
[2026-06-24 00:29:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 359): 13421.7 MiB
[2026-06-24 00:30:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2170, pooled_mean_dice=0.8181, per_class=['0.8181'], iou=0.6922, recall=0.9875, precision=0.6983, vol_sim=0.8284, mcc=0.8254, min_class_dice=0.8181, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9359, per_class_sd=['0.9359'], combined(w=0.50)=0.8770, balanced=0.8280
[2026-06-24 00:30:16] INFO segtask_v1.trainer.trainer: Epoch 360/400 | LR=2.61e-05 | loss=0.2539 | val_dice=0.8181 | best=0.8450 (ep298) | 07:33:33 | L_main=0.0528 L_aux_1=0.0549(w=0.5) L_aux_2=0.0617(w=0.5)
[2026-06-24 00:30:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 360): 13421.7 MiB
[2026-06-24 00:31:25] INFO segtask_v1.trainer.validation:   Val: loss=0.2206, pooled_mean_dice=0.8201, per_class=['0.8201'], iou=0.6950, recall=0.9875, precision=0.7012, vol_sim=0.8304, mcc=0.8277, min_class_dice=0.8201, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.9464, per_class_sd=['0.9464'], combined(w=0.50)=0.8832, balanced=0.8317
[2026-06-24 00:31:25] INFO segtask_v1.trainer.trainer: Epoch 361/400 | LR=2.48e-05 | loss=0.2553 | val_dice=0.8201 | best=0.8450 (ep298) | 07:34:42 | L_main=0.0535 L_aux_1=0.0556(w=0.5) L_aux_2=0.0626(w=0.5)
[2026-06-24 00:31:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 361): 13421.7 MiB
[2026-06-24 00:32:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2311, pooled_mean_dice=0.8093, per_class=['0.8093'], iou=0.6797, recall=0.9830, precision=0.6878, vol_sim=0.8233, mcc=0.8178, min_class_dice=0.8093, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9396, per_class_sd=['0.9396'], combined(w=0.50)=0.8745, balanced=0.8212
[2026-06-24 00:32:33] INFO segtask_v1.trainer.trainer: Epoch 362/400 | LR=2.36e-05 | loss=0.2567 | val_dice=0.8093 | best=0.8450 (ep298) | 07:35:50 | L_main=0.0535 L_aux_1=0.0558(w=0.5) L_aux_2=0.0631(w=0.5)
[2026-06-24 00:32:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 362): 13421.7 MiB
[2026-06-24 00:33:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2201, pooled_mean_dice=0.8115, per_class=['0.8115'], iou=0.6827, recall=0.9852, precision=0.6898, vol_sim=0.8236, mcc=0.8199, min_class_dice=0.8115, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9359, per_class_sd=['0.9359'], combined(w=0.50)=0.8737, balanced=0.8223
[2026-06-24 00:33:49] INFO segtask_v1.trainer.trainer: Epoch 363/400 | LR=2.25e-05 | loss=0.2564 | val_dice=0.8115 | best=0.8450 (ep298) | 07:37:06 | L_main=0.0528 L_aux_1=0.0550(w=0.5) L_aux_2=0.0620(w=0.5)
[2026-06-24 00:33:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 363): 13421.7 MiB
[2026-06-24 00:35:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2485, pooled_mean_dice=0.7986, per_class=['0.7986'], iou=0.6647, recall=0.9861, precision=0.6710, vol_sim=0.8098, mcc=0.8090, min_class_dice=0.7986, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9380, per_class_sd=['0.9380'], combined(w=0.50)=0.8683, balanced=0.8116
[2026-06-24 00:35:06] INFO segtask_v1.trainer.trainer: Epoch 364/400 | LR=2.13e-05 | loss=0.2595 | val_dice=0.7986 | best=0.8450 (ep298) | 07:38:22 | L_main=0.0538 L_aux_1=0.0559(w=0.5) L_aux_2=0.0630(w=0.5)
[2026-06-24 00:35:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 364): 13421.7 MiB
[2026-06-24 00:36:25] INFO segtask_v1.trainer.validation:   Val: loss=0.2256, pooled_mean_dice=0.8118, per_class=['0.8118'], iou=0.6832, recall=0.9869, precision=0.6894, vol_sim=0.8225, mcc=0.8202, min_class_dice=0.8118, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9405, per_class_sd=['0.9405'], combined(w=0.50)=0.8761, balanced=0.8234
[2026-06-24 00:36:25] INFO segtask_v1.trainer.trainer: Epoch 365/400 | LR=2.02e-05 | loss=0.2549 | val_dice=0.8118 | best=0.8450 (ep298) | 07:39:42 | L_main=0.0524 L_aux_1=0.0549(w=0.5) L_aux_2=0.0623(w=0.5)
[2026-06-24 00:36:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 365): 13421.7 MiB
[2026-06-24 00:37:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2072, pooled_mean_dice=0.8354, per_class=['0.8354'], iou=0.7173, recall=0.9868, precision=0.7242, vol_sim=0.8465, mcc=0.8404, min_class_dice=0.8354, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9481, per_class_sd=['0.9481'], combined(w=0.50)=0.8917, balanced=0.8452
[2026-06-24 00:37:52] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-24 00:37:52] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.8452 at epoch 366
[2026-06-24 00:37:52] INFO segtask_v1.trainer.trainer: Epoch 366/400 | LR=1.92e-05 | loss=0.2525 | val_dice=0.8354 | best=0.8452 (ep366) | 07:41:09 | L_main=0.0530 L_aux_1=0.0554(w=0.5) L_aux_2=0.0627(w=0.5)
[2026-06-24 00:37:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 366): 13421.7 MiB
[2026-06-24 00:39:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2263, pooled_mean_dice=0.8020, per_class=['0.8020'], iou=0.6694, recall=0.9833, precision=0.6771, vol_sim=0.8156, mcc=0.8118, min_class_dice=0.8020, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.9342, per_class_sd=['0.9342'], combined(w=0.50)=0.8681, balanced=0.8138
[2026-06-24 00:39:01] INFO segtask_v1.trainer.trainer: Epoch 367/400 | LR=1.81e-05 | loss=0.2524 | val_dice=0.8020 | best=0.8452 (ep366) | 07:42:18 | L_main=0.0526 L_aux_1=0.0548(w=0.5) L_aux_2=0.0619(w=0.5)
[2026-06-24 00:39:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 367): 13421.7 MiB
[2026-06-24 00:40:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2164, pooled_mean_dice=0.8122, per_class=['0.8122'], iou=0.6837, recall=0.9866, precision=0.6901, vol_sim=0.8232, mcc=0.8215, min_class_dice=0.8122, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9388, per_class_sd=['0.9388'], combined(w=0.50)=0.8755, balanced=0.8235
[2026-06-24 00:40:23] INFO segtask_v1.trainer.trainer: Epoch 368/400 | LR=1.71e-05 | loss=0.2577 | val_dice=0.8122 | best=0.8452 (ep366) | 07:43:40 | L_main=0.0546 L_aux_1=0.0569(w=0.5) L_aux_2=0.0644(w=0.5)
[2026-06-24 00:40:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 368): 13421.7 MiB
[2026-06-24 00:41:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2071, pooled_mean_dice=0.8263, per_class=['0.8263'], iou=0.7040, recall=0.9846, precision=0.7119, vol_sim=0.8393, mcc=0.8330, min_class_dice=0.8263, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9441, per_class_sd=['0.9441'], combined(w=0.50)=0.8852, balanced=0.8367
[2026-06-24 00:41:37] INFO segtask_v1.trainer.trainer: Epoch 369/400 | LR=1.61e-05 | loss=0.2581 | val_dice=0.8263 | best=0.8452 (ep366) | 07:44:54 | L_main=0.0543 L_aux_1=0.0566(w=0.5) L_aux_2=0.0641(w=0.5)
[2026-06-24 00:41:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 369): 13421.7 MiB
[2026-06-24 00:42:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2153, pooled_mean_dice=0.8187, per_class=['0.8187'], iou=0.6931, recall=0.9843, precision=0.7008, vol_sim=0.8318, mcc=0.8256, min_class_dice=0.8187, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9401, per_class_sd=['0.9401'], combined(w=0.50)=0.8794, balanced=0.8293
[2026-06-24 00:42:55] INFO segtask_v1.trainer.trainer: Epoch 370/400 | LR=1.52e-05 | loss=0.2479 | val_dice=0.8187 | best=0.8452 (ep366) | 07:46:12 | L_main=0.0514 L_aux_1=0.0534(w=0.5) L_aux_2=0.0599(w=0.5)
[2026-06-24 00:42:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 370): 13421.7 MiB
[2026-06-24 00:44:11] INFO segtask_v1.trainer.validation:   Val: loss=0.2284, pooled_mean_dice=0.8169, per_class=['0.8169'], iou=0.6904, recall=0.9865, precision=0.6970, vol_sim=0.8280, mcc=0.8250, min_class_dice=0.8169, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9413, per_class_sd=['0.9413'], combined(w=0.50)=0.8791, balanced=0.8280
[2026-06-24 00:44:11] INFO segtask_v1.trainer.trainer: Epoch 371/400 | LR=1.42e-05 | loss=0.2557 | val_dice=0.8169 | best=0.8452 (ep366) | 07:47:28 | L_main=0.0529 L_aux_1=0.0552(w=0.5) L_aux_2=0.0623(w=0.5)
[2026-06-24 00:44:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 371): 13421.7 MiB
[2026-06-24 00:45:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2166, pooled_mean_dice=0.8159, per_class=['0.8159'], iou=0.6890, recall=0.9846, precision=0.6965, vol_sim=0.8286, mcc=0.8241, min_class_dice=0.8159, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9466, per_class_sd=['0.9466'], combined(w=0.50)=0.8813, balanced=0.8282
[2026-06-24 00:45:26] INFO segtask_v1.trainer.trainer: Epoch 372/400 | LR=1.33e-05 | loss=0.2557 | val_dice=0.8159 | best=0.8452 (ep366) | 07:48:42 | L_main=0.0535 L_aux_1=0.0558(w=0.5) L_aux_2=0.0632(w=0.5)
[2026-06-24 00:45:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 372): 13421.7 MiB
[2026-06-24 00:46:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2124, pooled_mean_dice=0.8141, per_class=['0.8141'], iou=0.6865, recall=0.9855, precision=0.6935, vol_sim=0.8261, mcc=0.8236, min_class_dice=0.8141, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.9472, per_class_sd=['0.9472'], combined(w=0.50)=0.8807, balanced=0.8269
[2026-06-24 00:46:37] INFO segtask_v1.trainer.trainer: Epoch 373/400 | LR=1.25e-05 | loss=0.2539 | val_dice=0.8141 | best=0.8452 (ep366) | 07:49:54 | L_main=0.0534 L_aux_1=0.0555(w=0.5) L_aux_2=0.0629(w=0.5)
[2026-06-24 00:46:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 373): 13421.7 MiB
[2026-06-24 00:47:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2123, pooled_mean_dice=0.8192, per_class=['0.8192'], iou=0.6938, recall=0.9842, precision=0.7016, vol_sim=0.8323, mcc=0.8261, min_class_dice=0.8192, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9411, per_class_sd=['0.9411'], combined(w=0.50)=0.8802, balanced=0.8299
[2026-06-24 00:47:45] INFO segtask_v1.trainer.trainer: Epoch 374/400 | LR=1.16e-05 | loss=0.2535 | val_dice=0.8192 | best=0.8452 (ep366) | 07:51:02 | L_main=0.0530 L_aux_1=0.0555(w=0.5) L_aux_2=0.0629(w=0.5)
[2026-06-24 00:47:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 374): 13421.7 MiB
[2026-06-24 00:48:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2146, pooled_mean_dice=0.8248, per_class=['0.8248'], iou=0.7019, recall=0.9852, precision=0.7094, vol_sim=0.8372, mcc=0.8313, min_class_dice=0.8248, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9458, per_class_sd=['0.9458'], combined(w=0.50)=0.8853, balanced=0.8357
[2026-06-24 00:48:57] INFO segtask_v1.trainer.trainer: Epoch 375/400 | LR=1.08e-05 | loss=0.2613 | val_dice=0.8248 | best=0.8452 (ep366) | 07:52:14 | L_main=0.0555 L_aux_1=0.0578(w=0.5) L_aux_2=0.0652(w=0.5)
[2026-06-24 00:48:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 375): 13421.7 MiB
[2026-06-24 00:50:10] INFO segtask_v1.trainer.validation:   Val: loss=0.1862, pooled_mean_dice=0.8417, per_class=['0.8417'], iou=0.7267, recall=0.9874, precision=0.7335, vol_sim=0.8525, mcc=0.8463, min_class_dice=0.8417, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9499, per_class_sd=['0.9499'], combined(w=0.50)=0.8958, balanced=0.8510
[2026-06-24 00:50:14] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_multirf/best_model.pth
[2026-06-24 00:50:14] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.8510 at epoch 376
[2026-06-24 00:50:14] INFO segtask_v1.trainer.trainer: Epoch 376/400 | LR=1.01e-05 | loss=0.2581 | val_dice=0.8417 | best=0.8510 (ep376) | 07:53:31 | L_main=0.0551 L_aux_1=0.0568(w=0.5) L_aux_2=0.0637(w=0.5)
[2026-06-24 00:50:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 376): 13421.7 MiB
[2026-06-24 00:51:29] INFO segtask_v1.trainer.validation:   Val: loss=0.2354, pooled_mean_dice=0.8118, per_class=['0.8118'], iou=0.6832, recall=0.9844, precision=0.6907, vol_sim=0.8246, mcc=0.8205, min_class_dice=0.8118, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9408, per_class_sd=['0.9408'], combined(w=0.50)=0.8763, balanced=0.8235
[2026-06-24 00:51:29] INFO segtask_v1.trainer.trainer: Epoch 377/400 | LR=9.33e-06 | loss=0.2589 | val_dice=0.8118 | best=0.8510 (ep376) | 07:54:46 | L_main=0.0545 L_aux_1=0.0566(w=0.5) L_aux_2=0.0638(w=0.5)
[2026-06-24 00:51:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 377): 13421.7 MiB
[2026-06-24 00:52:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2176, pooled_mean_dice=0.8179, per_class=['0.8179'], iou=0.6919, recall=0.9858, precision=0.6989, vol_sim=0.8297, mcc=0.8258, min_class_dice=0.8179, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.9410, per_class_sd=['0.9410'], combined(w=0.50)=0.8795, balanced=0.8288
[2026-06-24 00:52:42] INFO segtask_v1.trainer.trainer: Epoch 378/400 | LR=8.63e-06 | loss=0.2550 | val_dice=0.8179 | best=0.8510 (ep376) | 07:55:59 | L_main=0.0532 L_aux_1=0.0554(w=0.5) L_aux_2=0.0626(w=0.5)
[2026-06-24 00:52:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 378): 13421.7 MiB
[2026-06-24 00:53:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2360, pooled_mean_dice=0.8039, per_class=['0.8039'], iou=0.6721, recall=0.9823, precision=0.6804, vol_sim=0.8184, mcc=0.8140, min_class_dice=0.8039, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9436, per_class_sd=['0.9436'], combined(w=0.50)=0.8737, balanced=0.8173
[2026-06-24 00:53:55] INFO segtask_v1.trainer.trainer: Epoch 379/400 | LR=7.95e-06 | loss=0.2573 | val_dice=0.8039 | best=0.8510 (ep376) | 07:57:12 | L_main=0.0532 L_aux_1=0.0551(w=0.5) L_aux_2=0.0623(w=0.5)
[2026-06-24 00:53:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 379): 13421.7 MiB
[2026-06-24 00:55:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2147, pooled_mean_dice=0.8189, per_class=['0.8189'], iou=0.6934, recall=0.9872, precision=0.6997, vol_sim=0.8296, mcc=0.8260, min_class_dice=0.8189, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9339, per_class_sd=['0.9339'], combined(w=0.50)=0.8764, balanced=0.8283
[2026-06-24 00:55:08] INFO segtask_v1.trainer.trainer: Epoch 380/400 | LR=7.31e-06 | loss=0.2564 | val_dice=0.8189 | best=0.8510 (ep376) | 07:58:24 | L_main=0.0537 L_aux_1=0.0563(w=0.5) L_aux_2=0.0638(w=0.5)
[2026-06-24 00:55:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 380): 13421.7 MiB
[2026-06-24 00:56:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2237, pooled_mean_dice=0.8089, per_class=['0.8089'], iou=0.6791, recall=0.9858, precision=0.6858, vol_sim=0.8205, mcc=0.8178, min_class_dice=0.8089, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9374, per_class_sd=['0.9374'], combined(w=0.50)=0.8731, balanced=0.8204
[2026-06-24 00:56:21] INFO segtask_v1.trainer.trainer: Epoch 381/400 | LR=6.69e-06 | loss=0.2546 | val_dice=0.8089 | best=0.8510 (ep376) | 07:59:38 | L_main=0.0534 L_aux_1=0.0556(w=0.5) L_aux_2=0.0630(w=0.5)
[2026-06-24 00:56:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 381): 13421.7 MiB
[2026-06-24 00:57:40] INFO segtask_v1.trainer.validation:   Val: loss=0.2307, pooled_mean_dice=0.8054, per_class=['0.8054'], iou=0.6743, recall=0.9841, precision=0.6817, vol_sim=0.8184, mcc=0.8143, min_class_dice=0.8054, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.9368, per_class_sd=['0.9368'], combined(w=0.50)=0.8711, balanced=0.8173
[2026-06-24 00:57:40] INFO segtask_v1.trainer.trainer: Epoch 382/400 | LR=6.11e-06 | loss=0.2540 | val_dice=0.8054 | best=0.8510 (ep376) | 08:00:57 | L_main=0.0522 L_aux_1=0.0543(w=0.5) L_aux_2=0.0613(w=0.5)
[2026-06-24 00:57:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 382): 13421.7 MiB
[2026-06-24 00:58:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2258, pooled_mean_dice=0.8092, per_class=['0.8092'], iou=0.6795, recall=0.9843, precision=0.6869, vol_sim=0.8220, mcc=0.8175, min_class_dice=0.8092, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9385, per_class_sd=['0.9385'], combined(w=0.50)=0.8738, balanced=0.8208
[2026-06-24 00:58:57] INFO segtask_v1.trainer.trainer: Epoch 383/400 | LR=5.56e-06 | loss=0.2536 | val_dice=0.8092 | best=0.8510 (ep376) | 08:02:14 | L_main=0.0527 L_aux_1=0.0549(w=0.5) L_aux_2=0.0620(w=0.5)
[2026-06-24 00:58:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 383): 13421.7 MiB
[2026-06-24 01:00:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2420, pooled_mean_dice=0.7919, per_class=['0.7919'], iou=0.6554, recall=0.9820, precision=0.6634, vol_sim=0.8064, mcc=0.8036, min_class_dice=0.7919, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9342, per_class_sd=['0.9342'], combined(w=0.50)=0.8630, balanced=0.8052
[2026-06-24 01:00:10] INFO segtask_v1.trainer.trainer: Epoch 384/400 | LR=5.04e-06 | loss=0.2556 | val_dice=0.7919 | best=0.8510 (ep376) | 08:03:27 | L_main=0.0528 L_aux_1=0.0548(w=0.5) L_aux_2=0.0619(w=0.5)
[2026-06-24 01:00:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 384): 13421.7 MiB
[2026-06-24 01:01:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2417, pooled_mean_dice=0.8030, per_class=['0.8030'], iou=0.6708, recall=0.9864, precision=0.6771, vol_sim=0.8140, mcc=0.8131, min_class_dice=0.8030, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9386, per_class_sd=['0.9386'], combined(w=0.50)=0.8708, balanced=0.8156
[2026-06-24 01:01:18] INFO segtask_v1.trainer.trainer: Epoch 385/400 | LR=4.55e-06 | loss=0.2583 | val_dice=0.8030 | best=0.8510 (ep376) | 08:04:35 | L_main=0.0538 L_aux_1=0.0561(w=0.5) L_aux_2=0.0637(w=0.5)
[2026-06-24 01:01:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 385): 13421.7 MiB
[2026-06-24 01:02:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2484, pooled_mean_dice=0.8036, per_class=['0.8036'], iou=0.6716, recall=0.9844, precision=0.6788, vol_sim=0.8163, mcc=0.8140, min_class_dice=0.8036, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9456, per_class_sd=['0.9456'], combined(w=0.50)=0.8746, balanced=0.8174
[2026-06-24 01:02:30] INFO segtask_v1.trainer.trainer: Epoch 386/400 | LR=4.09e-06 | loss=0.2631 | val_dice=0.8036 | best=0.8510 (ep376) | 08:05:47 | L_main=0.0559 L_aux_1=0.0581(w=0.5) L_aux_2=0.0659(w=0.5)
[2026-06-24 01:02:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 386): 13421.7 MiB
[2026-06-24 01:03:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2182, pooled_mean_dice=0.8115, per_class=['0.8115'], iou=0.6828, recall=0.9840, precision=0.6905, vol_sim=0.8247, mcc=0.8196, min_class_dice=0.8115, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.9388, per_class_sd=['0.9388'], combined(w=0.50)=0.8752, balanced=0.8229
[2026-06-24 01:03:43] INFO segtask_v1.trainer.trainer: Epoch 387/400 | LR=3.67e-06 | loss=0.2599 | val_dice=0.8115 | best=0.8510 (ep376) | 08:07:00 | L_main=0.0544 L_aux_1=0.0568(w=0.5) L_aux_2=0.0641(w=0.5)
[2026-06-24 01:03:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 387): 13421.7 MiB
[2026-06-24 01:04:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2111, pooled_mean_dice=0.8271, per_class=['0.8271'], iou=0.7051, recall=0.9845, precision=0.7130, vol_sim=0.8401, mcc=0.8325, min_class_dice=0.8271, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.9425, per_class_sd=['0.9425'], combined(w=0.50)=0.8848, balanced=0.8369
[2026-06-24 01:04:57] INFO segtask_v1.trainer.trainer: Epoch 388/400 | LR=3.27e-06 | loss=0.2559 | val_dice=0.8271 | best=0.8510 (ep376) | 08:08:13 | L_main=0.0529 L_aux_1=0.0554(w=0.5) L_aux_2=0.0629(w=0.5)
[2026-06-24 01:04:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 388): 13421.7 MiB
[2026-06-24 01:06:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2140, pooled_mean_dice=0.8307, per_class=['0.8307'], iou=0.7105, recall=0.9842, precision=0.7187, vol_sim=0.8441, mcc=0.8366, min_class_dice=0.8307, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.9463, per_class_sd=['0.9463'], combined(w=0.50)=0.8885, balanced=0.8409
[2026-06-24 01:06:09] INFO segtask_v1.trainer.trainer: Epoch 389/400 | LR=2.91e-06 | loss=0.2532 | val_dice=0.8307 | best=0.8510 (ep376) | 08:09:26 | L_main=0.0522 L_aux_1=0.0544(w=0.5) L_aux_2=0.0613(w=0.5)
[2026-06-24 01:06:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 389): 13421.7 MiB
[2026-06-24 01:07:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2321, pooled_mean_dice=0.8058, per_class=['0.8058'], iou=0.6748, recall=0.9841, precision=0.6822, vol_sim=0.8189, mcc=0.8152, min_class_dice=0.8058, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.9425, per_class_sd=['0.9425'], combined(w=0.50)=0.8741, balanced=0.8187
[2026-06-24 01:07:26] INFO segtask_v1.trainer.trainer: Epoch 390/400 | LR=2.58e-06 | loss=0.2531 | val_dice=0.8058 | best=0.8510 (ep376) | 08:10:42 | L_main=0.0540 L_aux_1=0.0561(w=0.5) L_aux_2=0.0633(w=0.5)
[2026-06-24 01:07:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 390): 13421.7 MiB
[2026-06-24 01:08:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2073, pooled_mean_dice=0.8214, per_class=['0.8214'], iou=0.6969, recall=0.9871, precision=0.7034, vol_sim=0.8322, mcc=0.8289, min_class_dice=0.8214, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9425, per_class_sd=['0.9425'], combined(w=0.50)=0.8820, balanced=0.8322
[2026-06-24 01:08:46] INFO segtask_v1.trainer.trainer: Epoch 391/400 | LR=2.28e-06 | loss=0.2605 | val_dice=0.8214 | best=0.8510 (ep376) | 08:12:03 | L_main=0.0545 L_aux_1=0.0570(w=0.5) L_aux_2=0.0642(w=0.5)
[2026-06-24 01:08:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 391): 13421.7 MiB
[2026-06-24 01:09:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2003, pooled_mean_dice=0.8257, per_class=['0.8257'], iou=0.7031, recall=0.9840, precision=0.7112, vol_sim=0.8391, mcc=0.8316, min_class_dice=0.8257, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9441, per_class_sd=['0.9441'], combined(w=0.50)=0.8849, balanced=0.8360
[2026-06-24 01:09:58] INFO segtask_v1.trainer.trainer: Epoch 392/400 | LR=2.01e-06 | loss=0.2534 | val_dice=0.8257 | best=0.8510 (ep376) | 08:13:15 | L_main=0.0530 L_aux_1=0.0552(w=0.5) L_aux_2=0.0623(w=0.5)
[2026-06-24 01:09:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 392): 13421.7 MiB
[2026-06-24 01:11:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2223, pooled_mean_dice=0.8122, per_class=['0.8122'], iou=0.6838, recall=0.9844, precision=0.6913, vol_sim=0.8251, mcc=0.8207, min_class_dice=0.8122, coverage=[83]/88 samples, pooled_mean_surface_dice@2px=0.9392, per_class_sd=['0.9392'], combined(w=0.50)=0.8757, balanced=0.8236
[2026-06-24 01:11:15] INFO segtask_v1.trainer.trainer: Epoch 393/400 | LR=1.77e-06 | loss=0.2572 | val_dice=0.8122 | best=0.8510 (ep376) | 08:14:32 | L_main=0.0544 L_aux_1=0.0566(w=0.5) L_aux_2=0.0638(w=0.5)
[2026-06-24 01:11:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 393): 13421.7 MiB
[2026-06-24 01:12:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2275, pooled_mean_dice=0.8083, per_class=['0.8083'], iou=0.6783, recall=0.9860, precision=0.6849, vol_sim=0.8198, mcc=0.8172, min_class_dice=0.8083, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.9371, per_class_sd=['0.9371'], combined(w=0.50)=0.8727, balanced=0.8198
[2026-06-24 01:12:28] INFO segtask_v1.trainer.trainer: Epoch 394/400 | LR=1.57e-06 | loss=0.2548 | val_dice=0.8083 | best=0.8510 (ep376) | 08:15:44 | L_main=0.0530 L_aux_1=0.0550(w=0.5) L_aux_2=0.0621(w=0.5)
[2026-06-24 01:12:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 394): 13421.7 MiB
[2026-06-24 01:13:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2125, pooled_mean_dice=0.8186, per_class=['0.8186'], iou=0.6929, recall=0.9860, precision=0.6998, vol_sim=0.8303, mcc=0.8263, min_class_dice=0.8186, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.9396, per_class_sd=['0.9396'], combined(w=0.50)=0.8791, balanced=0.8292
[2026-06-24 01:13:46] INFO segtask_v1.trainer.trainer: Epoch 395/400 | LR=1.39e-06 | loss=0.2548 | val_dice=0.8186 | best=0.8510 (ep376) | 08:17:02 | L_main=0.0527 L_aux_1=0.0552(w=0.5) L_aux_2=0.0626(w=0.5)
[2026-06-24 01:13:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 395): 13421.7 MiB
[2026-06-24 01:15:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2224, pooled_mean_dice=0.8196, per_class=['0.8196'], iou=0.6944, recall=0.9874, precision=0.7006, vol_sim=0.8301, mcc=0.8278, min_class_dice=0.8196, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.9351, per_class_sd=['0.9351'], combined(w=0.50)=0.8774, balanced=0.8292
[2026-06-24 01:15:00] INFO segtask_v1.trainer.trainer: Epoch 396/400 | LR=1.25e-06 | loss=0.2511 | val_dice=0.8196 | best=0.8510 (ep376) | 08:18:17 | L_main=0.0513 L_aux_1=0.0533(w=0.5) L_aux_2=0.0601(w=0.5)
[2026-06-24 01:15:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 396): 13421.7 MiB
[2026-06-24 01:16:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2267, pooled_mean_dice=0.8067, per_class=['0.8067'], iou=0.6760, recall=0.9854, precision=0.6829, vol_sim=0.8187, mcc=0.8167, min_class_dice=0.8067, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9389, per_class_sd=['0.9389'], combined(w=0.50)=0.8728, balanced=0.8189
[2026-06-24 01:16:18] INFO segtask_v1.trainer.trainer: Epoch 397/400 | LR=1.14e-06 | loss=0.2622 | val_dice=0.8067 | best=0.8510 (ep376) | 08:19:35 | L_main=0.0553 L_aux_1=0.0576(w=0.5) L_aux_2=0.0651(w=0.5)
[2026-06-24 01:16:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 397): 13421.7 MiB
[2026-06-24 01:17:43] INFO segtask_v1.trainer.validation:   Val: loss=0.1975, pooled_mean_dice=0.8334, per_class=['0.8334'], iou=0.7143, recall=0.9846, precision=0.7224, vol_sim=0.8464, mcc=0.8391, min_class_dice=0.8334, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.9422, per_class_sd=['0.9422'], combined(w=0.50)=0.8878, balanced=0.8423
[2026-06-24 01:17:43] INFO segtask_v1.trainer.trainer: Epoch 398/400 | LR=1.06e-06 | loss=0.2588 | val_dice=0.8334 | best=0.8510 (ep376) | 08:20:59 | L_main=0.0545 L_aux_1=0.0569(w=0.5) L_aux_2=0.0647(w=0.5)
[2026-06-24 01:17:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 398): 13421.7 MiB
[2026-06-24 01:19:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2358, pooled_mean_dice=0.7942, per_class=['0.7942'], iou=0.6586, recall=0.9799, precision=0.6677, vol_sim=0.8105, mcc=0.8044, min_class_dice=0.7942, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.9393, per_class_sd=['0.9393'], combined(w=0.50)=0.8667, balanced=0.8080
[2026-06-24 01:19:00] INFO segtask_v1.trainer.trainer: Epoch 399/400 | LR=1.02e-06 | loss=0.2568 | val_dice=0.7942 | best=0.8510 (ep376) | 08:22:17 | L_main=0.0533 L_aux_1=0.0558(w=0.5) L_aux_2=0.0632(w=0.5)
[2026-06-24 01:19:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 399): 13421.7 MiB
[2026-06-24 01:20:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2046, pooled_mean_dice=0.8286, per_class=['0.8286'], iou=0.7074, recall=0.9877, precision=0.7137, vol_sim=0.8389, mcc=0.8345, min_class_dice=0.8286, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.9416, per_class_sd=['0.9416'], combined(w=0.50)=0.8851, balanced=0.8381
[2026-06-24 01:20:16] INFO segtask_v1.trainer.trainer: Epoch 400/400 | LR=1.00e-06 | loss=0.2508 | val_dice=0.8286 | best=0.8510 (ep376) | 08:23:33 | L_main=0.0524 L_aux_1=0.0542(w=0.5) L_aux_2=0.0605(w=0.5)
[2026-06-24 01:20:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 400): 13421.7 MiB
[2026-06-24 01:20:17] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-24 01:20:17] INFO segtask_v1.trainer.trainer: Training complete. Best mean_balanced=0.8510 at epoch 376. Time: 08:23:34
[2026-06-24 01:20:17] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-24 01:20:17] INFO __main__: Best metrics: {'val_loss': 0.18621582199226727, 'dice_class_0': 0.8417225480079651, 'iou_class_0': 0.7267020344734192, 'recall_class_0': 0.9873914122581482, 'precision_class_0': 0.7335087656974792, 'vol_sim_class_0': 0.8524709939956665, 'mcc_class_0': 0.8463380336761475, 'mean_dice': 0.8417225480079651, 'mean_iou': 0.7267020344734192, 'mean_recall': 0.9873914122581482, 'mean_precision': 0.7335087656974792, 'mean_vol_sim': 0.8524709939956665, 'mean_mcc': 0.8463380336761475, 'min_class_dice': 0.8417225480079651, 'min_class_iou': 0.7267020344734192, 'surface_dice_class_0': 0.9498580098152161, 'mean_surface_dice': 0.9498580098152161, 'mean_combined': 0.8957902789115906, 'mean_balanced': 0.8510497212409973}


3D:
[2026-06-23 17:05:06] INFO __main__: Config loaded from: configs/segtest1.yaml
[2026-06-23 17:05:06] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-23 17:05:07] INFO __main__: Device: cuda
[2026-06-23 17:05:07] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-23 17:05:07] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-23 17:05:07] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-23 17:05:07] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-23 17:05:22] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-23 17:05:22] INFO segtask_v1.data.specs: Using CUBIC patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], max_scale=2.00) — SINGLE max-FOV cube extraction; trainer crops+resizes per view before the 3D forward.
[2026-06-23 17:05:22] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 88 npz packages...
[2026-06-23 17:05:57] INFO segtask_v1.data.dataset: NPZ cubic index: 88 volumes, 4400000 fg voxels sampled
[2026-06-23 17:05:57] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 22 npz packages...
[2026-06-23 17:06:07] INFO segtask_v1.data.dataset: NPZ cubic index: 22 volumes, 1100000 fg voxels sampled
[2026-06-23 17:06:07] INFO segtask_v1.data.loader: DataLoader: batch_size=1, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-23 17:06:07] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=36, num_workers=16 => up to ~131.10 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-23 17:06:07] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-23 17:06:08] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=35.17M, dec=17.20M, total=54.80M, channels=[64, 64, 128, 256, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=3 (fg=1, res=3), stem=dual(stride=1, n_views=1, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=True, ds=True, aux_seg=False(n_aux_heads=0, mode=conv)
[2026-06-23 17:06:09] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Patch3DNativeMultiResPipeline (patch_mode=cubic, n_views=3)
[2026-06-23 17:06:09] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=distance, loss=smooth_l1, iter=5, weight=0.300)
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: Model params: 54.80M
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=209.0 + grad=209.0 + optim(AdamW,2x)=418.1 + ema=209.1 = 1045.2 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=427.8 MiB, reserved=446.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: Train batches: 704, Val batches: 88
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: AMP=False (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: Grad accum=8, Effective batch=8
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: Pipeline=Patch3DNativeMultiResPipeline | n_views=3, n_aux_views=0, num_res_groups=3, slab_depth=0 | fg_classes=1, Loss=dice_focal
[2026-06-23 17:06:09] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-23 17:06:39] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 14468.2 MiB (forward + backward + optimizer.step + EMA update; accum=8 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-23 17:20:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8598, pooled_mean_dice=0.0616, per_class=['0.0616'], iou=0.0318, recall=0.0511, precision=0.0774, vol_sim=0.7956, mcc=-0.0884, min_class_dice=0.0616, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.3497, per_class_sd=['0.3497'], combined(w=0.50)=0.2056, balanced=0.0758
[2026-06-23 17:20:38] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 17:20:38] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0758 at epoch 1
[2026-06-23 17:20:38] INFO segtask_v1.trainer.trainer: Epoch 1/400 | LR=2.60e-05 | loss=0.4364 | val_dice=0.0616 | best=0.0758 (ep1) | 00:14:29 | L_res_0=0.3646 L_res_1=0.4084 L_res_2=0.5016
[2026-06-23 17:20:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 14468.2 MiB
[2026-06-23 17:34:43] INFO segtask_v1.trainer.validation:   Val: loss=0.8341, pooled_mean_dice=0.1940, per_class=['0.1940'], iou=0.1074, recall=0.2115, precision=0.1792, vol_sim=0.9174, mcc=0.0060, min_class_dice=0.1940, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4039, per_class_sd=['0.4039'], combined(w=0.50)=0.2990, balanced=0.2114
[2026-06-23 17:34:48] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 17:34:48] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2114 at epoch 2
[2026-06-23 17:34:48] INFO segtask_v1.trainer.trainer: Epoch 2/400 | LR=5.10e-05 | loss=0.2848 | val_dice=0.1940 | best=0.2114 (ep2) | 00:28:39 | L_res_0=0.2329 L_res_1=0.2884 L_res_2=0.3932
[2026-06-23 17:34:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 11924.2 MiB
[2026-06-23 17:48:52] INFO segtask_v1.trainer.validation:   Val: loss=0.8235, pooled_mean_dice=0.3040, per_class=['0.3040'], iou=0.1792, recall=0.4394, precision=0.2323, vol_sim=0.6918, mcc=0.1219, min_class_dice=0.3040, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4030, per_class_sd=['0.4030'], combined(w=0.50)=0.3535, balanced=0.3046
[2026-06-23 17:48:57] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 17:48:57] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.3046 at epoch 3
[2026-06-23 17:48:57] INFO segtask_v1.trainer.trainer: Epoch 3/400 | LR=7.59e-05 | loss=0.2257 | val_dice=0.3040 | best=0.3046 (ep3) | 00:42:48 | L_res_0=0.1877 L_res_1=0.2431 L_res_2=0.3318
[2026-06-23 17:48:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 11924.2 MiB
[2026-06-23 18:02:59] INFO segtask_v1.trainer.validation:   Val: loss=0.7944, pooled_mean_dice=0.4127, per_class=['0.4127'], iou=0.2600, recall=0.6538, precision=0.3015, vol_sim=0.6312, mcc=0.2617, min_class_dice=0.4127, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.3917, per_class_sd=['0.3917'], combined(w=0.50)=0.4022, balanced=0.3843
[2026-06-23 18:03:04] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 18:03:04] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.3843 at epoch 4
[2026-06-23 18:03:04] INFO segtask_v1.trainer.trainer: Epoch 4/400 | LR=1.01e-04 | loss=0.1943 | val_dice=0.4127 | best=0.3843 (ep4) | 00:56:54 | L_res_0=0.1630 L_res_1=0.2150 L_res_2=0.2945
[2026-06-23 18:03:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 4): 11923.2 MiB
[2026-06-23 18:17:05] INFO segtask_v1.trainer.validation:   Val: loss=0.7790, pooled_mean_dice=0.4600, per_class=['0.4600'], iou=0.2987, recall=0.8276, precision=0.3185, vol_sim=0.5558, mcc=0.3563, min_class_dice=0.4600, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4007, per_class_sd=['0.4007'], combined(w=0.50)=0.4304, balanced=0.4214
[2026-06-23 18:17:09] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 18:17:09] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.4214 at epoch 5
[2026-06-23 18:17:09] INFO segtask_v1.trainer.trainer: Epoch 5/400 | LR=1.26e-04 | loss=0.1740 | val_dice=0.4600 | best=0.4214 (ep5) | 01:11:00 | L_res_0=0.1508 L_res_1=0.1931 L_res_2=0.2673
[2026-06-23 18:17:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 5): 11923.1 MiB
[2026-06-23 18:31:11] INFO segtask_v1.trainer.validation:   Val: loss=0.7530, pooled_mean_dice=0.4917, per_class=['0.4917'], iou=0.3260, recall=0.8930, precision=0.3392, vol_sim=0.5506, mcc=0.4076, min_class_dice=0.4917, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4100, per_class_sd=['0.4100'], combined(w=0.50)=0.4508, balanced=0.4463
[2026-06-23 18:31:16] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 18:31:16] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.4463 at epoch 6
[2026-06-23 18:31:16] INFO segtask_v1.trainer.trainer: Epoch 6/400 | LR=1.51e-04 | loss=0.1533 | val_dice=0.4917 | best=0.4463 (ep6) | 01:25:06 | L_res_0=0.1405 L_res_1=0.1721 L_res_2=0.2331
[2026-06-23 18:31:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 6): 11924.0 MiB
[2026-06-23 18:45:15] INFO segtask_v1.trainer.validation:   Val: loss=0.7230, pooled_mean_dice=0.5219, per_class=['0.5219'], iou=0.3531, recall=0.9164, precision=0.3648, vol_sim=0.5695, mcc=0.4489, min_class_dice=0.5219, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4510, per_class_sd=['0.4510'], combined(w=0.50)=0.4864, balanced=0.4793
[2026-06-23 18:45:20] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 18:45:20] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.4793 at epoch 7
[2026-06-23 18:45:20] INFO segtask_v1.trainer.trainer: Epoch 7/400 | LR=1.76e-04 | loss=0.1420 | val_dice=0.5219 | best=0.4793 (ep7) | 01:39:11 | L_res_0=0.1340 L_res_1=0.1594 L_res_2=0.2064
[2026-06-23 18:45:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 7): 11924.2 MiB
[2026-06-23 18:59:22] INFO segtask_v1.trainer.validation:   Val: loss=0.6946, pooled_mean_dice=0.5540, per_class=['0.5540'], iou=0.3832, recall=0.9386, precision=0.3930, vol_sim=0.5903, mcc=0.4847, min_class_dice=0.5540, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4903, per_class_sd=['0.4903'], combined(w=0.50)=0.5222, balanced=0.5127
[2026-06-23 18:59:27] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 18:59:27] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.5127 at epoch 8
[2026-06-23 18:59:27] INFO segtask_v1.trainer.trainer: Epoch 8/400 | LR=2.01e-04 | loss=0.1269 | val_dice=0.5540 | best=0.5127 (ep8) | 01:53:18 | L_res_0=0.1232 L_res_1=0.1432 L_res_2=0.1858
[2026-06-23 18:59:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 8): 11923.0 MiB
[2026-06-23 19:13:27] INFO segtask_v1.trainer.validation:   Val: loss=0.6738, pooled_mean_dice=0.5858, per_class=['0.5858'], iou=0.4143, recall=0.9594, precision=0.4217, vol_sim=0.6106, mcc=0.5287, min_class_dice=0.5858, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.5339, per_class_sd=['0.5339'], combined(w=0.50)=0.5599, balanced=0.5478
[2026-06-23 19:13:32] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 19:13:32] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.5478 at epoch 9
[2026-06-23 19:13:32] INFO segtask_v1.trainer.trainer: Epoch 9/400 | LR=2.26e-04 | loss=0.1305 | val_dice=0.5858 | best=0.5478 (ep9) | 02:07:22 | L_res_0=0.1275 L_res_1=0.1379 L_res_2=0.1720
[2026-06-23 19:13:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 9): 11924.2 MiB
[2026-06-23 19:27:30] INFO segtask_v1.trainer.validation:   Val: loss=0.6317, pooled_mean_dice=0.6228, per_class=['0.6228'], iou=0.4522, recall=0.9649, precision=0.4598, vol_sim=0.6455, mcc=0.5660, min_class_dice=0.6228, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.5599, per_class_sd=['0.5599'], combined(w=0.50)=0.5914, balanced=0.5814
[2026-06-23 19:27:35] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 19:27:35] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.5814 at epoch 10
[2026-06-23 19:27:35] INFO segtask_v1.trainer.trainer: Epoch 10/400 | LR=2.51e-04 | loss=0.1164 | val_dice=0.6228 | best=0.5814 (ep10) | 02:21:26 | L_res_0=0.1224 L_res_1=0.1236 L_res_2=0.1414
[2026-06-23 19:27:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 10): 11923.6 MiB
[2026-06-23 19:41:36] INFO segtask_v1.trainer.validation:   Val: loss=0.6395, pooled_mean_dice=0.6208, per_class=['0.6208'], iou=0.4501, recall=0.9649, precision=0.4576, vol_sim=0.6433, mcc=0.5707, min_class_dice=0.6208, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.5960, per_class_sd=['0.5960'], combined(w=0.50)=0.6084, balanced=0.5896
[2026-06-23 19:41:41] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 19:41:41] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.5896 at epoch 11
[2026-06-23 19:41:41] INFO segtask_v1.trainer.trainer: Epoch 11/400 | LR=2.76e-04 | loss=0.1061 | val_dice=0.6208 | best=0.5896 (ep11) | 02:35:32 | L_res_0=0.1143 L_res_1=0.1103 L_res_2=0.1234
[2026-06-23 19:41:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 11): 11922.9 MiB
[2026-06-23 19:55:45] INFO segtask_v1.trainer.validation:   Val: loss=0.6246, pooled_mean_dice=0.6324, per_class=['0.6324'], iou=0.4624, recall=0.9719, precision=0.4687, vol_sim=0.6507, mcc=0.5914, min_class_dice=0.6324, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6111, per_class_sd=['0.6111'], combined(w=0.50)=0.6217, balanced=0.6027
[2026-06-23 19:55:49] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 19:55:49] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6027 at epoch 12
[2026-06-23 19:55:49] INFO segtask_v1.trainer.trainer: Epoch 12/400 | LR=3.01e-04 | loss=0.0883 | val_dice=0.6324 | best=0.6027 (ep12) | 02:49:40 | L_res_0=0.0969 L_res_1=0.0900 L_res_2=0.1019
[2026-06-23 19:55:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 12): 11923.3 MiB
[2026-06-23 20:09:56] INFO segtask_v1.trainer.validation:   Val: loss=0.5832, pooled_mean_dice=0.6647, per_class=['0.6647'], iou=0.4978, recall=0.9794, precision=0.5031, vol_sim=0.6787, mcc=0.6212, min_class_dice=0.6647, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6372, per_class_sd=['0.6372'], combined(w=0.50)=0.6510, balanced=0.6333
[2026-06-23 20:10:01] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 20:10:01] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6333 at epoch 13
[2026-06-23 20:10:01] INFO segtask_v1.trainer.trainer: Epoch 13/400 | LR=3.26e-04 | loss=0.1049 | val_dice=0.6647 | best=0.6333 (ep13) | 03:03:51 | L_res_0=0.1129 L_res_1=0.1052 L_res_2=0.1176
[2026-06-23 20:10:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 13): 11923.6 MiB
[2026-06-23 20:24:00] INFO segtask_v1.trainer.validation:   Val: loss=0.5747, pooled_mean_dice=0.6709, per_class=['0.6709'], iou=0.5048, recall=0.9800, precision=0.5100, vol_sim=0.6846, mcc=0.6279, min_class_dice=0.6709, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6291, per_class_sd=['0.6291'], combined(w=0.50)=0.6500, balanced=0.6360
[2026-06-23 20:24:05] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 20:24:05] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6360 at epoch 14
[2026-06-23 20:24:05] INFO segtask_v1.trainer.trainer: Epoch 14/400 | LR=3.51e-04 | loss=0.1134 | val_dice=0.6709 | best=0.6360 (ep14) | 03:17:55 | L_res_0=0.1214 L_res_1=0.1144 L_res_2=0.1257
[2026-06-23 20:24:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 14): 11924.6 MiB
[2026-06-23 20:38:08] INFO segtask_v1.trainer.validation:   Val: loss=0.5388, pooled_mean_dice=0.6981, per_class=['0.6981'], iou=0.5362, recall=0.9783, precision=0.5427, vol_sim=0.7136, mcc=0.6540, min_class_dice=0.6981, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6475, per_class_sd=['0.6475'], combined(w=0.50)=0.6728, balanced=0.6611
[2026-06-23 20:38:13] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 20:38:13] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6611 at epoch 15
[2026-06-23 20:38:13] INFO segtask_v1.trainer.trainer: Epoch 15/400 | LR=3.76e-04 | loss=0.0809 | val_dice=0.6981 | best=0.6611 (ep15) | 03:32:04 | L_res_0=0.0864 L_res_1=0.0822 L_res_2=0.0916
[2026-06-23 20:38:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 15): 11923.7 MiB
[2026-06-23 20:52:16] INFO segtask_v1.trainer.validation:   Val: loss=0.5790, pooled_mean_dice=0.6712, per_class=['0.6712'], iou=0.5052, recall=0.9778, precision=0.5110, vol_sim=0.6864, mcc=0.6315, min_class_dice=0.6712, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6538, per_class_sd=['0.6538'], combined(w=0.50)=0.6625, balanced=0.6426
[2026-06-23 20:52:16] INFO segtask_v1.trainer.trainer: Epoch 16/400 | LR=4.01e-04 | loss=0.1000 | val_dice=0.6712 | best=0.6611 (ep15) | 03:46:07 | L_res_0=0.1087 L_res_1=0.1004 L_res_2=0.1098
[2026-06-23 20:52:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 16): 11923.1 MiB
[2026-06-23 21:06:16] INFO segtask_v1.trainer.validation:   Val: loss=0.5492, pooled_mean_dice=0.6867, per_class=['0.6867'], iou=0.5228, recall=0.9815, precision=0.5280, vol_sim=0.6996, mcc=0.6473, min_class_dice=0.6867, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6766, per_class_sd=['0.6766'], combined(w=0.50)=0.6816, balanced=0.6600
[2026-06-23 21:06:16] INFO segtask_v1.trainer.trainer: Epoch 17/400 | LR=4.26e-04 | loss=0.0874 | val_dice=0.6867 | best=0.6611 (ep15) | 04:00:07 | L_res_0=0.0952 L_res_1=0.0889 L_res_2=0.0965
[2026-06-23 21:06:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 17): 11923.8 MiB
[2026-06-23 21:20:19] INFO segtask_v1.trainer.validation:   Val: loss=0.5410, pooled_mean_dice=0.6823, per_class=['0.6823'], iou=0.5178, recall=0.9852, precision=0.5219, vol_sim=0.6926, mcc=0.6427, min_class_dice=0.6823, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6433, per_class_sd=['0.6433'], combined(w=0.50)=0.6628, balanced=0.6484
[2026-06-23 21:20:19] INFO segtask_v1.trainer.trainer: Epoch 18/400 | LR=4.51e-04 | loss=0.1135 | val_dice=0.6823 | best=0.6611 (ep15) | 04:14:10 | L_res_0=0.1239 L_res_1=0.1122 L_res_2=0.1199
[2026-06-23 21:20:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 18): 11923.5 MiB
[2026-06-23 21:34:21] INFO segtask_v1.trainer.validation:   Val: loss=0.5588, pooled_mean_dice=0.6805, per_class=['0.6805'], iou=0.5157, recall=0.9810, precision=0.5209, vol_sim=0.6937, mcc=0.6361, min_class_dice=0.6805, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6570, per_class_sd=['0.6570'], combined(w=0.50)=0.6687, balanced=0.6501
[2026-06-23 21:34:21] INFO segtask_v1.trainer.trainer: Epoch 19/400 | LR=4.76e-04 | loss=0.0922 | val_dice=0.6805 | best=0.6611 (ep15) | 04:28:12 | L_res_0=0.0997 L_res_1=0.0924 L_res_2=0.1018
[2026-06-23 21:34:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 19): 11924.3 MiB
[2026-06-23 21:48:21] INFO segtask_v1.trainer.validation:   Val: loss=0.5461, pooled_mean_dice=0.6910, per_class=['0.6910'], iou=0.5279, recall=0.9852, precision=0.5321, vol_sim=0.7013, mcc=0.6509, min_class_dice=0.6910, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6710, per_class_sd=['0.6710'], combined(w=0.50)=0.6810, balanced=0.6619
[2026-06-23 21:48:26] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 21:48:26] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6619 at epoch 20
[2026-06-23 21:48:26] INFO segtask_v1.trainer.trainer: Epoch 20/400 | LR=5.01e-04 | loss=0.0738 | val_dice=0.6910 | best=0.6619 (ep20) | 04:42:17 | L_res_0=0.0781 L_res_1=0.0753 L_res_2=0.0833
[2026-06-23 21:48:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 20): 11923.4 MiB
[2026-06-23 22:02:26] INFO segtask_v1.trainer.validation:   Val: loss=0.5320, pooled_mean_dice=0.6879, per_class=['0.6879'], iou=0.5243, recall=0.9848, precision=0.5286, vol_sim=0.6985, mcc=0.6502, min_class_dice=0.6879, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6701, per_class_sd=['0.6701'], combined(w=0.50)=0.6790, balanced=0.6595
[2026-06-23 22:02:26] INFO segtask_v1.trainer.trainer: Epoch 21/400 | LR=5.25e-04 | loss=0.0815 | val_dice=0.6879 | best=0.6619 (ep20) | 04:56:16 | L_res_0=0.0864 L_res_1=0.0827 L_res_2=0.0899
[2026-06-23 22:02:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 21): 11923.6 MiB
[2026-06-23 22:16:28] INFO segtask_v1.trainer.validation:   Val: loss=0.4961, pooled_mean_dice=0.7091, per_class=['0.7091'], iou=0.5493, recall=0.9852, precision=0.5539, vol_sim=0.7198, mcc=0.6695, min_class_dice=0.7091, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6977, per_class_sd=['0.6977'], combined(w=0.50)=0.7034, balanced=0.6824
[2026-06-23 22:16:34] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 22:16:34] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6824 at epoch 22
[2026-06-23 22:16:34] INFO segtask_v1.trainer.trainer: Epoch 22/400 | LR=5.50e-04 | loss=0.0904 | val_dice=0.7091 | best=0.6824 (ep22) | 05:10:24 | L_res_0=0.0973 L_res_1=0.0914 L_res_2=0.0997
[2026-06-23 22:16:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 22): 11923.9 MiB
[2026-06-23 22:30:31] INFO segtask_v1.trainer.validation:   Val: loss=0.5050, pooled_mean_dice=0.7075, per_class=['0.7075'], iou=0.5474, recall=0.9894, precision=0.5506, vol_sim=0.7151, mcc=0.6694, min_class_dice=0.7075, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6931, per_class_sd=['0.6931'], combined(w=0.50)=0.7003, balanced=0.6802
[2026-06-23 22:30:31] INFO segtask_v1.trainer.trainer: Epoch 23/400 | LR=5.75e-04 | loss=0.0829 | val_dice=0.7075 | best=0.6824 (ep22) | 05:24:22 | L_res_0=0.0867 L_res_1=0.0857 L_res_2=0.0933
[2026-06-23 22:30:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 23): 11923.3 MiB
[2026-06-23 22:44:33] INFO segtask_v1.trainer.validation:   Val: loss=0.4719, pooled_mean_dice=0.7227, per_class=['0.7227'], iou=0.5658, recall=0.9855, precision=0.5706, vol_sim=0.7334, mcc=0.6816, min_class_dice=0.7227, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7029, per_class_sd=['0.7029'], combined(w=0.50)=0.7128, balanced=0.6941
[2026-06-23 22:44:38] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 22:44:38] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6941 at epoch 24
[2026-06-23 22:44:38] INFO segtask_v1.trainer.trainer: Epoch 24/400 | LR=6.00e-04 | loss=0.0746 | val_dice=0.7227 | best=0.6941 (ep24) | 05:38:28 | L_res_0=0.0799 L_res_1=0.0768 L_res_2=0.0837
[2026-06-23 22:44:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 24): 11923.0 MiB
[2026-06-23 22:58:43] INFO segtask_v1.trainer.validation:   Val: loss=0.5183, pooled_mean_dice=0.6968, per_class=['0.6968'], iou=0.5347, recall=0.9831, precision=0.5396, vol_sim=0.7087, mcc=0.6617, min_class_dice=0.6968, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7006, per_class_sd=['0.7006'], combined(w=0.50)=0.6987, balanced=0.6738
[2026-06-23 22:58:43] INFO segtask_v1.trainer.trainer: Epoch 25/400 | LR=6.25e-04 | loss=0.0859 | val_dice=0.6968 | best=0.6941 (ep24) | 05:52:33 | L_res_0=0.0916 L_res_1=0.0882 L_res_2=0.0956
[2026-06-23 22:58:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 25): 11923.8 MiB
[2026-06-23 23:12:46] INFO segtask_v1.trainer.validation:   Val: loss=0.4590, pooled_mean_dice=0.7311, per_class=['0.7311'], iou=0.5762, recall=0.9851, precision=0.5812, vol_sim=0.7421, mcc=0.6953, min_class_dice=0.7311, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7369, per_class_sd=['0.7369'], combined(w=0.50)=0.7340, balanced=0.7092
[2026-06-23 23:12:50] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 23:12:50] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7092 at epoch 26
[2026-06-23 23:12:50] INFO segtask_v1.trainer.trainer: Epoch 26/400 | LR=6.50e-04 | loss=0.0665 | val_dice=0.7311 | best=0.7092 (ep26) | 06:06:41 | L_res_0=0.0704 L_res_1=0.0682 L_res_2=0.0748
[2026-06-23 23:12:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 26): 11924.9 MiB
[2026-06-23 23:26:50] INFO segtask_v1.trainer.validation:   Val: loss=0.4467, pooled_mean_dice=0.7373, per_class=['0.7373'], iou=0.5839, recall=0.9831, precision=0.5898, vol_sim=0.7500, mcc=0.6990, min_class_dice=0.7373, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7322, per_class_sd=['0.7322'], combined(w=0.50)=0.7347, balanced=0.7128
[2026-06-23 23:26:55] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 23:26:55] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7128 at epoch 27
[2026-06-23 23:26:55] INFO segtask_v1.trainer.trainer: Epoch 27/400 | LR=6.75e-04 | loss=0.0676 | val_dice=0.7373 | best=0.7128 (ep27) | 06:20:45 | L_res_0=0.0712 L_res_1=0.0685 L_res_2=0.0758
[2026-06-23 23:26:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 27): 11923.8 MiB
[2026-06-23 23:40:55] INFO segtask_v1.trainer.validation:   Val: loss=0.4567, pooled_mean_dice=0.7384, per_class=['0.7384'], iou=0.5853, recall=0.9878, precision=0.5896, vol_sim=0.7475, mcc=0.7053, min_class_dice=0.7384, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7520, per_class_sd=['0.7520'], combined(w=0.50)=0.7452, balanced=0.7187
[2026-06-23 23:41:00] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-23 23:41:00] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7187 at epoch 28
[2026-06-23 23:41:00] INFO segtask_v1.trainer.trainer: Epoch 28/400 | LR=7.00e-04 | loss=0.0610 | val_dice=0.7384 | best=0.7187 (ep28) | 06:34:50 | L_res_0=0.0624 L_res_1=0.0626 L_res_2=0.0694
[2026-06-23 23:41:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 28): 11925.9 MiB
[2026-06-23 23:55:00] INFO segtask_v1.trainer.validation:   Val: loss=0.4449, pooled_mean_dice=0.7397, per_class=['0.7397'], iou=0.5869, recall=0.9889, precision=0.5908, vol_sim=0.7480, mcc=0.7063, min_class_dice=0.7397, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7456, per_class_sd=['0.7456'], combined(w=0.50)=0.7426, balanced=0.7183
[2026-06-23 23:55:00] INFO segtask_v1.trainer.trainer: Epoch 29/400 | LR=7.25e-04 | loss=0.0754 | val_dice=0.7397 | best=0.7187 (ep28) | 06:48:51 | L_res_0=0.0802 L_res_1=0.0760 L_res_2=0.0821
[2026-06-23 23:55:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 29): 11922.8 MiB
[2026-06-24 00:09:02] INFO segtask_v1.trainer.validation:   Val: loss=0.4532, pooled_mean_dice=0.7336, per_class=['0.7336'], iou=0.5793, recall=0.9920, precision=0.5820, vol_sim=0.7395, mcc=0.6971, min_class_dice=0.7336, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7291, per_class_sd=['0.7291'], combined(w=0.50)=0.7313, balanced=0.7093
[2026-06-24 00:09:02] INFO segtask_v1.trainer.trainer: Epoch 30/400 | LR=7.50e-04 | loss=0.0730 | val_dice=0.7336 | best=0.7187 (ep28) | 07:02:52 | L_res_0=0.0756 L_res_1=0.0746 L_res_2=0.0819
[2026-06-24 00:09:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 30): 11923.5 MiB
[2026-06-24 00:23:00] INFO segtask_v1.trainer.validation:   Val: loss=0.4549, pooled_mean_dice=0.7340, per_class=['0.7340'], iou=0.5797, recall=0.9918, precision=0.5825, vol_sim=0.7400, mcc=0.7012, min_class_dice=0.7340, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7410, per_class_sd=['0.7410'], combined(w=0.50)=0.7375, balanced=0.7127
[2026-06-24 00:23:00] INFO segtask_v1.trainer.trainer: Epoch 31/400 | LR=7.75e-04 | loss=0.0589 | val_dice=0.7340 | best=0.7187 (ep28) | 07:16:51 | L_res_0=0.0578 L_res_1=0.0610 L_res_2=0.0674
[2026-06-24 00:23:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 31): 11924.2 MiB
[2026-06-24 00:37:04] INFO segtask_v1.trainer.validation:   Val: loss=0.4376, pooled_mean_dice=0.7427, per_class=['0.7427'], iou=0.5907, recall=0.9888, precision=0.5946, vol_sim=0.7511, mcc=0.7129, min_class_dice=0.7427, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7465, per_class_sd=['0.7465'], combined(w=0.50)=0.7446, balanced=0.7211
[2026-06-24 00:37:09] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-24 00:37:09] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7211 at epoch 32
[2026-06-24 00:37:09] INFO segtask_v1.trainer.trainer: Epoch 32/400 | LR=8.00e-04 | loss=0.0566 | val_dice=0.7427 | best=0.7211 (ep32) | 07:30:59 | L_res_0=0.0570 L_res_1=0.0573 L_res_2=0.0642
[2026-06-24 00:37:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 32): 11923.5 MiB
[2026-06-24 00:51:14] INFO segtask_v1.trainer.validation:   Val: loss=0.3802, pooled_mean_dice=0.7809, per_class=['0.7809'], iou=0.6405, recall=0.9903, precision=0.6446, vol_sim=0.7886, mcc=0.7449, min_class_dice=0.7809, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7858, per_class_sd=['0.7858'], combined(w=0.50)=0.7834, balanced=0.7604
[2026-06-24 00:51:19] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-24 00:51:19] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7604 at epoch 33
[2026-06-24 00:51:19] INFO segtask_v1.trainer.trainer: Epoch 33/400 | LR=8.25e-04 | loss=0.0531 | val_dice=0.7809 | best=0.7604 (ep33) | 07:45:09 | L_res_0=0.0510 L_res_1=0.0548 L_res_2=0.0613
[2026-06-24 00:51:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 33): 11924.8 MiB
[2026-06-24 01:05:18] INFO segtask_v1.trainer.validation:   Val: loss=0.4092, pooled_mean_dice=0.7605, per_class=['0.7605'], iou=0.6136, recall=0.9926, precision=0.6164, vol_sim=0.7662, mcc=0.7290, min_class_dice=0.7605, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7662, per_class_sd=['0.7662'], combined(w=0.50)=0.7634, balanced=0.7399
[2026-06-24 01:05:18] INFO segtask_v1.trainer.trainer: Epoch 34/400 | LR=8.50e-04 | loss=0.0627 | val_dice=0.7605 | best=0.7604 (ep33) | 07:59:08 | L_res_0=0.0643 L_res_1=0.0637 L_res_2=0.0696
[2026-06-24 01:05:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 34): 11922.9 MiB
[2026-06-24 01:19:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3698, pooled_mean_dice=0.7830, per_class=['0.7830'], iou=0.6434, recall=0.9909, precision=0.6472, vol_sim=0.7902, mcc=0.7472, min_class_dice=0.7830, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7824, per_class_sd=['0.7824'], combined(w=0.50)=0.7827, balanced=0.7613
[2026-06-24 01:19:25] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-24 01:19:25] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7613 at epoch 35
[2026-06-24 01:19:25] INFO segtask_v1.trainer.trainer: Epoch 35/400 | LR=8.75e-04 | loss=0.0703 | val_dice=0.7830 | best=0.7613 (ep35) | 08:13:16 | L_res_0=0.0738 L_res_1=0.0714 L_res_2=0.0775
[2026-06-24 01:19:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 35): 11924.0 MiB
[2026-06-24 01:33:23] INFO segtask_v1.trainer.validation:   Val: loss=0.4109, pooled_mean_dice=0.7570, per_class=['0.7570'], iou=0.6090, recall=0.9906, precision=0.6125, vol_sim=0.7641, mcc=0.7241, min_class_dice=0.7570, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7587, per_class_sd=['0.7587'], combined(w=0.50)=0.7578, balanced=0.7351
[2026-06-24 01:33:23] INFO segtask_v1.trainer.trainer: Epoch 36/400 | LR=9.00e-04 | loss=0.0672 | val_dice=0.7570 | best=0.7613 (ep35) | 08:27:14 | L_res_0=0.0688 L_res_1=0.0690 L_res_2=0.0759
[2026-06-24 01:33:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 36): 11924.0 MiB
[2026-06-24 01:47:15] INFO segtask_v1.trainer.validation:   Val: loss=0.3870, pooled_mean_dice=0.7719, per_class=['0.7719'], iou=0.6285, recall=0.9920, precision=0.6317, vol_sim=0.7781, mcc=0.7400, min_class_dice=0.7719, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7770, per_class_sd=['0.7770'], combined(w=0.50)=0.7745, balanced=0.7515
[2026-06-24 01:47:15] INFO segtask_v1.trainer.trainer: Epoch 37/400 | LR=9.25e-04 | loss=0.0575 | val_dice=0.7719 | best=0.7613 (ep35) | 08:41:05 | L_res_0=0.0578 L_res_1=0.0590 L_res_2=0.0653
[2026-06-24 01:47:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 37): 11923.5 MiB
[2026-06-24 02:01:08] INFO segtask_v1.trainer.validation:   Val: loss=0.3561, pooled_mean_dice=0.7843, per_class=['0.7843'], iou=0.6452, recall=0.9938, precision=0.6478, vol_sim=0.7893, mcc=0.7519, min_class_dice=0.7843, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7779, per_class_sd=['0.7779'], combined(w=0.50)=0.7811, balanced=0.7616
[2026-06-24 02:01:12] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-24 02:01:12] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7616 at epoch 38
[2026-06-24 02:01:12] INFO segtask_v1.trainer.trainer: Epoch 38/400 | LR=9.50e-04 | loss=0.0537 | val_dice=0.7843 | best=0.7616 (ep38) | 08:55:03 | L_res_0=0.0518 L_res_1=0.0550 L_res_2=0.0616
[2026-06-24 02:01:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 38): 11924.0 MiB
[2026-06-24 02:15:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3950, pooled_mean_dice=0.7646, per_class=['0.7646'], iou=0.6190, recall=0.9925, precision=0.6219, vol_sim=0.7704, mcc=0.7350, min_class_dice=0.7646, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7811, per_class_sd=['0.7811'], combined(w=0.50)=0.7729, balanced=0.7468
[2026-06-24 02:15:05] INFO segtask_v1.trainer.trainer: Epoch 39/400 | LR=9.75e-04 | loss=0.1134 | val_dice=0.7646 | best=0.7616 (ep38) | 09:08:56 | L_res_0=0.1053 L_res_1=0.1161 L_res_2=0.1312
[2026-06-24 02:15:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 39): 11923.0 MiB
[2026-06-24 02:28:58] INFO segtask_v1.trainer.validation:   Val: loss=0.7952, pooled_mean_dice=0.4860, per_class=['0.4860'], iou=0.3210, recall=0.9842, precision=0.3227, vol_sim=0.4938, mcc=0.4177, min_class_dice=0.4860, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.3583, per_class_sd=['0.3583'], combined(w=0.50)=0.4222, balanced=0.4267
[2026-06-24 02:28:58] INFO segtask_v1.trainer.trainer: Epoch 40/400 | LR=1.00e-03 | loss=0.3412 | val_dice=0.4860 | best=0.7616 (ep38) | 09:22:49 | L_res_0=0.2785 L_res_1=0.3348 L_res_2=0.3996
[2026-06-24 02:28:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 40): 11923.1 MiB
[2026-06-24 02:42:51] INFO segtask_v1.trainer.validation:   Val: loss=0.9648, pooled_mean_dice=0.3872, per_class=['0.3872'], iou=0.2401, recall=0.9564, precision=0.2427, vol_sim=0.4048, mcc=0.2570, min_class_dice=0.3872, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2456, per_class_sd=['0.2456'], combined(w=0.50)=0.3164, balanced=0.3223
[2026-06-24 02:42:51] INFO segtask_v1.trainer.trainer: Epoch 41/400 | LR=1.00e-03 | loss=0.2820 | val_dice=0.3872 | best=0.7616 (ep38) | 09:36:41 | L_res_0=0.2565 L_res_1=0.2796 L_res_2=0.3203
[2026-06-24 02:42:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 41): 11923.0 MiB
[2026-06-24 02:56:43] INFO segtask_v1.trainer.validation:   Val: loss=0.9684, pooled_mean_dice=0.3726, per_class=['0.3726'], iou=0.2290, recall=0.9419, precision=0.2323, vol_sim=0.3956, mcc=0.2438, min_class_dice=0.3726, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2395, per_class_sd=['0.2395'], combined(w=0.50)=0.3061, balanced=0.3117
[2026-06-24 02:56:43] INFO segtask_v1.trainer.trainer: Epoch 42/400 | LR=1.00e-03 | loss=0.2226 | val_dice=0.3726 | best=0.7616 (ep38) | 09:50:34 | L_res_0=0.2138 L_res_1=0.2452 L_res_2=0.2540
[2026-06-24 02:56:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 42): 11923.1 MiB
[2026-06-24 03:10:37] INFO segtask_v1.trainer.validation:   Val: loss=0.9591, pooled_mean_dice=0.3753, per_class=['0.3753'], iou=0.2310, recall=0.9341, precision=0.2348, vol_sim=0.4018, mcc=0.2507, min_class_dice=0.3753, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2395, per_class_sd=['0.2395'], combined(w=0.50)=0.3074, balanced=0.3133
[2026-06-24 03:10:37] INFO segtask_v1.trainer.trainer: Epoch 43/400 | LR=1.00e-03 | loss=0.1865 | val_dice=0.3753 | best=0.7616 (ep38) | 10:04:27 | L_res_0=0.1845 L_res_1=0.2136 L_res_2=0.2134
[2026-06-24 03:10:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 43): 11922.8 MiB
[2026-06-24 03:24:29] INFO segtask_v1.trainer.validation:   Val: loss=0.9262, pooled_mean_dice=0.3942, per_class=['0.3942'], iou=0.2455, recall=0.9404, precision=0.2493, vol_sim=0.4192, mcc=0.2747, min_class_dice=0.3942, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2439, per_class_sd=['0.2439'], combined(w=0.50)=0.3190, balanced=0.3257
[2026-06-24 03:24:29] INFO segtask_v1.trainer.trainer: Epoch 44/400 | LR=1.00e-03 | loss=0.1566 | val_dice=0.3942 | best=0.7616 (ep38) | 10:18:19 | L_res_0=0.1497 L_res_1=0.1817 L_res_2=0.1943
[2026-06-24 03:24:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 44): 11923.4 MiB
[2026-06-24 03:38:22] INFO segtask_v1.trainer.validation:   Val: loss=0.9393, pooled_mean_dice=0.3889, per_class=['0.3889'], iou=0.2414, recall=0.9498, precision=0.2445, vol_sim=0.4094, mcc=0.2887, min_class_dice=0.3889, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2533, per_class_sd=['0.2533'], combined(w=0.50)=0.3211, balanced=0.3275
[2026-06-24 03:38:22] INFO segtask_v1.trainer.trainer: Epoch 45/400 | LR=1.00e-03 | loss=0.1364 | val_dice=0.3889 | best=0.7616 (ep38) | 10:32:12 | L_res_0=0.1310 L_res_1=0.1493 L_res_2=0.1731
[2026-06-24 03:38:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 45): 11923.4 MiB
[2026-06-24 03:52:13] INFO segtask_v1.trainer.validation:   Val: loss=0.9086, pooled_mean_dice=0.4152, per_class=['0.4152'], iou=0.2620, recall=0.9429, precision=0.2662, vol_sim=0.4404, mcc=0.3066, min_class_dice=0.4152, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2656, per_class_sd=['0.2656'], combined(w=0.50)=0.3404, balanced=0.3472
[2026-06-24 03:52:13] INFO segtask_v1.trainer.trainer: Epoch 46/400 | LR=1.00e-03 | loss=0.1159 | val_dice=0.4152 | best=0.7616 (ep38) | 10:46:04 | L_res_0=0.1165 L_res_1=0.1230 L_res_2=0.1333
[2026-06-24 03:52:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 46): 11923.2 MiB
[2026-06-24 04:06:06] INFO segtask_v1.trainer.validation:   Val: loss=0.8843, pooled_mean_dice=0.4263, per_class=['0.4263'], iou=0.2709, recall=0.9503, precision=0.2747, vol_sim=0.4486, mcc=0.3268, min_class_dice=0.4263, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2808, per_class_sd=['0.2808'], combined(w=0.50)=0.3535, balanced=0.3602
[2026-06-24 04:06:06] INFO segtask_v1.trainer.trainer: Epoch 47/400 | LR=1.00e-03 | loss=0.1209 | val_dice=0.4263 | best=0.7616 (ep38) | 10:59:57 | L_res_0=0.1217 L_res_1=0.1273 L_res_2=0.1270
[2026-06-24 04:06:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 47): 11923.6 MiB
[2026-06-24 04:20:00] INFO segtask_v1.trainer.validation:   Val: loss=0.8486, pooled_mean_dice=0.4592, per_class=['0.4592'], iou=0.2980, recall=0.9498, precision=0.3028, vol_sim=0.4834, mcc=0.3650, min_class_dice=0.4592, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.3083, per_class_sd=['0.3083'], combined(w=0.50)=0.3838, balanced=0.3906
[2026-06-24 04:20:00] INFO segtask_v1.trainer.trainer: Epoch 48/400 | LR=1.00e-03 | loss=0.1112 | val_dice=0.4592 | best=0.7616 (ep38) | 11:13:50 | L_res_0=0.1172 L_res_1=0.1122 L_res_2=0.1161
[2026-06-24 04:20:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 48): 11923.8 MiB
[2026-06-24 04:33:53] INFO segtask_v1.trainer.validation:   Val: loss=0.8218, pooled_mean_dice=0.4994, per_class=['0.4994'], iou=0.3328, recall=0.9613, precision=0.3373, vol_sim=0.5195, mcc=0.4223, min_class_dice=0.4994, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.3724, per_class_sd=['0.3724'], combined(w=0.50)=0.4359, balanced=0.4396
[2026-06-24 04:33:53] INFO segtask_v1.trainer.trainer: Epoch 49/400 | LR=1.00e-03 | loss=0.0867 | val_dice=0.4994 | best=0.7616 (ep38) | 11:27:43 | L_res_0=0.0890 L_res_1=0.0871 L_res_2=0.0926
[2026-06-24 04:33:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 49): 11923.1 MiB
[2026-06-24 04:47:46] INFO segtask_v1.trainer.validation:   Val: loss=0.8301, pooled_mean_dice=0.5047, per_class=['0.5047'], iou=0.3376, recall=0.9593, precision=0.3425, vol_sim=0.5262, mcc=0.4410, min_class_dice=0.5047, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4243, per_class_sd=['0.4243'], combined(w=0.50)=0.4645, balanced=0.4604
[2026-06-24 04:47:46] INFO segtask_v1.trainer.trainer: Epoch 50/400 | LR=1.00e-03 | loss=0.0839 | val_dice=0.5047 | best=0.7616 (ep38) | 11:41:37 | L_res_0=0.0845 L_res_1=0.0854 L_res_2=0.0905
[2026-06-24 04:47:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 50): 11922.9 MiB
[2026-06-24 05:01:41] INFO segtask_v1.trainer.validation:   Val: loss=0.7686, pooled_mean_dice=0.5538, per_class=['0.5538'], iou=0.3829, recall=0.9671, precision=0.3880, vol_sim=0.5726, mcc=0.4979, min_class_dice=0.5538, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.4621, per_class_sd=['0.4621'], combined(w=0.50)=0.5080, balanced=0.5053
[2026-06-24 05:01:41] INFO segtask_v1.trainer.trainer: Epoch 51/400 | LR=1.00e-03 | loss=0.0829 | val_dice=0.5538 | best=0.7616 (ep38) | 11:55:31 | L_res_0=0.0821 L_res_1=0.0840 L_res_2=0.0889
[2026-06-24 05:01:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 51): 11924.2 MiB
[2026-06-24 05:15:35] INFO segtask_v1.trainer.validation:   Val: loss=0.7234, pooled_mean_dice=0.5873, per_class=['0.5873'], iou=0.4158, recall=0.9742, precision=0.4204, vol_sim=0.6029, mcc=0.5322, min_class_dice=0.5873, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.5139, per_class_sd=['0.5139'], combined(w=0.50)=0.5506, balanced=0.5436
[2026-06-24 05:15:35] INFO segtask_v1.trainer.trainer: Epoch 52/400 | LR=1.00e-03 | loss=0.0866 | val_dice=0.5873 | best=0.7616 (ep38) | 12:09:25 | L_res_0=0.0866 L_res_1=0.0872 L_res_2=0.0950
[2026-06-24 05:15:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 52): 11924.9 MiB
[2026-06-24 05:29:28] INFO segtask_v1.trainer.validation:   Val: loss=0.6841, pooled_mean_dice=0.6073, per_class=['0.6073'], iou=0.4360, recall=0.9735, precision=0.4412, vol_sim=0.6238, mcc=0.5540, min_class_dice=0.6073, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.5381, per_class_sd=['0.5381'], combined(w=0.50)=0.5727, balanced=0.5645
[2026-06-24 05:29:28] INFO segtask_v1.trainer.trainer: Epoch 53/400 | LR=1.00e-03 | loss=0.0718 | val_dice=0.6073 | best=0.7616 (ep38) | 12:23:19 | L_res_0=0.0711 L_res_1=0.0732 L_res_2=0.0789
[2026-06-24 05:29:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 53): 11922.8 MiB
[2026-06-24 05:43:22] INFO segtask_v1.trainer.validation:   Val: loss=0.6447, pooled_mean_dice=0.6301, per_class=['0.6301'], iou=0.4599, recall=0.9802, precision=0.4643, vol_sim=0.6428, mcc=0.5834, min_class_dice=0.6301, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.5762, per_class_sd=['0.5762'], combined(w=0.50)=0.6031, balanced=0.5917
[2026-06-24 05:43:22] INFO segtask_v1.trainer.trainer: Epoch 54/400 | LR=1.00e-03 | loss=0.0707 | val_dice=0.6301 | best=0.7616 (ep38) | 12:37:13 | L_res_0=0.0699 L_res_1=0.0710 L_res_2=0.0774
[2026-06-24 05:43:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 54): 11924.8 MiB
[2026-06-24 05:57:15] INFO segtask_v1.trainer.validation:   Val: loss=0.6146, pooled_mean_dice=0.6530, per_class=['0.6530'], iou=0.4848, recall=0.9797, precision=0.4897, vol_sim=0.6665, mcc=0.6051, min_class_dice=0.6530, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.5914, per_class_sd=['0.5914'], combined(w=0.50)=0.6222, balanced=0.6125
[2026-06-24 05:57:15] INFO segtask_v1.trainer.trainer: Epoch 55/400 | LR=1.00e-03 | loss=0.0678 | val_dice=0.6530 | best=0.7616 (ep38) | 12:51:06 | L_res_0=0.0643 L_res_1=0.0697 L_res_2=0.0753
[2026-06-24 05:57:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 55): 11923.2 MiB
[2026-06-24 06:11:07] INFO segtask_v1.trainer.validation:   Val: loss=0.5651, pooled_mean_dice=0.6822, per_class=['0.6822'], iou=0.5177, recall=0.9817, precision=0.5227, vol_sim=0.6949, mcc=0.6350, min_class_dice=0.6822, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6216, per_class_sd=['0.6216'], combined(w=0.50)=0.6519, balanced=0.6421
[2026-06-24 06:11:07] INFO segtask_v1.trainer.trainer: Epoch 56/400 | LR=1.00e-03 | loss=0.0651 | val_dice=0.6822 | best=0.7616 (ep38) | 13:04:58 | L_res_0=0.0614 L_res_1=0.0672 L_res_2=0.0725
[2026-06-24 06:11:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 56): 11923.3 MiB
[2026-06-24 06:25:00] INFO segtask_v1.trainer.validation:   Val: loss=0.5689, pooled_mean_dice=0.6759, per_class=['0.6759'], iou=0.5105, recall=0.9848, precision=0.5145, vol_sim=0.6864, mcc=0.6336, min_class_dice=0.6759, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6385, per_class_sd=['0.6385'], combined(w=0.50)=0.6572, balanced=0.6422
[2026-06-24 06:25:00] INFO segtask_v1.trainer.trainer: Epoch 57/400 | LR=1.00e-03 | loss=0.0615 | val_dice=0.6759 | best=0.7616 (ep38) | 13:18:50 | L_res_0=0.0578 L_res_1=0.0632 L_res_2=0.0688
[2026-06-24 06:25:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 57): 11924.2 MiB
[2026-06-24 06:38:55] INFO segtask_v1.trainer.validation:   Val: loss=0.5462, pooled_mean_dice=0.6946, per_class=['0.6946'], iou=0.5321, recall=0.9788, precision=0.5383, vol_sim=0.7096, mcc=0.6512, min_class_dice=0.6946, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6677, per_class_sd=['0.6677'], combined(w=0.50)=0.6812, balanced=0.6636
[2026-06-24 06:38:55] INFO segtask_v1.trainer.trainer: Epoch 58/400 | LR=1.00e-03 | loss=0.0598 | val_dice=0.6946 | best=0.7616 (ep38) | 13:32:45 | L_res_0=0.0552 L_res_1=0.0616 L_res_2=0.0672
[2026-06-24 06:38:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 58): 11923.7 MiB
[2026-06-24 06:52:49] INFO segtask_v1.trainer.validation:   Val: loss=0.5386, pooled_mean_dice=0.6863, per_class=['0.6863'], iou=0.5225, recall=0.9850, precision=0.5266, vol_sim=0.6968, mcc=0.6508, min_class_dice=0.6863, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6754, per_class_sd=['0.6754'], combined(w=0.50)=0.6809, balanced=0.6597
[2026-06-24 06:52:49] INFO segtask_v1.trainer.trainer: Epoch 59/400 | LR=1.00e-03 | loss=0.0602 | val_dice=0.6863 | best=0.7616 (ep38) | 13:46:39 | L_res_0=0.0566 L_res_1=0.0617 L_res_2=0.0669
[2026-06-24 06:52:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 59): 11925.1 MiB
[2026-06-24 07:06:42] INFO segtask_v1.trainer.validation:   Val: loss=0.5214, pooled_mean_dice=0.6990, per_class=['0.6990'], iou=0.5373, recall=0.9794, precision=0.5435, vol_sim=0.7137, mcc=0.6621, min_class_dice=0.6990, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7008, per_class_sd=['0.7008'], combined(w=0.50)=0.6999, balanced=0.6755
[2026-06-24 07:06:42] INFO segtask_v1.trainer.trainer: Epoch 60/400 | LR=1.00e-03 | loss=0.0606 | val_dice=0.6990 | best=0.7616 (ep38) | 14:00:32 | L_res_0=0.0569 L_res_1=0.0623 L_res_2=0.0682
[2026-06-24 07:06:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 60): 11923.7 MiB
[2026-06-24 07:20:35] INFO segtask_v1.trainer.validation:   Val: loss=0.4882, pooled_mean_dice=0.7141, per_class=['0.7141'], iou=0.5554, recall=0.9819, precision=0.5611, vol_sim=0.7273, mcc=0.6792, min_class_dice=0.7141, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6968, per_class_sd=['0.6968'], combined(w=0.50)=0.7055, balanced=0.6865
[2026-06-24 07:20:35] INFO segtask_v1.trainer.trainer: Epoch 61/400 | LR=1.00e-03 | loss=0.0603 | val_dice=0.7141 | best=0.7616 (ep38) | 14:14:26 | L_res_0=0.0568 L_res_1=0.0618 L_res_2=0.0671
[2026-06-24 07:20:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 61): 11923.4 MiB
[2026-06-24 07:34:28] INFO segtask_v1.trainer.validation:   Val: loss=0.4369, pooled_mean_dice=0.7444, per_class=['0.7444'], iou=0.5929, recall=0.9882, precision=0.5971, vol_sim=0.7533, mcc=0.7050, min_class_dice=0.7444, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7152, per_class_sd=['0.7152'], combined(w=0.50)=0.7298, balanced=0.7141
[2026-06-24 07:34:28] INFO segtask_v1.trainer.trainer: Epoch 62/400 | LR=1.00e-03 | loss=0.0599 | val_dice=0.7444 | best=0.7616 (ep38) | 14:28:19 | L_res_0=0.0563 L_res_1=0.0621 L_res_2=0.0670
[2026-06-24 07:34:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 62): 11923.3 MiB
[2026-06-24 07:48:20] INFO segtask_v1.trainer.validation:   Val: loss=0.4753, pooled_mean_dice=0.7265, per_class=['0.7265'], iou=0.5705, recall=0.9889, precision=0.5742, vol_sim=0.7347, mcc=0.6934, min_class_dice=0.7265, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7257, per_class_sd=['0.7257'], combined(w=0.50)=0.7261, balanced=0.7032
[2026-06-24 07:48:20] INFO segtask_v1.trainer.trainer: Epoch 63/400 | LR=1.00e-03 | loss=0.0592 | val_dice=0.7265 | best=0.7616 (ep38) | 14:42:11 | L_res_0=0.0558 L_res_1=0.0610 L_res_2=0.0660
[2026-06-24 07:48:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 63): 11924.4 MiB
[2026-06-24 08:02:13] INFO segtask_v1.trainer.validation:   Val: loss=0.4817, pooled_mean_dice=0.7223, per_class=['0.7223'], iou=0.5654, recall=0.9862, precision=0.5699, vol_sim=0.7324, mcc=0.6850, min_class_dice=0.7223, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.7027, per_class_sd=['0.7027'], combined(w=0.50)=0.7125, balanced=0.6941
[2026-06-24 08:02:13] INFO segtask_v1.trainer.trainer: Epoch 64/400 | LR=1.00e-03 | loss=0.1249 | val_dice=0.7223 | best=0.7616 (ep38) | 14:56:03 | L_res_0=0.1338 L_res_1=0.1226 L_res_2=0.1281
[2026-06-24 08:02:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 64): 11923.7 MiB
[2026-06-24 08:16:07] INFO segtask_v1.trainer.validation:   Val: loss=0.5066, pooled_mean_dice=0.7018, per_class=['0.7018'], iou=0.5406, recall=0.9916, precision=0.5431, vol_sim=0.7078, mcc=0.6651, min_class_dice=0.7018, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6755, per_class_sd=['0.6755'], combined(w=0.50)=0.6886, balanced=0.6716
[2026-06-24 08:16:07] INFO segtask_v1.trainer.trainer: Epoch 65/400 | LR=1.00e-03 | loss=0.0863 | val_dice=0.7018 | best=0.7616 (ep38) | 15:09:58 | L_res_0=0.0862 L_res_1=0.0875 L_res_2=0.0935
[2026-06-24 08:16:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 65): 11923.0 MiB
[2026-06-24 08:30:00] INFO segtask_v1.trainer.validation:   Val: loss=0.5099, pooled_mean_dice=0.7039, per_class=['0.7039'], iou=0.5431, recall=0.9894, precision=0.5463, vol_sim=0.7115, mcc=0.6667, min_class_dice=0.7039, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6927, per_class_sd=['0.6927'], combined(w=0.50)=0.6983, balanced=0.6773
[2026-06-24 08:30:00] INFO segtask_v1.trainer.trainer: Epoch 66/400 | LR=1.00e-03 | loss=0.0643 | val_dice=0.7039 | best=0.7616 (ep38) | 15:23:51 | L_res_0=0.0611 L_res_1=0.0663 L_res_2=0.0722
[2026-06-24 08:30:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 66): 11922.7 MiB
[2026-06-24 08:43:52] INFO segtask_v1.trainer.validation:   Val: loss=0.4991, pooled_mean_dice=0.7058, per_class=['0.7058'], iou=0.5454, recall=0.9899, precision=0.5485, vol_sim=0.7131, mcc=0.6690, min_class_dice=0.7058, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6823, per_class_sd=['0.6823'], combined(w=0.50)=0.6941, balanced=0.6763
[2026-06-24 08:43:52] INFO segtask_v1.trainer.trainer: Epoch 67/400 | LR=1.00e-03 | loss=0.0637 | val_dice=0.7058 | best=0.7616 (ep38) | 15:37:43 | L_res_0=0.0621 L_res_1=0.0650 L_res_2=0.0702
[2026-06-24 08:43:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 67): 11924.1 MiB
[2026-06-24 08:57:48] INFO segtask_v1.trainer.validation:   Val: loss=0.5061, pooled_mean_dice=0.7051, per_class=['0.7051'], iou=0.5446, recall=0.9876, precision=0.5483, vol_sim=0.7140, mcc=0.6694, min_class_dice=0.7051, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.6947, per_class_sd=['0.6947'], combined(w=0.50)=0.6999, balanced=0.6789
[2026-06-24 08:57:48] INFO segtask_v1.trainer.trainer: Epoch 68/400 | LR=1.00e-03 | loss=0.0735 | val_dice=0.7051 | best=0.7616 (ep38) | 15:51:39 | L_res_0=0.0747 L_res_1=0.0742 L_res_2=0.0799
[2026-06-24 08:57:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 68): 11923.5 MiB
