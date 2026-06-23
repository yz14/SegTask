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


2 我用lungves0.yaml和lungves1.yaml训练，用lungves_weight_prep.py制作的区域权重，训练的指标看起来很低：
lungves0.yaml：
[2026-06-22 14:29:31] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-22 14:29:31] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-22 14:29:31] INFO __main__: Device: cuda
[2026-06-22 14:29:31] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-22 14:29:31] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-22 14:29:31] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-22 14:29:31] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-22 14:29:46] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-22 14:29:46] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-22 14:29:46] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-22 14:30:23] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-22 14:30:23] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-22 14:30:34] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-22 14:30:34] INFO segtask_v1.data.loader: DataLoader: batch_size=8, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-22 14:30:35] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-22 14:30:35] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 1, 1]
[2026-06-22 14:30:35] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.32M, total=47.12M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-22 14:30:36] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-22 14:30:36] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-22 14:30:36] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-22 14:30:36] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=centerline, loss=dice, iter=3, weight=0.300)
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: Model params: 47.12M
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.7 + grad=179.7 + optim(AdamW,2x)=359.5 + ema=179.8 = 898.8 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=362.5 MiB, reserved=376.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: Train batches: 88, Val batches: 11
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=8
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_cldice
[2026-06-22 14:30:36] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:31:01] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 22042.9 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-22 14:31:46] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-22 14:31:46] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-22 14:31:46] INFO __main__: Device: cuda
[2026-06-22 14:31:46] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-22 14:31:46] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-22 14:31:46] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-22 14:31:46] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-22 14:32:01] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-22 14:32:01] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-22 14:32:01] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-22 14:32:36] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-22 14:32:36] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-22 14:32:46] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-22 14:32:46] INFO segtask_v1.data.loader: DataLoader: batch_size=4, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-22 14:32:46] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-22 14:32:46] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 1, 1]
[2026-06-22 14:32:47] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.32M, total=47.12M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-22 14:32:48] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-22 14:32:48] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-22 14:32:48] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-22 14:32:48] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=centerline, loss=dice, iter=3, weight=0.300)
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: Model params: 47.12M
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.7 + grad=179.7 + optim(AdamW,2x)=359.5 + ema=179.8 = 898.8 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=362.5 MiB, reserved=376.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: Train batches: 176, Val batches: 22
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=4
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_cldice
[2026-06-22 14:32:48] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:33:00] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 11277.2 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-22 14:34:27] INFO segtask_v1.trainer.validation:   Val: loss=1.9383, pooled_mean_dice=0.0463, per_class=['0.0463'], iou=0.0237, recall=0.4985, precision=0.0243, vol_sim=0.0928, mcc=-0.0006, min_class_dice=0.0463, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.0734, per_class_sd=['0.0734'], combined(w=0.50)=0.0599, balanced=0.0504
[2026-06-22 14:34:28] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 14:34:28] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0504 at epoch 1
[2026-06-22 14:34:28] INFO segtask_v1.trainer.trainer: Epoch 1/400 | LR=2.01e-04 | loss=3.2729 | val_dice=0.0463 | best=0.0504 (ep1) | 00:01:40 | L_main=1.4460 L_aux_1=1.5219(w=0.5) L_aux_2=1.5371(w=0.5)
[2026-06-22 14:34:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 11675.0 MiB
[2026-06-22 14:35:49] INFO segtask_v1.trainer.validation:   Val: loss=1.9553, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[66]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:35:49] INFO segtask_v1.trainer.trainer: Epoch 2/400 | LR=4.01e-04 | loss=2.8155 | val_dice=0.0000 | best=0.0504 (ep1) | 00:03:00 | L_main=1.2044 L_aux_1=1.3072(w=0.5) L_aux_2=1.3309(w=0.5)
[2026-06-22 14:35:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 11851.6 MiB
[2026-06-22 14:37:10] INFO segtask_v1.trainer.validation:   Val: loss=1.9538, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:37:10] INFO segtask_v1.trainer.trainer: Epoch 3/400 | LR=6.00e-04 | loss=2.6123 | val_dice=0.0000 | best=0.0504 (ep1) | 00:04:21 | L_main=1.1098 L_aux_1=1.2485(w=0.5) L_aux_2=1.2731(w=0.5)
[2026-06-22 14:37:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 11851.6 MiB
[2026-06-22 14:38:30] INFO segtask_v1.trainer.validation:   Val: loss=1.9413, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:38:30] INFO segtask_v1.trainer.trainer: Epoch 4/400 | LR=8.00e-04 | loss=2.4985 | val_dice=0.0000 | best=0.0504 (ep1) | 00:05:41 | L_main=1.0529 L_aux_1=1.2158(w=0.5) L_aux_2=1.2447(w=0.5)
[2026-06-22 14:38:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 4): 11851.6 MiB
[2026-06-22 14:39:50] INFO segtask_v1.trainer.validation:   Val: loss=1.9459, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:39:50] INFO segtask_v1.trainer.trainer: Epoch 5/400 | LR=1.00e-03 | loss=2.3919 | val_dice=0.0000 | best=0.0504 (ep1) | 00:07:02 | L_main=0.9846 L_aux_1=1.1851(w=0.5) L_aux_2=1.2079(w=0.5)
[2026-06-22 14:39:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 5): 11851.6 MiB
[2026-06-22 14:41:11] INFO segtask_v1.trainer.validation:   Val: loss=1.9605, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:41:11] INFO segtask_v1.trainer.trainer: Epoch 6/400 | LR=1.00e-03 | loss=2.2892 | val_dice=0.0000 | best=0.0504 (ep1) | 00:08:23 | L_main=0.9273 L_aux_1=1.1560(w=0.5) L_aux_2=1.1552(w=0.5)
[2026-06-22 14:41:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 6): 11851.6 MiB
[2026-06-22 14:42:32] INFO segtask_v1.trainer.validation:   Val: loss=1.9693, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:42:32] INFO segtask_v1.trainer.trainer: Epoch 7/400 | LR=1.00e-03 | loss=2.1859 | val_dice=0.0000 | best=0.0504 (ep1) | 00:09:44 | L_main=0.8716 L_aux_1=1.1163(w=0.5) L_aux_2=1.1093(w=0.5)
[2026-06-22 14:42:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 7): 11851.6 MiB
[2026-06-22 14:43:52] INFO segtask_v1.trainer.validation:   Val: loss=1.9589, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:43:52] INFO segtask_v1.trainer.trainer: Epoch 8/400 | LR=1.00e-03 | loss=2.1327 | val_dice=0.0000 | best=0.0504 (ep1) | 00:11:04 | L_main=0.8458 L_aux_1=1.0936(w=0.5) L_aux_2=1.0802(w=0.5)
[2026-06-22 14:43:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 8): 11851.6 MiB
[2026-06-22 14:45:13] INFO segtask_v1.trainer.validation:   Val: loss=1.9774, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:45:13] INFO segtask_v1.trainer.trainer: Epoch 9/400 | LR=1.00e-03 | loss=2.0764 | val_dice=0.0000 | best=0.0504 (ep1) | 00:12:24 | L_main=0.8203 L_aux_1=1.0648(w=0.5) L_aux_2=1.0503(w=0.5)
[2026-06-22 14:45:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 9): 11851.6 MiB
[2026-06-22 14:46:33] INFO segtask_v1.trainer.validation:   Val: loss=1.9748, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:46:33] INFO segtask_v1.trainer.trainer: Epoch 10/400 | LR=1.00e-03 | loss=2.0829 | val_dice=0.0000 | best=0.0504 (ep1) | 00:13:44 | L_main=0.8274 L_aux_1=1.0617(w=0.5) L_aux_2=1.0494(w=0.5)
[2026-06-22 14:46:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 10): 11851.6 MiB
[2026-06-22 14:47:54] INFO segtask_v1.trainer.validation:   Val: loss=1.9816, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:47:54] INFO segtask_v1.trainer.trainer: Epoch 11/400 | LR=9.99e-04 | loss=1.9776 | val_dice=0.0000 | best=0.0504 (ep1) | 00:15:06 | L_main=0.7900 L_aux_1=0.9974(w=0.5) L_aux_2=0.9892(w=0.5)
[2026-06-22 14:47:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 11): 11851.6 MiB
[2026-06-22 14:49:14] INFO segtask_v1.trainer.validation:   Val: loss=1.9852, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:49:14] INFO segtask_v1.trainer.trainer: Epoch 12/400 | LR=9.99e-04 | loss=1.9987 | val_dice=0.0000 | best=0.0504 (ep1) | 00:16:26 | L_main=0.7946 L_aux_1=1.0100(w=0.5) L_aux_2=1.0056(w=0.5)
[2026-06-22 14:49:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 12): 11851.6 MiB
[2026-06-22 14:50:34] INFO segtask_v1.trainer.validation:   Val: loss=1.9880, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:50:34] INFO segtask_v1.trainer.trainer: Epoch 13/400 | LR=9.99e-04 | loss=1.9065 | val_dice=0.0000 | best=0.0504 (ep1) | 00:17:46 | L_main=0.7627 L_aux_1=0.9492(w=0.5) L_aux_2=0.9509(w=0.5)
[2026-06-22 14:50:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 13): 11851.6 MiB
[2026-06-22 14:51:54] INFO segtask_v1.trainer.validation:   Val: loss=1.9912, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 14:51:54] INFO segtask_v1.trainer.trainer: Epoch 14/400 | LR=9.99e-04 | loss=1.8983 | val_dice=0.0000 | best=0.0504 (ep1) | 00:19:06 | L_main=0.7555 L_aux_1=0.9450(w=0.5) L_aux_2=0.9568(w=0.5)
[2026-06-22 14:51:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 14): 11851.6 MiB
[2026-06-22 14:53:14] INFO segtask_v1.trainer.validation:   Val: loss=1.9834, pooled_mean_dice=0.0031, per_class=['0.0031'], iou=0.0016, recall=0.0016, precision=0.9521, vol_sim=0.0033, mcc=0.0379, min_class_dice=0.0031, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.0121, per_class_sd=['0.0121'], combined(w=0.50)=0.0076, balanced=0.0038
[2026-06-22 14:53:14] INFO segtask_v1.trainer.trainer: Epoch 15/400 | LR=9.98e-04 | loss=1.8245 | val_dice=0.0031 | best=0.0504 (ep1) | 00:20:26 | L_main=0.7358 L_aux_1=0.8983(w=0.5) L_aux_2=0.8959(w=0.5)
[2026-06-22 14:53:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 15): 11851.6 MiB
[2026-06-22 14:54:52] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-22 14:54:52] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-22 14:54:52] INFO __main__: Device: cuda
[2026-06-22 14:54:52] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-22 14:54:52] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-22 14:54:52] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-22 14:54:52] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-22 14:55:07] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-22 14:55:07] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-22 14:55:07] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-22 14:55:42] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-22 14:55:42] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-22 14:55:52] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-22 14:55:52] INFO segtask_v1.data.loader: DataLoader: batch_size=8, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-22 14:55:52] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-22 14:55:52] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-22 14:55:53] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.28M, total=47.08M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-22 14:55:54] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-22 14:55:54] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-22 14:55:54] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-22 14:55:54] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=centerline, loss=dice, iter=3, weight=0.300)
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: Model params: 47.08M
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.6 + grad=179.6 + optim(AdamW,2x)=359.2 + ema=179.7 = 898.1 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=364.0 MiB, reserved=374.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: Train batches: 88, Val batches: 11
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=8
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_cldice
[2026-06-22 14:55:54] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:56:13] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 21665.6 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-22 14:58:08] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-22 14:58:08] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-22 14:58:08] INFO __main__: Device: cuda
[2026-06-22 14:58:08] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-22 14:58:08] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-22 14:58:08] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-22 14:58:08] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-22 14:58:23] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-22 14:58:23] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-22 14:58:23] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-22 14:58:58] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-22 14:58:58] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-22 14:59:08] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-22 14:59:08] INFO segtask_v1.data.loader: DataLoader: batch_size=8, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-22 14:59:08] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-22 14:59:08] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-22 14:59:08] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.28M, total=47.08M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-22 14:59:09] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-22 14:59:09] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-22 14:59:09] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-22 14:59:09] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=centerline, loss=dice, iter=3, weight=0.300)
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: Model params: 47.08M
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.6 + grad=179.6 + optim(AdamW,2x)=359.2 + ema=179.7 = 898.1 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=364.0 MiB, reserved=374.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: Train batches: 88, Val batches: 11
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=8
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_cldice
[2026-06-22 14:59:09] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 14:59:29] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 21665.6 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-22 15:00:36] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-22 15:00:36] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-22 15:00:36] INFO __main__: Device: cuda
[2026-06-22 15:00:36] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-22 15:00:36] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-22 15:00:36] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-22 15:00:36] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-22 15:00:51] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-22 15:00:51] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-22 15:00:51] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-22 15:01:26] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-22 15:01:26] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-22 15:01:36] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-22 15:01:36] INFO segtask_v1.data.loader: DataLoader: batch_size=4, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-22 15:01:36] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-22 15:01:36] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 0, 1, 1, 1], dec_stages=[0, 0, 0, 0]
[2026-06-22 15:01:37] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=23.57M, dec=20.28M, total=47.08M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-22 15:01:38] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-22 15:01:38] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-22 15:01:38] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-22 15:01:38] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=centerline, loss=dice, iter=3, weight=0.300)
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: Model params: 47.08M
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=179.6 + grad=179.6 + optim(AdamW,2x)=359.2 + ema=179.7 = 898.1 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=364.0 MiB, reserved=374.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: Train batches: 176, Val batches: 22
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=4
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_cldice
[2026-06-22 15:01:38] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 15:01:51] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 11089.4 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-22 15:03:05] INFO segtask_v1.trainer.validation:   Val: loss=1.9374, pooled_mean_dice=0.0469, per_class=['0.0469'], iou=0.0240, recall=0.4165, precision=0.0248, vol_sim=0.1125, mcc=-0.0001, min_class_dice=0.0469, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.0886, per_class_sd=['0.0886'], combined(w=0.50)=0.0678, balanced=0.0524
[2026-06-22 15:03:09] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 15:03:09] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.0524 at epoch 1
[2026-06-22 15:03:09] INFO segtask_v1.trainer.trainer: Epoch 1/400 | LR=2.01e-04 | loss=3.2238 | val_dice=0.0469 | best=0.0524 (ep1) | 00:01:30 | L_main=1.4313 L_aux_1=1.4766(w=0.5) L_aux_2=1.5141(w=0.5)
[2026-06-22 15:03:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 11490.0 MiB
[2026-06-22 15:04:18] INFO segtask_v1.trainer.validation:   Val: loss=1.9527, pooled_mean_dice=0.0307, per_class=['0.0307'], iou=0.0156, recall=0.0833, precision=0.0188, vol_sim=0.3683, mcc=0.0000, min_class_dice=0.0307, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.0790, per_class_sd=['0.0790'], combined(w=0.50)=0.0548, balanced=0.0359
[2026-06-22 15:04:18] INFO segtask_v1.trainer.trainer: Epoch 2/400 | LR=4.01e-04 | loss=2.8835 | val_dice=0.0307 | best=0.0524 (ep1) | 00:02:40 | L_main=1.2490 L_aux_1=1.3144(w=0.5) L_aux_2=1.3646(w=0.5)
[2026-06-22 15:04:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 11666.1 MiB
[2026-06-22 15:05:27] INFO segtask_v1.trainer.validation:   Val: loss=1.9325, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:05:27] INFO segtask_v1.trainer.trainer: Epoch 3/400 | LR=6.00e-04 | loss=2.7276 | val_dice=0.0000 | best=0.0524 (ep1) | 00:03:49 | L_main=1.1745 L_aux_1=1.2751(w=0.5) L_aux_2=1.3200(w=0.5)
[2026-06-22 15:05:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 11666.1 MiB
[2026-06-22 15:06:36] INFO segtask_v1.trainer.validation:   Val: loss=1.9392, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:06:36] INFO segtask_v1.trainer.trainer: Epoch 4/400 | LR=8.00e-04 | loss=2.5489 | val_dice=0.0000 | best=0.0524 (ep1) | 00:04:57 | L_main=1.0826 L_aux_1=1.2303(w=0.5) L_aux_2=1.2661(w=0.5)
[2026-06-22 15:06:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 4): 11666.1 MiB
[2026-06-22 15:07:45] INFO segtask_v1.trainer.validation:   Val: loss=1.9414, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:07:45] INFO segtask_v1.trainer.trainer: Epoch 5/400 | LR=1.00e-03 | loss=2.5157 | val_dice=0.0000 | best=0.0524 (ep1) | 00:06:07 | L_main=1.0502 L_aux_1=1.2586(w=0.5) L_aux_2=1.2557(w=0.5)
[2026-06-22 15:07:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 5): 11666.1 MiB
[2026-06-22 15:08:54] INFO segtask_v1.trainer.validation:   Val: loss=1.9473, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:08:54] INFO segtask_v1.trainer.trainer: Epoch 6/400 | LR=1.00e-03 | loss=2.3269 | val_dice=0.0000 | best=0.0524 (ep1) | 00:07:16 | L_main=0.9404 L_aux_1=1.1989(w=0.5) L_aux_2=1.1679(w=0.5)
[2026-06-22 15:08:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 6): 11666.1 MiB
[2026-06-22 15:10:02] INFO segtask_v1.trainer.validation:   Val: loss=1.9520, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:10:02] INFO segtask_v1.trainer.trainer: Epoch 7/400 | LR=1.00e-03 | loss=2.2847 | val_dice=0.0000 | best=0.0524 (ep1) | 00:08:24 | L_main=0.8948 L_aux_1=1.2110(w=0.5) L_aux_2=1.1633(w=0.5)
[2026-06-22 15:10:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 7): 11666.1 MiB
[2026-06-22 15:11:11] INFO segtask_v1.trainer.validation:   Val: loss=1.9686, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:11:11] INFO segtask_v1.trainer.trainer: Epoch 8/400 | LR=1.00e-03 | loss=2.1914 | val_dice=0.0000 | best=0.0524 (ep1) | 00:09:32 | L_main=0.8432 L_aux_1=1.1833(w=0.5) L_aux_2=1.1161(w=0.5)
[2026-06-22 15:11:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 8): 11666.1 MiB
[2026-06-22 15:12:19] INFO segtask_v1.trainer.validation:   Val: loss=1.9617, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:12:19] INFO segtask_v1.trainer.trainer: Epoch 9/400 | LR=1.00e-03 | loss=2.1712 | val_dice=0.0000 | best=0.0524 (ep1) | 00:10:41 | L_main=0.8308 L_aux_1=1.1784(w=0.5) L_aux_2=1.1125(w=0.5)
[2026-06-22 15:12:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 9): 11666.1 MiB
[2026-06-22 15:13:28] INFO segtask_v1.trainer.validation:   Val: loss=1.9577, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:13:28] INFO segtask_v1.trainer.trainer: Epoch 10/400 | LR=1.00e-03 | loss=2.1559 | val_dice=0.0000 | best=0.0524 (ep1) | 00:11:50 | L_main=0.8315 L_aux_1=1.1566(w=0.5) L_aux_2=1.0960(w=0.5)
[2026-06-22 15:13:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 10): 11666.1 MiB
[2026-06-22 15:14:42] INFO segtask_v1.trainer.validation:   Val: loss=1.9611, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:14:42] INFO segtask_v1.trainer.trainer: Epoch 11/400 | LR=9.99e-04 | loss=2.1305 | val_dice=0.0000 | best=0.0524 (ep1) | 00:13:03 | L_main=0.8154 L_aux_1=1.1517(w=0.5) L_aux_2=1.0884(w=0.5)
[2026-06-22 15:14:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 11): 11666.1 MiB
[2026-06-22 15:15:51] INFO segtask_v1.trainer.validation:   Val: loss=1.9689, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:15:51] INFO segtask_v1.trainer.trainer: Epoch 12/400 | LR=9.99e-04 | loss=2.0898 | val_dice=0.0000 | best=0.0524 (ep1) | 00:14:13 | L_main=0.7929 L_aux_1=1.1286(w=0.5) L_aux_2=1.0739(w=0.5)
[2026-06-22 15:15:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 12): 11666.1 MiB
[2026-06-22 15:17:00] INFO segtask_v1.trainer.validation:   Val: loss=1.9731, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:17:00] INFO segtask_v1.trainer.trainer: Epoch 13/400 | LR=9.99e-04 | loss=2.0771 | val_dice=0.0000 | best=0.0524 (ep1) | 00:15:21 | L_main=0.7895 L_aux_1=1.1164(w=0.5) L_aux_2=1.0688(w=0.5)
[2026-06-22 15:17:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 13): 11666.1 MiB
[2026-06-22 15:18:09] INFO segtask_v1.trainer.validation:   Val: loss=1.9734, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=1.0000, vol_sim=0.0000, mcc=0.0000, min_class_dice=0.0000, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.0000, per_class_sd=['0.0000'], combined(w=0.50)=0.0000, balanced=0.0000
[2026-06-22 15:18:09] INFO segtask_v1.trainer.trainer: Epoch 14/400 | LR=9.99e-04 | loss=2.1005 | val_dice=0.0000 | best=0.0524 (ep1) | 00:16:30 | L_main=0.8013 L_aux_1=1.1315(w=0.5) L_aux_2=1.0814(w=0.5)
[2026-06-22 15:18:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 14): 11666.1 MiB
[2026-06-22 15:19:17] INFO segtask_v1.trainer.validation:   Val: loss=1.9746, pooled_mean_dice=0.0020, per_class=['0.0020'], iou=0.0010, recall=0.0010, precision=0.9609, vol_sim=0.0021, mcc=0.0305, min_class_dice=0.0020, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.0081, per_class_sd=['0.0081'], combined(w=0.50)=0.0050, balanced=0.0025
[2026-06-22 15:19:17] INFO segtask_v1.trainer.trainer: Epoch 15/400 | LR=9.98e-04 | loss=1.9763 | val_dice=0.0020 | best=0.0524 (ep1) | 00:17:39 | L_main=0.7405 L_aux_1=1.0689(w=0.5) L_aux_2=1.0253(w=0.5)
[2026-06-22 15:19:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 15): 11666.1 MiB
[2026-06-22 15:20:25] INFO segtask_v1.trainer.validation:   Val: loss=1.9586, pooled_mean_dice=0.0315, per_class=['0.0315'], iou=0.0160, recall=0.0160, precision=0.8949, vol_sim=0.0352, mcc=0.1183, min_class_dice=0.0315, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.0918, per_class_sd=['0.0918'], combined(w=0.50)=0.0617, balanced=0.0374
[2026-06-22 15:20:25] INFO segtask_v1.trainer.trainer: Epoch 16/400 | LR=9.98e-04 | loss=1.9956 | val_dice=0.0315 | best=0.0524 (ep1) | 00:18:47 | L_main=0.7558 L_aux_1=1.0621(w=0.5) L_aux_2=1.0270(w=0.5)
[2026-06-22 15:20:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 16): 11666.1 MiB
[2026-06-22 15:21:34] INFO segtask_v1.trainer.validation:   Val: loss=1.8260, pooled_mean_dice=0.1242, per_class=['0.1242'], iou=0.0662, recall=0.0671, precision=0.8387, vol_sim=0.1481, mcc=0.2334, min_class_dice=0.1242, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.2924, per_class_sd=['0.2924'], combined(w=0.50)=0.2083, balanced=0.1419
[2026-06-22 15:21:38] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 15:21:38] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.1419 at epoch 17
[2026-06-22 15:21:38] INFO segtask_v1.trainer.trainer: Epoch 17/400 | LR=9.98e-04 | loss=1.9340 | val_dice=0.1242 | best=0.1419 (ep17) | 00:20:00 | L_main=0.7286 L_aux_1=1.0337(w=0.5) L_aux_2=0.9987(w=0.5)
[2026-06-22 15:21:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 17): 11666.1 MiB
[2026-06-22 15:22:47] INFO segtask_v1.trainer.validation:   Val: loss=1.6411, pooled_mean_dice=0.2463, per_class=['0.2463'], iou=0.1405, recall=0.1481, precision=0.7309, vol_sim=0.3370, mcc=0.3231, min_class_dice=0.2463, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.4915, per_class_sd=['0.4915'], combined(w=0.50)=0.3689, balanced=0.2717
[2026-06-22 15:22:51] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 15:22:51] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2717 at epoch 18
[2026-06-22 15:22:51] INFO segtask_v1.trainer.trainer: Epoch 18/400 | LR=9.97e-04 | loss=1.9596 | val_dice=0.2463 | best=0.2717 (ep18) | 00:21:13 | L_main=0.7318 L_aux_1=1.0487(w=0.5) L_aux_2=1.0226(w=0.5)
[2026-06-22 15:22:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 18): 11666.1 MiB
[2026-06-22 15:24:01] INFO segtask_v1.trainer.validation:   Val: loss=1.4765, pooled_mean_dice=0.3306, per_class=['0.3306'], iou=0.1981, recall=0.2171, precision=0.6933, vol_sim=0.4769, mcc=0.3799, min_class_dice=0.3306, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.5922, per_class_sd=['0.5922'], combined(w=0.50)=0.4614, balanced=0.3568
[2026-06-22 15:24:04] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 15:24:04] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.3568 at epoch 19
[2026-06-22 15:24:04] INFO segtask_v1.trainer.trainer: Epoch 19/400 | LR=9.97e-04 | loss=1.9157 | val_dice=0.3306 | best=0.3568 (ep19) | 00:22:26 | L_main=0.7221 L_aux_1=1.0194(w=0.5) L_aux_2=0.9860(w=0.5)
[2026-06-22 15:24:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 19): 11666.1 MiB
[2026-06-22 15:25:13] INFO segtask_v1.trainer.validation:   Val: loss=1.3847, pooled_mean_dice=0.4457, per_class=['0.4457'], iou=0.2867, recall=0.3330, precision=0.6738, vol_sim=0.6615, mcc=0.4659, min_class_dice=0.4457, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.6882, per_class_sd=['0.6882'], combined(w=0.50)=0.5670, balanced=0.4679
[2026-06-22 15:25:17] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 15:25:17] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.4679 at epoch 20
[2026-06-22 15:25:17] INFO segtask_v1.trainer.trainer: Epoch 20/400 | LR=9.96e-04 | loss=1.9252 | val_dice=0.4457 | best=0.4679 (ep20) | 00:23:39 | L_main=0.7316 L_aux_1=1.0190(w=0.5) L_aux_2=0.9883(w=0.5)
[2026-06-22 15:25:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 20): 11666.1 MiB
[2026-06-22 15:26:26] INFO segtask_v1.trainer.validation:   Val: loss=1.1681, pooled_mean_dice=0.5819, per_class=['0.5819'], iou=0.4103, recall=0.4995, precision=0.6970, vol_sim=0.8349, mcc=0.5807, min_class_dice=0.5819, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.7588, per_class_sd=['0.7588'], combined(w=0.50)=0.6704, balanced=0.5936
[2026-06-22 15:26:30] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 15:26:30] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.5936 at epoch 21
[2026-06-22 15:26:30] INFO segtask_v1.trainer.trainer: Epoch 21/400 | LR=9.96e-04 | loss=1.9620 | val_dice=0.5819 | best=0.5936 (ep21) | 00:24:52 | L_main=0.7374 L_aux_1=1.0440(w=0.5) L_aux_2=1.0200(w=0.5)
[2026-06-22 15:26:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 21): 11666.1 MiB
[2026-06-22 15:27:39] INFO segtask_v1.trainer.validation:   Val: loss=1.1082, pooled_mean_dice=0.6423, per_class=['0.6423'], iou=0.4731, recall=0.6871, precision=0.6030, vol_sim=0.9348, mcc=0.6334, min_class_dice=0.6423, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.7869, per_class_sd=['0.7869'], combined(w=0.50)=0.7146, balanced=0.6487
[2026-06-22 15:27:43] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 15:27:43] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6487 at epoch 22
[2026-06-22 15:27:43] INFO segtask_v1.trainer.trainer: Epoch 22/400 | LR=9.95e-04 | loss=1.9245 | val_dice=0.6423 | best=0.6487 (ep22) | 00:26:05 | L_main=0.7264 L_aux_1=1.0203(w=0.5) L_aux_2=0.9963(w=0.5)
[2026-06-22 15:27:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 22): 11666.1 MiB
[2026-06-22 15:28:51] INFO segtask_v1.trainer.validation:   Val: loss=1.1246, pooled_mean_dice=0.6400, per_class=['0.6400'], iou=0.4706, recall=0.6966, precision=0.5919, vol_sim=0.9188, mcc=0.6335, min_class_dice=0.6400, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.7799, per_class_sd=['0.7799'], combined(w=0.50)=0.7099, balanced=0.6458
[2026-06-22 15:28:51] INFO segtask_v1.trainer.trainer: Epoch 23/400 | LR=9.95e-04 | loss=1.9094 | val_dice=0.6400 | best=0.6487 (ep22) | 00:27:13 | L_main=0.7413 L_aux_1=0.9960(w=0.5) L_aux_2=0.9706(w=0.5)
[2026-06-22 15:28:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 23): 11666.1 MiB
[2026-06-22 15:30:01] INFO segtask_v1.trainer.validation:   Val: loss=1.1075, pooled_mean_dice=0.6225, per_class=['0.6225'], iou=0.4519, recall=0.7522, precision=0.5310, vol_sim=0.8276, mcc=0.6212, min_class_dice=0.6225, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7551, per_class_sd=['0.7551'], combined(w=0.50)=0.6888, balanced=0.6272
[2026-06-22 15:30:01] INFO segtask_v1.trainer.trainer: Epoch 24/400 | LR=9.94e-04 | loss=1.8719 | val_dice=0.6225 | best=0.6487 (ep22) | 00:28:22 | L_main=0.7208 L_aux_1=0.9872(w=0.5) L_aux_2=0.9754(w=0.5)
[2026-06-22 15:30:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 24): 11666.1 MiB
[2026-06-22 15:31:09] INFO segtask_v1.trainer.validation:   Val: loss=1.0976, pooled_mean_dice=0.6409, per_class=['0.6409'], iou=0.4715, recall=0.7978, precision=0.5355, vol_sim=0.8033, mcc=0.6456, min_class_dice=0.6409, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.7878, per_class_sd=['0.7878'], combined(w=0.50)=0.7143, balanced=0.6487
[2026-06-22 15:31:09] INFO segtask_v1.trainer.trainer: Epoch 25/400 | LR=9.94e-04 | loss=1.8681 | val_dice=0.6409 | best=0.6487 (ep22) | 00:29:31 | L_main=0.7231 L_aux_1=0.9813(w=0.5) L_aux_2=0.9674(w=0.5)
[2026-06-22 15:31:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 25): 11666.1 MiB
[2026-06-22 15:32:18] INFO segtask_v1.trainer.validation:   Val: loss=1.0419, pooled_mean_dice=0.6140, per_class=['0.6140'], iou=0.4430, recall=0.8392, precision=0.4841, vol_sim=0.7317, mcc=0.6261, min_class_dice=0.6140, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.7633, per_class_sd=['0.7633'], combined(w=0.50)=0.6886, balanced=0.6225
[2026-06-22 15:32:18] INFO segtask_v1.trainer.trainer: Epoch 26/400 | LR=9.93e-04 | loss=1.8346 | val_dice=0.6140 | best=0.6487 (ep22) | 00:30:39 | L_main=0.7095 L_aux_1=0.9639(w=0.5) L_aux_2=0.9484(w=0.5)
[2026-06-22 15:32:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 26): 11666.1 MiB
[2026-06-22 15:33:26] INFO segtask_v1.trainer.validation:   Val: loss=1.0776, pooled_mean_dice=0.6181, per_class=['0.6181'], iou=0.4472, recall=0.8811, precision=0.4760, vol_sim=0.7014, mcc=0.6384, min_class_dice=0.6181, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.7759, per_class_sd=['0.7759'], combined(w=0.50)=0.6970, balanced=0.6286
[2026-06-22 15:33:26] INFO segtask_v1.trainer.trainer: Epoch 27/400 | LR=9.92e-04 | loss=1.8841 | val_dice=0.6181 | best=0.6487 (ep22) | 00:31:48 | L_main=0.7319 L_aux_1=0.9882(w=0.5) L_aux_2=0.9797(w=0.5)
[2026-06-22 15:33:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 27): 11666.1 MiB
[2026-06-22 15:34:35] INFO segtask_v1.trainer.validation:   Val: loss=1.0872, pooled_mean_dice=0.6055, per_class=['0.6055'], iou=0.4342, recall=0.8620, precision=0.4667, vol_sim=0.7025, mcc=0.6228, min_class_dice=0.6055, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.7445, per_class_sd=['0.7445'], combined(w=0.50)=0.6750, balanced=0.6126
[2026-06-22 15:34:35] INFO segtask_v1.trainer.trainer: Epoch 28/400 | LR=9.92e-04 | loss=1.7971 | val_dice=0.6055 | best=0.6487 (ep22) | 00:32:56 | L_main=0.7066 L_aux_1=0.9191(w=0.5) L_aux_2=0.9171(w=0.5)
[2026-06-22 15:34:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 28): 11666.1 MiB
[2026-06-22 15:35:44] INFO segtask_v1.trainer.validation:   Val: loss=1.0295, pooled_mean_dice=0.6045, per_class=['0.6045'], iou=0.4332, recall=0.8806, precision=0.4602, vol_sim=0.6865, mcc=0.6262, min_class_dice=0.6045, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.7646, per_class_sd=['0.7646'], combined(w=0.50)=0.6846, balanced=0.6154
[2026-06-22 15:35:44] INFO segtask_v1.trainer.trainer: Epoch 29/400 | LR=9.91e-04 | loss=1.7944 | val_dice=0.6045 | best=0.6487 (ep22) | 00:34:05 | L_main=0.7018 L_aux_1=0.9175(w=0.5) L_aux_2=0.9270(w=0.5)
[2026-06-22 15:35:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 29): 11666.1 MiB
[2026-06-22 15:36:53] INFO segtask_v1.trainer.validation:   Val: loss=1.1139, pooled_mean_dice=0.5981, per_class=['0.5981'], iou=0.4267, recall=0.7700, precision=0.4890, vol_sim=0.7767, mcc=0.6001, min_class_dice=0.5981, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.6934, per_class_sd=['0.6934'], combined(w=0.50)=0.6458, balanced=0.5963
[2026-06-22 15:36:53] INFO segtask_v1.trainer.trainer: Epoch 30/400 | LR=9.90e-04 | loss=1.9124 | val_dice=0.5981 | best=0.6487 (ep22) | 00:35:15 | L_main=0.7615 L_aux_1=0.9613(w=0.5) L_aux_2=0.9852(w=0.5)
[2026-06-22 15:36:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 30): 11666.1 MiB
[2026-06-22 15:38:04] INFO segtask_v1.trainer.validation:   Val: loss=1.1389, pooled_mean_dice=0.5453, per_class=['0.5453'], iou=0.3748, recall=0.8029, precision=0.4128, vol_sim=0.6791, mcc=0.5633, min_class_dice=0.5453, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.6880, per_class_sd=['0.6880'], combined(w=0.50)=0.6166, balanced=0.5529
[2026-06-22 15:38:04] INFO segtask_v1.trainer.trainer: Epoch 31/400 | LR=9.89e-04 | loss=1.7897 | val_dice=0.5453 | best=0.6487 (ep22) | 00:36:25 | L_main=0.6925 L_aux_1=0.9100(w=0.5) L_aux_2=0.9519(w=0.5)
[2026-06-22 15:38:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 31): 11666.1 MiB
[2026-06-22 15:39:13] INFO segtask_v1.trainer.validation:   Val: loss=1.1171, pooled_mean_dice=0.6296, per_class=['0.6296'], iou=0.4594, recall=0.7115, precision=0.5646, vol_sim=0.8849, mcc=0.6224, min_class_dice=0.6296, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.6918, per_class_sd=['0.6918'], combined(w=0.50)=0.6607, balanced=0.6208
[2026-06-22 15:39:13] INFO segtask_v1.trainer.trainer: Epoch 32/400 | LR=9.89e-04 | loss=1.9104 | val_dice=0.6296 | best=0.6487 (ep22) | 00:37:35 | L_main=0.7621 L_aux_1=0.9462(w=0.5) L_aux_2=1.0004(w=0.5)
[2026-06-22 15:39:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 32): 11666.1 MiB
[2026-06-22 15:40:22] INFO segtask_v1.trainer.validation:   Val: loss=1.1751, pooled_mean_dice=0.5891, per_class=['0.5891'], iou=0.4176, recall=0.6058, precision=0.5734, vol_sim=0.9725, mcc=0.5805, min_class_dice=0.5891, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.6670, per_class_sd=['0.6670'], combined(w=0.50)=0.6280, balanced=0.5834
[2026-06-22 15:40:22] INFO segtask_v1.trainer.trainer: Epoch 33/400 | LR=9.88e-04 | loss=1.7528 | val_dice=0.5891 | best=0.6487 (ep22) | 00:38:44 | L_main=0.6891 L_aux_1=0.8669(w=0.5) L_aux_2=0.9288(w=0.5)
[2026-06-22 15:40:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 33): 11666.1 MiB
[2026-06-22 15:41:32] INFO segtask_v1.trainer.validation:   Val: loss=1.0957, pooled_mean_dice=0.6248, per_class=['0.6248'], iou=0.4544, recall=0.7271, precision=0.5477, vol_sim=0.8593, mcc=0.6212, min_class_dice=0.6248, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.7260, per_class_sd=['0.7260'], combined(w=0.50)=0.6754, balanced=0.6238
[2026-06-22 15:41:32] INFO segtask_v1.trainer.trainer: Epoch 34/400 | LR=9.87e-04 | loss=1.7679 | val_dice=0.6248 | best=0.6487 (ep22) | 00:39:53 | L_main=0.6963 L_aux_1=0.8681(w=0.5) L_aux_2=0.9439(w=0.5)
[2026-06-22 15:41:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 34): 11666.1 MiB
[2026-06-22 15:42:41] INFO segtask_v1.trainer.validation:   Val: loss=1.0789, pooled_mean_dice=0.5982, per_class=['0.5982'], iou=0.4268, recall=0.7472, precision=0.4988, vol_sim=0.8007, mcc=0.6002, min_class_dice=0.5982, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.7188, per_class_sd=['0.7188'], combined(w=0.50)=0.6585, balanced=0.6010
[2026-06-22 15:42:41] INFO segtask_v1.trainer.trainer: Epoch 35/400 | LR=9.86e-04 | loss=1.7749 | val_dice=0.5982 | best=0.6487 (ep22) | 00:41:02 | L_main=0.7054 L_aux_1=0.8523(w=0.5) L_aux_2=0.9482(w=0.5)
[2026-06-22 15:42:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 35): 11666.1 MiB
[2026-06-22 15:43:51] INFO segtask_v1.trainer.validation:   Val: loss=0.9513, pooled_mean_dice=0.6325, per_class=['0.6325'], iou=0.4625, recall=0.8496, precision=0.5038, vol_sim=0.7445, mcc=0.6446, min_class_dice=0.6325, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7773, per_class_sd=['0.7773'], combined(w=0.50)=0.7049, balanced=0.6404
[2026-06-22 15:43:51] INFO segtask_v1.trainer.trainer: Epoch 36/400 | LR=9.85e-04 | loss=1.7267 | val_dice=0.6325 | best=0.6487 (ep22) | 00:42:12 | L_main=0.6951 L_aux_1=0.8130(w=0.5) L_aux_2=0.9135(w=0.5)
[2026-06-22 15:43:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 36): 11666.1 MiB
[2026-06-22 15:45:01] INFO segtask_v1.trainer.validation:   Val: loss=1.0086, pooled_mean_dice=0.6074, per_class=['0.6074'], iou=0.4361, recall=0.8405, precision=0.4755, vol_sim=0.7226, mcc=0.6212, min_class_dice=0.6074, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7309, per_class_sd=['0.7309'], combined(w=0.50)=0.6691, balanced=0.6115
[2026-06-22 15:45:01] INFO segtask_v1.trainer.trainer: Epoch 37/400 | LR=9.84e-04 | loss=1.8161 | val_dice=0.6074 | best=0.6487 (ep22) | 00:43:22 | L_main=0.7431 L_aux_1=0.8391(w=0.5) L_aux_2=0.9575(w=0.5)
[2026-06-22 15:45:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 37): 11666.1 MiB
[2026-06-22 15:46:10] INFO segtask_v1.trainer.validation:   Val: loss=1.0749, pooled_mean_dice=0.5879, per_class=['0.5879'], iou=0.4163, recall=0.8375, precision=0.4529, vol_sim=0.7019, mcc=0.6056, min_class_dice=0.5879, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.7135, per_class_sd=['0.7135'], combined(w=0.50)=0.6507, balanced=0.5926
[2026-06-22 15:46:10] INFO segtask_v1.trainer.trainer: Epoch 38/400 | LR=9.83e-04 | loss=1.7772 | val_dice=0.5879 | best=0.6487 (ep22) | 00:44:32 | L_main=0.7327 L_aux_1=0.8137(w=0.5) L_aux_2=0.9203(w=0.5)
[2026-06-22 15:46:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 38): 11666.1 MiB
[2026-06-22 15:47:20] INFO segtask_v1.trainer.validation:   Val: loss=1.0253, pooled_mean_dice=0.5980, per_class=['0.5980'], iou=0.4266, recall=0.8417, precision=0.4638, vol_sim=0.7105, mcc=0.6133, min_class_dice=0.5980, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7072, per_class_sd=['0.7072'], combined(w=0.50)=0.6526, balanced=0.5997
[2026-06-22 15:47:20] INFO segtask_v1.trainer.trainer: Epoch 39/400 | LR=9.82e-04 | loss=1.6676 | val_dice=0.5980 | best=0.6487 (ep22) | 00:45:42 | L_main=0.6766 L_aux_1=0.7614(w=0.5) L_aux_2=0.8840(w=0.5)
[2026-06-22 15:47:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 39): 11666.1 MiB
[2026-06-22 15:48:30] INFO segtask_v1.trainer.validation:   Val: loss=1.0135, pooled_mean_dice=0.6121, per_class=['0.6121'], iou=0.4411, recall=0.8759, precision=0.4705, vol_sim=0.6989, mcc=0.6299, min_class_dice=0.6121, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.7249, per_class_sd=['0.7249'], combined(w=0.50)=0.6685, balanced=0.6147
[2026-06-22 15:48:30] INFO segtask_v1.trainer.trainer: Epoch 40/400 | LR=9.81e-04 | loss=1.6674 | val_dice=0.6121 | best=0.6487 (ep22) | 00:46:52 | L_main=0.6834 L_aux_1=0.7539(w=0.5) L_aux_2=0.8784(w=0.5)
[2026-06-22 15:48:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 40): 11666.1 MiB
[2026-06-22 15:49:41] INFO segtask_v1.trainer.validation:   Val: loss=1.0698, pooled_mean_dice=0.5787, per_class=['0.5787'], iou=0.4072, recall=0.8682, precision=0.4340, vol_sim=0.6666, mcc=0.6040, min_class_dice=0.5787, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.7265, per_class_sd=['0.7265'], combined(w=0.50)=0.6526, balanced=0.5876
[2026-06-22 15:49:41] INFO segtask_v1.trainer.trainer: Epoch 41/400 | LR=9.80e-04 | loss=1.6777 | val_dice=0.5787 | best=0.6487 (ep22) | 00:48:03 | L_main=0.6881 L_aux_1=0.7506(w=0.5) L_aux_2=0.8875(w=0.5)
[2026-06-22 15:49:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 41): 11666.1 MiB
[2026-06-22 15:50:52] INFO segtask_v1.trainer.validation:   Val: loss=1.0517, pooled_mean_dice=0.5917, per_class=['0.5917'], iou=0.4201, recall=0.8836, precision=0.4447, vol_sim=0.6696, mcc=0.6155, min_class_dice=0.5917, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.7125, per_class_sd=['0.7125'], combined(w=0.50)=0.6521, balanced=0.5960
[2026-06-22 15:50:52] INFO segtask_v1.trainer.trainer: Epoch 42/400 | LR=9.79e-04 | loss=1.6822 | val_dice=0.5917 | best=0.6487 (ep22) | 00:49:13 | L_main=0.6885 L_aux_1=0.7524(w=0.5) L_aux_2=0.8921(w=0.5)
[2026-06-22 15:50:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 42): 11666.1 MiB
[2026-06-22 15:52:03] INFO segtask_v1.trainer.validation:   Val: loss=1.0453, pooled_mean_dice=0.5778, per_class=['0.5778'], iou=0.4063, recall=0.9110, precision=0.4231, vol_sim=0.6343, mcc=0.6077, min_class_dice=0.5778, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.7138, per_class_sd=['0.7138'], combined(w=0.50)=0.6458, balanced=0.5851
[2026-06-22 15:52:03] INFO segtask_v1.trainer.trainer: Epoch 43/400 | LR=9.77e-04 | loss=1.6664 | val_dice=0.5778 | best=0.6487 (ep22) | 00:50:25 | L_main=0.6784 L_aux_1=0.7475(w=0.5) L_aux_2=0.8956(w=0.5)
[2026-06-22 15:52:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 43): 11666.1 MiB
[2026-06-22 15:53:22] INFO segtask_v1.trainer.validation:   Val: loss=1.0131, pooled_mean_dice=0.6010, per_class=['0.6010'], iou=0.4296, recall=0.8927, precision=0.4530, vol_sim=0.6733, mcc=0.6232, min_class_dice=0.6010, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.7108, per_class_sd=['0.7108'], combined(w=0.50)=0.6559, balanced=0.6033
[2026-06-22 15:53:22] INFO segtask_v1.trainer.trainer: Epoch 44/400 | LR=9.76e-04 | loss=1.7193 | val_dice=0.6010 | best=0.6487 (ep22) | 00:51:44 | L_main=0.7074 L_aux_1=0.7731(w=0.5) L_aux_2=0.9076(w=0.5)
[2026-06-22 15:53:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 44): 11666.1 MiB
[2026-06-22 15:54:36] INFO segtask_v1.trainer.validation:   Val: loss=1.0383, pooled_mean_dice=0.5974, per_class=['0.5974'], iou=0.4259, recall=0.8661, precision=0.4559, vol_sim=0.6897, mcc=0.6170, min_class_dice=0.5974, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.7038, per_class_sd=['0.7038'], combined(w=0.50)=0.6506, balanced=0.5988
[2026-06-22 15:54:36] INFO segtask_v1.trainer.trainer: Epoch 45/400 | LR=9.75e-04 | loss=1.7345 | val_dice=0.5974 | best=0.6487 (ep22) | 00:52:58 | L_main=0.7148 L_aux_1=0.7773(w=0.5) L_aux_2=0.9258(w=0.5)
[2026-06-22 15:54:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 45): 11666.1 MiB
[2026-06-22 15:55:55] INFO segtask_v1.trainer.validation:   Val: loss=1.0137, pooled_mean_dice=0.6026, per_class=['0.6026'], iou=0.4313, recall=0.8914, precision=0.4552, vol_sim=0.6760, mcc=0.6260, min_class_dice=0.6026, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.7246, per_class_sd=['0.7246'], combined(w=0.50)=0.6636, balanced=0.6072
[2026-06-22 15:55:55] INFO segtask_v1.trainer.trainer: Epoch 46/400 | LR=9.74e-04 | loss=1.6540 | val_dice=0.6026 | best=0.6487 (ep22) | 00:54:17 | L_main=0.6756 L_aux_1=0.7380(w=0.5) L_aux_2=0.8856(w=0.5)
[2026-06-22 15:55:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 46): 11666.1 MiB
[2026-06-22 15:57:06] INFO segtask_v1.trainer.validation:   Val: loss=1.0116, pooled_mean_dice=0.5716, per_class=['0.5716'], iou=0.4001, recall=0.8866, precision=0.4217, vol_sim=0.6447, mcc=0.6004, min_class_dice=0.5716, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.7121, per_class_sd=['0.7121'], combined(w=0.50)=0.6418, balanced=0.5795
[2026-06-22 15:57:06] INFO segtask_v1.trainer.trainer: Epoch 47/400 | LR=9.72e-04 | loss=1.7260 | val_dice=0.5716 | best=0.6487 (ep22) | 00:55:27 | L_main=0.7188 L_aux_1=0.7692(w=0.5) L_aux_2=0.8945(w=0.5)
[2026-06-22 15:57:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 47): 11666.1 MiB
[2026-06-22 15:58:15] INFO segtask_v1.trainer.validation:   Val: loss=1.0331, pooled_mean_dice=0.5559, per_class=['0.5559'], iou=0.3850, recall=0.8838, precision=0.4055, vol_sim=0.6290, mcc=0.5888, min_class_dice=0.5559, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.7157, per_class_sd=['0.7157'], combined(w=0.50)=0.6358, balanced=0.5672
[2026-06-22 15:58:15] INFO segtask_v1.trainer.trainer: Epoch 48/400 | LR=9.71e-04 | loss=1.6363 | val_dice=0.5559 | best=0.6487 (ep22) | 00:56:37 | L_main=0.6741 L_aux_1=0.7285(w=0.5) L_aux_2=0.8595(w=0.5)
[2026-06-22 15:58:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 48): 11666.1 MiB
[2026-06-22 15:59:25] INFO segtask_v1.trainer.validation:   Val: loss=1.0298, pooled_mean_dice=0.5986, per_class=['0.5986'], iou=0.4271, recall=0.9080, precision=0.4464, vol_sim=0.6592, mcc=0.6268, min_class_dice=0.5986, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.7262, per_class_sd=['0.7262'], combined(w=0.50)=0.6624, balanced=0.6044
[2026-06-22 15:59:25] INFO segtask_v1.trainer.trainer: Epoch 49/400 | LR=9.70e-04 | loss=1.6945 | val_dice=0.5986 | best=0.6487 (ep22) | 00:57:46 | L_main=0.7026 L_aux_1=0.7540(w=0.5) L_aux_2=0.8956(w=0.5)
[2026-06-22 15:59:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 49): 11666.1 MiB
[2026-06-22 16:00:35] INFO segtask_v1.trainer.validation:   Val: loss=1.0037, pooled_mean_dice=0.5823, per_class=['0.5823'], iou=0.4107, recall=0.9138, precision=0.4273, vol_sim=0.6372, mcc=0.6137, min_class_dice=0.5823, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.7380, per_class_sd=['0.7380'], combined(w=0.50)=0.6602, balanced=0.5929
[2026-06-22 16:00:35] INFO segtask_v1.trainer.trainer: Epoch 50/400 | LR=9.68e-04 | loss=1.6001 | val_dice=0.5823 | best=0.6487 (ep22) | 00:58:57 | L_main=0.6583 L_aux_1=0.7054(w=0.5) L_aux_2=0.8459(w=0.5)
[2026-06-22 16:00:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 50): 11666.1 MiB
[2026-06-22 16:01:45] INFO segtask_v1.trainer.validation:   Val: loss=1.0192, pooled_mean_dice=0.5601, per_class=['0.5601'], iou=0.3890, recall=0.9307, precision=0.4006, vol_sim=0.6018, mcc=0.5966, min_class_dice=0.5601, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.7097, per_class_sd=['0.7097'], combined(w=0.50)=0.6349, balanced=0.5700
[2026-06-22 16:01:45] INFO segtask_v1.trainer.trainer: Epoch 51/400 | LR=9.67e-04 | loss=1.5912 | val_dice=0.5601 | best=0.6487 (ep22) | 01:00:07 | L_main=0.6568 L_aux_1=0.7045(w=0.5) L_aux_2=0.8365(w=0.5)
[2026-06-22 16:01:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 51): 11666.1 MiB
[2026-06-22 16:02:56] INFO segtask_v1.trainer.validation:   Val: loss=0.9841, pooled_mean_dice=0.5630, per_class=['0.5630'], iou=0.3918, recall=0.9314, precision=0.4035, vol_sim=0.6045, mcc=0.6017, min_class_dice=0.5630, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.7425, per_class_sd=['0.7425'], combined(w=0.50)=0.6528, balanced=0.5777
[2026-06-22 16:02:56] INFO segtask_v1.trainer.trainer: Epoch 52/400 | LR=9.66e-04 | loss=1.5756 | val_dice=0.5630 | best=0.6487 (ep22) | 01:01:17 | L_main=0.6456 L_aux_1=0.6996(w=0.5) L_aux_2=0.8381(w=0.5)
[2026-06-22 16:02:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 52): 11666.1 MiB
[2026-06-22 16:04:08] INFO segtask_v1.trainer.validation:   Val: loss=0.9867, pooled_mean_dice=0.5691, per_class=['0.5691'], iou=0.3977, recall=0.9424, precision=0.4076, vol_sim=0.6038, mcc=0.6064, min_class_dice=0.5691, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.7377, per_class_sd=['0.7377'], combined(w=0.50)=0.6534, balanced=0.5820
[2026-06-22 16:04:08] INFO segtask_v1.trainer.trainer: Epoch 53/400 | LR=9.64e-04 | loss=1.5977 | val_dice=0.5691 | best=0.6487 (ep22) | 01:02:30 | L_main=0.6645 L_aux_1=0.7051(w=0.5) L_aux_2=0.8243(w=0.5)
[2026-06-22 16:04:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 53): 11666.1 MiB
[2026-06-22 16:05:17] INFO segtask_v1.trainer.validation:   Val: loss=0.9701, pooled_mean_dice=0.5479, per_class=['0.5479'], iou=0.3773, recall=0.9488, precision=0.3852, vol_sim=0.5774, mcc=0.5930, min_class_dice=0.5479, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.7457, per_class_sd=['0.7457'], combined(w=0.50)=0.6468, balanced=0.5656
[2026-06-22 16:05:17] INFO segtask_v1.trainer.trainer: Epoch 54/400 | LR=9.63e-04 | loss=1.6574 | val_dice=0.5479 | best=0.6487 (ep22) | 01:03:39 | L_main=0.6861 L_aux_1=0.7345(w=0.5) L_aux_2=0.8722(w=0.5)
[2026-06-22 16:05:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 54): 11666.1 MiB
[2026-06-22 16:06:28] INFO segtask_v1.trainer.validation:   Val: loss=0.9374, pooled_mean_dice=0.5668, per_class=['0.5668'], iou=0.3955, recall=0.9382, precision=0.4061, vol_sim=0.6042, mcc=0.6080, min_class_dice=0.5668, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7789, per_class_sd=['0.7789'], combined(w=0.50)=0.6729, balanced=0.5865
[2026-06-22 16:06:28] INFO segtask_v1.trainer.trainer: Epoch 55/400 | LR=9.61e-04 | loss=1.5924 | val_dice=0.5668 | best=0.6487 (ep22) | 01:04:49 | L_main=0.6596 L_aux_1=0.7019(w=0.5) L_aux_2=0.8341(w=0.5)
[2026-06-22 16:06:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 55): 11666.1 MiB
[2026-06-22 16:07:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8650, pooled_mean_dice=0.6199, per_class=['0.6199'], iou=0.4492, recall=0.9504, precision=0.4600, vol_sim=0.6523, mcc=0.6483, min_class_dice=0.6199, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.7784, per_class_sd=['0.7784'], combined(w=0.50)=0.6992, balanced=0.6312
[2026-06-22 16:07:45] INFO segtask_v1.trainer.trainer: Epoch 56/400 | LR=9.59e-04 | loss=1.6175 | val_dice=0.6199 | best=0.6487 (ep22) | 01:06:06 | L_main=0.6690 L_aux_1=0.7098(w=0.5) L_aux_2=0.8532(w=0.5)
[2026-06-22 16:07:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 56): 11666.1 MiB
[2026-06-22 16:08:54] INFO segtask_v1.trainer.validation:   Val: loss=0.9066, pooled_mean_dice=0.5989, per_class=['0.5989'], iou=0.4274, recall=0.9608, precision=0.4350, vol_sim=0.6233, mcc=0.6347, min_class_dice=0.5989, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.7754, per_class_sd=['0.7754'], combined(w=0.50)=0.6871, balanced=0.6132
[2026-06-22 16:08:54] INFO segtask_v1.trainer.trainer: Epoch 57/400 | LR=9.58e-04 | loss=1.6255 | val_dice=0.5989 | best=0.6487 (ep22) | 01:07:15 | L_main=0.6811 L_aux_1=0.7110(w=0.5) L_aux_2=0.8434(w=0.5)
[2026-06-22 16:08:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 57): 11666.1 MiB
[2026-06-22 16:10:04] INFO segtask_v1.trainer.validation:   Val: loss=0.8699, pooled_mean_dice=0.6093, per_class=['0.6093'], iou=0.4381, recall=0.9550, precision=0.4473, vol_sim=0.6380, mcc=0.6426, min_class_dice=0.6093, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.7986, per_class_sd=['0.7986'], combined(w=0.50)=0.7039, balanced=0.6256
[2026-06-22 16:10:04] INFO segtask_v1.trainer.trainer: Epoch 58/400 | LR=9.56e-04 | loss=1.6520 | val_dice=0.6093 | best=0.6487 (ep22) | 01:08:26 | L_main=0.6902 L_aux_1=0.7241(w=0.5) L_aux_2=0.8560(w=0.5)
[2026-06-22 16:10:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 58): 11666.1 MiB
[2026-06-22 16:11:16] INFO segtask_v1.trainer.validation:   Val: loss=0.9219, pooled_mean_dice=0.5965, per_class=['0.5965'], iou=0.4250, recall=0.9563, precision=0.4334, vol_sim=0.6237, mcc=0.6333, min_class_dice=0.5965, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.7976, per_class_sd=['0.7976'], combined(w=0.50)=0.6970, balanced=0.6147
[2026-06-22 16:11:16] INFO segtask_v1.trainer.trainer: Epoch 59/400 | LR=9.55e-04 | loss=1.5969 | val_dice=0.5965 | best=0.6487 (ep22) | 01:09:37 | L_main=0.6654 L_aux_1=0.7002(w=0.5) L_aux_2=0.8320(w=0.5)
[2026-06-22 16:11:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 59): 11666.1 MiB
[2026-06-22 16:12:25] INFO segtask_v1.trainer.validation:   Val: loss=0.9130, pooled_mean_dice=0.5936, per_class=['0.5936'], iou=0.4221, recall=0.9605, precision=0.4295, vol_sim=0.6180, mcc=0.6324, min_class_dice=0.5936, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7952, per_class_sd=['0.7952'], combined(w=0.50)=0.6944, balanced=0.6119
[2026-06-22 16:12:25] INFO segtask_v1.trainer.trainer: Epoch 60/400 | LR=9.53e-04 | loss=1.6707 | val_dice=0.5936 | best=0.6487 (ep22) | 01:10:46 | L_main=0.7015 L_aux_1=0.7329(w=0.5) L_aux_2=0.8748(w=0.5)
[2026-06-22 16:12:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 60): 11666.1 MiB
[2026-06-22 16:13:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8924, pooled_mean_dice=0.5819, per_class=['0.5819'], iou=0.4103, recall=0.9533, precision=0.4187, vol_sim=0.6104, mcc=0.6194, min_class_dice=0.5819, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.7793, per_class_sd=['0.7793'], combined(w=0.50)=0.6806, balanced=0.5993
[2026-06-22 16:13:36] INFO segtask_v1.trainer.trainer: Epoch 61/400 | LR=9.51e-04 | loss=1.6459 | val_dice=0.5819 | best=0.6487 (ep22) | 01:11:57 | L_main=0.6935 L_aux_1=0.7188(w=0.5) L_aux_2=0.8487(w=0.5)
[2026-06-22 16:13:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 61): 11666.1 MiB
[2026-06-22 16:14:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8915, pooled_mean_dice=0.5960, per_class=['0.5960'], iou=0.4245, recall=0.9570, precision=0.4328, vol_sim=0.6228, mcc=0.6328, min_class_dice=0.5960, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.7845, per_class_sd=['0.7845'], combined(w=0.50)=0.6903, balanced=0.6123
[2026-06-22 16:14:45] INFO segtask_v1.trainer.trainer: Epoch 62/400 | LR=9.50e-04 | loss=1.5954 | val_dice=0.5960 | best=0.6487 (ep22) | 01:13:07 | L_main=0.6687 L_aux_1=0.6914(w=0.5) L_aux_2=0.8259(w=0.5)
[2026-06-22 16:14:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 62): 11666.1 MiB
[2026-06-22 16:15:54] INFO segtask_v1.trainer.validation:   Val: loss=0.9434, pooled_mean_dice=0.5623, per_class=['0.5623'], iou=0.3911, recall=0.9504, precision=0.3992, vol_sim=0.5916, mcc=0.6075, min_class_dice=0.5623, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7934, per_class_sd=['0.7934'], combined(w=0.50)=0.6778, balanced=0.5848
[2026-06-22 16:15:54] INFO segtask_v1.trainer.trainer: Epoch 63/400 | LR=9.48e-04 | loss=1.6901 | val_dice=0.5623 | best=0.6487 (ep22) | 01:14:16 | L_main=0.7220 L_aux_1=0.7338(w=0.5) L_aux_2=0.8506(w=0.5)
[2026-06-22 16:15:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 63): 11666.1 MiB
[2026-06-22 16:17:04] INFO segtask_v1.trainer.validation:   Val: loss=0.9087, pooled_mean_dice=0.5988, per_class=['0.5988'], iou=0.4273, recall=0.9468, precision=0.4378, vol_sim=0.6324, mcc=0.6329, min_class_dice=0.5988, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.7770, per_class_sd=['0.7770'], combined(w=0.50)=0.6879, balanced=0.6133
[2026-06-22 16:17:04] INFO segtask_v1.trainer.trainer: Epoch 64/400 | LR=9.46e-04 | loss=1.6650 | val_dice=0.5988 | best=0.6487 (ep22) | 01:15:26 | L_main=0.7075 L_aux_1=0.7237(w=0.5) L_aux_2=0.8593(w=0.5)
[2026-06-22 16:17:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 64): 11666.1 MiB
[2026-06-22 16:18:14] INFO segtask_v1.trainer.validation:   Val: loss=0.8694, pooled_mean_dice=0.6115, per_class=['0.6115'], iou=0.4404, recall=0.9579, precision=0.4491, vol_sim=0.6384, mcc=0.6444, min_class_dice=0.6115, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.7637, per_class_sd=['0.7637'], combined(w=0.50)=0.6876, balanced=0.6220
[2026-06-22 16:18:14] INFO segtask_v1.trainer.trainer: Epoch 65/400 | LR=9.44e-04 | loss=1.6341 | val_dice=0.6115 | best=0.6487 (ep22) | 01:16:35 | L_main=0.6900 L_aux_1=0.7102(w=0.5) L_aux_2=0.8372(w=0.5)
[2026-06-22 16:18:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 65): 11666.1 MiB
[2026-06-22 16:19:27] INFO segtask_v1.trainer.validation:   Val: loss=0.9118, pooled_mean_dice=0.5676, per_class=['0.5676'], iou=0.3962, recall=0.9549, precision=0.4038, vol_sim=0.5944, mcc=0.6101, min_class_dice=0.5676, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.7676, per_class_sd=['0.7676'], combined(w=0.50)=0.6676, balanced=0.5856
[2026-06-22 16:19:27] INFO segtask_v1.trainer.trainer: Epoch 66/400 | LR=9.42e-04 | loss=1.5408 | val_dice=0.5676 | best=0.6487 (ep22) | 01:17:48 | L_main=0.6448 L_aux_1=0.6639(w=0.5) L_aux_2=0.8066(w=0.5)
[2026-06-22 16:19:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 66): 11666.1 MiB
[2026-06-22 16:20:38] INFO segtask_v1.trainer.validation:   Val: loss=0.9049, pooled_mean_dice=0.5915, per_class=['0.5915'], iou=0.4199, recall=0.9568, precision=0.4281, vol_sim=0.6182, mcc=0.6281, min_class_dice=0.5915, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.7749, per_class_sd=['0.7749'], combined(w=0.50)=0.6832, balanced=0.6069
[2026-06-22 16:20:38] INFO segtask_v1.trainer.trainer: Epoch 67/400 | LR=9.40e-04 | loss=1.5799 | val_dice=0.5915 | best=0.6487 (ep22) | 01:18:59 | L_main=0.6660 L_aux_1=0.6786(w=0.5) L_aux_2=0.8165(w=0.5)
[2026-06-22 16:20:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 67): 11666.1 MiB
[2026-06-22 16:21:47] INFO segtask_v1.trainer.validation:   Val: loss=0.9056, pooled_mean_dice=0.5950, per_class=['0.5950'], iou=0.4234, recall=0.9580, precision=0.4315, vol_sim=0.6210, mcc=0.6321, min_class_dice=0.5950, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.7893, per_class_sd=['0.7893'], combined(w=0.50)=0.6922, balanced=0.6121
[2026-06-22 16:21:47] INFO segtask_v1.trainer.trainer: Epoch 68/400 | LR=9.39e-04 | loss=1.5583 | val_dice=0.5950 | best=0.6487 (ep22) | 01:20:09 | L_main=0.6599 L_aux_1=0.6658(w=0.5) L_aux_2=0.7970(w=0.5)
[2026-06-22 16:21:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 68): 11666.1 MiB
[2026-06-22 16:22:57] INFO segtask_v1.trainer.validation:   Val: loss=0.9331, pooled_mean_dice=0.5684, per_class=['0.5684'], iou=0.3971, recall=0.9666, precision=0.4026, vol_sim=0.5881, mcc=0.6116, min_class_dice=0.5684, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.7789, per_class_sd=['0.7789'], combined(w=0.50)=0.6737, balanced=0.5880
[2026-06-22 16:22:57] INFO segtask_v1.trainer.trainer: Epoch 69/400 | LR=9.37e-04 | loss=1.5700 | val_dice=0.5684 | best=0.6487 (ep22) | 01:21:19 | L_main=0.6646 L_aux_1=0.6680(w=0.5) L_aux_2=0.8060(w=0.5)
[2026-06-22 16:22:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 69): 11666.1 MiB
[2026-06-22 16:24:07] INFO segtask_v1.trainer.validation:   Val: loss=0.8826, pooled_mean_dice=0.6007, per_class=['0.6007'], iou=0.4292, recall=0.9707, precision=0.4349, vol_sim=0.6188, mcc=0.6386, min_class_dice=0.6007, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.7966, per_class_sd=['0.7966'], combined(w=0.50)=0.6986, balanced=0.6182
[2026-06-22 16:24:07] INFO segtask_v1.trainer.trainer: Epoch 70/400 | LR=9.35e-04 | loss=1.5763 | val_dice=0.6007 | best=0.6487 (ep22) | 01:22:29 | L_main=0.6651 L_aux_1=0.6740(w=0.5) L_aux_2=0.8158(w=0.5)
[2026-06-22 16:24:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 70): 11666.1 MiB
[2026-06-22 16:25:22] INFO segtask_v1.trainer.validation:   Val: loss=0.9358, pooled_mean_dice=0.5783, per_class=['0.5783'], iou=0.4068, recall=0.9655, precision=0.4128, vol_sim=0.5990, mcc=0.6208, min_class_dice=0.5783, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7908, per_class_sd=['0.7908'], combined(w=0.50)=0.6846, balanced=0.5983
[2026-06-22 16:25:22] INFO segtask_v1.trainer.trainer: Epoch 71/400 | LR=9.33e-04 | loss=1.5365 | val_dice=0.5783 | best=0.6487 (ep22) | 01:23:44 | L_main=0.6398 L_aux_1=0.6730(w=0.5) L_aux_2=0.7984(w=0.5)
[2026-06-22 16:25:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 71): 11666.1 MiB
[2026-06-22 16:26:31] INFO segtask_v1.trainer.validation:   Val: loss=0.9136, pooled_mean_dice=0.5649, per_class=['0.5649'], iou=0.3936, recall=0.9714, precision=0.3983, vol_sim=0.5816, mcc=0.6113, min_class_dice=0.5649, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7929, per_class_sd=['0.7929'], combined(w=0.50)=0.6789, balanced=0.5871
[2026-06-22 16:26:31] INFO segtask_v1.trainer.trainer: Epoch 72/400 | LR=9.31e-04 | loss=1.5733 | val_dice=0.5649 | best=0.6487 (ep22) | 01:24:52 | L_main=0.6624 L_aux_1=0.6874(w=0.5) L_aux_2=0.8030(w=0.5)
[2026-06-22 16:26:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 72): 11666.1 MiB
[2026-06-22 16:27:41] INFO segtask_v1.trainer.validation:   Val: loss=0.8885, pooled_mean_dice=0.6074, per_class=['0.6074'], iou=0.4362, recall=0.9662, precision=0.4429, vol_sim=0.6286, mcc=0.6441, min_class_dice=0.6074, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.8072, per_class_sd=['0.8072'], combined(w=0.50)=0.7073, balanced=0.6255
[2026-06-22 16:27:41] INFO segtask_v1.trainer.trainer: Epoch 73/400 | LR=9.29e-04 | loss=1.5725 | val_dice=0.6074 | best=0.6487 (ep22) | 01:26:02 | L_main=0.6647 L_aux_1=0.6775(w=0.5) L_aux_2=0.8036(w=0.5)
[2026-06-22 16:27:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 73): 11666.1 MiB
[2026-06-22 16:28:50] INFO segtask_v1.trainer.validation:   Val: loss=0.9546, pooled_mean_dice=0.5659, per_class=['0.5659'], iou=0.3946, recall=0.9588, precision=0.4014, vol_sim=0.5902, mcc=0.6121, min_class_dice=0.5659, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8023, per_class_sd=['0.8023'], combined(w=0.50)=0.6841, balanced=0.5893
[2026-06-22 16:28:50] INFO segtask_v1.trainer.trainer: Epoch 74/400 | LR=9.27e-04 | loss=1.5531 | val_dice=0.5659 | best=0.6487 (ep22) | 01:27:12 | L_main=0.6529 L_aux_1=0.6668(w=0.5) L_aux_2=0.8104(w=0.5)
[2026-06-22 16:28:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 74): 11666.1 MiB
[2026-06-22 16:30:00] INFO segtask_v1.trainer.validation:   Val: loss=0.8621, pooled_mean_dice=0.6031, per_class=['0.6031'], iou=0.4317, recall=0.9588, precision=0.4399, vol_sim=0.6290, mcc=0.6378, min_class_dice=0.6031, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.7916, per_class_sd=['0.7916'], combined(w=0.50)=0.6974, balanced=0.6193
[2026-06-22 16:30:00] INFO segtask_v1.trainer.trainer: Epoch 75/400 | LR=9.25e-04 | loss=1.5328 | val_dice=0.6031 | best=0.6487 (ep22) | 01:28:22 | L_main=0.6467 L_aux_1=0.6510(w=0.5) L_aux_2=0.7949(w=0.5)
[2026-06-22 16:30:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 75): 11666.1 MiB
[2026-06-22 16:31:08] INFO segtask_v1.trainer.validation:   Val: loss=0.8692, pooled_mean_dice=0.6099, per_class=['0.6099'], iou=0.4387, recall=0.9696, precision=0.4448, vol_sim=0.6290, mcc=0.6464, min_class_dice=0.6099, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8126, per_class_sd=['0.8126'], combined(w=0.50)=0.7112, balanced=0.6285
[2026-06-22 16:31:08] INFO segtask_v1.trainer.trainer: Epoch 76/400 | LR=9.22e-04 | loss=1.5420 | val_dice=0.6099 | best=0.6487 (ep22) | 01:29:30 | L_main=0.6509 L_aux_1=0.6564(w=0.5) L_aux_2=0.7985(w=0.5)
[2026-06-22 16:31:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 76): 11666.1 MiB
[2026-06-22 16:32:18] INFO segtask_v1.trainer.validation:   Val: loss=0.8866, pooled_mean_dice=0.6147, per_class=['0.6147'], iou=0.4437, recall=0.9541, precision=0.4534, vol_sim=0.6443, mcc=0.6487, min_class_dice=0.6147, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8051, per_class_sd=['0.8051'], combined(w=0.50)=0.7099, balanced=0.6314
[2026-06-22 16:32:18] INFO segtask_v1.trainer.trainer: Epoch 77/400 | LR=9.20e-04 | loss=1.6075 | val_dice=0.6147 | best=0.6487 (ep22) | 01:30:40 | L_main=0.6866 L_aux_1=0.6999(w=0.5) L_aux_2=0.8112(w=0.5)
[2026-06-22 16:32:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 77): 11666.1 MiB
[2026-06-22 16:33:28] INFO segtask_v1.trainer.validation:   Val: loss=0.8584, pooled_mean_dice=0.6224, per_class=['0.6224'], iou=0.4518, recall=0.9724, precision=0.4576, vol_sim=0.6400, mcc=0.6569, min_class_dice=0.6224, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8086, per_class_sd=['0.8086'], combined(w=0.50)=0.7155, balanced=0.6386
[2026-06-22 16:33:28] INFO segtask_v1.trainer.trainer: Epoch 78/400 | LR=9.18e-04 | loss=1.5512 | val_dice=0.6224 | best=0.6487 (ep22) | 01:31:49 | L_main=0.6584 L_aux_1=0.6723(w=0.5) L_aux_2=0.7818(w=0.5)
[2026-06-22 16:33:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 78): 11666.1 MiB
[2026-06-22 16:34:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8828, pooled_mean_dice=0.6047, per_class=['0.6047'], iou=0.4334, recall=0.9597, precision=0.4414, vol_sim=0.6301, mcc=0.6419, min_class_dice=0.6047, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8178, per_class_sd=['0.8178'], combined(w=0.50)=0.7112, balanced=0.6248
[2026-06-22 16:34:37] INFO segtask_v1.trainer.trainer: Epoch 79/400 | LR=9.16e-04 | loss=1.5411 | val_dice=0.6047 | best=0.6487 (ep22) | 01:32:58 | L_main=0.6543 L_aux_1=0.6589(w=0.5) L_aux_2=0.7886(w=0.5)
[2026-06-22 16:34:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 79): 11666.1 MiB
[2026-06-22 16:35:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8695, pooled_mean_dice=0.5964, per_class=['0.5964'], iou=0.4249, recall=0.9669, precision=0.4311, vol_sim=0.6167, mcc=0.6338, min_class_dice=0.5964, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.7955, per_class_sd=['0.7955'], combined(w=0.50)=0.6959, balanced=0.6143
[2026-06-22 16:35:47] INFO segtask_v1.trainer.trainer: Epoch 80/400 | LR=9.14e-04 | loss=1.5575 | val_dice=0.5964 | best=0.6487 (ep22) | 01:34:08 | L_main=0.6607 L_aux_1=0.6651(w=0.5) L_aux_2=0.7948(w=0.5)
[2026-06-22 16:35:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 80): 11666.1 MiB
[2026-06-22 16:36:57] INFO segtask_v1.trainer.validation:   Val: loss=0.8352, pooled_mean_dice=0.6192, per_class=['0.6192'], iou=0.4484, recall=0.9669, precision=0.4554, vol_sim=0.6404, mcc=0.6535, min_class_dice=0.6192, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8111, per_class_sd=['0.8111'], combined(w=0.50)=0.7152, balanced=0.6362
[2026-06-22 16:36:57] INFO segtask_v1.trainer.trainer: Epoch 81/400 | LR=9.11e-04 | loss=1.5019 | val_dice=0.6192 | best=0.6487 (ep22) | 01:35:18 | L_main=0.6339 L_aux_1=0.6396(w=0.5) L_aux_2=0.7775(w=0.5)
[2026-06-22 16:36:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 81): 11666.1 MiB
[2026-06-22 16:38:06] INFO segtask_v1.trainer.validation:   Val: loss=0.8483, pooled_mean_dice=0.6276, per_class=['0.6276'], iou=0.4573, recall=0.9648, precision=0.4650, vol_sim=0.6505, mcc=0.6594, min_class_dice=0.6276, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8055, per_class_sd=['0.8055'], combined(w=0.50)=0.7165, balanced=0.6424
[2026-06-22 16:38:06] INFO segtask_v1.trainer.trainer: Epoch 82/400 | LR=9.09e-04 | loss=1.5699 | val_dice=0.6276 | best=0.6487 (ep22) | 01:36:28 | L_main=0.6679 L_aux_1=0.6760(w=0.5) L_aux_2=0.7968(w=0.5)
[2026-06-22 16:38:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 82): 11666.1 MiB
[2026-06-22 16:39:16] INFO segtask_v1.trainer.validation:   Val: loss=0.8775, pooled_mean_dice=0.6002, per_class=['0.6002'], iou=0.4288, recall=0.9611, precision=0.4364, vol_sim=0.6245, mcc=0.6386, min_class_dice=0.6002, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8277, per_class_sd=['0.8277'], combined(w=0.50)=0.7140, balanced=0.6224
[2026-06-22 16:39:16] INFO segtask_v1.trainer.trainer: Epoch 83/400 | LR=9.07e-04 | loss=1.5145 | val_dice=0.6002 | best=0.6487 (ep22) | 01:37:38 | L_main=0.6414 L_aux_1=0.6412(w=0.5) L_aux_2=0.7811(w=0.5)
[2026-06-22 16:39:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 83): 11666.1 MiB
[2026-06-22 16:40:25] INFO segtask_v1.trainer.validation:   Val: loss=0.8894, pooled_mean_dice=0.6198, per_class=['0.6198'], iou=0.4490, recall=0.9540, precision=0.4589, vol_sim=0.6496, mcc=0.6552, min_class_dice=0.6198, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8450, per_class_sd=['0.8450'], combined(w=0.50)=0.7324, balanced=0.6418
[2026-06-22 16:40:25] INFO segtask_v1.trainer.trainer: Epoch 84/400 | LR=9.05e-04 | loss=1.4990 | val_dice=0.6198 | best=0.6487 (ep22) | 01:38:46 | L_main=0.6334 L_aux_1=0.6423(w=0.5) L_aux_2=0.7612(w=0.5)
[2026-06-22 16:40:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 84): 11666.1 MiB
[2026-06-22 16:41:34] INFO segtask_v1.trainer.validation:   Val: loss=0.8630, pooled_mean_dice=0.6146, per_class=['0.6146'], iou=0.4437, recall=0.9719, precision=0.4494, vol_sim=0.6324, mcc=0.6505, min_class_dice=0.6146, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8115, per_class_sd=['0.8115'], combined(w=0.50)=0.7131, balanced=0.6324
[2026-06-22 16:41:34] INFO segtask_v1.trainer.trainer: Epoch 85/400 | LR=9.02e-04 | loss=1.5407 | val_dice=0.6146 | best=0.6487 (ep22) | 01:39:56 | L_main=0.6541 L_aux_1=0.6609(w=0.5) L_aux_2=0.7809(w=0.5)
[2026-06-22 16:41:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 85): 11666.1 MiB
[2026-06-22 16:42:43] INFO segtask_v1.trainer.validation:   Val: loss=0.8815, pooled_mean_dice=0.6008, per_class=['0.6008'], iou=0.4294, recall=0.9665, precision=0.4359, vol_sim=0.6216, mcc=0.6384, min_class_dice=0.6008, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8000, per_class_sd=['0.8000'], combined(w=0.50)=0.7004, balanced=0.6189
[2026-06-22 16:42:43] INFO segtask_v1.trainer.trainer: Epoch 86/400 | LR=9.00e-04 | loss=1.5182 | val_dice=0.6008 | best=0.6487 (ep22) | 01:41:05 | L_main=0.6424 L_aux_1=0.6450(w=0.5) L_aux_2=0.7839(w=0.5)
[2026-06-22 16:42:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 86): 11666.1 MiB
[2026-06-22 16:43:52] INFO segtask_v1.trainer.validation:   Val: loss=0.8746, pooled_mean_dice=0.6328, per_class=['0.6328'], iou=0.4629, recall=0.9641, precision=0.4710, vol_sim=0.6564, mcc=0.6646, min_class_dice=0.6328, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8245, per_class_sd=['0.8245'], combined(w=0.50)=0.7287, balanced=0.6499
[2026-06-22 16:43:56] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 16:43:56] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6499 at epoch 87
[2026-06-22 16:43:56] INFO segtask_v1.trainer.trainer: Epoch 87/400 | LR=8.97e-04 | loss=1.5004 | val_dice=0.6328 | best=0.6499 (ep87) | 01:42:18 | L_main=0.6320 L_aux_1=0.6381(w=0.5) L_aux_2=0.7794(w=0.5)
[2026-06-22 16:43:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 87): 11666.1 MiB
[2026-06-22 16:45:07] INFO segtask_v1.trainer.validation:   Val: loss=0.8784, pooled_mean_dice=0.6012, per_class=['0.6012'], iou=0.4298, recall=0.9711, precision=0.4354, vol_sim=0.6191, mcc=0.6408, min_class_dice=0.6012, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8138, per_class_sd=['0.8138'], combined(w=0.50)=0.7075, balanced=0.6214
[2026-06-22 16:45:07] INFO segtask_v1.trainer.trainer: Epoch 88/400 | LR=8.95e-04 | loss=1.5209 | val_dice=0.6012 | best=0.6499 (ep87) | 01:43:29 | L_main=0.6406 L_aux_1=0.6516(w=0.5) L_aux_2=0.7834(w=0.5)
[2026-06-22 16:45:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 88): 11666.1 MiB
[2026-06-22 16:46:16] INFO segtask_v1.trainer.validation:   Val: loss=0.8954, pooled_mean_dice=0.6120, per_class=['0.6120'], iou=0.4409, recall=0.9643, precision=0.4482, vol_sim=0.6346, mcc=0.6489, min_class_dice=0.6120, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8205, per_class_sd=['0.8205'], combined(w=0.50)=0.7162, balanced=0.6315
[2026-06-22 16:46:16] INFO segtask_v1.trainer.trainer: Epoch 89/400 | LR=8.93e-04 | loss=1.5304 | val_dice=0.6120 | best=0.6499 (ep87) | 01:44:38 | L_main=0.6455 L_aux_1=0.6604(w=0.5) L_aux_2=0.7803(w=0.5)
[2026-06-22 16:46:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 89): 11666.1 MiB
[2026-06-22 16:47:27] INFO segtask_v1.trainer.validation:   Val: loss=0.8200, pooled_mean_dice=0.6311, per_class=['0.6311'], iou=0.4610, recall=0.9648, precision=0.4689, vol_sim=0.6541, mcc=0.6621, min_class_dice=0.6311, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8267, per_class_sd=['0.8267'], combined(w=0.50)=0.7289, balanced=0.6487
[2026-06-22 16:47:27] INFO segtask_v1.trainer.trainer: Epoch 90/400 | LR=8.90e-04 | loss=1.5835 | val_dice=0.6311 | best=0.6499 (ep87) | 01:45:48 | L_main=0.6812 L_aux_1=0.6844(w=0.5) L_aux_2=0.7811(w=0.5)
[2026-06-22 16:47:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 90): 11666.1 MiB
[2026-06-22 16:48:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8169, pooled_mean_dice=0.6391, per_class=['0.6391'], iou=0.4697, recall=0.9705, precision=0.4765, vol_sim=0.6586, mcc=0.6693, min_class_dice=0.6391, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8165, per_class_sd=['0.8165'], combined(w=0.50)=0.7278, balanced=0.6540
[2026-06-22 16:48:41] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 16:48:41] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6540 at epoch 91
[2026-06-22 16:48:41] INFO segtask_v1.trainer.trainer: Epoch 91/400 | LR=8.88e-04 | loss=1.4757 | val_dice=0.6391 | best=0.6540 (ep91) | 01:47:03 | L_main=0.6218 L_aux_1=0.6323(w=0.5) L_aux_2=0.7573(w=0.5)
[2026-06-22 16:48:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 91): 11666.1 MiB
[2026-06-22 16:49:53] INFO segtask_v1.trainer.validation:   Val: loss=0.8280, pooled_mean_dice=0.6213, per_class=['0.6213'], iou=0.4506, recall=0.9683, precision=0.4573, vol_sim=0.6416, mcc=0.6556, min_class_dice=0.6213, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.8199, per_class_sd=['0.8199'], combined(w=0.50)=0.7206, balanced=0.6393
[2026-06-22 16:49:53] INFO segtask_v1.trainer.trainer: Epoch 92/400 | LR=8.85e-04 | loss=1.5020 | val_dice=0.6213 | best=0.6540 (ep91) | 01:48:14 | L_main=0.6323 L_aux_1=0.6372(w=0.5) L_aux_2=0.7768(w=0.5)
[2026-06-22 16:49:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 92): 11666.1 MiB
[2026-06-22 16:51:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8557, pooled_mean_dice=0.6163, per_class=['0.6163'], iou=0.4454, recall=0.9662, precision=0.4525, vol_sim=0.6379, mcc=0.6506, min_class_dice=0.6163, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8081, per_class_sd=['0.8081'], combined(w=0.50)=0.7122, balanced=0.6333
[2026-06-22 16:51:03] INFO segtask_v1.trainer.trainer: Epoch 93/400 | LR=8.83e-04 | loss=1.5052 | val_dice=0.6163 | best=0.6540 (ep91) | 01:49:25 | L_main=0.6366 L_aux_1=0.6415(w=0.5) L_aux_2=0.7717(w=0.5)
[2026-06-22 16:51:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 93): 11666.1 MiB
[2026-06-22 16:52:16] INFO segtask_v1.trainer.validation:   Val: loss=0.8602, pooled_mean_dice=0.6023, per_class=['0.6023'], iou=0.4309, recall=0.9734, precision=0.4361, vol_sim=0.6188, mcc=0.6414, min_class_dice=0.6023, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8000, per_class_sd=['0.8000'], combined(w=0.50)=0.7012, balanced=0.6202
[2026-06-22 16:52:16] INFO segtask_v1.trainer.trainer: Epoch 94/400 | LR=8.80e-04 | loss=1.5106 | val_dice=0.6023 | best=0.6540 (ep91) | 01:50:37 | L_main=0.6371 L_aux_1=0.6469(w=0.5) L_aux_2=0.7793(w=0.5)
[2026-06-22 16:52:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 94): 11666.1 MiB
[2026-06-22 16:53:27] INFO segtask_v1.trainer.validation:   Val: loss=0.8778, pooled_mean_dice=0.6260, per_class=['0.6260'], iou=0.4556, recall=0.9718, precision=0.4618, vol_sim=0.6442, mcc=0.6590, min_class_dice=0.6260, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8123, per_class_sd=['0.8123'], combined(w=0.50)=0.7192, balanced=0.6422
[2026-06-22 16:53:27] INFO segtask_v1.trainer.trainer: Epoch 95/400 | LR=8.77e-04 | loss=1.5222 | val_dice=0.6260 | best=0.6540 (ep91) | 01:51:48 | L_main=0.6491 L_aux_1=0.6524(w=0.5) L_aux_2=0.7639(w=0.5)
[2026-06-22 16:53:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 95): 11666.1 MiB
[2026-06-22 16:54:38] INFO segtask_v1.trainer.validation:   Val: loss=0.8571, pooled_mean_dice=0.6306, per_class=['0.6306'], iou=0.4605, recall=0.9799, precision=0.4649, vol_sim=0.6436, mcc=0.6646, min_class_dice=0.6306, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8196, per_class_sd=['0.8196'], combined(w=0.50)=0.7251, balanced=0.6474
[2026-06-22 16:54:38] INFO segtask_v1.trainer.trainer: Epoch 96/400 | LR=8.75e-04 | loss=1.5273 | val_dice=0.6306 | best=0.6540 (ep91) | 01:53:00 | L_main=0.6483 L_aux_1=0.6533(w=0.5) L_aux_2=0.7750(w=0.5)
[2026-06-22 16:54:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 96): 11666.1 MiB
[2026-06-22 16:55:49] INFO segtask_v1.trainer.validation:   Val: loss=0.8564, pooled_mean_dice=0.6350, per_class=['0.6350'], iou=0.4652, recall=0.9577, precision=0.4750, vol_sim=0.6631, mcc=0.6668, min_class_dice=0.6350, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8462, per_class_sd=['0.8462'], combined(w=0.50)=0.7406, balanced=0.6551
[2026-06-22 16:55:53] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 16:55:53] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6551 at epoch 97
[2026-06-22 16:55:53] INFO segtask_v1.trainer.trainer: Epoch 97/400 | LR=8.72e-04 | loss=1.4806 | val_dice=0.6350 | best=0.6551 (ep97) | 01:54:15 | L_main=0.6268 L_aux_1=0.6293(w=0.5) L_aux_2=0.7587(w=0.5)
[2026-06-22 16:55:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 97): 11666.1 MiB
[2026-06-22 16:57:04] INFO segtask_v1.trainer.validation:   Val: loss=0.8274, pooled_mean_dice=0.6253, per_class=['0.6253'], iou=0.4549, recall=0.9726, precision=0.4608, vol_sim=0.6429, mcc=0.6582, min_class_dice=0.6253, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8188, per_class_sd=['0.8188'], combined(w=0.50)=0.7220, balanced=0.6426
[2026-06-22 16:57:04] INFO segtask_v1.trainer.trainer: Epoch 98/400 | LR=8.69e-04 | loss=1.4829 | val_dice=0.6253 | best=0.6551 (ep97) | 01:55:25 | L_main=0.6272 L_aux_1=0.6295(w=0.5) L_aux_2=0.7635(w=0.5)
[2026-06-22 16:57:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 98): 11666.1 MiB
[2026-06-22 16:58:14] INFO segtask_v1.trainer.validation:   Val: loss=0.8379, pooled_mean_dice=0.6441, per_class=['0.6441'], iou=0.4750, recall=0.9738, precision=0.4811, vol_sim=0.6614, mcc=0.6729, min_class_dice=0.6441, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8238, per_class_sd=['0.8238'], combined(w=0.50)=0.7340, balanced=0.6593
[2026-06-22 16:58:18] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 16:58:18] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6593 at epoch 99
[2026-06-22 16:58:18] INFO segtask_v1.trainer.trainer: Epoch 99/400 | LR=8.67e-04 | loss=1.5109 | val_dice=0.6441 | best=0.6593 (ep99) | 01:56:39 | L_main=0.6427 L_aux_1=0.6397(w=0.5) L_aux_2=0.7763(w=0.5)
[2026-06-22 16:58:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 99): 11666.1 MiB
[2026-06-22 16:59:28] INFO segtask_v1.trainer.validation:   Val: loss=0.8417, pooled_mean_dice=0.6153, per_class=['0.6153'], iou=0.4444, recall=0.9649, precision=0.4517, vol_sim=0.6377, mcc=0.6499, min_class_dice=0.6153, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8192, per_class_sd=['0.8192'], combined(w=0.50)=0.7173, balanced=0.6341
[2026-06-22 16:59:28] INFO segtask_v1.trainer.trainer: Epoch 100/400 | LR=8.64e-04 | loss=1.5233 | val_dice=0.6153 | best=0.6593 (ep99) | 01:57:50 | L_main=0.6457 L_aux_1=0.6524(w=0.5) L_aux_2=0.7729(w=0.5)
[2026-06-22 16:59:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 100): 11666.1 MiB
[2026-06-22 17:00:40] INFO segtask_v1.trainer.validation:   Val: loss=0.8213, pooled_mean_dice=0.6511, per_class=['0.6511'], iou=0.4827, recall=0.9678, precision=0.4905, vol_sim=0.6728, mcc=0.6797, min_class_dice=0.6511, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8308, per_class_sd=['0.8308'], combined(w=0.50)=0.7409, balanced=0.6664
[2026-06-22 17:00:44] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 17:00:44] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6664 at epoch 101
[2026-06-22 17:00:44] INFO segtask_v1.trainer.trainer: Epoch 101/400 | LR=8.61e-04 | loss=1.5295 | val_dice=0.6511 | best=0.6664 (ep101) | 01:59:05 | L_main=0.6516 L_aux_1=0.6604(w=0.5) L_aux_2=0.7678(w=0.5)
[2026-06-22 17:00:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 101): 11666.1 MiB
[2026-06-22 17:01:54] INFO segtask_v1.trainer.validation:   Val: loss=0.8317, pooled_mean_dice=0.6265, per_class=['0.6265'], iou=0.4562, recall=0.9726, precision=0.4621, vol_sim=0.6442, mcc=0.6588, min_class_dice=0.6265, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8044, per_class_sd=['0.8044'], combined(w=0.50)=0.7155, balanced=0.6413
[2026-06-22 17:01:54] INFO segtask_v1.trainer.trainer: Epoch 102/400 | LR=8.59e-04 | loss=1.5152 | val_dice=0.6265 | best=0.6664 (ep101) | 02:00:15 | L_main=0.6404 L_aux_1=0.6474(w=0.5) L_aux_2=0.7766(w=0.5)
[2026-06-22 17:01:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 102): 11666.1 MiB
[2026-06-22 17:03:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8688, pooled_mean_dice=0.6099, per_class=['0.6099'], iou=0.4388, recall=0.9749, precision=0.4438, vol_sim=0.6256, mcc=0.6492, min_class_dice=0.6099, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8139, per_class_sd=['0.8139'], combined(w=0.50)=0.7119, balanced=0.6290
[2026-06-22 17:03:03] INFO segtask_v1.trainer.trainer: Epoch 103/400 | LR=8.56e-04 | loss=1.5484 | val_dice=0.6099 | best=0.6664 (ep101) | 02:01:25 | L_main=0.6592 L_aux_1=0.6651(w=0.5) L_aux_2=0.7888(w=0.5)
[2026-06-22 17:03:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 103): 11666.1 MiB
[2026-06-22 17:04:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8951, pooled_mean_dice=0.5982, per_class=['0.5982'], iou=0.4267, recall=0.9720, precision=0.4320, vol_sim=0.6154, mcc=0.6403, min_class_dice=0.5982, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.8121, per_class_sd=['0.8121'], combined(w=0.50)=0.7051, balanced=0.6186
[2026-06-22 17:04:12] INFO segtask_v1.trainer.trainer: Epoch 104/400 | LR=8.53e-04 | loss=1.5030 | val_dice=0.5982 | best=0.6664 (ep101) | 02:02:34 | L_main=0.6388 L_aux_1=0.6469(w=0.5) L_aux_2=0.7584(w=0.5)
[2026-06-22 17:04:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 104): 11666.1 MiB
[2026-06-22 17:05:23] INFO segtask_v1.trainer.validation:   Val: loss=0.8509, pooled_mean_dice=0.6435, per_class=['0.6435'], iou=0.4744, recall=0.9733, precision=0.4806, vol_sim=0.6611, mcc=0.6754, min_class_dice=0.6435, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8234, per_class_sd=['0.8234'], combined(w=0.50)=0.7334, balanced=0.6590
[2026-06-22 17:05:23] INFO segtask_v1.trainer.trainer: Epoch 105/400 | LR=8.50e-04 | loss=1.4853 | val_dice=0.6435 | best=0.6664 (ep101) | 02:03:44 | L_main=0.6326 L_aux_1=0.6351(w=0.5) L_aux_2=0.7476(w=0.5)
[2026-06-22 17:05:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 105): 11666.1 MiB
[2026-06-22 17:06:34] INFO segtask_v1.trainer.validation:   Val: loss=0.8471, pooled_mean_dice=0.5976, per_class=['0.5976'], iou=0.4261, recall=0.9782, precision=0.4302, vol_sim=0.6109, mcc=0.6393, min_class_dice=0.5976, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.8134, per_class_sd=['0.8134'], combined(w=0.50)=0.7055, balanced=0.6183
[2026-06-22 17:06:34] INFO segtask_v1.trainer.trainer: Epoch 106/400 | LR=8.47e-04 | loss=1.4901 | val_dice=0.5976 | best=0.6664 (ep101) | 02:04:55 | L_main=0.6357 L_aux_1=0.6346(w=0.5) L_aux_2=0.7505(w=0.5)
[2026-06-22 17:06:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 106): 11666.1 MiB
[2026-06-22 17:07:43] INFO segtask_v1.trainer.validation:   Val: loss=0.8497, pooled_mean_dice=0.6181, per_class=['0.6181'], iou=0.4473, recall=0.9819, precision=0.4510, vol_sim=0.6295, mcc=0.6544, min_class_dice=0.6181, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.7975, per_class_sd=['0.7975'], combined(w=0.50)=0.7078, balanced=0.6333
[2026-06-22 17:07:43] INFO segtask_v1.trainer.trainer: Epoch 107/400 | LR=8.44e-04 | loss=1.4937 | val_dice=0.6181 | best=0.6664 (ep101) | 02:06:05 | L_main=0.6356 L_aux_1=0.6394(w=0.5) L_aux_2=0.7538(w=0.5)
[2026-06-22 17:07:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 107): 11666.1 MiB
[2026-06-22 17:08:53] INFO segtask_v1.trainer.validation:   Val: loss=0.8695, pooled_mean_dice=0.5963, per_class=['0.5963'], iou=0.4248, recall=0.9787, precision=0.4288, vol_sim=0.6093, mcc=0.6378, min_class_dice=0.5963, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8046, per_class_sd=['0.8046'], combined(w=0.50)=0.7004, balanced=0.6159
[2026-06-22 17:08:53] INFO segtask_v1.trainer.trainer: Epoch 108/400 | LR=8.42e-04 | loss=1.5210 | val_dice=0.5963 | best=0.6664 (ep101) | 02:07:15 | L_main=0.6545 L_aux_1=0.6511(w=0.5) L_aux_2=0.7500(w=0.5)
[2026-06-22 17:08:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 108): 11666.1 MiB
[2026-06-22 17:10:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8638, pooled_mean_dice=0.6211, per_class=['0.6211'], iou=0.4505, recall=0.9672, precision=0.4574, vol_sim=0.6422, mcc=0.6563, min_class_dice=0.6211, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8325, per_class_sd=['0.8325'], combined(w=0.50)=0.7268, balanced=0.6412
[2026-06-22 17:10:03] INFO segtask_v1.trainer.trainer: Epoch 109/400 | LR=8.39e-04 | loss=1.5197 | val_dice=0.6211 | best=0.6664 (ep101) | 02:08:25 | L_main=0.6532 L_aux_1=0.6508(w=0.5) L_aux_2=0.7562(w=0.5)
[2026-06-22 17:10:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 109): 11666.1 MiB
[2026-06-22 17:11:14] INFO segtask_v1.trainer.validation:   Val: loss=0.8686, pooled_mean_dice=0.6194, per_class=['0.6194'], iou=0.4487, recall=0.9788, precision=0.4531, vol_sim=0.6328, mcc=0.6558, min_class_dice=0.6194, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8066, per_class_sd=['0.8066'], combined(w=0.50)=0.7130, balanced=0.6358
[2026-06-22 17:11:14] INFO segtask_v1.trainer.trainer: Epoch 110/400 | LR=8.36e-04 | loss=1.4907 | val_dice=0.6194 | best=0.6664 (ep101) | 02:09:35 | L_main=0.6391 L_aux_1=0.6399(w=0.5) L_aux_2=0.7403(w=0.5)
[2026-06-22 17:11:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 110): 11666.1 MiB
[2026-06-22 17:12:25] INFO segtask_v1.trainer.validation:   Val: loss=0.8909, pooled_mean_dice=0.6135, per_class=['0.6135'], iou=0.4424, recall=0.9665, precision=0.4493, vol_sim=0.6347, mcc=0.6479, min_class_dice=0.6135, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.8098, per_class_sd=['0.8098'], combined(w=0.50)=0.7116, balanced=0.6310
[2026-06-22 17:12:25] INFO segtask_v1.trainer.trainer: Epoch 111/400 | LR=8.33e-04 | loss=1.5245 | val_dice=0.6135 | best=0.6664 (ep101) | 02:10:47 | L_main=0.6539 L_aux_1=0.6538(w=0.5) L_aux_2=0.7553(w=0.5)
[2026-06-22 17:12:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 111): 11666.1 MiB
[2026-06-22 17:13:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8409, pooled_mean_dice=0.6199, per_class=['0.6199'], iou=0.4492, recall=0.9719, precision=0.4551, vol_sim=0.6378, mcc=0.6540, min_class_dice=0.6199, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8179, per_class_sd=['0.8179'], combined(w=0.50)=0.7189, balanced=0.6378
[2026-06-22 17:13:36] INFO segtask_v1.trainer.trainer: Epoch 112/400 | LR=8.30e-04 | loss=1.5048 | val_dice=0.6199 | best=0.6664 (ep101) | 02:11:57 | L_main=0.6473 L_aux_1=0.6431(w=0.5) L_aux_2=0.7391(w=0.5)
[2026-06-22 17:13:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 112): 11666.1 MiB
[2026-06-22 17:14:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8658, pooled_mean_dice=0.6195, per_class=['0.6195'], iou=0.4488, recall=0.9621, precision=0.4568, vol_sim=0.6439, mcc=0.6528, min_class_dice=0.6195, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8186, per_class_sd=['0.8186'], combined(w=0.50)=0.7191, balanced=0.6375
[2026-06-22 17:14:47] INFO segtask_v1.trainer.trainer: Epoch 113/400 | LR=8.27e-04 | loss=1.5721 | val_dice=0.6195 | best=0.6664 (ep101) | 02:13:08 | L_main=0.6796 L_aux_1=0.6732(w=0.5) L_aux_2=0.7727(w=0.5)
[2026-06-22 17:14:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 113): 11666.1 MiB
[2026-06-22 17:15:56] INFO segtask_v1.trainer.validation:   Val: loss=0.8663, pooled_mean_dice=0.6191, per_class=['0.6191'], iou=0.4484, recall=0.9747, precision=0.4537, vol_sim=0.6352, mcc=0.6556, min_class_dice=0.6191, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8020, per_class_sd=['0.8020'], combined(w=0.50)=0.7106, balanced=0.6349
[2026-06-22 17:15:56] INFO segtask_v1.trainer.trainer: Epoch 114/400 | LR=8.24e-04 | loss=1.5270 | val_dice=0.6191 | best=0.6664 (ep101) | 02:14:18 | L_main=0.6484 L_aux_1=0.6500(w=0.5) L_aux_2=0.7762(w=0.5)
[2026-06-22 17:15:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 114): 11666.1 MiB
[2026-06-22 17:17:05] INFO segtask_v1.trainer.validation:   Val: loss=0.8362, pooled_mean_dice=0.6061, per_class=['0.6061'], iou=0.4348, recall=0.9785, precision=0.4391, vol_sim=0.6195, mcc=0.6441, min_class_dice=0.6061, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8022, per_class_sd=['0.8022'], combined(w=0.50)=0.7041, balanced=0.6238
[2026-06-22 17:17:05] INFO segtask_v1.trainer.trainer: Epoch 115/400 | LR=8.21e-04 | loss=1.5471 | val_dice=0.6061 | best=0.6664 (ep101) | 02:15:27 | L_main=0.6605 L_aux_1=0.6654(w=0.5) L_aux_2=0.7780(w=0.5)
[2026-06-22 17:17:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 115): 11666.1 MiB
[2026-06-22 17:18:15] INFO segtask_v1.trainer.validation:   Val: loss=0.8498, pooled_mean_dice=0.6342, per_class=['0.6342'], iou=0.4644, recall=0.9678, precision=0.4717, vol_sim=0.6554, mcc=0.6676, min_class_dice=0.6342, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8250, per_class_sd=['0.8250'], combined(w=0.50)=0.7296, balanced=0.6513
[2026-06-22 17:18:15] INFO segtask_v1.trainer.trainer: Epoch 116/400 | LR=8.18e-04 | loss=1.4899 | val_dice=0.6342 | best=0.6664 (ep101) | 02:16:37 | L_main=0.6373 L_aux_1=0.6400(w=0.5) L_aux_2=0.7362(w=0.5)
[2026-06-22 17:18:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 116): 11666.1 MiB
[2026-06-22 17:19:26] INFO segtask_v1.trainer.validation:   Val: loss=0.8426, pooled_mean_dice=0.6130, per_class=['0.6130'], iou=0.4419, recall=0.9701, precision=0.4480, vol_sim=0.6319, mcc=0.6503, min_class_dice=0.6130, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8199, per_class_sd=['0.8199'], combined(w=0.50)=0.7164, balanced=0.6323
[2026-06-22 17:19:26] INFO segtask_v1.trainer.trainer: Epoch 117/400 | LR=8.15e-04 | loss=1.4940 | val_dice=0.6130 | best=0.6664 (ep101) | 02:17:47 | L_main=0.6395 L_aux_1=0.6422(w=0.5) L_aux_2=0.7453(w=0.5)
[2026-06-22 17:19:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 117): 11666.1 MiB
[2026-06-22 17:20:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8745, pooled_mean_dice=0.5945, per_class=['0.5945'], iou=0.4230, recall=0.9625, precision=0.4301, vol_sim=0.6177, mcc=0.6344, min_class_dice=0.5945, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8186, per_class_sd=['0.8186'], combined(w=0.50)=0.7066, balanced=0.6162
[2026-06-22 17:20:36] INFO segtask_v1.trainer.trainer: Epoch 118/400 | LR=8.11e-04 | loss=1.4845 | val_dice=0.5945 | best=0.6664 (ep101) | 02:18:58 | L_main=0.6336 L_aux_1=0.6382(w=0.5) L_aux_2=0.7390(w=0.5)
[2026-06-22 17:20:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 118): 11666.1 MiB
[2026-06-22 17:21:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8721, pooled_mean_dice=0.6286, per_class=['0.6286'], iou=0.4584, recall=0.9576, precision=0.4679, vol_sim=0.6564, mcc=0.6607, min_class_dice=0.6286, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8269, per_class_sd=['0.8269'], combined(w=0.50)=0.7277, balanced=0.6466
[2026-06-22 17:21:45] INFO segtask_v1.trainer.trainer: Epoch 119/400 | LR=8.08e-04 | loss=1.4851 | val_dice=0.6286 | best=0.6664 (ep101) | 02:20:07 | L_main=0.6311 L_aux_1=0.6342(w=0.5) L_aux_2=0.7547(w=0.5)
[2026-06-22 17:21:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 119): 11666.1 MiB
[2026-06-22 17:23:01] INFO segtask_v1.trainer.validation:   Val: loss=0.8439, pooled_mean_dice=0.6007, per_class=['0.6007'], iou=0.4293, recall=0.9724, precision=0.4346, vol_sim=0.6178, mcc=0.6379, min_class_dice=0.6007, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8071, per_class_sd=['0.8071'], combined(w=0.50)=0.7039, balanced=0.6198
[2026-06-22 17:23:01] INFO segtask_v1.trainer.trainer: Epoch 120/400 | LR=8.05e-04 | loss=1.4822 | val_dice=0.6007 | best=0.6664 (ep101) | 02:21:23 | L_main=0.6301 L_aux_1=0.6332(w=0.5) L_aux_2=0.7482(w=0.5)
[2026-06-22 17:23:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 120): 11666.1 MiB
[2026-06-22 17:24:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8232, pooled_mean_dice=0.6370, per_class=['0.6370'], iou=0.4674, recall=0.9725, precision=0.4737, vol_sim=0.6551, mcc=0.6677, min_class_dice=0.6370, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8296, per_class_sd=['0.8296'], combined(w=0.50)=0.7333, balanced=0.6542
[2026-06-22 17:24:12] INFO segtask_v1.trainer.trainer: Epoch 121/400 | LR=8.02e-04 | loss=1.4930 | val_dice=0.6370 | best=0.6664 (ep101) | 02:22:33 | L_main=0.6337 L_aux_1=0.6366(w=0.5) L_aux_2=0.7586(w=0.5)
[2026-06-22 17:24:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 121): 11666.1 MiB
[2026-06-22 17:25:24] INFO segtask_v1.trainer.validation:   Val: loss=0.8500, pooled_mean_dice=0.6285, per_class=['0.6285'], iou=0.4583, recall=0.9769, precision=0.4633, vol_sim=0.6434, mcc=0.6645, min_class_dice=0.6285, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.8249, per_class_sd=['0.8249'], combined(w=0.50)=0.7267, balanced=0.6466
[2026-06-22 17:25:24] INFO segtask_v1.trainer.trainer: Epoch 122/400 | LR=7.99e-04 | loss=1.5508 | val_dice=0.6285 | best=0.6664 (ep101) | 02:23:45 | L_main=0.6669 L_aux_1=0.6672(w=0.5) L_aux_2=0.7737(w=0.5)
[2026-06-22 17:25:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 122): 11666.1 MiB
[2026-06-22 17:26:35] INFO segtask_v1.trainer.validation:   Val: loss=0.8395, pooled_mean_dice=0.6186, per_class=['0.6186'], iou=0.4478, recall=0.9722, precision=0.4536, vol_sim=0.6363, mcc=0.6531, min_class_dice=0.6186, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8178, per_class_sd=['0.8178'], combined(w=0.50)=0.7182, balanced=0.6368
[2026-06-22 17:26:35] INFO segtask_v1.trainer.trainer: Epoch 123/400 | LR=7.96e-04 | loss=1.4639 | val_dice=0.6186 | best=0.6664 (ep101) | 02:24:56 | L_main=0.6271 L_aux_1=0.6293(w=0.5) L_aux_2=0.7206(w=0.5)
[2026-06-22 17:26:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 123): 11666.1 MiB
[2026-06-22 17:27:44] INFO segtask_v1.trainer.validation:   Val: loss=0.8535, pooled_mean_dice=0.6223, per_class=['0.6223'], iou=0.4517, recall=0.9762, precision=0.4567, vol_sim=0.6375, mcc=0.6577, min_class_dice=0.6223, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8178, per_class_sd=['0.8178'], combined(w=0.50)=0.7200, balanced=0.6400
[2026-06-22 17:27:44] INFO segtask_v1.trainer.trainer: Epoch 124/400 | LR=7.92e-04 | loss=1.5004 | val_dice=0.6223 | best=0.6664 (ep101) | 02:26:06 | L_main=0.6373 L_aux_1=0.6530(w=0.5) L_aux_2=0.7559(w=0.5)
[2026-06-22 17:27:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 124): 11666.1 MiB
[2026-06-22 17:28:54] INFO segtask_v1.trainer.validation:   Val: loss=0.8279, pooled_mean_dice=0.6421, per_class=['0.6421'], iou=0.4729, recall=0.9693, precision=0.4801, vol_sim=0.6625, mcc=0.6715, min_class_dice=0.6421, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8217, per_class_sd=['0.8217'], combined(w=0.50)=0.7319, balanced=0.6573
[2026-06-22 17:28:54] INFO segtask_v1.trainer.trainer: Epoch 125/400 | LR=7.89e-04 | loss=1.4880 | val_dice=0.6421 | best=0.6664 (ep101) | 02:27:15 | L_main=0.6388 L_aux_1=0.6415(w=0.5) L_aux_2=0.7342(w=0.5)
[2026-06-22 17:28:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 125): 11666.1 MiB
[2026-06-22 17:30:05] INFO segtask_v1.trainer.validation:   Val: loss=0.8799, pooled_mean_dice=0.6264, per_class=['0.6264'], iou=0.4561, recall=0.9739, precision=0.4617, vol_sim=0.6432, mcc=0.6616, min_class_dice=0.6264, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8190, per_class_sd=['0.8190'], combined(w=0.50)=0.7227, balanced=0.6438
[2026-06-22 17:30:05] INFO segtask_v1.trainer.trainer: Epoch 126/400 | LR=7.86e-04 | loss=1.5251 | val_dice=0.6264 | best=0.6664 (ep101) | 02:28:26 | L_main=0.6549 L_aux_1=0.6560(w=0.5) L_aux_2=0.7543(w=0.5)
[2026-06-22 17:30:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 126): 11666.1 MiB
[2026-06-22 17:31:15] INFO segtask_v1.trainer.validation:   Val: loss=0.8385, pooled_mean_dice=0.6305, per_class=['0.6305'], iou=0.4603, recall=0.9636, precision=0.4685, vol_sim=0.6543, mcc=0.6624, min_class_dice=0.6305, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8337, per_class_sd=['0.8337'], combined(w=0.50)=0.7321, balanced=0.6493
[2026-06-22 17:31:15] INFO segtask_v1.trainer.trainer: Epoch 127/400 | LR=7.83e-04 | loss=1.4926 | val_dice=0.6305 | best=0.6664 (ep101) | 02:29:36 | L_main=0.6375 L_aux_1=0.6405(w=0.5) L_aux_2=0.7468(w=0.5)
[2026-06-22 17:31:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 127): 11666.1 MiB
[2026-06-22 17:32:24] INFO segtask_v1.trainer.validation:   Val: loss=0.8282, pooled_mean_dice=0.6389, per_class=['0.6389'], iou=0.4694, recall=0.9798, precision=0.4740, vol_sim=0.6521, mcc=0.6703, min_class_dice=0.6389, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8246, per_class_sd=['0.8246'], combined(w=0.50)=0.7318, balanced=0.6551
[2026-06-22 17:32:24] INFO segtask_v1.trainer.trainer: Epoch 128/400 | LR=7.79e-04 | loss=1.4639 | val_dice=0.6389 | best=0.6664 (ep101) | 02:30:46 | L_main=0.6243 L_aux_1=0.6302(w=0.5) L_aux_2=0.7331(w=0.5)
[2026-06-22 17:32:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 128): 11666.1 MiB
[2026-06-22 17:33:35] INFO segtask_v1.trainer.validation:   Val: loss=0.8316, pooled_mean_dice=0.6350, per_class=['0.6350'], iou=0.4652, recall=0.9762, precision=0.4705, vol_sim=0.6505, mcc=0.6697, min_class_dice=0.6350, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8408, per_class_sd=['0.8408'], combined(w=0.50)=0.7379, balanced=0.6545
[2026-06-22 17:33:35] INFO segtask_v1.trainer.trainer: Epoch 129/400 | LR=7.76e-04 | loss=1.4692 | val_dice=0.6350 | best=0.6664 (ep101) | 02:31:57 | L_main=0.6238 L_aux_1=0.6309(w=0.5) L_aux_2=0.7425(w=0.5)
[2026-06-22 17:33:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 129): 11666.1 MiB
[2026-06-22 17:34:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8270, pooled_mean_dice=0.6332, per_class=['0.6332'], iou=0.4633, recall=0.9716, precision=0.4697, vol_sim=0.6518, mcc=0.6654, min_class_dice=0.6332, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8199, per_class_sd=['0.8199'], combined(w=0.50)=0.7266, balanced=0.6496
[2026-06-22 17:34:45] INFO segtask_v1.trainer.trainer: Epoch 130/400 | LR=7.73e-04 | loss=1.4398 | val_dice=0.6332 | best=0.6664 (ep101) | 02:33:07 | L_main=0.6164 L_aux_1=0.6144(w=0.5) L_aux_2=0.7088(w=0.5)
[2026-06-22 17:34:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 130): 11666.1 MiB
[2026-06-22 17:36:00] INFO segtask_v1.trainer.validation:   Val: loss=0.8435, pooled_mean_dice=0.6300, per_class=['0.6300'], iou=0.4598, recall=0.9806, precision=0.4641, vol_sim=0.6425, mcc=0.6647, min_class_dice=0.6300, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8233, per_class_sd=['0.8233'], combined(w=0.50)=0.7266, balanced=0.6475
[2026-06-22 17:36:00] INFO segtask_v1.trainer.trainer: Epoch 131/400 | LR=7.69e-04 | loss=1.4542 | val_dice=0.6300 | best=0.6664 (ep101) | 02:34:22 | L_main=0.6213 L_aux_1=0.6212(w=0.5) L_aux_2=0.7307(w=0.5)
[2026-06-22 17:36:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 131): 11666.1 MiB
[2026-06-22 17:37:13] INFO segtask_v1.trainer.validation:   Val: loss=0.8738, pooled_mean_dice=0.6152, per_class=['0.6152'], iou=0.4443, recall=0.9770, precision=0.4490, vol_sim=0.6297, mcc=0.6539, min_class_dice=0.6152, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8304, per_class_sd=['0.8304'], combined(w=0.50)=0.7228, balanced=0.6360
[2026-06-22 17:37:13] INFO segtask_v1.trainer.trainer: Epoch 132/400 | LR=7.66e-04 | loss=1.4433 | val_dice=0.6152 | best=0.6664 (ep101) | 02:35:35 | L_main=0.6152 L_aux_1=0.6259(w=0.5) L_aux_2=0.7107(w=0.5)
[2026-06-22 17:37:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 132): 11666.1 MiB
[2026-06-22 17:38:23] INFO segtask_v1.trainer.validation:   Val: loss=0.8263, pooled_mean_dice=0.6475, per_class=['0.6475'], iou=0.4787, recall=0.9796, precision=0.4836, vol_sim=0.6610, mcc=0.6795, min_class_dice=0.6475, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8320, per_class_sd=['0.8320'], combined(w=0.50)=0.7398, balanced=0.6638
[2026-06-22 17:38:23] INFO segtask_v1.trainer.trainer: Epoch 133/400 | LR=7.63e-04 | loss=1.4464 | val_dice=0.6475 | best=0.6664 (ep101) | 02:36:44 | L_main=0.6175 L_aux_1=0.6251(w=0.5) L_aux_2=0.7143(w=0.5)
[2026-06-22 17:38:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 133): 11666.1 MiB
[2026-06-22 17:39:32] INFO segtask_v1.trainer.validation:   Val: loss=0.8304, pooled_mean_dice=0.6416, per_class=['0.6416'], iou=0.4723, recall=0.9772, precision=0.4775, vol_sim=0.6565, mcc=0.6727, min_class_dice=0.6416, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8166, per_class_sd=['0.8166'], combined(w=0.50)=0.7291, balanced=0.6562
[2026-06-22 17:39:32] INFO segtask_v1.trainer.trainer: Epoch 134/400 | LR=7.59e-04 | loss=1.4742 | val_dice=0.6416 | best=0.6664 (ep101) | 02:37:54 | L_main=0.6277 L_aux_1=0.6344(w=0.5) L_aux_2=0.7402(w=0.5)
[2026-06-22 17:39:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 134): 11666.1 MiB
[2026-06-22 17:40:41] INFO segtask_v1.trainer.validation:   Val: loss=0.8367, pooled_mean_dice=0.6513, per_class=['0.6513'], iou=0.4830, recall=0.9734, precision=0.4894, vol_sim=0.6691, mcc=0.6815, min_class_dice=0.6513, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8371, per_class_sd=['0.8371'], combined(w=0.50)=0.7442, balanced=0.6678
[2026-06-22 17:40:46] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 17:40:46] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6678 at epoch 135
[2026-06-22 17:40:46] INFO segtask_v1.trainer.trainer: Epoch 135/400 | LR=7.56e-04 | loss=1.4639 | val_dice=0.6513 | best=0.6678 (ep135) | 02:39:07 | L_main=0.6274 L_aux_1=0.6317(w=0.5) L_aux_2=0.7180(w=0.5)
[2026-06-22 17:40:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 135): 11666.1 MiB
[2026-06-22 17:41:57] INFO segtask_v1.trainer.validation:   Val: loss=0.7985, pooled_mean_dice=0.6474, per_class=['0.6474'], iou=0.4786, recall=0.9736, precision=0.4849, vol_sim=0.6650, mcc=0.6768, min_class_dice=0.6474, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8286, per_class_sd=['0.8286'], combined(w=0.50)=0.7380, balanced=0.6630
[2026-06-22 17:41:57] INFO segtask_v1.trainer.trainer: Epoch 136/400 | LR=7.53e-04 | loss=1.4619 | val_dice=0.6474 | best=0.6678 (ep135) | 02:40:18 | L_main=0.6236 L_aux_1=0.6242(w=0.5) L_aux_2=0.7282(w=0.5)
[2026-06-22 17:41:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 136): 11666.1 MiB
[2026-06-22 17:43:07] INFO segtask_v1.trainer.validation:   Val: loss=0.8335, pooled_mean_dice=0.6210, per_class=['0.6210'], iou=0.4503, recall=0.9824, precision=0.4540, vol_sim=0.6321, mcc=0.6580, min_class_dice=0.6210, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8197, per_class_sd=['0.8197'], combined(w=0.50)=0.7203, balanced=0.6393
[2026-06-22 17:43:07] INFO segtask_v1.trainer.trainer: Epoch 137/400 | LR=7.49e-04 | loss=1.4738 | val_dice=0.6210 | best=0.6678 (ep135) | 02:41:29 | L_main=0.6308 L_aux_1=0.6288(w=0.5) L_aux_2=0.7387(w=0.5)
[2026-06-22 17:43:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 137): 11666.1 MiB
[2026-06-22 17:44:17] INFO segtask_v1.trainer.validation:   Val: loss=0.8269, pooled_mean_dice=0.6498, per_class=['0.6498'], iou=0.4812, recall=0.9811, precision=0.4857, vol_sim=0.6623, mcc=0.6810, min_class_dice=0.6498, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8386, per_class_sd=['0.8386'], combined(w=0.50)=0.7442, balanced=0.6668
[2026-06-22 17:44:17] INFO segtask_v1.trainer.trainer: Epoch 138/400 | LR=7.46e-04 | loss=1.4914 | val_dice=0.6498 | best=0.6678 (ep135) | 02:42:38 | L_main=0.6384 L_aux_1=0.6382(w=0.5) L_aux_2=0.7462(w=0.5)
[2026-06-22 17:44:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 138): 11666.1 MiB
[2026-06-22 17:45:28] INFO segtask_v1.trainer.validation:   Val: loss=0.8473, pooled_mean_dice=0.6060, per_class=['0.6060'], iou=0.4348, recall=0.9779, precision=0.4391, vol_sim=0.6197, mcc=0.6447, min_class_dice=0.6060, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8197, per_class_sd=['0.8197'], combined(w=0.50)=0.7129, balanced=0.6264
[2026-06-22 17:45:28] INFO segtask_v1.trainer.trainer: Epoch 139/400 | LR=7.42e-04 | loss=1.5305 | val_dice=0.6060 | best=0.6678 (ep135) | 02:43:49 | L_main=0.6534 L_aux_1=0.6705(w=0.5) L_aux_2=0.7557(w=0.5)
[2026-06-22 17:45:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 139): 11666.1 MiB
[2026-06-22 17:46:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8604, pooled_mean_dice=0.6170, per_class=['0.6170'], iou=0.4461, recall=0.9768, precision=0.4509, vol_sim=0.6316, mcc=0.6543, min_class_dice=0.6170, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8239, per_class_sd=['0.8239'], combined(w=0.50)=0.7204, balanced=0.6364
[2026-06-22 17:46:37] INFO segtask_v1.trainer.trainer: Epoch 140/400 | LR=7.39e-04 | loss=1.4856 | val_dice=0.6170 | best=0.6678 (ep135) | 02:44:58 | L_main=0.6348 L_aux_1=0.6322(w=0.5) L_aux_2=0.7431(w=0.5)
[2026-06-22 17:46:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 140): 11666.1 MiB
[2026-06-22 17:47:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8405, pooled_mean_dice=0.6535, per_class=['0.6535'], iou=0.4853, recall=0.9650, precision=0.4940, vol_sim=0.6772, mcc=0.6838, min_class_dice=0.6535, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8570, per_class_sd=['0.8570'], combined(w=0.50)=0.7552, balanced=0.6728
[2026-06-22 17:47:51] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 17:47:51] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6728 at epoch 141
[2026-06-22 17:47:51] INFO segtask_v1.trainer.trainer: Epoch 141/400 | LR=7.35e-04 | loss=1.4872 | val_dice=0.6535 | best=0.6728 (ep141) | 02:46:13 | L_main=0.6345 L_aux_1=0.6392(w=0.5) L_aux_2=0.7471(w=0.5)
[2026-06-22 17:47:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 141): 11666.1 MiB
[2026-06-22 17:49:00] INFO segtask_v1.trainer.validation:   Val: loss=0.8079, pooled_mean_dice=0.6586, per_class=['0.6586'], iou=0.4910, recall=0.9792, precision=0.4962, vol_sim=0.6726, mcc=0.6878, min_class_dice=0.6586, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8243, per_class_sd=['0.8243'], combined(w=0.50)=0.7414, balanced=0.6719
[2026-06-22 17:49:00] INFO segtask_v1.trainer.trainer: Epoch 142/400 | LR=7.32e-04 | loss=1.5220 | val_dice=0.6586 | best=0.6728 (ep141) | 02:47:22 | L_main=0.6420 L_aux_1=0.6515(w=0.5) L_aux_2=0.7808(w=0.5)
[2026-06-22 17:49:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 142): 11666.1 MiB
[2026-06-22 17:50:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8222, pooled_mean_dice=0.6464, per_class=['0.6464'], iou=0.4775, recall=0.9818, precision=0.4818, vol_sim=0.6583, mcc=0.6767, min_class_dice=0.6464, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8224, per_class_sd=['0.8224'], combined(w=0.50)=0.7344, balanced=0.6612
[2026-06-22 17:50:12] INFO segtask_v1.trainer.trainer: Epoch 143/400 | LR=7.28e-04 | loss=1.5135 | val_dice=0.6464 | best=0.6728 (ep141) | 02:48:34 | L_main=0.6358 L_aux_1=0.6414(w=0.5) L_aux_2=0.7927(w=0.5)
[2026-06-22 17:50:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 143): 11666.1 MiB
[2026-06-22 17:51:23] INFO segtask_v1.trainer.validation:   Val: loss=0.8275, pooled_mean_dice=0.6321, per_class=['0.6321'], iou=0.4621, recall=0.9784, precision=0.4669, vol_sim=0.6461, mcc=0.6671, min_class_dice=0.6321, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8333, per_class_sd=['0.8333'], combined(w=0.50)=0.7327, balanced=0.6509
[2026-06-22 17:51:23] INFO segtask_v1.trainer.trainer: Epoch 144/400 | LR=7.25e-04 | loss=1.5011 | val_dice=0.6321 | best=0.6728 (ep141) | 02:49:45 | L_main=0.6373 L_aux_1=0.6373(w=0.5) L_aux_2=0.7599(w=0.5)
[2026-06-22 17:51:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 144): 11666.1 MiB
[2026-06-22 17:52:33] INFO segtask_v1.trainer.validation:   Val: loss=0.8264, pooled_mean_dice=0.6398, per_class=['0.6398'], iou=0.4704, recall=0.9778, precision=0.4755, vol_sim=0.6543, mcc=0.6720, min_class_dice=0.6398, coverage=[67]/88 samples, pooled_mean_surface_dice@2px=0.8188, per_class_sd=['0.8188'], combined(w=0.50)=0.7293, balanced=0.6551
[2026-06-22 17:52:33] INFO segtask_v1.trainer.trainer: Epoch 145/400 | LR=7.21e-04 | loss=1.5129 | val_dice=0.6398 | best=0.6728 (ep141) | 02:50:55 | L_main=0.6443 L_aux_1=0.6536(w=0.5) L_aux_2=0.7559(w=0.5)
[2026-06-22 17:52:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 145): 11666.1 MiB
[2026-06-22 17:53:42] INFO segtask_v1.trainer.validation:   Val: loss=0.8550, pooled_mean_dice=0.6179, per_class=['0.6179'], iou=0.4471, recall=0.9718, precision=0.4530, vol_sim=0.6358, mcc=0.6549, min_class_dice=0.6179, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8357, per_class_sd=['0.8357'], combined(w=0.50)=0.7268, balanced=0.6390
[2026-06-22 17:53:42] INFO segtask_v1.trainer.trainer: Epoch 146/400 | LR=7.17e-04 | loss=1.4777 | val_dice=0.6179 | best=0.6728 (ep141) | 02:52:04 | L_main=0.6316 L_aux_1=0.6308(w=0.5) L_aux_2=0.7430(w=0.5)
[2026-06-22 17:53:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 146): 11666.1 MiB
[2026-06-22 17:54:52] INFO segtask_v1.trainer.validation:   Val: loss=0.8373, pooled_mean_dice=0.6447, per_class=['0.6447'], iou=0.4757, recall=0.9745, precision=0.4817, vol_sim=0.6615, mcc=0.6764, min_class_dice=0.6447, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8364, per_class_sd=['0.8364'], combined(w=0.50)=0.7406, balanced=0.6620
[2026-06-22 17:54:52] INFO segtask_v1.trainer.trainer: Epoch 147/400 | LR=7.14e-04 | loss=1.4440 | val_dice=0.6447 | best=0.6728 (ep141) | 02:53:13 | L_main=0.6156 L_aux_1=0.6190(w=0.5) L_aux_2=0.7220(w=0.5)
[2026-06-22 17:54:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 147): 11666.1 MiB
[2026-06-22 17:56:01] INFO segtask_v1.trainer.validation:   Val: loss=0.8773, pooled_mean_dice=0.6168, per_class=['0.6168'], iou=0.4459, recall=0.9799, precision=0.4500, vol_sim=0.6295, mcc=0.6533, min_class_dice=0.6168, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8172, per_class_sd=['0.8172'], combined(w=0.50)=0.7170, balanced=0.6352
[2026-06-22 17:56:01] INFO segtask_v1.trainer.trainer: Epoch 148/400 | LR=7.10e-04 | loss=1.4497 | val_dice=0.6168 | best=0.6728 (ep141) | 02:54:22 | L_main=0.6160 L_aux_1=0.6237(w=0.5) L_aux_2=0.7301(w=0.5)
[2026-06-22 17:56:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 148): 11666.1 MiB
[2026-06-22 17:57:10] INFO segtask_v1.trainer.validation:   Val: loss=0.8393, pooled_mean_dice=0.6361, per_class=['0.6361'], iou=0.4663, recall=0.9793, precision=0.4710, vol_sim=0.6495, mcc=0.6700, min_class_dice=0.6361, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8333, per_class_sd=['0.8333'], combined(w=0.50)=0.7347, balanced=0.6542
[2026-06-22 17:57:10] INFO segtask_v1.trainer.trainer: Epoch 149/400 | LR=7.07e-04 | loss=1.4874 | val_dice=0.6361 | best=0.6728 (ep141) | 02:55:32 | L_main=0.6362 L_aux_1=0.6349(w=0.5) L_aux_2=0.7421(w=0.5)
[2026-06-22 17:57:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 149): 11666.1 MiB
[2026-06-22 17:58:19] INFO segtask_v1.trainer.validation:   Val: loss=0.8366, pooled_mean_dice=0.6256, per_class=['0.6256'], iou=0.4551, recall=0.9737, precision=0.4608, vol_sim=0.6424, mcc=0.6593, min_class_dice=0.6256, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8280, per_class_sd=['0.8280'], combined(w=0.50)=0.7268, balanced=0.6443
[2026-06-22 17:58:19] INFO segtask_v1.trainer.trainer: Epoch 150/400 | LR=7.03e-04 | loss=1.4274 | val_dice=0.6256 | best=0.6728 (ep141) | 02:56:41 | L_main=0.6070 L_aux_1=0.6122(w=0.5) L_aux_2=0.7165(w=0.5)
[2026-06-22 17:58:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 150): 11666.1 MiB
[2026-06-22 17:59:31] INFO segtask_v1.trainer.validation:   Val: loss=0.8184, pooled_mean_dice=0.6338, per_class=['0.6338'], iou=0.4640, recall=0.9804, precision=0.4683, vol_sim=0.6465, mcc=0.6671, min_class_dice=0.6338, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8304, per_class_sd=['0.8304'], combined(w=0.50)=0.7321, balanced=0.6518
[2026-06-22 17:59:31] INFO segtask_v1.trainer.trainer: Epoch 151/400 | LR=6.99e-04 | loss=1.4294 | val_dice=0.6338 | best=0.6728 (ep141) | 02:57:52 | L_main=0.6090 L_aux_1=0.6119(w=0.5) L_aux_2=0.7125(w=0.5)
[2026-06-22 17:59:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 151): 11666.1 MiB
[2026-06-22 18:00:40] INFO segtask_v1.trainer.validation:   Val: loss=0.8916, pooled_mean_dice=0.6317, per_class=['0.6317'], iou=0.4617, recall=0.9798, precision=0.4661, vol_sim=0.6448, mcc=0.6677, min_class_dice=0.6317, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8348, per_class_sd=['0.8348'], combined(w=0.50)=0.7333, balanced=0.6508
[2026-06-22 18:00:40] INFO segtask_v1.trainer.trainer: Epoch 152/400 | LR=6.96e-04 | loss=1.4374 | val_dice=0.6317 | best=0.6728 (ep141) | 02:59:02 | L_main=0.6122 L_aux_1=0.6131(w=0.5) L_aux_2=0.7183(w=0.5)
[2026-06-22 18:00:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 152): 11666.1 MiB
[2026-06-22 18:01:50] INFO segtask_v1.trainer.validation:   Val: loss=0.8345, pooled_mean_dice=0.6396, per_class=['0.6396'], iou=0.4702, recall=0.9805, precision=0.4746, vol_sim=0.6523, mcc=0.6729, min_class_dice=0.6396, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8271, per_class_sd=['0.8271'], combined(w=0.50)=0.7334, balanced=0.6563
[2026-06-22 18:01:50] INFO segtask_v1.trainer.trainer: Epoch 153/400 | LR=6.92e-04 | loss=1.4203 | val_dice=0.6396 | best=0.6728 (ep141) | 03:00:12 | L_main=0.6058 L_aux_1=0.6082(w=0.5) L_aux_2=0.7070(w=0.5)
[2026-06-22 18:01:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 153): 11666.1 MiB
[2026-06-22 18:02:59] INFO segtask_v1.trainer.validation:   Val: loss=0.8067, pooled_mean_dice=0.6441, per_class=['0.6441'], iou=0.4750, recall=0.9826, precision=0.4790, vol_sim=0.6555, mcc=0.6749, min_class_dice=0.6441, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8354, per_class_sd=['0.8354'], combined(w=0.50)=0.7397, balanced=0.6613
[2026-06-22 18:02:59] INFO segtask_v1.trainer.trainer: Epoch 154/400 | LR=6.88e-04 | loss=1.4548 | val_dice=0.6441 | best=0.6728 (ep141) | 03:01:20 | L_main=0.6190 L_aux_1=0.6242(w=0.5) L_aux_2=0.7294(w=0.5)
[2026-06-22 18:02:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 154): 11666.1 MiB
[2026-06-22 18:04:10] INFO segtask_v1.trainer.validation:   Val: loss=0.8121, pooled_mean_dice=0.6472, per_class=['0.6472'], iou=0.4784, recall=0.9826, precision=0.4824, vol_sim=0.6586, mcc=0.6783, min_class_dice=0.6472, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8259, per_class_sd=['0.8259'], combined(w=0.50)=0.7365, balanced=0.6625
[2026-06-22 18:04:10] INFO segtask_v1.trainer.trainer: Epoch 155/400 | LR=6.85e-04 | loss=1.5192 | val_dice=0.6472 | best=0.6728 (ep141) | 03:02:31 | L_main=0.6486 L_aux_1=0.6492(w=0.5) L_aux_2=0.7663(w=0.5)
[2026-06-22 18:04:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 155): 11666.1 MiB
[2026-06-22 18:05:20] INFO segtask_v1.trainer.validation:   Val: loss=0.8192, pooled_mean_dice=0.6374, per_class=['0.6374'], iou=0.4678, recall=0.9824, precision=0.4718, vol_sim=0.6489, mcc=0.6720, min_class_dice=0.6374, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8291, per_class_sd=['0.8291'], combined(w=0.50)=0.7333, balanced=0.6548
[2026-06-22 18:05:20] INFO segtask_v1.trainer.trainer: Epoch 156/400 | LR=6.81e-04 | loss=1.4724 | val_dice=0.6374 | best=0.6728 (ep141) | 03:03:41 | L_main=0.6311 L_aux_1=0.6262(w=0.5) L_aux_2=0.7355(w=0.5)
[2026-06-22 18:05:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 156): 11666.1 MiB
[2026-06-22 18:06:29] INFO segtask_v1.trainer.validation:   Val: loss=0.8372, pooled_mean_dice=0.6207, per_class=['0.6207'], iou=0.4500, recall=0.9814, precision=0.4538, vol_sim=0.6324, mcc=0.6571, min_class_dice=0.6207, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8255, per_class_sd=['0.8255'], combined(w=0.50)=0.7231, balanced=0.6398
[2026-06-22 18:06:29] INFO segtask_v1.trainer.trainer: Epoch 157/400 | LR=6.77e-04 | loss=1.5766 | val_dice=0.6207 | best=0.6728 (ep141) | 03:04:50 | L_main=0.6761 L_aux_1=0.6828(w=0.5) L_aux_2=0.7867(w=0.5)
[2026-06-22 18:06:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 157): 11666.1 MiB
[2026-06-22 18:07:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8455, pooled_mean_dice=0.6256, per_class=['0.6256'], iou=0.4551, recall=0.9834, precision=0.4587, vol_sim=0.6362, mcc=0.6630, min_class_dice=0.6256, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8285, per_class_sd=['0.8285'], combined(w=0.50)=0.7270, balanced=0.6446
[2026-06-22 18:07:37] INFO segtask_v1.trainer.trainer: Epoch 158/400 | LR=6.74e-04 | loss=1.4879 | val_dice=0.6256 | best=0.6728 (ep141) | 03:05:59 | L_main=0.6275 L_aux_1=0.6378(w=0.5) L_aux_2=0.7657(w=0.5)
[2026-06-22 18:07:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 158): 11666.1 MiB
[2026-06-22 18:08:46] INFO segtask_v1.trainer.validation:   Val: loss=0.8034, pooled_mean_dice=0.6469, per_class=['0.6469'], iou=0.4780, recall=0.9838, precision=0.4818, vol_sim=0.6575, mcc=0.6775, min_class_dice=0.6469, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8292, per_class_sd=['0.8292'], combined(w=0.50)=0.7380, balanced=0.6627
[2026-06-22 18:08:46] INFO segtask_v1.trainer.trainer: Epoch 159/400 | LR=6.70e-04 | loss=1.4522 | val_dice=0.6469 | best=0.6728 (ep141) | 03:07:08 | L_main=0.6170 L_aux_1=0.6202(w=0.5) L_aux_2=0.7288(w=0.5)
[2026-06-22 18:08:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 159): 11666.1 MiB
[2026-06-22 18:09:56] INFO segtask_v1.trainer.validation:   Val: loss=0.8045, pooled_mean_dice=0.6639, per_class=['0.6639'], iou=0.4969, recall=0.9830, precision=0.5013, vol_sim=0.6754, mcc=0.6925, min_class_dice=0.6639, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8292, per_class_sd=['0.8292'], combined(w=0.50)=0.7466, balanced=0.6773
[2026-06-22 18:10:00] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 18:10:00] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6773 at epoch 160
[2026-06-22 18:10:00] INFO segtask_v1.trainer.trainer: Epoch 160/400 | LR=6.66e-04 | loss=1.4741 | val_dice=0.6639 | best=0.6773 (ep160) | 03:08:22 | L_main=0.6242 L_aux_1=0.6299(w=0.5) L_aux_2=0.7509(w=0.5)
[2026-06-22 18:10:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 160): 11666.1 MiB
[2026-06-22 18:11:10] INFO segtask_v1.trainer.validation:   Val: loss=0.8413, pooled_mean_dice=0.6591, per_class=['0.6591'], iou=0.4915, recall=0.9818, precision=0.4960, vol_sim=0.6713, mcc=0.6882, min_class_dice=0.6591, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8266, per_class_sd=['0.8266'], combined(w=0.50)=0.7428, balanced=0.6727
[2026-06-22 18:11:10] INFO segtask_v1.trainer.trainer: Epoch 161/400 | LR=6.62e-04 | loss=1.4688 | val_dice=0.6591 | best=0.6773 (ep160) | 03:09:32 | L_main=0.6245 L_aux_1=0.6251(w=0.5) L_aux_2=0.7434(w=0.5)
[2026-06-22 18:11:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 161): 11666.1 MiB
[2026-06-22 18:12:19] INFO segtask_v1.trainer.validation:   Val: loss=0.7891, pooled_mean_dice=0.6585, per_class=['0.6585'], iou=0.4909, recall=0.9826, precision=0.4952, vol_sim=0.6702, mcc=0.6867, min_class_dice=0.6585, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8353, per_class_sd=['0.8353'], combined(w=0.50)=0.7469, balanced=0.6735
[2026-06-22 18:12:19] INFO segtask_v1.trainer.trainer: Epoch 162/400 | LR=6.59e-04 | loss=1.4580 | val_dice=0.6585 | best=0.6773 (ep160) | 03:10:40 | L_main=0.6208 L_aux_1=0.6227(w=0.5) L_aux_2=0.7319(w=0.5)
[2026-06-22 18:12:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 162): 11666.1 MiB
[2026-06-22 18:13:28] INFO segtask_v1.trainer.validation:   Val: loss=0.8337, pooled_mean_dice=0.6480, per_class=['0.6480'], iou=0.4793, recall=0.9814, precision=0.4837, vol_sim=0.6603, mcc=0.6803, min_class_dice=0.6480, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8407, per_class_sd=['0.8407'], combined(w=0.50)=0.7444, balanced=0.6657
[2026-06-22 18:13:28] INFO segtask_v1.trainer.trainer: Epoch 163/400 | LR=6.55e-04 | loss=1.4554 | val_dice=0.6480 | best=0.6773 (ep160) | 03:11:50 | L_main=0.6151 L_aux_1=0.6274(w=0.5) L_aux_2=0.7327(w=0.5)
[2026-06-22 18:13:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 163): 11666.1 MiB
[2026-06-22 18:14:38] INFO segtask_v1.trainer.validation:   Val: loss=0.8407, pooled_mean_dice=0.6289, per_class=['0.6289'], iou=0.4587, recall=0.9788, precision=0.4633, vol_sim=0.6425, mcc=0.6654, min_class_dice=0.6289, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8433, per_class_sd=['0.8433'], combined(w=0.50)=0.7361, balanced=0.6497
[2026-06-22 18:14:38] INFO segtask_v1.trainer.trainer: Epoch 164/400 | LR=6.51e-04 | loss=1.5025 | val_dice=0.6289 | best=0.6773 (ep160) | 03:12:59 | L_main=0.6326 L_aux_1=0.6617(w=0.5) L_aux_2=0.7507(w=0.5)
[2026-06-22 18:14:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 164): 11666.1 MiB
[2026-06-22 18:15:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8271, pooled_mean_dice=0.6304, per_class=['0.6304'], iou=0.4602, recall=0.9782, precision=0.4650, vol_sim=0.6444, mcc=0.6641, min_class_dice=0.6304, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8259, per_class_sd=['0.8259'], combined(w=0.50)=0.7281, balanced=0.6481
[2026-06-22 18:15:47] INFO segtask_v1.trainer.trainer: Epoch 165/400 | LR=6.47e-04 | loss=1.4683 | val_dice=0.6304 | best=0.6773 (ep160) | 03:14:08 | L_main=0.6235 L_aux_1=0.6462(w=0.5) L_aux_2=0.7224(w=0.5)
[2026-06-22 18:15:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 165): 11666.1 MiB
[2026-06-22 18:16:56] INFO segtask_v1.trainer.validation:   Val: loss=0.8270, pooled_mean_dice=0.6585, per_class=['0.6585'], iou=0.4909, recall=0.9782, precision=0.4963, vol_sim=0.6732, mcc=0.6883, min_class_dice=0.6585, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8454, per_class_sd=['0.8454'], combined(w=0.50)=0.7520, balanced=0.6753
[2026-06-22 18:16:56] INFO segtask_v1.trainer.trainer: Epoch 166/400 | LR=6.43e-04 | loss=1.5025 | val_dice=0.6585 | best=0.6773 (ep160) | 03:15:18 | L_main=0.6355 L_aux_1=0.6526(w=0.5) L_aux_2=0.7589(w=0.5)
[2026-06-22 18:16:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 166): 11666.1 MiB
[2026-06-22 18:18:05] INFO segtask_v1.trainer.validation:   Val: loss=0.8343, pooled_mean_dice=0.6488, per_class=['0.6488'], iou=0.4801, recall=0.9773, precision=0.4856, vol_sim=0.6638, mcc=0.6808, min_class_dice=0.6488, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8444, per_class_sd=['0.8444'], combined(w=0.50)=0.7466, balanced=0.6669
[2026-06-22 18:18:05] INFO segtask_v1.trainer.trainer: Epoch 167/400 | LR=6.40e-04 | loss=1.4670 | val_dice=0.6488 | best=0.6773 (ep160) | 03:16:27 | L_main=0.6204 L_aux_1=0.6293(w=0.5) L_aux_2=0.7428(w=0.5)
[2026-06-22 18:18:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 167): 11666.1 MiB
[2026-06-22 18:19:17] INFO segtask_v1.trainer.validation:   Val: loss=0.8289, pooled_mean_dice=0.6284, per_class=['0.6284'], iou=0.4581, recall=0.9806, precision=0.4623, vol_sim=0.6408, mcc=0.6637, min_class_dice=0.6284, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8297, per_class_sd=['0.8297'], combined(w=0.50)=0.7290, balanced=0.6471
[2026-06-22 18:19:17] INFO segtask_v1.trainer.trainer: Epoch 168/400 | LR=6.36e-04 | loss=1.4483 | val_dice=0.6284 | best=0.6773 (ep160) | 03:17:38 | L_main=0.6186 L_aux_1=0.6227(w=0.5) L_aux_2=0.7116(w=0.5)
[2026-06-22 18:19:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 168): 11666.1 MiB
[2026-06-22 18:20:26] INFO segtask_v1.trainer.validation:   Val: loss=0.8172, pooled_mean_dice=0.6419, per_class=['0.6419'], iou=0.4727, recall=0.9710, precision=0.4794, vol_sim=0.6611, mcc=0.6715, min_class_dice=0.6419, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8254, per_class_sd=['0.8254'], combined(w=0.50)=0.7337, balanced=0.6577
[2026-06-22 18:20:26] INFO segtask_v1.trainer.trainer: Epoch 169/400 | LR=6.32e-04 | loss=1.4465 | val_dice=0.6419 | best=0.6773 (ep160) | 03:18:47 | L_main=0.6172 L_aux_1=0.6228(w=0.5) L_aux_2=0.7192(w=0.5)
[2026-06-22 18:20:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 169): 11666.1 MiB
[2026-06-22 18:21:35] INFO segtask_v1.trainer.validation:   Val: loss=0.8425, pooled_mean_dice=0.6453, per_class=['0.6453'], iou=0.4763, recall=0.9773, precision=0.4817, vol_sim=0.6603, mcc=0.6774, min_class_dice=0.6453, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8315, per_class_sd=['0.8315'], combined(w=0.50)=0.7384, balanced=0.6618
[2026-06-22 18:21:35] INFO segtask_v1.trainer.trainer: Epoch 170/400 | LR=6.28e-04 | loss=1.4458 | val_dice=0.6453 | best=0.6773 (ep160) | 03:19:56 | L_main=0.6155 L_aux_1=0.6208(w=0.5) L_aux_2=0.7217(w=0.5)
[2026-06-22 18:21:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 170): 11666.1 MiB
[2026-06-22 18:22:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8080, pooled_mean_dice=0.6627, per_class=['0.6627'], iou=0.4956, recall=0.9810, precision=0.5004, vol_sim=0.6756, mcc=0.6927, min_class_dice=0.6627, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8580, per_class_sd=['0.8580'], combined(w=0.50)=0.7604, balanced=0.6810
[2026-06-22 18:22:49] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 18:22:49] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6810 at epoch 171
[2026-06-22 18:22:49] INFO segtask_v1.trainer.trainer: Epoch 171/400 | LR=6.24e-04 | loss=1.4595 | val_dice=0.6627 | best=0.6810 (ep171) | 03:21:10 | L_main=0.6222 L_aux_1=0.6237(w=0.5) L_aux_2=0.7335(w=0.5)
[2026-06-22 18:22:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 171): 11666.1 MiB
[2026-06-22 18:23:58] INFO segtask_v1.trainer.validation:   Val: loss=0.8533, pooled_mean_dice=0.6388, per_class=['0.6388'], iou=0.4692, recall=0.9808, precision=0.4736, vol_sim=0.6513, mcc=0.6727, min_class_dice=0.6388, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8326, per_class_sd=['0.8326'], combined(w=0.50)=0.7357, balanced=0.6565
[2026-06-22 18:23:58] INFO segtask_v1.trainer.trainer: Epoch 172/400 | LR=6.20e-04 | loss=1.4880 | val_dice=0.6388 | best=0.6810 (ep171) | 03:22:20 | L_main=0.6347 L_aux_1=0.6353(w=0.5) L_aux_2=0.7468(w=0.5)
[2026-06-22 18:23:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 172): 11666.1 MiB
[2026-06-22 18:25:08] INFO segtask_v1.trainer.validation:   Val: loss=0.8378, pooled_mean_dice=0.6465, per_class=['0.6465'], iou=0.4777, recall=0.9767, precision=0.4832, vol_sim=0.6619, mcc=0.6784, min_class_dice=0.6465, coverage=[85]/88 samples, pooled_mean_surface_dice@2px=0.8502, per_class_sd=['0.8502'], combined(w=0.50)=0.7484, balanced=0.6658
[2026-06-22 18:25:08] INFO segtask_v1.trainer.trainer: Epoch 173/400 | LR=6.17e-04 | loss=1.4383 | val_dice=0.6465 | best=0.6810 (ep171) | 03:23:29 | L_main=0.6108 L_aux_1=0.6134(w=0.5) L_aux_2=0.7226(w=0.5)
[2026-06-22 18:25:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 173): 11666.1 MiB
[2026-06-22 18:26:18] INFO segtask_v1.trainer.validation:   Val: loss=0.8197, pooled_mean_dice=0.6505, per_class=['0.6505'], iou=0.4820, recall=0.9807, precision=0.4866, vol_sim=0.6632, mcc=0.6812, min_class_dice=0.6505, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8343, per_class_sd=['0.8343'], combined(w=0.50)=0.7424, balanced=0.6666
[2026-06-22 18:26:18] INFO segtask_v1.trainer.trainer: Epoch 174/400 | LR=6.13e-04 | loss=1.4702 | val_dice=0.6505 | best=0.6810 (ep171) | 03:24:40 | L_main=0.6224 L_aux_1=0.6235(w=0.5) L_aux_2=0.7505(w=0.5)
[2026-06-22 18:26:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 174): 11666.1 MiB
[2026-06-22 18:27:27] INFO segtask_v1.trainer.validation:   Val: loss=0.8153, pooled_mean_dice=0.6404, per_class=['0.6404'], iou=0.4710, recall=0.9792, precision=0.4758, vol_sim=0.6540, mcc=0.6726, min_class_dice=0.6404, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8245, per_class_sd=['0.8245'], combined(w=0.50)=0.7324, balanced=0.6565
[2026-06-22 18:27:27] INFO segtask_v1.trainer.trainer: Epoch 175/400 | LR=6.09e-04 | loss=1.5090 | val_dice=0.6404 | best=0.6810 (ep171) | 03:25:49 | L_main=0.6409 L_aux_1=0.6427(w=0.5) L_aux_2=0.7639(w=0.5)
[2026-06-22 18:27:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 175): 11666.1 MiB
[2026-06-22 18:28:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8232, pooled_mean_dice=0.6562, per_class=['0.6562'], iou=0.4884, recall=0.9657, precision=0.4970, vol_sim=0.6795, mcc=0.6847, min_class_dice=0.6562, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8478, per_class_sd=['0.8478'], combined(w=0.50)=0.7520, balanced=0.6736
[2026-06-22 18:28:37] INFO segtask_v1.trainer.trainer: Epoch 176/400 | LR=6.05e-04 | loss=1.4675 | val_dice=0.6562 | best=0.6810 (ep171) | 03:26:59 | L_main=0.6251 L_aux_1=0.6263(w=0.5) L_aux_2=0.7371(w=0.5)
[2026-06-22 18:28:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 176): 11666.1 MiB
[2026-06-22 18:29:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8127, pooled_mean_dice=0.6370, per_class=['0.6370'], iou=0.4674, recall=0.9752, precision=0.4730, vol_sim=0.6532, mcc=0.6697, min_class_dice=0.6370, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8306, per_class_sd=['0.8306'], combined(w=0.50)=0.7338, balanced=0.6545
[2026-06-22 18:29:47] INFO segtask_v1.trainer.trainer: Epoch 177/400 | LR=6.01e-04 | loss=1.5411 | val_dice=0.6370 | best=0.6810 (ep171) | 03:28:08 | L_main=0.6559 L_aux_1=0.6540(w=0.5) L_aux_2=0.7858(w=0.5)
[2026-06-22 18:29:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 177): 11666.1 MiB
[2026-06-22 18:30:56] INFO segtask_v1.trainer.validation:   Val: loss=0.8246, pooled_mean_dice=0.6322, per_class=['0.6322'], iou=0.4622, recall=0.9718, precision=0.4685, vol_sim=0.6505, mcc=0.6648, min_class_dice=0.6322, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8339, per_class_sd=['0.8339'], combined(w=0.50)=0.7331, balanced=0.6508
[2026-06-22 18:30:56] INFO segtask_v1.trainer.trainer: Epoch 178/400 | LR=5.97e-04 | loss=1.4876 | val_dice=0.6322 | best=0.6810 (ep171) | 03:29:18 | L_main=0.6278 L_aux_1=0.6266(w=0.5) L_aux_2=0.7715(w=0.5)
[2026-06-22 18:30:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 178): 11666.1 MiB
[2026-06-22 18:32:06] INFO segtask_v1.trainer.validation:   Val: loss=0.8120, pooled_mean_dice=0.6388, per_class=['0.6388'], iou=0.4692, recall=0.9779, precision=0.4743, vol_sim=0.6532, mcc=0.6716, min_class_dice=0.6388, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8322, per_class_sd=['0.8322'], combined(w=0.50)=0.7355, balanced=0.6563
[2026-06-22 18:32:06] INFO segtask_v1.trainer.trainer: Epoch 179/400 | LR=5.93e-04 | loss=1.4841 | val_dice=0.6388 | best=0.6810 (ep171) | 03:30:27 | L_main=0.6277 L_aux_1=0.6297(w=0.5) L_aux_2=0.7593(w=0.5)
[2026-06-22 18:32:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 179): 11666.1 MiB
[2026-06-22 18:33:16] INFO segtask_v1.trainer.validation:   Val: loss=0.8275, pooled_mean_dice=0.6478, per_class=['0.6478'], iou=0.4790, recall=0.9682, precision=0.4867, vol_sim=0.6690, mcc=0.6770, min_class_dice=0.6478, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8396, per_class_sd=['0.8396'], combined(w=0.50)=0.7437, balanced=0.6650
[2026-06-22 18:33:16] INFO segtask_v1.trainer.trainer: Epoch 180/400 | LR=5.89e-04 | loss=1.4739 | val_dice=0.6478 | best=0.6810 (ep171) | 03:31:37 | L_main=0.6189 L_aux_1=0.6228(w=0.5) L_aux_2=0.7702(w=0.5)
[2026-06-22 18:33:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 180): 11666.1 MiB
[2026-06-22 18:34:26] INFO segtask_v1.trainer.validation:   Val: loss=0.8423, pooled_mean_dice=0.6385, per_class=['0.6385'], iou=0.4690, recall=0.9694, precision=0.4760, vol_sim=0.6586, mcc=0.6717, min_class_dice=0.6385, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8459, per_class_sd=['0.8459'], combined(w=0.50)=0.7422, balanced=0.6582
[2026-06-22 18:34:26] INFO segtask_v1.trainer.trainer: Epoch 181/400 | LR=5.85e-04 | loss=1.4890 | val_dice=0.6385 | best=0.6810 (ep171) | 03:32:47 | L_main=0.6276 L_aux_1=0.6318(w=0.5) L_aux_2=0.7689(w=0.5)
[2026-06-22 18:34:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 181): 11666.1 MiB
[2026-06-22 18:35:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8894, pooled_mean_dice=0.6151, per_class=['0.6151'], iou=0.4442, recall=0.9814, precision=0.4480, vol_sim=0.6268, mcc=0.6533, min_class_dice=0.6151, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8214, per_class_sd=['0.8214'], combined(w=0.50)=0.7183, balanced=0.6345
[2026-06-22 18:35:36] INFO segtask_v1.trainer.trainer: Epoch 182/400 | LR=5.82e-04 | loss=1.5112 | val_dice=0.6151 | best=0.6810 (ep171) | 03:33:57 | L_main=0.6377 L_aux_1=0.6426(w=0.5) L_aux_2=0.7776(w=0.5)
[2026-06-22 18:35:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 182): 11666.1 MiB
[2026-06-22 18:36:44] INFO segtask_v1.trainer.validation:   Val: loss=0.8002, pooled_mean_dice=0.6633, per_class=['0.6633'], iou=0.4963, recall=0.9794, precision=0.5015, vol_sim=0.6772, mcc=0.6891, min_class_dice=0.6633, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8361, per_class_sd=['0.8361'], combined(w=0.50)=0.7497, balanced=0.6777
[2026-06-22 18:36:44] INFO segtask_v1.trainer.trainer: Epoch 183/400 | LR=5.78e-04 | loss=1.4801 | val_dice=0.6633 | best=0.6810 (ep171) | 03:35:06 | L_main=0.6248 L_aux_1=0.6250(w=0.5) L_aux_2=0.7581(w=0.5)
[2026-06-22 18:36:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 183): 11666.1 MiB
[2026-06-22 18:37:54] INFO segtask_v1.trainer.validation:   Val: loss=0.8732, pooled_mean_dice=0.6509, per_class=['0.6509'], iou=0.4825, recall=0.9750, precision=0.4885, vol_sim=0.6676, mcc=0.6810, min_class_dice=0.6509, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8244, per_class_sd=['0.8244'], combined(w=0.50)=0.7377, balanced=0.6654
[2026-06-22 18:37:54] INFO segtask_v1.trainer.trainer: Epoch 184/400 | LR=5.74e-04 | loss=1.4790 | val_dice=0.6509 | best=0.6810 (ep171) | 03:36:16 | L_main=0.6223 L_aux_1=0.6200(w=0.5) L_aux_2=0.7681(w=0.5)
[2026-06-22 18:37:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 184): 11666.1 MiB
[2026-06-22 18:39:04] INFO segtask_v1.trainer.validation:   Val: loss=0.8244, pooled_mean_dice=0.6370, per_class=['0.6370'], iou=0.4674, recall=0.9664, precision=0.4751, vol_sim=0.6592, mcc=0.6664, min_class_dice=0.6370, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8341, per_class_sd=['0.8341'], combined(w=0.50)=0.7355, balanced=0.6548
[2026-06-22 18:39:04] INFO segtask_v1.trainer.trainer: Epoch 185/400 | LR=5.70e-04 | loss=1.4588 | val_dice=0.6370 | best=0.6810 (ep171) | 03:37:25 | L_main=0.6127 L_aux_1=0.6215(w=0.5) L_aux_2=0.7566(w=0.5)
[2026-06-22 18:39:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 185): 11666.1 MiB
[2026-06-22 18:40:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8098, pooled_mean_dice=0.6583, per_class=['0.6583'], iou=0.4906, recall=0.9770, precision=0.4964, vol_sim=0.6738, mcc=0.6877, min_class_dice=0.6583, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8520, per_class_sd=['0.8520'], combined(w=0.50)=0.7551, balanced=0.6761
[2026-06-22 18:40:12] INFO segtask_v1.trainer.trainer: Epoch 186/400 | LR=5.66e-04 | loss=1.5010 | val_dice=0.6583 | best=0.6810 (ep171) | 03:38:34 | L_main=0.6357 L_aux_1=0.6433(w=0.5) L_aux_2=0.7622(w=0.5)
[2026-06-22 18:40:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 186): 11666.1 MiB
[2026-06-22 18:41:22] INFO segtask_v1.trainer.validation:   Val: loss=0.8819, pooled_mean_dice=0.6295, per_class=['0.6295'], iou=0.4593, recall=0.9761, precision=0.4645, vol_sim=0.6449, mcc=0.6652, min_class_dice=0.6295, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.8475, per_class_sd=['0.8475'], combined(w=0.50)=0.7385, balanced=0.6508
[2026-06-22 18:41:22] INFO segtask_v1.trainer.trainer: Epoch 187/400 | LR=5.62e-04 | loss=1.4792 | val_dice=0.6295 | best=0.6810 (ep171) | 03:39:44 | L_main=0.6231 L_aux_1=0.6239(w=0.5) L_aux_2=0.7716(w=0.5)
[2026-06-22 18:41:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 187): 11666.1 MiB
[2026-06-22 18:42:31] INFO segtask_v1.trainer.validation:   Val: loss=0.8140, pooled_mean_dice=0.6591, per_class=['0.6591'], iou=0.4915, recall=0.9762, precision=0.4975, vol_sim=0.6751, mcc=0.6871, min_class_dice=0.6591, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8409, per_class_sd=['0.8409'], combined(w=0.50)=0.7500, balanced=0.6750
[2026-06-22 18:42:31] INFO segtask_v1.trainer.trainer: Epoch 188/400 | LR=5.58e-04 | loss=1.4985 | val_dice=0.6591 | best=0.6810 (ep171) | 03:40:53 | L_main=0.6283 L_aux_1=0.6356(w=0.5) L_aux_2=0.7841(w=0.5)
[2026-06-22 18:42:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 188): 11666.1 MiB
[2026-06-22 18:43:41] INFO segtask_v1.trainer.validation:   Val: loss=0.8280, pooled_mean_dice=0.6522, per_class=['0.6522'], iou=0.4839, recall=0.9811, precision=0.4885, vol_sim=0.6648, mcc=0.6826, min_class_dice=0.6522, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8383, per_class_sd=['0.8383'], combined(w=0.50)=0.7452, balanced=0.6687
[2026-06-22 18:43:41] INFO segtask_v1.trainer.trainer: Epoch 189/400 | LR=5.54e-04 | loss=1.4713 | val_dice=0.6522 | best=0.6810 (ep171) | 03:42:02 | L_main=0.6158 L_aux_1=0.6224(w=0.5) L_aux_2=0.7687(w=0.5)
[2026-06-22 18:43:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 189): 11666.1 MiB
[2026-06-22 18:44:51] INFO segtask_v1.trainer.validation:   Val: loss=0.8400, pooled_mean_dice=0.6570, per_class=['0.6570'], iou=0.4893, recall=0.9787, precision=0.4945, vol_sim=0.6714, mcc=0.6866, min_class_dice=0.6570, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8435, per_class_sd=['0.8435'], combined(w=0.50)=0.7503, balanced=0.6737
[2026-06-22 18:44:51] INFO segtask_v1.trainer.trainer: Epoch 190/400 | LR=5.50e-04 | loss=1.4746 | val_dice=0.6570 | best=0.6810 (ep171) | 03:43:12 | L_main=0.6182 L_aux_1=0.6197(w=0.5) L_aux_2=0.7739(w=0.5)
[2026-06-22 18:44:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 190): 11666.1 MiB
[2026-06-22 18:46:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8597, pooled_mean_dice=0.6549, per_class=['0.6549'], iou=0.4869, recall=0.9691, precision=0.4945, vol_sim=0.6758, mcc=0.6842, min_class_dice=0.6549, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8557, per_class_sd=['0.8557'], combined(w=0.50)=0.7553, balanced=0.6737
[2026-06-22 18:46:03] INFO segtask_v1.trainer.trainer: Epoch 191/400 | LR=5.46e-04 | loss=1.4637 | val_dice=0.6549 | best=0.6810 (ep171) | 03:44:25 | L_main=0.6161 L_aux_1=0.6162(w=0.5) L_aux_2=0.7603(w=0.5)
[2026-06-22 18:46:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 191): 11666.1 MiB
[2026-06-22 18:47:16] INFO segtask_v1.trainer.validation:   Val: loss=0.8347, pooled_mean_dice=0.6762, per_class=['0.6762'], iou=0.5108, recall=0.9704, precision=0.5189, vol_sim=0.6968, mcc=0.7014, min_class_dice=0.6762, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8614, per_class_sd=['0.8614'], combined(w=0.50)=0.7688, balanced=0.6929
[2026-06-22 18:47:20] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 18:47:20] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6929 at epoch 192
[2026-06-22 18:47:20] INFO segtask_v1.trainer.trainer: Epoch 192/400 | LR=5.42e-04 | loss=1.4708 | val_dice=0.6762 | best=0.6929 (ep192) | 03:45:42 | L_main=0.6172 L_aux_1=0.6198(w=0.5) L_aux_2=0.7662(w=0.5)
[2026-06-22 18:47:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 192): 11666.1 MiB
[2026-06-22 18:48:31] INFO segtask_v1.trainer.validation:   Val: loss=0.7989, pooled_mean_dice=0.6694, per_class=['0.6694'], iou=0.5031, recall=0.9745, precision=0.5098, vol_sim=0.6869, mcc=0.6952, min_class_dice=0.6694, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8526, per_class_sd=['0.8526'], combined(w=0.50)=0.7610, balanced=0.6856
[2026-06-22 18:48:31] INFO segtask_v1.trainer.trainer: Epoch 193/400 | LR=5.38e-04 | loss=1.4937 | val_dice=0.6694 | best=0.6929 (ep192) | 03:46:53 | L_main=0.6273 L_aux_1=0.6267(w=0.5) L_aux_2=0.7876(w=0.5)
[2026-06-22 18:48:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 193): 11666.1 MiB
[2026-06-22 18:49:41] INFO segtask_v1.trainer.validation:   Val: loss=0.8405, pooled_mean_dice=0.6569, per_class=['0.6569'], iou=0.4891, recall=0.9689, precision=0.4969, vol_sim=0.6780, mcc=0.6856, min_class_dice=0.6569, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8404, per_class_sd=['0.8404'], combined(w=0.50)=0.7486, balanced=0.6730
[2026-06-22 18:49:41] INFO segtask_v1.trainer.trainer: Epoch 194/400 | LR=5.34e-04 | loss=1.4622 | val_dice=0.6569 | best=0.6929 (ep192) | 03:48:03 | L_main=0.6117 L_aux_1=0.6222(w=0.5) L_aux_2=0.7615(w=0.5)
[2026-06-22 18:49:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 194): 11666.1 MiB
[2026-06-22 18:50:50] INFO segtask_v1.trainer.validation:   Val: loss=0.7733, pooled_mean_dice=0.6813, per_class=['0.6813'], iou=0.5167, recall=0.9733, precision=0.5241, vol_sim=0.7001, mcc=0.7045, min_class_dice=0.6813, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8567, per_class_sd=['0.8567'], combined(w=0.50)=0.7690, balanced=0.6964
[2026-06-22 18:50:54] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 18:50:54] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6964 at epoch 195
[2026-06-22 18:50:54] INFO segtask_v1.trainer.trainer: Epoch 195/400 | LR=5.30e-04 | loss=1.4400 | val_dice=0.6813 | best=0.6964 (ep195) | 03:49:16 | L_main=0.6037 L_aux_1=0.6249(w=0.5) L_aux_2=0.7300(w=0.5)
[2026-06-22 18:50:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 195): 11666.1 MiB
[2026-06-22 18:52:05] INFO segtask_v1.trainer.validation:   Val: loss=0.7854, pooled_mean_dice=0.6665, per_class=['0.6665'], iou=0.4998, recall=0.9777, precision=0.5056, vol_sim=0.6817, mcc=0.6937, min_class_dice=0.6665, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8499, per_class_sd=['0.8499'], combined(w=0.50)=0.7582, balanced=0.6828
[2026-06-22 18:52:05] INFO segtask_v1.trainer.trainer: Epoch 196/400 | LR=5.26e-04 | loss=1.4562 | val_dice=0.6665 | best=0.6964 (ep195) | 03:50:26 | L_main=0.6097 L_aux_1=0.6191(w=0.5) L_aux_2=0.7589(w=0.5)
[2026-06-22 18:52:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 196): 11666.1 MiB
[2026-06-22 18:53:14] INFO segtask_v1.trainer.validation:   Val: loss=0.8489, pooled_mean_dice=0.6475, per_class=['0.6475'], iou=0.4788, recall=0.9794, precision=0.4836, vol_sim=0.6611, mcc=0.6806, min_class_dice=0.6475, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8445, per_class_sd=['0.8445'], combined(w=0.50)=0.7460, balanced=0.6659
[2026-06-22 18:53:14] INFO segtask_v1.trainer.trainer: Epoch 197/400 | LR=5.22e-04 | loss=1.4649 | val_dice=0.6475 | best=0.6964 (ep195) | 03:51:36 | L_main=0.6149 L_aux_1=0.6167(w=0.5) L_aux_2=0.7658(w=0.5)
[2026-06-22 18:53:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 197): 11666.1 MiB
[2026-06-22 18:54:27] INFO segtask_v1.trainer.validation:   Val: loss=0.8537, pooled_mean_dice=0.6443, per_class=['0.6443'], iou=0.4752, recall=0.9672, precision=0.4830, vol_sim=0.6661, mcc=0.6755, min_class_dice=0.6443, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8387, per_class_sd=['0.8387'], combined(w=0.50)=0.7415, balanced=0.6620
[2026-06-22 18:54:27] INFO segtask_v1.trainer.trainer: Epoch 198/400 | LR=5.18e-04 | loss=1.4464 | val_dice=0.6443 | best=0.6964 (ep195) | 03:52:48 | L_main=0.6134 L_aux_1=0.6129(w=0.5) L_aux_2=0.7307(w=0.5)
[2026-06-22 18:54:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 198): 11666.1 MiB
[2026-06-22 18:55:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8320, pooled_mean_dice=0.6581, per_class=['0.6581'], iou=0.4905, recall=0.9673, precision=0.4987, vol_sim=0.6804, mcc=0.6873, min_class_dice=0.6581, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8614, per_class_sd=['0.8614'], combined(w=0.50)=0.7598, balanced=0.6775
[2026-06-22 18:55:36] INFO segtask_v1.trainer.trainer: Epoch 199/400 | LR=5.14e-04 | loss=1.4507 | val_dice=0.6581 | best=0.6964 (ep195) | 03:53:57 | L_main=0.6131 L_aux_1=0.6115(w=0.5) L_aux_2=0.7469(w=0.5)
[2026-06-22 18:55:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 199): 11666.1 MiB
[2026-06-22 18:56:45] INFO segtask_v1.trainer.validation:   Val: loss=0.7783, pooled_mean_dice=0.6855, per_class=['0.6855'], iou=0.5215, recall=0.9815, precision=0.5266, vol_sim=0.6984, mcc=0.7101, min_class_dice=0.6855, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8456, per_class_sd=['0.8456'], combined(w=0.50)=0.7655, balanced=0.6982
[2026-06-22 18:56:49] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 18:56:49] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.6982 at epoch 200
[2026-06-22 18:56:49] INFO segtask_v1.trainer.trainer: Epoch 200/400 | LR=5.10e-04 | loss=1.4627 | val_dice=0.6855 | best=0.6982 (ep200) | 03:55:10 | L_main=0.6231 L_aux_1=0.6220(w=0.5) L_aux_2=0.7364(w=0.5)
[2026-06-22 18:56:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 200): 11666.1 MiB
[2026-06-22 18:57:59] INFO segtask_v1.trainer.validation:   Val: loss=0.8209, pooled_mean_dice=0.6590, per_class=['0.6590'], iou=0.4915, recall=0.9783, precision=0.4969, vol_sim=0.6737, mcc=0.6875, min_class_dice=0.6590, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8496, per_class_sd=['0.8496'], combined(w=0.50)=0.7543, balanced=0.6763
[2026-06-22 18:57:59] INFO segtask_v1.trainer.trainer: Epoch 201/400 | LR=5.06e-04 | loss=1.4413 | val_dice=0.6590 | best=0.6982 (ep200) | 03:56:20 | L_main=0.6127 L_aux_1=0.6145(w=0.5) L_aux_2=0.7308(w=0.5)
[2026-06-22 18:57:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 201): 11666.1 MiB
[2026-06-22 18:59:08] INFO segtask_v1.trainer.validation:   Val: loss=0.8279, pooled_mean_dice=0.6321, per_class=['0.6321'], iou=0.4621, recall=0.9748, precision=0.4676, vol_sim=0.6484, mcc=0.6650, min_class_dice=0.6321, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8332, per_class_sd=['0.8332'], combined(w=0.50)=0.7326, balanced=0.6507
[2026-06-22 18:59:08] INFO segtask_v1.trainer.trainer: Epoch 202/400 | LR=5.02e-04 | loss=1.4113 | val_dice=0.6321 | best=0.6982 (ep200) | 03:57:30 | L_main=0.6005 L_aux_1=0.5969(w=0.5) L_aux_2=0.7135(w=0.5)
[2026-06-22 18:59:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 202): 11666.1 MiB
[2026-06-22 19:00:16] INFO segtask_v1.trainer.validation:   Val: loss=0.8697, pooled_mean_dice=0.6401, per_class=['0.6401'], iou=0.4707, recall=0.9707, precision=0.4775, vol_sim=0.6594, mcc=0.6745, min_class_dice=0.6401, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8595, per_class_sd=['0.8595'], combined(w=0.50)=0.7498, balanced=0.6618
[2026-06-22 19:00:16] INFO segtask_v1.trainer.trainer: Epoch 203/400 | LR=4.99e-04 | loss=1.4085 | val_dice=0.6401 | best=0.6982 (ep200) | 03:58:38 | L_main=0.5997 L_aux_1=0.5969(w=0.5) L_aux_2=0.7116(w=0.5)
[2026-06-22 19:00:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 203): 11666.1 MiB
[2026-06-22 19:01:29] INFO segtask_v1.trainer.validation:   Val: loss=0.7817, pooled_mean_dice=0.6714, per_class=['0.6714'], iou=0.5053, recall=0.9714, precision=0.5130, vol_sim=0.6911, mcc=0.6982, min_class_dice=0.6714, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8599, per_class_sd=['0.8599'], combined(w=0.50)=0.7656, balanced=0.6886
[2026-06-22 19:01:29] INFO segtask_v1.trainer.trainer: Epoch 204/400 | LR=4.95e-04 | loss=1.4406 | val_dice=0.6714 | best=0.6982 (ep200) | 03:59:51 | L_main=0.6103 L_aux_1=0.6178(w=0.5) L_aux_2=0.7203(w=0.5)
[2026-06-22 19:01:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 204): 11666.1 MiB
[2026-06-22 19:02:40] INFO segtask_v1.trainer.validation:   Val: loss=0.8359, pooled_mean_dice=0.6681, per_class=['0.6681'], iou=0.5016, recall=0.9830, precision=0.5060, vol_sim=0.6797, mcc=0.6974, min_class_dice=0.6681, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8527, per_class_sd=['0.8527'], combined(w=0.50)=0.7604, balanced=0.6848
[2026-06-22 19:02:40] INFO segtask_v1.trainer.trainer: Epoch 205/400 | LR=4.91e-04 | loss=1.4409 | val_dice=0.6681 | best=0.6982 (ep200) | 04:01:01 | L_main=0.6150 L_aux_1=0.6186(w=0.5) L_aux_2=0.7113(w=0.5)
[2026-06-22 19:02:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 205): 11666.1 MiB
[2026-06-22 19:03:49] INFO segtask_v1.trainer.validation:   Val: loss=0.8030, pooled_mean_dice=0.6834, per_class=['0.6834'], iou=0.5191, recall=0.9769, precision=0.5255, vol_sim=0.6996, mcc=0.7090, min_class_dice=0.6834, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8682, per_class_sd=['0.8682'], combined(w=0.50)=0.7758, balanced=0.7003
[2026-06-22 19:03:53] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 19:03:53] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7003 at epoch 206
[2026-06-22 19:03:53] INFO segtask_v1.trainer.trainer: Epoch 206/400 | LR=4.87e-04 | loss=1.4569 | val_dice=0.6834 | best=0.7003 (ep206) | 04:02:14 | L_main=0.6190 L_aux_1=0.6211(w=0.5) L_aux_2=0.7371(w=0.5)
[2026-06-22 19:03:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 206): 11666.1 MiB
[2026-06-22 19:05:02] INFO segtask_v1.trainer.validation:   Val: loss=0.7677, pooled_mean_dice=0.6688, per_class=['0.6688'], iou=0.5024, recall=0.9794, precision=0.5078, vol_sim=0.6829, mcc=0.6947, min_class_dice=0.6688, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8534, per_class_sd=['0.8534'], combined(w=0.50)=0.7611, balanced=0.6852
[2026-06-22 19:05:02] INFO segtask_v1.trainer.trainer: Epoch 207/400 | LR=4.83e-04 | loss=1.4742 | val_dice=0.6688 | best=0.7003 (ep206) | 04:03:23 | L_main=0.6254 L_aux_1=0.6325(w=0.5) L_aux_2=0.7448(w=0.5)
[2026-06-22 19:05:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 207): 11666.1 MiB
[2026-06-22 19:06:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8506, pooled_mean_dice=0.6564, per_class=['0.6564'], iou=0.4885, recall=0.9795, precision=0.4936, vol_sim=0.6701, mcc=0.6861, min_class_dice=0.6564, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8347, per_class_sd=['0.8347'], combined(w=0.50)=0.7455, balanced=0.6717
[2026-06-22 19:06:12] INFO segtask_v1.trainer.trainer: Epoch 208/400 | LR=4.79e-04 | loss=1.6083 | val_dice=0.6564 | best=0.7003 (ep206) | 04:04:34 | L_main=0.7000 L_aux_1=0.6964(w=0.5) L_aux_2=0.8028(w=0.5)
[2026-06-22 19:06:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 208): 11666.1 MiB
[2026-06-22 19:07:21] INFO segtask_v1.trainer.validation:   Val: loss=0.8361, pooled_mean_dice=0.6430, per_class=['0.6430'], iou=0.4738, recall=0.9767, precision=0.4792, vol_sim=0.6583, mcc=0.6757, min_class_dice=0.6430, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8399, per_class_sd=['0.8399'], combined(w=0.50)=0.7414, balanced=0.6612
[2026-06-22 19:07:21] INFO segtask_v1.trainer.trainer: Epoch 209/400 | LR=4.75e-04 | loss=1.4540 | val_dice=0.6430 | best=0.7003 (ep206) | 04:05:43 | L_main=0.6183 L_aux_1=0.6171(w=0.5) L_aux_2=0.7422(w=0.5)
[2026-06-22 19:07:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 209): 11666.1 MiB
[2026-06-22 19:08:34] INFO segtask_v1.trainer.validation:   Val: loss=0.8058, pooled_mean_dice=0.6557, per_class=['0.6557'], iou=0.4878, recall=0.9776, precision=0.4933, vol_sim=0.6707, mcc=0.6855, min_class_dice=0.6557, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8466, per_class_sd=['0.8466'], combined(w=0.50)=0.7511, balanced=0.6731
[2026-06-22 19:08:34] INFO segtask_v1.trainer.trainer: Epoch 210/400 | LR=4.71e-04 | loss=1.4605 | val_dice=0.6557 | best=0.7003 (ep206) | 04:06:55 | L_main=0.6228 L_aux_1=0.6223(w=0.5) L_aux_2=0.7325(w=0.5)
[2026-06-22 19:08:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 210): 11666.1 MiB
[2026-06-22 19:09:46] INFO segtask_v1.trainer.validation:   Val: loss=0.8423, pooled_mean_dice=0.6298, per_class=['0.6298'], iou=0.4597, recall=0.9720, precision=0.4658, vol_sim=0.6480, mcc=0.6638, min_class_dice=0.6298, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8405, per_class_sd=['0.8405'], combined(w=0.50)=0.7352, balanced=0.6499
[2026-06-22 19:09:46] INFO segtask_v1.trainer.trainer: Epoch 211/400 | LR=4.67e-04 | loss=1.4729 | val_dice=0.6298 | best=0.7003 (ep206) | 04:08:08 | L_main=0.6278 L_aux_1=0.6356(w=0.5) L_aux_2=0.7336(w=0.5)
[2026-06-22 19:09:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 211): 11666.1 MiB
[2026-06-22 19:10:57] INFO segtask_v1.trainer.validation:   Val: loss=0.8214, pooled_mean_dice=0.6545, per_class=['0.6545'], iou=0.4864, recall=0.9729, precision=0.4931, vol_sim=0.6727, mcc=0.6819, min_class_dice=0.6545, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.8304, per_class_sd=['0.8304'], combined(w=0.50)=0.7424, balanced=0.6692
[2026-06-22 19:10:57] INFO segtask_v1.trainer.trainer: Epoch 212/400 | LR=4.63e-04 | loss=1.4515 | val_dice=0.6545 | best=0.7003 (ep206) | 04:09:19 | L_main=0.6175 L_aux_1=0.6249(w=0.5) L_aux_2=0.7295(w=0.5)
[2026-06-22 19:10:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 212): 11666.1 MiB
[2026-06-22 19:12:05] INFO segtask_v1.trainer.validation:   Val: loss=0.8351, pooled_mean_dice=0.6572, per_class=['0.6572'], iou=0.4895, recall=0.9789, precision=0.4947, vol_sim=0.6714, mcc=0.6854, min_class_dice=0.6572, coverage=[84]/88 samples, pooled_mean_surface_dice@2px=0.8326, per_class_sd=['0.8326'], combined(w=0.50)=0.7449, balanced=0.6720
[2026-06-22 19:12:05] INFO segtask_v1.trainer.trainer: Epoch 213/400 | LR=4.59e-04 | loss=1.4305 | val_dice=0.6572 | best=0.7003 (ep206) | 04:10:27 | L_main=0.6089 L_aux_1=0.6065(w=0.5) L_aux_2=0.7188(w=0.5)
[2026-06-22 19:12:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 213): 11666.1 MiB
[2026-06-22 19:13:14] INFO segtask_v1.trainer.validation:   Val: loss=0.8141, pooled_mean_dice=0.6469, per_class=['0.6469'], iou=0.4781, recall=0.9784, precision=0.4832, vol_sim=0.6612, mcc=0.6779, min_class_dice=0.6469, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8334, per_class_sd=['0.8334'], combined(w=0.50)=0.7401, balanced=0.6634
[2026-06-22 19:13:14] INFO segtask_v1.trainer.trainer: Epoch 214/400 | LR=4.55e-04 | loss=1.4919 | val_dice=0.6469 | best=0.7003 (ep206) | 04:11:36 | L_main=0.6292 L_aux_1=0.6267(w=0.5) L_aux_2=0.7791(w=0.5)
[2026-06-22 19:13:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 214): 11666.1 MiB
[2026-06-22 19:14:24] INFO segtask_v1.trainer.validation:   Val: loss=0.8359, pooled_mean_dice=0.6400, per_class=['0.6400'], iou=0.4706, recall=0.9772, precision=0.4758, vol_sim=0.6549, mcc=0.6730, min_class_dice=0.6400, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8449, per_class_sd=['0.8449'], combined(w=0.50)=0.7424, balanced=0.6593
[2026-06-22 19:14:24] INFO segtask_v1.trainer.trainer: Epoch 215/400 | LR=4.51e-04 | loss=1.4618 | val_dice=0.6400 | best=0.7003 (ep206) | 04:12:45 | L_main=0.6178 L_aux_1=0.6140(w=0.5) L_aux_2=0.7536(w=0.5)
[2026-06-22 19:14:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 215): 11666.1 MiB
[2026-06-22 19:15:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8432, pooled_mean_dice=0.6464, per_class=['0.6464'], iou=0.4775, recall=0.9754, precision=0.4833, vol_sim=0.6627, mcc=0.6792, min_class_dice=0.6464, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8492, per_class_sd=['0.8492'], combined(w=0.50)=0.7478, balanced=0.6656
[2026-06-22 19:15:36] INFO segtask_v1.trainer.trainer: Epoch 216/400 | LR=4.47e-04 | loss=1.4674 | val_dice=0.6464 | best=0.7003 (ep206) | 04:13:57 | L_main=0.6152 L_aux_1=0.6093(w=0.5) L_aux_2=0.7708(w=0.5)
[2026-06-22 19:15:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 216): 11666.1 MiB
[2026-06-22 19:16:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8768, pooled_mean_dice=0.6410, per_class=['0.6410'], iou=0.4717, recall=0.9797, precision=0.4763, vol_sim=0.6543, mcc=0.6742, min_class_dice=0.6410, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8324, per_class_sd=['0.8324'], combined(w=0.50)=0.7367, balanced=0.6583
[2026-06-22 19:16:47] INFO segtask_v1.trainer.trainer: Epoch 217/400 | LR=4.43e-04 | loss=1.4791 | val_dice=0.6410 | best=0.7003 (ep206) | 04:15:08 | L_main=0.6168 L_aux_1=0.6175(w=0.5) L_aux_2=0.7862(w=0.5)
[2026-06-22 19:16:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 217): 11666.1 MiB
[2026-06-22 19:17:57] INFO segtask_v1.trainer.validation:   Val: loss=0.8253, pooled_mean_dice=0.6663, per_class=['0.6663'], iou=0.4996, recall=0.9752, precision=0.5060, vol_sim=0.6832, mcc=0.6941, min_class_dice=0.6663, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8562, per_class_sd=['0.8562'], combined(w=0.50)=0.7613, balanced=0.6837
[2026-06-22 19:17:57] INFO segtask_v1.trainer.trainer: Epoch 218/400 | LR=4.39e-04 | loss=1.5160 | val_dice=0.6663 | best=0.7003 (ep206) | 04:16:19 | L_main=0.6384 L_aux_1=0.6380(w=0.5) L_aux_2=0.7915(w=0.5)
[2026-06-22 19:17:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 218): 11666.1 MiB
[2026-06-22 19:19:06] INFO segtask_v1.trainer.validation:   Val: loss=0.7792, pooled_mean_dice=0.6730, per_class=['0.6730'], iou=0.5072, recall=0.9750, precision=0.5139, vol_sim=0.6903, mcc=0.6978, min_class_dice=0.6730, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8520, per_class_sd=['0.8520'], combined(w=0.50)=0.7625, balanced=0.6886
[2026-06-22 19:19:06] INFO segtask_v1.trainer.trainer: Epoch 219/400 | LR=4.35e-04 | loss=1.4890 | val_dice=0.6730 | best=0.7003 (ep206) | 04:17:28 | L_main=0.6323 L_aux_1=0.6353(w=0.5) L_aux_2=0.7594(w=0.5)
[2026-06-22 19:19:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 219): 11666.1 MiB
[2026-06-22 19:20:15] INFO segtask_v1.trainer.validation:   Val: loss=0.8338, pooled_mean_dice=0.6601, per_class=['0.6601'], iou=0.4926, recall=0.9801, precision=0.4976, vol_sim=0.6735, mcc=0.6885, min_class_dice=0.6601, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8381, per_class_sd=['0.8381'], combined(w=0.50)=0.7491, balanced=0.6754
[2026-06-22 19:20:15] INFO segtask_v1.trainer.trainer: Epoch 220/400 | LR=4.31e-04 | loss=1.4432 | val_dice=0.6601 | best=0.7003 (ep206) | 04:18:37 | L_main=0.6107 L_aux_1=0.6148(w=0.5) L_aux_2=0.7337(w=0.5)
[2026-06-22 19:20:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 220): 11666.1 MiB
[2026-06-22 19:21:26] INFO segtask_v1.trainer.validation:   Val: loss=0.8101, pooled_mean_dice=0.6585, per_class=['0.6585'], iou=0.4909, recall=0.9830, precision=0.4951, vol_sim=0.6699, mcc=0.6887, min_class_dice=0.6585, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8482, per_class_sd=['0.8482'], combined(w=0.50)=0.7533, balanced=0.6758
[2026-06-22 19:21:26] INFO segtask_v1.trainer.trainer: Epoch 221/400 | LR=4.27e-04 | loss=1.4588 | val_dice=0.6585 | best=0.7003 (ep206) | 04:19:48 | L_main=0.6139 L_aux_1=0.6183(w=0.5) L_aux_2=0.7555(w=0.5)
[2026-06-22 19:21:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 221): 11666.1 MiB
[2026-06-22 19:22:35] INFO segtask_v1.trainer.validation:   Val: loss=0.8082, pooled_mean_dice=0.6690, per_class=['0.6690'], iou=0.5027, recall=0.9831, precision=0.5071, vol_sim=0.6805, mcc=0.6974, min_class_dice=0.6690, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8449, per_class_sd=['0.8449'], combined(w=0.50)=0.7570, balanced=0.6843
[2026-06-22 19:22:36] INFO segtask_v1.trainer.trainer: Epoch 222/400 | LR=4.23e-04 | loss=1.4537 | val_dice=0.6690 | best=0.7003 (ep206) | 04:20:57 | L_main=0.6173 L_aux_1=0.6214(w=0.5) L_aux_2=0.7331(w=0.5)
[2026-06-22 19:22:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 222): 11666.1 MiB
[2026-06-22 19:23:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8567, pooled_mean_dice=0.6235, per_class=['0.6235'], iou=0.4530, recall=0.9794, precision=0.4573, vol_sim=0.6366, mcc=0.6602, min_class_dice=0.6235, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8282, per_class_sd=['0.8282'], combined(w=0.50)=0.7259, balanced=0.6427
[2026-06-22 19:23:45] INFO segtask_v1.trainer.trainer: Epoch 223/400 | LR=4.19e-04 | loss=1.4655 | val_dice=0.6235 | best=0.7003 (ep206) | 04:22:07 | L_main=0.6189 L_aux_1=0.6284(w=0.5) L_aux_2=0.7480(w=0.5)
[2026-06-22 19:23:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 223): 11666.1 MiB
[2026-06-22 19:24:54] INFO segtask_v1.trainer.validation:   Val: loss=0.8229, pooled_mean_dice=0.6601, per_class=['0.6601'], iou=0.4927, recall=0.9774, precision=0.4984, vol_sim=0.6754, mcc=0.6879, min_class_dice=0.6601, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8427, per_class_sd=['0.8427'], combined(w=0.50)=0.7514, balanced=0.6761
[2026-06-22 19:24:54] INFO segtask_v1.trainer.trainer: Epoch 224/400 | LR=4.16e-04 | loss=1.4260 | val_dice=0.6601 | best=0.7003 (ep206) | 04:23:15 | L_main=0.6064 L_aux_1=0.6033(w=0.5) L_aux_2=0.7193(w=0.5)
[2026-06-22 19:24:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 224): 11666.1 MiB
[2026-06-22 19:26:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8463, pooled_mean_dice=0.6491, per_class=['0.6491'], iou=0.4805, recall=0.9799, precision=0.4853, vol_sim=0.6624, mcc=0.6812, min_class_dice=0.6491, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8515, per_class_sd=['0.8515'], combined(w=0.50)=0.7503, balanced=0.6683
[2026-06-22 19:26:03] INFO segtask_v1.trainer.trainer: Epoch 225/400 | LR=4.12e-04 | loss=1.4487 | val_dice=0.6491 | best=0.7003 (ep206) | 04:24:24 | L_main=0.6153 L_aux_1=0.6156(w=0.5) L_aux_2=0.7289(w=0.5)
[2026-06-22 19:26:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 225): 11666.1 MiB
[2026-06-22 19:27:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8295, pooled_mean_dice=0.6244, per_class=['0.6244'], iou=0.4539, recall=0.9812, precision=0.4579, vol_sim=0.6364, mcc=0.6613, min_class_dice=0.6244, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8469, per_class_sd=['0.8469'], combined(w=0.50)=0.7356, balanced=0.6463
[2026-06-22 19:27:12] INFO segtask_v1.trainer.trainer: Epoch 226/400 | LR=4.08e-04 | loss=1.4612 | val_dice=0.6244 | best=0.7003 (ep206) | 04:25:34 | L_main=0.6225 L_aux_1=0.6185(w=0.5) L_aux_2=0.7436(w=0.5)
[2026-06-22 19:27:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 226): 11666.1 MiB
[2026-06-22 19:28:25] INFO segtask_v1.trainer.validation:   Val: loss=0.7746, pooled_mean_dice=0.6665, per_class=['0.6665'], iou=0.4998, recall=0.9829, precision=0.5041, vol_sim=0.6781, mcc=0.6938, min_class_dice=0.6665, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8403, per_class_sd=['0.8403'], combined(w=0.50)=0.7534, balanced=0.6812
[2026-06-22 19:28:25] INFO segtask_v1.trainer.trainer: Epoch 227/400 | LR=4.04e-04 | loss=1.4687 | val_dice=0.6665 | best=0.7003 (ep206) | 04:26:46 | L_main=0.6246 L_aux_1=0.6237(w=0.5) L_aux_2=0.7465(w=0.5)
[2026-06-22 19:28:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 227): 11666.1 MiB
[2026-06-22 19:29:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8170, pooled_mean_dice=0.6720, per_class=['0.6720'], iou=0.5060, recall=0.9809, precision=0.5111, vol_sim=0.6851, mcc=0.6998, min_class_dice=0.6720, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8511, per_class_sd=['0.8511'], combined(w=0.50)=0.7615, balanced=0.6878
[2026-06-22 19:29:36] INFO segtask_v1.trainer.trainer: Epoch 228/400 | LR=4.00e-04 | loss=1.4335 | val_dice=0.6720 | best=0.7003 (ep206) | 04:27:57 | L_main=0.6096 L_aux_1=0.6103(w=0.5) L_aux_2=0.7240(w=0.5)
[2026-06-22 19:29:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 228): 11666.1 MiB
[2026-06-22 19:30:45] INFO segtask_v1.trainer.validation:   Val: loss=0.7772, pooled_mean_dice=0.6761, per_class=['0.6761'], iou=0.5107, recall=0.9701, precision=0.5189, vol_sim=0.6970, mcc=0.7007, min_class_dice=0.6761, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8537, per_class_sd=['0.8537'], combined(w=0.50)=0.7649, balanced=0.6915
[2026-06-22 19:30:45] INFO segtask_v1.trainer.trainer: Epoch 229/400 | LR=3.96e-04 | loss=1.4352 | val_dice=0.6761 | best=0.7003 (ep206) | 04:29:06 | L_main=0.6107 L_aux_1=0.6090(w=0.5) L_aux_2=0.7241(w=0.5)
[2026-06-22 19:30:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 229): 11666.1 MiB
[2026-06-22 19:31:54] INFO segtask_v1.trainer.validation:   Val: loss=0.8044, pooled_mean_dice=0.6902, per_class=['0.6902'], iou=0.5269, recall=0.9810, precision=0.5323, vol_sim=0.7035, mcc=0.7147, min_class_dice=0.6902, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.8548, per_class_sd=['0.8548'], combined(w=0.50)=0.7725, balanced=0.7039
[2026-06-22 19:31:59] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 19:31:59] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7039 at epoch 230
[2026-06-22 19:31:59] INFO segtask_v1.trainer.trainer: Epoch 230/400 | LR=3.92e-04 | loss=1.4530 | val_dice=0.6902 | best=0.7039 (ep230) | 04:30:20 | L_main=0.6178 L_aux_1=0.6162(w=0.5) L_aux_2=0.7336(w=0.5)
[2026-06-22 19:31:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 230): 11666.1 MiB
[2026-06-22 19:33:09] INFO segtask_v1.trainer.validation:   Val: loss=0.7763, pooled_mean_dice=0.6784, per_class=['0.6784'], iou=0.5134, recall=0.9849, precision=0.5174, vol_sim=0.6889, mcc=0.7026, min_class_dice=0.6784, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8411, per_class_sd=['0.8411'], combined(w=0.50)=0.7597, balanced=0.6914
[2026-06-22 19:33:09] INFO segtask_v1.trainer.trainer: Epoch 231/400 | LR=3.88e-04 | loss=1.4415 | val_dice=0.6784 | best=0.7039 (ep230) | 04:31:31 | L_main=0.6143 L_aux_1=0.6162(w=0.5) L_aux_2=0.7192(w=0.5)
[2026-06-22 19:33:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 231): 11666.1 MiB
[2026-06-22 19:34:18] INFO segtask_v1.trainer.validation:   Val: loss=0.7804, pooled_mean_dice=0.6849, per_class=['0.6849'], iou=0.5208, recall=0.9790, precision=0.5267, vol_sim=0.6996, mcc=0.7090, min_class_dice=0.6849, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8409, per_class_sd=['0.8409'], combined(w=0.50)=0.7629, balanced=0.6969
[2026-06-22 19:34:18] INFO segtask_v1.trainer.trainer: Epoch 232/400 | LR=3.84e-04 | loss=1.4343 | val_dice=0.6849 | best=0.7039 (ep230) | 04:32:39 | L_main=0.6100 L_aux_1=0.6075(w=0.5) L_aux_2=0.7182(w=0.5)
[2026-06-22 19:34:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 232): 11666.1 MiB
[2026-06-22 19:35:26] INFO segtask_v1.trainer.validation:   Val: loss=0.8579, pooled_mean_dice=0.6541, per_class=['0.6541'], iou=0.4860, recall=0.9760, precision=0.4919, vol_sim=0.6702, mcc=0.6850, min_class_dice=0.6541, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8407, per_class_sd=['0.8407'], combined(w=0.50)=0.7474, balanced=0.6708
[2026-06-22 19:35:26] INFO segtask_v1.trainer.trainer: Epoch 233/400 | LR=3.81e-04 | loss=1.4278 | val_dice=0.6541 | best=0.7039 (ep230) | 04:33:48 | L_main=0.6062 L_aux_1=0.6061(w=0.5) L_aux_2=0.7253(w=0.5)
[2026-06-22 19:35:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 233): 11666.1 MiB
[2026-06-22 19:36:38] INFO segtask_v1.trainer.validation:   Val: loss=0.8168, pooled_mean_dice=0.6624, per_class=['0.6624'], iou=0.4952, recall=0.9700, precision=0.5029, vol_sim=0.6829, mcc=0.6892, min_class_dice=0.6624, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8564, per_class_sd=['0.8564'], combined(w=0.50)=0.7594, balanced=0.6802
[2026-06-22 19:36:38] INFO segtask_v1.trainer.trainer: Epoch 234/400 | LR=3.77e-04 | loss=1.4475 | val_dice=0.6624 | best=0.7039 (ep230) | 04:34:59 | L_main=0.6133 L_aux_1=0.6163(w=0.5) L_aux_2=0.7297(w=0.5)
[2026-06-22 19:36:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 234): 11666.1 MiB
[2026-06-22 19:37:50] INFO segtask_v1.trainer.validation:   Val: loss=0.8158, pooled_mean_dice=0.6624, per_class=['0.6624'], iou=0.4952, recall=0.9763, precision=0.5012, vol_sim=0.6785, mcc=0.6905, min_class_dice=0.6624, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8528, per_class_sd=['0.8528'], combined(w=0.50)=0.7576, balanced=0.6797
[2026-06-22 19:37:50] INFO segtask_v1.trainer.trainer: Epoch 235/400 | LR=3.73e-04 | loss=1.4729 | val_dice=0.6624 | best=0.7039 (ep230) | 04:36:12 | L_main=0.6250 L_aux_1=0.6280(w=0.5) L_aux_2=0.7464(w=0.5)
[2026-06-22 19:37:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 235): 11666.1 MiB
[2026-06-22 19:39:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8150, pooled_mean_dice=0.6482, per_class=['0.6482'], iou=0.4796, recall=0.9753, precision=0.4854, vol_sim=0.6646, mcc=0.6793, min_class_dice=0.6482, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8492, per_class_sd=['0.8492'], combined(w=0.50)=0.7487, balanced=0.6671
[2026-06-22 19:39:03] INFO segtask_v1.trainer.trainer: Epoch 236/400 | LR=3.69e-04 | loss=1.4144 | val_dice=0.6482 | best=0.7039 (ep230) | 04:37:24 | L_main=0.5997 L_aux_1=0.5992(w=0.5) L_aux_2=0.7130(w=0.5)
[2026-06-22 19:39:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 236): 11666.1 MiB
[2026-06-22 19:40:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8191, pooled_mean_dice=0.6549, per_class=['0.6549'], iou=0.4868, recall=0.9781, precision=0.4922, vol_sim=0.6696, mcc=0.6847, min_class_dice=0.6549, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8379, per_class_sd=['0.8379'], combined(w=0.50)=0.7464, balanced=0.6710
[2026-06-22 19:40:12] INFO segtask_v1.trainer.trainer: Epoch 237/400 | LR=3.65e-04 | loss=1.4294 | val_dice=0.6549 | best=0.7039 (ep230) | 04:38:34 | L_main=0.6071 L_aux_1=0.6097(w=0.5) L_aux_2=0.7240(w=0.5)
[2026-06-22 19:40:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 237): 11666.1 MiB
[2026-06-22 19:41:21] INFO segtask_v1.trainer.validation:   Val: loss=0.7928, pooled_mean_dice=0.6556, per_class=['0.6556'], iou=0.4877, recall=0.9768, precision=0.4934, vol_sim=0.6712, mcc=0.6840, min_class_dice=0.6556, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8393, per_class_sd=['0.8393'], combined(w=0.50)=0.7474, balanced=0.6717
[2026-06-22 19:41:21] INFO segtask_v1.trainer.trainer: Epoch 238/400 | LR=3.61e-04 | loss=1.4303 | val_dice=0.6556 | best=0.7039 (ep230) | 04:39:43 | L_main=0.6084 L_aux_1=0.6084(w=0.5) L_aux_2=0.7218(w=0.5)
[2026-06-22 19:41:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 238): 11666.1 MiB
[2026-06-22 19:42:31] INFO segtask_v1.trainer.validation:   Val: loss=0.8124, pooled_mean_dice=0.6518, per_class=['0.6518'], iou=0.4835, recall=0.9724, precision=0.4902, vol_sim=0.6703, mcc=0.6822, min_class_dice=0.6518, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8474, per_class_sd=['0.8474'], combined(w=0.50)=0.7496, balanced=0.6699
[2026-06-22 19:42:31] INFO segtask_v1.trainer.trainer: Epoch 239/400 | LR=3.58e-04 | loss=1.4055 | val_dice=0.6518 | best=0.7039 (ep230) | 04:40:52 | L_main=0.5951 L_aux_1=0.5977(w=0.5) L_aux_2=0.7138(w=0.5)
[2026-06-22 19:42:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 239): 11666.1 MiB
[2026-06-22 19:43:40] INFO segtask_v1.trainer.validation:   Val: loss=0.7872, pooled_mean_dice=0.6724, per_class=['0.6724'], iou=0.5065, recall=0.9783, precision=0.5123, vol_sim=0.6874, mcc=0.6980, min_class_dice=0.6724, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8524, per_class_sd=['0.8524'], combined(w=0.50)=0.7624, balanced=0.6882
[2026-06-22 19:43:40] INFO segtask_v1.trainer.trainer: Epoch 240/400 | LR=3.54e-04 | loss=1.4121 | val_dice=0.6724 | best=0.7039 (ep230) | 04:42:01 | L_main=0.5992 L_aux_1=0.5976(w=0.5) L_aux_2=0.7155(w=0.5)
[2026-06-22 19:43:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 240): 11666.1 MiB
[2026-06-22 19:44:50] INFO segtask_v1.trainer.validation:   Val: loss=0.8301, pooled_mean_dice=0.6500, per_class=['0.6500'], iou=0.4815, recall=0.9724, precision=0.4882, vol_sim=0.6684, mcc=0.6802, min_class_dice=0.6500, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8541, per_class_sd=['0.8541'], combined(w=0.50)=0.7520, balanced=0.6693
[2026-06-22 19:44:50] INFO segtask_v1.trainer.trainer: Epoch 241/400 | LR=3.50e-04 | loss=1.4291 | val_dice=0.6500 | best=0.7039 (ep230) | 04:43:12 | L_main=0.6056 L_aux_1=0.6061(w=0.5) L_aux_2=0.7222(w=0.5)
[2026-06-22 19:44:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 241): 11666.1 MiB
[2026-06-22 19:46:00] INFO segtask_v1.trainer.validation:   Val: loss=0.7743, pooled_mean_dice=0.6955, per_class=['0.6955'], iou=0.5332, recall=0.9771, precision=0.5399, vol_sim=0.7118, mcc=0.7167, min_class_dice=0.6955, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8634, per_class_sd=['0.8634'], combined(w=0.50)=0.7794, balanced=0.7097
[2026-06-22 19:46:04] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 19:46:04] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7097 at epoch 242
[2026-06-22 19:46:04] INFO segtask_v1.trainer.trainer: Epoch 242/400 | LR=3.46e-04 | loss=1.4116 | val_dice=0.6955 | best=0.7097 (ep242) | 04:44:25 | L_main=0.5972 L_aux_1=0.6024(w=0.5) L_aux_2=0.7174(w=0.5)
[2026-06-22 19:46:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 242): 11666.1 MiB
[2026-06-22 19:47:14] INFO segtask_v1.trainer.validation:   Val: loss=0.8023, pooled_mean_dice=0.6727, per_class=['0.6727'], iou=0.5068, recall=0.9781, precision=0.5127, vol_sim=0.6878, mcc=0.6997, min_class_dice=0.6727, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8509, per_class_sd=['0.8509'], combined(w=0.50)=0.7618, balanced=0.6883
[2026-06-22 19:47:14] INFO segtask_v1.trainer.trainer: Epoch 243/400 | LR=3.42e-04 | loss=1.4244 | val_dice=0.6727 | best=0.7097 (ep242) | 04:45:35 | L_main=0.6041 L_aux_1=0.6079(w=0.5) L_aux_2=0.7155(w=0.5)
[2026-06-22 19:47:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 243): 11666.1 MiB
[2026-06-22 19:48:24] INFO segtask_v1.trainer.validation:   Val: loss=0.7812, pooled_mean_dice=0.6741, per_class=['0.6741'], iou=0.5084, recall=0.9689, precision=0.5169, vol_sim=0.6958, mcc=0.6988, min_class_dice=0.6741, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8477, per_class_sd=['0.8477'], combined(w=0.50)=0.7609, balanced=0.6888
[2026-06-22 19:48:24] INFO segtask_v1.trainer.trainer: Epoch 244/400 | LR=3.39e-04 | loss=1.4351 | val_dice=0.6741 | best=0.7097 (ep242) | 04:46:45 | L_main=0.6096 L_aux_1=0.6115(w=0.5) L_aux_2=0.7215(w=0.5)
[2026-06-22 19:48:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 244): 11666.1 MiB
[2026-06-22 19:49:34] INFO segtask_v1.trainer.validation:   Val: loss=0.7808, pooled_mean_dice=0.6774, per_class=['0.6774'], iou=0.5122, recall=0.9778, precision=0.5182, vol_sim=0.6928, mcc=0.7033, min_class_dice=0.6774, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8571, per_class_sd=['0.8571'], combined(w=0.50)=0.7673, balanced=0.6933
[2026-06-22 19:49:34] INFO segtask_v1.trainer.trainer: Epoch 245/400 | LR=3.35e-04 | loss=1.4008 | val_dice=0.6774 | best=0.7097 (ep242) | 04:47:55 | L_main=0.5934 L_aux_1=0.5938(w=0.5) L_aux_2=0.7103(w=0.5)
[2026-06-22 19:49:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 245): 11666.1 MiB
[2026-06-22 19:50:43] INFO segtask_v1.trainer.validation:   Val: loss=0.7998, pooled_mean_dice=0.6725, per_class=['0.6725'], iou=0.5066, recall=0.9769, precision=0.5127, vol_sim=0.6884, mcc=0.6996, min_class_dice=0.6725, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8623, per_class_sd=['0.8623'], combined(w=0.50)=0.7674, balanced=0.6900
[2026-06-22 19:50:43] INFO segtask_v1.trainer.trainer: Epoch 246/400 | LR=3.31e-04 | loss=1.4054 | val_dice=0.6725 | best=0.7097 (ep242) | 04:49:05 | L_main=0.5965 L_aux_1=0.5953(w=0.5) L_aux_2=0.7058(w=0.5)
[2026-06-22 19:50:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 246): 11666.1 MiB
[2026-06-22 19:51:54] INFO segtask_v1.trainer.validation:   Val: loss=0.8769, pooled_mean_dice=0.6594, per_class=['0.6594'], iou=0.4919, recall=0.9647, precision=0.5009, vol_sim=0.6835, mcc=0.6882, min_class_dice=0.6594, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8527, per_class_sd=['0.8527'], combined(w=0.50)=0.7561, balanced=0.6772
[2026-06-22 19:51:54] INFO segtask_v1.trainer.trainer: Epoch 247/400 | LR=3.27e-04 | loss=1.4029 | val_dice=0.6594 | best=0.7097 (ep242) | 04:50:15 | L_main=0.5966 L_aux_1=0.5942(w=0.5) L_aux_2=0.7002(w=0.5)
[2026-06-22 19:51:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 247): 11666.1 MiB
[2026-06-22 19:53:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8173, pooled_mean_dice=0.6633, per_class=['0.6633'], iou=0.4962, recall=0.9736, precision=0.5030, vol_sim=0.6813, mcc=0.6908, min_class_dice=0.6633, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8513, per_class_sd=['0.8513'], combined(w=0.50)=0.7573, balanced=0.6802
[2026-06-22 19:53:03] INFO segtask_v1.trainer.trainer: Epoch 248/400 | LR=3.24e-04 | loss=1.4012 | val_dice=0.6633 | best=0.7097 (ep242) | 04:51:25 | L_main=0.5957 L_aux_1=0.5963(w=0.5) L_aux_2=0.7030(w=0.5)
[2026-06-22 19:53:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 248): 11666.1 MiB
[2026-06-22 19:54:12] INFO segtask_v1.trainer.validation:   Val: loss=0.7684, pooled_mean_dice=0.6952, per_class=['0.6952'], iou=0.5328, recall=0.9729, precision=0.5408, vol_sim=0.7146, mcc=0.7183, min_class_dice=0.6952, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8776, per_class_sd=['0.8776'], combined(w=0.50)=0.7864, balanced=0.7119
[2026-06-22 19:54:16] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 19:54:16] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7119 at epoch 249
[2026-06-22 19:54:16] INFO segtask_v1.trainer.trainer: Epoch 249/400 | LR=3.20e-04 | loss=1.4300 | val_dice=0.6952 | best=0.7119 (ep249) | 04:52:38 | L_main=0.6059 L_aux_1=0.6108(w=0.5) L_aux_2=0.7230(w=0.5)
[2026-06-22 19:54:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 249): 11666.1 MiB
[2026-06-22 19:55:26] INFO segtask_v1.trainer.validation:   Val: loss=0.8137, pooled_mean_dice=0.6683, per_class=['0.6683'], iou=0.5018, recall=0.9804, precision=0.5069, vol_sim=0.6817, mcc=0.6969, min_class_dice=0.6683, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8566, per_class_sd=['0.8566'], combined(w=0.50)=0.7624, balanced=0.6855
[2026-06-22 19:55:26] INFO segtask_v1.trainer.trainer: Epoch 250/400 | LR=3.16e-04 | loss=1.3874 | val_dice=0.6683 | best=0.7119 (ep249) | 04:53:48 | L_main=0.5857 L_aux_1=0.5920(w=0.5) L_aux_2=0.7067(w=0.5)
[2026-06-22 19:55:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 250): 11666.1 MiB
[2026-06-22 19:56:37] INFO segtask_v1.trainer.validation:   Val: loss=0.7893, pooled_mean_dice=0.6811, per_class=['0.6811'], iou=0.5164, recall=0.9678, precision=0.5254, vol_sim=0.7038, mcc=0.7055, min_class_dice=0.6811, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8725, per_class_sd=['0.8725'], combined(w=0.50)=0.7768, balanced=0.6989
[2026-06-22 19:56:37] INFO segtask_v1.trainer.trainer: Epoch 251/400 | LR=3.13e-04 | loss=1.4314 | val_dice=0.6811 | best=0.7119 (ep249) | 04:54:58 | L_main=0.6078 L_aux_1=0.6064(w=0.5) L_aux_2=0.7261(w=0.5)
[2026-06-22 19:56:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 251): 11666.1 MiB
[2026-06-22 19:57:46] INFO segtask_v1.trainer.validation:   Val: loss=0.7887, pooled_mean_dice=0.6770, per_class=['0.6770'], iou=0.5117, recall=0.9797, precision=0.5172, vol_sim=0.6911, mcc=0.7029, min_class_dice=0.6770, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8561, per_class_sd=['0.8561'], combined(w=0.50)=0.7666, balanced=0.6928
[2026-06-22 19:57:46] INFO segtask_v1.trainer.trainer: Epoch 252/400 | LR=3.09e-04 | loss=1.4136 | val_dice=0.6770 | best=0.7119 (ep249) | 04:56:08 | L_main=0.5986 L_aux_1=0.6011(w=0.5) L_aux_2=0.7180(w=0.5)
[2026-06-22 19:57:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 252): 11666.1 MiB
[2026-06-22 19:58:55] INFO segtask_v1.trainer.validation:   Val: loss=0.7746, pooled_mean_dice=0.6783, per_class=['0.6783'], iou=0.5132, recall=0.9771, precision=0.5195, vol_sim=0.6942, mcc=0.7039, min_class_dice=0.6783, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8605, per_class_sd=['0.8605'], combined(w=0.50)=0.7694, balanced=0.6946
[2026-06-22 19:58:55] INFO segtask_v1.trainer.trainer: Epoch 253/400 | LR=3.05e-04 | loss=1.4066 | val_dice=0.6783 | best=0.7119 (ep249) | 04:57:16 | L_main=0.5972 L_aux_1=0.5991(w=0.5) L_aux_2=0.7063(w=0.5)
[2026-06-22 19:58:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 253): 11666.1 MiB
[2026-06-22 20:00:04] INFO segtask_v1.trainer.validation:   Val: loss=0.7981, pooled_mean_dice=0.6670, per_class=['0.6670'], iou=0.5003, recall=0.9780, precision=0.5060, vol_sim=0.6820, mcc=0.6954, min_class_dice=0.6670, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8575, per_class_sd=['0.8575'], combined(w=0.50)=0.7622, balanced=0.6845
[2026-06-22 20:00:04] INFO segtask_v1.trainer.trainer: Epoch 254/400 | LR=3.02e-04 | loss=1.4065 | val_dice=0.6670 | best=0.7119 (ep249) | 04:58:25 | L_main=0.5979 L_aux_1=0.5980(w=0.5) L_aux_2=0.7028(w=0.5)
[2026-06-22 20:00:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 254): 11666.1 MiB
[2026-06-22 20:01:12] INFO segtask_v1.trainer.validation:   Val: loss=0.8327, pooled_mean_dice=0.6745, per_class=['0.6745'], iou=0.5089, recall=0.9770, precision=0.5150, vol_sim=0.6904, mcc=0.7009, min_class_dice=0.6745, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8463, per_class_sd=['0.8463'], combined(w=0.50)=0.7604, balanced=0.6890
[2026-06-22 20:01:12] INFO segtask_v1.trainer.trainer: Epoch 255/400 | LR=2.98e-04 | loss=1.4478 | val_dice=0.6745 | best=0.7119 (ep249) | 04:59:34 | L_main=0.6176 L_aux_1=0.6151(w=0.5) L_aux_2=0.7344(w=0.5)
[2026-06-22 20:01:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 255): 11666.1 MiB
[2026-06-22 20:02:22] INFO segtask_v1.trainer.validation:   Val: loss=0.7996, pooled_mean_dice=0.6834, per_class=['0.6834'], iou=0.5191, recall=0.9785, precision=0.5251, vol_sim=0.6984, mcc=0.7093, min_class_dice=0.6834, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8614, per_class_sd=['0.8614'], combined(w=0.50)=0.7724, balanced=0.6992
[2026-06-22 20:02:22] INFO segtask_v1.trainer.trainer: Epoch 256/400 | LR=2.94e-04 | loss=1.4070 | val_dice=0.6834 | best=0.7119 (ep249) | 05:00:44 | L_main=0.5977 L_aux_1=0.5967(w=0.5) L_aux_2=0.7048(w=0.5)
[2026-06-22 20:02:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 256): 11666.1 MiB
[2026-06-22 20:03:32] INFO segtask_v1.trainer.validation:   Val: loss=0.8106, pooled_mean_dice=0.6673, per_class=['0.6673'], iou=0.5007, recall=0.9675, precision=0.5093, vol_sim=0.6897, mcc=0.6945, min_class_dice=0.6673, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8633, per_class_sd=['0.8633'], combined(w=0.50)=0.7653, balanced=0.6856
[2026-06-22 20:03:32] INFO segtask_v1.trainer.trainer: Epoch 257/400 | LR=2.91e-04 | loss=1.4029 | val_dice=0.6673 | best=0.7119 (ep249) | 05:01:54 | L_main=0.5959 L_aux_1=0.5952(w=0.5) L_aux_2=0.7047(w=0.5)
[2026-06-22 20:03:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 257): 11666.1 MiB
[2026-06-22 20:04:41] INFO segtask_v1.trainer.validation:   Val: loss=0.7882, pooled_mean_dice=0.6755, per_class=['0.6755'], iou=0.5100, recall=0.9766, precision=0.5163, vol_sim=0.6916, mcc=0.7015, min_class_dice=0.6755, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8565, per_class_sd=['0.8565'], combined(w=0.50)=0.7660, balanced=0.6915
[2026-06-22 20:04:41] INFO segtask_v1.trainer.trainer: Epoch 258/400 | LR=2.87e-04 | loss=1.4514 | val_dice=0.6755 | best=0.7119 (ep249) | 05:03:03 | L_main=0.6139 L_aux_1=0.6166(w=0.5) L_aux_2=0.7425(w=0.5)
[2026-06-22 20:04:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 258): 11666.1 MiB
[2026-06-22 20:05:51] INFO segtask_v1.trainer.validation:   Val: loss=0.8216, pooled_mean_dice=0.6528, per_class=['0.6528'], iou=0.4846, recall=0.9709, precision=0.4917, vol_sim=0.6724, mcc=0.6831, min_class_dice=0.6528, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8468, per_class_sd=['0.8468'], combined(w=0.50)=0.7498, balanced=0.6706
[2026-06-22 20:05:51] INFO segtask_v1.trainer.trainer: Epoch 259/400 | LR=2.84e-04 | loss=1.4229 | val_dice=0.6528 | best=0.7119 (ep249) | 05:04:12 | L_main=0.6034 L_aux_1=0.6070(w=0.5) L_aux_2=0.7235(w=0.5)
[2026-06-22 20:05:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 259): 11666.1 MiB
[2026-06-22 20:07:00] INFO segtask_v1.trainer.validation:   Val: loss=0.8231, pooled_mean_dice=0.6406, per_class=['0.6406'], iou=0.4712, recall=0.9776, precision=0.4763, vol_sim=0.6552, mcc=0.6746, min_class_dice=0.6406, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8539, per_class_sd=['0.8539'], combined(w=0.50)=0.7472, balanced=0.6613
[2026-06-22 20:07:00] INFO segtask_v1.trainer.trainer: Epoch 260/400 | LR=2.80e-04 | loss=1.3800 | val_dice=0.6406 | best=0.7119 (ep249) | 05:05:22 | L_main=0.5886 L_aux_1=0.5859(w=0.5) L_aux_2=0.6868(w=0.5)
[2026-06-22 20:07:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 260): 11666.1 MiB
[2026-06-22 20:08:11] INFO segtask_v1.trainer.validation:   Val: loss=0.8735, pooled_mean_dice=0.6390, per_class=['0.6390'], iou=0.4695, recall=0.9794, precision=0.4742, vol_sim=0.6525, mcc=0.6740, min_class_dice=0.6390, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8374, per_class_sd=['0.8374'], combined(w=0.50)=0.7382, balanced=0.6575
[2026-06-22 20:08:11] INFO segtask_v1.trainer.trainer: Epoch 261/400 | LR=2.76e-04 | loss=1.4018 | val_dice=0.6390 | best=0.7119 (ep249) | 05:06:32 | L_main=0.5959 L_aux_1=0.5966(w=0.5) L_aux_2=0.7054(w=0.5)
[2026-06-22 20:08:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 261): 11666.1 MiB
[2026-06-22 20:09:20] INFO segtask_v1.trainer.validation:   Val: loss=0.8062, pooled_mean_dice=0.6582, per_class=['0.6582'], iou=0.4905, recall=0.9792, precision=0.4957, vol_sim=0.6722, mcc=0.6867, min_class_dice=0.6582, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8335, per_class_sd=['0.8335'], combined(w=0.50)=0.7458, balanced=0.6730
[2026-06-22 20:09:20] INFO segtask_v1.trainer.trainer: Epoch 262/400 | LR=2.73e-04 | loss=1.4114 | val_dice=0.6582 | best=0.7119 (ep249) | 05:07:42 | L_main=0.6007 L_aux_1=0.5997(w=0.5) L_aux_2=0.7101(w=0.5)
[2026-06-22 20:09:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 262): 11666.1 MiB
[2026-06-22 20:10:31] INFO segtask_v1.trainer.validation:   Val: loss=0.8046, pooled_mean_dice=0.6623, per_class=['0.6623'], iou=0.4952, recall=0.9719, precision=0.5023, vol_sim=0.6815, mcc=0.6907, min_class_dice=0.6623, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8515, per_class_sd=['0.8515'], combined(w=0.50)=0.7569, balanced=0.6795
[2026-06-22 20:10:31] INFO segtask_v1.trainer.trainer: Epoch 263/400 | LR=2.69e-04 | loss=1.4204 | val_dice=0.6623 | best=0.7119 (ep249) | 05:08:52 | L_main=0.6070 L_aux_1=0.6024(w=0.5) L_aux_2=0.7109(w=0.5)
[2026-06-22 20:10:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 263): 11666.1 MiB
[2026-06-22 20:11:40] INFO segtask_v1.trainer.validation:   Val: loss=0.7861, pooled_mean_dice=0.6775, per_class=['0.6775'], iou=0.5123, recall=0.9761, precision=0.5188, vol_sim=0.6941, mcc=0.7028, min_class_dice=0.6775, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8542, per_class_sd=['0.8542'], combined(w=0.50)=0.7658, balanced=0.6929
[2026-06-22 20:11:40] INFO segtask_v1.trainer.trainer: Epoch 264/400 | LR=2.66e-04 | loss=1.3970 | val_dice=0.6775 | best=0.7119 (ep249) | 05:10:02 | L_main=0.5930 L_aux_1=0.5899(w=0.5) L_aux_2=0.7063(w=0.5)
[2026-06-22 20:11:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 264): 11666.1 MiB
[2026-06-22 20:12:50] INFO segtask_v1.trainer.validation:   Val: loss=0.8017, pooled_mean_dice=0.6542, per_class=['0.6542'], iou=0.4861, recall=0.9779, precision=0.4915, vol_sim=0.6689, mcc=0.6837, min_class_dice=0.6542, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8525, per_class_sd=['0.8525'], combined(w=0.50)=0.7533, balanced=0.6726
[2026-06-22 20:12:50] INFO segtask_v1.trainer.trainer: Epoch 265/400 | LR=2.62e-04 | loss=1.4247 | val_dice=0.6542 | best=0.7119 (ep249) | 05:11:12 | L_main=0.6014 L_aux_1=0.6029(w=0.5) L_aux_2=0.7350(w=0.5)
[2026-06-22 20:12:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 265): 11666.1 MiB
[2026-06-22 20:14:01] INFO segtask_v1.trainer.validation:   Val: loss=0.8443, pooled_mean_dice=0.6307, per_class=['0.6307'], iou=0.4606, recall=0.9820, precision=0.4645, vol_sim=0.6422, mcc=0.6671, min_class_dice=0.6307, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8430, per_class_sd=['0.8430'], combined(w=0.50)=0.7368, balanced=0.6512
[2026-06-22 20:14:01] INFO segtask_v1.trainer.trainer: Epoch 266/400 | LR=2.59e-04 | loss=1.4224 | val_dice=0.6307 | best=0.7119 (ep249) | 05:12:22 | L_main=0.6026 L_aux_1=0.6031(w=0.5) L_aux_2=0.7174(w=0.5)
[2026-06-22 20:14:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 266): 11666.1 MiB
[2026-06-22 20:15:10] INFO segtask_v1.trainer.validation:   Val: loss=0.7911, pooled_mean_dice=0.6769, per_class=['0.6769'], iou=0.5117, recall=0.9749, precision=0.5185, vol_sim=0.6943, mcc=0.7026, min_class_dice=0.6769, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8529, per_class_sd=['0.8529'], combined(w=0.50)=0.7649, balanced=0.6922
[2026-06-22 20:15:10] INFO segtask_v1.trainer.trainer: Epoch 267/400 | LR=2.55e-04 | loss=1.4149 | val_dice=0.6769 | best=0.7119 (ep249) | 05:13:32 | L_main=0.5999 L_aux_1=0.5975(w=0.5) L_aux_2=0.7222(w=0.5)
[2026-06-22 20:15:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 267): 11666.1 MiB
[2026-06-22 20:16:20] INFO segtask_v1.trainer.validation:   Val: loss=0.7665, pooled_mean_dice=0.6785, per_class=['0.6785'], iou=0.5135, recall=0.9822, precision=0.5183, vol_sim=0.6908, mcc=0.7046, min_class_dice=0.6785, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8544, per_class_sd=['0.8544'], combined(w=0.50)=0.7664, balanced=0.6938
[2026-06-22 20:16:20] INFO segtask_v1.trainer.trainer: Epoch 268/400 | LR=2.52e-04 | loss=1.4225 | val_dice=0.6785 | best=0.7119 (ep249) | 05:14:41 | L_main=0.6035 L_aux_1=0.6004(w=0.5) L_aux_2=0.7211(w=0.5)
[2026-06-22 20:16:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 268): 11666.1 MiB
[2026-06-22 20:17:29] INFO segtask_v1.trainer.validation:   Val: loss=0.7924, pooled_mean_dice=0.6765, per_class=['0.6765'], iou=0.5111, recall=0.9675, precision=0.5201, vol_sim=0.6993, mcc=0.7022, min_class_dice=0.6765, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8617, per_class_sd=['0.8617'], combined(w=0.50)=0.7691, balanced=0.6933
[2026-06-22 20:17:29] INFO segtask_v1.trainer.trainer: Epoch 269/400 | LR=2.48e-04 | loss=1.3997 | val_dice=0.6765 | best=0.7119 (ep249) | 05:15:50 | L_main=0.5949 L_aux_1=0.5911(w=0.5) L_aux_2=0.7047(w=0.5)
[2026-06-22 20:17:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 269): 11666.1 MiB
[2026-06-22 20:18:39] INFO segtask_v1.trainer.validation:   Val: loss=0.7805, pooled_mean_dice=0.6807, per_class=['0.6807'], iou=0.5160, recall=0.9758, precision=0.5227, vol_sim=0.6976, mcc=0.7064, min_class_dice=0.6807, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8573, per_class_sd=['0.8573'], combined(w=0.50)=0.7690, balanced=0.6962
[2026-06-22 20:18:39] INFO segtask_v1.trainer.trainer: Epoch 270/400 | LR=2.45e-04 | loss=1.4100 | val_dice=0.6807 | best=0.7119 (ep249) | 05:17:00 | L_main=0.5986 L_aux_1=0.6008(w=0.5) L_aux_2=0.7080(w=0.5)
[2026-06-22 20:18:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 270): 11666.1 MiB
[2026-06-22 20:19:49] INFO segtask_v1.trainer.validation:   Val: loss=0.8005, pooled_mean_dice=0.6591, per_class=['0.6591'], iou=0.4915, recall=0.9807, precision=0.4964, vol_sim=0.6721, mcc=0.6885, min_class_dice=0.6591, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8432, per_class_sd=['0.8432'], combined(w=0.50)=0.7512, balanced=0.6755
[2026-06-22 20:19:49] INFO segtask_v1.trainer.trainer: Epoch 271/400 | LR=2.42e-04 | loss=1.3960 | val_dice=0.6591 | best=0.7119 (ep249) | 05:18:11 | L_main=0.5922 L_aux_1=0.5917(w=0.5) L_aux_2=0.7048(w=0.5)
[2026-06-22 20:19:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 271): 11666.1 MiB
[2026-06-22 20:20:59] INFO segtask_v1.trainer.validation:   Val: loss=0.7835, pooled_mean_dice=0.6500, per_class=['0.6500'], iou=0.4814, recall=0.9777, precision=0.4868, vol_sim=0.6648, mcc=0.6803, min_class_dice=0.6500, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8489, per_class_sd=['0.8489'], combined(w=0.50)=0.7494, balanced=0.6685
[2026-06-22 20:20:59] INFO segtask_v1.trainer.trainer: Epoch 272/400 | LR=2.38e-04 | loss=1.4086 | val_dice=0.6500 | best=0.7119 (ep249) | 05:19:20 | L_main=0.6008 L_aux_1=0.5976(w=0.5) L_aux_2=0.7087(w=0.5)
[2026-06-22 20:20:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 272): 11666.1 MiB
[2026-06-22 20:22:08] INFO segtask_v1.trainer.validation:   Val: loss=0.7928, pooled_mean_dice=0.6776, per_class=['0.6776'], iou=0.5124, recall=0.9755, precision=0.5191, vol_sim=0.6946, mcc=0.7034, min_class_dice=0.6776, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8622, per_class_sd=['0.8622'], combined(w=0.50)=0.7699, balanced=0.6943
[2026-06-22 20:22:08] INFO segtask_v1.trainer.trainer: Epoch 273/400 | LR=2.35e-04 | loss=1.4074 | val_dice=0.6776 | best=0.7119 (ep249) | 05:20:30 | L_main=0.5977 L_aux_1=0.5985(w=0.5) L_aux_2=0.7126(w=0.5)
[2026-06-22 20:22:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 273): 11666.1 MiB
[2026-06-22 20:23:18] INFO segtask_v1.trainer.validation:   Val: loss=0.8520, pooled_mean_dice=0.6332, per_class=['0.6332'], iou=0.4633, recall=0.9703, precision=0.4699, vol_sim=0.6526, mcc=0.6685, min_class_dice=0.6332, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8584, per_class_sd=['0.8584'], combined(w=0.50)=0.7458, balanced=0.6556
[2026-06-22 20:23:18] INFO segtask_v1.trainer.trainer: Epoch 274/400 | LR=2.32e-04 | loss=1.3927 | val_dice=0.6332 | best=0.7119 (ep249) | 05:21:39 | L_main=0.5899 L_aux_1=0.5930(w=0.5) L_aux_2=0.6990(w=0.5)
[2026-06-22 20:23:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 274): 11666.1 MiB
[2026-06-22 20:24:27] INFO segtask_v1.trainer.validation:   Val: loss=0.7797, pooled_mean_dice=0.6710, per_class=['0.6710'], iou=0.5049, recall=0.9780, precision=0.5107, vol_sim=0.6861, mcc=0.6985, min_class_dice=0.6710, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8519, per_class_sd=['0.8519'], combined(w=0.50)=0.7615, balanced=0.6870
[2026-06-22 20:24:27] INFO segtask_v1.trainer.trainer: Epoch 275/400 | LR=2.28e-04 | loss=1.3848 | val_dice=0.6710 | best=0.7119 (ep249) | 05:22:49 | L_main=0.5871 L_aux_1=0.5836(w=0.5) L_aux_2=0.6991(w=0.5)
[2026-06-22 20:24:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 275): 11666.1 MiB
[2026-06-22 20:25:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8032, pooled_mean_dice=0.6572, per_class=['0.6572'], iou=0.4894, recall=0.9793, precision=0.4945, vol_sim=0.6711, mcc=0.6869, min_class_dice=0.6572, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8470, per_class_sd=['0.8470'], combined(w=0.50)=0.7521, balanced=0.6744
[2026-06-22 20:25:37] INFO segtask_v1.trainer.trainer: Epoch 276/400 | LR=2.25e-04 | loss=1.4116 | val_dice=0.6572 | best=0.7119 (ep249) | 05:23:58 | L_main=0.6001 L_aux_1=0.6006(w=0.5) L_aux_2=0.7132(w=0.5)
[2026-06-22 20:25:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 276): 11666.1 MiB
[2026-06-22 20:26:46] INFO segtask_v1.trainer.validation:   Val: loss=0.8142, pooled_mean_dice=0.6759, per_class=['0.6759'], iou=0.5105, recall=0.9772, precision=0.5166, vol_sim=0.6917, mcc=0.7018, min_class_dice=0.6759, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8629, per_class_sd=['0.8629'], combined(w=0.50)=0.7694, balanced=0.6930
[2026-06-22 20:26:46] INFO segtask_v1.trainer.trainer: Epoch 277/400 | LR=2.22e-04 | loss=1.3980 | val_dice=0.6759 | best=0.7119 (ep249) | 05:25:08 | L_main=0.5925 L_aux_1=0.5957(w=0.5) L_aux_2=0.7029(w=0.5)
[2026-06-22 20:26:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 277): 11666.1 MiB
[2026-06-22 20:27:56] INFO segtask_v1.trainer.validation:   Val: loss=0.7864, pooled_mean_dice=0.6722, per_class=['0.6722'], iou=0.5063, recall=0.9817, precision=0.5111, vol_sim=0.6848, mcc=0.7012, min_class_dice=0.6722, coverage=[66]/88 samples, pooled_mean_surface_dice@2px=0.8618, per_class_sd=['0.8618'], combined(w=0.50)=0.7670, balanced=0.6898
[2026-06-22 20:27:56] INFO segtask_v1.trainer.trainer: Epoch 278/400 | LR=2.18e-04 | loss=1.3754 | val_dice=0.6722 | best=0.7119 (ep249) | 05:26:18 | L_main=0.5833 L_aux_1=0.5840(w=0.5) L_aux_2=0.6891(w=0.5)
[2026-06-22 20:27:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 278): 11666.1 MiB
[2026-06-22 20:29:06] INFO segtask_v1.trainer.validation:   Val: loss=0.8152, pooled_mean_dice=0.6604, per_class=['0.6604'], iou=0.4930, recall=0.9681, precision=0.5012, vol_sim=0.6822, mcc=0.6878, min_class_dice=0.6604, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8574, per_class_sd=['0.8574'], combined(w=0.50)=0.7589, balanced=0.6787
[2026-06-22 20:29:06] INFO segtask_v1.trainer.trainer: Epoch 279/400 | LR=2.15e-04 | loss=1.4068 | val_dice=0.6604 | best=0.7119 (ep249) | 05:27:27 | L_main=0.5965 L_aux_1=0.5957(w=0.5) L_aux_2=0.7101(w=0.5)
[2026-06-22 20:29:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 279): 11666.1 MiB
[2026-06-22 20:30:16] INFO segtask_v1.trainer.validation:   Val: loss=0.7841, pooled_mean_dice=0.6699, per_class=['0.6699'], iou=0.5037, recall=0.9738, precision=0.5106, vol_sim=0.6879, mcc=0.6961, min_class_dice=0.6699, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8497, per_class_sd=['0.8497'], combined(w=0.50)=0.7598, balanced=0.6856
[2026-06-22 20:30:16] INFO segtask_v1.trainer.trainer: Epoch 280/400 | LR=2.12e-04 | loss=1.3788 | val_dice=0.6699 | best=0.7119 (ep249) | 05:28:38 | L_main=0.5859 L_aux_1=0.5801(w=0.5) L_aux_2=0.6926(w=0.5)
[2026-06-22 20:30:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 280): 11666.1 MiB
[2026-06-22 20:31:27] INFO segtask_v1.trainer.validation:   Val: loss=0.8053, pooled_mean_dice=0.6679, per_class=['0.6679'], iou=0.5014, recall=0.9785, precision=0.5070, vol_sim=0.6826, mcc=0.6971, min_class_dice=0.6679, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8658, per_class_sd=['0.8658'], combined(w=0.50)=0.7669, balanced=0.6867
[2026-06-22 20:31:27] INFO segtask_v1.trainer.trainer: Epoch 281/400 | LR=2.09e-04 | loss=1.3766 | val_dice=0.6679 | best=0.7119 (ep249) | 05:29:48 | L_main=0.5830 L_aux_1=0.5856(w=0.5) L_aux_2=0.6964(w=0.5)
[2026-06-22 20:31:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 281): 11666.1 MiB
[2026-06-22 20:32:36] INFO segtask_v1.trainer.validation:   Val: loss=0.7840, pooled_mean_dice=0.6691, per_class=['0.6691'], iou=0.5027, recall=0.9794, precision=0.5081, vol_sim=0.6832, mcc=0.6947, min_class_dice=0.6691, coverage=[85]/88 samples, pooled_mean_surface_dice@2px=0.8547, per_class_sd=['0.8547'], combined(w=0.50)=0.7619, balanced=0.6856
[2026-06-22 20:32:36] INFO segtask_v1.trainer.trainer: Epoch 282/400 | LR=2.05e-04 | loss=1.4027 | val_dice=0.6691 | best=0.7119 (ep249) | 05:30:58 | L_main=0.5944 L_aux_1=0.5935(w=0.5) L_aux_2=0.7022(w=0.5)
[2026-06-22 20:32:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 282): 11666.1 MiB
[2026-06-22 20:33:46] INFO segtask_v1.trainer.validation:   Val: loss=0.7803, pooled_mean_dice=0.6655, per_class=['0.6655'], iou=0.4987, recall=0.9744, precision=0.5053, vol_sim=0.6830, mcc=0.6920, min_class_dice=0.6655, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8490, per_class_sd=['0.8490'], combined(w=0.50)=0.7572, balanced=0.6817
[2026-06-22 20:33:46] INFO segtask_v1.trainer.trainer: Epoch 283/400 | LR=2.02e-04 | loss=1.3826 | val_dice=0.6655 | best=0.7119 (ep249) | 05:32:07 | L_main=0.5838 L_aux_1=0.5905(w=0.5) L_aux_2=0.6981(w=0.5)
[2026-06-22 20:33:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 283): 11666.1 MiB
[2026-06-22 20:34:54] INFO segtask_v1.trainer.validation:   Val: loss=0.8200, pooled_mean_dice=0.6579, per_class=['0.6579'], iou=0.4902, recall=0.9669, precision=0.4986, vol_sim=0.6804, mcc=0.6860, min_class_dice=0.6579, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8548, per_class_sd=['0.8548'], combined(w=0.50)=0.7563, balanced=0.6761
[2026-06-22 20:34:54] INFO segtask_v1.trainer.trainer: Epoch 284/400 | LR=1.99e-04 | loss=1.3904 | val_dice=0.6579 | best=0.7119 (ep249) | 05:33:16 | L_main=0.5869 L_aux_1=0.5872(w=0.5) L_aux_2=0.7121(w=0.5)
[2026-06-22 20:34:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 284): 11666.1 MiB
[2026-06-22 20:36:03] INFO segtask_v1.trainer.validation:   Val: loss=0.7871, pooled_mean_dice=0.6758, per_class=['0.6758'], iou=0.5104, recall=0.9716, precision=0.5181, vol_sim=0.6956, mcc=0.7023, min_class_dice=0.6758, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8667, per_class_sd=['0.8667'], combined(w=0.50)=0.7713, balanced=0.6935
[2026-06-22 20:36:03] INFO segtask_v1.trainer.trainer: Epoch 285/400 | LR=1.96e-04 | loss=1.3687 | val_dice=0.6758 | best=0.7119 (ep249) | 05:34:25 | L_main=0.5826 L_aux_1=0.5775(w=0.5) L_aux_2=0.6852(w=0.5)
[2026-06-22 20:36:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 285): 11666.1 MiB
[2026-06-22 20:37:14] INFO segtask_v1.trainer.validation:   Val: loss=0.7637, pooled_mean_dice=0.7030, per_class=['0.7030'], iou=0.5421, recall=0.9766, precision=0.5492, vol_sim=0.7199, mcc=0.7222, min_class_dice=0.7030, coverage=[84]/88 samples, pooled_mean_surface_dice@2px=0.8605, per_class_sd=['0.8605'], combined(w=0.50)=0.7818, balanced=0.7155
[2026-06-22 20:37:17] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 20:37:17] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7155 at epoch 286
[2026-06-22 20:37:17] INFO segtask_v1.trainer.trainer: Epoch 286/400 | LR=1.93e-04 | loss=1.4216 | val_dice=0.7030 | best=0.7155 (ep286) | 05:35:39 | L_main=0.6013 L_aux_1=0.6096(w=0.5) L_aux_2=0.7170(w=0.5)
[2026-06-22 20:37:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 286): 11666.1 MiB
[2026-06-22 20:38:26] INFO segtask_v1.trainer.validation:   Val: loss=0.8037, pooled_mean_dice=0.6686, per_class=['0.6686'], iou=0.5021, recall=0.9725, precision=0.5094, vol_sim=0.6875, mcc=0.6950, min_class_dice=0.6686, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8583, per_class_sd=['0.8583'], combined(w=0.50)=0.7634, balanced=0.6859
[2026-06-22 20:38:26] INFO segtask_v1.trainer.trainer: Epoch 287/400 | LR=1.90e-04 | loss=1.3939 | val_dice=0.6686 | best=0.7155 (ep286) | 05:36:48 | L_main=0.5913 L_aux_1=0.5907(w=0.5) L_aux_2=0.7034(w=0.5)
[2026-06-22 20:38:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 287): 11666.1 MiB
[2026-06-22 20:39:36] INFO segtask_v1.trainer.validation:   Val: loss=0.8013, pooled_mean_dice=0.6836, per_class=['0.6836'], iou=0.5193, recall=0.9688, precision=0.5281, vol_sim=0.7056, mcc=0.7069, min_class_dice=0.6836, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8620, per_class_sd=['0.8620'], combined(w=0.50)=0.7728, balanced=0.6993
[2026-06-22 20:39:36] INFO segtask_v1.trainer.trainer: Epoch 288/400 | LR=1.86e-04 | loss=1.3865 | val_dice=0.6836 | best=0.7155 (ep286) | 05:37:58 | L_main=0.5851 L_aux_1=0.5891(w=0.5) L_aux_2=0.7072(w=0.5)
[2026-06-22 20:39:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 288): 11666.1 MiB
[2026-06-22 20:40:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8335, pooled_mean_dice=0.6660, per_class=['0.6660'], iou=0.4993, recall=0.9707, precision=0.5069, vol_sim=0.6862, mcc=0.6947, min_class_dice=0.6660, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8624, per_class_sd=['0.8624'], combined(w=0.50)=0.7642, balanced=0.6845
[2026-06-22 20:40:47] INFO segtask_v1.trainer.trainer: Epoch 289/400 | LR=1.83e-04 | loss=1.3964 | val_dice=0.6660 | best=0.7155 (ep286) | 05:39:08 | L_main=0.5927 L_aux_1=0.5902(w=0.5) L_aux_2=0.7056(w=0.5)
[2026-06-22 20:40:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 289): 11666.1 MiB
[2026-06-22 20:41:56] INFO segtask_v1.trainer.validation:   Val: loss=0.7717, pooled_mean_dice=0.6980, per_class=['0.6980'], iou=0.5360, recall=0.9772, precision=0.5428, vol_sim=0.7142, mcc=0.7191, min_class_dice=0.6980, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8623, per_class_sd=['0.8623'], combined(w=0.50)=0.7801, balanced=0.7116
[2026-06-22 20:41:56] INFO segtask_v1.trainer.trainer: Epoch 290/400 | LR=1.80e-04 | loss=1.4050 | val_dice=0.6980 | best=0.7155 (ep286) | 05:40:18 | L_main=0.5980 L_aux_1=0.5936(w=0.5) L_aux_2=0.7085(w=0.5)
[2026-06-22 20:41:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 290): 11666.1 MiB
[2026-06-22 20:43:07] INFO segtask_v1.trainer.validation:   Val: loss=0.7848, pooled_mean_dice=0.6862, per_class=['0.6862'], iou=0.5223, recall=0.9823, precision=0.5273, vol_sim=0.6986, mcc=0.7106, min_class_dice=0.6862, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8640, per_class_sd=['0.8640'], combined(w=0.50)=0.7751, balanced=0.7019
[2026-06-22 20:43:07] INFO segtask_v1.trainer.trainer: Epoch 291/400 | LR=1.77e-04 | loss=1.4228 | val_dice=0.6862 | best=0.7155 (ep286) | 05:41:28 | L_main=0.6041 L_aux_1=0.6040(w=0.5) L_aux_2=0.7148(w=0.5)
[2026-06-22 20:43:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 291): 11666.1 MiB
[2026-06-22 20:44:16] INFO segtask_v1.trainer.validation:   Val: loss=0.8014, pooled_mean_dice=0.6638, per_class=['0.6638'], iou=0.4967, recall=0.9809, precision=0.5016, vol_sim=0.6767, mcc=0.6940, min_class_dice=0.6638, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8712, per_class_sd=['0.8712'], combined(w=0.50)=0.7675, balanced=0.6840
[2026-06-22 20:44:17] INFO segtask_v1.trainer.trainer: Epoch 292/400 | LR=1.74e-04 | loss=1.3802 | val_dice=0.6638 | best=0.7155 (ep286) | 05:42:38 | L_main=0.5843 L_aux_1=0.5876(w=0.5) L_aux_2=0.6976(w=0.5)
[2026-06-22 20:44:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 292): 11666.1 MiB
[2026-06-22 20:45:27] INFO segtask_v1.trainer.validation:   Val: loss=0.7545, pooled_mean_dice=0.6800, per_class=['0.6800'], iou=0.5151, recall=0.9740, precision=0.5223, vol_sim=0.6981, mcc=0.7020, min_class_dice=0.6800, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8529, per_class_sd=['0.8529'], combined(w=0.50)=0.7664, balanced=0.6945
[2026-06-22 20:45:27] INFO segtask_v1.trainer.trainer: Epoch 293/400 | LR=1.71e-04 | loss=1.3821 | val_dice=0.6800 | best=0.7155 (ep286) | 05:43:48 | L_main=0.5827 L_aux_1=0.5927(w=0.5) L_aux_2=0.7050(w=0.5)
[2026-06-22 20:45:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 293): 11666.1 MiB
[2026-06-22 20:46:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8234, pooled_mean_dice=0.6742, per_class=['0.6742'], iou=0.5086, recall=0.9783, precision=0.5144, vol_sim=0.6892, mcc=0.7013, min_class_dice=0.6742, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8561, per_class_sd=['0.8561'], combined(w=0.50)=0.7651, balanced=0.6905
[2026-06-22 20:46:37] INFO segtask_v1.trainer.trainer: Epoch 294/400 | LR=1.68e-04 | loss=1.3902 | val_dice=0.6742 | best=0.7155 (ep286) | 05:44:58 | L_main=0.5899 L_aux_1=0.5958(w=0.5) L_aux_2=0.6954(w=0.5)
[2026-06-22 20:46:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 294): 11666.1 MiB
[2026-06-22 20:47:46] INFO segtask_v1.trainer.validation:   Val: loss=0.7980, pooled_mean_dice=0.6624, per_class=['0.6624'], iou=0.4952, recall=0.9752, precision=0.5015, vol_sim=0.6792, mcc=0.6891, min_class_dice=0.6624, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8396, per_class_sd=['0.8396'], combined(w=0.50)=0.7510, balanced=0.6775
[2026-06-22 20:47:46] INFO segtask_v1.trainer.trainer: Epoch 295/400 | LR=1.65e-04 | loss=1.3675 | val_dice=0.6624 | best=0.7155 (ep286) | 05:46:07 | L_main=0.5805 L_aux_1=0.5779(w=0.5) L_aux_2=0.6877(w=0.5)
[2026-06-22 20:47:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 295): 11666.1 MiB
[2026-06-22 20:48:55] INFO segtask_v1.trainer.validation:   Val: loss=0.8439, pooled_mean_dice=0.6463, per_class=['0.6463'], iou=0.4774, recall=0.9645, precision=0.4859, vol_sim=0.6700, mcc=0.6761, min_class_dice=0.6463, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8455, per_class_sd=['0.8455'], combined(w=0.50)=0.7459, balanced=0.6647
[2026-06-22 20:48:55] INFO segtask_v1.trainer.trainer: Epoch 296/400 | LR=1.62e-04 | loss=1.3731 | val_dice=0.6463 | best=0.7155 (ep286) | 05:47:17 | L_main=0.5806 L_aux_1=0.5828(w=0.5) L_aux_2=0.7031(w=0.5)
[2026-06-22 20:48:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 296): 11666.1 MiB
[2026-06-22 20:50:04] INFO segtask_v1.trainer.validation:   Val: loss=0.8073, pooled_mean_dice=0.6584, per_class=['0.6584'], iou=0.4907, recall=0.9818, precision=0.4952, vol_sim=0.6706, mcc=0.6886, min_class_dice=0.6584, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8521, per_class_sd=['0.8521'], combined(w=0.50)=0.7552, balanced=0.6763
[2026-06-22 20:50:04] INFO segtask_v1.trainer.trainer: Epoch 297/400 | LR=1.59e-04 | loss=1.3911 | val_dice=0.6584 | best=0.7155 (ep286) | 05:48:26 | L_main=0.5886 L_aux_1=0.5869(w=0.5) L_aux_2=0.7077(w=0.5)
[2026-06-22 20:50:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 297): 11666.1 MiB
[2026-06-22 20:51:13] INFO segtask_v1.trainer.validation:   Val: loss=0.7759, pooled_mean_dice=0.6903, per_class=['0.6903'], iou=0.5271, recall=0.9809, precision=0.5326, vol_sim=0.7038, mcc=0.7150, min_class_dice=0.6903, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8621, per_class_sd=['0.8621'], combined(w=0.50)=0.7762, balanced=0.7052
[2026-06-22 20:51:13] INFO segtask_v1.trainer.trainer: Epoch 298/400 | LR=1.57e-04 | loss=1.3802 | val_dice=0.6903 | best=0.7155 (ep286) | 05:49:35 | L_main=0.5836 L_aux_1=0.5857(w=0.5) L_aux_2=0.6988(w=0.5)
[2026-06-22 20:51:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 298): 11666.1 MiB
[2026-06-22 20:52:23] INFO segtask_v1.trainer.validation:   Val: loss=0.8101, pooled_mean_dice=0.6698, per_class=['0.6698'], iou=0.5035, recall=0.9782, precision=0.5093, vol_sim=0.6847, mcc=0.6985, min_class_dice=0.6698, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.8563, per_class_sd=['0.8563'], combined(w=0.50)=0.7630, balanced=0.6868
[2026-06-22 20:52:23] INFO segtask_v1.trainer.trainer: Epoch 299/400 | LR=1.54e-04 | loss=1.3677 | val_dice=0.6698 | best=0.7155 (ep286) | 05:50:44 | L_main=0.5799 L_aux_1=0.5805(w=0.5) L_aux_2=0.6832(w=0.5)
[2026-06-22 20:52:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 299): 11666.1 MiB
[2026-06-22 20:53:32] INFO segtask_v1.trainer.validation:   Val: loss=0.8115, pooled_mean_dice=0.6780, per_class=['0.6780'], iou=0.5128, recall=0.9773, precision=0.5190, vol_sim=0.6937, mcc=0.7051, min_class_dice=0.6780, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8701, per_class_sd=['0.8701'], combined(w=0.50)=0.7740, balanced=0.6960
[2026-06-22 20:53:32] INFO segtask_v1.trainer.trainer: Epoch 300/400 | LR=1.51e-04 | loss=1.3755 | val_dice=0.6780 | best=0.7155 (ep286) | 05:51:54 | L_main=0.5821 L_aux_1=0.5830(w=0.5) L_aux_2=0.6994(w=0.5)
[2026-06-22 20:53:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 300): 11666.1 MiB
[2026-06-22 20:54:45] INFO segtask_v1.trainer.validation:   Val: loss=0.8029, pooled_mean_dice=0.6449, per_class=['0.6449'], iou=0.4759, recall=0.9765, precision=0.4814, vol_sim=0.6604, mcc=0.6775, min_class_dice=0.6449, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8528, per_class_sd=['0.8528'], combined(w=0.50)=0.7488, balanced=0.6648
[2026-06-22 20:54:45] INFO segtask_v1.trainer.trainer: Epoch 301/400 | LR=1.48e-04 | loss=1.3723 | val_dice=0.6449 | best=0.7155 (ep286) | 05:53:06 | L_main=0.5811 L_aux_1=0.5830(w=0.5) L_aux_2=0.6929(w=0.5)
[2026-06-22 20:54:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 301): 11666.1 MiB
[2026-06-22 20:55:54] INFO segtask_v1.trainer.validation:   Val: loss=0.7982, pooled_mean_dice=0.6790, per_class=['0.6790'], iou=0.5140, recall=0.9808, precision=0.5192, vol_sim=0.6923, mcc=0.7061, min_class_dice=0.6790, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8653, per_class_sd=['0.8653'], combined(w=0.50)=0.7721, balanced=0.6961
[2026-06-22 20:55:54] INFO segtask_v1.trainer.trainer: Epoch 302/400 | LR=1.45e-04 | loss=1.3791 | val_dice=0.6790 | best=0.7155 (ep286) | 05:54:16 | L_main=0.5829 L_aux_1=0.5856(w=0.5) L_aux_2=0.6999(w=0.5)
[2026-06-22 20:55:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 302): 11666.1 MiB
[2026-06-22 20:57:03] INFO segtask_v1.trainer.validation:   Val: loss=0.8055, pooled_mean_dice=0.6825, per_class=['0.6825'], iou=0.5180, recall=0.9781, precision=0.5241, vol_sim=0.6978, mcc=0.7073, min_class_dice=0.6825, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8562, per_class_sd=['0.8562'], combined(w=0.50)=0.7694, balanced=0.6975
[2026-06-22 20:57:03] INFO segtask_v1.trainer.trainer: Epoch 303/400 | LR=1.42e-04 | loss=1.3747 | val_dice=0.6825 | best=0.7155 (ep286) | 05:55:25 | L_main=0.5822 L_aux_1=0.5811(w=0.5) L_aux_2=0.6899(w=0.5)
[2026-06-22 20:57:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 303): 11666.1 MiB
[2026-06-22 20:58:14] INFO segtask_v1.trainer.validation:   Val: loss=0.7659, pooled_mean_dice=0.7017, per_class=['0.7017'], iou=0.5405, recall=0.9785, precision=0.5470, vol_sim=0.7171, mcc=0.7242, min_class_dice=0.7017, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8790, per_class_sd=['0.8790'], combined(w=0.50)=0.7904, balanced=0.7178
[2026-06-22 20:58:18] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 20:58:18] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7178 at epoch 304
[2026-06-22 20:58:18] INFO segtask_v1.trainer.trainer: Epoch 304/400 | LR=1.40e-04 | loss=1.3849 | val_dice=0.7017 | best=0.7178 (ep304) | 05:56:39 | L_main=0.5881 L_aux_1=0.5856(w=0.5) L_aux_2=0.6987(w=0.5)
[2026-06-22 20:58:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 304): 11666.1 MiB
[2026-06-22 20:59:28] INFO segtask_v1.trainer.validation:   Val: loss=0.7928, pooled_mean_dice=0.6785, per_class=['0.6785'], iou=0.5135, recall=0.9775, precision=0.5196, vol_sim=0.6942, mcc=0.7028, min_class_dice=0.6785, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8486, per_class_sd=['0.8486'], combined(w=0.50)=0.7636, balanced=0.6927
[2026-06-22 20:59:28] INFO segtask_v1.trainer.trainer: Epoch 305/400 | LR=1.37e-04 | loss=1.3726 | val_dice=0.6785 | best=0.7178 (ep304) | 05:57:49 | L_main=0.5797 L_aux_1=0.5793(w=0.5) L_aux_2=0.6947(w=0.5)
[2026-06-22 20:59:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 305): 11666.1 MiB
[2026-06-22 21:00:37] INFO segtask_v1.trainer.validation:   Val: loss=0.7846, pooled_mean_dice=0.6873, per_class=['0.6873'], iou=0.5236, recall=0.9764, precision=0.5303, vol_sim=0.7039, mcc=0.7112, min_class_dice=0.6873, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8633, per_class_sd=['0.8633'], combined(w=0.50)=0.7753, balanced=0.7027
[2026-06-22 21:00:37] INFO segtask_v1.trainer.trainer: Epoch 306/400 | LR=1.34e-04 | loss=1.4133 | val_dice=0.6873 | best=0.7178 (ep304) | 05:58:59 | L_main=0.6009 L_aux_1=0.5947(w=0.5) L_aux_2=0.7130(w=0.5)
[2026-06-22 21:00:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 306): 11666.1 MiB
[2026-06-22 21:01:47] INFO segtask_v1.trainer.validation:   Val: loss=0.7838, pooled_mean_dice=0.6841, per_class=['0.6841'], iou=0.5199, recall=0.9844, precision=0.5242, vol_sim=0.6950, mcc=0.7090, min_class_dice=0.6841, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8563, per_class_sd=['0.8563'], combined(w=0.50)=0.7702, balanced=0.6989
[2026-06-22 21:01:47] INFO segtask_v1.trainer.trainer: Epoch 307/400 | LR=1.32e-04 | loss=1.3675 | val_dice=0.6841 | best=0.7178 (ep304) | 06:00:09 | L_main=0.5805 L_aux_1=0.5789(w=0.5) L_aux_2=0.6860(w=0.5)
[2026-06-22 21:01:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 307): 11666.1 MiB
[2026-06-22 21:02:57] INFO segtask_v1.trainer.validation:   Val: loss=0.7750, pooled_mean_dice=0.6893, per_class=['0.6893'], iou=0.5259, recall=0.9757, precision=0.5329, vol_sim=0.7065, mcc=0.7124, min_class_dice=0.6893, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8642, per_class_sd=['0.8642'], combined(w=0.50)=0.7768, balanced=0.7046
[2026-06-22 21:02:57] INFO segtask_v1.trainer.trainer: Epoch 308/400 | LR=1.29e-04 | loss=1.4068 | val_dice=0.6893 | best=0.7178 (ep304) | 06:01:19 | L_main=0.5960 L_aux_1=0.5959(w=0.5) L_aux_2=0.7165(w=0.5)
[2026-06-22 21:02:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 308): 11666.1 MiB
[2026-06-22 21:04:07] INFO segtask_v1.trainer.validation:   Val: loss=0.7829, pooled_mean_dice=0.6735, per_class=['0.6735'], iou=0.5078, recall=0.9847, precision=0.5118, vol_sim=0.6840, mcc=0.7012, min_class_dice=0.6735, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8571, per_class_sd=['0.8571'], combined(w=0.50)=0.7653, balanced=0.6901
[2026-06-22 21:04:07] INFO segtask_v1.trainer.trainer: Epoch 309/400 | LR=1.26e-04 | loss=1.4049 | val_dice=0.6735 | best=0.7178 (ep304) | 06:02:28 | L_main=0.5974 L_aux_1=0.5949(w=0.5) L_aux_2=0.7107(w=0.5)
[2026-06-22 21:04:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 309): 11666.1 MiB
[2026-06-22 21:05:17] INFO segtask_v1.trainer.validation:   Val: loss=0.7830, pooled_mean_dice=0.6733, per_class=['0.6733'], iou=0.5075, recall=0.9800, precision=0.5128, vol_sim=0.6870, mcc=0.7023, min_class_dice=0.6733, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8599, per_class_sd=['0.8599'], combined(w=0.50)=0.7666, balanced=0.6904
[2026-06-22 21:05:17] INFO segtask_v1.trainer.trainer: Epoch 310/400 | LR=1.24e-04 | loss=1.3696 | val_dice=0.6733 | best=0.7178 (ep304) | 06:03:38 | L_main=0.5799 L_aux_1=0.5774(w=0.5) L_aux_2=0.6878(w=0.5)
[2026-06-22 21:05:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 310): 11666.1 MiB
[2026-06-22 21:06:28] INFO segtask_v1.trainer.validation:   Val: loss=0.7313, pooled_mean_dice=0.7073, per_class=['0.7073'], iou=0.5471, recall=0.9794, precision=0.5535, vol_sim=0.7222, mcc=0.7271, min_class_dice=0.7073, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8639, per_class_sd=['0.8639'], combined(w=0.50)=0.7856, balanced=0.7198
[2026-06-22 21:06:32] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 21:06:32] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7198 at epoch 311
[2026-06-22 21:06:32] INFO segtask_v1.trainer.trainer: Epoch 311/400 | LR=1.21e-04 | loss=1.3599 | val_dice=0.7073 | best=0.7198 (ep311) | 06:04:54 | L_main=0.5763 L_aux_1=0.5752(w=0.5) L_aux_2=0.6864(w=0.5)
[2026-06-22 21:06:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 311): 11666.1 MiB
[2026-06-22 21:07:42] INFO segtask_v1.trainer.validation:   Val: loss=0.7560, pooled_mean_dice=0.6976, per_class=['0.6976'], iou=0.5357, recall=0.9824, precision=0.5409, vol_sim=0.7102, mcc=0.7207, min_class_dice=0.6976, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8739, per_class_sd=['0.8739'], combined(w=0.50)=0.7858, balanced=0.7134
[2026-06-22 21:07:42] INFO segtask_v1.trainer.trainer: Epoch 312/400 | LR=1.18e-04 | loss=1.3776 | val_dice=0.6976 | best=0.7198 (ep311) | 06:06:03 | L_main=0.5850 L_aux_1=0.5837(w=0.5) L_aux_2=0.6930(w=0.5)
[2026-06-22 21:07:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 312): 11666.1 MiB
[2026-06-22 21:08:51] INFO segtask_v1.trainer.validation:   Val: loss=0.7556, pooled_mean_dice=0.6912, per_class=['0.6912'], iou=0.5281, recall=0.9806, precision=0.5336, vol_sim=0.7048, mcc=0.7143, min_class_dice=0.6912, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8659, per_class_sd=['0.8659'], combined(w=0.50)=0.7785, balanced=0.7065
[2026-06-22 21:08:51] INFO segtask_v1.trainer.trainer: Epoch 313/400 | LR=1.16e-04 | loss=1.3768 | val_dice=0.6912 | best=0.7198 (ep311) | 06:07:13 | L_main=0.5828 L_aux_1=0.5798(w=0.5) L_aux_2=0.6919(w=0.5)
[2026-06-22 21:08:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 313): 11666.1 MiB
[2026-06-22 21:10:01] INFO segtask_v1.trainer.validation:   Val: loss=0.7696, pooled_mean_dice=0.6667, per_class=['0.6667'], iou=0.5000, recall=0.9685, precision=0.5083, vol_sim=0.6884, mcc=0.6930, min_class_dice=0.6667, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.8549, per_class_sd=['0.8549'], combined(w=0.50)=0.7608, balanced=0.6837
[2026-06-22 21:10:01] INFO segtask_v1.trainer.trainer: Epoch 314/400 | LR=1.13e-04 | loss=1.3985 | val_dice=0.6667 | best=0.7198 (ep311) | 06:08:23 | L_main=0.5918 L_aux_1=0.5916(w=0.5) L_aux_2=0.7132(w=0.5)
[2026-06-22 21:10:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 314): 11666.1 MiB
[2026-06-22 21:11:11] INFO segtask_v1.trainer.validation:   Val: loss=0.7907, pooled_mean_dice=0.6896, per_class=['0.6896'], iou=0.5262, recall=0.9789, precision=0.5323, vol_sim=0.7044, mcc=0.7136, min_class_dice=0.6896, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8678, per_class_sd=['0.8678'], combined(w=0.50)=0.7787, balanced=0.7055
[2026-06-22 21:11:11] INFO segtask_v1.trainer.trainer: Epoch 315/400 | LR=1.11e-04 | loss=1.3730 | val_dice=0.6896 | best=0.7198 (ep311) | 06:09:33 | L_main=0.5816 L_aux_1=0.5763(w=0.5) L_aux_2=0.6961(w=0.5)
[2026-06-22 21:11:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 315): 11666.1 MiB
[2026-06-22 21:12:21] INFO segtask_v1.trainer.validation:   Val: loss=0.8225, pooled_mean_dice=0.6835, per_class=['0.6835'], iou=0.5192, recall=0.9703, precision=0.5276, vol_sim=0.7044, mcc=0.7088, min_class_dice=0.6835, coverage=[69]/88 samples, pooled_mean_surface_dice@2px=0.8669, per_class_sd=['0.8669'], combined(w=0.50)=0.7752, balanced=0.7002
[2026-06-22 21:12:21] INFO segtask_v1.trainer.trainer: Epoch 316/400 | LR=1.08e-04 | loss=1.3993 | val_dice=0.6835 | best=0.7198 (ep311) | 06:10:42 | L_main=0.5933 L_aux_1=0.5895(w=0.5) L_aux_2=0.7148(w=0.5)
[2026-06-22 21:12:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 316): 11666.1 MiB
[2026-06-22 21:13:31] INFO segtask_v1.trainer.validation:   Val: loss=0.7838, pooled_mean_dice=0.6824, per_class=['0.6824'], iou=0.5179, recall=0.9792, precision=0.5237, vol_sim=0.6969, mcc=0.7066, min_class_dice=0.6824, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8536, per_class_sd=['0.8536'], combined(w=0.50)=0.7680, balanced=0.6969
[2026-06-22 21:13:31] INFO segtask_v1.trainer.trainer: Epoch 317/400 | LR=1.06e-04 | loss=1.3849 | val_dice=0.6824 | best=0.7198 (ep311) | 06:11:52 | L_main=0.5848 L_aux_1=0.5841(w=0.5) L_aux_2=0.7117(w=0.5)
[2026-06-22 21:13:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 317): 11666.1 MiB
[2026-06-22 21:14:40] INFO segtask_v1.trainer.validation:   Val: loss=0.7684, pooled_mean_dice=0.6850, per_class=['0.6850'], iou=0.5209, recall=0.9803, precision=0.5264, vol_sim=0.6988, mcc=0.7108, min_class_dice=0.6850, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8709, per_class_sd=['0.8709'], combined(w=0.50)=0.7780, balanced=0.7022
[2026-06-22 21:14:40] INFO segtask_v1.trainer.trainer: Epoch 318/400 | LR=1.04e-04 | loss=1.3565 | val_dice=0.6850 | best=0.7198 (ep311) | 06:13:02 | L_main=0.5727 L_aux_1=0.5715(w=0.5) L_aux_2=0.6910(w=0.5)
[2026-06-22 21:14:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 318): 11666.1 MiB
[2026-06-22 21:15:50] INFO segtask_v1.trainer.validation:   Val: loss=0.7640, pooled_mean_dice=0.7050, per_class=['0.7050'], iou=0.5444, recall=0.9662, precision=0.5549, vol_sim=0.7297, mcc=0.7247, min_class_dice=0.7050, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8748, per_class_sd=['0.8748'], combined(w=0.50)=0.7899, balanced=0.7197
[2026-06-22 21:15:50] INFO segtask_v1.trainer.trainer: Epoch 319/400 | LR=1.01e-04 | loss=1.3650 | val_dice=0.7050 | best=0.7198 (ep311) | 06:14:12 | L_main=0.5770 L_aux_1=0.5798(w=0.5) L_aux_2=0.6843(w=0.5)
[2026-06-22 21:15:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 319): 11666.1 MiB
[2026-06-22 21:17:00] INFO segtask_v1.trainer.validation:   Val: loss=0.7731, pooled_mean_dice=0.6809, per_class=['0.6809'], iou=0.5162, recall=0.9793, precision=0.5219, vol_sim=0.6953, mcc=0.7056, min_class_dice=0.6809, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8544, per_class_sd=['0.8544'], combined(w=0.50)=0.7677, balanced=0.6958
[2026-06-22 21:17:00] INFO segtask_v1.trainer.trainer: Epoch 320/400 | LR=9.87e-05 | loss=1.3576 | val_dice=0.6809 | best=0.7198 (ep311) | 06:15:21 | L_main=0.5759 L_aux_1=0.5724(w=0.5) L_aux_2=0.6813(w=0.5)
[2026-06-22 21:17:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 320): 11666.1 MiB
[2026-06-22 21:18:10] INFO segtask_v1.trainer.validation:   Val: loss=0.7997, pooled_mean_dice=0.6742, per_class=['0.6742'], iou=0.5085, recall=0.9784, precision=0.5143, vol_sim=0.6891, mcc=0.7004, min_class_dice=0.6742, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8482, per_class_sd=['0.8482'], combined(w=0.50)=0.7612, balanced=0.6891
[2026-06-22 21:18:10] INFO segtask_v1.trainer.trainer: Epoch 321/400 | LR=9.64e-05 | loss=1.3830 | val_dice=0.6742 | best=0.7198 (ep311) | 06:16:32 | L_main=0.5853 L_aux_1=0.5842(w=0.5) L_aux_2=0.7019(w=0.5)
[2026-06-22 21:18:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 321): 11666.1 MiB
[2026-06-22 21:19:20] INFO segtask_v1.trainer.validation:   Val: loss=0.8261, pooled_mean_dice=0.6662, per_class=['0.6662'], iou=0.4995, recall=0.9701, precision=0.5073, vol_sim=0.6867, mcc=0.6931, min_class_dice=0.6662, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8555, per_class_sd=['0.8555'], combined(w=0.50)=0.7608, balanced=0.6834
[2026-06-22 21:19:20] INFO segtask_v1.trainer.trainer: Epoch 322/400 | LR=9.41e-05 | loss=1.3478 | val_dice=0.6662 | best=0.7198 (ep311) | 06:17:41 | L_main=0.5700 L_aux_1=0.5683(w=0.5) L_aux_2=0.6868(w=0.5)
[2026-06-22 21:19:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 322): 11666.1 MiB
[2026-06-22 21:20:30] INFO segtask_v1.trainer.validation:   Val: loss=0.8319, pooled_mean_dice=0.6665, per_class=['0.6665'], iou=0.4998, recall=0.9778, precision=0.5055, vol_sim=0.6816, mcc=0.6958, min_class_dice=0.6665, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8640, per_class_sd=['0.8640'], combined(w=0.50)=0.7653, balanced=0.6852
[2026-06-22 21:20:30] INFO segtask_v1.trainer.trainer: Epoch 323/400 | LR=9.18e-05 | loss=1.3608 | val_dice=0.6665 | best=0.7198 (ep311) | 06:18:52 | L_main=0.5742 L_aux_1=0.5748(w=0.5) L_aux_2=0.6919(w=0.5)
[2026-06-22 21:20:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 323): 11666.1 MiB
[2026-06-22 21:21:40] INFO segtask_v1.trainer.validation:   Val: loss=0.7530, pooled_mean_dice=0.6957, per_class=['0.6957'], iou=0.5334, recall=0.9823, precision=0.5385, vol_sim=0.7082, mcc=0.7174, min_class_dice=0.6957, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8541, per_class_sd=['0.8541'], combined(w=0.50)=0.7749, balanced=0.7083
[2026-06-22 21:21:40] INFO segtask_v1.trainer.trainer: Epoch 324/400 | LR=8.95e-05 | loss=1.3581 | val_dice=0.6957 | best=0.7198 (ep311) | 06:20:01 | L_main=0.5781 L_aux_1=0.5729(w=0.5) L_aux_2=0.6786(w=0.5)
[2026-06-22 21:21:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 324): 11666.1 MiB
[2026-06-22 21:22:50] INFO segtask_v1.trainer.validation:   Val: loss=0.7633, pooled_mean_dice=0.6798, per_class=['0.6798'], iou=0.5150, recall=0.9737, precision=0.5222, vol_sim=0.6982, mcc=0.7039, min_class_dice=0.6798, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8578, per_class_sd=['0.8578'], combined(w=0.50)=0.7688, balanced=0.6954
[2026-06-22 21:22:50] INFO segtask_v1.trainer.trainer: Epoch 325/400 | LR=8.73e-05 | loss=1.3652 | val_dice=0.6798 | best=0.7198 (ep311) | 06:21:11 | L_main=0.5784 L_aux_1=0.5796(w=0.5) L_aux_2=0.6885(w=0.5)
[2026-06-22 21:22:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 325): 11666.1 MiB
[2026-06-22 21:23:58] INFO segtask_v1.trainer.validation:   Val: loss=0.7813, pooled_mean_dice=0.6745, per_class=['0.6745'], iou=0.5088, recall=0.9804, precision=0.5141, vol_sim=0.6880, mcc=0.7016, min_class_dice=0.6745, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8529, per_class_sd=['0.8529'], combined(w=0.50)=0.7637, balanced=0.6902
[2026-06-22 21:23:58] INFO segtask_v1.trainer.trainer: Epoch 326/400 | LR=8.50e-05 | loss=1.3877 | val_dice=0.6745 | best=0.7198 (ep311) | 06:22:20 | L_main=0.5891 L_aux_1=0.5886(w=0.5) L_aux_2=0.6992(w=0.5)
[2026-06-22 21:23:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 326): 11666.1 MiB
[2026-06-22 21:25:08] INFO segtask_v1.trainer.validation:   Val: loss=0.7475, pooled_mean_dice=0.7016, per_class=['0.7016'], iou=0.5404, recall=0.9701, precision=0.5496, vol_sim=0.7233, mcc=0.7212, min_class_dice=0.7016, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8609, per_class_sd=['0.8609'], combined(w=0.50)=0.7813, balanced=0.7144
[2026-06-22 21:25:08] INFO segtask_v1.trainer.trainer: Epoch 327/400 | LR=8.29e-05 | loss=1.3858 | val_dice=0.7016 | best=0.7198 (ep311) | 06:23:30 | L_main=0.5889 L_aux_1=0.5829(w=0.5) L_aux_2=0.6978(w=0.5)
[2026-06-22 21:25:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 327): 11666.1 MiB
[2026-06-22 21:26:18] INFO segtask_v1.trainer.validation:   Val: loss=0.8010, pooled_mean_dice=0.6737, per_class=['0.6737'], iou=0.5080, recall=0.9635, precision=0.5180, vol_sim=0.6993, mcc=0.6974, min_class_dice=0.6737, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8478, per_class_sd=['0.8478'], combined(w=0.50)=0.7608, balanced=0.6884
[2026-06-22 21:26:18] INFO segtask_v1.trainer.trainer: Epoch 328/400 | LR=8.07e-05 | loss=1.3667 | val_dice=0.6737 | best=0.7198 (ep311) | 06:24:40 | L_main=0.5802 L_aux_1=0.5778(w=0.5) L_aux_2=0.6908(w=0.5)
[2026-06-22 21:26:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 328): 11666.1 MiB
[2026-06-22 21:27:28] INFO segtask_v1.trainer.validation:   Val: loss=0.7878, pooled_mean_dice=0.6669, per_class=['0.6669'], iou=0.5003, recall=0.9745, precision=0.5070, vol_sim=0.6844, mcc=0.6948, min_class_dice=0.6669, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8556, per_class_sd=['0.8556'], combined(w=0.50)=0.7613, balanced=0.6841
[2026-06-22 21:27:28] INFO segtask_v1.trainer.trainer: Epoch 329/400 | LR=7.85e-05 | loss=1.3627 | val_dice=0.6669 | best=0.7198 (ep311) | 06:25:50 | L_main=0.5751 L_aux_1=0.5767(w=0.5) L_aux_2=0.6931(w=0.5)
[2026-06-22 21:27:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 329): 11666.1 MiB
[2026-06-22 21:28:38] INFO segtask_v1.trainer.validation:   Val: loss=0.8306, pooled_mean_dice=0.6762, per_class=['0.6762'], iou=0.5108, recall=0.9662, precision=0.5201, vol_sim=0.6998, mcc=0.7013, min_class_dice=0.6762, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8600, per_class_sd=['0.8600'], combined(w=0.50)=0.7681, balanced=0.6927
[2026-06-22 21:28:38] INFO segtask_v1.trainer.trainer: Epoch 330/400 | LR=7.64e-05 | loss=1.3479 | val_dice=0.6762 | best=0.7198 (ep311) | 06:26:59 | L_main=0.5718 L_aux_1=0.5700(w=0.5) L_aux_2=0.6727(w=0.5)
[2026-06-22 21:28:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 330): 11666.1 MiB
[2026-06-22 21:29:49] INFO segtask_v1.trainer.validation:   Val: loss=0.7497, pooled_mean_dice=0.6878, per_class=['0.6878'], iou=0.5242, recall=0.9827, precision=0.5291, vol_sim=0.7000, mcc=0.7121, min_class_dice=0.6878, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8637, per_class_sd=['0.8637'], combined(w=0.50)=0.7758, balanced=0.7033
[2026-06-22 21:29:49] INFO segtask_v1.trainer.trainer: Epoch 331/400 | LR=7.43e-05 | loss=1.3654 | val_dice=0.6878 | best=0.7198 (ep311) | 06:28:10 | L_main=0.5776 L_aux_1=0.5793(w=0.5) L_aux_2=0.6884(w=0.5)
[2026-06-22 21:29:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 331): 11666.1 MiB
[2026-06-22 21:30:59] INFO segtask_v1.trainer.validation:   Val: loss=0.7800, pooled_mean_dice=0.6833, per_class=['0.6833'], iou=0.5189, recall=0.9749, precision=0.5260, vol_sim=0.7009, mcc=0.7083, min_class_dice=0.6833, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8617, per_class_sd=['0.8617'], combined(w=0.50)=0.7725, balanced=0.6991
[2026-06-22 21:30:59] INFO segtask_v1.trainer.trainer: Epoch 332/400 | LR=7.23e-05 | loss=1.3610 | val_dice=0.6833 | best=0.7198 (ep311) | 06:29:21 | L_main=0.5769 L_aux_1=0.5742(w=0.5) L_aux_2=0.6843(w=0.5)
[2026-06-22 21:30:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 332): 11666.1 MiB
[2026-06-22 21:32:08] INFO segtask_v1.trainer.validation:   Val: loss=0.7598, pooled_mean_dice=0.6918, per_class=['0.6918'], iou=0.5288, recall=0.9802, precision=0.5345, vol_sim=0.7057, mcc=0.7150, min_class_dice=0.6918, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8647, per_class_sd=['0.8647'], combined(w=0.50)=0.7782, balanced=0.7068
[2026-06-22 21:32:08] INFO segtask_v1.trainer.trainer: Epoch 333/400 | LR=7.03e-05 | loss=1.3689 | val_dice=0.6918 | best=0.7198 (ep311) | 06:30:30 | L_main=0.5805 L_aux_1=0.5803(w=0.5) L_aux_2=0.6953(w=0.5)
[2026-06-22 21:32:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 333): 11666.1 MiB
[2026-06-22 21:33:18] INFO segtask_v1.trainer.validation:   Val: loss=0.7894, pooled_mean_dice=0.6935, per_class=['0.6935'], iou=0.5308, recall=0.9812, precision=0.5363, vol_sim=0.7068, mcc=0.7173, min_class_dice=0.6935, coverage=[80]/88 samples, pooled_mean_surface_dice@2px=0.8710, per_class_sd=['0.8710'], combined(w=0.50)=0.7822, balanced=0.7094
[2026-06-22 21:33:18] INFO segtask_v1.trainer.trainer: Epoch 334/400 | LR=6.83e-05 | loss=1.3707 | val_dice=0.6935 | best=0.7198 (ep311) | 06:31:40 | L_main=0.5784 L_aux_1=0.5797(w=0.5) L_aux_2=0.6958(w=0.5)
[2026-06-22 21:33:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 334): 11666.1 MiB
[2026-06-22 21:34:29] INFO segtask_v1.trainer.validation:   Val: loss=0.7925, pooled_mean_dice=0.7003, per_class=['0.7003'], iou=0.5388, recall=0.9809, precision=0.5445, vol_sim=0.7139, mcc=0.7228, min_class_dice=0.7003, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8608, per_class_sd=['0.8608'], combined(w=0.50)=0.7805, balanced=0.7134
[2026-06-22 21:34:29] INFO segtask_v1.trainer.trainer: Epoch 335/400 | LR=6.63e-05 | loss=1.3657 | val_dice=0.7003 | best=0.7198 (ep311) | 06:32:51 | L_main=0.5786 L_aux_1=0.5775(w=0.5) L_aux_2=0.6885(w=0.5)
[2026-06-22 21:34:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 335): 11666.1 MiB
[2026-06-22 21:35:38] INFO segtask_v1.trainer.validation:   Val: loss=0.8098, pooled_mean_dice=0.6840, per_class=['0.6840'], iou=0.5198, recall=0.9608, precision=0.5310, vol_sim=0.7119, mcc=0.7071, min_class_dice=0.6840, coverage=[68]/88 samples, pooled_mean_surface_dice@2px=0.8576, per_class_sd=['0.8576'], combined(w=0.50)=0.7708, balanced=0.6989
[2026-06-22 21:35:38] INFO segtask_v1.trainer.trainer: Epoch 336/400 | LR=6.43e-05 | loss=1.3821 | val_dice=0.6840 | best=0.7198 (ep311) | 06:34:00 | L_main=0.5852 L_aux_1=0.5851(w=0.5) L_aux_2=0.6941(w=0.5)
[2026-06-22 21:35:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 336): 11666.1 MiB
[2026-06-22 21:36:48] INFO segtask_v1.trainer.validation:   Val: loss=0.7825, pooled_mean_dice=0.6743, per_class=['0.6743'], iou=0.5086, recall=0.9805, precision=0.5139, vol_sim=0.6877, mcc=0.7014, min_class_dice=0.6743, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8612, per_class_sd=['0.8612'], combined(w=0.50)=0.7678, balanced=0.6914
[2026-06-22 21:36:48] INFO segtask_v1.trainer.trainer: Epoch 337/400 | LR=6.24e-05 | loss=1.3782 | val_dice=0.6743 | best=0.7198 (ep311) | 06:35:10 | L_main=0.5817 L_aux_1=0.5834(w=0.5) L_aux_2=0.6980(w=0.5)
[2026-06-22 21:36:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 337): 11666.1 MiB
[2026-06-22 21:37:58] INFO segtask_v1.trainer.validation:   Val: loss=0.7389, pooled_mean_dice=0.6920, per_class=['0.6920'], iou=0.5291, recall=0.9766, precision=0.5359, vol_sim=0.7086, mcc=0.7143, min_class_dice=0.6920, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8605, per_class_sd=['0.8605'], combined(w=0.50)=0.7762, balanced=0.7062
[2026-06-22 21:37:58] INFO segtask_v1.trainer.trainer: Epoch 338/400 | LR=6.05e-05 | loss=1.3574 | val_dice=0.6920 | best=0.7198 (ep311) | 06:36:20 | L_main=0.5750 L_aux_1=0.5725(w=0.5) L_aux_2=0.6883(w=0.5)
[2026-06-22 21:37:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 338): 11666.1 MiB
[2026-06-22 21:39:07] INFO segtask_v1.trainer.validation:   Val: loss=0.7943, pooled_mean_dice=0.7085, per_class=['0.7085'], iou=0.5486, recall=0.9710, precision=0.5578, vol_sim=0.7297, mcc=0.7276, min_class_dice=0.7085, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8724, per_class_sd=['0.8724'], combined(w=0.50)=0.7905, balanced=0.7223
[2026-06-22 21:39:11] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 21:39:11] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7223 at epoch 339
[2026-06-22 21:39:11] INFO segtask_v1.trainer.trainer: Epoch 339/400 | LR=5.86e-05 | loss=1.3719 | val_dice=0.7085 | best=0.7223 (ep339) | 06:37:33 | L_main=0.5807 L_aux_1=0.5800(w=0.5) L_aux_2=0.6944(w=0.5)
[2026-06-22 21:39:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 339): 11666.1 MiB
[2026-06-22 21:40:20] INFO segtask_v1.trainer.validation:   Val: loss=0.7567, pooled_mean_dice=0.7045, per_class=['0.7045'], iou=0.5438, recall=0.9818, precision=0.5493, vol_sim=0.7176, mcc=0.7252, min_class_dice=0.7045, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8738, per_class_sd=['0.8738'], combined(w=0.50)=0.7892, balanced=0.7192
[2026-06-22 21:40:20] INFO segtask_v1.trainer.trainer: Epoch 340/400 | LR=5.68e-05 | loss=1.3809 | val_dice=0.7045 | best=0.7223 (ep339) | 06:38:42 | L_main=0.5857 L_aux_1=0.5817(w=0.5) L_aux_2=0.6973(w=0.5)
[2026-06-22 21:40:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 340): 11666.1 MiB
[2026-06-22 21:41:31] INFO segtask_v1.trainer.validation:   Val: loss=0.7756, pooled_mean_dice=0.6874, per_class=['0.6874'], iou=0.5237, recall=0.9788, precision=0.5298, vol_sim=0.7023, mcc=0.7114, min_class_dice=0.6874, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8558, per_class_sd=['0.8558'], combined(w=0.50)=0.7716, balanced=0.7016
[2026-06-22 21:41:31] INFO segtask_v1.trainer.trainer: Epoch 341/400 | LR=5.50e-05 | loss=1.3470 | val_dice=0.6874 | best=0.7223 (ep339) | 06:39:53 | L_main=0.5716 L_aux_1=0.5688(w=0.5) L_aux_2=0.6767(w=0.5)
[2026-06-22 21:41:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 341): 11666.1 MiB
[2026-06-22 21:42:41] INFO segtask_v1.trainer.validation:   Val: loss=0.7635, pooled_mean_dice=0.6995, per_class=['0.6995'], iou=0.5378, recall=0.9775, precision=0.5446, vol_sim=0.7156, mcc=0.7209, min_class_dice=0.6995, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8614, per_class_sd=['0.8614'], combined(w=0.50)=0.7804, balanced=0.7128
[2026-06-22 21:42:41] INFO segtask_v1.trainer.trainer: Epoch 342/400 | LR=5.32e-05 | loss=1.3701 | val_dice=0.6995 | best=0.7223 (ep339) | 06:41:03 | L_main=0.5781 L_aux_1=0.5826(w=0.5) L_aux_2=0.6981(w=0.5)
[2026-06-22 21:42:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 342): 11666.1 MiB
[2026-06-22 21:43:51] INFO segtask_v1.trainer.validation:   Val: loss=0.8140, pooled_mean_dice=0.6581, per_class=['0.6581'], iou=0.4905, recall=0.9571, precision=0.5015, vol_sim=0.6877, mcc=0.6856, min_class_dice=0.6581, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8659, per_class_sd=['0.8659'], combined(w=0.50)=0.7620, balanced=0.6780
[2026-06-22 21:43:51] INFO segtask_v1.trainer.trainer: Epoch 343/400 | LR=5.15e-05 | loss=1.3767 | val_dice=0.6581 | best=0.7223 (ep339) | 06:42:12 | L_main=0.5836 L_aux_1=0.5841(w=0.5) L_aux_2=0.6885(w=0.5)
[2026-06-22 21:43:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 343): 11666.1 MiB
[2026-06-22 21:45:01] INFO segtask_v1.trainer.validation:   Val: loss=0.7674, pooled_mean_dice=0.6872, per_class=['0.6872'], iou=0.5234, recall=0.9729, precision=0.5312, vol_sim=0.7063, mcc=0.7103, min_class_dice=0.6872, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8651, per_class_sd=['0.8651'], combined(w=0.50)=0.7761, balanced=0.7029
[2026-06-22 21:45:01] INFO segtask_v1.trainer.trainer: Epoch 344/400 | LR=4.97e-05 | loss=1.3565 | val_dice=0.6872 | best=0.7223 (ep339) | 06:43:22 | L_main=0.5766 L_aux_1=0.5742(w=0.5) L_aux_2=0.6794(w=0.5)
[2026-06-22 21:45:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 344): 11666.1 MiB
[2026-06-22 21:46:10] INFO segtask_v1.trainer.validation:   Val: loss=0.8067, pooled_mean_dice=0.6667, per_class=['0.6667'], iou=0.5000, recall=0.9717, precision=0.5074, vol_sim=0.6861, mcc=0.6942, min_class_dice=0.6667, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8667, per_class_sd=['0.8667'], combined(w=0.50)=0.7667, balanced=0.6857
[2026-06-22 21:46:10] INFO segtask_v1.trainer.trainer: Epoch 345/400 | LR=4.80e-05 | loss=1.3744 | val_dice=0.6667 | best=0.7223 (ep339) | 06:44:32 | L_main=0.5810 L_aux_1=0.5857(w=0.5) L_aux_2=0.6945(w=0.5)
[2026-06-22 21:46:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 345): 11666.1 MiB
[2026-06-22 21:47:19] INFO segtask_v1.trainer.validation:   Val: loss=0.7788, pooled_mean_dice=0.6786, per_class=['0.6786'], iou=0.5136, recall=0.9719, precision=0.5213, vol_sim=0.6982, mcc=0.7040, min_class_dice=0.6786, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8626, per_class_sd=['0.8626'], combined(w=0.50)=0.7706, balanced=0.6952
[2026-06-22 21:47:19] INFO segtask_v1.trainer.trainer: Epoch 346/400 | LR=4.64e-05 | loss=1.3725 | val_dice=0.6786 | best=0.7223 (ep339) | 06:45:41 | L_main=0.5806 L_aux_1=0.5823(w=0.5) L_aux_2=0.7001(w=0.5)
[2026-06-22 21:47:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 346): 11666.1 MiB
[2026-06-22 21:48:30] INFO segtask_v1.trainer.validation:   Val: loss=0.7713, pooled_mean_dice=0.6914, per_class=['0.6914'], iou=0.5283, recall=0.9781, precision=0.5347, vol_sim=0.7069, mcc=0.7134, min_class_dice=0.6914, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8561, per_class_sd=['0.8561'], combined(w=0.50)=0.7738, balanced=0.7049
[2026-06-22 21:48:30] INFO segtask_v1.trainer.trainer: Epoch 347/400 | LR=4.47e-05 | loss=1.3675 | val_dice=0.6914 | best=0.7223 (ep339) | 06:46:51 | L_main=0.5785 L_aux_1=0.5783(w=0.5) L_aux_2=0.6944(w=0.5)
[2026-06-22 21:48:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 347): 11666.1 MiB
[2026-06-22 21:49:40] INFO segtask_v1.trainer.validation:   Val: loss=0.8048, pooled_mean_dice=0.6648, per_class=['0.6648'], iou=0.4979, recall=0.9784, precision=0.5035, vol_sim=0.6795, mcc=0.6945, min_class_dice=0.6648, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8597, per_class_sd=['0.8597'], combined(w=0.50)=0.7623, balanced=0.6831
[2026-06-22 21:49:40] INFO segtask_v1.trainer.trainer: Epoch 348/400 | LR=4.31e-05 | loss=1.3580 | val_dice=0.6648 | best=0.7223 (ep339) | 06:48:02 | L_main=0.5721 L_aux_1=0.5764(w=0.5) L_aux_2=0.6919(w=0.5)
[2026-06-22 21:49:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 348): 11666.1 MiB
[2026-06-22 21:50:50] INFO segtask_v1.trainer.validation:   Val: loss=0.7912, pooled_mean_dice=0.7030, per_class=['0.7030'], iou=0.5420, recall=0.9777, precision=0.5488, vol_sim=0.7190, mcc=0.7244, min_class_dice=0.7030, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8663, per_class_sd=['0.8663'], combined(w=0.50)=0.7846, balanced=0.7167
[2026-06-22 21:50:50] INFO segtask_v1.trainer.trainer: Epoch 349/400 | LR=4.15e-05 | loss=1.3990 | val_dice=0.7030 | best=0.7223 (ep339) | 06:49:12 | L_main=0.5886 L_aux_1=0.5952(w=0.5) L_aux_2=0.7196(w=0.5)
[2026-06-22 21:50:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 349): 11666.1 MiB
[2026-06-22 21:52:00] INFO segtask_v1.trainer.validation:   Val: loss=0.7621, pooled_mean_dice=0.6999, per_class=['0.6999'], iou=0.5383, recall=0.9807, precision=0.5440, vol_sim=0.7136, mcc=0.7209, min_class_dice=0.6999, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8584, per_class_sd=['0.8584'], combined(w=0.50)=0.7791, balanced=0.7125
[2026-06-22 21:52:00] INFO segtask_v1.trainer.trainer: Epoch 350/400 | LR=4.00e-05 | loss=1.3874 | val_dice=0.6999 | best=0.7223 (ep339) | 06:50:21 | L_main=0.5873 L_aux_1=0.5890(w=0.5) L_aux_2=0.6962(w=0.5)
[2026-06-22 21:52:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 350): 11666.1 MiB
[2026-06-22 21:53:11] INFO segtask_v1.trainer.validation:   Val: loss=0.7650, pooled_mean_dice=0.6811, per_class=['0.6811'], iou=0.5164, recall=0.9766, precision=0.5229, vol_sim=0.6974, mcc=0.7047, min_class_dice=0.6811, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8594, per_class_sd=['0.8594'], combined(w=0.50)=0.7703, balanced=0.6967
[2026-06-22 21:53:11] INFO segtask_v1.trainer.trainer: Epoch 351/400 | LR=3.85e-05 | loss=1.3740 | val_dice=0.6811 | best=0.7223 (ep339) | 06:51:32 | L_main=0.5811 L_aux_1=0.5817(w=0.5) L_aux_2=0.6968(w=0.5)
[2026-06-22 21:53:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 351): 11666.1 MiB
[2026-06-22 21:54:23] INFO segtask_v1.trainer.validation:   Val: loss=0.7333, pooled_mean_dice=0.7034, per_class=['0.7034'], iou=0.5425, recall=0.9780, precision=0.5492, vol_sim=0.7192, mcc=0.7237, min_class_dice=0.7034, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8619, per_class_sd=['0.8619'], combined(w=0.50)=0.7827, balanced=0.7161
[2026-06-22 21:54:23] INFO segtask_v1.trainer.trainer: Epoch 352/400 | LR=3.70e-05 | loss=1.3429 | val_dice=0.7034 | best=0.7223 (ep339) | 06:52:44 | L_main=0.5669 L_aux_1=0.5655(w=0.5) L_aux_2=0.6829(w=0.5)
[2026-06-22 21:54:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 352): 11666.1 MiB
[2026-06-22 21:55:32] INFO segtask_v1.trainer.validation:   Val: loss=0.7695, pooled_mean_dice=0.6795, per_class=['0.6795'], iou=0.5145, recall=0.9783, precision=0.5205, vol_sim=0.6946, mcc=0.7052, min_class_dice=0.6795, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8644, per_class_sd=['0.8644'], combined(w=0.50)=0.7719, balanced=0.6963
[2026-06-22 21:55:32] INFO segtask_v1.trainer.trainer: Epoch 353/400 | LR=3.55e-05 | loss=1.3497 | val_dice=0.6795 | best=0.7223 (ep339) | 06:53:54 | L_main=0.5724 L_aux_1=0.5684(w=0.5) L_aux_2=0.6811(w=0.5)
[2026-06-22 21:55:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 353): 11666.1 MiB
[2026-06-22 21:56:42] INFO segtask_v1.trainer.validation:   Val: loss=0.7327, pooled_mean_dice=0.6986, per_class=['0.6986'], iou=0.5368, recall=0.9784, precision=0.5432, vol_sim=0.7140, mcc=0.7192, min_class_dice=0.6986, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8677, per_class_sd=['0.8677'], combined(w=0.50)=0.7832, balanced=0.7130
[2026-06-22 21:56:42] INFO segtask_v1.trainer.trainer: Epoch 354/400 | LR=3.41e-05 | loss=1.3660 | val_dice=0.6986 | best=0.7223 (ep339) | 06:55:04 | L_main=0.5781 L_aux_1=0.5751(w=0.5) L_aux_2=0.6958(w=0.5)
[2026-06-22 21:56:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 354): 11666.1 MiB
[2026-06-22 21:57:52] INFO segtask_v1.trainer.validation:   Val: loss=0.7980, pooled_mean_dice=0.6685, per_class=['0.6685'], iou=0.5020, recall=0.9809, precision=0.5070, vol_sim=0.6815, mcc=0.6984, min_class_dice=0.6685, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8663, per_class_sd=['0.8663'], combined(w=0.50)=0.7674, balanced=0.6873
[2026-06-22 21:57:52] INFO segtask_v1.trainer.trainer: Epoch 355/400 | LR=3.27e-05 | loss=1.3408 | val_dice=0.6685 | best=0.7223 (ep339) | 06:56:13 | L_main=0.5665 L_aux_1=0.5665(w=0.5) L_aux_2=0.6770(w=0.5)
[2026-06-22 21:57:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 355): 11666.1 MiB
[2026-06-22 21:59:02] INFO segtask_v1.trainer.validation:   Val: loss=0.8565, pooled_mean_dice=0.6597, per_class=['0.6597'], iou=0.4922, recall=0.9681, precision=0.5003, vol_sim=0.6814, mcc=0.6898, min_class_dice=0.6597, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8650, per_class_sd=['0.8650'], combined(w=0.50)=0.7623, balanced=0.6794
[2026-06-22 21:59:02] INFO segtask_v1.trainer.trainer: Epoch 356/400 | LR=3.13e-05 | loss=1.3696 | val_dice=0.6597 | best=0.7223 (ep339) | 06:57:23 | L_main=0.5779 L_aux_1=0.5780(w=0.5) L_aux_2=0.6933(w=0.5)
[2026-06-22 21:59:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 356): 11666.1 MiB
[2026-06-22 22:00:12] INFO segtask_v1.trainer.validation:   Val: loss=0.7825, pooled_mean_dice=0.6855, per_class=['0.6855'], iou=0.5215, recall=0.9645, precision=0.5317, vol_sim=0.7107, mcc=0.7073, min_class_dice=0.6855, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8588, per_class_sd=['0.8588'], combined(w=0.50)=0.7722, balanced=0.7003
[2026-06-22 22:00:12] INFO segtask_v1.trainer.trainer: Epoch 357/400 | LR=2.99e-05 | loss=1.3697 | val_dice=0.6855 | best=0.7223 (ep339) | 06:58:33 | L_main=0.5793 L_aux_1=0.5782(w=0.5) L_aux_2=0.6952(w=0.5)
[2026-06-22 22:00:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 357): 11666.1 MiB
[2026-06-22 22:01:21] INFO segtask_v1.trainer.validation:   Val: loss=0.7560, pooled_mean_dice=0.6814, per_class=['0.6814'], iou=0.5168, recall=0.9790, precision=0.5226, vol_sim=0.6961, mcc=0.7059, min_class_dice=0.6814, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8533, per_class_sd=['0.8533'], combined(w=0.50)=0.7674, balanced=0.6960
[2026-06-22 22:01:21] INFO segtask_v1.trainer.trainer: Epoch 358/400 | LR=2.86e-05 | loss=1.3594 | val_dice=0.6814 | best=0.7223 (ep339) | 06:59:43 | L_main=0.5750 L_aux_1=0.5780(w=0.5) L_aux_2=0.6902(w=0.5)
[2026-06-22 22:01:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 358): 11666.1 MiB
[2026-06-22 22:02:31] INFO segtask_v1.trainer.validation:   Val: loss=0.7699, pooled_mean_dice=0.6851, per_class=['0.6851'], iou=0.5211, recall=0.9691, precision=0.5299, vol_sim=0.7070, mcc=0.7093, min_class_dice=0.6851, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8690, per_class_sd=['0.8690'], combined(w=0.50)=0.7771, balanced=0.7018
[2026-06-22 22:02:31] INFO segtask_v1.trainer.trainer: Epoch 359/400 | LR=2.73e-05 | loss=1.3409 | val_dice=0.6851 | best=0.7223 (ep339) | 07:00:53 | L_main=0.5675 L_aux_1=0.5648(w=0.5) L_aux_2=0.6810(w=0.5)
[2026-06-22 22:02:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 359): 11666.1 MiB
[2026-06-22 22:03:41] INFO segtask_v1.trainer.validation:   Val: loss=0.7987, pooled_mean_dice=0.6789, per_class=['0.6789'], iou=0.5139, recall=0.9746, precision=0.5209, vol_sim=0.6966, mcc=0.7035, min_class_dice=0.6789, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8527, per_class_sd=['0.8527'], combined(w=0.50)=0.7658, balanced=0.6938
[2026-06-22 22:03:41] INFO segtask_v1.trainer.trainer: Epoch 360/400 | LR=2.61e-05 | loss=1.3581 | val_dice=0.6789 | best=0.7223 (ep339) | 07:02:03 | L_main=0.5763 L_aux_1=0.5732(w=0.5) L_aux_2=0.6817(w=0.5)
[2026-06-22 22:03:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 360): 11666.1 MiB
[2026-06-22 22:04:52] INFO segtask_v1.trainer.validation:   Val: loss=0.7690, pooled_mean_dice=0.7136, per_class=['0.7136'], iou=0.5548, recall=0.9810, precision=0.5608, vol_sim=0.7275, mcc=0.7333, min_class_dice=0.7136, coverage=[81]/88 samples, pooled_mean_surface_dice@2px=0.8685, per_class_sd=['0.8685'], combined(w=0.50)=0.7910, balanced=0.7261
[2026-06-22 22:04:56] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 22:04:56] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7261 at epoch 361
[2026-06-22 22:04:56] INFO segtask_v1.trainer.trainer: Epoch 361/400 | LR=2.48e-05 | loss=1.3555 | val_dice=0.7136 | best=0.7261 (ep361) | 07:03:18 | L_main=0.5736 L_aux_1=0.5717(w=0.5) L_aux_2=0.6863(w=0.5)
[2026-06-22 22:04:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 361): 11666.1 MiB
[2026-06-22 22:06:05] INFO segtask_v1.trainer.validation:   Val: loss=0.7507, pooled_mean_dice=0.6792, per_class=['0.6792'], iou=0.5142, recall=0.9749, precision=0.5211, vol_sim=0.6967, mcc=0.7048, min_class_dice=0.6792, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8647, per_class_sd=['0.8647'], combined(w=0.50)=0.7719, balanced=0.6961
[2026-06-22 22:06:05] INFO segtask_v1.trainer.trainer: Epoch 362/400 | LR=2.36e-05 | loss=1.3501 | val_dice=0.6792 | best=0.7261 (ep361) | 07:04:27 | L_main=0.5689 L_aux_1=0.5706(w=0.5) L_aux_2=0.6943(w=0.5)
[2026-06-22 22:06:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 362): 11666.1 MiB
[2026-06-22 22:07:14] INFO segtask_v1.trainer.validation:   Val: loss=0.7678, pooled_mean_dice=0.6772, per_class=['0.6772'], iou=0.5120, recall=0.9797, precision=0.5174, vol_sim=0.6912, mcc=0.7044, min_class_dice=0.6772, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8773, per_class_sd=['0.8773'], combined(w=0.50)=0.7772, balanced=0.6965
[2026-06-22 22:07:14] INFO segtask_v1.trainer.trainer: Epoch 363/400 | LR=2.25e-05 | loss=1.3819 | val_dice=0.6772 | best=0.7261 (ep361) | 07:05:36 | L_main=0.5851 L_aux_1=0.5809(w=0.5) L_aux_2=0.7029(w=0.5)
[2026-06-22 22:07:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 363): 11666.1 MiB
[2026-06-22 22:08:25] INFO segtask_v1.trainer.validation:   Val: loss=0.7958, pooled_mean_dice=0.6764, per_class=['0.6764'], iou=0.5110, recall=0.9718, precision=0.5187, vol_sim=0.6960, mcc=0.7020, min_class_dice=0.6764, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8622, per_class_sd=['0.8622'], combined(w=0.50)=0.7693, balanced=0.6932
[2026-06-22 22:08:25] INFO segtask_v1.trainer.trainer: Epoch 364/400 | LR=2.13e-05 | loss=1.3603 | val_dice=0.6764 | best=0.7261 (ep361) | 07:06:46 | L_main=0.5732 L_aux_1=0.5788(w=0.5) L_aux_2=0.6971(w=0.5)
[2026-06-22 22:08:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 364): 11666.1 MiB
[2026-06-22 22:09:35] INFO segtask_v1.trainer.validation:   Val: loss=0.8008, pooled_mean_dice=0.6716, per_class=['0.6716'], iou=0.5056, recall=0.9764, precision=0.5118, vol_sim=0.6878, mcc=0.6993, min_class_dice=0.6716, coverage=[82]/88 samples, pooled_mean_surface_dice@2px=0.8563, per_class_sd=['0.8563'], combined(w=0.50)=0.7639, balanced=0.6883
[2026-06-22 22:09:35] INFO segtask_v1.trainer.trainer: Epoch 365/400 | LR=2.02e-05 | loss=1.3468 | val_dice=0.6716 | best=0.7261 (ep361) | 07:07:57 | L_main=0.5699 L_aux_1=0.5674(w=0.5) L_aux_2=0.6772(w=0.5)
[2026-06-22 22:09:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 365): 11666.1 MiB
[2026-06-22 22:10:44] INFO segtask_v1.trainer.validation:   Val: loss=0.7665, pooled_mean_dice=0.6858, per_class=['0.6858'], iou=0.5218, recall=0.9765, precision=0.5284, vol_sim=0.7023, mcc=0.7101, min_class_dice=0.6858, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8637, per_class_sd=['0.8637'], combined(w=0.50)=0.7747, balanced=0.7015
[2026-06-22 22:10:44] INFO segtask_v1.trainer.trainer: Epoch 366/400 | LR=1.92e-05 | loss=1.3456 | val_dice=0.6858 | best=0.7261 (ep361) | 07:09:05 | L_main=0.5701 L_aux_1=0.5685(w=0.5) L_aux_2=0.6785(w=0.5)
[2026-06-22 22:10:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 366): 11666.1 MiB
[2026-06-22 22:11:54] INFO segtask_v1.trainer.validation:   Val: loss=0.7811, pooled_mean_dice=0.6835, per_class=['0.6835'], iou=0.5192, recall=0.9804, precision=0.5246, vol_sim=0.6972, mcc=0.7083, min_class_dice=0.6835, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8579, per_class_sd=['0.8579'], combined(w=0.50)=0.7707, balanced=0.6986
[2026-06-22 22:11:54] INFO segtask_v1.trainer.trainer: Epoch 367/400 | LR=1.81e-05 | loss=1.3478 | val_dice=0.6835 | best=0.7261 (ep361) | 07:10:16 | L_main=0.5701 L_aux_1=0.5699(w=0.5) L_aux_2=0.6862(w=0.5)
[2026-06-22 22:11:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 367): 11666.1 MiB
[2026-06-22 22:13:04] INFO segtask_v1.trainer.validation:   Val: loss=0.7587, pooled_mean_dice=0.6846, per_class=['0.6846'], iou=0.5205, recall=0.9795, precision=0.5262, vol_sim=0.6990, mcc=0.7078, min_class_dice=0.6846, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8473, per_class_sd=['0.8473'], combined(w=0.50)=0.7660, balanced=0.6977
[2026-06-22 22:13:04] INFO segtask_v1.trainer.trainer: Epoch 368/400 | LR=1.71e-05 | loss=1.3660 | val_dice=0.6846 | best=0.7261 (ep361) | 07:11:25 | L_main=0.5791 L_aux_1=0.5754(w=0.5) L_aux_2=0.6887(w=0.5)
[2026-06-22 22:13:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 368): 11666.1 MiB
[2026-06-22 22:14:13] INFO segtask_v1.trainer.validation:   Val: loss=0.8046, pooled_mean_dice=0.6744, per_class=['0.6744'], iou=0.5087, recall=0.9798, precision=0.5141, vol_sim=0.6882, mcc=0.7025, min_class_dice=0.6744, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8754, per_class_sd=['0.8754'], combined(w=0.50)=0.7749, balanced=0.6938
[2026-06-22 22:14:13] INFO segtask_v1.trainer.trainer: Epoch 369/400 | LR=1.61e-05 | loss=1.3569 | val_dice=0.6744 | best=0.7261 (ep361) | 07:12:35 | L_main=0.5743 L_aux_1=0.5704(w=0.5) L_aux_2=0.6877(w=0.5)
[2026-06-22 22:14:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 369): 11666.1 MiB
[2026-06-22 22:15:22] INFO segtask_v1.trainer.validation:   Val: loss=0.7817, pooled_mean_dice=0.6885, per_class=['0.6885'], iou=0.5250, recall=0.9745, precision=0.5323, vol_sim=0.7065, mcc=0.7121, min_class_dice=0.6885, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8718, per_class_sd=['0.8718'], combined(w=0.50)=0.7802, balanced=0.7052
[2026-06-22 22:15:22] INFO segtask_v1.trainer.trainer: Epoch 370/400 | LR=1.52e-05 | loss=1.3399 | val_dice=0.6885 | best=0.7261 (ep361) | 07:13:44 | L_main=0.5684 L_aux_1=0.5653(w=0.5) L_aux_2=0.6738(w=0.5)
[2026-06-22 22:15:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 370): 11666.1 MiB
[2026-06-22 22:16:34] INFO segtask_v1.trainer.validation:   Val: loss=0.8004, pooled_mean_dice=0.6531, per_class=['0.6531'], iou=0.4848, recall=0.9716, precision=0.4918, vol_sim=0.6721, mcc=0.6845, min_class_dice=0.6531, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8766, per_class_sd=['0.8766'], combined(w=0.50)=0.7648, balanced=0.6755
[2026-06-22 22:16:34] INFO segtask_v1.trainer.trainer: Epoch 371/400 | LR=1.42e-05 | loss=1.3560 | val_dice=0.6531 | best=0.7261 (ep361) | 07:14:56 | L_main=0.5733 L_aux_1=0.5710(w=0.5) L_aux_2=0.6898(w=0.5)
[2026-06-22 22:16:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 371): 11666.1 MiB
[2026-06-22 22:17:44] INFO segtask_v1.trainer.validation:   Val: loss=0.7518, pooled_mean_dice=0.6960, per_class=['0.6960'], iou=0.5338, recall=0.9708, precision=0.5425, vol_sim=0.7169, mcc=0.7154, min_class_dice=0.6960, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8623, per_class_sd=['0.8623'], combined(w=0.50)=0.7791, balanced=0.7098
[2026-06-22 22:17:44] INFO segtask_v1.trainer.trainer: Epoch 372/400 | LR=1.33e-05 | loss=1.3299 | val_dice=0.6960 | best=0.7261 (ep361) | 07:16:05 | L_main=0.5633 L_aux_1=0.5586(w=0.5) L_aux_2=0.6748(w=0.5)
[2026-06-22 22:17:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 372): 11666.1 MiB
[2026-06-22 22:18:53] INFO segtask_v1.trainer.validation:   Val: loss=0.7640, pooled_mean_dice=0.6860, per_class=['0.6860'], iou=0.5221, recall=0.9774, precision=0.5285, vol_sim=0.7019, mcc=0.7092, min_class_dice=0.6860, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8570, per_class_sd=['0.8570'], combined(w=0.50)=0.7715, balanced=0.7005
[2026-06-22 22:18:53] INFO segtask_v1.trainer.trainer: Epoch 373/400 | LR=1.25e-05 | loss=1.3663 | val_dice=0.6860 | best=0.7261 (ep361) | 07:17:15 | L_main=0.5766 L_aux_1=0.5763(w=0.5) L_aux_2=0.6978(w=0.5)
[2026-06-22 22:18:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 373): 11666.1 MiB
[2026-06-22 22:20:03] INFO segtask_v1.trainer.validation:   Val: loss=0.7491, pooled_mean_dice=0.6982, per_class=['0.6982'], iou=0.5364, recall=0.9798, precision=0.5423, vol_sim=0.7126, mcc=0.7213, min_class_dice=0.6982, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8752, per_class_sd=['0.8752'], combined(w=0.50)=0.7867, balanced=0.7141
[2026-06-22 22:20:03] INFO segtask_v1.trainer.trainer: Epoch 374/400 | LR=1.16e-05 | loss=1.3717 | val_dice=0.6982 | best=0.7261 (ep361) | 07:18:24 | L_main=0.5787 L_aux_1=0.5817(w=0.5) L_aux_2=0.7050(w=0.5)
[2026-06-22 22:20:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 374): 11666.1 MiB
[2026-06-22 22:21:11] INFO segtask_v1.trainer.validation:   Val: loss=0.8100, pooled_mean_dice=0.6818, per_class=['0.6818'], iou=0.5173, recall=0.9722, precision=0.5250, vol_sim=0.7013, mcc=0.7065, min_class_dice=0.6818, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8559, per_class_sd=['0.8559'], combined(w=0.50)=0.7689, balanced=0.6968
[2026-06-22 22:21:11] INFO segtask_v1.trainer.trainer: Epoch 375/400 | LR=1.08e-05 | loss=1.3347 | val_dice=0.6818 | best=0.7261 (ep361) | 07:19:33 | L_main=0.5648 L_aux_1=0.5609(w=0.5) L_aux_2=0.6776(w=0.5)
[2026-06-22 22:21:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 375): 11666.1 MiB
[2026-06-22 22:22:21] INFO segtask_v1.trainer.validation:   Val: loss=0.7471, pooled_mean_dice=0.7002, per_class=['0.7002'], iou=0.5386, recall=0.9810, precision=0.5443, vol_sim=0.7137, mcc=0.7226, min_class_dice=0.7002, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8616, per_class_sd=['0.8616'], combined(w=0.50)=0.7809, balanced=0.7135
[2026-06-22 22:22:21] INFO segtask_v1.trainer.trainer: Epoch 376/400 | LR=1.01e-05 | loss=1.3551 | val_dice=0.7002 | best=0.7261 (ep361) | 07:20:42 | L_main=0.5728 L_aux_1=0.5729(w=0.5) L_aux_2=0.6932(w=0.5)
[2026-06-22 22:22:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 376): 11666.1 MiB
[2026-06-22 22:23:30] INFO segtask_v1.trainer.validation:   Val: loss=0.7950, pooled_mean_dice=0.6602, per_class=['0.6602'], iou=0.4927, recall=0.9798, precision=0.4978, vol_sim=0.6738, mcc=0.6886, min_class_dice=0.6602, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8428, per_class_sd=['0.8428'], combined(w=0.50)=0.7515, balanced=0.6762
[2026-06-22 22:23:30] INFO segtask_v1.trainer.trainer: Epoch 377/400 | LR=9.33e-06 | loss=1.3418 | val_dice=0.6602 | best=0.7261 (ep361) | 07:21:52 | L_main=0.5676 L_aux_1=0.5652(w=0.5) L_aux_2=0.6794(w=0.5)
[2026-06-22 22:23:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 377): 11666.1 MiB
[2026-06-22 22:24:40] INFO segtask_v1.trainer.validation:   Val: loss=0.7561, pooled_mean_dice=0.6819, per_class=['0.6819'], iou=0.5174, recall=0.9719, precision=0.5252, vol_sim=0.7017, mcc=0.7048, min_class_dice=0.6819, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8498, per_class_sd=['0.8498'], combined(w=0.50)=0.7659, balanced=0.6958
[2026-06-22 22:24:40] INFO segtask_v1.trainer.trainer: Epoch 378/400 | LR=8.63e-06 | loss=1.3515 | val_dice=0.6819 | best=0.7261 (ep361) | 07:23:01 | L_main=0.5715 L_aux_1=0.5681(w=0.5) L_aux_2=0.6854(w=0.5)
[2026-06-22 22:24:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 378): 11666.1 MiB
[2026-06-22 22:25:49] INFO segtask_v1.trainer.validation:   Val: loss=0.7784, pooled_mean_dice=0.6892, per_class=['0.6892'], iou=0.5258, recall=0.9711, precision=0.5342, vol_sim=0.7098, mcc=0.7116, min_class_dice=0.6892, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8737, per_class_sd=['0.8737'], combined(w=0.50)=0.7815, balanced=0.7060
[2026-06-22 22:25:49] INFO segtask_v1.trainer.trainer: Epoch 379/400 | LR=7.95e-06 | loss=1.3548 | val_dice=0.6892 | best=0.7261 (ep361) | 07:24:11 | L_main=0.5721 L_aux_1=0.5727(w=0.5) L_aux_2=0.6844(w=0.5)
[2026-06-22 22:25:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 379): 11666.1 MiB
[2026-06-22 22:26:58] INFO segtask_v1.trainer.validation:   Val: loss=0.7715, pooled_mean_dice=0.6836, per_class=['0.6836'], iou=0.5193, recall=0.9790, precision=0.5251, vol_sim=0.6982, mcc=0.7094, min_class_dice=0.6836, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8646, per_class_sd=['0.8646'], combined(w=0.50)=0.7741, balanced=0.6999
[2026-06-22 22:26:58] INFO segtask_v1.trainer.trainer: Epoch 380/400 | LR=7.31e-06 | loss=1.3755 | val_dice=0.6836 | best=0.7261 (ep361) | 07:25:20 | L_main=0.5815 L_aux_1=0.5789(w=0.5) L_aux_2=0.6961(w=0.5)
[2026-06-22 22:26:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 380): 11666.1 MiB
[2026-06-22 22:28:08] INFO segtask_v1.trainer.validation:   Val: loss=0.7752, pooled_mean_dice=0.7005, per_class=['0.7005'], iou=0.5391, recall=0.9795, precision=0.5453, vol_sim=0.7152, mcc=0.7222, min_class_dice=0.7005, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8723, per_class_sd=['0.8723'], combined(w=0.50)=0.7864, balanced=0.7156
[2026-06-22 22:28:08] INFO segtask_v1.trainer.trainer: Epoch 381/400 | LR=6.69e-06 | loss=1.3587 | val_dice=0.7005 | best=0.7261 (ep361) | 07:26:30 | L_main=0.5733 L_aux_1=0.5738(w=0.5) L_aux_2=0.6913(w=0.5)
[2026-06-22 22:28:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 381): 11666.1 MiB
[2026-06-22 22:29:18] INFO segtask_v1.trainer.validation:   Val: loss=0.7303, pooled_mean_dice=0.7107, per_class=['0.7107'], iou=0.5513, recall=0.9766, precision=0.5587, vol_sim=0.7278, mcc=0.7297, min_class_dice=0.7107, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8685, per_class_sd=['0.8685'], combined(w=0.50)=0.7896, balanced=0.7235
[2026-06-22 22:29:18] INFO segtask_v1.trainer.trainer: Epoch 382/400 | LR=6.11e-06 | loss=1.3567 | val_dice=0.7107 | best=0.7261 (ep361) | 07:27:39 | L_main=0.5729 L_aux_1=0.5739(w=0.5) L_aux_2=0.6884(w=0.5)
[2026-06-22 22:29:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 382): 11666.1 MiB
[2026-06-22 22:30:27] INFO segtask_v1.trainer.validation:   Val: loss=0.7891, pooled_mean_dice=0.6879, per_class=['0.6879'], iou=0.5243, recall=0.9741, precision=0.5317, vol_sim=0.7063, mcc=0.7129, min_class_dice=0.6879, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8859, per_class_sd=['0.8859'], combined(w=0.50)=0.7869, balanced=0.7071
[2026-06-22 22:30:27] INFO segtask_v1.trainer.trainer: Epoch 383/400 | LR=5.56e-06 | loss=1.3493 | val_dice=0.6879 | best=0.7261 (ep361) | 07:28:49 | L_main=0.5700 L_aux_1=0.5695(w=0.5) L_aux_2=0.6835(w=0.5)
[2026-06-22 22:30:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 383): 11666.1 MiB
[2026-06-22 22:31:37] INFO segtask_v1.trainer.validation:   Val: loss=0.7890, pooled_mean_dice=0.6962, per_class=['0.6962'], iou=0.5340, recall=0.9777, precision=0.5405, vol_sim=0.7121, mcc=0.7207, min_class_dice=0.6962, coverage=[73]/88 samples, pooled_mean_surface_dice@2px=0.8854, per_class_sd=['0.8854'], combined(w=0.50)=0.7908, balanced=0.7142
[2026-06-22 22:31:37] INFO segtask_v1.trainer.trainer: Epoch 384/400 | LR=5.04e-06 | loss=1.3385 | val_dice=0.6962 | best=0.7261 (ep361) | 07:29:58 | L_main=0.5671 L_aux_1=0.5628(w=0.5) L_aux_2=0.6738(w=0.5)
[2026-06-22 22:31:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 384): 11666.1 MiB
[2026-06-22 22:32:46] INFO segtask_v1.trainer.validation:   Val: loss=0.8046, pooled_mean_dice=0.6712, per_class=['0.6712'], iou=0.5052, recall=0.9752, precision=0.5117, vol_sim=0.6883, mcc=0.6979, min_class_dice=0.6712, coverage=[76]/88 samples, pooled_mean_surface_dice@2px=0.8482, per_class_sd=['0.8482'], combined(w=0.50)=0.7597, balanced=0.6866
[2026-06-22 22:32:46] INFO segtask_v1.trainer.trainer: Epoch 385/400 | LR=4.55e-06 | loss=1.3716 | val_dice=0.6712 | best=0.7261 (ep361) | 07:31:07 | L_main=0.5796 L_aux_1=0.5787(w=0.5) L_aux_2=0.7020(w=0.5)
[2026-06-22 22:32:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 385): 11666.1 MiB
[2026-06-22 22:33:55] INFO segtask_v1.trainer.validation:   Val: loss=0.8114, pooled_mean_dice=0.6563, per_class=['0.6563'], iou=0.4884, recall=0.9683, precision=0.4963, vol_sim=0.6777, mcc=0.6853, min_class_dice=0.6563, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8571, per_class_sd=['0.8571'], combined(w=0.50)=0.7567, balanced=0.6751
[2026-06-22 22:33:55] INFO segtask_v1.trainer.trainer: Epoch 386/400 | LR=4.09e-06 | loss=1.3465 | val_dice=0.6563 | best=0.7261 (ep361) | 07:32:17 | L_main=0.5696 L_aux_1=0.5654(w=0.5) L_aux_2=0.6867(w=0.5)
[2026-06-22 22:33:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 386): 11666.1 MiB
[2026-06-22 22:35:05] INFO segtask_v1.trainer.validation:   Val: loss=0.7395, pooled_mean_dice=0.7152, per_class=['0.7152'], iou=0.5567, recall=0.9824, precision=0.5623, vol_sim=0.7280, mcc=0.7340, min_class_dice=0.7152, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8749, per_class_sd=['0.8749'], combined(w=0.50)=0.7951, balanced=0.7285
[2026-06-22 22:35:09] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_multirf/best_model.pth
[2026-06-22 22:35:09] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.7285 at epoch 387
[2026-06-22 22:35:09] INFO segtask_v1.trainer.trainer: Epoch 387/400 | LR=3.67e-06 | loss=1.3645 | val_dice=0.7152 | best=0.7285 (ep387) | 07:33:30 | L_main=0.5761 L_aux_1=0.5736(w=0.5) L_aux_2=0.7001(w=0.5)
[2026-06-22 22:35:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 387): 11666.1 MiB
[2026-06-22 22:36:18] INFO segtask_v1.trainer.validation:   Val: loss=0.7667, pooled_mean_dice=0.6796, per_class=['0.6796'], iou=0.5147, recall=0.9804, precision=0.5200, vol_sim=0.6932, mcc=0.7045, min_class_dice=0.6796, coverage=[79]/88 samples, pooled_mean_surface_dice@2px=0.8622, per_class_sd=['0.8622'], combined(w=0.50)=0.7709, balanced=0.6960
[2026-06-22 22:36:18] INFO segtask_v1.trainer.trainer: Epoch 388/400 | LR=3.27e-06 | loss=1.3558 | val_dice=0.6796 | best=0.7285 (ep387) | 07:34:40 | L_main=0.5729 L_aux_1=0.5683(w=0.5) L_aux_2=0.6911(w=0.5)
[2026-06-22 22:36:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 388): 11666.1 MiB
[2026-06-22 22:37:28] INFO segtask_v1.trainer.validation:   Val: loss=0.7740, pooled_mean_dice=0.6852, per_class=['0.6852'], iou=0.5212, recall=0.9813, precision=0.5264, vol_sim=0.6983, mcc=0.7109, min_class_dice=0.6852, coverage=[71]/88 samples, pooled_mean_surface_dice@2px=0.8640, per_class_sd=['0.8640'], combined(w=0.50)=0.7746, balanced=0.7012
[2026-06-22 22:37:28] INFO segtask_v1.trainer.trainer: Epoch 389/400 | LR=2.91e-06 | loss=1.3661 | val_dice=0.6852 | best=0.7285 (ep387) | 07:35:50 | L_main=0.5780 L_aux_1=0.5745(w=0.5) L_aux_2=0.6901(w=0.5)
[2026-06-22 22:37:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 389): 11666.1 MiB
[2026-06-22 22:38:39] INFO segtask_v1.trainer.validation:   Val: loss=0.7695, pooled_mean_dice=0.6902, per_class=['0.6902'], iou=0.5270, recall=0.9789, precision=0.5330, vol_sim=0.7051, mcc=0.7153, min_class_dice=0.6902, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8706, per_class_sd=['0.8706'], combined(w=0.50)=0.7804, balanced=0.7066
[2026-06-22 22:38:39] INFO segtask_v1.trainer.trainer: Epoch 390/400 | LR=2.58e-06 | loss=1.3447 | val_dice=0.6902 | best=0.7285 (ep387) | 07:37:00 | L_main=0.5667 L_aux_1=0.5669(w=0.5) L_aux_2=0.6875(w=0.5)
[2026-06-22 22:38:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 390): 11666.1 MiB
[2026-06-22 22:39:50] INFO segtask_v1.trainer.validation:   Val: loss=0.7517, pooled_mean_dice=0.6915, per_class=['0.6915'], iou=0.5285, recall=0.9807, precision=0.5341, vol_sim=0.7051, mcc=0.7165, min_class_dice=0.6915, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8733, per_class_sd=['0.8733'], combined(w=0.50)=0.7824, balanced=0.7082
[2026-06-22 22:39:50] INFO segtask_v1.trainer.trainer: Epoch 391/400 | LR=2.28e-06 | loss=1.3283 | val_dice=0.6915 | best=0.7285 (ep387) | 07:38:12 | L_main=0.5622 L_aux_1=0.5567(w=0.5) L_aux_2=0.6686(w=0.5)
[2026-06-22 22:39:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 391): 11666.1 MiB
[2026-06-22 22:41:00] INFO segtask_v1.trainer.validation:   Val: loss=0.8226, pooled_mean_dice=0.6764, per_class=['0.6764'], iou=0.5110, recall=0.9626, precision=0.5213, vol_sim=0.7026, mcc=0.7016, min_class_dice=0.6764, coverage=[75]/88 samples, pooled_mean_surface_dice@2px=0.8753, per_class_sd=['0.8753'], combined(w=0.50)=0.7758, balanced=0.6953
[2026-06-22 22:41:00] INFO segtask_v1.trainer.trainer: Epoch 392/400 | LR=2.01e-06 | loss=1.3570 | val_dice=0.6764 | best=0.7285 (ep387) | 07:39:22 | L_main=0.5732 L_aux_1=0.5711(w=0.5) L_aux_2=0.6846(w=0.5)
[2026-06-22 22:41:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 392): 11666.1 MiB
[2026-06-22 22:42:10] INFO segtask_v1.trainer.validation:   Val: loss=0.7964, pooled_mean_dice=0.6725, per_class=['0.6725'], iou=0.5066, recall=0.9646, precision=0.5161, vol_sim=0.6971, mcc=0.6979, min_class_dice=0.6725, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8646, per_class_sd=['0.8646'], combined(w=0.50)=0.7686, balanced=0.6902
[2026-06-22 22:42:10] INFO segtask_v1.trainer.trainer: Epoch 393/400 | LR=1.77e-06 | loss=1.3915 | val_dice=0.6725 | best=0.7285 (ep387) | 07:40:32 | L_main=0.5882 L_aux_1=0.5874(w=0.5) L_aux_2=0.7101(w=0.5)
[2026-06-22 22:42:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 393): 11666.1 MiB
[2026-06-22 22:43:20] INFO segtask_v1.trainer.validation:   Val: loss=0.7680, pooled_mean_dice=0.6895, per_class=['0.6895'], iou=0.5261, recall=0.9744, precision=0.5335, vol_sim=0.7076, mcc=0.7110, min_class_dice=0.6895, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8610, per_class_sd=['0.8610'], combined(w=0.50)=0.7752, balanced=0.7041
[2026-06-22 22:43:20] INFO segtask_v1.trainer.trainer: Epoch 394/400 | LR=1.57e-06 | loss=1.3759 | val_dice=0.6895 | best=0.7285 (ep387) | 07:41:42 | L_main=0.5842 L_aux_1=0.5848(w=0.5) L_aux_2=0.6931(w=0.5)
[2026-06-22 22:43:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 394): 11666.1 MiB
[2026-06-22 22:44:30] INFO segtask_v1.trainer.validation:   Val: loss=0.7786, pooled_mean_dice=0.6850, per_class=['0.6850'], iou=0.5209, recall=0.9637, precision=0.5314, vol_sim=0.7109, mcc=0.7071, min_class_dice=0.6850, coverage=[77]/88 samples, pooled_mean_surface_dice@2px=0.8683, per_class_sd=['0.8683'], combined(w=0.50)=0.7767, balanced=0.7014
[2026-06-22 22:44:30] INFO segtask_v1.trainer.trainer: Epoch 395/400 | LR=1.39e-06 | loss=1.3447 | val_dice=0.6850 | best=0.7285 (ep387) | 07:42:52 | L_main=0.5679 L_aux_1=0.5639(w=0.5) L_aux_2=0.6824(w=0.5)
[2026-06-22 22:44:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 395): 11666.1 MiB
[2026-06-22 22:45:40] INFO segtask_v1.trainer.validation:   Val: loss=0.7808, pooled_mean_dice=0.6934, per_class=['0.6934'], iou=0.5306, recall=0.9797, precision=0.5365, vol_sim=0.7077, mcc=0.7169, min_class_dice=0.6934, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8532, per_class_sd=['0.8532'], combined(w=0.50)=0.7733, balanced=0.7063
[2026-06-22 22:45:40] INFO segtask_v1.trainer.trainer: Epoch 396/400 | LR=1.25e-06 | loss=1.3521 | val_dice=0.6934 | best=0.7285 (ep387) | 07:44:02 | L_main=0.5706 L_aux_1=0.5688(w=0.5) L_aux_2=0.6896(w=0.5)
[2026-06-22 22:45:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 396): 11666.1 MiB
[2026-06-22 22:46:50] INFO segtask_v1.trainer.validation:   Val: loss=0.7753, pooled_mean_dice=0.6973, per_class=['0.6973'], iou=0.5352, recall=0.9693, precision=0.5445, vol_sim=0.7194, mcc=0.7171, min_class_dice=0.6973, coverage=[78]/88 samples, pooled_mean_surface_dice@2px=0.8644, per_class_sd=['0.8644'], combined(w=0.50)=0.7808, balanced=0.7112
[2026-06-22 22:46:50] INFO segtask_v1.trainer.trainer: Epoch 397/400 | LR=1.14e-06 | loss=1.3551 | val_dice=0.6973 | best=0.7285 (ep387) | 07:45:11 | L_main=0.5716 L_aux_1=0.5716(w=0.5) L_aux_2=0.6862(w=0.5)
[2026-06-22 22:46:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 397): 11666.1 MiB
[2026-06-22 22:47:59] INFO segtask_v1.trainer.validation:   Val: loss=0.7713, pooled_mean_dice=0.6928, per_class=['0.6928'], iou=0.5299, recall=0.9723, precision=0.5381, vol_sim=0.7125, mcc=0.7143, min_class_dice=0.6928, coverage=[72]/88 samples, pooled_mean_surface_dice@2px=0.8610, per_class_sd=['0.8610'], combined(w=0.50)=0.7769, balanced=0.7069
[2026-06-22 22:47:59] INFO segtask_v1.trainer.trainer: Epoch 398/400 | LR=1.06e-06 | loss=1.3661 | val_dice=0.6928 | best=0.7285 (ep387) | 07:46:21 | L_main=0.5767 L_aux_1=0.5782(w=0.5) L_aux_2=0.6924(w=0.5)
[2026-06-22 22:47:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 398): 11666.1 MiB
[2026-06-22 22:49:10] INFO segtask_v1.trainer.validation:   Val: loss=0.8161, pooled_mean_dice=0.6713, per_class=['0.6713'], iou=0.5053, recall=0.9588, precision=0.5165, vol_sim=0.7002, mcc=0.6957, min_class_dice=0.6713, coverage=[70]/88 samples, pooled_mean_surface_dice@2px=0.8663, per_class_sd=['0.8663'], combined(w=0.50)=0.7688, balanced=0.6894
[2026-06-22 22:49:10] INFO segtask_v1.trainer.trainer: Epoch 399/400 | LR=1.02e-06 | loss=1.3554 | val_dice=0.6713 | best=0.7285 (ep387) | 07:47:31 | L_main=0.5734 L_aux_1=0.5751(w=0.5) L_aux_2=0.6838(w=0.5)
[2026-06-22 22:49:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 399): 11666.1 MiB
[2026-06-22 22:50:20] INFO segtask_v1.trainer.validation:   Val: loss=0.7918, pooled_mean_dice=0.6805, per_class=['0.6805'], iou=0.5157, recall=0.9807, precision=0.5210, vol_sim=0.6938, mcc=0.7075, min_class_dice=0.6805, coverage=[74]/88 samples, pooled_mean_surface_dice@2px=0.8683, per_class_sd=['0.8683'], combined(w=0.50)=0.7744, balanced=0.6979
[2026-06-22 22:50:20] INFO segtask_v1.trainer.trainer: Epoch 400/400 | LR=1.00e-06 | loss=1.3674 | val_dice=0.6805 | best=0.7285 (ep387) | 07:48:41 | L_main=0.5782 L_aux_1=0.5748(w=0.5) L_aux_2=0.6958(w=0.5)
[2026-06-22 22:50:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 400): 11666.1 MiB
[2026-06-22 22:50:21] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 22:50:21] INFO segtask_v1.trainer.trainer: Training complete. Best mean_balanced=0.7285 at epoch 387. Time: 07:48:42
[2026-06-22 22:50:21] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 22:50:21] INFO __main__: Best metrics: {'val_loss': 0.739473207430406, 'dice_class_0': 0.7152242064476013, 'iou_class_0': 0.5566918849945068, 'recall_class_0': 0.9824241399765015, 'precision_class_0': 0.5622919201850891, 'vol_sim_class_0': 0.7280197739601135, 'mcc_class_0': 0.7339785099029541, 'mean_dice': 0.7152242064476013, 'mean_iou': 0.5566918849945068, 'mean_recall': 0.9824241399765015, 'mean_precision': 0.5622919201850891, 'mean_vol_sim': 0.7280197739601135, 'mean_mcc': 0.7339785099029541, 'min_class_dice': 0.7152242064476013, 'min_class_iou': 0.5566918849945068, 'surface_dice_class_0': 0.8749454617500305, 'mean_surface_dice': 0.8749454617500305, 'mean_combined': 0.7950848340988159, 'mean_balanced': 0.7284972667694092}


lungves1.yaml:
[2026-06-22 15:36:17] INFO __main__: Config loaded from: configs/segtest1.yaml
[2026-06-22 15:36:17] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-22 15:36:17] INFO __main__: Device: cuda
[2026-06-22 15:36:17] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-22 15:36:17] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-22 15:36:17] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-22 15:36:17] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-22 15:36:32] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-22 15:36:32] INFO segtask_v1.data.specs: Using CUBIC patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], max_scale=2.00) — SINGLE max-FOV cube extraction; trainer crops+resizes per view before the 3D forward.
[2026-06-22 15:36:32] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 88 npz packages...
[2026-06-22 15:37:10] INFO segtask_v1.data.dataset: NPZ cubic index: 88 volumes, 4400000 fg voxels sampled
[2026-06-22 15:37:10] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 22 npz packages...
[2026-06-22 15:37:20] INFO segtask_v1.data.dataset: NPZ cubic index: 22 volumes, 1100000 fg voxels sampled
[2026-06-22 15:37:20] INFO segtask_v1.data.loader: DataLoader: batch_size=2, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-22 15:37:21] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=24, num_workers=16 => up to ~87.40 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-22 15:37:21] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=34.48M, dec=17.20M, total=54.11M, channels=[64, 64, 128, 256, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=3 (fg=1, res=3), stem=dual(stride=1, n_views=1, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=True, ds=True, aux_seg=False(n_aux_heads=0, mode=conv)
[2026-06-22 15:37:22] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Patch3DNativeMultiResPipeline (patch_mode=cubic, n_views=3)
[2026-06-22 15:37:22] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=distance, loss=smooth_l1, iter=5, weight=0.300)
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: Model params: 54.11M
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=206.4 + grad=206.4 + optim(AdamW,2x)=412.8 + ema=206.5 = 1032.2 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=422.1 MiB, reserved=442.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: Train batches: 352, Val batches: 44
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: AMP=False (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: Grad accum=4, Effective batch=8
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: Pipeline=Patch3DNativeMultiResPipeline | n_views=3, n_aux_views=0, num_res_groups=3, slab_depth=0 | fg_classes=1, Loss=dice_cldice
[2026-06-22 15:37:22] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 15:39:59] INFO __main__: Config loaded from: configs/segtest1.yaml
[2026-06-22 15:39:59] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-22 15:40:00] INFO __main__: Device: cuda
[2026-06-22 15:40:00] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-22 15:40:00] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-22 15:40:00] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-22 15:40:00] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-22 15:40:14] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-22 15:40:14] INFO segtask_v1.data.specs: Using CUBIC patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], max_scale=2.00) — SINGLE max-FOV cube extraction; trainer crops+resizes per view before the 3D forward.
[2026-06-22 15:40:14] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 88 npz packages...
[2026-06-22 15:40:50] INFO segtask_v1.data.dataset: NPZ cubic index: 88 volumes, 4400000 fg voxels sampled
[2026-06-22 15:40:50] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 22 npz packages...
[2026-06-22 15:41:00] INFO segtask_v1.data.dataset: NPZ cubic index: 22 volumes, 1100000 fg voxels sampled
[2026-06-22 15:41:00] INFO segtask_v1.data.loader: DataLoader: batch_size=1, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-22 15:41:00] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=36, num_workers=16 => up to ~131.10 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-22 15:41:01] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=34.48M, dec=17.20M, total=54.11M, channels=[64, 64, 128, 256, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=3 (fg=1, res=3), stem=dual(stride=1, n_views=1, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=True, ds=True, aux_seg=False(n_aux_heads=0, mode=conv)
[2026-06-22 15:41:02] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Patch3DNativeMultiResPipeline (patch_mode=cubic, n_views=3)
[2026-06-22 15:41:02] INFO segtask_v1.trainer.pipelines.factory: Aux topo head: ENABLED (target=distance, loss=smooth_l1, iter=5, weight=0.300)
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: Model params: 54.11M
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=206.4 + grad=206.4 + optim(AdamW,2x)=412.8 + ema=206.4 = 1032.1 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=421.9 MiB, reserved=442.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: Train batches: 704, Val batches: 88
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: AMP=False (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: Grad accum=8, Effective batch=8
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: Pipeline=Patch3DNativeMultiResPipeline | n_views=3, n_aux_views=0, num_res_groups=3, slab_depth=0 | fg_classes=1, Loss=dice_cldice
[2026-06-22 15:41:02] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-22 15:41:31] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 14411.5 MiB (forward + backward + optimizer.step + EMA update; accum=8 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-22 15:55:52] INFO segtask_v1.trainer.validation:   Val: loss=1.6579, pooled_mean_dice=0.2639, per_class=['0.2639'], iou=0.1520, recall=0.8287, precision=0.1569, vol_sim=0.3184, mcc=-0.0392, min_class_dice=0.2639, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.3133, per_class_sd=['0.3133'], combined(w=0.50)=0.2886, balanced=0.2558
[2026-06-22 15:55:54] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-22 15:55:54] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2558 at epoch 1
[2026-06-22 15:55:54] INFO segtask_v1.trainer.trainer: Epoch 1/400 | LR=2.60e-05 | loss=1.1212 | val_dice=0.2639 | best=0.2558 (ep1) | 00:14:51 | L_res_0=0.9839 L_res_1=1.0960 L_res_2=1.2420
[2026-06-22 15:55:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 14411.5 MiB
[2026-06-22 16:10:12] INFO segtask_v1.trainer.validation:   Val: loss=1.5883, pooled_mean_dice=0.2998, per_class=['0.2998'], iou=0.1764, recall=0.9966, precision=0.1765, vol_sim=0.3009, mcc=0.0137, min_class_dice=0.2998, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2005, per_class_sd=['0.2005'], combined(w=0.50)=0.2502, balanced=0.2506
[2026-06-22 16:10:12] INFO segtask_v1.trainer.trainer: Epoch 2/400 | LR=5.10e-05 | loss=0.8348 | val_dice=0.2998 | best=0.2558 (ep1) | 00:29:09 | L_res_0=0.7643 L_res_1=0.8651 L_res_2=1.0283
[2026-06-22 16:10:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 12583.8 MiB
[2026-06-22 16:24:33] INFO segtask_v1.trainer.validation:   Val: loss=1.5562, pooled_mean_dice=0.2982, per_class=['0.2982'], iou=0.1752, recall=1.0000, precision=0.1752, vol_sim=0.2982, mcc=0.0021, min_class_dice=0.2982, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1917, per_class_sd=['0.1917'], combined(w=0.50)=0.2449, balanced=0.2458
[2026-06-22 16:24:33] INFO segtask_v1.trainer.trainer: Epoch 3/400 | LR=7.59e-05 | loss=0.7969 | val_dice=0.2982 | best=0.2558 (ep1) | 00:43:31 | L_res_0=0.7349 L_res_1=0.8471 L_res_2=0.9966
[2026-06-22 16:24:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 12583.8 MiB
[2026-06-22 16:38:55] INFO segtask_v1.trainer.validation:   Val: loss=1.4952, pooled_mean_dice=0.3180, per_class=['0.3180'], iou=0.1890, recall=1.0000, precision=0.1890, vol_sim=0.3180, mcc=0.0003, min_class_dice=0.3180, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1944, per_class_sd=['0.1944'], combined(w=0.50)=0.2562, balanced=0.2568
[2026-06-22 16:39:00] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-22 16:39:00] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2568 at epoch 4
[2026-06-22 16:39:00] INFO segtask_v1.trainer.trainer: Epoch 4/400 | LR=1.01e-04 | loss=0.7430 | val_dice=0.3180 | best=0.2568 (ep4) | 00:57:58 | L_res_0=0.6763 L_res_1=0.7926 L_res_2=0.9443
[2026-06-22 16:39:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 4): 12583.8 MiB
[2026-06-22 16:53:22] INFO segtask_v1.trainer.validation:   Val: loss=1.5110, pooled_mean_dice=0.2982, per_class=['0.2982'], iou=0.1752, recall=1.0000, precision=0.1752, vol_sim=0.2982, mcc=0.0000, min_class_dice=0.2982, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1934, per_class_sd=['0.1934'], combined(w=0.50)=0.2458, balanced=0.2464
[2026-06-22 16:53:22] INFO segtask_v1.trainer.trainer: Epoch 5/400 | LR=1.26e-04 | loss=0.7442 | val_dice=0.2982 | best=0.2568 (ep4) | 01:12:20 | L_res_0=0.6613 L_res_1=0.7825 L_res_2=0.9341
[2026-06-22 16:53:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 5): 12583.8 MiB
[2026-06-22 17:07:30] INFO segtask_v1.trainer.validation:   Val: loss=1.4663, pooled_mean_dice=0.3025, per_class=['0.3025'], iou=0.1782, recall=1.0000, precision=0.1782, vol_sim=0.3025, mcc=0.0000, min_class_dice=0.3025, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1960, per_class_sd=['0.1960'], combined(w=0.50)=0.2493, balanced=0.2497
[2026-06-22 17:07:30] INFO segtask_v1.trainer.trainer: Epoch 6/400 | LR=1.51e-04 | loss=0.7772 | val_dice=0.3025 | best=0.2568 (ep4) | 01:26:28 | L_res_0=0.7230 L_res_1=0.8382 L_res_2=0.9568
[2026-06-22 17:07:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 6): 12583.8 MiB
[2026-06-22 17:21:51] INFO segtask_v1.trainer.validation:   Val: loss=1.4623, pooled_mean_dice=0.2846, per_class=['0.2846'], iou=0.1659, recall=1.0000, precision=0.1659, vol_sim=0.2846, mcc=0.0000, min_class_dice=0.2846, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1907, per_class_sd=['0.1907'], combined(w=0.50)=0.2377, balanced=0.2383
[2026-06-22 17:21:51] INFO segtask_v1.trainer.trainer: Epoch 7/400 | LR=1.76e-04 | loss=0.7151 | val_dice=0.2846 | best=0.2568 (ep4) | 01:40:49 | L_res_0=0.6629 L_res_1=0.7705 L_res_2=0.8775
[2026-06-22 17:21:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 7): 12583.8 MiB
[2026-06-22 17:36:14] INFO segtask_v1.trainer.validation:   Val: loss=1.4145, pooled_mean_dice=0.3010, per_class=['0.3010'], iou=0.1771, recall=1.0000, precision=0.1771, vol_sim=0.3010, mcc=0.0000, min_class_dice=0.3010, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1981, per_class_sd=['0.1981'], combined(w=0.50)=0.2496, balanced=0.2498
[2026-06-22 17:36:14] INFO segtask_v1.trainer.trainer: Epoch 8/400 | LR=2.01e-04 | loss=0.6780 | val_dice=0.3010 | best=0.2568 (ep4) | 01:55:12 | L_res_0=0.6187 L_res_1=0.7208 L_res_2=0.8125
[2026-06-22 17:36:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 8): 12583.8 MiB
[2026-06-22 17:50:40] INFO segtask_v1.trainer.validation:   Val: loss=1.4018, pooled_mean_dice=0.2941, per_class=['0.2941'], iou=0.1724, recall=1.0000, precision=0.1724, vol_sim=0.2941, mcc=0.0000, min_class_dice=0.2941, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1998, per_class_sd=['0.1998'], combined(w=0.50)=0.2469, balanced=0.2468
[2026-06-22 17:50:40] INFO segtask_v1.trainer.trainer: Epoch 9/400 | LR=2.26e-04 | loss=0.6588 | val_dice=0.2941 | best=0.2568 (ep4) | 02:09:38 | L_res_0=0.5849 L_res_1=0.6971 L_res_2=0.7835
[2026-06-22 17:50:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 9): 12583.8 MiB
[2026-06-22 18:05:02] INFO segtask_v1.trainer.validation:   Val: loss=1.3692, pooled_mean_dice=0.2937, per_class=['0.2937'], iou=0.1721, recall=1.0000, precision=0.1721, vol_sim=0.2937, mcc=0.0000, min_class_dice=0.2937, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1929, per_class_sd=['0.1929'], combined(w=0.50)=0.2433, balanced=0.2439
[2026-06-22 18:05:02] INFO segtask_v1.trainer.trainer: Epoch 10/400 | LR=2.51e-04 | loss=0.6530 | val_dice=0.2937 | best=0.2568 (ep4) | 02:24:00 | L_res_0=0.5645 L_res_1=0.6899 L_res_2=0.7757
[2026-06-22 18:05:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 10): 12583.8 MiB
[2026-06-22 18:19:10] INFO segtask_v1.trainer.validation:   Val: loss=1.3416, pooled_mean_dice=0.3013, per_class=['0.3013'], iou=0.1774, recall=1.0000, precision=0.1774, vol_sim=0.3013, mcc=0.0000, min_class_dice=0.3013, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1930, per_class_sd=['0.1930'], combined(w=0.50)=0.2472, balanced=0.2479
[2026-06-22 18:19:11] INFO segtask_v1.trainer.trainer: Epoch 11/400 | LR=2.76e-04 | loss=0.6480 | val_dice=0.3013 | best=0.2568 (ep4) | 02:38:08 | L_res_0=0.5497 L_res_1=0.6878 L_res_2=0.7747
[2026-06-22 18:19:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 11): 12583.8 MiB
[2026-06-22 18:33:21] INFO segtask_v1.trainer.validation:   Val: loss=1.3127, pooled_mean_dice=0.2983, per_class=['0.2983'], iou=0.1753, recall=1.0000, precision=0.1753, vol_sim=0.2983, mcc=0.0000, min_class_dice=0.2983, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1926, per_class_sd=['0.1926'], combined(w=0.50)=0.2455, balanced=0.2462
[2026-06-22 18:33:21] INFO segtask_v1.trainer.trainer: Epoch 12/400 | LR=3.01e-04 | loss=0.6333 | val_dice=0.2983 | best=0.2568 (ep4) | 02:52:19 | L_res_0=0.5270 L_res_1=0.6696 L_res_2=0.7574
[2026-06-22 18:33:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 12): 12583.8 MiB
[2026-06-22 18:47:28] INFO segtask_v1.trainer.validation:   Val: loss=1.2699, pooled_mean_dice=0.3067, per_class=['0.3067'], iou=0.1811, recall=1.0000, precision=0.1811, vol_sim=0.3067, mcc=0.0000, min_class_dice=0.3067, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2016, per_class_sd=['0.2016'], combined(w=0.50)=0.2541, balanced=0.2541
[2026-06-22 18:47:28] INFO segtask_v1.trainer.trainer: Epoch 13/400 | LR=3.26e-04 | loss=0.6273 | val_dice=0.3067 | best=0.2568 (ep4) | 03:06:26 | L_res_0=0.5149 L_res_1=0.6617 L_res_2=0.7520
[2026-06-22 18:47:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 13): 12583.8 MiB
[2026-06-22 19:01:37] INFO segtask_v1.trainer.validation:   Val: loss=1.2463, pooled_mean_dice=0.2984, per_class=['0.2984'], iou=0.1754, recall=1.0000, precision=0.1754, vol_sim=0.2984, mcc=0.0000, min_class_dice=0.2984, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1997, per_class_sd=['0.1997'], combined(w=0.50)=0.2491, balanced=0.2491
[2026-06-22 19:01:37] INFO segtask_v1.trainer.trainer: Epoch 14/400 | LR=3.51e-04 | loss=0.6363 | val_dice=0.2984 | best=0.2568 (ep4) | 03:20:35 | L_res_0=0.5304 L_res_1=0.6680 L_res_2=0.7629
[2026-06-22 19:01:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 14): 12583.8 MiB
[2026-06-22 19:15:46] INFO segtask_v1.trainer.validation:   Val: loss=1.1839, pooled_mean_dice=0.3171, per_class=['0.3171'], iou=0.1884, recall=1.0000, precision=0.1884, vol_sim=0.3171, mcc=0.0000, min_class_dice=0.3171, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1982, per_class_sd=['0.1982'], combined(w=0.50)=0.2576, balanced=0.2580
[2026-06-22 19:15:51] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-22 19:15:51] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2580 at epoch 15
[2026-06-22 19:15:51] INFO segtask_v1.trainer.trainer: Epoch 15/400 | LR=3.76e-04 | loss=0.6348 | val_dice=0.3171 | best=0.2580 (ep15) | 03:34:49 | L_res_0=0.5199 L_res_1=0.6637 L_res_2=0.7616
[2026-06-22 19:15:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 15): 12583.8 MiB
[2026-06-22 19:29:58] INFO segtask_v1.trainer.validation:   Val: loss=1.1499, pooled_mean_dice=0.3105, per_class=['0.3105'], iou=0.1838, recall=1.0000, precision=0.1838, vol_sim=0.3105, mcc=0.0000, min_class_dice=0.3105, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1992, per_class_sd=['0.1992'], combined(w=0.50)=0.2549, balanced=0.2551
[2026-06-22 19:29:58] INFO segtask_v1.trainer.trainer: Epoch 16/400 | LR=4.01e-04 | loss=0.6216 | val_dice=0.3105 | best=0.2580 (ep15) | 03:48:56 | L_res_0=0.5068 L_res_1=0.6441 L_res_2=0.7500
[2026-06-22 19:29:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 16): 12583.8 MiB
[2026-06-22 19:44:08] INFO segtask_v1.trainer.validation:   Val: loss=1.1306, pooled_mean_dice=0.2905, per_class=['0.2905'], iou=0.1700, recall=1.0000, precision=0.1700, vol_sim=0.2905, mcc=0.0000, min_class_dice=0.2905, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1996, per_class_sd=['0.1996'], combined(w=0.50)=0.2451, balanced=0.2449
[2026-06-22 19:44:08] INFO segtask_v1.trainer.trainer: Epoch 17/400 | LR=4.26e-04 | loss=0.6189 | val_dice=0.2905 | best=0.2580 (ep15) | 04:03:06 | L_res_0=0.5061 L_res_1=0.6312 L_res_2=0.7497
[2026-06-22 19:44:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 17): 12583.8 MiB
[2026-06-22 19:58:32] INFO segtask_v1.trainer.validation:   Val: loss=1.0916, pooled_mean_dice=0.2976, per_class=['0.2976'], iou=0.1748, recall=1.0000, precision=0.1748, vol_sim=0.2976, mcc=0.0000, min_class_dice=0.2976, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1928, per_class_sd=['0.1928'], combined(w=0.50)=0.2452, balanced=0.2459
[2026-06-22 19:58:32] INFO segtask_v1.trainer.trainer: Epoch 18/400 | LR=4.51e-04 | loss=0.6126 | val_dice=0.2976 | best=0.2580 (ep15) | 04:17:30 | L_res_0=0.4991 L_res_1=0.6044 L_res_2=0.7435
[2026-06-22 19:58:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 18): 12583.8 MiB
[2026-06-22 20:12:51] INFO segtask_v1.trainer.validation:   Val: loss=1.0524, pooled_mean_dice=0.3089, per_class=['0.3089'], iou=0.1826, recall=1.0000, precision=0.1826, vol_sim=0.3089, mcc=0.0000, min_class_dice=0.3089, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1993, per_class_sd=['0.1993'], combined(w=0.50)=0.2541, balanced=0.2543
[2026-06-22 20:12:51] INFO segtask_v1.trainer.trainer: Epoch 19/400 | LR=4.76e-04 | loss=0.6082 | val_dice=0.3089 | best=0.2580 (ep15) | 04:31:49 | L_res_0=0.4939 L_res_1=0.5972 L_res_2=0.7418
[2026-06-22 20:12:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 19): 12583.8 MiB
[2026-06-22 20:27:15] INFO segtask_v1.trainer.validation:   Val: loss=1.0560, pooled_mean_dice=0.2961, per_class=['0.2961'], iou=0.1738, recall=1.0000, precision=0.1738, vol_sim=0.2961, mcc=0.0000, min_class_dice=0.2961, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1978, per_class_sd=['0.1978'], combined(w=0.50)=0.2470, balanced=0.2471
[2026-06-22 20:27:15] INFO segtask_v1.trainer.trainer: Epoch 20/400 | LR=5.01e-04 | loss=0.6108 | val_dice=0.2961 | best=0.2580 (ep15) | 04:46:13 | L_res_0=0.4973 L_res_1=0.5998 L_res_2=0.7427
[2026-06-22 20:27:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 20): 12583.8 MiB
[2026-06-22 20:41:42] INFO segtask_v1.trainer.validation:   Val: loss=1.0459, pooled_mean_dice=0.3040, per_class=['0.3040'], iou=0.1793, recall=1.0000, precision=0.1793, vol_sim=0.3040, mcc=0.0000, min_class_dice=0.3040, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1950, per_class_sd=['0.1950'], combined(w=0.50)=0.2495, balanced=0.2501
[2026-06-22 20:41:42] INFO segtask_v1.trainer.trainer: Epoch 21/400 | LR=5.25e-04 | loss=0.6106 | val_dice=0.3040 | best=0.2580 (ep15) | 05:00:39 | L_res_0=0.4958 L_res_1=0.6003 L_res_2=0.7417
[2026-06-22 20:41:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 21): 12583.8 MiB
[2026-06-22 20:56:01] INFO segtask_v1.trainer.validation:   Val: loss=1.0348, pooled_mean_dice=0.3110, per_class=['0.3110'], iou=0.1842, recall=1.0000, precision=0.1842, vol_sim=0.3110, mcc=0.0000, min_class_dice=0.3110, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1981, per_class_sd=['0.1981'], combined(w=0.50)=0.2545, balanced=0.2549
[2026-06-22 20:56:01] INFO segtask_v1.trainer.trainer: Epoch 22/400 | LR=5.50e-04 | loss=0.6071 | val_dice=0.3110 | best=0.2580 (ep15) | 05:14:59 | L_res_0=0.4923 L_res_1=0.5971 L_res_2=0.7395
[2026-06-22 20:56:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 22): 12583.8 MiB
[2026-06-22 21:10:27] INFO segtask_v1.trainer.validation:   Val: loss=1.0442, pooled_mean_dice=0.3014, per_class=['0.3014'], iou=0.1774, recall=1.0000, precision=0.1774, vol_sim=0.3014, mcc=0.0000, min_class_dice=0.3014, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1998, per_class_sd=['0.1998'], combined(w=0.50)=0.2506, balanced=0.2507
[2026-06-22 21:10:27] INFO segtask_v1.trainer.trainer: Epoch 23/400 | LR=5.75e-04 | loss=0.6073 | val_dice=0.3014 | best=0.2580 (ep15) | 05:29:25 | L_res_0=0.4940 L_res_1=0.5972 L_res_2=0.7358
[2026-06-22 21:10:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 23): 12583.8 MiB
[2026-06-22 21:24:50] INFO segtask_v1.trainer.validation:   Val: loss=1.0456, pooled_mean_dice=0.2989, per_class=['0.2989'], iou=0.1757, recall=1.0000, precision=0.1757, vol_sim=0.2989, mcc=0.0000, min_class_dice=0.2989, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1981, per_class_sd=['0.1981'], combined(w=0.50)=0.2485, balanced=0.2487
[2026-06-22 21:24:50] INFO segtask_v1.trainer.trainer: Epoch 24/400 | LR=6.00e-04 | loss=0.6092 | val_dice=0.2989 | best=0.2580 (ep15) | 05:43:47 | L_res_0=0.4949 L_res_1=0.5985 L_res_2=0.7406
[2026-06-22 21:24:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 24): 12583.8 MiB
[2026-06-22 21:39:10] INFO segtask_v1.trainer.validation:   Val: loss=1.0418, pooled_mean_dice=0.3019, per_class=['0.3019'], iou=0.1778, recall=1.0000, precision=0.1778, vol_sim=0.3019, mcc=0.0000, min_class_dice=0.3019, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1902, per_class_sd=['0.1902'], combined(w=0.50)=0.2461, balanced=0.2470
[2026-06-22 21:39:10] INFO segtask_v1.trainer.trainer: Epoch 25/400 | LR=6.25e-04 | loss=0.6051 | val_dice=0.3019 | best=0.2580 (ep15) | 05:58:08 | L_res_0=0.4927 L_res_1=0.5932 L_res_2=0.7370
[2026-06-22 21:39:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 25): 12583.8 MiB
[2026-06-22 21:53:29] INFO segtask_v1.trainer.validation:   Val: loss=1.0342, pooled_mean_dice=0.3074, per_class=['0.3074'], iou=0.1816, recall=1.0000, precision=0.1816, vol_sim=0.3074, mcc=0.0000, min_class_dice=0.3074, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1935, per_class_sd=['0.1935'], combined(w=0.50)=0.2504, balanced=0.2512
[2026-06-22 21:53:29] INFO segtask_v1.trainer.trainer: Epoch 26/400 | LR=6.50e-04 | loss=0.6066 | val_dice=0.3074 | best=0.2580 (ep15) | 06:12:27 | L_res_0=0.4946 L_res_1=0.5930 L_res_2=0.7383
[2026-06-22 21:53:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 26): 12583.8 MiB
[2026-06-22 22:07:49] INFO segtask_v1.trainer.validation:   Val: loss=1.0433, pooled_mean_dice=0.2987, per_class=['0.2987'], iou=0.1756, recall=1.0000, precision=0.1756, vol_sim=0.2987, mcc=0.0000, min_class_dice=0.2987, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1975, per_class_sd=['0.1975'], combined(w=0.50)=0.2481, balanced=0.2484
[2026-06-22 22:07:49] INFO segtask_v1.trainer.trainer: Epoch 27/400 | LR=6.75e-04 | loss=0.6032 | val_dice=0.2987 | best=0.2580 (ep15) | 06:26:47 | L_res_0=0.4894 L_res_1=0.5900 L_res_2=0.7349
[2026-06-22 22:07:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 27): 12583.8 MiB
[2026-06-22 22:22:12] INFO segtask_v1.trainer.validation:   Val: loss=1.0265, pooled_mean_dice=0.3160, per_class=['0.3160'], iou=0.1876, recall=1.0000, precision=0.1876, vol_sim=0.3160, mcc=0.0000, min_class_dice=0.3160, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1969, per_class_sd=['0.1969'], combined(w=0.50)=0.2565, balanced=0.2569
[2026-06-22 22:22:12] INFO segtask_v1.trainer.trainer: Epoch 28/400 | LR=7.00e-04 | loss=0.6044 | val_dice=0.3160 | best=0.2580 (ep15) | 06:41:10 | L_res_0=0.4918 L_res_1=0.5914 L_res_2=0.7376
[2026-06-22 22:22:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 28): 12583.8 MiB
[2026-06-22 22:36:32] INFO segtask_v1.trainer.validation:   Val: loss=1.0426, pooled_mean_dice=0.3000, per_class=['0.3000'], iou=0.1765, recall=1.0000, precision=0.1765, vol_sim=0.3000, mcc=0.0000, min_class_dice=0.3000, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1963, per_class_sd=['0.1963'], combined(w=0.50)=0.2482, balanced=0.2486
[2026-06-22 22:36:32] INFO segtask_v1.trainer.trainer: Epoch 29/400 | LR=7.25e-04 | loss=0.6039 | val_dice=0.3000 | best=0.2580 (ep15) | 06:55:30 | L_res_0=0.4899 L_res_1=0.5907 L_res_2=0.7368
[2026-06-22 22:36:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 29): 12583.8 MiB
[2026-06-22 22:50:53] INFO segtask_v1.trainer.validation:   Val: loss=1.0327, pooled_mean_dice=0.3102, per_class=['0.3102'], iou=0.1836, recall=1.0000, precision=0.1836, vol_sim=0.3102, mcc=0.0000, min_class_dice=0.3102, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1996, per_class_sd=['0.1996'], combined(w=0.50)=0.2549, balanced=0.2551
[2026-06-22 22:50:53] INFO segtask_v1.trainer.trainer: Epoch 30/400 | LR=7.50e-04 | loss=0.6036 | val_dice=0.3102 | best=0.2580 (ep15) | 07:09:50 | L_res_0=0.4900 L_res_1=0.5905 L_res_2=0.7369
[2026-06-22 22:50:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 30): 12583.8 MiB
[2026-06-22 23:04:58] INFO segtask_v1.trainer.validation:   Val: loss=1.0226, pooled_mean_dice=0.3200, per_class=['0.3200'], iou=0.1904, recall=1.0000, precision=0.1904, vol_sim=0.3200, mcc=0.0000, min_class_dice=0.3200, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2024, per_class_sd=['0.2024'], combined(w=0.50)=0.2612, balanced=0.2612
[2026-06-22 23:05:03] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves3d/best_model.pth
[2026-06-22 23:05:03] INFO segtask_v1.trainer.trainer: ★ New best: mean_balanced=0.2612 at epoch 31
[2026-06-22 23:05:03] INFO segtask_v1.trainer.trainer: Epoch 31/400 | LR=7.75e-04 | loss=0.6038 | val_dice=0.3200 | best=0.2612 (ep31) | 07:24:01 | L_res_0=0.4896 L_res_1=0.5908 L_res_2=0.7388
[2026-06-22 23:05:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 31): 12583.8 MiB
[2026-06-22 23:19:05] INFO segtask_v1.trainer.validation:   Val: loss=1.0360, pooled_mean_dice=0.3059, per_class=['0.3059'], iou=0.1806, recall=1.0000, precision=0.1806, vol_sim=0.3059, mcc=0.0000, min_class_dice=0.3059, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1968, per_class_sd=['0.1968'], combined(w=0.50)=0.2513, balanced=0.2518
[2026-06-22 23:19:05] INFO segtask_v1.trainer.trainer: Epoch 32/400 | LR=8.00e-04 | loss=0.6057 | val_dice=0.3059 | best=0.2612 (ep31) | 07:38:03 | L_res_0=0.4935 L_res_1=0.5931 L_res_2=0.7394
[2026-06-22 23:19:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 32): 12583.8 MiB
[2026-06-22 23:33:11] INFO segtask_v1.trainer.validation:   Val: loss=1.0466, pooled_mean_dice=0.2947, per_class=['0.2947'], iou=0.1728, recall=1.0000, precision=0.1728, vol_sim=0.2947, mcc=0.0000, min_class_dice=0.2947, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2008, per_class_sd=['0.2008'], combined(w=0.50)=0.2477, balanced=0.2475
[2026-06-22 23:33:11] INFO segtask_v1.trainer.trainer: Epoch 33/400 | LR=8.25e-04 | loss=0.6010 | val_dice=0.2947 | best=0.2612 (ep31) | 07:52:09 | L_res_0=0.4877 L_res_1=0.5883 L_res_2=0.7361
[2026-06-22 23:33:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 33): 12583.8 MiB
[2026-06-22 23:47:14] INFO segtask_v1.trainer.validation:   Val: loss=1.0675, pooled_mean_dice=0.2739, per_class=['0.2739'], iou=0.1587, recall=1.0000, precision=0.1587, vol_sim=0.2739, mcc=0.0000, min_class_dice=0.2739, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1921, per_class_sd=['0.1921'], combined(w=0.50)=0.2330, balanced=0.2331
[2026-06-22 23:47:14] INFO segtask_v1.trainer.trainer: Epoch 34/400 | LR=8.50e-04 | loss=0.6007 | val_dice=0.2739 | best=0.2612 (ep31) | 08:06:12 | L_res_0=0.4870 L_res_1=0.5885 L_res_2=0.7351
[2026-06-22 23:47:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 34): 12583.8 MiB
[2026-06-23 00:01:18] INFO segtask_v1.trainer.validation:   Val: loss=1.0379, pooled_mean_dice=0.3034, per_class=['0.3034'], iou=0.1789, recall=1.0000, precision=0.1789, vol_sim=0.3034, mcc=0.0000, min_class_dice=0.3034, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1928, per_class_sd=['0.1928'], combined(w=0.50)=0.2481, balanced=0.2489
[2026-06-23 00:01:18] INFO segtask_v1.trainer.trainer: Epoch 35/400 | LR=8.75e-04 | loss=0.6013 | val_dice=0.3034 | best=0.2612 (ep31) | 08:20:16 | L_res_0=0.4886 L_res_1=0.5883 L_res_2=0.7358
[2026-06-23 00:01:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 35): 12583.8 MiB
[2026-06-23 00:15:20] INFO segtask_v1.trainer.validation:   Val: loss=1.0465, pooled_mean_dice=0.2940, per_class=['0.2940'], iou=0.1723, recall=1.0000, precision=0.1723, vol_sim=0.2940, mcc=0.0000, min_class_dice=0.2940, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1940, per_class_sd=['0.1940'], combined(w=0.50)=0.2440, balanced=0.2445
[2026-06-23 00:15:20] INFO segtask_v1.trainer.trainer: Epoch 36/400 | LR=9.00e-04 | loss=0.5996 | val_dice=0.2940 | best=0.2612 (ep31) | 08:34:17 | L_res_0=0.4859 L_res_1=0.5870 L_res_2=0.7338
[2026-06-23 00:15:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 36): 12583.8 MiB
[2026-06-23 00:29:20] INFO segtask_v1.trainer.validation:   Val: loss=1.0330, pooled_mean_dice=0.3084, per_class=['0.3084'], iou=0.1823, recall=1.0000, precision=0.1823, vol_sim=0.3084, mcc=0.0000, min_class_dice=0.3084, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2020, per_class_sd=['0.2020'], combined(w=0.50)=0.2552, balanced=0.2552
[2026-06-23 00:29:20] INFO segtask_v1.trainer.trainer: Epoch 37/400 | LR=9.25e-04 | loss=0.6034 | val_dice=0.3084 | best=0.2612 (ep31) | 08:48:18 | L_res_0=0.4908 L_res_1=0.5912 L_res_2=0.7376
[2026-06-23 00:29:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 37): 12583.8 MiB
[2026-06-23 00:43:22] INFO segtask_v1.trainer.validation:   Val: loss=1.0501, pooled_mean_dice=0.2906, per_class=['0.2906'], iou=0.1700, recall=1.0000, precision=0.1700, vol_sim=0.2906, mcc=0.0000, min_class_dice=0.2906, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1946, per_class_sd=['0.1946'], combined(w=0.50)=0.2426, balanced=0.2430
[2026-06-23 00:43:22] INFO segtask_v1.trainer.trainer: Epoch 38/400 | LR=9.50e-04 | loss=0.5999 | val_dice=0.2906 | best=0.2612 (ep31) | 09:02:19 | L_res_0=0.4868 L_res_1=0.5872 L_res_2=0.7327
[2026-06-23 00:43:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 38): 12583.8 MiB
[2026-06-23 00:57:23] INFO segtask_v1.trainer.validation:   Val: loss=1.0407, pooled_mean_dice=0.3003, per_class=['0.3003'], iou=0.1767, recall=1.0000, precision=0.1767, vol_sim=0.3003, mcc=0.0000, min_class_dice=0.3003, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1951, per_class_sd=['0.1951'], combined(w=0.50)=0.2477, balanced=0.2482
[2026-06-23 00:57:23] INFO segtask_v1.trainer.trainer: Epoch 39/400 | LR=9.75e-04 | loss=0.6040 | val_dice=0.3003 | best=0.2612 (ep31) | 09:16:20 | L_res_0=0.4923 L_res_1=0.5915 L_res_2=0.7396
[2026-06-23 00:57:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 39): 12583.8 MiB
[2026-06-23 01:11:26] INFO segtask_v1.trainer.validation:   Val: loss=1.0359, pooled_mean_dice=0.3048, per_class=['0.3048'], iou=0.1798, recall=1.0000, precision=0.1798, vol_sim=0.3048, mcc=0.0000, min_class_dice=0.3048, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1964, per_class_sd=['0.1964'], combined(w=0.50)=0.2506, balanced=0.2510
[2026-06-23 01:11:26] INFO segtask_v1.trainer.trainer: Epoch 40/400 | LR=1.00e-03 | loss=0.6005 | val_dice=0.3048 | best=0.2612 (ep31) | 09:30:24 | L_res_0=0.4865 L_res_1=0.5884 L_res_2=0.7347
[2026-06-23 01:11:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 40): 12583.8 MiB
[2026-06-23 01:25:30] INFO segtask_v1.trainer.validation:   Val: loss=1.0477, pooled_mean_dice=0.2928, per_class=['0.2928'], iou=0.1715, recall=1.0000, precision=0.1715, vol_sim=0.2928, mcc=0.0000, min_class_dice=0.2928, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1993, per_class_sd=['0.1993'], combined(w=0.50)=0.2460, balanced=0.2459
[2026-06-23 01:25:30] INFO segtask_v1.trainer.trainer: Epoch 41/400 | LR=1.00e-03 | loss=0.6345 | val_dice=0.2928 | best=0.2612 (ep31) | 09:44:28 | L_res_0=0.5248 L_res_1=0.6292 L_res_2=0.7686
[2026-06-23 01:25:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 41): 12583.8 MiB
[2026-06-23 01:39:30] INFO segtask_v1.trainer.validation:   Val: loss=1.0397, pooled_mean_dice=0.3005, per_class=['0.3005'], iou=0.1768, recall=1.0000, precision=0.1768, vol_sim=0.3005, mcc=0.0000, min_class_dice=0.3005, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1947, per_class_sd=['0.1947'], combined(w=0.50)=0.2476, balanced=0.2481
[2026-06-23 01:39:30] INFO segtask_v1.trainer.trainer: Epoch 42/400 | LR=1.00e-03 | loss=0.6042 | val_dice=0.3005 | best=0.2612 (ep31) | 09:58:28 | L_res_0=0.4883 L_res_1=0.5895 L_res_2=0.7423
[2026-06-23 01:39:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 42): 12583.8 MiB
[2026-06-23 01:53:35] INFO segtask_v1.trainer.validation:   Val: loss=1.0428, pooled_mean_dice=0.2965, per_class=['0.2965'], iou=0.1741, recall=1.0000, precision=0.1741, vol_sim=0.2965, mcc=0.0000, min_class_dice=0.2965, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1960, per_class_sd=['0.1960'], combined(w=0.50)=0.2463, balanced=0.2466
[2026-06-23 01:53:35] INFO segtask_v1.trainer.trainer: Epoch 43/400 | LR=1.00e-03 | loss=0.6037 | val_dice=0.2965 | best=0.2612 (ep31) | 10:12:32 | L_res_0=0.4893 L_res_1=0.5891 L_res_2=0.7399
[2026-06-23 01:53:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 43): 12583.8 MiB
[2026-06-23 02:07:35] INFO segtask_v1.trainer.validation:   Val: loss=1.0473, pooled_mean_dice=0.2919, per_class=['0.2919'], iou=0.1709, recall=1.0000, precision=0.1709, vol_sim=0.2919, mcc=0.0000, min_class_dice=0.2919, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1982, per_class_sd=['0.1982'], combined(w=0.50)=0.2450, balanced=0.2451
[2026-06-23 02:07:35] INFO segtask_v1.trainer.trainer: Epoch 44/400 | LR=1.00e-03 | loss=0.6006 | val_dice=0.2919 | best=0.2612 (ep31) | 10:26:33 | L_res_0=0.4846 L_res_1=0.5865 L_res_2=0.7389
[2026-06-23 02:07:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 44): 12583.8 MiB
[2026-06-23 02:21:37] INFO segtask_v1.trainer.validation:   Val: loss=1.0388, pooled_mean_dice=0.3001, per_class=['0.3001'], iou=0.1766, recall=1.0000, precision=0.1766, vol_sim=0.3001, mcc=0.0000, min_class_dice=0.3001, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2000, per_class_sd=['0.2000'], combined(w=0.50)=0.2500, balanced=0.2501
[2026-06-23 02:21:37] INFO segtask_v1.trainer.trainer: Epoch 45/400 | LR=1.00e-03 | loss=0.6024 | val_dice=0.3001 | best=0.2612 (ep31) | 10:40:35 | L_res_0=0.4869 L_res_1=0.5885 L_res_2=0.7435
[2026-06-23 02:21:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 45): 12583.8 MiB
[2026-06-23 02:35:40] INFO segtask_v1.trainer.validation:   Val: loss=1.0568, pooled_mean_dice=0.2838, per_class=['0.2838'], iou=0.1653, recall=1.0000, precision=0.1653, vol_sim=0.2838, mcc=0.0000, min_class_dice=0.2838, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1969, per_class_sd=['0.1969'], combined(w=0.50)=0.2403, balanced=0.2402
[2026-06-23 02:35:40] INFO segtask_v1.trainer.trainer: Epoch 46/400 | LR=1.00e-03 | loss=0.6034 | val_dice=0.2838 | best=0.2612 (ep31) | 10:54:38 | L_res_0=0.4874 L_res_1=0.5886 L_res_2=0.7448
[2026-06-23 02:35:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 46): 12583.8 MiB
[2026-06-23 02:49:41] INFO segtask_v1.trainer.validation:   Val: loss=1.0394, pooled_mean_dice=0.3021, per_class=['0.3021'], iou=0.1779, recall=1.0000, precision=0.1779, vol_sim=0.3021, mcc=0.0000, min_class_dice=0.3021, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1949, per_class_sd=['0.1949'], combined(w=0.50)=0.2485, balanced=0.2491
[2026-06-23 02:49:41] INFO segtask_v1.trainer.trainer: Epoch 47/400 | LR=1.00e-03 | loss=0.6066 | val_dice=0.3021 | best=0.2612 (ep31) | 11:08:39 | L_res_0=0.4910 L_res_1=0.5924 L_res_2=0.7463
[2026-06-23 02:49:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 47): 12583.8 MiB
[2026-06-23 03:03:45] INFO segtask_v1.trainer.validation:   Val: loss=1.0415, pooled_mean_dice=0.2987, per_class=['0.2987'], iou=0.1756, recall=1.0000, precision=0.1756, vol_sim=0.2987, mcc=0.0000, min_class_dice=0.2987, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2029, per_class_sd=['0.2029'], combined(w=0.50)=0.2508, balanced=0.2504
[2026-06-23 03:03:45] INFO segtask_v1.trainer.trainer: Epoch 48/400 | LR=1.00e-03 | loss=0.6022 | val_dice=0.2987 | best=0.2612 (ep31) | 11:22:43 | L_res_0=0.4858 L_res_1=0.5880 L_res_2=0.7436
[2026-06-23 03:03:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 48): 12583.8 MiB
[2026-06-23 03:17:47] INFO segtask_v1.trainer.validation:   Val: loss=1.0496, pooled_mean_dice=0.2937, per_class=['0.2937'], iou=0.1721, recall=1.0000, precision=0.1721, vol_sim=0.2937, mcc=0.0000, min_class_dice=0.2937, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1939, per_class_sd=['0.1939'], combined(w=0.50)=0.2438, balanced=0.2443
[2026-06-23 03:17:47] INFO segtask_v1.trainer.trainer: Epoch 49/400 | LR=1.00e-03 | loss=0.6015 | val_dice=0.2937 | best=0.2612 (ep31) | 11:36:45 | L_res_0=0.4840 L_res_1=0.5880 L_res_2=0.7427
[2026-06-23 03:17:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 49): 12583.8 MiB
[2026-06-23 03:31:51] INFO segtask_v1.trainer.validation:   Val: loss=1.0564, pooled_mean_dice=0.2868, per_class=['0.2868'], iou=0.1674, recall=1.0000, precision=0.1674, vol_sim=0.2868, mcc=0.0000, min_class_dice=0.2868, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1981, per_class_sd=['0.1981'], combined(w=0.50)=0.2424, balanced=0.2423
[2026-06-23 03:31:51] INFO segtask_v1.trainer.trainer: Epoch 50/400 | LR=1.00e-03 | loss=0.6015 | val_dice=0.2868 | best=0.2612 (ep31) | 11:50:48 | L_res_0=0.4855 L_res_1=0.5876 L_res_2=0.7417
[2026-06-23 03:31:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 50): 12583.8 MiB
[2026-06-23 03:45:52] INFO segtask_v1.trainer.validation:   Val: loss=1.0459, pooled_mean_dice=0.2976, per_class=['0.2976'], iou=0.1748, recall=1.0000, precision=0.1748, vol_sim=0.2976, mcc=0.0000, min_class_dice=0.2976, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1904, per_class_sd=['0.1904'], combined(w=0.50)=0.2440, balanced=0.2449
[2026-06-23 03:45:52] INFO segtask_v1.trainer.trainer: Epoch 51/400 | LR=1.00e-03 | loss=0.6035 | val_dice=0.2976 | best=0.2612 (ep31) | 12:04:50 | L_res_0=0.4876 L_res_1=0.5898 L_res_2=0.7445
[2026-06-23 03:45:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 51): 12583.8 MiB
[2026-06-23 03:59:54] INFO segtask_v1.trainer.validation:   Val: loss=1.0455, pooled_mean_dice=0.2954, per_class=['0.2954'], iou=0.1733, recall=1.0000, precision=0.1733, vol_sim=0.2954, mcc=0.0000, min_class_dice=0.2954, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1935, per_class_sd=['0.1935'], combined(w=0.50)=0.2444, balanced=0.2450
[2026-06-23 03:59:54] INFO segtask_v1.trainer.trainer: Epoch 52/400 | LR=1.00e-03 | loss=0.5998 | val_dice=0.2954 | best=0.2612 (ep31) | 12:18:52 | L_res_0=0.4840 L_res_1=0.5856 L_res_2=0.7400
[2026-06-23 03:59:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 52): 12583.8 MiB
[2026-06-23 04:13:57] INFO segtask_v1.trainer.validation:   Val: loss=1.0506, pooled_mean_dice=0.2902, per_class=['0.2902'], iou=0.1697, recall=1.0000, precision=0.1697, vol_sim=0.2902, mcc=0.0000, min_class_dice=0.2902, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1909, per_class_sd=['0.1909'], combined(w=0.50)=0.2405, balanced=0.2413
[2026-06-23 04:13:57] INFO segtask_v1.trainer.trainer: Epoch 53/400 | LR=1.00e-03 | loss=0.6032 | val_dice=0.2902 | best=0.2612 (ep31) | 12:32:55 | L_res_0=0.4877 L_res_1=0.5899 L_res_2=0.7452
[2026-06-23 04:13:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 53): 12583.8 MiB
[2026-06-23 04:28:00] INFO segtask_v1.trainer.validation:   Val: loss=1.0505, pooled_mean_dice=0.2891, per_class=['0.2891'], iou=0.1690, recall=1.0000, precision=0.1690, vol_sim=0.2891, mcc=0.0000, min_class_dice=0.2891, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1888, per_class_sd=['0.1888'], combined(w=0.50)=0.2389, balanced=0.2399
[2026-06-23 04:28:00] INFO segtask_v1.trainer.trainer: Epoch 54/400 | LR=1.00e-03 | loss=0.6035 | val_dice=0.2891 | best=0.2612 (ep31) | 12:46:58 | L_res_0=0.4880 L_res_1=0.5900 L_res_2=0.7440
[2026-06-23 04:28:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 54): 12583.8 MiB
[2026-06-23 04:42:04] INFO segtask_v1.trainer.validation:   Val: loss=1.0264, pooled_mean_dice=0.3139, per_class=['0.3139'], iou=0.1862, recall=1.0000, precision=0.1862, vol_sim=0.3139, mcc=0.0000, min_class_dice=0.3139, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2017, per_class_sd=['0.2017'], combined(w=0.50)=0.2578, balanced=0.2578
[2026-06-23 04:42:04] INFO segtask_v1.trainer.trainer: Epoch 55/400 | LR=1.00e-03 | loss=0.6024 | val_dice=0.3139 | best=0.2612 (ep31) | 13:01:02 | L_res_0=0.4863 L_res_1=0.5886 L_res_2=0.7443
[2026-06-23 04:42:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 55): 12583.8 MiB
[2026-06-23 04:56:05] INFO segtask_v1.trainer.validation:   Val: loss=1.0480, pooled_mean_dice=0.2910, per_class=['0.2910'], iou=0.1703, recall=1.0000, precision=0.1703, vol_sim=0.2910, mcc=0.0000, min_class_dice=0.2910, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1940, per_class_sd=['0.1940'], combined(w=0.50)=0.2425, balanced=0.2429
[2026-06-23 04:56:05] INFO segtask_v1.trainer.trainer: Epoch 56/400 | LR=1.00e-03 | loss=0.5997 | val_dice=0.2910 | best=0.2612 (ep31) | 13:15:03 | L_res_0=0.4857 L_res_1=0.5855 L_res_2=0.7397
[2026-06-23 04:56:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 56): 12583.8 MiB
[2026-06-23 05:10:09] INFO segtask_v1.trainer.validation:   Val: loss=1.0459, pooled_mean_dice=0.2927, per_class=['0.2927'], iou=0.1714, recall=1.0000, precision=0.1714, vol_sim=0.2927, mcc=0.0000, min_class_dice=0.2927, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1963, per_class_sd=['0.1963'], combined(w=0.50)=0.2445, balanced=0.2448
[2026-06-23 05:10:09] INFO segtask_v1.trainer.trainer: Epoch 57/400 | LR=1.00e-03 | loss=0.5998 | val_dice=0.2927 | best=0.2612 (ep31) | 13:29:06 | L_res_0=0.4852 L_res_1=0.5861 L_res_2=0.7412
[2026-06-23 05:10:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 57): 12583.8 MiB
[2026-06-23 05:24:09] INFO segtask_v1.trainer.validation:   Val: loss=1.0366, pooled_mean_dice=0.3038, per_class=['0.3038'], iou=0.1791, recall=1.0000, precision=0.1791, vol_sim=0.3038, mcc=0.0000, min_class_dice=0.3038, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1961, per_class_sd=['0.1961'], combined(w=0.50)=0.2499, balanced=0.2504
[2026-06-23 05:24:09] INFO segtask_v1.trainer.trainer: Epoch 58/400 | LR=1.00e-03 | loss=0.6023 | val_dice=0.3038 | best=0.2612 (ep31) | 13:43:07 | L_res_0=0.4862 L_res_1=0.5882 L_res_2=0.7459
[2026-06-23 05:24:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 58): 12583.8 MiB
[2026-06-23 05:38:14] INFO segtask_v1.trainer.validation:   Val: loss=1.0488, pooled_mean_dice=0.2912, per_class=['0.2912'], iou=0.1704, recall=1.0000, precision=0.1704, vol_sim=0.2912, mcc=0.0000, min_class_dice=0.2912, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1922, per_class_sd=['0.1922'], combined(w=0.50)=0.2417, balanced=0.2424
[2026-06-23 05:38:14] INFO segtask_v1.trainer.trainer: Epoch 59/400 | LR=1.00e-03 | loss=0.6056 | val_dice=0.2912 | best=0.2612 (ep31) | 13:57:12 | L_res_0=0.4911 L_res_1=0.5915 L_res_2=0.7481
[2026-06-23 05:38:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 59): 12583.8 MiB
[2026-06-23 05:52:15] INFO segtask_v1.trainer.validation:   Val: loss=1.0448, pooled_mean_dice=0.2943, per_class=['0.2943'], iou=0.1726, recall=1.0000, precision=0.1726, vol_sim=0.2943, mcc=0.0000, min_class_dice=0.2943, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1954, per_class_sd=['0.1954'], combined(w=0.50)=0.2449, balanced=0.2453
[2026-06-23 05:52:15] INFO segtask_v1.trainer.trainer: Epoch 60/400 | LR=1.00e-03 | loss=0.6024 | val_dice=0.2943 | best=0.2612 (ep31) | 14:11:13 | L_res_0=0.4871 L_res_1=0.5888 L_res_2=0.7457
[2026-06-23 05:52:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 60): 12583.8 MiB
[2026-06-23 06:06:21] INFO segtask_v1.trainer.validation:   Val: loss=1.0638, pooled_mean_dice=0.2764, per_class=['0.2764'], iou=0.1604, recall=1.0000, precision=0.1604, vol_sim=0.2764, mcc=0.0000, min_class_dice=0.2764, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1965, per_class_sd=['0.1965'], combined(w=0.50)=0.2364, balanced=0.2361
[2026-06-23 06:06:21] INFO segtask_v1.trainer.trainer: Epoch 61/400 | LR=1.00e-03 | loss=0.6014 | val_dice=0.2764 | best=0.2612 (ep31) | 14:25:19 | L_res_0=0.4868 L_res_1=0.5879 L_res_2=0.7424
[2026-06-23 06:06:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 61): 12583.8 MiB
[2026-06-23 06:20:27] INFO segtask_v1.trainer.validation:   Val: loss=1.0408, pooled_mean_dice=0.2988, per_class=['0.2988'], iou=0.1756, recall=1.0000, precision=0.1756, vol_sim=0.2988, mcc=0.0000, min_class_dice=0.2988, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2009, per_class_sd=['0.2009'], combined(w=0.50)=0.2499, balanced=0.2497
[2026-06-23 06:20:27] INFO segtask_v1.trainer.trainer: Epoch 62/400 | LR=1.00e-03 | loss=0.6005 | val_dice=0.2988 | best=0.2612 (ep31) | 14:39:25 | L_res_0=0.4859 L_res_1=0.5870 L_res_2=0.7441
[2026-06-23 06:20:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 62): 12583.8 MiB
[2026-06-23 06:34:30] INFO segtask_v1.trainer.validation:   Val: loss=1.0481, pooled_mean_dice=0.2914, per_class=['0.2914'], iou=0.1705, recall=1.0000, precision=0.1705, vol_sim=0.2914, mcc=0.0000, min_class_dice=0.2914, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1967, per_class_sd=['0.1967'], combined(w=0.50)=0.2440, balanced=0.2442
[2026-06-23 06:34:30] INFO segtask_v1.trainer.trainer: Epoch 63/400 | LR=1.00e-03 | loss=0.6009 | val_dice=0.2914 | best=0.2612 (ep31) | 14:53:28 | L_res_0=0.4861 L_res_1=0.5876 L_res_2=0.7434
[2026-06-23 06:34:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 63): 12583.8 MiB
[2026-06-23 06:48:34] INFO segtask_v1.trainer.validation:   Val: loss=1.0538, pooled_mean_dice=0.2854, per_class=['0.2854'], iou=0.1664, recall=1.0000, precision=0.1664, vol_sim=0.2854, mcc=0.0000, min_class_dice=0.2854, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1956, per_class_sd=['0.1956'], combined(w=0.50)=0.2405, balanced=0.2406
[2026-06-23 06:48:34] INFO segtask_v1.trainer.trainer: Epoch 64/400 | LR=1.00e-03 | loss=0.6026 | val_dice=0.2854 | best=0.2612 (ep31) | 15:07:32 | L_res_0=0.4883 L_res_1=0.5890 L_res_2=0.7447
[2026-06-23 06:48:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 64): 12583.8 MiB
[2026-06-23 07:02:41] INFO segtask_v1.trainer.validation:   Val: loss=1.0436, pooled_mean_dice=0.2964, per_class=['0.2964'], iou=0.1740, recall=1.0000, precision=0.1740, vol_sim=0.2964, mcc=0.0000, min_class_dice=0.2964, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1968, per_class_sd=['0.1968'], combined(w=0.50)=0.2466, balanced=0.2469
[2026-06-23 07:02:41] INFO segtask_v1.trainer.trainer: Epoch 65/400 | LR=1.00e-03 | loss=0.6022 | val_dice=0.2964 | best=0.2612 (ep31) | 15:21:39 | L_res_0=0.4857 L_res_1=0.5887 L_res_2=0.7461
[2026-06-23 07:02:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 65): 12583.8 MiB
[2026-06-23 07:16:43] INFO segtask_v1.trainer.validation:   Val: loss=1.0416, pooled_mean_dice=0.2975, per_class=['0.2975'], iou=0.1747, recall=1.0000, precision=0.1747, vol_sim=0.2975, mcc=0.0000, min_class_dice=0.2975, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1982, per_class_sd=['0.1982'], combined(w=0.50)=0.2478, balanced=0.2480
[2026-06-23 07:16:43] INFO segtask_v1.trainer.trainer: Epoch 66/400 | LR=1.00e-03 | loss=0.5991 | val_dice=0.2975 | best=0.2612 (ep31) | 15:35:41 | L_res_0=0.4831 L_res_1=0.5858 L_res_2=0.7425
[2026-06-23 07:16:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 66): 12583.8 MiB
[2026-06-23 07:30:48] INFO segtask_v1.trainer.validation:   Val: loss=1.0281, pooled_mean_dice=0.3103, per_class=['0.3103'], iou=0.1836, recall=1.0000, precision=0.1836, vol_sim=0.3103, mcc=0.0000, min_class_dice=0.3103, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1965, per_class_sd=['0.1965'], combined(w=0.50)=0.2534, balanced=0.2538
[2026-06-23 07:30:48] INFO segtask_v1.trainer.trainer: Epoch 67/400 | LR=1.00e-03 | loss=0.5996 | val_dice=0.3103 | best=0.2612 (ep31) | 15:49:46 | L_res_0=0.4833 L_res_1=0.5863 L_res_2=0.7424
[2026-06-23 07:30:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 67): 12583.8 MiB
[2026-06-23 07:44:53] INFO segtask_v1.trainer.validation:   Val: loss=1.0548, pooled_mean_dice=0.2850, per_class=['0.2850'], iou=0.1662, recall=1.0000, precision=0.1662, vol_sim=0.2850, mcc=0.0000, min_class_dice=0.2850, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1885, per_class_sd=['0.1885'], combined(w=0.50)=0.2367, balanced=0.2377
[2026-06-23 07:44:53] INFO segtask_v1.trainer.trainer: Epoch 68/400 | LR=1.00e-03 | loss=0.5998 | val_dice=0.2850 | best=0.2612 (ep31) | 16:03:51 | L_res_0=0.4863 L_res_1=0.5863 L_res_2=0.7405
[2026-06-23 07:44:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 68): 12583.8 MiB
[2026-06-23 07:58:59] INFO segtask_v1.trainer.validation:   Val: loss=1.0430, pooled_mean_dice=0.2957, per_class=['0.2957'], iou=0.1735, recall=1.0000, precision=0.1735, vol_sim=0.2957, mcc=0.0000, min_class_dice=0.2957, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1942, per_class_sd=['0.1942'], combined(w=0.50)=0.2449, balanced=0.2455
[2026-06-23 07:58:59] INFO segtask_v1.trainer.trainer: Epoch 69/400 | LR=1.00e-03 | loss=0.6016 | val_dice=0.2957 | best=0.2612 (ep31) | 16:17:57 | L_res_0=0.4867 L_res_1=0.5878 L_res_2=0.7435
[2026-06-23 07:58:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 69): 12583.8 MiB
[2026-06-23 08:13:03] INFO segtask_v1.trainer.validation:   Val: loss=1.0411, pooled_mean_dice=0.3001, per_class=['0.3001'], iou=0.1766, recall=1.0000, precision=0.1766, vol_sim=0.3001, mcc=0.0000, min_class_dice=0.3001, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1924, per_class_sd=['0.1924'], combined(w=0.50)=0.2463, balanced=0.2470
[2026-06-23 08:13:03] INFO segtask_v1.trainer.trainer: Epoch 70/400 | LR=1.00e-03 | loss=0.5997 | val_dice=0.3001 | best=0.2612 (ep31) | 16:32:01 | L_res_0=0.4843 L_res_1=0.5870 L_res_2=0.7444
[2026-06-23 08:13:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 70): 12583.8 MiB
[2026-06-23 08:27:08] INFO segtask_v1.trainer.validation:   Val: loss=1.0374, pooled_mean_dice=0.3013, per_class=['0.3013'], iou=0.1774, recall=1.0000, precision=0.1774, vol_sim=0.3013, mcc=0.0000, min_class_dice=0.3013, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1936, per_class_sd=['0.1936'], combined(w=0.50)=0.2475, balanced=0.2481
[2026-06-23 08:27:08] INFO segtask_v1.trainer.trainer: Epoch 71/400 | LR=1.00e-03 | loss=0.6010 | val_dice=0.3013 | best=0.2612 (ep31) | 16:46:06 | L_res_0=0.4859 L_res_1=0.5884 L_res_2=0.7434
[2026-06-23 08:27:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 71): 12583.8 MiB
[2026-06-23 08:41:12] INFO segtask_v1.trainer.validation:   Val: loss=1.0439, pooled_mean_dice=0.2958, per_class=['0.2958'], iou=0.1736, recall=1.0000, precision=0.1736, vol_sim=0.2958, mcc=0.0000, min_class_dice=0.2958, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1964, per_class_sd=['0.1964'], combined(w=0.50)=0.2461, balanced=0.2464
[2026-06-23 08:41:12] INFO segtask_v1.trainer.trainer: Epoch 72/400 | LR=1.00e-03 | loss=0.6025 | val_dice=0.2958 | best=0.2612 (ep31) | 17:00:10 | L_res_0=0.4885 L_res_1=0.5894 L_res_2=0.7457
[2026-06-23 08:41:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 72): 12583.8 MiB
[2026-06-23 08:55:19] INFO segtask_v1.trainer.validation:   Val: loss=1.0468, pooled_mean_dice=0.2927, per_class=['0.2927'], iou=0.1715, recall=1.0000, precision=0.1715, vol_sim=0.2927, mcc=0.0000, min_class_dice=0.2927, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1939, per_class_sd=['0.1939'], combined(w=0.50)=0.2433, balanced=0.2438
[2026-06-23 08:55:19] INFO segtask_v1.trainer.trainer: Epoch 73/400 | LR=1.00e-03 | loss=0.6001 | val_dice=0.2927 | best=0.2612 (ep31) | 17:14:17 | L_res_0=0.4863 L_res_1=0.5870 L_res_2=0.7421
[2026-06-23 08:55:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 73): 12583.8 MiB
[2026-06-23 09:09:20] INFO segtask_v1.trainer.validation:   Val: loss=1.0495, pooled_mean_dice=0.2904, per_class=['0.2904'], iou=0.1699, recall=1.0000, precision=0.1699, vol_sim=0.2904, mcc=0.0000, min_class_dice=0.2904, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1996, per_class_sd=['0.1996'], combined(w=0.50)=0.2450, balanced=0.2448
[2026-06-23 09:09:20] INFO segtask_v1.trainer.trainer: Epoch 74/400 | LR=1.00e-03 | loss=0.6024 | val_dice=0.2904 | best=0.2612 (ep31) | 17:28:18 | L_res_0=0.4883 L_res_1=0.5893 L_res_2=0.7454
[2026-06-23 09:09:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 74): 12583.8 MiB
[2026-06-23 09:23:24] INFO segtask_v1.trainer.validation:   Val: loss=1.0389, pooled_mean_dice=0.3004, per_class=['0.3004'], iou=0.1767, recall=1.0000, precision=0.1767, vol_sim=0.3004, mcc=0.0000, min_class_dice=0.3004, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1952, per_class_sd=['0.1952'], combined(w=0.50)=0.2478, balanced=0.2483
[2026-06-23 09:23:24] INFO segtask_v1.trainer.trainer: Epoch 75/400 | LR=1.00e-03 | loss=0.5995 | val_dice=0.3004 | best=0.2612 (ep31) | 17:42:22 | L_res_0=0.4835 L_res_1=0.5863 L_res_2=0.7427
[2026-06-23 09:23:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 75): 12583.8 MiB
[2026-06-23 09:37:32] INFO segtask_v1.trainer.validation:   Val: loss=1.0364, pooled_mean_dice=0.3025, per_class=['0.3025'], iou=0.1782, recall=1.0000, precision=0.1782, vol_sim=0.3025, mcc=0.0000, min_class_dice=0.3025, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1977, per_class_sd=['0.1977'], combined(w=0.50)=0.2501, balanced=0.2504
[2026-06-23 09:37:32] INFO segtask_v1.trainer.trainer: Epoch 76/400 | LR=1.00e-03 | loss=0.5997 | val_dice=0.3025 | best=0.2612 (ep31) | 17:56:30 | L_res_0=0.4856 L_res_1=0.5865 L_res_2=0.7422
[2026-06-23 09:37:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 76): 12583.8 MiB
[2026-06-23 09:51:35] INFO segtask_v1.trainer.validation:   Val: loss=1.0317, pooled_mean_dice=0.3075, per_class=['0.3075'], iou=0.1817, recall=1.0000, precision=0.1817, vol_sim=0.3075, mcc=0.0000, min_class_dice=0.3075, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1986, per_class_sd=['0.1986'], combined(w=0.50)=0.2530, balanced=0.2533
[2026-06-23 09:51:35] INFO segtask_v1.trainer.trainer: Epoch 77/400 | LR=1.00e-03 | loss=0.6017 | val_dice=0.3075 | best=0.2612 (ep31) | 18:10:32 | L_res_0=0.4868 L_res_1=0.5878 L_res_2=0.7444
[2026-06-23 09:51:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 77): 12583.8 MiB
[2026-06-23 10:05:40] INFO segtask_v1.trainer.validation:   Val: loss=1.0467, pooled_mean_dice=0.2937, per_class=['0.2937'], iou=0.1721, recall=1.0000, precision=0.1721, vol_sim=0.2937, mcc=0.0000, min_class_dice=0.2937, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1924, per_class_sd=['0.1924'], combined(w=0.50)=0.2430, balanced=0.2437
[2026-06-23 10:05:40] INFO segtask_v1.trainer.trainer: Epoch 78/400 | LR=1.00e-03 | loss=0.6004 | val_dice=0.2937 | best=0.2612 (ep31) | 18:24:38 | L_res_0=0.4856 L_res_1=0.5879 L_res_2=0.7429
[2026-06-23 10:05:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 78): 12583.8 MiB
[2026-06-23 10:19:44] INFO segtask_v1.trainer.validation:   Val: loss=1.0362, pooled_mean_dice=0.3028, per_class=['0.3028'], iou=0.1784, recall=1.0000, precision=0.1784, vol_sim=0.3028, mcc=0.0000, min_class_dice=0.3028, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1942, per_class_sd=['0.1942'], combined(w=0.50)=0.2485, balanced=0.2491
[2026-06-23 10:19:44] INFO segtask_v1.trainer.trainer: Epoch 79/400 | LR=1.00e-03 | loss=0.6021 | val_dice=0.3028 | best=0.2612 (ep31) | 18:38:41 | L_res_0=0.4875 L_res_1=0.5888 L_res_2=0.7439
[2026-06-23 10:19:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 79): 12583.8 MiB
[2026-06-23 10:33:47] INFO segtask_v1.trainer.validation:   Val: loss=1.0480, pooled_mean_dice=0.2918, per_class=['0.2918'], iou=0.1708, recall=1.0000, precision=0.1708, vol_sim=0.2918, mcc=0.0000, min_class_dice=0.2918, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1949, per_class_sd=['0.1949'], combined(w=0.50)=0.2433, balanced=0.2437
[2026-06-23 10:33:47] INFO segtask_v1.trainer.trainer: Epoch 80/400 | LR=1.00e-03 | loss=0.6022 | val_dice=0.2918 | best=0.2612 (ep31) | 18:52:45 | L_res_0=0.4879 L_res_1=0.5889 L_res_2=0.7449
[2026-06-23 10:33:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 80): 12583.8 MiB
[2026-06-23 10:47:53] INFO segtask_v1.trainer.validation:   Val: loss=1.0423, pooled_mean_dice=0.2985, per_class=['0.2985'], iou=0.1754, recall=1.0000, precision=0.1754, vol_sim=0.2985, mcc=0.0000, min_class_dice=0.2985, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1931, per_class_sd=['0.1931'], combined(w=0.50)=0.2458, balanced=0.2465
[2026-06-23 10:47:53] INFO segtask_v1.trainer.trainer: Epoch 81/400 | LR=1.00e-03 | loss=0.6022 | val_dice=0.2985 | best=0.2612 (ep31) | 19:06:50 | L_res_0=0.4876 L_res_1=0.5891 L_res_2=0.7445
[2026-06-23 10:47:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 81): 12583.8 MiB
[2026-06-23 11:01:58] INFO segtask_v1.trainer.validation:   Val: loss=1.0420, pooled_mean_dice=0.2973, per_class=['0.2973'], iou=0.1746, recall=1.0000, precision=0.1746, vol_sim=0.2973, mcc=0.0000, min_class_dice=0.2973, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1987, per_class_sd=['0.1987'], combined(w=0.50)=0.2480, balanced=0.2481
[2026-06-23 11:01:58] INFO segtask_v1.trainer.trainer: Epoch 82/400 | LR=1.00e-03 | loss=0.6023 | val_dice=0.2973 | best=0.2612 (ep31) | 19:20:56 | L_res_0=0.4882 L_res_1=0.5893 L_res_2=0.7453
[2026-06-23 11:01:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 82): 12583.8 MiB
[2026-06-23 11:16:01] INFO segtask_v1.trainer.validation:   Val: loss=1.0569, pooled_mean_dice=0.2828, per_class=['0.2828'], iou=0.1647, recall=1.0000, precision=0.1647, vol_sim=0.2828, mcc=0.0000, min_class_dice=0.2828, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1941, per_class_sd=['0.1941'], combined(w=0.50)=0.2384, balanced=0.2386
[2026-06-23 11:16:01] INFO segtask_v1.trainer.trainer: Epoch 83/400 | LR=1.00e-03 | loss=0.6015 | val_dice=0.2828 | best=0.2612 (ep31) | 19:34:59 | L_res_0=0.4864 L_res_1=0.5890 L_res_2=0.7439
[2026-06-23 11:16:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 83): 12583.8 MiB
[2026-06-23 11:30:05] INFO segtask_v1.trainer.validation:   Val: loss=1.0342, pooled_mean_dice=0.3043, per_class=['0.3043'], iou=0.1794, recall=1.0000, precision=0.1794, vol_sim=0.3043, mcc=0.0000, min_class_dice=0.3043, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1953, per_class_sd=['0.1953'], combined(w=0.50)=0.2498, balanced=0.2503
[2026-06-23 11:30:05] INFO segtask_v1.trainer.trainer: Epoch 84/400 | LR=1.00e-03 | loss=0.6027 | val_dice=0.3043 | best=0.2612 (ep31) | 19:49:03 | L_res_0=0.4888 L_res_1=0.5900 L_res_2=0.7464
[2026-06-23 11:30:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 84): 12583.8 MiB
[2026-06-23 11:44:06] INFO segtask_v1.trainer.validation:   Val: loss=1.0329, pooled_mean_dice=0.3059, per_class=['0.3059'], iou=0.1805, recall=1.0000, precision=0.1805, vol_sim=0.3059, mcc=0.0000, min_class_dice=0.3059, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1939, per_class_sd=['0.1939'], combined(w=0.50)=0.2499, balanced=0.2505
[2026-06-23 11:44:06] INFO segtask_v1.trainer.trainer: Epoch 85/400 | LR=1.00e-03 | loss=0.6019 | val_dice=0.3059 | best=0.2612 (ep31) | 20:03:04 | L_res_0=0.4870 L_res_1=0.5887 L_res_2=0.7455
[2026-06-23 11:44:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 85): 12583.8 MiB
[2026-06-23 11:58:11] INFO segtask_v1.trainer.validation:   Val: loss=1.0506, pooled_mean_dice=0.2894, per_class=['0.2894'], iou=0.1692, recall=1.0000, precision=0.1692, vol_sim=0.2894, mcc=0.0000, min_class_dice=0.2894, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1960, per_class_sd=['0.1960'], combined(w=0.50)=0.2427, balanced=0.2429
[2026-06-23 11:58:11] INFO segtask_v1.trainer.trainer: Epoch 86/400 | LR=9.99e-04 | loss=0.6007 | val_dice=0.2894 | best=0.2612 (ep31) | 20:17:09 | L_res_0=0.4874 L_res_1=0.5878 L_res_2=0.7432
[2026-06-23 11:58:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 86): 12583.8 MiB
[2026-06-23 12:12:14] INFO segtask_v1.trainer.validation:   Val: loss=1.0394, pooled_mean_dice=0.3002, per_class=['0.3002'], iou=0.1766, recall=1.0000, precision=0.1766, vol_sim=0.3002, mcc=0.0000, min_class_dice=0.3002, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1966, per_class_sd=['0.1966'], combined(w=0.50)=0.2484, balanced=0.2488
[2026-06-23 12:12:14] INFO segtask_v1.trainer.trainer: Epoch 87/400 | LR=9.99e-04 | loss=0.5981 | val_dice=0.3002 | best=0.2612 (ep31) | 20:31:11 | L_res_0=0.4834 L_res_1=0.5854 L_res_2=0.7416
[2026-06-23 12:12:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 87): 12583.8 MiB
[2026-06-23 12:26:19] INFO segtask_v1.trainer.validation:   Val: loss=1.0354, pooled_mean_dice=0.3029, per_class=['0.3029'], iou=0.1785, recall=1.0000, precision=0.1785, vol_sim=0.3029, mcc=0.0000, min_class_dice=0.3029, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.2026, per_class_sd=['0.2026'], combined(w=0.50)=0.2527, balanced=0.2525
[2026-06-23 12:26:19] INFO segtask_v1.trainer.trainer: Epoch 88/400 | LR=9.99e-04 | loss=0.5991 | val_dice=0.3029 | best=0.2612 (ep31) | 20:45:17 | L_res_0=0.4845 L_res_1=0.5864 L_res_2=0.7418
[2026-06-23 12:26:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 88): 12583.8 MiB
[2026-06-23 12:40:22] INFO segtask_v1.trainer.validation:   Val: loss=1.0365, pooled_mean_dice=0.3017, per_class=['0.3017'], iou=0.1777, recall=1.0000, precision=0.1777, vol_sim=0.3017, mcc=0.0000, min_class_dice=0.3017, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1997, per_class_sd=['0.1997'], combined(w=0.50)=0.2507, balanced=0.2508
[2026-06-23 12:40:22] INFO segtask_v1.trainer.trainer: Epoch 89/400 | LR=9.99e-04 | loss=0.6023 | val_dice=0.3017 | best=0.2612 (ep31) | 20:59:20 | L_res_0=0.4893 L_res_1=0.5887 L_res_2=0.7454
[2026-06-23 12:40:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 89): 12583.8 MiB
[2026-06-23 12:54:24] INFO segtask_v1.trainer.validation:   Val: loss=1.0470, pooled_mean_dice=0.2907, per_class=['0.2907'], iou=0.1701, recall=1.0000, precision=0.1701, vol_sim=0.2907, mcc=0.0000, min_class_dice=0.2907, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1949, per_class_sd=['0.1949'], combined(w=0.50)=0.2428, balanced=0.2432
[2026-06-23 12:54:24] INFO segtask_v1.trainer.trainer: Epoch 90/400 | LR=9.99e-04 | loss=0.5975 | val_dice=0.2907 | best=0.2612 (ep31) | 21:13:22 | L_res_0=0.4840 L_res_1=0.5851 L_res_2=0.7400
[2026-06-23 12:54:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 90): 12583.8 MiB
[2026-06-23 13:08:32] INFO segtask_v1.trainer.validation:   Val: loss=1.0494, pooled_mean_dice=0.2894, per_class=['0.2894'], iou=0.1692, recall=1.0000, precision=0.1692, vol_sim=0.2894, mcc=0.0000, min_class_dice=0.2894, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1970, per_class_sd=['0.1970'], combined(w=0.50)=0.2432, balanced=0.2433
[2026-06-23 13:08:32] INFO segtask_v1.trainer.trainer: Epoch 91/400 | LR=9.99e-04 | loss=0.6000 | val_dice=0.2894 | best=0.2612 (ep31) | 21:27:30 | L_res_0=0.4857 L_res_1=0.5878 L_res_2=0.7411
[2026-06-23 13:08:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 91): 12583.8 MiB
[2026-06-23 13:22:47] INFO segtask_v1.trainer.validation:   Val: loss=1.0312, pooled_mean_dice=0.3074, per_class=['0.3074'], iou=0.1816, recall=1.0000, precision=0.1816, vol_sim=0.3074, mcc=0.0000, min_class_dice=0.3074, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1946, per_class_sd=['0.1946'], combined(w=0.50)=0.2510, balanced=0.2516
[2026-06-23 13:22:47] INFO segtask_v1.trainer.trainer: Epoch 92/400 | LR=9.99e-04 | loss=0.6006 | val_dice=0.3074 | best=0.2612 (ep31) | 21:41:45 | L_res_0=0.4861 L_res_1=0.5876 L_res_2=0.7433
[2026-06-23 13:22:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 92): 12583.8 MiB
[2026-06-23 13:37:04] INFO segtask_v1.trainer.validation:   Val: loss=1.0397, pooled_mean_dice=0.2998, per_class=['0.2998'], iou=0.1763, recall=1.0000, precision=0.1763, vol_sim=0.2998, mcc=0.0000, min_class_dice=0.2998, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1948, per_class_sd=['0.1948'], combined(w=0.50)=0.2473, balanced=0.2478
[2026-06-23 13:37:04] INFO segtask_v1.trainer.trainer: Epoch 93/400 | LR=9.99e-04 | loss=0.6016 | val_dice=0.2998 | best=0.2612 (ep31) | 21:56:02 | L_res_0=0.4871 L_res_1=0.5888 L_res_2=0.7447
[2026-06-23 13:37:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 93): 12583.8 MiB
[2026-06-23 13:51:10] INFO segtask_v1.trainer.validation:   Val: loss=1.0354, pooled_mean_dice=0.3036, per_class=['0.3036'], iou=0.1790, recall=1.0000, precision=0.1790, vol_sim=0.3036, mcc=0.0000, min_class_dice=0.3036, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1935, per_class_sd=['0.1935'], combined(w=0.50)=0.2486, balanced=0.2492
[2026-06-23 13:51:10] INFO segtask_v1.trainer.trainer: Epoch 94/400 | LR=9.99e-04 | loss=0.6266 | val_dice=0.3036 | best=0.2612 (ep31) | 22:10:07 | L_res_0=0.5744 L_res_1=0.7016 L_res_2=0.7734
[2026-06-23 13:51:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 94): 12583.8 MiB
[2026-06-23 14:05:15] INFO segtask_v1.trainer.validation:   Val: loss=1.0392, pooled_mean_dice=0.3000, per_class=['0.3000'], iou=0.1764, recall=1.0000, precision=0.1764, vol_sim=0.3000, mcc=0.0000, min_class_dice=0.3000, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1979, per_class_sd=['0.1979'], combined(w=0.50)=0.2489, balanced=0.2492
[2026-06-23 14:05:15] INFO segtask_v1.trainer.trainer: Epoch 95/400 | LR=9.99e-04 | loss=0.6259 | val_dice=0.3000 | best=0.2612 (ep31) | 22:24:13 | L_res_0=0.5829 L_res_1=0.7119 L_res_2=0.7616
[2026-06-23 14:05:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 95): 12583.8 MiB
[2026-06-23 14:19:32] INFO segtask_v1.trainer.validation:   Val: loss=1.0532, pooled_mean_dice=0.2865, per_class=['0.2865'], iou=0.1672, recall=1.0000, precision=0.1672, vol_sim=0.2865, mcc=0.0000, min_class_dice=0.2865, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1941, per_class_sd=['0.1941'], combined(w=0.50)=0.2403, balanced=0.2406
[2026-06-23 14:19:32] INFO segtask_v1.trainer.trainer: Epoch 96/400 | LR=9.99e-04 | loss=0.6221 | val_dice=0.2865 | best=0.2612 (ep31) | 22:38:30 | L_res_0=0.5809 L_res_1=0.7087 L_res_2=0.7545
[2026-06-23 14:19:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 96): 12583.8 MiB
[2026-06-23 14:33:47] INFO segtask_v1.trainer.validation:   Val: loss=1.0324, pooled_mean_dice=0.3073, per_class=['0.3073'], iou=0.1815, recall=1.0000, precision=0.1815, vol_sim=0.3073, mcc=0.0000, min_class_dice=0.3073, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1933, per_class_sd=['0.1933'], combined(w=0.50)=0.2503, balanced=0.2510
[2026-06-23 14:33:47] INFO segtask_v1.trainer.trainer: Epoch 97/400 | LR=9.99e-04 | loss=0.6241 | val_dice=0.3073 | best=0.2612 (ep31) | 22:52:45 | L_res_0=0.5798 L_res_1=0.7089 L_res_2=0.7606
[2026-06-23 14:33:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 97): 12583.8 MiB
[2026-06-23 14:47:53] INFO segtask_v1.trainer.validation:   Val: loss=1.0499, pooled_mean_dice=0.2894, per_class=['0.2894'], iou=0.1692, recall=1.0000, precision=0.1692, vol_sim=0.2894, mcc=0.0000, min_class_dice=0.2894, coverage=[88]/88 samples, pooled_mean_surface_dice@2px=0.1977, per_class_sd=['0.1977'], combined(w=0.50)=0.2436, balanced=0.2436
[2026-06-23 14:47:53] INFO segtask_v1.trainer.trainer: Epoch 98/400 | LR=9.99e-04 | loss=0.6193 | val_dice=0.2894 | best=0.2612 (ep31) | 23:06:51 | L_res_0=0.5637 L_res_1=0.6807 L_res_2=0.7594
[2026-06-23 14:47:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 98): 12583.8 MiB
