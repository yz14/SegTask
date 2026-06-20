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


2 肺血管分割任务分析：我目前在进行肺血管分割任务，主要用到的两个方案是2.5D和3D的cubic，D:\codes\work-projects\SegTask\configs\segves2_5d.yaml和D:\codes\work-projects\SegTask\configs\segves3d.yaml。肺血管mask的预处理见D:\codes\work-projects\SegTask\tools\lungves_weight_prep.py。目前的分割结果感觉一般般，请你仔细分析，思考，调研，关于模型方面是否有优化的方案，关于数据处理方面是否有优化的空间，关于损失函数是否有针对性的更好的设计。我的训练集只有110例。我训练了2.5D的过程,用的D:\codes\work-projects\SegTask\configs\lungves0.yaml：
[2026-06-19 13:28:20] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-19 13:28:20] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-19 13:28:20] INFO __main__: Device: cuda
[2026-06-19 13:28:20] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-19 13:28:20] INFO segtask_v1.data.loader: Training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-19 13:28:20] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-19 13:28:20] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-19 13:28:40] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-19 13:28:40] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-19 13:28:40] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-06-19 13:29:15] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-06-19 13:29:15] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-06-19 13:29:26] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-06-19 13:29:26] INFO segtask_v1.data.loader: DataLoader: batch_size=8, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-19 13:29:26] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=12, num_workers=16 => up to ~43.70 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-19 13:29:27] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=22.39M, dec=20.28M, total=45.86M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-19 13:29:28] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-19 13:29:28] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-19 13:29:28] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Loading checkpoint: /data0/yzhen/timm_test/outputs/ves_resnet_bnorm/best_model.pth
[2026-06-19 13:29:28] WARNING segtask_v1.trainer.trainer: Failed to restore RNG state: RNG state must be a torch.ByteTensor
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Resumed from epoch 338, best=mean_dice=0.8271 (patience=0)
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Training: 1000 epochs, device=cuda
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Model params: 45.86M
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=174.9 + grad=174.9 + optim(AdamW,2x)=349.9 + ema=175.0 = 874.8 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=710.2 MiB, reserved=1488.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Train batches: 88, Val batches: 11
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=8
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_focal
[2026-06-19 13:29:28] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-19 13:29:49] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 12441.5 MiB (forward + backward + optimizer.step + EMA update; accum=1 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-19 13:30:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2847, pooled_mean_dice=0.8150, per_class=['0.8150'], iou=0.6878, recall=0.9846, precision=0.6953, vol_sim=0.8278, mcc=0.8222, min_class_dice=0.8150, coverage=[77]/88 samples
[2026-06-19 13:30:53] INFO segtask_v1.trainer.trainer: Epoch 339/1000 | LR=5.86e-05 | loss=0.2676 | val_dice=0.8150 | best=0.8271 (ep338) | 00:01:24 | L_main=0.1363 L_aux_1=0.1060(w=0.5) L_aux_2=0.1567(w=0.5)
[2026-06-19 13:30:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 339): 12512.0 MiB
[2026-06-19 13:31:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3088, pooled_mean_dice=0.8023, per_class=['0.8023'], iou=0.6698, recall=0.9788, precision=0.6797, vol_sim=0.8196, mcc=0.8108, min_class_dice=0.8023, coverage=[72]/88 samples
[2026-06-19 13:31:54] INFO segtask_v1.trainer.trainer: Epoch 340/1000 | LR=5.68e-05 | loss=0.2566 | val_dice=0.8023 | best=0.8271 (ep338) | 00:02:26 | L_main=0.1270 L_aux_1=0.1010(w=0.5) L_aux_2=0.1581(w=0.5)
[2026-06-19 13:31:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 340): 12688.6 MiB
[2026-06-19 13:32:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2921, pooled_mean_dice=0.8075, per_class=['0.8075'], iou=0.6772, recall=0.9776, precision=0.6879, vol_sim=0.8261, mcc=0.8157, min_class_dice=0.8075, coverage=[71]/88 samples
[2026-06-19 13:32:58] INFO segtask_v1.trainer.trainer: Epoch 341/1000 | LR=5.50e-05 | loss=0.2649 | val_dice=0.8075 | best=0.8271 (ep338) | 00:03:29 | L_main=0.1331 L_aux_1=0.1094(w=0.5) L_aux_2=0.1542(w=0.5)
[2026-06-19 13:32:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 341): 12688.6 MiB
[2026-06-19 13:33:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2786, pooled_mean_dice=0.8100, per_class=['0.8100'], iou=0.6807, recall=0.9832, precision=0.6887, vol_sim=0.8238, mcc=0.8173, min_class_dice=0.8100, coverage=[75]/88 samples
[2026-06-19 13:33:59] INFO segtask_v1.trainer.trainer: Epoch 342/1000 | LR=5.32e-05 | loss=0.2538 | val_dice=0.8100 | best=0.8271 (ep338) | 00:04:31 | L_main=0.1258 L_aux_1=0.1000(w=0.5) L_aux_2=0.1560(w=0.5)
[2026-06-19 13:33:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 342): 12688.6 MiB
[2026-06-19 13:35:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3373, pooled_mean_dice=0.8056, per_class=['0.8056'], iou=0.6745, recall=0.9831, precision=0.6824, vol_sim=0.8195, mcc=0.8140, min_class_dice=0.8056, coverage=[79]/88 samples
[2026-06-19 13:35:02] INFO segtask_v1.trainer.trainer: Epoch 343/1000 | LR=5.15e-05 | loss=0.2517 | val_dice=0.8056 | best=0.8271 (ep338) | 00:05:33 | L_main=0.1239 L_aux_1=0.1024(w=0.5) L_aux_2=0.1531(w=0.5)
[2026-06-19 13:35:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 343): 12688.6 MiB
[2026-06-19 13:36:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3154, pooled_mean_dice=0.7968, per_class=['0.7968'], iou=0.6623, recall=0.9787, precision=0.6720, vol_sim=0.8142, mcc=0.8069, min_class_dice=0.7968, coverage=[75]/88 samples
[2026-06-19 13:36:05] INFO segtask_v1.trainer.trainer: Epoch 344/1000 | LR=4.97e-05 | loss=0.2645 | val_dice=0.7968 | best=0.8271 (ep338) | 00:06:37 | L_main=0.1321 L_aux_1=0.1109(w=0.5) L_aux_2=0.1539(w=0.5)
[2026-06-19 13:36:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 344): 12688.6 MiB
[2026-06-19 13:37:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2970, pooled_mean_dice=0.8206, per_class=['0.8206'], iou=0.6958, recall=0.9831, precision=0.7042, vol_sim=0.8347, mcc=0.8270, min_class_dice=0.8206, coverage=[81]/88 samples
[2026-06-19 13:37:08] INFO segtask_v1.trainer.trainer: Epoch 345/1000 | LR=4.80e-05 | loss=0.2404 | val_dice=0.8206 | best=0.8271 (ep338) | 00:07:39 | L_main=0.1211 L_aux_1=0.0964(w=0.5) L_aux_2=0.1422(w=0.5)
[2026-06-19 13:37:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 345): 12688.6 MiB
[2026-06-19 13:38:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3086, pooled_mean_dice=0.7985, per_class=['0.7985'], iou=0.6646, recall=0.9803, precision=0.6736, vol_sim=0.8146, mcc=0.8085, min_class_dice=0.7985, coverage=[75]/88 samples
[2026-06-19 13:38:11] INFO segtask_v1.trainer.trainer: Epoch 346/1000 | LR=4.64e-05 | loss=0.2487 | val_dice=0.7985 | best=0.8271 (ep338) | 00:08:43 | L_main=0.1249 L_aux_1=0.1059(w=0.5) L_aux_2=0.1418(w=0.5)
[2026-06-19 13:38:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 346): 12688.6 MiB
[2026-06-19 13:39:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2884, pooled_mean_dice=0.8001, per_class=['0.8001'], iou=0.6668, recall=0.9816, precision=0.6752, vol_sim=0.8151, mcc=0.8099, min_class_dice=0.8001, coverage=[72]/88 samples
[2026-06-19 13:39:14] INFO segtask_v1.trainer.trainer: Epoch 347/1000 | LR=4.47e-05 | loss=0.2664 | val_dice=0.8001 | best=0.8271 (ep338) | 00:09:45 | L_main=0.1364 L_aux_1=0.1051(w=0.5) L_aux_2=0.1549(w=0.5)
[2026-06-19 13:39:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 347): 12688.6 MiB
[2026-06-19 13:40:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2858, pooled_mean_dice=0.8243, per_class=['0.8243'], iou=0.7012, recall=0.9802, precision=0.7112, vol_sim=0.8410, mcc=0.8306, min_class_dice=0.8243, coverage=[68]/88 samples
[2026-06-19 13:40:15] INFO segtask_v1.trainer.trainer: Epoch 348/1000 | LR=4.31e-05 | loss=0.2478 | val_dice=0.8243 | best=0.8271 (ep338) | 00:10:47 | L_main=0.1244 L_aux_1=0.0983(w=0.5) L_aux_2=0.1484(w=0.5)
[2026-06-19 13:40:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 348): 12688.6 MiB
[2026-06-19 13:41:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2975, pooled_mean_dice=0.8190, per_class=['0.8190'], iou=0.6934, recall=0.9835, precision=0.7016, vol_sim=0.8327, mcc=0.8256, min_class_dice=0.8190, coverage=[79]/88 samples
[2026-06-19 13:41:18] INFO segtask_v1.trainer.trainer: Epoch 349/1000 | LR=4.15e-05 | loss=0.2480 | val_dice=0.8190 | best=0.8271 (ep338) | 00:11:49 | L_main=0.1219 L_aux_1=0.1003(w=0.5) L_aux_2=0.1518(w=0.5)
[2026-06-19 13:41:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 349): 12688.6 MiB
[2026-06-19 13:42:21] INFO segtask_v1.trainer.validation:   Val: loss=0.3513, pooled_mean_dice=0.8000, per_class=['0.8000'], iou=0.6667, recall=0.9820, precision=0.6750, vol_sim=0.8147, mcc=0.8105, min_class_dice=0.8000, coverage=[78]/88 samples
[2026-06-19 13:42:21] INFO segtask_v1.trainer.trainer: Epoch 350/1000 | LR=4.00e-05 | loss=0.2479 | val_dice=0.8000 | best=0.8271 (ep338) | 00:12:52 | L_main=0.1226 L_aux_1=0.1039(w=0.5) L_aux_2=0.1468(w=0.5)
[2026-06-19 13:42:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 350): 12688.6 MiB
[2026-06-19 13:43:24] INFO segtask_v1.trainer.validation:   Val: loss=0.2848, pooled_mean_dice=0.8206, per_class=['0.8206'], iou=0.6957, recall=0.9811, precision=0.7052, vol_sim=0.8364, mcc=0.8272, min_class_dice=0.8206, coverage=[73]/88 samples
[2026-06-19 13:43:24] INFO segtask_v1.trainer.trainer: Epoch 351/1000 | LR=3.85e-05 | loss=0.2501 | val_dice=0.8206 | best=0.8271 (ep338) | 00:13:55 | L_main=0.1235 L_aux_1=0.1132(w=0.5) L_aux_2=0.1401(w=0.5)
[2026-06-19 13:43:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 351): 12688.6 MiB
[2026-06-19 13:44:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2975, pooled_mean_dice=0.8011, per_class=['0.8011'], iou=0.6681, recall=0.9819, precision=0.6765, vol_sim=0.8158, mcc=0.8109, min_class_dice=0.8011, coverage=[74]/88 samples
[2026-06-19 13:44:27] INFO segtask_v1.trainer.trainer: Epoch 352/1000 | LR=3.70e-05 | loss=0.2597 | val_dice=0.8011 | best=0.8271 (ep338) | 00:14:58 | L_main=0.1297 L_aux_1=0.1017(w=0.5) L_aux_2=0.1583(w=0.5)
[2026-06-19 13:44:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 352): 12688.6 MiB
[2026-06-19 13:45:29] INFO segtask_v1.trainer.validation:   Val: loss=0.2858, pooled_mean_dice=0.7984, per_class=['0.7984'], iou=0.6644, recall=0.9808, precision=0.6732, vol_sim=0.8140, mcc=0.8083, min_class_dice=0.7984, coverage=[73]/88 samples
[2026-06-19 13:45:29] INFO segtask_v1.trainer.trainer: Epoch 353/1000 | LR=3.55e-05 | loss=0.2436 | val_dice=0.7984 | best=0.8271 (ep338) | 00:16:00 | L_main=0.1205 L_aux_1=0.0966(w=0.5) L_aux_2=0.1496(w=0.5)
[2026-06-19 13:45:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 353): 12688.6 MiB
[2026-06-19 13:46:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2697, pooled_mean_dice=0.8052, per_class=['0.8052'], iou=0.6739, recall=0.9741, precision=0.6862, vol_sim=0.8266, mcc=0.8140, min_class_dice=0.8052, coverage=[68]/88 samples
[2026-06-19 13:46:31] INFO segtask_v1.trainer.trainer: Epoch 354/1000 | LR=3.41e-05 | loss=0.2554 | val_dice=0.8052 | best=0.8271 (ep338) | 00:17:03 | L_main=0.1258 L_aux_1=0.1012(w=0.5) L_aux_2=0.1578(w=0.5)
[2026-06-19 13:46:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 354): 12688.6 MiB
[2026-06-19 13:47:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3265, pooled_mean_dice=0.7844, per_class=['0.7844'], iou=0.6452, recall=0.9764, precision=0.6554, vol_sim=0.8033, mcc=0.7956, min_class_dice=0.7844, coverage=[80]/88 samples
[2026-06-19 13:47:34] INFO segtask_v1.trainer.trainer: Epoch 355/1000 | LR=3.27e-05 | loss=0.2722 | val_dice=0.7844 | best=0.8271 (ep338) | 00:18:05 | L_main=0.1354 L_aux_1=0.1092(w=0.5) L_aux_2=0.1644(w=0.5)
[2026-06-19 13:47:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 355): 12688.6 MiB
[2026-06-19 13:48:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2994, pooled_mean_dice=0.8247, per_class=['0.8247'], iou=0.7017, recall=0.9847, precision=0.7094, vol_sim=0.8375, mcc=0.8305, min_class_dice=0.8247, coverage=[77]/88 samples
[2026-06-19 13:48:36] INFO segtask_v1.trainer.trainer: Epoch 356/1000 | LR=3.13e-05 | loss=0.2249 | val_dice=0.8247 | best=0.8271 (ep338) | 00:19:07 | L_main=0.1104 L_aux_1=0.0963(w=0.5) L_aux_2=0.1326(w=0.5)
[2026-06-19 13:48:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 356): 12688.6 MiB
[2026-06-19 13:49:38] INFO segtask_v1.trainer.validation:   Val: loss=0.3376, pooled_mean_dice=0.8031, per_class=['0.8031'], iou=0.6710, recall=0.9843, precision=0.6783, vol_sim=0.8160, mcc=0.8124, min_class_dice=0.8031, coverage=[77]/88 samples
[2026-06-19 13:49:38] INFO segtask_v1.trainer.trainer: Epoch 357/1000 | LR=2.99e-05 | loss=0.2522 | val_dice=0.8031 | best=0.8271 (ep338) | 00:20:09 | L_main=0.1257 L_aux_1=0.1033(w=0.5) L_aux_2=0.1497(w=0.5)
[2026-06-19 13:49:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 357): 12688.6 MiB
[2026-06-19 13:50:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2908, pooled_mean_dice=0.8201, per_class=['0.8201'], iou=0.6950, recall=0.9814, precision=0.7043, vol_sim=0.8356, mcc=0.8268, min_class_dice=0.8201, coverage=[73]/88 samples
[2026-06-19 13:50:41] INFO segtask_v1.trainer.trainer: Epoch 358/1000 | LR=2.86e-05 | loss=0.2484 | val_dice=0.8201 | best=0.8271 (ep338) | 00:21:12 | L_main=0.1242 L_aux_1=0.1043(w=0.5) L_aux_2=0.1441(w=0.5)
[2026-06-19 13:50:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 358): 12688.6 MiB
[2026-06-19 13:51:43] INFO segtask_v1.trainer.validation:   Val: loss=0.3120, pooled_mean_dice=0.8123, per_class=['0.8123'], iou=0.6839, recall=0.9793, precision=0.6939, vol_sim=0.8294, mcc=0.8194, min_class_dice=0.8123, coverage=[75]/88 samples
[2026-06-19 13:51:43] INFO segtask_v1.trainer.trainer: Epoch 359/1000 | LR=2.73e-05 | loss=0.2436 | val_dice=0.8123 | best=0.8271 (ep338) | 00:22:15 | L_main=0.1209 L_aux_1=0.0990(w=0.5) L_aux_2=0.1465(w=0.5)
[2026-06-19 13:51:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 359): 12688.6 MiB
[2026-06-19 13:52:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2751, pooled_mean_dice=0.8173, per_class=['0.8173'], iou=0.6910, recall=0.9801, precision=0.7008, vol_sim=0.8339, mcc=0.8246, min_class_dice=0.8173, coverage=[71]/88 samples
[2026-06-19 13:52:46] INFO segtask_v1.trainer.trainer: Epoch 360/1000 | LR=2.61e-05 | loss=0.2544 | val_dice=0.8173 | best=0.8271 (ep338) | 00:23:17 | L_main=0.1273 L_aux_1=0.1046(w=0.5) L_aux_2=0.1494(w=0.5)
[2026-06-19 13:52:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 360): 12688.6 MiB
[2026-06-19 13:53:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2789, pooled_mean_dice=0.8118, per_class=['0.8118'], iou=0.6833, recall=0.9800, precision=0.6929, vol_sim=0.8284, mcc=0.8196, min_class_dice=0.8118, coverage=[74]/88 samples
[2026-06-19 13:53:50] INFO segtask_v1.trainer.trainer: Epoch 361/1000 | LR=2.48e-05 | loss=0.2485 | val_dice=0.8118 | best=0.8271 (ep338) | 00:24:21 | L_main=0.1247 L_aux_1=0.1041(w=0.5) L_aux_2=0.1435(w=0.5)
[2026-06-19 13:53:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 361): 12688.6 MiB
[2026-06-19 13:54:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2796, pooled_mean_dice=0.8012, per_class=['0.8012'], iou=0.6683, recall=0.9805, precision=0.6773, vol_sim=0.8171, mcc=0.8105, min_class_dice=0.8012, coverage=[69]/88 samples
[2026-06-19 13:54:53] INFO segtask_v1.trainer.trainer: Epoch 362/1000 | LR=2.36e-05 | loss=0.2562 | val_dice=0.8012 | best=0.8271 (ep338) | 00:25:24 | L_main=0.1304 L_aux_1=0.1022(w=0.5) L_aux_2=0.1493(w=0.5)
[2026-06-19 13:54:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 362): 12688.6 MiB
[2026-06-19 13:55:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2856, pooled_mean_dice=0.8130, per_class=['0.8130'], iou=0.6849, recall=0.9795, precision=0.6949, vol_sim=0.8300, mcc=0.8204, min_class_dice=0.8130, coverage=[72]/88 samples
[2026-06-19 13:55:56] INFO segtask_v1.trainer.trainer: Epoch 363/1000 | LR=2.25e-05 | loss=0.2407 | val_dice=0.8130 | best=0.8271 (ep338) | 00:26:27 | L_main=0.1202 L_aux_1=0.0929(w=0.5) L_aux_2=0.1482(w=0.5)
[2026-06-19 13:55:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 363): 12688.6 MiB
[2026-06-19 13:56:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2833, pooled_mean_dice=0.8140, per_class=['0.8140'], iou=0.6863, recall=0.9792, precision=0.6965, vol_sim=0.8313, mcc=0.8206, min_class_dice=0.8140, coverage=[76]/88 samples
[2026-06-19 13:56:59] INFO segtask_v1.trainer.trainer: Epoch 364/1000 | LR=2.13e-05 | loss=0.2394 | val_dice=0.8140 | best=0.8271 (ep338) | 00:27:30 | L_main=0.1202 L_aux_1=0.0984(w=0.5) L_aux_2=0.1399(w=0.5)
[2026-06-19 13:56:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 364): 12688.6 MiB
[2026-06-19 13:58:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3484, pooled_mean_dice=0.8027, per_class=['0.8027'], iou=0.6704, recall=0.9822, precision=0.6787, vol_sim=0.8172, mcc=0.8121, min_class_dice=0.8027, coverage=[78]/88 samples
[2026-06-19 13:58:02] INFO segtask_v1.trainer.trainer: Epoch 365/1000 | LR=2.02e-05 | loss=0.2547 | val_dice=0.8027 | best=0.8271 (ep338) | 00:28:33 | L_main=0.1269 L_aux_1=0.0965(w=0.5) L_aux_2=0.1591(w=0.5)
[2026-06-19 13:58:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 365): 12688.6 MiB
[2026-06-19 13:59:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3074, pooled_mean_dice=0.8118, per_class=['0.8118'], iou=0.6833, recall=0.9823, precision=0.6918, vol_sim=0.8265, mcc=0.8198, min_class_dice=0.8118, coverage=[74]/88 samples
[2026-06-19 13:59:05] INFO segtask_v1.trainer.trainer: Epoch 366/1000 | LR=1.92e-05 | loss=0.2419 | val_dice=0.8118 | best=0.8271 (ep338) | 00:29:36 | L_main=0.1205 L_aux_1=0.0998(w=0.5) L_aux_2=0.1430(w=0.5)
[2026-06-19 13:59:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 366): 12688.6 MiB
[2026-06-19 14:00:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3468, pooled_mean_dice=0.8067, per_class=['0.8067'], iou=0.6760, recall=0.9767, precision=0.6871, vol_sim=0.8260, mcc=0.8149, min_class_dice=0.8067, coverage=[79]/88 samples
[2026-06-19 14:00:09] INFO segtask_v1.trainer.trainer: Epoch 367/1000 | LR=1.81e-05 | loss=0.2493 | val_dice=0.8067 | best=0.8271 (ep338) | 00:30:40 | L_main=0.1273 L_aux_1=0.0972(w=0.5) L_aux_2=0.1469(w=0.5)
[2026-06-19 14:00:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 367): 12688.6 MiB
[2026-06-19 14:01:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3106, pooled_mean_dice=0.8088, per_class=['0.8088'], iou=0.6790, recall=0.9756, precision=0.6907, vol_sim=0.8290, mcc=0.8163, min_class_dice=0.8088, coverage=[73]/88 samples
[2026-06-19 14:01:11] INFO segtask_v1.trainer.trainer: Epoch 368/1000 | LR=1.71e-05 | loss=0.2495 | val_dice=0.8088 | best=0.8271 (ep338) | 00:31:42 | L_main=0.1254 L_aux_1=0.0993(w=0.5) L_aux_2=0.1490(w=0.5)
[2026-06-19 14:01:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 368): 12688.6 MiB
[2026-06-19 14:02:12] INFO segtask_v1.trainer.validation:   Val: loss=0.3322, pooled_mean_dice=0.8189, per_class=['0.8189'], iou=0.6933, recall=0.9836, precision=0.7014, vol_sim=0.8325, mcc=0.8264, min_class_dice=0.8189, coverage=[78]/88 samples
[2026-06-19 14:02:12] INFO segtask_v1.trainer.trainer: Epoch 369/1000 | LR=1.61e-05 | loss=0.2475 | val_dice=0.8189 | best=0.8271 (ep338) | 00:32:43 | L_main=0.1263 L_aux_1=0.0985(w=0.5) L_aux_2=0.1439(w=0.5)
[2026-06-19 14:02:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 369): 12688.6 MiB
[2026-06-19 14:03:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2887, pooled_mean_dice=0.8190, per_class=['0.8190'], iou=0.6934, recall=0.9795, precision=0.7036, vol_sim=0.8361, mcc=0.8244, min_class_dice=0.8190, coverage=[78]/88 samples
[2026-06-19 14:03:14] INFO segtask_v1.trainer.trainer: Epoch 370/1000 | LR=1.52e-05 | loss=0.2438 | val_dice=0.8190 | best=0.8271 (ep338) | 00:33:46 | L_main=0.1222 L_aux_1=0.0941(w=0.5) L_aux_2=0.1491(w=0.5)
[2026-06-19 14:03:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 370): 12688.6 MiB
[2026-06-19 14:04:18] INFO segtask_v1.trainer.validation:   Val: loss=0.3197, pooled_mean_dice=0.8195, per_class=['0.8195'], iou=0.6941, recall=0.9750, precision=0.7067, vol_sim=0.8405, mcc=0.8252, min_class_dice=0.8195, coverage=[79]/88 samples
[2026-06-19 14:04:18] INFO segtask_v1.trainer.trainer: Epoch 371/1000 | LR=1.42e-05 | loss=0.2451 | val_dice=0.8195 | best=0.8271 (ep338) | 00:34:49 | L_main=0.1208 L_aux_1=0.0998(w=0.5) L_aux_2=0.1488(w=0.5)
[2026-06-19 14:04:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 371): 12688.6 MiB
[2026-06-19 14:05:22] INFO segtask_v1.trainer.validation:   Val: loss=0.3269, pooled_mean_dice=0.8015, per_class=['0.8015'], iou=0.6688, recall=0.9819, precision=0.6771, vol_sim=0.8163, mcc=0.8108, min_class_dice=0.8015, coverage=[78]/88 samples
[2026-06-19 14:05:22] INFO segtask_v1.trainer.trainer: Epoch 372/1000 | LR=1.33e-05 | loss=0.2399 | val_dice=0.8015 | best=0.8271 (ep338) | 00:35:53 | L_main=0.1200 L_aux_1=0.0958(w=0.5) L_aux_2=0.1441(w=0.5)
[2026-06-19 14:05:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 372): 12688.6 MiB
[2026-06-19 14:06:25] INFO segtask_v1.trainer.validation:   Val: loss=0.2926, pooled_mean_dice=0.8086, per_class=['0.8086'], iou=0.6787, recall=0.9805, precision=0.6880, vol_sim=0.8247, mcc=0.8168, min_class_dice=0.8086, coverage=[73]/88 samples
[2026-06-19 14:06:25] INFO segtask_v1.trainer.trainer: Epoch 373/1000 | LR=1.25e-05 | loss=0.2394 | val_dice=0.8086 | best=0.8271 (ep338) | 00:36:56 | L_main=0.1208 L_aux_1=0.0960(w=0.5) L_aux_2=0.1412(w=0.5)
[2026-06-19 14:06:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 373): 12688.6 MiB
[2026-06-19 14:07:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2910, pooled_mean_dice=0.8092, per_class=['0.8092'], iou=0.6795, recall=0.9808, precision=0.6887, vol_sim=0.8250, mcc=0.8174, min_class_dice=0.8092, coverage=[74]/88 samples
[2026-06-19 14:07:28] INFO segtask_v1.trainer.trainer: Epoch 374/1000 | LR=1.16e-05 | loss=0.2394 | val_dice=0.8092 | best=0.8271 (ep338) | 00:37:59 | L_main=0.1202 L_aux_1=0.0949(w=0.5) L_aux_2=0.1435(w=0.5)
[2026-06-19 14:07:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 374): 12688.6 MiB
[2026-06-19 14:08:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3187, pooled_mean_dice=0.8044, per_class=['0.8044'], iou=0.6728, recall=0.9792, precision=0.6825, vol_sim=0.8215, mcc=0.8124, min_class_dice=0.8044, coverage=[78]/88 samples
[2026-06-19 14:08:31] INFO segtask_v1.trainer.trainer: Epoch 375/1000 | LR=1.08e-05 | loss=0.2485 | val_dice=0.8044 | best=0.8271 (ep338) | 00:39:03 | L_main=0.1216 L_aux_1=0.0994(w=0.5) L_aux_2=0.1544(w=0.5)
[2026-06-19 14:08:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 375): 12688.6 MiB
[2026-06-19 14:09:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2870, pooled_mean_dice=0.8150, per_class=['0.8150'], iou=0.6878, recall=0.9828, precision=0.6962, vol_sim=0.8293, mcc=0.8225, min_class_dice=0.8150, coverage=[72]/88 samples
[2026-06-19 14:09:33] INFO segtask_v1.trainer.trainer: Epoch 376/1000 | LR=1.01e-05 | loss=0.2363 | val_dice=0.8150 | best=0.8271 (ep338) | 00:40:04 | L_main=0.1203 L_aux_1=0.0938(w=0.5) L_aux_2=0.1381(w=0.5)
[2026-06-19 14:09:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 376): 12688.6 MiB
[2026-06-19 14:10:36] INFO segtask_v1.trainer.validation:   Val: loss=0.3294, pooled_mean_dice=0.8113, per_class=['0.8113'], iou=0.6825, recall=0.9812, precision=0.6916, vol_sim=0.8269, mcc=0.8189, min_class_dice=0.8113, coverage=[75]/88 samples
[2026-06-19 14:10:36] INFO segtask_v1.trainer.trainer: Epoch 377/1000 | LR=9.33e-06 | loss=0.2534 | val_dice=0.8113 | best=0.8271 (ep338) | 00:41:07 | L_main=0.1290 L_aux_1=0.1083(w=0.5) L_aux_2=0.1403(w=0.5)
[2026-06-19 14:10:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 377): 12688.6 MiB
[2026-06-19 14:11:38] INFO segtask_v1.trainer.validation:   Val: loss=0.2858, pooled_mean_dice=0.8152, per_class=['0.8152'], iou=0.6880, recall=0.9832, precision=0.6962, vol_sim=0.8292, mcc=0.8221, min_class_dice=0.8152, coverage=[75]/88 samples
[2026-06-19 14:11:38] INFO segtask_v1.trainer.trainer: Epoch 378/1000 | LR=8.63e-06 | loss=0.2552 | val_dice=0.8152 | best=0.8271 (ep338) | 00:42:10 | L_main=0.1284 L_aux_1=0.1048(w=0.5) L_aux_2=0.1489(w=0.5)
[2026-06-19 14:11:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 378): 12688.6 MiB
[2026-06-19 14:12:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3223, pooled_mean_dice=0.8195, per_class=['0.8195'], iou=0.6942, recall=0.9809, precision=0.7037, vol_sim=0.8355, mcc=0.8262, min_class_dice=0.8195, coverage=[74]/88 samples
[2026-06-19 14:12:40] INFO segtask_v1.trainer.trainer: Epoch 379/1000 | LR=7.95e-06 | loss=0.2335 | val_dice=0.8195 | best=0.8271 (ep338) | 00:43:11 | L_main=0.1158 L_aux_1=0.0955(w=0.5) L_aux_2=0.1399(w=0.5)
[2026-06-19 14:12:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 379): 12688.6 MiB
[2026-06-19 14:13:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2868, pooled_mean_dice=0.8170, per_class=['0.8170'], iou=0.6906, recall=0.9831, precision=0.6988, vol_sim=0.8310, mcc=0.8236, min_class_dice=0.8170, coverage=[78]/88 samples
[2026-06-19 14:13:42] INFO segtask_v1.trainer.trainer: Epoch 380/1000 | LR=7.31e-06 | loss=0.2641 | val_dice=0.8170 | best=0.8271 (ep338) | 00:44:13 | L_main=0.1300 L_aux_1=0.1081(w=0.5) L_aux_2=0.1601(w=0.5)
[2026-06-19 14:13:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 380): 12688.6 MiB
[2026-06-19 14:14:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2764, pooled_mean_dice=0.8149, per_class=['0.8149'], iou=0.6876, recall=0.9814, precision=0.6966, vol_sim=0.8303, mcc=0.8216, min_class_dice=0.8149, coverage=[72]/88 samples
[2026-06-19 14:14:46] INFO segtask_v1.trainer.trainer: Epoch 381/1000 | LR=6.69e-06 | loss=0.2305 | val_dice=0.8149 | best=0.8271 (ep338) | 00:45:17 | L_main=0.1159 L_aux_1=0.0938(w=0.5) L_aux_2=0.1354(w=0.5)
[2026-06-19 14:14:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 381): 12688.6 MiB
[2026-06-19 14:15:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3300, pooled_mean_dice=0.8032, per_class=['0.8032'], iou=0.6711, recall=0.9799, precision=0.6805, vol_sim=0.8196, mcc=0.8123, min_class_dice=0.8032, coverage=[76]/88 samples
[2026-06-19 14:15:49] INFO segtask_v1.trainer.trainer: Epoch 382/1000 | LR=6.11e-06 | loss=0.2520 | val_dice=0.8032 | best=0.8271 (ep338) | 00:46:20 | L_main=0.1256 L_aux_1=0.0938(w=0.5) L_aux_2=0.1590(w=0.5)
[2026-06-19 14:15:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 382): 12688.6 MiB
[2026-06-19 14:16:52] INFO segtask_v1.trainer.validation:   Val: loss=0.3132, pooled_mean_dice=0.8009, per_class=['0.8009'], iou=0.6680, recall=0.9810, precision=0.6767, vol_sim=0.8164, mcc=0.8106, min_class_dice=0.8009, coverage=[73]/88 samples
[2026-06-19 14:16:52] INFO segtask_v1.trainer.trainer: Epoch 383/1000 | LR=5.56e-06 | loss=0.2598 | val_dice=0.8009 | best=0.8271 (ep338) | 00:47:24 | L_main=0.1309 L_aux_1=0.0971(w=0.5) L_aux_2=0.1606(w=0.5)
[2026-06-19 14:16:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 383): 12688.6 MiB
[2026-06-19 14:17:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3381, pooled_mean_dice=0.8052, per_class=['0.8052'], iou=0.6739, recall=0.9810, precision=0.6828, vol_sim=0.8208, mcc=0.8144, min_class_dice=0.8052, coverage=[75]/88 samples
[2026-06-19 14:17:55] INFO segtask_v1.trainer.trainer: Epoch 384/1000 | LR=5.04e-06 | loss=0.2514 | val_dice=0.8052 | best=0.8271 (ep338) | 00:48:26 | L_main=0.1265 L_aux_1=0.1077(w=0.5) L_aux_2=0.1421(w=0.5)
[2026-06-19 14:17:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 384): 12688.6 MiB
[2026-06-19 14:18:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3002, pooled_mean_dice=0.8012, per_class=['0.8012'], iou=0.6683, recall=0.9841, precision=0.6756, vol_sim=0.8141, mcc=0.8107, min_class_dice=0.8012, coverage=[73]/88 samples
[2026-06-19 14:18:57] INFO segtask_v1.trainer.trainer: Epoch 385/1000 | LR=4.55e-06 | loss=0.2407 | val_dice=0.8012 | best=0.8271 (ep338) | 00:49:28 | L_main=0.1198 L_aux_1=0.0947(w=0.5) L_aux_2=0.1471(w=0.5)
[2026-06-19 14:18:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 385): 12688.6 MiB
[2026-06-19 14:20:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3310, pooled_mean_dice=0.8224, per_class=['0.8224'], iou=0.6983, recall=0.9809, precision=0.7080, vol_sim=0.8384, mcc=0.8281, min_class_dice=0.8224, coverage=[81]/88 samples
[2026-06-19 14:20:00] INFO segtask_v1.trainer.trainer: Epoch 386/1000 | LR=4.09e-06 | loss=0.2443 | val_dice=0.8224 | best=0.8271 (ep338) | 00:50:31 | L_main=0.1217 L_aux_1=0.0979(w=0.5) L_aux_2=0.1474(w=0.5)
[2026-06-19 14:20:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 386): 12688.6 MiB
[2026-06-19 14:21:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2784, pooled_mean_dice=0.8052, per_class=['0.8052'], iou=0.6740, recall=0.9789, precision=0.6839, vol_sim=0.8226, mcc=0.8137, min_class_dice=0.8052, coverage=[73]/88 samples
[2026-06-19 14:21:04] INFO segtask_v1.trainer.trainer: Epoch 387/1000 | LR=3.67e-06 | loss=0.2718 | val_dice=0.8052 | best=0.8271 (ep338) | 00:51:35 | L_main=0.1364 L_aux_1=0.1109(w=0.5) L_aux_2=0.1598(w=0.5)
[2026-06-19 14:21:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 387): 12688.6 MiB
[2026-06-19 14:22:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2872, pooled_mean_dice=0.8206, per_class=['0.8206'], iou=0.6957, recall=0.9801, precision=0.7057, vol_sim=0.8372, mcc=0.8266, min_class_dice=0.8206, coverage=[76]/88 samples
[2026-06-19 14:22:06] INFO segtask_v1.trainer.trainer: Epoch 388/1000 | LR=3.27e-06 | loss=0.2459 | val_dice=0.8206 | best=0.8271 (ep338) | 00:52:38 | L_main=0.1254 L_aux_1=0.0995(w=0.5) L_aux_2=0.1415(w=0.5)
[2026-06-19 14:22:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 388): 12688.6 MiB
[2026-06-19 14:23:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2923, pooled_mean_dice=0.8068, per_class=['0.8068'], iou=0.6762, recall=0.9790, precision=0.6861, vol_sim=0.8241, mcc=0.8143, min_class_dice=0.8068, coverage=[73]/88 samples
[2026-06-19 14:23:09] INFO segtask_v1.trainer.trainer: Epoch 389/1000 | LR=2.91e-06 | loss=0.2415 | val_dice=0.8068 | best=0.8271 (ep338) | 00:53:41 | L_main=0.1209 L_aux_1=0.0980(w=0.5) L_aux_2=0.1432(w=0.5)
[2026-06-19 14:23:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 389): 12688.6 MiB
[2026-06-19 14:24:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3435, pooled_mean_dice=0.8008, per_class=['0.8008'], iou=0.6678, recall=0.9764, precision=0.6788, vol_sim=0.8202, mcc=0.8102, min_class_dice=0.8008, coverage=[77]/88 samples
[2026-06-19 14:24:11] INFO segtask_v1.trainer.trainer: Epoch 390/1000 | LR=2.58e-06 | loss=0.2439 | val_dice=0.8008 | best=0.8271 (ep338) | 00:54:43 | L_main=0.1211 L_aux_1=0.1015(w=0.5) L_aux_2=0.1441(w=0.5)
[2026-06-19 14:24:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 390): 12688.6 MiB
[2026-06-19 14:25:15] INFO segtask_v1.trainer.validation:   Val: loss=0.3552, pooled_mean_dice=0.7979, per_class=['0.7979'], iou=0.6638, recall=0.9798, precision=0.6730, vol_sim=0.8144, mcc=0.8081, min_class_dice=0.7979, coverage=[72]/88 samples
[2026-06-19 14:25:15] INFO segtask_v1.trainer.trainer: Epoch 391/1000 | LR=2.28e-06 | loss=0.2301 | val_dice=0.7979 | best=0.8271 (ep338) | 00:55:46 | L_main=0.1157 L_aux_1=0.0995(w=0.5) L_aux_2=0.1294(w=0.5)
[2026-06-19 14:25:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 391): 12688.6 MiB
[2026-06-19 14:26:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2957, pooled_mean_dice=0.8156, per_class=['0.8156'], iou=0.6887, recall=0.9768, precision=0.7001, vol_sim=0.8350, mcc=0.8220, min_class_dice=0.8156, coverage=[76]/88 samples
[2026-06-19 14:26:17] INFO segtask_v1.trainer.trainer: Epoch 392/1000 | LR=2.01e-06 | loss=0.2563 | val_dice=0.8156 | best=0.8271 (ep338) | 00:56:48 | L_main=0.1296 L_aux_1=0.1036(w=0.5) L_aux_2=0.1498(w=0.5)
[2026-06-19 14:26:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 392): 12688.6 MiB
[2026-06-19 14:27:21] INFO segtask_v1.trainer.validation:   Val: loss=0.3366, pooled_mean_dice=0.7928, per_class=['0.7928'], iou=0.6567, recall=0.9783, precision=0.6664, vol_sim=0.8104, mcc=0.8033, min_class_dice=0.7928, coverage=[78]/88 samples
[2026-06-19 14:27:21] INFO segtask_v1.trainer.trainer: Epoch 393/1000 | LR=1.77e-06 | loss=0.2491 | val_dice=0.7928 | best=0.8271 (ep338) | 00:57:52 | L_main=0.1233 L_aux_1=0.1071(w=0.5) L_aux_2=0.1444(w=0.5)
[2026-06-19 14:27:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 393): 12688.6 MiB
[2026-06-19 14:28:24] INFO segtask_v1.trainer.validation:   Val: loss=0.3168, pooled_mean_dice=0.7858, per_class=['0.7858'], iou=0.6472, recall=0.9749, precision=0.6582, vol_sim=0.8061, mcc=0.7971, min_class_dice=0.7858, coverage=[74]/88 samples
[2026-06-19 14:28:24] INFO segtask_v1.trainer.trainer: Epoch 394/1000 | LR=1.57e-06 | loss=0.2362 | val_dice=0.7858 | best=0.8271 (ep338) | 00:58:56 | L_main=0.1181 L_aux_1=0.0996(w=0.5) L_aux_2=0.1365(w=0.5)
[2026-06-19 14:28:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 394): 12688.6 MiB
[2026-06-19 14:29:27] INFO segtask_v1.trainer.validation:   Val: loss=0.3053, pooled_mean_dice=0.8184, per_class=['0.8184'], iou=0.6927, recall=0.9787, precision=0.7033, vol_sim=0.8362, mcc=0.8245, min_class_dice=0.8184, coverage=[78]/88 samples
[2026-06-19 14:29:27] INFO segtask_v1.trainer.trainer: Epoch 395/1000 | LR=1.39e-06 | loss=0.2462 | val_dice=0.8184 | best=0.8271 (ep338) | 00:59:59 | L_main=0.1244 L_aux_1=0.0972(w=0.5) L_aux_2=0.1464(w=0.5)
[2026-06-19 14:29:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 395): 12688.6 MiB
[2026-06-19 14:30:30] INFO segtask_v1.trainer.validation:   Val: loss=0.3504, pooled_mean_dice=0.7941, per_class=['0.7941'], iou=0.6584, recall=0.9821, precision=0.6664, vol_sim=0.8085, mcc=0.8043, min_class_dice=0.7941, coverage=[81]/88 samples
[2026-06-19 14:30:30] INFO segtask_v1.trainer.trainer: Epoch 396/1000 | LR=1.25e-06 | loss=0.2366 | val_dice=0.7941 | best=0.8271 (ep338) | 01:01:01 | L_main=0.1181 L_aux_1=0.1010(w=0.5) L_aux_2=0.1360(w=0.5)
[2026-06-19 14:30:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 396): 12688.6 MiB
[2026-06-19 14:31:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3047, pooled_mean_dice=0.8156, per_class=['0.8156'], iou=0.6886, recall=0.9822, precision=0.6973, vol_sim=0.8304, mcc=0.8239, min_class_dice=0.8156, coverage=[73]/88 samples
[2026-06-19 14:31:32] INFO segtask_v1.trainer.trainer: Epoch 397/1000 | LR=1.14e-06 | loss=0.2501 | val_dice=0.8156 | best=0.8271 (ep338) | 01:02:03 | L_main=0.1262 L_aux_1=0.1059(w=0.5) L_aux_2=0.1418(w=0.5)
[2026-06-19 14:31:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 397): 12688.6 MiB
[2026-06-19 14:32:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2956, pooled_mean_dice=0.8065, per_class=['0.8065'], iou=0.6758, recall=0.9824, precision=0.6841, vol_sim=0.8210, mcc=0.8148, min_class_dice=0.8065, coverage=[77]/88 samples
[2026-06-19 14:32:33] INFO segtask_v1.trainer.trainer: Epoch 398/1000 | LR=1.06e-06 | loss=0.2260 | val_dice=0.8065 | best=0.8271 (ep338) | 01:03:04 | L_main=0.1114 L_aux_1=0.0880(w=0.5) L_aux_2=0.1412(w=0.5)
[2026-06-19 14:32:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 398): 12688.6 MiB
[2026-06-19 14:33:36] INFO segtask_v1.trainer.validation:   Val: loss=0.3045, pooled_mean_dice=0.8078, per_class=['0.8078'], iou=0.6776, recall=0.9827, precision=0.6858, vol_sim=0.8221, mcc=0.8166, min_class_dice=0.8078, coverage=[75]/88 samples
[2026-06-19 14:33:36] INFO segtask_v1.trainer.trainer: Epoch 399/1000 | LR=1.02e-06 | loss=0.2358 | val_dice=0.8078 | best=0.8271 (ep338) | 01:04:07 | L_main=0.1181 L_aux_1=0.0961(w=0.5) L_aux_2=0.1391(w=0.5)
[2026-06-19 14:33:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 399): 12688.6 MiB
[2026-06-19 14:34:39] INFO segtask_v1.trainer.validation:   Val: loss=0.2765, pooled_mean_dice=0.8210, per_class=['0.8210'], iou=0.6963, recall=0.9845, precision=0.7041, vol_sim=0.8340, mcc=0.8272, min_class_dice=0.8210, coverage=[73]/88 samples
[2026-06-19 14:34:39] INFO segtask_v1.trainer.trainer: Epoch 400/1000 | LR=1.00e-06 | loss=0.2570 | val_dice=0.8210 | best=0.8271 (ep338) | 01:05:10 | L_main=0.1297 L_aux_1=0.1028(w=0.5) L_aux_2=0.1518(w=0.5)
[2026-06-19 14:34:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 400): 12688.6 MiB
[2026-06-19 14:35:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2756, pooled_mean_dice=0.8063, per_class=['0.8063'], iou=0.6755, recall=0.9798, precision=0.6851, vol_sim=0.8230, mcc=0.8149, min_class_dice=0.8063, coverage=[72]/88 samples
[2026-06-19 14:35:43] INFO segtask_v1.trainer.trainer: Epoch 401/1000 | LR=1.02e-06 | loss=0.2480 | val_dice=0.8063 | best=0.8271 (ep338) | 01:06:14 | L_main=0.1225 L_aux_1=0.0997(w=0.5) L_aux_2=0.1512(w=0.5)
[2026-06-19 14:35:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 401): 12688.6 MiB
[2026-06-19 14:36:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3057, pooled_mean_dice=0.8093, per_class=['0.8093'], iou=0.6796, recall=0.9764, precision=0.6910, vol_sim=0.8288, mcc=0.8168, min_class_dice=0.8093, coverage=[77]/88 samples
[2026-06-19 14:36:45] INFO segtask_v1.trainer.trainer: Epoch 402/1000 | LR=1.06e-06 | loss=0.2480 | val_dice=0.8093 | best=0.8271 (ep338) | 01:07:16 | L_main=0.1258 L_aux_1=0.1010(w=0.5) L_aux_2=0.1434(w=0.5)
[2026-06-19 14:36:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 402): 12688.6 MiB
[2026-06-19 14:37:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3036, pooled_mean_dice=0.8244, per_class=['0.8244'], iou=0.7013, recall=0.9751, precision=0.7141, vol_sim=0.8455, mcc=0.8301, min_class_dice=0.8244, coverage=[74]/88 samples
[2026-06-19 14:37:48] INFO segtask_v1.trainer.trainer: Epoch 403/1000 | LR=1.14e-06 | loss=0.2478 | val_dice=0.8244 | best=0.8271 (ep338) | 01:08:19 | L_main=0.1263 L_aux_1=0.0999(w=0.5) L_aux_2=0.1433(w=0.5)
[2026-06-19 14:37:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 403): 12688.6 MiB
[2026-06-19 14:38:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2693, pooled_mean_dice=0.8035, per_class=['0.8035'], iou=0.6716, recall=0.9767, precision=0.6825, vol_sim=0.8227, mcc=0.8117, min_class_dice=0.8035, coverage=[71]/88 samples
[2026-06-19 14:38:51] INFO segtask_v1.trainer.trainer: Epoch 404/1000 | LR=1.25e-06 | loss=0.2413 | val_dice=0.8035 | best=0.8271 (ep338) | 01:09:22 | L_main=0.1207 L_aux_1=0.0982(w=0.5) L_aux_2=0.1430(w=0.5)
[2026-06-19 14:38:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 404): 12688.6 MiB
[2026-06-19 14:39:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3154, pooled_mean_dice=0.7964, per_class=['0.7964'], iou=0.6617, recall=0.9817, precision=0.6700, vol_sim=0.8113, mcc=0.8061, min_class_dice=0.7964, coverage=[81]/88 samples
[2026-06-19 14:39:53] INFO segtask_v1.trainer.trainer: Epoch 405/1000 | LR=1.39e-06 | loss=0.2411 | val_dice=0.7964 | best=0.8271 (ep338) | 01:10:24 | L_main=0.1220 L_aux_1=0.0973(w=0.5) L_aux_2=0.1410(w=0.5)
[2026-06-19 14:39:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 405): 12688.6 MiB
[2026-06-19 14:40:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2938, pooled_mean_dice=0.8026, per_class=['0.8026'], iou=0.6703, recall=0.9811, precision=0.6791, vol_sim=0.8181, mcc=0.8115, min_class_dice=0.8026, coverage=[76]/88 samples
[2026-06-19 14:40:56] INFO segtask_v1.trainer.trainer: Epoch 406/1000 | LR=1.57e-06 | loss=0.2366 | val_dice=0.8026 | best=0.8271 (ep338) | 01:11:27 | L_main=0.1177 L_aux_1=0.0948(w=0.5) L_aux_2=0.1429(w=0.5)
[2026-06-19 14:40:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 406): 12688.6 MiB
[2026-06-19 14:41:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3325, pooled_mean_dice=0.7945, per_class=['0.7945'], iou=0.6591, recall=0.9762, precision=0.6699, vol_sim=0.8139, mcc=0.8045, min_class_dice=0.7945, coverage=[77]/88 samples
[2026-06-19 14:41:59] INFO segtask_v1.trainer.trainer: Epoch 407/1000 | LR=1.77e-06 | loss=0.2543 | val_dice=0.7945 | best=0.8271 (ep338) | 01:12:30 | L_main=0.1287 L_aux_1=0.1007(w=0.5) L_aux_2=0.1505(w=0.5)
[2026-06-19 14:41:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 407): 12688.6 MiB
[2026-06-19 14:43:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3246, pooled_mean_dice=0.8202, per_class=['0.8202'], iou=0.6953, recall=0.9784, precision=0.7061, vol_sim=0.8383, mcc=0.8262, min_class_dice=0.8202, coverage=[81]/88 samples
[2026-06-19 14:43:00] INFO segtask_v1.trainer.trainer: Epoch 408/1000 | LR=2.01e-06 | loss=0.2343 | val_dice=0.8202 | best=0.8271 (ep338) | 01:13:31 | L_main=0.1168 L_aux_1=0.0923(w=0.5) L_aux_2=0.1427(w=0.5)
[2026-06-19 14:43:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 408): 12688.6 MiB
[2026-06-19 14:44:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3530, pooled_mean_dice=0.7963, per_class=['0.7963'], iou=0.6615, recall=0.9816, precision=0.6698, vol_sim=0.8112, mcc=0.8056, min_class_dice=0.7963, coverage=[78]/88 samples
[2026-06-19 14:44:04] INFO segtask_v1.trainer.trainer: Epoch 409/1000 | LR=2.28e-06 | loss=0.2431 | val_dice=0.7963 | best=0.8271 (ep338) | 01:14:35 | L_main=0.1218 L_aux_1=0.0979(w=0.5) L_aux_2=0.1446(w=0.5)
[2026-06-19 14:44:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 409): 12688.6 MiB
[2026-06-19 14:45:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3099, pooled_mean_dice=0.8026, per_class=['0.8026'], iou=0.6703, recall=0.9723, precision=0.6833, vol_sim=0.8254, mcc=0.8109, min_class_dice=0.8026, coverage=[74]/88 samples
[2026-06-19 14:45:07] INFO segtask_v1.trainer.trainer: Epoch 410/1000 | LR=2.58e-06 | loss=0.2578 | val_dice=0.8026 | best=0.8271 (ep338) | 01:15:38 | L_main=0.1288 L_aux_1=0.1040(w=0.5) L_aux_2=0.1540(w=0.5)
[2026-06-19 14:45:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 410): 12688.6 MiB
[2026-06-19 14:46:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2925, pooled_mean_dice=0.8151, per_class=['0.8151'], iou=0.6878, recall=0.9832, precision=0.6960, vol_sim=0.8290, mcc=0.8218, min_class_dice=0.8151, coverage=[77]/88 samples
[2026-06-19 14:46:10] INFO segtask_v1.trainer.trainer: Epoch 411/1000 | LR=2.91e-06 | loss=0.2407 | val_dice=0.8151 | best=0.8271 (ep338) | 01:16:41 | L_main=0.1207 L_aux_1=0.0991(w=0.5) L_aux_2=0.1409(w=0.5)
[2026-06-19 14:46:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 411): 12688.6 MiB
[2026-06-19 14:47:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2911, pooled_mean_dice=0.7996, per_class=['0.7996'], iou=0.6661, recall=0.9785, precision=0.6760, vol_sim=0.8172, mcc=0.8088, min_class_dice=0.7996, coverage=[72]/88 samples
[2026-06-19 14:47:12] INFO segtask_v1.trainer.trainer: Epoch 412/1000 | LR=3.27e-06 | loss=0.2428 | val_dice=0.7996 | best=0.8271 (ep338) | 01:17:44 | L_main=0.1212 L_aux_1=0.0974(w=0.5) L_aux_2=0.1458(w=0.5)
[2026-06-19 14:47:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 412): 12688.6 MiB
[2026-06-19 14:48:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2956, pooled_mean_dice=0.8054, per_class=['0.8054'], iou=0.6742, recall=0.9827, precision=0.6823, vol_sim=0.8196, mcc=0.8142, min_class_dice=0.8054, coverage=[75]/88 samples
[2026-06-19 14:48:15] INFO segtask_v1.trainer.trainer: Epoch 413/1000 | LR=3.67e-06 | loss=0.2290 | val_dice=0.8054 | best=0.8271 (ep338) | 01:18:46 | L_main=0.1139 L_aux_1=0.0978(w=0.5) L_aux_2=0.1323(w=0.5)
[2026-06-19 14:48:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 413): 12688.6 MiB
[2026-06-19 14:49:17] INFO segtask_v1.trainer.validation:   Val: loss=0.3173, pooled_mean_dice=0.8154, per_class=['0.8154'], iou=0.6883, recall=0.9835, precision=0.6963, vol_sim=0.8290, mcc=0.8226, min_class_dice=0.8154, coverage=[80]/88 samples
[2026-06-19 14:49:17] INFO segtask_v1.trainer.trainer: Epoch 414/1000 | LR=4.09e-06 | loss=0.2430 | val_dice=0.8154 | best=0.8271 (ep338) | 01:19:49 | L_main=0.1206 L_aux_1=0.1009(w=0.5) L_aux_2=0.1437(w=0.5)
[2026-06-19 14:49:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 414): 12688.6 MiB
[2026-06-19 14:50:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2724, pooled_mean_dice=0.8045, per_class=['0.8045'], iou=0.6729, recall=0.9807, precision=0.6819, vol_sim=0.8203, mcc=0.8134, min_class_dice=0.8045, coverage=[70]/88 samples
[2026-06-19 14:50:20] INFO segtask_v1.trainer.trainer: Epoch 415/1000 | LR=4.55e-06 | loss=0.2397 | val_dice=0.8045 | best=0.8271 (ep338) | 01:20:51 | L_main=0.1212 L_aux_1=0.0988(w=0.5) L_aux_2=0.1381(w=0.5)
[2026-06-19 14:50:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 415): 12688.6 MiB
[2026-06-19 14:51:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3017, pooled_mean_dice=0.8273, per_class=['0.8273'], iou=0.7054, recall=0.9829, precision=0.7142, vol_sim=0.8417, mcc=0.8322, min_class_dice=0.8273, coverage=[81]/88 samples
[2026-06-19 14:51:29] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_bnorm/best_model.pth
[2026-06-19 14:51:29] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8273 at epoch 416
[2026-06-19 14:51:29] INFO segtask_v1.trainer.trainer: Epoch 416/1000 | LR=5.04e-06 | loss=0.2437 | val_dice=0.8273 | best=0.8273 (ep416) | 01:22:00 | L_main=0.1208 L_aux_1=0.1044(w=0.5) L_aux_2=0.1415(w=0.5)
[2026-06-19 14:51:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 416): 12688.6 MiB
[2026-06-19 14:52:33] INFO segtask_v1.trainer.validation:   Val: loss=0.3345, pooled_mean_dice=0.8023, per_class=['0.8023'], iou=0.6699, recall=0.9756, precision=0.6813, vol_sim=0.8224, mcc=0.8110, min_class_dice=0.8023, coverage=[77]/88 samples
[2026-06-19 14:52:33] INFO segtask_v1.trainer.trainer: Epoch 417/1000 | LR=5.56e-06 | loss=0.2542 | val_dice=0.8023 | best=0.8273 (ep416) | 01:23:04 | L_main=0.1280 L_aux_1=0.0988(w=0.5) L_aux_2=0.1536(w=0.5)
[2026-06-19 14:52:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 417): 12688.6 MiB
[2026-06-19 14:53:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2820, pooled_mean_dice=0.8074, per_class=['0.8074'], iou=0.6770, recall=0.9771, precision=0.6879, vol_sim=0.8263, mcc=0.8151, min_class_dice=0.8074, coverage=[73]/88 samples
[2026-06-19 14:53:36] INFO segtask_v1.trainer.trainer: Epoch 418/1000 | LR=6.11e-06 | loss=0.2414 | val_dice=0.8074 | best=0.8273 (ep416) | 01:24:08 | L_main=0.1204 L_aux_1=0.0986(w=0.5) L_aux_2=0.1434(w=0.5)
[2026-06-19 14:53:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 418): 12688.6 MiB
[2026-06-19 14:54:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3131, pooled_mean_dice=0.8132, per_class=['0.8132'], iou=0.6852, recall=0.9790, precision=0.6954, vol_sim=0.8307, mcc=0.8198, min_class_dice=0.8132, coverage=[77]/88 samples
[2026-06-19 14:54:39] INFO segtask_v1.trainer.trainer: Epoch 419/1000 | LR=6.69e-06 | loss=0.2424 | val_dice=0.8132 | best=0.8273 (ep416) | 01:25:10 | L_main=0.1205 L_aux_1=0.0971(w=0.5) L_aux_2=0.1468(w=0.5)
[2026-06-19 14:54:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 419): 12688.6 MiB
[2026-06-19 14:55:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3181, pooled_mean_dice=0.8089, per_class=['0.8089'], iou=0.6791, recall=0.9842, precision=0.6866, vol_sim=0.8219, mcc=0.8166, min_class_dice=0.8089, coverage=[81]/88 samples
[2026-06-19 14:55:42] INFO segtask_v1.trainer.trainer: Epoch 420/1000 | LR=7.31e-06 | loss=0.2414 | val_dice=0.8089 | best=0.8273 (ep416) | 01:26:13 | L_main=0.1203 L_aux_1=0.0967(w=0.5) L_aux_2=0.1456(w=0.5)
[2026-06-19 14:55:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 420): 12688.6 MiB
[2026-06-19 14:56:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3008, pooled_mean_dice=0.8194, per_class=['0.8194'], iou=0.6941, recall=0.9788, precision=0.7046, vol_sim=0.8371, mcc=0.8253, min_class_dice=0.8194, coverage=[75]/88 samples
[2026-06-19 14:56:46] INFO segtask_v1.trainer.trainer: Epoch 421/1000 | LR=7.95e-06 | loss=0.2331 | val_dice=0.8194 | best=0.8273 (ep416) | 01:27:18 | L_main=0.1156 L_aux_1=0.1001(w=0.5) L_aux_2=0.1349(w=0.5)
[2026-06-19 14:56:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 421): 12688.6 MiB
[2026-06-19 14:57:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3552, pooled_mean_dice=0.7967, per_class=['0.7967'], iou=0.6622, recall=0.9797, precision=0.6714, vol_sim=0.8133, mcc=0.8071, min_class_dice=0.7967, coverage=[80]/88 samples
[2026-06-19 14:57:48] INFO segtask_v1.trainer.trainer: Epoch 422/1000 | LR=8.63e-06 | loss=0.2636 | val_dice=0.7967 | best=0.8273 (ep416) | 01:28:19 | L_main=0.1306 L_aux_1=0.1051(w=0.5) L_aux_2=0.1611(w=0.5)
[2026-06-19 14:57:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 422): 12688.6 MiB
[2026-06-19 14:58:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2973, pooled_mean_dice=0.8258, per_class=['0.8258'], iou=0.7033, recall=0.9781, precision=0.7146, vol_sim=0.8443, mcc=0.8310, min_class_dice=0.8258, coverage=[77]/88 samples
[2026-06-19 14:58:50] INFO segtask_v1.trainer.trainer: Epoch 423/1000 | LR=9.33e-06 | loss=0.2519 | val_dice=0.8258 | best=0.8273 (ep416) | 01:29:21 | L_main=0.1270 L_aux_1=0.0976(w=0.5) L_aux_2=0.1524(w=0.5)
[2026-06-19 14:58:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 423): 12688.6 MiB
[2026-06-19 14:59:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2906, pooled_mean_dice=0.8065, per_class=['0.8065'], iou=0.6757, recall=0.9805, precision=0.6849, vol_sim=0.8225, mcc=0.8152, min_class_dice=0.8065, coverage=[73]/88 samples
[2026-06-19 14:59:52] INFO segtask_v1.trainer.trainer: Epoch 424/1000 | LR=1.01e-05 | loss=0.2418 | val_dice=0.8065 | best=0.8273 (ep416) | 01:30:23 | L_main=0.1219 L_aux_1=0.0992(w=0.5) L_aux_2=0.1407(w=0.5)
[2026-06-19 14:59:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 424): 12688.6 MiB
[2026-06-19 15:00:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3164, pooled_mean_dice=0.8108, per_class=['0.8108'], iou=0.6817, recall=0.9797, precision=0.6915, vol_sim=0.8276, mcc=0.8181, min_class_dice=0.8108, coverage=[73]/88 samples
[2026-06-19 15:00:55] INFO segtask_v1.trainer.trainer: Epoch 425/1000 | LR=1.08e-05 | loss=0.2463 | val_dice=0.8108 | best=0.8273 (ep416) | 01:31:26 | L_main=0.1257 L_aux_1=0.0961(w=0.5) L_aux_2=0.1452(w=0.5)
[2026-06-19 15:00:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 425): 12688.6 MiB
[2026-06-19 15:01:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3300, pooled_mean_dice=0.7867, per_class=['0.7867'], iou=0.6484, recall=0.9831, precision=0.6558, vol_sim=0.8003, mcc=0.7988, min_class_dice=0.7867, coverage=[78]/88 samples
[2026-06-19 15:01:59] INFO segtask_v1.trainer.trainer: Epoch 426/1000 | LR=1.16e-05 | loss=0.2619 | val_dice=0.7867 | best=0.8273 (ep416) | 01:32:30 | L_main=0.1320 L_aux_1=0.0982(w=0.5) L_aux_2=0.1616(w=0.5)
[2026-06-19 15:01:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 426): 12688.6 MiB
[2026-06-19 15:03:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3675, pooled_mean_dice=0.7760, per_class=['0.7760'], iou=0.6340, recall=0.9790, precision=0.6428, vol_sim=0.7927, mcc=0.7890, min_class_dice=0.7760, coverage=[76]/88 samples
[2026-06-19 15:03:00] INFO segtask_v1.trainer.trainer: Epoch 427/1000 | LR=1.25e-05 | loss=0.2504 | val_dice=0.7760 | best=0.8273 (ep416) | 01:33:31 | L_main=0.1243 L_aux_1=0.1051(w=0.5) L_aux_2=0.1471(w=0.5)
[2026-06-19 15:03:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 427): 12688.6 MiB
[2026-06-19 15:04:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3098, pooled_mean_dice=0.8192, per_class=['0.8192'], iou=0.6938, recall=0.9817, precision=0.7029, vol_sim=0.8345, mcc=0.8253, min_class_dice=0.8192, coverage=[79]/88 samples
[2026-06-19 15:04:04] INFO segtask_v1.trainer.trainer: Epoch 428/1000 | LR=1.33e-05 | loss=0.2699 | val_dice=0.8192 | best=0.8273 (ep416) | 01:34:35 | L_main=0.1353 L_aux_1=0.1029(w=0.5) L_aux_2=0.1662(w=0.5)
[2026-06-19 15:04:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 428): 12688.6 MiB
[2026-06-19 15:05:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3288, pooled_mean_dice=0.8010, per_class=['0.8010'], iou=0.6680, recall=0.9816, precision=0.6765, vol_sim=0.8160, mcc=0.8101, min_class_dice=0.8010, coverage=[73]/88 samples
[2026-06-19 15:05:07] INFO segtask_v1.trainer.trainer: Epoch 429/1000 | LR=1.42e-05 | loss=0.2493 | val_dice=0.8010 | best=0.8273 (ep416) | 01:35:38 | L_main=0.1265 L_aux_1=0.0969(w=0.5) L_aux_2=0.1489(w=0.5)
[2026-06-19 15:05:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 429): 12688.6 MiB
[2026-06-19 15:06:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2981, pooled_mean_dice=0.7929, per_class=['0.7929'], iou=0.6569, recall=0.9801, precision=0.6657, vol_sim=0.8090, mcc=0.8035, min_class_dice=0.7929, coverage=[75]/88 samples
[2026-06-19 15:06:08] INFO segtask_v1.trainer.trainer: Epoch 430/1000 | LR=1.52e-05 | loss=0.2435 | val_dice=0.7929 | best=0.8273 (ep416) | 01:36:40 | L_main=0.1237 L_aux_1=0.0993(w=0.5) L_aux_2=0.1404(w=0.5)
[2026-06-19 15:06:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 430): 12688.6 MiB
[2026-06-19 15:07:13] INFO segtask_v1.trainer.validation:   Val: loss=0.2817, pooled_mean_dice=0.8216, per_class=['0.8216'], iou=0.6972, recall=0.9852, precision=0.7046, vol_sim=0.8340, mcc=0.8280, min_class_dice=0.8216, coverage=[78]/88 samples
[2026-06-19 15:07:13] INFO segtask_v1.trainer.trainer: Epoch 431/1000 | LR=1.61e-05 | loss=0.2506 | val_dice=0.8216 | best=0.8273 (ep416) | 01:37:44 | L_main=0.1274 L_aux_1=0.0927(w=0.5) L_aux_2=0.1539(w=0.5)
[2026-06-19 15:07:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 431): 12688.6 MiB
[2026-06-19 15:08:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2903, pooled_mean_dice=0.8106, per_class=['0.8106'], iou=0.6815, recall=0.9838, precision=0.6892, vol_sim=0.8239, mcc=0.8189, min_class_dice=0.8106, coverage=[72]/88 samples
[2026-06-19 15:08:16] INFO segtask_v1.trainer.trainer: Epoch 432/1000 | LR=1.71e-05 | loss=0.2293 | val_dice=0.8106 | best=0.8273 (ep416) | 01:38:47 | L_main=0.1145 L_aux_1=0.0924(w=0.5) L_aux_2=0.1372(w=0.5)
[2026-06-19 15:08:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 432): 12688.6 MiB
[2026-06-19 15:09:19] INFO segtask_v1.trainer.validation:   Val: loss=0.3207, pooled_mean_dice=0.7977, per_class=['0.7977'], iou=0.6635, recall=0.9795, precision=0.6728, vol_sim=0.8144, mcc=0.8070, min_class_dice=0.7977, coverage=[71]/88 samples
[2026-06-19 15:09:19] INFO segtask_v1.trainer.trainer: Epoch 433/1000 | LR=1.81e-05 | loss=0.2378 | val_dice=0.7977 | best=0.8273 (ep416) | 01:39:50 | L_main=0.1177 L_aux_1=0.0975(w=0.5) L_aux_2=0.1426(w=0.5)
[2026-06-19 15:09:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 433): 12688.6 MiB
[2026-06-19 15:10:21] INFO segtask_v1.trainer.validation:   Val: loss=0.3372, pooled_mean_dice=0.7784, per_class=['0.7784'], iou=0.6372, recall=0.9818, precision=0.6448, vol_sim=0.7928, mcc=0.7917, min_class_dice=0.7784, coverage=[76]/88 samples
[2026-06-19 15:10:21] INFO segtask_v1.trainer.trainer: Epoch 434/1000 | LR=1.92e-05 | loss=0.2517 | val_dice=0.7784 | best=0.8273 (ep416) | 01:40:52 | L_main=0.1287 L_aux_1=0.1010(w=0.5) L_aux_2=0.1450(w=0.5)
[2026-06-19 15:10:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 434): 12688.6 MiB
[2026-06-19 15:11:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3042, pooled_mean_dice=0.8059, per_class=['0.8059'], iou=0.6749, recall=0.9811, precision=0.6838, vol_sim=0.8215, mcc=0.8137, min_class_dice=0.8059, coverage=[81]/88 samples
[2026-06-19 15:11:23] INFO segtask_v1.trainer.trainer: Epoch 435/1000 | LR=2.02e-05 | loss=0.2580 | val_dice=0.8059 | best=0.8273 (ep416) | 01:41:55 | L_main=0.1291 L_aux_1=0.1050(w=0.5) L_aux_2=0.1530(w=0.5)
[2026-06-19 15:11:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 435): 12688.6 MiB
[2026-06-19 15:12:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3046, pooled_mean_dice=0.8048, per_class=['0.8048'], iou=0.6733, recall=0.9827, precision=0.6814, vol_sim=0.8189, mcc=0.8140, min_class_dice=0.8048, coverage=[69]/88 samples
[2026-06-19 15:12:26] INFO segtask_v1.trainer.trainer: Epoch 436/1000 | LR=2.13e-05 | loss=0.2387 | val_dice=0.8048 | best=0.8273 (ep416) | 01:42:57 | L_main=0.1194 L_aux_1=0.0968(w=0.5) L_aux_2=0.1416(w=0.5)
[2026-06-19 15:12:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 436): 12688.6 MiB
[2026-06-19 15:13:29] INFO segtask_v1.trainer.validation:   Val: loss=0.2898, pooled_mean_dice=0.8184, per_class=['0.8184'], iou=0.6926, recall=0.9830, precision=0.7010, vol_sim=0.8326, mcc=0.8260, min_class_dice=0.8184, coverage=[73]/88 samples
[2026-06-19 15:13:29] INFO segtask_v1.trainer.trainer: Epoch 437/1000 | LR=2.25e-05 | loss=0.2273 | val_dice=0.8184 | best=0.8273 (ep416) | 01:44:00 | L_main=0.1136 L_aux_1=0.0968(w=0.5) L_aux_2=0.1306(w=0.5)
[2026-06-19 15:13:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 437): 12688.6 MiB
[2026-06-19 15:14:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2894, pooled_mean_dice=0.8175, per_class=['0.8175'], iou=0.6913, recall=0.9850, precision=0.6987, vol_sim=0.8300, mcc=0.8248, min_class_dice=0.8175, coverage=[73]/88 samples
[2026-06-19 15:14:31] INFO segtask_v1.trainer.trainer: Epoch 438/1000 | LR=2.36e-05 | loss=0.2477 | val_dice=0.8175 | best=0.8273 (ep416) | 01:45:02 | L_main=0.1252 L_aux_1=0.1095(w=0.5) L_aux_2=0.1356(w=0.5)
[2026-06-19 15:14:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 438): 12688.6 MiB
[2026-06-19 15:15:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2994, pooled_mean_dice=0.8047, per_class=['0.8047'], iou=0.6732, recall=0.9781, precision=0.6835, vol_sim=0.8227, mcc=0.8123, min_class_dice=0.8047, coverage=[79]/88 samples
[2026-06-19 15:15:33] INFO segtask_v1.trainer.trainer: Epoch 439/1000 | LR=2.48e-05 | loss=0.2336 | val_dice=0.8047 | best=0.8273 (ep416) | 01:46:04 | L_main=0.1158 L_aux_1=0.0947(w=0.5) L_aux_2=0.1408(w=0.5)
[2026-06-19 15:15:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 439): 12688.6 MiB
[2026-06-19 15:16:35] INFO segtask_v1.trainer.validation:   Val: loss=0.3542, pooled_mean_dice=0.7841, per_class=['0.7841'], iou=0.6449, recall=0.9802, precision=0.6534, vol_sim=0.8000, mcc=0.7964, min_class_dice=0.7841, coverage=[79]/88 samples
[2026-06-19 15:16:35] INFO segtask_v1.trainer.trainer: Epoch 440/1000 | LR=2.61e-05 | loss=0.2538 | val_dice=0.7841 | best=0.8273 (ep416) | 01:47:06 | L_main=0.1280 L_aux_1=0.0985(w=0.5) L_aux_2=0.1530(w=0.5)
[2026-06-19 15:16:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 440): 12688.6 MiB
[2026-06-19 15:17:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3358, pooled_mean_dice=0.8292, per_class=['0.8292'], iou=0.7082, recall=0.9820, precision=0.7175, vol_sim=0.8443, mcc=0.8343, min_class_dice=0.8292, coverage=[77]/88 samples
[2026-06-19 15:17:44] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet_bnorm/best_model.pth
[2026-06-19 15:17:44] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8292 at epoch 441
[2026-06-19 15:17:44] INFO segtask_v1.trainer.trainer: Epoch 441/1000 | LR=2.73e-05 | loss=0.2358 | val_dice=0.8292 | best=0.8292 (ep441) | 01:48:16 | L_main=0.1176 L_aux_1=0.0963(w=0.5) L_aux_2=0.1402(w=0.5)
[2026-06-19 15:17:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 441): 12688.6 MiB
[2026-06-19 15:18:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2913, pooled_mean_dice=0.8153, per_class=['0.8153'], iou=0.6881, recall=0.9786, precision=0.6987, vol_sim=0.8331, mcc=0.8225, min_class_dice=0.8153, coverage=[72]/88 samples
[2026-06-19 15:18:47] INFO segtask_v1.trainer.trainer: Epoch 442/1000 | LR=2.86e-05 | loss=0.2650 | val_dice=0.8153 | best=0.8292 (ep441) | 01:49:18 | L_main=0.1358 L_aux_1=0.1020(w=0.5) L_aux_2=0.1564(w=0.5)
[2026-06-19 15:18:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 442): 12688.6 MiB
[2026-06-19 15:19:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2772, pooled_mean_dice=0.8237, per_class=['0.8237'], iou=0.7003, recall=0.9840, precision=0.7083, vol_sim=0.8371, mcc=0.8302, min_class_dice=0.8237, coverage=[71]/88 samples
[2026-06-19 15:19:50] INFO segtask_v1.trainer.trainer: Epoch 443/1000 | LR=2.99e-05 | loss=0.2462 | val_dice=0.8237 | best=0.8292 (ep441) | 01:50:21 | L_main=0.1231 L_aux_1=0.1033(w=0.5) L_aux_2=0.1430(w=0.5)
[2026-06-19 15:19:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 443): 12688.6 MiB
[2026-06-19 15:20:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3321, pooled_mean_dice=0.8021, per_class=['0.8021'], iou=0.6695, recall=0.9791, precision=0.6792, vol_sim=0.8192, mcc=0.8103, min_class_dice=0.8021, coverage=[80]/88 samples
[2026-06-19 15:20:53] INFO segtask_v1.trainer.trainer: Epoch 444/1000 | LR=3.13e-05 | loss=0.2529 | val_dice=0.8021 | best=0.8292 (ep441) | 01:51:25 | L_main=0.1268 L_aux_1=0.1037(w=0.5) L_aux_2=0.1485(w=0.5)
[2026-06-19 15:20:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 444): 12688.6 MiB
[2026-06-19 15:21:56] INFO segtask_v1.trainer.validation:   Val: loss=0.3113, pooled_mean_dice=0.8181, per_class=['0.8181'], iou=0.6922, recall=0.9759, precision=0.7042, vol_sim=0.8383, mcc=0.8244, min_class_dice=0.8181, coverage=[73]/88 samples
[2026-06-19 15:21:56] INFO segtask_v1.trainer.trainer: Epoch 445/1000 | LR=3.27e-05 | loss=0.2564 | val_dice=0.8181 | best=0.8292 (ep441) | 01:52:27 | L_main=0.1286 L_aux_1=0.1004(w=0.5) L_aux_2=0.1553(w=0.5)
[2026-06-19 15:21:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 445): 12688.6 MiB
[2026-06-19 15:22:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3110, pooled_mean_dice=0.8137, per_class=['0.8137'], iou=0.6860, recall=0.9824, precision=0.6945, vol_sim=0.8283, mcc=0.8213, min_class_dice=0.8137, coverage=[74]/88 samples
[2026-06-19 15:22:59] INFO segtask_v1.trainer.trainer: Epoch 446/1000 | LR=3.41e-05 | loss=0.2424 | val_dice=0.8137 | best=0.8292 (ep441) | 01:53:30 | L_main=0.1217 L_aux_1=0.0949(w=0.5) L_aux_2=0.1463(w=0.5)
[2026-06-19 15:22:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 446): 12688.6 MiB
[2026-06-19 15:24:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2969, pooled_mean_dice=0.8163, per_class=['0.8163'], iou=0.6896, recall=0.9815, precision=0.6986, vol_sim=0.8316, mcc=0.8226, min_class_dice=0.8163, coverage=[78]/88 samples
[2026-06-19 15:24:01] INFO segtask_v1.trainer.trainer: Epoch 447/1000 | LR=3.55e-05 | loss=0.2458 | val_dice=0.8163 | best=0.8292 (ep441) | 01:54:32 | L_main=0.1258 L_aux_1=0.0985(w=0.5) L_aux_2=0.1415(w=0.5)
[2026-06-19 15:24:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 447): 12688.6 MiB
[2026-06-19 15:25:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2942, pooled_mean_dice=0.8100, per_class=['0.8100'], iou=0.6807, recall=0.9832, precision=0.6887, vol_sim=0.8239, mcc=0.8175, min_class_dice=0.8100, coverage=[76]/88 samples
[2026-06-19 15:25:02] INFO segtask_v1.trainer.trainer: Epoch 448/1000 | LR=3.70e-05 | loss=0.2325 | val_dice=0.8100 | best=0.8292 (ep441) | 01:55:33 | L_main=0.1153 L_aux_1=0.0897(w=0.5) L_aux_2=0.1447(w=0.5)
[2026-06-19 15:25:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 448): 12688.6 MiB
[2026-06-19 15:26:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3254, pooled_mean_dice=0.8161, per_class=['0.8161'], iou=0.6893, recall=0.9786, precision=0.6999, vol_sim=0.8339, mcc=0.8223, min_class_dice=0.8161, coverage=[80]/88 samples
[2026-06-19 15:26:05] INFO segtask_v1.trainer.trainer: Epoch 449/1000 | LR=3.85e-05 | loss=0.2531 | val_dice=0.8161 | best=0.8292 (ep441) | 01:56:36 | L_main=0.1260 L_aux_1=0.1034(w=0.5) L_aux_2=0.1508(w=0.5)
[2026-06-19 15:26:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 449): 12688.6 MiB
[2026-06-19 15:27:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2782, pooled_mean_dice=0.8234, per_class=['0.8234'], iou=0.6998, recall=0.9820, precision=0.7089, vol_sim=0.8385, mcc=0.8281, min_class_dice=0.8234, coverage=[76]/88 samples
[2026-06-19 15:27:08] INFO segtask_v1.trainer.trainer: Epoch 450/1000 | LR=4.00e-05 | loss=0.2280 | val_dice=0.8234 | best=0.8292 (ep441) | 01:57:39 | L_main=0.1123 L_aux_1=0.0988(w=0.5) L_aux_2=0.1326(w=0.5)
[2026-06-19 15:27:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 450): 12688.6 MiB
[2026-06-19 15:28:12] INFO segtask_v1.trainer.validation:   Val: loss=0.3004, pooled_mean_dice=0.8056, per_class=['0.8056'], iou=0.6745, recall=0.9798, precision=0.6840, vol_sim=0.8223, mcc=0.8135, min_class_dice=0.8056, coverage=[76]/88 samples
[2026-06-19 15:28:12] INFO segtask_v1.trainer.trainer: Epoch 451/1000 | LR=4.15e-05 | loss=0.2425 | val_dice=0.8056 | best=0.8292 (ep441) | 01:58:43 | L_main=0.1199 L_aux_1=0.1000(w=0.5) L_aux_2=0.1451(w=0.5)
[2026-06-19 15:28:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 451): 12688.6 MiB
[2026-06-19 15:29:14] INFO segtask_v1.trainer.validation:   Val: loss=0.3087, pooled_mean_dice=0.8106, per_class=['0.8106'], iou=0.6816, recall=0.9846, precision=0.6889, vol_sim=0.8233, mcc=0.8187, min_class_dice=0.8106, coverage=[78]/88 samples
[2026-06-19 15:29:14] INFO segtask_v1.trainer.trainer: Epoch 452/1000 | LR=4.31e-05 | loss=0.2380 | val_dice=0.8106 | best=0.8292 (ep441) | 01:59:45 | L_main=0.1199 L_aux_1=0.0973(w=0.5) L_aux_2=0.1389(w=0.5)
[2026-06-19 15:29:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 452): 12688.6 MiB
[2026-06-19 15:30:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2984, pooled_mean_dice=0.8075, per_class=['0.8075'], iou=0.6771, recall=0.9809, precision=0.6861, vol_sim=0.8232, mcc=0.8155, min_class_dice=0.8075, coverage=[77]/88 samples
[2026-06-19 15:30:16] INFO segtask_v1.trainer.trainer: Epoch 453/1000 | LR=4.47e-05 | loss=0.2424 | val_dice=0.8075 | best=0.8292 (ep441) | 02:00:47 | L_main=0.1209 L_aux_1=0.0943(w=0.5) L_aux_2=0.1487(w=0.5)
[2026-06-19 15:30:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 453): 12688.6 MiB
[2026-06-19 15:31:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2841, pooled_mean_dice=0.8181, per_class=['0.8181'], iou=0.6921, recall=0.9867, precision=0.6986, vol_sim=0.8291, mcc=0.8249, min_class_dice=0.8181, coverage=[73]/88 samples
[2026-06-19 15:31:20] INFO segtask_v1.trainer.trainer: Epoch 454/1000 | LR=4.64e-05 | loss=0.2533 | val_dice=0.8181 | best=0.8292 (ep441) | 02:01:51 | L_main=0.1250 L_aux_1=0.1007(w=0.5) L_aux_2=0.1559(w=0.5)
[2026-06-19 15:31:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 454): 12688.6 MiB
[2026-06-19 15:32:21] INFO segtask_v1.trainer.validation:   Val: loss=0.3144, pooled_mean_dice=0.8123, per_class=['0.8123'], iou=0.6839, recall=0.9796, precision=0.6938, vol_sim=0.8292, mcc=0.8197, min_class_dice=0.8123, coverage=[71]/88 samples
[2026-06-19 15:32:21] INFO segtask_v1.trainer.trainer: Epoch 455/1000 | LR=4.80e-05 | loss=0.2369 | val_dice=0.8123 | best=0.8292 (ep441) | 02:02:52 | L_main=0.1180 L_aux_1=0.0963(w=0.5) L_aux_2=0.1416(w=0.5)
[2026-06-19 15:32:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 455): 12688.6 MiB
[2026-06-19 15:33:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3246, pooled_mean_dice=0.8273, per_class=['0.8273'], iou=0.7055, recall=0.9828, precision=0.7143, vol_sim=0.8418, mcc=0.8329, min_class_dice=0.8273, coverage=[79]/88 samples
[2026-06-19 15:33:23] INFO segtask_v1.trainer.trainer: Epoch 456/1000 | LR=4.97e-05 | loss=0.2376 | val_dice=0.8273 | best=0.8292 (ep441) | 02:03:55 | L_main=0.1189 L_aux_1=0.0951(w=0.5) L_aux_2=0.1422(w=0.5)
[2026-06-19 15:33:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 456): 12688.6 MiB
[2026-06-19 15:34:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2992, pooled_mean_dice=0.8054, per_class=['0.8054'], iou=0.6742, recall=0.9813, precision=0.6830, vol_sim=0.8208, mcc=0.8145, min_class_dice=0.8054, coverage=[72]/88 samples
[2026-06-19 15:34:26] INFO segtask_v1.trainer.trainer: Epoch 457/1000 | LR=5.15e-05 | loss=0.2502 | val_dice=0.8054 | best=0.8292 (ep441) | 02:04:58 | L_main=0.1252 L_aux_1=0.1023(w=0.5) L_aux_2=0.1477(w=0.5)
[2026-06-19 15:34:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 457): 12688.6 MiB
[2026-06-19 15:35:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2833, pooled_mean_dice=0.8030, per_class=['0.8030'], iou=0.6708, recall=0.9757, precision=0.6822, vol_sim=0.8230, mcc=0.8110, min_class_dice=0.8030, coverage=[74]/88 samples
[2026-06-19 15:35:30] INFO segtask_v1.trainer.trainer: Epoch 458/1000 | LR=5.32e-05 | loss=0.2432 | val_dice=0.8030 | best=0.8292 (ep441) | 02:06:01 | L_main=0.1214 L_aux_1=0.0878(w=0.5) L_aux_2=0.1558(w=0.5)
[2026-06-19 15:35:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 458): 12688.6 MiB
[2026-06-19 15:36:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3058, pooled_mean_dice=0.8168, per_class=['0.8168'], iou=0.6903, recall=0.9795, precision=0.7004, vol_sim=0.8338, mcc=0.8241, min_class_dice=0.8168, coverage=[74]/88 samples
[2026-06-19 15:36:34] INFO segtask_v1.trainer.trainer: Epoch 459/1000 | LR=5.50e-05 | loss=0.2522 | val_dice=0.8168 | best=0.8292 (ep441) | 02:07:05 | L_main=0.1263 L_aux_1=0.1047(w=0.5) L_aux_2=0.1470(w=0.5)
[2026-06-19 15:36:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 459): 12688.6 MiB
[2026-06-19 15:37:38] INFO segtask_v1.trainer.validation:   Val: loss=0.3170, pooled_mean_dice=0.8231, per_class=['0.8231'], iou=0.6993, recall=0.9830, precision=0.7079, vol_sim=0.8373, mcc=0.8297, min_class_dice=0.8231, coverage=[77]/88 samples
[2026-06-19 15:37:38] INFO segtask_v1.trainer.trainer: Epoch 460/1000 | LR=5.68e-05 | loss=0.2620 | val_dice=0.8231 | best=0.8292 (ep441) | 02:08:09 | L_main=0.1321 L_aux_1=0.0984(w=0.5) L_aux_2=0.1614(w=0.5)
[2026-06-19 15:37:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 460): 12688.6 MiB
[2026-06-19 15:38:40] INFO segtask_v1.trainer.validation:   Val: loss=0.2917, pooled_mean_dice=0.8096, per_class=['0.8096'], iou=0.6802, recall=0.9808, precision=0.6893, vol_sim=0.8255, mcc=0.8164, min_class_dice=0.8096, coverage=[79]/88 samples
[2026-06-19 15:38:40] INFO segtask_v1.trainer.trainer: Epoch 461/1000 | LR=5.86e-05 | loss=0.2315 | val_dice=0.8096 | best=0.8292 (ep441) | 02:09:11 | L_main=0.1180 L_aux_1=0.0928(w=0.5) L_aux_2=0.1342(w=0.5)
[2026-06-19 15:38:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 461): 12688.6 MiB
[2026-06-19 15:39:43] INFO segtask_v1.trainer.validation:   Val: loss=0.3320, pooled_mean_dice=0.8090, per_class=['0.8090'], iou=0.6793, recall=0.9774, precision=0.6902, vol_sim=0.8277, mcc=0.8158, min_class_dice=0.8090, coverage=[80]/88 samples
[2026-06-19 15:39:43] INFO segtask_v1.trainer.trainer: Epoch 462/1000 | LR=6.05e-05 | loss=0.2478 | val_dice=0.8090 | best=0.8292 (ep441) | 02:10:14 | L_main=0.1222 L_aux_1=0.0983(w=0.5) L_aux_2=0.1530(w=0.5)
[2026-06-19 15:39:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 462): 12688.6 MiB
[2026-06-19 15:40:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3242, pooled_mean_dice=0.8150, per_class=['0.8150'], iou=0.6877, recall=0.9811, precision=0.6970, vol_sim=0.8307, mcc=0.8225, min_class_dice=0.8150, coverage=[75]/88 samples
[2026-06-19 15:40:44] INFO segtask_v1.trainer.trainer: Epoch 463/1000 | LR=6.24e-05 | loss=0.2512 | val_dice=0.8150 | best=0.8292 (ep441) | 02:11:16 | L_main=0.1254 L_aux_1=0.0995(w=0.5) L_aux_2=0.1522(w=0.5)
[2026-06-19 15:40:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 463): 12688.6 MiB
[2026-06-19 15:41:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2922, pooled_mean_dice=0.8228, per_class=['0.8228'], iou=0.6989, recall=0.9794, precision=0.7093, vol_sim=0.8401, mcc=0.8288, min_class_dice=0.8228, coverage=[74]/88 samples
[2026-06-19 15:41:48] INFO segtask_v1.trainer.trainer: Epoch 464/1000 | LR=6.43e-05 | loss=0.2418 | val_dice=0.8228 | best=0.8292 (ep441) | 02:12:19 | L_main=0.1207 L_aux_1=0.0929(w=0.5) L_aux_2=0.1492(w=0.5)
[2026-06-19 15:41:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 464): 12688.6 MiB
[2026-06-19 15:42:50] INFO segtask_v1.trainer.validation:   Val: loss=0.3327, pooled_mean_dice=0.7978, per_class=['0.7978'], iou=0.6637, recall=0.9780, precision=0.6737, vol_sim=0.8157, mcc=0.8076, min_class_dice=0.7978, coverage=[78]/88 samples
[2026-06-19 15:42:50] INFO segtask_v1.trainer.trainer: Epoch 465/1000 | LR=6.63e-05 | loss=0.2692 | val_dice=0.7978 | best=0.8292 (ep441) | 02:13:21 | L_main=0.1349 L_aux_1=0.0975(w=0.5) L_aux_2=0.1711(w=0.5)
[2026-06-19 15:42:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 465): 12688.6 MiB
[2026-06-19 15:43:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3341, pooled_mean_dice=0.8091, per_class=['0.8091'], iou=0.6793, recall=0.9794, precision=0.6892, vol_sim=0.8260, mcc=0.8171, min_class_dice=0.8091, coverage=[78]/88 samples
[2026-06-19 15:43:54] INFO segtask_v1.trainer.trainer: Epoch 466/1000 | LR=6.83e-05 | loss=0.2568 | val_dice=0.8091 | best=0.8292 (ep441) | 02:14:25 | L_main=0.1288 L_aux_1=0.1002(w=0.5) L_aux_2=0.1560(w=0.5)
[2026-06-19 15:43:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 466): 12688.6 MiB
[2026-06-19 15:44:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3271, pooled_mean_dice=0.8111, per_class=['0.8111'], iou=0.6822, recall=0.9819, precision=0.6909, vol_sim=0.8260, mcc=0.8194, min_class_dice=0.8111, coverage=[80]/88 samples
[2026-06-19 15:44:57] INFO segtask_v1.trainer.trainer: Epoch 467/1000 | LR=7.03e-05 | loss=0.2456 | val_dice=0.8111 | best=0.8292 (ep441) | 02:15:28 | L_main=0.1236 L_aux_1=0.1051(w=0.5) L_aux_2=0.1388(w=0.5)
[2026-06-19 15:44:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 467): 12688.6 MiB
[2026-06-19 15:46:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2911, pooled_mean_dice=0.8205, per_class=['0.8205'], iou=0.6957, recall=0.9821, precision=0.7046, vol_sim=0.8355, mcc=0.8266, min_class_dice=0.8205, coverage=[75]/88 samples
[2026-06-19 15:46:00] INFO segtask_v1.trainer.trainer: Epoch 468/1000 | LR=7.23e-05 | loss=0.2417 | val_dice=0.8205 | best=0.8292 (ep441) | 02:16:31 | L_main=0.1201 L_aux_1=0.0970(w=0.5) L_aux_2=0.1460(w=0.5)
[2026-06-19 15:46:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 468): 12688.6 MiB
[2026-06-19 15:47:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3034, pooled_mean_dice=0.8135, per_class=['0.8135'], iou=0.6856, recall=0.9779, precision=0.6964, vol_sim=0.8318, mcc=0.8204, min_class_dice=0.8135, coverage=[77]/88 samples
[2026-06-19 15:47:01] INFO segtask_v1.trainer.trainer: Epoch 469/1000 | LR=7.43e-05 | loss=0.2564 | val_dice=0.8135 | best=0.8292 (ep441) | 02:17:32 | L_main=0.1251 L_aux_1=0.1054(w=0.5) L_aux_2=0.1572(w=0.5)
[2026-06-19 15:47:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 469): 12688.6 MiB
[2026-06-19 15:48:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2639, pooled_mean_dice=0.8176, per_class=['0.8176'], iou=0.6914, recall=0.9821, precision=0.7002, vol_sim=0.8324, mcc=0.8236, min_class_dice=0.8176, coverage=[78]/88 samples
[2026-06-19 15:48:04] INFO segtask_v1.trainer.trainer: Epoch 470/1000 | LR=7.64e-05 | loss=0.2388 | val_dice=0.8176 | best=0.8292 (ep441) | 02:18:35 | L_main=0.1183 L_aux_1=0.0960(w=0.5) L_aux_2=0.1449(w=0.5)
[2026-06-19 15:48:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 470): 12688.6 MiB
[2026-06-19 15:49:08] INFO segtask_v1.trainer.validation:   Val: loss=0.3104, pooled_mean_dice=0.8036, per_class=['0.8036'], iou=0.6716, recall=0.9806, precision=0.6806, vol_sim=0.8194, mcc=0.8125, min_class_dice=0.8036, coverage=[72]/88 samples
[2026-06-19 15:49:08] INFO segtask_v1.trainer.trainer: Epoch 471/1000 | LR=7.85e-05 | loss=0.2259 | val_dice=0.8036 | best=0.8292 (ep441) | 02:19:39 | L_main=0.1124 L_aux_1=0.0870(w=0.5) L_aux_2=0.1400(w=0.5)
[2026-06-19 15:49:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 471): 12688.6 MiB
[2026-06-19 15:50:11] INFO segtask_v1.trainer.validation:   Val: loss=0.2944, pooled_mean_dice=0.8134, per_class=['0.8134'], iou=0.6855, recall=0.9814, precision=0.6946, vol_sim=0.8288, mcc=0.8203, min_class_dice=0.8134, coverage=[78]/88 samples
[2026-06-19 15:50:11] INFO segtask_v1.trainer.trainer: Epoch 472/1000 | LR=8.07e-05 | loss=0.2528 | val_dice=0.8134 | best=0.8292 (ep441) | 02:20:42 | L_main=0.1286 L_aux_1=0.0959(w=0.5) L_aux_2=0.1523(w=0.5)
[2026-06-19 15:50:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 472): 12688.6 MiB
[2026-06-19 15:51:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3331, pooled_mean_dice=0.7948, per_class=['0.7948'], iou=0.6595, recall=0.9804, precision=0.6683, vol_sim=0.8107, mcc=0.8045, min_class_dice=0.7948, coverage=[81]/88 samples
[2026-06-19 15:51:13] INFO segtask_v1.trainer.trainer: Epoch 473/1000 | LR=8.29e-05 | loss=0.2633 | val_dice=0.7948 | best=0.8292 (ep441) | 02:21:44 | L_main=0.1311 L_aux_1=0.1062(w=0.5) L_aux_2=0.1582(w=0.5)
[2026-06-19 15:51:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 473): 12688.6 MiB
[2026-06-19 15:52:17] INFO segtask_v1.trainer.validation:   Val: loss=0.3141, pooled_mean_dice=0.8107, per_class=['0.8107'], iou=0.6816, recall=0.9842, precision=0.6892, vol_sim=0.8237, mcc=0.8189, min_class_dice=0.8107, coverage=[76]/88 samples
[2026-06-19 15:52:17] INFO segtask_v1.trainer.trainer: Epoch 474/1000 | LR=8.50e-05 | loss=0.2416 | val_dice=0.8107 | best=0.8292 (ep441) | 02:22:48 | L_main=0.1200 L_aux_1=0.0935(w=0.5) L_aux_2=0.1497(w=0.5)
[2026-06-19 15:52:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 474): 12688.6 MiB
[2026-06-19 15:53:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2883, pooled_mean_dice=0.8001, per_class=['0.8001'], iou=0.6668, recall=0.9797, precision=0.6761, vol_sim=0.8167, mcc=0.8090, min_class_dice=0.8001, coverage=[73]/88 samples
[2026-06-19 15:53:20] INFO segtask_v1.trainer.trainer: Epoch 475/1000 | LR=8.73e-05 | loss=0.2400 | val_dice=0.8001 | best=0.8292 (ep441) | 02:23:51 | L_main=0.1210 L_aux_1=0.1001(w=0.5) L_aux_2=0.1378(w=0.5)
[2026-06-19 15:53:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 475): 12688.6 MiB
[2026-06-19 15:54:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2954, pooled_mean_dice=0.7944, per_class=['0.7944'], iou=0.6590, recall=0.9854, precision=0.6655, vol_sim=0.8062, mcc=0.8051, min_class_dice=0.7944, coverage=[76]/88 samples
[2026-06-19 15:54:22] INFO segtask_v1.trainer.trainer: Epoch 476/1000 | LR=8.95e-05 | loss=0.2428 | val_dice=0.7944 | best=0.8292 (ep441) | 02:24:53 | L_main=0.1204 L_aux_1=0.0902(w=0.5) L_aux_2=0.1547(w=0.5)
[2026-06-19 15:54:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 476): 12688.6 MiB
[2026-06-19 15:55:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3252, pooled_mean_dice=0.8142, per_class=['0.8142'], iou=0.6867, recall=0.9827, precision=0.6951, vol_sim=0.8286, mcc=0.8229, min_class_dice=0.8142, coverage=[70]/88 samples
[2026-06-19 15:55:25] INFO segtask_v1.trainer.trainer: Epoch 477/1000 | LR=9.18e-05 | loss=0.2503 | val_dice=0.8142 | best=0.8292 (ep441) | 02:25:56 | L_main=0.1250 L_aux_1=0.0983(w=0.5) L_aux_2=0.1522(w=0.5)
[2026-06-19 15:55:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 477): 12688.6 MiB
[2026-06-19 15:56:27] INFO segtask_v1.trainer.validation:   Val: loss=0.3324, pooled_mean_dice=0.8111, per_class=['0.8111'], iou=0.6823, recall=0.9852, precision=0.6893, vol_sim=0.8233, mcc=0.8193, min_class_dice=0.8111, coverage=[79]/88 samples
[2026-06-19 15:56:27] INFO segtask_v1.trainer.trainer: Epoch 478/1000 | LR=9.41e-05 | loss=0.2447 | val_dice=0.8111 | best=0.8292 (ep441) | 02:26:58 | L_main=0.1228 L_aux_1=0.1020(w=0.5) L_aux_2=0.1417(w=0.5)
[2026-06-19 15:56:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 478): 12688.6 MiB
[2026-06-19 15:57:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2896, pooled_mean_dice=0.8079, per_class=['0.8079'], iou=0.6777, recall=0.9848, precision=0.6849, vol_sim=0.8204, mcc=0.8163, min_class_dice=0.8079, coverage=[76]/88 samples
[2026-06-19 15:57:30] INFO segtask_v1.trainer.trainer: Epoch 479/1000 | LR=9.64e-05 | loss=0.2607 | val_dice=0.8079 | best=0.8292 (ep441) | 02:28:01 | L_main=0.1310 L_aux_1=0.1052(w=0.5) L_aux_2=0.1543(w=0.5)
[2026-06-19 15:57:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 479): 12688.6 MiB
[2026-06-19 15:58:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3640, pooled_mean_dice=0.8060, per_class=['0.8060'], iou=0.6750, recall=0.9776, precision=0.6856, vol_sim=0.8245, mcc=0.8145, min_class_dice=0.8060, coverage=[78]/88 samples
[2026-06-19 15:58:31] INFO segtask_v1.trainer.trainer: Epoch 480/1000 | LR=9.87e-05 | loss=0.2569 | val_dice=0.8060 | best=0.8292 (ep441) | 02:29:02 | L_main=0.1293 L_aux_1=0.1008(w=0.5) L_aux_2=0.1544(w=0.5)
[2026-06-19 15:58:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 480): 12688.6 MiB
[2026-06-19 15:59:34] INFO segtask_v1.trainer.validation:   Val: loss=0.2988, pooled_mean_dice=0.8252, per_class=['0.8252'], iou=0.7024, recall=0.9832, precision=0.7109, vol_sim=0.8392, mcc=0.8308, min_class_dice=0.8252, coverage=[77]/88 samples
[2026-06-19 15:59:34] INFO segtask_v1.trainer.trainer: Epoch 481/1000 | LR=1.01e-04 | loss=0.2618 | val_dice=0.8252 | best=0.8292 (ep441) | 02:30:05 | L_main=0.1316 L_aux_1=0.1044(w=0.5) L_aux_2=0.1559(w=0.5)
[2026-06-19 15:59:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 481): 12688.6 MiB
[2026-06-19 16:00:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2947, pooled_mean_dice=0.7986, per_class=['0.7986'], iou=0.6647, recall=0.9797, precision=0.6740, vol_sim=0.8151, mcc=0.8077, min_class_dice=0.7986, coverage=[73]/88 samples
[2026-06-19 16:00:37] INFO segtask_v1.trainer.trainer: Epoch 482/1000 | LR=1.04e-04 | loss=0.2472 | val_dice=0.7986 | best=0.8292 (ep441) | 02:31:08 | L_main=0.1240 L_aux_1=0.1006(w=0.5) L_aux_2=0.1460(w=0.5)
[2026-06-19 16:00:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 482): 12688.6 MiB
[2026-06-19 16:01:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3182, pooled_mean_dice=0.8016, per_class=['0.8016'], iou=0.6689, recall=0.9837, precision=0.6764, vol_sim=0.8149, mcc=0.8108, min_class_dice=0.8016, coverage=[71]/88 samples
[2026-06-19 16:01:39] INFO segtask_v1.trainer.trainer: Epoch 483/1000 | LR=1.06e-04 | loss=0.2568 | val_dice=0.8016 | best=0.8292 (ep441) | 02:32:10 | L_main=0.1265 L_aux_1=0.1032(w=0.5) L_aux_2=0.1576(w=0.5)
[2026-06-19 16:01:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 483): 12688.6 MiB
[2026-06-19 16:02:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3346, pooled_mean_dice=0.7954, per_class=['0.7954'], iou=0.6603, recall=0.9848, precision=0.6671, vol_sim=0.8077, mcc=0.8063, min_class_dice=0.7954, coverage=[76]/88 samples
[2026-06-19 16:02:42] INFO segtask_v1.trainer.trainer: Epoch 484/1000 | LR=1.08e-04 | loss=0.2473 | val_dice=0.7954 | best=0.8292 (ep441) | 02:33:13 | L_main=0.1244 L_aux_1=0.0973(w=0.5) L_aux_2=0.1484(w=0.5)
[2026-06-19 16:02:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 484): 12688.6 MiB
[2026-06-19 16:03:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3262, pooled_mean_dice=0.7872, per_class=['0.7872'], iou=0.6491, recall=0.9834, precision=0.6563, vol_sim=0.8005, mcc=0.7987, min_class_dice=0.7872, coverage=[79]/88 samples
[2026-06-19 16:03:44] INFO segtask_v1.trainer.trainer: Epoch 485/1000 | LR=1.11e-04 | loss=0.2482 | val_dice=0.7872 | best=0.8292 (ep441) | 02:34:16 | L_main=0.1273 L_aux_1=0.1000(w=0.5) L_aux_2=0.1418(w=0.5)
[2026-06-19 16:03:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 485): 12688.6 MiB
[2026-06-19 16:04:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2720, pooled_mean_dice=0.8155, per_class=['0.8155'], iou=0.6885, recall=0.9787, precision=0.6990, vol_sim=0.8333, mcc=0.8224, min_class_dice=0.8155, coverage=[72]/88 samples
[2026-06-19 16:04:46] INFO segtask_v1.trainer.trainer: Epoch 486/1000 | LR=1.13e-04 | loss=0.2558 | val_dice=0.8155 | best=0.8292 (ep441) | 02:35:17 | L_main=0.1267 L_aux_1=0.1048(w=0.5) L_aux_2=0.1534(w=0.5)
[2026-06-19 16:04:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 486): 12688.6 MiB
[2026-06-19 16:05:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3142, pooled_mean_dice=0.8036, per_class=['0.8036'], iou=0.6716, recall=0.9785, precision=0.6817, vol_sim=0.8212, mcc=0.8114, min_class_dice=0.8036, coverage=[72]/88 samples
[2026-06-19 16:05:48] INFO segtask_v1.trainer.trainer: Epoch 487/1000 | LR=1.16e-04 | loss=0.2406 | val_dice=0.8036 | best=0.8292 (ep441) | 02:36:19 | L_main=0.1205 L_aux_1=0.0997(w=0.5) L_aux_2=0.1405(w=0.5)
[2026-06-19 16:05:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 487): 12688.6 MiB
[2026-06-19 16:06:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2973, pooled_mean_dice=0.8093, per_class=['0.8093'], iou=0.6796, recall=0.9854, precision=0.6866, vol_sim=0.8213, mcc=0.8179, min_class_dice=0.8093, coverage=[72]/88 samples
[2026-06-19 16:06:51] INFO segtask_v1.trainer.trainer: Epoch 488/1000 | LR=1.18e-04 | loss=0.2689 | val_dice=0.8093 | best=0.8292 (ep441) | 02:37:22 | L_main=0.1342 L_aux_1=0.1059(w=0.5) L_aux_2=0.1634(w=0.5)
[2026-06-19 16:06:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 488): 12688.6 MiB
[2026-06-19 16:07:54] INFO segtask_v1.trainer.validation:   Val: loss=0.2783, pooled_mean_dice=0.8226, per_class=['0.8226'], iou=0.6986, recall=0.9820, precision=0.7077, vol_sim=0.8376, mcc=0.8282, min_class_dice=0.8226, coverage=[78]/88 samples
[2026-06-19 16:07:54] INFO segtask_v1.trainer.trainer: Epoch 489/1000 | LR=1.21e-04 | loss=0.2426 | val_dice=0.8226 | best=0.8292 (ep441) | 02:38:25 | L_main=0.1228 L_aux_1=0.0949(w=0.5) L_aux_2=0.1448(w=0.5)
[2026-06-19 16:07:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 489): 12688.6 MiB
[2026-06-19 16:08:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2954, pooled_mean_dice=0.8028, per_class=['0.8028'], iou=0.6706, recall=0.9832, precision=0.6783, vol_sim=0.8165, mcc=0.8116, min_class_dice=0.8028, coverage=[75]/88 samples
[2026-06-19 16:08:56] INFO segtask_v1.trainer.trainer: Epoch 490/1000 | LR=1.24e-04 | loss=0.2393 | val_dice=0.8028 | best=0.8292 (ep441) | 02:39:28 | L_main=0.1219 L_aux_1=0.0996(w=0.5) L_aux_2=0.1353(w=0.5)
[2026-06-19 16:08:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 490): 12688.6 MiB
[2026-06-19 16:10:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2701, pooled_mean_dice=0.8122, per_class=['0.8122'], iou=0.6838, recall=0.9832, precision=0.6919, vol_sim=0.8261, mcc=0.8203, min_class_dice=0.8122, coverage=[70]/88 samples
[2026-06-19 16:10:00] INFO segtask_v1.trainer.trainer: Epoch 491/1000 | LR=1.26e-04 | loss=0.2276 | val_dice=0.8122 | best=0.8292 (ep441) | 02:40:31 | L_main=0.1134 L_aux_1=0.0942(w=0.5) L_aux_2=0.1342(w=0.5)
[2026-06-19 16:10:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 491): 12688.6 MiB
[2026-06-19 16:11:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2765, pooled_mean_dice=0.8016, per_class=['0.8016'], iou=0.6689, recall=0.9841, precision=0.6762, vol_sim=0.8146, mcc=0.8104, min_class_dice=0.8016, coverage=[75]/88 samples
[2026-06-19 16:11:03] INFO segtask_v1.trainer.trainer: Epoch 492/1000 | LR=1.29e-04 | loss=0.2598 | val_dice=0.8016 | best=0.8292 (ep441) | 02:41:34 | L_main=0.1297 L_aux_1=0.1124(w=0.5) L_aux_2=0.1479(w=0.5)
[2026-06-19 16:11:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 492): 12688.6 MiB
[2026-06-19 16:12:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2864, pooled_mean_dice=0.8123, per_class=['0.8123'], iou=0.6839, recall=0.9812, precision=0.6930, vol_sim=0.8278, mcc=0.8194, min_class_dice=0.8123, coverage=[74]/88 samples
[2026-06-19 16:12:04] INFO segtask_v1.trainer.trainer: Epoch 493/1000 | LR=1.32e-04 | loss=0.2474 | val_dice=0.8123 | best=0.8292 (ep441) | 02:42:36 | L_main=0.1232 L_aux_1=0.1012(w=0.5) L_aux_2=0.1471(w=0.5)
[2026-06-19 16:12:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 493): 12688.6 MiB
[2026-06-19 16:13:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2790, pooled_mean_dice=0.8095, per_class=['0.8095'], iou=0.6800, recall=0.9841, precision=0.6875, vol_sim=0.8226, mcc=0.8179, min_class_dice=0.8095, coverage=[75]/88 samples
[2026-06-19 16:13:07] INFO segtask_v1.trainer.trainer: Epoch 494/1000 | LR=1.34e-04 | loss=0.2564 | val_dice=0.8095 | best=0.8292 (ep441) | 02:43:38 | L_main=0.1293 L_aux_1=0.1007(w=0.5) L_aux_2=0.1536(w=0.5)
[2026-06-19 16:13:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 494): 12688.6 MiB
[2026-06-19 16:14:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3087, pooled_mean_dice=0.7987, per_class=['0.7987'], iou=0.6648, recall=0.9813, precision=0.6733, vol_sim=0.8138, mcc=0.8077, min_class_dice=0.7987, coverage=[79]/88 samples
[2026-06-19 16:14:09] INFO segtask_v1.trainer.trainer: Epoch 495/1000 | LR=1.37e-04 | loss=0.2578 | val_dice=0.7987 | best=0.8292 (ep441) | 02:44:40 | L_main=0.1309 L_aux_1=0.0996(w=0.5) L_aux_2=0.1542(w=0.5)
[2026-06-19 16:14:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 495): 12688.6 MiB
[2026-06-19 16:15:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3386, pooled_mean_dice=0.8063, per_class=['0.8063'], iou=0.6755, recall=0.9756, precision=0.6871, vol_sim=0.8265, mcc=0.8143, min_class_dice=0.8063, coverage=[78]/88 samples
[2026-06-19 16:15:11] INFO segtask_v1.trainer.trainer: Epoch 496/1000 | LR=1.40e-04 | loss=0.2582 | val_dice=0.8063 | best=0.8292 (ep441) | 02:45:43 | L_main=0.1311 L_aux_1=0.1007(w=0.5) L_aux_2=0.1533(w=0.5)
[2026-06-19 16:15:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 496): 12688.6 MiB
[2026-06-19 16:16:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3049, pooled_mean_dice=0.7993, per_class=['0.7993'], iou=0.6657, recall=0.9811, precision=0.6743, vol_sim=0.8147, mcc=0.8087, min_class_dice=0.7993, coverage=[76]/88 samples
[2026-06-19 16:16:13] INFO segtask_v1.trainer.trainer: Epoch 497/1000 | LR=1.42e-04 | loss=0.2429 | val_dice=0.7993 | best=0.8292 (ep441) | 02:46:44 | L_main=0.1229 L_aux_1=0.1025(w=0.5) L_aux_2=0.1375(w=0.5)
[2026-06-19 16:16:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 497): 12688.6 MiB
[2026-06-19 16:17:15] INFO segtask_v1.trainer.validation:   Val: loss=0.3246, pooled_mean_dice=0.8259, per_class=['0.8259'], iou=0.7035, recall=0.9854, precision=0.7109, vol_sim=0.8381, mcc=0.8330, min_class_dice=0.8259, coverage=[74]/88 samples
[2026-06-19 16:17:15] INFO segtask_v1.trainer.trainer: Epoch 498/1000 | LR=1.45e-04 | loss=0.2438 | val_dice=0.8259 | best=0.8292 (ep441) | 02:47:46 | L_main=0.1220 L_aux_1=0.1007(w=0.5) L_aux_2=0.1429(w=0.5)
[2026-06-19 16:17:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 498): 12688.6 MiB
[2026-06-19 16:18:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2762, pooled_mean_dice=0.8094, per_class=['0.8094'], iou=0.6798, recall=0.9842, precision=0.6873, vol_sim=0.8224, mcc=0.8172, min_class_dice=0.8094, coverage=[75]/88 samples
[2026-06-19 16:18:17] INFO segtask_v1.trainer.trainer: Epoch 499/1000 | LR=1.48e-04 | loss=0.2349 | val_dice=0.8094 | best=0.8292 (ep441) | 02:48:49 | L_main=0.1183 L_aux_1=0.0982(w=0.5) L_aux_2=0.1350(w=0.5)
[2026-06-19 16:18:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 499): 12688.6 MiB
[2026-06-19 16:19:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3538, pooled_mean_dice=0.8022, per_class=['0.8022'], iou=0.6697, recall=0.9821, precision=0.6780, vol_sim=0.8168, mcc=0.8115, min_class_dice=0.8022, coverage=[78]/88 samples
[2026-06-19 16:19:20] INFO segtask_v1.trainer.trainer: Epoch 500/1000 | LR=1.51e-04 | loss=0.2510 | val_dice=0.8022 | best=0.8292 (ep441) | 02:49:52 | L_main=0.1232 L_aux_1=0.1072(w=0.5) L_aux_2=0.1484(w=0.5)
[2026-06-19 16:19:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 500): 12688.6 MiB
[2026-06-19 16:20:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3089, pooled_mean_dice=0.7962, per_class=['0.7962'], iou=0.6614, recall=0.9800, precision=0.6704, vol_sim=0.8124, mcc=0.8069, min_class_dice=0.7962, coverage=[76]/88 samples
[2026-06-19 16:20:23] INFO segtask_v1.trainer.trainer: Epoch 501/1000 | LR=1.54e-04 | loss=0.2515 | val_dice=0.7962 | best=0.8292 (ep441) | 02:50:54 | L_main=0.1246 L_aux_1=0.1035(w=0.5) L_aux_2=0.1503(w=0.5)
[2026-06-19 16:20:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 501): 12688.6 MiB
[2026-06-19 16:21:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3247, pooled_mean_dice=0.8143, per_class=['0.8143'], iou=0.6867, recall=0.9835, precision=0.6947, vol_sim=0.8279, mcc=0.8224, min_class_dice=0.8143, coverage=[75]/88 samples
[2026-06-19 16:21:25] INFO segtask_v1.trainer.trainer: Epoch 502/1000 | LR=1.57e-04 | loss=0.2690 | val_dice=0.8143 | best=0.8292 (ep441) | 02:51:56 | L_main=0.1347 L_aux_1=0.1079(w=0.5) L_aux_2=0.1609(w=0.5)
[2026-06-19 16:21:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 502): 12688.6 MiB
[2026-06-19 16:22:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3127, pooled_mean_dice=0.8023, per_class=['0.8023'], iou=0.6698, recall=0.9783, precision=0.6799, vol_sim=0.8201, mcc=0.8111, min_class_dice=0.8023, coverage=[76]/88 samples
[2026-06-19 16:22:28] INFO segtask_v1.trainer.trainer: Epoch 503/1000 | LR=1.59e-04 | loss=0.2439 | val_dice=0.8023 | best=0.8292 (ep441) | 02:52:59 | L_main=0.1210 L_aux_1=0.0997(w=0.5) L_aux_2=0.1459(w=0.5)
[2026-06-19 16:22:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 503): 12688.6 MiB
[2026-06-19 16:23:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3063, pooled_mean_dice=0.7958, per_class=['0.7958'], iou=0.6608, recall=0.9826, precision=0.6686, vol_sim=0.8099, mcc=0.8053, min_class_dice=0.7958, coverage=[76]/88 samples
[2026-06-19 16:23:32] INFO segtask_v1.trainer.trainer: Epoch 504/1000 | LR=1.62e-04 | loss=0.2356 | val_dice=0.7958 | best=0.8292 (ep441) | 02:54:03 | L_main=0.1157 L_aux_1=0.0980(w=0.5) L_aux_2=0.1419(w=0.5)
[2026-06-19 16:23:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 504): 12688.6 MiB
[2026-06-19 16:24:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3092, pooled_mean_dice=0.8028, per_class=['0.8028'], iou=0.6706, recall=0.9798, precision=0.6799, vol_sim=0.8193, mcc=0.8114, min_class_dice=0.8028, coverage=[78]/88 samples
[2026-06-19 16:24:34] INFO segtask_v1.trainer.trainer: Epoch 505/1000 | LR=1.65e-04 | loss=0.2322 | val_dice=0.8028 | best=0.8292 (ep441) | 02:55:05 | L_main=0.1161 L_aux_1=0.0960(w=0.5) L_aux_2=0.1361(w=0.5)
[2026-06-19 16:24:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 505): 12688.6 MiB
[2026-06-19 16:25:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2704, pooled_mean_dice=0.8012, per_class=['0.8012'], iou=0.6684, recall=0.9834, precision=0.6760, vol_sim=0.8147, mcc=0.8099, min_class_dice=0.8012, coverage=[73]/88 samples
[2026-06-19 16:25:37] INFO segtask_v1.trainer.trainer: Epoch 506/1000 | LR=1.68e-04 | loss=0.2660 | val_dice=0.8012 | best=0.8292 (ep441) | 02:56:08 | L_main=0.1350 L_aux_1=0.1155(w=0.5) L_aux_2=0.1464(w=0.5)
[2026-06-19 16:25:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 506): 12688.6 MiB
[2026-06-19 16:26:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3192, pooled_mean_dice=0.8164, per_class=['0.8164'], iou=0.6898, recall=0.9855, precision=0.6968, vol_sim=0.8284, mcc=0.8234, min_class_dice=0.8164, coverage=[75]/88 samples
[2026-06-19 16:26:40] INFO segtask_v1.trainer.trainer: Epoch 507/1000 | LR=1.71e-04 | loss=0.2643 | val_dice=0.8164 | best=0.8292 (ep441) | 02:57:11 | L_main=0.1293 L_aux_1=0.1147(w=0.5) L_aux_2=0.1554(w=0.5)
[2026-06-19 16:26:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 507): 12688.6 MiB
[2026-06-19 16:27:43] INFO segtask_v1.trainer.validation:   Val: loss=0.3112, pooled_mean_dice=0.8020, per_class=['0.8020'], iou=0.6694, recall=0.9840, precision=0.6768, vol_sim=0.8150, mcc=0.8114, min_class_dice=0.8020, coverage=[78]/88 samples
[2026-06-19 16:27:43] INFO segtask_v1.trainer.trainer: Epoch 508/1000 | LR=1.74e-04 | loss=0.2416 | val_dice=0.8020 | best=0.8292 (ep441) | 02:58:14 | L_main=0.1208 L_aux_1=0.1013(w=0.5) L_aux_2=0.1404(w=0.5)
[2026-06-19 16:27:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 508): 12688.6 MiB
[2026-06-19 16:28:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3131, pooled_mean_dice=0.8160, per_class=['0.8160'], iou=0.6892, recall=0.9816, precision=0.6982, vol_sim=0.8313, mcc=0.8228, min_class_dice=0.8160, coverage=[73]/88 samples
[2026-06-19 16:28:45] INFO segtask_v1.trainer.trainer: Epoch 509/1000 | LR=1.77e-04 | loss=0.2422 | val_dice=0.8160 | best=0.8292 (ep441) | 02:59:16 | L_main=0.1228 L_aux_1=0.0931(w=0.5) L_aux_2=0.1458(w=0.5)
[2026-06-19 16:28:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 509): 12688.6 MiB
[2026-06-19 16:29:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3181, pooled_mean_dice=0.8195, per_class=['0.8195'], iou=0.6942, recall=0.9866, precision=0.7008, vol_sim=0.8306, mcc=0.8267, min_class_dice=0.8195, coverage=[78]/88 samples
[2026-06-19 16:29:48] INFO segtask_v1.trainer.trainer: Epoch 510/1000 | LR=1.80e-04 | loss=0.2542 | val_dice=0.8195 | best=0.8292 (ep441) | 03:00:20 | L_main=0.1287 L_aux_1=0.1026(w=0.5) L_aux_2=0.1485(w=0.5)
[2026-06-19 16:29:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 510): 12688.6 MiB
[2026-06-19 16:30:50] INFO segtask_v1.trainer.validation:   Val: loss=0.3039, pooled_mean_dice=0.8094, per_class=['0.8094'], iou=0.6799, recall=0.9784, precision=0.6902, vol_sim=0.8273, mcc=0.8158, min_class_dice=0.8094, coverage=[82]/88 samples
[2026-06-19 16:30:50] INFO segtask_v1.trainer.trainer: Epoch 511/1000 | LR=1.83e-04 | loss=0.2397 | val_dice=0.8094 | best=0.8292 (ep441) | 03:01:21 | L_main=0.1179 L_aux_1=0.0999(w=0.5) L_aux_2=0.1436(w=0.5)
[2026-06-19 16:30:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 511): 12688.6 MiB
[2026-06-19 16:31:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3186, pooled_mean_dice=0.8054, per_class=['0.8054'], iou=0.6743, recall=0.9823, precision=0.6826, vol_sim=0.8200, mcc=0.8145, min_class_dice=0.8054, coverage=[72]/88 samples
[2026-06-19 16:31:53] INFO segtask_v1.trainer.trainer: Epoch 512/1000 | LR=1.86e-04 | loss=0.2577 | val_dice=0.8054 | best=0.8292 (ep441) | 03:02:24 | L_main=0.1278 L_aux_1=0.1057(w=0.5) L_aux_2=0.1541(w=0.5)
[2026-06-19 16:31:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 512): 12688.6 MiB
[2026-06-19 16:32:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3174, pooled_mean_dice=0.8075, per_class=['0.8075'], iou=0.6772, recall=0.9817, precision=0.6858, vol_sim=0.8226, mcc=0.8155, min_class_dice=0.8075, coverage=[80]/88 samples
[2026-06-19 16:32:54] INFO segtask_v1.trainer.trainer: Epoch 513/1000 | LR=1.90e-04 | loss=0.2508 | val_dice=0.8075 | best=0.8292 (ep441) | 03:03:25 | L_main=0.1251 L_aux_1=0.1030(w=0.5) L_aux_2=0.1484(w=0.5)
[2026-06-19 16:32:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 513): 12688.6 MiB
[2026-06-19 16:33:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2977, pooled_mean_dice=0.8119, per_class=['0.8119'], iou=0.6833, recall=0.9847, precision=0.6907, vol_sim=0.8245, mcc=0.8202, min_class_dice=0.8119, coverage=[77]/88 samples
[2026-06-19 16:33:57] INFO segtask_v1.trainer.trainer: Epoch 514/1000 | LR=1.93e-04 | loss=0.2489 | val_dice=0.8119 | best=0.8292 (ep441) | 03:04:28 | L_main=0.1244 L_aux_1=0.1032(w=0.5) L_aux_2=0.1458(w=0.5)
[2026-06-19 16:33:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 514): 12688.6 MiB
[2026-06-19 16:34:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3097, pooled_mean_dice=0.8080, per_class=['0.8080'], iou=0.6779, recall=0.9822, precision=0.6864, vol_sim=0.8227, mcc=0.8164, min_class_dice=0.8080, coverage=[71]/88 samples
[2026-06-19 16:34:59] INFO segtask_v1.trainer.trainer: Epoch 515/1000 | LR=1.96e-04 | loss=0.2444 | val_dice=0.8080 | best=0.8292 (ep441) | 03:05:30 | L_main=0.1230 L_aux_1=0.0948(w=0.5) L_aux_2=0.1479(w=0.5)
[2026-06-19 16:34:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 515): 12688.6 MiB
[2026-06-19 16:36:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2905, pooled_mean_dice=0.8190, per_class=['0.8190'], iou=0.6935, recall=0.9827, precision=0.7021, vol_sim=0.8334, mcc=0.8248, min_class_dice=0.8190, coverage=[77]/88 samples
[2026-06-19 16:36:01] INFO segtask_v1.trainer.trainer: Epoch 516/1000 | LR=1.99e-04 | loss=0.2514 | val_dice=0.8190 | best=0.8292 (ep441) | 03:06:33 | L_main=0.1242 L_aux_1=0.0969(w=0.5) L_aux_2=0.1577(w=0.5)
[2026-06-19 16:36:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 516): 12688.6 MiB
[2026-06-19 16:37:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2885, pooled_mean_dice=0.8181, per_class=['0.8181'], iou=0.6921, recall=0.9790, precision=0.7026, vol_sim=0.8356, mcc=0.8244, min_class_dice=0.8181, coverage=[77]/88 samples
[2026-06-19 16:37:04] INFO segtask_v1.trainer.trainer: Epoch 517/1000 | LR=2.02e-04 | loss=0.2637 | val_dice=0.8181 | best=0.8292 (ep441) | 03:07:35 | L_main=0.1342 L_aux_1=0.1020(w=0.5) L_aux_2=0.1570(w=0.5)
[2026-06-19 16:37:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 517): 12688.6 MiB
[2026-06-19 16:38:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3191, pooled_mean_dice=0.8176, per_class=['0.8176'], iou=0.6915, recall=0.9834, precision=0.6997, vol_sim=0.8314, mcc=0.8241, min_class_dice=0.8176, coverage=[79]/88 samples
[2026-06-19 16:38:07] INFO segtask_v1.trainer.trainer: Epoch 518/1000 | LR=2.05e-04 | loss=0.2384 | val_dice=0.8176 | best=0.8292 (ep441) | 03:08:38 | L_main=0.1172 L_aux_1=0.1038(w=0.5) L_aux_2=0.1385(w=0.5)
[2026-06-19 16:38:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 518): 12688.6 MiB
[2026-06-19 16:39:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2896, pooled_mean_dice=0.8161, per_class=['0.8161'], iou=0.6893, recall=0.9797, precision=0.6993, vol_sim=0.8330, mcc=0.8226, min_class_dice=0.8161, coverage=[73]/88 samples
[2026-06-19 16:39:09] INFO segtask_v1.trainer.trainer: Epoch 519/1000 | LR=2.09e-04 | loss=0.2224 | val_dice=0.8161 | best=0.8292 (ep441) | 03:09:41 | L_main=0.1090 L_aux_1=0.0940(w=0.5) L_aux_2=0.1328(w=0.5)
[2026-06-19 16:39:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 519): 12688.6 MiB
[2026-06-19 16:40:12] INFO segtask_v1.trainer.validation:   Val: loss=0.3481, pooled_mean_dice=0.8008, per_class=['0.8008'], iou=0.6678, recall=0.9786, precision=0.6777, vol_sim=0.8184, mcc=0.8089, min_class_dice=0.8008, coverage=[80]/88 samples
[2026-06-19 16:40:12] INFO segtask_v1.trainer.trainer: Epoch 520/1000 | LR=2.12e-04 | loss=0.2578 | val_dice=0.8008 | best=0.8292 (ep441) | 03:10:43 | L_main=0.1284 L_aux_1=0.1052(w=0.5) L_aux_2=0.1536(w=0.5)
[2026-06-19 16:40:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 520): 12688.6 MiB
[2026-06-19 16:41:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2947, pooled_mean_dice=0.8013, per_class=['0.8013'], iou=0.6684, recall=0.9835, precision=0.6760, vol_sim=0.8147, mcc=0.8106, min_class_dice=0.8013, coverage=[76]/88 samples
[2026-06-19 16:41:16] INFO segtask_v1.trainer.trainer: Epoch 521/1000 | LR=2.15e-04 | loss=0.2393 | val_dice=0.8013 | best=0.8292 (ep441) | 03:11:47 | L_main=0.1191 L_aux_1=0.0954(w=0.5) L_aux_2=0.1449(w=0.5)
[2026-06-19 16:41:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 521): 12688.6 MiB
[2026-06-19 16:42:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2745, pooled_mean_dice=0.8039, per_class=['0.8039'], iou=0.6721, recall=0.9791, precision=0.6819, vol_sim=0.8211, mcc=0.8119, min_class_dice=0.8039, coverage=[73]/88 samples
[2026-06-19 16:42:19] INFO segtask_v1.trainer.trainer: Epoch 522/1000 | LR=2.18e-04 | loss=0.2494 | val_dice=0.8039 | best=0.8292 (ep441) | 03:12:50 | L_main=0.1238 L_aux_1=0.1030(w=0.5) L_aux_2=0.1482(w=0.5)
[2026-06-19 16:42:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 522): 12688.6 MiB
[2026-06-19 16:43:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3019, pooled_mean_dice=0.8047, per_class=['0.8047'], iou=0.6732, recall=0.9821, precision=0.6815, vol_sim=0.8193, mcc=0.8134, min_class_dice=0.8047, coverage=[76]/88 samples
[2026-06-19 16:43:20] INFO segtask_v1.trainer.trainer: Epoch 523/1000 | LR=2.22e-04 | loss=0.2526 | val_dice=0.8047 | best=0.8292 (ep441) | 03:13:52 | L_main=0.1268 L_aux_1=0.1002(w=0.5) L_aux_2=0.1513(w=0.5)
[2026-06-19 16:43:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 523): 12688.6 MiB
[2026-06-19 16:44:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2830, pooled_mean_dice=0.8010, per_class=['0.8010'], iou=0.6681, recall=0.9834, precision=0.6757, vol_sim=0.8145, mcc=0.8104, min_class_dice=0.8010, coverage=[74]/88 samples
[2026-06-19 16:44:23] INFO segtask_v1.trainer.trainer: Epoch 524/1000 | LR=2.25e-04 | loss=0.2568 | val_dice=0.8010 | best=0.8292 (ep441) | 03:14:54 | L_main=0.1285 L_aux_1=0.1092(w=0.5) L_aux_2=0.1475(w=0.5)
[2026-06-19 16:44:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 524): 12688.6 MiB
[2026-06-19 16:45:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3090, pooled_mean_dice=0.8145, per_class=['0.8145'], iou=0.6871, recall=0.9801, precision=0.6968, vol_sim=0.8311, mcc=0.8217, min_class_dice=0.8145, coverage=[74]/88 samples
[2026-06-19 16:45:26] INFO segtask_v1.trainer.trainer: Epoch 525/1000 | LR=2.28e-04 | loss=0.2572 | val_dice=0.8145 | best=0.8292 (ep441) | 03:15:57 | L_main=0.1279 L_aux_1=0.1011(w=0.5) L_aux_2=0.1574(w=0.5)
[2026-06-19 16:45:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 525): 12688.6 MiB
[2026-06-19 16:46:29] INFO segtask_v1.trainer.validation:   Val: loss=0.2900, pooled_mean_dice=0.8106, per_class=['0.8106'], iou=0.6816, recall=0.9827, precision=0.6898, vol_sim=0.8249, mcc=0.8183, min_class_dice=0.8106, coverage=[75]/88 samples
[2026-06-19 16:46:29] INFO segtask_v1.trainer.trainer: Epoch 526/1000 | LR=2.32e-04 | loss=0.2501 | val_dice=0.8106 | best=0.8292 (ep441) | 03:17:00 | L_main=0.1253 L_aux_1=0.1052(w=0.5) L_aux_2=0.1444(w=0.5)
[2026-06-19 16:46:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 526): 12688.6 MiB
[2026-06-19 16:47:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2603, pooled_mean_dice=0.8138, per_class=['0.8138'], iou=0.6860, recall=0.9838, precision=0.6938, vol_sim=0.8271, mcc=0.8214, min_class_dice=0.8138, coverage=[73]/88 samples
[2026-06-19 16:47:33] INFO segtask_v1.trainer.trainer: Epoch 527/1000 | LR=2.35e-04 | loss=0.2700 | val_dice=0.8138 | best=0.8292 (ep441) | 03:18:04 | L_main=0.1349 L_aux_1=0.1052(w=0.5) L_aux_2=0.1651(w=0.5)
[2026-06-19 16:47:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 527): 12688.6 MiB
[2026-06-19 16:48:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3156, pooled_mean_dice=0.7936, per_class=['0.7936'], iou=0.6578, recall=0.9859, precision=0.6641, vol_sim=0.8050, mcc=0.8040, min_class_dice=0.7936, coverage=[78]/88 samples
[2026-06-19 16:48:34] INFO segtask_v1.trainer.trainer: Epoch 528/1000 | LR=2.38e-04 | loss=0.2493 | val_dice=0.7936 | best=0.8292 (ep441) | 03:19:05 | L_main=0.1272 L_aux_1=0.1048(w=0.5) L_aux_2=0.1393(w=0.5)
[2026-06-19 16:48:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 528): 12688.6 MiB
[2026-06-19 16:49:38] INFO segtask_v1.trainer.validation:   Val: loss=0.3501, pooled_mean_dice=0.8021, per_class=['0.8021'], iou=0.6696, recall=0.9783, precision=0.6797, vol_sim=0.8199, mcc=0.8117, min_class_dice=0.8021, coverage=[77]/88 samples
[2026-06-19 16:49:38] INFO segtask_v1.trainer.trainer: Epoch 529/1000 | LR=2.42e-04 | loss=0.2549 | val_dice=0.8021 | best=0.8292 (ep441) | 03:20:09 | L_main=0.1267 L_aux_1=0.1041(w=0.5) L_aux_2=0.1522(w=0.5)
[2026-06-19 16:49:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 529): 12688.6 MiB
[2026-06-19 16:50:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3017, pooled_mean_dice=0.8147, per_class=['0.8147'], iou=0.6874, recall=0.9846, precision=0.6948, vol_sim=0.8275, mcc=0.8220, min_class_dice=0.8147, coverage=[74]/88 samples
[2026-06-19 16:50:39] INFO segtask_v1.trainer.trainer: Epoch 530/1000 | LR=2.45e-04 | loss=0.2777 | val_dice=0.8147 | best=0.8292 (ep441) | 03:21:11 | L_main=0.1425 L_aux_1=0.1067(w=0.5) L_aux_2=0.1637(w=0.5)
[2026-06-19 16:50:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 530): 12688.6 MiB
[2026-06-19 16:51:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3852, pooled_mean_dice=0.7897, per_class=['0.7897'], iou=0.6525, recall=0.9860, precision=0.6586, vol_sim=0.8009, mcc=0.8015, min_class_dice=0.7897, coverage=[80]/88 samples
[2026-06-19 16:51:42] INFO segtask_v1.trainer.trainer: Epoch 531/1000 | LR=2.48e-04 | loss=0.2510 | val_dice=0.7897 | best=0.8292 (ep441) | 03:22:13 | L_main=0.1262 L_aux_1=0.1019(w=0.5) L_aux_2=0.1478(w=0.5)
[2026-06-19 16:51:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 531): 12688.6 MiB
[2026-06-19 16:52:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3066, pooled_mean_dice=0.8197, per_class=['0.8197'], iou=0.6946, recall=0.9830, precision=0.7030, vol_sim=0.8339, mcc=0.8260, min_class_dice=0.8197, coverage=[81]/88 samples
[2026-06-19 16:52:45] INFO segtask_v1.trainer.trainer: Epoch 532/1000 | LR=2.52e-04 | loss=0.2653 | val_dice=0.8197 | best=0.8292 (ep441) | 03:23:16 | L_main=0.1349 L_aux_1=0.1031(w=0.5) L_aux_2=0.1579(w=0.5)
[2026-06-19 16:52:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 532): 12688.6 MiB
[2026-06-19 16:53:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2893, pooled_mean_dice=0.8239, per_class=['0.8239'], iou=0.7006, recall=0.9802, precision=0.7107, vol_sim=0.8406, mcc=0.8300, min_class_dice=0.8239, coverage=[74]/88 samples
[2026-06-19 16:53:47] INFO segtask_v1.trainer.trainer: Epoch 533/1000 | LR=2.55e-04 | loss=0.2654 | val_dice=0.8239 | best=0.8292 (ep441) | 03:24:18 | L_main=0.1330 L_aux_1=0.1066(w=0.5) L_aux_2=0.1582(w=0.5)
[2026-06-19 16:53:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 533): 12688.6 MiB
[2026-06-19 16:54:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2882, pooled_mean_dice=0.8084, per_class=['0.8084'], iou=0.6784, recall=0.9801, precision=0.6879, vol_sim=0.8248, mcc=0.8159, min_class_dice=0.8084, coverage=[74]/88 samples
[2026-06-19 16:54:48] INFO segtask_v1.trainer.trainer: Epoch 534/1000 | LR=2.59e-04 | loss=0.2576 | val_dice=0.8084 | best=0.8292 (ep441) | 03:25:19 | L_main=0.1296 L_aux_1=0.1136(w=0.5) L_aux_2=0.1424(w=0.5)
[2026-06-19 16:54:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 534): 12688.6 MiB
[2026-06-19 16:55:50] INFO segtask_v1.trainer.validation:   Val: loss=0.3106, pooled_mean_dice=0.8133, per_class=['0.8133'], iou=0.6854, recall=0.9840, precision=0.6931, vol_sim=0.8265, mcc=0.8206, min_class_dice=0.8133, coverage=[79]/88 samples
[2026-06-19 16:55:50] INFO segtask_v1.trainer.trainer: Epoch 535/1000 | LR=2.62e-04 | loss=0.2624 | val_dice=0.8133 | best=0.8292 (ep441) | 03:26:21 | L_main=0.1318 L_aux_1=0.1091(w=0.5) L_aux_2=0.1521(w=0.5)
[2026-06-19 16:55:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 535): 12688.6 MiB
[2026-06-19 16:56:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2978, pooled_mean_dice=0.8094, per_class=['0.8094'], iou=0.6798, recall=0.9824, precision=0.6882, vol_sim=0.8239, mcc=0.8169, min_class_dice=0.8094, coverage=[78]/88 samples
[2026-06-19 16:56:52] INFO segtask_v1.trainer.trainer: Epoch 536/1000 | LR=2.66e-04 | loss=0.2443 | val_dice=0.8094 | best=0.8292 (ep441) | 03:27:23 | L_main=0.1194 L_aux_1=0.1022(w=0.5) L_aux_2=0.1476(w=0.5)
[2026-06-19 16:56:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 536): 12688.6 MiB
[2026-06-19 16:57:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3091, pooled_mean_dice=0.8036, per_class=['0.8036'], iou=0.6717, recall=0.9864, precision=0.6780, vol_sim=0.8147, mcc=0.8123, min_class_dice=0.8036, coverage=[77]/88 samples
[2026-06-19 16:57:55] INFO segtask_v1.trainer.trainer: Epoch 537/1000 | LR=2.69e-04 | loss=0.2479 | val_dice=0.8036 | best=0.8292 (ep441) | 03:28:27 | L_main=0.1234 L_aux_1=0.1040(w=0.5) L_aux_2=0.1451(w=0.5)
[2026-06-19 16:57:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 537): 12688.6 MiB
[2026-06-19 16:58:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3140, pooled_mean_dice=0.7994, per_class=['0.7994'], iou=0.6658, recall=0.9835, precision=0.6733, vol_sim=0.8128, mcc=0.8085, min_class_dice=0.7994, coverage=[78]/88 samples
[2026-06-19 16:58:58] INFO segtask_v1.trainer.trainer: Epoch 538/1000 | LR=2.73e-04 | loss=0.2516 | val_dice=0.7994 | best=0.8292 (ep441) | 03:29:30 | L_main=0.1237 L_aux_1=0.1004(w=0.5) L_aux_2=0.1554(w=0.5)
[2026-06-19 16:58:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 538): 12688.6 MiB
[2026-06-19 17:00:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2992, pooled_mean_dice=0.8258, per_class=['0.8258'], iou=0.7033, recall=0.9799, precision=0.7136, vol_sim=0.8428, mcc=0.8316, min_class_dice=0.8258, coverage=[74]/88 samples
[2026-06-19 17:00:01] INFO segtask_v1.trainer.trainer: Epoch 539/1000 | LR=2.76e-04 | loss=0.2419 | val_dice=0.8258 | best=0.8292 (ep441) | 03:30:32 | L_main=0.1196 L_aux_1=0.0988(w=0.5) L_aux_2=0.1458(w=0.5)
[2026-06-19 17:00:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 539): 12688.6 MiB
[2026-06-19 17:01:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3017, pooled_mean_dice=0.8172, per_class=['0.8172'], iou=0.6908, recall=0.9866, precision=0.6974, vol_sim=0.8283, mcc=0.8242, min_class_dice=0.8172, coverage=[77]/88 samples
[2026-06-19 17:01:04] INFO segtask_v1.trainer.trainer: Epoch 540/1000 | LR=2.80e-04 | loss=0.2450 | val_dice=0.8172 | best=0.8292 (ep441) | 03:31:35 | L_main=0.1195 L_aux_1=0.1000(w=0.5) L_aux_2=0.1511(w=0.5)
[2026-06-19 17:01:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 540): 12688.6 MiB
[2026-06-19 17:02:06] INFO segtask_v1.trainer.validation:   Val: loss=0.3298, pooled_mean_dice=0.8037, per_class=['0.8037'], iou=0.6718, recall=0.9844, precision=0.6790, vol_sim=0.8164, mcc=0.8128, min_class_dice=0.8037, coverage=[78]/88 samples
[2026-06-19 17:02:06] INFO segtask_v1.trainer.trainer: Epoch 541/1000 | LR=2.84e-04 | loss=0.2409 | val_dice=0.8037 | best=0.8292 (ep441) | 03:32:38 | L_main=0.1176 L_aux_1=0.0978(w=0.5) L_aux_2=0.1489(w=0.5)
[2026-06-19 17:02:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 541): 12688.6 MiB
[2026-06-19 17:03:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3211, pooled_mean_dice=0.8072, per_class=['0.8072'], iou=0.6767, recall=0.9838, precision=0.6843, vol_sim=0.8205, mcc=0.8154, min_class_dice=0.8072, coverage=[79]/88 samples
[2026-06-19 17:03:09] INFO segtask_v1.trainer.trainer: Epoch 542/1000 | LR=2.87e-04 | loss=0.2476 | val_dice=0.8072 | best=0.8292 (ep441) | 03:33:40 | L_main=0.1217 L_aux_1=0.0995(w=0.5) L_aux_2=0.1522(w=0.5)
[2026-06-19 17:03:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 542): 12688.6 MiB
[2026-06-19 17:04:12] INFO segtask_v1.trainer.validation:   Val: loss=0.3012, pooled_mean_dice=0.8072, per_class=['0.8072'], iou=0.6768, recall=0.9833, precision=0.6846, vol_sim=0.8209, mcc=0.8152, min_class_dice=0.8072, coverage=[78]/88 samples
[2026-06-19 17:04:12] INFO segtask_v1.trainer.trainer: Epoch 543/1000 | LR=2.91e-04 | loss=0.2382 | val_dice=0.8072 | best=0.8292 (ep441) | 03:34:44 | L_main=0.1146 L_aux_1=0.1050(w=0.5) L_aux_2=0.1421(w=0.5)
[2026-06-19 17:04:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 543): 12688.6 MiB
[2026-06-19 17:05:15] INFO segtask_v1.trainer.validation:   Val: loss=0.3346, pooled_mean_dice=0.8077, per_class=['0.8077'], iou=0.6774, recall=0.9816, precision=0.6861, vol_sim=0.8228, mcc=0.8164, min_class_dice=0.8077, coverage=[77]/88 samples
[2026-06-19 17:05:15] INFO segtask_v1.trainer.trainer: Epoch 544/1000 | LR=2.94e-04 | loss=0.2445 | val_dice=0.8077 | best=0.8292 (ep441) | 03:35:47 | L_main=0.1179 L_aux_1=0.0929(w=0.5) L_aux_2=0.1603(w=0.5)
[2026-06-19 17:05:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 544): 12688.6 MiB
[2026-06-19 17:06:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2809, pooled_mean_dice=0.8132, per_class=['0.8132'], iou=0.6852, recall=0.9808, precision=0.6945, vol_sim=0.8291, mcc=0.8208, min_class_dice=0.8132, coverage=[75]/88 samples
[2026-06-19 17:06:17] INFO segtask_v1.trainer.trainer: Epoch 545/1000 | LR=2.98e-04 | loss=0.2509 | val_dice=0.8132 | best=0.8292 (ep441) | 03:36:48 | L_main=0.1237 L_aux_1=0.1068(w=0.5) L_aux_2=0.1475(w=0.5)
[2026-06-19 17:06:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 545): 12688.6 MiB
[2026-06-19 17:07:19] INFO segtask_v1.trainer.validation:   Val: loss=0.3182, pooled_mean_dice=0.7988, per_class=['0.7988'], iou=0.6650, recall=0.9858, precision=0.6714, vol_sim=0.8103, mcc=0.8093, min_class_dice=0.7988, coverage=[73]/88 samples
[2026-06-19 17:07:19] INFO segtask_v1.trainer.trainer: Epoch 546/1000 | LR=3.02e-04 | loss=0.2404 | val_dice=0.7988 | best=0.8292 (ep441) | 03:37:50 | L_main=0.1157 L_aux_1=0.0988(w=0.5) L_aux_2=0.1507(w=0.5)
[2026-06-19 17:07:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 546): 12688.6 MiB
[2026-06-19 17:08:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2937, pooled_mean_dice=0.8008, per_class=['0.8008'], iou=0.6678, recall=0.9818, precision=0.6762, vol_sim=0.8157, mcc=0.8103, min_class_dice=0.8008, coverage=[73]/88 samples
[2026-06-19 17:08:22] INFO segtask_v1.trainer.trainer: Epoch 547/1000 | LR=3.05e-04 | loss=0.2553 | val_dice=0.8008 | best=0.8292 (ep441) | 03:38:53 | L_main=0.1212 L_aux_1=0.1072(w=0.5) L_aux_2=0.1610(w=0.5)
[2026-06-19 17:08:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 547): 12688.6 MiB
[2026-06-19 17:09:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3022, pooled_mean_dice=0.8147, per_class=['0.8147'], iou=0.6874, recall=0.9822, precision=0.6960, vol_sim=0.8295, mcc=0.8217, min_class_dice=0.8147, coverage=[78]/88 samples
[2026-06-19 17:09:25] INFO segtask_v1.trainer.trainer: Epoch 548/1000 | LR=3.09e-04 | loss=0.2439 | val_dice=0.8147 | best=0.8292 (ep441) | 03:39:56 | L_main=0.1200 L_aux_1=0.0982(w=0.5) L_aux_2=0.1497(w=0.5)
[2026-06-19 17:09:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 548): 12688.6 MiB
[2026-06-19 17:10:27] INFO segtask_v1.trainer.validation:   Val: loss=0.3437, pooled_mean_dice=0.8165, per_class=['0.8165'], iou=0.6899, recall=0.9860, precision=0.6967, vol_sim=0.8281, mcc=0.8247, min_class_dice=0.8165, coverage=[77]/88 samples
[2026-06-19 17:10:27] INFO segtask_v1.trainer.trainer: Epoch 549/1000 | LR=3.13e-04 | loss=0.2782 | val_dice=0.8165 | best=0.8292 (ep441) | 03:40:58 | L_main=0.1371 L_aux_1=0.1217(w=0.5) L_aux_2=0.1606(w=0.5)
[2026-06-19 17:10:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 549): 12688.6 MiB
[2026-06-19 17:11:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2895, pooled_mean_dice=0.8048, per_class=['0.8048'], iou=0.6734, recall=0.9857, precision=0.6800, vol_sim=0.8165, mcc=0.8141, min_class_dice=0.8048, coverage=[72]/88 samples
[2026-06-19 17:11:30] INFO segtask_v1.trainer.trainer: Epoch 550/1000 | LR=3.16e-04 | loss=0.2540 | val_dice=0.8048 | best=0.8292 (ep441) | 03:42:01 | L_main=0.1236 L_aux_1=0.1114(w=0.5) L_aux_2=0.1494(w=0.5)
[2026-06-19 17:11:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 550): 12688.6 MiB
[2026-06-19 17:12:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3152, pooled_mean_dice=0.7983, per_class=['0.7983'], iou=0.6642, recall=0.9832, precision=0.6719, vol_sim=0.8119, mcc=0.8082, min_class_dice=0.7983, coverage=[72]/88 samples
[2026-06-19 17:12:34] INFO segtask_v1.trainer.trainer: Epoch 551/1000 | LR=3.20e-04 | loss=0.2542 | val_dice=0.7983 | best=0.8292 (ep441) | 03:43:06 | L_main=0.1234 L_aux_1=0.1019(w=0.5) L_aux_2=0.1597(w=0.5)
[2026-06-19 17:12:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 551): 12688.6 MiB
[2026-06-19 17:13:38] INFO segtask_v1.trainer.validation:   Val: loss=0.2764, pooled_mean_dice=0.8099, per_class=['0.8099'], iou=0.6805, recall=0.9811, precision=0.6895, vol_sim=0.8254, mcc=0.8182, min_class_dice=0.8099, coverage=[72]/88 samples
[2026-06-19 17:13:38] INFO segtask_v1.trainer.trainer: Epoch 552/1000 | LR=3.24e-04 | loss=0.2328 | val_dice=0.8099 | best=0.8292 (ep441) | 03:44:09 | L_main=0.1121 L_aux_1=0.0962(w=0.5) L_aux_2=0.1452(w=0.5)
[2026-06-19 17:13:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 552): 12688.6 MiB
[2026-06-19 17:14:39] INFO segtask_v1.trainer.validation:   Val: loss=0.2800, pooled_mean_dice=0.8216, per_class=['0.8216'], iou=0.6972, recall=0.9863, precision=0.7040, vol_sim=0.8330, mcc=0.8277, min_class_dice=0.8216, coverage=[79]/88 samples
[2026-06-19 17:14:39] INFO segtask_v1.trainer.trainer: Epoch 553/1000 | LR=3.27e-04 | loss=0.2326 | val_dice=0.8216 | best=0.8292 (ep441) | 03:45:10 | L_main=0.1096 L_aux_1=0.1042(w=0.5) L_aux_2=0.1418(w=0.5)
[2026-06-19 17:14:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 553): 12688.6 MiB
[2026-06-19 17:15:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2956, pooled_mean_dice=0.7899, per_class=['0.7899'], iou=0.6527, recall=0.9767, precision=0.6630, vol_sim=0.8087, mcc=0.8004, min_class_dice=0.7899, coverage=[73]/88 samples
[2026-06-19 17:15:43] INFO segtask_v1.trainer.trainer: Epoch 554/1000 | LR=3.31e-04 | loss=0.2539 | val_dice=0.7899 | best=0.8292 (ep441) | 03:46:15 | L_main=0.1234 L_aux_1=0.1079(w=0.5) L_aux_2=0.1532(w=0.5)
[2026-06-19 17:15:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 554): 12688.6 MiB
[2026-06-19 17:16:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3369, pooled_mean_dice=0.8112, per_class=['0.8112'], iou=0.6823, recall=0.9831, precision=0.6904, vol_sim=0.8251, mcc=0.8188, min_class_dice=0.8112, coverage=[84]/88 samples
[2026-06-19 17:16:46] INFO segtask_v1.trainer.trainer: Epoch 555/1000 | LR=3.35e-04 | loss=0.2709 | val_dice=0.8112 | best=0.8292 (ep441) | 03:47:17 | L_main=0.1323 L_aux_1=0.1213(w=0.5) L_aux_2=0.1559(w=0.5)
[2026-06-19 17:16:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 555): 12688.6 MiB
[2026-06-19 17:17:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3215, pooled_mean_dice=0.8120, per_class=['0.8120'], iou=0.6835, recall=0.9793, precision=0.6935, vol_sim=0.8291, mcc=0.8201, min_class_dice=0.8120, coverage=[72]/88 samples
[2026-06-19 17:17:49] INFO segtask_v1.trainer.trainer: Epoch 556/1000 | LR=3.39e-04 | loss=0.2568 | val_dice=0.8120 | best=0.8292 (ep441) | 03:48:20 | L_main=0.1203 L_aux_1=0.1145(w=0.5) L_aux_2=0.1585(w=0.5)
[2026-06-19 17:17:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 556): 12688.6 MiB
[2026-06-19 17:18:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3149, pooled_mean_dice=0.8011, per_class=['0.8011'], iou=0.6682, recall=0.9809, precision=0.6770, vol_sim=0.8167, mcc=0.8104, min_class_dice=0.8011, coverage=[74]/88 samples
[2026-06-19 17:18:53] INFO segtask_v1.trainer.trainer: Epoch 557/1000 | LR=3.42e-04 | loss=0.2198 | val_dice=0.8011 | best=0.8292 (ep441) | 03:49:25 | L_main=0.1031 L_aux_1=0.0978(w=0.5) L_aux_2=0.1356(w=0.5)
[2026-06-19 17:18:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 557): 12688.6 MiB
[2026-06-19 17:19:56] INFO segtask_v1.trainer.validation:   Val: loss=0.2770, pooled_mean_dice=0.8151, per_class=['0.8151'], iou=0.6879, recall=0.9789, precision=0.6983, vol_sim=0.8327, mcc=0.8220, min_class_dice=0.8151, coverage=[73]/88 samples
[2026-06-19 17:19:56] INFO segtask_v1.trainer.trainer: Epoch 558/1000 | LR=3.46e-04 | loss=0.2509 | val_dice=0.8151 | best=0.8292 (ep441) | 03:50:27 | L_main=0.1207 L_aux_1=0.1079(w=0.5) L_aux_2=0.1527(w=0.5)
[2026-06-19 17:19:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 558): 12688.6 MiB
[2026-06-19 17:20:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3275, pooled_mean_dice=0.8011, per_class=['0.8011'], iou=0.6681, recall=0.9815, precision=0.6766, vol_sim=0.8161, mcc=0.8092, min_class_dice=0.8011, coverage=[76]/88 samples
[2026-06-19 17:20:59] INFO segtask_v1.trainer.trainer: Epoch 559/1000 | LR=3.50e-04 | loss=0.2411 | val_dice=0.8011 | best=0.8292 (ep441) | 03:51:30 | L_main=0.1126 L_aux_1=0.1011(w=0.5) L_aux_2=0.1559(w=0.5)
[2026-06-19 17:20:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 559): 12688.6 MiB
[2026-06-19 17:22:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2936, pooled_mean_dice=0.8221, per_class=['0.8221'], iou=0.6980, recall=0.9822, precision=0.7069, vol_sim=0.8370, mcc=0.8287, min_class_dice=0.8221, coverage=[73]/88 samples
[2026-06-19 17:22:02] INFO segtask_v1.trainer.trainer: Epoch 560/1000 | LR=3.54e-04 | loss=0.2566 | val_dice=0.8221 | best=0.8292 (ep441) | 03:52:33 | L_main=0.1234 L_aux_1=0.1138(w=0.5) L_aux_2=0.1526(w=0.5)
[2026-06-19 17:22:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 560): 12688.6 MiB
[2026-06-19 17:23:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3308, pooled_mean_dice=0.8054, per_class=['0.8054'], iou=0.6742, recall=0.9844, precision=0.6815, vol_sim=0.8181, mcc=0.8144, min_class_dice=0.8054, coverage=[77]/88 samples
[2026-06-19 17:23:05] INFO segtask_v1.trainer.trainer: Epoch 561/1000 | LR=3.58e-04 | loss=0.2840 | val_dice=0.8054 | best=0.8292 (ep441) | 03:53:36 | L_main=0.1371 L_aux_1=0.1289(w=0.5) L_aux_2=0.1649(w=0.5)
[2026-06-19 17:23:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 561): 12688.6 MiB
[2026-06-19 17:24:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2930, pooled_mean_dice=0.7986, per_class=['0.7986'], iou=0.6647, recall=0.9832, precision=0.6723, vol_sim=0.8122, mcc=0.8077, min_class_dice=0.7986, coverage=[73]/88 samples
[2026-06-19 17:24:07] INFO segtask_v1.trainer.trainer: Epoch 562/1000 | LR=3.61e-04 | loss=0.3054 | val_dice=0.7986 | best=0.8292 (ep441) | 03:54:38 | L_main=0.1476 L_aux_1=0.1392(w=0.5) L_aux_2=0.1765(w=0.5)
[2026-06-19 17:24:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 562): 12688.6 MiB
[2026-06-19 17:25:08] INFO segtask_v1.trainer.validation:   Val: loss=0.3253, pooled_mean_dice=0.7958, per_class=['0.7958'], iou=0.6609, recall=0.9795, precision=0.6701, vol_sim=0.8124, mcc=0.8051, min_class_dice=0.7958, coverage=[78]/88 samples
[2026-06-19 17:25:08] INFO segtask_v1.trainer.trainer: Epoch 563/1000 | LR=3.65e-04 | loss=0.2974 | val_dice=0.7958 | best=0.8292 (ep441) | 03:55:39 | L_main=0.1446 L_aux_1=0.1297(w=0.5) L_aux_2=0.1758(w=0.5)
[2026-06-19 17:25:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 563): 12688.6 MiB
[2026-06-19 17:26:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3497, pooled_mean_dice=0.7944, per_class=['0.7944'], iou=0.6590, recall=0.9828, precision=0.6667, vol_sim=0.8084, mcc=0.8053, min_class_dice=0.7944, coverage=[80]/88 samples
[2026-06-19 17:26:11] INFO segtask_v1.trainer.trainer: Epoch 564/1000 | LR=3.69e-04 | loss=0.2773 | val_dice=0.7944 | best=0.8292 (ep441) | 03:56:42 | L_main=0.1329 L_aux_1=0.1232(w=0.5) L_aux_2=0.1655(w=0.5)
[2026-06-19 17:26:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 564): 12688.6 MiB
[2026-06-19 17:27:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2883, pooled_mean_dice=0.7947, per_class=['0.7947'], iou=0.6593, recall=0.9847, precision=0.6661, vol_sim=0.8070, mcc=0.8045, min_class_dice=0.7947, coverage=[73]/88 samples
[2026-06-19 17:27:14] INFO segtask_v1.trainer.trainer: Epoch 565/1000 | LR=3.73e-04 | loss=0.2474 | val_dice=0.7947 | best=0.8292 (ep441) | 03:57:45 | L_main=0.1193 L_aux_1=0.1072(w=0.5) L_aux_2=0.1490(w=0.5)
[2026-06-19 17:27:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 565): 12688.6 MiB
[2026-06-19 17:28:18] INFO segtask_v1.trainer.validation:   Val: loss=0.3009, pooled_mean_dice=0.8078, per_class=['0.8078'], iou=0.6776, recall=0.9845, precision=0.6848, vol_sim=0.8205, mcc=0.8166, min_class_dice=0.8078, coverage=[74]/88 samples
[2026-06-19 17:28:18] INFO segtask_v1.trainer.trainer: Epoch 566/1000 | LR=3.77e-04 | loss=0.2774 | val_dice=0.8078 | best=0.8292 (ep441) | 03:58:49 | L_main=0.1337 L_aux_1=0.1203(w=0.5) L_aux_2=0.1671(w=0.5)
[2026-06-19 17:28:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 566): 12688.6 MiB
[2026-06-19 17:29:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3219, pooled_mean_dice=0.7897, per_class=['0.7897'], iou=0.6524, recall=0.9810, precision=0.6608, vol_sim=0.8049, mcc=0.8005, min_class_dice=0.7897, coverage=[75]/88 samples
[2026-06-19 17:29:20] INFO segtask_v1.trainer.trainer: Epoch 567/1000 | LR=3.81e-04 | loss=0.2502 | val_dice=0.7897 | best=0.8292 (ep441) | 03:59:51 | L_main=0.1207 L_aux_1=0.1094(w=0.5) L_aux_2=0.1496(w=0.5)
[2026-06-19 17:29:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 567): 12688.6 MiB
[2026-06-19 17:30:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3163, pooled_mean_dice=0.7784, per_class=['0.7784'], iou=0.6372, recall=0.9832, precision=0.6442, vol_sim=0.7917, mcc=0.7913, min_class_dice=0.7784, coverage=[76]/88 samples
[2026-06-19 17:30:23] INFO segtask_v1.trainer.trainer: Epoch 568/1000 | LR=3.84e-04 | loss=0.2549 | val_dice=0.7784 | best=0.8292 (ep441) | 04:00:54 | L_main=0.1194 L_aux_1=0.1167(w=0.5) L_aux_2=0.1542(w=0.5)
[2026-06-19 17:30:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 568): 12688.6 MiB
[2026-06-19 17:31:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3319, pooled_mean_dice=0.7980, per_class=['0.7980'], iou=0.6638, recall=0.9840, precision=0.6711, vol_sim=0.8109, mcc=0.8077, min_class_dice=0.7980, coverage=[77]/88 samples
[2026-06-19 17:31:26] INFO segtask_v1.trainer.trainer: Epoch 569/1000 | LR=3.88e-04 | loss=0.2467 | val_dice=0.7980 | best=0.8292 (ep441) | 04:01:57 | L_main=0.1171 L_aux_1=0.1103(w=0.5) L_aux_2=0.1490(w=0.5)
[2026-06-19 17:31:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 569): 12688.6 MiB
[2026-06-19 17:32:29] INFO segtask_v1.trainer.validation:   Val: loss=0.3279, pooled_mean_dice=0.7877, per_class=['0.7877'], iou=0.6497, recall=0.9875, precision=0.6551, vol_sim=0.7977, mcc=0.7986, min_class_dice=0.7877, coverage=[81]/88 samples
[2026-06-19 17:32:29] INFO segtask_v1.trainer.trainer: Epoch 570/1000 | LR=3.92e-04 | loss=0.2607 | val_dice=0.7877 | best=0.8292 (ep441) | 04:03:00 | L_main=0.1282 L_aux_1=0.1107(w=0.5) L_aux_2=0.1542(w=0.5)
[2026-06-19 17:32:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 570): 12688.6 MiB
[2026-06-19 17:33:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3107, pooled_mean_dice=0.8023, per_class=['0.8023'], iou=0.6699, recall=0.9877, precision=0.6756, vol_sim=0.8123, mcc=0.8122, min_class_dice=0.8023, coverage=[75]/88 samples
[2026-06-19 17:33:31] INFO segtask_v1.trainer.trainer: Epoch 571/1000 | LR=3.96e-04 | loss=0.2449 | val_dice=0.8023 | best=0.8292 (ep441) | 04:04:02 | L_main=0.1176 L_aux_1=0.1064(w=0.5) L_aux_2=0.1481(w=0.5)
[2026-06-19 17:33:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 571): 12688.6 MiB
[2026-06-19 17:34:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3382, pooled_mean_dice=0.7661, per_class=['0.7661'], iou=0.6209, recall=0.9851, precision=0.6268, vol_sim=0.7777, mcc=0.7819, min_class_dice=0.7661, coverage=[71]/88 samples
[2026-06-19 17:34:34] INFO segtask_v1.trainer.trainer: Epoch 572/1000 | LR=4.00e-04 | loss=0.2376 | val_dice=0.7661 | best=0.8292 (ep441) | 04:05:06 | L_main=0.1150 L_aux_1=0.1009(w=0.5) L_aux_2=0.1442(w=0.5)
[2026-06-19 17:34:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 572): 12688.6 MiB
[2026-06-19 17:35:36] INFO segtask_v1.trainer.validation:   Val: loss=0.3050, pooled_mean_dice=0.7999, per_class=['0.7999'], iou=0.6665, recall=0.9812, precision=0.6751, vol_sim=0.8152, mcc=0.8088, min_class_dice=0.7999, coverage=[79]/88 samples
[2026-06-19 17:35:36] INFO segtask_v1.trainer.trainer: Epoch 573/1000 | LR=4.04e-04 | loss=0.2448 | val_dice=0.7999 | best=0.8292 (ep441) | 04:06:07 | L_main=0.1162 L_aux_1=0.1109(w=0.5) L_aux_2=0.1462(w=0.5)
[2026-06-19 17:35:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 573): 12688.6 MiB
[2026-06-19 17:36:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3063, pooled_mean_dice=0.7914, per_class=['0.7914'], iou=0.6547, recall=0.9825, precision=0.6625, vol_sim=0.8054, mcc=0.8015, min_class_dice=0.7914, coverage=[73]/88 samples
[2026-06-19 17:36:39] INFO segtask_v1.trainer.trainer: Epoch 574/1000 | LR=4.08e-04 | loss=0.2353 | val_dice=0.7914 | best=0.8292 (ep441) | 04:07:10 | L_main=0.1142 L_aux_1=0.0956(w=0.5) L_aux_2=0.1465(w=0.5)
[2026-06-19 17:36:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 574): 12688.6 MiB
[2026-06-19 17:37:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2631, pooled_mean_dice=0.8187, per_class=['0.8187'], iou=0.6930, recall=0.9851, precision=0.7004, vol_sim=0.8311, mcc=0.8257, min_class_dice=0.8187, coverage=[70]/88 samples
[2026-06-19 17:37:41] INFO segtask_v1.trainer.trainer: Epoch 575/1000 | LR=4.12e-04 | loss=0.2353 | val_dice=0.8187 | best=0.8292 (ep441) | 04:08:13 | L_main=0.1133 L_aux_1=0.1081(w=0.5) L_aux_2=0.1359(w=0.5)
[2026-06-19 17:37:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 575): 12688.6 MiB
[2026-06-19 17:38:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2962, pooled_mean_dice=0.7973, per_class=['0.7973'], iou=0.6630, recall=0.9853, precision=0.6696, vol_sim=0.8092, mcc=0.8076, min_class_dice=0.7973, coverage=[73]/88 samples
[2026-06-19 17:38:44] INFO segtask_v1.trainer.trainer: Epoch 576/1000 | LR=4.16e-04 | loss=0.2507 | val_dice=0.7973 | best=0.8292 (ep441) | 04:09:16 | L_main=0.1184 L_aux_1=0.0990(w=0.5) L_aux_2=0.1657(w=0.5)
[2026-06-19 17:38:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 576): 12688.6 MiB
[2026-06-19 17:39:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2909, pooled_mean_dice=0.8158, per_class=['0.8158'], iou=0.6889, recall=0.9836, precision=0.6969, vol_sim=0.8294, mcc=0.8227, min_class_dice=0.8158, coverage=[75]/88 samples
[2026-06-19 17:39:47] INFO segtask_v1.trainer.trainer: Epoch 577/1000 | LR=4.19e-04 | loss=0.2382 | val_dice=0.8158 | best=0.8292 (ep441) | 04:10:19 | L_main=0.1114 L_aux_1=0.1004(w=0.5) L_aux_2=0.1533(w=0.5)
[2026-06-19 17:39:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 577): 12688.6 MiB
[2026-06-19 17:40:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3152, pooled_mean_dice=0.7868, per_class=['0.7868'], iou=0.6486, recall=0.9826, precision=0.6561, vol_sim=0.8008, mcc=0.7984, min_class_dice=0.7868, coverage=[74]/88 samples
[2026-06-19 17:40:49] INFO segtask_v1.trainer.trainer: Epoch 578/1000 | LR=4.23e-04 | loss=0.2470 | val_dice=0.7868 | best=0.8292 (ep441) | 04:11:21 | L_main=0.1181 L_aux_1=0.1041(w=0.5) L_aux_2=0.1538(w=0.5)
[2026-06-19 17:40:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 578): 12688.6 MiB
[2026-06-19 17:41:52] INFO segtask_v1.trainer.validation:   Val: loss=0.3075, pooled_mean_dice=0.7968, per_class=['0.7968'], iou=0.6623, recall=0.9820, precision=0.6704, vol_sim=0.8114, mcc=0.8067, min_class_dice=0.7968, coverage=[73]/88 samples
[2026-06-19 17:41:52] INFO segtask_v1.trainer.trainer: Epoch 579/1000 | LR=4.27e-04 | loss=0.2924 | val_dice=0.7968 | best=0.8292 (ep441) | 04:12:23 | L_main=0.1392 L_aux_1=0.1269(w=0.5) L_aux_2=0.1797(w=0.5)
[2026-06-19 17:41:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 579): 12688.6 MiB
[2026-06-19 17:42:54] INFO segtask_v1.trainer.validation:   Val: loss=0.2939, pooled_mean_dice=0.8094, per_class=['0.8094'], iou=0.6799, recall=0.9837, precision=0.6876, vol_sim=0.8229, mcc=0.8177, min_class_dice=0.8094, coverage=[78]/88 samples
[2026-06-19 17:42:54] INFO segtask_v1.trainer.trainer: Epoch 580/1000 | LR=4.31e-04 | loss=0.2530 | val_dice=0.8094 | best=0.8292 (ep441) | 04:13:26 | L_main=0.1204 L_aux_1=0.1033(w=0.5) L_aux_2=0.1620(w=0.5)
[2026-06-19 17:42:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 580): 12688.6 MiB
[2026-06-19 17:43:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3351, pooled_mean_dice=0.7821, per_class=['0.7821'], iou=0.6422, recall=0.9813, precision=0.6501, vol_sim=0.7970, mcc=0.7945, min_class_dice=0.7821, coverage=[74]/88 samples
[2026-06-19 17:43:59] INFO segtask_v1.trainer.trainer: Epoch 581/1000 | LR=4.35e-04 | loss=0.2430 | val_dice=0.7821 | best=0.8292 (ep441) | 04:14:31 | L_main=0.1172 L_aux_1=0.1026(w=0.5) L_aux_2=0.1489(w=0.5)
[2026-06-19 17:43:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 581): 12688.6 MiB
[2026-06-19 17:45:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3207, pooled_mean_dice=0.8007, per_class=['0.8007'], iou=0.6676, recall=0.9831, precision=0.6753, vol_sim=0.8144, mcc=0.8097, min_class_dice=0.8007, coverage=[80]/88 samples
[2026-06-19 17:45:02] INFO segtask_v1.trainer.trainer: Epoch 582/1000 | LR=4.39e-04 | loss=0.2443 | val_dice=0.8007 | best=0.8292 (ep441) | 04:15:33 | L_main=0.1174 L_aux_1=0.1031(w=0.5) L_aux_2=0.1507(w=0.5)
[2026-06-19 17:45:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 582): 12688.6 MiB
[2026-06-19 17:46:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3272, pooled_mean_dice=0.8030, per_class=['0.8030'], iou=0.6708, recall=0.9850, precision=0.6777, vol_sim=0.8152, mcc=0.8125, min_class_dice=0.8030, coverage=[82]/88 samples
[2026-06-19 17:46:04] INFO segtask_v1.trainer.trainer: Epoch 583/1000 | LR=4.43e-04 | loss=0.2372 | val_dice=0.8030 | best=0.8292 (ep441) | 04:16:35 | L_main=0.1135 L_aux_1=0.0981(w=0.5) L_aux_2=0.1494(w=0.5)
[2026-06-19 17:46:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 583): 12688.6 MiB
[2026-06-19 17:47:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3403, pooled_mean_dice=0.7881, per_class=['0.7881'], iou=0.6504, recall=0.9832, precision=0.6577, vol_sim=0.8017, mcc=0.7996, min_class_dice=0.7881, coverage=[75]/88 samples
[2026-06-19 17:47:07] INFO segtask_v1.trainer.trainer: Epoch 584/1000 | LR=4.47e-04 | loss=0.2510 | val_dice=0.7881 | best=0.8292 (ep441) | 04:17:38 | L_main=0.1211 L_aux_1=0.1080(w=0.5) L_aux_2=0.1518(w=0.5)
[2026-06-19 17:47:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 584): 12688.6 MiB
[2026-06-19 17:48:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3479, pooled_mean_dice=0.8127, per_class=['0.8127'], iou=0.6845, recall=0.9829, precision=0.6927, vol_sim=0.8268, mcc=0.8205, min_class_dice=0.8127, coverage=[74]/88 samples
[2026-06-19 17:48:09] INFO segtask_v1.trainer.trainer: Epoch 585/1000 | LR=4.51e-04 | loss=0.2517 | val_dice=0.8127 | best=0.8292 (ep441) | 04:18:40 | L_main=0.1215 L_aux_1=0.1024(w=0.5) L_aux_2=0.1581(w=0.5)
[2026-06-19 17:48:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 585): 12688.6 MiB
[2026-06-19 17:49:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3367, pooled_mean_dice=0.8171, per_class=['0.8171'], iou=0.6907, recall=0.9872, precision=0.6969, vol_sim=0.8276, mcc=0.8255, min_class_dice=0.8171, coverage=[72]/88 samples
[2026-06-19 17:49:11] INFO segtask_v1.trainer.trainer: Epoch 586/1000 | LR=4.55e-04 | loss=0.2923 | val_dice=0.8171 | best=0.8292 (ep441) | 04:19:42 | L_main=0.1448 L_aux_1=0.1286(w=0.5) L_aux_2=0.1665(w=0.5)
[2026-06-19 17:49:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 586): 12688.6 MiB
[2026-06-19 17:50:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2980, pooled_mean_dice=0.8110, per_class=['0.8110'], iou=0.6821, recall=0.9863, precision=0.6886, vol_sim=0.8223, mcc=0.8190, min_class_dice=0.8110, coverage=[77]/88 samples
[2026-06-19 17:50:14] INFO segtask_v1.trainer.trainer: Epoch 587/1000 | LR=4.59e-04 | loss=0.3054 | val_dice=0.8110 | best=0.8292 (ep441) | 04:20:45 | L_main=0.1515 L_aux_1=0.1386(w=0.5) L_aux_2=0.1692(w=0.5)
[2026-06-19 17:50:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 587): 12688.6 MiB
[2026-06-19 17:51:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2847, pooled_mean_dice=0.8166, per_class=['0.8166'], iou=0.6901, recall=0.9863, precision=0.6967, vol_sim=0.8280, mcc=0.8235, min_class_dice=0.8166, coverage=[75]/88 samples
[2026-06-19 17:51:15] INFO segtask_v1.trainer.trainer: Epoch 588/1000 | LR=4.63e-04 | loss=0.2791 | val_dice=0.8166 | best=0.8292 (ep441) | 04:21:47 | L_main=0.1400 L_aux_1=0.1300(w=0.5) L_aux_2=0.1481(w=0.5)
[2026-06-19 17:51:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 588): 12688.6 MiB
[2026-06-19 17:52:18] INFO segtask_v1.trainer.validation:   Val: loss=0.3289, pooled_mean_dice=0.7942, per_class=['0.7942'], iou=0.6587, recall=0.9843, precision=0.6657, vol_sim=0.8069, mcc=0.8044, min_class_dice=0.7942, coverage=[76]/88 samples
[2026-06-19 17:52:19] INFO segtask_v1.trainer.trainer: Epoch 589/1000 | LR=4.67e-04 | loss=0.3090 | val_dice=0.7942 | best=0.8292 (ep441) | 04:22:50 | L_main=0.1466 L_aux_1=0.1515(w=0.5) L_aux_2=0.1732(w=0.5)
[2026-06-19 17:52:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 589): 12688.6 MiB
[2026-06-19 17:53:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2941, pooled_mean_dice=0.8024, per_class=['0.8024'], iou=0.6699, recall=0.9862, precision=0.6763, vol_sim=0.8136, mcc=0.8118, min_class_dice=0.8024, coverage=[75]/88 samples
[2026-06-19 17:53:20] INFO segtask_v1.trainer.trainer: Epoch 590/1000 | LR=4.71e-04 | loss=0.2685 | val_dice=0.8024 | best=0.8292 (ep441) | 04:23:51 | L_main=0.1334 L_aux_1=0.1249(w=0.5) L_aux_2=0.1453(w=0.5)
[2026-06-19 17:53:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 590): 12688.6 MiB
[2026-06-19 17:54:24] INFO segtask_v1.trainer.validation:   Val: loss=0.2662, pooled_mean_dice=0.8070, per_class=['0.8070'], iou=0.6765, recall=0.9825, precision=0.6847, vol_sim=0.8213, mcc=0.8150, min_class_dice=0.8070, coverage=[68]/88 samples
[2026-06-19 17:54:24] INFO segtask_v1.trainer.trainer: Epoch 591/1000 | LR=4.75e-04 | loss=0.2707 | val_dice=0.8070 | best=0.8292 (ep441) | 04:24:55 | L_main=0.1376 L_aux_1=0.1221(w=0.5) L_aux_2=0.1443(w=0.5)
[2026-06-19 17:54:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 591): 12688.6 MiB
[2026-06-19 17:55:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2931, pooled_mean_dice=0.8101, per_class=['0.8101'], iou=0.6808, recall=0.9866, precision=0.6872, vol_sim=0.8211, mcc=0.8165, min_class_dice=0.8101, coverage=[83]/88 samples
[2026-06-19 17:55:27] INFO segtask_v1.trainer.trainer: Epoch 592/1000 | LR=4.79e-04 | loss=0.2501 | val_dice=0.8101 | best=0.8292 (ep441) | 04:25:58 | L_main=0.1251 L_aux_1=0.1172(w=0.5) L_aux_2=0.1328(w=0.5)
[2026-06-19 17:55:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 592): 12688.6 MiB
[2026-06-19 17:56:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3095, pooled_mean_dice=0.7909, per_class=['0.7909'], iou=0.6542, recall=0.9843, precision=0.6611, vol_sim=0.8036, mcc=0.8017, min_class_dice=0.7909, coverage=[77]/88 samples
[2026-06-19 17:56:28] INFO segtask_v1.trainer.trainer: Epoch 593/1000 | LR=4.83e-04 | loss=0.2198 | val_dice=0.7909 | best=0.8292 (ep441) | 04:27:00 | L_main=0.1117 L_aux_1=0.1023(w=0.5) L_aux_2=0.1140(w=0.5)
[2026-06-19 17:56:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 593): 12688.6 MiB
[2026-06-19 17:57:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3491, pooled_mean_dice=0.7983, per_class=['0.7983'], iou=0.6644, recall=0.9824, precision=0.6724, vol_sim=0.8127, mcc=0.8080, min_class_dice=0.7983, coverage=[74]/88 samples
[2026-06-19 17:57:31] INFO segtask_v1.trainer.trainer: Epoch 594/1000 | LR=4.87e-04 | loss=0.2191 | val_dice=0.7983 | best=0.8292 (ep441) | 04:28:02 | L_main=0.1119 L_aux_1=0.0986(w=0.5) L_aux_2=0.1158(w=0.5)
[2026-06-19 17:57:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 594): 12688.6 MiB
[2026-06-19 17:58:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3302, pooled_mean_dice=0.7926, per_class=['0.7926'], iou=0.6565, recall=0.9855, precision=0.6629, vol_sim=0.8043, mcc=0.8028, min_class_dice=0.7926, coverage=[80]/88 samples
[2026-06-19 17:58:34] INFO segtask_v1.trainer.trainer: Epoch 595/1000 | LR=4.91e-04 | loss=0.2737 | val_dice=0.7926 | best=0.8292 (ep441) | 04:29:05 | L_main=0.1394 L_aux_1=0.1292(w=0.5) L_aux_2=0.1393(w=0.5)
[2026-06-19 17:58:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 595): 12688.6 MiB
[2026-06-19 17:59:38] INFO segtask_v1.trainer.validation:   Val: loss=0.3236, pooled_mean_dice=0.7794, per_class=['0.7794'], iou=0.6386, recall=0.9849, precision=0.6449, vol_sim=0.7914, mcc=0.7926, min_class_dice=0.7794, coverage=[74]/88 samples
[2026-06-19 17:59:38] INFO segtask_v1.trainer.trainer: Epoch 596/1000 | LR=4.95e-04 | loss=0.2425 | val_dice=0.7794 | best=0.8292 (ep441) | 04:30:09 | L_main=0.1249 L_aux_1=0.1075(w=0.5) L_aux_2=0.1278(w=0.5)
[2026-06-19 17:59:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 596): 12688.6 MiB
[2026-06-19 18:00:40] INFO segtask_v1.trainer.validation:   Val: loss=0.2951, pooled_mean_dice=0.8121, per_class=['0.8121'], iou=0.6837, recall=0.9842, precision=0.6913, vol_sim=0.8252, mcc=0.8200, min_class_dice=0.8121, coverage=[75]/88 samples
[2026-06-19 18:00:40] INFO segtask_v1.trainer.trainer: Epoch 597/1000 | LR=4.99e-04 | loss=0.2419 | val_dice=0.8121 | best=0.8292 (ep441) | 04:31:11 | L_main=0.1208 L_aux_1=0.1116(w=0.5) L_aux_2=0.1307(w=0.5)
[2026-06-19 18:00:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 597): 12688.6 MiB
[2026-06-19 18:01:43] INFO segtask_v1.trainer.validation:   Val: loss=0.3301, pooled_mean_dice=0.8031, per_class=['0.8031'], iou=0.6710, recall=0.9797, precision=0.6804, vol_sim=0.8197, mcc=0.8116, min_class_dice=0.8031, coverage=[73]/88 samples
[2026-06-19 18:01:43] INFO segtask_v1.trainer.trainer: Epoch 598/1000 | LR=5.02e-04 | loss=0.2320 | val_dice=0.8031 | best=0.8292 (ep441) | 04:32:14 | L_main=0.1177 L_aux_1=0.1039(w=0.5) L_aux_2=0.1246(w=0.5)
[2026-06-19 18:01:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 598): 12688.6 MiB
[2026-06-19 18:02:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3108, pooled_mean_dice=0.8017, per_class=['0.8017'], iou=0.6690, recall=0.9836, precision=0.6765, vol_sim=0.8151, mcc=0.8115, min_class_dice=0.8017, coverage=[72]/88 samples
[2026-06-19 18:02:45] INFO segtask_v1.trainer.trainer: Epoch 599/1000 | LR=5.06e-04 | loss=0.2455 | val_dice=0.8017 | best=0.8292 (ep441) | 04:33:17 | L_main=0.1277 L_aux_1=0.1099(w=0.5) L_aux_2=0.1256(w=0.5)
[2026-06-19 18:02:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 599): 12688.6 MiB
[2026-06-19 18:03:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3444, pooled_mean_dice=0.7678, per_class=['0.7678'], iou=0.6231, recall=0.9872, precision=0.6282, vol_sim=0.7778, mcc=0.7834, min_class_dice=0.7678, coverage=[72]/88 samples
[2026-06-19 18:03:48] INFO segtask_v1.trainer.trainer: Epoch 600/1000 | LR=5.10e-04 | loss=0.2411 | val_dice=0.7678 | best=0.8292 (ep441) | 04:34:19 | L_main=0.1264 L_aux_1=0.1090(w=0.5) L_aux_2=0.1204(w=0.5)
[2026-06-19 18:03:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 600): 12688.6 MiB
[2026-06-19 18:04:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2799, pooled_mean_dice=0.8012, per_class=['0.8012'], iou=0.6683, recall=0.9848, precision=0.6752, vol_sim=0.8135, mcc=0.8103, min_class_dice=0.8012, coverage=[73]/88 samples
[2026-06-19 18:04:50] INFO segtask_v1.trainer.trainer: Epoch 601/1000 | LR=5.14e-04 | loss=0.2369 | val_dice=0.8012 | best=0.8292 (ep441) | 04:35:22 | L_main=0.1214 L_aux_1=0.1059(w=0.5) L_aux_2=0.1250(w=0.5)
[2026-06-19 18:04:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 601): 12688.6 MiB
[2026-06-19 18:05:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3546, pooled_mean_dice=0.7971, per_class=['0.7971'], iou=0.6626, recall=0.9852, precision=0.6693, vol_sim=0.8090, mcc=0.8067, min_class_dice=0.7971, coverage=[81]/88 samples
[2026-06-19 18:05:53] INFO segtask_v1.trainer.trainer: Epoch 602/1000 | LR=5.18e-04 | loss=0.2657 | val_dice=0.7971 | best=0.8292 (ep441) | 04:36:24 | L_main=0.1364 L_aux_1=0.1237(w=0.5) L_aux_2=0.1350(w=0.5)
[2026-06-19 18:05:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 602): 12688.6 MiB
[2026-06-19 18:06:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2968, pooled_mean_dice=0.8002, per_class=['0.8002'], iou=0.6669, recall=0.9886, precision=0.6721, vol_sim=0.8094, mcc=0.8098, min_class_dice=0.8002, coverage=[72]/88 samples
[2026-06-19 18:06:55] INFO segtask_v1.trainer.trainer: Epoch 603/1000 | LR=5.22e-04 | loss=0.2503 | val_dice=0.8002 | best=0.8292 (ep441) | 04:37:26 | L_main=0.1286 L_aux_1=0.1159(w=0.5) L_aux_2=0.1274(w=0.5)
[2026-06-19 18:06:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 603): 12688.6 MiB
[2026-06-19 18:07:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3029, pooled_mean_dice=0.8103, per_class=['0.8103'], iou=0.6811, recall=0.9868, precision=0.6874, vol_sim=0.8211, mcc=0.8183, min_class_dice=0.8103, coverage=[76]/88 samples
[2026-06-19 18:07:57] INFO segtask_v1.trainer.trainer: Epoch 604/1000 | LR=5.26e-04 | loss=0.2270 | val_dice=0.8103 | best=0.8292 (ep441) | 04:38:28 | L_main=0.1150 L_aux_1=0.1063(w=0.5) L_aux_2=0.1179(w=0.5)
[2026-06-19 18:07:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 604): 12688.6 MiB
[2026-06-19 18:09:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3598, pooled_mean_dice=0.8017, per_class=['0.8017'], iou=0.6690, recall=0.9833, precision=0.6767, vol_sim=0.8153, mcc=0.8115, min_class_dice=0.8017, coverage=[79]/88 samples
[2026-06-19 18:09:00] INFO segtask_v1.trainer.trainer: Epoch 605/1000 | LR=5.30e-04 | loss=0.2402 | val_dice=0.8017 | best=0.8292 (ep441) | 04:39:31 | L_main=0.1242 L_aux_1=0.1114(w=0.5) L_aux_2=0.1205(w=0.5)
[2026-06-19 18:09:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 605): 12688.6 MiB
[2026-06-19 18:10:03] INFO segtask_v1.trainer.validation:   Val: loss=0.3348, pooled_mean_dice=0.7997, per_class=['0.7997'], iou=0.6663, recall=0.9855, precision=0.6728, vol_sim=0.8114, mcc=0.8092, min_class_dice=0.7997, coverage=[80]/88 samples
[2026-06-19 18:10:03] INFO segtask_v1.trainer.trainer: Epoch 606/1000 | LR=5.34e-04 | loss=0.2596 | val_dice=0.7997 | best=0.8292 (ep441) | 04:40:34 | L_main=0.1296 L_aux_1=0.1218(w=0.5) L_aux_2=0.1383(w=0.5)
[2026-06-19 18:10:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 606): 12688.6 MiB
[2026-06-19 18:11:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3056, pooled_mean_dice=0.7962, per_class=['0.7962'], iou=0.6614, recall=0.9832, precision=0.6690, vol_sim=0.8099, mcc=0.8062, min_class_dice=0.7962, coverage=[74]/88 samples
[2026-06-19 18:11:05] INFO segtask_v1.trainer.trainer: Epoch 607/1000 | LR=5.38e-04 | loss=0.2804 | val_dice=0.7962 | best=0.8292 (ep441) | 04:41:37 | L_main=0.1386 L_aux_1=0.1315(w=0.5) L_aux_2=0.1521(w=0.5)
[2026-06-19 18:11:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 607): 12688.6 MiB
[2026-06-19 18:12:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3070, pooled_mean_dice=0.8013, per_class=['0.8013'], iou=0.6685, recall=0.9836, precision=0.6761, vol_sim=0.8147, mcc=0.8111, min_class_dice=0.8013, coverage=[72]/88 samples
[2026-06-19 18:12:07] INFO segtask_v1.trainer.trainer: Epoch 608/1000 | LR=5.42e-04 | loss=0.2463 | val_dice=0.8013 | best=0.8292 (ep441) | 04:42:38 | L_main=0.1256 L_aux_1=0.1087(w=0.5) L_aux_2=0.1328(w=0.5)
[2026-06-19 18:12:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 608): 12688.6 MiB
[2026-06-19 18:13:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3277, pooled_mean_dice=0.8164, per_class=['0.8164'], iou=0.6898, recall=0.9848, precision=0.6972, vol_sim=0.8290, mcc=0.8236, min_class_dice=0.8164, coverage=[81]/88 samples
[2026-06-19 18:13:09] INFO segtask_v1.trainer.trainer: Epoch 609/1000 | LR=5.46e-04 | loss=0.2394 | val_dice=0.8164 | best=0.8292 (ep441) | 04:43:40 | L_main=0.1193 L_aux_1=0.1118(w=0.5) L_aux_2=0.1285(w=0.5)
[2026-06-19 18:13:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 609): 12688.6 MiB
[2026-06-19 18:14:12] INFO segtask_v1.trainer.validation:   Val: loss=0.3326, pooled_mean_dice=0.7905, per_class=['0.7905'], iou=0.6536, recall=0.9850, precision=0.6602, vol_sim=0.8026, mcc=0.8018, min_class_dice=0.7905, coverage=[76]/88 samples
[2026-06-19 18:14:12] INFO segtask_v1.trainer.trainer: Epoch 610/1000 | LR=5.50e-04 | loss=0.2736 | val_dice=0.7905 | best=0.8292 (ep441) | 04:44:43 | L_main=0.1399 L_aux_1=0.1240(w=0.5) L_aux_2=0.1435(w=0.5)
[2026-06-19 18:14:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 610): 12688.6 MiB
[2026-06-19 18:15:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2934, pooled_mean_dice=0.8034, per_class=['0.8034'], iou=0.6714, recall=0.9851, precision=0.6783, vol_sim=0.8156, mcc=0.8129, min_class_dice=0.8034, coverage=[66]/88 samples
[2026-06-19 18:15:15] INFO segtask_v1.trainer.trainer: Epoch 611/1000 | LR=5.54e-04 | loss=0.2496 | val_dice=0.8034 | best=0.8292 (ep441) | 04:45:46 | L_main=0.1271 L_aux_1=0.1135(w=0.5) L_aux_2=0.1315(w=0.5)
[2026-06-19 18:15:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 611): 12688.6 MiB
[2026-06-19 18:16:16] INFO segtask_v1.trainer.validation:   Val: loss=0.3507, pooled_mean_dice=0.7991, per_class=['0.7991'], iou=0.6654, recall=0.9851, precision=0.6721, vol_sim=0.8111, mcc=0.8093, min_class_dice=0.7991, coverage=[76]/88 samples
[2026-06-19 18:16:16] INFO segtask_v1.trainer.trainer: Epoch 612/1000 | LR=5.58e-04 | loss=0.2497 | val_dice=0.7991 | best=0.8292 (ep441) | 04:46:47 | L_main=0.1249 L_aux_1=0.1140(w=0.5) L_aux_2=0.1355(w=0.5)
[2026-06-19 18:16:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 612): 12688.6 MiB
[2026-06-19 18:17:17] INFO segtask_v1.trainer.validation:   Val: loss=0.3410, pooled_mean_dice=0.8000, per_class=['0.8000'], iou=0.6667, recall=0.9819, precision=0.6750, vol_sim=0.8147, mcc=0.8093, min_class_dice=0.8000, coverage=[81]/88 samples
[2026-06-19 18:17:17] INFO segtask_v1.trainer.trainer: Epoch 613/1000 | LR=5.62e-04 | loss=0.2346 | val_dice=0.8000 | best=0.8292 (ep441) | 04:47:49 | L_main=0.1212 L_aux_1=0.1075(w=0.5) L_aux_2=0.1192(w=0.5)
[2026-06-19 18:17:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 613): 12688.6 MiB
[2026-06-19 18:18:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3324, pooled_mean_dice=0.7941, per_class=['0.7941'], iou=0.6586, recall=0.9840, precision=0.6657, vol_sim=0.8071, mcc=0.8044, min_class_dice=0.7941, coverage=[74]/88 samples
[2026-06-19 18:18:20] INFO segtask_v1.trainer.trainer: Epoch 614/1000 | LR=5.66e-04 | loss=0.2407 | val_dice=0.7941 | best=0.8292 (ep441) | 04:48:51 | L_main=0.1228 L_aux_1=0.1117(w=0.5) L_aux_2=0.1241(w=0.5)
[2026-06-19 18:18:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 614): 12688.6 MiB
[2026-06-19 18:19:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3179, pooled_mean_dice=0.8070, per_class=['0.8070'], iou=0.6764, recall=0.9869, precision=0.6825, vol_sim=0.8177, mcc=0.8152, min_class_dice=0.8070, coverage=[79]/88 samples
[2026-06-19 18:19:23] INFO segtask_v1.trainer.trainer: Epoch 615/1000 | LR=5.70e-04 | loss=0.2968 | val_dice=0.8070 | best=0.8292 (ep441) | 04:49:55 | L_main=0.1492 L_aux_1=0.1398(w=0.5) L_aux_2=0.1555(w=0.5)
[2026-06-19 18:19:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 615): 12688.6 MiB
[2026-06-19 18:20:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3057, pooled_mean_dice=0.7924, per_class=['0.7924'], iou=0.6562, recall=0.9854, precision=0.6627, vol_sim=0.8042, mcc=0.8035, min_class_dice=0.7924, coverage=[72]/88 samples
[2026-06-19 18:20:26] INFO segtask_v1.trainer.trainer: Epoch 616/1000 | LR=5.74e-04 | loss=0.2651 | val_dice=0.7924 | best=0.8292 (ep441) | 04:50:57 | L_main=0.1309 L_aux_1=0.1271(w=0.5) L_aux_2=0.1413(w=0.5)
[2026-06-19 18:20:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 616): 12688.6 MiB
[2026-06-19 18:21:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2880, pooled_mean_dice=0.8030, per_class=['0.8030'], iou=0.6708, recall=0.9855, precision=0.6775, vol_sim=0.8148, mcc=0.8122, min_class_dice=0.8030, coverage=[73]/88 samples
[2026-06-19 18:21:27] INFO segtask_v1.trainer.trainer: Epoch 617/1000 | LR=5.78e-04 | loss=0.2710 | val_dice=0.8030 | best=0.8292 (ep441) | 04:51:59 | L_main=0.1354 L_aux_1=0.1239(w=0.5) L_aux_2=0.1473(w=0.5)
[2026-06-19 18:21:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 617): 12688.6 MiB
[2026-06-19 18:22:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2898, pooled_mean_dice=0.7877, per_class=['0.7877'], iou=0.6497, recall=0.9870, precision=0.6553, vol_sim=0.7980, mcc=0.7984, min_class_dice=0.7877, coverage=[73]/88 samples
[2026-06-19 18:22:31] INFO segtask_v1.trainer.trainer: Epoch 618/1000 | LR=5.82e-04 | loss=0.2402 | val_dice=0.7877 | best=0.8292 (ep441) | 04:53:02 | L_main=0.1208 L_aux_1=0.1104(w=0.5) L_aux_2=0.1284(w=0.5)
[2026-06-19 18:22:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 618): 12688.6 MiB
[2026-06-19 18:23:35] INFO segtask_v1.trainer.validation:   Val: loss=0.3206, pooled_mean_dice=0.8081, per_class=['0.8081'], iou=0.6780, recall=0.9860, precision=0.6846, vol_sim=0.8196, mcc=0.8163, min_class_dice=0.8081, coverage=[71]/88 samples
[2026-06-19 18:23:35] INFO segtask_v1.trainer.trainer: Epoch 619/1000 | LR=5.85e-04 | loss=0.2482 | val_dice=0.8081 | best=0.8292 (ep441) | 04:54:06 | L_main=0.1252 L_aux_1=0.1172(w=0.5) L_aux_2=0.1288(w=0.5)
[2026-06-19 18:23:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 619): 12688.6 MiB
[2026-06-19 18:24:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3498, pooled_mean_dice=0.7989, per_class=['0.7989'], iou=0.6651, recall=0.9842, precision=0.6723, vol_sim=0.8118, mcc=0.8086, min_class_dice=0.7989, coverage=[77]/88 samples
[2026-06-19 18:24:37] INFO segtask_v1.trainer.trainer: Epoch 620/1000 | LR=5.89e-04 | loss=0.3694 | val_dice=0.7989 | best=0.8292 (ep441) | 04:55:09 | L_main=0.1862 L_aux_1=0.1763(w=0.5) L_aux_2=0.1900(w=0.5)
[2026-06-19 18:24:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 620): 12688.6 MiB
[2026-06-19 18:25:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3263, pooled_mean_dice=0.7879, per_class=['0.7879'], iou=0.6500, recall=0.9845, precision=0.6567, vol_sim=0.8003, mcc=0.7990, min_class_dice=0.7879, coverage=[77]/88 samples
[2026-06-19 18:25:40] INFO segtask_v1.trainer.trainer: Epoch 621/1000 | LR=5.93e-04 | loss=0.2893 | val_dice=0.7879 | best=0.8292 (ep441) | 04:56:11 | L_main=0.1450 L_aux_1=0.1368(w=0.5) L_aux_2=0.1518(w=0.5)
[2026-06-19 18:25:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 621): 12688.6 MiB
[2026-06-19 18:26:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3361, pooled_mean_dice=0.7947, per_class=['0.7947'], iou=0.6594, recall=0.9845, precision=0.6663, vol_sim=0.8072, mcc=0.8043, min_class_dice=0.7947, coverage=[82]/88 samples
[2026-06-19 18:26:44] INFO segtask_v1.trainer.trainer: Epoch 622/1000 | LR=5.97e-04 | loss=0.2801 | val_dice=0.7947 | best=0.8292 (ep441) | 04:57:15 | L_main=0.1405 L_aux_1=0.1296(w=0.5) L_aux_2=0.1496(w=0.5)
[2026-06-19 18:26:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 622): 12688.6 MiB
[2026-06-19 18:27:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3153, pooled_mean_dice=0.7811, per_class=['0.7811'], iou=0.6408, recall=0.9854, precision=0.6469, vol_sim=0.7927, mcc=0.7927, min_class_dice=0.7811, coverage=[77]/88 samples
[2026-06-19 18:27:46] INFO segtask_v1.trainer.trainer: Epoch 623/1000 | LR=6.01e-04 | loss=0.3321 | val_dice=0.7811 | best=0.8292 (ep441) | 04:58:18 | L_main=0.1679 L_aux_1=0.1581(w=0.5) L_aux_2=0.1705(w=0.5)
[2026-06-19 18:27:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 623): 12688.6 MiB
[2026-06-19 18:28:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2955, pooled_mean_dice=0.7944, per_class=['0.7944'], iou=0.6590, recall=0.9862, precision=0.6651, vol_sim=0.8055, mcc=0.8046, min_class_dice=0.7944, coverage=[75]/88 samples
[2026-06-19 18:28:49] INFO segtask_v1.trainer.trainer: Epoch 624/1000 | LR=6.05e-04 | loss=0.2889 | val_dice=0.7944 | best=0.8292 (ep441) | 04:59:20 | L_main=0.1469 L_aux_1=0.1321(w=0.5) L_aux_2=0.1517(w=0.5)
[2026-06-19 18:28:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 624): 12688.6 MiB
[2026-06-19 18:29:51] INFO segtask_v1.trainer.validation:   Val: loss=0.3058, pooled_mean_dice=0.7921, per_class=['0.7921'], iou=0.6557, recall=0.9853, precision=0.6622, vol_sim=0.8038, mcc=0.8027, min_class_dice=0.7921, coverage=[69]/88 samples
[2026-06-19 18:29:51] INFO segtask_v1.trainer.trainer: Epoch 625/1000 | LR=6.09e-04 | loss=0.2762 | val_dice=0.7921 | best=0.8292 (ep441) | 05:00:22 | L_main=0.1403 L_aux_1=0.1286(w=0.5) L_aux_2=0.1431(w=0.5)
[2026-06-19 18:29:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 625): 12688.6 MiB
[2026-06-19 18:30:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3104, pooled_mean_dice=0.7899, per_class=['0.7899'], iou=0.6528, recall=0.9863, precision=0.6587, vol_sim=0.8009, mcc=0.8004, min_class_dice=0.7899, coverage=[77]/88 samples
[2026-06-19 18:30:54] INFO segtask_v1.trainer.trainer: Epoch 626/1000 | LR=6.13e-04 | loss=0.2675 | val_dice=0.7899 | best=0.8292 (ep441) | 05:01:26 | L_main=0.1307 L_aux_1=0.1230(w=0.5) L_aux_2=0.1506(w=0.5)
[2026-06-19 18:30:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 626): 12688.6 MiB
[2026-06-19 18:31:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3075, pooled_mean_dice=0.7962, per_class=['0.7962'], iou=0.6614, recall=0.9862, precision=0.6676, vol_sim=0.8074, mcc=0.8061, min_class_dice=0.7962, coverage=[76]/88 samples
[2026-06-19 18:31:58] INFO segtask_v1.trainer.trainer: Epoch 627/1000 | LR=6.17e-04 | loss=0.2354 | val_dice=0.7962 | best=0.8292 (ep441) | 05:02:29 | L_main=0.1212 L_aux_1=0.1053(w=0.5) L_aux_2=0.1229(w=0.5)
[2026-06-19 18:31:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 627): 12688.6 MiB
[2026-06-19 18:32:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2556, pooled_mean_dice=0.8165, per_class=['0.8165'], iou=0.6900, recall=0.9880, precision=0.6958, vol_sim=0.8265, mcc=0.8237, min_class_dice=0.8165, coverage=[70]/88 samples
[2026-06-19 18:32:59] INFO segtask_v1.trainer.trainer: Epoch 628/1000 | LR=6.20e-04 | loss=0.2269 | val_dice=0.8165 | best=0.8292 (ep441) | 05:03:30 | L_main=0.1174 L_aux_1=0.1044(w=0.5) L_aux_2=0.1145(w=0.5)
[2026-06-19 18:32:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 628): 12688.6 MiB
[2026-06-19 18:34:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2941, pooled_mean_dice=0.8035, per_class=['0.8035'], iou=0.6716, recall=0.9859, precision=0.6781, vol_sim=0.8150, mcc=0.8121, min_class_dice=0.8035, coverage=[75]/88 samples
[2026-06-19 18:34:02] INFO segtask_v1.trainer.trainer: Epoch 629/1000 | LR=6.24e-04 | loss=0.2943 | val_dice=0.8035 | best=0.8292 (ep441) | 05:04:33 | L_main=0.1522 L_aux_1=0.1379(w=0.5) L_aux_2=0.1463(w=0.5)
[2026-06-19 18:34:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 629): 12688.6 MiB
[2026-06-19 18:35:05] INFO segtask_v1.trainer.validation:   Val: loss=0.2898, pooled_mean_dice=0.8136, per_class=['0.8136'], iou=0.6858, recall=0.9855, precision=0.6928, vol_sim=0.8256, mcc=0.8205, min_class_dice=0.8136, coverage=[78]/88 samples
[2026-06-19 18:35:05] INFO segtask_v1.trainer.trainer: Epoch 630/1000 | LR=6.28e-04 | loss=0.3079 | val_dice=0.8136 | best=0.8292 (ep441) | 05:05:36 | L_main=0.1559 L_aux_1=0.1446(w=0.5) L_aux_2=0.1593(w=0.5)
[2026-06-19 18:35:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 630): 12688.6 MiB
[2026-06-19 18:36:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2859, pooled_mean_dice=0.7968, per_class=['0.7968'], iou=0.6622, recall=0.9839, precision=0.6695, vol_sim=0.8098, mcc=0.8063, min_class_dice=0.7968, coverage=[71]/88 samples
[2026-06-19 18:36:08] INFO segtask_v1.trainer.trainer: Epoch 631/1000 | LR=6.32e-04 | loss=0.2640 | val_dice=0.7968 | best=0.8292 (ep441) | 05:06:39 | L_main=0.1345 L_aux_1=0.1206(w=0.5) L_aux_2=0.1384(w=0.5)
[2026-06-19 18:36:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 631): 12688.6 MiB
[2026-06-19 18:37:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2988, pooled_mean_dice=0.7985, per_class=['0.7985'], iou=0.6646, recall=0.9824, precision=0.6726, vol_sim=0.8128, mcc=0.8071, min_class_dice=0.7985, coverage=[77]/88 samples
[2026-06-19 18:37:10] INFO segtask_v1.trainer.trainer: Epoch 632/1000 | LR=6.36e-04 | loss=0.2705 | val_dice=0.7985 | best=0.8292 (ep441) | 05:07:41 | L_main=0.1352 L_aux_1=0.1277(w=0.5) L_aux_2=0.1429(w=0.5)
[2026-06-19 18:37:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 632): 12688.6 MiB
[2026-06-19 18:38:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3172, pooled_mean_dice=0.7852, per_class=['0.7852'], iou=0.6463, recall=0.9815, precision=0.6543, vol_sim=0.7999, mcc=0.7973, min_class_dice=0.7852, coverage=[69]/88 samples
[2026-06-19 18:38:13] INFO segtask_v1.trainer.trainer: Epoch 633/1000 | LR=6.40e-04 | loss=0.2795 | val_dice=0.7852 | best=0.8292 (ep441) | 05:08:44 | L_main=0.1387 L_aux_1=0.1292(w=0.5) L_aux_2=0.1523(w=0.5)
[2026-06-19 18:38:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 633): 12688.6 MiB
[2026-06-19 18:39:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2938, pooled_mean_dice=0.7850, per_class=['0.7850'], iou=0.6461, recall=0.9824, precision=0.6536, vol_sim=0.7990, mcc=0.7967, min_class_dice=0.7850, coverage=[71]/88 samples
[2026-06-19 18:39:16] INFO segtask_v1.trainer.trainer: Epoch 634/1000 | LR=6.43e-04 | loss=0.2774 | val_dice=0.7850 | best=0.8292 (ep441) | 05:09:47 | L_main=0.1453 L_aux_1=0.1243(w=0.5) L_aux_2=0.1399(w=0.5)
[2026-06-19 18:39:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 634): 12688.6 MiB
[2026-06-19 18:40:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3191, pooled_mean_dice=0.7965, per_class=['0.7965'], iou=0.6618, recall=0.9874, precision=0.6674, vol_sim=0.8067, mcc=0.8055, min_class_dice=0.7965, coverage=[78]/88 samples
[2026-06-19 18:40:20] INFO segtask_v1.trainer.trainer: Epoch 635/1000 | LR=6.47e-04 | loss=0.3163 | val_dice=0.7965 | best=0.8292 (ep441) | 05:10:51 | L_main=0.1627 L_aux_1=0.1459(w=0.5) L_aux_2=0.1614(w=0.5)
[2026-06-19 18:40:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 635): 12688.6 MiB
[2026-06-19 18:41:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3133, pooled_mean_dice=0.7759, per_class=['0.7759'], iou=0.6339, recall=0.9838, precision=0.6405, vol_sim=0.7887, mcc=0.7889, min_class_dice=0.7759, coverage=[78]/88 samples
[2026-06-19 18:41:23] INFO segtask_v1.trainer.trainer: Epoch 636/1000 | LR=6.51e-04 | loss=0.2888 | val_dice=0.7759 | best=0.8292 (ep441) | 05:11:54 | L_main=0.1450 L_aux_1=0.1333(w=0.5) L_aux_2=0.1544(w=0.5)
[2026-06-19 18:41:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 636): 12688.6 MiB
[2026-06-19 18:42:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2986, pooled_mean_dice=0.7835, per_class=['0.7835'], iou=0.6441, recall=0.9846, precision=0.6506, vol_sim=0.7958, mcc=0.7954, min_class_dice=0.7835, coverage=[72]/88 samples
[2026-06-19 18:42:26] INFO segtask_v1.trainer.trainer: Epoch 637/1000 | LR=6.55e-04 | loss=0.2960 | val_dice=0.7835 | best=0.8292 (ep441) | 05:12:57 | L_main=0.1500 L_aux_1=0.1363(w=0.5) L_aux_2=0.1557(w=0.5)
[2026-06-19 18:42:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 637): 12688.6 MiB
[2026-06-19 18:43:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3476, pooled_mean_dice=0.7761, per_class=['0.7761'], iou=0.6341, recall=0.9847, precision=0.6404, vol_sim=0.7882, mcc=0.7888, min_class_dice=0.7761, coverage=[78]/88 samples
[2026-06-19 18:43:28] INFO segtask_v1.trainer.trainer: Epoch 638/1000 | LR=6.59e-04 | loss=0.2767 | val_dice=0.7761 | best=0.8292 (ep441) | 05:13:59 | L_main=0.1401 L_aux_1=0.1262(w=0.5) L_aux_2=0.1471(w=0.5)
[2026-06-19 18:43:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 638): 12688.6 MiB
[2026-06-19 18:44:30] INFO segtask_v1.trainer.validation:   Val: loss=0.3038, pooled_mean_dice=0.7863, per_class=['0.7863'], iou=0.6478, recall=0.9866, precision=0.6536, vol_sim=0.7970, mcc=0.7978, min_class_dice=0.7863, coverage=[69]/88 samples
[2026-06-19 18:44:30] INFO segtask_v1.trainer.trainer: Epoch 639/1000 | LR=6.62e-04 | loss=0.2414 | val_dice=0.7863 | best=0.8292 (ep441) | 05:15:01 | L_main=0.1244 L_aux_1=0.1086(w=0.5) L_aux_2=0.1255(w=0.5)
[2026-06-19 18:44:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 639): 12688.6 MiB
[2026-06-19 18:45:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3140, pooled_mean_dice=0.7580, per_class=['0.7580'], iou=0.6103, recall=0.9858, precision=0.6157, vol_sim=0.7689, mcc=0.7740, min_class_dice=0.7580, coverage=[74]/88 samples
[2026-06-19 18:45:31] INFO segtask_v1.trainer.trainer: Epoch 640/1000 | LR=6.66e-04 | loss=0.2599 | val_dice=0.7580 | best=0.8292 (ep441) | 05:16:03 | L_main=0.1323 L_aux_1=0.1215(w=0.5) L_aux_2=0.1337(w=0.5)
[2026-06-19 18:45:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 640): 12688.6 MiB
[2026-06-19 18:46:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3172, pooled_mean_dice=0.7965, per_class=['0.7965'], iou=0.6618, recall=0.9866, precision=0.6678, vol_sim=0.8073, mcc=0.8065, min_class_dice=0.7965, coverage=[76]/88 samples
[2026-06-19 18:46:34] INFO segtask_v1.trainer.trainer: Epoch 641/1000 | LR=6.70e-04 | loss=0.2821 | val_dice=0.7965 | best=0.8292 (ep441) | 05:17:05 | L_main=0.1443 L_aux_1=0.1333(w=0.5) L_aux_2=0.1423(w=0.5)
[2026-06-19 18:46:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 641): 12688.6 MiB
[2026-06-19 18:47:36] INFO segtask_v1.trainer.validation:   Val: loss=0.3231, pooled_mean_dice=0.7928, per_class=['0.7928'], iou=0.6567, recall=0.9871, precision=0.6624, vol_sim=0.8031, mcc=0.8031, min_class_dice=0.7928, coverage=[75]/88 samples
[2026-06-19 18:47:36] INFO segtask_v1.trainer.trainer: Epoch 642/1000 | LR=6.74e-04 | loss=0.2670 | val_dice=0.7928 | best=0.8292 (ep441) | 05:18:07 | L_main=0.1368 L_aux_1=0.1211(w=0.5) L_aux_2=0.1393(w=0.5)
[2026-06-19 18:47:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 642): 12688.6 MiB
[2026-06-19 18:48:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3300, pooled_mean_dice=0.7967, per_class=['0.7967'], iou=0.6621, recall=0.9836, precision=0.6695, vol_sim=0.8100, mcc=0.8061, min_class_dice=0.7967, coverage=[78]/88 samples
[2026-06-19 18:48:39] INFO segtask_v1.trainer.trainer: Epoch 643/1000 | LR=6.77e-04 | loss=0.2722 | val_dice=0.7967 | best=0.8292 (ep441) | 05:19:10 | L_main=0.1416 L_aux_1=0.1231(w=0.5) L_aux_2=0.1382(w=0.5)
[2026-06-19 18:48:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 643): 12688.6 MiB
[2026-06-19 18:49:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3189, pooled_mean_dice=0.7983, per_class=['0.7983'], iou=0.6643, recall=0.9840, precision=0.6716, vol_sim=0.8113, mcc=0.8065, min_class_dice=0.7983, coverage=[80]/88 samples
[2026-06-19 18:49:42] INFO segtask_v1.trainer.trainer: Epoch 644/1000 | LR=6.81e-04 | loss=0.2893 | val_dice=0.7983 | best=0.8292 (ep441) | 05:20:13 | L_main=0.1458 L_aux_1=0.1380(w=0.5) L_aux_2=0.1491(w=0.5)
[2026-06-19 18:49:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 644): 12688.6 MiB
[2026-06-19 18:50:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3147, pooled_mean_dice=0.7750, per_class=['0.7750'], iou=0.6327, recall=0.9855, precision=0.6386, vol_sim=0.7864, mcc=0.7874, min_class_dice=0.7750, coverage=[78]/88 samples
[2026-06-19 18:50:45] INFO segtask_v1.trainer.trainer: Epoch 645/1000 | LR=6.85e-04 | loss=0.2551 | val_dice=0.7750 | best=0.8292 (ep441) | 05:21:16 | L_main=0.1297 L_aux_1=0.1187(w=0.5) L_aux_2=0.1321(w=0.5)
[2026-06-19 18:50:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 645): 12688.6 MiB
[2026-06-19 18:51:47] INFO segtask_v1.trainer.validation:   Val: loss=0.3261, pooled_mean_dice=0.7906, per_class=['0.7906'], iou=0.6537, recall=0.9852, precision=0.6602, vol_sim=0.8025, mcc=0.8014, min_class_dice=0.7906, coverage=[77]/88 samples
[2026-06-19 18:51:47] INFO segtask_v1.trainer.trainer: Epoch 646/1000 | LR=6.88e-04 | loss=0.2880 | val_dice=0.7906 | best=0.8292 (ep441) | 05:22:18 | L_main=0.1482 L_aux_1=0.1343(w=0.5) L_aux_2=0.1454(w=0.5)
[2026-06-19 18:51:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 646): 12688.6 MiB
[2026-06-19 18:52:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3361, pooled_mean_dice=0.7724, per_class=['0.7724'], iou=0.6293, recall=0.9868, precision=0.6346, vol_sim=0.7828, mcc=0.7867, min_class_dice=0.7724, coverage=[75]/88 samples
[2026-06-19 18:52:49] INFO segtask_v1.trainer.trainer: Epoch 647/1000 | LR=6.92e-04 | loss=0.2667 | val_dice=0.7724 | best=0.8292 (ep441) | 05:23:20 | L_main=0.1345 L_aux_1=0.1222(w=0.5) L_aux_2=0.1420(w=0.5)
[2026-06-19 18:52:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 647): 12688.6 MiB
[2026-06-19 18:53:52] INFO segtask_v1.trainer.validation:   Val: loss=0.3141, pooled_mean_dice=0.7949, per_class=['0.7949'], iou=0.6596, recall=0.9860, precision=0.6658, vol_sim=0.8062, mcc=0.8048, min_class_dice=0.7949, coverage=[73]/88 samples
[2026-06-19 18:53:52] INFO segtask_v1.trainer.trainer: Epoch 648/1000 | LR=6.96e-04 | loss=0.2714 | val_dice=0.7949 | best=0.8292 (ep441) | 05:24:23 | L_main=0.1373 L_aux_1=0.1262(w=0.5) L_aux_2=0.1419(w=0.5)
[2026-06-19 18:53:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 648): 12688.6 MiB
[2026-06-19 18:54:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3465, pooled_mean_dice=0.7743, per_class=['0.7743'], iou=0.6317, recall=0.9814, precision=0.6393, vol_sim=0.7889, mcc=0.7868, min_class_dice=0.7743, coverage=[82]/88 samples
[2026-06-19 18:54:54] INFO segtask_v1.trainer.trainer: Epoch 649/1000 | LR=6.99e-04 | loss=0.2756 | val_dice=0.7743 | best=0.8292 (ep441) | 05:25:25 | L_main=0.1436 L_aux_1=0.1252(w=0.5) L_aux_2=0.1388(w=0.5)
[2026-06-19 18:54:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 649): 12688.6 MiB
[2026-06-19 18:55:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3362, pooled_mean_dice=0.7902, per_class=['0.7902'], iou=0.6531, recall=0.9836, precision=0.6603, vol_sim=0.8033, mcc=0.8005, min_class_dice=0.7902, coverage=[79]/88 samples
[2026-06-19 18:55:57] INFO segtask_v1.trainer.trainer: Epoch 650/1000 | LR=7.03e-04 | loss=0.2603 | val_dice=0.7902 | best=0.8292 (ep441) | 05:26:28 | L_main=0.1315 L_aux_1=0.1241(w=0.5) L_aux_2=0.1335(w=0.5)
[2026-06-19 18:55:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 650): 12688.6 MiB
[2026-06-19 18:57:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3262, pooled_mean_dice=0.8029, per_class=['0.8029'], iou=0.6706, recall=0.9862, precision=0.6770, vol_sim=0.8141, mcc=0.8116, min_class_dice=0.8029, coverage=[75]/88 samples
[2026-06-19 18:57:00] INFO segtask_v1.trainer.trainer: Epoch 651/1000 | LR=7.07e-04 | loss=0.2575 | val_dice=0.8029 | best=0.8292 (ep441) | 05:27:32 | L_main=0.1316 L_aux_1=0.1153(w=0.5) L_aux_2=0.1365(w=0.5)
[2026-06-19 18:57:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 651): 12688.6 MiB
[2026-06-19 18:58:03] INFO segtask_v1.trainer.validation:   Val: loss=0.3105, pooled_mean_dice=0.7982, per_class=['0.7982'], iou=0.6642, recall=0.9866, precision=0.6703, vol_sim=0.8091, mcc=0.8070, min_class_dice=0.7982, coverage=[78]/88 samples
[2026-06-19 18:58:03] INFO segtask_v1.trainer.trainer: Epoch 652/1000 | LR=7.10e-04 | loss=0.2404 | val_dice=0.7982 | best=0.8292 (ep441) | 05:28:35 | L_main=0.1243 L_aux_1=0.1108(w=0.5) L_aux_2=0.1214(w=0.5)
[2026-06-19 18:58:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 652): 12688.6 MiB
[2026-06-19 18:59:06] INFO segtask_v1.trainer.validation:   Val: loss=0.3205, pooled_mean_dice=0.7956, per_class=['0.7956'], iou=0.6606, recall=0.9845, precision=0.6676, vol_sim=0.8082, mcc=0.8063, min_class_dice=0.7956, coverage=[71]/88 samples
[2026-06-19 18:59:06] INFO segtask_v1.trainer.trainer: Epoch 653/1000 | LR=7.14e-04 | loss=0.2496 | val_dice=0.7956 | best=0.8292 (ep441) | 05:29:37 | L_main=0.1235 L_aux_1=0.1240(w=0.5) L_aux_2=0.1283(w=0.5)
[2026-06-19 18:59:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 653): 12688.6 MiB
[2026-06-19 19:00:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3323, pooled_mean_dice=0.8059, per_class=['0.8059'], iou=0.6749, recall=0.9868, precision=0.6810, vol_sim=0.8167, mcc=0.8152, min_class_dice=0.8059, coverage=[76]/88 samples
[2026-06-19 19:00:09] INFO segtask_v1.trainer.trainer: Epoch 654/1000 | LR=7.17e-04 | loss=0.2646 | val_dice=0.8059 | best=0.8292 (ep441) | 05:30:40 | L_main=0.1335 L_aux_1=0.1234(w=0.5) L_aux_2=0.1388(w=0.5)
[2026-06-19 19:00:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 654): 12688.6 MiB
[2026-06-19 19:01:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3379, pooled_mean_dice=0.8021, per_class=['0.8021'], iou=0.6696, recall=0.9830, precision=0.6774, vol_sim=0.8160, mcc=0.8099, min_class_dice=0.8021, coverage=[81]/88 samples
[2026-06-19 19:01:11] INFO segtask_v1.trainer.trainer: Epoch 655/1000 | LR=7.21e-04 | loss=0.2518 | val_dice=0.8021 | best=0.8292 (ep441) | 05:31:42 | L_main=0.1326 L_aux_1=0.1142(w=0.5) L_aux_2=0.1240(w=0.5)
[2026-06-19 19:01:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 655): 12688.6 MiB
[2026-06-19 19:02:14] INFO segtask_v1.trainer.validation:   Val: loss=0.3280, pooled_mean_dice=0.7894, per_class=['0.7894'], iou=0.6521, recall=0.9865, precision=0.6579, vol_sim=0.8002, mcc=0.8000, min_class_dice=0.7894, coverage=[75]/88 samples
[2026-06-19 19:02:14] INFO segtask_v1.trainer.trainer: Epoch 656/1000 | LR=7.25e-04 | loss=0.2722 | val_dice=0.7894 | best=0.8292 (ep441) | 05:32:46 | L_main=0.1383 L_aux_1=0.1281(w=0.5) L_aux_2=0.1398(w=0.5)
[2026-06-19 19:02:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 656): 12688.6 MiB
[2026-06-19 19:03:18] INFO segtask_v1.trainer.validation:   Val: loss=0.3459, pooled_mean_dice=0.7705, per_class=['0.7705'], iou=0.6267, recall=0.9796, precision=0.6349, vol_sim=0.7865, mcc=0.7843, min_class_dice=0.7705, coverage=[76]/88 samples
[2026-06-19 19:03:18] INFO segtask_v1.trainer.trainer: Epoch 657/1000 | LR=7.28e-04 | loss=0.2562 | val_dice=0.7705 | best=0.8292 (ep441) | 05:33:49 | L_main=0.1265 L_aux_1=0.1152(w=0.5) L_aux_2=0.1444(w=0.5)
[2026-06-19 19:03:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 657): 12688.6 MiB
[2026-06-19 19:04:19] INFO segtask_v1.trainer.validation:   Val: loss=0.3310, pooled_mean_dice=0.7907, per_class=['0.7907'], iou=0.6539, recall=0.9882, precision=0.6590, vol_sim=0.8002, mcc=0.8017, min_class_dice=0.7907, coverage=[78]/88 samples
[2026-06-19 19:04:19] INFO segtask_v1.trainer.trainer: Epoch 658/1000 | LR=7.32e-04 | loss=0.3041 | val_dice=0.7907 | best=0.8292 (ep441) | 05:34:50 | L_main=0.1566 L_aux_1=0.1409(w=0.5) L_aux_2=0.1539(w=0.5)
[2026-06-19 19:04:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 658): 12688.6 MiB
[2026-06-19 19:05:22] INFO segtask_v1.trainer.validation:   Val: loss=0.3183, pooled_mean_dice=0.7963, per_class=['0.7963'], iou=0.6616, recall=0.9862, precision=0.6678, vol_sim=0.8075, mcc=0.8062, min_class_dice=0.7963, coverage=[77]/88 samples
[2026-06-19 19:05:22] INFO segtask_v1.trainer.trainer: Epoch 659/1000 | LR=7.35e-04 | loss=0.2808 | val_dice=0.7963 | best=0.8292 (ep441) | 05:35:53 | L_main=0.1406 L_aux_1=0.1345(w=0.5) L_aux_2=0.1461(w=0.5)
[2026-06-19 19:05:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 659): 12688.6 MiB
[2026-06-19 19:06:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3187, pooled_mean_dice=0.7689, per_class=['0.7689'], iou=0.6245, recall=0.9841, precision=0.6309, vol_sim=0.7813, mcc=0.7829, min_class_dice=0.7689, coverage=[77]/88 samples
[2026-06-19 19:06:25] INFO segtask_v1.trainer.trainer: Epoch 660/1000 | LR=7.39e-04 | loss=0.3183 | val_dice=0.7689 | best=0.8292 (ep441) | 05:36:56 | L_main=0.1604 L_aux_1=0.1463(w=0.5) L_aux_2=0.1695(w=0.5)
[2026-06-19 19:06:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 660): 12688.6 MiB
[2026-06-19 19:07:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3418, pooled_mean_dice=0.7935, per_class=['0.7935'], iou=0.6577, recall=0.9870, precision=0.6635, vol_sim=0.8040, mcc=0.8039, min_class_dice=0.7935, coverage=[83]/88 samples
[2026-06-19 19:07:29] INFO segtask_v1.trainer.trainer: Epoch 661/1000 | LR=7.42e-04 | loss=0.2580 | val_dice=0.7935 | best=0.8292 (ep441) | 05:38:00 | L_main=0.1331 L_aux_1=0.1194(w=0.5) L_aux_2=0.1304(w=0.5)
[2026-06-19 19:07:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 661): 12688.6 MiB
[2026-06-19 19:08:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3223, pooled_mean_dice=0.8066, per_class=['0.8066'], iou=0.6759, recall=0.9818, precision=0.6844, vol_sim=0.8215, mcc=0.8146, min_class_dice=0.8066, coverage=[77]/88 samples
[2026-06-19 19:08:31] INFO segtask_v1.trainer.trainer: Epoch 662/1000 | LR=7.46e-04 | loss=0.2704 | val_dice=0.8066 | best=0.8292 (ep441) | 05:39:02 | L_main=0.1371 L_aux_1=0.1242(w=0.5) L_aux_2=0.1423(w=0.5)
[2026-06-19 19:08:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 662): 12688.6 MiB
[2026-06-19 19:09:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3180, pooled_mean_dice=0.7924, per_class=['0.7924'], iou=0.6562, recall=0.9828, precision=0.6638, vol_sim=0.8062, mcc=0.8017, min_class_dice=0.7924, coverage=[73]/88 samples
[2026-06-19 19:09:32] INFO segtask_v1.trainer.trainer: Epoch 663/1000 | LR=7.49e-04 | loss=0.2539 | val_dice=0.7924 | best=0.8292 (ep441) | 05:40:03 | L_main=0.1308 L_aux_1=0.1149(w=0.5) L_aux_2=0.1312(w=0.5)
[2026-06-19 19:09:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 663): 12688.6 MiB
[2026-06-19 19:10:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2540, pooled_mean_dice=0.8128, per_class=['0.8128'], iou=0.6847, recall=0.9854, precision=0.6917, vol_sim=0.8249, mcc=0.8206, min_class_dice=0.8128, coverage=[71]/88 samples
[2026-06-19 19:10:35] INFO segtask_v1.trainer.trainer: Epoch 664/1000 | LR=7.53e-04 | loss=0.2637 | val_dice=0.8128 | best=0.8292 (ep441) | 05:41:06 | L_main=0.1271 L_aux_1=0.1281(w=0.5) L_aux_2=0.1452(w=0.5)
[2026-06-19 19:10:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 664): 12688.6 MiB
[2026-06-19 19:11:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3496, pooled_mean_dice=0.7958, per_class=['0.7958'], iou=0.6608, recall=0.9839, precision=0.6680, vol_sim=0.8088, mcc=0.8057, min_class_dice=0.7958, coverage=[81]/88 samples
[2026-06-19 19:11:37] INFO segtask_v1.trainer.trainer: Epoch 665/1000 | LR=7.56e-04 | loss=0.2992 | val_dice=0.7958 | best=0.8292 (ep441) | 05:42:08 | L_main=0.1508 L_aux_1=0.1427(w=0.5) L_aux_2=0.1542(w=0.5)
[2026-06-19 19:11:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 665): 12688.6 MiB
[2026-06-19 19:12:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3380, pooled_mean_dice=0.7860, per_class=['0.7860'], iou=0.6475, recall=0.9832, precision=0.6547, vol_sim=0.7995, mcc=0.7972, min_class_dice=0.7860, coverage=[81]/88 samples
[2026-06-19 19:12:41] INFO segtask_v1.trainer.trainer: Epoch 666/1000 | LR=7.59e-04 | loss=0.2832 | val_dice=0.7860 | best=0.8292 (ep441) | 05:43:12 | L_main=0.1438 L_aux_1=0.1304(w=0.5) L_aux_2=0.1486(w=0.5)
[2026-06-19 19:12:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 666): 12688.6 MiB
[2026-06-19 19:13:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3207, pooled_mean_dice=0.8116, per_class=['0.8116'], iou=0.6829, recall=0.9834, precision=0.6909, vol_sim=0.8253, mcc=0.8192, min_class_dice=0.8116, coverage=[82]/88 samples
[2026-06-19 19:13:44] INFO segtask_v1.trainer.trainer: Epoch 667/1000 | LR=7.63e-04 | loss=0.2469 | val_dice=0.8116 | best=0.8292 (ep441) | 05:44:15 | L_main=0.1257 L_aux_1=0.1137(w=0.5) L_aux_2=0.1286(w=0.5)
[2026-06-19 19:13:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 667): 12688.6 MiB
[2026-06-19 19:14:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3190, pooled_mean_dice=0.7876, per_class=['0.7876'], iou=0.6496, recall=0.9816, precision=0.6576, vol_sim=0.8023, mcc=0.7991, min_class_dice=0.7876, coverage=[72]/88 samples
[2026-06-19 19:14:46] INFO segtask_v1.trainer.trainer: Epoch 668/1000 | LR=7.66e-04 | loss=0.2409 | val_dice=0.7876 | best=0.8292 (ep441) | 05:45:17 | L_main=0.1256 L_aux_1=0.1079(w=0.5) L_aux_2=0.1227(w=0.5)
[2026-06-19 19:14:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 668): 12688.6 MiB
[2026-06-19 19:15:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3420, pooled_mean_dice=0.7851, per_class=['0.7851'], iou=0.6462, recall=0.9844, precision=0.6529, vol_sim=0.7976, mcc=0.7971, min_class_dice=0.7851, coverage=[73]/88 samples
[2026-06-19 19:15:49] INFO segtask_v1.trainer.trainer: Epoch 669/1000 | LR=7.69e-04 | loss=0.2499 | val_dice=0.7851 | best=0.8292 (ep441) | 05:46:20 | L_main=0.1252 L_aux_1=0.1151(w=0.5) L_aux_2=0.1343(w=0.5)
[2026-06-19 19:15:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 669): 12688.6 MiB
[2026-06-19 19:16:51] INFO segtask_v1.trainer.validation:   Val: loss=0.3263, pooled_mean_dice=0.7980, per_class=['0.7980'], iou=0.6639, recall=0.9853, precision=0.6705, vol_sim=0.8099, mcc=0.8077, min_class_dice=0.7980, coverage=[77]/88 samples
[2026-06-19 19:16:51] INFO segtask_v1.trainer.trainer: Epoch 670/1000 | LR=7.73e-04 | loss=0.2280 | val_dice=0.7980 | best=0.8292 (ep441) | 05:47:22 | L_main=0.1133 L_aux_1=0.1043(w=0.5) L_aux_2=0.1252(w=0.5)
[2026-06-19 19:16:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 670): 12688.6 MiB
[2026-06-19 19:17:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3596, pooled_mean_dice=0.7814, per_class=['0.7814'], iou=0.6412, recall=0.9853, precision=0.6474, vol_sim=0.7930, mcc=0.7940, min_class_dice=0.7814, coverage=[72]/88 samples
[2026-06-19 19:17:55] INFO segtask_v1.trainer.trainer: Epoch 671/1000 | LR=7.76e-04 | loss=0.2481 | val_dice=0.7814 | best=0.8292 (ep441) | 05:48:27 | L_main=0.1230 L_aux_1=0.1131(w=0.5) L_aux_2=0.1373(w=0.5)
[2026-06-19 19:17:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 671): 12688.6 MiB
[2026-06-19 19:18:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3147, pooled_mean_dice=0.8034, per_class=['0.8034'], iou=0.6715, recall=0.9867, precision=0.6776, vol_sim=0.8143, mcc=0.8132, min_class_dice=0.8034, coverage=[74]/88 samples
[2026-06-19 19:18:58] INFO segtask_v1.trainer.trainer: Epoch 672/1000 | LR=7.79e-04 | loss=0.2545 | val_dice=0.8034 | best=0.8292 (ep441) | 05:49:29 | L_main=0.1302 L_aux_1=0.1177(w=0.5) L_aux_2=0.1309(w=0.5)
[2026-06-19 19:18:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 672): 12688.6 MiB
[2026-06-19 19:20:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2988, pooled_mean_dice=0.8104, per_class=['0.8104'], iou=0.6813, recall=0.9807, precision=0.6905, vol_sim=0.8264, mcc=0.8179, min_class_dice=0.8104, coverage=[76]/88 samples
[2026-06-19 19:20:00] INFO segtask_v1.trainer.trainer: Epoch 673/1000 | LR=7.83e-04 | loss=0.2470 | val_dice=0.8104 | best=0.8292 (ep441) | 05:50:31 | L_main=0.1260 L_aux_1=0.1131(w=0.5) L_aux_2=0.1288(w=0.5)
[2026-06-19 19:20:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 673): 12688.6 MiB
[2026-06-19 19:21:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3161, pooled_mean_dice=0.7916, per_class=['0.7916'], iou=0.6551, recall=0.9849, precision=0.6618, vol_sim=0.8038, mcc=0.8022, min_class_dice=0.7916, coverage=[76]/88 samples
[2026-06-19 19:21:02] INFO segtask_v1.trainer.trainer: Epoch 674/1000 | LR=7.86e-04 | loss=0.3694 | val_dice=0.7916 | best=0.8292 (ep441) | 05:51:33 | L_main=0.1800 L_aux_1=0.1793(w=0.5) L_aux_2=0.1995(w=0.5)
[2026-06-19 19:21:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 674): 12688.6 MiB
[2026-06-19 19:22:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2865, pooled_mean_dice=0.7974, per_class=['0.7974'], iou=0.6630, recall=0.9826, precision=0.6709, vol_sim=0.8115, mcc=0.8067, min_class_dice=0.7974, coverage=[70]/88 samples
[2026-06-19 19:22:04] INFO segtask_v1.trainer.trainer: Epoch 675/1000 | LR=7.89e-04 | loss=0.3349 | val_dice=0.7974 | best=0.8292 (ep441) | 05:52:36 | L_main=0.1677 L_aux_1=0.1585(w=0.5) L_aux_2=0.1759(w=0.5)
[2026-06-19 19:22:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 675): 12688.6 MiB
[2026-06-19 19:23:08] INFO segtask_v1.trainer.validation:   Val: loss=0.3191, pooled_mean_dice=0.7961, per_class=['0.7961'], iou=0.6613, recall=0.9805, precision=0.6701, vol_sim=0.8119, mcc=0.8061, min_class_dice=0.7961, coverage=[68]/88 samples
[2026-06-19 19:23:08] INFO segtask_v1.trainer.trainer: Epoch 676/1000 | LR=7.92e-04 | loss=0.2879 | val_dice=0.7961 | best=0.8292 (ep441) | 05:53:39 | L_main=0.1441 L_aux_1=0.1369(w=0.5) L_aux_2=0.1508(w=0.5)
[2026-06-19 19:23:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 676): 12688.6 MiB
[2026-06-19 19:24:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3153, pooled_mean_dice=0.7854, per_class=['0.7854'], iou=0.6466, recall=0.9859, precision=0.6526, vol_sim=0.7966, mcc=0.7967, min_class_dice=0.7854, coverage=[77]/88 samples
[2026-06-19 19:24:11] INFO segtask_v1.trainer.trainer: Epoch 677/1000 | LR=7.96e-04 | loss=0.2438 | val_dice=0.7854 | best=0.8292 (ep441) | 05:54:42 | L_main=0.1267 L_aux_1=0.1097(w=0.5) L_aux_2=0.1245(w=0.5)
[2026-06-19 19:24:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 677): 12688.6 MiB
[2026-06-19 19:25:16] INFO segtask_v1.trainer.validation:   Val: loss=0.3340, pooled_mean_dice=0.7899, per_class=['0.7899'], iou=0.6528, recall=0.9862, precision=0.6588, vol_sim=0.8010, mcc=0.8009, min_class_dice=0.7899, coverage=[73]/88 samples
[2026-06-19 19:25:16] INFO segtask_v1.trainer.trainer: Epoch 678/1000 | LR=7.99e-04 | loss=0.2853 | val_dice=0.7899 | best=0.8292 (ep441) | 05:55:47 | L_main=0.1481 L_aux_1=0.1302(w=0.5) L_aux_2=0.1442(w=0.5)
[2026-06-19 19:25:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 678): 12688.6 MiB
[2026-06-19 19:26:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2748, pooled_mean_dice=0.7945, per_class=['0.7945'], iou=0.6591, recall=0.9861, precision=0.6653, vol_sim=0.8057, mcc=0.8054, min_class_dice=0.7945, coverage=[68]/88 samples
[2026-06-19 19:26:17] INFO segtask_v1.trainer.trainer: Epoch 679/1000 | LR=8.02e-04 | loss=0.2697 | val_dice=0.7945 | best=0.8292 (ep441) | 05:56:49 | L_main=0.1415 L_aux_1=0.1214(w=0.5) L_aux_2=0.1351(w=0.5)
[2026-06-19 19:26:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 679): 12688.6 MiB
[2026-06-19 19:27:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2740, pooled_mean_dice=0.8047, per_class=['0.8047'], iou=0.6733, recall=0.9850, precision=0.6803, vol_sim=0.8170, mcc=0.8132, min_class_dice=0.8047, coverage=[75]/88 samples
[2026-06-19 19:27:20] INFO segtask_v1.trainer.trainer: Epoch 680/1000 | LR=8.05e-04 | loss=0.2310 | val_dice=0.8047 | best=0.8292 (ep441) | 05:57:51 | L_main=0.1199 L_aux_1=0.1039(w=0.5) L_aux_2=0.1183(w=0.5)
[2026-06-19 19:27:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 680): 12688.6 MiB
[2026-06-19 19:28:24] INFO segtask_v1.trainer.validation:   Val: loss=0.3158, pooled_mean_dice=0.7908, per_class=['0.7908'], iou=0.6540, recall=0.9840, precision=0.6610, vol_sim=0.8036, mcc=0.8004, min_class_dice=0.7908, coverage=[81]/88 samples
[2026-06-19 19:28:24] INFO segtask_v1.trainer.trainer: Epoch 681/1000 | LR=8.08e-04 | loss=0.2810 | val_dice=0.7908 | best=0.8292 (ep441) | 05:58:55 | L_main=0.1474 L_aux_1=0.1299(w=0.5) L_aux_2=0.1372(w=0.5)
[2026-06-19 19:28:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 681): 12688.6 MiB
[2026-06-19 19:29:27] INFO segtask_v1.trainer.validation:   Val: loss=0.3234, pooled_mean_dice=0.7980, per_class=['0.7980'], iou=0.6639, recall=0.9849, precision=0.6707, vol_sim=0.8103, mcc=0.8071, min_class_dice=0.7980, coverage=[81]/88 samples
[2026-06-19 19:29:27] INFO segtask_v1.trainer.trainer: Epoch 682/1000 | LR=8.11e-04 | loss=0.2481 | val_dice=0.7980 | best=0.8292 (ep441) | 05:59:59 | L_main=0.1286 L_aux_1=0.1151(w=0.5) L_aux_2=0.1240(w=0.5)
[2026-06-19 19:29:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 682): 12688.6 MiB
[2026-06-19 19:30:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3664, pooled_mean_dice=0.7859, per_class=['0.7859'], iou=0.6473, recall=0.9807, precision=0.6557, vol_sim=0.8014, mcc=0.7968, min_class_dice=0.7859, coverage=[78]/88 samples
[2026-06-19 19:30:31] INFO segtask_v1.trainer.trainer: Epoch 683/1000 | LR=8.15e-04 | loss=0.2688 | val_dice=0.7859 | best=0.8292 (ep441) | 06:01:02 | L_main=0.1389 L_aux_1=0.1241(w=0.5) L_aux_2=0.1356(w=0.5)
[2026-06-19 19:30:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 683): 12688.6 MiB
[2026-06-19 19:31:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3429, pooled_mean_dice=0.7901, per_class=['0.7901'], iou=0.6531, recall=0.9848, precision=0.6597, vol_sim=0.8024, mcc=0.8008, min_class_dice=0.7901, coverage=[76]/88 samples
[2026-06-19 19:31:34] INFO segtask_v1.trainer.trainer: Epoch 684/1000 | LR=8.18e-04 | loss=0.2778 | val_dice=0.7901 | best=0.8292 (ep441) | 06:02:05 | L_main=0.1397 L_aux_1=0.1313(w=0.5) L_aux_2=0.1449(w=0.5)
[2026-06-19 19:31:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 684): 12688.6 MiB
[2026-06-19 19:32:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3291, pooled_mean_dice=0.7965, per_class=['0.7965'], iou=0.6618, recall=0.9848, precision=0.6686, vol_sim=0.8088, mcc=0.8067, min_class_dice=0.7965, coverage=[80]/88 samples
[2026-06-19 19:32:37] INFO segtask_v1.trainer.trainer: Epoch 685/1000 | LR=8.21e-04 | loss=0.2870 | val_dice=0.7965 | best=0.8292 (ep441) | 06:03:08 | L_main=0.1439 L_aux_1=0.1368(w=0.5) L_aux_2=0.1492(w=0.5)
[2026-06-19 19:32:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 685): 12688.6 MiB
[2026-06-19 19:33:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3337, pooled_mean_dice=0.8001, per_class=['0.8001'], iou=0.6668, recall=0.9822, precision=0.6749, vol_sim=0.8146, mcc=0.8101, min_class_dice=0.8001, coverage=[75]/88 samples
[2026-06-19 19:33:41] INFO segtask_v1.trainer.trainer: Epoch 686/1000 | LR=8.24e-04 | loss=0.2554 | val_dice=0.8001 | best=0.8292 (ep441) | 06:04:12 | L_main=0.1332 L_aux_1=0.1173(w=0.5) L_aux_2=0.1271(w=0.5)
[2026-06-19 19:33:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 686): 12688.6 MiB
[2026-06-19 19:34:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3334, pooled_mean_dice=0.7959, per_class=['0.7959'], iou=0.6610, recall=0.9874, precision=0.6666, vol_sim=0.8061, mcc=0.8065, min_class_dice=0.7959, coverage=[74]/88 samples
[2026-06-19 19:34:42] INFO segtask_v1.trainer.trainer: Epoch 687/1000 | LR=8.27e-04 | loss=0.2586 | val_dice=0.7959 | best=0.8292 (ep441) | 06:05:13 | L_main=0.1353 L_aux_1=0.1166(w=0.5) L_aux_2=0.1299(w=0.5)
[2026-06-19 19:34:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 687): 12688.6 MiB
[2026-06-19 19:35:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2940, pooled_mean_dice=0.7875, per_class=['0.7875'], iou=0.6495, recall=0.9816, precision=0.6575, vol_sim=0.8023, mcc=0.7981, min_class_dice=0.7875, coverage=[77]/88 samples
[2026-06-19 19:35:45] INFO segtask_v1.trainer.trainer: Epoch 688/1000 | LR=8.30e-04 | loss=0.2543 | val_dice=0.7875 | best=0.8292 (ep441) | 06:06:17 | L_main=0.1312 L_aux_1=0.1191(w=0.5) L_aux_2=0.1272(w=0.5)
[2026-06-19 19:35:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 688): 12688.6 MiB
[2026-06-19 19:36:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3033, pooled_mean_dice=0.7913, per_class=['0.7913'], iou=0.6547, recall=0.9866, precision=0.6606, vol_sim=0.8021, mcc=0.8017, min_class_dice=0.7913, coverage=[74]/88 samples
[2026-06-19 19:36:49] INFO segtask_v1.trainer.trainer: Epoch 689/1000 | LR=8.33e-04 | loss=0.2789 | val_dice=0.7913 | best=0.8292 (ep441) | 06:07:20 | L_main=0.1375 L_aux_1=0.1338(w=0.5) L_aux_2=0.1490(w=0.5)
[2026-06-19 19:36:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 689): 12688.6 MiB
[2026-06-19 19:37:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2966, pooled_mean_dice=0.8015, per_class=['0.8015'], iou=0.6687, recall=0.9850, precision=0.6756, vol_sim=0.8137, mcc=0.8089, min_class_dice=0.8015, coverage=[83]/88 samples
[2026-06-19 19:37:51] INFO segtask_v1.trainer.trainer: Epoch 690/1000 | LR=8.36e-04 | loss=0.2662 | val_dice=0.8015 | best=0.8292 (ep441) | 06:08:22 | L_main=0.1333 L_aux_1=0.1269(w=0.5) L_aux_2=0.1389(w=0.5)
[2026-06-19 19:37:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 690): 12688.6 MiB
[2026-06-19 19:38:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3158, pooled_mean_dice=0.7945, per_class=['0.7945'], iou=0.6591, recall=0.9837, precision=0.6664, vol_sim=0.8077, mcc=0.8032, min_class_dice=0.7945, coverage=[77]/88 samples
[2026-06-19 19:38:53] INFO segtask_v1.trainer.trainer: Epoch 691/1000 | LR=8.39e-04 | loss=0.3331 | val_dice=0.7945 | best=0.8292 (ep441) | 06:09:24 | L_main=0.1672 L_aux_1=0.1581(w=0.5) L_aux_2=0.1739(w=0.5)
[2026-06-19 19:38:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 691): 12688.6 MiB
[2026-06-19 19:39:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3250, pooled_mean_dice=0.7889, per_class=['0.7889'], iou=0.6513, recall=0.9831, precision=0.6587, vol_sim=0.8024, mcc=0.7995, min_class_dice=0.7889, coverage=[79]/88 samples
[2026-06-19 19:39:55] INFO segtask_v1.trainer.trainer: Epoch 692/1000 | LR=8.42e-04 | loss=0.2945 | val_dice=0.7889 | best=0.8292 (ep441) | 06:10:26 | L_main=0.1458 L_aux_1=0.1347(w=0.5) L_aux_2=0.1627(w=0.5)
[2026-06-19 19:39:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 692): 12688.6 MiB
[2026-06-19 19:40:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3517, pooled_mean_dice=0.8012, per_class=['0.8012'], iou=0.6683, recall=0.9839, precision=0.6757, vol_sim=0.8143, mcc=0.8094, min_class_dice=0.8012, coverage=[82]/88 samples
[2026-06-19 19:40:59] INFO segtask_v1.trainer.trainer: Epoch 693/1000 | LR=8.44e-04 | loss=0.2997 | val_dice=0.8012 | best=0.8292 (ep441) | 06:11:30 | L_main=0.1493 L_aux_1=0.1417(w=0.5) L_aux_2=0.1589(w=0.5)
[2026-06-19 19:40:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 693): 12688.6 MiB
[2026-06-19 19:42:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3691, pooled_mean_dice=0.7840, per_class=['0.7840'], iou=0.6447, recall=0.9855, precision=0.6509, vol_sim=0.7955, mcc=0.7963, min_class_dice=0.7840, coverage=[78]/88 samples
[2026-06-19 19:42:01] INFO segtask_v1.trainer.trainer: Epoch 694/1000 | LR=8.47e-04 | loss=0.3262 | val_dice=0.7840 | best=0.8292 (ep441) | 06:12:33 | L_main=0.1623 L_aux_1=0.1523(w=0.5) L_aux_2=0.1756(w=0.5)
[2026-06-19 19:42:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 694): 12688.6 MiB
[2026-06-19 19:43:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3366, pooled_mean_dice=0.7927, per_class=['0.7927'], iou=0.6566, recall=0.9857, precision=0.6629, vol_sim=0.8042, mcc=0.8032, min_class_dice=0.7927, coverage=[81]/88 samples
[2026-06-19 19:43:04] INFO segtask_v1.trainer.trainer: Epoch 695/1000 | LR=8.50e-04 | loss=0.2806 | val_dice=0.7927 | best=0.8292 (ep441) | 06:13:36 | L_main=0.1438 L_aux_1=0.1287(w=0.5) L_aux_2=0.1450(w=0.5)
[2026-06-19 19:43:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 695): 12688.6 MiB
[2026-06-19 19:44:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3256, pooled_mean_dice=0.7742, per_class=['0.7742'], iou=0.6315, recall=0.9845, precision=0.6379, vol_sim=0.7863, mcc=0.7871, min_class_dice=0.7742, coverage=[75]/88 samples
[2026-06-19 19:44:05] INFO segtask_v1.trainer.trainer: Epoch 696/1000 | LR=8.53e-04 | loss=0.2665 | val_dice=0.7742 | best=0.8292 (ep441) | 06:14:36 | L_main=0.1379 L_aux_1=0.1192(w=0.5) L_aux_2=0.1379(w=0.5)
[2026-06-19 19:44:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 696): 12688.6 MiB
[2026-06-19 19:45:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3124, pooled_mean_dice=0.7814, per_class=['0.7814'], iou=0.6412, recall=0.9839, precision=0.6480, vol_sim=0.7941, mcc=0.7934, min_class_dice=0.7814, coverage=[77]/88 samples
[2026-06-19 19:45:09] INFO segtask_v1.trainer.trainer: Epoch 697/1000 | LR=8.56e-04 | loss=0.2399 | val_dice=0.7814 | best=0.8292 (ep441) | 06:15:40 | L_main=0.1210 L_aux_1=0.1109(w=0.5) L_aux_2=0.1271(w=0.5)
[2026-06-19 19:45:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 697): 12688.6 MiB
[2026-06-19 19:46:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3084, pooled_mean_dice=0.7925, per_class=['0.7925'], iou=0.6563, recall=0.9787, precision=0.6658, vol_sim=0.8097, mcc=0.8021, min_class_dice=0.7925, coverage=[75]/88 samples
[2026-06-19 19:46:11] INFO segtask_v1.trainer.trainer: Epoch 698/1000 | LR=8.59e-04 | loss=0.2512 | val_dice=0.7925 | best=0.8292 (ep441) | 06:16:42 | L_main=0.1297 L_aux_1=0.1171(w=0.5) L_aux_2=0.1260(w=0.5)
[2026-06-19 19:46:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 698): 12688.6 MiB
[2026-06-19 19:47:14] INFO segtask_v1.trainer.validation:   Val: loss=0.3288, pooled_mean_dice=0.7817, per_class=['0.7817'], iou=0.6416, recall=0.9834, precision=0.6487, vol_sim=0.7949, mcc=0.7937, min_class_dice=0.7817, coverage=[72]/88 samples
[2026-06-19 19:47:14] INFO segtask_v1.trainer.trainer: Epoch 699/1000 | LR=8.61e-04 | loss=0.2575 | val_dice=0.7817 | best=0.8292 (ep441) | 06:17:45 | L_main=0.1307 L_aux_1=0.1226(w=0.5) L_aux_2=0.1308(w=0.5)
[2026-06-19 19:47:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 699): 12688.6 MiB
[2026-06-19 19:48:17] INFO segtask_v1.trainer.validation:   Val: loss=0.3041, pooled_mean_dice=0.7903, per_class=['0.7903'], iou=0.6533, recall=0.9827, precision=0.6609, vol_sim=0.8042, mcc=0.8005, min_class_dice=0.7903, coverage=[74]/88 samples
[2026-06-19 19:48:17] INFO segtask_v1.trainer.trainer: Epoch 700/1000 | LR=8.64e-04 | loss=0.2612 | val_dice=0.7903 | best=0.8292 (ep441) | 06:18:48 | L_main=0.1318 L_aux_1=0.1227(w=0.5) L_aux_2=0.1362(w=0.5)
[2026-06-19 19:48:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 700): 12688.6 MiB
[2026-06-19 19:49:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3215, pooled_mean_dice=0.7885, per_class=['0.7885'], iou=0.6508, recall=0.9852, precision=0.6572, vol_sim=0.8003, mcc=0.7999, min_class_dice=0.7885, coverage=[72]/88 samples
[2026-06-19 19:49:20] INFO segtask_v1.trainer.trainer: Epoch 701/1000 | LR=8.67e-04 | loss=0.2623 | val_dice=0.7885 | best=0.8292 (ep441) | 06:19:51 | L_main=0.1358 L_aux_1=0.1231(w=0.5) L_aux_2=0.1298(w=0.5)
[2026-06-19 19:49:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 701): 12688.6 MiB
[2026-06-19 19:50:22] INFO segtask_v1.trainer.validation:   Val: loss=0.3117, pooled_mean_dice=0.8024, per_class=['0.8024'], iou=0.6700, recall=0.9844, precision=0.6772, vol_sim=0.8152, mcc=0.8116, min_class_dice=0.8024, coverage=[75]/88 samples
[2026-06-19 19:50:23] INFO segtask_v1.trainer.trainer: Epoch 702/1000 | LR=8.69e-04 | loss=0.3025 | val_dice=0.8024 | best=0.8292 (ep441) | 06:20:54 | L_main=0.1550 L_aux_1=0.1414(w=0.5) L_aux_2=0.1536(w=0.5)
[2026-06-19 19:50:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 702): 12688.6 MiB
[2026-06-19 19:51:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3184, pooled_mean_dice=0.7842, per_class=['0.7842'], iou=0.6450, recall=0.9823, precision=0.6525, vol_sim=0.7983, mcc=0.7962, min_class_dice=0.7842, coverage=[71]/88 samples
[2026-06-19 19:51:25] INFO segtask_v1.trainer.trainer: Epoch 703/1000 | LR=8.72e-04 | loss=0.2573 | val_dice=0.7842 | best=0.8292 (ep441) | 06:21:56 | L_main=0.1347 L_aux_1=0.1180(w=0.5) L_aux_2=0.1273(w=0.5)
[2026-06-19 19:51:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 703): 12688.6 MiB
[2026-06-19 19:52:27] INFO segtask_v1.trainer.validation:   Val: loss=0.3312, pooled_mean_dice=0.7640, per_class=['0.7640'], iou=0.6182, recall=0.9838, precision=0.6245, vol_sim=0.7766, mcc=0.7791, min_class_dice=0.7640, coverage=[71]/88 samples
[2026-06-19 19:52:27] INFO segtask_v1.trainer.trainer: Epoch 704/1000 | LR=8.75e-04 | loss=0.2580 | val_dice=0.7640 | best=0.8292 (ep441) | 06:22:58 | L_main=0.1342 L_aux_1=0.1194(w=0.5) L_aux_2=0.1282(w=0.5)
[2026-06-19 19:52:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 704): 12688.6 MiB
[2026-06-19 19:53:29] INFO segtask_v1.trainer.validation:   Val: loss=0.3072, pooled_mean_dice=0.7963, per_class=['0.7963'], iou=0.6615, recall=0.9862, precision=0.6677, vol_sim=0.8074, mcc=0.8058, min_class_dice=0.7963, coverage=[72]/88 samples
[2026-06-19 19:53:29] INFO segtask_v1.trainer.trainer: Epoch 705/1000 | LR=8.77e-04 | loss=0.2321 | val_dice=0.7963 | best=0.8292 (ep441) | 06:24:00 | L_main=0.1218 L_aux_1=0.1048(w=0.5) L_aux_2=0.1158(w=0.5)
[2026-06-19 19:53:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 705): 12688.6 MiB
[2026-06-19 19:54:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3155, pooled_mean_dice=0.8021, per_class=['0.8021'], iou=0.6696, recall=0.9807, precision=0.6785, vol_sim=0.8179, mcc=0.8089, min_class_dice=0.8021, coverage=[78]/88 samples
[2026-06-19 19:54:31] INFO segtask_v1.trainer.trainer: Epoch 706/1000 | LR=8.80e-04 | loss=0.2564 | val_dice=0.8021 | best=0.8292 (ep441) | 06:25:02 | L_main=0.1315 L_aux_1=0.1213(w=0.5) L_aux_2=0.1286(w=0.5)
[2026-06-19 19:54:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 706): 12688.6 MiB
[2026-06-19 19:55:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3463, pooled_mean_dice=0.7943, per_class=['0.7943'], iou=0.6588, recall=0.9814, precision=0.6671, vol_sim=0.8093, mcc=0.8044, min_class_dice=0.7943, coverage=[78]/88 samples
[2026-06-19 19:55:34] INFO segtask_v1.trainer.trainer: Epoch 707/1000 | LR=8.83e-04 | loss=0.2529 | val_dice=0.7943 | best=0.8292 (ep441) | 06:26:05 | L_main=0.1308 L_aux_1=0.1186(w=0.5) L_aux_2=0.1256(w=0.5)
[2026-06-19 19:55:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 707): 12688.6 MiB
[2026-06-19 19:56:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3149, pooled_mean_dice=0.7830, per_class=['0.7830'], iou=0.6433, recall=0.9819, precision=0.6510, vol_sim=0.7974, mcc=0.7948, min_class_dice=0.7830, coverage=[74]/88 samples
[2026-06-19 19:56:37] INFO segtask_v1.trainer.trainer: Epoch 708/1000 | LR=8.85e-04 | loss=0.2666 | val_dice=0.7830 | best=0.8292 (ep441) | 06:27:08 | L_main=0.1320 L_aux_1=0.1315(w=0.5) L_aux_2=0.1378(w=0.5)
[2026-06-19 19:56:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 708): 12688.6 MiB
[2026-06-19 19:57:39] INFO segtask_v1.trainer.validation:   Val: loss=0.2945, pooled_mean_dice=0.7919, per_class=['0.7919'], iou=0.6555, recall=0.9858, precision=0.6617, vol_sim=0.8033, mcc=0.8020, min_class_dice=0.7919, coverage=[70]/88 samples
[2026-06-19 19:57:39] INFO segtask_v1.trainer.trainer: Epoch 709/1000 | LR=8.88e-04 | loss=0.2795 | val_dice=0.7919 | best=0.8292 (ep441) | 06:28:10 | L_main=0.1403 L_aux_1=0.1295(w=0.5) L_aux_2=0.1488(w=0.5)
[2026-06-19 19:57:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 709): 12688.6 MiB
[2026-06-19 19:58:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3378, pooled_mean_dice=0.7741, per_class=['0.7741'], iou=0.6315, recall=0.9832, precision=0.6384, vol_sim=0.7874, mcc=0.7871, min_class_dice=0.7741, coverage=[78]/88 samples
[2026-06-19 19:58:42] INFO segtask_v1.trainer.trainer: Epoch 710/1000 | LR=8.90e-04 | loss=0.3434 | val_dice=0.7741 | best=0.8292 (ep441) | 06:29:13 | L_main=0.1693 L_aux_1=0.1647(w=0.5) L_aux_2=0.1835(w=0.5)
[2026-06-19 19:58:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 710): 12688.6 MiB
[2026-06-19 19:59:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2899, pooled_mean_dice=0.7941, per_class=['0.7941'], iou=0.6585, recall=0.9815, precision=0.6667, vol_sim=0.8090, mcc=0.8036, min_class_dice=0.7941, coverage=[75]/88 samples
[2026-06-19 19:59:46] INFO segtask_v1.trainer.trainer: Epoch 711/1000 | LR=8.93e-04 | loss=0.2713 | val_dice=0.7941 | best=0.8292 (ep441) | 06:30:17 | L_main=0.1400 L_aux_1=0.1281(w=0.5) L_aux_2=0.1343(w=0.5)
[2026-06-19 19:59:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 711): 12688.6 MiB
[2026-06-19 20:00:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3259, pooled_mean_dice=0.7807, per_class=['0.7807'], iou=0.6403, recall=0.9850, precision=0.6466, vol_sim=0.7926, mcc=0.7942, min_class_dice=0.7807, coverage=[67]/88 samples
[2026-06-19 20:00:48] INFO segtask_v1.trainer.trainer: Epoch 712/1000 | LR=8.95e-04 | loss=0.2655 | val_dice=0.7807 | best=0.8292 (ep441) | 06:31:20 | L_main=0.1349 L_aux_1=0.1214(w=0.5) L_aux_2=0.1397(w=0.5)
[2026-06-19 20:00:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 712): 12688.6 MiB
[2026-06-19 20:01:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2989, pooled_mean_dice=0.8107, per_class=['0.8107'], iou=0.6817, recall=0.9862, precision=0.6882, vol_sim=0.8220, mcc=0.8178, min_class_dice=0.8107, coverage=[78]/88 samples
[2026-06-19 20:01:51] INFO segtask_v1.trainer.trainer: Epoch 713/1000 | LR=8.97e-04 | loss=0.2689 | val_dice=0.8107 | best=0.8292 (ep441) | 06:32:23 | L_main=0.1371 L_aux_1=0.1274(w=0.5) L_aux_2=0.1362(w=0.5)
[2026-06-19 20:01:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 713): 12688.6 MiB
[2026-06-19 20:02:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3297, pooled_mean_dice=0.7926, per_class=['0.7926'], iou=0.6564, recall=0.9830, precision=0.6639, vol_sim=0.8062, mcc=0.8028, min_class_dice=0.7926, coverage=[75]/88 samples
[2026-06-19 20:02:54] INFO segtask_v1.trainer.trainer: Epoch 714/1000 | LR=9.00e-04 | loss=0.2722 | val_dice=0.7926 | best=0.8292 (ep441) | 06:33:25 | L_main=0.1369 L_aux_1=0.1277(w=0.5) L_aux_2=0.1429(w=0.5)
[2026-06-19 20:02:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 714): 12688.6 MiB
[2026-06-19 20:03:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3496, pooled_mean_dice=0.8121, per_class=['0.8121'], iou=0.6836, recall=0.9841, precision=0.6913, vol_sim=0.8252, mcc=0.8192, min_class_dice=0.8121, coverage=[80]/88 samples
[2026-06-19 20:03:57] INFO segtask_v1.trainer.trainer: Epoch 715/1000 | LR=9.02e-04 | loss=0.2890 | val_dice=0.8121 | best=0.8292 (ep441) | 06:34:28 | L_main=0.1470 L_aux_1=0.1345(w=0.5) L_aux_2=0.1494(w=0.5)
[2026-06-19 20:03:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 715): 12688.6 MiB
[2026-06-19 20:05:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3130, pooled_mean_dice=0.8096, per_class=['0.8096'], iou=0.6801, recall=0.9826, precision=0.6884, vol_sim=0.8240, mcc=0.8171, min_class_dice=0.8096, coverage=[75]/88 samples
[2026-06-19 20:05:00] INFO segtask_v1.trainer.trainer: Epoch 716/1000 | LR=9.05e-04 | loss=0.3075 | val_dice=0.8096 | best=0.8292 (ep441) | 06:35:31 | L_main=0.1556 L_aux_1=0.1397(w=0.5) L_aux_2=0.1640(w=0.5)
[2026-06-19 20:05:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 716): 12688.6 MiB
[2026-06-19 20:06:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3130, pooled_mean_dice=0.7952, per_class=['0.7952'], iou=0.6600, recall=0.9811, precision=0.6686, vol_sim=0.8106, mcc=0.8054, min_class_dice=0.7952, coverage=[75]/88 samples
[2026-06-19 20:06:01] INFO segtask_v1.trainer.trainer: Epoch 717/1000 | LR=9.07e-04 | loss=0.3538 | val_dice=0.7952 | best=0.8292 (ep441) | 06:36:32 | L_main=0.1778 L_aux_1=0.1688(w=0.5) L_aux_2=0.1832(w=0.5)
[2026-06-19 20:06:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 717): 12688.6 MiB
[2026-06-19 20:07:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3122, pooled_mean_dice=0.8074, per_class=['0.8074'], iou=0.6769, recall=0.9832, precision=0.6849, vol_sim=0.8212, mcc=0.8153, min_class_dice=0.8074, coverage=[78]/88 samples
[2026-06-19 20:07:04] INFO segtask_v1.trainer.trainer: Epoch 718/1000 | LR=9.09e-04 | loss=0.2826 | val_dice=0.8074 | best=0.8292 (ep441) | 06:37:35 | L_main=0.1438 L_aux_1=0.1307(w=0.5) L_aux_2=0.1468(w=0.5)
[2026-06-19 20:07:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 718): 12688.6 MiB
[2026-06-19 20:08:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3029, pooled_mean_dice=0.8106, per_class=['0.8106'], iou=0.6816, recall=0.9780, precision=0.6922, vol_sim=0.8288, mcc=0.8186, min_class_dice=0.8106, coverage=[76]/88 samples
[2026-06-19 20:08:05] INFO segtask_v1.trainer.trainer: Epoch 719/1000 | LR=9.11e-04 | loss=0.2864 | val_dice=0.8106 | best=0.8292 (ep441) | 06:38:37 | L_main=0.1488 L_aux_1=0.1325(w=0.5) L_aux_2=0.1428(w=0.5)
[2026-06-19 20:08:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 719): 12688.6 MiB
[2026-06-19 20:09:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2662, pooled_mean_dice=0.8117, per_class=['0.8117'], iou=0.6830, recall=0.9754, precision=0.6950, vol_sim=0.8321, mcc=0.8186, min_class_dice=0.8117, coverage=[74]/88 samples
[2026-06-19 20:09:07] INFO segtask_v1.trainer.trainer: Epoch 720/1000 | LR=9.14e-04 | loss=0.3031 | val_dice=0.8117 | best=0.8292 (ep441) | 06:39:38 | L_main=0.1514 L_aux_1=0.1424(w=0.5) L_aux_2=0.1610(w=0.5)
[2026-06-19 20:09:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 720): 12688.6 MiB
[2026-06-19 20:10:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2930, pooled_mean_dice=0.8088, per_class=['0.8088'], iou=0.6790, recall=0.9746, precision=0.6912, vol_sim=0.8299, mcc=0.8171, min_class_dice=0.8088, coverage=[70]/88 samples
[2026-06-19 20:10:09] INFO segtask_v1.trainer.trainer: Epoch 721/1000 | LR=9.16e-04 | loss=0.2810 | val_dice=0.8088 | best=0.8292 (ep441) | 06:40:41 | L_main=0.1472 L_aux_1=0.1282(w=0.5) L_aux_2=0.1395(w=0.5)
[2026-06-19 20:10:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 721): 12688.6 MiB
[2026-06-19 20:11:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2765, pooled_mean_dice=0.8098, per_class=['0.8098'], iou=0.6805, recall=0.9737, precision=0.6932, vol_sim=0.8317, mcc=0.8167, min_class_dice=0.8098, coverage=[72]/88 samples
[2026-06-19 20:11:10] INFO segtask_v1.trainer.trainer: Epoch 722/1000 | LR=9.18e-04 | loss=0.3098 | val_dice=0.8098 | best=0.8292 (ep441) | 06:41:41 | L_main=0.1560 L_aux_1=0.1431(w=0.5) L_aux_2=0.1646(w=0.5)
[2026-06-19 20:11:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 722): 12688.6 MiB
[2026-06-19 20:12:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3337, pooled_mean_dice=0.7980, per_class=['0.7980'], iou=0.6639, recall=0.9692, precision=0.6782, vol_sim=0.8234, mcc=0.8068, min_class_dice=0.7980, coverage=[80]/88 samples
[2026-06-19 20:12:13] INFO segtask_v1.trainer.trainer: Epoch 723/1000 | LR=9.20e-04 | loss=0.3346 | val_dice=0.7980 | best=0.8292 (ep441) | 06:42:44 | L_main=0.1701 L_aux_1=0.1580(w=0.5) L_aux_2=0.1709(w=0.5)
[2026-06-19 20:12:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 723): 12688.6 MiB
[2026-06-19 20:13:15] INFO segtask_v1.trainer.validation:   Val: loss=0.3129, pooled_mean_dice=0.8121, per_class=['0.8121'], iou=0.6836, recall=0.9729, precision=0.6969, vol_sim=0.8347, mcc=0.8183, min_class_dice=0.8121, coverage=[79]/88 samples
[2026-06-19 20:13:16] INFO segtask_v1.trainer.trainer: Epoch 724/1000 | LR=9.22e-04 | loss=0.2762 | val_dice=0.8121 | best=0.8292 (ep441) | 06:43:47 | L_main=0.1361 L_aux_1=0.1331(w=0.5) L_aux_2=0.1472(w=0.5)
[2026-06-19 20:13:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 724): 12688.6 MiB
[2026-06-19 20:14:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2919, pooled_mean_dice=0.8199, per_class=['0.8199'], iou=0.6948, recall=0.9791, precision=0.7053, vol_sim=0.8375, mcc=0.8261, min_class_dice=0.8199, coverage=[75]/88 samples
[2026-06-19 20:14:19] INFO segtask_v1.trainer.trainer: Epoch 725/1000 | LR=9.25e-04 | loss=0.3455 | val_dice=0.8199 | best=0.8292 (ep441) | 06:44:50 | L_main=0.1724 L_aux_1=0.1695(w=0.5) L_aux_2=0.1767(w=0.5)
[2026-06-19 20:14:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 725): 12688.6 MiB
[2026-06-19 20:15:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2660, pooled_mean_dice=0.8260, per_class=['0.8260'], iou=0.7035, recall=0.9793, precision=0.7142, vol_sim=0.8434, mcc=0.8316, min_class_dice=0.8260, coverage=[69]/88 samples
[2026-06-19 20:15:20] INFO segtask_v1.trainer.trainer: Epoch 726/1000 | LR=9.27e-04 | loss=0.2963 | val_dice=0.8260 | best=0.8292 (ep441) | 06:45:52 | L_main=0.1509 L_aux_1=0.1370(w=0.5) L_aux_2=0.1538(w=0.5)
[2026-06-19 20:15:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 726): 12688.6 MiB
[2026-06-19 20:16:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2859, pooled_mean_dice=0.8036, per_class=['0.8036'], iou=0.6717, recall=0.9724, precision=0.6848, vol_sim=0.8265, mcc=0.8121, min_class_dice=0.8036, coverage=[73]/88 samples
[2026-06-19 20:16:23] INFO segtask_v1.trainer.trainer: Epoch 727/1000 | LR=9.29e-04 | loss=0.2714 | val_dice=0.8036 | best=0.8292 (ep441) | 06:46:54 | L_main=0.1346 L_aux_1=0.1278(w=0.5) L_aux_2=0.1458(w=0.5)
[2026-06-19 20:16:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 727): 12688.6 MiB
[2026-06-19 20:17:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2985, pooled_mean_dice=0.8122, per_class=['0.8122'], iou=0.6838, recall=0.9749, precision=0.6960, vol_sim=0.8331, mcc=0.8192, min_class_dice=0.8122, coverage=[76]/88 samples
[2026-06-19 20:17:26] INFO segtask_v1.trainer.trainer: Epoch 728/1000 | LR=9.31e-04 | loss=0.2539 | val_dice=0.8122 | best=0.8292 (ep441) | 06:47:57 | L_main=0.1298 L_aux_1=0.1173(w=0.5) L_aux_2=0.1308(w=0.5)
[2026-06-19 20:17:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 728): 12688.6 MiB
[2026-06-19 20:18:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2776, pooled_mean_dice=0.7989, per_class=['0.7989'], iou=0.6651, recall=0.9781, precision=0.6751, vol_sim=0.8167, mcc=0.8077, min_class_dice=0.7989, coverage=[69]/88 samples
[2026-06-19 20:18:28] INFO segtask_v1.trainer.trainer: Epoch 729/1000 | LR=9.33e-04 | loss=0.2812 | val_dice=0.7989 | best=0.8292 (ep441) | 06:48:59 | L_main=0.1390 L_aux_1=0.1332(w=0.5) L_aux_2=0.1513(w=0.5)
[2026-06-19 20:18:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 729): 12688.6 MiB
[2026-06-19 20:19:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3078, pooled_mean_dice=0.8000, per_class=['0.8000'], iou=0.6666, recall=0.9714, precision=0.6800, vol_sim=0.8235, mcc=0.8086, min_class_dice=0.8000, coverage=[76]/88 samples
[2026-06-19 20:19:31] INFO segtask_v1.trainer.trainer: Epoch 730/1000 | LR=9.35e-04 | loss=0.2974 | val_dice=0.8000 | best=0.8292 (ep441) | 06:50:02 | L_main=0.1526 L_aux_1=0.1356(w=0.5) L_aux_2=0.1539(w=0.5)
[2026-06-19 20:19:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 730): 12688.6 MiB
[2026-06-19 20:20:34] INFO segtask_v1.trainer.validation:   Val: loss=0.2817, pooled_mean_dice=0.8099, per_class=['0.8099'], iou=0.6805, recall=0.9763, precision=0.6920, vol_sim=0.8296, mcc=0.8173, min_class_dice=0.8099, coverage=[73]/88 samples
[2026-06-19 20:20:34] INFO segtask_v1.trainer.trainer: Epoch 731/1000 | LR=9.37e-04 | loss=0.2967 | val_dice=0.8099 | best=0.8292 (ep441) | 06:51:06 | L_main=0.1492 L_aux_1=0.1388(w=0.5) L_aux_2=0.1563(w=0.5)
[2026-06-19 20:20:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 731): 12688.6 MiB
[2026-06-19 20:21:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2975, pooled_mean_dice=0.7990, per_class=['0.7990'], iou=0.6652, recall=0.9809, precision=0.6739, vol_sim=0.8145, mcc=0.8084, min_class_dice=0.7990, coverage=[78]/88 samples
[2026-06-19 20:21:36] INFO segtask_v1.trainer.trainer: Epoch 732/1000 | LR=9.39e-04 | loss=0.2668 | val_dice=0.7990 | best=0.8292 (ep441) | 06:52:07 | L_main=0.1331 L_aux_1=0.1259(w=0.5) L_aux_2=0.1415(w=0.5)
[2026-06-19 20:21:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 732): 12688.6 MiB
[2026-06-19 20:22:38] INFO segtask_v1.trainer.validation:   Val: loss=0.3038, pooled_mean_dice=0.8107, per_class=['0.8107'], iou=0.6816, recall=0.9778, precision=0.6923, vol_sim=0.8290, mcc=0.8185, min_class_dice=0.8107, coverage=[77]/88 samples
[2026-06-19 20:22:38] INFO segtask_v1.trainer.trainer: Epoch 733/1000 | LR=9.40e-04 | loss=0.2551 | val_dice=0.8107 | best=0.8292 (ep441) | 06:53:09 | L_main=0.1288 L_aux_1=0.1211(w=0.5) L_aux_2=0.1315(w=0.5)
[2026-06-19 20:22:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 733): 12688.6 MiB
[2026-06-19 20:23:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3006, pooled_mean_dice=0.8146, per_class=['0.8146'], iou=0.6872, recall=0.9830, precision=0.6955, vol_sim=0.8287, mcc=0.8212, min_class_dice=0.8146, coverage=[78]/88 samples
[2026-06-19 20:23:39] INFO segtask_v1.trainer.trainer: Epoch 734/1000 | LR=9.42e-04 | loss=0.2655 | val_dice=0.8146 | best=0.8292 (ep441) | 06:54:10 | L_main=0.1336 L_aux_1=0.1235(w=0.5) L_aux_2=0.1402(w=0.5)
[2026-06-19 20:23:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 734): 12688.6 MiB
[2026-06-19 20:24:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2886, pooled_mean_dice=0.7909, per_class=['0.7909'], iou=0.6542, recall=0.9768, precision=0.6645, vol_sim=0.8097, mcc=0.8003, min_class_dice=0.7909, coverage=[75]/88 samples
[2026-06-19 20:24:42] INFO segtask_v1.trainer.trainer: Epoch 735/1000 | LR=9.44e-04 | loss=0.2861 | val_dice=0.7909 | best=0.8292 (ep441) | 06:55:14 | L_main=0.1457 L_aux_1=0.1327(w=0.5) L_aux_2=0.1482(w=0.5)
[2026-06-19 20:24:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 735): 12688.6 MiB
[2026-06-19 20:25:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3130, pooled_mean_dice=0.8041, per_class=['0.8041'], iou=0.6724, recall=0.9788, precision=0.6824, vol_sim=0.8216, mcc=0.8123, min_class_dice=0.8041, coverage=[76]/88 samples
[2026-06-19 20:25:45] INFO segtask_v1.trainer.trainer: Epoch 736/1000 | LR=9.46e-04 | loss=0.2816 | val_dice=0.8041 | best=0.8292 (ep441) | 06:56:16 | L_main=0.1430 L_aux_1=0.1283(w=0.5) L_aux_2=0.1488(w=0.5)
[2026-06-19 20:25:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 736): 12688.6 MiB
[2026-06-19 20:26:47] INFO segtask_v1.trainer.validation:   Val: loss=0.3391, pooled_mean_dice=0.8105, per_class=['0.8105'], iou=0.6814, recall=0.9815, precision=0.6902, vol_sim=0.8258, mcc=0.8183, min_class_dice=0.8105, coverage=[78]/88 samples
[2026-06-19 20:26:47] INFO segtask_v1.trainer.trainer: Epoch 737/1000 | LR=9.48e-04 | loss=0.2476 | val_dice=0.8105 | best=0.8292 (ep441) | 06:57:18 | L_main=0.1267 L_aux_1=0.1131(w=0.5) L_aux_2=0.1288(w=0.5)
[2026-06-19 20:26:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 737): 12688.6 MiB
[2026-06-19 20:27:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3391, pooled_mean_dice=0.8033, per_class=['0.8033'], iou=0.6713, recall=0.9828, precision=0.6793, vol_sim=0.8174, mcc=0.8122, min_class_dice=0.8033, coverage=[78]/88 samples
[2026-06-19 20:27:49] INFO segtask_v1.trainer.trainer: Epoch 738/1000 | LR=9.50e-04 | loss=0.2334 | val_dice=0.8033 | best=0.8292 (ep441) | 06:58:20 | L_main=0.1193 L_aux_1=0.1095(w=0.5) L_aux_2=0.1188(w=0.5)
[2026-06-19 20:27:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 738): 12688.6 MiB
[2026-06-19 20:28:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2942, pooled_mean_dice=0.8054, per_class=['0.8054'], iou=0.6741, recall=0.9815, precision=0.6828, vol_sim=0.8206, mcc=0.8137, min_class_dice=0.8054, coverage=[72]/88 samples
[2026-06-19 20:28:51] INFO segtask_v1.trainer.trainer: Epoch 739/1000 | LR=9.51e-04 | loss=0.2475 | val_dice=0.8054 | best=0.8292 (ep441) | 06:59:23 | L_main=0.1236 L_aux_1=0.1157(w=0.5) L_aux_2=0.1320(w=0.5)
[2026-06-19 20:28:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 739): 12688.6 MiB
[2026-06-19 20:29:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3062, pooled_mean_dice=0.8128, per_class=['0.8128'], iou=0.6846, recall=0.9865, precision=0.6910, vol_sim=0.8239, mcc=0.8201, min_class_dice=0.8128, coverage=[77]/88 samples
[2026-06-19 20:29:55] INFO segtask_v1.trainer.trainer: Epoch 740/1000 | LR=9.53e-04 | loss=0.2973 | val_dice=0.8128 | best=0.8292 (ep441) | 07:00:26 | L_main=0.1503 L_aux_1=0.1396(w=0.5) L_aux_2=0.1545(w=0.5)
[2026-06-19 20:29:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 740): 12688.6 MiB
[2026-06-19 20:30:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3257, pooled_mean_dice=0.7799, per_class=['0.7799'], iou=0.6392, recall=0.9808, precision=0.6473, vol_sim=0.7952, mcc=0.7918, min_class_dice=0.7799, coverage=[78]/88 samples
[2026-06-19 20:30:58] INFO segtask_v1.trainer.trainer: Epoch 741/1000 | LR=9.55e-04 | loss=0.2687 | val_dice=0.7799 | best=0.8292 (ep441) | 07:01:29 | L_main=0.1368 L_aux_1=0.1239(w=0.5) L_aux_2=0.1400(w=0.5)
[2026-06-19 20:30:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 741): 12688.6 MiB
[2026-06-19 20:32:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3605, pooled_mean_dice=0.7900, per_class=['0.7900'], iou=0.6529, recall=0.9862, precision=0.6589, vol_sim=0.8010, mcc=0.8005, min_class_dice=0.7900, coverage=[81]/88 samples
[2026-06-19 20:32:00] INFO segtask_v1.trainer.trainer: Epoch 742/1000 | LR=9.56e-04 | loss=0.3043 | val_dice=0.7900 | best=0.8292 (ep441) | 07:02:31 | L_main=0.1476 L_aux_1=0.1474(w=0.5) L_aux_2=0.1660(w=0.5)
[2026-06-19 20:32:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 742): 12688.6 MiB
[2026-06-19 20:33:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3214, pooled_mean_dice=0.7967, per_class=['0.7967'], iou=0.6621, recall=0.9843, precision=0.6692, vol_sim=0.8094, mcc=0.8063, min_class_dice=0.7967, coverage=[77]/88 samples
[2026-06-19 20:33:02] INFO segtask_v1.trainer.trainer: Epoch 743/1000 | LR=9.58e-04 | loss=0.3376 | val_dice=0.7967 | best=0.8292 (ep441) | 07:03:33 | L_main=0.1669 L_aux_1=0.1636(w=0.5) L_aux_2=0.1777(w=0.5)
[2026-06-19 20:33:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 743): 12688.6 MiB
[2026-06-19 20:34:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3255, pooled_mean_dice=0.7777, per_class=['0.7777'], iou=0.6362, recall=0.9830, precision=0.6433, vol_sim=0.7911, mcc=0.7900, min_class_dice=0.7777, coverage=[74]/88 samples
[2026-06-19 20:34:05] INFO segtask_v1.trainer.trainer: Epoch 744/1000 | LR=9.59e-04 | loss=0.3525 | val_dice=0.7777 | best=0.8292 (ep441) | 07:04:36 | L_main=0.1803 L_aux_1=0.1637(w=0.5) L_aux_2=0.1807(w=0.5)
[2026-06-19 20:34:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 744): 12688.6 MiB
[2026-06-19 20:35:08] INFO segtask_v1.trainer.validation:   Val: loss=0.3374, pooled_mean_dice=0.7926, per_class=['0.7926'], iou=0.6565, recall=0.9847, precision=0.6632, vol_sim=0.8050, mcc=0.8032, min_class_dice=0.7926, coverage=[76]/88 samples
[2026-06-19 20:35:08] INFO segtask_v1.trainer.trainer: Epoch 745/1000 | LR=9.61e-04 | loss=0.3257 | val_dice=0.7926 | best=0.8292 (ep441) | 07:05:39 | L_main=0.1616 L_aux_1=0.1554(w=0.5) L_aux_2=0.1728(w=0.5)
[2026-06-19 20:35:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 745): 12688.6 MiB
[2026-06-19 20:36:10] INFO segtask_v1.trainer.validation:   Val: loss=0.3047, pooled_mean_dice=0.7813, per_class=['0.7813'], iou=0.6412, recall=0.9849, precision=0.6475, vol_sim=0.7933, mcc=0.7926, min_class_dice=0.7813, coverage=[78]/88 samples
[2026-06-19 20:36:10] INFO segtask_v1.trainer.trainer: Epoch 746/1000 | LR=9.63e-04 | loss=0.2621 | val_dice=0.7813 | best=0.8292 (ep441) | 07:06:42 | L_main=0.1364 L_aux_1=0.1174(w=0.5) L_aux_2=0.1339(w=0.5)
[2026-06-19 20:36:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 746): 12688.6 MiB
[2026-06-19 20:37:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3267, pooled_mean_dice=0.7963, per_class=['0.7963'], iou=0.6616, recall=0.9843, precision=0.6687, vol_sim=0.8091, mcc=0.8064, min_class_dice=0.7963, coverage=[71]/88 samples
[2026-06-19 20:37:13] INFO segtask_v1.trainer.trainer: Epoch 747/1000 | LR=9.64e-04 | loss=0.2646 | val_dice=0.7963 | best=0.8292 (ep441) | 07:07:45 | L_main=0.1355 L_aux_1=0.1229(w=0.5) L_aux_2=0.1355(w=0.5)
[2026-06-19 20:37:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 747): 12688.6 MiB
[2026-06-19 20:38:17] INFO segtask_v1.trainer.validation:   Val: loss=0.3596, pooled_mean_dice=0.7835, per_class=['0.7835'], iou=0.6441, recall=0.9861, precision=0.6500, vol_sim=0.7946, mcc=0.7941, min_class_dice=0.7835, coverage=[79]/88 samples
[2026-06-19 20:38:17] INFO segtask_v1.trainer.trainer: Epoch 748/1000 | LR=9.66e-04 | loss=0.2735 | val_dice=0.7835 | best=0.8292 (ep441) | 07:08:48 | L_main=0.1425 L_aux_1=0.1281(w=0.5) L_aux_2=0.1339(w=0.5)
[2026-06-19 20:38:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 748): 12688.6 MiB
[2026-06-19 20:39:19] INFO segtask_v1.trainer.validation:   Val: loss=0.3162, pooled_mean_dice=0.7821, per_class=['0.7821'], iou=0.6421, recall=0.9818, precision=0.6499, vol_sim=0.7966, mcc=0.7941, min_class_dice=0.7821, coverage=[72]/88 samples
[2026-06-19 20:39:19] INFO segtask_v1.trainer.trainer: Epoch 749/1000 | LR=9.67e-04 | loss=0.2698 | val_dice=0.7821 | best=0.8292 (ep441) | 07:09:51 | L_main=0.1367 L_aux_1=0.1285(w=0.5) L_aux_2=0.1377(w=0.5)
[2026-06-19 20:39:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 749): 12688.6 MiB
[2026-06-19 20:40:22] INFO segtask_v1.trainer.validation:   Val: loss=0.3289, pooled_mean_dice=0.7857, per_class=['0.7857'], iou=0.6470, recall=0.9861, precision=0.6530, vol_sim=0.7968, mcc=0.7973, min_class_dice=0.7857, coverage=[76]/88 samples
[2026-06-19 20:40:22] INFO segtask_v1.trainer.trainer: Epoch 750/1000 | LR=9.68e-04 | loss=0.2630 | val_dice=0.7857 | best=0.8292 (ep441) | 07:10:54 | L_main=0.1370 L_aux_1=0.1205(w=0.5) L_aux_2=0.1316(w=0.5)
[2026-06-19 20:40:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 750): 12688.6 MiB
[2026-06-19 20:41:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3098, pooled_mean_dice=0.7795, per_class=['0.7795'], iou=0.6387, recall=0.9853, precision=0.6448, vol_sim=0.7911, mcc=0.7917, min_class_dice=0.7795, coverage=[75]/88 samples
[2026-06-19 20:41:25] INFO segtask_v1.trainer.trainer: Epoch 751/1000 | LR=9.70e-04 | loss=0.2930 | val_dice=0.7795 | best=0.8292 (ep441) | 07:11:56 | L_main=0.1503 L_aux_1=0.1390(w=0.5) L_aux_2=0.1465(w=0.5)
[2026-06-19 20:41:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 751): 12688.6 MiB
[2026-06-19 20:42:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3426, pooled_mean_dice=0.7703, per_class=['0.7703'], iou=0.6264, recall=0.9832, precision=0.6331, vol_sim=0.7834, mcc=0.7836, min_class_dice=0.7703, coverage=[78]/88 samples
[2026-06-19 20:42:28] INFO segtask_v1.trainer.trainer: Epoch 752/1000 | LR=9.71e-04 | loss=0.2608 | val_dice=0.7703 | best=0.8292 (ep441) | 07:13:00 | L_main=0.1290 L_aux_1=0.1246(w=0.5) L_aux_2=0.1388(w=0.5)
[2026-06-19 20:42:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 752): 12688.6 MiB
[2026-06-19 20:43:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3509, pooled_mean_dice=0.7942, per_class=['0.7942'], iou=0.6586, recall=0.9852, precision=0.6652, vol_sim=0.8061, mcc=0.8047, min_class_dice=0.7942, coverage=[76]/88 samples
[2026-06-19 20:43:32] INFO segtask_v1.trainer.trainer: Epoch 753/1000 | LR=9.72e-04 | loss=0.2497 | val_dice=0.7942 | best=0.8292 (ep441) | 07:14:03 | L_main=0.1304 L_aux_1=0.1128(w=0.5) L_aux_2=0.1259(w=0.5)
[2026-06-19 20:43:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 753): 12688.6 MiB
[2026-06-19 20:44:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3170, pooled_mean_dice=0.7745, per_class=['0.7745'], iou=0.6320, recall=0.9866, precision=0.6375, vol_sim=0.7850, mcc=0.7880, min_class_dice=0.7745, coverage=[74]/88 samples
[2026-06-19 20:44:34] INFO segtask_v1.trainer.trainer: Epoch 754/1000 | LR=9.74e-04 | loss=0.2603 | val_dice=0.7745 | best=0.8292 (ep441) | 07:15:06 | L_main=0.1384 L_aux_1=0.1179(w=0.5) L_aux_2=0.1259(w=0.5)
[2026-06-19 20:44:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 754): 12688.6 MiB
[2026-06-19 20:45:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3047, pooled_mean_dice=0.7815, per_class=['0.7815'], iou=0.6413, recall=0.9860, precision=0.6472, vol_sim=0.7926, mcc=0.7927, min_class_dice=0.7815, coverage=[75]/88 samples
[2026-06-19 20:45:37] INFO segtask_v1.trainer.trainer: Epoch 755/1000 | LR=9.75e-04 | loss=0.2463 | val_dice=0.7815 | best=0.8292 (ep441) | 07:16:08 | L_main=0.1236 L_aux_1=0.1173(w=0.5) L_aux_2=0.1281(w=0.5)
[2026-06-19 20:45:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 755): 12688.6 MiB
[2026-06-19 20:46:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3201, pooled_mean_dice=0.7977, per_class=['0.7977'], iou=0.6634, recall=0.9846, precision=0.6704, vol_sim=0.8101, mcc=0.8073, min_class_dice=0.7977, coverage=[78]/88 samples
[2026-06-19 20:46:40] INFO segtask_v1.trainer.trainer: Epoch 756/1000 | LR=9.76e-04 | loss=0.2497 | val_dice=0.7977 | best=0.8292 (ep441) | 07:17:11 | L_main=0.1266 L_aux_1=0.1188(w=0.5) L_aux_2=0.1275(w=0.5)
[2026-06-19 20:46:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 756): 12688.6 MiB
[2026-06-19 20:47:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3227, pooled_mean_dice=0.7797, per_class=['0.7797'], iou=0.6389, recall=0.9867, precision=0.6445, vol_sim=0.7902, mcc=0.7931, min_class_dice=0.7797, coverage=[73]/88 samples
[2026-06-19 20:47:42] INFO segtask_v1.trainer.trainer: Epoch 757/1000 | LR=9.77e-04 | loss=0.2482 | val_dice=0.7797 | best=0.8292 (ep441) | 07:18:14 | L_main=0.1274 L_aux_1=0.1123(w=0.5) L_aux_2=0.1294(w=0.5)
[2026-06-19 20:47:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 757): 12688.6 MiB
[2026-06-19 20:48:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3331, pooled_mean_dice=0.7962, per_class=['0.7962'], iou=0.6614, recall=0.9862, precision=0.6676, vol_sim=0.8073, mcc=0.8067, min_class_dice=0.7962, coverage=[75]/88 samples
[2026-06-19 20:48:46] INFO segtask_v1.trainer.trainer: Epoch 758/1000 | LR=9.79e-04 | loss=0.2457 | val_dice=0.7962 | best=0.8292 (ep441) | 07:19:17 | L_main=0.1275 L_aux_1=0.1147(w=0.5) L_aux_2=0.1218(w=0.5)
[2026-06-19 20:48:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 758): 12688.6 MiB
[2026-06-19 20:49:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3385, pooled_mean_dice=0.7864, per_class=['0.7864'], iou=0.6480, recall=0.9852, precision=0.6544, vol_sim=0.7983, mcc=0.7979, min_class_dice=0.7864, coverage=[77]/88 samples
[2026-06-19 20:49:48] INFO segtask_v1.trainer.trainer: Epoch 759/1000 | LR=9.80e-04 | loss=0.2767 | val_dice=0.7864 | best=0.8292 (ep441) | 07:20:19 | L_main=0.1370 L_aux_1=0.1337(w=0.5) L_aux_2=0.1457(w=0.5)
[2026-06-19 20:49:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 759): 12688.6 MiB
[2026-06-19 20:50:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3273, pooled_mean_dice=0.7853, per_class=['0.7853'], iou=0.6465, recall=0.9844, precision=0.6532, vol_sim=0.7978, mcc=0.7962, min_class_dice=0.7853, coverage=[77]/88 samples
[2026-06-19 20:50:49] INFO segtask_v1.trainer.trainer: Epoch 760/1000 | LR=9.81e-04 | loss=0.2797 | val_dice=0.7853 | best=0.8292 (ep441) | 07:21:20 | L_main=0.1384 L_aux_1=0.1330(w=0.5) L_aux_2=0.1496(w=0.5)
[2026-06-19 20:50:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 760): 12688.6 MiB
[2026-06-19 20:51:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3392, pooled_mean_dice=0.7832, per_class=['0.7832'], iou=0.6437, recall=0.9827, precision=0.6511, vol_sim=0.7970, mcc=0.7949, min_class_dice=0.7832, coverage=[80]/88 samples
[2026-06-19 20:51:53] INFO segtask_v1.trainer.trainer: Epoch 761/1000 | LR=9.82e-04 | loss=0.2326 | val_dice=0.7832 | best=0.8292 (ep441) | 07:22:24 | L_main=0.1214 L_aux_1=0.1045(w=0.5) L_aux_2=0.1179(w=0.5)
[2026-06-19 20:51:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 761): 12688.6 MiB
[2026-06-19 20:52:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3256, pooled_mean_dice=0.7807, per_class=['0.7807'], iou=0.6403, recall=0.9821, precision=0.6478, vol_sim=0.7949, mcc=0.7934, min_class_dice=0.7807, coverage=[75]/88 samples
[2026-06-19 20:52:54] INFO segtask_v1.trainer.trainer: Epoch 762/1000 | LR=9.83e-04 | loss=0.2786 | val_dice=0.7807 | best=0.8292 (ep441) | 07:23:25 | L_main=0.1428 L_aux_1=0.1297(w=0.5) L_aux_2=0.1419(w=0.5)
[2026-06-19 20:52:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 762): 12688.6 MiB
[2026-06-19 20:53:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3388, pooled_mean_dice=0.7953, per_class=['0.7953'], iou=0.6601, recall=0.9832, precision=0.6677, vol_sim=0.8089, mcc=0.8043, min_class_dice=0.7953, coverage=[79]/88 samples
[2026-06-19 20:53:57] INFO segtask_v1.trainer.trainer: Epoch 763/1000 | LR=9.84e-04 | loss=0.2401 | val_dice=0.7953 | best=0.8292 (ep441) | 07:24:28 | L_main=0.1242 L_aux_1=0.1116(w=0.5) L_aux_2=0.1203(w=0.5)
[2026-06-19 20:53:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 763): 12688.6 MiB
[2026-06-19 20:54:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3369, pooled_mean_dice=0.8093, per_class=['0.8093'], iou=0.6796, recall=0.9866, precision=0.6860, vol_sim=0.8202, mcc=0.8168, min_class_dice=0.8093, coverage=[80]/88 samples
[2026-06-19 20:54:58] INFO segtask_v1.trainer.trainer: Epoch 764/1000 | LR=9.85e-04 | loss=0.4360 | val_dice=0.8093 | best=0.8292 (ep441) | 07:25:29 | L_main=0.2132 L_aux_1=0.2119(w=0.5) L_aux_2=0.2337(w=0.5)
[2026-06-19 20:54:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 764): 12688.6 MiB
[2026-06-19 20:56:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3266, pooled_mean_dice=0.7957, per_class=['0.7957'], iou=0.6607, recall=0.9825, precision=0.6686, vol_sim=0.8099, mcc=0.8058, min_class_dice=0.7957, coverage=[74]/88 samples
[2026-06-19 20:56:00] INFO segtask_v1.trainer.trainer: Epoch 765/1000 | LR=9.86e-04 | loss=0.3259 | val_dice=0.7957 | best=0.8292 (ep441) | 07:26:31 | L_main=0.1622 L_aux_1=0.1515(w=0.5) L_aux_2=0.1758(w=0.5)
[2026-06-19 20:56:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 765): 12688.6 MiB
[2026-06-19 20:57:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3249, pooled_mean_dice=0.7807, per_class=['0.7807'], iou=0.6403, recall=0.9760, precision=0.6505, vol_sim=0.7999, mcc=0.7925, min_class_dice=0.7807, coverage=[73]/88 samples
[2026-06-19 20:57:02] INFO segtask_v1.trainer.trainer: Epoch 766/1000 | LR=9.87e-04 | loss=0.3106 | val_dice=0.7807 | best=0.8292 (ep441) | 07:27:33 | L_main=0.1563 L_aux_1=0.1466(w=0.5) L_aux_2=0.1620(w=0.5)
[2026-06-19 20:57:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 766): 12688.6 MiB
[2026-06-19 20:58:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2893, pooled_mean_dice=0.7975, per_class=['0.7975'], iou=0.6632, recall=0.9816, precision=0.6715, vol_sim=0.8124, mcc=0.8066, min_class_dice=0.7975, coverage=[74]/88 samples
[2026-06-19 20:58:03] INFO segtask_v1.trainer.trainer: Epoch 767/1000 | LR=9.88e-04 | loss=0.3180 | val_dice=0.7975 | best=0.8292 (ep441) | 07:28:34 | L_main=0.1594 L_aux_1=0.1491(w=0.5) L_aux_2=0.1681(w=0.5)
[2026-06-19 20:58:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 767): 12688.6 MiB
[2026-06-19 20:59:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3235, pooled_mean_dice=0.7878, per_class=['0.7878'], iou=0.6499, recall=0.9777, precision=0.6597, vol_sim=0.8058, mcc=0.7973, min_class_dice=0.7878, coverage=[79]/88 samples
[2026-06-19 20:59:05] INFO segtask_v1.trainer.trainer: Epoch 768/1000 | LR=9.89e-04 | loss=0.2863 | val_dice=0.7878 | best=0.8292 (ep441) | 07:29:36 | L_main=0.1482 L_aux_1=0.1290(w=0.5) L_aux_2=0.1473(w=0.5)
[2026-06-19 20:59:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 768): 12688.6 MiB
[2026-06-19 21:00:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3174, pooled_mean_dice=0.8029, per_class=['0.8029'], iou=0.6707, recall=0.9786, precision=0.6807, vol_sim=0.8205, mcc=0.8111, min_class_dice=0.8029, coverage=[75]/88 samples
[2026-06-19 21:00:07] INFO segtask_v1.trainer.trainer: Epoch 769/1000 | LR=9.89e-04 | loss=0.3301 | val_dice=0.8029 | best=0.8292 (ep441) | 07:30:39 | L_main=0.1653 L_aux_1=0.1556(w=0.5) L_aux_2=0.1739(w=0.5)
[2026-06-19 21:00:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 769): 12688.6 MiB
[2026-06-19 21:01:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3146, pooled_mean_dice=0.7986, per_class=['0.7986'], iou=0.6647, recall=0.9834, precision=0.6723, vol_sim=0.8121, mcc=0.8068, min_class_dice=0.7986, coverage=[82]/88 samples
[2026-06-19 21:01:09] INFO segtask_v1.trainer.trainer: Epoch 770/1000 | LR=9.90e-04 | loss=0.3032 | val_dice=0.7986 | best=0.8292 (ep441) | 07:31:41 | L_main=0.1507 L_aux_1=0.1429(w=0.5) L_aux_2=0.1621(w=0.5)
[2026-06-19 21:01:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 770): 12688.6 MiB
[2026-06-19 21:02:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2856, pooled_mean_dice=0.7993, per_class=['0.7993'], iou=0.6657, recall=0.9785, precision=0.6756, vol_sim=0.8168, mcc=0.8081, min_class_dice=0.7993, coverage=[73]/88 samples
[2026-06-19 21:02:12] INFO segtask_v1.trainer.trainer: Epoch 771/1000 | LR=9.91e-04 | loss=0.2812 | val_dice=0.7993 | best=0.8292 (ep441) | 07:32:44 | L_main=0.1443 L_aux_1=0.1302(w=0.5) L_aux_2=0.1436(w=0.5)
[2026-06-19 21:02:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 771): 12688.6 MiB
[2026-06-19 21:03:16] INFO segtask_v1.trainer.validation:   Val: loss=0.3186, pooled_mean_dice=0.8013, per_class=['0.8013'], iou=0.6685, recall=0.9804, precision=0.6775, vol_sim=0.8173, mcc=0.8099, min_class_dice=0.8013, coverage=[75]/88 samples
[2026-06-19 21:03:16] INFO segtask_v1.trainer.trainer: Epoch 772/1000 | LR=9.92e-04 | loss=0.2952 | val_dice=0.8013 | best=0.8292 (ep441) | 07:33:47 | L_main=0.1506 L_aux_1=0.1363(w=0.5) L_aux_2=0.1529(w=0.5)
[2026-06-19 21:03:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 772): 12688.6 MiB
[2026-06-19 21:04:19] INFO segtask_v1.trainer.validation:   Val: loss=0.3100, pooled_mean_dice=0.7971, per_class=['0.7971'], iou=0.6627, recall=0.9779, precision=0.6728, vol_sim=0.8152, mcc=0.8056, min_class_dice=0.7971, coverage=[76]/88 samples
[2026-06-19 21:04:19] INFO segtask_v1.trainer.trainer: Epoch 773/1000 | LR=9.92e-04 | loss=0.2533 | val_dice=0.7971 | best=0.8292 (ep441) | 07:34:50 | L_main=0.1275 L_aux_1=0.1147(w=0.5) L_aux_2=0.1369(w=0.5)
[2026-06-19 21:04:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 773): 12688.6 MiB
[2026-06-19 21:05:21] INFO segtask_v1.trainer.validation:   Val: loss=0.3053, pooled_mean_dice=0.8065, per_class=['0.8065'], iou=0.6758, recall=0.9797, precision=0.6853, vol_sim=0.8232, mcc=0.8141, min_class_dice=0.8065, coverage=[78]/88 samples
[2026-06-19 21:05:21] INFO segtask_v1.trainer.trainer: Epoch 774/1000 | LR=9.93e-04 | loss=0.2609 | val_dice=0.8065 | best=0.8292 (ep441) | 07:35:53 | L_main=0.1302 L_aux_1=0.1196(w=0.5) L_aux_2=0.1418(w=0.5)
[2026-06-19 21:05:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 774): 12688.6 MiB
[2026-06-19 21:06:24] INFO segtask_v1.trainer.validation:   Val: loss=0.3020, pooled_mean_dice=0.8161, per_class=['0.8161'], iou=0.6893, recall=0.9814, precision=0.6985, vol_sim=0.8316, mcc=0.8232, min_class_dice=0.8161, coverage=[77]/88 samples
[2026-06-19 21:06:24] INFO segtask_v1.trainer.trainer: Epoch 775/1000 | LR=9.94e-04 | loss=0.2679 | val_dice=0.8161 | best=0.8292 (ep441) | 07:36:55 | L_main=0.1372 L_aux_1=0.1246(w=0.5) L_aux_2=0.1369(w=0.5)
[2026-06-19 21:06:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 775): 12688.6 MiB
[2026-06-19 21:07:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3014, pooled_mean_dice=0.8065, per_class=['0.8065'], iou=0.6758, recall=0.9768, precision=0.6868, vol_sim=0.8257, mcc=0.8148, min_class_dice=0.8065, coverage=[78]/88 samples
[2026-06-19 21:07:25] INFO segtask_v1.trainer.trainer: Epoch 776/1000 | LR=9.94e-04 | loss=0.2358 | val_dice=0.8065 | best=0.8292 (ep441) | 07:37:57 | L_main=0.1189 L_aux_1=0.1062(w=0.5) L_aux_2=0.1276(w=0.5)
[2026-06-19 21:07:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 776): 12688.6 MiB
[2026-06-19 21:08:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3232, pooled_mean_dice=0.8053, per_class=['0.8053'], iou=0.6740, recall=0.9804, precision=0.6832, vol_sim=0.8213, mcc=0.8136, min_class_dice=0.8053, coverage=[76]/88 samples
[2026-06-19 21:08:28] INFO segtask_v1.trainer.trainer: Epoch 777/1000 | LR=9.95e-04 | loss=0.3094 | val_dice=0.8053 | best=0.8292 (ep441) | 07:39:00 | L_main=0.1597 L_aux_1=0.1451(w=0.5) L_aux_2=0.1543(w=0.5)
[2026-06-19 21:08:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 777): 12688.6 MiB
[2026-06-19 21:09:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2723, pooled_mean_dice=0.8040, per_class=['0.8040'], iou=0.6723, recall=0.9762, precision=0.6835, vol_sim=0.8236, mcc=0.8116, min_class_dice=0.8040, coverage=[76]/88 samples
[2026-06-19 21:09:31] INFO segtask_v1.trainer.trainer: Epoch 778/1000 | LR=9.95e-04 | loss=0.2716 | val_dice=0.8040 | best=0.8292 (ep441) | 07:40:02 | L_main=0.1431 L_aux_1=0.1198(w=0.5) L_aux_2=0.1374(w=0.5)
[2026-06-19 21:09:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 778): 12688.6 MiB
[2026-06-19 21:10:33] INFO segtask_v1.trainer.validation:   Val: loss=0.3139, pooled_mean_dice=0.8102, per_class=['0.8102'], iou=0.6810, recall=0.9820, precision=0.6895, vol_sim=0.8250, mcc=0.8177, min_class_dice=0.8102, coverage=[77]/88 samples
[2026-06-19 21:10:33] INFO segtask_v1.trainer.trainer: Epoch 779/1000 | LR=9.96e-04 | loss=0.2717 | val_dice=0.8102 | best=0.8292 (ep441) | 07:41:04 | L_main=0.1355 L_aux_1=0.1283(w=0.5) L_aux_2=0.1441(w=0.5)
[2026-06-19 21:10:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 779): 12688.6 MiB
[2026-06-19 21:11:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3430, pooled_mean_dice=0.7868, per_class=['0.7868'], iou=0.6486, recall=0.9790, precision=0.6577, vol_sim=0.8037, mcc=0.7977, min_class_dice=0.7868, coverage=[77]/88 samples
[2026-06-19 21:11:34] INFO segtask_v1.trainer.trainer: Epoch 780/1000 | LR=9.96e-04 | loss=0.2473 | val_dice=0.7868 | best=0.8292 (ep441) | 07:42:06 | L_main=0.1270 L_aux_1=0.1136(w=0.5) L_aux_2=0.1270(w=0.5)
[2026-06-19 21:11:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 780): 12688.6 MiB
[2026-06-19 21:12:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3369, pooled_mean_dice=0.8062, per_class=['0.8062'], iou=0.6753, recall=0.9762, precision=0.6866, vol_sim=0.8259, mcc=0.8141, min_class_dice=0.8062, coverage=[74]/88 samples
[2026-06-19 21:12:37] INFO segtask_v1.trainer.trainer: Epoch 781/1000 | LR=9.97e-04 | loss=0.2844 | val_dice=0.8062 | best=0.8292 (ep441) | 07:43:08 | L_main=0.1440 L_aux_1=0.1352(w=0.5) L_aux_2=0.1458(w=0.5)
[2026-06-19 21:12:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 781): 12688.6 MiB
[2026-06-19 21:13:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3234, pooled_mean_dice=0.7966, per_class=['0.7966'], iou=0.6619, recall=0.9829, precision=0.6696, vol_sim=0.8104, mcc=0.8062, min_class_dice=0.7966, coverage=[75]/88 samples
[2026-06-19 21:13:40] INFO segtask_v1.trainer.trainer: Epoch 782/1000 | LR=9.97e-04 | loss=0.3325 | val_dice=0.7966 | best=0.8292 (ep441) | 07:44:11 | L_main=0.1669 L_aux_1=0.1571(w=0.5) L_aux_2=0.1741(w=0.5)
[2026-06-19 21:13:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 782): 12688.6 MiB
[2026-06-19 21:14:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2863, pooled_mean_dice=0.8070, per_class=['0.8070'], iou=0.6764, recall=0.9824, precision=0.6848, vol_sim=0.8215, mcc=0.8141, min_class_dice=0.8070, coverage=[75]/88 samples
[2026-06-19 21:14:43] INFO segtask_v1.trainer.trainer: Epoch 783/1000 | LR=9.98e-04 | loss=0.2909 | val_dice=0.8070 | best=0.8292 (ep441) | 07:45:14 | L_main=0.1507 L_aux_1=0.1361(w=0.5) L_aux_2=0.1442(w=0.5)
[2026-06-19 21:14:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 783): 12688.6 MiB
[2026-06-19 21:15:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3399, pooled_mean_dice=0.7914, per_class=['0.7914'], iou=0.6549, recall=0.9805, precision=0.6635, vol_sim=0.8072, mcc=0.8016, min_class_dice=0.7914, coverage=[78]/88 samples
[2026-06-19 21:15:45] INFO segtask_v1.trainer.trainer: Epoch 784/1000 | LR=9.98e-04 | loss=0.2545 | val_dice=0.7914 | best=0.8292 (ep441) | 07:46:17 | L_main=0.1323 L_aux_1=0.1142(w=0.5) L_aux_2=0.1302(w=0.5)
[2026-06-19 21:15:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 784): 12688.6 MiB
[2026-06-19 21:16:47] INFO segtask_v1.trainer.validation:   Val: loss=0.3268, pooled_mean_dice=0.7887, per_class=['0.7887'], iou=0.6511, recall=0.9777, precision=0.6609, vol_sim=0.8066, mcc=0.7990, min_class_dice=0.7887, coverage=[77]/88 samples
[2026-06-19 21:16:47] INFO segtask_v1.trainer.trainer: Epoch 785/1000 | LR=9.98e-04 | loss=0.2584 | val_dice=0.7887 | best=0.8292 (ep441) | 07:47:18 | L_main=0.1281 L_aux_1=0.1244(w=0.5) L_aux_2=0.1362(w=0.5)
[2026-06-19 21:16:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 785): 12688.6 MiB
[2026-06-19 21:17:50] INFO segtask_v1.trainer.validation:   Val: loss=0.3101, pooled_mean_dice=0.8006, per_class=['0.8006'], iou=0.6675, recall=0.9837, precision=0.6749, vol_sim=0.8139, mcc=0.8088, min_class_dice=0.8006, coverage=[80]/88 samples
[2026-06-19 21:17:50] INFO segtask_v1.trainer.trainer: Epoch 786/1000 | LR=9.99e-04 | loss=0.2565 | val_dice=0.8006 | best=0.8292 (ep441) | 07:48:22 | L_main=0.1270 L_aux_1=0.1201(w=0.5) L_aux_2=0.1390(w=0.5)
[2026-06-19 21:17:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 786): 12688.6 MiB
[2026-06-19 21:18:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3071, pooled_mean_dice=0.7890, per_class=['0.7890'], iou=0.6515, recall=0.9833, precision=0.6588, vol_sim=0.8024, mcc=0.8001, min_class_dice=0.7890, coverage=[72]/88 samples
[2026-06-19 21:18:53] INFO segtask_v1.trainer.trainer: Epoch 787/1000 | LR=9.99e-04 | loss=0.2544 | val_dice=0.7890 | best=0.8292 (ep441) | 07:49:24 | L_main=0.1309 L_aux_1=0.1184(w=0.5) L_aux_2=0.1286(w=0.5)
[2026-06-19 21:18:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 787): 12688.6 MiB
[2026-06-19 21:19:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3385, pooled_mean_dice=0.7999, per_class=['0.7999'], iou=0.6665, recall=0.9851, precision=0.6733, vol_sim=0.8120, mcc=0.8090, min_class_dice=0.7999, coverage=[80]/88 samples
[2026-06-19 21:19:55] INFO segtask_v1.trainer.trainer: Epoch 788/1000 | LR=9.99e-04 | loss=0.2602 | val_dice=0.7999 | best=0.8292 (ep441) | 07:50:26 | L_main=0.1314 L_aux_1=0.1223(w=0.5) L_aux_2=0.1351(w=0.5)
[2026-06-19 21:19:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 788): 12688.6 MiB
[2026-06-19 21:20:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3272, pooled_mean_dice=0.7967, per_class=['0.7967'], iou=0.6622, recall=0.9823, precision=0.6702, vol_sim=0.8111, mcc=0.8070, min_class_dice=0.7967, coverage=[75]/88 samples
[2026-06-19 21:20:58] INFO segtask_v1.trainer.trainer: Epoch 789/1000 | LR=9.99e-04 | loss=0.2720 | val_dice=0.7967 | best=0.8292 (ep441) | 07:51:29 | L_main=0.1373 L_aux_1=0.1304(w=0.5) L_aux_2=0.1390(w=0.5)
[2026-06-19 21:20:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 789): 12688.6 MiB
[2026-06-19 21:22:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3405, pooled_mean_dice=0.8123, per_class=['0.8123'], iou=0.6840, recall=0.9854, precision=0.6910, vol_sim=0.8244, mcc=0.8202, min_class_dice=0.8123, coverage=[82]/88 samples
[2026-06-19 21:22:01] INFO segtask_v1.trainer.trainer: Epoch 790/1000 | LR=1.00e-03 | loss=0.2668 | val_dice=0.8123 | best=0.8292 (ep441) | 07:52:32 | L_main=0.1320 L_aux_1=0.1244(w=0.5) L_aux_2=0.1452(w=0.5)
[2026-06-19 21:22:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 790): 12688.6 MiB
[2026-06-19 21:23:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3004, pooled_mean_dice=0.7890, per_class=['0.7890'], iou=0.6515, recall=0.9812, precision=0.6597, vol_sim=0.8041, mcc=0.8006, min_class_dice=0.7890, coverage=[69]/88 samples
[2026-06-19 21:23:04] INFO segtask_v1.trainer.trainer: Epoch 791/1000 | LR=1.00e-03 | loss=0.2386 | val_dice=0.7890 | best=0.8292 (ep441) | 07:53:35 | L_main=0.1217 L_aux_1=0.1102(w=0.5) L_aux_2=0.1236(w=0.5)
[2026-06-19 21:23:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 791): 12688.6 MiB
[2026-06-19 21:24:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3314, pooled_mean_dice=0.7977, per_class=['0.7977'], iou=0.6635, recall=0.9854, precision=0.6701, vol_sim=0.8095, mcc=0.8077, min_class_dice=0.7977, coverage=[73]/88 samples
[2026-06-19 21:24:07] INFO segtask_v1.trainer.trainer: Epoch 792/1000 | LR=1.00e-03 | loss=0.2523 | val_dice=0.7977 | best=0.8292 (ep441) | 07:54:38 | L_main=0.1261 L_aux_1=0.1136(w=0.5) L_aux_2=0.1388(w=0.5)
[2026-06-19 21:24:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 792): 12688.6 MiB
[2026-06-19 21:25:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2993, pooled_mean_dice=0.7991, per_class=['0.7991'], iou=0.6655, recall=0.9835, precision=0.6730, vol_sim=0.8125, mcc=0.8081, min_class_dice=0.7991, coverage=[77]/88 samples
[2026-06-19 21:25:10] INFO segtask_v1.trainer.trainer: Epoch 793/1000 | LR=1.00e-03 | loss=0.2310 | val_dice=0.7991 | best=0.8292 (ep441) | 07:55:41 | L_main=0.1191 L_aux_1=0.1078(w=0.5) L_aux_2=0.1159(w=0.5)
[2026-06-19 21:25:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 793): 12688.6 MiB
[2026-06-19 21:26:12] INFO segtask_v1.trainer.validation:   Val: loss=0.3148, pooled_mean_dice=0.7818, per_class=['0.7818'], iou=0.6417, recall=0.9803, precision=0.6501, vol_sim=0.7975, mcc=0.7935, min_class_dice=0.7818, coverage=[77]/88 samples
[2026-06-19 21:26:12] INFO segtask_v1.trainer.trainer: Epoch 794/1000 | LR=1.00e-03 | loss=0.2588 | val_dice=0.7818 | best=0.8292 (ep441) | 07:56:43 | L_main=0.1324 L_aux_1=0.1224(w=0.5) L_aux_2=0.1305(w=0.5)
[2026-06-19 21:26:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 794): 12688.6 MiB
[2026-06-19 21:27:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3512, pooled_mean_dice=0.7922, per_class=['0.7922'], iou=0.6559, recall=0.9826, precision=0.6636, vol_sim=0.8062, mcc=0.8019, min_class_dice=0.7922, coverage=[81]/88 samples
[2026-06-19 21:27:13] INFO segtask_v1.trainer.trainer: Epoch 795/1000 | LR=1.00e-03 | loss=0.2478 | val_dice=0.7922 | best=0.8292 (ep441) | 07:57:44 | L_main=0.1300 L_aux_1=0.1126(w=0.5) L_aux_2=0.1230(w=0.5)
[2026-06-19 21:27:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 795): 12688.6 MiB
[2026-06-19 21:28:16] INFO segtask_v1.trainer.validation:   Val: loss=0.3332, pooled_mean_dice=0.7731, per_class=['0.7731'], iou=0.6301, recall=0.9836, precision=0.6368, vol_sim=0.7859, mcc=0.7868, min_class_dice=0.7731, coverage=[75]/88 samples
[2026-06-19 21:28:16] INFO segtask_v1.trainer.trainer: Epoch 796/1000 | LR=1.00e-03 | loss=0.2591 | val_dice=0.7731 | best=0.8292 (ep441) | 07:58:47 | L_main=0.1285 L_aux_1=0.1262(w=0.5) L_aux_2=0.1350(w=0.5)
[2026-06-19 21:28:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 796): 12688.6 MiB
[2026-06-19 21:29:19] INFO segtask_v1.trainer.validation:   Val: loss=0.3022, pooled_mean_dice=0.8027, per_class=['0.8027'], iou=0.6705, recall=0.9836, precision=0.6781, vol_sim=0.8161, mcc=0.8111, min_class_dice=0.8027, coverage=[76]/88 samples
[2026-06-19 21:29:19] INFO segtask_v1.trainer.trainer: Epoch 797/1000 | LR=1.00e-03 | loss=0.2410 | val_dice=0.8027 | best=0.8292 (ep441) | 07:59:50 | L_main=0.1242 L_aux_1=0.1116(w=0.5) L_aux_2=0.1221(w=0.5)
[2026-06-19 21:29:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 797): 12688.6 MiB
[2026-06-19 21:30:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2919, pooled_mean_dice=0.7869, per_class=['0.7869'], iou=0.6487, recall=0.9880, precision=0.6539, vol_sim=0.7965, mcc=0.7979, min_class_dice=0.7869, coverage=[74]/88 samples
[2026-06-19 21:30:21] INFO segtask_v1.trainer.trainer: Epoch 798/1000 | LR=1.00e-03 | loss=0.2858 | val_dice=0.7869 | best=0.8292 (ep441) | 08:00:52 | L_main=0.1463 L_aux_1=0.1313(w=0.5) L_aux_2=0.1476(w=0.5)
[2026-06-19 21:30:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 798): 12688.6 MiB
[2026-06-19 21:31:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3243, pooled_mean_dice=0.7981, per_class=['0.7981'], iou=0.6640, recall=0.9827, precision=0.6718, vol_sim=0.8121, mcc=0.8078, min_class_dice=0.7981, coverage=[76]/88 samples
[2026-06-19 21:31:23] INFO segtask_v1.trainer.trainer: Epoch 799/1000 | LR=1.00e-03 | loss=0.2963 | val_dice=0.7981 | best=0.8292 (ep441) | 08:01:54 | L_main=0.1482 L_aux_1=0.1440(w=0.5) L_aux_2=0.1521(w=0.5)
[2026-06-19 21:31:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 799): 12688.6 MiB
[2026-06-19 21:32:24] INFO segtask_v1.trainer.validation:   Val: loss=0.3282, pooled_mean_dice=0.7795, per_class=['0.7795'], iou=0.6386, recall=0.9851, precision=0.6448, vol_sim=0.7913, mcc=0.7913, min_class_dice=0.7795, coverage=[77]/88 samples
[2026-06-19 21:32:24] INFO segtask_v1.trainer.trainer: Epoch 800/1000 | LR=1.00e-03 | loss=0.2980 | val_dice=0.7795 | best=0.8292 (ep441) | 08:02:56 | L_main=0.1459 L_aux_1=0.1442(w=0.5) L_aux_2=0.1599(w=0.5)
[2026-06-19 21:32:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 800): 12688.6 MiB
[2026-06-19 21:33:27] INFO segtask_v1.trainer.validation:   Val: loss=0.3156, pooled_mean_dice=0.7869, per_class=['0.7869'], iou=0.6486, recall=0.9846, precision=0.6553, vol_sim=0.7992, mcc=0.7979, min_class_dice=0.7869, coverage=[75]/88 samples
[2026-06-19 21:33:27] INFO segtask_v1.trainer.trainer: Epoch 801/1000 | LR=9.99e-04 | loss=0.2776 | val_dice=0.7869 | best=0.8292 (ep441) | 08:03:58 | L_main=0.1424 L_aux_1=0.1250(w=0.5) L_aux_2=0.1454(w=0.5)
[2026-06-19 21:33:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 801): 12688.6 MiB
[2026-06-19 21:34:29] INFO segtask_v1.trainer.validation:   Val: loss=0.3315, pooled_mean_dice=0.7736, per_class=['0.7736'], iou=0.6307, recall=0.9841, precision=0.6372, vol_sim=0.7861, mcc=0.7864, min_class_dice=0.7736, coverage=[76]/88 samples
[2026-06-19 21:34:29] INFO segtask_v1.trainer.trainer: Epoch 802/1000 | LR=9.99e-04 | loss=0.2961 | val_dice=0.7736 | best=0.8292 (ep441) | 08:05:00 | L_main=0.1532 L_aux_1=0.1350(w=0.5) L_aux_2=0.1509(w=0.5)
[2026-06-19 21:34:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 802): 12688.6 MiB
[2026-06-19 21:35:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3086, pooled_mean_dice=0.7944, per_class=['0.7944'], iou=0.6589, recall=0.9862, precision=0.6650, vol_sim=0.8054, mcc=0.8045, min_class_dice=0.7944, coverage=[76]/88 samples
[2026-06-19 21:35:31] INFO segtask_v1.trainer.trainer: Epoch 803/1000 | LR=9.99e-04 | loss=0.2882 | val_dice=0.7944 | best=0.8292 (ep441) | 08:06:03 | L_main=0.1453 L_aux_1=0.1363(w=0.5) L_aux_2=0.1495(w=0.5)
[2026-06-19 21:35:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 803): 12688.6 MiB
[2026-06-19 21:36:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3599, pooled_mean_dice=0.7794, per_class=['0.7794'], iou=0.6385, recall=0.9832, precision=0.6456, vol_sim=0.7927, mcc=0.7911, min_class_dice=0.7794, coverage=[79]/88 samples
[2026-06-19 21:36:34] INFO segtask_v1.trainer.trainer: Epoch 804/1000 | LR=9.99e-04 | loss=0.2712 | val_dice=0.7794 | best=0.8292 (ep441) | 08:07:05 | L_main=0.1373 L_aux_1=0.1299(w=0.5) L_aux_2=0.1379(w=0.5)
[2026-06-19 21:36:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 804): 12688.6 MiB
[2026-06-19 21:37:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3274, pooled_mean_dice=0.7864, per_class=['0.7864'], iou=0.6480, recall=0.9827, precision=0.6555, vol_sim=0.8003, mcc=0.7978, min_class_dice=0.7864, coverage=[75]/88 samples
[2026-06-19 21:37:37] INFO segtask_v1.trainer.trainer: Epoch 805/1000 | LR=9.98e-04 | loss=0.2735 | val_dice=0.7864 | best=0.8292 (ep441) | 08:08:08 | L_main=0.1405 L_aux_1=0.1283(w=0.5) L_aux_2=0.1376(w=0.5)
[2026-06-19 21:37:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 805): 12688.6 MiB
[2026-06-19 21:38:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3344, pooled_mean_dice=0.7933, per_class=['0.7933'], iou=0.6574, recall=0.9834, precision=0.6648, vol_sim=0.8067, mcc=0.8035, min_class_dice=0.7933, coverage=[79]/88 samples
[2026-06-19 21:38:40] INFO segtask_v1.trainer.trainer: Epoch 806/1000 | LR=9.98e-04 | loss=0.2504 | val_dice=0.7933 | best=0.8292 (ep441) | 08:09:11 | L_main=0.1294 L_aux_1=0.1149(w=0.5) L_aux_2=0.1271(w=0.5)
[2026-06-19 21:38:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 806): 12688.6 MiB
[2026-06-19 21:39:43] INFO segtask_v1.trainer.validation:   Val: loss=0.3259, pooled_mean_dice=0.7833, per_class=['0.7833'], iou=0.6438, recall=0.9854, precision=0.6500, vol_sim=0.7950, mcc=0.7948, min_class_dice=0.7833, coverage=[75]/88 samples
[2026-06-19 21:39:43] INFO segtask_v1.trainer.trainer: Epoch 807/1000 | LR=9.98e-04 | loss=0.2987 | val_dice=0.7833 | best=0.8292 (ep441) | 08:10:14 | L_main=0.1562 L_aux_1=0.1353(w=0.5) L_aux_2=0.1497(w=0.5)
[2026-06-19 21:39:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 807): 12688.6 MiB
[2026-06-19 21:40:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3157, pooled_mean_dice=0.7770, per_class=['0.7770'], iou=0.6353, recall=0.9845, precision=0.6417, vol_sim=0.7892, mcc=0.7895, min_class_dice=0.7770, coverage=[75]/88 samples
[2026-06-19 21:40:46] INFO segtask_v1.trainer.trainer: Epoch 808/1000 | LR=9.97e-04 | loss=0.3482 | val_dice=0.7770 | best=0.8292 (ep441) | 08:11:17 | L_main=0.1743 L_aux_1=0.1652(w=0.5) L_aux_2=0.1826(w=0.5)
[2026-06-19 21:40:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 808): 12688.6 MiB
[2026-06-19 21:41:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3157, pooled_mean_dice=0.7891, per_class=['0.7891'], iou=0.6517, recall=0.9858, precision=0.6579, vol_sim=0.8005, mcc=0.7992, min_class_dice=0.7891, coverage=[78]/88 samples
[2026-06-19 21:41:49] INFO segtask_v1.trainer.trainer: Epoch 809/1000 | LR=9.97e-04 | loss=0.2987 | val_dice=0.7891 | best=0.8292 (ep441) | 08:12:20 | L_main=0.1536 L_aux_1=0.1401(w=0.5) L_aux_2=0.1502(w=0.5)
[2026-06-19 21:41:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 809): 12688.6 MiB
[2026-06-19 21:42:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2898, pooled_mean_dice=0.7872, per_class=['0.7872'], iou=0.6490, recall=0.9849, precision=0.6556, vol_sim=0.7993, mcc=0.7985, min_class_dice=0.7872, coverage=[73]/88 samples
[2026-06-19 21:42:53] INFO segtask_v1.trainer.trainer: Epoch 810/1000 | LR=9.96e-04 | loss=0.2692 | val_dice=0.7872 | best=0.8292 (ep441) | 08:13:24 | L_main=0.1375 L_aux_1=0.1251(w=0.5) L_aux_2=0.1383(w=0.5)
[2026-06-19 21:42:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 810): 12688.6 MiB
[2026-06-19 21:43:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3245, pooled_mean_dice=0.7762, per_class=['0.7762'], iou=0.6342, recall=0.9840, precision=0.6409, vol_sim=0.7888, mcc=0.7886, min_class_dice=0.7762, coverage=[78]/88 samples
[2026-06-19 21:43:55] INFO segtask_v1.trainer.trainer: Epoch 811/1000 | LR=9.96e-04 | loss=0.2613 | val_dice=0.7762 | best=0.8292 (ep441) | 08:14:27 | L_main=0.1364 L_aux_1=0.1178(w=0.5) L_aux_2=0.1320(w=0.5)
[2026-06-19 21:43:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 811): 12688.6 MiB
[2026-06-19 21:44:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2806, pooled_mean_dice=0.8072, per_class=['0.8072'], iou=0.6767, recall=0.9820, precision=0.6852, vol_sim=0.8219, mcc=0.8146, min_class_dice=0.8072, coverage=[74]/88 samples
[2026-06-19 21:44:59] INFO segtask_v1.trainer.trainer: Epoch 812/1000 | LR=9.95e-04 | loss=0.2897 | val_dice=0.8072 | best=0.8292 (ep441) | 08:15:30 | L_main=0.1511 L_aux_1=0.1284(w=0.5) L_aux_2=0.1488(w=0.5)
[2026-06-19 21:44:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 812): 12688.6 MiB
[2026-06-19 21:46:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2971, pooled_mean_dice=0.7827, per_class=['0.7827'], iou=0.6429, recall=0.9846, precision=0.6494, vol_sim=0.7949, mcc=0.7938, min_class_dice=0.7827, coverage=[77]/88 samples
[2026-06-19 21:46:02] INFO segtask_v1.trainer.trainer: Epoch 813/1000 | LR=9.95e-04 | loss=0.2629 | val_dice=0.7827 | best=0.8292 (ep441) | 08:16:33 | L_main=0.1359 L_aux_1=0.1236(w=0.5) L_aux_2=0.1304(w=0.5)
[2026-06-19 21:46:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 813): 12688.6 MiB
[2026-06-19 21:47:03] INFO segtask_v1.trainer.validation:   Val: loss=0.3176, pooled_mean_dice=0.7767, per_class=['0.7767'], iou=0.6349, recall=0.9815, precision=0.6426, vol_sim=0.7913, mcc=0.7889, min_class_dice=0.7767, coverage=[73]/88 samples
[2026-06-19 21:47:03] INFO segtask_v1.trainer.trainer: Epoch 814/1000 | LR=9.94e-04 | loss=0.2671 | val_dice=0.7767 | best=0.8292 (ep441) | 08:17:34 | L_main=0.1364 L_aux_1=0.1250(w=0.5) L_aux_2=0.1363(w=0.5)
[2026-06-19 21:47:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 814): 12688.6 MiB
[2026-06-19 21:48:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2789, pooled_mean_dice=0.7860, per_class=['0.7860'], iou=0.6474, recall=0.9854, precision=0.6536, vol_sim=0.7976, mcc=0.7970, min_class_dice=0.7860, coverage=[71]/88 samples
[2026-06-19 21:48:06] INFO segtask_v1.trainer.trainer: Epoch 815/1000 | LR=9.94e-04 | loss=0.2566 | val_dice=0.7860 | best=0.8292 (ep441) | 08:18:38 | L_main=0.1306 L_aux_1=0.1153(w=0.5) L_aux_2=0.1369(w=0.5)
[2026-06-19 21:48:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 815): 12688.6 MiB
[2026-06-19 21:49:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3499, pooled_mean_dice=0.7708, per_class=['0.7708'], iou=0.6270, recall=0.9854, precision=0.6329, vol_sim=0.7822, mcc=0.7848, min_class_dice=0.7708, coverage=[77]/88 samples
[2026-06-19 21:49:09] INFO segtask_v1.trainer.trainer: Epoch 816/1000 | LR=9.93e-04 | loss=0.2189 | val_dice=0.7708 | best=0.8292 (ep441) | 08:19:40 | L_main=0.1165 L_aux_1=0.0969(w=0.5) L_aux_2=0.1079(w=0.5)
[2026-06-19 21:49:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 816): 12688.6 MiB
[2026-06-19 21:50:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3402, pooled_mean_dice=0.7892, per_class=['0.7892'], iou=0.6517, recall=0.9851, precision=0.6582, vol_sim=0.8011, mcc=0.7998, min_class_dice=0.7892, coverage=[77]/88 samples
[2026-06-19 21:50:11] INFO segtask_v1.trainer.trainer: Epoch 817/1000 | LR=9.92e-04 | loss=0.2552 | val_dice=0.7892 | best=0.8292 (ep441) | 08:20:43 | L_main=0.1258 L_aux_1=0.1233(w=0.5) L_aux_2=0.1355(w=0.5)
[2026-06-19 21:50:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 817): 12688.6 MiB
[2026-06-19 21:51:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3008, pooled_mean_dice=0.7909, per_class=['0.7909'], iou=0.6541, recall=0.9864, precision=0.6601, vol_sim=0.8018, mcc=0.8012, min_class_dice=0.7909, coverage=[78]/88 samples
[2026-06-19 21:51:13] INFO segtask_v1.trainer.trainer: Epoch 818/1000 | LR=9.92e-04 | loss=0.3431 | val_dice=0.7909 | best=0.8292 (ep441) | 08:21:45 | L_main=0.1704 L_aux_1=0.1622(w=0.5) L_aux_2=0.1832(w=0.5)
[2026-06-19 21:51:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 818): 12688.6 MiB
[2026-06-19 21:52:15] INFO segtask_v1.trainer.validation:   Val: loss=0.3573, pooled_mean_dice=0.7718, per_class=['0.7718'], iou=0.6285, recall=0.9845, precision=0.6347, vol_sim=0.7840, mcc=0.7857, min_class_dice=0.7718, coverage=[81]/88 samples
[2026-06-19 21:52:15] INFO segtask_v1.trainer.trainer: Epoch 819/1000 | LR=9.91e-04 | loss=0.3383 | val_dice=0.7718 | best=0.8292 (ep441) | 08:22:46 | L_main=0.1704 L_aux_1=0.1565(w=0.5) L_aux_2=0.1794(w=0.5)
[2026-06-19 21:52:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 819): 12688.6 MiB
[2026-06-19 21:53:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2943, pooled_mean_dice=0.7885, per_class=['0.7885'], iou=0.6508, recall=0.9883, precision=0.6559, vol_sim=0.7978, mcc=0.7994, min_class_dice=0.7885, coverage=[73]/88 samples
[2026-06-19 21:53:18] INFO segtask_v1.trainer.trainer: Epoch 820/1000 | LR=9.90e-04 | loss=0.2937 | val_dice=0.7885 | best=0.8292 (ep441) | 08:23:49 | L_main=0.1487 L_aux_1=0.1374(w=0.5) L_aux_2=0.1526(w=0.5)
[2026-06-19 21:53:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 820): 12688.6 MiB
[2026-06-19 21:54:22] INFO segtask_v1.trainer.validation:   Val: loss=0.3193, pooled_mean_dice=0.7853, per_class=['0.7853'], iou=0.6465, recall=0.9837, precision=0.6535, vol_sim=0.7983, mcc=0.7967, min_class_dice=0.7853, coverage=[76]/88 samples
[2026-06-19 21:54:22] INFO segtask_v1.trainer.trainer: Epoch 821/1000 | LR=9.89e-04 | loss=0.2659 | val_dice=0.7853 | best=0.8292 (ep441) | 08:24:54 | L_main=0.1350 L_aux_1=0.1228(w=0.5) L_aux_2=0.1390(w=0.5)
[2026-06-19 21:54:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 821): 12688.6 MiB
[2026-06-19 21:55:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3541, pooled_mean_dice=0.7597, per_class=['0.7597'], iou=0.6125, recall=0.9839, precision=0.6187, vol_sim=0.7721, mcc=0.7759, min_class_dice=0.7597, coverage=[74]/88 samples
[2026-06-19 21:55:26] INFO segtask_v1.trainer.trainer: Epoch 822/1000 | LR=9.89e-04 | loss=0.2648 | val_dice=0.7597 | best=0.8292 (ep441) | 08:25:57 | L_main=0.1380 L_aux_1=0.1219(w=0.5) L_aux_2=0.1318(w=0.5)
[2026-06-19 21:55:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 822): 12688.6 MiB
[2026-06-19 21:56:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3314, pooled_mean_dice=0.7824, per_class=['0.7824'], iou=0.6426, recall=0.9857, precision=0.6486, vol_sim=0.7938, mcc=0.7942, min_class_dice=0.7824, coverage=[71]/88 samples
[2026-06-19 21:56:28] INFO segtask_v1.trainer.trainer: Epoch 823/1000 | LR=9.88e-04 | loss=0.2522 | val_dice=0.7824 | best=0.8292 (ep441) | 08:26:59 | L_main=0.1307 L_aux_1=0.1148(w=0.5) L_aux_2=0.1282(w=0.5)
[2026-06-19 21:56:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 823): 12688.6 MiB
[2026-06-19 21:57:30] INFO segtask_v1.trainer.validation:   Val: loss=0.3512, pooled_mean_dice=0.7849, per_class=['0.7849'], iou=0.6460, recall=0.9847, precision=0.6526, vol_sim=0.7971, mcc=0.7963, min_class_dice=0.7849, coverage=[77]/88 samples
[2026-06-19 21:57:30] INFO segtask_v1.trainer.trainer: Epoch 824/1000 | LR=9.87e-04 | loss=0.2426 | val_dice=0.7849 | best=0.8292 (ep441) | 08:28:02 | L_main=0.1247 L_aux_1=0.1100(w=0.5) L_aux_2=0.1258(w=0.5)
[2026-06-19 21:57:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 824): 12688.6 MiB
[2026-06-19 21:58:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3446, pooled_mean_dice=0.7813, per_class=['0.7813'], iou=0.6411, recall=0.9845, precision=0.6476, vol_sim=0.7936, mcc=0.7941, min_class_dice=0.7813, coverage=[68]/88 samples
[2026-06-19 21:58:32] INFO segtask_v1.trainer.trainer: Epoch 825/1000 | LR=9.86e-04 | loss=0.2430 | val_dice=0.7813 | best=0.8292 (ep441) | 08:29:03 | L_main=0.1243 L_aux_1=0.1094(w=0.5) L_aux_2=0.1280(w=0.5)
[2026-06-19 21:58:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 825): 12688.6 MiB
[2026-06-19 21:59:35] INFO segtask_v1.trainer.validation:   Val: loss=0.3448, pooled_mean_dice=0.7767, per_class=['0.7767'], iou=0.6349, recall=0.9881, precision=0.6398, vol_sim=0.7861, mcc=0.7891, min_class_dice=0.7767, coverage=[77]/88 samples
[2026-06-19 21:59:35] INFO segtask_v1.trainer.trainer: Epoch 826/1000 | LR=9.85e-04 | loss=0.3273 | val_dice=0.7767 | best=0.8292 (ep441) | 08:30:06 | L_main=0.1676 L_aux_1=0.1514(w=0.5) L_aux_2=0.1678(w=0.5)
[2026-06-19 21:59:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 826): 12688.6 MiB
[2026-06-19 22:00:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3393, pooled_mean_dice=0.7406, per_class=['0.7406'], iou=0.5880, recall=0.9855, precision=0.5932, vol_sim=0.7515, mcc=0.7602, min_class_dice=0.7406, coverage=[73]/88 samples
[2026-06-19 22:00:37] INFO segtask_v1.trainer.trainer: Epoch 827/1000 | LR=9.84e-04 | loss=0.3056 | val_dice=0.7406 | best=0.8292 (ep441) | 08:31:08 | L_main=0.1569 L_aux_1=0.1420(w=0.5) L_aux_2=0.1553(w=0.5)
[2026-06-19 22:00:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 827): 12688.6 MiB
[2026-06-19 22:01:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3344, pooled_mean_dice=0.7776, per_class=['0.7776'], iou=0.6362, recall=0.9861, precision=0.6419, vol_sim=0.7886, mcc=0.7902, min_class_dice=0.7776, coverage=[72]/88 samples
[2026-06-19 22:01:39] INFO segtask_v1.trainer.trainer: Epoch 828/1000 | LR=9.83e-04 | loss=0.2876 | val_dice=0.7776 | best=0.8292 (ep441) | 08:32:10 | L_main=0.1446 L_aux_1=0.1316(w=0.5) L_aux_2=0.1543(w=0.5)
[2026-06-19 22:01:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 828): 12688.6 MiB
[2026-06-19 22:02:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3421, pooled_mean_dice=0.7847, per_class=['0.7847'], iou=0.6457, recall=0.9850, precision=0.6521, vol_sim=0.7967, mcc=0.7964, min_class_dice=0.7847, coverage=[75]/88 samples
[2026-06-19 22:02:41] INFO segtask_v1.trainer.trainer: Epoch 829/1000 | LR=9.82e-04 | loss=0.2543 | val_dice=0.7847 | best=0.8292 (ep441) | 08:33:12 | L_main=0.1258 L_aux_1=0.1183(w=0.5) L_aux_2=0.1389(w=0.5)
[2026-06-19 22:02:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 829): 12688.6 MiB
[2026-06-19 22:03:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3021, pooled_mean_dice=0.7868, per_class=['0.7868'], iou=0.6485, recall=0.9851, precision=0.6549, vol_sim=0.7987, mcc=0.7980, min_class_dice=0.7868, coverage=[74]/88 samples
[2026-06-19 22:03:44] INFO segtask_v1.trainer.trainer: Epoch 830/1000 | LR=9.81e-04 | loss=0.2664 | val_dice=0.7868 | best=0.8292 (ep441) | 08:34:16 | L_main=0.1341 L_aux_1=0.1252(w=0.5) L_aux_2=0.1393(w=0.5)
[2026-06-19 22:03:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 830): 12688.6 MiB
[2026-06-19 22:04:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3118, pooled_mean_dice=0.7740, per_class=['0.7740'], iou=0.6313, recall=0.9849, precision=0.6375, vol_sim=0.7858, mcc=0.7860, min_class_dice=0.7740, coverage=[77]/88 samples
[2026-06-19 22:04:48] INFO segtask_v1.trainer.trainer: Epoch 831/1000 | LR=9.80e-04 | loss=0.2634 | val_dice=0.7740 | best=0.8292 (ep441) | 08:35:19 | L_main=0.1303 L_aux_1=0.1239(w=0.5) L_aux_2=0.1423(w=0.5)
[2026-06-19 22:04:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 831): 12688.6 MiB
[2026-06-19 22:05:51] INFO segtask_v1.trainer.validation:   Val: loss=0.3251, pooled_mean_dice=0.7911, per_class=['0.7911'], iou=0.6544, recall=0.9860, precision=0.6606, vol_sim=0.8024, mcc=0.8015, min_class_dice=0.7911, coverage=[80]/88 samples
[2026-06-19 22:05:51] INFO segtask_v1.trainer.trainer: Epoch 832/1000 | LR=9.79e-04 | loss=0.2727 | val_dice=0.7911 | best=0.8292 (ep441) | 08:36:22 | L_main=0.1387 L_aux_1=0.1279(w=0.5) L_aux_2=0.1401(w=0.5)
[2026-06-19 22:05:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 832): 12688.6 MiB
[2026-06-19 22:06:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3503, pooled_mean_dice=0.7601, per_class=['0.7601'], iou=0.6130, recall=0.9815, precision=0.6202, vol_sim=0.7744, mcc=0.7752, min_class_dice=0.7601, coverage=[77]/88 samples
[2026-06-19 22:06:54] INFO segtask_v1.trainer.trainer: Epoch 833/1000 | LR=9.77e-04 | loss=0.2983 | val_dice=0.7601 | best=0.8292 (ep441) | 08:37:26 | L_main=0.1551 L_aux_1=0.1378(w=0.5) L_aux_2=0.1486(w=0.5)
[2026-06-19 22:06:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 833): 12688.6 MiB
[2026-06-19 22:07:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2892, pooled_mean_dice=0.7871, per_class=['0.7871'], iou=0.6489, recall=0.9863, precision=0.6548, vol_sim=0.7980, mcc=0.7991, min_class_dice=0.7871, coverage=[72]/88 samples
[2026-06-19 22:07:57] INFO segtask_v1.trainer.trainer: Epoch 834/1000 | LR=9.76e-04 | loss=0.2837 | val_dice=0.7871 | best=0.8292 (ep441) | 08:38:28 | L_main=0.1429 L_aux_1=0.1326(w=0.5) L_aux_2=0.1489(w=0.5)
[2026-06-19 22:07:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 834): 12688.6 MiB
[2026-06-19 22:09:00] INFO segtask_v1.trainer.validation:   Val: loss=0.3318, pooled_mean_dice=0.7860, per_class=['0.7860'], iou=0.6474, recall=0.9846, precision=0.6540, vol_sim=0.7983, mcc=0.7970, min_class_dice=0.7860, coverage=[78]/88 samples
[2026-06-19 22:09:00] INFO segtask_v1.trainer.trainer: Epoch 835/1000 | LR=9.75e-04 | loss=0.3676 | val_dice=0.7860 | best=0.8292 (ep441) | 08:39:31 | L_main=0.1841 L_aux_1=0.1763(w=0.5) L_aux_2=0.1906(w=0.5)
[2026-06-19 22:09:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 835): 12688.6 MiB
[2026-06-19 22:10:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2857, pooled_mean_dice=0.7972, per_class=['0.7972'], iou=0.6629, recall=0.9832, precision=0.6705, vol_sim=0.8109, mcc=0.8067, min_class_dice=0.7972, coverage=[74]/88 samples
[2026-06-19 22:10:03] INFO segtask_v1.trainer.trainer: Epoch 836/1000 | LR=9.74e-04 | loss=0.2716 | val_dice=0.7972 | best=0.8292 (ep441) | 08:40:34 | L_main=0.1352 L_aux_1=0.1296(w=0.5) L_aux_2=0.1433(w=0.5)
[2026-06-19 22:10:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 836): 12688.6 MiB
[2026-06-19 22:11:06] INFO segtask_v1.trainer.validation:   Val: loss=0.3089, pooled_mean_dice=0.7990, per_class=['0.7990'], iou=0.6653, recall=0.9874, precision=0.6710, vol_sim=0.8092, mcc=0.8084, min_class_dice=0.7990, coverage=[72]/88 samples
[2026-06-19 22:11:06] INFO segtask_v1.trainer.trainer: Epoch 837/1000 | LR=9.72e-04 | loss=0.2877 | val_dice=0.7990 | best=0.8292 (ep441) | 08:41:37 | L_main=0.1466 L_aux_1=0.1346(w=0.5) L_aux_2=0.1476(w=0.5)
[2026-06-19 22:11:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 837): 12688.6 MiB
[2026-06-19 22:12:10] INFO segtask_v1.trainer.validation:   Val: loss=0.3225, pooled_mean_dice=0.8071, per_class=['0.8071'], iou=0.6765, recall=0.9828, precision=0.6846, vol_sim=0.8212, mcc=0.8162, min_class_dice=0.8071, coverage=[76]/88 samples
[2026-06-19 22:12:10] INFO segtask_v1.trainer.trainer: Epoch 838/1000 | LR=9.71e-04 | loss=0.2492 | val_dice=0.8071 | best=0.8292 (ep441) | 08:42:41 | L_main=0.1287 L_aux_1=0.1129(w=0.5) L_aux_2=0.1282(w=0.5)
[2026-06-19 22:12:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 838): 12688.6 MiB
[2026-06-19 22:13:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3114, pooled_mean_dice=0.7870, per_class=['0.7870'], iou=0.6488, recall=0.9833, precision=0.6560, vol_sim=0.8004, mcc=0.7977, min_class_dice=0.7870, coverage=[79]/88 samples
[2026-06-19 22:13:13] INFO segtask_v1.trainer.trainer: Epoch 839/1000 | LR=9.70e-04 | loss=0.2384 | val_dice=0.7870 | best=0.8292 (ep441) | 08:43:44 | L_main=0.1212 L_aux_1=0.1101(w=0.5) L_aux_2=0.1243(w=0.5)
[2026-06-19 22:13:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 839): 12688.6 MiB
[2026-06-19 22:14:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2924, pooled_mean_dice=0.7827, per_class=['0.7827'], iou=0.6430, recall=0.9826, precision=0.6504, vol_sim=0.7966, mcc=0.7936, min_class_dice=0.7827, coverage=[74]/88 samples
[2026-06-19 22:14:16] INFO segtask_v1.trainer.trainer: Epoch 840/1000 | LR=9.68e-04 | loss=0.2394 | val_dice=0.7827 | best=0.8292 (ep441) | 08:44:47 | L_main=0.1260 L_aux_1=0.1095(w=0.5) L_aux_2=0.1173(w=0.5)
[2026-06-19 22:14:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 840): 12688.6 MiB
[2026-06-19 22:15:18] INFO segtask_v1.trainer.validation:   Val: loss=0.3185, pooled_mean_dice=0.7875, per_class=['0.7875'], iou=0.6495, recall=0.9818, precision=0.6574, vol_sim=0.8021, mcc=0.7975, min_class_dice=0.7875, coverage=[80]/88 samples
[2026-06-19 22:15:18] INFO segtask_v1.trainer.trainer: Epoch 841/1000 | LR=9.67e-04 | loss=0.2836 | val_dice=0.7875 | best=0.8292 (ep441) | 08:45:49 | L_main=0.1410 L_aux_1=0.1365(w=0.5) L_aux_2=0.1487(w=0.5)
[2026-06-19 22:15:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 841): 12688.6 MiB
[2026-06-19 22:16:21] INFO segtask_v1.trainer.validation:   Val: loss=0.3057, pooled_mean_dice=0.7798, per_class=['0.7798'], iou=0.6391, recall=0.9818, precision=0.6468, vol_sim=0.7942, mcc=0.7919, min_class_dice=0.7798, coverage=[74]/88 samples
[2026-06-19 22:16:21] INFO segtask_v1.trainer.trainer: Epoch 842/1000 | LR=9.66e-04 | loss=0.3058 | val_dice=0.7798 | best=0.8292 (ep441) | 08:46:52 | L_main=0.1515 L_aux_1=0.1439(w=0.5) L_aux_2=0.1646(w=0.5)
[2026-06-19 22:16:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 842): 12688.6 MiB
[2026-06-19 22:17:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3316, pooled_mean_dice=0.7918, per_class=['0.7918'], iou=0.6553, recall=0.9838, precision=0.6625, vol_sim=0.8048, mcc=0.8020, min_class_dice=0.7918, coverage=[79]/88 samples
[2026-06-19 22:17:23] INFO segtask_v1.trainer.trainer: Epoch 843/1000 | LR=9.64e-04 | loss=0.2737 | val_dice=0.7918 | best=0.8292 (ep441) | 08:47:55 | L_main=0.1404 L_aux_1=0.1249(w=0.5) L_aux_2=0.1418(w=0.5)
[2026-06-19 22:17:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 843): 12688.6 MiB
[2026-06-19 22:18:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2858, pooled_mean_dice=0.7981, per_class=['0.7981'], iou=0.6641, recall=0.9831, precision=0.6718, vol_sim=0.8119, mcc=0.8073, min_class_dice=0.7981, coverage=[72]/88 samples
[2026-06-19 22:18:26] INFO segtask_v1.trainer.trainer: Epoch 844/1000 | LR=9.63e-04 | loss=0.2546 | val_dice=0.7981 | best=0.8292 (ep441) | 08:48:57 | L_main=0.1329 L_aux_1=0.1137(w=0.5) L_aux_2=0.1298(w=0.5)
[2026-06-19 22:18:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 844): 12688.6 MiB
[2026-06-19 22:19:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3261, pooled_mean_dice=0.7979, per_class=['0.7979'], iou=0.6637, recall=0.9804, precision=0.6726, vol_sim=0.8138, mcc=0.8068, min_class_dice=0.7979, coverage=[80]/88 samples
[2026-06-19 22:19:28] INFO segtask_v1.trainer.trainer: Epoch 845/1000 | LR=9.61e-04 | loss=0.2759 | val_dice=0.7979 | best=0.8292 (ep441) | 08:49:59 | L_main=0.1426 L_aux_1=0.1248(w=0.5) L_aux_2=0.1418(w=0.5)
[2026-06-19 22:19:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 845): 12688.6 MiB
[2026-06-19 22:20:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3275, pooled_mean_dice=0.7788, per_class=['0.7788'], iou=0.6377, recall=0.9851, precision=0.6439, vol_sim=0.7905, mcc=0.7904, min_class_dice=0.7788, coverage=[78]/88 samples
[2026-06-19 22:20:31] INFO segtask_v1.trainer.trainer: Epoch 846/1000 | LR=9.59e-04 | loss=0.2801 | val_dice=0.7788 | best=0.8292 (ep441) | 08:51:02 | L_main=0.1442 L_aux_1=0.1312(w=0.5) L_aux_2=0.1404(w=0.5)
[2026-06-19 22:20:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 846): 12688.6 MiB
[2026-06-19 22:21:33] INFO segtask_v1.trainer.validation:   Val: loss=0.3174, pooled_mean_dice=0.7955, per_class=['0.7955'], iou=0.6604, recall=0.9852, precision=0.6671, vol_sim=0.8075, mcc=0.8052, min_class_dice=0.7955, coverage=[75]/88 samples
[2026-06-19 22:21:33] INFO segtask_v1.trainer.trainer: Epoch 847/1000 | LR=9.58e-04 | loss=0.2467 | val_dice=0.7955 | best=0.8292 (ep441) | 08:52:05 | L_main=0.1285 L_aux_1=0.1129(w=0.5) L_aux_2=0.1235(w=0.5)
[2026-06-19 22:21:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 847): 12688.6 MiB
[2026-06-19 22:22:35] INFO segtask_v1.trainer.validation:   Val: loss=0.3428, pooled_mean_dice=0.7826, per_class=['0.7826'], iou=0.6429, recall=0.9835, precision=0.6499, vol_sim=0.7957, mcc=0.7938, min_class_dice=0.7826, coverage=[80]/88 samples
[2026-06-19 22:22:35] INFO segtask_v1.trainer.trainer: Epoch 848/1000 | LR=9.56e-04 | loss=0.2513 | val_dice=0.7826 | best=0.8292 (ep441) | 08:53:06 | L_main=0.1298 L_aux_1=0.1171(w=0.5) L_aux_2=0.1260(w=0.5)
[2026-06-19 22:22:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 848): 12688.6 MiB
[2026-06-19 22:23:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3223, pooled_mean_dice=0.8010, per_class=['0.8010'], iou=0.6680, recall=0.9820, precision=0.6763, vol_sim=0.8157, mcc=0.8096, min_class_dice=0.8010, coverage=[77]/88 samples
[2026-06-19 22:23:37] INFO segtask_v1.trainer.trainer: Epoch 849/1000 | LR=9.55e-04 | loss=0.2472 | val_dice=0.8010 | best=0.8292 (ep441) | 08:54:08 | L_main=0.1266 L_aux_1=0.1125(w=0.5) L_aux_2=0.1287(w=0.5)
[2026-06-19 22:23:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 849): 12688.6 MiB
[2026-06-19 22:24:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3356, pooled_mean_dice=0.7939, per_class=['0.7939'], iou=0.6583, recall=0.9867, precision=0.6642, vol_sim=0.8046, mcc=0.8042, min_class_dice=0.7939, coverage=[73]/88 samples
[2026-06-19 22:24:39] INFO segtask_v1.trainer.trainer: Epoch 850/1000 | LR=9.53e-04 | loss=0.2400 | val_dice=0.7939 | best=0.8292 (ep441) | 08:55:11 | L_main=0.1218 L_aux_1=0.1120(w=0.5) L_aux_2=0.1244(w=0.5)
[2026-06-19 22:24:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 850): 12688.6 MiB
[2026-06-19 22:25:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3328, pooled_mean_dice=0.7646, per_class=['0.7646'], iou=0.6189, recall=0.9805, precision=0.6266, vol_sim=0.7797, mcc=0.7795, min_class_dice=0.7646, coverage=[76]/88 samples
[2026-06-19 22:25:44] INFO segtask_v1.trainer.trainer: Epoch 851/1000 | LR=9.51e-04 | loss=0.2975 | val_dice=0.7646 | best=0.8292 (ep441) | 08:56:15 | L_main=0.1496 L_aux_1=0.1420(w=0.5) L_aux_2=0.1537(w=0.5)
[2026-06-19 22:25:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 851): 12688.6 MiB
[2026-06-19 22:26:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2896, pooled_mean_dice=0.8070, per_class=['0.8070'], iou=0.6764, recall=0.9857, precision=0.6831, vol_sim=0.8187, mcc=0.8148, min_class_dice=0.8070, coverage=[75]/88 samples
[2026-06-19 22:26:47] INFO segtask_v1.trainer.trainer: Epoch 852/1000 | LR=9.50e-04 | loss=0.2668 | val_dice=0.8070 | best=0.8292 (ep441) | 08:57:18 | L_main=0.1351 L_aux_1=0.1237(w=0.5) L_aux_2=0.1398(w=0.5)
[2026-06-19 22:26:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 852): 12688.6 MiB
[2026-06-19 22:27:50] INFO segtask_v1.trainer.validation:   Val: loss=0.3177, pooled_mean_dice=0.7873, per_class=['0.7873'], iou=0.6492, recall=0.9866, precision=0.6550, vol_sim=0.7980, mcc=0.7991, min_class_dice=0.7873, coverage=[75]/88 samples
[2026-06-19 22:27:50] INFO segtask_v1.trainer.trainer: Epoch 853/1000 | LR=9.48e-04 | loss=0.2854 | val_dice=0.7873 | best=0.8292 (ep441) | 08:58:21 | L_main=0.1475 L_aux_1=0.1326(w=0.5) L_aux_2=0.1432(w=0.5)
[2026-06-19 22:27:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 853): 12688.6 MiB
[2026-06-19 22:28:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2980, pooled_mean_dice=0.7882, per_class=['0.7882'], iou=0.6504, recall=0.9856, precision=0.6567, vol_sim=0.7997, mcc=0.7998, min_class_dice=0.7882, coverage=[75]/88 samples
[2026-06-19 22:28:53] INFO segtask_v1.trainer.trainer: Epoch 854/1000 | LR=9.46e-04 | loss=0.2580 | val_dice=0.7882 | best=0.8292 (ep441) | 08:59:24 | L_main=0.1322 L_aux_1=0.1176(w=0.5) L_aux_2=0.1341(w=0.5)
[2026-06-19 22:28:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 854): 12688.6 MiB
[2026-06-19 22:29:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3015, pooled_mean_dice=0.7921, per_class=['0.7921'], iou=0.6558, recall=0.9839, precision=0.6629, vol_sim=0.8050, mcc=0.8025, min_class_dice=0.7921, coverage=[73]/88 samples
[2026-06-19 22:29:55] INFO segtask_v1.trainer.trainer: Epoch 855/1000 | LR=9.44e-04 | loss=0.2742 | val_dice=0.7921 | best=0.8292 (ep441) | 09:00:27 | L_main=0.1403 L_aux_1=0.1249(w=0.5) L_aux_2=0.1429(w=0.5)
[2026-06-19 22:29:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 855): 12688.6 MiB
[2026-06-19 22:30:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3429, pooled_mean_dice=0.7948, per_class=['0.7948'], iou=0.6595, recall=0.9855, precision=0.6659, vol_sim=0.8065, mcc=0.8041, min_class_dice=0.7948, coverage=[80]/88 samples
[2026-06-19 22:30:58] INFO segtask_v1.trainer.trainer: Epoch 856/1000 | LR=9.42e-04 | loss=0.2539 | val_dice=0.7948 | best=0.8292 (ep441) | 09:01:30 | L_main=0.1309 L_aux_1=0.1171(w=0.5) L_aux_2=0.1288(w=0.5)
[2026-06-19 22:30:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 856): 12688.6 MiB
[2026-06-19 22:32:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3280, pooled_mean_dice=0.8014, per_class=['0.8014'], iou=0.6686, recall=0.9842, precision=0.6759, vol_sim=0.8143, mcc=0.8106, min_class_dice=0.8014, coverage=[77]/88 samples
[2026-06-19 22:32:01] INFO segtask_v1.trainer.trainer: Epoch 857/1000 | LR=9.40e-04 | loss=0.2653 | val_dice=0.8014 | best=0.8292 (ep441) | 09:02:32 | L_main=0.1351 L_aux_1=0.1221(w=0.5) L_aux_2=0.1381(w=0.5)
[2026-06-19 22:32:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 857): 12688.6 MiB
[2026-06-19 22:33:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3156, pooled_mean_dice=0.8065, per_class=['0.8065'], iou=0.6757, recall=0.9829, precision=0.6837, vol_sim=0.8205, mcc=0.8139, min_class_dice=0.8065, coverage=[76]/88 samples
[2026-06-19 22:33:04] INFO segtask_v1.trainer.trainer: Epoch 858/1000 | LR=9.39e-04 | loss=0.3005 | val_dice=0.8065 | best=0.8292 (ep441) | 09:03:35 | L_main=0.1533 L_aux_1=0.1415(w=0.5) L_aux_2=0.1529(w=0.5)
[2026-06-19 22:33:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 858): 12688.6 MiB
[2026-06-19 22:34:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3196, pooled_mean_dice=0.7897, per_class=['0.7897'], iou=0.6524, recall=0.9852, precision=0.6589, vol_sim=0.8015, mcc=0.8007, min_class_dice=0.7897, coverage=[74]/88 samples
[2026-06-19 22:34:07] INFO segtask_v1.trainer.trainer: Epoch 859/1000 | LR=9.37e-04 | loss=0.2860 | val_dice=0.7897 | best=0.8292 (ep441) | 09:04:38 | L_main=0.1496 L_aux_1=0.1293(w=0.5) L_aux_2=0.1435(w=0.5)
[2026-06-19 22:34:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 859): 12688.6 MiB
[2026-06-19 22:35:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2990, pooled_mean_dice=0.7741, per_class=['0.7741'], iou=0.6315, recall=0.9861, precision=0.6371, vol_sim=0.7850, mcc=0.7878, min_class_dice=0.7741, coverage=[69]/88 samples
[2026-06-19 22:35:10] INFO segtask_v1.trainer.trainer: Epoch 860/1000 | LR=9.35e-04 | loss=0.2516 | val_dice=0.7741 | best=0.8292 (ep441) | 09:05:41 | L_main=0.1289 L_aux_1=0.1168(w=0.5) L_aux_2=0.1287(w=0.5)
[2026-06-19 22:35:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 860): 12688.6 MiB
[2026-06-19 22:36:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2982, pooled_mean_dice=0.7974, per_class=['0.7974'], iou=0.6630, recall=0.9851, precision=0.6697, vol_sim=0.8094, mcc=0.8064, min_class_dice=0.7974, coverage=[73]/88 samples
[2026-06-19 22:36:12] INFO segtask_v1.trainer.trainer: Epoch 861/1000 | LR=9.33e-04 | loss=0.2433 | val_dice=0.7974 | best=0.8292 (ep441) | 09:06:43 | L_main=0.1265 L_aux_1=0.1125(w=0.5) L_aux_2=0.1210(w=0.5)
[2026-06-19 22:36:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 861): 12688.6 MiB
[2026-06-19 22:37:15] INFO segtask_v1.trainer.validation:   Val: loss=0.3371, pooled_mean_dice=0.7800, per_class=['0.7800'], iou=0.6393, recall=0.9866, precision=0.6449, vol_sim=0.7905, mcc=0.7926, min_class_dice=0.7800, coverage=[77]/88 samples
[2026-06-19 22:37:15] INFO segtask_v1.trainer.trainer: Epoch 862/1000 | LR=9.31e-04 | loss=0.2484 | val_dice=0.7800 | best=0.8292 (ep441) | 09:07:46 | L_main=0.1321 L_aux_1=0.1122(w=0.5) L_aux_2=0.1203(w=0.5)
[2026-06-19 22:37:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 862): 12688.6 MiB
[2026-06-19 22:38:18] INFO segtask_v1.trainer.validation:   Val: loss=0.3281, pooled_mean_dice=0.7939, per_class=['0.7939'], iou=0.6582, recall=0.9841, precision=0.6653, vol_sim=0.8067, mcc=0.8041, min_class_dice=0.7939, coverage=[77]/88 samples
[2026-06-19 22:38:18] INFO segtask_v1.trainer.trainer: Epoch 863/1000 | LR=9.29e-04 | loss=0.2504 | val_dice=0.7939 | best=0.8292 (ep441) | 09:08:50 | L_main=0.1260 L_aux_1=0.1162(w=0.5) L_aux_2=0.1326(w=0.5)
[2026-06-19 22:38:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 863): 12688.6 MiB
[2026-06-19 22:39:22] INFO segtask_v1.trainer.validation:   Val: loss=0.3283, pooled_mean_dice=0.7858, per_class=['0.7858'], iou=0.6472, recall=0.9864, precision=0.6531, vol_sim=0.7967, mcc=0.7971, min_class_dice=0.7858, coverage=[79]/88 samples
[2026-06-19 22:39:22] INFO segtask_v1.trainer.trainer: Epoch 864/1000 | LR=9.27e-04 | loss=0.2494 | val_dice=0.7858 | best=0.8292 (ep441) | 09:09:53 | L_main=0.1249 L_aux_1=0.1160(w=0.5) L_aux_2=0.1329(w=0.5)
[2026-06-19 22:39:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 864): 12688.6 MiB
[2026-06-19 22:40:25] INFO segtask_v1.trainer.validation:   Val: loss=0.3289, pooled_mean_dice=0.7933, per_class=['0.7933'], iou=0.6574, recall=0.9874, precision=0.6629, vol_sim=0.8034, mcc=0.8033, min_class_dice=0.7933, coverage=[79]/88 samples
[2026-06-19 22:40:25] INFO segtask_v1.trainer.trainer: Epoch 865/1000 | LR=9.25e-04 | loss=0.2411 | val_dice=0.7933 | best=0.8292 (ep441) | 09:10:56 | L_main=0.1211 L_aux_1=0.1154(w=0.5) L_aux_2=0.1245(w=0.5)
[2026-06-19 22:40:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 865): 12688.6 MiB
[2026-06-19 22:41:28] INFO segtask_v1.trainer.validation:   Val: loss=0.3536, pooled_mean_dice=0.7783, per_class=['0.7783'], iou=0.6371, recall=0.9854, precision=0.6432, vol_sim=0.7899, mcc=0.7911, min_class_dice=0.7783, coverage=[80]/88 samples
[2026-06-19 22:41:28] INFO segtask_v1.trainer.trainer: Epoch 866/1000 | LR=9.22e-04 | loss=0.2616 | val_dice=0.7783 | best=0.8292 (ep441) | 09:11:59 | L_main=0.1282 L_aux_1=0.1246(w=0.5) L_aux_2=0.1421(w=0.5)
[2026-06-19 22:41:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 866): 12688.6 MiB
[2026-06-19 22:42:31] INFO segtask_v1.trainer.validation:   Val: loss=0.3248, pooled_mean_dice=0.7991, per_class=['0.7991'], iou=0.6654, recall=0.9822, precision=0.6735, vol_sim=0.8135, mcc=0.8082, min_class_dice=0.7991, coverage=[79]/88 samples
[2026-06-19 22:42:31] INFO segtask_v1.trainer.trainer: Epoch 867/1000 | LR=9.20e-04 | loss=0.2473 | val_dice=0.7991 | best=0.8292 (ep441) | 09:13:02 | L_main=0.1252 L_aux_1=0.1144(w=0.5) L_aux_2=0.1298(w=0.5)
[2026-06-19 22:42:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 867): 12688.6 MiB
[2026-06-19 22:43:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3231, pooled_mean_dice=0.8077, per_class=['0.8077'], iou=0.6774, recall=0.9844, precision=0.6847, vol_sim=0.8205, mcc=0.8151, min_class_dice=0.8077, coverage=[81]/88 samples
[2026-06-19 22:43:32] INFO segtask_v1.trainer.trainer: Epoch 868/1000 | LR=9.18e-04 | loss=0.2507 | val_dice=0.8077 | best=0.8292 (ep441) | 09:14:03 | L_main=0.1306 L_aux_1=0.1163(w=0.5) L_aux_2=0.1240(w=0.5)
[2026-06-19 22:43:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 868): 12688.6 MiB
[2026-06-19 22:44:34] INFO segtask_v1.trainer.validation:   Val: loss=0.2943, pooled_mean_dice=0.8030, per_class=['0.8030'], iou=0.6709, recall=0.9848, precision=0.6779, vol_sim=0.8155, mcc=0.8115, min_class_dice=0.8030, coverage=[75]/88 samples
[2026-06-19 22:44:34] INFO segtask_v1.trainer.trainer: Epoch 869/1000 | LR=9.16e-04 | loss=0.2381 | val_dice=0.8030 | best=0.8292 (ep441) | 09:15:06 | L_main=0.1234 L_aux_1=0.1099(w=0.5) L_aux_2=0.1195(w=0.5)
[2026-06-19 22:44:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 869): 12688.6 MiB
[2026-06-19 22:45:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3009, pooled_mean_dice=0.7887, per_class=['0.7887'], iou=0.6512, recall=0.9829, precision=0.6586, vol_sim=0.8024, mcc=0.8000, min_class_dice=0.7887, coverage=[73]/88 samples
[2026-06-19 22:45:37] INFO segtask_v1.trainer.trainer: Epoch 870/1000 | LR=9.14e-04 | loss=0.2347 | val_dice=0.7887 | best=0.8292 (ep441) | 09:16:08 | L_main=0.1211 L_aux_1=0.1075(w=0.5) L_aux_2=0.1196(w=0.5)
[2026-06-19 22:45:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 870): 12688.6 MiB
[2026-06-19 22:46:39] INFO segtask_v1.trainer.validation:   Val: loss=0.2914, pooled_mean_dice=0.7878, per_class=['0.7878'], iou=0.6499, recall=0.9866, precision=0.6557, vol_sim=0.7985, mcc=0.7994, min_class_dice=0.7878, coverage=[69]/88 samples
[2026-06-19 22:46:39] INFO segtask_v1.trainer.trainer: Epoch 871/1000 | LR=9.11e-04 | loss=0.2221 | val_dice=0.7878 | best=0.8292 (ep441) | 09:17:10 | L_main=0.1147 L_aux_1=0.0999(w=0.5) L_aux_2=0.1148(w=0.5)
[2026-06-19 22:46:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 871): 12688.6 MiB
[2026-06-19 22:47:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3140, pooled_mean_dice=0.8079, per_class=['0.8079'], iou=0.6777, recall=0.9859, precision=0.6843, vol_sim=0.8194, mcc=0.8159, min_class_dice=0.8079, coverage=[77]/88 samples
[2026-06-19 22:47:42] INFO segtask_v1.trainer.trainer: Epoch 872/1000 | LR=9.09e-04 | loss=0.2244 | val_dice=0.8079 | best=0.8292 (ep441) | 09:18:13 | L_main=0.1173 L_aux_1=0.1031(w=0.5) L_aux_2=0.1111(w=0.5)
[2026-06-19 22:47:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 872): 12688.6 MiB
[2026-06-19 22:48:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3896, pooled_mean_dice=0.7927, per_class=['0.7927'], iou=0.6566, recall=0.9862, precision=0.6627, vol_sim=0.8038, mcc=0.8036, min_class_dice=0.7927, coverage=[77]/88 samples
[2026-06-19 22:48:44] INFO segtask_v1.trainer.trainer: Epoch 873/1000 | LR=9.07e-04 | loss=0.2499 | val_dice=0.7927 | best=0.8292 (ep441) | 09:19:16 | L_main=0.1281 L_aux_1=0.1159(w=0.5) L_aux_2=0.1277(w=0.5)
[2026-06-19 22:48:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 873): 12688.6 MiB
[2026-06-19 22:49:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3196, pooled_mean_dice=0.8029, per_class=['0.8029'], iou=0.6707, recall=0.9834, precision=0.6783, vol_sim=0.8164, mcc=0.8118, min_class_dice=0.8029, coverage=[73]/88 samples
[2026-06-19 22:49:46] INFO segtask_v1.trainer.trainer: Epoch 874/1000 | LR=9.05e-04 | loss=0.2484 | val_dice=0.8029 | best=0.8292 (ep441) | 09:20:17 | L_main=0.1262 L_aux_1=0.1160(w=0.5) L_aux_2=0.1285(w=0.5)
[2026-06-19 22:49:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 874): 12688.6 MiB
[2026-06-19 22:50:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3500, pooled_mean_dice=0.8102, per_class=['0.8102'], iou=0.6810, recall=0.9840, precision=0.6887, vol_sim=0.8234, mcc=0.8185, min_class_dice=0.8102, coverage=[80]/88 samples
[2026-06-19 22:50:48] INFO segtask_v1.trainer.trainer: Epoch 875/1000 | LR=9.02e-04 | loss=0.2581 | val_dice=0.8102 | best=0.8292 (ep441) | 09:21:20 | L_main=0.1300 L_aux_1=0.1213(w=0.5) L_aux_2=0.1349(w=0.5)
[2026-06-19 22:50:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 875): 12688.6 MiB
[2026-06-19 22:51:50] INFO segtask_v1.trainer.validation:   Val: loss=0.3624, pooled_mean_dice=0.7794, per_class=['0.7794'], iou=0.6386, recall=0.9840, precision=0.6453, vol_sim=0.7921, mcc=0.7924, min_class_dice=0.7794, coverage=[77]/88 samples
[2026-06-19 22:51:50] INFO segtask_v1.trainer.trainer: Epoch 876/1000 | LR=9.00e-04 | loss=0.2404 | val_dice=0.7794 | best=0.8292 (ep441) | 09:22:22 | L_main=0.1246 L_aux_1=0.1111(w=0.5) L_aux_2=0.1206(w=0.5)
[2026-06-19 22:51:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 876): 12688.6 MiB
[2026-06-19 22:52:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2800, pooled_mean_dice=0.7935, per_class=['0.7935'], iou=0.6577, recall=0.9851, precision=0.6643, vol_sim=0.8055, mcc=0.8039, min_class_dice=0.7935, coverage=[71]/88 samples
[2026-06-19 22:52:53] INFO segtask_v1.trainer.trainer: Epoch 877/1000 | LR=8.97e-04 | loss=0.2439 | val_dice=0.7935 | best=0.8292 (ep441) | 09:23:24 | L_main=0.1291 L_aux_1=0.1119(w=0.5) L_aux_2=0.1177(w=0.5)
[2026-06-19 22:52:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 877): 12688.6 MiB
[2026-06-19 22:53:56] INFO segtask_v1.trainer.validation:   Val: loss=0.3337, pooled_mean_dice=0.8109, per_class=['0.8109'], iou=0.6819, recall=0.9843, precision=0.6894, vol_sim=0.8238, mcc=0.8192, min_class_dice=0.8109, coverage=[70]/88 samples
[2026-06-19 22:53:56] INFO segtask_v1.trainer.trainer: Epoch 878/1000 | LR=8.95e-04 | loss=0.2308 | val_dice=0.8109 | best=0.8292 (ep441) | 09:24:28 | L_main=0.1196 L_aux_1=0.1082(w=0.5) L_aux_2=0.1142(w=0.5)
[2026-06-19 22:53:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 878): 12688.6 MiB
[2026-06-19 22:54:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3479, pooled_mean_dice=0.7786, per_class=['0.7786'], iou=0.6375, recall=0.9842, precision=0.6441, vol_sim=0.7911, mcc=0.7921, min_class_dice=0.7786, coverage=[74]/88 samples
[2026-06-19 22:54:58] INFO segtask_v1.trainer.trainer: Epoch 879/1000 | LR=8.93e-04 | loss=0.2676 | val_dice=0.7786 | best=0.8292 (ep441) | 09:25:30 | L_main=0.1353 L_aux_1=0.1282(w=0.5) L_aux_2=0.1364(w=0.5)
[2026-06-19 22:54:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 879): 12688.6 MiB
[2026-06-19 22:56:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3028, pooled_mean_dice=0.7894, per_class=['0.7894'], iou=0.6521, recall=0.9857, precision=0.6584, vol_sim=0.8009, mcc=0.8001, min_class_dice=0.7894, coverage=[79]/88 samples
[2026-06-19 22:56:02] INFO segtask_v1.trainer.trainer: Epoch 880/1000 | LR=8.90e-04 | loss=0.2817 | val_dice=0.7894 | best=0.8292 (ep441) | 09:26:33 | L_main=0.1423 L_aux_1=0.1327(w=0.5) L_aux_2=0.1459(w=0.5)
[2026-06-19 22:56:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 880): 12688.6 MiB
[2026-06-19 22:57:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3251, pooled_mean_dice=0.7939, per_class=['0.7939'], iou=0.6583, recall=0.9864, precision=0.6643, vol_sim=0.8048, mcc=0.8034, min_class_dice=0.7939, coverage=[77]/88 samples
[2026-06-19 22:57:05] INFO segtask_v1.trainer.trainer: Epoch 881/1000 | LR=8.88e-04 | loss=0.2860 | val_dice=0.7939 | best=0.8292 (ep441) | 09:27:37 | L_main=0.1456 L_aux_1=0.1364(w=0.5) L_aux_2=0.1445(w=0.5)
[2026-06-19 22:57:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 881): 12688.6 MiB
[2026-06-19 22:58:08] INFO segtask_v1.trainer.validation:   Val: loss=0.3150, pooled_mean_dice=0.7931, per_class=['0.7931'], iou=0.6572, recall=0.9833, precision=0.6646, vol_sim=0.8066, mcc=0.8022, min_class_dice=0.7931, coverage=[79]/88 samples
[2026-06-19 22:58:08] INFO segtask_v1.trainer.trainer: Epoch 882/1000 | LR=8.85e-04 | loss=0.2639 | val_dice=0.7931 | best=0.8292 (ep441) | 09:28:39 | L_main=0.1367 L_aux_1=0.1204(w=0.5) L_aux_2=0.1341(w=0.5)
[2026-06-19 22:58:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 882): 12688.6 MiB
[2026-06-19 22:59:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2859, pooled_mean_dice=0.7870, per_class=['0.7870'], iou=0.6489, recall=0.9833, precision=0.6561, vol_sim=0.8004, mcc=0.7981, min_class_dice=0.7870, coverage=[74]/88 samples
[2026-06-19 22:59:10] INFO segtask_v1.trainer.trainer: Epoch 883/1000 | LR=8.83e-04 | loss=0.2473 | val_dice=0.7870 | best=0.8292 (ep441) | 09:29:42 | L_main=0.1286 L_aux_1=0.1145(w=0.5) L_aux_2=0.1229(w=0.5)
[2026-06-19 22:59:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 883): 12688.6 MiB
[2026-06-19 23:00:14] INFO segtask_v1.trainer.validation:   Val: loss=0.3415, pooled_mean_dice=0.7958, per_class=['0.7958'], iou=0.6608, recall=0.9816, precision=0.6691, vol_sim=0.8107, mcc=0.8054, min_class_dice=0.7958, coverage=[76]/88 samples
[2026-06-19 23:00:14] INFO segtask_v1.trainer.trainer: Epoch 884/1000 | LR=8.80e-04 | loss=0.2600 | val_dice=0.7958 | best=0.8292 (ep441) | 09:30:46 | L_main=0.1338 L_aux_1=0.1245(w=0.5) L_aux_2=0.1279(w=0.5)
[2026-06-19 23:00:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 884): 12688.6 MiB
[2026-06-19 23:01:17] INFO segtask_v1.trainer.validation:   Val: loss=0.3062, pooled_mean_dice=0.7902, per_class=['0.7902'], iou=0.6531, recall=0.9864, precision=0.6591, vol_sim=0.8011, mcc=0.8007, min_class_dice=0.7902, coverage=[75]/88 samples
[2026-06-19 23:01:17] INFO segtask_v1.trainer.trainer: Epoch 885/1000 | LR=8.77e-04 | loss=0.2216 | val_dice=0.7902 | best=0.8292 (ep441) | 09:31:48 | L_main=0.1144 L_aux_1=0.1036(w=0.5) L_aux_2=0.1109(w=0.5)
[2026-06-19 23:01:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 885): 12688.6 MiB
[2026-06-19 23:02:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3111, pooled_mean_dice=0.7866, per_class=['0.7866'], iou=0.6483, recall=0.9838, precision=0.6553, vol_sim=0.7995, mcc=0.7979, min_class_dice=0.7866, coverage=[75]/88 samples
[2026-06-19 23:02:20] INFO segtask_v1.trainer.trainer: Epoch 886/1000 | LR=8.75e-04 | loss=0.2127 | val_dice=0.7866 | best=0.8292 (ep441) | 09:32:52 | L_main=0.1092 L_aux_1=0.0993(w=0.5) L_aux_2=0.1078(w=0.5)
[2026-06-19 23:02:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 886): 12688.6 MiB
[2026-06-19 23:03:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3422, pooled_mean_dice=0.7895, per_class=['0.7895'], iou=0.6522, recall=0.9838, precision=0.6593, vol_sim=0.8025, mcc=0.8002, min_class_dice=0.7895, coverage=[83]/88 samples
[2026-06-19 23:03:23] INFO segtask_v1.trainer.trainer: Epoch 887/1000 | LR=8.72e-04 | loss=0.2229 | val_dice=0.7895 | best=0.8292 (ep441) | 09:33:54 | L_main=0.1171 L_aux_1=0.1026(w=0.5) L_aux_2=0.1090(w=0.5)
[2026-06-19 23:03:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 887): 12688.6 MiB
[2026-06-19 23:04:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3462, pooled_mean_dice=0.7929, per_class=['0.7929'], iou=0.6568, recall=0.9860, precision=0.6630, vol_sim=0.8041, mcc=0.8032, min_class_dice=0.7929, coverage=[77]/88 samples
[2026-06-19 23:04:26] INFO segtask_v1.trainer.trainer: Epoch 888/1000 | LR=8.69e-04 | loss=0.2796 | val_dice=0.7929 | best=0.8292 (ep441) | 09:34:57 | L_main=0.1392 L_aux_1=0.1322(w=0.5) L_aux_2=0.1487(w=0.5)
[2026-06-19 23:04:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 888): 12688.6 MiB
[2026-06-19 23:05:29] INFO segtask_v1.trainer.validation:   Val: loss=0.3554, pooled_mean_dice=0.7828, per_class=['0.7828'], iou=0.6432, recall=0.9787, precision=0.6523, vol_sim=0.7999, mcc=0.7944, min_class_dice=0.7828, coverage=[79]/88 samples
[2026-06-19 23:05:29] INFO segtask_v1.trainer.trainer: Epoch 889/1000 | LR=8.67e-04 | loss=0.2880 | val_dice=0.7828 | best=0.8292 (ep441) | 09:36:00 | L_main=0.1427 L_aux_1=0.1360(w=0.5) L_aux_2=0.1547(w=0.5)
[2026-06-19 23:05:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 889): 12688.6 MiB
[2026-06-19 23:06:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2925, pooled_mean_dice=0.8079, per_class=['0.8079'], iou=0.6777, recall=0.9842, precision=0.6852, vol_sim=0.8209, mcc=0.8163, min_class_dice=0.8079, coverage=[72]/88 samples
[2026-06-19 23:06:31] INFO segtask_v1.trainer.trainer: Epoch 890/1000 | LR=8.64e-04 | loss=0.2492 | val_dice=0.8079 | best=0.8292 (ep441) | 09:37:02 | L_main=0.1243 L_aux_1=0.1170(w=0.5) L_aux_2=0.1328(w=0.5)
[2026-06-19 23:06:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 890): 12688.6 MiB
[2026-06-19 23:07:33] INFO segtask_v1.trainer.validation:   Val: loss=0.3673, pooled_mean_dice=0.7898, per_class=['0.7898'], iou=0.6526, recall=0.9824, precision=0.6603, vol_sim=0.8039, mcc=0.8007, min_class_dice=0.7898, coverage=[82]/88 samples
[2026-06-19 23:07:33] INFO segtask_v1.trainer.trainer: Epoch 891/1000 | LR=8.61e-04 | loss=0.2600 | val_dice=0.7898 | best=0.8292 (ep441) | 09:38:04 | L_main=0.1377 L_aux_1=0.1168(w=0.5) L_aux_2=0.1278(w=0.5)
[2026-06-19 23:07:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 891): 12688.6 MiB
[2026-06-19 23:08:35] INFO segtask_v1.trainer.validation:   Val: loss=0.3273, pooled_mean_dice=0.7981, per_class=['0.7981'], iou=0.6640, recall=0.9869, precision=0.6699, vol_sim=0.8087, mcc=0.8084, min_class_dice=0.7981, coverage=[71]/88 samples
[2026-06-19 23:08:35] INFO segtask_v1.trainer.trainer: Epoch 892/1000 | LR=8.59e-04 | loss=0.2820 | val_dice=0.7981 | best=0.8292 (ep441) | 09:39:06 | L_main=0.1400 L_aux_1=0.1334(w=0.5) L_aux_2=0.1504(w=0.5)
[2026-06-19 23:08:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 892): 12688.6 MiB
[2026-06-19 23:09:38] INFO segtask_v1.trainer.validation:   Val: loss=0.3094, pooled_mean_dice=0.7990, per_class=['0.7990'], iou=0.6653, recall=0.9856, precision=0.6718, vol_sim=0.8107, mcc=0.8078, min_class_dice=0.7990, coverage=[77]/88 samples
[2026-06-19 23:09:38] INFO segtask_v1.trainer.trainer: Epoch 893/1000 | LR=8.56e-04 | loss=0.2870 | val_dice=0.7990 | best=0.8292 (ep441) | 09:40:09 | L_main=0.1443 L_aux_1=0.1336(w=0.5) L_aux_2=0.1517(w=0.5)
[2026-06-19 23:09:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 893): 12688.6 MiB
[2026-06-19 23:10:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3005, pooled_mean_dice=0.7785, per_class=['0.7785'], iou=0.6373, recall=0.9832, precision=0.6443, vol_sim=0.7918, mcc=0.7906, min_class_dice=0.7785, coverage=[74]/88 samples
[2026-06-19 23:10:39] INFO segtask_v1.trainer.trainer: Epoch 894/1000 | LR=8.53e-04 | loss=0.2641 | val_dice=0.7785 | best=0.8292 (ep441) | 09:41:10 | L_main=0.1367 L_aux_1=0.1220(w=0.5) L_aux_2=0.1328(w=0.5)
[2026-06-19 23:10:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 894): 12688.6 MiB
[2026-06-19 23:11:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3238, pooled_mean_dice=0.8104, per_class=['0.8104'], iou=0.6812, recall=0.9875, precision=0.6871, vol_sim=0.8206, mcc=0.8166, min_class_dice=0.8104, coverage=[84]/88 samples
[2026-06-19 23:11:42] INFO segtask_v1.trainer.trainer: Epoch 895/1000 | LR=8.50e-04 | loss=0.2588 | val_dice=0.8104 | best=0.8292 (ep441) | 09:42:13 | L_main=0.1314 L_aux_1=0.1204(w=0.5) L_aux_2=0.1343(w=0.5)
[2026-06-19 23:11:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 895): 12688.6 MiB
[2026-06-19 23:12:43] INFO segtask_v1.trainer.validation:   Val: loss=0.3231, pooled_mean_dice=0.7941, per_class=['0.7941'], iou=0.6585, recall=0.9836, precision=0.6658, vol_sim=0.8074, mcc=0.8035, min_class_dice=0.7941, coverage=[78]/88 samples
[2026-06-19 23:12:43] INFO segtask_v1.trainer.trainer: Epoch 896/1000 | LR=8.47e-04 | loss=0.2291 | val_dice=0.7941 | best=0.8292 (ep441) | 09:43:15 | L_main=0.1207 L_aux_1=0.1022(w=0.5) L_aux_2=0.1147(w=0.5)
[2026-06-19 23:12:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 896): 12688.6 MiB
[2026-06-19 23:13:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3441, pooled_mean_dice=0.7986, per_class=['0.7986'], iou=0.6647, recall=0.9865, precision=0.6708, vol_sim=0.8095, mcc=0.8081, min_class_dice=0.7986, coverage=[80]/88 samples
[2026-06-19 23:13:46] INFO segtask_v1.trainer.trainer: Epoch 897/1000 | LR=8.44e-04 | loss=0.2439 | val_dice=0.7986 | best=0.8292 (ep441) | 09:44:17 | L_main=0.1235 L_aux_1=0.1157(w=0.5) L_aux_2=0.1251(w=0.5)
[2026-06-19 23:13:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 897): 12688.6 MiB
[2026-06-19 23:14:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3155, pooled_mean_dice=0.7945, per_class=['0.7945'], iou=0.6590, recall=0.9842, precision=0.6661, vol_sim=0.8073, mcc=0.8043, min_class_dice=0.7945, coverage=[77]/88 samples
[2026-06-19 23:14:49] INFO segtask_v1.trainer.trainer: Epoch 898/1000 | LR=8.42e-04 | loss=0.2474 | val_dice=0.7945 | best=0.8292 (ep441) | 09:45:20 | L_main=0.1270 L_aux_1=0.1144(w=0.5) L_aux_2=0.1264(w=0.5)
[2026-06-19 23:14:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 898): 12688.6 MiB
[2026-06-19 23:15:51] INFO segtask_v1.trainer.validation:   Val: loss=0.3551, pooled_mean_dice=0.7934, per_class=['0.7934'], iou=0.6575, recall=0.9864, precision=0.6635, vol_sim=0.8043, mcc=0.8044, min_class_dice=0.7934, coverage=[79]/88 samples
[2026-06-19 23:15:51] INFO segtask_v1.trainer.trainer: Epoch 899/1000 | LR=8.39e-04 | loss=0.2410 | val_dice=0.7934 | best=0.8292 (ep441) | 09:46:23 | L_main=0.1267 L_aux_1=0.1095(w=0.5) L_aux_2=0.1191(w=0.5)
[2026-06-19 23:15:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 899): 12688.6 MiB
[2026-06-19 23:16:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3338, pooled_mean_dice=0.8037, per_class=['0.8037'], iou=0.6719, recall=0.9841, precision=0.6792, vol_sim=0.8167, mcc=0.8123, min_class_dice=0.8037, coverage=[76]/88 samples
[2026-06-19 23:16:53] INFO segtask_v1.trainer.trainer: Epoch 900/1000 | LR=8.36e-04 | loss=0.2536 | val_dice=0.8037 | best=0.8292 (ep441) | 09:47:24 | L_main=0.1280 L_aux_1=0.1223(w=0.5) L_aux_2=0.1287(w=0.5)
[2026-06-19 23:16:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 900): 12688.6 MiB
[2026-06-19 23:17:56] INFO segtask_v1.trainer.validation:   Val: loss=0.3359, pooled_mean_dice=0.7935, per_class=['0.7935'], iou=0.6577, recall=0.9817, precision=0.6659, vol_sim=0.8083, mcc=0.8033, min_class_dice=0.7935, coverage=[75]/88 samples
[2026-06-19 23:17:56] INFO segtask_v1.trainer.trainer: Epoch 901/1000 | LR=8.33e-04 | loss=0.2705 | val_dice=0.7935 | best=0.8292 (ep441) | 09:48:27 | L_main=0.1403 L_aux_1=0.1245(w=0.5) L_aux_2=0.1358(w=0.5)
[2026-06-19 23:17:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 901): 12688.6 MiB
[2026-06-19 23:18:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3022, pooled_mean_dice=0.7984, per_class=['0.7984'], iou=0.6645, recall=0.9826, precision=0.6724, vol_sim=0.8125, mcc=0.8084, min_class_dice=0.7984, coverage=[71]/88 samples
[2026-06-19 23:18:59] INFO segtask_v1.trainer.trainer: Epoch 902/1000 | LR=8.30e-04 | loss=0.2786 | val_dice=0.7984 | best=0.8292 (ep441) | 09:49:30 | L_main=0.1402 L_aux_1=0.1310(w=0.5) L_aux_2=0.1458(w=0.5)
[2026-06-19 23:18:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 902): 12688.6 MiB
[2026-06-19 23:20:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3363, pooled_mean_dice=0.8047, per_class=['0.8047'], iou=0.6733, recall=0.9852, precision=0.6801, vol_sim=0.8168, mcc=0.8134, min_class_dice=0.8047, coverage=[79]/88 samples
[2026-06-19 23:20:01] INFO segtask_v1.trainer.trainer: Epoch 903/1000 | LR=8.27e-04 | loss=0.2553 | val_dice=0.8047 | best=0.8292 (ep441) | 09:50:32 | L_main=0.1332 L_aux_1=0.1156(w=0.5) L_aux_2=0.1287(w=0.5)
[2026-06-19 23:20:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 903): 12688.6 MiB
[2026-06-19 23:21:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3149, pooled_mean_dice=0.7840, per_class=['0.7840'], iou=0.6447, recall=0.9849, precision=0.6511, vol_sim=0.7960, mcc=0.7951, min_class_dice=0.7840, coverage=[77]/88 samples
[2026-06-19 23:21:04] INFO segtask_v1.trainer.trainer: Epoch 904/1000 | LR=8.24e-04 | loss=0.2454 | val_dice=0.7840 | best=0.8292 (ep441) | 09:51:35 | L_main=0.1286 L_aux_1=0.1109(w=0.5) L_aux_2=0.1227(w=0.5)
[2026-06-19 23:21:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 904): 12688.6 MiB
[2026-06-19 23:22:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3524, pooled_mean_dice=0.7779, per_class=['0.7779'], iou=0.6366, recall=0.9821, precision=0.6440, vol_sim=0.7921, mcc=0.7911, min_class_dice=0.7779, coverage=[77]/88 samples
[2026-06-19 23:22:07] INFO segtask_v1.trainer.trainer: Epoch 905/1000 | LR=8.21e-04 | loss=0.2338 | val_dice=0.7779 | best=0.8292 (ep441) | 09:52:38 | L_main=0.1172 L_aux_1=0.1083(w=0.5) L_aux_2=0.1249(w=0.5)
[2026-06-19 23:22:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 905): 12688.6 MiB
[2026-06-19 23:23:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3521, pooled_mean_dice=0.8018, per_class=['0.8018'], iou=0.6692, recall=0.9823, precision=0.6774, vol_sim=0.8163, mcc=0.8110, min_class_dice=0.8018, coverage=[78]/88 samples
[2026-06-19 23:23:09] INFO segtask_v1.trainer.trainer: Epoch 906/1000 | LR=8.18e-04 | loss=0.2665 | val_dice=0.8018 | best=0.8292 (ep441) | 09:53:40 | L_main=0.1377 L_aux_1=0.1240(w=0.5) L_aux_2=0.1336(w=0.5)
[2026-06-19 23:23:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 906): 12688.6 MiB
[2026-06-19 23:24:11] INFO segtask_v1.trainer.validation:   Val: loss=0.3064, pooled_mean_dice=0.8009, per_class=['0.8009'], iou=0.6679, recall=0.9870, precision=0.6739, vol_sim=0.8114, mcc=0.8106, min_class_dice=0.8009, coverage=[76]/88 samples
[2026-06-19 23:24:11] INFO segtask_v1.trainer.trainer: Epoch 907/1000 | LR=8.15e-04 | loss=0.2563 | val_dice=0.8009 | best=0.8292 (ep441) | 09:54:42 | L_main=0.1315 L_aux_1=0.1177(w=0.5) L_aux_2=0.1319(w=0.5)
[2026-06-19 23:24:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 907): 12688.6 MiB
[2026-06-19 23:25:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3216, pooled_mean_dice=0.8044, per_class=['0.8044'], iou=0.6727, recall=0.9817, precision=0.6813, vol_sim=0.8194, mcc=0.8130, min_class_dice=0.8044, coverage=[76]/88 samples
[2026-06-19 23:25:13] INFO segtask_v1.trainer.trainer: Epoch 908/1000 | LR=8.11e-04 | loss=0.2686 | val_dice=0.8044 | best=0.8292 (ep441) | 09:55:45 | L_main=0.1376 L_aux_1=0.1254(w=0.5) L_aux_2=0.1366(w=0.5)
[2026-06-19 23:25:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 908): 12688.6 MiB
[2026-06-19 23:26:16] INFO segtask_v1.trainer.validation:   Val: loss=0.3562, pooled_mean_dice=0.7999, per_class=['0.7999'], iou=0.6665, recall=0.9872, precision=0.6723, vol_sim=0.8103, mcc=0.8092, min_class_dice=0.7999, coverage=[80]/88 samples
[2026-06-19 23:26:16] INFO segtask_v1.trainer.trainer: Epoch 909/1000 | LR=8.08e-04 | loss=0.2886 | val_dice=0.7999 | best=0.8292 (ep441) | 09:56:47 | L_main=0.1511 L_aux_1=0.1329(w=0.5) L_aux_2=0.1421(w=0.5)
[2026-06-19 23:26:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 909): 12688.6 MiB
[2026-06-19 23:27:17] INFO segtask_v1.trainer.validation:   Val: loss=0.3264, pooled_mean_dice=0.7964, per_class=['0.7964'], iou=0.6617, recall=0.9824, precision=0.6697, vol_sim=0.8107, mcc=0.8059, min_class_dice=0.7964, coverage=[80]/88 samples
[2026-06-19 23:27:17] INFO segtask_v1.trainer.trainer: Epoch 910/1000 | LR=8.05e-04 | loss=0.2759 | val_dice=0.7964 | best=0.8292 (ep441) | 09:57:49 | L_main=0.1401 L_aux_1=0.1318(w=0.5) L_aux_2=0.1397(w=0.5)
[2026-06-19 23:27:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 910): 12688.6 MiB
[2026-06-19 23:28:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3352, pooled_mean_dice=0.8072, per_class=['0.8072'], iou=0.6767, recall=0.9843, precision=0.6841, vol_sim=0.8200, mcc=0.8146, min_class_dice=0.8072, coverage=[81]/88 samples
[2026-06-19 23:28:20] INFO segtask_v1.trainer.trainer: Epoch 911/1000 | LR=8.02e-04 | loss=0.2377 | val_dice=0.8072 | best=0.8292 (ep441) | 09:58:51 | L_main=0.1211 L_aux_1=0.1115(w=0.5) L_aux_2=0.1219(w=0.5)
[2026-06-19 23:28:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 911): 12688.6 MiB
[2026-06-19 23:29:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3388, pooled_mean_dice=0.7990, per_class=['0.7990'], iou=0.6653, recall=0.9838, precision=0.6727, vol_sim=0.8122, mcc=0.8092, min_class_dice=0.7990, coverage=[74]/88 samples
[2026-06-19 23:29:20] INFO segtask_v1.trainer.trainer: Epoch 912/1000 | LR=7.99e-04 | loss=0.2674 | val_dice=0.7990 | best=0.8292 (ep441) | 09:59:51 | L_main=0.1356 L_aux_1=0.1260(w=0.5) L_aux_2=0.1377(w=0.5)
[2026-06-19 23:29:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 912): 12688.6 MiB
[2026-06-19 23:30:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2945, pooled_mean_dice=0.7998, per_class=['0.7998'], iou=0.6663, recall=0.9826, precision=0.6743, vol_sim=0.8140, mcc=0.8084, min_class_dice=0.7998, coverage=[78]/88 samples
[2026-06-19 23:30:22] INFO segtask_v1.trainer.trainer: Epoch 913/1000 | LR=7.96e-04 | loss=0.2357 | val_dice=0.7998 | best=0.8292 (ep441) | 10:00:53 | L_main=0.1202 L_aux_1=0.1094(w=0.5) L_aux_2=0.1217(w=0.5)
[2026-06-19 23:30:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 913): 12688.6 MiB
[2026-06-19 23:31:24] INFO segtask_v1.trainer.validation:   Val: loss=0.3190, pooled_mean_dice=0.7971, per_class=['0.7971'], iou=0.6626, recall=0.9853, precision=0.6692, vol_sim=0.8090, mcc=0.8071, min_class_dice=0.7971, coverage=[75]/88 samples
[2026-06-19 23:31:24] INFO segtask_v1.trainer.trainer: Epoch 914/1000 | LR=7.92e-04 | loss=0.2397 | val_dice=0.7971 | best=0.8292 (ep441) | 10:01:55 | L_main=0.1225 L_aux_1=0.1086(w=0.5) L_aux_2=0.1258(w=0.5)
[2026-06-19 23:31:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 914): 12688.6 MiB
[2026-06-19 23:32:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2931, pooled_mean_dice=0.7926, per_class=['0.7926'], iou=0.6564, recall=0.9850, precision=0.6631, vol_sim=0.8046, mcc=0.8035, min_class_dice=0.7926, coverage=[75]/88 samples
[2026-06-19 23:32:27] INFO segtask_v1.trainer.trainer: Epoch 915/1000 | LR=7.89e-04 | loss=0.2292 | val_dice=0.7926 | best=0.8292 (ep441) | 10:02:58 | L_main=0.1181 L_aux_1=0.0999(w=0.5) L_aux_2=0.1223(w=0.5)
[2026-06-19 23:32:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 915): 12688.6 MiB
[2026-06-19 23:33:30] INFO segtask_v1.trainer.validation:   Val: loss=0.3400, pooled_mean_dice=0.8019, per_class=['0.8019'], iou=0.6693, recall=0.9845, precision=0.6764, vol_sim=0.8145, mcc=0.8110, min_class_dice=0.8019, coverage=[79]/88 samples
[2026-06-19 23:33:30] INFO segtask_v1.trainer.trainer: Epoch 916/1000 | LR=7.86e-04 | loss=0.2367 | val_dice=0.8019 | best=0.8292 (ep441) | 10:04:01 | L_main=0.1261 L_aux_1=0.1058(w=0.5) L_aux_2=0.1154(w=0.5)
[2026-06-19 23:33:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 916): 12688.6 MiB
[2026-06-19 23:34:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3507, pooled_mean_dice=0.7895, per_class=['0.7895'], iou=0.6522, recall=0.9826, precision=0.6598, vol_sim=0.8034, mcc=0.8005, min_class_dice=0.7895, coverage=[76]/88 samples
[2026-06-19 23:34:32] INFO segtask_v1.trainer.trainer: Epoch 917/1000 | LR=7.83e-04 | loss=0.2368 | val_dice=0.7895 | best=0.8292 (ep441) | 10:05:03 | L_main=0.1246 L_aux_1=0.1086(w=0.5) L_aux_2=0.1158(w=0.5)
[2026-06-19 23:34:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 917): 12688.6 MiB
[2026-06-19 23:35:35] INFO segtask_v1.trainer.validation:   Val: loss=0.3506, pooled_mean_dice=0.7869, per_class=['0.7869'], iou=0.6486, recall=0.9838, precision=0.6556, vol_sim=0.7998, mcc=0.7993, min_class_dice=0.7869, coverage=[71]/88 samples
[2026-06-19 23:35:35] INFO segtask_v1.trainer.trainer: Epoch 918/1000 | LR=7.79e-04 | loss=0.2411 | val_dice=0.7869 | best=0.8292 (ep441) | 10:06:06 | L_main=0.1294 L_aux_1=0.1066(w=0.5) L_aux_2=0.1169(w=0.5)
[2026-06-19 23:35:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 918): 12688.6 MiB
[2026-06-19 23:36:38] INFO segtask_v1.trainer.validation:   Val: loss=0.3272, pooled_mean_dice=0.8020, per_class=['0.8020'], iou=0.6694, recall=0.9869, precision=0.6754, vol_sim=0.8126, mcc=0.8111, min_class_dice=0.8020, coverage=[80]/88 samples
[2026-06-19 23:36:38] INFO segtask_v1.trainer.trainer: Epoch 919/1000 | LR=7.76e-04 | loss=0.2322 | val_dice=0.8020 | best=0.8292 (ep441) | 10:07:09 | L_main=0.1210 L_aux_1=0.1081(w=0.5) L_aux_2=0.1144(w=0.5)
[2026-06-19 23:36:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 919): 12688.6 MiB
[2026-06-19 23:37:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3681, pooled_mean_dice=0.8118, per_class=['0.8118'], iou=0.6832, recall=0.9800, precision=0.6928, vol_sim=0.8283, mcc=0.8201, min_class_dice=0.8118, coverage=[69]/88 samples
[2026-06-19 23:37:41] INFO segtask_v1.trainer.trainer: Epoch 920/1000 | LR=7.73e-04 | loss=0.2175 | val_dice=0.8118 | best=0.8292 (ep441) | 10:08:12 | L_main=0.1122 L_aux_1=0.0975(w=0.5) L_aux_2=0.1131(w=0.5)
[2026-06-19 23:37:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 920): 12688.6 MiB
[2026-06-19 23:38:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3139, pooled_mean_dice=0.8058, per_class=['0.8058'], iou=0.6748, recall=0.9851, precision=0.6818, vol_sim=0.8180, mcc=0.8149, min_class_dice=0.8058, coverage=[72]/88 samples
[2026-06-19 23:38:44] INFO segtask_v1.trainer.trainer: Epoch 921/1000 | LR=7.69e-04 | loss=0.2297 | val_dice=0.8058 | best=0.8292 (ep441) | 10:09:15 | L_main=0.1196 L_aux_1=0.1028(w=0.5) L_aux_2=0.1174(w=0.5)
[2026-06-19 23:38:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 921): 12688.6 MiB
[2026-06-19 23:39:47] INFO segtask_v1.trainer.validation:   Val: loss=0.3170, pooled_mean_dice=0.7893, per_class=['0.7893'], iou=0.6520, recall=0.9863, precision=0.6579, vol_sim=0.8003, mcc=0.8009, min_class_dice=0.7893, coverage=[74]/88 samples
[2026-06-19 23:39:47] INFO segtask_v1.trainer.trainer: Epoch 922/1000 | LR=7.66e-04 | loss=0.2203 | val_dice=0.7893 | best=0.8292 (ep441) | 10:10:18 | L_main=0.1139 L_aux_1=0.0978(w=0.5) L_aux_2=0.1150(w=0.5)
[2026-06-19 23:39:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 922): 12688.6 MiB
[2026-06-19 23:40:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3266, pooled_mean_dice=0.7945, per_class=['0.7945'], iou=0.6590, recall=0.9840, precision=0.6662, vol_sim=0.8074, mcc=0.8044, min_class_dice=0.7945, coverage=[76]/88 samples
[2026-06-19 23:40:49] INFO segtask_v1.trainer.trainer: Epoch 923/1000 | LR=7.63e-04 | loss=0.2287 | val_dice=0.7945 | best=0.8292 (ep441) | 10:11:20 | L_main=0.1168 L_aux_1=0.1026(w=0.5) L_aux_2=0.1213(w=0.5)
[2026-06-19 23:40:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 923): 12688.6 MiB
[2026-06-19 23:41:52] INFO segtask_v1.trainer.validation:   Val: loss=0.3218, pooled_mean_dice=0.8075, per_class=['0.8075'], iou=0.6772, recall=0.9859, precision=0.6839, vol_sim=0.8191, mcc=0.8157, min_class_dice=0.8075, coverage=[77]/88 samples
[2026-06-19 23:41:52] INFO segtask_v1.trainer.trainer: Epoch 924/1000 | LR=7.59e-04 | loss=0.2185 | val_dice=0.8075 | best=0.8292 (ep441) | 10:12:23 | L_main=0.1124 L_aux_1=0.1008(w=0.5) L_aux_2=0.1114(w=0.5)
[2026-06-19 23:41:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 924): 12688.6 MiB
[2026-06-19 23:42:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3066, pooled_mean_dice=0.8181, per_class=['0.8181'], iou=0.6922, recall=0.9854, precision=0.6994, vol_sim=0.8303, mcc=0.8250, min_class_dice=0.8181, coverage=[72]/88 samples
[2026-06-19 23:42:55] INFO segtask_v1.trainer.trainer: Epoch 925/1000 | LR=7.56e-04 | loss=0.2589 | val_dice=0.8181 | best=0.8292 (ep441) | 10:13:27 | L_main=0.1331 L_aux_1=0.1220(w=0.5) L_aux_2=0.1294(w=0.5)
[2026-06-19 23:42:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 925): 12688.6 MiB
[2026-06-19 23:43:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3271, pooled_mean_dice=0.7998, per_class=['0.7998'], iou=0.6663, recall=0.9840, precision=0.6736, vol_sim=0.8128, mcc=0.8094, min_class_dice=0.7998, coverage=[75]/88 samples
[2026-06-19 23:43:57] INFO segtask_v1.trainer.trainer: Epoch 926/1000 | LR=7.53e-04 | loss=0.2433 | val_dice=0.7998 | best=0.8292 (ep441) | 10:14:28 | L_main=0.1224 L_aux_1=0.1177(w=0.5) L_aux_2=0.1241(w=0.5)
[2026-06-19 23:43:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 926): 12688.6 MiB
[2026-06-19 23:44:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3374, pooled_mean_dice=0.7948, per_class=['0.7948'], iou=0.6594, recall=0.9845, precision=0.6664, vol_sim=0.8073, mcc=0.8048, min_class_dice=0.7948, coverage=[81]/88 samples
[2026-06-19 23:44:58] INFO segtask_v1.trainer.trainer: Epoch 927/1000 | LR=7.49e-04 | loss=0.2499 | val_dice=0.7948 | best=0.8292 (ep441) | 10:15:30 | L_main=0.1282 L_aux_1=0.1144(w=0.5) L_aux_2=0.1291(w=0.5)
[2026-06-19 23:44:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 927): 12688.6 MiB
[2026-06-19 23:46:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3178, pooled_mean_dice=0.8043, per_class=['0.8043'], iou=0.6726, recall=0.9854, precision=0.6794, vol_sim=0.8162, mcc=0.8140, min_class_dice=0.8043, coverage=[75]/88 samples
[2026-06-19 23:46:01] INFO segtask_v1.trainer.trainer: Epoch 928/1000 | LR=7.46e-04 | loss=0.2247 | val_dice=0.8043 | best=0.8292 (ep441) | 10:16:33 | L_main=0.1118 L_aux_1=0.1030(w=0.5) L_aux_2=0.1230(w=0.5)
[2026-06-19 23:46:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 928): 12688.6 MiB
[2026-06-19 23:47:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2925, pooled_mean_dice=0.7928, per_class=['0.7928'], iou=0.6568, recall=0.9844, precision=0.6637, vol_sim=0.8054, mcc=0.8038, min_class_dice=0.7928, coverage=[70]/88 samples
[2026-06-19 23:47:04] INFO segtask_v1.trainer.trainer: Epoch 929/1000 | LR=7.42e-04 | loss=0.2277 | val_dice=0.7928 | best=0.8292 (ep441) | 10:17:35 | L_main=0.1162 L_aux_1=0.1068(w=0.5) L_aux_2=0.1163(w=0.5)
[2026-06-19 23:47:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 929): 12688.6 MiB
[2026-06-19 23:48:06] INFO segtask_v1.trainer.validation:   Val: loss=0.3292, pooled_mean_dice=0.7927, per_class=['0.7927'], iou=0.6566, recall=0.9825, precision=0.6643, vol_sim=0.8068, mcc=0.8039, min_class_dice=0.7927, coverage=[72]/88 samples
[2026-06-19 23:48:06] INFO segtask_v1.trainer.trainer: Epoch 930/1000 | LR=7.39e-04 | loss=0.2353 | val_dice=0.7927 | best=0.8292 (ep441) | 10:18:37 | L_main=0.1216 L_aux_1=0.1090(w=0.5) L_aux_2=0.1185(w=0.5)
[2026-06-19 23:48:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 930): 12688.6 MiB
[2026-06-19 23:49:08] INFO segtask_v1.trainer.validation:   Val: loss=0.3300, pooled_mean_dice=0.7884, per_class=['0.7884'], iou=0.6508, recall=0.9837, precision=0.6578, vol_sim=0.8015, mcc=0.8004, min_class_dice=0.7884, coverage=[76]/88 samples
[2026-06-19 23:49:08] INFO segtask_v1.trainer.trainer: Epoch 931/1000 | LR=7.35e-04 | loss=0.2473 | val_dice=0.7884 | best=0.8292 (ep441) | 10:19:40 | L_main=0.1192 L_aux_1=0.1154(w=0.5) L_aux_2=0.1408(w=0.5)
[2026-06-19 23:49:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 931): 12688.6 MiB
[2026-06-19 23:50:10] INFO segtask_v1.trainer.validation:   Val: loss=0.3155, pooled_mean_dice=0.7864, per_class=['0.7864'], iou=0.6480, recall=0.9822, precision=0.6557, vol_sim=0.8006, mcc=0.7978, min_class_dice=0.7864, coverage=[79]/88 samples
[2026-06-19 23:50:10] INFO segtask_v1.trainer.trainer: Epoch 932/1000 | LR=7.32e-04 | loss=0.2614 | val_dice=0.7864 | best=0.8292 (ep441) | 10:20:41 | L_main=0.1336 L_aux_1=0.1166(w=0.5) L_aux_2=0.1390(w=0.5)
[2026-06-19 23:50:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 932): 12688.6 MiB
[2026-06-19 23:51:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3348, pooled_mean_dice=0.7806, per_class=['0.7806'], iou=0.6402, recall=0.9850, precision=0.6465, vol_sim=0.7925, mcc=0.7937, min_class_dice=0.7806, coverage=[77]/88 samples
[2026-06-19 23:51:13] INFO segtask_v1.trainer.trainer: Epoch 933/1000 | LR=7.28e-04 | loss=0.2563 | val_dice=0.7806 | best=0.8292 (ep441) | 10:21:44 | L_main=0.1263 L_aux_1=0.1162(w=0.5) L_aux_2=0.1437(w=0.5)
[2026-06-19 23:51:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 933): 12688.6 MiB
[2026-06-19 23:52:16] INFO segtask_v1.trainer.validation:   Val: loss=0.3089, pooled_mean_dice=0.8026, per_class=['0.8026'], iou=0.6703, recall=0.9852, precision=0.6771, vol_sim=0.8147, mcc=0.8107, min_class_dice=0.8026, coverage=[78]/88 samples
[2026-06-19 23:52:16] INFO segtask_v1.trainer.trainer: Epoch 934/1000 | LR=7.25e-04 | loss=0.2482 | val_dice=0.8026 | best=0.8292 (ep441) | 10:22:47 | L_main=0.1287 L_aux_1=0.1157(w=0.5) L_aux_2=0.1234(w=0.5)
[2026-06-19 23:52:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 934): 12688.6 MiB
[2026-06-19 23:53:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2754, pooled_mean_dice=0.8049, per_class=['0.8049'], iou=0.6736, recall=0.9814, precision=0.6823, vol_sim=0.8202, mcc=0.8127, min_class_dice=0.8049, coverage=[77]/88 samples
[2026-06-19 23:53:18] INFO segtask_v1.trainer.trainer: Epoch 935/1000 | LR=7.21e-04 | loss=0.2302 | val_dice=0.8049 | best=0.8292 (ep441) | 10:23:50 | L_main=0.1201 L_aux_1=0.1020(w=0.5) L_aux_2=0.1183(w=0.5)
[2026-06-19 23:53:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 935): 12688.6 MiB
[2026-06-19 23:54:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2868, pooled_mean_dice=0.8071, per_class=['0.8071'], iou=0.6765, recall=0.9857, precision=0.6832, vol_sim=0.8188, mcc=0.8153, min_class_dice=0.8071, coverage=[75]/88 samples
[2026-06-19 23:54:22] INFO segtask_v1.trainer.trainer: Epoch 936/1000 | LR=7.17e-04 | loss=0.2281 | val_dice=0.8071 | best=0.8292 (ep441) | 10:24:53 | L_main=0.1166 L_aux_1=0.1039(w=0.5) L_aux_2=0.1193(w=0.5)
[2026-06-19 23:54:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 936): 12688.6 MiB
[2026-06-19 23:55:24] INFO segtask_v1.trainer.validation:   Val: loss=0.3283, pooled_mean_dice=0.8020, per_class=['0.8020'], iou=0.6695, recall=0.9823, precision=0.6776, vol_sim=0.8164, mcc=0.8120, min_class_dice=0.8020, coverage=[71]/88 samples
[2026-06-19 23:55:24] INFO segtask_v1.trainer.trainer: Epoch 937/1000 | LR=7.14e-04 | loss=0.2372 | val_dice=0.8020 | best=0.8292 (ep441) | 10:25:55 | L_main=0.1234 L_aux_1=0.1093(w=0.5) L_aux_2=0.1182(w=0.5)
[2026-06-19 23:55:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 937): 12688.6 MiB
[2026-06-19 23:56:27] INFO segtask_v1.trainer.validation:   Val: loss=0.3237, pooled_mean_dice=0.7960, per_class=['0.7960'], iou=0.6611, recall=0.9830, precision=0.6687, vol_sim=0.8097, mcc=0.8062, min_class_dice=0.7960, coverage=[74]/88 samples
[2026-06-19 23:56:27] INFO segtask_v1.trainer.trainer: Epoch 938/1000 | LR=7.10e-04 | loss=0.2263 | val_dice=0.7960 | best=0.8292 (ep441) | 10:26:58 | L_main=0.1179 L_aux_1=0.1008(w=0.5) L_aux_2=0.1159(w=0.5)
[2026-06-19 23:56:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 938): 12688.6 MiB
[2026-06-19 23:57:29] INFO segtask_v1.trainer.validation:   Val: loss=0.3017, pooled_mean_dice=0.8115, per_class=['0.8115'], iou=0.6828, recall=0.9842, precision=0.6904, vol_sim=0.8246, mcc=0.8191, min_class_dice=0.8115, coverage=[74]/88 samples
[2026-06-19 23:57:29] INFO segtask_v1.trainer.trainer: Epoch 939/1000 | LR=7.07e-04 | loss=0.2352 | val_dice=0.8115 | best=0.8292 (ep441) | 10:28:01 | L_main=0.1211 L_aux_1=0.1100(w=0.5) L_aux_2=0.1183(w=0.5)
[2026-06-19 23:57:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 939): 12688.6 MiB
[2026-06-19 23:58:32] INFO segtask_v1.trainer.validation:   Val: loss=0.2726, pooled_mean_dice=0.8086, per_class=['0.8086'], iou=0.6788, recall=0.9851, precision=0.6858, vol_sim=0.8209, mcc=0.8163, min_class_dice=0.8086, coverage=[74]/88 samples
[2026-06-19 23:58:32] INFO segtask_v1.trainer.trainer: Epoch 940/1000 | LR=7.03e-04 | loss=0.2321 | val_dice=0.8086 | best=0.8292 (ep441) | 10:29:03 | L_main=0.1169 L_aux_1=0.1115(w=0.5) L_aux_2=0.1189(w=0.5)
[2026-06-19 23:58:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 940): 12688.6 MiB
[2026-06-19 23:59:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2881, pooled_mean_dice=0.8134, per_class=['0.8134'], iou=0.6855, recall=0.9847, precision=0.6929, vol_sim=0.8260, mcc=0.8211, min_class_dice=0.8134, coverage=[75]/88 samples
[2026-06-19 23:59:36] INFO segtask_v1.trainer.trainer: Epoch 941/1000 | LR=6.99e-04 | loss=0.2508 | val_dice=0.8134 | best=0.8292 (ep441) | 10:30:07 | L_main=0.1267 L_aux_1=0.1166(w=0.5) L_aux_2=0.1316(w=0.5)
[2026-06-19 23:59:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 941): 12688.6 MiB
[2026-06-20 00:00:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3067, pooled_mean_dice=0.8012, per_class=['0.8012'], iou=0.6683, recall=0.9828, precision=0.6763, vol_sim=0.8153, mcc=0.8109, min_class_dice=0.8012, coverage=[73]/88 samples
[2026-06-20 00:00:37] INFO segtask_v1.trainer.trainer: Epoch 942/1000 | LR=6.96e-04 | loss=0.2449 | val_dice=0.8012 | best=0.8292 (ep441) | 10:31:09 | L_main=0.1267 L_aux_1=0.1160(w=0.5) L_aux_2=0.1203(w=0.5)
[2026-06-20 00:00:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 942): 12688.6 MiB
[2026-06-20 00:01:39] INFO segtask_v1.trainer.validation:   Val: loss=0.3026, pooled_mean_dice=0.8075, per_class=['0.8075'], iou=0.6771, recall=0.9846, precision=0.6844, vol_sim=0.8201, mcc=0.8159, min_class_dice=0.8075, coverage=[75]/88 samples
[2026-06-20 00:01:39] INFO segtask_v1.trainer.trainer: Epoch 943/1000 | LR=6.92e-04 | loss=0.2340 | val_dice=0.8075 | best=0.8292 (ep441) | 10:32:11 | L_main=0.1206 L_aux_1=0.1100(w=0.5) L_aux_2=0.1167(w=0.5)
[2026-06-20 00:01:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 943): 12688.6 MiB
[2026-06-20 00:02:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3213, pooled_mean_dice=0.8055, per_class=['0.8055'], iou=0.6743, recall=0.9829, precision=0.6823, vol_sim=0.8194, mcc=0.8135, min_class_dice=0.8055, coverage=[78]/88 samples
[2026-06-20 00:02:42] INFO segtask_v1.trainer.trainer: Epoch 944/1000 | LR=6.88e-04 | loss=0.2156 | val_dice=0.8055 | best=0.8292 (ep441) | 10:33:13 | L_main=0.1122 L_aux_1=0.0963(w=0.5) L_aux_2=0.1104(w=0.5)
[2026-06-20 00:02:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 944): 12688.6 MiB
[2026-06-20 00:03:45] INFO segtask_v1.trainer.validation:   Val: loss=0.3273, pooled_mean_dice=0.8037, per_class=['0.8037'], iou=0.6719, recall=0.9822, precision=0.6801, vol_sim=0.8183, mcc=0.8127, min_class_dice=0.8037, coverage=[82]/88 samples
[2026-06-20 00:03:45] INFO segtask_v1.trainer.trainer: Epoch 945/1000 | LR=6.85e-04 | loss=0.2253 | val_dice=0.8037 | best=0.8292 (ep441) | 10:34:17 | L_main=0.1168 L_aux_1=0.0993(w=0.5) L_aux_2=0.1178(w=0.5)
[2026-06-20 00:03:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 945): 12688.6 MiB
[2026-06-20 00:04:47] INFO segtask_v1.trainer.validation:   Val: loss=0.2938, pooled_mean_dice=0.8186, per_class=['0.8186'], iou=0.6929, recall=0.9812, precision=0.7022, vol_sim=0.8342, mcc=0.8256, min_class_dice=0.8186, coverage=[72]/88 samples
[2026-06-20 00:04:47] INFO segtask_v1.trainer.trainer: Epoch 946/1000 | LR=6.81e-04 | loss=0.2086 | val_dice=0.8186 | best=0.8292 (ep441) | 10:35:18 | L_main=0.1043 L_aux_1=0.0975(w=0.5) L_aux_2=0.1110(w=0.5)
[2026-06-20 00:04:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 946): 12688.6 MiB
[2026-06-20 00:05:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2749, pooled_mean_dice=0.8035, per_class=['0.8035'], iou=0.6716, recall=0.9836, precision=0.6792, vol_sim=0.8169, mcc=0.8124, min_class_dice=0.8035, coverage=[73]/88 samples
[2026-06-20 00:05:50] INFO segtask_v1.trainer.trainer: Epoch 947/1000 | LR=6.77e-04 | loss=0.2180 | val_dice=0.8035 | best=0.8292 (ep441) | 10:36:21 | L_main=0.1129 L_aux_1=0.1002(w=0.5) L_aux_2=0.1101(w=0.5)
[2026-06-20 00:05:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 947): 12688.6 MiB
[2026-06-20 00:06:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2999, pooled_mean_dice=0.7984, per_class=['0.7984'], iou=0.6644, recall=0.9802, precision=0.6735, vol_sim=0.8145, mcc=0.8077, min_class_dice=0.7984, coverage=[77]/88 samples
[2026-06-20 00:06:52] INFO segtask_v1.trainer.trainer: Epoch 948/1000 | LR=6.74e-04 | loss=0.2363 | val_dice=0.7984 | best=0.8292 (ep441) | 10:37:23 | L_main=0.1212 L_aux_1=0.1086(w=0.5) L_aux_2=0.1215(w=0.5)
[2026-06-20 00:06:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 948): 12688.6 MiB
[2026-06-20 00:07:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3079, pooled_mean_dice=0.7989, per_class=['0.7989'], iou=0.6652, recall=0.9864, precision=0.6713, vol_sim=0.8099, mcc=0.8098, min_class_dice=0.7989, coverage=[69]/88 samples
[2026-06-20 00:07:55] INFO segtask_v1.trainer.trainer: Epoch 949/1000 | LR=6.70e-04 | loss=0.2215 | val_dice=0.7989 | best=0.8292 (ep441) | 10:38:26 | L_main=0.1151 L_aux_1=0.0989(w=0.5) L_aux_2=0.1139(w=0.5)
[2026-06-20 00:07:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 949): 12688.6 MiB
[2026-06-20 00:08:57] INFO segtask_v1.trainer.validation:   Val: loss=0.3206, pooled_mean_dice=0.8174, per_class=['0.8174'], iou=0.6912, recall=0.9851, precision=0.6985, vol_sim=0.8298, mcc=0.8248, min_class_dice=0.8174, coverage=[75]/88 samples
[2026-06-20 00:08:57] INFO segtask_v1.trainer.trainer: Epoch 950/1000 | LR=6.66e-04 | loss=0.2166 | val_dice=0.8174 | best=0.8292 (ep441) | 10:39:28 | L_main=0.1147 L_aux_1=0.0982(w=0.5) L_aux_2=0.1057(w=0.5)
[2026-06-20 00:08:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 950): 12688.6 MiB
[2026-06-20 00:09:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3269, pooled_mean_dice=0.8026, per_class=['0.8026'], iou=0.6703, recall=0.9820, precision=0.6786, vol_sim=0.8173, mcc=0.8117, min_class_dice=0.8026, coverage=[73]/88 samples
[2026-06-20 00:09:59] INFO segtask_v1.trainer.trainer: Epoch 951/1000 | LR=6.62e-04 | loss=0.2394 | val_dice=0.8026 | best=0.8292 (ep441) | 10:40:31 | L_main=0.1226 L_aux_1=0.1115(w=0.5) L_aux_2=0.1221(w=0.5)
[2026-06-20 00:09:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 951): 12688.6 MiB
[2026-06-20 00:11:02] INFO segtask_v1.trainer.validation:   Val: loss=0.3215, pooled_mean_dice=0.8048, per_class=['0.8048'], iou=0.6733, recall=0.9842, precision=0.6807, vol_sim=0.8177, mcc=0.8143, min_class_dice=0.8048, coverage=[73]/88 samples
[2026-06-20 00:11:02] INFO segtask_v1.trainer.trainer: Epoch 952/1000 | LR=6.59e-04 | loss=0.2368 | val_dice=0.8048 | best=0.8292 (ep441) | 10:41:33 | L_main=0.1252 L_aux_1=0.1041(w=0.5) L_aux_2=0.1192(w=0.5)
[2026-06-20 00:11:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 952): 12688.6 MiB
[2026-06-20 00:12:05] INFO segtask_v1.trainer.validation:   Val: loss=0.2784, pooled_mean_dice=0.8151, per_class=['0.8151'], iou=0.6879, recall=0.9823, precision=0.6966, vol_sim=0.8298, mcc=0.8219, min_class_dice=0.8151, coverage=[71]/88 samples
[2026-06-20 00:12:05] INFO segtask_v1.trainer.trainer: Epoch 953/1000 | LR=6.55e-04 | loss=0.2096 | val_dice=0.8151 | best=0.8292 (ep441) | 10:42:36 | L_main=0.1098 L_aux_1=0.0957(w=0.5) L_aux_2=0.1040(w=0.5)
[2026-06-20 00:12:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 953): 12688.6 MiB
[2026-06-20 00:13:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3003, pooled_mean_dice=0.7857, per_class=['0.7857'], iou=0.6471, recall=0.9843, precision=0.6538, vol_sim=0.7982, mcc=0.7965, min_class_dice=0.7857, coverage=[78]/88 samples
[2026-06-20 00:13:07] INFO segtask_v1.trainer.trainer: Epoch 954/1000 | LR=6.51e-04 | loss=0.2255 | val_dice=0.7857 | best=0.8292 (ep441) | 10:43:38 | L_main=0.1179 L_aux_1=0.1006(w=0.5) L_aux_2=0.1146(w=0.5)
[2026-06-20 00:13:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 954): 12688.6 MiB
[2026-06-20 00:14:09] INFO segtask_v1.trainer.validation:   Val: loss=0.3371, pooled_mean_dice=0.7988, per_class=['0.7988'], iou=0.6650, recall=0.9829, precision=0.6728, vol_sim=0.8127, mcc=0.8087, min_class_dice=0.7988, coverage=[75]/88 samples
[2026-06-20 00:14:09] INFO segtask_v1.trainer.trainer: Epoch 955/1000 | LR=6.47e-04 | loss=0.2302 | val_dice=0.7988 | best=0.8292 (ep441) | 10:44:40 | L_main=0.1189 L_aux_1=0.1057(w=0.5) L_aux_2=0.1168(w=0.5)
[2026-06-20 00:14:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 955): 12688.6 MiB
[2026-06-20 00:15:11] INFO segtask_v1.trainer.validation:   Val: loss=0.2497, pooled_mean_dice=0.8116, per_class=['0.8116'], iou=0.6829, recall=0.9856, precision=0.6898, vol_sim=0.8234, mcc=0.8197, min_class_dice=0.8116, coverage=[66]/88 samples
[2026-06-20 00:15:11] INFO segtask_v1.trainer.trainer: Epoch 956/1000 | LR=6.43e-04 | loss=0.2424 | val_dice=0.8116 | best=0.8292 (ep441) | 10:45:42 | L_main=0.1250 L_aux_1=0.1094(w=0.5) L_aux_2=0.1255(w=0.5)
[2026-06-20 00:15:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 956): 12688.6 MiB
[2026-06-20 00:16:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3236, pooled_mean_dice=0.8112, per_class=['0.8112'], iou=0.6824, recall=0.9838, precision=0.6902, vol_sim=0.8246, mcc=0.8186, min_class_dice=0.8112, coverage=[77]/88 samples
[2026-06-20 00:16:13] INFO segtask_v1.trainer.trainer: Epoch 957/1000 | LR=6.40e-04 | loss=0.2245 | val_dice=0.8112 | best=0.8292 (ep441) | 10:46:44 | L_main=0.1185 L_aux_1=0.1004(w=0.5) L_aux_2=0.1117(w=0.5)
[2026-06-20 00:16:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 957): 12688.6 MiB
[2026-06-20 00:17:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2873, pooled_mean_dice=0.8177, per_class=['0.8177'], iou=0.6916, recall=0.9851, precision=0.6989, vol_sim=0.8301, mcc=0.8247, min_class_dice=0.8177, coverage=[74]/88 samples
[2026-06-20 00:17:15] INFO segtask_v1.trainer.trainer: Epoch 958/1000 | LR=6.36e-04 | loss=0.2272 | val_dice=0.8177 | best=0.8292 (ep441) | 10:47:46 | L_main=0.1156 L_aux_1=0.1063(w=0.5) L_aux_2=0.1170(w=0.5)
[2026-06-20 00:17:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 958): 12688.6 MiB
[2026-06-20 00:18:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2756, pooled_mean_dice=0.8107, per_class=['0.8107'], iou=0.6816, recall=0.9841, precision=0.6892, vol_sim=0.8238, mcc=0.8183, min_class_dice=0.8107, coverage=[68]/88 samples
[2026-06-20 00:18:18] INFO segtask_v1.trainer.trainer: Epoch 959/1000 | LR=6.32e-04 | loss=0.3110 | val_dice=0.8107 | best=0.8292 (ep441) | 10:48:49 | L_main=0.1595 L_aux_1=0.1437(w=0.5) L_aux_2=0.1593(w=0.5)
[2026-06-20 00:18:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 959): 12688.6 MiB
[2026-06-20 00:19:20] INFO segtask_v1.trainer.validation:   Val: loss=0.3197, pooled_mean_dice=0.8103, per_class=['0.8103'], iou=0.6810, recall=0.9860, precision=0.6877, vol_sim=0.8217, mcc=0.8174, min_class_dice=0.8103, coverage=[80]/88 samples
[2026-06-20 00:19:20] INFO segtask_v1.trainer.trainer: Epoch 960/1000 | LR=6.28e-04 | loss=0.2406 | val_dice=0.8103 | best=0.8292 (ep441) | 10:49:51 | L_main=0.1261 L_aux_1=0.1097(w=0.5) L_aux_2=0.1192(w=0.5)
[2026-06-20 00:19:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 960): 12688.6 MiB
[2026-06-20 00:20:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3465, pooled_mean_dice=0.7876, per_class=['0.7876'], iou=0.6496, recall=0.9839, precision=0.6566, vol_sim=0.8005, mcc=0.7989, min_class_dice=0.7876, coverage=[76]/88 samples
[2026-06-20 00:20:23] INFO segtask_v1.trainer.trainer: Epoch 961/1000 | LR=6.24e-04 | loss=0.2363 | val_dice=0.7876 | best=0.8292 (ep441) | 10:50:54 | L_main=0.1231 L_aux_1=0.1069(w=0.5) L_aux_2=0.1194(w=0.5)
[2026-06-20 00:20:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 961): 12688.6 MiB
[2026-06-20 00:21:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3212, pooled_mean_dice=0.7987, per_class=['0.7987'], iou=0.6648, recall=0.9865, precision=0.6709, vol_sim=0.8096, mcc=0.8078, min_class_dice=0.7987, coverage=[79]/88 samples
[2026-06-20 00:21:26] INFO segtask_v1.trainer.trainer: Epoch 962/1000 | LR=6.20e-04 | loss=0.2358 | val_dice=0.7987 | best=0.8292 (ep441) | 10:51:57 | L_main=0.1222 L_aux_1=0.1068(w=0.5) L_aux_2=0.1204(w=0.5)
[2026-06-20 00:21:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 962): 12688.6 MiB
[2026-06-20 00:22:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2823, pooled_mean_dice=0.8080, per_class=['0.8080'], iou=0.6779, recall=0.9832, precision=0.6858, vol_sim=0.8218, mcc=0.8157, min_class_dice=0.8080, coverage=[78]/88 samples
[2026-06-20 00:22:28] INFO segtask_v1.trainer.trainer: Epoch 963/1000 | LR=6.17e-04 | loss=0.2401 | val_dice=0.8080 | best=0.8292 (ep441) | 10:52:59 | L_main=0.1262 L_aux_1=0.1111(w=0.5) L_aux_2=0.1167(w=0.5)
[2026-06-20 00:22:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 963): 12688.6 MiB
[2026-06-20 00:23:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2794, pooled_mean_dice=0.8096, per_class=['0.8096'], iou=0.6801, recall=0.9831, precision=0.6881, vol_sim=0.8235, mcc=0.8172, min_class_dice=0.8096, coverage=[75]/88 samples
[2026-06-20 00:23:30] INFO segtask_v1.trainer.trainer: Epoch 964/1000 | LR=6.13e-04 | loss=0.2364 | val_dice=0.8096 | best=0.8292 (ep441) | 10:54:01 | L_main=0.1223 L_aux_1=0.1114(w=0.5) L_aux_2=0.1167(w=0.5)
[2026-06-20 00:23:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 964): 12688.6 MiB
[2026-06-20 00:24:33] INFO segtask_v1.trainer.validation:   Val: loss=0.3417, pooled_mean_dice=0.7954, per_class=['0.7954'], iou=0.6603, recall=0.9870, precision=0.6661, vol_sim=0.8059, mcc=0.8063, min_class_dice=0.7954, coverage=[74]/88 samples
[2026-06-20 00:24:33] INFO segtask_v1.trainer.trainer: Epoch 965/1000 | LR=6.09e-04 | loss=0.2682 | val_dice=0.7954 | best=0.8292 (ep441) | 10:55:04 | L_main=0.1383 L_aux_1=0.1221(w=0.5) L_aux_2=0.1375(w=0.5)
[2026-06-20 00:24:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 965): 12688.6 MiB
[2026-06-20 00:25:36] INFO segtask_v1.trainer.validation:   Val: loss=0.3103, pooled_mean_dice=0.7793, per_class=['0.7793'], iou=0.6385, recall=0.9849, precision=0.6448, vol_sim=0.7913, mcc=0.7920, min_class_dice=0.7793, coverage=[75]/88 samples
[2026-06-20 00:25:36] INFO segtask_v1.trainer.trainer: Epoch 966/1000 | LR=6.05e-04 | loss=0.3326 | val_dice=0.7793 | best=0.8292 (ep441) | 10:56:07 | L_main=0.1651 L_aux_1=0.1545(w=0.5) L_aux_2=0.1804(w=0.5)
[2026-06-20 00:25:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 966): 12688.6 MiB
[2026-06-20 00:26:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2849, pooled_mean_dice=0.7911, per_class=['0.7911'], iou=0.6544, recall=0.9847, precision=0.6611, vol_sim=0.8034, mcc=0.8023, min_class_dice=0.7911, coverage=[71]/88 samples
[2026-06-20 00:26:37] INFO segtask_v1.trainer.trainer: Epoch 967/1000 | LR=6.01e-04 | loss=0.3264 | val_dice=0.7911 | best=0.8292 (ep441) | 10:57:09 | L_main=0.1590 L_aux_1=0.1567(w=0.5) L_aux_2=0.1781(w=0.5)
[2026-06-20 00:26:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 967): 12688.6 MiB
[2026-06-20 00:27:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3045, pooled_mean_dice=0.8190, per_class=['0.8190'], iou=0.6935, recall=0.9859, precision=0.7005, vol_sim=0.8308, mcc=0.8253, min_class_dice=0.8190, coverage=[81]/88 samples
[2026-06-20 00:27:40] INFO segtask_v1.trainer.trainer: Epoch 968/1000 | LR=5.97e-04 | loss=0.2607 | val_dice=0.8190 | best=0.8292 (ep441) | 10:58:11 | L_main=0.1320 L_aux_1=0.1224(w=0.5) L_aux_2=0.1349(w=0.5)
[2026-06-20 00:27:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 968): 12688.6 MiB
[2026-06-20 00:28:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3143, pooled_mean_dice=0.7794, per_class=['0.7794'], iou=0.6386, recall=0.9847, precision=0.6450, vol_sim=0.7915, mcc=0.7917, min_class_dice=0.7794, coverage=[78]/88 samples
[2026-06-20 00:28:41] INFO segtask_v1.trainer.trainer: Epoch 969/1000 | LR=5.93e-04 | loss=0.2545 | val_dice=0.7794 | best=0.8292 (ep441) | 10:59:12 | L_main=0.1274 L_aux_1=0.1200(w=0.5) L_aux_2=0.1343(w=0.5)
[2026-06-20 00:28:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 969): 12688.6 MiB
[2026-06-20 00:29:44] INFO segtask_v1.trainer.validation:   Val: loss=0.3349, pooled_mean_dice=0.7996, per_class=['0.7996'], iou=0.6661, recall=0.9828, precision=0.6739, vol_sim=0.8135, mcc=0.8095, min_class_dice=0.7996, coverage=[74]/88 samples
[2026-06-20 00:29:44] INFO segtask_v1.trainer.trainer: Epoch 970/1000 | LR=5.89e-04 | loss=0.2583 | val_dice=0.7996 | best=0.8292 (ep441) | 11:00:15 | L_main=0.1285 L_aux_1=0.1229(w=0.5) L_aux_2=0.1367(w=0.5)
[2026-06-20 00:29:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 970): 12688.6 MiB
[2026-06-20 00:30:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3085, pooled_mean_dice=0.8082, per_class=['0.8082'], iou=0.6782, recall=0.9832, precision=0.6862, vol_sim=0.8221, mcc=0.8158, min_class_dice=0.8082, coverage=[73]/88 samples
[2026-06-20 00:30:46] INFO segtask_v1.trainer.trainer: Epoch 971/1000 | LR=5.85e-04 | loss=0.2566 | val_dice=0.8082 | best=0.8292 (ep441) | 11:01:17 | L_main=0.1310 L_aux_1=0.1195(w=0.5) L_aux_2=0.1318(w=0.5)
[2026-06-20 00:30:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 971): 12688.6 MiB
[2026-06-20 00:31:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2831, pooled_mean_dice=0.8037, per_class=['0.8037'], iou=0.6719, recall=0.9838, precision=0.6794, vol_sim=0.8170, mcc=0.8130, min_class_dice=0.8037, coverage=[73]/88 samples
[2026-06-20 00:31:48] INFO segtask_v1.trainer.trainer: Epoch 972/1000 | LR=5.82e-04 | loss=0.2396 | val_dice=0.8037 | best=0.8292 (ep441) | 11:02:19 | L_main=0.1233 L_aux_1=0.1099(w=0.5) L_aux_2=0.1229(w=0.5)
[2026-06-20 00:31:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 972): 12688.6 MiB
[2026-06-20 00:32:50] INFO segtask_v1.trainer.validation:   Val: loss=0.3392, pooled_mean_dice=0.7866, per_class=['0.7866'], iou=0.6483, recall=0.9851, precision=0.6548, vol_sim=0.7986, mcc=0.7986, min_class_dice=0.7866, coverage=[77]/88 samples
[2026-06-20 00:32:50] INFO segtask_v1.trainer.trainer: Epoch 973/1000 | LR=5.78e-04 | loss=0.2530 | val_dice=0.7866 | best=0.8292 (ep441) | 11:03:21 | L_main=0.1279 L_aux_1=0.1198(w=0.5) L_aux_2=0.1305(w=0.5)
[2026-06-20 00:32:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 973): 12688.6 MiB
[2026-06-20 00:33:53] INFO segtask_v1.trainer.validation:   Val: loss=0.3653, pooled_mean_dice=0.7924, per_class=['0.7924'], iou=0.6562, recall=0.9857, precision=0.6626, vol_sim=0.8040, mcc=0.8040, min_class_dice=0.7924, coverage=[78]/88 samples
[2026-06-20 00:33:53] INFO segtask_v1.trainer.trainer: Epoch 974/1000 | LR=5.74e-04 | loss=0.2383 | val_dice=0.7924 | best=0.8292 (ep441) | 11:04:24 | L_main=0.1224 L_aux_1=0.1091(w=0.5) L_aux_2=0.1227(w=0.5)
[2026-06-20 00:33:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 974): 12688.6 MiB
[2026-06-20 00:34:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3031, pooled_mean_dice=0.7961, per_class=['0.7961'], iou=0.6612, recall=0.9868, precision=0.6671, vol_sim=0.8067, mcc=0.8054, min_class_dice=0.7961, coverage=[77]/88 samples
[2026-06-20 00:34:55] INFO segtask_v1.trainer.trainer: Epoch 975/1000 | LR=5.70e-04 | loss=0.2608 | val_dice=0.7961 | best=0.8292 (ep441) | 11:05:27 | L_main=0.1361 L_aux_1=0.1170(w=0.5) L_aux_2=0.1326(w=0.5)
[2026-06-20 00:34:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 975): 12688.6 MiB
[2026-06-20 00:35:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3346, pooled_mean_dice=0.7922, per_class=['0.7922'], iou=0.6559, recall=0.9864, precision=0.6619, vol_sim=0.8031, mcc=0.8036, min_class_dice=0.7922, coverage=[76]/88 samples
[2026-06-20 00:35:58] INFO segtask_v1.trainer.trainer: Epoch 976/1000 | LR=5.66e-04 | loss=0.2595 | val_dice=0.7922 | best=0.8292 (ep441) | 11:06:29 | L_main=0.1337 L_aux_1=0.1216(w=0.5) L_aux_2=0.1299(w=0.5)
[2026-06-20 00:35:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 976): 12688.6 MiB
[2026-06-20 00:36:59] INFO segtask_v1.trainer.validation:   Val: loss=0.3077, pooled_mean_dice=0.7984, per_class=['0.7984'], iou=0.6644, recall=0.9874, precision=0.6701, vol_sim=0.8086, mcc=0.8076, min_class_dice=0.7984, coverage=[76]/88 samples
[2026-06-20 00:36:59] INFO segtask_v1.trainer.trainer: Epoch 977/1000 | LR=5.62e-04 | loss=0.2497 | val_dice=0.7984 | best=0.8292 (ep441) | 11:07:30 | L_main=0.1270 L_aux_1=0.1144(w=0.5) L_aux_2=0.1309(w=0.5)
[2026-06-20 00:36:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 977): 12688.6 MiB
[2026-06-20 00:38:01] INFO segtask_v1.trainer.validation:   Val: loss=0.3178, pooled_mean_dice=0.7915, per_class=['0.7915'], iou=0.6549, recall=0.9853, precision=0.6613, vol_sim=0.8032, mcc=0.8023, min_class_dice=0.7915, coverage=[76]/88 samples
[2026-06-20 00:38:01] INFO segtask_v1.trainer.trainer: Epoch 978/1000 | LR=5.58e-04 | loss=0.2348 | val_dice=0.7915 | best=0.8292 (ep441) | 11:08:32 | L_main=0.1206 L_aux_1=0.1080(w=0.5) L_aux_2=0.1204(w=0.5)
[2026-06-20 00:38:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 978): 12688.6 MiB
[2026-06-20 00:39:04] INFO segtask_v1.trainer.validation:   Val: loss=0.3074, pooled_mean_dice=0.8101, per_class=['0.8101'], iou=0.6809, recall=0.9844, precision=0.6883, vol_sim=0.8230, mcc=0.8178, min_class_dice=0.8101, coverage=[80]/88 samples
[2026-06-20 00:39:04] INFO segtask_v1.trainer.trainer: Epoch 979/1000 | LR=5.54e-04 | loss=0.2378 | val_dice=0.8101 | best=0.8292 (ep441) | 11:09:35 | L_main=0.1230 L_aux_1=0.1077(w=0.5) L_aux_2=0.1220(w=0.5)
[2026-06-20 00:39:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 979): 12688.6 MiB
[2026-06-20 00:40:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2902, pooled_mean_dice=0.8078, per_class=['0.8078'], iou=0.6775, recall=0.9812, precision=0.6865, vol_sim=0.8233, mcc=0.8160, min_class_dice=0.8078, coverage=[70]/88 samples
[2026-06-20 00:40:06] INFO segtask_v1.trainer.trainer: Epoch 980/1000 | LR=5.50e-04 | loss=0.2148 | val_dice=0.8078 | best=0.8292 (ep441) | 11:10:38 | L_main=0.1126 L_aux_1=0.0960(w=0.5) L_aux_2=0.1084(w=0.5)
[2026-06-20 00:40:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 980): 12688.6 MiB
[2026-06-20 00:41:10] INFO segtask_v1.trainer.validation:   Val: loss=0.3057, pooled_mean_dice=0.8146, per_class=['0.8146'], iou=0.6872, recall=0.9855, precision=0.6942, vol_sim=0.8266, mcc=0.8216, min_class_dice=0.8146, coverage=[77]/88 samples
[2026-06-20 00:41:10] INFO segtask_v1.trainer.trainer: Epoch 981/1000 | LR=5.46e-04 | loss=0.2181 | val_dice=0.8146 | best=0.8292 (ep441) | 11:11:41 | L_main=0.1136 L_aux_1=0.1016(w=0.5) L_aux_2=0.1074(w=0.5)
[2026-06-20 00:41:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 981): 12688.6 MiB
[2026-06-20 00:42:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2741, pooled_mean_dice=0.8048, per_class=['0.8048'], iou=0.6733, recall=0.9849, precision=0.6804, vol_sim=0.8171, mcc=0.8131, min_class_dice=0.8048, coverage=[74]/88 samples
[2026-06-20 00:42:12] INFO segtask_v1.trainer.trainer: Epoch 982/1000 | LR=5.42e-04 | loss=0.2308 | val_dice=0.8048 | best=0.8292 (ep441) | 11:12:43 | L_main=0.1189 L_aux_1=0.1076(w=0.5) L_aux_2=0.1161(w=0.5)
[2026-06-20 00:42:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 982): 12688.6 MiB
[2026-06-20 00:43:13] INFO segtask_v1.trainer.validation:   Val: loss=0.3057, pooled_mean_dice=0.7862, per_class=['0.7862'], iou=0.6477, recall=0.9857, precision=0.6539, vol_sim=0.7976, mcc=0.7974, min_class_dice=0.7862, coverage=[78]/88 samples
[2026-06-20 00:43:13] INFO segtask_v1.trainer.trainer: Epoch 983/1000 | LR=5.38e-04 | loss=0.2157 | val_dice=0.7862 | best=0.8292 (ep441) | 11:13:44 | L_main=0.1099 L_aux_1=0.1006(w=0.5) L_aux_2=0.1110(w=0.5)
[2026-06-20 00:43:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 983): 12688.6 MiB
[2026-06-20 00:44:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2922, pooled_mean_dice=0.8035, per_class=['0.8035'], iou=0.6716, recall=0.9870, precision=0.6776, vol_sim=0.8141, mcc=0.8121, min_class_dice=0.8035, coverage=[71]/88 samples
[2026-06-20 00:44:15] INFO segtask_v1.trainer.trainer: Epoch 984/1000 | LR=5.34e-04 | loss=0.2169 | val_dice=0.8035 | best=0.8292 (ep441) | 11:14:46 | L_main=0.1095 L_aux_1=0.1022(w=0.5) L_aux_2=0.1125(w=0.5)
[2026-06-20 00:44:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 984): 12688.6 MiB
[2026-06-20 00:45:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2949, pooled_mean_dice=0.8122, per_class=['0.8122'], iou=0.6838, recall=0.9809, precision=0.6930, vol_sim=0.8280, mcc=0.8200, min_class_dice=0.8122, coverage=[73]/88 samples
[2026-06-20 00:45:17] INFO segtask_v1.trainer.trainer: Epoch 985/1000 | LR=5.30e-04 | loss=0.2249 | val_dice=0.8122 | best=0.8292 (ep441) | 11:15:48 | L_main=0.1142 L_aux_1=0.1048(w=0.5) L_aux_2=0.1168(w=0.5)
[2026-06-20 00:45:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 985): 12688.6 MiB
[2026-06-20 00:46:19] INFO segtask_v1.trainer.validation:   Val: loss=0.3019, pooled_mean_dice=0.8017, per_class=['0.8017'], iou=0.6690, recall=0.9848, precision=0.6760, vol_sim=0.8141, mcc=0.8107, min_class_dice=0.8017, coverage=[74]/88 samples
[2026-06-20 00:46:19] INFO segtask_v1.trainer.trainer: Epoch 986/1000 | LR=5.26e-04 | loss=0.2119 | val_dice=0.8017 | best=0.8292 (ep441) | 11:16:50 | L_main=0.1062 L_aux_1=0.0974(w=0.5) L_aux_2=0.1140(w=0.5)
[2026-06-20 00:46:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 986): 12688.6 MiB
[2026-06-20 00:47:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2926, pooled_mean_dice=0.8111, per_class=['0.8111'], iou=0.6822, recall=0.9823, precision=0.6907, vol_sim=0.8257, mcc=0.8190, min_class_dice=0.8111, coverage=[76]/88 samples
[2026-06-20 00:47:21] INFO segtask_v1.trainer.trainer: Epoch 987/1000 | LR=5.22e-04 | loss=0.2262 | val_dice=0.8111 | best=0.8292 (ep441) | 11:17:53 | L_main=0.1166 L_aux_1=0.1034(w=0.5) L_aux_2=0.1157(w=0.5)
[2026-06-20 00:47:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 987): 12688.6 MiB
[2026-06-20 00:48:25] INFO segtask_v1.trainer.validation:   Val: loss=0.2857, pooled_mean_dice=0.8128, per_class=['0.8128'], iou=0.6846, recall=0.9811, precision=0.6937, vol_sim=0.8284, mcc=0.8195, min_class_dice=0.8128, coverage=[75]/88 samples
[2026-06-20 00:48:25] INFO segtask_v1.trainer.trainer: Epoch 988/1000 | LR=5.18e-04 | loss=0.2245 | val_dice=0.8128 | best=0.8292 (ep441) | 11:18:56 | L_main=0.1182 L_aux_1=0.1013(w=0.5) L_aux_2=0.1114(w=0.5)
[2026-06-20 00:48:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 988): 12688.6 MiB
[2026-06-20 00:49:26] INFO segtask_v1.trainer.validation:   Val: loss=0.3124, pooled_mean_dice=0.7946, per_class=['0.7946'], iou=0.6593, recall=0.9865, precision=0.6653, vol_sim=0.8055, mcc=0.8053, min_class_dice=0.7946, coverage=[74]/88 samples
[2026-06-20 00:49:26] INFO segtask_v1.trainer.trainer: Epoch 989/1000 | LR=5.14e-04 | loss=0.2206 | val_dice=0.7946 | best=0.8292 (ep441) | 11:19:57 | L_main=0.1097 L_aux_1=0.1027(w=0.5) L_aux_2=0.1191(w=0.5)
[2026-06-20 00:49:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 989): 12688.6 MiB
[2026-06-20 00:50:29] INFO segtask_v1.trainer.validation:   Val: loss=0.3113, pooled_mean_dice=0.7927, per_class=['0.7927'], iou=0.6566, recall=0.9826, precision=0.6643, vol_sim=0.8067, mcc=0.8036, min_class_dice=0.7927, coverage=[74]/88 samples
[2026-06-20 00:50:29] INFO segtask_v1.trainer.trainer: Epoch 990/1000 | LR=5.10e-04 | loss=0.2306 | val_dice=0.7927 | best=0.8292 (ep441) | 11:21:00 | L_main=0.1208 L_aux_1=0.1054(w=0.5) L_aux_2=0.1142(w=0.5)
[2026-06-20 00:50:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 990): 12688.6 MiB
[2026-06-20 00:51:32] INFO segtask_v1.trainer.validation:   Val: loss=0.3276, pooled_mean_dice=0.7994, per_class=['0.7994'], iou=0.6658, recall=0.9842, precision=0.6730, vol_sim=0.8122, mcc=0.8094, min_class_dice=0.7994, coverage=[76]/88 samples
[2026-06-20 00:51:32] INFO segtask_v1.trainer.trainer: Epoch 991/1000 | LR=5.06e-04 | loss=0.2085 | val_dice=0.7994 | best=0.8292 (ep441) | 11:22:03 | L_main=0.1111 L_aux_1=0.0954(w=0.5) L_aux_2=0.0996(w=0.5)
[2026-06-20 00:51:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 991): 12688.6 MiB
[2026-06-20 00:52:34] INFO segtask_v1.trainer.validation:   Val: loss=0.3278, pooled_mean_dice=0.8196, per_class=['0.8196'], iou=0.6944, recall=0.9858, precision=0.7014, vol_sim=0.8314, mcc=0.8271, min_class_dice=0.8196, coverage=[74]/88 samples
[2026-06-20 00:52:34] INFO segtask_v1.trainer.trainer: Epoch 992/1000 | LR=5.02e-04 | loss=0.2354 | val_dice=0.8196 | best=0.8292 (ep441) | 11:23:05 | L_main=0.1209 L_aux_1=0.1113(w=0.5) L_aux_2=0.1176(w=0.5)
[2026-06-20 00:52:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 992): 12688.6 MiB
[2026-06-20 00:53:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3288, pooled_mean_dice=0.8082, per_class=['0.8082'], iou=0.6781, recall=0.9849, precision=0.6852, vol_sim=0.8206, mcc=0.8164, min_class_dice=0.8082, coverage=[79]/88 samples
[2026-06-20 00:53:37] INFO segtask_v1.trainer.trainer: Epoch 993/1000 | LR=4.99e-04 | loss=0.2483 | val_dice=0.8082 | best=0.8292 (ep441) | 11:24:09 | L_main=0.1267 L_aux_1=0.1191(w=0.5) L_aux_2=0.1241(w=0.5)
[2026-06-20 00:53:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 993): 12688.6 MiB
[2026-06-20 00:54:40] INFO segtask_v1.trainer.validation:   Val: loss=0.3051, pooled_mean_dice=0.7965, per_class=['0.7965'], iou=0.6618, recall=0.9823, precision=0.6698, vol_sim=0.8109, mcc=0.8062, min_class_dice=0.7965, coverage=[74]/88 samples
[2026-06-20 00:54:40] INFO segtask_v1.trainer.trainer: Epoch 994/1000 | LR=4.95e-04 | loss=0.2330 | val_dice=0.7965 | best=0.8292 (ep441) | 11:25:11 | L_main=0.1180 L_aux_1=0.1101(w=0.5) L_aux_2=0.1199(w=0.5)
[2026-06-20 00:54:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 994): 12688.6 MiB
[2026-06-20 00:55:42] INFO segtask_v1.trainer.validation:   Val: loss=0.3066, pooled_mean_dice=0.8024, per_class=['0.8024'], iou=0.6700, recall=0.9813, precision=0.6786, vol_sim=0.8177, mcc=0.8115, min_class_dice=0.8024, coverage=[77]/88 samples
[2026-06-20 00:55:42] INFO segtask_v1.trainer.trainer: Epoch 995/1000 | LR=4.91e-04 | loss=0.2436 | val_dice=0.8024 | best=0.8292 (ep441) | 11:26:13 | L_main=0.1252 L_aux_1=0.1135(w=0.5) L_aux_2=0.1232(w=0.5)
[2026-06-20 00:55:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 995): 12688.6 MiB
[2026-06-20 00:56:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2918, pooled_mean_dice=0.7995, per_class=['0.7995'], iou=0.6660, recall=0.9826, precision=0.6740, vol_sim=0.8137, mcc=0.8091, min_class_dice=0.7995, coverage=[72]/88 samples
[2026-06-20 00:56:44] INFO segtask_v1.trainer.trainer: Epoch 996/1000 | LR=4.87e-04 | loss=0.2518 | val_dice=0.7995 | best=0.8292 (ep441) | 11:27:15 | L_main=0.1263 L_aux_1=0.1185(w=0.5) L_aux_2=0.1326(w=0.5)
[2026-06-20 00:56:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 996): 12688.6 MiB
[2026-06-20 00:57:47] INFO segtask_v1.trainer.validation:   Val: loss=0.3290, pooled_mean_dice=0.7916, per_class=['0.7916'], iou=0.6551, recall=0.9840, precision=0.6621, vol_sim=0.8045, mcc=0.8030, min_class_dice=0.7916, coverage=[78]/88 samples
[2026-06-20 00:57:47] INFO segtask_v1.trainer.trainer: Epoch 997/1000 | LR=4.83e-04 | loss=0.2321 | val_dice=0.7916 | best=0.8292 (ep441) | 11:28:19 | L_main=0.1160 L_aux_1=0.1078(w=0.5) L_aux_2=0.1242(w=0.5)
[2026-06-20 00:57:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 997): 12688.6 MiB
[2026-06-20 00:58:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2820, pooled_mean_dice=0.8016, per_class=['0.8016'], iou=0.6689, recall=0.9850, precision=0.6758, vol_sim=0.8138, mcc=0.8106, min_class_dice=0.8016, coverage=[73]/88 samples
[2026-06-20 00:58:50] INFO segtask_v1.trainer.trainer: Epoch 998/1000 | LR=4.79e-04 | loss=0.2508 | val_dice=0.8016 | best=0.8292 (ep441) | 11:29:21 | L_main=0.1293 L_aux_1=0.1143(w=0.5) L_aux_2=0.1288(w=0.5)
[2026-06-20 00:58:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 998): 12688.6 MiB
[2026-06-20 00:59:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2814, pooled_mean_dice=0.7994, per_class=['0.7994'], iou=0.6658, recall=0.9845, precision=0.6729, vol_sim=0.8119, mcc=0.8091, min_class_dice=0.7994, coverage=[75]/88 samples
[2026-06-20 00:59:52] INFO segtask_v1.trainer.trainer: Epoch 999/1000 | LR=4.75e-04 | loss=0.2399 | val_dice=0.7994 | best=0.8292 (ep441) | 11:30:24 | L_main=0.1220 L_aux_1=0.1103(w=0.5) L_aux_2=0.1257(w=0.5)
[2026-06-20 00:59:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 999): 12688.6 MiB
[2026-06-20 01:00:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2860, pooled_mean_dice=0.8104, per_class=['0.8104'], iou=0.6813, recall=0.9820, precision=0.6899, vol_sim=0.8253, mcc=0.8185, min_class_dice=0.8104, coverage=[75]/88 samples
[2026-06-20 01:00:55] INFO segtask_v1.trainer.trainer: Epoch 1000/1000 | LR=4.71e-04 | loss=0.2339 | val_dice=0.8104 | best=0.8292 (ep441) | 11:31:27 | L_main=0.1190 L_aux_1=0.1125(w=0.5) L_aux_2=0.1171(w=0.5)
[2026-06-20 01:00:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1000): 12688.6 MiB
[2026-06-20 01:00:56] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-20 01:00:56] INFO segtask_v1.trainer.trainer: Training complete. Best mean_dice=0.8292 at epoch 441. Time: 11:31:27
[2026-06-20 01:00:56] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-20 01:00:56] INFO __main__: Best metrics: {'val_loss': 0.3358045464212244, 'dice_class_0': 0.8291603922843933, 'iou_class_0': 0.7081758379936218, 'recall_class_0': 0.9820460081100464, 'precision_class_0': 0.7174649238586426, 'vol_sim_class_0': 0.844319224357605, 'mcc_class_0': 0.8342822194099426, 'mean_dice': 0.8291603922843933, 'mean_iou': 0.7081758379936218, 'mean_recall': 0.9820460081100464, 'mean_precision': 0.7174649238586426, 'mean_vol_sim': 0.844319224357605, 'mean_mcc': 0.8342822194099426, 'min_class_dice': 0.8291603922843933, 'min_class_iou': 0.7081758379936218}


我用到D:\codes\work-projects\SegTask\configs\lungves1.yaml训练：
[2026-06-18 15:12:41] INFO __main__: Config loaded from: configs/segtest1.yaml
[2026-06-18 15:12:41] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-18 15:12:41] INFO __main__: Device: cuda
[2026-06-18 15:12:41] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-18 15:12:41] INFO segtask_v1.data.loader: Training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-18 15:12:41] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-06-18 15:12:41] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-18 15:12:56] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-06-18 15:12:56] INFO segtask_v1.data.specs: Using CUBIC patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], max_scale=2.00) — SINGLE max-FOV cube extraction; trainer crops+resizes per view before the 3D forward.
[2026-06-18 15:12:56] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 88 npz packages...
[2026-06-18 15:13:32] INFO segtask_v1.data.dataset: NPZ cubic index: 88 volumes, 4400000 fg voxels sampled
[2026-06-18 15:13:32] INFO segtask_v1.data.dataset: Loading pre-computed fg coords from 22 npz packages...
[2026-06-18 15:13:42] INFO segtask_v1.data.dataset: NPZ cubic index: 22 volumes, 1100000 fg voxels sampled
[2026-06-18 15:13:42] INFO segtask_v1.data.loader: DataLoader: batch_size=2, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-18 15:13:42] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=24, num_workers=16 => up to ~87.40 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker).
[2026-06-18 15:13:42] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=34.48M, dec=17.11M, total=53.91M, channels=[64, 64, 128, 256, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=3 (fg=1, res=3), stem=dual(stride=1, n_views=1, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=False(n_aux_heads=0, mode=conv)
[2026-06-18 15:13:43] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Patch3DNativeMultiResPipeline (patch_mode=cubic, n_views=3)
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: Validation metric mode: medium (evaluator=PatchValEvaluator)
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: Training: 400 epochs, device=cuda
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: Model params: 53.91M
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=205.7 + grad=205.7 + optim(AdamW,2x)=411.3 + ema=205.7 = 1028.3 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=420.5 MiB, reserved=442.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: Train batches: 352, Val batches: 44
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: AMP=False (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: Grad accum=4, Effective batch=8
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: Pipeline=Patch3DNativeMultiResPipeline | n_views=3, n_aux_views=0, num_res_groups=3, slab_depth=0 | fg_classes=1, Loss=dice_focal
[2026-06-18 15:13:43] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-18 15:14:23] INFO segtask_v1.trainer.trainer: Actual one-step GPU peak: 19927.0 MiB (forward + backward + optimizer.step + EMA update; accum=4 micro-batches). Steady-state training peak should stay close to this; the full-epoch peak is reported separately at end of each epoch as 'GPU peak (epoch N)'.
[2026-06-18 15:25:10] INFO segtask_v1.trainer.validation:   Val: loss=0.8433, pooled_mean_dice=0.2921, per_class=['0.2921'], iou=0.1711, recall=1.0000, precision=0.1711, vol_sim=0.2921, mcc=0.0000, min_class_dice=0.2921, coverage=[88]/88 samples
[2026-06-18 15:25:11] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 15:25:11] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.2921 at epoch 1
[2026-06-18 15:25:11] INFO segtask_v1.trainer.trainer: Epoch 1/400 | LR=5.10e-05 | loss=0.3912 | val_dice=0.2921 | best=0.2921 (ep1) | 00:11:27 | L_res_0=0.3385 L_res_1=0.3735 L_res_2=0.4812
[2026-06-18 15:25:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 1): 19927.0 MiB
[2026-06-18 15:35:58] INFO segtask_v1.trainer.validation:   Val: loss=0.8321, pooled_mean_dice=0.3098, per_class=['0.3098'], iou=0.1833, recall=1.0000, precision=0.1833, vol_sim=0.3098, mcc=0.0000, min_class_dice=0.3098, coverage=[88]/88 samples
[2026-06-18 15:36:04] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 15:36:04] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.3098 at epoch 2
[2026-06-18 15:36:04] INFO segtask_v1.trainer.trainer: Epoch 2/400 | LR=1.01e-04 | loss=0.2360 | val_dice=0.3098 | best=0.3098 (ep2) | 00:22:20 | L_res_0=0.1776 L_res_1=0.2454 L_res_2=0.3460
[2026-06-18 15:36:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 2): 19517.6 MiB
[2026-06-18 15:46:52] INFO segtask_v1.trainer.validation:   Val: loss=0.8473, pooled_mean_dice=0.2883, per_class=['0.2883'], iou=0.1684, recall=1.0000, precision=0.1684, vol_sim=0.2883, mcc=0.0000, min_class_dice=0.2883, coverage=[88]/88 samples
[2026-06-18 15:46:52] INFO segtask_v1.trainer.trainer: Epoch 3/400 | LR=1.51e-04 | loss=0.1967 | val_dice=0.2883 | best=0.3098 (ep2) | 00:33:08 | L_res_0=0.1583 L_res_1=0.2102 L_res_2=0.2939
[2026-06-18 15:46:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 3): 19516.7 MiB
[2026-06-18 15:57:37] INFO segtask_v1.trainer.validation:   Val: loss=0.8267, pooled_mean_dice=0.3125, per_class=['0.3125'], iou=0.1852, recall=1.0000, precision=0.1852, vol_sim=0.3125, mcc=0.0000, min_class_dice=0.3125, coverage=[88]/88 samples
[2026-06-18 15:57:43] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 15:57:43] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.3125 at epoch 4
[2026-06-18 15:57:43] INFO segtask_v1.trainer.trainer: Epoch 4/400 | LR=2.01e-04 | loss=0.1693 | val_dice=0.3125 | best=0.3125 (ep4) | 00:43:59 | L_res_0=0.1461 L_res_1=0.1857 L_res_2=0.2503
[2026-06-18 15:57:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 4): 19516.3 MiB
[2026-06-18 16:08:29] INFO segtask_v1.trainer.validation:   Val: loss=0.8506, pooled_mean_dice=0.2797, per_class=['0.2797'], iou=0.1626, recall=0.9999, precision=0.1626, vol_sim=0.2797, mcc=0.0223, min_class_dice=0.2797, coverage=[88]/88 samples
[2026-06-18 16:08:29] INFO segtask_v1.trainer.trainer: Epoch 5/400 | LR=2.51e-04 | loss=0.1468 | val_dice=0.2797 | best=0.3125 (ep4) | 00:54:45 | L_res_0=0.1364 L_res_1=0.1639 L_res_2=0.2121
[2026-06-18 16:08:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 5): 19516.7 MiB
[2026-06-18 16:19:15] INFO segtask_v1.trainer.validation:   Val: loss=0.8309, pooled_mean_dice=0.3663, per_class=['0.3663'], iou=0.2242, recall=0.8457, precision=0.2338, vol_sim=0.4332, mcc=0.1969, min_class_dice=0.3663, coverage=[88]/88 samples
[2026-06-18 16:19:21] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 16:19:21] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.3663 at epoch 6
[2026-06-18 16:19:21] INFO segtask_v1.trainer.trainer: Epoch 6/400 | LR=3.01e-04 | loss=0.1280 | val_dice=0.3663 | best=0.3663 (ep6) | 01:05:37 | L_res_0=0.1268 L_res_1=0.1402 L_res_2=0.1742
[2026-06-18 16:19:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 6): 19516.2 MiB
[2026-06-18 16:30:07] INFO segtask_v1.trainer.validation:   Val: loss=0.8382, pooled_mean_dice=0.1292, per_class=['0.1292'], iou=0.0691, recall=0.1177, precision=0.1432, vol_sim=0.9021, mcc=-0.0240, min_class_dice=0.1292, coverage=[88]/88 samples
[2026-06-18 16:30:07] INFO segtask_v1.trainer.trainer: Epoch 7/400 | LR=3.51e-04 | loss=0.1263 | val_dice=0.1292 | best=0.3663 (ep6) | 01:16:23 | L_res_0=0.1345 L_res_1=0.1303 L_res_2=0.1476
[2026-06-18 16:30:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 7): 19516.3 MiB
[2026-06-18 16:40:51] INFO segtask_v1.trainer.validation:   Val: loss=0.8134, pooled_mean_dice=0.0543, per_class=['0.0543'], iou=0.0279, recall=0.0337, precision=0.1409, vol_sim=0.3858, mcc=-0.0247, min_class_dice=0.0543, coverage=[88]/88 samples
[2026-06-18 16:40:51] INFO segtask_v1.trainer.trainer: Epoch 8/400 | LR=4.01e-04 | loss=0.1076 | val_dice=0.0543 | best=0.3663 (ep6) | 01:27:07 | L_res_0=0.1176 L_res_1=0.1089 L_res_2=0.1229
[2026-06-18 16:40:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 8): 19516.3 MiB
[2026-06-18 16:51:33] INFO segtask_v1.trainer.validation:   Val: loss=0.8245, pooled_mean_dice=0.0281, per_class=['0.0281'], iou=0.0143, recall=0.0150, precision=0.2192, vol_sim=0.1284, mcc=0.0151, min_class_dice=0.0281, coverage=[88]/88 samples
[2026-06-18 16:51:33] INFO segtask_v1.trainer.trainer: Epoch 9/400 | LR=4.51e-04 | loss=0.0924 | val_dice=0.0281 | best=0.3663 (ep6) | 01:37:49 | L_res_0=0.1014 L_res_1=0.0934 L_res_2=0.1044
[2026-06-18 16:51:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 9): 19516.3 MiB
[2026-06-18 17:02:18] INFO segtask_v1.trainer.validation:   Val: loss=0.8337, pooled_mean_dice=0.0032, per_class=['0.0032'], iou=0.0016, recall=0.0016, precision=0.1371, vol_sim=0.0233, mcc=-0.0032, min_class_dice=0.0032, coverage=[88]/88 samples
[2026-06-18 17:02:18] INFO segtask_v1.trainer.trainer: Epoch 10/400 | LR=5.01e-04 | loss=0.0830 | val_dice=0.0032 | best=0.3663 (ep6) | 01:48:35 | L_res_0=0.0920 L_res_1=0.0849 L_res_2=0.0919
[2026-06-18 17:02:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 10): 19517.2 MiB
[2026-06-18 17:13:06] INFO segtask_v1.trainer.validation:   Val: loss=0.8342, pooled_mean_dice=0.0004, per_class=['0.0004'], iou=0.0002, recall=0.0002, precision=0.1278, vol_sim=0.0033, mcc=-0.0020, min_class_dice=0.0004, coverage=[88]/88 samples
[2026-06-18 17:13:06] INFO segtask_v1.trainer.trainer: Epoch 11/400 | LR=5.50e-04 | loss=0.0833 | val_dice=0.0004 | best=0.3663 (ep6) | 01:59:22 | L_res_0=0.0926 L_res_1=0.0841 L_res_2=0.0906
[2026-06-18 17:13:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 11): 19516.2 MiB
[2026-06-18 17:23:49] INFO segtask_v1.trainer.validation:   Val: loss=0.8374, pooled_mean_dice=0.0002, per_class=['0.0002'], iou=0.0001, recall=0.0001, precision=0.1450, vol_sim=0.0014, mcc=-0.0009, min_class_dice=0.0002, coverage=[88]/88 samples
[2026-06-18 17:23:49] INFO segtask_v1.trainer.trainer: Epoch 12/400 | LR=6.00e-04 | loss=0.0985 | val_dice=0.0002 | best=0.3663 (ep6) | 02:10:05 | L_res_0=0.1099 L_res_1=0.0999 L_res_2=0.1088
[2026-06-18 17:23:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 12): 19517.0 MiB
[2026-06-18 17:34:38] INFO segtask_v1.trainer.validation:   Val: loss=0.8355, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.1802, vol_sim=0.0002, mcc=-0.0000, min_class_dice=0.0000, coverage=[88]/88 samples
[2026-06-18 17:34:38] INFO segtask_v1.trainer.trainer: Epoch 13/400 | LR=6.50e-04 | loss=0.0924 | val_dice=0.0000 | best=0.3663 (ep6) | 02:20:54 | L_res_0=0.1027 L_res_1=0.0945 L_res_2=0.1003
[2026-06-18 17:34:38] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 13): 19517.0 MiB
[2026-06-18 17:45:28] INFO segtask_v1.trainer.validation:   Val: loss=0.8406, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.0848, vol_sim=0.0000, mcc=-0.0004, min_class_dice=0.0000, coverage=[88]/88 samples
[2026-06-18 17:45:28] INFO segtask_v1.trainer.trainer: Epoch 14/400 | LR=7.00e-04 | loss=0.0872 | val_dice=0.0000 | best=0.3663 (ep6) | 02:31:44 | L_res_0=0.0973 L_res_1=0.0882 L_res_2=0.0959
[2026-06-18 17:45:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 14): 19516.2 MiB
[2026-06-18 17:56:13] INFO segtask_v1.trainer.validation:   Val: loss=0.8364, pooled_mean_dice=0.0000, per_class=['0.0000'], iou=0.0000, recall=0.0000, precision=0.7821, vol_sim=0.0000, mcc=0.0033, min_class_dice=0.0000, coverage=[88]/88 samples
[2026-06-18 17:56:13] INFO segtask_v1.trainer.trainer: Epoch 15/400 | LR=7.50e-04 | loss=0.0741 | val_dice=0.0000 | best=0.3663 (ep6) | 02:42:29 | L_res_0=0.0799 L_res_1=0.0761 L_res_2=0.0823
[2026-06-18 17:56:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 15): 19515.8 MiB
[2026-06-18 18:06:58] INFO segtask_v1.trainer.validation:   Val: loss=0.8555, pooled_mean_dice=0.0001, per_class=['0.0001'], iou=0.0000, recall=0.0000, precision=0.9574, vol_sim=0.0001, mcc=0.0053, min_class_dice=0.0001, coverage=[88]/88 samples
[2026-06-18 18:06:58] INFO segtask_v1.trainer.trainer: Epoch 16/400 | LR=8.00e-04 | loss=0.0689 | val_dice=0.0001 | best=0.3663 (ep6) | 02:53:15 | L_res_0=0.0730 L_res_1=0.0711 L_res_2=0.0763
[2026-06-18 18:06:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 16): 19516.1 MiB
[2026-06-18 18:17:47] INFO segtask_v1.trainer.validation:   Val: loss=0.8760, pooled_mean_dice=0.0005, per_class=['0.0005'], iou=0.0002, recall=0.0002, precision=0.9775, vol_sim=0.0005, mcc=0.0140, min_class_dice=0.0005, coverage=[88]/88 samples
[2026-06-18 18:17:47] INFO segtask_v1.trainer.trainer: Epoch 17/400 | LR=8.50e-04 | loss=0.0705 | val_dice=0.0005 | best=0.3663 (ep6) | 03:04:03 | L_res_0=0.0747 L_res_1=0.0718 L_res_2=0.0783
[2026-06-18 18:17:47] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 17): 19515.9 MiB
[2026-06-18 18:28:37] INFO segtask_v1.trainer.validation:   Val: loss=0.9003, pooled_mean_dice=0.0021, per_class=['0.0021'], iou=0.0010, recall=0.0010, precision=0.9973, vol_sim=0.0021, mcc=0.0295, min_class_dice=0.0021, coverage=[88]/88 samples
[2026-06-18 18:28:37] INFO segtask_v1.trainer.trainer: Epoch 18/400 | LR=9.00e-04 | loss=0.0775 | val_dice=0.0021 | best=0.3663 (ep6) | 03:14:53 | L_res_0=0.0844 L_res_1=0.0784 L_res_2=0.0860
[2026-06-18 18:28:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 18): 19516.2 MiB
[2026-06-18 18:39:23] INFO segtask_v1.trainer.validation:   Val: loss=0.9021, pooled_mean_dice=0.0097, per_class=['0.0097'], iou=0.0049, recall=0.0049, precision=0.9889, vol_sim=0.0099, mcc=0.0627, min_class_dice=0.0097, coverage=[88]/88 samples
[2026-06-18 18:39:23] INFO segtask_v1.trainer.trainer: Epoch 19/400 | LR=9.50e-04 | loss=0.0867 | val_dice=0.0097 | best=0.3663 (ep6) | 03:25:40 | L_res_0=0.0946 L_res_1=0.0890 L_res_2=0.0951
[2026-06-18 18:39:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 19): 19517.1 MiB
[2026-06-18 18:50:04] INFO segtask_v1.trainer.validation:   Val: loss=0.9031, pooled_mean_dice=0.0335, per_class=['0.0335'], iou=0.0170, recall=0.0170, precision=0.9832, vol_sim=0.0341, mcc=0.1176, min_class_dice=0.0335, coverage=[88]/88 samples
[2026-06-18 18:50:04] INFO segtask_v1.trainer.trainer: Epoch 20/400 | LR=1.00e-03 | loss=0.0734 | val_dice=0.0335 | best=0.3663 (ep6) | 03:36:20 | L_res_0=0.0784 L_res_1=0.0757 L_res_2=0.0824
[2026-06-18 18:50:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 20): 19516.4 MiB
[2026-06-18 19:00:51] INFO segtask_v1.trainer.validation:   Val: loss=0.9459, pooled_mean_dice=0.0367, per_class=['0.0367'], iou=0.0187, recall=0.0187, precision=0.9773, vol_sim=0.0376, mcc=0.1218, min_class_dice=0.0367, coverage=[88]/88 samples
[2026-06-18 19:00:51] INFO segtask_v1.trainer.trainer: Epoch 21/400 | LR=1.00e-03 | loss=0.0664 | val_dice=0.0367 | best=0.3663 (ep6) | 03:47:07 | L_res_0=0.0699 L_res_1=0.0674 L_res_2=0.0749
[2026-06-18 19:00:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 21): 19516.7 MiB
[2026-06-18 19:11:35] INFO segtask_v1.trainer.validation:   Val: loss=0.9383, pooled_mean_dice=0.0657, per_class=['0.0657'], iou=0.0340, recall=0.0340, precision=0.9631, vol_sim=0.0682, mcc=0.1626, min_class_dice=0.0657, coverage=[88]/88 samples
[2026-06-18 19:11:35] INFO segtask_v1.trainer.trainer: Epoch 22/400 | LR=1.00e-03 | loss=0.0698 | val_dice=0.0657 | best=0.3663 (ep6) | 03:57:51 | L_res_0=0.0734 L_res_1=0.0708 L_res_2=0.0778
[2026-06-18 19:11:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 22): 19515.6 MiB
[2026-06-18 19:22:19] INFO segtask_v1.trainer.validation:   Val: loss=0.9576, pooled_mean_dice=0.0747, per_class=['0.0747'], iou=0.0388, recall=0.0389, precision=0.9425, vol_sim=0.0793, mcc=0.1705, min_class_dice=0.0747, coverage=[88]/88 samples
[2026-06-18 19:22:19] INFO segtask_v1.trainer.trainer: Epoch 23/400 | LR=1.00e-03 | loss=0.0624 | val_dice=0.0747 | best=0.3663 (ep6) | 04:08:36 | L_res_0=0.0641 L_res_1=0.0630 L_res_2=0.0700
[2026-06-18 19:22:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 23): 19517.6 MiB
[2026-06-18 19:33:03] INFO segtask_v1.trainer.validation:   Val: loss=0.9054, pooled_mean_dice=0.1382, per_class=['0.1382'], iou=0.0742, recall=0.0749, precision=0.8988, vol_sim=0.1538, mcc=0.2318, min_class_dice=0.1382, coverage=[88]/88 samples
[2026-06-18 19:33:03] INFO segtask_v1.trainer.trainer: Epoch 24/400 | LR=1.00e-03 | loss=0.0801 | val_dice=0.1382 | best=0.3663 (ep6) | 04:19:19 | L_res_0=0.0846 L_res_1=0.0815 L_res_2=0.0892
[2026-06-18 19:33:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 24): 19515.6 MiB
[2026-06-18 19:43:50] INFO segtask_v1.trainer.validation:   Val: loss=0.9248, pooled_mean_dice=0.1570, per_class=['0.1570'], iou=0.0852, recall=0.0863, precision=0.8692, vol_sim=0.1806, mcc=0.2422, min_class_dice=0.1570, coverage=[88]/88 samples
[2026-06-18 19:43:50] INFO segtask_v1.trainer.trainer: Epoch 25/400 | LR=1.00e-03 | loss=0.0608 | val_dice=0.1570 | best=0.3663 (ep6) | 04:30:06 | L_res_0=0.0634 L_res_1=0.0624 L_res_2=0.0685
[2026-06-18 19:43:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 25): 19515.4 MiB
[2026-06-18 19:54:35] INFO segtask_v1.trainer.validation:   Val: loss=0.8996, pooled_mean_dice=0.2094, per_class=['0.2094'], iou=0.1170, recall=0.1192, precision=0.8638, vol_sim=0.2424, mcc=0.2839, min_class_dice=0.2094, coverage=[88]/88 samples
[2026-06-18 19:54:35] INFO segtask_v1.trainer.trainer: Epoch 26/400 | LR=1.00e-03 | loss=0.0550 | val_dice=0.2094 | best=0.3663 (ep6) | 04:40:51 | L_res_0=0.0551 L_res_1=0.0564 L_res_2=0.0631
[2026-06-18 19:54:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 26): 19517.1 MiB
[2026-06-18 20:05:20] INFO segtask_v1.trainer.validation:   Val: loss=0.8763, pooled_mean_dice=0.2597, per_class=['0.2597'], iou=0.1493, recall=0.1532, precision=0.8522, vol_sim=0.3048, mcc=0.3208, min_class_dice=0.2597, coverage=[88]/88 samples
[2026-06-18 20:05:20] INFO segtask_v1.trainer.trainer: Epoch 27/400 | LR=1.00e-03 | loss=0.0668 | val_dice=0.2597 | best=0.3663 (ep6) | 04:51:36 | L_res_0=0.0712 L_res_1=0.0686 L_res_2=0.0720
[2026-06-18 20:05:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 27): 19516.4 MiB
[2026-06-18 20:16:01] INFO segtask_v1.trainer.validation:   Val: loss=0.8510, pooled_mean_dice=0.3107, per_class=['0.3107'], iou=0.1839, recall=0.1903, precision=0.8453, vol_sim=0.3675, mcc=0.3593, min_class_dice=0.3107, coverage=[88]/88 samples
[2026-06-18 20:16:01] INFO segtask_v1.trainer.trainer: Epoch 28/400 | LR=1.00e-03 | loss=0.0591 | val_dice=0.3107 | best=0.3663 (ep6) | 05:02:17 | L_res_0=0.0608 L_res_1=0.0609 L_res_2=0.0675
[2026-06-18 20:16:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 28): 19516.6 MiB
[2026-06-18 20:26:47] INFO segtask_v1.trainer.validation:   Val: loss=0.7349, pooled_mean_dice=0.4681, per_class=['0.4681'], iou=0.3055, recall=0.3236, precision=0.8456, vol_sim=0.5535, mcc=0.4733, min_class_dice=0.4681, coverage=[88]/88 samples
[2026-06-18 20:26:53] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 20:26:53] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.4681 at epoch 29
[2026-06-18 20:26:53] INFO segtask_v1.trainer.trainer: Epoch 29/400 | LR=1.00e-03 | loss=0.0605 | val_dice=0.4681 | best=0.4681 (ep29) | 05:13:09 | L_res_0=0.0620 L_res_1=0.0628 L_res_2=0.0682
[2026-06-18 20:26:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 29): 19516.8 MiB
[2026-06-18 20:37:36] INFO segtask_v1.trainer.validation:   Val: loss=0.7425, pooled_mean_dice=0.4819, per_class=['0.4819'], iou=0.3174, recall=0.3393, precision=0.8315, vol_sim=0.5796, mcc=0.4822, min_class_dice=0.4819, coverage=[88]/88 samples
[2026-06-18 20:37:42] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 20:37:42] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.4819 at epoch 30
[2026-06-18 20:37:42] INFO segtask_v1.trainer.trainer: Epoch 30/400 | LR=1.00e-03 | loss=0.0598 | val_dice=0.4819 | best=0.4819 (ep30) | 05:23:58 | L_res_0=0.0627 L_res_1=0.0619 L_res_2=0.0669
[2026-06-18 20:37:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 30): 19516.3 MiB
[2026-06-18 20:48:28] INFO segtask_v1.trainer.validation:   Val: loss=0.6902, pooled_mean_dice=0.5423, per_class=['0.5423'], iou=0.3721, recall=0.4057, precision=0.8176, vol_sim=0.6634, mcc=0.5259, min_class_dice=0.5423, coverage=[88]/88 samples
[2026-06-18 20:48:34] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 20:48:34] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.5423 at epoch 31
[2026-06-18 20:48:34] INFO segtask_v1.trainer.trainer: Epoch 31/400 | LR=1.00e-03 | loss=0.0523 | val_dice=0.5423 | best=0.5423 (ep31) | 05:34:51 | L_res_0=0.0520 L_res_1=0.0541 L_res_2=0.0589
[2026-06-18 20:48:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 31): 19517.3 MiB
[2026-06-18 20:59:19] INFO segtask_v1.trainer.validation:   Val: loss=0.5841, pooled_mean_dice=0.6672, per_class=['0.6672'], iou=0.5006, recall=0.5645, precision=0.8157, vol_sim=0.8180, mcc=0.6259, min_class_dice=0.6672, coverage=[88]/88 samples
[2026-06-18 20:59:25] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 20:59:25] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.6672 at epoch 32
[2026-06-18 20:59:25] INFO segtask_v1.trainer.trainer: Epoch 32/400 | LR=1.00e-03 | loss=0.0589 | val_dice=0.6672 | best=0.6672 (ep32) | 05:45:41 | L_res_0=0.0613 L_res_1=0.0615 L_res_2=0.0663
[2026-06-18 20:59:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 32): 19517.1 MiB
[2026-06-18 21:10:07] INFO segtask_v1.trainer.validation:   Val: loss=0.4858, pooled_mean_dice=0.7280, per_class=['0.7280'], iou=0.5723, recall=0.6683, precision=0.7995, vol_sim=0.9106, mcc=0.6786, min_class_dice=0.7280, coverage=[88]/88 samples
[2026-06-18 21:10:13] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 21:10:13] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.7280 at epoch 33
[2026-06-18 21:10:13] INFO segtask_v1.trainer.trainer: Epoch 33/400 | LR=1.00e-03 | loss=0.0561 | val_dice=0.7280 | best=0.7280 (ep33) | 05:56:29 | L_res_0=0.0577 L_res_1=0.0576 L_res_2=0.0634
[2026-06-18 21:10:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 33): 19516.6 MiB
[2026-06-18 21:20:55] INFO segtask_v1.trainer.validation:   Val: loss=0.3945, pooled_mean_dice=0.7940, per_class=['0.7940'], iou=0.6584, recall=0.7951, precision=0.7929, vol_sim=0.9987, mcc=0.7467, min_class_dice=0.7940, coverage=[88]/88 samples
[2026-06-18 21:21:01] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 21:21:01] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.7940 at epoch 34
[2026-06-18 21:21:01] INFO segtask_v1.trainer.trainer: Epoch 34/400 | LR=1.00e-03 | loss=0.0510 | val_dice=0.7940 | best=0.7940 (ep34) | 06:07:17 | L_res_0=0.0504 L_res_1=0.0528 L_res_2=0.0577
[2026-06-18 21:21:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 34): 19517.0 MiB
[2026-06-18 21:31:46] INFO segtask_v1.trainer.validation:   Val: loss=0.3791, pooled_mean_dice=0.8012, per_class=['0.8012'], iou=0.6684, recall=0.8396, precision=0.7662, vol_sim=0.9543, mcc=0.7591, min_class_dice=0.8012, coverage=[88]/88 samples
[2026-06-18 21:31:52] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 21:31:52] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8012 at epoch 35
[2026-06-18 21:31:52] INFO segtask_v1.trainer.trainer: Epoch 35/400 | LR=1.00e-03 | loss=0.0506 | val_dice=0.8012 | best=0.8012 (ep35) | 06:18:08 | L_res_0=0.0516 L_res_1=0.0515 L_res_2=0.0564
[2026-06-18 21:31:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 35): 19516.6 MiB
[2026-06-18 21:42:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3496, pooled_mean_dice=0.8034, per_class=['0.8034'], iou=0.6714, recall=0.8449, precision=0.7658, vol_sim=0.9508, mcc=0.7568, min_class_dice=0.8034, coverage=[88]/88 samples
[2026-06-18 21:42:43] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 21:42:43] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8034 at epoch 36
[2026-06-18 21:42:43] INFO segtask_v1.trainer.trainer: Epoch 36/400 | LR=1.00e-03 | loss=0.0503 | val_dice=0.8034 | best=0.8034 (ep36) | 06:28:59 | L_res_0=0.0510 L_res_1=0.0510 L_res_2=0.0561
[2026-06-18 21:42:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 36): 19516.1 MiB
[2026-06-18 21:53:23] INFO segtask_v1.trainer.validation:   Val: loss=0.3061, pooled_mean_dice=0.8235, per_class=['0.8235'], iou=0.6999, recall=0.9103, precision=0.7518, vol_sim=0.9046, mcc=0.7861, min_class_dice=0.8235, coverage=[88]/88 samples
[2026-06-18 21:53:29] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 21:53:29] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8235 at epoch 37
[2026-06-18 21:53:29] INFO segtask_v1.trainer.trainer: Epoch 37/400 | LR=1.00e-03 | loss=0.0478 | val_dice=0.8235 | best=0.8235 (ep37) | 06:39:45 | L_res_0=0.0464 L_res_1=0.0495 L_res_2=0.0540
[2026-06-18 21:53:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 37): 19516.4 MiB
[2026-06-18 22:04:13] INFO segtask_v1.trainer.validation:   Val: loss=0.2992, pooled_mean_dice=0.8196, per_class=['0.8196'], iou=0.6943, recall=0.9231, precision=0.7369, vol_sim=0.8878, mcc=0.7842, min_class_dice=0.8196, coverage=[88]/88 samples
[2026-06-18 22:04:13] INFO segtask_v1.trainer.trainer: Epoch 38/400 | LR=1.00e-03 | loss=0.0651 | val_dice=0.8196 | best=0.8235 (ep37) | 06:50:29 | L_res_0=0.0667 L_res_1=0.0677 L_res_2=0.0712
[2026-06-18 22:04:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 38): 19516.4 MiB
[2026-06-18 22:14:58] INFO segtask_v1.trainer.validation:   Val: loss=0.3211, pooled_mean_dice=0.8147, per_class=['0.8147'], iou=0.6874, recall=0.8877, precision=0.7529, vol_sim=0.9178, mcc=0.7743, min_class_dice=0.8147, coverage=[88]/88 samples
[2026-06-18 22:14:58] INFO segtask_v1.trainer.trainer: Epoch 39/400 | LR=1.00e-03 | loss=0.0773 | val_dice=0.8147 | best=0.8235 (ep37) | 07:01:14 | L_res_0=0.0839 L_res_1=0.0791 L_res_2=0.0855
[2026-06-18 22:14:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 39): 19516.4 MiB
[2026-06-18 22:25:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3411, pooled_mean_dice=0.8081, per_class=['0.8081'], iou=0.6780, recall=0.8918, precision=0.7388, vol_sim=0.9062, mcc=0.7695, min_class_dice=0.8081, coverage=[88]/88 samples
[2026-06-18 22:25:41] INFO segtask_v1.trainer.trainer: Epoch 40/400 | LR=1.00e-03 | loss=0.0689 | val_dice=0.8081 | best=0.8235 (ep37) | 07:11:57 | L_res_0=0.0713 L_res_1=0.0713 L_res_2=0.0787
[2026-06-18 22:25:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 40): 19516.2 MiB
[2026-06-18 22:36:24] INFO segtask_v1.trainer.validation:   Val: loss=0.3570, pooled_mean_dice=0.8046, per_class=['0.8046'], iou=0.6731, recall=0.8622, precision=0.7543, vol_sim=0.9333, mcc=0.7621, min_class_dice=0.8046, coverage=[88]/88 samples
[2026-06-18 22:36:24] INFO segtask_v1.trainer.trainer: Epoch 41/400 | LR=1.00e-03 | loss=0.0579 | val_dice=0.8046 | best=0.8235 (ep37) | 07:22:40 | L_res_0=0.0578 L_res_1=0.0603 L_res_2=0.0668
[2026-06-18 22:36:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 41): 19516.2 MiB
[2026-06-18 22:47:07] INFO segtask_v1.trainer.validation:   Val: loss=0.3183, pooled_mean_dice=0.8196, per_class=['0.8196'], iou=0.6944, recall=0.8882, precision=0.7609, vol_sim=0.9228, mcc=0.7806, min_class_dice=0.8196, coverage=[88]/88 samples
[2026-06-18 22:47:07] INFO segtask_v1.trainer.trainer: Epoch 42/400 | LR=1.00e-03 | loss=0.0541 | val_dice=0.8196 | best=0.8235 (ep37) | 07:33:23 | L_res_0=0.0569 L_res_1=0.0557 L_res_2=0.0601
[2026-06-18 22:47:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 42): 19516.4 MiB
[2026-06-18 22:57:49] INFO segtask_v1.trainer.validation:   Val: loss=0.3043, pooled_mean_dice=0.8254, per_class=['0.8254'], iou=0.7027, recall=0.9214, precision=0.7476, vol_sim=0.8959, mcc=0.7899, min_class_dice=0.8254, coverage=[88]/88 samples
[2026-06-18 22:57:53] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 22:57:53] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8254 at epoch 43
[2026-06-18 22:57:53] INFO segtask_v1.trainer.trainer: Epoch 43/400 | LR=9.99e-04 | loss=0.0532 | val_dice=0.8254 | best=0.8254 (ep43) | 07:44:10 | L_res_0=0.0537 L_res_1=0.0549 L_res_2=0.0604
[2026-06-18 22:57:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 43): 19516.2 MiB
[2026-06-18 23:08:37] INFO segtask_v1.trainer.validation:   Val: loss=0.3213, pooled_mean_dice=0.8101, per_class=['0.8101'], iou=0.6808, recall=0.8919, precision=0.7421, vol_sim=0.9084, mcc=0.7735, min_class_dice=0.8101, coverage=[88]/88 samples
[2026-06-18 23:08:37] INFO segtask_v1.trainer.trainer: Epoch 44/400 | LR=9.99e-04 | loss=0.0512 | val_dice=0.8101 | best=0.8254 (ep43) | 07:54:53 | L_res_0=0.0519 L_res_1=0.0528 L_res_2=0.0575
[2026-06-18 23:08:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 44): 19516.9 MiB
[2026-06-18 23:19:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2592, pooled_mean_dice=0.8437, per_class=['0.8437'], iou=0.7297, recall=0.9639, precision=0.7502, vol_sim=0.8753, mcc=0.8119, min_class_dice=0.8437, coverage=[88]/88 samples
[2026-06-18 23:19:23] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-18 23:19:23] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8437 at epoch 45
[2026-06-18 23:19:23] INFO segtask_v1.trainer.trainer: Epoch 45/400 | LR=9.99e-04 | loss=0.0511 | val_dice=0.8437 | best=0.8437 (ep45) | 08:05:39 | L_res_0=0.0520 L_res_1=0.0524 L_res_2=0.0564
[2026-06-18 23:19:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 45): 19516.6 MiB
[2026-06-18 23:30:05] INFO segtask_v1.trainer.validation:   Val: loss=0.3020, pooled_mean_dice=0.8207, per_class=['0.8207'], iou=0.6960, recall=0.9349, precision=0.7314, vol_sim=0.8779, mcc=0.7870, min_class_dice=0.8207, coverage=[88]/88 samples
[2026-06-18 23:30:05] INFO segtask_v1.trainer.trainer: Epoch 46/400 | LR=9.99e-04 | loss=0.0718 | val_dice=0.8207 | best=0.8437 (ep45) | 08:16:22 | L_res_0=0.0767 L_res_1=0.0746 L_res_2=0.0812
[2026-06-18 23:30:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 46): 19516.0 MiB
[2026-06-18 23:40:48] INFO segtask_v1.trainer.validation:   Val: loss=0.3004, pooled_mean_dice=0.8299, per_class=['0.8299'], iou=0.7093, recall=0.9318, precision=0.7481, vol_sim=0.8907, mcc=0.7936, min_class_dice=0.8299, coverage=[88]/88 samples
[2026-06-18 23:40:48] INFO segtask_v1.trainer.trainer: Epoch 47/400 | LR=9.99e-04 | loss=0.0718 | val_dice=0.8299 | best=0.8437 (ep45) | 08:27:04 | L_res_0=0.0778 L_res_1=0.0742 L_res_2=0.0794
[2026-06-18 23:40:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 47): 19516.3 MiB
[2026-06-18 23:51:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2846, pooled_mean_dice=0.8303, per_class=['0.8303'], iou=0.7098, recall=0.9395, precision=0.7438, vol_sim=0.8837, mcc=0.7974, min_class_dice=0.8303, coverage=[88]/88 samples
[2026-06-18 23:51:30] INFO segtask_v1.trainer.trainer: Epoch 48/400 | LR=9.99e-04 | loss=0.0540 | val_dice=0.8303 | best=0.8437 (ep45) | 08:37:46 | L_res_0=0.0559 L_res_1=0.0565 L_res_2=0.0615
[2026-06-18 23:51:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 48): 19517.2 MiB
[2026-06-19 00:02:13] INFO segtask_v1.trainer.validation:   Val: loss=0.2921, pooled_mean_dice=0.8279, per_class=['0.8279'], iou=0.7064, recall=0.9322, precision=0.7447, vol_sim=0.8881, mcc=0.7936, min_class_dice=0.8279, coverage=[88]/88 samples
[2026-06-19 00:02:13] INFO segtask_v1.trainer.trainer: Epoch 49/400 | LR=9.99e-04 | loss=0.0499 | val_dice=0.8279 | best=0.8437 (ep45) | 08:48:29 | L_res_0=0.0505 L_res_1=0.0520 L_res_2=0.0568
[2026-06-19 00:02:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 49): 19518.3 MiB
[2026-06-19 00:12:54] INFO segtask_v1.trainer.validation:   Val: loss=0.3087, pooled_mean_dice=0.8166, per_class=['0.8166'], iou=0.6901, recall=0.9158, precision=0.7369, vol_sim=0.8917, mcc=0.7797, min_class_dice=0.8166, coverage=[88]/88 samples
[2026-06-19 00:12:54] INFO segtask_v1.trainer.trainer: Epoch 50/400 | LR=9.99e-04 | loss=0.0483 | val_dice=0.8166 | best=0.8437 (ep45) | 08:59:10 | L_res_0=0.0476 L_res_1=0.0507 L_res_2=0.0551
[2026-06-19 00:12:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 50): 19516.1 MiB
[2026-06-19 00:23:41] INFO segtask_v1.trainer.validation:   Val: loss=0.3049, pooled_mean_dice=0.8179, per_class=['0.8179'], iou=0.6919, recall=0.9269, precision=0.7318, vol_sim=0.8824, mcc=0.7832, min_class_dice=0.8179, coverage=[88]/88 samples
[2026-06-19 00:23:41] INFO segtask_v1.trainer.trainer: Epoch 51/400 | LR=9.99e-04 | loss=0.0485 | val_dice=0.8179 | best=0.8437 (ep45) | 09:09:57 | L_res_0=0.0485 L_res_1=0.0507 L_res_2=0.0547
[2026-06-19 00:23:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 51): 19517.3 MiB
[2026-06-19 00:34:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2987, pooled_mean_dice=0.8166, per_class=['0.8166'], iou=0.6901, recall=0.9433, precision=0.7200, vol_sim=0.8657, mcc=0.7840, min_class_dice=0.8166, coverage=[88]/88 samples
[2026-06-19 00:34:23] INFO segtask_v1.trainer.trainer: Epoch 52/400 | LR=9.99e-04 | loss=0.0516 | val_dice=0.8166 | best=0.8437 (ep45) | 09:20:40 | L_res_0=0.0495 L_res_1=0.0544 L_res_2=0.0596
[2026-06-19 00:34:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 52): 19516.7 MiB
[2026-06-19 00:45:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2685, pooled_mean_dice=0.8346, per_class=['0.8346'], iou=0.7162, recall=0.9627, precision=0.7367, vol_sim=0.8670, mcc=0.8044, min_class_dice=0.8346, coverage=[88]/88 samples
[2026-06-19 00:45:06] INFO segtask_v1.trainer.trainer: Epoch 53/400 | LR=9.99e-04 | loss=0.0602 | val_dice=0.8346 | best=0.8437 (ep45) | 09:31:23 | L_res_0=0.0628 L_res_1=0.0619 L_res_2=0.0689
[2026-06-19 00:45:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 53): 19517.3 MiB
[2026-06-19 00:55:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2671, pooled_mean_dice=0.8343, per_class=['0.8343'], iou=0.7157, recall=0.9740, precision=0.7297, vol_sim=0.8566, mcc=0.8037, min_class_dice=0.8343, coverage=[88]/88 samples
[2026-06-19 00:55:50] INFO segtask_v1.trainer.trainer: Epoch 54/400 | LR=9.99e-04 | loss=0.0772 | val_dice=0.8343 | best=0.8437 (ep45) | 09:42:06 | L_res_0=0.0803 L_res_1=0.0782 L_res_2=0.0871
[2026-06-19 00:55:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 54): 19516.1 MiB
[2026-06-19 01:06:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2907, pooled_mean_dice=0.8114, per_class=['0.8114'], iou=0.6827, recall=0.9795, precision=0.6926, vol_sim=0.8284, mcc=0.7859, min_class_dice=0.8114, coverage=[88]/88 samples
[2026-06-19 01:06:31] INFO segtask_v1.trainer.trainer: Epoch 55/400 | LR=9.99e-04 | loss=0.0581 | val_dice=0.8114 | best=0.8437 (ep45) | 09:52:47 | L_res_0=0.0588 L_res_1=0.0606 L_res_2=0.0670
[2026-06-19 01:06:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 55): 19517.4 MiB
[2026-06-19 01:17:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2881, pooled_mean_dice=0.8233, per_class=['0.8233'], iou=0.6996, recall=0.9755, precision=0.7121, vol_sim=0.8439, mcc=0.7939, min_class_dice=0.8233, coverage=[88]/88 samples
[2026-06-19 01:17:16] INFO segtask_v1.trainer.trainer: Epoch 56/400 | LR=9.99e-04 | loss=0.0593 | val_dice=0.8233 | best=0.8437 (ep45) | 10:03:32 | L_res_0=0.0584 L_res_1=0.0626 L_res_2=0.0699
[2026-06-19 01:17:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 56): 19517.6 MiB
[2026-06-19 01:28:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2717, pooled_mean_dice=0.8372, per_class=['0.8372'], iou=0.7201, recall=0.9790, precision=0.7314, vol_sim=0.8552, mcc=0.8067, min_class_dice=0.8372, coverage=[88]/88 samples
[2026-06-19 01:28:00] INFO segtask_v1.trainer.trainer: Epoch 57/400 | LR=9.99e-04 | loss=0.0531 | val_dice=0.8372 | best=0.8437 (ep45) | 10:14:16 | L_res_0=0.0544 L_res_1=0.0549 L_res_2=0.0608
[2026-06-19 01:28:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 57): 19516.7 MiB
[2026-06-19 01:38:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2912, pooled_mean_dice=0.8190, per_class=['0.8190'], iou=0.6934, recall=0.9764, precision=0.7053, vol_sim=0.8388, mcc=0.7913, min_class_dice=0.8190, coverage=[88]/88 samples
[2026-06-19 01:38:43] INFO segtask_v1.trainer.trainer: Epoch 58/400 | LR=9.99e-04 | loss=0.0495 | val_dice=0.8190 | best=0.8437 (ep45) | 10:25:00 | L_res_0=0.0487 L_res_1=0.0515 L_res_2=0.0576
[2026-06-19 01:38:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 58): 19516.4 MiB
[2026-06-19 01:49:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2658, pooled_mean_dice=0.8336, per_class=['0.8336'], iou=0.7147, recall=0.9829, precision=0.7237, vol_sim=0.8481, mcc=0.8067, min_class_dice=0.8336, coverage=[88]/88 samples
[2026-06-19 01:49:27] INFO segtask_v1.trainer.trainer: Epoch 59/400 | LR=9.98e-04 | loss=0.0511 | val_dice=0.8336 | best=0.8437 (ep45) | 10:35:43 | L_res_0=0.0505 L_res_1=0.0530 L_res_2=0.0579
[2026-06-19 01:49:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 59): 19516.7 MiB
[2026-06-19 02:00:13] INFO segtask_v1.trainer.validation:   Val: loss=0.2937, pooled_mean_dice=0.8182, per_class=['0.8182'], iou=0.6923, recall=0.9707, precision=0.7071, vol_sim=0.8429, mcc=0.7887, min_class_dice=0.8182, coverage=[88]/88 samples
[2026-06-19 02:00:13] INFO segtask_v1.trainer.trainer: Epoch 60/400 | LR=9.98e-04 | loss=0.0583 | val_dice=0.8182 | best=0.8437 (ep45) | 10:46:29 | L_res_0=0.0604 L_res_1=0.0612 L_res_2=0.0657
[2026-06-19 02:00:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 60): 19517.3 MiB
[2026-06-19 02:10:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2946, pooled_mean_dice=0.8150, per_class=['0.8150'], iou=0.6878, recall=0.9851, precision=0.6951, vol_sim=0.8274, mcc=0.7881, min_class_dice=0.8150, coverage=[88]/88 samples
[2026-06-19 02:10:58] INFO segtask_v1.trainer.trainer: Epoch 61/400 | LR=9.98e-04 | loss=0.0466 | val_dice=0.8150 | best=0.8437 (ep45) | 10:57:14 | L_res_0=0.0457 L_res_1=0.0489 L_res_2=0.0534
[2026-06-19 02:10:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 61): 19518.1 MiB
[2026-06-19 02:21:42] INFO segtask_v1.trainer.validation:   Val: loss=0.2717, pooled_mean_dice=0.8331, per_class=['0.8331'], iou=0.7139, recall=0.9887, precision=0.7198, vol_sim=0.8426, mcc=0.8063, min_class_dice=0.8331, coverage=[88]/88 samples
[2026-06-19 02:21:42] INFO segtask_v1.trainer.trainer: Epoch 62/400 | LR=9.98e-04 | loss=0.0483 | val_dice=0.8331 | best=0.8437 (ep45) | 11:07:58 | L_res_0=0.0462 L_res_1=0.0507 L_res_2=0.0558
[2026-06-19 02:21:42] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 62): 19517.2 MiB
[2026-06-19 02:32:24] INFO segtask_v1.trainer.validation:   Val: loss=0.2744, pooled_mean_dice=0.8303, per_class=['0.8303'], iou=0.7098, recall=0.9903, precision=0.7148, vol_sim=0.8385, mcc=0.8001, min_class_dice=0.8303, coverage=[88]/88 samples
[2026-06-19 02:32:24] INFO segtask_v1.trainer.trainer: Epoch 63/400 | LR=9.98e-04 | loss=0.0604 | val_dice=0.8303 | best=0.8437 (ep45) | 11:18:40 | L_res_0=0.0617 L_res_1=0.0632 L_res_2=0.0679
[2026-06-19 02:32:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 63): 19515.8 MiB
[2026-06-19 02:43:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2884, pooled_mean_dice=0.8192, per_class=['0.8192'], iou=0.6937, recall=0.9878, precision=0.6997, vol_sim=0.8293, mcc=0.7945, min_class_dice=0.8192, coverage=[88]/88 samples
[2026-06-19 02:43:09] INFO segtask_v1.trainer.trainer: Epoch 64/400 | LR=9.98e-04 | loss=0.0551 | val_dice=0.8192 | best=0.8437 (ep45) | 11:29:25 | L_res_0=0.0554 L_res_1=0.0576 L_res_2=0.0626
[2026-06-19 02:43:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 64): 19516.0 MiB
[2026-06-19 02:53:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2802, pooled_mean_dice=0.8225, per_class=['0.8225'], iou=0.6985, recall=0.9900, precision=0.7034, vol_sim=0.8308, mcc=0.7985, min_class_dice=0.8225, coverage=[88]/88 samples
[2026-06-19 02:53:52] INFO segtask_v1.trainer.trainer: Epoch 65/400 | LR=9.98e-04 | loss=0.0508 | val_dice=0.8225 | best=0.8437 (ep45) | 11:40:08 | L_res_0=0.0508 L_res_1=0.0530 L_res_2=0.0579
[2026-06-19 02:53:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 65): 19517.0 MiB
[2026-06-19 03:04:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2651, pooled_mean_dice=0.8259, per_class=['0.8259'], iou=0.7034, recall=0.9901, precision=0.7084, vol_sim=0.8342, mcc=0.8014, min_class_dice=0.8259, coverage=[88]/88 samples
[2026-06-19 03:04:33] INFO segtask_v1.trainer.trainer: Epoch 66/400 | LR=9.98e-04 | loss=0.0462 | val_dice=0.8259 | best=0.8437 (ep45) | 11:50:49 | L_res_0=0.0452 L_res_1=0.0485 L_res_2=0.0525
[2026-06-19 03:04:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 66): 19516.4 MiB
[2026-06-19 03:15:16] INFO segtask_v1.trainer.validation:   Val: loss=0.2718, pooled_mean_dice=0.8319, per_class=['0.8319'], iou=0.7122, recall=0.9909, precision=0.7169, vol_sim=0.8396, mcc=0.8070, min_class_dice=0.8319, coverage=[88]/88 samples
[2026-06-19 03:15:16] INFO segtask_v1.trainer.trainer: Epoch 67/400 | LR=9.98e-04 | loss=0.0472 | val_dice=0.8319 | best=0.8437 (ep45) | 12:01:32 | L_res_0=0.0474 L_res_1=0.0493 L_res_2=0.0536
[2026-06-19 03:15:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 67): 19517.0 MiB
[2026-06-19 03:25:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2552, pooled_mean_dice=0.8395, per_class=['0.8395'], iou=0.7234, recall=0.9865, precision=0.7307, vol_sim=0.8510, mcc=0.8119, min_class_dice=0.8395, coverage=[88]/88 samples
[2026-06-19 03:25:57] INFO segtask_v1.trainer.trainer: Epoch 68/400 | LR=9.98e-04 | loss=0.0440 | val_dice=0.8395 | best=0.8437 (ep45) | 12:12:13 | L_res_0=0.0422 L_res_1=0.0459 L_res_2=0.0502
[2026-06-19 03:25:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 68): 19516.6 MiB
[2026-06-19 03:36:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2513, pooled_mean_dice=0.8429, per_class=['0.8429'], iou=0.7285, recall=0.9845, precision=0.7370, vol_sim=0.8562, mcc=0.8168, min_class_dice=0.8429, coverage=[88]/88 samples
[2026-06-19 03:36:43] INFO segtask_v1.trainer.trainer: Epoch 69/400 | LR=9.98e-04 | loss=0.0448 | val_dice=0.8429 | best=0.8437 (ep45) | 12:22:59 | L_res_0=0.0442 L_res_1=0.0462 L_res_2=0.0502
[2026-06-19 03:36:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 69): 19516.2 MiB
[2026-06-19 03:47:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2457, pooled_mean_dice=0.8444, per_class=['0.8444'], iou=0.7307, recall=0.9898, precision=0.7362, vol_sim=0.8531, mcc=0.8187, min_class_dice=0.8444, coverage=[88]/88 samples
[2026-06-19 03:47:34] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 03:47:34] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8444 at epoch 70
[2026-06-19 03:47:34] INFO segtask_v1.trainer.trainer: Epoch 70/400 | LR=9.98e-04 | loss=0.0452 | val_dice=0.8444 | best=0.8444 (ep70) | 12:33:50 | L_res_0=0.0439 L_res_1=0.0480 L_res_2=0.0516
[2026-06-19 03:47:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 70): 19517.7 MiB
[2026-06-19 03:58:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2578, pooled_mean_dice=0.8351, per_class=['0.8351'], iou=0.7168, recall=0.9911, precision=0.7215, vol_sim=0.8425, mcc=0.8110, min_class_dice=0.8351, coverage=[88]/88 samples
[2026-06-19 03:58:19] INFO segtask_v1.trainer.trainer: Epoch 71/400 | LR=9.97e-04 | loss=0.0433 | val_dice=0.8351 | best=0.8444 (ep70) | 12:44:35 | L_res_0=0.0414 L_res_1=0.0455 L_res_2=0.0496
[2026-06-19 03:58:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 71): 19516.8 MiB
[2026-06-19 04:08:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2432, pooled_mean_dice=0.8421, per_class=['0.8421'], iou=0.7273, recall=0.9901, precision=0.7327, vol_sim=0.8506, mcc=0.8164, min_class_dice=0.8421, coverage=[88]/88 samples
[2026-06-19 04:08:59] INFO segtask_v1.trainer.trainer: Epoch 72/400 | LR=9.97e-04 | loss=0.0429 | val_dice=0.8421 | best=0.8444 (ep70) | 12:55:15 | L_res_0=0.0406 L_res_1=0.0451 L_res_2=0.0489
[2026-06-19 04:08:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 72): 19516.5 MiB
[2026-06-19 04:19:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2421, pooled_mean_dice=0.8533, per_class=['0.8533'], iou=0.7441, recall=0.9919, precision=0.7486, vol_sim=0.8602, mcc=0.8279, min_class_dice=0.8533, coverage=[88]/88 samples
[2026-06-19 04:19:48] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 04:19:48] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8533 at epoch 73
[2026-06-19 04:19:48] INFO segtask_v1.trainer.trainer: Epoch 73/400 | LR=9.97e-04 | loss=0.0432 | val_dice=0.8533 | best=0.8533 (ep73) | 13:06:04 | L_res_0=0.0410 L_res_1=0.0449 L_res_2=0.0496
[2026-06-19 04:19:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 73): 19516.2 MiB
[2026-06-19 04:30:31] INFO segtask_v1.trainer.validation:   Val: loss=0.2439, pooled_mean_dice=0.8505, per_class=['0.8505'], iou=0.7400, recall=0.9916, precision=0.7446, vol_sim=0.8577, mcc=0.8245, min_class_dice=0.8505, coverage=[88]/88 samples
[2026-06-19 04:30:31] INFO segtask_v1.trainer.trainer: Epoch 74/400 | LR=9.97e-04 | loss=0.0433 | val_dice=0.8505 | best=0.8533 (ep73) | 13:16:47 | L_res_0=0.0418 L_res_1=0.0452 L_res_2=0.0491
[2026-06-19 04:30:31] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 74): 19515.9 MiB
[2026-06-19 04:41:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2426, pooled_mean_dice=0.8452, per_class=['0.8452'], iou=0.7319, recall=0.9879, precision=0.7385, vol_sim=0.8556, mcc=0.8205, min_class_dice=0.8452, coverage=[88]/88 samples
[2026-06-19 04:41:15] INFO segtask_v1.trainer.trainer: Epoch 75/400 | LR=9.97e-04 | loss=0.0559 | val_dice=0.8452 | best=0.8533 (ep73) | 13:27:32 | L_res_0=0.0570 L_res_1=0.0594 L_res_2=0.0625
[2026-06-19 04:41:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 75): 19517.3 MiB
[2026-06-19 04:51:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2479, pooled_mean_dice=0.8398, per_class=['0.8398'], iou=0.7238, recall=0.9872, precision=0.7307, vol_sim=0.8507, mcc=0.8150, min_class_dice=0.8398, coverage=[88]/88 samples
[2026-06-19 04:51:57] INFO segtask_v1.trainer.trainer: Epoch 76/400 | LR=9.97e-04 | loss=0.0668 | val_dice=0.8398 | best=0.8533 (ep73) | 13:38:13 | L_res_0=0.0725 L_res_1=0.0693 L_res_2=0.0739
[2026-06-19 04:51:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 76): 19515.9 MiB
[2026-06-19 05:02:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2406, pooled_mean_dice=0.8525, per_class=['0.8525'], iou=0.7430, recall=0.9897, precision=0.7487, vol_sim=0.8614, mcc=0.8290, min_class_dice=0.8525, coverage=[88]/88 samples
[2026-06-19 05:02:43] INFO segtask_v1.trainer.trainer: Epoch 77/400 | LR=9.97e-04 | loss=0.0621 | val_dice=0.8525 | best=0.8533 (ep73) | 13:48:59 | L_res_0=0.0648 L_res_1=0.0650 L_res_2=0.0688
[2026-06-19 05:02:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 77): 19516.1 MiB
[2026-06-19 05:13:25] INFO segtask_v1.trainer.validation:   Val: loss=0.2536, pooled_mean_dice=0.8391, per_class=['0.8391'], iou=0.7228, recall=0.9926, precision=0.7268, vol_sim=0.8454, mcc=0.8130, min_class_dice=0.8391, coverage=[88]/88 samples
[2026-06-19 05:13:25] INFO segtask_v1.trainer.trainer: Epoch 78/400 | LR=9.97e-04 | loss=0.0629 | val_dice=0.8391 | best=0.8533 (ep73) | 13:59:42 | L_res_0=0.0678 L_res_1=0.0658 L_res_2=0.0710
[2026-06-19 05:13:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 78): 19516.4 MiB
[2026-06-19 05:24:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2609, pooled_mean_dice=0.8350, per_class=['0.8350'], iou=0.7168, recall=0.9908, precision=0.7215, vol_sim=0.8427, mcc=0.8085, min_class_dice=0.8350, coverage=[88]/88 samples
[2026-06-19 05:24:09] INFO segtask_v1.trainer.trainer: Epoch 79/400 | LR=9.97e-04 | loss=0.0497 | val_dice=0.8350 | best=0.8533 (ep73) | 14:10:25 | L_res_0=0.0517 L_res_1=0.0516 L_res_2=0.0565
[2026-06-19 05:24:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 79): 19516.0 MiB
[2026-06-19 05:34:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2626, pooled_mean_dice=0.8322, per_class=['0.8322'], iou=0.7126, recall=0.9900, precision=0.7178, vol_sim=0.8406, mcc=0.8056, min_class_dice=0.8322, coverage=[88]/88 samples
[2026-06-19 05:34:55] INFO segtask_v1.trainer.trainer: Epoch 80/400 | LR=9.96e-04 | loss=0.0462 | val_dice=0.8322 | best=0.8533 (ep73) | 14:21:11 | L_res_0=0.0451 L_res_1=0.0484 L_res_2=0.0539
[2026-06-19 05:34:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 80): 19516.9 MiB
[2026-06-19 05:45:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2641, pooled_mean_dice=0.8382, per_class=['0.8382'], iou=0.7214, recall=0.9900, precision=0.7268, vol_sim=0.8467, mcc=0.8124, min_class_dice=0.8382, coverage=[88]/88 samples
[2026-06-19 05:45:37] INFO segtask_v1.trainer.trainer: Epoch 81/400 | LR=9.96e-04 | loss=0.0464 | val_dice=0.8382 | best=0.8533 (ep73) | 14:31:53 | L_res_0=0.0449 L_res_1=0.0491 L_res_2=0.0529
[2026-06-19 05:45:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 81): 19517.1 MiB
[2026-06-19 05:56:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2542, pooled_mean_dice=0.8440, per_class=['0.8440'], iou=0.7302, recall=0.9924, precision=0.7343, vol_sim=0.8505, mcc=0.8158, min_class_dice=0.8440, coverage=[88]/88 samples
[2026-06-19 05:56:23] INFO segtask_v1.trainer.trainer: Epoch 82/400 | LR=9.96e-04 | loss=0.0564 | val_dice=0.8440 | best=0.8533 (ep73) | 14:42:39 | L_res_0=0.0602 L_res_1=0.0579 L_res_2=0.0618
[2026-06-19 05:56:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 82): 19516.0 MiB
[2026-06-19 06:07:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2839, pooled_mean_dice=0.8206, per_class=['0.8206'], iou=0.6958, recall=0.9873, precision=0.7021, vol_sim=0.8312, mcc=0.7952, min_class_dice=0.8206, coverage=[88]/88 samples
[2026-06-19 06:07:07] INFO segtask_v1.trainer.trainer: Epoch 83/400 | LR=9.96e-04 | loss=0.0483 | val_dice=0.8206 | best=0.8533 (ep73) | 14:53:24 | L_res_0=0.0488 L_res_1=0.0510 L_res_2=0.0546
[2026-06-19 06:07:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 83): 19517.3 MiB
[2026-06-19 06:17:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2596, pooled_mean_dice=0.8372, per_class=['0.8372'], iou=0.7200, recall=0.9914, precision=0.7246, vol_sim=0.8445, mcc=0.8120, min_class_dice=0.8372, coverage=[88]/88 samples
[2026-06-19 06:17:52] INFO segtask_v1.trainer.trainer: Epoch 84/400 | LR=9.96e-04 | loss=0.0463 | val_dice=0.8372 | best=0.8533 (ep73) | 15:04:08 | L_res_0=0.0456 L_res_1=0.0484 L_res_2=0.0525
[2026-06-19 06:17:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 84): 19516.8 MiB
[2026-06-19 06:28:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2433, pooled_mean_dice=0.8461, per_class=['0.8461'], iou=0.7333, recall=0.9918, precision=0.7377, vol_sim=0.8531, mcc=0.8195, min_class_dice=0.8461, coverage=[88]/88 samples
[2026-06-19 06:28:36] INFO segtask_v1.trainer.trainer: Epoch 85/400 | LR=9.96e-04 | loss=0.0435 | val_dice=0.8461 | best=0.8533 (ep73) | 15:14:52 | L_res_0=0.0424 L_res_1=0.0454 L_res_2=0.0493
[2026-06-19 06:28:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 85): 19517.3 MiB
[2026-06-19 06:39:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2445, pooled_mean_dice=0.8478, per_class=['0.8478'], iou=0.7358, recall=0.9930, precision=0.7397, vol_sim=0.8538, mcc=0.8224, min_class_dice=0.8478, coverage=[88]/88 samples
[2026-06-19 06:39:22] INFO segtask_v1.trainer.trainer: Epoch 86/400 | LR=9.96e-04 | loss=0.0436 | val_dice=0.8478 | best=0.8533 (ep73) | 15:25:39 | L_res_0=0.0407 L_res_1=0.0459 L_res_2=0.0502
[2026-06-19 06:39:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 86): 19515.7 MiB
[2026-06-19 06:50:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2490, pooled_mean_dice=0.8438, per_class=['0.8438'], iou=0.7298, recall=0.9883, precision=0.7361, vol_sim=0.8537, mcc=0.8191, min_class_dice=0.8438, coverage=[88]/88 samples
[2026-06-19 06:50:07] INFO segtask_v1.trainer.trainer: Epoch 87/400 | LR=9.96e-04 | loss=0.0432 | val_dice=0.8438 | best=0.8533 (ep73) | 15:36:23 | L_res_0=0.0413 L_res_1=0.0454 L_res_2=0.0492
[2026-06-19 06:50:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 87): 19516.7 MiB
[2026-06-19 07:00:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2549, pooled_mean_dice=0.8431, per_class=['0.8431'], iou=0.7288, recall=0.9916, precision=0.7334, vol_sim=0.8503, mcc=0.8191, min_class_dice=0.8431, coverage=[88]/88 samples
[2026-06-19 07:00:51] INFO segtask_v1.trainer.trainer: Epoch 88/400 | LR=9.95e-04 | loss=0.0419 | val_dice=0.8431 | best=0.8533 (ep73) | 15:47:08 | L_res_0=0.0393 L_res_1=0.0442 L_res_2=0.0479
[2026-06-19 07:00:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 88): 19515.6 MiB
[2026-06-19 07:11:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2713, pooled_mean_dice=0.8363, per_class=['0.8363'], iou=0.7187, recall=0.9875, precision=0.7253, vol_sim=0.8469, mcc=0.8136, min_class_dice=0.8363, coverage=[88]/88 samples
[2026-06-19 07:11:36] INFO segtask_v1.trainer.trainer: Epoch 89/400 | LR=9.95e-04 | loss=0.0431 | val_dice=0.8363 | best=0.8533 (ep73) | 15:57:53 | L_res_0=0.0424 L_res_1=0.0448 L_res_2=0.0480
[2026-06-19 07:11:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 89): 19516.4 MiB
[2026-06-19 07:22:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2499, pooled_mean_dice=0.8408, per_class=['0.8408'], iou=0.7254, recall=0.9875, precision=0.7321, vol_sim=0.8515, mcc=0.8164, min_class_dice=0.8408, coverage=[88]/88 samples
[2026-06-19 07:22:20] INFO segtask_v1.trainer.trainer: Epoch 90/400 | LR=9.95e-04 | loss=0.0521 | val_dice=0.8408 | best=0.8533 (ep73) | 16:08:36 | L_res_0=0.0532 L_res_1=0.0546 L_res_2=0.0591
[2026-06-19 07:22:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 90): 19517.1 MiB
[2026-06-19 07:33:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2378, pooled_mean_dice=0.8542, per_class=['0.8542'], iou=0.7455, recall=0.9895, precision=0.7514, vol_sim=0.8632, mcc=0.8299, min_class_dice=0.8542, coverage=[88]/88 samples
[2026-06-19 07:33:09] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 07:33:09] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8542 at epoch 91
[2026-06-19 07:33:09] INFO segtask_v1.trainer.trainer: Epoch 91/400 | LR=9.95e-04 | loss=0.0442 | val_dice=0.8542 | best=0.8542 (ep91) | 16:19:26 | L_res_0=0.0424 L_res_1=0.0466 L_res_2=0.0509
[2026-06-19 07:33:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 91): 19516.5 MiB
[2026-06-19 07:43:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2376, pooled_mean_dice=0.8509, per_class=['0.8509'], iou=0.7405, recall=0.9906, precision=0.7457, vol_sim=0.8590, mcc=0.8265, min_class_dice=0.8509, coverage=[88]/88 samples
[2026-06-19 07:43:51] INFO segtask_v1.trainer.trainer: Epoch 92/400 | LR=9.95e-04 | loss=0.0462 | val_dice=0.8509 | best=0.8542 (ep91) | 16:30:07 | L_res_0=0.0450 L_res_1=0.0481 L_res_2=0.0522
[2026-06-19 07:43:51] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 92): 19515.9 MiB
[2026-06-19 07:54:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2478, pooled_mean_dice=0.8523, per_class=['0.8523'], iou=0.7427, recall=0.9899, precision=0.7483, vol_sim=0.8610, mcc=0.8300, min_class_dice=0.8523, coverage=[88]/88 samples
[2026-06-19 07:54:35] INFO segtask_v1.trainer.trainer: Epoch 93/400 | LR=9.95e-04 | loss=0.0483 | val_dice=0.8523 | best=0.8542 (ep91) | 16:40:52 | L_res_0=0.0478 L_res_1=0.0507 L_res_2=0.0547
[2026-06-19 07:54:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 93): 19517.0 MiB
[2026-06-19 08:05:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2332, pooled_mean_dice=0.8570, per_class=['0.8570'], iou=0.7498, recall=0.9889, precision=0.7561, vol_sim=0.8666, mcc=0.8335, min_class_dice=0.8570, coverage=[88]/88 samples
[2026-06-19 08:05:25] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 08:05:25] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8570 at epoch 94
[2026-06-19 08:05:25] INFO segtask_v1.trainer.trainer: Epoch 94/400 | LR=9.95e-04 | loss=0.0441 | val_dice=0.8570 | best=0.8570 (ep94) | 16:51:41 | L_res_0=0.0432 L_res_1=0.0465 L_res_2=0.0498
[2026-06-19 08:05:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 94): 19517.7 MiB
[2026-06-19 08:16:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2348, pooled_mean_dice=0.8586, per_class=['0.8586'], iou=0.7523, recall=0.9911, precision=0.7574, vol_sim=0.8664, mcc=0.8344, min_class_dice=0.8586, coverage=[88]/88 samples
[2026-06-19 08:16:14] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 08:16:14] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8586 at epoch 95
[2026-06-19 08:16:14] INFO segtask_v1.trainer.trainer: Epoch 95/400 | LR=9.94e-04 | loss=0.0479 | val_dice=0.8586 | best=0.8586 (ep95) | 17:02:31 | L_res_0=0.0460 L_res_1=0.0509 L_res_2=0.0542
[2026-06-19 08:16:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 95): 19516.2 MiB
[2026-06-19 08:26:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2577, pooled_mean_dice=0.8425, per_class=['0.8425'], iou=0.7279, recall=0.9898, precision=0.7334, vol_sim=0.8512, mcc=0.8198, min_class_dice=0.8425, coverage=[88]/88 samples
[2026-06-19 08:26:58] INFO segtask_v1.trainer.trainer: Epoch 96/400 | LR=9.94e-04 | loss=0.0544 | val_dice=0.8425 | best=0.8586 (ep95) | 17:13:14 | L_res_0=0.0575 L_res_1=0.0562 L_res_2=0.0593
[2026-06-19 08:26:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 96): 19516.2 MiB
[2026-06-19 08:37:39] INFO segtask_v1.trainer.validation:   Val: loss=0.2358, pooled_mean_dice=0.8548, per_class=['0.8548'], iou=0.7464, recall=0.9919, precision=0.7509, vol_sim=0.8617, mcc=0.8305, min_class_dice=0.8548, coverage=[88]/88 samples
[2026-06-19 08:37:39] INFO segtask_v1.trainer.trainer: Epoch 97/400 | LR=9.94e-04 | loss=0.0533 | val_dice=0.8548 | best=0.8586 (ep95) | 17:23:55 | L_res_0=0.0538 L_res_1=0.0558 L_res_2=0.0619
[2026-06-19 08:37:39] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 97): 19515.2 MiB
[2026-06-19 08:48:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2329, pooled_mean_dice=0.8530, per_class=['0.8530'], iou=0.7437, recall=0.9895, precision=0.7496, vol_sim=0.8621, mcc=0.8313, min_class_dice=0.8530, coverage=[88]/88 samples
[2026-06-19 08:48:23] INFO segtask_v1.trainer.trainer: Epoch 98/400 | LR=9.94e-04 | loss=0.0463 | val_dice=0.8530 | best=0.8586 (ep95) | 17:34:40 | L_res_0=0.0461 L_res_1=0.0489 L_res_2=0.0526
[2026-06-19 08:48:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 98): 19516.5 MiB
[2026-06-19 08:59:07] INFO segtask_v1.trainer.validation:   Val: loss=0.2228, pooled_mean_dice=0.8622, per_class=['0.8622'], iou=0.7577, recall=0.9883, precision=0.7645, vol_sim=0.8723, mcc=0.8378, min_class_dice=0.8622, coverage=[88]/88 samples
[2026-06-19 08:59:14] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 08:59:14] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8622 at epoch 99
[2026-06-19 08:59:14] INFO segtask_v1.trainer.trainer: Epoch 99/400 | LR=9.94e-04 | loss=0.0438 | val_dice=0.8622 | best=0.8622 (ep99) | 17:45:30 | L_res_0=0.0426 L_res_1=0.0461 L_res_2=0.0498
[2026-06-19 08:59:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 99): 19516.4 MiB
[2026-06-19 09:09:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2215, pooled_mean_dice=0.8627, per_class=['0.8627'], iou=0.7585, recall=0.9876, precision=0.7658, vol_sim=0.8735, mcc=0.8392, min_class_dice=0.8627, coverage=[88]/88 samples
[2026-06-19 09:10:04] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 09:10:04] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8627 at epoch 100
[2026-06-19 09:10:04] INFO segtask_v1.trainer.trainer: Epoch 100/400 | LR=9.94e-04 | loss=0.0427 | val_dice=0.8627 | best=0.8627 (ep100) | 17:56:20 | L_res_0=0.0411 L_res_1=0.0450 L_res_2=0.0483
[2026-06-19 09:10:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 100): 19516.9 MiB
[2026-06-19 09:20:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2232, pooled_mean_dice=0.8649, per_class=['0.8649'], iou=0.7620, recall=0.9908, precision=0.7674, vol_sim=0.8730, mcc=0.8409, min_class_dice=0.8649, coverage=[88]/88 samples
[2026-06-19 09:20:54] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 09:20:54] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8649 at epoch 101
[2026-06-19 09:20:54] INFO segtask_v1.trainer.trainer: Epoch 101/400 | LR=9.94e-04 | loss=0.0418 | val_dice=0.8649 | best=0.8649 (ep101) | 18:07:11 | L_res_0=0.0402 L_res_1=0.0436 L_res_2=0.0473
[2026-06-19 09:20:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 101): 19515.8 MiB
[2026-06-19 09:31:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2443, pooled_mean_dice=0.8521, per_class=['0.8521'], iou=0.7423, recall=0.9882, precision=0.7490, vol_sim=0.8623, mcc=0.8279, min_class_dice=0.8521, coverage=[88]/88 samples
[2026-06-19 09:31:37] INFO segtask_v1.trainer.trainer: Epoch 102/400 | LR=9.93e-04 | loss=0.0601 | val_dice=0.8521 | best=0.8649 (ep101) | 18:17:53 | L_res_0=0.0626 L_res_1=0.0634 L_res_2=0.0651
[2026-06-19 09:31:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 102): 19516.2 MiB
[2026-06-19 09:42:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2486, pooled_mean_dice=0.8484, per_class=['0.8484'], iou=0.7368, recall=0.9904, precision=0.7421, vol_sim=0.8567, mcc=0.8258, min_class_dice=0.8484, coverage=[88]/88 samples
[2026-06-19 09:42:20] INFO segtask_v1.trainer.trainer: Epoch 103/400 | LR=9.93e-04 | loss=0.0489 | val_dice=0.8484 | best=0.8649 (ep101) | 18:28:37 | L_res_0=0.0481 L_res_1=0.0515 L_res_2=0.0561
[2026-06-19 09:42:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 103): 19517.2 MiB
[2026-06-19 09:53:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2250, pooled_mean_dice=0.8563, per_class=['0.8563'], iou=0.7487, recall=0.9881, precision=0.7555, vol_sim=0.8666, mcc=0.8324, min_class_dice=0.8563, coverage=[88]/88 samples
[2026-06-19 09:53:06] INFO segtask_v1.trainer.trainer: Epoch 104/400 | LR=9.93e-04 | loss=0.0457 | val_dice=0.8563 | best=0.8649 (ep101) | 18:39:23 | L_res_0=0.0456 L_res_1=0.0474 L_res_2=0.0515
[2026-06-19 09:53:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 104): 19516.5 MiB
[2026-06-19 10:03:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2267, pooled_mean_dice=0.8578, per_class=['0.8578'], iou=0.7510, recall=0.9894, precision=0.7571, vol_sim=0.8670, mcc=0.8351, min_class_dice=0.8578, coverage=[88]/88 samples
[2026-06-19 10:03:50] INFO segtask_v1.trainer.trainer: Epoch 105/400 | LR=9.93e-04 | loss=0.0423 | val_dice=0.8578 | best=0.8649 (ep101) | 18:50:07 | L_res_0=0.0412 L_res_1=0.0444 L_res_2=0.0481
[2026-06-19 10:03:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 105): 19515.6 MiB
[2026-06-19 10:14:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2263, pooled_mean_dice=0.8595, per_class=['0.8595'], iou=0.7536, recall=0.9878, precision=0.7607, vol_sim=0.8701, mcc=0.8340, min_class_dice=0.8595, coverage=[88]/88 samples
[2026-06-19 10:14:36] INFO segtask_v1.trainer.trainer: Epoch 106/400 | LR=9.93e-04 | loss=0.0414 | val_dice=0.8595 | best=0.8649 (ep101) | 19:00:52 | L_res_0=0.0388 L_res_1=0.0436 L_res_2=0.0476
[2026-06-19 10:14:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 106): 19516.0 MiB
[2026-06-19 10:25:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2158, pooled_mean_dice=0.8661, per_class=['0.8661'], iou=0.7639, recall=0.9890, precision=0.7704, vol_sim=0.8758, mcc=0.8436, min_class_dice=0.8661, coverage=[88]/88 samples
[2026-06-19 10:25:25] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 10:25:25] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8661 at epoch 107
[2026-06-19 10:25:25] INFO segtask_v1.trainer.trainer: Epoch 107/400 | LR=9.93e-04 | loss=0.0405 | val_dice=0.8661 | best=0.8661 (ep107) | 19:11:41 | L_res_0=0.0381 L_res_1=0.0427 L_res_2=0.0461
[2026-06-19 10:25:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 107): 19515.9 MiB
[2026-06-19 10:36:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2097, pooled_mean_dice=0.8680, per_class=['0.8680'], iou=0.7667, recall=0.9877, precision=0.7741, vol_sim=0.8787, mcc=0.8454, min_class_dice=0.8680, coverage=[88]/88 samples
[2026-06-19 10:36:17] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 10:36:17] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8680 at epoch 108
[2026-06-19 10:36:17] INFO segtask_v1.trainer.trainer: Epoch 108/400 | LR=9.92e-04 | loss=0.0402 | val_dice=0.8680 | best=0.8680 (ep108) | 19:22:33 | L_res_0=0.0375 L_res_1=0.0425 L_res_2=0.0459
[2026-06-19 10:36:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 108): 19517.1 MiB
[2026-06-19 10:46:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2471, pooled_mean_dice=0.8497, per_class=['0.8497'], iou=0.7387, recall=0.9811, precision=0.7494, vol_sim=0.8661, mcc=0.8284, min_class_dice=0.8497, coverage=[88]/88 samples
[2026-06-19 10:46:59] INFO segtask_v1.trainer.trainer: Epoch 109/400 | LR=9.92e-04 | loss=0.0409 | val_dice=0.8497 | best=0.8680 (ep108) | 19:33:15 | L_res_0=0.0400 L_res_1=0.0425 L_res_2=0.0459
[2026-06-19 10:46:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 109): 19516.2 MiB
[2026-06-19 10:57:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2273, pooled_mean_dice=0.8606, per_class=['0.8606'], iou=0.7553, recall=0.9836, precision=0.7650, vol_sim=0.8749, mcc=0.8371, min_class_dice=0.8606, coverage=[88]/88 samples
[2026-06-19 10:57:43] INFO segtask_v1.trainer.trainer: Epoch 110/400 | LR=9.92e-04 | loss=0.0422 | val_dice=0.8606 | best=0.8680 (ep108) | 19:43:59 | L_res_0=0.0407 L_res_1=0.0445 L_res_2=0.0479
[2026-06-19 10:57:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 110): 19516.8 MiB
[2026-06-19 11:08:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2036, pooled_mean_dice=0.8706, per_class=['0.8706'], iou=0.7709, recall=0.9892, precision=0.7774, vol_sim=0.8801, mcc=0.8481, min_class_dice=0.8706, coverage=[88]/88 samples
[2026-06-19 11:08:32] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 11:08:32] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8706 at epoch 111
[2026-06-19 11:08:32] INFO segtask_v1.trainer.trainer: Epoch 111/400 | LR=9.92e-04 | loss=0.0393 | val_dice=0.8706 | best=0.8706 (ep111) | 19:54:49 | L_res_0=0.0366 L_res_1=0.0414 L_res_2=0.0449
[2026-06-19 11:08:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 111): 19516.8 MiB
[2026-06-19 11:19:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2034, pooled_mean_dice=0.8719, per_class=['0.8719'], iou=0.7729, recall=0.9853, precision=0.7819, vol_sim=0.8849, mcc=0.8478, min_class_dice=0.8719, coverage=[88]/88 samples
[2026-06-19 11:19:24] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 11:19:24] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8719 at epoch 112
[2026-06-19 11:19:24] INFO segtask_v1.trainer.trainer: Epoch 112/400 | LR=9.92e-04 | loss=0.0395 | val_dice=0.8719 | best=0.8719 (ep112) | 20:05:40 | L_res_0=0.0370 L_res_1=0.0413 L_res_2=0.0449
[2026-06-19 11:19:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 112): 19516.4 MiB
[2026-06-19 11:30:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2127, pooled_mean_dice=0.8725, per_class=['0.8725'], iou=0.7739, recall=0.9885, precision=0.7809, vol_sim=0.8826, mcc=0.8516, min_class_dice=0.8725, coverage=[88]/88 samples
[2026-06-19 11:30:12] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 11:30:12] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8725 at epoch 113
[2026-06-19 11:30:12] INFO segtask_v1.trainer.trainer: Epoch 113/400 | LR=9.91e-04 | loss=0.0400 | val_dice=0.8725 | best=0.8725 (ep113) | 20:16:28 | L_res_0=0.0374 L_res_1=0.0420 L_res_2=0.0455
[2026-06-19 11:30:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 113): 19516.1 MiB
[2026-06-19 11:40:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2057, pooled_mean_dice=0.8721, per_class=['0.8721'], iou=0.7732, recall=0.9871, precision=0.7811, vol_sim=0.8835, mcc=0.8479, min_class_dice=0.8721, coverage=[88]/88 samples
[2026-06-19 11:40:55] INFO segtask_v1.trainer.trainer: Epoch 114/400 | LR=9.91e-04 | loss=0.0393 | val_dice=0.8721 | best=0.8725 (ep113) | 20:27:12 | L_res_0=0.0369 L_res_1=0.0411 L_res_2=0.0445
[2026-06-19 11:40:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 114): 19517.5 MiB
[2026-06-19 11:51:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2239, pooled_mean_dice=0.8588, per_class=['0.8588'], iou=0.7526, recall=0.9861, precision=0.7606, vol_sim=0.8709, mcc=0.8369, min_class_dice=0.8588, coverage=[88]/88 samples
[2026-06-19 11:51:41] INFO segtask_v1.trainer.trainer: Epoch 115/400 | LR=9.91e-04 | loss=0.0400 | val_dice=0.8588 | best=0.8725 (ep113) | 20:37:57 | L_res_0=0.0374 L_res_1=0.0423 L_res_2=0.0456
[2026-06-19 11:51:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 115): 19516.4 MiB
[2026-06-19 12:02:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2107, pooled_mean_dice=0.8719, per_class=['0.8719'], iou=0.7728, recall=0.9885, precision=0.7798, vol_sim=0.8820, mcc=0.8512, min_class_dice=0.8719, coverage=[88]/88 samples
[2026-06-19 12:02:22] INFO segtask_v1.trainer.trainer: Epoch 116/400 | LR=9.91e-04 | loss=0.0401 | val_dice=0.8719 | best=0.8725 (ep113) | 20:48:38 | L_res_0=0.0386 L_res_1=0.0416 L_res_2=0.0448
[2026-06-19 12:02:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 116): 19516.2 MiB
[2026-06-19 12:13:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2083, pooled_mean_dice=0.8648, per_class=['0.8648'], iou=0.7618, recall=0.9874, precision=0.7692, vol_sim=0.8758, mcc=0.8438, min_class_dice=0.8648, coverage=[88]/88 samples
[2026-06-19 12:13:06] INFO segtask_v1.trainer.trainer: Epoch 117/400 | LR=9.91e-04 | loss=0.0405 | val_dice=0.8648 | best=0.8725 (ep113) | 20:59:22 | L_res_0=0.0377 L_res_1=0.0425 L_res_2=0.0460
[2026-06-19 12:13:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 117): 19516.0 MiB
[2026-06-19 12:23:51] INFO segtask_v1.trainer.validation:   Val: loss=0.2044, pooled_mean_dice=0.8737, per_class=['0.8737'], iou=0.7757, recall=0.9876, precision=0.7833, vol_sim=0.8846, mcc=0.8525, min_class_dice=0.8737, coverage=[88]/88 samples
[2026-06-19 12:23:57] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 12:23:57] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8737 at epoch 118
[2026-06-19 12:23:57] INFO segtask_v1.trainer.trainer: Epoch 118/400 | LR=9.91e-04 | loss=0.0386 | val_dice=0.8737 | best=0.8737 (ep118) | 21:10:13 | L_res_0=0.0363 L_res_1=0.0404 L_res_2=0.0441
[2026-06-19 12:23:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 118): 19515.6 MiB
[2026-06-19 12:34:39] INFO segtask_v1.trainer.validation:   Val: loss=0.1949, pooled_mean_dice=0.8763, per_class=['0.8763'], iou=0.7798, recall=0.9885, precision=0.7870, vol_sim=0.8865, mcc=0.8558, min_class_dice=0.8763, coverage=[88]/88 samples
[2026-06-19 12:34:45] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 12:34:45] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8763 at epoch 119
[2026-06-19 12:34:45] INFO segtask_v1.trainer.trainer: Epoch 119/400 | LR=9.90e-04 | loss=0.0373 | val_dice=0.8763 | best=0.8763 (ep119) | 21:21:01 | L_res_0=0.0342 L_res_1=0.0391 L_res_2=0.0426
[2026-06-19 12:34:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 119): 19516.7 MiB
[2026-06-19 12:45:28] INFO segtask_v1.trainer.validation:   Val: loss=0.1962, pooled_mean_dice=0.8742, per_class=['0.8742'], iou=0.7765, recall=0.9844, precision=0.7862, vol_sim=0.8881, mcc=0.8524, min_class_dice=0.8742, coverage=[88]/88 samples
[2026-06-19 12:45:28] INFO segtask_v1.trainer.trainer: Epoch 120/400 | LR=9.90e-04 | loss=0.0394 | val_dice=0.8742 | best=0.8763 (ep119) | 21:31:44 | L_res_0=0.0358 L_res_1=0.0411 L_res_2=0.0456
[2026-06-19 12:45:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 120): 19516.5 MiB
[2026-06-19 12:56:14] INFO segtask_v1.trainer.validation:   Val: loss=0.1901, pooled_mean_dice=0.8804, per_class=['0.8804'], iou=0.7864, recall=0.9899, precision=0.7928, vol_sim=0.8894, mcc=0.8590, min_class_dice=0.8804, coverage=[88]/88 samples
[2026-06-19 12:56:20] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 12:56:20] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8804 at epoch 121
[2026-06-19 12:56:20] INFO segtask_v1.trainer.trainer: Epoch 121/400 | LR=9.90e-04 | loss=0.0398 | val_dice=0.8804 | best=0.8804 (ep121) | 21:42:36 | L_res_0=0.0382 L_res_1=0.0412 L_res_2=0.0447
[2026-06-19 12:56:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 121): 19516.7 MiB
[2026-06-19 13:07:05] INFO segtask_v1.trainer.validation:   Val: loss=0.1915, pooled_mean_dice=0.8795, per_class=['0.8795'], iou=0.7848, recall=0.9879, precision=0.7925, vol_sim=0.8902, mcc=0.8585, min_class_dice=0.8795, coverage=[88]/88 samples
[2026-06-19 13:07:05] INFO segtask_v1.trainer.trainer: Epoch 122/400 | LR=9.90e-04 | loss=0.0385 | val_dice=0.8795 | best=0.8804 (ep121) | 21:53:22 | L_res_0=0.0365 L_res_1=0.0401 L_res_2=0.0436
[2026-06-19 13:07:05] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 122): 19517.0 MiB
[2026-06-19 13:17:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2024, pooled_mean_dice=0.8789, per_class=['0.8789'], iou=0.7839, recall=0.9873, precision=0.7919, vol_sim=0.8902, mcc=0.8569, min_class_dice=0.8789, coverage=[88]/88 samples
[2026-06-19 13:17:48] INFO segtask_v1.trainer.trainer: Epoch 123/400 | LR=9.90e-04 | loss=0.0380 | val_dice=0.8789 | best=0.8804 (ep121) | 22:04:04 | L_res_0=0.0352 L_res_1=0.0401 L_res_2=0.0434
[2026-06-19 13:17:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 123): 19516.0 MiB
[2026-06-19 13:28:30] INFO segtask_v1.trainer.validation:   Val: loss=0.1905, pooled_mean_dice=0.8817, per_class=['0.8817'], iou=0.7885, recall=0.9889, precision=0.7955, vol_sim=0.8916, mcc=0.8597, min_class_dice=0.8817, coverage=[88]/88 samples
[2026-06-19 13:28:36] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 13:28:36] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8817 at epoch 124
[2026-06-19 13:28:36] INFO segtask_v1.trainer.trainer: Epoch 124/400 | LR=9.89e-04 | loss=0.0378 | val_dice=0.8817 | best=0.8817 (ep124) | 22:14:52 | L_res_0=0.0345 L_res_1=0.0398 L_res_2=0.0432
[2026-06-19 13:28:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 124): 19516.5 MiB
[2026-06-19 13:39:24] INFO segtask_v1.trainer.validation:   Val: loss=0.1970, pooled_mean_dice=0.8798, per_class=['0.8798'], iou=0.7854, recall=0.9885, precision=0.7926, vol_sim=0.8900, mcc=0.8579, min_class_dice=0.8798, coverage=[88]/88 samples
[2026-06-19 13:39:24] INFO segtask_v1.trainer.trainer: Epoch 125/400 | LR=9.89e-04 | loss=0.0374 | val_dice=0.8798 | best=0.8817 (ep124) | 22:25:41 | L_res_0=0.0342 L_res_1=0.0394 L_res_2=0.0428
[2026-06-19 13:39:24] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 125): 19515.8 MiB
[2026-06-19 13:50:07] INFO segtask_v1.trainer.validation:   Val: loss=0.1986, pooled_mean_dice=0.8763, per_class=['0.8763'], iou=0.7799, recall=0.9884, precision=0.7871, vol_sim=0.8866, mcc=0.8559, min_class_dice=0.8763, coverage=[88]/88 samples
[2026-06-19 13:50:07] INFO segtask_v1.trainer.trainer: Epoch 126/400 | LR=9.89e-04 | loss=0.0370 | val_dice=0.8763 | best=0.8817 (ep124) | 22:36:23 | L_res_0=0.0343 L_res_1=0.0387 L_res_2=0.0424
[2026-06-19 13:50:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 126): 19516.2 MiB
[2026-06-19 14:00:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2009, pooled_mean_dice=0.8772, per_class=['0.8772'], iou=0.7812, recall=0.9891, precision=0.7880, vol_sim=0.8868, mcc=0.8568, min_class_dice=0.8772, coverage=[88]/88 samples
[2026-06-19 14:00:49] INFO segtask_v1.trainer.trainer: Epoch 127/400 | LR=9.89e-04 | loss=0.0370 | val_dice=0.8772 | best=0.8817 (ep124) | 22:47:05 | L_res_0=0.0338 L_res_1=0.0388 L_res_2=0.0424
[2026-06-19 14:00:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 127): 19516.0 MiB
[2026-06-19 14:11:35] INFO segtask_v1.trainer.validation:   Val: loss=0.1895, pooled_mean_dice=0.8846, per_class=['0.8846'], iou=0.7930, recall=0.9882, precision=0.8006, vol_sim=0.8951, mcc=0.8642, min_class_dice=0.8846, coverage=[88]/88 samples
[2026-06-19 14:11:41] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 14:11:41] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8846 at epoch 128
[2026-06-19 14:11:41] INFO segtask_v1.trainer.trainer: Epoch 128/400 | LR=9.89e-04 | loss=0.0384 | val_dice=0.8846 | best=0.8846 (ep128) | 22:57:58 | L_res_0=0.0360 L_res_1=0.0401 L_res_2=0.0435
[2026-06-19 14:11:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 128): 19515.7 MiB
[2026-06-19 14:22:26] INFO segtask_v1.trainer.validation:   Val: loss=0.1980, pooled_mean_dice=0.8789, per_class=['0.8789'], iou=0.7839, recall=0.9905, precision=0.7898, vol_sim=0.8873, mcc=0.8566, min_class_dice=0.8789, coverage=[88]/88 samples
[2026-06-19 14:22:26] INFO segtask_v1.trainer.trainer: Epoch 129/400 | LR=9.88e-04 | loss=0.0683 | val_dice=0.8789 | best=0.8846 (ep128) | 23:08:42 | L_res_0=0.0721 L_res_1=0.0693 L_res_2=0.0778
[2026-06-19 14:22:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 129): 19517.1 MiB
[2026-06-19 14:33:13] INFO segtask_v1.trainer.validation:   Val: loss=0.1937, pooled_mean_dice=0.8817, per_class=['0.8817'], iou=0.7885, recall=0.9910, precision=0.7942, vol_sim=0.8897, mcc=0.8589, min_class_dice=0.8817, coverage=[88]/88 samples
[2026-06-19 14:33:13] INFO segtask_v1.trainer.trainer: Epoch 130/400 | LR=9.88e-04 | loss=0.0520 | val_dice=0.8817 | best=0.8846 (ep128) | 23:19:29 | L_res_0=0.0516 L_res_1=0.0547 L_res_2=0.0588
[2026-06-19 14:33:13] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 130): 19516.5 MiB
[2026-06-19 14:43:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2043, pooled_mean_dice=0.8762, per_class=['0.8762'], iou=0.7797, recall=0.9914, precision=0.7850, vol_sim=0.8838, mcc=0.8524, min_class_dice=0.8762, coverage=[88]/88 samples
[2026-06-19 14:43:59] INFO segtask_v1.trainer.trainer: Epoch 131/400 | LR=9.88e-04 | loss=0.0525 | val_dice=0.8762 | best=0.8846 (ep128) | 23:30:15 | L_res_0=0.0529 L_res_1=0.0556 L_res_2=0.0604
[2026-06-19 14:43:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 131): 19516.0 MiB
[2026-06-19 14:54:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2114, pooled_mean_dice=0.8672, per_class=['0.8672'], iou=0.7655, recall=0.9924, precision=0.7701, vol_sim=0.8739, mcc=0.8437, min_class_dice=0.8672, coverage=[88]/88 samples
[2026-06-19 14:54:44] INFO segtask_v1.trainer.trainer: Epoch 132/400 | LR=9.88e-04 | loss=0.0531 | val_dice=0.8672 | best=0.8846 (ep128) | 23:41:00 | L_res_0=0.0562 L_res_1=0.0564 L_res_2=0.0595
[2026-06-19 14:54:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 132): 19516.3 MiB
[2026-06-19 15:05:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2176, pooled_mean_dice=0.8643, per_class=['0.8643'], iou=0.7610, recall=0.9912, precision=0.7662, vol_sim=0.8720, mcc=0.8430, min_class_dice=0.8643, coverage=[88]/88 samples
[2026-06-19 15:05:27] INFO segtask_v1.trainer.trainer: Epoch 133/400 | LR=9.87e-04 | loss=0.0427 | val_dice=0.8643 | best=0.8846 (ep128) | 23:51:43 | L_res_0=0.0415 L_res_1=0.0454 L_res_2=0.0491
[2026-06-19 15:05:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 133): 19517.1 MiB
[2026-06-19 15:16:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2074, pooled_mean_dice=0.8744, per_class=['0.8744'], iou=0.7768, recall=0.9924, precision=0.7814, vol_sim=0.8811, mcc=0.8506, min_class_dice=0.8744, coverage=[88]/88 samples
[2026-06-19 15:16:14] INFO segtask_v1.trainer.trainer: Epoch 134/400 | LR=9.87e-04 | loss=0.0430 | val_dice=0.8744 | best=0.8846 (ep128) | 24:02:30 | L_res_0=0.0418 L_res_1=0.0454 L_res_2=0.0490
[2026-06-19 15:16:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 134): 19516.1 MiB
[2026-06-19 15:26:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2224, pooled_mean_dice=0.8602, per_class=['0.8602'], iou=0.7547, recall=0.9862, precision=0.7628, vol_sim=0.8722, mcc=0.8365, min_class_dice=0.8602, coverage=[88]/88 samples
[2026-06-19 15:26:59] INFO segtask_v1.trainer.trainer: Epoch 135/400 | LR=9.87e-04 | loss=0.0415 | val_dice=0.8602 | best=0.8846 (ep128) | 24:13:15 | L_res_0=0.0390 L_res_1=0.0438 L_res_2=0.0477
[2026-06-19 15:26:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 135): 19516.1 MiB
[2026-06-19 15:37:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2187, pooled_mean_dice=0.8621, per_class=['0.8621'], iou=0.7576, recall=0.9895, precision=0.7638, vol_sim=0.8713, mcc=0.8394, min_class_dice=0.8621, coverage=[88]/88 samples
[2026-06-19 15:37:45] INFO segtask_v1.trainer.trainer: Epoch 136/400 | LR=9.87e-04 | loss=0.0389 | val_dice=0.8621 | best=0.8846 (ep128) | 24:24:01 | L_res_0=0.0360 L_res_1=0.0412 L_res_2=0.0449
[2026-06-19 15:37:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 136): 19516.6 MiB
[2026-06-19 15:48:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2124, pooled_mean_dice=0.8685, per_class=['0.8685'], iou=0.7676, recall=0.9893, precision=0.7740, vol_sim=0.8779, mcc=0.8455, min_class_dice=0.8685, coverage=[88]/88 samples
[2026-06-19 15:48:28] INFO segtask_v1.trainer.trainer: Epoch 137/400 | LR=9.87e-04 | loss=0.0382 | val_dice=0.8685 | best=0.8846 (ep128) | 24:34:44 | L_res_0=0.0353 L_res_1=0.0403 L_res_2=0.0439
[2026-06-19 15:48:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 137): 19515.9 MiB
[2026-06-19 15:59:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2259, pooled_mean_dice=0.8625, per_class=['0.8625'], iou=0.7583, recall=0.9888, precision=0.7649, vol_sim=0.8723, mcc=0.8392, min_class_dice=0.8625, coverage=[88]/88 samples
[2026-06-19 15:59:15] INFO segtask_v1.trainer.trainer: Epoch 138/400 | LR=9.86e-04 | loss=0.0377 | val_dice=0.8625 | best=0.8846 (ep128) | 24:45:31 | L_res_0=0.0346 L_res_1=0.0398 L_res_2=0.0437
[2026-06-19 15:59:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 138): 19517.1 MiB
[2026-06-19 16:10:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2183, pooled_mean_dice=0.8708, per_class=['0.8708'], iou=0.7712, recall=0.9903, precision=0.7771, vol_sim=0.8794, mcc=0.8490, min_class_dice=0.8708, coverage=[88]/88 samples
[2026-06-19 16:10:00] INFO segtask_v1.trainer.trainer: Epoch 139/400 | LR=9.86e-04 | loss=0.0380 | val_dice=0.8708 | best=0.8846 (ep128) | 24:56:16 | L_res_0=0.0350 L_res_1=0.0400 L_res_2=0.0437
[2026-06-19 16:10:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 139): 19515.8 MiB
[2026-06-19 16:20:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2028, pooled_mean_dice=0.8764, per_class=['0.8764'], iou=0.7800, recall=0.9891, precision=0.7868, vol_sim=0.8861, mcc=0.8540, min_class_dice=0.8764, coverage=[88]/88 samples
[2026-06-19 16:20:45] INFO segtask_v1.trainer.trainer: Epoch 140/400 | LR=9.86e-04 | loss=0.0373 | val_dice=0.8764 | best=0.8846 (ep128) | 25:07:01 | L_res_0=0.0347 L_res_1=0.0393 L_res_2=0.0427
[2026-06-19 16:20:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 140): 19516.5 MiB
[2026-06-19 16:31:29] INFO segtask_v1.trainer.validation:   Val: loss=0.2196, pooled_mean_dice=0.8666, per_class=['0.8666'], iou=0.7646, recall=0.9851, precision=0.7736, vol_sim=0.8797, mcc=0.8451, min_class_dice=0.8666, coverage=[88]/88 samples
[2026-06-19 16:31:29] INFO segtask_v1.trainer.trainer: Epoch 141/400 | LR=9.86e-04 | loss=0.0371 | val_dice=0.8666 | best=0.8846 (ep128) | 25:17:45 | L_res_0=0.0341 L_res_1=0.0389 L_res_2=0.0425
[2026-06-19 16:31:29] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 141): 19515.5 MiB
[2026-06-19 16:42:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2075, pooled_mean_dice=0.8732, per_class=['0.8732'], iou=0.7749, recall=0.9888, precision=0.7818, vol_sim=0.8831, mcc=0.8508, min_class_dice=0.8732, coverage=[88]/88 samples
[2026-06-19 16:42:12] INFO segtask_v1.trainer.trainer: Epoch 142/400 | LR=9.85e-04 | loss=0.0363 | val_dice=0.8732 | best=0.8846 (ep128) | 25:28:28 | L_res_0=0.0332 L_res_1=0.0380 L_res_2=0.0416
[2026-06-19 16:42:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 142): 19517.1 MiB
[2026-06-19 16:52:54] INFO segtask_v1.trainer.validation:   Val: loss=0.2091, pooled_mean_dice=0.8692, per_class=['0.8692'], iou=0.7687, recall=0.9887, precision=0.7755, vol_sim=0.8792, mcc=0.8475, min_class_dice=0.8692, coverage=[88]/88 samples
[2026-06-19 16:52:54] INFO segtask_v1.trainer.trainer: Epoch 143/400 | LR=9.85e-04 | loss=0.0369 | val_dice=0.8692 | best=0.8846 (ep128) | 25:39:10 | L_res_0=0.0339 L_res_1=0.0387 L_res_2=0.0425
[2026-06-19 16:52:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 143): 19517.3 MiB
[2026-06-19 17:03:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2066, pooled_mean_dice=0.8767, per_class=['0.8767'], iou=0.7805, recall=0.9891, precision=0.7873, vol_sim=0.8864, mcc=0.8552, min_class_dice=0.8767, coverage=[88]/88 samples
[2026-06-19 17:03:37] INFO segtask_v1.trainer.trainer: Epoch 144/400 | LR=9.85e-04 | loss=0.0363 | val_dice=0.8767 | best=0.8846 (ep128) | 25:49:53 | L_res_0=0.0332 L_res_1=0.0382 L_res_2=0.0416
[2026-06-19 17:03:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 144): 19516.1 MiB
[2026-06-19 17:14:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2129, pooled_mean_dice=0.8692, per_class=['0.8692'], iou=0.7686, recall=0.9876, precision=0.7761, vol_sim=0.8801, mcc=0.8482, min_class_dice=0.8692, coverage=[88]/88 samples
[2026-06-19 17:14:23] INFO segtask_v1.trainer.trainer: Epoch 145/400 | LR=9.85e-04 | loss=0.0370 | val_dice=0.8692 | best=0.8846 (ep128) | 26:00:39 | L_res_0=0.0344 L_res_1=0.0390 L_res_2=0.0423
[2026-06-19 17:14:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 145): 19518.4 MiB
[2026-06-19 17:25:06] INFO segtask_v1.trainer.validation:   Val: loss=0.2104, pooled_mean_dice=0.8719, per_class=['0.8719'], iou=0.7728, recall=0.9891, precision=0.7794, vol_sim=0.8814, mcc=0.8509, min_class_dice=0.8719, coverage=[88]/88 samples
[2026-06-19 17:25:06] INFO segtask_v1.trainer.trainer: Epoch 146/400 | LR=9.84e-04 | loss=0.0363 | val_dice=0.8719 | best=0.8846 (ep128) | 26:11:22 | L_res_0=0.0332 L_res_1=0.0381 L_res_2=0.0416
[2026-06-19 17:25:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 146): 19516.6 MiB
[2026-06-19 17:35:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2013, pooled_mean_dice=0.8808, per_class=['0.8808'], iou=0.7870, recall=0.9884, precision=0.7944, vol_sim=0.8912, mcc=0.8598, min_class_dice=0.8808, coverage=[88]/88 samples
[2026-06-19 17:35:50] INFO segtask_v1.trainer.trainer: Epoch 147/400 | LR=9.84e-04 | loss=0.0364 | val_dice=0.8808 | best=0.8846 (ep128) | 26:22:06 | L_res_0=0.0330 L_res_1=0.0383 L_res_2=0.0417
[2026-06-19 17:35:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 147): 19516.4 MiB
[2026-06-19 17:46:34] INFO segtask_v1.trainer.validation:   Val: loss=0.1991, pooled_mean_dice=0.8772, per_class=['0.8772'], iou=0.7812, recall=0.9880, precision=0.7887, vol_sim=0.8878, mcc=0.8550, min_class_dice=0.8772, coverage=[88]/88 samples
[2026-06-19 17:46:34] INFO segtask_v1.trainer.trainer: Epoch 148/400 | LR=9.84e-04 | loss=0.0360 | val_dice=0.8772 | best=0.8846 (ep128) | 26:32:50 | L_res_0=0.0327 L_res_1=0.0377 L_res_2=0.0413
[2026-06-19 17:46:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 148): 19517.2 MiB
[2026-06-19 17:57:19] INFO segtask_v1.trainer.validation:   Val: loss=0.1987, pooled_mean_dice=0.8742, per_class=['0.8742'], iou=0.7765, recall=0.9905, precision=0.7823, vol_sim=0.8826, mcc=0.8540, min_class_dice=0.8742, coverage=[88]/88 samples
[2026-06-19 17:57:19] INFO segtask_v1.trainer.trainer: Epoch 149/400 | LR=9.84e-04 | loss=0.0366 | val_dice=0.8742 | best=0.8846 (ep128) | 26:43:36 | L_res_0=0.0339 L_res_1=0.0381 L_res_2=0.0415
[2026-06-19 17:57:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 149): 19515.9 MiB
[2026-06-19 18:08:01] INFO segtask_v1.trainer.validation:   Val: loss=0.1954, pooled_mean_dice=0.8805, per_class=['0.8805'], iou=0.7865, recall=0.9865, precision=0.7951, vol_sim=0.8926, mcc=0.8594, min_class_dice=0.8805, coverage=[88]/88 samples
[2026-06-19 18:08:01] INFO segtask_v1.trainer.trainer: Epoch 150/400 | LR=9.83e-04 | loss=0.0382 | val_dice=0.8805 | best=0.8846 (ep128) | 26:54:17 | L_res_0=0.0357 L_res_1=0.0396 L_res_2=0.0437
[2026-06-19 18:08:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 150): 19516.8 MiB
[2026-06-19 18:18:45] INFO segtask_v1.trainer.validation:   Val: loss=0.1877, pooled_mean_dice=0.8844, per_class=['0.8844'], iou=0.7927, recall=0.9880, precision=0.8004, vol_sim=0.8951, mcc=0.8630, min_class_dice=0.8844, coverage=[88]/88 samples
[2026-06-19 18:18:45] INFO segtask_v1.trainer.trainer: Epoch 151/400 | LR=9.83e-04 | loss=0.0428 | val_dice=0.8844 | best=0.8846 (ep128) | 27:05:01 | L_res_0=0.0413 L_res_1=0.0454 L_res_2=0.0476
[2026-06-19 18:18:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 151): 19517.0 MiB
[2026-06-19 18:29:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2043, pooled_mean_dice=0.8757, per_class=['0.8757'], iou=0.7788, recall=0.9901, precision=0.7849, vol_sim=0.8844, mcc=0.8539, min_class_dice=0.8757, coverage=[88]/88 samples
[2026-06-19 18:29:27] INFO segtask_v1.trainer.trainer: Epoch 152/400 | LR=9.83e-04 | loss=0.0379 | val_dice=0.8757 | best=0.8846 (ep128) | 27:15:44 | L_res_0=0.0356 L_res_1=0.0398 L_res_2=0.0432
[2026-06-19 18:29:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 152): 19516.1 MiB
[2026-06-19 18:40:11] INFO segtask_v1.trainer.validation:   Val: loss=0.1980, pooled_mean_dice=0.8759, per_class=['0.8759'], iou=0.7792, recall=0.9875, precision=0.7869, vol_sim=0.8870, mcc=0.8546, min_class_dice=0.8759, coverage=[88]/88 samples
[2026-06-19 18:40:11] INFO segtask_v1.trainer.trainer: Epoch 153/400 | LR=9.83e-04 | loss=0.0421 | val_dice=0.8759 | best=0.8846 (ep128) | 27:26:27 | L_res_0=0.0412 L_res_1=0.0438 L_res_2=0.0472
[2026-06-19 18:40:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 153): 19517.8 MiB
[2026-06-19 18:50:55] INFO segtask_v1.trainer.validation:   Val: loss=0.2121, pooled_mean_dice=0.8774, per_class=['0.8774'], iou=0.7815, recall=0.9905, precision=0.7874, vol_sim=0.8858, mcc=0.8548, min_class_dice=0.8774, coverage=[88]/88 samples
[2026-06-19 18:50:55] INFO segtask_v1.trainer.trainer: Epoch 154/400 | LR=9.82e-04 | loss=0.0436 | val_dice=0.8774 | best=0.8846 (ep128) | 27:37:11 | L_res_0=0.0428 L_res_1=0.0457 L_res_2=0.0486
[2026-06-19 18:50:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 154): 19516.7 MiB
[2026-06-19 19:01:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2166, pooled_mean_dice=0.8653, per_class=['0.8653'], iou=0.7625, recall=0.9913, precision=0.7676, vol_sim=0.8728, mcc=0.8449, min_class_dice=0.8653, coverage=[88]/88 samples
[2026-06-19 19:01:36] INFO segtask_v1.trainer.trainer: Epoch 155/400 | LR=9.82e-04 | loss=0.0386 | val_dice=0.8653 | best=0.8846 (ep128) | 27:47:52 | L_res_0=0.0365 L_res_1=0.0406 L_res_2=0.0433
[2026-06-19 19:01:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 155): 19516.5 MiB
[2026-06-19 19:12:21] INFO segtask_v1.trainer.validation:   Val: loss=0.2009, pooled_mean_dice=0.8787, per_class=['0.8787'], iou=0.7836, recall=0.9900, precision=0.7899, vol_sim=0.8875, mcc=0.8572, min_class_dice=0.8787, coverage=[88]/88 samples
[2026-06-19 19:12:21] INFO segtask_v1.trainer.trainer: Epoch 156/400 | LR=9.82e-04 | loss=0.0363 | val_dice=0.8787 | best=0.8846 (ep128) | 27:58:37 | L_res_0=0.0335 L_res_1=0.0378 L_res_2=0.0416
[2026-06-19 19:12:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 156): 19515.7 MiB
[2026-06-19 19:23:02] INFO segtask_v1.trainer.validation:   Val: loss=0.2076, pooled_mean_dice=0.8717, per_class=['0.8717'], iou=0.7726, recall=0.9852, precision=0.7817, vol_sim=0.8848, mcc=0.8506, min_class_dice=0.8717, coverage=[88]/88 samples
[2026-06-19 19:23:02] INFO segtask_v1.trainer.trainer: Epoch 157/400 | LR=9.82e-04 | loss=0.0375 | val_dice=0.8717 | best=0.8846 (ep128) | 28:09:19 | L_res_0=0.0352 L_res_1=0.0390 L_res_2=0.0421
[2026-06-19 19:23:02] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 157): 19517.0 MiB
[2026-06-19 19:33:45] INFO segtask_v1.trainer.validation:   Val: loss=0.2030, pooled_mean_dice=0.8754, per_class=['0.8754'], iou=0.7785, recall=0.9871, precision=0.7865, vol_sim=0.8869, mcc=0.8546, min_class_dice=0.8754, coverage=[88]/88 samples
[2026-06-19 19:33:45] INFO segtask_v1.trainer.trainer: Epoch 158/400 | LR=9.81e-04 | loss=0.0360 | val_dice=0.8754 | best=0.8846 (ep128) | 28:20:01 | L_res_0=0.0327 L_res_1=0.0378 L_res_2=0.0415
[2026-06-19 19:33:45] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 158): 19516.9 MiB
[2026-06-19 19:44:26] INFO segtask_v1.trainer.validation:   Val: loss=0.1865, pooled_mean_dice=0.8852, per_class=['0.8852'], iou=0.7940, recall=0.9899, precision=0.8005, vol_sim=0.8942, mcc=0.8632, min_class_dice=0.8852, coverage=[88]/88 samples
[2026-06-19 19:44:32] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-19 19:44:32] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8852 at epoch 159
[2026-06-19 19:44:32] INFO segtask_v1.trainer.trainer: Epoch 159/400 | LR=9.81e-04 | loss=0.0359 | val_dice=0.8852 | best=0.8852 (ep159) | 28:30:49 | L_res_0=0.0335 L_res_1=0.0375 L_res_2=0.0406
[2026-06-19 19:44:32] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 159): 19516.7 MiB
[2026-06-19 19:55:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2076, pooled_mean_dice=0.8776, per_class=['0.8776'], iou=0.7819, recall=0.9890, precision=0.7888, vol_sim=0.8874, mcc=0.8557, min_class_dice=0.8776, coverage=[88]/88 samples
[2026-06-19 19:55:15] INFO segtask_v1.trainer.trainer: Epoch 160/400 | LR=9.81e-04 | loss=0.0354 | val_dice=0.8776 | best=0.8852 (ep159) | 28:41:31 | L_res_0=0.0329 L_res_1=0.0369 L_res_2=0.0402
[2026-06-19 19:55:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 160): 19517.1 MiB
[2026-06-19 20:05:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2062, pooled_mean_dice=0.8767, per_class=['0.8767'], iou=0.7804, recall=0.9890, precision=0.7872, vol_sim=0.8864, mcc=0.8570, min_class_dice=0.8767, coverage=[88]/88 samples
[2026-06-19 20:05:58] INFO segtask_v1.trainer.trainer: Epoch 161/400 | LR=9.80e-04 | loss=0.0353 | val_dice=0.8767 | best=0.8852 (ep159) | 28:52:15 | L_res_0=0.0322 L_res_1=0.0371 L_res_2=0.0405
[2026-06-19 20:05:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 161): 19516.4 MiB
[2026-06-19 20:16:41] INFO segtask_v1.trainer.validation:   Val: loss=0.2125, pooled_mean_dice=0.8773, per_class=['0.8773'], iou=0.7814, recall=0.9902, precision=0.7875, vol_sim=0.8860, mcc=0.8570, min_class_dice=0.8773, coverage=[88]/88 samples
[2026-06-19 20:16:41] INFO segtask_v1.trainer.trainer: Epoch 162/400 | LR=9.80e-04 | loss=0.0342 | val_dice=0.8773 | best=0.8852 (ep159) | 29:02:57 | L_res_0=0.0304 L_res_1=0.0358 L_res_2=0.0393
[2026-06-19 20:16:41] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 162): 19516.6 MiB
[2026-06-19 20:27:26] INFO segtask_v1.trainer.validation:   Val: loss=0.1862, pooled_mean_dice=0.8851, per_class=['0.8851'], iou=0.7939, recall=0.9892, precision=0.8009, vol_sim=0.8948, mcc=0.8641, min_class_dice=0.8851, coverage=[88]/88 samples
[2026-06-19 20:27:26] INFO segtask_v1.trainer.trainer: Epoch 163/400 | LR=9.80e-04 | loss=0.0354 | val_dice=0.8851 | best=0.8852 (ep159) | 29:13:42 | L_res_0=0.0323 L_res_1=0.0371 L_res_2=0.0405
[2026-06-19 20:27:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 163): 19517.1 MiB
[2026-06-19 20:38:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2040, pooled_mean_dice=0.8796, per_class=['0.8796'], iou=0.7852, recall=0.9895, precision=0.7917, vol_sim=0.8889, mcc=0.8588, min_class_dice=0.8796, coverage=[88]/88 samples
[2026-06-19 20:38:10] INFO segtask_v1.trainer.trainer: Epoch 164/400 | LR=9.80e-04 | loss=0.0359 | val_dice=0.8796 | best=0.8852 (ep159) | 29:24:27 | L_res_0=0.0322 L_res_1=0.0377 L_res_2=0.0415
[2026-06-19 20:38:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 164): 19517.1 MiB
[2026-06-19 20:48:55] INFO segtask_v1.trainer.validation:   Val: loss=0.1963, pooled_mean_dice=0.8824, per_class=['0.8824'], iou=0.7896, recall=0.9870, precision=0.7979, vol_sim=0.8940, mcc=0.8602, min_class_dice=0.8824, coverage=[88]/88 samples
[2026-06-19 20:48:55] INFO segtask_v1.trainer.trainer: Epoch 165/400 | LR=9.79e-04 | loss=0.0478 | val_dice=0.8824 | best=0.8852 (ep159) | 29:35:11 | L_res_0=0.0492 L_res_1=0.0500 L_res_2=0.0533
[2026-06-19 20:48:55] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 165): 19516.0 MiB
[2026-06-19 20:59:40] INFO segtask_v1.trainer.validation:   Val: loss=0.2064, pooled_mean_dice=0.8737, per_class=['0.8737'], iou=0.7756, recall=0.9890, precision=0.7824, vol_sim=0.8834, mcc=0.8530, min_class_dice=0.8737, coverage=[88]/88 samples
[2026-06-19 20:59:40] INFO segtask_v1.trainer.trainer: Epoch 166/400 | LR=9.79e-04 | loss=0.0393 | val_dice=0.8737 | best=0.8852 (ep159) | 29:45:56 | L_res_0=0.0378 L_res_1=0.0411 L_res_2=0.0448
[2026-06-19 20:59:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 166): 19516.2 MiB
[2026-06-19 21:10:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2309, pooled_mean_dice=0.8628, per_class=['0.8628'], iou=0.7587, recall=0.9893, precision=0.7650, vol_sim=0.8721, mcc=0.8413, min_class_dice=0.8628, coverage=[88]/88 samples
[2026-06-19 21:10:27] INFO segtask_v1.trainer.trainer: Epoch 167/400 | LR=9.79e-04 | loss=0.0602 | val_dice=0.8628 | best=0.8852 (ep159) | 29:56:43 | L_res_0=0.0655 L_res_1=0.0617 L_res_2=0.0686
[2026-06-19 21:10:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 167): 19516.5 MiB
[2026-06-19 21:21:12] INFO segtask_v1.trainer.validation:   Val: loss=0.2239, pooled_mean_dice=0.8666, per_class=['0.8666'], iou=0.7647, recall=0.9905, precision=0.7703, vol_sim=0.8749, mcc=0.8440, min_class_dice=0.8666, coverage=[88]/88 samples
[2026-06-19 21:21:12] INFO segtask_v1.trainer.trainer: Epoch 168/400 | LR=9.79e-04 | loss=0.0538 | val_dice=0.8666 | best=0.8852 (ep159) | 30:07:29 | L_res_0=0.0576 L_res_1=0.0565 L_res_2=0.0601
[2026-06-19 21:21:12] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 168): 19515.8 MiB
[2026-06-19 21:31:58] INFO segtask_v1.trainer.validation:   Val: loss=0.2074, pooled_mean_dice=0.8734, per_class=['0.8734'], iou=0.7753, recall=0.9922, precision=0.7800, vol_sim=0.8803, mcc=0.8517, min_class_dice=0.8734, coverage=[88]/88 samples
[2026-06-19 21:31:58] INFO segtask_v1.trainer.trainer: Epoch 169/400 | LR=9.78e-04 | loss=0.0418 | val_dice=0.8734 | best=0.8852 (ep159) | 30:18:14 | L_res_0=0.0405 L_res_1=0.0443 L_res_2=0.0480
[2026-06-19 21:31:58] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 169): 19516.9 MiB
[2026-06-19 21:42:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2218, pooled_mean_dice=0.8700, per_class=['0.8700'], iou=0.7700, recall=0.9900, precision=0.7760, vol_sim=0.8788, mcc=0.8464, min_class_dice=0.8700, coverage=[88]/88 samples
[2026-06-19 21:42:44] INFO segtask_v1.trainer.trainer: Epoch 170/400 | LR=9.78e-04 | loss=0.0389 | val_dice=0.8700 | best=0.8852 (ep159) | 30:29:00 | L_res_0=0.0366 L_res_1=0.0412 L_res_2=0.0451
[2026-06-19 21:42:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 170): 19517.1 MiB
[2026-06-19 21:53:26] INFO segtask_v1.trainer.validation:   Val: loss=0.2193, pooled_mean_dice=0.8671, per_class=['0.8671'], iou=0.7654, recall=0.9883, precision=0.7724, vol_sim=0.8774, mcc=0.8446, min_class_dice=0.8671, coverage=[88]/88 samples
[2026-06-19 21:53:26] INFO segtask_v1.trainer.trainer: Epoch 171/400 | LR=9.78e-04 | loss=0.0373 | val_dice=0.8671 | best=0.8852 (ep159) | 30:39:42 | L_res_0=0.0349 L_res_1=0.0394 L_res_2=0.0431
[2026-06-19 21:53:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 171): 19516.2 MiB
[2026-06-19 22:04:08] INFO segtask_v1.trainer.validation:   Val: loss=0.2227, pooled_mean_dice=0.8677, per_class=['0.8677'], iou=0.7662, recall=0.9900, precision=0.7722, vol_sim=0.8764, mcc=0.8459, min_class_dice=0.8677, coverage=[88]/88 samples
[2026-06-19 22:04:08] INFO segtask_v1.trainer.trainer: Epoch 172/400 | LR=9.77e-04 | loss=0.0362 | val_dice=0.8677 | best=0.8852 (ep159) | 30:50:24 | L_res_0=0.0332 L_res_1=0.0383 L_res_2=0.0421
[2026-06-19 22:04:08] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 172): 19516.1 MiB
[2026-06-19 22:14:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2073, pooled_mean_dice=0.8712, per_class=['0.8712'], iou=0.7717, recall=0.9900, precision=0.7778, vol_sim=0.8800, mcc=0.8509, min_class_dice=0.8712, coverage=[88]/88 samples
[2026-06-19 22:14:52] INFO segtask_v1.trainer.trainer: Epoch 173/400 | LR=9.77e-04 | loss=0.0393 | val_dice=0.8712 | best=0.8852 (ep159) | 31:01:08 | L_res_0=0.0377 L_res_1=0.0414 L_res_2=0.0446
[2026-06-19 22:14:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 173): 19516.7 MiB
[2026-06-19 22:25:37] INFO segtask_v1.trainer.validation:   Val: loss=0.2128, pooled_mean_dice=0.8718, per_class=['0.8718'], iou=0.7727, recall=0.9900, precision=0.7787, vol_sim=0.8806, mcc=0.8479, min_class_dice=0.8718, coverage=[88]/88 samples
[2026-06-19 22:25:37] INFO segtask_v1.trainer.trainer: Epoch 174/400 | LR=9.77e-04 | loss=0.0376 | val_dice=0.8718 | best=0.8852 (ep159) | 31:11:53 | L_res_0=0.0350 L_res_1=0.0398 L_res_2=0.0437
[2026-06-19 22:25:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 174): 19517.1 MiB
[2026-06-19 22:36:20] INFO segtask_v1.trainer.validation:   Val: loss=0.2128, pooled_mean_dice=0.8743, per_class=['0.8743'], iou=0.7767, recall=0.9890, precision=0.7835, vol_sim=0.8841, mcc=0.8527, min_class_dice=0.8743, coverage=[88]/88 samples
[2026-06-19 22:36:20] INFO segtask_v1.trainer.trainer: Epoch 175/400 | LR=9.76e-04 | loss=0.0358 | val_dice=0.8743 | best=0.8852 (ep159) | 31:22:37 | L_res_0=0.0325 L_res_1=0.0379 L_res_2=0.0413
[2026-06-19 22:36:20] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 175): 19516.4 MiB
[2026-06-19 22:47:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2246, pooled_mean_dice=0.8627, per_class=['0.8627'], iou=0.7586, recall=0.9876, precision=0.7659, vol_sim=0.8736, mcc=0.8432, min_class_dice=0.8627, coverage=[88]/88 samples
[2026-06-19 22:47:04] INFO segtask_v1.trainer.trainer: Epoch 176/400 | LR=9.76e-04 | loss=0.0354 | val_dice=0.8627 | best=0.8852 (ep159) | 31:33:21 | L_res_0=0.0319 L_res_1=0.0377 L_res_2=0.0408
[2026-06-19 22:47:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 176): 19517.3 MiB
[2026-06-19 22:57:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2161, pooled_mean_dice=0.8734, per_class=['0.8734'], iou=0.7753, recall=0.9897, precision=0.7816, vol_sim=0.8825, mcc=0.8541, min_class_dice=0.8734, coverage=[88]/88 samples
[2026-06-19 22:57:50] INFO segtask_v1.trainer.trainer: Epoch 177/400 | LR=9.76e-04 | loss=0.0361 | val_dice=0.8734 | best=0.8852 (ep159) | 31:44:06 | L_res_0=0.0339 L_res_1=0.0380 L_res_2=0.0412
[2026-06-19 22:57:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 177): 19516.7 MiB
[2026-06-19 23:08:34] INFO segtask_v1.trainer.validation:   Val: loss=0.2074, pooled_mean_dice=0.8791, per_class=['0.8791'], iou=0.7843, recall=0.9882, precision=0.7917, vol_sim=0.8896, mcc=0.8578, min_class_dice=0.8791, coverage=[88]/88 samples
[2026-06-19 23:08:34] INFO segtask_v1.trainer.trainer: Epoch 178/400 | LR=9.76e-04 | loss=0.0354 | val_dice=0.8791 | best=0.8852 (ep159) | 31:54:51 | L_res_0=0.0323 L_res_1=0.0372 L_res_2=0.0407
[2026-06-19 23:08:34] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 178): 19516.5 MiB
[2026-06-19 23:19:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2218, pooled_mean_dice=0.8661, per_class=['0.8661'], iou=0.7638, recall=0.9886, precision=0.7706, vol_sim=0.8761, mcc=0.8436, min_class_dice=0.8661, coverage=[88]/88 samples
[2026-06-19 23:19:15] INFO segtask_v1.trainer.trainer: Epoch 179/400 | LR=9.75e-04 | loss=0.0407 | val_dice=0.8661 | best=0.8852 (ep159) | 32:05:32 | L_res_0=0.0396 L_res_1=0.0425 L_res_2=0.0465
[2026-06-19 23:19:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 179): 19516.4 MiB
[2026-06-19 23:29:59] INFO segtask_v1.trainer.validation:   Val: loss=0.2159, pooled_mean_dice=0.8770, per_class=['0.8770'], iou=0.7809, recall=0.9900, precision=0.7871, vol_sim=0.8858, mcc=0.8549, min_class_dice=0.8770, coverage=[88]/88 samples
[2026-06-19 23:29:59] INFO segtask_v1.trainer.trainer: Epoch 180/400 | LR=9.75e-04 | loss=0.0366 | val_dice=0.8770 | best=0.8852 (ep159) | 32:16:15 | L_res_0=0.0335 L_res_1=0.0385 L_res_2=0.0424
[2026-06-19 23:29:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 180): 19516.3 MiB
[2026-06-19 23:40:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2115, pooled_mean_dice=0.8751, per_class=['0.8751'], iou=0.7779, recall=0.9887, precision=0.7849, vol_sim=0.8851, mcc=0.8529, min_class_dice=0.8751, coverage=[88]/88 samples
[2026-06-19 23:40:44] INFO segtask_v1.trainer.trainer: Epoch 181/400 | LR=9.75e-04 | loss=0.0353 | val_dice=0.8751 | best=0.8852 (ep159) | 32:27:01 | L_res_0=0.0323 L_res_1=0.0371 L_res_2=0.0407
[2026-06-19 23:40:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 181): 19516.0 MiB
[2026-06-19 23:51:27] INFO segtask_v1.trainer.validation:   Val: loss=0.2155, pooled_mean_dice=0.8722, per_class=['0.8722'], iou=0.7733, recall=0.9892, precision=0.7799, vol_sim=0.8817, mcc=0.8518, min_class_dice=0.8722, coverage=[88]/88 samples
[2026-06-19 23:51:27] INFO segtask_v1.trainer.trainer: Epoch 182/400 | LR=9.74e-04 | loss=0.0342 | val_dice=0.8722 | best=0.8852 (ep159) | 32:37:43 | L_res_0=0.0305 L_res_1=0.0361 L_res_2=0.0394
[2026-06-19 23:51:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 182): 19516.9 MiB
[2026-06-20 00:02:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2132, pooled_mean_dice=0.8716, per_class=['0.8716'], iou=0.7724, recall=0.9877, precision=0.7799, vol_sim=0.8825, mcc=0.8504, min_class_dice=0.8716, coverage=[88]/88 samples
[2026-06-20 00:02:10] INFO segtask_v1.trainer.trainer: Epoch 183/400 | LR=9.74e-04 | loss=0.0347 | val_dice=0.8716 | best=0.8852 (ep159) | 32:48:26 | L_res_0=0.0320 L_res_1=0.0362 L_res_2=0.0399
[2026-06-20 00:02:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 183): 19515.3 MiB
[2026-06-20 00:12:52] INFO segtask_v1.trainer.validation:   Val: loss=0.2038, pooled_mean_dice=0.8762, per_class=['0.8762'], iou=0.7796, recall=0.9858, precision=0.7885, vol_sim=0.8887, mcc=0.8554, min_class_dice=0.8762, coverage=[88]/88 samples
[2026-06-20 00:12:52] INFO segtask_v1.trainer.trainer: Epoch 184/400 | LR=9.74e-04 | loss=0.0342 | val_dice=0.8762 | best=0.8852 (ep159) | 32:59:09 | L_res_0=0.0311 L_res_1=0.0357 L_res_2=0.0394
[2026-06-20 00:12:52] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 184): 19515.5 MiB
[2026-06-20 00:23:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2060, pooled_mean_dice=0.8766, per_class=['0.8766'], iou=0.7804, recall=0.9870, precision=0.7885, vol_sim=0.8881, mcc=0.8558, min_class_dice=0.8766, coverage=[88]/88 samples
[2026-06-20 00:23:36] INFO segtask_v1.trainer.trainer: Epoch 185/400 | LR=9.73e-04 | loss=0.0345 | val_dice=0.8766 | best=0.8852 (ep159) | 33:09:53 | L_res_0=0.0317 L_res_1=0.0361 L_res_2=0.0397
[2026-06-20 00:23:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 185): 19516.0 MiB
[2026-06-20 00:34:20] INFO segtask_v1.trainer.validation:   Val: loss=0.1860, pooled_mean_dice=0.8869, per_class=['0.8869'], iou=0.7968, recall=0.9873, precision=0.8050, vol_sim=0.8983, mcc=0.8664, min_class_dice=0.8869, coverage=[88]/88 samples
[2026-06-20 00:34:26] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-20 00:34:26] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8869 at epoch 186
[2026-06-20 00:34:26] INFO segtask_v1.trainer.trainer: Epoch 186/400 | LR=9.73e-04 | loss=0.0331 | val_dice=0.8869 | best=0.8869 (ep186) | 33:20:42 | L_res_0=0.0297 L_res_1=0.0348 L_res_2=0.0383
[2026-06-20 00:34:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 186): 19515.8 MiB
[2026-06-20 00:45:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2014, pooled_mean_dice=0.8815, per_class=['0.8815'], iou=0.7881, recall=0.9873, precision=0.7961, vol_sim=0.8928, mcc=0.8619, min_class_dice=0.8815, coverage=[88]/88 samples
[2026-06-20 00:45:10] INFO segtask_v1.trainer.trainer: Epoch 187/400 | LR=9.73e-04 | loss=0.0338 | val_dice=0.8815 | best=0.8869 (ep186) | 33:31:27 | L_res_0=0.0300 L_res_1=0.0355 L_res_2=0.0390
[2026-06-20 00:45:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 187): 19515.8 MiB
[2026-06-20 00:55:53] INFO segtask_v1.trainer.validation:   Val: loss=0.1926, pooled_mean_dice=0.8825, per_class=['0.8825'], iou=0.7897, recall=0.9884, precision=0.7971, vol_sim=0.8928, mcc=0.8623, min_class_dice=0.8825, coverage=[88]/88 samples
[2026-06-20 00:55:53] INFO segtask_v1.trainer.trainer: Epoch 188/400 | LR=9.72e-04 | loss=0.0335 | val_dice=0.8825 | best=0.8869 (ep186) | 33:42:10 | L_res_0=0.0303 L_res_1=0.0350 L_res_2=0.0387
[2026-06-20 00:55:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 188): 19516.3 MiB
[2026-06-20 01:06:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2037, pooled_mean_dice=0.8777, per_class=['0.8777'], iou=0.7821, recall=0.9879, precision=0.7897, vol_sim=0.8885, mcc=0.8594, min_class_dice=0.8777, coverage=[88]/88 samples
[2026-06-20 01:06:35] INFO segtask_v1.trainer.trainer: Epoch 189/400 | LR=9.72e-04 | loss=0.0332 | val_dice=0.8777 | best=0.8869 (ep186) | 33:52:52 | L_res_0=0.0294 L_res_1=0.0349 L_res_2=0.0386
[2026-06-20 01:06:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 189): 19516.5 MiB
[2026-06-20 01:17:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2001, pooled_mean_dice=0.8802, per_class=['0.8802'], iou=0.7861, recall=0.9892, precision=0.7929, vol_sim=0.8899, mcc=0.8596, min_class_dice=0.8802, coverage=[88]/88 samples
[2026-06-20 01:17:22] INFO segtask_v1.trainer.trainer: Epoch 190/400 | LR=9.72e-04 | loss=0.0332 | val_dice=0.8802 | best=0.8869 (ep186) | 34:03:38 | L_res_0=0.0299 L_res_1=0.0347 L_res_2=0.0383
[2026-06-20 01:17:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 190): 19516.8 MiB
[2026-06-20 01:28:06] INFO segtask_v1.trainer.validation:   Val: loss=0.1902, pooled_mean_dice=0.8822, per_class=['0.8822'], iou=0.7893, recall=0.9871, precision=0.7975, vol_sim=0.8938, mcc=0.8625, min_class_dice=0.8822, coverage=[88]/88 samples
[2026-06-20 01:28:06] INFO segtask_v1.trainer.trainer: Epoch 191/400 | LR=9.71e-04 | loss=0.0331 | val_dice=0.8822 | best=0.8869 (ep186) | 34:14:22 | L_res_0=0.0295 L_res_1=0.0347 L_res_2=0.0383
[2026-06-20 01:28:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 191): 19517.2 MiB
[2026-06-20 01:38:50] INFO segtask_v1.trainer.validation:   Val: loss=0.1841, pooled_mean_dice=0.8873, per_class=['0.8873'], iou=0.7974, recall=0.9878, precision=0.8053, vol_sim=0.8983, mcc=0.8671, min_class_dice=0.8873, coverage=[88]/88 samples
[2026-06-20 01:38:56] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-20 01:38:56] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8873 at epoch 192
[2026-06-20 01:38:56] INFO segtask_v1.trainer.trainer: Epoch 192/400 | LR=9.71e-04 | loss=0.0334 | val_dice=0.8873 | best=0.8873 (ep192) | 34:25:12 | L_res_0=0.0297 L_res_1=0.0351 L_res_2=0.0388
[2026-06-20 01:38:56] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 192): 19516.2 MiB
[2026-06-20 01:49:40] INFO segtask_v1.trainer.validation:   Val: loss=0.1885, pooled_mean_dice=0.8842, per_class=['0.8842'], iou=0.7924, recall=0.9859, precision=0.8015, vol_sim=0.8968, mcc=0.8629, min_class_dice=0.8842, coverage=[88]/88 samples
[2026-06-20 01:49:40] INFO segtask_v1.trainer.trainer: Epoch 193/400 | LR=9.71e-04 | loss=0.0358 | val_dice=0.8842 | best=0.8873 (ep192) | 34:35:56 | L_res_0=0.0324 L_res_1=0.0381 L_res_2=0.0417
[2026-06-20 01:49:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 193): 19515.9 MiB
[2026-06-20 02:00:25] INFO segtask_v1.trainer.validation:   Val: loss=0.1869, pooled_mean_dice=0.8841, per_class=['0.8841'], iou=0.7922, recall=0.9900, precision=0.7987, vol_sim=0.8930, mcc=0.8639, min_class_dice=0.8841, coverage=[88]/88 samples
[2026-06-20 02:00:25] INFO segtask_v1.trainer.trainer: Epoch 194/400 | LR=9.70e-04 | loss=0.0389 | val_dice=0.8841 | best=0.8873 (ep192) | 34:46:41 | L_res_0=0.0379 L_res_1=0.0406 L_res_2=0.0436
[2026-06-20 02:00:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 194): 19516.4 MiB
[2026-06-20 02:11:09] INFO segtask_v1.trainer.validation:   Val: loss=0.2013, pooled_mean_dice=0.8807, per_class=['0.8807'], iou=0.7868, recall=0.9914, precision=0.7922, vol_sim=0.8883, mcc=0.8609, min_class_dice=0.8807, coverage=[88]/88 samples
[2026-06-20 02:11:09] INFO segtask_v1.trainer.trainer: Epoch 195/400 | LR=9.70e-04 | loss=0.0548 | val_dice=0.8807 | best=0.8873 (ep192) | 34:57:26 | L_res_0=0.0543 L_res_1=0.0584 L_res_2=0.0650
[2026-06-20 02:11:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 195): 19517.1 MiB
[2026-06-20 02:21:54] INFO segtask_v1.trainer.validation:   Val: loss=0.1982, pooled_mean_dice=0.8800, per_class=['0.8800'], iou=0.7857, recall=0.9889, precision=0.7927, vol_sim=0.8899, mcc=0.8586, min_class_dice=0.8800, coverage=[88]/88 samples
[2026-06-20 02:21:54] INFO segtask_v1.trainer.trainer: Epoch 196/400 | LR=9.70e-04 | loss=0.0411 | val_dice=0.8800 | best=0.8873 (ep192) | 35:08:10 | L_res_0=0.0390 L_res_1=0.0437 L_res_2=0.0478
[2026-06-20 02:21:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 196): 19517.1 MiB
[2026-06-20 02:32:40] INFO segtask_v1.trainer.validation:   Val: loss=0.2225, pooled_mean_dice=0.8662, per_class=['0.8662'], iou=0.7640, recall=0.9891, precision=0.7705, vol_sim=0.8758, mcc=0.8430, min_class_dice=0.8662, coverage=[88]/88 samples
[2026-06-20 02:32:40] INFO segtask_v1.trainer.trainer: Epoch 197/400 | LR=9.69e-04 | loss=0.0382 | val_dice=0.8662 | best=0.8873 (ep192) | 35:18:56 | L_res_0=0.0363 L_res_1=0.0406 L_res_2=0.0445
[2026-06-20 02:32:40] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 197): 19516.0 MiB
[2026-06-20 02:43:22] INFO segtask_v1.trainer.validation:   Val: loss=0.2185, pooled_mean_dice=0.8662, per_class=['0.8662'], iou=0.7640, recall=0.9884, precision=0.7709, vol_sim=0.8764, mcc=0.8451, min_class_dice=0.8662, coverage=[88]/88 samples
[2026-06-20 02:43:22] INFO segtask_v1.trainer.trainer: Epoch 198/400 | LR=9.69e-04 | loss=0.0364 | val_dice=0.8662 | best=0.8873 (ep192) | 35:29:38 | L_res_0=0.0334 L_res_1=0.0386 L_res_2=0.0421
[2026-06-20 02:43:22] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 198): 19517.0 MiB
[2026-06-20 02:54:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2253, pooled_mean_dice=0.8646, per_class=['0.8646'], iou=0.7615, recall=0.9885, precision=0.7683, vol_sim=0.8747, mcc=0.8417, min_class_dice=0.8646, coverage=[88]/88 samples
[2026-06-20 02:54:04] INFO segtask_v1.trainer.trainer: Epoch 199/400 | LR=9.69e-04 | loss=0.0351 | val_dice=0.8646 | best=0.8873 (ep192) | 35:40:20 | L_res_0=0.0317 L_res_1=0.0371 L_res_2=0.0411
[2026-06-20 02:54:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 199): 19515.6 MiB
[2026-06-20 03:04:50] INFO segtask_v1.trainer.validation:   Val: loss=0.2271, pooled_mean_dice=0.8657, per_class=['0.8657'], iou=0.7632, recall=0.9904, precision=0.7689, vol_sim=0.8741, mcc=0.8446, min_class_dice=0.8657, coverage=[88]/88 samples
[2026-06-20 03:04:50] INFO segtask_v1.trainer.trainer: Epoch 200/400 | LR=9.68e-04 | loss=0.0354 | val_dice=0.8657 | best=0.8873 (ep192) | 35:51:06 | L_res_0=0.0325 L_res_1=0.0373 L_res_2=0.0410
[2026-06-20 03:04:50] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 200): 19517.1 MiB
[2026-06-20 03:15:33] INFO segtask_v1.trainer.validation:   Val: loss=0.2104, pooled_mean_dice=0.8718, per_class=['0.8718'], iou=0.7727, recall=0.9893, precision=0.7792, vol_sim=0.8812, mcc=0.8503, min_class_dice=0.8718, coverage=[88]/88 samples
[2026-06-20 03:15:33] INFO segtask_v1.trainer.trainer: Epoch 201/400 | LR=9.68e-04 | loss=0.0348 | val_dice=0.8718 | best=0.8873 (ep192) | 36:01:49 | L_res_0=0.0322 L_res_1=0.0364 L_res_2=0.0400
[2026-06-20 03:15:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 201): 19516.6 MiB
[2026-06-20 03:26:18] INFO segtask_v1.trainer.validation:   Val: loss=0.2165, pooled_mean_dice=0.8685, per_class=['0.8685'], iou=0.7676, recall=0.9883, precision=0.7746, vol_sim=0.8788, mcc=0.8476, min_class_dice=0.8685, coverage=[88]/88 samples
[2026-06-20 03:26:18] INFO segtask_v1.trainer.trainer: Epoch 202/400 | LR=9.68e-04 | loss=0.0379 | val_dice=0.8685 | best=0.8873 (ep192) | 36:12:34 | L_res_0=0.0367 L_res_1=0.0400 L_res_2=0.0426
[2026-06-20 03:26:18] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 202): 19516.1 MiB
[2026-06-20 03:37:01] INFO segtask_v1.trainer.validation:   Val: loss=0.2088, pooled_mean_dice=0.8728, per_class=['0.8728'], iou=0.7744, recall=0.9898, precision=0.7806, vol_sim=0.8818, mcc=0.8516, min_class_dice=0.8728, coverage=[88]/88 samples
[2026-06-20 03:37:01] INFO segtask_v1.trainer.trainer: Epoch 203/400 | LR=9.67e-04 | loss=0.0350 | val_dice=0.8728 | best=0.8873 (ep192) | 36:23:17 | L_res_0=0.0326 L_res_1=0.0365 L_res_2=0.0399
[2026-06-20 03:37:01] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 203): 19516.3 MiB
[2026-06-20 03:47:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2159, pooled_mean_dice=0.8734, per_class=['0.8734'], iou=0.7753, recall=0.9877, precision=0.7829, vol_sim=0.8843, mcc=0.8525, min_class_dice=0.8734, coverage=[88]/88 samples
[2026-06-20 03:47:44] INFO segtask_v1.trainer.trainer: Epoch 204/400 | LR=9.67e-04 | loss=0.0338 | val_dice=0.8734 | best=0.8873 (ep192) | 36:34:00 | L_res_0=0.0307 L_res_1=0.0355 L_res_2=0.0391
[2026-06-20 03:47:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 204): 19517.3 MiB
[2026-06-20 03:58:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2088, pooled_mean_dice=0.8780, per_class=['0.8780'], iou=0.7825, recall=0.9887, precision=0.7896, vol_sim=0.8880, mcc=0.8569, min_class_dice=0.8780, coverage=[88]/88 samples
[2026-06-20 03:58:28] INFO segtask_v1.trainer.trainer: Epoch 205/400 | LR=9.67e-04 | loss=0.0337 | val_dice=0.8780 | best=0.8873 (ep192) | 36:44:44 | L_res_0=0.0299 L_res_1=0.0355 L_res_2=0.0393
[2026-06-20 03:58:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 205): 19516.7 MiB
[2026-06-20 04:09:11] INFO segtask_v1.trainer.validation:   Val: loss=0.2125, pooled_mean_dice=0.8689, per_class=['0.8689'], iou=0.7682, recall=0.9879, precision=0.7754, vol_sim=0.8795, mcc=0.8471, min_class_dice=0.8689, coverage=[88]/88 samples
[2026-06-20 04:09:11] INFO segtask_v1.trainer.trainer: Epoch 206/400 | LR=9.66e-04 | loss=0.0529 | val_dice=0.8689 | best=0.8873 (ep192) | 36:55:27 | L_res_0=0.0528 L_res_1=0.0549 L_res_2=0.0598
[2026-06-20 04:09:11] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 206): 19516.3 MiB
[2026-06-20 04:19:54] INFO segtask_v1.trainer.validation:   Val: loss=0.2214, pooled_mean_dice=0.8677, per_class=['0.8677'], iou=0.7664, recall=0.9910, precision=0.7718, vol_sim=0.8757, mcc=0.8461, min_class_dice=0.8677, coverage=[88]/88 samples
[2026-06-20 04:19:54] INFO segtask_v1.trainer.trainer: Epoch 207/400 | LR=9.66e-04 | loss=0.0423 | val_dice=0.8677 | best=0.8873 (ep192) | 37:06:10 | L_res_0=0.0402 L_res_1=0.0442 L_res_2=0.0487
[2026-06-20 04:19:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 207): 19516.3 MiB
[2026-06-20 04:30:36] INFO segtask_v1.trainer.validation:   Val: loss=0.2201, pooled_mean_dice=0.8710, per_class=['0.8710'], iou=0.7715, recall=0.9894, precision=0.7779, vol_sim=0.8804, mcc=0.8494, min_class_dice=0.8710, coverage=[88]/88 samples
[2026-06-20 04:30:36] INFO segtask_v1.trainer.trainer: Epoch 208/400 | LR=9.66e-04 | loss=0.0367 | val_dice=0.8710 | best=0.8873 (ep192) | 37:16:52 | L_res_0=0.0342 L_res_1=0.0386 L_res_2=0.0423
[2026-06-20 04:30:36] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 208): 19517.0 MiB
[2026-06-20 04:41:23] INFO segtask_v1.trainer.validation:   Val: loss=0.2102, pooled_mean_dice=0.8709, per_class=['0.8709'], iou=0.7713, recall=0.9905, precision=0.7771, vol_sim=0.8792, mcc=0.8501, min_class_dice=0.8709, coverage=[88]/88 samples
[2026-06-20 04:41:23] INFO segtask_v1.trainer.trainer: Epoch 209/400 | LR=9.65e-04 | loss=0.0352 | val_dice=0.8709 | best=0.8873 (ep192) | 37:27:39 | L_res_0=0.0315 L_res_1=0.0371 L_res_2=0.0408
[2026-06-20 04:41:23] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 209): 19516.3 MiB
[2026-06-20 04:52:04] INFO segtask_v1.trainer.validation:   Val: loss=0.2419, pooled_mean_dice=0.8566, per_class=['0.8566'], iou=0.7492, recall=0.9898, precision=0.7551, vol_sim=0.8655, mcc=0.8348, min_class_dice=0.8566, coverage=[88]/88 samples
[2026-06-20 04:52:04] INFO segtask_v1.trainer.trainer: Epoch 210/400 | LR=9.65e-04 | loss=0.0347 | val_dice=0.8566 | best=0.8873 (ep192) | 37:38:21 | L_res_0=0.0322 L_res_1=0.0365 L_res_2=0.0397
[2026-06-20 04:52:04] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 210): 19516.5 MiB
[2026-06-20 05:02:49] INFO segtask_v1.trainer.validation:   Val: loss=0.2312, pooled_mean_dice=0.8669, per_class=['0.8669'], iou=0.7651, recall=0.9882, precision=0.7722, vol_sim=0.8773, mcc=0.8437, min_class_dice=0.8669, coverage=[88]/88 samples
[2026-06-20 05:02:49] INFO segtask_v1.trainer.trainer: Epoch 211/400 | LR=9.64e-04 | loss=0.0359 | val_dice=0.8669 | best=0.8873 (ep192) | 37:49:05 | L_res_0=0.0342 L_res_1=0.0372 L_res_2=0.0398
[2026-06-20 05:02:49] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 211): 19516.8 MiB
[2026-06-20 05:13:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2239, pooled_mean_dice=0.8660, per_class=['0.8660'], iou=0.7637, recall=0.9873, precision=0.7713, vol_sim=0.8772, mcc=0.8468, min_class_dice=0.8660, coverage=[88]/88 samples
[2026-06-20 05:13:30] INFO segtask_v1.trainer.trainer: Epoch 212/400 | LR=9.64e-04 | loss=0.0340 | val_dice=0.8660 | best=0.8873 (ep192) | 37:59:46 | L_res_0=0.0309 L_res_1=0.0359 L_res_2=0.0390
[2026-06-20 05:13:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 212): 19516.8 MiB
[2026-06-20 05:24:14] INFO segtask_v1.trainer.validation:   Val: loss=0.2065, pooled_mean_dice=0.8735, per_class=['0.8735'], iou=0.7755, recall=0.9900, precision=0.7816, vol_sim=0.8823, mcc=0.8518, min_class_dice=0.8735, coverage=[88]/88 samples
[2026-06-20 05:24:14] INFO segtask_v1.trainer.trainer: Epoch 213/400 | LR=9.64e-04 | loss=0.0349 | val_dice=0.8735 | best=0.8873 (ep192) | 38:10:30 | L_res_0=0.0319 L_res_1=0.0372 L_res_2=0.0402
[2026-06-20 05:24:14] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 213): 19515.8 MiB
[2026-06-20 05:34:57] INFO segtask_v1.trainer.validation:   Val: loss=0.2124, pooled_mean_dice=0.8772, per_class=['0.8772'], iou=0.7812, recall=0.9888, precision=0.7882, vol_sim=0.8871, mcc=0.8535, min_class_dice=0.8772, coverage=[88]/88 samples
[2026-06-20 05:34:57] INFO segtask_v1.trainer.trainer: Epoch 214/400 | LR=9.63e-04 | loss=0.0333 | val_dice=0.8772 | best=0.8873 (ep192) | 38:21:13 | L_res_0=0.0295 L_res_1=0.0351 L_res_2=0.0389
[2026-06-20 05:34:57] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 214): 19516.3 MiB
[2026-06-20 05:45:44] INFO segtask_v1.trainer.validation:   Val: loss=0.2101, pooled_mean_dice=0.8729, per_class=['0.8729'], iou=0.7745, recall=0.9865, precision=0.7828, vol_sim=0.8849, mcc=0.8519, min_class_dice=0.8729, coverage=[88]/88 samples
[2026-06-20 05:45:44] INFO segtask_v1.trainer.trainer: Epoch 215/400 | LR=9.63e-04 | loss=0.0333 | val_dice=0.8729 | best=0.8873 (ep192) | 38:32:00 | L_res_0=0.0294 L_res_1=0.0351 L_res_2=0.0389
[2026-06-20 05:45:44] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 215): 19516.3 MiB
[2026-06-20 05:56:30] INFO segtask_v1.trainer.validation:   Val: loss=0.2052, pooled_mean_dice=0.8785, per_class=['0.8785'], iou=0.7832, recall=0.9864, precision=0.7918, vol_sim=0.8906, mcc=0.8582, min_class_dice=0.8785, coverage=[88]/88 samples
[2026-06-20 05:56:30] INFO segtask_v1.trainer.trainer: Epoch 216/400 | LR=9.63e-04 | loss=0.0326 | val_dice=0.8785 | best=0.8873 (ep192) | 38:42:46 | L_res_0=0.0290 L_res_1=0.0342 L_res_2=0.0376
[2026-06-20 05:56:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 216): 19516.5 MiB
[2026-06-20 06:07:15] INFO segtask_v1.trainer.validation:   Val: loss=0.1914, pooled_mean_dice=0.8816, per_class=['0.8816'], iou=0.7882, recall=0.9884, precision=0.7956, vol_sim=0.8919, mcc=0.8598, min_class_dice=0.8816, coverage=[88]/88 samples
[2026-06-20 06:07:15] INFO segtask_v1.trainer.trainer: Epoch 217/400 | LR=9.62e-04 | loss=0.0327 | val_dice=0.8816 | best=0.8873 (ep192) | 38:53:31 | L_res_0=0.0293 L_res_1=0.0343 L_res_2=0.0378
[2026-06-20 06:07:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 217): 19516.8 MiB
[2026-06-20 06:18:00] INFO segtask_v1.trainer.validation:   Val: loss=0.2042, pooled_mean_dice=0.8795, per_class=['0.8795'], iou=0.7849, recall=0.9890, precision=0.7918, vol_sim=0.8892, mcc=0.8585, min_class_dice=0.8795, coverage=[88]/88 samples
[2026-06-20 06:18:00] INFO segtask_v1.trainer.trainer: Epoch 218/400 | LR=9.62e-04 | loss=0.0385 | val_dice=0.8795 | best=0.8873 (ep192) | 39:04:16 | L_res_0=0.0360 L_res_1=0.0406 L_res_2=0.0444
[2026-06-20 06:18:00] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 218): 19517.2 MiB
[2026-06-20 06:28:43] INFO segtask_v1.trainer.validation:   Val: loss=0.2111, pooled_mean_dice=0.8725, per_class=['0.8725'], iou=0.7739, recall=0.9888, precision=0.7808, vol_sim=0.8824, mcc=0.8501, min_class_dice=0.8725, coverage=[88]/88 samples
[2026-06-20 06:28:43] INFO segtask_v1.trainer.trainer: Epoch 219/400 | LR=9.61e-04 | loss=0.0473 | val_dice=0.8725 | best=0.8873 (ep192) | 39:15:00 | L_res_0=0.0469 L_res_1=0.0500 L_res_2=0.0530
[2026-06-20 06:28:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 219): 19516.6 MiB
[2026-06-20 06:39:28] INFO segtask_v1.trainer.validation:   Val: loss=0.2000, pooled_mean_dice=0.8829, per_class=['0.8829'], iou=0.7904, recall=0.9869, precision=0.7988, vol_sim=0.8947, mcc=0.8610, min_class_dice=0.8829, coverage=[88]/88 samples
[2026-06-20 06:39:28] INFO segtask_v1.trainer.trainer: Epoch 220/400 | LR=9.61e-04 | loss=0.0370 | val_dice=0.8829 | best=0.8873 (ep192) | 39:25:44 | L_res_0=0.0331 L_res_1=0.0393 L_res_2=0.0434
[2026-06-20 06:39:28] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 220): 19516.3 MiB
[2026-06-20 06:50:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2223, pooled_mean_dice=0.8678, per_class=['0.8678'], iou=0.7665, recall=0.9894, precision=0.7729, vol_sim=0.8772, mcc=0.8450, min_class_dice=0.8678, coverage=[88]/88 samples
[2026-06-20 06:50:10] INFO segtask_v1.trainer.trainer: Epoch 221/400 | LR=9.61e-04 | loss=0.0350 | val_dice=0.8678 | best=0.8873 (ep192) | 39:36:26 | L_res_0=0.0320 L_res_1=0.0370 L_res_2=0.0402
[2026-06-20 06:50:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 221): 19516.3 MiB
[2026-06-20 07:00:53] INFO segtask_v1.trainer.validation:   Val: loss=0.1948, pooled_mean_dice=0.8787, per_class=['0.8787'], iou=0.7836, recall=0.9877, precision=0.7913, vol_sim=0.8896, mcc=0.8575, min_class_dice=0.8787, coverage=[88]/88 samples
[2026-06-20 07:00:53] INFO segtask_v1.trainer.trainer: Epoch 222/400 | LR=9.60e-04 | loss=0.0343 | val_dice=0.8787 | best=0.8873 (ep192) | 39:47:09 | L_res_0=0.0311 L_res_1=0.0361 L_res_2=0.0396
[2026-06-20 07:00:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 222): 19515.8 MiB
[2026-06-20 07:11:37] INFO segtask_v1.trainer.validation:   Val: loss=0.1826, pooled_mean_dice=0.8880, per_class=['0.8880'], iou=0.7986, recall=0.9879, precision=0.8065, vol_sim=0.8989, mcc=0.8659, min_class_dice=0.8880, coverage=[88]/88 samples
[2026-06-20 07:11:43] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-20 07:11:43] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8880 at epoch 223
[2026-06-20 07:11:43] INFO segtask_v1.trainer.trainer: Epoch 223/400 | LR=9.60e-04 | loss=0.0341 | val_dice=0.8880 | best=0.8880 (ep223) | 39:58:00 | L_res_0=0.0307 L_res_1=0.0358 L_res_2=0.0394
[2026-06-20 07:11:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 223): 19517.3 MiB
[2026-06-20 07:22:27] INFO segtask_v1.trainer.validation:   Val: loss=0.1939, pooled_mean_dice=0.8834, per_class=['0.8834'], iou=0.7912, recall=0.9872, precision=0.7994, vol_sim=0.8948, mcc=0.8630, min_class_dice=0.8834, coverage=[88]/88 samples
[2026-06-20 07:22:27] INFO segtask_v1.trainer.trainer: Epoch 224/400 | LR=9.59e-04 | loss=0.0330 | val_dice=0.8834 | best=0.8880 (ep223) | 40:08:43 | L_res_0=0.0303 L_res_1=0.0345 L_res_2=0.0379
[2026-06-20 07:22:27] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 224): 19516.8 MiB
[2026-06-20 07:33:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2138, pooled_mean_dice=0.8707, per_class=['0.8707'], iou=0.7711, recall=0.9881, precision=0.7783, vol_sim=0.8813, mcc=0.8498, min_class_dice=0.8707, coverage=[88]/88 samples
[2026-06-20 07:33:10] INFO segtask_v1.trainer.trainer: Epoch 225/400 | LR=9.59e-04 | loss=0.0331 | val_dice=0.8707 | best=0.8880 (ep223) | 40:19:26 | L_res_0=0.0301 L_res_1=0.0348 L_res_2=0.0381
[2026-06-20 07:33:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 225): 19516.2 MiB
[2026-06-20 07:43:53] INFO segtask_v1.trainer.validation:   Val: loss=0.2168, pooled_mean_dice=0.8706, per_class=['0.8706'], iou=0.7708, recall=0.9881, precision=0.7781, vol_sim=0.8811, mcc=0.8506, min_class_dice=0.8706, coverage=[88]/88 samples
[2026-06-20 07:43:53] INFO segtask_v1.trainer.trainer: Epoch 226/400 | LR=9.59e-04 | loss=0.0329 | val_dice=0.8706 | best=0.8880 (ep223) | 40:30:09 | L_res_0=0.0292 L_res_1=0.0345 L_res_2=0.0381
[2026-06-20 07:43:53] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 226): 19516.0 MiB
[2026-06-20 07:54:35] INFO segtask_v1.trainer.validation:   Val: loss=0.2086, pooled_mean_dice=0.8747, per_class=['0.8747'], iou=0.7773, recall=0.9874, precision=0.7850, vol_sim=0.8858, mcc=0.8560, min_class_dice=0.8747, coverage=[88]/88 samples
[2026-06-20 07:54:35] INFO segtask_v1.trainer.trainer: Epoch 227/400 | LR=9.58e-04 | loss=0.0327 | val_dice=0.8747 | best=0.8880 (ep223) | 40:40:52 | L_res_0=0.0295 L_res_1=0.0346 L_res_2=0.0380
[2026-06-20 07:54:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 227): 19517.3 MiB
[2026-06-20 08:05:19] INFO segtask_v1.trainer.validation:   Val: loss=0.2037, pooled_mean_dice=0.8775, per_class=['0.8775'], iou=0.7817, recall=0.9887, precision=0.7888, vol_sim=0.8875, mcc=0.8567, min_class_dice=0.8775, coverage=[88]/88 samples
[2026-06-20 08:05:19] INFO segtask_v1.trainer.trainer: Epoch 228/400 | LR=9.58e-04 | loss=0.0331 | val_dice=0.8775 | best=0.8880 (ep223) | 40:51:35 | L_res_0=0.0303 L_res_1=0.0348 L_res_2=0.0380
[2026-06-20 08:05:19] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 228): 19517.2 MiB
[2026-06-20 08:16:03] INFO segtask_v1.trainer.validation:   Val: loss=0.2060, pooled_mean_dice=0.8789, per_class=['0.8789'], iou=0.7840, recall=0.9882, precision=0.7914, vol_sim=0.8894, mcc=0.8583, min_class_dice=0.8789, coverage=[88]/88 samples
[2026-06-20 08:16:03] INFO segtask_v1.trainer.trainer: Epoch 229/400 | LR=9.57e-04 | loss=0.0338 | val_dice=0.8789 | best=0.8880 (ep223) | 41:02:19 | L_res_0=0.0308 L_res_1=0.0350 L_res_2=0.0386
[2026-06-20 08:16:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 229): 19516.1 MiB
[2026-06-20 08:26:46] INFO segtask_v1.trainer.validation:   Val: loss=0.2187, pooled_mean_dice=0.8722, per_class=['0.8722'], iou=0.7734, recall=0.9898, precision=0.7796, vol_sim=0.8812, mcc=0.8524, min_class_dice=0.8722, coverage=[88]/88 samples
[2026-06-20 08:26:46] INFO segtask_v1.trainer.trainer: Epoch 230/400 | LR=9.57e-04 | loss=0.0332 | val_dice=0.8722 | best=0.8880 (ep223) | 41:13:02 | L_res_0=0.0295 L_res_1=0.0347 L_res_2=0.0382
[2026-06-20 08:26:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 230): 19516.7 MiB
[2026-06-20 08:37:30] INFO segtask_v1.trainer.validation:   Val: loss=0.1984, pooled_mean_dice=0.8826, per_class=['0.8826'], iou=0.7898, recall=0.9871, precision=0.7981, vol_sim=0.8941, mcc=0.8614, min_class_dice=0.8826, coverage=[88]/88 samples
[2026-06-20 08:37:30] INFO segtask_v1.trainer.trainer: Epoch 231/400 | LR=9.57e-04 | loss=0.0326 | val_dice=0.8826 | best=0.8880 (ep223) | 41:23:47 | L_res_0=0.0289 L_res_1=0.0343 L_res_2=0.0380
[2026-06-20 08:37:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 231): 19516.8 MiB
[2026-06-20 08:48:15] INFO segtask_v1.trainer.validation:   Val: loss=0.2148, pooled_mean_dice=0.8771, per_class=['0.8771'], iou=0.7811, recall=0.9846, precision=0.7908, vol_sim=0.8908, mcc=0.8586, min_class_dice=0.8771, coverage=[88]/88 samples
[2026-06-20 08:48:15] INFO segtask_v1.trainer.trainer: Epoch 232/400 | LR=9.56e-04 | loss=0.0319 | val_dice=0.8771 | best=0.8880 (ep223) | 41:34:31 | L_res_0=0.0279 L_res_1=0.0335 L_res_2=0.0370
[2026-06-20 08:48:15] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 232): 19516.3 MiB
[2026-06-20 08:58:57] INFO segtask_v1.trainer.validation:   Val: loss=0.1809, pooled_mean_dice=0.8898, per_class=['0.8898'], iou=0.8015, recall=0.9893, precision=0.8086, vol_sim=0.8995, mcc=0.8686, min_class_dice=0.8898, coverage=[88]/88 samples
[2026-06-20 08:59:03] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-20 08:59:03] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8898 at epoch 233
[2026-06-20 08:59:03] INFO segtask_v1.trainer.trainer: Epoch 233/400 | LR=9.56e-04 | loss=0.0316 | val_dice=0.8898 | best=0.8898 (ep233) | 41:45:19 | L_res_0=0.0280 L_res_1=0.0330 L_res_2=0.0367
[2026-06-20 08:59:03] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 233): 19517.6 MiB
[2026-06-20 09:09:46] INFO segtask_v1.trainer.validation:   Val: loss=0.1956, pooled_mean_dice=0.8825, per_class=['0.8825'], iou=0.7896, recall=0.9873, precision=0.7978, vol_sim=0.8938, mcc=0.8628, min_class_dice=0.8825, coverage=[88]/88 samples
[2026-06-20 09:09:46] INFO segtask_v1.trainer.trainer: Epoch 234/400 | LR=9.55e-04 | loss=0.0316 | val_dice=0.8825 | best=0.8898 (ep233) | 41:56:02 | L_res_0=0.0276 L_res_1=0.0332 L_res_2=0.0368
[2026-06-20 09:09:46] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 234): 19516.7 MiB
[2026-06-20 09:20:30] INFO segtask_v1.trainer.validation:   Val: loss=0.1857, pooled_mean_dice=0.8865, per_class=['0.8865'], iou=0.7962, recall=0.9844, precision=0.8064, vol_sim=0.9006, mcc=0.8650, min_class_dice=0.8865, coverage=[88]/88 samples
[2026-06-20 09:20:30] INFO segtask_v1.trainer.trainer: Epoch 235/400 | LR=9.55e-04 | loss=0.0317 | val_dice=0.8865 | best=0.8898 (ep233) | 42:06:47 | L_res_0=0.0280 L_res_1=0.0333 L_res_2=0.0369
[2026-06-20 09:20:30] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 235): 19516.4 MiB
[2026-06-20 09:31:16] INFO segtask_v1.trainer.validation:   Val: loss=0.1841, pooled_mean_dice=0.8871, per_class=['0.8871'], iou=0.7971, recall=0.9890, precision=0.8042, vol_sim=0.8970, mcc=0.8671, min_class_dice=0.8871, coverage=[88]/88 samples
[2026-06-20 09:31:16] INFO segtask_v1.trainer.trainer: Epoch 236/400 | LR=9.55e-04 | loss=0.0316 | val_dice=0.8871 | best=0.8898 (ep233) | 42:17:32 | L_res_0=0.0278 L_res_1=0.0333 L_res_2=0.0368
[2026-06-20 09:31:16] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 236): 19516.1 MiB
[2026-06-20 09:42:00] INFO segtask_v1.trainer.validation:   Val: loss=0.1797, pooled_mean_dice=0.8916, per_class=['0.8916'], iou=0.8044, recall=0.9901, precision=0.8109, vol_sim=0.9005, mcc=0.8715, min_class_dice=0.8916, coverage=[88]/88 samples
[2026-06-20 09:42:06] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-20 09:42:06] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8916 at epoch 237
[2026-06-20 09:42:06] INFO segtask_v1.trainer.trainer: Epoch 237/400 | LR=9.54e-04 | loss=0.0322 | val_dice=0.8916 | best=0.8916 (ep237) | 42:28:22 | L_res_0=0.0286 L_res_1=0.0337 L_res_2=0.0371
[2026-06-20 09:42:06] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 237): 19516.6 MiB
[2026-06-20 09:52:48] INFO segtask_v1.trainer.validation:   Val: loss=0.2026, pooled_mean_dice=0.8805, per_class=['0.8805'], iou=0.7865, recall=0.9873, precision=0.7945, vol_sim=0.8918, mcc=0.8618, min_class_dice=0.8805, coverage=[88]/88 samples
[2026-06-20 09:52:48] INFO segtask_v1.trainer.trainer: Epoch 238/400 | LR=9.54e-04 | loss=0.0323 | val_dice=0.8805 | best=0.8916 (ep237) | 42:39:04 | L_res_0=0.0289 L_res_1=0.0342 L_res_2=0.0371
[2026-06-20 09:52:48] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 238): 19516.7 MiB
[2026-06-20 10:03:33] INFO segtask_v1.trainer.validation:   Val: loss=0.1772, pooled_mean_dice=0.8909, per_class=['0.8909'], iou=0.8033, recall=0.9858, precision=0.8127, vol_sim=0.9037, mcc=0.8705, min_class_dice=0.8909, coverage=[88]/88 samples
[2026-06-20 10:03:33] INFO segtask_v1.trainer.trainer: Epoch 239/400 | LR=9.53e-04 | loss=0.0318 | val_dice=0.8909 | best=0.8916 (ep237) | 42:49:49 | L_res_0=0.0277 L_res_1=0.0333 L_res_2=0.0373
[2026-06-20 10:03:33] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 239): 19516.2 MiB
[2026-06-20 10:14:17] INFO segtask_v1.trainer.validation:   Val: loss=0.2027, pooled_mean_dice=0.8770, per_class=['0.8770'], iou=0.7809, recall=0.9852, precision=0.7902, vol_sim=0.8902, mcc=0.8578, min_class_dice=0.8770, coverage=[88]/88 samples
[2026-06-20 10:14:17] INFO segtask_v1.trainer.trainer: Epoch 240/400 | LR=9.53e-04 | loss=0.0315 | val_dice=0.8770 | best=0.8916 (ep237) | 43:00:33 | L_res_0=0.0275 L_res_1=0.0330 L_res_2=0.0367
[2026-06-20 10:14:17] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 240): 19516.0 MiB
[2026-06-20 10:24:59] INFO segtask_v1.trainer.validation:   Val: loss=0.1814, pooled_mean_dice=0.8914, per_class=['0.8914'], iou=0.8041, recall=0.9886, precision=0.8116, vol_sim=0.9017, mcc=0.8722, min_class_dice=0.8914, coverage=[88]/88 samples
[2026-06-20 10:24:59] INFO segtask_v1.trainer.trainer: Epoch 241/400 | LR=9.53e-04 | loss=0.0313 | val_dice=0.8914 | best=0.8916 (ep237) | 43:11:16 | L_res_0=0.0275 L_res_1=0.0328 L_res_2=0.0364
[2026-06-20 10:24:59] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 241): 19515.8 MiB
[2026-06-20 10:35:43] INFO segtask_v1.trainer.validation:   Val: loss=0.1895, pooled_mean_dice=0.8838, per_class=['0.8838'], iou=0.7918, recall=0.9881, precision=0.7994, vol_sim=0.8944, mcc=0.8645, min_class_dice=0.8838, coverage=[88]/88 samples
[2026-06-20 10:35:43] INFO segtask_v1.trainer.trainer: Epoch 242/400 | LR=9.52e-04 | loss=0.0310 | val_dice=0.8838 | best=0.8916 (ep237) | 43:21:59 | L_res_0=0.0270 L_res_1=0.0326 L_res_2=0.0359
[2026-06-20 10:35:43] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 242): 19516.0 MiB
[2026-06-20 10:46:26] INFO segtask_v1.trainer.validation:   Val: loss=0.1818, pooled_mean_dice=0.8915, per_class=['0.8915'], iou=0.8043, recall=0.9862, precision=0.8134, vol_sim=0.9040, mcc=0.8704, min_class_dice=0.8915, coverage=[88]/88 samples
[2026-06-20 10:46:26] INFO segtask_v1.trainer.trainer: Epoch 243/400 | LR=9.52e-04 | loss=0.0316 | val_dice=0.8915 | best=0.8916 (ep237) | 43:32:42 | L_res_0=0.0277 L_res_1=0.0331 L_res_2=0.0367
[2026-06-20 10:46:26] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 243): 19516.4 MiB
[2026-06-20 10:57:10] INFO segtask_v1.trainer.validation:   Val: loss=0.2066, pooled_mean_dice=0.8802, per_class=['0.8802'], iou=0.7861, recall=0.9877, precision=0.7939, vol_sim=0.8912, mcc=0.8603, min_class_dice=0.8802, coverage=[88]/88 samples
[2026-06-20 10:57:10] INFO segtask_v1.trainer.trainer: Epoch 244/400 | LR=9.51e-04 | loss=0.0320 | val_dice=0.8802 | best=0.8916 (ep237) | 43:43:26 | L_res_0=0.0282 L_res_1=0.0335 L_res_2=0.0370
[2026-06-20 10:57:10] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 244): 19516.5 MiB
[2026-06-20 11:07:54] INFO segtask_v1.trainer.validation:   Val: loss=0.1916, pooled_mean_dice=0.8866, per_class=['0.8866'], iou=0.7964, recall=0.9890, precision=0.8034, vol_sim=0.8965, mcc=0.8669, min_class_dice=0.8866, coverage=[88]/88 samples
[2026-06-20 11:07:54] INFO segtask_v1.trainer.trainer: Epoch 245/400 | LR=9.51e-04 | loss=0.0318 | val_dice=0.8866 | best=0.8916 (ep237) | 43:54:10 | L_res_0=0.0282 L_res_1=0.0332 L_res_2=0.0369
[2026-06-20 11:07:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 245): 19516.3 MiB
[2026-06-20 11:18:35] INFO segtask_v1.trainer.validation:   Val: loss=0.1998, pooled_mean_dice=0.8808, per_class=['0.8808'], iou=0.7871, recall=0.9865, precision=0.7956, vol_sim=0.8929, mcc=0.8623, min_class_dice=0.8808, coverage=[88]/88 samples
[2026-06-20 11:18:35] INFO segtask_v1.trainer.trainer: Epoch 246/400 | LR=9.50e-04 | loss=0.0317 | val_dice=0.8808 | best=0.8916 (ep237) | 44:04:51 | L_res_0=0.0276 L_res_1=0.0333 L_res_2=0.0369
[2026-06-20 11:18:35] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 246): 19516.3 MiB
[2026-06-20 11:29:19] INFO segtask_v1.trainer.validation:   Val: loss=0.1862, pooled_mean_dice=0.8930, per_class=['0.8930'], iou=0.8066, recall=0.9858, precision=0.8161, vol_sim=0.9059, mcc=0.8734, min_class_dice=0.8930, coverage=[88]/88 samples
[2026-06-20 11:29:25] INFO segtask_v1.trainer.trainer: Best model saved: outputs/ves_resnet3d_bnorm/best_model.pth
[2026-06-20 11:29:25] INFO segtask_v1.trainer.trainer: ★ New best: mean_dice=0.8930 at epoch 247
[2026-06-20 11:29:25] INFO segtask_v1.trainer.trainer: Epoch 247/400 | LR=9.50e-04 | loss=0.0317 | val_dice=0.8930 | best=0.8930 (ep247) | 44:15:42 | L_res_0=0.0282 L_res_1=0.0331 L_res_2=0.0365
[2026-06-20 11:29:25] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 247): 19515.7 MiB
[2026-06-20 11:40:09] INFO segtask_v1.trainer.validation:   Val: loss=0.1804, pooled_mean_dice=0.8928, per_class=['0.8928'], iou=0.8064, recall=0.9865, precision=0.8153, vol_sim=0.9050, mcc=0.8729, min_class_dice=0.8928, coverage=[88]/88 samples
[2026-06-20 11:40:09] INFO segtask_v1.trainer.trainer: Epoch 248/400 | LR=9.50e-04 | loss=0.0314 | val_dice=0.8928 | best=0.8930 (ep247) | 44:26:26 | L_res_0=0.0283 L_res_1=0.0327 L_res_2=0.0363
[2026-06-20 11:40:09] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 248): 19516.7 MiB
[2026-06-20 11:50:54] INFO segtask_v1.trainer.validation:   Val: loss=0.1836, pooled_mean_dice=0.8906, per_class=['0.8906'], iou=0.8028, recall=0.9859, precision=0.8121, vol_sim=0.9034, mcc=0.8712, min_class_dice=0.8906, coverage=[88]/88 samples
[2026-06-20 11:50:54] INFO segtask_v1.trainer.trainer: Epoch 249/400 | LR=9.49e-04 | loss=0.0315 | val_dice=0.8906 | best=0.8930 (ep247) | 44:37:10 | L_res_0=0.0279 L_res_1=0.0329 L_res_2=0.0365
[2026-06-20 11:50:54] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 249): 19516.1 MiB
[2026-06-20 12:01:37] INFO segtask_v1.trainer.validation:   Val: loss=0.1916, pooled_mean_dice=0.8888, per_class=['0.8888'], iou=0.7999, recall=0.9870, precision=0.8084, vol_sim=0.9005, mcc=0.8692, min_class_dice=0.8888, coverage=[88]/88 samples
[2026-06-20 12:01:37] INFO segtask_v1.trainer.trainer: Epoch 250/400 | LR=9.49e-04 | loss=0.0313 | val_dice=0.8888 | best=0.8930 (ep247) | 44:47:54 | L_res_0=0.0274 L_res_1=0.0328 L_res_2=0.0364
[2026-06-20 12:01:37] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 250): 19516.8 MiB
[2026-06-20 12:12:21] INFO segtask_v1.trainer.validation:   Val: loss=0.1935, pooled_mean_dice=0.8893, per_class=['0.8893'], iou=0.8006, recall=0.9890, precision=0.8078, vol_sim=0.8992, mcc=0.8695, min_class_dice=0.8893, coverage=[88]/88 samples
[2026-06-20 12:12:21] INFO segtask_v1.trainer.trainer: Epoch 251/400 | LR=9.48e-04 | loss=0.0454 | val_dice=0.8893 | best=0.8930 (ep247) | 44:58:37 | L_res_0=0.0451 L_res_1=0.0489 L_res_2=0.0527
[2026-06-20 12:12:21] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 251): 19516.7 MiB
[2026-06-20 12:23:07] INFO segtask_v1.trainer.validation:   Val: loss=0.1996, pooled_mean_dice=0.8823, per_class=['0.8823'], iou=0.7894, recall=0.9887, precision=0.7966, vol_sim=0.8924, mcc=0.8624, min_class_dice=0.8823, coverage=[88]/88 samples
[2026-06-20 12:23:07] INFO segtask_v1.trainer.trainer: Epoch 252/400 | LR=9.48e-04 | loss=0.0521 | val_dice=0.8823 | best=0.8930 (ep247) | 45:09:24 | L_res_0=0.0527 L_res_1=0.0562 L_res_2=0.0591
[2026-06-20 12:23:07] INFO segtask_v1.trainer.trainer:   GPU peak (epoch 252): 19515.9 MiB
