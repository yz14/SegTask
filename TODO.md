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

4 有些代码太大，逻辑绕口，让人读起来费劲，维护困难，例如config.py文件，如果某个参数会被自动重写或者更新，那么它就不应该暴露接口出来，例如save_best_metric。前全面检查代码其它地方是否有类似的问题。我需要的是让人读起来代码来不那么费劲，绕圈，甚至多个文件反复查看确认才能理解代码。

5 在服务器上运行报错
 python -m segtask_v1.train --config configs/segtest0.yaml
Config key 'aux_keep_native_d' is deprecated; use 'keep_native_view_depth' instead (auto-remapped for backward compatibility).
Config key 'context_fusion' is deprecated; use 'stem_fusion_mode' instead (auto-remapped for backward compatibility).
[2026-06-03 09:49:01] INFO __main__: Config loaded from: configs/segtest0.yaml
[2026-06-03 09:49:01] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-06-03 09:49:02] INFO __main__: Device: cuda
[2026-06-03 09:49:02] INFO __main__: GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
[2026-06-03 09:49:02] INFO segtask_v1.data.loader: Training source: npz packages under /data0/yzhen/data/totalseg/body_prep (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-06-03 09:49:02] INFO segtask_v1.data.loader: Discovered 404 npz package(s) under /data0/yzhen/data/totalseg/body_prep.
[2026-06-03 09:49:02] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-06-03 09:50:13] INFO segtask_v1.data.loader: Stratified split: 323 train, 81 val (strata sizes: {'1': 404})
[2026-06-03 09:50:13] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-06-03 09:50:13] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 323 npz packages...
[2026-06-03 09:50:51] INFO segtask_v1.data.dataset: NPZ index built: 323 volumes, 75782/76530 foreground slices
[2026-06-03 09:50:51] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 81 npz packages...
[2026-06-03 09:51:01] INFO segtask_v1.data.dataset: NPZ index built: 81 volumes, 20154/20446 foreground slices
[2026-06-03 09:51:01] INFO segtask_v1.data.loader: DataLoader: batch_size=8, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-06-03 09:51:01] INFO segtask_v1.data.loader: Volume cache estimate: ~138.19 MiB per volume (image fp32 + label int16, bbox-cropped); effective cap=12, num_workers=16 => up to ~25.91 GiB RAM (all workers, caches only; transient decode peaks add ~92.13 MiB/worker).
[2026-06-03 09:51:01] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=11.68M, dec=5.66M, total=18.14M, channels=[32, 64, 128, 256, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv)
[2026-06-03 09:51:05] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-06-03 09:51:05] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-06-03 09:51:05] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: ============================================================
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: Training: 100 epochs, device=cuda
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: Model params: 18.14M
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=69.2 + grad=69.2 + optim(AdamW,2x)=138.4 + ema=69.3 = 346.1 MiB (real peak reported per-epoch as 'GPU peak')
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=143.9 MiB, reserved=160.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: Train batches: 323, Val batches: 41
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: Grad accum=1, Effective batch=8
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_focal
[2026-06-03 09:51:05] INFO segtask_v1.trainer.trainer: ============================================================
Traceback (most recent call last):
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data0/yzhen/timm_test/segtask_v1/train.py", line 112, in <module>
    main()
  File "/data0/yzhen/timm_test/segtask_v1/train.py", line 105, in main
    best_metrics = trainer.fit()
  File "/data0/yzhen/timm_test/segtask_v1/trainer/trainer.py", line 266, in fit
    train_metrics = self._train_epoch(epoch)
  File "/data0/yzhen/timm_test/segtask_v1/trainer/trainer.py", line 408, in _train_epoch
    pred = self.model(image)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/data0/yzhen/timm_test/segtask_v1/models/unet.py", line 417, in forward
    dec_features = self.decoder(enc_features)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/data0/yzhen/timm_test/segtask_v1/models/unet.py", line 256, in forward
    x        = level(x, encoder_features[skip_idx])
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/data0/yzhen/timm_test/segtask_v1/models/unet.py", line 184, in forward
    x = self.upsample(x)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/data0/yzhen/timm_test/segtask_v1/models/blocks.py", line 754, in forward
    x = F.interpolate(
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/nn/functional.py", line 3950, in interpolate
    return torch._C._nn.upsample_bilinear2d(input, output_size, align_corners, scale_factors)
RuntimeError: "upsample_bilinear2d_out_frame" not implemented for 'BFloat16'
Exception in thread Thread-1 (_pin_memory_loop):
Traceback (most recent call last):
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
    self.run()
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/threading.py", line 953, in run
    self._target(*self._args, **self._kwargs)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/utils/data/_utils/pin_memory.py", line 49, in _pin_memory_loop
    do_one_step()
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/utils/data/_utils/pin_memory.py", line 26, in do_one_step
    r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/multiprocessing/queues.py", line 122, in get
    return _ForkingPickler.loads(res)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/torch/multiprocessing/reductions.py", line 305, in rebuild_storage_fd
    fd = df.detach()
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/multiprocessing/resource_sharer.py", line 86, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/multiprocessing/connection.py", line 502, in Client
    c = SocketClient(address)
  File "/data0/yzhen/py3/envs/py310/lib/python3.10/multiprocessing/connection.py", line 630, in SocketClient
    s.connect(address)
ConnectionRefusedError: [Errno 111] Connection refused