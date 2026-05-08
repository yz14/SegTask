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


测试环境为: **conda activate torch27_env**!!!!  


这是我写的2.5D/3D分割代码，训练入口在D:\codes\work-projects\SegTask\segtask_v1\train.py。这里有3个3D方案，z轴滑块（只在z轴滑动切块，x,y为全尺寸）；cubic滑块（在x,y,z轴滑动中心切块）；whole（直接输入整个图像）。一个2.5D方案，它和z轴滑块的单分辨率/感受野方案非常的相似，区别是：a 在train的时候，当数据增强结束后，将3D数据B,1,D,H,W变为B,D,H,W作为2D输入,D张切片代表D个通道；b 模型采用2D模型。计算损失也和现有框架一致，模型输出为B,num_fgxD,H,W然后拆分为num_fg个B,D,H,W单标签预测，各自计算单标签损失。这里有一份小数据集作为测试：F:\med_data\Totalsegmentator_dataset_v201\small_data\nii，F:\med_data\Totalsegmentator_dataset_v201\small_data\mask，
F:\med_data\Totalsegmentator_dataset_v201\small_data\bbox，
F:\med_data\Totalsegmentator_dataset_v201\small_data\region_weihgt。  


# TODO  
1. 目前除了3D的whole模式外，其它的都支持多分辨率/感受野输入，而且数据读取应该都是单分辨率处理，只有数据增强结束后，在输入模型的那一刻才制作多分辨率输入。如果是单分辨率输入，那么数据读取自然是单分辨率。如果是多分辨率输入，那么核心的(B,D,H,W)/(B,C,D,H,W)和辅助的1.5倍，2倍分辨率中，2倍分辨率包含了1.5倍和1倍的内容，所以仍然可以是单分辨率，只不过是按最大的分辨率来处理（而且不resize到targe size），只有数据增强结束后在输入模型的那一刻，才通过中心剪裁制作多分辨率输入，且resize到target size。此时，模型的stem对应的进行应对，同时seg head也对应的应对。除了这种多分辨率/感受野的方法外，还有什么模型上的优化可以让2.5D模型拥有3D的信息吗？  

2. 经常OOM
python -m segtask_v1.train --config configs/seg2_5d.yaml
[2026-05-08 15:16:32] INFO __main__: Config loaded from: configs/seg2_5d.yaml
[2026-05-08 15:16:32] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-05-08 15:16:32] INFO __main__: Device: cuda
[2026-05-08 15:16:32] INFO __main__: GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU (17.2 GB)
[2026-05-08 15:16:32] INFO segtask_v1.data.loader: Found 207 matched image-label pairs.
[2026-05-08 15:16:32] INFO segtask_v1.data.loader: Label values: [0, 1, 2], num_classes: 3, num_fg: 2
[2026-05-08 15:18:09] INFO segtask_v1.data.loader: Stratified split: 166 train, 41 val (strata sizes: {'1': 207, '2': 0})
[2026-05-08 15:18:09] INFO segtask_v1.data.loader: Matched 207 bbox files under F:\BaiduNetdiskDownload\bone_bbox.
[2026-05-08 15:18:09] INFO segtask_v1.data.loader: Matched 207 regionweight files under F:\BaiduNetdiskDownload\lung_weight.
[2026-05-08 15:18:09] INFO segtask_v1.data.loader: Using 2.5D patch mode + aux_keep_native_d=True (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00) — SINGLE max-FOV cube extraction (depth=24), trainer center-crops per view at native depth before forward.
[2026-05-08 15:19:02] INFO segtask_v1.data.dataset: BBox precomputed: 166/166 masks have foreground; mean (D,H,W)=(309.4, 298.9, 468.2), min=(190, 198, 368), max=(695, 412, 512)
[2026-05-08 15:19:02] INFO segtask_v1.data.dataset: Building dataset index for 166 volumes...
[2026-05-08 15:19:37] INFO segtask_v1.data.dataset: Index built: 166 volumes, 51366/51366 foreground slices
[2026-05-08 15:19:51] INFO segtask_v1.data.dataset: BBox precomputed: 41/41 masks have foreground; mean (D,H,W)=(330.3, 301.4, 470.7), min=(180, 240, 367), max=(621, 357, 512)
[2026-05-08 15:19:51] INFO segtask_v1.data.dataset: Building dataset index for 41 volumes...
[2026-05-08 15:20:00] INFO segtask_v1.data.dataset: Index built: 41 volumes, 13544/13544 foreground slices
[2026-05-08 15:20:00] INFO segtask_v1.data.loader: DataLoader: batch_size=4, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4
[2026-05-08 15:20:01] INFO segtask_v1.data.loader: Volume cache estimate: ~526.50 MiB per volume (image+label, int16 label); effective cap=2, num_workers=4 => up to ~4.11 GiB RAM (all workers).
[2026-05-08 15:20:01] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=5.56M, dec=5.09M, total=10.66M, channels=[32, 64, 128, 256, 256], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=24 (fg=2, res=1), stem=conv3(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=False, aux_seg=True(n_aux_heads=2, mode=linear)
[2026-05-08 15:20:01] INFO segtask_v1.trainer: Loss: dice_focal [2.5D, reduction=per_volume], num_slices=12, fg_classes=2
[2026-05-08 15:20:01] INFO segtask_v1.trainer: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.25], fusion=multi_stem_proj
[2026-05-08 15:20:05] INFO segtask_v1.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda).
[2026-05-08 15:20:05] INFO segtask_v1.trainer: Trainer aux_keep_native_d=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-05-08 15:20:05] INFO segtask_v1.trainer: ============================================================
[2026-05-08 15:20:05] INFO segtask_v1.trainer: Training: 100 epochs, device=cuda
[2026-05-08 15:20:05] INFO segtask_v1.trainer: Model params: 10.66M
[2026-05-08 15:20:05] INFO segtask_v1.trainer: Train batches: 332, Val batches: 41
[2026-05-08 15:20:05] INFO segtask_v1.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-05-08 15:20:05] INFO segtask_v1.trainer: Grad accum=1, Effective batch=4
[2026-05-08 15:20:05] INFO segtask_v1.trainer: Foreground classes: 2, Loss: dice_focal
[2026-05-08 15:20:05] INFO segtask_v1.trainer: ============================================================
NIfTI read failed (attempt 1/4) for F:\BaiduNetdiskDownload\lung_nii\CE025005-0007575842-T125.nii.gz: Exception thrown in SimpleITK ImageFileReader_Execute: D:\a\SimpleITK\SimpleITK\bld\ITK\Modules\IO\NIFTI\src\itkNiftiImageIO.cxx:597:
ITK ERROR: NiftiImageIO(00000199AF126C20): nifti_image_load failed for file: F:\BaiduNetdiskDownload\lung_nii\CE025005-0007575842-T125.nii.gz — retrying in 0.50s
NIfTI read failed (attempt 1/4) for F:\BaiduNetdiskDownload\lung_nii\CE021001-P00079153-7173.nii.gz: Exception thrown in SimpleITK ImageFileReader_Execute: D:\a\SimpleITK\SimpleITK\bld\ITK\Modules\IO\NIFTI\src\itkNiftiImageIO.cxx:597:
ITK ERROR: NiftiImageIO(00000199AF128A20): nifti_image_load failed for file: F:\BaiduNetdiskDownload\lung_nii\CE021001-P00079153-7173.nii.gz — retrying in 0.50s
NIfTI read failed (attempt 1/4) for F:\BaiduNetdiskDownload\lung_weight\CE021003-136401001430951-175463.nii.gz: Exception thrown in SimpleITK ImageFileReader_Execute: D:\a\SimpleITK\SimpleITK\bld\ITK\Modules\IO\NIFTI\src\itkNiftiImageIO.cxx:597:
ITK ERROR: NiftiImageIO(000002661D4DE7A0): nifti_image_load failed for file: F:\BaiduNetdiskDownload\lung_weight\CE021003-136401001430951-175463.nii.gz — retrying in 0.50s
Traceback (most recent call last):
  File "D:\miniconda\envs\torch27_env\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "D:\miniconda\envs\torch27_env\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "D:\codes\work-projects\SegTask\segtask_v1\train.py", line 120, in <module>
    main()
  File "D:\codes\work-projects\SegTask\segtask_v1\train.py", line 113, in main
    best_metrics = trainer.fit()
  File "D:\codes\work-projects\SegTask\segtask_v1\trainer.py", line 738, in fit
    train_metrics = self._train_epoch(epoch)
  File "D:\codes\work-projects\SegTask\segtask_v1\trainer.py", line 860, in _train_epoch
    for step, batch in enumerate(self.train_loader):
  File "D:\miniconda\envs\torch27_env\lib\site-packages\torch\utils\data\dataloader.py", line 733, in __next__
    data = self._next_data()
  File "D:\miniconda\envs\torch27_env\lib\site-packages\torch\utils\data\dataloader.py", line 1515, in _next_data
    return self._process_data(data, worker_id)
  File "D:\miniconda\envs\torch27_env\lib\site-packages\torch\utils\data\dataloader.py", line 1550, in _process_data
    data.reraise()
  File "D:\miniconda\envs\torch27_env\lib\site-packages\torch\_utils.py", line 750, in reraise
    raise exception
MemoryError: Caught MemoryError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py", line 85, in _sitk_read_with_retry
    return sitk.ReadImage(*read_args)
  File "D:\miniconda\envs\torch27_env\lib\site-packages\SimpleITK\extra.py", line 384, in ReadImage
    return reader.Execute()
  File "D:\miniconda\envs\torch27_env\lib\site-packages\SimpleITK\SimpleITK.py", line 8534, in Execute
    return _SimpleITK.ImageFileReader_Execute(self)
RuntimeError: Exception thrown in SimpleITK ImageFileReader_Execute: bad allocation

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\miniconda\envs\torch27_env\lib\site-packages\torch\utils\data\_utils\worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "D:\miniconda\envs\torch27_env\lib\site-packages\torch\utils\data\_utils\fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\miniconda\envs\torch27_env\lib\site-packages\torch\utils\data\_utils\fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py", line 753, in __getitem__
    img, lbl = self._load_image(vol_idx), self._load_label(vol_idx)
  File "D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py", line 698, in _load_image
    img = load_nifti(path)
  File "D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py", line 148, in load_nifti
    img = _sitk_read_with_retry(read_args, path)
  File "D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py", line 91, in _sitk_read_with_retry
    raise MemoryError(
MemoryError: NIfTI read aborted (host OOM) for F:\BaiduNetdiskDownload\lung_nii\CN010013-82136731-38159-304.nii.gz: Exception thrown in SimpleITK ImageFileReader_Execute: bad allocation

NIfTI read failed (attempt 1/4) for F:\BaiduNetdiskDownload\lung_nii\CE021001-P00104820-41906.nii.gz: Exception thrown in SimpleITK ImageFileReader_Execute: D:\a\SimpleITK\SimpleITK\bld\ITK\Modules\IO\NIFTI\src\itkNiftiImageIO.cxx:597:
ITK ERROR: NiftiImageIO(000002661D4DE3E0): nifti_image_load failed for file: F:\BaiduNetdiskDownload\lung_nii\CE021001-P00104820-41906.nii.gz — retrying in 0.50s