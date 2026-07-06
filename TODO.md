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


本地测试环境为: **D:\miniconda\envs\torch27_env\python.exe**。  


内容（详情见各自对应README.md）：  
segtask_v1是2.5D/3D分割项目（项目起源）。  
ssltask是自监督学习项目（基于segtask_v1改造），主要是对分割，生成，分类，检测的backbone（部分/全部）进行预训练。  
clstask是分类项目（基于segtask_v1改造）。  
dettask是检测项目（基于segtask_v1改造）。  
gentask是生成/超分项目（基于segtask_v1改造）。注意：  
生成/超分不一定有img-lable对；可能只有lable(高质图像)，img要对应任务的退化。可能有bbox, region_weight。  
任务可能是：  
厚层生成薄层(d,H,W -> D,H,W)，可能需要逼真的薄层模拟出厚层（例如部分容积效应等等）。  
面内超分(D,h,w -> D,H,W)。  


# TODO  
1 优化gentask里面的代码，目前主要用于超分，先针对CT超分进行修改、适配和优化（如需用到分割项目的代码，可先拷贝py文件后修改）：  
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


2 分割项目代码审查：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程。

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增

评审范围与规模
项目 5 大部分与对应文件已梳理清楚:

1. 数据读取 loader.py(~790行 发现-匹配-切分-建 loader)、dataset.py(~50KB 三个 Dataset + IO/patch/bbox/npz)、specs.py(patch_mode 策略)、make_data.py(npz 预打包)、mixed_sampler.py

2. 模型构建 models/:blocks.py(~57KB 积木)、stem.py(多视图融合)、resnet.py/convnext.py/mednext.py(编码骨干)、unet.py/unetpp.py/unet3p.py(解码)、topology.py+factory.py(装配)、adm_unet.py/edm2_unet.py

3. 数据增强/处理 augment.py(GPU 共享 warp)、dataset.py 内 preprocess/oversample/中心裁剪

4. 训练全流程(含 val) trainer.py(~55KB)、pipelines/(4 种 patch 管线策略)、validation.py、optim.py/amp.py/memory.py/breakdown.py/checkpoint.py/views.py、losses/(~45KB)

5. 推理全流程 predictor/:predictor.py、sliding.py(4 种滑窗)、inputs.py、forwards.py(TTA)、blending.py、io.py、adabn.py

横切:config.py(~108KB dataclass 配置系统),会在各部分评审时按需引用。

进展： 

数据读取
总体评价
架构成熟、职责分离清晰,是这套代码里质量很高的一层:

策略模式收敛 patch_mode:specs.py 把"选 Dataset 类 + 模式专属 kwargs"收敛到 3 个 DatasetSpec + build_data_spec,注释明确声明"data 侧唯一允许 patch_mode if/elif 的地方",可扩展性好。
npz 离线预烘(nnU-Net 风格):make_data.py 先 bbox 裁剪、预算前景索引、ZIP_STORED 不压缩以便多 worker 共享 OS page cache——设计思路专业。
fork RNG 陷阱已正确处理:_rng()/_sample_rng() 用 torch 逐 worker 种子重建 np.random.Generator,避开了"多 worker 复制 numpy 全局 RNG 导致重复采样"这一经典 bug;验证集用 (_VAL_SAMPLING_SEED, sample_idx) 派生确定性 RNG,保证 best/early-stop 不被采样噪声驱动——很到位。
单 max-FOV 提取原则:只提一份最大 FOV cube、增强只跑一次共享 warp、逐视图裁剪推迟到 trainer,消除多次 zoom 的高频损失,保证多视图几何一致。
DDP 细节:ValBatchShardSampler 按 batch 块无 padding/无重复切分;scaled_num_workers 防止每卡 worker/缓存 RAM 随卡数线性翻倍;混合训练与 DDP 的不兼容被显式拒绝。这些都是踩过坑才会有的设计。

正确性问题
[中] _build_index 泄漏 npz 文件句柄 @d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:694-702 与 @d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:926-934:



python
for path in self._npz_paths:
    f = _open_npz(path)          # 未 with / 未 f.close()
    fg = np.asarray(f["fg_slices"], ...)
    D  = int(f["image"].shape[0])
其余读取函数都用 with _open_npz(path) as f,唯独两处 _build_index 没有关闭。大数据集下会在 dataset 初始化期堆积 N 个未关闭的 NpzFile 句柄(依赖 GC 回收),在 Windows/worker 环境下可能触发 "too many open files" 或延迟释放。

[中] _build_index 为取 shape 而全量解码 image 同两处的 f["image"].shape / f["image"].shape[0]:NpzFile["image"] 是惰性的,一旦下标访问就会把整卷 image 从 zip 解出到内存,仅为读一个 shape。相当于 dataset 初始化时把每个样本的整卷 image 都解码一遍,启动开销随数据集规模线性增长。应改为解析 .npy header(np.lib.format.read_array_header_1_0)或把 shape 写进 meta(见优化建议)。这两点(泄漏 + 全解码)是同一处代码最值得修的。

[低] 返回张量可能与 LRU 缓存 numpy 数组别名 _getitem_max_fov 中当 resize_3d 尺寸恰好匹配为 no-op(@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:376-377)时直接返回原数组,随后 torch.from_numpy(img_s[None].astype(..., copy=False)) 与缓存数组共享内存。当前因 GPU 增强前会经 collate(stack 复制)而安全,但属易碎设计:whole 模式或恰好等尺寸时若未来有人在 augment 前做 in-place,会污染缓存。建议在返回前对命中缓存的路径显式 copy 或注释锁定该不变量。

[低] 随机划分与分层划分的取整不一致 train_val_split 用 int(n*val_ratio)(向下取整,@d:\codes\work-projects\SegTask\segtask_v1\data\loader.py:384),stratified_train_val_split 用 int(round(len*val_ratio))(@d:\codes\work-projects\SegTask\segtask_v1\data\loader.py:451)。同一 val_ratio 在两条路径下 val 比例略有差异,且 n=1 时随机划分会产出空 train。属边界瑕疵。

[低] 死分支 _extract_z_single 的 use_padded=False 分支(@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:791-795)在现有调用点(唯一调用处 line 739 传 use_padded=True)从不触达。

优化空间
把 shape/spacing 写进 npz meta:make_data 的 meta 已含 bbox、label_values 等,补上 image_shape 后,_build_index 只需读 meta(0-d object,极廉价)即可拿到 D/shape,彻底消除上面的全量解码与句柄泄漏两个问题。这是性价比最高的一处改进。
detect_label_values 全量串行扫描(主进程,@d:\codes\work-projects\SegTask\segtask_v1\data\loader.py:356-363):大数据集下是训练启动的串行瓶颈。既然 make_data 已逐样本落盘 label_values 到 meta,可让 loader 优先从任一 npz 的 meta 读取标签全集,退化时才全量扫描。
VolumeCache 三份独立缓存:image/label/rw 各持一个 cache_max_volumes LRU,实际内存是 cap × (img+lbl+rw)——loader 的估算已正确反映,但缓存键都是同一 path,可合并为单个按 path 存元组的缓存,减少一次 dict 查找与三份 OrderedDict 开销(优先级低)。

可借鉴 / 建议新增(业界高质量做法)
前景强度统计归一化(nnU-Net):当前用固定 intensity_min/max 窗 + minmax/zscore。CT 建议在 make_data 阶段采集前景体素的 0.5/99.5 百分位与 mean/std 作为数据集指纹,zscore 用前景统计。这对跨机器/协议的 CT 更稳。spacing 归一化你已按 nnU-Net 中位数指纹实现,强度指纹是自然的下一步。
划分可复现性增强:目前单一 split_seed。可考虑加入 K-fold(nnU-Net 5 折)接口,分层逻辑已具备,扩展成本低。


数据增强/处理 代码审查
审查文件：augment.py（GPU 共享 warp）、dataset.py（extract/oversample/edge-pad）、views.py + trainer/pipelines/* + trainer/trainer.py（center-crop、多视图拆分）。

整体数据流（关键前提）： __getitem__ 抽单张 max-FOV cube（含 oversample × max_scale 余量）→ collate → to(GPU) → GPUAugmentor 一次性增强 → views.center_crop 去 oversample 余量 → pipeline.prepare_batch 逐视图 center-crop + resize 拆分。见 @d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:714-721。

总体评价： 这一层设计思路是正确且高质量的——"单次共享 warp + 惰性拆视图"从根上保证了多分辨率几何一致性，这是比"逐视图各自增强"专业得多的做法。下面按你点名的 4 个子项给结论，含少量真问题与若干可优化点。

一、共享 warp 的几何一致性 — 正确，无实质问题
做对的地方：

flip/affine/elastic 每个子阶段对被选样本只算一次 grid/mask，然后 image(bilinear)/label(nearest)/wmap(可配) 用同一 grid、同一 padding_mode='border' 采样（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:159-171、@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:257-266）。采样坐标完全一致、仅插值模式不同 → 几何严格对齐，label 用 nearest 保 one-hot、不产生跨类渗漏。
所有视图从同一张已增强 cube 里 center-crop/resize 得到（split_views_native_3d/split_views_native_d），因此各 FOV 视图共享同一形变，几何一致性天然成立。
elastic 位移/单位网格的轴序处理正确：voxel_to_grid=[2/W,2/H,2/D] 配 grid[::-1] 与 align_corners=False，与 affine_grid(align_corners=False) 一致（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:250-277）。
grid_sample 的 grid 按 batch 广播到所有通道，label 的多前景通道与 image 单通道共用同一 grid → 正确。
已知限制（非 bug，文档已明示）： aspect_correct 只按 voxel-count 比例（W,H,D）校正，不含真实 spacing；spacing_normalization=False 时厚层 z 的 out-of-plane 旋转物理不准（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:114-123）。这点标注得很清楚，可接受。

二、多 FOV 安全性 — 机制健全，一处需注意
做对的地方：

patch3d 交叉校验每个视图 native 尺寸 ≤ max-FOV target，拦截浮点漂移（@d:\codes\work-projects\SegTask\segtask_v1\trainer\pipelines\patch3d.py:79-87）。
multi_res_scales[0]==1.0、全部 ≥1.0 断言；max_scale>1.0 强制 edge_pad（@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:672-678）——否则 stretch 会破坏跨 scale 物理 z-FOV 一致性。这是正确的硬约束。
elastic 的 alpha /= max_scale，使最大物理位移随 FOV 归一（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:57-62），考虑周到。
cubic 的 _safe_center_range 把中心夹到界内，避免 max-scale cube >50% 体素来自边界复制（@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:988-1008）。
[中] random_translate_range 与 oversample 余量未协调（多 FOV / 前景采样安全性）： 平移在 max-FOV cube 上、center-crop 之前施加，量级是归一化 [-1,1] 全轴（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:145-149）。若平移量超过 oversample 余量，_sample_z/_sample_center 精心保证的前景中心会被平移出最终裁剪窗，稀释 foreground_oversample_ratio 的前景过采样保证，且大平移会把 border-复制内容推进有效区。建议：把平移上限与 oversample 余量挂钩，或在文档里明确"translate 需 ≤ (oversample-1)/2 的归一化占比"。默认若为 [0,0] 则无影响。

三、oversample + 中心裁剪去边界 — 逻辑正确，两处不对称
做对的地方： 两类余量被正确区分——center_crop 只去 oversample 余量（target=round(p*max_scale)，needs_crop = oversample>1.0），max_scale 余量保留给大 FOV 视图（@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:716-718、@d:\codes\work-projects\SegTask\segtask_v1\trainer\views.py:20-34）。拆视图时 native-d/native-3d 还断言深度轴已等于 target，确保 oversample 余量确实被去掉（@d:\codes\work-projects\SegTask\segtask_v1\trainer\views.py:134-139）。设计闭环、自检到位。

[中] z_axis 的 oversample 是 z-only，面内旋转 border 伪影无余量吸收： SegDataset3D.extract_size=(round(pD*oversample), pH, pW)（@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:659-661），面内直接 resize 到 (pH,pW)，无过采样余量。而 SegDataset3DCubic 是三轴同乘 oversample（@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:901-902）。后果：2.5D 里最常见的面内旋转（绕 z）产生的 border 复制伪影落在 H/W 边缘，没有余量可裁除，会残留进每张 slab。这与"oversample 后中心裁去边界"的初衷在面内是没有实现的。取舍上可接受（nnU-Net patch 边缘同样有），但值得在设计上明确：z_axis 若开面内旋转，H/W 边缘伪影是固有的。

[低] grid_dropout 在 oversample 余量区打的洞会被裁掉，稀释有效 dropout 比例： 洞随机落在整张 max-FOV cube（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:301-320），center-crop 后靠近边缘的洞被丢弃，最终 patch 上的有效遮挡比例低于配置值且空间非均匀（中心洞更易保留）。gaussian noise/blur/gamma 是位置无关的，不受影响；仅 grid_dropout 有此偏差。低优先级，若要精确可在 crop 后再做 dropout。

四、GPU 增强数值/边界处理 — 稳健，无严重数值 bug
做对的地方：

_gaussian_blur_3d：可分离卷积 + F.pad(mode="replicate") 避免暗边；核长统一取 sigma 上界、逐样本核归一化（小 sigma 仅多近零尾部），grouped-conv 向量化正确（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:403-441）。数值健全。
_random_gamma：先逐通道 minmax 归一到 [0,1] 再 pow，rng.clamp(min=1e-7) 处理常数通道，负值（zscore）也安全（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:361-386）。仅空间轴 reduce、通道独立 → 多分辨率安全。
_simulate_lowres：按目标尺寸分组批量 interpolate，max(1,int(D*z)) 防零维（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:444-470）。正确。
clone/inplace 契约清晰，默认克隆不污染上游、且 collate 已复制，快路径安全（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:37-43）。
强度增强逐样本逐通道 reduce（reduce_dims=range(2,ndim)），对 2.5D 折通道前的 rank-5 也成立。
[低] intensity_clamp 语义较激进（已文档化）： 记录增强前逐样本逐通道 min/max，全部强度增强后夹回（@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:70-81）。这会把 brightness/noise 越界部分压回原范围，等于部分抵消增强强度。注释已声明"比 nnU-Net 更激进、可关"。属设计取舍，非 bug；建议默认关或明确推荐值。

[低] 无显式 NaN/inf 兜底： intensity_clamp 关闭时，brightness/gaussian_noise 可把值推出归一化范围；一般训练可接受，但缺少极端值保护。低优先级。

未发现的严重问题： flip 的 image[idx]=torch.flip(image[idx],[axis]) 高级索引 scatter-写回正确；affine 各向异性共轭 A⁻¹RA 只对旋转生效（各同性 scale 与对角阵可交换）数学正确；elastic 与 affine 的 align_corners/轴序一致。这些都对。

结论汇总（按优先级）
[中] translate 量级与 oversample 余量未协调（augment.py:145-149）：大平移会把前景中心移出最终裁剪窗，稀释前景过采样并引入 border 内渗。
[中] z_axis 面内无 oversample 余量（dataset.py:659-661）：面内旋转 border 伪影无法被中心裁去除，与 cubic 的三轴过采样不对称。
[低] grid_dropout 洞落在被裁余量里 → 有效遮挡比例被稀释且空间非均匀（augment.py:301-320）。
[低] intensity_clamp 偏激进、无 NaN/inf 兜底（augment.py:70-81）：属可配置取舍。
共享 warp 的几何一致性、多 FOV 尺寸校验/edge_pad 约束、oversample 双余量区分、GPU 核数值处理——均正确且实现质量高。


模型构建代码审查
总体评价
装配架构成熟、分层清晰，是高质量的一层：build_topology 单一真相源把 patch_mode × mode-flags → 全部派生量 收敛到一处决策树；block 仓库用统一 Stage(in,out,num_blocks) 接口 + _StatefulStageBuilder 让 4 种 backbone（resnet/convnext/mednext + adm/edm2）可插拔；上/下采样算子（BlurPool/PixelShuffle/CARAFE/DySample）与注意力（SE/ECA/CBAM/Coord + softmax/linear/window/grid 自注意力 + RoPE）覆盖面广且维度无关（spatial_dims 分派）。ICNR、DropPath fp32 采样、zero-init proj、checkpoint_if 的 use_reentrant=False+preserve_rng_state 都是踩过坑的细节。

下面按你点名的 5 个子项给结论，含 2 处真 bug。

一、topology 单一真相源 — 设计正确，一处未彻底
build_topology 把三处历史重复推导收敛为唯一入口（@d:\codes\work-projects\SegTask\segtask_v1\models\topology.py:74-160），决策树覆盖 2.5D lift/native_d/普通 与 3D 三分支，num_fg_classes property 反算自洽。正确。

[低] build_model 未完全走单一真相源：build_model 大部分读 topo.*，但 encoder 的 in_channels 直接读 mc.in_channels（@d:\codes\work-projects\SegTask\segtask_v1\models\factory.py:439），而非 topo.in_channels。二者仅在 Config.sync 已把 topology 写回 mc.in_channels 时才一致——这依赖调用时序，弱化了"唯一入口"的保证。建议统一用 topo.in_channels（out_classes 已经这么做了）。

二、多视图融合 Plan A / Plan C — 发现一处会崩溃的 bug
[高] Plan C（hierarchical）aux 分割头分辨率不匹配，训练首个 forward 即 RuntimeError。

构造期：hierarchical 分支让 aux 头 k 读低分辨率特征 dec_features[n_dec-1-k]（@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py:462-467）。
前向期：对每个 aux 输出断言 ao.shape[2:] == target_size（=输入尺寸），否则 raise RuntimeError，且不做上采样（@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py:503-511）。
由于 feat_idx = n_dec-1-k（k≥1）恒低于最高分辨率 dec[-1]，其空间尺寸 = 输入/2^k ≠ 输入 → 必然触发 RuntimeError。
这与两处 docstring 明确承诺的"aux 上采到 main 尺寸"（@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py:363 与 :415-416）直接矛盾。Plan A（shared_stem/multi_stem_proj）因 aux 都读 dec[-1]（=input 尺寸）不受影响；只要用户同时开 stem_fusion_mode='hierarchical' + aux_seg_supervision=True 就崩，而 config 层对二者无互斥校验。修法：aux 输出在尺寸检查前 F.interpolate 到 target_size（兑现 docstring），或仅对 Plan A 保留等尺寸断言。

其余 Plan A/C 机制正确：MultiStemProj 逐 view stem→cat→1×1、HierarchicalStems 的 aux stem stride=s0·2^k 对齐 encoder 第 k 级、Encoder.aux_fuse 逐级 cat→1×1、以及 forward 内对 aux 空间不匹配的显式 RuntimeError 护栏（@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py:150-160）都对。

三、编解码装配 — 发现一处 patchN stem 失效
[中] patch2/patch4 stem 与 UNet 解码拓扑不配套，主头 forward 会 RuntimeError。

stem_mode=patchN 使 stem 各向同性降 N 倍（@d:\codes\work-projects\SegTask\segtask_v1\models\stem.py:107-114），config/stem 注释均称"UNet 末尾上采补回 / 主输出加上采样"（config.py:315、stem.py:3）。
但 Decoder 只镜像 encoder 的 n-1 次 stage 间下采样（@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py:271-288），没有任何一级补偿 stem 的 N 倍。总下采样 = N·2^(n-1)，解码只还原 2^(n-1)，主输出停在 输入/N。
UNet3D.forward 又断言 main_out.shape[2:] == target_size 否则 raise（@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py:494-499），且注释坦承"forward 不做上采样补偿"（:393-395）——与 config 注释自相矛盾。
结果：patchN 目前对 unet/unetpp/unet3p 三种 decoder 都不可用。修法二选一：在主头/DS 头输出后按 stem_stride 补一次上采样；或从 STEM_MODES 暂时下线 patchN 并更正注释。unetpp/unet3p 有 F.interpolate 回退保尺寸对齐（unetpp.py:97-108、unet3p.py:81-91），装配本身正确。

编解码其余部分正确：out_channels 全程 low-res→high-res 一致；DS 头用 ConvSegHead（3×3+1×1）、主头 1×1（nnU-Net 风）；各向异性 stride 的兼容性校验（拒绝 convnext LN-first / 非 unet decoder / 不兼容的 up/down mode）在 factory.py:413-435 到位。

四、注意力算子 — 正确，两处工程建议
softmax 走 SDPA（可选 RoPE）、linear 是 Shen 2021 的 KᵀV O(N)、window/grid 各自 partition + padding mask + SDPA，RoPE 对 linear/grid 显式拒绝（@d:\codes\work-projects\SegTask\segtask_v1\models\blocks.py:839-848）——数学与边界都对；proj/FFN zero-init 保证初始恒等残差。
[中] self-attention 与 MultiRF 仅对 backbone='resnet' 生效：factory.py:371-407 只在 resnet 分支解析 selfattn/multirf 逐 stage 配置，convnext/mednext 分支静默忽略。用户在 convnext/mednext 下开这些开关不会报错也不会生效。建议在非 resnet backbone 且开关开启时 logger.warning。

[低] Coord/AttentionGate 默认 BatchNorm：CoordAttention3D 用 _BN[d]（blocks.py:309）、AttentionGate3D 默认 norm_type='batch'（blocks.py:911）。3D 小 batch 下 BN 统计噪声大；建议默认 instance/group 或文档提示。

五、上/下采样算子 — 稳健，无实质问题
Downsample/Upsample 各向异性 stride 与核结构约束一致（blurpool/pixelshuffle/carafe/dysample 因核结构限各向同性 2，已显式报错，blocks.py:1113-1130、1328-1353）。
CARAFE 的 reassembly、DySample 的 offset 归一化+分组 grid_sample、ICNR 与 PixelShuffle 的 (c r1 r2 r3) 通道分组匹配（repeat_interleave(rd) → NN 初始化）均正确。
trilinear/nearest 在 bf16/fp16 下先转 fp32 再插值，兼顾后端支持与确定性。
[低] DropPath 也施加于 decoder：enc/dec 共用同一 _make_*_stage_builder，各自按 np.linspace(0, drop_path_rate, ...) 生成随机深度（factory.py:38-41）。惯例上 stochastic depth 只用于 encoder；decoder 也开是非主流取舍，值得确认是否有意。

可借鉴 / 建议新增（业界高质量做法）
MedNeXt UpKern / DilatedReparam 已实现（mednext.py），质量高；但 upkern_remap 用 align_corners=True（注释已说明与官方 False 不同），小核影响小，建议对齐官方默认以免复现差异。
多视图融合可考虑加轻量 cross-view attention：当前 Plan A 是 cat→1×1 早融合，可选加一层 view 维注意力（复用已有 SelfAttentionBlock）作为 Plan B 消融。
get_norm 的 group 静默折半（blocks.py:137-138）：MedNeXt 分支已改成显式报错，建议全局统一为显式报错或至少一次性 warning，避免 group 数被悄悄退化到 1 组。
结论汇总（按优先级）
[高] Plan C hierarchical + aux_seg_supervision → aux 头读低分辨率特征却断言等于输入尺寸、未上采样 → 训练即 RuntimeError（unet.py:462-511，与 docstring 矛盾）。
[中] patch2/patch4 stem 无解码端补偿 → 主头 forward RuntimeError；config/stem 注释宣称的"末尾上采补回"未实现（stem.py:107-114 vs unet.py:494-499）。
[中] selfattn/multirf 仅 resnet 生效，convnext/mednext 静默忽略，缺 warning（factory.py:371-407）。
[低] build_model encoder 用 mc.in_channels 而非 topo.in_channels，弱化单一真相源；Coord/AttentionGate 默认 BN 小 batch 不稳；DropPath 亦作用于 decoder；get_norm group 静默折半。
其余（topology 决策树、Plan A、注意力数学与边界、上/下采样各向异性约束、ICNR/DropPath/checkpoint 细节）均正确且实现质量高。


训练全流程(含 val) 代码审查
审查文件:trainer.py、trainer/pipelines/*(base/vanilla3d/patch3d/slab25d/lift25d/factory)、optim.py、amp.py、validation.py、views.py、breakdown.py、checkpoint.py、losses.py,并交叉核对 utils.py(ModelEMA / pooled 指标)与 config.py(选模派生量)。

总体评价
这是全项目成熟度最高的一层之一。Trainer 被彻底收窄为"编排器",所有"如何把 batch 拆成 (model_input, supervision)、如何折损失"的模式分支全部下沉到 ViewPipeline 策略对象(唯一 if/elif 在 factory.py,且派生量与 models.factory 共用 ModelTopology 单一真相源)。fp32 损失、GradScaler 跳步保护、尾组有效 accum、EMA CPU offload、RNG 位精确 resume、ZeRO consolidate 早于 rank 早退、DDP no_sync 把 forward 一并纳入、验证 EMA try/finally 换回、pooled 指标 all-reduce 的严格可加性——都是踩过坑才有的实现。下面按你点名的 6 个子项给结论。

一、Trainer 控制流 — 正确,一处注释过强
fit → _train_epoch / _validate 主干清晰:优化步以 ceil(len/accum) 对齐 scheduler horizon;尾组不满 accum 时 _effective_accum 用真实尾长作分母(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:579-587),is_step_boundary 用 (step+1)==total_steps 兜住尾组触发一步——与 scheduler 步数自洽。非有限 loss:fp16 交 GradScaler 跳步,bf16/fp32 走 skip_optim_step 丢弃本组梯度但仍推进 scheduler/EMA(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:786-815),避免 NaN 永久污染权重/EMA。DDP no_sync 正确包住 forward+backward(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:727-741),_save_checkpoint 先 consolidate_state_dict(to=0) 再 if not is_main: return(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:974-980)——顺序对,否则集合通信挂死。

[中] "math-equivalent to single-GPU under grad-accum" 的注释对"区域比值型"损失过强(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:222-224、738-741)。 该等价只对逐样本可分损失(BCE/Focal/per-sample Dice 的样本均值)成立:此时 loss/accum 求和 + DDP 梯度均值 = 全局样本均值。但 batch_dice=True 的 Dice / Tversky / GDL 是在 batch 维汇总分子分母的比值,其"批"在实现里 = 单卡单 micro-batch;grad-accum 越大或卡越多,batch_dice 的有效统计窗越小、越抖,与"single-GPU 全 batch pooled dice"并不等价。这不是 bug(与 nnU-Net 不用 accum 的取舍一致),但注释会误导用 batch_dice+大 accum 的用户。建议注释区分"样本可分损失严格等价、比值型损失仅近似"。

二、pipelines 策略 — 设计正确,实现干净
7 个 pipeline 各司其职:Vanilla3D(pass-through/eager MR)、Patch3DNativeMultiRes(懒 MR,max-FOV cube 逐视图裁+resize 堆通道)、Slab2_5D{,-Aux,-NativeD}、Lift2_5D{,-Aux}。SupervisionPack 只填各自需要的字段(@d:\codes\work-projects\SegTask\segtask_v1\trainer\pipelines\base.py:34-49),compute_loss 内 arity 校验齐全(preds/weights/losses/labels 四元对齐,见 @d:\codes\work-projects\SegTask\segtask_v1\trainer\pipelines\slab25d.py:314-320)。split_views_native_3d/_d 都断言"增强 oversample 余量已被中心裁去掉"(@d:\codes\work-projects\SegTask\segtask_v1\trainer\views.py:62-67、134-139),自检闭环。aux 权重缺省几何衰减 0.5^(k+1)(@d:\codes\work-projects\SegTask\segtask_v1\trainer\pipelines\slab25d.py:36-37),契合"FOV 越宽对齐越差权重越小"。topo 辅助头损失由 factory 统一注入、与具体 pipeline 解耦(@d:\codes\work-projects\SegTask\segtask_v1\trainer\pipelines\factory.py:70-77)。此项未见正确性问题。

三、AMP / EMA / grad-clip / scheduler
AMP: amp_dtype='auto' 按设备解析 bf16/fp16;scaler 仅 fp16 启用(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:170-174);损失强制 fp32 且 logits clamp ±50 防 BCE 出 NaN(@d:\codes\work-projects\SegTask\segtask_v1\trainer\amp.py:73-94)。正确。[低] 该 clamp 在 bf16/fp32 下也生效,对 |logit|>50 的梯度有极小改变,可忽略。

EMA: foreach 向量化 + dtype 分组;CPU offload 用 pinned staging 异步 D2H+单次流同步,数学等价(@d:/codes/work-projects/SegTask/segtask_v1/utils.py:103-127);整型 buffer 直接跟随;apply/restore 用 _swapped 防重入;pretrain 后重对齐 shadow 防随机初始泄露(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:1117-1123)。正确。

grad-clip: 仅边界步 unscale_ 后 clip_grad_norm_,复用范数喂健康监测(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:770-785)。fp16 下即便 clip 把 inf 梯度缩成 NaN,scaler.step 仍会跳步,不污染权重——安全。

scheduler: WarmupScheduler 线性 warmup→base;one_cycle 关外层 warmup 用 pct_start 映射,避免双重叠加(@d:\codes\work-projects\SegTask\segtask_v1\trainer\optim.py:139-148、trainer.py:153-158);cosine/poly/step horizon 按 post_warmup_steps 对齐,退火恰好到尾。resume 校验 warmup 配置漂移。

[低/设计] 跳步仍推进 scheduler/EMA。 bf16/fp32 guard-skip 明确推进(已文档化、内部一致);但 fp16 下 scaler.step 内部跳步对 Trainer 不可见(未比较 get_scale() 前后),scheduler.step()(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:831)照常推进,启动期标定的少数跳步会让 LR 提前漂移几步。影响很小,可选:检测 scale 下降来判定跳步。
[低/设计] plateau 的 patience 单位是"验证 epoch"而非 epoch。 step_epoch 仅在 metric is not None 时步进(@d:\codes\work-projects\SegTask\segtask_v1\trainer\optim.py:189-194),而 metric 只在 val epoch 有值,故有效耐心 ≈ plateau_patience × val_every。文档提示即可。
四、DS / MultiRes / aux 损失组合 — 正确
DS×MR 组合口径一致: DeepSupervisionLoss 对每个低分辨率 pred 把整数 label_raw(B,C_res,*) 与 wmap 一起 nearest 下采到 pred 尺寸,再交 MultiResolutionLoss 按通道 [:, r*num_fg:(r+1)*num_fg] / label[:, r] 拆分(@d:\codes\work-projects\SegTask\segtask_v1\losses\losses.py:324-340、762-777)。C_res 通道在下采样中保留,layout 严格一致。
诊断无热路径同步: MR 每次 forward 只 detach 追加历史,pop_per_res_diag 每步一次 .item();DS 多次调用时对尺度取均(@d:\codes\work-projects\SegTask\segtask_v1\losses\losses.py:745-753、breakdown.py:24-39)。设计好。
DS 权重归一化(@d:\codes\work-projects\SegTask\segtask_v1\losses\losses.py:301-306)使各尺度权重和为 1,主损失量纲稳定;数量不匹配显式 ValueError(:318-321)。
SliceChannelLoss 逐类单通道喂 base_loss、per_slice/per_volume 两种 reduction、_aggregate_per_class 重读 base_loss.class_weights 做归一化加权均(@d:\codes\work-projects\SegTask\segtask_v1\losses\losses.py:974-1015)——避免 cw 在包装层被"折叠成无操作"。正确。
aux 三条路径(folded 共享 / native_d 逐视图 / lift MR)标签取用与损失函数选择均与 image 折叠方式对应,无错位。
未发现该子项的正确性问题。 唯一提示:DS 权重默认 4 项,若模型实际 DS 头数不同则运行时 ValueError——建议 config 层预校验 len(deep_supervision_weights) 与解码深度一致(属模型装配交叉项,可选)。

五、validation 的 pooled Dice — 正确且 nnU-Net 一致
MetricAccumulator 逐 batch/卷累加 inter/pred_sum/target_sum/voxels/cov(+可选 sd 分子分母),compute() 末闭式导出 dice/iou/recall/precision/vol_sim/mcc/surface_dice/balanced(@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:114-268)。pooled dice = (2ΣTP+ε)/(2ΣTP+ΣFP+ΣFN+ε),是 nnU-Net 训练期"global/pseudo dice"口径。做对的关键:

多卡严格等价: 各 rank 处理不相交样本,可加混淆量 all-reduce(SUM) 后与单卡在全集累加逐位相等;空 rank 按正确 shape/dtype 零初始化避免 collective 死锁,voxels 保 float64 防大计数溢出(@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:140-188)。
coverage 掩码: 全 val 无 GT 的类(cov==0)从 mean/min 剔除,全空退回全类(@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:222-232)——nnU-Net ignore_empty 思想。
high 模式 bbox 裁剪后回传整卷 voxels 使 MCC 的 TN 口径不变、sd 按 tol+1 外扩边距保严格等价(@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:442-458)。
选模一致性(核对确认非 bug):save_best_metric/mode 是 save_best_criterion 的派生只读 property(@d:\codes\work-projects\SegTask\segtask_v1\config.py:743-754),而 MetricAccumulator 也用 save_best_criterion 决定是否算 surface/balanced——被追踪指标必然被产出,不会出现"设了指标却从不置 best"的坑。
[中] medium 与 high 的二值化阈值不一致。 medium 走 dice_batch_stats 内固定 sigmoid>0.5(@d:/codes/work-projects/SegTask/segtask_v1/utils.py:253-254),high 用 predictor.threshold(可为逐类调优阈值,@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:436-440)。若部署阈值≠0.5,medium 选出的 best 未必对齐部署行为。属口径差异(medium 本就文档化为"快而乐观"),但值得在文档明确:medium 选模恒在 0.5 阈值下进行。
[低,已文档化] pooled≠mean-per-case: pooled dice 会被大目标主导、掩盖个别病例失败;且 medium 是"每卷一 patch"上 pooled,偏乐观。用于选模可接受(与 nnU-Net 一致),但最终报告建议另算 per-case。
六、checkpoint / resume — 正确
在线权重存 model_state_dict,best 时把 EMA 提为 primary、在线放 model_online_state_dict(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:960-969);resume 优先读 model_online_state_dict 续训、RNG 四源(torch cpu/cuda+numpy+python)快照恢复、rank>0 resume 后 _reseed_rank_rng 防各卡 RNG 退化成 rank0(@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:1039-1057);keep-last-k 剪枝不动 best。pretrain 仅载权重、UpKern 重映射、不动 optim/RNG/epoch。正确。

结论汇总(按优先级)
[中] grad-accum/DDP 的"与单卡严格等价"注释对 batch_dice/Dice/Tversky/GDL 等区域比值型损失过强;其有效统计窗是单卡单 micro-batch,随 accum/卡数缩小(trainer.py:222-224、738-741)。非 bug,注释需修正。
[中] medium(0.5)与 high(predictor.threshold)验证阈值不一致,medium 选模不反映调优阈值(validation.py vs utils.py:253)。
[低/设计] fp16 scaler 跳步对 Trainer 不可见,scheduler 仍推进,启动期 LR 微漂;plateau patience 实为"验证 epoch"单位(=plateau_patience×val_every)。
[低] compute_loss_fp32 在 bf16/fp32 下也 clamp logits ±50(可忽略);DS 权重数与解码深度仅运行时 ValueError,建议 config 预校验。
正确且高质量: pipeline 策略收敛、DS×MR×aux 损失组合口径一致、fp32 损失与跳步保护、EMA offload/换回、pooled 指标多卡严格可加与 coverage 掩码、选模单一真相源、RNG 位精确 resume、ZeRO consolidate 顺序——均无实质问题。


推理全流程代码审查
审查范围
predictor.py(调度/单卷编排)、sliding.py(4 种滑窗 + blend 累加)、blending.py(位置/权重/prob→label)、forwards.py(3 种 forward + TTA)、inputs.py(6 个窗口 builder)、adabn.py(测试时 BN 自适应)、io.py(ckpt/精度/run_inference),并交叉核对 config.PredictConfig。

总体评价
这一层与训练层同为高成熟度：Predictor 只做编排,几何/权重下沉到 blending,窗口构造下沉到 inputs(纯函数 + 类侧 shim),forward/TTA 下沉到 forwards,mode 派生量统一来自 ModelTopology 单一真相源(消除了与 Config.sync/build_model 的三处重复)。累加器 dtype/落点(acc_dtype/accumulate_on_cpu)、整卷 vol_dtype=fp16 常驻、fp32 sigmoid、NaN→背景兜底 + 诊断、ckpt 的 compile/EMA/best-primary 兼容、< 半参数加载即硬错——都是踩过坑的实现。下面按你点名的 5 个子项给结论,含 2 处真问题。

一、滑窗与高斯融合 — 机制正确,发现一处 blend_mode 失效
做对的地方:

compute_1d_positions 尾窗反推 (length-patch, length) 保证全覆盖;length<=patch 单窗;stride=max(1,·) 防死循环(@d:\codes\work-projects\SegTask\segtask_v1\predictor\blending.py:34-43)。正确。
z 轴累加器 acc_weight 形状 (1,D,1,1) 只沿 z,H/W 每 z 恒一窗全覆盖 → 广播正确;cubic 用可分离 3D 外积权重 + 按 [:ad,:ah,:aw] trim 尾窗(@d:\codes\work-projects\SegTask\segtask_v1\predictor\sliding.py:341-347)。自归一化加权平均 acc_pred/acc_weight 数学正确。
edge_pad+ad<pD 时对称去 pad(pad_before=(pD-ad)//2)取中心 ad 切片、build_1d_weight(ad) 对称权重(@d:\codes\work-projects\SegTask\segtask_v1\predictor\sliding.py:167-184);且该分支仅在小卷单窗触达,尾窗恒 ad==pD。闭环。
fp16 累加器 eps 按 dtype 抬到 6.1e-5 防下溢清零(@d:\codes\work-projects\SegTask\segtask_v1\predictor\sliding.py:393)。稳健。
[中] z 轴滑窗恒用 gaussian,忽略 blend_mode="average"。 sliding_window_z 的两处权重 build_1d_weight(pD) 与 build_1d_weight(ad)(@d:\codes\work-projects\SegTask\segtask_v1\predictor\sliding.py:90、183)均未传 mode,而 build_1d_weight 默认 mode="gaussian"(@d:\codes\work-projects\SegTask\segtask_v1\predictor\blending.py:49)。因此无论 predict.blend_mode 设成 "average" 与否,z 轴/2.5D 路径始终高斯融合;偏偏日志还打印 blend=%s(@d:\codes\work-projects\SegTask\segtask_v1\predictor\sliding.py:78)会显示 average,误导。cubic 路径经 build_3d_weight(...,p.blend_mode) 正确响应(@d:\codes\work-projects\SegTask\segtask_v1\predictor\sliding.py:295)。属真实不一致:要么把 p.blend_mode 传进两处 build_1d_weight,要么在 z 路径日志显式标注"z 轴恒 gaussian"。因 gaussian 是默认且推荐值,实际影响面小,但语义与日志确实错位。

[低] 高斯 σ=n/4 比 nnU-Net 的 n/8 平缓(@d:\codes\work-projects\SegTask\segtask_v1\predictor\blending.py:57):窗边权重 ≈exp(-2)=0.135 偏高、中心加权更弱、更接近均匀,缝合处抑制略逊于 nnU-Net 默认。可选调优,非 bug。

二、TTA — 正确,批量化与串行严格等价
规格表 _FLIP_SPECS_3D(7 组 D/H/W)与 _FLIP_SPECS_2_5D(仅 H/W,不翻通道轴 D)分别记录 x 与 prob 的翻转轴(布局不同),映射正确(@d:\codes\work-projects\SegTask\segtask_v1\predictor\forwards.py:47-58)。2.5D 不翻 D(输入通道轴,翻转会打乱物理切片顺序)——判断正确。
_flip_tta_batched 按 tta_batch_size 沿 batch cat 成大 batch 一次前向、逐变体反 flip 后累加、除以 1+len(specs)(@d:\codes\work-projects\SegTask\segtask_v1\predictor\forwards.py:74-97);eval 下 BN 用 running stats、变体间无 batch 耦合,与逐变体串行逐像素等价,累加顺序保持"原图→变体序"浮点同序。正确。
所有 TTA 调用都发生在各 forward 的 autocast 上下文内(@d:\codes\work-projects\SegTask\segtask_v1\predictor\forwards.py:257-268、285-294、305-318),精度一致;post_fn 统一 sigmoid(pred.float()) fp32。lift 模式复用 3D ensemble(D 为真空间轴,翻转合法)。未见正确性问题。
三、bbox — 逻辑正确,顺序闭环
bbox 形状对齐检查在裁剪/重采样之前对原始 raw_vol 做(@d:\codes\work-projects\SegTask\segtask_v1\predictor\predictor.py:352-356);空 bbox 警告并回退全卷(bbox=None,不裁不拼)。
顺序正确:bbox 裁剪 → pre_resample_shape=裁剪后形状 → spacing 重采样 → 推理 → 概率回采到裁剪原生分辨率 → 贴回 (num_fg,D_orig,H_orig,W_orig) 画布、bbox 外留 0(@d:\codes\work-projects\SegTask\segtask_v1\predictor\predictor.py:371-431)。read_nifti_spacing 取整图 spacing(与是否裁剪无关)喂重采样,正确。_log_inroi_prob_stats 在贴回前算,避免 ROI 外 0 偏移统计。未见问题。
四、AdaBN — 发现一处 global 与 per_volume 的语义不一致
做对的地方:collect_bn_modules 只纳入 track_running_stats=True 且 buffer 非空的 _BatchNorm(instance/group 自动排除);bn_estimation_mode 临时 train()+momentum=None(累积平均)配合 reset_running_stats,退出恢复 training/momentum 而保留新 running stats,且全程 no_grad(@d:\codes\work-projects\SegTask\segtask_v1\predictor\adabn.py:33-97)。per_volume 双趟(先估计再预测)、_diag_first_batch_logged 护孔管理均正确(@d:\codes\work-projects\SegTask\segtask_v1\predictor\predictor.py:394-412)。

[中] global AdaBN 未设 _adabn_estimating → BN 估计期 TTA 变体被拼成大 batch。 per_volume 路径明确置 self._adabn_estimating=True,使 _tta_chunk_size 强制返回 1、串行前向,理由是"BN 处于 train+累积平均,把多个 flip 变体拼成大 batch 会改变 BN 见到的 batch 统计构成与 running stats 累积"(@d:\codes\work-projects\SegTask\segtask_v1\predictor\forwards.py:61-71、@d:\codes\work-projects\SegTask\segtask_v1\predictor\predictor.py:401-409)。但 global 模式的 _warmup 里 predict_volume 全程都没有置该标志(@d:\codes\work-projects\SegTask\segtask_v1\predictor\io.py:197-208),于是 global BN 重估时 TTA 仍按 tta_batch_size 批量化,momentum=None 的逐 forward 累积平均对 batch 构成敏感 → global 的 BN 统计会依赖 tta_batch_size,与 per_volume 刻意规避的语义直接矛盾。修法:global 预热前后同样置/复位 _adabn_estimating(或预热时临时关 tta_flip)。

[低] 两种模式的 BN 估计都在跑 TTA(把翻转输入喂进 BN 统计估计)。 conv 非 flip-等变,翻转输入会产生不同激活,相当于用增广样本估 BN。虽不致命,但纯粹的目标域 BN 统计通常只用原图前向;可考虑估计期一律关 TTA(顺带解决上一条)。

五、精度 / ckpt 加载 — 正确
_select_state_dict 的 auto 优先 EMA、ema 缺失回退 online 并警告、online 取 model_online_state_dict(@d:\codes\work-projects\SegTask\segtask_v1\predictor\io.py:42-66);与训练侧"best 把 EMA 提为 primary、online 另存"契合——best ckpt 无独立 ema_state_dict 时 auto 落到 primary(即 EMA),结果正确。_unwrap_ema_state 拆 {shadow,decay}、_strip_compile_prefix 在选权重后剥 _orig_mod.,次序对。
load_state_dict(strict=False) 后 加载 <半数参数即硬错,拒绝随机权重静默推理(@d:\codes\work-projects\SegTask\segtask_v1\predictor\io.py:131-144);weights_only=False 有注释说明(ckpt 含 Config/RNG,PyTorch 2.6+ 需显式)。MedNeXt reparam_deploy 在 .to(device) 前 reparameterize(@d:\codes\work-projects\SegTask\segtask_v1\predictor\io.py:146-149),次序对。
精度:auto 跟随 train.amp_dtype,fp16 仅 CUDA 下 model.half() 并醒目警告 LayerNorm 溢出;Predictor 侧检测到 fp16/bf16 权重时关 autocast 改为逐前向 cast 输入(@d:\codes\work-projects\SegTask\segtask_v1\predictor\predictor.py:243-249、@d:\codes\work-projects\SegTask\segtask_v1\predictor\io.py:154-168)。逻辑自洽。
[低] CPU + precision=fp16 静默降级。 io.py:155 仅在 device.type=="cuda" 时 half(),CPU 上请求 fp16 会静默按 fp32 跑(use_amp 亦为 False)。合理但无提示,可补一行 info。

[低] whole/cubic 的 CPU-blend 路径(forward_batch_numpy)不发 forward 诊断,仅 GPU 路径发(@d:\codes\work-projects\SegTask\segtask_v1\predictor\forwards.py:275-281)。已文档化,属复刻旧行为的取舍。

结论汇总(按优先级)
[中] z 轴滑窗恒用 gaussian,忽略 blend_mode="average"(build_1d_weight 两处未传 mode),日志却打印 blend_mode,与 cubic 不一致(sliding.py:90、183 vs blending.py:49)。
[中] global AdaBN 未置 _adabn_estimating,BN 重估期 TTA flip 变体被批量化拼 batch,使 BN 统计依赖 tta_batch_size,与 per_volume 明确规避的语义矛盾(io.py:197-208 vs forwards.py:61-71)。
[低] AdaBN 估计期(两模式)仍跑 TTA,把翻转输入喂入 BN 统计估计——建议估计期一律关 TTA。
[低] 高斯 σ=n/4 较 nnU-Net(n/8)平缓;CPU+fp16 静默降级;whole/cubic-CPU 路径不发 diag(已文档化)。
正确且高质量:滑窗全覆盖与自归一化加权、cubic 可分离权重与尾窗 trim、TTA 批量化↔串行严格等价与轴映射、bbox 裁剪/重采样/贴回顺序闭环、AdaBN 的 BN 收集与累积平均/状态恢复、ckpt 的 EMA/compile/best-primary 兼容与半参数硬错、精度分派——均无实质问题。

跨模块代码审查  
一、跨模块一致性
已贯通的高质量横切模式(值得保留)
单一真相源 ModelTopology 真正落地:build_topology(cfg) 一处把 patch_mode × 5 flag 派生成全部几何/通道量,Config.sync 写回只读 property,models.factory / pipelines.factory / predictor 三处均读它(@d:\codes\work-projects\SegTask\segtask_v1\models\topology.py:1-160)。新增 patch_mode 只改一处决策树——架构收敛干净。
Config.validate() 作为统一前置门:_validate_model 覆盖 arch×patch_mode、r2plus1d×2.5D、save_best_criterion='loss'×high 模式等大量跨模块互斥(@d:\codes\work-projects\SegTask\segtask_v1\config.py:1119-1345),质量很高。
RNG 确定性纪律一致:data(逐 worker 重建)、val(确定性种子)、trainer(四源位精确 resume)口径统一。
共享 warp 几何一致性链:augment 单次 warp → views/pipelines 逐视图裁剪 → losses(MR/DS)通道 layout,端到端严格对齐。
pooled Dice 口径 train↔(high)inference 一致,与 nnU-Net global dice 对齐。
真正的跨模块不一致(按优先级)
[高] config↔model 校验缺口 → hierarchical + aux_seg_supervision 首个 forward 崩:_validate_model 对 arch=='unet' 放行 stem_fusion_mode='hierarchical'(@d:\codes\work-projects\SegTask\segtask_v1\config.py:1188-1192),但与 aux_seg_supervision=True 无互斥校验,而 aux 头读低分辨率特征却断言等于输入尺寸(unet.py:462-511)。这是"强 validate 层里的一个精确漏检"。
[中] config↔model 校验缺口 → patchN stem 无解码补偿:config 注释宣称"末尾上采补回",但 validate 未校验 decoder 是否补偿 N 倍下采样(stem.py:107-114 vs unet.py:494-499)。
[中] trainer/utils ↔ predictor 二值化阈值不统一:medium 恒 sigmoid>0.5(utils.py:253),high 用 predictor.threshold(validation.py:436)。部署阈值≠0.5 时,medium 选出的 best 不反映部署行为。
[中] config↔predictor blend_mode 语义/日志错位:z 轴/2.5D 滑窗恒 gaussian、忽略 blend_mode='average',日志却打印 blend_mode(sliding.py:90/183 vs blending.py:49)。
[中] trainer↔losses 注释过强:"grad-accum/DDP 与单卡严格等价"仅对逐样本可分损失成立,对 batch_dice/Tversky/GDL 等区域比值型不成立(trainer.py:222-224)。
[中] predictor 内部语义不一致:global AdaBN 未置 _adabn_estimating,BN 重估期 TTA 变体被拼 batch,与 per_volume 刻意规避的语义矛盾(io.py:197-208 vs forwards.py:61-71)。
纠错(交叉核验推翻了原模型评审的一条)
原"模型构建"评审记:selfattn/multirf 在 convnext/mednext 下静默忽略、缺 warning。实际不成立:_validate_selfattn/_validate_multirf 均 硬报错 backbone=='resnet'(@d:\codes\work-projects\SegTask\segtask_v1\config.py:1385-1388、1479-1482)。因此这不是"静默 no-op",而是启动即 ConfigError。这一条应从模型评审的 [中] 缺陷里撤下(config 层已正确拦截)。
二、架构级建议(按优先级)
[最高性价比] 把剩余跨模块互斥/依赖补进既有 Config.validate()。项目已有成熟 validate 框架,只需补 3 条 _require,即可把"运行期首个 forward 崩"提前到"启动即清晰报错":
hierarchical + aux_seg_supervision 互斥(或改为 aux 头上采,兑现 docstring)。
stem_mode ∈ {patch2,patch4} 时要求主头/DS 头有对应上采补偿。
len(deep_supervision_weights) == 实际解码 DS 头数(现仅运行时 ValueError)。
统一 train↔inference 口径的单一来源:threshold / blend_mode / 二值化方式应从同一 config 字段派生、两侧一致消费,消除 #3、#4 两处一致性缺口。
单一真相源再彻底一步:build_model 的 encoder in_channels 改读 topo.in_channels(现读 mc.in_channels,依赖 sync 时序);DS 头数由 topo 派生并回校权重长度。
引入"数据集指纹"元数据层:在 make_data 的 meta 里补 image_shape/spacing/前景强度统计。一处架构改动同时解决:_build_index 句柄泄漏 + 全量解码(dataset.py:694/926)、detect_label_values 主进程串行瓶颈(loader.py:356)、以及下面的强度归一化——是回报最高的抽象。
三、可借鉴的业界高质量做法(nnU-Net / MONAI,按优先级)
高
nnU-Net 前景强度指纹归一化:采集前景体素 0.5/99.5 百分位 + mean/std,CT 先 clip 再 zscore。你已实现 spacing 中位数指纹,强度指纹是自然下一步,跨机构/协议 CT 更稳(接建议 #4 一并落)。
滑窗高斯 σ 对齐 nnU-Net 的 n/8(你用 n/4 偏平缓,窗边权重≈0.135),改善缝合抑制,近乎零成本。
中
nnU-Net 5-fold 交叉验证接口:分层划分逻辑已具备,扩展成本低,提升选模稳健性与可复现性。
per-case Dice 报告:pooled 用于选模(与 nnU-Net 一致),但最终报告另算 per-case,避免大目标掩盖个别病例失败。
MONAI MetaTensor / 真实 spacing 追踪:可解决 aspect_correct 不含真实 spacing 导致厚层 out-of-plane 旋转物理不准的已知限制(augment.py:114-123)。
AdaBN 估计期一律关 TTA:纯目标域 BN 统计只用原图前向,顺带解决 #6(global/per_volume 语义分歧)。
低
以 MONAI SlidingWindowInferer / DynUNet 作对照基线,给你的滑窗/装配做回归验证(不引入运行期依赖,仅测试期)。
四、单一优先级 backlog(合并全部模块真问题,供执行排期)
P0 [高]:hierarchical+aux 崩(unet.py:462-511)+ config 前置校验。
P1 [中]:patchN stem 无解码补偿崩(stem.py vs unet.py:494);_build_index 句柄泄漏+全量解码(改 meta shape);translate 与 oversample 余量未协调(augment.py:145);z 轴忽略 blend_mode(sliding.py:90/183);global AdaBN estimating flag(io.py:197);medium/high 阈值不一致。
P2 [中/注释]:batch_dice 等价注释修正(trainer.py:222);z_axis 面内无 oversample 余量说明。
纠错:selfattn/multirf 非 resnet —— config 已硬报错,非缺陷,从模型评审移除。