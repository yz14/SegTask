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

进展： 
数据读取审查报告  
审查范围：dataset.py（IO/预处理/bbox/缓存/3 个 Dataset）、loader.py（发现-匹配-划分-DataLoader 工厂）、specs.py（策略选择）、mixed_sampler.py（双源混批）、make_data.py（npz 烘焙）。

一、总体评价
数据层设计成熟、职责清晰，几处工程细节达到业界水准：

npz-only 训练契约（SegDatasetNpzBase.__init__ 的 assert，@/Users/.../dataset.py:497-498）配合 OS page cache 共享的 OOM 论证，取舍合理。
逐 worker 采样 RNG（@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:530-544）正确规避了"DataLoader fork 后各 worker 复制全局 numpy RNG 导致跨 worker 重复采样"这一经典坑。
验证确定性采样（@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:546-555，(seed, sample_idx) 派生）保证 save_best/early-stop 不被采样噪声驱动。
类均衡前景采样（先选类再选点/切片，@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:770-774 与 :1021-1030）对稀有小结构友好，优于合并前景采样。
单 max-FOV 一次抽取、多分辨率延迟到 trainer：消除多次 zoom 的高频损失且保证多 view 几何一致，是这套框架的核心亮点。
二、正确性 / 潜在 Bug
1. resize_3d 依赖 scipy zoom(factor)，输出尺寸可能 off-by-one（中风险）



dataset.py:378-388
    factors = [target_d / D, target_h / H, target_w / W]
...
return zoom(arr, factors, order=order).astype(arr.dtype, copy=False)
scipy.ndimage.zoom 的输出尺寸是 round(input_size * factor)。当 factor = target/current 时，理论上等于 target，但浮点舍入在某些比值下会得到 target±1。一旦 batch 内不同样本产生不同尺寸，default_collate 的 torch.stack 会直接报错。cubic 多 scale 的"抽取→resize 回"往返最易触发。建议：改用 torch.nn.functional.interpolate(size=...) 或对 zoom 结果做尺寸兜底裁/补，保证输出严格等于目标尺寸。

2. cubic 图像 patch 会 alias 到 LRU 缓存卷（低风险，潜在隐患） _extract_cubic_patch 在无越界填充时返回的是缓存卷的视图：



dataset.py:845-855
patch = vol[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
if any(pb > 0 or pa > 0 ...):
    patch = np.pad(...)
return patch
_getitem_max_fov 对 image 用 astype(np.float32, copy=False)（@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:975），image 已是 fp32 → 不复制，返回张量与 _img_cache 中的卷共享内存；label 则用了 np.ascontiguousarray 保护（:976）。当前安全（collate 会 stack 复制、增强在设备端新张量上进行），但只要未来某处对样本做 CPU in-place 操作就会污染缓存。建议：image 同样走 np.ascontiguousarray（顺带修复视图非连续、利于 pin_memory）。

3. SegDataset3DWhole.__getitem__ 未写 self._sample_idx（极低风险，一致性问题）



dataset.py:1091-1093
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    vol_idx    = idx % len(self.image_paths)
另两个 dataset 入口都会 self._sample_idx = idx。whole 目前无采样 RNG 依赖故无害，但一致性缺口——若将来 whole 也引入确定性采样会成隐性 bug。建议补一行保持一致。

三、优化空间
1. 启动期标签扫描重复解码全部 label 卷（性能） build_dataloaders 里 detect_label_values + stratified_train_val_split 通过 load_npz_label_for_split 逐个全量解码 label 卷（@d:\codes\work-projects\SegTask\segtask_v1\data\loader.py:646-674）。而 make_data 早已计算并写入 _manifest.json 的 label_values（@d:\codes\work-projects\SegTask\segtask_v1\data\make_data.py:449），且逐类前景信息已存在每个 npz 的 fg_slices_cls 中。大数据集下这是明显的启动开销。建议：优先读 _manifest.json 的 label_values；分层划分的主类可从 npz 的 fg_coords_cls/fg_slices_cls（已在内存索引里）直接统计，免全量 label 解码。

2. 验证集 patch 采样是随机/部分覆盖，指标偏噪（合理性） z_axis/cubic 验证走的是"逐样本确定性但随机位置"的 patch（_sample_z val 分支 rng.integers(0,D_vol)，@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:775；fg_ratio val=0，specs.py:134-136）。虽有确定性，但仍是部分覆盖、位置随机，patch-Dice 噪声较大。项目另有整卷 high-res 验证（validation.py，步骤 4 细看）——若模型选择已完全依赖整卷验证，则 patch-val 主要是训练期廉价监控；否则建议 patch-val 改为系统性网格覆盖以降噪。

3. 默认值文档/代码不一致（低） config.py 中 cache_max_volumes 默认 1（@d:\codes\work-projects\SegTask\segtask_v1\config.py:151），而 SegDataset3D 签名默认 0、README 早期段落也提 0。以 config 为准（=1，安全），但建议统一叙述避免误读。

四、可借鉴 / 可新增（业界做法）
数据集指纹复用（nnU-Net 风格）：make_data 已产出 manifest（label_values、spacing 中位数思路见 _resolve_target_spacing，@d:\codes\work-projects\SegTask\segtask_v1\data\make_data.py:295-316），但 loader 未复用。把 manifest 作为"指纹单一真相源"接进 loader，可省掉启动扫描，也更接近 nnU-Net 的 fingerprint 流程。
page-cache 优先的缓存策略：warning 里已建议 cache_mode:"none" 靠 OS page cache 共享（@d:\codes\work-projects\SegTask\segtask_v1\data\loader.py:829-836），可考虑在 npz 模式把它作为默认，减少逐 worker LRU 的 RAM 倍增。
对 TODO 1（gentask）的直接价值：本层"单 max-FOV 一次抽取 + 增强后中心裁 + oversample 余量 + region_weight 优先级 + 类均衡前景采样"的模式，正是 TODO 1 要复用的 2.5D 读取骨架。SegDataset3D（z 轴滑窗）几乎可原样改造为"z 轴取起止切片做 VFI 输入 / 在 x 轴取切片做 SISR 输入"，只需把 label 从二值通道换成高分辨率 GT。建议后续 gentask 直接拷贝 SegDatasetNpzBase + SegDataset3D 为起点。


模型构建审查报告
审查范围：models/ 全部——拓扑派生(topology.py)、装配(factory.py)、UNet 主体(unet.py)、三种 decoder(unet/unetpp/unet3p)、四类骨干(resnet/convnext/mednext + block 仓库 blocks.py)、stem 与多 FOV 融合(stem.py)。

一、总体评价
模型层是整个项目最出彩的部分，架构清晰、可组合性强、算法实现忠于原论文：

拓扑单一真相源（build_topology，@d:\codes\work-projects\SegTask\segtask_v1\models\topology.py:74-160）把 in_channels/out_classes/spatial_dims/aux 拓扑 一次算齐，Config.sync 与 factory 都只读不再重推——这是很好的防漂移设计。
工厂 + 有状态 stage builder + block 注册表（factory.py、resnet._BLOCK_REGISTRY）让"骨干 × decoder × block × 注意力 × 上下采样"任意拼装，扩展成本低。
积木库覆盖面达到业界水准：SE/ECA/CBAM/Coord 注意力；自注意力 softmax/linear/window/grid + nD RoPE + GEGLU FFN + zero-init 残差；BlurPool、CARAFE、DySample、PixelShuffle(+ICNR)；MedNeXt 的 UniRepLKNet 式 DilatedReparamBlock（训练多分支、推理重参数化）与 UpKern 权重迁移。均忠实对应论文。
nnU-Net 式各向异性下采样自动调度（_auto_anisotropic_strides，@d:\codes\work-projects\SegTask\segtask_v1\models\factory.py:257-283）配合 encoder/decoder stride 镜像（unet.py Decoder），对薄 z 轴医学体积很关键，且对 unetpp/unet3p/ConvNeXt-LN 下采样做了明确的兼容性拦截。
深监督 / 多 FOV aux 头 / 拓扑 aux 头、逐 stage 梯度检查点（checkpoint_if 用 use_reentrant=False + preserve_rng_state=True）都实现正确。
二、正确性 / 潜在 Bug
1. patch2/patch4 stem 与 UNet decoder 拓扑冲突，主输出必然分辨率不匹配（高价值，需确认） config.py:315 注释称 "patchN 降 N 倍分辨率（UNet3D 主输出加上采样）"，但 UNet3D 明确声明不做上采样补偿、不匹配即 RuntimeError：



unet.py:393-396
# 主头读最高分辨率 decoder 特征。stem_stride>1 时 decoder 最高分辨率仍低于
# 输入，forward 不做上采样补偿，而是显式 RuntimeError（要求 decoder 拓扑
# 与 stem_stride 配套，保证输出 = 输入分辨率）。DS 头保留各自分辨率。
而 Decoder 的上采样级数恒为 n_levels-1，只镜像 encoder 各 stage 间的 n-1 次下采样，不包含 stem 的 N 倍下采样（@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py:271-288）。因此 stem_mode=patch2/patch4 时主头输出恒为 输入/N，必触发 unet.py:495 的 RuntimeError。config.validate 也未禁止该组合。结论：patchN stem 对 decoder_type='unet' 分割实际不可用，且 config 注释与实现自相矛盾。建议二选一——要么在 UNet3D 主/DS/aux 头后补一次 stem_stride 上采样（兑现 config 注释），要么在 validate 里显式禁止 patchN × unet decoder 并更正注释。

2. DualConvStem 文档与实现不符（低，误导性）



stem.py:20-35
class DualConvStem(nn.Module):
    """两个堆叠 3×3×3 conv-norm-act（nnU-Net stem）。"""
    ...
        self.block1 = ConvNormAct(
            in_ch, out_ch, kernel_size=7, stride=1, padding=3,  # 第一层用7x7x7会不会好？
docstring 说"两个 3×3×3"，实现却是 7×7×7 + 3×3×3，且留有未决问句注释。参数量/感受野与描述不符，建议更正 docstring 或定回 3×3（并移除临时注释，符合规则五）。

3. 全局 RoPE cos/sin 缓存无界增长（低，长跑内存） _ROPE_ND_CACHE（@d:\codes\work-projects\SegTask\segtask_v1\models\blocks.py:29）以 (spatial_shape, device, dtype, axis, ...) 为键缓存且从不淘汰。滑窗推理/多分辨率训练会不断产生新形状键，长期运行缓存条目单调增长。建议改为有界 LRU 或按 module 生命周期管理。

三、优化空间
1. 小 batch 3D 下多处硬编码 BatchNorm（合理性/稳定性） CoordAttention3D（@d:\codes\work-projects\SegTask\segtask_v1\models\blocks.py:309 用 _BN）与 AttentionGate3D 默认 norm_type='batch'——其 psi 甚至在 1 通道特征图上做 BatchNorm（blocks.py:926-929）。3D 分割常用 batch_size=2（config 默认，见步骤 1），此时 BN 统计极噪。建议这些注意力/门控子模块的 norm 跟随全局 norm_type（instance/group）或至少默认 group，避免与主干 InstanceNorm 的稳定性取向相悖。

2. get_norm group 回退是静默降级（可维护性）



blocks.py:136-139
elif norm_type == "group":
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)
不整除时静默把组数折半（最坏退化为 1 组 ≈ LayerNorm）。MultiRFBlock 已选择显式报错（resnet.py:351-362），但全局其余路径仍静默。建议至少 warning 一次，避免用户以为在用 8 组实则 1 组。

3. _LinearQKVAttention 的数值细节（低） 线性注意力对 K 在 token 维 softmax（blocks.py:803），当序列很长时 KᵀV 聚合可行但对分布偏移较敏感；当前无 eps 稳定项。属可接受实现，若后续放到浅层大分辨率处使用，建议加小 eps/温度以稳住训练早期。

四、可借鉴 / 可新增（业界做法）
对 TODO 1（gentask 超分/VFI）的直接价值最大：blocks.py 里的 PixelShuffle3d(+ICNR)、CARAFE3d、DySample3d、BlurPool3d 正是超分/插帧上采样的主力算子，可直接迁移到 gentask 的上采样端；SelfAttentionBlock（window/grid/RoPE）可作为 SISR transformer 分支的骨架。这套 dim-agnostic(2D/3D) block 仓库让 gentask "在 z 轴 SISR / 在 x 轴取切片" 的两种方案都能复用同一批算子。
MedNeXt 档位 B 补全：当前 mednext.py 档位 A 复用通用重采样，原生"重采样残差块"（stride 融进深度卷积 + 1×1 残差）与更完整的 UpKern 尚未落地（文件头已注明）。若追求 MedNeXt SOTA 表现值得补齐。
`nnU-Net ResEnc 预设 + 各向异性调度已具备，可进一步引入 nnU-Net v2 的"逐轴 kernel size 各向异性"（薄 z 轴用 1×3×3 卷积核），当前各向异性只体现在 stride，卷积核仍各向同性 3。
arch=adm|edm2 生成式骨干已在库中（magnitude-preserving conv/SiLU），是 gentask 的现成参考实现，TODO 1 可优先评估复用。


数据增强 / 处理 审查报告
审查范围：augment.py（GPU 空间/强度增强）、views.py（center_crop / 3D·2.5D 懒多分辨率拆分 / 2.5D fold）、trainer/pipelines/*（6 类 ViewPipeline 的 prepare_batch 视图重塑）、trainer.py:705-723 的「H2D→增强→中心裁→视图重塑」时序、config.py 的 AugConfig。

一、总体评价
增强/处理层与数据读取层衔接自洽，几处设计达到业界水准：

单 max-FOV 抽取 + 增强后中心裁 + 懒多分辨率闭环完整：dataset 只发一份 max-FOV cube，trainer 先在带 oversample 余量的张量上做空间增强、再 views.center_crop 回 target_patch_size（@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:714-718），最后由 pipeline 懒拆多分辨率视图。空间增强只做一次、几何天然一致，是核心亮点，也正是 TODO 1(gentask) 要复用的骨架。
增强全程 GPU、逐样本独立：flip/affine/elastic/dropout 用逐样本 Bernoulli mask + nonzero 选中子集处理，grid_sample 一次对整个选中子集向量化，weight_map 随空间变换同步（插值模式可配）。
ViewPipeline 策略化彻底消除模式分支漂移：build_pipeline（@d:\codes\work-projects\SegTask\segtask_v1\trainer\pipelines\factory.py:44-68）是唯一 if/elif 处，派生量全来自 ModelTopology；prepare_batch/compute_loss/prepare_val_batch 三接口把「视图重塑 + 损失聚合」封装干净，Trainer._train_epoch 无需预知任何模式。
强度增强 nnU-Net 式 clamp：增强前记录逐样本逐通道 min/max、全部强度增强后夹回（augment.py:70-81），避免 brightness/contrast/noise 叠加越界污染 gamma 语义。
gamma 仅在空间轴 reduce、通道独立（augment.py:372-386），对 eager 多分辨率(每通道不同 scale 不同强度范围)是正确的；gaussian_blur 用 groups=n*C 分组卷积把「逐样本不同 σ」一次并行，工程实现优雅。
二、正确性 / 潜在 Bug
1. _grid_dropout 在 hole 尺寸 ≥ 维度长时会索引越界（低风险，边界）



augment.py:296-304
frac = (ratio / max(num_holes, 1)) ** (1.0 / 3.0)
hd = max(1, int(D * frac))
...
d0 = torch.randint(0, max(D - hd, 1), (B, num_holes), device=device)
当 hd > D（即 ratio 很大或 num_holes 很小导致 frac>1）时，max(D-hd,1)=1 → d0=0，而 ds = d0 + arange(hd) 会达到 hd-1 > D-1，hole_mask[..., ds, ...] 越界报错。默认 ratio=0.3, holes=4（frac≈0.42）安全，但缺乏对 hd/hh/hw ≤ D/H/W 的兜底。建议：hd = min(D, max(1, int(D*frac)))（三轴同理）。

2. 弹性形变 alpha 与「体素位移」语义不严格一致（低，文档性） disp 由粗网格 randn(std=1) 经 trilinear 上采样得到（augment.py:247-248），插值使内部点方差 <1，故真实最大位移 < alpha·(2/N) 体素，config 注释「位移幅度(voxel)常 3–12」略高估实际幅度；再叠加 effective_alpha = alpha/max_scale(augment.py:58)。功能正确，仅标称单位偏乐观。建议：注释注明「近似幅度、受平滑衰减」，或对位移场按范数归一后乘 alpha 以兑现「最大体素位移」语义。

3. config 注释「Affine…合成单次 grid_sample」与实际两次重采样（低，误导性） config.py:178 称仿射合成单次 grid_sample —— 对 affine 自身成立，但 __call__ 中 _random_affine 与 _elastic_deform 是两次独立 grid_sample（augment.py:50-62），选中样本被连续重采样两遍，产生双重插值模糊。属实现取舍，但注释易让人以为全空间变换只采样一次。建议：把 affine 的仿射位移与 elastic 位移场相加后合成单次 grid_sample（既消双重模糊又省一次重采样，见优化 1）。

4. scale_range 的 grid_sample 语义为「反向」（低，文档性） m = scales * rot（augment.py:216）作用在采样网格坐标上：scale>1 使网格坐标外扩、采样范围变大 → 物体在输出中变小。直觉上 random_scale_range=[0.85,1.15] 的「1.15」实际是缩小而非放大。功能对称无害，但语义方向与直觉相反，值得在注释澄清。

以上均非高危项——数据增强层未发现会导致训练崩溃或标签错配的正确性硬伤（flip/affine/elastic 对 image/label/wmap 用同一变换、label/wmap 走 nearest 保离散，均正确）。

三、优化空间
1. Affine + Elastic 融合为单次 grid_sample（性能 + 质量） 当前对同一批选中样本先仿射重采样、再弹性重采样两遍。可将仿射网格与弹性位移场在归一化坐标里相加后只 grid_sample 一次：减少一次三线性重采样开销，并消除双重插值导致的额外高频损失。nnU-Net/MONAI 的 RandDeformGrid+Affine 组合即走单次 warp。

2. 每个增强算子的隐式 device 同步累积（性能） _random_affine 的 mask.sum().item()（augment.py:133）、_random_contrast 的 mask.sum().item()（:356）、_simulate_lowres 的 .tolist()（:457-458），以及各算子的 mask.nonzero()（flip/affine/elastic/noise/blur）——每个都会强制一次 CUDA→CPU 同步。一个 step 约 8–10 次同步，掩盖 H2D 与 kernel 重叠。建议：整批处理 + torch.where(mask, aug, orig) 混合代替 nonzero+scatter，或一次性预采样所有 Bernoulli mask，减少同步点。

3. inplace=False 下的多次额外分配（显存/性能） 入口 clone 一份后，brightness/contrast/gamma 各返回新张量（image + shift 等），叠加时产生多份 batch 体积的瞬时分配。可让强度算子在选中子集上做 in-place（如 image[idx].add_(shift)）以复用缓冲；与已存在的 inplace 契约方向一致。

4. grid_dropout 的 python for k in range(num_holes) 循环（低） 逐 hole 循环做高级索引赋值；holes 少时无所谓，可用一次性构造 mask 的向量化写法，属可选清理。

四、可借鉴 / 可新增（业界做法）
对 TODO 1(gentask) 的直接价值最大：
空间增强(flip/affine/elastic)天然对 image 与 label 施加同一几何变换，SR/VFI 的「退化 LR ↔ 高清 GT」正需要这种成对一致变换，可原样复用。
_simulate_lowres（augment.py:444-470，trilinear 下采→上采）本质就是一个退化模型，是 gentask「面内超分/厚层模拟」degradation pipeline 的现成起点；可扩展各向异性 zoom（仅 z 轴下采样以模拟厚层部分容积效应）。
views.split_views_native_d（2.5D 逐视图抽 slab）与 split_views_native_3d（多 FOV 中心裁+resize）正是 gentask「z 轴取起止切片做 VFI / x 轴取切片做 SISR」的多 FOV 读取骨架。
物理 spacing-aware 旋转：当前 aspect_correct 只按 voxel-count 各向同性校正（augment.py:151-155,212-214），未含真实 spacing；docstring 已注明局限。可在 data.spacing_normalization=False 时引入按物理 spacing 的出面旋转限幅（nnU-Net 对薄 z 轴限制 out-of-plane 旋转角），避免厚层数据的几何失真。
gamma invert / 通道级 gamma（nnU-Net 对反相图再做一次 gamma）与 mirroring-consistent TTA 可作为廉价增益补充。
MONAI/nnU-Net 的融合 warp 与 batched RandGaussianNoise：若后续追求吞吐，可对齐其「单 warp + 少同步」实现范式。


训练全流程(含val流程) 审查报告
审查范围：train.py（CLI 入口 / DDP spawn / 信号与孤儿兜底）、trainer/trainer.py（fit 主循环、_train_epoch、_validate、checkpoint/resume/pretrain、健康监测）、trainer/optim.py（optimizer/scheduler 工厂 + WarmupScheduler）、trainer/amp.py（AMP shim + fp32 损失）、trainer/validation.py（MetricAccumulator + medium/high 两 evaluator）、trainer/pipelines/*（compute_loss 聚合）、trainer/dist_utils.py、trainer/memory.py、trainer/checkpoint.py、utils.py（ModelEMA、pooled 指标算子）。

一、总体评价
训练层是工程完成度最高的一层，若干处达到或超过业界主流框架水准：

模式无关的训练循环：``Trainer._train_epoch`` 完全不判训练模式，所有「视图重塑 + 损失聚合」经 ``self.pipeline``（@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:721,737）归口，Trainer 只协调 optimizer/scaler/EMA/DDP。这套 R2 重构消除了模式分支漂移，扩展新任务（如 gentask）只需换 pipeline+loss+metric。
AMP 数值取舍正确：forward 走 autocast、损失强制 fp32（@d:\codes\work-projects\SegTask\segtask_v1\trainer\amp.py:73-94，logit 先 clamp(±50) 再 fp32 dice/bce），从源头规避 fp16 下 Dice/BCE 汇总溢出→NaN 的经典坑；``amp_dtype='auto'`` 按 bf16 能力自动择型（amp.py:66-70）。
非有限值的双路防护：fp16 交给 GradScaler 跳步；bf16/fp32 无 scaler 保护时显式丢弃整个 accum 组梯度（@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:786-815），且 scheduler/EMA 照常推进——避免 NaN 永久污染权重/EMA，这是很多自研 trainer 缺失的护栏。
梯度累积 × DDP 数学等价：非边界步用 ``fwd_model.no_sync()`` 免 all-reduce（trainer.py:727-733），边界步按有符号尾长 ``_effective_accum`` 归一（trainer.py:579-587），配合 DDP 的 mean-all-reduce，等效 batch=batch_size×accum×world_size 与单卡严格等价，遵循 PyTorch 官方 no_sync 惯例。
pooled 指标累加器（nnU-Net 风格）：``MetricAccumulator`` 逐 batch 累加 inter/pred_sum/target_sum/voxels/cov（@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:84-138），闭式一次导出 dice/iou/recall/precision/vol_sim/mcc/surface_dice/balanced，且多卡 all-reduce(SUM) 与单卡累加逐位相等（validation.py:140-188）——因混淆量可加，这是最正确的分布式指标聚合方式（优于各 rank 求 dice 再平均）。
medium/high 双评估策略同构输出：两 evaluator 唯一差别是 (pred,target) 来源，累加/导出共用 MetricAccumulator，故选模/调度/ckpt 无需分支（validation.py:290-471）。high 模式直接复用 ``Predictor`` 整卷滑窗并按推理阈值二值化喂入（``pred_is_binary``，validation.py:434-458），与部署口径一致。
resume 的位精确性与 DDP 分流：checkpoint 快照 torch/cuda/numpy/python 四路 RNG（trainer.py:937-944），resume 后 rank>0 用 ``_reseed_rank_rng`` 重新分流避免所有 rank 退化成 rank0 随机流（trainer.py:77-83,1054-1057）。best_model 以 EMA 为主权重、在线权重另存 ``model_online_state_dict``（trainer.py:960-969），resume 优先读在线权重、pretrain 可选读 EMA，方向正确。
进程健壮性：DDP 子进程 ``PR_SET_PDEATHSIG`` 父死即死 + SIGTERM/SIGINT 优雅销毁 pg + NCCL 异步超时（@d:\codes\work-projects\SegTask\segtask_v1\train.py:110-200），是"孤儿进程卡 NCCL 永久占卡"的成熟兜底，工业级细节。
监测/健康指标全程异常隔离（trainer.py:505-559,881-898）：任何监测失败仅告警、绝不打断训练，符合"副作用不影响主流程"的设计纪律。

二、正确性 / 潜在 Bug
1. 每 micro-step 多次 ``.item()`` 强制 host-device 同步（中风险·性能正确性） ``_train_epoch`` 每步都传入非空 breakdown 调 ``compute_loss``，pipeline 内对 ``L_main/L_aux_k/L_total`` 逐个 ``.detach().item()``（如 @d:\codes\work-projects\SegTask\segtask_v1\trainer\pipelines\slab25d.py:50,202,206），叠加主循环 ``step_loss = loss.item()``（trainer.py:745-746）。即使当前 step 不打日志（``log_every`` 未命中），这些 ``.item()`` 也照常执行——每 step 约 3~5 次 CUDA→CPU 同步，打断 forward/backward 与 H2D 重叠。这与「数据增强报告」里指出的同步问题同源。建议：分量在 GPU 上以张量累加，仅在 ``log_every`` / epoch 末一次性 ``.item()``；或让 ``compute_loss`` 在 ``breakdown is None`` 时跳过标量抽取，主循环仅在需要写 meter 的步传 breakdown。

2. EMA 在"被跳过的优化步"上仍 ``update()``（低风险，副作用） fp16 scaler 跳步路径（trainer.py:828-833）与 bf16 显式跳步路径（trainer.py:811-812）都会调用 ``self.ema.update()``。权重未变时 shadow 向同值收敛，数值无害，但 ``num_updates`` 仍自增，推进 EMA warmup 的有效 decay ``(1+n)/(10+n)``（@d:\codes\work-projects\SegTask\segtask_v1\utils.py:109-110），且在 ema_device='cpu' 时白白触发一次 D2H staging 同步（utils.py:111-119）。建议：优化步真正生效后再 ``ema.update()``（把 update 移入未跳步分支）。

3. 训练 loss 与验证 loss 口径不一致（低风险，监控口径） medium 验证 loss 用裸 ``base_loss`` 在 1x reshape 后的张量上算（@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:338），而训练 loss 走 ``SliceChannelLoss``/``MultiResolutionLoss``（含 reduction、DS、aux 加权）。两者非同一泛函，监控面板上 train_loss 与 val_loss 不可直接比较；high 模式更是不产 val_loss（validation.py:97,355）。功能无误（选模只用 mean_dice/combined 等，不用 val_loss），但建议在文档/图例注明口径差异，避免误读为过拟合/欠拟合信号。

4. OneCycleLR 与 total_steps 的强耦合在 resume 漂移下会抛错（低风险，边界） OneCycle 的 ``total_steps = epochs×steps_per_epoch``（@d:\codes\work-projects\SegTask\segtask_v1\trainer\optim.py:141-148），且每个边界步（含被跳过的优化步）都调用一次 ``scheduler.step()``。正常/早停下 step 次数 ≤ total_steps 安全；但 resume 时若改了 ``epochs``/``grad_accum_steps``/数据量，恢复的 OneCycle 内部计数与新 total_steps 不符，可能触发 "Tried to step N times but total_steps=M"。``WarmupScheduler.load_state_dict`` 已对 warmup 漂移告警（optim.py:210-234），但未覆盖 base scheduler 的 horizon 漂移。建议：resume 时对 one_cycle 检测 total_steps 变化并告警/兜底重建。

5. 训练期 ``"dice"`` 为稀疏抽样、却按 epoch 级上报（低风险，监控口径） train dice 仅在 ``(step+1)%log_every==0 or step==0`` 时计算并入 ``dice_meter``（trainer.py:870-877），故 ``train_metrics["dice"]`` 是抽样均值而非全 epoch。作为廉价训练监控可接受，但严格说与 val dice 不同口径，建议注明或按需全量。

以上均非会导致训练崩溃/标签错配的高危硬伤；#1 为最值得优先处理项（吞吐相关，改动局部）。

三、优化空间
1. 消除每步同步点（吞吐） 同「二.1」。foreach 化的健康指标已很克制（trainer.py:592-641 仅每 epoch 少量 ``.item()``），但主损失/分量的 per-step ``.item()`` 是稳态吞吐的主要隐性同步源。GPU 端累加、稀疏落地即可显著改善 GPU 利用率，尤其小 batch 3D。

2. high 验证每 epoch 全量整卷滑窗成本高（可配置降频） ``VolumeValEvaluator.evaluate`` 每次验证对全部 val 卷做一遍部署级滑窗推理（@d:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:396-460），最可靠但最慢。当前靠 ``val_every`` 全局降频。建议提供「前中期 medium、后段/里程碑 high」的混合策略，或对 high 单独设更大间隔，兼顾选模可靠性与训练墙钟。

3. 验证前后 ``empty_cache`` 仅在 ``val_empty_cache`` 且 CUDA 时（trainer.py:917-926） 对整卷大累加器 OOM 是有效兜底，但默认关。可在 val_metric_mode='high' 时默认开启（high 才需要连续大显存），减少用户踩坑。

4. checkpoint 保存为阻塞 ``torch.save``（trainer.py:986,990） 大模型 + 频繁 save_every 时阻塞主进程。可选异步/后台线程落盘（rank0），或对周期 ckpt 采用 ``torch.save(..., _use_new_zipfile_serialization=True)`` 之外的分离 I/O。属可选优化。

四、可借鉴 / 可新增（业界做法）
对 TODO 1(gentask) 的直接价值最大：整套 Trainer 骨架（AMP-fp32 损失、EMA、DS、grad-accum、DDP no_sync、warmup+scheduler 工厂、resume/pretrain/UpKern、健康监测、监测隔离）几乎可原样迁移到 gentask，只需替换 ①pipeline（SR/VFI 的退化-重建视图重塑）②loss（L1/Charbonnier/感知/SSIM/GAN）③metric。EMA 对 SR/超分尤其关键（SOTA SR 普遍用 EMA 稳定），此处实现（含 CPU offload、foreach、warmup）可直接复用。
指标累加器的可加性前提需注意：``MetricAccumulator`` 依赖「混淆量跨样本可加」才能 pooled + all-reduce 严格等价（validation.py:140-188）。PSNR/SSIM 不是线性可加量（PSNR 含 log、SSIM 为比值），gentask 若照搬需改为「逐卷算指标再算数平均 + 计数 all-reduce」，或累加 MSE(可加) 后统一转 PSNR。建议在 gentask 复用时显式区分「可加混淆量」与「不可加评分」。
high 模式的整卷滑窗验证正是 SR 的部署级评估范式（validation.py:350-460）：gentask 可复用「val 阶段跑一遍与推理一致的滑窗/拼接」来对齐训练-部署差异。
选模标准的单一真相源（save_best_criterion→(metric,mode) 派生，config.py:743-754,831）值得推广到 gentask：把 "psnr/ssim/lpips/combined" 也做成映射表，避免 metric 名与 mode 手配不一致。
可新增：梯度累积下的 BN/统计校正——若 gentask 用 BatchNorm 且 batch 很小，可引入 nnU-Net 式 batch_dice 类比的「跨累积组统计」或改 GroupNorm（与模型报告中「小 batch 3D 慎用 BN」呼应）。
可新增：EMA 权重同时保存"在线+EMA"已具备（trainer.py:960-969），但验证仅用 EMA；建议提供「同一 epoch 同时报 online 与 EMA 指标」的可选诊断，便于判断 EMA 是否真正带来增益（长跑早期 EMA 可能拖累）。