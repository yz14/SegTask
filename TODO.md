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


segtask_v1是2.5D/3D分割项目(见对应README.md)
gentask是生成/超分项目。大多数设计和代码和分割代码非常类似，注意：  
生成/超分不一定有img-lable对；可能只有lable(高质图像)，img要对应任务的退化。可能有bbox, region_weight。
任务可能是：  
厚层生成薄层(d,H,W -> D,H,W)，可能需要逼真的薄层模拟出厚层（例如部分容积效应等等）。  
面内超分(D,h,w -> D,H,W)。  
ssltask是自监督学习项目，主要是对分割，生成，分类（暂缺），检测（暂缺）的backbone（部分/全部）进行预训练，提升下游模型指标。


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

2 分割项目代码分析审查，需要认真，详细的分析，思考和调研。保证高质量完成。为了保证高质量完成，本轮不动任何代码/文档：  
我希望你关注模型/算法架构的方向，例如unet是否有升级的地方，注意力是否有更好的算法，这些可能需要细致的调研现有的高质量论文/项目。我说一下我之前怎么做的，我先实现了unet，然后升级了unet++，然后升级加入了multirf, qkv attn, linear attn等等这些。我希望你也可以类似这样的给我一些算法层面的建议。

进展：  
A. 正确性问题（高优先级，疑似 bug）
A1. LR 调度器 horizon 与梯度累积不匹配


trainer.py:120-123
steps_per_epoch = len(train_loader)
warmup_steps    = tc.warmup_epochs * steps_per_epoch
total_steps     = tc.epochs * steps_per_epoch
total_steps/warmup_steps 按 micro-batch 数计算，但 self.scheduler.step() 只在优化步边界调用（trainer.py:705-741）。当 grad_accum_steps = N > 1 时：

warmup 实际持续 N× 个 epoch（warmup 以优化步计数，但目标步数按 micro-step 给定）；
cosine/poly 永远走不完：整个训练只推进 horizon 的 1/N，结束时 LR 停在 cos(π/N) 对应值而非 cosine_min_lr。
修复很简单：steps_per_epoch = ceil(len(train_loader) / accum)。grad_accum=1 时无影响，所以此 bug 可能一直没被现有实验暴露。

A2. EMA × torch.compile 的 state_dict key 不匹配（需验证）
@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:154 先用裸模型构建 ModelEMA（shadow key 无前缀）；
trainer.py:171 随后 self.model = torch.compile(self.model)；
trainer.py:742-743 调 self.ema.update(self.model)——此时 model.state_dict() 的 key 带 _orig_mod. 前缀；
@d:\codes\work-projects\SegTask\segtask_v1\utils.py:55-59 的 _build_pairs 直接 self.shadow[k] 索引 → 首个优化步即 KeyError（apply_shadow/restore 同理）。
推测现有环境（Windows 无 Triton，compile 回退 eager）从未真正触发。建议：EMA 内部对 key 做 _orig_mod. 归一化，或 EMA 始终持有 unwrap 后的裸模型引用；并补一条 compile+EMA 冒烟测试。

A3. ModelEMA 文档自相矛盾
utils.py:40 写"仅单卡，不兼容 DDP/FSDP"，但 Trainer 在 DDP 下无条件启用 EMA。实际上因为梯度 all-reduce 后各 rank 参数一致，EMA 数学上是对的——应更新注释而非代码，但目前的措辞会误导维护者。

A4. bf16/fp32 下非有限 loss 不会被跳过，NaN 会永久污染权重与 EMA（高优先级）


trainer.py:150-151
self._scaler_active = self.use_amp and self.amp_dtype == torch.float16
self.scaler = GradScaler("cuda", enabled=self._scaler_active)
trainer.py:793-797 出现非有限 loss 时仅告警，并声称 "GradScaler will skip this optimizer step" —— 这只在 fp16（scaler 启用）时成立。当 amp_dtype 为 bf16（auto 在 Ampere+ 会解析为 bf16）或纯 fp32 时，GradScaler(enabled=False).step() 直接调用 optimizer.step()，不做 inf/NaN 检查：一个 NaN loss → NaN 梯度 → 权重全 NaN → EMA shadow 随后也被污染，训练不可恢复。

建议：优化步边界处，若本 accum 组内出现过非有限 loss（或检测到梯度非有限），zero_grad 并跳过该步（含 scheduler/EMA），与 fp16 路径语义对齐；同时修正误导性告警文案。

A5. medium 验证的 patch 每 epoch 随机重采，选模指标含采样噪声
dataset.py 的 _sample_z（686-693 行）在 is_train=False 时走均匀随机采样，且 per-worker RNG 从 torch.initial_seed() 派生（dataset.py:477-482），每个 epoch worker 种子不同 → 每次验证看到的是不同的随机 patch 集合。
后果链：save_best_metric、early-stopping patience、plateau 调度（trainer.py:352-353）全部被同一采样噪声驱动。小验证集 + 稀有类时，best checkpoint 可能选中"运气好"的 epoch。
建议：val 模式用确定性采样（固定种子，或每卷固定网格取 patch），或推荐生产配置用 val_metric_mode=high（整卷滑窗，validation.py 已支持且实现质量很好）。

A6. one_cycle 的 horizon 同属 A1 bug 家族


optim.py:75
total_steps = tc.epochs * steps_per_epoch
build_scheduler 内部 one_cycle 分支自行用 epochs * steps_per_epoch（micro-step 数）计算 total_steps，A1 修 steps_per_epoch = ceil(len(loader)/accum) 时此处会一并修好，但需注意 pct_start 下限逻辑（optim.py:79）也依赖 total_steps，修复时应同步验证。

A7. DDP 全 rank 同 seed，随机流跨 rank 完全相同（需结合实验确认影响）


train.py:183
seed_everything(cfg.train.seed, cfg.train.deterministic)
_train_worker 中所有 rank 用同一个种子：

GPUAugmentor 用 torch.rand(device)，各 rank CUDA RNG 相同 → 每 step 各 rank 抽到完全相同的增强参数序列（作用在不同数据上，非致命，但增强多样性打折）；
DataLoader worker 的 numpy RNG 由 torch.initial_seed() 派生 → 各 rank 的 fg/均匀采样决策序列相同。
建议：seed_everything(cfg.train.seed + local_rank, ...)（数据切分由 DistributedSampler 保证，不受影响）。另注意 _load_checkpoint 会把 rank0 的 RNG 状态恢复到所有 rank（trainer.py:934-947），resume 后同样全 rank 同流。

B. 算法/模型架构层面（重点）
B1. 缺失物理 spacing 归一化——最大的系统性短板
全管线（make_data → dataset → augment → model）没有任何 target-spacing 重采样；spacing 只在推理 z-interleave 里读过一次（dataset.py:95-114）。后果：

训练分布中同一解剖结构的体素尺度不一致（不同设备/协议的 CT，面内 0.5–1.0 mm、层厚 0.5–5 mm 都有可能），模型被迫学习尺度不变性，浪费容量；
z 轴 slab 的物理 FOV 随层厚漂移：patch_size[0]=64 在 1 mm 数据上是 64 mm，在 5 mm 数据上是 320 mm，多 FOV multi_res_scales 的语义也随之漂移；
nnU-Net 的经验表明 spacing 归一化 + 由数据指纹推 patch/stride 是 3D 分割最重要的"免费"增益之一。
建议方向：在 make_data.py 烘焙阶段加入 target-spacing 重采样（中位数 spacing 或各向异性策略），npz meta 里已有落点；推理侧 Predictor 镜像重采样 + 概率图回采样。这是对现契约（"数据集只产单分辨率 max-FOV cube"）侵入最小的插入点。

B2. z_axis/2.5D 的面内"整片 resize 到 patch"损失过大


dataset.py:666-668
# 面内 resize 到 (eH,eW)；D 轴保持 eD_max（不重采样）。
img_s = resize_3d(img_s, eD_max, eH, eW, is_label=False)
lbl_s = resize_3d(lbl_s, eD_max, eH, eW, is_label=True)
默认 patch_size=[64,128,128]，而 CT 面内原生 512×512（bbox 裁剪后也常 >256）。这意味着：

4× 下采样直接抹掉小结构与边界细节，且 label 用 order=0 resize，GT 本身被量化；训练/推理都在低分辨率坐标系里做，最后概率图再插值回原尺寸，边界 Dice/表面指标的上限被数据层锁死；
面内没有随机 crop —— 每个样本的面内内容恒定（只有 z 在动），面内平移多样性 = 0，等效增强量明显低于 cubic 模式；
每个 __getitem__ 对 eD_max×H×W 做 scipy zoom（单线程 C 循环），是 CPU 数据管线的最大热点；这个 resize 结果对同一 volume 是恒定的，却每个 sample 重复计算——至少应把面内 resize 挪进 make_data 一次性烘焙，或在 GPU 上做。
建议：优先评估"面内原生分辨率 + 随机 crop（如 64×256×256 或 D×192×192 滑窗）"方案；若显存受限，至少把固定 resize 前移到烘焙阶段。

B3. 空间增强的物理正确性
@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py:92-129 的仿射在 affine_grid 归一化坐标系里做三轴欧拉旋转。归一化坐标把 (D,H,W) 都拉成 [-1,1]，当 patch 各向异性（64 vs 128）或 voxel spacing 各向异性时，旋转实际是旋转+剪切+非均匀缩放的混合形变，且绕 x/y 轴的"旋转"在厚层数据上物理失真最严重。同一 random_rotate_range 施加于三轴也不符合 CT 惯例（业界通常面内 ±180°/±30°，出面小角度或禁用）。

建议：a) 旋转矩阵左右乘 aspect 校正对角阵，使旋转在"物理各向同性"坐标里进行；b) 三轴角度范围分开配置。elastic 形变有同样的归一化坐标问题（voxel_to_grid 已把位移逐轴换算，位移幅度是对的，但形变场平滑尺度 sigma 也是逐轴体素数意义上的，各向异性下平滑不均，影响较小）。

B4. 前景过采样不感知类别——稀有类饥饿
_sample_z（dataset.py:685-693）与 _sample_center（dataset.py:923-941）的 fg 索引把所有前景类合并（fg_slices / fg_coords 在 make_data 中不区分类别）。多类任务中体积大的器官会统治采样，稀有小类可能长期见不到。nnU-Net 的做法是：先随机选一个类，再从该类的 voxel 里采中心。改动点集中在 make_data（按类存 fg_coords）+ 两个 _sample_*，成本低收益明确。

B5. Dice 默认配置对稀疏前景不友好
config.py:467-469：batch_dice=False、ignore_empty=False 是默认。patch 训练下大量 patch 对某些类是空 GT，per-sample Dice 在空类上恒 ≈1（只由 smooth 支配），既抬高 loss 基线又稀释有效梯度；dice_bce 里 BCE 部分能兜底，但 nnU-Net 的默认（batch_dice=True）在稀疏前景上更稳。建议把生产 YAML 的推荐默认改为 batch_dice=True（代码已支持，纯配置问题），或至少在 default.yaml 注释里给出明确指导。

B6. Plan A 的 aux 头与主头读同一特征，监督冗余


unet.py:450-455
else:
    in_ch = decoder.out_channels[-1]
    for k in range(1, n_views):
        self.aux_feat_indices.append(n_dec - 1)  # 用最后一个dec特征
shared_stem/multi_stem_proj 下所有 aux 头都挂在 dec[-1]，与主头同源；aux 监督给 encoder/decoder 主干的梯度与主监督高度共线，多 FOV 信息"必须被主干保留"的正则效果打折。Plan C 挂在不同深度是对的。可考虑：Plan A 的 aux 头改挂 view-specific stem 输出或中间 decoder 层，或直接用消融（run_aux_sweep.py）验证 Plan A aux 是否真有增益，无增益就简化掉。

B7. one-vs-rest sigmoid vs softmax
全框架锁定"逐前景类独立 sigmoid、背景隐含"（losses.py:1-6）。对互斥多器官任务，softmax+CE+Dice 是更强的归纳偏置（类间竞争、概率归一）；sigmoid 适合重叠结构/region-based 训练。目前推理端对多个二值预测冲突的解决是 "argmax-ish"，语义不如 softmax 干净。

B8. 空间增强缺"平移"自由度，且 z_axis 模式面内增强量进一步受限
augment.py 的仿射只有旋转+各向同性缩放（_build_rotation_matrices 无平移分量，augment.py:156-160 平移列恒 0）。3D cubic 模式下随机 crop 本身提供平移多样性，尚可；但 z_axis 模式面内是整片 resize（B2），面内平移多样性 = 0 且仿射也不补。业界（batchgenerators/MONAI RandAffine）默认含随机平移。建议仿射矩阵加小幅随机平移，成本一行。

B9. 强度增强后无重裁剪/夹取
brightness/contrast/noise 后不 clamp（augment.py:63-68）。normalize 后数据分布已知（如 z-score 或 [0,1]），nnU-Net 在 contrast 增强后按原 min/max 夹取以避免分布外值。影响小，但对 [0,1] 归一化配置，gamma 前的 minmax 归一（_random_gamma）与 brightness 叠加可能产生越界值改变 gamma 语义。

B10. _gaussian_blur_3d / _simulate_lowres 逐样本 Python 循环
augment.py:353 / augment.py:387 逐选中样本循环 + 每次 torch.empty(1).uniform_()（CPU→GPU 同步点）。batch 大时是热路径上的隐性串行。可向量化（分组同 sigma / 同 zoom），优先级低于 B2 的 CPU resize 热点。

B11. 主头分辨率与 stem_stride 的注释-实现漂移
unet.py:377 注释称 "stem_stride>1 时 forward 末尾上采回输入分辨率"，但 forward（unet.py:477-481）实际是尺寸不符直接 raise，没有任何上采样。若 factory 保证 stem_stride=1 或 decoder 已镜像补偿则无 bug，但注释会误导后续维护（与 A3 同性质：文档漂移）。

B12. 优化器无参数分组：norm/bias 也被 weight decay


optim.py:24-26
params = [p for p in model.parameters() if p.requires_grad]
if   tc.optimizer == "adamw":
    return torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
InstanceNorm/GroupNorm 的 affine 参数与所有 bias 都吃了 weight decay。惯例（AdamW 场景尤其）是 norm/bias 免 decay。对分割影响通常温和，但这是零风险改进；同时为 ssltask 预训练权重的下游微调预留 param-group 机制（layer-wise LR decay）也需要这个分组入口。

B13. EMA decay 无 warmup
ModelEMA（utils.py:39-71）固定 decay=0.999。训练早期 shadow 被随机初始拖累，常见做法是 min(decay, (1+step)/(10+step)) 式 ramp-up（timm）。当前 val_every 首次验证若较早，EMA 权重可能显著落后在线权重，导致早期 best 判定失真。

C. 训练流程/工程
C1. DDP 下 medium 验证浪费 (N-1)/N 的 DataLoader CPU
validation.py:318-320：每个 rank 完整迭代 val_loader，靠 i % world_size != rank 跳过 —— 但 batch 已经被 worker 完整生产（含 B2 提到的昂贵 resize）。val 集大时验证阶段 CPU 开销是单卡的 N 倍。建议 val 也走 DistributedSampler（drop_last=False + 去重）或在 Dataset 层分片。

C2. 周期 checkpoint 无保留策略
trainer.py:422-423 每 save_every 写 checkpoint_epoch_{N}.pth，从不清理。长训练 + 大模型会安静地吃满磁盘。建议 keep-last-k。

C3. _validate 中 compile 模型 shape 抖动（低优先级，需验证）
VolumeValEvaluator 复用 trainer.model（可能是 compiled 模块）做滑窗推理，滑窗尾窗/不同卷尺寸会触发 recompile；且 EMA swap（A2 的 key 问题修复后）每 epoch 换权重两次对 compile 缓存无影响但对 cudagraph 模式有。Windows 无 Triton 环境下未暴露，Linux 上开 compile + val_metric_mode=high 时值得冒烟验证。

D. 推理端（predictor）
D1. 概率累加器全程 GPU fp32，大卷 × 多类有 OOM 风险
sliding.py:87-91 / sliding.py:283-287：acc_pred 为 (num_fg, D, H, W) fp32 常驻 GPU。512×512×800 的 CT + 4 类 ≈ 3.1 GiB，加上 vol_t、batch 与模型激活，消费级卡易 OOM，且失败发生在推理后半程。nnU-Net 提供 half 累加/CPU 累加开关。建议：acc_pred 用 fp16 累加（blend 权重归一后精度足够）或加 accumulate_on_cpu 配置作逃生门。

D2. TTA 已实现且质量好（更正预期）
forwards.py 有完整 3D 7-flip / 2.5D 3-flip TTA 且支持变体批量化，无需补。唯一建议：flip TTA 与 B3 的旋转增强正交，若后续按 B3 收紧出面旋转，TTA 收益可能上升，值得在 run_aux_sweep.py 风格的消融里量化。

D3. prob_to_label 全类共享单一阈值
blending.py:111：max_prob < threshold → bg，所有前景类共用 threshold。one-vs-rest sigmoid 框架下（B7），不同类的最优操作点常差异很大（小结构类偏低阈值）。低成本改进：支持逐类阈值（配置为标量或列表），并可在 val 集上自动扫描。另外没有连通域后处理选项（nnU-Net 的 largest-CC 消融是常规免费增益），可作为可选后处理加入 predictor。


核对：

## A. 正确性问题核对

### A1. LR 调度 horizon 与梯度累积不匹配 —— ✅ 成立（确认为真 bug）
- `trainer/trainer.py:120-123`：`steps_per_epoch = len(train_loader)`，warmup/total 均按 micro-batch 数计。
- `trainer/trainer.py:705-741`：`self.scheduler.step()` 只在 `is_step_boundary`（优化步边界）调用。
- `WarmupScheduler.step()`（`trainer/optim.py:113-121`）按调用次数推进 → grad_accum=N 时整个训练只推进 horizon 的 1/N；warmup 实际持续 N× 个 epoch。
- 你的修复方案正确：`steps_per_epoch = ceil(len(train_loader) / accum)`。注意 `_train_epoch` 中 `is_step_boundary` 在 epoch 尾部（`(step+1)==total_steps`）也会触发一次优化步，所以每 epoch 的真实优化步数恰好是 `ceil(len/accum)`，与该修复严格一致。

### A2. EMA × torch.compile key 不匹配 —— ✅ 成立（比你判断的更严重一点）
- 顺序确认：`trainer.py:154` 用裸模型建 EMA → `trainer.py:171` `self.model = torch.compile(...)` → `trainer.py:742-743` `ema.update(self.model)`。
- `OptimizedModule.state_dict()` 的 key 带 `_orig_mod.` 前缀，`utils.py:55-59` `_build_pairs` 直接 `self.shadow[k]` → 首个优化步 KeyError。
- 补充：不只 update/apply_shadow/restore —— **best checkpoint 保存路径也会炸**：`trainer.py:882` `self.ema.apply_shadow(self.model)` 传入的是 compiled 模块（尽管同函数内 `unwrap_compile` 已用于 state_dict 提取）。
- 你"现有 Windows 环境 compile 回退 eager 故未触发"的推测与代码一致：`trainer.py:159-172` 无 Triton 时 `_compile_enabled=False`，`self.model` 保持裸模块。Linux + Triton 环境首步必炸。建议方案（key 归一化或 EMA 持有 unwrap 后引用 + compile+EMA 冒烟测试）合理，后者更干净。

### A3. ModelEMA 文档自相矛盾 —— ✅ 成立
- `utils.py:40` 注释"仅单卡，不兼容 DDP/FSDP"；但 Trainer 在 DDP 下无条件启用 EMA（`trainer.py:154` 无 DDP 判断）。
- 数学上你的分析正确：DDP 不包装 `self.model` 本体（`trainer.py:174-199` 注释明确参数张量与单卡一致，DDP 只挂 all-reduce 钩子），梯度同步后各 rank 参数一致，EMA 等价。确为"改注释而非改代码"级别的问题。FSDP 部分注释仍然成立（参数分片下确实不兼容）。

### A4. bf16/fp32 下非有限 loss 不被跳过 —— ✅ 成立（高优先级判断正确）
- `trainer.py:150-151`：scaler 仅 fp16 启用。`trainer.py:793-797` 的告警文案 "GradScaler will skip this optimizer step" 在 bf16/fp32 下是**错误陈述**：`GradScaler(enabled=False).step()` 直接透传 `optimizer.step()`，无 inf/NaN 检查。
- 后果链成立：NaN loss → NaN 梯度 → AdamW 一步后权重 NaN → 下一次 `ema.update` 污染 shadow，不可恢复。且 `resolve_auto_amp_dtype` 在 Ampere+ 默认解析为 bf16，即生产默认配置正处于无保护路径。你的修复建议（accum 组内出现非有限 loss 时 zero_grad 跳过该优化步，含 scheduler/EMA，并修文案）与 fp16 语义对齐，正确。

### A5. medium 验证 patch 每 epoch 随机重采 —— ✅ 成立
- `dataset.py:685-693` `_sample_z`：`is_train=False` 时走 `rng.integers(0, D_vol)` 均匀随机；cubic 的 `_sample_center`（923-941）同理。
- `dataset.py:469-483` per-worker RNG 种子来自 `info.seed`，而 DataLoader 每次 `__iter__` 从全局 torch RNG 抽新 base_seed，且 `loader.py:725-728` 的 val_loader 未传固定 `generator` → 每个 epoch 验证集 patch 确实不同。
- 后果链（save_best/early-stopping/plateau 被采样噪声驱动）成立：`trainer.py:352-378` 全部以该指标为准。建议方向（val 确定性采样或推荐 `val_metric_mode=high`）合理；`validation.py` 的 VolumeValEvaluator 实现确实完整可用。

### A6. one_cycle horizon 同族 bug —— ✅ 成立
- `optim.py:75` `total_steps = tc.epochs * steps_per_epoch`（micro-step 数）。修 A1 时传入的 `steps_per_epoch` 改为优化步数即一并修好；`optim.py:78-79` `pct_start` 下限 `2.0/max(total_steps,4)` 依赖 total_steps，你"修复时同步验证"的提醒是对的（total_steps 变小后下限变松，不会出错但 warmup 段占比会变）。

### A7. DDP 全 rank 同 seed —— ✅ 成立
- `train.py:183`：`_train_worker` 内所有 rank 用 `cfg.train.seed`。GPUAugmentor 全部用 `torch.rand(..., device)`（`augment.py` 各处）→ 各 rank CUDA RNG 流相同，每 step 增强参数序列完全一致；DataLoader worker numpy RNG 派生自 torch 种子，同理。
- 定性正确：非致命（数据不同）但增强多样性打折。`seed_everything(seed + local_rank)` 是标准做法（nnU-Net/timm 均如此）。resume 时 `trainer.py:934-947` 各 rank 恢复同一份 RNG 状态的补充观察也属实。

---

## B. 算法/架构层面核对

### B1. 缺失 spacing 归一化 —— ✅ 成立（同意"最大系统性短板"的定级）
- `make_data.py` 全文无 spacing/resample 处理；spacing 仅在 `dataset.py:95-114` `load_nifti_with_spacing` 被推理 z-interleave 读取一次。npz meta 有落点，你提出的"烘焙阶段插入 target-spacing 重采样 + Predictor 镜像回采"是侵入最小的正确位置。这也是 nnU-Net 数据指纹里权重最高的一环，同意优先级排在 B 组首位。

### B2. z_axis 面内整片 resize —— ✅ 成立
- `dataset.py:666-668` 每个 `__getitem__` 对 `(eD_max,H,W)` 做 scipy `zoom`（`resize_3d`，label order=0）；面内无随机 crop（z 是唯一随机自由度）。三条后果（细节上限锁死、面内平移多样性=0、CPU 热点重复计算）均与代码一致。"resize 前移到 make_data 烘焙或改面内原生分辨率+随机 crop"两个方案都成立，后者收益更大但要重估显存。

### B3. 空间增强物理正确性 —— ✅ 成立
- `augment.py:92-129` 仿射在 `affine_grid` 归一化坐标系做三轴欧拉旋转，无 aspect 校正；patch (64,128,128) 各向异性下旋转即混入剪切/非均匀缩放。`random_rotate_range` 三轴共用（`augment.py:109-110`）。elastic 的补充判断也对：`voxel_to_grid`（190-195）位移幅度逐轴换算正确，但粗网格 `cD/cH/cW = round(dim/sigma)`（184-186）的平滑尺度是体素意义的，各向异性下不均。建议（aspect 校正对角阵 + 三轴角度分开配置）是标准做法（batchgenerators 即如此）。

### B4. 前景采样不感知类别 —— ✅ 成立
- `make_data.py:56-75` `_compute_fg_indices`：`fg_mask = label != bg` 把所有前景类合并；`fg_coords` cap=50000 的均匀下采样进一步按体积比例分配 → 大器官统治采样。nnU-Net "先随机选类、再从该类 voxel 采中心"的对照准确。改动点（make_data 按类存 + 两个 `_sample_*`）评估正确，属低成本高收益。

### B5. Dice 默认配置 —— ✅ 成立
- `config.py:467-469`：`batch_dice=False`、`ignore_empty=False` 均为默认。分析正确（空 GT patch 的 per-sample Dice ≈1 抬高基线、稀释梯度）。代码已支持 batch_dice，确属纯配置/文档建议。

### B6. Plan A aux 头同源 —— ✅ 成立
- `unet.py:450-455`：非 hierarchical 时所有 aux 头 `feat_idx = n_dec - 1`，与主头（`unet.py:476` 读 `dec_features[-1]`）完全同源。hierarchical（Plan C）挂 `n_dec-1-k`（444-449）确实不同深度。"用 run_aux_sweep 消融验证 Plan A aux 是否有增益，无则简化"是合理的处置。

### B7. sigmoid vs softmax —— ✅ 成立（属设计权衡而非 bug）
- `losses.py:1-6` 明确全框架锁定逐类 sigmoid、背景隐含；推理端 `blending.py:108-111` 的 max/argmax 解决冲突。对互斥器官 softmax+CE+Dice 归纳偏置更强的判断符合社区共识（nnU-Net 默认 softmax，region-based 才用 sigmoid）。这是框架级改动，成本高，建议先在一个互斥多类任务上 AB 验证再决定。

### B8. 缺平移自由度 —— ✅ 成立
- `_build_rotation_matrices`（`augment.py:156-160`）平移列恒 `zeros`。cubic 模式随机 crop 提供平移、z_axis 模式面内完全没有的分析正确。仿射矩阵补随机平移确实是一行级改动。

### B9. 强度增强后无 clamp —— ✅ 成立（同意影响小）
- `augment.py:63-68` brightness/contrast/noise 后无重裁剪。gamma 内部有 `clamp(0,1)`（317）但之后 `*rng+mn` 反归一，与 brightness 叠加可产生越界值。低优先级判断合适。

### B10. blur/lowres 逐样本循环 —— ✅ 基本成立，一处细节修正
- `augment.py:353`、`augment.py:387` 逐选中样本 Python 循环属实，热路径隐性串行成立。
- **修正**：`torch.empty(1).uniform_()` 创建的是 **CPU 张量**，其 `float()` 不构成 CUDA→CPU 同步点（没有 GPU 参与）。真正的同步点是各函数里的 `mask.any()` / `mask.sum().item()` / `.nonzero().tolist()`（整个 augment 管线普遍存在）。结论（可向量化、优先级低于 B2）不变。

### B11. stem_stride 注释-实现漂移 —— ✅ 成立
- `unet.py:377` 注释称"stem_stride>1 时 forward 末尾上采回输入分辨率"，`unet.py:477-481` 实际尺寸不符直接 raise，全 forward 无上采样逻辑。`stem.py:82`：patchN stem 的 stride=N，配置上可达 stem_stride>1 → 注释确会误导。与 A3 同性质（文档漂移），改注释即可（或真正实现上采样，但需先确认是否有 patchN 使用场景）。

### B12. 优化器无参数分组 —— ✅ 成立
- `optim.py:24-26` 单参数组，norm affine 与 bias 均被 decay。零风险改进 + 为 ssltask layer-wise LR decay 预留入口的理由都成立。

### B13. EMA decay 无 warmup —— ✅ 成立
- `utils.py:42` 固定 decay=0.999。timm 式 `min(decay, (1+step)/(10+step))` ramp-up 建议标准。补充：`_load_pretrain`（`trainer.py:989-995`）已把 EMA shadow 对齐加载权重，所以 pretrain 场景问题不大，主要影响 from-scratch 且 `val_every` 较小的配置。

---

## C. 训练流程/工程核对

### C1. DDP medium 验证浪费 CPU —— ✅ 成立
- `validation.py:318-320`：每 rank 完整迭代 val_loader、`i % world_size != rank` 跳过，batch 已被 worker 完整生产（含 B2 的 resize）。val 走 DistributedSampler(drop_last=False)+去重 或 Dataset 层分片的建议正确；注意去重逻辑（DistributedSampler 会 padding 补齐）需要小心处理以免指标偏差。

### C2. 周期 checkpoint 无保留策略 —— ✅ 成立
- `trainer.py:422-423` + `_save_checkpoint`（904 行）：`checkpoint_epoch_{N}.pth` 从不清理。keep-last-k 建议标准。

### C3. compile 模型滑窗 shape 抖动 —— ✅ 方向成立，定性为"需验证"恰当
- `VolumeValEvaluator` 确实复用 `trainer.model`（可能 compiled）做滑窗；不同卷尺寸/尾窗会触发 recompile 属 torch.compile 已知行为。属 Linux+Triton+`val_metric_mode=high` 组合下值得冒烟验证的项，与 A2 可共用同一条 compile 冒烟测试。

---

## D. 推理端核对

### D1. 概率累加器 GPU fp32 OOM 风险 —— ✅ 成立
- `sliding.py:87-91`（z 轴）与 `sliding.py:283-287`（cubic）：`acc_pred (num_fg,D,H,W) fp32` + `acc_weight` 常驻 GPU；cubic 的 `acc_weight` 还是全尺寸 `(1,D,H,W)`。512³×800×4 类的量级估算正确。fp16 累加（blend 权重归一后精度足够）或 `accumulate_on_cpu` 逃生门都是 nnU-Net 已验证的方案。

### D2. TTA 已实现且质量好 —— ✅ 成立
- `forwards.py`：3D 7-flip / 2.5D 3-flip、`tta_batch_size` 变体批量化、2.5D 不翻 D 轴的理由（注释 52-53 行）都正确。无需补。

### D3. 全类共享单一阈值 —— ✅ 成立
- `blending.py:111`：`max_prob < threshold → bg`，单标量阈值。one-vs-rest 下逐类阈值 + val 集自动扫描是低成本改进；largest-CC 后处理选项确实缺失，均同意。

---

## 算法层面建议

**1. 先吃数据/训练侧的"免费增益"，再动架构。**
2024 年 nnU-Net 团队的系统性对照（*nnU-Net Revisited*, arXiv:2404.09556）结论很硬：绝大多数 2020-2024 的"新架构"（含多数 transformer/Mamba 变体）在同等训练预算下**打不过配置正确的纯 CNN U-Net**；宣称的提升多来自不公平的 baseline。对本仓库，B1（spacing 归一化）、B4（类感知采样）、B5（batch_dice）、A 组 bug 的期望收益大概率高于任何架构升级，建议先落地并建立可信消融基线。

**2. 编码器现代化：残差缩放 > 换范式。**
- **nnU-Net ResEnc (M/L/XL)**：残差 encoder + 按显存预算缩放深度/宽度，是上文对照中的最强配置。你的 `stage_builder` 注入机制天然支持，只需加残差 block preset 与更深的 stage 配比，成本低、证据强，**首推**。
- **MedNeXt（MICCAI 2023）**：3D ConvNeXt 风格 + 大核（3→5→7 kernel 上采样迁移训练），在多个 CT 任务上稳定超 nnU-Net baseline，是"大感受野"路线里证据最好的；与你的 multirf 思路同向，可作为 multirf block 的升级替换。
- **STU-Net（2023）**：结构与 nnU-Net 兼容、在 TotalSegmentator 上有预训练权重，可与 ssltask 的预训练路线互补（直接拿监督预训练权重微调）。

**3. 注意力升级：优先"工程正确"而非换算法。**
- 现有 `_SoftmaxQKVAttention`（`blocks.py:361`）是手写 O(N²) 实现，建议改走 `F.scaled_dot_product_attention`（自动用 FlashAttention/内存高效 kernel），同精度下显著省显存提速——这比换注意力公式收益更确定。
- 线性注意力（Shen 2021）之上，若追求全局上下文，**Mamba/SSM 系**（U-Mamba、SegMamba，2024）是当前热点且复杂度 O(N)，但复现口碑分化、且被 nnU-Net Revisited 点名质疑，建议只作小规模消融，不作主线。
- 3D 分割中注意力的实证甜点位是：**仅在 bottleneck 和最低两级分辨率放注意力**（N 小、O(N²) 可承受，softmax 注意力即可），高分辨率级保持纯卷积。你现有的 per-level 注入机制可直接表达该配置。
- CBAM/SE/ECA/CoordAttention 这类通道注意力（blocks.py 已备齐）在 3D 分割的期望增益 <0.5 Dice，不建议再投入。

**4. skip/解码器路线：UNet++ 之后不必追 UNet3+。**
UNet3+ 全尺度 skip 的独立复现证据弱。更值得试的是：a) skip 上的 Attention Gate（已实现，`skip_attention`）与 deep supervision（已实现）保持；b) 解码器上采样从 transpose 换 DySample/CARAFE（已实现，做一次系统消融定夺）；c) 若做 B1 spacing 归一化，记得按各向异性 spacing 配 `downsample_strides`（代码已支持各向异性 stride，这是 nnU-Net plan 的核心自由度，目前可能未被配置利用）。

**5. 损失/后处理侧。**
clDice 已实现（管状结构任务记得用）；可补 boundary-based loss（HD loss / boundary DoU）作为可选项；D3 的逐类阈值 + largest-CC 是推理端最便宜的两个增益点。
