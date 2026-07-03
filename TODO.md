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
本轮关注模型/算法架构的方向，例如unet是否有升级的地方，注意力是否有更多更好的算法，这些可能需要细致的调研现有的高质量论文/项目来借鉴（公认高质量自然图像，NLP，LLM论文，项目算法）。我说一下我之前怎么做的，先实现了unet，升级到unet++，升级加入multirf, qkv attn, linear attn等等这些。我希望你也类似朝算法方向给我一些高质量模型/算法提质的建议。  

有没有加速训练，减少训练/推理GPU占用高质量技巧，目前我只实现了grad_checkpointing, 多卡训练，混合精度训练等等。  

进展：  

一、现状盘点（已实现能力，作为建议的基线）
Decoder：UNet / UNet++ / UNet3+（@d:/codes/work-projects/SegTask/segtask_v1/models/factory.py:316-321），deep supervision、aux 头、attention gate。
Backbone：ResNet（post/pre-act/bottleneck）、ConvNeXt（drop-path + LayerScale）、MedNeXt 档位 A、ADM/EDM2 UNet。
注意力：通道类 SE/ECA/CBAM/Coord（@d:/codes/work-projects/SegTask/segtask_v1/models/blocks.py:309-331）；内容自注意力 softmax QKV（O(N²)）与 linear attention（Shen 2021），逐 stage 可配、zero-init 残差。
重采样：BlurPool / PixelShuffle+ICNR / CARAFE / DySample，nnU-Net 式各向异性 stride 调度。
训练技巧：grad checkpointing（逐 stage/level）、DDP、AMP（fp16+scaler / bf16）、EMA（timm 式 warmup + 热路径缓存）、梯度累积、torch.compile 配置项、cudnn.benchmark、no-decay 参数分组、非有限梯度守卫。
基线已经相当完善，下面的建议聚焦"还没做、且有公认高质量出处"的增量。

二、模型/算法架构升级建议（按性价比排序）
A1（高优先）：_SoftmaxQKVAttention 换用 SDPA / FlashAttention
@d:/codes/work-projects/SegTask/segtask_v1/models/blocks.py:361-377 手写 einsum + 物化 O(N²) 注意力矩阵。3D 瓶颈层 N=8³=512 尚可，但放到次深层（N=16³=4096）时 fp32 权重矩阵单头就要 64MB+。换 F.scaled_dot_product_attention（PyTorch 2.x 内置 FlashAttention/mem-efficient 后端）：

收益：显存 O(N²)→O(N)，速度 2–4×，数值上等价（且内部自动用 fp32 accumulate，可删掉手写的 softmax(.float()) 处理）；
成本：约 10 行改动，权重全兼容（QKV/proj 卷积不变）；
这是质量零损、速度显存双赢的第一优先项。
A2（高优先）：给自注意力块补齐 Transformer 化的两个缺口
当前 SelfAttentionBlock 是"纯 attention 残差"，对照公认 ViT/LLM 块设计缺两样：

位置信息：QKV 完全内容寻址，无任何位置编码。分割里空间先验重要，推荐 3D 分解式 RoPE（按 D/H/W 轴分配 head_dim，业界 LLM/ViT 公认做法，无参数、对分辨率外推友好）或退而求其次的 learnable 相对位置偏置（Swin 式）。
FFN 缺失：标准 Transformer 块 = Attn + FFN 成对。可加可配的 GEGLU/SwiGLU FFN（LLaMA 系公认），同样 zero-init 输出投影保持"初始恒等"的现有哲学。
A3（中优先）：窗口/网格注意力，把 attention 推到更浅层
全局 O(N²) 只能放最深两层。要在高分辨率层获益，推荐实现其一：

Swin 式 window + shift（3D 窗口 4³/8³），SwinUNETR 已验证于医学 3D；
MaxViT 式 block+grid 交替（局部窗口 + 稀疏全局网格），实现更简单、无 shift mask 麻烦，天然适配你的"逐 stage 可配"框架（新增 selfattn_type='window'|'grid'）。
配合 A1 的 SDPA，窗口注意力的每窗序列短，速度非常好。

A4（中优先）：大核深度卷积升级 —— MedNeXt 档位 B / 膨胀重参数
mednext.py 自己注明档位 B 未做：UpKern（k=3 权重插值初始化 k=5 大核）与重采样残差块，是 MedNeXt 论文报告增益的主要来源之一；
更进一步可借鉴 UniRepLKNet / RepLK 的 Dilated Reparam Block：训练期"大核 = 并行多支小核+膨胀核"，推理期重参数化合并为单一大核——与你已有的 MultiRFBlock（@d:/codes/work-projects/SegTask/segtask_v1/models/resnet.py:257）思想同源，但推理零开销，等于给 MultiRF 加一个"可折叠"档位。
A5（中优先）：块级微升级（低风险、逐项可消融）
Stochastic Depth 推广：drop-path 目前仅 ConvNeXt 有；给 ResNet/MedNeXt stage 也加线性递增 drop-path（He/Huang 2016，深 encoder 正则标配）。
GRN（ConvNeXt-V2, 2023）：在 MedNeXt/ConvNeXt 块 FFN 中加 Global Response Normalization，替代/叠加 LayerScale，V2 论文显示对小模型也有稳定增益，实现约 15 行。
AttentionGate3D 用 BatchNorm 的隐患：@d:/codes/work-projects/SegTask/segtask_v1/models/blocks.py:461-472 内部是 _BN，与全库 instance/group norm 及小 batch（3D 常 batch=2）设定相悖，小 batch 下 BN 统计噪声大。建议改为跟随 norm_type（保留 BN 为兼容默认或直接切 GN）。
A6（选做，方向性）：线性复杂度全局建模 —— Mamba/SSM
U-Mamba / SegMamba（2024）在 3D 分割上报告了超过 Transformer 基线的结果，O(N) 全局感受野。但：依赖 mamba-ssm+causal-conv1d（CUDA 扩展，Windows 编译困难），且训练稳定性不如 attention 成熟。建议列为观察项而非本轮实现；若要"线性全局"，先用你已有的 linear attention + A2 的 RoPE 组合压榨。

A7（算法层，非结构）：任务相关损失/后处理
clDice（CVPR 2021）：管状结构（血管/气道）拓扑保持损失，与你的 save_best_preset: vessel/airway 天然配套；
largest-CC 后处理：即 TODO 进展 P7 的确认项，仍未实现，建议与 clDice 同一轮补。
三、训练加速 / 降显存建议
B1（高优先，确定性收益）：DDP 梯度累积缺 no_sync()
_train_epoch 每个 micro-step 都直接 backward()（@d:/codes/work-projects/SegTask/segtask_v1/trainer/trainer.py:706），DDP 下每个 micro-batch 都做一次全量梯度 all-reduce。数学上等价（注释也承认），但 grad_accum_steps=k 时白白多付 (k−1)/k 的通信。标准做法：非边界步包 self.fwd_model.no_sync()。多卡+累积场景通信量直接除以 k。

B2（高优先，零风险三连）
TF32：全库未开。torch.set_float32_matmul_precision("high") + torch.backends.cudnn.allow_tf32=True，对 AMP 外残留的 fp32 matmul/conv（如 fp32 损失、验证）在 Ampere+ 上有免费加速；
fused AdamW：torch.optim.AdamW(..., fused=True)（CUDA 下），单 kernel 更新全部参数，参数多的 3D UNet 每步省数百次 kernel launch；
EMA foreach 化：ModelEMA.update 是 Python 逐张量循环（@d:/codes/work-projects/SegTask/segtask_v1/utils.py:80-85），换 torch._foreach_lerp_，一次调用完成全部 shadow 更新。
B3（中优先）：选择性激活检查点
目前 grad_checkpointing 是全 stage 一刀切。3D UNet 激活显存集中在最高分辨率的 1–2 层，只 checkpoint 浅层 stage（如新增 grad_ckpt_stages 掩码，风格与 selfattn_encoder_stages 一致）通常能拿到 ~70% 的省显存效果、只付 ~30% 的重算代价。

B4（中优先）：torch.compile 实用化
配置项已有但默认关。建议：(a) 文档/预设推荐 mode="default" + regional compile（只 compile stage 而非整模型，规避 2.5D/deep-supervision 动态形状导致的反复重编译）；(b) 落实进展 P4 提到的 compile+EMA 冒烟测试。PyTorch 2.7（你的 torch27_env）对此支持成熟。

B5（选做）
ZeroRedundancyOptimizer：DDP 下切分 optimizer state（AdamW 的 m/v 是 2× 参数量显存），一行包装；
channels_last_3d：PyTorch 2.x 对 3D conv 支持渐好，AMP 下可能有 10–20% 提速，但需 benchmark 验证，不保证正收益；
8-bit optimizer（bitsandbytes）：省 75% optimizer 显存，但引入重依赖 + Windows 支持差，不推荐现在做。



P1：one_cycle 的 pct_start 是死代码，内建 warmup 实际不可配置
@d:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:131-134 强制 one_cycle 必须 warmup_epochs=0，但：



optim.py:93-94
pct_start = tc.warmup_epochs / max(tc.epochs, 1)
pct_start = min(max(pct_start, 2.0 / max(total_steps, 4)), 0.9)
warmup_epochs 恒为 0 → pct_start 恒被夹到 2/total_steps（约 2 步），OneCycle 事实上没有 warmup，而报错文案却声称"OneCycleLR has built-in warmup (pct_start)"。二者自相矛盾：要么允许 warmup_epochs 在 one_cycle 下映射到 pct_start（推荐，删掉 trainer 的 raise），要么给 one_cycle 单独的 pct_start 配置项。

P2：A7 残留——resume 后各 rank RNG 流重新同一
原进展 A7 已点名"_load_checkpoint 会把 rank0 的 RNG 状态恢复到所有 rank"，此次未修：trainer.py:981-994 在每个 rank 上恢复同一份（rank0 的）RNG 状态，seed + local_rank 的解耦在 resume 后即失效。最小修法：非 rank0 恢复 RNG 后再做一次 manual_seed(restored ⊕ rank) 式偏移，或仅 rank0 恢复、其余 rank 重播种。

P3：A4 残留——"loss 有限但梯度非有限"在 bf16/fp32 下仍会污染权重
跳步守卫只看 loss（trainer.py:711）。bf16 反传中间溢出可产生 finite loss + NaN 梯度；此时开着 grad_clip_norm（默认 12），clip_grad_norm_ 返回的范数 gn 已经免费拿到（trainer.py:757），却没有用它做非有限检查。零成本改进：skip_optim_step |= not math.isfinite(gn)（scaler 未激活时）。原进展文本中"或检测到梯度非有限"这半句未落实。

P4：缺少针对本轮修复的回归测试
tests/ 中最新的 test_round2_fixes.py 覆盖的是旧轮次问题。本轮的关键修复均无测试，尤其：

A2：进展文本明确建议"补一条 compile+EMA 冒烟测试"，未见；
A1：scheduler horizon 与 accum 的对齐（很容易被后续重构再次破坏）；
A5：val patch 跨"epoch"确定性；C2 keep-last-k；B3 aspect 校正数值（如 90° 面内旋转在 64×128×128 上应精确置换）；D3 逐类阈值。
P5（备注级）：B9 clamp 范围比 nnU-Net 更激进
nnU-Net 仅在 contrast 后夹取；当前实现把 brightness/noise/gamma/blur 全部结果夹回增强前 min/max（augment.py:67-78）。对 minmax [0,1] 数据，brightness +0.1 的顶端 10% 动态范围会被削成饱和平台，部分抵消该增强本身。属于可辩护的取舍，但建议在注释或消融里确认无副作用。

P6（备注级）：B3 校正只含体素计数、不含物理 spacing
A=diag(W,H,D)（augment.py:148-149）只消除 grid 归一化坐标的形状各向异性；spacing_normalization=False 时体素本身的物理各向异性（层厚 vs 面内）仍会使出面旋转失真。开 B1 后此问题自然消解，建议在 random_affine_aspect_correct 注释中点明这一依赖关系。

P7（确认项）：D3 建议的 largest-CC 后处理未实现
全库无 largest_cc/连通域相关代码。原建议列为"可选"，若是有意延后请在 TODO 中记一笔，避免遗忘。


