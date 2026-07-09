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
gentask是生成/超分项目（基于segtask_v1改造）。


# TODO  
1 


2 分割项目代码审查：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程。

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  

进展：
分割进展：  
最高优先级（已亲自核验）
arch='adm' 与 arch='edm2' 分割模型构建即崩 ✅ P0（硬 bug） build_adm_seg_model(adm_unet.py:739) 和 build_edm2_seg_model(edm2_unet.py:676) 都传 out_channels=out_classes，但 ADMSegModel.__init__(adm_unet.py:483) 与 EDM2SegModel.__init__(edm2_unet.py:481) 只接受 num_fg_classes、根本没有 out_channels 形参 → TypeError 直接崩（且必填的 num_fg_classes 未传）。对应 tests/test_adm_edm2_seg_smoke.py 应当是失败/未跑状态。主 unet 路径不受影响，但这两个 arch 目前完全不可用。→ 改传 num_fg_classes=out_classes，并让主/DS/aux 头都用该通道数。

ADM/EDM2 主头输出 num_fg 而非 num_fg×D ✅（即使修了①仍存在） adm_unet.py:533/edm2_unet.py:520 头通道是 per-slice num_fg，但 2.5D topology 的 out_classes = num_fg×D(topology.py:106)。→ 头通道对齐 build_topology().out_classes。

UNet++ 的 attention-gate 参数接反 ✅ unet.py:237 是 attn_gate(skip, x_decoder)（用解码信号门控 skip，符合 Oktay），但 unetpp.py:110 是 gates[key](up, x[i][0])——用编码 skip 去门控解码特征，语义相反。skip_attention=True 下换 decoder_type 行为不一致（非崩溃，但影响分割质量与可比性）。→ 统一为门控 skip。

数据增强：label 的 grid_sample 用 padding_mode='border' ✅ P1 augment.py:274-275 image 与 label 共用 border（label 用 nearest 正确）。仿射/弹性/平移采样越界时会把边缘类别复制进外插区，而非背景，配合损失里 label==fg 精确判定会注入假前景。nnU-Net/MONAI 对 label 用零填充。→ label 用 zeros（背景）padding，image 保留 border/reflection。

z_axis 模式只对 D 过采样，H/W 无增强余量 ✅（dataset.py:731-733） 面内旋转/平移仍用 border 且无 post-crop 余量 → 面内边缘复制伪影残留（cubic 模式三轴都过采样，安全）。→ 面内也留余量，或 z_axis 下收紧面内几何增强幅度。

分区详述（保留要点，均带 file:line）
1) 数据读取（成熟，问题多为"共享继承"型）
[正确性] z_boundary_mode='stretch' 存了不用（训练恒 edge_pad，dataset.py:809；config 在 config.py:1175 自动升级为 edge_pad 兜底，但 dataset 参数是误导性死状态，且 tests/test_z_boundary_mode.py 相关期望已 stale）。cubic patch 无 padding 分支返回缓存卷视图（dataset.py:922-932，输出侧 ascontiguousarray 兜底但 helper 与 predictor 共享，脆弱）。resize_3d 早返回透传引用（dataset.py:427）同类缓存别名隐患。
[正确性] 混合训练 MixedBatchSampler 每 epoch 丢弃 coarse 尾样本（mixed_sampler.py:135 floor 除）；且与 DDP 显式不兼容（loader.py:618 抛错）。n=1 时 val 空（train 正常）。
[设计/优化] 三份独立 LRU（img/lbl/rw）→ 冷缓存每样本开 3-4 次 npz、RAM≈3×（dataset.py:578,630）。→ 合并 _load_npz_bundle 单开填三缓存。
[做得好] spec 策略分派、per-worker RNG 隔离、val 确定性 SeedSequence、Halton/等距 val 覆盖、ValBatchShardSampler、DistributedSampler.set_epoch、npz 不解码 peek、类均衡前景采样、稳健 NIfTI IO。
2) 模型构建
[正确性] 主线①②③ ✅；UNet3D.num_fg_classes 命名误导（实为含 z-fold 的 out_channels，unet.py:403）；window self-attn + RoPE 丢失跨窗全局偏移（blocks.py:838，各窗坐标从 0 起，纯窗内位置）。
[设计] ADM/EDM2 绕过 build_topology 自行重推通道（adm_unet.py:648），与 R5 单一真相源背离；nn.Sequential(stage, SelfAttentionBlock) 使梯度检查点粒度只能整 stage（factory.py:124）；self-attn/MultiRF/各向异性仅 decoder_type='unet' 支持（factory.py:366 其余抛错）；UNet++ 尺寸不匹配静默 interpolate 而非报错（unetpp.py:97，与经典 Decoder 的严格 RuntimeError 不一致）；三处线性注意力实现重复；EDM2 忽略 stem_mode（edm2_unet.py:233，与 gentask 同源问题）。
[优化] param_count() 漏统计 ds/aux/topo 头（unet.py:558）；decoder 无 per-stage 检查点掩码；瓶颈全量 softmax attn 仍可能（token cap 仅对 softmax）。
[做得好] ModelTopology 单推、2D/3D 统一 dispatch、各向异性 stride 端到端、self-attn flatten/unflatten 与 spatial_dims 无关 + SDPA + zero-init 残差 + config 侧 OOM/整除校验、MultiRF 强制 dilation=1 分支且 axes='hw' 避开薄 z、梯度检查点 use_reentrant=False+RNG 保留、DS/aux 头契约与 topology 对齐、构建期防御式通道校验、丰富构建日志。
3) 数据增强 / 处理
[正确性] 主线④⑤ ✅；float label + 精确等值二值化在几何增强后脆弱（losses.py:787，无 round 守卫，先修 padding）；simulate_lowres/gaussian_blur 只动 image 不动 label（多为有意，但与真实重采样不符）；make_data 空/越界 bbox 静默回退整卷读（make_data.py:141）；各向同性仿射 scale 忽略体素各向异性（spacing_normalization 默认关）。
[设计] 单体 GPUAugmentor.__call__ 组合性弱；CPU/GPU RNG 混用无 per-step Generator（严格复现受限）；make_data 前景子采样种子硬编码 42、与训练种子脱钩；强度增强幅度假定 minmax[0,1]。
[优化] grid_dropout 全 batch 生成再门控、simulate_lowres Python 分组循环、gaussian_blur 每步重建核、flip 逐轴多趟——均可向量化。
[做得好] 仿射+弹性融合单次 grid_sample（避免二次插值模糊）、image/label 插值模式正确分离、强度仅 image 且在空间变换后、CPU Bernoulli 掩码避免 CUDA 同步、align_corners=False 一致、原子 npz 写、训练/验证增强策略分离。
[借鉴] nnU-Net 的增强后前景保证/retry（本实现只在 patch 原点采前景，旋转/平移可能把小结构移出画面）；解剖学镜像约束（默认允许左右翻转）；B-spline 弹性形变。
4) 训练 / 验证
[正确性] ✅ 训练步 dice 日志用默认阈值 0.5（trainer.py:975），val 用 predict.threshold（validation.py:324）→ 阈值≠0.5 时训练/验证 dice 口径不一致（仅日志，不影响梯度）。✅ skipped optim step 仍 scheduler.step()（trainer.py:903 非有限跳步、931 GradScaler 跳步后均推进 LR，EMA 已正确跳过）→ 不稳定期 LR 与实际更新解耦。SWA 的 BN 重校准在 DDP 下每 rank 只见本 shard（仅用 BatchNorm 时有影响；默认 instance/group norm 则无碍）；EMA 验证不重校 BN。DDP+batch_dice+grad_accum 非位等价（已在代码注释声明为近似）。
[已核验正确] grad accum 尾批 _effective_accum 缩放、unscale→clip→step→update 顺序、非有限 DDP all_reduce_flag_any 一致跳步、val metric all-reduce 数学精确、bbox-crop + voxels_override 语义、ZeRO consolidate 在 rank 早退前、resume 载在线权重 + best 存 EMA 为主键 + relocate_optimizer_state + rank>0 RNG reseed。
[设计/做得好] pipeline 策略模式（Trainer 热路径零模式分支）、round-2 纯函数拆分（views/amp/breakdown/checkpoint/optim/dist_utils）、train.py PR_SET_PDEATHSIG+NCCL 异步错误、save-best 单一 criterion 派生、config 校验不兼容组合。
[借鉴] nnU-Net DS 权重按尺度衰减（当前等权/固定归一）；高模式 val 每次跑全 Predictor 很慢（可子集轮换/降频）；torch.compile 在 Triton+CUDA 时自动开。
5) 推理 / 采样
[正确性] GPU 面内/z resize 用 F.interpolate(trilinear)，训练用 scipy zoom(order=1)（inputs.py:89 vs dataset.py:815）→ 亚体素平滑差异（CPU 回退路径反而更贴训练）；3D 多分辨率推理只取 pred[:,:num_fg] 丢弃其余 view（forwards.py:279，训练监督了全部 view——若非有意应 ensemble）；stretch 推理分支与训练分叉（config 兜底后 OK）；legacy CPU 多分辨率路径（multi-res 且非 keep_native）对应的训练流已不存在 → 误配会跑没训过的路径（建议 Predictor.__init__ 直接拦）；skip_empty_windows 可能漏检低对比前景。
[做得好] ModelTopology 单一真相源、复用训练几何原语（extract_z/cubic/preprocess/resize）、VolumeValEvaluator 复用 Predictor、native multi-res/native_d 与 trainer 对称、2.5D TTA 不翻 D 轴、sigmoid + per-class 阈值无 softmax 误用、NIfTI affine/spacing/bbox 回贴正确、fp16 acc/CPU 累积等显存逃生门、批量 flip-TTA、诊断日志完善。
[借鉴] GPU/CPU 重采样器统一以彻底消除 train/deploy 亚体素差异；多 view 概率平均 ensemble；MONAI SlidingWindowInferer 的 sigma_scale 可配。

核验：  
最高优先级五条
你的结论	核验结果
① ADM/EDM2 构建即崩（out_channels 传参）
✅ 完全属实。adm_unet.py:739 / edm2_unet.py:676 传 out_channels=out_classes，而 ADMSegModel.__init__（adm_unet.py:467-492）与 EDM2SegModel.__init__（edm2_unet.py:461-486）只有必填的 num_fg_classes → TypeError。tests/test_adm_edm2_seg_smoke.py 存在但当前必挂。
② 主头输出 num_fg 而非 num_fg×D（"即使修了①仍存在"）
⚠️ 需要修正。头通道就是 num_fg_classes 形参本身（adm_unet.py:533-537、edm2_unet.py:520-521），而工厂里 out_classes = num_fg*D 已经算对了（adm_unet.py:652）。所以改传 num_fg_classes=out_classes 一步同时修好①和②，②不是修①后的独立残留 bug。真正的残留问题是另一个：ADM/EDM2 硬编码 num_fg*D 绕过 build_topology，导致 lift_2_5d_to_3d=True 时（topology 期望 out_classes=num_fg、spatial_dims=3，而 ADM/EDM2 硬编码 2D，adm_unet.py:494）通道与几何双重冲突，且 config 校验不拦这个组合。
③ UNet++ attention gate 接反
✅ 属实。AttentionGate3D.forward(x, g) 用 g 门控 x（blocks.py:1028-1035）；unet.py:239 是 attn_gate(skip, x)（门控 skip，符合 Oktay），unetpp.py:110 是 gates[key](up, x[i][0])（门控解码分支），语义相反。
④ label grid_sample 用 border 填充
✅ 属实。augment.py:274-275，label nearest 正确但 padding 是 border，外插区复制边缘类别。另注意 z 轴 edge_pad 抽取（dataset.py:878-895）同样会把边界切片的前景复制进 pad 区，是同性质的第二处。
⑤ z_axis 面内无增强余量
✅ 属实。dataset.py:731-733 只对 D 乘 oversample；cubic（dataset.py:980-981）/whole（1174-1176）三轴都有余量。config.py:1721-1737 的平移告警也没区分 z_axis 面内无余量的情况。
分区详述抽查（你标了 file:line 的项）
基本全部核验属实，仅列差异：

z_boundary_mode 死状态 ✅（dataset.py:746 存、809 恒 edge_pad；config.py:1175-1185 sync 兜底）。补充：launcher manifest 仍暴露 stretch 选项（launcher/manifest.py:54），绕过 sync() 直接构造 Predictor 时推理侧 stretch 分支（inputs.py:106-124）仍会与训练分叉——你的判断"config 兜底后 OK"成立但有旁路。
cubic 无 padding 返回缓存视图 ✅ 且比你写的更实：ascontiguousarray 对本就连续的切片不复制（dataset.py:922-932 vs 1057-1059），aug.inplace=True 时是真实的 LRU 污染路径；z_axis 侧有强制 .copy()（dataset.py:895）对比鲜明。
MixedBatchSampler 尾样本丢弃 + DDP 不兼容 ✅（mixed_sampler.py:135、loader.py:618-623）。
三份独立 LRU ✅（img/lbl/rw 各占条目，cache_max_volumes=1 时甚至可能只缓存 image）。
window RoPE 丢全局偏移 ✅ 且定位更准：_window_partition_tokens 已经算出了 meta["offsets"]（blocks.py:640-643），但 blocks.py:841 调 _apply_rope_nd 时没传 position_offsets——修复成本极低。
EDM2 忽略 stem_mode ✅（_build_edm2_stem 调用无 stem_mode 形参，edm2_unet.py:650-656，日志 696 却打印 mc.stem_mode）。
训练 dice 阈值 0.5 vs val 用 predict.threshold ✅（trainer.py:975 vs validation.py:318-324）。
跳步仍 scheduler.step() ✅（trainer.py:903/931），且代码注释表明是有意为之、EMA 两条路径都正确跳过（904-905、932-933）——定性为"取舍"而非 bug 更准确。
推理 GPU interpolate vs 训练 scipy zoom ✅（inputs.py:91-93 vs dataset.py:423-448，你写 inputs.py:89 差 2 行）；3D 多 view 只取 view0 ✅（forwards.py:235/279/304 三处 pred[:, :num_fg]）；legacy CPU 多分辨率死路径 ✅（sliding.py:149-157，对应训练流已不成立）。
losses float label 精确 == ✅（losses.py:787），现有 nearest 链路下 mostly 安全，属"缺 round 兜底"的建议级，与你判断一致。
二、你清单之外的新发现
审查还发现了几条你的进展里没有的：

正确性（值得进你的 P1/P2）

验证与部署的二值化语义不一致：high 模式 val 逐通道 prob > thr 独立二值（validation.py:455-459），而部署 prob_to_label 是 argmax 赢者通吃再阈值（blending.py:110-122）——多类都超阈时选模指标与最终 NIfTI 标签会分裂。
adm_attention_levels 默认行为与注释相反：config 注释说"空=最深两级"（config.py:491-492），但 default_factory=list 给 []，而 [] 走"不加注意力"分支（adm_unet.py:235-238）——默认配置实际无注意力。
deep_supervision=True + deep_supervision_weights=[]：空列表为 falsy，pipeline 不包 DS loss，但模型 forward 返回 list，运行期才炸（config.py:1769-1777 只在非空时预警）。
SWA BN 重校准 DDP 下只见本 rank shard（trainer.py:1104-1118）——你已列在 P3，核验属实且是确定问题（仅 BatchNorm 时）。
设计/健壮性（P3 级）

ADM 用 assert 拦 patch_mode，python -O 下失效（adm_unet.py:649）；EDM2 静默忽略 aux_head_mode（edm2_unet.py:549）；cache_mode 拼错静默禁用缓存、samples_per_volume/val_ratio 无下界校验；build_dataloaders 副作用回写 cfg.data；test_model_topology.py 文件头承诺的 build_model 一致性测试实际不存在。
三、核验总结论
你的进展质量很高：五条主线中四条完全属实、file:line 精确；分区详述抽查无一虚报。唯一需要修正的是 主线②——它与①是同一处修复（num_fg_classes=out_classes），不是独立残留；其位置上真正的独立问题是"ADM/EDM2 绕过 topology + 不拦 lift 组合"（你在分区详述里其实已经写到了 adm_unet.py:648 绕过 topology，只是没和②合并归因）。另外"跳步仍推 scheduler"建议从 bug 降级为"已文档化的取舍，可加开关"。

修订后的优先级建议（相对你的表只有三处变化）：

P0 两条合并为一条（传参修复即同时解决通道数）；
新增 P1：val/部署二值化语义不一致；ADM/EDM2+lift 组合校验缺失（含 adm_attention_levels 默认注释反）；
P2 追加：DS 空权重列表晚失败。
五份分区完整报告（各带 file:line 与做得好/借鉴清单）分别来自 数据读取、模型构建、数据增强、训练验证、推理流程。


segtask_v1 数据读取代码审查报告
审查范围：dataset.py、loader.py、specs.py、mixed_sampler.py、config.py（数据相关）、make_data.py（读取接口约定）。以下为按类别整理的发现，每条附 file:line 证据。

[正确性]
z_boundary_mode 在训练 dataset 中不生效（恒为 edge_pad）
SegDataset3D 保存了 self.z_boundary_mode（dataset.py:746），但 __getitem__ 路径始终调用 _extract_z_patch_padded（dataset.py:809-810），self.z_boundary_mode 在抽取逻辑中从未被读取。stretch 分支（_extract_z_single(..., use_padded=False)，dataset.py:866-872）在生产路径中不可达。
缓解：Config.sync() 会把 stretch 自动升级为 edge_pad 并告警（config.py:1175-1185），正常训练入口下 train/infer 几何一致。
残留风险：绕过 sync() 直接构造 SegDataset3D(z_boundary_mode="stretch") 时，配置名与行为不符。

Cubic 无 padding 时 patch 可能返回缓存卷视图（别名风险）
_extract_cubic_patch 在无越界时直接 return patch（dataset.py:922-932），patch 是 vol[...] 切片视图。SegDataset3DCubic 对 label 仅做 np.ascontiguousarray(lbl_s[None])（dataset.py:1057-1059）；对已是 C-contiguous 的切片，ascontiguousarray 不复制，torch.from_numpy 可与 _lbl_cache 共享内存。
对比：z_axis 的 extract_z_patch_padded 末尾强制 .copy()（dataset.py:895）；cubic 注释（dataset.py:1054-1056）假设总会复制，与无 padding 路径不一致。
默认 aug.inplace=False 时不易触发；aug.inplace=True 时存在污染 LRU 缓存的潜在路径。

MixedBatchSampler 每 epoch 丢弃粗标尾样本
__len__ = n_secondary // coarse_per_batch（mixed_sampler.py:135），n_secondary % coarse_per_batch 个粗标样本每 epoch 不参与训练。测试明确接受此行为（tests/test_mixed_sampler.py:78-80）。属设计性样本损失，非崩溃 bug，但粗标利用率 <100%。

DDP 训练 drop_last=True 丢弃各 rank 尾 batch
loader.py:772,779 使用 DistributedSampler(..., drop_last=True) + DataLoader(..., drop_last=True)，每 rank 每 epoch 会丢少量金标准 patch。DDP 常规做法，但小数据集 + 多卡时丢样比例可观。

n=1 时 val 集为空
train_val_split 在 n=1 时 n_val=0（loader.py:403-406），仅告警不报错；MetricAccumulator.compute 对空累加器返回退化指标（validation.py:209-212）。不崩溃，但 mean_dice=0、val_loss=nan 可能误导选模。

[设计]
z_boundary_mode 职责分裂：Config 宣称废弃 stretch，Dataset 仍保留无效字段

Config 默认 edge_pad，注释写明训练侧从未用 stretch（config.py:123-125）
ZCubeSpec 仍把 dc.z_boundary_mode 传入 Dataset（specs.py:203）
Dataset __init__ 仍校验 stretch|edge_pad（dataset.py:726-729）
launcher manifest 仍暴露 stretch 选项（launcher/manifest.py:54）
单一真相源不清晰：配置层、dataset 层、predictor 层语义不一致。
DatasetSpec 是良好的 train/val 差异收敛点
specs.py:123-136 集中处理 aug_oversample、samples_per_volume（val 减半）、foreground_oversample_ratio（val=0）；WholeSpec val 强制 samples_per_volume=1（specs.py:165-166），避免 whole 模式 val 重复白算。loader.py 不再散落 train/val kwargs。

npz-only 路径与 make_data 接口契约清晰
dataset.py:221-231 文档化 npz 键语义；make_data.py:204-237 写入 meta.image_shape、label_counts、逐类 fg 索引；loader.py:671-677 用 load_npz_label_counts 快路避免全量解码 label。职责分离合理。

双源混合与 DDP 互斥是显式设计
loader.py:618-623 在 world_size>1 + npz_dir_secondary 时硬拒绝，因 MixedBatchSampler 无 rank 感知（mixed_sampler.py:85-175）。错误信息明确，比静默错误好。

build_dataloaders 副作用修改 cfg.data
loader.py:675-689 自动探测并写回 label_values、num_classes，再 cfg.sync()。方便但 Config 对象在数据构建后可能被 mutate，不利于复现/日志快照。

cache_mode 无 Config.validate 校验
仅 launcher manifest 限制 none|memory（launcher/manifest.py:53）；specs.py:69 用 == "memory" 判断。拼写错误会静默禁用内存缓存，无告警。

samples_per_volume / val_ratio 缺少下界校验
config.py _validate_data 未约束 samples_per_volume >= 1、0 < val_ratio < 1；samples_per_volume=0 会导致 __len__=0（dataset.py:670-671）。

make_data 模块 doc 与实现不符
make_data.py:1 写「训练时 mmap 多 worker 共享」；dataset.py:231 明确「numpy 对 .npz 忽略 mmap_mode，逐 worker 为 owned ndarray」。接口文档易误导性能预期。

[优化]
同卷 cache miss 时多次打开 npz
冷启动单 worker 对同一卷可能：npz_has_rw（dataset.py:293-294）、load_npz_image（dataset.py:266-267）、load_npz_label（dataset.py:276-277）、load_npz_region_weight（dataset.py:282-285）各开一次 zip。可合并为单次 _open_npz + 多键读取（类似 _build_index，dataset.py:767-771）。

LRU 按「路径」计条目，不按字节
VolumeCache（dataset.py:479-507）max_volumes=1（Config 默认，config.py:178）只缓存 1 个路径，但 image/label/rw 各占 1 条（dataset.py:578-580），同一卷 3 个 key。cache_max_volumes=1 时可能只缓存 image，label 仍反复 IO。

内存估算假设所有卷同尺寸
loader.py:822-824 仅用首个 npz 的 image.shape 估算；bbox 裁剪后体素数差异大时会低估/高估（loader.py:842-850）。

_build_index 启动期 N 次 npz 打开
SegDataset3D._build_index（dataset.py:766-771）、SegDataset3DCubic._build_index（dataset.py:1005-1012）逐卷 with _open_npz；大 cohort 启动慢。可考虑惰性加载或一次性 meta 索引文件。

resize_3d 形状已匹配时返回原数组引用
dataset.py:427-428,432-433 直接 return arr。Whole 模式（dataset.py:1189-1191）若 extract_size == vol.shape，img_r 与缓存卷别名；后续若 in-place 操作会污染缓存（概率低）。

DDP num_workers 平摊策略
loader.py:589-597,629-642 按 world_size 平摊 worker，控制全机 LRU RAM 与单卡基线一致。是务实的多卡内存治理。

[做得好]
逐 worker 独立 numpy RNG，避免 fork 后重复采样
dataset.py:592-606 用 torch.utils.data.get_worker_info().seed 惰性创建 np.random.Generator，解决 DataLoader fork 后全局 RNG 复制问题。

验证采样 worker 无关的确定性
_sample_rng val 分支用 (_VAL_SAMPLING_SEED, self._sample_idx)（dataset.py:608-617），不依赖 worker RNG；配合 shuffle=False（loader.py:803）保证 epoch 间同一 idx 同一 patch，利于 save_best/early-stop。

val_grid_coverage 实现扎实
z_axis：卷内等距 z bin 中心（dataset.py:837-841）；cubic：Halton(2,3,5) 低差异 3D 铺点（dataset.py:1099-1106）。比纯随机 val 更稳定。

z_axis patch 抽取防别名
extract_z_patch_padded 末尾 .copy()（dataset.py:895）；load_nifti_cropped 从 sitk view 显式 copy（dataset.py:208-210）。

VolumeCache.__getstate__ 清空 store
dataset.py:515-518 Windows spawn 传 worker 时不序列化缓存，避免管道超限且无意义的跨进程缓存传输。

ValBatchShardSampler DDP val 切分
loader.py:37-67 按 batch 块切分、无 padding/重复；validation.py:340-342 注释与实现一致，避免每 rank 跑全集 val。

NIfTI 读取重试 + OOM 识别
dataset.py:66-88 区分瞬态 IO 与 bad_alloc，OOM 不重试直接 MemoryError。

免解码读 npz 形状
_read_npz_image_shape（dataset.py:239-255）优先 meta.image_shape，旧包回退解析 .npy 头，避免仅为形状解压整卷。

前景过采样类均衡
dataset.py:847-850,1113-1115 有 fg_*_cls 时先均匀选类再选点，缓解大器官挤压稀有类（nnU-Net 思路）。

preprocess_image 单次分配 + in-place clip
dataset.py:360-384 减少强度归一化临时 buffer。

分层划分复用 label_counts meta
loader.py:693-704 避免启动期二次全量扫 label。

MixedBatchSampler 可复现且 epoch 间不同
mixed_sampler.py:159-161 seed + _epoch；tests/test_mixed_sampler.py:94-109 有单测覆盖。

[借鉴]（nnU-Net / MONAI 等可对照点）
nnU-Net 式前景过采样 + 类均衡
预计算 fg_slices/fg_coords（make_data.py:64-106）+ 训练时 foreground_oversample_ratio（dataset.py:833-852），与 nnU-Net 预计算前景位置再 stochastic crop 一致。比 MONAI PosNegLabelCropd 运行时 argwhere 更省 IO。

nnU-Net 式 pooled 验证指标 + 确定性 val patch
val 用固定种子派生 RNG（dataset.py:26-27,617）+ 可选网格覆盖（config.py:167-171），对齐 nnU-Net「固定验证 crop 集合」思路，减少选模噪声。

nnU-Net 式 spacing 指纹
make_data.py:303-324 扫描头信息取逐轴中位数作 target_spacing；烘焙进 npz（make_data.py:219-222），训练侧零成本读取。符合 nnU-Net 预处理烘焙哲学。

MONAI CacheDataset / LMDB 对照
当前 VolumeCache 是进程内 LRU（dataset.py:479-507），worker 间不共享；MONAI CacheDataset 或 nnU-Net npz 烘焙 + OS page cache 是更成熟的「跨 worker 共享」方案。本项目已走 npz 烘焙路线（loader.py:605-608），cache_mode=none 依赖 OS page cache 是合理备选（loader.py:864-865）。

MONAI DistributedSampler + partition_weights 对照
双源混合用自定义 MixedBatchSampler（mixed_sampler.py:85）而非 PyTorch 内置；DDP 直接禁用（loader.py:618-623）。业界常见做法是 rank-aware 混合 sampler 或每源独立 DataLoader + zip，可参考 MONAI 多数据源训练模式扩展 DDP 支持。

Halton 序列做 val 空间覆盖
dataset.py:30-38,1103-1106 比均匀随机 grid 覆盖更均匀，是低差异采样在医学影像 val 中的实用做法。

intensity clamp after aug
config.py:200-202 注释引用 nnU-Net 惯例；数据读取侧通过 preprocess_image 窗宽剪枝（dataset.py:367）与增强侧 clamp 分工清晰。

专题结论摘要
专题	结论
z_boundary_mode
训练 dataset 不读该字段，恒 edge_pad；Config sync() 把 stretch 升级掉。非 bug（有缓解），但是死配置 + 文档/测试部分过时（tests/test_z_boundary_mode.py 仍期望 stretch 行为）。
cubic/z_axis 缓存别名
z_axis 安全（强制 copy）；cubic 无 padding 时有 label 别名风险（dataset.py:932 vs 1059）。
LRU
按路径条目、pickle 清空、DDP worker 平摊；不按字节、三缓存独立计数。
MixedBatchSampler + DDP
显式不支持；粗标尾样本丢弃；金标准循环过采样。
n=1
val 空集，告警 + 退化指标，不崩溃。
val 确定性
实现优秀：idx 驱动种子 + 可选 grid；与 worker 数无关。


segtask_v1 模型构建代码审查报告
审查范围：segtask_v1/models/ 全部 12 个实现文件、config.py 模型相关配置/校验、tests/ 模型相关测试。以下为按维度分类的结构化清单（每条附 file:line，已逐行核对）。

测试状态说明
tests/test_adm_edm2_seg_smoke.py 存在（D:\codes\work-projects\SegTask\tests\test_adm_edm2_seg_smoke.py），覆盖 ADM/EDM2 构建、eval/train forward、aux/DS 形状与 backward。
当前代码下应无法通过：build_adm_seg_model / build_edm2_seg_model 传入 out_channels=...，而 ADMSegModel / EDM2SegModel.__init__ 要求 num_fg_classes（见下方 [正确性] #1）。本环境未能成功执行 pytest，但静态分析可确定首条用例会在 build_model(cfg) 处 TypeError。
其余模型测试较全：test_model_topology.py、test_unetpp.py、test_grad_checkpointing.py、test_attention.py、test_selfattn.py 等。
[正确性] 确定 Bug
ADM 构建即崩：工厂 kwarg 与 __init__ 不匹配
build_adm_seg_model 传 out_channels=out_classes（adm_unet.py:739），但 ADMSegModel.__init__ 签名只有 num_fg_classes、无 out_channels（adm_unet.py:483）。运行 build_model(cfg) 会 TypeError: unexpected keyword argument 'out_channels'，且缺少必填 num_fg_classes。

EDM2 构建即崩：同上
build_edm2_seg_model 传 out_channels=out_classes（edm2_unet.py:676），EDM2SegModel.__init__ 要求 num_fg_classes（edm2_unet.py:481）。与 ADM 为同一类未完成重构（对比 gentask/models/adm_unet.py 已用 num_fg_classes=out_classes）。

主头通道语义：修复 #1/#2 后应传 num_fg_classes=out_classes（即 num_fg×D）
2.5D 折叠模式下 topology.out_classes = num_fg * D（topology.py:106）；UNet3D 主头用 out_channels=topo.out_classes（factory.py:519）。ADM/EDM2 的 seg_head 第二参数即输出通道数（adm_unet.py:533-536、edm2_unet.py:520-521），修复后应传 num_fg*D，不能传裸 num_fg。

UNet++ attention gate 参数顺序与 unet.py/Oktay 原文相反
AttentionGate3D.forward(x, g) 用 g 门控 x（blocks.py:1028-1035）。DecoderLevel 正确：attn_gate(skip, x)，门控 skip（unet.py:237-239）。UNetPPDecoder 写 gates[key](up, x[i][0])（unetpp.py:109-110），门控的是上采样 decoder 分支，encoder skip 作 gating signal，与 Oktay 2018（门控 skip）及 unet.py 不一致。

窗口 self-attention + RoPE 每窗从 0 起算，丢失全局位置
_window_partition_tokens 计算了 offsets（blocks.py:640-643），但 _WindowQKVAttention 调用 _apply_rope_nd(q, k, meta["token_sizes"]) 未传 position_offsets（blocks.py:838-841），RoPE 坐标恒为 0..window_size-1。全图 softmax+RoPE 用完整 spatial_shape（blocks.py:800-803），仅 window 模式丢全局位置。

ADM/EDM2 硬编码 out_classes=num_fg*D，未走 build_topology，与 lift 模式冲突
build_adm_seg_model/build_edm2_seg_model 均 out_classes = num_fg * D（adm_unet.py:652、edm2_unet.py:609），而 lift_2_5d_to_3d=True 时 topology 为 out_classes=num_fg（topology.py:116）。config.validate 不禁止 arch=adm/edm2 + lift（config.py:1880-1901），组合下通道契约错误。

arch=adm/edm2 + lift_2_5d_to_3d：2D 模型 vs 3D 数据几何
ADMSegModel.spatial_dims = 2 硬编码（adm_unet.py:494），全链路 Conv2d；lift 后 topology.spatial_dims=3（topology.py:114-116）。配置可通过校验但模型与数据管线几何不匹配。

[正确性] 建议关注（非构建即崩，但易埋雷）
ADM/EDM2 decoder skip 尺寸不匹配时静默 interpolate
adm_unet.py:445-449、edm2_unet.py:421-424 对 skip 做 bilinear 对齐；DecoderLevel 在同类情况 RuntimeError（unet.py:231-235）。静默插值可能掩盖 stride/配置错误。

UNet++ 上采样尺寸不匹配时 warn + interpolate
unetpp.py:97-108 与 unet.py:231-235 严格报错策略不一致，同类问题表现不同。

EDM2 忽略 aux_head_mode
文档写「'linear' only」（edm2_unet.py:484），但 aux 固定 _MPSegHead（edm2_unet.py:549），cfg.model.aux_head_mode='conv' 被静默忽略；ADM 走 build_head（adm_unet.py:570-579）。

ADM 用 assert 而非 raise ValueError
assert cfg.data.patch_mode == "2_5d"（adm_unet.py:649），python -O 下断言被剥离，错误配置可能拖到更深处才失败。

[设计]
ADM/EDM2 绕过 ModelTopology 单一真相源
factory.build_model 对 unet 用 build_topology(cfg)（factory.py:334-339）；ADM/EDM2 自行算 D、n_views、in_channels、out_classes、aux_head_out_channels（adm_unet.py:648-706、edm2_unet.py:606-647）。新增 patch_mode/flag 时易与 topology 漂移。

decoder_type 支持矩阵不完整
MultiRF/self-attn 仅 decoder_type=='unet' 生效（factory.py:366-377）；各向异性下采样拒 unetpp/unet3p（factory.py:424-428）。unetpp/unet3p 功能子集未在 config 层集中文档化。

梯度检查点粒度不对称
Encoder 支持 grad_ckpt_encoder_stages 逐 stage 掩码（unet.py:127-136、config.py:1468-1480）；decoder 仅全局 grad_checkpointing（unet.py:314-315、unetpp.py:112-113），无 decoder stage 掩码。

线性注意力两套实现
blocks.py:_LinearQKVAttention（blocks.py:880-902）与 adm_unet.py:_LinearAttention（adm_unet.py:149-179）算法相近、接口不同，维护成本高。

num_fg_classes 命名在 ADM/EDM2 中误导
ADMSegModel 参数名 num_fg_classes 实际表示主头总输出通道（应为 out_channels，与 UNet3D 对齐，unet.py:385-404），加剧 #1 类重构遗漏。

config 注释与代码默认行为不一致
adm_attention_levels 注释「空=默认最深两级」（config.py:491-492），但 default_factory=list 得 []，_resolve_attention_levels 仅在 None 时用默认（adm_unet.py:235-238）。实际默认是无注意力，非最深两级。

decoder_blocks_per_stage 校验长度与 ADM/EDM2 实际拓扑不一致
config 要求 len == n_levels-1（config.py:1460-1464）；ADM/EDM2 内部用 dec_bps_full = enc_bps（长度 n_levels，adm_unet.py:672），用户配置被忽略仅 warning（adm_unet.py:664-671），校验与运行语义脱节。

test_model_topology.py 文档承诺未实现
文件头声明测 build_model 与 topology 一致（test_model_topology.py:10），实际无对应用例；ADM/EDM2 通道回归无单测保护。

[优化]
param_count() 不完整
UNet3D 只统计 enc/dec/seg_head（unet.py:558-563），不含 ds_heads/aux_heads/topo_head。ADM 缺 ds/aux（adm_unet.py:622-631）；EDM2 缺 ds/aux（edm2_unet.py:585-591）。日志「total」正确，分项低估 head 占比。

RoPE 已算 offsets 未用于 window 路径
blocks.py:671-678 的 meta["offsets"] 在 blocks.py:841 未传入 _apply_rope_nd，修复 #5 时可直接复用。

ADM/EDM2 不支持 grad_checkpointing
config.grad_checkpointing 对 ADM/EDM2 无 wiring；大模型 2.5D 显存优化空间未利用。

config._validate_model 不校验 adm_*/edm2_* 索引范围
越界仅在 build_*_seg_model 抛 ValueError（adm_unet.py:241-243），宜前移到 validate() 与 unet 的 selfattn 校验对齐（config.py:1501-1502）。

[做得好]
ModelTopology + build_topology 单一真相源（unet 路径）
冻结 dataclass、决策树注释清晰（topology.py:31-161），factory.build_model 读 topo 全字段（factory.py:334-339），消除历史双处推导。

各向异性下采样构造期硬失败
ConvNeXt LN-first、hierarchical stem、unetpp/unet3p、down/up mode 不兼容时构造即 ValueError（factory.py:418-445），避免 forward 才爆。

checkpoint_if 实现规范
use_reentrant=False + preserve_rng_state=True + grad 门控（blocks.py:49-64），兼顾 DropPath 与 DecoderLevel 双输入。

ADM/EDM2 skip-stack 拓扑与论文对齐
encoder 逐块 push skip、decoder pop 融合（adm_unet.py:309-327、352-426），skip_stack 不平衡时 RuntimeError（adm_unet.py:423-426），forward 可诊断。

Aux/DS forward 合同与 UNet3D 统一
train+aux 返 {"main","aux"}、eval 不返 dict（adm_unet.py:584-620、edm2_unet.py:551-583），与 unet.py:503-556 及 smoke test 契约一致。

_StatefulStageBuilder 消除双计数器
factory.py:44-63 单索引维护 stage 序号，MultiRF/self-attn 与 block 数对齐可靠。

EDM2 magnitude-preserving 原语忠实
_normalize、_mp_cat、_mp_sum、训练期 weight renorm（edm2_unet.py:24-89）与 Karras 2024 设计一致。

[借鉴] 可吸收的业界做法
nnU-Net 式各向异性 stride 调度
_auto_anisotropic_strides + _MIN_FEATURE_SIZE=4（factory.py:242-285）是医学 3D 薄 z 轴场景的标准工程化方案。

Attention U-Net gate 的 norm 可配 + auto 跟随全局 norm
attn_gate_norm='auto'（factory.py:467-469）避免 3D 小 batch 下 gate 用 BN 的不稳定——与 MONAI/nnU-Net 社区经验一致。

SelfAttention 零初始化输出投影 + 可选 FFN
SelfAttentionBlock 的 zero_init proj（blocks.py:968-970）与 ViT/Swin 式「注意力块初始近似恒等」一致，利于深层叠加。

深监督头用 ConvSegmentationHead、主头用 1×1
unet.py:411-428 区分主/辅头容量，对齐 nnU-Net 深监督实践。

构造期拒绝未实现的 stem 组合
ADM/EDM2 显式拒 hierarchical（adm_unet.py:690-694、edm2_unet.py:634-637），比 runtime 静默错误更可维护。

RoPE cos/sin LRU 缓存
blocks.py:33-34、_rope_cache_key（blocks.py:499-515）对滑窗/多分辨率推理友好，可参考 LLM/ViT 推理优化套路。

优先修复建议（仅指导，Ask 模式不改代码）
优先级	项	动作
P0
#1 #2
out_channels → num_fg_classes=out_classes（值仍为 num_fg*D）
P1
#4
unetpp.py:110 改为 x[i][0] = self.gates[key](x[i][0], up) 或等效，与 unet.py:239 对齐
P1
#5
blocks.py:841 传入 position_offsets=meta["offsets"][batch_idx] 或 per-window offset
P2
#6 #12
ADM/EDM2 改为 topo = build_topology(cfg) 驱动 in/out/aux 通道
P2
#17
统一 config 注释与 _resolve_attention_levels(None) 默认策略


segtask_v1 数据增强/处理 — 代码审查报告
审查范围：augment.py、dataset.py（过采样/抽取）、make_data.py、losses.py（label 二值化）。以下为只读结论，附 file:line。

[正确性]
确定问题 / 已文档化但会污染训练分布

label 的 grid_sample 使用 padding_mode='border'，外插区复制边缘体素而非 0 — 旋转/平移/弹性时，若 patch 边缘含前景，border 会把边缘类别“拉进”采样域，产生假前景条带；与 nnU-Net/MONAI 对 mask 用零填充的惯例不符。image 用 bilinear+border 合理，label 应 nearest+zeros。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:274-275（同函数注释 :211 亦写明三路均为 border）

z_axis 模式仅 D 轴过采样，H/W 无余量 — extract_size = (round(pD*ratio), pH, pW)，面内尺寸等于最终 patch，增强后 center_crop 只能裁 D，无法消掉 H/W 方向 border 伪影。
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:731-733
项目已在配置中承认该限制：D:\codes\work-projects\SegTask\segtask_v1\config.py:228-229

z_axis + 面内仿射/平移 → 伪影必留 patch 内（与 #2 联动） — 训练流为 augment → center_crop → forward；H/W 无 oversample 时，面内旋转/平移引入的 border 复制无法被裁掉。
D:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:787-791 + dataset.py:733

cubic / whole 三轴均有 oversample 余量（对比 z_axis 正确） — cubic：extract_size 三轴乘 ratio；whole 同理。
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:980-981
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:1174-1176

建议级（非必现 bug，但有风险）

loss 对 float label 用精确 == 二值化，增强后无 round — 训练侧 .float() 后走 nearest 插值，0/1/2 等 int 标签在 fp32 下通常仍精确；但若链路中出现非 nearest 或更大标签值，等值判定会变脆。dataset 侧 preprocess_label 有 np.round，loss 侧没有对称兜底。
D:\codes\work-projects\SegTask\segtask_v1\trainer\trainer.py:780
D:\codes\work-projects\SegTask\segtask_v1\losses\losses.py:780-787（2.5D 同模式 :869、:943）
对比：D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:411-417

仿射 scale 为各向同性单标量，未用真实 spacing — scales 为 (na,1) 统一缩放；aspect_correct 只用体素数比例 diag(W,H,D)，注释明确“不代替真实 spacing 校正”。make_data 可选 spacing 归一化，但增强侧不读 spacing。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:242-243,253-259,128-130
D:\codes\work-projects\SegTask\segtask_v1\data\make_data.py:177-190

配置称弹性为 “B-spline”，实现为 coarse randn + trilinear 平滑 — 行为接近 MONAI 的 Gaussian-smoothed field，但不是 B-spline control-point 形变。
D:\codes\work-projects\SegTask\segtask_v1\config.py:232-236
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:170-186

z 轴抽取 edge pad 会在 label 边界复制类别 — 与 grid_sample border 不同层，但 z 越界时 label 同样复制边界值（注释认为对 label 安全）；若边界切片含前景，pad 区也是假前景。
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:878-895

已核验为正确

simulate_lowres / gaussian_blur_3d 仅作用于 image — 强度链路只接收 image；label 不参与。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:86-94,407-474

grid_dropout 只掩 image，label/wmap 不变 — 符合“遮挡输入、标签完整”的设计。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:289,330

多分辨率视图拆分：label 用 nearest，image 用 trilinear — 避免 label 被双线性污染。
D:\codes\work-projects\SegTask\segtask_v1\trainer\views.py:86-89

[设计]
增强管线组合清晰：空间 → 强度，inplace/clamp 契约明确 — 空间 flip → 仿射+弹性单次 warp → grid-dropout；强度仅 image；可选增强前 min/max clamp。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:64-98

RNG 混合 CPU/GPU，有意避免 CUDA 同步 — Bernoulli/标量参数 CPU 采样（_bernoulli_mask），再 async 上 GPU；但 elastic/noise/grid_dropout 仍用 GPU 默认 RNG，全链路复现需 seed_everything + deterministic。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:20-22,215-250,177,403,307
D:\codes\work-projects\SegTask\segtask_v1\utils.py:494-512

Dataset 侧 per-worker np.random.Generator，验证用确定性 RNG — 避免 fork 后 numpy 全局状态重复；val 用固定种子 + sample_idx。
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:592-617,26-27

train/val oversample 分离 — 训练用 aug_oversample_ratio，验证强制 1.0；val 不走 augmentor。
D:\codes\work-projects\SegTask\segtask_v1\data\specs.py:123-127
D:\codes\work-projects\SegTask\segtask_v1\trainer\validation.py:345-347

make_data 前景索引固定 seed=42、逐类 cap — 可复现；但每个样本 _compute_fg_indices 都 RandomState(42) 重置，跨病例 subsample 索引模式相同（非 per-pid 独立 RNG）。
D:\codes\work-projects\SegTask\segtask_v1\data\make_data.py:57-79,91-93

平移幅度 vs oversample 余量：启动期 warning，但未区分 z_axis H/W 无余量 — 只按全局 ratio 估算 margin。
D:\codes\work-projects\SegTask\segtask_v1\config.py:1721-1737

raw 整数 label 延迟二值化 — dataset 输出 int16 原始标签，loss 内 _label_to_binary 再 ==；与 validation 整卷路径 preprocess_label 分离，职责清楚。
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:819-820
D:\codes\work-projects\SegTask\segtask_v1\losses\losses.py:780-787

[优化]
_grid_dropout 对 num_holes 的 Python 循环 — 可向量化合并多 hole 掩膜，减少小 batch 上的 kernel launch。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:315-325

_simulate_lowres 按目标尺寸分组后逐组 interpolate — 已做分组，但 idxs.tolist() + Python dict 仍有 host 同步；可考虑按 zoom 分桶或固定少量档位。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:461-473

_random_flip 逐轴串行 — 三轴可合并为一次随机翻转决策张量，减少 Python 循环。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:109-116

_gaussian_blur_3d grouped conv 三轴可分离 — 已向量化选中样本，设计已较优；大 sigma 时统一 ks 略浪费算力（注释已说明取舍）。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:411-412,425

make_data 多进程无 per-worker 随机种子 — fg 索引 deterministic 无妨；若未来加入随机增广预处理需注意。
D:\codes\work-projects\SegTask\segtask_v1\data\make_data.py:402-406

[做得好]
仿射 + 弹性合成单次 grid_sample — G(x)=Θ(x+d) 与两次 warp 采样位置一致，省算力且避免双重模糊；image/label/wmap 插值模式分离（bilinear / nearest / 可配 wmap）。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:199-204,272-279

CPU 采样 Bernoulli 避免训练 step 内多次 device→host 同步 — 模块 docstring 与实现一致，对 GPU 流水线友好。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:3-6,20-22

intensity_clamp 在增强前采集 per-sample min/max — 比事后 clamp 更语义正确，不受 border/dropout 污染基准。
D:\codes\work-projects\SegTask\segtask_v1\data\augment.py:57-62,95-96

cubic 前景采样 _safe_center_range + clip — 限制中心域，避免 >50% 体素来自 edge-pad。
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:1071-1121

make_data 流水线完整 — bbox 裁剪、可选 spacing 归一、逐类 fg 索引/计数、原子写、meta 自描述。
D:\codes\work-projects\SegTask\segtask_v1\data\make_data.py:141-246

resize 路径 label/image 插值分离 — resize_3d(is_label=True) 用 order=0；rw 同样 nearest。
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:423-437,815-816,822-823

多分辨率 lazy split 与训练/推理对齐 — dataset 发 max-FOV cube，trainer center_crop 后再 split_views_native_3d，与 predictor 侧 cubic/z 逻辑一致。
D:\codes\work-projects\SegTask\segtask_v1\trainer\views.py:40-103
D:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:1018-1020

[借鉴]（相对 nnU-Net / MONAI 可补强项）
label/mask 几何变换用 padding_mode='zeros' — 当前 border 需配合足够 oversample；z_axis 面内无解，应优先改 label padding 或禁用面内大角旋转。
现状：augment.py:274-275；nnU-Net 惯例：mask 零填充。

增强后前景保证 / retry — 无“warp 后前景 voxel 数低于阈值则重采样”逻辑；稀疏结构训练可能偶发空 patch。可仿 nnU-Net 的 ensure_foreground。

解剖学镜像约束 — random_flip_axes=[2,3,4] 三轴独立 Bernoulli，无左-右器官约束或仅面内翻转策略。
D:\codes\work-projects\SegTask\segtask_v1\config.py:209-210

真 B-spline 弹性形变 — 可借鉴 MONAI Rand3DElasticd 的 B-spline 控制点 + 物理 spacing 感知的位移幅度。

spacing-aware 旋转/缩放 — make_data 已支持 median spacing 重采样；增强侧 aspect_correct 仅用 voxel 计数，可扩展为读 npz meta 的 target_spacing 做物理各向同性旋转。

loss 二值化 — 建议在 _label_to_binary 中对 warp 后 label 做 torch.round 或 torch.isclose，与 dataset.preprocess_label 对齐，降低 float 链路脆弱性。
D:\codes\work-projects\SegTask\segtask_v1\losses\losses.py:787 vs dataset.py:413

z_axis 面内 oversample — 若坚持 2.5D/z-slab 工作流且要用面内仿射，可考虑 extract_size 的 H/W 也乘 aug_oversample_ratio（或仅对 in-plane 轴），与 cubic 对齐。

审查结论摘要
审查点	结论
label grid_sample padding
确定问题：nearest+border，非零填充
z_axis vs cubic 余量
确定：z_axis 仅 D；cubic/whole 三轴
float == 二值化
建议：现链路 mostly 安全，缺 round 兜底
lowres/blur 仅 image
正确
scale 各向异性
设计局限：各向同性 scale + voxel-count aspect，非 spacing
单次 grid_sample 融合
高质量实现


segtask_v1 训练/验证全流程代码审查
审查基准：当前工作区未提交改动（train.py、trainer/*.py、tests/test_checkpoint_resume.py）。

[正确性]
确定 bug
训练步 dice 阈值与验证/部署不一致：训练日志 dice 调用 compute_dice_per_class(...) 使用默认 threshold=0.5，验证 MetricAccumulator 使用 cfg.predict.threshold（validation.py:318-324）。当 predict.threshold ≠ 0.5 或为逐类列表时，训练 dice 与选模指标口径分裂。segtask_v1/trainer/trainer.py:975-976，segtask_v1/utils.py:288-293，segtask_v1/trainer/validation.py:70-74,324。

SWA BN 重校准在 DDP 下只见本 rank 训练 shard：_swa_recalibrate_bn 遍历 self.train_loader（DDP 下经 DistributedSampler 切分），各 rank 独立累积 BN running stats，与单卡全量数据重估不等价；rank0 落盘的 SWA 权重绑定的 BN stats 有偏。segtask_v1/trainer/trainer.py:1104-1118，segtask_v1/train.py:90-92，segtask_v1/data/loader.py:768-770。

建议 / 边界（非必改，但应知情）
EMA 验证不做 BN 重校准：_validate 仅 _ema_swapped() 换权重，无 AdaBN；ModelEMA 文档已说明 BN buffer 被 EMA 平滑且评估前不重估。若 backbone 含 BatchNorm，EMA 验证指标可能系统性偏低/偏高。segtask_v1/trainer/trainer.py:1031-1032，segtask_v1/utils.py:47-49。

非有限跳步时 scheduler 仍推进、EMA 正确跳过（两处均已核验）：

bf16/fp32 手动 guard：skip_optim_step=True 时 scheduler.step() 仍调用（903），EMA 不更新（904-905），注释写明有意为之。
fp16 GradScaler 内部跳步：通过 scale_before > get_scale() 识别（927-929），EMA 跳过（932-933），scheduler 仍 step（931）。
后果：跳步 epoch 内 LR 按“虚拟优化步”前进，与真实权重更新次数可能长期错位（OneCycle/cosine 尤甚）。segtask_v1/trainer/trainer.py:877-905,921-933。
fp16 路径非有限 loss 不做 all_reduce 手动跳步：skip_optim_step 仅在 not self._scaler_active 时启用（884-886），fp16 完全依赖 GradScaler 检测梯度 inf/NaN；DDP 反传 all-reduce 后各 rank 梯度一致，通常安全。非 log 的 accum 边界上 pending loss 不 flush（833-834），loss guard 对该步无效，仍靠 GradScaler。segtask_v1/trainer/trainer.py:833-834,877-886,180-181。

梯度累积 / unscale / clip / step 顺序正确：尾批 _effective_accum 缩放（622-630,816-817）；accum 边界 unscale_ → clip → scaler.step → update → zero_grad（850-924）；非边界 no_sync()（801-803,811-819）。

DDP 非有限标志 all_reduce 一致：bf16/fp32 用 all_reduce_flag_any(loss_nonfinite or grad_nonfinite)（884-886），保证各 rank 同步跳步决策。segtask_v1/trainer/dist_utils.py:71-82。

验证指标 all-reduce 数学正确：可加混淆量（inter/pred_sum/target_sum/voxels/cov/sd_*）做 SUM；空 rank 零初始化保 shape 对齐；voxels 用 float64；loss sum/count 同步后再算 avg。与单进程全集 pooled 严格相等。segtask_v1/trainer/validation.py:154-201，segtask_v1/utils.py:347-351,362-409。

Checkpoint 续训逻辑整体正确：

RNG：torch CPU/CUDA + numpy + python 快照与 bytes roundtrip（checkpoint.py:47-79，trainer.py:1131-1138,1252-1259）。
best 主键：is_best=True 时 ema_as_primary=True，model_state_dict 存 EMA，在线权重进 model_online_state_dict（1125-1167,1178）；resume 优先加载 model_online_state_dict（1229-1231）。
Optimizer：relocate_optimizer_state 对齐 ZeRO/fused 混合设备（1233-1234，checkpoint.py:101-127）。
rank>0 RNG：restore 后 _reseed_rank_rng(seed, rank, start_epoch, ...) 避免 DDP rank 退化为 rank0 随机流（1260-1263，84-90）。
ZeRO：consolidate_state_dict(to=0) 在 rank 早退之前（1173-1177）。
训练 dice 与验证 dice 聚合语义本就不同（非 bug，需知）：训练 dice 仅 log 步抽样 + per-batch ignore_empty；验证为全集 pooled + cov 掩码。代码已注释（988-989）。与阈值问题（#1）叠加时更易误判收敛。

deep_supervision=True 但 deep_supervision_weights=[]：pipeline 不包 DeepSupervisionLoss（权重空为 falsy），forward 可能收到 list 却在 runtime 报错；validate 仅在 weights 非空时预警（config.py:1769-1777，各 pipeline 如 vanilla3d.py:38-42）。

[设计]
Pipeline 策略模式清晰：build_pipeline 唯一 mode 分支入口（factory.py:28-78），ViewPipeline 统一 prepare_batch / compute_loss / split_for_metrics（base.py:52-147），Trainer 只做协调（trainer.py:122-124,794-815）。

验证策略模式对称：PatchValEvaluator / VolumeValEvaluator 共享 MetricAccumulator，产出同结构 metrics dict（validation.py:306-499）。

进程守护完善：Linux PR_SET_PDEATHSIG + 竞态自杀（train.py:116-135）；SIGTERM/SIGINT 销毁 process group（138-150）；NCCL async error handling（168）；DDP 异常路径跳过 barrier（198-206）。

Config 校验较全：load_config 必调 validate()（config.py:2228-2237）；深监督权重长度预警（1769-1781）；predict.threshold 范围/列表校验（2032-2043）；ZeRO 需多卡（2061-2063）。

DDP 与 compile/EMA/checkpoint 分离合理：optimizer/EMA/checkpoint 绑裸 self.model，前向走 self.fwd_model（DDP 包装）（trainer.py:214-254）；unwrap_compile 统一剥壳（checkpoint.py:199-201）。

val_loss 与 train loss 口径刻意分离：medium 验证用裸 base_loss（无 DS/aux/topo），文档已说明不可直接对比（validation.py:355-357）。

early stopping / patience 以“有验证的 epoch”计次：仅当 save_best_metric in val_metrics 且非 best 时 patience_counter += 1（418-439）；val_every>1 时跳过验证的 epoch 不计 patience，与 config 注释一致（config.py:812-813）。

[优化]
训练步 dice 应对齐 predict.threshold：一行传参即可消除 #1，并让 monitor 曲线与选模一致。trainer.py:975。

SWA BN 重校准 DDP 方案：rank0 用全量 train_loader / 或跨 rank 同步 BN 累积统计 / 或 estimate_bn_stats 前临时禁用 DistributedSampler。trainer.py:1085-1120。

EMA 验证可选 BN 重校准：配置开关 + 少量无增强 train batch，复用 _swa_recalibrate_bn / AdaBN 逻辑。trainer.py:1007-1035，predictor/adabn.py:79-97。

跳步时 scheduler 是否同步跳过：可增 train.skip_scheduler_on_bad_step 或在 GradScaler skip 时不调 scheduler.step()，使 LR 与有效优化步对齐。trainer.py:903,931。

high 模式验证成本：每 epoch 全 val 滑窗；可借鉴 val 子集 / 降频 + medium 快筛。trainer.py:407-410，validation.py:369-479。

fp16 非 log accum 边界可提前 flush pending：在 is_step_boundary 时无条件 _flush_pending()，使 loss guard 与日志更一致（GradScaler 仍为主保护）。trainer.py:833-834。

Async checkpoint 队列无上限：极端 save_every=1 + 慢盘可能堆积；可加 max_inflight 或同步 best 保存。checkpoint.py:147-196。

grad_norm_lazy_sync 跳 D2H 但 clip 仍执行：正确；若需 health 指标与 lazy 共存，文档化取舍。trainer.py:854-858。

torch.compile 仅在 Triton 可用时启用：否则 eager fallback（199-212）；可考虑 torch.compile(..., backend="eager") 或 Inductor 无 Triton 路径的配置提示。

VolumeValEvaluator 懒加载 Predictor：首 epoch high 模式有冷启动；可训练开始前 warmup 构建。validation.py:402-413。

[做得好]
尾批梯度累积分母修正：_effective_accum 避免最后不满 accum 组被过度缩小（622-630），注释与实现一致。

延迟 loss .item() 同步：pending GPU 缓存 + 边界/日志步批量 flush，减少 micro-step D2H（749-776,825-834）。

Loss fp32 + logit clamp 防 AMP NaN：forward AMP、loss 强制 fp32（amp.py:33-37,75-96）；pipeline 统一 compute_loss_fp32（如 vanilla3d.py:60-61）。

MetricAccumulator pooled 设计：GPU 累加、一次 all-reduce、闭式导出 dice/iou/mcc/balanced；ignore_empty 用 cov 掩码（validation.py:51-300）。

Checkpoint 工程化：async CPU 快照 + RNG bytes 打包 + optimizer 设备 relocate + ZeRO consolidate 顺序（checkpoint.py:130-196,101-127）；tests/test_checkpoint_resume.py 覆盖 RNG roundtrip 与 fused AdamW 续训。

ModelEMA CPU offload + foreach 热路径：pinned staging、单次流同步（utils.py:71-127）。

DDP 文档化 batch-pooled loss 非严格等价：trainer 初始化日志明确 batch_dice/Tversky/GDL 在 DDP+accum 下为近似（243-248）。

OneCycle resume horizon 漂移 reconcile：按比例 fast-forward last_epoch（optim.py:237-261）。

Val high 模式 bbox crop + voxels_override：裁剪后仍按整卷体素数算 TN/MCC（validation.py:461-472）。

2.5D aux 权重默认几何衰减 0.5^(k+1)：与 nnU-Net 多尺度思想一致（slab25d.py:33-37）。

[借鉴]
深监督权重：当前默认 [1, 0.5, 0.25, 0.125] 且 DeepSupervisionLoss(normalize_weights=True) 会归一化到 sum=1（losses.py:291-306,568-569）；nnU-Net 通常按尺度衰减但不重归一化——若要对齐论文行为可设 normalize_weights=False 或显式配置未归一化权重表。

验证降频/子集：已有 val_every（config.py:770，trainer.py:407）；可再加 val_subset_ratio / 仅 best 候选 epoch 跑 high 模式，medium 日常监控。

torch.compile：已支持 compile_mode + Triton 探测（trainer.py:196-212）；医学 3D 大 patch 可试 mode="reduce-overhead" 或 selective compile（仅 encoder）。

nnU-Net ignore_empty：验证侧在 MetricAccumulator.compute 用 cov 掩码（235-239）；训练 loss 侧 BinaryDiceLoss.ignore_empty 与 config 联动——保持 train/val 空类策略文档化即可。

AdaBN 推理链路可复用到训练尾段：predictor/adabn.py 已实现；SWA/EMA 收尾可统一调用 estimate_bn_stats，避免各写一套 forward loop。

MultiResolutionLoss 各分辨率等权平均（778）：非 nnU-Net 式按尺度衰减；多 FOV 任务可考虑与 deep_supervision_weights 类似的 per-res 权重配置。

Monitor / health 指标：nonfinite_steps、grad_clip_frac、update_ratio 等（686-719）可导出到 nnU-Net 式 progress png 或 W&B，便于长训诊断。

关键路径核验摘要
核验项	结论
训练 dice 阈值 vs predict.threshold
不一致（#1，默认 0.5 时无感）
非有限跳步 → EMA
两处均跳过 EMA
非有限跳步 → scheduler
两处均仍 step（有意）
尾批 accum 缩放
正确
unscale→clip→step→update
正确
DDP 非有限 all_reduce
bf16/fp32 正确；fp16 靠 GradScaler
SWA BN + DDP
各 rank 只见本地 shard（#2）
EMA 验证 BN
不重校准（#3）
checkpoint RNG/EMA/online/ZeRO
正确
val metric all-reduce
数学正确


segtask_v1 推理全流程代码审查
审查范围：predict.py、predictor/ 全包，对照 data/dataset.py 几何原语与 trainer/validation.py 的 VolumeValEvaluator。

[正确性]
确定问题 / 高风险

训练 H/W 重采样用 scipy.zoom，推理 GPU 主路径用 F.interpolate，存在系统性亚体素偏差
训练 dataset 对面内 resize 走 resize_3d（scipy.ndimage.zoom, order=1）：dataset.py:423-448、dataset.py:815-816。
推理 z 滑窗 GPU builder 对面内 resize 走 trilinear F.interpolate：inputs.py:91-93、inputs.py:122-124；sliding.py:81-82 注释亦写明替代 scipy。
二者边界处理与 align_corners=False 网格定义不同，属 train/deploy 几何不一致（keep_native_multi_res 主路径亦受影响：max-FOV slab 的 H/W 步）。

legacy CPU 多分辨率路径对应已不存在的训练流，误配风险高
当 len(multi_res_scales)>1 且 keep_native_multi_res=False 时，训练走 Vanilla3DPipeline 直通 batch（vanilla3d.py:52-53），但 dataset 仍只发 (1, eD_max, …) 单通道 cube（dataset.py:780-820），与 ModelTopology 期望 in_channels=n_views（topology.py:118-122）不匹配；推理却走 CPU 路径 build_z_window_cpu_multi_res：sliding.py:149-157、inputs.py:180-206。
现代有效多分辨率训练要求 keep_native_multi_res=True（config.py:1811-1828），legacy 路径实为死代码/误配陷阱。

z_boundary_mode='stretch' 推理分支与训练几何不一致（若绕过 sync()）
训练 dataset 恒 edge-pad 抽 patch，无 stretch 分支（dataset.py:809-810、config.py:1175-1185）。
推理 GPU 单分辨率仍实现 stretch：inputs.py:106-107、inputs.py:111-124；blend 逆变换亦分支：sliding.py:208-220。
load_config/predict.py 会 sync() 自动升级 stretch→edge_pad（config.py:1178-1185、predict.py:62-66），但直接 Predictor(...) 且未 sync 时仍会分叉。

3D 多分辨率推理只取 view0 输出通道，丢弃其余 view 的预测
训练 MultiResolutionLoss 监督全部 n_views 组通道（losses.py:763-778）；模型 out_classes = num_fg × n_views（topology.py:120-121）。
推理统一 pred[:, :num_fg]：forwards.py:235、forwards.py:279、forwards.py:304——仅 canonical 1× view 概率进入 blend；view1+ 的 sigmoid 输出被丢弃（与 split_for_metrics 取 view0 一致：losses.py:792-794，属设计取舍，非实现错误，但部署未利用多 view 预测）。

VolumeValEvaluator 二值化语义与部署 prob_to_label 不一致
验证：逐通道 (prob > thr) 独立二值（validation.py:455-459）。
部署：argmax 赢者通吃再阈值（blending.py:110-122）。
多类概率均超阈时，high val Dice 与最终 NIfTI label map 可能偏离。

建议 / 条件风险

skip_empty_windows 启用后可能漏检低对比前景
判据为归一化窗内 max() <= threshold（sliding.py:117-118、sliding.py:403-404）；z-score/低对比病灶窗内 max 可很低，被当纯背景跳过；默认 threshold=0.0 且默认关闭较保守（config.py:1038-1041）。

spacing 回采与 bbox 回贴逻辑正确
resample_to_spacing/resize_3d（scipy）前后对称（dataset.py:148-159、predictor.py:396-450）；bbox 裁切后按原 (D_orig,H_orig,W_orig) 画布拼回（predictor.py:453-458），索引与 affine 一致。

sigmoid 假设正确，无 softmax 分割后处理
全路径 torch.sigmoid（forwards.py:118、forwards.py:279）；多标签独立通道，与训练假设一致（predictor.py:38-39）。

per-class 阈值实现正确
标量/列表校验（predictor.py:155-159）；prob_to_label 按 argmax 类取对应阈（blending.py:113-121）。

NIfTI affine/spacing/direction 回写正确
CopyInformation(ref_img)（predictor.py:528-533）；数组 (D,H,W) 对应 sitk (Z,Y,X)，无需转置（predictor.py:522-523）。

2.5D TTA 正确避免翻 D 轴
仅 H/W flip，注释与 _FLIP_SPECS_2_5D 轴映射明确（forwards.py:66-72、forwards.py:122-134）；lift 模式走 3D TTA，D 为真空间轴（forwards.py:217-238）。

CPU 回退路径明确
非 keep_native_multi_res 且多分辨率 z 滑窗：build_z_window_cpu_multi_res + resize_3d（scipy）（sliding.py:149-157、inputs.py:204）；whole_volume_forward 全程 scipy（sliding.py:46-53）。

[设计]
ModelTopology 作为模式派生单一真相源，Predictor 正确复用（predictor.py:169-178、topology.py:74-160），与 build_model/build_pipeline 对齐。

VolumeValEvaluator 直接复用 Predictor.predict_preprocessed_array，关闭 log_progress（validation.py:369-442、validation.py:408-412），实现 train/val/deploy 滑窗一致（除 npz 无 z_spacing 时 z-interleave 回退：predictor.py:484-485）。

显存逃生门设计完整：acc_dtype=fp16、accumulate_on_cpu、vol_dtype=fp16（predictor.py:66-71、predictor.py:88-91）；_finalize_accumulators 避免 GPU fp32 峰值（sliding.py:444-456）。

几何原语部分复用、部分重复：z/cubic patch 抽取与 dataset 共享 extract_z_patch_padded/_extract_cubic_patch（inputs.py:27）；但 _mr_native_sizes/_mr_target_shape 在 Predictor 重算（predictor.py:214-240），与 patch3d.py:55-78 逻辑平行，维护成本偏高。

AdaBN 与 TTA 交互有意识设计：估计期强制 TTA 串行（forwards.py:82-83、predictor.py:280-283、adabn.py:58-76）。

推理精度分层清晰：run_inference 形状预检、EMA 选择、MedNeXt deploy reparam（io.py:131-163、io.py:96-183）。

[优化]
z 滑窗已按 batch_size 攒批 + TTA 分块（sliding.py:160-167、forwards.py:75-111）；cubic 同理（sliding.py:357-393）。

_blend_z_batch 按 actual_d 分组合并 interpolate（sliding.py:194-220），减少冗余上采样。

可改进：3D 多 view 输出 ensemble（对 view1+ 的 num_fg 通道做平均/加权），当前仅 view0（forwards.py:279）。

可改进：Gaussian blend σ 固定 n/4（blending.py:55-58），不可配；尾窗权重未按 MONAI 做 sigma_scale 自适应。

可改进：统一重采样器——训练 dataset 与推理 GPU 窗口 builder 共用同一实现（全 GPU trilinear 或全 scipy），消除 #1 偏差。

skip_empty_windows 可用更鲁棒判据（如归一化后 percentile / 方差），替代单纯 max()（sliding.py:117-118）。

[做得好]
R6 模块化拆分清晰：inputs/forwards/sliding/blending/io 职责边界明确，纯函数 + 显式参数，可单测（inputs.py:1-16、forwards.py:1-7）。

keep_native_multi_res GPU 路径与训练 split_views_native_3d 几何同构（中心裁 + trilinear 回 patch_size）：inputs.py:128-154 vs views.py:40-103。

滑窗 overlap blend 实现扎实：1D/3D 可分离高斯权重、尾窗全覆盖（blending.py:26-43、blending.py:49-71）；cubic 居中 edge-pad 与训练一致（sliding.py:408-423、dataset.py:901-930）。

诊断体系完善：首 batch logits/prob 统计、NaN→bf16 提示（forwards.py:140-201）；饱和前景训练侧告警（predictor.py:312-338）。

bbox ROI 推理 + 全尺寸画布回贴 + sitk 元数据保留端到端闭环（predictor.py:370-458、predictor.py:515-544）。

2.5D native_d / lift / folded 三条 forward 分派严谨，rank 校验充分（forwards.py:217-268）。

z-interleave 子流互斥缝回，无跨流加权需求（sliding.py:253-294）。

TTA 批量化与串行严格等价并有测试覆盖（forwards.py:88-96、tests/test_tta_batched_equivalence.py）。

[借鉴]
MONAI SlidingWindowInferer：sigma_scale 可配、overlap 模式统一；当前 build_1d_weight σ=n/4 硬编码（blending.py:55-58）可参数化。

统一 GPU/CPU 重采样器：参考 MONAI Resample/Spacing，将 resize_3d（scipy）与 F.interpolate 收敛到单一后端，并固定 align_corners/边界模式。

多 view ensemble：训练已监督全 view，推理可借鉴 nnU-Net/MONAI 多尺度融合，对 view0..K 的 num_fg 通道做加权平均而非只取 :num_fg。

跳窗策略：可参考 nnU-Net 的 "only predict patches containing foreground in low-res preview"，比 max()<=thr 更稳。

VolumeValEvaluator 与部署后处理对齐：验证指标应走同一 prob_to_label（或明确文档化差异），避免选模与部署 label 语义分裂（validation.py:455-459 vs blending.py:77-134）。

快速结论
维度	结论
主路径（keep_native_multi_res + GPU 滑窗 + edge_pad）
与训练几何高度对齐，工程成熟
最大隐患
scipy zoom（训练面内）vs F.interpolate（推理面内）混用
legacy CPU multi-res
对应无效训练配置，误配风险
3D 多分辨率输出
仅用 view0 预测，其余 view 监督权重未在部署兑现
TTA / NIfTI / bbox
实现正确且文档充分


生成任务进展：  
gentask 的整体架构是清晰、连贯的：topology(几何真相源) → factory(装配) → backbone → generation(回归/扩散统一 forward/restore/degrade 接口) → trainer/predictor。在线退化管线顺序正确（增强→裁剪→退化），EDM 预条件/Karras schedule 忠于原论文，SSP 层厚建模、decimate 相位对齐、Gaussian 滑窗融合等都是高质量实现。

但作为从 segtask_v1 剥离的工程，它在几处出现了**"配置/注释承诺了、代码没兑现"的失配，以及训练基建相对母项目退化**。最需要优先处理的是下面 4 条跨层主线问题。

跨层主线问题（最高优先级，均已亲自核验）
UNet3D 在 patch2/patch4 stem 下必崩 ✅ unet.py:396 注释写"stem_stride>1 时 forward 末尾上采回输入分辨率"，但 forward 在 unet.py:479-484 直接 RuntimeError，没有任何上采样。decoder 只还原 n_levels-1 次下采样，stem 的 2/4 倍从不恢复。ADM/EDM2 路径用输出插值规避了，唯独通用 UNet 没有。→ 要么补 F.interpolate 到 target_size，要么在校验层拒绝 arch=unet + patch stem。

训练/部署几何不一致（2.5D 面内分辨率） ✅ 训练 Volume3D 把 H/W resize 到 patch_size（core.py:329-331，且 core.py:185 注释自称"与 predictor 一致"），但推理 _slab_views_2_5d 用原生 H/W、只滑 z（gen_predictor.py:187）。全卷积虽不崩，但模型在"缩放过的面内尺度"上训练、却在原生尺度上推理，SR 这种对内容尺度敏感的任务会掉质量。推理侧多处（SISR 用 HR patch_size 在 LR 网格滑窗 B2、whole 模式不 resize B3）都是同源的几何漂移。→ 抽取一个共享的 inference_geometry，让 trainer pipelines 与 predictor 复用同一套裁剪/resize，从根上消除漂移。

best_model 存的是在线权重，不是验证用的 EMA 权重 ✅ 验证在 EMA 影子权重下测指标，但 _save_best(gen_trainer.py:378-389) 主键 model_state_dict 存 bare.state_dict()（在线权重），EMA 仅放 ema_state_dict。直接 load(model_state_dict) 部署 ≠ 验证时的模型。segtask_v1 是以 EMA 为主键存 best。→ best 以 EMA 写主键（并保留 online 副本）。

训练基建相对 segtask_v1 明显退化（多项已核验）

断点续训不保存 RNG（_save_checkpoint 无 rng_state）→ 续训后增强/退化/扩散噪声序列错位，严格复现失败。
无非有限 loss/grad 守卫 → NaN/Inf 会污染权重（母项目有 group_has_nonfinite 跳步）。
DDP / ZeRO / metric all-reduce / torch.compile / pretrain 加载 全部未接线（配置字段 compile_mode/pretrain* 是死字段）。若计划多卡，需从母项目移植。
分区详述（保留最重要项，均带 file:line）
1) 数据读取
[正确性] z_boundary_mode='stretch' 被校验/存储但从不生效，训练恒走 edge_pad（core.py 存 self.z_boundary_mode 后再无读取）→ 与推理假设可能不一致。建议实现或从配置移除。
[正确性] cubic patch 无 padding 分支直接返回缓存卷的视图（core.py:444-454，z 路径则 .copy()），共享内存，未来 in-place 会污染 LRU 缓存。→ 无条件 copy。
[正确性] n=1 时 train_val_split 产出空训练集（n_val=max(1,...)）。→ n<2 特判。
[设计] 四个独立 LRU 缓存（img/lbl/rw/cond）各自按 cache_max_volumes 计数，实际 RAM≈4×，而诊断只估一份（loader.py:596-625 低估）。→ 合并为单条目多字段缓存 + 按字节预算。
[优化] 冷缓存每样本开 4-5 次 npz（_open_npz 反复解 ZIP 目录）。→ 单次 _load_npz_bundle 一并填四缓存。
[借鉴] MONAI CacheDataset/SmartCache（缓存 patch 级而非整卷）、mmap/Zarr 后端、pos/neg 运行时裁剪重试。
1) 模型构建
[正确性] 主线①(UNet3D patch stem) ✅；ADM/EDM2 硬编码 2D（spatial_dims=2），真 3D 扩散 SR 不支持；EDM2 忽略 stem_mode（edm2_unet.py 恒建默认 stem，却日志假装应用）；扩散多视图在 factory 不拦、只在 trainer 崩（晚失败）。
[设计] 通用/ADM/EDM2 三套 UNet 各自复制 aux/深监督/forward 契约 ~3 份 → 抽共享 head router；生成项目仍用 SegmentationHead/num_fg_classes 分割遗名，易误读。
[优化] models/ 全无梯度检查点（3D/宽通道易 OOM）；EDM2 _MPConv 训练期原地 copy_ 归一化权重，与 Adam 动量交互异常，建议对照官方实现改为 forward-only detached normalize。
[借鉴] SR 首选 pixelshuffle+ICNR（已具备，建议设为 SR 默认）；z-SISR 各向异性 pixelshuffle 已很好；可选 SwinIR/HAT/RRDB；(2+1)D 伪 3D。
1) 数据增强 / 退化
[正确性] spacing 已烘焙进 npz 但训练从不使用（load_npz_spacing 无训练侧调用），只有推理用 target_z_spacing 自适应 → 异质层厚队列存在 train/deploy 分布偏移；sr_noise_std 作用在归一化空间而非 HU，未标定；三角 SSP 核实际半宽比注释窄 1 体素；simulate_lowres 只增强 image 不动 cond（启用则与 cond 错位）。
[设计] 管线顺序/训练验证 RNG 门控/cond 空间同步/make_data 原子写与原始 HU 存储都正确且良好。RNG 后端混用（random/torch/numpy）削弱严格复现。
[借鉴] Real-ESRGAN 二阶退化、BSRGAN 随机化、spacing 感知的物理 mm 空间退化、Rician/Poisson 噪声模型。
1) 训练 / 验证
[正确性] 主线③④(best 存权重、无 RNG/守卫、无 DDP) ✅；扩散验证不固定种子→ val PSNR/SSIM 抖动大、选模噪声；EDM 训练 σ 采样无 [σ_min,σ_max] 截断（截断只用于采样）；sigma_data=0.5 对 minmax[0,1] 数据（std≈0.2–0.3）未必匹配，影响预条件与损失权重。
⚠️ 已修正专项过度结论：专项称"验证 SSIM 按整 batch 池化、与 per-image 口径不一致"。我核验 recon.py:159 ssim_map.mean()：等尺寸 patch 下全局均值恒等于逐图均值再平均（且 drop_last=True 无尾差），故此项数值上基本无影响，仅属"逐图标量更规范"的风格建议，不作为 bug。
[已核验正确的关键路径] 梯度累积缩放 loss/accum + 尾 batch 触发、GradScaler 仅 fp16、unscale_→clip→step 顺序、warmup 按优化器步、扩散+多视图显式拒绝——均正确。
[借鉴] 默认 amp_dtype: auto(bf16 扩散更稳)、LPIPS/DISTS 感知指标、EMA warmup/ramp、参数分组 LR、keep-last-k/异步 checkpoint。
1) 推理 / 采样
[正确性] 主线②(2.5D 不 resize / SISR LR-HR 窗口 / whole 不 resize) ✅；扩散采样端到端不可复现（restore 不传 generator、predict.py 不 seed_everything、DDIM/DDPM 随机步无 generator）；DDPM x0.clamp(-1.5,1.5) 与 minmax[0,1] 不一致；TTA 未实现；input_grid='hr' 不做 degrade（喂干净 HR 会静默错误条件）；checkpoint strict=False 50% 阈值可能半随机权重跑起来；CLI 宣传的 predict.batch_size 实际不存在。
[设计/优化] predictor 重实现了 trainer 的多视图几何(漂移根源)；逐 patch 串行无批处理；扩散成本×z-slab×滑窗（可达上百次 UNet/slice）；全 HR 累积张量在 GPU 易 OOM。
[借鉴] Gaussian 滑窗(已具备,建议对齐 MONAI SlidingWindowInferer)、DPM-Solver++/s_churn、8-fold flip 自集成、inference_mode+bf16。
建议优先级（仅排序，不实现）
级别	问题	位置
P0
UNet3D patch stem 输出不上采（必崩/注释误导）
unet.py:479-484
P0
2.5D/ SISR/whole 训练-推理几何漂移
gen_predictor.py:187 vs core.py:329
P0
best 存在线权重≠验证的 EMA
gen_trainer.py:378-389
P1
续训无 RNG + 无非有限守卫
gen_trainer.py:396-436
P1
扩散验证/采样不可复现（种子）
diffusion.py/predict.py
P1
EDM2 忽略 stem_mode / ADM-EDM2 仅 2D
edm2_unet.py
P1
σ 采样无截断 + sigma_data 校准
diffusion.py:77-84
P2
spacing 训练侧未用 / noise HU 未标定
degradation.py
P2
缓存别名(cubic) + 四缓存内存 + 冷缓存多开 npz
data/
P2
三套 UNet 重复 / 无梯度检查点
models/
P3
TTA / 更快采样器 / 感知指标 / 批处理推理
多处





3 


4 模型流可视化乱套了：
感觉模型流可视化仍然设计的不够通用，不够好，我增加了一些模块，例如自注意力等等，完整参数见seg2_5d.yaml和seg3d.yaml（这两个配置里面的参数几乎可以囊括所有可能情况），模型流可视化便乱套了。

一、现状理解（现有实现怎么工作）
模型流可视化是一套自研的三段式管线（零第三方依赖，输出单文件 HTML）：

采集（model_flow.py _trace_modules）：给所有模块注册 forward-pre/forward hook，跑一次 dummy 前向；用 id(tensor) + 一个 TorchFunctionMode（_make_prov_mode）追踪张量"血缘"（provenance），试图穿过 cat/+/interpolate 等 functional 算子。
重建（_ModelGraphBuilder + _emit_*）：把叶子按硬编码命名规则聚合成容器框，再靠血缘反查连边、识别残差/融合。
布局渲染（graph.py assign_grid_layout + render.py 的 JS）：CSS Grid 摆 (rank,col,colspan)，再用一大段手写 SVG 逻辑做正交折线、外缘车道、跳连分侧、聚焦布线等。
二、根因分析：为什么"每次改模块就乱套"
问题不是某个 bug，而是架构层面的耦合。当前实现把"分组、连边、残差、布局"四件事都建立在对具体模块命名/结构的硬编码假设上，任何新模块只要不符合这些假设，就至少打破其中一环：

根因 1：分组靠"名字白名单"，不靠模块树本身。 _top_key / _block_key 依赖写死的段名集合：


model_flow.py
Lines 118-121
_INDEXED_SEGS = {
    "stages", "downsamples", "levels", "upsamples", "level_blocks",
    "level_entries", "blocks", "gates", "branches", "fusions",
}
你加 selfattn 时，factory 把 stage 包成 nn.Sequential(ResNetStage, SelfAttentionBlock)（factory.py:126），于是模块路径变成 encoder.stages.4.1.norm 这种；再加任何新容器名（新 backbone、新头、新融合），只要不在这个白名单里，分组就归错框。新增模块 = 必须回来改白名单，这正是"改一次乱一次"的直接来源。

根因 2：残差/捷径靠"类型清单 + 属性名"猜。


model_flow.py
Lines 244-281
def _residual_block_types() -> Tuple[type, ...]:
    ...  # 手工列举 ResNetBlock / ConvNeXtBlock / _ResBlockNoEmb / _EDM2Block
_SHORTCUT_ATTRS = ("shortcut", "skip_connection", "conv_skip")
SelfAttentionBlock 的残差是 x = x + h（blocks.py:991），它既不在类型清单里、也没有名叫 shortcut 的子模块 → _has_residual 判 False → 不 reseal、不标残差。于是它的 + 把"块输入血缘"和"注意力支血缘"混在一起，连边错乱。每种新残差写法都要回来登记。

根因 3：数据流靠 id(tensor) + functional 血缘，脆弱且随实现细节漂移。 SelfAttentionBlock 内部是 rearrange(flatten) → Conv1d → sdpa/permute/einsum → unflatten → +（blocks.py:984-999）。这些 reshape/permute/chunk 全是 functional，血缘要靠 _ProvMode 把 id 关系一路缝合，还要处理 id 复用、透传张量、inplace、weakref 校验（_supersede、reseal、"悬空 shortcut 兜底"等一大堆特判，见 _emit_leaf_flow 近 150 行）。Conv1d 作用在 (B,C,HW) 上，导致框里显示的形状维度都和 3D 语义对不上（_container_io 又要启发式回退）。追踪正确性依赖模块内部怎么写，换个等价写法就可能变样。

根因 4：布局是"为已知架构手工调"的，不是通用图布局。 render.py 里注释明确写着这些逻辑是针对 unetpp/unet3p 密集解码器、deep-supervision 头→loss、hierarchical 反馈边 等具体结构调出来的（外缘车道两色划分、Rloss 束、聚焦星形布线…）。新拓扑一旦产生新的边型分布，这套手工规则就没覆盖到 → 连线交叉/穿框/溢出。

一句话根因："分组"和"连边"是两条各自依赖硬编码假设的管线，且都以模块命名/类型为锚点；布局又针对已知架构手工特化。三者任何一处遇到未登记的新模块都会失配，表现为"乱套"。

三、业界通用做法调研
方案	机制	能否处理本项目（动态返回 dict/list、if training、reshape 残差）	依赖
torch.fx symbolic_trace
符号追踪成静态 IR
❌ 官方明确：不支持 data-dependent 动态控制流；本模型 if self.training/返回结构随分支变化会 trace 失败或失真
内置
torch.export
更强的图捕获
⚠️ 比 fx 强但更严格，动态返回结构仍需改造模型
内置
torchview (draw_graph)
__torch_function__ eager 追踪 一次前向，输出 graphviz
✅ 支持 Skip/Sequential/嵌套/nested，depth 参数按模块树折叠；expand_nested、rolling
+torchview +graphviz
torchlens
eager Python 函数包裹，捕获真实运行
✅ 任意架构含动态/递归/融合注意力，一行出图
+torchlens +graphviz
Netron (ONNX 导出)
导出 ONNX 后看图
⚠️ 需能 export ONNX；算子层级、非交互定制弱
onnx + 外部查看器
关键结论：

静态符号方案（fx/export）在本模型上不可行（动态控制流 + 动态返回结构）。
业界成熟可视化（torchview / torchlens）与你现有采集思路同源（都是 eager 追踪），区别在于它们用模块树做分组、用 graphviz 做分层布局，因此对新模块天然鲁棒——这恰好对应你缺的两点。
因此正确方向不是"再打一个补丁修 selfattn"，而是把分组从命名白名单改为模块树、把布局从手工特化改为通用分层算法。
四、通用设计方案（核心思想）
用三条"结构驱动、去硬编码"的原则替换四类特判：

分组 = nn.Module 树本身：容器框 = 模块树节点；折叠深度 = 通用 depth 参数（类似 torchview）。删除 _INDEXED_SEGS/_top_key/_block_key 的名字白名单。新模块因为本就是树里的节点，自动获得正确的框，无需登记。
连边 = op 级 DAG + "按当前可见深度上提(lift)"：底层只维护一张"张量→产出op"的真值 DAG；显示时把叶子边统一上提到"当前可见的祖先框"。这是一个统一操作，取代所有 per-container 补边/reseal/supersede/悬空兜底。
残差/融合 = 从 DAG 结构判定：融合 = 任何有 ≥2 张量输入的 op（cat/add/mul）；残差 = 某 add 的一个输入是另一个输入的祖先。不再靠类型清单/属性名。
布局二选一（见下方决策）：要么引入一个通用分层布局（Sugiyama/graphviz），要么保留自研 HTML 但把布局改成"对任意 DAG 通用"的分层，删掉架构特化路由。

五、可执行计划（分步、可独立验收）
每步都可单独交付、单独验证；先做"采集/分组"内核，再做布局，最后清理。

步骤 0（对齐，仅确认不写码）：定下两个决策点（见文末问题）：是否允许引入依赖；布局走"引入 graphviz/JS 分层库"还是"自研通用分层"。
步骤 1｜统一的结构化 IR 内核：新增一个"追踪→op级 DAG"的模块，产出 {节点(叶子/op), 边(张量流), 模块树父子} 三张表。复用现有 hook + TorchFunctionMode 采集，但只产出真值 op-DAG，不做任何命名判断。验收：对 seg2_5d.yaml/seg3d.yaml 各跑一次，导出的 DAG 节点数=被执行叶子+融合op数，边无悬空、无自环（单测断言）。
步骤 2｜树驱动分组 + 边上提：用模块树做容器层级，实现"边 lift 到可见深度"的通用算法，替换 _top_key/_block_key/_emit_intra_edges/_emit_leaf_flow。验收：unet（含 selfattn+multirf）、unetpp、unet3p、一个扩散 backbone 四种配置下，每个 stage 框的 in/out 形状正确、encoder↔decoder 跳连数量与代码一致（单测按拓扑断言）。
步骤 3｜结构化残差/融合判定：用"祖先关系"判残差、用"≥2 输入"判融合，删除 _residual_block_types/_SHORTCUT_ATTRS。验收：SelfAttentionBlock、ResNetBlock、ConvNeXtBlock 的残差都被标出且只标一条，无"悬空 shortcut"。
步骤 4｜通用布局替换：按步骤 0 的决策，把 render.py 的手工车道/rank 特化换成对任意 DAG 通用的分层布局。验收：四种配置渲染无连线穿框、无右侧溢出被裁；同一配置两次渲染布局稳定（幂等）。
步骤 5｜清理与回归：移除死代码与特判，补 README 说明新架构；跑通全量可视化回归。验收：git grep 不再有架构名白名单；已有 inspect.getsource 类测试更新通过。
范围边界：只重构 segtask_v1/visualization/（model_flow 为主，必要时 graph/render），不改模型代码（models/）。data_flow / predict_flow 复用同一 IR 与渲染层，但本任务只保证 model_flow 达标，另两个作为兼容性验证。

主要风险：① 若选"自研通用分层"，Sugiyama 类算法有一定实现量（可控，约 1 步）；② 引入 graphviz 是系统级依赖（需你许可）；③ eager 追踪对"动态返回结构"已被现有代码验证可行，风险低。

依赖放开后的关键变化
之前四个根因里，根因 3（id 血缘追踪脆弱）和根因 4（布局手工特化） 现在都能直接交给成熟库解决，而不是自己再造轮子：

追踪/建图引擎：用 eager 追踪类库（torchlens 或 torchview），它们和你现有采集同源（都是 __torch_function__/函数包裹跑一次真实前向），因此照样支持你模型的动态返回（dict/list）与 if self.training，但把"op 级真值 DAG + 模块树层级 + 形状/元数据"做成了经过 1 万+架构验证的稳定能力。新增自注意力、multirf、乃至新 backbone，都不需要回来改可视化。
布局引擎：用 graphviz（dot 分层布局）算坐标——通用、对任意 DAG 成立，彻底取代 render.py 里那套为 unetpp/unet3p/深监督手工调的车道路由。
两者选型对比（本项目视角）：

库	定位	优点	代价
torchlens
追踪引擎（程序化 API 最强）
文档化的 per-op 图 + 父子层级 + 元数据，明确支持动态/递归/融合注意力；最适合"喂给我们自己的渲染器"
需 graphviz 画图；引擎较重
torchview
可视化（graphviz 直出）
depth/expand_nested 按模块树折叠、形状、Sequential/skip 都支持，出图快
稳定公开产物是 graphviz Digraph，程序化取节点图偏内部
graphviz
布局引擎
通用分层布局、可导出坐标(json/plain)供自绘
系统级依赖（需装 graphviz 二进制）
推荐架构（最高质量、保留交互）
引擎 + 布局用成熟库，渲染仍用你现有的交互式单文件 HTML，三个 tab（Data/Model/Predict）保持一致体验：

torchlens（或 torchview） 跑一次前向 → 产出 op 级 DAG + 模块树层级 + 形状 → 映射到你现有的 VisGraph IR（graph.py）。→ 消除根因 1/2/3（分组改由模块树、连边/残差由真值 DAG 结构判定，删掉所有命名白名单与类型清单）。
graphviz dot 对该 DAG 算节点坐标（分层）→ 你的渲染器按坐标画 SVG。→ 消除根因 4（删架构特化路由）。
保留 你的 HTML 渲染层：折叠框、单击聚焦、详情抽屉、按 kind 配色、Data/Predict 两个自研概念流（它们不是 nn.Module，无法被上述库追踪，继续用现渲染器）。
这样：模型流的"追踪+布局"交给成熟库（不再乱套），交互与三流一致性由你现有渲染层保留。

更新后的执行计划（分步、可独立验收）
步骤 1｜选型 spike：在 seg2_5d.yaml / seg3d.yaml 两个"几乎囊括所有情况"的配置上，分别用 torchlens 与 torchview 各跑一次，评估：能否拿到 op 级图+模块层级+形状、对 selfattn/multirf/dict 返回是否正确、程序化取图是否稳定。产出：定选型 + 版本（遵守"以最新官方文档为准"，届时按当时文档 API 写）。验收：两配置都能生成非空、结构正确的图对象。
步骤 2｜引擎图 → VisGraph 映射：写适配层把库的图映射到现有 VisGraph（节点 kind/形状/detail、模块树 parent_id、边 forward/residual/skip）。删除 model_flow.py 里的命名白名单/类型清单/血缘特判。验收：四类配置（unet+selfattn+multirf / unetpp / unet3p / 一个扩散 backbone）跳连数、stage in/out 形状与代码一致（单测断言）。
步骤 3｜graphviz 布局接入：用 dot 算坐标，渲染器按坐标绘边，替换手工车道/rank 路由。验收：四类配置无穿框、无溢出被裁；两次渲染稳定幂等。
步骤 4｜清理与回归：移除死代码，更新依赖声明（requirements）与 README，跑通全量可视化回归；更新受影响的既有测试。验收：git grep 无架构名白名单；三 tab 均正常。
范围：只动 segtask_v1/visualization/ + 依赖声明；不改 models/。新增依赖会在引入前在提交说明里写明用途与必要性。

（推荐）用 torchlens/torchview 作追踪引擎 + graphviz 算布局，但保留你现有的交互式单文件 HTML 渲染层（三 tab/聚焦/详情抽屉不变）
方向已确认：成熟库作追踪引擎 + graphviz 算布局，保留你现有交互式 HTML 渲染层。规划到此完成，方案完全锁定。

最终锁定方案
依赖：允许引入高质量第三方库（追踪引擎 torchlens/torchview 二选一 + graphviz），引入前在提交说明写明用途与必要性。
架构：库负责"模型追踪→op级DAG+模块树+形状"与"分层布局坐标"；你现有渲染层负责交互（三 tab、折叠、单击聚焦、详情抽屉），Data/Predict 两流不动。
彻底删除乱套根源：命名白名单（_INDEXED_SEGS/_top_key/_block_key）、残差类型清单（_residual_block_types/_SHORTCUT_ATTRS）、自研 id 血缘特判、架构特化布局路由。
范围：只改 segtask_v1/visualization/ + 依赖声明；不动 models/。
执行步骤（每轮专注一步、做透）
选型 spike：seg2_5d/seg3d 上对比 torchlens vs torchview，定引擎 + 版本
引擎图 → VisGraph 映射，删命名/类型/血缘特判
graphviz 布局接入，替换手工路由
清理 + 依赖声明 + README + 回归