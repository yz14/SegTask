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


2 生成项目代码审查：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程。

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  

进展：

## 代码审查（TODO 2）
审查计划：R1 数据读取 → R2 数据增强/退化/处理 → R3 模型构建 → R4 训练全流程(含val) → R5 推理全流程 → R6 配置+跨模块一致性+综合。每轮一部分，详细报告在对话中给出，此处仅记简洁摘要。

### R1 数据读取（已完成）
范围：`data/loader.py`、`data/specs.py`、`data/dataset/{core,io,cache,__init__}.py`。
整体质量较高（npz 预烘焙 + fg 索引 + bbox 流式裁剪 + 逐 worker RNG + 策略模式收敛 patch_mode）。关键问题：
- 高 H1：`aug_oversample>1` / `multi_res_scales>1` 发出的超尺寸 max-FOV cube 无 trainer 裁剪/拆视图消费者（core.py:246/510/701 vs gen_trainer 缺失），会与 topology 的 in_channels/D 契约不符 —— 即 TODO 1 多 FOV 输入的落地缺口。
- 高 H2：z_axis/2.5D 模式把整幅切片 resize 到 (pH,pW)（core.py:330-331），破坏/降低 in-plane 分辨率、非方阵扭曲长宽比；对 TODO 1b 面内超分为方向性错误，需保留原生 in-plane 像素的取块模式。
- 高 H3：npz 仅存 HR(`image`)+organ label，无真实 LR 槽位；LR 只能在线合成，无法承载 TODO 1 中「image_dir 低分辨率输入可能存在」的真实配对训练。
- 中 M1 缓存无上限/多缓存 OOM 风险与内存估计漏算 cond；M2 z 轴中心采样无安全夹匯致边界重复切片；M3 面内无过采样余量。
- 低：preprocess_label 死导入、_extract_z_single 死分支、image/label_paths 命名误导、残留空注释标题、极小数据集划分无校验、通配导出。

### R2 数据增强/退化/处理（已完成）
范围：`data/degradation.py`、`data/make_data.py`，及 DataConfig/TaskConfig 相关字段、全仓库增强代码存在性核查。
亮点：area 核↔部分容积效应、各向异性逐轴倍率(厚层只超分z)、blur/decimate 两范式；make_data 原子写/manifest/failures/进程池/fg 索引 seed 一致。关键问题：
- 高 H4：数据契约语义与 TODO 1 相反。现状=「干净 HR(image_dir) + 器官 label(label_dir)，在线合成 LR」；TODO 1 要「LR(image_dir,可缺失) + HR-GT(label_dir)」，image/label 角色互换（make_data.py:115-118 强制 image==label 同 shape；label 仅用于 fg 采样/分层）。需先统一契约（HR-GT 归 npz image、器官掩码作可选 label/region_weight、真实 LR 增独立槽位），否则数据侧全程错位。这是 R1-H3 的根因。
- 高 H5：增强管线整体缺失。全仓库无任何增强/中心裁实现，但 config(aug_oversample_ratio, 注释「增强后中心裁回」)与 degradation.py:13 均承诺其存在 → oversample>1 产出超尺寸 patch 无人裁(呼应 R1-H1)；SR 常用 flip/rot/transpose×8 与「HR→增强→中心裁→退化」既定管线未实现。
- 中 M4：blur 仅 box/area 平均，缺 CT 层敏感度剖面(SSP，近高斯/三角)，厚层真实性不足(TODO 1a)。
- 中 M5：decimate 是抽稀采样而非层内平均，物理与「厚层=平均」不符；且 kept 帧与 align_corners=False 上采样网格错位，VFI 线性基线有相位偏移(degradation.py:129-139)。
- 中 M6：退化为固定单一确定算子，缺随机退化池(Real-ESRGAN/BSRGAN 高阶退化)以适配多样真实 LR。
- 中 M7：npz 未烘焙 spacing，sr_scale 为固定 int，异质层厚下无法 spacing-aware 退化(io.load_nifti_with_spacing 仅推理用)。
- 低：sr_scale 仅整数；噪声仅同方差高斯；_resolve_label_values 大数据集全量扫描。

### R3 模型构建（已完成）
范围：`models/{generation,topology,diffusion,factory,stem,unet,blocks}.py`（并核对 adm/edm2/resnet/convnext/unetpp/unet3p 结构与 config 输出通道派生）。
亮点：topology 单一真相源、generation 统一 forward/restore/degrade、EDM/DDPM 预条件与采样忠于论文、blocks 基础件丰富且高质量、factory 各向异性调度完善；生成输出通道派生正确(num_fg_classes→out_channels)。关键问题：
- 高 H6：build_generation_model 用局部 `spatial_dims=2 if 2_5d else 3`(generation.py:214) 而非 topology.spatial_dims；lift_2_5d_to_3d=True 时 backbone 按 3D 建、退化/打包按 2D → 维度不匹配，生成+lift 路径不可用（bug）。
- 高 H7：缺专用 SISR/VFI 架构（TODO 1 核心要求）。仅通用分割 UNet + ADM/EDM2 扩散；无 EDSR/RCAN/RDN/SwinIR/HAT，无 VFI(光流/形变/核)。residual 仅 VDSR 式全局残差。
- 高 H8：整网锁定同尺寸 in→out(unet.py:480 严格校验+1×1 主头)，只支持 pre-upsampling SISR；blocks 有 PixelShuffle/ICNR/CARAFE/DySample 却未接成真正上采头，无 EDSR/ESPCN 式 post-upsampling SR，对 D→2D/4D 实际增采样是结构限制。
- 中 M8：扩散数据尺度不符（minmax[0,1] 均值~0.5 vs DDPM/EDM 假设零中心[-1,1]）；建议扩散用 zscore/[-1,1]。
- 中 M9：3D 扩散不可用(diffusion 仅 2D + ADM/EDM2 断言 2.5D)。
- 中 M10：EDM sampler=="ddim" 实为关 Heun 的确定性 Euler，命名误导。M11：_StatefulStageBuilder 有序可变状态脆弱。
- 低：DualConvStem 首层实为 7×7 且残留犹豫注释；ADM/EDM2 忽略 decoder_blocks_per_stage；keep_native_multi_res 无消费者；2.5D 下 CARAFE/DySample 不可用。
- 业界(2026)：通用 SISR 优先 EDSR/RCAN/RDN/SwinIR/HAT（3D 变体可作 backbone）；自然 VFI(RIFE/FILM) 直接迁移 CT 效果差，宜借鉴 CT 专用切片插值(SAINT/I3Net/TVSRN/ArSSR/跨视图纹理迁移/空频Swin/样条+数据保真)。

### R4 训练全流程含val（已完成）
范围：`trainer/{gen_trainer,optim,amp,checkpoint,memory}.py`、`losses/recon.py`、`utils.py`、`train.py`。
亮点：AMP+累积+裁剪模式正确、损失模块自身 .float() 上采(AMP 安全)、EMA 缓存+try/finally 恢复、WarmupScheduler 漂移检测、checkpoint 多格式兼容、SR 损失工具箱(Charbonnier/SSIM/梯度/区域加权)合理。无致命 bug。关键问题：
- 中 M12：梯度累积下 warmup/调度 horizon 计算错误。warmup_steps/total_steps 按 batch 计(gen_trainer.py:63-65)，但 scheduler.step 按优化器步触发 → accum>1 时 warmup 时长放大 accum 倍、cosine/poly horizon 错位；应换算为优化器步(//accum)。accum==1 正确。
- 中 M13：验证为 patch 级(非整卷，与部署不一致)；且 clamp(0,1) 与 psnr/ssim 默认 data_range=1.0 硬编码 [0,1]，zscore 下口径错(训练损失用 2.0，自相矛盾)；PSNR 用整 batch MSE→dB 非逐图平均，有系统偏差。
- 中 M14：无续训/周期保存/早停(TrainConfig 有字段但未实现)；_save_best 不存 optimizer/scheduler/scaler → 无法断点续训。
- 低：one_cycle 与外层 warmup 冲突未禁止；每步 loss.item() 同步拖慢；NaN 损失静默跳过；compute_loss_fp32/memory.estimate 死代码；_save_best pickle 整个 cfg；仍 collate 未用 label + 扩散 loss 不支持 weight_map；train.py 仅 override 时才 sync/validate(待 R6 核实)。
- 业界：SR 应整卷逐图 PSNR/SSIM 并匹配 data_range；可加 LPIPS/频域损失；续训需存 optimizer/scheduler/scaler。

### R5 推理全流程（已完成）
范围：`predictor/gen_predictor.py`、`predict.py`。
亮点：权重选择(auto/ema/online)+<50% 加载保护、几何 CopyInformation 保留、2.5D 尾窗对齐、cond 路径匹配。关键问题：
- 高 H9：训练-推理退化/网格不一致。训练 degrade=下采+上采回 HR 网格；推理直接 restore 不 degrade、不上采(gen_predictor.py:112-117)，而整网同尺寸 in→out → 喂真实厚层(D/2)输出仍 D/2、z 维不增采。真实 LR 须先重采样到 HR 网格再入网，代码/文档未处理。这是能否产出真实超分的核心缺口。
- 中 M15：3D 模式(z_axis/cubic/whole)整卷单次前向(gen_predictor.py:66-71)，真实 CT 易 OOM 且与训练 patch 尺度不一致(仅 whole 一致)；缺 3D 滑窗聚合。
- 中 M16：2.5D slab 非重叠(步长=slab深)+均匀计数平均 → 交界接缝；应重叠+高斯加权融合。
- 中 M17：短体零填充(vs 训练 edge_pad)边界策略不一致。
- 中 M18：输出未反归一化，保存 [0,1]/zscore 值而非 HU(gen_predictor.py:123)，丢失标定。
- 低：扩散推理无种子不可复现；默认 input=image_dir(HR，语义混淆)+递归收集误收 nii；全程 fp32。
- 业界：滑窗重叠+高斯加权融合(nnU-Net/MONAI)；推理输入须与训练 LR 网格/退化一致；输出须反归一化回 HU。

### R6 配置+跨模块一致性+综合（已完成）
范围：`config/{validation,io,dataclasses}.py` + 全局跨模块契约。
配置层：validate 对 model/task/2.5d/data 覆盖细致、别名/弃用派生键处理优雅、load_config 统一 sync+validate(R4-L20 解除)。问题：
- 中 M19：校验放行了 trainer/predictor 未实现的组合(keep_native_*/multi_res>1/aug_oversample>1) → 运行期形状崩溃；实现补齐前应拒绝或强告警。
- 低 L24：diffusion+arch=unet 未在 validate 拦截(build 期才报)；one_cycle+warmup 未禁；zscore 允许但 val/推理/扩散多处硬编码 minmax；gen_out_channels 疑似无用；io.py 注释重复；topology docstring 引用不存在的 trainer.pipelines.factory。
跨模块契约漂移：①多FOV/oversample/中心裁 —— config/dataset/topology 就绪但 trainer/predictor 无消费者(最严重)；②spatial_dims 在 generation/trainer/predictor 各自本地重算、绕过 topology(H6 根因)；③数据契约语义(image=HR vs TODO label=HR-GT)；④归一化域不统一；⑤训练vs推理路径漂移(在线退化/边界/patch-整卷)。

## 综合结论与整改路线图
gentask 工程质量高但仍带分割起源骨架 + 「HR+器官label→合成LR、同尺寸pre-upsampling」假设链。按优先级：
- 第1层(阻塞 TODO 1，必须先定/补)：①数据契约(H4/H3/M7 image-label角色/真实LR槽位/spacing) ②多FOV管线(H1/H5/M19) ③增采架构+推理网格(H8/H9) ④缺经典 SISR/VFI(H7)。
- 第2层(正确性 bug)：H6 spatial_dims 绕过 topology；M12 累积 LR 调度错位；M5 decimate 物理/相位。
- 第3层(真实性/效果)：SSP 厚层核(M4)/随机退化(M6)/spacing 感知；扩散数据尺度(M8)；z 采样重复(M2)/面内保原生像素取块(H2)；整卷逐图指标(M13)/滑窗重叠高斯融合(M16)/反归一化回HU(M18)。
- 第4层(工程)：续训(M14)/3D滑窗(M15)/缓存(M1)/死代码/命名/stale docs。
建议顺序：先定契约 → 补多FOV+中心裁+几何增强管线 → 落地经典 SISR+真正增采头(先 1b 面内超分风险最低，再 1a z-SISR/VFI 借鉴 CT 专用切片插值) → 修 bug/真实性/评估推理 → 补工程完整性。

审查完成（R1–R6 全部完成；审查阶段未改动任何项目代码/文档，仅记录进展）。

## 复检对账（2026-07-08，对照当前代码逐条核验）
自审查以来代码已大量整改（新增 data/augment.py、trainer/views.py、trainer/pipelines/{base,vanilla,stacked,native_d,factory}.py 等）。逐条核验结果如下。

### 已修复（核验确认）
- H1+H5 多FOV/oversample/增强管线：完整落地。GPU 增强（augment.py：flip/affine+elastic 融合单次 warp/grid-dropout + 强度类，image/cond/wmap 同步）在 max-FOV cube 上执行 → pipelines 中心裁过采样余量 → 逐视图拆分/resize/打包（vanilla/stacked/native_d 三管线，factory 与 build_topology 决策对齐）。R6-①契约漂移同步解除。
- H6 spatial_dims：generation.py:227 改由 topology.spatial_dims 单一真相源；sync() 统一写回 model._spatial_dims。R6-②解除。
- M1 缓存：VolumeCache 有 cache_max_volumes 上限（LRU）。
- M2 z 采样：z 抽取走 edge-pad 路径（_extract_z_patch_padded），不再中心夹带来的重复/错位。
- M5 decimate：改相位对齐线性插值（_phase_aligned_linear_upsample），保留帧逐体素精确保留（lr[k*sc]==hr[k*sc]），无相位偏移；噪声施加在保留帧（LR 域）。
- M12 累积调度：horizon 按优化器步 ceil(steps/accum) 计（gen_trainer.py:79-93）；one_cycle 自动禁用外层 warmup（原 L 项 one_cycle 冲突一并解除）。
- M19 校验放行：validate 已补 keep_native_view_depth / keep_native_multi_res / multi_res 组合、whole 模式限制等；diffusion×多视图在 trainer fit 入口显式拒绝。
- L 项部分：diffusion+arch=unet 已在 validate 拦截；_extract_z_single 已被实际调用（不再是死分支）。

### 部分修复
- H3/H4 数据契约：npz 新增 cond 槽位（cond_dirs → 与 image 同 bbox 裁剪、独立归一化、训练/推理全链路可用），真实 LR 可作为条件体进入模型。但契约主体未变：image 仍必须是干净 HR（make_data 仍强制 image==label 同 shape）、训练输入 LR 仍 100% 在线合成——「真实 LR 直接作为网络输入」的配对训练仍不支持，仅有 cond 旁路。
- M9 3D 扩散：仍不可用，但已由 validate 显式拦截（diffusion 限 2.5D、禁 lift），从"静默崩溃"降级为"显式拒绝"。
- M13 验证口径：PSNR 改逐图平均（recon.py:163-171，SR 标准口径）；data_range 与 normalize 匹配（minmax=1.0 / zscore=2.0），zscore 不再错误 clamp。仍是 patch 级验证（非整卷、与部署不一致）——剩余部分未做。

### 仍未处理（对齐原编号）
- 阻塞 TODO 1：H2（z_axis/2_5d 面内整幅 resize 到 patch 尺寸，不保留原生面内像素；1b 需用 cubic 模式规避）、H7（无专用 SISR/VFI 架构，factory 仍仅 unet/adm/edm2）、H8（整网同尺寸 in→out，无 post-upsampling 增采头）、H9（推理直接 restore，真实厚层输入不重采样到 HR 网格、z 维不增采——核心缺口）。
- 真实性/效果：M4（无 SSP/高斯层敏感度剖面核）、M6（无随机退化池）、M7（npz 不烘 spacing、sr_scale 固定 int）、M8（扩散数据尺度 minmax[0,1] vs 零中心假设，validate 亦无警告）、M3（面内无过采样余量，z_axis/2_5d 仅 z 有余量；grid_sample border padding 部分缓解）。
- 工程：M10（EDM sampler=='ddim' 实为关 Heun 的 Euler，命名误导仍在）、M14（TrainConfig 有 resume/save_every/early_stopping/val_every 字段但 gen_trainer 全未实现；_save_best 仍不存 optimizer/scheduler/scaler，无法断点续训）、M15（3D 模式仍整卷单次前向，无 3D 滑窗）、M16（2.5D 滑窗步长=slab 深、除尾窗外不重叠+均匀平均，无高斯融合）、M17（短体零填充 vs 训练 edge_pad）、M18（输出不反归一化回 HU）。
- 低项：每步 loss.item() 同步、NaN 损失静默跳过、扩散推理无种子、predict 默认 input=image_dir（HR）语义混淆、compute_loss_fp32/estimate_train_memory 死代码、preprocess_label 死导入 等大多仍在。

### 新发现（本轮补充审查）
- 高 H10：多视图训练-推理不对称。多视图（multi_res_scales>1 / keep_native_view_depth）训练侧已由 pipelines 完整支持，但 GenerationPredictor 完全按单视图工作（2.5D 只喂 depth=patch_size[0] 的单 slab、3D 整卷单视图），多视图模型 in_channels 不匹配 → 推理直接崩溃；predict 入口与 validate 均无拦截。当前多视图"只能训、不能推"。修法：predictor 按 view_sizes 从整卷构造多 FOV 视图（与训练 pipelines 同几何），或至少 validate 拒绝多视图配置进入 predict。
- 低 L25：gen_trainer 为单进程单卡（无 DDP），train.gpus 等多卡字段对 gentask 无效；val_every 字段同样未消费（每 epoch 必验）。属工程完整性，与 M14 同层。

### 复检后的路线图（更新）
- 第1层（阻塞 TODO 1）：仅剩 ③增采架构+推理网格（H8/H9）与 ④经典 SISR/VFI（H7）；①数据契约剩「真实 LR 作输入」半截（H4 残留 + M7 spacing），②多FOV管线已完成但需补 H10 推理侧对称。
- 第2层（正确性）：H10（新）＞ M8 扩散尺度 ＞ M10 命名。
- 第3层（真实性/效果）：M4/M6/M7、H2 面内原生像素、M13 整卷验证、M16 高斯融合、M18 反归一化。
- 第4层（工程）：M14 续训/周期保存/早停、M15 3D 滑窗、M17、L 项清理。
建议顺序不变：先补 H8/H9+H10（推理能真正增采并支持多视图）→ 落地经典 SISR（1b 先行）→ 真实性/评估 → 工程完整性。


3 分割训练在服务器报错

data:
  npz_dir: "/data0/yzhen/data/tx_ves/npz_data"

  label_values: [0, 1]
  num_classes : 2

  patch_size: [12, 320, 400]  ##################################################################
  patch_mode: "2_5d"          ##################################################################

  multi_res_scales    : [1.0, 1.5, 2.0]
  aug_oversample_ratio: 1.5

  keep_native_view_depth: true

  z_boundary_mode: "edge_pad"

  intensity_min: -1024.0
  intensity_max: 1024.0
  normalize    : "minmax"

  val_ratio       : 0.2
  split_seed      : 42
  stratified_split: true

  batch_size     : 8
  num_workers    : 24
  prefetch_factor: 8
  pin_memory     : true

  foreground_oversample_ratio: 0.5
  samples_per_volume         : 8

  cache_mode       : "memory"
  cache_max_volumes: 24


augment:
  enabled         : true
  intensity_clamp : true
  wmap_interp_mode: "nearest"

  random_flip_prob: 0.2
  random_flip_axes: [2, 3, 4]

  random_affine_prob          : 0.3
  random_rotate_range         : [-20.0, 20.0]
  random_scale_range          : [0.80, 1.2]
  random_translate_range      : [-0.1, 0.1]
  random_affine_aspect_correct: true

  elastic_deform_prob : 0.0
  elastic_deform_sigma: 5.0
  elastic_deform_alpha: 1.0

  random_brightness_prob : 0.3
  random_brightness_range: [-0.1, 0.1]

  random_contrast_prob : 0.3
  random_contrast_range: [0.8, 1.2]

  random_gamma_prob : 0.2
  random_gamma_range: [0.8, 1.2]

  gaussian_noise_prob: 0.2
  gaussian_noise_std : 0.09

  gaussian_blur_prob : 0.2
  gaussian_blur_sigma: [0.5, 1.5]

  simulate_lowres_prob: 0.2
  simulate_lowres_zoom: [0.5, 1.0]


model:
  arch                    : "unet"
  backbone                : "resnet"
  encoder_channels        : [64, 128, 256, 512, 512]
  blocks_per_level        : 2
  encoder_blocks_per_stage: [2, 2, 2, 2, 2]
  decoder_blocks_per_stage: [2, 2, 2, 2]
  block_type              : "basic"

  norm_type               : "batch"
  norm_groups             : 8
  activation              : "leakyrelu"
  dropout                 : 0.0
  drop_path_rate          : 0.0

  stem_mode               : "dual"
  decoder_type            : "unet"
  downsample_mode         : "conv"
  downsample_strides      : []
  anisotropic_pooling     : false
  upsample_mode           : "trilinear"
  upsample_norm_act       : true
  skip_mode               : "cat"

  attention_type          : "none"  ##################################################################
  se_reduction            : 16
  skip_attention          : false
  attn_gate_norm          : "batch"

  deep_supervision        : true
  aux_seg_supervision     : true
  aux_head_mode           : "conv"

  stem_fusion_mode        : "multi_stem_proj"
  lift_2_5d_to_3d         : false

  grad_checkpointing      : true
  grad_ckpt_encoder_stages: []

  multirf_enabled       : true
  multirf_dilations     : [1, 2, 3]
  multirf_mode          : "split"
  multirf_fusion        : "concat_proj"
  multirf_axes          : "hw"
  multirf_encoder_stages: [0, 1, 1, 1, 1]
  multirf_decoder_stages: [1, 1, 1, 0]
  multirf_branch_norm_act: true

  selfattn_enabled       : true
  selfattn_type          : "softmax"
  selfattn_num_heads     : 4
  selfattn_head_dim      : -1
  selfattn_zero_init     : true
  selfattn_rope          : false
  selfattn_ffn           : false
  selfattn_ffn_ratio     : 4.0
  selfattn_encoder_stages: [0, 0, 0, 0, softmax]  ##################################################################
  selfattn_decoder_stages: [0, 0, 0, 0]


loss:
  name                    : "dice_focal"
  compound_weights        : [1.0, 1.0]
  class_weights           : [1.0]
  region_weights          : [0.0, 4.0]
  slice_loss_reduction    : "per_volume"
  deep_supervision_weights: [1.0, 0.5, 0.25, 0.125]
  aux_supervision_weights : [0.5, 0.5]

  batch_dice              : true
  ignore_empty            : false

  dice_smooth : 1.0e-5
  dice_squared: false

  focal_alpha: 0.5
  focal_gamma: 2.0

  tversky_alpha: 0.3
  tversky_beta : 0.7


train:
  epochs                       : 1000
  seed                         : 42
  deterministic                : false
  gpus                         : [1,3]
  output_dir                   : "outputs/ves_multirf2d_mmore_attn2"  ##################################################################

  ddp_find_unused_parameters   : false
  ddp_static_graph             : true
  ddp_gradient_as_bucket_view  : true
  ddp_master_port              : 0
  ddp_scale_dataloader_per_rank: true
  ddp_timeout_minutes          : 30
  zero_redundancy_optimizer    : true

  optimizer                    : "adamw"
  adamw_fused                  : true
  lr                           : 1.0e-4
  weight_decay                 : 1.0e-4

  scheduler                    : "cosine"
  warmup_epochs                : 5
  warmup_lr                    : 1.0e-6
  cosine_min_lr                : 1.0e-6

  grad_accum_steps             : 2
  grad_clip_norm               : 12.0

  use_amp                      : true
  amp_dtype                    : "auto"
  compile_mode                 : "default"
  channels_last                : false
  cuda_expandable_segments     : false
  val_empty_cache              : null

  use_ema                      : true
  ema_decay                    : 0.999
  ema_warmup                   : true
  ema_device                   : ""
  swa_enabled                  : false  ##################################################################
  swa_start_ratio              : 0.75
  swa_bn_update_steps          : 50

  val_every                    : 1
  val_metric_mode              : "high"
  val_metric_bbox_crop         : true
  surface_dice_tolerance       : 1
  surface_dice_weight          : 0.5
  save_best_preset             : "vessel"
  early_stopping               : 0

  save_every                   : 10
  save_keep_last               : 3
  save_async                   : true

  log_every                    : 10
  vis_every                    : 10

  resume                       : ""
  pretrain                     : ""
  pretrain_strict              : false
  pretrain_load_ema            : true


predict:
  z_overlap              : 0.5
  blend_mode             : "gaussian"
  batch_size             : 2

  tta_flip               : true
  tta_batch_size         : 1
  threshold              : 0.5

  acc_dtype              : "fp32"
  vol_dtype              : "fp32"
  accumulate_on_cpu      : false

  cudnn_benchmark        : false
  use_inference_mode     : true
  channels_last          : false

  z_interleave_enabled   : false
  z_interleave_thresholds: [1.0, 1.5, 3]
  z_interleave_factors   : [4, 3, 2, 1]

  output_dir             : ""
  save_probabilities     : false


vis:
  enabled          : true
  output_dir       : ""
  filename         : "pipeline_vis.html"
  flows            : ["data", "model", "predict"]
  trace_shapes     : true
  max_detail_params: 200


monitor:
  enabled            : true
  output_dir         : ""
  filename           : "training_monitor.html"
  update_every       : 1
  auto_reload_seconds: 10
  run_name           : ""
  compare_runs                 : []
  health_monitor               : true
  health_grad_norm_when_no_clip: true
  health_update_ratio          : false

est$ python -m segtask_v1.train --config configs/segtest0.yaml
[2026-07-08 15:05:43] INFO __mp_main__: DDP launched: world_size=2 on physical GPUs [1, 3] (MASTER_PORT=38171).
[2026-07-08 15:05:43] INFO segtask_v1.utils: Seed set to 42 (deterministic=False)
[2026-07-08 15:05:43] INFO segtask_v1.data.loader: DDP dataloader scaling: num_workers 24 -> 12 per rank (world_size=2; aggregate 24 workers across ranks matches the single-GPU baseline). Per-worker LRU cache is unchanged, so aggregate cache RAM also matches single-GPU. Set train.ddp_scale_dataloader_per_rank=false to keep full num_workers on every rank.
[2026-07-08 15:05:43] INFO segtask_v1.data.loader: Primary (gold) training source: npz packages under /data0/yzhen/data/tx_ves/npz_data (suffix=.npz). NIfTI fields image_dir/label_dir/bbox_dir/region_weight_dir are consumed only by make_data when the npz cache must be built.
[2026-07-08 15:05:43] INFO segtask_v1.data.loader: Discovered 110 npz package(s) under /data0/yzhen/data/tx_ves/npz_data.
[2026-07-08 15:05:43] INFO segtask_v1.data.loader: Label values: [0, 1], num_classes: 2, num_fg: 1
[2026-07-08 15:05:55] INFO segtask_v1.data.loader: Stratified split: 88 train, 22 val (strata sizes: {'1': 110})
[2026-07-08 15:05:55] INFO segtask_v1.data.specs: Using 2_5D patch mode (oversample=1.50, scales=[1.0, 1.5, 2.0], n_views=3, max_scale=2.00, z_boundary=edge_pad) — SINGLE max-FOV z-cube extraction; trainer crops+resizes per view before forward.
[2026-07-08 15:05:55] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 88 npz packages...
[2026-07-08 15:05:55] INFO segtask_v1.data.dataset: NPZ index built: 88 volumes, 20793/25183 foreground slices
[2026-07-08 15:05:55] INFO segtask_v1.data.dataset: Loading pre-computed fg indices from 22 npz packages...
[2026-07-08 15:05:55] INFO segtask_v1.data.dataset: NPZ index built: 22 volumes, 5279/6409 foreground slices
[2026-07-08 15:05:55] INFO segtask_v1.data.loader: DDP DistributedSampler: rank=0/2, ~352 samples/rank (train).
[2026-07-08 15:05:55] INFO segtask_v1.data.loader: DataLoader: batch_size=8, num_workers=12 (per rank), pin_memory=True, persistent_workers=True, prefetch_factor=8
[2026-07-08 15:05:55] INFO segtask_v1.data.loader: Volume cache estimate: ~233.06 MiB per volume (image fp32 + label int16 + region_weight fp32, bbox-cropped); effective cap=24, num_workers=12 => up to ~65.55 GiB RAM (all workers, caches only; transient decode peaks add ~93.22 MiB/worker) [per rank; x2 ranks => ~131.10 GiB machine-wide aggregate].
[2026-07-08 15:05:55] INFO segtask_v1.models.factory: MultiRF ENABLED: dilations=[1, 2, 3], mode=split, fusion=concat_proj, axes=hw, enc_stages=[0, 1, 1, 1, 1], dec_stages=[1, 1, 1, 0]
[2026-07-08 15:05:55] INFO segtask_v1.models.factory: SelfAttention ENABLED: default_type=softmax, num_heads=4, head_dim=-1, zero_init=True, enc_types=[None, None, None, None, 'softmax'], dec_types=[None, None, None, None]
[2026-07-08 15:05:55] INFO segtask_v1.models.factory: Built UNet3D [resnet/basic, decoder=unet, preset=none]: enc=24.66M, dec=20.97M, total=48.82M, channels=[64, 128, 256, 512, 512], enc_blocks=[2, 2, 2, 2, 2], dec_blocks=[2, 2, 2, 2], out_classes=12 (fg=1, res=1), stem=dual(stride=1, n_views=3, fusion=multi_stem_proj), down=conv, up=trilinear, skip=cat, attn=none, skip_attn=False, ds=True, aux_seg=True(n_aux_heads=2, mode=conv), grad_ckpt=True
[2026-07-08 15:05:55] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-07-08 15:05:55] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-07-08 15:05:55] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
[2026-07-08 15:05:57] INFO segtask_v1.visualization: Pipeline visualization HTML written: outputs/ves_multirf2d_mmore_attn2/visualization/pipeline_vis.html
[2026-07-08 15:05:57] INFO __mp_main__: Pipeline visualization written to: outputs/ves_multirf2d_mmore_attn2/visualization/pipeline_vis.html
[2026-07-08 15:05:57] INFO segtask_v1.trainer.pipelines.factory: ViewPipeline selected: Slab2_5DNativeDPipeline (patch_mode=2_5d, n_views=3)
[2026-07-08 15:05:57] INFO segtask_v1.trainer.pipelines.slab25d: Aux seg supervision: ENABLED (native depth), n_aux_views=2, per-view depths=[18, 24], weights=[0.5, 0.5], fusion=multi_stem_proj
[2026-07-08 15:05:57] INFO segtask_v1.trainer.pipelines.slab25d: Trainer keep_native_view_depth=True: max-FOV crop D=24, per-view depths=[12, 18, 24], channel layout sum=54.
/data0/yzhen/timm_test/segtask_v1/trainer/optim.py:59: DeprecationWarning: `TorchScript` support for functional optimizers is deprecated and will be removed in a future PyTorch release. Consider using the `torch.compile` optimizer instead.
  from torch.distributed.optim import ZeroRedundancyOptimizer
/data0/yzhen/timm_test/segtask_v1/trainer/optim.py:59: DeprecationWarning: `TorchScript` support for functional optimizers is deprecated and will be removed in a future PyTorch release. Consider using the `torch.compile` optimizer instead.
  from torch.distributed.optim import ZeroRedundancyOptimizer
[2026-07-08 15:05:57] INFO segtask_v1.trainer.optim: ZeroRedundancyOptimizer enabled: AdamW state sharded across ranks.
[2026-07-08 15:05:57] INFO segtask_v1.trainer.trainer: amp_dtype='auto' resolved to 'bfloat16' (device=cuda:1).
[2026-07-08 15:05:57] INFO segtask_v1.trainer.trainer: Compiling model with mode='default'
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: DDP enabled: rank=0/2, device=cuda:1, find_unused_parameters=False, gradient_as_bucket_view=True, static_graph=True. Training grads all-reduce per backward. Note: math-equivalence to single-GPU under grad-accum holds for per-sample separable losses (BCE/Focal/per-sample Dice); batch-pooled ratio losses (batch_dice/Tversky/GDL) pool over the per-rank micro-batch, so their effective statistics window shrinks with accum/ranks (approximate, not strictly equivalent).
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Validation metric mode: high (evaluator=VolumeValEvaluator)
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Training monitor enabled → metrics: outputs/ves_multirf2d_mmore_attn2/monitor | dashboard: outputs/ves_multirf2d_mmore_attn2/training_monitor.html
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: ============================================================
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Training: 1000 epochs, device=cuda:1
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Model params: 48.82M
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Static GPU mem (persistent, excl. activations): param=186.3 + grad=186.3 + optim(AdamW,2x)=186.3 + ema=186.4 = 745.2 MiB (real peak reported per-epoch as 'GPU peak')
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Gradient checkpointing: ON — encoder/decoder activations recomputed in backward (~+20-33%% compute, much lower activation memory; numerics unchanged vs OFF).
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: CUDA mem at training start: allocated=566.8 MiB, reserved=768.0 MiB (model already on device; activations/workspace will add on top during forward).
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Train batches: 44, Val batches: 6
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: AMP=True (dtype=auto, resolved=bfloat16, scaler=False), EMA=True (decay=0.9990)
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Grad accum=2, Effective batch=16
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: Pipeline=Slab2_5DNativeDPipeline | n_views=3, n_aux_views=2, num_res_groups=1, slab_depth=12 | fg_classes=1, Loss=dice_focal
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: torch.compile mode: default (active=True)
[2026-07-08 15:06:00] INFO segtask_v1.trainer.trainer: ============================================================
W0708 15:06:47.291000 140518365775680 torch/multiprocessing/spawn.py:146] Terminating process 2497284 via signal SIGTERM
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/data0/yzhen/timm_test/segtask_v1/train.py", line 276, in <module>
    main()
  File "/data0/yzhen/timm_test/segtask_v1/train.py", line 245, in main
    mp.spawn(
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 282, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 238, in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 189, in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException:

-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 76, in _wrap
    fn(i, *args)
  File "/data0/yzhen/timm_test/segtask_v1/train.py", line 189, in _train_worker
    _build_and_fit(cfg, device)
  File "/data0/yzhen/timm_test/segtask_v1/train.py", line 105, in _build_and_fit
    best_metrics = trainer.fit()
                   ^^^^^^^^^^^^^
  File "/data0/yzhen/timm_test/segtask_v1/trainer/trainer.py", line 397, in fit
    train_metrics = self._train_epoch(epoch)
                    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/timm_test/segtask_v1/trainer/trainer.py", line 811, in _train_epoch
    pred = self.fwd_model(image)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1636, in forward
    else self._run_ddp_forward(*inputs, **kwargs)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1454, in _run_ddp_forward
    return self.module(*inputs, **kwargs)  # type: ignore[index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 433, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1110, in __call__
    return hijacked_callback(
           ^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 948, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 472, in __call__
    return _compile(
           ^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_utils_internal.py", line 84, in wrapper_function
    return StrobelightCompileTimeProfiler.profile_compile_time(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_strobelight/compile_time_profiler.py", line 129, in profile_compile_time
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 817, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 231, in time_wrapper
    r = func(*args, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 636, in compile_inner
    out_code = transform_code_object(code, transform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1185, in transform_code_object
    transformations(instructions, code_options)
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 178, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 582, in transform
    tracer.run()
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2451, in run
    super().run()
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 893, in run
    while self.step():
          ^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 805, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2642, in RETURN_VALUE
    self._return(inst)
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2627, in _return
    self.output.compile_subgraph(
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/output_graph.py", line 1123, in compile_subgraph
    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/output_graph.py", line 1318, in compile_and_call_fx_graph
    compiled_fn = self.call_user_compiler(gm)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 231, in time_wrapper
    r = func(*args, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/output_graph.py", line 1409, in call_user_compiler
    raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/output_graph.py", line 1390, in call_user_compiler
    compiled_fn = compiler_fn(gm, self.example_inputs())
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data0/yzhen/py3/envs/llm/lib/python3.12/site-packages/torch/_dynamo/backends/distributed.py", line 490, in compile_fn
    raise NotImplementedError(
torch._dynamo.exc.BackendCompilerFailed: backend='compile_fn' raised:
NotImplementedError: DDPOptimizer backend: Found a higher order op in the graph. This is not supported. Please turn off DDP optimizer using torch._dynamo.config.optimize_ddp=False. Note that this can cause performance degradation because there will be one bucket for the entire Dynamo graph. Please refer to this issue - https://github.com/pytorch/pytorch/issues/104674.

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True