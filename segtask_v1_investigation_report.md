# 分割主线（segtask_v1）现状调研报告 — Phase 1（只调研，不动代码）

> 依据 TODO 第 2 项：按「数据读取 / 模型构建 / 数据增强·处理 / 训练全流程(含 val) / 推理全流程」五部分，
> 审查代码·算法·设计·架构的正确性、合理性、优化空间，以及可借鉴/适配/新增的高质量内容。
> 本轮**不修改任何代码/文档**。所有结论均基于对 `segtask_v1` 源码的逐文件精读，并结合
> `README.md` / `docs/DESIGN.md` 的设计契约。

---

## 0. 总体架构评价

`segtask_v1` 是仓库的**基建真相源**，2.5D/3D 医学分割的完整工程主线，其他任务（ssltask/clstask/dettask/gentask）皆由此改造。整体质量在开源医学分割实现中属**参考级**，核心优点：

- **几何与实现分离**：所有训练/推理几何派生量收敛到单一真相源 `ModelTopology`（`models/topology.py`），`config.sync`、`factory.build_model`、`trainer.pipelines.factory`、`Predictor` 全部从这里读，杜绝了「同一派生量多处重算、新增 patch_mode 漏改一处」的经典缺陷。
- **策略对象化**：patch_mode 的分支被收敛到 `data/specs.py::build_data_spec`（data 侧唯一 if/elif）与 `trainer/pipelines/*`（监督侧策略对象），Trainer 主循环无模式分支。
- **npz 预烘 + 逐 worker LRU 卷缓存**：`make_data.py` 把 NIfTI 烘成可 mmap 的 ZIP_STORED npz（含逐类 fg 索引、精确 label 计数、物理 spacing、几何指纹），配 `VolumeCache` LRU（`dataset.py:508-553`），IO 路径成熟。
- **训练循环工程健壮性极高**：优化步为单位的调度、梯度累积尾组补齐、**跨 rank 非有限 loss/grad 守护并跳步不推进 scheduler/EMA**、延迟 `.item()` 削同步点、DDP `no_sync`、异步 checkpoint、EMA/SWA、ZeRO-1、channels_last、grad-checkpoint——这正是 ssltask 报告中 SSL trainer 缺失、需要「继承」的那套实现。
- **训练/推理几何严格镜像**：`Predictor` 与 dataset/trainer 共用 topology 与同名几何量，spacing 归一化的正/逆变换、z-interleave、边界处理口径一致。

下面逐域给出发现的问题与优化点，按 `[高]/[中]/[低]` 标注。**需强调：五域整体正确，未发现影响正确性的实质缺陷**；下列多为设计取舍说明与 opt-in 增强建议。

---

## 1. 数据读取（Domain 1）

**文件**：`data/dataset.py`、`data/loader.py`、`data/specs.py`、`data/make_data.py`、`data/mixed_sampler.py`

### 现状（正确/合理之处）
- **单分辨率最大 FOV cube 读取原则**：dataset 只发一份 max-FOV cube，多分辨率/多视图裁剪推迟到 trainer，避免多次 zoom 的高频损失、保证同体素跨 view 几何一致（`dataset.py:808-860`）。
- **确定性验证采样**：训练用逐 worker 流式 RNG，验证用「样本序号派生的确定性 RNG」+ 可选 `val_grid_coverage`（z 轴等距 bin 中心 / cubic 用 Halton(2,3,5) 低差异序列铺满安全中心域，`dataset.py:648-654,862-882,1126-1139`），使 save_best/early-stop 不被采样噪声驱动——比多数实现更严谨。
- **逐类前景索引 + 类均衡采样**：make_data 逐类 argwhere（每类独立 cap，避免稀有小结构被大器官淹没）并写 `fg_coords_cls`；采样时「先均匀选类再选点/切片」（`make_data.py:58-107`、`dataset.py:876-880,1146-1148`）。
- **物理几何 fail-fast**：make_data 只读头校验 label/bbox/rw 与 image 共 spacing/origin/direction，不共坐标系直接报错（`make_data.py:117-139`）——防止「shape 相等但未共注册」的静默错配。
- **流式 bbox 裁剪读取**：`load_nifti_cropped` 用 sitk `SetExtractIndex/Size` 只读 ROI，峰值内存与裁后体积同量级（`dataset.py:176-229`）。
- **切分策略完备**：随机 / 分层（主前景类）/ 组级（患者级正则，防泄漏并断言 train/val 组互斥，`loader.py:411-458`）；label_values 与逐样本计数可从 npz meta 快路读取，启动期免解码 label 卷。
- **双批混合采样**：`MixedBatchSampler`（金/粗定配比、粗标每 epoch 消费一遍、金标准循环过采样、DDP 各 rank 同 seed+epoch 不相交切分，`mixed_sampler.py:87-215`），DDP 语义严谨。
- **DDP 验证** `ValBatchShardSampler` 按 batch 块不相交切分、worker 只产本 rank batch、无 padding/重复（`loader.py:38-68`）。
- **缓存足迹估计 + OOM 告警**（`loader.py:879-928`）：`cache_max_volumes=0` 且卷×worker 多时给出建议 cap，工程细节到位。

### 问题与优化空间

**[中] 1.1 `z_axis`/`2_5d` 模式把 H/W 一次性 resize 到 `patch_size`（`dataset.py:844-845`）。**
z 轴保留全 z-FOV，但面内直接缩到 `(pH,pW)`。对大面内体积（如 512×512 CT）这会**牺牲面内分辨率**，成为细小结构（细血管、微结节）分割精度的上界。训练与推理口径一致（`sliding.py` 面内同样 resize 回原尺寸），不是 bug，但属**架构级上界**。若目标含细小面内结构，建议评估改用 `cubic`（面内也滑窗、不缩放）或提高 `patch_size` 的面内尺度。

**[中] 1.2 cubic 前景中心被夹到 `_safe_center_range`（`dataset.py:1126-1155`）。**
为避免 max-FOV cube 越界产生 >50% 边界复制体素，fg 坐标被 clip 到安全区间。**靠近体积边界的小前景**会被系统性地往内拉，patch 中心偏离真实前景中心，轻微偏移训练分布。可接受（边界填充更糟），但对「贴边器官/血管末梢」是潜在偏置点，值得在稀疏边界目标上验证。

**[低] 1.3 分层划分的「主前景类」= argmax 体素计数（`loader.py:461-473,506-511`）。** 多器官/多标签卷被归约为单一层（体素最多的类），分层的类均衡性对真正多标签数据偏弱。可接受，属常见简化。

**[低] 1.4 逐 worker LRU 卷缓存内存随 worker 数线性放大。** 已有估计与告警（1.0 现状），非缺陷；大语料建议显式设 `cache_max_volumes` 或 `cache_mode:"none"` 依赖 OS page cache（npz 为 ZIP_STORED 可跨 worker 共享）。

**优化建议（opt-in，低风险）**
- **内容/强度偏置的拒绝采样**：`fg_ratio` 未命中的分支仍是整卷均匀采样（`dataset.py:881,1156-1158`），对空气占比高的 CT 会产出较多「近空」patch。可加一个廉价的方差/非背景占比阈值重采样（最多重试 K 次），提升有效监督密度——尤其利于 `z_axis` 面内被缩放后仍空的样本。这与 nnU-Net 前景/背景采样比是同族思想。

---

## 2. 模型构建（Domain 2）

**文件**：`models/factory.py`、`topology.py`、`stem.py`、`unet.py`、`blocks.py`、`resnet.py`/`convnext.py`/`mednext.py`/`unetpp.py`/`unet3p.py`

### 现状（正确/合理之处）
- **算子库丰富且实现正确**（`blocks.py`）：DropPath、GRN(ConvNeXt-V2)、SE/ECA/CBAM/CoordAttention/LKA(VAN)/MSCA(SegNeXt)、内容自注意力（softmax/linear/window/grid + nD RoPE 带缓存）、AttentionGate(Oktay)、BlurPool 抗混叠、CARAFE/DySample/PixelShuffle(ICNR)。覆盖面超过多数分割库，且都做了 2D/3D 通用化。
- **backbone/decoder 解耦装配**（`factory.py`）：resnet(basic/preact/bottleneck/r2plus1d) / convnext / mednext × unet/unetpp/unet3p，逐 stage block 数、drop_path 线性调度、逐 stage MultiRF 与 SelfAttn mask 正交叠加。
- **nnU-Net 式各向异性下采样自动调度**（`factory.py:259-312`）：逐级仅对「分辨率仍偏大且减半后≥min_size」的轴降采样，保持各轴分辨率彼此 2× 以内，并对不兼容组合（ConvNeXt LN-first 下采/非 unet decoder/hierarchical stem）**构造期显式报错**而非 forward 期崩。
- **stem 多 FOV 融合**（`stem.py`）：shared_stem / multi_stem_proj / hierarchical(逐级注入 aux)，patchN stem 的 stem_stride 由 UNet 主头/topo 头末端插值补回（`unet.py:509-517`）。
- **深监督 + 多类辅助头 + 拓扑辅助头**统一在 `UNet3D.forward` 以 dict 输出，尺寸不匹配处**显式 RuntimeError**（`unet.py:503-556`）——不静默 resize 掩盖几何错误。
- **残差块规范**：pre-act/post-act 均标准，norm/bias 参数在优化器侧免 weight decay（`optim.py:22-34`）；ConvNeXt/MedNeXt 块内固定 LN/GN+GELU 并对被忽略的 norm/act 设置**告警**。

### 问题与优化空间

**[低] 2.1 特性表面积很大，测试/维护成本高。** 7 种注意力 × 4 种自注意力 × 多种上/下采样 × 3 decoder × 3 backbone 的组合空间庞大；许多高级特性仅对特定组合支持（如各向异性下采样仅 `decoder_type='unet'`、MultiRF/SelfAttn 主要在 resnet 分支，`factory.py:366-378`）。这些边界都有报错保护，非正确性问题，但属**可维护性/组合爆炸**关注点——建议文档明确「推荐组合矩阵」，避免用户组合到未充分验证的路径。

**[低] 2.2 `arch=='adm'/'edm2'` 分支忽略大多数 block/backbone 选项**（`factory.py:319-327`），使用论文原生 GN+SiLU。属有意设计，但与 unet 主线的配置语义割裂，用户易误配后被静默忽略。建议在切到 adm/edm2 时对「被忽略的 mc.* 字段」打一次汇总告警（与 convnext/mednext 的告警一致）。

**优化/借鉴建议**
- **明确 ResEnc U-Net 为下游默认强基线**：`resenc_preset` 已存在。结合 ssltask 报告与 CVPR 2025 3D MAE 工作的结论（**下游同构 Residual Encoder U-Net + 大数据 + 严格评测 > 不断换 pretext**），建议把 ResEnc 预设作为文档推荐默认，并与 SSL 预训练骨干对齐（同名同形已由 ssltask 保证）。
- **面向细长/血管结构**：`MSCA`（条形核，各向异性友好）与 `CoordAttention` 已内置，可作为血管/气道任务的推荐注意力档位写入文档配方。

---

## 3. 数据增强 / 处理（Domain 3）

**文件**：`data/augment.py`（GPU 共享空间增强）+ `dataset.py` 预处理路径

### 现状（正确/合理之处）
- **仿射+弹性融合为单次 `grid_sample`**（`augment.py:193-296`）：合成采样网格 `G(x)=Θ(x+d)`，与「先 affine 后 elastic 两次重采样」采样位置一致但**只插值一次**，避免双重插值叠加模糊——高质量实现。
- **零同步设计**：选样 Bernoulli 掩码与逐样本标量参数全部 CPU 采样后异步搬设备，避免 `mask.any()/.item()/.nonzero()` 打断 CUDA 流水（模块 docstring 明示约束）。
- **越界填充语义正确**：affine/elastic 越界体素，image 用 `border`、**label 填背景 `label_fill`、weight_map 填中性 1.0**（`augment.py:280-294`），避免 border 复制把边缘前景外推成假监督。
- **强度增强夹回增强前范围**（`intensity_clamp`，`augment.py:60-100`）：在任何增强前采集逐样本逐通道 min/max 作为基准。
- **可分离 3D 高斯模糊向量化**（`augment.py:422-460`）：核长统一取上界、grouped conv 并行逐样本核；`simulate_lowres` 按目标尺寸分组批量 interpolate。
- **`aspect_correct`** 做旋转的 voxel-count 各向同性共轭校正 `R←A⁻¹RA`（`augment.py:124-171`），并诚实注明「不代替真实 spacing 校正」。

### 问题与优化空间

**[中] 3.1 `aspect_correct` 是 voxel-count 各向同性、非物理 spacing 各向同性。**
对大层厚各向异性数据，旋转在物理空间并非各向同性（除非启用 `spacing_normalization` 把体素做成物理各向同性）。已在 docstring 声明，属**已知取舍**；对各向异性数据+旋转增强的组合，建议默认配合 spacing 归一化，或在文档强调该限制。

**[低] 3.2 高斯噪声 std 为绝对值（`augment.py:408-419`），非按强度方差自适应。** nnU-Net 的加性噪声按体素方差缩放；此处对 minmax[0,1] 与 z-score 输入用同一 `gaussian_noise_std` 语义不同，需用户按归一化方式设值。建议：可选「按逐样本 std 缩放噪声」的 opt-in。

**[低] 3.3 `grid_dropout` 仅置零 image、不动 label（`augment.py:299-345`）。** 这是输入级正则（正确做法），此处只是提示：它不等于 CutOut 式的标签遮挡。

---

## 4. 训练全流程（含 val）（Domain 4）

**文件**：`trainer/trainer.py`、`validation.py`、`optim.py`、`amp.py`、`checkpoint.py`、`dist_utils.py`、`pipelines/*`、`losses/*`

### 现状（正确/合理之处）——本域是全仓质量标杆
- **调度以优化步为单位**（`trainer.py:150-167`）：`steps_per_epoch=ceil(len/accum)`，warmup/total/post_warmup 全部基于优化步；`one_cycle` 把 warmup_epochs 映射为 `pct_start` 且外层关闭线性 warmup，**避免 warmup 双重叠加**。→ 正是 ssltask 报告标为 P0 的问题，此处**已正确**。
- **训练步守护严谨**（`trainer.py:855-947`）：梯度累积尾组补齐；边界步在 step 前算全局 grad norm 并检查 loss/grad 有限性（**bf16/fp32 无 scaler 路径同样守护**）；非有限则 `zero_grad` 跳过且**不推进 scheduler/EMA/global step**；fp16 下据 `GradScaler` scale 回退识别跳步同样不推进——权重未变则时钟不走。这套「非成功更新一律不推进状态」正是 ssltask 报告要求 SSL trainer 继承的。
- **削同步点**：未缩放 loss 先以 GPU 张量缓存，`.item()` 延迟到日志步/边界（`trainer.py:760-846`）；分量 breakdown 仅日志步抽取。
- **DDP**：非边界步 `no_sync`（含 forward）；跳步决策基于 all-reduce 后各 rank 一致的梯度，判定天然同步。
- **优化器/调度**（`optim.py`）：norm/bias 免衰减；ZeRO-1 分片（数值等价）；`ReduceLROnPlateau` 逐 epoch、其余逐 step；**OneCycleLR resume 时 horizon 漂移的按比例折算兜底**（`optim.py:237-261`）——细节极到位。
- **验证器**（`validation.py`）：`MetricAccumulator` 累加 pooled 混淆量后闭式导出 dice/iou/recall/precision/vol_sim/mcc/**spacing-aware NSD(surface dice)**/balanced；**DDP all-reduce 与单进程全集累加严格相等**；ignore-empty 类掩码（cov==0 的类不污染 mean/min）。medium(随机 patch) / high(整卷复用 Predictor 滑窗，与部署一致) 双模，metrics dict 结构一致、选模逻辑无分支；`val_every` 控频（`trainer.py:418`）使 high 模式成本可控。
- **surface dice 实现正确**（`utils.py:461-517`）：真·边界提取 + 各向异性欧氏 EDT 的对称 NSD（MONAI 口径），一侧空表面按分母惩罚。→ 直接对应 ssltask 报告对「HD95 非标准 surface 距离」的诉求，此处采用更稳的 NSD 且 spacing-aware。
- **损失库完整正确**（`losses/losses.py`）：Dice(squared/batch_dice/ignore_empty)、BCE、Focal、Tversky、GDL(体积倒数加权+w_max)、FocalTversky、Lovász-Hinge(向量化)、clDice(可微 soft skeleton)，及复合与深监督包装、`MultiResolutionLoss` / `SliceChannelLoss`(per_slice/per_volume)。weight_map 作为求和权、class_weights 归一化加权均值使幅值稳定，AMP 下 target 转 pred dtype。

### 问题与优化空间

**[低] 4.1 深监督默认把 target 近邻下采样到各 pred 尺度**（`losses.py:334-339`）。低分辨率尺度上，微小结构可能整块消失，DS 监督对极稀疏目标偏弱。已提供 `upsample_pred`（把 pred 上采到 target）作为替代，属标准 nnU-Net 取舍，此处仅提示按目标稀疏度选择。

**[低] 4.2 `clDice` 有意忽略 `weight_map`**（`losses.py:569-589`，拓扑指标无逐体素权重一致语义）。合理，但与 region_weight 工作流组合时用户需知晓 clDice 分量不受区域权重影响。

**评价**：本域**未发现正确性问题**。建议把此 trainer/validation 作为其他任务（尤其 ssltask）的实现基线来对齐——ssltask 报告中 P0/P1（非有限守护、削同步点、异步保存、resume）在这里都已具备。

---

## 5. 推理全流程（Domain 5）

**文件**：`predictor/predictor.py`、`sliding.py`、`forwards.py`、`inputs.py`、`blending.py`、`io.py`、`adabn.py`

### 现状（正确/合理之处）
- **与训练几何严格镜像**：所有 mode 派生量来自 `build_topology(cfg)`（`predictor.py:203-273`），不再重算；`patch_mode`/`multi_res_scales`/`z_boundary_mode`/spacing 与 dataset/trainer 同源。
- **滑窗主循环高质量**（`sliding.py`）：gaussian/uniform blending 权重（可分离 3D 外积）、尾窗反推全覆盖、`addcmul_` 融合累加、跳空窗（低强度启发式）+ **跳窗比例>50% 无条件告警**（防阈值/归一化不匹配静默丢前景）、**fp16 累加器 / CPU 累加逃生门**（大卷×多类显存）、z-interleave（按物理 z spacing 拆 k 个互斥子流缝回，2.5D 各向异性）。
- **TTA 正确**（`forwards.py`）：flip 在概率空间平均；**2.5D 只翻 H/W**（D 是通道轴，翻转会反转物理切片顺序造成分布偏移），变体按 `tta_batch_size` 批量化且与串行严格等价。
- **spacing 归一化正/逆**（`predictor.py:422-479`）：输入重采样到 target_spacing、概率图回采到 pre-resample 形状；target_spacing 显式配置优先，否则回读 `npz_dir/_manifest.json`（make_data 写入）——推理复现契约闭环。
- **bbox 裁剪+拼回**、**AdaBN**（global/per_volume transductive BN，估计期强制 TTA 串行保 BN 统计一致）、channels_last、inference_mode、饱和/NaN 诊断（区分「模型饱和 vs 后处理坍塌」）。
- **ckpt 加载稳健**（`io.py`）：EMA/online variant、compile 前缀剥离、**形状预校验**、`strict=False` + missing/unexpected 日志、**加载<半参数时硬报错拒绝随机权重推理**、fp16 LayerNorm NaN 警示。
- **pretrain(迁移/SSL 权重) 加载**（`trainer/checkpoint.py` + `trainer.py:1291-1352`）：`_extract_pretrain_state_dict` 识别 `model_state_dict/model_online_state_dict/state_dict/裸 OrderedDict`、`strip_common_prefixes`、EMA shadow 重对齐、UpKern 跨核迁移、strict=False。→ **正好解决 ssltask 报告 Domain 5.1 的疑点**：segtask 侧确实能吃 SSL ckpt 的 `model_state_dict` 包装，不会静默命中 0 key。
- **NaN-safe label 化**（`blending.py:77-145`）：标量阈值 argmax / 逐类 eligible-mask（每类过自身阈值再取最高，避免先 argmax 再门控丢弃次高类）、NaN 强制背景、最小整型 dtype。

### 问题与优化空间

**[低] 5.1 `z_axis`/`whole` 的面内分辨率上界**继承自 Domain 1.1（面内 resize），推理侧一致。属架构取舍，非 bug。

**[低] 5.2 无内置连通域后处理**（`prob_to_label` 只做阈值+eligible argmax）。nnU-Net 提供可选「保留最大连通域 / 去小碎块」后处理，对单发/大器官任务能稳定去除假阳碎片。建议作为 **opt-in 后处理**（`scipy.ndimage.label` + 体积阈值 / keep-largest），默认关闭以免误伤多灶目标。

**评价**：推理域**未发现正确性问题**，逃生门与诊断齐全。

---

## 6. 可借鉴 / 适配 / 新增内容（供 Phase 2 选型，全部 opt-in、不破坏现状）

按「收益/成本/风险」排序：

| 优先 | 项 | 说明 | 依赖 | 关联域 |
|---|---|---|---|---|
| **P1** | 内容偏置拒绝采样 | `fg_ratio` 未命中分支避免近空 patch；提升有效监督密度（尤利 z_axis 面内缩放后仍空） | 无 | 1 |
| **P1** | 明确 ResEnc U-Net 为下游默认 + SSL 骨干对齐配方 | 呼应 CVPR2025 结论：同构 ResEnc + 大数据 + 严格评测 > 换 pretext | 无（`resenc_preset` 已有） | 2/迁移 |
| **P2** | 可选连通域后处理（keep-largest / 去小碎块） | 单发/大器官任务稳定去假阳；默认关，避免误伤多灶 | scipy(已依赖) | 5 |
| **P2** | 各向异性数据默认配合 spacing 归一化 + 强度自适应噪声 | 让旋转/噪声增强在物理空间更一致 | 无 | 3 |
| **P2** | 推荐组合矩阵文档 | 收敛「backbone×decoder×注意力×各向异性」的已验证组合，降低误配到弱验证路径 | 无 | 2 |
| **P3** | 边界类损失（Boundary DoU / Active Boundary）作为 clDice 之外的补充 | 面向边界精度；与现有 loss 工厂同构接入 | 无 | 4 |
| **P3** | adm/edm2 切换时汇总告警被忽略的 mc.* 字段 | 与 convnext/mednext 告警一致，减少静默误配 | 无 | 2 |

> 说明：本项目**算法与工程覆盖度已很高**，边际收益更多在「面向下游任务的默认配方沉淀 + 少量 opt-in 后处理/采样增强」，而非补基础能力。

---

## 7. 结论

- **正确性**：五域整体正确，**未发现影响正确性的实质缺陷**。训练步守护、调度时钟、DDP 指标汇总、spacing-aware NSD、推理几何镜像、ckpt/pretrain 加载均实现严谨。
- **合理性/架构**：`ModelTopology` 单一真相源 + 策略对象化 + 配置驱动 + pid/几何 fail-fast 契约，是本仓最高质量的部分；`trainer/`、`validation.py`、`optim.py`、`predictor/` 应作为其他任务（尤其 ssltask）对齐的基线实现。
- **主要取舍（非缺陷，需按目标确认）**：`z_axis`/`whole` 的面内分辨率上界（1.1/5.1）；cubic 边界前景中心夹取偏置（1.2）；`aspect_correct` 非物理各向同性（3.1）。若任务含细小面内/贴边稀疏结构，建议优先 `cubic` + spacing 归一化。
- **可新增**：内容偏置采样、ResEnc 默认配方、可选连通域后处理为三条低风险高性价比项。

**下一步**：请确认是否需要我在 Phase 2 落地其中某几项（建议先 P1 两项）。确认后我再出分步实施计划并按项推进。
