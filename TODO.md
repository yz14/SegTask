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


内容（**注意**：需查看各自对应README.md理解设计）：  
segtask_v1是2.5D/3D分割项目（项目起源，目前最完善）。  
ssltask是自监督学习项目（基于segtask_v1改造），主要是对分割，生成，分类，检测的backbone（部分/全部）进行预训练。  
clstask是分类项目（基于segtask_v1改造）。  
dettask是检测项目（基于segtask_v1改造）。  
gentask是生成/超分项目（基于segtask_v1改造）。


# TODO  
1 SSL项目代码审查（需要结合对应readme一起分析）：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程、推理全流程。

审查主要内容为代码、算法、设计、架构等等：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  
在针对下游分割项目的预训练模块/方向上是否有更好的高质量内容可以借鉴、适配或新增。  

进展：




2 分割项目代码审查：需要认真、仔细、严谨的理解、分析、思考和调研。为了保证高质量完成，本轮不动任何代码/文档：  
项目代码可大致分为5部分，数据读取、模型构建、数据增强/处理、训练全流程(含val流程)、推理全流程。

审查主要内容为代码、算法、设计、架构等等，需要结合对应readme文档一起审核：  
是否正确、合理；是否有优化空间；是否有更好的高质量内容可以借鉴、适配或新增。  

进展：
# segtask_v1 TODO 2 · Phase 2 正式代码审查报告

> 审查类型：只读静态审查 + 公式/边界条件最小验证  
> 审查范围：`segtask_v1` 配置、数据、增强、模型拓扑、损失、训练、验证、推理、NIfTI I/O 与现有测试  
> 代码基线：本会话保存的最新镜像 `/home/ubuntu/SegTask_latest`  
> 结论日期：2026-07-10  
> 约束：未修改项目代码或文档，未创建分支、提交或 PR

---

## 1. 执行摘要

### 1.1 总体评价

`segtask_v1` 不是简单脚本式训练工程，而是已经形成清晰架构边界的医学分割框架。其突出优点包括：

1. `ModelTopology` 集中派生 2.5D/3D、多分辨率、native-view、lift、aux 等几何契约；
2. 数据 pipeline、训练 pipeline 和验证 evaluator 均使用策略对象隔离模式分支；
3. 梯度累积尾组、DDP `no_sync`、AMP、EMA/SWA、ZeRO checkpoint 等复杂状态已有大量防御；
4. `weight_map` 在空间增强、损失广播和 oversample 裁剪间保持了较一致的契约；
5. 推理滑窗实现具有正权重覆盖、边界裁剪、概率融合、TTA、NIfTI 元数据复制等完整能力；
6. 代码注释总体高质量，许多性能取舍和非等价优化被明确说明。

但审查确认了若干跨模块高风险点。它们主要不是“网络能否运行”的基础问题，而是：

- 数学公式与论文定义不一致；
- optimizer、scheduler、EMA 三者在跳步时状态不一致；
- DDP 与双源混合采样没有闭环；
- 医学图像被按数组 shape 而非物理空间校验；
- 训练/验证/部署阈值和 spacing 契约存在分叉；
- 部分“high validation”仍不能复现启用 z-interleave 的部署路径。

### 1.2 严重级别汇总

| 级别 | 数量 | 说明 |
|---|---:|---|
| P0 | 0 | 未发现对所有合法配置均必现的全局阻断或无条件数据破坏 |
| P1 | 7 | 已确认的数学错误，或在明确配置下影响训练状态、数据正确性、模型选择/部署一致性 |
| P2 | 11 | 条件性正确性风险、可复现性缺口、默认评估偏差或重要架构债务 |
| P3 | 3 | 小型采样偏差、性能/文档/可维护性优化 |

### 1.3 最优先处理的五项

1. **修正 Soft clDice 分母**：完美预测当前仍产生约 `1/3` 损失。
2. **scheduler 只在 optimizer 真正更新后推进**：当前 AMP/非 AMP 非有限跳步都会消耗 LR schedule。
3. **为 MixedBatchSampler 增加 rank-aware 分片**：当前 DDP 各 rank 生成同一组 volume 索引序列。
4. **在 NPZ 烘焙及 bbox 推理前校验 origin/spacing/direction**：相同 shape 不代表物理对齐。
5. **统一验证与部署的离散化逻辑**：精确等于阈值及逐类阈值下当前行为不一致。

---

## 2. 审查方法、证据等级与优先级

### 2.1 证据等级

- **A — 确认**：可由当前源码控制流、数学代入或官方定义直接证明；
- **B — 高置信条件风险**：触发配置明确，但实际影响大小依赖数据/模型；
- **C — 优化建议**：不会直接证明结果错误，但影响扩展性、性能或诊断质量。

### 2.2 优先级定义

- **P0**：无条件阻断、广泛静默数据破坏或安全问题；
- **P1**：明确配置下会产生错误结果、错误状态推进、错误模型选择或部署失败；
- **P2**：条件性偏差、复现缺口、评估不充分或局部不一致；
- **P3**：轻微偏差、性能、文档与测试盲区。

### 2.3 官方基线

- PyTorch AMP：<https://docs.pytorch.org/docs/stable/notes/amp_examples.html>
- PyTorch DataLoader / DistributedSampler：<https://docs.pytorch.org/docs/stable/data.html>
- PyTorch 可复现性：<https://docs.pytorch.org/docs/stable/notes/randomness.html>
- MONAI Surface Dice：<https://docs.monai.io/en/latest/metrics.html>
- SimpleITK 物理空间：<https://simpleitk.readthedocs.io/en/latest/fundamentalConcepts.html>
- clDice 官方实现：<https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py>
- nnU-Net 文档：<https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation>

---

## 3. Step 0：配置、拓扑与接口契约

### 3.1 模式矩阵

权威派生入口为 `segtask_v1/models/topology.py:74-150`。

| 模式 | 模型维度 | 输入通道 | 主输出通道 | 主要输出语义 |
|---|---:|---:|---:|---|
| `whole` | 3D | `n_views`，通常 1 | `num_fg × n_views` | 每分辨率组一组前景 logits |
| `z_axis` | 3D | `n_views` | `num_fg × n_views` | z slab 的多分辨率监督 |
| `cubic` | 3D | `n_views` | `num_fg × n_views` | 3D cube 的多分辨率监督 |
| `2_5d` folded | 2D | `D × n_views` | `num_fg × D` | class-major：`(c,d)` 折叠到通道 |
| `2_5d` native depth | 2D | `ΣD_k` | `num_fg × D` | 各 view 保留原生深度后 stem 融合 |
| `2_5d` lift | 3D | `n_views` | `num_fg` | slab 被 lift 成 3D 特征 |

2.5D 折叠/展开契约在 `segtask_v1/losses/losses.py:871-925` 使用：

```python
rearrange(pred, 'b (c d) h w -> (b d) c h w', c=num_fg, d=D)
rearrange(pred, 'b (c d) h w -> b c d h w', c=num_fg, d=D)
```

该实现明确采用 **class-major、slice-minor** 排列，审查未发现模型、损失与指标间的直接排列冲突。

### 3.2 优秀实践

**[GOOD-01] Topology 单一真相源**

- 位置：`models/topology.py:31-69,74-150`
- 证据：几何与通道字段一次派生为冻结 dataclass；factory 和 pipeline 读取拓扑而非各自重算。
- 价值：显著降低 2.5D、3D、多视图组合中的 shape drift。
- 迁移建议：`ssltask/clstask/dettask/gentask` 均应采用类似的 immutable task topology。

**[GOOD-02] Pipeline strategy 隔离模式分支**

- 位置：`trainer/pipelines/base.py`、`slab25d.py`、`lift25d.py` 等。
- 价值：Trainer 只协调设备、AMP、optimizer 和状态机，不直接理解所有视图布局。

### 3.3 发现

#### [P1-07] 自动 spacing 指纹没有成为可持久化的配置真相源

- **位置**：
  - `data/make_data.py:303-324,349-355`
  - `predictor/predictor.py:131-145`
  - `config.py:1843-1849`
- **证据**：
  - `spacing_normalization=True` 且 `target_spacing=None` 时，`make_data` 自动计算数据集逐轴中位数；
  - 计算结果只写日志和每个 NPZ 的 meta，不写回/导出解析后的运行配置；
  - Predictor 对相同配置直接抛错，要求用户从日志手工复制中位数。
- **触发条件**：启用 spacing normalization，但 YAML 不显式写 `target_spacing`。
- **影响**：训练数据可以成功烘焙和训练，但原配置无法直接推理；日志丢失或复制错误会导致训练—推理 spacing 不一致。
- **严重度**：P1（在该合法配置下部署阻断）。
- **建议**：
  1. `make_data` 输出数据集级 manifest/fingerprint；
  2. manifest 至少持久化 resolved spacing、label values、归一化参数、版本与源数据摘要；
  3. Trainer/Predictor 从同一 manifest 读取，配置值只能显式覆盖并进行一致性检查。
- **置信度**：A。

---

## 4. Step 1：数据读取、划分、采样与缓存

### 4.1 已确认的稳健设计

1. 训练 patch RNG 使用 `torch.utils.data.get_worker_info().seed` 派生每 worker 的 `np.random.Generator`：`data/dataset.py:592-606`。
2. 验证 patch 由固定种子和样本序号派生：`dataset.py:608-617`，能降低 save-best 抖动。
3. NPZ 写入使用临时文件 + rename：`data/make_data.py:227-242`。
4. cache 容量会估算每 worker、每 rank 的聚合内存：`data/loader.py:817-850`。
5. 双源 sampler 明确规定粗标整轮覆盖、金标循环过采样：`data/mixed_sampler.py:85-175`。

### 4.2 发现

#### [P1-01] 双源 MixedBatchSampler 在 DDP 下不是 rank-aware

- **位置**：
  - `data/loader.py:734-766`
  - `data/loader.py:767-783`
  - `data/mixed_sampler.py:126-175`
- **证据**：
  - loader 先进入 `if use_mixed`，因此不会进入后面的 `elif world_size > 1` / `DistributedSampler` 分支；
  - 每个 rank 都以相同 `split_seed` 构造 `MixedBatchSampler`；
  - sampler RNG 仅为 `base_seed + epoch`，无 `rank/world_size`；
  - 各 rank 因而产生相同的 primary/secondary volume 索引序列。
- **触发条件**：`npz_dir_secondary` 非空且 DDP 多卡。
- **影响**：
  - 各 rank 不满足 volume 索引互斥；
  - world size 增加不会按预期扩展每 epoch 的 volume 覆盖；
  - 不同 worker seed 可能从同卷抽到不同 patch，但病例构成高度相关；
  - 与代码中普通 DDP 分支“训练样本不相交”的契约不一致。
- **官方对照**：PyTorch `DistributedSampler` 的核心语义是让每个进程加载原始数据集的独占子集。
- **严重度**：P1。
- **建议**：
  - 新建 `DistributedMixedBatchSampler(rank, world_size)`；
  - 先生成全局可复现的两源序列，再按 batch 或源内索引分片；
  - 明确定义 coarse 是“全局每 epoch 覆盖一次”还是“每 rank 覆盖一次”；
  - `__len__` 在所有 rank 必须一致，且补齐/丢弃策略显式测试。
- **置信度**：A。

#### [P2-01] checkpoint 不能恢复 persistent worker 内部 patch RNG 流

- **位置**：
  - `config.py:157`：`persistent_workers=True`
  - `data/dataset.py:586-617`：worker 内缓存 `np.random.Generator`
  - `trainer/trainer.py:1131-1150,1252-1263`：只保存主进程 torch/CUDA/numpy/python RNG
- **证据**：worker 进程中的 `_rng_cache` 状态不在 checkpoint 中。不中断训练时 persistent workers 延续原流；resume 后 workers 重建并从新 base seed 初始化。
- **触发条件**：`num_workers>0`、persistent workers、随机 patch 采样、从 epoch checkpoint resume。
- **影响**：模型/optimizer 可恢复，但下一 epoch 的 patch 序列不能做到位精确重放；注释“位精确 resume”范围过宽。
- **严重度**：P2。
- **建议**：
  - 若要求位精确，patch RNG 应成为 `f(seed, epoch, rank, sample_index, draw_index)` 的无状态函数；
  - 或保存 sampler/dataset 的可序列化随机状态并在 worker 初始化时恢复；
  - 文档区分“训练状态恢复”与“完整数据流位精确恢复”。
- **置信度**：A。

#### [P2-02] DDP resume 只精确恢复 rank0 RNG，其他 rank 被重新播种

- **位置**：`trainer/trainer.py:1252-1263`。
- **证据**：checkpoint 只由 rank0 写入；rank>0 读取同一 RNG 后立即调用 `_reseed_rank_rng(seed, rank, epoch, ...)`。
- **触发条件**：DDP resume。
- **影响**：这是合理的随机流分叉策略，但不等于各 rank 的位精确续接；可能改变增强/DropPath 序列。
- **严重度**：P2。
- **建议**：保存 `rng_state_by_rank`，或把 reproducibility contract 改为“统计可复现、非逐位续接”。
- **置信度**：A。

#### [P2-03] “患者级隔离”依赖“一文件即一患者”的隐式假设

- **位置**：`data/loader.py:693-714` 及 split helpers。
- **证据**：划分对象是 NPZ path/index；未见显式 patient/group ID 聚合。
- **触发条件**：一个患者存在多个序列、时点或重建文件，且文件 stem 不等于唯一患者 ID。
- **影响**：同一患者的不同 volume 可能进入 train 与 val。
- **严重度**：P2。
- **建议**：在 manifest 中增加 `patient_id/study_id/series_id`，split 接收 group key；启动时检查 group 交集为空。
- **置信度**：B；若数据规范保证每患者仅一卷，则不触发。

---

## 5. Step 2：预处理与增强

### 5.1 已确认的稳健设计

1. affine 与 elastic 合并成一次 `grid_sample`：`data/augment.py:189-281`，避免双重插值。
2. image 使用 bilinear、label 使用 nearest、weight map 可配置 nearest/bilinear。
3. 三路均使用同一 grid；空间同步契约正确。
4. `padding_mode='border'` 可避免 weight map oversample slack 区域被填成 0。
5. 随机选样和标量参数在 CPU 生成，减少 CUDA→host 隐式同步。
6. oversample 后在 Trainer 统一中心裁剪：`trainer/trainer.py:787-795`。

### 5.2 发现

#### [P2-04] affine 的 aspect correction 不是物理 spacing correction

- **位置**：
  - `data/augment.py:120-131,156-159,252-259`
  - `config.py:219-224`
- **证据**：
  - 共轭矩阵使用 `A=diag(W,H,D)`；
  - augment 实现注释明确说明“不代替真实 spacing 校正”；
  - config 注释却称“在物理各向同性坐标里做旋转”。
- **触发条件**：原始/target spacing 各向异性，尤其 z spacing 显著大于 x/y。
- **影响**：相同角度对应的物理旋转并不准确，可能混入非期望剪切/缩放；配置文档会让用户误认为已做物理校正。
- **严重度**：P2。
- **建议**：
  - batch 携带 spacing，使用 `diag(W*sx, H*sy, D*sz)` 或直接在物理坐标构造变换；
  - 若数据已重采样到各向同性 spacing，则显式验证；
  - 修正文档，不把 voxel-count correction 称为 physical correction。
- **置信度**：A。

#### [P2-05] label 的 border padding 可复制边缘前景

- **位置**：`data/augment.py:272-279`。
- **证据**：image、label、weight map 全部使用 `padding_mode='border'`；label 为 nearest。
- **触发条件**：旋转/平移采样越界，且 label 边界处有前景；`z_axis` H/W 无 oversample 余量时尤其明显。
- **影响**：边界前景会沿越界区域复制，产生非真实监督；config 已承认 z-axis 面内伪影会保留。
- **严重度**：P2。
- **建议**：
  - image 可保持 border；
  - label 越界使用 background constant；
  - weight map 越界使用语义中性值 1；
  - 增加前景贴边的旋转/平移合成测试。
- **置信度**：B。

#### [P3-01] Grid Dropout 最后一个合法起点永远不会被采样

- **位置**：`data/augment.py:300-309`。
- **证据**：`torch.randint(0, max(D-hd,1))` 的上界不包含；合法起点应为 `0..D-hd`，高位应传 `D-hd+1`。H/W 同样。
- **触发条件**：hole 小于对应轴。
- **影响**：洞位置轻微偏向左/前/上侧，永远不能严格贴右/后/下边界。
- **严重度**：P3。
- **建议**：上界改为 `max(D-hd+1,1)`，并做位置覆盖测试。
- **置信度**：A。

---

## 6. Step 3：模型、拓扑、损失与输出契约

### 6.1 已确认的稳健设计

1. `ModelTopology` 是模型几何唯一入口。
2. `SliceChannelLoss` 对 `(B,num_fg*D,H,W)` 的 class-major 展开明确。
3. Dice/Tversky/GDL 中 `weight_map` 作为求和权重，同时作用于分子与分母。
4. clDice 明确忽略 weight map，而不是静默部分使用。
5. loss 计算被强制到 fp32，避免 fp16 大体素求和溢出。

### 6.2 发现

#### [P1-02] Soft clDice 最终调和公式错误，完美预测损失不为 0

- **位置**：`losses/losses.py:607-614`。
- **实现**：

```python
tprec = (... + smooth) / (... + smooth)
tsens = (... + smooth) / (... + smooth)
cldice = 2 * tprec * tsens / (tprec + tsens + smooth)
```

- **官方定义**：clDice 官方实现最终分母为 `tprec + tsens`，不再加 `smooth`。
- **最小数学验证**：
  - 完美预测时 `tprec=1, tsens=1`；
  - 当前默认 `smooth=1`：`clDice=2/(1+1+1)=2/3`；
  - 当前 loss=`1/3`，而正确值应为 0。
- **触发条件**：使用 `cldice` 或 `dice_cldice`。
- **影响**：
  - clDice 分量具有错误下限；
  - 复合损失最优点被常量和非线性缩放改变；
  - 监控曲线和 loss 权重语义失真。
- **严重度**：P1。
- **建议**：最终式改为 `2*tprec*tsens/(tprec+tsens)`；仅在需要防 `0/0` 时加入极小 epsilon，而不是使用 numerator smoothing 常数。
- **必测不变量**：完美预测→0；全空/全空行为显式定义；单线段完全匹配→0；断裂预测比连通预测损失高。
- **置信度**：A。

#### [P2-06] 需要把 topology 契约测试从 smoke 提升为全组合参数化测试

- **位置**：`models/topology.py`、`trainer/pipelines/*`、`losses/losses.py`。
- **证据**：当前架构已集中，但合法组合仍多：patch mode × views × native × lift × aux × DS。
- **触发条件**：新增模型/decoder/head 或修改 multi-res 规则。
- **影响**：单一真相源能降低风险，但不能自动证明 dataset/pipeline/model/predictor 都遵守同一排列。
- **严重度**：P2。
- **建议**：对每种合法 topology 自动生成 dummy 输入，断言 dataset 输出、model logits、loss reshape、metric split、predictor 输出完整闭环。
- **置信度**：C。

---

## 7. Step 4：训练、优化、分布式与 checkpoint 状态机

### 7.1 已确认的稳健设计

1. 尾部不足 accumulation 的分母使用真实尾长：`trainer/trainer.py:623-631,798-818`。
2. DDP `no_sync()` 包住 forward 和 backward：`trainer.py:800-819`，符合 PyTorch 要求。
3. 非 fp16 路径对 loss/grad non-finite 做跨 rank `any` 共识。
4. GradScaler 跳步后 EMA 不推进。
5. ZeRO 在 rank early-return 前 collective consolidate。
6. async checkpoint 深拷到 CPU，避免 state_dict 与在线参数共享存储。
7. checkpoint 包含 model/online/EMA/SWA/optimizer/scheduler/scaler/best/patience/RNG。

### 7.2 状态机核查

| 事件 | optimizer | scheduler | EMA | 当前结果 |
|---|---|---|---|---|
| 正常步 | step | step | update | 一致 |
| fp16 GradScaler 检测到 inf | skip | **step** | skip | 不一致 |
| bf16/fp32 non-finite guard | skip | **step** | skip | 不一致 |

### 7.3 发现

#### [P1-03] optimizer 被跳过时 scheduler 仍推进

- **位置**：
  - 非 scaler 跳步：`trainer/trainer.py:873-908`
  - GradScaler 跳步：`trainer/trainer.py:921-933`
- **证据**：
  - `skip_optim_step` 分支执行 `scheduler.step()` 后 `continue`；
  - fp16 路径通过 scale 回退识别 `scaler_skipped`，但 `scheduler.step()` 无条件执行；
  - 只有 EMA 被条件保护。
- **官方对照**：PyTorch 明确说明 `GradScaler.step()` 在 inf/NaN 时不调用底层 `optimizer.step()`。
- **触发条件**：AMP scale 校准或训练中出现 inf/NaN；bf16/fp32 non-finite guard。
- **影响**：
  - LR schedule 按“尝试步数”而非“参数更新次数”前进；
  - warmup、OneCycle、cosine 的有效更新点减少；
  - PyTorch scheduler 可能认为首次 scheduler step 早于 optimizer step；
  - resume 后 scheduler 状态虽可恢复，但恢复的是已经漂移的状态。
- **严重度**：P1。
- **建议**：
  - 定义单一 `did_optimizer_step`；
  - 仅 `did_optimizer_step=True` 时执行 per-step scheduler、EMA、SWA update counter 与 optimizer-step 健康计数；
  - plateau/epoch scheduler 保持其独立粒度。
- **置信度**：A。

#### [P2-07] SWA BN 重估在 DDP 下未聚合全局统计

- **位置**：
  - `trainer/trainer.py:1085-1120`
  - `predictor/adabn.py:48-97`
- **证据**：
  - 每 rank 在自己的 `train_loader` shard 上 reset 并累计 BN；
  - 未见 BN moments 的 all-reduce；
  - 最终只保存 rank0 权重/buffer。
- **触发条件**：DDP + `swa_enabled=True` + 模型含 BatchNorm。
- **影响**：最终 SWA checkpoint 的 BN running stats 主要代表 rank0 数据 shard，而非全训练集；不同 world size 可能得到不同结果。
- **严重度**：P2。
- **建议**：
  - 只在 rank0 用非分片 loader 重估，然后 broadcast；
  - 或累加每层 `count/sum/sumsq` 并跨 rank reduce；
  - 增加单卡与 2-rank SWA 输出接近性测试。
- **置信度**：B。

#### [P2-08] checkpoint 写入最终路径，不具备崩溃原子性

- **位置**：
  - `trainer/checkpoint.py:147-196`
  - `trainer/trainer.py:1181-1201`
- **证据**：同步和异步路径均直接 `torch.save(state, final_path)`；没有 temp + atomic replace。
- **触发条件**：保存过程中进程崩溃、磁盘满、节点重启。
- **影响**：目标文件可能部分写入；`best_model.pth` 可能覆盖掉此前可用 best。相比之下，`make_data` 已采用临时文件原子替换。
- **严重度**：P2。
- **建议**：同目录临时文件写入、flush/fsync（按可靠性需求）、`os.replace`；成功后再 prune。
- **置信度**：A。

---

## 8. Step 5：验证与指标

### 8.1 已确认的稳健设计

1. `MetricAccumulator` 累加 TP/pred/target/voxels 等可加量，DDP 最后 all-reduce；不是 batch Dice 简单平均。
2. 空 GT 类通过 coverage mask 从 mean/min 排除；空病例上的 false positive 仍进入 pooled pred sum。
3. medium/high evaluator 产出相同 metrics schema。
4. high 模式复用 Predictor，减少验证—部署滑窗代码漂移。
5. threshold 来源统一读取 `cfg.predict.threshold`。

### 8.2 发现

#### [P1-04] Surface Dice 是 voxel Chebyshev 邻域，不是物理空间表面距离

- **位置**：
  - `utils.py:430-490`
  - `trainer/validation.py:141-149,258-270`
  - `config.py:790-805`
- **证据**：
  - 边界由 3×3×3 pooling 提取；
  - tolerance 通过 kernel=`2τ+1` 的 max-pool 膨胀；
  - 日志明确为 `@Npx`，config 明确为“像素，Chebyshev 邻域”；
  - 没有输入 spacing。
- **官方对照**：MONAI Surface Dice 基于双向最近表面距离，支持 `distance_metric`、`spacing` 和 class-specific thresholds。
- **触发条件**：
  - `save_best_criterion` 为 `dice+surface_dice` 或 `balanced`；
  - 不同病例 spacing 不同，或 spacing 各向异性；
  - 任务要求 mm 级边界容差。
- **影响**：
  - 相同 `τ=1` 在不同轴和不同病例代表不同毫米距离；
  - Chebyshev 邻域把对角偏移也视作同一 tolerance，和 Euclidean 距离不等价；
  - 可导致模型选择与临床边界标准不一致。
- **严重度**：P1（当其参与选模）；仅作快速像素指标时为 P2。
- **建议**：
  - 将当前指标更名为 `voxel_chebyshev_surface_overlap`；
  - 正式选模增加 spacing-aware Euclidean NSD；
  - tolerance 使用 mm，并支持逐类阈值；
  - NPZ manifest 必须提供 resolved spacing。
- **置信度**：A。

#### [P2-09] 默认 medium validation 是确定性 patch 指标，不是部署级 volume 指标

- **位置**：`config.py:770-781`、`trainer/validation.py:331-366`。
- **证据**：默认 `val_metric_mode="medium"`；只对 val loader patch 前向。
- **触发条件**：保持默认设置并据此 save-best/early-stop。
- **影响**：
  - z 上下文、滑窗融合、bbox、边缘、空病例和全卷 false positive 不能完整反映；
  - 确定性消除了采样抖动，但不能消除代表性偏差。
- **严重度**：P2。
- **建议**：medium 用于每 epoch 快速监控，high 用于周期性或候选 best 复核；最终 best 必须由 full-volume 指标确认。
- **置信度**：A。

#### [P2-10] high validation 在启用 z-interleave 时仍不等于部署路径

- **位置**：
  - `config.py:773-780`
  - `predictor/predictor.py:473-497`
  - `trainer/validation.py:435-477`
- **证据**：NPZ 不向 Predictor 传 z spacing；`predict_preprocessed_array(..., z_spacing=None)` 会回退标准 z sliding。
- **触发条件**：2.5D + `z_interleave_enabled=True`。
- **影响**：名为 high/full-volume 的选模预测与部署使用不同 z 邻域。
- **严重度**：P2。
- **建议**：从 NPZ meta 读取 normalized/original z spacing，明确传给 Predictor；增加 interleave on/off 的 full-volume parity 测试。
- **置信度**：A。

#### [P3-02] val_loss 与 train loss 不同口径，但共享“loss”命名

- **位置**：`trainer/validation.py:355-364`。
- **证据**：验证只计算裸 `base_loss`，不含 deep supervision、aux、topology、多分辨率权重；注释已说明不可直接对比。
- **影响**：图表使用者仍容易把 train loss 与 val loss 当作同一目标判断过拟合。
- **严重度**：P3。
- **建议**：命名为 `val_base_loss`；若需要可比曲线，额外计算 `val_objective_loss`。
- **置信度**：A。

---

## 9. Step 6：推理、滑窗、TTA 与医学图像 I/O

### 9.1 已确认的稳健设计

1. `compute_1d_positions` 强制加入尾窗，覆盖完整轴：`predictor/blending.py:26-43`。
2. Gaussian 权重严格为正，不存在边缘零权重。
3. cubic/z 累加器按权重归一化，fp16 使用可表示的 clamp 下界。
4. bbox 推理后概率图拼回原数组 shape。
5. 输出 NIfTI 对 label 和每类概率执行 `CopyInformation(ref_img)`。
6. NaN 概率显式报警并回退背景。
7. whole/z/cubic/2.5D 的 mode dispatch 清晰。

### 9.2 发现

#### [P1-05] image/label/bbox/region-weight 只校验 array shape，不校验物理空间

- **位置**：
  - 烘焙：`data/make_data.py:141-190`
  - 推理 bbox：`predictor/predictor.py:370-391`
  - NIfTI 元数据读取只在 image spacing 使用：`data/dataset.py:132-159`
- **证据**：
  - image 与 label 仅比较 crop 后 shape；
  - bbox 推理只比较 bbox array shape 与 image array shape；
  - 未比较 origin、spacing、direction；
  - spacing normalization 使用 image spacing 同时重采样 label/rw。
- **官方对照**：SimpleITK 定义图像物理区域由 origin、spacing、size、direction 共同决定；相同 size 不表示同一物理区域。
- **触发条件**：label/bbox/rw 与 image shape 相同，但 affine、方向、origin 或 spacing 不同。
- **影响**：
  - 可静默裁错 ROI、错位监督或错误重采样 label；
  - 最终输出虽然复制 image 元数据，但预测内容可能源自错位输入；
  - 属于医学影像中高风险静默错误。
- **严重度**：P1。
- **建议**：
  - make_data 前读取四类图像 header，比较 size/origin/spacing/direction；
  - 浮点元数据使用明确容差；
  - 不一致时默认 fail-fast；
  - 若允许自动配准/重采样，必须以 image 为 reference 使用 SimpleITK physical resampling，而不是只按 shape zoom；
  - 将校验结果和几何摘要写入 manifest。
- **置信度**：A。

#### [P1-06] 验证与部署在“概率恰等于阈值”时结论相反

- **位置**：
  - 验证：`utils.py:338-339`、`trainer/validation.py:453-459`
  - 部署：`predictor/blending.py:110-122`
- **证据**：
  - 验证使用 `prob > threshold`；
  - `prob_to_label` 使用 `below = max_prob < threshold`，因此 `prob == threshold` 被判为前景；
  - docstring 又声明“max fg 概率 > threshold 才取前景”，与实现不符。
- **最小验证**：`prob=0.5, threshold=0.5` 时，验证为背景，部署为第一个前景类。
- **触发条件**：概率精确等于阈值；默认阈值 0.5 对应 logit=0。
- **影响**：未训练/零初始化/量化或部分对称输出可能出现明显验证—部署分叉。
- **严重度**：P1。
- **建议**：提取唯一 `discretize_probabilities()`，验证、高验证、CLI 输出和测试全部复用；统一采用 `>` 或 `>=` 并写入契约。
- **置信度**：A。

#### [P2-11] 逐类阈值下先 argmax 再判阈值会丢弃本可接受的次高类

- **位置**：`predictor/blending.py:110-122`。
- **证据**：

```python
max_class = prob.argmax(axis=0)
below = max_prob < threshold[max_class]
```

- **最小例**：
  - 概率 `[0.60, 0.59]`，阈值 `[0.70, 0.50]`；
  - class 1 未达 0.70，class 2 已达 0.50；
  - 当前先选 class 1，再回退背景，class 2 被忽略。
- **触发条件**：多前景类 + 逐类不同 threshold。
- **影响**：per-class threshold 的语义并非“每类先过门槛，再在合格类中选最优”；同时验证按每类独立阈值统计，部署却强制互斥 argmax。
- **严重度**：P2。
- **建议**：先构造 eligible mask，再定义合格类间的选择分数；并明确多标签指标与互斥 label-map 指标的区别。
- **置信度**：A。

#### [P3-03] skip-empty-window 是强数据假设，不应称作纯背景证明

- **位置**：`config.py:1031-1042`、`predictor/sliding.py:401-406`。
- **证据**：仅依据归一化后窗口最大强度判断，不使用 body mask 或 ROI。
- **触发条件**：用户启用该优化，且目标结构强度不高于阈值，或归一化方式改变。
- **影响**：窗口直接被强制为 0 概率。
- **严重度**：P3；默认关闭且文档已有风险说明。
- **建议**：改名为 `skip_low_intensity_windows`；要求显式数据预设或 ROI mask；记录跳过区域比例并提供安全上限。
- **置信度**：A。

---

## 10. Step 7：跨模块闭环结论

### 10.1 闭环状态

| 链路 | 状态 | 说明 |
|---|---|---|
| Config → Topology | 良好 | 单一派生入口，模式约束集中 |
| Topology → Dataset/Pipeline/Model | 良好 | 主要 shape 契约一致 |
| Model → Loss | 良好但有公式缺陷 | 2.5D 排列一致；clDice 公式错误 |
| Loss → Trainer | 良好 | fp32 loss、weight map、DS/aux 分工清晰 |
| Trainer → Scheduler/EMA | **不闭环** | optimizer skip 时 scheduler 仍前进 |
| Data → DDP | **不闭环** | mixed source bypass distributed sharding |
| NIfTI → NPZ | **不闭环** | shape 校验代替物理空间校验 |
| make_data → Predictor | **不闭环** | 自动 target spacing 未持久化 |
| Validation → Deployment | **不闭环** | 阈值边界、per-class threshold、z-interleave 不一致 |
| Predictor → NIfTI output | 良好 | shape 回填及 metadata copy 正确，前提是输入物理对齐 |

### 10.2 Findings Ledger

| ID | 级别 | 类别 | 简述 | 置信度 |
|---|---|---|---|---|
| P1-01 | P1 | 分布式数据 | MixedBatchSampler 无 rank 分片 | A |
| P1-02 | P1 | 损失数学 | clDice 完美预测仍有 1/3 损失 | A |
| P1-03 | P1 | 训练状态机 | optimizer skip 但 scheduler step | A |
| P1-04 | P1 | 指标 | Surface Dice 非物理空间 NSD | A |
| P1-05 | P1 | 医学 I/O | 只校验 shape，不校验 origin/spacing/direction | A |
| P1-06 | P1 | 部署一致性 | 阈值相等时 val 与 infer 相反 | A |
| P1-07 | P1 | spacing 契约 | 自动 median spacing 无法由同配置推理 | A |
| P2-01 | P2 | 可复现性 | persistent worker RNG 不可 resume | A |
| P2-02 | P2 | 可复现性 | DDP 非 rank0 RNG 非位精确恢复 | A |
| P2-03 | P2 | 数据泄漏 | split 隐式假设一文件一患者 | B |
| P2-04 | P2 | 增强 | aspect correction 非物理 spacing | A |
| P2-05 | P2 | 增强 | label border padding 可复制前景 | B |
| P2-06 | P2 | 测试 | topology 全组合契约测试不足 | C |
| P2-07 | P2 | SWA/DDP | BN stats 未做全局聚合 | B |
| P2-08 | P2 | checkpoint | 写盘不具崩溃原子性 | A |
| P2-09 | P2 | 验证 | 默认 medium patch 指标用于选模 | A |
| P2-10 | P2 | 验证/推理 | high val 不复现 z-interleave | A |
| P2-11 | P2 | 阈值 | per-class threshold 先 argmax 后门控 | A |
| P3-01 | P3 | 增强 | Grid Dropout 起点上界少 1 | A |
| P3-02 | P3 | 监控 | val_loss 与 train loss 不同口径 | A |
| P3-03 | P3 | 推理优化 | skip-empty 实为强度启发式 | A |

---

## 11. 改进路线

### 11.1 第一批：立即修复，改动小、收益高

| 项目 | 预计范围 | 验证 |
|---|---|---|
| clDice 最终分母 | 1 个 loss 文件 + 单测 | 完美/空/断裂/连通不变量 |
| scheduler 跳步保护 | Trainer 状态机 | fp16 inf、bf16 NaN、正常步 |
| 阈值离散化单一函数 | blending + validation | exact threshold、scalar/per-class |
| Grid Dropout `+1` | augment + 单测 | 所有合法边界起点可达 |
| checkpoint 原子写 | checkpoint helper | 故障注入与旧文件保留 |

### 11.2 第二批：跨模块契约修复

| 项目 | 预计范围 | 验证 |
|---|---|---|
| DistributedMixedBatchSampler | sampler + loader + DDP tests | rank 互斥、比例、覆盖、等长 |
| dataset manifest/fingerprint | make_data + loader + predictor | spacing/label/normalize 端到端 |
| 物理几何一致性校验 | NIfTI reader/make_data/predictor | origin/spacing/direction mismatch fail-fast |
| z-interleave high val parity | NPZ meta + evaluator | full-volume val 与 CLI 同输入同输出 |
| SWA BN 全局重估 | Trainer distributed helper | 1-rank/2-rank parity |

### 11.3 第三批：评估与医学语义升级

1. 增加 spacing-aware Euclidean NSD/HD95；
2. 保存 per-case/per-class 指标，同时保留 pooled 指标；
3. 最终模型选择采用 full-volume deployment-consistent metric；
4. split manifest 显式 patient/study grouping；
5. 物理空间增强使用 spacing-aware transform。

---

## 12. 推荐测试矩阵

### 12.1 单元测试

| 模块 | 场景 | 核心断言 |
|---|---|---|
| clDice | perfect prediction | loss≈0 |
| clDice | broken vs connected tube | broken loss 更高 |
| discretization | prob==threshold | val/infer 完全相同 |
| discretization | per-class thresholds | 合格类选择符合定义 |
| Grid Dropout | 统计大量起点 | `0` 与 `length-hole` 均可出现 |
| surface metric | anisotropic spacing | mm 距离符合预期 |
| NIfTI geometry | same shape, different origin | fail-fast |
| NIfTI geometry | same shape, flipped direction | fail-fast |
| checkpoint | save 中断 | 旧 checkpoint 仍可加载 |

### 12.2 合成契约测试

对以下组合自动构造小张量：

```text
patch_mode = whole / z_axis / cubic / 2_5d
n_views = 1 / 2
2_5d = folded / native_depth / lift
deep_supervision = off / on
aux_seg = off / on
num_fg = 1 / 3
```

逐组合断言：

1. topology 派生；
2. dataset batch shape；
3. model logits schema；
4. pipeline main/aux/DS 拆分；
5. loss 可反向；
6. metric reshape；
7. predictor 输出 `(num_fg,D,H,W)`。

### 12.3 分布式集成测试

| 配置 | 必测点 |
|---|---|
| DDP + ordinary sampler | rank index 互斥、set_epoch 改变顺序 |
| DDP + mixed sampler | 两源比例、全局覆盖、rank 互斥 |
| DDP + accum tail | 与单卡等效 batch 的梯度接近 |
| DDP + fp16 overflow | 所有 rank 同步 skip；scheduler/EMA 不推进 |
| DDP + bf16 NaN | all-reduce any；所有状态不推进 |
| DDP + SWA BN | 1-rank/2-rank logits 接近 |
| DDP resume | 明确验证“逐位”或“统计”复现等级 |

### 12.4 端到端医学 I/O 测试

1. 生成带非单位 spacing、非零 origin、非 identity direction 的 SimpleITK image；
2. 生成严格共注册 label/bbox，验证 bake→predict→save 后元数据不变；
3. 分别扰动 label 的 origin/spacing/direction，确认 bake 拒绝；
4. spacing normalization 自动 fingerprint 后，用同 manifest 推理；
5. bbox crop + resample + paste-back 后检查物理点映射；
6. full-volume validation 与 CLI inference 对同一 NPZ/NIfTI 比较概率。

---

## 13. 可迁移到其他 task 的高质量模式

### 13.1 建议直接复用

1. **Immutable topology**  
   适用于 `clstask` 多视图输入、`dettask` FPN 尺度、`gentask` latent/condition layout。

2. **Strategy pipeline**  
   将数据布局、loss 拆分、metric 拆分放入 task strategy，Trainer 不理解任务细节。

3. **有效尾组 accumulation**  
   `_effective_accum` 可直接复用，防止 epoch 尾部梯度被低估。

4. **DDP no_sync 包含 forward**  
   这是容易遗漏但实现正确的分布式模板。

5. **可加统计量后 all-reduce**  
   指标先累加充分统计量，最后一次 collective，优于每 batch 汇总均值。

6. **异步保存前深拷 CPU**  
   可避免在线 state_dict 被后续更新污染；再补原子 replace 即可成为通用组件。

7. **部署高验证复用 Predictor**  
   验证和 CLI 共用相同 forward/sliding path，是减少训练—部署漂移的正确方向。

8. **显式 weight-map contract**  
   统一 `(B,1,*spatial)` 并在 loss 侧广播，适用于检测 heatmap、生成 mask、SSL confidence map。

### 13.2 不应原样迁移

1. 不能把 shape equality 当作多模态/医学影像对齐证明；
2. 自定义 sampler 必须从设计开始支持 rank/world size；
3. scheduler/EMA/SWA 必须绑定“真实 optimizer update”而非 loop boundary；
4. threshold/discretization 必须是共享函数，不能在 metric 与 export 各写一份；
5. 任何“physical”指标或增强必须携带 spacing/origin/direction 契约。

---

## 14. 证据缺口与审查边界

1. 本报告基于保存的最新源码镜像；隧道此前发生 SSL EOF，无法证明镜像之后用户本地没有新增未同步改动。
2. 当前审查环境未安装 PyTorch，因此未执行 GPU/模型 smoke tests；高风险结论均来自静态控制流、公式代入和官方定义。已用无第三方依赖的最小数学脚本验证 clDice 与阈值边界：
   - 当前 perfect clDice=`2/3`，loss=`1/3`；
   - `prob=threshold=0.5` 时 infer=foreground、val=background；
   - `[0.60,0.59]` 配 `[0.70,0.50]` 时部署回退背景，但第二类独立过阈值。
3. 未使用真实临床数据，因此不对精度、泛化或临床有效性下结论。
4. 未修改测试来“证明”发现，也未实施任何修复。

---

## 15. 最终结论

该项目的核心架构方向正确，尤其是 topology-driven 派生、pipeline strategy、累积/DDP/EMA/checkpoint 防御及 Predictor 复用，质量明显高于常见单任务训练仓库。

当前不建议直接进行大规模重构。最合理的下一阶段是：

1. 先用小 PR/小批次修复 clDice、scheduler skip、threshold、Grid Dropout 与原子 checkpoint；
2. 再单独设计 distributed mixed sampler 与 dataset fingerprint；
3. 最后升级物理空间校验和 spacing-aware metrics。

在 P1 项修复并通过契约测试前，以下配置应视为高风险：

- `loss=cldice` / `dice_cldice`；
- DDP + secondary mixed source；
- `save_best_criterion` 使用 surface dice/balanced；
- image/label/bbox 来自可能未严格共注册的 NIfTI；
- `spacing_normalization=True, target_spacing=None`；
- 多类逐类阈值；
- 训练期间出现 AMP/non-finite optimizer skip。

---

## 16. 核验记录与第一批修复（2026-07-10，Devin）

### 16.1 核验结论（逐条对照最新代码，只读核验）

21 条 findings 中 **19 条属实**（行号、控制流、公式代入均与代码一致）；2 条需修正：

- **[P1-01] 触发条件不成立，降级为功能缺口（P3/增强）**：`data/loader.py:618-623` 已在
  `world_size>1 且 npz_dir_secondary 非空` 时显式 `raise ValueError`（fail-fast），报告所述
  "各 rank 静默产生相同序列"不可达。MixedBatchSampler 确实非 rank-aware，
  "实现 DistributedMixedBatchSampler" 作为功能增强仍成立。相应修正：§1.3 第 3 条、
  §10.1 "Data → DDP 不闭环"（实为"显式互斥，功能缺口"）、§15 高风险配置
  "DDP + secondary mixed source"（该组合直接报错，非静默风险）。
- **[P1-03] 行为属实，但为代码内明示的有意取舍**：`trainer.py` 注释明确写有
  "scheduler 照常推进，EMA 不推进"；与 PyTorch 官方语义分歧仍在，修复建议合理。

表述微调：

- **[P1-07]**：make_data 实际已写数据集级 `_manifest.json`（只是缺 target_spacing 字段）；
  且 DESIGN.md 声称 NPZ meta 中的 spacing "方便推理时做逆变换"，但 Predictor 并未读取。
  修复成本比报告暗示的低：扩展现有 manifest 并让 Predictor 读取即可。
- **[P2-06]**："smoke 级"低估现状：已有 test_model_topology / test_model_flow /
  test_keep_native_* / test_lift_aux_ds / test_pipelines 等多个拓扑单测；
  缺的是全组合参数化闭环测试。
- **[P2-10]**：`config.py` 注释已明示 medium 无 z-interleave 属已知取舍；
  NPZ meta 已含 spacing，报告的修复建议可行。

### 16.2 第一批修复（已实施并通过测试）

对应 §11 第一批（均为局部小改，配套回归测试 `tests/test_review_batch1_fixes.py`，9 项全过）：

1. **P1-02**：`losses.py` SoftCLDiceLoss 最终调和均值分母去掉 `+ smooth`
   （改为 `clamp(min=1e-8)` 兜底 smooth=0），完美预测损失 ≈ 0。
2. **P1-03**：`trainer.py` 两条路径（bf16/fp32 non-finite guard 与 fp16 GradScaler 跳步）
   均改为仅在 optimizer 真正更新后推进 scheduler 与 EMA；跳步决策各 rank 一致
   （all-reduce / all-reduce 后梯度），DDP 同步不变量保持。
   同步更新 `tests/test_round2_fixes.py` 中两处编码旧行为的断言（current_step 1→0）。
3. **P1-06**：`blending.py` prob_to_label 改为 `max_prob <= threshold` 判背景
   （即严格 `>` 取前景），与验证侧 `sigmoid(pred) > threshold` 及 docstring 一致；
   标量与逐类阈值两分支同步修正。
4. **P3-01**：`augment.py` _grid_dropout 起点 `randint` 上界改为 `axis - hole + 1`，
   末端起点可达且不越界。
5. **P2-08**：新增 `checkpoint.atomic_torch_save`（同目录 tmp + `os.replace`，失败清理重抛），
   接管 best/周期/SWA 同步保存与 AsyncCheckpointSaver 后台保存共 4 处 `torch.save`。

验证：CPU torch 2.7.1 下 `tests/test_review_batch1_fixes.py` 9/9 通过；
`test_new_losses / test_checkpoint_resume / test_todo_p_regressions / test_round2_fixes`
相关用例通过（test_round2_fixes 中 bug1/bug6/bug10b/bug11 等 4 项失败为陈旧测试
与演进后代码不匹配，改动前即失败，与本批修复无关）。

### 16.3 第二批修复（已实施并通过测试，2026-07-10）

配套回归测试 `tests/test_review_batch2_fixes.py`（14 项全过）；全仓测试基线
无新增失败（陈旧失败集合与改动前一致）：

1. **P1-01（功能增强）**：`mixed_sampler.py` MixedBatchSampler 增加
   `rank`/`world_size` 与 `set_epoch`（DistributedSampler 同款接口）：各 rank
   共享同一全局 batch 序列（同 seed+epoch），按 batch 取 strided 不相交切片、
   等长（尾部不整除丢弃，类比 drop_last）。`loader.py` 移除 DDP+双源互斥
   fail-fast，混合采样器直接接 rank/world_size；`trainer.py` 采样器识别改为
   按 set_epoch 协议鸭子识别 sampler/batch_sampler 两处，每 epoch set_epoch
   对齐重洗。
2. **P1-07 / P2-04（部分）**：`make_data.py` manifest 增记
   `spacing_normalization` 与解析后的 `target_spacing`（自动中位数不再只存在
   于日志）；`predictor.py` 在 `data.target_spacing` 未显式配置时从
   `npz_dir/_manifest.json` 回读（显式配置仍优先，均缺失才报错）。
   `_TOOL_VERSION` 1.4→1.5。
3. **P1-04/P1-05（几何部分）**：`dataset.py` 新增 `read_nifti_geometry`
   （只读头返 spacing/origin/direction）；`make_data.prepare_one` 开头对
   label/bbox/rw 与 image 做物理坐标系一致性校验（容差 spacing/direction
   1e-3、origin 1e-2 mm），不一致 fail-fast（与 shape 校验同级），
   杜绝"shape 相同但物理坐标系不同"的静默错配。

### 16.4 后续批次（未实施）

- 第三批：spacing-aware Surface Dice、SWA/AdaBN BN buffer 跨 rank 聚合、
  high-val z-interleave 一致性。




3 


4 模型流可视化需要有层次化，结构化，美化，可以清晰看到计算流的走向，可以清晰理解模型架构，可以清晰的溯源。总之：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判、讨厌"自动布局默认输出"式的无设计感结果。以下是一些例子：  

- 聚焦模式到stem, stage这个层级为止：
点击模块A，进入聚焦模式，模块群B和A有联系，模块群C和A没有联系，所以模块群C谈出，模块群B突显。我希望到stem，stage这个级别的模块能进入聚焦，再进一步的子模块例如stem，stage里面的子模块则不进入聚焦。  

- 连线走线需要清晰、不重叠、不交叉、美观、可以溯源：
需要清晰的看到不同模块的关系，并能溯源输入输出等等

- 位置清晰，层次清晰，严格遵守各自的位置关系：
例如输入后可能同时结果多个stem，那么这几个stem就是位置并列的；例如如果有deep supervision，且在dec level 0后有ds head 2, dec level 1后有ds head 1等等，那么ds head 2位置就应该和dec level 1并列，因为它们就是dec level 0的下一个计算。

- 其它的我暂时没有想到，请你根据我的喜好推荐，注意，原则是：层次化/结构化/位置即计算次序、走线可溯源不交叉、方案通用无架构特判。

进展：
