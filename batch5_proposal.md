# Batch 5 方案（P2-04 / P2-01+02 / P2-09）

Batch 4 已完成（P2-11 / P2-05 / P2-03 / P3-02 / P3-03 / P2-06）。本文档给出剩余三组
开放项的具体实施方案与代价评估，供决策后再动手。

---

## 一、P2-04 — affine 物理 spacing correction

### 现状
`_build_rotation_matrices` 的 aspect 共轭校正用 `A = diag(W, H, D)`（voxel-count），
使旋转在"体素数各向同性"坐标进行；当原始/target spacing 各向异性（尤其 z≫x/y）时，
相同角度对应的物理旋转不准确，会混入剪切/缩放。config 文档已在 batch 前修正表述，
但功能本身仍是 voxel-count 校正。

### 方案（推荐 B）
- **A. 全量 per-batch spacing 校正**：dataset 每个样本随 batch 携带
  `spacing (sz, sy, sx)`（npz meta 已有 z_spacing，xy 需 make_data 补记），
  `GPUAugmentor` 把 aspect 换成 `A = diag(W·sx, H·sy, D·sz)`（物理 extent）。
  代价：make_data manifest 格式升级 + dataset/trainer/augment 三层管道传参 +
  混 spacing batch 时逐样本矩阵（已是逐样本，无额外开销）。约 150–200 行改动。
- **B. 单一 dataset spacing 快速路径（推荐）**：绝大多数管线在 bake 阶段做
  spacing normalization → 全数据集单一 `target_spacing`。直接把
  `cfg.data.target_spacing`（或 manifest 回读值，与 `_resolve_sd_spacing` 同源）
  传进 `GPUAugmentor.__init__`，aspect 由 `diag(W,H,D)` 改为
  `diag(W·sx, H·sy, D·sz)`；未归一化且未设 target_spacing 时保持现状 + 一次性
  warning。改动集中在 augment.py/trainer.py，约 40 行；不改数据格式。
- 两案均补合成测试：各向异性 spacing 下旋转 90° 的几何 ground-truth 校验。

### 风险
改变增强分布 → 与旧 run 的训练曲线不可直接对比；建议加开关
`augment.physical_aspect_correct`（默认 off 保持复现，文档推荐 on）。

---

## 二、P2-01 + P2-02 — 位精确 resume（patch RNG 流 / DDP 多 rank RNG）

### 现状
- P2-01：persistent worker 内 `_rng_cache`（np Generator）不入 checkpoint，
  resume 后 worker 重建重播种 → patch 序列不可位精确重放。
- P2-02：checkpoint 只存 rank0 RNG；rank>0 恢复后 `_reseed_rank_rng` 重新分流，
  是"统计可复现"而非逐位续接。

### 方案（推荐 A；A 与 B 可各自独立落地）
- **A. 无状态计数派生 RNG（推荐，根治 P2-01）**：训练态 `_sample_rng()` 不再用
  流式 `_rng_cache`，改为
  `np.random.default_rng((train_seed, rank, epoch, sample_global_idx))`——
  与验证态 `(_VAL_SAMPLING_SEED, sample_idx)` 同构。epoch 由 sampler 的
  `set_epoch` 传入 dataset（DistributedSampler 已有该机制，MixedBatchSampler
  需补一个 `set_epoch` 透传）。RNG 变成纯函数后 **无状态可丢**，checkpoint
  无需保存任何 worker 状态，persistent_workers/num_workers 变化也不影响重放。
  代价：约 60 行 + 契约测试（同 (seed,epoch,idx) 两次采样 patch 位置逐位一致；
  resume 后下一 epoch 序列与不间断 run 一致）。
  注意：GPU 端增强 RNG（torch）另属 rank 主进程流，已由 rng_state 快照覆盖。
- **B. `rng_state_by_rank`（根治 P2-02）**：checkpoint 保存前 all_gather 各 rank
  的 `{torch_cpu, torch_cuda, numpy, python}` 状态（CPU tensor，几 KB/rank），
  rank0 写入 `rng_state_by_rank`；resume 时各 rank 取自己的槽位，找不到
  （world_size 变化）才回退 `_reseed_rank_rng`。代价：约 50 行；world_size
  改变时自动降级为现行为并 warning。
- **C. 最低成本替代**：不改代码，只把注释/文档的"位精确 resume"改为
  "模型/优化器状态位精确；数据流统计可复现"。若你认为逐位重放价值低，选 C。

---

## 三、P2-09 — medium patch 指标选模 → full-volume 复核

### 现状
默认 `val_metric_mode="medium"`：确定性 patch 指标（已消除采样抖动），但 z 上下文
/滑窗融合/空病例/全卷 FP 无法反映；save_best/early-stop 全依赖它。high 模式已存在
（整卷滑窗 + z_spacing 已接通，P2-10 已修），但每 epoch 跑代价大。

### 方案：混合评估器（medium 驱动 + high 复核）
新增 `train.val_high_confirm_every_n: int = 0`（0=关闭，保持现状）：
1. 每 epoch 照常跑 medium → 驱动 plateau/early-stop/日志（快、稳定）。
2. 当 medium 产生候选 best **或** 距上次 high 复核 ≥ N epoch 时，追加一次 high
   full-volume 评估；`is_best` 判定与 `best.pth` 落盘改由 **high 指标**
   （`val_high/*` 前缀入 history，图表单独成线）。
3. 训练结束时若最终 best 未经 high 确认，补跑一次。
实现点：`build_val_evaluator` 返回组合评估器（复用现有 Medium/High 两个类，
无需新指标代码）；Trainer best 决策块改读 confirm 结果；monitor 增
`val_high_*` 序列（charts 已按 key 动态渲染，基本免改）。约 120 行 + 测试
（用 stub evaluator 断言：候选 best 触发 high、best.pth 由 high 指标决定、
N-epoch 周期触发、关闭时行为与现状逐位一致）。

---

## 建议排期
| 批次 | 内容 | 预估改动 |
|---|---|---|
| batch5a | P2-04 方案 B（含开关）+ 合成测试 | ~60 行 |
| batch5b | P2-01 方案 A + P2-02 方案 B + resume 契约测试 | ~150 行 |
| batch5c | P2-09 混合评估器 + 测试 | ~150 行 |

三者互不依赖，可任选子集/顺序。回复形如"5a+5b"即可开工；对某项想改走
其他子方案（如 P2-04 走 A、P2-01/02 走 C）也请一并说明。
