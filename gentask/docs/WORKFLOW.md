# gentask 生成（超分/复原）流程总览

> 入口：`python -m gentask.train --config configs/gensr_2_5d_regression.yaml | gensr_2_5d_diffusion_adm.yaml`；
> 推理：`python -m gentask.predict --config ... --ckpt ...`。
> YAML `task.type='generation'` 启用生成任务（当前退化仅超分 `superres`），`task.algorithm` 选两类范式：
> **regression**（前馈回归复原）与 **diffusion**（条件扩散采样）。数据/几何/优化基建与分割同构，
> 输入输出几何、通道布局和视图关系统一由 `models.topology` 推导（唯一真相源）。

---

## 0. 共享主干

```
配置加载（task/data/model/train/predict dataclass + 校验）→ npz 发现（make_data 预烘包）
 → Dataset 抽 max-FOV cube → GPU 同步 3D 增强（生成变体）→ Pipeline 中心裁剪/视图拆分
 → 模型 forward(hr)（内部在线退化 HR → LR 条件图，GPU、batch 级）→ 回归/扩散损失 → backward → EMA
 → 验证（degrade → restore → PSNR/SSIM 选模）
 → 推理：NIfTI 整卷 → 滑窗 restore → 反归一化写出 *_sr.nii.gz
```

### 几何（`patch_mode`，与分割同语义）

| 几何 | patch_mode | 模型输入 | 退化作用轴 |
|---|---|---|---|
| 3D | `whole` / `z_axis` / `cubic` | (B, 1, D, H, W) | 逐空间轴 (D,H,W)，`[s,1,1]` 即只超分 z 轴 |
| 2.5D | `2_5d` | (B, D, H, W)，D 折进通道 | 仅 (H,W) 面内（D 视作通道，逐切片 2D） |

### 数据契约

- 训练只读 npz（`data.npz_dir` 必设；空目录且 `npz_auto_build=true` 时内联调 `make_data.prepare_dataset` 从 NIfTI 烘焙）；
- `make_data`：bbox 裁剪 + 预计算 fg 索引 → `<pid>.npz`，多 worker mmap 共享 OS page cache；
- 可选逐样本条件卷（`data.cond_dirs`，如 mask/预分割）：独立强度窗归一化，随 image 同步空间变换；
- 无 label 参与训练：干净 HR 图自身即重建目标。

---

## 1. 训练流程

```
【Dataset，CPU worker → (B, 1, eD, eH, eW) max-FOV 过采样 cube】
npz 读取（image + 可选 cond/rw + fg 索引）→ 预处理（img 归一化）
 → 按 patch_mode 抽 cube（whole 整卷 resize / z_axis z 滑动抖中心 + H/W 保持全尺寸整面 resize
   / cubic 三轴随机中心；越界 edge 复制；oversample 留增强余量）

【Trainer，GPU】
GPU 同步 3D 增强（生成变体：空间变换 image+cond+wmap 同步 warp，强度变换仅 image）
 → Pipeline 中心裁剪/视图拆分（见下）
 → 模型 forward(hr)：内部先在线退化造 LR 条件图——
   HR --下采样(sr_kernel，area≈部分容积)--> LR 域（可选高斯噪声）
   --上采样(sr_kernel_up)--> 与 HR 同尺寸的 LR 输入（pre-upsampling；SISR 网络则保持真 LR 网格）
   （sr_scale 各向同性 / sr_scale_per_axis 逐轴，CT 厚层→薄层 = [s,1,1] 只退化 z；
    sr_kernel_pool / sr_noise_std_range 训练随机退化，验证/推理固定，指标可比）
 → 前向 → {pred, target(, weight)} → fp32 损失 → backward → optimizer/EMA
```

### 视图 Pipeline（dataset 恒发单条 max-FOV cube，几何全在 GPU 侧完成）

```
【Pipeline：VanillaPipeline → 模型布局】
单视图（n_views=1）通用：仅中心裁剪掉过采样余量回 patch_size

【Pipeline：StackedMultiResPipeline → (B, n_views, pD, pH, pW)】
逐视图同中心裁 FOV + resize 回 patch_size → 通道堆叠
（覆盖 z_axis/cubic 3D 多分辨率、2.5D 统一深度与 lift_2_5d_to_3d；
 2.5D 非 lift 由模型折叠为 (B, n_views·D, H, W)，lift 保持 rank-5 走真 3D）

【Pipeline：NativeDPipeline → (B, ΣD_k, H, W)】
2.5D keep_native_view_depth：逐视图中心裁原生深度 z-slab（不 resize）→ 按通道拼接
```

### 两类算法（`task.algorithm`，统一 `forward(hr) / restore(lr) / degrade(hr)` 接口）

```
【regression】前馈回归复原
backbone：UNet 系（unet/unetpp/unet3p × resnet/convnext，2.5D/3D）| ADM | EDM2（仅 2.5D）
 | EDSR / RCAN（post-upsampling SISR：真 LR 网格提特征 → AnisoPixelShuffle+ICNR 上采头，
   支持各向异性倍率，配 degrade(keep_lr_size=True)）
可选残差学习（task.residual：预测 HR−LR，输出 = pred + LR）
损失 ReconstructionLoss = Charbonnier/L1/MSE + ssim_weight·(1−SSIM) + grad_weight·梯度L1

【diffusion】条件扩散（类 SR3/Palette，仅 2.5D，backbone = ADM / EDM2 扩散版）
x_cat = cat([噪声图(预条件缩放), LR 条件图], dim=1)，c_noise 标量嵌入贯穿各 ResBlock
 → 参数化 edm（Karras σ 加权）| ddpm_eps（ε-预测，linear/cosine β）
 → DiffusionLoss：逐样本加权 MSE（σ/时间步权重由 wrapper 给出）
推理 sample(cond) 迭代去噪：edm_heun（二阶）/ edm_euler / ddpm / ddim，sample_steps 步
```

---

## 2. 通用训练技巧（`train.*`）

| 技巧 | 说明 |
|---|---|
| 混合精度 AMP | `use_amp` / `amp_dtype` + GradScaler；损失 fp32 |
| EMA | `use_ema`；验证与 best 保存均用 EMA shadow |
| 非有限守护 | loss 非有限不 backward、整个 accum 组丢弃；unscale 后梯度范数非有限跳过 optimizer step；仅有效步更新 EMA |
| fused AdamW / wd 分组 | `adamw_fused`（默认 true，仅 CUDA 生效）；norm/bias 参数免 weight decay（口径同 seg） |
| 梯度累积 / 裁剪 | `grad_accum_steps` / `grad_clip_norm` |
| warmup + scheduler | `warmup_epochs` + cosine/poly/step/plateau 等 |
| torch.compile | `compile_mode` |
| 选模 / 早停 | patch 级 PSNR 越大越好；`val_full_volume=true` 时改用整卷 PSNR；`early_stopping`；扩散验证采样用固定 seed generator，选模/早停/plateau 不受采样噪声干扰 |
| 整卷验证 | 与部署同口径：在线退化整卷 → 推理器滑窗复原（复用 predict.overlap/blend）→ 逐卷 PSNR/SSIM；`val_full_volume_max` 控耗时 |
| 续训 / 迁移 | `resume`（模型+optimizer/scheduler/scaler/EMA，history.json 续接）；`pretrain`（strict 可配，可载 EMA 权重） |
| 训练历史落盘 | history.json 逐 epoch 原子写（epoch/lr/训练与验证指标） |
| 离线预烘包 | `python -m gentask.data.make_data`：NIfTI → bbox 裁剪 npz + fg 索引，训练零重扫 |

---

## 3. 验证指标

- patch 级：`degrade(hr)` → `restore(lr)` → PSNR / SSIM（归一化域，data_range 按 minmax/zscore 自适应），并报告 LR 基线 PSNR 作参照（增益读数）；
- 整卷（可选）：`vol_psnr / vol_ssim / vol_psnr_lr`，滑窗路径与推理完全一致；
- 验证/推理固定 `sr_kernel` / `sr_noise_std`（训练侧的随机退化池不影响可比性）；
- 扩散采样：验证/推理传固定 seed 的 torch.Generator（初始噪声与 DDIM/祖先采样逐步噪声均走 generator），采样逐位可复现；训练未设 generator 时仍随机。

---

## 4. 推理（整卷 → 复原 NIfTI）

```
NIfTI 读取 → 归一化 → 入网网格（predict.input_grid）
  hr：输入已在 HR 网格（在线退化实验 / 已插值体）
  lr：真实厚层输入，先按训练同参重采样到 HR 网格（blur 用 sr_kernel_up 插值；
      decimate 用相位对齐线性插值）；target_z_spacing>0 时逐体读 z spacing 自适应倍率
 → 复原
  whole：整卷单次前向
  z_axis / cubic：3D 三轴滑窗（stride = size·(1−overlap)，末窗贴边）
   → 重叠区加权融合（blend：gaussian 中心高权消接缝 / uniform 等权）
  2_5d：沿 z 逐 slab 滑窗（同 overlap/blend，z 向权重融合）；slab 内同中心多 FOV 视图与训练一致
 → denormalize（归一化域 → 原强度 HU，保留物理标定）→ 写出 *_sr.nii.gz
```

- 推理 AMP：`predict.use_amp`（默认 true，仅 CUDA autocast，dtype 同 `train.amp_dtype`）；
- 翻转 TTA：`predict.tta_flips`（默认 false），输入与 cond 同翻、预测回翻后均值；
  2.5D 非 lift 仅 H/W 轴；`sr_sampling='decimate'` 时自动排除被退化轴（防相位错位）；
- 扩散推理采样用固定 seed generator，两次推理逐位一致；
- 小数据集拦截：训练集样本数 < batch_size 时（drop_last=True 会产生零批次）显式报错。

条件卷推理侧按 `cond_dirs/cond_suffixes` 逐体配对，同窗裁剪送入 `restore(lr, cond=...)`；cond 整卷与输入形状不符时显式报错。

---

## 5. 一致性契约

- 训练-推理退化同参：`sr_scale(_per_axis)` / `sr_kernel(_up)` / `sr_sampling` 决定训练造 LR 与推理入网重采样，必须一致；
- patch 几何一致：推理滑窗 patch 尺寸 = 训练 `data.patch_size`，2.5D slab 深度 = `patch_size[0]`，多视图 FOV/深度与训练同拓扑（`build_topology` 统一推导）；
- 2.5D 恒把 D 折进通道：退化只作用 (H,W)，SSIM/梯度损失逐通道计算，与"D 视作通道"设定一致；
- 条件卷 cond 与 image 空间对齐：训练侧同 warp、推理侧同窗裁剪，强度归一化各自独立；
- SISR（edsr/rcan）为 post-upsampling：输出网格 = 输入 × 倍率，倍率固定，不支持 `target_z_spacing` 自适应；
- 已知待定（暂缓）：whole/z_axis/2_5d 训练侧整卷/面内 resize 到 patch 尺寸，而推理侧在原生分辨率滑窗/整卷前向，两侧频谱分布不等价；修改方向（候选：训练侧改原生分辨率裁剪）涉及已训模型兼容，尚未实施。
