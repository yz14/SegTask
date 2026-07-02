## 总结性描述

gentask 是从 segtask_v1 剥离出的超分生成工程：`train.py` / `predict.py` 提供训练与推理入口，`data.degradation.SuperResDegradation` 负责在线退化，`models/generation.py` 统一回归与扩散两条生成范式，`trainer/gen_trainer.py` 与 `predictor/gen_predictor.py` 承载训练 / 预测闭环。它支持 3D z-axis、3D cubic、2.5D 单视图、2.5D 多视图等生成几何，并已经补齐了生成专属适配特性：重要性图加权、multi-view aux reconstruction、auxiliary input conditioning。

## 详细说明

### 总览与模块树

```text
gentask/
├── __init__.py  # 包入口 / 对外导出
├── __main__.py  # 模块 CLI 入口
├── train.py  # 训练 CLI 入口
├── predict.py  # 推理 CLI 入口
├── logging_utils.py  # 日志初始化工具
├── utils.py  # 通用工具
├── docs/  # 生成模型综述与设计备忘
├── config/  # 配置系统（见下）
├── data/  # 数据子系统（见下）
├── losses/  # 损失库（见下）
├── models/  # 网络与生成接口（见下）
├── predictor/  # 推理器（见下）
└── trainer/  # 训练循环（见下）
```

### 配置系统

```text
gentask/config/
├── __init__.py  # 配置包入口
├── dataclasses.py  # Config / Task / Model dataclass 定义
├── io.py  # YAML 读写与 resolved config 保存
└── validation.py  # 生成任务配置校验
```

### 数据子系统

```text
gentask/data/
├── __init__.py  # 数据子系统入口
├── degradation.py  # 在线超分退化算子
├── loader.py  # 路径匹配与 dataloader 工厂
├── make_data.py  # 离线预打包 npz
├── specs.py  # patch / view 规格推导
└── dataset/  # npz 数据集子包（见下）
```

```text
gentask/data/dataset/
├── __init__.py  # dataset 子包入口
├── cache.py  # 缓存与 LRU 管理
├── core.py  # npz dataset 主逻辑
└── io.py  # npz / NIfTI 读取工具
```

### 损失库

```text
gentask/losses/
├── __init__.py  # 损失包入口
└── recon.py  # 重建损失与加权封装
```

### 模型与生成接口

```text
gentask/models/
├── __init__.py  # 模型包入口
├── adm_unet.py  # ADM backbone
├── blocks.py  # 共享积木与注意力模块
├── convnext.py  # ConvNeXt stage / block
├── diffusion.py  # 扩散 sampler / scheduler
├── edm2_unet.py  # EDM2 backbone
├── factory.py  # generation model 装配工厂
├── generation.py  # 回归 / 扩散统一接口
├── resnet.py  # ResNet block / stage
├── stem.py  # stem 与多视图融合
├── topology.py  # 派生输入/输出几何真相源
├── unet.py  # UNet encoder/decoder 主体
├── unet3p.py  # UNet3+ decoder
└── unetpp.py  # UNet++ decoder
```

### 推理器

```text
gentask/predictor/
├── __init__.py  # 预测器包入口
└── gen_predictor.py  # generation 推理器
```

### 训练循环

```text
gentask/trainer/
├── __init__.py  # 训练包入口
├── amp.py  # AMP / GradScaler 工具
├── checkpoint.py  # checkpoint 读写与兼容
├── gen_trainer.py  # generation 训练循环
├── memory.py  # 显存统计工具
└── optim.py  # 优化器 / 调度器 / warmup
```

### 关键设计

#### 两类生成范式

| 范式 | `task.algorithm` | 思路 | 复用 backbone |
|---|---|---|---|
| **回归复原** | `regression` | 单次前向把低分图直接映射回高分图，等价于 DnCNN / SRCNN / U-Net regression；可选残差学习 `HR-LR` | `unet` / `adm` / `edm2` 都可作图到图网络 |
| **条件扩散** | `diffusion` | 以低分图为条件，从噪声出发迭代去噪采样出高分图，思路接近 SR3 / Palette；支持 EDM 与 DDPM 两种参数化 | 仅 `adm` / `edm2`，并重新启用 timestep / σ 条件 |

#### 退化算子

- `data/degradation.py::SuperResDegradation` 负责在线退化：先从 HR 做下采样，再上采样回 HR 网格，得到低分输入。2.5D 时 D 维折进通道，按通道逐一退化。
- `sr_kernel` 支持 `area`、`trilinear`、`nearest`，可选高斯噪声。
- `sr_scale` 是各向同性缩放；`sr_scale_per_axis` 是各向异性缩放，3D 顺序按 `(D, H, W)`，2.5D 顺序按 `(H, W)`。CT 厚层→薄层通常写成 `[2, 1, 1]`，即只让 z 轴降采样。
- `sr_sampling` 控制抽样方式：
  - `blur`：默认的 SISR 路径，降采样后再上采样，形成同尺寸模糊输入；
  - `decimate`：VFI 插帧路径，沿退化轴抽稀保留帧，再线性插值填回原尺寸。

#### 损失、深监督与条件 backbone

- `losses/recon.py` 中的 `ReconstructionLoss` 供回归使用：Charbonnier / L1 / MSE，可选 `(1 - SSIM)` 与梯度 L1；`DiffusionLoss` 供扩散使用，为逐样本加权 MSE。验证仍报告 `psnr` / `ssim`。
- `model.deep_supervision=true` 只对回归路径生效：复用 backbone 的多尺度解码头，并由 `loss.deep_supervision_weights` 聚合各尺度重建损失。
- ADM / EDM2 的 timestep / σ 条件只在扩散路径启用；非扩散路径（`emb_channels==0`）会整体跳过条件分支，行为与以前一致。

#### 扩散框架与统一模型接口

- `models/diffusion.py` 提供两个 sampler：
  - `EDMDiffusion`：去噪预条件 + Heun 二阶采样；
  - `DDPMDiffusion`：ε-预测 + 祖先 / DDIM 采样。
- `models/generation.py` 统一回归与扩散接口：
  - `forward(hr)`：在线退化并返回训练所需三元组；
  - `restore(lr)`：从低分图恢复高分图；
  - `degrade(hr)`：验证时单独查看退化结果；
  - `models/factory.build_model` 会根据 `cfg.is_generation` 自动分派到生成路径。

### 生成专属适配特性

- **重要性图加权**：`data.region_weight_dir` / `data.region_weight_suffix` 指向预计算的 per-voxel importance map（label-independent，读取时做 `+1` offset），`make_data` 会把它与 image / label 一起烘焙进 npz，训练时进入 `batch["weight_map"]` 并被 `ReconstructionLoss` 使用；若同时设置 `data.region_weights`（按标签的旧路径），预计算重要性图优先生效。
- **多视图辅助重建**：`model.aux_seg_supervision=true` 仍沿用旧 YAML 键名，但当前语义是“辅助重建监督”。仅在 `patch_mode="2_5d"` 且 `len(data.multi_res_scales)>1` 时启用；`data.multi_res_scales` 必须全部 `>= 1.0`（例如 `[1.0, 2.0]`，辅助视图是更大物理 FOV）。`loss.aux_recon_weights` 可显式指定各辅助视图权重，空列表则回退到 `0.5^k`。
- **辅助输入条件**：`data.cond_dirs` / `data.cond_suffixes` / `data.cond_normalize` / `data.cond_intensity_*` / `data.cond_global_*` 指定与图像严格配准的外部条件体（mask / 其他模态 / 预分割等）；条件体不做退化，直接拼到模型输入作为独立融合流。当前支持单视图回归、单视图 2.5D、扩散，以及多视图 Plan A；`2.5D-lift + cond` 仍是未支持并带明确报错。

### 运行与示例

- 训练 / 推理入口：`python -m gentask.train --config configs/gensr_*.yaml`、`python -m gentask.predict --config ...`
- 示例配置：`configs/gensr_2_5d_regression.yaml`、`configs/gensr_2_5d_diffusion_adm.yaml`、`configs/gensr_3d_zaxis_regression.yaml`、`configs/gensr_3d_zaxis_vfi.yaml`、`configs/gensr_3d_zaxis_region_weight.yaml`、`configs/gensr_2_5d_multiview_aux.yaml`、`configs/gensr_3d_zaxis_cond.yaml`。
- 注意：生成任务忽略分割标签（训练目标是干净图自身），但 dataloader 仍需 `label_dir`；`task.out_channels==1`（CT 灰度）仍是当前约束，其他几何/条件特性按各自配置项启用。
- 冒烟测试：`python tests/test_generation_smoke.py`、`python tests/test_data_pipeline_smoke.py`。
