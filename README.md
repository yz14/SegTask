# SegTask · 通用 2.5D / 3D 医学图像分割框架

> 一个面向科研场景的分割训练管线：**单分辨率 NIfTI / pre-baked npz** 数据 → **YAML 全参数驱动** 的训练/推理 → **2.5D / z-slab / cubic / whole** 四种 patch 模式 → **ResNet / R(2+1)D / ConvNeXt × UNet / UNet++ / UNet3+** 任意拼装 → 多 FOV 上下文融合（Plan A 早融合 / Plan C 分层注入）+ aux 监督 + Plan A lift-to-3D。
>
> 训练入口：`@d:\codes\work-projects\SegTask\segtask_v1\train.py`。**`segtask_v1/`** 是当前活跃的代码路径；`segtask/` 是早期 v0 原型，保留作参考。

---

## 目录

- [SegTask · 通用 2.5D / 3D 医学图像分割框架](#segtask--通用-25d--3d-医学图像分割框架)
  - [目录](#目录)
  - [1. 整体架构](#1-整体架构)
    - [1.1 技术栈](#11-技术栈)
    - [1.2 顶层目录结构](#12-顶层目录结构)
    - [1.3 端到端数据流](#13-端到端数据流)
    - [1.4 Patch 模式与多分辨率/感受野（FOV）矩阵](#14-patch-模式与多分辨率感受野fov矩阵)
    - [1.5 文件命名约定与 pid](#15-文件命名约定与-pid)
  - [2. 快速开始](#2-快速开始)
    - [2.1 环境](#21-环境)
    - [2.2 训练](#22-训练)
    - [2.3 预测](#23-预测)
    - [2.4 npz 预打包（可选，强烈推荐）](#24-npz-预打包可选强烈推荐)
    - [2.5 排除坏 NIfTI 样本](#25-排除坏-nifti-样本)
  - [3. `segtask_v1/` 模块详解](#3-segtask_v1-模块详解)
    - [3.1 `train.py` / `predict.py` / `__main__.py` —— CLI 入口](#31-trainpy--predictpy--__main__py--cli-入口)
    - [3.2 `config.py` —— dataclass + YAML 配置系统](#32-configpy--dataclass--yaml-配置系统)
    - [3.3 `utils.py` —— 通用工具](#33-utilspy--通用工具)
    - [3.4 `data/` —— 数据发现 / IO / patch / 增强](#34-data--数据发现--io--patch--增强)
    - [3.5 `models/` —— UNet 家族 + Block 仓库](#35-models--unet-家族--block-仓库)
    - [3.6 `losses/` —— 二元 sigmoid 损失库 + 多分辨率/2.5D 包装器](#36-losses--二元-sigmoid-损失库--多分辨率25d-包装器)
    - [3.7 `trainer.py` —— 训练循环](#37-trainerpy--训练循环)
    - [3.8 `predictor.py` —— 滑动窗口推理](#38-predictorpy--滑动窗口推理)
  - [4. `configs/` —— YAML 配置 + 实验脚本](#4-configs--yaml-配置--实验脚本)
  - [5. `tools/` —— 数据集体检工具](#5-tools--数据集体检工具)
  - [6. 根目录测试脚本](#6-根目录测试脚本)
  - [7. `segtask/` —— 早期 v0 原型（已冻结）](#7-segtask--早期-v0-原型已冻结)
  - [8. 关键交互时序](#8-关键交互时序)
    - [8.1 一次训练 step（2.5D 多 FOV）](#81-一次训练-step25d-多-fov)
    - [8.2 一次推理（z\_axis 滑窗 + TTA）](#82-一次推理z_axis-滑窗--tta)
  - [9. 扩展指南](#9-扩展指南)

---

## 1. 整体架构

### 1.1 技术栈

| 层 | 技术 | 关键文件 |
|---|---|---|
| 训练循环 | 原生 PyTorch ≥ 2.0（AMP fp16/bf16、`torch.compile`、单 GPU） | `@d:\codes\work-projects\SegTask\segtask_v1\trainer.py` |
| 配置 | `@dataclass` 嵌套 + PyYAML，CLI dot-notation override | `@d:\codes\work-projects\SegTask\segtask_v1\config.py` |
| 医学图像 IO | SimpleITK（首选，带退避重试）+ numpy `npz mmap` | `@d:\codes\work-projects\SegTask\segtask_v1\data\dataset.py:32-419` |
| 数据增强 | 纯 GPU（`grid_sample` 共享 warp、弹性形变、强度抖动） | `@d:\codes\work-projects\SegTask\segtask_v1\data\augment.py` |
| 模型 | UNet / UNet++ / UNet3+ × ResNet(basic/preact/bottleneck/R(2+1)D) / ConvNeXt | `@d:\codes\work-projects\SegTask\segtask_v1\models\unet.py` |
| 多 FOV 融合 | shared\_stem / multi\_stem\_proj (Plan A) / hierarchical (Plan C) | `@d:\codes\work-projects\SegTask\segtask_v1\models\stem.py` |
| 注意力 | SE / ECA / CBAM / CoordAttention + AttentionGate (skip) | `@d:\codes\work-projects\SegTask\segtask_v1\models\blocks.py:146-435` |
| 上下采样 | conv / max / avg / **BlurPool** / PixelUnShuffle ↔ transpose / linear / nearest / PixelShuffle / **CARAFE** / **DySample** | `@d:\codes\work-projects\SegTask\segtask_v1\models\blocks.py:611-948` |
| 损失 | Dice / BCE / Focal / Tversky / **GDL** / **FocalTversky** / **Lovász-Hinge** / **clDice** + compound + DS + MultiRes + SliceChannel | `@d:\codes\work-projects\SegTask\segtask_v1\losses\losses.py` |
| 推理 | z\_axis / cubic / whole / z-interleave 滑窗 + Gaussian/uniform blend + TTA-flip + bbox crop | `@d:\codes\work-projects\SegTask\segtask_v1\predictor\` |

### 1.2 顶层目录结构

```text
SegTask/
├── README.md                          # 本文件
├── TODO.md                            # 工作规范 + 待办（核心原则在最前）
├── .gitignore
├── configs/                           # YAML 配置 + 实验脚本（§4）
│   ├── default.yaml                   # 全参数注释版默认配置（2.5D body 任务示例）
│   ├── seg2_5d.yaml                   # 2.5D 生产配置（多 FOV + aux 监督）
│   ├── seg3d.yaml                     # 3D 生产配置（z_axis / cubic / whole）
│   ├── test_e2e.yaml                  # 端到端 smoke 用的小配置
│   └── experiments/
│       ├── lift_a_baseline_2_5d.yaml  # Plan A lift 对照：纯 2.5D baseline
│       ├── lift_a_planA.yaml          # Plan A lift：2.5D → 3D + R(2+1)D
│       ├── lift_a_planA_aux.yaml      # Plan A lift + aux_seg_supervision
│       ├── lift_a_ref_3d.yaml         # 纯 3D 参考实验
│       ├── seg2_5d_small_base.yaml    # small_data 12 例的最小配置
│       └── run_aux_sweep.py           # 自动跑 A0/A1/C1/C2 4 组对比
│
├── segtask_v1/                        # 当前活跃的训练管线（§3）
│   ├── __init__.py
│   ├── __main__.py                    # 让 `python -m segtask_v1` 直接跑训练
│   ├── train.py                       # 训练 CLI（YAML + --override）
│   ├── predict.py                     # 推理 CLI（单文件/目录、可选 bbox 裁剪）
│   ├── config.py                      # ~1450 行的全部 @dataclass 配置系统（§3.2）
│   ├── utils.py                       # AverageMeter / ModelEMA / Timer / Dice
│   ├── trainer/                       # 训练循环（R1–R3 模块化包）
│   │   ├── trainer.py                 # ~700 行 Trainer 类（控制流）
│   │   ├── pipelines/                 # ViewPipeline 策略 + 5 个具体实现
│   │   ├── views.py / optim.py / amp.py / memory.py / breakdown.py / checkpoint.py
│   ├── predictor/                     # 滑动窗口推理（R6 模块化包）
│   │   ├── predictor.py               # ~450 行 Predictor 类（shim + 入口）
│   │   ├── sliding.py                 # 4 种 sliding 主循环（whole/z/z-interleave/cubic）
│   │   ├── inputs.py                  # 6 个 window/batch builders
│   │   ├── forwards.py                # forward + TTA + diag
│   │   ├── blending.py                # 几何/概率 helpers
│   │   └── io.py                      # run_inference + ckpt + precision
│   ├── data/                          # 数据子系统（§3.4）
│   │   ├── __init__.py
│   │   ├── dataset.py                 # 3 个 Dataset + 所有 IO / patch / bbox / npz 工具
│   │   ├── specs.py                   # DatasetSpec 策略对象（R4 引入）：模式选 dataset 类
│   │   ├── loader.py                  # 发现-匹配-切分-build_dataloaders（已瘦身到 ~530 行）
│   │   ├── augment.py                 # GPUAugmentor（共享 warp，多 FOV 安全）
│   │   └── make_data.py               # 一次性把 NIfTI 打包成 bbox-cropped npz
│   ├── models/                        # 网络结构（§3.5）
│   │   ├── __init__.py
│   │   ├── blocks.py                  # ~950 行通用积木（conv/norm/attn/up-down sample）
│   │   ├── stem.py                    # 5 种 stem + 3 种多视图融合策略
│   │   ├── resnet.py                  # 4 种 ResNet 残差块 + ResNetStage
│   │   ├── convnext.py                # ConvNeXt 块 + 论文 norm-first downsample
│   │   ├── unet.py                    # 通用 UNet3D（spatial_dims=2 兼容 2.5D）
│   │   ├── unetpp.py                  # UNet++ 嵌套稠密 decoder
│   │   ├── unet3p.py                  # UNet3+ 全尺度跳连 decoder
│   │   ├── topology.py                # ModelTopology + build_topology（R5 引入）：派生量单一真相源
│   │   └── factory.py                 # 从 Config 装配 Encoder/Decoder/UNet3D（读 topology）
│   └── losses/                        # 损失函数（§3.6）
│       ├── __init__.py
│       └── losses.py                  # ~1270 行：基础 loss × wrapper × build_loss 工厂
│
├── tools/                             # 数据集体检脚本（§5）
│   ├── scan_bad_nifti.py              # 扫描 SimpleITK 读不开的非正交 NIfTI
│   └── bad_seg2_5d/                   # scan_bad_nifti 的输出落地（pid 列表 + CSV）
│
├── segtask/                           # ⚠ 已冻结的 v0 原型，仅作参考（§7）
│   ├── train.py / predict.py / trainer.py / predictor.py / config.py
│   ├── utils.py / visualization.py
│   ├── data/ ·  models/ ·  losses/
│
├── test_*.py / smoke_test_*.py        # 19 份功能 / 回归测试（§6）
├── experiments/   outputs/            # 训练副产物（checkpoints / logs / resolved_config.yaml）
├── .pytest_cache/ · __pycache__/      # 工具产物
```

### 1.3 端到端数据流

```text
                                            训练时
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  train.py                                                                            │
│    │  load_config(yaml) + apply_overrides  →  cfg.sync() + cfg.validate()            │
│    ▼                                                                                 │
│  data.loader.build_dataloaders(cfg)                                                  │
│    │     discover_samples → exclude_list 过滤 → (可选) match_bbox / match_rw         │
│    │     detect_label_values → stratified_train_val_split                            │
│    │     data.specs.build_data_spec(cfg) → spec.make_split(train|val, common)        │
│    │     spec ∈ { ZCubeSpec | WholeSpec | CubicSpec }（唯一 patch_mode 决策点）      │
│    ▼                                                                                 │
│  __getitem__:  load_nifti_cropped → preprocess_image / label → 抽 patch              │
│                 (z_axis = 沿 z 滑窗；cubic = 3D 立方；whole = 整卷 resize；          │
│                  2_5d = 复用 z_axis，多 FOV 沿 D 拼通道)                              │
│    ▼  (B, C_res, D, H, W) 单分辨率 cube — 多 FOV 仍是单分辨率，只是更大的最大 FOV     │
│  GPUAugmentor (CUDA, 共享 grid_sample 的 affine / elastic warp)                      │
│    ▼                                                                                 │
│  Trainer._split_views_*      做 per-view 中心裁剪 + resize（多 FOV 在此刻产生不同分辨率）│
│    │   2.5D 模式  squeeze C_res=1 把 D 折进通道：(B, D*n_views, H, W)                │
│    │   3D 模式    保持 (B, C_res, D, H, W)                                            │
│    ▼                                                                                 │
│  UNet3D(spatial_dims=2 或 3)  →  main + DS + per-view aux 输出                       │
│    ▼                                                                                 │
│  Loss = MultiResolutionLoss(DeepSupervisionLoss(SliceChannelLoss(base)))             │
│    │   + Σ_k aux_w_k * L_aux(view_k)                                                 │
│    ▼   AMP backward → grad-clip → optimizer.step → scheduler.step → ema.update       │
│  validate → pooled Dice → 保存 best / last checkpoint                                │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Patch 模式与多分辨率/感受野（FOV）矩阵

| 模式 | 输入张量 | 切块方式 | 多 FOV 语义 (`multi_res_scales`) | 模型 `spatial_dims` |
|---|---|---|---|---|
| `z_axis` | `(B, C_res, D, H, W)` | 沿 z 滑窗，H/W 取全分辨率 | 仅 z 轴缩放：scale=k 取 `round(D·k)` 切片后 resize 回 D | 3 |
| `cubic` | `(B, C_res, D, H, W)` | 3D 立方滑窗，xyz 全可移动 | 三轴一起缩放：取 `extract_size·k` 后 resize 回 `extract_size` | 3 |
| `whole` | `(B, 1, D, H, W)` | 整卷 resize 到 `patch_size` | 必须为 `[1.0]`（缩放无物理意义） | 3 |
| `2_5d` | `(B, D·n_views, H, W)`（默认）<br/>或 `(B, Σ_k D_k, H, W)`（`aux_keep_native_d=True`） | 复用 z_axis 路径，aug 后 `squeeze(C_res=1)` 把 D 折入通道 | 每个 FOV 占 D 个通道；可选 aux 监督独立预测每个 view | 2（lift 模式恢复为 3） |

**单分辨率读取原则**（贯穿数据层）：数据集只按 *最大* FOV 提一份 cube，augmentation 也只跑一次共享 warp；按 view 的中心裁剪 + resize 推迟到 trainer 进入模型前一刻才做（`data.keep_native_multi_res=True` / `data.aux_keep_native_d=True`）。这样既消除多次 zoom 引入的高频损失，又保证多 view 之间的几何完全一致。

### 1.5 文件命名约定与 pid

- **pid**：`image_path.name` 去掉 `data.image_suffix`（默认 `.nii.gz`）后的部分。整个管线（npz、bbox、region\_weight、exclude\_list）都以 pid 为索引键。
- **配对契约**：当 `bbox_dir` / `region_weight_dir` / `npz_dir` 非空时，**每个**图像 pid 必须有对应文件，否则 `FileNotFoundError`。这是有意的强契约，避免静默丢样本。
- **npz 包**（`make_data` 产物）：`<npz_dir>/<pid>.npz`，键 `image / label / [rw] / fg_slices / fg_coords / meta`。详见 `@d:\codes\work-projects\SegTask\segtask_v1\data\make_data.py:32-70`。

---

## 2. 快速开始

### 2.1 环境

> 测试环境：**`conda activate torch27_env`**。`conda run` 会有明显启动开销，推荐直接调用环境内的 `python.exe`。

依赖：`torch ≥ 2.0`、`SimpleITK`、`numpy`、`scipy`、`pyyaml`、`tqdm`。

### 2.2 训练

```powershell
# 最简：YAML 全权决定
python -m segtask_v1.train --config configs/seg2_5d.yaml

# 临时 override（dot-notation，类型按 dataclass 字段自动 cast）
python -m segtask_v1.train --config configs/seg2_5d.yaml `
    --override train.epochs=50 model.backbone=convnext data.batch_size=4

# 也可直接跑文件
python segtask_v1/train.py --config configs/seg3d.yaml
```

每次训练在 `cfg.train.output_dir` 下写：

- `train.log` —— 控制台 + 文件日志
- `resolved_config.yaml` —— `sync()` + override 之后的最终配置（用来复现/继续训练）
- `last_model.pth` / `best_model.pth` / `epoch_*.pth` —— full-state checkpoint（model / EMA / optimizer / scheduler / scaler / RNG / early-stop 计数）

### 2.3 预测

```powershell
# 单文件
python -m segtask_v1.predict --config configs/seg2_5d.yaml `
    --checkpoint outputs/body/best_model.pth --input case_001.nii.gz

# 整个目录（递归）
python -m segtask_v1.predict --config configs/seg2_5d.yaml `
    --checkpoint outputs/body/best_model.pth `
    --input F:/BaiduNetdiskDownload/lung_nii --output F:/.../body_pred

# 强制使用 EMA 权重 + 保存概率图 + 临时调推理参数
python -m segtask_v1.predict ... --weights ema --save-probs `
    --override predict.batch_size=4 predict.tta_flip=true

# 显式 bbox 裁剪（优先级：--bbox 文件/目录 > cfg.data.bbox_dir > 无 bbox）
python -m segtask_v1.predict ... --bbox F:/path/to/bbox_dir
# 强制不用 bbox（即使 cfg 里配了 bbox_dir）
python -m segtask_v1.predict ... --bbox ""
```

### 2.4 npz 预打包（可选，强烈推荐）

`.nii.gz` 不能流式 seek —— ITK 必须把整卷解 gzip 到 native dtype 才能 ROI 裁剪。当 `num_workers ≥ 4` × `(image, label, rw)` = 12 个并发 gzip-decompress 时，16 GiB 机器很容易 OOM。`make_data` 把每个 pid 一次性烘焙成 bbox-cropped npz（`np.load(..., mmap_mode='r')` 可被多 worker 共享 OS page cache）：

```powershell
python -m segtask_v1.data.make_data --config configs/seg2_5d.yaml `
    --out-dir F:/cache/seg2_5d_npz --workers 8
```

随后只要在 YAML 把 `data.npz_dir` 指过去，整个数据层即换成 mmap 路径（`bbox_dir / image_dir / label_dir / region_weight_dir` 全部被忽略）。`data.npz_auto_build=True`（默认）时若 `npz_dir` 空目录，trainer 启动会自动跑一次构建。

### 2.5 排除坏 NIfTI 样本

部分扫描的 qform/sform 非正交，SimpleITK 直接抛 `ITK only supports orthonormal direction cosines`：

```powershell
python tools/scan_bad_nifti.py --config configs/seg2_5d.yaml --out tools/bad_seg2_5d
# 然后在 YAML 里设置 data.exclude_list: "tools/bad_seg2_5d/bad_pids.txt"
```

---

## 3. `segtask_v1/` 模块详解

### 3.1 `train.py` / `predict.py` / `__main__.py` —— CLI 入口

```text
segtask_v1/
├── __init__.py        # 占位包 docstring
├── __main__.py        # 让 `python -m segtask_v1` 等价于 `python -m segtask_v1.train`
├── train.py           # 训练 CLI；负责 logging / seed / device / 调度 Trainer.fit()
└── predict.py         # 推理 CLI；解析 --input/--bbox/--weights/--save-probs，调度 run_inference()
```

`train.py` 关键函数：
- `setup_logging(output_dir, level)`：同时写 stdout 和 `output_dir/train.log`。
- `apply_overrides(cfg, ["a.b.c=42", ...])`：dot-notation override，按字段当前类型 cast（bool/int/float/list-JSON/str）。
- `main()`：`load_config → override → sync → validate → seed → build_dataloaders → build_model → save_config → Trainer.fit`。

`predict.py` 关键函数：
- `_gather_nifti(input, recursive)`：把 `--input` 展开成 `.nii / .nii.gz` 列表。
- `_resolve_bbox_paths(--bbox, image_paths, cfg)`：实现 `--bbox` 优先级（显式文件/目录 > cfg.data.bbox\_dir > 关闭）。
- `main()`：解析 → 复用 `train.setup_logging` / `apply_overrides` → `run_inference(cfg, ckpt, image_paths, weight_variant, bbox_paths)`。

### 3.2 `config.py` —— dataclass + YAML 配置系统

~1450 行，6 个 `@dataclass` + 顶层装配 + `sync()` / `validate()` / `load_config()` / `save_config()`：

```text
config.py
├── DataConfig            # 数据路径、patch_mode、multi_res_scales、bbox/rw、npz、aug_oversample、
│                         # z_boundary_mode（stretch | edge_pad）、aux_keep_native_d、
│                         # keep_native_multi_res、stratified_split、dataloader 参数…
├── AugConfig             # GPU 增强各类概率与范围（flip / affine / elastic / dropout / brightness …）
├── ModelConfig           # backbone (resnet | convnext) / spatial_dims / block_type
│                         # (basic | preact | bottleneck | r2plus1d) / resenc_preset (none|S|M|L|XL)
│                         # / encoder_channels / blocks_per_stage / norm / activation / dropout
│                         # / attention_type / skip_attention / deep_supervision
│                         # / aux_seg_supervision + aux_head_mode + lift_2_5d_to_3d
│                         # / stem_mode + context_fusion (shared_stem | multi_stem_proj | hierarchical)
│                         # / decoder_type (unet | unetpp | unet3p) + unet3p_cat_channels
│                         # / downsample_mode + upsample_mode + skip_mode + drop_path_rate
│                         # / convnext_layer_scale_init + convnext_downsample_lnfirst
├── LossConfig            # name (dice | bce | focal | tversky | gdl | focal_tversky | lovasz | cldice
│                         # | dice_bce | dice_focal | dice_tversky | …) + compound_weights
│                         # / class_weights / region_weights / batch_dice / ignore_empty
│                         # / gdl_weight_type / focal_tversky_gamma / lovasz_per_sample / cldice_iter
│                         # / deep_supervision_weights / slice_loss_reduction (per_slice | per_volume)
│                         # / aux_supervision_weights
├── TrainConfig           # epochs / optimizer / scheduler (含 warmup / cosine_warm_restarts / poly /
│                         # step / plateau / one_cycle) / grad_accum / grad_clip / AMP (use_amp +
│                         # amp_dtype: auto | fp16 | bf16) / compile_mode / EMA / output_dir /
│                         # save_every / save_best_metric / early_stopping / seed / deterministic
│                         # / resume / pretrain (+ strict / load_ema)
└── PredictConfig         # z_overlap / blend_mode (gaussian | average) / batch_size / tta_flip /
                          # threshold / output_dir / save_probabilities
```

- `Config.sync()`：把派生字段填齐——`num_classes ← len(label_values)`、`z_boundary_mode` 自动升级到 `edge_pad`、`resenc_preset` 展开成 `encoder_blocks_per_stage` / `decoder_blocks_per_stage`；**`spatial_dims` / `in_channels` 由 `models.topology.build_topology` 一次性算出再写回 `cfg.model`**（R5：单一真相源，避免与 `factory.build_model` 重复推导）。
- `Config.validate()`：所有枚举字段全 `assert`；强制禁掉 `r2plus1d × 2.5D`、`lift_2_5d_to_3d × aux_keep_native_d` 等不合理组合。
- `load_config(path)`：YAML → 嵌套 dict → `_dataclass_from_dict` 递归构造 → `sync()` + `validate()`。
- `save_config(cfg, path)`：`asdict` → YAML 落盘到 `output_dir/resolved_config.yaml`。

### 3.3 `utils.py` —— 通用工具

```text
utils.py
├── AverageMeter             # 简单计数 + 均值
├── ModelEMA                 # 指数滑动平均；apply_shadow / restore 都做 in-place copy_，
│                            # 避免每次 val 都 deepcopy 大模型；带 _swapped 幂等保护
├── Timer                    # 经过时间格式化 HH:MM:SS
├── compute_dice_per_class   # sigmoid + 阈值 → 单 batch per-class mean Dice，
│                            # ignore_empty=True 时跳过 GT 空的类（nnU-Net 约定）
├── dice_batch_stats         # 返回 inter / denom / n_with_gt，便于跨 batch 池化 Dice
└── seed_everything          # random / numpy / torch / PYTHONHASHSEED；cudnn deterministic vs benchmark
```

### 3.4 `data/` —— 数据发现 / IO / patch / 增强

```text
segtask_v1/data/
├── __init__.py
├── dataset.py        # 单文件巨型模块：IO + 预处理 + bbox + npz + 3 个 Dataset
├── loader.py         # 路径匹配、stratified split、build_dataloaders 工厂
├── augment.py        # GPUAugmentor（仅一处共享 grid_sample 的入口）
└── make_data.py      # 离线把 NIfTI 烘焙成 bbox-cropped npz（多进程）
```

**`dataset.py`**（~2230 行）按职责分块：

- *NIfTI IO（带退避重试）*
  - `_sitk_read_with_retry(read_callable, path)`：bounded 指数退避重试 `SimpleITK ReadImage`；对内存型错误（`bad allocation`）立即转抛 `MemoryError` 不重试。环境变量 `SEGTASK_NIFTI_READ_RETRIES` / `SEGTASK_NIFTI_READ_BACKOFF_S` 可调。
  - `load_nifti(path, dtype)`：sitk 读 → numpy `(D, H, W)`。
  - `load_nifti_cropped(path, bbox, dtype)`：bbox-stream 读，不解码整卷（在 `bbox_dir` 模式省内存）。
- *npz IO（mmap）*：`_open_npz / load_npz_image / load_npz_label / load_npz_region_weight / npz_has_rw / load_npz_fg_slices / load_npz_fg_coords / load_npz_label_for_split`。
- *预处理*：`preprocess_image`（窗位 + minmax/zscore 归一化）、`preprocess_label`（整型标签 → per-fg-class 二值通道）、`load_region_weight_volume`（+1 shift）、`compute_region_weight_map`（按 `loss.region_weights` 静态生成）、`resize_3d`（image/label 区分插值阶数）。
- *bbox*：`compute_bbox_from_volume / apply_bbox / precompute_bboxes`。
- *Patch 抽取（z 边界安全）*：`extract_z_patch_padded`（沿 z 边缘 replicate）、`_extract_cubic_patch`（3D 立方 + zero-pad）。
- *Volume LRU 缓存*：`VolumeCache(max_volumes)` —— 线程安全 OrderedDict，按 `cache_mode=memory` + `cache_max_volumes` 配置。
- *3 个 Dataset 类*
  - `SegDataset3D`（`patch_mode in {"z_axis", "2_5d"}`）—— 沿 z 滑窗；2.5D 在此输出 `(C_res=1, D·n_views, H, W)` 由 trainer 后续 squeeze；支持 `aux_keep_native_d` 单 max-FOV cube 路径。
  - `SegDataset3DCubic`（`patch_mode="cubic"`）—— 3D 立方采样，前景过采样 + 三轴 multi-res。
  - `SegDataset3DWhole`（`patch_mode="whole"`）—— 整卷直接 resize 到 `patch_size`，最简但显存最大。

**`specs.py`** —— **R4 引入的 data 侧策略层**。把"按 `patch_mode` 选 dataset 类 + 准备 split-dependent kwargs"从 `loader.py` 抽出：

- `DatasetCommonCfg.from_cfg(cfg)` —— 把 11 个跨模式公共参数（`patch_size` / `intensity_min` / `cache_*` / `region_weights` / ...）冻结成一个不可变 dataclass；**避免 `loader.py` 重复构造 `common_kwargs` dict**。
- `SplitPaths` —— 单 split 的路径三元组（image / label / npz）。
- `DatasetSpec` ABC + 3 个子类：`WholeSpec` / `ZCubeSpec`（`z_axis` + `2_5d` 共用）/ `CubicSpec`。每个 spec 自己知道：要选哪个 dataset 类、需要哪些"模式专属 kwargs"（`multi_res_scales` / `z_boundary_mode` / `foreground_oversample_ratio`）、`is_train` 切换时如何调整 `aug_oversample_ratio` / `samples_per_volume` / `fg_ratio`。
- `build_data_spec(cfg)` —— 整个 data 子包**唯一允许 patch_mode if/elif 的地方**，与 `trainer/pipelines/factory.py` 同节奏。
- 新增一种 patch 模式：在 `dataset.py` 加 `SegDataset3DXxx` → 在 `specs.py` 加 `XxxSpec(DatasetSpec)` + `build_data_spec` 决策树加 1 行；**`loader.py` 完全不动**。

**`loader.py`** 全部都是顶层函数：

- `_load_exclude_pids / _filter_by_exclude`：读 `data.exclude_list` 把坏 pid 从配对列表里剔除。
- `discover_samples(image_dir, label_dir, suffixes)`：按文件名配对，warning + 跳过无 label 的样本。
- `_match_per_sample_paths`：通用 per-sample 路径配对器（被 bbox / region\_weight 复用）。
- `match_bbox_paths` / `match_region_weight_paths`：强契约——缺失即报错。
- `detect_label_values`：扫描 N 个 label 取并集自动推断 `label_values`。
- `train_val_split` / `_volume_primary_class` / `stratified_train_val_split`：按主导前景类分层切分。
- `discover_npz_samples`：从 `npz_dir`（可选 `_manifest.json`）发现 pid。
- `build_dataloaders(cfg)`：组装 Dataset → `DataLoader`（含 `persistent_workers / prefetch_factor / pin_memory`）。

**`augment.py`** —— `GPUAugmentor(AugConfig, max_scale)` 一个类。所有空间变换共用一份 `grid_sample`（per-sample 独立的仿射 + 弹性 displacement），保证 image / label / weight\_map 严格对齐；`max_scale` 用于把 `elastic_deform_alpha` 按最大 FOV 缩放，避免大 FOV 通道被等量 displacement 拉得过远。强度变换只作用在 image。

**`make_data.py`** —— 离线 npz 烘焙：

- `prepare_one(pid, image, label, bbox, rw, out_dir, ...)`：单样本 worker，做 bbox 裁剪 + 计算 `fg_slices` / `fg_coords`（seed=42 子采样到 N≤50000）+ 落盘 `<pid>.npz`。
- `_build_sample_table(cfg)` / `_resolve_label_values(cfg, samples)`：把 `Config` 翻译成可并行的样本表。
- `prepare_dataset(cfg, out_dir, workers, overwrite)`：`ProcessPoolExecutor` 并发执行 + 进度统计 + 失败 CSV。
- `main()` CLI：`python -m segtask_v1.data.make_data --config ... --out-dir ... --workers 8 [--overwrite]`。

### 3.5 `models/` —— UNet 家族 + Block 仓库

```text
segtask_v1/models/
├── __init__.py
├── blocks.py     # 所有通用积木（spatial_dims=2/3 共用）
├── stem.py       # 5 种 stem + 3 种多 FOV 上下文融合策略
├── resnet.py     # 4 种残差块 + ResNetStage
├── convnext.py   # ConvNeXt 块 + 论文 norm-first downsample
├── unet.py       # UNet3D 主网络（兼容 2D）
├── unetpp.py     # UNet++ 嵌套稠密 decoder（接口和 unet.Decoder 一致）
├── unet3p.py     # UNet3+ 全尺度跳连 decoder
└── factory.py    # 从 Config 装配 Encoder / Decoder / UNet3D
```

**`blocks.py`**（~950 行）所有积木都按 `spatial_dims∈{2,3}` 一份代码共用，类名保留 `*3D` 后缀只为 API 稳定：

- 工厂：`_CONV / _NORM / _DROP / get_activation / get_norm / make_attention`。
- 基础：`ConvNormAct`。
- 通道/空间注意力：`SqueezeExcite3D`（SE，Hu 2018）、`ECA3D`（Wang CVPR 2020）、`CBAM3D`（Woo ECCV 2018）、`CoordAttention3D`（Hou CVPR 2021）。
- Skip 注意力：`AttentionGate3D`（Oktay MIDL 2018）。
- 抗混叠 / 无参重排：`BlurPool3d`（Zhang ICML 2019）、`PixelUnshuffle3d` ↔ `PixelShuffle3d`。
- 高级上采样：`CARAFE3d`（Wang ICCV 2019，3D-only）、`DySample3d`（Liu ICCV 2023，3D-only）。
- 多模下/上采样：`Downsample`（conv / maxpool / avgpool / blurpool / pixelunshuffle）、`Upsample`（transpose / linear / nearest / pixelshuffle / carafe / dysample）。

**`stem.py`** —— 输入层 + 多 FOV 融合：

- 单视图 stem（`build_stem`）：`conv3` / `conv7` / `dual`（nnU-Net）/ `patch2` / `patch4`（Swin / ConvNeXt 标准）。
- 多视图融合（`build_context_stem(mode, fusion, n_views, ...)`）：
  - `shared_stem` —— 所有 `n_views·D` 通道一起进同一个 stem（最便宜）。
  - `multi_stem_proj` —— **Plan A**：`n_views` 个独立 stem → 通道拼接 → 1×1 fuse，推荐默认。
  - `hierarchical` —— **Plan C**：view 0 进主 stem；view k(k≥1) 用 stride 为 `main_stem_stride·2^k` 的 patchify stem，在 encoder 第 k 个 stage 入口处 cat-fuse；由 `HierarchicalStems` 类持有所有 aux stem。

**`resnet.py`** —— 4 个残差块共享 `ResNetStage(N blocks, attention, ...)`：

- `ResNetBlock`（`basic`）经典 post-act ResNet。
- `PreActResNetBlock`（`preact`）He et al. ECCV 2016，深层稳定。
- `BottleneckBlock`（`bottleneck`）1×1→3×3→1×1 × 4 扩展，nnU-Net ResEnc-XL。
- `R2Plus1DBlock`（`r2plus1d`，**3D-only**）—— (1,3,3) spatial conv + (3,1,1) temporal conv（Tran CVPR 2018 / Qiu ICCV 2017），Plan A 的 z 上下文注入；在 2.5D 模式被 validate 拒绝。
- `_BLOCK_REGISTRY` + `_make_block` 把字符串 → 类映射集中管理。

**`convnext.py`** —— ConvNeXt（Liu 2022）3D 移植：

- `DropPath` 随机深度。
- `LayerNorm3d` channel-first LN（与官方 channels-last 数学等价）。
- `ConvNeXtBlock`（7×7×7 DW + 4× 扩展 PW + LayerScale `1e-6`）+ `ConvNeXtAdaptBlock`（通道适配版）。
- `ConvNeXtStage`（N blocks）+ `ConvNeXtDownsample`（论文版 LN → stride-2 Conv，由 `model.convnext_downsample_lnfirst` 切换）。

**`unet.py`** —— UNet3D 主网络：

- `Encoder` —— stem + N 个 stage，stage 之间夹 `Downsample`；2.5D + 多 FOV + Plan C 时 stage 入口处插 `HierarchicalStems` 输出。
- `DecoderLevel` —— 单层 decoder：Upsample → 可选 AttentionGate → skip concat → stage blocks。
- `Decoder` —— N-1 层堆叠（接口稳定，UNet++/UNet3+ 实现同样接口可热替换）。
- `SegmentationHead` 1×1×1 / `ConvSegmentationHead` 3×3 + 1×1（可作 aux head 高容量变体）。
- `_build_aux_head(mode, ...)` —— `linear`（1×1×1）或 `conv`（3×3×3 + 1×1×1）。
- `UNet3D` —— 把上述拼起来；负责 deep-supervision 多尺度输出、aux head 调度、patchN stem 末尾的还原 upsample；公开 `count_params()` 拆分 encoder/decoder/head 参数。
- `_match_size` —— bilinear/trilinear resize helper。

**`unetpp.py`** —— `UNetPPDecoder` 嵌套稠密 grid，对外暴露对角线 `X[i, n-1-i]`，保证和 `unet.Decoder` 的多尺度输出 contract 一致（DS / seg\_head 不用改）。

**`unet3p.py`** —— `UNet3PDecoder` 全尺度跳连：每个 decoder 节点融合所有 encoder（max-pool 到本层） + 所有更深的 decoder（trilinear up 到本层），每条分支走 `cat_channels` 宽的 ConvNormAct 然后总 fuse。

**`factory.py`** —— 整网装配：

- `_resolve_blocks_per_stage` 调和 explicit 列表 vs fallback。
- `_StatefulStageBuilder` —— 按调用顺序消费每个 stage 的 block 数。
- `_make_resnet_stage_builder` / `_make_convnext_stage_builder` / `_make_convnext_downsample_builder` 构造对应工厂。
- `build_model(cfg) -> UNet3D` —— 单入口：**所有 mode 派生量（`out_classes` / `spatial_dims` / `context_n_views` / `in_ch_per_view_list` / `aux_head_out_channels` / `aux_seg_active` 门控）读 `ModelTopology`**（R5），本函数仅负责 backbone × decoder_type × stem 装配，不再做 patch_mode 分支。

**`topology.py`**（R5 引入，~150 行）—— **训练几何 / 通道布局派生量的单一真相源**：

- `@dataclass(frozen=True) ModelTopology` —— 一次性冻结 12 个派生字段：原始 mode flags（`patch_mode` / `lift_2_5d_to_3d` / `aux_keep_native_d` / `keep_native_multi_res`）+ 几何量（`n_views` / `num_res_groups` / `slab_depth` / `aux_view_depths`）+ I/O 通道（`in_channels` / `out_classes` / `spatial_dims` / `context_n_views` / `in_ch_per_view_list`）+ aux 拓扑（`aux_seg_active` / `aux_head_out_channels`）。
- `build_topology(cfg) -> ModelTopology` —— **整个 codebase 唯一推导入口**。重构前同一组派生量被 `Config.sync` 与 `models.factory.build_model` 各算一遍；R5 后 `Config.sync` / `factory.build_model` / `trainer.pipelines.factory.build_pipeline` / `Config.aux_view_depths` 全部委托至此。
- `aux_seg_active = aux_seg_supervision AND n_views > 1` —— 把原来散布于 `Config.validate` / `factory.py:255` / `unet.py:337` 的三处门控**合并到 topology 一处**；`UNet3D.__init__` 若收到 `aux_seg_supervision=True` 但 `n_views<=1` 现在会直接 `ValueError`。
- 新增 patch_mode：仅需修改 `build_topology` 内决策树即可同步影响 dataset / model / pipeline 三方。

### 3.6 `losses/` —— 二元 sigmoid 损失库 + 多分辨率/2.5D 包装器

```text
segtask_v1/losses/losses.py    # 全部损失都在这里，~1270 行
```

公共契约：`pred / target` 都是 `(B, num_fg, *spatial)` 的 per-class 独立 sigmoid（背景不预测）；可选 `weight_map: (B, 1, *spatial)` 广播到所有类。

| 损失类 / 工厂 | 一句话说明 |
|---|---|
| `BinaryDiceLoss` | per-class sigmoid Dice，可切 `batch_dice` / `ignore_empty` / `squared`。 |
| `BCELoss` | 带 `class_weights` 归一化的 BCE-with-logits。 |
| `BinaryFocalLoss` | 标准 `alpha_t · (1-p_t)^γ · log(p_t)`，pos/neg 都被 alpha 平衡。 |
| `BinaryTverskyLoss` | 非对称 Dice（FP/FN 各自权重）。 |
| `CompoundLoss` | 任意 loss 的加权和。 |
| `DeepSupervisionLoss` | 多尺度输出 → 自动把 target 下采样到每个尺度后求加权和（nnU-Net 风格）。 |
| `GeneralizedDiceLoss` | Sudre DLMIA 2017，自动逆体积类权（square / simple / uniform）。 |
| `BinaryFocalTverskyLoss` | Abraham ISBI 2019，`(1-TI)^γ` 放大难类。 |
| `LovaszHingeLoss` | Berman CVPR 2018，per-class IoU 替代物，支持 per-sample / batch 排序。 |
| `SoftCLDiceLoss` | Shit CVPR 2021，软骨架 + Dice 联合，保拓扑。 |
| `MultiResolutionLoss` | 多 FOV / 多 view label 包装器；按 view 拆 label，对每个 view 调用基础 loss。 |
| `SliceChannelLoss` | **2.5D 专用**：把 `(B, num_fg·D, H, W)` 的折叠输出按 `slice_loss_reduction=per_slice` (reshape 到 `(B·D, 1, H, W)`) 或 `per_volume` (reshape 到 `(B, num_fg, D, H, W)`) 跑基础 Dice/Tversky/BCE。 |
| `build_loss(cfg.loss)` | 工厂：name → 类；compound 自动按 `compound_weights` 组合。Trainer 在外层再包 `MultiResolutionLoss(DeepSupervisionLoss(SliceChannelLoss(base)))`。 |

私有辅助：`_check_inputs / _register_class_weights / _weighted_mean_over_classes / _weighted_voxel_mean / _interp_mode_smooth / _jaccard_from_sorted_errors / _soft_skel`。

### 3.7 `trainer/` —— 训练循环（Round 1–3 模块化）

原单文件 `trainer.py` (~1700 行) 已重构为 `trainer/` 包：基础设施（AMP / 优化器 / 内存核算 / breakdown / checkpoint）按职责拆分成独立模块；模式分支（whole / patch3d / 2.5d folded / 2.5d aux / 2.5d native_d / 2.5d lift / 2.5d lift+aux）由 **`ViewPipeline` 策略对象**统一封装，`Trainer` 不再判断模式。

```text
segtask_v1/trainer/
├── __init__.py             # re-export Trainer / build_optimizer / build_scheduler / WarmupScheduler（旧 import 100% 兼容）
├── trainer.py              # Trainer 主类：__init__ / fit / _train_epoch / _validate / checkpoint I/O
├── optim.py                # build_optimizer / build_scheduler / WarmupScheduler
├── amp.py                  # GradScaler shim / autocast / resolve_auto_amp_dtype / compute_loss_fp32
├── memory.py               # estimate_train_memory（参数 + 梯度 + 优化器 + EMA 静态预算）
├── breakdown.py            # collect_multi_res_breakdown / format_breakdown
├── checkpoint.py           # unwrap_compile / extract_model_state_dict / strip_common_prefixes
├── views.py                # 5 个无状态视图函数（center_crop / split_views_native_3d/d / squeeze_2_5d/_keep_views）
└── pipelines/              # ViewPipeline 策略对象（7 类，1 个工厂；唯一允许的 if/elif 集中地）
    ├── base.py             #   ViewPipeline ABC + SupervisionPack dataclass
    ├── factory.py          #   build_pipeline(cfg, base_loss) — 由 cfg flag 选子类
    ├── vanilla3d.py        #   Vanilla3DPipeline           (whole / 3D 单/eager 多分辨率)
    ├── patch3d.py          #   Patch3DNativeMultiResPipeline (3D keep_native_multi_res)
    ├── slab25d.py          #   Slab2_5DPipeline / Slab2_5DAuxPipeline / Slab2_5DNativeDPipeline
    └── lift25d.py          #   Lift2_5DPipeline / Lift2_5DAuxPipeline
```

`Trainer` 核心职责：

- **Pipeline 选择** —— `__init__` 中 `self.pipeline = build_pipeline(cfg, base_loss)` 一次性决定模式；后续 `_train_epoch` / `_validate` 只调用 `pipeline.prepare_batch` / `pipeline.compute_loss` / `pipeline.prepare_val_batch`，不再有任何模式 if 分支。
- **AMP** —— `use_amp + amp_dtype=auto|fp16|bf16`，fp16 走 `GradScaler`，bf16 跳过 scaler；损失计算强制升 fp32 防止 Dice 分母溢出（见 `trainer.amp.compute_loss_fp32`）。
- **梯度累积 + 裁剪** —— `grad_accum_steps` 末尾自动补齐 partial-tail（`Trainer._effective_accum`），`grad_clip_norm` 在 unscale 之后做。
- **EMA** —— `ModelEMA` 用 in-place swap；验证时 `ema.apply_shadow → validate → ema.restore`，异常安全（context manager `_ema_swapped`）。
- **`torch.compile`** —— `compile_mode` ∈ `none/default/reduce-overhead/max-autotune`，存盘前自动 unwrap（`trainer.checkpoint.unwrap_compile`）。
- **Checkpoint** —— full-state（model + EMA shadow + optimizer + scheduler + scaler + epoch + RNG + early-stop 计数器）；`resume` 完整恢复，`pretrain` 仅加载 model 权重（可选 `pretrain_load_ema=True` 用 EMA shadow 作为迁移起点；`pretrain_strict=False` 允许 head 形状不一致）。
- **指标** —— 训练用 `compute_dice_per_class`（单 batch），验证使用 `dice_batch_stats` 跨 batch 池化 Dice（nnU-Net 约定）；best 默认按 `mean_dice` 取 max。
- **Early stopping** —— `train.early_stopping=N` 启用，0 关闭。

> **新增模式怎么办？** 只需在 `pipelines/` 加一个 `XxxPipeline(ViewPipeline)`、在 `factory.py` 决策树里加一行 if，**`Trainer` 无需任何改动**。等价性请在 `test_pipelines.py::TestComputeLossEquivalence` 仿照现有用例补一条。

### 3.8 `predictor/` —— 滑动窗口推理（R6 模块化包）

原单文件 `predictor.py` (~1412 行) 已重构为 `predictor/` 包：mode 派生量改读 `ModelTopology`（与 trainer / build_model 共用同一真相源，R5 契约扩展），4 种 sliding 主循环 + 6 种 window/batch builders + 3 种 forward 变体 + 2 种 TTA 全部抽到模块级函数；`Predictor` 类保留 ~30 个 thin shim 方法以维持原私有 API（数百行单元测试通过 `Predictor.__new__(Predictor)` + 私有方法直调进行白盒测试，shim 让这些测试无需改一行）。

```text
segtask_v1/predictor/
├── __init__.py        # 完整 re-export（外部 API 100% 兼容）
├── predictor.py       # ~450 行：Predictor 类外壳 + __init__（topology 化）+ predict_volume 入口 + thin shims
├── sliding.py         # ~280 行：whole / z / z_interleave / cubic 四种主循环
├── inputs.py          # ~240 行：6 个 window/batch builders + 共享 padding helper
├── forwards.py        # ~270 行：3 种 forward（3D / 2.5D folded / 2.5D lift）+ 2 种 TTA + diag
├── blending.py        # ~100 行：compute_1d_positions / build_*_weight / prob_to_label（纯 numpy）
└── io.py              # ~145 行：run_inference + checkpoint helpers + precision resolution

Predictor                # 滑窗推理器（外壳；__init__ 读 ModelTopology 消除 ~80 行 R5 单一真相源违反）
├── predict_volume       # 顶层入口：load → bbox → preprocess → dispatch → blend → label_map
├── _build_z_window_*    # ──┐  3 个 z 轴 GPU + 1 个 CPU multi-res 窗口建造（thin shim → inputs.py）
├── _build_batch_*       # ──┴─ 2 个 cubic batch builders
├── _sliding_window_*    # ──── 4 种主循环（thin shim → sliding.py）
├── _forward_batch_*     # ──── 3 种 forward 变体（thin shim → forwards.py）
├── _tta_flip_ensemble*  # ──── 2 种 TTA（thin shim → forwards.py）
├── _compute_1d_positions / _build_1d_weight / _build_3d_weight / _prob_to_label
│                          # ──── 几何 / 概率 helpers（thin shim → blending.py）
└── _save_predictions    # NIfTI 写出（保 affine / origin / spacing / direction）

# Module-level helpers (in io.py)
_strip_compile_prefix    # 剥 torch.compile 的 _orig_mod. 前缀
_unwrap_ema_state        # ModelEMA shadow → 普通 state_dict
_select_state_dict       # 按 --weights {auto, ema, online} 选权重
_resolve_inference_precision   # auto → 跟随 cfg.train.amp_dtype
run_inference(cfg, ckpt_path, image_paths, weight_variant, bbox_paths, precision)
                         # 顶层入口：build_model → load ckpt → for image_path: Predictor.predict_volume → 写 NIfTI
```

关键契约：

- 假设 channel 排序：前 `num_fg` 个 1× 分辨率通道严格对齐 `cfg.data.label_values[1:]`，在构造和每 batch 都 assert。
- 几何一致 —— `z_boundary_mode`、`multi_res_scales`、`keep_native_multi_res` 等所有训练侧 toggle 都被 Predictor 镜像消费，避免训练-推理几何错位。
- 输出 —— 保留原 NIfTI 的 affine / origin / spacing / direction；`--save-probs` 额外保存 per-class sigmoid 概率图。

---

## 4. `configs/` —— YAML 配置 + 实验脚本

```text
configs/
├── default.yaml                # 全字段 + 注释，作教学版起点；任务为 TotalSegmentator body_pred 2.5D
├── seg2_5d.yaml                # 2.5D 多 FOV 生产配置（multi_res_scales + aux_seg_supervision）
├── seg3d.yaml                  # 3D 模式生产配置（z_axis / cubic / whole 任选）
├── test_e2e.yaml               # 用 small_data 跑通端到端的最小可用配置
└── experiments/
    ├── seg2_5d_small_base.yaml      # small_data 12 例 2.5D baseline
    ├── lift_a_baseline_2_5d.yaml    # Plan A 对照：纯 2.5D（不 lift）
    ├── lift_a_planA.yaml            # Plan A：lift_2_5d_to_3d + r2plus1d
    ├── lift_a_planA_aux.yaml        # Plan A + aux_seg_supervision
    ├── lift_a_ref_3d.yaml           # 纯 3D z_axis 参考
    └── run_aux_sweep.py             # 4 组 A/B/C/D 对比的自动化驱动脚本
```

`run_aux_sweep.py`：在 small\_data 上跑 4 组 30-epoch 训练（A0 baseline / A1 Plan A+aux / C1 Plan C+linear head / C2 Plan C+conv head），其他超参全部固定。每组结果输出到 `outputs/aux_exp/<id>/`，结束后打印 `best mean_dice` 与最后 epoch 的 `L_aux_k` 分解。

---

## 5. `tools/` —— 数据集体检工具

```text
tools/
├── scan_bad_nifti.py     # 扫一遍 image_dir / label_dir / bbox_dir，把所有
│                         # SimpleITK 读不开（非正交 direction cosines）的 pid
│                         # 落盘成 bad_pids.txt + bad_files.csv
└── bad_seg2_5d/          # scan_bad_nifti 的实际输出（按 dataset 命名一个子目录）
```

使用：见 §2.5。

---

## 6. 根目录测试脚本

所有测试用 `pytest` 或 `python <file>` 直接跑（每个文件末尾都有 `if __name__ == "__main__"`）：

```text
SegTask/
├── smoke_test_bbox.py                          # bbox 数据管线 smoke
├── smoke_test_bbox_predict.py                  # bbox-aware Predictor.predict_volume smoke
├── test_segtask_v1.py                          # 配置/数据/模型/损失/EMA/Dice 综合单元测试
├── test_load_nifti_memory.py                   # load_nifti 内存路径验证（避免 float64 提升）
├── test_attention.py                           # ECA/CBAM/Coord/AttentionGate 形状+梯度+集成
├── test_blocks_sampling.py                     # BlurPool/PixelShuffle/CARAFE/DySample 模式
├── test_blocks_2d_smoke.py                     # blocks spatial_dims=2 兼容性 smoke
├── test_stem_and_unet3p.py                     # 5 种 stem × decoder × deep_supervision 组合
├── test_unetpp.py                              # UNet++ decoder + 端到端 + 配置校验
├── test_resenc.py                              # preact/bottleneck 残差块 + ResEnc 预设
├── test_new_losses.py                          # GDL / FocalTversky / Lovász / clDice
├── test_round2_fixes.py                        # Round 2 BUG-1/2/3 回归
├── test_slice_loss_reduction.py                # SliceChannelLoss per_slice vs per_volume
├── test_z_boundary_mode.py                     # z_boundary_mode stretch vs edge_pad
├── test_2_5d_smoke.py                          # 2.5D 端到端 smoke（dataset+model+loss）
├── test_aux_seg_supervision.py                 # 2.5D 多 FOV aux 监督完整通路
├── test_aux_keep_native_d.py                   # aux_keep_native_d 单 max-FOV cube 路径
├── test_keep_native_multi_res.py               # 3D 单 max-FOV cube（dataset 层 R1）
├── test_keep_native_multi_res_trainer.py       # 3D 单 cube 在 trainer 侧的 per-view 切分 (R2)
├── test_keep_native_multi_res_predictor.py     # 3D 单 cube 推理侧 (R3)
└── test_lift_aux_ds.py                         # Plan A lift_2_5d_to_3d + aux + DS 集成
```

> 测试环境：`conda activate torch27_env`，建议直接调用环境的 `python.exe`（`conda run` 启动慢）。

---

## 7. `segtask/` —— 早期 v0 原型（已冻结）

`segtask/` 是项目最早的实验性原型，**已不再维护**。结构如下，新工作请一律基于 `segtask_v1/`：

```text
segtask/
├── train.py / predict.py
├── trainer.py / predictor.py / config.py
├── utils.py / visualization.py
├── data/        # 老版本的数据集 + DataLoader
├── models/      # 老版本 UNet（不支持 spatial_dims=2 / multi-FOV / aux 监督）
└── losses/      # 老版本损失函数
```

仅在需要复现旧实验时使用；新增功能、bug fix、配置都只走 `segtask_v1/`。

---

## 8. 关键交互时序

### 8.1 一次训练 step（2.5D 多 FOV）

```text
DataLoader worker (CPU)
    SegDataset3D.__getitem__
      ├── load_nifti_cropped(image, bbox)        # int16 → fp32，单分辨率最大 FOV cube
      ├── load_nifti_cropped(label, bbox)        # int16 raw labels
      ├── (optional) load_region_weight_volume
      └── extract_z_patch_padded(D · max_scale)  # 单 max-FOV cube，仍是单分辨率
    → (1, C_res=1, eD_max, H, W) image
      (1, C_fg,    eD_max, H, W) label_binary    (preprocess_label 拆类)
      (1, 1,       eD_max, H, W) weight_map?

GPU
    GPUAugmentor (共享 grid_sample) → image / label / weight_map 同步 warp
    Trainer._split_views_native_2_5d                       ← 单 max-FOV cube → 多 view 张量
        for k in views: center_crop(D_k=round(D·s_k)) + resize(H, W)
        concat → (B, sum_k D_k, H, W)  (aux_keep_native_d) OR (B, D·n_views, H, W)

Model
    UNet3D(spatial_dims=2)
        Stem(context_fusion = shared_stem | multi_stem_proj | hierarchical)
            └── multi_stem_proj: n_views 独立 stem → cat → 1×1 fuse
            └── hierarchical:    view 0 进主 stem；view k 进 stage-k 入口 cat
        Encoder / Decoder
        Main seg_head:  (B, num_fg·D, H, W)
        Aux  seg_head_k: (B, num_fg·D, H, W)   k = 1..n_views-1

Loss
    L_main = MultiResolutionLoss( DeepSupervisionLoss( SliceChannelLoss(base) ) )(main, label_view0)
    L_aux  = Σ_k aux_w_k · MultiResolutionLoss(SliceChannelLoss(base), num_res=1)(aux_k, label_view_k)
    L_total = L_main + L_aux                              # 上述都强制 fp32 算 reduction

Backward
    grad-clip → optimizer.step → scheduler.step → ema.update
```

### 8.2 一次推理（z\_axis 滑窗 + TTA）

```text
predict.py
    load_config → run_inference(cfg, ckpt_path, [image_paths], weight_variant)
        build_model(cfg) → Predictor(model, cfg, device)
        _select_state_dict(ckpt, weight_variant) → model.load_state_dict
        for each image_path:
            Predictor.predict_volume(image_path, bbox_path=?)
                load_nifti + preprocess_image
                (optional) apply_bbox
                z 滑窗:
                    for each window center z:
                        _build_z_window_input(...)     # 含 multi_res_scales 几何
                        (optional) flip-TTA            # forward 2× 取均值
                        forward → sigmoid → _gaussian_blend_kernel 加权累加
                threshold → argmax-ish (per-class binary)
                (optional) save_probs → .nii.gz
                (optional) un-crop bbox 回原坐标
                写出  <output_dir>/<pid>_pred.nii.gz   保留源 affine
```

---

## 9. 扩展指南

- **新增一种损失**：在 `@d:\codes\work-projects\SegTask\segtask_v1\losses\losses.py` 里加 `class XxxLoss(nn.Module)`，在 `build_loss` 工厂分支里注册名字，在 `Config.validate()` 的 `loss.name` 白名单里加上字符串，最后写一份回归测试到 `test_new_losses.py`。
- **新增一种 backbone**：在 `models/` 下加 `xxx.py`（参考 `resnet.py` / `convnext.py`），实现一个 `XxxStage(in_ch, out_ch, n_blocks, ...)`，在 `models/factory.py` 添加 `_make_xxx_stage_builder` + 把 `cfg.model.backbone == "xxx"` 接进 `build_model`，并在 `ModelConfig.validate` 加白名单。
- **新增一种 patch 模式**（R5–R6 后大幅简化，仅 5 处改动）：
  1. `data/dataset.py` 加 `SegDataset3DXxx`（如果几何抽取语义有别于现有 3 种）
  2. `data/specs.py` 加 `XxxSpec(DatasetSpec)` 并在 `build_data_spec` 决策树追加 1 行（`build_dataloaders` 不动）
  3. **`models/topology.py::build_topology` 内的决策树**追加 1 个分支以填写 `in_channels` / `out_classes` / `spatial_dims` 等派生量 —— **`Config.sync` / `factory.build_model` / `pipelines/factory.build_pipeline` / `Predictor.__init__` 全部自动同步，无需修改**
  4. `trainer/pipelines/` 加对应 `ViewPipeline(...)` 子类 + `factory.py` 决策树追加 1 行
  5. `predictor/inputs.py` 加对应 window builder + `predictor/sliding.py` 主循环 builder 分派加 1 行（如果几何不同于现有 6 种）；`predictor/forwards.py` 通常无需改动（forward 路径仅按 `patch_mode == '2_5d'` 与 `lift_2_5d_to_3d` 二分）
  
  注意保持「数据集只产单分辨率最大 FOV cube；多分辨率在 trainer/predictor 入模型前做」的契约。
- **新增一个 stem 或多 FOV 融合策略**：在 `models/stem.py` 加类 + 在 `build_stem` / `build_context_stem` 注册；`Config.validate` 的 `stem_mode` / `context_fusion` 白名单。
- **改命名/路径约定**：所有规则集中在 `data/loader._match_per_sample_paths` 和 `data/make_data._stem`，改一处即可。

> 任何带 `validate()` 报错信息的改动，请同步更新 `configs/default.yaml` 注释与本 README 的对应表格。
