# ssltask — 与分割主线共骨干的自监督预训练

ssltask 是一套与 `segtask_v1` 共骨干的自监督预训练工程：它负责在无标注或少标注数据上做表示学习，并把学到的 encoder 权重迁移回分割、分类与检测任务。方法库覆盖重建式、对比式、自蒸馏式与掩码建模式多种 SSL 路线，评测侧则提供在线 probe 与离线 few-shot 评测。

> 端到端预训练/评测流程见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

## 模块树

```text
ssltask/
├── README.md  # 本文件
├── __init__.py  # 包入口与注册表
├── __main__.py  # python -m ssltask 入口
├── config.py  # 复用分割配置并叠加 SSLConfig
├── pretrain.py  # 自监督预训练 CLI
├── evaluate.py  # 离线 few-shot 评测 CLI
├── data/  # 数据子系统
│   ├── __init__.py  # 数据包入口
│   ├── corruptions.py  # Genesis 式破坏变换
│   ├── masking.py  # 掩码建模共享工具
│   ├── multicrop.py  # 多裁剪视图生成器
│   ├── ssl_dataset.py  # image-only / labeled npz 数据集
│   └── vesselness.py  # Frangi vesselness 目标
├── docs/  # 流程文档
│   └── WORKFLOW.md  # 端到端预训练/评测流程
├── eval/  # 评测子系统
│   ├── __init__.py  # 评测包入口
│   ├── cls_probe.py  # 在线分类探针
│   ├── metrics.py  # AUC / F1 / HD95 指标
│   ├── pipeline.py  # 离线评测 pipeline
│   ├── probe.py  # 在线分割探针
│   └── split.py  # 组级（患者级）train/val 划分
├── methods/  # SSL 方法注册与实现
│   ├── __init__.py  # 方法注册表 / build_method
│   ├── base.py  # SSLMethod 抽象接口
│   ├── byol.py  # BYOL
│   ├── dino.py  # DINO
│   ├── dino_gram.py  # DINO + Gram anchoring（staged recipe）
│   ├── genesis.py  # Models Genesis 式重建
│   ├── ibot.py  # iBOT / DINOv2 混合
│   ├── jepa.py  # JEPA 隐空间预测（CNN 适配，非原版 I-JEPA）
│   ├── moco.py  # MoCo
│   ├── prior.py  # Frangi vesselness 回归
│   ├── simmim.py  # SimMIM
│   ├── spark.py  # SparK
│   ├── sparkdino.py  # SparK + DINO 联合目标
│   └── vicregl.py  # VICRegL 局部一致性
├── models/  # SSL 专有模型模块
│   ├── __init__.py  # 模型包入口
│   ├── dino_modules.py  # DINOHead / DINONet
│   ├── ibot_modules.py  # iBOT 密集头工具
│   ├── jepa_modules.py  # JEPA encoder / predictor builder
│   ├── spark_modules.py  # SparK 稀疏编码与轻量解码器
│   ├── ssl_models.py  # 重建 / MIM 模型与 head builder
│   └── vicregl_modules.py  # VICRegL 投影 / 局部头工具
└── trainer/  # SSL 训练循环
    ├── __init__.py  # 训练包入口
    └── ssl_trainer.py  # 方法无关的 SSL 训练循环
```

## 关键概念

- **共骨干迁移**：ssltask 的主要价值是产出可迁移 encoder；分割、分类、检测都可以直接复用这份权重。
- **方法注册表**：不同 SSL 方法通过统一的注册入口装配，训练循环尽量保持方法无关。
- **数据视图**：image-only npz、多裁剪、masking、corruption、vesselness 都是围绕视图构造展开的。
- **评测探针**：在线 probe 与离线 few-shot 评测并存，既看表示可分性，也看下游迁移效果。
- **方法家族**：重建、对比、自蒸馏、掩码建模和先验回归几类方法都在同一仓库内对齐实现。
- **配置入口**：所有方法都从 `configs/ssltask_*.yaml` 进入，和主线分割配置保持一致的数据与几何约束。

## 用法

```bash
# 预训练
python -m ssltask.pretrain --config configs/ssltask_genesis.yaml
python -m ssltask.pretrain --config configs/ssltask_dino.yaml

# 评测
python -m ssltask.evaluate --config configs/ssltask_prior.yaml

# 示例配置（configs/）
# ssltask_byol / ssltask_dino / ssltask_dino_gram / ssltask_genesis /
# ssltask_ibot / ssltask_jepa / ssltask_moco / ssltask_prior /
# ssltask_simmim / ssltask_spark / ssltask_sparkdino / ssltask_vicregl
```

## 训练 / 评测行为约定

以下为需要注意的正确性与语义约定（配置项默认值见 `config.py`、示例见对应 YAML）：

- **优化步时钟**：梯度累积下，scheduler / warmup / global_step / 方法内部调度均按**真实 optimizer.step 边界**推进，而非 micro-batch；尾批按实际累积长度归一。
- **AMP 跳步一致性**：loss 非有限或 GradScaler 因 inf/NaN 内部跳步时，`on_after_step(global_step, stepped=False)`——调度时钟照常推进，但 EMA / center / MoCo queue / DINO-Gram 快照等状态一律冻结；同一累积组内的方法状态延到优化步边界统一施加，跳步则整组丢弃。
- **teacher 恒为 eval**：所有冻结 EMA 分支（DINO/iBOT/DINO-Gram teacher、BYOL target、MoCo key、JEPA target）通过 `train()` 覆写始终保持 eval 模式。
- **DINO last-layer freeze**：前 `dino_freeze_last_layer_frac` 比例的优化步内取消学生投影头末层梯度（官方稳定化技巧）。
- **DINO-Gram staged recipe**：λ 生效前 Gram teacher 不刷新；首次生效时从当前 EMA teacher 锚定快照，此后每 `dino_gram_refresh_steps` 步刷新（默认 1000）。
- **VICRegL**：方差用总体方差（unbiased=False，N=1 返回 0）；稠密匹配为 overlap-aware 双向位置 + 可选特征空间匹配，不重叠视图不强造正样本。
- **multicrop**：`scale` 按 **体积占比**（RandomResizedCrop 约定）各向同性采样，跨样本可比；裁剪+重采样+翻转折叠进单次 `grid_sample` 批处理。iBOT / DINO-Gram 的密集分支复用 DINO 主损失的同一批 global 裁剪（不重复采样）。
- **HD95**：spacing-aware 双向表面距离；空掩码在 batch / 报告层显式计数（`probe_hd95_empty_frac`），并新增 `probe_fg_recall` 监控漏检。
- **组级划分**：probe / 离线 eval 的 train/val 按患者（文件名 stem，`*_group_regex` 第 1 个捕获组）划分，同组绝不跨集；单组默认抛错，`*_allow_single_group=True` 才退回 train==val。验证 patch 确定性 + 前景感知。
- **Frangi 先验**：`prior_spacing` 非空时 `prior_scales` 按物理单位解释（各向异性感知）；空则保持体素单位旧行为。
- **checkpoint**：原子写（临时文件 + `os.replace`）+ 状态指纹；续训校验指纹不匹配即报错。DDP 下仅 rank 0 落盘，epoch loss 按样本数加权 all-reduce。
