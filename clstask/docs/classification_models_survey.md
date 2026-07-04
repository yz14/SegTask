# 分类模型综述（clstask 四模板选型依据）

仿 `dettask/docs/detection_models_survey.md` / `gentask/docs/generative_models_survey.md`：
梳理四大分类骨干的原理、医学 3D 场景适配与本工程的实现取舍。

## 1. ResNet（`cls.backbone: encoder` + `model.backbone: resnet`）

* **原理**（He et al., CVPR 2016）：残差捷径使深网络可优化；分类范式为
  「stage 递降分辨率 → 全局池化 → 线性头」。
* **医学地位**：3D ResNet 系（Med3D、MedicalNet 等）长期是体数据分类的
  默认基线；与分割 encoder 同构意味着**同一权重可在分割/SSL/分类间无缝
  流转**——这是本工程把 ResNet 模板直接落在 `segtask_v1` Encoder 上
  （而非另写一份）的核心理由。
* **实现取舍**：复用 `build_seg_model(cfg).encoder`，几何（2.5D 折叠 /
  3D）与通道数由 topology 单一真相源派生；SSL `encoder.*` strict=False
  全量命中；`cls.encoder_lr_mult` / `freeze_encoder` 支持微调与
  linear-probe 两种迁移协议。

## 2. ConvNeXt（`cls.backbone: encoder` + `model.backbone: convnext`）

* **原理**（Liu et al., CVPR 2022）：以 ViT 的宏观设计（大核深度卷积、
  LN、GELU、倒瓶颈、layer scale）现代化 ResNet，纯卷积达到 ViT 精度。
* **医学价值**：MedNeXt（Roy et al., MICCAI 2023）验证了该家族在医学
  3D 上的有效性；卷积归纳偏置在小数据集上通常优于 ViT。
* **实现取舍**：与 ResNet 同一构建路径（`model.backbone: convnext`，
  含 GRN / layer-scale / stochastic-depth 等 segtask 已有开关），
  分类侧零新增代码——模板差异完全收敛在 `cfg.model`，SSL 迁移同样成立。

## 3. DenseNet-BC（`cls.backbone: densenet`）

* **原理**（Huang et al., CVPR 2017）：层间密集连接 + 瓶颈/压缩
  （BC），特征复用使参数量小、抗过拟合——医学分类文献中最常见的
  CNN 之一（胸片、结节良恶性等）。
* **实现取舍**：`clstask/models/densenet.py` 以 `spatial_dims` 参数化
  2D/3D 同一实现；growth-rate / block-layers / compression 可配；
  norm/act 跟随 `cfg.model`（GroupNorm 友好小 batch）。与 segtask
  Encoder 拓扑不同，故 SSL 迁移不适用（README 已标注）。

## 4. ViT（`cls.backbone: vit`）

* **原理**（Dosovitskiy et al., ICLR 2021）：图像切 patch 作 token，
  纯 Transformer 编码 + CLS/池化分类；容量上限高，但小数据依赖
  预训练或强增强。
* **医学定位**：探索性（体数据 token 数大、医学数据集小）；配合
  MAE/DINO 类 SSL 时上限最高，作为四模板的「上限探针」。
* **实现取舍**：`clstask/models/vit.py` 以 `vit_patch_size`（3 元，2.5D
  自动取 yx 二元）参数化 2D/3D；位置编码按网格插值以适配不同
  patch_size；`slice` 粒度下保留 z 分辨率的 token 池化。

## 5. 横向对比与选型建议

| 维度 | ResNet(encoder) | ConvNeXt(encoder) | DenseNet-BC | ViT |
|------|-----------------|-------------------|-------------|-----|
| 范式 | 残差 CNN | 现代化 CNN | 密集连接 CNN | Transformer |
| 参数效率 | 中 | 中 | 高 | 低 |
| 小数据稳健性 | 高 | 高 | 高 | 低（需 SSL） |
| SSL/分割权重迁移 | ✅ 全量 | ✅ 全量 | ✗ | ✗（结构不同） |
| 医学证据 | ★★★ | ★★（MedNeXt 系） | ★★★ | ★（新兴） |

**默认路线**：ResNet-encoder（可吃 SSL 权重）作生产基线 → ConvNeXt 做
现代化消融 → DenseNet 验证轻量上限 → ViT 配 SSL 追上限。

## 6. 与本仓库设计的对应关系

* 几何双轨：四模板全部由 `spatial_dims` 参数化，2.5D（slab 折叠进
  通道，`(B, D, H, W)`）与 3D（`(B, 1, D, H, W)`）同一代码路径，由
  `segtask_v1.models.topology.build_topology` 单一真相源派生；
* 标签双粒度：`volume`（(B, K)）与 `slice`（(B, K, D)）两种头
  （`clstask/models/classifier.py`），mask 弱标签 / csv-json 标签表
  两种来源；
* 推理 MIL：patch 网格 → `agg_mode ∈ {mean, max, lse, topk}` 聚合为
  卷级概率，slice 粒度逐 z 回填（重叠取 max）；
* 依赖克制：AUC/F1/acc 纯 torch 秩统计实现，未引入 sklearn / timm /
  monai。
