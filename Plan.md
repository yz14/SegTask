# Plan：分类（clstask）与检测（dettask）项目设计方案（完整覆盖 3D 与 2.5D 双几何）

> 本文为阶段一（调研 / 分析 / 规划）产出的设计方案，开放问题已经确认（见文末 §8），作为后续执行阶段的依据。
>
> 复述目标：在已有 `segtask_v1`（分割）/ `gentask`（生成·超分）/ `ssltask`（自监督预训练）之上，
> 参照分割项目的代码风格新增 **分类** 与 **检测** 两个工程；各用 4 个「最经典 / 公认 SOTA」算法作模板样例；
> 自监督统一为分割 / 生成 / 分类 / 检测的骨干预训练来源；能复用分割代码处直接拷贝后修改，避免重复造轮子。


---

## 0. 结论先行（TL;DR）

1. **完全复刻分割的模块化骨架**：`config/ | data/ | models/ | losses/ | trainer/ | predictor/ | docs/`，两个新工程与 `gentask` 同构。
2. **几何双轨是分割体系的一等公民，必须原样继承**：分割通过 `patch_mode ∈ {whole, z_axis, cubic}`（3D）与 `2_5d`（2.5D 折叠 / lift）+ `ModelTopology` 单一真相源 + `ViewPipeline` 策略族，让 **同一套 Encoder/Decoder 代码同时服务 2D 与 3D**（所有 block 都按 `spatial_dims` 参数化）。分类 / 检测沿用同一机制，各配一族任务 pipeline，即可同时得到 3D 与 2.5D 两套方案。
3. **复用枢纽**：共享 `Encoder`（多尺度特征金字塔）/ `Decoder`（FPN 式金字塔）+ 统一 `build_model` + `strict=False` 预训练交接。任务头命名 `cls_head` / `det_head`，即可零成本继承 `ssltask` 全部预训练权重（`cls_probe.py` 已验证分类迁移路径）。
4. **分类 4 模板**（每个都同时支持 3D 与 2.5D，见 §2.4）：`ResNet`（最经典，≈现有 Encoder+GAP）、`DenseNet`（医学最常用、CheXNet 血统）、`ConvNeXt`（现代 CNN SOTA，`convnext.py` 已有）、`ViT`（Transformer SOTA 代表）。
5. **检测 4 模板**（每个都同时支持 3D 与 2.5D，见 §3.5）：`RetinaNet / Retina U-Net`（医学检测公认 SOTA 基线，nnDetection 同款）、`Faster R-CNN`（两阶段经典）、`FCOS`（anchor-free 现代范式）、`Deformable DETR`（Transformer 集合预测 SOTA）。检测头挂在共享 Encoder+Decoder 金字塔上，编/解码器 SSL 权重均可迁移。
6. **2.5D 检测有独特工程红利**：2.5D 分支的框是 2D 的，可直接对接成熟的 2D 检测算子生态（ROIAlign / NMS 均有现成高效实现），推理端做「逐 slab 2D 框 → 跨层拼接成 3D 框」；而 3D 分支需自实现 ROIAlign-3D / 3D NMS。先 2.5D 立基线、后 3D 冲上限，是风险最低的路线。

---

## 1. 现有架构复用分析（设计依据）

### 1.1 三个工程的同构骨架

`segtask_v1` 最成熟；`gentask` = 分割骨架 + 「退化 + 生成范式」；`ssltask` = 复用 `segtask_v1.config.Config` 与 `build_model`，叠加自监督目标。三者共享「dataclass+YAML 配置、几何派生、GPU 增强、AMP/EMA/warmup 训练循环」。新工程照此复刻。

### 1.2 3D / 2.5D 双几何机制（本次修订的核心分析）

分割用三层抽象把「几何」从模型 / 训练代码里剥离，这是 2D/3D 一套代码通吃的关键：

**(a) `patch_mode` 与拓扑派生（`models/topology.py::build_topology`，唯一推导入口）**

| patch_mode | spatial_dims | 模型输入 | 主头输出 | 说明 |
| --- | --- | --- | --- | --- |
| `whole` / `z_axis` / `cubic` | 3 | `in_ch = n_views`（多分辨率视图按通道 cat） | `num_fg × n_views`（`num_res_groups=n_views`，逐视图组监督） | 纯 3D 卷积 |
| `2_5d`（折叠，默认） | **2** | `in_ch = D × n_views`（slab 深度折叠进通道） | `num_fg × D`（逐 slice 通道监督） | 2D 卷积处理 3D 上下文 |
| `2_5d` + `keep_native_view_depth` | 2 | `in_ch = Σ D_k`（各视图保原生深度） | `num_fg × D` + 逐视图 aux 头 `num_fg × D_k` | 多 FOV 异深 |
| `2_5d` + `lift_2_5d_to_3d` | 3 | `in_ch = n_views` | `num_fg` | 2.5D 数据升维回 3D 卷积 |

另有 3D 侧的 `keep_native_multi_res`（多 FOV 懒加载，trainer 逐视图裁剪+resize）。

**(b) stem 多视图融合（`models/stem.py`）**：仅 2.5D 在 stem 处融合多 FOV 视图（`num_stem_fusion_views = n_views`），三种模式 `shared_stem / multi_stem_proj / hierarchical`；3D 的多分辨率走输出端分组监督、不经 stem 融合。

**(c) `ViewPipeline` 策略族（`trainer/pipelines/`）**：`factory.build_pipeline` 按 topology 选择 `Vanilla3DPipeline / Patch3DNativeMultiResPipeline / Slab2_5DPipeline / Slab2_5DAuxPipeline / Slab2_5DNativeDPipeline / Lift2_5DPipeline / Lift2_5DAuxPipeline`——「batch → (model_input, supervision) → loss 折叠」全部归口 pipeline，Trainer 与模型对几何无感知。

**推论**：分类 / 检测只要（i）沿用 `build_topology` 并扩展各自派生量、（ii）各写一族任务 pipeline、（iii）模型侧继续用 `spatial_dims` 参数化的共享 block，就能像分割一样 **一套代码同时提供 3D 与 2.5D 方案**，且新增 patch_mode 仍然「只改 topology 一处」。

### 1.3 复用枢纽：`build_model` → Encoder / Decoder / 独立命名 Head

`Encoder.forward` 返回多尺度特征列表 `[level_0, …, bottleneck]`，`Decoder` 输出 FPN 式金字塔；分割挂 `seg_head`，SSL 挂 `recon_head`（`ssltask/models/ssl_models.py` 复用同一 `build_model`，逐参数同名同形 → 下游 `train.pretrain` `strict=False` 干净交接）。分类 / 检测同法：挂 `cls_head` / `det_head`，`encoder.*`（乃至 `decoder.*`）全部命中、任务头随机。

### 1.4 训练与数据基建

`Trainer` 外壳（AMP/EMA/warmup/grad-accum/梯度检查点/compile/DDP）与 `DataConfig` 全套能力（2.5D/3D patch 采样、多分辨率 FOV、oversample→中心裁剪、前景过采样、npz 预打包、缓存、`bbox_dir`、GPU 增强）直接继承；仅替换标签语义与评估器。

---

## 2. 分类项目 clstask 设计（3D 与 2.5D 双几何）

### 2.1 目录结构（复刻 gentask）

```text
clstask/
├── __init__.py / __main__.py / train.py / predict.py / utils.py / logging_utils.py
├── docs/            # classification_models_survey.md（仿 gentask/docs 综述）
├── config/          # dataclasses.py | io.py | validation.py
├── data/            # loader.py | make_data.py | specs.py | dataset/  （拷后改标签语义）
├── losses/          # cls.py（CE / BCE / Focal / LabelSmoothing / mixup-cutmix）
├── models/          # blocks/resnet/convnext/mednext/stem/unet/topology 直接拷；新增 densenet.py + vit.py + classifier.py + factory.py
├── trainer/         # cls_trainer.py + pipelines/{cls3d.py, clsslab25d.py, clslift25d.py} + amp/optim/checkpoint/memory/ema/dist（直接拷）
└── predictor/       # cls_predictor.py（整卷 / patch 投票 / slab 聚合 / TTA）
```

### 2.2 配置系统

复刻 `gentask/config/dataclasses.py`（`spatial_dims`/`in_channels` 仍由 topology 派生为只读 property）。新增 `ClsTaskConfig`：

- `num_classes`、`multi_label`、`label_key`（沿用 `cls_probe.py` 的 `cls_label_key` 约定）；
- `label_granularity`：`volume`（体级标签）| `slice`（2.5D 逐 slice 标签，见 2.3）；
- `input_mode`：`whole` | `patch`（patch 分类 + 推理聚合）；**几何仍由 `data.patch_mode` 决定**（whole/z_axis/cubic=3D，2_5d=2.5D），与分割完全一致，不另设开关；
- `arch`：`resnet | densenet | convnext | vit`；`pool`：`gap | gem | attention`（attention=MIL 注意力池化）；
- `loss`：`ce | bce | focal`、`label_smoothing`、`mixup_alpha`/`cutmix_alpha`、`class_weights`。

### 2.3 双几何下的分类拓扑（`build_topology` 扩展）

| 几何 | 模型输入 | cls_head 输出 | 监督 |
| --- | --- | --- | --- |
| 3D（whole/z_axis/cubic） | `(B, n_views, D, H, W)` | `num_classes`（GAP over 3D） | 体级 / patch 级标签 |
| 2.5D 折叠 | `(B, D×n_views, H, W)`（复用 Slab 折叠） | `label_granularity=volume`：`num_classes`；`slice`：`num_classes × D`（对应分割 `num_fg × D` 的逐 slice 通道思想） | slab 级；或逐 slice 标签 → 体级由 MIL/max/attention 聚合 |
| 2.5D + `keep_native_view_depth` | `(B, ΣD_k, H, W)` | 同上 | stem 多视图融合原样复用 |
| 2.5D + `lift_2_5d_to_3d` | `(B, n_views, D, H, W)`（3D 卷积） | `num_classes` | 同 3D |

对应 pipeline 族：`Cls3DPipeline / ClsSlab2_5DPipeline / ClsLift2_5DPipeline`（仿分割 `factory.build_pipeline` 的决策树；分类无逐像素监督，pipeline 主要负责输入折叠 / 标签聚合 / mixup 软标签折算）。

### 2.4 模型：统一 `Backbone → Neck(Pool) → Head`，4 模板均双几何

新增 `models/classifier.py::ClassifierModel`（对标 `gentask/models/generation.py` 的封装角色）：

```
feats  = backbone(x)          # 复用 Encoder 契约（2D 或 3D 由 topology 决定）
pooled = neck(feats[-1])      # gap / gem / attention（MIL）
logits = cls_head(pooled)     # 命名 cls_head → SSL strict=False 干净交接
# 可选：多 level 辅助分类头（借鉴分割 deep_supervision）
```

由于共享 block 全部按 `spatial_dims` 参数化，**4 个模板天然同时提供 2D（2.5D 用）与 3D 两个版本**，factory 按 `cls.arch × topo.spatial_dims` 构建：

1. **ResNet（最经典 / 强基线）**——即现有 `Encoder`（resnet backbone，2D/3D 现成）+ GAP + 线性头；就是 `cls_probe.py` 已跑通的路径，零新增骨干代码。SSL 迁移第一等公民（2D：ImageNet/CheXNet 血统；3D：MedicalNet/Med3D 血统）。
2. **DenseNet（医学最常用）**——MONAI 3D 分类默认、CheXNet（2D 胸片）血统。新增 `densenet.py`（仿 `resnet.py` 的 stage/block 写法，`DenseBlock`+`Transition`，`spatial_dims` 参数化一次写两用）。
3. **ConvNeXt（现代 CNN SOTA）**——`convnext.py` 已有（`ConvNeXtStage`/`ConvNeXtDownsample`，已 dims 参数化），factory 走通分类头即可，几乎零新增。
4. **ViT（Transformer SOTA 代表）**——新增 `vit.py`：patch-embed（2D 版给 2.5D 折叠输入 / 3D 版给体数据）+ Transformer encoder + `[CLS]`/平均池化；注意力复用 `blocks.py` 已 SDPA 化的实现。说明：ViT 不复用 CNN Encoder，不吃当前 CNN 自监督；作为 Transformer 范式模板，配 DINO/iBOT/JEPA（需 ssltask 增 transformer 骨干分支）或后续 MAE。取舍显式标注。

### 2.5 损失 / 训练 / 推理

- `losses/cls.py`：CE / BCE（`cls_probe.py` 已用）/ Focal / LabelSmoothing + mixup/cutmix 软标签；2.5D 逐 slice 监督时提供 `SliceChannelClsLoss`（对应分割 `SliceChannelLoss` 思想）。
- `trainer/cls_trainer.py`：拷 `gen_trainer.py`，评估器换 AUC/F1/Acc/Balanced-Acc（直接复用 `ssltask/eval/metrics.py::macro_cls_metrics`）；`best` 以 val AUC 选。
- `predictor/cls_predictor.py`：3D=整卷或滑窗 patch 概率平均；2.5D=沿 z 滑 slab → slab 级概率 → 体级聚合（mean/max/attention，与训练端 MIL 一致）；可选 TTA（flip/rot90）。
- 数据：显式 `cls_label` 键优先；否则由分割掩码派生「每类是否出现」多标签目标（`LabeledPatchDataset` + `_target_from_batch` 已验证）——弱监督冷启动，2.5D 下天然得到逐 slice 标签（掩码逐 slice 判定），正好喂 `label_granularity=slice`。

### 2.6 与 ssltask 集成

`train.pretrain` + `strict=False` → `encoder.*` 命中、`cls_head.*` 随机。**注意几何一致性**：SSL ckpt 与下游必须同 `patch_mode` 几何（2.5D 折叠的 `in_ch=D×n_views` 与 3D 的 `in_ch=n_views` 参数形状不同）——这与现在分割↔SSL 的约束一致，ssltask 本就复用同一 Config、天然支持在 2.5D 或 3D 几何下预训练，无需改动。

---

## 3. 检测项目 dettask 设计（3D 与 2.5D 双几何）

### 3.1 目录结构

```text
dettask/
├── (顶层同构) + docs/detection_models_survey.md
├── config/          # DetTaskConfig（anchor / fpn / nms / assigner / 2.5D 拼接参数）
├── data/            # targets.py（bbox 编解码/匹配/几何联动，2D 与 3D 双实现）；复用 bbox_dir
├── losses/          # det.py（Focal + GIoU/DIoU + centerness + 匈牙利匹配）
├── models/          # 拷共享 backbone/Encoder/Decoder；新增 fpn.py + heads/{retina,frcnn,fcos,detr}.py + detector.py + factory.py + topology.py
├── trainer/         # det_trainer.py + pipelines/{det3d.py, detslab25d.py}
└── predictor/       # det_predictor.py（3D：滑窗+3D NMS；2.5D：逐 slab 2D 框 → 跨层拼接 3D 框）
```

### 3.2 主干：共享 Encoder + Decoder = 天然 FPN（双几何通用）

分割 `Decoder` 输出 `[dec_low_res,…,dec_high_res]` 即特征金字塔——检测头挂其上（nnDetection「Retina U-Net」思路）。2.5D 时它是 2D 金字塔、3D 时是 3D 金字塔，**同一份代码**。编码器吃 SSL `encoder.*`；重建式 SSL（Genesis/SparK/Prior）还可迁移 `decoder.*`。新增 `fpn.py` 仅做通道对齐 / 额外 P6-P7（dims 参数化）。

### 3.3 双几何下的检测形态

| 几何 | 框 | 头 / 算子 | 优劣 |
| --- | --- | --- | --- |
| 3D（z_axis/cubic） | 3D 框 `[z1,y1,x1,z2,y2,x2]`，3D anchor | 3D 卷积头；需自实现 ROIAlign-3D、3D NMS | 上限高（完整 3D 上下文）；算子工程量大、显存高 |
| 2.5D 折叠 | slab 内 2D 框 `[y1,x1,y2,x2]`（带 slab z 索引） | 2D 卷积头；**直接用 torchvision 成熟 2D ROIAlign / NMS**；slab 折叠通道提供跨层上下文 | 工程量小、速度快、可吃 2D 生态；跨 slab 需推理端拼接 |
| 2.5D + lift | 3D 框 | 同 3D | 2.5D 数据管线 + 3D 检测头的折中 |

2.5D 检测是医学（肺结节等）经典且强力的路线：逐 slab 出 2D 框 → 相邻层框按 IoU/中心距离链接 → 合成 3D 框（predictor 内 `stitching`，参数：`link_iou`、`min_span`）。**建议以 2.5D RetinaNet 为 D 系列第一个落地基线**（复用最多、风险最低），3D 随后。

对应 pipeline 族：`Det3DPipeline / DetSlab2_5DPipeline`（负责：输入折叠、oversample 裁剪后同步裁框、增强几何作用于框、target 分配调用）。

### 3.4 配置与数据

- `DetTaskConfig`：`num_classes`、`fpn_levels`、`anchor_sizes/ratios/scales`（2D 三元 / 3D 增 z 向）、`assigner`（`iou`/`atss`/`hungarian`）、`nms_iou`/`score_thresh`/`max_dets`、`num_queries`（DETR）、`reg_loss`、`focal_alpha/gamma`、`stitch_link_iou`/`stitch_min_span`（2.5D 拼接）。`build_topology` 扩展派生 anchor 数、各层步长、框维度（4 或 6）。
- 数据：扩展现有 `bbox_dir` 为框标注（3D：`[z1,y1,x1,z2,y2,x2,cls]`；2.5D 可由 3D 框逐层切片自动派生 2D 框——**只需存 3D 真值一份**）。`targets.py` 以 `spatial_dims` 参数化实现 anchor 生成、IoU/ATSS 分配、框编解码（Δcxcywh(d) / distance-to-boundary）、crop/flip/rot90 框联动（配单测）。

### 3.5 模型：4 模板（`detector.py` 统一封装，均双几何）

`DetectorModel` 封装 `backbone(Encoder) → decoder/FPN → det_head`，factory 按 `det.arch × topo.spatial_dims` 分派：

1. **RetinaNet / Retina U-Net（医学检测公认 SOTA 基线）**——一阶段 anchor + Focal Loss，nnDetection 同款；头是纯卷积（dims 参数化即得 2D/3D 双版），最能体现「复用分割解码器 + SSL 迁移」红利。必选、首个落地。
2. **Faster R-CNN（两阶段经典）**——RPN + ROIAlign + R-CNN 头。2.5D 版直接用 torchvision ROIAlign；3D 版需自实现 ROIAlign-3D（主要新增件）。
3. **FCOS（anchor-free 现代范式）**——逐点预测 + centerness，无 anchor 超参，头最轻，2D/3D 同构。
4. **Deformable DETR（Transformer 集合预测 SOTA）**——queries + 可变形注意力 + 匈牙利匹配，免 NMS。2.5D 版可用官方 2D 可变形注意力思路（纯 PyTorch grid_sample 实现避免 CUDA 扩展）；3D 化是探索性「冲 SOTA」模板。

### 3.6 损失 / 训练 / 推理

- `losses/det.py`：Focal + GIoU/DIoU/L1（2D/3D 各一版 IoU 计算）+ centerness + 匈牙利集合损失；多损失聚合走 Trainer 已有 breakdown 机制。
- `trainer/det_trainer.py`：评估器 = mAP（COCO 式）+ **FROC**（医学标准；2.5D 在拼接后的 3D 框上算，保证与 3D 分支同一读数口径）。
- `predictor/det_predictor.py`：3D=滑窗 → 全图坐标 → 跨窗合并 + 3D NMS；2.5D=沿 z 滑 slab → 2D NMS → 跨层拼接 3D 框 →（可选）3D 框级二次抑制。复用分割 predictor 滑窗坐标框架。

### 3.7 与 ssltask 集成

同 §2.6：几何一致的 SSL ckpt 经 `strict=False` 迁移 `encoder.*`（+重建式方法的 `decoder.*`）；`fpn.*`/`det_head.*` 随机。可选增强：ssltask 评测侧加「检测探针」（冻结 encoder + 轻量 Retina 头 few-shot 读数），与 seg/cls 探针并列。

---

## 4. 4+4 模型选型总览（均含 2.5D 与 3D 双版本）

| 工程 | 模板 | 范式 | 2.5D | 3D | 复用共享 Encoder | 吃 CNN-SSL | 主要新增件 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 分类 | ResNet | 经典 CNN 基线 | ✅ | ✅ | ✅（现成） | ✅ | ≈0（仅头） |
| 分类 | DenseNet | 医学最常用 | ✅ | ✅ | ✅ | ✅ | `densenet.py`（dims 参数化） |
| 分类 | ConvNeXt | 现代 CNN SOTA | ✅ | ✅ | ✅（convnext.py 已有） | ✅ | ≈0 |
| 分类 | ViT | Transformer SOTA | ✅ | ✅ | ❌（自带 patch-embed） | ⚠️ 需 DINO/MAE 系 | `vit.py` |
| 检测 | RetinaNet / Retina U-Net | 一阶段 anchor（医学 SOTA 基线） | ✅（首个落地） | ✅ | ✅ Enc+Dec | ✅（含解码器） | retina 头 + fpn |
| 检测 | Faster R-CNN | 两阶段经典 | ✅（torchvision ROIAlign） | ✅（需 ROIAlign-3D） | ✅ Enc+Dec | ✅ | RPN + ROIAlign(-3D) |
| 检测 | FCOS | anchor-free 现代 | ✅ | ✅ | ✅ Enc+Dec | ✅ | fcos 头（最轻） |
| 检测 | Deformable DETR | Transformer 集合预测 SOTA | ✅ | ✅（探索性） | ✅ Enc | ✅ | 可变形注意力 + 匈牙利匹配 |

---

## 5. 复用清单（直接拷 / 改造 / 新增）

- **直接拷贝几乎不改**：`trainer/{amp,optim,checkpoint,memory,dist_utils,ema}`、`models/{blocks,resnet,convnext,mednext,stem,unet}`（含全部 2D/3D dims 参数化 block 与 2.5D stem 融合）、`config/io.py`、`data/{loader,make_data,cache}` 骨架、`ssltask/eval/metrics.py`。
- **改造**：`config/dataclasses.py`（换 TaskConfig）、`models/topology.py`（扩展任务派生量，保留 2.5D/3D 决策树）、`trainer/pipelines/`（换成 Cls/Det pipeline 族，保留 3D/Slab/Lift 三形态）、`data/dataset`（换标签语义）、`trainer/*_trainer.py`（换评估器）、`predictor`、`models/factory.py`。
- **新增**：分类 `classifier.py`+`densenet.py`+`vit.py`+`losses/cls.py`；检测 `detector.py`+`fpn.py`+`heads/*`+`targets.py`+`losses/det.py`+predictor 拼接/NMS；3D 专属算子 ROIAlign-3D、3D NMS。

---

## 6. 分阶段实施计划（执行阶段，确认后逐步进行）

> 每步独立可验收；**每步的验收都同时覆盖 2.5D 与 3D 两个冒烟配置**（与分割的双几何测试口径一致）。

**分类 clstask**
- C1：搭骨架（目录/config/topology/pipeline 族/trainer/predictor 外壳），ResNet+GAP 最小训练打通。验收：2.5D 折叠与 3D cubic 两配置各跑 1 epoch，loss 下降、val AUC 有输出、`train.pretrain` 从对应几何 ssl ckpt 命中 `encoder.*`。
- C2：数据层（显式标签 + 掩码派生弱标签 + 逐 slice 标签 + mixup/cutmix + MIL 聚合）。验收：volume/slice 两种 granularity 各一冒烟。
- C3：DenseNet / ConvNeXt。验收：双几何 × 三骨干各出 AUC。
- C4：ViT + `classification_models_survey.md`。验收：4 模板齐、文档成稿。

**检测 dettask**
- D1：骨架 + bbox 数据层 + `targets.py`（2D/3D 双实现 + 单测：裁剪/翻转框联动、anchor 生成、编解码往返）。
- D2：**2.5D RetinaNet** 打通（Enc+Dec 金字塔 + Focal+GIoU + 2D NMS + 跨层拼接 3D 框 + FROC）。验收：小数据过拟合、FROC 有读数、SSL 迁移命中。
- D3：**3D RetinaNet**（3D anchor/NMS）+ FCOS（双几何）。验收：mAP/FROC 双几何读数。
- D4：Faster R-CNN（2.5D 先行；3D 含 ROIAlign-3D）。
- D5：Deformable DETR（2.5D 先行；3D 探索性）+ `detection_models_survey.md`。

> 按 §8 确认：本轮仅聚焦 clstask / dettask 本体，ssltask 的检测探针 / transformer 骨干分支不在范围内（后续另议）。

---

## 7. 难点与风险

1. **几何一致性约束**：SSL ckpt 与下游必须同 patch_mode 几何（in_channels/spatial_dims 形状耦合）——沿用现状约束即可，但需在 validation.py 里显式校验并给出清晰报错。
2. **ViT / DETR 不吃当前 CNN 自监督**：显式标注；建议配 ssltask 的 DINO/iBOT/JEPA 增 transformer 骨干分支，或后续 MAE。不阻塞 CNN 主线。
3. **3D 检测算子成本**：ROIAlign-3D、3D NMS、可变形注意力 3D 化是最大工程量——所以 D 系列按「2.5D 先行、3D 递进」排期，2.5D 直接吃 torchvision 成熟算子。
4. **框几何与增强联动**：oversample/裁剪/翻转/旋转必须同步作用于框（分割无此问题），`targets.py` 配套单测；2.5D 还需保证 slab 切片派生 2D 框与 3D 真值一致。
5. **2.5D 拼接的召回损失**：跨层链接参数（link_iou/min_span）影响小病灶召回，FROC 评估统一在拼接后的 3D 框上进行，保证与 3D 分支可比。
6. **弱标注冷启动**：分类先用分割掩码派生多标签目标（已验证），2.5D 下天然得到逐 slice 标签。
7. **依赖克制**：优先纯 PyTorch 自实现（grid_sample 版可变形注意力、自写 3D NMS）；确需第三方先说明必要性。

---

## 8. 已确认的决策（用户确认，2026-07）

1. **分类标签粒度：两种都要支持，由参数控制**。`cls.label_granularity = volume | slice`：`volume` 对该样本（体/patch）分类；`slice` 对切片上可能的目标逐 slice 分类（2.5D 下 cls_head 输出 `num_classes × D`，体级结果由 MIL/max/attention 聚合）。
2. **检测排期确认**：2.5D RetinaNet 首个落地、3D 递进；框标注只存 3D 真值，2.5D 自动切片派生 2D 框。
3. **4+4 模板清单确认**：分类 ResNet / DenseNet / ConvNeXt / ViT；检测 RetinaNet / Faster R-CNN / FCOS / Deformable DETR。
4. **范围确认**：仅聚焦分类与检测两个工程本体，不同步扩展 ssltask。
