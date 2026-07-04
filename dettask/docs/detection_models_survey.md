# 检测模型综述（dettask 四模板选型依据）

仿 `gentask/docs/generative_models_survey.md`：梳理四大检测范式的原理、
医学 3D 场景适配与本工程的实现取舍。

## 1. 一阶段 anchor —— RetinaNet / Retina U-Net（`det.arch: retinanet`）

* **原理**（Lin et al., ICCV 2017）：FPN 各层密铺 anchor，分类分支用
  Focal Loss 解决一阶段极端正负失衡，回归分支预测 anchor→gt 的
  Δcenter/log-size。
* **医学地位**：nnDetection（Baumgartner et al., MICCAI 2021）以
  Retina U-Net（Jaeger et al., 2020）为唯一基线横扫 10+ 医学检测榜单——
  U-Net decoder 天然提供高分辨率金字塔，对小病灶召回至关重要。
  这是本工程「共享 Encoder+Decoder + FPNAdapter」设计的直接依据，也是
  排期上 2.5D RetinaNet 首个落地的原因。
* **实现取舍**：anchor z-scale 独立配置（结节 z 向常更扁）；分配器
  max-IoU（`pos_iou=0.3~0.4`，医学 3D IoU 天然偏低）或 ATSS
  （Zhang et al., CVPR 2020：候选 IoU 均值+方差自适应阈值，免调参）；
  回归默认 GIoU（比 smooth-L1 对小框更稳）。

## 2. 两阶段 —— Faster R-CNN（`det.arch: faster_rcnn`）

* **原理**（Ren et al., NeurIPS 2015）：RPN 产生类无关 proposal，
  ROIAlign（He et al., ICCV 2017 提出替代 ROIPool）取区域特征后二次
  分类+回归。两阶段精排在假阳性控制上仍是强基线。
* **医学适配**：3D ROIAlign 无官方算子——本工程用 `grid_sample` 自实现
  （`dettask/ops.roi_align`），2D/3D 同一路径、可反传，免 CUDA 扩展；
  ROI 取金字塔最高分辨率层（小病灶特征在浅层）。
* **实现取舍**：类无关框回归 + K+1 softmax（医学类数少，够用且省参）；
  proposal 训练时混入 gt 框保证 ROI 头正样本供给。

## 3. anchor-free —— FCOS（`det.arch: fcos`）

* **原理**（Tian et al., ICCV 2019）：逐特征点直接回归到框四（六）边的
  距离，centerness 分支压低边缘点的低质量预测，层间按回归距离范围分工。
* **医学价值**：免 anchor 设计（医学目标尺寸分布偏斜时 anchor 超参敏感），
  3D 化后自然扩展为 6 距离 + 3D centerness（逐轴 min/max 比几何平均）。
* **实现取舍**：中心采样简化为「点在框内 + 层范围过滤 + 多框取最小体积」；
  推理分数 = cls × centerness。

## 4. Transformer 集合预测 —— Deformable DETR（`det.arch: detr`）

* **原理**（Carion et al., ECCV 2020 DETR；Zhu et al., ICLR 2021
  Deformable）：可学习 query 经匈牙利匹配一对一绑定 gt，端到端免
  NMS；可变形注意力仅在参考点附近采稀疏点，收敛快、可及高分辨率。
* **医学定位**：探索性（体数据 token 数大、小数据集匹配不稳），但集合
  预测对「每卷少量目标」的医学场景在概念上契合；2.5D 先行。
* **实现取舍**（Plan §7-7 依赖克制）：可变形注意力以 `grid_sample`
  纯 PyTorch 实现（query 预测 offset+权重，单尺度特征采样），2D/3D 同一
  实现；可学习参考点 + 逐层框细化（inverse-sigmoid 空间累加）；集合损失
  = focal-BCE + L1 + GIoU（Deformable-DETR 权重口径 2/5/2）。

## 5. 横向对比与选型建议

| 维度 | RetinaNet | Faster R-CNN | FCOS | Deformable DETR |
|------|-----------|--------------|------|-----------------|
| 范式 | 一阶段 anchor | 两阶段 | anchor-free | 集合预测 |
| 速度 | 快 | 慢（ROI 二次前向） | 快 | 中 |
| 小目标召回 | 高（高分辨率金字塔） | 高 | 中-高 | 中（探索性） |
| 假阳控制 | 中 | 高（二阶段精排） | 中 | 高（一对一匹配） |
| 超参敏感 | anchor 设计 | anchor + 采样比 | 层范围 | 匹配权重/收敛 |
| 医学证据 | ★★★（nnDetection SOTA） | ★★ | ★★ | ★（新兴） |

**默认路线**：RetinaNet（2.5D 先行）作生产基线 → FCOS 做 anchor 消融 →
Faster R-CNN 压假阳 → DETR 追端到端上限。

## 6. 与本仓库设计的对应关系

* 金字塔来源：`segtask_v1` Encoder+Decoder（Retina U-Net 思路），
  `FPNAdapter` 仅做通道对齐——SSL 预训练 encoder/decoder 全量可迁移；
* 双几何：全部头由 `spatial_dims` 参数化；2.5D 逐 slab 2D 框 + 推理跨层
  拼接（`stitch_link_iou`/`stitch_min_span` 是小病灶召回的关键权衡，
  Plan §7-5），FROC 统一在拼接后的 3D 框上评估；
* 依赖克制：NMS/ROIAlign/可变形注意力全部纯 PyTorch 自实现，未引入
  torchvision / mmdet / CUDA 扩展。
