# clstask — 3D / 2.5D 医学影像分类

复用 segtask_v1 基建（配置 / 几何拓扑 / 预处理 / Encoder / 优化器 / AMP / EMA）的
分类工程。SSL（ssltask）预训练 encoder 权重可直接迁移。

## 双几何

与分割同一套 `patch_mode` 语义，由 `segtask_v1` topology 单一真相源派生
`spatial_dims` / `in_channels`：

| 几何 | patch_mode | spatial_dims | 输入布局 |
|------|-----------|--------------|----------|
| 3D   | `whole` / `z_axis` / `cubic` | 3 | `(B, 1, D, H, W)` |
| 2.5D | `2_5d`（slab 深度折进通道） | 2 | `(B, D, H, W)` |

2.5D 目前限单 FOV（`multi_res_scales: [1.0]`），与 image-only SSL 预训练几何
一致；不支持 `lift_2_5d_to_3d`。

## 四个 backbone 模板

| 模板 | 配置 | SSL 迁移 |
|------|------|---------|
| ResNet   | `cls.backbone: encoder` + `model.backbone: resnet`   | ✅（同一构建路径） |
| ConvNeXt | `cls.backbone: encoder` + `model.backbone: convnext` | ✅ |
| DenseNet-BC | `cls.backbone: densenet`（`clstask/models/densenet.py`，2D/3D） | ✗ |
| ViT      | `cls.backbone: vit`（`clstask/models/vit.py`，2D/3D，位置编码可插值） | ✗ |

## 标签

* **粒度** `cls.label_granularity`：
  * `volume` —— 对整个 patch/样本分类，logits `(B, K)`；
  * `slice` —— 对每个 z 切片分类，logits `(B, K, D)`（2.5D 头把特征图按 D
    切片池化；3D 头沿 z 保留分辨率）。
* **来源** `cls.label_source`：
  * `mask` —— 由分割 mask 派生弱标签（每前景类"是否出现"），volume/slice 均支持；
  * `table` —— 显式标签表 csv（`pid,label` 或 `pid,c1..cK`）/ json，仅 volume；
    `cls.multi_label=false` 时为单标签 softmax CE。

## 损失 / 增强 / 选模

* 损失：`bce`（多标签）/ `focal` / `ce`（单标签），支持 label smoothing、
  class weights，fp32 计算 + logit clamp（承接 segtask AMP 口径）。
* Mixup / CutMix（仅 volume 粒度）：软标签，CutMix λ 按实际裁剪体积重算。
* 选模：`cls.save_best_metric ∈ {auc, f1, acc, loss}`；AUC 为无第三方依赖的
  秩统计实现（Mann-Whitney，带并列校正）。

## SSL / 分割权重迁移

```yaml
cls:
  backbone: encoder
  pretrained_ckpt: outputs/ssl_xxx/best.pt   # 只取 encoder.*，strict=False
  freeze_encoder: false                       # true = linear probe
  encoder_lr_mult: 0.1                        # encoder 参数组学习率倍率
```

命中/缺失张量数打日志；0 命中直接报错（几何或 backbone 不匹配时不静默）。

## 推理（patch → volume 的 MIL 聚合）

`ClsPredictor` 按几何做网格采样（2.5D 沿 z；3D cubic 三轴网格），patch 概率经
`cls.agg_mode ∈ {mean, max, lse, topk}` 聚合为卷级概率；slice 粒度另输出逐
z-slice 概率（重叠 patch 取 max）。

## 使用

```bash
# 训练（3D cubic / 2.5D 折叠两套参考配置）
python -m clstask.train --config configs/cls3d_cubic.yaml
python -m clstask.train --config configs/cls2_5d.yaml \
    --override train.epochs=100 cls.agg_mode=topk

# 推理
python -m clstask.predict --config configs/cls3d_cubic.yaml \
    --ckpt outputs/cls3d_cubic/best_model.pth \
    --npz-dir /path/to/npz --out-dir predictions/cls [--use-ema]
```

## 冒烟测试

```bash
python tests/test_clstask_smoke.py
```

覆盖：双几何配置派生、数据集形状、损失/指标、四模板×双几何前向、
table 标签 + mixup/cutmix、双几何 3-epoch 训练（loss 下降 + val AUC）、
ckpt `encoder.*` 键 + strict=False 迁移命中、整卷推理。
