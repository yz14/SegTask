# SegTask — 通用医学图像分割框架

基于 UNet 架构的模块化分割项目，支持任意 encoder/decoder backbone 组合、2D/2.5D/3D 数据输入，所有参数通过 YAML 配置驱动。

## 项目结构

```
SegTask/
├── configs/
│   └── default.yaml              # 默认配置（含详细注释）
├── segtask/
│   ├── __init__.py
│   ├── config.py                 # 配置系统（dataclass + YAML）
│   ├── utils.py                  # 工具函数（指标、EMA、日志）
│   ├── trainer.py                # 训练流水线（AMP、EMA、调度器）
│   ├── predictor.py              # 推理流水线（滑动窗口、TTA）
│   ├── data/
│   │   ├── matching.py           # 数据匹配（image ↔ label 配对）
│   │   ├── dataset.py            # Dataset 类（2D、2.5D、3D）
│   │   ├── transforms.py         # GPU 数据增强
│   │   ├── sampler.py            # 类平衡采样器
│   │   └── loader.py             # DataLoader 工厂
│   ├── models/
│   │   ├── blocks.py             # 通用构建块（Conv、Norm、Up/Down）
│   │   ├── unet.py               # 通用 UNet 架构
│   │   ├── factory.py            # 模型工厂（从配置构建模型）
│   │   ├── encoders/
│   │   │   ├── vgg.py            # VGG 风格编码器
│   │   │   ├── resnet.py         # ResNet 残差编码器
│   │   │   └── vit.py            # ViT Transformer 编码器
│   │   └── decoders/
│   │       ├── vgg.py            # VGG 风格解码器
│   │       ├── resnet.py         # ResNet 残差解码器
│   │       └── vit.py            # ViT Transformer 解码器
│   └── losses/
│       └── losses.py             # 损失函数（Dice、CE、Focal、Tversky、边界加权）
├── train.py                      # 训练入口
├── predict.py                    # 预测入口
└── test_segtask.py               # 测试脚本（31 项测试）
```

## 环境

```bash
conda activate py310
```

依赖：`torch`, `nibabel`, `numpy`, `pyyaml`, `pandas`, `scipy`（可选，用于边界加权和后处理）

## 快速开始

### 训练

```bash
# 使用默认配置训练
python train.py --config configs/default.yaml

# 覆盖配置参数
python train.py --config configs/default.yaml --train.epochs 100 --train.lr 0.001 --data.batch_size 16

# 指定 GPU
python train.py --config configs/default.yaml --device cuda
```

### 预测

```bash
# 单文件预测
python predict.py --config configs/default.yaml --checkpoint outputs/best_model.pth --input path/to/image.nii.gz

# 批量预测
python predict.py --config configs/default.yaml --checkpoint outputs/best_model.pth --input-dir path/to/images/

# 启用 TTA
python predict.py --config configs/default.yaml --checkpoint outputs/best_model.pth --input path/to/image.nii.gz --tta
```

### 测试

```bash
python test_segtask.py
```

## 核心特性

### 数据模式

| 模式 | 输入形状 | 说明 |
|------|---------|------|
| `2d` | `(B, 1, H, W)` | 单切片，适用于 X-ray 或自然图像 |
| `2.5d` | `(B, C, H, W)` | C = 2×num_slices_per_side+1 张连续切片，训练时对所有切片计算损失，预测时只保留中间切片 |
| `3d` | `(B, 1, D, H, W)` | 3D 体积块，随机裁剪训练 |

### Encoder/Decoder 组合

通过 YAML 配置任意组合：

```yaml
model:
  encoder_name: "resnet"   # "vgg", "resnet", "vit"
  decoder_name: "resnet"   # "vgg", "resnet", "vit"
  encoder_channels: [32, 64, 128, 256, 512]
  encoder_blocks_per_level: [2, 2, 2, 2, 2]
```

- **VGG**: 经典多层卷积堆叠
- **ResNet**: 残差连接，深层网络更稳定
- **ViT**: Transformer 自注意力，捕获全局上下文

### 损失函数

```yaml
loss:
  name: "dice_ce"          # dice, ce, focal, tversky, dice_ce, dice_focal
  class_weights: [0.1, 1.0, 0.5]   # 每类权重
  spatial_weight_mode: "border"      # 边界加权（U-Net 论文方法）
```

### 训练技巧

- **混合精度（AMP）**: 自动开启，显著加速
- **EMA**: 指数移动平均，平滑模型权重
- **学习率调度**: cosine, poly, step, plateau, one_cycle
- **Warmup**: 线性预热
- **梯度裁剪**: 按范数或值
- **前景过采样**: 训练时偏向包含前景的切片/块
- **类平衡采样**: 基于逆频率的加权采样
- **深度监督**: 多尺度输出，改善梯度流
- **边界加权**: U-Net 论文的距离变换边界权重

### 数据增强（GPU）

所有增强在 GPU 上执行，零额外 CPU 开销：

- 随机翻转、旋转、缩放
- 亮度、对比度、Gamma 调整
- 高斯噪声、高斯模糊
- Mixup

## 配置说明

所有参数集中在 YAML 文件中，按模块组织：

- `data`: 数据路径、模式、预处理、采样
- `augment`: 数据增强开关和参数
- `model`: 模型架构（encoder/decoder 选择、通道数、层数）
- `loss`: 损失函数类型和参数
- `train`: 训练超参（优化器、调度器、AMP、EMA）
- `predict`: 推理设置（滑动窗口、TTA、后处理）

详见 `configs/default.yaml` 中的注释。
