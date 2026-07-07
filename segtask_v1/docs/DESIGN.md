# segtask_v1 设计文档

本文档整理 `segtask_v1` 的架构约定、数据流、训练与推理时序、扩展方式和关键契约。README 负责导航，本文件负责把实现背后的设计原则和跨文件关系讲清楚。

## 1. 总体架构

`segtask_v1` 的核心思想是把「几何」与「实现」分开：

- **几何** 由配置决定，包括 `patch_mode`、`multi_res_scales`、`keep_native_view_depth`、`lift_2_5d_to_3d`、`spacing_normalization` 等。
- **实现** 由 `data/`、`models/`、`losses/`、`trainer/`、`predictor/` 这些子模块承接。
- **真相源** 集中在 `config.py` 与 `models/topology.py`，避免同一组派生字段在多个地方重复推导。

### 1.1 训练时数据流

```text
train.py
  └── load_config / override / sync / validate
        └── data.loader.build_dataloaders(cfg)
              ├── discover_samples / exclude / match bbox / match rw
              ├── split train / val
              └── build_data_spec(cfg)
                    └── 选择对应 Dataset 与 split 参数
                          └── __getitem__
                                ├── NIfTI / npz 读取
                                ├── 预处理 image / label / weight map
                                ├── 抽 patch 或整卷 resize
                                └── 返回单分辨率最大 FOV cube
                                      └── GPUAugmentor
                                            └── 共享空间增强
                                                  └── Trainer / Pipeline
                                                        ├── view 切分
                                                        ├── 送入模型
                                                        └── 计算损失与指标
```

训练阶段的关键点是：**数据集只负责产出单分辨率的最大 FOV cube**，多视图和多分辨率的裁剪、下采样、拼接推迟到 trainer 侧完成，这样可以保证几何一致，同时减少重复 IO 与重复插值带来的误差。

### 1.2 推理时数据流

```text
predict.py
  └── load_config / override / sync / validate
        └── build_model(cfg)
              └── Predictor(model, cfg)
                    ├── 读取 checkpoint
                    ├── 读取图像 / bbox
                    ├── 做与训练一致的预处理
                    ├── 按 patch_mode 进入滑窗逻辑
                    ├── 做必要的 TTA / blend
                    └── 反归一化与写出 NIfTI
```

推理阶段必须镜像训练阶段的几何约定，尤其是 `patch_mode`、`multi_res_scales`、`keep_native_multi_res` 和 `z_boundary_mode`。只要这些字段一致，推理结果才可以与训练时的 patch 语义对齐。

## 2. Patch 模式与 FOV 约定

### 2.1 `patch_mode`

- `z_axis`：沿 z 方向滑窗，H/W 保持全分辨率。
- `cubic`：三维立方滑窗，xyz 都可移动。
- `whole`：整卷 resize 到目标 patch。
- `2_5d`：沿 z 组织 slab，把深度折进通道。

### 2.2 单分辨率读取原则

数据集只读取一份最大 FOV 的原始 cube，之后的多分辨率只是在模型前做中心裁剪与 resize。这样有三个好处：

1. 避免多次 zoom 造成的高频损失。
2. 保证同一体素在不同 view 里几何一致。
3. 让缓存、npz 烘焙和推理逻辑保持统一。

### 2.3 `ModelTopology`

`models/topology.py` 负责推导训练几何与通道布局相关的派生量：

- 输入通道数
- 输出类别数
- `spatial_dims`
- 视图数与每视图深度
- aux 头是否激活

这一层是单一真相源。`config.py`、`factory.py`、`trainer/pipelines/factory.py` 和 `Predictor` 都应该从这里读取结果，而不是各自重复计算。

## 3. 文件命名与数据契约

### 3.1 pid 约定

`pid` 由图像文件名去掉图像后缀得到。所有配套文件都使用 pid 作为索引键：

- image
- label
- bbox
- region weight
- npz
- exclude list

任何需要强配对的路径，一旦缺失就应报错。这样可以尽早发现样本丢失、命名错误或目录不一致。

### 3.2 npz 烘焙

`make_data.py` 把 NIfTI 预处理成 npz，主要目的是减少训练期的 gzip 解压与随机 IO 开销。npz 路径通常包含：

- `image`
- `label`
- `rw`
- `fg_slices`
- `fg_coords`
- `meta`

如果启用 spacing normalization，烘焙阶段会把样本重采样到统一 spacing，并把原始 spacing 与目标 spacing 一并写入 meta，方便推理时做逆变换。

### 3.3 bbox 与 region weight

bbox 用来缩小有效读盘范围；region weight 用来在损失里强调关注区域。两者都必须以 pid 严格对齐，训练和推理对它们的使用方式也要一致。

## 4. 训练时序

### 4.1 一次 step 的路径

```text
DataLoader worker
  └── Dataset.__getitem__
        ├── load image / label / rw
        ├── crop / pad / resize
        └── return batch item
GPU
  └── GPUAugmentor
        └── 同步空间增强
Trainer
  └── ViewPipeline
        ├── 组织主视图与辅助视图
        ├── 送入模型
        ├── 构造监督包
        └── 计算多分辨率损失
Backward
  └── AMP / grad clip / optimizer / scheduler / EMA
```

### 4.2 训练里的关键设计

- **AMP**：损失计算尽量在 fp32 中进行，避免 Dice / 分母类损失数值不稳定。
- **EMA**：验证时可临时切到 EMA shadow，验证后再恢复在线参数。
- **Checkpoint**：训练保存完整状态，保证恢复时优化器、调度器与随机状态都能继续。
- **Pipeline**：不同 patch_mode 的监督逻辑由策略对象接管，Trainer 本身不应该充满模式分支。

## 5. 推理时序

### 5.1 一次推理的路径

```text
Predictor
  ├── load checkpoint
  ├── 预处理 image / bbox
  ├── 按 patch_mode 切窗
  ├── forward + 可选 TTA
  ├── blend / threshold
  └── 写出预测结果
```

### 5.2 关键一致性

推理必须和训练保持同一套：

- 输入几何
- 通道组织
- patch 滑窗方式
- 归一化与反归一化规则
- bbox 裁剪策略

如果这些不一致，即使模型参数正确，输出也会出现几何错位。

## 6. 训练 / 推理 / 预打包 / 启动器 / 监控

### 6.1 训练

训练入口统一走 `train.py`，配置来自 YAML，必要时用 override 临时改参。训练产物会包括日志、最终配置和 checkpoint。

### 6.2 推理

推理入口统一走 `predict.py`，支持单文件与目录，支持显式 bbox，支持选择 online / EMA 权重。

### 6.3 npz 预打包

`make_data.py` 适合在正式训练前把原始 NIfTI 烘焙成 npz。它对多 worker 场景更友好，也能把大体积样本的随机读盘成本降下来。

### 6.4 launcher

launcher 提供一个本地网页表单，把 YAML 参数转成可编辑的表单项，并在页面内直接触发训练或推理。它适合不想手写配置、但又需要快速试验的场景。

### 6.5 monitor

monitor 负责把训练过程做成可离线打开的仪表盘。它不干预训练逻辑，只在训练侧写入历史，在需要时把历史渲染为 HTML。

### 6.6 visualization

visualization 关注静态结构图：数据流图、模型流图和预测流图。它与 monitor 正交，一个偏结构，一个偏时序。

## 7. 设计细节补充

### 7.1 模块职责边界

- `data/` 负责样本读写与几何抽取。
- `models/` 负责骨干、解码器和拓扑装配。
- `losses/` 负责目标函数。
- `trainer/` 负责优化、验证与 checkpoint。
- `predictor/` 负责推理、融合与写出。
- `launcher/`、`monitor/`、`visualization/` 负责工程可用性工具。

### 7.2 扩展原则

新增功能时优先遵守这几条：

1. 几何派生优先放到 topology。
2. 新 patch 模式优先在 data / pipeline / predictor 三处同步。
3. 新骨干优先在 models/factory 统一装配。
4. 新损失优先在 losses 建立单独类，再由工厂暴露。
5. 训练、推理和可视化工具要尽量保持松耦合。

## 8. 扩展指南

### 8.1 新增损失

在 `losses/losses.py` 加入新类，并在 `build_loss` 中注册名称。若该损失依赖特定配置字段，再同步更新 `config.py` 的校验逻辑与示例配置。

### 8.2 新增 backbone

在 `models/` 增加新模块，实现对应 stage / block，再在 `factory.py` 装配。若 backbone 影响输入通道、残差块数量或 stage 结构，尽量让 topology 先推导相关派生量。

### 8.3 新增 patch 模式

新增 patch 模式时，至少需要同步修改：

- `data/`：新增 Dataset 或抽样逻辑
- `models/topology.py`：补齐派生量
- `trainer/pipelines/`：新增监督策略
- `predictor/`：新增窗口构造与滑窗入口

### 8.4 新增多视图或上下文融合策略

多视图策略优先放在 `models/stem.py` 和 `trainer/views.py` 里管理，避免在主训练循环里硬编码分支。

### 8.5 改命名或路径约定

所有和命名、配对、路径规则相关的逻辑，优先集中在 `data/loader.py` 与 `data/make_data.py`，不要在各个下游文件里重复写判断。

## 9. 参考用法

```bash
# 训练
python -m segtask_v1.train --config configs/seg2_5d.yaml
python -m segtask_v1.train --config configs/seg3d.yaml
python -m segtask_v1.train --config configs/test_e2e.yaml

# 预测
python -m segtask_v1.predict --config configs/seg2_5d.yaml

# npz 预打包
python -m segtask_v1.data.make_data --config configs/seg2_5d.yaml --out-dir /path/to/npz --workers 8

# launcher / monitor
python -m segtask_v1.launcher
python -m segtask_v1.monitor runs/exp_a
```
