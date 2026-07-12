# gentask — 医学影像超分与生成任务

gentask 是从 `segtask_v1` 剥离出的生成工程，覆盖医学影像超分、复原和条件生成两条主线。它既支持回归式重建，也支持条件扩散式采样，并把在线退化、训练、推理和数据预打包都放进统一的工程入口里。

## 模块树

```text
gentask/
├── README.md  # 本文件
├── __init__.py  # 包入口与对外导出
├── __main__.py  # python -m gentask 入口
├── train.py  # 训练 CLI
├── predict.py  # 推理 CLI
├── logging_utils.py  # 日志初始化工具
├── utils.py  # 通用工具
├── config/  # 配置系统
│   ├── __init__.py  # 配置包入口
│   ├── dataclasses.py  # Config / Task / Model dataclass 定义
│   ├── io.py  # YAML 读写与 resolved config 保存
│   └── validation.py  # 生成任务配置校验
├── data/  # 数据与退化系统
│   ├── __init__.py  # 数据包入口
│   ├── augment.py  # GPU 3D 生成数据增强
│   ├── degradation.py  # 在线超分退化算子
│   ├── loader.py  # 路径匹配与 dataloader 工厂
│   ├── make_data.py  # 离线预打包 npz
│   ├── specs.py  # patch / view 规格推导
│   └── dataset/  # npz 数据集子包
│       ├── __init__.py  # dataset 子包入口
│       ├── cache.py  # 缓存与 LRU 管理
│       ├── core.py  # npz dataset 主逻辑
│       └── io.py  # npz / NIfTI 读取工具
├── docs/  # 生成模型综述与设计备忘
│   └── generative_models_survey.md  # 生成方案综述
├── losses/  # 损失库
│   ├── __init__.py  # 损失包入口
│   └── recon.py  # 重建损失与加权封装
├── models/  # 网络与生成接口
│   ├── __init__.py  # 模型包入口
│   ├── adm_unet.py  # ADM backbone
│   ├── blocks.py  # 共享积木与注意力模块
│   ├── convnext.py  # ConvNeXt stage / block
│   ├── diffusion.py  # 扩散 sampler / scheduler
│   ├── edm2_unet.py  # EDM2 backbone
│   ├── factory.py  # generation model 装配工厂
│   ├── generation.py  # 回归 / 扩散统一接口
│   ├── resnet.py  # ResNet block / stage
│   ├── sisr.py  # 经典 SISR backbone（EDSR / RCAN）
│   ├── stem.py  # stem 与多视图融合
│   ├── topology.py  # 派生输入 / 输出几何真相源
│   ├── unet.py  # UNet encoder / decoder 主体
│   ├── unet3p.py  # UNet3+ decoder
│   └── unetpp.py  # UNet++ decoder
├── predictor/  # 推理器
│   ├── __init__.py  # 预测器包入口
│   └── gen_predictor.py  # generation 推理器
└── trainer/  # 训练循环
    ├── __init__.py  # 训练包入口
    ├── amp.py  # AMP / GradScaler 工具
    ├── checkpoint.py  # checkpoint 读写与兼容
    ├── gen_trainer.py  # generation 训练循环
    ├── optim.py  # 优化器 / 调度器 / warmup
    ├── views.py  # 多视图消费侧几何原语
    └── pipelines/  # 多视图消费管线
        ├── __init__.py  # 管线包入口
        ├── base.py  # 管线抽象基类
        ├── native_d.py  # 2.5D 原生深度多视图管线
        ├── stacked.py  # 多视图堆叠管线（逐视图裁剪+resize+通道堆叠）
        └── vanilla.py  # 单视图管线（中心裁剪回 patch_size）
```

## 关键概念

- **两类生成范式**：回归复原适合直接映射，条件扩散适合从噪声迭代采样，两个范式共享大部分数据和配置体系。
- **在线退化**：训练时先把 HR 做退化再回到 HR 网格，数据路径和输入几何都在同一模块里定义。
- **多视图与条件输入**：2.5D 多视图、辅助重建、外部条件输入都围绕同一套数据契约展开。
- **统一模型接口**：`generation.py` 把回归与扩散接口统一成同一套 `forward / restore / degrade` 语义。
- **拓扑真相源**：输入输出几何、通道布局和视图关系都由 topology 层统一推导，不在别处重复计算。
- **预打包数据**：`make_data.py` 支持把 NIfTI 烘焙成 npz，减少训练时的 IO 压力。

## 用法

```bash
# 训练
python -m gentask.train --config configs/gensr_2_5d_regression.yaml
python -m gentask.train --config configs/gensr_2_5d_diffusion_adm.yaml

# 推理
python -m gentask.predict --config configs/gensr_3d_zaxis_regression.yaml

# 冒烟测试
python tests/test_generation_smoke.py
python tests/test_data_pipeline_smoke.py
```
