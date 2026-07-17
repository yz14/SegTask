# gentask — 医学影像超分与生成任务

gentask 是从 `segtask_v1` 剥离出的生成工程，覆盖医学影像超分、复原和条件生成两条主线。它既支持回归式重建，也支持条件扩散式采样，并把在线退化、训练、推理和数据预打包都放进统一的工程入口里。公共骨干 / 配置公共段 / 训练推理工程件已下沉到顶层公共包 `taskcore`，本包保留生成任务层；被迁移模块以 shim 保留旧 import 路径，行为不变（模块树中以 `[shim → taskcore.*]` 标注）。

> 端到端训练/推理流程见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

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
│   ├── dataclasses.py  # Config dataclass：公共段继承 taskcore.config.core，叠加生成任务段
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
│   ├── generative_models_survey.md  # 生成方案综述
│   └── WORKFLOW.md  # 端到端训练/推理流程
├── losses/  # 损失库
│   ├── __init__.py  # 损失包入口
│   └── recon.py  # 重建损失与加权封装
├── models/  # 网络与生成接口
│   ├── __init__.py  # 模型包入口
│   ├── adm_unet.py  # [shim → taskcore.models.adm_unet] ADM backbone
│   ├── blocks.py  # [shim → taskcore.models.blocks] 共享积木与注意力模块
│   ├── convnext.py  # [shim → taskcore.models.convnext] ConvNeXt stage / block
│   ├── diffusion.py  # 扩散 sampler / scheduler
│   ├── edm2_unet.py  # [shim → taskcore.models.edm2_unet] EDM2 backbone
│   ├── factory.py  # generation model 装配工厂
│   ├── generation.py  # 回归 / 扩散统一接口
│   ├── resnet.py  # [shim → taskcore.models.resnet] ResNet block / stage
│   ├── sisr.py  # 经典 SISR backbone（EDSR / RCAN）
│   ├── stem.py  # [shim → taskcore.models.stem] stem 与多视图融合
│   ├── topology.py  # [shim → taskcore.models.topology] 派生输入 / 输出几何真相源
│   ├── unet.py  # [shim → taskcore.models.unet] UNet encoder / decoder 主体
│   ├── unet3p.py  # [shim → taskcore.models.unet3p] UNet3+ decoder
│   └── unetpp.py  # [shim → taskcore.models.unetpp] UNet++ decoder
├── predictor/  # 推理器
│   ├── __init__.py  # 预测器包入口
│   └── gen_predictor.py  # generation 推理器（继承 taskcore.engine.BasePredictor）
└── trainer/  # 训练循环
    ├── __init__.py  # 训练包入口
    ├── amp.py  # [shim → taskcore.engine.amp] AMP / GradScaler 工具
    ├── checkpoint.py  # [shim → taskcore.engine.checkpoint] checkpoint 读写与兼容
    ├── gen_trainer.py  # generation 训练循环（继承 taskcore.engine.BaseTrainer）
    ├── optim.py  # [shim → taskcore.engine.optim] 优化器 / 调度器 / warmup
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
- **训练稳健性**：非有限 loss/梯度丢弃 accum 组（仅有效步更新 EMA），history.json 逐 epoch 落盘并随 `resume` 续接；`train.adamw_fused`（CUDA）与 norm/bias 免 weight decay 分组。
- **显存与多卡**：`model.grad_checkpointing` UNet 系 backbone 支持（非 UNet 架构开启时 warning 提示忽略）；`train.gpus` 配多卡即启用 DDP（mp.spawn 每卡一进程，PSNR/SSIM 加权跨卡归约，落盘仅 rank0），单卡/CPU 路径零变化。
- **推理可复现与加速**：扩散验证/推理采样用固定 seed generator 逐位可复现；`predict.use_amp` 推理 autocast、`predict.tta_flips` 可选翻转 TTA（decimate 被退化轴自动排除）。

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
