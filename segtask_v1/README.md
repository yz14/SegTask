# segtask_v1 — 分割主线导航地图

`segtask_v1` 是当前活跃的 2.5D / 3D 医学图像分割工程：训练、预测、launcher、monitor、visualization 都围绕这条主线展开。它把数据读取、patch 几何、模型装配、损失、训练循环与推理流程收拢到一套配置驱动的体系里。公共基建（配置 / 数据 / 模型 / 训练推理工程件 / 工具）现已下沉到顶层公共包 `taskcore`，本包保留分割任务层；被迁移模块以 shim 保留旧 import 路径，行为不变（模块树中以 `[shim → taskcore.*]` 标注）。

> 设计级细节已拆到 [`docs/DESIGN.md`](docs/DESIGN.md)，端到端训练/推理流程见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。这里仅保留导航地图和关键约定。

## 模块树

```text
segtask_v1/
├── README.md  # 本文件：导航地图
├── __init__.py  # 包入口
├── __main__.py  # python -m segtask_v1 入口
├── train.py  # 训练 CLI
├── predict.py  # 推理 CLI
├── config.py  # [shim → taskcore.config.core] 配置系统与派生字段
├── logging_utils.py  # [shim → taskcore.utils.logging_utils] 日志工具
├── utils.py  # [shim → taskcore.utils.common] 通用工具
├── data/  # [shim → taskcore.data] 数据发现 / IO / patch / 预打包
│   ├── __init__.py  # 数据包入口
│   ├── augment.py  # GPU 共享增强
│   ├── dataset.py  # NIfTI / npz 读取、patch 抽取、bbox / 缓存、Dataset
│   ├── loader.py  # 样本发现、切分、dataloader 工厂
│   ├── make_data.py  # 离线烘焙 npz
│   ├── mixed_sampler.py  # 混合采样器
│   └── specs.py  # patch_mode / split 策略层
├── docs/  # 设计与流程文档
│   ├── DESIGN.md  # 设计级细节
│   └── WORKFLOW.md  # 四方案端到端训练/推理流程
├── launcher/  # 本地网页启动器
│   ├── __init__.py  # 包入口
│   ├── __main__.py  # launcher CLI
│   ├── assets.py  # 前端资源与样式
│   ├── build.py  # 页面构建
│   ├── manifest.py  # 配置清单与表单元数据
│   ├── page.py  # 页面渲染
│   ├── process.py  # 子进程启动与管理
│   ├── schema.py  # 表单 schema
│   └── server.py  # HTTP 服务
├── losses/  # 损失库
│   ├── __init__.py  # 损失包入口
│   ├── losses.py  # Dice / BCE / Focal / Tversky / GDL / clDice / wrapper
│   └── topo_aux.py  # 拓扑辅助损失
├── models/  # [shim → taskcore.models] 模型骨架
│   ├── __init__.py  # 模型包入口
│   ├── adm_unet.py  # ADM UNet
│   ├── blocks.py  # 通用积木与采样算子
│   ├── convnext.py  # ConvNeXt 块
│   ├── edm2_unet.py  # EDM2 UNet
│   ├── factory.py  # 模型装配工厂
│   ├── mednext.py  # MedNeXt 骨干
│   ├── resnet.py  # ResNet 骨干
│   ├── stem.py  # 输入 stem 与多视图融合
│   ├── topology.py  # 几何与通道布局的单一真相源
│   ├── unet.py  # UNet 主体
│   ├── unet3p.py  # UNet3+
│   └── unetpp.py  # UNet++
├── monitor/  # 训练监控仪表盘
│   ├── __init__.py  # 监控包入口
│   ├── __main__.py  # monitor CLI
│   ├── assets.py  # 页面样式与脚本
│   ├── charts.py  # 指标整理与图表数据
│   ├── dashboard.py  # HTML 仪表盘渲染
│   └── history.py  # 训练历史与落盘
├── predictor/  # 滑窗推理
│   ├── __init__.py  # 推理包入口
│   ├── adabn.py  # 推理期 AdaBN
│   ├── blending.py  # 概率融合
│   ├── forwards.py  # forward / TTA 变体
│   ├── inputs.py  # 窗口与 batch 构造
│   ├── io.py  # checkpoint / precision / run_inference
│   ├── predictor.py  # Predictor 外壳与入口（继承 taskcore.engine.BasePredictor）
│   └── sliding.py  # whole / z / interleave / cubic 滑窗主循环
├── trainer/  # 训练循环与策略管线
│   ├── __init__.py  # 训练包入口
│   ├── amp.py  # [shim → taskcore.engine.amp] AMP 相关封装
│   ├── breakdown.py  # 多分辨率损失分解
│   ├── checkpoint.py  # [shim → taskcore.engine.checkpoint] checkpoint I/O
│   ├── dist_utils.py  # [shim → taskcore.engine.dist_utils] 分布式辅助
│   ├── memory.py  # [shim → taskcore.engine.memory] 显存预算与统计
│   ├── optim.py  # [shim → taskcore.engine.optim] 优化器 / 调度器 / warmup
│   ├── trainer.py  # Trainer 主类（继承 taskcore.engine.BaseTrainer）
│   ├── validation.py  # 验证逻辑
│   ├── views.py  # 视图切分与拼接
│   └── pipelines/  # ViewPipeline 策略对象
│       ├── __init__.py  # 管线包入口
│       ├── base.py  # 管线基类与监督包
│       ├── factory.py  # 管线装配入口
│       ├── lift25d.py  # 2.5D lift 管线
│       ├── patch3d.py  # 3D patch 管线
│       ├── slab25d.py  # 2.5D slab 管线
│       └── vanilla3d.py  # 3D whole 管线
└── visualization/  # 数据流 / 模型流 / 预测流可视化
    ├── __init__.py  # 可视化入口
    ├── data_flow.py  # 数据流图构建
    ├── graph.py  # 图中间表示
    ├── model_flow.py  # 模型流图构建
    ├── predict_flow.py  # 预测流图构建
    └── render.py  # HTML 渲染
```

## 关键概念

- **Patch 模式是主几何开关**：`z_axis`、`cubic`、`whole`、`2_5d` 都围绕同一套几何与数据契约工作，训练和推理必须保持一致。
- **FOV 只在一个层次上变化**：数据集先给出单分辨率最大 FOV cube，多分辨率和多视图的裁剪、缩放、拼接在后续阶段完成。
- **Topology 是单一真相源**：`ModelTopology` 负责推导输入通道、输出类别、`spatial_dims`、多视图数量和 aux 头开关，避免多处重复计算。
- **pid 命名契约**：图像、标签、bbox、region weight、npz、exclude list 都围绕 pid 对齐；缺失配对默认报错，不做静默跳过。
- **npz 预打包**：`make_data.py` 负责把 NIfTI 烘焙成可 mmap 的 npz，减轻 gzip 解码和重复 IO 压力。
- **设计细节下沉**：训练时序、预测时序、扩展指南和数据流图放到 `docs/DESIGN.md`；whole / cubic / zaxis / 2.5d 四方案的端到端流程与通用训练/推理技巧见 `docs/WORKFLOW.md`。README 只保留导航。

## 用法

```bash
# 训练
python -m segtask_v1.train --config configs/seg2_5d.yaml
python -m segtask_v1.train --config configs/seg3d.yaml
python -m segtask_v1.train --config configs/test_e2e.yaml

# 推理
python -m segtask_v1.predict --config configs/seg2_5d.yaml

# npz 预打包
python -m segtask_v1.data.make_data --config configs/seg2_5d.yaml --out-dir /path/to/npz --workers 8

# 本地 launcher 与 monitor
python -m segtask_v1.launcher
python -m segtask_v1.monitor runs/exp_a
```
