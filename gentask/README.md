# gentask — 医学影像超分与生成任务

gentask 覆盖医学影像超分、复原与条件生成（回归 / 条件扩散）。公共骨干、配置公共段与训练推理工程件在 `taskcore`；本包保留生成任务层（退化、扩散、SISR、条件通道、pipelines）。配置为 **taskcore 子类扩展**（非平行 fork）：`io`/`validation` 委托 core，`make_data` 委托 `prepare_one`。部分模块仍为兼容 shim，**新代码请直连 `taskcore`**。

> 端到端流程见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

## 模块树

```text
gentask/
├── train.py / predict.py / __main__.py
├── logging_utils.py / utils.py    # [shim] → taskcore.utils.*
├── config/
│   ├── dataclasses.py             # 继承 taskcore 公共段 + 生成任务段
│   ├── io.py                      # YAML I/O（委托 taskcore dataclass_from_dict）
│   └── validation.py              # 任务/SISR/扩散专属校验；data/augment/2.5d 几何委托 core
├── data/
│   ├── augment.py                 # 薄封装（cond/weight_map 契约）
│   ├── degradation.py             # 在线超分退化
│   ├── loader.py / specs.py       # 发现/划分复用 core；gen DatasetCommonCfg 扩展
│   ├── make_data.py               # 样本发现 + 委托 taskcore.prepare_one
│   └── dataset/                   # Volume3D* + cond mixin；cache/io 部分 re-export
├── losses/recon.py
├── models/
│   ├── generation.py / diffusion.py / sisr.py / factory.py   # 任务侧
│   └── adm/edm2/unet*/…           # 多为 [shim] → taskcore.models.*
├── predictor/gen_predictor.py     # BasePredictor 子类
└── trainer/
    ├── gen_trainer.py             # BaseTrainer 子类（_save_best 用父类）
    ├── views.py                   # 多视图消费侧几何原语（任务侧）
    ├── pipelines/                 # 多视图消费管线
    └── amp/checkpoint/optim/prefetch  # [shim] → taskcore.engine.*
```

## 关键概念

- **两类范式**：回归复原 vs 条件扩散；共享数据与大部分配置。
- **在线退化**：训练时 HR→退化→回 HR 网格，几何在同一模块定义。
- **条件通道**：`data.cond_*` 使 `in_channels` 含 cond；2.5D 校验不套用分割的 `D*n_views` 通道等式。
- **统一接口**：`generation.py` 的 `forward` / `restore` / `degrade`。
- **拓扑真相源**：`taskcore.models.topology.build_topology`。
- **预打包**：`make_data` 与 seg 同口径（spacing/fg/meta skip），仅多 cond 配对。
- **训练稳健性**：非有限丢弃 accum 组、history 续接、fused AdamW、DDP（`train.gpus`）。

## 用法

```bash
python -m gentask.train --config configs/gensr_2_5d_regression.yaml
python -m gentask.train --config configs/gensr_2_5d_diffusion_adm.yaml
python -m gentask.predict --config configs/gensr_3d_zaxis_regression.yaml
```
