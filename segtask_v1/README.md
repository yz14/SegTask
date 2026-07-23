# segtask_v1 — 分割主线导航地图

`segtask_v1` 是当前活跃的 2.5D / 3D 医学图像分割工程：训练、预测、launcher、visualization 都围绕这条主线展开。公共基建在顶层包 `taskcore`；本包保留分割任务层（损失、滑窗推理、ViewPipeline、launcher）。配置经 `seg_config` 加载为 `SegBundle`（core + `seg:`）；部分旧路径仍以 re-export shim 兼容，**新代码请直连 `taskcore`**。

> 设计级细节见 [`docs/DESIGN.md`](docs/DESIGN.md)，端到端流程见 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

## 模块树

```text
segtask_v1/
├── train.py / predict.py / __main__.py
├── seg_config.py          # 正式配置入口 → SegBundle（seg: loss/predict）
├── config.py              # [shim] → taskcore.config.core（pickle/旧 import）
├── logging_utils.py / utils.py   # [shim] → taskcore.utils.*
├── data/                  # 多为 [shim] → taskcore.data.*；CLI: make_data
├── models/                # 多为 [shim] → taskcore.models.*
├── monitor/               # [shim] → taskcore.monitor（保留 python -m 入口）
├── losses/                # 分割损失（Dice/BCE/…、topo_aux）
├── predictor/             # 滑窗 / blend / AdaBN / TTA（任务侧）
├── trainer/               # SegTrainer + ViewPipeline；工程件 shim → taskcore.engine
├── launcher/              # 本地网页启动器
├── visualization/         # 数据流 / 模型流 / 预测流
└── docs/
```

## 关键概念

- **配置**：YAML 写公共段 + `seg:`（`loss`/`predict`；仓库示例已迁入）。旧式顶层仍兼容（hoist）；与 `seg.*` 并存则报错。运行期 `cfg.loss` / `cfg.predict` 仍可用（`SegBundle`）。
- **Patch 模式是主几何开关**：`z_axis` / `cubic` / `whole` / `2_5d`；训练与推理必须一致。
- **Topology 是单一真相源**：`taskcore.models.topology.build_topology`。
- **pid 命名契约**：图像/标签/bbox/权重/npz 按 pid 对齐；缺失配对报错。
- **npz 预打包**：`python -m segtask_v1.data.make_data --out ...`（实现在 `taskcore.data.make_data`）。
- **选模**：走 `BaseTrainer._save_best`（与 cls/det/gen 槽位一致）。
- **AdaBN**：Predictor 编排；BN 统计在 `taskcore.engine.bn_stats`（`predictor/adabn.py` 为 shim）。

## 用法

```bash
# 训练
python -m segtask_v1.train --config configs/seg2_5d.yaml
python -m segtask_v1.train --config configs/seg3d.yaml

# 推理
python -m segtask_v1.predict --config configs/seg2_5d.yaml

# npz 预打包
python -m segtask_v1.data.make_data --config configs/seg2_5d.yaml --out /path/to/npz --workers 8

# launcher / monitor
python -m segtask_v1.launcher
python -m segtask_v1.monitor runs/exp_a
```
