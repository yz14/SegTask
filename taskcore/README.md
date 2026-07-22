# taskcore — 五任务公共框架层

seg / cls / det / gen / ssl 五个任务共用的框架代码。设计取向：**不吞并任务主流程**，只把逐字重复的工程件收敛为公共类 / 函数，任务层按原有顺序显式调用。各任务的整体设计与使用方式见各自 `README.md` 与 `docs/WORKFLOW.md`（seg 另有 `docs/DESIGN.md`）。

## 模块地图

```
taskcore/
├── config/
│   ├── core.py            # 核心 Config（data/model/train/aug/loss/predict/vis/monitor 段）、
│   │                      #   ConfigError、校验器；注意 loss/predict 目前仍是 seg 形（P2a 待下沉）
│   ├── registry.py        # 任务段注册表：cls/det/ssl 经 register_task_section 装配
│   │                      #   （组合式任务跳过 loss/predict 校验器）；gen 走 Config 子类 fork
│   ├── task_io.py         # 「核心 Config + 顶层任务段」的 YAML 加载/保存/dotted override
│   └── model_migration.py # 旧 checkpoint 配置迁移
├── data/
│   ├── dataset.py         # 3 个 dataset 基类（3D / cubic / whole）、npz IO、patch 抽取、预处理
│   ├── loader.py          # split 扫描/配对、DataLoader 装配
│   ├── specs.py           # DatasetCommonCfg（12 个公共构造参数）、SplitPaths
│   ├── augment.py         # 增强管线（伴随张量 companion 机制）
│   ├── mixed_sampler.py   # 双源混采 MixedBatchSampler（DDP 下按 world_size 丢余数 batch）
│   ├── make_data.py       # 打包预处理（spacing 归一化、前景索引、几何校验）
│   └── patch_*.py / sampling.py
├── models/
│   ├── factory.py         # build_model（完整 UNet+头，seg 形）/ build_backbone（cls/det 用）
│   │                      #   / UNet 家族 (encoder, decoder) 构造
│   ├── topology.py        # build_topology：patch_mode × mode flags → 全部几何派生量
│   ├── blocks.py / stem.py / unet*.py / resnet.py / convnext.py / mednext.py
│   └── adm_unet.py / edm2_unet.py   # 扩散 backbone（gen 用）
├── engine/
│   ├── base_trainer.py    # BaseTrainer：EMA/SWA/AMP/DDP/checkpoint/非有限守护等工程件
│   ├── base_predictor.py  # BasePredictor：仅推理 AMP + flip TTA 组合枚举（有意薄；
│   │                      #   滑窗/blend/网格几何在任务侧）
│   ├── checkpoint.py      # 原子写、AsyncCheckpointSaver、extract_model_state_dict、unwrap_compile
│   ├── optim.py           # AdamW(fused) + wd 分组、warmup/调度、ZeRO-1
│   ├── amp.py / prefetch.py (CudaPrefetcher) / bn_stats.py (AdaBN/SWA BN 重估)
│   ├── dist_utils.py / launch.py / memory.py / views.py
├── monitor/               # 训练监控 dashboard（history.json 消费端）
├── metrics.py             # 可加混淆量式指标（DDP all-reduce 严格等价）
└── utils/                 # logging、AverageMeter/Timer 等
```

## 各任务接入方式

| 任务 | Config 接入 | 模型入口 | Trainer | Predictor |
|---|---|---|---|---|
| seg | 直接用核心 `Config`（历史即 seg 形） | `build_model` | 自管 trainer（槽位约定与 BaseTrainer 一致） | 自有 predictor（滑窗/blend/AdaBN） |
| cls | `register_task_section("cls")` | `build_backbone` + 任务头 | `BaseTrainer` 子类（用 `_save_best`） | `BasePredictor` 子类 |
| det | `register_task_section("det")` | `build_backbone` + FPN + 四头 | `BaseTrainer` 子类（用 `_save_best`） | `BasePredictor` 子类（自写滑窗） |
| gen | **Config 子类 fork**（`gentask/config/dataclasses.py`） | 任务包 SISR + core ADM/EDM2 | `BaseTrainer` 子类（自管 `_save_best`） | `BasePredictor` 子类 |
| ssl | `register_task_section("ssl")` | 方法插件自组（backbone 经 core blocks） | `BaseTrainer` 子类（自管保存） | —（下游评估走 probe） |

已知双轨：cls/det/ssl 走 registry、gen 走子类 fork；`loss`/`predict` 段仍常驻 core（组合式任务仅跳过校验）。P2a 后续计划见 `config/registry.py` 模块 docstring。

## Checkpoint 槽位约定

统一读取入口：`taskcore.engine.checkpoint.extract_model_state_dict(ckpt, prefer_ema)`，兼容全部布局；跨任务 pretrain/resume 一律经它取权重，不要手工读槽位。

| 生产者 | best 的 `model_state_dict` | 在线权重 | EMA |
|---|---|---|---|
| BaseTrainer._save_best（cls/det） | EMA 权重（选模/部署口径） | `model_online_state_dict` | `ema_state_dict` |
| seg trainer（自管） | 同上（best 时 `ema_as_primary`） | `model_online_state_dict` | `ema_state_dict` |
| gen trainer（自管） | 同上（EMA 作 primary，已与 seg/cls/det 对齐） | `model_online_state_dict` | `ema_state_dict` |
| ssl trainer（自管） | `export_backbone_state_dict()` 导出的 backbone 键 | — | `ema_state_dict` |

注意：ssl 的 primary 语义不同（预训练产物即 backbone 导出，属有意设计）；gen 的历史 best checkpoint（对齐前训练的）primary 为在线权重，经 `extract_model_state_dict` / `_select_state_dict` 读取仍兼容。所有 checkpoint 均原子写（临时文件 + `os.replace`），DDP 下仅 rank0 落盘，可选 `AsyncCheckpointSaver` 后台写。

## 公开符号与旧名别名

以下符号曾以下划线私有名被任务层跨包引用，现已转正为公开名（旧下划线名保留为别名，兼容存量代码/pickle/tests，新代码一律用公开名）：

| 公开名 | 旧名（别名保留） | 位置 | 下游使用 |
|---|---|---|---|
| `open_npz` | `_open_npz` | `data.dataset` | cls/det/gen/ssl |
| `extract_cubic_patch` | `_extract_cubic_patch` | `data.patch_ops`（dataset 转发） | ssl、seg predictor |
| `get_conv(spatial_dims)` | `_CONV[d]` 字典（仍为模块内部） | `models.blocks` | det heads/fpn、ssl modules/probe |
| `resize_logits` | `_resize_logits` | `models.unet` | ssl |
| `LOGIT_CLAMP` | `_LOGIT_CLAMP` | `engine.amp` | cls |
| `reseed_rank_rng` | `_reseed_rank_rng` | `engine.base_trainer` | ssl、seg trainer |
| `check_physical_geometry` | `_check_physical_geometry` | `data.make_data` | gen |
| `nested_dataclass_type` | `_nested_dataclass_type` | `config.core` | gen config io |

## 其它约定

- 配置校验统一抛 `ConfigError`（`config/core.py`）；data/model 侧仍有少量 `assert`（`python -O` 下会静默失效，见 TODO.md）。
- 旧路径 shim：segtask_v1 下大量 `[shim] 已迁移至 taskcore...` 别名模块，行为不变，仅为兼容旧 import/pickle；新代码一律直接 import taskcore。
- DDP 指标：`metrics.py` 采用可加混淆量 + all-reduce(SUM)，与单进程全集计算严格相等；训练侧 batch 内池化的比值损失（如 batch_dice）在 DDP 下为近似（每卡池化后梯度均值）。
