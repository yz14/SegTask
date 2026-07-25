# taskcore — 五任务公共框架层

seg / cls / det / gen / ssl 五个任务共用的框架代码。设计取向：**不吞并任务主流程**，只把逐字重复的工程件收敛为公共类 / 函数，任务层按原有顺序显式调用。各任务的整体设计与使用方式见各自 `README.md` 与 `docs/WORKFLOW.md`（seg 另有 `docs/DESIGN.md`）。

## 模块地图

```
taskcore/
├── config/
│   ├── core.py              # 核心 Config（data/augment/model/train/vis/monitor）
│   │                        #   + ConfigError、段校验、DataclassLoadContext
│   ├── section_validators.py # 跨任务共享校验（stem/stage 长度等）
│   ├── seg_task.py          # SegTaskConfig（loss/predict）+ hoist_legacy_seg_sections
│   ├── seg_bundle.py        # SegBundle：运行期 core + seg 统一视图
│   ├── registry.py          # 任务段注册表（cls/det/ssl/seg）
│   ├── task_io.py           # 「核心 Config + 顶层任务段」YAML I/O / override
│   └── model_migration.py   # 旧 checkpoint 配置迁移
├── data/
│   ├── dataset.py           # SegDatasetNpzBase + 3 模式类（z/cubic/whole）、npz IO
│   ├── loader.py            # split 扫描/配对、DataLoader 装配
│   ├── specs.py             # DatasetCommonCfg、SplitPaths、patch_mode 策略
│   ├── augment.py           # GPU 增强（Companion 伴随张量）
│   ├── mixed_sampler.py     # 双源混采 MixedBatchSampler
│   ├── make_data.py         # 打包预处理（spacing/fg/meta skip/几何校验）
│   └── patch_*.py / patch_dataset_base.py / sampling.py  # cls/det 共用 patch 基类
├── models/
│   ├── factory.py           # build_model / build_backbone / UNet 家族构造
│   ├── topology.py          # build_topology：几何派生单一真相源
│   ├── blocks.py / stem.py / unet*.py / resnet.py / convnext.py / mednext.py
│   └── adm_unet.py / edm2_unet.py   # 扩散 backbone（gen 用）
├── engine/
│   ├── base_trainer.py      # BaseTrainer：EMA/SWA/AMP/DDP/_save_best/非有限守护等
│   ├── base_predictor.py    # BasePredictor：推理 AMP + flip TTA（有意薄）
│   ├── checkpoint.py        # 原子写、AsyncCheckpointSaver、extract_model_state_dict
│   ├── optim.py             # AdamW(fused) + wd 分组、warmup/调度、ZeRO-1
│   ├── amp.py / prefetch.py / bn_stats.py
│   ├── dist_utils.py / launch.py / memory.py / views.py
├── monitor/                 # 训练监控 dashboard
├── metrics.py               # 可加混淆量式指标（DDP all-reduce）
└── utils/                   # logging、AverageMeter/Timer 等
```

## 各任务接入方式

| 任务 | Config 接入 | 模型入口 | Trainer | Predictor |
|---|---|---|---|---|
| seg | `SegBundle` + `register_task_section("seg")`（`seg:` loss/predict） | `build_model` | `BaseTrainer` 子类（`_save_best`） | `BasePredictor` 子类（滑窗/blend/AdaBN；基类仅 AMP+flip TTA） |
| cls | `register_task_section("cls")` | `build_backbone` + 任务头 | `BaseTrainer` 子类 | `BasePredictor` 子类 |
| det | `register_task_section("det")` | `build_backbone` + FPN + 四头 | `BaseTrainer` 子类 | `BasePredictor` 子类 |
| gen | 段 dataclass 子类化 core；顶层 Config 组合 + 委托校验/I/O（model allowlist 任务侧） | 任务包 SISR + core ADM/EDM2 | `BaseTrainer` 子类（`_save_best`） | `BasePredictor` 子类 |
| ssl | `register_task_section("ssl")` | 方法插件自组 | `BaseTrainer` 子类（backbone 导出） | —（在线 SegProbe；离线 evaluate） |

- `loss`/`predict` 已下沉为 `seg:` 任务段；运行期经 `SegBundle` 仍可用 `cfg.loss` / `cfg.predict`。仓库示例 YAML 已迁入 `seg:`（旧式顶层仍 hoist 兼容）。
- gen：`io.py` 复用 `dataclass_from_dict`；`validation` 的 data/augment 委托 `CoreConfig`，`2_5d` 委托几何段（`check_channel_layout=False`）；`make_data` 委托 `prepare_one`。

## Checkpoint 槽位约定

统一读取入口：`taskcore.engine.checkpoint.extract_model_state_dict(ckpt, prefer_ema)`；跨任务 pretrain/resume 一律经它取权重。

| 生产者 | best / 导出 `model_state_dict` | 在线权重 | EMA 槽 |
|---|---|---|---|
| `BaseTrainer._save_best`（seg/cls/det/gen） | EMA 权重（选模/部署口径） | `model_online_state_dict` | `ema_state_dict` |
| ssl `ssl_best.pt` / `ssl_last.pt` | `export_backbone_state_dict()`（若启用 EMA则已 apply_shadow 烘焙进导出） | — | **无**独立 `ema_state_dict` |
| ssl `ssl_resume.pt` | method 全状态（非 backbone 导出） | — | 有 `ema_state_dict`（续训用） |

ssl 导出快照与 resume 全状态正交。历史 gen best（对齐前）primary 可能为在线权重，经 `extract_model_state_dict` 仍兼容。checkpoint 均原子写；DDP 仅 rank0 落盘；可选 `AsyncCheckpointSaver`。

## 公开符号与旧名别名

以下符号曾以下划线私有名被任务层跨包引用，现已转正为公开名（旧名保留为别名）：

| 公开名 | 旧名（别名保留） | 位置 | 下游使用 |
|---|---|---|---|
| `open_npz` | `_open_npz` | `data.dataset` | cls/det/gen/ssl |
| `extract_cubic_patch` | `_extract_cubic_patch`（别名在 `data.dataset`） | `data.patch_ops` | ssl、seg predictor |
| `get_conv(spatial_dims)` | （内部表 `_CONV`；优先用 `get_conv`） | `models.blocks` | det、ssl |
| `resize_logits` | `_resize_logits` | `models.unet` | ssl |
| `LOGIT_CLAMP` | `_LOGIT_CLAMP` | `engine.amp` | cls |
| `reseed_rank_rng` | `_reseed_rank_rng` | `engine.base_trainer` | ssl、seg |
| `check_physical_geometry` | `_check_physical_geometry` | `data.make_data` | `prepare_one`（gen make_data 委托） |
| `nested_dataclass_type` | `_nested_dataclass_type` | `config.core` | gen config io |

## 其它约定

- 配置校验统一抛 `ConfigError`；data 路径非法配置用 `ValueError`/`FileNotFoundError`；模型构造期少量 `assert` 为内部不变量。
- 数据层通过显式 `finalize_from_data(cfg, label_values, ...)` 提交探测结果；
  loader 不再自行回写配置。`data.group_id_regex` 统一优先启用组级划分，
  空值时保持任务原有分层/随机语义。
- `make_data` metadata 保留历史 spacing 键，并额外记录
  `achieved_spacing`、`pre_resample_shape`、`origin`、`direction`；旧 npz
  缺少新键时由读取端沿用既有 fallback。
- seg/gen 包内仍有兼容旧 import 的 re-export 模块（`[shim]`）；**新代码一律 `import taskcore...`**。声明式入口：`python -m segtask_v1.monitor`、`python -m segtask_v1.data.make_data`（`--out`）；`segtask_v1.config` 为旧路径→core 的 shim（新配置走 `segtask_v1.seg_config` / `SegBundle`）。
- DDP 指标：可加混淆量 + all-reduce(SUM)；batch 内池化比值损失（如 batch_dice）在 DDP 下为近似。
