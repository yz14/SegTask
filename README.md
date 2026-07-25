# SegTask — 医学影像多任务工程索引

SegTask 把分割、生成、自监督、分类、检测五个子项目放在同一仓库，底层共享顶层公共包 `taskcore`（配置 / 数据 / 模型 / 训练推理工程件 / 通用工具）。仓库级 README 只做导航，不承载实现细节；各子项目的约定、命令和模块树都以各自 README 为准。

## 仓库树

```text
SegTask/
├── taskcore/    # 五任务公共框架：config / data / models / engine（BaseTrainer、BasePredictor 等）/ utils
├── segtask_v1/  # 2.5D / 3D 分割主线：训练、预测、launcher、monitor、visualization
├── gentask/     # 生成 / 超分：回归与扩散两条路线
├── ssltask/     # 自监督预训练：骨干/配置经 taskcore（register_task_section("ssl")）
├── clstask/     # 3D / 2.5D 分类：复用基建并支持 SSL 权重迁移
├── dettask/     # 3D / 2.5D 检测：RetinaNet / FCOS / Faster R-CNN / DETR 四头
├── configs/     # 五个子项目共用的 YAML 示例配置
├── tools/       # 数据集体检与维护脚本
├── tests/       # 冒烟与回归测试
├── img_process/ # 图像处理辅助脚本
└── TODO.md      # 审查与重构进度（TODO 3 已关闭）
```

## 公共包 taskcore

五任务共用的工程基建统一住在 `taskcore/`，分六层：

- `taskcore.config` —— 公共配置（Data/Aug/Model/Train/Vis/Monitor）+ 任务段（`seg:`/`cls:`/`det:`/`ssl:`）与 YAML 加载/校验；分割运行期为 `SegBundle`；
- `taskcore.data` —— 数据发现/划分、NPZ 预处理（make_data）、dataset/loader/增强；
- `taskcore.models` —— 公共骨干与拓扑（UNet/UNet++/MedNeXt/ADM/EDM2 等）；
- `taskcore.engine` —— 训练/推理工程件与 `BaseTrainer` / `BasePredictor`；
- `taskcore.monitor` —— 训练监控仪表盘；
- `taskcore.utils` —— seed/计量/EMA/SWA/日志等。

仓内新代码直连 `taskcore`。seg/gen 包内仍保留若干旧路径 re-export（`[shim]`），供外部脚本与 legacy pickle；优先不要再新增对 shim 的依赖。

## 数据读取方案（五任务同语义）

- `data.patch_mode` 四模式全仓统一口径：`z_axis`（z 滑块 + H/W 面内整体 resize）、`2_5d`（同 z_axis 抽取，D 折进通道驱动 2D 模型）、`cubic`（三轴滑块）、`whole`（整体 resize）；seg/cls/det/gen 四模式全支持且抽取口径逐位一致（保证预训练 encoder 输入分布一致）；ssl 支持除 `whole` 外的三种（有意不支持，见 ssltask 文档）。
- 2.5D 折叠时机契约：默认 dataset 发未折叠 3D，GPU 增强后、**送模型前**才折叠。例外：det 恒在 dataset 折叠（框几何联动）；cls 在关闭 GPU 增强时也在 dataset 折叠。各任务细节见各自 WORKFLOW。
- 数据划分公共优先级：`data.group_id_regex` 非空时按组隔离，其次按任务配置分层，最后回退随机；空 regex 保持各任务原有分支。
- 批 1–3 的可选数值开关默认保持旧行为：`augment.elastic_field_mode=legacy`、
  `augment.elastic_normalize_displacement=false`、`data.split_rounding_mode=legacy`、
  `data.split_manifest_path=""`、`data.resize_antialias=false`、
  `train.pretrain_upkern_normalize=false`、`model.init_strategy=legacy`。
  z 轴采样默认 `data.z_sampling_mode=safe`；设为 `legacy` 可复现批 1
  之前的全域训练中心与验证 z-grid，但跨版本验证指标不可直接比较。

## 子项目 README

- [`segtask_v1/README.md`](segtask_v1/README.md)
- [`gentask/README.md`](gentask/README.md)
- [`ssltask/README.md`](ssltask/README.md)
- [`clstask/README.md`](clstask/README.md)
- [`dettask/README.md`](dettask/README.md)
