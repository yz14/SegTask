# SegTask — 医学影像多任务工程索引

SegTask 把分割、生成、自监督、分类、检测五个子项目放在同一仓库，底层共享顶层公共包 `taskcore`（配置 / 数据 / 模型 / 训练推理工程件 / 通用工具）。仓库级 README 只做导航，不承载实现细节；各子项目的约定、命令和模块树都以各自 README 为准。

## 仓库树

```text
SegTask/
├── taskcore/    # 五任务公共框架：config / data / models / engine（BaseTrainer、BasePredictor 等）/ utils
├── segtask_v1/  # 2.5D / 3D 分割主线：训练、预测、launcher、monitor、visualization
├── gentask/     # 生成 / 超分：回归与扩散两条路线
├── ssltask/     # 自监督预训练：复用 segtask_v1 骨干与配置核心
├── clstask/     # 3D / 2.5D 分类：复用基建并支持 SSL 权重迁移
├── dettask/     # 3D / 2.5D 检测：RetinaNet / FCOS / Faster R-CNN / DETR 四头
├── configs/     # 五个子项目共用的 YAML 示例配置
├── tools/       # 数据集体检与维护脚本
├── tests/       # 冒烟与回归测试
├── img_process/ # 图像处理辅助脚本
└── segtask/     # 早期 v0 原型，已冻结，仅供参考
```

## 公共包 taskcore

五任务共用的工程基建统一住在 `taskcore/`，分六层：

- `taskcore.config` —— 公共配置 dataclass（Data/Aug/Model/Loss/Train/Predict 等）与 YAML 加载/校验；各任务继承公共段并叠加自己的任务段；
- `taskcore.data` —— 数据发现/划分、NPZ 预处理（make_data）、dataset/loader/增强；
- `taskcore.models` —— 公共骨干与拓扑（UNet/UNet++/ADM/EDM2/SISR 等，含条件通道、扩散变体）；
- `taskcore.engine` —— 训练/推理工程件：AMP、优化器/调度、checkpoint、分布式（DDP 装配、mp.spawn 启动工程 `launch`、指标归约 `dist_utils`）、预取，以及共用基类 `BaseTrainer` / `BasePredictor`（各任务训练器/推理器子类化，只保留任务自己的主循环）；
- `taskcore.monitor` —— 训练监控仪表盘（jsonl + HTML，含离线渲染 CLI；rank0 守卫、失败隔离）；
- `taskcore.utils` —— seed/计量/EMA/SWA/日志等通用工具。

仓内代码均已直连 `taskcore`；旧 import 路径（如 `segtask_v1.config`、`segtask_v1.trainer.optim`、`gentask.trainer.checkpoint`）仍通过 shim 模块保留，供外部脚本与 legacy checkpoint pickle 使用，行为不变。

## 数据读取方案（五任务同语义）

- `data.patch_mode` 四模式全仓统一口径：`z_axis`（z 滑块 + H/W 面内整体 resize）、`2_5d`（同 z_axis 抽取，D 折进通道驱动 2D 模型）、`cubic`（三轴滑块）、`whole`（整体 resize）；seg/cls/det/gen 四模式全支持且抽取口径逐位一致（保证预训练 encoder 输入分布一致）；ssl 支持除 `whole` 外的三种（有意不支持，见 ssltask 文档）。
- 2.5D 折叠时机契约：dataset 恒发未折叠 3D（max-FOV × `aug_oversample_ratio` 余量），GPU 3D 增强后、**送模型前**才折叠；唯一例外是 det（折叠需与框几何联动，在 dataset 层完成）。各任务细节见各自 WORKFLOW。

## 子项目 README

- [`segtask_v1/README.md`](segtask_v1/README.md)
- [`gentask/README.md`](gentask/README.md)
- [`ssltask/README.md`](ssltask/README.md)
- [`clstask/README.md`](clstask/README.md)
- [`dettask/README.md`](dettask/README.md)
