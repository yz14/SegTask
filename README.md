# 仓库索引

本仓库包含三个并列的子项目：

- `segtask_v1/` —— 2.5D / 3D 医学图像分割，含训练、预测、网页启动器、可视化与训练监控。
- `gentask/` —— 生成 / 超分辨率，含回归与扩散两条范式及生成专属适配特性。
- `ssltask/` —— 自监督预训练，复用 `segtask_v1` 的骨干与配置核心。

各子项目的详细说明见其各自的 README：

- [`segtask_v1/README.md`](segtask_v1/README.md)
- [`gentask/README.md`](gentask/README.md)
- [`ssltask/README.md`](ssltask/README.md)

仓库级共享目录：

- `configs/` —— 三个项目的 YAML 示例配置。
- `tools/` —— 数据集体检 / 维护工具。
- `tests/` —— 冒烟与回归测试。
- `outputs/` —— 本地实验产物。
- `img_process/` —— 图像处理辅助脚本。
- `segtask/` —— 早期 v0 原型，已冻结，仅供参考。
