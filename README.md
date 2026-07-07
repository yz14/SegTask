# SegTask — 医学影像多任务工程索引

SegTask 把分割、生成、自监督、分类、检测五个子项目放在同一仓库，底层共享 `segtask_v1` 的配置、拓扑、数据、训练与推理基础设施。仓库级 README 只做导航，不承载实现细节；各子项目的约定、命令和模块树都以各自 README 为准。

## 仓库树

```text
SegTask/
├── segtask_v1/  # 2.5D / 3D 分割主线：训练、预测、launcher、monitor、visualization 的基建真相源
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

## 子项目 README

- [`segtask_v1/README.md`](segtask_v1/README.md)
- [`gentask/README.md`](gentask/README.md)
- [`ssltask/README.md`](ssltask/README.md)
- [`clstask/README.md`](clstask/README.md)
- [`dettask/README.md`](dettask/README.md)
