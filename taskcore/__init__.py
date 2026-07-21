"""taskcore — seg / gen / ssl / cls / det 五任务共用的框架层（唯一真相源）。

分层：
* ``taskcore.config``  —— 公共配置 dataclass 与 YAML 加载/校验；
* ``taskcore.data``    —— 数据发现/划分、NPZ 预烘焙、dataset/loader/增强；
* ``taskcore.models``  —— 公共骨干与拓扑（UNet 家族 / ADM / EDM2 等）；
* ``taskcore.engine``  —— 训练/推理工程件：AMP / optim / checkpoint / DDP /
  prefetch，以及共用基类 ``BaseTrainer`` / ``BasePredictor``；
* ``taskcore.monitor`` —— 训练监控仪表盘（jsonl + HTML，失败隔离）；
* ``taskcore.utils``   —— 日志、随机种子、AverageMeter / ModelEMA 等通用工具。

旧路径（如 ``segtask_v1.utils``）保留 re-export shim，行为不变。
"""
