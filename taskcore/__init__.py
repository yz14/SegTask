"""taskcore — seg / gen / ssl / cls / det 五任务共用的框架层（唯一真相源）。

分层：
* ``taskcore.utils``  —— 日志、随机种子、AverageMeter / ModelEMA 等通用工具；
* ``taskcore.engine`` —— 训练/推理工程件：amp / optim / checkpoint / dist_utils /
  memory / prefetch（后续步骤将加入 BaseTrainer / BasePredictor）；
* ``taskcore.config`` / ``taskcore.data`` / ``taskcore.models`` —— 按重构计划分步迁入。

迁移期间旧路径（如 ``segtask_v1.utils``）保留 re-export shim，行为不变。
"""
