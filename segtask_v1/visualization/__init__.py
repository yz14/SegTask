"""全流程可视化分析工具（TODO #2）。

由 ``cfg.vis.enabled`` 守卫：训练启动时把「数据流 / 模型流 / 预测流」三视图导出为
一份**自包含 HTML**（零外部依赖、可离线打开），用于人工核对"数据流与模型架构是否
符合 yaml、是否有优化空间"。生成过程仅用 CPU dummy 张量，不读盘、不依赖 GPU 与真实数据。

公开入口：``generate_visualization(cfg, model) -> str``（返回写出的 HTML 路径）。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch.nn as nn

from taskcore.config.core import Config
from taskcore.models.topology import build_topology
from .data_flow import build_data_flow
from .graph import VisGraph
from .model_flow import build_model_flow
from .predict_flow import build_predict_flow
from .render import render_html

logger = logging.getLogger(__name__)


def _resolve_output_path(cfg: Config) -> str:
    """落盘路径：``vis.output_dir`` 为空时退到 ``train.output_dir/visualization``。"""
    out_dir = cfg.vis.output_dir.strip()
    if not out_dir:
        out_dir = os.path.join(cfg.train.output_dir, "visualization")
    return os.path.join(out_dir, cfg.vis.filename or "pipeline_vis.html")


def _cap_detail(graphs: dict, max_params: int) -> None:
    """限制每个节点 ``detail`` 的条数，超出截断并标注，防止详情面板爆炸。"""
    if max_params <= 0:
        return
    for g in graphs.values():
        for node in g.nodes:
            if len(node.detail) > max_params:
                kept = dict(list(node.detail.items())[:max_params])
                kept["…"] = f"(+{len(node.detail) - max_params} more, 已截断)"
                node.detail = kept


def generate_visualization(cfg: Config, model: Optional[nn.Module]) -> str:
    """构建三视图 IR、渲染 HTML 并落盘；返回写出的文件路径。

    * ``cfg.vis.flows`` 控制生成哪些视图及标签页顺序（默认 data / model / predict）。
    * ``model`` 为 None 时模型流退化为纯结构（无法做形状追踪）。
    * 单个视图构建失败不影响其余视图（记录告警、跳过）。
    """
    topo = build_topology(cfg)
    flows = list(cfg.vis.flows) or ["data", "model", "predict"]

    graphs = {}
    for flow in flows:
        try:
            if flow == "data":
                graphs["data"] = build_data_flow(cfg, topo)
            elif flow == "model":
                if model is None:
                    logger.warning(
                        "vis: model 为 None，模型流退化为空视图。")
                    graphs["model"] = VisGraph(title="模型流 Model Flow")
                else:
                    graphs["model"] = build_model_flow(
                        cfg, model, topo, trace_shapes=cfg.vis.trace_shapes)
            elif flow == "predict":
                graphs["predict"] = build_predict_flow(cfg, topo)
            else:
                logger.warning("vis: 未知 flow %r，跳过。", flow)
        except Exception as e:  # 单视图失败不连累整体
            logger.warning("vis: 构建 %s 流失败: %s", flow, e)

    _cap_detail(graphs, int(cfg.vis.max_detail_params))

    order = [f for f in flows if f in graphs]
    html_str = render_html(graphs, order)

    out_path = _resolve_output_path(cfg)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    logger.info("Pipeline visualization HTML written: %s", out_path)
    return out_path


__all__ = ["generate_visualization", "build_data_flow",
           "build_model_flow", "build_predict_flow", "render_html"]
