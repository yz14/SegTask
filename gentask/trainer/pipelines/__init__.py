"""生成任务多视图消费管线（batch 几何准备）。"""

from .base import GenViewPipeline
from .factory import build_pipeline
from .native_d import NativeDPipeline
from .stacked import StackedMultiResPipeline
from .vanilla import VanillaPipeline

__all__ = [
    "GenViewPipeline", "VanillaPipeline", "StackedMultiResPipeline",
    "NativeDPipeline", "build_pipeline",
]
