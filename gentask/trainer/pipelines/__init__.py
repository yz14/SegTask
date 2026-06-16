"""``ViewPipeline`` 策略对象与工厂。"""

from .base import SupervisionPack, ViewPipeline
from .factory import build_pipeline
from .lift25d import Lift2_5DAuxPipeline, Lift2_5DPipeline
from .patch3d import Patch3DNativeMultiResPipeline
from .slab25d import (
    Slab2_5DAuxPipeline,
    Slab2_5DNativeDPipeline,
    Slab2_5DPipeline,
)
from .vanilla3d import Vanilla3DPipeline

__all__ = [
    "SupervisionPack", "ViewPipeline", "build_pipeline",
    "Vanilla3DPipeline", "Patch3DNativeMultiResPipeline",
    "Slab2_5DPipeline", "Slab2_5DAuxPipeline", "Slab2_5DNativeDPipeline",
    "Lift2_5DPipeline", "Lift2_5DAuxPipeline",
]
