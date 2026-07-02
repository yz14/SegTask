"""Generation predictor entry points.

The super-resolution path only needs :class:`GenerationPredictor` and
``run_generation_inference``.
"""

from __future__ import annotations

from .gen_predictor import GenerationPredictor, run_generation_inference  # noqa: F401

__all__ = [
    "GenerationPredictor",
    "run_generation_inference",
]
