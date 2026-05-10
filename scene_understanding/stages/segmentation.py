"""Segmentation stage: mask detections from segmentation pipeline."""

from __future__ import annotations

from typing import Any

from ..pipeline_context import PipelineContext


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Generate mask detections using package segmentation pipeline."""
    seg_pipe = pipeline._get_segmentation_pipeline()
    detections = seg_pipe.generate(
        image_rgb=ctx.img_rgb,
        use_primary=True,
        use_secondary=False,
        use_fallback=True,
    )
    ctx.detections = list(detections)
    return ctx
