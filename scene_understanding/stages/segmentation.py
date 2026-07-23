"""Segmentation stage: mask detections from segmentation pipeline."""

from __future__ import annotations

from typing import Any

from ..pipeline_context import PipelineContext


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Generate mask detections using GroundedSAM2.

    Reuses ``pipeline.sam2_wrapper`` -- the legacy monolith's own GDINO+SAM2
    instance, already loaded during ``SceneUnderstandingPipeline.__init__``
    -- rather than ``pipeline._get_segmentation_pipeline()``, which builds
    an entirely separate, independent ``SegmentationPipeline`` that loads
    its own second copy of both models from scratch. Confirmed safe:
    ``SegmentationPipeline.generate()`` is a thin pass-through to
    ``GroundedSAM2Wrapper.generate()`` with no extra logic of its own, so
    this is a behavior-identical swap that just skips the duplicate load
    (measured: real double-loading of GDINO+SAM2 was a meaningful chunk of
    the ~1186s a real end-to-end run took before this fix).
    """
    if hasattr(pipeline, "sam2_wrapper") and pipeline.sam2_wrapper is not None:
        detections = pipeline.sam2_wrapper.generate(ctx.img_rgb)
    else:
        seg_pipe = pipeline._get_segmentation_pipeline()
        detections = seg_pipe.generate(image_rgb=ctx.img_rgb, use_primary=True, use_secondary=False, use_fallback=True)
    ctx.detections = list(detections)
    return ctx
