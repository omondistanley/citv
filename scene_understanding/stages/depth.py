"""Depth stage: load or infer metric depth."""

from __future__ import annotations

from typing import Any

from ..pipeline_context import PipelineContext


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Infer/reuse metric depth via DepthCoordinator."""
    depth_dir = ctx.extra["depth_dir"]
    reuse_existing = bool(getattr(pipeline, "_reuse_cached_depth", True))
    metric_depth = pipeline.depth_coordinator.load_or_infer_depth(
        image_rgb=ctx.img_rgb,
        output_dir=depth_dir,
        image_stem=ctx.stem,
        reuse_existing=reuse_existing,
    )
    ctx.metric_depth = metric_depth
    return ctx
