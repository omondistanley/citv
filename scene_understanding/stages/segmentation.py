"""Segmentation stage: mask detections from segmentation pipeline."""

from __future__ import annotations

from typing import Any

from ..core.prompting import refresh_gdino_query_for_staged
from ..pipeline_context import PipelineContext


def _should_run_pre_sam_gdino_query_refresh(cfg: Any) -> bool:
    if not cfg:
        return False
    mode = str(getattr(cfg, "query_builder_mode", "inherit") or "inherit").strip().lower()
    if mode == "static":
        return False
    if not bool(getattr(cfg, "rampp_enabled", True)):
        return False
    return mode in ("inherit", "rampp_full", "rampp_region_crops")


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Generate mask detections using package segmentation pipeline."""
    seg_pipe = pipeline._get_segmentation_pipeline()
    cfg = getattr(pipeline, "config", None)
    # Defaults: primary GroundedSAM2 path unless AMG is explicitly enabled.
    run_both = bool(getattr(cfg, "run_both_segmentors", False)) if cfg else False
    fallback = bool(getattr(cfg, "grounded_sam2_fallback_to_amg", False)) if cfg else False

    if _should_run_pre_sam_gdino_query_refresh(cfg):
        try:
            rampp = pipeline._ensure_rampp_for_gdino_query()
            tags, q_used = refresh_gdino_query_for_staged(
                cfg=cfg,
                img_rgb=ctx.img_rgb,
                region_partition_meta=list(ctx.region_partition_meta or []),
                width=ctx.width,
                height=ctx.height,
                seg_pipe=seg_pipe,
                rampp=rampp,
            )
            ctx.extra["gdino_query_tags_pre_sam"] = tags
            ctx.extra["gdino_query_used_pre_sam"] = q_used
        except Exception as exc:
            print(f"  [Segmentation] pre-SAM GDINO query refresh skipped: {exc}")

    detections = seg_pipe.generate(
        image_rgb=ctx.img_rgb,
        use_primary=True,
        use_secondary=run_both,
        use_fallback=fallback,
    )
    ctx.detections = list(detections)
    return ctx
