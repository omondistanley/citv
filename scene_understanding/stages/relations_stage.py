"""Relations stage: Pix2SG (and package RelationsPipeline)."""

from __future__ import annotations

from typing import Any

from ..pipeline_context import PipelineContext


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Predict relation triplets from staged objects."""
    rel_pipe = pipeline._get_relations_pipeline()
    objects = list(ctx.extra.get("objects", []))
    relations = rel_pipe.predict_relations(
        image=ctx.img_bgr,
        image_stem=ctx.stem,
        detections=objects,
        iou_func=pipeline._bbox_iou_xyxy,
    )
    ctx.relations = list(relations)
    return ctx
