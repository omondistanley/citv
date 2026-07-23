"""Relations stage: Pix2SG (and package RelationsPipeline)."""

from __future__ import annotations

import json
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

    # Optional raw-triplet debug export (SCENE_GRAPH_DEEP_DIVE.md §8 item 5):
    # per-layer triplets before dedup, for relation-fusion QA/experiments.
    cfg = getattr(pipeline, "config", None)
    if bool(getattr(cfg, "export_raw_relation_triplets", True)) if cfg else True:
        pix2sg = getattr(rel_pipe, "pix2sg", None)
        debug = getattr(pix2sg, "last_debug", None)
        if debug:
            staged_dir = ctx.output_dir / "scene_graph" / "staged"
            staged_dir.mkdir(parents=True, exist_ok=True)
            debug_path = staged_dir / f"{ctx.stem}_relations_raw_triplets.json"
            debug_path.write_text(json.dumps(debug, indent=2))
            ctx.path_exports["relations_raw_triplets_json"] = str(debug_path)
    return ctx
