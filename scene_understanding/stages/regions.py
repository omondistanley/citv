"""Optional depth regions partition stage."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..pipeline_context import PipelineContext


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Optional depth K-means + CC regions (writes label map + PNG under depth/)."""
    cfg = getattr(pipeline, "config", None)
    if not bool(getattr(cfg, "regions_enabled", False)) if cfg else False:
        return ctx
    if ctx.metric_depth is None:
        return ctx
    try:
        from ..regions.partitioner import label_map_to_bgr, partition_depth_regions
    except Exception as e:
        print(f"  [Regions] skipped (import failed): {e}")
        return ctx

    depth_dir = ctx.extra["depth_dir"]
    _dm = str(getattr(pipeline.depth_estimator, "backend_name", "DepthAnythingV2"))
    _k = int(getattr(cfg, "regions_k", 4)) if cfg else 4
    _min_px = int(getattr(cfg, "regions_min_region_px", 500)) if cfg else 500
    _blur = float(getattr(cfg, "regions_blur_sigma", 0.0)) if cfg else 0.0
    _seed = int(getattr(cfg, "regions_seed", 42)) if cfg else 42
    _part = partition_depth_regions(
        ctx.metric_depth,
        k=_k,
        min_region_px=_min_px,
        blur_sigma=_blur,
        seed=_seed,
        depth_model_id=_dm,
    )
    ctx.region_label_map = _part.label_map
    ctx.region_partition_meta = list(_part.regions)
    _bgr = label_map_to_bgr(_part.label_map, _part.palette)
    rpng = depth_dir / f"{ctx.stem}_regions.png"
    cv2.imwrite(str(rpng), _bgr)
    np.save(str(depth_dir / f"{ctx.stem}_regions_label_map.npy"), _part.label_map.astype(np.int32))
    ctx.extra["regions_png"] = str(rpng)
    print(f"  [Regions] partitioned into {len(ctx.region_partition_meta)} regions → {rpng.name}")
    return ctx
