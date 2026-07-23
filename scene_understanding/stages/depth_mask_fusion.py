"""Depth-mask fusion export: pairs each labelled object with its mask and
per-mask depth statistics into ``{stem}_depth_mask_A.json``, mirroring the
monolith's ``_build_depth_mask_json``/mode-A ("detection-first") output.

Mode A is fully derivable from the staged pipeline today: ``stages/labelling.py``
already attaches ``_sam2_mask_array`` (the matched mask) and depth stats to
every object in ``ctx.extra["objects"]``, so mode A here is a re-serialization
of already-computed data, not new matching logic.

Mode B ("mask-first": IoU-matching *unlabelled* SAM2 AMG masks against
detections, to surface objects the detector missed) is intentionally NOT
implemented here -- it needs the raw AMG mask set, and the staged segmentation
stage (``stages/segmentation.py``) currently calls
``seg_pipe.generate(..., use_secondary=False)``, so no AMG masks exist in
``ctx`` to match against. Wiring mode B is a segmentation-stage change
(exposing AMG masks into the context), not something this stage can do on
its own -- so it's skipped with an explicit reason in the output rather than
silently omitted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..pipeline_context import PipelineContext


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _mapping_image(metric_depth: np.ndarray, objects: List[Dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Depth colormap restricted to matched-object pixels; everything else black."""
    combined_mask = np.zeros((h, w), dtype=bool)
    for obj in objects:
        m = obj.get("_sam2_mask_array")
        if m is None:
            continue
        m = np.asarray(m)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask |= (m.astype(bool))
    d = np.asarray(metric_depth, dtype=np.float32)
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if not combined_mask.any():
        return vis
    d_min, d_max = float(d[combined_mask].min()), float(d[combined_mask].max())
    if d_max - d_min < 1e-8:
        d_max = d_min + 1.0
    norm = np.clip(((d - d_min) / (d_max - d_min) * 255.0), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    vis[combined_mask] = colored[combined_mask]
    return vis


def _object_entry(obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "id": obj.get("id", f"obj_{index}"),
        "entity_kind": "object",
        "label": obj.get("label", "unknown"),
        "bbox": obj.get("bbox", []),
        "source_model": obj.get("segmentor", ""),
        "segmentor": obj.get("segmentor", ""),
        "depth_stats": obj.get("depth_stats", {}),
        "coordinates_3d_from_mask": obj.get("coordinates_3d", {}),
        "mask_centroid_2d": obj.get("mask_centroid_2d", []),
    }


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    objects: List[Dict[str, Any]] = ctx.extra.get("objects", [])
    if not objects or ctx.metric_depth is None:
        return ctx

    cfg = getattr(pipeline, "config", None)
    h, w = ctx.height, ctx.width
    depth = np.asarray(ctx.metric_depth, dtype=np.float32)
    finite = np.isfinite(depth)

    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = staged_dir / f"{ctx.stem}_depth_mask_mapping.png"
    cv2.imwrite(str(mapping_path), _mapping_image(depth, objects, h, w))

    json_objects = [_object_entry(obj, i) for i, obj in enumerate(objects)]
    payload = {
        "metadata": {
            "image_path": str(ctx.image_path),
            "image_stem": ctx.stem,
            "timestamp": ctx.timestamp,
            "image_size": [w, h],
            "matching_mode": "A",
        },
        "depth": {
            "depth_map_path": f"depth/{ctx.stem}_depth_metric.npy",
            "depth_map_image_path": f"depth/{ctx.stem}_depth_colormap.png",
            "depth_min": float(depth[finite].min()) if finite.any() else 0.0,
            "depth_max": float(depth[finite].max()) if finite.any() else 0.0,
            "depth_mean": float(depth[finite].mean()) if finite.any() else 0.0,
        },
        "segmentation": {
            "model": "GroundedSAM2",
            "mode": "detection_first",
            "segmentation_map_image_path": f"scene_graph/staged/{ctx.stem}_segmentation.png",
            "num_auto_masks": len(objects),
            "match_strategy": "detection_first_mask_attached_at_labelling_stage",
        },
        "depth_mask": {
            "mapping_image_path": mapping_path.name,
            "objects": json_objects,
        },
        "mode_b": {
            "attempted": False,
            "reason": "staged segmentation stage runs with use_secondary=False; no raw AMG mask set is available in ctx to IoU-match against detections",
        },
    }
    out_path = staged_dir / f"{ctx.stem}_depth_mask_A.json"
    out_path.write_text(json.dumps(payload, indent=2))
    ctx.path_exports["depth_mask_a_json"] = str(out_path)
    print(f"  [DepthMaskFusion] wrote {len(json_objects)} matched objects -> {out_path.name}")
    return ctx
