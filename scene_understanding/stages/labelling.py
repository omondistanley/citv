"""Labelling + per-mask depth geometry for staged pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..pipeline_context import PipelineContext


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Attach labels and depth-backed geometry to each detection."""
    if ctx.metric_depth is None:
        return ctx
    cfg = getattr(pipeline, "config", None)
    label_mode = str(getattr(cfg, "post_sam_label_mode", "mask_chain") if cfg else "mask_chain").strip().lower()
    use_mask_chain = label_mode == "mask_chain" and hasattr(pipeline, "_label_mask")
    labellers = None
    if label_mode == "gdino_only":
        pass
    elif use_mask_chain:
        pipeline._load_labellers()
    else:
        labellers = pipeline._get_labeling_pipeline()
    objects: List[Dict[str, Any]] = []
    lm = ctx.region_label_map
    try:
        from ..regions.partitioner import majority_region_index as _majority_ri
    except Exception:
        _majority_ri = None  # type: ignore[assignment]
    for idx, det in enumerate(ctx.detections):
        seg = det.get("segmentation")
        if seg is None:
            continue
        mask_bin = np.asarray(seg) > 0
        if mask_bin.sum() == 0:
            continue
        x, y, w, h = [int(v) for v in det.get("bbox", [0, 0, ctx.width, ctx.height])]
        x1, y1 = max(0, x), max(0, y)
        x2 = min(ctx.width, x + w)
        y2 = min(ctx.height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        _ri = 0
        if lm is not None and _majority_ri is not None:
            try:
                _ri = int(_majority_ri(mask_bin, lm))
            except Exception:
                _ri = 0
        amg_entry = dict(det)
        if "bbox" not in amg_entry or amg_entry.get("bbox") is None:
            amg_entry["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        if label_mode == "gdino_only":
            label_val = str(det.get("label", "object"))
            conf_val = float(det.get("gdino_conf", det.get("predicted_iou", 0.0)))
        elif use_mask_chain:
            det_out = pipeline._label_mask(ctx.img_bgr, mask_bin, amg_entry, label_map=lm, region_index=_ri)
            label_val = str(det_out.get("label", det.get("label", "object")))
            conf_val = float(det_out.get("conf", det.get("gdino_conf", 0.0)))
        else:
            assert labellers is not None
            crop = ctx.img_bgr[y1:y2, x1:x2].copy()
            label_payload = labellers.label_crop(crop)
            label_val = str(label_payload.get("label", det.get("label", "object")))
            conf_val = float(label_payload.get("conf", det.get("gdino_conf", 0.0)))
        depth_stats, coords_3d, centroid = pipeline._mask_depth_stats_and_3d(
            metric_depth=ctx.metric_depth,
            K=ctx.intrinsics,
            mask=mask_bin,
            detection=det,
            use_erosion=True,
        )
        _oid = f"obj_{idx}"
        objects.append(
            {
                "id": _oid,
                "graph_id": _oid,
                "label": label_val,
                "conf": conf_val,
                "bbox": [x1, y1, x2, y2],
                "segmentor": str(det.get("source_model", "unknown")),
                "depth_stats": depth_stats,
                "coordinates_3d": coords_3d,
                "mask_centroid_2d": centroid,
                "_sam2_mask_array": mask_bin,
            }
        )
    ctx.extra["objects"] = objects
    return ctx
