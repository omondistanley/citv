"""Relations stage: Pix2SG (and package RelationsPipeline) plus 3D-aware refinement."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from ..pipeline_context import PipelineContext

# Predicates that are computed purely from screen-space ordering / overlap and
# carry little physical meaning when 3D evidence is available. See plan §1.4
# and `path_updates.md` §3 Pitfall 4.
_2D_ONLY_PREDICATES = {"overlapping", "left_of", "right_of", "above", "below"}


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Predict relation triplets from staged objects."""
    rel_pipe = pipeline._get_relations_pipeline()
    objects = list(ctx.extra.get("objects", []))
    print(f"  [Relations] predicting relations for {len(objects)} objects...")
    relations = rel_pipe.predict_relations(
        image=ctx.img_bgr,
        image_stem=ctx.stem,
        detections=objects,
        iou_func=pipeline._bbox_iou_xyxy,
    )
    relations = list(relations)

    refined, demote_count, promote_count = _refine_relations(
        relations,
        objects,
        metric_depth=ctx.metric_depth,
        mask_hierarchy=ctx.mask_hierarchy,
    )
    ctx.relations = refined

    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    out_path = staged_dir / f"{ctx.stem}_relations.json"
    out_path.write_text(json.dumps(ctx.relations, indent=2, default=str))

    print(
        f"  [Relations] {len(ctx.relations)} relation triplets "
        f"(demoted_2d={demote_count}, promoted_from_hierarchy={promote_count})"
    )
    return ctx


def _depth_at(arr: Optional[np.ndarray], uv: Optional[List[float]]) -> Optional[float]:
    if arr is None or uv is None or len(uv) < 2:
        return None
    try:
        x = int(round(float(uv[0])))
        y = int(round(float(uv[1])))
    except (TypeError, ValueError):
        return None
    h, w = arr.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return None
    val = float(arr[y, x])
    if not np.isfinite(val) or val <= 0.0:
        return None
    return val


def _has_depth_corroboration(
    rel: Dict[str, Any],
    objects_by_id: Dict[str, Dict[str, Any]],
    metric_depth: Optional[np.ndarray],
) -> bool:
    if metric_depth is None:
        return False
    sub = objects_by_id.get(str(rel.get("sub_id") or rel.get("subject_id") or ""))
    obj = objects_by_id.get(str(rel.get("obj_id") or rel.get("object_id") or ""))
    if not sub or not obj:
        return False
    z_s = _depth_at(metric_depth, sub.get("mask_centroid_2d"))
    z_o = _depth_at(metric_depth, obj.get("mask_centroid_2d"))
    if z_s is None or z_o is None:
        return False
    pred = str(rel.get("pred") or rel.get("predicate") or "")
    delta = z_s - z_o
    if pred == "behind" and delta > 0.10:
        return True
    if pred == "in_front_of" and delta < -0.10:
        return True
    # left_of/right_of/above/below have no depth confirmation but the depth
    # signal does *not* contradict them either; we return False so they are
    # demoted unless intra_region.
    return False


def _refine_relations(
    relations: List[Dict[str, Any]],
    objects: List[Dict[str, Any]],
    *,
    metric_depth: Optional[np.ndarray],
    mask_hierarchy: Optional[Dict[str, Any]],
) -> tuple:
    """Demote 2D-only predicates when depth available; promote contains/supports.

    Returns ``(refined_relations, demote_count, promote_count)``.
    """
    objects_by_id: Dict[str, Dict[str, Any]] = {
        str(o.get("id", "")): o for o in objects if isinstance(o, dict) and o.get("id")
    }
    label_by_id: Dict[str, str] = {
        oid: str(o.get("canonical_label") or o.get("label", ""))
        for oid, o in objects_by_id.items()
    }

    refined: List[Dict[str, Any]] = []
    demote_count = 0
    seen_pairs: set = set()
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        pred = str(rel.get("pred") or rel.get("predicate") or "")
        tier = str(rel.get("relation_tier", ""))
        sid = str(rel.get("sub_id") or rel.get("subject_id") or "")
        oid = str(rel.get("obj_id") or rel.get("object_id") or "")
        seen_pairs.add((sid, oid, pred))

        # Keep all relations when no 3D substitute is computable for this image.
        if metric_depth is None:
            refined.append(rel)
            continue

        if pred in _2D_ONLY_PREDICATES and tier != "intra_region":
            if not _has_depth_corroboration(rel, objects_by_id, metric_depth):
                # Demote: keep the record but flag and downscore.
                rec = dict(rel)
                rec["demoted_reason"] = "screen_space_predicate_without_3d_support"
                rec["score"] = float(rec.get("score", 0.0)) * 0.4
                rec["relation_tier"] = "demoted_2d"
                refined.append(rec)
                demote_count += 1
                continue
        refined.append(rel)

    # Promote contains/on/supports from mask hierarchy edges when present.
    promote_count = 0
    edges = []
    if isinstance(mask_hierarchy, dict):
        edges = list(mask_hierarchy.get("edges") or [])
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        parent = str(edge.get("parent_object_id") or "")
        child = str(edge.get("child_object_id") or "")
        ratio = float(edge.get("containment_ratio") or 0.0)
        if not parent or not child:
            continue
        # Only emit when the child is meaningfully contained by the parent;
        # ratio is the fraction of the child mask inside the parent mask.
        if ratio < 0.6:
            continue
        triple = (parent, child, "contains")
        if triple in seen_pairs:
            continue
        seen_pairs.add(triple)
        refined.append({
            "sub": label_by_id.get(parent, parent),
            "pred": "contains",
            "obj": label_by_id.get(child, child),
            "sub_id": parent,
            "obj_id": child,
            "score": float(min(1.0, 0.6 + ratio * 0.4)),
            "relation_tier": "mask_hierarchy",
            "source": "mask_hierarchy.containment_ratio",
        })
        promote_count += 1
        # Reciprocal "on" / "supports": the parent supports the child only if
        # the child sits *physically below* the parent's centre line in image
        # space (a coarse but safe heuristic without surface normals).
        sub_obj = objects_by_id.get(parent)
        ch_obj = objects_by_id.get(child)
        if sub_obj and ch_obj:
            sc = sub_obj.get("mask_centroid_2d") or []
            cc = ch_obj.get("mask_centroid_2d") or []
            if (
                len(sc) >= 2
                and len(cc) >= 2
                and float(cc[1]) < float(sc[1])
            ):
                triple_on = (child, parent, "on")
                if triple_on not in seen_pairs:
                    seen_pairs.add(triple_on)
                    refined.append({
                        "sub": label_by_id.get(child, child),
                        "pred": "on",
                        "obj": label_by_id.get(parent, parent),
                        "sub_id": child,
                        "obj_id": parent,
                        "score": float(min(1.0, 0.55 + ratio * 0.35)),
                        "relation_tier": "mask_hierarchy",
                        "source": "mask_hierarchy.containment_ratio+geometric_above",
                    })
                    promote_count += 1
    return refined, demote_count, promote_count
