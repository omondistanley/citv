"""Object hierarchy export: mask-containment parent/child edges (shared
``regions/mask_hierarchy.py``, used by both the monolith and this stage) plus
relation-derived occlusion/containment bookkeeping (``contains``/
``contained_by``/``occludes``/``occluded_by``), written as
``{stem}_mask_hierarchy.json`` and ``{stem}_mask_hierarchy_detailed.json``
(the detailed variant additionally carries each node's ``containment_depth``
and the human-readable edge/occlusion summary used by the PNG overlay).

This closes the staged-pipeline gap flagged in SCENE_GRAPH_DEEP_DIVE.md §8
item 2: the hierarchy *algorithm* already existed (``regions/mask_hierarchy.py``,
shared with the monolith) but was never wired into the staged stage chain --
``ctx.mask_hierarchy`` stayed ``None`` and nothing populated it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..pipeline_context import PipelineContext
from ..regions.mask_hierarchy import build_mask_hierarchy
from scene_understanding.pathing.goal_anchors import mask_centroid_px

_EDGE_LABELS = {
    "object_object_part": "part of",
    "region_object_membership": "member of",
    "region_region_containment": "inside region",
}
_EDGE_COLORS_BGR = {
    "object_object_part": (255, 0, 255),
    "region_object_membership": (0, 165, 255),
    "region_region_containment": (255, 200, 0),
}


def apply_relation_derived_fields(objects: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
    """Mutates ``objects`` in place, adding ``contains``/``contained_by``
    (seeded from the mask-containment parent/child edges, then extended by
    ``contains``/``inside_of``/``around`` relation predicates) and
    ``occludes``/``occluded_by`` (purely relation-derived, from
    ``in_front_of``/``behind`` predicates -- mask containment alone can't
    tell you occlusion order, only spatial nesting)."""
    id_to_obj = {str(obj.get("id")): obj for obj in objects}
    for obj in objects:
        obj.setdefault("contains", list(obj.get("child_object_ids", [])))
        parent_id = obj.get("parent_object_id")
        obj.setdefault("contained_by", [parent_id] if parent_id else [])
        obj.setdefault("occludes", [])
        obj.setdefault("occluded_by", [])

    for rel in relations:
        sub_id = str(rel.get("subject_id"))
        obj_id = rel.get("object_id")
        if obj_id is None:
            continue
        obj_id = str(obj_id)
        subject, target = id_to_obj.get(sub_id), id_to_obj.get(obj_id)
        if subject is None or target is None:
            continue
        predicate = str(rel.get("predicate", ""))
        if predicate in {"contains", "around"}:
            if obj_id not in subject["contains"]:
                subject["contains"].append(obj_id)
            if sub_id not in target["contained_by"]:
                target["contained_by"].append(sub_id)
        elif predicate in {"inside_of", "inside"}:
            if obj_id not in subject["contained_by"]:
                subject["contained_by"].append(obj_id)
            if sub_id not in target["contains"]:
                target["contains"].append(sub_id)
        elif predicate == "in_front_of":
            if obj_id not in subject["occludes"]:
                subject["occludes"].append(obj_id)
            if sub_id not in target["occluded_by"]:
                target["occluded_by"].append(sub_id)
        elif predicate == "behind":
            if obj_id not in subject["occluded_by"]:
                subject["occluded_by"].append(obj_id)
            if sub_id not in target["occludes"]:
                target["occludes"].append(sub_id)


def _draw_hierarchy_png(img_bgr: np.ndarray, objects: List[Dict[str, Any]], hierarchy: Dict[str, Any], h: int, w: int) -> np.ndarray:
    canvas = img_bgr.copy()
    id_to_obj = {str(obj.get("id")): obj for obj in objects}
    centroid_cache = {oid: mask_centroid_px(obj, h, w) for oid, obj in id_to_obj.items()}

    for oid, obj in id_to_obj.items():
        c = centroid_cache.get(oid)
        if c is None:
            continue
        cx, cy = int(c[0]), int(c[1])
        cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1)
        label = str(obj.get("canonical_name") or obj.get("label") or oid)
        cv2.putText(canvas, label, (cx + 8, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    for edge in hierarchy.get("edges", []):
        p = centroid_cache.get(str(edge.get("parent_object_id")))
        c = centroid_cache.get(str(edge.get("child_object_id")))
        if p is None or c is None:
            continue
        etype = str(edge.get("edge_type", "object_object_part"))
        color = _EDGE_COLORS_BGR.get(etype, (255, 0, 255))
        cv2.arrowedLine(canvas, (int(p[0]), int(p[1])), (int(c[0]), int(c[1])), color, 2, cv2.LINE_AA, tipLength=0.10)
        ratio = edge.get("containment_ratio")
        text = f"{_EDGE_LABELS.get(etype, etype)} ({ratio:.2f})" if isinstance(ratio, float) else _EDGE_LABELS.get(etype, etype)
        mx, my = (int(p[0]) + int(c[0])) // 2, (int(p[1]) + int(c[1])) // 2
        cv2.putText(canvas, text, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    y = 18
    for etype, color in _EDGE_COLORS_BGR.items():
        cv2.putText(canvas, _EDGE_LABELS[etype], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        y += 16
    return canvas


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    objects: List[Dict[str, Any]] = ctx.extra.get("objects", [])
    if not objects:
        return ctx

    cfg = getattr(pipeline, "config", None)
    hierarchy = build_mask_hierarchy(
        objects,
        hierarchy_enable_region_region_edges=bool(getattr(cfg, "hierarchy_enable_region_region_edges", False)) if cfg else False,
        hierarchy_region_region_containment_min=float(getattr(cfg, "hierarchy_region_region_containment_min", 0.97)) if cfg else 0.97,
    )
    apply_relation_derived_fields(objects, ctx.relations)

    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    stem = ctx.stem

    summary_payload = {"edges": hierarchy["edges"], "root_object_ids": hierarchy["root_object_ids"], "num_edges": hierarchy["num_edges"]}
    (staged_dir / f"{stem}_mask_hierarchy.json").write_text(json.dumps(summary_payload, indent=2))

    detailed_nodes = [
        {
            "id": obj.get("id"),
            "label": obj.get("label"),
            "parent_object_id": obj.get("parent_object_id"),
            "child_object_ids": obj.get("child_object_ids", []),
            "containment_depth": obj.get("containment_depth", 0),
            "part_mask_ids": obj.get("part_mask_ids", []),
            "contains": obj.get("contains", []),
            "contained_by": obj.get("contained_by", []),
            "occludes": obj.get("occludes", []),
            "occluded_by": obj.get("occluded_by", []),
        }
        for obj in objects
    ]
    detailed_payload = {
        **summary_payload,
        "max_containment_depth": hierarchy.get("max_containment_depth", 0),
        "nodes": detailed_nodes,
    }
    detailed_path = staged_dir / f"{stem}_mask_hierarchy_detailed.json"
    detailed_path.write_text(json.dumps(detailed_payload, indent=2))

    png_path = staged_dir / f"{stem}_mask_hierarchy.png"
    cv2.imwrite(str(png_path), _draw_hierarchy_png(ctx.img_bgr, objects, hierarchy, ctx.height, ctx.width))

    ctx.mask_hierarchy = summary_payload
    ctx.path_exports["mask_hierarchy_json"] = str(staged_dir / f"{stem}_mask_hierarchy.json")
    ctx.path_exports["mask_hierarchy_detailed_json"] = str(detailed_path)
    print(f"  [HierarchyExport] {hierarchy['num_edges']} containment edges, max depth {hierarchy.get('max_containment_depth', 0)} -> {detailed_path.name}")
    return ctx
