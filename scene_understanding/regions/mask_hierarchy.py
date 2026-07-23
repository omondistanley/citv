"""Mask containment hierarchy over SAM2 / region object rows (mutates objects in place)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def mask_area(mask: Any) -> int:
    if mask is None:
        return 0
    return int(np.sum(np.asarray(mask) > 0))


def build_mask_hierarchy(
    objects_3d: List[Dict[str, Any]],
    *,
    hierarchy_enable_region_region_edges: bool = False,
    hierarchy_region_region_containment_min: float = 0.97,
) -> Dict[str, Any]:
    edges: List[Dict[str, Any]] = []
    parent_for: Dict[str, str] = {}
    child_lists: Dict[str, List[str]] = {}
    edge_scores: Dict[Tuple[str, str], float] = {}

    for child in objects_3d:
        child_mask = child.get("_sam2_mask_array")
        child_area = mask_area(child_mask)
        if child_area <= 0:
            continue

        best_parent = None
        best_score = 0.0
        best_edge: Optional[Dict[str, Any]] = None
        child_id = str(child.get("id"))

        for parent in objects_3d:
            parent_id = str(parent.get("id"))
            if parent_id == child_id:
                continue
            parent_mask = parent.get("_sam2_mask_array")
            parent_area = mask_area(parent_mask)
            if parent_area <= int(child_area * 1.1):
                continue
            if parent_mask is None:
                continue

            parent_bin = np.asarray(parent_mask) > 0
            child_bin = np.asarray(child_mask) > 0
            if parent_bin.shape != child_bin.shape:
                parent_bin = cv2.resize(
                    parent_bin.astype(np.uint8),
                    (child_bin.shape[1], child_bin.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

            inter = int(np.logical_and(parent_bin, child_bin).sum())
            if inter <= 0:
                continue
            contain_ratio = inter / float(max(child_area, 1))
            parent_cover = inter / float(max(parent_area, 1))
            if contain_ratio < 0.92 or parent_cover < 0.03:
                continue

            _parent_kind = str(parent.get("entity_kind", "object"))
            _child_kind = str(child.get("entity_kind", "object"))
            _both_regions = _parent_kind == "region" and _child_kind == "region"
            if _both_regions:
                if (
                    not hierarchy_enable_region_region_edges
                    or contain_ratio < hierarchy_region_region_containment_min
                ):
                    continue

            score = contain_ratio + min(parent_cover, 0.25)
            if score > best_score:
                best_parent = parent
                best_score = score
                if _parent_kind == "region" or _child_kind == "region":
                    _edge_type = "region_object_membership"
                else:
                    _edge_type = "object_object_part"
                best_edge = {
                    "parent_object_id": parent_id,
                    "child_object_id": child_id,
                    "parent_mask_index": parent.get("sam2_mask_index"),
                    "child_mask_index": child.get("sam2_mask_index"),
                    "containment_ratio": round(contain_ratio, 4),
                    "parent_overlap_ratio": round(parent_cover, 4),
                    "edge_type": _edge_type,
                }

        if best_parent is not None and best_edge is not None:
            parent_id = str(best_parent.get("id"))
            parent_for[child_id] = parent_id
            child_lists.setdefault(parent_id, []).append(child_id)
            edge_scores[(parent_id, child_id)] = best_score
            edges.append(best_edge)

    root_object_ids = [str(obj.get("id")) for obj in objects_3d if str(obj.get("id")) not in parent_for]

    def _containment_depth(obj_id: str, _seen: Optional[frozenset] = None) -> int:
        """Distance from ``obj_id`` up to its root ancestor (0 = root itself).
        Cycle-guarded: ``parent_for`` is built from best-scoring single-parent
        edges so cycles shouldn't occur, but a defensive guard is cheap and
        this is the kind of bug that's silent (infinite loop) if it ever does."""
        seen = _seen or frozenset()
        parent_id = parent_for.get(obj_id)
        if parent_id is None or obj_id in seen:
            return 0
        return 1 + _containment_depth(parent_id, seen | {obj_id})

    for obj in objects_3d:
        obj_id = str(obj.get("id"))
        child_ids = child_lists.get(obj_id, [])
        parent_id = parent_for.get(obj_id)
        obj["parent_object_id"] = parent_id
        obj["child_object_ids"] = child_ids
        obj["containment_depth"] = _containment_depth(obj_id)
        obj["part_mask_ids"] = [
            child.get("sam2_mask_index")
            for child in objects_3d
            if str(child.get("id")) in child_ids and child.get("sam2_mask_index") is not None
        ]

    return {
        "edges": edges,
        "root_object_ids": root_object_ids,
        "num_edges": len(edges),
        "max_containment_depth": max((int(obj.get("containment_depth", 0)) for obj in objects_3d), default=0),
    }
