"""Stage: visualization exports for the staged pipeline.

Writes all per-image PNGs that the monolithic pipeline produces:
  - {stem}_segmentation.png        — colored mask per object
  - {stem}_overlay.png             — semi-transparent tint over image
  - {stem}_3d_viz.png              — bounding boxes + relation edges
  - {stem}_relations_map.png       — relation arrows (all entities)
  - {stem}_relations_map_objects.png  — relation arrows (objects only)
  - {stem}_relations_map_regions.png  — relation arrows (regions only)
  - {stem}_mask_hierarchy.png      — containment graph
  - {stem}_region_segmentation.png — colored region map with geometry
  - {stem}_region_sam2_style_segmentation.png
  - {stem}_region_tinted_overlay.png

All writes are guarded so a single failure does not abort the stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..pipeline_context import PipelineContext

# ---------------------------------------------------------------------------
# Deterministic per-index BGR colour palette (16 maximally-distinct hues).
# ---------------------------------------------------------------------------

_PALETTE_RGB = [
    (204,  51,  51), ( 51, 153, 255), ( 51, 204,  51), (255, 153,  51),
    (153,  51, 204), ( 51, 204, 204), (255,  51, 153), (204, 204,  51),
    ( 51,  51, 204), (204, 102,  51), (102, 204,  51), ( 51, 102, 204),
    (204,  51, 102), ( 51, 204, 102), (204, 153,  51), (102,  51, 204),
]

_PRED_COL_BGR: Dict[str, Tuple[int, int, int]] = {
    "in_front_of": (  0, 230,   0),
    "behind":      (  0,   0, 230),
    "left_of":     (230, 200,   0),
    "right_of":    (  0, 140, 255),
    "above":       (  0, 240, 240),
    "below":       (255,   0, 210),
    "adjacent_to": (220, 220, 220),
    "overlapping": (255, 200,   0),
    "contains":    (200,   0, 200),
    "inside_of":   (170,   0, 170),
    "on":          (  0, 215, 255),
    "holds":       (  0, 200, 120),
}

_HIER_EDGE_COL: Dict[str, Tuple[int, int, int]] = {
    "object_object_part":        (255,   0, 255),
    "region_object_membership":  (  0, 165, 255),
    "region_region_containment": (255, 200,   0),
    "region_depth_order":        (120, 255, 255),
}

_HIER_EDGE_LABEL: Dict[str, str] = {
    "object_object_part":        "part of",
    "region_object_membership":  "member of",
    "region_region_containment": "inside region",
    "region_depth_order":        "depth order",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_colour_bgr(idx: int) -> Tuple[int, int, int]:
    r, g, b = _PALETTE_RGB[idx % len(_PALETTE_RGB)]
    return b, g, r


def _obj_centroid(obj: Dict[str, Any]) -> Tuple[int, int]:
    mc = obj.get("mask_centroid_2d")
    if mc and len(mc) >= 2:
        return int(mc[0]), int(mc[1])
    bbox = obj.get("bbox", [0, 0, 0, 0])
    return (int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2


def _draw_label(
    canvas: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    mask_area: int = 0,
) -> None:
    if not text:
        return
    scale = float(np.clip(np.sqrt(max(mask_area, 1)) / 250.0, 0.35, 0.70))
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 3
    x0 = max(0, cx - tw // 2 - pad)
    y0 = max(0, cy - th - pad)
    x1 = min(canvas.shape[1] - 1, cx + tw // 2 + pad)
    y1 = min(canvas.shape[0] - 1, cy + baseline + pad)
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    if roi.size > 0:
        roi[:] = roi * 0.35
        canvas[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    cv2.putText(canvas, text, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _build_object_visual_cache(
    objects: List[Dict[str, Any]],
    h: int,
    w: int,
) -> Optional[Dict[str, Any]]:
    """Build label_map + palette_lut + contours shared across segmentation/overlay.

    Paints far→near so nearer objects win on overlap (matches monolith ordering).
    Returns None when no objects have usable masks.
    """
    def _z(o: Dict[str, Any]) -> float:
        try:
            return float((o.get("coordinates_3d") or {}).get("z", 0.0) or 0.0)
        except Exception:
            return 0.0

    order = sorted(range(len(objects)), key=lambda i: -_z(objects[i]))
    label_map = np.zeros((h, w), dtype=np.int32)
    resized: Dict[int, np.ndarray] = {}

    for idx in order:
        m = objects[idx].get("_sam2_mask_array")
        if m is None:
            continue
        arr = np.asarray(m, dtype=np.uint8)
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        bin_mask = arr > 0
        resized[idx] = bin_mask
        label_map[bin_mask] = idx + 1

    if not resized:
        return None

    max_idx = max(resized.keys()) + 1
    palette_lut = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for idx in resized:
        palette_lut[idx + 1] = _mask_colour_bgr(idx)

    contours: Dict[int, Any] = {}
    for idx, bin_mask in resized.items():
        cnts, _ = cv2.findContours(bin_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours[idx] = cnts

    return {
        "h": h, "w": w,
        "label_map": label_map,
        "palette_lut": palette_lut,
        "binary": label_map > 0,
        "contours": contours,
        "masks": resized,
    }


def _rel_id(rel: Dict[str, Any], key_a: str, key_b: str) -> str:
    v = rel.get(key_a) or rel.get(key_b, "")
    return str(v)


# ---------------------------------------------------------------------------
# Object visualization writers
# ---------------------------------------------------------------------------

def _write_segmentation(
    objects: List[Dict[str, Any]],
    h: int, w: int,
    staged_dir: Path,
    stem: str,
) -> Optional[str]:
    try:
        cache = _build_object_visual_cache(objects, h, w)
        if cache is None:
            return None
        canvas = cache["palette_lut"][cache["label_map"]]
        for cnts in cache["contours"].values():
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 2)
        for i, obj in enumerate(objects):
            if i not in cache["masks"]:
                continue
            label = str(obj.get("label", "object"))
            cx, cy = _obj_centroid(obj)
            area = int(np.sum(cache["masks"][i]))
            _draw_label(canvas, label, cx, cy, area)
        out = staged_dir / f"{stem}_segmentation.png"
        cv2.imwrite(str(out), canvas)
        return f"scene_graph/staged/{stem}_segmentation.png"
    except Exception as exc:
        print(f"  [VizExport] segmentation skipped: {exc}")
        return None


def _write_tinted_overlay(
    objects: List[Dict[str, Any]],
    img_rgb: np.ndarray,
    h: int, w: int,
    staged_dir: Path,
    stem: str,
    alpha: float = 0.45,
) -> Optional[str]:
    try:
        cache = _build_object_visual_cache(objects, h, w)
        if cache is None:
            return None
        src = img_rgb.astype(np.float32)
        if src.shape[:2] != (h, w):
            src = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        # palette_lut is BGR; convert to RGB for blending against img_rgb
        palette_rgb = cache["palette_lut"][:, ::-1].astype(np.float32)
        tint = palette_rgb[cache["label_map"]]
        binary = cache["binary"]
        src[binary] = src[binary] * (1.0 - alpha) + tint[binary] * alpha
        out_bgr = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for cnts in cache["contours"].values():
            cv2.drawContours(out_bgr, cnts, -1, (255, 255, 255), 2)
        for i, obj in enumerate(objects):
            if i not in cache["masks"]:
                continue
            label = str(obj.get("label", "object"))
            cx, cy = _obj_centroid(obj)
            area = int(np.sum(cache["masks"][i]))
            _draw_label(out_bgr, label, cx, cy, area)
        out = staged_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(out), out_bgr)
        return f"scene_graph/staged/{stem}_overlay.png"
    except Exception as exc:
        print(f"  [VizExport] overlay skipped: {exc}")
        return None


def _write_3d_viz(
    objects: List[Dict[str, Any]],
    img_bgr: np.ndarray,
    relations: List[Dict[str, Any]],
    staged_dir: Path,
    stem: str,
) -> Optional[str]:
    try:
        canvas = img_bgr.copy()
        id_to_obj = {str(o.get("id", "")): o for o in objects}
        for obj in objects:
            bbox = obj.get("bbox", [])
            if len(bbox) < 4:
                continue
            color = (0, 255, 0)
            label = str(obj.get("label", "object")) + " [M]"
            cv2.rectangle(canvas, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(canvas, label, (int(bbox[0]), max(0, int(bbox[1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            mc = obj.get("mask_centroid_2d")
            if mc and len(mc) >= 2:
                cv2.circle(canvas, (int(mc[0]), int(mc[1])), 3, color, -1)
        for rel in relations:
            sub_id = _rel_id(rel, "sub_id", "subject_id")
            obj_id = _rel_id(rel, "obj_id", "object_id")
            src = id_to_obj.get(sub_id)
            tgt = id_to_obj.get(obj_id)
            if src is None or tgt is None:
                continue
            cx_a, cy_a = _obj_centroid(src)
            cx_b, cy_b = _obj_centroid(tgt)
            cv2.line(canvas, (cx_a, cy_a), (cx_b, cy_b), (0, 255, 255), 1)
            pred = str(rel.get("pred", rel.get("predicate", "")))
            mid = ((cx_a + cx_b) // 2, (cy_a + cy_b) // 2)
            cv2.putText(canvas, pred, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        out = staged_dir / f"{stem}_3d_viz.png"
        cv2.imwrite(str(out), canvas)
        return f"scene_graph/staged/{stem}_3d_viz.png"
    except Exception as exc:
        print(f"  [VizExport] 3d_viz skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# Relations map
# ---------------------------------------------------------------------------

def _write_relations_map(
    objects: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    img_bgr: np.ndarray,
    regions: List[Dict[str, Any]],
    staged_dir: Path,
    stem: str,
    view: str = "all",
) -> Optional[str]:
    suffix = {"all": "", "objects_only": "_objects", "regions_only": "_regions"}[view]
    fname = f"{stem}_relations_map{suffix}.png"
    try:
        canvas = img_bgr.copy()
        h, w = canvas.shape[:2]
        id_to_obj: Dict[str, Dict[str, Any]] = {str(o.get("id", "")): o for o in objects}
        rid_to_region: Dict[str, Dict[str, Any]] = {str(r.get("id", "")): r for r in regions}
        region_ids = set(rid_to_region.keys()) - {""}
        obj_ids = set(id_to_obj.keys()) - {""}

        occupied: List[Tuple[int, int, int, int]] = []

        def _overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        def _place_label(bx: int, by: int, text: str, scale: float = 0.4) -> Tuple[int, int]:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            for tx, ty in [(bx + 8, by - 8), (bx + 8, by + 14),
                           (bx - tw - 8, by - 8), (bx - tw - 8, by + 14)]:
                box = (max(0, tx - 2), max(0, ty - th - 2),
                       min(w - 1, tx + tw + 2), min(h - 1, ty + 2))
                if not any(_overlap(box, o) for o in occupied):
                    occupied.append(box)
                    return tx, ty
            occupied.append((max(0, bx - 2), max(0, by - th - 2),
                             min(w - 1, bx + tw + 2), min(h - 1, by + 2)))
            return bx, by

        # Draw entity anchors
        if view in ("all", "objects_only"):
            for obj in objects:
                name = str(obj.get("label", "object"))
                ox, oy = _obj_centroid(obj)
                ox = int(min(max(0, ox), w - 1))
                oy = int(min(max(0, oy), h - 1))
                cv2.circle(canvas, (ox, oy), 6, (0, 220, 80), -1)
                cv2.circle(canvas, (ox, oy), 6, (0, 0, 0), 1)
                tx, ty = _place_label(ox, oy, name, 0.44)
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(canvas, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                            (255, 255, 255), 1, cv2.LINE_AA)

        if view in ("all", "regions_only"):
            for r in regions:
                rid = str(r.get("id", ""))
                if not rid:
                    continue
                sem = str(r.get("semantic_label", "") or r.get("type", "region")).strip() or "region"
                c = r.get("centroid_2d_px") or [w // 2, h // 2]
                ox = int(min(max(0, float(c[0])), w - 1))
                oy = int(min(max(0, float(c[1])), h - 1))
                cv2.circle(canvas, (ox, oy), 8, (0, 140, 255), -1)
                cv2.circle(canvas, (ox, oy), 8, (0, 0, 0), 1)
                tx, ty = _place_label(ox, oy, sem, 0.44)
                (tw, th), _ = cv2.getTextSize(sem, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (30, 30, 30), -1)
                cv2.putText(canvas, sem, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                            (255, 200, 100), 1, cv2.LINE_AA)

        # Deduplicate and draw relation arrows
        seen: set = set()
        for rel in relations:
            sub_id = _rel_id(rel, "sub_id", "subject_id")
            obj_id = _rel_id(rel, "obj_id", "object_id")
            pred = str(rel.get("pred", rel.get("predicate", "related_to")))
            key = (sub_id, pred, obj_id)
            if key in seen or not sub_id or not obj_id:
                continue
            seen.add(key)

            if view == "objects_only" and (sub_id in region_ids or obj_id in region_ids):
                continue
            if view == "regions_only":
                tier = str(rel.get("relation_tier", ""))
                if tier != "region_region" and (sub_id not in region_ids or obj_id not in region_ids):
                    continue

            # Resolve endpoints
            if sub_id in id_to_obj:
                ax, ay = _obj_centroid(id_to_obj[sub_id])
            elif sub_id in rid_to_region:
                c = rid_to_region[sub_id].get("centroid_2d_px") or [w // 2, h // 2]
                ax, ay = int(float(c[0])), int(float(c[1]))
            else:
                continue

            if obj_id in id_to_obj:
                bx, by = _obj_centroid(id_to_obj[obj_id])
            elif obj_id in rid_to_region:
                c = rid_to_region[obj_id].get("centroid_2d_px") or [w // 2, h // 2]
                bx, by = int(float(c[0])), int(float(c[1]))
            else:
                continue

            colour = _PRED_COL_BGR.get(pred, (200, 200, 200))
            cv2.arrowedLine(canvas, (ax, ay), (bx, by), colour, 2, cv2.LINE_AA, tipLength=0.12)
            mx, my = (ax + bx) // 2, (ay + by) // 2
            (tw, th), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(canvas, (mx - 2, my - th - 2), (mx + tw + 2, my + 2), (0, 0, 0), -1)
            cv2.putText(canvas, pred, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        colour, 1, cv2.LINE_AA)

        out = staged_dir / fname
        cv2.imwrite(str(out), canvas)
        return f"scene_graph/staged/{fname}"
    except Exception as exc:
        print(f"  [VizExport] {fname} skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# Mask hierarchy map
# ---------------------------------------------------------------------------

def _write_hierarchy_map(
    objects: List[Dict[str, Any]],
    img_bgr: np.ndarray,
    hierarchy: Dict[str, Any],
    regions: List[Dict[str, Any]],
    staged_dir: Path,
    stem: str,
) -> Optional[str]:
    try:
        canvas = img_bgr.copy()
        h, w = canvas.shape[:2]
        id_to_node: Dict[str, Dict[str, Any]] = {str(o.get("id", "")): o for o in objects}
        for r in regions:
            id_to_node[str(r.get("id", ""))] = r

        for nid, node in id_to_node.items():
            cx, cy = _obj_centroid(node)
            cx = int(min(max(0, cx), w - 1))
            cy = int(min(max(0, cy), h - 1))
            is_region = str(node.get("entity_kind", "object")) == "region"
            if is_region:
                cv2.drawMarker(canvas, (cx, cy), (0, 165, 255), cv2.MARKER_DIAMOND, 12, 2)
            else:
                cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)
                cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1)
            label = str(node.get("label", nid))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            lx, ly = cx + 8, cy - 4
            roi = canvas[max(0, ly - th - 2):min(h, ly + 3), max(0, lx - 2):min(w, lx + tw + 2)]
            if roi.size > 0:
                canvas[max(0, ly - th - 2):min(h, ly + 3),
                       max(0, lx - 2):min(w, lx + tw + 2)] = (
                    np.clip(roi.astype(np.float32) * 0.3, 0, 255).astype(np.uint8))
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)

        for edge in hierarchy.get("edges", []):
            parent = id_to_node.get(str(edge.get("parent_object_id", "")))
            child = id_to_node.get(str(edge.get("child_object_id", "")))
            if parent is None or child is None:
                continue
            px, py = _obj_centroid(parent)
            cx, cy = _obj_centroid(child)
            px, py = int(min(max(0, px), w - 1)), int(min(max(0, py), h - 1))
            cx, cy = int(min(max(0, cx), w - 1)), int(min(max(0, cy), h - 1))
            etype = str(edge.get("edge_type", "object_object_part"))
            colour = _HIER_EDGE_COL.get(etype, (255, 0, 255))
            cv2.arrowedLine(canvas, (px, py), (cx, cy), colour, 2, cv2.LINE_AA, tipLength=0.10)
            readable = _HIER_EDGE_LABEL.get(etype, etype)
            ratio = edge.get("containment_ratio", "")
            txt = f"{readable} ({ratio:.2f})" if isinstance(ratio, float) else readable
            mid_x, mid_y = (px + cx) // 2, (py + cy) // 2
            cv2.putText(canvas, txt, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        colour, 1, cv2.LINE_AA)

        # Legend
        y = 18
        for txt, col in _HIER_EDGE_COL.items():
            label_txt = _HIER_EDGE_LABEL.get(txt, txt)
            cv2.putText(canvas, label_txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        col, 1, cv2.LINE_AA)
            y += 16

        out = staged_dir / f"{stem}_mask_hierarchy.png"
        cv2.imwrite(str(out), canvas)
        return f"scene_graph/staged/{stem}_mask_hierarchy.png"
    except Exception as exc:
        print(f"  [VizExport] mask_hierarchy.png skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# Region visualization helpers
# ---------------------------------------------------------------------------

def _build_region_visual_objects(
    regions: List[Dict[str, Any]],
    lm: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build pseudo-objects from region metadata for shared segmentation/overlay writers."""
    arr = np.asarray(lm, dtype=np.int32)
    h, w = arr.shape[:2]
    rows: List[Dict[str, Any]] = []
    for r in regions:
        ridx = int(r.get("region_index", 0) or 0)
        if ridx <= 0:
            continue
        mask = arr == ridx
        if not np.any(mask):
            continue
        bx = r.get("bbox_px") or [0, 0, w - 1, h - 1]
        x1, y1, x2, y2 = [int(v) for v in bx[:4]]
        sem = str(r.get("semantic_label", "")).strip().lower()
        typ = str(r.get("type", "region"))
        base = sem if sem else typ
        oids = r.get("object_ids") or []
        tag = f"{base} ({len(oids)})" if oids else base
        cxy = r.get("centroid_2d_px") or [(x1 + x2) / 2, (y1 + y2) / 2]
        rows.append({
            "id": str(r.get("id", f"region_{ridx}")),
            "label": tag,
            "_sam2_mask_array": mask,
            "mask_centroid_2d": [float(cxy[0]), float(cxy[1])],
            "bbox": [x1, y1, x2, y2],
        })
    return rows


def _annotate_region_geometry(
    canvas: np.ndarray,
    regions: List[Dict[str, Any]],
    adjacency_edges: List[Dict[str, Any]],
) -> None:
    h, w = canvas.shape[:2]
    centroids: Dict[str, Tuple[int, int]] = {}
    for r in regions:
        rid = str(r.get("id", ""))
        c = r.get("centroid_2d_px") or [w // 2, h // 2]
        cx = int(min(max(0, float(c[0])), w - 1))
        cy = int(min(max(0, float(c[1])), h - 1))
        centroids[rid] = (cx, cy)
        cv2.circle(canvas, (cx, cy), 6, (80, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(canvas, rid, (cx + 7, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 255, 200), 1, cv2.LINE_AA)
    for edge in adjacency_edges:
        a = centroids.get(str(edge.get("subject", "")))
        b = centroids.get(str(edge.get("object", "")))
        if a and b:
            cv2.line(canvas, a, b, (0, 180, 255), 1, cv2.LINE_AA)


def _write_region_segmentation(
    reg_objs: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    lm: np.ndarray,
    adjacency_edges: List[Dict[str, Any]],
    staged_dir: Path,
    stem: str,
) -> Optional[str]:
    try:
        h, w = lm.shape[:2]
        cache = _build_object_visual_cache(reg_objs, h, w)
        if cache is None:
            return None
        canvas = cache["palette_lut"][cache["label_map"]].copy()
        for cnts in cache["contours"].values():
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
        _annotate_region_geometry(canvas, regions, adjacency_edges)
        out = staged_dir / f"{stem}_region_segmentation.png"
        cv2.imwrite(str(out), canvas)
        return f"scene_graph/staged/{stem}_region_segmentation.png"
    except Exception as exc:
        print(f"  [VizExport] region_segmentation skipped: {exc}")
        return None


def _write_region_sam2_segmentation(
    reg_objs: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    lm: np.ndarray,
    adjacency_edges: List[Dict[str, Any]],
    staged_dir: Path,
    stem: str,
) -> Optional[str]:
    try:
        h, w = lm.shape[:2]
        cache = _build_object_visual_cache(reg_objs, h, w)
        if cache is None:
            return None
        canvas = cache["palette_lut"][cache["label_map"]].copy()
        for cnts in cache["contours"].values():
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
        for i, obj in enumerate(reg_objs):
            if i not in cache["masks"]:
                continue
            label = str(obj.get("label", ""))
            cx, cy = _obj_centroid(obj)
            area = int(np.sum(cache["masks"][i]))
            _draw_label(canvas, label, cx, cy, area)
        _annotate_region_geometry(canvas, regions, adjacency_edges)
        out = staged_dir / f"{stem}_region_sam2_style_segmentation.png"
        cv2.imwrite(str(out), canvas)
        return f"scene_graph/staged/{stem}_region_sam2_style_segmentation.png"
    except Exception as exc:
        print(f"  [VizExport] region_sam2_style_segmentation skipped: {exc}")
        return None


def _write_region_tinted_overlay(
    reg_objs: List[Dict[str, Any]],
    img_rgb: np.ndarray,
    regions: List[Dict[str, Any]],
    lm: np.ndarray,
    adjacency_edges: List[Dict[str, Any]],
    staged_dir: Path,
    stem: str,
    alpha: float = 0.45,
) -> Optional[str]:
    try:
        h, w = lm.shape[:2]
        cache = _build_object_visual_cache(reg_objs, h, w)
        if cache is None:
            return None
        src = img_rgb.astype(np.float32)
        if src.shape[:2] != (h, w):
            src = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        palette_rgb = cache["palette_lut"][:, ::-1].astype(np.float32)
        tint = palette_rgb[cache["label_map"]]
        binary = cache["binary"]
        src[binary] = src[binary] * (1.0 - alpha) + tint[binary] * alpha
        out_bgr = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for cnts in cache["contours"].values():
            cv2.drawContours(out_bgr, cnts, -1, (255, 255, 255), 1)
        _annotate_region_geometry(out_bgr, regions, adjacency_edges)
        out = staged_dir / f"{stem}_region_tinted_overlay.png"
        cv2.imwrite(str(out), out_bgr)
        return f"scene_graph/staged/{stem}_region_tinted_overlay.png"
    except Exception as exc:
        print(f"  [VizExport] region_tinted_overlay skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Write all visualization PNGs for this image."""
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    stem = ctx.stem
    h, w = ctx.height, ctx.width

    objects: List[Dict[str, Any]] = list(ctx.extra.get("objects", []))
    regions = list(ctx.region_partition_meta or [])
    lm = ctx.region_label_map
    relations = list(ctx.relations or [])

    # Adjacency edges for region annotation (from regions_block adjacency graph)
    adj_edges: List[Dict[str, Any]] = []
    if ctx.regions_block:
        for e in (ctx.regions_block.get("region_adjacency_graph") or {}).get("edges", []):
            adj_edges.append(e)

    exports: Dict[str, str] = {}

    # --- Object visualizations (require masks) ---
    if objects:
        v = _write_segmentation(objects, h, w, staged_dir, stem)
        if v:
            exports["segmentation_image"] = v
            exports["sam2_segmentation_image"] = v

        v = _write_tinted_overlay(objects, ctx.img_rgb, h, w, staged_dir, stem)
        if v:
            exports["sam2_tinted_overlay_image"] = v
            exports["overlay_image"] = v

        v = _write_3d_viz(objects, ctx.img_bgr, relations, staged_dir, stem)
        if v:
            exports["3d_viz_image"] = v

    # --- Relations maps (draw even when objects=[], will show only region nodes) ---
    v = _write_relations_map(objects, relations, ctx.img_bgr, regions, staged_dir, stem, view="all")
    if v:
        exports["relations_map_image"] = v
    v = _write_relations_map(objects, relations, ctx.img_bgr, regions, staged_dir, stem, view="objects_only")
    if v:
        exports["relations_map_objects_image"] = v
    v = _write_relations_map(objects, relations, ctx.img_bgr, regions, staged_dir, stem, view="regions_only")
    if v:
        exports["relations_map_regions_image"] = v

    # --- Mask hierarchy map ---
    if ctx.mask_hierarchy:
        v = _write_hierarchy_map(objects, ctx.img_bgr, ctx.mask_hierarchy, regions, staged_dir, stem)
        if v:
            exports["mask_hierarchy_image"] = v

    # --- Region visualizations ---
    if regions and lm is not None:
        reg_objs = _build_region_visual_objects(regions, lm)
        if reg_objs:
            v = _write_region_segmentation(reg_objs, regions, lm, adj_edges, staged_dir, stem)
            if v:
                exports["region_segmentation_image"] = v
            v = _write_region_sam2_segmentation(reg_objs, regions, lm, adj_edges, staged_dir, stem)
            if v:
                exports["region_sam2_segmentation_image"] = v
            v = _write_region_tinted_overlay(reg_objs, ctx.img_rgb, regions, lm, adj_edges, staged_dir, stem)
            if v:
                exports["region_tinted_overlay_image"] = v

    # Store all export keys so scene_write can embed them in metadata
    ctx.path_exports.update(exports)

    written = len(exports)
    print(f"  [VizExport] {stem}: {written} visualization PNGs written")
    return ctx
