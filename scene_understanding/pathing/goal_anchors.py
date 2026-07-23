"""Approach-anchor repair: move path endpoints off object silhouettes onto
walkable ground, instead of at a mask's raw centroid (which usually lands
inside the object, e.g. the seat of a chair).

This has to be genuinely mask-aware, not bbox-aware, because the axis-
aligned bounding box of an irregular object (an L-shaped sofa, a ladder
leaning at an angle, a plant with sprawling leaves) is a poor proxy for
where its silhouette actually touches the ground. Wherever the real
per-pixel segmentation mask is available it is the source of truth; the
bbox is only a fallback when no mask array is attached to the object
record. Region identity (``region_label_map`` + region metadata) is used
too, so a contact point isn't accepted just because a handful of stray
pixels happen to pass a support-mask test -- it has to sit in a region
that is itself a substantial, contiguous walkable area.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scene_understanding.pathing.ground_plane import snap_uv_down_to_support, zero_shot_classify_text

_FACADE_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a tall vertical building facade, wall, or freestanding structure",
    "a pole, column, or trunk",
)
_NON_FACADE_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a small object sitting on the ground",
    "furniture, a person, or a vehicle",
    "a horizontal or low surface",
)


def object_mask_array(obj: Dict[str, Any], h: int, w: int) -> Optional[np.ndarray]:
    """The object's real per-pixel mask, resized to the working resolution.

    Checked under a few field names since different upstream stages
    (GroundedSAM2 vs. SAM2 AMG vs. staged package) have stored it under
    slightly different keys historically.
    """
    for key in ("_sam2_mask_array", "mask_array", "segmentation_mask", "mask"):
        m = obj.get(key)
        if m is None:
            continue
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)
    return None


def _bbox_to_corners(bbox: Sequence[float], h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    """Resolve a 4-tuple bbox to ``(x0, y0, x1, y1)`` corners regardless of
    whether it was stored as ``[x, y, w, h]`` (older/legacy convention, e.g.
    ``scene_understanding.py`` detections) or ``[x1, y1, x2, y2]`` corners
    (the staged package's ``stages/labelling.py`` output convention) --
    both exist in this codebase. Disambiguated by which interpretation
    yields a valid, in-bounds, non-degenerate box; corners are preferred on
    a tie since that's what objects reaching this module are produced with."""
    if len(bbox) != 4:
        return None
    a, b, c, d = (float(v) for v in bbox)

    def _valid_corners() -> Optional[Tuple[int, int, int, int]]:
        x0, y0, x1, y1 = int(a), int(b), int(c), int(d)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None

    def _valid_xywh() -> Optional[Tuple[int, int, int, int]]:
        x0, y0 = max(0, int(a)), max(0, int(b))
        x1, y1 = min(w, int(a + c)), min(h, int(b + d))
        return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None

    corners = _valid_corners()
    xywh = _valid_xywh()
    if corners is not None and xywh is not None and corners != xywh:
        # Both parse validly but disagree -- prefer whichever's width/height
        # look like a plausible object size relative to the image (an xywh
        # box misread as corners tends to blow up to a much larger area).
        area_c = (corners[2] - corners[0]) * (corners[3] - corners[1])
        area_x = (xywh[2] - xywh[0]) * (xywh[3] - xywh[1])
        img_area = max(1, h * w)
        return corners if area_c <= img_area else xywh if area_x <= img_area else corners
    return corners if corners is not None else xywh


def object_bbox_mask(obj: Dict[str, Any], h: int, w: int) -> Optional[np.ndarray]:
    """Fallback rectangular mask built from ``obj['bbox']`` when no real mask exists."""
    bbox = obj.get("bbox")
    if not bbox:
        return None
    corners = _bbox_to_corners(bbox, h, w)
    if corners is None:
        return None
    x0, y0, x1, y1 = corners
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def object_footprint_mask(obj: Dict[str, Any], h: int, w: int) -> Optional[np.ndarray]:
    """Real mask if present, otherwise the bbox rectangle as a coarse stand-in."""
    m = object_mask_array(obj, h, w)
    if m is not None and m.any():
        return m
    return object_bbox_mask(obj, h, w)


def mask_bbox_px(obj: Dict[str, Any], h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounding box derived from the *real* mask when available (falls
    back to the stored bbox metadata only if no mask array exists), so a
    stale/approximate stored bbox never overrides the true footprint."""
    mask = object_mask_array(obj, h, w)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bbox = obj.get("bbox")
    if bbox:
        return _bbox_to_corners(bbox, h, w)
    return None


def mask_centroid_px(obj: Dict[str, Any], h: int, w: int) -> Optional[Tuple[float, float]]:
    """Pixel-weighted centroid of the real mask (not the bbox center, which can
    sit far from an irregular object's actual visual mass)."""
    mask = object_mask_array(obj, h, w)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), float(ys.mean())
    bbox = mask_bbox_px(obj, h, w)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def mask_bottom_contour(mask: np.ndarray) -> Dict[int, int]:
    """For every column the mask occupies, the row index of its lowest
    (largest-y) True pixel -- i.e. the real silhouette's ground contact
    edge, not a flat line at the bbox's bottom."""
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {}
    bottom: Dict[int, int] = {}
    # argsort by column then take the last (max-y) occurrence per column.
    order = np.argsort(xs, kind="stable")
    xs_sorted, ys_sorted = xs[order], ys[order]
    for x, y in zip(xs_sorted.tolist(), ys_sorted.tolist()):
        prev = bottom.get(x)
        if prev is None or y > prev:
            bottom[x] = y
    return bottom


def vertical_structure_heuristic(
    obj: Dict[str, Any],
    h: int,
    w: int,
    aspect_thresh: float = 1.35,
    min_height_frac: float = 0.14,
    device: str = "cpu",
) -> bool:
    """True when the object's real mask silhouette looks like a tall vertical
    structure (facade/pole/trunk) -- geometry is derived from the mask's
    tight bbox (see ``mask_bbox_px``), not stored/approximate metadata."""
    bbox = mask_bbox_px(obj, h, w)
    geometry_says_vertical = False
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        geometry_says_vertical = (bh >= h * min_height_frac) and (bh / bw >= aspect_thresh)

    text = " ".join(
        str(obj.get(k, "")).strip()
        for k in ("canonical_name", "label", "caption", "semantic_label")
        if obj.get(k)
    ).strip()
    text_verdict = zero_shot_classify_text(
        text, _FACADE_CONCEPT_PROMPTS, _NON_FACADE_CONCEPT_PROMPTS, device=device,
    )
    if text_verdict is None:
        return geometry_says_vertical
    # Geometry (from the real mask) and open-vocab text agreeing raises
    # confidence; on disagreement, trust the mask-derived geometry over
    # potentially noisy/generic label text.
    return geometry_says_vertical or text_verdict


def _region_area_fraction(region_label_map: Optional[np.ndarray], region_id: int) -> float:
    if region_label_map is None:
        return 1.0
    lm = np.asarray(region_label_map)
    total = lm.size
    if total == 0:
        return 0.0
    return float((lm == region_id).sum()) / float(total)


def ground_intercept_goal_uv(
    obj: Dict[str, Any],
    support_mask: np.ndarray,
    feasible: np.ndarray,
    h: int,
    w: int,
    region_label_map: Optional[np.ndarray] = None,
    min_region_area_fraction: float = 0.01,
    max_down_search_px: int = 48,
    search_half_width_px: int = 40,
) -> Optional[Tuple[int, int]]:
    """Find where an object's real silhouette meets walkable ground.

    Unlike a bbox-center vertical scan, this walks the mask's actual bottom
    contour (every column the object occupies, at that column's true lowest
    mask pixel) and only accepts a contact point that (a) is walkable
    (``support_mask & feasible``) and (b) sits inside a region that is
    itself a non-trivial, contiguous walkable area -- not a few stray
    support-mask pixels caused by segmentation noise at the object's edge.
    """
    footprint = object_footprint_mask(obj, h, w)
    if footprint is None or not footprint.any():
        return None
    walkable = np.asarray(support_mask, dtype=bool) & np.asarray(feasible, dtype=bool)
    bottom = mask_bottom_contour(footprint)
    if not bottom:
        return None

    cx_mass = mask_centroid_px(obj, h, w)
    cx = int(round(cx_mass[0])) if cx_mass else int(round(np.mean(list(bottom.keys()))))

    def _region_ok(x: int, y: int) -> bool:
        if region_label_map is None:
            return True
        lm = np.asarray(region_label_map)
        rid = int(lm[y, x])
        if rid <= 0:
            return False
        return _region_area_fraction(lm, rid) >= min_region_area_fraction

    # Walk outward from the footprint's horizontal center of mass along its
    # own bottom contour (not a fixed-width band around the bbox center),
    # so a diagonally-oriented object still gets a contact point that sits
    # directly beneath its actual silhouette.
    columns_by_distance = sorted(bottom.keys(), key=lambda x: abs(x - cx))
    for x in columns_by_distance:
        if abs(x - cx) > search_half_width_px * 3:
            break
        y_edge = bottom[x]
        for dy in range(1, min(max_down_search_px, h - y_edge) + 1):
            y = y_edge + dy
            if y >= h:
                break
            if walkable[y, x] and _region_ok(x, y):
                return snap_uv_down_to_support((x, y), walkable)

    # Fallback: scan the full row just below the lowest contour point for
    # the nearest walkable+region-valid column.
    y_floor = min(max(bottom.values()) + 1, h - 1)
    row = walkable[y_floor, :]
    idx = np.nonzero(row)[0]
    idx = np.array([i for i in idx if _region_ok(int(i), y_floor)]) if region_label_map is not None else idx
    if idx.size:
        ix = int(idx[np.argmin(np.abs(idx - cx))])
        return ix, y_floor
    return None


def pair_passes_locomotion_gates(
    src: Dict[str, Any],
    tgt: Dict[str, Any],
    relations: Optional[List[Dict[str, Any]]],
    cfg: Any,
    h: int,
    w: int,
) -> bool:
    """Filter degenerate source/target pairs before spending an FMM solve on them."""
    min_span = float(getattr(cfg, "path_pair_min_span_px", 24.0)) if cfg else 24.0
    max_depth_delta = float(getattr(cfg, "path_pair_max_depth_delta_m", 6.0)) if cfg else 6.0
    require_relation = bool(getattr(cfg, "path_pair_require_relation", False)) if cfg else False

    src_c = mask_centroid_px(src, h, w)
    tgt_c = mask_centroid_px(tgt, h, w)
    if src_c and tgt_c:
        if ((src_c[0] - tgt_c[0]) ** 2 + (src_c[1] - tgt_c[1]) ** 2) ** 0.5 < min_span:
            return False

    def _z(obj: Dict[str, Any]) -> Optional[float]:
        stats = obj.get("depth_stats") or {}
        for k in ("median", "mean", "mode"):
            if stats.get(k) is not None:
                return float(stats[k])
        coords = obj.get("coordinates_3d") or {}
        return float(coords["z"]) if coords.get("z") is not None else None

    z0, z1 = _z(src), _z(tgt)
    if z0 is not None and z1 is not None and abs(z0 - z1) > max_depth_delta:
        return False

    src_vertical = vertical_structure_heuristic(src, h, w)
    tgt_vertical = vertical_structure_heuristic(tgt, h, w)
    if src_vertical and tgt_vertical:
        return False  # skip facade-like-to-facade-like pairs (degenerate wall<->wall paths)

    if require_relation and relations:
        src_id, tgt_id = src.get("id"), tgt.get("id")
        linked = any(
            {rel.get("source_id"), rel.get("target_id")} == {src_id, tgt_id}
            for rel in relations
        )
        if not linked:
            return False

    return True
