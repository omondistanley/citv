"""Goal UV repair for locomotion: ground intercept in front of vertical structures."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def _mask_bbox_px(obj: Mapping[str, Any], h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    bbox = obj.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x0 = int(max(0, min(w - 1, round(x))))
        y0 = int(max(0, min(h - 1, round(y))))
        x1 = int(max(0, min(w - 1, round(x + bw))))
        y1 = int(max(0, min(h - 1, round(y + bh))))
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(h - 1, y0 + 1)
        return x0, y0, x1, y1
    raw = obj.get("_sam2_mask_array")
    if raw is None:
        return None
    mm = np.asarray(raw, dtype=bool)
    if mm.shape[:2] != (h, w):
        if cv2 is None:
            return None
        mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.where(mm)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def vertical_structure_heuristic(
    obj: Mapping[str, Any],
    h: int,
    w: int,
    *,
    aspect_thresh: float = 1.35,
    min_height_frac: float = 0.14,
) -> bool:
    """True when mask reads as tall vertical (façade / pole / tree trunk-like)."""
    bb = _mask_bbox_px(obj, h, w)
    if bb is None:
        return False
    x0, y0, x1, y1 = bb
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    if bh < h * float(min_height_frac):
        return False
    return (bh / float(bw)) >= float(aspect_thresh)


def ground_intercept_goal_uv(
    obj: Mapping[str, Any],
    support_mask: Optional[np.ndarray],
    feasible: np.ndarray,
    h: int,
    w: int,
    *,
    snap_walkable,
) -> Optional[Tuple[int, int]]:
    """Pick a UV on support near the bottom-center of the object bbox."""
    if support_mask is None or not np.asarray(support_mask, dtype=bool).any():
        return None
    sm = np.asarray(support_mask, dtype=bool)
    bb = _mask_bbox_px(obj, h, w)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    cx = int(round(0.5 * (x0 + x1)))
    row = min(y1, h - 1)
    # Search upward a few rows along bottom strip for support ∩ feasible
    for dy in range(0, min(48, h)):
        y = min(h - 1, row + dy)
        x_lo = max(0, cx - 40)
        x_hi = min(w - 1, cx + 40)
        row_sup = np.where(sm[y, x_lo : x_hi + 1] & feasible[y, x_lo : x_hi + 1])[0]
        if row_sup.size > 0:
            ix = x_lo + int(np.median(row_sup))
            return snap_walkable(ix, y, feasible, w, h)
    # Full-width scan along bottom row of bbox extension
    y = min(h - 1, y1)
    row_sup = np.where(sm[y, :] & feasible[y, :])[0]
    if row_sup.size > 0:
        ix = int(row_sup[np.argmin(np.abs(row_sup - cx))])
        return snap_walkable(ix, y, feasible, w, h)
    return None


def pair_passes_locomotion_gates(
    src: Mapping[str, Any],
    tgt: Mapping[str, Any],
    relations: List[Mapping[str, Any]],
    cfg: Any,
    *,
    h: int,
    w: int,
) -> Tuple[bool, str]:
    """Filter degenerate or implausible object pairs for ribbon FMM."""
    sid = str(src.get("id", ""))
    tid = str(tgt.get("id", ""))
    min_span = float(getattr(cfg, "path_pair_min_span_px", 0.0) or 0.0)
    if min_span > 0:
        c0 = src.get("mask_centroid_2d") or [0, 0]
        c1 = tgt.get("mask_centroid_2d") or [0, 0]
        try:
            d = math.hypot(float(c0[0]) - float(c1[0]), float(c0[1]) - float(c1[1]))
        except (TypeError, IndexError, ValueError):
            d = 0.0
        if d < min_span:
            return False, "below_min_span_px"

    max_depth_d = float(getattr(cfg, "path_pair_max_depth_delta_m", 0.0) or 0.0)
    if max_depth_d > 0:
        z0 = _depth_m(src)
        z1 = _depth_m(tgt)
        if z0 > 1e-6 and z1 > 1e-6 and abs(z0 - z1) > max_depth_d:
            return False, "depth_delta_exceeded"

    if bool(getattr(cfg, "path_pair_skip_facade_facade", False)):
        lab0 = normalize_label_str(src.get("canonical_label") or src.get("label") or "")
        lab1 = normalize_label_str(tgt.get("canonical_label") or tgt.get("label") or "")
        facade_terms = {"building", "wall", "house", "facade", "skyscraper"}
        if lab0 in facade_terms and lab1 in facade_terms:
            req_vert = bool(getattr(cfg, "path_pair_facade_skip_requires_vertical", False))
            if not req_vert or (
                vertical_structure_heuristic(src, h, w) and vertical_structure_heuristic(tgt, h, w)
            ):
                return False, "facade_facade_skip"

    if bool(getattr(cfg, "path_pair_require_relation", False)):
        if not _relation_links_ids(relations, sid, tid):
            return False, "no_relation"

    return True, ""


def normalize_label_str(s: Any) -> str:
    from ..evidence import normalize_label

    return normalize_label(str(s or ""))


def _depth_m(obj: Mapping[str, Any]) -> float:
    ds = obj.get("depth_stats") or {}
    for k in ("median", "mean", "mode"):
        v = ds.get(k)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
    c3 = obj.get("coordinates_3d") or {}
    z = c3.get("z")
    if isinstance(z, (int, float)) and math.isfinite(float(z)):
        return float(z)
    return 0.0


def _relation_links_ids(relations: List[Mapping[str, Any]], a: str, b: str) -> bool:
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        s = str(rel.get("sub_id") or rel.get("subject_id") or rel.get("subject") or "")
        o = str(rel.get("obj_id") or rel.get("object_id") or rel.get("object") or "")
        if {s, o} == {a, b}:
            return True
    return False


__all__ = [
    "vertical_structure_heuristic",
    "ground_intercept_goal_uv",
    "pair_passes_locomotion_gates",
]
