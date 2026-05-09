"""Ground / support surface estimation for path anchoring (CPU-first).

Plan §2.1 deliverable. Two complementary sources are unioned:

- ``fit_ground_plane`` — RANSAC plane fit on depth-back-projected camera-space
  points sampled from the lower 40% of the image (avoiding object pixels).
- ``build_support_mask`` — semantic union with regions whose ``region_type`` /
  ``semantic_label`` matches floor/ground/road/stair/sidewalk/tabletop tokens
  from ``scene_understanding/resources/path_action_ontology.json``.

Returns boolean ``(H, W)`` masks suitable for intersecting with the walkable
mask in ``walkable_mask.build_path_walkable_mask`` (plan §2.4) and for
snap-to-support in ``polyline_3d.lift_polyline_2d_to_3d`` (plan §2.5).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Tokens that mark a region as a real physical support surface for actors.
# Kept here (not in ontology JSON) because these are the *role* labels we
# project onto regions, not action prompts. Synonyms are case-insensitive
# substring matches so "stair_tread", "stairtread", "stairs", "stair" all hit.
_SUPPORT_TOKENS: Tuple[str, ...] = (
    "floor",
    "ground",
    "road",
    "path",
    "sidewalk",
    "stair",
    "step",
    "tread",
    "tabletop",
    "table top",
    "table_top",
    "platform",
    "bridge",
    "water_surface",
    "water surface",
    "deck",
    "ramp",
    "rug",
    "carpet",
    "lawn",
    "grass",
    "terrain",
)


def _support_label_match(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    return any(tok in s for tok in _SUPPORT_TOKENS)


def fit_ground_plane(
    metric_depth_m: Optional[np.ndarray],
    intrinsics: Optional[Dict[str, float]],
    *,
    object_mask: Optional[np.ndarray] = None,
    sample_rows: Tuple[float, float] = (0.6, 1.0),
    inlier_thresh_m: float = 0.10,
    max_iter: int = 80,
    min_samples: int = 200,
) -> Optional[Dict[str, Any]]:
    """RANSAC fit a plane n . X + d = 0 in camera coords.

    Returns ``None`` when depth/intrinsics are missing or the fit collapses
    (too few inliers). Otherwise returns:

    ``{"normal": (3,), "d": float, "inliers_uv": (N, 2), "inlier_mask": (H, W)}``

    The plane equation in camera coordinates is ``n.x*X + n.y*Y + n.z*Z + d = 0``.
    Distance of a point P to the plane is ``|n . P + d|``.
    """
    if metric_depth_m is None or intrinsics is None:
        return None
    dm = np.asarray(metric_depth_m, dtype=np.float32)
    if dm.ndim != 2 or dm.size == 0:
        return None
    h, w = dm.shape[:2]
    fx = float(intrinsics.get("fx") or 0.0)
    fy = float(intrinsics.get("fy") or 0.0)
    cx = float(intrinsics.get("cx") or w * 0.5)
    cy = float(intrinsics.get("cy") or h * 0.5)
    if fx <= 0.0 or fy <= 0.0:
        return None

    # Sample mask: lower band of the image, finite depth, off objects.
    y_lo = int(max(0, min(h - 1, sample_rows[0] * h)))
    y_hi = int(max(0, min(h, sample_rows[1] * h)))
    band = np.zeros((h, w), dtype=bool)
    band[y_lo:y_hi, :] = True
    finite = np.isfinite(dm) & (dm > 0.0)
    cand = band & finite
    if object_mask is not None:
        om = np.asarray(object_mask, dtype=bool)
        if om.shape == cand.shape:
            cand &= ~om
    ys, xs = np.where(cand)
    if xs.size < min_samples:
        # Fall back to all finite depths.
        ys, xs = np.where(finite)
        if xs.size < min_samples:
            return None

    z = dm[ys, xs].astype(np.float64)
    X = (xs.astype(np.float64) - cx) * z / fx
    Y = (ys.astype(np.float64) - cy) * z / fy
    Z = z
    pts = np.stack([X, Y, Z], axis=1)

    # Subsample for RANSAC speed cap. 4000 points is plenty for a plane.
    if pts.shape[0] > 4000:
        idx = np.random.default_rng(seed=0).choice(pts.shape[0], 4000, replace=False)
        pts = pts[idx]
        ys = ys[idx]
        xs = xs[idx]

    rng = np.random.default_rng(seed=1)
    best_inliers = None
    best_count = 0
    n_pts = pts.shape[0]
    for _ in range(int(max_iter)):
        sample_idx = rng.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts[sample_idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -float(np.dot(normal, p1))
        # Distances to plane.
        dist = np.abs(pts @ normal + d)
        inliers = dist < inlier_thresh_m
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if best_inliers is None or best_count < max(50, n_pts // 20):
        return None

    # Refit on inliers via SVD for a tighter plane.
    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    centred = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    normal /= max(1e-9, float(np.linalg.norm(normal)))
    d = -float(np.dot(normal, centroid))

    # Orient normal so it points "up" (negative Y in image-camera convention).
    if normal[1] > 0:
        normal = -normal
        d = -d

    inlier_xy = np.stack([xs[best_inliers], ys[best_inliers]], axis=1)
    inlier_mask = np.zeros((h, w), dtype=bool)
    inlier_mask[inlier_xy[:, 1], inlier_xy[:, 0]] = True
    return {
        "normal": normal.astype(np.float32),
        "d": float(d),
        "inliers_uv": inlier_xy.astype(np.int32),
        "inlier_mask": inlier_mask,
        "inlier_count": int(best_count),
        "sample_count": int(n_pts),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def project_inlier_mask(
    plane: Dict[str, Any],
    metric_depth_m: np.ndarray,
    intrinsics: Optional[Dict[str, float]],
    *,
    inlier_thresh_m: float = 0.10,
) -> np.ndarray:
    """Densify the plane to a full ``(H, W)`` mask using the depth map."""
    dm = np.asarray(metric_depth_m, dtype=np.float32)
    h, w = dm.shape[:2]
    if intrinsics is None or plane is None:
        return np.zeros((h, w), dtype=bool)
    fx = float(intrinsics.get("fx") or plane.get("fx") or 0.0)
    fy = float(intrinsics.get("fy") or plane.get("fy") or 0.0)
    cx = float(intrinsics.get("cx") or plane.get("cx") or w * 0.5)
    cy = float(intrinsics.get("cy") or plane.get("cy") or h * 0.5)
    if fx <= 0.0 or fy <= 0.0:
        return np.zeros((h, w), dtype=bool)
    normal = np.asarray(plane["normal"], dtype=np.float32)
    d = float(plane["d"])
    finite = np.isfinite(dm) & (dm > 0.0)
    if not finite.any():
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.indices((h, w), dtype=np.float32)
    z = dm
    X = (xx - cx) * z / fx
    Y = (yy - cy) * z / fy
    Z = z
    dist = np.abs(normal[0] * X + normal[1] * Y + normal[2] * Z + d)
    return finite & (dist < float(inlier_thresh_m))


def build_semantic_support_mask(
    region_label_map: Optional[np.ndarray],
    regions_meta: Optional[List[Dict[str, Any]]],
) -> np.ndarray:
    """Mask where the region label maps to a support-role token."""
    if region_label_map is None:
        return np.zeros((1, 1), dtype=bool)
    lm = np.asarray(region_label_map, dtype=np.int32)
    h, w = lm.shape[:2]
    if not regions_meta:
        return np.zeros((h, w), dtype=bool)
    support_indices: List[int] = []
    for r in regions_meta:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("region_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        if idx <= 0:
            continue
        text_blob = " ".join([
            str(r.get("region_type", "")),
            str(r.get("semantic_label", "")),
            str(r.get("layer_type", "")),
        ])
        if _support_label_match(text_blob):
            support_indices.append(idx)
    if not support_indices:
        return np.zeros((h, w), dtype=bool)
    mask = np.isin(lm, np.asarray(support_indices, dtype=np.int32))
    return np.asarray(mask, dtype=bool)


def build_support_mask(
    metric_depth_m: Optional[np.ndarray],
    intrinsics: Optional[Dict[str, float]],
    region_label_map: Optional[np.ndarray],
    regions_meta: Optional[List[Dict[str, Any]]],
    *,
    object_mask: Optional[np.ndarray] = None,
    inlier_thresh_m: float = 0.10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Union of (a) ground-plane inliers and (b) semantic support regions.

    Returns ``(support_mask, info)`` where ``info`` records the ground plane
    fit (or ``None`` when it failed) and how many pixels each source
    contributed. When both sources fail, an empty mask is returned and
    callers should fall back to the existing walkable behaviour.
    """
    sem = build_semantic_support_mask(region_label_map, regions_meta)
    h, w = sem.shape[:2] if sem.size else (
        int(metric_depth_m.shape[0]) if metric_depth_m is not None else 1,
        int(metric_depth_m.shape[1]) if metric_depth_m is not None else 1,
    )
    plane = fit_ground_plane(
        metric_depth_m,
        intrinsics,
        object_mask=object_mask,
        inlier_thresh_m=inlier_thresh_m,
    )
    plane_mask = np.zeros((h, w), dtype=bool)
    if plane is not None and metric_depth_m is not None:
        try:
            plane_mask = project_inlier_mask(
                plane, metric_depth_m, intrinsics, inlier_thresh_m=inlier_thresh_m
            )
        except Exception:
            plane_mask = plane.get("inlier_mask", plane_mask)
    if plane_mask.shape != (h, w):
        plane_mask = np.zeros((h, w), dtype=bool)
    union = sem | plane_mask
    return union, {
        "plane": (
            None
            if plane is None
            else {
                "normal": plane["normal"].tolist(),
                "d": plane["d"],
                "inlier_count": plane.get("inlier_count", 0),
                "sample_count": plane.get("sample_count", 0),
            }
        ),
        "semantic_pixel_count": int(sem.sum()),
        "plane_pixel_count": int(plane_mask.sum()),
        "union_pixel_count": int(union.sum()),
    }


def snap_uv_down_to_support(
    uv: Tuple[int, int],
    support_mask: Optional[np.ndarray],
    *,
    max_search_px: int = 200,
) -> Tuple[int, int]:
    """Walk down from ``uv`` until we hit ``support_mask``; clip to bounds.

    Used by ``polyline_3d.lift_polyline_2d_to_3d`` (plan §2.5) and by the
    foot/support anchor helper in ``affordances_export`` (plan §2.2). When
    the column never enters support_mask, return the bottom-most pixel.
    """
    x, y = int(uv[0]), int(uv[1])
    if support_mask is None:
        return x, y
    sm = np.asarray(support_mask, dtype=bool)
    if sm.size == 0:
        return x, y
    h, w = sm.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    end_y = min(h - 1, y + int(max_search_px))
    col = sm[y:end_y + 1, x]
    if col.any():
        return x, y + int(np.argmax(col))
    return x, end_y
