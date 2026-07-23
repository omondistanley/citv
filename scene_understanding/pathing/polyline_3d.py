"""Lift a 2D (+ per-pixel depth) polyline to metric 3D world coordinates.

Uses the same pinhole back-projection convention as the monolith's
``SceneUnderstandingPipeline._back_project`` (``x=(u-cx)*z/fx``,
``y=(v-cy)*z/fy``) so a lifted path's coordinates are directly comparable to
per-object ``coordinates_3d`` computed elsewhere in the pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scene_understanding.pathing.ground_plane import snap_uv_down_to_support

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def _bilinear_sample(depth: np.ndarray, u: float, v: float) -> Optional[float]:
    h, w = depth.shape[:2]
    if u < 0 or v < 0 or u > w - 1 or v > h - 1:
        return None
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = u - x0, v - y0
    d00, d01 = depth[y0, x0], depth[y0, x1]
    d10, d11 = depth[y1, x0], depth[y1, x1]
    vals = [d00, d01, d10, d11]
    if any((not np.isfinite(d)) or d <= 0 for d in vals):
        finite_vals = [d for d in vals if np.isfinite(d) and d > 0]
        return float(np.mean(finite_vals)) if finite_vals else None
    top = d00 * (1 - fx) + d01 * fx
    bot = d10 * (1 - fx) + d11 * fx
    return float(top * (1 - fy) + bot * fy)


def _snap_uv_down(uv: Point2D, support_mask: np.ndarray, max_search_px: int = 48) -> Point2D:
    x, y = snap_uv_down_to_support((int(round(uv[0])), int(round(uv[1]))), support_mask, max_search_px=max_search_px)
    return float(x), float(y)


def lift_polyline_2d_to_3d(
    polyline_2d: Sequence[Point2D],
    depth: np.ndarray,
    invalid_z: float = -1.0,
    support_mask: Optional[np.ndarray] = None,
    snap_search_px: int = 48,
) -> Tuple[List[List[float]], bool]:
    """Returns ``([[u, v, z], ...], any_point_snapped)``. ``z`` stays in camera
    space (metric depth at that pixel), not yet unprojected to world XYZ --
    that's what ``smooth_polyline_in_3d`` does once smoothing needs true 3D."""
    dm = np.asarray(depth, dtype=np.float32)
    out: List[List[float]] = []
    snapped_any = False
    for u, v in polyline_2d:
        uv = (float(u), float(v))
        if support_mask is not None:
            snapped_uv = _snap_uv_down(uv, support_mask, snap_search_px)
            if snapped_uv != uv:
                snapped_any = True
            uv = snapped_uv
        z = _bilinear_sample(dm, uv[0], uv[1])
        if z is None or z <= 0:
            out.append([uv[0], uv[1], invalid_z])
        else:
            out.append([uv[0], uv[1], float(z)])
    return out, snapped_any


def attach_polyline_3d_to_paths(
    paths: Sequence[Dict[str, Any]],
    depth: np.ndarray,
    invalid_z: float = -1.0,
    support_mask: Optional[np.ndarray] = None,
) -> None:
    """Mutates each path dict in place, adding ``polyline_3d`` (and
    ``polyline_2d_support_snapped`` when any point was snapped to support)."""
    for path in paths:
        poly2d = path.get("polyline_2d")
        if not poly2d:
            continue
        poly3d, snapped = lift_polyline_2d_to_3d(poly2d, depth, invalid_z=invalid_z, support_mask=support_mask)
        path["polyline_3d"] = poly3d
        if snapped:
            path["polyline_2d_support_snapped"] = [[p[0], p[1]] for p in poly3d]


def _unproject_uvz(polyline_3d: Sequence[Sequence[float]], intrinsics: Dict[str, float]) -> np.ndarray:
    fx = float(intrinsics.get("fx", 1.0))
    fy = float(intrinsics.get("fy", 1.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    arr = np.asarray(polyline_3d, dtype=np.float64)
    u, v, z = arr[:, 0], arr[:, 1], arr[:, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def smooth_polyline_in_3d(
    polyline_3d: Sequence[Sequence[float]],
    intrinsics: Dict[str, float],
    smoothing_window: int = 5,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Back-project (u,v,z) -> world XYZ, moving-average smooth X and Z only
    (Y/vertical is left raw so staircase steps, curbs, etc. aren't blurred
    away), then reproject to get a smoothed 2D display polyline alongside
    the smoothed 3D one.

    Returns ``(smoothed_3d_xyz, reprojected_2d)``.
    """
    pts = _unproject_uvz(polyline_3d, intrinsics)
    x, y, z = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
    valid = z > 0
    if valid.sum() >= 2:
        x[valid] = _moving_average(x[valid], smoothing_window)
        z[valid] = _moving_average(z[valid], smoothing_window)

    fx = float(intrinsics.get("fx", 1.0))
    fy = float(intrinsics.get("fy", 1.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    u = np.where(z > 0, x * fx / np.maximum(z, 1e-6) + cx, np.asarray(polyline_3d)[:, 0])
    v = np.where(z > 0, y * fy / np.maximum(z, 1e-6) + cy, np.asarray(polyline_3d)[:, 1])

    smoothed_3d = np.stack([x, y, z], axis=-1).tolist()
    reprojected_2d = np.stack([u, v], axis=-1).tolist()
    return smoothed_3d, reprojected_2d
