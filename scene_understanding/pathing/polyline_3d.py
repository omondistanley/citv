"""Lift 2D polylines to (u, v, z_m) using metric depth.

Plan §2.5: when a ``support_mask`` is provided, every centerline vertex is
first snapped down to the nearest support pixel before depth is sampled.
This eliminates per-vertex depth jitter when the centerline grazes a
foreground object's mask, which used to push the rendered polyline onto a
table top, a chair seat, or a person's head.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _snap_uv_down(uv: Tuple[float, float], support_mask: np.ndarray, max_search_px: int = 48) -> Tuple[float, float]:
    """Walk down column ``u`` until support_mask is True; clip to bounds."""
    sm = np.asarray(support_mask, dtype=bool)
    if sm.size == 0:
        return uv
    h, w = sm.shape[:2]
    x = int(round(max(0.0, min(float(w - 1), uv[0]))))
    y = int(round(max(0.0, min(float(h - 1), uv[1]))))
    end_y = min(h - 1, y + int(max_search_px))
    col = sm[y:end_y + 1, x]
    if col.any():
        return float(x), float(y + int(np.argmax(col)))
    return float(x), float(end_y)


def lift_polyline_2d_to_3d(
    polyline_2d: List[Any],
    metric_depth_m: np.ndarray,
    *,
    invalid_z: float = -1.0,
    support_mask: Optional[np.ndarray] = None,
    snap_search_px: int = 48,
) -> List[List[float]]:
    """Sample depth at each vertex (bilinear). Returns [[u, v, z], ...] in pixel + meters.

    When ``support_mask`` is supplied, each (u, v) is first walked down to the
    nearest support pixel (plan §2.5). The output uses the snapped (u, v) so
    rendering can follow the support surface exactly.
    """
    dep = np.asarray(metric_depth_m, dtype=np.float32)
    if dep.ndim != 2 or not polyline_2d:
        return []
    h, w = dep.shape[:2]
    out: List[List[float]] = []
    for xy in polyline_2d:
        if not xy or len(xy) < 2:
            continue
        try:
            u = float(xy[0])
            v = float(xy[1])
        except (TypeError, ValueError):
            continue
        if support_mask is not None:
            u, v = _snap_uv_down((u, v), support_mask, max_search_px=snap_search_px)
        xi = max(0.0, min(float(w - 1), u))
        yi = max(0.0, min(float(h - 1), v))
        x0 = int(np.floor(xi))
        y0 = int(np.floor(yi))
        x1 = min(w - 1, x0 + 1)
        y1 = min(h - 1, y0 + 1)
        tx = xi - x0
        ty = yi - y0
        z00 = float(dep[y0, x0])
        z10 = float(dep[y0, x1])
        z01 = float(dep[y1, x0])
        z11 = float(dep[y1, x1])
        z = (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11
        if not np.isfinite(z) or z <= 0.0:
            z = float(invalid_z)
        out.append([float(u), float(v), float(z)])
    return out


def attach_polyline_3d_to_paths(
    paths: List[Any],
    metric_depth_m: np.ndarray,
    cfg: Any,
    *,
    support_mask: Optional[np.ndarray] = None,
) -> None:
    if not bool(getattr(cfg, "path_export_polyline_3d", True)) if cfg else True:
        return
    dep = np.asarray(metric_depth_m, dtype=np.float32)
    if dep.ndim != 2 or dep.size == 0:
        return
    inv = float(getattr(cfg, "path_polyline_3d_invalid_depth_value", -1.0)) if cfg else -1.0
    snap_px = int(getattr(cfg, "polyline_support_snap_max_px", 48)) if cfg else 48
    for p in paths or []:
        if not isinstance(p, dict):
            continue
        pl = p.get("polyline_2d") or []
        if len(pl) < 2:
            continue
        p3 = lift_polyline_2d_to_3d(
            pl,
            dep,
            invalid_z=inv,
            support_mask=support_mask,
            snap_search_px=snap_px,
        )
        if len(p3) >= 2:
            p["polyline_3d"] = p3
            # If we snapped, also expose the snapped 2D so renderers don't
            # need to recompute the snap (plan §2.5 + §2.6).
            if support_mask is not None:
                p["polyline_2d_support_snapped"] = [[row[0], row[1]] for row in p3]


def smooth_polyline_in_3d(
    polyline_3d: List[List[float]],
    intrinsics: Optional[Dict[str, float]],
    *,
    smoothing_window: int = 5,
) -> Dict[str, List[List[float]]]:
    """Plan §2.7: smooth the 3D polyline in camera-space (X, Z) and reproject.

    Returns ``{"polyline_3d_smoothed": [...], "polyline_2d_reprojected": [...]}``.
    Uses a simple moving-average filter on (X, Z); Y is taken from the support
    point (already snapped during ``lift_polyline_2d_to_3d``). When intrinsics
    are missing the function returns empty lists.
    """
    if not polyline_3d or intrinsics is None:
        return {"polyline_3d_smoothed": [], "polyline_2d_reprojected": []}
    fx = float(intrinsics.get("fx") or 0.0)
    fy = float(intrinsics.get("fy") or 0.0)
    cx = float(intrinsics.get("cx") or 0.0)
    cy = float(intrinsics.get("cy") or 0.0)
    if fx <= 0.0 or fy <= 0.0:
        return {"polyline_3d_smoothed": [], "polyline_2d_reprojected": []}

    arr = np.asarray(polyline_3d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return {"polyline_3d_smoothed": [], "polyline_2d_reprojected": []}

    u = arr[:, 0]
    v = arr[:, 1]
    z = arr[:, 2]
    valid = np.isfinite(z) & (z > 0.0)
    if not valid.any():
        return {"polyline_3d_smoothed": [], "polyline_2d_reprojected": []}

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    win = max(1, int(smoothing_window))
    if win > 1 and X.size >= win:
        kernel = np.ones(win, dtype=np.float64) / float(win)
        X_s = np.convolve(X, kernel, mode="same")
        Z_s = np.convolve(Z, kernel, mode="same")
        # Y is left as-is so vertical staircase steps are preserved.
        Y_s = Y
    else:
        X_s, Y_s, Z_s = X, Y, Z

    Z_safe = np.where(Z_s > 1e-3, Z_s, np.maximum(z, 1e-3))
    u_re = (X_s * fx / Z_safe) + cx
    v_re = (Y_s * fy / Z_safe) + cy

    smoothed_3d = [[float(X_s[i]), float(Y_s[i]), float(Z_s[i])] for i in range(arr.shape[0])]
    reprojected_2d = [[float(u_re[i]), float(v_re[i])] for i in range(arr.shape[0])]
    return {"polyline_3d_smoothed": smoothed_3d, "polyline_2d_reprojected": reprojected_2d}
