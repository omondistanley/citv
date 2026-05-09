"""Shared 2D polyline extraction for paths (NaN-safe, optional image clamp)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def int_xy_from_vertex(
    xy: Any,
    width: int,
    height: int,
    *,
    clamp: bool = True,
) -> Optional[Tuple[int, int]]:
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None
    try:
        fx = float(xy[0])
        fy = float(xy[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fx) or not math.isfinite(fy):
        return None
    ix = int(round(fx))
    iy = int(round(fy))
    if clamp and width > 0 and height > 0:
        ix = max(0, min(width - 1, ix))
        iy = max(0, min(height - 1, iy))
    return (ix, iy)


def int_polyline_from_path_dict(
    path_dict: Dict[str, Any],
    *,
    prefer_geodesic: bool = False,
    width: int = 0,
    height: int = 0,
    clamp: bool = True,
) -> List[Tuple[int, int]]:
    """Vertices from validated display geometry, else geodesic/reprojected/raw."""
    seq: List[Any] = []
    display = path_dict.get("display_polyline_2d")
    if isinstance(display, list) and len(display) >= 2:
        seq = display
    if not seq and prefer_geodesic:
        geo = path_dict.get("polyline_geodesic_2d")
        if isinstance(geo, list) and len(geo) >= 2:
            seq = geo
    if not seq:
        seq = path_dict.get("polyline_2d_validated") or path_dict.get("polyline_2d_reprojected") or path_dict.get("polyline_2d") or []
    if len(seq) < 2:
        return []
    out: List[Tuple[int, int]] = []
    for xy in seq:
        pt = int_xy_from_vertex(xy, width, height, clamp=clamp)
        if pt is None:
            continue
        if not out or out[-1] != pt:
            out.append(pt)
    return out if len(out) >= 2 else []
