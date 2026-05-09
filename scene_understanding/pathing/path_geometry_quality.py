"""Geometry validation and display-centerline helpers for path QA.

The staged path exporter may produce routes from raster backtraces, mask
centroids, or fallback anchors.  Those are useful planning traces, but they are
not always suitable for primary QA or animation.  This module keeps the raw
route, creates a constrained display route, and reports measurable geometry
defects so bad shapes can be downranked instead of silently rendered.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


def _float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def clean_polyline(polyline: Iterable[Any], width: int = 0, height: int = 0) -> List[List[float]]:
    """Return finite ``[[x, y], ...]`` points, clipped to image bounds if known."""
    out: List[List[float]] = []
    for xy in polyline or []:
        if not isinstance(xy, Sequence) or len(xy) < 2:
            continue
        x = _float(xy[0], float("nan"))
        y = _float(xy[1], float("nan"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if width > 0:
            x = max(0.0, min(float(width - 1), x))
        if height > 0:
            y = max(0.0, min(float(height - 1), y))
        if not out or math.hypot(out[-1][0] - x, out[-1][1] - y) >= 0.75:
            out.append([float(x), float(y)])
    return out


def polyline_length(polyline: Sequence[Sequence[float]]) -> float:
    if len(polyline) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(_float(polyline[i][0]) - _float(polyline[i - 1][0]), _float(polyline[i][1]) - _float(polyline[i - 1][1]))
            for i in range(1, len(polyline))
        )
    )


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _rdp(points: List[Point], epsilon: float) -> List[Point]:
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    max_d = -1.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i], a, b)
        if d > max_d:
            max_d = d
            idx = i
    if max_d > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [a, b]


def simplify_polyline(polyline: Sequence[Sequence[float]], epsilon_px: float = 3.0) -> List[List[float]]:
    pts = [(float(p[0]), float(p[1])) for p in polyline if len(p) >= 2]
    if len(pts) < 3:
        return [[x, y] for x, y in pts]
    simplified = _rdp(pts, max(0.5, float(epsilon_px)))
    return [[round(x, 3), round(y, 3)] for x, y in simplified]


def _chaikin_once(polyline: Sequence[Sequence[float]]) -> List[List[float]]:
    if len(polyline) < 3:
        return [[float(p[0]), float(p[1])] for p in polyline]
    out: List[List[float]] = [[float(polyline[0][0]), float(polyline[0][1])]]
    for i in range(len(polyline) - 1):
        x0, y0 = float(polyline[i][0]), float(polyline[i][1])
        x1, y1 = float(polyline[i + 1][0]), float(polyline[i + 1][1])
        out.append([0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1])
        out.append([0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1])
    out.append([float(polyline[-1][0]), float(polyline[-1][1])])
    return out


def smooth_polyline(polyline: Sequence[Sequence[float]], iterations: int = 1) -> List[List[float]]:
    pts = [[float(p[0]), float(p[1])] for p in polyline if len(p) >= 2]
    if len(pts) < 3:
        return pts
    for _ in range(max(0, int(iterations))):
        pts = _chaikin_once(pts)
    return [[round(x, 3), round(y, 3)] for x, y in pts]


def _sample_points(polyline: Sequence[Sequence[float]], samples_per_segment: int = 6) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    if len(polyline) < 2:
        return pts
    for i in range(1, len(polyline)):
        x0, y0 = float(polyline[i - 1][0]), float(polyline[i - 1][1])
        x1, y1 = float(polyline[i][0]), float(polyline[i][1])
        n = max(2, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / max(1.0, samples_per_segment))))
        for j in range(n):
            t = j / float(max(1, n - 1))
            pts.append((int(round((1.0 - t) * x0 + t * x1)), int(round((1.0 - t) * y0 + t * y1))))
    out: List[Tuple[int, int]] = []
    for p in pts:
        if not out or out[-1] != p:
            out.append(p)
    return out


def _valid_fraction(polyline: Sequence[Sequence[float]], mask: Optional[np.ndarray]) -> Tuple[float, int]:
    if mask is None or len(polyline) < 2:
        return 1.0, 0
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return 1.0, 0
    h, w = arr.shape[:2]
    samples = _sample_points(polyline)
    if not samples:
        return 0.0, 0
    good = 0
    transitions = 0
    prev: Optional[bool] = None
    for x, y in samples:
        inside = 0 <= x < w and 0 <= y < h and bool(arr[y, x])
        good += int(inside)
        if prev is not None and prev != inside:
            transitions += 1
        prev = inside
    return float(good) / float(max(1, len(samples))), transitions


def build_display_geometry(
    raw_polyline: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    feasible_mask: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    """Create validated/display centerlines without changing the raw route."""
    raw = clean_polyline(raw_polyline, width, height)
    if len(raw) < 2:
        return {
            "polyline_2d_raw": raw,
            "polyline_2d_validated": raw,
            "display_polyline_2d": raw,
            "geometry_smoothing_provenance": {
                "source": "raw_too_short",
                "smoothability_status": "too_short",
                "raw_vertex_count": len(raw),
                "display_vertex_count": len(raw),
            },
        }

    epsilon = _float(getattr(cfg, "path_display_simplify_epsilon_px", 3.0), 3.0) if cfg else 3.0
    iterations = int(getattr(cfg, "path_display_smoothing_iterations", 1)) if cfg else 1
    min_valid = _float(getattr(cfg, "path_display_min_valid_fraction", 0.92), 0.92) if cfg else 0.92
    simplified = simplify_polyline(raw, epsilon_px=epsilon)
    smoothed = smooth_polyline(simplified, iterations=iterations)
    smoothed = clean_polyline(smoothed, width, height)

    feasible_fraction, feasible_crossings = _valid_fraction(smoothed, feasible_mask)
    support_fraction, support_crossings = _valid_fraction(smoothed, support_mask)
    smooth_valid = feasible_fraction >= min_valid and support_fraction >= max(0.50, min_valid - 0.20)

    if smooth_valid and len(smoothed) >= 2:
        display = smoothed
        status = "smoothed_validated"
        source = "simplified_chaikin_constrained"
    else:
        simp_feasible, simp_feas_cross = _valid_fraction(simplified, feasible_mask)
        simp_support, simp_sup_cross = _valid_fraction(simplified, support_mask)
        if simp_feasible >= min_valid and simp_support >= max(0.50, min_valid - 0.20):
            display = clean_polyline(simplified, width, height)
            status = "simplified_validated"
            source = "rdp_simplified"
            feasible_fraction, support_fraction = simp_feasible, simp_support
            feasible_crossings, support_crossings = simp_feas_cross, simp_sup_cross
        else:
            display = raw
            status = "raw_retained_smoothing_invalid"
            source = "raw_planning_polyline"

    return {
        "polyline_2d_raw": [[round(float(x), 3), round(float(y), 3)] for x, y in raw],
        "polyline_2d_validated": [[round(float(x), 3), round(float(y), 3)] for x, y in display],
        "display_polyline_2d": [[round(float(x), 3), round(float(y), 3)] for x, y in display],
        "geometry_smoothing_provenance": {
            "source": source,
            "smoothability_status": status,
            "raw_vertex_count": len(raw),
            "simplified_vertex_count": len(simplified),
            "display_vertex_count": len(display),
            "simplification_epsilon_px": round(float(epsilon), 3),
            "smoothing_iterations": int(iterations),
            "validation": {
                "feasible_fraction": round(float(feasible_fraction), 4),
                "support_fraction": round(float(support_fraction), 4),
                "feasible_boundary_crossing_count": int(feasible_crossings),
                "support_boundary_crossing_count": int(support_crossings),
                "min_valid_fraction": round(float(min_valid), 4),
            },
        },
    }


def _turn_angles(polyline: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    angles: List[float] = []
    signs: List[float] = []
    if len(polyline) < 3:
        return angles, signs
    for i in range(1, len(polyline) - 1):
        ax = float(polyline[i][0]) - float(polyline[i - 1][0])
        ay = float(polyline[i][1]) - float(polyline[i - 1][1])
        bx = float(polyline[i + 1][0]) - float(polyline[i][0])
        by = float(polyline[i + 1][1]) - float(polyline[i][1])
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la <= 1e-6 or lb <= 1e-6:
            continue
        dot = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        cross = ax * by - ay * bx
        angles.append(float(math.degrees(math.acos(dot))))
        signs.append(1.0 if cross > 0.0 else -1.0 if cross < 0.0 else 0.0)
    return angles, signs


def _support_snap_stats(raw: Sequence[Sequence[float]], poly3d: Sequence[Sequence[float]]) -> Dict[str, float]:
    n = min(len(raw), len(poly3d))
    if n <= 0:
        return {"mean_px": 0.0, "max_px": 0.0}
    dists: List[float] = []
    for i in range(n):
        if len(raw[i]) < 2 or len(poly3d[i]) < 2:
            continue
        dists.append(math.hypot(float(poly3d[i][0]) - float(raw[i][0]), float(poly3d[i][1]) - float(raw[i][1])))
    if not dists:
        return {"mean_px": 0.0, "max_px": 0.0}
    return {"mean_px": float(np.mean(dists)), "max_px": float(max(dists))}


def evaluate_geometry_quality(
    *,
    raw_polyline: Sequence[Sequence[float]],
    display_polyline: Sequence[Sequence[float]],
    polyline_3d: Optional[Sequence[Sequence[float]]] = None,
    feasible_mask: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    boundary_trace: Optional[Dict[str, Any]] = None,
    kinematic_signatures: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    raw = clean_polyline(raw_polyline)
    display = clean_polyline(display_polyline)
    angles, signs = _turn_angles(display)
    p95 = float(np.percentile(np.asarray(angles, dtype=np.float32), 95)) if angles else 0.0
    zig_turns = [s for a, s in zip(angles, signs) if a >= 18.0 and abs(s) > 0.0]
    alternations = sum(1 for i in range(1, len(zig_turns)) if zig_turns[i] != zig_turns[i - 1])
    zigzag_score = float(alternations) / float(max(1, len(zig_turns) - 1)) if len(zig_turns) >= 2 else 0.0
    curvature = float(np.mean([(a / 180.0) ** 2 for a in angles])) if angles else 0.0

    total_len = polyline_length(display)
    vertical_len = 0.0
    for i in range(1, len(display)):
        dx = abs(float(display[i][0]) - float(display[i - 1][0]))
        dy = abs(float(display[i][1]) - float(display[i - 1][1]))
        seg = math.hypot(dx, dy)
        if seg >= 8.0 and dy > max(12.0, dx * 3.0):
            vertical_len += seg
    vertical_shoot_score = min(1.0, vertical_len / max(1.0, total_len))

    depth_jump_count = 0
    if polyline_3d:
        zs = [
            _float(p[2], -1.0)
            for p in polyline_3d
            if isinstance(p, Sequence) and len(p) >= 3 and _float(p[2], -1.0) > 0.0
        ]
        for i in range(1, len(zs)):
            if abs(zs[i] - zs[i - 1]) >= 0.35:
                depth_jump_count += 1

    support_stats = _support_snap_stats(raw, polyline_3d or [])
    feasible_fraction, feasible_crossings = _valid_fraction(display, feasible_mask)
    support_fraction, support_crossings = _valid_fraction(display, support_mask)
    boundary = boundary_trace or {}
    kin = list(kinematic_signatures or [])
    explained_vertical = bool(kin) or str(boundary.get("boundary_interaction", "")).lower() in {"stairs", "portal", "occlusion_transition"}

    max_snap = _float(getattr(cfg, "path_geometry_max_support_snap_px", 32.0), 32.0) if cfg else 32.0
    max_turn = _float(getattr(cfg, "path_geometry_max_turn_angle_p95_deg", 132.0), 132.0) if cfg else 132.0
    max_zigzag = _float(getattr(cfg, "path_geometry_max_zigzag_score", 0.58), 0.58) if cfg else 0.58
    max_vertical = _float(getattr(cfg, "path_geometry_max_vertical_shoot_score", 0.34), 0.34) if cfg else 0.34
    min_feasible = _float(getattr(cfg, "path_geometry_min_feasible_fraction", 0.82), 0.82) if cfg else 0.82

    rejection: List[str] = []
    if len(display) < 2:
        rejection.append("geometry_missing_display_polyline")
    if zigzag_score > max_zigzag:
        rejection.append("geometry_high_zigzag")
    if p95 > max_turn and not explained_vertical:
        rejection.append("geometry_sharp_turns_unexplained")
    if vertical_shoot_score > max_vertical and not explained_vertical:
        rejection.append("geometry_vertical_shoot_unexplained")
    if support_stats["max_px"] > max_snap:
        rejection.append("geometry_support_snap_displacement_too_large")
    if depth_jump_count > 0 and not explained_vertical:
        rejection.append("geometry_depth_jump_unexplained")
    if feasible_mask is not None and feasible_fraction < min_feasible:
        rejection.append("geometry_low_feasible_fraction")
    if support_mask is not None and support_fraction < 0.45:
        rejection.append("geometry_low_support_fraction")

    smoothability = "accepted" if not rejection else "rejected"
    return {
        "schema": "citv_path_geometry_quality_v1",
        "zigzag_score": round(float(zigzag_score), 4),
        "turn_angle_p95": round(float(p95), 3),
        "curvature_energy": round(float(curvature), 5),
        "vertical_shoot_score": round(float(vertical_shoot_score), 4),
        "depth_jump_count": int(depth_jump_count),
        "support_snap_displacement_px": {
            "mean": round(float(support_stats["mean_px"]), 3),
            "max": round(float(support_stats["max_px"]), 3),
        },
        "mask_boundary_crossing_count": int(feasible_crossings + support_crossings),
        "terrain_transition_mismatch": bool((depth_jump_count > 0 or vertical_shoot_score > max_vertical) and not explained_vertical),
        "display_feasible_fraction": round(float(feasible_fraction), 4),
        "display_support_fraction": round(float(support_fraction), 4),
        "smoothability_status": smoothability,
        "geometry_rejection_reasons": sorted(set(rejection)),
    }


__all__ = [
    "build_display_geometry",
    "clean_polyline",
    "evaluate_geometry_quality",
    "polyline_length",
    "simplify_polyline",
    "smooth_polyline",
]
