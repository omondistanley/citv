"""Fast Marching Method planner over the fused semantic/geometry/physics cost
field from ``semantic_cost.py``.

Replaces a hard-obstacle-only Dijkstra search with a continuous speed field
(``skfmm.travel_time``): every pixel gets a traversal speed derived from cost
and precision, one arrival-time field ``T`` is solved per goal, and multiple
diverse routes from different sources reuse that single solve. Falls back to
the package's existing ``traversability.grid_dijkstra_path``/``k_diverse_grid_paths``
when ``scikit-fmm`` isn't installed, so callers never hard-fail for a missing
optional dependency.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import skfmm
    _HAS_SKFMM = True
except ImportError:  # pragma: no cover - optional dependency
    skfmm = None
    _HAS_SKFMM = False

from scene_understanding.pathing.traversability import grid_dijkstra_path, k_diverse_grid_paths

Point = Tuple[int, int]


def has_fmm() -> bool:
    return _HAS_SKFMM


def speed_from_cost(cost: np.ndarray, precision: Optional[np.ndarray] = None, speed_floor: float = 1e-3) -> np.ndarray:
    """Blend cost toward an uninformative 0.5 wherever precision is low, then
    invert to a traversal speed (high speed = easy to cross)."""
    c = np.asarray(cost, dtype=np.float64)
    if precision is not None:
        p = np.clip(np.asarray(precision, dtype=np.float64), 0.0, 1.0)
        c = p * c + (1.0 - p) * 0.5
    speed = np.clip(1.0 - c, speed_floor, 1.0)
    return speed.astype(np.float64)


def time_of_arrival(speed: np.ndarray, goal: Point) -> Optional[np.ndarray]:
    """Solve the Eikonal equation for arrival time to ``goal`` under ``speed``."""
    if not _HAS_SKFMM:
        return None
    h, w = speed.shape[:2]
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    phi = np.ones((h, w), dtype=np.float64)
    phi[gy, gx] = -1.0
    try:
        return skfmm.travel_time(phi, speed, dx=1.0)
    except Exception:
        return None


def _descend_gradient(T: np.ndarray, start: Point, max_steps: int = 5000) -> List[Point]:
    """8-neighbor steepest-descent walk on the arrival-time field back to the goal."""
    h, w = T.shape[:2]
    x, y = int(start[0]), int(start[1])
    path = [(x, y)]
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    for _ in range(max_steps):
        cur = T[y, x]
        if not np.isfinite(cur) or cur <= 0:
            break
        best = None
        best_val = cur
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and np.isfinite(T[ny, nx]) and T[ny, nx] < best_val:
                best_val = T[ny, nx]
                best = (nx, ny)
        if best is None:
            break
        x, y = best
        path.append((x, y))
    return path


def fmm_backtrace(T: np.ndarray, start: Point) -> List[Point]:
    return _descend_gradient(T, start)


def backtrace_from_T(T: np.ndarray, start: Point) -> List[Point]:
    return fmm_backtrace(T, start)


def k_diverse_from_T(
    T: np.ndarray,
    start: Point,
    k: int,
    edge_penalty: float = 0.35,
    dilate_radius: int = 2,
) -> List[List[Point]]:
    """K diverse routes from ONE precomputed arrival-time field: after each
    backtrace, inflate T along a dilated version of the chosen path so the
    next descent finds an adjacent valley instead of retracing it."""
    import cv2

    working = T.copy()
    out: List[List[Point]] = []
    for _ in range(max(1, int(k))):
        path = _descend_gradient(working, start)
        if len(path) < 2:
            break
        out.append(path)
        mask = np.zeros(T.shape[:2], dtype=np.uint8)
        for x, y in path:
            mask[y, x] = 1
        if dilate_radius > 0:
            kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), np.uint8)
            mask = cv2.dilate(mask, kernel)
        working = np.where(mask > 0, working * (1.0 + edge_penalty), working)
    return out


def k_diverse_fmm_paths(
    speed: np.ndarray,
    start: Point,
    goal: Point,
    k: int,
    edge_penalty: float = 0.35,
) -> List[List[Point]]:
    """Re-solve per alternate (additive cost bump instead of reusing one T);
    used when routes need to diverge earlier than a single T's local geometry allows."""
    working_speed = speed.copy()
    out: List[List[Point]] = []
    for _ in range(max(1, int(k))):
        T = time_of_arrival(working_speed, goal)
        if T is None:
            break
        path = _descend_gradient(T, start)
        if len(path) < 2 or path in out:
            break
        out.append(path)
        for x, y in path:
            working_speed[y, x] = max(1e-3, working_speed[y, x] * (1.0 - edge_penalty))
    return out


def theta_star_on_fmm(path: List[Point], cost: np.ndarray, line_of_sight_tol: float = 0.05) -> List[Point]:
    """Any-angle shortcut: drop intermediate vertices whose direct line has
    mean cost within ``line_of_sight_tol`` of the reference path segment."""
    if len(path) < 3:
        return path
    c = np.asarray(cost, dtype=np.float32)
    h, w = c.shape[:2]

    def _line_mean_cost(p0: Point, p1: Point) -> float:
        n = max(2, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        xs = np.linspace(p0[0], p1[0], n).astype(int).clip(0, w - 1)
        ys = np.linspace(p0[1], p1[1], n).astype(int).clip(0, h - 1)
        return float(np.mean(c[ys, xs]))

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        ref_cost = _line_mean_cost(path[i], path[i + 1])
        while j > i + 1:
            direct_cost = _line_mean_cost(path[i], path[j])
            if direct_cost <= ref_cost + line_of_sight_tol:
                break
            j -= 1
        out.append(path[j])
        i = j
    return out


def multiscale_fmm_plan(
    speed: np.ndarray,
    start: Point,
    goal: Point,
    coarse_factor: int = 4,
    refine_margin_px: int = 24,
) -> Optional[List[Point]]:
    """Coarse full-frame FMM solve to get an approximate route, then a full-resolution
    solve restricted to a local bbox around that route (cheaper than full-res everywhere)."""
    import cv2

    h, w = speed.shape[:2]
    ch, cw = max(1, h // coarse_factor), max(1, w // coarse_factor)
    coarse_speed = cv2.resize(speed.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
    coarse_goal = (max(0, min(cw - 1, goal[0] // coarse_factor)), max(0, min(ch - 1, goal[1] // coarse_factor)))
    coarse_start = (max(0, min(cw - 1, start[0] // coarse_factor)), max(0, min(ch - 1, start[1] // coarse_factor)))
    T_coarse = time_of_arrival(coarse_speed.astype(np.float64), coarse_goal)
    if T_coarse is None:
        return None
    coarse_path = _descend_gradient(T_coarse, coarse_start)
    if len(coarse_path) < 2:
        return None

    xs = [p[0] * coarse_factor for p in coarse_path]
    ys = [p[1] * coarse_factor for p in coarse_path]
    x0 = max(0, min(xs) - refine_margin_px)
    x1 = min(w, max(xs) + refine_margin_px)
    y0 = max(0, min(ys) - refine_margin_px)
    y1 = min(h, max(ys) + refine_margin_px)
    local_speed = speed[y0:y1, x0:x1]
    local_start = (start[0] - x0, start[1] - y0)
    local_goal = (goal[0] - x0, goal[1] - y0)
    T_fine = time_of_arrival(local_speed, local_goal)
    if T_fine is None:
        return [(x + x0, y + y0) for x, y in coarse_path]
    fine_path = _descend_gradient(T_fine, local_start)
    return [(x + x0, y + y0) for x, y in fine_path]


def plan_path(
    algorithm: str,
    speed_or_cost: np.ndarray,
    start: Point,
    goal: Point,
    precision: Optional[np.ndarray] = None,
) -> List[Point]:
    """Single dispatch entrypoint: ``'fmm'`` (falls back to Dijkstra if skfmm
    is missing), or ``'dijkstra'`` explicitly."""
    if algorithm == "dijkstra" or not _HAS_SKFMM:
        speed_map = speed_or_cost if algorithm == "dijkstra" else speed_from_cost(speed_or_cost, precision)
        return grid_dijkstra_path(speed_map, start, goal)
    speed = speed_from_cost(speed_or_cost, precision) if precision is not None else speed_or_cost
    T = time_of_arrival(speed, goal)
    if T is None:
        return grid_dijkstra_path(speed, start, goal)
    return fmm_backtrace(T, start)


def time_of_arrival_from_speed(speed_map: np.ndarray, goal: Point) -> Optional[np.ndarray]:
    """Alias kept for call sites that already have a prebuilt speed map
    (e.g. reused across multiple sources sharing one goal)."""
    return time_of_arrival(speed_map, goal)


def k_diverse_paths(
    speed_or_cost: np.ndarray,
    start: Point,
    goal: Point,
    k: int,
    edge_penalty: float = 0.35,
    precision: Optional[np.ndarray] = None,
) -> List[List[Point]]:
    """K-diverse dispatch mirroring ``plan_path``'s fallback behavior."""
    if not _HAS_SKFMM:
        return k_diverse_grid_paths(speed_or_cost, start, goal, k, edge_penalty)
    speed = speed_from_cost(speed_or_cost, precision) if precision is not None else speed_or_cost
    T = time_of_arrival(speed, goal)
    if T is None:
        return k_diverse_grid_paths(speed, start, goal, k, edge_penalty)
    return k_diverse_from_T(T, start, k, edge_penalty)
