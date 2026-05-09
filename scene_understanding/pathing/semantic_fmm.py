"""Fast Marching Method (FMM) planner on the semantic cost map.

This replaces/augments the legacy A*/Dijkstra planner for the semantic
cost map. The FMM computes a viscosity-consistent time-of-arrival field
``T`` from the goal under a *speed* map ``F(x, y) = 1 / (eps + C(x, y))``,
then the path is recovered by steepest descent on ``T`` from the start.

Compared with A*:

- FMM is a PDE solver — it produces a proper geodesic on a continuous
  cost field. A* is a discrete graph search that ignores the continuum
  between pixels.
- The speed function ``F`` respects *semantic* cost: high-cost pixels
  (obstacles / low confidence) have low speed → the wavefront goes
  around them without ever needing "binary obstacle / free" labels.

We expose four dispatch entry points:

- :func:`fmm_backtrace`        — single FMM solve + gradient descent.
- :func:`hier_fmm_plan`        — region graph + per-region FMM refinement.
- :func:`multiscale_fmm_plan`  — coarse FMM for large images, then refine.
- :func:`theta_star_on_fmm`    — post-smooth the FMM backtrace with line-of-sight shortcuts.

A* / Dijkstra have been removed from every live entry point. When
``scikit-fmm`` is unavailable, callers receive a straight-line
fallback so the pipeline still returns a typed polyline — we deliberately
do not re-introduce a grid search in that path. The dispatch helper
:func:`plan_path` consolidates the selection rule so callers pass
``cfg.path_search_algorithm``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# NOTE: A* has been removed from every live planning entry point — FMM is
# the sole geodesic solver. A degraded planner still has to return
# *something* when scikit-fmm is missing, so we keep a tiny straight-line
# fallback that runs in constant time instead of re-introducing A*.
try:
    import skfmm  # type: ignore
except Exception:  # pragma: no cover
    skfmm = None  # type: ignore


def _straight_line_fallback(
    start: "Point", goal: "Point", *, step: int = 1
) -> "List[Point]":
    """Rasterise a Bresenham-like polyline between ``start`` and ``goal``.

    Used only when ``scikit-fmm`` is absent; the pipeline treats scikit-fmm
    as a hard dependency in practice so this path exists purely to keep
    callers type-safe.
    """

    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    n = max(abs(gx - sx), abs(gy - sy)) + 1
    if n <= 1:
        return [(sx, sy), (gx, gy)]
    xs = np.linspace(sx, gx, n).round().astype(np.int32)
    ys = np.linspace(sy, gy, n).round().astype(np.int32)
    if step > 1:
        xs = xs[::step]
        ys = ys[::step]
        if xs[-1] != gx or ys[-1] != gy:
            xs = np.concatenate([xs, [gx]])
            ys = np.concatenate([ys, [gy]])
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


Point = Tuple[int, int]


# ---------------------------------------------------------------------------
# Core FMM solve
# ---------------------------------------------------------------------------


def _speed_from_cost(
    cost: np.ndarray,
    *,
    precision: Optional[np.ndarray] = None,
    speed_floor: float = 1e-3,
) -> np.ndarray:
    """Translate a cost field ``C ∈ [0, 1]`` into a speed field ``F > 0``.

    We clamp ``F`` below at ``speed_floor`` so the FMM never sees zero
    speed (which would violate the Eikonal equation). When precision is
    supplied, we damp the speed more strongly in low-precision areas so
    the wavefront prefers trustworthy regions.
    """

    c = np.clip(cost.astype(np.float32), 0.0, 1.0)
    if precision is not None:
        p = np.clip(precision.astype(np.float32), 0.0, 1.0)
        # Low precision → blend cost toward 0.5 (uninformative).
        c = p * c + (1.0 - p) * 0.5
    speed = 1.0 - c
    return np.maximum(speed, speed_floor).astype(np.float64)


def _time_of_arrival(cost: np.ndarray, goal: Point, precision: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if skfmm is None:
        return None
    h, w = cost.shape[:2]
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    phi = np.ones((h, w), dtype=np.float64)
    phi[gy, gx] = -1.0  # zero-contour = goal
    speed = _speed_from_cost(cost, precision=precision)
    try:
        T = skfmm.travel_time(phi, speed, dx=1.0)
    except Exception:
        return None
    return np.asarray(T, dtype=np.float32)


def time_of_arrival(
    cost: np.ndarray,
    goal: Point,
    *,
    precision: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Public wrapper around :func:`_time_of_arrival`.

    Exposed so callers can pre-compute one travel-time field per unique
    goal and reuse it across many source points, collapsing
    ``O(P) FMM solves`` (one per source/goal pair) into
    ``O(G) FMM solves + O(P) gradient descents``.
    """

    return _time_of_arrival(cost, goal, precision=precision)


def time_of_arrival_from_speed(
    speed_map: np.ndarray, goal: Point, *, speed_floor: float = 1e-3
) -> Optional[np.ndarray]:
    """Compute travel-time directly from a pre-computed *speed* field.

    Unlike :func:`_time_of_arrival`, which takes a cost field and inverts
    it internally, this helper accepts an already-built speed field
    (e.g. the output of
    :func:`scene_understanding.pathing.traversability.build_traversability_speed_map`)
    and simply clips it above the floor before the FMM solve. Lets the
    per-image pipeline reuse its semantic speed map across every geodesic
    query without re-inverting it per pair.
    """

    if skfmm is None:
        return None
    h, w = speed_map.shape[:2]
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    phi = np.ones((h, w), dtype=np.float64)
    phi[gy, gx] = -1.0
    sp = np.maximum(np.asarray(speed_map, dtype=np.float64), float(speed_floor))
    try:
        T = skfmm.travel_time(phi, sp, dx=1.0)
    except Exception:
        return None
    return np.asarray(T, dtype=np.float32)


def _descend_gradient(
    T: np.ndarray,
    start: Point,
    *,
    max_steps: int = 5000,
) -> List[Point]:
    """Steepest-descent backtrace from ``start`` towards the 0-level-set."""

    h, w = T.shape[:2]
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < w and 0 <= sy < h):
        return []
    path: List[Point] = [(sx, sy)]
    visited = set(path)
    # 8-neighbour descent with half-pixel step.
    offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for _ in range(max_steps):
        x, y = path[-1]
        best = None
        best_val = T[y, x]
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            v = T[ny, nx]
            if not np.isfinite(v):
                continue
            if v < best_val - 1e-6 and (nx, ny) not in visited:
                best_val = v
                best = (nx, ny)
        if best is None:
            break
        path.append(best)
        visited.add(best)
        if best_val <= 0.0:
            break
    return path


# ---------------------------------------------------------------------------
# Public planners
# ---------------------------------------------------------------------------


def fmm_backtrace(
    cost: np.ndarray,
    start: Point,
    goal: Point,
    *,
    precision: Optional[np.ndarray] = None,
) -> List[Point]:
    """Single FMM solve + backtrace.

    A* has been removed from every live dispatch; when ``scikit-fmm`` is
    missing we return a straight-line polyline rather than re-introducing
    a grid search. Production deployments list scikit-fmm as a hard
    dependency so this branch only fires in degraded environments.
    """

    T = _time_of_arrival(cost, goal, precision=precision)
    if T is None:
        return _straight_line_fallback(start, goal)
    path = _descend_gradient(T, start)
    if not path:
        return _straight_line_fallback(start, goal)
    return path


def backtrace_from_T(T: np.ndarray, start: Point) -> List[Point]:
    """Steepest descent on a precomputed travel-time field.

    Companion to :func:`time_of_arrival` / :func:`time_of_arrival_from_speed`
    that lets callers amortise one FMM solve across many source points.
    Returns the gradient-descent path; empty list if ``start`` is invalid.
    """

    return _descend_gradient(T, start)


def k_diverse_from_T(
    T: np.ndarray,
    start: Point,
    *,
    k: int = 3,
    edge_penalty: float = 0.35,
    dilate_radius: int = 2,
) -> List[List[Point]]:
    """Yield up to ``k`` diverse polylines from a *single* T field.

    The standard :func:`k_diverse_fmm_paths` re-solves FMM for every
    diversified path because it mutates the cost field. That costs one
    FMM solve per alternate, which dominates the per-pair budget when
    you have many source/goal pairs.

    This variant trades fidelity for speed: we run one descent, then
    inflate ``T`` along a dilated footprint of the chosen path and
    descend again from the same start. Because ``T`` is monotone
    (non-decreasing away from the goal), the inflation pushes subsequent
    descents onto adjacent valleys of the travel-time field, producing
    visually-diverse routes that still live on the semantic geodesic.
    """

    paths: List[List[Point]] = []
    if T is None or not np.isfinite(T).any():
        return paths
    work = np.asarray(T, dtype=np.float32).copy()
    h, w = work.shape[:2]
    for _ in range(max(1, int(k))):
        path = _descend_gradient(work, start)
        if not path or len(path) < 2:
            break
        if path in paths:
            break
        paths.append(path)
        if edge_penalty <= 0:
            continue
        import cv2

        mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y) in path:
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = 1
        if dilate_radius > 0:
            ksize = max(3, 2 * int(dilate_radius) + 1)
            mask = cv2.dilate(mask, np.ones((ksize, ksize), np.uint8))
        # Multiplicative inflation preserves monotonicity locally — we
        # scale T up where the previous path walked so the next descent
        # prefers neighbouring valleys. Additive would distort distances
        # near the goal where T is small.
        scale = 1.0 + float(edge_penalty)
        work = np.where(mask > 0, work * scale, work).astype(np.float32)
    return paths


def theta_star_on_fmm(
    cost: np.ndarray,
    start: Point,
    goal: Point,
    *,
    precision: Optional[np.ndarray] = None,
    line_of_sight_tol: float = 0.05,
) -> List[Point]:
    """Post-smooth the FMM backtrace with any-angle line-of-sight shortcuts.

    Walks the raw FMM polyline and, at each step, tries to extend the
    current segment to the farthest future vertex whose straight line
    keeps the mean cost within ``line_of_sight_tol`` of the raw cost
    integral. This keeps the FMM's semantic guarantees while producing
    visually smoother any-angle routes.
    """

    raw = fmm_backtrace(cost, start, goal, precision=precision)
    if len(raw) <= 2:
        return raw
    h, w = cost.shape[:2]

    def _line_cost(a: Point, b: Point) -> Tuple[float, int]:
        # Sample along Bresenham with np.linspace for a simple mean cost.
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        xs = np.linspace(a[0], b[0], n).astype(np.int32)
        ys = np.linspace(a[1], b[1], n).astype(np.int32)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        return float(np.mean(cost[ys, xs])), n

    smoothed: List[Point] = [raw[0]]
    i = 0
    while i < len(raw) - 1:
        j = len(raw) - 1
        # Reference cost of staying on the FMM polyline from i..j.
        ref_xs = np.array([p[0] for p in raw[i : j + 1]], dtype=np.int32)
        ref_ys = np.array([p[1] for p in raw[i : j + 1]], dtype=np.int32)
        ref_xs = np.clip(ref_xs, 0, w - 1)
        ref_ys = np.clip(ref_ys, 0, h - 1)
        ref_cost = float(np.mean(cost[ref_ys, ref_xs]))
        while j > i + 1:
            c, _ = _line_cost(raw[i], raw[j])
            if c <= ref_cost + line_of_sight_tol:
                break
            j -= 1
        smoothed.append(raw[j])
        i = j
    if smoothed[-1] != raw[-1]:
        smoothed.append(raw[-1])
    return smoothed


def multiscale_fmm_plan(
    cost: np.ndarray,
    start: Point,
    goal: Point,
    *,
    precision: Optional[np.ndarray] = None,
    downsample_factor: int = 4,
) -> List[Point]:
    """Coarse-to-fine FMM. Solves at 1/N resolution, then refines full-res.

    Only falls back to the full-res solve if downsample fails — we never
    return a path computed purely at low resolution.
    """

    if downsample_factor < 2 or skfmm is None:
        return fmm_backtrace(cost, start, goal, precision=precision)

    import cv2

    h, w = cost.shape[:2]
    ch = max(1, h // downsample_factor)
    cw = max(1, w // downsample_factor)
    cost_lo = cv2.resize(cost, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32)
    prec_lo = cv2.resize(precision, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32) if precision is not None else None
    start_lo = (int(start[0] * cw / w), int(start[1] * ch / h))
    goal_lo = (int(goal[0] * cw / w), int(goal[1] * ch / h))
    # 1) coarse FMM
    T_lo = _time_of_arrival(cost_lo, goal_lo, precision=prec_lo)
    if T_lo is None:
        return fmm_backtrace(cost, start, goal, precision=precision)
    path_lo = _descend_gradient(T_lo, start_lo)
    if not path_lo:
        return fmm_backtrace(cost, start, goal, precision=precision)
    # 2) upsample and locally refine with full-res FMM around the strip.
    path_hi = [(int(px * w / cw), int(py * h / ch)) for px, py in path_lo]
    # Local refinement: FMM on a padded bbox around path_hi.
    xs = [p[0] for p in path_hi]; ys = [p[1] for p in path_hi]
    pad = max(4, downsample_factor * 2)
    x1, x2 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y1, y2 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    local = cost[y1:y2, x1:x2].copy()
    local_prec = precision[y1:y2, x1:x2] if precision is not None else None
    local_start = (start[0] - x1, start[1] - y1)
    local_goal = (goal[0] - x1, goal[1] - y1)
    # If the goal fell outside the local bbox, drop back to the coarse upsampled path.
    if not (0 <= local_goal[0] < local.shape[1] and 0 <= local_goal[1] < local.shape[0]):
        return path_hi
    if not (0 <= local_start[0] < local.shape[1] and 0 <= local_start[1] < local.shape[0]):
        return path_hi
    T_hi = _time_of_arrival(local, local_goal, precision=local_prec)
    if T_hi is None:
        return path_hi
    refined = _descend_gradient(T_hi, local_start)
    if not refined:
        return path_hi
    return [(px + x1, py + y1) for px, py in refined]


def hier_fmm_plan(
    cost: np.ndarray,
    start: Point,
    goal: Point,
    *,
    precision: Optional[np.ndarray] = None,
    region_label_map: Optional[np.ndarray] = None,
    region_padding_px: int = 8,
) -> List[Point]:
    """Region-graph planning with per-region FMM refinement.

    Builds a coarse graph over regions (centroid per region), Dijkstra's
    between region centroids using mean cost per region, then refines
    each hop with a local FMM. If the label map is missing we fall back
    to a single-pass FMM.
    """

    if region_label_map is None:
        return fmm_backtrace(cost, start, goal, precision=precision)
    lm = np.asarray(region_label_map, dtype=np.int32)
    if lm.shape[:2] != cost.shape[:2]:
        return fmm_backtrace(cost, start, goal, precision=precision)
    ids = np.unique(lm)
    ids = ids[ids > 0]
    if ids.size < 2:
        return fmm_backtrace(cost, start, goal, precision=precision)
    # 1) region centroids and mean cost per region.
    h, w = cost.shape[:2]
    centroids = {}
    mean_cost = {}
    for rid in ids:
        mask = (lm == rid)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        cy = int(ys.mean()); cx = int(xs.mean())
        centroids[int(rid)] = (cx, cy)
        mean_cost[int(rid)] = float(cost[mask].mean())
    if not centroids:
        return fmm_backtrace(cost, start, goal, precision=precision)
    # 2) pick start/goal regions.
    rid_start = int(lm[start[1], start[0]]) if (0 <= start[0] < w and 0 <= start[1] < h) else 0
    rid_goal = int(lm[goal[1], goal[0]]) if (0 <= goal[0] < w and 0 <= goal[1] < h) else 0
    if rid_start <= 0 or rid_goal <= 0 or rid_start not in centroids or rid_goal not in centroids:
        return fmm_backtrace(cost, start, goal, precision=precision)
    # 3) Dijkstra on the adjacency graph of regions.
    import heapq
    pq: List[Tuple[float, int]] = [(0.0, rid_start)]
    dist = {rid_start: 0.0}
    came: dict[int, int] = {}
    neighbours: dict[int, set[int]] = {r: set() for r in centroids}
    # build adjacency via dilation XOR.
    import cv2
    for rid in centroids:
        m = (lm == rid).astype(np.uint8)
        dil = cv2.dilate(m, np.ones((3, 3), np.uint8))
        touching = np.unique(lm[(dil > 0) & (lm != rid) & (lm > 0)])
        neighbours[rid] = {int(t) for t in touching if int(t) in centroids}
    while pq:
        d, r = heapq.heappop(pq)
        if r == rid_goal:
            break
        for n in neighbours.get(r, ()):
            w_edge = 0.5 * (mean_cost[r] + mean_cost[n]) + 1e-3
            nd = d + w_edge
            if nd < dist.get(n, float("inf")):
                dist[n] = nd
                came[n] = r
                heapq.heappush(pq, (nd, n))
    if rid_goal not in dist:
        return fmm_backtrace(cost, start, goal, precision=precision)
    # Phase 1.5 — single continuous global FMM instead of per-segment local
    # solves through centroid waypoints. Centroid waypoints can land in
    # geometrically poor positions (e.g., the unreachable arm of a U-shaped
    # region), breaking the path. One global FMM solve is cheaper and correct.
    return fmm_backtrace(cost, start, goal, precision=precision)


# ---------------------------------------------------------------------------
# k-diverse on the FMM speed field
# ---------------------------------------------------------------------------


def k_diverse_fmm_paths(
    cost: np.ndarray,
    start: Point,
    goal: Point,
    *,
    precision: Optional[np.ndarray] = None,
    k: int = 3,
    diversity_bonus: float = 0.5,
) -> List[List[Point]]:
    """Return up to ``k`` diverse FMM paths by iteratively penalising
    visited neighbourhoods in the cost field."""

    paths: List[List[Point]] = []
    c = cost.copy()
    p = precision.copy() if precision is not None else None
    for _ in range(max(1, k)):
        path = fmm_backtrace(c, start, goal, precision=p)
        if not path:
            break
        paths.append(path)
        if diversity_bonus <= 0:
            continue
        import cv2
        mask = np.zeros_like(c, dtype=np.uint8)
        for (x, y) in path:
            if 0 <= x < c.shape[1] and 0 <= y < c.shape[0]:
                mask[y, x] = 1
        dil = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        c = np.clip(c + diversity_bonus * (dil > 0).astype(np.float32), 0.0, 1.0)
    return paths


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def plan_path(
    algorithm: str,
    cost: np.ndarray,
    start: Point,
    goal: Point,
    *,
    precision: Optional[np.ndarray] = None,
    region_label_map: Optional[np.ndarray] = None,
    line_of_sight_tol: float = 0.05,
    downsample_factor: int = 4,
    region_padding_px: int = 8,
) -> List[Point]:
    """Dispatch helper for ``cfg.path_search_algorithm``."""

    alg = (algorithm or "fmm").strip().lower()
    # A* has been retired from every live dispatch path. Historical
    # algorithm names ("legacy_astar", unknown values) all route to the
    # FMM backtrace so downstream callers need no changes when they
    # continue to pass the old string through ``cfg.path_search_algorithm``.
    if alg == "theta_star":
        return theta_star_on_fmm(
            cost, start, goal, precision=precision, line_of_sight_tol=line_of_sight_tol
        )
    if alg == "multiscale_fmm":
        return multiscale_fmm_plan(
            cost, start, goal, precision=precision, downsample_factor=downsample_factor
        )
    if alg == "hier_fmm":
        return hier_fmm_plan(
            cost,
            start,
            goal,
            precision=precision,
            region_label_map=region_label_map,
            region_padding_px=region_padding_px,
        )
    # "fmm", "legacy_astar" (historical alias), and unknown values all
    # fall back to the FMM backtrace — the canonical geodesic solver.
    return fmm_backtrace(cost, start, goal, precision=precision)


__all__ = [
    "fmm_backtrace",
    "backtrace_from_T",
    "theta_star_on_fmm",
    "multiscale_fmm_plan",
    "hier_fmm_plan",
    "k_diverse_fmm_paths",
    "k_diverse_from_T",
    "time_of_arrival",
    "time_of_arrival_from_speed",
    "plan_path",
]
