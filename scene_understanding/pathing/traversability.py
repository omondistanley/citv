"""
Depth- and region-aware traversability for path refinement (geodesic on a speed field).

Moved from ``scene_path_traversability.py`` so reviewer docs and imports can cite
``scene_understanding.pathing`` only. Root ``scene_path_traversability`` re-exports
for backward compatibility.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..action_ontology import load_action_ontology, prompt_bank
from ..evidence import normalize_label

def is_soft_obstacle(label: str, ontology: Optional[Dict[str, Any]] = None, cfg: Any = None) -> bool:
    """Return True for jumpable/climbable objects — should be cost spikes, not hard exclusions."""
    onto = ontology or load_action_ontology(cfg)
    bank = prompt_bank(onto, "role_prompts")
    terms = {normalize_label(v) for v in bank.get("soft_obstacle", []) if normalize_label(v)}
    label_n = normalize_label(label)
    if not label_n:
        return False
    return any(term == label_n or term in label_n or label_n in term for term in terms)


def build_traversability_speed_map(
    metric_depth_m: Optional[np.ndarray],
    region_label_map: np.ndarray,
    obstacle_mask: np.ndarray,
    img_bgr: np.ndarray,
    cfg: Any,
    K: Optional[Dict[str, float]] = None,
    *,
    soft_obstacle_mask: Optional[np.ndarray] = None,
    actor_type: str = "ground",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build traversability speed map (high = easy traversal).

    Phase 2.2: ``soft_obstacle_mask`` marks jumpable/climbable objects.
    These are *not* excluded from feasibility — instead they receive a
    strong speed penalty (×0.1) so FMM routes around them when possible
    but can still traverse them at high cost.

    Phase 3.2: ``actor_type`` selects the cost channel:
    - ``"ground"`` (default): standard depth-flatness + edge cost.
    - ``"aerial"``: ignores ground constraints, treats all labelled regions as equally feasible.
    - ``"fluid"``: applies extra penalty for non-flat, high-gradient regions.
    """
    lm = np.asarray(region_label_map, dtype=np.int32)
    obs = np.asarray(obstacle_mask, dtype=bool)
    h, w = lm.shape[:2]
    if actor_type == "aerial":
        # Aerial actors ignore ground obstacles and depth constraints.
        feasible = lm > 0
    else:
        feasible = (lm > 0) & (~obs)

    gray = cv2.cvtColor(np.asarray(img_bgr), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    gmax = float(np.max(grad))
    edge_norm = (grad / gmax) if gmax > 1e-6 else np.zeros_like(grad, dtype=np.float32)
    w_edge = float(getattr(cfg, "trav_weight_image_edge", 0.25)) if cfg else 0.25
    smooth = np.clip(1.0 - w_edge * edge_norm, 0.0, 1.0).astype(np.float32)

    flat = np.ones((h, w), dtype=np.float32)
    if metric_depth_m is not None:
        dm = np.asarray(metric_depth_m, dtype=np.float32)
        finite = np.isfinite(dm) & (dm > 1e-6)
        if finite.any():
            gx_d = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
            gy_d = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
            
            # Phase 2.1: True 2.5D Slope calculation
            if K is not None:
                fx = K.get("fx", 1000.0)
                fy = K.get("fy", 1000.0)
                slope_x = gx_d * fx / np.maximum(dm, 1e-3)
                slope_y = gy_d * fy / np.maximum(dm, 1e-3)
                gmd = np.hypot(slope_x, slope_y)
                gmd = np.where(finite, gmd, 0.0)
                slope_penalty = np.clip(gmd, 0.0, 2.0) / 2.0
                flat = np.exp(-slope_penalty * 4.0).astype(np.float32)
                flat = np.where(finite, flat, 0.15)
            else:
                gmd = cv2.magnitude(gx_d, gy_d)
                gmd = np.where(finite, gmd, 0.0)
                med = float(np.percentile(gmd[finite], 50.0))
                sigma = float(getattr(cfg, "trav_depth_grad_sigma_m", 0.35)) if cfg else 0.35
                flat = np.exp(-gmd / max(med, sigma)).astype(np.float32)
                flat = np.where(finite, flat, 0.15)

    w_flat = float(getattr(cfg, "trav_weight_depth_flatness", 0.55)) if cfg else 0.55
    w_smooth = float(getattr(cfg, "trav_weight_image_smooth", 0.45)) if cfg else 0.45
    ws = max(1e-6, w_flat + w_smooth)
    w_flat, w_smooth = w_flat / ws, w_smooth / ws
    speed = (w_flat * flat + w_smooth * smooth) * feasible.astype(np.float32)

    # Phase 3.2: aerial actor ignores geometry constraints — flat speed field.
    if actor_type == "aerial":
        speed = np.where(feasible, 0.85, 0.0).astype(np.float32)
    elif actor_type == "fluid":
        # Fluid actors prefer flat/low-gradient terrain; extra penalty on slopes.
        gx_f = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
        gy_f = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
        slope_f = np.clip(cv2.magnitude(gx_f, gy_f), 0.0, 1.0)
        speed = np.clip(speed * (1.0 - 0.6 * slope_f), 0.0, 1.0).astype(np.float32)

    # Phase 2.2: soft obstacles get a strong speed penalty instead of hard exclusion.
    if soft_obstacle_mask is not None:
        soft = np.asarray(soft_obstacle_mask, dtype=bool)
        if soft.shape[:2] == (h, w):
            speed = np.where(soft & feasible, speed * 0.1, speed).astype(np.float32)

    s_min = float(getattr(cfg, "trav_speed_floor", 0.06)) if cfg else 0.06
    speed = np.clip(speed, s_min, 1.0).astype(np.float32)
    meta = {
        "feasible_ratio": float(feasible.mean()) if feasible.size else 0.0,
        "speed_mean": float(np.mean(speed[feasible])) if feasible.any() else 0.0,
        "actor_type": actor_type,
        "soft_obstacles_applied": soft_obstacle_mask is not None,
    }
    return speed, meta


def grid_dijkstra_path(
    speed_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    inv_speed_clip: Tuple[float, float] = (1.0, 200.0),
) -> List[Tuple[int, int]]:
    """8-connected Dijkstra; edge length ~ mean(inv_speed) * geometric step."""
    sm = np.asarray(speed_map, dtype=np.float32)
    h, w = sm.shape[:2]
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return []
    lo, hi = inv_speed_clip
    inv = 1.0 / np.clip(sm, 1.0 / hi, 1.0 / lo)
    pq: List[Tuple[float, int, int]] = [(0.0, sx, sy)]
    dist: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    seen: set[Tuple[int, int]] = set()
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    while pq:
        d, x, y = heapq.heappop(pq)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if (x, y) == (gx, gy):
            break
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            step = 1.41421356 if (dx != 0 and dy != 0) else 1.0
            nd = d + step * 0.5 * (float(inv[y, x]) + float(inv[ny, nx]))
            key = (nx, ny)
            if nd < dist.get(key, 1e30):
                dist[key] = nd
                prev[key] = (x, y)
                heapq.heappush(pq, (nd, nx, ny))
    if (gx, gy) not in prev and (gx, gy) != (sx, sy):
        return []
    path = [(gx, gy)]
    cur = (gx, gy)
    while cur in prev:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return [(int(a), int(b)) for a, b in path]


def k_diverse_grid_paths(
    speed_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    k: int,
    edge_penalty: float,
) -> List[List[Tuple[int, int]]]:
    """K diverse paths by inflating inverse-speed along previously chosen routes."""
    base = np.asarray(speed_map, dtype=np.float32)
    penalty = np.ones_like(base, dtype=np.float32)
    out: List[List[Tuple[int, int]]] = []
    pen = float(edge_penalty)
    for _ in range(max(1, int(k))):
        sm = base / penalty
        p = grid_dijkstra_path(sm, start, goal)
        if len(p) < 2:
            break
        if p in out:
            break
        out.append(p)
        for x, y in p:
            yy, xx = int(y), int(x)
            penalty[yy, xx] *= 1.0 + pen
    return out


def heading_from_depth_at(metric_depth_m: Optional[np.ndarray], x: float, y: float, window: int = 9) -> Optional[float]:
    """Heading (rad) perpendicular to local depth gradient (walk along iso-depth)."""
    if metric_depth_m is None:
        return None
    dm = np.asarray(metric_depth_m, dtype=np.float32)
    h, w = dm.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    r = max(1, int(window) // 2)
    x1, y1 = max(0, xi - r), max(0, yi - r)
    x2, y2 = min(w, xi + r + 1), min(h, yi + r + 1)
    patch = dm[y1:y2, x1:x2]
    if patch.size < 4 or not np.isfinite(patch).any():
        return None
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    cx, cy = xi - x1, yi - y1
    cx = max(0, min(gx.shape[1] - 1, cx))
    cy = max(0, min(gx.shape[0] - 1, cy))
    gxv = float(gx[cy, cx])
    gyv = float(gy[cy, cx])
    n = math.hypot(gxv, gyv)
    if n < 1e-6:
        return None
    tx, ty = -gyv / n, gxv / n
    return math.atan2(ty, tx)
