"""Per-pixel cost field for image-space path refinement (A*)."""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def distance_transform_centering(mask_free: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask_free).astype(np.uint8) * 255)
    if m.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    mx = float(np.max(dt))
    if mx <= 1e-6:
        return np.zeros_like(dt, dtype=np.float32)
    return dt / mx


def build_path_cost_map(
    img_bgr: np.ndarray,
    region_label_map: np.ndarray,
    obstacle_mask: np.ndarray,
    cfg: Any,
    soft_obstacle_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    gmax = float(np.max(grad))
    edge_cost = (grad / gmax) if gmax > 1e-6 else np.zeros_like(grad, dtype=np.float32)

    obs = np.asarray(obstacle_mask).astype(np.uint8)
    obs_cost = cv2.GaussianBlur(obs.astype(np.float32), (0, 0), 1.5)
    if np.max(obs_cost) > 1e-6:
        obs_cost = obs_cost / float(np.max(obs_cost))

    if soft_obstacle_mask is not None:
        sobs = np.asarray(soft_obstacle_mask).astype(np.uint8)
        sobs_cost = cv2.GaussianBlur(sobs.astype(np.float32), (0, 0), 1.5)
        if np.max(sobs_cost) > 1e-6:
            # Phase 2.2: Soft obstacles incur high cost but are not infinite walls
            sobs_cost = (sobs_cost / float(np.max(sobs_cost))) * 0.85
        obs_cost = np.maximum(obs_cost, sobs_cost)

    lm = np.asarray(region_label_map, dtype=np.int32)
    region_prior = (lm <= 0).astype(np.float32)
    if np.max(region_prior) > 1e-6:
        region_prior = region_prior / float(np.max(region_prior))

    center_bonus = distance_transform_centering((~obs.astype(bool)))
    center_cost = 1.0 - center_bonus

    we = float(getattr(cfg, "path_cost_weight_edges", 0.35)) if cfg else 0.35
    wo = float(getattr(cfg, "path_cost_weight_obstacle", 0.35)) if cfg else 0.35
    wr = float(getattr(cfg, "path_cost_weight_region_prior", 0.15)) if cfg else 0.15
    wc = float(getattr(cfg, "path_cost_weight_centering", 0.15)) if cfg else 0.15
    ws = max(1e-6, we + wo + wr + wc)
    we, wo, wr, wc = we / ws, wo / ws, wr / ws, wc / ws
    cm = (we * edge_cost) + (wo * obs_cost) + (wr * region_prior) + (wc * center_cost)
    cm = np.clip(cm, 0.0, 1.0).astype(np.float32)
    return cm


def astar_on_cost_map(cost_map: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    # Phase D.5 — prefer the flat-array A* from the numerics module. It
    # uses contiguous ``gscore`` / ``came`` buffers (``y*W + x`` flat ids)
    # instead of ``Dict[Tuple[int, int], float]``; the legacy dict-based
    # code below is kept as a strict fallback only.
    try:
        from scene_understanding.numerics import astar_shortest_path
        return astar_shortest_path(cost_map, start, goal)
    except Exception:
        pass
    h, w = cost_map.shape[:2]
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return []
    pq: List[Tuple[float, int, int]] = [(0.0, sx, sy)]
    gscore: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    while pq:
        _f, x, y = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            break
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            step = 1.4142 if (dx != 0 and dy != 0) else 1.0
            ng = gscore[(x, y)] + step * (1.0 + float(cost_map[ny, nx]))
            key = (nx, ny)
            if ng < gscore.get(key, 1e18):
                gscore[key] = ng
                came[key] = (x, y)
                hcost = math.hypot(float(gx - nx), float(gy - ny))
                heapq.heappush(pq, (ng + hcost, nx, ny))
    if (gx, gy) not in came and (gx, gy) != (sx, sy):
        return []
    path = [(gx, gy)]
    cur = (gx, gy)
    while cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return [(int(x), int(y)) for (x, y) in path]
