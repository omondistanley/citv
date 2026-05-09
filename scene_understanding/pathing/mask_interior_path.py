"""Interior shortest path inside an eroded instance mask (intra motion, not contour)."""
from __future__ import annotations

from collections import deque
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np


def _largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """Keep largest 8-connected component."""
    binary = np.asarray(binary, dtype=np.uint8)
    if not binary.any():
        return binary.astype(bool)
    num, labels = cv2.connectedComponents(binary, connectivity=8)
    if num <= 1:
        return binary.astype(bool)
    best = 1
    best_ct = 0
    for k in range(1, num):
        ct = int((labels == k).sum())
        if ct > best_ct:
            best_ct = ct
            best = k
    return (labels == best).astype(bool)


def _pick_two_seeds(m: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    ys, xs = np.where(m)
    if xs.size < 10:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    i_near = int(np.argmin(d))
    i_far = int(np.argmax(d))
    return (int(xs[i_near]), int(ys[i_near])), (int(xs[i_far]), int(ys[i_far]))


def mask_interior_geodesic_points(
    mask: np.ndarray,
    w: int,
    h: int,
    *,
    erode_iters: int = 2,
    kernel_px: int = 3,
    min_component_px: int = 400,
) -> List[Tuple[int, int]]:
    """
    BFS shortest path between two seeds in eroded mask interior.
    Returns pixel path including endpoints; empty if not feasible.
    """
    mm = np.asarray(mask, dtype=bool)
    if mm.shape[:2] != (h, w):
        mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    k = max(1, int(kernel_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    er = mm.astype(np.uint8) * 255
    for _ in range(max(0, int(erode_iters))):
        er = cv2.erode(er, kernel)
    interior = er > 0
    interior = _largest_component_mask(interior.astype(np.uint8))
    if int(interior.sum()) < int(min_component_px):
        return []
    seeds = _pick_two_seeds(interior)
    if seeds is None:
        return []
    start, goal = seeds
    if not interior[start[1], start[0]] or not interior[goal[1], goal[0]]:
        return []
    # 4-neighbor BFS
    prev: dict = {}
    q: deque = deque([start])
    prev[start] = None
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    found = False
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            found = True
            break
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if not interior[ny, nx]:
                continue
            nb = (nx, ny)
            if nb in prev:
                continue
            prev[nb] = (x, y)
            q.append(nb)
    if not found:
        return []
    out: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = goal
    while cur is not None:
        out.append(cur)
        cur = prev.get(cur)  # type: ignore[assignment]
    out.reverse()
    return out
