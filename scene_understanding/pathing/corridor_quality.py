"""Along-polyline corridor statistics for rejecting weak region sequences."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def polyline_corridor_stats(
    pts: List[Tuple[int, int]],
    cost_map: np.ndarray,
    path_walkable: Optional[np.ndarray],
    w: int,
    h: int,
    samples_per_seg: int = 30,
) -> Dict[str, float]:
    """Mean cost and fraction of samples not on path_walkable (when provided)."""
    cm = np.clip(np.asarray(cost_map, dtype=np.float32), 0.0, 1.0)
    ww = np.asarray(path_walkable, dtype=bool) if path_walkable is not None else None
    if ww is not None and ww.shape[:2] != (h, w):
        ww = None
    total = 0
    cost_sum = 0.0
    off_walk = 0
    for p0, p1 in zip(pts, pts[1:]):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        n = max(2, int(samples_per_seg))
        for t in range(n):
            a = t / float(n - 1) if n > 1 else 0.0
            xi = int(round((1.0 - a) * x0 + a * x1))
            yi = int(round((1.0 - a) * y0 + a * y1))
            if xi < 0 or yi < 0 or xi >= w or yi >= h:
                continue
            total += 1
            cost_sum += float(cm[yi, xi])
            if ww is not None and not bool(ww[yi, xi]):
                off_walk += 1
    mean_cost = float(cost_sum / max(1, total))
    off_ratio = float(off_walk / max(1, total))
    return {"mean_cost": mean_cost, "walkable_off_ratio": off_ratio, "samples": float(total)}


def corridor_rejects(
    pts: List[Tuple[int, int]],
    cost_map: np.ndarray,
    path_walkable: Optional[np.ndarray],
    w: int,
    h: int,
    cfg: Any,
) -> Tuple[bool, str, Dict[str, float]]:
    mc_thr = float(getattr(cfg, "path_sequence_reject_mean_cost_above", 0.0)) if cfg else 0.0
    wr_thr = float(getattr(cfg, "path_sequence_reject_walkable_off_ratio_above", 0.0)) if cfg else 0.0
    if mc_thr <= 1e-9 and wr_thr <= 1e-9:
        return False, "", {}
    n_samp = int(getattr(cfg, "path_corridor_samples_per_seg", 30)) if cfg else 30
    n_samp = max(2, n_samp)
    st = polyline_corridor_stats(pts, cost_map, path_walkable, w, h, samples_per_seg=n_samp)
    if mc_thr > 1e-9 and st["mean_cost"] > mc_thr:
        return True, "corridor_mean_cost", st
    if wr_thr > 1e-9 and st["walkable_off_ratio"] > wr_thr:
        return True, "corridor_walkable_off_ratio", st
    return False, "", st
