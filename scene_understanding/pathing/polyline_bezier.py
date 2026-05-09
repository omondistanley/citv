"""Quadratic Bézier corner rounding for piecewise-linear paths (smooth, parabolic blends)."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1]) + 1e-9
    n2 = math.hypot(v2[0], v2[1]) + 1e-9
    cos_t = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return float(math.degrees(math.acos(cos_t)))


def _norm(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy) + 1e-9
    return dx / n, dy / n


def bezier_round_polyline(
    pts: List[Tuple[int, int]],
    *,
    w: int,
    h: int,
    walkable: Optional[np.ndarray] = None,
    corner_deg_min: float = 22.0,
    handle_frac: float = 0.32,
    max_handle_px: float = 72.0,
    samples_per_corner: int = 14,
) -> List[Tuple[int, int]]:
    """
    Replace sharp corners (turn angle >= corner_deg_min) with quadratic Bézier
    segments P0 -- B -- P2 (control point B is the original vertex).
    """
    if len(pts) < 3:
        return pts
    corner_deg_min = float(max(0.0, corner_deg_min))
    handle_frac = float(max(0.05, min(0.49, handle_frac)))
    max_handle_px = float(max(4.0, max_handle_px))
    n_samples = max(4, int(samples_per_corner))

    out: List[Tuple[int, int]] = [pts[0]]
    ww = np.asarray(walkable, dtype=bool) if walkable is not None else None

    def clip_xy(x: float, y: float) -> Tuple[int, int]:
        return int(max(0, min(w - 1, round(x)))), int(max(0, min(h - 1, round(y))))

    def sample_ok(xy_list: List[Tuple[int, int]]) -> bool:
        if ww is None:
            return True
        for x, y in xy_list:
            if y < 0 or x < 0 or y >= ww.shape[0] or x >= ww.shape[1] or not bool(ww[y, x]):
                return False
        return True

    for i in range(1, len(pts) - 1):
        a = (float(out[-1][0]), float(out[-1][1]))
        b = (float(pts[i][0]), float(pts[i][1]))
        c = (float(pts[i + 1][0]), float(pts[i + 1][1]))
        ang = _angle_deg(a, b, c)
        if ang < corner_deg_min - 1e-6:
            p = clip_xy(b[0], b[1])
            if not out or out[-1] != p:
                out.append(p)
            continue

        d_ba = math.hypot(b[0] - a[0], b[1] - a[1])
        d_bc = math.hypot(c[0] - b[0], c[1] - b[1])
        d = min(max_handle_px, handle_frac * min(d_ba, d_bc))
        uax, uay = _norm(a[0] - b[0], a[1] - b[1])
        ucx, ucy = _norm(c[0] - b[0], c[1] - b[1])
        p0 = (b[0] + d * uax, b[1] + d * uay)
        p2 = (b[0] + d * ucx, b[1] + d * ucy)

        seg: List[Tuple[int, int]] = []
        for t in range(n_samples + 1):
            s = t / float(n_samples)
            oms = 1.0 - s
            x = oms * oms * p0[0] + 2 * oms * s * b[0] + s * s * p2[0]
            y = oms * oms * p0[1] + 2 * oms * s * b[1] + s * s * p2[1]
            seg.append(clip_xy(x, y))

        if not sample_ok(seg):
            # shorten handle until feasible or fall back to corner point
            ok_seg = None
            for scale in (0.65, 0.4, 0.22, 0.0):
                dd = d * scale
                p0s = (b[0] + dd * uax, b[1] + dd * uay)
                p2s = (b[0] + dd * ucx, b[1] + dd * ucy)
                seg_try: List[Tuple[int, int]] = []
                for t in range(n_samples + 1):
                    s = t / float(n_samples)
                    oms = 1.0 - s
                    x = oms * oms * p0s[0] + 2 * oms * s * b[0] + s * s * p2s[0]
                    y = oms * oms * p0s[1] + 2 * oms * s * b[1] + s * s * p2s[1]
                    seg_try.append(clip_xy(x, y))
                if scale == 0.0:
                    ok_seg = [clip_xy(b[0], b[1])]
                    break
                if sample_ok(seg_try):
                    ok_seg = seg_try
                    break
            seg = ok_seg or [clip_xy(b[0], b[1])]

        for p in seg[1:]:
            if not out or out[-1] != p:
                out.append(p)

    last = clip_xy(float(pts[-1][0]), float(pts[-1][1]))
    if not out or out[-1] != last:
        out.append(last)
    return out if len(out) >= 2 else pts
