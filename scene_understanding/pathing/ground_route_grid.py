"""Coarse support-grid routing: FMM on downsampled support cells (Pitfall 2 mitigation).

Maps polylines from a coarse grid back to image UV. Falls back to empty path on failure.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def _cell_centers(
    h: int,
    w: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (cy, cx) integer centers per coarse cell, plus coarse shape."""
    step = max(2, int(step))
    ch = int(np.ceil(h / step))
    cw = int(np.ceil(w / step))
    cy = np.arange(ch, dtype=np.float32) * step + (step * 0.5)
    cx = np.arange(cw, dtype=np.float32) * step + (step * 0.5)
    return cy, cx, ch, cw


def image_uv_to_coarse(
    u: int,
    v: int,
    step: int,
    ch: int,
    cw: int,
) -> Tuple[int, int]:
    ci = int(np.clip(v // step, 0, ch - 1))
    cj = int(np.clip(u // step, 0, cw - 1))
    return ci, cj


def coarse_center_to_uv(ci: int, cj: int, step: int) -> Tuple[int, int]:
    u = int(cj * step + step // 2)
    v = int(ci * step + step // 2)
    return u, v


def build_coarse_speed_field(
    speed_map: np.ndarray,
    support_mask: np.ndarray,
    feasible: np.ndarray,
    step: int,
    *,
    support_frac_thresh: float = 0.25,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Average speed onto coarse grid; mask cells with low support fraction.

    Returns (coarse_speed, coarse_feasible_bool, step) or (None, None, step).
    """
    if cv2 is None:
        return None, None, step
    sm = np.asarray(speed_map, dtype=np.float32)
    sup = np.asarray(support_mask, dtype=bool)
    feas = np.asarray(feasible, dtype=bool)
    h, w = sm.shape[:2]
    step = max(2, int(step))
    cy, cx, ch, cw = _cell_centers(h, w, step)
    coarse_sp = np.zeros((ch, cw), dtype=np.float32)
    coarse_mask = np.zeros((ch, cw), dtype=bool)
    for ci in range(ch):
        y0, y1 = ci * step, min(h, (ci + 1) * step)
        for cj in range(cw):
            x0, x1 = cj * step, min(w, (cj + 1) * step)
            patch_sup = sup[y0:y1, x0:x1]
            patch_feas = feas[y0:y1, x0:x1]
            if patch_sup.size == 0:
                continue
            frac = float(patch_sup.mean())
            if frac < support_frac_thresh:
                continue
            if not (patch_feas & patch_sup).any():
                continue
            patch_speed = sm[y0:y1, x0:x1]
            valid = patch_feas & patch_sup
            if not valid.any():
                continue
            coarse_sp[ci, cj] = float(np.mean(patch_speed[valid]))
            coarse_mask[ci, cj] = True
    if not coarse_mask.any():
        return None, None, step
    coarse_sp = np.where(coarse_mask, coarse_sp, 1e-3)
    return coarse_sp, coarse_mask, step


def plan_coarse_support_path(
    speed_map: np.ndarray,
    support_mask: np.ndarray,
    feasible: np.ndarray,
    start_uv: Tuple[int, int],
    goal_uv: Tuple[int, int],
    *,
    step: int = 14,
) -> List[Tuple[int, int]]:
    """Run FMM on coarse grid restricted to support-feasible cells; return image-space polyline."""
    from .semantic_fmm import backtrace_from_T, time_of_arrival_from_speed

    coarse_sp, coarse_feas, step = build_coarse_speed_field(
        speed_map, support_mask, feasible, step
    )
    if coarse_sp is None or coarse_feas is None:
        return []
    ch, cw = coarse_sp.shape[:2]
    su, sv = int(start_uv[0]), int(start_uv[1])
    gu, gv = int(goal_uv[0]), int(goal_uv[1])
    sci, scj = image_uv_to_coarse(su, sv, step, ch, cw)
    gci, gcj = image_uv_to_coarse(gu, gv, step, ch, cw)
    if not coarse_feas[sci, scj] or not coarse_feas[gci, gcj]:
        # Snap to nearest feasible coarse cell
        def _nearest(fi: int, fj: int) -> Optional[Tuple[int, int]]:
            ys, xs = np.where(coarse_feas)
            if ys.size == 0:
                return None
            d = (ys - fi) ** 2 + (xs - fj) ** 2
            k = int(np.argmin(d))
            return int(ys[k]), int(xs[k])

        a = _nearest(sci, scj)
        b = _nearest(gci, gcj)
        if a is None or b is None:
            return []
        sci, scj = a
        gci, gcj = b

    sm_coarse = np.where(coarse_feas, coarse_sp, coarse_sp * 0.02)
    Tg = time_of_arrival_from_speed(sm_coarse, (gcj, gci))
    if Tg is None:
        return []
    raw = backtrace_from_T(Tg, (scj, sci))
    if len(raw) < 2:
        return []
    out: List[Tuple[int, int]] = []
    for x, y in raw:
        u, v = coarse_center_to_uv(int(y), int(x), step)
        out.append((u, v))
    return out


__all__ = [
    "build_coarse_speed_field",
    "plan_coarse_support_path",
    "image_uv_to_coarse",
    "coarse_center_to_uv",
]
