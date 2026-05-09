"""Feasible-pixel masks for staged FMM routing (base vs support-relaxed).

``feasible_base`` matches the historical (lm>0) ∧ ¬obstacles ∧ physical_speed_gate.

``feasible_routing`` optionally unions a *support-only* relaxed gate so thin
support surfaces (e.g. stair treads) stay connected for geodesics without
opening the entire image.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def build_feasible_base(
    lm: np.ndarray,
    obs_mask: np.ndarray,
    speed_map: np.ndarray,
    *,
    speed_floor: float,
    physical_mult: float = 1.5,
) -> np.ndarray:
    """Return boolean (H,W) feasible mask used for strict routing."""
    lm = np.asarray(lm, dtype=np.int32)
    obs = np.asarray(obs_mask, dtype=bool)
    sp = np.asarray(speed_map, dtype=np.float32)
    physical_gate = sp > (float(speed_floor) * float(physical_mult))
    return (lm > 0) & (~obs) & physical_gate


def build_feasible_routing(
    feasible_base: np.ndarray,
    lm: np.ndarray,
    obs_mask: np.ndarray,
    speed_map: np.ndarray,
    support_mask: Optional[np.ndarray],
    cfg: Any,
    *,
    speed_floor: float,
) -> Tuple[np.ndarray, str]:
    """Return (feasible_routing, variant_name).

    When ``path_routing_relax_on_support`` is false or ``support_mask`` is empty,
    returns ``feasible_base`` unchanged with variant ``base``.
    """
    base = np.asarray(feasible_base, dtype=bool)
    if not bool(getattr(cfg, "path_routing_relax_on_support", False)) if cfg else False:
        return base, "base"
    if support_mask is None:
        return base, "base"
    sm = np.asarray(support_mask, dtype=bool)
    if sm.shape != base.shape or not sm.any():
        return base, "base"

    sp = np.asarray(speed_map, dtype=np.float32)
    mult = float(getattr(cfg, "path_routing_support_speed_floor_mult", 0.72)) if cfg else 0.72
    relaxed_gate = sp > (float(speed_floor) * float(mult) * 1.5)
    lm0 = np.asarray(lm, dtype=np.int32) > 0
    obs = np.asarray(obs_mask, dtype=bool)

    bridge = sm & lm0 & (~obs) & relaxed_gate
    out = base | bridge

    close_px = int(getattr(cfg, "path_routing_support_close_px", 0)) if cfg else 0
    if close_px > 0:
        try:
            import cv2

            k = max(3, 2 * close_px + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            core = (out & sm).astype(np.uint8) * 255
            dil = cv2.dilate(core, kernel) > 0
            out = out | (dil & sm & lm0 & (~obs) & relaxed_gate)
        except Exception:
            pass

    if not out.any():
        return base, "base"
    return out, "routing_support"


def build_feasible_bridge(
    feasible_routing: np.ndarray,
    lm: np.ndarray,
    obs_mask: np.ndarray,
    speed_map: np.ndarray,
    support_mask: Optional[np.ndarray],
    cfg: Any,
    *,
    speed_floor: float,
) -> np.ndarray:
    """Union extra support pixels (lower speed floor) to merge CCs for portal recovery."""
    fr = np.asarray(feasible_routing, dtype=bool)
    if support_mask is None:
        return fr
    sm = np.asarray(support_mask, dtype=bool)
    if sm.shape != fr.shape or not sm.any():
        return fr
    sp = np.asarray(speed_map, dtype=np.float32)
    mult = float(getattr(cfg, "path_routing_bridge_speed_floor_mult", 0.55)) if cfg else 0.55
    gate = sp > (float(speed_floor) * float(mult) * 1.5)
    lm0 = np.asarray(lm, dtype=np.int32) > 0
    obs = np.asarray(obs_mask, dtype=bool)
    extra = sm & lm0 & (~obs) & gate
    out = fr | extra
    return out if out.any() else fr


def cc_label_at(
    labels: np.ndarray,
    uv: Tuple[int, int],
    h: int,
    w: int,
) -> int:
    x, y = int(uv[0]), int(uv[1])
    if not (0 <= y < h and 0 <= x < w):
        return 0
    return int(labels[y, x])


def connected_labels(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (label_map, num_labels) with 0 = background."""
    m = np.asarray(mask, dtype=bool)
    try:
        import cv2

        n, lab = cv2.connectedComponents(m.astype(np.uint8))
        return lab, int(n) - 1
    except Exception:
        # NumPy fallback: single component
        return m.astype(np.int32), int(m.any())
