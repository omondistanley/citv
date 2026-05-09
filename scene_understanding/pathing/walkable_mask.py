"""Scene- and traversability-aware walkable mask for path validation (single source of truth)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _semantic_locomotion_block_mask(
    lm: np.ndarray,
    regions_meta: Optional[List[Dict[str, Any]]],
    semantic_layer: Optional[Dict[str, Any]],
    *,
    exclude_far_background: bool,
    exclude_obstacle_regions: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pixels to remove from geometric walkable: region labels whose affordance is
    non-traversable (far_background / obstacle) per semantic_layer + regions_meta.
    """
    info: Dict[str, Any] = {"labels_blocked": 0}
    block = np.zeros_like(lm, dtype=bool)
    if not regions_meta or semantic_layer is None:
        return block, info
    afford_list = semantic_layer.get("region_affordances") or []
    if not afford_list:
        return block, info
    rid_to_aff: Dict[str, str] = {}
    for a in afford_list:
        if not isinstance(a, dict):
            continue
        rid = str(a.get("region_id", "") or "").strip()
        if rid:
            rid_to_aff[rid] = str(a.get("affordance", "interaction_zone") or "interaction_zone")
    index_to_rid: Dict[int, str] = {}
    for r in regions_meta:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("region_index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        rid = str(r.get("id", "") or "").strip()
        if idx > 0 and rid:
            index_to_rid[idx] = rid
    excluded = set()
    if exclude_far_background:
        excluded.add("far_background")
    if exclude_obstacle_regions:
        excluded.add("obstacle")
    if not excluded:
        return block, info
    n_blocked = 0
    for li in np.unique(lm):
        lii = int(li)
        if lii <= 0:
            continue
        rid = index_to_rid.get(lii, "")
        aff = rid_to_aff.get(rid, "interaction_zone")
        if aff in excluded:
            block[lm == lii] = True
            n_blocked += 1
    info["labels_blocked"] = int(n_blocked)
    return block, info


def _keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return a bool mask retaining only the largest connected component.

    Discards isolated walkable islands so FMM never encounters T=∞ pockets
    that arise when a disconnected island is used as a start/goal anchor.
    """
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if n_labels <= 2:
        return mask.astype(bool)
    sizes = [(labels == i).sum() for i in range(1, n_labels)]
    largest = int(np.argmax(sizes)) + 1
    return (labels == largest).astype(bool)


def build_path_walkable_mask(
    region_label_map: np.ndarray,
    obstacle_mask: np.ndarray,
    speed_map: np.ndarray,
    cfg: Any,
    *,
    semantic_layer: Optional[Dict[str, Any]] = None,
    regions_meta: Optional[List[Dict[str, Any]]] = None,
    support_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Boolean (H,W) mask: pixels where inter-region / locomotion paths are allowed.

    Combines region support (label > 0), free of merged obstacle instances,
    optional traversability speed threshold, and optional morphological erosion
    to stay off fragile region borders. When ``support_mask`` is supplied
    (Phase 2.4) the result is intersected with it so paths cannot wander onto
    walls / sofas / tabletops, only onto floor / ground / road / stair surfaces.
    """
    lm = np.asarray(region_label_map, dtype=np.int32)
    obs = np.asarray(obstacle_mask, dtype=bool)
    sp = np.clip(np.asarray(speed_map, dtype=np.float32), 0.0, 1.0)
    h, w = lm.shape[:2]
    use_obs = bool(getattr(cfg, "path_walkable_use_obstacles", True)) if cfg else True
    base = (lm > 0).astype(bool)
    if use_obs:
        base &= ~obs
    if support_mask is not None:
        sm = np.asarray(support_mask, dtype=bool)
        if sm.shape == base.shape and sm.any():
            intersected = base & sm
            if intersected.any():
                base = intersected

    smin = float(getattr(cfg, "path_walkable_min_speed", 0.0)) if cfg else 0.0
    if smin > 1e-6:
        base &= sp >= smin

    q = float(getattr(cfg, "path_walkable_speed_quantile", 0.0)) if cfg else 0.0
    if q > 1e-6 and base.any():
        vals = sp[base]
        if vals.size > 10:
            thr = float(np.percentile(vals, max(0.0, min(100.0, q * 100.0))))
            base &= sp >= thr

    erode = int(getattr(cfg, "path_walkable_erode_px", 0)) if cfg else 0
    if erode > 0:
        k = max(1, 2 * erode + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        er = cv2.erode(base.astype(np.uint8) * 255, kernel) > 0
        if er.any():
            base = er
        else:
            base = (lm > 0).astype(bool)
            if use_obs:
                base &= ~obs

    if not base.any() and lm.size:
        base = (lm > 0).astype(bool)
        if use_obs:
            base &= ~obs

    # Phase 1.1 — keep only the largest connected component so isolated
    # walkable islands don't produce T=∞ in FMM when used as anchors.
    if base.any():
        filtered = _keep_largest_cc(base)
        if filtered.any():
            base = filtered

    fusion_on = bool(getattr(cfg, "path_walkable_semantic_fusion_enabled", True)) if cfg else True
    ex_fb = bool(getattr(cfg, "path_walkable_semantic_exclude_far_background", True)) if cfg else True
    ex_obs_reg = bool(getattr(cfg, "path_walkable_semantic_exclude_obstacle_regions", True)) if cfg else True
    meta_sem: Dict[str, Any] = {
        "semantic_fusion_applied": 0.0,
        "semantic_fusion_fallback_geometric": 0.0,
        "semantic_labels_blocked": 0.0,
    }
    base_geo = base.copy()
    if fusion_on and semantic_layer is not None and regions_meta:
        sem_block, sem_info = _semantic_locomotion_block_mask(
            lm,
            regions_meta,
            semantic_layer,
            exclude_far_background=ex_fb,
            exclude_obstacle_regions=ex_obs_reg,
        )
        meta_sem["semantic_labels_blocked"] = float(sem_info.get("labels_blocked", 0))
        if sem_block.any():
            fused = base & ~sem_block
            meta_sem["semantic_fusion_applied"] = 1.0
            if fused.any():
                base = fused
            else:
                meta_sem["semantic_fusion_fallback_geometric"] = 1.0
                base = base_geo

    meta: Dict[str, Any] = {
        "walkable_ratio": float(base.mean()) if base.size else 0.0,
        "use_obstacles": float(1.0 if use_obs else 0.0),
        "min_speed": float(smin),
        "speed_quantile_setting": float(q),
        "erode_px": float(erode),
        "semantic_fusion_applied": float(meta_sem["semantic_fusion_applied"]),
        "semantic_fusion_fallback_geometric": float(meta_sem["semantic_fusion_fallback_geometric"]),
        "semantic_labels_blocked": float(meta_sem["semantic_labels_blocked"]),
        "support_intersection_applied": float(1.0 if support_mask is not None else 0.0),
    }
    return base, meta


def per_actor_connected_components(
    walkable: np.ndarray,
    actor_uvs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, List[int]]:
    """Plan §2.4: return the per-actor CC label and the full CC label array.

    The walkable mask may be split into several connected components
    (e.g. two rooms joined only by a door masked as obstacle). The vanilla
    ``_keep_largest_cc`` discards every minor component, which silently
    snaps actors from a small island onto the dominant CC and produces
    paths that "hop" across visually-disjoint regions. This helper instead
    returns:

    - ``labels``: ``(H, W)`` int32 with 1..N CC ids (0 = non-walkable).
    - ``per_actor_cc``: list aligned with ``actor_uvs`` giving the CC id of
      each actor's foot/support anchor, or 0 if the anchor is not on any CC.

    Callers can then group actor pairs by ``cc`` to decide whether to route
    a normal path or emit a portal_path candidate.
    """
    ww = np.asarray(walkable, dtype=np.uint8)
    n_labels, labels = cv2.connectedComponents(ww)
    h, w = labels.shape[:2]
    per_actor: List[int] = []
    for uv in actor_uvs:
        try:
            x = int(round(float(uv[0])))
            y = int(round(float(uv[1])))
        except (TypeError, ValueError):
            per_actor.append(0)
            continue
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        per_actor.append(int(labels[y, x]))
    return labels.astype(np.int32), per_actor


def snap_uv_to_medial_interior(
    x: int,
    y: int,
    mask: np.ndarray,
    w: int,
    h: int,
) -> Tuple[int, int]:
    """Snap to a medial interior pixel inside ``mask``, closest to (x, y).

    Phase 1.2 fix: instead of taking the global argmax of the distance
    transform (which can land in an unreachable arm of a U-shaped region),
    we collect all pixels whose DT value exceeds half the maximum (the
    approximate medial zone) and pick the one nearest the query point.
    This keeps the snap obstacle-aware and reachability-respecting.
    """
    mm = np.asarray(mask, dtype=bool)
    if mm.shape[:2] != (h, w) or not mm.any():
        return snap_uv_to_mask(x, y, mask, w, h)
    inv = (~mm).astype(np.uint8) * 255
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dt[~mm] = 0.0
    dt_max = float(np.max(dt))
    if dt_max < 1.0:
        return snap_uv_to_mask(x, y, mask, w, h)
    # Medial zone = pixels with DT ≥ 50 % of global max.
    medial = dt >= (dt_max * 0.5)
    mys, mxs = np.where(medial)
    if mxs.size == 0:
        iy, ix = np.unravel_index(int(np.argmax(dt)), dt.shape)
        return int(ix), int(iy)
    d = (mxs.astype(np.float64) - float(x)) ** 2 + (mys.astype(np.float64) - float(y)) ** 2
    j = int(np.argmin(d))
    return int(mxs[j]), int(mys[j])


def snap_uv_to_mask(
    x: int,
    y: int,
    mask: np.ndarray,
    w: int,
    h: int,
) -> Tuple[int, int]:
    """Nearest pixel inside bool mask; clip to bounds if mask empty."""
    mm = np.asarray(mask, dtype=bool)
    xi = int(max(0, min(w - 1, x)))
    yi = int(max(0, min(h - 1, y)))
    if mm.shape[:2] != (h, w):
        return xi, yi
    if mm[yi, xi]:
        return xi, yi
    ys, xs = np.where(mm)
    if xs.size == 0:
        return xi, yi
    d = (xs.astype(np.float64) - float(x)) ** 2 + (ys.astype(np.float64) - float(y)) ** 2
    j = int(np.argmin(d))
    return int(xs[j]), int(ys[j])


def snap_uv_to_walkable(x: int, y: int, walkable: np.ndarray, w: int, h: int) -> Tuple[int, int]:
    """Nearest walkable pixel to (x,y)."""
    ww = np.asarray(walkable, dtype=bool)
    xi = int(max(0, min(w - 1, x)))
    yi = int(max(0, min(h - 1, y)))
    if ww.shape[:2] != (h, w):
        return xi, yi
    if ww[yi, xi]:
        return xi, yi
    ys, xs = np.where(ww)
    if xs.size == 0:
        return xi, yi
    d = (xs.astype(np.float64) - float(x)) ** 2 + (ys.astype(np.float64) - float(y)) ** 2
    j = int(np.argmin(d))
    return int(xs[j]), int(ys[j])
