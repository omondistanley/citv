"""Manifold-specific support/grounding fit scoring for path contracts."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if val == val else default
    except (TypeError, ValueError):
        return default


_SUPPORT_CHANNELS = (
    "support_surface_score",
    "route_corridor_score",
    "contact_target_score",
    "occlusion_edge_score",
    "contour_boundary_score",
    "open_air_score",
    "portal_score",
    "interior_motion_score",
    "effect_host_score",
    "confirmed_blocker_score",
    "uncertainty_score",
)


def support_channel_means(support_trace: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {k: 0.0 for k in _SUPPORT_CHANNELS}
    n = 0
    for row in support_trace or []:
        if not isinstance(row, dict):
            continue
        n += 1
        for k in _SUPPORT_CHANNELS:
            sums[k] += _float(row.get(k), 0.0)
    if n <= 0:
        return {k: 0.0 for k in _SUPPORT_CHANNELS}
    return {k: max(0.0, min(1.0, sums[k] / float(n))) for k in _SUPPORT_CHANNELS}


def _mean_visible(visibility_profile: Iterable[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for row in visibility_profile or []:
        if isinstance(row, dict):
            vals.append(_float(row.get("visible_fraction"), 1.0))
    if not vals:
        return 1.0
    return max(0.0, min(1.0, sum(vals) / float(max(1, len(vals)))))


def _geometry_contract_score(geometry_quality: Dict[str, Any]) -> float:
    if not isinstance(geometry_quality, dict) or not geometry_quality:
        return 0.45
    score = 1.0
    score -= 0.30 * _float(geometry_quality.get("zigzag_score"), 0.0)
    score -= 0.08 * min(4.0, _float(geometry_quality.get("depth_jump_count"), 0.0))
    score -= 0.35 * _float((geometry_quality.get("support_snap_displacement_px") or {}).get("max"), 0.0) / 48.0
    reasons = list(geometry_quality.get("geometry_rejection_reasons") or [])
    if reasons:
        score -= min(0.45, 0.08 * len(reasons))
    return max(0.0, min(1.0, score))


def _manifold_support_fit(manifold: str, ch: Dict[str, float]) -> float:
    # Route-like manifolds rely on support and corridor evidence.
    if manifold == "ribbon_path":
        return max(
            0.0,
            min(
                1.0,
                0.55 * ch["support_surface_score"]
                + 0.25 * ch["route_corridor_score"]
                + 0.10 * ch["contour_boundary_score"]
                + 0.10 * ch["portal_score"]
                - 0.70 * ch["confirmed_blocker_score"],
            ),
        )
    if manifold == "contact_patch":
        return max(0.0, min(1.0, 0.55 * ch["contact_target_score"] + 0.20 * ch["contour_boundary_score"] + 0.25 * (1.0 - ch["confirmed_blocker_score"])))
    if manifold == "occlusion_pulse":
        return max(0.0, min(1.0, 0.50 * ch["occlusion_edge_score"] + 0.25 * ch["contour_boundary_score"] + 0.25 * (1.0 - ch["confirmed_blocker_score"])))
    if manifold == "contour_path":
        return max(0.0, min(1.0, 0.60 * ch["contour_boundary_score"] + 0.20 * ch["contact_target_score"] + 0.20 * (1.0 - ch["confirmed_blocker_score"])))
    if manifold == "volume_path":
        return max(0.0, min(1.0, 0.70 * ch["open_air_score"] + 0.30 * (1.0 - ch["confirmed_blocker_score"])))
    if manifold == "portal_path":
        return max(0.0, min(1.0, 0.65 * ch["portal_score"] + 0.20 * ch["contour_boundary_score"] + 0.15 * (1.0 - ch["confirmed_blocker_score"])))
    if manifold in {"blob_path", "interior_path"}:
        return max(0.0, min(1.0, 0.65 * ch["interior_motion_score"] + 0.20 * ch["contact_target_score"] + 0.15 * (1.0 - ch["confirmed_blocker_score"])))
    if manifold == "effect_field":
        return max(0.0, min(1.0, 0.65 * ch["effect_host_score"] + 0.20 * ch["contact_target_score"] + 0.15 * (1.0 - ch["confirmed_blocker_score"])))
    return max(0.0, min(1.0, 0.45 * ch["support_surface_score"] + 0.30 * ch["route_corridor_score"] + 0.25 * (1.0 - ch["confirmed_blocker_score"])))


def _contradiction_score(manifold: str, ch: Dict[str, float], geometry_quality: Dict[str, Any]) -> float:
    base = 0.0
    blocker = ch.get("confirmed_blocker_score", 0.0)
    if manifold in {"ribbon_path", "contour_path", "portal_path", "blob_path", "interior_path"}:
        base += 0.75 * blocker
    else:
        base += 0.35 * blocker
    if isinstance(geometry_quality, dict):
        if list(geometry_quality.get("geometry_rejection_reasons") or []):
            base += 0.22
        base += 0.12 * min(3.0, _float(geometry_quality.get("depth_jump_count"), 0.0))
    return max(0.0, min(1.0, base))


def compute_manifold_fit_scores(
    *,
    manifold_type: str,
    support_trace: Iterable[Dict[str, Any]],
    visibility_profile: Iterable[Dict[str, Any]],
    geometry_quality: Dict[str, Any],
    local_grounding_score: float,
) -> Dict[str, float]:
    ch = support_channel_means(support_trace)
    manifold = str(manifold_type or "ribbon_path")
    manifold_fit = _manifold_support_fit(manifold, ch)
    renderability = _mean_visible(visibility_profile)
    geometry_score = _geometry_contract_score(geometry_quality)
    contradiction = _contradiction_score(manifold, ch, geometry_quality)
    uncertainty = max(0.0, min(1.0, 0.65 * ch.get("uncertainty_score", 0.0) + 0.35 * (1.0 - local_grounding_score)))
    return {
        "manifold_fit_score": round(float(manifold_fit), 4),
        "local_grounding_score": round(float(max(0.0, min(1.0, local_grounding_score))), 4),
        "geometry_contract_score": round(float(geometry_score), 4),
        "renderability_score": round(float(renderability), 4),
        "contradiction_score": round(float(contradiction), 4),
        "uncertainty_score": round(float(uncertainty), 4),
        "support_channel_means": {k: round(float(v), 4) for k, v in ch.items()},
    }


__all__ = ["compute_manifold_fit_scores", "support_channel_means"]
