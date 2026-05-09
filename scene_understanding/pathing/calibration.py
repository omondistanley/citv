"""Data-driven calibration helpers for path scoring and filtering."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _safe_quantile(values: Iterable[float], q: float, fallback: float) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return float(fallback)
    return float(np.quantile(arr, max(0.0, min(1.0, float(q)))))


def calibrate_path_policy(
    *,
    cfg: Any,
    paths: List[Dict[str, Any]],
    semantic_scores: Iterable[float],
    far_ratios: Iterable[float],
    obs_ratios: Iterable[float],
    walk_ratios: Iterable[float],
    motion_scores: Iterable[float],
    hybrid_scores_for_min_conf: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Build calibrated thresholds from current-scene path distributions."""

    enabled = bool(getattr(cfg, "path_calibration_enabled", True)) if cfg else True
    if not enabled:
        return {
            "min_conf": float(getattr(cfg, "path_min_confidence", 0.55)) if cfg else 0.55,
            "semantic_validity_floor": float(getattr(cfg, "path_semantic_validity_floor", 0.30)) if cfg else 0.30,
            "max_far": float(getattr(cfg, "path_semantic_max_far_background_ratio", 0.40)) if cfg else 0.40,
            "max_obs": float(getattr(cfg, "path_semantic_max_obstacle_ratio", 0.35)) if cfg else 0.35,
            "min_walk": float(getattr(cfg, "path_semantic_min_walkable_ratio", 0.20)) if cfg else 0.20,
            "motion_primary_threshold": float(getattr(cfg, "path_motion_primary_threshold", 0.45)) if cfg else 0.45,
        }

    sem_q = float(getattr(cfg, "path_calibration_semantic_valid_quantile", 0.30)) if cfg else 0.30
    far_q = float(getattr(cfg, "path_calibration_max_far_quantile", 0.85)) if cfg else 0.85
    obs_q = float(getattr(cfg, "path_calibration_max_obs_quantile", 0.85)) if cfg else 0.85
    walk_q = float(getattr(cfg, "path_calibration_min_walk_quantile", 0.20)) if cfg else 0.20
    motion_q = float(getattr(cfg, "path_calibration_motion_primary_quantile", 0.60)) if cfg else 0.60

    if hybrid_scores_for_min_conf is not None:
        conf_vals = [float(x) for x in hybrid_scores_for_min_conf if np.isfinite(float(x))]
        if not conf_vals:
            conf_vals = [float(getattr(cfg, "path_min_confidence", 0.55)) if cfg else 0.55]
        conf_q = float(getattr(cfg, "path_calibration_min_hybrid_quantile", 0.15)) if cfg else 0.15
    else:
        conf_vals = [
            float((p.get("scores") or {}).get("overall_confidence", 0.0))
            for p in (paths or [])
        ]
        conf_q = float(getattr(cfg, "path_calibration_min_conf_quantile", 0.55)) if cfg else 0.55
    min_conf = _safe_quantile(conf_vals, conf_q, float(getattr(cfg, "path_min_confidence", 0.55)) if cfg else 0.55)
    if hybrid_scores_for_min_conf is not None:
        cap = float(getattr(cfg, "path_calibration_hybrid_min_conf_cap", 0.56)) if cfg else 0.56
        if 0.0 < cap < 1.0:
            min_conf = min(float(min_conf), cap)
    sem_floor = _safe_quantile(
        semantic_scores,
        sem_q,
        float(getattr(cfg, "path_semantic_validity_floor", 0.30)) if cfg else 0.30,
    )
    max_far = _safe_quantile(
        far_ratios,
        far_q,
        float(getattr(cfg, "path_semantic_max_far_background_ratio", 0.40)) if cfg else 0.40,
    )
    max_obs = _safe_quantile(
        obs_ratios,
        obs_q,
        float(getattr(cfg, "path_semantic_max_obstacle_ratio", 0.35)) if cfg else 0.35,
    )
    min_walk = _safe_quantile(
        walk_ratios,
        walk_q,
        float(getattr(cfg, "path_semantic_min_walkable_ratio", 0.20)) if cfg else 0.20,
    )
    motion_thr = _safe_quantile(
        motion_scores,
        motion_q,
        float(getattr(cfg, "path_motion_primary_threshold", 0.45)) if cfg else 0.45,
    )
    # Optional global scaling from config.
    scale = float(getattr(cfg, "path_calibration_threshold_scale", 1.0)) if cfg else 1.0
    min_conf = max(0.0, min(1.0, min_conf * scale))
    sem_floor = max(0.0, min(1.0, sem_floor * scale))
    return {
        "min_conf": float(min_conf),
        "semantic_validity_floor": float(sem_floor),
        "max_far": float(max_far),
        "max_obs": float(max_obs),
        "min_walk": float(min_walk),
        "motion_primary_threshold": float(motion_thr),
    }


__all__ = ["calibrate_path_policy"]

