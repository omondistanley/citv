"""Validation window helpers (Phase C.7).

During the validation window the pipeline emits BOTH the legacy A* route
and the new semantic FMM route per image so the QA diff harness can
compare them without flipping the default planner.

Usage from the path export site::

    from scene_understanding.pathing.semantic_validation import plan_with_validation

    result = plan_with_validation(cfg, cost_map, semantic_cost_map, start, goal,
                                  region_label_map=label_map)
    legacy_path = result["legacy"]["path"]
    semantic_path = result["semantic"]["path"] if result["semantic"] else None

The default ``path_search_algorithm="legacy_astar"`` ensures pipelines
that have not opted in stay byte-identical — this helper is a no-op
when ``cfg.path_cost_semantic_enabled`` is False and
``cfg.path_cost_emit_both_paths`` is False.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from scene_understanding.pathing.cost_map import astar_on_cost_map
from scene_understanding.pathing.semantic_cost import SemanticCostMap
from scene_understanding.pathing.semantic_fmm import plan_path

Point = Tuple[int, int]


def plan_with_validation(
    cfg: Any,
    cost_legacy: np.ndarray,
    scm: Optional[SemanticCostMap],
    start: Point,
    goal: Point,
    *,
    region_label_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return ``{"legacy": {...}, "semantic": {...} | None}``.

    The caller decides which path is *promoted* into the canonical
    ``path_hypotheses.json`` based on ``cfg.path_search_algorithm``; the
    other is surfaced only when ``cfg.path_cost_emit_both_paths`` is True.
    """

    out: Dict[str, Any] = {"legacy": {}, "semantic": None}
    # Legacy A* on the existing cost map — identical to pre-phase-C output.
    legacy_path = astar_on_cost_map(cost_legacy, start, goal)
    out["legacy"] = {"algorithm": "legacy_astar", "path": list(legacy_path)}

    semantic_enabled = bool(getattr(cfg, "path_cost_semantic_enabled", False))
    if not semantic_enabled or scm is None:
        return out

    algorithm = str(getattr(cfg, "path_search_algorithm", "legacy_astar"))
    if algorithm == "legacy_astar":
        # Semantic cost exists but caller still wants legacy A* on top of it.
        sem_path = astar_on_cost_map(scm.cost, start, goal)
    else:
        sem_path = plan_path(
            algorithm,
            scm.cost,
            start,
            goal,
            precision=scm.precision,
            region_label_map=region_label_map,
            line_of_sight_tol=float(getattr(cfg, "path_fmm_theta_star_line_of_sight_tol", 0.05)),
            downsample_factor=int(getattr(cfg, "path_fmm_multiscale_downsample_factor", 4)),
            region_padding_px=int(getattr(cfg, "path_fmm_hier_region_padding_px", 8)),
        )
    out["semantic"] = {"algorithm": algorithm, "path": list(sem_path)}
    return out


__all__ = ["plan_with_validation"]
