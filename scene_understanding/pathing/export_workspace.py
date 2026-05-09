"""Early setup for path hypothesis export (directories + feasible mask)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def prepare_path_hypotheses_workspace(
    cfg: Any,
    img_bgr: np.ndarray,
    path_stem: str,
    track_dir: Path,
    region_label_map: Optional[np.ndarray],
    regions_block: Optional[Dict[str, Any]],
    region_adjacency: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Return workspace dict if export should proceed; ``None`` if disabled or missing regions.

    Keys: ``paths_root_dir``, ``images_root_dir``, ``images_region_dir``, ``images_object_dir``,
    ``images_mask_dir``, ``h``, ``w``, ``lm``, ``regions_meta``, ``region_by_id``, ``feasible``.

    The boolean **walkable** mask used for path validation (``(lm>0) ∧ ¬obstacles`` plus optional
    speed / erosion) is built in the staged path-export flow after instance masks and the
    traversability speed field exist — see ``scene_understanding.pathing.walkable_mask``.
    """
    enabled = bool(getattr(cfg, "export_path_hypotheses", True)) if cfg else True
    if not enabled:
        return None
    if region_label_map is None or not regions_block or not region_adjacency:
        return None

    paths_root_dir = track_dir / f"{path_stem}_paths"
    paths_root_dir.mkdir(parents=True, exist_ok=True)
    images_root_dir = paths_root_dir / "images"
    images_root_dir.mkdir(parents=True, exist_ok=True)

    images_region_dir = images_root_dir / "region"
    images_object_dir = images_root_dir / "object"
    images_mask_dir = images_root_dir / "mask"
    images_region_dir.mkdir(parents=True, exist_ok=True)
    images_object_dir.mkdir(parents=True, exist_ok=True)
    images_mask_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_bgr.shape[:2]
    lm = np.asarray(region_label_map, dtype=np.int32)
    regions_meta = list((regions_block or {}).get("regions", []) or [])
    region_by_id = {str(r.get("id", "")): r for r in regions_meta if str(r.get("id", "")).strip()}
    feasible = lm > 0

    return {
        "paths_root_dir": paths_root_dir,
        "images_root_dir": images_root_dir,
        "images_region_dir": images_region_dir,
        "images_object_dir": images_object_dir,
        "images_mask_dir": images_mask_dir,
        "h": h,
        "w": w,
        "lm": lm,
        "regions_meta": regions_meta,
        "region_by_id": region_by_id,
        "feasible": feasible,
    }
