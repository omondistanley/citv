"""Bridge: lets the monolithic pipeline call path-hypothesis export without a circular import."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def invoke_path_hypotheses_export_for_track(
    pipeline: Any,
    *,
    img_bgr: np.ndarray,
    path_stem: str,
    track_dir_name: str,
    track_dir: Path,
    objects_3d_with_masks: List[Dict[str, Any]],
    regions_block: Optional[Dict[str, Any]],
    region_label_map: Optional[np.ndarray],
    region_adjacency: Optional[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    metric_depth_m: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Delegate to ``pipeline._export_path_hypotheses_for_track`` (monolithic entry point)."""
    fn = getattr(pipeline, "_export_path_hypotheses_for_track", None)
    if fn is None:
        return {}
    try:
        result = fn(
            img_bgr=img_bgr,
            path_stem=path_stem,
            track_dir_name=track_dir_name,
            track_dir=track_dir,
            objects_3d_with_masks=objects_3d_with_masks,
            regions_block=regions_block,
            region_label_map=region_label_map,
            region_adjacency=region_adjacency,
            relations=relations,
            metric_depth_m=metric_depth_m,
        )
        return dict(result) if result else {}
    except Exception as exc:
        print(f"  [PathExportHook] export failed: {exc}")
        return {}
