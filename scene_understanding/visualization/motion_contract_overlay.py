"""Motion contract QA overlay: legacy polylines, geodesic, trajectory instant_prior arrows."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def write_motion_contract_overlay(
    img_bgr: np.ndarray,
    paths_sorted: List[Dict[str, Any]],
    traj_bundle: Dict[str, Any],
    out_path: Path,
    cfg: Optional[Any] = None,
) -> None:
    """
    QA overlay (bottom to top draw order):
    1) Legacy path polylines polyline_2d — per-path colors (same hash scheme as path maps).
    2) Geodesic polylines polyline_geodesic_2d — green (when present).
    3) Trajectory instant_prior — magenta arrows from trajectory_hypotheses.
    """
    canvas = np.asarray(img_bgr).copy()
    h_img, w_img = canvas.shape[:2]

    def _path_color(pid: str) -> Tuple[int, int, int]:
        hsh = abs(hash(str(pid)))
        return (int(50 + (hsh % 205)), int(50 + ((hsh // 7) % 205)), int(50 + ((hsh // 49) % 205)))

    max_paths = int(getattr(cfg, "path_motion_contract_overlay_max_paths", 24)) if cfg else 24
    lw_legacy = int(getattr(cfg, "path_motion_contract_legacy_line_px", 2)) if cfg else 2
    lw_geo = int(getattr(cfg, "path_motion_contract_geodesic_line_px", 3)) if cfg else 3

    for p in paths_sorted[: max(0, max_paths)]:
        pts = p.get("polyline_2d") or []
        if isinstance(pts, list) and len(pts) >= 2:
            arr = np.array(
                [
                    [
                        max(0, min(w_img - 1, int(float(xy[0])))),
                        max(0, min(h_img - 1, int(float(xy[1])))),
                    ]
                    for xy in pts
                ],
                dtype=np.int32,
            )
            pid = str(p.get("path_id", ""))
            cv2.polylines(canvas, [arr], False, _path_color(pid), lw_legacy, lineType=cv2.LINE_AA)

    for p in paths_sorted[: max(0, max_paths)]:
        g = p.get("polyline_geodesic_2d")
        if isinstance(g, list) and len(g) >= 2:
            arr = np.array(
                [
                    [
                        max(0, min(w_img - 1, int(float(xy[0])))),
                        max(0, min(h_img - 1, int(float(xy[1])))),
                    ]
                    for xy in g
                ],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [arr], False, (40, 220, 60), lw_geo, lineType=cv2.LINE_AA)

    for th in traj_bundle.get("hypotheses") or []:
        for samp in (th.get("samples") or [])[:1]:
            sts = samp.get("states_t") or []
            if len(sts) >= 2:
                p0 = (
                    max(0, min(w_img - 1, int(float(sts[0]["x_px"])))),
                    max(0, min(h_img - 1, int(float(sts[0]["y_px"])))),
                )
                p1 = (
                    max(0, min(w_img - 1, int(float(sts[1]["x_px"])))),
                    max(0, min(h_img - 1, int(float(sts[1]["y_px"])))),
                )
                cv2.arrowedLine(canvas, p0, p1, (200, 60, 255), 3, cv2.LINE_AA, tipLength=0.22)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
