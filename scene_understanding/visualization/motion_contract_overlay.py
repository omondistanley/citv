"""Motion contract QA overlay (plan §2.9).

Renders one panel per scene proving the support / foot / visibility chain:

- support_mask outlined in cyan;
- per-object foot/support anchors as filled dots;
- top-K ribbons coloured by ``manifold_type``;
- visibility-tagged segments (solid / dashed / faded) using the new
  ``tapered_polyline_draw`` helper;
- trajectory instant_prior arrows in magenta;
- a small legend in the upper-left.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Fixed manifold-type palette so ribbons keep their semantic identity across
# panels and frames (BGR, OpenCV order).
_MANIFOLD_COLORS: Dict[str, Tuple[int, int, int]] = {
    "ribbon_path": (240, 200, 30),
    "centerline_path": (220, 180, 30),
    "blob_path": (60, 200, 240),
    "volume_path": (255, 130, 200),
    "contour_path": (90, 220, 90),
    "interior_path": (40, 200, 90),
    "portal_path": (200, 60, 255),
    "occlusion_pulse": (40, 90, 230),
    "contact_patch": (90, 240, 200),
    "effect_field": (200, 220, 240),
}


def _color_for_manifold(mtype: str) -> Tuple[int, int, int]:
    return _MANIFOLD_COLORS.get(str(mtype or ""), (60, 200, 255))


def _draw_support_outline(canvas: np.ndarray, support_mask: Optional[np.ndarray]) -> None:
    if support_mask is None:
        return
    sm = np.asarray(support_mask, dtype=bool)
    if sm.size == 0 or not sm.any():
        return
    edges = cv2.Canny(sm.astype(np.uint8) * 255, 50, 100)
    canvas[edges > 0] = (200, 220, 60)


def _draw_anchor_dots(
    canvas: np.ndarray,
    object_affordances: Optional[Dict[str, Any]],
    width: int,
    height: int,
) -> int:
    if not object_affordances:
        return 0
    drawn = 0
    for o in object_affordances.get("objects") or []:
        anchors = o.get("anchors") or {}
        sc = anchors.get("support_contact_uv") or anchors.get("foot_uv")
        if not sc or len(sc) < 2:
            continue
        try:
            x = int(round(float(sc[0])))
            y = int(round(float(sc[1])))
        except (TypeError, ValueError):
            continue
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        cv2.circle(canvas, (x, y), 6, (50, 220, 240), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6, (10, 10, 10), 1, cv2.LINE_AA)
        drawn += 1
    return drawn


def _draw_legend(canvas: np.ndarray, manifold_counts: Dict[str, int]) -> None:
    if not manifold_counts:
        return
    items = [(k, v) for k, v in manifold_counts.items() if v > 0]
    if not items:
        return
    pad = 8
    line_h = 18
    box_w = 220
    box_h = pad * 2 + len(items) * line_h + line_h
    x0, y0 = 10, 10
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0.0, dst=canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (60, 60, 60), 1)
    cv2.putText(
        canvas, "Motion contract overlay",
        (x0 + pad, y0 + line_h - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )
    for i, (name, cnt) in enumerate(items):
        cy = y0 + pad + (i + 1) * line_h + 6
        col = _color_for_manifold(name)
        cv2.line(canvas, (x0 + pad, cy - 6), (x0 + pad + 28, cy - 6), col, 3, cv2.LINE_AA)
        cv2.putText(
            canvas, f"{name}: {cnt}",
            (x0 + pad + 36, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA,
        )


def write_motion_contract_overlay(
    img_bgr: np.ndarray,
    paths_sorted: List[Dict[str, Any]],
    traj_bundle: Dict[str, Any],
    out_path: Path,
    cfg: Optional[Any] = None,
    *,
    support_mask: Optional[np.ndarray] = None,
    object_affordances: Optional[Dict[str, Any]] = None,
    metric_depth_m: Optional[np.ndarray] = None,
) -> None:
    """Render the §2.9 motion-contract QA panel."""
    canvas = np.asarray(img_bgr).copy()
    h_img, w_img = canvas.shape[:2]

    max_paths = int(getattr(cfg, "path_motion_contract_overlay_max_paths", 24)) if cfg else 24

    # 1. Support mask outline.
    _draw_support_outline(canvas, support_mask)

    # 2. Foot / support anchors.
    _draw_anchor_dots(canvas, object_affordances, w_img, h_img)

    # 3. Visibility-aware ribbons coloured by manifold type.
    manifold_counts: Dict[str, int] = {}
    try:
        from scene_understanding.pathing.path_canvas import (
            tapered_polyline_draw,
            draw_direction_heads,
            sample_depth_along_polyline,
        )
        sw = int(getattr(cfg, "path_stroke_start_width_px", 7)) if cfg else 7
        ew = int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2
        a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95
        a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.45)) if cfg else 0.45
        for p in paths_sorted[: max(0, max_paths)]:
            mtype = str(p.get("manifold_type", ""))
            manifold_counts[mtype] = manifold_counts.get(mtype, 0) + 1
            raw = p.get("polyline_2d_reprojected") or p.get("polyline_2d") or []
            pts = [
                (
                    max(0, min(w_img - 1, int(float(xy[0])))),
                    max(0, min(h_img - 1, int(float(xy[1])))),
                )
                for xy in raw
                if isinstance(xy, (list, tuple)) and len(xy) >= 2
            ]
            if len(pts) < 2:
                continue
            col = _color_for_manifold(mtype)
            depth_vals = sample_depth_along_polyline(pts, metric_depth_m)
            tapered_polyline_draw(
                canvas,
                pts,
                col,
                sw,
                ew,
                a0,
                a1,
                depth_values=depth_vals,
                width_profile_px=p.get("width_profile_px"),
                visibility_profile=p.get("visibility_profile"),
                metric_depth_m=metric_depth_m,
                occlusion_compositing=True,
            )
            draw_direction_heads(
                canvas, pts, col,
                thickness=2, tip_len=0.18,
                visibility_profile=p.get("visibility_profile"),
            )
    except Exception:
        # Fallback to simple polylines.
        for p in paths_sorted[: max(0, max_paths)]:
            pts = p.get("polyline_2d") or []
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            arr = np.array(
                [
                    [
                        max(0, min(w_img - 1, int(float(xy[0])))),
                        max(0, min(h_img - 1, int(float(xy[1])))),
                    ]
                    for xy in pts
                    if isinstance(xy, (list, tuple)) and len(xy) >= 2
                ],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [arr], False, _color_for_manifold(str(p.get("manifold_type", ""))), 2, cv2.LINE_AA)

    # 4. Trajectory instant_prior arrows (magenta).
    for th in (traj_bundle or {}).get("hypotheses") or []:
        for samp in (th.get("samples") or [])[:1]:
            sts = samp.get("states_t") or []
            if len(sts) >= 2:
                try:
                    p0 = (
                        max(0, min(w_img - 1, int(float(sts[0]["x_px"])))),
                        max(0, min(h_img - 1, int(float(sts[0]["y_px"])))),
                    )
                    p1 = (
                        max(0, min(w_img - 1, int(float(sts[1]["x_px"])))),
                        max(0, min(h_img - 1, int(float(sts[1]["y_px"])))),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                cv2.arrowedLine(canvas, p0, p1, (200, 60, 255), 3, cv2.LINE_AA, tipLength=0.22)

    # 5. Legend.
    _draw_legend(canvas, manifold_counts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
