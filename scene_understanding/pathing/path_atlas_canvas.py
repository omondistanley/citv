"""Line-only ranked path / trajectory panels for visual QA (trajs-upt atlas)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from scene_understanding.pathing.path_canvas import (
    draw_direction_heads,
    draw_objects_boxes_bgr,
    draw_regions_contours_bgr,
    sample_depth_along_polyline,
    tapered_polyline_draw,
)
from scene_understanding.pathing.path_colors import (
    bgr_from_stable_id,
    bgr_from_stable_id_trajectory,
    bgr_list_with_min_hue_separation,
)
from scene_understanding.pathing.path_polyline_utils import int_polyline_from_path_dict, int_xy_from_vertex


def _stable_lane_index(path_id: str, lane_count: int) -> int:
    if lane_count <= 1:
        return 0
    return int(abs(hash(str(path_id))) % max(1, lane_count))


def _path_midpoint_heading(pts: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], float]:
    if not pts:
        return (0, 0), 0.0
    if len(pts) == 1:
        return pts[0], 0.0
    if len(pts) == 2:
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        mx = int(round((float(x0) + float(x1)) * 0.5))
        my = int(round((float(y0) + float(y1)) * 0.5))
        theta = math.atan2(float(y1 - y0), float(x1 - x0))
        return (mx, my), theta
    mid = max(1, min(len(pts) - 2, len(pts) // 2))
    x0, y0 = pts[mid - 1]
    x1, y1 = pts[mid + 1]
    theta = math.atan2(float(y1 - y0), float(x1 - x0))
    return pts[mid], theta


def _draw_separated_tapered_polyline(
    canvas: np.ndarray,
    pts: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    *,
    sw: int,
    ew: int,
    a0: float,
    a1: float,
    alpha_scale: float,
    metric_depth_m: Optional[np.ndarray] = None,
    width_profile_px: Any = None,
    visibility_profile: Any = None,
    occlusion_compositing: bool = True,
) -> None:
    if len(pts) < 2:
        return
    # Dark underlay halo so overlapping neighboring paths remain separable.
    tapered_polyline_draw(
        canvas,
        pts,
        (12, 12, 12),
        max(1, int(sw) + 3),
        max(1, int(ew) + 2),
        min(1.0, float(a0) * 0.9),
        min(1.0, float(a1) * 0.9),
        alpha_scale=max(0.12, float(alpha_scale) * 0.68),
        depth_values=sample_depth_along_polyline(pts, metric_depth_m),
        width_profile_px=width_profile_px,
        visibility_profile=visibility_profile,
        metric_depth_m=metric_depth_m,
        occlusion_compositing=False,
    )
    tapered_polyline_draw(
        canvas,
        pts,
        color_bgr,
        max(1, int(sw)),
        max(1, int(ew)),
        float(a0),
        float(a1),
        alpha_scale=float(alpha_scale),
        depth_values=sample_depth_along_polyline(pts, metric_depth_m),
        width_profile_px=width_profile_px,
        visibility_profile=visibility_profile,
        metric_depth_m=metric_depth_m,
        occlusion_compositing=bool(occlusion_compositing),
    )


def _draw_directional_manifold_marker(
    canvas: np.ndarray,
    path: Dict[str, Any],
    pts: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    *,
    alpha: float = 0.55,
) -> None:
    if len(pts) < 2:
        return
    manifold = str(path.get("manifold_type", "")).strip().lower()
    if not manifold:
        return
    (cx, cy), theta = _path_midpoint_heading(pts)
    ang = -math.degrees(theta)
    overlay = canvas.copy()
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    if manifold == "contact_patch":
        # Directional contact capsule: indicates hold/touch orientation.
        cv2.ellipse(overlay, (cx, cy), (15, 6), ang, 0, 360, color_bgr, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx, cy), (15, 6), ang, 0, 360, (240, 240, 240), 1, cv2.LINE_AA)
        tail = (int(round(cx - cos_t * 12.0)), int(round(cy - sin_t * 12.0)))
        cv2.line(overlay, tail, (cx, cy), (245, 245, 245), 2, cv2.LINE_AA)
    elif manifold in {"blob_path", "interior_path"}:
        # Blob is not a circle: elongated by motion direction for readability.
        cv2.ellipse(overlay, (cx, cy), (17, 10), ang, 0, 360, color_bgr, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx, cy), (17, 10), ang, 0, 360, (240, 240, 240), 1, cv2.LINE_AA)
    elif manifold == "occlusion_pulse":
        start = int(ang - 120)
        end = int(ang + 120)
        cv2.ellipse(overlay, (cx, cy), (14, 14), 0, start, end, color_bgr, 2, cv2.LINE_AA)
        tip = (int(round(cx + cos_t * 14.0)), int(round(cy + sin_t * 14.0)))
        cv2.arrowedLine(overlay, (cx, cy), tip, color_bgr, 2, cv2.LINE_AA, tipLength=0.3)
    elif manifold == "portal_path":
        p0 = (int(round(cx - cos_t * 8.0)), int(round(cy - sin_t * 8.0)))
        p1 = (int(round(cx + cos_t * 8.0)), int(round(cy + sin_t * 8.0)))
        cv2.ellipse(overlay, p0, (9, 5), ang, 0, 360, (210, 120, 255), 2, cv2.LINE_AA)
        cv2.ellipse(overlay, p1, (9, 5), ang, 0, 360, (210, 120, 255), 2, cv2.LINE_AA)
    elif manifold == "volume_path":
        cv2.ellipse(overlay, (cx, cy), (18, 8), ang, 0, 360, (130, 225, 255), 2, cv2.LINE_AA)
    elif manifold == "effect_field":
        c0 = (cx, cy)
        c1 = (int(round(cx + cos_t * 9.0)), int(round(cy + sin_t * 9.0)))
        c2 = (int(round(cx - cos_t * 9.0)), int(round(cy - sin_t * 9.0)))
        cv2.circle(overlay, c0, 7, (230, 90, 220), 2, cv2.LINE_AA)
        cv2.circle(overlay, c1, 5, (230, 90, 220), 2, cv2.LINE_AA)
        cv2.circle(overlay, c2, 5, (230, 90, 220), 2, cv2.LINE_AA)
    else:
        cv2.arrowedLine(
            overlay,
            (int(round(cx - cos_t * 9.0)), int(round(cy - sin_t * 9.0))),
            (int(round(cx + cos_t * 9.0)), int(round(cy + sin_t * 9.0))),
            color_bgr,
            2,
            cv2.LINE_AA,
            tipLength=0.28,
        )
    a = max(0.15, min(0.95, float(alpha)))
    cv2.addWeighted(overlay, a, canvas, 1.0 - a, 0.0, dst=canvas)


def write_path_rank_panels_line_only(
    *,
    paths_root_dir: Path,
    height: int,
    width: int,
    panels: List[List[Dict[str, Any]]],
    cfg: Any,
) -> List[str]:
    """
    Draw each panel on a flat BGR canvas (no scene, regions, or boxes).
    Returns list of absolute paths written (panel_01 .. panel_N).
    """
    bg = getattr(cfg, "path_atlas_background_bgr", (26, 26, 26))
    if not isinstance(bg, (list, tuple)) or len(bg) < 3:
        bg = (26, 26, 26)
    bg_bgr = (int(bg[0]), int(bg[1]), int(bg[2]))
    prefer_geo = bool(getattr(cfg, "path_atlas_prefer_geodesic", False)) if cfg else False
    min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
    sw = int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8
    ew = int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2
    a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95
    a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35
    dots = bool(getattr(cfg, "path_atlas_endpoint_dots", False)) if cfg else False

    written: List[str] = []
    for pi, plist in enumerate(panels, start=1):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = bg_bgr
        ids = [str(p.get("path_id", "")) for p in plist if str(p.get("path_id", "")).strip()]
        colors = (
            bgr_list_with_min_hue_separation(ids, bgr_from_stable_id, min_sep)
            if ids
            else []
        )
        id_to_color = dict(zip(ids, colors))
        drawable_count = max(1, len(plist))
        for draw_i, p in enumerate(plist):
            pid = str(p.get("path_id", "")).strip()
            if not pid:
                continue
            pts = int_polyline_from_path_dict(
                p, prefer_geodesic=prefer_geo, width=width, height=height, clamp=True
            )
            if len(pts) < 2:
                continue
            draw_pts = _lane_offset_polyline(pts, draw_i, drawable_count, width, height)
            col = id_to_color.get(pid) or bgr_from_stable_id(pid)
            _draw_separated_tapered_polyline(
                canvas,
                draw_pts,
                col,
                sw=sw,
                ew=ew,
                a0=a0,
                a1=a1,
                alpha_scale=1.0,
            )
            _draw_directional_manifold_marker(canvas, p, draw_pts, col, alpha=0.65)
            if dots and len(draw_pts) >= 2:
                cv2.circle(canvas, draw_pts[0], 3, col, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, draw_pts[-1], 3, col, -1, lineType=cv2.LINE_AA)
        out_path = paths_root_dir / f"path_atlas_ranked_panel_{pi:02d}.png"
        cv2.imwrite(str(out_path), canvas)
        written.append(str(out_path))
    return written


def write_path_rank_panels_overlay(
    *,
    paths_root_dir: Path,
    base_img_bgr: np.ndarray,
    label_map: Optional[np.ndarray],
    objects: Optional[List[Dict[str, Any]]],
    metric_depth_m: Optional[np.ndarray],
    panels: List[List[Dict[str, Any]]],
    cfg: Any,
) -> List[str]:
    """Draw ranked panel overlays on the original input image."""
    prefer_geo = bool(getattr(cfg, "path_atlas_prefer_geodesic", False)) if cfg else False
    min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
    sw = int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8
    ew = int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2
    a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95
    a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35
    with_context = bool(getattr(cfg, "path_atlas_overlay_include_context", True)) if cfg else True
    occ = bool(getattr(cfg, "path_render_occlusion_compositing", True)) if cfg else True
    written: List[str] = []
    for pi, plist in enumerate(panels, start=1):
        canvas = np.asarray(base_img_bgr, dtype=np.uint8).copy()
        ch, cw = canvas.shape[:2]
        if with_context:
            if label_map is not None:
                draw_regions_contours_bgr(canvas, label_map)
            if objects:
                draw_objects_boxes_bgr(canvas, list(objects), max_boxes=60)
        ids = [str(p.get("path_id", "")) for p in plist if str(p.get("path_id", "")).strip()]
        colors = (
            bgr_list_with_min_hue_separation(ids, bgr_from_stable_id, min_sep)
            if ids
            else []
        )
        id_to_color = dict(zip(ids, colors))
        drawable_count = max(1, len(plist))
        for draw_i, p in enumerate(plist):
            pid = str(p.get("path_id", "")).strip()
            if not pid:
                continue
            pts = int_polyline_from_path_dict(
                p, prefer_geodesic=prefer_geo, width=cw, height=ch, clamp=True
            )
            if len(pts) < 2:
                continue
            draw_pts = _lane_offset_polyline(pts, draw_i, drawable_count, cw, ch)
            col = id_to_color.get(pid) or bgr_from_stable_id(pid)
            vp = p.get("visibility_profile") or None
            wp = p.get("width_profile_px") or None
            _draw_separated_tapered_polyline(
                canvas,
                draw_pts,
                col,
                sw=sw,
                ew=ew,
                a0=a0,
                a1=a1,
                alpha_scale=1.0,
                metric_depth_m=metric_depth_m,
                occlusion_compositing=occ,
                width_profile_px=wp,
                visibility_profile=vp,
            )
            draw_direction_heads(canvas, draw_pts, col, thickness=2, tip_len=0.12, visibility_profile=vp)
            _draw_directional_manifold_marker(canvas, p, draw_pts, col, alpha=0.58)
        out_path = paths_root_dir / f"path_atlas_ranked_panel_{pi:02d}_context.png"
        cv2.imwrite(str(out_path), canvas)
        written.append(str(out_path))
    return written


def write_path_rank_panels_paths_trajectories_overlay(
    *,
    paths_root_dir: Path,
    base_img_bgr: np.ndarray,
    metric_depth_m: Optional[np.ndarray],
    panels: List[List[Dict[str, Any]]],
    traj_bundle: Optional[Dict[str, Any]],
    cfg: Any,
) -> List[str]:
    """Draw ranked panel overlays with paths + trajectories on input image."""
    prefer_geo = bool(getattr(cfg, "path_atlas_prefer_geodesic", False)) if cfg else False
    min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
    sw = int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8
    ew = int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2
    a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95
    a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35
    tsw = max(2, int(getattr(cfg, "path_atlas_trajectory_stroke_start_px", 5)) if cfg else 5)
    tew = max(1, int(getattr(cfg, "path_atlas_trajectory_stroke_end_px", 2)) if cfg else 2)
    ta0 = float(getattr(cfg, "path_stroke_alpha_start", 0.9)) if cfg else 0.9
    ta1 = float(getattr(cfg, "path_stroke_alpha_end", 0.4)) if cfg else 0.4
    hyps = (traj_bundle or {}).get("hypotheses") or []
    t_ids = [str(h.get("trajectory_id", "")).strip() for h in hyps if str(h.get("trajectory_id", "")).strip()]
    t_cols = bgr_list_with_min_hue_separation(t_ids, bgr_from_stable_id_trajectory, min_sep) if t_ids else []
    t_map = dict(zip(t_ids, t_cols))
    written: List[str] = []
    for pi, plist in enumerate(panels, start=1):
        canvas = np.asarray(base_img_bgr, dtype=np.uint8).copy()
        ch, cw = canvas.shape[:2]
        ids = [str(p.get("path_id", "")) for p in plist if str(p.get("path_id", "")).strip()]
        panel_path_ids = set(ids)
        colors = bgr_list_with_min_hue_separation(ids, bgr_from_stable_id, min_sep) if ids else []
        id_to_color = dict(zip(ids, colors))
        drawable_count = max(1, len(plist))
        for draw_i, p in enumerate(plist):
            pid = str(p.get("path_id", "")).strip()
            if not pid:
                continue
            pts = int_polyline_from_path_dict(
                p, prefer_geodesic=prefer_geo, width=cw, height=ch, clamp=True
            )
            if len(pts) < 2:
                continue
            draw_pts = _lane_offset_polyline(pts, draw_i, drawable_count, cw, ch)
            col = id_to_color.get(pid) or bgr_from_stable_id(pid)
            vp = p.get("visibility_profile") or None
            wp = p.get("width_profile_px") or None
            _draw_separated_tapered_polyline(
                canvas,
                draw_pts,
                col,
                sw=sw,
                ew=ew,
                a0=a0,
                a1=a1,
                alpha_scale=1.0,
                metric_depth_m=metric_depth_m,
                occlusion_compositing=bool(getattr(cfg, "path_render_occlusion_compositing", True)) if cfg else True,
                width_profile_px=wp,
                visibility_profile=vp,
            )
            draw_direction_heads(canvas, draw_pts, col, thickness=2, tip_len=0.12, visibility_profile=vp)
            _draw_directional_manifold_marker(canvas, p, draw_pts, col, alpha=0.58)
        for hyp in hyps:
            tid = str(hyp.get("trajectory_id", "")).strip()
            if not tid:
                continue
            pid = str((hyp.get("action_context") or {}).get("path_id") or hyp.get("continues_from_path_id") or "")
            if panel_path_ids and pid not in panel_path_ids:
                continue
            tpts = _polyline_from_trajectory_hypothesis(hyp, canvas.shape[1], canvas.shape[0])
            if len(tpts) < 2:
                continue
            tcol = t_map.get(tid) or bgr_from_stable_id_trajectory(tid)
            tapered_polyline_draw(canvas, tpts, tcol, tsw, tew, ta0, ta1, alpha_scale=0.85)
            draw_direction_heads(canvas, tpts, tcol, thickness=2, tip_len=0.14)
        out_path = paths_root_dir / f"path_atlas_ranked_panel_{pi:02d}_paths_trajectories.png"
        cv2.imwrite(str(out_path), canvas)
        written.append(str(out_path))
    return written


def _polyline_from_trajectory_hypothesis(
    hyp: Dict[str, Any],
    width: int = 0,
    height: int = 0,
) -> List[Tuple[int, int]]:
    """2D trajectory polyline from ``trajectory_points`` or first sample's ``states_t`` (NaN-safe)."""
    raw = hyp.get("trajectory_points")
    if isinstance(raw, list) and raw:
        pts_tp: List[Tuple[int, int]] = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            try:
                fx = float(t.get("x_px", 0))
                fy = float(t.get("y_px", 0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fx) or not math.isfinite(fy):
                continue
            ix, iy = int(round(fx)), int(round(fy))
            if width > 0 and height > 0:
                ix = max(0, min(width - 1, ix))
                iy = max(0, min(height - 1, iy))
            pts_tp.append((ix, iy))
        out_tp: List[Tuple[int, int]] = []
        for xy in pts_tp:
            if not out_tp or out_tp[-1] != xy:
                out_tp.append(xy)
        if len(out_tp) >= 2:
            return out_tp

    samples = hyp.get("samples") or []
    if not samples:
        return []
    st0 = (samples[0] or {}).get("states_t") or []
    pts: List[Tuple[int, int]] = []
    for row in st0:
        if not isinstance(row, dict):
            continue
        try:
            fx = float(row.get("x_px", 0))
            fy = float(row.get("y_px", 0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fx) or not math.isfinite(fy):
            continue
        ix, iy = int(round(fx)), int(round(fy))
        if width > 0 and height > 0:
            ix = max(0, min(width - 1, ix))
            iy = max(0, min(height - 1, iy))
        pts.append((ix, iy))
    out: List[Tuple[int, int]] = []
    for xy in pts:
        if not out or out[-1] != xy:
            out.append(xy)
    return out


def write_trajectory_atlas_line_only(
    *,
    paths_root_dir: Path,
    height: int,
    width: int,
    traj_bundle: Dict[str, Any],
    cfg: Any,
) -> Optional[str]:
    """Single line-only canvas for all trajectory hypotheses; returns path string or None."""
    hyps = traj_bundle.get("hypotheses") or []
    if not hyps:
        return None
    bg = getattr(cfg, "path_atlas_background_bgr", (26, 26, 26))
    if not isinstance(bg, (list, tuple)) or len(bg) < 3:
        bg = (26, 26, 26)
    bg_bgr = (int(bg[0]), int(bg[1]), int(bg[2]))
    min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
    sw = max(2, int(getattr(cfg, "path_atlas_trajectory_stroke_start_px", 5)) if cfg else 5)
    ew = max(1, int(getattr(cfg, "path_atlas_trajectory_stroke_end_px", 2)) if cfg else 2)
    a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.9)) if cfg else 0.9
    a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.4)) if cfg else 0.4

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = bg_bgr
    ids = [str(h.get("trajectory_id", "")) for h in hyps if str(h.get("trajectory_id", "")).strip()]
    colors = bgr_list_with_min_hue_separation(ids, bgr_from_stable_id_trajectory, min_sep)
    id_to_color = dict(zip(ids, colors))
    for hyp in hyps:
        tid = str(hyp.get("trajectory_id", "")).strip()
        if not tid:
            continue
        pts = _polyline_from_trajectory_hypothesis(hyp, width, height)
        if len(pts) < 2:
            continue
        col = id_to_color.get(tid) or bgr_from_stable_id_trajectory(tid)
        tapered_polyline_draw(canvas, pts, col, sw, ew, a0, a1, 1.0)
    out_path = paths_root_dir / "traj_atlas_line_only.png"
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def _traj_hyp_by_path_id(traj_bundle: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for hyp in list((traj_bundle or {}).get("hypotheses") or []):
        if not isinstance(hyp, dict):
            continue
        pid = str((hyp.get("action_context") or {}).get("path_id", ""))
        if not pid:
            pid = str(hyp.get("continues_from_path_id", ""))
        if pid and pid not in out:
            out[pid] = hyp
    return out


def _draw_path_id_label(
    canvas: np.ndarray,
    pts: List[Tuple[int, int]],
    label: str,
    color_bgr: Tuple[int, int, int],
    *,
    scale: float = 0.32,
) -> None:
    if canvas is None or len(pts) < 2 or not label:
        return
    idx = max(0, min(len(pts) - 1, len(pts) // 2))
    x, y = pts[idx]
    x = max(0, min(canvas.shape[1] - 1, int(x)))
    y = max(0, min(canvas.shape[0] - 1, int(y)))
    text = str(label)[:34]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x0 = max(0, min(canvas.shape[1] - tw - 6, x + 4))
    y0 = max(th + 4, min(canvas.shape[0] - 2, y - 4))
    cv2.rectangle(canvas, (x0 - 2, y0 - th - 3), (x0 + tw + 3, y0 + 3), (10, 10, 10), -1)
    cv2.putText(canvas, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, 1, cv2.LINE_AA)


def _int_polyline_from_key(
    path: Dict[str, Any],
    key: str,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    raw = path.get(key) or []
    if not isinstance(raw, list) or len(raw) < 2:
        return []
    out: List[Tuple[int, int]] = []
    for xy in raw:
        pt = int_xy_from_vertex(xy, width, height, clamp=True)
        if pt is None:
            continue
        if not out or out[-1] != pt:
            out.append(pt)
    return out if len(out) >= 2 else []


def _turn_defect_indices(pts: List[Tuple[int, int]], min_deg: float = 120.0) -> List[int]:
    bad: List[int] = []
    for i in range(1, len(pts) - 1):
        ax, ay = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        bx, by = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        la, lb = math.hypot(ax, ay), math.hypot(bx, by)
        if la <= 1e-6 or lb <= 1e-6:
            continue
        dot = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        if math.degrees(math.acos(dot)) >= min_deg:
            bad.append(i)
    return bad


def _draw_geometry_debug(
    canvas: np.ndarray,
    path: Dict[str, Any],
    display_pts: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    width: int,
    height: int,
) -> None:
    raw_pts = _int_polyline_from_key(path, "polyline_2d_raw", width, height)
    valid_pts = _int_polyline_from_key(path, "polyline_2d_validated", width, height)
    if len(raw_pts) >= 2:
        cv2.polylines(canvas, [np.asarray(raw_pts, dtype=np.int32)], False, (150, 150, 150), 1, cv2.LINE_AA)
    if len(valid_pts) >= 2 and valid_pts != display_pts:
        cv2.polylines(canvas, [np.asarray(valid_pts, dtype=np.int32)], False, (245, 245, 245), 1, cv2.LINE_AA)
    for idx in _turn_defect_indices(display_pts):
        cv2.circle(canvas, display_pts[idx], 5, (30, 30, 255), 2, cv2.LINE_AA)
    raw = path.get("polyline_2d_raw") or []
    p3d = path.get("polyline_3d") or []
    n = min(len(raw), len(p3d))
    if n >= 2:
        step = max(1, n // 12)
        for i in range(0, n, step):
            try:
                p0 = int_xy_from_vertex(raw[i], width, height, clamp=True)
                p1 = int_xy_from_vertex(p3d[i], width, height, clamp=True)
            except Exception:
                continue
            if not p0 or not p1:
                continue
            if math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])) >= 4.0:
                cv2.arrowedLine(canvas, p0, p1, (0, 80, 255), 1, cv2.LINE_AA, tipLength=0.28)
    quality = path.get("path_geometry_quality") if isinstance(path.get("path_geometry_quality"), dict) else {}
    reasons = list(quality.get("geometry_rejection_reasons") or [])
    if reasons and display_pts:
        x, y = display_pts[min(len(display_pts) - 1, max(0, len(display_pts) // 2))]
        txt = ",".join(str(r).replace("geometry_", "") for r in reasons[:2])[:48]
        _draw_path_id_label(canvas, display_pts, txt, (80, 190, 255), scale=0.28)
    if len(display_pts) >= 2:
        cv2.circle(canvas, display_pts[0], 4, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, display_pts[-1], 4, (255, 255, 255), 1, cv2.LINE_AA)


def _lane_offset_polyline(
    pts: List[Tuple[int, int]],
    lane_index: int,
    lane_count: int,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    if len(pts) < 2 or lane_count <= 1:
        return pts
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dx, dy = float(x1 - x0), float(y1 - y0)
    mag = math.hypot(dx, dy)
    if mag <= 1e-6:
        return pts
    nx, ny = -dy / mag, dx / mag
    offset = (float(lane_index) - (float(lane_count) - 1.0) * 0.5) * 4.0
    offset = max(-10.0, min(10.0, offset))
    out: List[Tuple[int, int]] = []
    for x, y in pts:
        ox = max(0, min(width - 1, int(round(float(x) + nx * offset))))
        oy = max(0, min(height - 1, int(round(float(y) + ny * offset))))
        if not out or out[-1] != (ox, oy):
            out.append((ox, oy))
    return out if len(out) >= 2 else pts


def write_scene_path_trajectories_overview(
    *,
    paths_root_dir: Path,
    base_img_bgr: np.ndarray,
    label_map: Optional[np.ndarray],
    objects: Optional[List[Dict[str, Any]]],
    metric_depth_m: Optional[np.ndarray],
    ranked_paths: List[Dict[str, Any]],
    traj_bundle: Optional[Dict[str, Any]],
    cfg: Any,
) -> Optional[str]:
    """Write ``path_trajectories.png`` as a single all-path scene overview.

    The detailed per-path id mapping remains in ``path_trajectories_batch_*.png``
    and ``path_visual_qa.json``; this overview is intentionally dense and shows
    every ranked path after the accepted-path cap/dedupe stage.
    """
    if not bool(getattr(cfg, "path_scene_trajectory_overview_enabled", True)) if cfg else True:
        return None
    if base_img_bgr is None or not ranked_paths:
        return None
    accepted_only = bool(getattr(cfg, "path_scene_trajectory_overview_accepted_only", True)) if cfg else True
    include_plausible = bool(getattr(cfg, "path_scene_trajectory_include_plausible_uncertain", True)) if cfg else True
    plausible_min_conf = float(getattr(cfg, "path_scene_trajectory_plausible_min_confidence", 0.30)) if cfg else 0.30
    if accepted_only:
        primary_statuses = {"accepted"}
        if include_plausible:
            primary_statuses.add("plausible_uncertain")
        primary = []
        for p in ranked_paths:
            st = str(p.get("acceptance_status", "accepted"))
            if st not in primary_statuses:
                continue
            if st == "plausible_uncertain":
                conf = float((p.get("scores") or {}).get("overall_confidence", 0.0) or 0.0)
                if conf < plausible_min_conf:
                    continue
            primary.append(p)
        if primary:
            ranked_paths = primary

    canvas = np.asarray(base_img_bgr, dtype=np.uint8).copy()
    ch, cw = canvas.shape[:2]
    with_ctx = bool(getattr(cfg, "path_scene_trajectory_overview_include_context", False)) if cfg else False
    if with_ctx:
        if label_map is not None:
            draw_regions_contours_bgr(canvas, label_map)
        if objects:
            draw_objects_boxes_bgr(canvas, list(objects), max_boxes=60)

    prefer_geo = bool(getattr(cfg, "path_atlas_prefer_geodesic", False)) if cfg else False
    min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
    ids = [str(p.get("path_id", "")) for p in ranked_paths if str(p.get("path_id", "")).strip()]
    colors = bgr_list_with_min_hue_separation(ids, bgr_from_stable_id, min_sep) if ids else []
    id_to_color = dict(zip(ids, colors))
    occ = bool(getattr(cfg, "path_render_occlusion_compositing", True)) if cfg else True
    sw = max(1, int(getattr(cfg, "path_scene_overview_path_width_px", 3)) if cfg else 3)
    ew = max(1, int(getattr(cfg, "path_scene_overview_path_end_width_px", 1)) if cfg else 1)
    alpha_scale = float(getattr(cfg, "path_scene_overview_alpha_scale", 0.38)) if cfg else 0.38
    top_label_n = int(getattr(cfg, "path_scene_overview_label_top_n", 60)) if cfg else 60
    traj_by_pid = _traj_hyp_by_path_id(traj_bundle)

    for rank, p in enumerate(ranked_paths, start=1):
        pid = str(p.get("path_id", "")).strip()
        if not pid:
            continue
        pts = int_polyline_from_path_dict(
            p, prefer_geodesic=prefer_geo, width=cw, height=ch, clamp=True
        )
        if len(pts) < 2:
            continue
        lane_idx = _stable_lane_index(pid, 9)
        draw_pts = _lane_offset_polyline(pts, lane_idx, 9, cw, ch)
        col = id_to_color.get(pid) or bgr_from_stable_id(pid)
        _draw_separated_tapered_polyline(
            canvas,
            draw_pts,
            col,
            sw=sw,
            ew=ew,
            a0=0.75,
            a1=0.28,
            alpha_scale=alpha_scale,
            metric_depth_m=metric_depth_m,
            occlusion_compositing=occ,
            width_profile_px=p.get("width_profile_px") or None,
            visibility_profile=p.get("visibility_profile") or None,
        )
        _draw_directional_manifold_marker(canvas, p, draw_pts, col, alpha=0.45)
        if rank <= top_label_n:
            _draw_path_id_label(canvas, draw_pts, f"#{rank}", col, scale=0.34)

    # Draw linked path-follow trajectories faintly on top so trajectory coverage
    # is visible without overwhelming the scene.
    for p in ranked_paths:
        pid = str(p.get("path_id", "")).strip()
        hyp = traj_by_pid.get(pid)
        if not hyp:
            continue
        tpts = _polyline_from_trajectory_hypothesis(hyp, cw, ch)
        if len(tpts) < 2:
            continue
        col = id_to_color.get(pid) or bgr_from_stable_id(pid)
        lane_idx = _stable_lane_index(pid, 9)
        tdraw = _lane_offset_polyline(tpts, lane_idx, 9, cw, ch)
        tapered_polyline_draw(canvas, tdraw, col, 2, 1, 0.55, 0.22, alpha_scale=0.35)

    title = f"path_trajectories overview: {len(ranked_paths)} accepted paths; detailed ids in batch PNGs + JSON"
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (min(cw - 1, 760), 36), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0.0, dst=canvas)
    cv2.putText(
        canvas,
        title[:110],
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    out_path = paths_root_dir / "path_trajectories.png"
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def write_scene_path_trajectory_batches(
    *,
    paths_root_dir: Path,
    base_img_bgr: np.ndarray,
    label_map: Optional[np.ndarray],
    objects: Optional[List[Dict[str, Any]]],
    metric_depth_m: Optional[np.ndarray],
    ranked_paths: List[Dict[str, Any]],
    traj_bundle: Optional[Dict[str, Any]],
    cfg: Any,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Scene-backed PNGs with strict 10 paths per image (batched over *all* ranked paths),
    each path in a distinct color; trajectories linked by ``path_id`` are drawn on top.

    Filenames: ``path_trajectories_batch_{idx:03d}.png``.
    Returns ``(absolute_paths_written, batch_metadata_dicts)``.
    """
    if not bool(getattr(cfg, "path_scene_trajectory_batches_enabled", True)) if cfg else True:
        return [], []
    if base_img_bgr is None or not ranked_paths:
        return [], []

    accepted_only = bool(getattr(cfg, "path_scene_trajectory_batches_accepted_only", True)) if cfg else True
    include_plausible = bool(getattr(cfg, "path_scene_trajectory_include_plausible_uncertain", True)) if cfg else True
    plausible_min_conf = float(getattr(cfg, "path_scene_trajectory_plausible_min_confidence", 0.30)) if cfg else 0.30
    if accepted_only:
        primary_statuses = {"accepted"}
        if include_plausible:
            primary_statuses.add("plausible_uncertain")
        primary = []
        for p in ranked_paths:
            st = str(p.get("acceptance_status", "accepted"))
            if st not in primary_statuses:
                continue
            if st == "plausible_uncertain":
                conf = float((p.get("scores") or {}).get("overall_confidence", 0.0) or 0.0)
                if conf < plausible_min_conf:
                    continue
            primary.append(p)
        if primary:
            ranked_paths = primary

    batch_sz = int(getattr(cfg, "path_scene_trajectory_batch_size", 10)) if cfg else 10
    batch_sz = max(1, batch_sz)
    prefer_geo = bool(getattr(cfg, "path_atlas_prefer_geodesic", False)) if cfg else False
    min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
    sw = int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8
    ew = int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2
    a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95
    a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35
    tsw = max(2, int(getattr(cfg, "path_atlas_trajectory_stroke_start_px", 5)) if cfg else 5)
    tew = max(1, int(getattr(cfg, "path_atlas_trajectory_stroke_end_px", 2)) if cfg else 2)
    ta0 = float(getattr(cfg, "path_stroke_alpha_start", 0.9)) if cfg else 0.9
    ta1 = float(getattr(cfg, "path_stroke_alpha_end", 0.4)) if cfg else 0.4
    with_ctx = bool(getattr(cfg, "path_scene_batch_include_context", False)) if cfg else False
    with_ctx_debug = bool(getattr(cfg, "path_scene_batch_include_context_debug", True)) if cfg else True
    occ = bool(getattr(cfg, "path_render_occlusion_compositing", True)) if cfg else True

    traj_by_pid = _traj_hyp_by_path_id(traj_bundle)
    rank_by_pid: Dict[str, int] = {}
    for i, rp in enumerate(ranked_paths):
        pk = str(rp.get("path_id", "")).strip()
        if pk and pk not in rank_by_pid:
            rank_by_pid[pk] = i + 1

    batches: List[List[Dict[str, Any]]] = []
    for lo in range(0, len(ranked_paths), batch_sz):
        batches.append(ranked_paths[lo : lo + batch_sz])

    written_abs: List[str] = []
    meta: List[Dict[str, Any]] = []

    for bi, plist in enumerate(batches):
        canvas = np.asarray(base_img_bgr, dtype=np.uint8).copy()
        ch, cw = canvas.shape[:2]
        if with_ctx:
            if label_map is not None:
                draw_regions_contours_bgr(canvas, label_map)
            if objects:
                draw_objects_boxes_bgr(canvas, list(objects), max_boxes=60)
        debug_canvas = canvas.copy()
        if with_ctx_debug:
            if label_map is not None:
                draw_regions_contours_bgr(debug_canvas, label_map)
            if objects:
                draw_objects_boxes_bgr(debug_canvas, list(objects), max_boxes=60)

        ids = [str(p.get("path_id", "")) for p in plist if str(p.get("path_id", "")).strip()]
        colors = bgr_list_with_min_hue_separation(ids, bgr_from_stable_id, min_sep) if ids else []
        id_to_color = dict(zip(ids, colors))

        path_ids: List[str] = []
        drawable_count = max(1, len(plist))
        for draw_i, p in enumerate(plist):
            pid = str(p.get("path_id", "")).strip()
            if not pid:
                continue
            pts = int_polyline_from_path_dict(
                p, prefer_geodesic=prefer_geo, width=cw, height=ch, clamp=True
            )
            if len(pts) < 2:
                continue
            draw_pts = _lane_offset_polyline(pts, draw_i, drawable_count, cw, ch)
            col = id_to_color.get(pid) or bgr_from_stable_id(pid)
            vp = p.get("visibility_profile") or None
            wp = p.get("width_profile_px") or None
            _draw_separated_tapered_polyline(
                canvas,
                draw_pts,
                col,
                sw=sw,
                ew=ew,
                a0=a0,
                a1=a1,
                alpha_scale=1.0,
                metric_depth_m=metric_depth_m,
                occlusion_compositing=occ,
                width_profile_px=wp,
                visibility_profile=vp,
            )
            draw_direction_heads(canvas, draw_pts, col, thickness=2, tip_len=0.12, visibility_profile=vp)
            _draw_directional_manifold_marker(canvas, p, draw_pts, col, alpha=0.58)
            tapered_polyline_draw(
                debug_canvas,
                pts,
                col,
                sw,
                ew,
                a0,
                a1,
                1.0,
                depth_values=sample_depth_along_polyline(pts, metric_depth_m),
                width_profile_px=wp,
                visibility_profile=vp,
                metric_depth_m=metric_depth_m,
                occlusion_compositing=occ,
            )
            draw_direction_heads(debug_canvas, pts, col, thickness=2, tip_len=0.12, visibility_profile=vp)
            if with_ctx_debug:
                _draw_geometry_debug(debug_canvas, p, pts, col, cw, ch)
            if bool(getattr(cfg, "path_scene_batch_draw_path_labels", True)) if cfg else True:
                gr = int(rank_by_pid.get(pid, 0))
                manifold = str(p.get("manifold_type", "")).replace("_path", "").replace("_", "/")
                action = str(p.get("action_family", "")).replace("_", "/")
                _draw_path_id_label(canvas, draw_pts, f"#{gr} {action or manifold}", col, scale=0.34)
                if with_ctx_debug:
                    _draw_path_id_label(debug_canvas, pts, f"#{gr} {pid[:16]}", col, scale=0.30)
            path_ids.append(pid)

        # Trajectories for paths in this batch only (same colors as trajectory_id).
        t_ids_batch: List[str] = []
        for pid in path_ids:
            hyp = traj_by_pid.get(pid)
            if not hyp:
                continue
            tid = str(hyp.get("trajectory_id", "")).strip()
            if tid:
                t_ids_batch.append(tid)
        t_cols = (
            bgr_list_with_min_hue_separation(t_ids_batch, bgr_from_stable_id_trajectory, min_sep)
            if t_ids_batch
            else []
        )
        t_map = dict(zip(t_ids_batch, t_cols))

        for pid in path_ids:
            hyp = traj_by_pid.get(pid)
            if not hyp:
                continue
            tid = str(hyp.get("trajectory_id", "")).strip()
            tpts = _polyline_from_trajectory_hypothesis(hyp, cw, ch)
            if len(tpts) < 2:
                continue
            tcol = t_map.get(tid) or bgr_from_stable_id_trajectory(tid or pid)
            tapered_polyline_draw(canvas, tpts, tcol, tsw, tew, ta0, ta1, alpha_scale=0.88)
            draw_direction_heads(canvas, tpts, tcol, thickness=2, tip_len=0.14)
            tapered_polyline_draw(debug_canvas, tpts, tcol, tsw, tew, ta0, ta1, alpha_scale=0.88)
            draw_direction_heads(debug_canvas, tpts, tcol, thickness=2, tip_len=0.14)

        # Legend: global rank, path_id, confidence, trajectory id
        leg_x, leg_y = 8, 8
        leg_line = 18
        legend_rows: List[Tuple[int, str, float, str, str, str, Tuple[int, int, int]]] = []
        for p in plist:
            pid = str(p.get("path_id", "")).strip()
            if not pid:
                continue
            if pid not in path_ids:
                continue
            gr = int(rank_by_pid.get(pid, 0))
            conf = float((p.get("scores") or {}).get("overall_confidence", 0.0))
            hyp = traj_by_pid.get(pid)
            tid = str(hyp.get("trajectory_id", "")) if hyp else ""
            pcol = id_to_color.get(pid) or bgr_from_stable_id(pid)
            manifold = str(p.get("manifold_type", ""))
            action = str(p.get("action_family", ""))
            legend_rows.append((gr, pid, conf, tid, manifold, action, pcol))

        leg_w = 420
        leg_h = 28 + len(legend_rows) * leg_line
        ov = canvas.copy()
        cv2.rectangle(ov, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.68, canvas, 0.32, 0.0, dst=canvas)
        title = f"path_trajectories batch {bi + 1}/{len(batches)} (strict {batch_sz} paths)"
        cv2.putText(
            canvas,
            title[:80],
            (leg_x + 4, leg_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        for j, (gr, pid, conf, tid, manifold, action, pcol) in enumerate(legend_rows):
            ey = leg_y + 28 + j * leg_line
            cv2.line(canvas, (leg_x + 4, ey - 4), (leg_x + 22, ey - 4), pcol, 3, cv2.LINE_AA)
            traj_note = f" | T:{tid[:18]}" if tid else ""
            label_note = f"{action or manifold}".replace("_", "/")[:18]
            line = f"#{gr} {label_note}  {pid[:20]}  conf={conf:.2f}{traj_note}"
            cv2.putText(
                canvas,
                line[:72],
                (leg_x + 26, ey),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

        out_path = paths_root_dir / f"path_trajectories_batch_{bi:03d}.png"
        cv2.imwrite(str(out_path), canvas)
        if with_ctx_debug:
            debug_out_path = paths_root_dir / f"path_context_debug_batch_{bi:03d}.png"
            cv2.imwrite(str(debug_out_path), debug_canvas)
        written_abs.append(str(out_path))
        meta.append({
            "batch_index": bi,
            "file": out_path.name,
            "path_ids": list(path_ids),
            "global_ranks": [rank_by_pid.get(pid, 0) for pid in path_ids],
        })

    return written_abs, meta
