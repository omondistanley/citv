"""BGR drawing helpers for path context composites and related overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from scene_understanding.pathing.path_polyline_utils import int_polyline_from_path_dict


def draw_regions_contours_bgr(img_bgr: np.ndarray, label_map: np.ndarray) -> None:
    """Draw thin region boundaries over the image (in-place)."""
    if img_bgr is None or label_map is None:
        return
    lm = np.asarray(label_map, dtype=np.int32)
    h, w = lm.shape[:2]
    b = np.zeros((h, w), dtype=np.uint8)
    b[1:, :] |= (lm[1:, :] != lm[:-1, :]).astype(np.uint8) * 255
    b[:, 1:] |= (lm[:, 1:] != lm[:, :-1]).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    b = cv2.dilate(b, k, iterations=1)
    m = b > 0
    if np.any(m):
        img_bgr[m] = (0, 255, 255)


def draw_objects_boxes_bgr(img_bgr: np.ndarray, objects: List[Dict[str, Any]], max_boxes: int = 40) -> None:
    """Draw lightweight bbox + label for context (in-place)."""
    if img_bgr is None:
        return
    count = 0
    for obj in (objects or []):
        if count >= max_boxes:
            break
        if str(obj.get("entity_kind", "object")) == "region":
            continue
        bbox = obj.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        label = str(obj.get("canonical_name") or obj.get("name") or obj.get("label") or "obj")
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 180, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            img_bgr,
            label[:18],
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 180, 0),
            1,
            lineType=cv2.LINE_AA,
        )
        count += 1


def _draw_dashed_segment(
    canvas: np.ndarray,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    thickness: int,
    dash_px: int = 8,
    gap_px: int = 6,
) -> None:
    """Draw a dashed line between p0 and p1."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = float(x1 - x0), float(y1 - y0)
    length = float(np.hypot(dx, dy))
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    s = 0.0
    while s < length:
        e = min(length, s + float(dash_px))
        a = (int(round(x0 + ux * s)), int(round(y0 + uy * s)))
        b = (int(round(x0 + ux * e)), int(round(y0 + uy * e)))
        cv2.line(canvas, a, b, color_bgr, max(1, thickness), cv2.LINE_AA)
        s = e + float(gap_px)


def tapered_polyline_draw(
    img_bgr: np.ndarray,
    pts: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    start_w: int,
    end_w: int,
    alpha_start: float,
    alpha_end: float,
    alpha_scale: float = 1.0,
    depth_values: Optional[List[float]] = None,
    width_profile_px: Optional[List[Dict[str, Any]]] = None,
    visibility_profile: Optional[List[Dict[str, Any]]] = None,
    metric_depth_m: Optional[np.ndarray] = None,
    occlusion_compositing: bool = True,
) -> None:
    """Per-segment draw with depth + width + visibility-aware compositing.

    Plan §2.6: when ``visibility_profile`` is supplied, segments whose
    ``render_layer == "behind_object"`` are drawn first then the foreground
    occluder pixels (from ``metric_depth_m``) are re-blitted on top so the
    line passes physically behind. ``partially_occluded`` segments use a
    dashed style with reduced alpha. ``in_front`` segments are solid.
    """
    if img_bgr is None or len(pts) < 2:
        return
    asc = max(0.0, min(1.0, float(alpha_scale)))
    nseg = max(1, len(pts) - 1)
    depth_arr: Optional[np.ndarray] = None
    if depth_values is not None and len(depth_values) >= len(pts):
        arr = np.asarray(depth_values[: len(pts)], dtype=np.float32)
        finite = np.isfinite(arr)
        if np.any(finite):
            dmin = float(np.nanpercentile(arr[finite], 5))
            dmax = float(np.nanpercentile(arr[finite], 95))
            if dmax > dmin + 1e-6:
                depth_arr = (arr - dmin) / (dmax - dmin)
                depth_arr = np.clip(depth_arr, 0.0, 1.0)

    # Width profile: prefer the per-vertex physical width when supplied
    # (plan §11 width-from-depth formula). Falls back to start_w/end_w taper.
    explicit_widths: Optional[List[float]] = None
    if width_profile_px and len(width_profile_px) >= len(pts):
        explicit_widths = []
        for row in width_profile_px[: len(pts)]:
            try:
                wpx = float((row or {}).get("width_px", 0.0))
            except (TypeError, ValueError):
                wpx = 0.0
            explicit_widths.append(max(1.0, min(80.0, wpx)))

    occluder_pixels: Optional[np.ndarray] = None
    canvas_h, canvas_w = img_bgr.shape[:2]

    for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
        t = i / max(1, nseg - 1)
        if depth_arr is not None:
            # Depth-aware styling only: nearer segments are thicker/less transparent.
            t = float(0.5 * t + 0.5 * depth_arr[i])
        if explicit_widths is not None:
            w = int(round(explicit_widths[i]))
        else:
            w = int(round(start_w + (end_w - start_w) * t))
        a = float(alpha_start + (alpha_end - alpha_start) * t)
        w = max(1, w)
        a = max(0.0, min(1.0, a * asc))

        layer = "in_front"
        visible_fraction = 1.0
        if visibility_profile and i < len(visibility_profile):
            vp = visibility_profile[i] or {}
            layer = str(vp.get("render_layer", "in_front"))
            try:
                visible_fraction = float(vp.get("visible_fraction", 1.0))
            except (TypeError, ValueError):
                visible_fraction = 1.0

        if layer == "partially_occluded":
            a *= 0.55
            overlay = img_bgr.copy()
            _draw_dashed_segment(overlay, p0, p1, color_bgr, w)
            cv2.addWeighted(overlay, a, img_bgr, 1.0 - a, 0.0, dst=img_bgr)
        elif layer == "behind_object" and occlusion_compositing and metric_depth_m is not None:
            # Draw the segment, then re-blit foreground depth-closer pixels on top.
            overlay = img_bgr.copy()
            cv2.line(overlay, p0, p1, color_bgr, w, lineType=cv2.LINE_AA)
            # Determine the actor depth at this segment from depth_values (if any).
            actor_z = None
            if depth_values and i < len(depth_values):
                try:
                    actor_z = float(depth_values[i])
                except (TypeError, ValueError):
                    actor_z = None
            if actor_z is not None and np.isfinite(actor_z):
                if occluder_pixels is None:
                    dm = np.asarray(metric_depth_m, dtype=np.float32)
                    occluder_pixels = dm
                # Per-segment foreground mask (depth closer than actor by margin).
                margin_m = 0.05
                fg = (occluder_pixels < (actor_z - margin_m)) & np.isfinite(occluder_pixels)
                if fg.any():
                    # Restrict to a tight bbox around the segment for speed.
                    x0p, y0p = p0
                    x1p, y1p = p1
                    pad = max(20, w * 2)
                    x_lo = max(0, min(x0p, x1p) - pad)
                    x_hi = min(canvas_w, max(x0p, x1p) + pad)
                    y_lo = max(0, min(y0p, y1p) - pad)
                    y_hi = min(canvas_h, max(y0p, y1p) + pad)
                    region_fg = fg[y_lo:y_hi, x_lo:x_hi]
                    if region_fg.any():
                        overlay_region = overlay[y_lo:y_hi, x_lo:x_hi]
                        original_region = img_bgr[y_lo:y_hi, x_lo:x_hi]
                        overlay_region[region_fg] = original_region[region_fg]
            cv2.addWeighted(overlay, a, img_bgr, 1.0 - a, 0.0, dst=img_bgr)
        else:
            overlay = img_bgr.copy()
            cv2.line(overlay, p0, p1, color_bgr, w, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, a, img_bgr, 1.0 - a, 0.0, dst=img_bgr)


def draw_direction_heads(
    img_bgr: np.ndarray,
    pts: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    tip_len: float = 0.14,
    visibility_profile: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Draw explicit direction head from the final visible segment.

    Plan §2.6: when ``visibility_profile`` is supplied, the arrowhead is
    placed on the last segment whose ``render_layer == "in_front"`` so
    arrows don't end up floating on top of an occluder.
    """
    if img_bgr is None or len(pts) < 2:
        return
    p0 = None
    p1 = None
    n = len(pts) - 1
    for offset in range(n):
        idx = n - 1 - offset
        if visibility_profile and idx < len(visibility_profile):
            layer = str((visibility_profile[idx] or {}).get("render_layer", "in_front"))
            if layer != "in_front":
                continue
        a = pts[idx]
        b = pts[idx + 1]
        if a != b:
            p0 = a
            p1 = b
            break
    if p0 is None or p1 is None:
        # Fallback: any non-degenerate segment.
        for a, b in zip(reversed(pts[:-1]), reversed(pts[1:])):
            if a != b:
                p0 = a
                p1 = b
                break
    if p0 is None or p1 is None:
        return
    cv2.arrowedLine(img_bgr, p0, p1, color_bgr, max(1, int(thickness)), cv2.LINE_AA, tipLength=float(tip_len))


def sample_depth_along_polyline(
    pts: List[Tuple[int, int]],
    metric_depth_m: Optional[np.ndarray],
) -> Optional[List[float]]:
    if metric_depth_m is None or len(pts) < 2:
        return None
    dm = np.asarray(metric_depth_m, dtype=np.float32)
    h, w = dm.shape[:2]
    out: List[float] = []
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h and np.isfinite(dm[y, x]):
            out.append(float(dm[y, x]))
        else:
            out.append(np.nan)
    return out


def path_color_from_path_id(pid: str) -> Tuple[int, int, int]:
    hsh = abs(hash(str(pid)))
    return (int(50 + (hsh % 205)), int(50 + ((hsh // 7) % 205)), int(50 + ((hsh // 49) % 205)))


def write_path_context_top5_png(
    *,
    paths_root_dir: Path,
    img_bgr: np.ndarray,
    lm: np.ndarray,
    objs: List[Dict[str, Any]],
    paths: List[Dict[str, Any]],
    cfg: Any,
    metric_depth_m: Optional[np.ndarray] = None,
) -> None:
    """Write ``path_context_top5.png`` (filename fixed; K from ``path_context_top_k``)."""
    export_ctx = bool(getattr(cfg, "path_export_context_composites", True)) if cfg else True
    if not export_ctx:
        return
    ctx_top_k = int(getattr(cfg, "path_context_top_k", 5)) if cfg else 5
    ctx_top_k = max(0, ctx_top_k)
    if ctx_top_k <= 0:
        return
    ranked = sorted(
        paths,
        key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
        reverse=True,
    )[:ctx_top_k]

    ctx_all = img_bgr.copy()
    ih, iw = ctx_all.shape[:2]
    draw_regions_contours_bgr(ctx_all, lm)
    draw_objects_boxes_bgr(ctx_all, objs, max_boxes=50)
    for p in ranked:
        pid = str(p.get("path_id", ""))
        pts = int_polyline_from_path_dict(
            p, prefer_geodesic=False, width=iw, height=ih, clamp=True
        )
        if len(pts) < 2:
            continue
        col = path_color_from_path_id(pid)
        vp = p.get("visibility_profile") or None
        wp = p.get("width_profile_px") or None
        tapered_polyline_draw(
            ctx_all,
            pts,
            col,
            start_w=int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8,
            end_w=int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2,
            alpha_start=float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95,
            alpha_end=float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35,
            depth_values=sample_depth_along_polyline(pts, metric_depth_m),
            width_profile_px=wp,
            visibility_profile=vp,
            metric_depth_m=metric_depth_m,
            occlusion_compositing=bool(getattr(cfg, "path_render_occlusion_compositing", True)) if cfg else True,
        )
        draw_direction_heads(ctx_all, pts, col, thickness=2, tip_len=0.12, visibility_profile=vp)
    ctx_all_path = paths_root_dir / "path_context_top5.png"
    cv2.imwrite(str(ctx_all_path), ctx_all)
