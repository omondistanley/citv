"""
Composite QA figure: RGB + cost + speed + text legend for path planning fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _resize_to_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return bgr
    if h == target_h:
        return bgr
    scale = target_h / float(h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)


def _title_bar(img: np.ndarray, title: str, bar_h: int = 34) -> np.ndarray:
    """Stack a dark title strip above image."""
    h, w = img.shape[:2]
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:bar_h, :] = (36, 36, 36)
    out[bar_h:, :] = img
    t = title[:72] if len(title) > 72 else title
    cv2.putText(
        out,
        t,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return out


def _draw_path_overlay(canvas: np.ndarray, pts: List[Tuple[int, int]], bgr: Tuple[int, int, int], thick: int = 2) -> None:
    if len(pts) < 2:
        return
    h, w = canvas.shape[:2]
    arr = np.array([[max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y)))] for x, y in pts], dtype=np.int32)
    cv2.polylines(canvas, [arr], False, bgr, thick, lineType=cv2.LINE_AA)


def build_path_fields_legend_payload(
    path_stem: str,
    trav_meta: Dict[str, Any],
    export_trav_speed: bool,
) -> Dict[str, Any]:
    return {
        "schema": "citv_path_fields_legend_v1",
        "image_stem": path_stem,
        "panels": [
            {
                "id": "rgb",
                "title": "RGB + sample paths",
                "description": "Scene color. Colored polylines = top-ranked legacy routes (polyline_2d). Green = geodesic when present.",
            },
            {
                "id": "path_cost_map",
                "title": "path_cost_map (legacy A*)",
                "value_range": [0.0, 1.0],
                "colormap": "INFERNO",
                "bright_means": "higher cost (more expensive to traverse)",
                "components": [
                    "RGB Sobel edges (strong edges cost more)",
                    "Gaussian-smoothed union of object masks (obstacle cost)",
                    "Penalty outside depth-partition regions (label <= 0)",
                    "1 - distance-to-free-space from obstacles (centering / open corridor bias)",
                ],
                "does_not_use": ["metric_depth_m"],
            },
            {
                "id": "path_traversability_speed",
                "title": "path_traversability_speed (geodesic)",
                "value_range": "trav_speed_floor .. 1.0 (see config)",
                "colormap": "VIRIDIS",
                "bright_means": "higher speed (easier motion; geodesic prefers these pixels)",
                "components": [
                    "Feasible = (region label > 0) AND NOT obstacle mask; elsewhere low speed floor",
                    "Depth-gradient flatness: exp(-|grad z| / scale) when metric depth is available",
                    "Mild RGB edge penalty: 1 - w_edge * normalized Sobel magnitude",
                ],
                "exported_speed_png": bool(export_trav_speed),
                "traversability_meta": dict(trav_meta) if trav_meta else {},
            },
        ],
        "reading_tip": "Compare cost vs speed: texture can dominate cost, while depth jumps dominate speed.",
    }


def build_path_fields_explainer_image(
    img_bgr: np.ndarray,
    cost_map: np.ndarray,
    speed_map: np.ndarray,
    paths_sorted: List[Dict[str, Any]],
    cfg: Any,
) -> np.ndarray:
    """
    2x2 layout: RGB+paths | cost+paths / speed+paths | legend text.
    """
    panel_h = int(getattr(cfg, "path_fields_explainer_panel_h", 380)) if cfg else 380
    max_paths = int(getattr(cfg, "path_fields_explainer_max_paths", 4)) if cfg else 4
    bar_h = 34

    rgb = _resize_to_height(np.asarray(img_bgr), panel_h)
    h0, w0 = rgb.shape[:2]

    cost_u8 = np.clip(np.asarray(cost_map, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    cost_rgb = cv2.applyColorMap(cost_u8, cv2.COLORMAP_INFERNO)
    cost_rgb = _resize_to_height(cost_rgb, panel_h)
    sm = np.asarray(speed_map, dtype=np.float32)
    sm_u8 = np.clip(sm * 255.0, 0, 255).astype(np.uint8)
    speed_rgb = cv2.applyColorMap(sm_u8, cv2.COLORMAP_VIRIDIS)
    speed_rgb = _resize_to_height(speed_rgb, panel_h)

    def _color(pid: str) -> Tuple[int, int, int]:
        hsh = abs(hash(str(pid)))
        return (int(50 + (hsh % 205)), int(50 + ((hsh // 7) % 205)), int(50 + ((hsh // 49) % 205)))

    for target in (rgb, cost_rgb, speed_rgb):
        th, tw = target.shape[:2]
        scale_x = tw / float(w0) if w0 else 1.0
        scale_y = th / float(h0) if h0 else 1.0
        for p in paths_sorted[:max_paths]:
            pts = [(int(xy[0] * scale_x), int(xy[1] * scale_y)) for xy in (p.get("polyline_2d") or [])]
            if len(pts) < 2:
                continue
            _draw_path_overlay(target, pts, _color(str(p.get("path_id", ""))), thick=2)
            geo = p.get("polyline_geodesic_2d")
            if isinstance(geo, list) and len(geo) >= 2:
                gpts = [(int(xy[0] * scale_x), int(xy[1] * scale_y)) for xy in geo]
                _draw_path_overlay(target, gpts, (60, 255, 120), thick=2)

    p1 = _title_bar(rgb, "A) RGB + paths  (color=legacy  green=geodesic)")
    p2 = _title_bar(cost_rgb, "B) path_cost_map  INFERNO  bright = HIGH cost (A*)")
    p3 = _title_bar(speed_rgb, "C) path_traversability_speed  VIRIDIS  bright = HIGH speed")

    w_cell = max(p1.shape[1], p2.shape[1], p3.shape[1])
    h_cell = max(p1.shape[0], p2.shape[0], p3.shape[0])

    def _pad_to_cell(img: np.ndarray) -> np.ndarray:
        ih, iw = img.shape[:2]
        out = np.ones((h_cell, w_cell, 3), dtype=np.uint8) * 240
        out[:ih, :iw] = img
        return out

    p1c = _pad_to_cell(p1)
    p2c = _pad_to_cell(p2)
    p3c = _pad_to_cell(p3)

    legend_w = w_cell
    legend_inner_h = max(1, h_cell - bar_h)
    legend = np.ones((legend_inner_h, legend_w, 3), dtype=np.uint8) * 255
    lines = [
        "D) How to read",
        "",
        "A: Scene + top paths.",
        "Colored = polyline_2d.",
        "Green = geodesic.",
        "",
        "B: path_cost_map 0..1",
        "INFERNO: bright =",
        "expensive (A*).",
        "RGB edges + masks +",
        "regions + corridor.",
        "No metric depth.",
        "",
        "C: speed map ..1",
        "VIRIDIS: bright =",
        "easy (Dijkstra).",
        "Depth flatness +",
        "mild RGB penalty.",
        "",
        "JSON: path_fields",
        "_legend.json",
    ]
    y = 20
    for line in lines:
        cv2.putText(
            legend,
            line[:44],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        y += 24
    p4 = _title_bar(legend, "D) Legend (see path_fields_legend.json)")

    gap = 10
    sep = np.full((h_cell, gap, 3), 220, dtype=np.uint8)
    row1 = np.hstack([p1c, sep, p2c])
    row2 = np.hstack([p3c, sep, p4])
    hsep = np.full((gap, row1.shape[1], 3), 220, dtype=np.uint8)
    out = np.vstack([row1, hsep, row2])
    return out
