"""BGR drawing helpers for path context composites and related overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


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


def tapered_polyline_draw(
    img_bgr: np.ndarray,
    pts: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    start_w: int,
    end_w: int,
    alpha_start: float,
    alpha_end: float,
    alpha_scale: float = 1.0,
) -> None:
    if img_bgr is None or len(pts) < 2:
        return
    asc = max(0.0, min(1.0, float(alpha_scale)))
    nseg = max(1, len(pts) - 1)
    for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
        t = i / max(1, nseg - 1)
        w = int(round(start_w + (end_w - start_w) * t))
        a = float(alpha_start + (alpha_end - alpha_start) * t)
        w = max(1, w)
        a = max(0.0, min(1.0, a * asc))
        overlay = img_bgr.copy()
        cv2.line(overlay, p0, p1, color_bgr, w, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, a, img_bgr, 1.0 - a, 0.0, dst=img_bgr)


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
    draw_regions_contours_bgr(ctx_all, lm)
    draw_objects_boxes_bgr(ctx_all, objs, max_boxes=50)
    for p in ranked:
        pid = str(p.get("path_id", ""))
        pts = [tuple(map(int, xy)) for xy in (p.get("polyline_2d") or [])]
        if len(pts) < 2:
            continue
        col = path_color_from_path_id(pid)
        tapered_polyline_draw(
            ctx_all,
            pts,
            col,
            start_w=int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8,
            end_w=int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2,
            alpha_start=float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95,
            alpha_end=float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35,
        )
        sx, sy = pts[0]
        gx, gy = pts[-1]
        cv2.arrowedLine(ctx_all, (sx, sy), (gx, gy), col, 2, cv2.LINE_AA, tipLength=0.12)
    ctx_all_path = paths_root_dir / "path_context_top5.png"
    cv2.imwrite(str(ctx_all_path), ctx_all)
