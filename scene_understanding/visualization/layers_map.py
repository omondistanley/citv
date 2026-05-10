"""Layer-colored bbox map (contrasts with path_context which does not encode layer on boxes)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def save_layers_map_bgr(
    image_bgr: np.ndarray,
    objects_3d: List[Dict[str, Any]],
    out_path: Path,
    regions_meta: Optional[List[Dict[str, Any]]] = None,
) -> None:
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    colour_for = {
        "foreground": (0, 255, 0),
        "midground": (0, 165, 255),
        "background": (255, 0, 0),
        "unassigned": (160, 160, 160),
    }

    occupied: List[Tuple[int, int, int, int]] = []

    def _overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def _place(bx: int, by: int, text: str, scale: float = 0.38) -> Tuple[int, int]:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        for tx, ty in [(bx + 4, by - 4), (bx + 4, by + th + 4), (bx - tw - 4, by - 4)]:
            box = (max(0, tx - 2), max(0, ty - th - 2), min(w - 1, tx + tw + 2), min(h - 1, ty + 2))
            if not any(_overlap(box, o) for o in occupied):
                occupied.append(box)
                return tx, ty
        box = (max(0, bx - 2), max(0, by - th - 6), min(w - 1, bx + tw + 2), min(h - 1, by + 2))
        occupied.append(box)
        return bx, by

    for obj in objects_3d:
        bbox = obj.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        layer_type = str(obj.get("layer_type", "unassigned"))
        colour = colour_for.get(layer_type, (160, 160, 160))
        label = f"{obj.get('label', 'object')} ({layer_type})"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        tx, ty = _place(x1, max(16, y1 - 4), label)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    if regions_meta:
        for r in regions_meta:
            layer_type = str(r.get("layer_type", r.get("type", "unassigned")))
            colour = colour_for.get(layer_type, (255, 165, 0))
            c = r.get("centroid_2d_px") or [w // 2, h // 2]
            cx = int(min(max(0, float(c[0])), w - 1))
            cy = int(min(max(0, float(c[1])), h - 1))
            cv2.drawMarker(canvas, (cx, cy), colour, cv2.MARKER_DIAMOND, 14, 2)
            r_sem = (
                str(r.get("semantic_label", "") or r.get("canonical_name", "") or layer_type).strip().lower()
                or layer_type
            )
            rid = str(r.get("id", ""))
            label = f"[{rid}] {r_sem} ({layer_type})" if rid else f"{r_sem} ({layer_type})"
            tx, ty = _place(cx + 8, cy, label, scale=0.36)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, colour, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
