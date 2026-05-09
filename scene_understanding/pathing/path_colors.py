"""Process-stable BGR colors from string ids (SHA-256), for atlas / QA exports."""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """h,s,v in 0..1 (h wraps); returns BGR uint8."""
    h2 = (h % 1.0) * 179.0
    s2 = max(0.0, min(1.0, s)) * 255.0
    v2 = max(0.0, min(1.0, v)) * 255.0
    pix = cv2.cvtColor(
        np.uint8([[[h2, s2, v2]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    return (int(pix[0]), int(pix[1]), int(pix[2]))


def _hsv_from_bgr(bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    x = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    return float(hsv[0]) / 179.0, float(hsv[1]) / 255.0, float(hsv[2]) / 255.0


def bgr_from_stable_id(id_str: str) -> Tuple[int, int, int]:
    """
    Deterministic saturated BGR from any string (e.g. path_id).
    Uses SHA-256 so colors match across machines and Python runs (unlike hash()).
    """
    raw = hashlib.sha256(str(id_str).encode("utf-8")).digest()
    u = int.from_bytes(raw[:4], "big", signed=False)
    hue = (u % 10000) / 10000.0
    s = 0.72 + (raw[4] % 28) / 100.0
    v = 0.82 + (raw[5] % 18) / 100.0
    return _hsv_to_bgr(hue, s, v)


def bgr_from_stable_id_trajectory(id_str: str) -> Tuple[int, int, int]:
    """Distinct palette: slightly lower saturation, for trajectory_id lines."""
    raw = hashlib.sha256(("traj:" + str(id_str)).encode("utf-8")).digest()
    u = int.from_bytes(raw[:4], "big", signed=False)
    hue = (u % 10000) / 10000.0
    s = 0.45 + (raw[4] % 35) / 100.0
    v = 0.88 + (raw[5] % 12) / 100.0
    return _hsv_to_bgr(hue + 0.03, s, v)


def bgr_to_hex_rgb(bgr: Tuple[int, int, int]) -> str:
    """Markdown-friendly #RRGGBB (sRGB, for human legend)."""
    b, g, r = (int(max(0, min(255, x))) for x in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def bgr_list_with_min_hue_separation(
    ids: Sequence[str],
    base_fn,
    min_hue_sep: float = 0.07,
) -> List[Tuple[int, int, int]]:
    """
    Assign colors per id; nudge hue if consecutive colors would be too similar.
    base_fn: bgr_from_stable_id or bgr_from_stable_id_trajectory
    """
    out: List[Tuple[int, int, int]] = []
    last_h: float | None = None
    for sid in ids:
        bgr = base_fn(sid)
        if min_hue_sep <= 0:
            out.append(bgr)
            last_h = _hsv_from_bgr(bgr)[0]
            continue
        h, s, v = _hsv_from_bgr(bgr)
        if last_h is not None:
            dh = abs(h - last_h)
            dh = min(dh, 1.0 - dh)
            if dh < min_hue_sep:
                h = (last_h + min_hue_sep) % 1.0
                bgr = _hsv_to_bgr(h, min(0.95, s + 0.05), min(0.98, v))
        out.append(bgr)
        last_h = _hsv_from_bgr(bgr)[0]
    return out
