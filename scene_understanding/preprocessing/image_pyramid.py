"""Image pyramid helper — Phase B.6.

Builds every downscaled image view the pipeline needs exactly once and
passes them, by name, to the wrappers that need them. Previous code paths
rebuilt SAM2 / GDINO / Florence-2 / depth processor inputs independently
from the same source image, which duplicated cv2.resize + colour-space
conversions several times per frame.

Usage::

    pyramid = ImagePyramid(
        full_bgr=img_bgr,
        max_sides={
            "gdino": 1280,
            "sam2": 1024,
            "depth": 1024,
            "caption": 768,
        },
    )
    bgr_for_gdino = pyramid.bgr("gdino")
    rgb_for_depth = pyramid.rgb("depth")

Every getter is memoised so repeated lookups cost only a dict probe. Levels
whose max side is >= the source image just return the source (no copy) to
avoid throwing away fine detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _Level:
    max_side: int
    bgr: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    scale: float = 1.0
    size: Tuple[int, int] = (0, 0)  # (w, h) at this level

    def resolve_scale(self, src_h: int, src_w: int) -> float:
        if self.max_side <= 0:
            return 1.0
        long_edge = max(src_h, src_w)
        if long_edge <= self.max_side:
            return 1.0
        return float(self.max_side) / float(long_edge)


class ImagePyramid:
    """Lazy multi-resolution cache for a single source image.

    All views share the same colour-space mapping rules:
        ``bgr(level)`` returns OpenCV's native BGR layout.
        ``rgb(level)`` returns RGB (e.g. for PIL, HF processors, SAM2).
    """

    __slots__ = ("_src_bgr", "_src_rgb", "_levels", "_src_shape")

    def __init__(
        self,
        full_bgr: np.ndarray,
        max_sides: Mapping[str, int] | None = None,
    ) -> None:
        if full_bgr is None or full_bgr.size == 0:
            raise ValueError("ImagePyramid requires a non-empty BGR image")
        if full_bgr.ndim != 3 or full_bgr.shape[2] != 3:
            raise ValueError(
                f"ImagePyramid expects an HxWx3 BGR image, got shape={full_bgr.shape}"
            )
        self._src_bgr = full_bgr
        self._src_rgb: Optional[np.ndarray] = None
        self._src_shape = (int(full_bgr.shape[0]), int(full_bgr.shape[1]))
        defaults: Dict[str, int] = {
            "full": 0,
            "gdino": 1280,
            "sam2": 1024,
            "depth": 1024,
            "caption": 768,
        }
        if max_sides:
            for k, v in max_sides.items():
                defaults[str(k)] = int(v)
        self._levels: Dict[str, _Level] = {
            name: _Level(max_side=int(ms)) for name, ms in defaults.items()
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def source_hw(self) -> Tuple[int, int]:
        return self._src_shape

    def configured_levels(self) -> Dict[str, int]:
        return {name: lvl.max_side for name, lvl in self._levels.items()}

    def register_level(self, name: str, max_side: int) -> None:
        """Add or override a level after construction."""
        key = str(name)
        self._levels[key] = _Level(max_side=int(max_side))

    def scale_for(self, level: str) -> float:
        lvl = self._levels.get(level) or self._levels["full"]
        if lvl.bgr is None:
            _ = self.bgr(level)
        return lvl.scale

    def size_for(self, level: str) -> Tuple[int, int]:
        lvl = self._levels.get(level) or self._levels["full"]
        if lvl.bgr is None:
            _ = self.bgr(level)
        return lvl.size

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def bgr(self, level: str = "full") -> np.ndarray:
        lvl = self._levels.get(level)
        if lvl is None:
            # Unknown level — synthesise a passthrough level so downstream
            # callers do not crash; use source dims as the implicit bound.
            lvl = _Level(max_side=0)
            self._levels[level] = lvl
        if lvl.bgr is not None:
            return lvl.bgr
        h, w = self._src_shape
        scale = lvl.resolve_scale(h, w)
        if scale >= 1.0:
            lvl.bgr = self._src_bgr
            lvl.scale = 1.0
            lvl.size = (w, h)
            return lvl.bgr
        nw, nh = int(round(w * scale)), int(round(h * scale))
        lvl.bgr = cv2.resize(self._src_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        lvl.scale = scale
        lvl.size = (nw, nh)
        return lvl.bgr

    def rgb(self, level: str = "full") -> np.ndarray:
        lvl = self._levels.get(level)
        if lvl is None:
            lvl = _Level(max_side=0)
            self._levels[level] = lvl
        if lvl.rgb is not None:
            return lvl.rgb
        bgr = self.bgr(level)
        # When the level is the full source and we already converted the
        # source once, return the cached copy.
        if lvl.scale == 1.0 and self._src_rgb is not None and bgr is self._src_bgr:
            lvl.rgb = self._src_rgb
            return lvl.rgb
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if lvl.scale == 1.0 and bgr is self._src_bgr:
            self._src_rgb = rgb
        lvl.rgb = rgb
        return rgb

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def scale_box_from_source(self, level: str, box_xyxy) -> list:
        """Map an xyxy box from source resolution onto ``level``."""
        s = self.scale_for(level)
        if s == 1.0:
            return [float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])]
        return [float(box_xyxy[0]) * s, float(box_xyxy[1]) * s, float(box_xyxy[2]) * s, float(box_xyxy[3]) * s]

    def scale_box_to_source(self, level: str, box_xyxy) -> list:
        """Inverse of :meth:`scale_box_from_source`."""
        s = self.scale_for(level)
        if s == 1.0 or s <= 0.0:
            return [float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])]
        inv = 1.0 / s
        return [
            float(box_xyxy[0]) * inv,
            float(box_xyxy[1]) * inv,
            float(box_xyxy[2]) * inv,
            float(box_xyxy[3]) * inv,
        ]
