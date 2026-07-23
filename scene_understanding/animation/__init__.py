"""Photorealistic animation renderers.

``compositor`` is Tier 2 (2D render-and-composite, CPU-only). Every renderer
in this package shares one contract: the original uploaded photo's pixels
are the immutable background plate in every output frame -- only the actor
(a real segmentation-mask cutout here; a rendered RGBA pass in Tier 3
elsewhere) plus a shadow/contact layer are ever composited on top.
"""
from __future__ import annotations

from scene_understanding.animation.compositor import render_animation

__all__ = ["render_animation"]
