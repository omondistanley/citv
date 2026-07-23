"""Tier 1: CPU-only 3D render-and-composite via headless Blender.

``render_job`` is the main-process side (no ``bpy`` dependency -- importable
and testable without Blender installed). ``blender_render_actor.py`` is the
script Blender itself executes (``bpy``-only, cannot be imported or unit
tested outside a real Blender install) and is invoked as a subprocess.
"""
from __future__ import annotations

from scene_understanding.animation.tier1_blender.render_job import (
    build_job_spec,
    composite_blender_render,
    find_blender_executable,
    render_tier1_animation,
    run_blender_render,
)

__all__ = [
    "build_job_spec",
    "composite_blender_render",
    "find_blender_executable",
    "render_tier1_animation",
    "run_blender_render",
]
