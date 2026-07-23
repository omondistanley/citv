"""Tier 1 orchestrator: builds a render job from a path hypothesis, runs
headless Blender (``blender_render_actor.py``) as a subprocess, and
composites the resulting per-frame (actor RGBA, shadow alpha) pair onto the
real photo -- the photo is the immutable background in every frame, exactly
like Tier 2; only the rendered actor + its shadow are ever drawn on top.

**Environment limitation, stated plainly:** this module has no ``bpy``
import and is fully unit-testable (job-spec building, compositing math), but
``run_blender_render``/the end-to-end render itself requires a real Blender
install, which is not available in this development environment (no
``blender`` binary here). The job-spec and compositing logic below are
tested against synthetic frame files standing in for Blender's output; the
actual Blender subprocess call and ``blender_render_actor.py`` script are
unverified pending a real Blender install (the user's machine or the
deployment VM) -- flag this explicitly rather than claim untested confidence.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from scene_understanding.animation.compositor import PathTrace, frame_schedule, write_gif, write_mp4
from scene_understanding.animation.render_qa import run_render_qa

_BLENDER_SCRIPT = Path(__file__).parent / "blender_render_actor.py"
# Default actor body attributes (see motion_contract.py's
# _DEFAULT_ACTOR_ATTRS / classify_actor_attributes_openvocab) -- a generic
# upright creature. Kept in sync manually since blender_render_actor.py
# can't import from the rest of the package (bpy-only Python).
_DEFAULT_ACTOR_ATTRS: Dict[str, Any] = {"leg_count": 2, "has_wings": False, "has_tail": False, "elongated": False}
_DEFAULT_ACTOR_COLOR: Tuple[float, float, float] = (0.55, 0.55, 0.55)


def _sanitize_actor_attrs(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Clamps/coerces whatever actor_attrs the caller passed into the exact
    shape blender_render_actor.py's ``_build_creature`` expects, so a bad or
    partial dict degrades to safe defaults instead of a Blender-side crash
    (that subprocess has no way to report a clean Python traceback back)."""
    attrs = attrs or {}
    leg_count = attrs.get("leg_count", 2)
    if leg_count not in (0, 2, 4):
        leg_count = 2
    return {
        "leg_count": leg_count,
        "has_wings": bool(attrs.get("has_wings", False)),
        "has_tail": bool(attrs.get("has_tail", False)),
        "elongated": bool(attrs.get("elongated", False)),
    }


def find_blender_executable() -> Optional[str]:
    return shutil.which("blender")


def build_job_spec(
    hypothesis: Dict[str, Any],
    intrinsics: Dict[str, float],
    image_size: Tuple[int, int],
    fps: int = 24,
    duration_s: float = 4.0,
    actor_attrs: Optional[Dict[str, Any]] = None,
    color_rgb: Optional[Tuple[float, float, float]] = None,
    action_style: Optional[Dict[str, Any]] = None,
    retargeted_motion: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Per-frame world-space actor states (position/heading/motion/scale),
    reusing Tier 2's ``PathTrace``/``frame_schedule`` kinematic-timing logic
    against ``polyline_3d_smoothed`` (true metric world XYZ) instead of pixel
    positions -- WHEN each frame happens and WHERE the actor is don't need
    to be recomputed for a 3D renderer, only WHAT draws it changes.

    ``retargeted_motion`` (from ``motion_contract.ground_motion_contract``'s
    ``GroundedMotionContract.retargeted_motion``, itself from
    ``motion_retargeting.build_retargeted_motion``) is already resampled to
    exactly ``len(schedule)`` frames by the caller's contract, so it's
    attached here index-for-index -- no further resampling needed.
    """
    if "polyline_3d_smoothed" not in hypothesis:
        raise ValueError("hypothesis has no polyline_3d_smoothed -- run it through paths_export.py's 3D lift first")
    trace = PathTrace(hypothesis, position_source="polyline_3d_smoothed")
    schedule = frame_schedule(trace, fps, duration_s)
    reference_width = float(np.median(trace.width_profile_px)) or 1.0

    frames: List[Dict[str, Any]] = []
    for frame_idx, s in enumerate(schedule):
        state = trace.sample(s)
        idx = max(0.0, min(1.0, s)) * (trace.n - 1)
        i0 = int(np.floor(idx))
        cur = trace.polyline_2d[i0]  # (x, y, z) world meters despite the pixel-oriented field name
        nxt = trace.polyline_2d[min(i0 + 1, trace.n - 1)]
        # World heading in the horizontal (X-Z) plane -- Y is "up" per
        # smooth_polyline_in_3d's convention, so it doesn't drive heading.
        heading_deg = float(np.degrees(np.arctan2(nxt[0] - cur[0], nxt[2] - cur[2])))
        frame_entry: Dict[str, Any] = {
            "world_xyz": [float(cur[0]), float(cur[1]), float(cur[2])],
            "heading_deg": heading_deg,
            "motion": trace.motion_at(s),
            "scale": float(np.clip(state["width_px"] / reference_width, 0.6, 1.8)),
            "alpha": state["alpha"],
        }
        if retargeted_motion:
            for key in ("bob", "lean_deg", "limb_swing"):
                values = retargeted_motion.get(key)
                if values and frame_idx < len(values):
                    frame_entry[f"retargeted_{key}"] = float(values[frame_idx])
        frames.append(frame_entry)

    return {
        "schema": "citv_tier1_render_job_v1",
        "frames": frames,
        "fps": fps,
        "intrinsics": {k: float(v) for k, v in intrinsics.items()},
        "image_size": [int(image_size[0]), int(image_size[1])],
        "actor_attrs": _sanitize_actor_attrs(actor_attrs),
        "color_rgb": list(color_rgb) if color_rgb is not None else list(_DEFAULT_ACTOR_COLOR),
        "action_style": {
            "is_static": bool((action_style or {}).get("is_static", False)),
            "energy": (action_style or {}).get("energy", "normal") if (action_style or {}).get("energy") in ("high", "low", "normal") else "normal",
        },
    }


def run_blender_render(job_spec: Dict[str, Any], work_dir: Path, blender_executable: Optional[str] = None, timeout_s: int = 1800) -> Dict[str, List[str]]:
    """Runs Blender headless on ``job_spec``; returns per-frame actor-RGBA
    and shadow-alpha PNG paths. Raises ``RuntimeError`` (with a clear reason)
    if Blender isn't installed or the render fails -- callers should catch
    this and fall back to Tier 2, since only Tier 1 has this external
    dependency."""
    blender_bin = blender_executable or find_blender_executable()
    if not blender_bin or not (shutil.which(blender_bin) or Path(blender_bin).is_file()):
        raise RuntimeError(
            f"Blender executable not found{f' at {blender_bin!r}' if blender_bin else ' on PATH'}. "
            "Tier 1 (3D render) needs a local, headless-capable Blender install "
            "(e.g. `apt install blender` or https://www.blender.org/download/)."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / "job.json"
    job_path.write_text(json.dumps(job_spec))
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    cmd = [
        blender_bin, "--background", "--factory-startup", "--python", str(_BLENDER_SCRIPT),
        "--", "--job", str(job_path), "--out", str(frames_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender render failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    n_frames = len(job_spec["frames"])
    actor_frames = [str(frames_dir / f"actor_{i:04d}.png") for i in range(n_frames)]
    shadow_frames = [str(frames_dir / f"shadow_{i:04d}.png") for i in range(n_frames)]
    missing = [p for p in actor_frames + shadow_frames if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Blender render finished but {len(missing)} expected frame(s) are missing (e.g. {missing[0]})")
    return {"actor_frames": actor_frames, "shadow_frames": shadow_frames}


def composite_blender_render(
    original_bgr: np.ndarray,
    render_result: Dict[str, List[str]],
    fps: int,
    occluder_ids_per_frame: List[List[Any]],
    object_masks_by_id: Dict[Any, np.ndarray],
    out_gif_path: str,
    out_mp4_path: Optional[str] = None,
    shadow_strength: float = 0.4,
    qa_states: Optional[List[Dict[str, Any]]] = None,
    duration_s: float = 0.0,
) -> Dict[str, Any]:
    """Composite each (actor RGBA, shadow alpha) frame pair onto the real
    photo: shadow darkens first, then the actor alpha-blends on top, then
    any occluding object's real pixels are restored over the actor -- same
    photo-is-truth / real-silhouette-occlusion rules as Tier 2's compositor.

    Blender renders the actor already correctly placed/scaled in full-frame
    camera space (that's what the intrinsics-matched camera in
    ``blender_render_actor.py`` is for), so unlike Tier 2 there's no
    crop/paste-position math here -- just a full-frame alpha composite.
    """
    frames_out: List[np.ndarray] = []
    actor_paths, shadow_paths = render_result["actor_frames"], render_result["shadow_frames"]
    for i, (actor_path, shadow_path) in enumerate(zip(actor_paths, shadow_paths)):
        canvas = original_bgr.copy()

        shadow_img = cv2.imread(shadow_path, cv2.IMREAD_GRAYSCALE)
        if shadow_img is not None:
            shadow_alpha = shadow_img.astype(np.float32) / 255.0
            canvas[:] = np.clip(canvas.astype(np.float32) * (1.0 - shadow_strength * shadow_alpha[..., None]), 0, 255).astype(np.uint8)

        actor_rgba = cv2.imread(actor_path, cv2.IMREAD_UNCHANGED)
        if actor_rgba is not None:
            if actor_rgba.shape[2] == 3:
                actor_rgba = np.dstack([actor_rgba, np.full(actor_rgba.shape[:2], 255, np.uint8)])
            alpha = actor_rgba[..., 3:4].astype(np.float32) / 255.0
            canvas[:] = np.clip(
                canvas.astype(np.float32) * (1 - alpha) + actor_rgba[..., :3].astype(np.float32) * alpha, 0, 255,
            ).astype(np.uint8)

        for occ_id in (occluder_ids_per_frame[i] if i < len(occluder_ids_per_frame) else []):
            occ_mask = object_masks_by_id.get(occ_id)
            if occ_mask is not None:
                m = np.asarray(occ_mask, dtype=bool)
                if m.shape[:2] == canvas.shape[:2]:
                    canvas[m] = original_bgr[m]
        frames_out.append(canvas)

    write_gif(frames_out, out_gif_path, fps)
    if out_mp4_path:
        write_mp4(frames_out, out_mp4_path, fps)

    result: Dict[str, Any] = {"frame_count": len(frames_out), "gif_path": out_gif_path, "mp4_path": out_mp4_path}
    if qa_states is not None:
        result["qa"] = run_render_qa(frames_out, original_bgr, qa_states, object_masks_by_id, fps, duration_s)
    return result


def render_tier1_animation(
    original_bgr: np.ndarray,
    hypothesis: Dict[str, Any],
    intrinsics: Dict[str, float],
    object_masks_by_id: Dict[Any, np.ndarray],
    work_dir: Path,
    out_gif_path: str,
    out_mp4_path: Optional[str] = None,
    fps: int = 24,
    duration_s: float = 4.0,
    actor_attrs: Optional[Dict[str, Any]] = None,
    color_rgb: Optional[Tuple[float, float, float]] = None,
    action_style: Optional[Dict[str, Any]] = None,
    retargeted_motion: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """End-to-end Tier 1 entrypoint: build job -> run Blender -> composite.
    Raises ``RuntimeError`` (propagated from ``run_blender_render``) if
    Blender isn't available -- callers (e.g. ``app.py``) should catch this
    and fall back to Tier 2 rather than fail the whole request."""
    h, w = original_bgr.shape[:2]
    job_spec = build_job_spec(
        hypothesis, intrinsics, (w, h), fps=fps, duration_s=duration_s, actor_attrs=actor_attrs, color_rgb=color_rgb,
        action_style=action_style,
        retargeted_motion=retargeted_motion,
    )
    render_result = run_blender_render(job_spec, work_dir)

    visibility = hypothesis.get("visibility_profile") or []
    trace = PathTrace(hypothesis, position_source="polyline_3d_smoothed")
    schedule = frame_schedule(trace, fps, duration_s)
    occluder_ids_per_frame = []
    for s in schedule:
        idx = int(max(0.0, min(1.0, s)) * (trace.n - 1))
        occluder_ids_per_frame.append(visibility[idx].get("occluder_ids", []) if idx < len(visibility) else [])

    # Separate pixel-space trace purely for render_qa: `trace` above samples
    # polyline_3d_smoothed (true metric world XYZ, what Blender's camera
    # needs) so its own "position_px"/"width_px" aren't screen pixels --
    # render_qa's checks (background-preserved circle, occlusion-overlap
    # window) need actual on-screen pixel coordinates, which is what the
    # default position_source (polyline_2d) gives, on the same index space.
    qa_trace = PathTrace(hypothesis)
    qa_states = [qa_trace.sample(s) for s in schedule]

    return composite_blender_render(
        original_bgr, render_result, fps, occluder_ids_per_frame, object_masks_by_id, out_gif_path, out_mp4_path,
        qa_states=qa_states, duration_s=duration_s,
    )
