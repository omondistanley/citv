"""Tier 2 photorealistic animation: real-mask cutout + depth-aware compositing
directly onto the original uploaded photo.

Every frame this module produces starts from an unmodified copy of the real
photo; the only things ever drawn on top are (in order) a contact shadow,
the actor sprite (alpha-blended, depth-scaled, color-harmonized), and any
occluding object's *original* pixels restored on top of the sprite where a
foreground object's real silhouette should hide it. Nothing here rebuilds or
repaints the scene itself.

Consumes the ``path_hypotheses.json`` v3 fields ``paths_export.py`` already
produces (``polyline_2d``, ``width_profile_px``, ``alpha_profile``,
``visibility_profile``, ``kinematic_signatures``) -- this module does not
recompute any of that, only renders it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from scene_understanding.animation.render_qa import run_render_qa

# Relative speed multipliers used to convert kinematic-signature motion
# labels into non-uniform frame timing (a "jump" segment should take a
# fixed, short real-world time regardless of how many path-samples it
# spans; "walk"/"run"/"crawl" segments consume path length at their own pace).
_SPEED_MAP: Dict[str, float] = {
    "walk": 1.0, "run": 2.2, "crawl": 0.35, "climb": 0.5,
    "jump": 0.9, "descend": 0.8,
    # swim/fly: continuous, path-length-consuming motions like walk/run/crawl
    # (not fixed-duration point events like jump/climb/descend below), added
    # for paths_export.py's terrain-aware motion override (Phase 2.5).
    "swim": 0.6, "fly": 1.3,
}
_FIXED_DURATION_MOTIONS = {"jump": 0.5, "climb": 0.9, "descend": 0.6}


def extract_actor_sprite(
    img_bgr: np.ndarray, mask_bool: np.ndarray, feather_px: int = 2, pad_px: int = 3,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Real per-pixel silhouette cutout (not an ellipse/bbox approximation).

    Returns ``(rgba_uint8, (origin_x, origin_y))`` where the RGBA crop's
    top-left corner sits at ``origin_xy`` in the source image. ``feather_px``
    only softens the mask *boundary* by a couple of pixels for anti-aliasing
    -- it does not shrink the silhouette toward an ellipse.
    """
    mask = np.asarray(mask_bool, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty mask -- nothing to extract a sprite from")
    h, w = mask.shape[:2]
    x0, x1 = max(0, xs.min() - pad_px), min(w, xs.max() + 1 + pad_px)
    y0, y1 = max(0, ys.min() - pad_px), min(h, ys.max() + 1 + pad_px)

    crop_bgr = img_bgr[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].astype(np.float32)
    if feather_px > 0:
        crop_mask = cv2.GaussianBlur(crop_mask, (0, 0), feather_px)
    alpha = np.clip(crop_mask * 255.0, 0, 255).astype(np.uint8)

    rgba = np.dstack([crop_bgr, alpha])
    return rgba, (int(x0), int(y0))


def _interp(arr: Sequence[Any], idx_f: float) -> Any:
    n = len(arr)
    if n == 0:
        return None
    if n == 1:
        return arr[0]
    idx_f = max(0.0, min(float(n - 1), idx_f))
    i0 = int(np.floor(idx_f))
    i1 = min(i0 + 1, n - 1)
    frac = idx_f - i0
    a, b = arr[i0], arr[i1]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * frac
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [av + (bv - av) * frac for av, bv in zip(a, b)]
    return a  # discrete/non-numeric fields (e.g. render_layer) -- take the floor sample


class PathTrace:
    """Continuous-index sampler over a ``path_hypotheses.json`` v3 hypothesis's
    parallel per-sample arrays, so rendering isn't limited to the coarse
    (<=12 point) ``animation_render_contract.sample_state_preview``."""

    def __init__(self, hypothesis: Dict[str, Any], position_source: Optional[str] = None):
        """``position_source`` overrides which hypothesis field supplies
        positions (default ``polyline_2d``, pixel coords). Tier 1's
        Blender renderer passes ``"polyline_3d_smoothed"`` instead (true
        metric world XYZ) to drive the same kinematic-timing logic without
        duplicating it -- the rest of this class (width/alpha/occlusion
        sampling, ``frame_schedule``) is position-representation-agnostic."""
        self.polyline_2d: List[List[float]] = hypothesis[position_source or "polyline_2d"]
        self.n = len(self.polyline_2d)
        self.width_profile_px: List[float] = hypothesis.get("width_profile_px") or [12.0] * self.n
        self.alpha_profile: List[float] = hypothesis.get("alpha_profile") or [1.0] * self.n
        visibility = hypothesis.get("visibility_profile") or []
        self.render_layers: List[str] = [v.get("render_layer", "in_front") for v in visibility] or ["in_front"] * self.n
        self.occluder_ids: List[List[Any]] = [v.get("occluder_ids", []) for v in visibility] or [[] for _ in range(self.n)]
        self.z_m_profile: List[Optional[float]] = [v.get("z_m") for v in visibility] or [None] * self.n
        self.kinematic_signatures: List[Dict[str, Any]] = hypothesis.get("kinematic_signatures") or []

    def sample(self, s: float) -> Dict[str, Any]:
        idx_f = max(0.0, min(1.0, s)) * (self.n - 1)
        i0 = int(np.floor(idx_f))
        position = _interp(self.polyline_2d, idx_f)
        width = _interp(self.width_profile_px, idx_f)
        alpha = _interp(self.alpha_profile, idx_f)
        nxt = self.polyline_2d[min(i0 + 1, self.n - 1)]
        heading = float(np.degrees(np.arctan2(nxt[1] - position[1], nxt[0] - position[0])))
        z_m = self.z_m_profile[min(i0, len(self.z_m_profile) - 1)] if self.z_m_profile else None
        return {
            "position_px": (float(position[0]), float(position[1])),
            "width_px": float(width),
            "alpha": float(np.clip(alpha, 0.0, 1.0)),
            "render_layer": self.render_layers[min(i0, len(self.render_layers) - 1)],
            "occluder_ids": self.occluder_ids[min(i0, len(self.occluder_ids) - 1)],
            "heading_deg": heading,
            "z_m": z_m,
        }

    def motion_at(self, s: float) -> str:
        idx = max(0.0, min(1.0, s)) * (self.n - 1)
        for seg in self.kinematic_signatures:
            if seg["start_idx"] <= idx <= seg["end_idx"]:
                return str(seg.get("motion", "walk"))
        return "walk"


def frame_schedule(trace: PathTrace, fps: int, duration_s: float) -> List[float]:
    """Kinematic-signature-driven frame -> path-progress ``s`` map: motions
    with a fixed real-world duration (jump/climb/descend) consume a fixed
    time slice regardless of path length; walk/run/crawl consume path length
    at a speed proportional to ``_SPEED_MAP``. Falls back to uniform timing
    when no kinematic signatures are available."""
    n_frames = max(1, int(round(fps * duration_s)))
    if not trace.kinematic_signatures:
        return [i / max(1, n_frames - 1) for i in range(n_frames)]

    segs = trace.kinematic_signatures
    total_idx = max(1, trace.n - 1)
    seg_durations = []
    for seg in segs:
        motion = str(seg.get("motion", "walk"))
        span = max(1, seg["end_idx"] - seg["start_idx"])
        if motion in _FIXED_DURATION_MOTIONS:
            seg_durations.append(_FIXED_DURATION_MOTIONS[motion])
        else:
            speed = _SPEED_MAP.get(motion, 1.0)
            seg_durations.append((span / total_idx) * duration_s / max(speed, 1e-3))
    total_duration = sum(seg_durations) or duration_s
    time_scale = duration_s / total_duration

    schedule: List[float] = []
    for i in range(n_frames):
        t = (i / max(1, n_frames - 1)) * duration_s
        acc_t, acc_idx = 0.0, 0
        s = 1.0
        for seg, seg_dur in zip(segs, seg_durations):
            seg_dur_scaled = seg_dur * time_scale
            if t <= acc_t + seg_dur_scaled or seg is segs[-1]:
                frac = 0.0 if seg_dur_scaled <= 1e-9 else (t - acc_t) / seg_dur_scaled
                idx = seg["start_idx"] + frac * (seg["end_idx"] - seg["start_idx"])
                s = float(np.clip(idx / total_idx, 0.0, 1.0))
                break
            acc_t += seg_dur_scaled
        schedule.append(s)
    return schedule


def cast_contact_shadow(
    canvas_bgr: np.ndarray, foot_xy: Tuple[float, float], width_px: float,
    alpha: float = 0.35, support_mask: Optional[np.ndarray] = None,
) -> None:
    """Soft elliptical contact shadow, darkened multiplicatively into the
    photo beneath the actor's feet -- mutates ``canvas_bgr`` in place."""
    h, w = canvas_bgr.shape[:2]
    cx, cy = int(round(foot_xy[0])), int(round(foot_xy[1]))
    rx, ry = max(2, int(width_px * 0.55)), max(1, int(width_px * 0.22))
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(shadow_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), max(1.0, ry * 0.4))
    if support_mask is not None:
        shadow_mask *= np.asarray(support_mask, dtype=np.float32)
    darken = 1.0 - alpha * shadow_mask[..., None]
    canvas_bgr[:] = np.clip(canvas_bgr.astype(np.float32) * darken, 0, 255).astype(np.uint8)


def harmonize_sprite_color(sprite_rgba: np.ndarray, background_bgr: np.ndarray, blend: float = 0.3) -> np.ndarray:
    """Shift the sprite's mean color/brightness partway toward the local
    background patch it's about to be placed on, so it reads as lit by the
    same scene rather than pasted on. Keeps the sprite's own texture/detail
    -- this is a mean-color shift, not a full histogram transplant."""
    rgb = sprite_rgba[..., :3].astype(np.float32)
    alpha = sprite_rgba[..., 3].astype(np.float32) / 255.0
    fg_mask = alpha > 0.05
    if not fg_mask.any():
        return sprite_rgba
    sprite_mean = rgb[fg_mask].mean(axis=0)
    bg_mean = background_bgr.reshape(-1, 3).astype(np.float32).mean(axis=0)
    shift = (bg_mean - sprite_mean) * blend
    rgb = np.clip(rgb + shift[None, None, :], 0, 255).astype(np.uint8)
    out = sprite_rgba.copy()
    out[..., :3] = rgb
    return out


def _paste_rgba(canvas_bgr: np.ndarray, sprite_rgba: np.ndarray, center_xy: Tuple[float, float], scale: float, global_alpha: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Alpha-blend a scaled RGBA sprite onto ``canvas_bgr`` centered at
    ``center_xy``. Returns the resized alpha mask placed at full canvas
    resolution (for occlusion fix-up) and the paste origin."""
    h, w = canvas_bgr.shape[:2]
    sh, sw = sprite_rgba.shape[:2]
    new_w, new_h = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(sprite_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    ox, oy = int(round(center_xy[0] - new_w / 2)), int(round(center_xy[1] - new_h))  # feet at center_xy
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + new_w), min(h, oy + new_h)
    full_alpha = np.zeros((h, w), dtype=np.float32)
    if x1 <= x0 or y1 <= y0:
        return full_alpha, (ox, oy)

    sx0, sy0 = x0 - ox, y0 - oy
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    sprite_crop = resized[sy0:sy1, sx0:sx1]
    a = (sprite_crop[..., 3].astype(np.float32) / 255.0) * global_alpha
    region = canvas_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = region * (1 - a[..., None]) + sprite_crop[..., :3].astype(np.float32) * a[..., None]
    canvas_bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    full_alpha[y0:y1, x0:x1] = a
    return full_alpha, (ox, oy)


def composite_frame(
    original_bgr: np.ndarray,
    sprite_rgba: np.ndarray,
    state: Dict[str, Any],
    object_masks_by_id: Dict[Any, np.ndarray],
    reference_width_px: float,
    support_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One fully-composited frame: original photo -> contact shadow -> actor
    (alpha- and depth-scaled) -> real-silhouette occluder restore."""
    canvas = original_bgr.copy()
    position = state["position_px"]
    scale = state["width_px"] / max(reference_width_px, 1e-3)

    cast_contact_shadow(canvas, position, state["width_px"], alpha=0.3, support_mask=support_mask)

    bg_patch = original_bgr[
        max(0, int(position[1]) - 10):int(position[1]) + 2,
        max(0, int(position[0]) - 15):int(position[0]) + 15,
    ]
    harmonized = harmonize_sprite_color(sprite_rgba, bg_patch) if bg_patch.size else sprite_rgba

    _paste_rgba(canvas, harmonized, position, scale, state["alpha"])

    # Real-silhouette occlusion: restore the occluder's own pixels on top of
    # the actor wherever that specific foreground object's mask covers it,
    # instead of a uniform frame-wide dim.
    if state["render_layer"] != "in_front":
        for occ_id in state.get("occluder_ids", []):
            occ_mask = object_masks_by_id.get(occ_id)
            if occ_mask is None:
                continue
            m = np.asarray(occ_mask, dtype=bool)
            if m.shape[:2] != canvas.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            canvas[m] = original_bgr[m]
    return canvas


def render_animation(
    original_bgr: np.ndarray,
    hypothesis: Dict[str, Any],
    actor_mask: Optional[np.ndarray],
    object_masks_by_id: Dict[Any, np.ndarray],
    out_gif_path: str,
    out_mp4_path: Optional[str] = None,
    fps: int = 24,
    duration_s: float = 4.0,
    support_mask: Optional[np.ndarray] = None,
    max_width: int = 960,
    actor_sprite_rgba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Render ``hypothesis`` (a ``path_hypotheses.json`` v3 entry) as an
    animation of an actor moving along it, composited onto ``original_bgr``.
    Writes both a GIF and (if ``out_mp4_path`` given) an MP4; every frame's
    background pixels outside the composited region are the original photo,
    unchanged.

    The actor sprite normally comes from cutting out ``actor_mask``'s real
    silhouette from the photo (Tier 2's core "no ellipse cutout" rule). Pass
    ``actor_sprite_rgba`` instead (Phase 3: an uploaded actor image) to use
    it directly as the sprite -- it takes priority over ``actor_mask``,
    which becomes optional in that case (RGB-only assets are treated as
    fully opaque).
    """
    if actor_sprite_rgba is not None:
        sprite_rgba = actor_sprite_rgba if actor_sprite_rgba.shape[2] == 4 else np.dstack(
            [actor_sprite_rgba, np.full(actor_sprite_rgba.shape[:2], 255, np.uint8)]
        )
    elif actor_mask is not None:
        sprite_rgba, _ = extract_actor_sprite(original_bgr, actor_mask)
    else:
        raise ValueError("render_animation needs either actor_mask or actor_sprite_rgba")
    reference_width_px = float(np.median(hypothesis.get("width_profile_px") or [12.0]))
    trace = PathTrace(hypothesis)
    schedule = frame_schedule(trace, fps, duration_s)

    h, w = original_bgr.shape[:2]
    scale = min(1.0, max_width / w)
    render_w, render_h = int(w * scale), int(h * scale)
    base_resized = cv2.resize(original_bgr, (render_w, render_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else original_bgr

    def _resize_mask(m: np.ndarray) -> np.ndarray:
        if scale >= 1.0:
            return m
        return cv2.resize(np.asarray(m, dtype=np.uint8), (render_w, render_h), interpolation=cv2.INTER_NEAREST) > 0

    resized_masks_by_id = {k: _resize_mask(v) for k, v in object_masks_by_id.items()}
    resized_support = _resize_mask(support_mask) if support_mask is not None else None
    if scale < 1.0:
        sprite_resized = cv2.resize(sprite_rgba, (max(1, int(sprite_rgba.shape[1] * scale)), max(1, int(sprite_rgba.shape[0] * scale))))
    else:
        sprite_resized = sprite_rgba

    frames_bgr: List[np.ndarray] = []
    states: List[Dict[str, Any]] = []
    for s in schedule:
        state = trace.sample(s)
        state["position_px"] = (state["position_px"][0] * scale, state["position_px"][1] * scale)
        state["width_px"] = state["width_px"] * scale
        frame = composite_frame(
            base_resized, sprite_resized, state, resized_masks_by_id,
            reference_width_px * scale, support_mask=resized_support,
        )
        frames_bgr.append(frame)
        states.append(state)

    write_gif(frames_bgr, out_gif_path, fps)
    if out_mp4_path:
        write_mp4(frames_bgr, out_mp4_path, fps)

    qa = run_render_qa(frames_bgr, base_resized, states, resized_masks_by_id, fps, duration_s)

    return {
        "frame_count": len(frames_bgr), "fps": fps, "duration_s": duration_s,
        "qa": qa,
        "gif_path": out_gif_path, "mp4_path": out_mp4_path,
        "motions_used": sorted({trace.motion_at(s) for s in schedule}),
    }


def write_gif(frames_bgr: Sequence[np.ndarray], out_path: str, fps: int) -> None:
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=128, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG) for f in frames_bgr]
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:], duration=int(1000 / max(1, fps)), loop=0)


def write_mp4(frames_bgr: Sequence[np.ndarray], out_path: str, fps: int) -> None:
    h, w = frames_bgr[0].shape[:2]
    for codec in ("avc1", "mp4v", "MJPG"):
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            break
    else:
        raise RuntimeError("no working video codec found for MP4 export")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
