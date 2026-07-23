"""Lightweight automated render-quality checks (Phase 2.5 item 3).

Four cheap, concrete checks against real failure modes of *this* pipeline's
own photo-compositing invariants -- not photoreal/aesthetic scoring, and not
legacy's cartoon-sprite-specific flicker/depth-percentile checks (that
system validated a stand-in sprite's own styling, not real compositing).
Run on real in-memory frames right before either tier writes its GIF/MP4, so
a broken render (wrong occlusion order, actor not scaling with depth, wrong
frame count, or a compositing bug that leaks outside the actor region) is
flagged before it reaches the user instead of only being caught by eye.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np


def check_frame_count(frames_bgr: Sequence[np.ndarray], fps: int, duration_s: float) -> Tuple[bool, str]:
    expected = max(1, round(fps * duration_s))
    actual = len(frames_bgr)
    return actual == expected, f"frame_count: expected {expected}, got {actual}"


def check_background_preserved(
    frames_bgr: Sequence[np.ndarray],
    original_bgr: np.ndarray,
    states: Sequence[Dict[str, Any]],
    object_masks_by_id: Dict[Any, np.ndarray],
    pad_factor: float = 2.0,
) -> Tuple[bool, str]:
    """Every pixel outside a dilated actor/occluder region must be
    bit-identical to the original photo -- this codebase's own stated
    compositing invariant (restated in both compositor.py's and
    render_job.py's module docstrings), verified directly rather than
    assumed."""
    if not frames_bgr:
        return True, "background_preserved: no frames to check (skipped)"
    base = original_bgr
    fh, fw = frames_bgr[0].shape[:2]
    if base.shape[:2] != (fh, fw):
        base = cv2.resize(base, (fw, fh), interpolation=cv2.INTER_AREA)
    violations = 0
    for frame, state in zip(frames_bgr, states):
        protected = np.zeros((fh, fw), dtype=np.uint8)
        cx, cy = state.get("position_px", (fw / 2.0, fh / 2.0))
        r = max(4, int(state.get("width_px", 12.0) * pad_factor))
        cv2.circle(protected, (int(cx), int(cy)), r, 1, -1)
        for occ_id in state.get("occluder_ids", []) or []:
            m = object_masks_by_id.get(occ_id)
            if m is None:
                continue
            m = np.asarray(m, dtype=np.uint8)
            if m.shape[:2] != (fh, fw):
                m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
            protected |= (m > 0).astype(np.uint8)
        diff = np.any(frame != base, axis=-1)
        if (diff & ~protected.astype(bool)).any():
            violations += 1
    ok = violations == 0
    return ok, f"background_preserved: {violations}/{len(frames_bgr)} frame(s) modified pixels outside the protected actor/occluder region"


def check_occlusion_order_consistency(
    states: Sequence[Dict[str, Any]],
    object_masks_by_id: Dict[Any, np.ndarray],
    frame_shape: Tuple[int, int],
    radius_px: float = 10.0,
) -> Tuple[bool, str]:
    """A frame claiming occlusion (``render_layer != "in_front"``) must have
    at least one of its ``occluder_ids``' real masks actually overlapping
    the actor's on-screen position there -- catches a stale/wrong
    ``visibility_profile`` rather than trusting it blindly."""
    h, w = frame_shape
    bad = 0
    checked = 0
    for state in states:
        if state.get("render_layer") == "in_front":
            continue
        occluder_ids = state.get("occluder_ids", []) or []
        if not occluder_ids:
            continue
        checked += 1
        cx, cy = state.get("position_px", (0.0, 0.0))
        x0, x1 = max(0, int(cx - radius_px)), min(w, int(cx + radius_px))
        y0, y1 = max(0, int(cy - radius_px)), min(h, int(cy + radius_px))
        if x1 <= x0 or y1 <= y0:
            bad += 1
            continue
        found = False
        for occ_id in occluder_ids:
            m = object_masks_by_id.get(occ_id)
            if m is None:
                continue
            m = np.asarray(m, dtype=bool)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if m[y0:y1, x0:x1].any():
                found = True
                break
        if not found:
            bad += 1
    ok = bad == 0
    return ok, f"occlusion_order_consistency: {bad}/{max(1, checked)} occluded frame(s) had no real occluder-mask overlap near the actor"


def check_depth_scale_monotonicity(states: Sequence[Dict[str, Any]], min_z_variance: float = 1e-3) -> Tuple[bool, str]:
    """``width_px`` should correlate negatively (or near-zero) with ``z_m``
    across the frame schedule -- farther should not render larger. Skips
    (passes) when there aren't enough real depth samples or depth barely
    varies along the path, rather than false-flagging a flat-depth path."""
    pairs = [
        (s.get("z_m"), s.get("width_px"))
        for s in states
        if s.get("z_m") is not None and s.get("width_px") is not None
    ]
    if len(pairs) < 3:
        return True, "depth_scale_monotonicity: not enough depth samples to evaluate (skipped)"
    z_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    w_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    if float(np.var(z_arr)) < min_z_variance:
        return True, "depth_scale_monotonicity: depth is ~flat across the path (skipped)"
    corr = float(np.corrcoef(z_arr, w_arr)[0, 1])
    ok = corr <= 0.2  # allow noise near zero; only flag a clearly inverted (positive) relationship
    return ok, f"depth_scale_monotonicity: correlation(z_m, width_px) = {corr:.3f} (expect <= 0.2)"


def run_render_qa(
    frames_bgr: Sequence[np.ndarray],
    original_bgr: np.ndarray,
    states: Sequence[Dict[str, Any]],
    object_masks_by_id: Dict[Any, np.ndarray],
    fps: int,
    duration_s: float,
) -> Dict[str, Any]:
    """Runs all four checks, returns a combined pass/fail summary to attach
    as ``result["qa"]`` from either render tier."""
    frame_shape = frames_bgr[0].shape[:2] if frames_bgr else (0, 0)
    checks = [
        check_frame_count(frames_bgr, fps, duration_s),
        check_background_preserved(frames_bgr, original_bgr, states, object_masks_by_id),
        check_occlusion_order_consistency(states, object_masks_by_id, frame_shape),
        check_depth_scale_monotonicity(states),
    ]
    failed = [msg for ok, msg in checks if not ok]
    return {"passed": len(failed) == 0, "checks": [msg for _, msg in checks], "failed_checks": failed}
