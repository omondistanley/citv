"""Motion retargeting from an uploaded video clip.

Extracts real body-pose keypoints per frame via MediaPipe's Pose Landmarker
(CPU-only Tasks API -- no GPU dependency, unlike Tier 3) and derives a
normalized motion signal (vertical hip bob, torso lean, leg-spread/stride
proxy) resampled to this project's own render frame count -- the uploaded
clip's motion *shape and timing character* carry over regardless of its
original fps/frame count/duration, which is the "preserve timing and style,
then ground to the scene" policy this project's design reference (the
codex branch's `uploaded_animation.retargeting_policy`) already named.

This is the Tier 1 (procedural puppet) retargeting path specifically --
Tier 2 is a photo cutout with no rig to retarget motion onto, and Tier 3's
own use of an uploaded clip (if any) is a generative-model conditioning
concern, documented separately in ``docs/TIER3_GENERATIVE_VIDEO_DESIGN.md``.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_MODEL_CACHE_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "pose_landmarker_lite.task"

# MediaPipe's 33-point pose landmark indices used to derive the motion signal.
_LEFT_HIP, _RIGHT_HIP = 23, 24
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_ANKLE, _RIGHT_ANKLE = 27, 28

Landmarks = Dict[int, Tuple[float, float, float, float]]  # index -> (x, y, z, visibility), normalized image coords


def _ensure_model_downloaded(cache_path: Path = _MODEL_CACHE_PATH) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, str(cache_path))
    return cache_path


def extract_pose_landmarks_from_video(video_path: str, max_frames: int = 300, model_path: Optional[Path] = None) -> List[Optional[Landmarks]]:
    """Runs MediaPipe Pose (CPU Tasks API) over every frame of ``video_path``.
    Returns one entry per frame -- a landmark dict, or ``None`` for frames
    where no person was detected (e.g. actor out of frame, clip doesn't
    contain a person at all)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    resolved_model = model_path or _ensure_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=str(resolved_model))
    options = mp_vision.PoseLandmarkerOptions(base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: List[Optional[Landmarks]] = []
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened() and frame_idx < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * (1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                results.append({i: (p.x, p.y, p.z, p.visibility) for i, p in enumerate(lm)})
            else:
                results.append(None)
            frame_idx += 1
    cap.release()
    return results


def derive_motion_signal(landmarks_sequence: List[Optional[Landmarks]]) -> Dict[str, List[float]]:
    """From raw per-frame landmarks, derive three signals over the clip's
    own timeline: ``bob`` (vertical hip-center oscillation around its own
    mean; positive = higher), ``lean_deg`` (torso angle from vertical,
    forward/back), ``limb_swing`` (normalized left-right ankle spread, a
    proxy for stride phase -- not a full per-joint retarget, but enough to
    drive the procedural puppet's own limb-swing parameter meaningfully).
    Frames with no detection are linearly interpolated from neighboring
    valid frames. Returns empty lists (caller falls back to the canned
    per-motion style) if not a single frame had a detected person."""
    n = len(landmarks_sequence)
    hip_y = np.full(n, np.nan)
    lean_deg = np.full(n, np.nan)
    limb_spread = np.full(n, np.nan)

    for i, lm in enumerate(landmarks_sequence):
        if lm is None:
            continue
        lhip, rhip = lm[_LEFT_HIP], lm[_RIGHT_HIP]
        lsh, rsh = lm[_LEFT_SHOULDER], lm[_RIGHT_SHOULDER]
        lank, rank = lm[_LEFT_ANKLE], lm[_RIGHT_ANKLE]
        hip_center = ((lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0)
        shoulder_center = ((lsh[0] + rsh[0]) / 2.0, (lsh[1] + rsh[1]) / 2.0)
        hip_y[i] = -hip_center[1]  # image y grows downward -- flip so "up" reads as positive
        dx, dy = shoulder_center[0] - hip_center[0], hip_center[1] - shoulder_center[1]
        lean_deg[i] = float(np.degrees(np.arctan2(dx, max(dy, 1e-6))))
        limb_spread[i] = float(abs(lank[0] - rank[0]))

    def _fill_gaps(arr: np.ndarray) -> List[float]:
        valid = ~np.isnan(arr)
        if not valid.any():
            return []
        idx = np.arange(len(arr))
        return np.interp(idx, idx[valid], arr[valid]).tolist()

    hip_y_filled = _fill_gaps(hip_y)
    if not hip_y_filled:
        return {"bob": [], "lean_deg": [], "limb_swing": []}
    hip_y_arr = np.array(hip_y_filled)
    bob = (hip_y_arr - hip_y_arr.mean()).tolist()
    limb_swing = _fill_gaps(limb_spread)
    if limb_swing:
        arr = np.array(limb_swing)
        rng = arr.max() - arr.min()
        limb_swing = ((arr - arr.mean()) / rng).tolist() if rng > 1e-6 else [0.0] * len(limb_swing)
    return {"bob": bob, "lean_deg": _fill_gaps(lean_deg), "limb_swing": limb_swing}


def resample_motion_signal(signal: Dict[str, List[float]], n_frames: int) -> Dict[str, List[float]]:
    """Resamples each channel (arbitrary source length) to exactly
    ``n_frames`` samples via linear interpolation over normalized [0, 1]
    time -- the "preserve timing/style, adapt to our own duration" step: the
    clip's motion shape carries over regardless of its original fps/length."""
    out: Dict[str, List[float]] = {}
    for key, values in signal.items():
        if not values:
            out[key] = []
            continue
        src_t = np.linspace(0.0, 1.0, len(values))
        dst_t = np.linspace(0.0, 1.0, max(1, n_frames))
        out[key] = np.interp(dst_t, src_t, values).tolist()
    return out


def build_retargeted_motion(video_path: str, n_frames: int, model_path: Optional[Path] = None) -> Optional[Dict[str, List[float]]]:
    """End-to-end: video -> landmarks -> motion signal -> resampled to
    ``n_frames``. Returns ``None`` if no person was detected anywhere in the
    clip -- caller should fall back to the canned per-motion sine-wave style
    rather than animate with an all-zero/meaningless signal."""
    landmarks = extract_pose_landmarks_from_video(video_path, model_path=model_path)
    signal = derive_motion_signal(landmarks)
    if not signal["bob"]:
        return None
    return resample_motion_signal(signal, n_frames)
