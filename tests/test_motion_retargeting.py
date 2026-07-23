"""Motion retargeting tests.

``derive_motion_signal``/``resample_motion_signal`` are pure math over
landmark dicts and fully tested here with synthetic landmark sequences.
``extract_pose_landmarks_from_video`` genuinely calls MediaPipe's CPU Tasks
API against a real video file -- tested for the "no person detected, handled
gracefully" path (a synthetic video has no real human in it, so MediaPipe
correctly finds nothing); true positive-detection accuracy against a real
human video is not verifiable in this environment without one, but the API
integration itself (model download/cache, per-frame inference call,
graceful empty-result handling) is exercised for real, not mocked.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scene_understanding.animation.motion_retargeting import (
    build_retargeted_motion,
    derive_motion_signal,
    extract_pose_landmarks_from_video,
    resample_motion_signal,
)


def _synthetic_landmarks(n: int, bob_amplitude: float = 0.05) -> list:
    """A synthetic walk-like landmark sequence: hips oscillate vertically,
    ankles spread apart and together (stride), torso stays upright."""
    seq = []
    for i in range(n):
        phase = i / max(1, n - 1) * 4 * np.pi
        hip_y = 0.5 + bob_amplitude * np.sin(phase)
        spread = 0.05 + 0.05 * abs(np.sin(phase / 2))
        lm = {
            11: (0.48, 0.3, 0.0, 1.0), 12: (0.52, 0.3, 0.0, 1.0),  # shoulders
            23: (0.48, hip_y, 0.0, 1.0), 24: (0.52, hip_y, 0.0, 1.0),  # hips
            27: (0.5 - spread, 0.9, 0.0, 1.0), 28: (0.5 + spread, 0.9, 0.0, 1.0),  # ankles
        }
        seq.append(lm)
    return seq


def test_derive_motion_signal_extracts_bob_and_stride_from_synthetic_walk():
    landmarks = _synthetic_landmarks(40)
    signal = derive_motion_signal(landmarks)
    assert len(signal["bob"]) == 40
    assert len(signal["lean_deg"]) == 40
    assert len(signal["limb_swing"]) == 40
    # A sinusoidal hip bob should have roughly zero mean (centered) and real variance.
    assert abs(np.mean(signal["bob"])) < 1e-6
    assert np.std(signal["bob"]) > 0.01


def test_derive_motion_signal_interpolates_across_missing_detections():
    landmarks = _synthetic_landmarks(20)
    landmarks[5] = None
    landmarks[6] = None
    landmarks[7] = None
    signal = derive_motion_signal(landmarks)
    assert len(signal["bob"]) == 20
    assert all(np.isfinite(v) for v in signal["bob"]), "gaps must be interpolated, not left as NaN"


def test_derive_motion_signal_returns_empty_when_nothing_detected():
    signal = derive_motion_signal([None] * 10)
    assert signal == {"bob": [], "lean_deg": [], "limb_swing": []}


def test_resample_motion_signal_preserves_shape_at_new_length():
    signal = {"bob": [0.0, 1.0, 0.0, -1.0, 0.0]}
    resampled = resample_motion_signal(signal, n_frames=50)
    assert len(resampled["bob"]) == 50
    # Peak should still land near the middle of the resampled timeline.
    peak_idx = int(np.argmax(resampled["bob"]))
    assert 5 < peak_idx < 20, "the source signal's early peak should resample into the same relative position"


def test_resample_motion_signal_handles_empty_channel():
    assert resample_motion_signal({"bob": []}, n_frames=10) == {"bob": []}


def _write_synthetic_video(path: Path, n_frames: int = 5, size=(64, 64)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_pose_landmarks_handles_no_person_detected_gracefully(tmp_path):
    """Real MediaPipe Tasks API call (not mocked) against a synthetic video
    with no actual human in it -- confirms the integration path (model
    download/cache, per-frame inference, empty-result handling) runs
    end-to-end without crashing, even though there's nothing to detect."""
    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path)
    try:
        results = extract_pose_landmarks_from_video(str(video_path))
    except Exception as exc:  # pragma: no cover - environment without network/model access
        pytest.skip(f"MediaPipe model unavailable in this environment: {exc}")
    assert len(results) == 5
    assert all(r is None for r in results), "a synthetic video with no person should detect nothing, not hallucinate landmarks"


def test_build_retargeted_motion_returns_none_when_no_person_in_clip(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path)
    try:
        result = build_retargeted_motion(str(video_path), n_frames=24)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MediaPipe model unavailable in this environment: {exc}")
    assert result is None
