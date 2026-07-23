"""Unit tests for terrain-aware motion classification (Phase 2.5 item 2):
``ground_plane.classify_terrain_openvocab`` and
``paths_export._apply_terrain_aware_motion``. CLIP itself is monkeypatched
out (via ``zero_shot_classify_text``) so these run fast and don't need a
real model load."""
from __future__ import annotations

import scene_understanding.pathing.ground_plane as ground_plane
import scene_understanding.stages.paths_export as paths_export


def _patch_zero_shot(monkeypatch, water_result, open_air_result):
    """``classify_terrain_openvocab`` makes up to two zero_shot_classify_text
    calls in order: water-vs-solid, then open_air-vs-grounded. Feed each in
    turn regardless of the actual prompt tuples passed."""
    calls = iter([water_result, open_air_result])

    def _fake(text, positive_prompts, negative_prompts, device="cpu", **kwargs):
        return next(calls)

    monkeypatch.setattr(ground_plane, "zero_shot_classify_text", _fake)


def test_classify_terrain_water_short_circuits_to_swimmable(monkeypatch):
    _patch_zero_shot(monkeypatch, water_result=True, open_air_result=None)
    assert ground_plane.classify_terrain_openvocab("a swimming pool") == "swimmable"


def test_classify_terrain_open_air(monkeypatch):
    _patch_zero_shot(monkeypatch, water_result=False, open_air_result=True)
    assert ground_plane.classify_terrain_openvocab("open sky above a cliff") == "open_air"


def test_classify_terrain_walkable(monkeypatch):
    _patch_zero_shot(monkeypatch, water_result=False, open_air_result=False)
    assert ground_plane.classify_terrain_openvocab("a paved sidewalk") == "walkable"


def test_classify_terrain_ambiguous_returns_none(monkeypatch):
    _patch_zero_shot(monkeypatch, water_result=None, open_air_result=None)
    assert ground_plane.classify_terrain_openvocab("an indistinct texture") is None


def _support_trace(n: int, region_type: str):
    return [{"region_type": region_type} for _ in range(n)]


def test_apply_terrain_aware_motion_overrides_walk_to_swim(monkeypatch):
    monkeypatch.setattr(paths_export, "classify_terrain_openvocab", lambda text, device="cpu": "swimmable")
    kinematic_sigs = [{"start_idx": 0, "end_idx": 3, "motion": "walk", "dz_m": 0.0}]
    support_trace = _support_trace(4, "a pool")
    out = paths_export._apply_terrain_aware_motion(kinematic_sigs, support_trace)
    assert out[0]["motion"] == "swim"
    assert out[0]["terrain_type"] == "swimmable"


def test_apply_terrain_aware_motion_overrides_crawl_to_fly(monkeypatch):
    monkeypatch.setattr(paths_export, "classify_terrain_openvocab", lambda text, device="cpu": "open_air")
    kinematic_sigs = [{"start_idx": 0, "end_idx": 2, "motion": "crawl", "dz_m": 0.0}]
    support_trace = _support_trace(3, "open sky")
    out = paths_export._apply_terrain_aware_motion(kinematic_sigs, support_trace)
    assert out[0]["motion"] == "fly"
    assert out[0]["terrain_type"] == "open_air"


def test_apply_terrain_aware_motion_swimmable_never_overrides_jump_climb_descend(monkeypatch):
    monkeypatch.setattr(paths_export, "classify_terrain_openvocab", lambda text, device="cpu": "swimmable")
    kinematic_sigs = [
        {"start_idx": 0, "end_idx": 1, "motion": "jump", "dz_m": 0.4},
        {"start_idx": 1, "end_idx": 2, "motion": "climb", "dz_m": 0.2},
        {"start_idx": 2, "end_idx": 3, "motion": "descend", "dz_m": -0.3},
    ]
    support_trace = _support_trace(4, "a pool")
    out = paths_export._apply_terrain_aware_motion(kinematic_sigs, support_trace)
    assert [seg["motion"] for seg in out] == ["jump", "climb", "descend"]
    # terrain_type is still recorded for downstream consumers even when not applied.
    assert all(seg["terrain_type"] == "swimmable" for seg in out)


def test_apply_terrain_aware_motion_open_air_overrides_jump_climb_descend_too(monkeypatch):
    # Real-run bug: a flying path's "descend" segment (losing altitude while
    # flying) never got promoted to "fly" because the original override only
    # touched walk/crawl -- there's no ground over open air for jump/climb/
    # descend to mean anything, so open_air must override all of them.
    monkeypatch.setattr(paths_export, "classify_terrain_openvocab", lambda text, device="cpu": "open_air")
    kinematic_sigs = [
        {"start_idx": 0, "end_idx": 1, "motion": "jump", "dz_m": 0.4},
        {"start_idx": 1, "end_idx": 2, "motion": "climb", "dz_m": 0.2},
        {"start_idx": 2, "end_idx": 3, "motion": "descend", "dz_m": -0.3},
    ]
    support_trace = _support_trace(4, "open sky")
    out = paths_export._apply_terrain_aware_motion(kinematic_sigs, support_trace)
    assert [seg["motion"] for seg in out] == ["fly", "fly", "fly"]
    assert all(seg["terrain_type"] == "open_air" for seg in out)


def test_apply_terrain_aware_motion_walkable_leaves_walk_unchanged(monkeypatch):
    monkeypatch.setattr(paths_export, "classify_terrain_openvocab", lambda text, device="cpu": "walkable")
    kinematic_sigs = [{"start_idx": 0, "end_idx": 2, "motion": "walk", "dz_m": 0.0}]
    support_trace = _support_trace(3, "a sidewalk")
    out = paths_export._apply_terrain_aware_motion(kinematic_sigs, support_trace)
    assert out[0]["motion"] == "walk"
    assert out[0]["terrain_type"] == "walkable"


def test_apply_terrain_aware_motion_clip_unavailable_degrades_safely(monkeypatch):
    monkeypatch.setattr(paths_export, "classify_terrain_openvocab", lambda text, device="cpu": None)
    kinematic_sigs = [{"start_idx": 0, "end_idx": 2, "motion": "walk", "dz_m": 0.0}]
    support_trace = _support_trace(3, "an indistinct surface")
    out = paths_export._apply_terrain_aware_motion(kinematic_sigs, support_trace)
    assert out[0]["motion"] == "walk"
    assert out[0]["terrain_type"] == "unknown"


def test_speed_map_has_swim_and_fly_entries():
    from scene_understanding.animation.compositor import _FIXED_DURATION_MOTIONS, _SPEED_MAP

    assert "swim" in _SPEED_MAP and "fly" in _SPEED_MAP
    # swim/fly are continuous motions, not fixed-duration point events.
    assert "swim" not in _FIXED_DURATION_MOTIONS and "fly" not in _FIXED_DURATION_MOTIONS


def test_blender_motion_style_has_swim_and_fly_entries():
    # blender_render_actor.py has an unconditional `import bpy` (only importable
    # inside Blender's bundled Python -- the existing Tier 1 test suite deliberately
    # never imports this module, see test_tier1_render_job.py's own docstring), so
    # this checks the source text directly rather than importing the module.
    import pathlib

    src = pathlib.Path("scene_understanding/animation/tier1_blender/blender_render_actor.py").read_text()
    style_block = src[src.index("_MOTION_STYLE = {") : src.index("_MOTION_STYLE = {") + 800]
    assert '"swim"' in style_block
    assert '"fly"' in style_block
