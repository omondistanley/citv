"""Tests for staged full-run orchestration (legacy-equivalent outputs)."""

from pathlib import Path

import pytest

from scene_understanding.pipeline import LegacySceneUnderstandingPipeline, SceneUnderstandingPipeline
from scene_understanding.stages import full_run


def test_use_modular_chain_only_false_by_default(monkeypatch):
    monkeypatch.delenv("CITV_STAGED_MODULAR_CHAIN_ONLY", raising=False)
    assert full_run.use_modular_chain_only() is False


@pytest.mark.parametrize("val", ("1", "true", "yes", "on", "TRUE"))
def test_use_modular_chain_only_true(monkeypatch, val):
    monkeypatch.setenv("CITV_STAGED_MODULAR_CHAIN_ONLY", val)
    assert full_run.use_modular_chain_only() is True


def test_preferred_scene_json_paths_prefers_grounded_sam2(tmp_path):
    stem = "img1"
    sg = tmp_path / "scene_graph" / "grounded_sam2"
    sg.mkdir(parents=True)
    p = sg / f"{stem}_scene.json"
    p.write_text("{}")
    paths = full_run._preferred_scene_json_paths(tmp_path, stem)
    assert paths and paths[0] == p


def test_process_image_staged_invokes_legacy_when_not_modular_only(monkeypatch, tmp_path):
    monkeypatch.delenv("CITV_STAGED_MODULAR_CHAIN_ONLY", raising=False)
    calls = []

    def _legacy(_self, ip, od):
        calls.append((ip, od))

    monkeypatch.setattr(LegacySceneUnderstandingPipeline, "process_image", _legacy)
    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "o"
    pipe.process_image_staged(str(img), str(out))
    assert len(calls) == 1
    assert calls[0][0] == str(img)
    assert calls[0][1] == str(out)
