"""Tests for package pipeline stage skeleton wiring."""

from pathlib import Path
import json
import cv2
import numpy as np

from scene_understanding.pipeline import LegacySceneUnderstandingPipeline, SceneUnderstandingPipeline


def _write_dummy_image(path: Path) -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_stage_preprocess_builds_context_and_dirs(tmp_path):
    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    image_path = tmp_path / "image01.jpg"
    _write_dummy_image(image_path)
    out_dir = tmp_path / "out"

    ctx = pipe._stage_preprocess(str(image_path), str(out_dir))

    assert ctx.image_path == image_path
    assert ctx.output_dir == out_dir
    assert ctx.stem == "image01"
    assert (out_dir / "depth").exists()
    assert (out_dir / "scene_graph").exists()


def test_process_image_uses_legacy_when_env_legacy(monkeypatch, tmp_path):
    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    image_path = tmp_path / "img.png"
    _write_dummy_image(image_path)
    output_dir = tmp_path / "run"
    calls = []
    monkeypatch.setenv("CITV_SCENE_PIPELINE_MODE", "legacy")

    def _legacy_process(_self, legacy_image_path, legacy_output_dir):
        calls.append("legacy_export")
        return {"image": legacy_image_path, "output": legacy_output_dir}

    monkeypatch.setattr(LegacySceneUnderstandingPipeline, "process_image", _legacy_process)

    result = pipe.process_image(str(image_path), str(output_dir))

    assert calls == ["legacy_export"]
    assert result == {"image": str(image_path), "output": str(output_dir)}


def test_process_image_defaults_to_legacy_when_no_config(monkeypatch, tmp_path):
    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    pipe.config = None
    image_path = tmp_path / "img_default.png"
    _write_dummy_image(image_path)
    output_dir = tmp_path / "run_default"
    calls = []

    def _staged(image_arg, output_arg):
        calls.append("staged")
        return {"scene_json": f"{output_arg}/scene_graph/staged/{Path(image_arg).stem}_scene.json"}

    def _legacy(_self, *_args, **_kwargs):
        calls.append("legacy")
        return {"legacy": True}

    monkeypatch.setattr(pipe, "process_image_staged", _staged)
    monkeypatch.setattr(LegacySceneUnderstandingPipeline, "process_image", _legacy)

    out = pipe.process_image(str(image_path), str(output_dir))
    assert calls == ["legacy"]
    assert out == {"legacy": True}


def test_process_image_uses_staged_when_config_staged(monkeypatch, tmp_path):
    class _Cfg:
        scene_pipeline_mode = "staged"

    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    pipe.config = _Cfg()
    image_path = tmp_path / "img_staged_cfg.png"
    _write_dummy_image(image_path)
    output_dir = tmp_path / "run_staged_cfg"
    calls = []

    def _staged(image_arg, output_arg):
        calls.append("staged")
        return {"scene_json": f"{output_arg}/scene_graph/staged/{Path(image_arg).stem}_scene.json"}

    def _legacy(_self, *_args, **_kwargs):
        calls.append("legacy")
        return {"legacy": True}

    monkeypatch.setattr(pipe, "process_image_staged", _staged)
    monkeypatch.setattr(LegacySceneUnderstandingPipeline, "process_image", _legacy)

    out = pipe.process_image(str(image_path), str(output_dir))
    assert calls == ["staged"]
    assert out["scene_json"].endswith("img_staged_cfg_scene.json")


def test_process_image_uses_legacy_when_mode_legacy(monkeypatch, tmp_path):
    class _Cfg:
        scene_pipeline_mode = "legacy"

    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    pipe.config = _Cfg()
    image_path = tmp_path / "img_legacy.png"
    _write_dummy_image(image_path)
    output_dir = tmp_path / "run_legacy"
    calls = []

    def _staged(*_args, **_kwargs):
        calls.append("staged")
        return {"scene_json": "unused"}

    def _legacy(_self, legacy_image_path, legacy_output_dir):
        calls.append("legacy")
        return {"image": legacy_image_path, "output": legacy_output_dir}

    monkeypatch.setattr(pipe, "process_image_staged", _staged)
    monkeypatch.setattr(LegacySceneUnderstandingPipeline, "process_image", _legacy)

    out = pipe.process_image(str(image_path), str(output_dir))
    assert calls == ["legacy"]
    assert out == {"image": str(image_path), "output": str(output_dir)}


def test_process_image_env_overrides_config(monkeypatch, tmp_path):
    class _Cfg:
        scene_pipeline_mode = "legacy"

    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    pipe.config = _Cfg()
    image_path = tmp_path / "img_env.png"
    _write_dummy_image(image_path)
    output_dir = tmp_path / "run_env"
    calls = []

    monkeypatch.setenv("CITV_SCENE_PIPELINE_MODE", "staged")

    def _staged(*_args, **_kwargs):
        calls.append("staged")
        return {"scene_json": "ok"}

    def _legacy(_self, *_args, **_kwargs):
        calls.append("legacy")
        return {"legacy": True}

    monkeypatch.setattr(pipe, "process_image_staged", _staged)
    monkeypatch.setattr(LegacySceneUnderstandingPipeline, "process_image", _legacy)

    out = pipe.process_image(str(image_path), str(output_dir))
    assert calls == ["staged"]
    assert out == {"scene_json": "ok"}


def test_run_stage_chain_executes_stages_in_order(monkeypatch, tmp_path):
    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    original_preprocess = SceneUnderstandingPipeline._stage_preprocess
    image_path = tmp_path / "img2.png"
    _write_dummy_image(image_path)
    output_dir = tmp_path / "run2"
    calls = []

    def _preprocess(image_arg, output_arg):
        calls.append("preprocess")
        return original_preprocess(pipe, image_arg, output_arg)

    def _record(name):
        def _fn(ctx):
            calls.append(name)
            return ctx

        return _fn

    monkeypatch.setattr(pipe, "_stage_preprocess", _preprocess)
    monkeypatch.setattr(pipe, "_stage_depth", _record("depth"))
    monkeypatch.setattr(pipe, "_stage_regions", _record("regions"))
    monkeypatch.setattr(pipe, "_stage_segment", _record("segment"))
    monkeypatch.setattr(pipe, "_stage_label_and_geometry", _record("label_geometry"))
    monkeypatch.setattr(pipe, "_stage_relations", _record("relations"))
    monkeypatch.setattr(pipe, "_stage_unload_labellers", _record("unload_labellers"))

    ctx = pipe._run_stage_chain(str(image_path), str(output_dir))

    assert calls == ["preprocess", "depth", "regions", "segment", "label_geometry", "relations", "unload_labellers"]
    assert ctx.stem == "img2"


def test_process_image_staged_writes_scene_json(monkeypatch, tmp_path):
    monkeypatch.setenv("CITV_STAGED_MODULAR_CHAIN_ONLY", "1")
    pipe = SceneUnderstandingPipeline.__new__(SceneUnderstandingPipeline)
    image_path = tmp_path / "img3.png"
    _write_dummy_image(image_path)
    output_dir = tmp_path / "run3"

    def _fake_chain(_image_path, _output_dir):
        ctx = pipe._stage_preprocess(_image_path, _output_dir)
        ctx.detections = [{"id": "det_0"}]
        ctx.extra["objects"] = [
            {
                "id": "obj_0",
                "label": "object",
                "conf": 0.75,
                "bbox": [1, 2, 3, 4],
                "coordinates_3d": {"x": 0.1, "y": 0.2, "z": 1.5},
                "depth_stats": {"mean": 1.5},
                "mask_centroid_2d": [2, 3],
            }
        ]
        ctx.relations = [{"sub": "obj_0", "pred": "near", "obj": "obj_0"}]
        return ctx

    monkeypatch.setattr(pipe, "_run_stage_chain", _fake_chain)

    result = pipe.process_image_staged(str(image_path), str(output_dir))
    scene_path = Path(result["scene_json"])
    assert scene_path.exists()

    payload = json.loads(scene_path.read_text())
    assert payload["metadata"]["segmentor"] == "package_staged_pipeline"
    assert payload["metadata"]["relation_debug"]["num_detections"] == 1
    assert "models" in payload["metadata"]
    assert "relation_sources" in payload["metadata"]
    assert len(payload["objects"]) == 1
    assert payload["objects"][0]["confidence"] == 0.75
    assert payload["objects"][0]["sources"]["Pix2SG"]["relations"] == []
    assert len(payload["relations"]) == 1
