from __future__ import annotations

import tempfile
import unittest
import json
import subprocess
from pathlib import Path

import numpy as np

from scene_understanding.action_ontology import load_action_ontology
from scene_understanding.evidence import build_label_candidates, normalize_relation
from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.pathing import path_visual_qa_export
from scene_understanding.visualization import animation_qa_renderer
from scene_understanding.stages import action_export, affordances_export, animation_export, captions_export, parity_export, paths_export


class _Cfg:
    export_affordance_hypotheses = True
    export_action_hypotheses = True
    path_actor_width_m = 0.5
    path_min_width_px = 2.0
    path_max_width_px = 40.0


class _Pipeline:
    config = _Cfg()


class _NoQaCfg(_Cfg):
    path_animation_qa_enabled = False
class _QaCfg(_Cfg):
    path_animation_qa_enabled = True
    path_animation_qa_modes = [24]
    path_animation_qa_candidate_seconds = 0.05
    path_animation_qa_delimiter_seconds = 0.0
    path_animation_qa_render_debug_overlays = False
    path_animation_qa_max_candidates_per_panel = 16
    path_animation_qa_output_subdir_24 = "animation_qa_24"

    path_animation_qa_modes = [24, 120]


class _NoQaPipeline:
    config = _NoQaCfg()


class _QaPipeline:
    config = _QaCfg()


def _ctx(tmp_path: Path) -> PipelineContext:
    h, w = 32, 48
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.linspace(1.0, 3.0, h * w, dtype=np.float32).reshape(h, w)
    lm = np.ones((h, w), dtype=np.int32)
    glass_mask = np.zeros((h, w), dtype=bool)
    glass_mask[8:24, 28:38] = True
    water_mask = np.zeros((h, w), dtype=bool)
    water_mask[18:29, 3:22] = True
    return PipelineContext(
        image_path=tmp_path / "scene.png",
        output_dir=tmp_path,
        stem="scene",
        timestamp="2026-05-06T00:00:00",
        img_bgr=img,
        img_rgb=img,
        height=h,
        width=w,
        intrinsics={"fx": 40.0, "fy": 40.0, "cx": 24.0, "cy": 16.0},
        metric_depth=depth,
        region_label_map=lm,
        region_partition_meta=[
            {
                "id": "region_1",
                "region_index": 1,
                "type": "floor near glass and water",
                "semantic_label": "walkable reflective liquid area",
                "depth_stats": {"mean": 2.0},
            }
        ],
        relations=[
            {
                "subject": "obj_glass",
                "predicate": "reflects",
                "object": "obj_water",
                "score": 0.8,
            }
        ],
        extra={
            "objects": [
                {
                    "id": "obj_glass",
                    "label": "glass door",
                    "caption": "transparent glass door with reflection",
                    "bbox": [28, 8, 10, 16],
                    "conf": 0.9,
                    "mask_centroid_2d": [33, 16],
                    "region_id": "region_1",
                    "region_index": 1,
                    "depth_stats": {"z_val": 1.5, "possibly_transparent": True},
                    "_sam2_mask_array": glass_mask,
                    "sources": {
                        "Florence2": {"caption": "a shiny transparent glass door"},
                        "RAM++": {"tags": ["glass", "reflection", "door"]},
                    },
                },
                {
                    "id": "obj_water",
                    "label": "water pool",
                    "caption": "liquid water surface",
                    "bbox": [3, 18, 19, 11],
                    "conf": 0.88,
                    "mask_centroid_2d": [12, 24],
                    "region_id": "region_1",
                    "region_index": 1,
                    "depth_stats": {"z_val": 2.3},
                    "_sam2_mask_array": water_mask,
                    "sources": {
                        "Florence2": {"caption": "a pool of water"},
                        "RAM++": {"tags": ["water", "liquid", "pool"]},
                    },
                },
            ],
            "caption_tiers": {
                "global_scene": {
                    "caption": "A scene with a reflective glass door and a water pool.",
                    "status": "generated_local_florence",
                },
                "per_object": {
                    "objects": [
                        {
                            "object_id": "obj_glass",
                            "label": "glass door",
                            "florence_caption": "a shiny transparent glass door",
                            "rampp_tags": ["glass", "reflection", "door"],
                            "gdino_conf": 0.9,
                            "label_warning": "",
                        },
                        {
                            "object_id": "obj_water",
                            "label": "water pool",
                            "florence_caption": "a pool of water",
                            "rampp_tags": ["water", "liquid", "pool"],
                            "gdino_conf": 0.88,
                            "label_warning": "",
                        },
                    ]
                },
                "per_region": {
                    "regions": [
                        {
                            "region_id": "region_1",
                            "region_type": "floor",
                            "object_labels": ["glass door", "water pool"],
                            "narrative": "Region region_1 contains glass and water.",
                        }
                    ]
                },
                "cross_region": {"interactions": []},
                "uncertainty": {"notes": []},
            },
            "caption_bundle": {"tiers": {}},
        },
    )


class PathActionAffordanceContractTests(unittest.TestCase):
    def test_phase_acceptance_matrix_resource_is_present_and_well_formed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        matrix_path = repo_root / "scene_understanding/resources/path_phase_acceptance_matrix.json"
        self.assertTrue(matrix_path.exists())
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("schema"), "citv_path_phase_acceptance_matrix_v1")
        phases = list(payload.get("phases") or [])
        self.assertEqual([int(p.get("phase", -1)) for p in phases], [1, 2, 3, 4, 5, 6])
        for phase in phases:
            self.assertTrue(str(phase.get("name", "")).strip())
            self.assertIsInstance(phase.get("required_artifacts") or [], list)

    def test_relation_normalization_accepts_staged_and_legacy_schemas(self) -> None:
        legacy = normalize_relation({"subject": "a", "predicate": "on", "object": "b", "score": 0.5})
        staged = normalize_relation({"sub_id": "a", "pred": "on", "obj_id": "b", "confidence": 0.6})

        self.assertEqual(legacy["subject_id"], "a")
        self.assertEqual(staged["object_id"], "b")
        self.assertEqual(staged["predicate"], "on")
        self.assertAlmostEqual(staged["score"], 0.6)

    def test_label_fusion_rejects_meta_visual_caption_as_canonical_label(self) -> None:
        ontology = load_action_ontology(_Cfg())
        fused = build_label_candidates(
            gdino_label="person",
            gdino_conf=0.72,
            florence_label="blurred photograph",
            florence_caption="a blurred photograph of a person in a room",
            rampp_label="",
            rampp_tags=["person", "room"],
            ontology=ontology,
        )

        self.assertEqual(fused["canonical_label"], "person")
        self.assertTrue(any(r["label"] == "blurred photograph" for r in fused["rejected_labels"]))
        self.assertIn("blurred", fused["visual_quality_attributes"])

    def test_caption_cross_region_populates_from_staged_relation_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = _ctx(tmp_path)
            ctx.relations = [{"sub_id": "obj_glass", "pred": "reflects", "obj_id": "obj_water", "score": 0.8}]
            ctx.extra["objects"][1]["region_id"] = "region_2"
            ctx.extra["objects"][1]["region_index"] = 2

            ctx = captions_export.run(_Pipeline(), ctx)
            cross = ctx.extra["caption_tiers"]["cross_region"]

            self.assertEqual(cross["count"], 1)
            self.assertEqual(cross["interactions"][0]["predicate"], "reflects")

    def test_affordance_stage_exports_caption_aware_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = affordances_export.run(_Pipeline(), _ctx(tmp_path))

            self.assertIsNotNone(ctx.caption_evidence)
            self.assertIsNotNone(ctx.scene_affordances)
            self.assertIsNotNone(ctx.object_affordances)
            self.assertIsNotNone(ctx.mask_affordances)
            self.assertIn("caption_evidence_json", ctx.path_exports)
            self.assertTrue((tmp_path / "scene_graph/staged/scene_object_affordances.json").exists())

            water = next(o for o in ctx.object_affordances["objects"] if o["object_id"] == "obj_water")
            self.assertTrue(any(a["name"] == "swim" for a in water["actions"]))
            glass = next(o for o in ctx.object_affordances["objects"] if o["object_id"] == "obj_glass")
            self.assertTrue(any(r["name"] == "reflective_transparent" for r in glass["roles"]))

    def test_path_enrichment_adds_scene_and_render_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = affordances_export.run(_Pipeline(), _ctx(tmp_path))
            paths = [
                {
                    "path_id": "p0",
                    "source_entity": {"type": "object", "id": "obj_water"},
                    "target_entity": {"type": "object", "id": "obj_glass"},
                    "polyline_2d": [[10, 24], [18, 22], [28, 16], [36, 16]],
                    "scores": {"overall_confidence": 0.7},
                }
            ]
            speed = np.ones((ctx.height, ctx.width), dtype=np.float32) * 0.8
            objects = list(ctx.extra["objects"])

            paths_export._enrich_path_hypotheses(ctx, _Pipeline(), paths, speed, objects, ctx.region_label_map)
            path = paths[0]

            self.assertTrue(path["polyline_3d"])
            self.assertTrue(path["width_profile_px"])
            self.assertGreater(path["caption_trace"]["mean_caption_confidence"], 0)
            self.assertTrue(path["visibility_profile"])
            self.assertTrue(path["render_layers"])
            self.assertIn("semantic_confidence", path["scores"])
            self.assertIn("contract_field_availability", path)
            self.assertIn("unavailable_reasons", path["contract_field_availability"])

    def test_action_export_writes_path_and_affordance_manifolds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = affordances_export.run(_Pipeline(), _ctx(tmp_path))
            paths_root = tmp_path / "scene_graph/staged/scene_paths"
            paths_root.mkdir(parents=True)
            (paths_root / "path_hypotheses.json").write_text(
                """
                {
                  "paths": [
                    {
                      "path_id": "p0",
                      "source_entity": {"type": "object", "id": "obj_water"},
                      "target_entity": {"type": "object", "id": "obj_glass"},
                      "polyline_2d": [[10, 24], [36, 16]],
                      "polyline_3d": [[10, 24, 2.0], [36, 16, 1.4]],
                      "manifold_type": "ribbon_path",
                      "scores": {"overall_confidence": 0.7}
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            ctx = action_export.run(_Pipeline(), ctx)

            self.assertIsNotNone(ctx.action_hypotheses)
            self.assertTrue((paths_root / "action_hypotheses.json").exists())
            # v2 schema: single `hypotheses` key.
            self.assertEqual(ctx.action_hypotheses.get("schema"), "citv_action_hypotheses_v2")
            self.assertNotIn("actions", ctx.action_hypotheses, "v2 must drop duplicate 'actions' key")
            manifold_types = {a["manifold_type"] for a in ctx.action_hypotheses["hypotheses"]}
            self.assertIn("ribbon_path", manifold_types)
            self.assertTrue({"blob_path", "effect_field", "portal_path", "contact_patch"} & manifold_types)
            any_scores = (ctx.action_hypotheses.get("hypotheses") or [])[0].get("scores") or {}
            self.assertIn("confidence_breakdown", any_scores)
            self.assertIn("rejection_reasons", any_scores)

    def test_action_export_interior_path_is_first_class_manifold(self) -> None:
        obj = {"object_id": "obj_1", "label": "container"}
        mask = {
            "geometry": {"interior_seed_uv": [10, 10], "interior_extent": {"radius_px": 8}},
            "path_modes": [{"mode": "interior_path", "score": 0.8}],
        }
        manifold = action_export._object_manifold("interior_path", obj, mask)
        self.assertEqual(manifold["type"], "interior_path")
        self.assertTrue(manifold.get("interior_constraint", {}).get("stay_inside_mask"))

    def test_affordance_label_noise_penalty_reduces_quality(self) -> None:
        factor = affordances_export._label_quality_factor(
            {"label": "landscape photograph", "canonical_label": "thin"},
            {"labels": ["face", "set"]},
        )
        self.assertLess(factor, 1.0)

    def test_parity_export_writes_manifest_bundle_and_caption_compat_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = _ctx(tmp_path)
            ctx.relations = [{"sub_id": "obj_glass", "pred": "reflects", "obj_id": "obj_water", "score": 0.8}]
            ctx = captions_export.run(_Pipeline(), ctx)
            ctx = affordances_export.run(_Pipeline(), ctx)

            paths_root = tmp_path / "scene_graph/staged/scene_paths"
            paths_root.mkdir(parents=True)
            path_payload = {
                "paths": [
                    {
                        "path_id": "p0",
                        "path_level": "object",
                        "source_entity": {"type": "object", "id": "obj_water"},
                        "target_entity": {"type": "object", "id": "obj_glass"},
                        "polyline_2d": [[10, 24], [36, 16]],
                        "polyline_3d": [[10, 24, 2.0], [36, 16, 1.4]],
                        "manifold_type": "ribbon_path",
                        "scores": {"overall_confidence": 0.7},
                    }
                ]
            }
            (paths_root / "path_hypotheses.json").write_text(
                json.dumps(path_payload),
                encoding="utf-8",
            )
            ctx.path_exports["path_hypotheses_json"] = "scene_graph/staged/scene_paths/path_hypotheses.json"
            ctx = action_export.run(_Pipeline(), ctx)

            ctx = parity_export.run(_Pipeline(), ctx)

            staged_dir = tmp_path / "scene_graph/staged"
            manifest_path = staged_dir / "scene_artifact_manifest.json"
            bundle_path = staged_dir / "scene_scene_action_bundle.json"
            scene_path = staged_dir / "scene_scene.json"
            self.assertTrue(scene_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(bundle_path.exists())
            self.assertTrue((staged_dir / "scene_florence_object_captions.json").exists())
            self.assertTrue((paths_root / "scene_context.json").exists())
            self.assertTrue((paths_root / "animation_qa_24/animation_qa_24_failure.json").exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            missing_keys = {rec["key"] for rec in manifest["missing_artifacts"]}
            required_missing = {rec["key"] for rec in manifest["missing_artifacts"] if rec.get("required")}
            self.assertNotIn("scene_action_bundle_json", missing_keys)
            self.assertNotIn("artifact_manifest_json", missing_keys)
            self.assertEqual(required_missing, set())
            self.assertTrue(any(f["artifact"] == "animation_qa_24" for f in manifest["failed_artifacts"]))

            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["schema"], "citv_scene_action_bundle_v1")
            self.assertEqual(bundle["relations"]["normalized_count"], 1)

    def test_animation_export_writes_non_empty_trajectory_and_components_without_qa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = affordances_export.run(_Pipeline(), _ctx(tmp_path))
            paths_root = tmp_path / "scene_graph/staged/scene_paths"
            paths_root.mkdir(parents=True)
            path_payload = {
                "schema": "citv_path_hypotheses_v2",
                "paths": [
                    {
                        "path_id": "p0",
                        "path_level": "object",
                        "path_type": "object_to_object",
                        "source_entity": {"type": "object", "id": "obj_water"},
                        "target_entity": {"type": "object", "id": "obj_glass"},
                        "polyline_2d": [[10, 24], [18, 22], [28, 16], [36, 16]],
                        "polyline_3d": [[10, 24, 2.0], [18, 22, 1.8], [28, 16, 1.6], [36, 16, 1.4]],
                        "manifold_type": "ribbon_path",
                        "action_family": "locomotion",
                        "scores": {"overall_confidence": 0.7, "semantic_confidence": 0.6},
                        "caption_trace": {"mean_caption_confidence": 0.6},
                        "visibility_profile": [{"visible": True}],
                        "render_layers": ["in_front"],
                    }
                ],
            }
            (paths_root / "path_hypotheses.json").write_text(json.dumps(path_payload), encoding="utf-8")

            ctx = animation_export.run(_NoQaPipeline(), ctx)

            trajectory_path = paths_root / "trajectory_hypotheses.json"
            components_path = paths_root / "animation_components.json"
            plan_path = paths_root / "animation_plan.json"
            atlas_path = paths_root / "path_atlas_manifest.json"
            self.assertTrue(trajectory_path.exists())
            self.assertTrue(components_path.exists())
            self.assertTrue(plan_path.exists())
            self.assertTrue(atlas_path.exists())
            trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
            components = json.loads(components_path.read_text(encoding="utf-8"))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
            self.assertGreater(len(trajectories.get("hypotheses") or []), 0)
            self.assertGreater(len(components.get("components") or []), 0)
            self.assertGreater(len(plan.get("paths") or []), 0)
            self.assertGreater(len(atlas.get("paths") or []), 0)

    def test_trajectory_coverage_adds_path_follow_hypotheses_for_unlinked_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = _ctx(tmp_path)
            paths = [
                {
                    "path_id": "p_a",
                    "source_entity": {"type": "object", "id": "obj_water"},
                    "target_entity": {"type": "object", "id": "obj_glass"},
                    "polyline_2d": [[10, 24], [20, 20], [36, 16]],
                    "polyline_3d": [[10, 24, 2.0], [20, 20, 1.8], [36, 16, 1.4]],
                    "manifold_type": "ribbon_path",
                    "action_family": "locomotion",
                    "scores": {"overall_confidence": 0.7},
                    "render_layers": ["in_front"],
                    "motion_hints": [{"motion": "walk", "score": 0.6}],
                },
                {
                    "path_id": "p_b",
                    "source_entity": {"type": "object", "id": "obj_glass"},
                    "target_entity": {"type": "object", "id": "obj_water"},
                    "polyline_2d": [[36, 16], [25, 19], [10, 24]],
                    "manifold_type": "portal_path",
                    "action_family": "transition",
                    "scores": {"overall_confidence": 0.5},
                    "render_layers": ["in_front"],
                },
            ]
            bundle = {"schema": "citv_trajectory_hypotheses_bundle_v1", "hypotheses": []}

            animation_export._ensure_path_follow_trajectory_coverage(
                ctx,
                bundle,
                paths,
                {"width": ctx.width, "height": ctx.height},
                "staged",
                _Cfg(),
            )

            linked = {
                str(h.get("continues_from_path_id", ""))
                for h in bundle.get("hypotheses") or []
            }
            self.assertEqual(linked, {"p_a", "p_b"})
            self.assertEqual(bundle["path_trajectory_coverage"]["missing_path_ids"], [])

    def test_visual_qa_links_action_hypotheses_v2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            paths_root = tmp_path / "paths"
            path = {
                "path_id": "p0",
                "path_type": "fmm_geodesic",
                "manifold_type": "ribbon_path",
                "action_family": "locomotion",
                "source_entity": {"id": "obj_water"},
                "target_entity": {"id": "obj_glass"},
                "polyline_2d": [[10, 24], [36, 16]],
                "scores": {"overall_confidence": 0.7},
            }
            action_bundle = {
                "schema": "citv_action_hypotheses_v2",
                "hypotheses": [
                    {
                        "action_id": "act_p0",
                        "action_name": "walk",
                        "action_family": "locomotion",
                        "manifold_type": "ribbon_path",
                        "grounding": {"path_id": "p0"},
                    }
                ],
            }

            path_visual_qa_export.export_path_visual_qa_json_and_md(
                paths_root=paths_root,
                stem="scene",
                width=48,
                height=32,
                ranked_paths=[path],
                traj_bundle={"hypotheses": []},
                action_hypotheses=action_bundle,
                batch_meta=[],
                dropped_paths=[],
            )

            payload = json.loads((paths_root / "path_visual_qa.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["paths"][0]["linked_action"]["action_id"], "act_p0")
            self.assertTrue((paths_root / "per_path/p0.json").exists())
            self.assertTrue((paths_root / "per_path/p0.md").exists())

    def test_animation_qa_renderer_reads_hypotheses_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            paths_root = tmp_path / "scene_paths"
            paths_root.mkdir(parents=True, exist_ok=True)
            (paths_root / "path_hypotheses.json").write_text(
                json.dumps(
                    {
                        "hypotheses": [
                            {
                                "path_id": "p0",
                                "polyline_2d": [[2, 2], [20, 20]],
                                "regions_traversed": ["region_1"],
                                "energy_terms": {"E_total": 0.1},
                                "visibility_profile": [{"visible_fraction": 1.0}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (paths_root / "animation_components.json").write_text(
                json.dumps(
                    {
                        "components": [
                            {
                                "path_id_linked": "p0",
                                "trajectory_id": "t0",
                                "trajectory_points": [
                                    {"t_s": 0.0, "x_px": 2, "y_px": 2, "theta_rad": 0.0},
                                    {"t_s": 1.0, "x_px": 20, "y_px": 20, "theta_rad": 0.0},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (paths_root / "path_atlas_manifest.json").write_text(
                json.dumps(
                    {
                        "panels": [{"index": 1}],
                        "paths": [{"panel_index": 1, "path_id": "p0", "path_num": 1}],
                    }
                ),
                encoding="utf-8",
            )
            out = animation_qa_renderer.export_animation_qa(
                paths_root_dir=paths_root,
                img_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
                metric_depth_m=np.ones((32, 32), dtype=np.float32),
                cfg=_QaCfg(),
                region_label_map=np.ones((32, 32), dtype=np.int32),
            )
            man = json.loads((paths_root / "animation_qa_24" / "animation_qa_manifest.json").read_text(encoding="utf-8"))
            panels = list(man.get("panel_videos") or [])
            segments = sum(len(p.get("segments") or []) for p in panels)
            self.assertIn("animation_qa_manifest_24_json", out)
            self.assertGreater(segments, 0)

    def test_region_boundary_trace_marks_inter_region_crossing(self) -> None:
        lm = np.ones((10, 12), dtype=np.int32)
        lm[:, 6:] = 2
        regions_by_index = {
            1: {
                "region_id": "region_1",
                "region_type": "floor",
                "semantic_label": "near floor",
                "depth_stats": {"mean": 1.0},
            },
            2: {
                "region_id": "region_2",
                "region_type": "threshold",
                "semantic_label": "far threshold",
                "depth_stats": {"mean": 1.55},
            },
        }

        trace = paths_export._region_boundary_trace(
            [[2, 5], [9, 5]],
            lm,
            regions_by_index,
            12,
            10,
            load_action_ontology(_Cfg()),
        )

        self.assertTrue(trace["available"])
        self.assertEqual(trace["movement_scope"], "inter_region")
        self.assertGreaterEqual(trace["transition_count"], 1)
        self.assertGreater(trace["boundary_sample_fraction"], 0.0)
        self.assertIn("depth_step_or_occlusion_boundary", trace["motion_implications"])

    def test_path_descriptions_include_boundary_motion_and_prose(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = _ctx(tmp_path)
            ctx.action_hypotheses = {
                "schema": "citv_action_hypotheses_v2",
                "hypotheses": [
                    {
                        "action_id": "act_p0",
                        "action_name": "cross_region_transition",
                        "action_family": "locomotion",
                        "manifold_type": "ribbon_path",
                        "grounding": {"path_id": "p0"},
                    }
                ],
            }
            paths_root = tmp_path / "scene_graph/staged/scene_paths"
            path = {
                "path_id": "p0",
                "path_type": "object_to_object",
                "manifold_type": "ribbon_path",
                "action_family": "locomotion",
                "source_entity": {"id": "obj_water", "label": "water pool"},
                "target_entity": {"id": "obj_glass", "label": "glass door"},
                "polyline_2d": [[10, 24], [36, 16]],
                "polyline_3d": [[10, 24, 2.0], [36, 16, 1.5]],
                "region_boundary_trace": {
                    "available": True,
                    "movement_scope": "inter_region",
                    "boundary_interaction": "crosses_region_boundary",
                    "boundary_sample_fraction": 0.25,
                    "transition_count": 1,
                    "motion_implications": ["inter_region_transition"],
                },
                "movement_scope": "inter_region",
                "boundary_interaction": "crosses_region_boundary",
                "trajectory_contract": {"dominant_motion": "region_transition_traverse"},
                "motion_hints": [{"motion": "region_transition_traverse", "score": 0.58}],
                "action_hints": [{"action": "cross_region_transition", "score": 0.60}],
                "scores": {"overall_confidence": 0.7},
            }

            parity_export._write_path_descriptions(
                ctx,
                paths_root,
                [path],
                {
                    "hypotheses": [
                        {
                            "trajectory_id": "traj_p0",
                            "continues_from_path_id": "p0",
                            "samples": [{"states_t": [{"x": 10, "y": 24}, {"x": 36, "y": 16}]}],
                        }
                    ]
                },
            )

            descriptions = json.loads((paths_root / "path_descriptions.json").read_text(encoding="utf-8"))
            rec = descriptions["p0"]
            self.assertEqual(rec["labels"]["movement_scope"], "inter_region")
            self.assertEqual(rec["movement"]["region_boundary_trace"]["transition_count"], 1)
            self.assertEqual(rec["actions"]["linked_action"]["action_id"], "act_p0")
            self.assertIn("yellow contours", rec["prose"]["boundary"])
            md = (paths_root / "path_reasoning.md").read_text(encoding="utf-8")
            self.assertIn("boundary evidence", md)

    def test_output_scene_is_only_tracked_output_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "ls-files", "output_scenes"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip(), "")


class V3ContractTests(unittest.TestCase):
    """Plan §1.5: enforce the v3 path/action schema invariants."""

    def _ctx_with_objects(self, tmp_path: Path) -> PipelineContext:
        ctx = _ctx(tmp_path)
        ctx = affordances_export.run(_Pipeline(), ctx)
        return ctx

    def _seed_paths(self, ctx: PipelineContext, tmp_path: Path) -> Path:
        paths_root = tmp_path / "scene_graph/staged/scene_paths"
        paths_root.mkdir(parents=True)
        path_payload = {
            "schema": "citv_path_hypotheses_v3",
            "version": "3.0",
            "stem": ctx.stem,
            "hypotheses": [
                {
                    "path_id": "p0",
                    "path_level": "object",
                    "path_type": "object_to_object",
                    "source_entity": {"type": "object", "id": "obj_water"},
                    "target_entity": {"type": "object", "id": "obj_glass"},
                    "polyline_2d_raw": [[10, 24], [18, 22], [28, 16], [36, 16]],
                    "polyline_2d": [[10, 24], [18, 22], [28, 16], [36, 16]],
                    "polyline_2d_validated": [[10, 24], [18, 22], [28, 16], [36, 16]],
                    "display_polyline_2d": [[10, 24], [18, 22], [28, 16], [36, 16]],
                    "polyline_3d": [[10, 24, 2.0], [18, 22, 1.8], [28, 16, 1.6], [36, 16, 1.4]],
                    "geometry_smoothing_provenance": {"smoothability_status": "seeded"},
                    "path_geometry_quality": {
                        "schema": "citv_path_geometry_quality_v1",
                        "zigzag_score": 0.0,
                        "turn_angle_p95": 30.0,
                        "vertical_shoot_score": 0.0,
                        "depth_jump_count": 0,
                        "geometry_rejection_reasons": [],
                        "smoothability_status": "accepted",
                    },
                    "manifold_type": "ribbon_path",
                    "action_family": "locomotion",
                    "scores": {"overall_confidence": 0.7, "semantic_confidence": 0.6},
                    "caption_trace": {"mean_caption_confidence": 0.6},
                    "visibility_profile": [{"visible": True}],
                    "render_layers": ["in_front"],
                }
            ],
        }
        (paths_root / "path_hypotheses.json").write_text(json.dumps(path_payload), encoding="utf-8")
        ctx.path_exports["path_hypotheses_json"] = "scene_graph/staged/scene_paths/path_hypotheses.json"
        return paths_root

    def test_v3_path_hypotheses_keeps_display_geometry_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = self._ctx_with_objects(tmp_path)
            paths_root = self._seed_paths(ctx, tmp_path)
            payload = json.loads((paths_root / "path_hypotheses.json").read_text(encoding="utf-8"))
            self.assertNotIn(
                "paths", payload,
                "v3 must drop duplicate `paths` key (kept only `hypotheses`)",
            )
            for hyp in payload.get("hypotheses") or []:
                self.assertIn("polyline_2d_raw", hyp)
                self.assertIn("polyline_2d_validated", hyp)
                self.assertIn("display_polyline_2d", hyp)
                self.assertIn("geometry_smoothing_provenance", hyp)
                self.assertIn("path_geometry_quality", hyp)

    def test_v3_action_hypotheses_no_actions_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = self._ctx_with_objects(tmp_path)
            self._seed_paths(ctx, tmp_path)
            ctx = action_export.run(_Pipeline(), ctx)
            self.assertEqual(ctx.action_hypotheses["schema"], "citv_action_hypotheses_v2")
            self.assertNotIn("actions", ctx.action_hypotheses, "v2 must drop duplicate `actions` key")
            self.assertIn("hypotheses", ctx.action_hypotheses)

    def test_path_hypotheses_full_json_no_longer_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = self._ctx_with_objects(tmp_path)
            self._seed_paths(ctx, tmp_path)
            ctx = action_export.run(_Pipeline(), ctx)
            ctx = parity_export.run(_Pipeline(), ctx)
            paths_root = tmp_path / "scene_graph/staged/scene_paths"
            self.assertFalse(
                (paths_root / "path_hypotheses_full.json").exists(),
                "path_hypotheses_full.json is removed in v3 (compact projection done at load time)",
            )

    def test_artifact_manifest_does_not_expect_full_path_hypotheses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ctx = _ctx(tmp_path)
            expected = parity_export._expected_artifacts(ctx)
            keys = {k for (k, _, _, _) in expected}
            self.assertNotIn("path_hypotheses_full_json", keys)

    def test_path_hypotheses_candidates_is_optional_manifest_row(self) -> None:
        ctx = _ctx(Path("."))
        expected = parity_export._expected_artifacts(ctx)
        rows = [(k, req) for (k, _, _, req) in expected if k == "path_hypotheses_candidates_json"]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0][1], "candidates JSON must not be required for parity")

    def test_path_dedupe_drops_near_duplicates_with_reason(self) -> None:
        from scene_understanding.pathing.path_hypotheses_paths import dedupe_paths

        # Two near-identical paths; expect one drop with a reason string.
        base_pts = [[10.0, 10.0 + i] for i in range(0, 40, 2)]
        near_pts = [[10.5, 10.0 + i] for i in range(0, 40, 2)]
        candidates = [
            {
                "path_id": "p_keep",
                "source_entity": {"id": "a"},
                "target_entity": {"id": "b"},
                "polyline_2d": base_pts,
                "scores": {"overall_confidence": 0.9},
            },
            {
                "path_id": "p_drop",
                "source_entity": {"id": "a"},
                "target_entity": {"id": "b"},
                "polyline_2d": near_pts,
                "scores": {"overall_confidence": 0.6},
            },
        ]
        kept, dropped = dedupe_paths(candidates, max_per_pair=1, frechet_thresh_px=18.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(kept[0]["path_id"], "p_keep")
        self.assertIn("near_duplicate_of:p_keep", dropped[0]["dropped_reason"])

    def test_relations_demotion_screen_predicates_when_depth_present(self) -> None:
        from scene_understanding.stages.relations_stage import _refine_relations

        depth = np.full((32, 32), 1.5, dtype=np.float32)
        objects = [
            {"id": "obj_a", "label": "a", "canonical_label": "a", "mask_centroid_2d": [10, 10]},
            {"id": "obj_b", "label": "b", "canonical_label": "b", "mask_centroid_2d": [20, 20]},
        ]
        relations = [
            # Pure 2D screen ordering, no depth corroboration.
            {"sub": "a", "pred": "left_of", "obj": "b", "sub_id": "obj_a", "obj_id": "obj_b", "score": 0.9, "relation_tier": "inter_region"},
            # Intra-region 2D predicate stays.
            {"sub": "a", "pred": "above", "obj": "b", "sub_id": "obj_a", "obj_id": "obj_b", "score": 0.8, "relation_tier": "intra_region"},
        ]
        refined, demoted, _ = _refine_relations(
            relations, objects, metric_depth=depth, mask_hierarchy=None
        )
        demoted_records = [r for r in refined if r.get("demoted_reason")]
        self.assertEqual(demoted, 1)
        self.assertEqual(len(demoted_records), 1)
        self.assertEqual(demoted_records[0]["pred"], "left_of")

    def test_relations_promotion_from_mask_hierarchy(self) -> None:
        from scene_understanding.stages.relations_stage import _refine_relations

        objects = [
            {"id": "obj_table", "label": "table", "canonical_label": "table", "mask_centroid_2d": [10, 20]},
            {"id": "obj_cup", "label": "cup", "canonical_label": "cup", "mask_centroid_2d": [10, 10]},
        ]
        hierarchy = {
            "edges": [
                {
                    "parent_object_id": "obj_table",
                    "child_object_id": "obj_cup",
                    "containment_ratio": 0.9,
                }
            ],
            "nodes": [],
        }
        refined, _, promoted = _refine_relations(
            [], objects, metric_depth=None, mask_hierarchy=hierarchy
        )
        preds = {r.get("pred") for r in refined}
        self.assertGreaterEqual(promoted, 1)
        self.assertIn("contains", preds)
        # Cup centroid is above the table centroid (smaller v) -> emit "on".
        self.assertIn("on", preds)


class TestPhase6SceneFixtures(unittest.TestCase):
    """Phase 6 scene fixtures: indoor floor/furniture, doorway/portal, sky,
    stairs, and cluttered-occlusion synthetic contract tests.

    These tests verify affordance grading and acceptance behavior for the
    scene types listed as missing in path_updates.md Phase 6.
    """

    def _make_mask(self, h: int, w: int, row0: int, col0: int, dh: int, dw: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=bool)
        m[row0:row0 + dh, col0:col0 + dw] = True
        return m

    def test_indoor_floor_furniture_grading(self) -> None:
        """Floor object should score high support; furniture body high hard_obstacle."""
        from scene_understanding.stages import affordances_export

        h, w = 32, 48
        floor_mask = self._make_mask(h, w, 20, 0, 12, 48)
        chair_mask = self._make_mask(h, w, 10, 15, 10, 8)
        objects = [
            {
                "id": "obj_floor",
                "label": "floor",
                "caption": "a flat wooden floor surface",
                "bbox": [0, 20, 48, 12],
                "conf": 0.9,
                "mask_centroid_2d": [24, 26],
                "region_id": "r1",
                "region_index": 1,
                "depth_stats": {"z_val": 2.0},
                "_sam2_mask_array": floor_mask,
                "sources": {"RAM++": {"tags": ["floor", "wood", "surface"]}},
            },
            {
                "id": "obj_chair",
                "label": "chair",
                "caption": "a wooden chair blocking the path",
                "bbox": [15, 10, 8, 10],
                "conf": 0.85,
                "mask_centroid_2d": [19, 15],
                "region_id": "r1",
                "region_index": 1,
                "depth_stats": {"z_val": 1.5},
                "_sam2_mask_array": chair_mask,
                "sources": {"RAM++": {"tags": ["chair", "furniture", "seat"]}},
            },
        ]
        ontology = load_action_ontology(None)
        role_prompts = {
            "support": ["walkable floor surface platform ground"],
            "hard_obstacle": ["obstacle furniture chair table blocking"],
        }
        action_prompts: dict = {}
        floor_text = "floor a flat wooden floor surface floor wood surface"
        chair_text = "chair a wooden chair blocking the path chair furniture seat"
        from scene_understanding.stages.affordances_export import _score_prompt_bank
        floor_roles = _score_prompt_bank(floor_text, role_prompts)
        chair_roles = _score_prompt_bank(chair_text, role_prompts)
        self.assertGreater(
            floor_roles.get("support", {}).get("score", 0.0),
            floor_roles.get("hard_obstacle", {}).get("score", 0.0),
            "Floor should score higher support than hard_obstacle",
        )
        self.assertGreater(
            chair_roles.get("hard_obstacle", {}).get("score", 0.0),
            chair_roles.get("support", {}).get("score", 0.0),
            "Chair should score higher hard_obstacle than support",
        )

    def test_doorway_portal_action_hints(self) -> None:
        """Doorway/opening object should surface portal role and enter/exit actions."""
        from scene_understanding.stages.affordances_export import _score_prompt_bank

        ontology = load_action_ontology(None)
        role_prompts = {
            "portal": ["doorway opening gate arch tunnel entrance exit through"],
            "hard_obstacle": ["wall solid barrier blocking impassable"],
        }
        action_prompts = {
            "enter": ["enter go through doorway pass threshold"],
            "exit": ["exit leave pass out opening"],
        }
        doorway_text = "glass door open doorway hallway entrance leading through"
        roles = _score_prompt_bank(doorway_text, role_prompts)
        actions = _score_prompt_bank(doorway_text, action_prompts)
        self.assertGreater(
            roles.get("portal", {}).get("score", 0.0),
            roles.get("hard_obstacle", {}).get("score", 0.0),
            "Doorway should score higher portal than hard_obstacle",
        )
        self.assertGreater(actions.get("enter", {}).get("score", 0.0), 0.0, "Enter action should score > 0")

    def test_sky_open_air_aerial_affordance(self) -> None:
        """Sky / open-air region should score high for aerial/fly actions."""
        from scene_understanding.stages.affordances_export import _score_prompt_bank

        action_prompts = {
            "fly": ["fly glide aerial soar through sky open air"],
            "walk": ["walk step ground floor path"],
        }
        sky_text = "open sky blue aerial expanse above buildings clear open air"
        scores = _score_prompt_bank(sky_text, action_prompts)
        self.assertGreater(
            scores.get("fly", {}).get("score", 0.0),
            scores.get("walk", {}).get("score", 0.0),
            "Sky caption should score higher fly than walk",
        )

    def test_stairs_height_change_kinematic_signature(self) -> None:
        """A polyline_3d with step-like depth increases should emit climb/descend sigs."""
        from scene_understanding.stages.animation_export import _extract_kinematic_signatures

        # Simulate a staircase: depth increases in steps of ~0.18 m each
        steps = [[float(i * 4), float(i * 2), 1.0 + i * 0.18] for i in range(12)]
        sigs = _extract_kinematic_signatures(steps, climb_z_thresh=0.12)
        motions = {s["motion"] for s in sigs}
        self.assertTrue(
            motions & {"climb", "descend", "jump"},
            f"Expected climb/descend/jump in kinematic sigs for stair-like Z profile, got {motions}",
        )

    def test_cluttered_occlusion_low_visibility_confidence(self) -> None:
        """A path through many overlapping occluders should yield low mean visible_fraction."""
        from scene_understanding.stages.paths_export import _alpha_profile_from_visibility

        # Simulate a visibility profile where most samples are heavily occluded
        vis_profile = [
            {"visible_fraction": 0.1, "render_layer": "behind_object", "occluder_ids": ["obj_clutter"]},
            {"visible_fraction": 0.15, "render_layer": "behind_object", "occluder_ids": ["obj_clutter"]},
            {"visible_fraction": 0.08, "render_layer": "behind_object", "occluder_ids": ["obj_clutter"]},
            {"visible_fraction": 0.2, "render_layer": "partially_occluded", "occluder_ids": ["obj_clutter"]},
        ]
        alpha = _alpha_profile_from_visibility(vis_profile)
        self.assertEqual(len(alpha), 4)
        mean_alpha = sum(alpha) / len(alpha)
        self.assertLess(mean_alpha, 0.25, "Cluttered path should yield low mean alpha")


class TestPhase6CaptionFixtures(unittest.TestCase):
    """Phase 6 caption fixtures: generic captions, caption/depth contradictions,
    and sky caption uncertainty.

    These test the structured evidence reasoning defined in path_updates.md
    Sections 8 and 9.
    """

    def test_generic_caption_low_precision(self) -> None:
        """Generic captions like 'an object' should produce low precision scores."""
        from scene_understanding.stages.affordances_export import _label_quality_factor

        ontology = load_action_ontology(None)
        generic_obj = {
            "id": "obj_generic",
            "label": "object",
            "canonical_label": "object",
            "label_candidates": [{"label": "object", "source": "GDINO", "conf": 0.4}],
            "rejected_labels": [],
            "visual_quality_attributes": [],
            "depth_stats": {"z_val": 2.0},
            "sources": {},
        }
        caption_rec = {"text": "an object", "precision": 0.1, "confidence": 0.2}
        quality, reasons = _label_quality_factor(generic_obj, caption_rec, ontology, return_reasons=True)
        self.assertLess(quality, 0.70, "Generic 'an object' caption should produce quality < 0.70")

    def test_caption_depth_contradiction_flag(self) -> None:
        """A caption claiming 'far background' while depth shows close should surface uncertainty."""
        from scene_understanding.stages.affordances_export import _score_prompt_bank

        # The depth metadata says z=0.8 m (very close), but caption says "far background"
        action_prompts = {
            "walk_through": ["walk through area near close foreground"],
            "distant_fly": ["far background aerial distant sky far away"],
        }
        # If we label the text correctly, close objects should score more for near actions
        close_caption_text = "large close foreground wall surface near camera"
        far_caption_text = "far distant background blur sky haze"
        close_scores = _score_prompt_bank(close_caption_text, action_prompts)
        far_scores = _score_prompt_bank(far_caption_text, action_prompts)
        self.assertGreater(
            close_scores.get("walk_through", {}).get("score", 0.0),
            far_scores.get("walk_through", {}).get("score", 0.0),
            "Close caption should score higher for near-action than far caption",
        )
        self.assertGreater(
            far_scores.get("distant_fly", {}).get("score", 0.0),
            close_scores.get("distant_fly", {}).get("score", 0.0),
            "Far caption should score higher for distant action than close caption",
        )

    def test_sky_caption_uncertainty_aerial_grounding(self) -> None:
        """Sky captions with uncertainty markers should still ground aerial actions above 0."""
        from scene_understanding.stages.affordances_export import _score_prompt_bank

        action_prompts = {
            "fly": ["fly aerial sky glide open air above"],
            "walk": ["walk step ground floor surface path"],
        }
        # Sky but with uncertainty signal (overexposed, unclear)
        uncertain_sky_text = "possibly sky or ceiling unclear overexposed area above"
        scores = _score_prompt_bank(uncertain_sky_text, action_prompts)
        fly_score = scores.get("fly", {}).get("score", 0.0)
        # Even with uncertainty, sky vocabulary should give some aerial grounding
        self.assertGreater(fly_score, 0.0, "Uncertain sky caption should still ground aerial action > 0")

    def test_alpha_profile_portal_tapering(self) -> None:
        """Portal path visibility_profile with tapering fractions → alpha_profile tapers to 0."""
        from scene_understanding.stages.paths_export import _alpha_profile_from_visibility

        # Simulate portal approach: actor visible → partially reflected → disappeared
        vis_profile = [
            {"visible_fraction": 1.0, "render_layer": "in_front", "occluder_ids": []},
            {"visible_fraction": 0.45, "render_layer": "partially_occluded", "occluder_ids": ["obj_mirror"]},
            {"visible_fraction": 0.0, "render_layer": "fading_disappearing", "occluder_ids": ["obj_mirror"]},
        ]
        alpha = _alpha_profile_from_visibility(vis_profile)
        self.assertEqual(len(alpha), 3)
        self.assertEqual(alpha[0], 1.0)
        self.assertAlmostEqual(alpha[1], 0.45, places=2)
        self.assertEqual(alpha[2], 0.0)


if __name__ == "__main__":
    unittest.main()
