"""Phase 7 unit tests: staged semantic cost fusion, pair gates, coarse grid, manifest MP4 QA."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

try:
    import skfmm  # type: ignore
except Exception:  # pragma: no cover
    skfmm = None

from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.pathing.goal_anchors import pair_passes_locomotion_gates, vertical_structure_heuristic
from scene_understanding.pathing.ground_route_grid import plan_coarse_support_path
from scene_understanding.pathing.path_contract_compliance import evaluate_paths_root
from scene_understanding.pathing.staged_semantic_cost import blend_traversability_with_fused_cost, build_staged_semantic_cost_map
from scene_understanding.stages import parity_export


class _FusedCfg:
    path_cost_use_vlm_layer = False
    path_cost_ontology_caption_layer = True
    path_cost_caption_layer_precision_cap = 0.22
    path_cost_agent_profile_file = ""


class _PairCfg:
    path_pair_min_span_px = 100.0
    path_pair_max_depth_delta_m = 0.0
    path_pair_skip_facade_facade = False
    path_pair_facade_skip_requires_vertical = False
    path_pair_require_relation = False


class StagedSemanticCostTests(unittest.TestCase):
    def test_build_staged_semantic_cost_map_shape_and_range(self) -> None:
        h, w = 24, 32
        img = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 2.0, dtype=np.float32)
        obs = np.zeros((h, w), dtype=bool)
        scm = build_staged_semantic_cost_map(img, depth, obs, [], _FusedCfg(), {"fx": 90.0, "fy": 90.0, "cx": w / 2, "cy": h / 2})
        self.assertEqual(scm.cost.shape, (h, w))
        self.assertTrue(np.all(scm.cost >= 0.0))
        self.assertTrue(np.all(scm.cost <= 1.0))

    def test_blend_traversability_with_fused_cost_respects_floor(self) -> None:
        base = np.ones((4, 4), dtype=np.float32)
        fused = np.ones((4, 4), dtype=np.float32)
        out = blend_traversability_with_fused_cost(base, fused, blend=1.0, speed_floor=0.06)
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(np.all(out >= 0.06 - 1e-5))


class PairGateTests(unittest.TestCase):
    def test_min_span_rejects_close_centroids(self) -> None:
        src = {"id": "a", "mask_centroid_2d": [10.0, 10.0], "canonical_label": "person"}
        tgt = {"id": "b", "mask_centroid_2d": [12.0, 10.0], "canonical_label": "chair"}
        ok, reason = pair_passes_locomotion_gates(src, tgt, [], _PairCfg(), h=100, w=100)
        self.assertFalse(ok)
        self.assertEqual(reason, "below_min_span_px")

    def test_facade_pair_skip(self) -> None:
        class Cfg:
            path_pair_min_span_px = 0.0
            path_pair_max_depth_delta_m = 0.0
            path_pair_skip_facade_facade = True
            path_pair_facade_skip_requires_vertical = False
            path_pair_require_relation = False

        src = {"id": "a", "mask_centroid_2d": [10.0, 50.0], "canonical_label": "building"}
        tgt = {"id": "b", "mask_centroid_2d": [80.0, 50.0], "canonical_label": "wall"}
        ok, reason = pair_passes_locomotion_gates(src, tgt, [], Cfg(), h=100, w=100)
        self.assertFalse(ok)
        self.assertEqual(reason, "facade_facade_skip")


class VerticalHeuristicTests(unittest.TestCase):
    def test_tall_bbox_triggers_vertical_heuristic(self) -> None:
        h, w = 120, 80
        mask = np.zeros((h, w), dtype=bool)
        mask[10:100, 40:48] = True
        obj = {"_sam2_mask_array": mask}
        self.assertTrue(vertical_structure_heuristic(obj, h, w, aspect_thresh=1.2, min_height_frac=0.1))


@unittest.skipUnless(skfmm is not None, "scikit-fmm required for coarse-grid FMM")
class CoarseSupportGridTests(unittest.TestCase):
    def test_plan_coarse_support_path_connects_strip(self) -> None:
        h, w = 80, 80
        speed = np.ones((h, w), dtype=np.float32)
        support = np.zeros((h, w), dtype=bool)
        feasible = np.zeros((h, w), dtype=bool)
        support[60:, :] = True
        feasible[60:, :] = True
        poly = plan_coarse_support_path(speed, support, feasible, (5, 65), (75, 65), step=10)
        self.assertGreaterEqual(len(poly), 2)
        # Endpoints land near requested coarse cells along the bottom strip.
        self.assertGreater(poly[0][1], h // 2)
        self.assertGreater(poly[-1][1], h // 2)


class ManifestMp4QualityTests(unittest.TestCase):
    def test_tiny_animation_qa_mp4_marked_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rel = "scene_graph/staged/t_paths/animation_qa_12/panel_00_paths_trajectories.mp4"
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"mv")

            ctx = PipelineContext(
                image_path=tmp / "in.jpg",
                output_dir=tmp,
                stem="t",
                timestamp="0",
                img_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
                img_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                height=4,
                width=4,
                intrinsics={},
            )

            class Cfg:
                artifact_manifest_mp4_min_bytes = 256

            issue = parity_export._artifact_quality_issue(ctx, "animation_qa_video_12", rel, False, {}, Cfg())
            self.assertIn("mp4_too_small", issue)


class DedupeOffTests(unittest.TestCase):
    def test_dedupe_paths_max_zero_keeps_all(self) -> None:
        from scene_understanding.pathing.path_hypotheses_paths import dedupe_paths

        base = [[float(i), 0.0] for i in range(6)]
        dup = [[0.1 + float(i), 0.1] for i in range(6)]
        cands = [
            {"path_id": "p1", "polyline_2d": base, "source_entity": {"id": "a"}, "target_entity": {"id": "b"}, "scores": {"overall_confidence": 0.9}},
            {"path_id": "p2", "polyline_2d": dup, "source_entity": {"id": "a"}, "target_entity": {"id": "b"}, "scores": {"overall_confidence": 0.5}},
        ]
        kept, dropped = dedupe_paths(cands, max_per_pair=0)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])


class PathGeometryQualityTests(unittest.TestCase):
    def test_display_geometry_simplifies_micro_zigzags(self) -> None:
        from scene_understanding.pathing.path_geometry_quality import build_display_geometry, evaluate_geometry_quality

        raw = [[0.0, 20.0], [5.0, 22.0], [10.0, 18.0], [15.0, 22.0], [20.0, 18.0], [30.0, 20.0]]
        feasible = np.ones((40, 40), dtype=bool)
        geom = build_display_geometry(raw, width=40, height=40, feasible_mask=feasible, cfg=None)
        self.assertIn("display_polyline_2d", geom)
        self.assertLessEqual(len(geom["display_polyline_2d"]), max(2, len(raw) * 2))
        quality = evaluate_geometry_quality(
            raw_polyline=geom["polyline_2d_raw"],
            display_polyline=geom["display_polyline_2d"],
            feasible_mask=feasible,
            cfg=None,
        )
        self.assertIn("zigzag_score", quality)
        self.assertIn("geometry_rejection_reasons", quality)

    def test_large_support_snap_displacement_is_rejected(self) -> None:
        from scene_understanding.pathing.path_geometry_quality import evaluate_geometry_quality

        raw = [[5.0, 2.0], [10.0, 2.0]]
        snapped_3d = [[5.0, 80.0, 2.0], [10.0, 80.0, 2.0]]
        quality = evaluate_geometry_quality(
            raw_polyline=raw,
            display_polyline=raw,
            polyline_3d=snapped_3d,
            cfg=None,
        )
        self.assertIn("geometry_support_snap_displacement_too_large", quality["geometry_rejection_reasons"])


class ComplianceAuditTests(unittest.TestCase):
    def test_evaluate_paths_root_flags_missing_contract_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths_root = root / "scene_paths"
            paths_root.mkdir(parents=True, exist_ok=True)
            (paths_root / "path_hypotheses.json").write_text(
                '{"hypotheses":[{"path_id":"p1"},{"path_id":"p2"}]}',
                encoding="utf-8",
            )
            (paths_root / "trajectory_hypotheses.json").write_text(
                '{"hypotheses":[{"continues_from_path_id":"p1"}]}',
                encoding="utf-8",
            )
            (paths_root / "path_atlas_manifest.json").write_text(
                '{"paths":[{"path_id":"p1"}]}',
                encoding="utf-8",
            )
            (paths_root / "path_trajectories_batches.json").write_text(
                '{"batches":[{"path_ids":["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11"]}]}',
                encoding="utf-8",
            )
            (paths_root / "path_visual_qa.json").write_text('{"paths":[{"path_id":"p1"}]}', encoding="utf-8")
            report = evaluate_paths_root(paths_root)
            self.assertFalse(report["ok"])
            self.assertIn("atlas_path_count_mismatch", report["failures"])
            self.assertIn("missing_trajectory_links", report["failures"])
            self.assertIn("batch_size_exceeds_10", report["failures"])

    def test_evaluate_paths_root_passes_when_contract_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths_root = root / "scene_paths"
            (paths_root / "animation_qa_24").mkdir(parents=True, exist_ok=True)
            (paths_root / "per_path").mkdir(parents=True, exist_ok=True)
            (paths_root / "path_hypotheses.json").write_text(
                '{"hypotheses":[{"path_id":"p1"},{"path_id":"p2"}]}',
                encoding="utf-8",
            )
            (paths_root / "trajectory_hypotheses.json").write_text(
                '{"hypotheses":[{"continues_from_path_id":"p1"},{"continues_from_path_id":"p2"}]}',
                encoding="utf-8",
            )
            (paths_root / "path_atlas_manifest.json").write_text(
                '{"paths":[{"path_id":"p1"},{"path_id":"p2"}]}',
                encoding="utf-8",
            )
            (paths_root / "path_trajectories_batches.json").write_text(
                '{"batches":[{"path_ids":["p1","p2"]}]}',
                encoding="utf-8",
            )
            (paths_root / "path_visual_qa.json").write_text(
                '{"paths":[{"path_id":"p1"},{"path_id":"p2"}]}',
                encoding="utf-8",
            )
            (paths_root / "per_path/p1.json").write_text("{}", encoding="utf-8")
            (paths_root / "per_path/p2.json").write_text("{}", encoding="utf-8")
            (paths_root / "per_path/p1.md").write_text("x", encoding="utf-8")
            (paths_root / "per_path/p2.md").write_text("x", encoding="utf-8")
            (paths_root / "animation_qa_24/animation_qa_manifest.json").write_text(
                '{"panel_videos":[{"segments":[{"path_id":"p1"}]}]}',
                encoding="utf-8",
            )
            (paths_root / "animation_qa_24/animation_qa_scores.json").write_text(
                '{"panels":[{"metrics":[{"path_id":"p1"}]}]}',
                encoding="utf-8",
            )
            report = evaluate_paths_root(paths_root)
            self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
