"""Tests A-D: VLM cost layer, FMM corridor preference, precision weighting, end-to-end path quality."""

from __future__ import annotations

import unittest
import warnings
from typing import List
from unittest.mock import patch

import numpy as np

try:
    import skfmm  # type: ignore
except Exception:
    skfmm = None

from scene_understanding.pathing.semantic_cost import CostLayer, SemanticCostMap, precision_weighted_fuse
from scene_understanding.pathing.staged_semantic_cost import (
    blend_traversability_with_fused_cost,
    build_staged_semantic_cost_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_img(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _uniform_depth(h: int, w: int, val: float = 2.0) -> np.ndarray:
    return np.full((h, w), val, dtype=np.float32)


def _cost_layer(name: str, cost_val: float, prec_val: float, h: int = 10, w: int = 10) -> CostLayer:
    return CostLayer(
        name=name,
        cost=np.full((h, w), cost_val, dtype=np.float32),
        precision=np.full((h, w), prec_val, dtype=np.float32),
    )


def _make_vlm_layer(h: int, w: int, cost_val: float, prec_val: float) -> CostLayer:
    return CostLayer(
        name="B_vlm",
        cost=np.full((h, w), cost_val, dtype=np.float32),
        precision=np.full((h, w), prec_val, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# A — VLM layer changes the fused cost surface
# ---------------------------------------------------------------------------

class VLMLayerInfluenceTests(unittest.TestCase):
    """A: enabling the VLM layer must produce a measurably different cost map."""

    def _cfg(self, use_vlm: bool):
        class Cfg:
            path_cost_use_vlm_layer = use_vlm
            path_cost_ontology_caption_layer = False
            path_cost_caption_layer_precision_cap = 0.22
            path_cost_agent_profile_file = ""
        return Cfg()

    def test_vlm_layer_alters_fused_cost_when_enabled(self):
        """Injecting a synthetic VLM layer with non-trivial precision shifts the fused cost."""
        h, w = 20, 20
        # Base layers present in both cases
        geometry = _cost_layer("A_geometry", 0.2, 0.6, h, w)
        depth_noise = _cost_layer("F_depth_noise", 0.1, 0.3, h, w)

        fused_without = precision_weighted_fuse([geometry, depth_noise])

        # VLM layer says high cost (0.85) with meaningful precision (0.5)
        vlm = _make_vlm_layer(h, w, 0.85, 0.5)
        fused_with = precision_weighted_fuse([geometry, depth_noise, vlm])

        diff = float(np.mean(np.abs(fused_with.cost - fused_without.cost)))
        self.assertGreater(diff, 0.05, "VLM layer had negligible effect on fused cost")

    def test_zero_precision_vlm_layer_does_not_change_cost(self):
        """A VLM layer with precision=0 must not shift the fused cost at all."""
        h, w = 10, 10
        geometry = _cost_layer("A_geometry", 0.3, 0.7, h, w)
        vlm_zero_prec = _make_vlm_layer(h, w, 0.99, 0.0)

        fused_without = precision_weighted_fuse([geometry])
        fused_with = precision_weighted_fuse([geometry, vlm_zero_prec])

        diff = float(np.mean(np.abs(fused_with.cost - fused_without.cost)))
        self.assertAlmostEqual(diff, 0.0, places=4,
            msg="Zero-precision VLM layer should not affect fused cost")

    def test_vlm_failure_emits_warning_not_exception(self):
        """When VLM layer raises, build_staged_semantic_cost_map warns and continues."""
        h, w = 16, 20
        img = _blank_img(h, w)
        depth = _uniform_depth(h, w)
        obs = np.zeros((h, w), dtype=bool)

        class Cfg:
            path_cost_use_vlm_layer = True
            path_cost_ontology_caption_layer = False
            path_cost_caption_layer_precision_cap = 0.22
            path_cost_agent_profile_file = ""
            path_cost_prompts_yaml = "/nonexistent/path/prompts.yaml"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scm = build_staged_semantic_cost_map(
                img, depth, obs, [], Cfg(),
                {"fx": 90.0, "fy": 90.0, "cx": w / 2, "cy": h / 2},
            )

        self.assertEqual(scm.cost.shape, (h, w))
        self.assertTrue(
            any("VLM" in str(w.message) or "vlm" in str(w.message).lower() for w in caught),
            "Expected a RuntimeWarning mentioning VLM failure",
        )

    def test_vlm_off_and_on_produce_different_layer_counts(self):
        """Enabling VLM adds B_vlm to the layer list; disabling omits it."""
        h, w = 16, 20
        img = _blank_img(h, w)
        depth = _uniform_depth(h, w)
        obs = np.zeros((h, w), dtype=bool)
        intr = {"fx": 90.0, "fy": 90.0, "cx": w / 2, "cy": h / 2}

        class CfgOff:
            path_cost_use_vlm_layer = False
            path_cost_ontology_caption_layer = False
            path_cost_caption_layer_precision_cap = 0.22
            path_cost_agent_profile_file = ""

        scm_off = build_staged_semantic_cost_map(img, depth, obs, [], CfgOff(), intr)
        names_off = {lyr.name for lyr in scm_off.layers}
        self.assertNotIn("B_vlm", names_off)

        # Inject a synthetic working layer_vlm via mock so we don't need GPU
        synthetic_vlm = _make_vlm_layer(h, w, 0.7, 0.4)

        class CfgOn:
            path_cost_use_vlm_layer = True
            path_cost_ontology_caption_layer = False
            path_cost_caption_layer_precision_cap = 0.22
            path_cost_agent_profile_file = ""
            path_cost_prompts_yaml = "cost_map_prompts.yaml"

        with patch(
            "scene_understanding.pathing.semantic_vlm.layer_vlm",
            return_value=synthetic_vlm,
        ), patch(
            "scene_understanding.pathing.semantic_vlm.load_cost_prompts",
            return_value=object(),  # non-None sentinel so the branch fires
        ):
            scm_on = build_staged_semantic_cost_map(img, depth, obs, [], CfgOn(), intr)

        names_on = {lyr.name for lyr in scm_on.layers}
        self.assertIn("B_vlm", names_on)
        self.assertGreater(len(scm_on.layers), len(scm_off.layers))


# ---------------------------------------------------------------------------
# B — FMM prefers low-cost corridors
# ---------------------------------------------------------------------------

@unittest.skipUnless(skfmm is not None, "scikit-fmm required")
class FMMCorridorPreferenceTests(unittest.TestCase):
    """B: the FMM planner must route through cheap pixels, not expensive ones."""

    def _backtrace(self, cost: np.ndarray, start, goal) -> List:
        from scene_understanding.pathing.semantic_fmm import fmm_backtrace
        return fmm_backtrace(cost, start=start, goal=goal)

    def test_path_stays_in_free_half(self):
        """Left half free, right half wall — start and goal both on left side."""
        h, w = 40, 80
        cost = np.full((h, w), 0.05, dtype=np.float32)
        cost[:, 40:] = 0.95

        path = self._backtrace(cost, start=(2, 2), goal=(2, 37))
        self.assertGreater(len(path), 1, "Expected a non-trivial path")
        max_x = max(p[0] for p in path)
        self.assertLess(max_x, 40, f"Path crossed into high-cost wall (max_x={max_x})")

    def test_path_routes_around_horizontal_wall(self):
        """Horizontal wall with a gap at the right — path must use the gap."""
        h, w = 60, 80
        cost = np.full((h, w), 0.05, dtype=np.float32)
        # Solid wall rows 20-40, columns 0-65 (gap at columns 66-79)
        cost[20:40, 0:66] = 0.98

        path = self._backtrace(cost, start=(10, 10), goal=(50, 10))
        self.assertGreater(len(path), 1)
        in_wall = [(x, y) for (x, y) in path if 0 <= x < 66 and 20 <= y < 40]
        self.assertEqual(len(in_wall), 0,
            f"Path entered solid wall region at: {in_wall[:5]}")

    def test_single_corridor_is_used(self):
        """Two expensive strips leave one cheap corridor — path must use it."""
        h, w = 50, 50
        cost = np.full((h, w), 0.9, dtype=np.float32)
        # Cheap corridor: column 20-25
        cost[:, 20:26] = 0.05

        path = self._backtrace(cost, start=(22, 2), goal=(22, 47))
        self.assertGreater(len(path), 1)
        outside_corridor = [(x, y) for (x, y) in path if not (20 <= x < 26)]
        # Allow start/end to deviate slightly but the bulk must stay in corridor
        self.assertLess(len(outside_corridor), 5,
            f"Path left the cheap corridor too often ({len(outside_corridor)} pts outside)")

    def test_speed_floor_prevents_zero_speed(self):
        """A cost map of all-ones must still return a path (speed floor prevents deadlock)."""
        from scene_understanding.pathing.semantic_fmm import fmm_backtrace
        cost = np.ones((20, 20), dtype=np.float32)
        path = fmm_backtrace(cost, start=(0, 0), goal=(0, 19))
        self.assertGreater(len(path), 0, "All-ones cost map should still yield a path via speed floor")


# ---------------------------------------------------------------------------
# C — Precision weighting suppresses low-confidence layers
# ---------------------------------------------------------------------------

class PrecisionWeightingTests(unittest.TestCase):
    """C: low-precision layers must not dominate the fused cost."""

    def test_high_precision_layer_dominates(self):
        """High-prec layer cost=0.8 vs low-prec layer cost=0.1 → fused closer to 0.8."""
        h, w = 10, 10
        hi = _cost_layer("hi", 0.8, 0.9, h, w)
        lo = _cost_layer("lo", 0.1, 0.05, h, w)
        fused = precision_weighted_fuse([hi, lo])
        mean = float(np.mean(fused.cost))
        self.assertGreater(mean, 0.6,
            f"High-precision layer should dominate (fused mean={mean:.3f})")

    def test_equal_precision_averages_costs(self):
        """Equal precision → fused cost is the mean of the two layer costs."""
        h, w = 8, 8
        a = _cost_layer("a", 0.2, 0.5, h, w)
        b = _cost_layer("b", 0.8, 0.5, h, w)
        fused = precision_weighted_fuse([a, b])
        mean = float(np.mean(fused.cost))
        self.assertAlmostEqual(mean, 0.5, delta=0.01,
            msg=f"Equal-precision layers should average to 0.5, got {mean:.4f}")

    def test_zero_precision_layer_ignored(self):
        """A layer with precision=0 everywhere contributes nothing to fused cost."""
        h, w = 6, 6
        base = _cost_layer("base", 0.4, 0.6, h, w)
        ghost = _cost_layer("ghost", 0.99, 0.0, h, w)
        fused = precision_weighted_fuse([base])
        fused_with_ghost = precision_weighted_fuse([base, ghost])
        diff = float(np.mean(np.abs(fused.cost - fused_with_ghost.cost)))
        self.assertAlmostEqual(diff, 0.0, places=4,
            msg="Zero-precision ghost layer should not affect fused cost")

    def test_fused_precision_increases_with_more_layers(self):
        """Adding a layer with nonzero precision must increase fused precision."""
        h, w = 5, 5
        base = _cost_layer("a", 0.5, 0.4, h, w)
        extra = _cost_layer("b", 0.5, 0.3, h, w)
        fused_one = precision_weighted_fuse([base])
        fused_two = precision_weighted_fuse([base, extra])
        self.assertTrue(
            np.all(fused_two.precision >= fused_one.precision - 1e-5),
            "Adding a non-zero precision layer must not decrease fused precision",
        )
        self.assertGreater(
            float(np.mean(fused_two.precision)),
            float(np.mean(fused_one.precision)),
            "Adding a layer with precision>0 should increase mean fused precision",
        )

    def test_precision_blend_suppresses_uncertain_vlm(self):
        """Simulate a noisy VLM layer: its low precision must not corrupt geometry evidence."""
        h, w = 12, 12
        # Geometry: confident, says low cost (open area)
        geometry = _cost_layer("A_geometry", 0.15, 0.85, h, w)
        # Noisy VLM: says very high cost but very uncertain
        noisy_vlm = _cost_layer("B_vlm", 0.95, 0.08, h, w)

        fused = precision_weighted_fuse([geometry, noisy_vlm])
        mean = float(np.mean(fused.cost))
        # Geometry should dominate: fused cost should stay closer to 0.15 than 0.95
        self.assertLess(mean, 0.5,
            f"Uncertain VLM should not override confident geometry (fused={mean:.3f})")


# ---------------------------------------------------------------------------
# D — End-to-end: semantic cost → FMM → polyline avoids obstacles
# ---------------------------------------------------------------------------

@unittest.skipUnless(skfmm is not None, "scikit-fmm required")
class EndToEndPathQualityTests(unittest.TestCase):
    """D: the full cost-to-path pipeline must produce obstacle-free routes."""

    def _run_pipeline(self, cost: np.ndarray, start, goal) -> List:
        from scene_understanding.pathing.semantic_fmm import fmm_backtrace
        return fmm_backtrace(cost, start=start, goal=goal)

    def test_room_with_doorway(self):
        """Room divided by a wall with one doorway — path must pass through the door."""
        h, w = 60, 60
        cost = np.full((h, w), 0.05, dtype=np.float32)
        # Wall at row 30, with a doorway at columns 25-34
        cost[28:32, :25] = 0.98
        cost[28:32, 35:] = 0.98

        path = self._run_pipeline(cost, start=(30, 5), goal=(30, 55))
        self.assertGreater(len(path), 1)

        # No point should be in the solid wall (outside the door)
        in_wall = [(x, y) for (x, y) in path
                   if 28 <= y < 32 and not (25 <= x < 35)]
        self.assertEqual(len(in_wall), 0,
            f"Path went through solid wall, not door: {in_wall[:5]}")

    def test_staged_cost_map_feeds_fmm_correctly(self):
        """build_staged_semantic_cost_map output plugged into FMM gives a valid, in-bounds path."""
        from scene_understanding.pathing.semantic_fmm import fmm_backtrace

        h, w = 60, 80
        img = _blank_img(h, w)
        depth = _uniform_depth(h, w)
        obs = np.zeros((h, w), dtype=bool)
        obs[25:35, 20:60] = True  # horizontal obstacle strip

        class Cfg:
            path_cost_use_vlm_layer = False
            path_cost_ontology_caption_layer = False
            path_cost_caption_layer_precision_cap = 0.22
            path_cost_agent_profile_file = ""

        scm = build_staged_semantic_cost_map(
            img, depth, obs, [], Cfg(),
            {"fx": 90.0, "fy": 90.0, "cx": w / 2, "cy": h / 2},
        )
        self.assertEqual(scm.cost.shape, (h, w))
        self.assertTrue(np.all(scm.cost >= 0.0) and np.all(scm.cost <= 1.0))

        path = fmm_backtrace(scm.cost, start=(40, 5), goal=(40, 55))
        self.assertGreater(len(path), 1, "Should find a path")

        # All waypoints must be within image bounds
        oob = [(x, y) for (x, y) in path if not (0 <= x < w and 0 <= y < h)]
        self.assertEqual(len(oob), 0, f"Path has out-of-bounds points: {oob[:5]}")

        # Obstacle pixels must cost more than well-separated free pixels
        obs_cost = float(scm.cost[obs].mean())
        free_mask = np.zeros((h, w), dtype=bool)
        free_mask[:10, :10] = True  # top-left corner, far from obstacle
        free_cost = float(scm.cost[free_mask].mean())
        self.assertGreater(obs_cost, free_cost,
            f"Obstacle pixels (cost={obs_cost:.3f}) should cost more than free pixels (cost={free_cost:.3f})")

    def test_blend_traversability_applies_cost_map_penalty(self):
        """High-cost pixels in the semantic map must reduce speed after blending."""
        h, w = 8, 8
        base_speed = np.ones((h, w), dtype=np.float32)
        fused_cost = np.zeros((h, w), dtype=np.float32)
        fused_cost[3:5, 3:5] = 0.9  # small high-cost patch

        blended = blend_traversability_with_fused_cost(base_speed, fused_cost, blend=0.72)
        patch_speed = float(blended[3:5, 3:5].mean())
        free_speed = float(np.concatenate([blended[:3, :].ravel(), blended[5:, :].ravel()]).mean())

        self.assertLess(patch_speed, free_speed,
            "High-cost patch must have lower speed after blending")
        self.assertGreater(patch_speed, 0.0, "Speed floor must prevent zero speed")

    def test_cost_map_obstacle_region_has_higher_cost_than_free(self):
        """The geometry layer alone must assign higher cost near obstacles than in open space."""
        from scene_understanding.pathing.semantic_cost import layer_geometry

        h, w = 40, 60
        img = _blank_img(h, w)
        depth = _uniform_depth(h, w)
        obs = np.zeros((h, w), dtype=bool)
        obs[15:25, 20:40] = True  # rectangular obstacle

        layer = layer_geometry(img, depth, obs)
        obs_cost = float(layer.cost[obs].mean())
        free_cost = float(layer.cost[~obs].mean())

        self.assertGreater(obs_cost, free_cost,
            f"Obstacle pixels cost={obs_cost:.3f} should exceed free pixels cost={free_cost:.3f}")


if __name__ == "__main__":
    unittest.main()
