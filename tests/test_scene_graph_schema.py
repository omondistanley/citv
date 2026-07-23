import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scene_understanding.output.scene_graph_builder import SceneGraphBuilder
from scene_understanding.output.savers import VisualizationSaver
from scene_understanding.pipeline import SceneUnderstandingPipeline as PackagePipeline


REPO_ROOT = Path(__file__).resolve().parent.parent
TRUTH_DIR = REPO_ROOT / "output_scene-truth" / "scene_graph"
LEGACY_METADATA_KEYS = {
    "timestamp",
    "segmentor",
    "intrinsics",
    "models",
    "rampp_tags",
    "gdino_query_used",
    "relation_sources",
    "relation_debug",
    "depth_map",
    "segmentation_image",
    "sam2_segmentation_image",
    "sam2_tinted_overlay_image",
}
LEGACY_OBJECT_KEYS = {
    "id",
    "label",
    "confidence",
    "conf",
    "bbox",
    "segmentor",
    "coordinates_3d",
    "depth_stats",
    "mask_centroid_2d",
    "coordinates_3d_no_erosion",
    "depth_stats_no_erosion",
    "mask_centroid_2d_no_erosion",
    "sam2_mask_index",
    "mask_matched",
    "mask_path",
    "depth_map_path",
    "sources",
}
ADDITIVE_TOP_LEVEL_KEYS = {"relations", "mask_hierarchy", "layers", "regions"}
ADDITIVE_METADATA_KEYS = {
    "regions_json",
    "regions_image",
    "regions_overlay_image",
    "region_segmentation_image",
    "region_sam2_segmentation_image",
    "region_tinted_overlay_image",
}
ADDITIVE_OBJECT_KEYS = {
    "name",
    "canonical_name",
    "aliases",
    "category",
    "source_labels",
    "layer_type",
    "parent_object_id",
    "child_object_ids",
    "part_mask_ids",
    "contains",
    "contained_by",
    "occludes",
    "occluded_by",
    "region_index",
    "region_id",
    "coordinates_3d_region_relative",
    "region_depth_percentile",
    "depth_plausibility_score",
    "label_warning",
}
LEGACY_SOURCE_KEYS = {"GroundedSAM2", "Florence2", "RAM++", "Pix2SG"}
LEGACY_NESTED_REL_KEYS = {"predicate", "target_id", "target_label", "target_caption"}
OPTIONAL_NESTED_REL_KEYS = {"relation_tier", "score", "source_layer"}


def _load_legacy_pipeline_class():
    module_path = REPO_ROOT / "scene_understanding.py"
    spec = importlib.util.spec_from_file_location("_scene_understanding_legacy_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.SceneUnderstandingPipeline


def _truth_scene_files():
    files = sorted(TRUTH_DIR.glob("*_scene.json"))
    return files


def _dummy_objects():
    parent_mask = np.zeros((24, 24), dtype=np.uint8)
    parent_mask[3:21, 3:21] = 1
    child_mask = np.zeros((24, 24), dtype=np.uint8)
    child_mask[8:14, 8:14] = 1

    base_objects = [
        {
            "id": "obj_parent",
            "label": "brown",
            "confidence": 0.91,
            "conf": 0.91,
            "bbox": [3, 3, 20, 20],
            "segmentor": "GroundedSAM2",
            "coordinates_3d": {"x": 0.0, "y": 0.0, "z": 3.0},
            "depth_stats": {"mean": 3.0, "median": 3.0},
            "mask_centroid_2d": [11, 11],
            "coordinates_3d_no_erosion": {"x": 0.0, "y": 0.0, "z": 3.1},
            "depth_stats_no_erosion": {"mean": 3.1, "median": 3.1},
            "mask_centroid_2d_no_erosion": [11, 11],
            "sam2_mask_index": 0,
            "mask_matched": True,
            "mask_path": None,
            "depth_map_path": None,
            "sources": {
                "GroundedSAM2": {"caption": "brown | person", "label": "brown", "confidence": 0.91},
                "Florence2": {"label": "person", "caption": "person"},
                "RAM++": {"label": "person", "caption": "person", "tags": ["person", "brown"]},
                "Pix2SG": {
                    "relations": [
                        {
                            "predicate": "in_front_of",
                            "target_id": "obj_child",
                            "target_label": "charger",
                            "target_caption": "charger",
                            "score": 0.75,
                        }
                    ]
                },
            },
            "_sam2_mask_array": parent_mask,
            "name": "person",
            "canonical_name": "person",
            "aliases": ["person", "brown"],
            "category": "person",
            "source_labels": {
                "GroundedSAM2": {"label": "brown", "caption": "brown | person", "confidence": 0.91},
                "Florence2": {"label": "person", "caption": "person"},
                "RAM++": {"label": "person", "caption": "person", "tags": ["person", "brown"]},
            },
            "layer_type": "unassigned",
            "parent_object_id": None,
            "child_object_ids": [],
            "part_mask_ids": [],
            "contains": [],
            "contained_by": [],
            "occludes": [],
            "occluded_by": [],
            "region_index": None,
            "region_id": None,
            "coordinates_3d_region_relative": None,
            "region_depth_percentile": None,
            "depth_plausibility_score": None,
            "label_warning": None,
        },
        {
            "id": "obj_child",
            "label": "charger",
            "confidence": 0.88,
            "conf": 0.88,
            "bbox": [8, 8, 13, 13],
            "segmentor": "GroundedSAM2",
            "coordinates_3d": {"x": 0.0, "y": 0.0, "z": 5.5},
            "depth_stats": {"mean": 5.5, "median": 5.5},
            "mask_centroid_2d": [10, 10],
            "coordinates_3d_no_erosion": {"x": 0.0, "y": 0.0, "z": 5.6},
            "depth_stats_no_erosion": {"mean": 5.6, "median": 5.6},
            "mask_centroid_2d_no_erosion": [10, 10],
            "sam2_mask_index": 1,
            "mask_matched": True,
            "mask_path": None,
            "depth_map_path": None,
            "sources": {
                "GroundedSAM2": {"caption": "charger", "label": "charger", "confidence": 0.88},
                "Florence2": {"label": "charger", "caption": "charger"},
                "RAM++": {"label": "charger", "caption": "charger", "tags": ["charger"]},
                "Pix2SG": {"relations": []},
            },
            "_sam2_mask_array": child_mask,
            "name": "charger",
            "canonical_name": "charger",
            "aliases": ["charger"],
            "category": "electronics",
            "source_labels": {
                "GroundedSAM2": {"label": "charger", "caption": "charger", "confidence": 0.88},
                "Florence2": {"label": "charger", "caption": "charger"},
                "RAM++": {"label": "charger", "caption": "charger", "tags": ["charger"]},
            },
            "layer_type": "unassigned",
            "parent_object_id": None,
            "child_object_ids": [],
            "part_mask_ids": [],
            "contains": [],
            "contained_by": [],
            "occludes": [],
            "occluded_by": [],
            "region_index": None,
            "region_id": None,
            "coordinates_3d_region_relative": None,
            "region_depth_percentile": None,
            "depth_plausibility_score": None,
            "label_warning": None,
        },
    ]
    return base_objects


def test_truth_scene_graphs_define_legacy_contract():
    truth_files = _truth_scene_files()
    if not truth_files:
        pytest.skip(f"No truth scene fixtures in {TRUTH_DIR} (optional golden set)")
    non_empty_files = 0
    for scene_path in truth_files:
        payload = json.loads(scene_path.read_text())
        assert "metadata" in payload
        assert "objects" in payload
        assert LEGACY_METADATA_KEYS.issubset(payload["metadata"].keys())
        if not payload["objects"]:
            continue
        non_empty_files += 1

        for obj in payload["objects"]:
            assert LEGACY_OBJECT_KEYS.issubset(obj.keys())
            assert LEGACY_SOURCE_KEYS.issubset(obj["sources"].keys())
            for rel in obj["sources"]["Pix2SG"].get("relations", []):
                assert LEGACY_NESTED_REL_KEYS.issubset(rel.keys())
    assert non_empty_files > 0


def test_enhanced_scene_graph_builder_is_legacy_superset():
    scene_graph = SceneGraphBuilder.build_enhanced_scene_json(
        metadata={key: f"value_for_{key}" for key in LEGACY_METADATA_KEYS},
        objects=[{key: None for key in LEGACY_OBJECT_KEYS | ADDITIVE_OBJECT_KEYS}],
        relations=[],
        mask_hierarchy={"edges": [], "root_object_ids": [], "num_edges": 0},
        layers={"ordering": [], "bands": [], "depth_quantiles": {}},
        regions=[],
    )
    assert set(scene_graph.keys()) == {
        "metadata",
        "objects",
        "relations",
        "mask_hierarchy",
        "layers",
        "regions",
    }
    assert LEGACY_METADATA_KEYS.issubset(scene_graph["metadata"].keys())
    assert LEGACY_OBJECT_KEYS.issubset(scene_graph["objects"][0].keys())
    assert ADDITIVE_TOP_LEVEL_KEYS.issubset(scene_graph.keys())
    assert ADDITIVE_OBJECT_KEYS.issubset(scene_graph["objects"][0].keys())


def test_root_and_package_pipelines_match_additive_scene_helpers():
    LegacyPipeline = _load_legacy_pipeline_class()
    legacy_pipe = LegacyPipeline.__new__(LegacyPipeline)
    package_pipe = PackagePipeline.__new__(PackagePipeline)

    legacy_objects = copy.deepcopy(_dummy_objects())
    package_objects = copy.deepcopy(_dummy_objects())

    legacy_relations, legacy_hierarchy, legacy_layers = legacy_pipe._derive_scene_additions(legacy_objects)
    package_relations, package_hierarchy, package_layers = package_pipe._derive_scene_additions(package_objects)

    assert len(legacy_relations) == len(package_relations)
    assert legacy_hierarchy["num_edges"] == package_hierarchy["num_edges"]
    assert [obj["layer_type"] for obj in legacy_objects] == [obj["layer_type"] for obj in package_objects]

    for obj in legacy_objects:
        assert LEGACY_OBJECT_KEYS.issubset(obj.keys())
        assert ADDITIVE_OBJECT_KEYS.issubset(obj.keys())

    assert any(rel["predicate"] == "contains" for rel in legacy_relations)
    assert any(rel["predicate"] == "inside_of" for rel in legacy_relations)
    assert any(obj["occludes"] for obj in legacy_objects)
    assert any(obj["contained_by"] for obj in legacy_objects)
    assert legacy_layers["bands"]


def test_relations_map_sidecar_is_written(tmp_path):
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    objects = _dummy_objects()
    relations = [
        {
            "subject_id": "obj_parent",
            "predicate": "contains",
            "object_id": "obj_child",
        }
    ]
    out_path = tmp_path / "relations_map.png"
    VisualizationSaver.save_relations_map(image, objects, relations, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
