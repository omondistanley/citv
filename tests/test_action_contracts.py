import json

import numpy as np
import pytest

from scene_understanding.action_contracts import (
    ActorSpec,
    JsonExtractionError,
    MotionContract,
    UserGeometry,
    adapt_motion_contract_to_scene,
    extract_first_json_object,
    extract_first_json_value,
    motion_contract_from_json,
)


def test_extract_first_json_object_ignores_arrays_inside_object():
    text = '''Claude says:
    {
      "polyline_2d": [[590, 150], [610, 220], [597, 750]],
      "depth_trace_m": [2.1, 2.3, 2.7],
      "action_text": "cobra rears up and strikes"
    }
    Thanks.'''
    payload = extract_first_json_object(text)
    assert payload["polyline_2d"][0] == [590, 150]
    assert payload["depth_trace_m"][-1] == 2.7


def test_extract_first_json_object_rejects_naked_coordinate_array():
    text = '[[590, 150], [610, 220]], "depth_trace_m": [2.1, 2.3]'
    with pytest.raises(JsonExtractionError):
        extract_first_json_object(text)
    assert extract_first_json_value(text) == [[590, 150], [610, 220]]


def test_motion_contract_from_json_preserves_open_vocab_actor_and_path():
    contract = motion_contract_from_json(
        {
            "contract_id": "take_1",
            "actor": {"actor_text": "photorealistic cobra", "actor_source": "generated_asset"},
            "action_text": "cobra slithers then rears up",
            "user_geometry": {
                "mode": "polyline",
                "points": [[10, 20], [30, 40], [50, 60]],
                "source": "user_drawn",
            },
        }
    )
    assert contract.actor.actor_text == "photorealistic cobra"
    assert contract.manifold_type == "ribbon_path"
    assert contract.user_geometry.points == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]


def test_scene_adapter_keeps_raw_user_geometry_and_adds_traces():
    depth = np.ones((100, 100), dtype=np.float32) * 2.0
    regions = np.ones((100, 100), dtype=np.int32)
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:70, 40:70] = True
    scene_graph = {
        "objects": [{"id": "obj_cup", "label": "cup", "mask_centroid_2d": [55, 55]}],
        "regions": {"regions": [{"region_index": 1, "semantic_label": "tabletop"}]},
    }
    contract = MotionContract(
        contract_id="take_2",
        actor=ActorSpec(actor_text="soda can", actor_source="uploaded_asset", asset_ref="can.png"),
        action_text="rolls across the table and bumps the cup",
        user_geometry=UserGeometry(mode="start_end", points=[(10, 50), (90, 50)]),
    )
    grounded = adapt_motion_contract_to_scene(
        contract,
        scene_graph=scene_graph,
        metric_depth_m=depth,
        region_label_map=regions,
        object_masks={"obj_cup": mask},
        sample_count=8,
    )
    out = grounded.to_json()
    assert out["contract"]["user_geometry"]["points"] == [[10.0, 50.0], [90.0, 50.0]]
    assert out["grounded_geometry"]["user_polyline_2d"] == [[10.0, 50.0], [90.0, 50.0]]
    assert len(out["traces"]["depth_trace_m"]) == 8
    assert "obj_cup" in sum(out["traces"]["occluder_ids"], [])
    assert out["rendering"]["asset_policy"]["no_hard_coded_actor_fallback"] is True
    assert out["report"]["scores"]["user_geometry_preservation"] == 1.0
