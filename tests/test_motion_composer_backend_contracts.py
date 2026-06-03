from scene_understanding.action_contracts import motion_contract_from_json


def test_start_end_path_contract_from_ui_is_backend_compatible():
    contract = motion_contract_from_json(
        {
            "contract_id": "take_start_end_path",
            "actor": {"actor_text": "uploaded mascot", "actor_source": "uploaded_asset"},
            "action_text": "follow my curve from start to end",
            "uploaded_animation_ref": "browser_uploaded_animation",
            "uploaded_animation": {
                "name": "walk_cycle.asset",
                "type": "application/octet-stream",
                "size_bytes": 1234,
                "retargeting_policy": "preserve_timing_and_style_then_ground_to_scene",
            },
            "user_geometry": {
                "mode": "start_end_path",
                "start_point": [10, 20],
                "drawn_path_2d": [[20, 25], [30, 35]],
                "end_point": [40, 50],
                "points": [[10, 20], [20, 25], [30, 35], [40, 50]],
                "source": "user_start_end_plus_drawn_path",
            },
        }
    )

    assert contract.user_geometry.mode == "start_end_path"
    assert contract.user_geometry.points[0] == (10.0, 20.0)
    assert contract.user_geometry.points[-1] == (40.0, 50.0)
    assert contract.user_geometry.metadata["start_point"] == [10, 20]
    assert contract.user_geometry.metadata["drawn_path_2d"] == [[20, 25], [30, 35]]
    assert contract.uploaded_animation_ref == "browser_uploaded_animation"
    assert contract.metadata["uploaded_animation"]["name"] == "walk_cycle.asset"
