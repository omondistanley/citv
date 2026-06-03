"""Scene-adaptive user motion contract helpers.

This package keeps user-authored actor/action/path intent as the source of
truth, then enriches it with scene evidence so UI/rendering layers can adapt
motion naturally without silently replacing the user's directions.
"""

from scene_understanding.action_contracts.contracts import (
    ActorSpec,
    AdaptationPolicy,
    GroundedMotionContract,
    MotionContract,
    SceneAdaptationReport,
    UserGeometry,
    infer_manifold_type,
    motion_contract_from_json,
)
from scene_understanding.action_contracts.json_extraction import (
    JsonExtractionError,
    extract_first_json_object,
    extract_first_json_value,
)
from scene_understanding.action_contracts.scene_adapter import adapt_motion_contract_to_scene

__all__ = [
    "ActorSpec",
    "AdaptationPolicy",
    "GroundedMotionContract",
    "MotionContract",
    "SceneAdaptationReport",
    "UserGeometry",
    "JsonExtractionError",
    "adapt_motion_contract_to_scene",
    "extract_first_json_object",
    "extract_first_json_value",
    "infer_manifold_type",
    "motion_contract_from_json",
]
