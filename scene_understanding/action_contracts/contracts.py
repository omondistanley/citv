"""Data contracts for user-authored, scene-adaptive motion.

User input is authoritative; scene evidence is additive. A user-authored path,
tap point, uploaded animation, actor description, or interaction requirement must
not be silently replaced by a model-generated path. Adapters may add grounded
traces, warnings, preview corrections, and alternatives, but they must preserve
raw user geometry and report changes explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

Point2D = Tuple[float, float]
GeometryMode = Literal["point", "start_end", "start_end_path", "polyline", "region"]
ManifoldType = Literal[
    "centerline_path", "ribbon_path", "blob_path", "volume_path",
    "contour_path", "interior_path", "portal_path", "occlusion_pulse",
    "contact_patch", "effect_field",
]
ContractStatus = Literal[
    "accepted", "accepted_with_warnings", "low_confidence", "rejected", "accepted_surreal",
]


@dataclass(frozen=True)
class UserGeometry:
    """Immutable geometry created by the user.

    ``mode='start_end_path'`` means the UI captured explicit start/end anchors
    plus a drawn route. The fused ``points`` list is used for sampling, while raw
    start/end/path parts are preserved in ``metadata``.
    """

    mode: GeometryMode
    points: List[Point2D]
    source: str = "user"
    corridor_radius_px: float = 28.0
    closed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pts = [_coerce_point2d(p) for p in self.points]
        object.__setattr__(self, "points", pts)
        min_pts = {"point": 1, "start_end": 2, "start_end_path": 2, "polyline": 2, "region": 3}[self.mode]
        if len(pts) < min_pts:
            raise ValueError(f"{self.mode} geometry requires at least {min_pts} point(s)")
        if self.corridor_radius_px < 0:
            raise ValueError("corridor_radius_px must be non-negative")

    @property
    def start_point(self) -> Point2D:
        return self.points[0]

    @property
    def end_point(self) -> Point2D:
        return self.points[-1]

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class ActorSpec:
    """Open-vocabulary actor description."""

    actor_text: str
    actor_source: Literal["text", "uploaded_asset", "scene_object", "generated_asset", "drawn_asset"] = "text"
    asset_ref: Optional[str] = None
    scene_object_id: Optional[str] = None
    visual_style: Literal["photorealistic", "drawn", "stylized", "source_preserving"] = "source_preserving"
    physical_profile: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = (self.actor_text or "").strip()
        if not text and not self.asset_ref and not self.scene_object_id:
            raise ValueError("ActorSpec needs actor_text, asset_ref, or scene_object_id")
        object.__setattr__(self, "actor_text", text)

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class AdaptationPolicy:
    """Controls how scene evidence may adapt user input."""

    preserve_user_geometry: bool = True
    allow_path_bending: bool = True
    allow_endpoint_snap: bool = False
    allow_surreal_motion: bool = True
    max_path_deviation_px: float = 36.0
    obstacle_clearance_px: float = 8.0
    required_object_ids: List[str] = field(default_factory=list)
    avoid_object_ids: List[str] = field(default_factory=list)
    avoid_region_ids: List[str] = field(default_factory=list)
    must_render_behind_object_ids: List[str] = field(default_factory=list)
    protect_important_scene_regions: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class MotionContract:
    """User-authored action contract before scene grounding."""

    contract_id: str
    actor: ActorSpec
    action_text: str
    user_geometry: UserGeometry
    manifold_type: Optional[ManifoldType] = None
    duration_s: float = 4.0
    source: Literal["user_authored", "pipeline_suggested", "text_only"] = "user_authored"
    uploaded_animation_ref: Optional[str] = None
    policy: AdaptationPolicy = field(default_factory=AdaptationPolicy)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.contract_id or "").strip():
            raise ValueError("contract_id is required")
        text = (self.action_text or "").strip()
        if not text:
            raise ValueError("action_text is required")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        object.__setattr__(self, "action_text", text)

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class SceneAdaptationReport:
    """Transparent record of preserved, adapted, and warned constraints."""

    status: ContractStatus
    preserved: List[str] = field(default_factory=list)
    adapted: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rejected_reasons: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


@dataclass(frozen=True)
class GroundedMotionContract:
    """Scene-enriched contract consumed by renderers and export layers."""

    contract: MotionContract
    grounded_geometry: Dict[str, Any]
    traces: Dict[str, Any]
    rendering: Dict[str, Any]
    report: SceneAdaptationReport
    nearest_scene_entities: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: str = "citv.grounded_motion_contract.v1"

    def to_json(self) -> Dict[str, Any]:
        return _dataclass_json(self)


_MANIFOLD_KEYWORDS = (
    ("occlusion_pulse", ("peek", "hide", "duck", "emerge from behind", "behind")),
    ("portal_path", ("enter", "exit", "disappear", "appear", "through doorway", "portal")),
    ("contact_patch", ("hold", "touch", "grab", "sit", "lean", "land on", "place on", "bump", "push")),
    ("volume_path", ("fly", "glide", "hover", "float upward", "air")),
    ("blob_path", ("swim", "float", "drift", "smoke", "liquid", "water")),
    ("effect_field", ("shimmer", "glow", "ripple", "wobble", "dissolve", "reflection", "logo appears")),
    ("contour_path", ("circle", "orbit", "around", "inspect", "trace edge")),
    ("ribbon_path", ("walk", "run", "crawl", "slither", "roll", "slide", "drive")),
)


def infer_manifold_type(action_text: str, geometry: UserGeometry) -> ManifoldType:
    """Infer a default manifold without closing the action vocabulary."""

    text = (action_text or "").lower()
    for manifold, keys in _MANIFOLD_KEYWORDS:
        if any(k in text for k in keys):
            return manifold  # type: ignore[return-value]
    if geometry.mode == "point":
        return "effect_field"
    if geometry.mode == "region":
        return "blob_path"
    return "ribbon_path"


def motion_contract_from_json(payload: Dict[str, Any]) -> MotionContract:
    actor_payload = payload.get("actor") or {}
    geom_payload = payload.get("user_geometry") or payload.get("geometry") or {}
    policy_payload = payload.get("policy") or {}
    geometry = UserGeometry(
        mode=_normalize_geometry_mode(geom_payload.get("mode", "polyline")),
        points=geom_payload.get("points") or geom_payload.get("polyline_2d") or [],
        source=geom_payload.get("source", "user"),
        corridor_radius_px=float(geom_payload.get("corridor_radius_px", 28.0)),
        closed=bool(geom_payload.get("closed", False)),
        metadata={
            **dict(geom_payload.get("metadata") or {}),
            "start_point": geom_payload.get("start_point"),
            "end_point": geom_payload.get("end_point"),
            "drawn_path_2d": geom_payload.get("drawn_path_2d") or [],
            "region_polygon_2d": geom_payload.get("region_polygon_2d") or [],
        },
    )
    actor = ActorSpec(
        actor_text=str(actor_payload.get("actor_text") or actor_payload.get("text") or payload.get("actor_text") or ""),
        actor_source=actor_payload.get("actor_source", "text"),
        asset_ref=actor_payload.get("asset_ref"),
        scene_object_id=actor_payload.get("scene_object_id"),
        visual_style=actor_payload.get("visual_style", "source_preserving"),
        physical_profile=dict(actor_payload.get("physical_profile") or {}),
        metadata=dict(actor_payload.get("metadata") or {}),
    )
    policy = AdaptationPolicy(
        preserve_user_geometry=bool(policy_payload.get("preserve_user_geometry", True)),
        allow_path_bending=bool(policy_payload.get("allow_path_bending", True)),
        allow_endpoint_snap=bool(policy_payload.get("allow_endpoint_snap", False)),
        allow_surreal_motion=bool(policy_payload.get("allow_surreal_motion", True)),
        max_path_deviation_px=float(policy_payload.get("max_path_deviation_px", 36.0)),
        obstacle_clearance_px=float(policy_payload.get("obstacle_clearance_px", 8.0)),
        required_object_ids=list(policy_payload.get("required_object_ids") or []),
        avoid_object_ids=list(policy_payload.get("avoid_object_ids") or []),
        avoid_region_ids=list(policy_payload.get("avoid_region_ids") or []),
        must_render_behind_object_ids=list(policy_payload.get("must_render_behind_object_ids") or []),
        protect_important_scene_regions=bool(policy_payload.get("protect_important_scene_regions", True)),
        metadata=dict(policy_payload.get("metadata") or {}),
    )
    metadata = dict(payload.get("metadata") or {})
    if payload.get("uploaded_animation") is not None:
        metadata["uploaded_animation"] = payload.get("uploaded_animation")
    manifold = payload.get("manifold_type")
    contract = MotionContract(
        contract_id=str(payload.get("contract_id") or payload.get("id") or "user_contract"),
        actor=actor,
        action_text=str(payload.get("action_text") or payload.get("action") or ""),
        user_geometry=geometry,
        manifold_type=manifold,
        duration_s=float(payload.get("duration_s", 4.0)),
        source=payload.get("source", "user_authored"),
        uploaded_animation_ref=payload.get("uploaded_animation_ref"),
        policy=policy,
        metadata=metadata,
    )
    if contract.manifold_type is None:
        return MotionContract(
            contract_id=contract.contract_id,
            actor=contract.actor,
            action_text=contract.action_text,
            user_geometry=contract.user_geometry,
            manifold_type=infer_manifold_type(contract.action_text, contract.user_geometry),
            duration_s=contract.duration_s,
            source=contract.source,
            uploaded_animation_ref=contract.uploaded_animation_ref,
            policy=contract.policy,
            metadata=contract.metadata,
        )
    return contract


def _normalize_geometry_mode(mode: str) -> GeometryMode:
    allowed = {"point", "start_end", "start_end_path", "polyline", "region"}
    mode_s = str(mode or "polyline")
    if mode_s not in allowed:
        return "polyline"
    return mode_s  # type: ignore[return-value]


def _coerce_point2d(value: Sequence[float]) -> Point2D:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"Invalid 2D point: {value!r}")
    return (float(value[0]), float(value[1]))


def _dataclass_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: _dataclass_json(v) for k, v in asdict(value).items()}
    if isinstance(value, tuple):
        return [_dataclass_json(v) for v in value]
    if isinstance(value, list):
        return [_dataclass_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _dataclass_json(v) for k, v in value.items()}
    return value
