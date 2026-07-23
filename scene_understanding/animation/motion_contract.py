"""Open-ended, field-optional custom-input contract (Phase 3).

Not a fixed menu of input modes ("drawn path OR start/end OR upload actor").
Every field on ``MotionContract`` is independently optional; any combination
may be set. ``ground_motion_contract`` fills in whatever the user left unset
by pulling from every artifact the pipeline has already computed by this
point (relations, affordances, region semantics, mask-hierarchy occlusion
order, the precomputed path-hypothesis library, kinematic signatures) --
never a generic/hardcoded default when real evidence exists. Whatever the
user *did* set is preserved exactly and never overridden; grounding only
adds derived traces/enrichment alongside it, mirroring the same rule
``paths_export.py`` already applies to auto-generated paths.

Adding a genuinely new kind of custom input later is one optional field +
one grounding rule here, not a new branch of UI/backend logic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from scene_understanding.animation.motion_retargeting import build_retargeted_motion
from scene_understanding.pathing.goal_anchors import ground_intercept_goal_uv, mask_centroid_px, object_footprint_mask
from scene_understanding.pathing.ground_plane import classify_action_style_openvocab, classify_actor_attributes_openvocab
from scene_understanding.pathing.semantic_fmm import k_diverse_from_T, plan_path, speed_from_cost, time_of_arrival
from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages.paths_export import build_hypothesis, build_scene_cost_field

# Default actor body attributes when actor_text is unset -- a generic
# upright creature, matching the old binary classifier's humanoid fallback.
_DEFAULT_ACTOR_ATTRS: Dict[str, Any] = {"leg_count": 2, "has_wings": False, "has_tail": False, "elongated": False}

# Colors are an inherently finite/closed domain (unlike open-ended object
# categories), so a direct name lookup is used rather than a CLIP zero-shot
# call -- consistent with why classify_actor_attributes_openvocab handles
# shape/body attributes but not color.
_COLOR_NAME_TO_RGB: Dict[str, Tuple[float, float, float]] = {
    "red": (0.75, 0.08, 0.08), "orange": (0.85, 0.42, 0.05), "yellow": (0.85, 0.75, 0.05),
    "green": (0.08, 0.55, 0.12), "blue": (0.08, 0.25, 0.75), "purple": (0.45, 0.1, 0.6),
    "pink": (0.85, 0.4, 0.6), "brown": (0.4, 0.24, 0.1), "black": (0.03, 0.03, 0.03),
    "white": (0.9, 0.9, 0.9), "gray": (0.5, 0.5, 0.5), "grey": (0.5, 0.5, 0.5),
    "gold": (0.75, 0.6, 0.1), "silver": (0.7, 0.7, 0.75), "tan": (0.7, 0.55, 0.35),
}
_DEFAULT_ACTOR_COLOR: Tuple[float, float, float] = (0.55, 0.55, 0.55)  # neutral gray

# Default action style when action_text is unset -- unmodified geometric
# motion style (the path's own kinematic_signatures label drives everything).
_DEFAULT_ACTION_STYLE: Dict[str, Any] = {"is_static": False, "energy": "normal"}


@dataclass
class MotionContract:
    """Every field is optional and independent. ``None``/empty means
    "not specified by the user" -- grounding fills it in; it is never
    treated as an explicit request for a default value."""

    # Actor: which real object moves, or what to render if there's no
    # matching scene object (open-vocabulary text, or an uploaded image).
    actor_object_id: Optional[str] = None
    actor_text: Optional[str] = None
    actor_asset_rgba: Optional[np.ndarray] = None

    # Geometry: a hand-drawn path takes precedence over a custom start/end
    # pair, which takes precedence over picking existing detected objects.
    drawn_polyline_2d: Optional[List[Tuple[float, float]]] = None
    start_point_2d: Optional[Tuple[float, float]] = None
    end_point_2d: Optional[Tuple[float, float]] = None
    target_object_id: Optional[str] = None

    # Action (open-vocabulary; not wired into a fixed action taxonomy).
    action_text: Optional[str] = None

    # Constraints: avoid/behind are hard overrides; required is a soft cost
    # bias (see module docstring in ``_apply_required_bias`` for why this
    # isn't a guaranteed waypoint constraint).
    avoid_object_ids: List[str] = field(default_factory=list)
    required_object_ids: List[str] = field(default_factory=list)
    must_render_behind_object_ids: List[str] = field(default_factory=list)

    # Render knobs.
    duration_s: Optional[float] = None
    fps: Optional[int] = None

    # A local video file path (a real clip the user uploaded). Motion is
    # extracted via MediaPipe Pose (CPU, see motion_retargeting.py) and
    # resampled to this animation's own frame count -- the clip's motion
    # *shape and timing* carry over regardless of its own fps/length. Only
    # meaningful for Tier 1 (the procedural puppet has bob/lean/limb-swing
    # parameters to drive); Tier 2 is a photo cutout with no rig to retarget
    # onto. Preserved and reported even when no person is detected in the
    # clip, rather than silently ignored.
    uploaded_animation_ref: Optional[str] = None


@dataclass
class GroundedMotionContract:
    hypothesis: Dict[str, Any]
    actor_mask: Optional[np.ndarray]
    actor_asset_rgba: Optional[np.ndarray]
    archetype: str  # human-readable description for status text, e.g. "winged 2-legged creature"
    actor_attrs: Dict[str, Any]  # structured body attributes Tier 1 actually builds from -- see classify_actor_attributes_openvocab
    color_rgb: Tuple[float, float, float]
    action_style: Dict[str, Any]  # {"is_static": bool, "energy": "high"|"low"|"normal"} -- see classify_action_style_openvocab
    tier2_available: bool
    report: Dict[str, List[str]]
    retargeted_motion: Optional[Dict[str, List[float]]] = None


def _load_precomputed_hypotheses(ctx: PipelineContext) -> List[Dict[str, Any]]:
    hyp_path = ctx.path_exports.get("path_hypotheses_json")
    if not hyp_path or not Path(hyp_path).exists():
        return []
    return json.loads(Path(hyp_path).read_text()).get("hypotheses", [])


def _placeholder_object(obj_id: str) -> Dict[str, Any]:
    return {"id": obj_id, "label": obj_id}


def _apply_avoid_and_required(
    cost: np.ndarray, objects_by_id: Dict[str, Any], avoid_ids: List[str], required_ids: List[str], h: int, w: int,
) -> np.ndarray:
    """``avoid``: a hard cost wall (impassable) over the object's real mask.
    ``required``: a soft cost discount in a dilated region around the
    object -- a *preference*, not a guaranteed waypoint. Genuinely forcing a
    route through a specific point is a multi-goal routing problem (solve to
    the required point, then from there to the real goal); that's a
    reasonable follow-up but is more machinery than a soft bias for what's
    likely to be an occasional, non-critical constraint today."""
    out = cost.copy()
    for oid in avoid_ids:
        obj = objects_by_id.get(oid)
        if obj is None:
            continue
        m = object_footprint_mask(obj, h, w)
        if m is not None:
            out[m] = 1.0
    for oid in required_ids:
        obj = objects_by_id.get(oid)
        if obj is None:
            continue
        m = object_footprint_mask(obj, h, w)
        if m is None:
            continue
        dilated = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        out[dilated] = np.clip(out[dilated] - 0.25, 0.0, 1.0)
    return out


def _force_behind_objects(hypothesis: Dict[str, Any], objects_by_id: Dict[str, Any], behind_ids: List[str], h: int, w: int) -> None:
    """Mutates ``hypothesis`` in place: any path sample landing inside a
    ``must_render_behind_object_ids`` object's (dilated) mask gets its
    render_layer/alpha/occluder_ids forced to the same values Tier 2/Tier 1
    already know how to render for real computed occlusion -- this is a
    user-authored hard override, applied the same way a computed one would
    be, not a separate rendering code path."""
    if not behind_ids:
        return
    masks = []
    for oid in behind_ids:
        obj = objects_by_id.get(oid)
        m = object_footprint_mask(obj, h, w) if obj is not None else None
        if m is not None:
            masks.append((oid, cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0))
    if not masks:
        return
    visibility = hypothesis.get("visibility_profile", [])
    alpha_profile = hypothesis.get("alpha_profile", [])
    for i, sample in enumerate(visibility):
        u, v = int(sample.get("u", 0)), int(sample.get("v", 0))
        if not (0 <= v < h and 0 <= u < w):
            continue
        hits = [oid for oid, m in masks if m[v, u]]
        if hits:
            sample["render_layer"] = "behind_object"
            sample["visible_fraction"] = 0.15
            sample["occluder_ids"] = sorted(set(sample.get("occluder_ids", [])) | set(hits))
            if i < len(alpha_profile):
                alpha_profile[i] = 0.15
    occluder_ids_all = {oid for _, sample in enumerate(visibility) for oid in sample.get("occluder_ids", [])}
    hypothesis["occlusion_trace"]["occluder_ids"] = sorted(occluder_ids_all)


def _describe_obstacle_crossing(
    start_idx: int, end_idx: int, support_trace: List[Dict[str, Any]], objects_by_id: Dict[str, Any], h: int, w: int,
) -> Dict[str, Any]:
    """Identifies which real object's mask covers the midpoint of a
    contiguous 'blocking' run, for a human-readable warning."""
    mid = support_trace[(start_idx + end_idx) // 2]
    xi = max(0, min(w - 1, int(mid.get("u", 0))))
    yi = max(0, min(h - 1, int(mid.get("v", 0))))
    object_id, object_name = None, "an unidentified obstacle"
    for oid, obj in objects_by_id.items():
        mask = object_footprint_mask(obj, h, w)
        if mask is not None and mask[yi, xi]:
            object_id = oid
            object_name = obj.get("canonical_name") or obj.get("label", "object")
            break
    return {"start_idx": start_idx, "end_idx": end_idx, "object_id": object_id, "object_name": object_name}


def _flag_obstacle_crossings(
    hypothesis: Dict[str, Any], objects_by_id: Dict[str, Any], h: int, w: int,
) -> List[Dict[str, Any]]:
    """Groups contiguous 'blocking' support_trace points into crossing runs.
    ``support_trace``'s ``support_kind`` field (paths_export.py's
    ``_support_trace``) already indexes the real obstacle mask per point --
    it was previously write-only, recorded in the exported hypothesis JSON
    but never read back by anything. Read-only: only annotates the
    hypothesis, never mutates the path geometry itself, so "user intent is
    ground truth" holds regardless of what this finds."""
    support_trace = hypothesis.get("support_trace") or []
    crossings: List[Dict[str, Any]] = []
    run_start: Optional[int] = None
    for i, point in enumerate(support_trace):
        blocking = point.get("support_kind") == "blocking"
        if blocking and run_start is None:
            run_start = i
        elif not blocking and run_start is not None:
            crossings.append(_describe_obstacle_crossing(run_start, i - 1, support_trace, objects_by_id, h, w))
            run_start = None
    if run_start is not None:
        crossings.append(_describe_obstacle_crossing(run_start, len(support_trace) - 1, support_trace, objects_by_id, h, w))
    return crossings


def reroute_around_obstacles(
    drawn_polyline_2d: List[Tuple[float, float]],
    obstacle_crossings: List[Dict[str, Any]],
    speed_field: Optional[np.ndarray],
) -> List[Tuple[float, float]]:
    """Explicit, user-triggered adjustment (never automatic): locally
    reroutes only the flagged obstacle-crossing sub-segments of a drawn
    path around real obstacles, using the same FMM machinery Mode B's
    custom start/end search already uses (``plan_path("fmm", ...)`` against
    the scene's own precomputed speed field, ``ctx.extra["_paths_speed_field"]``).
    Every point outside a crossing run is left byte-for-byte as the user
    drew it -- "user intent is ground truth" stays the default behavior;
    this is an opt-in alternative the caller (the UI) can offer the user to
    accept or discard, never applied silently.

    A crossing run that has no route around it (``plan_path`` returns
    fewer than 2 points) is left untouched rather than guessed at -- the
    caller should still see it flagged in a re-grounded hypothesis.
    """
    if not obstacle_crossings or speed_field is None:
        return list(drawn_polyline_2d)

    points = list(drawn_polyline_2d)
    # Apply from the end backwards so splicing in a replacement sub-path of
    # a different length never shifts the indices of runs not yet applied.
    for crossing in sorted(obstacle_crossings, key=lambda c: c["start_idx"], reverse=True):
        start_idx, end_idx = crossing["start_idx"], crossing["end_idx"]
        entry_idx = max(0, start_idx - 1)
        exit_idx = min(len(points) - 1, end_idx + 1)
        if entry_idx >= exit_idx:
            continue
        entry = (int(round(points[entry_idx][0])), int(round(points[entry_idx][1])))
        exit_point = (int(round(points[exit_idx][0])), int(round(points[exit_idx][1])))
        rerouted = plan_path("fmm", speed_field, entry, exit_point)
        if len(rerouted) < 2:
            continue
        points[entry_idx : exit_idx + 1] = [(float(x), float(y)) for x, y in rerouted]
    return points


def _resolve_geometry(
    contract: MotionContract, ctx: PipelineContext, objects_by_id: Dict[str, Any], report: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    h, w = ctx.height, ctx.width
    cfg = None  # cost-field recompute path (only hit if Analyze didn't stash it) needs no pipeline config beyond defaults

    # Mode A: a hand-drawn path -- skips FMM entirely, preserved verbatim.
    if contract.drawn_polyline_2d and len(contract.drawn_polyline_2d) >= 2:
        report["preserved"].append("drawn_polyline_2d (skips path search entirely, per the plan's rule)")
        src = objects_by_id.get(contract.actor_object_id) or _placeholder_object("user_drawn_start")
        tgt = objects_by_id.get(contract.target_object_id) or _placeholder_object("user_drawn_end")
        hypothesis = build_hypothesis(
            "user_drawn_path", src, tgt, [(float(x), float(y)) for x, y in contract.drawn_polyline_2d],
            ctx, cfg, ctx.extra.get("_paths_obstacle_mask", np.zeros((h, w), bool)), ctx.extra.get("_paths_support_mask"),
        )
        if hypothesis is not None:
            crossings = _flag_obstacle_crossings(hypothesis, objects_by_id, h, w)
            hypothesis["obstacle_crossings"] = crossings
            for crossing in crossings:
                n_points = crossing["end_idx"] - crossing["start_idx"] + 1
                report["warnings"].append(
                    f"drawn path crosses a real obstacle ('{crossing['object_name']}') for {n_points} point(s) -- "
                    "path preserved exactly as drawn; use auto-adjust to reroute around it if desired"
                )
        return hypothesis

    # Mode B: a custom start/end point pair -- FMM search between exactly
    # these two points (not object anchors), with avoid/required applied.
    if contract.start_point_2d and contract.end_point_2d:
        report["preserved"].append("start_point_2d/end_point_2d (custom FMM search, not object anchors)")
        speed = ctx.extra.get("_paths_speed_field")
        if speed is None:
            speed = build_scene_cost_field(_FakePipelineShim(ctx), ctx)["speed"]
        obstacle_mask = ctx.extra.get("_paths_obstacle_mask", np.zeros((h, w), bool))
        # Approximate cost = 1 - speed (the inverse of how speed_from_cost derived
        # it) since only the final speed field is stashed in ctx.extra, not the
        # raw cost -- fine for a soft avoid/required adjustment, not bit-exact.
        adjusted_cost = _apply_avoid_and_required(1.0 - speed, objects_by_id, contract.avoid_object_ids, contract.required_object_ids, h, w)
        speed = speed_from_cost(adjusted_cost, None)
        start = (int(round(contract.start_point_2d[0])), int(round(contract.start_point_2d[1])))
        end = (int(round(contract.end_point_2d[0])), int(round(contract.end_point_2d[1])))
        raw_path = plan_path("fmm", speed, start, end)
        if len(raw_path) < 2:
            report["warnings"].append("no route found between the given start/end points -- they may be unreachable/inside an obstacle")
            return None
        src = objects_by_id.get(contract.actor_object_id) or _placeholder_object("user_start")
        tgt = objects_by_id.get(contract.target_object_id) or _placeholder_object("user_end")
        return build_hypothesis("user_start_end_path", src, tgt, raw_path, ctx, cfg, obstacle_mask, ctx.extra.get("_paths_support_mask"))

    # Modes C/D/E unify: filter the precomputed hypothesis library by
    # whichever of actor/target the user set; unset ends naturally widen the
    # filter (target unset -> best hypothesis for that actor; both unset ->
    # best hypothesis overall), which IS the grounding rule for this field.
    hypotheses = _load_precomputed_hypotheses(ctx)
    candidates = hypotheses
    if contract.actor_object_id:
        report["preserved"].append("actor_object_id")
        candidates = [h_ for h_ in candidates if h_.get("source_id") == contract.actor_object_id]
    else:
        report["grounded"].append("actor not specified -- considering all precomputed path sources")
    if contract.target_object_id:
        report["preserved"].append("target_object_id")
        candidates = [h_ for h_ in candidates if h_.get("target_id") == contract.target_object_id]
    else:
        report["grounded"].append("target not specified -- picking the highest-confidence match")
    if not candidates:
        report["warnings"].append("no precomputed path hypothesis matches the given actor/target -- Analyze may need to be re-run")
        return None
    return max(candidates, key=lambda h_: h_.get("overall_confidence", 0.0))


class _FakePipelineShim:
    """``build_scene_cost_field`` only reads ``pipeline.config``/``pipeline.depth_estimator.device``
    -- this avoids threading the real pipeline object through just for that."""

    def __init__(self, ctx: PipelineContext):
        self.config = None

        class _D:
            device = "cpu"

        self.depth_estimator = _D()


def _resolve_actor(
    contract: MotionContract, ctx: PipelineContext, objects_by_id: Dict[str, Any], hypothesis: Dict[str, Any], report: Dict[str, List[str]],
) -> Tuple[Optional[np.ndarray], Dict[str, Any], Tuple[float, float, float], str, bool]:
    h, w = ctx.height, ctx.width
    # CPU-only per Phase 4's always-on deployment target -- PipelineContext
    # doesn't carry the pipeline's device (that lives on the pipeline
    # object, not ctx), and every other open-vocab call site in this
    # codebase (ground_plane.py, goal_anchors.py) defaults to "cpu" too.
    device = "cpu"

    actor_attrs = _infer_actor_attrs(contract.actor_text, device)
    color_rgb = _extract_color_from_text(contract.actor_text)
    archetype = _describe_actor_attrs(actor_attrs)
    if contract.actor_text:
        report["preserved"].append("actor_text")

    # An uploaded asset always wins as the Tier 2 sprite *source* -- it's
    # independent of where the actor's mask/geometry comes from, so it's
    # checked here rather than as a mutually-exclusive alternative to
    # actor_object_id (a user can give both: "animate along actor_0's path,
    # but render this uploaded image instead of actor_0's own cutout").
    if contract.actor_asset_rgba is not None:
        report["preserved"].append("actor_asset_rgba (used as the Tier 2 sprite instead of a mask cutout)")

    # Mask/position source, independent of the sprite-source decision above.
    # Important: the "ground to the path's own source object" fallback only
    # applies when the user gave NO actor identity at all (no object, no
    # asset, no text) -- if they gave actor_text (an explicit open-vocabulary
    # identity override), silently reusing an unrelated real object's own
    # photo cutout as its mask would misrepresent that identity (e.g.
    # labelling a chair's cutout as "a red fox"), so that case is left
    # correctly mask-less for Tier 2 (Tier 1/3 render the described actor
    # instead) rather than grounded to the wrong pixels.
    if contract.actor_object_id and contract.actor_object_id in objects_by_id:
        report["preserved"].append("actor_object_id")
        mask = object_footprint_mask(objects_by_id[contract.actor_object_id], h, w)
    elif not contract.actor_text and contract.actor_asset_rgba is None:
        source_id = hypothesis.get("source_id")
        if source_id in objects_by_id:
            report["grounded"].append(f"actor not specified -- grounded to the path's own source object ({source_id})")
            mask = object_footprint_mask(objects_by_id[source_id], h, w)
        else:
            mask = None
    else:
        mask = None

    tier2_available = mask is not None or contract.actor_asset_rgba is not None
    if not tier2_available:
        if contract.actor_text:
            report["warnings"].append("open-vocabulary actor_text with no scene mask/asset -- Tier 2 unavailable, use Tier 1 or Tier 3")
        else:
            report["warnings"].append("no actor specified and none could be grounded from the path -- rendering will have no visible actor")
    return mask, actor_attrs, color_rgb, archetype, tier2_available


def _infer_actor_attrs(actor_text: Optional[str], device: str) -> Dict[str, Any]:
    if not actor_text:
        return dict(_DEFAULT_ACTOR_ATTRS)
    return classify_actor_attributes_openvocab(actor_text, device=device)


def _extract_color_from_text(actor_text: Optional[str]) -> Tuple[float, float, float]:
    if not actor_text:
        return _DEFAULT_ACTOR_COLOR
    text = actor_text.lower()
    for name, rgb in _COLOR_NAME_TO_RGB.items():
        if name in text:
            return rgb
    return _DEFAULT_ACTOR_COLOR


def _describe_actor_attrs(attrs: Dict[str, Any]) -> str:
    parts: List[str] = []
    if attrs.get("has_wings"):
        parts.append("winged")
    if attrs.get("elongated"):
        parts.append("elongated")
    parts.append(f"{attrs.get('leg_count', 2)}-legged")
    if attrs.get("has_tail"):
        parts.append("tailed")
    return " ".join(parts) + " creature"


def ground_motion_contract(contract: MotionContract, ctx: PipelineContext) -> Optional[GroundedMotionContract]:
    """The single grounding entrypoint: resolves geometry, actor, and
    constraints from whatever subset of ``contract`` is set, filling in the
    rest from the pipeline's own evidence. Returns ``None`` (with warnings in
    the report) if nothing usable could be resolved at all."""
    objects = ctx.extra.get("objects", [])
    objects_by_id = {o.get("id"): o for o in objects}
    report: Dict[str, List[str]] = {"preserved": [], "grounded": [], "warnings": []}

    hypothesis = _resolve_geometry(contract, ctx, objects_by_id, report)
    if hypothesis is None:
        return None

    actor_mask, actor_attrs, color_rgb, archetype, tier2_available = _resolve_actor(contract, ctx, objects_by_id, hypothesis, report)

    # action_text previously existed on the contract but was never consumed
    # by rendering at all -- "static"/"energetic"/"peeking" etc. had no
    # effect. Now independently modulates HOW the actor moves (Tier 1 only;
    # Tier 2 is a photo cutout with no bob/limb-swing rig to modulate).
    action_style = classify_action_style_openvocab(contract.action_text, device="cpu") if contract.action_text else dict(_DEFAULT_ACTION_STYLE)
    if contract.action_text:
        report["preserved"].append("action_text (modulates Tier 1 motion style: static/energy)")

    if contract.avoid_object_ids:
        report["preserved"].append("avoid_object_ids")
    if contract.required_object_ids:
        report["preserved"].append("required_object_ids (soft cost preference, not a guaranteed waypoint)")
    if contract.must_render_behind_object_ids:
        report["preserved"].append("must_render_behind_object_ids")
        _force_behind_objects(hypothesis, objects_by_id, contract.must_render_behind_object_ids, ctx.height, ctx.width)

    retargeted_motion: Optional[Dict[str, List[float]]] = None
    if contract.uploaded_animation_ref:
        fps = contract.fps or 24
        duration_s = contract.duration_s or 4.0
        n_frames = max(1, round(fps * duration_s))
        try:
            retargeted_motion = build_retargeted_motion(contract.uploaded_animation_ref, n_frames)
        except Exception as exc:  # pragma: no cover -- depends on the MediaPipe model being reachable/cached
            report["warnings"].append(f"uploaded_animation_ref preserved but retargeting failed ({exc}) -- falling back to the canned motion style")
        else:
            if retargeted_motion is None:
                report["warnings"].append("uploaded_animation_ref preserved but no person was detected in the clip -- falling back to the canned motion style")
            else:
                report["preserved"].append("uploaded_animation_ref (motion retargeted via MediaPipe Pose, resampled to this render's frame count)")

    return GroundedMotionContract(
        hypothesis=hypothesis, actor_mask=actor_mask, actor_asset_rgba=contract.actor_asset_rgba,
        archetype=archetype, actor_attrs=actor_attrs, color_rgb=color_rgb, action_style=action_style,
        tier2_available=tier2_available, report=report, retargeted_motion=retargeted_motion,
    )
