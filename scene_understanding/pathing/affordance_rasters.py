"""Rasterize per-object affordance evidence into dense (H, W) cost-map inputs.

Every raster is painted using the object's *real* segmentation mask
(``goal_anchors.object_footprint_mask`` -- real mask when present, bbox
rectangle only as a last resort), never a flat bbox fill, so a channel like
"blocker" reflects the actual obstacle silhouette rather than its rectangle.

Affordance scores per object/channel come from two sources, in priority
order:
  1. Precomputed ``{stem}_object_affordances.json`` / ``{stem}_mask_affordances.json``
     (produced by a richer affordance-export stage, when one has run and left
     that data on ``obj["affordances"]``) -- used as-is if present.
  2. Otherwise, an open-vocabulary fallback: each channel is defined by a
     natural-language concept pair and the object's own label/caption text is
     zero-shot classified against it (see ``ground_plane.zero_shot_classify_text``),
     so there is no fixed closed vocabulary of "affordance types" -- any
     object description the labelling stages produced can still populate
     every channel.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scene_understanding.pathing.goal_anchors import object_footprint_mask
from scene_understanding.pathing.ground_plane import zero_shot_classify_text

RASTER_CHANNELS: Tuple[str, ...] = (
    "support_surface",
    "blocker",
    "contact_target",
    "occlusion_edge",
    "portal",
    "open_air",
    "interior_motion",
    "uncertainty",
    "depth_discontinuity",
)

# (channel, positive concepts, negative concepts) for the open-vocab fallback.
# ``support_surface`` intentionally mirrors ground_plane's walkable-surface
# concepts so the two signals agree by construction where they overlap.
_CHANNEL_CONCEPTS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "support_surface": (
        ("a walkable floor, ground, or platform surface", "stairs, a ramp, or a walkable surface"),
        ("a wall or vertical obstacle", "an object you cannot stand on"),
    ),
    "blocker": (
        ("a solid obstacle blocking movement", "furniture, a wall, or a large solid object"),
        ("open floor or empty space", "something you can walk through or around easily"),
    ),
    "contact_target": (
        ("something to sit on, hold, or interact with", "a handle, seat, or interactive object"),
        ("background scenery not meant to be touched", "empty space"),
    ),
    "occlusion_edge": (
        ("an object something could hide behind or peek around", "a corner, doorway edge, or tall obstacle"),
        ("an open area with nothing to hide behind", "flat open ground"),
    ),
    "portal": (
        ("a doorway, gate, or passage to walk through", "an opening, archway, or entrance"),
        ("a solid wall with no opening", "a closed surface"),
    ),
    "open_air": (
        ("open air, sky, or empty space above the ground", "a place to fly, float, or hover"),
        ("solid ground or a solid object", "an enclosed interior space"),
    ),
    "interior_motion": (
        ("a hollow interior space something could move inside or through", "a tunnel, pipe, or container interior"),
        ("a solid exterior surface", "the outside of an object"),
    ),
}


def _paint_mask_max(raster: np.ndarray, mask: np.ndarray, value: float) -> None:
    if value <= 0.0:
        return
    raster[mask] = np.maximum(raster[mask], value)


def _object_text(obj: Dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(k, "")).strip()
        for k in ("canonical_name", "label", "caption", "semantic_label")
        if obj.get(k)
    ).strip()


def _precomputed_channel_scores(obj: Dict[str, Any]) -> Optional[Dict[str, float]]:
    affordances = obj.get("affordances")
    if not isinstance(affordances, dict):
        return None
    roles = affordances.get("roles") or {}
    actions = affordances.get("actions") or {}
    scores: Dict[str, float] = {}
    scores["support_surface"] = max(float(roles.get("support", 0.0)), float(actions.get("walk", 0.0)), float(actions.get("climb", 0.0)))
    scores["blocker"] = max(float(roles.get("hard_obstacle", 0.0)), 0.55 * float(roles.get("soft_obstacle", 0.0)))
    scores["contact_target"] = max(
        float(roles.get("interaction_target", 0.0)), float(actions.get("hold", 0.0)),
        float(actions.get("touch", 0.0)), float(actions.get("sit", 0.0)),
    )
    scores["occlusion_edge"] = max(float(roles.get("occluder", 0.0)), float(actions.get("peek", 0.0)), float(actions.get("hide", 0.0)))
    scores["portal"] = max(float(roles.get("portal", 0.0)), float(actions.get("enter", 0.0)), float(actions.get("exit", 0.0)))
    scores["open_air"] = max(float(roles.get("sky_open_air", 0.0)), float(actions.get("fly", 0.0)), float(actions.get("hover", 0.0)))
    scores["interior_motion"] = max(float(roles.get("interior", 0.0)), float(actions.get("inside", 0.0)), float(actions.get("float", 0.0)))
    scores["uncertainty"] = 0.85 if roles.get("contradictions") else (0.55 if roles.get("uncertainty") else 0.0)
    return scores if any(v > 0 for v in scores.values()) else None


def _openvocab_channel_scores(obj: Dict[str, Any], device: str) -> Dict[str, float]:
    text = _object_text(obj)
    scores: Dict[str, float] = {c: 0.0 for c in RASTER_CHANNELS}
    if not text:
        scores["uncertainty"] = 0.55
        return scores
    hits = 0
    for channel, (pos, neg) in _CHANNEL_CONCEPTS.items():
        verdict = zero_shot_classify_text(text, pos, neg, device=device)
        if verdict is None:
            continue
        hits += 1
        scores[channel] = 0.6 if verdict else 0.0
    scores["uncertainty"] = 0.0 if hits else 0.55
    return scores


def build_affordance_rasters(
    objects: Sequence[Dict[str, Any]],
    support_mask: Optional[np.ndarray],
    obstacle_mask: Optional[np.ndarray],
    metric_depth_m: Optional[np.ndarray],
    shape: Tuple[int, int],
    device: str = "cpu",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build the 9 (H, W) float32 rasters described in ``RASTER_CHANNELS``."""
    h, w = shape
    rasters = {c: np.zeros((h, w), dtype=np.float32) for c in RASTER_CHANNELS}

    if support_mask is not None:
        _paint_mask_max(rasters["support_surface"], np.asarray(support_mask, dtype=bool), 0.65)
    if obstacle_mask is not None:
        _paint_mask_max(rasters["blocker"], np.asarray(obstacle_mask, dtype=bool), 0.55)

    if metric_depth_m is not None:
        dm = np.asarray(metric_depth_m, dtype=np.float32)
        finite = np.isfinite(dm)
        dm_f = np.where(finite, dm, 0.0)
        gx = np.abs(np.diff(dm_f, axis=1, prepend=dm_f[:, :1]))
        gy = np.abs(np.diff(dm_f, axis=0, prepend=dm_f[:1, :]))
        grad = gx + gy
        p95 = float(np.percentile(grad[finite], 95)) if finite.any() else 0.0
        disc = np.clip(grad / p95, 0.0, 1.0) if p95 > 1e-6 else np.zeros_like(grad)
        rasters["depth_discontinuity"] = disc.astype(np.float32)
        rasters["occlusion_edge"] = np.maximum(rasters["occlusion_edge"], disc * 0.45)

    painted = 0
    for obj in objects:
        mask = object_footprint_mask(obj, h, w)
        if mask is None or not mask.any():
            continue
        scores = _precomputed_channel_scores(obj)
        if scores is None:
            scores = _openvocab_channel_scores(obj, device=device)
        for channel, value in scores.items():
            _paint_mask_max(rasters[channel], mask, value)
        painted += 1

    meta = {
        "channels": list(RASTER_CHANNELS),
        "objects_painted": painted,
        "objects_total": len(objects),
        "raster_stats": {
            c: {"min": float(r.min()), "max": float(r.max()), "nonzero_fraction": float((r > 0).mean())}
            for c, r in rasters.items()
        },
    }
    return rasters, meta


def save_affordance_rasters(rasters: Dict[str, np.ndarray], meta: Dict[str, Any], out_path: str) -> None:
    """``np.savez_compressed`` + a small manifest, mirroring the reference layout."""
    import json
    from pathlib import Path

    np.savez_compressed(out_path, **rasters)
    manifest_path = Path(out_path).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
