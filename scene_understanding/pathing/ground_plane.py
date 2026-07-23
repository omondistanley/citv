"""Ground/support-surface estimation from metric depth.

Two independent estimates are unioned into one support mask:
  1. A RANSAC-fit ground plane in camera space (works even when regions are
     mislabelled or missing).
  2. An open-vocabulary semantic mask: each region's own label/caption text
     (whatever the labelling stages produced — GDINO/Florence-2/RAM++ output,
     not a closed set) is zero-shot classified against a small set of
     natural-language "walkable surface" vs. "not walkable" concept prompts
     using the same CLIP text encoder already loaded elsewhere in the
     pipeline (see ``depth.py::SceneClassifier``). This generalizes to any
     phrasing the labelling stages produce instead of requiring a literal
     substring match against a fixed keyword list.

Both feed ``build_support_mask`` used by ``goal_anchors`` and the FMM cost
layers to know where an actor can actually stand.
"""
from __future__ import annotations

import random
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Anchor concepts for the zero-shot support/not-support classification below.
# These are natural-language *descriptions*, matched by embedding similarity
# to whatever text a region actually carries -- not substrings to search for.
_SUPPORT_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a walkable floor or ground surface a person can stand or walk on",
    "a road, sidewalk, or paved path",
    "stairs, steps, or a ramp",
    "a platform, deck, bridge, or tabletop surface to stand on",
    "grass, lawn, dirt, or natural terrain",
    "a rug, carpet, or mat on the floor",
)
_NON_SUPPORT_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a wall or vertical surface",
    "furniture or an object that cannot be walked on",
    "the sky, ceiling, or open air",
    "a person, animal, or vehicle",
    "water, glass, or an unstable surface",
)

_clip_lock = threading.Lock()
_clip_state: Dict[str, Any] = {"model": None, "processor": None, "device": "cpu"}


def _load_clip(device: str = "cpu") -> bool:
    if _clip_state["model"] is not None:
        return True
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(device)
        model.eval()
        _clip_state.update({"model": model, "processor": processor, "device": device, "torch": torch})
        return True
    except Exception as exc:  # pragma: no cover - offline/broken install fallback
        print(f"  [ground_plane] CLIP load failed ({exc}); open-vocab support classification disabled.")
        return False


def unload_clip() -> None:
    """Free the CLIP model (call once the pipeline no longer needs region classification)."""
    with _clip_lock:
        model = _clip_state.get("model")
        if model is not None:
            del model
        _clip_state.update({"model": None, "processor": None})


def _text_embeddings(texts: List[str]) -> Optional[Any]:
    processor = _clip_state["processor"]
    model = _clip_state["model"]
    torch = _clip_state["torch"]
    with torch.no_grad():
        inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(_clip_state["device"]) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feats


@lru_cache(maxsize=4096)
def zero_shot_classify_text(
    text: str,
    positive_prompts: Tuple[str, ...],
    negative_prompts: Tuple[str, ...],
    device: str = "cpu",
    margin: float = 0.02,
    min_confidence: float = 0.15,
) -> Optional[bool]:
    """Generic open-vocabulary zero-shot classifier: does ``text`` read closer to
    ``positive_prompts`` or ``negative_prompts``?

    Both prompt sets are natural-language concept descriptions compared to
    ``text`` by CLIP text-embedding cosine similarity -- there is no fixed
    keyword/label list, so any phrasing the upstream labelling stages
    produce (including synonyms/novel surface or structure descriptions)
    is handled the same way. Returns ``None`` when CLIP is unavailable, the
    text is empty, or the match to *either* concept set is too weak to
    trust (caller should fall back to a non-text signal instead of
    guessing).

    ``@lru_cache``: measured on a real 16-object scene, ``goal_anchors.py``'s
    ``vertical_structure_heuristic`` (called once per object per *candidate
    pair* while building path hypotheses, up to C(16,2)=120 pairs x 2 calls
    = 240 calls) was re-classifying the exact same 16 objects' text over and
    over -- this collapses that to 16 real CLIP calls, the rest are cache
    hits. All arguments are hashable (str/tuple/float), so this is safe with
    no behavior change, just no repeated work for an identical query.
    """
    text = (text or "").strip()
    if not text:
        return None
    with _clip_lock:
        if not _load_clip(device):
            return None
        torch = _clip_state["torch"]
        prompts = list(positive_prompts) + list(negative_prompts)
        concept_feats = _text_embeddings(prompts)
        query_feat = _text_embeddings([text])
        sims = (query_feat @ concept_feats.T).squeeze(0)
        n_pos = len(positive_prompts)
        pos_score = float(torch.max(sims[:n_pos]))
        neg_score = float(torch.max(sims[n_pos:]))
    if max(pos_score, neg_score) < min_confidence:
        return None
    return pos_score > (neg_score + margin)


def classify_support_text_openvocab(text: str, device: str = "cpu", margin: float = 0.02) -> Optional[bool]:
    """Zero-shot: does ``text`` describe a walkable surface?"""
    return zero_shot_classify_text(
        text, _SUPPORT_CONCEPT_PROMPTS, _NON_SUPPORT_CONCEPT_PROMPTS, device=device, margin=margin,
    )


# Terrain-type concepts for animation motion selection (Phase 2.5): a path
# segment's own region_type/semantic_label text (same field _support_trace
# already produces) is classified the same open-vocab way as support/not-
# support above, rather than a fixed keyword list, so any phrasing the
# labelling stages produce still generalizes.
_WATER_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a body of water such as a pool, lake, river, or ocean surface",
    "a wet, submerged, or floating surface",
)
_SOLID_GROUND_CONCEPT_PROMPTS: Tuple[str, ...] = ("a solid dry ground or floor surface",)
_OPEN_AIR_CONCEPT_PROMPTS: Tuple[str, ...] = ("open air, sky, or empty space with no ground surface below",)
_GROUNDED_CONCEPT_PROMPTS: Tuple[str, ...] = ("a solid supporting surface or structure",)


def classify_terrain_openvocab(text: str, device: str = "cpu") -> Optional[str]:
    """Zero-shot: what kind of terrain does ``text`` describe for animation
    motion selection? Returns ``"swimmable"`` | ``"open_air"`` | ``"walkable"``,
    or ``None`` when CLIP is unavailable or the classification is ambiguous
    (callers should leave the existing Z-profile motion label untouched in
    that case, never guess)."""
    is_water = zero_shot_classify_text(text, _WATER_CONCEPT_PROMPTS, _SOLID_GROUND_CONCEPT_PROMPTS, device=device)
    if is_water:
        return "swimmable"
    is_open_air = zero_shot_classify_text(text, _OPEN_AIR_CONCEPT_PROMPTS, _GROUNDED_CONCEPT_PROMPTS, device=device)
    if is_open_air:
        return "open_air"
    if is_water is False and is_open_air is False:
        return "walkable"
    return None


# Actor-appearance attribute concepts for Tier 1 shape composition (Phase
# 2.6): open-vocab actor_text (e.g. "a red bird", "a snake", "a lion") gets
# classified against independent, COMPOSABLE body attributes rather than a
# fixed archetype enum. A fixed set of named categories forces every
# description into the closest of a handful of predefined buckets (a bird
# request was landing on "quadruped" with no flying-creature category at
# all); attributes compose freely instead, so a genuinely novel creature
# description -- a snake, a fish, a six-legged bug description -- still gets
# a plausible silhouette built from the attributes it implies, instead of
# being forced into the nearest wrong bucket. Each attribute is its own
# independent zero_shot_classify_text call and degrades to a safe default
# when CLIP can't decide, rather than guessing.
#
# Prompt-engineering note: CLIP's TEXT-text similarity (this is text vs
# text, not image vs text) is measurably unreliable for these attributes
# when the concept prompts are short abstract descriptions ("a creature
# with visible legs") -- empirically verified against the real installed
# CLIP model, e.g. "a red fox" and "a dog" both misclassified as legless
# and "a bird" misclassified as wingless with the first version of these
# prompts. Concrete, varied named exemplars per side perform far better and
# were verified against a real fox/bird/dog/person/fish/snake/lion/eel/
# goose/bat test set before landing here -- keep this pattern (many named
# examples, not one abstract description) for any future attribute added.
_LEGGED_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a dog", "a cat", "a horse", "a fox", "a lion", "a bird", "a person",
    "a deer", "a rabbit", "a bear", "an animal that walks on legs",
)
_LEGLESS_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a snake", "a fish", "a worm", "an eel", "a slug",
    "an animal with no legs that slithers or swims",
)
_FOUR_LEGGED_CONCEPT_PROMPTS: Tuple[str, ...] = ("a dog", "a cat", "a horse", "a fox", "a lion", "a bear", "a four-legged animal")
_TWO_LEGGED_CONCEPT_PROMPTS: Tuple[str, ...] = ("a person", "a bird", "a human", "a two-legged upright creature")
_WINGED_CONCEPT_PROMPTS: Tuple[str, ...] = ("a bird", "an eagle", "a pigeon", "a goose", "a bat", "a butterfly", "a dragonfly", "a winged flying creature")
_WINGLESS_CONCEPT_PROMPTS: Tuple[str, ...] = ("a dog", "a cat", "a fox", "a lion", "a person", "a fish", "a snake", "a creature that cannot fly")
_TAILED_CONCEPT_PROMPTS: Tuple[str, ...] = ("a dog", "a cat", "a fox", "a lion", "a lizard", "a bird", "an animal with a visible tail")
_TAILLESS_CONCEPT_PROMPTS: Tuple[str, ...] = ("a person", "a human", "a frog", "an animal with no tail")
_ELONGATED_CONCEPT_PROMPTS: Tuple[str, ...] = ("a snake", "an eel", "a worm", "a long slender serpentine creature")
_COMPACT_CONCEPT_PROMPTS: Tuple[str, ...] = ("a dog", "a person", "a bird", "a compact rounded creature")


def classify_actor_attributes_openvocab(text: str, device: str = "cpu") -> Dict[str, Any]:
    """Zero-shot, composable creature attributes inferred from open-vocab
    actor_text. Returns ``{"leg_count": 0|2|4, "has_wings": bool,
    "has_tail": bool, "elongated": bool}``. Every attribute independently
    falls back to a safe default (2 legs, no wings, no tail, not elongated
    -- a generic upright creature) when CLIP is unavailable or ambiguous on
    that specific attribute, rather than guessing."""
    has_legs = zero_shot_classify_text(text, _LEGGED_CONCEPT_PROMPTS, _LEGLESS_CONCEPT_PROMPTS, device=device)
    if has_legs is False:
        leg_count = 0
    else:
        four_legged = zero_shot_classify_text(text, _FOUR_LEGGED_CONCEPT_PROMPTS, _TWO_LEGGED_CONCEPT_PROMPTS, device=device)
        leg_count = 4 if four_legged else 2  # None (ambiguous) or explicit False both default to 2 (upright)

    winged = zero_shot_classify_text(text, _WINGED_CONCEPT_PROMPTS, _WINGLESS_CONCEPT_PROMPTS, device=device)
    tailed = zero_shot_classify_text(text, _TAILED_CONCEPT_PROMPTS, _TAILLESS_CONCEPT_PROMPTS, device=device)
    elongated = zero_shot_classify_text(text, _ELONGATED_CONCEPT_PROMPTS, _COMPACT_CONCEPT_PROMPTS, device=device)

    return {
        "leg_count": leg_count,
        "has_wings": bool(winged),  # None (ambiguous/CLIP unavailable) -> False, the safe default
        "has_tail": bool(tailed) if tailed is not None else (leg_count == 4),  # quadrupeds usually have visible tails
        "elongated": bool(elongated),
    }


# Action-style concepts (Phase 2.6): open-vocab action_text (e.g. "stays
# still and peeks out", "hops around excitedly") independently modulates HOW
# the actor moves along the path, on top of WHERE the path's own geometry
# says it should be -- action_text previously existed as a MotionContract
# field but was never consumed anywhere in rendering, so a user's explicit
# "static"/"energetic"/etc. description had no effect at all. Not a fixed
# motion-name enum -- two independent, composable style dimensions instead,
# so a genuinely novel description still lands on a plausible style instead
# of being forced into the nearest of a few named presets.
_STATIC_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a still, static, motionless, frozen, or stationary pose",
)
_ACTIVE_CONCEPT_PROMPTS: Tuple[str, ...] = ("an actively moving action",)
_HIGH_ENERGY_CONCEPT_PROMPTS: Tuple[str, ...] = ("an energetic, fast, vigorous, or excited action",)
_LOW_ENERGY_CONCEPT_PROMPTS: Tuple[str, ...] = ("a calm, slow, cautious, or subtle action such as peeking or creeping",)


def classify_action_style_openvocab(text: str, device: str = "cpu") -> Dict[str, Any]:
    """Zero-shot action-style modulation from open-vocab action_text.
    Returns ``{"is_static": bool, "energy": "high"|"low"|"normal"}``.
    ``is_static`` freezes the actor's bob/limb-swing regardless of what the
    path geometry's own motion label would otherwise drive; ``energy``
    scales bob/limb-swing amplitude up or down. Degrades to "normal"
    (unmodified geometric motion style) when CLIP is unavailable or
    ambiguous, never guesses a style."""
    is_static = zero_shot_classify_text(text, _STATIC_CONCEPT_PROMPTS, _ACTIVE_CONCEPT_PROMPTS, device=device)
    if is_static:
        return {"is_static": True, "energy": "low"}
    is_high_energy = zero_shot_classify_text(text, _HIGH_ENERGY_CONCEPT_PROMPTS, _LOW_ENERGY_CONCEPT_PROMPTS, device=device)
    if is_high_energy is True:
        energy = "high"
    elif is_high_energy is False:
        energy = "low"
    else:
        energy = "normal"
    return {"is_static": False, "energy": energy}


def _unproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def fit_ground_plane(
    depth: np.ndarray,
    intrinsics: Dict[str, float],
    object_mask: Optional[np.ndarray] = None,
    sample_band: Tuple[float, float] = (0.6, 1.0),
    inlier_thresh_m: float = 0.10,
    max_iter: int = 80,
    min_samples: int = 200,
    max_points: int = 4000,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """RANSAC-fit a plane ``normal . X + d = 0`` in camera coordinates.

    Samples only the lower ``sample_band`` fraction of image rows (where the
    ground plane is most likely to be visible) and excludes pixels flagged
    as belonging to a detected object.
    """
    dm = np.asarray(depth, dtype=np.float32)
    h, w = dm.shape[:2]
    fx = float(intrinsics.get("fx", w))
    fy = float(intrinsics.get("fy", h))
    cx = float(intrinsics.get("cx", w / 2.0))
    cy = float(intrinsics.get("cy", h / 2.0))

    y0 = int(h * sample_band[0])
    y1 = int(h * sample_band[1])
    band = np.zeros((h, w), dtype=bool)
    band[y0:y1, :] = True
    finite = np.isfinite(dm) & (dm > 1e-6)
    valid = band & finite
    if object_mask is not None:
        valid &= ~np.asarray(object_mask, dtype=bool)

    ys, xs = np.nonzero(valid)
    if xs.size < min_samples:
        return None
    if xs.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xs.size, size=max_points, replace=False)
        ys, xs = ys[idx], xs[idx]

    z = dm[ys, xs].astype(np.float64)
    pts = _unproject(xs.astype(np.float64), ys.astype(np.float64), z, fx, fy, cx, cy)

    rng = random.Random(seed)
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    n = pts.shape[0]
    need = max(50, n // 20)
    for _ in range(max_iter):
        i0, i1, i2 = rng.sample(range(n), 3)
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(normal @ p0)
        resid = np.abs(pts @ normal + d)
        inliers = resid < inlier_thresh_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < need:
        return None

    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(1e-9, float(np.linalg.norm(normal)))
    if normal[1] > 0:
        normal = -normal
    d = -float(normal @ centroid)

    inlier_uv = np.stack([xs[best_inliers], ys[best_inliers]], axis=-1)
    return {
        "normal": normal.astype(np.float64).tolist(),
        "d": d,
        "inlier_uv": inlier_uv,
        "inlier_count": int(best_inliers.sum()),
        "sample_count": int(n),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
    }


def project_inlier_mask(plane: Dict[str, Any], depth: np.ndarray, inlier_thresh_m: float = 0.10) -> np.ndarray:
    """Densify a fitted plane to a full-image support mask."""
    dm = np.asarray(depth, dtype=np.float32)
    h, w = dm.shape[:2]
    fx, fy, cx, cy = plane["fx"], plane["fy"], plane["cx"], plane["cy"]
    normal = np.asarray(plane["normal"], dtype=np.float64)
    d = float(plane["d"])
    finite = np.isfinite(dm) & (dm > 1e-6)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    pts = _unproject(us.astype(np.float64), vs.astype(np.float64), dm.astype(np.float64), fx, fy, cx, cy)
    resid = np.abs(pts @ normal + d)
    mask = finite & (resid < inlier_thresh_m)
    return mask


def build_semantic_support_mask(
    region_label_map: Optional[np.ndarray],
    regions_meta: Optional[Dict[str, Any]],
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Open-vocabulary mask of regions whose own text reads as a walkable surface."""
    if region_label_map is None or not regions_meta:
        return None
    lm = np.asarray(region_label_map)
    mask = np.zeros(lm.shape[:2], dtype=bool)
    regions = regions_meta.get("regions") if isinstance(regions_meta, dict) else regions_meta
    if not regions:
        return None
    any_classified = False
    for region in regions:
        region_id = region.get("region_id", region.get("id"))
        if region_id is None:
            continue
        text = " ".join(
            str(region.get(k, "")).strip()
            for k in ("region_type", "semantic_label", "layer_type", "label", "caption")
            if region.get(k)
        ).strip()
        verdict = classify_support_text_openvocab(text, device=device)
        if verdict is None:
            continue
        any_classified = True
        if verdict:
            mask |= (lm == region_id)
    return mask if any_classified and mask.any() else None


def build_support_mask(
    depth: np.ndarray,
    intrinsics: Dict[str, float],
    object_mask: Optional[np.ndarray] = None,
    region_label_map: Optional[np.ndarray] = None,
    regions_meta: Optional[Dict[str, Any]] = None,
    inlier_thresh_m: float = 0.10,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Union of open-vocabulary region support and RANSAC ground-plane support."""
    sem_mask = build_semantic_support_mask(region_label_map, regions_meta, device=device)
    plane = fit_ground_plane(depth, intrinsics, object_mask=object_mask, inlier_thresh_m=inlier_thresh_m)
    plane_mask = project_inlier_mask(plane, depth, inlier_thresh_m=inlier_thresh_m) if plane else None

    h, w = np.asarray(depth).shape[:2]
    if sem_mask is None and plane_mask is None:
        combined = np.zeros((h, w), dtype=bool)
    elif sem_mask is None:
        combined = plane_mask
    elif plane_mask is None:
        combined = sem_mask
    else:
        combined = sem_mask | plane_mask

    info = {
        "has_semantic_support": sem_mask is not None,
        "has_plane_support": plane is not None,
        "plane_normal": plane["normal"] if plane else None,
        "plane_d": plane.get("d") if plane else None,
        "plane_inlier_count": plane.get("inlier_count") if plane else 0,
        "support_ratio": float(combined.mean()) if combined.size else 0.0,
    }
    return combined, info


def snap_uv_down_to_support(uv: Tuple[int, int], support_mask: np.ndarray, max_search_px: int = 200) -> Tuple[int, int]:
    """Walk straight down from ``uv`` until the support mask is True."""
    x, y = int(round(uv[0])), int(round(uv[1]))
    h, w = support_mask.shape[:2]
    x = max(0, min(w - 1, x))
    y0 = max(0, min(h - 1, y))
    for dy in range(0, max_search_px + 1):
        yy = y0 + dy
        if yy >= h:
            break
        if support_mask[yy, x]:
            return x, yy
    return x, h - 1
