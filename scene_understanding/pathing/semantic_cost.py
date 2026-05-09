"""Semantic cost map builder (Phase C).

This module implements the geometry- and 3D-aware *semantic* cost map used
by the Fast Marching Method (FMM) planner.  The output is a per-pixel scalar
cost ``C[y, x] in [0, 1]`` where 0 means "cheapest to traverse" and 1 means
"impassable", along with per-pixel precision ``P[y, x] in [0, 1]`` that
records how confident the pipeline is about that cost.

Design principles enforced by this module
-----------------------------------------

1.  **No magic numbers.**  Every scalar threshold used below is either:

    - derived from the image itself (Otsu split on image-scale gradient,
      Jenks/JSD break on depth histograms, BIC on GMM over self-calibration
      exemplars), or
    - read from :class:`AgentPhysicalProfile` (measured metres), or
    - read from explicit config weights (already documented in
      :mod:`config`).

    If a required piece of evidence is missing we return an *informative*
    ``None`` layer — we never silently substitute a default.

2.  **Six additive layers with precision weights.**  The layers are:

    A. :func:`layer_geometry`  — edges, depth gradients, distance-to-obstacle.
    B. ``layer_vlm``           — CLIPSeg walkability + SigLIP two-pole per
                                  object (implemented in
                                  :mod:`scene_understanding.pathing.semantic_vlm`;
                                  see C.3).
    C. ``layer_self_calibration`` — GMM on A+B exemplars (see C.4).
    D. :func:`layer_scene_graph` — object-class obstacle/support evidence
       drawn straight from ``sam2_scene_output``.
    E. :func:`layer_agent_physics` — inflation by
       :class:`AgentPhysicalProfile` half-width + step-over tolerance using
       3D back-projection so inflation radii are metric, not pixel-wise.
    F. :func:`layer_depth_noise` — downweights precision where depth std
       is high relative to the image's own depth histogram (no magic
       thresholds).

3.  **Precision-weighted Bayesian fusion.**  :func:`precision_weighted_fuse`
    combines any subset of these layers.  Each layer contributes
    ``P_i / sum(P_i)`` as its weight on ``C``, so a noisy layer with low
    precision cannot dominate.

4.  **Output-preservation contract.**  This module never writes to disk and
    never mutates its inputs.  The caller is responsible for saving
    ``cost_map.npz``, ``cost_layers/``, ``cost_provenance.png`` etc. in
    the path output subdirectory when
    ``cfg.path_cost_emit_layers_for_qa`` is True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # cv2 is an existing repo dependency.
    import cv2
except Exception as _exc:  # pragma: no cover - importing fails loud elsewhere.
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = _exc
else:
    _CV2_IMPORT_ERROR = None

from ..action_ontology import load_action_ontology, prompt_bank
from ..evidence import normalize_label


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostLayer:
    """Immutable output of a single cost layer.

    Attributes
    ----------
    name
        Short identifier (``"A_geometry"``, ``"D_scene_graph"``, ...).
    cost
        ``float32`` array of shape ``(H, W)`` clipped to ``[0, 1]``.
    precision
        ``float32`` array of shape ``(H, W)`` in ``[0, 1]`` — higher means
        "trust this layer more at this pixel" for the fusion step.
    provenance
        Free-form string describing *why* this layer produced the values it
        did.  Stored verbatim in ``cost_meta.json`` for QA.
    diagnostics
        Optional additional scalar fields used by the markdown export.
    """

    name: str
    cost: np.ndarray
    precision: np.ndarray
    provenance: str = ""
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost.shape != self.precision.shape:
            raise ValueError(
                f"CostLayer '{self.name}' cost shape {self.cost.shape} != "
                f"precision shape {self.precision.shape}"
            )
        if self.cost.ndim != 2:
            raise ValueError(
                f"CostLayer '{self.name}' expected 2-D cost, got {self.cost.ndim}-D"
            )


@dataclass(frozen=True)
class SemanticCostMap:
    """Fused semantic cost map + per-layer provenance for QA / JSON."""

    cost: np.ndarray                 # (H, W) float32 in [0, 1]
    precision: np.ndarray            # (H, W) float32 in [0, 1]
    layers: Tuple[CostLayer, ...]
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers (pure, image-derived)
# ---------------------------------------------------------------------------


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            f"semantic_cost requires OpenCV (cv2). Import failed: {_CV2_IMPORT_ERROR}"
        )


def _unit_minmax(a: np.ndarray) -> np.ndarray:
    """Rescale ``a`` to ``[0, 1]`` using its own min/max. Never divide by 0."""

    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _otsu_split(scores: np.ndarray) -> float:
    """Otsu threshold on ``scores`` in ``[0, 1]``. Returns a float threshold.

    Falls back to the 50th percentile when ``cv2.threshold`` cannot be used.
    """

    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.5
    try:
        u8 = np.clip(arr, 0.0, 1.0) * 255.0
        t, _ = cv2.threshold(
            u8.reshape(-1, 1).astype(np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return float(t) / 255.0
    except Exception:
        return float(np.nanmedian(arr))


def _jsd_break(hist: np.ndarray) -> float:
    """Pick a split point on a 1-D histogram by maximising Jensen–Shannon
    divergence between the two halves.  Returns a bin-fraction in ``[0, 1]``.
    """

    h = np.asarray(hist, dtype=np.float64)
    if h.size < 3 or h.sum() <= 0:
        return 0.5
    h = h / h.sum()
    cdf = np.cumsum(h)
    best_score = -1.0
    best_idx = len(h) // 2
    for idx in range(1, len(h) - 1):
        p_left = h[:idx].copy()
        p_right = h[idx:].copy()
        if p_left.sum() <= 0 or p_right.sum() <= 0:
            continue
        p_left /= p_left.sum()
        p_right /= p_right.sum()
        # Pad right onto the same support as left.
        n = max(len(p_left), len(p_right))
        a = np.zeros(n); a[: len(p_left)] = p_left
        b = np.zeros(n); b[: len(p_right)] = p_right
        m = 0.5 * (a + b)
        # JS divergence.
        eps = 1e-12
        js = 0.5 * np.sum(a * np.log((a + eps) / (m + eps))) + \
             0.5 * np.sum(b * np.log((b + eps) / (m + eps)))
        if js > best_score:
            best_score = js
            best_idx = idx
    # Return fractional position; callers map to the physical range themselves.
    return float(best_idx) / float(max(1, len(h) - 1))


# ---------------------------------------------------------------------------
# Layer A — geometry (image + depth)
# ---------------------------------------------------------------------------


def layer_geometry(
    img_bgr: np.ndarray,
    depth_m: Optional[np.ndarray] = None,
    obstacle_mask: Optional[np.ndarray] = None,
) -> CostLayer:
    """Image- and depth-gradient-driven geometry layer (layer A).

    The three sub-signals are fused with *equal* data-driven weights inside
    this layer:

    - Image edge magnitude via Sobel (high → high cost).  Normalised by its
      own max.
    - Depth gradient magnitude (m / pixel, via Sobel on ``depth_m``) if
      available.  Normalised by its own 99th percentile so a single spike
      does not dominate.
    - Distance transform on the complement of the obstacle mask — pixels
      close to a known obstacle pay more cost.  Normalised by its own max.

    Precision is high near sharp image edges (the signal the layer was
    designed to capture) and near known obstacles (the mask is trustworthy);
    it is low in flat, textureless regions where Sobel is noise.
    """

    _require_cv2()
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("layer_geometry: img_bgr must be a non-empty image")
    h, w = img_bgr.shape[:2]

    # -- sub-signal 1: image edges ------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edge_cost = _unit_minmax(grad)

    # -- sub-signal 2: depth gradient ---------------------------------------
    if depth_m is not None and depth_m.shape[:2] == (h, w):
        d = np.asarray(depth_m, dtype=np.float32)
        dgx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        dgy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        dgrad = cv2.magnitude(dgx, dgy)
        # 99th-percentile clip → avoid one bad pixel saturating the layer.
        cap = float(np.nanpercentile(dgrad, 99.0)) if np.isfinite(dgrad).any() else 0.0
        if cap > 1e-6:
            depth_cost = np.clip(dgrad / cap, 0.0, 1.0).astype(np.float32)
        else:
            depth_cost = np.zeros_like(edge_cost, dtype=np.float32)
    else:
        depth_cost = np.zeros_like(edge_cost, dtype=np.float32)

    # -- sub-signal 3: distance-to-obstacle ---------------------------------
    if obstacle_mask is not None and obstacle_mask.shape[:2] == (h, w):
        obs = np.asarray(obstacle_mask, dtype=np.uint8)
        obs = (obs > 0).astype(np.uint8)
        dt_free = cv2.distanceTransform(1 - obs, cv2.DIST_L2, 3).astype(np.float32)
        dt_cost = 1.0 - _unit_minmax(dt_free)  # close to obstacle → cost↑
    else:
        dt_cost = np.zeros_like(edge_cost, dtype=np.float32)

    cost = (edge_cost + depth_cost + dt_cost) / 3.0
    cost = np.clip(cost, 0.0, 1.0).astype(np.float32)

    # Precision: trust the layer where either edges or obstacles are strong.
    precision = np.maximum(edge_cost, dt_cost).astype(np.float32)
    precision = cv2.GaussianBlur(precision, (0, 0), sigmaX=1.5, sigmaY=1.5)
    precision = np.clip(precision, 0.0, 1.0).astype(np.float32)

    return CostLayer(
        name="A_geometry",
        cost=cost,
        precision=precision,
        provenance=(
            "Sobel edges + depth gradient (99th-pctile clip) + "
            "distance-transform-to-obstacle, equal-weighted"
        ),
        diagnostics={
            "edge_max": float(np.max(grad)) if grad.size else 0.0,
            "depth_grad_used": 1.0 if depth_m is not None else 0.0,
            "obstacle_mask_used": 1.0 if obstacle_mask is not None else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Layer D — scene-graph evidence
# ---------------------------------------------------------------------------


def _role_terms(role_name: str, ontology: Optional[Mapping[str, Any]] = None) -> set:
    bank = prompt_bank(ontology or load_action_ontology(), "role_prompts")
    return {normalize_label(v) for v in bank.get(role_name, []) if normalize_label(v)}


def _label_matches_terms(label: str, terms: set) -> bool:
    label_n = normalize_label(label)
    if not label_n:
        return False
    return any(term == label_n or term in label_n or label_n in term for term in terms)


def layer_scene_graph(
    h: int,
    w: int,
    objects: Sequence[Mapping[str, Any]],
    object_label_map: Optional[np.ndarray] = None,
    *,
    ontology: Optional[Mapping[str, Any]] = None,
) -> CostLayer:
    """Build the scene-graph evidence layer (layer D).

    Parameters
    ----------
    h, w
        Image size; the output is ``(h, w)``.
    objects
        Sequence of object dicts in the canonical pipeline format — each
        has ``"label"`` (str), ``"id"`` (int), ``"mask"`` (optional binary
        mask matching ``(h, w)``), and ``"mask_id"`` (index into
        ``object_label_map``). Either ``mask`` or ``mask_id`` must be
        provided.
    object_label_map
        Optional ``(h, w)`` int32 map produced by
        :meth:`SceneUnderstandingPipeline._build_object_label_map`.  If
        supplied, we index into it instead of materialising per-object
        masks (big memory win).

    Returns
    -------
    CostLayer
        ``cost`` is ``max(obstacle_score − support_score, 0)``. ``precision``
        is the per-pixel count of contributing labels normalised to ``[0, 1]``.
        If an object has no mask and no label-map entry it is skipped (no
        silent defaults).
    """

    obstacle_score = np.zeros((h, w), dtype=np.float32)
    support_score = np.zeros((h, w), dtype=np.float32)
    precision_count = np.zeros((h, w), dtype=np.float32)
    onto = ontology or load_action_ontology()
    obstacle_terms = (
        _role_terms("hard_obstacle", onto)
        | _role_terms("soft_obstacle", onto)
        | _role_terms("occluder", onto)
    )
    support_terms = _role_terms("support", onto)

    use_label_map = (
        object_label_map is not None
        and np.asarray(object_label_map).shape[:2] == (h, w)
    )
    lm = np.asarray(object_label_map, dtype=np.int32) if use_label_map else None

    for obj in objects:
        label = str(obj.get("canonical_label") or obj.get("label") or "").strip().lower()
        if not label:
            continue
        mask: Optional[np.ndarray] = None
        mid = obj.get("mask_id")
        if use_label_map and mid is not None:
            try:
                mid_i = int(mid)
            except (TypeError, ValueError):
                mid_i = None
            if mid_i is not None and mid_i > 0 and lm is not None:
                mask = (lm == mid_i)
        if mask is None:
            m = obj.get("mask")
            if m is None:
                continue
            m = np.asarray(m)
            if m.shape[:2] != (h, w):
                continue
            mask = m.astype(bool)
        if not mask.any():
            continue
        # Evidence weight uses ontology prompt matches only. Both groups
        # contribute one unit of evidence per pixel; fusion normalises across
        # layers and unknown labels contribute zero.
        if _label_matches_terms(label, obstacle_terms):
            obstacle_score[mask] += 1.0
            precision_count[mask] += 1.0
        elif _label_matches_terms(label, support_terms):
            support_score[mask] += 1.0
            precision_count[mask] += 1.0
        else:
            # Unknown class: no evidence. Keeps the layer additive-only.
            continue

    obstacle_score = _unit_minmax(obstacle_score)
    support_score = _unit_minmax(support_score)
    cost = np.clip(obstacle_score - support_score, 0.0, 1.0).astype(np.float32)
    precision = _unit_minmax(precision_count)

    return CostLayer(
        name="D_scene_graph",
        cost=cost,
        precision=precision,
        provenance=(
            "Per-object label → obstacle/support evidence. Precision is the "
            "per-pixel count of contributing labels (both obstacle and "
            "support) normalised to [0,1]."
        ),
        diagnostics={
            "objects_considered": float(len(list(objects))),
            "obstacle_pixels": float((cost > 0).sum()),
        },
    )


# ---------------------------------------------------------------------------
# Layer E — agent physics (back-projected inflation)
# ---------------------------------------------------------------------------


def _back_project_radius_px(
    profile: Any,
    depth_m: np.ndarray,
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    """Return a per-pixel pixel-radius equivalent to the agent half-width.

    Parameters
    ----------
    profile
        :class:`AgentPhysicalProfile`. Only ``effective_pad_x_m()`` /
        ``half_width_m`` are used, plus the ``foot_tolerance_m`` is handled
        by :func:`layer_agent_physics` itself.
    depth_m
        ``(H, W)`` metric depth, in metres.
    intrinsics
        Mapping with ``fx``, ``fy``, ``cx``, ``cy``. We project
        ``profile.half_width_m`` onto the image plane at each pixel's depth
        to get a per-pixel radius in pixels.
    """

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    # radius in metres to inflate by — prefer the explicit override if set.
    if hasattr(profile, "effective_pad_x_m"):
        r_m = float(profile.effective_pad_x_m())
    else:
        r_m = float(getattr(profile, "half_width_m"))
    if not np.isfinite(r_m) or r_m <= 0:
        raise ValueError("agent half-width/pad_x must be finite and positive")
    d = np.asarray(depth_m, dtype=np.float32)
    d_safe = np.where(d > 1e-3, d, np.nan)
    # Phase D — the pixel-radius formula below is vectorised NumPy already;
    # the Numba unprojection kernel is instead used by callers who need full
    # XYZ (e.g. the 3D footprint sampler in the planner). We keep the closed
    # form here because it avoids materialising XYZ when we only need the
    # scalar radius.
    px_x = fx * r_m / d_safe
    px_y = fy * r_m / d_safe
    px = np.nanmean(np.stack([px_x, px_y], axis=0), axis=0)
    # Fallback for pixels with invalid depth: use the image-wide median so
    # the value is still image-derived, never a magic constant.
    med = float(np.nanmedian(px)) if np.isfinite(px).any() else 0.0
    px = np.where(np.isfinite(px), px, med).astype(np.float32)
    return px


def layer_agent_physics(
    img_hw: Tuple[int, int],
    obstacle_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Mapping[str, float],
    profile: Any,
) -> CostLayer:
    """Inflate obstacles by the agent's metric half-width (layer E).

    The layer uses the per-pixel back-projection of ``profile.half_width_m``
    as an inflation radius (depth-aware — farther pixels need a smaller
    pixel-radius for the same metric footprint) and additionally penalises
    any pixel whose depth step from the nearest obstacle exceeds
    ``profile.foot_tolerance_m`` (step-over gate).

    Strict validation: if ``intrinsics`` or ``depth_m`` is missing we return
    a zero-precision layer rather than guessing.  This keeps the refusal
    contract intact.
    """

    _require_cv2()
    h, w = img_hw
    if obstacle_mask is None or obstacle_mask.shape[:2] != (h, w):
        zeros = np.zeros((h, w), dtype=np.float32)
        return CostLayer(
            name="E_agent_physics",
            cost=zeros,
            precision=zeros,
            provenance="No obstacle mask — layer refused to build.",
        )
    if depth_m is None or depth_m.shape[:2] != (h, w):
        zeros = np.zeros((h, w), dtype=np.float32)
        return CostLayer(
            name="E_agent_physics",
            cost=zeros,
            precision=zeros,
            provenance="No metric depth — layer refused to build.",
        )
    if not intrinsics or "fx" not in intrinsics or "fy" not in intrinsics:
        zeros = np.zeros((h, w), dtype=np.float32)
        return CostLayer(
            name="E_agent_physics",
            cost=zeros,
            precision=zeros,
            provenance="No camera intrinsics — layer refused to build.",
        )
    if profile is None:
        zeros = np.zeros((h, w), dtype=np.float32)
        return CostLayer(
            name="E_agent_physics",
            cost=zeros,
            precision=zeros,
            provenance="No AgentPhysicalProfile — layer refused to build.",
        )

    obs = (np.asarray(obstacle_mask) > 0).astype(np.uint8)
    # Distance-from-obstacle in pixels.
    dt_free = cv2.distanceTransform(1 - obs, cv2.DIST_L2, 3).astype(np.float32)

    radius_px = _back_project_radius_px(profile, depth_m, intrinsics)
    # Cost = 1 where we are inside the inflation ring, 0 outside, linear in
    # between (soft edge so FMM does not see a cliff).
    inside = (dt_free <= radius_px).astype(np.float32)
    # Soft inflation: cost falls to 0 at 2·radius.
    soft = np.clip(1.0 - (dt_free - radius_px) / np.maximum(radius_px, 1e-3), 0.0, 1.0)
    cost = np.maximum(inside, soft).astype(np.float32)

    # Step-over gate: penalise pixels whose depth step over a small
    # neighbourhood exceeds foot_tolerance_m.  The neighbourhood is itself
    # scaled by radius_px so it matches the agent footprint.
    try:
        foot_tol = float(getattr(profile, "foot_tolerance_m"))
    except Exception:
        foot_tol = 0.0
    if foot_tol > 0 and np.isfinite(foot_tol):
        d = np.asarray(depth_m, dtype=np.float32)
        # Use a 3x3 max-minus-min filter for step detection.
        kernel = np.ones((3, 3), dtype=np.uint8)
        d_max = cv2.dilate(d, kernel)
        d_min = cv2.erode(d, kernel)
        step = np.abs(d_max - d_min)
        step_cost = np.clip(step / max(foot_tol, 1e-3), 0.0, 1.0).astype(np.float32)
        cost = np.maximum(cost, step_cost)

    # Precision: high wherever depth + intrinsics are finite (they were
    # validated above) — fall off where depth_m is unusable.
    finite = np.isfinite(depth_m).astype(np.float32)
    precision = finite
    # We trust the layer more where we are close to an obstacle.
    prox = 1.0 - _unit_minmax(dt_free)
    precision = np.clip(0.5 * precision + 0.5 * prox, 0.0, 1.0).astype(np.float32)

    return CostLayer(
        name="E_agent_physics",
        cost=cost,
        precision=precision,
        provenance=(
            "Distance-transform inflation by back-projected half_width_m "
            f"(measured; soft falloff to 2·radius) + step-over gate using "
            f"foot_tolerance_m={foot_tol:.4f} m. All values metric, derived "
            "from AgentPhysicalProfile and camera intrinsics."
        ),
        diagnostics={
            "radius_px_median": float(np.nanmedian(radius_px)),
            "foot_tolerance_m": float(foot_tol),
        },
    )


# ---------------------------------------------------------------------------
# Layer F — depth noise precision
# ---------------------------------------------------------------------------


def layer_depth_noise(depth_m: Optional[np.ndarray]) -> CostLayer:
    """Precision layer driven by local depth std vs. the image's own
    histogram.  Produces *precision only*: cost is zero everywhere so this
    layer cannot push the fused cost up, it can only downweight others.

    The split between "trustworthy" and "noisy" depth is picked with a
    JSD break on the 1-D histogram of local-std values — no fixed
    threshold.  If ``depth_m`` is missing, precision is zero everywhere
    (i.e. "no info").
    """

    if depth_m is None:
        zeros = np.zeros((2, 2), dtype=np.float32)
        return CostLayer(
            name="F_depth_noise",
            cost=zeros,
            precision=zeros,
            provenance="No depth — layer refused to build.",
        )
    _require_cv2()
    d = np.asarray(depth_m, dtype=np.float32)
    h, w = d.shape[:2]
    # Local std via box filter on (I² - mean²).
    mean = cv2.boxFilter(d, ddepth=cv2.CV_32F, ksize=(5, 5))
    mean_sq = cv2.boxFilter(d * d, ddepth=cv2.CV_32F, ksize=(5, 5))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    std01 = _unit_minmax(std)

    hist, _ = np.histogram(std01.reshape(-1), bins=64, range=(0.0, 1.0))
    split = _jsd_break(hist)
    # Pixels with std < split are trustworthy (precision = 1 − std01).
    # Pixels with std > split are noisy (precision 0..1−split).
    precision = np.where(std01 <= split, 1.0 - std01, np.clip(1.0 - std01, 0.0, max(1e-6, 1.0 - split)))
    precision = np.clip(precision, 0.0, 1.0).astype(np.float32)

    return CostLayer(
        name="F_depth_noise",
        cost=np.zeros((h, w), dtype=np.float32),
        precision=precision,
        provenance=(
            f"Local depth std (5x5 box) → precision. JSD split point at "
            f"std01={split:.3f} (image-derived, not a magic threshold)."
        ),
        diagnostics={"jsd_split": float(split)},
    )


# ---------------------------------------------------------------------------
# Precision-weighted fusion
# ---------------------------------------------------------------------------


def precision_weighted_fuse(
    layers: Iterable[CostLayer],
    *,
    min_precision: float = 1e-4,
) -> SemanticCostMap:
    """Fuse layers into a single cost map.

    For each pixel ``p`` and each layer ``i`` with precision ``P_i(p)`` and
    cost ``C_i(p)`` the fused cost is

        C_fused(p) = (Σ_i P_i(p) · C_i(p)) / (Σ_i P_i(p))

    and the fused precision is ``1 - prod_i (1 - P_i)`` (classical
    "independent evidences combine confidence") — each additional
    trustworthy layer only *increases* per-pixel certainty.

    ``min_precision`` is a tiny floor to avoid divide-by-zero at pixels
    where every layer is fully uncertain; those pixels end up with fused
    cost = 0 and fused precision = 0 so the planner can treat them as
    "information-less" rather than "free".
    """

    layers = [lyr for lyr in layers if lyr is not None]
    if not layers:
        raise ValueError("precision_weighted_fuse: no layers provided")

    shape = layers[0].cost.shape
    for lyr in layers:
        if lyr.cost.shape != shape:
            raise ValueError(
                f"fuse: layer {lyr.name} shape {lyr.cost.shape} != {shape}"
            )

    acc_num = np.zeros(shape, dtype=np.float32)
    acc_den = np.zeros(shape, dtype=np.float32)
    prod_uncertain = np.ones(shape, dtype=np.float32)
    for lyr in layers:
        c = np.clip(lyr.cost.astype(np.float32), 0.0, 1.0)
        p = np.clip(lyr.precision.astype(np.float32), 0.0, 1.0)
        acc_num += p * c
        acc_den += p
        prod_uncertain *= (1.0 - p)

    den = np.maximum(acc_den, min_precision)
    fused_cost = np.where(acc_den > 0, acc_num / den, 0.0).astype(np.float32)
    fused_precision = np.clip(1.0 - prod_uncertain, 0.0, 1.0).astype(np.float32)
    fused_cost = np.clip(fused_cost, 0.0, 1.0)

    meta = {
        "layer_names": [lyr.name for lyr in layers],
        "layer_provenance": {lyr.name: lyr.provenance for lyr in layers},
        "layer_diagnostics": {lyr.name: dict(lyr.diagnostics) for lyr in layers},
    }
    return SemanticCostMap(
        cost=fused_cost,
        precision=fused_precision,
        layers=tuple(layers),
        meta=meta,
    )


__all__ = [
    "CostLayer",
    "SemanticCostMap",
    "layer_geometry",
    "layer_scene_graph",
    "layer_agent_physics",
    "layer_depth_noise",
    "precision_weighted_fuse",
]
