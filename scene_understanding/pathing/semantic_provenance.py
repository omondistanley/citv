"""QA artefacts for the semantic cost map (Phase C.6).

Public entry point :func:`save_semantic_cost_artifacts` writes, into a
caller-supplied output directory, the following files:

- ``cost_map.npz`` — compressed cost + precision arrays (fp16 for disk
  economy; dtype noted in the sidecar).
- ``cost_layers/<name>.png`` — heatmap PNG per layer (cost only).
- ``cost_layers/<name>_precision.png`` — heatmap PNG per layer
  (precision only).
- ``cost_provenance.png`` — argmax-provenance visualisation (for each
  pixel, which layer has the highest per-layer ``P_i`` — helps answer
  "why did the planner go that way?").
- ``cost_meta.json`` — layer names, provenance strings, per-layer
  diagnostics, and the semantic cost map version.

:func:`annotate_path_with_provenance` adds ``cost_layers_local`` to every
waypoint of a path (the six per-layer costs at that pixel) and a
``semantic_summary`` dict at the trajectory level so path JSON can answer
"which layer drove the planner here".

All outputs are **additive**: existing outputs are not modified. Callers
are responsible for honouring ``cfg.path_cost_emit_layers_for_qa`` before
invoking the writer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scene_understanding.pathing.semantic_cost import CostLayer, SemanticCostMap

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _heatmap_png(path: Path, arr: np.ndarray) -> None:
    if cv2 is None:
        return
    a = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)
    u8 = (a * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    # Phase F — QA heatmaps are diagnostic artefacts, never final
    # deliverables, so we use zlib compression level 1 for ~3× faster
    # encoding at modest file-size cost.
    cv2.imwrite(str(path), heat, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])


def _provenance_map(layers: Sequence[CostLayer]) -> np.ndarray:
    """Per-pixel argmax over ``precision * cost`` — "which layer drove here".

    Returns a uint8 map with values ``1..len(layers)`` (0 = no evidence).
    """

    if not layers:
        return np.zeros((1, 1), dtype=np.uint8)
    shape = layers[0].cost.shape
    stack = np.stack([np.clip(l.cost, 0, 1) * np.clip(l.precision, 0, 1) for l in layers], axis=0)
    total = stack.sum(axis=0)
    idx = np.argmax(stack, axis=0)
    # Where total == 0, mark as 0 (no evidence).
    mp = np.where(total > 1e-6, idx.astype(np.int32) + 1, 0).astype(np.uint8)
    return mp


def _palette_png(path: Path, prov_map: np.ndarray, layer_names: Sequence[str]) -> None:
    if cv2 is None:
        return
    h, w = prov_map.shape[:2]
    palette = np.zeros((256, 3), dtype=np.uint8)
    # Fixed evenly-spaced hues; layer_names drive the count but not the values.
    n = max(1, len(layer_names))
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
        palette[i + 1] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    rgb = palette[prov_map]
    cv2.imwrite(str(path), rgb, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])


def save_semantic_cost_artifacts(
    out_dir: str | Path,
    scm: SemanticCostMap,
    *,
    version: str = "2026-04-21.a",
) -> Dict[str, str]:
    """Write cost_map.npz, cost_layers/, cost_provenance.png, cost_meta.json.

    Returns a dict of ``{artefact_name: path_str}``.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers_dir = out / "cost_layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out / "cost_map.npz"
    np.savez_compressed(
        npz_path,
        cost=scm.cost.astype(np.float16),
        precision=scm.precision.astype(np.float16),
    )

    layer_file_index: Dict[str, Dict[str, str]] = {}
    for lyr in scm.layers:
        c_path = layers_dir / f"{lyr.name}_cost.png"
        p_path = layers_dir / f"{lyr.name}_precision.png"
        _heatmap_png(c_path, lyr.cost)
        _heatmap_png(p_path, lyr.precision)
        layer_file_index[lyr.name] = {
            "cost_png": str(c_path.relative_to(out)),
            "precision_png": str(p_path.relative_to(out)),
        }

    prov_map = _provenance_map(scm.layers)
    prov_png = out / "cost_provenance.png"
    _palette_png(prov_png, prov_map, [l.name for l in scm.layers])

    meta = {
        "version": version,
        "shape": list(scm.cost.shape),
        "layers": [
            {
                "name": lyr.name,
                "provenance": lyr.provenance,
                "diagnostics": dict(lyr.diagnostics),
                **layer_file_index.get(lyr.name, {}),
            }
            for lyr in scm.layers
        ],
        "fused_cost_npz": "cost_map.npz",
        "provenance_png": "cost_provenance.png",
        "meta_from": scm.meta,
    }
    meta_path = out / "cost_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "cost_map_npz": str(npz_path),
        "cost_meta_json": str(meta_path),
        "cost_provenance_png": str(prov_png),
        "cost_layers_dir": str(layers_dir),
    }


# ---------------------------------------------------------------------------
# Per-waypoint / per-trajectory path JSON enrichment
# ---------------------------------------------------------------------------


def annotate_path_with_provenance(
    path: Sequence[Tuple[int, int]],
    scm: SemanticCostMap,
) -> Dict[str, Any]:
    """Return ``{waypoints: [...], summary: {...}}`` describing the path's
    cost origin. Shapes:

    - ``waypoints[i].layers`` maps layer name → cost value at that pixel.
    - ``summary.layer_share`` maps layer name → fractional share of
      cost-weight along the path.
    """

    if not path:
        return {"waypoints": [], "summary": {}}
    h, w = scm.cost.shape[:2]
    waypoints: List[Dict[str, Any]] = []
    layer_weights: Dict[str, float] = {lyr.name: 0.0 for lyr in scm.layers}
    for (x, y) in path:
        xi = int(max(0, min(w - 1, x)))
        yi = int(max(0, min(h - 1, y)))
        wp_layers: Dict[str, float] = {}
        for lyr in scm.layers:
            c_here = float(lyr.cost[yi, xi])
            p_here = float(lyr.precision[yi, xi])
            wp_layers[lyr.name] = round(c_here, 4)
            layer_weights[lyr.name] += p_here * c_here
        waypoints.append({
            "x": xi,
            "y": yi,
            "fused_cost": round(float(scm.cost[yi, xi]), 4),
            "fused_precision": round(float(scm.precision[yi, xi]), 4),
            "layers": wp_layers,
        })
    total = sum(layer_weights.values()) or 1e-6
    share = {k: round(v / total, 4) for k, v in layer_weights.items()}
    dominant = max(share.items(), key=lambda kv: kv[1])[0] if share else ""
    return {
        "waypoints": waypoints,
        "summary": {
            "layer_share": share,
            "dominant_layer": dominant,
            "waypoint_count": len(waypoints),
        },
    }


__all__ = ["save_semantic_cost_artifacts", "annotate_path_with_provenance"]
