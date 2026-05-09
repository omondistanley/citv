"""Stable relative filenames + candidate path dedupe helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import math

PATH_HYPOTHESES_JSON_NAME = "path_hypotheses.json"
# v3 retired: kept only for backward path lookup helpers used by tests/tools.
PATH_HYPOTHESES_FULL_JSON_NAME = "path_hypotheses_full.json"


def path_hypotheses_json_path(paths_root_dir: Path) -> Path:
    return paths_root_dir / PATH_HYPOTHESES_JSON_NAME


def path_hypotheses_full_json_path(paths_root_dir: Path) -> Path:
    return paths_root_dir / PATH_HYPOTHESES_FULL_JSON_NAME


def _resample_polyline(poly: List[Any], n: int) -> List[Tuple[float, float]]:
    """Uniform arc-length resample of a 2D polyline to ``n`` samples."""
    if not poly or n <= 1:
        return [(float(poly[0][0]), float(poly[0][1]))] if poly else []
    pts: List[Tuple[float, float]] = []
    for xy in poly:
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            try:
                pts.append((float(xy[0]), float(xy[1])))
            except (TypeError, ValueError):
                continue
    if len(pts) < 2:
        return pts
    seg_len = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1)]
    total = sum(seg_len)
    if total <= 1e-6:
        return [pts[0]] * n
    cum = [0.0]
    for s in seg_len:
        cum.append(cum[-1] + s)
    out: List[Tuple[float, float]] = []
    for i in range(n):
        target = total * (i / max(1, n - 1))
        # find segment containing target
        j = 0
        while j + 1 < len(cum) and cum[j + 1] < target:
            j += 1
        if j >= len(pts) - 1:
            out.append(pts[-1])
            continue
        denom = max(1e-9, cum[j + 1] - cum[j])
        t = (target - cum[j]) / denom
        x = pts[j][0] + t * (pts[j + 1][0] - pts[j][0])
        y = pts[j][1] + t * (pts[j + 1][1] - pts[j][1])
        out.append((x, y))
    return out


def _hausdorff(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    """Symmetric directed Hausdorff distance in pixels."""
    if not a or not b:
        return float("inf")

    def directed(p: List[Tuple[float, float]], q: List[Tuple[float, float]]) -> float:
        m = 0.0
        for px, py in p:
            best = float("inf")
            for qx, qy in q:
                d = (px - qx) * (px - qx) + (py - qy) * (py - qy)
                if d < best:
                    best = d
            if best > m:
                m = best
        return math.sqrt(m)

    return max(directed(a, b), directed(b, a))


def _pair_key(path: Dict[str, Any]) -> Tuple[Any, ...]:
    s = str((path.get("source_entity") or {}).get("id", ""))
    t = str((path.get("target_entity") or {}).get("id", ""))
    if s and t and t < s:
        s, t = t, s
    ptype = str(path.get("path_type", "") or "")
    # Do not collapse portals / composites / object–region routes with same-id geodesics.
    if ptype in ("portal", "composite_cc_chain", "object_region_fmm", "region_fmm"):
        return (s, t, ptype)
    return (s, t)


def dedupe_paths(
    paths: List[Dict[str, Any]],
    *,
    max_per_pair: int = 1,
    frechet_thresh_px: float = 18.0,
    samples: int = 16,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deduplicate near-identical FMM path hypotheses.

    Returns ``(kept, dropped)`` where every dropped record gains a
    ``dropped_reason`` string for ``path_diagnostics.json``.
    """
    if not paths:
        return [], []
    if int(max_per_pair) <= 0:
        return list(paths), []
    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for p in paths:
        by_pair.setdefault(_pair_key(p), []).append(p)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for pair, group in by_pair.items():
        # Keep best path first, then remaining diverse paths up to max_per_pair.
        group_sorted = sorted(
            group,
            key=lambda r: float((r.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=True,
        )
        kept_in_pair: List[Tuple[Dict[str, Any], List[Tuple[float, float]]]] = []
        for cand in group_sorted:
            poly = _resample_polyline(cand.get("polyline_2d") or [], samples)
            if not poly:
                rec = dict(cand)
                rec["dropped_reason"] = "empty_polyline"
                dropped.append(rec)
                continue
            duplicate_of = None
            for keep_path, keep_poly in kept_in_pair:
                if _hausdorff(poly, keep_poly) <= frechet_thresh_px:
                    duplicate_of = str(keep_path.get("path_id", ""))
                    break
            if duplicate_of is not None:
                rec = dict(cand)
                rec["dropped_reason"] = f"near_duplicate_of:{duplicate_of}"
                dropped.append(rec)
                continue
            if len(kept_in_pair) >= max_per_pair:
                rec = dict(cand)
                rec["dropped_reason"] = (
                    f"max_per_pair:{max_per_pair}_top_kept:"
                    f"{','.join(str(k.get('path_id', '')) for k, _ in kept_in_pair)}"
                )
                dropped.append(rec)
                continue
            kept_in_pair.append((cand, poly))
            kept.append(cand)
    return kept, dropped


def dedupe_by_route_signature(
    paths: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deduplicate strictly identical ``route_signature`` records.

    Useful for atlas manifest paths where the same route could be appended
    multiple times by different rankers.
    """
    seen: Dict[str, str] = {}
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for p in paths:
        sig = str(p.get("route_signature") or "")
        if not sig:
            kept.append(p)
            continue
        if sig in seen:
            rec = dict(p)
            rec["dropped_reason"] = f"duplicate_route_signature_of:{seen[sig]}"
            dropped.append(rec)
            continue
        seen[sig] = str(p.get("path_id", ""))
        kept.append(p)
    return kept, dropped
