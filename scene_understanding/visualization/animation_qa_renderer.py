"""Animation QA renderer for dual-FPS path/trajectory verification."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_sheet_frames(sheet_bgr: np.ndarray) -> List[np.ndarray]:
    cell = 512
    frames: List[np.ndarray] = []
    for r in range(4):
        for c in range(4):
            y0 = r * cell
            x0 = c * cell
            frames.append(sheet_bgr[y0:y0 + cell, x0:x0 + cell].copy())
    return frames


def _load_sprite_registry(cfg: Any) -> Dict[str, List[np.ndarray]]:
    reg: Dict[str, List[np.ndarray]] = {}
    mapping = dict(getattr(cfg, "path_animation_sprite_sheets", {}) or {})
    for motion, p in mapping.items():
        path = Path(str(p))
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or img.shape[0] != 2048 or img.shape[1] != 2048:
            continue
        reg[str(motion).lower()] = _split_sheet_frames(img)
    return reg


def _depth_norm(z: float, z_low: float, z_high: float, eps: float) -> float:
    return float(max(0.0, min(1.0, (z - z_low) / max(eps, z_high - z_low))))


def _style_from_depth(z_norm: float, cfg: Any) -> Tuple[int, float]:
    t_min = int(getattr(cfg, "path_animation_thickness_min_px", 1))
    t_max = int(getattr(cfg, "path_animation_thickness_max_px", 7))
    g_t = float(getattr(cfg, "path_animation_thickness_gamma", 1.35))
    a_min = float(getattr(cfg, "path_animation_alpha_min", 0.28))
    a_max = float(getattr(cfg, "path_animation_alpha_max", 0.95))
    g_a = float(getattr(cfg, "path_animation_alpha_gamma", 1.10))
    near = 1.0 - z_norm
    thickness = int(round(t_min + (near**g_t) * (t_max - t_min)))
    alpha = float(a_min + (near**g_a) * (a_max - a_min))
    return max(1, thickness), max(0.0, min(1.0, alpha))


def _scene_depth_bounds(dm: np.ndarray, cfg: Any) -> Tuple[float, float]:
    """Scene-wide depth percentiles for consistent character scaling in QA clips."""
    arr = np.asarray(dm, dtype=np.float32).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(arr, float(getattr(cfg, "path_animation_depth_percentile_low", 5.0))))
    hi = float(np.percentile(arr, float(getattr(cfg, "path_animation_depth_percentile_high", 95.0))))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def _depth_at(dm: Optional[np.ndarray], x: int, y: int) -> float:
    if dm is None:
        return float("nan")
    d = np.asarray(dm, dtype=np.float32)
    if 0 <= y < d.shape[0] and 0 <= x < d.shape[1]:
        v = float(d[y, x])
        if np.isfinite(v):
            return v
    return float("nan")


def _polyline_cumulative_lengths(pts: List[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
    if len(pts) < 2:
        return np.zeros((0,), dtype=np.float64), 0.0
    acc: List[float] = [0.0]
    for i in range(1, len(pts)):
        dx = float(pts[i][0] - pts[i - 1][0])
        dy = float(pts[i][1] - pts[i - 1][1])
        acc.append(acc[-1] + math.hypot(dx, dy))
    total = float(acc[-1])
    return np.asarray(acc, dtype=np.float64), max(total, 1e-6)


def _polyline_sample(
    pts: List[Tuple[int, int]],
    cum: np.ndarray,
    total_len: float,
    s: float,
) -> Tuple[float, float, float]:
    """Sample point and tangent along polyline; s in [0, 1] by arc length."""
    s = float(max(0.0, min(1.0, s)))
    if len(pts) < 2:
        return float(pts[0][0]), float(pts[0][1]), 0.0
    target = s * total_len
    # find segment i where cum[i] <= target <= cum[i+1]
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    idx = max(0, min(len(pts) - 2, idx))
    seg_len = float(cum[idx + 1] - cum[idx]) or 1e-6
    u = (target - float(cum[idx])) / seg_len
    x0, y0 = float(pts[idx][0]), float(pts[idx][1])
    x1, y1 = float(pts[idx + 1][0]), float(pts[idx + 1][1])
    x = x0 + u * (x1 - x0)
    y = y0 + u * (y1 - y0)
    theta = math.atan2(y1 - y0, x1 - x0)
    return x, y, theta


def _semantic_feasibility(component: Dict[str, Any]) -> float:
    sem = component.get("scene_semantic_constraints") or {}
    score = float(sem.get("semantic_validity_score", 0.0))
    if not bool(sem.get("semantic_valid", True)):
        score *= 0.5
    return max(0.0, min(1.0, score))


def _path_semantic_multiplier(path: Dict[str, Any]) -> float:
    if not path:
        return 1.0
    base = float(path.get("semantic_validity_score", 0.75))
    if not bool(path.get("semantic_valid", True)):
        base *= 0.55
    return max(0.15, min(1.0, base))


def _region_feet_alignment(
    lm: Optional[np.ndarray],
    x: int,
    y: int,
    regions_traversed: List[Any],
) -> float:
    """1.0 if current label is plausibly on the path corridor; soft gate only."""
    if lm is None or not regions_traversed:
        return 1.0
    h, w = lm.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return 0.85
    rid = lm[y, x]
    tokens = {str(r).strip().lower() for r in regions_traversed if str(r).strip()}
    if not tokens:
        return 1.0
    if str(int(rid)) in tokens or f"region_{int(rid)}" in tokens:
        return 1.0
    return 0.75


def _draw_depth_tapered_polyline(
    canvas: np.ndarray,
    pts: List[Tuple[int, int]],
    depth_values: List[float],
    color: Tuple[int, int, int],
    cfg: Any,
    semantic_multiplier: float = 1.0,
    style_beta: float = 0.35,
) -> Dict[str, float]:
    if len(pts) < 2 or len(depth_values) < len(pts):
        return {"thickness_depth_monotonicity": 1.0, "temporal_flicker_proxy": 0.0}
    valid_depth = np.asarray([d for d in depth_values if np.isfinite(d)], dtype=np.float32)
    if valid_depth.size == 0:
        valid_depth = np.asarray([1.0], dtype=np.float32)
    z_low = float(np.percentile(valid_depth, float(getattr(cfg, "path_animation_depth_percentile_low", 5.0))))
    z_high = float(np.percentile(valid_depth, float(getattr(cfg, "path_animation_depth_percentile_high", 95.0))))
    eps = float(getattr(cfg, "path_animation_depth_eps", 1e-6))

    smooth_thick = None
    smooth_alpha = None
    thicknesses: List[float] = []
    norms: List[float] = []
    alpha_jumps: List[float] = []
    for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
        z = float(depth_values[i]) if np.isfinite(depth_values[i]) else z_low
        z_norm = _depth_norm(z, z_low, z_high, eps)
        raw_t, raw_a = _style_from_depth(z_norm, cfg)
        raw_a *= semantic_multiplier
        if smooth_thick is None:
            smooth_thick = float(raw_t)
            smooth_alpha = float(raw_a)
        else:
            smooth_thick = style_beta * smooth_thick + (1.0 - style_beta) * float(raw_t)
            next_alpha = style_beta * float(smooth_alpha) + (1.0 - style_beta) * float(raw_a)
            alpha_jumps.append(abs(next_alpha - float(smooth_alpha)))
            smooth_alpha = next_alpha
        w = max(1, int(round(float(smooth_thick))))
        a = max(0.0, min(1.0, float(smooth_alpha)))
        overlay = canvas.copy()
        cv2.line(overlay, p0, p1, color, w, cv2.LINE_AA)
        cv2.addWeighted(overlay, a, canvas, 1.0 - a, 0.0, dst=canvas)
        thicknesses.append(float(w))
        norms.append(float(z_norm))

    mono = 1.0
    if len(thicknesses) >= 2:
        v = np.asarray(thicknesses, dtype=np.float32)
        d = np.asarray(norms, dtype=np.float32)
        corr = np.corrcoef(v, d)[0, 1] if np.std(v) > 1e-6 and np.std(d) > 1e-6 else 0.0
        mono = float(max(0.0, min(1.0, 1.0 - max(0.0, corr))))
    flicker = float(np.mean(alpha_jumps)) if alpha_jumps else 0.0
    return {"thickness_depth_monotonicity": mono, "temporal_flicker_proxy": flicker}


def _interp_state(points: List[Dict[str, Any]], t_s: float) -> Dict[str, float]:
    if not points:
        return {"x_px": 0.0, "y_px": 0.0, "theta_rad": 0.0, "z_m": 0.0}
    if t_s <= float(points[0].get("t_s", 0.0)):
        return dict(points[0])
    if t_s >= float(points[-1].get("t_s", 0.0)):
        return dict(points[-1])
    for a, b in zip(points, points[1:]):
        ta = float(a.get("t_s", 0.0))
        tb = float(b.get("t_s", ta))
        if ta <= t_s <= tb and tb > ta:
            u = (t_s - ta) / (tb - ta)
            return {
                "x_px": (1.0 - u) * float(a.get("x_px", 0.0)) + u * float(b.get("x_px", 0.0)),
                "y_px": (1.0 - u) * float(a.get("y_px", 0.0)) + u * float(b.get("y_px", 0.0)),
                "theta_rad": (1.0 - u) * float(a.get("theta_rad", 0.0)) + u * float(b.get("theta_rad", 0.0)),
                "z_m": (1.0 - u) * float(a.get("z_m", 0.0) or 0.0) + u * float(b.get("z_m", 0.0) or 0.0),
                "t_s": t_s,
            }
    return dict(points[-1])


def _active_motion(component: Dict[str, Any], t_s: float) -> str:
    def _terms(value: Any) -> List[str]:
        txt = str(value or "").strip().lower()
        if not txt:
            return []
        for sep in ("_or_", "/", "|", ",", ";"):
            txt = txt.replace(sep, " ")
        out: List[str] = []
        for tok in txt.split():
            t = tok.strip()
            if t and t not in out:
                out.append(t)
        return out

    def _vocab() -> List[str]:
        out: List[str] = []
        for rec in list(component.get("timeline_records") or []):
            for tok in _terms((rec or {}).get("motion", "")):
                if tok not in out:
                    out.append(tok)
        for cand in list(component.get("motion_mode_candidates") or []):
            for tok in _terms((cand or {}).get("motion", "")):
                if tok not in out:
                    out.append(tok)
        ctx = component.get("action_context") if isinstance(component.get("action_context"), dict) else {}
        rc = ctx.get("animation_render_contract") if isinstance(ctx.get("animation_render_contract"), dict) else {}
        for item in list(rc.get("motion_labels") or []):
            for tok in _terms(item):
                if tok not in out:
                    out.append(tok)
        return out

    def _norm(motion: str, vocab: List[str]) -> str:
        tokens = _terms(motion)
        if not tokens:
            return vocab[0] if vocab else "traverse"
        joined = "_".join(tokens)
        if joined in vocab:
            return joined
        for tok in tokens:
            if tok in vocab:
                return tok
        return tokens[0]

    vocab = _vocab()
    timeline = list(component.get("timeline_records") or [])
    for rec in timeline:
        t0 = float(rec.get("t0_s", 0.0))
        t1 = float(rec.get("t1_s", t0))
        if t0 <= t_s <= t1:
            return _norm(str(rec.get("motion", "")), vocab)
    cand = list(component.get("motion_mode_candidates") or [])
    if cand:
        return _norm(str(cand[0].get("motion", "")), vocab)
    return vocab[0] if vocab else "traverse"


def _traj_pixel_points(traj_pts: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for st in traj_pts:
        out.append((int(round(float(st.get("x_px", 0.0)))), int(round(float(st.get("y_px", 0.0))))))
    return out


def _draw_trajectory_depth_tapered(
    canvas: np.ndarray,
    traj_pts: List[Dict[str, Any]],
    dm: Optional[np.ndarray],
    cfg: Any,
    color: Tuple[int, int, int],
    sem_mult: float,
    beta: float,
) -> None:
    pts = _traj_pixel_points(traj_pts)
    if len(pts) < 2:
        return
    depth_vals: List[float] = []
    if dm is not None:
        for x, y in pts:
            depth_vals.append(_depth_at(dm, x, y))
    else:
        depth_vals = [0.0 for _ in pts]
    _draw_depth_tapered_polyline(canvas, pts, depth_vals, color, cfg, semantic_multiplier=sem_mult, style_beta=beta)


def _make_procedural_bunny(size: int = 112) -> np.ndarray:
    """Simple BGR bunny (black background) for QA when sprite sheets are absent."""
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, int(size * 0.55)
    thick = max(2, size // 56)
    outline = (0, 0, 0)
    # Saturated fills for visibility on varied backgrounds; dark stroke for silhouette.
    body = (255, 100, 255)
    head = (255, 165, 255)
    ears = (255, 85, 240)
    nose = (255, 255, 255)
    # body
    cv2.ellipse(tile, (cx, cy), (int(size * 0.22), int(size * 0.16)), 0, 0, 360, body, -1, cv2.LINE_AA)
    cv2.ellipse(tile, (cx, cy), (int(size * 0.22), int(size * 0.16)), 0, 0, 360, outline, thick, cv2.LINE_AA)
    # head
    hr = int(size * 0.14)
    cv2.circle(tile, (cx, int(size * 0.32)), hr, head, -1, cv2.LINE_AA)
    cv2.circle(tile, (cx, int(size * 0.32)), hr, outline, thick, cv2.LINE_AA)
    # ears
    for sign in (-1, 1):
        cv2.ellipse(
            tile,
            (cx + sign * size // 6, int(size * 0.14)),
            (size // 18, size // 4),
            sign * 18,
            0,
            360,
            ears,
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            tile,
            (cx + sign * size // 6, int(size * 0.14)),
            (size // 18, size // 4),
            sign * 18,
            0,
            360,
            outline,
            thick,
            cv2.LINE_AA,
        )
    # nose
    cv2.circle(tile, (cx, int(size * 0.34)), max(2, size // 35), nose, -1, cv2.LINE_AA)
    return tile


_BUNNY_TILE: Optional[np.ndarray] = None


def _bunny_tile() -> np.ndarray:
    global _BUNNY_TILE
    if _BUNNY_TILE is None:
        _BUNNY_TILE = _make_procedural_bunny()
    return _BUNNY_TILE


def _paste_rotated_bgr_alpha(
    canvas: np.ndarray,
    sprite_bgr: np.ndarray,
    cx: float,
    cy: float,
    theta_rad: float,
    scale: float,
    alpha: float,
) -> None:
    """Paste sprite with rotation, scale, and global alpha (black = transparent)."""
    if sprite_bgr is None or alpha <= 1e-3:
        return
    h_img, w_img = canvas.shape[:2]
    sh, sw = sprite_bgr.shape[:2]
    ang_deg = -math.degrees(theta_rad)
    M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), ang_deg, float(scale))
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = max(4, int((sh * sin + sw * cos)))
    nh = max(4, int((sh * cos + sw * sin)))
    M[0, 2] += (nw / 2.0) - sw / 2.0
    M[1, 2] += (nh / 2.0) - sh / 2.0
    warped = cv2.warpAffine(
        sprite_bgr,
        M,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    x0 = int(round(cx - nw / 2.0))
    y0 = int(round(cy - nh / 2.0))
    x1 = x0 + nw
    y1 = y0 + nh
    if x1 <= 0 or y1 <= 0 or x0 >= w_img or y0 >= h_img:
        return
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = nw - max(0, x1 - w_img)
    sy1 = nh - max(0, y1 - h_img)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    crop = warped[sy0:sy1, sx0:sx1]
    if crop.size == 0:
        return
    roi = canvas[dy0 : dy0 + crop.shape[0], dx0 : dx0 + crop.shape[1]]
    mask = (np.max(crop, axis=2) > 10).astype(np.float32) * float(alpha)
    for c in range(3):
        roi[:, :, c] = (roi[:, :, c] * (1.0 - mask) + crop[:, :, c].astype(np.float32) * mask).astype(np.uint8)


def _draw_delimiter_frame(
    canvas: np.ndarray,
    *,
    title: str,
    lines: List[str],
    accent_bgr: Tuple[int, int, int],
    progress: float,
) -> None:
    """Full-frame delimiter so reviewers see exact boundaries between candidates."""
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0.0, dst=canvas)
    cv2.rectangle(canvas, (0, 0), (w, 10), accent_bgr, -1)
    cv2.rectangle(canvas, (0, h - 10), (w, h), accent_bgr, -1)
    bar_w = int(max(0, min(1.0, progress)) * (w - 40))
    cv2.rectangle(canvas, (20, h - 28), (20 + bar_w, h - 18), accent_bgr, -1)
    cv2.putText(canvas, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    y = 92
    for ln in lines[:10]:
        cv2.putText(canvas, ln[:110], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        y += 28


def _load_scene_context_for_animation_qa(paths_root_dir: Path) -> Dict[str, Any]:
    """Load sibling scene_graph metadata + light stats from relations/hierarchy JSON."""
    out: Dict[str, Any] = {
        "scene_json": "",
        "loaded": False,
        "relations_count": 0,
        "mask_hierarchy_count": 0,
        "region_hierarchy_count": 0,
        "layers_loaded": False,
        "hierarchy_scale_mult": 1.0,
        "fallback_artifacts_loaded": [],
        "scene_affordance_regions": 0,
        "object_affordance_count": 0,
        "mask_affordance_count": 0,
        "caption_record_count": 0,
    }
    name = paths_root_dir.name
    if not name.endswith("_paths"):
        return out
    stem = name[: -len("_paths")]
    track_dir = paths_root_dir.parent
    scene_path = track_dir / f"{stem}_scene.json"
    if not scene_path.exists():
        alt = paths_root_dir.parent / f"{stem}_scene.json"
        if alt.exists():
            scene_path = alt
    if not scene_path.exists():
        fallback_specs = {
            "scene_affordances": track_dir / f"{stem}_scene_affordances.json",
            "object_affordances": track_dir / f"{stem}_object_affordances.json",
            "mask_affordances": track_dir / f"{stem}_mask_affordances.json",
            "caption_evidence": track_dir / f"{stem}_caption_evidence.json",
            "region_relations": track_dir / f"{stem}_region_relations.json",
            "mask_hierarchy": track_dir / f"{stem}_mask_hierarchy_detailed.json",
            "mask_hierarchy_levels": track_dir / f"{stem}_mask_hierarchy_levels.json",
            "layers": track_dir / f"{stem}_layers.json",
        }
        loaded_names: List[str] = []
        for name_key, path in fallback_specs.items():
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            loaded_names.append(name_key)
            if name_key == "scene_affordances" and isinstance(data, dict):
                out["scene_affordance_regions"] = len(data.get("regions") or [])
                out["relations_count"] = max(int(out["relations_count"]), len(data.get("relations") or []))
            elif name_key == "object_affordances" and isinstance(data, dict):
                out["object_affordance_count"] = len(data.get("objects") or [])
            elif name_key == "mask_affordances" and isinstance(data, dict):
                out["mask_affordance_count"] = len(data.get("masks") or [])
            elif name_key == "caption_evidence" and isinstance(data, dict):
                out["caption_record_count"] = len(data.get("records") or [])
            elif name_key == "region_relations" and isinstance(data, dict):
                out["relations_count"] = len(data.get("relations") or [])
            elif name_key.startswith("mask_hierarchy") and isinstance(data, dict):
                out["mask_hierarchy_count"] = max(int(out["mask_hierarchy_count"]), len(data.get("levels") or data.get("hierarchy") or data))
            elif name_key == "layers":
                out["layers_loaded"] = True
        out["fallback_artifacts_loaded"] = loaded_names
        out["loaded"] = bool(loaded_names)
        out["hierarchy_scale_mult"] = float(
            1.0
            + 0.02 * (1.0 if out["layers_loaded"] else 0.0)
            + 0.01 * min(5.0, float(out["relations_count"]) / 40.0)
            + 0.01 * min(5.0, float(out["scene_affordance_regions"]) / 25.0)
        )
        return out
    try:
        root = json.loads(scene_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    meta = root.get("metadata") if isinstance(root, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    out["scene_json"] = str(scene_path)
    out["loaded"] = True
    output_root = paths_root_dir.parent.parent.parent

    def _try_json(key: str) -> Any:
        rel = meta.get(key)
        if not rel or not isinstance(rel, str) or not str(rel).strip():
            return None
        p = output_root / rel
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    rel = _try_json("relations_json")
    if isinstance(rel, list):
        out["relations_count"] = len(rel)
    mh = _try_json("mask_hierarchy_json")
    if isinstance(mh, dict):
        nodes = mh.get("nodes")
        out["mask_hierarchy_count"] = len(nodes) if isinstance(nodes, list) else len(mh)
    rh = _try_json("region_hierarchy_json")
    if isinstance(rh, dict):
        nodes = rh.get("nodes")
        out["region_hierarchy_count"] = len(nodes) if isinstance(nodes, list) else len(rh)
    ly = _try_json("layers_json")
    out["layers_loaded"] = ly is not None
    out["hierarchy_scale_mult"] = float(
        1.0 + 0.02 * (1.0 if out["layers_loaded"] else 0.0) + 0.01 * min(5.0, float(out["relations_count"]) / 40.0)
    )
    return out


def _stable_accent_bgr(pid: str) -> Tuple[int, int, int]:
    hsh = abs(hash(str(pid)))
    return (
        int(40 + (hsh % 180)),
        int(60 + ((hsh // 5) % 160)),
        int(120 + ((hsh // 25) % 120)),
    )


def _open_video_writer(path: Path, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    """Open MP4 writer with codec fallback for better playback compatibility."""
    for codec in ("avc1", "mp4v", "MJPG"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _as_point_list(rows: Any, *, limit: int = 64) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for row in list(rows or [])[:limit]:
        if isinstance(row, dict) and isinstance(row.get("position_px"), list) and len(row.get("position_px") or []) >= 2:
            xy = row["position_px"]
        else:
            xy = row
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            try:
                pts.append((int(round(float(xy[0]))), int(round(float(xy[1])))))
            except (TypeError, ValueError):
                continue
    return pts


def _path_render_polyline(path: Dict[str, Any], *, limit: int = 128) -> List[Tuple[int, int]]:
    return _as_point_list(
        path.get("display_polyline_2d")
        or path.get("polyline_2d_validated")
        or path.get("polyline_2d_reprojected")
        or path.get("polyline_2d"),
        limit=limit,
    )


def _contract_from_path_or_comp(path: Dict[str, Any], comp: Dict[str, Any]) -> Dict[str, Any]:
    ctx = comp.get("action_context") if isinstance(comp.get("action_context"), dict) else {}
    for candidate in (
        path.get("animation_render_contract"),
        comp.get("animation_render_contract"),
        ctx.get("animation_render_contract"),
    ):
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _draw_pulse_ring(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    alpha: float,
    thickness: int = 2,
) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, max(1, radius), color, max(1, thickness), cv2.LINE_AA)
    cv2.addWeighted(overlay, max(0.0, min(1.0, alpha)), frame, 1.0 - max(0.0, min(1.0, alpha)), 0.0, dst=frame)


def _draw_manifold_action_primitive(
    frame: np.ndarray,
    path: Dict[str, Any],
    comp: Dict[str, Any],
    manifold: str,
    progress: float,
    bx: float,
    by: float,
    heading_rad: float,
    color: Tuple[int, int, int],
) -> None:
    """Render the action manifold itself, not only the moving character."""
    h, w = frame.shape[:2]
    contract = _contract_from_path_or_comp(path, comp)
    anchors = contract.get("host_mask_contact_anchors") if isinstance(contract.get("host_mask_contact_anchors"), dict) else {}
    pulse = 0.5 + 0.5 * math.sin(progress * math.tau)
    center = (int(round(max(0, min(w - 1, bx)))), int(round(max(0, min(h - 1, by)))))
    ang = -math.degrees(float(heading_rad))
    hx = math.cos(float(heading_rad))
    hy = math.sin(float(heading_rad))

    if manifold == "contact_patch":
        contact_pts = _as_point_list(anchors.get("contact_points") or path.get("contact_points"), limit=8)
        approach_pts = _as_point_list(anchors.get("approach_points") or path.get("approach_points"), limit=4)
        for pt in contact_pts[:3]:
            overlay = frame.copy()
            cv2.ellipse(overlay, pt, (15, 6), ang, 0, 360, (80, 235, 235), -1, cv2.LINE_AA)
            cv2.ellipse(overlay, pt, (15, 6), ang, 0, 360, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.52, frame, 0.48, 0.0, dst=frame)
        if approach_pts and contact_pts:
            overlay = frame.copy()
            cv2.arrowedLine(overlay, approach_pts[0], contact_pts[0], (80, 235, 235), 3, cv2.LINE_AA, tipLength=0.24)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, dst=frame)
        return

    if manifold == "occlusion_pulse":
        anchor_pts = _as_point_list(anchors.get("anchor_points") or path.get("anchor_points"), limit=12)
        if not anchor_pts:
            anchor_pts = [center]
        for pt in anchor_pts:
            _draw_pulse_ring(frame, pt, int(5 + 12 * pulse), (70, 145, 255), 0.30 + 0.35 * pulse, 2)
        if len(anchor_pts) >= 2:
            overlay = frame.copy()
            cv2.polylines(overlay, [np.asarray(anchor_pts, dtype=np.int32)], False, (70, 145, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.42, frame, 0.58, 0.0, dst=frame)
        return

    if manifold == "portal_path":
        entry = anchors.get("portal_entry_uv") or path.get("portal_entry_uv")
        entry_pts = _as_point_list([entry], limit=1)
        target = entry_pts[0] if entry_pts else center
        for k in range(3):
            _draw_pulse_ring(frame, target, int(10 + 12 * k + 16 * progress), (175, 95, 255), 0.22 + 0.18 * (1.0 - k / 3.0), 2)
        overlay = frame.copy()
        cv2.line(overlay, center, target, (175, 95, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0.0, dst=frame)
        return

    if manifold in {"blob_path", "interior_path", "volume_path"}:
        pts = _as_point_list(path.get("volume_samples_2d") or path.get("display_polyline_2d") or path.get("polyline_2d"), limit=48)
        if len(pts) >= 3:
            overlay = frame.copy()
            cv2.polylines(overlay, [np.asarray(pts, dtype=np.int32)], manifold != "volume_path", (120, 210, 255) if manifold == "volume_path" else (230, 170, 70), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, dst=frame)
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            center,
            (int(16 + 5 * pulse), int(8 + 2 * pulse)),
            ang,
            0,
            360,
            (120, 210, 255) if manifold == "volume_path" else (230, 170, 70),
            2,
            cv2.LINE_AA,
        )
        tip = (int(round(center[0] + hx * 14.0)), int(round(center[1] + hy * 14.0)))
        cv2.arrowedLine(overlay, center, tip, (240, 240, 240), 1, cv2.LINE_AA, tipLength=0.32)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, dst=frame)
        return

    if manifold == "effect_field":
        effect_center = anchors.get("effect_center_uv") or path.get("effect_center_uv")
        pts = _as_point_list([effect_center], limit=1)
        c = pts[0] if pts else center
        radius = int(float((contract.get("effect_parameters") or {}).get("oscillation_radius_px") or path.get("oscillation_radius_px") or 10))
        for k in range(3):
            _draw_pulse_ring(frame, c, int(radius + k * 10 + 8 * pulse), (230, 80, 210), 0.28, 2)
        return

    if manifold == "contour_path":
        pts = _path_render_polyline(path, limit=96)
        if len(pts) >= 3:
            overlay = frame.copy()
            cv2.polylines(overlay, [np.asarray(pts, dtype=np.int32)], False, (255, 170, 90), 3, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, dst=frame)


def _draw_motion_style_overlay(
    frame: np.ndarray,
    motion_style: str,
    center: Tuple[int, int],
    heading_rad: float,
    progress: float,
) -> None:
    style_key = str(motion_style or "").strip().lower()
    overlay = frame.copy()
    cx, cy = int(center[0]), int(center[1])
    hx = math.cos(float(heading_rad))
    hy = math.sin(float(heading_rad))
    if style_key == "fluid":
        for k in range(3):
            r = int(6 + 6 * k + 4 * (0.5 + 0.5 * math.sin((progress + 0.18 * k) * math.tau)))
            cv2.ellipse(overlay, (cx, cy), (r, max(2, int(r * 0.45))), -math.degrees(heading_rad), 0, 360, (255, 190, 80), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0.0, dst=frame)
    elif style_key == "aerial":
        wing = int(10 + 4 * math.sin(progress * math.tau * 2.0))
        p0 = (int(round(cx - hx * 3.0)), int(round(cy - hy * 3.0)))
        left = (int(round(cx - hy * wing)), int(round(cy + hx * wing)))
        right = (int(round(cx + hy * wing)), int(round(cy - hx * wing)))
        cv2.line(overlay, p0, left, (220, 220, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, p0, right, (220, 220, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0.0, dst=frame)


def _ground_motion_style(path: Dict[str, Any], comp: Dict[str, Any]) -> str:
    cls = {}
    if isinstance(path.get("ground_object_classification"), dict):
        cls = dict(path.get("ground_object_classification") or {})
    elif isinstance(comp.get("ground_object_classification"), dict):
        cls = dict(comp.get("ground_object_classification") or {})
    elif isinstance((comp.get("action_context") if isinstance(comp.get("action_context"), dict) else {}).get("ground_object_classification"), dict):
        cls = dict((comp.get("action_context") or {}).get("ground_object_classification") or {})
    sw = float(cls.get("swimmable_fraction", 0.0) or 0.0)
    air = float(cls.get("open_air_fraction", 0.0) or 0.0)
    walk = float(cls.get("walkable_fraction", 0.0) or 0.0)
    if sw >= max(air, walk) and sw >= 0.20:
        return "fluid"
    if air >= max(sw, walk) and air >= 0.20:
        return "aerial"
    return "ground"


def export_animation_qa(
    *,
    paths_root_dir: Path,
    img_bgr: np.ndarray,
    metric_depth_m: Optional[np.ndarray],
    cfg: Any,
    region_label_map: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not bool(getattr(cfg, "path_animation_qa_enabled", True)):
        return out

    man_path = paths_root_dir / "path_atlas_manifest.json"
    comp_path = paths_root_dir / "animation_components.json"
    hyp_path = paths_root_dir / "path_hypotheses.json"
    if not (man_path.exists() and comp_path.exists() and hyp_path.exists()):
        return out

    manifest = _load_json(man_path)
    components_bundle = _load_json(comp_path)
    path_hyps = _load_json(hyp_path)
    path_rows = list(path_hyps.get("hypotheses") or path_hyps.get("paths") or [])
    path_by_id = {str(p.get("path_id", "")): p for p in path_rows if isinstance(p, dict)}
    comp_by_path = {
        str(c.get("path_id_linked", "")): c
        for c in (components_bundle.get("components") or [])
        if str(c.get("path_id_linked", "")).strip()
    }

    # Load live animation_plan.json — takes precedence over stale animation_components.json
    # for kinematic signatures and motion mode candidates.
    anim_plan_path = paths_root_dir / "animation_plan.json"
    live_plan_by_pid: Dict[str, Any] = {}
    if anim_plan_path.exists():
        try:
            anim_plan = _load_json(anim_plan_path)
            for plan_path in (anim_plan.get("paths") or []):
                pid_key = str(plan_path.get("path_id") or plan_path.get("trajectory_id") or "")
                if pid_key:
                    live_plan_by_pid[pid_key] = plan_path
        except Exception:
            pass
    sprite_registry = _load_sprite_registry(cfg)
    use_fallback_bunny = bool(getattr(cfg, "path_animation_qa_procedural_bunny_fallback", True))
    follow_path = bool(getattr(cfg, "path_animation_qa_follow_path_geometry", True))
    delim_s = float(getattr(cfg, "path_animation_qa_delimiter_seconds", 0.55))
    base_scale = float(getattr(cfg, "path_animation_qa_character_base_scale", 0.42))
    depth_scale_span = float(getattr(cfg, "path_animation_qa_depth_scale_span", 0.35))

    panels = list(manifest.get("panels") or [])
    path_rows = list(manifest.get("paths") or [])
    modes = [int(x) for x in (getattr(cfg, "path_animation_qa_modes", [24, 120]) or [24, 120])]
    max_per_panel = int(getattr(cfg, "path_animation_qa_max_candidates_per_panel", 10))
    seconds_per_candidate = float(getattr(cfg, "path_animation_qa_candidate_seconds", 4.0))
    render_debug = bool(getattr(cfg, "path_animation_qa_render_debug_overlays", True))

    dm = np.asarray(metric_depth_m, dtype=np.float32) if metric_depth_m is not None else None
    z_scene_lo, z_scene_hi = _scene_depth_bounds(dm, cfg) if dm is not None else (0.0, 1.0)
    scene_ctx = _load_scene_context_for_animation_qa(paths_root_dir)
    hier_scale = float(scene_ctx.get("hierarchy_scale_mult", 1.0))
    idle_follow_path = bool(getattr(cfg, "path_animation_qa_idle_follow_path", True))

    for fps in modes:
        subdir_name = (
            str(getattr(cfg, "path_animation_qa_output_subdir_120", "animation_qa_120"))
            if fps >= 120
            else str(getattr(cfg, "path_animation_qa_output_subdir_24", "animation_qa_24"))
        )
        mode_dir = paths_root_dir / subdir_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        beta = float(
            getattr(cfg, "path_animation_style_beta_120", 0.70)
            if fps >= 120
            else getattr(cfg, "path_animation_style_beta_24", 0.35)
        )
        trail_n = int(
            getattr(cfg, "path_animation_trail_samples_120", 20)
            if fps >= 120
            else getattr(cfg, "path_animation_trail_samples_24", 6)
        )
        trail_decay = float(getattr(cfg, "path_animation_trail_decay", 0.80))

        mode_scores: Dict[str, Any] = {"panels": [], "fps": fps}
        mode_manifest: Dict[str, Any] = {
            "schema": "citv_animation_qa_manifest_v2",
            "fps": fps,
            "panel_videos": [],
            "scene_context": scene_ctx,
            "animation_qa_scene_context_ingested": bool(scene_ctx.get("loaded")),
        }

        for pinfo in panels:
            pidx = int(pinfo.get("index", 0))
            panel_rows = [r for r in path_rows if int(r.get("panel_index", -1)) == pidx]
            panel_rows = sorted(panel_rows, key=lambda r: int(r.get("path_num", 10**9)))
            if max_per_panel > 0:
                panel_rows = panel_rows[:max_per_panel]
            panel_path_ids = [str(r.get("path_id", "")) for r in panel_rows if str(r.get("path_id", "")).strip()]

            video_path = mode_dir / f"panel_{pidx:02d}_paths_trajectories.mp4"
            h, w = img_bgr.shape[:2]
            writer = _open_video_writer(video_path, float(fps), (w, h))
            if writer is None:
                continue

            panel_metric_rows: List[Dict[str, Any]] = []
            segment_manifest: List[Dict[str, Any]] = []
            frame_cursor = 0
            n_delim = max(0, int(round(delim_s * fps)))
            for cand_i, pid in enumerate(panel_path_ids):
                path = path_by_id.get(pid) or {}
                comp = comp_by_path.get(pid) or {}
                # Prefer live animation_plan kinematic signatures over stale components.
                live_plan = live_plan_by_pid.get(pid) or {}
                if live_plan:
                    comp = {**comp, **{k: live_plan[k] for k in (
                        "timeline_records", "motion_mode_candidates",
                        "kinematic_signatures", "confidence_breakdown",
                        "relation_guardrails",
                        "path_shape_contract", "animation_render_contract",
                        "direction_profile", "acceptance_status", "rejection_reasons",
                        "uncertainty_reasons", "contradiction_reasons", "grounding_evidence",
                        "path_geometry_quality", "geometry_smoothing_provenance",
                    ) if k in live_plan}}
                traj_id = str(live_plan.get("trajectory_id") or comp.get("trajectory_id", "") or "")
                path_pts = _path_render_polyline(path, limit=256)
                traj_pts = [dict(x) for x in (comp.get("trajectory_points") or [])]
                if len(path_pts) < 2:
                    continue
                manifold = str(comp.get("manifold_type") or path.get("manifold_type") or "ribbon_path")
                render_contract = _contract_from_path_or_comp(path, comp)
                shape_contract = (
                    path.get("path_shape_contract")
                    or comp.get("path_shape_contract")
                    or ((comp.get("action_context") or {}).get("path_shape_contract") if isinstance(comp.get("action_context"), dict) else {})
                    or {}
                )
                direction_profile = (
                    render_contract.get("direction_profile")
                    or shape_contract.get("direction_profile")
                    or {}
                )

                if cand_i > 0 and n_delim > 0:
                    accent = _stable_accent_bgr(pid)
                    for di in range(n_delim):
                        prog = (di + 1) / max(1, n_delim)
                        fr = img_bgr.copy()
                        regions = path.get("regions_traversed") or []
                        en = path.get("energy_terms") or {}
                        lines = [
                            f"Candidate {cand_i + 1} / {len(panel_path_ids)}",
                            f"path_id: {pid}",
                            f"trajectory_id: {traj_id or '(unlinked)'}",
                            f"semantic_valid={bool(path.get('semantic_valid', True))} score={float(path.get('semantic_validity_score', 0.0)):.2f}",
                            f"regions: {', '.join(map(str, regions[:6]))}{'...' if len(regions) > 6 else ''}",
                            f"E_total={float(en.get('E_total', 0.0)):.3f}",
                        ]
                        _draw_delimiter_frame(
                            fr,
                            title="NEXT PATH / TRAJECTORY",
                            lines=lines,
                            accent_bgr=accent,
                            progress=prog,
                        )
                        writer.write(fr)
                        frame_cursor += 1

                cum, total_len = _polyline_cumulative_lengths(path_pts)
                depth_vals_path: List[float] = []
                if dm is not None:
                    for x, y in path_pts:
                        depth_vals_path.append(_depth_at(dm, x, y))
                else:
                    depth_vals_path = [float("nan") for _ in path_pts]

                sem_mult = _semantic_feasibility(comp) if comp else 1.0
                sem_mult *= _path_semantic_multiplier(path)

                n_frames = max(1, int(round(seconds_per_candidate * fps)))
                horizon = float(max([float(x.get("t_s", 0.0)) for x in traj_pts], default=0.16))
                horizon = max(horizon, 1e-3)

                seg_frame0 = frame_cursor
                metric_acc: List[Dict[str, float]] = []
                motion_style = _ground_motion_style(path, comp)
                for fi in range(n_frames):
                    s = fi / max(1, n_frames - 1)
                    t_motion = s * horizon
                    motion = _active_motion(comp, t_motion)
                    manifold_follow = manifold in {"ribbon_path", "centerline_path", "contour_path", "portal_path"}
                    s_path = 0.0 if (follow_path and idle_follow_path and motion == "idle") else s
                    if follow_path:
                        if manifold_follow:
                            bx, by, ptheta = _polyline_sample(path_pts, cum, total_len, s_path)
                        else:
                            # Non-line manifolds are rendered as local anchored motions.
                            bx, by, ptheta = _polyline_sample(path_pts, cum, total_len, 0.5)
                    else:
                        t_clip = (fi / max(1, n_frames - 1)) * horizon
                        st = _interp_state(traj_pts, t_clip)
                        bx, by, ptheta = float(st["x_px"]), float(st["y_px"]), float(st.get("theta_rad", 0.0))

                    if motion_style == "fluid":
                        by += 3.5 * math.sin((s * 3.0) * math.tau)
                        ptheta += 0.16 * math.sin((s * 2.5) * math.tau)
                    elif motion_style == "aerial":
                        by += 5.0 * math.sin((s * 2.0) * math.tau)
                        ptheta += 0.10 * math.sin((s * 1.6) * math.tau)

                    ix, iy = int(round(bx)), int(round(by))

                    frame = img_bgr.copy()
                    stats = _draw_depth_tapered_polyline(
                        frame, path_pts, depth_vals_path, (20, 220, 20), cfg, semantic_multiplier=sem_mult, style_beta=beta
                    )
                    metric_acc.append(stats)

                    if len(traj_pts) >= 2:
                        _draw_trajectory_depth_tapered(
                            frame, traj_pts, dm, cfg, (80, 170, 255), sem_mult, beta
                        )

                    if len(traj_pts) >= 2:
                        trail_states: List[Dict[str, float]] = []
                        for k in range(max(1, trail_n)):
                            t_back = max(0.0, t_motion - (k / max(1, fps)))
                            trail_states.append(_interp_state(traj_pts, t_back))
                        for k in range(len(trail_states) - 1):
                            a = trail_decay**k
                            p0 = (int(trail_states[k]["x_px"]), int(trail_states[k]["y_px"]))
                            p1 = (int(trail_states[k + 1]["x_px"]), int(trail_states[k + 1]["y_px"]))
                            overlay = frame.copy()
                            cv2.line(overlay, p0, p1, (60, 200, 255), 2, cv2.LINE_AA)
                            cv2.addWeighted(overlay, min(1.0, 0.75 * a), frame, 1.0 - min(1.0, 0.75 * a), 0.0, dst=frame)

                    _draw_manifold_action_primitive(
                        frame,
                        path,
                        comp,
                        manifold,
                        s,
                        bx,
                        by,
                        ptheta,
                        _stable_accent_bgr(pid),
                    )
                    _draw_motion_style_overlay(frame, motion_style, (ix, iy), ptheta, s)

                    z_foot = _depth_at(dm, ix, iy)
                    z_norm = _depth_norm(z_foot, z_scene_lo, z_scene_hi, float(getattr(cfg, "path_animation_depth_eps", 1e-6))) if np.isfinite(z_foot) else 0.45
                    reg_align = _region_feet_alignment(region_label_map, ix, iy, list(path.get("regions_traversed") or []))
                    # Plan §2.6: dim the sprite when the path's visibility profile
                    # marks the current vertex behind / partially occluded.
                    vis_mult = 1.0
                    try:
                        vp_list = list(path.get("visibility_profile") or [])
                        if vp_list:
                            vp_i = vp_list[min(len(vp_list) - 1, max(0, fi))] if isinstance(vp_list, list) else None
                            if isinstance(vp_i, dict):
                                vfrac = float(vp_i.get("visible_fraction", 1.0))
                                vis_mult = max(0.25, min(1.0, vfrac))
                    except Exception:
                        vis_mult = 1.0
                    char_alpha = float(max(0.25, min(1.0, sem_mult * reg_align * vis_mult * (0.55 + 0.45 * (1.0 - z_norm)))))
                    char_scale = float(
                        base_scale
                        * (1.0 + depth_scale_span * (1.0 - z_norm))
                        * (0.85 + 0.15 * reg_align)
                        * hier_scale
                    )

                    frames = sprite_registry.get(motion) or sprite_registry.get("walk") or []
                    if frames:
                        sp_i = int((fi * len(frames)) / max(1, n_frames)) % len(frames)
                        _paste_rotated_bgr_alpha(frame, frames[sp_i], bx, by, ptheta, char_scale, char_alpha)
                    elif use_fallback_bunny:
                        # Procedural fallback: keep near-opaque so depth/semantic alpha does not wash it out.
                        bunny_alpha = min(1.0, max(float(char_alpha), 0.92))
                        _paste_rotated_bgr_alpha(frame, _bunny_tile(), bx, by, ptheta, char_scale * 1.15, bunny_alpha)
                    else:
                        cv2.circle(frame, (ix, iy), max(4, int(8 * char_scale)), (255, 255, 255), -1, cv2.LINE_AA)

                    if render_debug:
                        hud = frame.copy()
                        cv2.rectangle(hud, (4, 4), (min(w - 4, 640), 128), (12, 12, 12), -1)
                        cv2.addWeighted(hud, 0.42, frame, 0.58, 0.0, dst=frame)
                        cv2.rectangle(frame, (4, 4), (min(w - 4, 640), 128), (70, 70, 70), 1)
                        cv2.putText(
                            frame,
                            f"P{pidx} | {pid} | traj={traj_id or '-'} | motion={motion}",
                            (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (250, 250, 250),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            f"arc_s={s:.2f} z_norm={z_norm:.2f} sem={sem_mult:.2f} reg={reg_align:.2f}",
                            (10, 44),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (220, 220, 220),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            f"GREEN=path | CYAN=trajectory | manifold={manifold}",
                            (10, 66),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (200, 230, 200),
                            1,
                            cv2.LINE_AA,
                        )
                        support_counts = dict((path.get("semantic_trace") or {}).get("support_kind_counts") or {})
                        support_txt = ",".join(f"{k}:{v}" for k, v in list(support_counts.items())[:4])
                        cv2.putText(
                            frame,
                            f"status={path.get('acceptance_status', '') or comp.get('acceptance_status', '')} | support={support_txt[:52]}",
                            (10, 88),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.43,
                            (210, 220, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            f"dir={direction_profile.get('image_direction', '')}/{direction_profile.get('camera_depth_trend', '')} | render={render_contract.get('render_primitive', '')[:42]}",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (220, 210, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    writer.write(frame)
                    frame_cursor += 1

                m_mono = float(np.mean([m.get("thickness_depth_monotonicity", 1.0) for m in metric_acc])) if metric_acc else 1.0
                m_flicker = float(np.mean([m.get("temporal_flicker_proxy", 0.0) for m in metric_acc])) if metric_acc else 0.0
                panel_metric_rows.append(
                    {
                        "path_id": pid,
                        "trajectory_id": traj_id,
                        "thickness_depth_monotonicity": m_mono,
                        "temporal_flicker_score": m_flicker,
                        "depth_semantic_coherence": float(max(0.0, min(1.0, sem_mult * m_mono))),
                    }
                )
                segment_manifest.append(
                    {
                        "path_id": pid,
                        "trajectory_id": traj_id,
                        "candidate_index": cand_i,
                        "frame_start": seg_frame0,
                        "frame_end_exclusive": frame_cursor,
                        "delimiter_frames_before": n_delim if cand_i > 0 else 0,
                        "follow_path_geometry": follow_path,
                        "manifold_type": manifold,
                        "motion_style": motion_style,
                        "render_primitive": str(render_contract.get("render_primitive", "")),
                        "direction_profile": dict(direction_profile),
                        "acceptance_status": str(path.get("acceptance_status", comp.get("acceptance_status", ""))),
                        "rejection_reasons": list(path.get("rejection_reasons") or comp.get("rejection_reasons") or []),
                        "uncertainty_reasons": list(path.get("uncertainty_reasons") or comp.get("uncertainty_reasons") or []),
                        "contradiction_reasons": list(path.get("contradiction_reasons") or comp.get("contradiction_reasons") or []),
                        "grounding_evidence": dict(path.get("grounding_evidence") or comp.get("grounding_evidence") or {}),
                        "path_geometry_quality": dict(path.get("path_geometry_quality") or comp.get("path_geometry_quality") or {}),
                        "geometry_smoothing_provenance": dict(path.get("geometry_smoothing_provenance") or comp.get("geometry_smoothing_provenance") or {}),
                        "sprite_motions_resolved": bool(sprite_registry),
                        "procedural_bunny_used": bool(not sprite_registry and use_fallback_bunny),
                    }
                )
            writer.release()
            rel_video = f"{subdir_name}/panel_{pidx:02d}_paths_trajectories.mp4"
            mode_manifest["panel_videos"].append(
                {
                    "panel_index": pidx,
                    "video": rel_video,
                    "candidate_count": len(panel_path_ids),
                    "segments": segment_manifest,
                }
            )
            mode_scores["panels"].append({"panel_index": pidx, "metrics": panel_metric_rows})
            if video_path.exists() and video_path.stat().st_size > 0:
                out[f"animation_qa_video_{fps}"] = rel_video

        man_out = mode_dir / "animation_qa_manifest.json"
        sco_out = mode_dir / "animation_qa_scores.json"
        man_out.write_text(json.dumps(mode_manifest, indent=2), encoding="utf-8")
        sco_out.write_text(json.dumps(mode_scores, indent=2), encoding="utf-8")
        out[f"animation_qa_manifest_{fps}_json"] = f"{subdir_name}/animation_qa_manifest.json"
        out[f"animation_qa_scores_{fps}_json"] = f"{subdir_name}/animation_qa_scores.json"
    return out
