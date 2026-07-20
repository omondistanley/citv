#!/usr/bin/env python3
"""
CITV Studio — Turn any photo into an animation.
Run:  ./venv/bin/python app.py
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import gradio as gr


# ────────────────────────────────────────────────────────────────────────────
# Palette + colour helpers
# ────────────────────────────────────────────────────────────────────────────

_PAL: List[Tuple[int, int, int]] = [
    (255,  82,  82), (30, 190, 255), (30, 230, 110), (255, 185,  0),
    (200,  50, 255), ( 0, 230, 200), (255, 240,  0), (255, 110, 200),
    (140, 255,  60), (255, 195, 100), (110, 150, 255), (255,  60, 140),
    ( 60, 210, 255), (255, 150,  50), (180, 255, 190), (255,  90, 210),
]
_ACTOR_COL  = (255, 220,  0)   # gold
_TARGET_COL = (  0, 230, 255)  # cyan

def _pal(i: int) -> Tuple[int, int, int]:
    return _PAL[int(i) % len(_PAL)]

def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ────────────────────────────────────────────────────────────────────────────
# Scene loading
# ────────────────────────────────────────────────────────────────────────────

def discover_scenes(root: str) -> List[str]:
    """Return sorted list of CITV paths-dirs (staged or grounded_sam2 format)."""
    found: set = set()
    for p in Path(root).rglob("scene_grounding_index.json"):
        found.add(str(p.parent))
    for p in Path(root).rglob("path_hypotheses.json"):
        found.add(str(p.parent))
    return sorted(found)

def _jload(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pth(p: Path) -> Optional[str]:
    return str(p) if p.is_file() else None

def _norm_entities_new(depth_mask_objs: List[Dict], sem_layer: Dict) -> List[Dict]:
    """Convert grounded_sam2 depth_mask objects → standard entity list."""
    sem_ents = {e["id"]: e for e in sem_layer.get("entities", [])} if isinstance(sem_layer, dict) else {}
    out = []
    for obj in depth_mask_objs:
        eid  = obj.get("id", "")
        lbl  = obj.get("label", eid)
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        role = sem_ents.get(eid, {}).get("role", "actor")
        out.append({
            "entity_id":  eid,
            "label":      lbl,
            "role":       role,
            "kind":       obj.get("entity_kind", "object"),
            "geometry":   {"bbox_xyxy": [x1, y1, x2, y2], "centroid_xy": [cx, cy]},
            "depth_z":    obj.get("depth_stats", {}).get("z_val", 3.0),
        })
    return out


def _norm_hypotheses_new(paths: List[Dict]) -> List[Dict]:
    """Convert grounded_sam2 path list → standard hypothesis list."""
    out = []
    for p in paths:
        src = p.get("source_entity", {})
        tgt = p.get("target_entity", {})
        out.append({
            **p,
            "depth_trace_m":      [],
            "kinematic_signatures": [],
            "source_entity":      src,
            "target_entity":      tgt,
        })
    return out


def load_scene(paths_dir: str) -> Dict[str, Any]:
    """Load all data from a CITV paths output directory (staged or grounded_sam2 format)."""
    base = Path(paths_dir).expanduser()
    empty: Dict[str, Any] = {"valid": False, "dir": str(base)}

    gi_path = base / "scene_grounding_index.json"
    ph_path = base / "path_hypotheses.json"

    # ── Staged format (old) ──────────────────────────────────────────────────
    if gi_path.is_file():
        gi = _jload(gi_path)
        if not gi:
            return empty
        ph = _jload(ph_path) or {}
        ac = _jload(base / "animation_components.json") or {}
        ap = _jload(base / "animation_plan.json") or {}

        stem    = gi.get("stem", base.stem.replace("_paths", ""))
        vis_dir = base.parent
        img_size = (ac.get("image_size", {}) if isinstance(ac, dict) else {})
        pipe_w   = int(img_size.get("width",  945))
        pipe_h   = int(img_size.get("height", 1024))

        entities: List[Dict]   = gi.get("entities", [])
        hypotheses: List[Dict] = ph.get("hypotheses", []) if isinstance(ph, dict) else []
        ac_comps = ac.get("components", []) if isinstance(ac, dict) else []
        relations = gi.get("relations", [])

    # ── grounded_sam2 format (new) ───────────────────────────────────────────
    elif ph_path.is_file():
        ph = _jload(ph_path) or {}
        ap = _jload(base / "animation_plan.json") or {}

        stem    = base.stem.replace("_paths", "")
        vis_dir = base.parent   # …/grounded_sam2/

        img_size = ph.get("image_size", {}) if isinstance(ph, dict) else {}
        pipe_w   = int(img_size.get("width",  945))
        pipe_h   = int(img_size.get("height", 1024))

        # Entities from depth_mask_A (has bbox for both objects and regions)
        dm_path = vis_dir / f"{stem}_depth_mask_A.json"
        dm = _jload(dm_path) or {}
        dm_objs = (dm.get("depth_mask") or {}).get("objects", [])

        # Also include plain scene objects as fallback
        sc_path = vis_dir / f"{stem}_scene.json"
        sc = _jload(sc_path) or {}
        if not dm_objs:
            dm_objs = sc.get("objects", [])

        sem = _jload(vis_dir / f"{stem}_semantic_layer.json") or \
              _jload(base / "semantic_layer.json") or {}

        entities   = _norm_entities_new(dm_objs, sem)
        raw_paths  = ph.get("paths", []) if isinstance(ph, dict) else []
        hypotheses = _norm_hypotheses_new(raw_paths)
        ac_comps   = []
        relations  = []

    else:
        return empty

    # ── Common: build indexes & find original image ──────────────────────────
    ent_by_id: Dict[str, Dict] = {e["entity_id"]: e for e in entities}

    hyp_index: Dict[Tuple[str, str], List[Dict]] = {}
    for h in hypotheses:
        src = (h.get("source_entity") or {}).get("id")
        tgt = (h.get("target_entity") or {}).get("id")
        if src and tgt:
            hyp_index.setdefault((src, tgt), []).append(h)

    prj_root = Path(__file__).parent
    orig: Optional[str] = None
    for search_dir in (base / "images", prj_root / "images", vis_dir):
        if not search_dir.is_dir():
            continue
        for ext in ("png", "jpg", "jpeg", "heic", "heif"):
            for fp in search_dir.glob(f"{stem}.{ext}"):
                orig = str(fp); break
            if orig:
                break
        if orig:
            break
    if not orig:
        for cand in (f"{stem}_sam2_tinted_overlay", f"{stem}_overlay",
                     f"{stem}_segmentation", f"{stem}_regions_overlay"):
            fp = vis_dir / f"{cand}.png"
            if fp.is_file():
                orig = str(fp); break

    def _vis(name: str) -> Optional[str]:
        for ext in (".png", ".jpg"):
            for d in (vis_dir, base):
                fp = d / f"{name}{ext}"
                if fp.is_file():
                    return str(fp)
        return None

    return {
        "valid":         True,
        "stem":          stem,
        "dir":           str(base),
        "paths_dir":     str(base),
        "vis_dir":       str(vis_dir),
        "pipe_w":        pipe_w,
        "pipe_h":        pipe_h,
        "entities":      entities,
        "ent_by_id":     ent_by_id,
        "relations":     relations,
        "hypotheses":    hypotheses,
        "hyp_index":     hyp_index,
        "anim_components": ac_comps,
        "anim_plan":     ap,
        "orig_image":    orig,
        # visualisations
        "img_overlay":        _vis(f"{stem}_overlay"),
        "img_seg":            _vis(f"{stem}_segmentation"),
        "img_regions":        _vis(f"{stem}_region_segmentation") or _vis(f"{stem}_regions_overlay"),
        "img_depth":          _vis(f"{stem}_depth_map") or _vis("../../../depth/" + stem + "_depth_map"),
        "img_relations":      _vis(f"{stem}_relations_map_objects"),
        "img_3d":             _vis(f"{stem}_3d_viz"),
        "img_path_overlay":   _pth(base / "path_overlay.png"),
        "img_trajectories":   _pth(base / "path_trajectories.png") or _pth(base / f"{stem}_path_map_all.png"),
        "img_manifold":       _pth(base / "action_manifold_overlay.png"),
        "img_trav":           _pth(base / "path_traversability_speed.png"),
        "img_motion":         _pth(base / "motion_contracts_overlay.png"),
        "img_support":        _pth(base / "support_trace_overlay.png"),
        "img_semantic":       _pth(base / "semantic_trace_overlay.png"),
        "img_visibility":     _pth(base / "visibility_profile_overlay.png"),
        "img_occlusion":      _pth(base / "occlusion_state_overlay.png"),
        "img_nav":            _pth(base / "navigation_zones.png"),
        "img_atlas":          _pth(base / "path_atlas_ranked_panel_01.png"),
        "img_atlas_ctx":      _pth(base / "path_atlas_ranked_panel_01_context.png"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Image helpers
# ────────────────────────────────────────────────────────────────────────────

def _open(path: Optional[str]) -> Optional[Image.Image]:
    if not path or not os.path.isfile(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def _base_image(scene: Dict) -> Optional[Image.Image]:
    """Return the scene image at pipeline coordinates (pipe_w × pipe_h)."""
    pw, ph = scene.get("pipe_w", 945), scene.get("pipe_h", 1024)
    img = (
        _open(scene.get("orig_image"))
        or _open(scene.get("img_overlay"))
        or _open(scene.get("img_seg"))
    )
    if img is None:
        return None
    if img.size != (pw, ph):
        img = img.resize((pw, ph), Image.LANCZOS)
    return img


# ────────────────────────────────────────────────────────────────────────────
# Scene annotator (bounding-box overlays)
# ────────────────────────────────────────────────────────────────────────────

def annotate_scene(
    scene: Dict,
    actor_id:  Optional[str],
    target_id: Optional[str],
) -> Optional[Image.Image]:
    """Return scene image with coloured entity boxes + legend."""
    base = _base_image(scene)
    if base is None:
        return None

    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    fbadge  = _font(max(10, W // 70))
    flabel  = _font(max(9,  W // 85))

    for idx, ent in enumerate(scene.get("entities", [])):
        eid   = ent["entity_id"]
        geo   = ent.get("geometry", {})
        bbox  = geo.get("bbox_xyxy")
        cxy   = geo.get("centroid_xy")
        label = ent.get("label", eid)

        is_a  = (eid == actor_id)
        is_t  = (eid == target_id)
        col   = _ACTOR_COL if is_a else (_TARGET_COL if is_t else _pal(idx))
        lw    = 4 if (is_a or is_t) else 2

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            draw.rectangle([x1, y1, x2, y2], fill=col + (40,), outline=col + (220,), width=lw)
            r = max(9, W // 75)
            bx, by = x1, max(0, y1 - r * 2 + 2)
            draw.ellipse([bx, by, bx + r*2, by + r*2], fill=col + (220,))
            draw.text((bx + r, by + r), str(idx), fill=(0, 0, 0), font=fbadge, anchor="mm")
            pfx   = "★ " if is_a else ("▶ " if is_t else "")
            short = (pfx + label[:18]).strip()
            draw.text((x1 + 2, y2 + 3), short, fill=col + (255,), font=flabel,
                      stroke_width=1, stroke_fill=(0, 0, 0, 180))
        elif cxy:
            cx, cy = int(cxy[0]), int(cxy[1])
            r = max(8, W // 80)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col + (180,), outline=col + (255,), width=lw)
            draw.text((cx, cy), str(idx), fill=(0, 0, 0), font=fbadge, anchor="mm")

    merged = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    # Legend bar
    lh = max(28, H // 22)
    legend = Image.new("RGB", (W, lh), (18, 18, 28))
    ld = ImageDraw.Draw(legend)
    x = 6
    for idx, ent in enumerate(scene.get("entities", [])[:16]):
        eid = ent["entity_id"]
        col = _ACTOR_COL if eid == actor_id else (_TARGET_COL if eid == target_id else _pal(idx))
        pfx = "★ " if eid == actor_id else ("▶ " if eid == target_id else f"{idx}: ")
        txt = pfx + ent.get("label", eid)[:12]
        ld.text((x, lh // 2), txt, fill=col, font=flabel, anchor="lm")
        try:
            tw = int(flabel.getlength(txt))
        except AttributeError:
            tw = len(txt) * 7
        x += tw + 10
        if x > W - 50:
            break

    out = Image.new("RGB", (W, H + lh))
    out.paste(merged, (0, 0))
    out.paste(legend, (0, H))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Hit-test: find entity under a click
# ────────────────────────────────────────────────────────────────────────────

def entity_at(x: float, y: float, scene: Dict) -> Optional[str]:
    """Return entity_id closest to pixel click (x, y), preferring insiders."""
    entities = scene.get("entities", [])
    inside_id, inside_area = None, float("inf")
    best_id,   best_dist   = None, float("inf")

    for ent in entities:
        geo  = ent.get("geometry", {})
        bbox = geo.get("bbox_xyxy")
        cxy  = geo.get("centroid_xy")
        eid  = ent["entity_id"]

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            area   = (x2 - x1) * (y2 - y1)
            if x1 <= x <= x2 and y1 <= y <= y2:
                if area < inside_area:
                    inside_area, inside_id = area, eid
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
        elif cxy:
            dist = math.sqrt((x - cxy[0])**2 + (y - cxy[1])**2)
        else:
            continue

        if dist < best_dist:
            best_dist, best_id = dist, eid

    return inside_id or best_id


# ────────────────────────────────────────────────────────────────────────────
# Path search
# ────────────────────────────────────────────────────────────────────────────

def find_paths(scene: Dict, a_id: str, b_id: str) -> List[Dict]:
    idx = scene.get("hyp_index", {})
    paths = list(idx.get((a_id, b_id), [])) + list(idx.get((b_id, a_id), []))
    return sorted(paths, key=lambda h: (h.get("scores") or {}).get("overall_confidence", 0), reverse=True)


# ────────────────────────────────────────────────────────────────────────────
# Text grounding
# ────────────────────────────────────────────────────────────────────────────

_MOTION: Dict[str, str] = {
    "walk": "walk", "run": "run", "sprint": "run", "jog": "run",
    "fly": "fly", "float": "fly", "glide": "fly", "soar": "fly",
    "jump": "jump", "leap": "jump", "hop": "jump",
    "climb": "climb", "crawl": "crawl", "swim": "swim", "slide": "slide",
    "hover": "hover", "drift": "hover",
    "interact": "interact", "touch": "interact", "grab": "interact", "use": "interact",
    "peek": "peek", "hide": "hide", "sneak": "careful",
    "approach": "walk", "move": "walk", "go": "walk", "head": "walk",
    "reach": "walk", "cross": "walk", "travel": "walk",
}

def ground_text(text: str, scene: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (actor_id, target_id, motion_hint) from free-form text."""
    words = re.split(r"\W+", text.lower())
    motion = next((_MOTION[w] for w in words if w in _MOTION), None)

    scored: List[Tuple[int, str]] = []
    for ent in scene.get("entities", []):
        lbl_words = set(re.split(r"\W+", ent.get("label", "").lower()))
        score     = sum(1 for w in words if len(w) > 2 and w in lbl_words)
        if score:
            scored.append((score, ent["entity_id"]))
    scored.sort(reverse=True)

    a_id = scored[0][1] if len(scored) > 0 else None
    b_id = scored[1][1] if len(scored) > 1 else None

    if a_id and not b_id:
        for (src, tgt) in scene.get("hyp_index", {}):
            if src == a_id:
                b_id = tgt
                break

    return a_id, b_id, motion


# ────────────────────────────────────────────────────────────────────────────
# Sprite extraction
# ────────────────────────────────────────────────────────────────────────────

def extract_sprite(scene: Dict, entity_id: str) -> Optional[Image.Image]:
    """
    Crop the actor's image region, apply feathered ellipse alpha.
    Returns RGBA PIL Image ready to paste.
    """
    ent  = scene.get("ent_by_id", {}).get(entity_id, {})
    geo  = ent.get("geometry", {})
    bbox = geo.get("bbox_xyxy")
    if not bbox or len(bbox) < 4:
        return None

    base = _base_image(scene)
    if base is None:
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    pad = max(4, (x2 - x1 + y2 - y1) // 8)
    x1c = max(0, x1 - pad)
    y1c = max(0, y1 - pad)
    x2c = min(base.width,  x2 + pad)
    y2c = min(base.height, y2 + pad)

    crop = base.crop([x1c, y1c, x2c, y2c]).convert("RGBA")
    cw, ch = crop.size

    # Feathered ellipse mask
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([0, 0, cw, ch], fill=255)
    blur_r = max(3, min(cw, ch) // 6)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
    crop.putalpha(mask)

    return crop


# ────────────────────────────────────────────────────────────────────────────
# Animation engine
# ────────────────────────────────────────────────────────────────────────────

_SPEED_MAP: Dict[str, float] = {
    "walk": 1.0, "run": 2.4, "sprint": 3.0, "jog": 1.8,
    "crawl": 0.35, "climb": 0.55, "descend": 0.75,
    "fly": 1.7, "hover": 0.25, "glide": 1.2,
    "jump": 2.2, "leap": 2.5, "hop": 1.5,
    "interact": 0.2, "idle": 0.1, "careful": 0.45,
    "peek": 0.3, "hide": 0.4,
}

_MOTION_ICON: Dict[str, str] = {
    "walk": "walk", "run": "run →", "fly": "fly ✦", "climb": "climb ↑",
    "crawl": "crawl", "jump": "jump ↑↑", "hover": "hover ~",
    "interact": "interact", "descend": "descend ↓", "careful": "careful ⚠",
    "peek": "peek", "hide": "hide",
}


def _depth_at_pt(pt_idx: int, n_pts: int, depth_pts: List[Dict]) -> float:
    """Interpolate z_m at path point index from sampled depth trace."""
    if not depth_pts:
        return 3.0
    s_target = pt_idx / max(n_pts - 1, 1)
    s_vals   = [d["s"] for d in depth_pts]
    z_vals   = [d["z_m"] for d in depth_pts]
    i = bisect_left(s_vals, s_target)
    if i == 0:
        return z_vals[0]
    if i >= len(s_vals):
        return z_vals[-1]
    t = (s_target - s_vals[i - 1]) / max(s_vals[i] - s_vals[i - 1], 1e-9)
    return z_vals[i - 1] * (1 - t) + z_vals[i] * t


def _motion_at(pt_idx: int, sigs: List[Dict]) -> str:
    for sig in sigs:
        if sig.get("start_idx", 0) <= pt_idx <= sig.get("end_idx", 0):
            return sig.get("motion", "walk")
    return "walk"


def _frame_schedule(n_pts: int, sigs: List[Dict], motion_override: Optional[str], n_frames: int) -> List[int]:
    """Return list[n_frames] mapping frame_i → path_point_index."""
    speeds = np.ones(n_pts, dtype=float)
    if motion_override and motion_override != "custom":
        speeds[:] = _SPEED_MAP.get(motion_override, 1.0)
    else:
        for sig in sigs:
            s_i = int(sig.get("start_idx", 0))
            e_i = min(int(sig.get("end_idx", n_pts)), n_pts)
            spd = _SPEED_MAP.get(sig.get("motion", "walk"), 1.0)
            speeds[s_i:e_i + 1] = spd

    inv   = 1.0 / np.maximum(speeds, 0.04)
    cum   = np.cumsum(inv)
    total = cum[-1]
    times = np.linspace(0, total, n_frames)
    return [min(int(np.searchsorted(cum, t)), n_pts - 1) for t in times]


def generate_animation(
    scene:           Dict,
    path_hyp:        Dict,
    actor_id:        str,
    target_id:       str,
    motion_override: Optional[str] = None,
    fps:             int   = 12,
    duration_s:      float = 3.5,
    max_width:       int   = 520,
    sprite_scale:    float = 1.0,
) -> Optional[str]:
    """
    Generate an animated GIF of the actor sprite traversing the path.
    Returns path to the output .gif file, or None on failure.
    """
    base = _base_image(scene)
    if base is None:
        return None

    # Scale down to max_width for manageable GIF size
    orig_W, orig_H = base.size
    if orig_W > max_width:
        scale_ratio = max_width / orig_W
        new_H = int(orig_H * scale_ratio)
        base = base.resize((max_width, new_H), Image.LANCZOS)
    else:
        scale_ratio = 1.0

    W, H = base.size
    sprite = extract_sprite(scene, actor_id)
    # Scale sprite: both for display resolution AND user-controlled size multiplier
    if sprite is not None:
        combined = scale_ratio * sprite_scale
        sw = max(4, int(sprite.width  * combined))
        sh = max(4, int(sprite.height * combined))
        sprite = sprite.resize((sw, sh), Image.LANCZOS)

    raw_polyline: List[List[float]] = path_hyp.get("polyline_2d", [])
    raw_trace: List[Any]            = path_hyp.get("depth_trace_m", [])
    sigs: List[Dict]                = path_hyp.get("kinematic_signatures", [])

    # Scale polyline to output resolution
    polyline = [[pt[0] * scale_ratio, pt[1] * scale_ratio] for pt in raw_polyline]

    # Filter valid depth-trace items
    depth_pts = [d for d in raw_trace if isinstance(d, dict) and "s" in d and "z_m" in d]
    ref_depth = float(np.median([d["z_m"] for d in depth_pts])) if depth_pts else 3.0

    n_pts    = max(len(polyline), 1)
    n_frames = int(fps * duration_s)
    schedule = _frame_schedule(n_pts, sigs, motion_override, n_frames)

    # ── start / end markers (scaled) ──
    _raw_start = (path_hyp.get("source_entity") or {}).get("start_uv")
    _raw_goal  = (path_hyp.get("target_entity") or {}).get("goal_uv")
    start_uv = [_raw_start[0] * scale_ratio, _raw_start[1] * scale_ratio] if _raw_start else None
    goal_uv  = [_raw_goal[0]  * scale_ratio, _raw_goal[1]  * scale_ratio] if _raw_goal  else None

    frames: List[Image.Image] = []
    font_sm  = _font(max(12, W // 65))
    trail_n  = 25

    for fi in range(n_frames):
        pt_idx = schedule[fi]
        if pt_idx >= n_pts:
            pt_idx = n_pts - 1
        px, py = float(polyline[pt_idx][0]), float(polyline[pt_idx][1])

        depth = _depth_at_pt(pt_idx, n_pts, depth_pts)
        scale = min(2.8, max(0.2, ref_depth / max(depth, 0.15)))

        motion_now = _motion_at(pt_idx, sigs) if motion_override in (None, "custom") else motion_override

        # ── build frame ──────────────────────────────────────────────────
        frame = base.convert("RGBA")
        drw   = ImageDraw.Draw(frame)

        # dimmed path (full route, low alpha)
        if len(polyline) >= 2:
            for i in range(0, n_pts - 1, max(1, n_pts // 200)):
                t  = i / max(n_pts - 1, 1)
                r, g = int(200*(1-t)), int(120*t)
                p1 = (int(polyline[i][0]),     int(polyline[i][1]))
                p2 = (int(polyline[i+1][0]),   int(polyline[i+1][1]))
                drw.line([p1, p2], fill=(r, g, 30, 55), width=1)

        # gradient trail (recent history)
        t_start = max(0, pt_idx - trail_n)
        for i in range(t_start, pt_idx):
            trail_t = (i - t_start) / trail_n
            alpha   = int(200 * trail_t)
            t_full  = i / max(n_pts - 1, 1)
            r2  = int(255 * (1 - t_full))
            g2  = int(215 * t_full)
            p1  = (int(polyline[i][0]),     int(polyline[i][1]))
            p2  = (int(polyline[i + 1][0]), int(polyline[i + 1][1])) if i + 1 < n_pts else p1
            drw.line([p1, p2], fill=(r2, g2, 30, alpha), width=max(2, int(3.5 * scale)))

        # start dot (green)
        if start_uv:
            sx, sy = int(start_uv[0]), int(start_uv[1])
            sr = max(6, int(9 * scale))
            drw.ellipse([sx-sr, sy-sr, sx+sr, sy+sr], fill=(40, 240, 80, 220), outline=(0, 160, 0, 255), width=2)
            drw.text((sx, sy), "S", fill=(0, 0, 0, 255), font=font_sm, anchor="mm")

        # end dot (red)
        if goal_uv:
            ex, ey = int(goal_uv[0]), int(goal_uv[1])
            er = max(6, int(9 * scale))
            drw.ellipse([ex-er, ey-er, ex+er, ey+er], fill=(240, 50, 50, 220), outline=(160, 0, 0, 255), width=2)
            drw.text((ex, ey), "E", fill=(255, 255, 255, 255), font=font_sm, anchor="mm")

        # ── actor sprite ─────────────────────────────────────────────────
        if sprite is not None:
            sw = max(5, int(sprite.width  * scale))
            sh = max(5, int(sprite.height * scale))
            spr = sprite.resize((sw, sh), Image.LANCZOS)
            ox  = int(px - sw / 2)
            oy  = int(py - sh / 2)
            # soft glow behind sprite
            glow = Image.new("RGBA", (sw + 8, sh + 8), (0, 0, 0, 0))
            gd   = ImageDraw.Draw(glow)
            gd.ellipse([2, 2, sw + 6, sh + 6], fill=(*_ACTOR_COL, 80))
            glow = glow.filter(ImageFilter.GaussianBlur(radius=4))
            frame.paste(glow, (ox - 4, oy - 4), glow)
            frame.paste(spr,  (ox,     oy),     spr)
        else:
            # fallback: coloured circle
            cr = max(7, int(13 * scale))
            drw.ellipse([px-cr, py-cr, px+cr, py+cr], fill=(*_ACTOR_COL, 230), outline=(0,0,0,200), width=2)

        # ── motion label (top-left) ───────────────────────────────────────
        motion_txt = _MOTION_ICON.get(motion_now, motion_now)
        drw.text((9, 9), motion_txt,
                 fill=(255, 255, 255, 220), font=font_sm,
                 stroke_width=1, stroke_fill=(0, 0, 0, 200))

        # ── progress bar (bottom) ─────────────────────────────────────────
        prog  = pt_idx / max(n_pts - 1, 1)
        bar_h = max(4, H // 100)
        drw.rectangle([0, H - bar_h, W, H], fill=(30, 30, 30, 160))
        drw.rectangle([0, H - bar_h, int(W * prog), H], fill=(*_ACTOR_COL, 210))

        frames.append(frame.convert("RGB"))

    if not frames:
        return None

    # Quantise to a shared 128-colour palette → much smaller GIF
    tf = tempfile.NamedTemporaryFile(suffix=".gif", delete=False, prefix="citv_anim_")
    try:
        # Build a global palette from the first frame
        pal_frame = frames[0].quantize(colors=64, dither=Image.Dither.FLOYDSTEINBERG)
        palette   = pal_frame.getpalette()
        q_frames  = [f.quantize(colors=64, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]

        q_frames[0].save(
            tf.name,
            save_all=True,
            append_images=q_frames[1:],
            loop=0,
            duration=1000 // fps,
            optimize=True,
            palette=palette,
        )
        return tf.name
    except Exception:
        # Plain fallback if quantise fails
        try:
            frames[0].save(tf.name, save_all=True, append_images=frames[1:],
                           loop=0, duration=1000 // fps, optimize=False)
            return tf.name
        except Exception:
            return None


# ────────────────────────────────────────────────────────────────────────────
# Cross-image animation
# ────────────────────────────────────────────────────────────────────────────

def generate_cross_animation(
    scene_a:  Dict,
    scene_b:  Dict,
    actor_id: str,   # entity in scene A
    dest_id:  str,   # entity in scene B (destination)
    fps:      int   = 12,
    hold_s:   float = 1.0,
    max_width: int  = 520,
    fade_s:   float = 0.8,
    arrive_s: float = 1.0,
) -> Optional[str]:
    """
    Two-image animation: show actor in scene A, cross-dissolve into scene B,
    actor appears at destination.
    """
    base_a = _base_image(scene_a)
    base_b = _base_image(scene_b)
    if base_a is None or base_b is None:
        return None

    # Scale both to max_width
    orig_W, orig_H = base_a.size
    if orig_W > max_width:
        r = max_width / orig_W
        base_a = base_a.resize((max_width, int(orig_H * r)), Image.LANCZOS)
    W, H = base_a.size
    base_b = base_b.resize((W, H), Image.LANCZOS)

    sprite_a = extract_sprite(scene_a, actor_id)
    sprite_b = extract_sprite(scene_b, dest_id) if dest_id else None

    ent_a   = scene_a["ent_by_id"].get(actor_id, {})
    bbox_a  = (ent_a.get("geometry") or {}).get("bbox_xyxy")
    cxy_a   = (ent_a.get("geometry") or {}).get("centroid_xy") or ([bbox_a[0] + (bbox_a[2]-bbox_a[0])//2, bbox_a[1] + (bbox_a[3]-bbox_a[1])//2] if bbox_a else [W//2, H//2])

    ent_b   = scene_b["ent_by_id"].get(dest_id, {}) if dest_id else {}
    bbox_b  = (ent_b.get("geometry") or {}).get("bbox_xyxy")
    cxy_b   = (ent_b.get("geometry") or {}).get("centroid_xy") or ([bbox_b[0] + (bbox_b[2]-bbox_b[0])//2, bbox_b[1] + (bbox_b[3]-bbox_b[1])//2] if bbox_b else [W//2, H//2])

    hold_n   = int(fps * hold_s)
    fade_n   = int(fps * fade_s)
    arrive_n = int(fps * arrive_s)

    font  = _font(max(13, W // 60))
    frames: List[Image.Image] = []

    # Phase 1: hold on scene A with actor highlighted
    for fi in range(hold_n):
        frame = base_a.convert("RGBA")
        drw   = ImageDraw.Draw(frame)
        if sprite_a:
            # Pulse effect
            pulse = 1.0 + 0.15 * math.sin(fi / hold_n * 4 * math.pi)
            sw = max(5, int(sprite_a.width  * pulse))
            sh = max(5, int(sprite_a.height * pulse))
            spr = sprite_a.resize((sw, sh), Image.LANCZOS)
            ox  = int(cxy_a[0] - sw / 2)
            oy  = int(cxy_a[1] - sh / 2)
            frame.paste(spr, (ox, oy), spr)
        if bbox_a:
            x1, y1, x2, y2 = (int(v) for v in bbox_a)
            drw.rectangle([x1, y1, x2, y2], outline=(*_ACTOR_COL, 230), width=3)
        drw.text((9, 9), f"Scene A  ·  {ent_a.get('label','actor')}",
                 fill=(255,255,255,220), font=font, stroke_width=1, stroke_fill=(0,0,0,200))
        frames.append(frame.convert("RGB"))

    # Phase 2: cross-dissolve A → B
    for fi in range(fade_n):
        t = fi / max(fade_n - 1, 1)
        blended = Image.blend(base_a, base_b, alpha=t)
        frame   = blended.convert("RGBA")
        drw     = ImageDraw.Draw(frame)
        # Fade sprite from A
        if sprite_a and t < 0.6:
            alpha_mult = max(0, 1 - t / 0.6)
            sw = max(5, sprite_a.width)
            sh = max(5, sprite_a.height)
            spr = sprite_a.copy()
            r, g, b, a = spr.split()
            a = a.point(lambda v: int(v * alpha_mult))
            spr = Image.merge("RGBA", (r, g, b, a))
            ox = int(cxy_a[0] - sw / 2)
            oy = int(cxy_a[1] - sh / 2)
            frame.paste(spr, (ox, oy), spr)
        drw.text((9, 9), "crossing →",
                 fill=(255,255,255,200), font=font, stroke_width=1, stroke_fill=(0,0,0,200))
        frames.append(frame.convert("RGB"))

    # Phase 3: arrive in scene B
    for fi in range(arrive_n):
        t = fi / max(arrive_n - 1, 1)
        frame = base_b.convert("RGBA")
        drw   = ImageDraw.Draw(frame)
        if sprite_b:
            # fade in
            sw = max(5, int(sprite_b.width  * (0.5 + 0.5 * t)))
            sh = max(5, int(sprite_b.height * (0.5 + 0.5 * t)))
            spr = sprite_b.resize((sw, sh), Image.LANCZOS)
            r2, g2, b2, a2 = spr.split()
            a2 = a2.point(lambda v: int(v * t))
            spr = Image.merge("RGBA", (r2, g2, b2, a2))
            ox = int(cxy_b[0] - sw / 2)
            oy = int(cxy_b[1] - sh / 2)
            frame.paste(spr, (ox, oy), spr)
        if bbox_b:
            x1, y1, x2, y2 = (int(v) for v in bbox_b)
            a_border = int(255 * t)
            drw.rectangle([x1, y1, x2, y2], outline=(*_TARGET_COL, a_border), width=3)
        drw.text((9, 9), f"Scene B  ·  arrived",
                 fill=(255,255,255,220), font=font, stroke_width=1, stroke_fill=(0,0,0,200))
        frames.append(frame.convert("RGB"))

    if not frames:
        return None

    tf = tempfile.NamedTemporaryFile(suffix=".gif", delete=False, prefix="citv_cross_")
    try:
        q_frames = [f.quantize(colors=64, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
        q_frames[0].save(tf.name, save_all=True, append_images=q_frames[1:],
                         loop=0, duration=1000 // fps, optimize=True)
    except Exception:
        frames[0].save(tf.name, save_all=True, append_images=frames[1:],
                       loop=0, duration=1000 // fps, optimize=False)
    return tf.name


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap: discover & pre-load default scene
# ────────────────────────────────────────────────────────────────────────────

_ROOT    = str(Path(__file__).parent)

# ── Fix 1: Pipeline singleton ─────────────────────────────────────────────
# Models are loaded once and kept alive for the entire app session.
# Re-instantiating on every button click was the root cause of the 30-45 s wait.
# Only reloads depth model when the user explicitly switches Indoor ↔ Outdoor.
_PIPE_CACHE: Dict[str, Any] = {
    "cfg":        None,
    "depth":      None,
    "pipe":       None,
    "scene_type": None,   # "auto" | "indoor" | "outdoor"
}

def _ensure_pipeline(img_path: str, scene_type: str = "auto"):
    """Return (cfg, depth_estimator, pipe), reusing cached instances when possible."""
    import sys
    sys.path.insert(0, _ROOT)
    from config import PreprocessConfig
    from depth import DepthEstimator
    from scene_understanding.pipeline import SceneUnderstandingPipeline

    cached_type = _PIPE_CACHE["scene_type"]
    pipe_alive  = _PIPE_CACHE["pipe"] is not None
    # Only reload when: never loaded, OR user changed to an explicit scene type
    need_reload = (not pipe_alive) or (
        scene_type != "auto" and scene_type != cached_type
    )

    if need_reload:
        cfg = PreprocessConfig()
        if scene_type != "auto":
            cfg.depth_model_variant = scene_type   # skip CLIP auto-classification
        depth = DepthEstimator(cfg, first_image=img_path)
        pipe  = SceneUnderstandingPipeline(depth, config=cfg)
        _PIPE_CACHE.update(cfg=cfg, depth=depth, pipe=pipe, scene_type=scene_type)

    return _PIPE_CACHE["cfg"], _PIPE_CACHE["depth"], _PIPE_CACHE["pipe"]


# ── Fix 2: Per-image isolated output directories ──────────────────────────
# Writing all images to the same flat output_studio/ caused cross-contamination
# and stale results. Each image now gets its own subdirectory.

def _out_dir_for(img_path: str) -> str:
    """Isolated output directory for one image: output_studio/{stem}/"""
    stem = Path(img_path).stem
    d = str(Path(_ROOT) / "output_studio" / stem)
    os.makedirs(d, exist_ok=True)
    return d


def _already_processed(img_path: str) -> Optional[str]:
    """Return paths_dir if this image was already analyzed, else None (instant cache)."""
    d = Path(_ROOT) / "output_studio" / Path(img_path).stem
    if not d.is_dir():
        return None
    scenes = discover_scenes(str(d))
    return scenes[0] if scenes else None


_SCENES  = discover_scenes(_ROOT)
_SC_LABS = [Path(s).stem.replace("_paths", "") for s in _SCENES]
_SC_OPTS = list(zip(_SC_LABS, _SCENES)) if _SCENES else []

_DEFAULT: Dict = load_scene(_SCENES[0]) if _SCENES else {"valid": False}

def _ent_choices(scene: Dict) -> List[Tuple[str, str]]:
    return [(f"{e.get('label','?')}  [{e['entity_id']}]", e["entity_id"])
            for e in scene.get("entities", [])]

_DEF_CHOICES = _ent_choices(_DEFAULT)
_DEF_IMG     = annotate_scene(_DEFAULT, None, None)


# ────────────────────────────────────────────────────────────────────────────
# Gradio application
# ────────────────────────────────────────────────────────────────────────────

_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1400px !important; margin: auto; }
#app-title { font-size: 1.7rem; font-weight: 800; margin-bottom: 2px; }
#app-sub   { color: #888; font-size: 0.95rem; margin-bottom: 16px; }
#hint-box  { background: #1e2130; border-radius: 8px; padding: 12px 14px;
             color: #a0c4ff; font-size: 0.88rem; line-height: 1.5; margin-top: 4px; }
.step-label { font-weight: 700; font-size: 1rem; color: #c0d0ff; }
"""

with gr.Blocks(title="CITV Studio") as demo:

    # ── app state ─────────────────────────────────────────────────────────
    _sc_a    = gr.State(_DEFAULT)
    _sc_b    = gr.State({"valid": False})
    _act_st  = gr.State(None)
    _tgt_st  = gr.State(None)

    # ── header ────────────────────────────────────────────────────────────
    gr.Markdown("# CITV Studio", elem_id="app-title")
    gr.Markdown("Turn any photo into an animation — tap two objects, watch them move.",
                elem_id="app-sub")

    # ═══════════════════════════════════════════════════════════════════════
    # Two-panel layout
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Row(equal_height=False):

        # ── LEFT panel ───────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.Markdown("### 1 · Load a scene", elem_classes=["step-label"])

            with gr.Tab("Upload photo"):
                upload_img    = gr.Image(type="filepath", label="Drop photo here", height=160)
                scene_type_rd = gr.Radio(
                    choices=[("✨ Auto-detect", "auto"),
                             ("🌆 Indoor / room", "indoor"),
                             ("🌳 Outdoor / street", "outdoor")],
                    value="auto",
                    label="Scene type",
                    info="Auto uses AI to pick the right depth model. "
                         "Set manually to skip that step and save ~4 s. "
                         "Switching Indoor ↔ Outdoor reloads the depth model (~8 s once).",
                )
                run_btn = gr.Button("Analyze →", variant="primary")
                run_log = gr.Markdown("")

            with gr.Tab("Demo / saved"):
                demo_dd = gr.Dropdown(
                    choices=_SC_OPTS or _SCENES,
                    value=_SCENES[0] if _SCENES else None,
                    label="Saved scene",
                    interactive=True,
                )
                load_btn = gr.Button("Load", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### 2 · Select actor & target", elem_classes=["step-label"])
            gr.Markdown(
                "**Tap two objects** on the scene image to the right → first tap = actor ★, second = target ▶  \n"
                "Or use the dropdowns below.",
                elem_id="hint-box",
            )

            actor_dd  = gr.Dropdown(choices=_DEF_CHOICES, label="★ Actor (moves)", value=None, interactive=True)
            actor_md  = gr.Markdown("*No actor selected.*")
            target_dd = gr.Dropdown(choices=_DEF_CHOICES, label="▶ Target (destination)", value=None, interactive=True)
            target_md = gr.Markdown("*No target selected.*")
            clear_sel = gr.Button("Clear selection", size="sm")

            gr.Markdown("---")
            gr.Markdown("### 3 · Define the action", elem_classes=["step-label"])

            action_txt = gr.Textbox(
                label="Describe anything you want (natural language)",
                placeholder='"the person walks to the chair"  ·  "fly over the fence"  ·  "anything imaginable"',
                lines=2,
            )
            auto_btn   = gr.Button("Auto-detect from text", size="sm", variant="secondary")

            motion_dd  = gr.Radio(
                choices=["walk", "run", "fly", "climb", "crawl", "interact", "hover", "jump", "custom"],
                label="Motion type",
                value="walk",
                info="Overrides the AI-detected kinematic signature. "
                     "'custom' lets the path's own signature control speed.",
            )

            with gr.Accordion("⚙  Animation options", open=False):
                gr.Markdown(
                    "All controls are optional — defaults produce a good result. "
                    "These are fully yours to adjust.",
                    elem_id="hint-box",
                )
                duration_sl = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.5, step=0.5,
                    label="Duration (seconds)",
                    info="How long the animation plays before looping.",
                )
                fps_sl = gr.Slider(
                    minimum=6, maximum=24, value=12, step=2,
                    label="Frames per second",
                    info="Higher = smoother but larger file. 12 is a good balance.",
                )
                path_n_sl = gr.Slider(
                    minimum=1, maximum=10, value=1, step=1,
                    label="Path variant (1 = AI best, higher = alternatives)",
                    info="The AI ranks paths by confidence. Variant 1 is the best match; "
                         "try 2–5 for more creative or scenic routes.",
                )
                sprite_sl = gr.Slider(
                    minimum=0.3, maximum=2.5, value=1.0, step=0.1,
                    label="Actor size multiplier",
                    info="Scale the moving actor sprite up or down.",
                )

            animate_btn  = gr.Button("Animate →", variant="primary", size="lg")
            dl_btn       = gr.Button("Download GIF", size="sm")
            dl_file      = gr.File(label="Download", visible=False)

            gr.Markdown("---")
            gr.Markdown("### Cross-image mode", elem_classes=["step-label"])
            gr.Markdown("Upload a second photo — actor jumps from Scene A into Scene B.")
            upload_b    = gr.Image(type="filepath", label="Scene B photo", height=130)
            run_b_btn   = gr.Button("Analyze Scene B →", variant="secondary", size="sm")
            run_b_log   = gr.Markdown("")
            target_b_dd = gr.Dropdown(choices=[], label="▶ Destination in Scene B", value=None, interactive=True)
            cross_btn   = gr.Button("Cross-image Animate →", variant="secondary", size="sm")

        # ── RIGHT panel ──────────────────────────────────────────────────
        with gr.Column(scale=2):

            scene_img = gr.Image(
                label="Scene (click to select actor / target)",
                interactive=True,
                value=_DEF_IMG,
                height=520,
            )
            status_md = gr.Markdown(
                "Ready. Tap an object to select actor ★, tap another for target ▶, then click **Animate →**"
            )

            result_img = gr.Image(
                label="Animation result",
                height=520,
            )

            with gr.Accordion("Scene outputs & details", open=False):
                with gr.Row():
                    out_seg     = gr.Image(label="Segmentation",       value=_DEFAULT.get("img_seg"))
                    out_regions = gr.Image(label="Regions",            value=_DEFAULT.get("img_regions"))
                    out_rel     = gr.Image(label="Relations",          value=_DEFAULT.get("img_relations"))
                with gr.Row():
                    out_path    = gr.Image(label="Path overlay",       value=_DEFAULT.get("img_path_overlay"))
                    out_trav    = gr.Image(label="Traversability",     value=_DEFAULT.get("img_trav"))
                    out_mfold   = gr.Image(label="Action manifold",    value=_DEFAULT.get("img_manifold"))
                with gr.Row():
                    out_motion  = gr.Image(label="Motion contracts",   value=_DEFAULT.get("img_motion"))
                    out_supp    = gr.Image(label="Support trace",      value=_DEFAULT.get("img_support"))
                    out_vis     = gr.Image(label="Visibility profile", value=_DEFAULT.get("img_visibility"))
                with gr.Row():
                    out_atlas   = gr.Image(label="Path atlas",         value=_DEFAULT.get("img_atlas"))
                    out_depth   = gr.Image(label="Depth map",          value=_DEFAULT.get("img_depth"))
                    out_occ     = gr.Image(label="Occlusion state",    value=_DEFAULT.get("img_occlusion"))

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers that update all scene-dependent outputs at once
    # ═══════════════════════════════════════════════════════════════════════

    _SCENE_IMG_OUTS = [
        scene_img,
        actor_dd, target_dd,
        out_seg, out_regions, out_rel,
        out_path, out_trav, out_mfold,
        out_motion, out_supp, out_vis,
        out_atlas, out_depth, out_occ,
        status_md,
    ]

    def _populate_scene(scene: Dict):
        img     = annotate_scene(scene, None, None)
        choices = _ent_choices(scene)
        n_e     = len(scene.get("entities", []))
        n_h     = len(scene.get("hypotheses", []))
        st      = f"✅ **{scene.get('stem','')}** — {n_e} objects · {n_h} paths · Tap actor then target"
        return [
            img,
            gr.update(choices=choices, value=None),
            gr.update(choices=choices, value=None),
            scene.get("img_seg"),
            scene.get("img_regions"),
            scene.get("img_relations"),
            scene.get("img_path_overlay"),
            scene.get("img_trav"),
            scene.get("img_manifold"),
            scene.get("img_motion"),
            scene.get("img_support"),
            scene.get("img_visibility"),
            scene.get("img_atlas"),
            scene.get("img_depth"),
            scene.get("img_occlusion"),
            st,
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # Event: load demo / saved scene
    # ═══════════════════════════════════════════════════════════════════════

    def _on_load_demo(path: str):
        scene = load_scene(path) if path else {"valid": False}
        if not scene.get("valid"):
            return [{"valid": False}] + [gr.update()] * len(_SCENE_IMG_OUTS)
        return [scene] + _populate_scene(scene)

    load_btn.click(_on_load_demo, [demo_dd], [_sc_a] + _SCENE_IMG_OUTS)
    demo_dd.change( _on_load_demo, [demo_dd], [_sc_a] + _SCENE_IMG_OUTS)

    # ═══════════════════════════════════════════════════════════════════════
    # Event: analyze new image (Scene A)
    # ═══════════════════════════════════════════════════════════════════════

    # Fix 2+3: generator so every yield immediately clears stale UI.
    # Fix 1: uses _ensure_pipeline (singleton) — no reload after first run.
    # Fix 2: uses _out_dir_for (per-image dirs) + _already_processed (cache).
    def _on_analyze(img_path: str, scene_type: str, progress=gr.Progress(track_tqdm=True)):
        _blank = [{"valid": False}] + [gr.update()] * len(_SCENE_IMG_OUTS)

        if not img_path:
            yield _blank + ["⚠ No image selected — drop a photo above."]
            return

        # ── Step 0: immediately wipe old scene so dropdowns are never stale ──
        yield _blank + ["⏳ Checking cache…"]

        # ── Step 1: instant cache hit ─────────────────────────────────────────
        cached = _already_processed(img_path)
        if cached:
            scene = load_scene(cached)
            if scene.get("valid"):
                n_e = len(scene.get("entities", []))
                n_h = len(scene.get("hypotheses", []))
                yield [scene] + _populate_scene(scene) + [
                    f"⚡ **Loaded from cache** (instant) — "
                    f"**{n_e} objects** · **{n_h} paths**  \n"
                    "Tap actor ★ on the scene image, then tap the target ▶, then **Animate →**"
                ]
                return

        # ── Step 2: first-time or scene-type-change load ─────────────────────
        first_run = _PIPE_CACHE["pipe"] is None
        type_changed = (
            scene_type != "auto"
            and _PIPE_CACHE["scene_type"] not in (None, scene_type)
        )
        if first_run:
            yield _blank + [
                "⏳ **Step 1 / 5** — Loading AI models for the first time…  \n"
                "This takes ~30 s once; every image after this is much faster."
            ]
        elif type_changed:
            yield _blank + [
                f"⏳ Reloading depth model for **{scene_type}** scenes (~8 s)…"
            ]
        else:
            yield _blank + [
                "⏳ **Step 1 / 5** — Models already loaded — starting analysis…"
            ]

        try:
            progress(0.05, desc="Loading AI models…")
            cfg, depth_est, pipe = _ensure_pipeline(img_path, scene_type)

            # ── Step 3: depth ─────────────────────────────────────────────────
            yield _blank + ["⏳ **Step 2 / 5** — Estimating scene depth…"]
            progress(0.20, desc="Depth estimation…")

            # ── Step 4: segment ───────────────────────────────────────────────
            yield _blank + [
                "⏳ **Step 3 / 5** — Detecting & segmenting every object…  \n"
                "*(Grounding DINO + SAM2 — this is the longest step, ~10–20 s)*"
            ]
            progress(0.35, desc="Detecting objects…")

            # ── Step 5: full pipeline ─────────────────────────────────────────
            yield _blank + ["⏳ **Step 4 / 5** — Labelling objects & planning paths…"]
            progress(0.50, desc="Running pipeline…")

            out_dir = _out_dir_for(img_path)
            pipe.process_image(img_path, out_dir)

            # ── Step 6: load results ──────────────────────────────────────────
            yield _blank + ["⏳ **Step 5 / 5** — Loading results…"]
            progress(0.90, desc="Loading results…")

            new_scenes = discover_scenes(out_dir)
            if new_scenes:
                scene = load_scene(new_scenes[0])
                n_e   = len(scene.get("entities",   []))
                n_h   = len(scene.get("hypotheses", []))
                progress(1.0)
                yield [scene] + _populate_scene(scene) + [
                    f"✅ Done! Found **{n_e} objects** · **{n_h} paths**  \n"
                    "**How to animate:** tap actor ★ on the scene image → tap target ▶ → "
                    "choose motion → click **Animate →**  \n"
                    "Or describe the action in the text box and click **Auto-detect**."
                ]
            else:
                yield _blank + [
                    "⚠ Pipeline finished but no scene data was written.  \n"
                    "Check the terminal for errors."
                ]

        except Exception as ex:
            import traceback; traceback.print_exc()
            yield _blank + [f"❌ **Error:** {ex}"]

    run_btn.click(
        _on_analyze,
        [upload_img, scene_type_rd],
        [_sc_a] + _SCENE_IMG_OUTS + [run_log],
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Event: analyze Scene B
    # ═══════════════════════════════════════════════════════════════════════

    def _on_analyze_b(img_path: str, progress=gr.Progress(track_tqdm=True)):
        _blank3 = [{"valid": False}, gr.update()]

        if not img_path:
            yield _blank3 + ["⚠ No Scene B image selected."]
            return

        yield _blank3 + ["⏳ Checking cache for Scene B…"]

        cached = _already_processed(img_path)
        if cached:
            scene = load_scene(cached)
            if scene.get("valid"):
                choices = _ent_choices(scene)
                yield [scene, gr.update(choices=choices, value=None),
                       "⚡ **Scene B loaded from cache** — pick a destination object below."]
                return

        yield _blank3 + [
            "⏳ Analyzing Scene B — reusing loaded models (no reload needed)…"
        ]

        try:
            progress(0.1, desc="Preparing Scene B…")
            cfg, depth_est, pipe = _ensure_pipeline(img_path, "auto")
            progress(0.3, desc="Running pipeline on Scene B…")

            out_dir = _out_dir_for(img_path)
            pipe.process_image(img_path, out_dir)

            progress(0.9)
            new_scenes = discover_scenes(out_dir)
            if new_scenes:
                scene   = load_scene(new_scenes[0])
                choices = _ent_choices(scene)
                n_e     = len(scene.get("entities", []))
                yield [scene, gr.update(choices=choices, value=None),
                       f"✅ Scene B ready — **{n_e} objects** found. "
                       "Pick a destination object in the dropdown below."]
            else:
                yield _blank3 + ["⚠ Scene B processed but no data found."]

        except Exception as ex:
            import traceback; traceback.print_exc()
            yield _blank3 + [f"❌ Scene B error: {ex}"]

    run_b_btn.click(_on_analyze_b, [upload_b], [_sc_b, target_b_dd, run_b_log])

    # ═══════════════════════════════════════════════════════════════════════
    # Event: click on scene image → select actor / target
    # ═══════════════════════════════════════════════════════════════════════

    _CLICK_OUTS = [_act_st, _tgt_st, scene_img, actor_dd, target_dd, actor_md, target_md, status_md]

    def _on_click(scene, actor, target, evt: gr.SelectData):
        if not scene or not scene.get("valid"):
            return [actor, target] + [gr.update()] * (len(_CLICK_OUTS) - 2)

        x, y = (evt.index[0], evt.index[1]) if evt.index else (0, 0)
        # Ignore clicks in the legend strip (below pipe_h)
        if y > scene.get("pipe_h", 1024):
            return [actor, target] + [gr.update()] * (len(_CLICK_OUTS) - 2)

        clicked = entity_at(x, y, scene)
        if clicked is None:
            return [actor, target] + [gr.update()] * (len(_CLICK_OUTS) - 2)

        lbl = scene["ent_by_id"].get(clicked, {}).get("label", clicked)

        if actor is None:
            actor = clicked
            hint  = f"★ **Actor:** {lbl} · now tap the target ▶"
        elif target is None:
            if clicked == actor:
                hint = f"★ Actor is already **{lbl}** — tap a *different* object for target"
            else:
                target = clicked
                hint   = f"▶ **Target:** {lbl} · ready — click **Animate →**"
        else:
            actor  = clicked
            target = None
            hint   = f"★ **New actor:** {lbl} · tap the target ▶"

        img     = annotate_scene(scene, actor, target)
        choices = _ent_choices(scene)

        a_info = _ent_card(scene, actor)  if actor  else "*No actor selected.*"
        t_info = _ent_card(scene, target) if target else "*No target selected.*"

        return [
            actor, target,
            img,
            gr.update(choices=choices, value=actor),
            gr.update(choices=choices, value=target),
            a_info, t_info,
            hint,
        ]

    scene_img.select(_on_click, [_sc_a, _act_st, _tgt_st], _CLICK_OUTS)

    # ─── dropdowns change ────────────────────────────────────────────────

    def _on_dd_change(scene, a_val, t_val):
        img    = annotate_scene(scene, a_val or None, t_val or None) if scene.get("valid") else None
        a_info = _ent_card(scene, a_val) if a_val and scene.get("valid") else "*No actor selected.*"
        t_info = _ent_card(scene, t_val) if t_val and scene.get("valid") else "*No target selected.*"
        return img, a_info, t_info

    actor_dd.change( _on_dd_change, [_sc_a, actor_dd,  target_dd], [scene_img, actor_md,  target_md])
    target_dd.change(_on_dd_change, [_sc_a, actor_dd,  target_dd], [scene_img, actor_md,  target_md])

    # ─── clear selection ─────────────────────────────────────────────────

    def _on_clear(scene):
        img     = annotate_scene(scene, None, None) if scene.get("valid") else None
        choices = _ent_choices(scene)
        return (None, None, img,
                gr.update(choices=choices, value=None),
                gr.update(choices=choices, value=None),
                "*No actor selected.*", "*No target selected.*",
                "Selection cleared. Tap actor ★ then target ▶")

    clear_sel.click(_on_clear, [_sc_a],
                    [_act_st, _tgt_st, scene_img, actor_dd, target_dd, actor_md, target_md, status_md])

    # ─── auto-ground text ────────────────────────────────────────────────

    def _on_auto(text, scene):
        if not text or not scene.get("valid"):
            return gr.update(), gr.update()
        a, b, _ = ground_text(text, scene)
        choices  = _ent_choices(scene)
        return gr.update(choices=choices, value=a), gr.update(choices=choices, value=b)

    auto_btn.click(_on_auto, [action_txt, _sc_a], [actor_dd, target_dd])

    # ═══════════════════════════════════════════════════════════════════════
    # Event: Animate →
    # ═══════════════════════════════════════════════════════════════════════

    def _on_animate(scene, actor, target, text, motion,
                    duration, fps, path_variant, sprite_mult,
                    progress=gr.Progress()):
        if not scene or not scene.get("valid"):
            return None, "❌ No scene loaded. Upload and analyze a photo first."

        # ── Text grounding fallback ───────────────────────────────────────────
        if text and not (actor and target):
            a2, b2, m2 = ground_text(text, scene)
            actor  = actor  or a2
            target = target or b2
            if m2 and motion == "walk":
                motion = m2

        if not actor:
            return None, (
                "⚠ No actor selected.  \n"
                "**Options:**  \n"
                "• Tap an object on the scene image (first tap = actor ★)  \n"
                "• Choose from the **★ Actor** dropdown  \n"
                "• Type a description and click **Auto-detect from text**"
            )

        # ── Auto-pick closest target if not specified ─────────────────────────
        if not target:
            for (src, tgt) in scene.get("hyp_index", {}):
                if src == actor:
                    target = tgt
                    break
        if not target:
            return None, (
                "⚠ No target found.  \n"
                "**Options:**  \n"
                "• Tap a second object on the scene image (second tap = target ▶)  \n"
                "• Choose from the **▶ Target** dropdown  \n"
                "• The AI will auto-pick the most plausible destination if you leave it empty "
                "and there is at least one path from the selected actor."
            )

        progress(0.1, desc="Finding paths…")
        paths = find_paths(scene, actor, target)

        # ── Motion-type filter (soft — keeps all if none match) ───────────────
        if motion != "custom":
            flt = [p for p in paths if any(
                s.get("motion") == motion for s in p.get("kinematic_signatures", [])
            )]
            if flt:
                paths = flt

        if not paths:
            return None, (
                f"⚠ No path found between **{actor}** and **{target}**.  \n"
                "Try a different actor / target pair, or a different motion type."
            )

        # ── Pick the n-th path variant (user-controlled) ──────────────────────
        variant_idx = max(0, min(int(path_variant) - 1, len(paths) - 1))
        chosen_path = paths[variant_idx]
        n_paths     = len(paths)

        progress(0.3, desc="Extracting actor sprite…")
        progress(0.5, desc="Rendering frames…")

        gif_path = generate_animation(
            scene, chosen_path, actor, target,
            motion_override = (motion if motion != "custom" else None),
            fps             = int(fps),
            duration_s      = float(duration),
            max_width       = 520,
            sprite_scale    = float(sprite_mult),
        )

        if gif_path is None:
            return None, "❌ Animation render failed — check terminal for details."

        a_lbl   = scene["ent_by_id"].get(actor,  {}).get("label", actor)
        t_lbl   = scene["ent_by_id"].get(target, {}).get("label", target)
        scores  = chosen_path.get("scores") or {}
        conf    = round(scores.get("overall_confidence", scores.get("hybrid_overall", 0)), 2)
        mfold   = chosen_path.get("manifold_type", chosen_path.get("trajectory_type", "—"))
        sigs    = chosen_path.get("kinematic_signatures", [])
        seq     = " → ".join(dict.fromkeys(s.get("motion", "?") for s in sigs[:6])) or motion

        progress(1.0)
        st = (
            f"🎬 **{a_lbl}** → **{t_lbl}**  ·  `{motion}` motion  \n"
            f"Path variant **{variant_idx + 1} / {n_paths}**  ·  "
            f"confidence `{conf}`  ·  type `{mfold}`  \n"
            f"Kinematic sequence: `{seq}`  ·  "
            f"{int(fps)} fps · {float(duration):.1f} s  \n"
            "*Try other variants, motion types or the sliders in ⚙ Animation options.*"
        )
        return gif_path, st

    animate_btn.click(
        _on_animate,
        [_sc_a, actor_dd, target_dd, action_txt, motion_dd,
         duration_sl, fps_sl, path_n_sl, sprite_sl],
        [result_img, status_md],
    )

    # ─── download GIF ────────────────────────────────────────────────────

    def _on_download(scene, actor, target, motion):
        paths = find_paths(scene, actor or "", target or "") if scene.get("valid") else []
        if not paths:
            return gr.update(visible=False)
        gif = generate_animation(scene, paths[0], actor or "", target or "",
                                 motion_override=(motion if motion != "custom" else None))
        if gif:
            return gr.update(value=gif, visible=True)
        return gr.update(visible=False)

    dl_btn.click(_on_download, [_sc_a, actor_dd, target_dd, motion_dd], [dl_file])

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-image animation
    # ═══════════════════════════════════════════════════════════════════════

    def _on_cross(scene_a, scene_b, actor, dest):
        if not scene_a.get("valid"):
            return None, "⚠ Load Scene A first."
        if not scene_b.get("valid"):
            return None, "⚠ Analyze Scene B first (use the panel on the left)."
        if not actor:
            return None, "⚠ Select actor in Scene A."

        gif = generate_cross_animation(scene_a, scene_b, actor, dest or "")
        if gif:
            return gif, f"🎬 Cross-image animation: **{scene_a.get('stem','')}** → **{scene_b.get('stem','')}**"
        return None, "❌ Cross-image animation failed."

    cross_btn.click(_on_cross, [_sc_a, _sc_b, actor_dd, target_b_dd], [result_img, status_md])


# ────────────────────────────────────────────────────────────────────────────
# Entity info card helper (defined after Blocks to avoid closure issues)
# ────────────────────────────────────────────────────────────────────────────

def _ent_card(scene: Dict, eid: str) -> str:
    ent = scene.get("ent_by_id", {}).get(eid)
    if not ent:
        return f"*`{eid}` not found*"
    geo    = ent.get("geometry", {})
    bbox   = geo.get("bbox_xyxy", [])
    cxy    = geo.get("centroid_xy", [])
    roles  = ent.get("roles",   {}) or {}
    acts   = ent.get("actions", {}) or {}
    lines  = [f"**{ent.get('label', eid)}** `{eid}`"]
    if bbox:
        lines.append(f"bbox `[{', '.join(str(int(v)) for v in bbox)}]`")
    if cxy:
        lines.append(f"centroid `({cxy[0]:.0f}, {cxy[1]:.0f})`")
    if roles:
        top = sorted(roles.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append("roles: " + "  ·  ".join(f"`{r}`" for r, _ in top))
    if acts:
        top = sorted(acts.items(), key=lambda x: x[1], reverse=True)[:4]
        lines.append("actions: " + "  ·  ".join(f"`{a}`" for a, _ in top))
    return "  \n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=_CSS,
    )
