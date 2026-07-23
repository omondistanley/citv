"""CITV Studio -- turn any photo into a photorealistic animation.

Golden path: upload a photo -> Analyze (runs the staged scene-understanding
pipeline: depth, segmentation, labelling, relations, hierarchy, path
hypotheses) -> either just pick an actor/target object, OR give any
combination of custom input (drawn path, free-text actor/action, an
uploaded actor image, avoid/required/behind-object constraints) -> Animate.

Phase 3's ``MotionContract`` (``scene_understanding/animation/motion_contract.py``)
is the open-ended input layer: every custom field below is independently
optional. Whatever you set is preserved exactly; whatever you leave unset is
grounded from the pipeline's own evidence (see ``ground_motion_contract``).

Rebuilt fresh against the staged package pipeline (``SceneUnderstandingPipeline
._run_stage_chain``) rather than the legacy monolith entrypoint, so it
exercises the same stage chain this session's work wired up end-to-end.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# The upstream sam2 repo is cloned into <repo_root>/sam2/ with its real
# installed package one level deeper, at <repo_root>/sam2/sam2/. Because
# this file lives at <repo_root>, Python puts <repo_root> on sys.path[0]
# automatically -- which makes the default import resolution treat the
# outer <repo_root>/sam2 (no __init__.py of its own) as an implicit
# namespace package matching the name "sam2", shadowing the real
# editable-installed package. sam2's own build_sam.py detects exactly this
# shadow and aborts. Load the real package by its exact file path and
# cache it in sys.modules before anything else can shadow it, so every
# later `import sam2` anywhere in the pipeline reuses this correct module.
_sam2_real_init = Path(__file__).resolve().parent / "sam2" / "sam2" / "__init__.py"
if "sam2" not in sys.modules and _sam2_real_init.exists():
    _spec = importlib.util.spec_from_file_location(
        "sam2", _sam2_real_init, submodule_search_locations=[str(_sam2_real_init.parent)]
    )
    _sam2_module = importlib.util.module_from_spec(_spec)
    sys.modules["sam2"] = _sam2_module
    _spec.loader.exec_module(_sam2_module)

import json
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from skimage.morphology import skeletonize

from config import PreprocessConfig
from depth import DepthEstimator
from scene_understanding.animation import render_animation
from scene_understanding.animation.motion_contract import MotionContract, ground_motion_contract, reroute_around_obstacles
from scene_understanding.animation.sketch_scene import SketchRegion, SketchScene, run_sketch_scene_pipeline
from scene_understanding.animation.tier1_blender import find_blender_executable, render_tier1_animation
from scene_understanding.pipeline import SceneUnderstandingPipeline
from scene_understanding.pipeline_context import PipelineContext

_OUTPUT_ROOT = Path("output_scene_understanding")
TIER_3D, TIER_2D = "3D (Blender, default)", "2D (fast, always available)"
_SKETCH_CANVAS_SIZE = (640, 480)  # (width, height)
_SKETCH_REGION_COLORS = {"floor": (235, 206, 135), "wall": (200, 200, 200), "object": (140, 230, 140)}  # BGR

# Heavy models are loaded once per process and kept resident across requests
# (mirrors the lazy-singleton pattern used elsewhere in this codebase for the
# same reason -- reloading SAM2/GDINO/Florence2/RAM++ per request is far too
# slow, especially on the CPU-only deployment target from Phase 4).
_pipeline_cache: Dict[str, SceneUnderstandingPipeline] = {}


def _get_pipeline(first_image_path: str) -> SceneUnderstandingPipeline:
    if "pipeline" not in _pipeline_cache:
        cfg = PreprocessConfig()
        depth_estimator = DepthEstimator(cfg, first_image=first_image_path)
        _pipeline_cache["pipeline"] = SceneUnderstandingPipeline(depth_estimator, config=cfg)
    return _pipeline_cache["pipeline"]


def _release_heavy_models(pipeline: SceneUnderstandingPipeline) -> None:
    """Frees GDINO/SAM2/depth-estimator weights after a full Analyze run
    completes, and drops the cached pipeline so the next Analyze click
    reconstructs everything from scratch. These were previously kept
    resident for the entire process lifetime (the whole point of
    _pipeline_cache) to avoid slow reloads -- but on this 8GB, CPU-only
    machine that sustained footprint, on top of Florence-2/RAM++ being
    reloaded every run, has already caused a real kernel panic under memory
    pressure. Florence-2/RAM++ are already freed mid-chain by
    _stage_unload_labellers; this releases the rest. Trades a slower next
    cold-start (GDINO+SAM2+DepthAnythingV2 reload from disk) for a much
    lower sustained memory floor between runs."""
    try:
        pipeline.depth_estimator.unload_backend()
    except Exception as exc:
        print(f"  [Studio] depth_estimator unload failed (non-fatal): {exc}")
    try:
        pipeline.sam2_wrapper.unload()
    except Exception as exc:
        print(f"  [Studio] sam2_wrapper unload failed (non-fatal): {exc}")
    _pipeline_cache.clear()


def _object_labels(objects: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    labels: List[str] = []
    id_by_label: Dict[str, Any] = {}
    for obj in objects:
        label = f"{obj.get('canonical_name') or obj.get('label', 'object')} ({obj.get('id')})"
        labels.append(label)
        id_by_label[label] = obj.get("id")
    return labels, id_by_label


def _to_bgra(img_rgb_or_rgba: np.ndarray) -> np.ndarray:
    """Gradio's numpy image is RGB(A); every render function in this project
    works in BGR (OpenCV convention, matching ``ctx.img_bgr``)."""
    if img_rgb_or_rgba.shape[2] == 4:
        rgb, alpha = img_rgb_or_rgba[..., :3], img_rgb_or_rgba[..., 3]
    else:
        rgb, alpha = img_rgb_or_rgba, np.full(img_rgb_or_rgba.shape[:2], 255, np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.dstack([bgr, alpha])


def analyze_scene(image_path: Optional[str]):
    """Run the staged pipeline chain and populate every picker."""
    if not image_path:
        empty = gr.update(choices=[], value=None)
        return empty, empty, empty, empty, empty, None, None, "Upload a photo first."

    pipeline = _get_pipeline(image_path)
    stem = Path(image_path).stem
    output_dir = _OUTPUT_ROOT / f"{stem}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx: PipelineContext = pipeline._run_stage_chain(image_path, str(output_dir))
    pipeline._stage_export_package(ctx)  # writes scene.json for inspection; not needed by this UI

    objects = ctx.extra.get("objects", [])
    labels, id_by_label = _object_labels(objects)

    # path_hypotheses_count is already computed by paths_export.py -- reading
    # and fully parsing the actual JSON here just to report a count was
    # loading the entire (real-world: hundreds of MB, hundreds of
    # hypotheses each with several full per-point profiles) file into
    # memory and then keeping it alive for the whole session via gr.State,
    # even though nothing downstream ever reads it back (the actual render
    # path, ground_motion_contract, independently re-reads the one
    # hypothesis it needs straight from disk). A real, measured multi-
    # hundred-MB memory spike on every Analyze click, for a number.
    n_hypotheses = int(ctx.path_exports.get("path_hypotheses_count", 0))

    state = {"ctx": ctx, "id_by_label": id_by_label}
    status = (
        f"Found **{len(objects)}** objects and **{n_hypotheses}** path hypotheses "
        f"(output: `{output_dir}`). Pick an actor/target below, or use the custom inputs, then Animate."
    )
    actor_default = labels[0] if labels else None
    target_default = labels[-1] if len(labels) > 1 else None
    img_rgb = cv2.cvtColor(ctx.img_bgr, cv2.COLOR_BGR2RGB)
    _release_heavy_models(pipeline)
    return (
        gr.update(choices=labels, value=actor_default),
        gr.update(choices=labels, value=target_default),
        gr.update(choices=labels, value=[]),  # avoid
        gr.update(choices=labels, value=[]),  # required
        gr.update(choices=labels, value=[]),  # behind
        state,
        img_rgb,
        status,
    )


def inspect_object_at_point(state: Optional[Dict[str, Any]], evt: gr.SelectData) -> str:
    """Click-to-inspect: hit-tests the clicked pixel against every real
    object mask and reports its name, so a user can build a clearer mental
    model of the scene before picking an actor/target or writing custom
    text -- surfaces the rich per-object naming fields stages/labelling.py
    already captures (canonical_name/category/aliases) but that had no UI
    consumer until now."""
    if not state:
        return "Run Analyze first."
    ctx: PipelineContext = state["ctx"]
    x, y = evt.index
    objects = ctx.extra.get("objects", [])
    xi, yi = int(round(x)), int(round(y))
    xi = max(0, min(ctx.width - 1, xi))
    yi = max(0, min(ctx.height - 1, yi))
    hits = [o for o in objects if o.get("_sam2_mask_array") is not None and o["_sam2_mask_array"][yi, xi]]
    if not hits:
        return "No detected object at that point."
    # Multiple masks can overlap at a point (e.g. a part nested inside a
    # bigger object) -- the smallest-area hit is the most specific one.
    obj = min(hits, key=lambda o: int(np.asarray(o["_sam2_mask_array"]).sum()))
    name = obj.get("canonical_name") or obj.get("label", "object")
    bits = [f"**{name}**"]
    category = obj.get("category")
    if category and category != name:
        bits.append(f"({category})")
    aliases = [a for a in obj.get("aliases", []) if a != name]
    if aliases:
        bits.append(f"— also seen as: {', '.join(aliases[:4])}")
    return " ".join(bits)


_PATH_COLOR = (235, 64, 52)  # normal drawn segment (unchanged from before)
_OBSTACLE_WARNING_COLOR = (255, 140, 0)  # orange -- this segment crosses a real detected object


def _segment_crosses_obstacle(p0: Tuple[float, float], p1: Tuple[float, float], obstacle_mask: Optional[np.ndarray]) -> bool:
    """Samples along a drawn line segment against the real obstacle mask
    (same ``_paths_obstacle_mask`` the render/grounding path already uses,
    see motion_contract.py's ``_flag_obstacle_crossings``) so the canvas can
    show a crossing before the user ever clicks Animate."""
    if obstacle_mask is None:
        return False
    h, w = obstacle_mask.shape[:2]
    n_samples = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    for t in np.linspace(0.0, 1.0, n_samples):
        x = int(round(p0[0] + t * (p1[0] - p0[0])))
        y = int(round(p0[1] + t * (p1[1] - p0[1])))
        if 0 <= y < h and 0 <= x < w and obstacle_mask[y, x]:
            return True
    return False


def _redraw_canvas(
    img_rgb: np.ndarray, points: List[Tuple[float, float]], obstacle_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    canvas = img_rgb.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(canvas, (int(x), int(y)), 5, _PATH_COLOR, -1)
        if i > 0:
            crosses = _segment_crosses_obstacle(points[i - 1], (x, y), obstacle_mask)
            color = _OBSTACLE_WARNING_COLOR if crosses else _PATH_COLOR
            cv2.line(canvas, (int(points[i - 1][0]), int(points[i - 1][1])), (int(x), int(y)), color, 2)
    return canvas


def _walk_prev(prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [end]
    cur = end
    while prev.get(cur) is not None:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def _bfs(coord_set: set, start: Tuple[int, int]):
    dist = {start: 0}
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        r, c = cur
        head += 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in coord_set and nb not in dist:
                    dist[nb] = dist[cur] + 1
                    prev[nb] = cur
                    queue.append(nb)
    return dist, prev


def _rescale_points(
    points: List[Tuple[float, float]], actual_shape: Tuple[int, int], expected_shape: Tuple[int, int],
) -> List[Tuple[float, float]]:
    ah, aw = actual_shape[:2]
    eh, ew = expected_shape[:2]
    if ah == 0 or aw == 0 or (ah == eh and aw == ew):
        return points
    sx, sy = ew / aw, eh / ah
    return [(x * sx, y * sy) for x, y in points]


def _vectorize_editor_stroke(
    editor_value: Optional[Dict[str, Any]],
    expected_shape: Tuple[int, int],
    max_points: int = 60,
    min_ink_px: int = 8,
    alpha_threshold: int = 32,
) -> List[Tuple[float, float]]:
    """Converts a freehand brush stroke on the ImageEditor's single drawable
    layer into a plain (x, y) pixel polyline -- the same shape
    ``MotionContract.drawn_polyline_2d`` already accepted from the old
    click-based mechanism, so nothing downstream (``ground_motion_contract``,
    ``build_hypothesis``, ``paths_export.py``) needs to change; only how the
    polyline is produced changes.

    Deliberately conservative: always returns *something* usable or an empty
    list, never raises, even on a messy/self-intersecting/looped stroke.
      1. Read layers[0]'s alpha channel as the "ink" mask -- robust against
         brush-color/photo-color collisions, unlike diffing the composite
         against the background.
      2. Keep only the largest connected ink component (multiple disjoint
         strokes are not concatenated -- there's no principled way to infer
         intended order between two separate blobs).
      3. Skeletonize to a 1px-wide curve, walk it into an ordered point list
         via BFS over its 8-connected pixel graph. With >=2 skeleton
         endpoints, picks the pair with the greatest BFS shortest-path
         length (the "BFS diameter") -- this silently excludes short stray
         branches/brush wobble at either end. With 0-1 endpoints (a closed
         loop or blob), falls back to the two most pairwise-distant ink
         pixels so a doodle never crashes the vectorizer.
      4. cv2.approxPolyDP decimates while preserving curvature (unlike
         uniform-stride sampling) -- this is what keeps the result a real
         curve rather than a re-derived straight polyline.
    """
    if not editor_value:
        return []
    layers = editor_value.get("layers") or []
    if not layers or layers[0] is None:
        return []
    layer = np.asarray(layers[0])
    if layer.ndim != 3 or layer.shape[2] < 4:
        return []
    ink = layer[..., 3] >= alpha_threshold
    if int(ink.sum()) < min_ink_px:
        return []

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return []
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component_mask = labels_im == largest_label

    skeleton = skeletonize(component_mask)
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return []
    if len(ys) == 1:
        x, y = float(xs[0]), float(ys[0])
        return _rescale_points([(x, y), (x, y)], skeleton.shape, expected_shape)

    coords = list(zip(ys.tolist(), xs.tolist()))
    coord_set = set(coords)
    degree = {}
    for r, c in coords:
        d = sum(
            1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0) and (r + dr, c + dc) in coord_set
        )
        degree[(r, c)] = d
    endpoints = [rc for rc, d in degree.items() if d == 1]

    if len(endpoints) >= 2:
        dist0, _ = _bfs(coord_set, endpoints[0])
        reachable = [e for e in endpoints if e in dist0] or [endpoints[0]]
        far1 = max(reachable, key=lambda e: dist0[e])
        dist1, prev1 = _bfs(coord_set, far1)
        reachable2 = [e for e in endpoints if e in dist1] or [far1]
        far2 = max(reachable2, key=lambda e: dist1[e])
        path = _walk_prev(prev1, far2)
    else:
        ink_yx = np.column_stack(np.nonzero(component_mask))
        hull = cv2.convexHull(ink_yx[:, ::-1].astype(np.int32)).reshape(-1, 2)  # (x, y)
        best_pair, best_dist = (hull[0], hull[0]), -1.0
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                d = float(np.hypot(*(hull[i].astype(np.float64) - hull[j].astype(np.float64))))
                if d > best_dist:
                    best_dist, best_pair = d, (hull[i], hull[j])
        skel_arr = np.array(coords)  # (row, col)

        def _nearest_skel(xy):
            x, y = xy
            d2 = (skel_arr[:, 0] - y) ** 2 + (skel_arr[:, 1] - x) ** 2
            return tuple(skel_arr[int(np.argmin(d2))])

        start_rc, end_rc = _nearest_skel(best_pair[0]), _nearest_skel(best_pair[1])
        dist_s, prev_s = _bfs(coord_set, start_rc)
        path = _walk_prev(prev_s, end_rc) if end_rc in dist_s else [start_rc, end_rc]

    if len(path) < 2:
        return []

    pts_xy = np.array([[c, r] for r, c in path], dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts_xy, 2.5, False).reshape(-1, 2)
    points = [(float(x), float(y)) for x, y in approx]
    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        decimated = points[::stride]
        if decimated[-1] != points[-1]:
            decimated.append(points[-1])
        points = decimated

    return _rescale_points(points, skeleton.shape, expected_shape)


def _ground_drawn_points(state: Dict[str, Any], points: List[Tuple[float, float]]):
    ctx: PipelineContext = state["ctx"]
    return ground_motion_contract(MotionContract(drawn_polyline_2d=points), ctx)


def _obstacle_status_text(state: Dict[str, Any], points: List[Tuple[float, float]]) -> str:
    grounded = _ground_drawn_points(state, points)
    crossings = list(grounded.hypothesis.get("obstacle_crossings", [])) if grounded else []
    if not crossings:
        return "Path clear of detected objects."
    names = sorted({c["object_name"] for c in crossings if c.get("object_name")})
    return f"⚠ Crosses: **{', '.join(names)}**. Use *Auto-adjust around obstacles* to preview a reroute, or just redraw."


def on_stroke_drawn(state: Optional[Dict[str, Any]], editor_value: Optional[Dict[str, Any]]):
    """Fires on every completed freehand edit to the drawing canvas (bound
    to ``.input()``, user-driven only -- never fires from the programmatic
    value pushes ``analyze_scene``/``auto_adjust_around_obstacles`` make).
    Vectorizes the current brush stroke into a dense curved polyline and
    re-checks it against real obstacles, same as the old per-click handler
    did, just from a freehand stroke instead of accumulated clicks."""
    if not state:
        return [], None, ""
    ctx: PipelineContext = state["ctx"]
    points = _vectorize_editor_stroke(editor_value, (ctx.height, ctx.width))
    if len(points) < 2:
        return [], None, ""
    img_rgb = cv2.cvtColor(ctx.img_bgr, cv2.COLOR_BGR2RGB)
    preview = _redraw_canvas(img_rgb, points, ctx.extra.get("_paths_obstacle_mask"))
    status = _obstacle_status_text(state, points)
    return points, preview, status


def auto_adjust_around_obstacles(state: Optional[Dict[str, Any]], points: List[Tuple[float, float]]):
    """Explicit, user-triggered obstacle reroute (never automatic -- see
    reroute_around_obstacles's docstring). Grounds the current drawn points
    to find real obstacle crossings, then replaces only those crossing
    sub-segments; everything else in the drawn path is left untouched."""
    if not state or len(points) < 2:
        return None, points, "Draw at least two points first."
    ctx: PipelineContext = state["ctx"]
    img_rgb = cv2.cvtColor(ctx.img_bgr, cv2.COLOR_BGR2RGB)
    grounded = _ground_drawn_points(state, points)
    crossings = list(grounded.hypothesis.get("obstacle_crossings", [])) if grounded else []
    if not crossings:
        return _redraw_canvas(img_rgb, points, ctx.extra.get("_paths_obstacle_mask")), points, "No obstacle crossings detected -- path unchanged."
    adjusted = reroute_around_obstacles(points, crossings, ctx.extra.get("_paths_speed_field"))
    canvas = _redraw_canvas(img_rgb, adjusted, ctx.extra.get("_paths_obstacle_mask"))
    status = (
        f"Rerouted {len(crossings)} obstacle crossing(s) around real objects; the rest of your drawn path is "
        "unchanged. This replaces the path shown here -- Animate to use it, or draw again to discard."
    )
    return canvas, adjusted, status


def _blank_sketch_canvas() -> np.ndarray:
    w, h = _SKETCH_CANVAS_SIZE
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _redraw_sketch_canvas(regions: List[SketchRegion], current_points: List[Tuple[float, float]]) -> np.ndarray:
    canvas = _blank_sketch_canvas()
    for region in regions:
        pts = np.array([[int(x), int(y)] for x, y in region.polygon_2d], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], _SKETCH_REGION_COLORS.get(region.region_type, (180, 180, 180)))
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
        cv2.polylines(canvas, [pts], True, (60, 60, 60), 2)
        label_pt = pts.mean(axis=0).astype(int)
        cv2.putText(canvas, region.label or region.region_type, tuple(label_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(current_points):
        cv2.circle(canvas, (int(x), int(y)), 4, (52, 64, 235), -1)
        if i > 0:
            cv2.line(canvas, (int(current_points[i - 1][0]), int(current_points[i - 1][1])), (int(x), int(y)), (52, 64, 235), 2)
    return canvas


def on_sketch_canvas_click(regions: List[SketchRegion], current_points: List[Tuple[float, float]], evt: gr.SelectData):
    x, y = evt.index
    current_points = list(current_points) + [(float(x), float(y))]
    return _redraw_sketch_canvas(regions, current_points), current_points


def undo_sketch_point(regions: List[SketchRegion], current_points: List[Tuple[float, float]]):
    current_points = list(current_points)[:-1]
    return _redraw_sketch_canvas(regions, current_points), current_points


def finish_sketch_region(
    regions: List[SketchRegion], current_points: List[Tuple[float, float]], region_type: str, region_label: str,
):
    if len(current_points) < 3:
        return _redraw_sketch_canvas(regions, current_points), regions, current_points, "Need at least 3 points to close a region."
    new_region = SketchRegion(region_type=region_type, polygon_2d=list(current_points), label=(region_label or "").strip() or None)
    regions = list(regions) + [new_region]
    return _redraw_sketch_canvas(regions, []), regions, [], f"Added a '{region_type}' region ({len(regions)} total)."


def clear_sketch_scene():
    return _blank_sketch_canvas(), [], [], ""


def build_scene_from_sketch_ui(regions: List[SketchRegion]):
    """Same output shape as ``analyze_scene`` (photo path), so everything
    below it in the UI -- actor/target pickers, custom inputs, Animate --
    works identically regardless of which scene-input method was used."""
    empty = gr.update(choices=[], value=None)
    if not regions or not any(r.region_type == "floor" for r in regions):
        return empty, empty, empty, empty, empty, None, None, "Draw at least a floor region before building the scene."

    w, h = _SKETCH_CANVAS_SIZE
    scene = SketchScene(regions=regions, image_size=(w, h))
    canvas_bgr = _redraw_sketch_canvas(regions, [])
    stem = f"sketch_{uuid.uuid4().hex[:8]}"
    output_dir = _OUTPUT_ROOT / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = run_sketch_scene_pipeline(scene, canvas_bgr, stem=stem, output_dir=output_dir)
    objects = ctx.extra.get("objects", [])
    labels, id_by_label = _object_labels(objects)
    # See analyze_scene's identical fix: use the pre-computed count instead
    # of loading the entire (potentially very large) hypotheses file just
    # to report a number, then keeping it alive for the whole session.
    n_hypotheses = int(ctx.path_exports.get("path_hypotheses_count", 0))

    state = {"ctx": ctx, "id_by_label": id_by_label}
    status = (
        f"Built a scene from {len(regions)} drawn regions: **{len(objects)}** objects, **{n_hypotheses}** "
        f"path hypotheses (output: `{output_dir}`). No photo, no diffusion model, no GPU -- depth/masks came "
        f"directly from what you drew. Pick an actor/target below, then Animate."
    )
    actor_default = labels[0] if labels else None
    target_default = labels[-1] if len(labels) > 1 else None
    return (
        gr.update(choices=labels, value=actor_default),
        gr.update(choices=labels, value=target_default),
        gr.update(choices=labels, value=[]),
        gr.update(choices=labels, value=[]),
        gr.update(choices=labels, value=[]),
        state,
        cv2.cvtColor(ctx.img_bgr, cv2.COLOR_BGR2RGB),
        status,
    )


def _qa_note(result: Dict[str, Any]) -> str:
    """Formats render_qa's pass/fail summary (see render_qa.run_render_qa)
    into the same status-string pattern report_note already uses, rather
    than a separate UI element."""
    qa = result.get("qa")
    if not qa:
        return ""
    if qa.get("passed"):
        return "QA: OK"
    return "QA: " + "; ".join(qa.get("failed_checks", []))


def animate_scene(
    state: Optional[Dict[str, Any]],
    actor_label: Optional[str], target_label: Optional[str],
    fps: float, duration_s: float, tier: str,
    drawn_points: List[Tuple[float, float]],
    actor_text: str, action_text: str,
    actor_asset_img: Optional[np.ndarray],
    animation_clip_path: Optional[str],
    avoid_labels: List[str], required_labels: List[str], behind_labels: List[str],
):
    if not state:
        return None, None, "Run Analyze first."
    ctx: PipelineContext = state["ctx"]
    id_by_label: Dict[str, Any] = state["id_by_label"]

    contract = MotionContract(
        actor_object_id=id_by_label.get(actor_label) if actor_label else None,
        target_object_id=id_by_label.get(target_label) if target_label else None,
        drawn_polyline_2d=list(drawn_points) if drawn_points and len(drawn_points) >= 2 else None,
        actor_text=(actor_text or "").strip() or None,
        action_text=(action_text or "").strip() or None,
        actor_asset_rgba=_to_bgra(actor_asset_img) if actor_asset_img is not None else None,
        uploaded_animation_ref=animation_clip_path or None,
        avoid_object_ids=[id_by_label[lbl] for lbl in (avoid_labels or []) if lbl in id_by_label],
        required_object_ids=[id_by_label[lbl] for lbl in (required_labels or []) if lbl in id_by_label],
        must_render_behind_object_ids=[id_by_label[lbl] for lbl in (behind_labels or []) if lbl in id_by_label],
        duration_s=float(duration_s), fps=int(fps),
    )

    grounded = ground_motion_contract(contract, ctx)
    if grounded is None:
        reasons = "; ".join(["Could not build an animation from the given inputs."])
        return None, None, reasons

    report_note = "; ".join(grounded.report["grounded"] + grounded.report["warnings"])
    objects = ctx.extra.get("objects", [])
    object_masks_by_id = {o.get("id"): o.get("_sam2_mask_array") for o in objects if o.get("_sam2_mask_array") is not None}

    tmp_dir = Path(tempfile.mkdtemp(prefix="citv_animate_"))
    gif_path, mp4_path = str(tmp_dir / "animation.gif"), str(tmp_dir / "animation.mp4")

    if tier == TIER_3D:
        try:
            result = render_tier1_animation(
                ctx.img_bgr, grounded.hypothesis, ctx.intrinsics, object_masks_by_id,
                work_dir=tmp_dir / "blender_work", out_gif_path=gif_path, out_mp4_path=mp4_path,
                fps=int(fps), duration_s=float(duration_s), actor_attrs=grounded.actor_attrs, color_rgb=grounded.color_rgb,
                action_style=grounded.action_style, retargeted_motion=grounded.retargeted_motion,
            )
        except RuntimeError as exc:
            report_note = f"{report_note} | Tier 1 fallback to Tier 2: {exc}" if report_note else f"Tier 1 fallback to Tier 2: {exc}"
        else:
            status = f"[Tier 1, Blender, {grounded.archetype}] Rendered **{result['frame_count']}** frames at {fps:.0f} fps. {report_note} {_qa_note(result)}"
            return gif_path, mp4_path, status

    if not grounded.tier2_available:
        return None, None, f"Tier 2 can't render this actor (no real photo pixels or uploaded asset available). {report_note}"

    support_mask = ctx.extra.get("_paths_support_mask")
    result = render_animation(
        ctx.img_bgr, grounded.hypothesis, grounded.actor_mask, object_masks_by_id,
        out_gif_path=gif_path, out_mp4_path=mp4_path,
        fps=int(fps), duration_s=float(duration_s), support_mask=support_mask,
        actor_sprite_rgba=grounded.actor_asset_rgba,
    )
    status = f"[Tier 2, 2D] Rendered **{result['frame_count']}** frames ({', '.join(result['motions_used'])}) at {fps:.0f} fps. {report_note} {_qa_note(result)}"
    return gif_path, mp4_path, status


with gr.Blocks(title="CITV Studio", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# CITV Studio\nTurn any photo (or a hand-drawn sketch) into a photorealistic animation.")

    with gr.Tabs():
        with gr.Tab("Upload a photo"):
            with gr.Row():
                image_in = gr.Image(type="filepath", label="Upload a photo")
                analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Tab("Sketch a scene"):
            gr.Markdown(
                "No photo needed. Draw a **floor** region first, then any **wall**/**object** regions "
                "(click points in order, then \"Finish region\"). Depth and masks are built directly from what "
                "you draw -- no diffusion model, no GPU, no hallucinated photo."
            )
            with gr.Row():
                with gr.Column():
                    sketch_canvas = gr.Image(type="numpy", label="Click to outline a region", interactive=False, value=_blank_sketch_canvas())
                    with gr.Row():
                        sketch_region_type = gr.Radio(["floor", "wall", "object"], value="floor", label="Region type")
                        sketch_region_label = gr.Textbox(label="Label (e.g. 'a chair')")
                    with gr.Row():
                        undo_sketch_point_btn = gr.Button("Undo last point")
                        finish_region_btn = gr.Button("Finish this region")
                        clear_sketch_btn = gr.Button("Clear sketch")
                    sketch_status_md = gr.Markdown()
                with gr.Column():
                    build_scene_btn = gr.Button("Build scene from sketch", variant="primary")

    gr.Markdown("---")
    with gr.Group():
        with gr.Row():
            actor_dd = gr.Dropdown(label="Actor (who moves)", choices=[])
            target_dd = gr.Dropdown(label="Target (where to)", choices=[])

    with gr.Group():
        gr.Markdown(
            "**Draw the actor's path** -- freehand, right on the photo: curves, loops, whatever the motion "
            "actually looks like. Click without dragging to inspect what's under a point instead of drawing. "
            "The path is always followed exactly as drawn; the preview below turns a segment "
            "<span style='color:#ff8c00'>orange</span> if it crosses a real detected object. Use "
            "**Auto-adjust around obstacles** to preview a rerouted version of just the crossing segments "
            "(opt-in -- your drawn path is never changed automatically)."
        )
        draw_canvas = gr.ImageEditor(
            type="numpy", label="Draw a path", image_mode="RGBA",
            sources=(), transforms=(),
            brush=gr.Brush(colors=["rgb(235, 64, 52)"], color_mode="fixed", default_size=6),
            eraser=gr.Eraser(default_size=10),
            layers=gr.LayerOptions(allow_additional_layers=False),
            fixed_canvas=False, interactive=True,
        )
        inspect_output_md = gr.Markdown("Click a point (without dragging) to see what's there.")
        with gr.Row():
            obstacle_preview_img = gr.Image(label="Path preview (orange = crosses a detected object)", interactive=False)
            draw_obstacle_status_md = gr.Markdown()
        auto_adjust_btn = gr.Button("Auto-adjust around obstacles")

    with gr.Accordion("Actor & action (optional)", open=False):
        gr.Markdown(
            "Whatever you set here is preserved exactly. Whatever you leave unset is filled in from "
            "the scene evidence above (relations, affordances, occlusion order, the path-hypothesis library)."
        )
        with gr.Row():
            actor_text_in = gr.Textbox(label="Actor description (open-vocabulary, optional)", placeholder="e.g. a photorealistic red fox")
            action_text_in = gr.Textbox(label="Action (optional)", placeholder="e.g. trots over to the table")
        with gr.Row():
            actor_asset_in = gr.Image(type="numpy", image_mode="RGBA", label="Upload an actor image (optional)")
            animation_clip_in = gr.Video(
                label="Upload an animation clip to retarget (optional, Tier 1 / 3D only)",
                sources=["upload"],
            )
        with gr.Row():
            avoid_cb = gr.CheckboxGroup(label="Avoid these objects (optional)", choices=[])
            required_cb = gr.CheckboxGroup(label="Prefer passing near these objects (optional, soft preference)", choices=[])
            behind_cb = gr.CheckboxGroup(label="Always render behind these objects (optional)", choices=[])

    with gr.Accordion("Advanced render settings", open=False):
        with gr.Row():
            tier_radio = gr.Radio(
                [TIER_3D, TIER_2D], value=TIER_3D, label="Render tier",
                info="3D needs a local Blender install; falls back to 2D automatically if unavailable.",
            )
            fps_slider = gr.Slider(6, 30, value=24, step=1, label="FPS")
            duration_slider = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="Duration (s)")

    animate_btn = gr.Button("Animate", variant="primary")
    status_md = gr.Markdown()

    with gr.Row():
        gif_out = gr.Image(label="Animation (GIF)")
        mp4_out = gr.Video(label="Animation (MP4)")

    scene_state = gr.State()
    drawn_points_state = gr.State([])
    sketch_regions_state = gr.State([])
    sketch_points_state = gr.State([])

    analyze_btn.click(
        analyze_scene, inputs=[image_in],
        outputs=[actor_dd, target_dd, avoid_cb, required_cb, behind_cb, scene_state, draw_canvas, status_md],
        trigger_mode="once",  # a real run takes minutes with no progress feedback in the UI --
        # without this, an impatient extra click launches a second full,
        # independent pipeline run (confirmed via a real log: two complete
        # ~220s+ runs back-to-back for the same image), roughly doubling
        # wall-clock time and peak memory for nothing.
    )
    undo_sketch_point_btn.click(
        undo_sketch_point, inputs=[sketch_regions_state, sketch_points_state],
        outputs=[sketch_canvas, sketch_points_state],
    )
    sketch_canvas.select(
        on_sketch_canvas_click, inputs=[sketch_regions_state, sketch_points_state],
        outputs=[sketch_canvas, sketch_points_state],
    )
    finish_region_btn.click(
        finish_sketch_region, inputs=[sketch_regions_state, sketch_points_state, sketch_region_type, sketch_region_label],
        outputs=[sketch_canvas, sketch_regions_state, sketch_points_state, sketch_status_md],
    )
    clear_sketch_btn.click(clear_sketch_scene, outputs=[sketch_canvas, sketch_regions_state, sketch_points_state, sketch_status_md])
    build_scene_btn.click(
        build_scene_from_sketch_ui, inputs=[sketch_regions_state],
        outputs=[actor_dd, target_dd, avoid_cb, required_cb, behind_cb, scene_state, draw_canvas, status_md],
        trigger_mode="once",  # same double-submit risk as analyze_btn above
    )
    draw_canvas.select(inspect_object_at_point, inputs=[scene_state], outputs=[inspect_output_md])
    draw_canvas.input(
        on_stroke_drawn, inputs=[scene_state, draw_canvas],
        outputs=[drawn_points_state, obstacle_preview_img, draw_obstacle_status_md],
    )
    auto_adjust_btn.click(
        auto_adjust_around_obstacles, inputs=[scene_state, drawn_points_state],
        outputs=[obstacle_preview_img, drawn_points_state, draw_obstacle_status_md],
    )
    animate_btn.click(
        animate_scene,
        inputs=[
            scene_state, actor_dd, target_dd, fps_slider, duration_slider, tier_radio,
            drawn_points_state, actor_text_in, action_text_in, actor_asset_in, animation_clip_in,
            avoid_cb, required_cb, behind_cb,
        ],
        outputs=[gif_out, mp4_out, status_md],
        trigger_mode="once",  # same double-submit risk as analyze_btn above
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
