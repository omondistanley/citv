"""Hand-drawn scene input: direct construction, not diffusion translation.

Corrects an earlier design idea (a diffusion model hallucinating a whole
photorealistic room from a sketch) that was caught during review as a real
divergence from this project's grounding principle -- every other part of
this codebase treats real segmentation masks, real depth, and real photo
pixels as ground truth, and Phase 3's own `MotionContract` rule is "user
intent is preserved exactly, never invented over." A generated fake photo
as the foundation for the entire pipeline breaks that rule at the root.

Instead: the user's sketch *annotations* (drawn+labelled floor/wall/object
regions) are treated exactly like a user-drawn path elsewhere in Phase 3 --
literal, authoritative input -- and used to *directly construct* the same
plain-data structures the ML stages would have produced from a real photo
(``ctx.metric_depth``, ``ctx.extra["objects"]``, ``ctx.region_label_map``),
via simple, deterministic planar-geometry depth synthesis and polygon
rasterization. SAM2/GroundingDINO/Depth-Anything-V2 are not involved at all
for this input type -- there's no photo for photo-only models to run on,
and no diffusion model inventing one. No GPU dependency either.

Once these structures exist, every later stage (relations, hierarchy,
paths, captions) runs completely unmodified against them -- those stages
only ever read plain ``ctx`` fields and don't know or care whether the
values came from ML inference on a photo or direct construction from a
sketch.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scene_understanding.pipeline_context import PipelineContext
from scene_understanding.stages import captions_export, hierarchy_export, paths_export, relations_stage

_DEFAULT_NEAR_DEPTH_M = 1.0
_DEFAULT_FAR_DEPTH_M = 8.0
_DEFAULT_WALL_DEPTH_M = 5.0
_DEFAULT_FOV_DEGREES = 60.0


@dataclass
class SketchRegion:
    """One user-drawn, user-labelled region on the sketch canvas.

    ``region_type`` is open-vocabulary-ish in spirit but only three values
    drive actual depth-synthesis behavior: ``"floor"`` (drives the ground-
    plane depth gradient every other region anchors to), ``"wall"``
    (a roughly-constant-depth vertical plane), and ``"object"`` (a labelled
    thing standing on the floor). ``relative_depth_m`` is an optional user
    override; when absent it's inferred from the floor's own depth gradient
    at that region's position, never a hardcoded universal constant.
    """

    region_type: str  # "floor" | "wall" | "object"
    polygon_2d: List[Tuple[float, float]]
    label: Optional[str] = None
    relative_depth_m: Optional[float] = None


@dataclass
class SketchScene:
    regions: List[SketchRegion] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)  # (width, height)


def _rasterize_polygon(polygon_2d: Sequence[Tuple[float, float]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon_2d], dtype=np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _estimate_intrinsics(w: int, h: int, fov_degrees: float = _DEFAULT_FOV_DEGREES) -> Dict[str, float]:
    """Same FOV-based fallback convention used elsewhere in this codebase
    (``scene_understanding.py::_estimate_intrinsics``) when no camera
    calibration is available -- there's no EXIF/calibration for a sketch."""
    fx = (w / 2.0) / np.tan(np.radians(fov_degrees) / 2.0)
    fy = fx  # square pixels assumed, same simplification as the photo-path fallback
    return {"fx": float(fx), "fy": float(fy), "cx": w / 2.0, "cy": h / 2.0}


def _back_project(u: float, v: float, z: float, k: Dict[str, float]) -> Dict[str, float]:
    x = (u - k["cx"]) * z / k["fx"]
    y = (v - k["cy"]) * z / k["fy"]
    return {"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3)}


class _FloorDepthField:
    """Linear perspective depth gradient over a floor polygon's own vertical
    extent: nearer to the camera (larger image ``v``) is shallower depth,
    farther (smaller ``v``, toward the drawn horizon) is deeper -- the
    simplest planar-ground assumption that still gives every object
    standing on the floor a sensible, monotonic sense of distance."""

    def __init__(self, floor_mask: np.ndarray, near_depth_m: float, far_depth_m: float):
        ys, _ = np.nonzero(floor_mask)
        self.v_min = float(ys.min()) if ys.size else 0.0
        self.v_max = float(ys.max()) if ys.size else float(floor_mask.shape[0] - 1)
        self.near_depth_m = near_depth_m
        self.far_depth_m = far_depth_m

    def depth_at(self, v: float) -> float:
        span = max(1e-6, self.v_max - self.v_min)
        frac = np.clip((self.v_max - v) / span, 0.0, 1.0)  # 0 at bottom (near), 1 at top (far)
        return float(self.near_depth_m + frac * (self.far_depth_m - self.near_depth_m))

    def dense_depth_map(self, h: int, w: int) -> np.ndarray:
        vs = np.arange(h, dtype=np.float32)
        span = max(1e-6, self.v_max - self.v_min)
        frac = np.clip((self.v_max - vs) / span, 0.0, 1.0)
        depth_per_row = self.near_depth_m + frac * (self.far_depth_m - self.near_depth_m)
        return np.tile(depth_per_row[:, None], (1, w)).astype(np.float32)


def build_scene_from_sketch(scene: SketchScene) -> Dict[str, Any]:
    """Directly constructs the plain-data structures the ML stages would
    normally produce from a real photo: ``metric_depth`` (H, W float32),
    ``objects`` (same dict shape as ``stages/labelling.py`` produces --
    drop-in compatible with ``ctx.extra["objects"]``), ``region_label_map``
    (H, W int32), ``region_partition_meta``, and ``intrinsics``.
    """
    w, h = scene.image_size
    if w <= 0 or h <= 0:
        raise ValueError("SketchScene.image_size must be a positive (width, height)")

    intrinsics = _estimate_intrinsics(w, h)
    floor_regions = [r for r in scene.regions if r.region_type == "floor"]
    wall_regions = [r for r in scene.regions if r.region_type == "wall"]
    object_regions = [r for r in scene.regions if r.region_type == "object"]

    region_label_map = np.zeros((h, w), dtype=np.int32)
    region_partition_meta: List[Dict[str, Any]] = []
    metric_depth = np.full((h, w), _DEFAULT_WALL_DEPTH_M, dtype=np.float32)  # background fallback for undrawn area

    floor_field: Optional[_FloorDepthField] = None
    next_region_id = 1
    if floor_regions:
        # Only one floor is meaningful for this simple planar model; if the
        # user drew more than one, the first drawn one is authoritative and
        # the rest are ignored rather than silently averaged into something
        # that doesn't correspond to what was actually drawn.
        floor = floor_regions[0]
        floor_mask = _rasterize_polygon(floor.polygon_2d, h, w)
        far_depth = floor.relative_depth_m if floor.relative_depth_m is not None else _DEFAULT_FAR_DEPTH_M
        floor_field = _FloorDepthField(floor_mask, _DEFAULT_NEAR_DEPTH_M, far_depth)
        dense = floor_field.dense_depth_map(h, w)
        metric_depth[floor_mask] = dense[floor_mask]
        region_label_map[floor_mask] = next_region_id
        region_partition_meta.append({
            "region_id": next_region_id, "region_index": next_region_id,
            "region_type": "floor", "semantic_label": floor.label or "floor",
            "depth_stats": {"mean": float(dense[floor_mask].mean())},
        })
        next_region_id += 1

    for wall in wall_regions:
        wall_mask = _rasterize_polygon(wall.polygon_2d, h, w)
        if wall.relative_depth_m is not None:
            wall_depth = wall.relative_depth_m
        elif floor_field is not None:
            ys, _ = np.nonzero(wall_mask)
            wall_depth = floor_field.depth_at(float(ys.max())) if ys.size else _DEFAULT_WALL_DEPTH_M
        else:
            wall_depth = _DEFAULT_WALL_DEPTH_M
        metric_depth[wall_mask] = wall_depth
        region_label_map[wall_mask] = next_region_id
        region_partition_meta.append({
            "region_id": next_region_id, "region_index": next_region_id,
            "region_type": "wall", "semantic_label": wall.label or "wall",
            "depth_stats": {"mean": float(wall_depth)},
        })
        next_region_id += 1

    objects: List[Dict[str, Any]] = []
    for i, obj in enumerate(object_regions):
        obj_mask = _rasterize_polygon(obj.polygon_2d, h, w)
        if not obj_mask.any():
            continue
        ys, xs = np.nonzero(obj_mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if obj.relative_depth_m is not None:
            obj_depth = obj.relative_depth_m
        elif floor_field is not None:
            obj_depth = floor_field.depth_at(float(y1))  # bottom of the object's polygon = its floor-contact point
        else:
            obj_depth = _DEFAULT_WALL_DEPTH_M
        metric_depth[obj_mask] = obj_depth  # flat/billboard simplification -- explicit, not a full 3D reconstruction
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        coords_3d = _back_project(cx, cy, obj_depth, intrinsics)
        obj_id = f"sketch_obj_{i}"
        objects.append({
            "id": obj_id, "graph_id": obj_id,
            "label": obj.label or "object", "conf": 1.0,  # user-labelled, not a model confidence score
            "bbox": [x0, y0, x1, y1],
            "segmentor": "user_sketch",
            "depth_stats": {"median": obj_depth, "mean": obj_depth, "min": obj_depth, "max": obj_depth, "std": 0.0, "num_pixels": int(obj_mask.sum())},
            "coordinates_3d": coords_3d,
            "mask_centroid_2d": [cx, cy],
            "_sam2_mask_array": obj_mask,
        })

    return {
        "metric_depth": metric_depth,
        "objects": objects,
        "region_label_map": region_label_map,
        "region_partition_meta": region_partition_meta,
        "intrinsics": intrinsics,
    }


def _bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


class _SketchPipelineShim:
    """Stand-in for ``SceneUnderstandingPipeline`` used only for sketch-mode
    scenes, where no photo (and therefore no heavy ML models) exist at all.
    Exposes exactly the attributes/methods the later stages
    (``relations_stage``, ``hierarchy_export``, ``paths_export``,
    ``captions_export``) actually read -- no SAM2/GroundingDINO/
    Depth-Anything-V2/RAM++ is loaded or needed."""

    def __init__(self) -> None:
        self.config = None
        self._relations_pipeline = None

        class _FakeDepthEstimator:
            device = "cpu"

        self.depth_estimator = _FakeDepthEstimator()

    def _get_relations_pipeline(self):
        if self._relations_pipeline is None:
            import torch

            from scene_understanding.relations import RelationsPipeline

            # florence2_wrapper=None: Pix2SG's spatial-scaffold backend needs
            # no model weights at all; Florence2 semantic enrichment is an
            # optional add-on this pipeline correctly runs without.
            self._relations_pipeline = RelationsPipeline(torch.device("cpu"), florence2_wrapper=None)
        return self._relations_pipeline

    def _bbox_iou_xyxy(self, box_a: Sequence[float], box_b: Sequence[float]) -> float:
        return _bbox_iou_xyxy(box_a, box_b)


def run_sketch_scene_pipeline(scene: SketchScene, canvas_bgr: np.ndarray, stem: str, output_dir: Path) -> PipelineContext:
    """Builds the sketch-derived context and runs every later stage
    (relations -> hierarchy -> paths -> captions) against it, exactly the
    same stages a real photo goes through after its own depth/segmentation/
    labelling -- just skipping the photo-only ML stages that produced
    ``ctx.metric_depth``/``ctx.extra["objects"]`` in that case, since here
    they're constructed directly from the user's own drawn annotations.

    ``canvas_bgr`` is the rendered sketch image itself (line art) -- used as
    ``ctx.img_bgr`` so image-derived signals (e.g. the cost map's edge
    layer) key off the user's own drawn structure, which is meaningful
    input here, not a substitute photo.
    """
    built = build_scene_from_sketch(scene)
    h, w = canvas_bgr.shape[:2]
    ctx = PipelineContext(
        image_path=Path(f"{stem}_sketch.png"), output_dir=output_dir, stem=stem,
        timestamp=datetime.datetime.now().isoformat(),
        img_bgr=canvas_bgr, img_rgb=cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB),
        height=h, width=w, intrinsics=built["intrinsics"], metric_depth=built["metric_depth"],
        region_label_map=built["region_label_map"], region_partition_meta=built["region_partition_meta"],
    )
    ctx.extra["objects"] = built["objects"]

    pipeline_shim = _SketchPipelineShim()
    ctx = relations_stage.run(pipeline_shim, ctx)
    ctx = hierarchy_export.run(pipeline_shim, ctx)
    ctx = paths_export.run(pipeline_shim, ctx)
    ctx = captions_export.run(pipeline_shim, ctx)
    return ctx
