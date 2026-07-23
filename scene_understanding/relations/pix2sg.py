"""Pix2SG — canonical implementation (synced from scene_understanding legacy module)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch


class Pix2SGWrapper:
    """
    Spatial relation scaffold + Florence-2 semantic enrichment.
    Layer 1: pixel IoU / depth-axis / centroid direction.
    Layer 2: Florence-2 RED/BLUE colour-overlay captions for overlapping pairs.
    See docs/LABELLING_AND_RELATIONS.md for formulas and predicate table.
    """
    def __init__(
        self,
        device: torch.device,
        triplets_dir: str = "pix2sg_triplets",
        max_relations_per_object: int = 8,
        mask_overlap_thresh: float = 0.05,
        depth_near_threshold: float = 1.0,
        depth_far_threshold: float = 3.0,
        florence2: Optional["Florence2Wrapper"] = None,
        relation_min_mask_overlap: float = 0.05,
        region_relation_mode: str = "all",
        relation_bbox_touch_margin_px: int = 2,
    ):
        self.device = device
        print("Initializing Pix2SG...")
        self.model = None
        self.triplets_dir = Path(triplets_dir)
        self.max_relations_per_object = max(1, int(max_relations_per_object))
        self._mask_overlap_thresh = float(mask_overlap_thresh)
        self._depth_near_threshold = float(depth_near_threshold)
        self._depth_far_threshold = float(depth_far_threshold)
        # Fix 5.6: Florence-2 semantic enrichment
        self._florence2 = florence2
        self._relation_min_mask_overlap = float(relation_min_mask_overlap)
        self._region_relation_mode = str(region_relation_mode)
        self._relation_bbox_touch_margin_px = max(0, int(relation_bbox_touch_margin_px))
        self.backend = "spatial_scaffold"
        self.active = True
        self.disabled_reason = ""
        # Populated per-call by predict() -- raw per-layer triplets before
        # dedup, for debug/QA export (SCENE_GRAPH_DEEP_DIVE.md §8 item 5).
        self.last_debug: Dict[str, List[Dict[str, Any]]] = {
            "precomputed": [], "spatial_scaffold": [], "florence2": [], "final_deduped": [],
        }
        if self.triplets_dir.exists():
            self.backend = "precomputed_triplets"
            print(f"Pix2SG precomputed triplets backend enabled: {self.triplets_dir.resolve()}")
        else:
            self.disabled_reason = (
                f"No precomputed triplets dir at {self.triplets_dir.resolve()}. "
                "Using spatial scaffold backend."
            )
            print(f"Pix2SG notice: {self.disabled_reason}")

    def is_active(self) -> bool:
        return bool(self.active)

    def status(self) -> Dict[str, Any]:
        return {
            "active": self.is_active(),
            "backend": self.backend,
            "reason": self.disabled_reason,
        }

    def _load_precomputed_triplets(self, image_stem: str) -> List[Dict[str, Any]]:
        if self.backend != "precomputed_triplets":
            return []
        json_path = self.triplets_dir / f"{image_stem}.json"
        if not json_path.exists():
            return []
        try:
            with open(json_path, "r") as f:
                payload = json.load(f)
            triplets = payload.get("triplets", payload)
            if not isinstance(triplets, list):
                return []
            cleaned: List[Dict[str, Any]] = []
            for t in triplets:
                if not isinstance(t, dict):
                    continue
                sub = str(t.get("sub", "")).strip()
                pred = str(t.get("pred", "")).strip()
                obj = str(t.get("obj", "")).strip()
                if not (sub and pred and obj):
                    continue
                cleaned.append({
                    "sub": sub.lower(),
                    "pred": pred.lower(),
                    "obj": obj.lower(),
                    "score": float(t.get("score", 1.0)),
                    "sub_id": t.get("sub_id"),
                    "obj_id": t.get("obj_id"),
                })
            return cleaned
        except Exception as e:
            print(f"Pix2SG precomputed triplets parse failed ({json_path}): {e}")
            return []

    @staticmethod
    def _center(box: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _get_centroid(obj: Dict[str, Any]) -> Tuple[float, float]:
        """Return (cx, cy) preferring mask_centroid_2d over bbox center.

        When an object was matched to a SAM2 mask, mask_centroid_2d holds the
        centre-of-mass of the actual foreground pixels — more accurate than the
        midpoint of the bounding box, especially for non-rectangular objects.
        """
        mc = obj.get("mask_centroid_2d")
        if mc and len(mc) == 2:
            return float(mc[0]), float(mc[1])
        box = obj.get("bbox", [])
        if len(box) >= 4:
            return Pix2SGWrapper._center(box)
        return 0.0, 0.0

    @staticmethod
    def _bboxes_xyxy_might_touch(
        box_a: List[float],
        box_b: List[float],
        margin_px: int,
    ) -> bool:
        """True if axis-aligned boxes could touch/overlap after expanding each side by margin_px."""
        try:
            x1a, y1a, x2a, y2a = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
            x1b, y1b, x2b, y2b = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
        except (IndexError, TypeError, ValueError):
            return True
        m = float(max(0, int(margin_px)))
        if (x2a + m) < (x1b - m) or (x2b + m) < (x1a - m):
            return False
        if (y2a + m) < (y1b - m) or (y2b + m) < (y1a - m):
            return False
        return True

    @staticmethod
    def _spatial_predicate_bbox(
        box_a: List[float],
        box_b: List[float],
        image_w: int,
        image_h: int,
        iou_func,
    ) -> str:
        """Bbox-based spatial predicate — kept as fallback when masks are unavailable."""
        iou = float(iou_func(box_a, box_b))
        if iou >= 0.1:
            return "overlapping"
        ax, ay = Pix2SGWrapper._center(box_a)
        bx, by = Pix2SGWrapper._center(box_b)
        dx = bx - ax
        dy = by - ay
        if abs(dx) >= abs(dy):
            return "left_of" if dx > 0 else "right_of"
        return "above" if dy > 0 else "below"

    @staticmethod
    def _resize_to_match(mask_bin: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        if mask_bin.shape[:2] == shape_hw:
            return mask_bin
        return cv2.resize(mask_bin.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0

    @classmethod
    def _containment_predicate(
        cls, sub_m: np.ndarray, obj_m: np.ndarray, inter: int,
        contain_ratio_thresh: float = 0.92, size_ratio_thresh: float = 1.1,
    ) -> Optional[str]:
        """``sub`` contains ``obj`` (or vice versa) when the smaller mask is
        almost entirely inside the larger one -- same containment math as
        ``regions/mask_hierarchy.py`` (kept independent/inline here rather than
        imported, so this predicate doesn't depend on hierarchy-stage ordering)."""
        sub_area, obj_area = int(sub_m.sum()), int(obj_m.sum())
        if sub_area <= 0 or obj_area <= 0:
            return None
        if obj_area > int(sub_area * size_ratio_thresh) and (inter / max(sub_area, 1)) >= contain_ratio_thresh:
            return "inside_of"  # sub is inside obj
        if sub_area > int(obj_area * size_ratio_thresh) and (inter / max(obj_area, 1)) >= contain_ratio_thresh:
            return "contains"  # sub contains obj
        return None

    @staticmethod
    def _resting_on_predicate(sub_m: np.ndarray, obj_m: np.ndarray, contact_margin_px: int = 3) -> Optional[str]:
        """``sub`` rests on ``obj`` when sub's mask sits directly above obj's
        mask with real horizontal overlap and near-zero vertical gap between
        sub's bottom edge and obj's top edge (contact, not just proximity)."""
        sub_ys, sub_xs = np.nonzero(sub_m)
        obj_ys, obj_xs = np.nonzero(obj_m)
        if sub_ys.size == 0 or obj_ys.size == 0:
            return None
        x_lo, x_hi = max(sub_xs.min(), obj_xs.min()), min(sub_xs.max(), obj_xs.max())
        if x_hi < x_lo:
            return None  # no horizontal overlap at all
        sub_bottom_by_col: Dict[int, int] = {}
        for x, y in zip(sub_xs.tolist(), sub_ys.tolist()):
            if x_lo <= x <= x_hi and (x not in sub_bottom_by_col or y > sub_bottom_by_col[x]):
                sub_bottom_by_col[x] = y
        obj_top_by_col: Dict[int, int] = {}
        for x, y in zip(obj_xs.tolist(), obj_ys.tolist()):
            if x_lo <= x <= x_hi and (x not in obj_top_by_col or y < obj_top_by_col[x]):
                obj_top_by_col[x] = y
        shared_cols = set(sub_bottom_by_col) & set(obj_top_by_col)
        if not shared_cols:
            return None
        gaps = [obj_top_by_col[x] - sub_bottom_by_col[x] for x in shared_cols]
        contact_cols = sum(1 for g in gaps if 0 <= g <= contact_margin_px)
        # Require genuine contact along a meaningful fraction of the shared span,
        # not just one accidental touching column.
        if contact_cols >= max(3, int(0.15 * len(shared_cols))):
            return "resting_on"
        return None

    def _spatial_predicate_mask(self, sub: Dict[str, Any], obj: Dict[str, Any]) -> str:
        """Return spatial predicate using pixel-mask IoU, containment,
        contact, and depth-weighted centroids. Priority order: overlapping ->
        containment -> resting_on -> touching (near-contact, no overlap) ->
        depth-aware adjacency (in_front_of/behind) -> 2D positional fallback."""
        sub_mask = sub.get("_sam2_mask_array")
        obj_mask = obj.get("_sam2_mask_array")

        if sub_mask is not None and obj_mask is not None:
            sub_m = np.asarray(sub_mask) > 0
            obj_m = self._resize_to_match(np.asarray(obj_mask) > 0, sub_m.shape[:2])
            inter = int(np.logical_and(sub_m, obj_m).sum())
            union = int(np.logical_or(sub_m, obj_m).sum())
            if union > 0 and (inter / (union + 1e-8)) >= self._mask_overlap_thresh:
                return "overlapping"

            containment = self._containment_predicate(sub_m, obj_m, inter)
            if containment is not None:
                return containment

            resting = self._resting_on_predicate(sub_m, obj_m)
            if resting is not None:
                return resting

            # Near-contact ("touching"): no IoU overlap, but a small dilation
            # of one mask reaches the other -- objects that sit right next to
            # each other on a shelf/floor, not overlapping but not far apart either.
            kernel = np.ones((11, 11), np.uint8)  # ~5px reach each direction
            sub_dilated = cv2.dilate(sub_m.astype(np.uint8), kernel) > 0
            if np.logical_and(sub_dilated, obj_m).any():
                return "touching"

        sub_z = sub.get("coordinates_3d", {}).get("z")
        obj_z = obj.get("coordinates_3d", {}).get("z")
        if sub_z is not None and obj_z is not None:
            depth_diff = abs(float(obj_z) - float(sub_z))
            if depth_diff >= self._depth_far_threshold:
                return "in_front_of" if float(sub_z) < float(obj_z) else "behind"

        sx, sy = self._get_centroid(sub)
        ox, oy = self._get_centroid(obj)
        dx, dy = ox - sx, oy - sy
        if abs(dx) >= abs(dy):
            return "left_of" if dx > 0 else "right_of"
        return "above" if dy > 0 else "below"

    def _build_spatial_scaffold_triplets(
        self,
        detections: List[Dict[str, Any]],
        image_h: int,
        image_w: int,
        iou_func,
    ) -> List[Dict[str, Any]]:
        if len(detections) < 2:
            return []
        triplets: List[Dict[str, Any]] = []
        for i, sub in enumerate(detections):
            sub_label = str(sub.get("label", "object")).lower()
            sub_id = sub.get("graph_id", sub.get("id"))
            sub_box = sub.get("bbox", [])
            if len(sub_box) < 4:
                continue
            scored_neighbors: List[Tuple[float, int]] = []
            # Use mask centroid for neighbour distance sorting (falls back to bbox center)
            sx, sy = self._get_centroid(sub)
            for j, obj in enumerate(detections):
                if i == j:
                    continue
                obj_box = obj.get("bbox", [])
                if len(obj_box) < 4:
                    continue
                ox, oy = self._get_centroid(obj)
                dist = float(np.hypot(ox - sx, oy - sy))
                scored_neighbors.append((dist, j))
            scored_neighbors.sort(key=lambda x: x[0])
            for _, j in scored_neighbors[: self.max_relations_per_object]:
                obj = detections[j]
                obj_label = str(obj.get("label", "object")).lower()
                obj_id = obj.get("graph_id", obj.get("id"))
                obj_box = obj.get("bbox", [])
                if self._region_relation_mode == "intra_region_only":
                    ri_s = int(sub.get("region_index", 0) or 0)
                    ri_o = int(obj.get("region_index", 0) or 0)
                    if ri_s > 0 and ri_o > 0 and ri_s != ri_o:
                        continue
                # Use mask-native predicate when at least one object has a mask;
                # otherwise fall back to the original bbox-based predicate.
                if sub.get("_sam2_mask_array") is not None or obj.get("_sam2_mask_array") is not None:
                    pred = self._spatial_predicate_mask(sub, obj)
                else:
                    pred = self._spatial_predicate_bbox(sub_box, obj_box, image_w, image_h, iou_func)
                if pred in {"contains", "inside_of"}:
                    score = 0.90  # strongest geometric signal: near-total mask containment
                elif pred == "resting_on":
                    score = 0.85  # verified bottom/top contact + horizontal overlap
                elif pred == "overlapping":
                    score = 0.85
                elif pred == "touching":
                    score = 0.72  # near-contact but no overlap/containment/support verified
                elif pred in {"in_front_of", "behind"}:
                    score = 0.75
                else:
                    score = 0.70
                ri_s = int(sub.get("region_index", 0) or 0)
                ri_o = int(obj.get("region_index", 0) or 0)
                tier = "intra_region"
                if ri_s > 0 and ri_o > 0 and ri_s != ri_o:
                    tier = "inter_region"
                triplets.append({
                    "sub": sub_label,
                    "pred": pred,
                    "obj": obj_label,
                    "sub_id": sub_id,
                    "obj_id": obj_id,
                    "score": score,
                    "relation_tier": tier,
                })
        return triplets

    @staticmethod
    def _triplet_dedupe_key(t: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            str(t.get("sub_id", "") or ""),
            str(t.get("obj_id", "") or ""),
            str(t.get("pred", "")).strip().lower(),
            str(t.get("source_layer", "") or ""),
        )

    @classmethod
    def _dedupe_triplets(cls, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, str, str, str]] = set()
        out: List[Dict[str, Any]] = []
        for t in triplets:
            k = cls._triplet_dedupe_key(t)
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out

    def predict(
        self,
        image: np.ndarray,
        image_stem: str = "",
        detections: Optional[List[Dict[str, Any]]] = None,
        iou_func=None,
    ) -> List[Dict[str, Any]]:
        """
        Generate relation triplets.

        Layer 1: Precomputed triplets (when present) OR spatial scaffold.
        Layer 2: Florence-2 semantic enrichment only when ``self._florence2`` is set
          and active (see ``PreprocessConfig.florence2_relation_enabled``); merged on
          top of layer 1 / precomputed triplets when enabled.
        """
        self.last_debug = {"precomputed": [], "spatial_scaffold": [], "florence2": [], "final_deduped": []}
        if not self.is_active():
            return []
        if detections is None:
            return []
        h, w = image.shape[:2]
        if iou_func is None:
            iou_func = lambda _a, _b: 0.0

        triplets: List[Dict[str, Any]] = []
        precomputed = self._load_precomputed_triplets(image_stem) if image_stem else []
        self.last_debug["precomputed"] = list(precomputed)
        if precomputed:
            triplets.extend(precomputed)
        if not triplets:
            scaffold = self._build_spatial_scaffold_triplets(detections, h, w, iou_func)
            self.last_debug["spatial_scaffold"] = list(scaffold)
            triplets = scaffold

        if self._florence2 is not None and self._florence2.active and len(detections) >= 2:
            florence_triplets = self._enrich_with_florence2(image, detections)
            self.last_debug["florence2"] = list(florence_triplets)
            triplets.extend(florence_triplets)

        deduped = self._dedupe_triplets(triplets)
        self.last_debug["final_deduped"] = list(deduped)
        return deduped

    def _enrich_with_florence2(
        self,
        image_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For every pair of objects whose bboxes may touch (cheap gate) and whose
        masks overlap above relation_min_mask_overlap, query Florence-2 for a
        semantic predicate.

        Skips pairs whose expanded bboxes cannot intersect (no touching/overlap
        possible) before any mask resize or IoU work, then applies mask IoU.

        Returns additional triplets with source_layer="florence2".
        """
        extra: List[Dict[str, Any]] = []
        n = len(detections)
        for i in range(n):
            sub = detections[i]
            sub_mask = sub.get("_sam2_mask_array")
            if sub_mask is None:
                continue
            sub_m = np.asarray(sub_mask) > 0
            sub_label = str(sub.get("label", "object"))
            sub_id = sub.get("graph_id") or sub.get("id")

            for j in range(n):
                if i == j:
                    continue
                obj = detections[j]
                obj_mask = obj.get("_sam2_mask_array")
                if obj_mask is None:
                    continue
                sub_box = sub.get("bbox") or []
                obj_box = obj.get("bbox") or []
                if len(sub_box) >= 4 and len(obj_box) >= 4:
                    if not self._bboxes_xyxy_might_touch(
                        sub_box, obj_box, self._relation_bbox_touch_margin_px
                    ):
                        continue

                obj_m = np.asarray(obj_mask) > 0
                if sub_m.shape != obj_m.shape:
                    obj_m = cv2.resize(
                        obj_m.astype(np.uint8),
                        (sub_m.shape[1], sub_m.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                inter = int(np.logical_and(sub_m, obj_m).sum())
                union = int(np.logical_or(sub_m, obj_m).sum())
                if union == 0 or (inter / union) < self._relation_min_mask_overlap:
                    continue

                obj_label = str(obj.get("label", "object"))
                obj_id = obj.get("graph_id") or obj.get("id")

                if self._region_relation_mode == "intra_region_only":
                    ri_s = int(sub.get("region_index", 0) or 0)
                    ri_o = int(obj.get("region_index", 0) or 0)
                    if ri_s > 0 and ri_o > 0 and ri_s != ri_o:
                        continue

                pred = self._florence2.predict_relation(
                    image_bgr, sub_m, obj_m, sub_label, obj_label
                )
                if pred is not None:
                    _ri_a = int(sub.get("region_index", 0) or 0)
                    _ri_b = int(obj.get("region_index", 0) or 0)
                    _tier = "intra_region"
                    if _ri_a > 0 and _ri_b > 0 and _ri_a != _ri_b:
                        _tier = "inter_region"
                    extra.append({
                        "sub": sub_label.lower(),
                        "pred": pred,
                        "obj": obj_label.lower(),
                        "sub_id": sub_id,
                        "obj_id": obj_id,
                        "score": 0.75,
                        "source_layer": "florence2",
                        "relation_tier": _tier,
                    })
        return extra
