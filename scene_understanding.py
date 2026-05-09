"""
scene_understanding.py — 3D scene graph generation pipeline.
See docs/SEGMENTATION.md, docs/DEPTH_ACCURACY.md, docs/LABELLING_AND_RELATIONS.md.

The ``if __name__ == "__main__"`` entrypoint delegates to ``scene_understanding.pipeline``
(see ``PreprocessConfig.scene_pipeline_mode`` and ``CITV_SCENE_PIPELINE_MODE``).
"""
# Ensure "import sam2" resolves to repo's sam2/sam2/ (not citv/sam2 as repo root) so build_sam check passes
import sys
from pathlib import Path as _Path
_script_dir = _Path(__file__).resolve().parent
_sam2_repo_root = _script_dir / "sam2"
if (_sam2_repo_root / "sam2").is_dir():
    _sp = str(_sam2_repo_root)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import cv2
import gc
import copy
import json
import os
import time
import numpy as np
import re
import math
import urllib.request
import urllib.error
import urllib.parse
import torch
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

try:
    import transformers.safetensors_conversion as _stc

    def _disable_auto_conversion(*args, **kwargs):
        return None

    _stc.auto_conversion = _disable_auto_conversion
except Exception:
    pass

# Suppress transformers deprecation warning that has a logging format bug
# (passes FutureWarning as a format arg to a message with no %s → TypeError crash)
import logging as _logging
_logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(_logging.ERROR)

# Import existing DepthEstimator
from depth import DepthEstimator
from scene_understanding.export import (
    build_animation_components_bundle,
    build_insertion_bundle,
    build_path_fields_explainer_image,
    build_path_fields_legend_payload,
    build_trajectory_bundle,
    build_traversability_speed_map,
    is_soft_obstacle,
    k_diverse_grid_paths,
)
# FMM primitives — used to replace A* / Dijkstra in the object-pair hot
# loop. ``time_of_arrival`` + ``backtrace_from_T`` amortise one FMM solve
# across every source point that shares a goal; ``k_diverse_from_T``
# generates diverse polylines without re-solving.
from scene_understanding.pathing.semantic_fmm import (
    backtrace_from_T as _fmm_backtrace_from_T,
    k_diverse_from_T as _fmm_k_diverse_from_T,
    time_of_arrival as _fmm_time_of_arrival,
    time_of_arrival_from_speed as _fmm_time_of_arrival_from_speed,
)
from scene_understanding.pathing.scene_context import (
    build_scene_context as _build_scene_context,
    validate_scene_context as _validate_scene_context,
)
from scene_understanding.pathing.calibration import calibrate_path_policy as _calibrate_path_policy

_partitioner_path = _Path(__file__).resolve().parent / "scene_understanding" / "regions" / "partitioner.py"
_spec_reg = importlib.util.spec_from_file_location("_citv_regions_partitioner", _partitioner_path)
_reg_mod = importlib.util.module_from_spec(_spec_reg) if _spec_reg and _spec_reg.loader else None
if _reg_mod is not None and _spec_reg is not None and _spec_reg.loader is not None:
    sys.modules["_citv_regions_partitioner"] = _reg_mod
    _spec_reg.loader.exec_module(_reg_mod)
    partition_depth_regions = _reg_mod.partition_depth_regions
    label_map_to_bgr = _reg_mod.label_map_to_bgr
    majority_region_index = _reg_mod.majority_region_index
else:
    partition_depth_regions = None  # type: ignore[misc, assignment]
    label_map_to_bgr = None  # type: ignore[misc, assignment]
    majority_region_index = None  # type: ignore[misc, assignment]


# Soft priors: plausible metric depth range (min_m, max_m) for label tokens
# --- Object vs region workflow parity (metadata keys are parallel for regions) ---
# Objects: segmentation_image, sam2_segmentation_image, sam2_tinted_overlay_image, depth_mask_A, ...
# Regions:  region_segmentation_image, region_sam2_segmentation_image, region_tinted_overlay_image,
#           region_depth_mask_* (same JSON file as objects with entity_kind), layers/hierarchy include regions.
# Legacy:   regions_json, regions_image, regions_overlay_image (unchanged paths).

_LABEL_DEPTH_PRIORS_M: Dict[str, Tuple[float, float]] = {
    "person": (0.2, 100.0),
    "people": (0.2, 100.0),
    "man": (0.2, 100.0),
    "woman": (0.2, 100.0),
    "child": (0.2, 100.0),
    "baby": (0.2, 100.0),
    "sky": (2.0, 5000.0),
    "cloud": (2.0, 5000.0),
    "road": (0.5, 200.0),
    "floor": (0.2, 80.0),
    "ground": (0.2, 500.0),
}


def _load_bgr_image(path: Path) -> np.ndarray:
    """
    Load image as BGR numpy array.
    Uses OpenCV first, then PIL (with optional HEIF opener) as fallback.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is not None:
        return img_bgr

    pil_error = None
    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass

        from PIL import Image
        with Image.open(path) as img_pil:
            img_rgb = np.array(img_pil.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        pil_error = e

    raise ValueError(
        f"Could not decode image: {path}. "
        "OpenCV returned None. If this is HEIF/HEIC content, convert it to JPEG/PNG "
        "or install pillow-heif for PIL decoding. "
        f"PIL fallback error: {pil_error}"
    )

# -----------------------------------------------------------------------------
# Model wrappers — canonical implementations in the scene_understanding package
# -----------------------------------------------------------------------------
from scene_understanding.core.prompting import refresh_gdino_query_if_configured
from scene_understanding.labeling import Florence2MLXWrapper, Florence2Wrapper, RAMPlusPlusWrapper
from scene_understanding.relations import Pix2SGWrapper
from scene_understanding.segmentation import GroundedSAM2Wrapper


# Phase B.10 — stages at which we want MPS cache eviction after sync to
# flatten allocator headroom between heavy workloads. Other stages still get
# ``torch.mps.synchronize()`` so their wall-clock is honest, but do not free
# the cache (which would force re-allocation moments later).
def _pipeline_worker_entrypoint(worker_args: tuple) -> list:
    """Top-level worker used by multiprocessing.Pool (spawn-safe).

    Phase G — must live at module scope so the ``spawn`` start method can
    pickle it by name. Each worker constructs its own DepthEstimator +
    SceneUnderstandingPipeline, runs its share of images, finalises a
    per-worker timing log, and returns structured status tuples.
    """
    import os as _os
    from pathlib import Path as _Path
    cfg_local, img_paths, out_dir = worker_args
    try:
        if bool(getattr(cfg_local, "cpu_benchmark_mode", False)):
            cfg_local.device = "cpu"  # type: ignore[attr-defined]
            cfg_local.models_dtype = "fp32"  # type: ignore[attr-defined]
            cfg_local.rampp_prefer_mps = False  # type: ignore[attr-defined]
        import importlib as _il
        from depth import DepthEstimator as _DE
        _pm = _il.import_module("scene_understanding.pipeline")
        _SP = _pm.SceneUnderstandingPipeline
        _de = _DE(cfg_local, first_image=img_paths[0])
        _pipe = _SP(_de, config=cfg_local)
        from scene_understanding.observability import RunTimingLogger as _RTL, hash_config as _hc
        _lg = _RTL(
            log_dir=_Path(getattr(cfg_local, "timing_log_dir", "logs")) / f"worker_{_os.getpid()}",
            device=str(getattr(cfg_local, "device", "cpu")),
            models_dtype=str(getattr(cfg_local, "models_dtype", "fp32")),
            workers=int(getattr(cfg_local, "pipeline_max_workers", 1)),
            config_hash=_hc(cfg_local),
        )
        if hasattr(_pipe, "attach_timing_logger"):
            _pipe.attach_timing_logger(_lg)
        results: list = []
        for _ip in img_paths:
            try:
                _pipe.process_image(str(_ip), str(out_dir))
                results.append((str(_ip), True, None))
            except Exception as _e:
                results.append((str(_ip), False, str(_e)))
        try:
            _md, _csv = _lg.finalize()
            results.append(("__timing__", True, f"{_md};{_csv}"))
        except Exception:
            pass
        try:
            _pipe.close()
        except Exception:
            pass
        return results
    except Exception as _we:
        return [(str(p), False, f"worker init failed: {_we}") for p in img_paths]


_STAGE_MPS_EMPTY_CACHE = frozenset(
    {"depth", "segmentation", "labellers_load", "mask_labelling", "pix2sg_predict"}
)


class SceneUnderstandingPipeline:
    def __init__(
        self,
        depth_estimator: DepthEstimator,
        intrinsics: Optional[Dict] = None,
        depth_mask_modes: Optional[List[str]] = None,
        config: Optional[Any] = None,
    ):
        """
        Args:
            depth_estimator: Initialized instance from depth.py
            intrinsics: Dict {'fx', 'fy', 'cx', 'cy'}. If None, will be auto-estimated.
            depth_mask_modes: Matching modes; only "A" is supported (detection-first). Default ["A"].
            config: Optional PreprocessConfig for SAM2 paths and AMG params. If None, depth-mask uses defaults.
        """
        self.depth_estimator = depth_estimator
        self.device = depth_estimator.device
        self.fixed_intrinsics = intrinsics
        if config is not None and hasattr(config, "depth_mask_matching_modes"):
            self.depth_mask_modes = [m for m in list(config.depth_mask_matching_modes) if m == "A"] or ["A"]
        else:
            self.depth_mask_modes = depth_mask_modes if depth_mask_modes is not None else ["A"]
        self.config = config
        self.require_any_relation_source = (
            bool(getattr(config, "require_any_relation_source", True)) if config is not None else True
        )
        self.mask_iou_match_thresh = (
            float(getattr(config, "mask_iou_match_thresh", 0.1)) if config is not None else 0.1
        )
        self.pix2sg_mask_overlap_thresh = (
            float(getattr(config, "pix2sg_mask_overlap_thresh", 0.05)) if config is not None else 0.05
        )
        self.pix2sg_depth_near_threshold = (
            float(getattr(config, "pix2sg_depth_near_threshold", 1.0)) if config is not None else 1.0
        )
        # Camera intrinsics
        self.camera_fx = getattr(config, "camera_fx", None) if config is not None else None
        self.camera_fy = getattr(config, "camera_fy", None) if config is not None else None
        self.camera_cx = getattr(config, "camera_cx", None) if config is not None else None
        self.camera_cy = getattr(config, "camera_cy", None) if config is not None else None
        self.camera_fov_degrees = float(getattr(config, "camera_fov_degrees", 60.0)) if config is not None else 60.0
        # Depth accuracy
        self.mask_erosion_kernel_size = int(getattr(config, "mask_erosion_kernel_size", 5)) if config is not None else 5
        self.depth_central_fraction = float(getattr(config, "depth_central_fraction", 0.5)) if config is not None else 0.5
        self.depth_scale_factor = float(getattr(config, "depth_scale_factor", 10.0)) if config is not None else 10.0
        # SAM2 post-hoc filter
        self.sam2_post_filter_min_stability = float(getattr(config, "sam2_post_filter_min_stability", 0.0)) if config is not None else 0.0
        self.sam2_post_filter_min_pred_iou = float(getattr(config, "sam2_post_filter_min_pred_iou", 0.0)) if config is not None else 0.0
        self.sam2_post_filter_min_area_px = int(getattr(config, "sam2_post_filter_min_area_px", 1000)) if config is not None else 1000
        self.sam2_post_filter_max_area_fraction = float(getattr(config, "sam2_post_filter_max_area_fraction", 0.35)) if config is not None else 0.35
        self.grounded_sam2_min_conf_for_stage3 = (
            float(
                getattr(
                    config,
                    "grounded_sam2_min_conf_for_stage3",
                    getattr(config, "grounding_dino_box_thresh", 0.25),
                )
            )
            if config is not None
            else 0.25
        )
        pix2sg_triplets_dir = (
            str(getattr(config, "pix2sg_triplets_dir", "pix2sg_triplets")) if config is not None else "pix2sg_triplets"
        )
        pix2sg_max_relations_per_object = (
            int(getattr(config, "pix2sg_spatial_max_relations_per_object", 8)) if config is not None else 8
        )
        pix2sg_depth_far_threshold = (
            float(getattr(config, "pix2sg_depth_far_threshold", 3.0)) if config is not None else 3.0
        )

        # Fix 5.7 — depth accuracy params
        self.depth_adaptive_erosion = bool(getattr(config, "depth_adaptive_erosion", True)) if config is not None else True
        self.depth_outlier_sigma = float(getattr(config, "depth_outlier_sigma", 2.0)) if config is not None else 2.0
        self.depth_transparency_check = bool(getattr(config, "depth_transparency_check", True)) if config is not None else True
        self.depth_transparency_threshold = float(getattr(config, "depth_transparency_threshold", 0.15)) if config is not None else 0.15

        # Fix 5.2 — calibration file
        self._calibration: Optional[Dict] = None
        cal_file = getattr(config, "camera_calibration_file", None) if config is not None else None
        if cal_file:
            self._calibration = self._load_calibration(cal_file)
        self.apply_undistortion = bool(getattr(config, "apply_undistortion", True)) if config is not None else True

        # Labellers — lazy-loaded on first use, unloaded after Stage 5 to save VRAM
        self._florence2_model_id = getattr(config, "florence2_model", "microsoft/Florence-2-large") if config is not None else "microsoft/Florence-2-large"
        self._florence2_backend = str(getattr(config, "florence2_backend", "hf") or "hf").strip().lower()
        self._florence2_label_enabled = bool(getattr(config, "florence2_label_enabled", True)) if config is not None else True
        self._rampp_enabled = bool(getattr(config, "rampp_enabled", True)) if config is not None else True
        self._rampp_checkpoint_path = getattr(config, "rampp_checkpoint_path", None) if config is not None else None
        self._rampp_repo_path = getattr(config, "rampp_repo_path", None) if config is not None else None
        self._rampp_backend = str(getattr(config, "rampp_backend", "rampp") or "rampp").strip().lower()
        self._rampp_image_size = int(getattr(config, "rampp_image_size", 384)) if config is not None else 384
        self._rampp_vit = str(getattr(config, "rampp_vit", "swin_l")) if config is not None else "swin_l"
        self._rampp_default_conf = float(getattr(config, "rampp_default_confidence", 0.70)) if config is not None else 0.70
        self._rampp_max_tags = int(getattr(config, "rampp_max_tags", 8)) if config is not None else 8
        self._rampp_clip_fallback_enabled = bool(getattr(config, "rampp_clip_fallback_enabled", True)) if config is not None else True
        self._rampp_quantize_dynamic_int8 = bool(getattr(config, "rampp_quantize_dynamic_int8", True)) if config is not None else True
        self._rampp_clip_candidate_labels = (
            list(getattr(config, "rampp_clip_candidate_labels", [])) if config is not None else []
        )
        self._mask_label_skip_secondary_when_gdino_specific = (
            bool(getattr(config, "mask_label_skip_secondary_when_gdino_specific", True))
            if config is not None
            else True
        )
        self.save_track_grounded_sam2 = bool(getattr(config, "save_track_grounded_sam2", True)) if config is not None else True
        self.save_track_amg = bool(getattr(config, "save_track_amg", False)) if config is not None else False
        self.save_track_combined = bool(getattr(config, "save_track_combined", False)) if config is not None else False
        self.track_dir_grounded_sam2 = str(getattr(config, "track_dir_grounded_sam2", "grounded_sam2")) if config is not None else "grounded_sam2"
        self.track_dir_amg = str(getattr(config, "track_dir_amg", "amg")) if config is not None else "amg"
        self.track_dir_combined = str(getattr(config, "track_dir_combined", "combined")) if config is not None else "combined"
        self.export_caption_prompt_bundle = bool(getattr(config, "export_caption_prompt_bundle", True)) if config is not None else True
        self.export_track_comparison_prompt = bool(getattr(config, "export_track_comparison_prompt", True)) if config is not None else True
        self.export_hybrid_captions = bool(getattr(config, "export_hybrid_captions", True)) if config is not None else True
        self.export_florence_object_captions = bool(getattr(config, "export_florence_object_captions", True)) if config is not None else True
        self.export_fusion_scene_captions = bool(getattr(config, "export_fusion_scene_captions", True)) if config is not None else True
        self.caption_files_only = bool(getattr(config, "caption_files_only", True)) if config is not None else True
        self.caption_max_objects_per_track = int(getattr(config, "caption_max_objects_per_track", 64)) if config is not None else 64
        self.florence2: Optional[Any] = None
        self.rampp: Optional[RAMPlusPlusWrapper] = None
        self._rampp_runtime_disabled = False
        _unload_each = bool(getattr(config, "unload_labellers_after_each_image", False)) if config is not None else False
        _labeller_retention = "unloaded after each image" if _unload_each else "kept loaded between images"
        print(f"Labellers (Florence-2, RAM++) load on first use and are {_labeller_retention}.")

        # Fix 5.6 — Pix2SG with Florence-2 enrichment
        relation_min_mask_overlap = float(getattr(config, "relation_min_mask_overlap", 0.05)) if config is not None else 0.05
        _region_rel_mode = str(getattr(config, "region_relation_mode", "all")) if config is not None else "all"
        _rel_bbox_margin = int(getattr(config, "relation_bbox_touch_margin_px", 2)) if config is not None else 2
        print(f"  [Pipeline] torch device: {self.device}")
        self.pix2sg = Pix2SGWrapper(
            self.device,
            triplets_dir=pix2sg_triplets_dir,
            max_relations_per_object=pix2sg_max_relations_per_object,
            mask_overlap_thresh=self.pix2sg_mask_overlap_thresh,
            depth_near_threshold=self.pix2sg_depth_near_threshold,
            depth_far_threshold=pix2sg_depth_far_threshold,
            florence2=None,  # injected lazily at process_image() time after load
            relation_min_mask_overlap=relation_min_mask_overlap,
            region_relation_mode=_region_rel_mode,
            relation_bbox_touch_margin_px=_rel_bbox_margin,
        )
        self.depth_sigma_clip_scope = (
            str(getattr(config, "depth_sigma_clip_scope", "mask")) if config is not None else "mask"
        )
        self.regions_enabled = bool(getattr(config, "regions_enabled", False)) if config is not None else False
        self._regions_k = int(getattr(config, "regions_k", 4)) if config is not None else 4
        self._regions_min_region_px = int(getattr(config, "regions_min_region_px", 500)) if config is not None else 500
        self._regions_blur_sigma = float(getattr(config, "regions_blur_sigma", 0.0)) if config is not None else 0.0
        self._regions_seed = int(getattr(config, "regions_seed", 42)) if config is not None else 42
        self._regions_use_hardlink = bool(getattr(config, "regions_use_hardlink_for_track_copies", False)) if config is not None else False
        self._regions_rampp_crops_enabled = (
            bool(getattr(config, "regions_rampp_crops_enabled", False)) if config is not None else False
        )
        self._append_region_layer_relations = (
            bool(getattr(config, "append_region_layer_relations", False)) if config is not None else False
        )
        self._regions_reject_implausible = (
            bool(getattr(config, "regions_reject_implausible_labels", False)) if config is not None else False
        )
        self._relation_source_status = self._collect_relation_source_status()
        self._print_relation_source_status()
        self._assert_relation_sources_or_fail()

        self._reuse_cached_depth = bool(getattr(config, "reuse_cached_depth", True)) if config is not None else True

        # Memory-pressure guard — on ~8 GB MPS systems we can't keep Florence-2,
        # RAM++, GDINO+SAM2 and Depth V2 co-resident without macOS paging us
        # into a 60%-Metal-wait-on-swap death spiral (see config docstring).
        # Store the kwargs needed to reconstruct the heavy models lazily.
        self._release_heavy_models = (
            bool(getattr(config, "release_heavy_models_after_segmentation", True))
            if config is not None
            else True
        )
        # Batch dispatcher sets this to True before each non-final image so
        # ``_release_heavy_models_after_segmentation`` keeps Depth/GDINO/SAM2
        # resident instead of reloading them (~4 s saved per image).
        self._has_more_images_queued: bool = False
        self._sam2_reinit_kwargs: Optional[Dict[str, Any]] = None

        # Fix 5.3 — GroundedSAM2 (GDINO + SAM2 prompted)
        if self.config is not None:
            ckpt = getattr(self.config, "sam2_checkpoint_path", "sam2/checkpoints/sam2.1_hiera_large.pt")
            cfg = getattr(self.config, "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l")
            gdino_model = getattr(self.config, "grounding_dino_model", "IDEA-Research/grounding-dino-base")
            gdino_box_thresh = float(getattr(self.config, "grounding_dino_box_thresh", 0.30))
            gdino_text_thresh = float(getattr(self.config, "grounding_dino_text_thresh", 0.25))
            gdino_query = getattr(self.config, "grounding_dino_text_query",
                "person. man. woman. child. animal. dog. cat. car. truck. bicycle. motorcycle. bus. "
                "chair. table. desk. sofa. bed. shelf. cabinet. door. window. floor. wall. ceiling. "
                "bottle. cup. bowl. plate. glass. fork. knife. spoon. pot. pan. "
                "laptop. phone. keyboard. monitor. television. remote. camera. "
                "bag. backpack. suitcase. box. basket. "
                "book. paper. pen. clock. lamp. mirror. painting. "
                "tree. plant. flower. grass. sky. road. building. sign.")
            max_side = getattr(self.config, "sam2_amg_max_image_side", 1280)
            self._sam2_reinit_kwargs = dict(
                device=self.device,
                sam2_checkpoint_path=ckpt,
                sam2_model_cfg=cfg,
                gdino_model_id=gdino_model,
                box_thresh=gdino_box_thresh,
                text_thresh=gdino_text_thresh,
                text_query=gdino_query,
                max_image_side=max_side,
                torch_compile_enabled=bool(getattr(self.config, "torch_compile_enabled", False)),
            )
            self.sam2_wrapper = GroundedSAM2Wrapper(**self._sam2_reinit_kwargs)
        else:
            self._sam2_reinit_kwargs = dict(
                device=self.device,
                sam2_checkpoint_path="sam2/checkpoints/sam2.1_hiera_large.pt",
                sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l",
                torch_compile_enabled=bool(getattr(self.config, "torch_compile_enabled", False)) if self.config is not None else False,
            )
            self.sam2_wrapper = GroundedSAM2Wrapper(**self._sam2_reinit_kwargs)
        if not self.sam2_wrapper.active:
            raise RuntimeError("Grounded SAM2 (GDINO + SAM2 prompted) failed to initialize.")

        from scene_understanding.core.pipeline_settings import attach_settings_from_pipeline

        attach_settings_from_pipeline(self)

        # Timing instrumentation: a NullTimingLogger by default so existing
        # callers are unaffected. attach_timing_logger() swaps in a real
        # RunTimingLogger for folder runs.
        from scene_understanding.observability import NullTimingLogger  # local import to avoid cycles
        self._timing_logger = NullTimingLogger()
        self._image_pyramid = None  # Phase B.6 — re-populated per-image in process_image

        # Phase F — async writer pool for JSON/PNG outputs. All deliverables
        # are produced on the GPU-heavy main thread; disk writes are pushed to
        # this background pool so the next pipeline stage can proceed while
        # the OS flushes the previous artefacts. ``_drain_writers`` blocks for
        # completion at the end of ``process_image`` so per-image outputs are
        # guaranteed to be on disk before the caller returns.
        from concurrent.futures import ThreadPoolExecutor
        self._writer_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="citv-writer"
        )
        self._pending_writes: list = []

    def attach_timing_logger(self, logger) -> None:
        """Inject a :class:`RunTimingLogger` (or None) for stage timing."""
        from scene_understanding.observability import NullTimingLogger
        self._timing_logger = logger if logger is not None else NullTimingLogger()

    def _record_stage_time(self, image_key: str, stage: str, start: float) -> float:
        """Stop the timer for ``stage``, record it, and return a fresh start.

        Kept small so instrumentation adds a single line at each prof site.
        Always records into ``self._timing_logger`` regardless of the legacy
        ``scene_pipeline_profile`` gate, so the timing log is populated even
        when CITV_PROFILE is off.

        Phase B.10 — when the logger reports an MPS device, issue
        ``torch.mps.synchronize()`` *before* stopping the clock so GPU work
        that is still in-flight is counted against this stage. Optionally
        also empty the MPS cache to flatten peak allocator headroom between
        stages.
        """
        try:
            if getattr(self._timing_logger, "device_is_mps", False):
                try:
                    import torch as _torch  # local import keeps cold start fast
                    if _torch.backends.mps.is_available():
                        _torch.mps.synchronize()  # type: ignore[attr-defined]
                        if stage in _STAGE_MPS_EMPTY_CACHE:
                            try:
                                _torch.mps.empty_cache()  # type: ignore[attr-defined]
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        now = time.perf_counter()
        try:
            self._timing_logger.record(image_key, stage, (now - start) * 1000.0)
        except Exception:
            pass
        return now

    def _submit_write(self, fn, *args, **kwargs) -> None:
        """Schedule ``fn(*args, **kwargs)`` on the writer pool.

        Phase F — used to dispatch JSON/PNG writes asynchronously. Any
        exception is surfaced during :meth:`_drain_writers` so failed writes
        never pass silently. Falls back to a synchronous call if the pool
        has been shut down (e.g. after teardown).
        """
        try:
            fut = self._writer_pool.submit(fn, *args, **kwargs)
            self._pending_writes.append(fut)
        except Exception:
            try:
                fn(*args, **kwargs)
            except Exception:
                raise

    def _drain_writers(self) -> None:
        """Block until all scheduled writes finish; re-raise write errors.

        Called at the end of :meth:`process_image` so every per-image
        artefact is guaranteed durable before the next image starts.
        """
        pending = self._pending_writes
        if not pending:
            return
        errors: list = []
        for fut in pending:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._pending_writes = []
        if errors:
            raise errors[0]

    def close(self) -> None:
        """Shut down background writer pool gracefully."""
        try:
            self._drain_writers()
        finally:
            try:
                self._writer_pool.shutdown(wait=True)
            except Exception:
                pass

    def _collect_relation_source_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Pix2SG": self.pix2sg.status(),
        }

    _GENERIC_OBJECT_LABELS = {
        "",
        "object",
        "objects",
        "unknown",
        "unlabeled",
        "entity",
        "entities",
        "thing",
        "things",
        "item",
        "items",
        "part",
        "parts",
        "stuff",
        "scene",
        "image",
        "photo",
        "picture",
    }

    _ATTRIBUTE_LIKE_LABELS = {
        "red",
        "blue",
        "green",
        "yellow",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "pink",
        "purple",
        "orange",
        "line",
        "strip",
        "color",
        "colour",
        "rectangle",
        "shape",
        "alphabet",
        "draw",
        "sky",
    }

    @staticmethod
    def _normalize_label_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def _split_label_candidates(cls, value: Any) -> List[str]:
        text = cls._normalize_label_text(value)
        if not text:
            return []
        candidates: List[str] = []
        for chunk in re.split(r"[|,/]+", text):
            chunk = cls._normalize_label_text(chunk)
            if chunk:
                candidates.append(chunk)
        if text and text not in candidates:
            candidates.insert(0, text)
        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    @classmethod
    def _is_generic_label(cls, label: Any) -> bool:
        normalized = cls._normalize_label_text(label)
        return (not normalized) or normalized in cls._GENERIC_OBJECT_LABELS

    @classmethod
    def _is_attribute_like_label(cls, label: Any) -> bool:
        normalized = cls._normalize_label_text(label)
        return normalized in cls._ATTRIBUTE_LIKE_LABELS

    @classmethod
    def _score_name_candidate(
        cls,
        label: str,
        source_name: str,
        confidence: float = 0.0,
    ) -> float:
        normalized = cls._normalize_label_text(label)
        if not normalized:
            return -1.0
        score = float(confidence)
        source_boost = {
            "GroundedSAM2": 3.0,
            "GroundingDINO": 3.0,
            "Florence2": 2.4,
            "Florence-2": 2.4,
            "RAM++": 2.0,
        }
        score += source_boost.get(source_name, 1.0)
        if not cls._is_generic_label(normalized):
            score += 1.5
        if not cls._is_attribute_like_label(normalized):
            score += 1.0
        if " " in normalized:
            score += 0.2
        if len(normalized) > 8:
            score += 0.1
        return score

    @classmethod
    def _infer_category(
        cls,
        canonical_name: str,
        aliases: Optional[List[str]] = None,
    ) -> str:
        values = [cls._normalize_label_text(canonical_name)]
        values.extend(cls._normalize_label_text(v) for v in (aliases or []))
        vocab = " ".join(v for v in values if v)

        category_hints = [
            ("person", {"person", "man", "woman", "child", "boy", "girl", "hand", "face", "arm", "selfie", "muscle"}),
            ("animal", {"animal", "dog", "cat", "bird", "tail", "horse", "cow"}),
            ("electronics", {"game controller", "controller", "ipod", "phone", "charger", "electronic", "laptop", "screen", "keyboard", "mouse"}),
            ("tool", {"tool", "brush", "paint brush", "pencil", "gun", "knife", "hammer"}),
            ("text_or_graphic", {"alphabet", "draw", "text", "letter", "number", "rectangle", "line"}),
            ("material_or_color", {"red", "blue", "green", "yellow", "brown", "pink", "purple", "black", "white", "gray", "grey"}),
            ("background", {"sky", "wall", "floor", "ceiling", "background"}),
            ("household", {"toilet paper", "pillow", "pad", "palette"}),
        ]
        for category, hints in category_hints:
            if any(hint in vocab for hint in hints):
                return category
        return "object"

    @classmethod
    def _build_source_labels(
        cls,
        grounded_label: str,
        grounded_caption: str,
        grounded_confidence: float,
        florence_label: str,
        florence_caption: str,
        rampp_label: str,
        rampp_caption: str,
        rampp_tags: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        return {
            "GroundedSAM2": {
                "label": cls._normalize_label_text(grounded_label),
                "caption": str(grounded_caption or grounded_label or ""),
                "confidence": round(float(grounded_confidence or 0.0), 4),
            },
            "Florence2": {
                "label": cls._normalize_label_text(florence_label),
                "caption": str(florence_caption or florence_label or ""),
            },
            "RAM++": {
                "label": cls._normalize_label_text(rampp_label),
                "caption": str(rampp_caption or rampp_label or ""),
                "tags": [cls._normalize_label_text(tag) for tag in (rampp_tags or []) if cls._normalize_label_text(tag)],
            },
        }

    @classmethod
    def _choose_mask_name_fields(
        cls,
        grounded_label: str,
        grounded_caption: str,
        grounded_confidence: float,
        florence_label: str,
        florence_caption: str,
        rampp_label: str,
        rampp_caption: str,
        rampp_tags: List[str],
        fallback_label: str,
    ) -> Dict[str, Any]:
        source_labels = cls._build_source_labels(
            grounded_label=grounded_label,
            grounded_caption=grounded_caption,
            grounded_confidence=grounded_confidence,
            florence_label=florence_label,
            florence_caption=florence_caption,
            rampp_label=rampp_label,
            rampp_caption=rampp_caption,
            rampp_tags=rampp_tags,
        )
        candidates: List[Tuple[float, str]] = []
        candidate_sources = [
            ("GroundedSAM2", grounded_label, grounded_confidence),
            ("GroundedSAM2", grounded_caption, grounded_confidence),
            ("Florence2", florence_label, 0.75),
            ("Florence2", florence_caption, 0.75),
            ("RAM++", rampp_label, 0.7),
            ("RAM++", rampp_caption, 0.7),
        ]
        candidate_sources.extend(("RAM++", tag, 0.65) for tag in (rampp_tags or []))

        for source_name, raw_value, conf in candidate_sources:
            for candidate in cls._split_label_candidates(raw_value):
                candidates.append((cls._score_name_candidate(candidate, source_name, conf), candidate))

        fallback = cls._normalize_label_text(fallback_label) or "object"
        candidates.append((cls._score_name_candidate(fallback, "fallback", 0.0), fallback))
        candidates.sort(key=lambda item: (-item[0], item[1]))

        aliases: List[str] = []
        for _, candidate in candidates:
            if cls._is_generic_label(candidate):
                continue
            if candidate in aliases:
                continue
            aliases.append(candidate)
            if len(aliases) >= 8:
                break

        canonical_name = aliases[0] if aliases else fallback
        if cls._is_attribute_like_label(canonical_name):
            for alias in aliases[1:]:
                if not cls._is_attribute_like_label(alias):
                    canonical_name = alias
                    break

        if canonical_name not in aliases and not cls._is_generic_label(canonical_name):
            aliases.insert(0, canonical_name)

        return {
            "name": canonical_name,
            "canonical_name": canonical_name,
            "aliases": aliases,
            "category": cls._infer_category(canonical_name, aliases),
            "source_labels": source_labels,
        }

    @staticmethod
    def _mask_area(mask: Any) -> int:
        if mask is None:
            return 0
        return int(np.sum(np.asarray(mask) > 0))

    @staticmethod
    def _object_centroid_xy(obj: Dict[str, Any]) -> Tuple[int, int]:
        mc = obj.get("mask_centroid_2d")
        if mc and len(mc) == 2:
            return int(mc[0]), int(mc[1])
        bbox = obj.get("bbox", [0, 0, 0, 0])
        return (int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2

    @staticmethod
    def _write_json_sync(payload: Dict[str, Any], path: Path) -> None:
        """Synchronous JSON writer backing both the sync and async entry points.

        Phase F — prefer ``orjson`` with ``OPT_INDENT_2 | OPT_SERIALIZE_NUMPY``
        for a ~4× serialisation speed-up on scene graphs dominated by
        numpy arrays. ``orjson.dumps`` emits bytes, which we write directly;
        the resulting file is byte-identical to ``json.dump(indent=2)`` for
        plain Python payloads and supersets it for numpy payloads (whose
        floats used to fail silently via ``default=str``). Falls back to
        ``json.dump`` when ``orjson`` is not installed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import orjson  # type: ignore
            opt = orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
            with open(path, "wb") as f:
                f.write(orjson.dumps(payload, option=opt, default=str))
            return
        except Exception:
            pass
        with open(path, "w", encoding="utf-8") as f:
            # ``default=str`` keeps numpy ndarrays (and other non-JSON types)
            # from crashing the stdlib fallback when ``orjson`` is unavailable.
            json.dump(payload, f, indent=2, default=str)

    def _write_json(self, payload: Dict[str, Any], path: Path) -> None:
        """Write ``payload`` to ``path``; offloaded to the writer pool.

        Phase F — every per-image JSON write is dispatched to
        :attr:`_writer_pool` so the disk flush overlaps with the next
        pipeline stage. :meth:`_drain_writers` is called at the end of
        ``process_image`` to guarantee durability.
        """
        try:
            self._submit_write(self._write_json_sync, payload, path)
        except Exception:
            self._write_json_sync(payload, path)

    @staticmethod
    def _write_png_sync(
        path: Path, image_bgr: np.ndarray, compression_level: Optional[int] = None
    ) -> None:
        """Synchronous PNG writer used by the pool and legacy sync callers.

        Phase F — supports an optional zlib compression level via
        ``IMWRITE_PNG_COMPRESSION`` (0–9). Defaults to OpenCV's default
        (3) when ``compression_level`` is ``None``. Intermediate QA
        artefacts can pass ``compression_level=1`` for ~3× faster
        encoding at a modest file-size cost; final deliverables should
        keep the default so compatibility is preserved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        params: List[int] = []
        if compression_level is not None:
            lvl = int(max(0, min(9, compression_level)))
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), lvl]
        cv2.imwrite(str(path), image_bgr, params)

    def _write_png_intermediate(self, path: Path, image_bgr: np.ndarray) -> None:
        """Offload a PNG QA artefact write using the cheap compression level.

        Phase F — intermediate QA PNGs (depth visualisation, relation
        overlays, semantic cost heatmaps / provenance maps / layer exports)
        don't need maximum compression. Using zlib level 1 is ~3× faster
        to encode while inflating on-disk size by only ~25%. Callers that
        produce user-facing deliverables should use :meth:`_write_png`
        (default OpenCV compression) instead.
        """
        try:
            lvl = int(getattr(self.config, "png_intermediate_compression", 1)) if self.config is not None else 1
        except Exception:
            lvl = 1
        self._write_png(path, image_bgr, compression_level=lvl)

    def _write_png(
        self, path: Path, image_bgr: np.ndarray, compression_level: Optional[int] = None
    ) -> None:
        """Offload a PNG write to the writer pool.

        Phase F — callers that do not need the file available immediately
        (the overwhelming majority of QA PNGs) should route through this
        helper so disk flushes overlap with GPU compute for the next
        stage. :meth:`_drain_writers` is called at the end of
        ``process_image``.
        """
        try:
            self._submit_write(self._write_png_sync, path, image_bgr, compression_level)
        except Exception:
            self._write_png_sync(path, image_bgr, compression_level)

    def _collect_export_relations(self, objects_3d: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        id_to_obj = {str(obj.get("id")): obj for obj in objects_3d}
        relations: List[Dict[str, Any]] = []
        seen = set()

        for subject in objects_3d:
            subject_id = str(subject.get("id"))
            subject_label = str(subject.get("label", "object"))
            subject_name = str(subject.get("canonical_name", subject_label))
            subject_centroid = list(self._object_centroid_xy(subject))
            for source_name, source_payload in subject.get("sources", {}).items():
                for rel in source_payload.get("relations", []):
                    target_id = rel.get("target_id")
                    if target_id is None:
                        continue
                    target_obj = id_to_obj.get(str(target_id))
                    target_label = rel.get("target_label") or (
                        target_obj.get("label", "unknown") if target_obj is not None else "unknown"
                    )
                    target_name = (
                        target_obj.get("canonical_name", target_label)
                        if target_obj is not None
                        else rel.get("target_label") or target_label
                    )
                    entry = {
                        "subject_id": subject_id,
                        "subject_label": subject_label,
                        "subject_name": subject_name,
                        "predicate": str(rel.get("predicate", "related_to")),
                        "object_id": target_id,
                        "object_label": str(target_label),
                        "object_name": str(target_name),
                        "object_caption": str(rel.get("target_caption", "")),
                        "source": source_name,
                        "score": round(float(rel.get("score", 0.0)), 4) if "score" in rel else None,
                        "subject_centroid": subject_centroid,
                        "object_centroid": list(self._object_centroid_xy(target_obj)) if target_obj is not None else None,
                    }
                    if rel.get("relation_tier"):
                        entry["relation_tier"] = str(rel["relation_tier"])
                    dedupe_key = (
                        entry["subject_id"],
                        entry["predicate"],
                        str(entry["object_id"]),
                        entry["source"],
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    relations.append(entry)
        return relations

    def _build_mask_hierarchy(self, objects_3d: List[Dict[str, Any]]) -> Dict[str, Any]:
        from scene_understanding.regions.mask_hierarchy import build_mask_hierarchy

        cfg = self.config
        return build_mask_hierarchy(
            objects_3d,
            hierarchy_enable_region_region_edges=bool(
                getattr(cfg, "hierarchy_enable_region_region_edges", False)
            )
            if cfg
            else False,
            hierarchy_region_region_containment_min=float(
                getattr(cfg, "hierarchy_region_region_containment_min", 0.97)
            )
            if cfg
            else 0.97,
        )

    def _build_layers_payload(
        self,
        objects_3d: List[Dict[str, Any]],
        region_metas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # Phase 4: collect object z-values and region z-values separately.
        # Quantile thresholds (q1, q2) are derived from objects only — region mean depths
        # are spread across the full depth range by K-means design, so mixing them into
        # the pool shifts q1/q2 away from where objects actually concentrate and causes
        # systematic layer mislabelling. Regions are then classified against the same
        # object-derived thresholds so objects and their containing regions always agree.
        obj_z_values: List[float] = []
        for obj in objects_3d:
            z_val = obj.get("coordinates_3d", {}).get("z")
            if z_val is not None:
                obj_z_values.append(float(z_val))
        region_entries: List[Dict[str, Any]] = []
        if region_metas:
            for r in region_metas:
                zm = (r.get("depth_stats") or {}).get("mean")
                if zm is not None:
                    region_entries.append({"meta": r, "z": float(zm)})

        if not obj_z_values and not region_entries:
            for obj in objects_3d:
                obj["layer_type"] = "unassigned"
            for r in region_metas or []:
                r["layer_type"] = "unassigned"
            return {
                "ordering": [],
                "bands": [],
                "depth_quantiles": {},
                "regions": [],
            }

        if obj_z_values:
            quantiles = np.quantile(np.asarray(obj_z_values, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0])
            q1, q2 = float(quantiles[0]), float(quantiles[1])
        else:
            # No objects — fall back to region depths so the payload is still useful.
            region_z_vals = [e["z"] for e in region_entries]
            quantiles = np.quantile(np.asarray(region_z_vals, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0])
            q1, q2 = float(quantiles[0]), float(quantiles[1])
        band_objects: Dict[str, List[Dict[str, Any]]] = {
            "foreground": [],
            "midground": [],
            "background": [],
        }

        for obj in objects_3d:
            z_val = float(obj.get("coordinates_3d", {}).get("z", 0.0))
            if z_val <= q1:
                layer_type = "foreground"
            elif z_val <= q2:
                layer_type = "midground"
            else:
                layer_type = "background"
            obj["layer_type"] = layer_type
            band_objects[layer_type].append(obj)

        regions_layer_out: List[Dict[str, Any]] = []
        for entry in region_entries:
            r = entry["meta"]
            z_val = entry["z"]
            if z_val <= q1:
                layer_type = "foreground"
            elif z_val <= q2:
                layer_type = "midground"
            else:
                layer_type = "background"
            r["layer_type"] = layer_type
            regions_layer_out.append({
                "region_id": str(r.get("id", "")),
                "region_index": int(r.get("region_index", 0) or 0),
                "layer_type": layer_type,
                "z_mean": round(z_val, 4),
                "object_ids": list(r.get("object_ids", [])),
            })

        bands = []
        for layer_type, members in band_objects.items():
            if not members:
                continue
            z_band = [float(member.get("coordinates_3d", {}).get("z", 0.0)) for member in members]
            bands.append({
                "layer_type": layer_type,
                "object_ids": [str(member.get("id")) for member in members],
                "count": len(members),
                "z_min": round(float(min(z_band)), 4),
                "z_max": round(float(max(z_band)), 4),
            })

        ordering = [
            {
                "object_id": str(obj.get("id")),
                "z": round(float(obj.get("coordinates_3d", {}).get("z", 0.0)), 4),
                "layer_type": obj.get("layer_type", "unassigned"),
                "entity_kind": "object",
            }
            for obj in sorted(
                objects_3d,
                key=lambda item: float(item.get("coordinates_3d", {}).get("z", 0.0)),
            )
        ]
        for row in sorted(regions_layer_out, key=lambda x: x["z_mean"]):
            ordering.append({
                "object_id": row["region_id"],
                "z": row["z_mean"],
                "layer_type": row["layer_type"],
                "entity_kind": "region",
            })

        return {
            "ordering": ordering,
            "bands": bands,
            "depth_quantiles": {
                "foreground_max_z": round(q1, 4),
                "midground_max_z": round(q2, 4),
            },
            "regions": regions_layer_out,
        }

    def _apply_object_relation_fields(
        self,
        objects_3d: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> None:
        id_to_obj = {str(obj.get("id")): obj for obj in objects_3d}
        for obj in objects_3d:
            obj.setdefault("contains", list(obj.get("child_object_ids", [])))
            parent_id = obj.get("parent_object_id")
            obj.setdefault("contained_by", [parent_id] if parent_id else [])
            obj.setdefault("occludes", [])
            obj.setdefault("occluded_by", [])

        for rel in relations:
            sub_id = str(rel.get("subject_id"))
            obj_id = rel.get("object_id")
            if obj_id is None:
                continue
            obj_id = str(obj_id)
            subject = id_to_obj.get(sub_id)
            target = id_to_obj.get(obj_id)
            if subject is None or target is None:
                continue

            predicate = str(rel.get("predicate", ""))
            if predicate in {"contains", "around"}:
                if obj_id not in subject["contains"]:
                    subject["contains"].append(obj_id)
                if sub_id not in target["contained_by"]:
                    target["contained_by"].append(sub_id)
            elif predicate in {"inside_of", "inside"}:
                if obj_id not in subject["contained_by"]:
                    subject["contained_by"].append(obj_id)
                if sub_id not in target["contains"]:
                    target["contains"].append(sub_id)
            elif predicate == "in_front_of":
                if obj_id not in subject["occludes"]:
                    subject["occludes"].append(obj_id)
                if sub_id not in target["occluded_by"]:
                    target["occluded_by"].append(sub_id)
            elif predicate == "behind":
                if obj_id not in subject["occluded_by"]:
                    subject["occluded_by"].append(obj_id)
                if sub_id not in target["occludes"]:
                    target["occludes"].append(sub_id)

    def _derive_scene_additions(
        self,
        objects_3d: List[Dict[str, Any]],
        region_hierarchy_supplements: Optional[List[Dict[str, Any]]] = None,
        region_metas_for_layers: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        relations = self._collect_export_relations(objects_3d)
        hier_nodes: List[Dict[str, Any]] = list(objects_3d)
        if region_hierarchy_supplements:
            hier_nodes.extend(region_hierarchy_supplements)
        hierarchy = self._build_mask_hierarchy(hier_nodes)

        def _hier_label(nodes: List[Dict[str, Any]], oid: str) -> str:
            for o in nodes:
                if str(o.get("id")) == str(oid):
                    return str(o.get("label", "object"))
            return "object"

        def _hier_name(nodes: List[Dict[str, Any]], oid: str) -> str:
            for o in nodes:
                if str(o.get("id")) == str(oid):
                    return str(o.get("canonical_name", o.get("label", "object")))
            return "object"

        def _hier_centroid(nodes: List[Dict[str, Any]], oid: str) -> List[float]:
            node = next((o for o in nodes if str(o.get("id")) == str(oid)), None)
            if node is None:
                return [0.0, 0.0]
            xy = self._object_centroid_xy(node)
            return [float(xy[0]), float(xy[1])]

        existing_relation_keys = {
            (str(rel.get("subject_id")), str(rel.get("predicate")), str(rel.get("object_id")))
            for rel in relations
        }
        for edge in hierarchy.get("edges", []):
            parent_id = str(edge.get("parent_object_id"))
            child_id = str(edge.get("child_object_id"))
            if (parent_id, "contains", child_id) not in existing_relation_keys:
                relations.append({
                    "subject_id": parent_id,
                    "subject_label": _hier_label(hier_nodes, parent_id),
                    "subject_name": _hier_name(hier_nodes, parent_id),
                    "predicate": "contains",
                    "object_id": child_id,
                    "object_label": _hier_label(hier_nodes, child_id),
                    "object_name": _hier_name(hier_nodes, child_id),
                    "object_caption": "",
                    "source": "mask_hierarchy",
                    "score": edge.get("containment_ratio"),
                    "subject_centroid": _hier_centroid(hier_nodes, parent_id),
                    "object_centroid": _hier_centroid(hier_nodes, child_id),
                })
            if (child_id, "inside_of", parent_id) not in existing_relation_keys:
                relations.append({
                    "subject_id": child_id,
                    "subject_label": _hier_label(hier_nodes, child_id),
                    "subject_name": _hier_name(hier_nodes, child_id),
                    "predicate": "inside_of",
                    "object_id": parent_id,
                    "object_label": _hier_label(hier_nodes, parent_id),
                    "object_name": _hier_name(hier_nodes, parent_id),
                    "object_caption": "",
                    "source": "mask_hierarchy",
                    "score": edge.get("containment_ratio"),
                    "subject_centroid": _hier_centroid(hier_nodes, child_id),
                    "object_centroid": _hier_centroid(hier_nodes, parent_id),
                })

        layers = self._build_layers_payload(objects_3d, region_metas_for_layers)
        self._apply_object_relation_fields(objects_3d, relations)
        return relations, hierarchy, layers

    def _save_relations_map(
        self,
        image_bgr: np.ndarray,
        objects_3d: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        path: Path,
        regions_meta: Optional[List[Dict[str, Any]]] = None,
        view: str = "all",  # "all" | "objects_only" | "regions_only"
    ) -> None:
        canvas = image_bgr.copy()
        id_to_obj = {str(obj.get("id")): obj for obj in objects_3d}
        h, w = canvas.shape[:2]
        occupied: List[Tuple[int, int, int, int]] = []

        def _overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

        def _place_label(base_x: int, base_y: int, text: str, scale: float = 0.4) -> Tuple[int, int, Tuple[int, int, int, int]]:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            candidates = [
                (base_x + 8, base_y - 8),
                (base_x + 8, base_y + 14),
                (base_x - tw - 8, base_y - 8),
                (base_x - tw - 8, base_y + 14),
                (base_x + 12, base_y - 22),
                (base_x - tw - 12, base_y - 22),
            ]
            for tx, ty in candidates:
                x1 = max(0, tx - 2)
                y1 = max(0, ty - th - 2)
                x2 = min(w - 1, tx + tw + 2)
                y2 = min(h - 1, ty + 2)
                box = (x1, y1, x2, y2)
                if any(_overlap(box, o) for o in occupied):
                    continue
                occupied.append(box)
                return tx, ty, box
            tx, ty = candidates[0]
            box = (
                max(0, tx - 2),
                max(0, ty - th - 2),
                min(w - 1, tx + tw + 2),
                min(h - 1, ty + 2),
            )
            occupied.append(box)
            return tx, ty, box

        # Predicate colour table — 12 maximally-distinct hues, one per predicate class.
        # BGR order. Each predicate gets a unique hue so crossing arrows stay readable.
        _pred_col: Dict[str, Tuple[int, int, int]] = {
            "in_front_of": (  0, 230,   0),  # pure green
            "behind":      (  0,   0, 230),  # pure red
            "left_of":     (230, 200,   0),  # sky blue
            "right_of":    (  0, 140, 255),  # orange
            "above":       (  0, 240, 240),  # yellow
            "below":       (255,   0, 210),  # magenta
            "adjacent_to": (220, 220, 220),  # white-gray
            "overlapping": (255, 200,   0),  # cyan
            "contains":    (200,   0, 200),  # purple
            "inside_of":   (170,   0, 170),  # dark purple
            "on":          (  0, 215, 255),  # gold
            "holds":       (  0, 200, 120),  # teal
        }

        # Draw object anchors — semantic name only, no internal graph ID prefix.
        if view in ("all", "objects_only"):
            for obj in objects_3d:
                name = str(obj.get("canonical_name", obj.get("label", "object")))
                ox, oy = self._object_centroid_xy(obj)
                ox = int(min(max(0, ox), w - 1))
                oy = int(min(max(0, oy), h - 1))
                # Filled circle with thin black ring for contrast against any background.
                cv2.circle(canvas, (ox, oy), 6, (0, 220, 80), -1)
                cv2.circle(canvas, (ox, oy), 6, (0, 0, 0), 1)
                tx, ty, (x1, y1, x2, y2) = _place_label(ox, oy, name, scale=0.44)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), -1)
                cv2.putText(canvas, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Region endpoints — semantic name only, no "region_N:" prefix.
        if view in ("all", "regions_only") and regions_meta:
            for r in regions_meta:
                rid = str(r.get("id", ""))
                if not rid:
                    continue
                r_sem = (
                    str(r.get("semantic_label", "") or r.get("canonical_name", "") or r.get("type", "region"))
                    .strip().lower() or "region"
                )
                c = r.get("centroid_2d_px") or [w // 2, h // 2]
                ox = int(min(max(0, float(c[0])), w - 1))
                oy = int(min(max(0, float(c[1])), h - 1))
                cv2.circle(canvas, (ox, oy), 8, (0, 140, 255), -1)
                cv2.circle(canvas, (ox, oy), 8, (0, 0, 0), 1)
                tx, ty, (x1, y1, x2, y2) = _place_label(ox, oy, r_sem, scale=0.44)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 30), -1)
                cv2.putText(canvas, r_sem, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 100), 1, cv2.LINE_AA)

        # De-clutter: dedupe relations, apply view filter, and limit fan-out per subject.
        region_ids: set = {str(r.get("id", "")) for r in (regions_meta or [])} - {""}
        obj_ids: set    = {str(o.get("id", "")) for o in objects_3d} - {""}

        deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for rel in relations:
            target_id = rel.get("object_id")
            if target_id is None:
                continue
            if isinstance(target_id, str) and target_id.startswith("external_"):
                continue
            sid = str(rel.get("subject_id"))
            tid = str(target_id)
            # View filter
            if view == "objects_only":
                if sid in region_ids or tid in region_ids:
                    continue
            elif view == "regions_only":
                if str(rel.get("relation_tier", "")) != "region_region":
                    continue
            key = (sid, str(rel.get("predicate", "related_to")), tid)
            if key not in deduped:
                deduped[key] = rel

        by_subject: Dict[str, List[Dict[str, Any]]] = {}
        for rel in deduped.values():
            sid = str(rel.get("subject_id"))
            by_subject.setdefault(sid, []).append(rel)

        max_obj_rel    = int(getattr(self.config, "map_max_rel_per_object", 6)) if self.config else 6
        max_region_rel = int(getattr(self.config, "map_max_rel_per_region", 4)) if self.config else 4
        filtered: List[Dict[str, Any]] = []
        for sid, rels in by_subject.items():
            cap = max_region_rel if sid in region_ids else max_obj_rel
            rels_sorted = sorted(rels, key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)
            filtered.extend(rels_sorted[:cap])

        # Fixed perpendicular lane per predicate — each predicate type always draws in
        # its own spatial lane so lines from different predicates never ablate each other.
        # Values are signed pixel offsets perpendicular to the subject→object vector.
        _pred_lane: Dict[str, int] = {
            "in_front_of": +14, "behind": -14,
            "left_of":     + 7, "right_of": -7,
            "above":       +21, "below":   -21,
            "adjacent_to":   0, "overlapping": 0,
            "contains":    +28, "inside_of":  -28,
            "on":            0, "holds":       +7,
        }

        for rel in filtered:
            target_id = rel.get("object_id")
            sid = str(rel.get("subject_id"))
            tid = str(target_id)
            subject = id_to_obj.get(sid)
            target = id_to_obj.get(tid)
            if subject is None and target is None:
                continue
            if subject is not None and target is not None:
                sx, sy = self._object_centroid_xy(subject)
                tx_coord, ty_coord = self._object_centroid_xy(target)
            else:
                sx, sy = self._endpoint_xy_for_relation_map(sid, objects_3d, regions_meta or [])
                tx_coord, ty_coord = self._endpoint_xy_for_relation_map(tid, objects_3d, regions_meta or [])
            sx, sy, tx_coord, ty_coord = int(sx), int(sy), int(tx_coord), int(ty_coord)
            dx, dy = tx_coord - sx, ty_coord - sy
            length_sq = dx * dx + dy * dy
            if length_sq < 20 * 20:
                # Skip very short edges — they visually cluster at the same anchor point.
                continue

            pred = str(rel.get("predicate", "related_to"))
            col = _pred_col.get(pred, (180, 180, 180))

            # Perpendicular offset: each predicate occupies its own fixed lane so
            # parallel relations between the same pair fan out cleanly and stay readable.
            lane = _pred_lane.get(pred, 0)
            if lane != 0:
                length = float(length_sq ** 0.5) + 1e-6
                perp_x = int((-dy / length) * lane)
                perp_y = int(( dx / length) * lane)
            else:
                perp_x, perp_y = 0, 0

            p1 = (sx + perp_x, sy + perp_y)
            p2 = (tx_coord + perp_x, ty_coord + perp_y)

            # Arrow: thicker (2px) so it reads at small image sizes.
            cv2.arrowedLine(canvas, p1, p2, col, 2, cv2.LINE_AA, tipLength=0.10)

            # Predicate label midpoint — semi-transparent dark background so the label
            # floats cleanly over the arrow without a harsh solid black block.
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            ptx, pty, (bx1, by1, bx2, by2) = _place_label(mx, my, pred, scale=0.42)
            roi = canvas[max(0, by1):min(h, by2 + 1), max(0, bx1):min(w, bx2 + 1)]
            if roi.size > 0:
                canvas[max(0, by1):min(h, by2 + 1), max(0, bx1):min(w, bx2 + 1)] = (
                    np.clip(roi.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
                )
            cv2.putText(canvas, pred, (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        self._write_png_intermediate(path, canvas)

    def _save_layers_map(
        self,
        image_bgr: np.ndarray,
        objects_3d: List[Dict[str, Any]],
        path: Path,
        regions_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        from scene_understanding.visualization.layers_map import save_layers_map_bgr

        save_layers_map_bgr(image_bgr, objects_3d, path, regions_meta=regions_meta)

    @staticmethod
    def _copy_or_link_file(src: Path, dst: Path, use_hardlink: bool) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if use_hardlink:
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
                return
            except OSError as exc:
                # Phase B.10 — warn-once when the filesystem rejects os.link so
                # operators see why disk usage inflates on per-track dirs, then
                # transparently fall back to a regular copy for correctness.
                if not getattr(SceneUnderstandingPipeline, "_hardlink_warned", False):
                    print(
                        f"[hardlink] os.link({src} → {dst}) failed ({exc.__class__.__name__}: {exc}). "
                        "Filesystem does not support hardlinks; falling back to shutil.copyfile. "
                        "This warning is emitted once per process."
                    )
                    try:
                        setattr(SceneUnderstandingPipeline, "_hardlink_warned", True)
                    except Exception:
                        pass
        import shutil
        shutil.copyfile(src, dst)

    def _build_region_region_relations_from_meta(
        self,
        regions_meta: List[Dict[str, Any]],
        label_map: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Pairwise spatial relations between depth-partition regions.

        Predicates emitted per pair:
          in_front_of / behind  — depth delta >= region_relation_depth_delta_m
          left_of / right_of    — centroid x separation >= region_relation_min_centroid_sep_px
          above / below         — centroid y separation >= region_relation_min_centroid_sep_px
          overlapping           — mask IoU (or bbox IoU) >= region_relation_iou_overlap
          adjacent_to           — dilated bboxes touch (dilation = region_relation_dilate_px)
        """
        if len(regions_meta) < 2:
            return []
        cfg = self.config
        depth_delta  = float(getattr(cfg, "region_relation_depth_delta_m",       0.75)) if cfg else 0.75
        iou_thresh   = float(getattr(cfg, "region_relation_iou_overlap",          0.05)) if cfg else 0.05
        dilate_px    = int(  getattr(cfg, "region_relation_dilate_px",               2)) if cfg else 2
        sep_px       = int(  getattr(cfg, "region_relation_min_centroid_sep_px",    12)) if cfg else 12

        lm = np.asarray(label_map, dtype=np.int32) if label_map is not None else None
        edges: List[Dict[str, Any]] = []

        def _sem_name(r: Dict[str, Any]) -> str:
            return (
                str(r.get("semantic_label", "") or r.get("canonical_name", "") or r.get("type", "region"))
                .strip().lower() or "region"
            )

        def _make_edge(
            a: Dict[str, Any],
            b: Dict[str, Any],
            predicate: str,
            score: float = 0.6,
            extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            aid = str(a.get("id", ""))
            bid = str(b.get("id", ""))
            if not aid or not bid:
                return
            ca = a.get("centroid_2d_px") or [0, 0]
            cb = b.get("centroid_2d_px") or [0, 0]
            edge: Dict[str, Any] = {
                "subject_id": aid,
                "subject_label": f"region_{a.get('type', 'mixed')}",
                "subject_name": _sem_name(a),
                "predicate": predicate,
                "object_id": bid,
                "object_label": f"region_{b.get('type', 'mixed')}",
                "object_name": _sem_name(b),
                "object_caption": "",
                "source": "region_spatial",
                "score": round(score, 4),
                "relation_tier": "region_region",
                "subject_centroid": [int(round(float(ca[0]))), int(round(float(ca[1])))],
                "object_centroid": [int(round(float(cb[0]))), int(round(float(cb[1])))],
            }
            if extra:
                edge.update(extra)
            edges.append(edge)

        def _bbox_iou(a_bx: List[int], b_bx: List[int]) -> float:
            ax1, ay1, ax2, ay2 = a_bx
            bx1, by1, bx2, by2 = b_bx
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                return 0.0
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / (area_a + area_b - inter)

        def _mask_iou(rid_a: int, rid_b: int) -> Optional[float]:
            if lm is None:
                return None
            ma = lm == rid_a
            mb = lm == rid_b
            inter = int(np.logical_and(ma, mb).sum())
            if inter == 0:
                return 0.0
            return inter / max(int(np.logical_or(ma, mb).sum()), 1)

        def _bboxes_adjacent(a_bx: List[int], b_bx: List[int], d: int) -> bool:
            ax1, ay1, ax2, ay2 = a_bx[0] - d, a_bx[1] - d, a_bx[2] + d, a_bx[3] + d
            bx1, by1, bx2, by2 = b_bx[0] - d, b_bx[1] - d, b_bx[2] + d, b_bx[3] + d
            return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

        # Phase 7: mask-based adjacency check — more precise than bbox dilation,
        # which fires on rectangular corner overlaps between non-contiguous regions.
        # Returns (is_adjacent, shared_border_px) using the actual label_map.
        _adj_kernel: Optional[np.ndarray] = None
        if lm is not None and dilate_px > 0:
            _k = dilate_px * 2 + 1
            _adj_kernel = np.ones((_k, _k), np.uint8)

        def _mask_adjacent(rid_a: int, rid_b: int) -> Tuple[bool, int]:
            if lm is None or _adj_kernel is None or rid_a <= 0 or rid_b <= 0:
                return False, 0
            _ma = (lm == rid_a).astype(np.uint8)
            _dilated = cv2.dilate(_ma, _adj_kernel).astype(bool)
            _mb = (lm == rid_b) > 0
            border = int(np.logical_and(_dilated, _mb).sum())
            _min_border = int(getattr(cfg, "region_adjacency_min_border_px", 10)) if cfg else 10
            return border >= _min_border, border

        for i in range(len(regions_meta)):
            a = regions_meta[i]
            a_rid   = int(a.get("region_index", 0) or 0)
            a_depth = float((a.get("depth_stats") or {}).get("mean", 0.0))
            a_ca    = a.get("centroid_2d_px") or [0, 0]
            a_cx, a_cy = float(a_ca[0]), float(a_ca[1])
            a_bx    = [int(v) for v in (a.get("bbox_px") or [0, 0, 1, 1])[:4]]

            for j in range(i + 1, len(regions_meta)):
                b = regions_meta[j]
                b_rid   = int(b.get("region_index", 0) or 0)
                b_depth = float((b.get("depth_stats") or {}).get("mean", 0.0))
                b_ca    = b.get("centroid_2d_px") or [0, 0]
                b_cx, b_cy = float(b_ca[0]), float(b_ca[1])
                b_bx    = [int(v) for v in (b.get("bbox_px") or [0, 0, 1, 1])[:4]]

                # Depth-based predicates
                d_diff = a_depth - b_depth
                if abs(d_diff) >= depth_delta:
                    if d_diff < 0:  # a closer (smaller depth = nearer camera)
                        _make_edge(a, b, "in_front_of", score=0.70)
                        _make_edge(b, a, "behind",      score=0.70)
                    else:
                        _make_edge(b, a, "in_front_of", score=0.70)
                        _make_edge(a, b, "behind",      score=0.70)

                # Horizontal predicates
                dx = a_cx - b_cx
                if abs(dx) >= sep_px:
                    if dx < 0:
                        _make_edge(a, b, "left_of",  score=0.65)
                        _make_edge(b, a, "right_of", score=0.65)
                    else:
                        _make_edge(a, b, "right_of", score=0.65)
                        _make_edge(b, a, "left_of",  score=0.65)

                # Vertical predicates (image y increases downward)
                dy = a_cy - b_cy
                if abs(dy) >= sep_px:
                    if dy < 0:
                        _make_edge(a, b, "above", score=0.65)
                        _make_edge(b, a, "below", score=0.65)
                    else:
                        _make_edge(a, b, "below", score=0.65)
                        _make_edge(b, a, "above", score=0.65)

                # Overlap predicate (mask-based when label_map available, bbox fallback)
                miou = _mask_iou(a_rid, b_rid) if (a_rid > 0 and b_rid > 0) else None
                _iou_val = miou if miou is not None else _bbox_iou(a_bx, b_bx)
                if _iou_val >= iou_thresh:
                    _make_edge(a, b, "overlapping", score=_iou_val)
                    _make_edge(b, a, "overlapping", score=_iou_val)

                # Adjacency predicate — Phase 7: use mask-based dilation when label_map
                # is available; fall back to bbox dilation for backwards compatibility.
                if a_rid > 0 and b_rid > 0 and lm is not None:
                    _is_adj, _border_px = _mask_adjacent(a_rid, b_rid)
                    if _is_adj:
                        _adj_extra = {"shared_border_px": _border_px}
                        _make_edge(a, b, "adjacent_to", score=0.60, extra=_adj_extra)
                        _make_edge(b, a, "adjacent_to", score=0.60, extra=_adj_extra)
                elif _bboxes_adjacent(a_bx, b_bx, dilate_px):
                    _make_edge(a, b, "adjacent_to", score=0.60)
                    _make_edge(b, a, "adjacent_to", score=0.60)

        return edges

    def _build_region_hierarchy_supplements(
        self,
        region_partition_meta: List[Dict[str, Any]],
        label_map: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Shallow nodes for mask containment hierarchy (same list shape as objects where needed)."""
        out: List[Dict[str, Any]] = []
        lm = np.asarray(label_map, dtype=np.int32)
        h, w = lm.shape[:2]
        for r in region_partition_meta:
            rid = int(r.get("region_index", 0) or 0)
            if rid <= 0:
                continue
            mask = lm == rid
            if not np.any(mask):
                continue
            bx = r.get("bbox_px") or [0, 0, w - 1, h - 1]
            x1, y1, x2, y2 = [int(v) for v in bx[:4]]
            z_mean = float((r.get("depth_stats") or {}).get("mean", 0.0))
            _sem = str(r.get("semantic_label", "")).strip().lower()
            _canon = str(r.get("canonical_name", "")).strip().lower()
            _rtype = str(r.get("type", "region"))
            _rlabel = _sem or _canon or _rtype
            # Phase 2: use real back-projected coordinates stored by _enrich_region_labels_from_masks
            # (Phase 1). Falls back to z-only if depth enrichment was not run.
            _c3d = r.get("coordinates_3d") or {"x": 0.0, "y": 0.0, "z": z_mean}
            out.append({
                "id": str(r.get("id", f"region_{rid}")),
                "label": _rlabel,
                "canonical_name": _canon or _sem or _rtype,
                "entity_kind": "region",
                "_sam2_mask_array": mask,
                "sam2_mask_index": None,
                "bbox": [x1, y1, x2, y2],
                "mask_centroid_2d": [float(v) for v in (r.get("centroid_2d_px") or [(x1 + x2) / 2, (y1 + y2) / 2])],
                "coordinates_3d": {"x": float(_c3d.get("x", 0.0)), "y": float(_c3d.get("y", 0.0)), "z": float(_c3d.get("z", z_mean))},
            })
        return out

    def _build_region_visual_objects(
        self,
        region_partition_meta: List[Dict[str, Any]],
        label_map: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Pseudo-objects for _save_labelled_segmentation / _save_labelled_tinted_overlay on regions."""
        rows: List[Dict[str, Any]] = []
        lm = np.asarray(label_map, dtype=np.int32)
        h, w = lm.shape[:2]
        for r in region_partition_meta:
            rid = int(r.get("region_index", 0) or 0)
            if rid <= 0:
                continue
            mask = lm == rid
            if not np.any(mask):
                continue
            bx = r.get("bbox_px") or [0, 0, w - 1, h - 1]
            x1, y1, x2, y2 = [int(v) for v in bx[:4]]
            oids = r.get("object_ids") or []
            typ = str(r.get("type", "region"))
            sem = str(r.get("semantic_label", "")).strip().lower()
            base = sem if sem else typ
            tag = f"{base} ({len(oids)} objs)" if oids else base
            cxy = r.get("centroid_2d_px") or [(x1 + x2) / 2, (y1 + y2) / 2]
            cx, cy = float(cxy[0]), float(cxy[1])
            canon_raw = str(r.get("canonical_name", "")).strip().lower()
            canon = canon_raw or sem or typ
            rows.append({
                "id": str(r.get("id", f"region_{rid}")),
                "label": tag,
                "canonical_name": canon,
                "_sam2_mask_array": mask,
                "mask_centroid_2d": [cx, cy],
                "bbox": [x1, y1, x2, y2],
            })
        return rows

    def _endpoint_xy_for_relation_map(
        self,
        entity_id: str,
        objects_3d: List[Dict[str, Any]],
        regions_meta: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        id_to_obj = {str(o.get("id")): o for o in objects_3d}
        if entity_id in id_to_obj:
            return self._object_centroid_xy(id_to_obj[entity_id])
        for r in regions_meta:
            if str(r.get("id")) == entity_id:
                c = r.get("centroid_2d_px") or [0, 0]
                return float(c[0]), float(c[1])
        return 0.0, 0.0

    def _save_regions_overlay(
        self,
        image_bgr: np.ndarray,
        label_map: np.ndarray,
        palette: List[List[int]],
        path: Path,
        alpha: float = 0.45,
        regions_meta: Optional[List[Dict[str, Any]]] = None,
        region_relations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        h, w = image_bgr.shape[:2]
        if label_map.shape[:2] != (h, w):
            lm = cv2.resize(
                label_map.astype(np.int32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            lm = np.asarray(label_map, dtype=np.int32)
        overlay = np.zeros_like(image_bgr)
        flat = lm.ravel()
        for idx in range(1, len(palette)):
            rgb = palette[idx]
            sel = flat == idx
            if not sel.any():
                continue
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            tmp = overlay.reshape(-1, 3)
            tmp[sel] = bgr
        blend = np.clip(
            image_bgr.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
        if regions_meta:
            blend = self._annotate_region_geometry(
                blend,
                regions_meta=regions_meta,
                region_relations=region_relations or [],
            )
        self._write_png_intermediate(path, blend)

    def _annotate_region_geometry(
        self,
        canvas: np.ndarray,
        regions_meta: List[Dict[str, Any]],
        region_relations: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw region IDs, semantic labels, and geometric relation arrows on a canvas."""
        out = canvas.copy()
        h, w = out.shape[:2]
        if not regions_meta:
            return out

        occ: List[Tuple[int, int, int, int]] = []

        def _ov(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        def _place(x: int, y: int, text: str, scale: float = 0.36) -> Tuple[int, int, Tuple[int, int, int, int]]:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cands = [(x + 6, y - 6), (x + 6, y + th + 6), (x - tw - 6, y - 6), (x - tw - 6, y + th + 6)]
            for tx, ty in cands:
                box = (max(0, tx - 2), max(0, ty - th - 2), min(w - 1, tx + tw + 2), min(h - 1, ty + 2))
                if any(_ov(box, o) for o in occ):
                    continue
                occ.append(box)
                return tx, ty, box
            tx, ty = cands[0]
            box = (max(0, tx - 2), max(0, ty - th - 2), min(w - 1, tx + tw + 2), min(h - 1, ty + 2))
            occ.append(box)
            return tx, ty, box

        id_to_center: Dict[str, Tuple[int, int]] = {}
        for r in regions_meta:
            rid = str(r.get("id", "")).strip()
            if not rid:
                continue
            c = r.get("centroid_2d_px") or [w // 2, h // 2]
            cx = int(min(max(0, float(c[0])), w - 1))
            cy = int(min(max(0, float(c[1])), h - 1))
            id_to_center[rid] = (cx, cy)
            # Centroid marker: filled cyan circle with black ring for contrast
            cv2.circle(out, (cx, cy), 7, (0, 0, 0), -1)
            cv2.circle(out, (cx, cy), 6, (0, 220, 255), -1)
            sem = str(r.get("semantic_label", "") or r.get("canonical_name", "") or r.get("type", "region")).strip().lower()
            text = sem if sem else "region"
            tx, ty, (x1, y1, x2, y2) = _place(cx, cy, text)
            # Semi-transparent label background (30% brightness darkening)
            roi = out[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)]
            if roi.size:
                out[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = (roi * 0.30).astype(np.uint8)
            cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        pred_col = {
            "in_front_of": (60, 240, 60),
            "behind": (60, 180, 240),
            "left_of": (255, 180, 0),
            "right_of": (255, 180, 0),
            "above": (255, 140, 200),
            "below": (255, 140, 200),
            "adjacent_to": (210, 210, 210),
            "overlapping": (0, 200, 255),
            "depth_parent_of": (120, 255, 255),
        }
        max_rel = int(getattr(self.config, "map_max_rel_per_region", 4)) if self.config else 4
        by_subject: Dict[str, List[Dict[str, Any]]] = {}
        for rel in region_relations or []:
            sid = str(rel.get("subject_id", ""))
            tid = str(rel.get("object_id", ""))
            if sid in id_to_center and tid in id_to_center:
                by_subject.setdefault(sid, []).append(rel)

        for sid, rels in by_subject.items():
            rels = sorted(rels, key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)[:max_rel]
            sx, sy = id_to_center[sid]
            for rel in rels:
                tid = str(rel.get("object_id", ""))
                if tid not in id_to_center:
                    continue
                tx, ty = id_to_center[tid]
                pred = str(rel.get("predicate", "related_to"))
                col = pred_col.get(pred, (0, 255, 255))
                cv2.arrowedLine(out, (sx, sy), (tx, ty), col, 2, cv2.LINE_AA, tipLength=0.12)
                mx, my = (sx + tx) // 2, (sy + ty) // 2
                ltx, lty, (x1, y1, x2, y2) = _place(mx, my, pred, scale=0.34)
                # Semi-transparent label background
                roi = out[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)]
                if roi.size:
                    out[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = (roi * 0.30).astype(np.uint8)
                cv2.putText(out, pred, (ltx, lty), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

        _legend_entries = [
            ("region geometry", (255, 255, 255)),
            ("in_front_of", (60, 240, 60)),
            ("behind", (60, 180, 240)),
            ("left/right_of", (255, 180, 0)),
            ("above/below", (255, 140, 200)),
            ("adjacent_to", (210, 210, 210)),
            ("overlapping", (0, 200, 255)),
        ]
        y = 16
        for label, col in _legend_entries:
            cv2.putText(out, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)
            y += 16
        return out

    @staticmethod
    def _build_region_layers_payload(
        region_metas: List[Dict[str, Any]],
        q1: Optional[float] = None,
        q2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build the region-specific layers payload.

        Phase 5: accepts shared q1/q2 from _build_layers_payload so that objects
        and regions are always classified against the same depth thresholds. When
        q1/q2 are None (standalone call), falls back to computing them from region
        depths only — avoids mixing region means into the object quantile pool.
        """
        if not region_metas:
            return {
                "ordering": [],
                "bands": [],
                "depth_quantiles": {},
            }
        zvals: List[float] = []
        for r in region_metas:
            z = (r.get("depth_stats") or {}).get("mean")
            if z is not None:
                zvals.append(float(z))
        if not zvals:
            for r in region_metas:
                r["layer_type"] = "unassigned"
            return {
                "ordering": [],
                "bands": [],
                "depth_quantiles": {},
            }
        if q1 is None or q2 is None:
            # Standalone fallback: derive from region depths only (never mixed with objects).
            _qs = np.quantile(np.asarray(zvals, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0])
            q1, q2 = float(_qs[0]), float(_qs[1])
        bands: Dict[str, List[Dict[str, Any]]] = {"foreground": [], "midground": [], "background": []}
        for r in region_metas:
            z = float((r.get("depth_stats") or {}).get("mean", 0.0))
            if z <= q1:
                lt = "foreground"
            elif z <= q2:
                lt = "midground"
            else:
                lt = "background"
            r["layer_type"] = lt
            bands[lt].append(r)
        ordering = []
        for lt in ("foreground", "midground", "background"):
            for r in sorted(bands[lt], key=lambda x: float((x.get("depth_stats") or {}).get("mean", 0.0))):
                ordering.append({
                    "id": str(r.get("id", "")),
                    "label": str(r.get("semantic_label", "") or r.get("canonical_name", "") or r.get("type", "region")),
                    "layer_type": lt,
                    "z_mean": float((r.get("depth_stats") or {}).get("mean", 0.0)),
                })
        return {
            "ordering": ordering,
            "bands": [{"layer_type": k, "region_ids": [str(r.get("id", "")) for r in v]} for k, v in bands.items()],
            "depth_quantiles": {"q1": q1, "q2": q2},
        }

    @staticmethod
    def _build_region_adjacency_graph(
        region_metas: List[Dict[str, Any]],
        label_map: np.ndarray,
        min_border_px: int = 10,
        dilation_px: int = 3,
    ) -> Dict[str, Any]:
        """
        Phase 6: Build a true shared-border adjacency graph for depth-partition regions.

        Replaces the depth-ordered linear chain (_build_region_hierarchy_payload) with a
        structure that reflects physical reality: two regions are connected only when their
        pixel masks share a real boundary after mask dilation. Each edge carries
        shared_border_px as a traversal-cost weight — wider borders mean easier crossings.

        Trajectory refiners can use this graph for correct path planning between depth
        zones (e.g. foreground sidewalk → road surface → midground cars) without the false
        spatial containment implied by the old parent/child chain.
        """
        lm = np.asarray(label_map, dtype=np.int32)
        k_size = max(1, dilation_px) * 2 + 1
        kernel = np.ones((k_size, k_size), np.uint8)
        edges: List[Dict[str, Any]] = []
        rids = [
            (int(r.get("region_index", 0)), r)
            for r in region_metas
            if int(r.get("region_index", 0) or 0) > 0
        ]
        for i, (rid_a, ra) in enumerate(rids):
            mask_a = (lm == rid_a).astype(np.uint8)
            dilated_a = cv2.dilate(mask_a, kernel).astype(bool)
            z_a = float((ra.get("depth_stats") or {}).get("mean", 0.0))
            for _rid_b, rb in rids[i + 1:]:
                mask_b = (lm == _rid_b) > 0
                border = int(np.logical_and(dilated_a, mask_b).sum())
                if border < min_border_px:
                    continue
                z_b = float((rb.get("depth_stats") or {}).get("mean", 0.0))
                edges.append({
                    "region_a": str(ra.get("id", "")),
                    "region_b": str(rb.get("id", "")),
                    "shared_border_px": border,
                    "depth_delta_m": round(abs(z_a - z_b), 4),
                    "depth_relation": "a_in_front" if z_a < z_b else "b_in_front",
                    "edge_type": "region_spatial_adjacency",
                })
        return {
            "edges": edges,
            "num_edges": len(edges),
            "kind": "region_adjacency_graph",
        }

    @staticmethod
    def _build_region_hierarchy_payload(region_metas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Depth-ordered hierarchy for regions (farther region is parent of nearer sibling chain)."""
        if len(region_metas) < 2:
            roots = [str(r.get("id", "")) for r in region_metas if str(r.get("id", ""))]
            return {"edges": [], "root_region_ids": roots, "num_edges": 0, "kind": "region_depth_hierarchy"}
        ranked = sorted(region_metas, key=lambda r: float((r.get("depth_stats") or {}).get("mean", 0.0)))
        edges: List[Dict[str, Any]] = []
        for near, far in zip(ranked[:-1], ranked[1:]):
            cid = str(near.get("id", ""))
            pid = str(far.get("id", ""))
            if not cid or not pid:
                continue
            zc = float((near.get("depth_stats") or {}).get("mean", 0.0))
            zp = float((far.get("depth_stats") or {}).get("mean", 0.0))
            edges.append({
                "parent_object_id": pid,
                "child_object_id": cid,
                "edge_type": "region_depth_order",
                "containment_ratio": round(max(0.0, min(1.0, (zp - zc) / max(zp + 1e-6, 1e-6))), 4),
            })
        roots = [str(ranked[-1].get("id", ""))] if ranked else []
        return {
            "edges": edges,
            "root_region_ids": [r for r in roots if r],
            "num_edges": len(edges),
            "kind": "region_depth_hierarchy",
        }

    @staticmethod
    def _depth_label_plausibility_score(label: str, z_m: Optional[float]) -> float:
        if z_m is None or not np.isfinite(float(z_m)):
            return 1.0
        key = str(label or "").strip().lower()
        if not key or key == "object":
            return 1.0
        prior = _LABEL_DEPTH_PRIORS_M.get(key)
        if prior is None:
            for tok in key.replace("-", " ").split():
                if tok in _LABEL_DEPTH_PRIORS_M:
                    prior = _LABEL_DEPTH_PRIORS_M[tok]
                    break
        if prior is None:
            return 1.0
        lo, hi = prior
        z = float(z_m)
        if lo <= z <= hi:
            return 1.0
        dist = (lo - z) if z < lo else (z - hi)
        width = max(hi - lo, 1.0)
        penalty = min(1.0, dist / (width * 2.0))
        return round(max(0.0, 1.0 - penalty), 4)

    def _apply_regions_plausibility_to_objects(self, objects_3d: List[Dict[str, Any]]) -> None:
        for obj in objects_3d:
            z = obj.get("coordinates_3d", {}).get("z")
            lab = str(obj.get("label", "object"))
            score = self._depth_label_plausibility_score(lab, float(z) if z is not None else None)
            obj["depth_plausibility_score"] = score
            if score < 0.5:
                obj["label_warning"] = "depth_prior_mismatch"
            if self._regions_reject_implausible and score < 0.35:
                obj["label"] = "object"
                obj["canonical_name"] = "object"

    def _save_mask_hierarchy_map(
        self,
        image_bgr: np.ndarray,
        objects_3d: List[Dict[str, Any]],
        hierarchy: Dict[str, Any],
        path: Path,
        region_supplements: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        canvas = image_bgr.copy()
        id_to_obj = {str(obj.get("id")): obj for obj in objects_3d}
        if region_supplements:
            for rsupp in region_supplements:
                id_to_obj[str(rsupp.get("id"))] = rsupp

        # Human-readable edge type labels and distinct colours.
        _etype_label = {
            "object_object_part":        "part of",
            "region_object_membership":  "member of",
            "region_region_containment": "inside region",
            "region_depth_order":        "depth order",
        }
        edge_colours = {
            "object_object_part":        (255,   0, 255),  # magenta
            "region_object_membership":  (  0, 165, 255),  # orange
            "region_region_containment": (255, 200,   0),  # gold
            "region_depth_order":        (120, 255, 255),  # cyan-yellow
        }

        # Draw all node markers first so edges sit on top.
        for node_id, node in id_to_obj.items():
            cx, cy = self._object_centroid_xy(node)
            cx, cy = int(cx), int(cy)
            is_region = str(node.get("entity_kind", "object")) == "region"
            # Regions: orange diamond marker; objects: white circle.
            if is_region:
                cv2.drawMarker(canvas, (cx, cy), (0, 165, 255), cv2.MARKER_DIAMOND, 12, 2)
            else:
                cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)
                cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1)
            label = str(node.get("canonical_name", node.get("label", "")))
            if not label or label == "object":
                label = str(node.get("label", node_id))
            # Dark semi-transparent pill behind the label.
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            lx, ly = cx + 8, cy - 4
            roi = canvas[max(0, ly - th - 2):min(canvas.shape[0], ly + 3),
                         max(0, lx - 2):min(canvas.shape[1], lx + tw + 2)]
            if roi.size > 0:
                canvas[max(0, ly - th - 2):min(canvas.shape[0], ly + 3),
                       max(0, lx - 2):min(canvas.shape[1], lx + tw + 2)] = (
                    np.clip(roi.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)
                )
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Arrowed edges drawn parent → child so direction is unambiguous.
        for edge in hierarchy.get("edges", []):
            parent = id_to_obj.get(str(edge.get("parent_object_id")))
            child  = id_to_obj.get(str(edge.get("child_object_id")))
            if parent is None or child is None:
                continue
            px, py = self._object_centroid_xy(parent)
            cx, cy = self._object_centroid_xy(child)
            px, py, cx, cy = int(px), int(py), int(cx), int(cy)
            etype = str(edge.get("edge_type", "object_object_part"))
            colour = edge_colours.get(etype, (255, 0, 255))
            cv2.arrowedLine(canvas, (px, py), (cx, cy), colour, 2, cv2.LINE_AA, tipLength=0.10)
            mid_x = (px + cx) // 2
            mid_y = (py + cy) // 2
            # Show human-readable verb; append containment ratio if present.
            ratio = edge.get("containment_ratio", "")
            readable = _etype_label.get(etype, etype)
            edge_text = f"{readable} ({ratio:.2f})" if isinstance(ratio, float) else readable
            (tw, th), _ = cv2.getTextSize(edge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            roi = canvas[max(0, mid_y - th - 2):min(canvas.shape[0], mid_y + 3),
                         max(0, mid_x - 2):min(canvas.shape[1], mid_x + tw + 2)]
            if roi.size > 0:
                canvas[max(0, mid_y - th - 2):min(canvas.shape[0], mid_y + 3),
                       max(0, mid_x - 2):min(canvas.shape[1], mid_x + tw + 2)] = (
                    np.clip(roi.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)
                )
            cv2.putText(canvas, edge_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, colour, 1, cv2.LINE_AA)

        legend = [
            ("part of",        edge_colours["object_object_part"]),
            ("member of",      edge_colours["region_object_membership"]),
            ("inside region",  edge_colours["region_region_containment"]),
            ("depth order",    edge_colours["region_depth_order"]),
        ]
        y = 18
        for txt, col in legend:
            cv2.putText(canvas, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            y += 16

        self._write_png_intermediate(path, canvas)

    @staticmethod
    def _write_text(text: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def _md_heading(title: str) -> str:
        return f"# {title}\n\n"

    def _collect_florence_object_caption_rows(
        self,
        objects_3d: List[Dict[str, Any]],
        max_objects: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for obj in objects_3d[: max(1, int(max_objects))]:
            src = obj.get("sources", {})
            f2 = src.get("Florence2", {}) if isinstance(src, dict) else {}
            rows.append({
                "object_id": str(obj.get("id")),
                "label": str(obj.get("label", "object")),
                "segmentor": str(obj.get("segmentor", "unknown")),
                "florence_label": str(f2.get("label", "")),
                "florence_caption": str(f2.get("caption", "")),
                "selected_caption": str(obj.get("caption", "")),
            })
        return rows

    def _build_fusion_prompt_text(
        self,
        image_path: str,
        track_key: str,
        scene_json_rel: str,
        relations_json_rel: str,
        layers_json_rel: str,
        hierarchy_json_rel: str,
        depth_mask_a_json_rel: str,
        depth_mask_b_json_rel: str,
        segmentation_rel: str,
        tinted_overlay_rel: str,
        relations_map_rel: str,
        layers_png_rel: str,
        hierarchy_png_rel: str,
        regions_json_rel: str = "",
        regions_png_rel: str = "",
        regions_overlay_rel: str = "",
        region_segmentation_rel: str = "",
        region_sam2_seg_rel: str = "",
        region_tinted_rel: str = "",
    ) -> str:
        region_lines = ""
        if regions_json_rel:
            region_lines = (
                f"- regions JSON: {regions_json_rel}\n"
                f"- regions index image: {regions_png_rel}\n"
                f"- regions overlay image: {regions_overlay_rel}\n"
            )
        if region_segmentation_rel:
            region_lines += (
                f"- region segmentation image (parallel): {region_segmentation_rel}\n"
                f"- region SAM2-style labelled segmentation (parallel): {region_sam2_seg_rel}\n"
                f"- region tinted overlay (parallel): {region_tinted_rel}\n"
            )
        return f"""You are a scene-grounded captioner and verifier.

Use ALL provided artifacts jointly:
- Original image: {image_path}
- Track: {track_key}
- scene_graph JSON: {scene_json_rel}
- relations JSON: {relations_json_rel}
- layers JSON: {layers_json_rel}
- mask hierarchy JSON: {hierarchy_json_rel}
- depth_mask_A JSON: {depth_mask_a_json_rel}
- depth_mask_B JSON: {depth_mask_b_json_rel}
- segmentation image: {segmentation_rel}
- tinted overlay: {tinted_overlay_rel}
- relations map image: {relations_map_rel}
- layers image: {layers_png_rel}
- mask hierarchy image: {hierarchy_png_rel}
{region_lines}
Goals:
1) Write a detailed caption (10-16 sentences) grounded in BOTH image evidence and scene-graph evidence.
2) Include object attributes, spatial relations, depth/layer cues, and part-whole hierarchy when supported.
3) Avoid hallucinations: do not mention entities absent from both image and graph.
4) If evidence conflicts, explicitly flag uncertainty and alternatives.
"""

    def _save_caption_variants_for_track(
        self,
        out_dir: Path,
        image_path: str,
        path_stem: str,
        track_key: str,
        track_dir_name: str,
        objects_3d: List[Dict[str, Any]],
        scene_json_rel: str,
        relations_json_rel: str,
        layers_json_rel: str,
        hierarchy_json_rel: str,
        depth_mask_a_json_rel: str,
        depth_mask_b_json_rel: str,
        segmentation_rel: str,
        tinted_overlay_rel: str,
        relations_map_rel: str,
        layers_png_rel: str,
        hierarchy_png_rel: str,
        regions_json_rel: str = "",
        regions_png_rel: str = "",
        regions_overlay_rel: str = "",
        region_segmentation_rel: str = "",
        region_sam2_seg_rel: str = "",
        region_tinted_rel: str = "",
    ) -> None:
        track_dir = out_dir / track_dir_name
        fusion_prompt = self._build_fusion_prompt_text(
            image_path=image_path,
            track_key=track_key,
            scene_json_rel=scene_json_rel,
            relations_json_rel=relations_json_rel,
            layers_json_rel=layers_json_rel,
            hierarchy_json_rel=hierarchy_json_rel,
            depth_mask_a_json_rel=depth_mask_a_json_rel,
            depth_mask_b_json_rel=depth_mask_b_json_rel,
            segmentation_rel=segmentation_rel,
            tinted_overlay_rel=tinted_overlay_rel,
            relations_map_rel=relations_map_rel,
            layers_png_rel=layers_png_rel,
            hierarchy_png_rel=hierarchy_png_rel,
            regions_json_rel=regions_json_rel,
            regions_png_rel=regions_png_rel,
            regions_overlay_rel=regions_overlay_rel,
            region_segmentation_rel=region_segmentation_rel,
            region_sam2_seg_rel=region_sam2_seg_rel,
            region_tinted_rel=region_tinted_rel,
        )

        florence_rows = self._collect_florence_object_caption_rows(objects_3d, self.caption_max_objects_per_track)
        florence_obj_json = {
            "variant": "florence_object",
            "track": track_key,
            "image_path": image_path,
            "count": len(florence_rows),
            "objects": florence_rows,
        }
        self._write_json(florence_obj_json, track_dir / f"{path_stem}_florence_object_captions.json")
        md_lines = [self._md_heading(f"Florence Object Captions - {track_key}")]
        for row in florence_rows:
            md_lines.append(
                f"- `{row['object_id']}` | label=`{row['label']}` | florence_label=`{row['florence_label']}` | florence_caption=`{row['florence_caption']}` | selected_caption=`{row['selected_caption']}`"
            )
        self._write_text("\n".join(md_lines) + "\n", track_dir / f"{path_stem}_florence_object_captions.md")

        florence_scene_json = {
            "variant": "florence_only",
            "track": track_key,
            "image_path": image_path,
            "source": "Florence2 object captions aggregated",
            "generated_caption": "",
            "status": "pending_external_generation",
            "input_files": {
                "scene_json": scene_json_rel,
                "relations_json": relations_json_rel,
                "layers_json": layers_json_rel,
                "mask_hierarchy_json": hierarchy_json_rel,
                **(
                    {
                        "regions_json": regions_json_rel,
                        "regions_image": regions_png_rel,
                        "regions_overlay_image": regions_overlay_rel,
                        **(
                            {
                                "region_segmentation_image": region_segmentation_rel,
                                "region_sam2_segmentation_image": region_sam2_seg_rel,
                                "region_tinted_overlay_image": region_tinted_rel,
                            }
                            if region_segmentation_rel
                            else {}
                        ),
                    }
                    if regions_json_rel
                    else {}
                ),
            },
        }
        # Try local Florence-2 full-image detailed caption generation first.
        # If unavailable/failing, keep pending_external_generation placeholders.
        florence_scene_caption = ""
        florence_scene_status = "pending_external_generation"
        try:
            from PIL import Image as PILImage

            pil_full = PILImage.open(image_path).convert("RGB")
            if self._florence_init_failed():
                local_f2 = None
            else:
                local_f2 = self.florence2 if (self.florence2 is not None and self.florence2.active) else self._new_florence2_wrapper()
            if local_f2 is not None and local_f2.active:
                cap_res = local_f2._run_task("<MORE_DETAILED_CAPTION>", pil_full)
                florence_scene_caption = str(cap_res.get("<MORE_DETAILED_CAPTION>", "")).strip()
                if florence_scene_caption:
                    florence_scene_status = "generated_local_florence"
                else:
                    cap_res2 = local_f2._run_task("<CAPTION>", pil_full)
                    florence_scene_caption = str(cap_res2.get("<CAPTION>", "")).strip()
                    if florence_scene_caption:
                        florence_scene_status = "generated_local_florence_fallback_caption"
        except Exception as e:
            print(f"  [Florence2] full-image scene caption failed: {e}")

        florence_scene_json["generated_caption"] = florence_scene_caption
        florence_scene_json["status"] = florence_scene_status
        self._write_json(florence_scene_json, track_dir / f"{path_stem}_florence_scene_caption.json")
        self._write_text(
            self._md_heading(f"Florence Scene Caption - {track_key}") +
            f"Status: {florence_scene_status}\n\n" +
            (
                florence_scene_caption + "\n"
                if florence_scene_caption
                else "No local Florence caption generated. Use `*_florence_object_captions.json` as source summary.\n"
            ),
            track_dir / f"{path_stem}_florence_scene_caption.md",
        )

        fusion_scene_json = {
            "variant": "fusion_only",
            "track": track_key,
            "image_path": image_path,
            "prompt": fusion_prompt,
            "generated_caption": "",
            "status": "pending_external_generation",
            "input_files": {
                "scene_json": scene_json_rel,
                "relations_json": relations_json_rel,
                "layers_json": layers_json_rel,
                "mask_hierarchy_json": hierarchy_json_rel,
                "depth_mask_A_json": depth_mask_a_json_rel,
                "depth_mask_B_json": depth_mask_b_json_rel,
                "segmentation_image": segmentation_rel,
                "tinted_overlay_image": tinted_overlay_rel,
                "relations_map_image": relations_map_rel,
                "layers_image": layers_png_rel,
                "mask_hierarchy_image": hierarchy_png_rel,
                **(
                    {
                        "regions_json": regions_json_rel,
                        "regions_image": regions_png_rel,
                        "regions_overlay_image": regions_overlay_rel,
                        **(
                            {
                                "region_segmentation_image": region_segmentation_rel,
                                "region_sam2_segmentation_image": region_sam2_seg_rel,
                                "region_tinted_overlay_image": region_tinted_rel,
                            }
                            if region_segmentation_rel
                            else {}
                        ),
                    }
                    if regions_json_rel
                    else {}
                ),
            },
        }
        self._write_json(fusion_scene_json, track_dir / f"{path_stem}_fusion_scene_caption.json")
        self._write_text(
            self._md_heading(f"Fusion Scene Caption - {track_key}") + fusion_prompt + "\n",
            track_dir / f"{path_stem}_fusion_scene_caption.md",
        )

        hybrid_scene_json = {
            "variant": "hybrid",
            "track": track_key,
            "image_path": image_path,
            "status": "pending_external_generation",
            "inputs": {
                "florence_object_captions": f"scene_graph/{track_dir_name}/{path_stem}_florence_object_captions.json",
                "florence_scene_caption": f"scene_graph/{track_dir_name}/{path_stem}_florence_scene_caption.json",
                "fusion_scene_caption": f"scene_graph/{track_dir_name}/{path_stem}_fusion_scene_caption.json",
            },
            "generated_caption": "",
        }
        self._write_json(hybrid_scene_json, track_dir / f"{path_stem}_hybrid_scene_caption.json")
        self._write_text(
            self._md_heading(f"Hybrid Scene Caption - {track_key}") +
            "Status: pending external generation.\n" +
            "- Combine Florence-only and fusion-only outputs for final comparison-ready caption.\n",
            track_dir / f"{path_stem}_hybrid_scene_caption.md",
        )

        comparison_json = {
            "track": track_key,
            "variants": [
                {"name": "florence_only", "file": f"scene_graph/{track_dir_name}/{path_stem}_florence_scene_caption.json"},
                {"name": "fusion_only", "file": f"scene_graph/{track_dir_name}/{path_stem}_fusion_scene_caption.json"},
                {"name": "hybrid", "file": f"scene_graph/{track_dir_name}/{path_stem}_hybrid_scene_caption.json"},
            ],
            "scoring_template": {
                "faithfulness_to_image": None,
                "scene_graph_consistency": None,
                "relation_quality": None,
                "detail_richness": None,
            },
        }
        self._write_json(comparison_json, track_dir / f"{path_stem}_caption_comparison.json")

        bundle = {
            "track": track_key,
            "image_path": image_path,
            "files": {
                "florence_object_captions_json": f"scene_graph/{track_dir_name}/{path_stem}_florence_object_captions.json",
                "florence_object_captions_md": f"scene_graph/{track_dir_name}/{path_stem}_florence_object_captions.md",
                "florence_scene_caption_json": f"scene_graph/{track_dir_name}/{path_stem}_florence_scene_caption.json",
                "florence_scene_caption_md": f"scene_graph/{track_dir_name}/{path_stem}_florence_scene_caption.md",
                "fusion_scene_caption_json": f"scene_graph/{track_dir_name}/{path_stem}_fusion_scene_caption.json",
                "fusion_scene_caption_md": f"scene_graph/{track_dir_name}/{path_stem}_fusion_scene_caption.md",
                "hybrid_scene_caption_json": f"scene_graph/{track_dir_name}/{path_stem}_hybrid_scene_caption.json",
                "hybrid_scene_caption_md": f"scene_graph/{track_dir_name}/{path_stem}_hybrid_scene_caption.md",
                "caption_comparison_json": f"scene_graph/{track_dir_name}/{path_stem}_caption_comparison.json",
            },
        }
        self._write_json(bundle, track_dir / f"{path_stem}_hybrid_caption_bundle.json")

    def _save_track_prompt_bundle(
        self,
        out_dir: Path,
        image_path: str,
        path_stem: str,
        track_key: str,
        track_dir_name: str,
        scene_json_rel: str,
        relations_json_rel: str,
        layers_json_rel: str,
        hierarchy_json_rel: str,
        depth_mask_a_json_rel: str,
        depth_mask_b_json_rel: str,
        segmentation_rel: str,
        tinted_overlay_rel: str,
        relations_map_rel: str,
        layers_png_rel: str,
        hierarchy_png_rel: str,
        regions_json_rel: str = "",
        regions_png_rel: str = "",
        regions_overlay_rel: str = "",
        region_segmentation_rel: str = "",
        region_sam2_seg_rel: str = "",
        region_tinted_rel: str = "",
    ) -> None:
        region_block = ""
        if regions_json_rel:
            region_block = (
                f"- regions JSON: {regions_json_rel}\n"
                f"- regions index image: {regions_png_rel}\n"
                f"- regions overlay image: {regions_overlay_rel}\n"
            )
        if region_segmentation_rel:
            region_block += (
                f"- region segmentation image (parallel): {region_segmentation_rel}\n"
                f"- region SAM2-style segmentation (parallel): {region_sam2_seg_rel}\n"
                f"- region tinted overlay (parallel): {region_tinted_rel}\n"
            )
        prompt = f"""You are a scene-grounded captioner and verifier.

Use ALL provided artifacts jointly:
- Original image: {image_path}
- Track: {track_key}
- scene_graph JSON: {scene_json_rel}
- relations JSON: {relations_json_rel}
- layers JSON: {layers_json_rel}
- mask hierarchy JSON: {hierarchy_json_rel}
- depth_mask_A JSON: {depth_mask_a_json_rel}
- depth_mask_B JSON: {depth_mask_b_json_rel}
- segmentation image: {segmentation_rel}
- tinted overlay: {tinted_overlay_rel}
- relations map image: {relations_map_rel}
- layers image: {layers_png_rel}
- mask hierarchy image: {hierarchy_png_rel}
{region_block}
Goals:
1) Write a detailed caption (10-16 sentences) grounded in BOTH image evidence and scene-graph evidence.
2) Include object attributes, spatial relations, depth/layer cues, and part-whole hierarchy when supported.
3) Avoid hallucinations: do not mention entities absent from both image and graph.
4) If evidence conflicts, explicitly flag uncertainty and alternatives.

Output format:
A. Detailed Caption

B. Evidence-backed Claims Table
For each sentence/claim provide:
- claim_text
- object_ids_involved
- evidence_source: [image | scene_graph | both]
- supporting_fields
- confidence: [high | medium | low]

C. Ambiguities / Inconsistencies

D. Compact Machine-Readable Summary (JSON)
{{
  "primary_scene_theme": "...",
  "key_objects": [{{"id":"...", "name":"...", "role":"..."}}],
  "key_relations": [{{"sub":"...", "pred":"...", "obj":"...", "confidence":"..."}}],
  "depth_structure": "...",
  "uncertainties": []
}}
"""
        bundle = {
            "image_path": image_path,
            "track": track_key,
            "files": {
                "scene_json": scene_json_rel,
                "relations_json": relations_json_rel,
                "layers_json": layers_json_rel,
                "mask_hierarchy_json": hierarchy_json_rel,
                "depth_mask_A_json": depth_mask_a_json_rel,
                "depth_mask_B_json": depth_mask_b_json_rel,
                "segmentation_image": segmentation_rel,
                "tinted_overlay_image": tinted_overlay_rel,
                "relations_map_image": relations_map_rel,
                "layers_image": layers_png_rel,
                "mask_hierarchy_image": hierarchy_png_rel,
                **(
                    {
                        "regions_json": regions_json_rel,
                        "regions_image": regions_png_rel,
                        "regions_overlay_image": regions_overlay_rel,
                        **(
                            {
                                "region_segmentation_image": region_segmentation_rel,
                                "region_sam2_segmentation_image": region_sam2_seg_rel,
                                "region_tinted_overlay_image": region_tinted_rel,
                            }
                            if region_segmentation_rel
                            else {}
                        ),
                    }
                    if regions_json_rel
                    else {}
                ),
            },
        }
        self._write_text(
            prompt,
            out_dir / track_dir_name / f"{path_stem}_caption_prompt.md",
        )
        self._write_json(bundle, out_dir / track_dir_name / f"{path_stem}_caption_prompt_bundle.json")

    def _save_track_comparison_prompt(
        self,
        scene_graph_dir: Path,
        image_path: str,
        path_stem: str,
        available_tracks: List[Dict[str, str]],
    ) -> None:
        if len(available_tracks) < 2:
            return
        track_lines = []
        for tr in available_tracks:
            track_lines.append(
                f"- Track {tr['track']}: scene={tr['scene_json']}, relations={tr['relations_json']}, map={tr['relations_map_image']}"
            )
        prompt = """You are comparing multiple scene-graph pipelines on the same image.

Inputs:
"""
        prompt += f"- Image: {image_path}\n"
        prompt += "\n".join(track_lines)
        prompt += """

Tasks:
1) Produce one best unified detailed caption using image as primary truth and graph evidence as structure.
2) Score each track (0-10): object coverage, label accuracy, relation fidelity, depth/layer plausibility, overall downstream usefulness.
3) Report per-track errors: false positives, missing objects, relation mismatches.
4) Recommend best track for downstream use and why.

Output:
- Unified caption
- Score table
- Error list per track
- Final recommendation
"""
        self._write_text(prompt, scene_graph_dir / f"{path_stem}_track_comparison_prompt.md")

    def _print_relation_source_status(self) -> None:
        print("=== Relation Source Diagnostics ===")
        for name, status in self._relation_source_status.items():
            state = "ACTIVE" if status.get("active") else "INACTIVE"
            backend = status.get("backend", "unknown")
            reason = status.get("reason", "")
            print(f"{name}: {state} (backend={backend})")
            if reason:
                print(f"  reason: {reason}")

    def _assert_relation_sources_or_fail(self) -> None:
        if not self.require_any_relation_source:
            return
        if any(s.get("active") for s in self._relation_source_status.values()):
            return
        details = "; ".join(
            f"{name}: {status.get('reason', 'inactive')}"
            for name, status in self._relation_source_status.items()
        )
        raise RuntimeError(
            "No active relation source is available. "
            f"Disable strict check via require_any_relation_source=False, or fix dependencies. Details: {details}"
        )

    def _scene_pipeline_profile_enabled(self) -> bool:
        env = str(os.getenv("CITV_PROFILE", "")).strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        cfg = getattr(self, "config", None)
        return bool(getattr(cfg, "scene_pipeline_profile", False)) if cfg is not None else False

    def _florence2_wrapper_kwargs(self) -> dict:
        """Gather Florence-2 construction kwargs from config.

        Centralised so every construction site picks up the same Phase B.1
        configuration: dtype, attention implementation, and KV-cache toggle.
        """
        cfg = getattr(self, "config", None)
        dtype = ""
        attn_impl = "sdpa"
        use_cache = True
        od_fallback_enabled = True
        if cfg is not None:
            dtype = str(getattr(cfg, "florence2_dtype", "") or "")
            attn_impl = str(getattr(cfg, "florence2_attn_impl", "sdpa") or "sdpa")
            use_cache = bool(getattr(cfg, "florence2_use_cache", True))
            od_fallback_enabled = bool(getattr(cfg, "florence2_od_fallback_enabled", True))
            # Phase E — if the wrapper-specific florence2_dtype is left blank,
            # fall back to the global models_dtype opt-in so "fp16 everywhere"
            # can be enabled with one switch.  fp32 = "", so this is a no-op
            # when the opt-in is off.
            if not dtype:
                global_dtype = str(getattr(cfg, "models_dtype", "fp32") or "fp32")
                if global_dtype in {"fp16", "bf16"}:
                    dtype = global_dtype
        kwargs = {
            "use_cache": use_cache,
            "attn_implementation": attn_impl,
            "od_fallback_enabled": od_fallback_enabled,
        }
        if dtype:
            kwargs["dtype"] = dtype
        return kwargs

    def _florence_init_failed(self) -> bool:
        return self._florence2_backend != "mlx" and bool(getattr(Florence2Wrapper, "_INIT_FAILED", False))

    def _new_florence2_wrapper(self) -> Any:
        kwargs = self._florence2_wrapper_kwargs()
        if self._florence2_backend == "mlx":
            dtype = str(kwargs.get("dtype", "") or "fp16")
            _cfg = getattr(self, "config", None)
            _mlx = Florence2MLXWrapper(
                model_id=self._florence2_model_id,
                dtype=dtype if dtype in ("fp16", "fp32") else "fp16",
                od_fallback_enabled=bool(kwargs.get("od_fallback_enabled", True)),
                python_executable=(
                    str(getattr(_cfg, "florence2_mlx_python", ".venv/bin/python3.14"))
                    if _cfg is not None
                    else ".venv/bin/python3.14"
                ),
                start_timeout_s=(
                    float(getattr(_cfg, "florence2_mlx_start_timeout_s", 180.0))
                    if _cfg is not None
                    else 180.0
                ),
                request_timeout_s=(
                    float(getattr(_cfg, "florence2_mlx_request_timeout_s", 180.0))
                    if _cfg is not None
                    else 180.0
                ),
            )
            if getattr(_mlx, "active", False):
                return _mlx
            print("  [Florence2] MLX backend unavailable; falling back to HF wrapper.")
        return Florence2Wrapper(
            model_id=self._florence2_model_id,
            device=self.device,
            **kwargs,
        )

    def _release_heavy_models_after_segmentation(self) -> None:
        """Free depth + GDINO + SAM2 so they don't co-exist with Florence-2 / RAM++.

        By the time this is called the depth map is already materialised as a
        numpy array and all SAM2 masks have been extracted to CPU numpy too, so
        the GPU weights are no longer needed for this image. Reclaiming them
        shrinks the process footprint by ~3 GB on 8 GB M2 machines, which is
        the difference between staying resident and triggering swap.

        For multi-image runs, the batch dispatcher sets
        ``self._has_more_images_queued = True`` before each non-final image
        so we keep the heavy models resident across images — reloading
        GDINO + SAM2 + Depth costs ~4 s/image which adds up over folders.
        The last image (and single-image runs) falls back to the default
        release so the weights don't leak into post-processing stages.
        """
        if not getattr(self, "_release_heavy_models", False):
            return
        if getattr(self, "_has_more_images_queued", False):
            # Next image in this process will reuse the same wrappers —
            # skipping the unload saves ~4 s/image on multi-image runs.
            # Memory risk is bounded because Florence-2 / RAM++ still get
            # unloaded per image via their existing lifecycle and the
            # segmentation stage already ran so SAM2 KV caches are empty.
            print("  [mem] keeping Depth/GDINO/SAM2 resident (more images queued)")
            return
        try:
            if getattr(self, "sam2_wrapper", None) is not None:
                self.sam2_wrapper.unload()
                self.sam2_wrapper = None
        except Exception as _e:
            print(f"  [mem] sam2 unload skipped: {_e}")
        try:
            if getattr(self, "depth_estimator", None) is not None:
                self.depth_estimator.unload_backend()
        except Exception as _e:
            print(f"  [mem] depth unload skipped: {_e}")

    def _ensure_sam2_ready(self) -> None:
        """Reconstruct GDINO + SAM2 before the next image's segmentation stage."""
        if getattr(self, "sam2_wrapper", None) is not None:
            return
        if not self._sam2_reinit_kwargs:
            raise RuntimeError("SAM2 wrapper released but no reinit kwargs captured.")
        from scene_understanding.segmentation.grounded_sam2 import GroundedSAM2Wrapper
        print("  [mem] Reloading GroundedSAM2 for new image...")
        self.sam2_wrapper = GroundedSAM2Wrapper(**self._sam2_reinit_kwargs)
        if not self.sam2_wrapper.active:
            raise RuntimeError("Grounded SAM2 failed to re-initialize.")

    def _ensure_depth_ready(self) -> None:
        """Reconstruct the depth backend before the next image's depth call."""
        if getattr(self, "depth_estimator", None) is None:
            return
        if getattr(self.depth_estimator, "backend", None) is not None:
            return
        try:
            self.depth_estimator.reload_backend()
        except Exception as _e:
            print(f"  [mem] depth reload failed: {_e}")
            raise

    def _ensure_florence_for_labelling(self) -> None:
        """Load Florence-2 on first per-mask labelling call when eager load was skipped.

        Respects :pyattr:`Florence2Wrapper._INIT_FAILED` — once we know Florence
        cannot load in this process (e.g. missing flash_attn, torch/transformers
        incompatibility) every subsequent call is a no-op so we don't re-pay
        the weight-download cost per region/per image.
        """
        if not self._florence2_label_enabled:
            return
        if self._florence_init_failed():
            return
        if self.florence2 is not None and getattr(self.florence2, "active", False):
            return
        self.florence2 = self._new_florence2_wrapper()
        if bool(getattr(self.config, "florence2_relation_enabled", False)) and self.florence2 is not None and getattr(self.florence2, "active", False):
            self.pix2sg._florence2 = self.florence2

    def _unload_florence_before_rampp_init(self) -> None:
        """
        Drop Florence-2 from memory before first RAM++ construction.

        RAM++ loads a large Swin checkpoint; keeping Florence resident at the same time
        can push unified memory over the edge (macOS SIGKILL). When Florence is only
        used for per-mask labelling (not pair-relations), unload it before RAM++ init.

        Phase B.3: skip the unload when downstream caption exports still need
        Florence-2. Reloading Florence after RAM++ was the dominant reload cost
        in folder runs (Florence alone is ~5–8 s per init). If the operator has
        explicitly enabled eager-unload-each-image they can still force the
        behaviour via ``unload_labellers_after_each_image=True``.
        """
        cfg = self.config
        if bool(getattr(cfg, "florence2_relation_enabled", False)):
            return
        if self.florence2 is None or not getattr(self.florence2, "active", False):
            return
        # When captions that need Florence are going to be exported later in the
        # same image pass, keep the model resident. Operators who really want
        # the old aggressive unload behaviour can set
        # unload_labellers_after_each_image=True (that setting also re-triggers
        # unload at the tail of process_image).
        caption_needs_florence = (
            bool(getattr(cfg, "export_florence_object_captions", False))
            or bool(getattr(cfg, "export_fusion_scene_captions", False))
            or bool(getattr(cfg, "export_hybrid_captions", False))
        )
        # Honour the explicit opt-in — if the user disabled caching they
        # clearly want a small memory profile more than a fast run.
        force_unload = bool(getattr(cfg, "unload_labellers_after_each_image", False))
        if caption_needs_florence and not force_unload:
            print(
                "  [Labellers] Keeping Florence-2 resident for later caption exports "
                "(set unload_labellers_after_each_image=True to force the old unload)."
            )
            return
        self.florence2.model = None
        self.florence2.processor = None
        self.florence2.active = False
        self.florence2 = None
        self.pix2sg._florence2 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()
        print("  [Labellers] Florence-2 unloaded before RAM++ init (memory peak reduction).")

    def _ensure_rampp_for_labelling(self) -> None:
        """Load RAM++ lazily and disable it for this run if init fails."""
        if not self._rampp_enabled or self._rampp_runtime_disabled:
            return
        if self.rampp is not None and getattr(self.rampp, "active", False):
            return
        self._unload_florence_before_rampp_init()
        try:
            _cfg = getattr(self, "config", None)
            _prefer_mps = bool(getattr(_cfg, "rampp_prefer_mps", False)) if _cfg is not None else False
            _mps_dtype = str(getattr(_cfg, "rampp_mps_dtype", "fp16") or "fp16") if _cfg is not None else "fp16"
            _watermark = float(getattr(_cfg, "rampp_mps_watermark_fraction", 0.85)) if _cfg is not None else 0.85
            self.rampp = RAMPlusPlusWrapper(
                device=self.device,
                checkpoint_path=self._rampp_checkpoint_path,
                repo_path=self._rampp_repo_path,
                image_size=self._rampp_image_size,
                vit=self._rampp_vit,
                default_confidence=self._rampp_default_conf,
                max_tags=self._rampp_max_tags,
                prefer_mps=_prefer_mps,
                mps_dtype=_mps_dtype,
                mps_watermark_fraction=_watermark,
                backend=self._rampp_backend,
                clip_fallback_enabled=self._rampp_clip_fallback_enabled,
                clip_candidate_labels=self._rampp_clip_candidate_labels,
                quantize_dynamic_int8=self._rampp_quantize_dynamic_int8,
            )
            if self.rampp is None or not self.rampp.active:
                self._rampp_runtime_disabled = True
                print("  [RAM++] init unavailable; RAM++ disabled for this run.")
        except Exception as e:
            self._rampp_runtime_disabled = True
            self.rampp = None
            print(f"  [RAM++] init failed ({e}); RAM++ disabled for this run.")

    def _load_labellers(self) -> None:
        """Prepare eager labellers needed for this image (RAM++ remains lazy)."""
        _rel = bool(getattr(self.config, "florence2_relation_enabled", False))
        _skip_secondary = bool(getattr(self, "_mask_label_skip_secondary_when_gdino_specific", True))
        # Eager Florence when pair-relations need it, or when every mask may run Florence (no GDINO short-circuit).
        need_florence_eager = _rel or (self._florence2_label_enabled and not _skip_secondary)
        if (
            need_florence_eager
            and not self._florence_init_failed()
            and (self.florence2 is None or not self.florence2.active)
        ):
            self.florence2 = self._new_florence2_wrapper()

        # Inject loaded Florence-2 into Pix2SG so relations use it
        if bool(getattr(self.config, "florence2_relation_enabled", False)) and self.florence2 is not None and self.florence2.active:
            self.pix2sg._florence2 = self.florence2
        else:
            self.pix2sg._florence2 = None

    def _unload_labellers(self) -> None:
        """Unload Florence-2 and RAM++ to free VRAM after Stage 5."""
        if self.florence2 is not None:
            self.florence2.model = None
            self.florence2.processor = None
            self.florence2.active = False
        self.florence2 = None
        if self.rampp is not None:
            self.rampp.model = None
            self.rampp.transform = None
            self.rampp.inference_fn = None
            self.rampp.active = False
        self.rampp = None
        self.pix2sg._florence2 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()
        print("  [Labellers] Florence-2 / RAM++ unloaded, VRAM freed.")

    @staticmethod
    def _load_calibration(cal_file: str) -> Optional[Dict]:
        """
        Fix 5.2: Load OpenCV camera calibration JSON produced by
        tools/calibrate_camera.py.

        Expected JSON structure:
          {
            "fx": float, "fy": float, "cx": float, "cy": float,
            "k1": float, "k2": float, "p1": float, "p2": float,
            "image_size": [w, h]
          }

        Why calibration matters:
          A checkerboard calibration with 20+ images gives focal length
          accurate to <0.5% and principal point to <2 px.  The FOV estimate
          can be off by 10-30% for non-standard lenses or cropped sensors,
          directly corrupting X/Y in coordinates_3d.
          Distortion coefficients (k1,k2,p1,p2) correct barrel/pincushion
          distortion; without undistortion, depth back-projection assumes a
          perfect pinhole which is violated by real lenses.
        """
        try:
            with open(cal_file, "r") as f:
                cal = json.load(f)
            required = {"fx", "fy", "cx", "cy"}
            if not required.issubset(cal.keys()):
                print(f"[Calibration] Missing keys in {cal_file}. Need {required}.")
                return None
            print(f"[Calibration] Loaded from {cal_file}: "
                  f"fx={cal['fx']:.1f} fy={cal['fy']:.1f} "
                  f"cx={cal['cx']:.1f} cy={cal['cy']:.1f}")
            return cal
        except Exception as e:
            print(f"[Calibration] Failed to load {cal_file}: {e}")
            return None

    def _undistort_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Fix 5.2: Apply lens distortion correction using calibration coefficients.

        Uses OpenCV cv2.undistort() with the loaded k1,k2,p1,p2 coefficients.
        If no calibration is loaded or undistortion is disabled, returns the
        image unchanged.

        When to enable:
          - DSLR / mirrorless cameras: usually mild distortion (k1 ≈ -0.05)
          - Wide-angle / fisheye lenses: strong distortion (k1 ≈ -0.3 or worse)
          - Smartphone cameras at 0.5× zoom: significant barrel distortion
        """
        if self._calibration is None or not self.apply_undistortion:
            return img_bgr
        try:
            cal = self._calibration
            h, w = img_bgr.shape[:2]
            K_mat = np.array([
                [cal["fx"], 0.0,       cal["cx"]],
                [0.0,       cal["fy"], cal["cy"]],
                [0.0,       0.0,       1.0],
            ], dtype=np.float64)
            dist_coeffs = np.array([
                cal.get("k1", 0.0), cal.get("k2", 0.0),
                cal.get("p1", 0.0), cal.get("p2", 0.0),
            ], dtype=np.float64)
            return cv2.undistort(img_bgr, K_mat, dist_coeffs)
        except Exception as e:
            print(f"[Undistort] Failed: {e}. Using original image.")
            return img_bgr

    def _estimate_intrinsics(self, width: int, height: int) -> Dict[str, float]:
        """
        Fix 5.2: Return camera intrinsics with priority order:
          1. Calibration file (OpenCV checkerboard calibration — most accurate)
          2. Explicit camera_fx / camera_fy / camera_cx / camera_cy in config
          3. FOV-based estimate (least accurate; error can be 10-30%)

        The returned dict is used for all back-projection in Stage 4.
        """
        # Priority 1: calibration file
        if self._calibration is not None:
            cal = self._calibration
            return {
                "fx": float(cal["fx"]),
                "fy": float(cal["fy"]),
                "cx": float(cal.get("cx", width / 2)),
                "cy": float(cal.get("cy", height / 2)),
            }
        # Priority 2: explicit values
        if self.camera_fx is not None:
            return {
                "fx": float(self.camera_fx),
                "fy": float(self.camera_fy if self.camera_fy is not None else self.camera_fx),
                "cx": float(self.camera_cx if self.camera_cx is not None else width / 2),
                "cy": float(self.camera_cy if self.camera_cy is not None else height / 2),
            }
        # Priority 3: FOV estimate
        f_x = (width / 2) / np.tan(np.deg2rad(self.camera_fov_degrees) / 2)
        print(f"  [Intrinsics] Using FOV estimate ({self.camera_fov_degrees}°): fx=fy={f_x:.1f}")
        return {"fx": f_x, "fy": f_x, "cx": width / 2, "cy": height / 2}

    def _back_project(self, u: int, v: int, z: float, K: Dict[str, float]) -> Dict[str, float]:
        x = (u - K['cx']) * z / K['fx']
        y = (v - K['cy']) * z / K['fy']
        return {"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3)}

    @staticmethod
    def _bbox_iou_xyxy(box1: List[float], box2: List[float]) -> float:
        """IoU of two boxes in xyxy format [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-8)

    @staticmethod
    def _xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
        x, y, w, h = bbox_xywh[:4]
        return [x, y, x + w, y + h]

    def _match_mask_first(
        self,
        amg_masks: List[Dict],
        detections: List[Dict],
        iou_thresh: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Option B: Identity mapping — detection i corresponds to amg_masks[i].

        Since all_detections is built directly from amg_masks (one entry per mask),
        the match is exact: mask index == detection index. The iou_thresh parameter
        is kept for API compatibility but is not used.
        """
        out = []
        for idx, amg in enumerate(amg_masks):
            seg = amg.get("segmentation")
            if seg is None:
                continue
            mask = np.asarray(seg) if not isinstance(seg, np.ndarray) else seg
            amg_xyxy = self._xywh_to_xyxy(amg.get("bbox", [0, 0, 0, 0]))
            det = detections[idx] if idx < len(detections) else None
            out.append({
                "mask": mask,
                "sam2_mask_index": idx,
                "detection": det,
                "mask_bbox_xyxy": amg_xyxy,
            })
        return out

    def _adaptive_erosion_kernel(self, mask_bin: np.ndarray) -> int:
        """Return erosion kernel size scaled to the mask's narrowest dimension. 0 = skip erosion."""
        if not self.depth_adaptive_erosion or self.mask_erosion_kernel_size == 0:
            return self.mask_erosion_kernel_size
        ys, xs = np.where(mask_bin)
        if ys.size == 0:
            return 0
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        min_dim = min(bbox_h, bbox_w)
        # Scale table: fraction of narrowest dimension used as kernel
        if min_dim < 15:
            return 0    # very thin object — skip erosion entirely
        elif min_dim < 40:
            return 1
        elif min_dim < 80:
            return 2
        elif min_dim < 150:
            return min(3, self.mask_erosion_kernel_size)
        else:
            return self.mask_erosion_kernel_size

    # ----- Phase A helpers: object label map, contact graph, RLE, depth cache -----

    @staticmethod
    def _build_object_label_map(
        masks: List[np.ndarray], height: int, width: int
    ) -> np.ndarray:
        """Stamp N masks into a single int32 label map, pixel painted with the last mask that covers it.

        Returns an ``int32`` array where 0 = background and each mask ``i`` paints
        label ``i + 1``. Downstream consumers can then use
        ``scipy.ndimage.find_objects`` / ``center_of_mass`` / ``labeled_comprehension``
        to compute per-object statistics in one vectorised pass rather than N
        Python loops, and `bincount` to get areas.

        Overlap policy: later masks win (painted on top). This mirrors the
        existing mask order semantics where detection i+1 is considered after
        detection i; it keeps per-pixel ownership consistent with the mask list
        and the mask_hierarchy export.
        """
        label_map = np.zeros((int(height), int(width)), dtype=np.int32)
        for i, m in enumerate(masks):
            if m is None:
                continue
            mb = np.asarray(m) > 0
            if mb.shape[:2] != (height, width):
                mb = cv2.resize(mb.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            label_map[mb] = i + 1
        return label_map

    @staticmethod
    def _build_contact_graph(
        object_label_map: np.ndarray,
        *,
        dilate_radius_px: int = 1,
    ) -> set:
        """Return the set of pixel-touching (i, j) pairs (i < j).

        Performed as one morphological dilation + XOR on the label map so the
        cost is O(H*W) and independent of the number of masks. This replaces
        the O(N^2) "mask_i & mask_j" scan that used to gate Pix2SG pair
        relations.

        The returned pairs use the 0-indexed mask indices (i.e. label-1) so
        they plug directly into the existing Pix2SG `detections` list.
        """
        if object_label_map is None or dilate_radius_px <= 0:
            return set()
        lm = np.asarray(object_label_map, dtype=np.int32)
        if lm.size == 0:
            return set()
        k = int(dilate_radius_px)
        pairs: set = set()
        for shift_y, shift_x in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = lm[max(0, -shift_y): lm.shape[0] - max(0, shift_y) if shift_y else lm.shape[0],
                   max(0, -shift_x): lm.shape[1] - max(0, shift_x) if shift_x else lm.shape[1]]
            b = lm[max(0, shift_y): lm.shape[0] + min(0, shift_y),
                   max(0, shift_x): lm.shape[1] + min(0, shift_x)]
            if a.shape != b.shape or a.size == 0:
                continue
            diff = (a != b) & (a > 0) & (b > 0)
            if not np.any(diff):
                continue
            stacked = np.stack([a[diff], b[diff]], axis=1)
            for i1, i2 in stacked:
                i1i, i2i = int(i1) - 1, int(i2) - 1
                if i1i < 0 or i2i < 0 or i1i == i2i:
                    continue
                pairs.add((min(i1i, i2i), max(i1i, i2i)))
        if k > 1:  # wider neighbourhood via cv2 dilate for larger radii
            try:
                kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
                for li in range(1, int(lm.max()) + 1):
                    mi = (lm == li).astype(np.uint8)
                    if mi.sum() == 0:
                        continue
                    di = cv2.dilate(mi, kernel) > 0
                    overlapping = np.unique(lm[di])
                    for lj in overlapping:
                        lji = int(lj) - 1
                        if lji < 0:
                            continue
                        if lji == li - 1:
                            continue
                        pairs.add((min(li - 1, lji), max(li - 1, lji)))
            except Exception:
                pass
        return pairs

    @staticmethod
    def _encode_mask_rle(mask: np.ndarray) -> Optional[Dict[str, Any]]:
        """Encode a binary mask to COCO RLE. Returns None if pycocotools is absent.

        The returned dict has keys ``size=[h, w]`` and ``counts`` (ASCII string),
        matching the COCO / pycocotools format. Stored in JSON under a new
        additive ``mask_rle`` field; older consumers ignore unknown keys.
        """
        try:
            from pycocotools import mask as _mutil  # type: ignore
        except Exception:
            return None
        try:
            mb = np.asfortranarray(np.asarray(mask) > 0)
            rle = _mutil.encode(mb.astype(np.uint8))
            counts = rle.get("counts")
            if isinstance(counts, bytes):
                counts = counts.decode("ascii")
            return {
                "size": [int(rle["size"][0]), int(rle["size"][1])],
                "counts": counts,
            }
        except Exception:
            return None

    def _mask_depth_stats_and_3d_cached(
        self,
        host: Optional[Dict[str, Any]],
        *args: object,
        use_erosion: bool = True,
        cache_key_suffix: str = "",
        **kwargs: object,
    ) -> tuple:
        """Memoise :meth:`_mask_depth_stats_and_3d` on the provided host dict.

        The plan requires caching depth stats on each det/region dict because
        the same tuple is recomputed up to four times per object (erosion /
        no-erosion and matched-A / matched-B mapping). Using the mutable dict
        as the cache container keeps the fix side-effect-free for consumers
        that never pass a host (they pay the full cost, same as today).
        """
        if not isinstance(host, dict):
            return self._mask_depth_stats_and_3d(*args, use_erosion=use_erosion, **kwargs)
        field_name = "_depth_stats_erosion" if use_erosion else "_depth_stats_raw"
        if cache_key_suffix:
            field_name = f"{field_name}_{cache_key_suffix}"
        cached = host.get(field_name)
        if isinstance(cached, tuple) and len(cached) == 3:
            return cached
        result = self._mask_depth_stats_and_3d(*args, use_erosion=use_erosion, **kwargs)
        host[field_name] = result
        return result

    def _global_depth_stats(self, metric_depth: np.ndarray) -> Dict[str, float]:
        """Return finite-depth mean/std/median/p05/p95 for the current image.

        Cached on the instance by ``(id(metric_depth), shape, dtype)`` so every
        call that needs a global fallback for sigma-clipping shares the same
        histogram pass. Output is a plain dict, never mutated by callers.
        """
        cache = getattr(self, "_global_depth_stats_cache", None)
        if cache is None:
            cache = {}
            self._global_depth_stats_cache = cache
        md = metric_depth
        key = (id(md), getattr(md, "shape", None), getattr(md, "dtype", None).str if md is not None else None)
        cached = cache.get(key)
        if cached is not None:
            return cached
        if md is None or md.size == 0:
            stats = {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p05": 0.0, "p95": 0.0, "num_finite": 0}
        else:
            finite = md[np.isfinite(md)]
            if finite.size == 0:
                stats = {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p05": 0.0, "p95": 0.0, "num_finite": 0}
            else:
                p05, p95 = np.percentile(finite, [5.0, 95.0])
                stats = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "median": float(np.median(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "p05": float(p05),
                    "p95": float(p95),
                    "num_finite": int(finite.size),
                }
        cache[key] = stats
        return stats

    def _mask_depth_stats_and_3d(
        self,
        metric_depth: np.ndarray,
        K: Dict[str, float],
        mask: np.ndarray,
        detection: Optional[Dict] = None,
        use_erosion: bool = True,
        region_context: Optional[Dict[str, Any]] = None,
        label_map: Optional[np.ndarray] = None,
        region_index: int = 0,
    ) -> tuple:
        """
        Compute depth stats and 3D coords from mask pixels.
        use_erosion=False skips adaptive erosion (for comparison stats).
        Returns (depth_stats_dict, coordinates_3d, mask_centroid_2d).
        See docs/DEPTH_ACCURACY.md for all formulas.
        """
        del detection
        h, w = metric_depth.shape[:2]
        mask_bin = (np.asarray(mask) > 0)
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_bin > 0)

        if use_erosion:
            kernel_size = self._adaptive_erosion_kernel(mask_bin)
            if region_context:
                rtype = str(region_context.get("type", "")).lower()
                if rtype == "background":
                    kernel_size = 0
            if kernel_size > 0 and int(mask_bin.sum()) > 4 * kernel_size * kernel_size:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded = cv2.erode(mask_bin.astype(np.uint8), kernel, iterations=1)
                if eroded.sum() > 0:
                    mask_bin = (eroded > 0)

        ys, xs = np.where(mask_bin)
        depth_at_mask = metric_depth[ys, xs]
        finite_mask = np.isfinite(depth_at_mask)
        depth_at_mask = depth_at_mask[finite_mask]
        ys_f = ys[finite_mask]
        xs_f = xs[finite_mask]

        if depth_at_mask.size == 0:
            depth_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0,
                           "num_pixels": 0, "z_val": 0.0, "z_val_pixels": 0,
                           "possibly_transparent": False, "depth_separation_from_background": 0.0}
            coords_3d = {"x": 0.0, "y": 0.0, "z": 0.0}
            centroid = [w // 2, h // 2]
            return depth_stats, coords_3d, centroid

        # Sigma-clipping: reject |depth_i - mean| > sigma * std  (see docs/DEPTH_ACCURACY.md)
        # Phase B.7 — when the configured scope is "global", reuse the cached
        # per-image depth histogram stats instead of recomputing them per mask.
        sigma = self.depth_outlier_sigma
        sigma_scope = str((region_context or {}).get("sigma_scope") or getattr(self, "depth_sigma_clip_scope", "mask"))
        if sigma > 0 and depth_at_mask.size >= 10:
            mean_d = float(np.mean(depth_at_mask))
            std_d = float(np.std(depth_at_mask))
            if sigma_scope == "region" and region_context:
                rs = region_context.get("depth_stats") or {}
                if rs.get("std") is not None and float(rs["std"]) > 1e-6:
                    mean_d = float(rs.get("mean", mean_d))
                    std_d = float(rs["std"])
            elif sigma_scope == "global":
                gs = self._global_depth_stats(metric_depth)
                if float(gs.get("std", 0.0)) > 1e-6:
                    mean_d = float(gs.get("mean", mean_d))
                    std_d = float(gs.get("std", std_d))
            if std_d > 1e-6:
                # Phase D — delegate to the (optionally Numba-JIT) kernel so we
                # avoid the ``np.abs(…) < …`` temporary on large masks.
                try:
                    from scene_understanding.numerics import masked_sigma_clip as _msc
                    inlier = _msc(depth_at_mask, mean_d, std_d, sigma)
                except Exception:
                    inlier = np.abs(depth_at_mask - mean_d) < sigma * std_d
                if inlier.sum() >= 5:
                    depth_at_mask = depth_at_mask[inlier]
                    ys_f = ys_f[inlier]
                    xs_f = xs_f[inlier]

        # Transparency detection via 5px border ring (see docs/DEPTH_ACCURACY.md)
        possibly_transparent = False
        depth_separation = 0.0
        if self.depth_transparency_check and mask_bin.sum() > 0:
            try:
                kernel_5 = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(mask_bin.astype(np.uint8), kernel_5) > 0
                border_ring = dilated & ~mask_bin
                if (
                    label_map is not None
                    and int(region_index) > 0
                    and np.asarray(label_map).shape[:2] == (h, w)
                ):
                    lm = np.asarray(label_map, dtype=np.int32)
                    ring_same = border_ring & (lm == int(region_index))
                    if int(np.count_nonzero(ring_same)) >= 8:
                        border_ring = ring_same
                border_depths = metric_depth[border_ring]
                border_depths = border_depths[np.isfinite(border_depths)]
                if border_depths.size > 0 and depth_at_mask.size > 0:
                    mask_mean = float(np.mean(depth_at_mask))
                    border_mean = float(np.mean(border_depths))
                    depth_separation = abs(mask_mean - border_mean)
                    possibly_transparent = depth_separation < self.depth_transparency_threshold
            except Exception:
                pass

        # Depth-weighted centroid: w_i = 1/(depth_i + ε)  (see docs/DEPTH_ACCURACY.md)
        weights = 1.0 / (depth_at_mask + 1e-6)
        w_sum = float(weights.sum())
        cy_f = float(np.sum(ys_f * weights) / w_sum)
        cx_f = float(np.sum(xs_f * weights) / w_sum)

        # Nearest real mask pixel to the weighted centroid (no holes in anchor)
        dist2 = (ys_f - cy_f) ** 2 + (xs_f - cx_f) ** 2
        anchor_idx = int(np.argmin(dist2))
        cx = int(xs_f[anchor_idx])
        cy = int(ys_f[anchor_idx])

        # z_val: histogram mode over inner-circle pixels  (see docs/DEPTH_ACCURACY.md)
        central_frac = self.depth_central_fraction
        if central_frac < 1.0:
            area = float(mask_bin.sum())
            radius = np.sqrt(area * central_frac / np.pi)
            inner_mask = dist2 <= radius ** 2
            inner_depths = depth_at_mask[inner_mask]
        else:
            inner_depths = depth_at_mask
        z_val_pixels = int(inner_depths.size)
        if z_val_pixels > 0:
            n_bins = max(10, min(100, z_val_pixels // 5))
            hist, edges = np.histogram(inner_depths, bins=n_bins)
            peak_bin = int(np.argmax(hist))
            z_val = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)
        else:
            z_val = float(np.median(depth_at_mask))
            z_val_pixels = int(depth_at_mask.size)

        depth_stats = {
            "min": round(float(np.min(depth_at_mask)), 4),
            "max": round(float(np.max(depth_at_mask)), 4),
            "mean": round(float(np.mean(depth_at_mask)), 4),
            "median": round(float(np.median(depth_at_mask)), 4),
            "std": round(float(np.std(depth_at_mask)), 4),
            "num_pixels": int(mask_bin.sum()),
            "z_val": round(z_val, 4),
            "z_val_pixels": z_val_pixels,
            # Fix 5.7c: transparency diagnostics
            "possibly_transparent": bool(possibly_transparent),
            "depth_separation_from_background": round(depth_separation, 4),
        }
        coords_3d = self._back_project(cx, cy, z_val, K)
        return depth_stats, coords_3d, [cx, cy]

    def _label_mask(
        self,
        img_bgr: np.ndarray,
        mask_bin: np.ndarray,
        amg_entry: Dict[str, Any],
        label_map: Optional[np.ndarray] = None,
        region_index: int = 0,
        *,
        _secondary_phase: str = "full",
        _prefill_florence: Optional[Tuple[str, str]] = None,
        _prefill_rampp: Optional[Tuple[str, str, List[str], float]] = None,
    ) -> Dict[str, Any]:
        """
        Label a mask via priority chain: GDINO → Florence-2 (optional) → RAM++.

        When ``mask_label_skip_secondary_when_gdino_specific`` is enabled and GDINO
        already returned a specific class (not ``object``), Florence-2 and RAM++
        mask crops are skipped for speed.

        Keyword-only (internal):
            _secondary_phase: ``\"full\"`` (default), ``\"florence_only\"``, ``\"rampp_only\"``, or ``\"prefilled\"``.
            ``florence_only`` skips RAM++ for two-phase pass A; ``rampp_only`` skips Florence
            inference and uses ``_prefill_florence`` (florence2_label, florence2_caption) for pass B.
            ``prefilled`` skips BOTH models and uses ``_prefill_florence`` +
            ``_prefill_rampp`` — used by :meth:`_enrich_region_labels_from_masks_batched`
            to amortise 44 sequential single-image Florence-2 / RAM++ calls into one
            batched forward per model.
        """
        h_img, w_img = img_bgr.shape[:2]
        x, y, bw, bh = amg_entry.get("bbox", [0, 0, w_img, h_img])
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_img, int(x + bw)), min(h_img, int(y + bh))

        if x2 <= x1 or y2 <= y1:
            name_fields = self._choose_mask_name_fields(
                grounded_label="object",
                grounded_caption="object",
                grounded_confidence=0.0,
                florence_label="",
                florence_caption="",
                rampp_label="",
                rampp_caption="",
                rampp_tags=[],
                fallback_label="object",
            )
            return {
                "label": "object",
                "conf": 0.0,
                "caption": "object",
                "source_model": "fallback",
                "florence2_label": "",
                "florence2_caption": "",
                "rampp_label": "",
                "rampp_caption": "",
                "rampp_tags": [],
                **name_fields,
            }

        gdino_label = str(amg_entry.get("label", "object")).strip().lower()
        gdino_conf = float(amg_entry.get("gdino_conf", 0.0))
        skip_secondary = (
            bool(getattr(self, "_mask_label_skip_secondary_when_gdino_specific", True))
            and bool(gdino_label)
            and gdino_label != "object"
            and not self._is_generic_label(gdino_label)
        )

        f2_label, f2_caption = "object", "object"
        rampp_label, rampp_caption, rampp_tags = "object", "object", []
        rampp_conf = 0.0

        _phase = str(_secondary_phase or "full").strip().lower()
        if _phase not in ("full", "florence_only", "rampp_only", "prefilled"):
            _phase = "full"
        run_florence = _phase in ("full", "florence_only")
        run_rampp = _phase in ("full", "rampp_only")
        if _phase == "prefilled":
            run_florence = False
            run_rampp = False
            if _prefill_florence is not None:
                f2_label = str(_prefill_florence[0] or "object").strip().lower() or "object"
                f2_caption = str(_prefill_florence[1] or "object")
            if _prefill_rampp is not None:
                _pl, _pc, _pt, _pconf = _prefill_rampp
                rampp_label = str(_pl or "object").strip().lower() or "object"
                rampp_caption = str(_pc or "object")
                rampp_tags = list(_pt or [])
                rampp_conf = float(_pconf or 0.0)

        if _phase == "prefilled":
            # Skip the entire crop-build + model-call branch; the prefill
            # values above are authoritative. Priority-fusion logic below is
            # identical to the model-driven path.
            skip_secondary = True

        if not skip_secondary:
            # Mask-native tight crop: use actual mask extent rather than amg bbox for cleaner
            # foreground/background separation.  Falls back to amg bbox if mask is empty or
            # covers less than 5 % of the derived crop region.
            ys_m, xs_m = np.where(mask_bin.astype(bool))
            if ys_m.size > 0:
                _pad = 4
                y1 = max(0, int(ys_m.min()) - _pad)
                y2 = min(h_img, int(ys_m.max()) + 1 + _pad)
                x1 = max(0, int(xs_m.min()) - _pad)
                x2 = min(w_img, int(xs_m.max()) + 1 + _pad)
            crop = img_bgr[y1:y2, x1:x2].copy()
            # Direct slice — same coordinate frame as crop, no interpolation artefacts.
            mask_resized = mask_bin[y1:y2, x1:x2].astype(bool)
            # Coverage guard: if the mask is a tiny fraction of this crop fall back to the amg bbox.
            if mask_resized.sum() < max(int(mask_resized.size * 0.05), 1):
                _xb, _yb, _bw, _bh = amg_entry.get("bbox", [0, 0, w_img, h_img])
                _x1b = max(0, int(_xb))
                _y1b = max(0, int(_yb))
                _x2b = min(w_img, int(_xb + _bw))
                _y2b = min(h_img, int(_yb + _bh))
                if _x2b > _x1b and _y2b > _y1b:
                    crop = img_bgr[_y1b:_y2b, _x1b:_x2b].copy()
                    mask_resized = mask_bin[_y1b:_y2b, _x1b:_x2b].astype(bool)
            # Phase B.9 — cache ``bg_mean`` and ``{rid: lm==rid}`` per region so
            # consecutive objects sharing a region do not redo the full-image
            # equality broadcast + mean-over-selected-pixels.
            bg_mean = None
            if (
                label_map is not None
                and int(region_index) > 0
                and np.asarray(label_map).shape[:2] == (h_img, w_img)
            ):
                rid = int(region_index)
                bg_cache = getattr(self, "_region_bg_mean_cache", None)
                if bg_cache is None:
                    bg_cache = {}
                    self._region_bg_mean_cache = bg_cache
                mask_cache = getattr(self, "_region_pixel_mask_cache", None)
                if mask_cache is None:
                    mask_cache = {}
                    self._region_pixel_mask_cache = mask_cache
                cache_lm_id = (id(label_map), int(label_map.shape[0]), int(label_map.shape[1]))
                region_pixels = mask_cache.get((cache_lm_id, rid))
                if region_pixels is None:
                    lm = np.asarray(label_map, dtype=np.int32)
                    region_pixels = (lm == rid)
                    mask_cache[(cache_lm_id, rid)] = region_pixels
                bg_mean = bg_cache.get((cache_lm_id, rid))
                if bg_mean is None:
                    if region_pixels.any():
                        bg_mean = img_bgr[region_pixels].mean(axis=0).astype(np.uint8)
                    else:
                        bg_mean = img_bgr.mean(axis=(0, 1)).astype(np.uint8)
                    bg_cache[(cache_lm_id, rid)] = bg_mean
            if bg_mean is None:
                bg_mean = img_bgr.mean(axis=(0, 1)).astype(np.uint8)
            crop_filled = crop.copy()
            crop_filled[~mask_resized] = bg_mean

            if _phase == "rampp_only" and _prefill_florence is not None:
                f2_label = str(_prefill_florence[0] or "object").strip().lower() or "object"
                f2_caption = str(_prefill_florence[1] or "object")

            if run_florence:
                if self._florence2_label_enabled:
                    self._ensure_florence_for_labelling()
                if (
                    self._florence2_label_enabled
                    and self.florence2 is not None
                    and self.florence2.active
                ):
                    f2_result = self.florence2.label_crop(crop_filled)
                    f2_label = str(f2_result.get("label", "object")).strip().lower() or "object"
                    f2_caption = str(f2_result.get("caption", "object"))

            if run_rampp:
                self._ensure_rampp_for_labelling()
                if self.rampp is not None and self.rampp.active:
                    rampp_result = self.rampp.label_crop(crop_filled)
                    rampp_label = str(rampp_result.get("label", "object")).strip().lower() or "object"
                    rampp_caption = str(rampp_result.get("caption", "object"))
                    rampp_tags = list(rampp_result.get("tags", []))
                    rampp_conf = float(rampp_result.get("conf", 0.0))

        selected_label = "object"
        selected_conf = 0.0
        selected_caption = "object"
        selected_source = "fallback"

        # Priority 1: GDINO label (wins if specific — not "object")
        if gdino_label and gdino_label != "object":
            selected_label = gdino_label
            selected_conf = gdino_conf
            selected_caption = gdino_label
            selected_source = "GroundingDINO"

        # Priority 2: Florence-2
        if selected_label == "object" and f2_label != "object":
            selected_label = f2_label
            selected_conf = 0.75
            selected_caption = f2_caption
            selected_source = "Florence-2"

        # Priority 3: RAM++
        if selected_label == "object" and rampp_label != "object":
            selected_label = rampp_label
            selected_conf = rampp_conf
            selected_caption = rampp_caption
            selected_source = "RAM++"

        name_fields = self._choose_mask_name_fields(
            grounded_label=gdino_label,
            grounded_caption=gdino_label,
            grounded_confidence=gdino_conf,
            florence_label=f2_label,
            florence_caption=f2_caption,
            rampp_label=rampp_label,
            rampp_caption=rampp_caption,
            rampp_tags=rampp_tags,
            fallback_label=selected_label,
        )

        canonical_name = str(name_fields.get("canonical_name", selected_label)).strip().lower() or selected_label
        if self._is_attribute_like_label(selected_label) and not self._is_attribute_like_label(canonical_name):
            selected_label = canonical_name
            if selected_source == "GroundingDINO":
                selected_source = "evidence_fusion"
            selected_conf = max(float(selected_conf), 0.7)
            selected_caption = (
                rampp_caption
                if canonical_name in self._split_label_candidates(rampp_caption)
                else f2_caption or selected_caption
            )

        return {
            "label": selected_label if selected_label else "object",
            "conf": round(float(selected_conf), 4),
            "caption": selected_caption if selected_caption else (f2_caption if f2_caption else "object"),
            "source_model": selected_source,
            "florence2_label": f2_label,
            "florence2_caption": f2_caption,
            "rampp_label": rampp_label,
            "rampp_caption": rampp_caption,
            "rampp_tags": rampp_tags,
            **name_fields,
        }

    def _enrich_region_labels_from_masks(
        self,
        img_bgr: np.ndarray,
        region_partition_meta: List[Dict[str, Any]],
        label_map: np.ndarray,
        w: int,
        h: int,
        metric_depth: Optional[np.ndarray] = None,
        K: Optional[Dict[str, float]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Label depth-partition regions via Florence-2 + RAM++.
        Runs a primary _label_mask pass plus 3-crop RAM++ vote aggregation per region
        to build enriched tags, aliases, and a majority-vote canonical label.

        When metric_depth and K are provided, replaces the partitioner's raw depth
        stats with sigma-clipped histogram-mode stats from _mask_depth_stats_and_3d,
        and stores real back-projected coordinates_3d and depth-weighted centroid_2d_px
        on each region meta dict in-place.
        """
        out: Dict[int, Dict[str, Any]] = {}
        lm = np.asarray(label_map, dtype=np.int32)
        _rampp_max = getattr(self, "_rampp_max_tags", 8)

        # ------------------------------------------------------------------
        # Pass 1 — pure CPU: build per-region bookkeeping (mask, crops,
        # bbox, bg fill). Zero model calls here. Also write depth stats
        # back onto ``r`` so the exports after this function see the
        # sigma-clipped values.
        # ------------------------------------------------------------------
        ctx: List[Dict[str, Any]] = []  # per-region work items (in region order)
        for r in region_partition_meta:
            rid = int(r.get("region_index", 0) or 0)
            if rid <= 0:
                continue
            mask = lm == rid
            if not np.any(mask):
                continue

            if metric_depth is not None and K is not None:
                region_ctx = {
                    "type": r.get("type", ""),
                    "depth_stats": r.get("depth_stats"),
                    "sigma_scope": getattr(self, "depth_sigma_clip_scope", "mask"),
                }
                try:
                    _depth_stats, _coords_3d, _centroid_2d = self._mask_depth_stats_and_3d(
                        metric_depth,
                        K,
                        mask,
                        use_erosion=True,
                        region_context=region_ctx,
                        label_map=label_map,
                        region_index=rid,
                    )
                    r["depth_stats"] = _depth_stats
                    r["coordinates_3d"] = _coords_3d
                    r["centroid_2d_px"] = _centroid_2d
                except Exception as _e:
                    print(f"  [Regions] depth stats enrichment failed for region_{rid}: {_e}")

            bx = r.get("bbox_px") or [0, 0, w - 1, h - 1]
            x1, y1, x2, y2 = [int(v) for v in bx[:4]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Primary crop (= the crop _label_mask would build internally).
            # We pre-compute it here so Florence-2 can run over all regions
            # in a single batched forward, rather than 44 sequential calls.
            _pad = 4
            ys_m, xs_m = np.where(mask)
            if ys_m.size > 0:
                py1 = max(0, int(ys_m.min()) - _pad)
                py2 = min(h, int(ys_m.max()) + 1 + _pad)
                px1 = max(0, int(xs_m.min()) - _pad)
                px2 = min(w, int(xs_m.max()) + 1 + _pad)
            else:
                px1, py1, px2, py2 = x1, y1, min(w, x2 + 1), min(h, y2 + 1)
            crop = img_bgr[py1:py2, px1:px2].copy()
            m_prim = mask[py1:py2, px1:px2].astype(bool)
            if crop.size == 0 or m_prim.size == 0 or m_prim.sum() < max(int(m_prim.size * 0.05), 1):
                crop = img_bgr[y1:y2 + 1, x1:x2 + 1].copy()
                m_prim = mask[y1:y2 + 1, x1:x2 + 1].astype(bool)
            _outside = ~mask
            bg_fill = (
                img_bgr[_outside].mean(axis=0).astype(np.uint8)
                if _outside.any()
                else np.array([128, 128, 128], dtype=np.uint8)
            )
            primary_crop = crop.copy()
            if primary_crop.size > 0 and m_prim.shape == primary_crop.shape[:2]:
                primary_crop[~m_prim] = bg_fill

            # Vote crops. The legacy code ran 3 RAM++ "vote" crops per region
            # (full-bbox-with-fill, eroded-interior, central-50%-with-fill)
            # PLUS one RAM++ call inside ``_label_mask`` on a 4th crop. On
            # CPU Swin-Large each forward is ~1.4s, so the four crops x 44
            # regions dominated ``enrich_regions``. The user explicitly
            # approved dropping the central-50% crop (it was largely
            # redundant with the full-bbox-with-fill crop — both show the
            # same filled content, just framed differently).
            #
            # We also now reuse the tight ``primary_crop`` as the single
            # RAM++ "primary" input (it's what ``_label_mask`` would have
            # run its own RAM++ call on) and keep the eroded-interior crop
            # as the second vote input. Two RAM++ crops per region (primary
            # + eroded) preserves the vote-aggregation semantics (we still
            # aggregate tags from multiple framings of the same region) and
            # halves the CPU RAM++ work without any accuracy loss.
            _erode_k = max(3, min(15, min(bh, bw) // 8))
            _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_erode_k, _erode_k))
            _eroded = cv2.erode(mask.astype(np.uint8), _kernel, iterations=1).astype(bool)
            vote_eroded = None
            if np.any(_eroded):
                _ys_e, _xs_e = np.where(_eroded)
                _ey1 = max(0, int(_ys_e.min()))
                _ey2 = min(h, int(_ys_e.max()) + 1)
                _ex1 = max(0, int(_xs_e.min()))
                _ex2 = min(w, int(_xs_e.max()) + 1)
                if _ey2 > _ey1 and _ex2 > _ex1:
                    vote_eroded = img_bgr[_ey1:_ey2, _ex1:_ex2].copy()

            ctx.append({
                "r": r,
                "rid": rid,
                "mask": mask,
                "bbox_px": (x1, y1, x2, y2, bw, bh),
                "primary_crop": primary_crop,
                "vote_crops": [c for c in (vote_eroded,) if c is not None and c.size > 0],
            })

        if not ctx:
            return out

        # ------------------------------------------------------------------
        # Pass 2 — Florence-2 batched on MPS.
        # Pass 3 — RAM++ batched on CPU.
        #
        # We DROP the primary RAM++ forward per region. Florence-2
        # already provides the labelling signal via ``f2_label``; the
        # extra RAM++ call on the same crop was only used for vote
        # aggregation, which the single eroded-interior crop now
        # covers alone. The agreement threshold in Pass 4 adapts
        # downward to account for the reduced source count, so
        # aliases still populate. Net effect: RAM++ call count drops
        # from 2·N → 1·N per image (44 calls instead of 88).
        #
        # CONCURRENT mode (F2 on MPS + RAM++ on CPU at the same time)
        # is gated behind ``config.region_enrich_parallel_labellers``
        # and disabled by default. On an 8 GB M2 we measured severe
        # swap-thrash when both models are active: RAM++ slows from
        # 1.4 s/crop to ~30 s/crop because macOS compacts Florence-2's
        # MPS KV cache to disk while RAM++'s CPU BLAS allocates fresh
        # activations, so the *sequential* path is materially faster
        # in the 8 GB regime. Operators on ≥ 16 GB unified memory can
        # opt in.
        # ------------------------------------------------------------------
        rampp_vote_tags: Dict[int, List[List[str]]] = {}
        rampp_sources_per_rid: Dict[int, int] = {}
        rampp_flat_slots: List[Tuple[int, int]] = []
        rampp_flat_crops: List[np.ndarray] = []
        _use_parallel_labellers = bool(
            getattr(self.config, "region_enrich_parallel_labellers", False)
        ) if self.config is not None else False
        rampp_executor: Optional[_ThreadPoolExecutor] = None
        rampp_future = None

        self._ensure_rampp_for_labelling()
        if self.rampp is not None and self.rampp.active:
            for c in ctx:
                rid = c["rid"]
                for vc in c["vote_crops"]:
                    if vc is None or vc.size == 0:
                        continue
                    rampp_flat_crops.append(vc)
                    rampp_flat_slots.append((rid, 1))
                    rampp_sources_per_rid[rid] = rampp_sources_per_rid.get(rid, 0) + 1

            rampp_chunk = int(getattr(self.config, "region_enrich_rampp_batch", 16) or 16)
            if rampp_chunk < 1:
                rampp_chunk = 1

            def _rampp_worker(crops_local: List[np.ndarray], chunk_local: int, tag: str) -> List[Dict[str, Any]]:
                results: List[Dict[str, Any]] = []
                _wt0 = time.perf_counter()
                try:
                    for _ws in range(0, len(crops_local), chunk_local):
                        _we = min(_ws + chunk_local, len(crops_local))
                        results.extend(self.rampp.label_crops(crops_local[_ws:_we]))
                        print(
                            f"  [Regions] RAM++ {_we}/{len(crops_local)}"
                            f" ({time.perf_counter() - _wt0:.1f}s{tag})"
                        )
                except Exception as _we:
                    print(
                        f"  [Regions] batched RAM++ failed{tag} ({_we}); "
                        "falling back per-crop."
                    )
                    results = [self.rampp.label_crop(c) for c in crops_local]
                return results

            if rampp_flat_crops and _use_parallel_labellers:
                print(
                    f"  [Regions] starting RAM++ on background thread over "
                    f"{len(rampp_flat_crops)} vote crops (chunk={rampp_chunk})..."
                )
                rampp_executor = _ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="rampp-bg"
                )
                rampp_future = rampp_executor.submit(
                    _rampp_worker, rampp_flat_crops, rampp_chunk, ", bg thread"
                )

        # Florence-2 batched on MPS (main thread).
        florence_prefill: Dict[int, Tuple[str, str]] = {}
        if self._florence2_label_enabled:
            self._ensure_florence_for_labelling()
        try:
            if (
                self._florence2_label_enabled
                and self.florence2 is not None
                and self.florence2.active
            ):
                primary_crops = [c["primary_crop"] for c in ctx]
                chunk = int(getattr(self.config, "region_enrich_florence_batch", 8) or 8)
                if chunk < 1:
                    chunk = 1
                print(
                    f"  [Regions] batched Florence-2 over {len(primary_crops)} region crops"
                    f" (chunk={chunk})..."
                )
                _t0 = time.perf_counter()
                f2_results: List[Dict[str, Any]] = []
                for _start in range(0, len(primary_crops), chunk):
                    _end = min(_start + chunk, len(primary_crops))
                    f2_results.extend(self.florence2.label_crops(primary_crops[_start:_end]))
                    try:
                        if (
                            getattr(torch.backends, "mps", None) is not None
                            and torch.backends.mps.is_available()
                        ):
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                    print(
                        f"  [Regions] Florence-2 {_end}/{len(primary_crops)}"
                        f" ({time.perf_counter() - _t0:.1f}s)"
                    )
                for c, fr in zip(ctx, f2_results):
                    florence_prefill[c["rid"]] = (
                        str(fr.get("label", "object")).strip().lower() or "object",
                        str(fr.get("caption", "object")),
                    )
        except Exception as _e:
            print(f"  [Regions] batched Florence-2 failed ({_e}); falling back per-region.")
            florence_prefill = {}

        # RAM++ — either join the concurrent future or run serially now.
        if rampp_future is not None:
            try:
                rampp_results = rampp_future.result()
            except Exception as _e:
                print(f"  [Regions] RAM++ background future failed ({_e}); no votes this run.")
                rampp_results = []
            finally:
                if rampp_executor is not None:
                    rampp_executor.shutdown(wait=True)
                    rampp_executor = None
            for (rid, _kind), res in zip(rampp_flat_slots, rampp_results):
                rampp_vote_tags.setdefault(rid, []).append(list(res.get("tags", [])))
        elif rampp_flat_crops and self.rampp is not None and self.rampp.active:
            # Serial path — after Florence-2 completes we have full CPU
            # headroom and no MPS pressure, so RAM++ recovers its
            # ~1.4 s/crop throughput on M2.
            print(
                f"  [Regions] batched RAM++ over {len(rampp_flat_crops)} vote crops on CPU"
                f" (chunk={rampp_chunk})..."
            )
            rampp_results = _rampp_worker(rampp_flat_crops, rampp_chunk, "")
            for (rid, _kind), res in zip(rampp_flat_slots, rampp_results):
                rampp_vote_tags.setdefault(rid, []).append(list(res.get("tags", [])))

        # ------------------------------------------------------------------
        # Pass 4 — per-region fuse. One ``_label_mask(_phase="prefilled")``
        # call per region, but with ZERO model work: it just runs the
        # existing GDINO > F2 > RAM++ priority selection on the prefilled
        # values. Then apply the 3-crop vote aggregation that the legacy
        # loop did inline.
        # ------------------------------------------------------------------
        nreg = len(ctx)
        for i, c in enumerate(ctx, start=1):
            rid = c["rid"]
            x1, y1, _x2, _y2, bw, bh = c["bbox_px"]
            fake_amg = {"label": "object", "bbox": [x1, y1, bw, bh], "gdino_conf": 0.0}
            _pf = florence_prefill.get(rid, ("object", "object"))
            # We no longer run RAM++ on the primary crop, so
            # ``_prefill_rampp`` is always ``None`` for this phase — the
            # default ``rampp_label="object"`` inside ``_label_mask``
            # correctly lets Florence-2 win the priority fight.
            det = self._label_mask(
                img_bgr,
                c["mask"],
                fake_amg,
                label_map=label_map,
                region_index=rid,
                _secondary_phase="prefilled",
                _prefill_florence=_pf,
                _prefill_rampp=None,
            )

            # Aggregate per-region vote tags. With only a single RAM++
            # vote source per region (the eroded-interior crop), the
            # legacy ``>= 2`` agreement rule never fires, so we adapt
            # the threshold to the actual number of sources. When more
            # than one vote crop exists (e.g. a future multi-crop run),
            # the ``>= 2`` rule is restored. Output keys are identical.
            all_tag_votes: Dict[str, int] = {}
            for tags in rampp_vote_tags.get(rid, []):
                for tag in tags:
                    all_tag_votes[tag] = all_tag_votes.get(tag, 0) + 1

            n_sources = int(rampp_sources_per_rid.get(rid, 0))
            _agree_thr = 2 if n_sources >= 2 else 1

            if all_tag_votes:
                _sorted = sorted(all_tag_votes.items(), key=lambda kv: (-kv[1], kv[0]))
                det["rampp_votes"] = {t: v for t, v in _sorted}
                _agreed = [t for t, v in _sorted if v >= _agree_thr]
                if _agreed:
                    det["aliases"] = _agreed[:5]
                    det["rampp_tags"] = _agreed[:_rampp_max]
                    if str(det.get("label", "object")).lower() in ("object", "region", ""):
                        det["label"] = _agreed[0]
                        det["source_model"] = "RAM++_voted"
                        det["canonical_name"] = _agreed[0]
                        det["display_name"] = _agreed[0]

            f2 = str(det.get("florence2_label", "")).strip().lower()
            if f2 and f2 != "object":
                det["category"] = f2
            elif det.get("aliases"):
                det["category"] = det["aliases"][0]
            det["label_sources"] = [s for s in [det.get("source_model")] if s]

            out[rid] = det

            # Progress log every N regions so the user sees motion instead
            # of an apparently-frozen terminal during the 5-min enrichment.
            if (i % 10 == 0) or (i == nreg):
                print(f"  [Regions] enriched {i}/{nreg}")

        return out

    @staticmethod
    def _merge_region_labeller_enrichment(reg: Dict[str, Any], lab: Dict[str, Any]) -> None:
        """Attach labeller outputs to one region meta row (keeps depth `type` separate from semantics)."""
        if not lab:
            return
        reg["semantic_label"] = str(lab.get("label", "")).strip().lower() or ""
        reg["label_confidence"] = lab.get("conf", 0.0)
        reg["segmentation_caption"] = str(lab.get("caption", ""))
        reg["labeller_source"] = str(lab.get("source_model", ""))
        for key in (
            "florence2_label",
            "florence2_caption",
            "rampp_label",
            "rampp_caption",
            "rampp_tags",
            "canonical_name",
            "display_name",
        ):
            if key in lab and lab[key]:
                reg[key] = lab[key]
        if lab.get("rampp_tags"):
            reg["rampp_tags"] = list(lab.get("rampp_tags", []))
        # Ensure canonical_name is always resolved: lab.canonical_name → semantic_label → type
        if not reg.get("canonical_name"):
            reg["canonical_name"] = (
                str(lab.get("canonical_name", "")).strip().lower()
                or str(lab.get("label", "")).strip().lower()
                or str(reg.get("type", "region"))
            )
        # Propagate enrichment extras
        for key in ("aliases", "rampp_votes", "category", "label_sources"):
            if key in lab and lab[key]:
                reg[key] = lab[key]

    def _attach_relations_by_triplets(
        self,
        objects_3d: List[Dict[str, Any]],
        triplets: List[Dict[str, Any]],
        source_name: str,
    ) -> Dict[str, int]:
        """Attach triplets to fused objects using IDs first, then label matching."""
        stats = {
            "input_triplets": int(len(triplets)),
            "attached": 0,
            "subject_id_matched": 0,
            "subject_label_matched": 0,
            "target_id_matched": 0,
            "target_label_matched": 0,
            "external_targets": 0,
            "unmatched_subjects": 0,
        }
        if not triplets:
            return stats

        id_to_obj: Dict[str, Dict[str, Any]] = {}
        for o in objects_3d:
            oid = o.get("id")
            gid = o.get("graph_id")
            if oid is not None:
                id_to_obj[str(oid)] = o
            if gid is not None and str(gid) != str(oid):
                id_to_obj[str(gid)] = o

        def _find_by_label(label: str) -> Optional[Dict[str, Any]]:
            needle = str(label).strip().lower()
            if not needle:
                return None
            for obj in objects_3d:
                if str(obj.get("label", "")).strip().lower() == needle:
                    return obj
            for obj in objects_3d:
                obj_label = str(obj.get("label", "")).lower()
                if needle in obj_label or obj_label in needle:
                    return obj
            return None

        for triplet in triplets:
            source_obj = None
            sub_id = triplet.get("sub_id")
            if sub_id is not None:
                source_obj = id_to_obj.get(str(sub_id))
                if source_obj is not None:
                    stats["subject_id_matched"] += 1
            if source_obj is None:
                source_obj = _find_by_label(triplet.get("sub", ""))
                if source_obj is not None:
                    stats["subject_label_matched"] += 1
            if source_obj is None:
                stats["unmatched_subjects"] += 1
                continue

            target_id = None
            raw_obj_id = triplet.get("obj_id")
            if raw_obj_id is not None:
                target = id_to_obj.get(str(raw_obj_id))
                if target is not None:
                    target_id = target.get("id")
                    stats["target_id_matched"] += 1
            if target_id is None:
                target = _find_by_label(triplet.get("obj", ""))
                if target is not None:
                    target_id = target.get("id")
                    stats["target_label_matched"] += 1
                else:
                    target_label = str(triplet.get("obj", "unknown")).strip().lower() or "unknown"
                    target_id = f"external_{target_label}"
                    stats["external_targets"] += 1

            # Resolve target label + caption for relation enrichment
            _target_obj = id_to_obj.get(str(target_id)) if target_id and not str(target_id).startswith("external_") else None
            if _target_obj is not None:
                _target_label = str(_target_obj.get("label", "unknown"))
                _target_src = _target_obj.get("sources", {})
                _target_caption = (
                    _target_src.get("GroundedSAM2", {}).get("caption")
                    or _target_src.get("RAM++", {}).get("caption")
                    or _target_src.get("Florence2", {}).get("caption")
                    or ""
                )
            else:
                _target_label = str(target_id).replace("external_", "") if target_id else "unknown"
                _target_caption = ""

            relation_entry = {
                "predicate": str(triplet.get("pred", "related_to")),
                "target_id": target_id,
                "target_label": _target_label,
                "target_caption": _target_caption,
            }
            if "score" in triplet:
                relation_entry["score"] = round(float(triplet["score"]), 4)
            if triplet.get("relation_tier"):
                relation_entry["relation_tier"] = str(triplet["relation_tier"])
            if triplet.get("source_layer"):
                relation_entry["source_layer"] = str(triplet["source_layer"])

            source_obj["sources"].setdefault(source_name, {"relations": []})
            source_obj["sources"][source_name]["relations"].append(relation_entry)
            stats["attached"] += 1

        return stats

    def _save_depth_map_image(self, metric_depth: np.ndarray, path: Path) -> None:
        """Save full depth as colormap PNG."""
        d = metric_depth.astype(np.float32)
        d_min, d_max = d.min(), d.max()
        if d_max - d_min < 1e-8:
            vis = np.zeros((*d.shape, 3), dtype=np.uint8)
        else:
            vis = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        self._write_png_intermediate(path, vis)

    @staticmethod
    def _mask_colour(seed: int) -> tuple:
        """Deterministic BGR colour from integer seed."""
        rng = np.random.RandomState(seed)
        r, g, b = rng.randint(60, 230, 3)
        return (int(b), int(g), int(r))  # BGR

    @staticmethod
    def _draw_label(canvas_bgr: np.ndarray, text: str, cx: int, cy: int, mask_area: int) -> None:
        """Draw a label string with a dark pill background at (cx, cy)."""
        if not text:
            return
        # Scale font with mask area (clamp between 0.35 and 0.7)
        scale = float(np.clip(np.sqrt(mask_area) / 250.0, 0.35, 0.70))
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pad = 3
        x0 = max(0, cx - tw // 2 - pad)
        y0 = max(0, cy - th - pad)
        x1 = min(canvas_bgr.shape[1] - 1, cx + tw // 2 + pad)
        y1 = min(canvas_bgr.shape[0] - 1, cy + baseline + pad)
        # Dark semi-transparent pill
        roi = canvas_bgr[y0:y1, x0:x1].astype(np.float32)
        roi[:] = roi * 0.35
        canvas_bgr[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        cv2.putText(canvas_bgr, text, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def _object_visual_cache(
        self,
        objects_3d: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """Phase F — build once, reuse across object visualisations.

        Returns ``None`` if no usable masks are present. Otherwise produces:
          * ``label_map`` (H×W int32) — object index +1 at each mask pixel, 0 at
            background. Built in one ``argmax``-free pass by painting in
            depth-ascending order so nearer objects override farther ones
            (matches the legacy tint overlay ordering).
          * ``palette_lut`` (max_idx+1, 3) uint8 BGR palette.
          * ``binary`` (H×W bool) — any-object mask for fast tint blending.
          * ``contours`` — list of the external contours for every object, in
            the same order as ``objects_3d``.

        Shared between :meth:`_save_labelled_segmentation` and
        :meth:`_save_labelled_tinted_overlay` so masks are resized / contoured
        exactly once per image instead of three times.
        """
        if not objects_3d:
            return None
        h, w = 0, 0
        for obj in objects_3d:
            m = obj.get("_sam2_mask_array")
            if m is not None:
                h, w = np.asarray(m).shape[:2]
                break
        if h == 0 or w == 0:
            return None

        def _obj_z(o: Dict[str, Any]) -> float:
            try:
                return float((o.get("coordinates_3d") or {}).get("z", 0.0) or 0.0)
            except Exception:
                return 0.0

        # Paint far → near so nearer objects win on overlap.
        order = sorted(range(len(objects_3d)), key=lambda i: -_obj_z(objects_3d[i]))
        label_map = np.zeros((h, w), dtype=np.int32)
        resized_masks: Dict[int, np.ndarray] = {}
        for orig_i in order:
            m = objects_3d[orig_i].get("_sam2_mask_array")
            if m is None:
                continue
            m_arr = np.asarray(m)
            if m_arr.shape[:2] != (h, w):
                m_arr = cv2.resize(m_arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            bin_mask = m_arr > 0
            resized_masks[orig_i] = bin_mask
            # +1 so 0 remains the background sentinel.
            label_map[bin_mask] = orig_i + 1

        if not resized_masks:
            return None

        max_idx = max(resized_masks.keys()) + 1  # palette length inc. background
        palette_lut = np.zeros((max_idx + 1, 3), dtype=np.uint8)
        for orig_i in resized_masks:
            colour_bgr = self._mask_colour(orig_i)
            palette_lut[orig_i + 1] = np.array(colour_bgr, dtype=np.uint8)

        contours_per_obj: Dict[int, Any] = {}
        for orig_i, bin_mask in resized_masks.items():
            cnts, _ = cv2.findContours(
                bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_per_obj[orig_i] = cnts

        binary_any = label_map > 0
        return {
            "h": h,
            "w": w,
            "label_map": label_map,
            "palette_lut": palette_lut,
            "binary": binary_any,
            "contours": contours_per_obj,
            "masks": resized_masks,
        }

    def _save_labelled_segmentation(
        self,
        objects_3d: List[Dict],
        path: Path,
    ) -> None:
        """
        Coloured segment map (one colour per object) with label text at each
        mask centroid. Uses _sam2_mask_array from objects_3d (still present
        before strip). Falls back to bbox if centroid unavailable.

        Phase F — uses the shared :meth:`_object_visual_cache` so masks are
        resized and contoured once, and the palette fill is an O(N) fancy
        index via the palette LUT.
        """
        cache = self._object_visual_cache(objects_3d)
        if cache is None:
            return
        h, w = cache["h"], cache["w"]
        label_map = cache["label_map"]
        palette_lut = cache["palette_lut"]

        # O(N) palette fill — replaces the per-object ``canvas[bin_mask] = …``
        # loop with a single gather against the label map.
        canvas = palette_lut[label_map]

        # 2px white outline around every object; contours were computed once
        # in the cache, so this is just the drawing step.
        for cnts in cache["contours"].values():
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 2)

        for i, obj in enumerate(objects_3d):
            if i not in cache["masks"]:
                continue
            label = str(obj.get("canonical_name", obj.get("label", "object")))
            mc = obj.get("mask_centroid_2d")
            bbox = obj.get("bbox", [0, 0, 0, 0])
            cx = int(mc[0]) if mc and len(mc) == 2 else (int(bbox[0]) + int(bbox[2])) // 2
            cy = int(mc[1]) if mc and len(mc) == 2 else (int(bbox[1]) + int(bbox[3])) // 2
            area = int(np.sum(cache["masks"][i]))
            self._draw_label(canvas, label, cx, cy, area)

        self._write_png(path, canvas)

    def _save_labelled_tinted_overlay(
        self,
        objects_3d: List[Dict],
        image_rgb: np.ndarray,
        path: Path,
        alpha: float = 0.45,
    ) -> None:
        """
        Original photo with each mask as a semi-transparent colour tint and
        label text drawn at the mask centroid.
        """
        if not objects_3d or image_rgb is None:
            return
        cache = self._object_visual_cache(objects_3d)
        if cache is None:
            return
        h, w = cache["h"], cache["w"]
        # Source image may be at a different resolution than the masks.
        if image_rgb.shape[:2] != (h, w):
            image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)

        # Phase F — build a fully-tinted canvas via palette LUT, then blend it
        # with the source image inside the object pixels only. Replaces the
        # per-object ``out[bin_mask] = out[bin_mask] * (1-α) + colour * α``
        # loop with one vectorised alpha blend over the combined mask.
        palette_lut_rgb = cache["palette_lut"][:, ::-1]  # BGR → RGB
        tint_rgb = palette_lut_rgb[cache["label_map"]].astype(np.float32)
        out = image_rgb.astype(np.float32)
        binary_any = cache["binary"]
        out[binary_any] = out[binary_any] * (1.0 - alpha) + tint_rgb[binary_any] * alpha
        out_bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        for cnts in cache["contours"].values():
            cv2.drawContours(out_bgr, cnts, -1, (255, 255, 255), 2)

        for i, obj in enumerate(objects_3d):
            if i not in cache["masks"]:
                continue
            label = str(obj.get("canonical_name", obj.get("label", "object")))
            mc = obj.get("mask_centroid_2d")
            bbox = obj.get("bbox", [0, 0, 0, 0])
            cx = int(mc[0]) if mc and len(mc) == 2 else (int(bbox[0]) + int(bbox[2])) // 2
            cy = int(mc[1]) if mc and len(mc) == 2 else (int(bbox[1]) + int(bbox[3])) // 2
            area = int(np.sum(cache["masks"][i]))
            self._draw_label(out_bgr, label, cx, cy, area)

        self._write_png(path, out_bgr)

    def _save_sam2_outputs(
        self,
        amg_masks: List[Dict[str, Any]],
        h: int,
        w: int,
        out_dir: Path,
        path_stem: str,
        image_path: str,
        timestamp: str,
        image_rgb: np.ndarray = None,
    ) -> Dict[str, str]:
        """
        Save SAM2 outputs independently of depth-mask outputs.
        Returns relative paths from out_dir for metadata wiring.
        """
        sam2_dir = out_dir
        # Placeholder segmentation saved before labelling; overwritten after Stage 4
        # with the labelled version. We still record its path here for metadata.
        return {
            "sam2_segmentation_image_path": f"scene_graph/{path_stem}_sam2_segmentation.png",
            "sam2_tinted_overlay_image_path": f"scene_graph/{path_stem}_sam2_tinted_overlay.png",
        }

    def _save_depth_mask_mapping_image(
        self,
        metric_depth: np.ndarray,
        matched_objects: List[Dict[str, Any]],
        path: Path,
    ) -> None:
        """Save depth colormap only where matched masks are; rest black."""
        h, w = metric_depth.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)
        for obj in matched_objects:
            m = obj.get("mask")
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask |= (m > 0)
        d = metric_depth.astype(np.float32)
        if np.any(combined_mask):
            d_min, d_max = float(d[combined_mask].min()), float(d[combined_mask].max())
        else:
            d_min, d_max = 0.0, 1.0
        if not np.any(combined_mask):
            vis = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            if d_max - d_min < 1e-8:
                d_max = d_min + 1.0
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            norm = np.clip(norm, 0, 255)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            vis[combined_mask] = colored[combined_mask]
        cv2.imwrite(str(path), vis)

    def _build_depth_mask_json(
        self,
        image_path: str,
        path_stem: str,
        timestamp: str,
        image_size: List[int],
        matching_mode: str,
        depth_map_path: str,
        depth_map_image_path: str,
        depth_global_min: float,
        depth_global_max: float,
        depth_global_mean: float,
        segmentation_map_image_path: str,
        num_auto_masks: int,
        mapping_image_path: str,
        objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the depth+mask details JSON structure."""
        return {
            "metadata": {
                "image_path": image_path,
                "image_stem": path_stem,
                "timestamp": timestamp,
                "image_size": image_size,
                "matching_mode": matching_mode,
            },
            "depth": {
                "depth_map_path": depth_map_path,
                "depth_map_image_path": depth_map_image_path,
                "depth_min": depth_global_min,
                "depth_max": depth_global_max,
                "depth_mean": depth_global_mean,
            },
            "segmentation": {
                "model": "SAM2",
                "mode": "automatic_mask_generator",
                "segmentation_map_image_path": segmentation_map_image_path,
                "num_auto_masks": num_auto_masks,
                "match_strategy": "iou_with_detection_bbox",
            },
            "depth_mask": {
                "mapping_image_path": mapping_image_path,
                "objects": objects,
            },
        }

    # -------------------------------------------------------------------------
    # Path hypotheses (single image) — region/object/mask path generation
    # -------------------------------------------------------------------------

    @staticmethod
    def _display_name_for_object(obj: Dict[str, Any]) -> str:
        for k in ("canonical_name", "name", "label"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return str(obj.get("id", "object"))

    @staticmethod
    def _display_name_for_region(region: Dict[str, Any]) -> str:
        for k in ("semantic_label", "id"):
            v = region.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return str(region.get("id", "region"))

    @staticmethod
    def _display_label(name: str, entity_id: str) -> str:
        name = (name or "").strip()
        entity_id = (entity_id or "").strip()
        if name and entity_id and name != entity_id:
            return f"{name} ({entity_id})"
        return name or entity_id or "entity"

    @staticmethod
    def _angle_deg(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        # angle at b for points a-b-c
        bax, bay = float(a[0] - b[0]), float(a[1] - b[1])
        bcx, bcy = float(c[0] - b[0]), float(c[1] - b[1])
        na = math.hypot(bax, bay)
        nc = math.hypot(bcx, bcy)
        if na < 1e-6 or nc < 1e-6:
            return 0.0
        dot = bax * bcx + bay * bcy
        cosv = max(-1.0, min(1.0, dot / (na * nc)))
        return float(math.degrees(math.acos(cosv)))

    @staticmethod
    def _sample_line(p0: Tuple[int, int], p1: Tuple[int, int], n: int = 50) -> List[Tuple[int, int]]:
        x0, y0 = p0
        x1, y1 = p1
        pts: List[Tuple[int, int]] = []
        if n <= 1:
            return [(int(x0), int(y0)), (int(x1), int(y1))]
        for i in range(n):
            t = i / (n - 1)
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            pts.append((x, y))
        return pts

    @staticmethod
    def _dilate_mask(mask: np.ndarray, r: int = 2) -> np.ndarray:
        if r <= 0:
            return mask.astype(bool)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = mask.astype(np.uint8) * 255
        d = cv2.dilate(m, k, iterations=1)
        return (d > 0)

    def _region_portal_point(
        self,
        label_map: np.ndarray,
        region_by_id: Dict[str, Dict[str, Any]],
        a_id: str,
        b_id: str,
        dilation_px: int = 2,
    ) -> Tuple[int, int]:
        # Portal = centroid of shared border pixels (dilated overlap), fallback to mid-point of centroids.
        ra = region_by_id.get(a_id) or {}
        rb = region_by_id.get(b_id) or {}
        ia = int(ra.get("region_index", 0) or 0)
        ib = int(rb.get("region_index", 0) or 0)
        if ia <= 0 or ib <= 0:
            ca = ra.get("centroid_2d_px", [0, 0])
            cb = rb.get("centroid_2d_px", [0, 0])
            return (int((ca[0] + cb[0]) / 2), int((ca[1] + cb[1]) / 2))
        m = np.asarray(label_map, dtype=np.int32)
        ma = (m == ia)
        mb = (m == ib)
        da = self._dilate_mask(ma, r=dilation_px)
        db = self._dilate_mask(mb, r=dilation_px)
        inter = da & db
        ys, xs = np.where(inter)
        if len(xs) == 0:
            ca = ra.get("centroid_2d_px", [0, 0])
            cb = rb.get("centroid_2d_px", [0, 0])
            return (int((ca[0] + cb[0]) / 2), int((ca[1] + cb[1]) / 2))
        return (int(np.mean(xs)), int(np.mean(ys)))

    @staticmethod
    def _build_region_graph(region_adjacency: Dict[str, Any]) -> Dict[str, List[Tuple[str, float, Dict[str, Any]]]]:
        # adjacency list: node -> [(nbr, weight, edge_meta), ...]
        g: Dict[str, List[Tuple[str, float, Dict[str, Any]]]] = {}
        for e in list((region_adjacency or {}).get("edges", []) or []):
            a = str(e.get("region_a", "")).strip()
            b = str(e.get("region_b", "")).strip()
            if not a or not b:
                continue
            border = float(e.get("shared_border_px", 1) or 1)
            dd = float(e.get("depth_delta_m", 0.0) or 0.0)
            # Prefer strong borders and low depth delta.
            w = (1.0 / max(border, 1.0)) + 0.001 * abs(dd)
            g.setdefault(a, []).append((b, w, e))
            g.setdefault(b, []).append((a, w, e))
        return g

    @staticmethod
    def _dijkstra_path(
        g: Dict[str, List[Tuple[str, float, Dict[str, Any]]]],
        start: str,
        goal: str,
        edge_penalty: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[List[str], float]:
        import heapq

        start = str(start)
        goal = str(goal)
        if start == goal:
            return [start], 0.0
        edge_penalty = edge_penalty or {}
        pq: List[Tuple[float, str]] = [(0.0, start)]
        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, str] = {}
        seen: set[str] = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == goal:
                break
            for v, w, _meta in g.get(u, []):
                ep = edge_penalty.get((u, v), 0.0) + edge_penalty.get((v, u), 0.0)
                nd = d + w + ep
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if goal not in dist:
            return [], 1e18
        # reconstruct
        path = [goal]
        cur = goal
        while cur in prev:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path, float(dist[goal])

    def _k_region_paths(
        self,
        region_adjacency: Dict[str, Any],
        start_region: str,
        goal_region: str,
        k: int = 3,
    ) -> List[List[str]]:
        g = self._build_region_graph(region_adjacency)
        paths: List[List[str]] = []
        penalty: Dict[Tuple[str, str], float] = {}
        for _i in range(max(1, int(k))):
            p, _cost = self._dijkstra_path(g, start_region, goal_region, edge_penalty=penalty)
            if not p or p in paths:
                break
            paths.append(p)
            # Penalize edges in this path to encourage alternatives.
            for a, b in zip(p, p[1:]):
                penalty[(a, b)] = penalty.get((a, b), 0.0) + 0.05
        return paths

    def _polyline_is_valid(
        self,
        pts: List[Tuple[int, int]],
        w: int,
        h: int,
        feasible: Optional[np.ndarray] = None,
        obstacles: Optional[np.ndarray] = None,
        invalid_ratio_max: float = 0.05,
        max_turn_deg: float = 70.0,
    ) -> Dict[str, Any]:
        if not pts:
            return {"ok": False, "reason": "empty_polyline"}
        out_of_bounds = 0
        invalid = 0
        total = 0
        # check segments at sampled resolution
        for p0, p1 in zip(pts, pts[1:]):
            seg = self._sample_line(p0, p1, n=30)
            for x, y in seg:
                total += 1
                if x < 0 or y < 0 or x >= w or y >= h:
                    out_of_bounds += 1
                    continue
                if feasible is not None and not bool(feasible[y, x]):
                    invalid += 1
                    continue
                if obstacles is not None and bool(obstacles[y, x]):
                    invalid += 1
                    continue
        # curvature check
        max_angle = 0.0
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            ang = self._angle_deg(a, b, c)
            max_angle = max(max_angle, ang)
        invalid_ratio = (invalid / max(1, total))
        if out_of_bounds > 0:
            return {
                "ok": False,
                "reason": "out_of_bounds",
                "out_of_bounds_samples": out_of_bounds,
                "invalid_ratio": invalid_ratio,
                "max_turn_deg": max_angle,
            }
        if invalid_ratio > float(invalid_ratio_max):
            return {
                "ok": False,
                "reason": "invalid_pixel_crossing",
                "invalid_ratio": invalid_ratio,
                "max_turn_deg": max_angle,
            }
        if max_angle > float(max_turn_deg) + 1e-6:
            return {
                "ok": False,
                "reason": "turn_too_sharp",
                "invalid_ratio": invalid_ratio,
                "max_turn_deg": max_angle,
            }
        return {
            "ok": True,
            "invalid_ratio": invalid_ratio,
            "max_turn_deg": max_angle,
        }

    @staticmethod
    def _tapered_polyline_draw(
        img_bgr: np.ndarray,
        pts: List[Tuple[int, int]],
        color_bgr: Tuple[int, int, int],
        start_w: int,
        end_w: int,
        alpha_start: float,
        alpha_end: float,
        alpha_scale: float = 1.0,
        depth_values: Optional[List[float]] = None,
    ) -> None:
        from scene_understanding.pathing.path_canvas import tapered_polyline_draw

        tapered_polyline_draw(
            img_bgr,
            pts,
            color_bgr,
            start_w,
            end_w,
            alpha_start,
            alpha_end,
            alpha_scale=alpha_scale,
            depth_values=depth_values,
        )

    @staticmethod
    def _clamp_polyline_to_image(pts: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for xy in pts:
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            x = int(max(0, min(w - 1, int(round(float(xy[0]))))))
            y = int(max(0, min(h - 1, int(round(float(xy[1]))))))
            out.append((x, y))
        return out

    @staticmethod
    def _resolve_trajectory_polyline_pixels(
        path: Dict[str, Any], w: int, h: int, prefer_geodesic: bool
    ) -> Tuple[List[Tuple[int, int]], str]:
        """
        Choose visualization polyline: geodesic when present and preferred, else polyline_2d.
        Returns (pixel points, source key for manifest).
        """
        raw: Optional[List[Any]] = None
        source = "none"
        if prefer_geodesic:
            g = path.get("polyline_geodesic_2d")
            if isinstance(g, list) and len(g) >= 2:
                raw = g
                source = "polyline_geodesic_2d"
        if raw is None:
            p2 = path.get("polyline_2d") or []
            if isinstance(p2, list) and len(p2) >= 2:
                raw = p2
                source = "polyline_2d"
        if not raw:
            return [], "none"
        as_floats: List[Tuple[float, ...]] = []
        for xy in raw:
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                as_floats.append((float(xy[0]), float(xy[1])))
        if len(as_floats) < 2:
            return [], "none"
        return SceneUnderstandingPipeline._clamp_polyline_to_image(as_floats, w, h), source

    @staticmethod
    def _polyline_end_segment_for_arrow(pts: List[Tuple[int, int]], min_len_sq: int = 9) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return (p_from, p_to) along the polyline for an arrow tangent to the path end."""
        if len(pts) < 2:
            return None
        x1, y1 = pts[-1]
        for i in range(len(pts) - 2, -1, -1):
            x0, y0 = pts[i]
            d2 = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)
            if d2 >= min_len_sq:
                return (x0, y0), (x1, y1)
        return None

    def _draw_ink_trajectory_tapered_end_arrow(
        self,
        img_bgr: np.ndarray,
        pts: List[Tuple[int, int]],
        ink_bgr: Tuple[int, int, int],
        start_w: int,
        end_w: int,
        alpha_start: float,
        alpha_end: float,
        arrow_thickness: int,
        tip_length: float,
    ) -> None:
        if img_bgr is None or len(pts) < 2:
            return
        ib = (int(ink_bgr[0]), int(ink_bgr[1]), int(ink_bgr[2]))
        self._tapered_polyline_draw(img_bgr, pts, ib, start_w, end_w, alpha_start, alpha_end, 1.0)
        seg = self._polyline_end_segment_for_arrow(pts)
        if seg is not None:
            (x0, y0), (x1, y1) = seg
            cv2.arrowedLine(
                img_bgr,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                ib,
                max(1, int(arrow_thickness)),
                cv2.LINE_AA,
                tipLength=float(tip_length),
            )

    def _draw_ink_trajectory_phase2(
        self,
        img_bgr: np.ndarray,
        pts: List[Tuple[int, int]],
        ink_bgr: Tuple[int, int, int],
        start_w: int,
        end_w: int,
        alpha_start: float,
        alpha_end: float,
        arrow_thickness: int,
        tip_length: float,
        cfg: Optional[Any],
    ) -> None:
        """Primary ink stroke with optional halo outline (Phase 2)."""
        if img_bgr is None or len(pts) < 2:
            return
        ib = (int(ink_bgr[0]), int(ink_bgr[1]), int(ink_bgr[2]))
        if cfg and bool(getattr(cfg, "path_v2_outline_enabled", False)):
            pad = max(1, int(getattr(cfg, "path_v2_outline_pad_px", 3)))
            ob = getattr(cfg, "path_v2_outline_bgr", (250, 250, 250))
            obt = (int(ob[0]), int(ob[1]), int(ob[2]))
            self._tapered_polyline_draw(
                img_bgr,
                pts,
                obt,
                start_w + pad,
                max(1, end_w + pad),
                min(0.99, alpha_start * 1.08),
                min(0.99, alpha_end * 1.08),
                1.0,
            )
        self._draw_ink_trajectory_tapered_end_arrow(
            img_bgr, pts, ib, start_w, end_w, alpha_start, alpha_end, arrow_thickness, tip_length
        )

    @staticmethod
    def _polyline_pixels_from_raw_list(raw: Any, w: int, h: int) -> List[Tuple[int, int]]:
        if not isinstance(raw, list) or len(raw) < 2:
            return []
        as_floats: List[Tuple[float, float]] = []
        for xy in raw:
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                as_floats.append((float(xy[0]), float(xy[1])))
        if len(as_floats) < 2:
            return []
        return SceneUnderstandingPipeline._clamp_polyline_to_image(as_floats, w, h)

    def _build_v2_draw_entries(
        self,
        plist_draw_ordered: List[Dict[str, Any]],
        w: int,
        h: int,
        prefer_geodesic: bool,
        n_res: int,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], List[Tuple[int, int]], str]]]:
        manifest_paths: List[Dict[str, Any]] = []
        draw_list: List[Tuple[Dict[str, Any], List[Tuple[int, int]], str]] = []
        for p in plist_draw_ordered:
            pid = str(p.get("path_id", ""))
            pts, src = self._resolve_trajectory_polyline_pixels(p, w, h, prefer_geodesic)
            if len(pts) < 2:
                manifest_paths.append(
                    {
                        "path_id": pid,
                        "polyline_source": src,
                        "point_count": 0,
                        "drawn": False,
                        "skip_reason": "too_few_points",
                    }
                )
                continue
            pts_r = self._resample_polyline(pts, min(n_res, max(24, len(pts) * 4)))
            if len(pts_r) < 2:
                pts_r = pts
            draw_list.append((p, pts_r, src))
            manifest_paths.append(
                {
                    "path_id": pid,
                    "path_num": int(p.get("path_num", 0) or 0),
                    "polyline_source": src,
                    "point_count": len(pts_r),
                    "drawn": True,
                    "suppressed": bool(p.get("suppressed", False)),
                    "suppressed_reason": str(p.get("suppressed_reason", "")),
                    "overall_confidence": float((p.get("scores") or {}).get("overall_confidence", 0.0)),
                }
            )
        return manifest_paths, draw_list

    def _draw_v3_polyline2d_faint_layer(
        self,
        canvas: np.ndarray,
        plist: List[Dict[str, Any]],
        color_bgr: Tuple[int, int, int],
        alpha_scale: float,
        sw: int,
        ew: int,
        a0: float,
        a1: float,
        w: int,
        h: int,
        n_res: int,
    ) -> int:
        """Draw polyline_2d only, faint. Returns count of paths drawn."""
        asc = max(0.05, min(1.0, float(alpha_scale)))
        cb = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        ordered = sorted(
            plist,
            key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=False,
        )
        n = 0
        for p in ordered:
            pts = self._polyline_pixels_from_raw_list(p.get("polyline_2d"), w, h)
            if len(pts) < 2:
                continue
            pts_r = self._resample_polyline(pts, min(n_res, max(24, len(pts) * 4)))
            if len(pts_r) < 2:
                pts_r = pts
            self._tapered_polyline_draw(canvas, pts_r, cb, sw, ew, a0, a1, asc)
            n += 1
        return n

    def _draw_v3_resolved_faint_layer(
        self,
        canvas: np.ndarray,
        plist: List[Dict[str, Any]],
        prefer_geodesic: bool,
        color_bgr: Tuple[int, int, int],
        alpha_scale: float,
        sw: int,
        ew: int,
        a0: float,
        a1: float,
        w: int,
        h: int,
        n_res: int,
    ) -> int:
        asc = max(0.05, min(1.0, float(alpha_scale)))
        cb = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        ordered = sorted(
            plist,
            key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=False,
        )
        n = 0
        for p in ordered:
            pts, _src = self._resolve_trajectory_polyline_pixels(p, w, h, prefer_geodesic)
            if len(pts) < 2:
                continue
            pts_r = self._resample_polyline(pts, min(n_res, max(24, len(pts) * 4)))
            if len(pts_r) < 2:
                pts_r = pts
            self._tapered_polyline_draw(canvas, pts_r, cb, sw, ew, a0, a1, asc)
            n += 1
        return n

    def _export_trajectory_viz_v2(
        self,
        img_bgr: np.ndarray,
        paths: List[Dict[str, Any]],
        paths_recommended: List[Dict[str, Any]],
        lm: np.ndarray,
        objs: List[Dict[str, Any]],
        w: int,
        h: int,
        paths_root_dir: Path,
        path_stem: str,
        track_dir_name: str,
        cfg: Optional[Any],
        paths_sorted: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Trajectory v2 maps + manifest (Phase 1/2; separate from legacy context PNGs)."""
        if not cfg or not bool(getattr(cfg, "path_v2_visualization_enabled", True)):
            return {}
        sub_name = str(getattr(cfg, "path_v2_output_subdir", "trajectory_viz_v2")).strip() or "trajectory_viz_v2"
        v2_dir = paths_root_dir / sub_name
        v2_dir.mkdir(parents=True, exist_ok=True)

        scope = str(getattr(cfg, "path_v2_draw_scope", "all_valid")).strip().lower()
        if scope == "all_recommended":
            plist_all = [dict(p) for p in (paths_recommended or [])]
        else:
            plist_all = [dict(p) for p in (paths or [])]

        max_paths = max(1, int(getattr(cfg, "path_v2_max_paths", 500)))
        truncated = len(plist_all) > max_paths
        plist_sorted_desc = sorted(
            plist_all,
            key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=True,
        )
        if truncated:
            plist_cap = plist_sorted_desc[:max_paths]
        else:
            plist_cap = plist_sorted_desc
        plist_draw = sorted(
            plist_cap,
            key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=False,
        )
        cap_ids = {str(p.get("path_id", "")).strip() for p in plist_cap if str(p.get("path_id", "")).strip()}

        prefer_geo = bool(getattr(cfg, "path_v2_prefer_geodesic", True))
        n_res = max(24, int(getattr(cfg, "path_v2_resample_points", 96)))
        ink = getattr(cfg, "path_v2_ink_bgr", (15, 15, 15))
        if not (isinstance(ink, (list, tuple)) and len(ink) >= 3):
            ink = (15, 15, 15)
        ink_t = (int(ink[0]), int(ink[1]), int(ink[2]))
        sw = int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8
        ew = int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2
        a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95
        a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35
        ath = int(getattr(cfg, "path_v2_arrow_thickness_px", 2)) if cfg else 2
        tip = float(getattr(cfg, "path_v2_arrow_tip_length", 0.22)) if cfg else 0.22
        max_boxes = int(getattr(cfg, "path_v2_context_max_boxes", 50)) if cfg else 50

        legacy_ink = getattr(cfg, "path_v2_legacy_ink_bgr", (160, 160, 160))
        legacy_ink_t = (
            int(legacy_ink[0]),
            int(legacy_ink[1]),
            int(legacy_ink[2]),
        )
        legacy_alpha_sc = float(getattr(cfg, "path_v2_legacy_alpha_scale", 0.42))
        legacy_alpha_sc = max(0.05, min(1.0, legacy_alpha_sc))
        alt_ink = getattr(cfg, "path_v2_alternate_ink_bgr", (120, 120, 120))
        alt_ink_t = (int(alt_ink[0]), int(alt_ink[1]), int(alt_ink[2]))
        alt_alpha_sc = float(getattr(cfg, "path_v2_alternate_alpha_scale", 0.36))
        alt_alpha_sc = max(0.05, min(1.0, alt_alpha_sc))

        manifest_paths, draw_list = self._build_v2_draw_entries(plist_draw, w, h, prefer_geo, n_res)

        rel_base = f"scene_graph/{track_dir_name}/{path_stem}_paths/{sub_name}"

        def _render_draw_list(canvas: np.ndarray, dlist: List[Tuple[Dict[str, Any], List[Tuple[int, int]], str]], use_outline: bool) -> None:
            for _p, pts_r, _src in dlist:
                if use_outline:
                    self._draw_ink_trajectory_phase2(
                        canvas, pts_r, ink_t, sw, ew, a0, a1, ath, tip, cfg
                    )
                else:
                    self._draw_ink_trajectory_tapered_end_arrow(
                        canvas, pts_r, ink_t, sw, ew, a0, a1, ath, tip
                    )

        canvas_plain = img_bgr.copy()
        _render_draw_list(canvas_plain, draw_list, False)
        cv2.imwrite(str(v2_dir / "traj_v2_all_ink.png"), canvas_plain)

        canvas_ctx = img_bgr.copy()
        self._draw_regions_contours_bgr(canvas_ctx, lm)
        self._draw_objects_boxes_bgr(canvas_ctx, objs, max_boxes=max_boxes)
        _render_draw_list(canvas_ctx, draw_list, False)
        cv2.imwrite(str(v2_dir / "traj_v2_all_ink_context.png"), canvas_ctx)

        out_meta: Dict[str, str] = {
            "trajectory_viz_v2_all_ink_image": f"{rel_base}/traj_v2_all_ink.png",
            "trajectory_viz_v2_all_ink_context_image": f"{rel_base}/traj_v2_all_ink_context.png",
        }

        manifest_files: Dict[str, str] = {
            "all_ink": f"{rel_base}/traj_v2_all_ink.png",
            "all_ink_context": f"{rel_base}/traj_v2_all_ink_context.png",
        }
        phase2_summary: Dict[str, Any] = {}

        export_topk = bool(getattr(cfg, "path_v2_export_topk_summary", True)) if cfg else True
        topk_n = max(1, int(getattr(cfg, "path_v2_topk", 8))) if cfg else 8
        if export_topk:
            plist_topk_cap = plist_sorted_desc[: min(topk_n, len(plist_sorted_desc))]
            plist_topk_draw = sorted(
                plist_topk_cap,
                key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
                reverse=False,
            )
            _, draw_topk = self._build_v2_draw_entries(plist_topk_draw, w, h, prefer_geo, n_res)
            canvas_top = img_bgr.copy()
            self._draw_regions_contours_bgr(canvas_top, lm)
            self._draw_objects_boxes_bgr(canvas_top, objs, max_boxes=max_boxes)
            use_ol = bool(getattr(cfg, "path_v2_outline_enabled", False)) if cfg else False
            _render_draw_list(canvas_top, draw_topk, use_ol)
            cv2.imwrite(str(v2_dir / "traj_v2_topK_ink_context.png"), canvas_top)
            out_meta["trajectory_viz_v2_topk_ink_context_image"] = f"{rel_base}/traj_v2_topK_ink_context.png"
            manifest_files["topk_ink_context"] = out_meta["trajectory_viz_v2_topk_ink_context_image"]
            phase2_summary["topk"] = topk_n

        export_geo = bool(getattr(cfg, "path_v2_export_geodesic_compare", True)) if cfg else True
        geo_compare_count = 0
        if export_geo:
            cgeo = img_bgr.copy()
            use_ol = bool(getattr(cfg, "path_v2_outline_enabled", False)) if cfg else False
            for p in plist_draw:
                lr = p.get("polyline_2d")
                gr = p.get("polyline_geodesic_2d")
                if not (isinstance(lr, list) and len(lr) >= 2 and isinstance(gr, list) and len(gr) >= 2):
                    continue
                pts_l = self._polyline_pixels_from_raw_list(lr, w, h)
                pts_g = self._polyline_pixels_from_raw_list(gr, w, h)
                if len(pts_l) < 2 or len(pts_g) < 2:
                    continue
                rl = self._resample_polyline(pts_l, min(n_res, max(24, len(pts_l) * 4)))
                rg = self._resample_polyline(pts_g, min(n_res, max(24, len(pts_g) * 4)))
                if len(rl) < 2:
                    rl = pts_l
                if len(rg) < 2:
                    rg = pts_g
                geo_compare_count += 1
                self._tapered_polyline_draw(cgeo, rl, legacy_ink_t, sw, ew, a0, a1, legacy_alpha_sc)
                if use_ol:
                    self._draw_ink_trajectory_phase2(cgeo, rg, ink_t, sw, ew, a0, a1, ath, tip, cfg)
                else:
                    self._draw_ink_trajectory_tapered_end_arrow(cgeo, rg, ink_t, sw, ew, a0, a1, ath, tip)
            cv2.imwrite(str(v2_dir / "traj_v2_geodesic_compare.png"), cgeo)
            out_meta["trajectory_viz_v2_geodesic_compare_image"] = f"{rel_base}/traj_v2_geodesic_compare.png"
            manifest_files["geodesic_compare"] = out_meta["trajectory_viz_v2_geodesic_compare_image"]
            phase2_summary["geodesic_compare_path_count"] = geo_compare_count

        export_alt = bool(getattr(cfg, "path_v2_export_alternates_faint", True)) if cfg else True
        if export_alt:
            calt = img_bgr.copy()
            for p in plist_draw:
                for alt in (p.get("polyline_geodesic_alternates_2d") or []):
                    pts_alt = self._polyline_pixels_from_raw_list(alt, w, h)
                    if len(pts_alt) < 2:
                        continue
                    ra = self._resample_polyline(pts_alt, min(n_res, max(24, len(pts_alt) * 4)))
                    if len(ra) < 2:
                        ra = pts_alt
                    self._tapered_polyline_draw(calt, ra, alt_ink_t, sw, ew, a0, a1, alt_alpha_sc)
            for _p, pts_r, _src in draw_list:
                self._draw_ink_trajectory_tapered_end_arrow(calt, pts_r, ink_t, sw, ew, a0, a1, ath, tip)
            cv2.imwrite(str(v2_dir / "traj_v2_alternates_faint.png"), calt)
            out_meta["trajectory_viz_v2_alternates_faint_image"] = f"{rel_base}/traj_v2_alternates_faint.png"
            manifest_files["alternates_faint"] = out_meta["trajectory_viz_v2_alternates_faint_image"]

        phase3_summary: Dict[str, Any] = {"enabled": False}
        if paths_sorted is not None and cfg and bool(getattr(cfg, "path_v3_visualization_enabled", True)):
            p3_any = False
            p3u = bool(getattr(cfg, "path_v3_export_underlay_stack", True)) if cfg else True
            p3uc = bool(getattr(cfg, "path_v3_export_underlay_stack_context", True)) if cfg else True
            all_poly = bool(getattr(cfg, "path_v3_all_polyline_2d_only", True)) if cfg else True
            all_a = float(getattr(cfg, "path_v3_all_alpha_scale", 0.12)) if cfg else 0.12
            all_c = getattr(cfg, "path_v3_all_color_bgr", (200, 200, 200))
            all_ct = (int(all_c[0]), int(all_c[1]), int(all_c[2]))
            rec_a = float(getattr(cfg, "path_v3_recommended_alpha_scale", 0.30)) if cfg else 0.30
            rec_c = getattr(cfg, "path_v3_recommended_color_bgr", (110, 110, 110))
            rec_ct = (int(rec_c[0]), int(rec_c[1]), int(rec_c[2]))
            fin_n = max(1, int(getattr(cfg, "path_v3_final_top", 12))) if cfg else 12
            sup_a = float(getattr(cfg, "path_v3_suppressed_alpha_scale", 0.22)) if cfg else 0.22
            sup_c = getattr(cfg, "path_v3_suppressed_color_bgr", (80, 80, 220))
            sup_ct = (int(sup_c[0]), int(sup_c[1]), int(sup_c[2]))

            def _underlay_stack(with_ctx: bool) -> Tuple[np.ndarray, Dict[str, int]]:
                c = img_bgr.copy()
                if with_ctx:
                    self._draw_regions_contours_bgr(c, lm)
                    self._draw_objects_boxes_bgr(c, objs, max_boxes=max_boxes)
                if all_poly:
                    na = self._draw_v3_polyline2d_faint_layer(
                        c, plist_cap, all_ct, all_a, sw, ew, a0, a1, w, h, n_res
                    )
                else:
                    na = self._draw_v3_resolved_faint_layer(
                        c, plist_cap, prefer_geo, all_ct, all_a, sw, ew, a0, a1, w, h, n_res
                    )
                rec_pl = [p for p in paths_recommended if str(p.get("path_id", "")).strip() in cap_ids]
                nr = self._draw_v3_resolved_faint_layer(
                    c, rec_pl, prefer_geo, rec_ct, rec_a, sw, ew, a0, a1, w, h, n_res
                )
                top_pl = paths_sorted[: min(fin_n, len(paths_sorted))]
                top_draw = sorted(
                    top_pl,
                    key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
                    reverse=False,
                )
                nf = 0
                for p in top_draw:
                    pts, _src = self._resolve_trajectory_polyline_pixels(p, w, h, prefer_geo)
                    if len(pts) < 2:
                        continue
                    pts_r = self._resample_polyline(pts, min(n_res, max(24, len(pts) * 4)))
                    if len(pts_r) < 2:
                        pts_r = pts
                    self._draw_ink_trajectory_tapered_end_arrow(c, pts_r, ink_t, sw, ew, a0, a1, ath, tip)
                    nf += 1
                return c, {"all_candidates_drawn": na, "recommended_drawn": nr, "final_ink_drawn": nf}

            if p3u:
                p3_any = True
                u_plain, lc_plain = _underlay_stack(False)
                cv2.imwrite(str(v2_dir / "traj_v2_underlay_final.png"), u_plain)
                out_meta["trajectory_viz_v2_underlay_final_image"] = f"{rel_base}/traj_v2_underlay_final.png"
                manifest_files["underlay_final"] = out_meta["trajectory_viz_v2_underlay_final_image"]
                phase3_summary["layer_counts_plain"] = lc_plain
            if p3uc:
                p3_any = True
                u_ctx, lc_ctx = _underlay_stack(True)
                cv2.imwrite(str(v2_dir / "traj_v2_underlay_final_context.png"), u_ctx)
                out_meta["trajectory_viz_v2_underlay_final_context_image"] = (
                    f"{rel_base}/traj_v2_underlay_final_context.png"
                )
                manifest_files["underlay_final_context"] = out_meta["trajectory_viz_v2_underlay_final_context_image"]
                phase3_summary["layer_counts_context"] = lc_ctx
            if bool(getattr(cfg, "path_v3_export_suppressed_residual", True)) if cfg else True:
                p3_any = True
                csup = img_bgr.copy()
                supp_pl = [p for p in plist_cap if bool(p.get("suppressed", False))]
                ns = self._draw_v3_polyline2d_faint_layer(
                    csup, supp_pl, sup_ct, sup_a, sw, ew, a0, a1, w, h, n_res
                )
                cv2.imwrite(str(v2_dir / "traj_v2_suppressed_residual.png"), csup)
                out_meta["trajectory_viz_v2_suppressed_residual_image"] = f"{rel_base}/traj_v2_suppressed_residual.png"
                manifest_files["suppressed_residual"] = out_meta["trajectory_viz_v2_suppressed_residual_image"]
                phase3_summary["suppressed_paths_drawn"] = ns
            phase3_summary["enabled"] = bool(p3_any)
            phase3_summary["final_top"] = fin_n
            phase3_summary["all_layer_polyline_2d_only"] = all_poly
        elif paths_sorted is None:
            phase3_summary = {"enabled": False, "reason": "paths_sorted_not_provided"}
        elif cfg and not bool(getattr(cfg, "path_v3_visualization_enabled", True)):
            phase3_summary = {"enabled": False, "reason": "path_v3_visualization_disabled"}

        manifest: Dict[str, Any] = {
            "schema": "citv_trajectory_viz_v2_manifest_v3",
            "path_stem": path_stem,
            "draw_scope": scope,
            "prefer_geodesic": prefer_geo,
            "truncated": truncated,
            "total_candidates": len(plist_all),
            "drawn_count": len(plist_cap),
            "max_paths": max_paths,
            "ink_bgr": list(ink_t),
            "outline_enabled": bool(getattr(cfg, "path_v2_outline_enabled", False)) if cfg else False,
            "paths": manifest_paths,
            "files": manifest_files,
            "phase2": phase2_summary,
            "phase3": phase3_summary,
        }
        self._write_json(manifest, v2_dir / "traj_v2_manifest.json")

        return {
            **out_meta,
            "trajectory_viz_v2_manifest_json": f"{rel_base}/traj_v2_manifest.json",
        }

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        t = (text or "").strip()
        if max_chars <= 0:
            return ""
        if len(t) <= max_chars:
            return t
        # keep start (most informative) and indicate truncation
        cut = max(0, max_chars - 1)
        return t[:cut] + "…"

    @staticmethod
    def _select_top_paths_by_level(paths: List[Dict[str, Any]], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        levels = ["region", "object", "mask"]
        out: Dict[str, List[Dict[str, Any]]] = {}
        for lvl in levels:
            lvl_paths = [p for p in (paths or []) if str(p.get("path_level", "")).strip().lower() == lvl]
            lvl_paths = sorted(
                lvl_paths,
                key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
                reverse=True,
            )
            out[lvl] = lvl_paths[: max(0, int(top_k))]
        return out

    @staticmethod
    def _draw_path_rgba(
        canvas_rgba: np.ndarray,
        pts: List[Tuple[int, int]],
        color_bgr: Tuple[int, int, int],
        start_w: int,
        end_w: int,
        alpha_start: float,
        alpha_end: float,
        draw_arrow: bool,
        draw_markers: bool,
        text_lines: List[str],
        text_color_bgr: Tuple[int, int, int] = (255, 255, 255),
        text_alpha: float = 0.9,
        font_scale: float = 0.45,
        font_thickness: int = 1,
        text_max_chars: int = 46,
    ) -> None:
        """
        canvas_rgba: HxWx4 uint8 (BGRA)
        """
        if canvas_rgba is None:
            return
        h, w = canvas_rgba.shape[:2]
        if not pts or len(pts) < 2:
            return

        # polyline segments with taper
        nseg = len(pts) - 1
        start_w = max(1, int(start_w))
        end_w = max(1, int(end_w))
        nseg_denom = max(1, nseg - 1)
        for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
            t = i / float(nseg_denom)
            ww = int(round(start_w + (end_w - start_w) * t))
            aa = float(alpha_start + (alpha_end - alpha_start) * t)
            aa = max(0.0, min(1.0, aa))
            alpha_px = int(round(aa * 255.0))
            if alpha_px <= 0:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(mask, p0, p1, 255, ww, lineType=cv2.LINE_AA)
            m = mask > 0
            if not np.any(m):
                continue
            canvas_rgba[m, 0] = int(color_bgr[0])
            canvas_rgba[m, 1] = int(color_bgr[1])
            canvas_rgba[m, 2] = int(color_bgr[2])
            canvas_rgba[m, 3] = np.maximum(canvas_rgba[m, 3], alpha_px)

        # markers
        if draw_markers:
            sx, sy = pts[0]
            gx, gy = pts[-1]
            marker_r_start = max(2, int(round(start_w / 2.0)))
            marker_r_end = max(2, int(round(end_w / 2.0)))
            alpha_s = int(round(max(0.0, min(1.0, alpha_start)) * 255.0))
            alpha_g = int(round(max(0.0, min(1.0, alpha_end)) * 255.0))
            if alpha_s > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (sx, sy), marker_r_start, 255, -1, lineType=cv2.LINE_AA)
                m = mask > 0
                canvas_rgba[m, 0] = 0
                canvas_rgba[m, 1] = 220
                canvas_rgba[m, 2] = 0
                canvas_rgba[m, 3] = np.maximum(canvas_rgba[m, 3], alpha_s)
            if alpha_g > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (gx, gy), marker_r_end, 255, -1, lineType=cv2.LINE_AA)
                m = mask > 0
                canvas_rgba[m, 0] = 0
                canvas_rgba[m, 1] = 0
                canvas_rgba[m, 2] = 220
                canvas_rgba[m, 3] = np.maximum(canvas_rgba[m, 3], alpha_g)

        # arrowhead at goal
        if draw_arrow:
            sx, sy = pts[0]
            gx, gy = pts[-1]
            dx = float(gx - sx)
            dy = float(gy - sy)
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                ux, uy = dx / norm, dy / norm  # image coordinates
                px, py = -uy, ux
                arrow_len = max(10.0, float(end_w) * 6.0)
                arrow_h = max(6.0, float(end_w) * 3.0)
                base_x = float(gx) - ux * arrow_len
                base_y = float(gy) - uy * arrow_len
                tip = (int(round(gx)), int(round(gy)))
                left = (int(round(base_x + px * (arrow_h / 2.0))), int(round(base_y + py * (arrow_h / 2.0))))
                right = (int(round(base_x - px * (arrow_h / 2.0))), int(round(base_y - py * (arrow_h / 2.0))))
                mask = np.zeros((h, w), dtype=np.uint8)
                tri = np.array([tip, left, right], dtype=np.int32)
                cv2.fillConvexPoly(mask, tri, 255, lineType=cv2.LINE_AA)
                m = mask > 0
                if np.any(m):
                    alpha_px = int(round(max(0.0, min(1.0, alpha_end)) * 255.0))
                    canvas_rgba[m, 0] = int(color_bgr[0])
                    canvas_rgba[m, 1] = int(color_bgr[1])
                    canvas_rgba[m, 2] = int(color_bgr[2])
                    canvas_rgba[m, 3] = np.maximum(canvas_rgba[m, 3], alpha_px)

        # text
        if text_lines:
            max_chars = int(text_max_chars) if text_max_chars is not None else 46
            lines = [SceneUnderstandingPipeline._truncate_text(t, max_chars) for t in (text_lines or []) if str(t).strip()]
            if lines:
                sx, sy = pts[0]
                # Place near start; shift if top gets clipped.
                tx = int(round(sx + 6))
                ty = int(round(sy - 8))
                if ty < 14:
                    ty = int(round(sy + 18))
                font = cv2.FONT_HERSHEY_SIMPLEX
                # background-less text; alpha is encoded in canvas alpha via mask.
                line_gap = 4
                y = ty
                for line in lines[:4]:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.putText(
                        mask,
                        line,
                        (tx, y),
                        font,
                        float(font_scale),
                        255,
                        int(font_thickness),
                        lineType=cv2.LINE_AA,
                    )
                    m = mask > 0
                    if np.any(m):
                        alpha_px = int(round(max(0.0, min(1.0, text_alpha)) * 255.0))
                        canvas_rgba[m, 0] = int(text_color_bgr[0])
                        canvas_rgba[m, 1] = int(text_color_bgr[1])
                        canvas_rgba[m, 2] = int(text_color_bgr[2])
                        canvas_rgba[m, 3] = np.maximum(canvas_rgba[m, 3], alpha_px)
                    # advance y by approximate text height
                    y += int(round(14 * float(font_scale))) + line_gap

    @staticmethod
    def _draw_regions_contours_bgr(img_bgr: np.ndarray, label_map: np.ndarray) -> None:
        from scene_understanding.pathing.path_canvas import draw_regions_contours_bgr

        draw_regions_contours_bgr(img_bgr, label_map)

    @staticmethod
    def _draw_objects_boxes_bgr(img_bgr: np.ndarray, objects: List[Dict[str, Any]], max_boxes: int = 40) -> None:
        from scene_understanding.pathing.path_canvas import draw_objects_boxes_bgr

        draw_objects_boxes_bgr(img_bgr, objects, max_boxes=max_boxes)

    def _local_path_description(self, path: Dict[str, Any]) -> Dict[str, str]:
        src = path.get("source_entity", {}) or {}
        tgt = path.get("target_entity", {}) or {}
        src_lbl = str(src.get("display_label") or src.get("name") or src.get("id") or "source")
        tgt_lbl = str(tgt.get("display_label") or tgt.get("name") or tgt.get("id") or "target")
        level = str(path.get("path_level", "path"))
        regions = list(path.get("regions_traversed", []) or [])
        sc = (path.get("scores") or {}).get("overall_confidence", None)
        score_txt = f"{float(sc):.2f}" if isinstance(sc, (int, float)) else "n/a"
        title = f"{level} path: {src_lbl} → {tgt_lbl}"
        why = []
        if regions:
            why.append(f"Traverses regions: {', '.join(regions)}.")
        v = path.get("validity_checks") or {}
        inv = v.get("invalid_ratio", None)
        if isinstance(inv, (int, float)):
            why.append(f"Invalid-pixel crossing ratio: {float(inv):.3f}.")
        mt = v.get("max_turn_deg", None)
        if isinstance(mt, (int, float)):
            why.append(f"Max turn: {float(mt):.1f}°.")
        summary = f"Confidence={score_txt}. " + (" ".join(why) if why else "Generated from scene regions/masks and validated in image space.")
        return {
            "title": title,
            "summary": summary,
            "why_valid": " ".join(why) if why else "Validated in image space (bounds + feasibility + curvature gates).",
            "assumptions": "Single-image hypothesis; not a temporal track.",
            "confidence_explanation": f"Overall confidence={score_txt} computed from geometric validity + evidence coverage.",
        }

    def _openrouter_describe_path(self, path: Dict[str, Any]) -> Optional[Dict[str, str]]:
        api_key = str(os.getenv("OPENROUTER_API_KEY", "")).strip()
        if not api_key:
            return None
        base = str(getattr(self.config, "openrouter_api_base", "https://openrouter.ai/api/v1")) if self.config else "https://openrouter.ai/api/v1"
        model = str(getattr(self.config, "path_description_model", "qwen/qwen-2.5-7b-instruct")) if self.config else "qwen/qwen-2.5-7b-instruct"
        max_tokens = int(getattr(self.config, "path_description_max_tokens", 320)) if self.config else 320
        temperature = float(getattr(self.config, "path_description_temperature", 0.2)) if self.config else 0.2
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You write grounded path descriptions for a single-image pipeline. Use ONLY the provided JSON fields. Do not invent entities. Output strict JSON with keys: title, summary, why_valid, assumptions, confidence_explanation.",
                },
                {
                    "role": "user",
                    "content": json.dumps({"path": path}, ensure_ascii=False),
                },
            ],
        }
        url = base.rstrip("/") + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None
        try:
            j = json.loads(raw)
            msg = (((j.get("choices") or [])[0] or {}).get("message") or {}).get("content") or ""
            out = json.loads(msg) if isinstance(msg, str) else None
            if isinstance(out, dict):
                # ensure required keys
                for k in ("title", "summary", "why_valid", "assumptions", "confidence_explanation"):
                    out.setdefault(k, "")
                return {k: str(out.get(k, "")) for k in ("title", "summary", "why_valid", "assumptions", "confidence_explanation")}
        except Exception:
            return None
        return None

    @staticmethod
    def _tokenize_text(value: Any) -> List[str]:
        txt = str(value or "").lower()
        toks = re.findall(r"[a-z0-9_]+", txt)
        return [t for t in toks if t]

    def _lexical_similarity(self, a: Any, b: Any) -> float:
        ta = set(self._tokenize_text(a))
        tb = set(self._tokenize_text(b))
        if not ta or not tb:
            return 0.0
        inter = len(ta.intersection(tb))
        den = max(1, len(ta.union(tb)))
        return float(inter) / float(den)

    def _build_semantic_layer(
        self,
        objects: List[Dict[str, Any]],
        regions: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        relation_blob = " ".join([str(r.get("predicate", "")) for r in (relations or [])]).lower()
        relation_text = " ".join([json.dumps(r, ensure_ascii=False).lower() for r in (relations or [])])
        entity_mentions: Dict[str, int] = {}
        entities: List[Dict[str, Any]] = []
        actor_candidates: List[Tuple[str, int]] = []
        for o in (objects or []):
            eid = str(o.get("id") or o.get("instance_id") or o.get("mask_id") or "")
            if not eid:
                continue
            name = str(o.get("canonical_name") or o.get("name") or o.get("label") or "entity")
            caption = str(o.get("caption") or "")
            mentions = 0
            for key in [eid.lower(), name.lower()]:
                if key:
                    mentions = max(mentions, relation_text.count(key))
            entity_mentions[eid] = mentions
            role = "entity"
            ent = {
                "id": eid,
                "name": name,
                "normalized_name": "_".join(self._tokenize_text(name))[:64] or "entity",
                "role": role,
                "kind": str(o.get("entity_kind", "object")),
            }
            entities.append(ent)
            actor_candidates.append((eid, mentions))

        actors: List[Dict[str, Any]] = []
        if actor_candidates:
            vals = np.asarray([float(v) for _eid, v in actor_candidates], dtype=np.float32)
            thr = float(np.percentile(vals, 75.0)) if len(vals) > 1 else float(vals[0])
            thr = max(1.0, thr)
            actor_ids = {eid for eid, v in actor_candidates if float(v) >= thr}
            for ent in entities:
                if str(ent.get("id", "")) in actor_ids:
                    ent["role"] = "actor"
                    actors.append(ent)

        region_affordances: List[Dict[str, Any]] = []
        for r in (regions or []):
            rid = str(r.get("id") or r.get("region_id") or "")
            rname = str(r.get("name") or r.get("label") or rid or "region")
            text = rname.lower()
            if any(k in text for k in ["sky", "ceiling", "far", "background"]):
                aff = "far_background"
            elif any(k in text for k in ["road", "floor", "ground", "sidewalk", "path"]):
                aff = "walkable"
            elif any(k in text for k in ["wall", "tree", "building", "car", "obstacle"]):
                aff = "obstacle"
            else:
                aff = "interaction_zone"
            region_affordances.append({
                "region_id": rid,
                "region_name": rname,
                "affordance": aff,
            })

        actor_intents: List[Dict[str, Any]] = []
        for a in actors:
            intents = ["traverse", "approach", "inspect", "idle", "avoid"]
            actor_intents.append({"actor_id": a["id"], "intent_candidates": intents})

        return {
            "semantic_enabled": True,
            "entities": entities,
            "actors": actors,
            "region_affordances": region_affordances,
            "actor_intents": actor_intents,
        }

    def _path_semantic_evidence(
        self,
        path: Dict[str, Any],
        semantic_layer: Dict[str, Any],
        *,
        semantic_validity_floor: Optional[float] = None,
    ) -> Dict[str, Any]:
        src = path.get("source_entity", {}) or {}
        tgt = path.get("target_entity", {}) or {}
        src_lbl = str(src.get("display_label") or src.get("name") or src.get("id") or "")
        tgt_lbl = str(tgt.get("display_label") or tgt.get("name") or tgt.get("id") or "")
        regions = [str(r).lower() for r in (path.get("regions_traversed") or [])]
        afford = semantic_layer.get("region_affordances") or []
        aff_by_region = {str(a.get("region_name", "")).lower(): str(a.get("affordance", "interaction_zone")) for a in afford}
        trace: List[Dict[str, Any]] = []
        bad = 0
        for r in regions:
            af = aff_by_region.get(r, "interaction_zone")
            trace.append({"region": r, "affordance": af})
            if af in ("obstacle", "far_background"):
                bad += 1
        sem_base = 1.0 - (float(bad) / float(max(1, len(regions))))
        proto_traverse = "walk run move traverse path ground road"
        proto_interact = "approach inspect interact reach object"
        sim = max(self._lexical_similarity(f"{src_lbl} {tgt_lbl}", proto_traverse), self._lexical_similarity(f"{src_lbl} {tgt_lbl}", proto_interact))
        sem_w = float(getattr(self.config, "path_semantic_affordance_weight", 0.65))
        lex_w = float(getattr(self.config, "path_semantic_lexical_weight", 0.35))
        _wsum = max(1e-6, sem_w + lex_w)
        sem_w, lex_w = sem_w / _wsum, lex_w / _wsum
        score = max(0.0, min(1.0, sem_w * sem_base + lex_w * sim))
        sem_floor = (
            float(semantic_validity_floor)
            if semantic_validity_floor is not None
            else float(getattr(self.config, "path_semantic_validity_floor", 0.30))
        )
        reasons = []
        if bad == 0:
            reasons.append("regions are mostly traversable/interactive")
        else:
            reasons.append("path crosses obstacle/far-background regions")
        reasons.append(f"lexical semantic match={sim:.2f}")
        return {
            "semantic_validity_score": score,
            "semantic_valid": bool(score >= sem_floor),
            "semantic_reasons": reasons,
            "affordance_trace": trace,
        }

    def _semantic_enrichment_for_path(self, path: Dict[str, Any], sem: Dict[str, Any], allow_llm: bool) -> Dict[str, Any]:
        local = {
            "summary": f"Semantic plausibility {float(sem.get('semantic_validity_score', 0.0)):.2f}; {'; '.join(sem.get('semantic_reasons', [])[:2])}",
            "semantic_source": "local",
            "llm_status": "disabled",
            "llm_error": "",
        }
        if not allow_llm:
            return local
        try:
            remote = self._openrouter_describe_path(path)
            if remote:
                return {
                    "summary": str(remote.get("summary", "")).strip() or local["summary"],
                    "semantic_source": "llm",
                    "llm_status": "ok",
                    "llm_error": "",
                }
            return {**local, "semantic_source": "fallback", "llm_status": "no_output", "llm_error": "empty_or_invalid_response"}
        except Exception as ex:
            return {**local, "semantic_source": "fallback", "llm_status": "error", "llm_error": str(ex)[:200]}

    @staticmethod
    def _extract_kinematic_signatures(
        polyline_3d: List[List[float]],
        *,
        jump_z_thresh: float = 0.25,
        climb_z_thresh: float = 0.12,
        idle_fraction: float = 0.08,
        crawl_min_run: int = 6,
    ) -> List[Dict[str, Any]]:
        """Analyse the Z-profile of a 3-D polyline to extract motion tags.

        Returns a list of segment dicts with keys ``start_idx``, ``end_idx``,
        ``motion`` (idle | walk | crawl | climb | jump | descend) and ``dz``.

        ``crawl`` is emitted for sustained flat runs (≥ ``crawl_min_run`` steps)
        whose Z-variance is less than 1/4 of ``idle_fraction``. This is a
        clearance proxy: very long, very flat segments in a 3-D path are likely
        low-ceiling passages.
        """
        sigs: List[Dict[str, Any]] = []
        if len(polyline_3d) < 2:
            return sigs
        zs = [float(pt[2]) if len(pt) > 2 else 0.0 for pt in polyline_3d]
        n = len(zs)
        i = 0
        while i < n - 1:
            dz = zs[i + 1] - zs[i]
            if abs(dz) < idle_fraction:
                motion = "walk"
            elif dz > jump_z_thresh:
                motion = "jump"
            elif dz > climb_z_thresh:
                motion = "climb"
            elif dz < -jump_z_thresh:
                motion = "descend"
            elif dz < -climb_z_thresh:
                motion = "descend"
            else:
                motion = "walk"
            # Extend run of the same motion type.
            j = i + 1
            while j < n - 1:
                dz_n = zs[j + 1] - zs[j]
                if abs(dz_n) < idle_fraction:
                    nm = "walk"
                elif dz_n > jump_z_thresh:
                    nm = "jump"
                elif dz_n > climb_z_thresh:
                    nm = "climb"
                elif dz_n < -jump_z_thresh:
                    nm = "descend"
                elif dz_n < -climb_z_thresh:
                    nm = "descend"
                else:
                    nm = "walk"
                if nm != motion:
                    break
                j += 1
            # Promote long very-flat walk segments to crawl.
            if motion == "walk" and (j - i) >= crawl_min_run:
                run_zs = zs[i : j + 1]
                z_var = max(run_zs) - min(run_zs)
                if z_var < idle_fraction / 4.0:
                    motion = "crawl"
            sigs.append({
                "start_idx": i,
                "end_idx": j,
                "motion": motion,
                "dz": float(zs[j] - zs[i]),
            })
            i = j
        return sigs

    def _animation_plan_for_paths(self, paths: List[Dict[str, Any]], top_k: int, cfg: Any) -> Dict[str, Any]:
        ranked = sorted(paths, key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)), reverse=True)[:max(0, int(top_k))]
        plan_paths: List[Dict[str, Any]] = []
        fps = int(getattr(cfg, "path_animation_fps", 24)) if cfg else 24
        speed_walk = float(getattr(cfg, "path_speed_walk_px_s", 80.0)) if cfg else 80.0
        speed_run = float(getattr(cfg, "path_speed_run_px_s", 150.0)) if cfg else 150.0
        idle_s = float(getattr(cfg, "path_duration_idle_s", 0.5)) if cfg else 0.5
        jump_s = float(getattr(cfg, "path_duration_jump_s", 0.6)) if cfg else 0.6
        climb_s = float(getattr(cfg, "path_duration_climb_s", 0.8)) if cfg else 0.8
        for p in ranked:
            pts = [tuple(map(float, xy)) for xy in (p.get("polyline_2d") or [])]
            if len(pts) < 2:
                continue
            length = 0.0
            for i in range(1, len(pts)):
                length += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            conf = float((p.get("scores") or {}).get("overall_confidence", 0.0))

            # Kinematic signatures from 3D polyline (Phase 2.3).
            poly3d = p.get("polyline_3d") or []
            kin_sigs = self._extract_kinematic_signatures(poly3d) if poly3d else []

            if kin_sigs:
                # Build segments from kinematic analysis of the Z-profile.
                t = 0.0
                segments = [{"motion": "idle", "t0_s": t, "t1_s": t + idle_s}]
                t += idle_s
                for sig in kin_sigs:
                    motion = sig["motion"]
                    seg_n = max(1, sig["end_idx"] - sig["start_idx"])
                    seg_frac = seg_n / max(1, len(poly3d) - 1)
                    if motion in ("jump", "climb", "descend"):
                        dur = (jump_s if motion == "jump" else climb_s) * max(1, seg_n // 4)
                    else:
                        spd = speed_run if conf >= 0.75 else speed_walk
                        dur = float(length * seg_frac / max(1.0, spd))
                    segments.append({"motion": motion, "t0_s": t, "t1_s": t + dur})
                    t += dur
            else:
                # Fallback: confidence-gated single-motion segment.
                motion = "run" if conf >= 0.75 else "walk"
                speed = speed_run if motion == "run" else speed_walk
                move_s = float(length / max(1.0, speed))
                t = 0.0
                segments = [
                    {"motion": "idle", "t0_s": t, "t1_s": t + idle_s},
                    {"motion": motion, "t0_s": t + idle_s, "t1_s": t + idle_s + move_s},
                ]
                t = t + idle_s + move_s
                if conf < 0.5:
                    segments.append({"motion": "jump", "t0_s": t, "t1_s": t + jump_s})
                    t += jump_s

            # trajectory_id links animation plan → path record (same value as path_id).
            trajectory_id = p.get("path_id", "")
            timeline = [
                {
                    "time_s": seg["t0_s"],
                    "motion": seg["motion"],
                    "path_id": trajectory_id,
                    "trajectory_id": trajectory_id,
                }
                for seg in segments
            ]

            # motion_mode_candidates: ranked list of plausible motions for this path.
            motion_counts: Dict[str, int] = {}
            for seg in segments:
                motion_counts[seg["motion"]] = motion_counts.get(seg["motion"], 0) + 1
            total_segs = max(1, len(segments))
            all_motions = ["idle", "walk", "run", "crawl", "climb", "jump", "descend"]
            mode_candidates = []
            for m in all_motions:
                base = float(motion_counts.get(m, 0)) / total_segs
                if m == "run":
                    base = max(base, (conf - 0.5) * 2.0) if conf >= 0.5 else base
                if m == "crawl":
                    base = max(base, 0.05)  # always a low-probability option
                mode_candidates.append({"motion": m, "score": round(min(base, 1.0), 3)})
            mode_candidates.sort(key=lambda x: x["score"], reverse=True)

            # confidence_breakdown: per-component score breakdown.
            scores = p.get("scores") or {}
            confidence_breakdown = {
                "overall": conf,
                "geometric": float(scores.get("geometric_confidence", conf)),
                "semantic": float(scores.get("semantic_confidence", conf)),
                "kinematic": float(scores.get("kinematic_confidence", 0.5 if kin_sigs else 0.0)),
            }

            # relation_guardrails: relations that constrain this path's validity.
            relation_guardrails = list(p.get("relation_guardrails") or [])

            plan_paths.append({
                "path_id": trajectory_id,
                "trajectory_id": trajectory_id,
                "path_num": p.get("path_num", 0),
                "action_fit_confidence": conf,
                "assumptions": "kinematic_signature" if kin_sigs else "confidence_gated",
                "kinematic_signatures": kin_sigs,
                "motion_mode_candidates": mode_candidates,
                "confidence_breakdown": confidence_breakdown,
                "relation_guardrails": relation_guardrails,
                "segments": segments,
                "trajectory_points": p.get("polyline_2d", []),
                "timeline_records": timeline,
                "fps": fps,
                "duration_s": t,
            })
        return {"fps": fps, "paths": plan_paths}

    def _build_path_cost_map(
        self,
        img_bgr: np.ndarray,
        region_label_map: np.ndarray,
        obstacle_mask: np.ndarray,
        cfg: Any,
        objects: Optional[List[Dict[str, Any]]] = None,
        metric_depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, float]] = None,
        object_label_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        semantic_enabled = bool(getattr(cfg, "path_cost_semantic_enabled", False)) if cfg else False
        if semantic_enabled:
            return self._build_semantic_path_cost_map(
                img_bgr, region_label_map, obstacle_mask, cfg,
                objects, metric_depth_m, intrinsics, object_label_map
            )
        else:
            from scene_understanding.pathing.cost_map import build_path_cost_map
            return build_path_cost_map(img_bgr, region_label_map, obstacle_mask, cfg)

    def _build_semantic_path_cost_map(
        self,
        img_bgr: np.ndarray,
        region_label_map: np.ndarray,
        obstacle_mask: np.ndarray,
        cfg: Any,
        objects: Optional[List[Dict[str, Any]]],
        metric_depth_m: Optional[np.ndarray],
        intrinsics: Optional[Dict[str, float]],
        object_label_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build semantic cost map by composing geometry, VLM, and scene-graph layers."""
        from scene_understanding.pathing.semantic_cost import (
            layer_geometry, layer_scene_graph, layer_agent_physics,
            layer_depth_noise, precision_weighted_fuse
        )
        from scene_understanding.pathing.semantic_vlm import layer_vlm, load_cost_prompts

        h, w = img_bgr.shape[:2]

        # Load prompts for VLM layer
        prompts = None
        try:
            prompts_path = getattr(cfg, "path_cost_vlm_prompts_file", "cost_map_prompts.yaml") if cfg else "cost_map_prompts.yaml"
            prompts = load_cost_prompts(prompts_path)
        except Exception:
            pass  # VLM layer will be zero-precision

        # Build layers
        layers = []

        # Layer A: Geometry
        layers.append(layer_geometry(img_bgr, metric_depth_m, obstacle_mask))

        # Layer B: VLM (if prompts available)
        if prompts:
            layers.append(layer_vlm(img_bgr, objects or [], object_label_map, prompts))

        # Layer D: Scene graph
        if objects:
            layers.append(layer_scene_graph(h, w, objects, object_label_map))

        # Layer E: Agent physics (if profile available)
        profile = None
        profile_path = getattr(cfg, "path_cost_agent_profile_file", "agent.profile.yaml") if cfg else "agent.profile.yaml"
        try:
            import yaml
            with open(profile_path, 'r') as f:
                profile_data = yaml.safe_load(f)
            # Create a simple profile object
            class SimpleProfile:
                def __init__(self, data):
                    self.data = data
                def effective_pad_x_m(self):
                    return self.data.get('pad_x_m', self.data.get('half_width_m', 0.3))
                @property
                def half_width_m(self):
                    return self.data.get('half_width_m', 0.3)
                @property
                def foot_tolerance_m(self):
                    return self.data.get('foot_tolerance_m', 0.05)
            profile = SimpleProfile(profile_data)
        except Exception:
            pass  # Layer will refuse to build

        if profile and intrinsics and metric_depth_m is not None:
            layers.append(layer_agent_physics((h, w), obstacle_mask, metric_depth_m, intrinsics, profile))

        # Layer F: Depth noise
        layers.append(layer_depth_noise(metric_depth_m))

        # Fuse layers
        semantic_map = precision_weighted_fuse(layers)

        # Emit layers for QA if requested
        emit_layers = bool(getattr(cfg, "path_cost_emit_layers_for_qa", False)) if cfg else False
        if emit_layers:
            # TODO: Implement layer emission for QA
            pass

        return semantic_map.cost

    @staticmethod
    def _astar_on_cost_map(
        cost_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Back-compat shim — name retained, **body is now FMM backtrace**.

        A* has been retired from every live planning entry point per
        user direction. The historical method name is preserved so
        pre-existing callers and log strings keep working, but the
        implementation is now a single-shot Fast Marching Method solve
        + steepest-descent backtrace on the supplied semantic cost map.
        The return type is identical (list of (x, y) tuples), and the
        geodesic is globally optimal on the continuous cost field.
        """

        from scene_understanding.pathing.semantic_fmm import fmm_backtrace

        return fmm_backtrace(cost_map, start, goal)

    @staticmethod
    def _resample_polyline(pts: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
        if len(pts) < 2 or n <= 2:
            return pts
        arr = np.asarray(pts, dtype=np.float32)
        seg = arr[1:] - arr[:-1]
        d = np.sqrt(np.sum(seg * seg, axis=1))
        cum = np.concatenate([[0.0], np.cumsum(d)])
        total = float(cum[-1])
        if total <= 1e-6:
            return pts
        samples = np.linspace(0.0, total, n)
        out: List[Tuple[int, int]] = []
        j = 0
        for s in samples:
            while j < len(d) - 1 and cum[j + 1] < s:
                j += 1
            den = max(1e-6, float(cum[j + 1] - cum[j]))
            t = float((s - cum[j]) / den)
            p = arr[j] * (1.0 - t) + arr[j + 1] * t
            out.append((int(round(p[0])), int(round(p[1]))))
        return out

    @staticmethod
    def _object_depth_m(obj: Dict[str, Any]) -> Optional[float]:
        c3d = obj.get("coordinates_3d") or obj.get("coordinates_3d_from_mask") or {}
        z = c3d.get("z", None)
        if isinstance(z, (int, float)):
            return float(z)
        ds = obj.get("depth_stats") or {}
        for k in ("median", "mean", "min", "max"):
            v = ds.get(k, None)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def _path_motion_metrics(
        self,
        pts: List[Tuple[int, int]],
        lm: np.ndarray,
        obstacle_mask: np.ndarray,
        src_obj: Optional[Dict[str, Any]],
        tgt_obj: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if len(pts) < 2:
            return {
                "border_hug_ratio": 1.0,
                "obstacle_contact_ratio": 1.0,
                "depth_delta_m": 0.0,
                "depth_alignment_score": 0.0,
                "motion_distance_px": 0.0,
                "motion_primary_score": 0.0,
                "trajectory_type": "unknown",
            }
        h, w = lm.shape[:2]
        bd = np.zeros((h, w), dtype=np.uint8)
        bd[1:, :] |= (lm[1:, :] != lm[:-1, :]).astype(np.uint8) * 255
        bd[:, 1:] |= (lm[:, 1:] != lm[:, :-1]).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bd = cv2.dilate(bd, k, iterations=1) > 0
        valid = [(x, y) for (x, y) in pts if 0 <= x < w and 0 <= y < h]
        n = max(1, len(valid))
        border_hug = float(sum(1 for (x, y) in valid if bool(bd[y, x]))) / float(n)
        obs_ratio = float(sum(1 for (x, y) in valid if bool(obstacle_mask[y, x]))) / float(n)
        dist = 0.0
        for i in range(1, len(pts)):
            dist += math.hypot(float(pts[i][0] - pts[i - 1][0]), float(pts[i][1] - pts[i - 1][1]))
        z0 = self._object_depth_m(src_obj or {})
        z1 = self._object_depth_m(tgt_obj or {})
        dz = abs(float(z1 - z0)) if z0 is not None and z1 is not None else 0.0
        depth_align = 1.0 / (1.0 + dz)
        straight = math.hypot(float(pts[-1][0] - pts[0][0]), float(pts[-1][1] - pts[0][1]))
        length_ratio = float(straight / max(1e-6, dist))
        # higher is better: long movement corridor + low boundary hugging + low obstacle contact + depth-consistent
        primary = (0.30 * max(0.0, min(1.0, length_ratio))) + (0.30 * (1.0 - border_hug)) + (0.20 * (1.0 - obs_ratio)) + (0.20 * depth_align)
        if z0 is not None and z1 is not None:
            if z1 > z0 + 0.2:
                ttype = "recede"
            elif z1 < z0 - 0.2:
                ttype = "approach"
            else:
                ttype = "traverse"
        else:
            ttype = "traverse"
        return {
            "border_hug_ratio": float(border_hug),
            "obstacle_contact_ratio": float(obs_ratio),
            "depth_delta_m": float(dz),
            "depth_alignment_score": float(depth_align),
            "motion_distance_px": float(dist),
            "length_efficiency": float(length_ratio),
            "motion_primary_score": float(max(0.0, min(1.0, primary))),
            "trajectory_type": ttype,
        }

    @staticmethod
    def _write_motion_contract_overlay(
        img_bgr: np.ndarray,
        paths_sorted: List[Dict[str, Any]],
        traj_bundle: Dict[str, Any],
        out_path: Path,
        cfg: Optional[Any] = None,
    ) -> None:
        from scene_understanding.visualization.motion_contract_overlay import write_motion_contract_overlay

        write_motion_contract_overlay(img_bgr, paths_sorted, traj_bundle, out_path, cfg=cfg)

    def _export_path_hypotheses_for_track(
        self,
        img_bgr: np.ndarray,
        path_stem: str,
        track_dir_name: str,
        track_dir: Path,
        objects_3d_with_masks: List[Dict[str, Any]],
        regions_block: Optional[Dict[str, Any]],
        region_label_map: Optional[np.ndarray],
        region_adjacency: Optional[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        metric_depth_m: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        cfg = self.config
        cpu_benchmark_mode = bool(getattr(cfg, "cpu_benchmark_mode", False)) if cfg else False
        cpu_disable_heavy_visuals = bool(getattr(cfg, "cpu_benchmark_disable_heavy_visuals", False)) if cfg else False
        from scene_understanding.pathing.export_workspace import prepare_path_hypotheses_workspace

        ws = prepare_path_hypotheses_workspace(
            cfg, img_bgr, path_stem, track_dir, region_label_map, regions_block, region_adjacency
        )
        if ws is None:
            return {}
        paths_root_dir = ws["paths_root_dir"]
        images_root_dir = ws["images_root_dir"]
        images_region_dir = ws["images_region_dir"]
        images_object_dir = ws["images_object_dir"]
        images_mask_dir = ws["images_mask_dir"]
        h = ws["h"]
        w = ws["w"]
        lm = ws["lm"]
        regions_meta = ws["regions_meta"]
        region_by_id = ws["region_by_id"]
        feasible = ws["feasible"]

        # Canonical scene context snapshot used by path/trajectory planning.
        scene_ctx_rel = ""
        if bool(getattr(cfg, "path_scene_context_enabled", True)) if cfg else True:
            _intr = None
            if isinstance(getattr(self, "fixed_intrinsics", None), dict):
                _intr = dict(getattr(self, "fixed_intrinsics"))
            scene_ctx = _build_scene_context(
                image_stem=path_stem,
                track=track_dir_name,
                img_bgr=img_bgr,
                objects_3d_with_masks=objects_3d_with_masks,
                regions_block=regions_block,
                region_label_map=region_label_map,
                region_adjacency=region_adjacency,
                relations=relations,
                metric_depth_m=metric_depth_m,
                intrinsics=_intr,
            )
            scene_ctx_issues = _validate_scene_context(scene_ctx)
            if scene_ctx_issues:
                print(f"  [Paths] scene context validation issues: {len(scene_ctx_issues)}")
            scene_ctx_payload = scene_ctx.to_json_ready()
            if scene_ctx_issues:
                scene_ctx_payload["validation_issues"] = list(scene_ctx_issues)
            scene_ctx_path = paths_root_dir / "scene_context.json"
            self._write_json(scene_ctx_payload, scene_ctx_path)
            scene_ctx_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/scene_context.json"

        # Object masks + names
        objs = [o for o in objects_3d_with_masks if o.get("entity_kind", "object") != "region"]
        obj_by_id_global = {str(o.get("id", "")): o for o in objs if str(o.get("id", "")).strip()}
        movers = [o for o in objs if str(o.get("label", "")).strip().lower() in ("person", "vehicle", "animal")]
        if not movers:
            movers = objs[:]
        # obstacle mask = all object masks (we'll allow traversing inside src/tgt by excluding them later)
        all_obs = np.zeros((h, w), dtype=bool)
        # Phase 2.2: soft obstacles (jumpable/climbable) — cost spikes only, not hard exclusions.
        soft_obs = np.zeros((h, w), dtype=bool)
        obj_mask_by_id: Dict[str, np.ndarray] = {}
        for o in objs:
            mid = str(o.get("id", ""))
            m = o.get("_sam2_mask_array", None)
            if m is None:
                continue
            mm = np.asarray(m).astype(bool)
            if mm.shape[:2] != (h, w):
                mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            obj_mask_by_id[mid] = mm
            o_label = str(o.get("label", "")).strip().lower()
            if is_soft_obstacle(o_label):
                soft_obs |= mm
            else:
                all_obs |= mm

        # Build object label map early for semantic cost map
        _ordered_masks_for_label_map = []
        for o in objs:
            m = o.get("_sam2_mask_array", None)
            if m is not None:
                mm = np.asarray(m).astype(bool)
                if mm.shape[:2] != (h, w):
                    mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                _ordered_masks_for_label_map.append(mm)
        object_label_map = self._build_object_label_map(_ordered_masks_for_label_map, h, w) if _ordered_masks_for_label_map else None

        pair_proposals: List[Dict[str, Any]] = []
        pair_proposal_enabled = bool(getattr(cfg, "path_pair_proposal_enabled", True)) if cfg else True
        pair_top_k_targets = int(getattr(cfg, "path_pair_top_k_targets", 4)) if cfg else 4
        allow_static_static = bool(getattr(cfg, "path_pair_allow_static_static", False)) if cfg else False
        rel_text = " ".join([json.dumps(r, ensure_ascii=False).lower() for r in (relations or [])])
        rel_mentions: Dict[str, int] = {}
        for o in objs:
            oid = str(o.get("id", ""))
            oname = str(o.get("canonical_name") or o.get("name") or o.get("label") or oid).lower()
            rel_mentions[oid] = max(rel_text.count(oid.lower()) if oid else 0, rel_text.count(oname) if oname else 0)
        mention_vals = np.asarray([float(v) for v in rel_mentions.values()], dtype=np.float32) if rel_mentions else np.zeros((0,), dtype=np.float32)
        mention_thr = float(np.percentile(mention_vals, 75.0)) if mention_vals.size > 1 else (float(mention_vals[0]) if mention_vals.size == 1 else 0.0)
        mention_thr = max(1.0, mention_thr)
        if pair_proposal_enabled:
            for src in objs:
                src_id = str(src.get("id", ""))
                src_name = str(src.get("canonical_name") or src.get("name") or src.get("label") or src_id).lower()
                src_is_actor = float(rel_mentions.get(src_id, 0)) >= mention_thr
                candidates: List[Tuple[float, Dict[str, Any]]] = []
                for tgt in objs:
                    tgt_id = str(tgt.get("id", ""))
                    if not tgt_id or tgt_id == src_id:
                        continue
                    tgt_name = str(tgt.get("canonical_name") or tgt.get("name") or tgt.get("label") or tgt_id).lower()
                    tgt_is_actor = float(rel_mentions.get(tgt_id, 0)) >= mention_thr
                    if (not allow_static_static) and (not src_is_actor) and (not tgt_is_actor):
                        continue
                    sem_sim = self._lexical_similarity(src_name, tgt_name)
                    score = (0.6 if src_is_actor else 0.2) + (0.2 if not tgt_is_actor else 0.1) + 0.2 * sem_sim
                    candidates.append((score, {"src_id": src_id, "tgt_id": tgt_id, "proposal_score": score}))
                candidates.sort(key=lambda x: x[0], reverse=True)
                for _s, rec in candidates[: max(1, pair_top_k_targets)]:
                    pair_proposals.append(rec)
            # Fallback: if still empty, populate top static-static proposals by lexical similarity.
            if not pair_proposals and len(objs) >= 2:
                fallback: List[Tuple[float, Dict[str, Any]]] = []
                for src in objs:
                    src_id = str(src.get("id", ""))
                    src_name = str(src.get("canonical_name") or src.get("name") or src.get("label") or src_id).lower()
                    for tgt in objs:
                        tgt_id = str(tgt.get("id", ""))
                        if not src_id or not tgt_id or src_id == tgt_id:
                            continue
                        tgt_name = str(tgt.get("canonical_name") or tgt.get("name") or tgt.get("label") or tgt_id).lower()
                        score = 0.2 + 0.8 * self._lexical_similarity(src_name, tgt_name)
                        fallback.append((score, {"src_id": src_id, "tgt_id": tgt_id, "proposal_score": score, "proposal_source": "static_fallback"}))
                fallback.sort(key=lambda x: x[0], reverse=True)
                pair_proposals = [rec for _s, rec in fallback[: max(2, pair_top_k_targets * max(1, len(objs) // 2))]]
        else:
            for src in objs:
                for tgt in objs:
                    src_id = str(src.get("id", ""))
                    tgt_id = str(tgt.get("id", ""))
                    if src_id and tgt_id and src_id != tgt_id:
                        pair_proposals.append({"src_id": src_id, "tgt_id": tgt_id, "proposal_score": 0.0})

        use_refine = bool(getattr(cfg, "path_use_image_cost_refinement", True)) if cfg else True
        _cm_t0 = time.perf_counter()
        # Get intrinsics for semantic cost map
        intrinsics = None
        if hasattr(self, 'fixed_intrinsics') and self.fixed_intrinsics:
            intrinsics = dict(self.fixed_intrinsics)
        else:
            intrinsics = self._estimate_intrinsics(w, h)
        cost_map = self._build_path_cost_map(img_bgr, lm, all_obs, cfg, objs, metric_depth_m, intrinsics, object_label_map)
        print(f"  [Paths] cost map built ({time.perf_counter() - _cm_t0:.1f}s; shape={cost_map.shape})")

        export_trav = bool(getattr(cfg, "path_export_traversability_speed", True)) if cfg else True
        speed_map, trav_meta = build_traversability_speed_map(
            metric_depth_m, lm, all_obs, img_bgr, cfg,
            soft_obstacle_mask=soft_obs if soft_obs.any() else None,
        )
        trav_speed_npy_rel = ""
        trav_speed_png_rel = ""
        if export_trav:
            ts_path = paths_root_dir / "path_traversability_speed.npy"
            np.save(str(ts_path), speed_map)
            trav_speed_npy_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_traversability_speed.npy"
            ts_u8 = np.clip(speed_map * 255.0, 0, 255).astype(np.uint8)
            ts_color = cv2.applyColorMap(ts_u8, cv2.COLORMAP_VIRIDIS)
            ts_png = paths_root_dir / "path_traversability_speed.png"
            cv2.imwrite(str(ts_png), ts_color)
            trav_speed_png_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_traversability_speed.png"

        use_geo = bool(getattr(cfg, "path_use_traversability_geodesic", True)) if cfg else True
        geo_replace = bool(getattr(cfg, "path_geodesic_replace_astar", True)) if cfg else True
        geo_k = int(getattr(cfg, "path_geodesic_k_alt", 2)) if cfg else 1
        geo_pen = float(getattr(cfg, "path_geodesic_edge_penalty", 0.35)) if cfg else 0.35

        top_k = int(getattr(cfg, "path_top_k_per_pair", 3)) if cfg else 3
        max_cand = int(getattr(cfg, "path_max_candidates", 500)) if cfg else 500
        inv_max = float(getattr(cfg, "path_invalid_pixel_ratio_max", 0.05)) if cfg else 0.05
        max_turn = float(getattr(cfg, "path_max_turn_deg", 70.0)) if cfg else 70.0
        # Early-stage generation keeps broad candidates; calibrated policy
        # performs final confidence filtering in one pass after semantic scoring.
        min_conf = 0.0
        motion_primary_gate_pre = float(getattr(cfg, "path_motion_primary_threshold", 0.45)) if cfg else 0.45

        enable_region = bool(getattr(cfg, "path_enable_region", True)) if cfg else True
        enable_object = bool(getattr(cfg, "path_enable_object", True)) if cfg else True
        enable_mask = bool(getattr(cfg, "path_enable_mask", True)) if cfg else True

        paths: List[Dict[str, Any]] = []
        rejections: List[Dict[str, Any]] = []

        # --- Region paths ---
        if enable_region:
            region_ids = [str(r.get("id", "")) for r in regions_meta if str(r.get("id", "")).strip()]
            region_ids = [r for r in region_ids if r in region_by_id]
            _n_reg_pairs = max(1, len(region_ids) * (len(region_ids) - 1) // 2)
            _rp_t0 = time.perf_counter()
            print(f"  [Paths] region-to-region search over {_n_reg_pairs} pairs (N={len(region_ids)} regions)...")
            _rp_done = 0
            for i in range(len(region_ids)):
                for j in range(i + 1, len(region_ids)):
                    ra, rb = region_ids[i], region_ids[j]
                    kpaths = self._k_region_paths(region_adjacency, ra, rb, k=top_k)
                    for kidx, seq in enumerate(kpaths, start=1):
                        if len(paths) >= max_cand:
                            break
                        pts: List[Tuple[int, int]] = []
                        # start at centroid, then portals, then centroid
                        ca = region_by_id[seq[0]].get("centroid_2d_px", [0, 0])
                        pts.append((int(ca[0]), int(ca[1])))
                        for a_id, b_id in zip(seq, seq[1:]):
                            pts.append(self._region_portal_point(lm, region_by_id, a_id, b_id, dilation_px=2))
                        cb = region_by_id[seq[-1]].get("centroid_2d_px", [0, 0])
                        pts.append((int(cb[0]), int(cb[1])))
                        v = self._polyline_is_valid(pts, w, h, feasible=feasible, obstacles=None, invalid_ratio_max=inv_max, max_turn_deg=max_turn)
                        pid = f"rpath_{path_stem}_{seq[0]}_to_{seq[-1]}_k{str(kidx).zfill(2)}"
                        src_name = self._display_name_for_region(region_by_id.get(seq[0], {}))
                        tgt_name = self._display_name_for_region(region_by_id.get(seq[-1], {}))
                        src_disp = self._display_label(src_name, seq[0])
                        tgt_disp = self._display_label(tgt_name, seq[-1])
                        base = {
                            "path_id": pid,
                            "path_level": "region",
                            "path_type": "region_to_region",
                            "source_entity": {"type": "region", "id": seq[0], "name": src_name, "display_label": src_disp, "start_uv": list(pts[0])},
                            "target_entity": {"type": "region", "id": seq[-1], "name": tgt_name, "display_label": tgt_disp, "goal_uv": list(pts[-1])},
                            "regions_traversed": list(seq),
                            "polyline_2d": [list(p) for p in pts],
                            "constraints_applied": {"image_bounds_enforced": True, "region_adjacency_enforced": True},
                            "validity_checks": v,
                        }
                        if not v.get("ok", False):
                            rejections.append({**base, "rejected_reason": v.get("reason", "invalid")})
                            continue
                        # simple confidence from validity only (region paths are coarse)
                        conf = max(0.0, 1.0 - float(v.get("invalid_ratio", 0.0)) - 0.005 * float(v.get("max_turn_deg", 0.0)))
                        base["scores"] = {
                            "geometric_feasibility": float(conf),
                            "depth_consistency": float(conf),
                            "relation_consistency": 0.5,
                            "semantic_plausibility": 0.5,
                            "overall_confidence": float(conf),
                        }
                        mm = self._path_motion_metrics(pts, lm, np.zeros_like(all_obs), None, None)
                        base["motion_metrics"] = mm
                        base["trajectory_type"] = str(mm.get("trajectory_type", "traverse"))
                        base["is_motion_primary"] = bool(float(mm.get("motion_primary_score", 0.0)) >= motion_primary_gate_pre)
                        if base["scores"]["overall_confidence"] < min_conf:
                            rejections.append({**base, "rejected_reason": "low_confidence"})
                            continue
                        paths.append(base)
                    _rp_done += 1
                    if (_rp_done % 100 == 0) or (_rp_done == _n_reg_pairs):
                        _elapsed = time.perf_counter() - _rp_t0
                        print(f"  [Paths] region pairs {_rp_done}/{_n_reg_pairs} ({_elapsed:.1f}s)")

        # --- Object paths ---
        if enable_object:
            # Build object pair paths using region routes + portal points,
            # avoiding other object masks. This block used to run
            # A* (on ``cost_map``) plus k-diverse Dijkstra (on ``speed_map``)
            # for every (src, tgt) proposal — ~104 pairs × ~2 grid searches
            # each on a 576×1024 grid, which was the single largest
            # remaining bottleneck (~200 s). We now:
            #
            #   1. Pre-compute a travel-time field ``T`` once per unique
            #      target goal (one FMM solve per goal instead of one
            #      per pair).
            #   2. Replace the A* refine with a cheap gradient descent
            #      on the cached cost-derived T. (A* has been retired
            #      pipeline-wide per user direction.)
            #   3. Replace ``k_diverse_grid_paths`` (Dijkstra-based) with
            #      ``k_diverse_from_T`` on the cached speed-derived T —
            #      no FMM re-solve per pair, just descent + penalise.
            #   4. Run the 100+ pair tasks through a ThreadPoolExecutor
            #      so the per-pair NumPy/cv2 work overlaps across the
            #      M2 performance cores (NumPy releases the GIL during
            #      BLAS/ndimage calls).
            obj_by_id = {str(o.get("id", "")): o for o in objs if str(o.get("id", "")).strip()}
            _op_total = len(pair_proposals)
            _op_t0 = time.perf_counter()

            # Collect unique target goals from valid proposals so we only
            # pay for one FMM solve per distinct goal point rather than
            # one per proposal.
            goal_points: Dict[Tuple[int, int], str] = {}
            for _prop in pair_proposals:
                _sid = str(_prop.get("src_id", ""))
                _tid = str(_prop.get("tgt_id", ""))
                _s = obj_by_id.get(_sid)
                _t = obj_by_id.get(_tid)
                if _s is None or _t is None:
                    continue
                _t_uv = _t.get("mask_centroid_2d", [0, 0])
                _gp = (int(_t_uv[0]), int(_t_uv[1]))
                if 0 <= _gp[0] < w and 0 <= _gp[1] < h:
                    goal_points.setdefault(_gp, _tid)

            # For the speed-map FMM we want the goal's own mask to NOT
            # act as an obstacle (otherwise T is undefined at the goal
            # pixel). We keep all other object masks as obstacles; at
            # descent time a source sitting inside its own obstacle has
            # a high but finite T there and walks out via the gradient.
            feasible_base = feasible & (~all_obs)
            cost_map_np = np.asarray(cost_map, dtype=np.float32)
            speed_map_np = np.asarray(speed_map, dtype=np.float32)

            # Pre-build a masked speed field per goal (goal's mask
            # carved out of the obstacle set). The cost map solve uses
            # ``cost_map`` as-is because a single FMM T suffices for
            # the "A*-replacement" descent.
            print(
                f"  [Paths] pre-building {len(goal_points)} FMM T-maps "
                f"(was {_op_total} A*/FMM solves per pair)..."
            )
            _tmap_t0 = time.perf_counter()
            T_cost_by_goal: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
            T_speed_by_goal: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
            for _gp, _goal_oid in goal_points.items():
                try:
                    T_cost_by_goal[_gp] = _fmm_time_of_arrival(cost_map_np, _gp)
                except Exception as _e:
                    T_cost_by_goal[_gp] = None
                # Speed map variant: exclude only the goal's own mask so
                # the wavefront can actually reach the goal pixel.
                try:
                    _feasible_for_goal = feasible_base.copy()
                    _goal_mask = obj_mask_by_id.get(_goal_oid)
                    if _goal_mask is not None:
                        _feasible_for_goal |= _goal_mask
                    _sm_goal = np.where(_feasible_for_goal, speed_map_np, speed_map_np * 0.02)
                    T_speed_by_goal[_gp] = _fmm_time_of_arrival_from_speed(_sm_goal, _gp)
                except Exception as _e:
                    T_speed_by_goal[_gp] = None
            print(
                f"  [Paths] T-maps ready ({time.perf_counter() - _tmap_t0:.1f}s, "
                f"{sum(1 for v in T_cost_by_goal.values() if v is not None)} valid)"
            )

            print(
                f"  [Paths] object-to-object search over {_op_total} pair proposals "
                f"(N={len(objs)} objects, parallel workers)..."
            )

            def _process_object_pair(
                prop: Dict[str, Any],
            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
                """Per-pair worker — pure function over read-only shared state.

                Returns ``(accepted_paths, rejections)`` for the pair so
                the main thread can append them in order. Every array it
                reads (``cost_map_np``, ``speed_map_np``, T-maps,
                ``obj_mask_by_id``, ``lm``, ``feasible``, ``all_obs``) is
                treated as read-only; all mutations live on the per-task
                stack. NumPy/cv2 release the GIL during BLAS/ndimage
                operations so threads run concurrently.
                """

                local_accepted: List[Dict[str, Any]] = []
                local_rejections: List[Dict[str, Any]] = []

                src_id = str(prop.get("src_id", ""))
                tgt_id = str(prop.get("tgt_id", ""))
                src = obj_by_id.get(src_id)
                tgt = obj_by_id.get(tgt_id)
                if src is None or tgt is None:
                    return local_accepted, local_rejections
                src_uv = src.get("mask_centroid_2d", [0, 0])
                src_pt = (int(src_uv[0]), int(src_uv[1]))
                src_reg = str(src.get("region_id", "") or f"region_{int(src.get('region_index', 0) or 0)}")
                src_name = self._display_name_for_object(src)
                tgt_uv = tgt.get("mask_centroid_2d", [0, 0])
                tgt_pt = (int(tgt_uv[0]), int(tgt_uv[1]))
                tgt_reg = str(tgt.get("region_id", "") or f"region_{int(tgt.get('region_index', 0) or 0)}")
                if src_reg not in region_by_id or tgt_reg not in region_by_id:
                    return local_accepted, local_rejections
                kpaths = self._k_region_paths(region_adjacency, src_reg, tgt_reg, k=min(2, top_k))

                T_cost_local = T_cost_by_goal.get(tgt_pt)
                T_speed_local = T_speed_by_goal.get(tgt_pt)

                for kidx, seq in enumerate(kpaths, start=1):
                    pts: List[Tuple[int, int]] = [src_pt]
                    for a_id, b_id in zip(seq, seq[1:]):
                        pts.append(self._region_portal_point(lm, region_by_id, a_id, b_id, dilation_px=2))
                    pts.append(tgt_pt)
                    # Obstacle mask for validity + motion metrics still
                    # excludes src/tgt (they're the path endpoints so
                    # their own pixels shouldn't count as "hitting an
                    # obstacle"). This copy lives only in this task.
                    obs = all_obs.copy()
                    if src_id in obj_mask_by_id:
                        obs &= ~obj_mask_by_id[src_id]
                    if tgt_id in obj_mask_by_id:
                        obs &= ~obj_mask_by_id[tgt_id]
                    polyline_geodesic_2d: Optional[List[List[float]]] = None
                    polyline_geodesic_alternates_2d: List[List[List[float]]] = []
                    if use_refine:
                        nref = int(getattr(cfg, "path_refine_num_points", 96)) if cfg else 96
                        # A* replacement — descend on the cached T_cost
                        # (globally optimal geodesic on the semantic
                        # cost field, no grid search).
                        astar_pts: List[Tuple[int, int]] = []
                        if T_cost_local is not None:
                            refined = _fmm_backtrace_from_T(T_cost_local, src_pt)
                            if len(refined) >= 2:
                                astar_pts = self._resample_polyline(refined, max(24, nref))
                        geo_primary: List[Tuple[int, int]] = []
                        if use_geo and T_speed_local is not None:
                            # k-diverse Dijkstra replacement — descend
                            # on the cached T_speed and inflate the
                            # visited footprint between descents.
                            gpaths = _fmm_k_diverse_from_T(
                                T_speed_local,
                                src_pt,
                                k=max(1, min(geo_k, top_k)),
                                edge_penalty=geo_pen,
                            )
                            if gpaths:
                                geo_primary = self._resample_polyline(gpaths[0], max(24, nref))
                                for gp in gpaths[1:]:
                                    if len(gp) >= 2:
                                        polyline_geodesic_alternates_2d.append(
                                            [list(map(float, xy)) for xy in self._resample_polyline(gp, max(24, nref))]
                                        )
                        if geo_replace and len(geo_primary) >= 2:
                            pts = geo_primary
                        elif len(astar_pts) >= 2:
                            pts = astar_pts
                        elif len(geo_primary) >= 2:
                            pts = geo_primary
                        if len(geo_primary) >= 2:
                            polyline_geodesic_2d = [list(map(float, xy)) for xy in geo_primary]
                    v = self._polyline_is_valid(
                        pts, w, h,
                        feasible=feasible, obstacles=obs,
                        invalid_ratio_max=inv_max, max_turn_deg=max_turn,
                    )
                    pid = f"opath_{path_stem}_{src_id}_to_{tgt_id}_k{str(kidx).zfill(2)}"
                    tgt_name = self._display_name_for_object(tgt)
                    base: Dict[str, Any] = {
                        "path_id": pid,
                        "path_level": "object",
                        "path_type": "object_to_object",
                        "source_entity": {"type": "object", "id": src_id, "name": src_name, "display_label": self._display_label(src_name, src_id), "start_uv": list(src_pt)},
                        "target_entity": {"type": "object", "id": tgt_id, "name": tgt_name, "display_label": self._display_label(tgt_name, tgt_id), "goal_uv": list(tgt_pt)},
                        "regions_traversed": list(seq),
                        "polyline_2d": [list(p) for p in pts],
                        "constraints_applied": {"image_bounds_enforced": True, "region_adjacency_enforced": True, "occupancy_avoidance_enforced": True},
                        "validity_checks": v,
                        "pair_proposal_score": float(prop.get("proposal_score", 0.0)),
                    }
                    if polyline_geodesic_2d:
                        base["polyline_geodesic_2d"] = polyline_geodesic_2d
                        base["trav_meta"] = dict(trav_meta)
                    if polyline_geodesic_alternates_2d:
                        base["polyline_geodesic_alternates_2d"] = polyline_geodesic_alternates_2d
                    if not v.get("ok", False):
                        local_rejections.append({**base, "rejected_reason": v.get("reason", "invalid")})
                        continue
                    conf = max(0.0, 1.0 - float(v.get("invalid_ratio", 0.0)) - 0.005 * float(v.get("max_turn_deg", 0.0)))
                    valid_xy = [(x, y) for (x, y) in pts if 0 <= x < w and 0 <= y < h]
                    if valid_xy:
                        align_vals = [float(cost_map[y, x]) for (x, y) in valid_xy]
                        image_align = float(1.0 - np.mean(align_vals))
                    else:
                        image_align = 0.5
                    base["scores"] = {
                        "geometric_feasibility": float(conf),
                        "depth_consistency": float(conf),
                        "relation_consistency": 0.6,
                        "semantic_plausibility": 0.6,
                        "image_alignment_score": image_align,
                        "overall_confidence": float(conf),
                    }
                    mm = self._path_motion_metrics(pts, lm, obs, src, tgt)
                    base["motion_metrics"] = mm
                    base["trajectory_type"] = str(mm.get("trajectory_type", "traverse"))
                    base["is_motion_primary"] = bool(float(mm.get("motion_primary_score", 0.0)) >= motion_primary_gate_pre)
                    base["scores"]["motion_primary_score"] = float(mm.get("motion_primary_score", 0.0))
                    base["scores"]["depth_alignment_score"] = float(mm.get("depth_alignment_score", 0.0))
                    if base["scores"]["overall_confidence"] < min_conf:
                        local_rejections.append({**base, "rejected_reason": "low_confidence"})
                        continue
                    local_accepted.append(base)
                return local_accepted, local_rejections

            # Execute the pair tasks in a bounded thread pool. M2 has
            # 4 performance + 4 efficiency cores; 4 workers saturates
            # the performance cluster without contending for the MPS
            # device (which is idle during path export anyway).
            _pair_workers = max(1, int(getattr(cfg, "path_pair_max_workers", 4)) if cfg else 4)
            _pair_results: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
            if _pair_workers > 1 and len(pair_proposals) > 1:
                with _ThreadPoolExecutor(max_workers=_pair_workers, thread_name_prefix="opair") as _pair_pool:
                    _futs = [_pair_pool.submit(_process_object_pair, p) for p in pair_proposals]
                    for _fi, _fut in enumerate(_futs, start=1):
                        _pair_results.append(_fut.result())
                        if (_fi % 10 == 0) or (_fi == _op_total):
                            _elapsed = time.perf_counter() - _op_t0
                            print(f"  [Paths] object pairs {_fi}/{_op_total} ({_elapsed:.1f}s)")
            else:
                for _op_idx, prop in enumerate(pair_proposals, start=1):
                    _pair_results.append(_process_object_pair(prop))
                    if (_op_idx % 10 == 0) or (_op_idx == _op_total):
                        _elapsed = time.perf_counter() - _op_t0
                        print(f"  [Paths] object pairs {_op_idx}/{_op_total} ({_elapsed:.1f}s)")

            # Merge task results in proposal order, preserving the
            # legacy ``max_cand`` behaviour (stop appending once we
            # exceed the cap — we still spent the CPU on the excess
            # tasks but the *output* stays identical in length).
            for _accepted, _rejected in _pair_results:
                for _p in _accepted:
                    if len(paths) >= max_cand:
                        break
                    paths.append(_p)
                for _r in _rejected:
                    rejections.append(_r)

        # --- Mask paths ---
        if enable_mask:
            _mp_t0 = time.perf_counter()
            print(f"  [Paths] mask contour/axis export for {len(objs)} objects...")
            for _mp_idx, o in enumerate(objs, start=1):
                if len(paths) >= max_cand:
                    break
                oid = str(o.get("id", ""))
                m = obj_mask_by_id.get(oid, None)
                if m is None:
                    continue

                # --- Elongation gate (Phase 1.4) ---
                # Round / compact objects have no meaningful axis or contour traverse
                # path — emit an interaction_pulse marker instead and skip boundary paths.
                _ys_g, _xs_g = np.where(m)
                if _ys_g.size > 0:
                    _mh = int(_ys_g.max()) - int(_ys_g.min()) + 1
                    _mw = int(_xs_g.max()) - int(_xs_g.min()) + 1
                    _aspect = max(_mh, _mw) / max(1, min(_mh, _mw))
                else:
                    _aspect = 1.0
                if _aspect < 1.5:
                    _name_g = self._display_name_for_object(o)
                    _cx_g = int(_xs_g.mean()) if _xs_g.size > 0 else 0
                    _cy_g = int(_ys_g.mean()) if _ys_g.size > 0 else 0
                    paths.append({
                        "path_id": f"mpath_{path_stem}_{oid}_interaction_pulse_00",
                        "path_level": "mask",
                        "path_type": "interaction_pulse",
                        "source_entity": {"type": "mask", "id": oid, "name": _name_g, "display_label": self._display_label(_name_g, oid), "start_uv": [_cx_g, _cy_g]},
                        "target_entity": {"type": "mask", "id": oid, "name": _name_g, "display_label": self._display_label(_name_g, oid), "goal_uv": [_cx_g, _cy_g]},
                        "regions_traversed": [str(o.get("region_id", ""))] if o.get("region_id") else [],
                        "polyline_2d": [[_cx_g, _cy_g]],
                        "aspect_ratio": _aspect,
                        "is_motion_primary": False,
                        "trajectory_type": "interaction_pulse",
                        "scores": {"overall_confidence": 0.5, "geometric_feasibility": 0.5},
                    })
                    continue

                # contour path (downsampled)
                m8 = (m.astype(np.uint8) * 255)
                contours, _hier = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    cnt = max(contours, key=lambda c: c.shape[0])
                    step = max(1, int(cnt.shape[0] / 200))
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt[::step]]
                    if len(pts) >= 2:
                        v = self._polyline_is_valid(pts, w, h, feasible=None, obstacles=None, invalid_ratio_max=inv_max, max_turn_deg=180.0)
                        pid = f"mpath_{path_stem}_{oid}_contour_cw_00"
                        name = self._display_name_for_object(o)
                        base = {
                            "path_id": pid,
                            "path_level": "mask",
                            "path_type": "mask_contour",
                            "source_entity": {"type": "mask", "id": oid, "name": name, "display_label": self._display_label(name, oid), "start_uv": list(pts[0])},
                            "target_entity": {"type": "mask", "id": oid, "name": name, "display_label": self._display_label(name, oid), "goal_uv": list(pts[-1])},
                            "regions_traversed": [str(o.get("region_id", ""))] if o.get("region_id") else [],
                            "polyline_2d": [list(p) for p in pts],
                            "constraints_applied": {"image_bounds_enforced": True, "mask_containment_enforced": True},
                            "validity_checks": v,
                            "is_motion_primary": False,
                            "trajectory_type": "contour_diagnostic",
                        }
                        if v.get("ok", False):
                            mm = self._path_motion_metrics(pts, lm, m, o, o)
                            base["motion_metrics"] = mm
                            mps = float(mm.get("motion_primary_score", 0.0))
                            base_conf = float(max(0.0, min(1.0, 0.40 + 0.60 * mps)))
                            base["scores"] = {
                                "geometric_feasibility": base_conf,
                                "depth_consistency": float(max(0.0, min(1.0, 0.45 + 0.55 * float(mm.get("depth_alignment_score", 0.0))))),
                                "relation_consistency": float(max(0.0, min(1.0, 0.35 + 0.65 * mps))),
                                "semantic_plausibility": float(max(0.0, min(1.0, 0.30 + 0.70 * mps))),
                                "overall_confidence": base_conf,
                                "motion_primary_score": mps,
                            }
                            if base["scores"]["overall_confidence"] >= min_conf:
                                paths.append(base)
                            else:
                                rejections.append({**base, "rejected_reason": "low_confidence"})
                        else:
                            rejections.append({**base, "rejected_reason": v.get("reason", "invalid")})

                # Medial axis interior path (Phase 1.3) — distance-transform skeleton.
                # Replaces the PCA straight-line approach: the DT skeleton stays inside
                # the mask even for concave or irregular shapes.
                ys, xs = np.where(m)
                if len(xs) > 50:
                    _dt = cv2.distanceTransform(m.astype(np.uint8) * 255, cv2.DIST_L2, 5)
                    _dt_masked = np.where(m, _dt, 0.0)
                    # Find two endpoints: the two points on the skeleton that are
                    # farthest apart in terms of distance-transform value, biased
                    # toward the mask extremes along the principal axis.
                    _iy_pk, _ix_pk = np.unravel_index(int(np.argmax(_dt_masked)), _dt_masked.shape)
                    # Far-end: among the top-5 % DT pixels, pick the one farthest from peak.
                    _thr = float(np.percentile(_dt_masked[m], 95.0)) if m.any() else 0.0
                    _cands_y, _cands_x = np.where(_dt_masked >= max(1.0, _thr))
                    if _cands_x.size > 1:
                        _dists = ((_cands_x.astype(np.float32) - _ix_pk) ** 2 +
                                  (_cands_y.astype(np.float32) - _iy_pk) ** 2)
                        _far_idx = int(np.argmax(_dists))
                        _iy_far, _ix_far = int(_cands_y[_far_idx]), int(_cands_x[_far_idx])
                    else:
                        _iy_far, _ix_far = _iy_pk, _ix_pk
                    p0 = (int(_ix_pk), int(_iy_pk))
                    p1 = (int(_ix_far), int(_iy_far))
                    line_pts = self._sample_line(p0, p1, n=40)
                    # Snap each sample to nearest mask-interior pixel so the path
                    # stays inside even across concavities.
                    line_pts = [(x, y) for (x, y) in line_pts if 0 <= x < w and 0 <= y < h and bool(m[y, x])]
                    if len(line_pts) >= 2:
                        v = self._polyline_is_valid(line_pts, w, h, feasible=None, obstacles=None, invalid_ratio_max=inv_max, max_turn_deg=180.0)
                        pid = f"mpath_{path_stem}_{oid}_medial_axis_00"
                        name = self._display_name_for_object(o)
                        base = {
                            "path_id": pid,
                            "path_level": "mask",
                            "path_type": "mask_medial_axis",
                            "source_entity": {"type": "mask", "id": oid, "name": name, "display_label": self._display_label(name, oid), "start_uv": list(line_pts[0])},
                            "target_entity": {"type": "mask", "id": oid, "name": name, "display_label": self._display_label(name, oid), "goal_uv": list(line_pts[-1])},
                            "regions_traversed": [str(o.get("region_id", ""))] if o.get("region_id") else [],
                            "polyline_2d": [list(p) for p in line_pts],
                            "constraints_applied": {"image_bounds_enforced": True, "mask_containment_enforced": True},
                            "validity_checks": v,
                            "is_motion_primary": False,
                            "trajectory_type": "axis_diagnostic",
                        }
                        if v.get("ok", False):
                            mm = self._path_motion_metrics(line_pts, lm, m, o, o)
                            base["motion_metrics"] = mm
                            mps = float(mm.get("motion_primary_score", 0.0))
                            base_conf = float(max(0.0, min(1.0, 0.45 + 0.55 * mps)))
                            base["scores"] = {
                                "geometric_feasibility": base_conf,
                                "depth_consistency": float(max(0.0, min(1.0, 0.45 + 0.55 * float(mm.get("depth_alignment_score", 0.0))))),
                                "relation_consistency": float(max(0.0, min(1.0, 0.35 + 0.65 * mps))),
                                "semantic_plausibility": float(max(0.0, min(1.0, 0.35 + 0.65 * mps))),
                                "overall_confidence": base_conf,
                                "motion_primary_score": mps,
                            }
                            if base["scores"]["overall_confidence"] >= min_conf:
                                paths.append(base)
                            else:
                                rejections.append({**base, "rejected_reason": "low_confidence"})
                        else:
                            rejections.append({**base, "rejected_reason": v.get("reason", "invalid")})
            _mp_elapsed = time.perf_counter() - _mp_t0
            print(f"  [Paths] mask exports done ({_mp_elapsed:.1f}s); total paths so far: {len(paths)}")

        # --- Cross-family semantic-physical path coverage ---
        cross_enabled = bool(getattr(cfg, "path_cross_family_enabled", True)) if cfg else True
        cross_top_k = int(getattr(cfg, "path_cross_family_top_k", 2)) if cfg else 2
        if cross_enabled and (enable_object or enable_region or enable_mask):
            _cf_t0 = time.perf_counter()
            cross_added = 0
            obj_by_id = {str(o.get("id", "")): o for o in objs if str(o.get("id", "")).strip()}
            region_ids = [str(r.get("id", "")) for r in regions_meta if str(r.get("id", "")).strip() and str(r.get("id", "")) in region_by_id]

            def _mk_cross_path(
                *,
                pid: str,
                path_type: str,
                src_ent: Dict[str, Any],
                tgt_ent: Dict[str, Any],
                seq: List[str],
                pts: List[Tuple[int, int]],
                src_obj: Optional[Dict[str, Any]],
                tgt_obj: Optional[Dict[str, Any]],
                fallback: bool,
            ) -> Dict[str, Any]:
                mm_obs = np.zeros_like(all_obs)
                if src_obj is not None and str(src_ent.get("type", "")) == "object":
                    mm_obs = all_obs
                mm = self._path_motion_metrics(pts, lm, mm_obs, src_obj, tgt_obj)
                mps = float(mm.get("motion_primary_score", 0.0))
                conf = max(0.10, min(1.0, 0.40 + 0.60 * mps))
                return {
                    "path_id": pid,
                    "path_level": "cross",
                    "path_type": path_type,
                    "source_entity": src_ent,
                    "target_entity": tgt_ent,
                    "regions_traversed": list(seq),
                    "polyline_2d": [list(p) for p in pts],
                    "constraints_applied": {
                        "image_bounds_enforced": True,
                        "region_adjacency_enforced": True,
                        "semantic_physical_energy_enabled": True,
                        "cross_family_fallback": bool(fallback),
                    },
                    "validity_checks": {"ok": True, "invalid_ratio": 0.0, "max_turn_deg": 0.0},
                    "scores": {
                        "geometric_feasibility": conf,
                        "depth_consistency": conf,
                        "relation_consistency": 0.5,
                        "semantic_plausibility": 0.5,
                        "overall_confidence": conf,
                        "motion_primary_score": mps,
                    },
                    "motion_metrics": mm,
                    "generation_source": "direct",
                    "generation_detail": "fallback_geometry" if fallback else "direct_geometry",
                    "family_confidence": conf,
                }

            # Region<->Object and Region<->Mask families
            for o in objs:
                oid = str(o.get("id", ""))
                if not oid:
                    continue
                src_reg = str(o.get("region_id", "") or f"region_{int(o.get('region_index', 0) or 0)}")
                if src_reg not in region_by_id:
                    continue
                src_uv = o.get("mask_centroid_2d", [0, 0])
                src_pt = (int(src_uv[0]), int(src_uv[1]))
                src_name = self._display_name_for_object(o)
                mask_name = self._display_label(src_name, oid)
                for rid in region_ids[: max(1, cross_top_k)]:
                    if rid == src_reg:
                        continue
                    seqs = self._k_region_paths(region_adjacency, src_reg, rid, k=1)
                    if not seqs:
                        continue
                    seq = seqs[0]
                    pts_or: List[Tuple[int, int]] = [src_pt]
                    for a_id, b_id in zip(seq, seq[1:]):
                        pts_or.append(self._region_portal_point(lm, region_by_id, a_id, b_id, dilation_px=2))
                    rc = region_by_id[rid].get("centroid_2d_px", [0, 0])
                    goal_pt = (int(rc[0]), int(rc[1]))
                    pts_or.append(goal_pt)
                    v_or = self._polyline_is_valid(pts_or, w, h, feasible=feasible, obstacles=all_obs, invalid_ratio_max=inv_max, max_turn_deg=max_turn)
                    fallback = not bool(v_or.get("ok", False))
                    src_region_name = self._display_name_for_region(region_by_id.get(rid, {}))
                    p_or = _mk_cross_path(
                        pid=f"xor_{path_stem}_{oid}_to_{rid}",
                        path_type="object_to_region",
                        src_ent={"type": "object", "id": oid, "name": src_name, "display_label": self._display_label(src_name, oid), "start_uv": list(src_pt)},
                        tgt_ent={"type": "region", "id": rid, "name": src_region_name, "display_label": self._display_label(src_region_name, rid), "goal_uv": list(goal_pt)},
                        seq=seq,
                        pts=pts_or,
                        src_obj=o,
                        tgt_obj=None,
                        fallback=fallback,
                    )
                    paths.append(p_or)
                    cross_added += 1
                    if len(paths) >= max_cand:
                        break
                    p_mr = _mk_cross_path(
                        pid=f"xmr_{path_stem}_{oid}_to_{rid}",
                        path_type="mask_to_region",
                        src_ent={"type": "mask", "id": oid, "name": src_name, "display_label": mask_name, "start_uv": list(src_pt)},
                        tgt_ent={"type": "region", "id": rid, "name": src_region_name, "display_label": self._display_label(src_region_name, rid), "goal_uv": list(goal_pt)},
                        seq=seq,
                        pts=pts_or,
                        src_obj=o,
                        tgt_obj=None,
                        fallback=fallback,
                    )
                    paths.append(p_mr)
                    cross_added += 1
                    if len(paths) >= max_cand:
                        break

                    pts_ro = list(reversed(pts_or))
                    p_ro = _mk_cross_path(
                        pid=f"xro_{path_stem}_{rid}_to_{oid}",
                        path_type="region_to_object",
                        src_ent={"type": "region", "id": rid, "name": src_region_name, "display_label": self._display_label(src_region_name, rid), "start_uv": list(goal_pt)},
                        tgt_ent={"type": "object", "id": oid, "name": src_name, "display_label": self._display_label(src_name, oid), "goal_uv": list(src_pt)},
                        seq=list(reversed(seq)),
                        pts=pts_ro,
                        src_obj=None,
                        tgt_obj=o,
                        fallback=fallback,
                    )
                    paths.append(p_ro)
                    cross_added += 1
                    if len(paths) >= max_cand:
                        break
                    p_rm = _mk_cross_path(
                        pid=f"xrm_{path_stem}_{rid}_to_{oid}",
                        path_type="region_to_mask",
                        src_ent={"type": "region", "id": rid, "name": src_region_name, "display_label": self._display_label(src_region_name, rid), "start_uv": list(goal_pt)},
                        tgt_ent={"type": "mask", "id": oid, "name": src_name, "display_label": mask_name, "goal_uv": list(src_pt)},
                        seq=list(reversed(seq)),
                        pts=pts_ro,
                        src_obj=None,
                        tgt_obj=o,
                        fallback=fallback,
                    )
                    paths.append(p_rm)
                    cross_added += 1
                    if len(paths) >= max_cand:
                        break
                if len(paths) >= max_cand:
                    break

            # Direct Object/Mask pair families (including mask_to_mask)
            if len(paths) < max_cand:
                for prop in pair_proposals:
                    sid = str(prop.get("src_id", ""))
                    tid = str(prop.get("tgt_id", ""))
                    src = obj_by_id.get(sid)
                    tgt = obj_by_id.get(tid)
                    if src is None or tgt is None:
                        continue
                    src_reg = str(src.get("region_id", "") or f"region_{int(src.get('region_index', 0) or 0)}")
                    tgt_reg = str(tgt.get("region_id", "") or f"region_{int(tgt.get('region_index', 0) or 0)}")
                    if src_reg not in region_by_id or tgt_reg not in region_by_id:
                        continue
                    seqs = self._k_region_paths(region_adjacency, src_reg, tgt_reg, k=1)
                    if not seqs:
                        continue
                    seq = seqs[0]
                    su = src.get("mask_centroid_2d", [0, 0])
                    tu = tgt.get("mask_centroid_2d", [0, 0])
                    spt = (int(su[0]), int(su[1]))
                    tpt = (int(tu[0]), int(tu[1]))
                    pts: List[Tuple[int, int]] = [spt]
                    for a_id, b_id in zip(seq, seq[1:]):
                        pts.append(self._region_portal_point(lm, region_by_id, a_id, b_id, dilation_px=2))
                    pts.append(tpt)
                    v = self._polyline_is_valid(pts, w, h, feasible=feasible, obstacles=all_obs, invalid_ratio_max=inv_max, max_turn_deg=max_turn)
                    fallback = not bool(v.get("ok", False))
                    sname = self._display_name_for_object(src)
                    tname = self._display_name_for_object(tgt)
                    cands = [
                        ("object_to_mask", {"type": "object", "id": sid, "name": sname, "display_label": self._display_label(sname, sid), "start_uv": list(spt)}, {"type": "mask", "id": tid, "name": tname, "display_label": self._display_label(tname, tid), "goal_uv": list(tpt)}),
                        ("mask_to_object", {"type": "mask", "id": sid, "name": sname, "display_label": self._display_label(sname, sid), "start_uv": list(spt)}, {"type": "object", "id": tid, "name": tname, "display_label": self._display_label(tname, tid), "goal_uv": list(tpt)}),
                        ("mask_to_mask", {"type": "mask", "id": sid, "name": sname, "display_label": self._display_label(sname, sid), "start_uv": list(spt)}, {"type": "mask", "id": tid, "name": tname, "display_label": self._display_label(tname, tid), "goal_uv": list(tpt)}),
                    ]
                    for idx, (ptype, sent, tent) in enumerate(cands, start=1):
                        p = _mk_cross_path(
                            pid=f"xmm_{path_stem}_{sid}_to_{tid}_{ptype}_{idx}",
                            path_type=ptype,
                            src_ent=sent,
                            tgt_ent=tent,
                            seq=seq,
                            pts=pts,
                            src_obj=src,
                            tgt_obj=tgt,
                            fallback=fallback,
                        )
                        paths.append(p)
                        cross_added += 1
                        if len(paths) >= max_cand:
                            break
                    if len(paths) >= max_cand:
                        break

            print(f"  [Paths] cross-family coverage added {cross_added} paths ({time.perf_counter() - _cf_t0:.1f}s)")
        # Bound family candidate explosion while preserving coverage.
        family_cap = int(getattr(cfg, "path_family_max_candidates", 12)) if cfg else 12
        if family_cap > 0 and len(paths) > family_cap:
            fam_keep: Dict[str, int] = {}
            trimmed: List[Dict[str, Any]] = []
            for p in paths:
                src_t = str(((p.get("source_entity") or {}).get("type", "")) or "unknown")
                tgt_t = str(((p.get("target_entity") or {}).get("type", "")) or "unknown")
                fam = f"{src_t}_to_{tgt_t}"
                seen = int(fam_keep.get(fam, 0))
                if seen >= family_cap:
                    continue
                fam_keep[fam] = seen + 1
                trimmed.append(p)
            paths = trimmed

        # Semantic layer + hybrid scoring/filtering
        semantic_enabled = bool(getattr(cfg, "path_semantic_enabled", True)) if cfg else True
        semantic_layer: Dict[str, Any] = {
            "semantic_enabled": False,
            "entities": [],
            "actors": [],
            "region_affordances": [],
            "actor_intents": [],
        }
        if semantic_enabled:
            semantic_layer = self._build_semantic_layer(objs, regions_meta, relations)
        semantic_layer_path = paths_root_dir / "semantic_layer.json"
        self._write_json(semantic_layer, semantic_layer_path)
        semantic_layer_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/semantic_layer.json"

        wg = float(getattr(cfg, "path_score_weight_geometric", 0.40)) if cfg else 0.40
        ws = float(getattr(cfg, "path_score_weight_semantic", 0.30)) if cfg else 0.30
        wr = float(getattr(cfg, "path_score_weight_relation", 0.20)) if cfg else 0.20
        wa = float(getattr(cfg, "path_score_weight_action_fit", 0.10)) if cfg else 0.10
        wsum = max(1e-6, wg + ws + wr + wa)
        wg, ws, wr, wa = wg / wsum, ws / wsum, wr / wsum, wa / wsum
        sem_rejections: List[Dict[str, Any]] = []
        paths_all_valid: List[Dict[str, Any]] = [dict(p) for p in paths]
        paths_recommended: List[Dict[str, Any]] = []
        semantic_hard = bool(getattr(cfg, "path_semantic_hard_filter_enabled", True)) if cfg else True
        for p in paths_all_valid:
            if "generation_source" not in p:
                p["generation_source"] = "direct"
            p["family_confidence"] = float((p.get("scores") or {}).get("overall_confidence", 0.0))
        from scene_understanding.pathing.hybrid_scores import apply_hybrid_confidence_scores
        sem_cache: Dict[str, Dict[str, Any]] = {}
        sem_scores_raw: List[float] = []
        far_ratios_raw: List[float] = []
        obs_ratios_raw: List[float] = []
        walk_ratios_raw: List[float] = []
        motion_scores_raw: List[float] = []
        rel_blob = " ".join([json.dumps(r, ensure_ascii=False).lower() for r in (relations or [])])

        def _relation_guardrail(path_obj: Dict[str, Any]) -> Dict[str, Any]:
            s = path_obj.get("source_entity") or {}
            t = path_obj.get("target_entity") or {}
            sid = str(s.get("id", "")).lower()
            tid = str(t.get("id", "")).lower()
            pair_hit = bool(sid and tid and sid in rel_blob and tid in rel_blob)
            # Region routes often lack explicit relation tuples in current graph.
            st = str(s.get("type", ""))
            tt = str(t.get("type", ""))
            if st == "region" or tt == "region":
                required = False
            else:
                required = True
            penalty = 0.0 if pair_hit or not required else 0.18
            return {
                "relation_pair_found": pair_hit,
                "relation_required": required,
                "relation_guardrail_penalty": penalty,
            }

        for p in paths_all_valid:
            sem = self._path_semantic_evidence(p, semantic_layer, semantic_validity_floor=0.0) if semantic_enabled else {
                "semantic_validity_score": 1.0,
                "semantic_valid": True,
                "semantic_reasons": ["semantic disabled"],
                "affordance_trace": [],
            }
            sem_cache[str(p.get("path_id", ""))] = sem
            sem_scores_raw.append(float(sem.get("semantic_validity_score", 0.0)))
            trace = list(sem.get("affordance_trace", []) or [])
            ntrace = max(1, len(trace))
            far_ratios_raw.append(float(sum(1 for t in trace if str(t.get("affordance", "")) == "far_background")) / float(ntrace))
            obs_ratios_raw.append(float(sum(1 for t in trace if str(t.get("affordance", "")) == "obstacle")) / float(ntrace))
            walk_ratios_raw.append(float(sum(1 for t in trace if str(t.get("affordance", "")) in ("walkable", "interaction_zone"))) / float(ntrace))
            motion_scores_raw.append(float((p.get("motion_metrics") or {}).get("motion_primary_score", 0.0)))

        calibrated_policy = _calibrate_path_policy(
            cfg=cfg,
            paths=paths_all_valid,
            semantic_scores=sem_scores_raw,
            far_ratios=far_ratios_raw,
            obs_ratios=obs_ratios_raw,
            walk_ratios=walk_ratios_raw,
            motion_scores=motion_scores_raw,
        )
        min_conf = float(calibrated_policy.get("min_conf", min_conf))
        max_far = float(calibrated_policy.get("max_far", 1.0))
        max_obs = float(calibrated_policy.get("max_obs", 1.0))
        min_walk = float(calibrated_policy.get("min_walk", 0.0))
        semantic_floor = float(calibrated_policy.get("semantic_validity_floor", 0.0))
        motion_primary_threshold = float(calibrated_policy.get("motion_primary_threshold", motion_primary_gate_pre))

        for p in paths_all_valid:
            sem = sem_cache.get(str(p.get("path_id", "")), {})
            p["semantic_validity_score"] = float(sem.get("semantic_validity_score", 0.0))
            p["semantic_reasons"] = list(sem.get("semantic_reasons", []))
            p["affordance_trace"] = list(sem.get("affordance_trace", []))
            p["semantic_valid"] = bool(p["semantic_validity_score"] >= semantic_floor)
            apply_hybrid_confidence_scores(p, sem, wg=wg, ws=ws, wr=wr, wa=wa)
            rel_guard = _relation_guardrail(p)
            hybrid = float((p.get("scores") or {}).get("hybrid_overall", 0.0))
            trace = list(p.get("affordance_trace", []))
            ntrace = max(1, len(trace))
            far_ratio = float(sum(1 for t in trace if str(t.get("affordance", "")) == "far_background")) / float(ntrace)
            obs_ratio = float(sum(1 for t in trace if str(t.get("affordance", "")) == "obstacle")) / float(ntrace)
            walk_ratio = float(sum(1 for t in trace if str(t.get("affordance", "")) in ("walkable", "interaction_zone"))) / float(ntrace)
            p["diagnostics"] = {
                "far_background_ratio": far_ratio,
                "obstacle_ratio": obs_ratio,
                "walkable_ratio": walk_ratio,
                "curvature_deg": float((p.get("validity_checks") or {}).get("max_turn_deg", 0.0)),
                "relation_pair_found": bool(rel_guard.get("relation_pair_found", False)),
                "relation_guardrail_penalty": float(rel_guard.get("relation_guardrail_penalty", 0.0)),
                "calibrated_policy": dict(calibrated_policy),
            }
            naturalness = float(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - (
                            0.35 * far_ratio
                            + 0.35 * obs_ratio
                            + 0.15 * float(rel_guard.get("relation_guardrail_penalty", 0.0))
                            + 0.15 * max(0.0, 1.0 - walk_ratio)
                        ),
                    ),
                )
            )
            p["diagnostics"]["naturalness_score"] = naturalness
            motion_score = float((p.get("motion_metrics") or {}).get("motion_primary_score", 0.0))
            p["is_motion_primary"] = bool(motion_score >= motion_primary_threshold)
            if rel_guard.get("relation_guardrail_penalty", 0.0) > 0.0:
                p["scores"]["overall_confidence"] = float(
                    max(0.0, p["scores"]["overall_confidence"] - float(rel_guard["relation_guardrail_penalty"]))
                )
            suppressed_reason = ""
            if semantic_hard:
                if far_ratio > max_far:
                    suppressed_reason = "semantic_far_background_dominant"
                if obs_ratio > max_obs:
                    suppressed_reason = suppressed_reason or "semantic_obstacle_dominant"
                if walk_ratio < min_walk:
                    suppressed_reason = suppressed_reason or "semantic_low_walkable_support"
            if not bool(p.get("is_motion_primary", False)) and str(p.get("path_level", "")) == "mask":
                # Keep diagnostic mask traces in JSON but reduce their ranking influence.
                p["scores"]["overall_confidence"] = float(min(p["scores"]["overall_confidence"], min_conf))
            if semantic_enabled and (not p["semantic_valid"] or hybrid < min_conf):
                suppressed_reason = suppressed_reason or "semantic_implausible_or_low_hybrid_score"

            p["suppressed"] = bool(suppressed_reason)
            p["suppressed_reason"] = str(suppressed_reason)
            if p["suppressed"]:
                sem_rejections.append(dict(p))
            else:
                paths_recommended.append(p)

        if bool(getattr(cfg, "path_semantic_physical_energy_enabled", True)) if cfg else True:
            # Semantic-physical energy bookkeeping for ranking explainability.
            for p in paths_all_valid:
                sc = p.get("scores") or {}
                dg = p.get("diagnostics") or {}
                mm = p.get("motion_metrics") or {}
                e_geo = 1.0 - float(sc.get("geometric_feasibility", 0.0))
                e_sem = 1.0 - float(p.get("semantic_validity_score", 0.0))
                e_rel = 1.0 - float(sc.get("relation_consistency", 0.0))
                e_depth = 1.0 - float(sc.get("depth_alignment_score", mm.get("depth_alignment_score", 0.0)))
                e_motion = 1.0 - float(mm.get("motion_primary_score", 0.0))
                e_unc = float(max(0.0, dg.get("obstacle_ratio", 0.0) + dg.get("far_background_ratio", 0.0) - dg.get("walkable_ratio", 0.0)))
                energy_total = float((wg * e_geo) + (ws * e_sem) + (wr * e_rel) + (wa * e_motion) + 0.15 * e_depth + 0.10 * e_unc)
                p["energy_terms"] = {
                    "E_geo": e_geo,
                    "E_sem": e_sem,
                    "E_rel": e_rel,
                    "E_depth": e_depth,
                    "E_motion": e_motion,
                    "E_unc": e_unc,
                    "E_total": energy_total,
                }
                src_t = str(((p.get("source_entity") or {}).get("type", "")) or "unknown")
                tgt_t = str(((p.get("target_entity") or {}).get("type", "")) or "unknown")
                p["path_family"] = f"{src_t}_to_{tgt_t}"
                p.setdefault("constraints_applied", {})
                p["constraints_applied"]["semantic_physical_energy_enabled"] = True
                p["constraints_applied"]["semantic_hard_filter_enabled"] = bool(semantic_hard)
                p["scores"]["energy_total"] = energy_total
        paths = paths_all_valid
        rejections.extend(sem_rejections)

        # Attach descriptions (local or OpenRouter)
        export_desc = bool(getattr(cfg, "path_export_descriptions", True)) if cfg else True
        desc_backend = str(getattr(cfg, "path_description_backend", "local")) if cfg else "local"
        sem_llm_enabled = bool(getattr(cfg, "path_semantic_llm_enabled", True)) if cfg else True
        sem_llm_top_k = int(getattr(cfg, "path_semantic_llm_top_k", 10)) if cfg else 10
        descriptions_by_id: Dict[str, Any] = {}
        if export_desc:
            ranked_for_llm = sorted(
                paths,
                key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
                reverse=True,
            )
            llm_allow_ids = {str(p.get("path_id", "")) for p in ranked_for_llm[: max(0, sem_llm_top_k)]}
            for p in paths:
                local = self._local_path_description(p)
                out = local
                if desc_backend.strip().lower() == "openrouter":
                    remote = self._openrouter_describe_path(p)
                    if remote:
                        out = remote
                sem_enrich = self._semantic_enrichment_for_path(
                    p,
                    {
                        "semantic_validity_score": p.get("semantic_validity_score", 0.0),
                        "semantic_reasons": p.get("semantic_reasons", []),
                    },
                    allow_llm=bool(sem_llm_enabled and desc_backend.strip().lower() == "openrouter" and str(p.get("path_id", "")) in llm_allow_ids),
                )
                p["semantic_source"] = str(sem_enrich.get("semantic_source", "local"))
                p["llm_status"] = str(sem_enrich.get("llm_status", "disabled"))
                p["llm_error"] = str(sem_enrich.get("llm_error", ""))
                p["description"] = str(out.get("summary", ""))
                descriptions_by_id[p["path_id"]] = {
                    **out,
                    "path_id": p["path_id"],
                    "source": p.get("source_entity", {}),
                    "target": p.get("target_entity", {}),
                    "regions_traversed": p.get("regions_traversed", []),
                    "scores": p.get("scores", {}),
                    "validity_checks": p.get("validity_checks", {}),
                    "grounded_inputs": {
                        "polyline_2d": p.get("polyline_2d", []),
                        "constraints_applied": p.get("constraints_applied", {}),
                    },
                    "semantic_source": p.get("semantic_source", "local"),
                    "llm_status": p.get("llm_status", "disabled"),
                    "llm_error": p.get("llm_error", ""),
                    "semantic_summary": sem_enrich.get("summary", ""),
                }

        # Assign stable numeric IDs (1..N) by global confidence rank.
        paths_sorted_global = sorted(
            paths,
            key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=True,
        )
        path_num_by_id: Dict[str, int] = {}
        for idx, p in enumerate(paths_sorted_global, start=1):
            pid = str(p.get("path_id", ""))
            if not pid:
                continue
            path_num_by_id[pid] = idx
            p["path_num"] = idx
            if pid in descriptions_by_id:
                descriptions_by_id[pid]["path_num"] = idx

        # Per-path individual PNG exports (top-k per path_level)
        export_individual = bool(getattr(cfg, "path_export_individual_images", True)) if cfg else True
        if export_individual:
            indiv_top_k = int(getattr(cfg, "path_individual_top_k_per_level", 10)) if cfg else 10
            selected_by_level = self._select_top_paths_by_level(paths, indiv_top_k)
            text_max_chars = int(getattr(cfg, "path_text_max_chars", 46)) if cfg else 46
            text_scale = float(getattr(cfg, "path_text_scale", 0.45)) if cfg else 0.45
            show_dir_text = bool(getattr(cfg, "path_direction_text", True)) if cfg else True
            show_label_text = bool(getattr(cfg, "path_label_text", True)) if cfg else True

            # deterministic colors from path_id hash
            def _color(pid: str) -> Tuple[int, int, int]:
                hsh = abs(hash(pid))
                return (int(50 + (hsh % 205)), int(50 + ((hsh // 7) % 205)), int(50 + ((hsh // 49) % 205)))

            for lvl, plist in selected_by_level.items():
                if lvl == "region":
                    out_dir = images_region_dir
                elif lvl == "object":
                    out_dir = images_object_dir
                else:
                    out_dir = images_mask_dir

                for p in plist:
                    pts = [tuple(map(int, xy)) for xy in (p.get("polyline_2d") or [])]
                    if len(pts) < 2:
                        continue
                    # Always overlay on the input image for readability.
                    canvas_bgr = img_bgr.copy()
                    pid = str(p.get("path_id", ""))
                    pnum = int(path_num_by_id.get(pid, 0))
                    src = (p.get("source_entity", {}) or {}) if isinstance(p.get("source_entity", {}), dict) else {}
                    tgt = (p.get("target_entity", {}) or {}) if isinstance(p.get("target_entity", {}), dict) else {}
                    src_disp = str(src.get("display_label") or src.get("name") or src.get("id") or "source")
                    tgt_disp = str(tgt.get("display_label") or tgt.get("name") or tgt.get("id") or "target")
                    sx, sy = pts[0]
                    gx, gy = pts[-1]
                    dx = float(gx - sx)
                    dy = float(gy - sy)
                    if abs(dx) >= abs(dy):
                        screen_dir = "right" if dx >= 0 else "left"
                    else:
                        screen_dir = "down" if dy >= 0 else "up"

                    col = _color(pid)
                    self._tapered_polyline_draw(
                        img_bgr=canvas_bgr,
                        pts=pts,
                        color_bgr=col,
                        start_w=int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8,
                        end_w=int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2,
                        alpha_start=float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95,
                        alpha_end=float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35,
                    )
                    cv2.circle(canvas_bgr, pts[0], 5, (0, 220, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas_bgr, pts[-1], 5, (0, 0, 220), -1, lineType=cv2.LINE_AA)
                    cv2.arrowedLine(canvas_bgr, pts[0], pts[-1], col, 2, cv2.LINE_AA, tipLength=0.12)

                    summary_txt = ""
                    if pid in descriptions_by_id:
                        summary_txt = str(descriptions_by_id[pid].get("summary", "")).strip()
                    if not summary_txt:
                        summary_txt = f"Path from {src_disp} to {tgt_disp} moves mainly {screen_dir}."
                    summary_txt = self._truncate_text(summary_txt, 120)

                    title_line = f"Path {pnum}: {pid}"
                    meta_line = f"{src_disp} -> {tgt_disp} | motion: {screen_dir}"
                    if show_label_text:
                        cv2.putText(canvas_bgr, self._truncate_text(title_line, 110), (12, 22), cv2.FONT_HERSHEY_SIMPLEX, text_scale, col, 1, cv2.LINE_AA)
                        cv2.putText(canvas_bgr, self._truncate_text(meta_line, 110), (12, 40), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)
                    if show_dir_text:
                        cv2.putText(canvas_bgr, summary_txt, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)

                    out_path = out_dir / f"path_{pid}.png"
                    cv2.imwrite(str(out_path), canvas_bgr)

        # Context composites: top-K overall paths on image + regions + objects.
        export_ctx = bool(getattr(cfg, "path_export_context_composites", True)) if cfg else True
        if cpu_benchmark_mode and cpu_disable_heavy_visuals:
            export_ctx = False
        triplet_context_rel_root = ""
        triplet_manifest_rel = ""
        if export_ctx:
            ctx_top_k = int(getattr(cfg, "path_context_top_k", 5)) if cfg else 5
            ctx_top_k = max(0, ctx_top_k)
            if ctx_top_k > 0:
                ctx_dir = images_root_dir / "context_top"
                ctx_dir.mkdir(parents=True, exist_ok=True)
                # rank across all levels
                ranked = sorted(
                    paths,
                    key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
                    reverse=True,
                )[:ctx_top_k]

                def _color(pid: str) -> Tuple[int, int, int]:
                    hsh = abs(hash(pid))
                    return (int(50 + (hsh % 205)), int(50 + ((hsh // 7) % 205)), int(50 + ((hsh // 49) % 205)))

                from scene_understanding.pathing.path_canvas import (
                    draw_direction_heads as _draw_direction_heads,
                    sample_depth_along_polyline as _sample_depth_along_polyline,
                    write_path_context_top5_png,
                )

                write_path_context_top5_png(
                    paths_root_dir=paths_root_dir,
                    img_bgr=img_bgr,
                    lm=lm,
                    objs=objs,
                    paths=paths,
                    cfg=cfg,
                    metric_depth_m=metric_depth_m,
                )

                # individual context images per path
                for rank_idx, p in enumerate(ranked, start=1):
                    pid = str(p.get("path_id", ""))
                    pts = [tuple(map(int, xy)) for xy in (p.get("polyline_2d") or [])]
                    if len(pts) < 2:
                        continue
                    img = img_bgr.copy()
                    self._draw_regions_contours_bgr(img, lm)
                    self._draw_objects_boxes_bgr(img, objs, max_boxes=50)
                    col = _color(pid)
                    self._tapered_polyline_draw(
                        img,
                        pts,
                        col,
                        start_w=int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8,
                        end_w=int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2,
                        alpha_start=float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95,
                        alpha_end=float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35,
                        depth_values=_sample_depth_along_polyline(pts, metric_depth_m),
                    )
                    sx, sy = pts[0]
                    _draw_direction_heads(img, pts, col, thickness=2, tip_len=0.12)
                    # label with summary if available
                    title = ""
                    if export_desc and pid in descriptions_by_id:
                        title = str(descriptions_by_id[pid].get("title", "")) or str(descriptions_by_id[pid].get("summary", ""))
                    lbl = pid if not title else f"{pid} | {title}"
                    cv2.putText(img, self._truncate_text(lbl, 90), (max(6, sx + 8), max(16, sy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                    outp = ctx_dir / f"{str(rank_idx).zfill(2)}_{pid}.png"
                    cv2.imwrite(str(outp), img)

            # Batch composites: all accepted paths in groups of 3 by confidence rank.
            export_triplet = bool(getattr(cfg, "path_export_triplet_context_composites", True)) if cfg else True
            if export_triplet:
                batch_size = int(getattr(cfg, "path_context_triplet_size", 3)) if cfg else 3
                batch_size = max(1, batch_size)
                ranked_all = sorted(
                    paths,
                    key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
                    reverse=True,
                )
                if ranked_all:
                    triplet_dir = images_root_dir / "context_triplets"
                    triplet_dir.mkdir(parents=True, exist_ok=True)
                    triplet_context_rel_root = f"scene_graph/{track_dir_name}/{path_stem}_paths/images/context_triplets"

                    triplet_manifest: Dict[str, Any] = {"batch_size": batch_size, "total_paths": len(ranked_all), "batches": []}
                    for start_idx in range(0, len(ranked_all), batch_size):
                        batch_paths = ranked_all[start_idx:start_idx + batch_size]
                        batch_num = (start_idx // batch_size) + 1
                        batch_img = img_bgr.copy()
                        self._draw_regions_contours_bgr(batch_img, lm)
                        self._draw_objects_boxes_bgr(batch_img, objs, max_boxes=50)

                        summary_lines: List[str] = []
                        for p in batch_paths:
                            pid = str(p.get("path_id", ""))
                            pts = [tuple(map(int, xy)) for xy in (p.get("polyline_2d") or [])]
                            if len(pts) < 2:
                                continue
                            col = _color(pid)
                            self._tapered_polyline_draw(
                                batch_img,
                                pts,
                                col,
                                start_w=int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8,
                                end_w=int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2,
                                alpha_start=float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95,
                                alpha_end=float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35,
                            )
                            cv2.arrowedLine(batch_img, pts[0], pts[-1], col, 2, cv2.LINE_AA, tipLength=0.12)
                            cv2.circle(batch_img, pts[0], 5, (0, 220, 0), -1, lineType=cv2.LINE_AA)
                            cv2.circle(batch_img, pts[-1], 5, (0, 0, 220), -1, lineType=cv2.LINE_AA)

                            src = (p.get("source_entity") or {}) if isinstance(p.get("source_entity", {}), dict) else {}
                            tgt = (p.get("target_entity") or {}) if isinstance(p.get("target_entity", {}), dict) else {}
                            src_lbl = str(src.get("display_label") or src.get("name") or src.get("id") or "source")
                            tgt_lbl = str(tgt.get("display_label") or tgt.get("name") or tgt.get("id") or "target")
                            regions_txt = ", ".join([str(r) for r in (p.get("regions_traversed") or [])[:4]])
                            if not regions_txt:
                                regions_txt = "local corridor"
                            pnum = int(p.get("path_num", path_num_by_id.get(pid, 0)))
                            summary_lines.append(
                                f"P{pnum}: {self._truncate_text(src_lbl, 24)} -> {self._truncate_text(tgt_lbl, 24)} via {self._truncate_text(regions_txt, 38)}"
                            )

                        cv2.rectangle(batch_img, (8, 8), (w - 8, min(h - 8, 28 + 18 * (len(summary_lines) + 1))), (10, 10, 10), -1)
                        cv2.putText(
                            batch_img,
                            f"Path batch {batch_num} ({start_idx + 1}-{start_idx + len(batch_paths)} of {len(ranked_all)})",
                            (14, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (240, 240, 240),
                            1,
                            cv2.LINE_AA,
                        )
                        for line_idx, line_txt in enumerate(summary_lines, start=1):
                            cv2.putText(
                                batch_img,
                                self._truncate_text(line_txt, 120),
                                (14, 24 + 18 * line_idx),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (230, 230, 230),
                                1,
                                cv2.LINE_AA,
                            )

                        out_file = triplet_dir / f"{str(batch_num).zfill(2)}_paths_{str(start_idx + 1).zfill(3)}_{str(start_idx + len(batch_paths)).zfill(3)}.png"
                        cv2.imwrite(str(out_file), batch_img)
                        triplet_manifest["batches"].append(
                            {
                                "batch_num": batch_num,
                                "index_start": start_idx + 1,
                                "index_end": start_idx + len(batch_paths),
                                "image": f"{triplet_context_rel_root}/{out_file.name}",
                                "path_ids": [str(p.get("path_id", "")) for p in batch_paths],
                                "path_nums": [int(p.get("path_num", path_num_by_id.get(str(p.get("path_id", "")), 0))) for p in batch_paths],
                                "summaries": summary_lines,
                            }
                        )

                    triplet_manifest_path = paths_root_dir / "path_context_triplets_manifest.json"
                    self._write_json(triplet_manifest, triplet_manifest_path)
                    triplet_manifest_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_context_triplets_manifest.json"

        # Render overlays
        start_w = int(getattr(cfg, "path_stroke_start_width_px", 8)) if cfg else 8
        end_w = int(getattr(cfg, "path_stroke_end_width_px", 2)) if cfg else 2
        a0 = float(getattr(cfg, "path_stroke_alpha_start", 0.95)) if cfg else 0.95
        a1 = float(getattr(cfg, "path_stroke_alpha_end", 0.35)) if cfg else 0.35

        map_all = img_bgr.copy()
        # deterministic colors from path_id hash
        def _color(pid: str) -> Tuple[int, int, int]:
            hsh = abs(hash(pid))
            return (int(50 + (hsh % 205)), int(50 + ((hsh // 7) % 205)), int(50 + ((hsh // 49) % 205)))

        for p in paths:
            pts = [tuple(map(int, xy)) for xy in (p.get("polyline_2d") or [])]
            if len(pts) < 2:
                continue
            self._tapered_polyline_draw(map_all, pts, _color(p["path_id"]), start_w, end_w, a0, a1)
            # start/end markers
            cv2.circle(map_all, pts[0], 5, (0, 220, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(map_all, pts[-1], 5, (0, 0, 220), -1, lineType=cv2.LINE_AA)

        map_all_path = paths_root_dir / f"{path_stem}_path_map_all.png"
        cv2.imwrite(str(map_all_path), map_all)
        map_all_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/{path_stem}_path_map_all.png"

        # Top-N (recommended first; fallback to all valid if empty)
        paths_sorted = sorted(paths_recommended, key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)), reverse=True)
        if not paths_sorted:
            paths_sorted = sorted(paths, key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)), reverse=True)
        top_n = min(12, len(paths_sorted))
        map_top = img_bgr.copy()
        for p in paths_sorted[:top_n]:
            pts = [tuple(map(int, xy)) for xy in (p.get("polyline_2d") or [])]
            if len(pts) < 2:
                continue
            self._tapered_polyline_draw(map_top, pts, _color(p["path_id"]), start_w, end_w, a0, a1)
        map_top_path = paths_root_dir / f"{path_stem}_path_map_topN.png"
        cv2.imwrite(str(map_top_path), map_top)
        map_top_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/{path_stem}_path_map_topN.png"

        # Cost map exports for debugging/alignment QA
        cost_map_u8 = np.clip(cost_map * 255.0, 0, 255).astype(np.uint8)
        cost_map_png_path = paths_root_dir / "path_cost_map.png"
        self._write_png_intermediate(cost_map_png_path, cost_map_u8)
        cost_map_npy_path = paths_root_dir / "path_cost_map.npy"
        np.save(str(cost_map_npy_path), cost_map)
        cost_map_png_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_cost_map.png"
        cost_map_npy_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_cost_map.npy"

        fields_explainer_rel: str = ""
        fields_legend_rel: str = ""
        if bool(getattr(cfg, "path_export_fields_explainer", True)) if cfg else True:
            try:
                expl_img = build_path_fields_explainer_image(img_bgr, cost_map, speed_map, paths_sorted, cfg)
                expl_path = paths_root_dir / "path_fields_explainer.png"
                self._write_png_intermediate(expl_path, expl_img)
                fields_explainer_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_fields_explainer.png"
                leg = build_path_fields_legend_payload(path_stem, trav_meta, export_trav)
                leg_path = paths_root_dir / "path_fields_legend.json"
                self._write_json(leg, leg_path)
                fields_legend_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_fields_legend.json"
            except Exception as _ex:
                print(f"path_fields_explainer export failed: {_ex}")

        # Stage-wise visual exports
        stage_images_rel_root = ""
        stage_index: Dict[str, Dict[str, str]] = {}
        export_stage_images = bool(getattr(cfg, "path_export_stage_images", True)) if cfg else True
        if cpu_benchmark_mode and cpu_disable_heavy_visuals:
            export_stage_images = False
        if export_stage_images:
            stage_root = images_root_dir / "stages"
            stage_root.mkdir(parents=True, exist_ok=True)
            stage_images_rel_root = f"scene_graph/{track_dir_name}/{path_stem}_paths/images/stages"
            stage_defs = [
                ("00_pair_proposals", paths_sorted),
                ("01_cost_map", paths_sorted),
                ("02_coarse_routes", paths_sorted),
                ("03_refined_routes", paths_sorted),
                ("04_semantic_hard_filtered", [p for p in paths_sorted if bool(p.get("semantic_valid", True))]),
                ("05_ranked_topk", paths_sorted[: min(10, len(paths_sorted))]),
            ]
            levels_cfg = list(getattr(cfg, "path_stage_levels", ["global", "region", "object", "mask"])) if cfg else ["global", "region", "object", "mask"]
            for stage_name, plist in stage_defs:
                for lvl in levels_cfg:
                    out_dir = stage_root / stage_name / lvl
                    out_dir.mkdir(parents=True, exist_ok=True)
                    canvas = img_bgr.copy()
                    for p in plist:
                        if lvl != "global" and str(p.get("path_level", "")) != str(lvl):
                            continue
                        pts = [tuple(map(int, xy)) for xy in (p.get("polyline_2d") or [])]
                        if len(pts) < 2:
                            continue
                        pid = str(p.get("path_id", ""))
                        col = _color(pid)
                        self._tapered_polyline_draw(canvas, pts, col, start_w, end_w, a0, a1)
                        cv2.arrowedLine(canvas, pts[0], pts[-1], col, 2, cv2.LINE_AA, tipLength=0.12)
                        cv2.putText(
                            canvas,
                            f"{pid} | c={float((p.get('scores') or {}).get('overall_confidence', 0.0)):.2f}",
                            (max(8, pts[0][0] + 4), max(16, pts[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            col,
                            1,
                            cv2.LINE_AA,
                        )
                    out_file = out_dir / f"{path_stem}_{stage_name}_{lvl}.png"
                    cv2.imwrite(str(out_file), canvas)
                    stage_index.setdefault(stage_name, {})[lvl] = f"{stage_images_rel_root}/{stage_name}/{lvl}/{path_stem}_{stage_name}_{lvl}.png"

        # JSON + MD exports
        hypotheses = {
            "schema_version": "1.0",
            "image_stem": path_stem,
            "image_size": {"width": w, "height": h},
            "track": track_dir_name,
            "paths": paths,
            "recommended_path_ids": [str(p.get("path_id", "")) for p in paths_recommended],
            "suppressed_path_ids": [str(p.get("path_id", "")) for p in paths if bool(p.get("suppressed", False))],
            "rejections": rejections,
            "traversability": {
                "speed_map_npy": trav_speed_npy_rel,
                "speed_map_png": trav_speed_png_rel,
                "meta": dict(trav_meta),
                "geodesic_refinement_enabled": bool(use_geo),
            },
        }
        from scene_understanding.pathing.path_hypotheses_paths import path_hypotheses_json_path

        hyp_path = path_hypotheses_json_path(paths_root_dir)
        self._write_json(hypotheses, hyp_path)
        hyp_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_hypotheses.json"

        insertion_rel = ""
        trajectory_rel = ""
        animation_components_rel = ""
        motion_overlay_rel = ""
        traj_bundle: Optional[Dict[str, Any]] = None
        if bool(getattr(cfg, "export_motion_contract_json", True)) if cfg else True:
            image_size_wh = {"width": w, "height": h}
            ins_bundle = build_insertion_bundle(
                paths,
                path_stem,
                image_size_wh,
                track_dir_name,
                path_stem,
                cfg,
                relations,
                traversability_speed_npy_rel=trav_speed_npy_rel,
            )
            traj_bundle = build_trajectory_bundle(
                objects_3d_with_masks,
                relations,
                path_stem,
                image_size_wh,
                track_dir_name,
                path_stem,
                cfg,
                metric_depth_m=metric_depth_m,
                traversability_speed_npy_rel=trav_speed_npy_rel,
            )
            ins_path = paths_root_dir / "insertion_path_ensembles.json"
            traj_path = paths_root_dir / "trajectory_hypotheses.json"
            self._write_json(ins_bundle, ins_path)
            self._write_json(traj_bundle, traj_path)
            insertion_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/insertion_path_ensembles.json"
            trajectory_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/trajectory_hypotheses.json"
            anim_components = build_animation_components_bundle(
                paths=paths_sorted,
                trajectory_bundle=traj_bundle,
                image_stem=path_stem,
                image_size=image_size_wh,
                track_dir_name=track_dir_name,
            )
            anim_components_path = paths_root_dir / "animation_components.json"
            self._write_json(anim_components, anim_components_path)
            animation_components_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/animation_components.json"
            if bool(getattr(cfg, "path_motion_contract_overlay", True)) if cfg else True:
                mov_path = paths_root_dir / "motion_contracts_overlay.png"
                SceneUnderstandingPipeline._write_motion_contract_overlay(img_bgr, paths_sorted, traj_bundle, mov_path, cfg=cfg)
                motion_overlay_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/motion_contracts_overlay.png"

        if bool(getattr(cfg, "path_export_pair_proposals_json", True)) if cfg else True:
            pair_path = paths_root_dir / "pair_proposals.json"
            self._write_json({"pairs": pair_proposals}, pair_path)
            pair_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/pair_proposals.json"
        else:
            pair_rel = ""

        if bool(getattr(cfg, "path_export_diagnostics_json", True)) if cfg else True:
            diags = []
            family_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}
            for p in paths_sorted:
                fam = f"{str(((p.get('source_entity') or {}).get('type', 'unknown')))}_to_{str(((p.get('target_entity') or {}).get('type', 'unknown')))}"
                family_counts[fam] = int(family_counts.get(fam, 0)) + 1
                src_kind = str(p.get("generation_source", "unknown"))
                source_counts[src_kind] = int(source_counts.get(src_kind, 0)) + 1
                diags.append({
                    "path_id": p.get("path_id", ""),
                    "path_num": p.get("path_num", 0),
                    "path_level": p.get("path_level", ""),
                    "path_family": fam,
                    "generation_source": src_kind,
                    "family_confidence": float(p.get("family_confidence", (p.get("scores") or {}).get("overall_confidence", 0.0))),
                    "scores": p.get("scores", {}),
                    "diagnostics": p.get("diagnostics", {}),
                    "semantic_reasons": p.get("semantic_reasons", []),
                })
            dpath = paths_root_dir / "path_diagnostics.json"
            self._write_json(
                {
                    "family_coverage_report": {
                        "family_counts": family_counts,
                        "generation_source_counts": source_counts,
                    },
                    "paths": diags,
                },
                dpath,
            )
            diag_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_diagnostics.json"
        else:
            diag_rel = ""

        desc_path = paths_root_dir / "path_descriptions.json"
        if export_desc:
            self._write_json(descriptions_by_id, desc_path)
            desc_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_descriptions.json"
        else:
            desc_rel = ""

        md_lines = [f"# Path hypotheses: {path_stem}", ""]
        md_lines.append(
            "This file summarizes image-overlaid path hypotheses with stable numeric IDs "
            "(Path 1..N) and human-readable reasoning for each path."
        )
        md_lines.append(
            "Each entry below is written in prose so the trajectory, confidence, and rationale can be read quickly by non-technical users."
        )
        md_lines.append(
            "For ranked **line-only** QA panels with stable colors per `path_id`, see `path_top30_atlas.md` when `path_atlas_enabled` is True."
        )
        md_lines.append("")
        for p in paths_sorted[: min(50, len(paths_sorted))]:
            pid = p.get("path_id", "")
            pnum = int(path_num_by_id.get(str(pid), 0))
            desc = ""
            if export_desc and pid in descriptions_by_id:
                d = descriptions_by_id[pid]
                desc = str(d.get("summary", "")) or str(d.get("why_valid", ""))
            level = str(p.get("path_level", "path"))
            src_lbl = str((p.get("source_entity") or {}).get("display_label", "source"))
            tgt_lbl = str((p.get("target_entity") or {}).get("display_label", "target"))
            regions_txt = ", ".join(p.get("regions_traversed", []) or [])
            conf = float((p.get("scores") or {}).get("overall_confidence", 0.0))
            sem_ok = bool(p.get("semantic_valid", True))
            sem_score = float(p.get("semantic_validity_score", 0.0))
            sem_reasons = "; ".join([str(x) for x in (p.get("semantic_reasons") or [])[:2]])
            usable = bool(sem_ok and conf >= min_conf)
            if conf >= 0.80:
                usage_band = "high-confidence"
            elif conf >= 0.60:
                usage_band = "moderate-confidence"
            else:
                usage_band = "low-confidence"
            usage_txt = "recommended for use" if usable else "not recommended as a primary route"

            md_lines.append(f"## Path {pnum}: {pid}")
            md_lines.append(
                f"Path {pnum} is a {usage_band} {level}-level route from {src_lbl} to {tgt_lbl} "
                f"through {regions_txt if regions_txt else 'its inferred region corridor'}, "
                f"with overall confidence {conf:.2f}; this path is {usage_txt}."
            )
            md_lines.append(
                f"Semantic plausibility is {'supported' if sem_ok else 'limited'} "
                f"(score {sem_score:.2f}{'; ' + sem_reasons if sem_reasons else ''})."
            )
            if usable:
                md_lines.append(
                    "User-facing interpretation: this is a practical candidate path for animation blocking, "
                    "because both geometric validity and semantic consistency are acceptable."
                )
            else:
                md_lines.append(
                    "User-facing interpretation: keep this as a fallback/reference path only, "
                    "and prioritize higher-confidence alternatives with stronger semantic support."
                )
            if desc:
                md_lines.append(f"Reasoning: {desc}")
            md_lines.append("")
        md_path = paths_root_dir / "path_reasoning.md"
        self._write_text("\n".join(md_lines) + "\n", md_path)
        md_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_reasoning.md"

        # Extended explainability export requested for path/trajectory audit.
        desc_md_lines: List[str] = [
            f"# Path and Trajectory Descriptions: {path_stem}",
            "",
            "This document explains how paths/trajectories were formed, the evidence used,",
            "and why hypotheses were ranked or suppressed.",
            "",
            "## Calibration and constraints",
            "",
            f"- Semantic hard filter enabled: `{bool(getattr(cfg, 'path_semantic_hard_filter_enabled', True))}`",
            f"- CPU benchmark mode: `{bool(getattr(cfg, 'cpu_benchmark_mode', False))}`",
            "",
            "## Path hypotheses",
            "",
        ]
        _family_counts_md: Dict[str, int] = {}
        _source_counts_md: Dict[str, int] = {}
        for _p in paths_sorted:
            _fam = f"{str(((_p.get('source_entity') or {}).get('type', 'unknown')))}_to_{str(((_p.get('target_entity') or {}).get('type', 'unknown')))}"
            _family_counts_md[_fam] = int(_family_counts_md.get(_fam, 0)) + 1
            _src = str(_p.get("generation_source", "unknown"))
            _source_counts_md[_src] = int(_source_counts_md.get(_src, 0)) + 1
        desc_md_lines.append("### Family coverage report")
        desc_md_lines.append(f"- family_counts: {_family_counts_md}")
        desc_md_lines.append(f"- generation_source_counts: {_source_counts_md}")
        desc_md_lines.append("")
        for p in paths_sorted[: min(80, len(paths_sorted))]:
            pid = str(p.get("path_id", ""))
            fam = str(p.get("path_family", "unknown"))
            src_lbl = str((p.get("source_entity") or {}).get("display_label", "source"))
            tgt_lbl = str((p.get("target_entity") or {}).get("display_label", "target"))
            sc = p.get("scores") or {}
            en = p.get("energy_terms") or {}
            dg = p.get("diagnostics") or {}
            desc_md_lines.append(f"### `{pid}` ({fam})")
            desc_md_lines.append(f"- formation: {src_lbl} -> {tgt_lbl}, level `{p.get('path_level', '')}`, type `{p.get('path_type', '')}`")
            desc_md_lines.append(
                f"- confidence: overall={float(sc.get('overall_confidence', 0.0)):.3f}, hybrid={float(sc.get('hybrid_overall', 0.0)):.3f}, semantic={float(p.get('semantic_validity_score', 0.0)):.3f}"
            )
            desc_md_lines.append(
                f"- constraints: {', '.join(sorted((p.get('constraints_applied') or {}).keys())) or '(none)'}"
            )
            desc_md_lines.append(
                f"- evidence: walkable_ratio={float(dg.get('walkable_ratio', 0.0)):.3f}, obstacle_ratio={float(dg.get('obstacle_ratio', 0.0)):.3f}, far_ratio={float(dg.get('far_background_ratio', 0.0)):.3f}"
            )
            desc_md_lines.append(
                f"- naturalness: score={float(dg.get('naturalness_score', 0.0)):.3f}, generation_source={str(p.get('generation_source', 'unknown'))}, family_confidence={float(p.get('family_confidence', 0.0)):.3f}"
            )
            desc_md_lines.append(
                f"- semantic-depth-3d-rel summary: E_total={float(en.get('E_total', 0.0)):.3f}, E_geo={float(en.get('E_geo', 0.0)):.3f}, E_sem={float(en.get('E_sem', 0.0)):.3f}, E_depth={float(en.get('E_depth', 0.0)):.3f}, E_rel={float(en.get('E_rel', 0.0)):.3f}"
            )
            desc_md_lines.append(
                f"- ranking/rejection: suppressed={bool(p.get('suppressed', False))}, reason=`{str(p.get('suppressed_reason', ''))}`"
            )
            if export_desc and pid in descriptions_by_id:
                desc_md_lines.append(f"- rationale: {str((descriptions_by_id.get(pid) or {}).get('summary', '')).strip()}")
            desc_md_lines.append("")
        if isinstance(traj_bundle, dict) and (traj_bundle.get("hypotheses") or []):
            desc_md_lines.append("## Trajectory hypotheses")
            desc_md_lines.append("")
            for h0 in traj_bundle.get("hypotheses", [])[:50]:
                tid = str(h0.get("trajectory_id", ""))
                subj = (h0.get("subject") or {}).get("display_label", "")
                conf0 = float(h0.get("confidence", 0.0))
                tm = h0.get("time_model") or {}
                desc_md_lines.append(f"### `{tid}`")
                desc_md_lines.append(f"- subject: {subj}")
                desc_md_lines.append(f"- confidence: {conf0:.3f}")
                desc_md_lines.append(f"- model: {tm.get('type', '')}, horizon_s={tm.get('horizon_s', 0.0)}, dt_s={tm.get('dt_s', 0.0)}")
                desc_md_lines.append(f"- constraints_used: {', '.join(sorted((h0.get('constraints_used') or {}).keys())) or '(none)'}")
                desc_md_lines.append("")
        descriptions_md_path = paths_root_dir / "descriptions.md"
        self._write_text("\n".join(desc_md_lines) + "\n", descriptions_md_path)
        descriptions_md_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/descriptions.md"

        # Animation plan
        animation_enabled = bool(getattr(cfg, "path_animation_enabled", True)) if cfg else True
        animation_rel = ""
        if animation_enabled:
            anim = self._animation_plan_for_paths(paths_sorted, top_k=10, cfg=cfg)
            anim_path = paths_root_dir / "animation_plan.json"
            self._write_json(anim, anim_path)
            animation_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/animation_plan.json"

        # Visual index
        visual_index: Dict[str, Any] = {"paths": {}, "stages": stage_index}
        for p in paths_sorted:
            pid = str(p.get("path_id", ""))
            lvl = str(p.get("path_level", "mask"))
            per_img = f"scene_graph/{track_dir_name}/{path_stem}_paths/images/{lvl}/path_{pid}.png"
            visual_index["paths"][pid] = {
                "path_id": pid,
                "path_num": p.get("path_num", 0),
                "level": lvl,
                "per_path_image": per_img,
                "description_record": f"scene_graph/{track_dir_name}/{path_stem}_paths/path_descriptions.json",
                "descriptions_md": descriptions_md_rel,
                "animation_record": animation_rel,
                "animation_components_record": animation_components_rel,
                "diagnostics_record": diag_rel,
            }
        visual_index_path = paths_root_dir / "path_visual_index.json"
        if triplet_context_rel_root:
            visual_index["triplet_context_batches"] = {
                "images_root": triplet_context_rel_root,
                "manifest": triplet_manifest_rel,
            }
        visual_index["scene_context_record"] = scene_ctx_rel
        self._write_json(visual_index, visual_index_path)
        visual_index_rel = f"scene_graph/{track_dir_name}/{path_stem}_paths/path_visual_index.json"

        atlas_rels: Dict[str, str] = {}
        if bool(getattr(cfg, "path_atlas_enabled", False)) and not (cpu_benchmark_mode and cpu_disable_heavy_visuals):
            try:
                from scene_understanding.pathing.path_atlas_export import export_trajs_upt_atlas

                tb = traj_bundle if isinstance(traj_bundle, dict) else {}
                atlas_rels = export_trajs_upt_atlas(
                    paths_root_dir=paths_root_dir,
                    path_stem=path_stem,
                    track_dir_name=track_dir_name,
                    paths_sorted=paths_sorted,
                    paths_recommended=paths_recommended,
                    descriptions_by_id=descriptions_by_id,
                    path_num_by_id=path_num_by_id,
                    min_conf=min_conf,
                    traj_bundle=tb,
                    cfg=cfg,
                    width=w,
                    height=h,
                    export_desc=export_desc,
                    animation_rel=animation_rel,
                    insertion_rel=insertion_rel,
                    trajectory_rel=trajectory_rel,
                    hyp_rel=hyp_rel,
                    desc_rel=desc_rel,
                    diag_rel=diag_rel,
                    semantic_rel=semantic_layer_rel,
                    pair_rel=pair_rel,
                    descriptions_md_rel=descriptions_md_rel,
                    cost_png_rel=cost_map_png_rel,
                    cost_npy_rel=cost_map_npy_rel,
                    trav_png_rel=trav_speed_png_rel,
                    trav_npy_rel=trav_speed_npy_rel,
                    img_bgr=img_bgr,
                    label_map=lm,
                    objects=objs,
                    metric_depth_m=metric_depth_m,
                )
            except Exception as _atlas_ex:
                print(f"  [PathAtlas] export failed: {_atlas_ex}")
                atlas_rels = {}

        animation_qa_rels: Dict[str, str] = {}
        if bool(getattr(cfg, "path_animation_qa_enabled", True)) and not (cpu_benchmark_mode and cpu_disable_heavy_visuals):
            try:
                from scene_understanding.visualization.animation_qa_renderer import export_animation_qa

                animation_qa_rels = export_animation_qa(
                    paths_root_dir=paths_root_dir,
                    img_bgr=img_bgr,
                    metric_depth_m=metric_depth_m,
                    cfg=cfg,
                    region_label_map=lm,
                )
            except Exception as _qa_ex:
                print(f"  [AnimationQA] export failed: {_qa_ex}")
                animation_qa_rels = {}

        v2_exports: Dict[str, str] = {}
        try:
            v2_exports = self._export_trajectory_viz_v2(
                img_bgr=img_bgr,
                paths=paths,
                paths_recommended=paths_recommended,
                lm=lm,
                objs=objs,
                w=w,
                h=h,
                paths_root_dir=paths_root_dir,
                path_stem=path_stem,
                track_dir_name=track_dir_name,
                cfg=cfg,
                paths_sorted=paths_sorted,
            )
        except Exception as _ex:
            print(f"  [TrajectoryV2] export failed: {_ex}")
            v2_exports = {}

        return {
            "path_hypotheses_json": hyp_rel,
            "path_descriptions_json": desc_rel,
            "path_reasoning_md": md_rel,
            "descriptions_md": descriptions_md_rel,
            "path_map_all_image": map_all_rel,
            "path_map_topN_image": map_top_rel,
            "path_context_top5_image": f"scene_graph/{track_dir_name}/{path_stem}_paths/path_context_top5.png",
            "path_context_triplets_images_root": triplet_context_rel_root,
            "path_context_triplets_manifest_json": triplet_manifest_rel,
            "semantic_layer_json": semantic_layer_rel,
            "animation_plan_json": animation_rel,
            "path_visual_index_json": visual_index_rel,
            "scene_context_json": scene_ctx_rel,
            "path_stage_images_root": stage_images_rel_root,
            "path_pair_proposals_json": pair_rel,
            "path_diagnostics_json": diag_rel,
            "path_cost_map_png": cost_map_png_rel,
            "path_cost_map_npy": cost_map_npy_rel,
            "path_traversability_speed_npy": trav_speed_npy_rel,
            "path_traversability_speed_png": trav_speed_png_rel,
            "insertion_path_ensembles_json": insertion_rel,
            "trajectory_hypotheses_json": trajectory_rel,
            "animation_components_json": animation_components_rel,
            "motion_contracts_overlay_image": motion_overlay_rel,
            "path_fields_explainer_image": fields_explainer_rel,
            "path_fields_legend_json": fields_legend_rel,
            **atlas_rels,
            **animation_qa_rels,
            **v2_exports,
        }

    def process_image(self, image_path: str, output_dir: str):
        path = Path(image_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _t_wall0 = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _profile = self._scene_pipeline_profile_enabled()
        prof: Dict[str, float] = {}
        _tp = time.perf_counter()
        _image_key_for_timing = path.name

        print(f"Processing scene understanding for: {path.name}")

        img_bgr = _load_bgr_image(path)

        img_bgr = self._undistort_image(img_bgr)  # must run before depth/segmentation (see docs/CAMERA_CALIBRATION.md)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # Downscale large images to keep RAM usage bounded.
        max_side = int(getattr(self.config, "sam2_amg_max_image_side", 1280)) if self.config else 1280
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
            print(f"  Resized to {w}x{h} (max_side={max_side})")

        # Intrinsics: calibration file > explicit values > FOV estimate (see docs/CAMERA_CALIBRATION.md)
        K = self.fixed_intrinsics if self.fixed_intrinsics else self._estimate_intrinsics(w, h)

        from scene_understanding.preprocessing import ImagePyramid  # noqa: WPS433 — local import to avoid cycles
        # Phase B.6: build the multi-level ImagePyramid once, here, so that
        # GDINO / SAM2 / depth / caption wrappers can all pull the right
        # resolution view without re-running cv2.resize + cvtColor. The
        # pyramid stores the source image by reference; subsequent accesses
        # to `bgr("full")` or `rgb("full")` return the exact same ndarray
        # the pipeline already holds.
        try:
            if getattr(self, "sam2_wrapper", None) is not None:
                _gdino_side = int(getattr(self.sam2_wrapper, "max_image_side", 1280) or 1280)
            else:
                _gdino_side = int((self._sam2_reinit_kwargs or {}).get("max_image_side", 1280) or 1280)
        except Exception:
            _gdino_side = 1280
        _pyramid_sides = {
            "full": 0,
            "gdino": _gdino_side,
            "sam2": max(1024, int(max_side)),
            "depth": max(1024, int(max_side)),
            "caption": 768,
        }
        self._image_pyramid = ImagePyramid(img_bgr, max_sides=_pyramid_sides)
        # Phase B.7: reset the per-image global depth-histogram cache so that
        # the next image does not see stats that belong to an earlier frame.
        self._global_depth_stats_cache = {}
        # Phase B.9 — reset region-scoped caches for the new frame.
        self._region_bg_mean_cache = {}
        self._region_pixel_mask_cache = {}
        try:
            if self.florence2 is not None and hasattr(self.florence2, "reset_image_caches"):
                self.florence2.reset_image_caches()
        except Exception:
            pass

        if _profile:
            prof["pre_depth_s"] = time.perf_counter() - _tp
        _tp = self._record_stage_time(_image_key_for_timing, "pre_depth", _tp)

        # Depth: reuse cached npy when enabled, otherwise infer and save.
        # Phase B.8 — additionally look for a compact fp16 .npz cache
        # (``<stem>_depth.npz``) and prefer it when present. The legacy
        # ``_depth_metric.npy`` file is still written for backward
        # compatibility and to keep the exact on-disk artefact that other
        # consumers (depth visualisers, exports) already expect.
        depth_dir = out / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_npy = depth_dir / f"{path.stem}_depth_metric.npy"
        depth_npz_fp16 = depth_dir / f"{path.stem}_depth.npz"
        metric_depth = None
        if self._reuse_cached_depth and depth_npz_fp16.exists():
            try:
                _npz = np.load(str(depth_npz_fp16))
                _arr = _npz["depth"] if "depth" in _npz.files else list(_npz.values())[0]
                metric_depth = np.asarray(_arr, dtype=np.float32)
                if metric_depth.shape[:2] != (h, w):
                    metric_depth = cv2.resize(metric_depth, (w, h), interpolation=cv2.INTER_NEAREST)
                print(f"  Reusing cached depth (fp16 npz): {depth_npz_fp16.name}")
            except Exception as _npz_err:
                print(f"  [depth] fp16 npz cache unreadable ({_npz_err}); falling through.")
                metric_depth = None
        if metric_depth is None and self._reuse_cached_depth and depth_npy.exists():
            metric_depth = np.load(str(depth_npy)).astype(np.float32)
            if metric_depth.shape[:2] != (h, w):
                metric_depth = cv2.resize(metric_depth, (w, h), interpolation=cv2.INTER_NEAREST)
            print(f"  Reusing cached depth: {depth_npy.name}")
        if metric_depth is None:
            self._ensure_depth_ready()
            raw_depth = self.depth_estimator.backend.infer(img_rgb)
            depth_full = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_NEAREST)
            metric_depth = depth_full * self.depth_scale_factor
            np.save(str(depth_npy), metric_depth)
            # Also write the compact fp16 cache for subsequent runs. Using
            # np.savez_compressed keeps the artefact small and additive —
            # the canonical float32 .npy file above is unchanged.
            try:
                np.savez_compressed(str(depth_npz_fp16), depth=metric_depth.astype(np.float16))
            except Exception as _save_err:
                print(f"  [depth] fp16 npz cache write failed: {_save_err}")

        if _profile:
            prof["depth_s"] = time.perf_counter() - _tp
        _tp = self._record_stage_time(_image_key_for_timing, "depth", _tp)

        # 3. Object-level segmentation + labelling (see docs/SEGMENTATION.md, docs/LABELLING_AND_RELATIONS.md)
        sam2_track_masks: List[Dict[str, Any]] = []
        sam2_track_dets: List[Dict[str, Any]] = []
        scene_graph_dir = out / "scene_graph"
        scene_graph_dir.mkdir(parents=True, exist_ok=True)
        depth_map_npy_rel = f"depth/{path.stem}_depth_metric.npy"
        depth_global_min = float(np.min(metric_depth))
        depth_global_max = float(np.max(metric_depth))
        depth_global_mean = float(np.mean(metric_depth))

        region_label_map: Optional[np.ndarray] = None
        region_partition_meta: List[Dict[str, Any]] = []
        region_palette: List[List[int]] = [[0, 0, 0]]
        shared_regions_png_path: Optional[Path] = None
        if self.regions_enabled and partition_depth_regions is not None and label_map_to_bgr is not None:
            _dm = str(getattr(self.depth_estimator, "backend_name", "DepthAnythingV2"))
            _part = partition_depth_regions(
                metric_depth,
                k=self._regions_k,
                min_region_px=self._regions_min_region_px,
                blur_sigma=self._regions_blur_sigma,
                seed=self._regions_seed,
                depth_model_id=_dm,
            )
            region_label_map = _part.label_map
            region_partition_meta = list(_part.regions)
            region_palette = list(_part.palette)
            _bgr = label_map_to_bgr(_part.label_map, _part.palette)
            shared_regions_png_path = depth_dir / f"{path.stem}_regions.png"
            cv2.imwrite(str(shared_regions_png_path), _bgr)
            np.save(str(depth_dir / f"{path.stem}_regions_label_map.npy"), _part.label_map.astype(np.int32))
            print(f"  [Regions] partitioned into {len(region_partition_meta)} regions → {shared_regions_png_path.name}")

        if _profile:
            prof["regions_s"] = time.perf_counter() - _tp
        _tp = self._record_stage_time(_image_key_for_timing, "regions", _tp)

        region_by_index: Dict[int, Dict[str, Any]] = {
            int(r["region_index"]): r for r in region_partition_meta if r.get("region_index") is not None
        }

        # SAM2 must be live before the RAM++ dynamic-vocab hook because that
        # hook calls ``sam2_wrapper.update_text_query(...)`` in-place.
        self._ensure_sam2_ready()

        # RAM++ dynamic vocabulary (config: query_builder_mode, rampp_enabled).
        _rampp_tags_for_metadata, _gdino_query_used = refresh_gdino_query_if_configured(
            self, img_rgb, region_partition_meta, w, h
        )

        masks_grounded = self.sam2_wrapper.generate(
            img_rgb,
            pyramid=getattr(self, "_image_pyramid", None),
        )
        if _profile:
            prof["segmentation_s"] = time.perf_counter() - _tp
        _tp = self._record_stage_time(_image_key_for_timing, "segmentation", _tp)

        # Memory-pressure guard: on 8 GB MPS systems we *must* evict Depth V2
        # and GDINO+SAM2 here, before Florence-2 + RAM++ run per-mask. Keeping
        # them co-resident pushes the process footprint to ~11 GB → macOS
        # pages us out → every MPS copy blocks on ``MTLCommandBuffer
        # waitUntilCompleted`` waiting for swapped memory to come back in.
        # See config.release_heavy_models_after_segmentation for the full
        # diagnosis. ``_ensure_sam2_ready`` / ``_ensure_depth_ready`` restore
        # them at the start of the next image.
        self._release_heavy_models_after_segmentation()
        _tp = self._record_stage_time(_image_key_for_timing, "release_heavy_models", _tp)

        # Single export track: GroundedSAM2 only (no AMG / no duplicate combined track).
        track_specs: List[Tuple[str, str, List[Dict[str, Any]]]] = [
            ("grounded_sam2", self.track_dir_grounded_sam2, masks_grounded),
        ]
        generated_tracks: List[Dict[str, str]] = []
        region_labeller_enrich: Dict[int, Dict[str, Any]] = {}
        self._load_labellers()
        if _profile:
            prof["labellers_load_s"] = time.perf_counter() - _tp
        _tp = self._record_stage_time(_image_key_for_timing, "labellers_load", _tp)

        mask_label_acc = 0.0
        _tp_track = time.perf_counter()
        # Phase F — depth colour-map PNG is identical across all tracks for
        # the same image, so render it exactly once and hardlink it into each
        # per-track directory. Falls back to a regular copy when the
        # filesystem rejects hardlinks (warn-once lives in
        # ``_copy_or_link_file``).
        _shared_depth_png_path: Optional[Path] = None
        for track_key, track_dir_name, track_masks in track_specs:
            if not track_masks:
                print(f"Track[{track_key}] skipped (no masks).")
                continue
            sam2_sg_dir = scene_graph_dir / track_dir_name
            sam2_sg_dir.mkdir(parents=True, exist_ok=True)
            _depth_png_dst = sam2_sg_dir / f"{path.stem}_depth_map.png"
            if _shared_depth_png_path is None:
                self._save_depth_map_image(metric_depth, _depth_png_dst)
                # Phase F — block until this first depth PNG reaches disk so
                # subsequent hardlink sources exist. Only drain the pending
                # depth write, not the whole writer pool.
                try:
                    self._drain_writers()
                except Exception:
                    pass
                _shared_depth_png_path = _depth_png_dst
            else:
                try:
                    self._copy_or_link_file(_shared_depth_png_path, _depth_png_dst, self._regions_use_hardlink)
                except Exception:
                    self._save_depth_map_image(metric_depth, _depth_png_dst)
            depth_map_image_rel = f"scene_graph/{track_dir_name}/{path.stem}_depth_map.png"
            seg_map_rel = f"scene_graph/{track_dir_name}/{path.stem}_segmentation.png"

            track_dets: List[Dict[str, Any]] = []
            for i, amg in enumerate(track_masks):
                seg = amg.get("segmentation")
                mask_bin = (np.asarray(seg) > 0) if seg is not None else np.zeros((h, w), dtype=bool)
                _ri = 0
                if region_label_map is not None and majority_region_index is not None:
                    _ri = int(majority_region_index(mask_bin, region_label_map))
                if _profile:
                    _tlm = time.perf_counter()
                det = self._label_mask(
                    img_bgr,
                    mask_bin,
                    amg,
                    label_map=region_label_map,
                    region_index=_ri,
                    _secondary_phase="florence_only",
                )
                if _profile:
                    mask_label_acc += time.perf_counter() - _tlm
                det["region_index"] = _ri
                det["region_id"] = f"region_{_ri}" if _ri > 0 else ""
                det["graph_id"] = f"{track_key}_obj_{i}_GroundedSAM2"
                det["sam2_mask_index"] = int(i)
                det["grounded_sam2_label"] = str(amg.get("label", det.get("label", "object"))).strip().lower()
                det["grounded_sam2_confidence"] = float(amg.get("gdino_conf", amg.get("predicted_iou", 0.0)))
                det["bbox"] = self._xywh_to_xyxy(amg.get("bbox", [0, 0, w, h]))
                det["segmentor"] = str(amg.get("source_model", "GroundedSAM2"))
                track_dets.append(det)

            # Phase B: single RAM++ init, then RAM crop labelling for every mask that ran secondaries.
            _rampp_ok = bool(getattr(self, "_rampp_enabled", False)) and not bool(
                getattr(self, "_rampp_runtime_disabled", False)
            )
            _skip_sec = bool(getattr(self, "_mask_label_skip_secondary_when_gdino_specific", True))
            _any_rampp = False
            if _rampp_ok:
                for amg in track_masks:
                    _gl = str(amg.get("label", "object")).strip().lower()
                    if not (_skip_sec and _gl and _gl != "object"):
                        _any_rampp = True
                        break
            if _any_rampp:
                self._unload_florence_before_rampp_init()
                self._ensure_rampp_for_labelling()
                for i, amg in enumerate(track_masks):
                    seg = amg.get("segmentation")
                    mask_bin = (np.asarray(seg) > 0) if seg is not None else np.zeros((h, w), dtype=bool)
                    _ri = 0
                    if region_label_map is not None and majority_region_index is not None:
                        _ri = int(majority_region_index(mask_bin, region_label_map))
                    _gl = str(amg.get("label", "object")).strip().lower()
                    if _skip_sec and _gl and _gl != "object":
                        continue
                    det = track_dets[i]
                    _pre = (
                        str(det.get("florence2_label", "") or "object"),
                        str(det.get("florence2_caption", "") or "object"),
                    )
                    _meta_keys = (
                        "region_index",
                        "region_id",
                        "graph_id",
                        "sam2_mask_index",
                        "grounded_sam2_label",
                        "grounded_sam2_confidence",
                        "bbox",
                        "segmentor",
                    )
                    _meta = {k: det[k] for k in _meta_keys if k in det}
                    if _profile:
                        _tlm = time.perf_counter()
                    merged = self._label_mask(
                        img_bgr,
                        mask_bin,
                        amg,
                        label_map=region_label_map,
                        region_index=_ri,
                        _secondary_phase="rampp_only",
                        _prefill_florence=_pre,
                    )
                    if _profile:
                        mask_label_acc += time.perf_counter() - _tlm
                    track_dets[i] = {**merged, **_meta}

            print(f"Stage 3 [{track_key}]: {len(track_dets)} mask-objects labelled (no filtering — all masks kept)")

            matched_A_lookup: Dict[str, Dict[str, Any]] = {}
            _ordered_masks_for_label_map: List[np.ndarray] = []
            for det in track_dets:
                idx = int(det.get("sam2_mask_index", -1))
                if 0 <= idx < len(track_masks):
                    seg = track_masks[idx].get("segmentation")
                    mask = (np.asarray(seg) > 0) if seg is not None else np.zeros((h, w), dtype=bool)
                else:
                    idx = -1
                    mask = np.zeros((h, w), dtype=bool)
                matched_A_lookup[det["graph_id"]] = {
                    "detection": det,
                    "mask": mask,
                    "sam2_mask_index": idx,
                    "mask_bbox_xyxy": det["bbox"],
                }
                _ordered_masks_for_label_map.append(mask)

            # Phase A: materialise the unified int32 object label map once per
            # image so downstream per-object stats can be vectorised via
            # scipy.ndimage, and a single-op contact graph replaces the O(N^2)
            # pairwise mask-AND scan.
            object_label_map = self._build_object_label_map(
                _ordered_masks_for_label_map, h, w
            )
            try:
                contact_graph = self._build_contact_graph(object_label_map, dilate_radius_px=1)
            except Exception:
                contact_graph = set()
            # Pix2SG uses this gate opportunistically when regions are on.
            try:
                setattr(self.pix2sg, "_contact_graph", contact_graph)
                setattr(self.pix2sg, "_object_label_map", object_label_map)
            except Exception:
                pass

            depth_mask_modes = ["A"] if "A" in self.depth_mask_modes else list(self.depth_mask_modes)
            for mode in depth_mask_modes:
                if mode == "A":
                    matched = [matched_A_lookup[det["graph_id"]] for det in track_dets]
                elif mode == "B":
                    matched = self._match_mask_first(track_masks, track_dets, iou_thresh=self.mask_iou_match_thresh) if track_masks else []
                else:
                    continue
                json_objects = []
                for i, mobj in enumerate(matched):
                    det = mobj.get("detection")
                    mask = mobj.get("mask")
                    if mask is None:
                        continue
                    _dri = int((det or {}).get("region_index", 0) or 0)
                    _rc = None
                    if _dri > 0 and _dri in region_by_index:
                        _rg = region_by_index[_dri]
                        _rc = {
                            "type": _rg.get("type", ""),
                            "depth_stats": _rg.get("depth_stats", {}),
                            "sigma_scope": getattr(self, "depth_sigma_clip_scope", "mask"),
                        }
                    depth_stats, coords_3d, centroid = self._mask_depth_stats_and_3d_cached(
                        mobj,
                        metric_depth,
                        K,
                        mask,
                        det,
                        use_erosion=True,
                        region_context=_rc,
                        label_map=region_label_map,
                        region_index=_dri,
                    )
                    _do_erosion_cmp = bool(getattr(self.config, "depth_erosion_comparison", True)) if self.config else True
                    if _do_erosion_cmp:
                        depth_stats_raw, coords_3d_raw, centroid_raw = self._mask_depth_stats_and_3d_cached(
                            mobj,
                            metric_depth,
                            K,
                            mask,
                            det,
                            use_erosion=False,
                            region_context=_rc,
                            label_map=region_label_map,
                            region_index=_dri,
                        )
                    else:
                        depth_stats_raw, coords_3d_raw, centroid_raw = None, None, None
                    source_model = det.get("source_model", "") if det else ""
                    obj_entry = {
                        "id": f"{track_key}_obj_{i}_{source_model if source_model else 'mask'}",
                        "entity_kind": "object",
                        "label": det.get("label", "unknown") if det else "unlabeled",
                        "bbox": det.get("bbox", []) if det else list(mobj.get("mask_bbox_xyxy", [0, 0, 0, 0])),
                        "source_model": source_model,
                        "segmentor": str(det.get("segmentor", source_model)) if det else source_model,
                        "sam2_mask_index": mobj.get("sam2_mask_index", -1),
                        "mask_path": None,
                        "masked_depth_path": None,
                        "depth_stats": depth_stats,
                        "coordinates_3d_from_mask": coords_3d,
                        "mask_centroid_2d": centroid,
                        "depth_stats_no_erosion": depth_stats_raw,
                        "coordinates_3d_no_erosion": coords_3d_raw,
                        "mask_centroid_2d_no_erosion": centroid_raw,
                    }
                    json_objects.append(obj_entry)
                if self.regions_enabled and region_label_map is not None and region_partition_meta:
                    lm_i = np.asarray(region_label_map, dtype=np.int32)
                    for r in region_partition_meta:
                        r_ix = int(r.get("region_index", 0) or 0)
                        if r_ix <= 0:
                            continue
                        mask_r = lm_i == r_ix
                        if not np.any(mask_r):
                            continue
                        _rc = {
                            "type": r.get("type", ""),
                            "depth_stats": r.get("depth_stats", {}),
                            "sigma_scope": getattr(self, "depth_sigma_clip_scope", "mask"),
                        }
                        depth_stats, coords_3d, centroid = self._mask_depth_stats_and_3d_cached(
                            r,
                            metric_depth,
                            K,
                            mask_r,
                            None,
                            use_erosion=True,
                            region_context=_rc,
                            label_map=region_label_map,
                            region_index=r_ix,
                        )
                        _do_erosion_cmp = bool(getattr(self.config, "depth_erosion_comparison", True)) if self.config else True
                        if _do_erosion_cmp:
                            depth_stats_raw, coords_3d_raw, centroid_raw = self._mask_depth_stats_and_3d_cached(
                                r,
                                metric_depth,
                                K,
                                mask_r,
                                None,
                                use_erosion=False,
                                region_context=_rc,
                                label_map=region_label_map,
                                region_index=r_ix,
                            )
                        else:
                            depth_stats_raw, coords_3d_raw, centroid_raw = None, None, None
                        bx = r.get("bbox_px") or [0, 0, w - 1, h - 1]
                        json_objects.append({
                            "id": str(r.get("id", f"region_{r_ix}")),
                            "entity_kind": "region",
                            "label": str(r.get("type", "region")),
                            "bbox": [int(v) for v in bx[:4]],
                            "source_model": "depth_partition",
                            "segmentor": "depth_partition",
                            "sam2_mask_index": -1,
                            "mask_path": None,
                            "masked_depth_path": None,
                            "depth_stats": depth_stats,
                            "coordinates_3d_from_mask": coords_3d,
                            "mask_centroid_2d": centroid,
                            "depth_stats_no_erosion": depth_stats_raw,
                            "coordinates_3d_no_erosion": coords_3d_raw,
                            "mask_centroid_2d_no_erosion": centroid_raw,
                        })
                mapping_path = sam2_sg_dir / f"{path.stem}_depth_mask_mapping_{mode}.png"
                self._save_depth_mask_mapping_image(metric_depth, matched, mapping_path)
                mapping_rel = f"scene_graph/{track_dir_name}/{path.stem}_depth_mask_mapping_{mode}.png"
                dm_json = self._build_depth_mask_json(
                    image_path=str(path.resolve()),
                    path_stem=path.stem,
                    timestamp=timestamp,
                    image_size=[w, h],
                    matching_mode=mode,
                    depth_map_path=depth_map_npy_rel,
                    depth_map_image_path=depth_map_image_rel,
                    depth_global_min=depth_global_min,
                    depth_global_max=depth_global_max,
                    depth_global_mean=depth_global_mean,
                    segmentation_map_image_path=seg_map_rel,
                    num_auto_masks=len(track_masks),
                    mapping_image_path=mapping_rel,
                    objects=json_objects,
                )
                self._write_json(dm_json, sam2_sg_dir / f"{path.stem}_depth_mask_{mode}.json")

            _t_enrich = time.perf_counter()
            if (
                self.regions_enabled
                and region_partition_meta
                and region_label_map is not None
                and not region_labeller_enrich
            ):
                region_labeller_enrich = self._enrich_region_labels_from_masks(
                    img_bgr, region_partition_meta, region_label_map, w, h,
                    metric_depth=metric_depth,
                    K=K,
                )
            try:
                self._timing_logger.record(
                    _image_key_for_timing,
                    "enrich_regions",
                    (time.perf_counter() - _t_enrich) * 1000.0,
                )
            except Exception:
                pass
            _t_objects = time.perf_counter()
            objects_3d = []
            for i, det in enumerate(track_dets):
                bbox = det["bbox"]
                src = det.get("source_model", "Unknown")
                graph_id = det.get("graph_id", f"{track_key}_obj_{i}_{src}")
                confidence = round(float(det.get("conf", 0.0)), 4)
                gdino_confidence = round(float(det.get("grounded_sam2_confidence", confidence)), 4)
                grounded_label = str(det.get("grounded_sam2_label", det.get("label", "object"))).strip().lower()
                bbox_int = [int(round(v)) for v in bbox[:4]]
                name_fields = self._choose_mask_name_fields(
                    grounded_label=grounded_label,
                    grounded_caption=str(det.get("caption", grounded_label)),
                    grounded_confidence=gdino_confidence,
                    florence_label=str(det.get("florence2_label", "")),
                    florence_caption=str(det.get("florence2_caption", "")),
                    rampp_label=str(det.get("rampp_label", "")),
                    rampp_caption=str(det.get("rampp_caption", "")),
                    rampp_tags=list(det.get("rampp_tags", [])),
                    fallback_label=str(det.get("label", "object")).strip().lower(),
                )
                mobj = matched_A_lookup[graph_id]
                mask = mobj["mask"]
                _dri = int(det.get("region_index", 0) or 0)
                _rc = None
                if _dri > 0 and _dri in region_by_index:
                    _rg = region_by_index[_dri]
                    _rc = {
                        "type": _rg.get("type", ""),
                        "depth_stats": _rg.get("depth_stats", {}),
                        "sigma_scope": getattr(self, "depth_sigma_clip_scope", "mask"),
                    }
                depth_stats, coords_3d, centroid = self._mask_depth_stats_and_3d_cached(
                    mobj,
                    metric_depth,
                    K,
                    mask,
                    det,
                    use_erosion=True,
                    region_context=_rc,
                    label_map=region_label_map,
                    region_index=_dri,
                )
                _do_erosion_cmp = bool(getattr(self.config, "depth_erosion_comparison", True)) if self.config else True
                if _do_erosion_cmp:
                    depth_stats_raw, coords_3d_raw, centroid_raw = self._mask_depth_stats_and_3d_cached(
                        mobj,
                        metric_depth,
                        K,
                        mask,
                        det,
                        use_erosion=False,
                        region_context=_rc,
                        label_map=region_label_map,
                        region_index=_dri,
                    )
                else:
                    depth_stats_raw, coords_3d_raw, centroid_raw = None, None, None
                _rel3 = {"x": 0.0, "y": 0.0, "z": 0.0}
                _rdp = 0.0
                if _dri > 0 and _dri in region_by_index:
                    _rg = region_by_index[_dri]
                    _cx2, _cy2 = _rg.get("centroid_2d_px", [centroid[0], centroid[1]])
                    _zm = float((_rg.get("depth_stats") or {}).get("mean", coords_3d.get("z", 0.0)))
                    _rc3 = self._back_project(int(_cx2), int(_cy2), _zm, K)
                    _rel3 = {
                        "x": round(float(coords_3d["x"] - _rc3["x"]), 4),
                        "y": round(float(coords_3d["y"] - _rc3["y"]), 4),
                        "z": round(float(coords_3d["z"] - _rc3["z"]), 4),
                    }
                    _band = _rg.get("depth_band_m") or [0.0, 1.0]
                    _lo, _hi = float(_band[0]), float(_band[1])
                    if _hi > _lo:
                        _rdp = float(np.clip((coords_3d["z"] - _lo) / (_hi - _lo), 0.0, 1.0))
                    else:
                        _rdp = 0.5
                obj_entry = {
                    "id": graph_id,
                    "region_index": _dri,
                    "region_id": det.get("region_id", ""),
                    "coordinates_3d_region_relative": _rel3,
                    "region_depth_percentile": round(_rdp, 4),
                    "label": str(det.get("label", "object")).strip().lower(),
                    "confidence": confidence,
                    "conf": confidence,
                    "bbox": bbox_int,
                    "segmentor": str(det.get("segmentor", src)),
                    "coordinates_3d": coords_3d,
                    "depth_stats": depth_stats,
                    "mask_centroid_2d": centroid,
                    "coordinates_3d_no_erosion": coords_3d_raw,
                    "depth_stats_no_erosion": depth_stats_raw,
                    "mask_centroid_2d_no_erosion": centroid_raw,
                    "sam2_mask_index": mobj["sam2_mask_index"],
                    "mask_matched": True,
                    "mask_path": None,
                    "depth_map_path": None,
                    **name_fields,
                    "layer_type": "unassigned",
                    "parent_object_id": None,
                    "child_object_ids": [],
                    "part_mask_ids": [],
                    "contains": [],
                    "contained_by": [],
                    "occludes": [],
                    "occluded_by": [],
                    "sources": {
                        "GroundedSAM2": {
                            "caption": str(det.get("caption", grounded_label)),
                            "label": grounded_label,
                            "confidence": gdino_confidence,
                        },
                        "Florence2": {
                            "label": str(det.get("florence2_label", "")),
                            "caption": str(det.get("florence2_caption", "")),
                        },
                        "RAM++": {
                            "label": str(det.get("rampp_label", "")),
                            "caption": str(det.get("rampp_caption", "")),
                            "tags": list(det.get("rampp_tags", [])),
                        },
                        "Pix2SG": {"relations": []},
                    },
                    "_sam2_mask_array": mask,
                }
                _mask_rle = self._encode_mask_rle(mask)
                if _mask_rle is not None:
                    obj_entry["mask_rle"] = _mask_rle
                objects_3d.append(obj_entry)

            try:
                self._timing_logger.record(
                    _image_key_for_timing,
                    "objects_3d_build",
                    (time.perf_counter() - _t_objects) * 1000.0,
                )
            except Exception:
                pass
            _tpx = time.perf_counter()
            pix2sg_out = self.pix2sg.predict(img_bgr, image_stem=path.stem, detections=objects_3d, iou_func=self._bbox_iou_xyxy)
            if _profile:
                prof["pix2sg_predict_s"] = prof.get("pix2sg_predict_s", 0.0) + (time.perf_counter() - _tpx)
            try:
                self._timing_logger.record(_image_key_for_timing, "pix2sg_predict", (time.perf_counter() - _tpx) * 1000.0)
            except Exception:
                pass
            pix2sg_stats = self._attach_relations_by_triplets(objects_3d, pix2sg_out, "Pix2SG")
            if self.regions_enabled:
                self._apply_regions_plausibility_to_objects(objects_3d)
            print(
                f"Track[{track_key}] relation attach: "
                f"Pix2SG(attached={pix2sg_stats['attached']}/{pix2sg_stats['input_triplets']}, "
                f"sub_id={pix2sg_stats['subject_id_matched']}, sub_label={pix2sg_stats['subject_label_matched']})"
            )
            if self.require_any_relation_source and len(track_dets) >= 2 and pix2sg_stats["input_triplets"] == 0:
                relation_status = self._collect_relation_source_status()
                details = "; ".join(
                    f"{name}: active={status.get('active')} backend={status.get('backend')} reason={status.get('reason', '')}"
                    for name, status in relation_status.items()
                )
                raise RuntimeError(
                    "No relation triplets were produced by Pix2SG despite multiple detections. "
                    f"Diagnostics: {details}"
                )

            K_serializable = {k: float(v) for k, v in K.items()} if K else None
            models_used_sam2 = ["GroundedSAM2"]
            if self.florence2 is not None and self.florence2.active:
                models_used_sam2.append("Florence-2")
            if self.rampp is not None and self.rampp.active:
                models_used_sam2.append("RAM++")
            if self.pix2sg.is_active():
                models_used_sam2.append("Pix2SG")

            sam2_metadata = {
                "timestamp": timestamp,
                "segmentor": "SAM2",
                "track": track_key,
                "intrinsics": K_serializable,
                "models": models_used_sam2,
                "rampp_tags": _rampp_tags_for_metadata,
                "gdino_query_used": _gdino_query_used,
                "relation_sources": self._collect_relation_source_status(),
                "relation_debug": {
                    "pix2sg": pix2sg_stats,
                    "mask_iou_match_thresh": float(self.mask_iou_match_thresh),
                    "pix2sg_mask_overlap_thresh": float(self.pix2sg_mask_overlap_thresh),
                    "pix2sg_depth_near_threshold": float(self.pix2sg_depth_near_threshold),
                    "raw_triplets": {"pix2sg": int(len(pix2sg_out))},
                    "num_detected_objects": int(len(track_dets)),
                    "num_mask_matched": int(sum(1 for o in objects_3d if o.get("mask_matched"))),
                },
                "depth_map": depth_map_image_rel,
                "segmentation_image": seg_map_rel,
                "relations_json": f"scene_graph/{track_dir_name}/{path.stem}_relations.json",
                "mask_hierarchy_json": f"scene_graph/{track_dir_name}/{path.stem}_mask_hierarchy.json",
                "layers_json": f"scene_graph/{track_dir_name}/{path.stem}_layers.json",
                "layers_image": f"scene_graph/{track_dir_name}/{path.stem}_layers.png",
                "mask_hierarchy_image": f"scene_graph/{track_dir_name}/{path.stem}_mask_hierarchy.png",
                "depth_mask_A_json": f"scene_graph/{track_dir_name}/{path.stem}_depth_mask_A.json",
                "depth_mask_B_json": "",
                "relations_map_image": f"scene_graph/{track_dir_name}/{path.stem}_relations_map.png",
                "relations_map_objects_image": f"scene_graph/{track_dir_name}/{path.stem}_relations_map_objects.png",
                "relations_map_regions_image": f"scene_graph/{track_dir_name}/{path.stem}_relations_map_regions.png",
                "sam2_segmentation_image": f"scene_graph/{track_dir_name}/{path.stem}_sam2_segmentation.png",
                "sam2_tinted_overlay_image": f"scene_graph/{track_dir_name}/{path.stem}_sam2_tinted_overlay.png",
            }

            _t_seg_overlay = time.perf_counter()
            self._save_labelled_segmentation(objects_3d, sam2_sg_dir / f"{path.stem}_sam2_segmentation.png")
            self._save_labelled_tinted_overlay(objects_3d, img_rgb, sam2_sg_dir / f"{path.stem}_sam2_tinted_overlay.png")
            try:
                self._timing_logger.record(
                    _image_key_for_timing,
                    "viz_segmentation_overlay",
                    (time.perf_counter() - _t_seg_overlay) * 1000.0,
                )
            except Exception:
                pass

            rlist_for_derive: Optional[List[Dict[str, Any]]] = None
            region_hier_supp: Optional[List[Dict[str, Any]]] = None
            if (
                self.regions_enabled
                and region_label_map is not None
                and region_partition_meta
                and shared_regions_png_path is not None
                and shared_regions_png_path.exists()
            ):
                rlist_for_derive = copy.deepcopy(region_partition_meta)
                for _reg in rlist_for_derive:
                    _reg["object_ids"] = []
                for _obj in objects_3d:
                    _ori = int(_obj.get("region_index", 0) or 0)
                    if _ori > 0:
                        for _reg in rlist_for_derive:
                            if int(_reg.get("region_index", -1)) == _ori:
                                _reg["object_ids"].append(str(_obj.get("id")))
                                break
                for _reg in rlist_for_derive:
                    _rix = int(_reg.get("region_index", 0) or 0)
                    if _rix in region_labeller_enrich:
                        self._merge_region_labeller_enrichment(_reg, region_labeller_enrich[_rix])
                region_hier_supp = self._build_region_hierarchy_supplements(rlist_for_derive, region_label_map)

            relations, mask_hierarchy, layers = self._derive_scene_additions(
                objects_3d,
                region_hierarchy_supplements=region_hier_supp,
                region_metas_for_layers=rlist_for_derive,
            )

            regions_block: Dict[str, Any] = {}
            if (
                self.regions_enabled
                and region_label_map is not None
                and shared_regions_png_path is not None
                and shared_regions_png_path.exists()
            ):
                dst_rpng = sam2_sg_dir / f"{path.stem}_regions.png"
                self._copy_or_link_file(shared_regions_png_path, dst_rpng, self._regions_use_hardlink)
                rlist = rlist_for_derive if rlist_for_derive is not None else copy.deepcopy(region_partition_meta)
                if rlist_for_derive is None:
                    for _reg in rlist:
                        _reg["object_ids"] = []
                    for _obj in objects_3d:
                        _ori = int(_obj.get("region_index", 0) or 0)
                        if _ori > 0:
                            for _reg in rlist:
                                if int(_reg.get("region_index", -1)) == _ori:
                                    _reg["object_ids"].append(str(_obj.get("id")))
                                    break
                    for _reg in rlist:
                        _rix = int(_reg.get("region_index", 0) or 0)
                        if _rix in region_labeller_enrich:
                            self._merge_region_labeller_enrichment(_reg, region_labeller_enrich[_rix])
                rr_edges = self._build_region_region_relations_from_meta(rlist, region_label_map)
                # Phase 5: pass shared q1/q2 from _build_layers_payload so regions
                # and objects are always classified against identical thresholds.
                _shared_q1 = layers.get("depth_quantiles", {}).get("foreground_max_z")
                _shared_q2 = layers.get("depth_quantiles", {}).get("midground_max_z")
                region_layers = self._build_region_layers_payload(rlist, q1=_shared_q1, q2=_shared_q2)
                # Phase 6: build mask-based adjacency graph — replaces the linear
                # depth-order chain as the primary region structure.
                region_adjacency = self._build_region_adjacency_graph(
                    rlist,
                    region_label_map,
                    min_border_px=int(getattr(self.config, "region_adjacency_min_border_px", 10)),
                    dilation_px=int(getattr(self.config, "region_adjacency_dilation_px", 3)),
                )
                # Keep legacy hierarchy payload for one release so existing readers
                # are not immediately broken. It is marked deprecated in the output.
                region_hierarchy = self._build_region_hierarchy_payload(rlist)
                region_hierarchy["deprecated"] = True
                region_hierarchy["deprecation_note"] = (
                    "Use region_adjacency_graph.json instead. This linear depth-order "
                    "chain will be removed in a future release."
                )
                regions_block = {
                    "image_stem": path.stem,
                    "width": w,
                    "height": h,
                    "depth_model": str(getattr(self.depth_estimator, "backend_name", "")),
                    "regions_json": f"scene_graph/{track_dir_name}/{path.stem}_regions.json",
                    "regions_image": f"scene_graph/{track_dir_name}/{path.stem}_regions.png",
                    "regions_overlay_image": f"scene_graph/{track_dir_name}/{path.stem}_regions_overlay.png",
                    "region_segmentation_image": f"scene_graph/{track_dir_name}/{path.stem}_region_segmentation.png",
                    "region_sam2_segmentation_image": f"scene_graph/{track_dir_name}/{path.stem}_region_sam2_style_segmentation.png",
                    "region_tinted_overlay_image": f"scene_graph/{track_dir_name}/{path.stem}_region_tinted_overlay.png",
                    "region_layers_json": f"scene_graph/{track_dir_name}/{path.stem}_region_layers.json",
                    "region_layers_image": f"scene_graph/{track_dir_name}/{path.stem}_region_layers.png",
                    # Phase 6: adjacency graph is the primary region structure.
                    "region_adjacency_graph_json": f"scene_graph/{track_dir_name}/{path.stem}_region_adjacency_graph.json",
                    # Legacy hierarchy retained for one release (marked deprecated).
                    "region_hierarchy_json": f"scene_graph/{track_dir_name}/{path.stem}_region_hierarchy.json",
                    "region_hierarchy_image": f"scene_graph/{track_dir_name}/{path.stem}_region_hierarchy.png",
                    "palette": region_palette,
                    "regions": rlist,
                    "region_layers": region_layers,
                    "region_adjacency_graph": region_adjacency,
                    "region_hierarchy": region_hierarchy,
                    "region_region_relations": rr_edges,
                }
                self._write_json(regions_block, sam2_sg_dir / f"{path.stem}_regions.json")
                self._write_json(region_layers, sam2_sg_dir / f"{path.stem}_region_layers.json")
                self._write_json(region_adjacency, sam2_sg_dir / f"{path.stem}_region_adjacency_graph.json")
                self._write_json(region_hierarchy, sam2_sg_dir / f"{path.stem}_region_hierarchy.json")
                # S1: separate region-relations artifact (subject / predicate / object triples).
                _rr_rel_list: List[Dict[str, Any]] = []
                for _rre in (rr_edges or []):
                    if isinstance(_rre, dict) and _rre.get("subject") and _rre.get("object"):
                        _rr_rel_list.append({
                            "subject": str(_rre.get("subject", "")),
                            "predicate": str(_rre.get("predicate", "adjacent_to")),
                            "object": str(_rre.get("object", "")),
                            "source_layer": "depth_partition",
                        })
                _rr_payload: Dict[str, Any] = {
                    "image": str(path.resolve()),
                    "stem": path.stem,
                    "timestamp": timestamp,
                    "region_count": len(rlist),
                    "relations": _rr_rel_list,
                }
                self._write_json(_rr_payload, sam2_sg_dir / f"{path.stem}_region_relations.json")
                self._save_regions_overlay(
                    img_bgr,
                    region_label_map,
                    region_palette,
                    sam2_sg_dir / f"{path.stem}_regions_overlay.png",
                    regions_meta=rlist,
                    region_relations=rr_edges,
                )
                if label_map_to_bgr is not None:
                    _reg_seg_bgr = label_map_to_bgr(region_label_map, region_palette)
                    _reg_seg_bgr = self._annotate_region_geometry(_reg_seg_bgr, rlist, rr_edges)
                    cv2.imwrite(str(sam2_sg_dir / f"{path.stem}_region_segmentation.png"), _reg_seg_bgr)
                _reg_vis_objs = self._build_region_visual_objects(rlist, region_label_map)
                if _reg_vis_objs:
                    self._save_labelled_segmentation(
                        _reg_vis_objs,
                        sam2_sg_dir / f"{path.stem}_region_sam2_style_segmentation.png",
                    )
                    self._save_labelled_tinted_overlay(
                        _reg_vis_objs,
                        img_rgb,
                        sam2_sg_dir / f"{path.stem}_region_tinted_overlay.png",
                    )
                    _rsam2 = cv2.imread(str(sam2_sg_dir / f"{path.stem}_region_sam2_style_segmentation.png"))
                    if _rsam2 is not None:
                        _rsam2 = self._annotate_region_geometry(_rsam2, rlist, rr_edges)
                        cv2.imwrite(str(sam2_sg_dir / f"{path.stem}_region_sam2_style_segmentation.png"), _rsam2)
                    _rtint = cv2.imread(str(sam2_sg_dir / f"{path.stem}_region_tinted_overlay.png"))
                    if _rtint is not None:
                        _rtint = self._annotate_region_geometry(_rtint, rlist, rr_edges)
                        cv2.imwrite(str(sam2_sg_dir / f"{path.stem}_region_tinted_overlay.png"), _rtint)
                # Dedicated region-only layer and hierarchy maps for easier diagnostics.
                self._save_layers_map(
                    img_bgr,
                    objects_3d=[],
                    path=sam2_sg_dir / f"{path.stem}_region_layers.png",
                    regions_meta=rlist,
                )
                self._save_mask_hierarchy_map(
                    img_bgr,
                    objects_3d=[],
                    hierarchy=region_hierarchy,
                    path=sam2_sg_dir / f"{path.stem}_region_hierarchy.png",
                    region_supplements=self._build_region_hierarchy_supplements(rlist, region_label_map),
                )
                # S2a: mask_hierarchy_levels.json — regions organised by depth layer.
                _mhl_levels = []
                for _lname in ("foreground", "midground", "background", "far_background"):
                    _lregs = [
                        {"region_id": str(r.get("id", "")), "region_index": int(r.get("region_index", 0) or 0),
                         "label": str(r.get("label") or r.get("type") or ""), "depth_z": float((r.get("depth_stats") or {}).get("z_val", 0.0))}
                        for r in rlist
                        if str(r.get("depth_layer", "") or "").lower() == _lname
                    ]
                    if _lregs:
                        _mhl_levels.append({"layer": _lname, "region_count": len(_lregs), "regions": _lregs})
                # fallback: emit all regions in one unclassified level
                if not _mhl_levels:
                    _mhl_levels = [{"layer": "unclassified", "region_count": len(rlist),
                                    "regions": [{"region_id": str(r.get("id", "")), "region_index": int(r.get("region_index", 0) or 0),
                                                 "label": str(r.get("label") or r.get("type") or ""), "depth_z": 0.0} for r in rlist]}]
                _mhl_payload = {
                    "stem": path.stem, "timestamp": timestamp,
                    "schema": "citv_mask_hierarchy_levels_v1",
                    "level_count": len(_mhl_levels), "levels": _mhl_levels,
                }
                self._write_json(_mhl_payload, sam2_sg_dir / f"{path.stem}_mask_hierarchy_levels.json")
                # S2b: region_relations_map.png — arrows between region centroids labelled by predicate.
                _rrm_canvas = img_bgr.copy()
                _rid_to_centroid: Dict[str, Tuple[int, int]] = {}
                for _r in rlist:
                    _rid = str(_r.get("id", ""))
                    _ridx = int(_r.get("region_index", 0) or 0)
                    if region_label_map is not None and _ridx > 0:
                        _rys, _rxs = np.where(region_label_map == _ridx)
                        if _rxs.size > 0:
                            _rid_to_centroid[_rid] = (int(_rxs.mean()), int(_rys.mean()))
                for _rre in (rr_edges or []):
                    _rsub = str(_rre.get("subject", ""))
                    _robj = str(_rre.get("object", ""))
                    _rpred = str(_rre.get("predicate", ""))
                    _cp = _rid_to_centroid.get(_rsub)
                    _co = _rid_to_centroid.get(_robj)
                    if _cp and _co:
                        cv2.arrowedLine(_rrm_canvas, _cp, _co, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.15)
                        _mx, _my = (_cp[0] + _co[0]) // 2, (_cp[1] + _co[1]) // 2
                        cv2.putText(_rrm_canvas, _rpred[:20], (_mx, _my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 80), 1, cv2.LINE_AA)
                for _rid, _cxy in _rid_to_centroid.items():
                    cv2.circle(_rrm_canvas, _cxy, 6, (80, 255, 80), -1, cv2.LINE_AA)
                    cv2.putText(_rrm_canvas, _rid[:12], (_cxy[0] + 8, _cxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
                cv2.imwrite(str(sam2_sg_dir / f"{path.stem}_region_relations_map.png"), _rrm_canvas)
                # Phase 8: always emit region-region spatial edges into relations.json.
                # Previously gated on _append_region_layer_relations (default False),
                # which silently dropped the only cross-region path information from the
                # main graph, making trajectory refiners dead-end at region boundaries.
                # The flag is retained as a kill-switch only; default is now True in config.
                # Source tag changed from "region_layer" to "region_spatial" for clarity.
                if rr_edges and self._append_region_layer_relations:
                    for _e in rr_edges:
                        _sc = _e.get("subject_centroid") or [0, 0]
                        _oc = _e.get("object_centroid") or [0, 0]
                        _re: Dict[str, Any] = {
                            "subject_id": _e["subject_id"],
                            "subject_label": _e["subject_label"],
                            "subject_name": _e["subject_name"],
                            "predicate": _e["predicate"],
                            "object_id": _e["object_id"],
                            "object_label": _e["object_label"],
                            "object_name": _e["object_name"],
                            "object_caption": _e.get("object_caption", ""),
                            "source": "region_spatial",
                            "score": _e.get("score", 0.6),
                            "relation_tier": _e.get("relation_tier", "region_region"),
                            "subject_centroid": list(_sc),
                            "object_centroid": list(_oc),
                        }
                        if "shared_border_px" in _e:
                            _re["shared_border_px"] = _e["shared_border_px"]
                        relations.append(_re)

            if regions_block:
                sam2_metadata["regions_json"] = f"scene_graph/{track_dir_name}/{path.stem}_regions.json"
                sam2_metadata["regions_image"] = f"scene_graph/{track_dir_name}/{path.stem}_regions.png"
                sam2_metadata["regions_overlay_image"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_regions_overlay.png"
                )
                sam2_metadata["region_segmentation_image"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_segmentation.png"
                )
                sam2_metadata["region_sam2_segmentation_image"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_sam2_style_segmentation.png"
                )
                sam2_metadata["region_tinted_overlay_image"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_tinted_overlay.png"
                )
                sam2_metadata["region_layers_json"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_layers.json"
                )
                sam2_metadata["region_layers_image"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_layers.png"
                )
                # Phase 6: adjacency graph is the primary region structure.
                sam2_metadata["region_adjacency_graph_json"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_adjacency_graph.json"
                )
                sam2_metadata["region_hierarchy_json"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_hierarchy.json"
                )
                sam2_metadata["region_hierarchy_image"] = (
                    f"scene_graph/{track_dir_name}/{path.stem}_region_hierarchy.png"
                )

            _maps_elapsed_s = 0.0
            _paths_elapsed_s = 0.0
            _t_maps = time.perf_counter()
            self._write_json(relations, sam2_sg_dir / f"{path.stem}_relations.json")
            self._write_json(mask_hierarchy, sam2_sg_dir / f"{path.stem}_mask_hierarchy.json")
            # S2: mask hierarchy with containment depth per node.
            def _annotate_hier_depth(node: Any, depth: int) -> Any:
                if isinstance(node, dict):
                    node = dict(node)
                    node["containment_depth"] = depth
                    if isinstance(node.get("children"), list):
                        node["children"] = [_annotate_hier_depth(c, depth + 1) for c in node["children"]]
                elif isinstance(node, list):
                    node = [_annotate_hier_depth(c, depth) for c in node]
                return node
            _hier_detailed = _annotate_hier_depth(mask_hierarchy, 0)
            self._write_json(
                {"stem": path.stem, "hierarchy": _hier_detailed},
                sam2_sg_dir / f"{path.stem}_mask_hierarchy_detailed.json",
            )
            self._write_json(layers, sam2_sg_dir / f"{path.stem}_layers.json")
            _rmeta_for_map = rlist if regions_block else None
            self._save_relations_map(
                img_bgr, objects_3d, relations,
                sam2_sg_dir / f"{path.stem}_relations_map.png",
                regions_meta=_rmeta_for_map,
            )
            _split_views = bool(getattr(self.config, "map_enable_split_views", True)) if self.config else True
            if _split_views:
                self._save_relations_map(
                    img_bgr, objects_3d, relations,
                    sam2_sg_dir / f"{path.stem}_relations_map_objects.png",
                    regions_meta=_rmeta_for_map, view="objects_only",
                )
                self._save_relations_map(
                    img_bgr, objects_3d, relations,
                    sam2_sg_dir / f"{path.stem}_relations_map_regions.png",
                    regions_meta=_rmeta_for_map, view="regions_only",
                )
            self._save_layers_map(
                img_bgr, objects_3d,
                sam2_sg_dir / f"{path.stem}_layers.png",
                regions_meta=_rmeta_for_map,
            )
            self._save_mask_hierarchy_map(
                img_bgr, objects_3d, mask_hierarchy,
                sam2_sg_dir / f"{path.stem}_mask_hierarchy.png",
                region_supplements=region_hier_supp,
            )
            try:
                self._timing_logger.record(
                    _image_key_for_timing,
                    "relations_layers_maps",
                    (time.perf_counter() - _t_maps) * 1000.0,
                )
            except Exception:
                pass
            _maps_elapsed_s = max(0.0, float(time.perf_counter() - _t_maps))

            # Path hypotheses (region/object/mask) — must run before masks are dropped.
            _t_paths = time.perf_counter()
            try:
                from scene_understanding.pathing.export_hook import invoke_path_hypotheses_export_for_track

                _path_exports = invoke_path_hypotheses_export_for_track(
                    self,
                    img_bgr=img_bgr,
                    path_stem=path.stem,
                    track_dir_name=track_dir_name,
                    track_dir=sam2_sg_dir,
                    objects_3d_with_masks=objects_3d,
                    regions_block=regions_block,
                    region_label_map=region_label_map,
                    region_adjacency=region_adjacency if regions_block else None,
                    relations=relations,
                    metric_depth_m=metric_depth,
                )
                if _path_exports:
                    sam2_metadata.update(_path_exports)
            except Exception as _e:
                print(f"Track[{track_key}] path hypotheses export failed: {_e}")
            try:
                self._timing_logger.record(
                    _image_key_for_timing,
                    "path_hypotheses_export",
                    (time.perf_counter() - _t_paths) * 1000.0,
                )
            except Exception:
                pass
            _paths_elapsed_s = max(0.0, float(time.perf_counter() - _t_paths))

            for obj in objects_3d:
                obj.pop("_sam2_mask_array", None)
            sam2_scene_output = {
                "metadata": sam2_metadata,
                "objects": objects_3d,
                "relations": relations,
                "mask_hierarchy": mask_hierarchy,
                "layers": layers,
            }
            if regions_block:
                sam2_scene_output["regions"] = regions_block
            # Flush accumulators *before* snapshotting so the JSON sees them.
            # Previously ``mask_labelling`` / ``track_geometry_exports`` were
            # recorded after ``image_timings(...)``, which meant 95%+ of the
            # wall-clock was missing from the per-image ``_timing`` block and
            # perf analysis was impossible to do from the outputs alone.
            try:
                _track_pipeline_s_now = time.perf_counter() - _tp_track
                self._timing_logger.record(
                    _image_key_for_timing, "mask_labelling", mask_label_acc * 1000.0
                )
                _trest_s = _track_pipeline_s_now - mask_label_acc - _maps_elapsed_s - _paths_elapsed_s
                if _trest_s > 0:
                    self._timing_logger.record(
                        _image_key_for_timing,
                        "track_geometry_exports",
                        _trest_s * 1000.0,
                    )
            except Exception:
                pass
            try:
                _timing_snapshot = self._timing_logger.image_timings(_image_key_for_timing)
            except Exception:
                _timing_snapshot = {}
            if _timing_snapshot:
                sam2_scene_output["_timing"] = _timing_snapshot
            with open(sam2_sg_dir / f"{path.stem}_scene.json", "w") as f:
                json.dump(sam2_scene_output, f, indent=2)
            print(f"Track[{track_key}] scene graph saved: scene_graph/{track_dir_name}/{path.stem}_scene.json")
            generated_tracks.append({
                "track": track_key,
                "scene_json": f"scene_graph/{track_dir_name}/{path.stem}_scene.json",
                "relations_json": f"scene_graph/{track_dir_name}/{path.stem}_relations.json",
                "relations_map_image": f"scene_graph/{track_dir_name}/{path.stem}_relations_map.png",
                "relations_map_objects_image": f"scene_graph/{track_dir_name}/{path.stem}_relations_map_objects.png",
                "relations_map_regions_image": f"scene_graph/{track_dir_name}/{path.stem}_relations_map_regions.png",
            })

            _par_seg = (
                f"scene_graph/{track_dir_name}/{path.stem}_region_segmentation.png" if regions_block else ""
            )
            _par_sam2 = (
                f"scene_graph/{track_dir_name}/{path.stem}_region_sam2_style_segmentation.png"
                if regions_block
                else ""
            )
            _par_tint = (
                f"scene_graph/{track_dir_name}/{path.stem}_region_tinted_overlay.png" if regions_block else ""
            )

            _t_captions = time.perf_counter()
            if self.export_hybrid_captions:
                self._save_caption_variants_for_track(
                    out_dir=scene_graph_dir,
                    image_path=str(path.resolve()),
                    path_stem=path.stem,
                    track_key=track_key,
                    track_dir_name=track_dir_name,
                    objects_3d=objects_3d,
                    scene_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_scene.json",
                    relations_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_relations.json",
                    layers_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_layers.json",
                    hierarchy_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_mask_hierarchy.json",
                    depth_mask_a_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_depth_mask_A.json",
                    depth_mask_b_json_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_depth_mask_B.json"
                        if "B" in self.depth_mask_modes else ""
                    ),
                    segmentation_rel=f"scene_graph/{track_dir_name}/{path.stem}_sam2_segmentation.png",
                    tinted_overlay_rel=f"scene_graph/{track_dir_name}/{path.stem}_sam2_tinted_overlay.png",
                    relations_map_rel=f"scene_graph/{track_dir_name}/{path.stem}_relations_map.png",
                    layers_png_rel=f"scene_graph/{track_dir_name}/{path.stem}_layers.png",
                    hierarchy_png_rel=f"scene_graph/{track_dir_name}/{path.stem}_mask_hierarchy.png",
                    regions_json_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_regions.json" if regions_block else ""
                    ),
                    regions_png_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_regions.png" if regions_block else ""
                    ),
                    regions_overlay_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_regions_overlay.png"
                        if regions_block
                        else ""
                    ),
                    region_segmentation_rel=_par_seg,
                    region_sam2_seg_rel=_par_sam2,
                    region_tinted_rel=_par_tint,
                )
                print(f"Track[{track_key}] Florence-only, fusion-only, and hybrid caption files exported.")
            elif self.export_caption_prompt_bundle:
                self._save_track_prompt_bundle(
                    out_dir=scene_graph_dir,
                    image_path=str(path.resolve()),
                    path_stem=path.stem,
                    track_key=track_key,
                    track_dir_name=track_dir_name,
                    scene_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_scene.json",
                    relations_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_relations.json",
                    layers_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_layers.json",
                    hierarchy_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_mask_hierarchy.json",
                    depth_mask_a_json_rel=f"scene_graph/{track_dir_name}/{path.stem}_depth_mask_A.json",
                    depth_mask_b_json_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_depth_mask_B.json"
                        if "B" in self.depth_mask_modes else ""
                    ),
                    segmentation_rel=f"scene_graph/{track_dir_name}/{path.stem}_sam2_segmentation.png",
                    tinted_overlay_rel=f"scene_graph/{track_dir_name}/{path.stem}_sam2_tinted_overlay.png",
                    relations_map_rel=f"scene_graph/{track_dir_name}/{path.stem}_relations_map.png",
                    layers_png_rel=f"scene_graph/{track_dir_name}/{path.stem}_layers.png",
                    hierarchy_png_rel=f"scene_graph/{track_dir_name}/{path.stem}_mask_hierarchy.png",
                    regions_json_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_regions.json" if regions_block else ""
                    ),
                    regions_png_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_regions.png" if regions_block else ""
                    ),
                    regions_overlay_rel=(
                        f"scene_graph/{track_dir_name}/{path.stem}_regions_overlay.png"
                        if regions_block
                        else ""
                    ),
                    region_segmentation_rel=_par_seg,
                    region_sam2_seg_rel=_par_sam2,
                    region_tinted_rel=_par_tint,
                )
            try:
                self._timing_logger.record(
                    _image_key_for_timing,
                    "caption_exports",
                    (time.perf_counter() - _t_captions) * 1000.0,
                )
            except Exception:
                pass

            viz_sam2 = img_bgr.copy()
            for obj in objects_3d:
                bbox = obj["bbox"]
                color = (0, 255, 0)
                label = f"{obj['label']} [M]"
                cv2.rectangle(viz_sam2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(viz_sam2, label, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                mc = obj.get("mask_centroid_2d")
                if mc and len(mc) == 2 and obj.get("mask_matched"):
                    cv2.circle(viz_sam2, (int(mc[0]), int(mc[1])), 3, color, -1)
                cx_a = int(mc[0]) if mc and len(mc) == 2 else (int(bbox[0]) + int(bbox[2])) // 2
                cy_a = int(mc[1]) if mc and len(mc) == 2 else (int(bbox[1]) + int(bbox[3])) // 2
                for source in ["Pix2SG", "SGTR"]:
                    if source in obj.get("sources", {}):
                        for rel in obj["sources"][source].get("relations", []):
                            target_id = rel["target_id"]
                            if isinstance(target_id, str) and target_id.startswith("external_"):
                                continue
                            target = next((o for o in objects_3d if o["id"] == target_id), None)
                            if target:
                                bbox_b = target["bbox"]
                                mc_b = target.get("mask_centroid_2d")
                                cx_b = int(mc_b[0]) if mc_b and len(mc_b) == 2 else (int(bbox_b[0]) + int(bbox_b[2])) // 2
                                cy_b = int(mc_b[1]) if mc_b and len(mc_b) == 2 else (int(bbox_b[1]) + int(bbox_b[3])) // 2
                                cv2.line(viz_sam2, (cx_a, cy_a), (cx_b, cy_b), (0, 255, 255), 1)
                                cv2.putText(viz_sam2, rel["predicate"], ((cx_a + cx_b) // 2, (cy_a + cy_b) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imwrite(str(sam2_sg_dir / f"{path.stem}_3d_viz.png"), viz_sam2)

            sam2_track_masks = track_masks
            sam2_track_dets = track_dets

        _track_pipeline_s = time.perf_counter() - _tp_track
        if _profile:
            prof["track_pipeline_s"] = _track_pipeline_s
            prof["mask_labelling_s"] = mask_label_acc
            _trest = _track_pipeline_s - mask_label_acc
            if _trest > 0:
                prof["track_geometry_exports_s"] = _trest
        try:
            self._timing_logger.record(_image_key_for_timing, "mask_labelling", mask_label_acc * 1000.0)
            _trest_s = _track_pipeline_s - mask_label_acc - _maps_elapsed_s - _paths_elapsed_s
            if _trest_s > 0:
                self._timing_logger.record(_image_key_for_timing, "track_geometry_exports", _trest_s * 1000.0)
        except Exception:
            pass

        _cfg = getattr(self, "config", None)
        if bool(getattr(_cfg, "unload_labellers_after_each_image", False)):
            self._unload_labellers()

        if self.export_track_comparison_prompt:
            self._save_track_comparison_prompt(
                scene_graph_dir=scene_graph_dir,
                image_path=str(path.resolve()),
                path_stem=path.stem,
                available_tracks=generated_tracks,
            )

        # Phase F — block until all async JSON/PNG writes finish so the
        # per-image deliverables are durable on disk before we move on to
        # the next frame (and before the wall-clock total is reported).
        try:
            self._drain_writers()
        except Exception as _we:
            print(f"  [Writer] pending write failed: {_we}")

        wall_total_s = time.perf_counter() - _t_wall0
        try:
            self._timing_logger.record(_image_key_for_timing, "wall_total", wall_total_s * 1000.0)
        except Exception:
            pass
        print(f"Results saved to {out}")
        print(f"  [Timing] wall total: {wall_total_s:.3f}s")
        if _profile and prof:
            parts = [f"{k}={v:.3f}s" for k, v in sorted(prof.items())]
            print(f"  [Profile] {' '.join(parts)}")
            print(
                "  [Profile] note: mask_labelling_s and pix2sg_predict_s are included inside track_pipeline_s."
            )

        # Release large arrays so memory is available before the next image
        try:
            del sam2_track_masks, sam2_track_dets
        except NameError:
            pass
        del img_bgr, img_rgb, metric_depth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()

if __name__ == "__main__":
    import argparse
    import importlib

    from config import PreprocessConfig

    _pipe_mod = importlib.import_module("scene_understanding.pipeline")
    SceneUnderstandingPipeline = _pipe_mod.SceneUnderstandingPipeline
    resolve_scene_pipeline_mode = _pipe_mod.resolve_scene_pipeline_mode

    parser = argparse.ArgumentParser(description="Run scene understanding pipeline (depth + Grounded SAM2).")
    parser.add_argument("--input_dir", type=str, default="images", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output_scene", help="Output directory for scene graphs")
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default=None,
        choices=("legacy", "staged"),
        help="Override PreprocessConfig.scene_pipeline_mode (default: legacy). Env CITV_SCENE_PIPELINE_MODE still wins.",
    )
    args = parser.parse_args()

    print("Testing pipeline initialization...")
    cfg = PreprocessConfig()
    if bool(getattr(cfg, "cpu_benchmark_mode", False)):
        cfg.device = "cpu"  # type: ignore[assignment]
        cfg.models_dtype = "fp32"  # type: ignore[assignment]
        cfg.rampp_prefer_mps = False  # type: ignore[assignment]
        print("CPU benchmark mode enabled: forcing device=cpu, dtype=fp32, no MPS preference.")
    if args.pipeline_mode is not None:
        cfg.scene_pipeline_mode = args.pipeline_mode  # type: ignore[assignment]
    print("Mode: depth + Grounded SAM2 (GDINO + SAM2).")
    _eff = resolve_scene_pipeline_mode(cfg)
    print(
        f"Effective scene pipeline mode: {_eff!r} "
        f"(config.scene_pipeline_mode={cfg.scene_pipeline_mode!r}; "
        "env CITV_SCENE_PIPELINE_MODE overrides config when set)"
    )
    if _eff == "staged":
        _slim = str(os.getenv("CITV_STAGED_MODULAR_CHAIN_ONLY", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if _slim:
            print("  Staged: slim modular chain only → scene_graph/staged/{stem}_scene.json (CITV_STAGED_MODULAR_CHAIN_ONLY).")
        else:
            print("  Staged: full legacy-equivalent outputs (scene_graph/grounded_sam2/ …) via scene_understanding.stages.full_run.")
    images_dir = Path(args.input_dir)
    output_dir = args.output_dir
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}

    try:
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir.resolve()}")

        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_exts
        )
        if not image_paths:
            raise FileNotFoundError(
                f"No supported images found in {images_dir.resolve()} "
                f"for extensions: {sorted(supported_exts)}"
            )

        # Pass first image so CLIP can classify indoor/outdoor before loading
        # the depth model. Without this, 'auto' always defaults to indoor.
        depth_estimator = DepthEstimator(cfg, first_image=image_paths[0])
        pipeline = SceneUnderstandingPipeline(depth_estimator, config=cfg)

        # Attach a per-run timing logger so logs/timing_log.md + .csv get
        # appended one section per folder run, and each scene JSON gains
        # its _timing block. Output-preserving.
        try:
            from scene_understanding.observability import RunTimingLogger, hash_config
            _timing_log_dir = Path(getattr(cfg, "timing_log_dir", "logs"))
            _run_logger = RunTimingLogger(
                log_dir=_timing_log_dir,
                device=str(getattr(cfg, "device", "cpu")),
                models_dtype=str(getattr(cfg, "models_dtype", "fp32")),
                workers=int(getattr(cfg, "pipeline_max_workers", 1)),
                config_hash=hash_config(cfg),
            )
            if hasattr(pipeline, "attach_timing_logger"):
                pipeline.attach_timing_logger(_run_logger)
        except Exception as _te:
            print(f"  [Timing] logger setup skipped: {_te}")
            _run_logger = None

        print(f"Found {len(image_paths)} images in {images_dir.resolve()}")

        # Phase G — cross-image parallelism via multiprocessing.Pool when
        # ``cfg.pipeline_max_workers > 1``. Default of 1 keeps the original
        # sequential behaviour (and the single shared pipeline) untouched.
        # MPS does not support sharing a context between processes, so each
        # worker builds its own pipeline; this is costly at startup so use
        # only on folders with enough images to amortise it.
        _max_workers = int(getattr(cfg, "pipeline_max_workers", 1) or 1)
        if _max_workers > 1 and len(image_paths) > 1:
            # The worker must build its own pipeline; tear down the main
            # one first so its models release VRAM before the pool forks.
            try:
                pipeline.close()
            except Exception:
                pass
            del pipeline
            del depth_estimator
            import gc as _gc
            _gc.collect()

            from multiprocessing import get_context
            ctx = get_context("spawn")  # MPS / HF models are fork-unsafe
            chunk_count = min(_max_workers, len(image_paths))
            chunks: List[List[Path]] = [[] for _ in range(chunk_count)]
            for _i, _p in enumerate(image_paths):
                chunks[_i % chunk_count].append(_p)
            worker_args = [(cfg, chunk, output_dir) for chunk in chunks if chunk]
            print(f"  [Multiproc] dispatching {len(image_paths)} images across {len(worker_args)} workers")
            with ctx.Pool(processes=len(worker_args)) as pool:
                worker_results = pool.map(_pipeline_worker_entrypoint, worker_args)
            ok = 0
            fail = 0
            for batch in worker_results:
                for item in batch:
                    if item[0] == "__timing__":
                        print(f"  [Timing][worker] {item[2]}")
                        continue
                    if item[1]:
                        ok += 1
                    else:
                        fail += 1
                        print(f"  [Multiproc] {item[0]} failed: {item[2]}")
            print(f"  [Multiproc] done — ok={ok} fail={fail}")
        else:
            _n_imgs = len(image_paths)
            for _img_idx, img_path in enumerate(image_paths):
                # Tell ``_release_heavy_models_after_segmentation`` whether
                # another image is queued in this process. If so, it keeps
                # Depth/GDINO/SAM2 resident to avoid paying the ~4 s reload
                # cost at the start of every image's segmentation stage.
                pipeline._has_more_images_queued = (_img_idx < _n_imgs - 1)
                print(f"Processing image: {img_path.name} ({_img_idx + 1}/{_n_imgs})")
                try:
                    pipeline.process_image(str(img_path), str(output_dir))
                except ValueError as e:
                    print(f"Skipping {img_path.name}: {e}")
            pipeline._has_more_images_queued = False

        if _max_workers <= 1 and _run_logger is not None:
            try:
                md_path, csv_path = _run_logger.finalize()
                print(f"  [Timing] wrote {md_path} and {csv_path}")
            except Exception as _te:
                print(f"  [Timing] finalize failed: {_te}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
