"""Config-driven Grounding DINO query refresh (RAM++ tags → dynamic text query)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _default_query_from_wrapper(sam2_wrapper: Any) -> str:
    return str(getattr(sam2_wrapper, "text_query", "") or "")


def refresh_gdino_query_if_configured(
    pipeline: Any,
    img_rgb: np.ndarray,
    region_partition_meta: List[Dict[str, Any]],
    width: int,
    height: int,
) -> Tuple[List[str], str]:
    """
    Optionally run RAM++ and update ``sam2_wrapper`` text query.

    Returns ``(tags_for_metadata, query_used)``.
    """
    cfg = getattr(pipeline, "config", None)
    mode = str(getattr(cfg, "query_builder_mode", "rampp_full") or "rampp_full").strip().lower()

    if mode == "static":
        return [], _default_query_from_wrapper(pipeline.sam2_wrapper)

    rampp_on = bool(getattr(pipeline, "_rampp_enabled", False))
    force_rampp = mode in ("rampp_full", "rampp_region_crops")
    if mode == "inherit":
        rampp = getattr(pipeline, "rampp", None)
        if rampp is None or not getattr(rampp, "active", False):
            # In inherit mode we do not instantiate RAM++; only reuse an already-live instance.
            return [], _default_query_from_wrapper(pipeline.sam2_wrapper)
    if not force_rampp and not rampp_on:
        return [], _default_query_from_wrapper(pipeline.sam2_wrapper)
    if bool(getattr(pipeline, "_rampp_runtime_disabled", False)):
        return [], _default_query_from_wrapper(pipeline.sam2_wrapper)
    tags_meta: List[str] = []
    query_used = _default_query_from_wrapper(pipeline.sam2_wrapper)

    use_region_crops = bool(getattr(pipeline, "_regions_rampp_crops_enabled", False)) and bool(region_partition_meta)
    if mode == "rampp_region_crops":
        use_region_crops = bool(region_partition_meta)
    elif mode == "rampp_full":
        use_region_crops = False
    elif mode == "inherit":
        use_region_crops = use_region_crops and bool(region_partition_meta)
    else:
        use_region_crops = use_region_crops and bool(region_partition_meta)

    if getattr(pipeline, "rampp", None) is None or not getattr(pipeline.rampp, "active", False):
        ensure = getattr(pipeline, "_ensure_rampp_for_labelling", None)
        if callable(ensure):
            ensure()

    rampp = getattr(pipeline, "rampp", None)
    if rampp is None or not getattr(rampp, "active", False):
        return [], query_used

    max_tags = int(getattr(pipeline, "_rampp_max_tags", 8))
    w, h = width, height

    if use_region_crops:
        # Phase B.5: gather crops up-front and run one batched RAM++ forward
        # instead of N sequential tag_image calls. Output-preserving — union
        # order follows the original iteration order of region_partition_meta.
        import cv2 as _cv2  # local import — prompting.py is otherwise numpy-only.
        crops_bgr: List[np.ndarray] = []
        for reg in region_partition_meta:
            bx = reg.get("bbox_px") or [0, 0, w - 1, h - 1]
            if len(bx) < 4:
                continue
            x1, y1, x2, y2 = [int(max(0, v)) for v in bx[:4]]
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_rgb = img_rgb[y1 : y2 + 1, x1 : x2 + 1]
            if crop_rgb.size == 0:
                continue
            crops_bgr.append(_cv2.cvtColor(crop_rgb, _cv2.COLOR_RGB2BGR))
        union_tags: List[str] = []
        if crops_bgr:
            try:
                batched = rampp.label_crops(crops_bgr)
            except Exception as e:
                print(f"  [RAM++] region crop batched tag failed: {e}; per-crop fallback.")
                batched = None
            if batched is None:
                for _bgr in crops_bgr:
                    try:
                        tr = rampp.label_crop(_bgr)
                        union_tags.extend(list(tr.get("tags", [])))
                    except Exception as e:
                        print(f"  [RAM++] region crop tag failed: {e}")
            else:
                for tr in batched:
                    union_tags.extend(list(tr.get("tags", [])))
        tags_meta = list(dict.fromkeys(union_tags))[: max(max_tags * 2, 16)]
        if tags_meta:
            print(f"  [RAM++] Region-crop tags ({len(tags_meta)})")

    if not tags_meta:
        tag_result = rampp.tag_image(img_rgb)
        tags_meta = list(tag_result.get("tags", []))
        if tags_meta:
            print(f"  [RAM++] Full-image tags: {', '.join(tags_meta)}")

    if tags_meta:
        dynamic_query = ". ".join(tags_meta) + "."
        pipeline.sam2_wrapper.update_text_query(dynamic_query)
        query_used = dynamic_query
        print(f"  [RAM++] GDINO query updated ({len(tags_meta)} tags)")
    else:
        print("  [RAM++] No tags returned — using default GDINO query")

    return tags_meta, query_used


def refresh_gdino_query_for_staged(
    *,
    cfg: Any,
    img_rgb: np.ndarray,
    region_partition_meta: List[Dict[str, Any]],
    width: int,
    height: int,
    seg_pipe: Any,
    rampp: Any,
) -> Tuple[List[str], str]:
    """Staged pipeline: RAM++ tags → update GroundedSAM2 / GDINO text query before ``generate``.

    Same semantics as :func:`refresh_gdino_query_if_configured`, but takes explicit
    ``seg_pipe`` (``SegmentationPipeline``) and ``rampp`` (``RAMPlusPlusWrapper``)
    instead of monolith ``pipeline.sam2_wrapper`` / ``pipeline.rampp``.

    ``query_builder_mode``:
    - ``static``: no refresh
    - ``rampp_full`` / ``rampp_region_crops``: require active ``rampp``
    - ``inherit``: refresh when ``rampp`` is active (staged default when RAM++ loads)
    """
    mode = str(getattr(cfg, "query_builder_mode", "inherit") or "inherit").strip().lower()
    default_q = str(getattr(seg_pipe, "text_query", "") or "")

    if mode == "static":
        return [], default_q

    rampp_on = bool(getattr(cfg, "rampp_enabled", True))
    force_rampp = mode in ("rampp_full", "rampp_region_crops")

    if mode == "inherit":
        if rampp is None or not getattr(rampp, "active", False):
            return [], default_q
    else:
        if not force_rampp and not rampp_on:
            return [], default_q

    if rampp is None or not getattr(rampp, "active", False):
        return [], default_q

    max_tags = int(getattr(cfg, "rampp_max_tags", 8))
    use_region_crops = bool(getattr(cfg, "regions_rampp_crops_enabled", False)) and bool(region_partition_meta)
    if mode == "rampp_region_crops":
        use_region_crops = bool(region_partition_meta)
    elif mode == "rampp_full":
        use_region_crops = False
    elif mode == "inherit":
        use_region_crops = use_region_crops and bool(region_partition_meta)
    else:
        use_region_crops = use_region_crops and bool(region_partition_meta)

    tags_meta: List[str] = []
    w, h = width, height

    if use_region_crops:
        import cv2 as _cv2

        crops_bgr: List[np.ndarray] = []
        for reg in region_partition_meta:
            bx = reg.get("bbox_px") or [0, 0, w - 1, h - 1]
            if len(bx) < 4:
                continue
            x1, y1, x2, y2 = [int(max(0, v)) for v in bx[:4]]
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_rgb = img_rgb[y1 : y2 + 1, x1 : x2 + 1]
            if crop_rgb.size == 0:
                continue
            crops_bgr.append(_cv2.cvtColor(crop_rgb, _cv2.COLOR_RGB2BGR))
        union_tags: List[str] = []
        if crops_bgr:
            try:
                batched = rampp.label_crops(crops_bgr)
            except Exception as e:
                print(f"  [RAM++] region crop batched tag failed: {e}; per-crop fallback.")
                batched = None
            if batched is None:
                for _bgr in crops_bgr:
                    try:
                        tr = rampp.label_crop(_bgr)
                        union_tags.extend(list(tr.get("tags", [])))
                    except Exception as e:
                        print(f"  [RAM++] region crop tag failed: {e}")
            else:
                for tr in batched:
                    union_tags.extend(list(tr.get("tags", [])))
        tags_meta = list(dict.fromkeys(union_tags))[: max(max_tags * 2, 16)]
        if tags_meta:
            print(f"  [RAM++] Region-crop tags ({len(tags_meta)})")

    if not tags_meta:
        tag_result = rampp.tag_image(img_rgb)
        tags_meta = list(tag_result.get("tags", []))
        if tags_meta:
            print(f"  [RAM++] Full-image tags: {', '.join(tags_meta)}")

    if tags_meta:
        dynamic_query = ". ".join(tags_meta) + "."
        seg_pipe.update_text_query(dynamic_query)
        print(f"  [RAM++] GDINO query updated ({len(tags_meta)} tags)")
        return tags_meta, dynamic_query
    print("  [RAM++] No tags returned — using default GDINO query")
    return [], default_q
