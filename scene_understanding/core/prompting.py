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
    if mode == "inherit" and not rampp_on:
        return [], _default_query_from_wrapper(pipeline.sam2_wrapper)
    if not force_rampp and not rampp_on:
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

    from scene_understanding.labeling import RAMPlusPlusWrapper

    if getattr(pipeline, "rampp", None) is None or not getattr(pipeline.rampp, "active", False):
        pipeline.rampp = RAMPlusPlusWrapper(
            device=pipeline.device,
            checkpoint_path=getattr(pipeline, "_rampp_checkpoint_path", None),
            repo_path=getattr(pipeline, "_rampp_repo_path", None),
            image_size=int(getattr(pipeline, "_rampp_image_size", 384)),
            vit=str(getattr(pipeline, "_rampp_vit", "swin_l")),
            default_confidence=float(getattr(pipeline, "_rampp_default_conf", 0.7)),
            max_tags=int(getattr(pipeline, "_rampp_max_tags", 8)),
        )

    rampp = getattr(pipeline, "rampp", None)
    if rampp is None or not getattr(rampp, "active", False):
        return [], query_used

    max_tags = int(getattr(pipeline, "_rampp_max_tags", 8))
    w, h = width, height

    if use_region_crops:
        union_tags: List[str] = []
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
            try:
                tr = rampp.tag_image(crop_rgb)
                union_tags.extend(list(tr.get("tags", [])))
            except Exception as e:
                print(f"  [RAM++] region crop tag failed: {e}")
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
