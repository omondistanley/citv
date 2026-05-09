"""Track-local region artifact relative paths (parallel metadata keys)."""

from __future__ import annotations

from typing import Dict


def region_track_relpaths(track_dir_name: str, stem: str) -> Dict[str, str]:
    """Relative POSIX paths under ``scene_graph/{track}/`` matching ``regions_block`` / sam2_metadata."""
    base = f"scene_graph/{track_dir_name}/{stem}"
    return {
        "regions_json": f"{base}_regions.json",
        "regions_image": f"{base}_regions.png",
        "regions_overlay_image": f"{base}_regions_overlay.png",
        "region_segmentation_image": f"{base}_region_segmentation.png",
        "region_sam2_segmentation_image": f"{base}_region_sam2_style_segmentation.png",
        "region_tinted_overlay_image": f"{base}_region_tinted_overlay.png",
    }
