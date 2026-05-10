"""
Full per-image scene graph run (legacy-equivalent artifacts).

``staged`` pipeline mode uses this module so outputs match
``SceneUnderstandingPipeline.process_image`` in ``scene_understanding.py``
(``scene_graph/grounded_sam2/``, depth-mask JSON, paths, captions, etc.).

The implementation **delegates** to ``LegacySceneUnderstandingPipeline.process_image``
until the monolith body is mechanically moved here; callers import this module
instead of reaching into the script file, which is the strangling seam for
eventually deleting the thin class shell in ``scene_understanding.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _preferred_scene_json_paths(output_dir: Path, stem: str) -> List[Path]:
    """Prefer grounded_sam2 track layout, then any ``{stem}_scene.json`` under scene_graph/."""
    sg = output_dir / "scene_graph"
    ordered: List[Path] = []
    for track in ("grounded_sam2", "amg", "combined"):
        p = sg / track / f"{stem}_scene.json"
        if p.is_file():
            ordered.append(p)
    if not ordered and sg.is_dir():
        ordered = sorted(sg.rglob(f"{stem}_scene.json"))
    return ordered


def run_legacy_equivalent_process_image(pipeline: Any, image_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Run the full legacy pipeline body on *pipeline* (same files as monolithic ``process_image``).

    Returns a small dict including ``scene_json`` (primary track scene file) for API parity
    with the older slim staged exporter.
    """
    from scene_understanding.pipeline import LegacySceneUnderstandingPipeline

    LegacySceneUnderstandingPipeline.process_image(pipeline, image_path, output_dir)

    stem = Path(image_path).stem
    out = Path(output_dir)
    paths = _preferred_scene_json_paths(out, stem)
    primary: Optional[Path] = paths[0] if paths else None
    return {
        "runner": "legacy_equivalent",
        "output_dir": str(out.resolve()),
        "scene_json": str(primary.resolve()) if primary else "",
        "scene_json_paths": [str(p.resolve()) for p in paths],
    }


def use_modular_chain_only() -> bool:
    """When True, ``process_image_staged`` uses the slim PR2 chain + ``scene_graph/staged/`` JSON only."""
    v = str(os.getenv("CITV_STAGED_MODULAR_CHAIN_ONLY", "")).strip().lower()
    return v in ("1", "true", "yes", "on")
