"""First real, end-to-end run of the staged pipeline chain against actual
models (SAM2, Depth-Anything-V2, GroundingDINO, Florence-2, RAM++) -- every
Phase 1-3 stage this session built/rewired (paths_export, hierarchy_export,
depth_mask_fusion, captions_export, relations) was verified all session
against synthetic fixtures and fake pipeline shims, but never against real
model output until now. Run this before touching the Gradio UI, so any
real integration bug shows up here with a clean traceback, not buried under
Gradio's own error handling.

Usage:
    python scripts/smoke_test_pipeline.py [image_path]

Defaults to images/IMG-6392.png if no path is given.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The upstream sam2 repo is cloned into REPO_ROOT/sam2/ with its real
# installed package one level deeper, at REPO_ROOT/sam2/sam2/. Once
# REPO_ROOT goes on sys.path (next line, needed for `config`/`depth`/
# `scene_understanding`), Python's default resolution treats the outer
# REPO_ROOT/sam2 (no __init__.py of its own) as an implicit namespace
# package matching the name "sam2", shadowing the real editable-installed
# package -- sam2's own build_sam.py detects exactly this shadow and
# aborts. Load the real package by its exact file path and cache it in
# sys.modules *before* REPO_ROOT touches sys.path, so every later
# `import sam2` anywhere in the pipeline reuses this correct module
# instead of re-resolving (and re-breaking) it.
_sam2_real_init = REPO_ROOT / "sam2" / "sam2" / "__init__.py"
if "sam2" not in sys.modules and _sam2_real_init.exists():
    _spec = importlib.util.spec_from_file_location(
        "sam2", _sam2_real_init, submodule_search_locations=[str(_sam2_real_init.parent)]
    )
    _sam2_module = importlib.util.module_from_spec(_spec)
    sys.modules["sam2"] = _sam2_module
    _spec.loader.exec_module(_sam2_module)

sys.path.insert(0, str(REPO_ROOT))

from config import PreprocessConfig
from depth import DepthEstimator
from scene_understanding.animation.compositor import render_animation
from scene_understanding.animation.motion_contract import MotionContract, ground_motion_contract
from scene_understanding.pipeline import SceneUnderstandingPipeline


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "images" / "IMG-6392.png")
    if not Path(image_path).exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    output_dir = REPO_ROOT / "output_scene_understanding" / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading models (first run downloads/caches are already done by setup.sh)...")
    t0 = time.time()
    cfg = PreprocessConfig()
    depth_estimator = DepthEstimator(cfg, first_image=image_path)
    pipeline = SceneUnderstandingPipeline(depth_estimator, config=cfg)
    print(f"  models loaded in {time.time() - t0:.1f}s")

    print(f"\nRunning _run_stage_chain on {image_path} ...")
    t0 = time.time()
    ctx = pipeline._run_stage_chain(image_path, str(output_dir))
    elapsed = time.time() - t0
    print(f"  stage chain finished in {elapsed:.1f}s")

    objects = ctx.extra.get("objects", [])
    print(f"\n=== Results ===")
    print(f"objects detected:        {len(objects)}")
    print(f"relations:                {len(ctx.relations)}")
    print(f"mask_hierarchy edges:     {len(ctx.mask_hierarchy.get('edges', [])) if ctx.mask_hierarchy else 'N/A (stage did not run)'}")

    hyp_path = ctx.path_exports.get("path_hypotheses_json")
    if hyp_path and Path(hyp_path).exists():
        hypotheses = json.loads(Path(hyp_path).read_text())["hypotheses"]
        print(f"path hypotheses:          {len(hypotheses)}")
        if hypotheses:
            sample = hypotheses[0]
            print(f"  sample path_id:         {sample['path_id']}")
            print(f"  sample polyline length: {len(sample['polyline_2d'])} points (>2 means non-degenerate)")
            print(f"  sample kinematic tags:  {[s['motion'] for s in sample['kinematic_signatures']]}")
    else:
        print("path hypotheses:          NONE WRITTEN -- paths_export stage did not produce output, this needs debugging")

    caption_bundle = ctx.path_exports.get("caption_bundle_json")
    print(f"caption bundle:           {'written -> ' + caption_bundle if caption_bundle else 'NOT WRITTEN'}")

    print(f"\n=== Phase 3 custom-input flow (drawn path + free-text actor/action) ===")
    print("Reuses this same real ctx (real masks/depth/relations) instead of a second model load --")
    print("exercises exactly what a user does in the 'Custom inputs' section of app.py.")
    h, w = ctx.img_bgr.shape[:2]
    # A synthetic hand-drawn diagonal path, standing in for mouse clicks
    # (+ the new undo/backtrack button) on the Gradio canvas -- same field
    # (`drawn_polyline_2d`) either way. Paired with a real detected object as
    # the actor (rather than open-vocabulary actor_text with no mask/asset
    # to cut a sprite from) since that combination is Tier-2-renderable by
    # design -- see `_resolve_actor`'s documented mask-less fallback.
    drawn_points = [(w * 0.15, h * 0.85), (w * 0.5, h * 0.55), (w * 0.85, h * 0.25)]
    objects_for_actor = ctx.extra.get("objects", [])
    actor_object_id = objects_for_actor[0]["id"] if objects_for_actor else None
    contract = MotionContract(
        actor_object_id=actor_object_id,
        drawn_polyline_2d=drawn_points,
        action_text="walks along the path",  # the free-text ("words") input
        duration_s=3.0, fps=12,
    )
    try:
        grounded = ground_motion_contract(contract, ctx)
    except Exception as exc:
        print(f"  FAILED: ground_motion_contract raised: {exc}")
        grounded = None

    if grounded is None:
        print("  FAILED: ground_motion_contract returned None for a drawn path + free text -- this needs debugging.")
    else:
        print(f"  grounded ok. archetype={grounded.archetype}, tier2_available={grounded.tier2_available}")
        print(f"  report: {grounded.report}")
        objects = ctx.extra.get("objects", [])
        object_masks_by_id = {o.get("id"): o.get("_sam2_mask_array") for o in objects if o.get("_sam2_mask_array") is not None}
        custom_gif = str(output_dir / "phase3_smoke_test.gif")
        custom_mp4 = str(output_dir / "phase3_smoke_test.mp4")
        try:
            result = render_animation(
                ctx.img_bgr, grounded.hypothesis, grounded.actor_mask, object_masks_by_id,
                out_gif_path=custom_gif, out_mp4_path=custom_mp4, fps=12, duration_s=3.0,
                actor_sprite_rgba=grounded.actor_asset_rgba,
            )
            print(f"  Tier 2 rendered {result['frame_count']} frames -> {custom_gif}")
        except Exception as exc:
            print(f"  FAILED: Tier 2 render_animation raised: {exc}")

    print(f"\nFull output tree: {output_dir}")
    print("If everything above looks non-empty and non-degenerate, the real pipeline integration is solid --")
    print("proceed to `python app.py` and test the Gradio UI golden path next.")


if __name__ == "__main__":
    main()
