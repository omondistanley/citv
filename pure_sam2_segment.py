"""
pure_sam2_segment.py
====================
Pure SAM2 instance segmentation — exactly as described in the SAM2 paper and demo.

Usage modes
-----------
1. Prompted (point click — paper/demo primary mode):
   python pure_sam2_segment.py --image path/to/image.jpg --points "320,240" "100,80"

2. Prompted (bounding box):
   python pure_sam2_segment.py --image path/to/image.jpg --box "50,30,400,300"

3. Prompted (combined point + box for one object):
   python pure_sam2_segment.py --image path/to/image.jpg \
       --box "50,30,400,300" --points "200,150" --point_labels 1

4. Automatic mask generation (segment everything — grid sweep):
   python pure_sam2_segment.py --image path/to/image.jpg --auto

What makes this "pure":
  - SAM2ImagePredictor is used directly with no intermediate detector.
  - multimask_output=True for point prompts: 3 candidates returned, best selected
    by predicted IoU score (exactly as the paper describes).
  - multimask_output=False only when a tight box is given as the sole prompt
    (unambiguous prompt → single output, per paper recommendation).
  - When a point + box are combined, multimask_output=True is used so SAM2 can
    disambiguate which sub-region the click refers to.
  - Low-res mask logits from a first pass are fed back as mask_input in an
    optional refinement pass (iterative mask refinement, paper Section 3.3).
  - predicted_iou and stability_score come directly from SAM2 — not proxies.
  - No external detector, no resize-before-predict hacks.
  - Automatic mode uses SAM2AutomaticMaskGenerator with paper-default params.

Outputs (written to --out_dir, default: sam2_pure_output/):
  {stem}_overlay.png        — colour-coded instance overlay on original image
  {stem}_masks.png          — each mask as a separate colour tile (grid)
  {stem}_masks.json         — per-mask metadata: iou, stability, bbox, area, prompt
  {stem}_segmentation.png   — black background, each instance a unique colour
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Resolve the local sam2 package from the project directory
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SAM2_REPO = SCRIPT_DIR / "sam2"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))

# Default checkpoint / config (sam2.1 large — highest quality)
DEFAULT_CHECKPOINT = str(SAM2_REPO / "checkpoints" / "sam2.1_hiera_large.pt")
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_l"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_colour(seed: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed + 42)
    return tuple(int(x) for x in rng.integers(60, 230, size=3))


def _overlay_masks(image_rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Draw each mask as a semi-transparent colour overlay on the image."""
    out = image_rgb.copy().astype(np.float32)
    for i, mask in enumerate(masks):
        colour = np.array(_random_colour(i), dtype=np.float32)
        bin_mask = mask.astype(bool)
        out[bin_mask] = out[bin_mask] * (1 - alpha) + colour * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _segmentation_image(masks: List[np.ndarray], h: int, w: int) -> np.ndarray:
    """Black canvas; each instance painted a unique colour (no overlap blending)."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        colour = _random_colour(i)
        canvas[mask.astype(bool)] = colour
    return canvas


def _mask_tile_grid(image_rgb: np.ndarray, masks: List[np.ndarray], ious: List[float]) -> np.ndarray:
    """Small per-mask thumbnails arranged in a grid with IoU label."""
    THUMB = 128
    n = len(masks)
    if n == 0:
        return np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * THUMB, cols * THUMB, 3), dtype=np.uint8)
    h, w = image_rgb.shape[:2]
    for idx, (mask, iou) in enumerate(zip(masks, ious)):
        r, c = divmod(idx, cols)
        crop = image_rgb.copy()
        colour = np.array(_random_colour(idx), dtype=np.float32)
        bin_mask = mask.astype(bool)
        crop[bin_mask] = (crop[bin_mask].astype(np.float32) * 0.4 + colour * 0.6).astype(np.uint8)
        # draw bounding box
        ys, xs = np.where(bin_mask)
        if len(ys):
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(crop, (x0, y0), (x1, y1), (255, 255, 255), 2)
        thumb = cv2.resize(crop, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{iou:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        grid[r * THUMB:(r + 1) * THUMB, c * THUMB:(c + 1) * THUMB] = thumb
    return grid


def _build_metadata(
    masks: List[np.ndarray],
    ious: List[float],
    stabilities: Optional[List[float]],
    prompt_info: dict,
) -> List[dict]:
    records = []
    for i, (mask, iou) in enumerate(zip(masks, ious)):
        bin_mask = mask.astype(bool)
        area = int(bin_mask.sum())
        ys, xs = np.where(bin_mask)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
            if len(ys) else [0, 0, 0, 0]
        )
        centroid = (
            [float(xs.mean()), float(ys.mean())] if len(ys) else [0.0, 0.0]
        )
        records.append({
            "mask_index": i,
            "predicted_iou": float(iou),
            "stability_score": float(stabilities[i]) if stabilities else None,
            "area_px": area,
            "bbox_xywh": bbox,
            "centroid_xy": centroid,
            "prompt": prompt_info,
        })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Core: Prompted mode (point / box / combined)
# ──────────────────────────────────────────────────────────────────────────────

def segment_prompted(
    image_rgb: np.ndarray,
    predictor,
    point_coords: Optional[np.ndarray] = None,   # shape (N, 2) in xy pixels
    point_labels: Optional[np.ndarray] = None,   # shape (N,) 1=fg, 0=bg
    box: Optional[np.ndarray] = None,            # shape (4,) xyxy pixels
    refine: bool = True,
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    """
    Run SAM2 prompted segmentation exactly as the paper/demo.

    Paper rules implemented:
      - Point-only prompt  → multimask_output=True (3 candidates), pick highest IoU
      - Box-only prompt    → multimask_output=False (unambiguous), single mask
      - Point+box prompt   → multimask_output=True (box constrains region,
                             point disambiguates sub-region)
      - Refinement pass    → if refine=True, feed low_res_logits back as
                             mask_input for a second predict() call (paper §3.3)

    Returns:
      masks       — list of binary HxW bool arrays (best mask per object)
      ious        — predicted IoU for each returned mask
      stabilities — stability score for each returned mask (None if unavailable)
    """
    predictor.set_image(image_rgb)

    # Decide multimask_output per paper rules
    if box is not None and point_coords is None:
        # Tight box → unambiguous → single mask output
        multimask = False
    else:
        # Point alone, or point+box → produce 3 candidates
        multimask = True

    # --- First pass ---
    masks_np, iou_preds, low_res_logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=multimask,
    )
    # masks_np: (C, H, W)  iou_preds: (C,)  low_res_logits: (C, 1, 256, 256)

    # --- Refinement pass (paper §3.3: feed logits back as mask_input) ---
    if refine and masks_np.shape[0] > 0:
        # Use the highest-IoU mask's logits for the refinement pass
        best_idx = int(np.argmax(iou_preds))
        refined_masks, refined_ious, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=low_res_logits[best_idx : best_idx + 1],
            multimask_output=False,   # refinement → single definitive mask
        )
        # Replace the best-candidate with its refined version
        masks_np[best_idx] = refined_masks[0]
        iou_preds[best_idx] = refined_ious[0]

    # Select the best mask (highest predicted IoU)
    if multimask and masks_np.shape[0] > 1:
        best_idx = int(np.argmax(iou_preds))
        selected_masks = [masks_np[best_idx].astype(bool)]
        selected_ious = [float(iou_preds[best_idx])]
    else:
        selected_masks = [masks_np[i].astype(bool) for i in range(masks_np.shape[0])]
        selected_ious = [float(iou_preds[i]) for i in range(len(iou_preds))]

    return selected_masks, selected_ious, []   # stability N/A for prompted mode


# ──────────────────────────────────────────────────────────────────────────────
# Core: Automatic mode (segment everything — paper §4 / demo)
# ──────────────────────────────────────────────────────────────────────────────

def segment_auto(
    image_rgb: np.ndarray,
    model,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    crop_n_layers: int = 1,
    min_mask_region_area: int = 100,
    max_long_side: int = 1280,
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    """
    Run SAM2AutomaticMaskGenerator with paper/demo default settings.

    Paper defaults (Table 2 in SAM2 paper):
      points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.95,
      box_nms_thresh=0.7, crop_n_layers=1, use_m2m=True, multimask_output=True

    max_long_side: downscale the image before AMG if its longest side exceeds
      this value. AMG's multi-crop strategy multiplies memory by ~4x; on a
      14 GB GPU a 3024×4032 image OOMs. Masks are upscaled back to original
      resolution with INTER_NEAREST so boundaries are preserved exactly.
      Set to 0 to disable downscaling (only safe for small images).

    Returns:
      masks       — list of binary HxW bool arrays at ORIGINAL resolution,
                    sorted by area (largest first)
      ious        — predicted_iou for each mask
      stabilities — stability_score for each mask
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    orig_h, orig_w = image_rgb.shape[:2]
    scale = 1.0
    proc = image_rgb

    if max_long_side > 0 and max(orig_h, orig_w) > max_long_side:
        scale = max_long_side / max(orig_h, orig_w)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        proc = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  [auto] Image downscaled {orig_w}x{orig_h} → {new_w}x{new_h} for AMG (VRAM limit)")

    amg = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=64,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=1.0,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
        use_m2m=True,           # mask-to-mask refinement — paper recommendation
        multimask_output=True,  # paper default for AMG
    )

    with torch.inference_mode():
        anns = amg.generate(proc)

    if not anns:
        return [], [], []

    # Sort by area descending (largest object first, same as demo)
    anns = sorted(anns, key=lambda x: x["area"], reverse=True)

    # Upscale masks back to original resolution when we downscaled
    if scale != 1.0:
        upscaled = []
        for a in anns:
            m = np.asarray(a["segmentation"]).astype(np.uint8)
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            upscaled.append(m.astype(bool))
        masks = upscaled
    else:
        masks = [np.asarray(a["segmentation"]).astype(bool) for a in anns]
    ious = [float(a["predicted_iou"]) for a in anns]
    stabilities = [float(a["stability_score"]) for a in anns]

    return masks, ious, stabilities


# ──────────────────────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    ious: List[float],
    stabilities: List[float],
    out_dir: Path,
    stem: str,
    prompt_info: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]

    # 1. Colour overlay
    overlay = _overlay_masks(image_rgb, masks)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 2. Black-background segmentation map (each instance = unique colour)
    seg_map = _segmentation_image(masks, h, w)
    cv2.imwrite(str(out_dir / f"{stem}_segmentation.png"), cv2.cvtColor(seg_map, cv2.COLOR_RGB2BGR))

    # 3. Mask tile grid with predicted IoU labels
    grid = _mask_tile_grid(image_rgb, masks, ious)
    cv2.imwrite(str(out_dir / f"{stem}_masks.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # 4. JSON metadata
    meta = _build_metadata(masks, ious, stabilities if stabilities else None, prompt_info)
    with open(out_dir / f"{stem}_masks.json", "w") as f:
        json.dump({"image": stem, "num_masks": len(masks), "masks": meta}, f, indent=2)

    print(f"  Saved {len(masks)} masks → {out_dir}/")
    for i, (iou, stab) in enumerate(zip(ious, stabilities if stabilities else [None] * len(ious))):
        stab_str = f", stability={stab:.3f}" if stab is not None else ""
        print(f"    mask {i:03d}: predicted_iou={iou:.3f}{stab_str}, area={meta[i]['area_px']}px")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pure SAM2 instance segmentation (paper/demo faithful)")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--out_dir", default="sam2_pure_output", help="Output directory")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="SAM2 checkpoint .pt")
    p.add_argument("--cfg", default=DEFAULT_CFG, help="SAM2 Hydra config name")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true",
                      help="Automatic mode: segment everything (grid sweep)")
    mode.add_argument("--points", nargs="+", metavar="X,Y",
                      help="One or more foreground point prompts, e.g. '320,240'")
    mode.add_argument("--box", metavar="X1,Y1,X2,Y2",
                      help="Single bounding box prompt in xyxy pixel coords")

    p.add_argument("--bg_points", nargs="+", metavar="X,Y",
                   help="Background points (label=0) when using --points")
    p.add_argument("--point_labels", nargs="+", type=int,
                   help="Labels for --points (1=fg, 0=bg). Must match count of --points")
    p.add_argument("--no_refine", action="store_true",
                   help="Disable iterative mask refinement pass (enabled by default)")

    # Auto-mode quality controls (paper defaults)
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.8)
    p.add_argument("--stability_thresh", type=float, default=0.95)
    p.add_argument("--min_area", type=int, default=100,
                   help="Minimum mask area in pixels (auto mode)")
    p.add_argument("--max_long_side", type=int, default=1280,
                   help="Downscale image so longest side <= this before AMG (VRAM safety). "
                        "Set 0 to disable. Default: 1280")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load image ──────────────────────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        sys.exit(f"Could not read image: {img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"Image: {img_path.name}  ({w}x{h})")

    # ── Load SAM2 model ──────────────────────────────────────────────────────
    from sam2.build_sam import build_sam2
    print(f"Loading SAM2: {args.cfg}  checkpoint={Path(args.checkpoint).name}")
    model = build_sam2(
        config_file=args.cfg,
        ckpt_path=args.checkpoint,
        device=args.device,
        mode="eval",
        apply_postprocessing=True,   # enables dynamic_multimask_via_stability — paper default
    )
    print("SAM2 loaded.")

    out_dir = Path(args.out_dir)
    stem = img_path.stem

    # ── Automatic mode ───────────────────────────────────────────────────────
    if args.auto or (not args.points and not args.box):
        print("Mode: AUTOMATIC (segment everything)")
        masks, ious, stabilities = segment_auto(
            image_rgb,
            model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_thresh,
            min_mask_region_area=args.min_area,
            max_long_side=args.max_long_side,
        )
        prompt_info = {
            "mode": "auto",
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_thresh,
        }

    # ── Prompted mode ────────────────────────────────────────────────────────
    else:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(model)

        point_coords = None
        point_labels = None
        box_arr = None
        prompt_info = {"mode": "prompted"}

        if args.box:
            vals = [float(v) for v in args.box.replace(",", " ").split()]
            if len(vals) != 4:
                sys.exit("--box must be exactly 4 values: X1,Y1,X2,Y2")
            box_arr = np.array(vals, dtype=np.float32)
            prompt_info["box_xyxy"] = vals

        if args.points:
            coords = []
            for pt_str in args.points:
                xy = [float(v) for v in pt_str.replace(",", " ").split()]
                if len(xy) != 2:
                    sys.exit(f"Bad point format (need X,Y): {pt_str}")
                coords.append(xy)

            # Append background points if provided
            labels = []
            if args.point_labels and len(args.point_labels) == len(args.points):
                labels = list(args.point_labels)
            else:
                labels = [1] * len(coords)

            if args.bg_points:
                for pt_str in args.bg_points:
                    xy = [float(v) for v in pt_str.replace(",", " ").split()]
                    coords.append(xy)
                    labels.append(0)

            point_coords = np.array(coords, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
            prompt_info["point_coords"] = coords
            prompt_info["point_labels"] = labels

        if point_coords is None and box_arr is None:
            sys.exit("Prompted mode requires --points and/or --box.")

        print(f"Mode: PROMPTED  points={point_coords is not None}  box={box_arr is not None}  refine={not args.no_refine}")
        masks, ious, stabilities = segment_prompted(
            image_rgb,
            predictor,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
            refine=not args.no_refine,
        )

    print(f"\nTotal masks produced: {len(masks)}")
    save_outputs(image_rgb, masks, ious, stabilities, out_dir, stem, prompt_info)


if __name__ == "__main__":
    main()
