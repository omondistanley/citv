#!/usr/bin/env python3
"""
eval_detectors.py
-----------------
Compares the four object detectors (GRiT, FasterRCNN, DETR, Pix2Seq / OWL-ViT)
across all images that have already been processed by scene_understanding.py.

Usage:
    python eval_detectors.py                          # default output dir
    python eval_detectors.py --output_dir output_scene

What it does (no model inference — reads existing JSON files):
  - Counts detections per model per image
  - Reports confidence score distributions (mean, median, min, max)
  - Reports unique label diversity per model
  - Computes overlap between models (how many labels do two models both detect?)
  - Computes per-model SAM2 match rate from scene JSON (mask_matched flag)
  - Computes per-model 3D depth stats quality (depth_stats.std — lower = cleaner mask)
  - Prints a ranked summary table
  - Optionally saves results to eval_detectors.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = ["grit", "faster_rcnn", "detr", "pix2seq"]
# source_model string as written by each wrapper
SOURCE_MODEL_MAP = {
    "grit": "GRiT",
    "faster_rcnn": "FasterRCNN",
    "detr": "DETR",
    "pix2seq": "Pix2Seq",
}
# OWL-ViT is the fallback inside Pix2SeqWrapper
OWL_VIT_SOURCE = "OWL-ViT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(lst: List[float]) -> Optional[float]:
    return round(sum(lst) / len(lst), 4) if lst else None


def _safe_median(lst: List[float]) -> Optional[float]:
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    return round((s[mid] + s[mid - 1]) / 2 if n % 2 == 0 else s[mid], 4)


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100 * num / denom:.1f}%"


def _bar(value: float, max_val: float, width: int = 20) -> str:
    if max_val == 0:
        return " " * width
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_per_model_jsons(output_dir: Path) -> Dict[str, List[List[Dict]]]:
    """
    Returns {model_name: [list_of_detections_per_image, ...]}
    Each element is the detection list for one image.
    """
    per_model: Dict[str, List[List[Dict]]] = defaultdict(list)
    stems_found: set = set()

    # Walk the output dir looking for {stem}_{model}.json under {model}/ subdirs
    for model in MODELS:
        model_dir = output_dir / model
        if not model_dir.exists():
            continue
        for jf in sorted(model_dir.glob(f"*_{model}.json")):
            try:
                with open(jf) as f:
                    dets = json.load(f)
                per_model[model].append(dets)
                stem = jf.name[: -(len(model) + 6)]  # strip _{model}.json
                stems_found.add(stem)
            except Exception as e:
                print(f"  Warning: could not read {jf}: {e}")

    return per_model, sorted(stems_found)


def collect_scene_jsons(output_dir: Path) -> Dict[str, List[Dict]]:
    """
    Returns {source_model_str: [obj_entry, ...]} across all scene JSONs.
    Reads objects_3d from {stem}_scene.json to get mask_matched and depth_stats.
    """
    scene_dir = output_dir / "scene_graph"
    per_source: Dict[str, List[Dict]] = defaultdict(list)

    if not scene_dir.exists():
        return per_source

    for jf in sorted(scene_dir.glob("*_scene.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
            objects = data.get("objects", [])
            for obj in objects:
                sources = obj.get("sources", {})
                # The first key in 'sources' is the detector that produced this object
                src = next(iter(sources), None)
                if src:
                    per_source[src].append(obj)
        except Exception as e:
            print(f"  Warning: could not read {jf}: {e}")

    return per_source


# ---------------------------------------------------------------------------
# Per-model statistics
# ---------------------------------------------------------------------------

def compute_model_stats(
    model: str,
    images_dets: List[List[Dict]],
    scene_objs: List[Dict],
) -> Dict[str, Any]:
    """Compute stats for one model across all processed images."""
    all_confs: List[float] = []
    all_labels: List[str] = []
    det_counts: List[int] = []

    for img_dets in images_dets:
        det_counts.append(len(img_dets))
        for d in img_dets:
            c = d.get("conf")
            if c is not None:
                all_confs.append(float(c))
            lbl = d.get("label", "")
            if lbl:
                all_labels.append(lbl.lower().strip())

    n_images = len(images_dets)
    n_total = sum(det_counts)
    unique_labels = sorted(set(all_labels))

    # SAM2 match rate from scene objects
    matched = sum(1 for o in scene_objs if o.get("mask_matched", False))
    unmatched = len(scene_objs) - matched

    # Depth quality: std of depth over mask (lower = cleaner)
    depth_stds: List[float] = []
    for o in scene_objs:
        ds = o.get("depth_stats")
        if ds and ds.get("std") is not None:
            depth_stds.append(float(ds["std"]))

    return {
        "model": model,
        "n_images": n_images,
        "n_total_detections": n_total,
        "avg_per_image": round(n_total / n_images, 1) if n_images else 0,
        "min_per_image": min(det_counts) if det_counts else 0,
        "max_per_image": max(det_counts) if det_counts else 0,
        "conf_mean": _safe_mean(all_confs),
        "conf_median": _safe_median(all_confs),
        "conf_min": round(min(all_confs), 4) if all_confs else None,
        "conf_max": round(max(all_confs), 4) if all_confs else None,
        "n_unique_labels": len(unique_labels),
        "top_labels": unique_labels[:15],
        "n_scene_objects": len(scene_objs),
        "mask_matched": matched,
        "mask_unmatched": unmatched,
        "mask_match_rate": matched / len(scene_objs) if scene_objs else None,
        "depth_std_mean": _safe_mean(depth_stds),   # lower is better
        "depth_std_median": _safe_median(depth_stds),
    }


def compute_overlap(stats_list: List[Dict]) -> None:
    """Print pairwise label overlap between models."""
    label_sets: Dict[str, set] = {}
    for s in stats_list:
        model = s["model"]
        # Rebuild label set from top_labels (partial, but indicative)
        label_sets[model] = set(s["top_labels"])

    models = [s["model"] for s in stats_list]
    print("\n  Pairwise label overlap (# shared labels in top-15 per model):")
    print(f"  {'':>12}", end="")
    for m in models:
        print(f"  {m:>12}", end="")
    print()
    for m1 in models:
        print(f"  {m1:>12}", end="")
        for m2 in models:
            if m1 == m2:
                n = len(label_sets[m1])
                print(f"  {'(self)':>12}", end="")
            else:
                shared = len(label_sets[m1] & label_sets[m2])
                print(f"  {shared:>12}", end="")
        print()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SEP = "─" * 90


def render_summary(stats_list: List[Dict]) -> None:
    print(f"\n{'═' * 90}")
    print("  DETECTOR COMPARISON REPORT")
    print(f"{'═' * 90}\n")

    # --- Detection volume ---
    print("  1. DETECTION VOLUME (per image)")
    print(f"  {SEP}")
    max_avg = max((s["avg_per_image"] for s in stats_list), default=1)
    print(f"  {'Model':<14} {'Images':>7} {'Total':>7} {'Avg/img':>8} {'Min':>5} {'Max':>5}  Bar")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*8} {'-'*5} {'-'*5}  {'─'*20}")
    for s in stats_list:
        bar = _bar(s["avg_per_image"], max_avg)
        print(
            f"  {s['model']:<14} {s['n_images']:>7} {s['n_total_detections']:>7} "
            f"{s['avg_per_image']:>8.1f} {s['min_per_image']:>5} {s['max_per_image']:>5}  {bar}"
        )

    # --- Confidence scores ---
    print(f"\n  2. CONFIDENCE SCORES (model's own threshold applies)")
    print(f"  {SEP}")
    print(f"  {'Model':<14} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for s in stats_list:
        def _fmt(v):
            return f"{v:.4f}" if v is not None else "  n/a "
        print(
            f"  {s['model']:<14} {_fmt(s['conf_mean']):>8} {_fmt(s['conf_median']):>8} "
            f"{_fmt(s['conf_min']):>8} {_fmt(s['conf_max']):>8}"
        )

    # --- Label diversity ---
    print(f"\n  3. LABEL DIVERSITY (unique class labels seen)")
    print(f"  {SEP}")
    max_ul = max((s["n_unique_labels"] for s in stats_list), default=1)
    print(f"  {'Model':<14} {'Unique labels':>14}  Bar")
    print(f"  {'-'*14} {'-'*14}  {'─'*20}")
    for s in stats_list:
        bar = _bar(s["n_unique_labels"], max_ul)
        print(f"  {s['model']:<14} {s['n_unique_labels']:>14}  {bar}")

    print()
    for s in stats_list:
        labels_str = ", ".join(s["top_labels"][:10]) or "(none)"
        print(f"    {s['model']}: {labels_str}")

    # --- SAM2 match rate ---
    print(f"\n  4. SAM2 MASK MATCH RATE (higher = more objects get mask-native 3D coords)")
    print(f"  {SEP}")
    print(f"  {'Model':<14} {'Scene objs':>11} {'Matched':>8} {'Unmatched':>10} {'Rate':>8}  Bar")
    print(f"  {'-'*14} {'-'*11} {'-'*8} {'-'*10} {'-'*8}  {'─'*20}")
    for s in stats_list:
        rate = s["mask_match_rate"]
        bar = _bar(rate if rate is not None else 0, 1.0)
        rate_str = f"{rate:.1%}" if rate is not None else "  n/a"
        print(
            f"  {s['model']:<14} {s['n_scene_objects']:>11} {s['mask_matched']:>8} "
            f"{s['mask_unmatched']:>10} {rate_str:>8}  {bar}"
        )

    # --- Depth quality ---
    print(f"\n  5. DEPTH QUALITY — std of depth over mask (lower = cleaner mask, less background bleed)")
    print(f"  {SEP}")
    max_std = max(
        (s["depth_std_mean"] for s in stats_list if s["depth_std_mean"] is not None),
        default=1.0,
    )
    print(f"  {'Model':<14} {'Depth std mean':>16} {'Depth std median':>17}  Bar (lower=better)")
    print(f"  {'-'*14} {'-'*16} {'-'*17}  {'─'*20}")
    for s in stats_list:
        dm = s["depth_std_mean"]
        dd = s["depth_std_median"]
        dm_str = f"{dm:.4f}" if dm is not None else "  n/a  "
        dd_str = f"{dd:.4f}" if dd is not None else "  n/a  "
        # Invert bar: shorter bar = lower std = better
        inv_val = max_std - (dm if dm is not None else max_std)
        bar = _bar(inv_val, max_std)
        print(f"  {s['model']:<14} {dm_str:>16} {dd_str:>17}  {bar}")

    # --- Ranked summary ---
    print(f"\n  6. RANKED SUMMARY")
    print(f"  {SEP}")
    print("  Scoring rubric (higher is better for each criterion):")
    print("    A. Avg detections/image     — volume")
    print("    B. Mean confidence          — quality of detections reported")
    print("    C. Unique labels            — diversity (good for scene graphs)")
    print("    D. SAM2 match rate          — 3D coord quality")
    print("    E. Depth cleanliness (inv.) — lower std = better")

    # Normalise each metric to [0, 1] and sum for a rough composite score
    def norm(vals, invert=False):
        clean = [v for v in vals if v is not None]
        if not clean or max(clean) == min(clean):
            return [0.5] * len(vals)
        lo, hi = min(clean), max(clean)
        normed = [(v - lo) / (hi - lo) if v is not None else 0.0 for v in vals]
        return [1.0 - n for n in normed] if invert else normed

    avgs   = [s["avg_per_image"]      for s in stats_list]
    confs  = [s["conf_mean"]          for s in stats_list]
    divs   = [s["n_unique_labels"]    for s in stats_list]
    rates  = [s["mask_match_rate"]    for s in stats_list]
    stds   = [s["depth_std_mean"]     for s in stats_list]

    n_avg  = norm(avgs)
    n_conf = norm(confs)
    n_div  = norm(divs)
    n_rate = norm(rates)
    n_std  = norm(stds, invert=True)  # lower depth_std is better

    weights = {"volume": 0.15, "conf": 0.20, "diversity": 0.20, "match": 0.30, "depth": 0.15}
    scores = [
        weights["volume"] * a
        + weights["conf"] * c
        + weights["diversity"] * d
        + weights["match"] * r
        + weights["depth"] * s
        for a, c, d, r, s in zip(n_avg, n_conf, n_div, n_rate, n_std)
    ]

    ranked = sorted(zip(scores, stats_list), reverse=True)
    print(f"\n  {'Rank':<5} {'Model':<14} {'Score':>7}  {'Volume':>7}  {'Conf':>6}  {'Diversity':>9}  {'Match%':>7}  {'Depth(↓)':>9}")
    print(f"  {'-'*5} {'-'*14} {'-'*7}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*9}")
    for rank, (score, s) in enumerate(ranked, 1):
        rate_str = f"{s['mask_match_rate']:.1%}" if s["mask_match_rate"] is not None else "  n/a"
        std_str  = f"{s['depth_std_mean']:.3f}"  if s["depth_std_mean"]  is not None else "  n/a"
        conf_str = f"{s['conf_mean']:.3f}"        if s["conf_mean"]       is not None else "  n/a"
        print(
            f"  #{rank:<4} {s['model']:<14} {score:>7.3f}  "
            f"{s['avg_per_image']:>7.1f}  {conf_str:>6}  {s['n_unique_labels']:>9}  "
            f"{rate_str:>7}  {std_str:>9}"
        )

    print(f"\n  Weights: volume×{weights['volume']}, conf×{weights['conf']}, "
          f"diversity×{weights['diversity']}, SAM2-match×{weights['match']}, depth-clean×{weights['depth']}")
    print(f"  (Adjust weights at top of _compute_scores() in this file to reprioritize)")
    print(f"\n{'═' * 90}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare detector outputs from scene_understanding.py")
    parser.add_argument("--output_dir", default="output_scene", help="Pipeline output directory")
    parser.add_argument("--save_json", default="eval_detectors.json", help="Path to save results JSON ('' to skip)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory '{output_dir}' does not exist.")
        print("Run scene_understanding.py on at least one image first.")
        sys.exit(1)

    print(f"Reading detector outputs from: {output_dir.resolve()}")

    # Collect per-model JSONs
    per_model_jsons, stems = collect_per_model_jsons(output_dir)
    if not any(per_model_jsons.values()):
        print("No per-model JSON files found. Ensure scene_understanding.py has run.")
        sys.exit(1)
    print(f"  Found outputs for {len(stems)} image(s): {', '.join(stems[:5])}{'...' if len(stems) > 5 else ''}")

    # Collect scene JSON objects for SAM2 match rate and depth quality
    scene_objs_by_source = collect_scene_jsons(output_dir)

    # Compute stats per model
    stats_list = []
    for model in MODELS:
        images_dets = per_model_jsons.get(model, [])
        source_str  = SOURCE_MODEL_MAP[model]
        # Also include OWL-ViT under pix2seq's scene objects
        scene_objs = scene_objs_by_source.get(source_str, [])
        if model == "pix2seq":
            scene_objs += scene_objs_by_source.get(OWL_VIT_SOURCE, [])
        stats = compute_model_stats(model, images_dets, scene_objs)
        stats_list.append(stats)

    render_summary(stats_list)
    compute_overlap(stats_list)

    if args.save_json:
        out_path = Path(args.save_json)
        with open(out_path, "w") as f:
            json.dump(stats_list, f, indent=2)
        print(f"\n  Full results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
