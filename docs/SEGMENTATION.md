# Segmentation

## Overview

The pipeline uses two complementary segmentation strategies that run simultaneously and are merged
before any further processing. This dual-segmentor design ensures complete scene coverage:
object-level masks from GroundingDINO + SAM2, and part-level / small-object masks from SAM2 AMG.

---

## Mode 1 — GroundedSAM2 (Object-Level)

**Models**: GroundingDINO (`IDEA-Research/grounding-dino-base`) + SAM2 (`SAM2ImagePredictor`)

**Flow**:
```
Text query → GroundingDINO → bounding boxes + labels
                                      ↓
                          SAM2 (prompted per bbox)
                                      ↓
                     One high-quality binary mask per detected entity
```

**Text query strategy**: A broad category string covers all common scene graph objects:
```
"person. vehicle. furniture. electronics. food. clothing. animal.
 building. plant. container. tool. appliance. sign. ..."
```
GDINO's open-vocabulary architecture handles novel objects not in this list gracefully.

**Why object-level matters**: SAM2 AMG alone produces part-level masks — a car generates
separate masks for door, wheel, window, roof. GroundingDINO detects at the entity level ("car" =
one bounding box), so SAM2 generates one clean mask for that one entity. This gives the correct
granularity for scene graph nodes.

**Label attachment**: GDINO attaches text label + confidence directly to each mask at detection
time. These become `grounded_sam2_label` and `grounded_sam2_confidence` in the output.

**Config**:
```python
grounding_dino_box_thresh  = 0.15   # detection confidence gate
grounding_dino_text_thresh = 0.15   # text-image alignment gate
```

---

## Mode 2 — SAM2 AMG (Automatic, Part-Level)

**Model**: SAM2 `SAM2AutomaticMaskGenerator`

**Flow**:
```
Dense point grid over image → SAM2 segments every coherent region
                                          ↓
                      Many binary masks (no labels attached)
```

**What it captures that GroundedSAM2 misses**:
- Parts of objects (drawer handle, wheel spoke, label on a bottle)
- Very small objects below GDINO's detection threshold
- Background / texture regions (sky, floor, wall patches)
- Objects outside the text query vocabulary

**No labels**: AMG masks carry no semantic label — labelling happens later in `_label_mask()`
using Florence-2 or GRiT on the masked crop.

**Config**:
```python
sam2_amg_pred_iou_thresh          = 0.80   # AMG-level quality filter
sam2_amg_stability_score_thresh   = 0.92   # AMG-level stability filter
sam2_amg_min_mask_region_area     = 100    # minimum mask area in pixels
```

---

## Dual-Segmentor Merge

When `run_both_segmentors = True` (default), both modes run on every image and their results
are merged with IoU-based deduplication.

**Algorithm**:
```
gdino_masks = GroundedSAM2.generate(image)      ← object-level
amg_masks   = SAM2_AMG.generate(image)          ← part-level

for each amg_mask in amg_masks:
    for each gdino_mask in gdino_masks:
        IoU(amg_mask, gdino_mask) = |amg ∩ gdino| / |amg ∪ gdino|
        if IoU >= iou_dedup_threshold (0.7):
            mark amg_mask as duplicate → discard
            break
    if not duplicate:
        keep amg_mask (it covers something GDINO missed)

final_masks = gdino_masks + [surviving AMG masks]
```

Each surviving mask is tagged:
```python
mask["source_model"] = "GroundedSAM2"  # or "SAM2_AMG"
```

This tag becomes the `segmentor` field in the scene JSON.

**Rationale for IoU threshold 0.7**: A threshold of 0.7 allows for moderate segmentation
disagreement (GDINO mask may be slightly tighter or looser than AMG's). Masks with IoU > 0.7
represent the same object — the GDINO-prompted mask is kept because it is typically cleaner
(single prompted point vs. grid point). Masks with IoU < 0.7 represent genuinely different
regions.

**Config**:
```python
run_both_segmentors          = True
run_both_segmentors_iou_dedup = 0.7
```

---

## AMG-Only Fallback

If GroundingDINO fails to import or produces zero detections above threshold, the pipeline
automatically falls back to AMG-only mode. `grounded_sam2_fallback_to_amg = True` in config
controls this behaviour. The pipeline never fails silently.

---

## All Filtering Disabled

All post-hoc mask filters are disabled by default — every mask from both segmentors passes into
the scene graph:

| Filter | Config field | Default |
|---|---|---|
| Minimum SAM2 stability score | `sam2_post_filter_min_stability` | `0.0` (off) |
| Minimum SAM2 predicted IoU | `sam2_post_filter_min_pred_iou` | `0.0` (off) |
| Minimum mask area (pixels) | `sam2_post_filter_min_area_px` | `0` (off) |
| Maximum mask area fraction | `sam2_post_filter_max_area_fraction` | `1.0` (off) |

This ensures small objects (< 1500px), background patches (> 30% of image), and low-stability
masks all appear in the output for full scene coverage.

---

## Output: Scene JSON Object Entry

Every mask produces one object entry in `{stem}_scene.json`. Fields:

```json
{
  "id": "obj_0",
  "label": "sofa",
  "conf": 0.87,
  "segmentor": "GroundedSAM2",

  "bbox": [120, 80, 540, 410],
  "sam2_mask_index": 0,

  "grounded_sam2_label": "sofa",
  "grounded_sam2_confidence": 0.87,

  "mask_centroid_2d": [330, 245],

  "depth_stats": {
    "min": 1.82,
    "max": 2.95,
    "mean": 2.28,
    "median": 2.31,
    "std": 0.14,
    "num_pixels": 8420,
    "z_val": 2.31,
    "z_val_pixels": 1240,
    "possibly_transparent": false,
    "depth_separation_from_background": 0.89
  },

  "depth_stats_no_erosion": {
    "z_val": 2.18,
    "..."  : "..."
  },

  "coordinates_3d": { "x": -0.42, "y": 0.15, "z": 2.31 },
  "coordinates_3d_no_erosion": { "x": -0.41, "y": 0.14, "z": 2.18 },
  "mask_centroid_2d_no_erosion": [332, 246],

  "sources": { "GRiT": false, "Pix2SG": true, "Florence2": true }
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `id` | str | Unique object ID in this image (`obj_N`) |
| `label` | str | Best available semantic label (4-priority chain) |
| `conf` | float | Confidence of the winning label source |
| `segmentor` | str | `"GroundedSAM2"` or `"SAM2_AMG"` |
| `bbox` | [x1,y1,x2,y2] | SAM2 bounding box in pixel coordinates |
| `sam2_mask_index` | int | Index in the amg_masks list for this image |
| `grounded_sam2_label` | str | GDINO text label (or fallback label if AMG) |
| `grounded_sam2_confidence` | float | GDINO detection confidence |
| `mask_centroid_2d` | [cx, cy] | Depth-weighted centre-of-mass of the mask |
| `depth_stats` | dict | Depth statistics with adaptive erosion applied |
| `depth_stats_no_erosion` | dict | Same stats computed without erosion (comparison) |
| `coordinates_3d` | {x,y,z} | 3D world position in metres (with erosion) |
| `coordinates_3d_no_erosion` | {x,y,z} | 3D world position without erosion (comparison) |
| `mask_centroid_2d_no_erosion` | [cx,cy] | Centroid computed without erosion |

### `depth_stats` Sub-fields

| Field | Description |
|---|---|
| `z_val` | **Primary depth estimate** — histogram mode of inner-circle pixels (metres). More robust than mean/median. See [DEPTH_ACCURACY.md](DEPTH_ACCURACY.md). |
| `z_val_pixels` | Number of pixels used to compute `z_val` |
| `min / max` | Depth range across all (sigma-clipped) mask pixels |
| `mean / median / std` | Distribution statistics of mask depth values |
| `num_pixels` | Total foreground pixel count of the binary mask |
| `possibly_transparent` | `true` if the mask depth is indistinguishable from the surrounding background — indicates glass, water, or transparent material |
| `depth_separation_from_background` | Absolute depth difference (metres) between mask mean depth and the 5px border ring around the mask. Low value → transparent. |

---

## Output Directory Layout

```
output_scene/
├── depth/
│   └── {stem}_depth_metric.npy         ← float32 metric depth array, shape (H, W)
├── scene_graph/
│   ├── {stem}_depth_map.png            ← colourised depth visualisation
│   ├── {stem}_sam2_segmentation.png    ← coloured mask overlay on original image
│   ├── depth_mask/
│   │   └── {stem}_obj_N_depth_mask.png ← per-object: original image + mask overlay + depth
│   └── masks/
│       └── {stem}_obj_N_mask.png       ← per-object: binary mask PNG
└── {stem}_scene.json                   ← full scene graph
```

---

## Relation Graph in Scene JSON

After segmentation and labelling, Pix2SG + Florence-2 build a relation graph:

```json
"relations": [
  {
    "subject": "obj_0",
    "predicate": "in_front_of",
    "object": "obj_2",
    "score": 0.75,
    "source_layer": "spatial_scaffold"
  },
  {
    "subject": "obj_1",
    "predicate": "holds",
    "object": "obj_3",
    "score": 0.75,
    "source_layer": "florence2"
  }
]
```

`source_layer` indicates which system predicted the relation:
- `"spatial_scaffold"` — geometric rule from Pix2SG (centroid direction, depth difference, pixel overlap)
- `"florence2"` — semantic predicate from Florence-2 colour-overlay caption

See [LABELLING_AND_RELATIONS.md](LABELLING_AND_RELATIONS.md) for details.
