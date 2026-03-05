# CITV Pipeline — Full Technical Notes

Everything discussed across all sessions: what the pipeline does, every fix applied,
the mathematics, the reasons, and ASCII flow diagrams.

---

## Table of Contents

1. [What the Pipeline Does](#1-what-the-pipeline-does)
2. [Full Pipeline Flow Diagram](#2-full-pipeline-flow-diagram)
3. [Model Stack — Why Each Model Was Chosen](#3-model-stack--why-each-model-was-chosen)
4. [Stage-by-Stage Breakdown with Math](#4-stage-by-stage-breakdown-with-math)
5. [Fixes Applied (5.1 – 5.7)](#5-fixes-applied-51--57)
6. [Dead Code Removed](#6-dead-code-removed)
7. [Depth Accuracy — Full Explanation](#7-depth-accuracy--full-explanation)
8. [Coordinate Accuracy — Back-Projection Math](#8-coordinate-accuracy--back-projection-math)
9. [Dual Segmentor Strategy](#9-dual-segmentor-strategy)
10. [Dual Erosion Comparison](#10-dual-erosion-comparison)
11. [Relation Pipeline](#11-relation-pipeline)
12. [Output Structure](#12-output-structure)
13. [Config Reference](#13-config-reference)
14. [Reproduction / Setup](#14-reproduction--setup)

---

## 1. What the Pipeline Does

**Goal:** Given a folder of RGB images, produce a 3D scene graph for each image.

A scene graph is a structured representation of what objects are in the scene,
where they are in 3D space, and how they relate to each other
(e.g. "person — sitting on — chair", "cup — on — table").

**Output per image:**
- `{stem}_scene.json` — full scene graph: objects with 3D coordinates, depth stats,
  labels, confidence scores, and semantic relations
- `{stem}_depth.png` — colourised depth map
- `{stem}_depth_16bit.png` — 16-bit depth map (optional)
- `{stem}_viz.png` — annotated image with mask overlays, object labels, depth values
- Per-object mask PNGs (optional)
- Depth `.npy` arrays (optional)

---

## 2. Full Pipeline Flow Diagram

```
INPUT IMAGE
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 0 — Undistortion (optional)                   │
│                                                     │
│  If camera_calibration_file set:                    │
│    Load JSON → K matrix + dist_coeffs               │
│    cv2.undistort(img, K, dist_coeffs) → img_corr    │
│  Else: use raw image                                │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1 — Camera Intrinsics                         │
│                                                     │
│  Priority 1: calibration JSON (most accurate)       │
│  Priority 2: explicit config fx/fy/cx/cy            │
│  Priority 3: FOV estimate                           │
│    fx = W / (2 * tan(FOV/2))                        │
│    cx = W/2, cy = H/2                               │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2 — Depth Estimation                          │
│                                                     │
│  CLIP (loads, classifies, UNLOADS to free VRAM)     │
│    image → "indoor" or "outdoor"                    │
│         │                                           │
│         ├─ indoor  → Depth-Anything-V2-Indoor-Large │
│         └─ outdoor → Depth-Anything-V2-Outdoor-Large│
│                                                     │
│  Model → metric_depth  (float32, metres, H×W)       │
│  No scaling needed (metric output = true metres)    │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3 — Segmentation (Dual Segmentor)             │
│                                                     │
│  GroundedSAM2 (primary):                            │
│    GroundingDINO → bboxes + labels per object       │
│    SAM2 prompted per bbox → object-level masks      │
│    (one clean mask per detected entity)             │
│         │                                           │
│  SAM2 AMG (secondary, run_both_segmentors=True):    │
│    Automatic mask generation (grid of points)       │
│    Catches: parts, small objects, background        │
│         │                                           │
│  IoU dedup: AMG mask dropped if IoU > 0.7 with      │
│    any GroundedSAM2 mask (GDINO mask is better)     │
│         │                                           │
│  → merged amg_masks list (all masks, no filtering)  │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4 — Labelling + Depth per Object              │
│                                                     │
│  For each mask:                                     │
│    Label priority:                                  │
│      1. GDINO label (if mask from GroundedSAM2)     │
│      2. Florence-2 <OD> on cropped mask region      │
│      3. GRiT dense captioning (last resort)         │
│      4. YOLOv8x-cls (final fallback)                │
│                                                     │
│    Depth extraction:                                │
│      mask_pixels = metric_depth[mask_binary]        │
│      → adaptive erosion (shrink mask edges)         │
│      → sigma-clipping (remove outliers)             │
│      → histogram mode → z_val (metres)              │
│      → transparency check (border ring compare)     │
│      → back-project → X, Y, Z in metres            │
│                                                     │
│    (If depth_erosion_comparison=True: run twice,    │
│     store both eroded and raw depth stats)          │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5 — Relations (Pix2SG + Florence-2)           │
│                                                     │
│  For each pair of objects where mask IoU > 0.02:    │
│    Render crop with subject=RED, object=BLUE        │
│    Florence-2 <CAPTION> → sentence                  │
│    Extract canonical predicate (verb phrase)        │
│    → relation entry: {subject, predicate, object}   │
│                                                     │
│  Also: spatial scaffold from Pix2SG                 │
│    (left-of, above, near, far, etc. from 3D coords) │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 6 — Serialise                                 │
│                                                     │
│  Strip numpy arrays (_sam2_mask_array)              │
│  json.dump → {stem}_scene.json                      │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Stage 7 — Visualisation                             │
│                                                     │
│  Draw: mask overlays, label [M] tags,               │
│  depth values, bounding boxes, centroid dots        │
│  → {stem}_viz.png                                   │
└─────────────────────────────────────────────────────┘
     │
     ▼
OUTPUT: scene_graph/{stem}_scene.json
        depth_maps/{stem}_depth.png
        visualizations/{stem}_viz.png
        masks/{stem}_obj_{i}_mask.png  (optional)
```

---

## 3. Model Stack — Why Each Model Was Chosen

| Model | Role | Why |
|---|---|---|
| **CLIP ViT-B/32** | Indoor/outdoor classifier | Fast, 340MB, zero-shot; unloaded after use to free VRAM |
| **Depth Anything V2 Metric** | Metric depth (metres) | State-of-art monocular depth; metric variant gives true metres without scaling |
| **GroundingDINO v2** | Open-vocab object detection | Text-prompted detection; finds any object category without fixed class list |
| **SAM2** | Instance segmentation | Produces clean, prompt-able masks; AMG mode for part-level coverage |
| **Florence-2** | Labelling + relation prediction | Single model for both tasks; <OD> for labels, <CAPTION> for relations |
| **GRiT** | Dense captioning fallback | Produces captions for any crop; last resort when GDINO+Florence-2 fail |
| **YOLOv8x-cls** | Classification fallback | Fast top-1 class for any crop |

---

## 4. Stage-by-Stage Breakdown with Math

### Stage 0 — Lens Undistortion

OpenCV lens model:

```
x_distorted = x(1 + k1*r² + k2*r⁴) + 2*p1*x*y + p2*(r² + 2x²)
y_distorted = y(1 + k1*r² + k2*r⁴) + p1*(r² + 2y²) + 2*p2*x*y

where r² = x² + y²
k1, k2 = radial distortion coefficients
p1, p2 = tangential distortion coefficients
```

`cv2.undistort(img, K, dist_coeffs)` inverts this to produce a rectified image.
Critical: all depth and mask processing runs on the undistorted image, so back-projected
3D coordinates correspond to real-world geometry.

### Stage 1 — Camera Intrinsics

The camera matrix K maps 3D camera-space points to 2D image pixels:

```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]

fx, fy = focal lengths in pixels
cx, cy = principal point (ideally image centre W/2, H/2)
```

FOV fallback (least accurate, ~10–30% error vs calibration):
```
fx = W / (2 * tan(FOV_degrees * π/360))
fy = fx   (assuming square pixels)
cx = W / 2
cy = H / 2
```

Calibration accuracy (OpenCV checkerboard):
```
fx, fy error: < 0.5%
cx, cy error: < 2 pixels
RMS reprojection error target: < 0.5 px (excellent), < 1.0 px (acceptable)
```

### Stage 2 — Depth Model

**Key bug fixed (Fix 5.1):**
HuggingFace depth-estimation pipeline returns two keys:
- `out["depth"]` → PIL Image, uint8, range 0–255 (NOT metres — this was the bug)
- `out["predicted_depth"]` → raw float32 tensor, range 0–∞ metres ← correct key

Indoor model: trained on NYUv2 (indoor rooms)
Outdoor model: trained on KITTI (streets, outdoor scenes)
CLIP automatically selects the right model per image.

**Output:** `metric_depth` — float32 numpy array, shape (H, W), values in metres.

### Stage 4 — Per-Object Depth Stats

**Step 1: Extract mask pixels**
```python
ys, xs = np.where(mask_binary)
depth_at_mask = metric_depth[ys, xs]   # shape: (N_pixels,)
```

**Step 2: Adaptive erosion** (shrink mask inward to avoid boundary bleed)
```
min_dim = min(mask_height, mask_width)

kernel_size:
  0  if min_dim < 15px   (tiny object — don't erode at all)
  1  if min_dim < 30px
  2  if min_dim < 60px
  config.mask_erosion_kernel_size  otherwise (default: 5)

eroded_mask = cv2.erode(mask, kernel(kernel_size))
```

Reason: mask edges often bleed onto background depth values (the mask boundary
covers pixels that are partly foreground, partly background). Erosion removes these.

**Step 3: Sigma-clipping** (remove statistical outliers)
```
μ = mean(depth_at_mask)
σ = std(depth_at_mask)
keep = |depth_pixel - μ| < outlier_sigma * σ    (outlier_sigma = 2.0)
depth_clean = depth_at_mask[keep]
```

Reason: a glass window or translucent object produces bimodal depth distribution.
Clipping removes the background mode, keeping only the foreground depth.

**Step 4: Histogram mode (z_val)**
```
Use inner 50% of mask pixels (depth_central_fraction = 0.5)
Bin them into a histogram
z_val = centre of the bin with the most pixels
```

Reason: the histogram mode is more robust than mean for objects with
non-uniform depth (curved surfaces, occluded areas). Mean would be pulled
toward background bleed. Mode picks the dominant depth layer.

**Step 5: Depth-weighted centroid**
```
weight_i = 1 / (depth_i + ε)   where ε = 1e-6

centroid_x = Σ(weight_i * x_i) / Σ(weight_i)
centroid_y = Σ(weight_i * y_i) / Σ(weight_i)
```

Reason: closer pixels (lower depth, higher weight) dominate. This pulls the
centroid toward the nearest surface of the object — more physically meaningful
than a pure geometric centroid.

**Step 6: Transparency detection**
```
border_ring = dilate(mask, 3px) AND NOT mask
border_depth = mean(metric_depth[border_ring])
mask_depth   = mean(depth_clean)
separation   = |mask_depth - border_depth|

if separation < depth_transparency_threshold (0.15m):
    possibly_transparent = True
```

Reason: glass, plastic wrap, and water have depth values nearly identical to
the background behind them (the sensor "sees through" them). The border ring
comparison detects this.

### Stage 8 — Back-Projection to 3D

```
Given pixel (u, v) and depth z (metres):

X = (u - cx) * z / fx      ← metres, horizontal
Y = (v - cy) * z / fy      ← metres, vertical (positive = down)
Z = z                       ← metres, depth from camera

3D point = [X, Y, Z]
```

Accuracy depends on z_val accuracy:
- With calibration + metric model: X, Y accurate to ~2–5% for well-lit, non-transparent objects
- With FOV estimate only: X, Y error ~10–30% due to fx/fy uncertainty
- Depth (Z) accuracy: ~5–10% for Depth Anything V2 Metric at training distribution scenes

---

## 5. Fixes Applied (5.1 – 5.7)

### Fix 5.1 — Depth Model Variant Selection (auto/indoor/outdoor)

**Problem:** A single depth model trained on indoor data gives wrong scale for outdoor
scenes (e.g. a street appears only 2m wide instead of 10m wide).

**Solution:** CLIP classifies each image → loads matching metric model.
```
depth_model_variant = "auto"  →  CLIP classifies → indoor or outdoor model
depth_model_variant = "indoor"  →  always indoor model (NYUv2-trained)
depth_model_variant = "outdoor" →  always outdoor model (KITTI-trained)
```

CLIP loads, classifies, then **immediately unloads** (~340MB VRAM freed) before the
depth model loads. This is critical on GPUs with 8–12GB VRAM.

**Also fixed:** `out["depth"]` → `out["predicted_depth"]` — the HuggingFace pipeline
was returning a PIL uint8 image (0–255) instead of the raw metric float32 tensor.
This caused all depth values to be 0–255 instead of true metres. Root cause of the
most impactful accuracy bug in the pipeline.

### Fix 5.2 — Camera Intrinsics via OpenCV Calibration

**Problem:** FOV estimate (60°) gives ~10–30% error in fx/fy → proportional error in X, Y.

**Solution:** OpenCV checkerboard calibration produces K matrix with < 0.5% error.

```
tools/calibrate_camera.py --images calib_imgs/ --pattern 9x6 --square_size 0.025 --out calibration.json
```

Output JSON: `{fx, fy, cx, cy, k1, k2, p1, p2, image_size, rms_reprojection_error}`

Priority order in pipeline:
1. `camera_calibration_file` JSON (most accurate)
2. Explicit `camera_fx/fy/cx/cy` in config
3. `camera_fov_degrees` FOV estimate (least accurate)

### Fix 5.3 — Grounded SAM2 (GroundingDINO + SAM2)

**Problem:** SAM2 AMG (automatic grid) produces part-level masks (e.g. "shirt sleeve"
instead of "person"). No semantic labels. Cannot identify specific objects.

**Solution:** GroundingDINO detects objects by text query → SAM2 prompted per bbox.
Each detected entity gets one clean, object-level mask with a semantic label.

Text query (broad to catch everything):
```
"person. animal. vehicle. furniture. appliance. food. clothing.
 container. tool. building. plant. electronics. object."
```

SAM2 AMG kept as fallback (and now also run alongside GDINO for parts/small objects).

### Fix 5.4 — Florence-2 Object Labelling

**Problem:** GRiT produces verbose captions ("a red ceramic mug on a wooden table")
instead of clean labels ("mug"). Heuristic noun extraction was fragile.

**Solution:** Florence-2 `<OD>` task returns structured `{label, bbox}` pairs.
Highest-confidence label taken for the mask crop. Clean, short labels.

Label priority chain:
```
GDINO label → Florence-2 <OD> → GRiT caption → YOLOv8x-cls
```

### Fix 5.5 — Post-hoc Mask Filters (now fully disabled)

**Original:** Strict quality filters removed low-confidence masks:
- `stability_score >= 0.85`
- `pred_iou >= 0.82`
- `area >= 1500px` (removed small objects)
- `area_fraction <= 0.30` (removed background)

**Current state (fully disabled):** All thresholds set to 0 / 1.0.
Every mask is kept — including tiny objects (buttons, coins), background regions
(sky, floor), and part-level masks.

Reason: research/analysis use case — need complete coverage for scene graph.
Re-enable by setting non-zero thresholds in config.py.

### Fix 5.6 — Relation Enhancement (Florence-2 + Pix2SG)

**Problem:** SGTR was the only relation source — removed due to complex setup,
poor generalisation, dead code.

**Solution:** Two-source relation pipeline:
1. **Pix2SG spatial scaffold** — geometric relations from 3D positions
   (left-of, above, near/far based on depth thresholds)
2. **Florence-2 semantic relations** — for overlapping mask pairs:
   - Render crop: subject=RED overlay, object=BLUE overlay
   - Florence-2 `<CAPTION>` → sentence
   - Extract canonical predicate (verb phrase)
   - Stored as `source_layer: "florence2"`

Overlap threshold: `relation_min_mask_overlap = 0.02` (2% pixel IoU).
Only pairs with spatial contact get semantic relation prediction (avoids running
VQA on 10,000 non-touching pairs in a dense scene).

### Fix 5.7 — Depth Accuracy: Adaptive Erosion + Outlier Rejection

Three improvements to per-object depth extraction:

**Adaptive erosion** (kernel scales to object size):
```
Prevents destroying thin objects (pencil, wire, finger) while still
removing boundary bleed on large objects.
```

**Sigma-clipping** (`depth_outlier_sigma = 2.0`):
```
Removes pixels more than 2σ from mask mean depth.
Handles: glass, reflective surfaces, partially occluded objects.
```

**Transparency detection** (border ring comparison):
```
Objects where mask depth ≈ background depth → flagged possibly_transparent.
Downstream code can then treat depth value as unreliable.
```

---

## 6. Dead Code Removed

The following were in `scene_understanding.py` and completely removed.
They either never worked in this codebase, depended on unavailable checkpoints,
or were superseded by better approaches.

| Removed | Lines (approx) | Reason |
|---|---|---|
| `FasterRCNNWrapper` | ~120 lines | Replaced by GroundedSAM2 for detection |
| `DETRWrapper` | ~80 lines | Replaced by GroundingDINO |
| `Pix2SeqWrapper` / OWLViT | ~100 lines | OWLViT checkpoint unavailable; never ran |
| `SGSGWrapper` (SGTR) | ~330 lines | SGTR repo removed; checkpoint unavailable |
| SGTR path setup | ~20 lines | Dead without SGTR repo |
| `_OIV6_PREDICATES` dict | ~70 lines | Only used by SGSGWrapper |
| `_save_model_output` method | ~20 lines | Never called anywhere |
| All `sgtr_stats` references | ~5 lines | Caused NameError after SGTR removal |

Total removed: ~745 lines (~22% of original file).

---

## 7. Depth Accuracy — Full Explanation

### What the depth model produces

Depth Anything V2 Metric is a monocular depth estimation model.
"Monocular" = single camera, single image — no stereo baseline.

**Absolute accuracy (metric model, good conditions):**
- Indoor (NYUv2-trained): MAE ~5–8cm for objects within 0.5–5m
- Outdoor (KITTI-trained): MAE ~10–30cm for objects within 5–50m
- Challenging: glass, mirrors, transparent objects, very dark surfaces

**Why metric models are better than relative models:**
```
Relative model output: arbitrary scale (e.g. 0–1 or 0–255)
  → requires a depth_scale_factor hack (was: ×10) to approximate metres
  → inconsistent across images

Metric model output: true metres, consistent across images
  → depth_scale_factor = 1.0 always
  → can directly compare depths across frames
```

### How depth map + mask masking works

```
metric_depth shape: (H, W)   — every pixel has a depth value in metres

mask_binary shape: (H, W)    — True where this object's pixels are

depth extraction:
    ys, xs = np.where(mask_binary)     # pixel coordinates of mask
    depth_values = metric_depth[ys, xs]  # depth at each mask pixel
```

This is a direct index into the depth array using the mask's pixel coordinates.
The result is an array of depth measurements (one per mask pixel) in true metres.

The pipeline then applies erosion → sigma-clipping → histogram mode
to get a single robust representative depth value `z_val`.

### Sources of depth error

1. **Model error** (~5–10%): monocular depth is fundamentally ambiguous
2. **Boundary bleed** (→ fixed by erosion): mask edges cover background pixels
3. **Transparent objects** (→ flagged): glass/water depth ≈ background depth
4. **Wrong model variant** (→ fixed by CLIP): indoor model on outdoor scene or vice versa
5. **Distortion** (→ fixed by undistortion): barrel/pincushion distortion shifts pixel positions

---

## 8. Coordinate Accuracy — Back-Projection Math

### Full derivation

Camera model (pinhole):
```
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

Inverting to get 3D from pixel + depth:
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = z_val   (from depth model, metres)
```

`coordinates_3d` in scene JSON stores `[X, Y, Z]` in metres relative to camera.

### Centroid pixel used for back-projection

The depth-weighted centroid `(u_c, v_c)` is computed from all mask pixels:
```
weight_i = 1 / (depth_i + ε)
u_c = Σ(weight_i * u_i) / Σ weight_i
v_c = Σ(weight_i * v_i) / Σ weight_i
```

This centroid is back-projected with `z_val` to produce `coordinates_3d`.

### Dual erosion comparison

Every object gets computed **twice**:

| Field | With adaptive erosion | Without erosion |
|---|---|---|
| `depth_stats` | eroded mask depth | raw mask depth |
| `coordinates_3d` | from eroded z_val | from raw z_val |
| `mask_centroid_2d` | eroded centroid | raw centroid |
| suffix | (default) | `_no_erosion` |

This lets you compare how much erosion affects depth accuracy per object.
Thin objects (pencil, finger) show large differences; large flat objects show small differences.

---

## 9. Dual Segmentor Strategy

```
run_both_segmentors = True  →  both run on every image

GroundedSAM2                        SAM2 AMG
────────────────                    ──────────────────
Text query → GDINO                  Grid of points
→ object bboxes                     → all possible masks
→ SAM2 per bbox                     → parts, small objects,
→ one clean mask                      background regions
  per entity

        IoU deduplication
        ─────────────────
        For each AMG mask:
            IoU with any GDINO mask > 0.7?
            YES → discard (GDINO mask is better quality)
            NO  → keep (new information: part or small object)

        Merged list:
        [GDINO masks] + [non-duplicate AMG masks]

        Every mask gets:
            segmentor: "GroundedSAM2"  or  "SAM2_AMG"
```

Why this matters:
- GDINO alone misses: buttons, coins, text on packaging, textures, background sky
- AMG alone gives: part-level masks (shirt sleeve ≠ person), no semantic labels
- Together: complete coverage at both object and part level

---

## 10. Dual Erosion Comparison

`depth_erosion_comparison = True` causes every mask to be processed twice
through `_mask_depth_stats_and_3d(use_erosion=True/False)`.

Use cases:
- **Thin objects** (pencil, wire): erosion destroys the mask → use `_no_erosion` fields
- **Large objects** (wall, floor): erosion removes boundary bleed → use eroded fields
- **Debugging**: compare both to understand how boundary effects influence depth
- **Downstream analysis**: choose which set to use per object based on `min_dim`

The `use_erosion` parameter was added to `_mask_depth_stats_and_3d()` as a clean
parameter (no global state mutation) to support this without code duplication.

---

## 11. Relation Pipeline

```
objects_3d list (all objects with 3D coords)
         │
         ▼
Pix2SG.predict(objects_3d)
    │
    ├── Spatial scaffold
    │     For each object pair:
    │       depth A vs B → near/far (thresholds: 1.0m, 3.0m)
    │       bbox position → left-of, right-of, above, below
    │       mask overlap  → on-top-of, contains
    │
    └── Florence-2 semantic enrichment
          For each overlapping pair (mask IoU > 0.02):
            crop = bounding box of union
            overlay: subject pixels → RED tint
                     object pixels  → BLUE tint
            Florence-2 <CAPTION> → "a person sitting on a chair"
            extract predicate: "sitting on"
            canonical form: "sitting_on"
            → relation: {predicate, target_id, target_label, target_caption}

relation entry structure:
{
  "predicate":      "sitting_on",
  "target_id":      "obj_3_GroundedSAM2",
  "target_label":   "chair",
  "target_caption": "a wooden dining chair"
}
```

---

## 12. Output Structure

```
output_scene/
├── scene_graph/
│   └── {stem}_scene.json          ← main output
├── depth_maps/
│   ├── {stem}_depth.png           ← colourised depth
│   └── {stem}_depth_16bit.png     ← 16-bit depth (optional)
├── visualizations/
│   └── {stem}_viz.png             ← annotated image
└── masks/
    └── {stem}_obj_{i}_mask.png    ← per-object masks (optional)
```

### scene JSON structure (per object)

```json
{
  "id": "obj_0_GroundedSAM2",
  "label": "person",
  "conf": 0.87,
  "segmentor": "GroundedSAM2",
  "bbox": [x1, y1, x2, y2],
  "coordinates_3d": [X, Y, Z],
  "coordinates_3d_no_erosion": [X, Y, Z],
  "depth_stats": {
    "z_val": 2.34,
    "z_val_pixels": 2310,
    "possibly_transparent": false,
    "depth_separation_from_background": 0.82
  },
  "depth_stats_no_erosion": { ... },
  "mask_centroid_2d": [u, v],
  "mask_centroid_2d_no_erosion": [u, v],
  "sam2_mask_index": 0,
  "mask_matched": true,
  "mask_path": "masks/..._obj_0_mask.png",
  "sources": {
    "GroundedSAM2": {"label": "person", "conf": 0.87},
    "Florence2": {"label": "person", "caption": "a man in a blue shirt"},
    "GRiT": {"caption": "a man standing near a table"},
    "Pix2SG": {"relations": [...]}
  },
  "relations": [
    {
      "predicate": "standing_near",
      "target_id": "obj_2_GroundedSAM2",
      "target_label": "table",
      "target_caption": "a wooden dining table"
    }
  ]
}
```

---

## 13. Config Reference

All settings live in `config.py` → `PreprocessConfig` dataclass.

### Depth

| Field | Default | Description |
|---|---|---|
| `depth_model_variant` | `"auto"` | `"auto"` \| `"indoor"` \| `"outdoor"` |
| `depth_scale_factor` | `1.0` | Always 1.0 for metric models |
| `depth_adaptive_erosion` | `True` | Scale erosion kernel to object size |
| `mask_erosion_kernel_size` | `5` | Max erosion kernel (px) |
| `depth_central_fraction` | `0.5` | Inner fraction used for histogram |
| `depth_outlier_sigma` | `2.0` | Sigma-clipping threshold (0 = off) |
| `depth_transparency_check` | `True` | Enable border ring comparison |
| `depth_transparency_threshold` | `0.15` | Metres separation to flag transparent |
| `depth_erosion_comparison` | `True` | Compute both eroded and raw depth stats |

### Camera

| Field | Default | Description |
|---|---|---|
| `camera_calibration_file` | `None` | Path to calibration JSON |
| `apply_undistortion` | `True` | Apply cv2.undistort() |
| `camera_fx/fy/cx/cy` | `None` | Explicit intrinsics (priority 2) |
| `camera_fov_degrees` | `60.0` | FOV fallback (priority 3) |

### Segmentation

| Field | Default | Description |
|---|---|---|
| `run_both_segmentors` | `True` | Run GDINO+SAM2 AND SAM2 AMG |
| `run_both_segmentors_iou_dedup` | `0.7` | IoU threshold to dedup AMG masks |
| `grounding_dino_box_thresh` | `0.15` | GDINO detection confidence |
| `grounding_dino_text_thresh` | `0.15` | GDINO token confidence |
| `sam2_amg_min_mask_region_area` | `100` | Min mask area in px (AMG) |
| `grounded_sam2_fallback_to_amg` | `True` | Fall back to AMG if GDINO fails |

### Filtering (all disabled)

| Field | Default | Description |
|---|---|---|
| `sam2_post_filter_min_stability` | `0.0` | 0 = disabled |
| `sam2_post_filter_min_pred_iou` | `0.0` | 0 = disabled |
| `sam2_post_filter_min_area_px` | `0` | 0 = allow all sizes |
| `sam2_post_filter_max_area_fraction` | `1.0` | 1.0 = allow background |

### Relations

| Field | Default | Description |
|---|---|---|
| `florence2_relation_enabled` | `True` | Florence-2 semantic relations |
| `relation_min_mask_overlap` | `0.02` | Min IoU to attempt relation |
| `pix2sg_depth_near_threshold` | `1.0` | Metres — near/far boundary |
| `pix2sg_depth_far_threshold` | `3.0` | Metres — far boundary |

---

## 14. Reproduction / Setup

```bash
# 1. Clone your repo
git clone https://github.com/<you>/citv.git
cd citv

# 2. Run one-shot setup (clones GRiT+SAM2, downloads checkpoints,
#    builds detectron2+SAM2 C extension, installs all deps,
#    pre-downloads all HuggingFace models)
bash setup.sh

# 3. Run the pipeline
python scene_understanding.py --input_dir images --output_dir output_scene

# Optional: camera calibration for accurate intrinsics
python tools/calibrate_camera.py \
    --images path/to/checkerboard_frames/ \
    --pattern 9x6 \
    --square_size 0.025 \
    --out calibration.json
# Then set in config.py: camera_calibration_file = "calibration.json"
```

**Requirements:**
- Python 3.9–3.11
- CUDA GPU (≥ 8 GB VRAM)
- `nvcc` on PATH (`sudo apt install nvidia-cuda-toolkit`)
- ~20 GB free disk space

**Disk breakdown:**

| Component | Size |
|---|---|
| PyTorch + CUDA | ~5 GB |
| Depth-Anything-V2 × 2 | ~2.6 GB |
| Florence-2-large | ~900 MB |
| GRiT + detectron2 build | ~3 GB |
| SAM2 checkpoint | ~900 MB |
| GroundingDINO | ~341 MB |
| CLIP | ~340 MB |
| YOLOv8x-cls | ~130 MB |
| **Total** | **~13–15 GB** |
