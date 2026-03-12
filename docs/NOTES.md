# CITV Pipeline — Complete Technical Reference

A code-grounded, formula-complete explanation of every stage of the pipeline,
derived directly from `scene_understanding.py` and `depth.py`.

---

## Table of Contents

1. [Overview and Objective](#1-overview-and-objective)
2. [Full Pipeline Flow](#2-full-pipeline-flow)
3. [Stage 0 — Lens Undistortion](#3-stage-0--lens-undistortion)
4. [Stage 1 — Camera Intrinsics](#4-stage-1--camera-intrinsics)
5. [Stage 2 — Scene Classification (CLIP) and Depth Estimation](#5-stage-2--scene-classification-clip-and-depth-estimation)
6. [Stage 3 — RAM++ Dynamic Vocabulary + Dual Segmentation (GroundedSAM2 + SAM2 AMG)](#6-stage-3--dual-segmentation-groundedsam2--sam2-amg)
7. [Stage 4 — Per-Object Depth Stats and 3D Coordinates](#7-stage-4--per-object-depth-stats-and-3d-coordinates)
8. [Stage 4b — Object Labelling](#8-stage-4b--object-labelling)
9. [Stage 5 — Relation Prediction (Pix2SG + Florence-2)](#9-stage-5--relation-prediction-pix2sg--florence-2)
10. [Stage 6 — Serialisation](#10-stage-6--serialisation)
11. [Stage 7 — Visualisation](#11-stage-7--visualisation)
12. [All Formulas in One Place](#12-all-formulas-in-one-place)
13. [Fixes Applied (5.1–5.8)](#13-fixes-applied-51-58)
14. [Dead Code Removed](#14-dead-code-removed)
15. [Output File Structure](#15-output-file-structure)
16. [Config Reference](#16-config-reference)
17. [Reproduction](#17-reproduction)
18. [Inference Time and Performance Notes](#18-inference-time-and-performance-notes)

---

## 1. Overview and Objective

**Input:** A folder of RGB images (any resolution, any camera).

**Output per image:**
- `{stem}_scene.json` — structured scene graph: every object with its 3D position,
  depth statistics, semantic label, confidence, and relations to other objects
- `{stem}_depth.png` — false-colour depth map (INFERNO colormap)
- `{stem}_viz.png` — annotated overlay: mask colours, labels, depth values
- Per-object mask PNGs (optional)

**What "scene graph" means:** A labelled directed graph where:
- **Nodes** = detected objects (person, chair, cup …) with 3D coordinates [X, Y, Z]
- **Edges** = semantic relations (sitting_on, holding, next_to …) between pairs

The pipeline builds this entirely from a single RGB image — no depth sensor, no stereo,
no IMU. Everything is inferred.

---

## 2. Full Pipeline Flow

```
INPUT IMAGE FILE
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 0 — Undistortion (optional, if calibration JSON set)   │
│   cv2.undistort(img, K, dist_coeffs) → rectified image       │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 1 — Camera Intrinsics                                   │
│   Priority 1: calibration JSON  (< 0.5% error)               │
│   Priority 2: explicit fx/fy/cx/cy in config                  │
│   Priority 3: FOV estimate  (~10–30% error)                   │
│   → K = {fx, fy, cx, cy}                                      │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 2 — Depth Estimation                                    │
│                                                               │
│   CLIP (loads → classifies → UNLOADS to free 340MB VRAM)     │
│     image → softmax over indoor/outdoor prompts              │
│     → "indoor" or "outdoor"                                   │
│                                                               │
│   Depth Anything V2 Metric (matching variant loads)           │
│     img_rgb → out["predicted_depth"] → float32 metres H×W    │
│     resize to image dims with INTER_NEAREST                   │
│     → metric_depth  (true metres, no scaling)                 │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 3 — Segmentation                                        │
│                                                               │
│   RAM++ (runs first):                                         │
│     full image → Swin-L tagger → pipe-separated tags         │
│     → dynamic GDINO query (max 8 tags, period-separated)     │
│     sam2_wrapper.update_text_query(dynamic_query)            │
│     stored in metadata: rampp_tags, gdino_query_used         │
│                                                               │
│   GroundedSAM2 (primary):                                     │
│     dynamic query → GroundingDINO → N bboxes + labels        │
│     SAM2ImagePredictor.predict(bbox) → 1 mask per entity      │
│                                                               │
│   SAM2 AMG (secondary, run_both_segmentors=True):             │
│     grid of points → SamAutomaticMaskGenerator → all masks   │
│                                                               │
│   IoU deduplication:                                          │
│     for each AMG mask: IoU with any GDINO mask > 0.7?        │
│       YES → drop (GDINO mask is better quality)               │
│       NO  → keep (part / small object / background)           │
│                                                               │
│   → merged amg_masks list (no filtering, all masks kept)      │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 4 — Per-Object Depth + 3D + Labelling                   │
│                                                               │
│   For each mask in amg_masks:                                 │
│                                                               │
│   4a. Labelling (_label_mask):                                │
│       Priority: GDINO → Florence-2 → RAM++                   │
│       Florence-2: <MORE_DETAILED_CAPTION> → noun extraction  │
│                   <OD> fallback if caption gave "object"     │
│       Crop = bbox region, out-of-mask pixels → image mean     │
│                                                               │
│   4b. Depth stats (_mask_depth_stats_and_3d):                 │
│       i.   Adaptive erosion (kernel sized to min_dim)         │
│       ii.  depth_at_mask = metric_depth[ys, xs]               │
│       iii. Sigma-clipping: reject |d - μ| > σ * std          │
│       iv.  Transparency check (border ring comparison)        │
│       v.   Depth-weighted centroid (cx, cy)                   │
│       vi.  Histogram mode over inner 50% → z_val (metres)     │
│       vii. Back-projection → X, Y, Z                          │
│                                                               │
│       If depth_erosion_comparison=True: run steps i–vii       │
│       twice (use_erosion=True AND False) → store both sets    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 5 — Relations (Pix2SG.predict)                          │
│                                                               │
│   Layer 1: Spatial scaffold                                   │
│     For each object pair: depth, bbox position, mask overlap  │
│     → near/far, left-of, above, on-top-of                     │
│                                                               │
│   Layer 2: Florence-2 semantic (_enrich_with_florence2)       │
│     For pairs where mask IoU > relation_min_mask_overlap:     │
│       union bbox crop → subject=RED tint, object=BLUE tint    │
│       Florence-2 <MORE_DETAILED_CAPTION> (standalone)        │
│       → _parse_relation_phrase → canonical predicate         │
│       → add triplet                                           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 6 — Serialise                                           │
│   Strip _sam2_mask_array (numpy, not JSON-serialisable)       │
│   json.dump → {stem}_scene.json                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 7 — Visualisation                                       │
│   Solid-fill segmentation + α=0.45 tinted overlay + labels   │
│   → {stem}_sam2_segmentation.png                             │
│   → {stem}_sam2_tinted_overlay.png                           │
│   → {stem}_3d_viz.png                                        │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 0 — Lens Undistortion

**Why:** Every real lens introduces distortion. Barrel distortion (wide-angle) bends
straight lines outward. Pincushion distortion bends them inward. If uncorrected,
pixels near image edges are in the wrong position, making all depth back-projections
geometrically incorrect.

**OpenCV lens distortion model:**

```
Let (x, y) = normalised camera coordinates (pixel - principal point, divided by focal length)
r² = x² + y²

Radial distortion correction:
  x_r = x · (1 + k1·r² + k2·r⁴)
  y_r = y · (1 + k1·r² + k2·r⁴)

Tangential distortion correction (lens not perfectly parallel to sensor):
  x_t = 2·p1·x·y + p2·(r² + 2x²)
  y_t = p1·(r² + 2y²) + 2·p2·x·y

Full corrected coordinates:
  x_corrected = x_r + x_t
  y_corrected = y_r + y_t

Coefficients from calibration JSON:
  k1, k2 = radial distortion (most important — highest magnitude)
  p1, p2 = tangential distortion (usually small)
```

`cv2.undistort(img, K, dist_coeffs)` inverts this mapping analytically, producing a
rectified image where straight lines in the world appear straight in the image.

**When it runs:** Only if `camera_calibration_file` is set in config and
`apply_undistortion = True`. All subsequent processing (depth, masks, back-projection)
runs on the undistorted image.

**Camera calibration tool:**
```bash
python tools/calibrate_camera.py \
    --images path/to/checkerboard_frames/ \
    --pattern 9x6 \
    --square_size 0.025 \
    --out calibration.json
```

OpenCV's checkerboard method: finds N·M inner corners at known 3D positions
(z=0 plane, spacing = square_size metres), minimises reprojection error across
20+ images to solve for K and [k1, k2, p1, p2].

Accuracy: `fx, fy < 0.5% error`, `cx, cy < 2px error`, `RMS < 0.5px = excellent`.

---

## 4. Stage 1 — Camera Intrinsics

The **camera matrix K** maps 3D camera-space points to 2D image pixels:

```
        [ fx    0   cx ]
K  =    [  0   fy   cy ]
        [  0    0    1 ]

fx = focal length in pixels, x-axis
fy = focal length in pixels, y-axis  (fx ≈ fy for square pixels)
cx = principal point x  (ideally W/2)
cy = principal point y  (ideally H/2)
```

**Priority 1 — calibration JSON** (most accurate):
```
Load: fx, fy, cx, cy directly from JSON
Also load: k1, k2, p1, p2 for undistortion
Error: < 0.5% for fx/fy, < 2px for cx/cy
```

**Priority 2 — explicit config values:**
```
camera_fx, camera_fy, camera_cx, camera_cy set directly in config.py
Useful when manufacturer provides pixel-level specs (e.g. from EXIF)
```

**Priority 3 — FOV estimate** (least accurate):
```
fx = W / (2 · tan(FOV_degrees · π / 360))
fy = fx                 ← assumes square pixels
cx = W / 2
cy = H / 2

Default: FOV = 60°, so for W=1920:
  fx = 1920 / (2 · tan(30°)) = 1920 / (2 · 0.5774) = 1662 px

Error: 10–30% vs. real lens, depending on actual focal length
```

---

## 5. Stage 2 — Scene Classification (CLIP) and Depth Estimation

### 5a. CLIP Scene Classifier (`SceneTypeClassifier`)

**Why:** Depth Anything V2 has two metric variants:
- **Indoor** (NYUv2-trained): range ~0.1–10m, tuned for rooms/furniture
- **Outdoor** (KITTI-trained): range ~0.5–80m, tuned for streets/buildings

Using the wrong variant causes systematic scale errors that corrupt all coordinates.

**How CLIP classifies (from `depth.py:132–176`):**

```
Prompts:
  INDOOR  = ["a photo of an indoor room",
              "an interior space with furniture",
              "inside a building with walls and ceiling"]
  OUTDOOR = ["a photo taken outside with sky or open space",
              "an outdoor scene with trees, streets or buildings",
              "a landscape or street scene outside"]

Step 1: Encode image + all 6 texts with CLIP ViT-B/32
Step 2: logits = outputs.logits_per_image[0]   # shape: (6,)
Step 3: probs = softmax(logits)                 # sums to 1 across all 6 texts
Step 4: indoor_score  = mean(probs[0:3])
        outdoor_score = mean(probs[3:6])
Step 5: scene_type = "indoor" if indoor_score >= outdoor_score else "outdoor"
```

CLIP's contrastive training makes these softmax scores directly comparable —
they reflect how well the image matches each text prompt relative to all others.

**VRAM management:** CLIP (~340MB) loads → classifies one image → immediately
`unload()` is called (deletes model, calls `torch.cuda.empty_cache()`) before
the depth model loads. This is essential on 8–12GB GPUs.

### 5b. Depth Anything V2 Metric (`DepthAnythingV2Backend`)

**Critical implementation detail (from `depth.py:248–254`):**
```python
out = self.pipe(pil_img)

# out["depth"]            → PIL Image, uint8, range 0-255  ← WRONG (visualization only)
# out["predicted_depth"]  → torch.Tensor, float32, metres  ← CORRECT
depth = out["predicted_depth"]
```

This was the most impactful bug in the original pipeline: using `out["depth"]` gave
values 0–255 instead of true metres, making all coordinates wrong by ~100×.

**Resize strategy (`depth.py:36–50`):**
```python
cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
```

`INTER_NEAREST` is mandatory: it copies the nearest depth value without blending.
`INTER_LINEAR` would average adjacent depths at object boundaries, creating
"ghost" depth values where foreground and background blend — corrupting
per-mask depth extraction in Stage 4.

**Output:** `metric_depth` — float32 numpy array, shape (H, W), values in metres.
No `depth_scale_factor` scaling needed (metric models output true metres; config
keeps `depth_scale_factor = 1.0`).

**16-bit PNG saving (optional):**
```
max_range = 20.0m (indoor) or 80.0m (outdoor)
depth_16bit = clip(depth / max_range * 65535, 0, 65535).astype(uint16)
```

---

## 6. Stage 3 — Dual Segmentation (GroundedSAM2 + SAM2 AMG)

### 6a. RAM++ Dynamic Vocabulary (runs before GroundedSAM2)

Before any segmentation happens, **RAM++ (Recognize Anything Model++)** tags the
full scene image to produce a per-image vocabulary. The result is used as the
GroundingDINO text query, replacing the previous static generic category list.

```
RAM++ input:  full scene image (RGB numpy, resized to 384×384 internally)
RAM++ output: pipe-separated tag string → parsed to list
  e.g. beach image → ["beach", "cloudy", "coast", "footprint", "sea", "sand",
                       "shoreline", "sky"]
  e.g. kitchen  → ["alcohol", "bottle", "cake", "car", "counter top", "plate",
                   "paper plate", "platter"]

Dynamic GDINO query:
  "beach. cloudy. coast. footprint. sea. sand. shoreline. sky."
  (period-separated, max 8 tags from config.rampp_max_tags)
```

`sam2_wrapper.update_text_query(dynamic_query)` is called before `.generate()`.
Both the tags list and the exact query string used are stored in scene JSON metadata
under `rampp_tags` and `gdino_query_used`.

**Why this improves detection:** A static query like "person. furniture. vehicle."
misses scene-specific objects. A beach image gets no detections for "footprint" or
"shoreline" with the old query; RAM++ identifies exactly what's in the scene.

### 6b. GroundedSAM2 (primary)

**GroundingDINO** is an open-vocabulary detector. Given a text query, it returns
bounding boxes for every instance of every mentioned category:

```
Text query: dynamic per-image from RAM++ (see 6a above)

Output: [(bbox1, "sand", conf=0.84), (bbox2, "sky", conf=0.91), ...]

Thresholds (lowered from defaults to catch weak/small detections):
  grounding_dino_box_thresh = 0.15   ← minimum detection confidence
  grounding_dino_text_thresh = 0.15  ← minimum token confidence
```

**SAM2ImagePredictor** then generates a precise pixel mask for each detected box:
```
For each (bbox, label, conf) from GroundingDINO:
    predictor.set_image(img_rgb)
    masks, scores, logits = predictor.predict(box=bbox, multimask_output=True)
    best_mask = masks[argmax(scores)]
    → amg_entry = {segmentation: best_mask, label: label, gdino_conf: conf}
```

This gives **one clean object-level mask per entity** — much better than the
generic AMG grid which produces part-level masks.

### 6c. SAM2 AMG (secondary, `run_both_segmentors = True`)

SAM2's Automatic Mask Generator places a dense grid of points across the image
and generates masks for every stable region it finds:

```
Points per side: 32  → 32×32 = 1024 seed points
Predicted IoU threshold: 0.80    ← permissive to get candidates
Stability score threshold: 0.92
Min mask area: 100px
```

AMG catches what GDINO misses: texture regions, small objects (buttons, coins),
background sky/floor/wall, part-level anatomy (shirt, collar, sole of shoe).

### 6d. IoU Deduplication

```python
# From process_image() in scene_understanding.py:
iou_thresh = config.run_both_segmentors_iou_dedup  # 0.7

for amg_m in amg_only_masks:
    seg = np.asarray(amg_m["segmentation"]) > 0
    duplicate = False
    for gdino_mask_bin in gdino_masks:
        inter = np.logical_and(seg, gdino_mask_bin).sum()
        union = np.logical_or(seg, gdino_mask_bin).sum()
        if union > 0 and inter / union >= iou_thresh:
            duplicate = True
            break
    if not duplicate:
        amg_m["source_model"] = "SAM2_AMG"
        extra.append(amg_m)

amg_masks = gdino_masks + extra
```

**IoU formula:**
```
IoU(A, B) = |A ∩ B| / |A ∪ B|

where |·| = number of True pixels in the boolean mask
```

An AMG mask with IoU > 0.7 against any GDINO mask is dropped — the GDINO-prompted
mask is more semantically accurate for that object. The remaining AMG masks
(IoU ≤ 0.7 with all GDINO masks) are new information and kept.

---

## 7. Stage 4 — Per-Object Depth Stats and 3D Coordinates

This is the core computation, implemented in `_mask_depth_stats_and_3d()`.

### Step i — Adaptive Erosion (`_adaptive_erosion_kernel`)

**Why erosion:** Mask edges overlap background pixels. The depth model at a boundary
pixel produces a blend of foreground and background depth — typically too high for
the foreground object. Eroding the mask inward removes these boundary pixels.

**Why adaptive:** A fixed kernel of 5px destroys thin objects (a 6px-wide pole
disappears after erosion). The kernel is sized to the object:

```
min_dim = min(mask_height, mask_width)

kernel_size:
  0   if min_dim < 15px    → too thin to erode safely; skip
  1   if min_dim < 30px    → minimal erosion
  2   if min_dim < 60px    → moderate erosion
  config.mask_erosion_kernel_size  otherwise  (default: 5)

Applied only if mask has > 4 * kernel² pixels (won't disappear after erosion):
  eroded = cv2.erode(mask_bin.astype(uint8), ones(kernel, kernel), iterations=1)
  if eroded.sum() > 0: mask_bin = eroded
```

When `use_erosion=False` (the no-erosion comparison run), this entire block is skipped.

### Step ii — Extract Depth at Mask Pixels

```python
ys, xs = np.where(mask_bin)            # pixel row, col coordinates of mask
depth_at_mask = metric_depth[ys, xs]   # direct array index → depth values in metres
depth_at_mask = depth_at_mask[np.isfinite(depth_at_mask)]  # remove NaN/Inf
```

This is a direct numpy index into the H×W depth array using the mask coordinates.
Result: 1D array of N depth measurements, one per mask pixel, in metres.

### Step iii — Sigma-Clipping (Outlier Rejection)

**Why:** Transparent objects, reflective surfaces, and partially occluded objects
produce bimodal depth distributions. The background mode must be rejected.

**From `scene_understanding.py:1763–1771`:**
```python
sigma = self.depth_outlier_sigma   # default: 2.0
if sigma > 0 and depth_at_mask.size >= 10:
    mean_d = float(np.mean(depth_at_mask))
    std_d  = float(np.std(depth_at_mask))
    if std_d > 1e-6:
        inlier = np.abs(depth_at_mask - mean_d) < sigma * std_d
        if inlier.sum() >= 5:
            depth_at_mask = depth_at_mask[inlier]
```

**Formula:**
```
μ = mean(depth_at_mask)
σ = std(depth_at_mask)
keep pixel i  iff  |depth_i - μ| < outlier_sigma · σ

With outlier_sigma = 2.0: keeps all pixels within 2 standard deviations of the mean.
~95.4% of a Gaussian distribution is within 2σ — outliers are the tail beyond.
```

Example: glass vase, 250 pixels:
- 200 pixels at ~0.8m (vase surface), 50 at ~2.5m (background through glass)
- μ = 1.04m, σ = 0.42m
- Threshold: |d - 1.04| < 2.0 × 0.42 = 0.84m → keep [0.2m, 1.88m]
- 2.5m pixels: |2.5 - 1.04| = 1.46 > 0.84 → rejected

### Step iv — Transparency Detection

**From `scene_understanding.py:1780–1791`:**
```python
kernel_5 = np.ones((5, 5), np.uint8)
dilated    = cv2.dilate(mask_bin.astype(uint8), kernel_5) > 0
border_ring = dilated & ~mask_bin          # pixels just outside the mask

border_depths = metric_depth[border_ring]   # background depth samples
mask_mean     = mean(depth_at_mask)
border_mean   = mean(border_depths)
depth_separation = |mask_mean - border_mean|

possibly_transparent = depth_separation < depth_transparency_threshold  # 0.15m
```

**Logic:** A solid opaque object at 1.5m should have background at 3m+ → separation
= 1.5m >> 0.15m → not transparent. A glass window at 2m with wall behind at 2.1m
→ separation = 0.1m < 0.15m → flagged `possibly_transparent = True`.

This flag tells downstream code the depth value is unreliable for this object.

### Step v — Depth-Weighted Centroid

**From `scene_understanding.py:1798–1807`:**
```python
weights = 1.0 / (depth_at_mask + 1e-6)    # closer = higher weight
w_sum   = weights.sum()
cy_f = np.sum(ys_f * weights) / w_sum      # weighted centroid y (row)
cx_f = np.sum(xs_f * weights) / w_sum      # weighted centroid x (col)

# Snap to nearest real mask pixel (avoids landing in a hole)
dist2 = (ys_f - cy_f)² + (xs_f - cx_f)²
anchor = argmin(dist2)
cx, cy = xs_f[anchor], ys_f[anchor]
```

**Formula:**
```
w_i = 1 / (d_i + ε)      where d_i = depth at pixel i, ε = 1e-6

cx = Σ(w_i · x_i) / Σ(w_i)
cy = Σ(w_i · y_i) / Σ(w_i)
```

**Why depth-weighted:** A pure geometric centroid of an asymmetric object lands
at the visual centre, which may be inside an occluded region. The depth-weighted
centroid is pulled toward the nearest visible surface — more physically meaningful
for 3D anchoring.

### Step vi — z_val via Histogram Mode

**From `scene_understanding.py:1810–1826`:**
```python
# Inner circle: pixels within sqrt(area * central_frac / π) radius of centroid
area   = float(mask_bin.sum())
radius = np.sqrt(area * central_frac / np.pi)   # central_frac = 0.5
inner_mask = dist2 <= radius²
inner_depths = depth_at_mask[inner_mask]

# Histogram over inner depths
n_bins = max(10, min(100, inner_depths.size // 5))
hist, edges = np.histogram(inner_depths, bins=n_bins)
peak_bin = argmax(hist)
z_val = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
```

**Why histogram mode instead of mean:**
- Mean is biased by outliers even after sigma-clipping
- For a curved surface (bowl, sphere), near edge pixels are farther than centre pixels
- Mode picks the depth of the largest coherent surface layer → most representative
  single depth for the object

**Inner circle rationale:** The outer pixels of a mask are most likely to be:
- boundary bleed (even after erosion)
- occluded partial pixels at depth transitions

Using only the inner 50% of the mask area (by equivalent circle radius) concentrates
on the central, most reliable depth measurements.

### Step vii — Back-Projection to 3D

**From `_back_project()` in `scene_understanding.py`:**
```
Given:
  (cx, cy) = centroid pixel coordinates (col, row)
  z_val    = representative depth in metres
  K        = {fx, fy, cx_principal, cy_principal}

3D camera-space coordinates:
  X = (cx - K["cx"]) * z_val / K["fx"]     ← metres, horizontal (right = positive)
  Y = (cy - K["cy"]) * z_val / K["fy"]     ← metres, vertical (down = positive)
  Z = z_val                                  ← metres, depth from camera

coordinates_3d = {"x": X, "y": Y, "z": Z}
```

**Derivation:** The pinhole camera model projects 3D point (X, Y, Z) to pixel (u, v):
```
u = fx · (X/Z) + cx_principal
v = fy · (Y/Z) + cy_principal
```
Inverting:
```
X = (u - cx_principal) · Z / fx
Y = (v - cy_principal) · Z / fy
```

This is exact for a perfect pinhole camera with no distortion (after undistortion
in Stage 0, this assumption holds to < 0.5px residual error).

### Dual Erosion Comparison

When `depth_erosion_comparison = True`, `_mask_depth_stats_and_3d` is called twice:

```python
# Call 1: with erosion (standard)
depth_stats, coords_3d, centroid = self._mask_depth_stats_and_3d(
    metric_depth, K, mask, detection, use_erosion=True)

# Call 2: without erosion (comparison)
depth_stats_ne, coords_3d_ne, centroid_ne = self._mask_depth_stats_and_3d(
    metric_depth, K, mask, detection, use_erosion=False)
```

Both sets are stored in the scene JSON under separate keys:
```
depth_stats              ← with adaptive erosion
depth_stats_no_erosion   ← without erosion
coordinates_3d           ← back-projected from eroded z_val
coordinates_3d_no_erosion← back-projected from raw z_val
mask_centroid_2d         ← eroded centroid pixel
mask_centroid_2d_no_erosion ← raw centroid pixel
```

---

## 8. Stage 4b — Object Labelling (`_label_mask`)

**Three-priority label chain** (GRiT and YOLO removed as dead code — see Section 14):

### Priority 1: GroundingDINO label
```python
gdino_label = amg_entry.get("label")
gdino_conf  = amg_entry.get("gdino_conf", 0.0)
if gdino_label and gdino_label != "object":
    return {"label": gdino_label, "conf": gdino_conf, "source_model": "GroundingDINO"}
```
Used directly when mask came from GroundedSAM2. GDINO labels are the most semantically
accurate because they come from the text-prompted detection. Note: the confidence
threshold is intentionally omitted — since the GDINO query is now scene-specific
(from RAM++), even low-confidence detections are relevant.

### Priority 2: Florence-2 on masked crop (`label_crop`)

`label_crop` runs a **two-step** process on a tight crop of the object:

**Step 2a — `<MORE_DETAILED_CAPTION>` (primary):**
```python
cap_result = florence2._run_task("<MORE_DETAILED_CAPTION>", pil_crop)
caption = cap_result.get("<MORE_DETAILED_CAPTION>", "")
# e.g. "a wooden dining chair with a padded seat and curved back legs"
label = florence2._extract_label_from_caption(caption)
# skips stopwords → first meaningful noun → "chair"
```

`_extract_label_from_caption` skips ~50 stopwords (articles, prepositions, common
adjectives like "wooden", "large", colour words) and returns the first remaining
noun. The full caption is stored in `sources.Florence2.caption` in the scene JSON.

**Step 2b — `<OD>` fallback (when caption gave "object"):**
```python
od_result = florence2._run_task("<OD>", pil_crop)
# Returns: {labels: ["chair", "table"], bboxes: [[x1,y1,x2,y2], ...]}
# Pick label from largest bbox area → dominant object in crop
best_label = max(zip(labels, bboxes), key=lambda lb: bbox_area(lb[1]))[0]
```

**Why mean-fill background:** Out-of-mask pixels are replaced with the image mean
colour (not zero/black). This preserves natural brightness/colour statistics so
Florence-2 behaves as trained.

**Why largest bbox:** A crop of a "person" might detect ["person", "shirt", "hand"].
The person bbox is largest → correct dominant label.

### Priority 3: RAM++ per-crop (`label_crop`)
```python
rampp_result = rampp.label_crop(crop_filled)  # crop is BGR
# Returns: {label: "chair", conf: 0.70, caption: "chair | table | ...", tags: [...]}
label = rampp_result["label"]   # first tag = most salient detected object
```
RAM++ runs its Swin-L tagger on the masked crop. The first tag (highest salience)
is used as the label. Unlike whole-image tagging (which feeds the GDINO query),
this per-crop call gives RAM++ a chance to label AMG masks that GDINO and Florence-2
both missed.

If all three priorities return "object", the final label is `"object"` with
`source_model = "fallback"`.

---

## 9. Stage 5 — Relation Prediction (Pix2SG + Florence-2)

### Layer 1: Spatial Scaffold

For every ordered pair of objects (A, B), geometric relations are derived from
3D coordinates and mask positions:

```
Depth-based:
  if Z_A < near_threshold (1.0m): A is "near"
  if Z_A > far_threshold (3.0m):  A is "far"
  if Z_A < Z_B - 0.5m: A is "in_front_of" B

Positional:
  bbox centroid comparison → "left_of", "right_of", "above", "below"

Mask overlap:
  if mask_IoU(A, B) > pix2sg_mask_overlap_thresh (0.05):
    if A is smaller and inside B bbox: A "on" B
    → "contains", "on_top_of", "overlapping"
```

### Layer 2: Florence-2 Semantic Relations (`_enrich_with_florence2`)

**From `scene_understanding.py:542–603`:**

```python
for i in range(n):                      # subject
    for j in range(n):                  # object
        if i == j: continue

        # Mask IoU check (from code lines 582-584):
        inter = np.logical_and(sub_m, obj_m).sum()
        union = np.logical_or(sub_m, obj_m).sum()
        iou   = inter / union
        if iou < relation_min_mask_overlap:   # 0.02
            continue

        pred = florence2.predict_relation(
            image_bgr, sub_mask, obj_mask, sub_label, obj_label)
```

**`predict_relation` method:**

```python
# Step 1: Union bounding box of both masks (10px padding)
union_m = sub_m | obj_m
ys, xs  = np.where(union_m)
x1 = max(0, int(xs.min()) - 10)
x2 = min(W, int(xs.max()) + 10)
y1 = max(0, int(ys.min()) - 10)
y2 = min(H, int(ys.max()) + 10)
crop = full_img_bgr[y1:y2, x1:x2]

# Step 2: Color overlay (alpha=0.45)
crop_rgb = cv2.cvtColor(crop, BGR2RGB).astype(float32)
crop_rgb[sub_crop, 0] = clip(crop_rgb[sub_crop, 0] * 0.55 + 255 * 0.45, 0, 255)  # RED channel
crop_rgb[obj_crop, 2] = clip(crop_rgb[obj_crop, 2] * 0.55 + 255 * 0.45, 0, 255)  # BLUE channel

# Step 3: Florence-2 <MORE_DETAILED_CAPTION> (standalone task token — no appended text)
result = florence2._run_task("<MORE_DETAILED_CAPTION>", pil_crop)
raw    = result.get("<MORE_DETAILED_CAPTION>", "")
# → "The red person is sitting on the blue chair near a wooden table."

# Step 4: Map to canonical predicate via phrase lookup table
predicate = florence2._parse_relation_phrase(raw.lower())
# "sitting on" → "on",  "next to" → "is_next_to",  "holding" → "holds", etc.
# Returns None if no phrase from the lookup table matched
```

**Important:** The task token `<MORE_DETAILED_CAPTION>` must be the **only** text
passed to `_run_task` — no appended prompt text. transformers 5.x enforces this
strictly and raises `"task token should be the only token in the text"` otherwise.
Earlier versions used `<CAPTION>` with an appended prompt; that approach no longer
works and was replaced.

**Why color overlay works:** Florence-2 understands natural language colour references
("the red X", "the blue Y"). The overlay makes the two regions unambiguous within
the crop, so the caption naturally describes the relationship between exactly those
two objects even when other objects appear in the same crop.

**`_parse_relation_phrase` lookup table (abbreviated):**
```
"on top of" / "resting on" / "sitting on" / "standing on"  → "on"
"under" / "below" / "beneath"                               → "under"
"next to" / "beside" / "adjacent"                           → "is_next_to"
"in front of"                                               → "in_front_of"
"behind"                                                    → "behind"
"inside" / "within" / "contained in"                        → "inside_of"
"holding" / "carrying" / "gripping"                         → "holds"
"wearing" / "dressed in"                                    → "wears"
"riding" / "mounted on"                                     → "rides"
... (20 entries total)
```
Returns `None` if no phrase matched — that pair gets no Florence-2 relation entry.

**Triplet structure:**
```json
{
  "sub": "person",
  "pred": "sitting_on",
  "obj": "chair",
  "sub_id": "obj_0_GroundedSAM2",
  "obj_id": "obj_3_GroundedSAM2",
  "score": 0.75,
  "source_layer": "florence2"
}
```

### Relation Attachment (`_attach_relations_by_triplets`)

Triplets are attached to `objects_3d` entries by ID first, then label fallback:

```python
id_to_obj = {str(o["id"]): o for o in objects_3d}

for triplet in triplets:
    # Match subject
    source_obj = id_to_obj.get(str(triplet["sub_id"]))
    if source_obj is None:
        source_obj = find_by_label(triplet["sub"])   # partial string match

    # Match target
    target = id_to_obj.get(str(triplet["obj_id"]))
    if target is None:
        target = find_by_label(triplet["obj"])
    if target is None:
        target_id = f"external_{triplet['obj']}"     # object not in scene graph

    # Build relation entry (includes target_label and target_caption)
    relation_entry = {
        "predicate":      triplet["pred"],
        "target_id":      target_id,
        "target_label":   target_obj["label"],
        "target_caption": target_obj["sources"][...]["caption"],
    }
    source_obj["sources"][source_name]["relations"].append(relation_entry)
```

---

## 10. Stage 6 — Serialisation

Before `json.dump`, `_sam2_mask_array` (a numpy bool array, not JSON-serialisable)
is stripped from every object entry. All other fields are Python native types
(float, int, str, list, dict) and serialise cleanly.

Output: `{output_dir}/scene_graph/{stem}_scene.json`

---

## 11. Stage 7 — Visualisation

Two labelled overlay images are produced per image. Both are generated **before**
`_sam2_mask_array` is stripped from `objects_3d` (they need the numpy masks).

**`_sam2_segmentation.png`** — solid mask fills + contours + labels:
- Each mask gets a deterministic colour from `_mask_colour(seed=mask_index)`:
  `np.random.RandomState(seed).randint(60, 230, 3)` → stable colour per object
- Solid fill drawn at full opacity, then contour outline in same colour
- Label text on dark (0,0,0,180 alpha) pill background at mask centroid
- Font scale: `sqrt(mask_area) / 250`, clamped to [0.35, 0.70]

**`_sam2_tinted_overlay.png`** — original photo with semi-transparent tints + labels:
- Original image copied, then each mask region blended: `α=0.45` tint over photo
- Same label rendering as above (pill background, centroid-anchored)

**`_depth_map.png`** — INFERNO false-colour depth map (cv2.COLORMAP_INFERNO).

**`_3d_viz.png`** — 3D coordinate overlay with centroid positions annotated.

No bounding box rectangles are drawn. No `[M]` tag prefix. No depth value in the label.

---

## 12. All Formulas in One Place

```
─── CLIP scene classification ──────────────────────────────────
logits  = CLIP_similarity(image, [text_1 ... text_6])     shape: (6,)
probs   = softmax(logits)
indoor_score  = mean(probs[0:3])
outdoor_score = mean(probs[3:6])
scene_type = argmax([indoor_score, outdoor_score])

─── FOV intrinsics fallback ────────────────────────────────────
fx = W / (2 · tan(FOV_deg · π / 360))
fy = fx
cx = W/2,  cy = H/2

─── Lens distortion correction ────────────────────────────────
r² = x² + y²
x_corr = x(1 + k1·r² + k2·r⁴) + 2·p1·x·y + p2·(r² + 2x²)
y_corr = y(1 + k1·r² + k2·r⁴) + p1·(r² + 2y²) + 2·p2·x·y

─── Mask IoU (dedup + overlap + relations) ──────────────────────
IoU(A, B) = |A ∩ B| / |A ∪ B|         (pixel-level boolean masks)

─── Adaptive erosion kernel ────────────────────────────────────
min_dim = min(mask_h, mask_w)
k = 0 if min_dim < 15
  = 1 if min_dim < 30
  = 2 if min_dim < 60
  = config.mask_erosion_kernel_size  otherwise

─── Depth extraction ───────────────────────────────────────────
depth_at_mask = metric_depth[ys, xs]       direct numpy index

─── Sigma clipping ─────────────────────────────────────────────
μ = mean(depth_at_mask)
σ = std(depth_at_mask)
keep_i iff |depth_i - μ| < outlier_sigma · σ      (outlier_sigma = 2.0)

─── Transparency detection ──────────────────────────────────────
border_ring = dilate(mask, 5×5) AND NOT mask
depth_sep   = |mean(depth_at_mask) - mean(metric_depth[border_ring])|
possibly_transparent = depth_sep < depth_transparency_threshold   (0.15m)

─── Depth-weighted centroid ─────────────────────────────────────
w_i = 1 / (depth_i + 1e-6)
cx  = Σ(w_i · x_i) / Σ(w_i)
cy  = Σ(w_i · y_i) / Σ(w_i)

─── Inner circle radius ────────────────────────────────────────
area   = number of True pixels in mask
radius = sqrt(area · central_frac / π)     (central_frac = 0.5)
inner  = pixels where (x-cx)² + (y-cy)² ≤ radius²

─── Histogram mode (z_val) ──────────────────────────────────────
n_bins = clamp(inner.size // 5,  min=10, max=100)
hist, edges = histogram(inner_depths, bins=n_bins)
peak   = argmax(hist)
z_val  = (edges[peak] + edges[peak+1]) / 2

─── Back-projection ─────────────────────────────────────────────
X = (u - cx_principal) · z_val / fx
Y = (v - cy_principal) · z_val / fy
Z = z_val

─── RAM++ dynamic GDINO query ───────────────────────────────────
tags  = ram_plus.inference(transform(full_image), model)   → pipe-separated string
tags_list = parse(tags)[:max_tags]                         → list of nouns
gdino_query = ". ".join(tags_list) + "."                   → GDINO text input

─── Florence-2 labelling (label_crop) ───────────────────────────
Primary:   caption = Florence2("<MORE_DETAILED_CAPTION>", crop)
           label   = first non-stopword noun in caption
Fallback:  od      = Florence2("<OD>", crop)
           label   = label from bbox with largest area in od["<OD>"]["bboxes"]

─── Florence-2 color overlay (relation prediction) ─────────────
# alpha = 0.45
crop_rgb[sub_pixels, 0] = clip(crop_rgb[sub_pixels, 0] · 0.55 + 255 · 0.45, 0, 255)  # RED
crop_rgb[obj_pixels, 2] = clip(crop_rgb[obj_pixels, 2] · 0.55 + 255 · 0.45, 0, 255)  # BLUE
raw_caption = Florence2("<MORE_DETAILED_CAPTION>", crop)   # standalone task token only
predicate   = _parse_relation_phrase(raw_caption)          # phrase → canonical predicate

─── 16-bit depth PNG ────────────────────────────────────────────
max_range = 20.0m (indoor) or 80.0m (outdoor)
depth_16  = clip(depth / max_range · 65535,  0, 65535).astype(uint16)

─── EMA temporal depth filtering (multi-frame, optional) ────────
filtered[t] = α · depth[t] + (1-α) · filtered[t-1]     α = 0.6
```

---

## 13. Fixes Applied (5.1–5.8)

### Fix 5.1 — Correct depth key + CLIP model selection

**Bug:** `out["depth"]` returns a PIL uint8 image (0–255), not metres.
**Fix:** Use `out["predicted_depth"]` (raw float32 tensor in metres).

**Bug:** Single depth model used regardless of scene type → wrong scale.
**Fix:** CLIP classifies indoor/outdoor, matching metric model loads.
CLIP unloads immediately after to free VRAM (~340MB).

**Bug:** `_resize_to_target` was called inside `infer()` forcing a 512×512 square
resize before `process_image` resized again → double resize, destroyed aspect ratio.
**Fix:** Removed the resize from `infer()`; `process_image` does one correct resize.

### Fix 5.2 — Camera calibration (OpenCV checkerboard)

Priority system: calibration JSON > explicit fx/fy > FOV estimate.
`tools/calibrate_camera.py` produces calibration JSON with < 0.5% fx/fy error.
Distortion coefficients (k1, k2, p1, p2) applied via `cv2.undistort` before any
depth or mask processing.

### Fix 5.3 — GroundedSAM2 replaces SAM2 AMG as primary segmentor

GDINO text-prompted detection → one clean object-level mask per entity.
SAM2 AMG kept as secondary (parts, small objects, background).
`grounded_sam2_fallback_to_amg = True` as safety net.

### Fix 5.4 — Florence-2 replaces GRiT heuristic as primary labeller; GRiT + YOLO removed

GRiT's "last word of caption" heuristic failed ~30% of time. GRiT and YOLOv8x-cls were
removed entirely from `scene_understanding.py` (see Section 14 for full dead-code list).
Florence-2 `label_crop` replaced them with a two-step approach:
1. `<MORE_DETAILED_CAPTION>` → full sentence → noun extracted by `_extract_label_from_caption`
   (skips ~50 stopwords including articles, prepositions, adjectives, colour words)
2. `<OD>` fallback if caption extraction yielded "object" → largest bbox label

The full caption is stored in `sources.Florence2.caption` in the scene JSON for reference.
The label chain is now **GDINO → Florence-2 → RAM++** (three priorities, no GRiT/YOLO).

### Fix 5.5 — Post-hoc mask filters (now fully disabled)

Original strict filters removed small objects, background, low-confidence masks.
All thresholds set to 0 / 1.0 → every mask retained.
Re-enable selectively in config.py for production use.

### Fix 5.6 — Florence-2 semantic relation prediction

For overlapping mask pairs (IoU > 0.02):
1. Crop to union bbox
2. RED tint on subject, BLUE tint on object
3. Florence-2 `<MORE_DETAILED_CAPTION>` (standalone) → `_parse_relation_phrase` → canonical predicate
4. Stored alongside spatial scaffold relations

### Fix 5.7 — Depth accuracy: adaptive erosion + sigma-clipping + transparency

Three independent improvements to per-mask depth quality:
1. **Adaptive erosion** — kernel scales to min(mask_h, mask_w) to protect thin objects
2. **Sigma-clipping** — reject pixels beyond 2σ from mask mean depth
3. **Transparency detection** — border ring depth comparison → `possibly_transparent` flag

`use_erosion` parameter added to `_mask_depth_stats_and_3d` for dual-run comparison
(no global state mutation, clean parameter interface).

### Fix 5.8 — RAM++ dynamic GDINO vocabulary + improved labelling

**Problem:** GDINO used a static generic category list for every image, causing missed
detections for scene-specific objects and false positives for irrelevant categories.
Florence-2 `predict_relation` used `<CAPTION>` with appended text — broken in
transformers 5.x. Florence-2 `label_crop` only used `<OD>`, missing the richer
`<MORE_DETAILED_CAPTION>` output.

**Fixes:**
1. **RAM++ whole-image tagging** — runs before GroundedSAM2; builds a per-image
   GDINO query from scene-specific tags. `update_text_query()` added to
   `GroundedSAM2Wrapper`.
2. **`label_crop` two-step** — primary: `<MORE_DETAILED_CAPTION>` → noun extraction
   with `_extract_label_from_caption` (~50-word stoplist); fallback: `<OD>` largest-bbox.
3. **`predict_relation` task token** — changed from `<CAPTION>` + appended text to
   `<MORE_DETAILED_CAPTION>` standalone. Fixes `"task token should be the only token"`
   error in transformers 5.x; relation prediction now executes correctly.
4. **`all_tied_weights_keys` property setter** — RAM++ shim added a read-only property
   to `PreTrainedModel` that blocked Florence-2 loading after RAM++. Fixed by adding
   a setter that stores override via `self.__dict__`.
5. **RAM++ per-crop labelling** — added as Priority 3 in the label chain, between
   Florence-2 and GRiT.
6. **Output cleanup** — per-object mask PNGs removed; output reduced to 7 flat files
   per image; labelled overlay images (`_sam2_segmentation.png`,
   `_sam2_tinted_overlay.png`) added with pill-background labels at mask centroids.

---

## 14. Dead Code Removed

Removed from `scene_understanding.py` (~900 lines, ~27% of original file):

| Class / Code | Lines | Reason |
|---|---|---|
| `FasterRCNNWrapper` | ~120 | Superseded by GroundedSAM2 |
| `DETRWrapper` | ~80 | Superseded by GroundingDINO |
| `Pix2SeqWrapper` / OWLViT | ~100 | Checkpoint unavailable; never ran |
| `SGSGWrapper` (SGTR) | ~330 | SGTR repo removed; no checkpoint |
| SGTR path setup | ~20 | Dead without SGTR |
| `_OIV6_PREDICATES` dict | ~70 | Only used by SGSGWrapper |
| `_save_model_output` | ~20 | Never called |
| `sgtr_stats` references | ~5 | Caused NameError post-removal |
| `GRiTWrapper` | ~80 | Replaced by Florence-2 (Fix 5.4); GRiT also removed from setup.sh |
| `YOLOWrapper` | ~60 | Replaced by Florence-2 + RAM++ (Fix 5.4/5.8) |

GRiT and its dependency chain (detectron2, CenterNet2) have been fully removed from
the codebase — not just from `scene_understanding.py` but also from `setup.sh` and
`requirements.txt`. The label chain is now **GDINO → Florence-2 → RAM++** exclusively.

---

## 15. Output File Structure

All outputs land flat inside `output_scene/scene_graph/`. Exactly **7 files per image**,
no subdirectories, no per-object mask PNGs.

```
output_scene/
├── depth/
│   └── {stem}_depth_metric.npy        ← float32 H×W metres (always saved)
└── scene_graph/
    ├── {stem}_scene.json               ← full scene graph (metadata + objects)
    ├── {stem}_depth_map.png            ← INFERNO false-colour depth map
    ├── {stem}_3d_viz.png               ← 3D coordinate overlay
    ├── {stem}_depth_mask_mapping_A.png ← depth/mask matching visualisation (mode A)
    ├── {stem}_depth_mask_mapping_B.png ← depth/mask matching visualisation (mode B)
    ├── {stem}_sam2_segmentation.png    ← solid-colour mask fills + contours + labels
    └── {stem}_sam2_tinted_overlay.png  ← original photo with α=0.45 tinted overlays + labels
```

**Per-object mask PNGs are not saved** (`save_per_object_masks = False` hardcoded).
`mask_path` and `depth_map_path` in the scene JSON are always `null`.

**The labelled overlay images** (`_sam2_segmentation.png`, `_sam2_tinted_overlay.png`)
are generated **before** `_sam2_mask_array` is stripped — they use the transient numpy
mask arrays still present on `objects_3d` entries at that point. Each label is drawn
as white text on a dark pill background, centred at the mask centroid, with font
size scaled to `sqrt(mask_area) / 250` (clamped 0.35–0.70).

**Scene JSON — top-level structure:**
```json
{
  "metadata": {
    "timestamp": "2026-03-09 14:22:01",
    "segmentor": "SAM2",
    "intrinsics": {"fx": 623.5, "fy": 623.5, "cx": 360.0, "cy": 640.0},
    "models": ["DepthAnythingV2-Metric-Indoor", "GroundingDINO", "SAM2", "Florence-2", "RAM++"],
    "rampp_tags": ["beach", "cloudy", "coast", "footprint", "sea", "sand", "shoreline", "sky"],
    "gdino_query_used": "beach. cloudy. coast. footprint. sea. sand. shoreline. sky.",
    "relation_sources": { "...": "..." },
    "relation_debug": { "...": "..." },
    "depth_map": "scene_graph/{stem}_depth_map.png",
    "segmentation_image": "scene_graph/{stem}_segmentation.png",
    "sam2_segmentation_image": "scene_graph/{stem}_sam2_segmentation.png",
    "sam2_tinted_overlay_image": "scene_graph/{stem}_sam2_tinted_overlay.png"
  },
  "objects": [ ... ]
}
```

**Per-object entry in scene JSON:**
```json
{
  "id": "obj_0_GroundedSAM2",
  "label": "person",
  "confidence": 0.87,
  "conf": 0.87,
  "segmentor": "GroundedSAM2",
  "bbox": [359, 4, 572, 319],
  "coordinates_3d": {"x": -0.23, "y": 0.41, "z": 2.34},
  "coordinates_3d_no_erosion": {"x": -0.25, "y": 0.43, "z": 2.41},
  "depth_stats": {
    "min": 2.01, "max": 2.89, "mean": 2.38, "median": 2.35, "std": 0.18,
    "num_pixels": 4210, "z_val": 2.34, "z_val_pixels": 2310,
    "possibly_transparent": false,
    "depth_separation_from_background": 0.82
  },
  "depth_stats_no_erosion": { "...": "..." },
  "mask_centroid_2d": [312, 445],
  "mask_centroid_2d_no_erosion": [315, 448],
  "sam2_mask_index": 0,
  "mask_matched": true,
  "mask_path": null,
  "depth_map_path": null,
  "sources": {
    "GroundedSAM2": {"label": "person", "conf": 0.87, "caption": "person"},
    "Florence2": {
      "label": "person",
      "caption": "a man in a blue shirt standing near a wooden table"
    },
    "RAM++": {
      "label": "person",
      "caption": "person | clothing | shirt",
      "tags": ["person", "clothing", "shirt"]
    },
    "Pix2SG": {
      "relations": [
        {
          "predicate": "is_next_to",
          "target_id": "obj_2_GroundedSAM2",
          "target_label": "table",
          "target_caption": "a brown wooden dining table",
          "score": 0.85
        }
      ]
    }
  }
}
```

---

## 16. Config Reference

All settings in `config.py` → `PreprocessConfig` dataclass.

### Depth
| Field | Default | Meaning |
|---|---|---|
| `depth_model_variant` | `"auto"` | `"auto"` / `"indoor"` / `"outdoor"` |
| `depth_scale_factor` | `1.0` | Always 1.0 for metric models |
| `depth_adaptive_erosion` | `True` | Scale kernel to object size |
| `mask_erosion_kernel_size` | `5` | Max erosion kernel (px) |
| `depth_central_fraction` | `0.5` | Inner fraction for histogram |
| `depth_outlier_sigma` | `2.0` | Sigma-clipping threshold (0 = off) |
| `depth_transparency_check` | `True` | Enable border ring comparison |
| `depth_transparency_threshold` | `0.15` | Metres below which → transparent |
| `depth_erosion_comparison` | `True` | Store both eroded and raw stats |

### Camera
| Field | Default | Meaning |
|---|---|---|
| `camera_calibration_file` | `None` | Path to calibration JSON |
| `apply_undistortion` | `True` | Run cv2.undistort() |
| `camera_fx/fy/cx/cy` | `None` | Explicit intrinsics (priority 2) |
| `camera_fov_degrees` | `60.0` | FOV fallback (priority 3) |

### Segmentation
| Field | Default | Meaning |
|---|---|---|
| `run_both_segmentors` | `True` | GDINO+SAM2 AND SAM2 AMG |
| `run_both_segmentors_iou_dedup` | `0.7` | IoU threshold to drop AMG duplicates |
| `grounding_dino_box_thresh` | `0.15` | Min GDINO detection confidence |
| `grounding_dino_text_thresh` | `0.15` | Min GDINO token confidence |
| `sam2_amg_min_mask_region_area` | `100` | Min AMG mask area (px) |
| `grounded_sam2_fallback_to_amg` | `True` | Use AMG if GDINO fails |

### Filtering (all disabled)
| Field | Default | Meaning |
|---|---|---|
| `sam2_post_filter_min_stability` | `0.0` | 0 = disabled |
| `sam2_post_filter_min_pred_iou` | `0.0` | 0 = disabled |
| `sam2_post_filter_min_area_px` | `0` | 0 = allow all sizes |
| `sam2_post_filter_max_area_fraction` | `1.0` | 1.0 = allow background |

### Relations
| Field | Default | Meaning |
|---|---|---|
| `florence2_relation_enabled` | `True` | Florence-2 semantic relations |
| `relation_min_mask_overlap` | `0.02` | Min IoU to attempt relation |
| `pix2sg_depth_near_threshold` | `1.0` | Near boundary (metres) |
| `pix2sg_depth_far_threshold` | `3.0` | Far boundary (metres) |

### RAM++
| Field | Default | Meaning |
|---|---|---|
| `rampp_enabled` | `True` | Enable RAM++ for GDINO query + per-crop labelling |
| `rampp_checkpoint_path` | `"checkpoints/ram_plus_swin_large_14m.pth"` | Local .pth path |
| `rampp_image_size` | `384` | Input resolution for RAM++ Swin-L |
| `rampp_vit` | `"swin_l"` | Backbone variant (`"swin_l"` or `"swin_b"`) |
| `rampp_default_confidence` | `0.70` | Confidence assigned to RAM++ labels |
| `rampp_max_tags` | `8` | Max tags kept for dynamic GDINO query |

---

## 17. Reproduction

```bash
# 1. Clone repo
git clone https://github.com/omondistanley/citv.git
cd citv

# 2. One-shot setup (clones SAM2, downloads SAM2 checkpoint,
#    builds SAM2_C CUDA extension, installs all deps, pre-downloads
#    all HuggingFace models — Depth Anything V2, CLIP, GDINO, Florence-2)
bash setup.sh

# 3. Run pipeline
python scene_understanding.py --input_dir images --output_dir output_scene

# Optional: camera calibration
python tools/calibrate_camera.py \
    --images path/to/checkerboard_frames/ \
    --pattern 9x6 \
    --square_size 0.025 \
    --out calibration.json
# Then set in config.py: camera_calibration_file = "calibration.json"
```

**Requirements:** Python 3.9–3.11, CUDA GPU ≥ 8GB VRAM, nvcc on PATH, ~20GB disk.

---

## 18. Inference Time and Performance Notes

### Why the pipeline runs long

The pipeline is sequential — each stage waits for the previous one. Five heavyweight
GPU models run back-to-back per image with no shared state:

| Stage | Model | Typical GPU time |
|---|---|---|
| 2 | CLIP + Depth Anything V2 Metric | 5–15 s |
| 3 | RAM++ + GroundingDINO + SAM2 | 10–30 s |
| 4 | Florence-2 labelling (×N objects) | **dominant — see below** |
| 5 | Pix2SG + Florence-2 relations (×M pairs) | **dominant — see below** |

Models also load and unload per image. Florence-2-large (~1.5 GB) loads fresh for
Stage 4, stays loaded through Stage 5, then `_unload_labellers()` frees VRAM.
CLIP (~340 MB) loads and unloads within Stage 2 alone.

### Florence-2 is the main bottleneck

Florence-2 is called **once per object** (labelling) and **once per overlapping pair**
(relations). Both loops are sequential — no batching.

**Stage 4 — labelling** (`_label_mask` → `label_crop`):

Every mask that lacks a confident GDINO label triggers `label_crop()`, which makes
**two Florence-2 forward passes**:

```
Pass 1: <MORE_DETAILED_CAPTION>  → noun extraction
Pass 2: <OD>                     → largest-bbox label  (only if pass 1 gave "object")
```

With 50 detected objects, worst case: **100 Florence-2 forward passes**.

**Stage 5 — relation prediction** (`_enrich_with_florence2`):

Nested O(N²) loop over all ordered object pairs. Every pair whose mask IoU exceeds
`relation_min_mask_overlap` (default: 0.02 — very permissive) triggers one
`predict_relation()` call = one `<MORE_DETAILED_CAPTION>` forward pass.

With 50 objects and 10% overlap rate: 50×49 = 2450 pairs checked → ~245 relation calls.

**Total per image:**
```
Florence-2 calls ≈ 2 × (objects without GDINO label) + overlapping_pairs
                 ≈ 50 – 350 forward passes per image
                   × ~200–800 ms each on GPU
                 = 10 s – 5 min per image for Florence-2 alone
```

### Tuning for speed

To reduce inference time without losing core capability:

| Change | Config field | Impact |
|---|---|---|
| Raise relation overlap threshold | `relation_min_mask_overlap = 0.15` | Largest single reduction: cuts Stage 5 calls by ~5× |
| Use Florence-2-base | `florence2_model = "microsoft/Florence-2-base"` | ~4× faster per call, minor quality drop |
| Disable dual segmentor | `run_both_segmentors = False` | Halves Stage 3 time |
| Lower GDINO thresholds to get more labels | `grounding_dino_box_thresh = 0.25` | More GDINO labels → fewer Florence-2 `label_crop` calls |
| Disable dual erosion comparison | `depth_erosion_comparison = False` | Halves Stage 4 depth computation |

The most impactful single change is raising `relation_min_mask_overlap` from 0.02 to 0.15,
which eliminates the majority of weak/distant-object relation calls while keeping all
meaningful touching/overlapping pairs.
