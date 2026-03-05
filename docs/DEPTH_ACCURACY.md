# Depth Accuracy — Per-Mask Depth Analysis

## Overview

Raw depth values at mask pixels are not directly usable as object depth estimates. Three sources
of contamination exist:

1. **Boundary bleed** — the depth network blends foreground and background depth at object edges,
   producing pixels with intermediate "halo" depth values inside the mask boundary.
2. **Background bleed through transparency** — for glass, water, or thin plastic, the depth
   network sees the background through the object, producing distant background depth values
   inside the mask.
3. **Outliers from occlusion** — partially occluded masks include background pixels whose depth
   is far from the object's true depth.

`_mask_depth_stats_and_3d()` in `scene_understanding.py` addresses all three in sequence.

---

## Stage 1 — Adaptive Mask Erosion

**Purpose**: remove boundary pixels where fg+bg depth values are blended.

A fixed erosion kernel fails for thin objects: a 5px kernel destroys the entire mask of a 3px-wide
pole. The kernel is therefore scaled to the object's narrowest dimension:

```
ys, xs   = np.where(mask_bin)
bbox_h   = ys.max() - ys.min() + 1
bbox_w   = xs.max() - xs.min() + 1
min_dim  = min(bbox_h, bbox_w)

kernel = 0   if min_dim < 15     # very thin object — skip erosion
         1   if min_dim < 40
         2   if min_dim < 80
         3   if min_dim < 150
         config.mask_erosion_kernel_size   otherwise   (default 5)
```

Erosion is skipped entirely (`kernel = 0`) when the eroded mask would have fewer pixels than
`4 × kernel²` — preventing complete destruction of very small masks.

Controlled by `use_erosion: bool` parameter:
- `use_erosion=True` (default) — applies adaptive erosion; stored in `depth_stats`
- `use_erosion=False` — skips erosion entirely; stored in `depth_stats_no_erosion`

**Config**: `depth_adaptive_erosion = True`, `mask_erosion_kernel_size = 5`

---

## Stage 2 — Sigma-Clipping (Outlier Rejection)

**Purpose**: reject depth pixels that belong to the background, not the object surface.

```
μ   = mean(depth_pixels_in_mask)
σ   = std(depth_pixels_in_mask)

keep pixel i  iff  |depth_i − μ| < outlier_sigma × σ
```

**Example**: a glass vase mask contains 200 pixels at ~0.8 m (vase surface) and 50 pixels at
~2.5 m (background seen through glass).

```
μ ≈ 1.0 m,  σ ≈ 0.4 m,  outlier_sigma = 2.0
threshold = 1.0 ± 2.0×0.4 = [0.2, 1.8]
→ pixels at 2.5 m are rejected  (|2.5 - 1.0| = 1.5 > 0.8)
```

Clipping is only applied when:
- `outlier_sigma > 0` (0 disables it)
- at least 10 depth pixels are available
- the retained inlier set has ≥ 5 pixels

**Config**: `depth_outlier_sigma = 2.0`

---

## Stage 3 — Transparency Detection

**Purpose**: flag objects whose depth is indistinguishable from the surrounding background,
indicating transparent material (glass, water, clear plastic).

```
border_ring = dilate(mask, 5×5 kernel) \ mask        # 5px ring around the mask

mask_mean   = mean(depth_pixels[mask])
border_mean = mean(depth_pixels[border_ring])

separation  = |mask_mean − border_mean|               # in metres
possibly_transparent = (separation < depth_transparency_threshold)   # default 0.15 m
```

The 5px dilation ensures the border ring samples the immediately surrounding background, not
distant scene content.

Output fields set by this stage:

| Field | Type | Meaning |
|---|---|---|
| `possibly_transparent` | bool | True if separation < threshold |
| `depth_separation_from_background` | float (m) | The computed separation value |

**Config**: `depth_transparency_check = True`, `depth_transparency_threshold = 0.15`

---

## Stage 4 — Depth-Weighted Centroid

**Purpose**: locate the 2D centroid of the mask, weighted toward the nearest visible surface
face of the object (rather than the geometric centre, which may fall on a distant part or
inside a concave shape).

```
w_i = 1 / (depth_i + ε)         ε = 1e-6  (avoid division by zero)

cx  = Σ(x_i × w_i) / Σ(w_i)
cy  = Σ(y_i × w_i) / Σ(w_i)
```

Since the weighted centroid may fall in an empty region of the mask (e.g. a donut shape), the
nearest actual mask pixel to (cx, cy) is used as the anchor point:

```
distances² = (x_i − cx)² + (y_i − cy)²
anchor     = argmin(distances²)
```

This anchor pixel's `(x, y)` becomes `mask_centroid_2d` and is used in back-projection.

---

## Stage 5 — z_val: Histogram Mode Depth

**Purpose**: compute a robust single depth value representing the object's primary surface.
Mean and median are biased by halo pixels and background bleed; the histogram mode picks the
dominant depth bin.

Only the **inner circle** of the mask is used — pixels within a circle of equivalent area
scaled by `depth_central_fraction`:

```
area    = sum(mask_pixels)
radius  = sqrt(area × central_frac / π)    central_frac = 0.5 (config)

inner_pixels = {pixel i : dist(i, anchor)² ≤ radius²}
```

Histogram over inner pixel depths:

```
n_bins = clamp(len(inner_pixels) // 5, 10, 100)
hist, edges = histogram(inner_depths, bins=n_bins)
peak_bin    = argmax(hist)
z_val       = (edges[peak_bin] + edges[peak_bin+1]) / 2
```

If no inner pixels exist (very small mask), `z_val` falls back to `median(all_mask_depths)`.

**Config**: `depth_central_fraction = 0.5`

---

## Stage 6 — Back-Projection to 3D

**Purpose**: convert the 2D anchor pixel + z_val into 3D world coordinates using the pinhole
camera model.

```
X = (u − cx_K) × z / fx
Y = (v − cy_K) × z / fy
Z = z_val
```

Where:
- `(u, v)` = 2D anchor pixel from Stage 4
- `z` = `z_val` from Stage 5
- `fx, fy, cx_K, cy_K` = camera intrinsics (from calibration file, explicit config, or FOV estimate)

See [CAMERA_CALIBRATION.md](CAMERA_CALIBRATION.md) for intrinsics priority.

---

## Dual Erosion Comparison

Every object is processed twice: once with erosion (`use_erosion=True`) and once without
(`use_erosion=False`). Both sets of results are stored in the scene JSON:

```json
{
  "depth_stats":             { "z_val": 2.31, "..." },   // with erosion
  "coordinates_3d":          { "x": -0.42, "y": 0.15, "z": 2.31 },
  "mask_centroid_2d":        [330, 245],

  "depth_stats_no_erosion":  { "z_val": 2.18, "..." },   // without erosion
  "coordinates_3d_no_erosion": { "x": -0.41, "y": 0.14, "z": 2.18 },
  "mask_centroid_2d_no_erosion": [332, 246]
}
```

The comparison quantifies how much boundary-bleed bias erosion removes. For objects with thin
masks (where `kernel=0` and erosion is skipped), both sets are identical.

**Config**: `depth_erosion_comparison = True`
