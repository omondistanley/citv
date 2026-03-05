# Camera Calibration

## Overview

All depth-to-3D back-projection in the pipeline uses the pinhole camera model:

```
X = (u − cx) × z / fx
Y = (v − cy) × z / fy
Z = z
```

Where `fx, fy` are focal lengths in pixels and `cx, cy` is the principal point (image centre).
These intrinsics must be correct for accurate 3D coordinates. The pipeline supports three
sources in priority order.

---

## Intrinsics Priority

```
1. Calibration file (highest accuracy)
   camera_calibration_file = "calibration.json"
   → fx, fy, cx, cy from cv2.calibrateCamera()

2. Explicit config values
   camera_fx = ...   camera_fy = ...
   camera_cx = ...   camera_cy = ...

3. FOV estimate (fallback, ~10% error for typical lenses)
   fx = W / (2 × tan(FOV_deg × π / 360))
   fy = fx
   cx = W / 2
   cy = H / 2
```

The default `camera_fov_degrees = 60.0` is a reasonable estimate for smartphones and webcams.
Error is proportional to how far the true FOV deviates from 60°. For a wide-angle lens at 90°
actual FOV, the 60° estimate gives ~40% error in `fx`, which translates directly to ~40% error
in computed X/Y world coordinates.

---

## Lens Distortion

Real lenses introduce distortion that violates the pinhole model. Uncorrected distortion
causes objects near image edges to back-project to incorrect 3D positions — the error is
proportional to the object's distance from the image centre.

Two distortion components are modelled:

**Radial distortion** (barrel or pincushion):
```
r² = x² + y²     (normalised image coordinates)

x_r = x × (1 + k1·r² + k2·r⁴)
y_r = y × (1 + k1·r² + k2·r⁴)
```

**Tangential distortion** (lens misalignment with sensor plane):
```
x_t = x + 2·p1·x·y + p2·(r² + 2x²)
y_t = y + p1·(r² + 2y²) + 2·p2·x·y
```

Total correction: `(x_corrected, y_corrected) = (x_r + x_t, y_r + y_t)`

Undistortion is applied as the very first processing step (`cv2.undistort()`) before depth
estimation, segmentation, or intrinsics are used. All subsequent models and algorithms then
operate on a geometrically correct image.

---

## Calibration Tool

Calibrate any camera using a printed checkerboard pattern:

```bash
# Capture 20+ checkerboard images from different angles
python tools/calibrate_camera.py \
    --images   path/to/checkerboard_images/ \
    --out      calibration.json \
    --pattern_size  9x6          \   # interior corner count (cols×rows)
    --square_size_mm  25             # physical square size in mm
```

The tool:
1. Detects interior corners with `cv2.findChessboardCorners()` + `cv2.cornerSubPix()` for
   sub-pixel accuracy.
2. Collects 3D object points (known from pattern geometry) and 2D image points.
3. Runs `cv2.calibrateCamera()` → camera matrix K and distortion coefficients D.
4. Computes reprojection error (RMS pixels) as a quality metric.
5. Saves to JSON:
   ```json
   {
     "fx": 1423.7, "fy": 1421.2,
     "cx": 960.3,  "cy": 540.1,
     "k1": -0.31,  "k2": 0.12,
     "p1": 0.001,  "p2": -0.003
   }
   ```

**Good calibration**: RMS reprojection error < 1.0 pixel. Use at least 20 images from varied
angles including oblique shots and corners of the frame.

Then set in `config.py`:
```python
camera_calibration_file = "calibration.json"
apply_undistortion      = True
```

---

## Why Undistort Before Everything Else

The pipeline applies `cv2.undistort()` as Stage 0 — before depth estimation, segmentation,
and intrinsics are used.

If undistortion were applied after segmentation, the mask pixel coordinates would be in the
distorted image space but the intrinsics K would assume the undistorted pinhole model.
Back-projection `X = (u − cx)·z/fx` would then be incorrect for any pixel not at the image
centre, with error increasing toward the edges.

By undistorting first, every subsequent model and formula operates on a geometrically correct
pinhole image.

---

## Example: Effect of Calibration vs FOV Estimate

For a smartphone camera with true `fx = 1400 px` (28mm equivalent) on a 1920×1080 image:

```
FOV estimate (default 60°): fx = 1920 / (2·tan(30°)) = 1920 / 1.155 = 1662 px
True fx: 1400 px
Error: (1662−1400)/1400 = 18.7%

→ X coordinates are 18.7% too small
→ Object at true X=2.0m back-projects to X=1.63m
```

For critical 3D measurements, calibration is strongly recommended.
