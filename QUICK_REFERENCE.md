# Quick Reference: Using the Refactored Scene Understanding Package

## Installation
```python
# The package is located at: scene_understanding/
# Already in your citv directory
import sys
sys.path.insert(0, '/home/omondistanley_oduor/citv')
```

## 1. Preprocessing - Image Loading & Camera Calibration

### Load Images
```python
from scene_understanding.preprocessing import load_bgr_image
import cv2

# Supports: JPEG, PNG, HEIF, HEIC, BMP, TIFF, WebP
img_bgr = load_bgr_image("path/to/image.jpg")  # Returns HxWx3 BGR

# Or HEIF:
img_bgr = load_bgr_image("path/to/image.heic")  # Graceful fallback
```

### Camera Calibration
```python
from scene_understanding.preprocessing import CameraCalibration

# Method 1: From OpenCV calibration file
calib = CameraCalibration(
    calibration_dict={
        "fx": 1435.8, "fy": 1435.8,
        "cx": 960, "cy": 540,
        "k1": -0.05, "k2": 0.01, "p1": 0, "p2": 0
    }
)

# Method 2: Explicit parameters
calib = CameraCalibration(
    camera_fx=1435.8,
    camera_fy=1435.8,
    camera_cx=960,
    camera_cy=540
)

# Method 3: FOV-based (automatic fallback)
calib = CameraCalibration(camera_fov_degrees=71.0)

# Use it
K = calib.get_intrinsics(width=1920, height=1080)
print(K)  # {"fx": 1435.8, "fy": 1435.8, "cx": 960, "cy": 540}

# Apply lens distortion correction
img_undistorted = calib.undistort_image(img_bgr)

# Back-project pixel to 3D
point_3d = CameraCalibration.back_project(u=960, v=540, z=2.5, K=K)
# Returns: {"x": 0.0, "y": 0.0, "z": 2.5}
```

---

## 2. Image Processing

### Resizing & Color Conversion
```python
from scene_understanding.preprocessing import (
    resize_image_if_needed,
    rgba_to_bgr,
    bgr_to_rgb
)
import cv2

# Intelligent downscaling
img_bgr, img_rgb, scale, new_size = resize_image_if_needed(
    img_bgr=img_bgr,
    img_rgb=img_rgb,
    max_side=1280  # Won't resize if smaller
)
print(f"Scale: {scale}, Size: {new_size}")

# Color conversion
img_rgb = bgr_to_rgb(img_bgr)
img_bgr_back = rgb_to_bgr(img_rgb)
```

---

## 3. Utilities - Bounding Box & Mask Operations

### Bbox Conversion
```python
from scene_understanding.utils import (
    xywh_to_xyxy,
    xyxy_to_xywh,
    iou_xyxy,
    mask_iou
)

# Format conversion
bbox_xywh = [100, 200, 300, 400]  # [x, y, width, height]
bbox_xyxy = xywh_to_xyxy(bbox_xywh)
# Result: [100, 200, 400, 600]

bbox_xywh_back = xyxy_to_xywh(bbox_xyxy)
# Result: [100, 200, 300, 400]

# Compute IoU
box1 = [0, 0, 100, 100]
box2 = [50, 50, 150, 150]
iou = iou_xyxy(box1, box2)
print(f"IoU: {iou:.3f}")  # IoU: 0.143

# Mask IoU
import numpy as np
mask1 = np.zeros((100, 100), dtype=bool)
mask1[10:50, 10:50] = True

mask2 = np.zeros((100, 100), dtype=bool)
mask2[30:70, 30:70] = True

miou = mask_iou(mask1, mask2)
print(f"Mask IoU: {miou:.3f}")  # Mask IoU: 0.250
```

---

## 4. Output - Visualization & Scene Graphs

### Save Visualizations
```python
from scene_understanding.output import VisualizationSaver
from pathlib import Path
import numpy as np

saver = VisualizationSaver()

# Save depth map as colormap
depth_map = np.random.rand(720, 1280) * 10  # 0-10 meters
saver.save_depth_map(
    metric_depth=depth_map,
    path=Path("output/depth_map.png"),
    colormap="inferno"  # or "viridis", "hot", etc.
)

# Save segmentation (colored masks with labels)
objects = [
    {
        "label": "person",
        "mask_centroid_2d": [640, 360],
        "bbox": [100, 200, 300, 400],
        "_sam2_mask_array": np.random.rand(720, 1280) > 0.5,
    },
    {
        "label": "chair",
        "mask_centroid_2d": [800, 500],
        "bbox": [500, 400, 700, 600],
        "_sam2_mask_array": np.random.rand(720, 1280) > 0.6,
    }
]

saver.save_segmentation_map(
    objects=objects,
    path=Path("output/segmentation.png")
)

# Save original + tinted overlay (45% transparency)
img_rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
saver.save_tinted_overlay(
    objects=objects,
    image_rgb=img_rgb,
    path=Path("output/overlay.png"),
    alpha=0.45
)

# Save depth only where objects are
saver.save_depth_mask_mapping(
    metric_depth=depth_map,
    matched_objects=[{"mask": obj["_sam2_mask_array"]} for obj in objects],
    path=Path("output/depth_mapping.png")
)
```

### Build Scene Graph JSON
```python
from scene_understanding.output import SceneGraphBuilder
from datetime import datetime

builder = SceneGraphBuilder()

scene_graph = builder.build_scene_depth_mask_json(
    image_path="/path/to/image.jpg",
    path_stem="image_01",
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    image_size=[1920, 1080],
    matching_mode="A",
    depth_map_path="depth/image_01_depth_metric.npy",
    depth_map_image_path="scene_graph/image_01_depth_map.png",
    depth_global_min=0.5,
    depth_global_max=10.0,
    depth_global_mean=3.2,
    segmentation_map_image_path="scene_graph/image_01_segmentation.png",
    num_auto_masks=12,
    mapping_image_path="scene_graph/image_01_depth_mask_mapping.png",
    objects=[
        {
            "id": "obj_0",
            "label": "person",
            "confidence": 0.94,
            "bbox_xyxy": [100, 200, 400, 600],
            "coordinates_3d": {"x": 0.5, "y": 0.2, "z": 2.5},
            "depth_stats": {"z_min": 2.3, "z_max": 2.7, "z_mean": 2.5},
        }
    ],
)

# Save to JSON
builder.save_scene_json(scene_graph, Path("output/image_01_scene.json"))
```

---

## 5. Core Types & Constants

### Use TypedDict for Type Safety
```python
from scene_understanding.core import ObjectDetection, SceneGraph

# IDE will provide autocomplete!
obj: ObjectDetection = {
    "id": "obj_0",
    "label": "person",
    "confidence": 0.94,
    "bbox_xyxy": [100, 200, 400, 600],
    "mask_centroid_2d": [250, 400],
    "area_pixels": 25000,
    "coordinates_3d": {
        "x": 0.5,
        "y": 0.2,
        "z": 2.5,
        "z_min": 2.3,
        "z_max": 2.7,
        "z_mean": 2.5,
    },
}
```

### Access Constants
```python
from scene_understanding.core import (
    GDINO_DEFAULTS,
    PIX2SG_DEFAULTS,
    SAM2_AMG_DEFAULTS,
    DEFAULT_GDINO_QUERY,
)

print(GDINO_DEFAULTS)
# {
#   "box_threshold": 0.3,
#   "text_threshold": 0.25,
# }

print(PIX2SG_DEFAULTS)
# {
#   "mask_overlap_thresh": 0.05,
#   "relation_min_mask_overlap": 0.02,
#   ...
# }

print(DEFAULT_GDINO_QUERY)
# "person. animal. vehicle. ..."
```

---

## 6. Integration Example

```python
from scene_understanding.preprocessing import load_bgr_image, CameraCalibration
from scene_understanding.preprocessing import resize_image_if_needed, bgr_to_rgb
from scene_understanding.output import VisualizationSaver, SceneGraphBuilder
from scene_understanding.utils import xywh_to_xyxy
from pathlib import Path
import cv2

# Step 1: Load and preprocess
img_bgr = load_bgr_image("input/image.jpg")
img_rgb = bgr_to_rgb(img_bgr)
img_bgr, img_rgb, scale, (w, h) = resize_image_if_needed(img_bgr, img_rgb)

# Step 2: Get camera calibration
calib = CameraCalibration(camera_fov_degrees=71.0)
K = calib.get_intrinsics(w, h)

# Step 3: Undistort (if needed)
img_bgr = calib.undistort_image(img_bgr)

# ... [Run segmentation, labeling, etc.] ...

# Step 4: Save outputs
output_dir = Path("output/scene")
output_dir.mkdir(parents=True, exist_ok=True)

saver = VisualizationSaver()
saver.save_depth_map(depth_map, output_dir / "depth_map.png")
saver.save_segmentation_map(objects, output_dir / "segmentation.png")
saver.save_tinted_overlay(objects, img_rgb, output_dir / "overlay.png")

# Step 5: Build and save scene graph
builder = SceneGraphBuilder()
scene_graph = builder.build_scene_depth_mask_json(...)
builder.save_scene_json(scene_graph, output_dir / "scene.json")

print("✓ Pipeline complete!")
```

---

## 📚 Available Modules

| Module | Purpose | Usage |
|--------|---------|-------|
| `core.types` | TypedDict definitions | `from scene_understanding.core import ObjectDetection` |
| `core.constants` | Configuration constants | `from scene_understanding.core import DEFAULT_GDINO_QUERY` |
| `preprocessing.image_loader` | Image loading | `from scene_understanding.preprocessing import load_bgr_image` |
| `preprocessing.calibration` | Camera calibration | `from scene_understanding.preprocessing import CameraCalibration` |
| `preprocessing.image_processing` | Image transforms | `from scene_understanding.preprocessing import resize_image_if_needed` |
| `output.savers` | Visualization saving | `from scene_understanding.output import VisualizationSaver` |
| `output.scene_graph_builder` | Scene graph building | `from scene_understanding.output import SceneGraphBuilder` |
| `utils` | Bbox/mask utilities | `from scene_understanding.utils import xywh_to_xyxy` |

---

## 🔍 Debugging

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
from scene_understanding.preprocessing import load_bgr_image
img = load_bgr_image("test.jpg")
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# Validate types
from scene_understanding.core import ObjectDetection
assert isinstance(obj, dict)  # TypedDicts are just dicts at runtime
print(f"Object keys: {list(obj.keys())}")
```

---

For Phase 3+ modules (segmentation, labeling, geometry, relations, depth), see REFACTORING_GUIDE.md
