## Overview

CITV takes a directory of images and, for each image, produces a structured JSON scene graph describing:

- Every segmented object (mask, bounding box, 2D centroid, depth-weighted 3D coordinates)
- Metric depth statistics per object — with and without adaptive mask erosion for comparison
- Semantic labels ranked by a 4-model priority chain (GDINO → Florence-2 → GRiT → YOLOv8)
- Spatial and semantic relations between overlapping objects (Pix2SG + Florence-2)
- Optional lens-undistortion using a calibrated camera matrix

The pipeline runs entirely locally after a one-shot `setup.sh` and produces no calls to paid APIs.

---

## Pipeline Architecture

```
Input image
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 0  Camera undistortion (optional)                     │
│           cv2.undistort() using calibration JSON             │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1  Camera intrinsics                                  │
│           Priority: calibration file > explicit fx/fy > FOV  │
│           FOV fallback: fx = W / (2·tan(FOV·π/360))          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2  Metric depth estimation                            │
│           CLIP classifies scene → indoor/outdoor             │
│           Depth Anything V2 Metric → float32 metres          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3  Instance segmentation (dual-segmentor)             │
│           GroundingDINO → SAM2 per-bbox  (object-level)      │
│           SAM2 AMG grid-based            (part/small-object) │
│           IoU deduplication (threshold 0.7) → merged masks   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 4  Per-mask depth analysis                            │
│           Adaptive erosion → sigma-clipping → histogram mode │
│           Back-projection → (X, Y, Z) in metres              │
│           Transparency detection via border-ring comparison  │
│           Dual erosion: stats stored with AND without erosion│
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 5  Semantic labelling                                 │
│           GDINO text label > Florence-2 <OD> >               │
│           GRiT dense caption > YOLOv8-cls                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 6  Relation graph (Pix2SG + Florence-2)               │
│           Spatial scaffold for overlapping mask pairs        │
│           Florence-2 <CAPTION> over RED/BLUE overlay crops   │
│           Canonical predicate extraction                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
              {stem}_scene.json
```

## Setup

### Requirements

- Python 3.9–3.11
- CUDA-capable GPU (≥ 8 GB VRAM recommended)
- `nvcc` on PATH (`sudo apt install nvidia-cuda-toolkit`)
- ~20 GB free disk space

### One-shot install

```bash
git clone https://github.com/omondistanley/citv.git
cd citv
bash setup.sh
```

### Pipeline quick test

```bash
python scene_understanding.py --input_dir images --output_dir output_scene
```

Place image(s) in `images/` and inspect `output_scene/<stem>_scene.json`.

## Output Format

Each image produces `{stem}_scene.json`:

```json
{
  "image": "images/living_room.jpg",
  "scene_type": "indoor",
  "objects": [
    {
      "id": "obj_0",
      "label": "sofa",
      "conf": 0.87,
      "segmentor": "GroundedSAM2",
      "bbox": [120, 80, 540, 410],
      "mask_centroid_2d": [330, 245],
      "coordinates_3d": {"x": -0.42, "y": 0.15, "z": 2.31},
      "depth_stats": {
        "z_val": 2.31,
        "z_val_pixels": 2.31,
        "possibly_transparent": false,
        "depth_separation_from_background": 0.89
      },
      "depth_stats_no_erosion": { "z_val": 2.28, "..." : "..." },
      "coordinates_3d_no_erosion": {"x": -0.41, "y": 0.14, "z": 2.28},
      "grounded_sam2_label": "sofa",
      "grounded_sam2_confidence": 0.87
    }
  ],
  "relations": [
    {
      "subject": "obj_0",
      "predicate": "in front of",
      "object": "obj_2",
      "source_layer": "florence2"
    }
  ]
}
```

Every object carries **dual depth stats** — one set computed with adaptive mask erosion, one without — so you can compare the effect of erosion on depth accuracy.

---

## Model Stack

| Model | Role | Source |
|---|---|---|
| **CLIP ViT-B/32** | Indoor/outdoor scene classification | [Radford et al., 2021](https://arxiv.org/abs/2103.00020) |
| **Depth Anything V2 Metric** | Metric monocular depth (metres) | [Yang et al., 2024](https://arxiv.org/abs/2406.09414) |
| **GroundingDINO** | Open-vocabulary object detection | [Liu et al., 2023](https://arxiv.org/abs/2303.05499) |
| **SAM2** | Prompted + automatic mask generation | [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) |
| **Florence-2** | Semantic labelling (`<OD>`) + relation captions | [Xiao et al., 2023](https://arxiv.org/abs/2311.06242) |
| **YOLOv8-cls** | Classification fallback label | [Jocher et al., 2023](https://github.com/ultralytics/ultralytics) |
| **Pix2SG** | Spatial relation scaffold | [Yao et al., 2024](https://arxiv.org/abs/2401.03600) |
