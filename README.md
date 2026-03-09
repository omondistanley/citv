# CITV — Camera-Informed 3D Scene Understanding

> **Multi-model pipeline** that converts a single RGB image into a fully-labelled, metric 3D scene graph: per-object depth, 3D coordinates, semantic labels, and spatial relations.

![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.11-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Model Stack](#model-stack)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Output Format](#output-format)
7. [Configuration](#configuration)
8. [Camera Calibration](#camera-calibration)
9. [Technical Notes](#technical-notes)
10. [Citations](#citations)

---

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
│           FOV fallback: fx = W / (2·tan(FOV·π/360))         │
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
│           IoU deduplication (threshold 0.7) → merged masks  │
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

---

## Model Stack

| Model | Role | Source |
|---|---|---|
| **CLIP ViT-B/32** | Indoor/outdoor scene classification | [Radford et al., 2021](https://arxiv.org/abs/2103.00020) |
| **Depth Anything V2 Metric** | Metric monocular depth (metres) | [Yang et al., 2024](https://arxiv.org/abs/2406.09414) |
| **GroundingDINO** | Open-vocabulary object detection | [Liu et al., 2023](https://arxiv.org/abs/2303.05499) |
| **SAM2** | Prompted + automatic mask generation | [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) |
| **Florence-2** | Semantic labelling (`<OD>`) + relation captions | [Xiao et al., 2023](https://arxiv.org/abs/2311.06242) |
| **GRiT** | Dense captioning fallback label | [Wu et al., 2022](https://arxiv.org/abs/2212.00280) |
| **YOLOv8-cls** | Classification fallback label | [Jocher et al., 2023](https://github.com/ultralytics/ultralytics) |
| **Pix2SG** | Spatial relation scaffold | [Yao et al., 2024](https://arxiv.org/abs/2401.03600) |

All models run locally after setup. No API keys required (HuggingFace public repos only).

---

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

`setup.sh` performs these steps automatically:

| Step | Action |
|---|---|
| 0 | Disk-space check (warns if < 20 GB free) |
| 1 | Clone `GRiT` and `sam2` repos (skips if already present) |
| 2 | Download SAM2 checkpoint `sam2.1_hiera_large.pt` (~900 MB) |
| 3 | Download GRiT weights `grit_b_densecap_objectdet.pth` (~900 MB) |
| 4 | Install PyTorch with correct CUDA version (auto-detected via `nvcc`) |
| 5 | Install Python dependencies from `requirements.txt` |
| 6 | Build `detectron2` from `GRiT/third_party/CenterNet2` (5–15 min) |
| 7 | Install GRiT Python deps (excluding conflicting pins) |
| 8 | Build SAM2 `_C` CUDA extension |
| 9 | Pre-download all HuggingFace models (~6 GB; idempotent) |

After setup, the first pipeline run is instant — all models are pre-cached.

---

## Usage

### Run the pipeline

```bash
gcloud compute ssh --zone "zone here" "instance here" --project "name"
```

| Argument | Description |
|---|---|
| `--input_dir` | Directory of input images (JPG, PNG) |
| `--output_dir` | Output directory for JSON scene graphs and visualisations |

### Optional: use a calibration file

```bash
python scene_understanding.py \
  --input_dir images \
  --output_dir output_scene \
  --camera_calibration_file calibration.json
```

### Single-image quick test

```bash
python scene_understanding.py --input_dir images --output_dir output_scene
```

Place one image in `images/` and inspect `output_scene/<stem>_scene.json`.

---

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

## Configuration

Edit `config.py` to tune the pipeline. Key fields:

### Depth

| Field | Default | Description |
|---|---|---|
| `depth_model_variant` | `"auto"` | `"auto"` / `"indoor"` / `"outdoor"` |
| `depth_scale_factor` | `1.0` | Metric output needs no scaling |
| `depth_adaptive_erosion` | `True` | Erode mask before depth sampling |
| `mask_erosion_kernel_size` | `5` | Max erosion kernel (adapts to mask size) |
| `depth_outlier_sigma` | `2.0` | Sigma-clip rejection threshold (0 = disabled) |
| `depth_transparency_check` | `True` | Flag possibly-transparent objects |
| `depth_transparency_threshold` | `0.15` | Min depth separation (metres) to flag transparent |
| `depth_erosion_comparison` | `True` | Save depth stats with AND without erosion |

### Segmentation

| Field | Default | Description |
|---|---|---|
| `grounding_dino_box_thresh` | `0.15` | GDINO detection confidence threshold |
| `grounding_dino_text_thresh` | `0.15` | GDINO text alignment threshold |
| `grounded_sam2_fallback_to_amg` | `True` | Fall back to SAM2 AMG if GDINO fails |
| `run_both_segmentors` | `True` | Run GDINO+SAM2 AND SAM2 AMG; merge masks |
| `run_both_segmentors_iou_dedup` | `0.7` | IoU threshold to dedup overlapping masks |
| `sam2_amg_min_mask_region_area` | `100` | Minimum mask area in pixels for AMG |

### Filtering (all disabled by default — every mask passes)

| Field | Default | Description |
|---|---|---|
| `sam2_post_filter_min_stability` | `0.0` | Min SAM2 stability score (0 = no filter) |
| `sam2_post_filter_min_pred_iou` | `0.0` | Min SAM2 predicted IoU (0 = no filter) |
| `sam2_post_filter_min_area_px` | `0` | Min mask area in pixels (0 = no filter) |
| `sam2_post_filter_max_area_fraction` | `1.0` | Max fraction of image (1.0 = no filter) |

### Relations

| Field | Default | Description |
|---|---|---|
| `florence2_relation_enabled` | `True` | Run Florence-2 relation prediction |
| `relation_min_mask_overlap` | `0.02` | Min IoU between masks to predict relation |
| `pix2sg_mask_overlap_thresh` | `0.05` | Pix2SG spatial scaffold overlap threshold |

---

## Camera Calibration

For accurate metric depth and 3D coordinates, calibrate your camera using a checkerboard:

```bash
# Capture 20+ checkerboard images from different angles, save to calib_imgs/
python tools/calibrate_camera.py \
  --images calib_imgs/ \
  --out calibration.json \
  --pattern_size 9x6 \
  --square_size_mm 25
```

Then set in `config.py`:

```python
camera_calibration_file = "calibration.json"
apply_undistortion = True
```

The calibration JSON stores `fx, fy, cx, cy` (intrinsics) and `k1, k2, p1, p2` (distortion coefficients). Without calibration the pipeline estimates intrinsics from a 60° horizontal FOV assumption, which is accurate to within ~10% for typical smartphone and webcam lenses.

---

## Technical Notes

See [`NOTES.md`](NOTES.md) for in-depth documentation of every algorithm, formula, and design decision — including:

- Exact depth formula derivations (pinhole model, histogram mode z_val, depth-weighted centroid, back-projection)
- Adaptive erosion sizing logic
- Sigma-clipping implementation
- Transparency detection algorithm
- Dual-segmentor IoU deduplication
- Florence-2 RED/BLUE colour-overlay relation extraction
- All fixes applied (5.1 – 5.7) with problem/solution/rationale

---

## Citations

### Models

```bibtex
@article{yang2024depthv2,
  title   = {Depth Anything V2},
  author  = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen
             and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal = {arXiv:2406.09414},
  year    = {2024}
}

@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and
             Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and
             Khedr, Haitham and Rädle, Roman and Rolland, Chloe and
             Gustafson, Laura and Mintun, Eric and Pan, Junting and
             Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan
             and Girshick, Ross and Dollár, Piotr and Feichtenhofer, Christoph},
  journal = {arXiv:2408.00714},
  year    = {2024}
}

@article{liu2023groundingdino,
  title   = {Grounding DINO: Marrying DINO with Grounded Pre-Training for
             Open-Set Object Detection},
  author  = {Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng
             and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei
             and Su, Hang and Zhu, Jun and Zhang, Lei},
  journal = {arXiv:2303.05499},
  year    = {2023}
}

@article{xiao2023florence2,
  title   = {Florence-2: Advancing a Unified Representation for a Variety
             of Vision Tasks},
  author  = {Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and
             Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and
             Yuan, Lu},
  journal = {arXiv:2311.06242},
  year    = {2023}
}

@article{radford2021clip,
  title   = {Learning Transferable Visual Models From Natural Language
             Supervision},
  author  = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and
             Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and
             Sastry, Girish and Askell, Amanda and Mishkin, Pamela and
             Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  journal = {arXiv:2103.00020},
  year    = {2021}
}

@article{wu2022grit,
  title   = {GRiT: A Generative Region-to-text Transformer for Object
             Understanding},
  author  = {Wu, Jialian and Wang, Jianfeng and Yang, Zhengyuan and
             Gan, Zhe and Liu, Zicheng and Yuan, Junsong and Wang, Lijuan},
  journal = {arXiv:2212.00280},
  year    = {2022}
}

@software{jocher2023yolov8,
  title   = {Ultralytics YOLOv8},
  author  = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics}
}

@article{yao2024pix2sg,
  title   = {Pix2SG: Pixel-level Scene Graph Generation with
             Positional and Semantic Priors},
  author  = {Yao, Yuan and others},
  journal = {arXiv:2401.03600},
  year    = {2024}
}
```

### Methods and Formulas

```bibtex
@book{hartley2004mvg,
  title     = {Multiple View Geometry in Computer Vision},
  author    = {Hartley, Richard and Zisserman, Andrew},
  edition   = {2nd},
  publisher = {Cambridge University Press},
  year      = {2004},
  note      = {Pinhole camera model: X=(u-cx)z/fx, Y=(v-cy)z/fy}
}

@article{chauvenet1863sigma,
  title   = {Method of Least Squares},
  author  = {Chauvenet, William},
  journal = {Appendix to Manual of Spherical and Practical Astronomy},
  year    = {1863},
  note    = {Sigma-clipping: reject pixels where |depth_i - mean| > sigma * std}
}

@article{bradski2000opencv,
  title   = {The OpenCV Library},
  author  = {Bradski, Gary},
  journal = {Dr. Dobb's Journal of Software Tools},
  year    = {2000},
  note    = {cv2.undistort, cv2.findChessboardCorners, cv2.calibrateCamera}
}
```

