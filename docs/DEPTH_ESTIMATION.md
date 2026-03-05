# Depth Estimation

## Overview

Metric depth is estimated by **Depth Anything V2 Metric** — a monocular depth model that outputs
true metres, not a relative/affine-invariant scale. Two variants are available, trained on different
datasets for different depth ranges:

| Variant | Training set | Depth range | Use |
|---|---|---|---|
| `Depth-Anything-V2-Metric-Indoor-Large-hf` | NYUv2 | 0 – 10 m | Rooms, corridors, offices |
| `Depth-Anything-V2-Metric-Outdoor-Large-hf` | KITTI | 0 – 80 m | Streets, outdoor scenes |

Using the wrong variant introduces systematic scale and shape errors that cascade into wrong Z
coordinates and broken depth-gated spatial relations (in_front_of / behind).

---

## Scene Classification with CLIP

Before loading the depth model, **CLIP ViT-B/32** classifies the scene as `indoor` or `outdoor`
so the correct variant is selected automatically (when `depth_model_variant = "auto"` in config).

**How it works** (`depth.py` — `SceneTypeClassifier.classify()`):

1. Encode the image and 6 text prompts with CLIP.
2. Compute softmax similarity between image and all prompts.
3. Average scores within each group.
4. Select the group with the higher average.

```
all_prompts = [indoor_0, indoor_1, indoor_2, outdoor_0, outdoor_1, outdoor_2]
logits       = CLIP(image, all_prompts)          # shape (6,)
probs        = softmax(logits)

score_indoor  = mean(probs[0:3])
score_outdoor = mean(probs[3:6])

scene_type = "indoor" if score_indoor >= score_outdoor else "outdoor"
```

Text prompts used:

```python
INDOOR  = ["a photo of an indoor room",
           "an interior space with furniture",
           "inside a building with walls and ceiling"]

OUTDOOR = ["a photo taken outside with sky or open space",
           "an outdoor scene with trees, streets or buildings",
           "a landscape or street scene outside"]
```

Full sentences are used (not single words) because CLIP's contrastive pre-training aligns better
with natural language image descriptions.

---

## VRAM Lifecycle

CLIP ViT-B/32 occupies ~340 MB of GPU VRAM. Loading it simultaneously with the depth model
(~1.8 GB) and SAM2 (~850 MB) can exhaust memory on smaller cards.

Strategy:

```
1. Load CLIP (~340 MB)
2. Classify scene → "indoor" / "outdoor"
3. Unload CLIP — free 340 MB before next step
4. Load depth model (~1.8 GB)
```

`SceneTypeClassifier.unload()` calls `del model`, `del processor`,
`torch.cuda.empty_cache()` to ensure the memory is actually freed.

---

## Metric Output

`DepthAnythingV2Backend.infer()` returns `out["predicted_depth"]` — a `float32` tensor in
**true metres**. This is the correct key for the metric HuggingFace checkpoints.

```python
# depth.py — DepthAnythingV2Backend.infer()
outputs = self._model(**inputs)
depth = outputs.predicted_depth          # float32, metres
depth_np = depth.squeeze().cpu().numpy() # shape (H, W)
```

`depth_scale_factor = 1.0` in config — the output is already metres, no scaling is applied.
The legacy `× 10` hack used for relative models has been removed.

---

## Depth Map Resizing

The depth model outputs at its own internal resolution, which is then resized to the input image
dimensions:

```python
depth_full = cv2.resize(raw_depth, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
```

`INTER_NEAREST` (nearest-neighbour) is used instead of `INTER_LINEAR` (bilinear) because:
- `INTER_LINEAR` blends adjacent pixel values — at object boundaries this averages foreground
  depth with background depth, producing "halo" pixels with intermediate depth values.
- These halo pixels contaminate per-mask depth stats in Stage 4, shifting `z_val` toward the
  background.
- `INTER_NEAREST` preserves the actual metric values output by the model.

---

## Depth Map Saving

Two outputs are saved per image:

| File | Format | Content |
|---|---|---|
| `depth/{stem}_depth_metric.npy` | float32 NumPy array | True metric depth in metres |
| `scene_graph/{stem}_depth_map.png` | uint8 PNG (colourised) | Visualisation only |

The colourised PNG is for human inspection. All per-mask depth computations use the `.npy` array.

16-bit depth PNG encoding (used in `_save_depth_maps()`):

```
max_range = 10.0 m (indoor) or 80.0 m (outdoor)
depth_uint16 = clip(depth_metres / max_range, 0, 1) × 65535
```

This encodes the full metric range into a 16-bit PNG with ~1.5mm precision for indoor scenes.
