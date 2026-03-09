# Labelling and Relations

## Overview

Every mask needs a semantic label and the scene graph needs predicates between objects.
Labelling uses a 3-stage priority chain; relation prediction uses a two-layer system (Pix2SG
spatial scaffold enriched by Florence-2 semantic captions).

---

## Labelling — 3-Priority Chain

`_label_mask()` in `scene_understanding.py` tries three sources in priority order, stopping at
the first that returns a non-generic label:

```
Priority 1 — GroundingDINO label
  if label != "object":
      use GDINO label directly
      → most accurate; label came from the same model that drew the bbox

Priority 2 — Florence-2 (optional for labelling)
  crop = image[bbox]
  crop[~mask] = mean(full_image)    ← mean-fill background
  result = Florence2.label_crop(crop)
  if result["label"] != "object":
      use Florence-2 label

Priority 3 — RAM++
  result = RAMPP.label_crop(crop)
  if result["label"] != "object":
      use RAM++ label
  else:
      label = "object"
```

### Florence-2 vs RAM++ for Labelling

Florence-2 provides richer local context and often better labels for visually complex crops,
but it is slower and heavier. RAM++ is usually better as an open-vocabulary fallback when
GroundingDINO and Florence-2 both return generic labels.

### Mean-Fill Background

Out-of-mask pixels are set to the mean image colour before passing to Florence-2 or RAM++:

```python
bg_mean = img_bgr.mean(axis=(0, 1)).astype(np.uint8)
crop[~mask_resized] = bg_mean
```

Setting background to black (zeros) creates an unnatural brightness/colour distribution.
Mean-fill preserves natural image statistics so both Florence-2 and RAM++ behave as trained.

---

## Relations — Two-Layer System

### Layer 1 — Pix2SG Spatial Scaffold

Generates geometric predicates for every pair of objects, using three decision levels in
priority order:

**Level 1 — Pixel overlap (overlapping)**
```
inter = sum(mask_sub & mask_obj)
union = sum(mask_sub | mask_obj)

if inter / union >= mask_overlap_thresh (0.05):
    predicate = "overlapping"
```

Pixel mask IoU is used rather than bounding-box IoU because bbox IoU fires "overlapping"
for any two rectangles that share area — even objects far apart in 3D. Pixel intersection
measures actual foreground contact.

**Level 2 — Depth-axis (in_front_of / behind)**
```
depth_diff = |z_obj − z_sub|

if depth_diff >= depth_far_threshold (3.0 m):
    predicate = "in_front_of" if z_sub < z_obj else "behind"
```

When objects are > 3 m apart in depth, the dominant spatial relation is along the depth axis,
not the 2D image plane. This threshold avoids incorrectly labelling adjacent objects (e.g.,
two chairs 0.5 m apart) as "in_front_of".

**Level 3 — 2D direction (left_of / right_of / above / below)**
```
(sx, sy) = mask_centroid_2d of subject
(ox, oy) = mask_centroid_2d of object

dx = ox − sx
dy = oy − sy

if |dx| >= |dy|:
    predicate = "left_of" if dx > 0 else "right_of"
else:
    predicate = "above" if dy > 0 else "below"
```

Mask centroid (centre-of-mass of foreground pixels) is used rather than the bbox midpoint
for more accurate left/right/above/below classification of asymmetric objects
(e.g., an L-shaped sofa).

**Neighbour selection**: only the `max_relations_per_object` (default 8) nearest objects
by centroid distance are considered, to avoid O(N²) Florence-2 calls on large scenes.

**Config**:
```python
pix2sg_mask_overlap_thresh  = 0.05
pix2sg_depth_near_threshold = 1.0
pix2sg_depth_far_threshold  = 3.0
```

---

### Layer 2 — Florence-2 Semantic Enrichment

For every object pair whose masks overlap by IoU > `relation_min_mask_overlap` (0.02),
Florence-2 is queried for a semantic predicate.

**Colour-overlay method** (`Florence2Wrapper.predict_relation()`):

```
1. Compute union bbox of both masks (+ 10px padding).
2. Crop the full image to the union region.
3. Apply colour overlay:
     pixel ∈ mask_subject: R channel += 0.45 × (255 − R)   → red tint
     pixel ∈ mask_object:  B channel += 0.45 × (255 − B)   → blue tint
4. Prompt Florence-2:
     "<CAPTION> Describe the relationship between the red {sub} and the blue {obj}
      in one short phrase."
5. Parse the free-form response → canonical predicate.
```

Alpha 0.45 provides a strong enough colour signal for Florence-2 to distinguish the two
regions while preserving enough original texture for contextual understanding.

Florence-2 does not have a native region-pair relation task, but its `<CAPTION>` task
understands colour references in natural language very well because of its diverse
vision-language pre-training data.

**Canonical predicate mapping** — free-form phrases are matched against a lookup table:

| Phrases | Canonical predicate |
|---|---|
| on top of, resting on, sitting on, lying on | `on` |
| under, below, beneath | `under` |
| next to, beside, adjacent, alongside | `is_next_to` |
| in front of | `in_front_of` |
| behind, in back of | `behind` |
| inside, within, contained in | `inside_of` |
| hanging from, suspended from | `hangs_from` |
| leaning on, leaning against | `leans_on` |
| holding, carrying, gripping | `holds` |
| wearing, dressed in | `wears` |
| riding, mounted on | `rides` |
| eating, consuming | `eats` |
| drinking | `drinks` |
| reading | `reads` |
| using, operating | `uses` |
| looking at, gazing at | `looks_at` |
| playing | `plays` |
| kicking | `kicks` |

If no phrase matches, `None` is returned and no triplet is added — avoiding noisy relations.

**Config**: `florence2_relation_enabled = True`, `relation_min_mask_overlap = 0.02`

---

## Scene JSON — Relations Format

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

`source_layer` values:
- `"spatial_scaffold"` — geometric rule from Pix2SG
- `"florence2"` — semantic predicate from Florence-2 colour-overlay caption

Spatial scaffold always runs. Florence-2 enrichment only runs for overlapping pairs.
