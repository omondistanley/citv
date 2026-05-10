# Scene Graph Deep Dive

This document tracks **how CITV builds scene-graph artifacts today**: segmentation sources, labelling, relations, and what lands in `{stem}_scene.json`. It is maintained next to **`scene_understanding.py`** (monolithic pipeline body) and **`scene_understanding/pipeline.py`** (package entry: legacy vs staged).

**Pipeline routing:** `SceneUnderstandingPipeline.process_image` (`scene_understanding/pipeline.py`) runs **`legacy`** mode by default (delegates to `scene_understanding.py` `process_image`). With **`CITV_SCENE_PIPELINE_MODE=staged`** (or config `scene_pipeline_mode`), **staged** mode runs: either the **legacy-equivalent** path via `scene_understanding/stages/full_run.py` (same `scene_graph/<track>/` tree as legacy), or the **slim modular chain** when `CITV_STAGED_MODULAR_CHAIN_ONLY=1`, which writes `scene_graph/staged/{stem}_scene.json` via `scene_understanding/stages/scene_write.py` only.

## Executive Summary

- The repo is not choosing between per-object and per-mask segmentation. It already uses both.
- The effective production path is a hybrid:
  - `GroundedSAM2` gives object-level instance masks.
  - `SAM2 AMG` gives mask-level "segment everything" regions.
  - The two are merged with IoU deduplication.
- For scene graphs, object-level segmentation is the better primary representation because graph nodes should correspond to entities, not arbitrary parts.
- Mask-level segmentation is still necessary as a recall layer for small, occluded, layered, and missed objects.
- The best direction for this codebase is not "pick one", but "make object-level masks the canonical graph nodes, and keep mask-level segments as supporting evidence, child parts, and recovery candidates."
- **Serialization:** Full-track `{stem}_scene.json` includes **`metadata`**, **`objects`**, a **top-level `relations`** array, **`mask_hierarchy`**, **`layers`**, and optional **`regions`**; companion JSON/PNG paths are linked from `metadata`. (See §6.)

## 1. How Segmentation Is Currently Done

### Effective runtime path

End-to-end orchestration happens in two cooperating layers:

1. **Package entry** — `SceneUnderstandingPipeline.process_image` in [`scene_understanding/pipeline.py`](../scene_understanding/pipeline.py): chooses **legacy** vs **staged** (`resolve_scene_pipeline_mode`, env `CITV_SCENE_PIPELINE_MODE`).
2. **Legacy body** — `SceneUnderstandingPipeline.process_image` in [`scene_understanding.py`](../scene_understanding.py) (~5913+): undistort → depth → regions (optional) → segmentation tracks → per-mask geometry + `_label_mask` → Pix2SG / Florence relations → **writes** `scene_graph/<track_dir_name>/{stem}_scene.json` and siblings per track.

**Staged modes:**

- **Legacy-equivalent (default staged):** [`scene_understanding/stages/full_run.py`](../scene_understanding/stages/full_run.py) calls `LegacySceneUnderstandingPipeline.process_image`, so outputs match the monolith layout (`scene_graph/grounded_sam2/`, etc.).
- **Slim staged:** `CITV_STAGED_MODULAR_CHAIN_ONLY=1` runs `_run_stage_chain` + [`scene_write.export_staged_package`](../scene_understanding/stages/scene_write.py) → `scene_graph/staged/{stem}_scene.json` with a thinner baseline payload (metadata merged from `ctx.sam2_metadata` when present).

Important implementation anchors (line numbers drift — search by symbol):

- Per-mask labelling: `_label_mask()` in `scene_understanding.py`.
- Relation assembly and attachment: Pix2SG pipeline under [`scene_understanding/relations/`](../scene_understanding/relations/) and callsites in `scene_understanding.py` that populate the list passed into `sam2_scene_output["relations"]`.
- Final per-track scene dict: assignment to `sam2_scene_output` immediately before `json.dump` of `{stem}_scene.json` in `scene_understanding.py` (~6701+).

The modular package **mirrors** segmentation/labelling/relations in [`scene_understanding/segmentation/`](../scene_understanding/segmentation/), [`scene_understanding/labeling/`](../scene_understanding/labeling/), [`scene_understanding/relations/`](../scene_understanding/relations/); the CLI still loads the legacy class body from the root script via `pipeline.py`.

### Current segmentation modes

#### A. Per-object segmentation

`scene_understanding/segmentation/grounded_sam2.py` is clearly object-level.

- Grounding DINO predicts boxes and labels: `scene_understanding/segmentation/grounded_sam2.py:140`
- SAM2 then turns each detected box into one mask: `scene_understanding/segmentation/grounded_sam2.py:201`
- The wrapper explicitly says "one binary mask per detected entity": `scene_understanding/segmentation/grounded_sam2.py:205`

This is instance segmentation driven by object detection prompts.

#### B. Per-mask segmentation

`scene_understanding/segmentation/sam2_amg.py` is mask-level / region-level segmentation.

- It runs SAM2 Automatic Mask Generator with no prompts: `scene_understanding/segmentation/sam2_amg.py:18`
- It returns many binary regions, including parts and background-like regions: `scene_understanding/segmentation/sam2_amg.py:114`

This is not object-centric. It is region-centric.

#### C. Hybrid merge

The default config enables both:

- `run_both_segmentors = True`: see [`config.py`](../config.py) (`run_both_segmentors`)
- IoU dedup is **0.7**: see `iou_merge_threshold` / related keys in [`config.py`](../config.py)

The merge happens in `scene_understanding.py` (search for `generate(` / AMG merge after GroundedSAM2). Typical pattern:

- GroundedSAM2 produces primary masks.
- AMG adds masks that are not heavily overlapping with existing ones (IoU dedup threshold from config, default **0.7**).

Important detail: the first pass is already `GroundedSAM2`, with fallback-to-AMG behavior inside the wrapper, so the naming `amg_masks` is misleading.

### Conclusion on current behavior

The repo currently does:

- per-object segmentation as the main graph-node source
- per-mask segmentation as a supplemental recall source

That is the right overall shape for scene graph generation.

## 2. Which Segmentation Style Is More Accurate and More Useful Here

### For graph nodes: per-object is more useful

Why:

- Scene-graph nodes should represent entities, not arbitrary fragments.
- `GroundedSAM2` produces one mask per language-grounded entity when detection succeeds.
- That aligns better with downstream labels and relations than raw AMG regions do.

This is consistent with the OpenPSG framing of panoptic scene graph generation, where nodes are grounded by segmentation masks rather than loose boxes, and relations are mask-to-mask, not box-to-box:

- OpenPSG repo: https://github.com/Jingkang50/OpenPSG
- Relevant lines: `turn1view0` lines 311-325

### For coverage and layered scenes: per-mask is necessary

Why:

- AMG can recover small objects, partial occlusions, texture regions, and parts that detection misses.
- SAM2 is designed as a promptable segmentation foundation model and is strong in both prompted and automatic modes:
  - SAM 2 paper: https://arxiv.org/abs/2408.00714
- Grounding DINO is open-set detection with language prompts:
  - Grounding DINO paper: https://arxiv.org/abs/2303.05499

### Practical answer for this repo

Use:

- per-object segmentation as the canonical object layer
- per-mask segmentation as:
  - a recovery layer for missed objects
  - a child-part layer
  - a relation support layer for occlusion, containment, and attachment

Do not flatten both into one undifferentiated object list the way the current pipeline mostly does.

## 3. How To Make Segmentation More Efficient While Preserving Layered Objects and Relations

### What is already efficient

- GroundedSAM2 batches all boxes through SAM2 in one image pass: `scene_understanding/segmentation/grounded_sam2.py:205`
- SAM2 AMG resizes large images before generation: `scene_understanding/segmentation/sam2_amg.py:167`
- The pipeline defers Florence-2 loading until after segmentation to save VRAM: `scene_understanding.py:2828`

### What is currently inefficient or lossy

#### 1. AMG is run broadly, then flattened

The code keeps all masks because all post-filters are disabled:

- `sam2_post_filter_min_stability = 0.0`: `config.py:63`
- `sam2_post_filter_min_pred_iou = 0.0`: `config.py:64`
- `sam2_post_filter_min_area_px = 0`: `config.py:65`
- `sam2_post_filter_max_area_fraction = 1.0`: `config.py:66`

That improves recall, but it also pushes background and part noise into later stages.

#### 2. Layered objects are not fully explicit in the flat `objects` list

The pipeline stores a flat `objects` list. Provenance is compressed into fields like `segmentor`, while richer structure is partly pushed to **`mask_hierarchy`**, **`layers`**, and optional **`regions`** blocks saved alongside the same scene JSON.

#### 3. Depth-mask JSON

Depth-mask association (`dm_json`) is built in **`_build_depth_mask_json`** and **written** when the depth-mask export branch runs: `_write_json(dm_json, sam2_sg_dir / f"{path.stem}_depth_mask_{mode}.json")` in `scene_understanding.py` (search `_write_json(dm_json`). Whether this appears for a given run depends on **`depth_mask_modes`** / matching configuration—not all pipelines persist mode `B`.

### Recommended efficiency upgrades

#### A. Split masks into three tiers

After segmentation, classify each mask as:

- `entity_candidate`
- `part_candidate`
- `stuff_candidate`

Use features already available:

- mask area
- overlap with GDINO masks
- label specificity
- depth stability
- containment ratio

This lets you keep layered detail without forcing every mask to become a top-level graph object.

#### B. Add a containment graph before final object selection

For every mask pair, compute:

- IoU
- intersection over smaller mask
- centroid inclusion
- depth ordering

Then infer:

- `part_of`
- `inside`
- `occludes`
- `attached_to`

This is the missing structural layer for "layered objects and their relations."

#### C. Run AMG selectively, not uniformly

Current default is broad. A better schedule:

- run `GroundedSAM2` first
- run AMG only on:
  - uncovered image regions
  - high-entropy regions
  - masks around low-confidence GDINO detections
  - images with small-object-heavy domains

That preserves recall while cutting noise and cost.

#### D. Keep raw masks and merged masks separately

Store:

- `raw_grounded_masks`
- `raw_amg_masks`
- `merged_objects`
- `part_masks`

Right now, once the merge is done, provenance is mostly compressed into `segmentor`.

#### E. Add explicit mask hierarchy fields

Each final object should be able to carry:

- `parent_object_id`
- `child_object_ids`
- `part_mask_ids`
- `occluded_by`
- `occludes`
- `contains`
- `contained_by`

That will improve both querying and relation extraction.

## 4. How Labeling Works Now, and How To Get Actual Object Names

### Current label flow

The runtime per-mask priority chain is implemented in **`_label_mask()`** in [`scene_understanding.py`](../scene_understanding.py) (docstring: GDINO → Florence-2 → RAM++):

1. **GroundingDINO** label wins when it is already specific (not `object`).
2. **Florence-2** crop labelling fills in when the running label is still `object`.
3. **RAM++** crop labelling is the next fallback; note there is a **duplicate conditional block** that applies RAM++ twice in sequence — worth consolidating when touching this code.

When GDINO already has a specific class, **`mask_label_skip_secondary_when_gdino_specific`** can skip Florence-2 / RAM++ crops for speed.

Whole-image RAM++ tagging also feeds detection: **`refresh_gdino_query_if_configured`** ([`scene_understanding/core/prompting.py`](../scene_understanding/core/prompting.py)) can rewrite the GroundingDINO text query from RAM++ tags before boxes are predicted — see call sites around **`refresh_gdino_query_if_configured`** in `scene_understanding.py`.

This is **not** the README’s older “GRiT + YOLOv8” story; those are not applied in `_label_mask` in the current monolith body.

### Current strengths

- If GDINO already has a specific class, that is usually the best low-latency label.
- Florence-2 gives richer open-vocabulary recovery on mask crops.
- RAM++ improves vocabulary recall for crops and supplies **whole-image tags** that influence the GroundingDINO text query when RAM++ is enabled upstream of detection (see `_apply_rampp_tags_to_detections` / tagging paths in `scene_understanding.py`).

### Current weaknesses

#### 1. Florence-2 label extraction is too naive

`_extract_label_from_caption()` returns the first non-stopword token:

- `scene_understanding/labeling/florence2.py:312`

That is too weak for names like:

- `coffee mug`
- `dining table`
- `office chair`
- `tv remote`

It will often collapse a noun phrase into a single token.

#### 2. GDINO wins too early

If GDINO says a broad label like `container`, `furniture`, or `electronics`, it still wins as long as it is not literally `object` — see the “Priority 1: GDINO label” branch in `_label_mask`.

That blocks better crop-specific labels from Florence-2 or RAM++.

#### 3. There is no ontology normalization layer

The system stores one final `label`, but does not normalize:

- synonyms
- noun phrases
- singular/plural
- hypernym/hyponym conflicts

#### 4. Names are not fused across sources

You already store source evidence in **`sources`** on each object, which is good — search where **`sources`** is populated before scene write.

But there is no final label arbitration beyond hard priority.

### Recommended labeling upgrade

#### A. Replace hard priority with evidence fusion

For each object, keep:

- `canonical_name`
- `display_name`
- `aliases`
- `source_votes`

Example source votes:

- GDINO: `cup`
- Florence-2: `coffee mug`
- RAM++: `mug`, `cup`, `ceramic mug`

Then select:

- `canonical_name = coffee_mug`
- `display_name = coffee mug`
- `aliases = ["cup", "mug"]`

#### B. Treat broad GDINO labels as provisional

Introduce a "specificity gate." If GDINO returns a generic class family, let Florence-2 and RAM++ override it.

#### C. Extract noun phrases, not first tokens

Florence-2 is a general promptable vision model:

- Florence-2 paper: https://arxiv.org/abs/2311.06242

Use that strength by extracting the best noun phrase from the caption or `<OD>` output, not just the first word.

#### D. Preserve multiple label fields in output

Add these fields per object:

- `name`
- `canonical_name`
- `aliases`
- `category`
- `source_labels`

That will let scene graphs stay queryable even when the naming is uncertain.

#### E. Use RAM++ tags as label candidates, not just fallback

RAM is explicitly a tagging model:

- RAM paper: https://arxiv.org/abs/2306.03514

That means its output is best treated as candidate evidence, not just a last-resort single label.

## 5. How To Improve Relation Accuracy

### Current relation pipeline

The runtime relation flow centers on:

- Spatial / scaffold logic in [`scene_understanding/relations/pix2sg.py`](../scene_understanding/relations/pix2sg.py) (Pix2SG wrapper + geometry).
- Optional Florence-2 enrichment on overlapping mask pairs.
- Attachment into **`sources.Pix2SG.relations`** on objects **and** accumulation into the **graph-global `relations` list** that is serialized at the top level of `{stem}_scene.json` (see §6).

### What it gets right

- Relations are mask-aware, not only bbox-aware.
- Centroid and depth are used.
- Florence-2 is only called on overlapping masks, which contains cost.

### Main accuracy limitations

#### 1. Only one flat predicate is produced per neighbor

`_spatial_predicate_mask()` returns one label from:

- overlapping
- in_front_of / behind
- left_of / right_of
- above / below

That is too coarse for layered scenes.

#### 2. Florence-2 enrichment only runs for overlapping masks

See `scene_understanding/relations/pix2sg.py:361-402`.

That misses many valid relations:

- `next to`
- `in front of`
- `behind`
- `looking at`
- `leaning against`
- `hanging from`

when masks are close but not overlapping.

#### 3. Relation attachment can mis-bind duplicates

If IDs fail, `_attach_relations_by_triplets()` falls back to substring label matching — search its definition in `scene_understanding.py`.

That is risky when there are multiple objects with the same label.

#### 4. Dual representation: graph-global vs object-local

Consumers now get a **top-level `relations`** array in `{stem}_scene.json`, which is the portable scene-graph view. Per-object **`sources.Pix2SG.relations`** may still duplicate or supplement provenance; tooling should prefer the top-level array for graph algorithms unless debugging source attribution.

### Recommended relation upgrade

#### A. Add pair-candidate generation before predicate classification

Build a pair shortlist using:

- mask overlap
- distance between mask centroids
- depth proximity
- containment ratio
- boundary contact ratio
- vertical support heuristic

That is better than "nearest N neighbors only."

#### B. Predict relation families separately

Instead of one predicate function, predict:

- topology: `intersects`, `contains`, `inside`, `attached_to`
- support: `on`, `under`, `leaning_on`, `hanging_from`
- directional: `left_of`, `right_of`, `above`, `below`
- depth: `in_front_of`, `behind`

Then fuse them into the final predicate set.

#### C. Keep multi-relations when justified

Some pairs genuinely have multiple truths:

- `cup on table`
- `cup in front of plate`
- `person holding cup`

The current pipeline forces mostly one predicate per generation path.

#### D. Use relation-specific prompts with object names

`Florence2Wrapper.predict_relation()` currently ignores `subject_label` and `object_label`:

- `scene_understanding/labeling/florence2.py:245-259`

That is a missed opportunity. Use prompts like:

- "What is the relationship between the red coffee mug and the blue table?"

This should improve disambiguation.

#### E. Add a true panoptic scene-graph backend

The config already anticipates this:

- `psgtr_enabled`: `config.py:208`
- `univrd_enabled`: `config.py:210`

That is the right direction. For this repo, the strongest enhancement path is:

- keep current geometric scaffold as guaranteed fallback
- add a panoptic or visual-relationship model as a second relation expert
- fuse results by confidence and agreement

OpenPSG is the most directly aligned research direction because it is built around mask-grounded scene graphs rather than box-grounded ones:

- https://github.com/Jingkang50/OpenPSG

## 6. How `{stem}_scene.json` Is Built and Where It Lives

### Output path layout

`<output_dir>` is whatever you pass to `process_image`. Scene graphs are **per segmentation track**, not a single flat file at the repo root:

| Layout | Meaning |
|--------|---------|
| `{output_dir}/scene_graph/<track_dir_name>/{stem}_scene.json` | Primary scene graph for track **`grounded_sam2`**, **`amg`**, **`combined`**, or **`staged`** (depending on config / pipeline mode). |
| `{output_dir}/scene_graph/staged/{stem}_scene.json` | Written by **slim** staged export (`CITV_STAGED_MODULAR_CHAIN_ONLY=1`) via `scene_write.write_staged_scene_json`. |

Legacy-equivalent **staged** runs still use `full_run.run_legacy_equivalent_process_image`, which writes the **same** multi-track tree as legacy (`scene_graph/grounded_sam2/` …).

### Top-level keys (full track write)

When the monolithic track saver builds `sam2_scene_output` in `scene_understanding.py` (search **`sam2_scene_output = {`** immediately before writing `{stem}_scene.json`), the JSON typically contains:

| Key | Role |
|-----|------|
| **`metadata`** | Timestamp, intrinsics, active models, palette, and **relative paths** to companion artifacts (depth NPZ/NPY, region JSON/PNGs, segmentation overlays, caption JSONs, path bundle under `{stem}_paths/`, etc.). |
| **`objects`** | Final object list after `_sam2_mask_array` is stripped at save time; includes labels, bbox, depth stats, 3D coords, **`sources`** (GroundingDINO / Florence-2 / RAM++ / Pix2SG stubs). |
| **`relations`** | **Graph-global** relation triplets / records (object–object, Florence/Pix2SG layers, optional **region–region** edges appended when regions + `append_region_layer_relations` allow). |
| **`mask_hierarchy`** | Mask / containment hierarchy used for maps and downstream reasoning. |
| **`layers`** | Layer summary used with visualization exports. |
| **`regions`** | Present when depth-region partitioning is enabled: region list / stats bound into the same scene file. |

### Companion artifacts (same folder as `{stem}_scene.json`)

Non-exhaustive: `{stem}_relations.json`, `{stem}_relations_map*.png`, `{stem}_mask_hierarchy*.json`, `{stem}_layers.json`, `{stem}_regions*.json`, `{stem}_depth_mask_A.json` (when depth-mask export runs), `{stem}_paths/` subtree for traversability + path hypotheses + animation plans, caption bundles when hybrid captions are enabled. Paths appear both on disk and as **`metadata`** string fields.

### What is still a good extension (incremental)

The core scene file is now **much closer** to a portable scene graph than older revisions (top-level **`relations`**, **`regions`**, **`mask_hierarchy`**). Remaining wins tend to be **quality**, not missing top-level keys:

- Raw Pix2SG triplet dumps **before** dedup (debug / training).
- Richer **parent/child** edges inside objects (not only hierarchy sidecars).
- Normalized **relation_summary** or deduped edge index for analytics.

### Historical note

Earlier drafts of this doc claimed there was **no** top-level `relations` array and that **`dm_json`** was never written. The current `scene_understanding.py` track saver **does** emit **`relations`** alongside **`objects`** and **can** persist **`{stem}_depth_mask_{mode}.json`** when that branch executes.

## 7. Important Gaps and Risks Found in the Current Code

### 1. Documentation drift (partially addressed)

The README **Overview** and pipeline diagram should track **`_label_mask`**: **GroundingDINO → Florence-2 → RAM++** on crops, plus RAM++-driven query cues upstream — not a GRiT/YOLOv8 chain unless those are wired elsewhere.

### 2. Two truths: monolith vs slim staged

**`scene_understanding/pipeline.py`** is the supported entrypoint, but the heavy logic still lives in the root **`scene_understanding.py`** class body. When fixing bugs, confirm whether the repro uses **legacy**, **staged legacy-equivalent**, or **`CITV_STAGED_MODULAR_CHAIN_ONLY`** — outputs differ in completeness.

### 3. All masks are kept, including noisy ones

This helps recall but hurts graph quality because background and part masks enter the same object pool.

See [`config.py`](../config.py) post-filter keys for AMG.

### 4. Graph ids encode track name but still say `GroundedSAM2`

Detections receive:

- `det["graph_id"] = f"{track_key}_obj_{i}_GroundedSAM2"`

around the track assembly loop in `scene_understanding.py` (search **`graph_id`**).

even when the actual **`segmentor`** is **`SAM2_AMG`** or another source.

That can confuse provenance and debugging.

### 5. Relation output can be redundant

Top-level **`relations`** plus per-object **`sources.Pix2SG.relations`** can overlap; consumers should define a single canonical representation for their application.

## 8. Recommended Implementation Order

If the goal is "better and more accurate scene graphs" with minimal wasted effort, I would do the work in this order:

1. **Reduce noise in `objects`** (AMG filtering tiers, specificity gates on GDINO “almost specific” labels).

2. **Add object hierarchy and layered-mask bookkeeping** inside or beside the flat list:
   - parent/child
   - contains/inside
   - occlusion order

3. **Upgrade naming from hard priority to evidence fusion.**
   - noun phrase extraction
   - generic-label override
   - alias storage

4. **Expand relation candidate generation beyond overlapping masks.**
   - near-contact
   - containment
   - support
   - depth-aware adjacency

5. **Optional:** Raw triplet / debug exports for Pix2SG and relation fusion experiments.

6. **Integrate a panoptic scene-graph backend** (optional).
   - OpenPSG-style / PSGTR-style relation grounding
   - keep Pix2SG as fallback

## External References

- SAM 2 paper: https://arxiv.org/abs/2408.00714
- Grounding DINO paper: https://arxiv.org/abs/2303.05499
- Florence-2 paper: https://arxiv.org/abs/2311.06242
- RAM paper: https://arxiv.org/abs/2306.03514
- OpenPSG repo: https://github.com/Jingkang50/OpenPSG
- Pix2SG paper: https://arxiv.org/abs/2303.10944
