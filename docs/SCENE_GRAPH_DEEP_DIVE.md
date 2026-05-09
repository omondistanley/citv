# Scene Graph Deep Dive

This report is based on the code that currently runs from `scene_understanding.py`, not just the refactor scaffolding under `scene_understanding/`.

## Executive Summary

- The repo is not choosing between per-object and per-mask segmentation. It already uses both.
- The effective production path is a hybrid:
  - `GroundedSAM2` gives object-level instance masks.
  - `SAM2 AMG` gives mask-level "segment everything" regions.
  - The two are merged with IoU deduplication.
- For scene graphs, object-level segmentation is the better primary representation because graph nodes should correspond to entities, not arbitrary parts.
- Mask-level segmentation is still necessary as a recall layer for small, occluded, layered, and missed objects.
- The best direction for this codebase is not "pick one", but "make object-level masks the canonical graph nodes, and keep mask-level segments as supporting evidence, child parts, and recovery candidates."

## 1. How Segmentation Is Currently Done

### Effective runtime path

The actual orchestration lives in `scene_understanding.py`, especially:

- `SceneUnderstandingPipeline.process_image()` at `scene_understanding.py:2561`
- `_label_mask()` at `scene_understanding.py:2053`
- relation attachment at `scene_understanding.py:2912`
- final scene JSON save at `scene_understanding.py:2962`

The modular package mirrors this design, but the CLI and README point to the monolithic script.

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

- `run_both_segmentors = True`: `config.py:191`
- IoU dedup is `0.7`: `config.py:192`

The merge happens in `scene_understanding.py:2670`.

- First pass: `self.sam2_wrapper.generate(img_rgb)` at `scene_understanding.py:2668`
- Second pass: raw AMG fallback run again at `scene_understanding.py:2674`
- Extra AMG masks are kept if they do not overlap strongly with the first set: `scene_understanding.py:2685`

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

#### 2. Layered objects are not modeled explicitly

The pipeline stores a flat `objects` list. It does not preserve:

- parent/child part structure
- containment hierarchy
- occlusion order
- multiple candidate masks for one entity

#### 3. Depth-mask JSON is built but effectively discarded

`dm_json` is created in `scene_understanding.py:2811`, but it is not written anywhere.

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

The real runtime labeling priority chain is in `_label_mask()`:

- GDINO label wins first: `scene_understanding.py:2101`
- Florence-2 second: `scene_understanding.py:2117`
- RAM++ third: `scene_understanding.py:2131`

There is also whole-image RAM++ tagging that updates the GDINO query before segmentation:

- `scene_understanding.py:2645-2664`

### Current strengths

- If GDINO already has a specific class, that is usually the best low-latency label.
- Florence-2 gives richer open-vocabulary recovery.
- RAM++ improves vocabulary recall for unlabeled crops.

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

If GDINO says a broad label like `container`, `furniture`, or `electronics`, it still wins as long as it is not literally `object`:

- `scene_understanding.py:2104`

That blocks better crop-specific labels from Florence-2 or RAM++.

#### 3. There is no ontology normalization layer

The system stores one final `label`, but does not normalize:

- synonyms
- noun phrases
- singular/plural
- hypernym/hyponym conflicts

#### 4. Names are not fused across sources

You already store source evidence in `sources`, which is good:

- `scene_understanding.py:2891`

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

The runtime relation flow is:

- spatial scaffold in `scene_understanding/relations/pix2sg.py:227`
- optional Florence-2 enrichment in `scene_understanding/relations/pix2sg.py:355`
- attachment into object-local `sources.Pix2SG.relations` in `scene_understanding.py:2165`

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

If IDs fail, `_attach_relations_by_triplets()` falls back to substring label matching:

- `scene_understanding.py:2187-2227`

That is risky when there are multiple objects with the same label.

#### 4. The output graph is object-local, not graph-global

Relations end up nested under each object's source block:

- `sources.Pix2SG.relations`

There is no clean top-level `relations` array in the final saved JSON, even though that would be more scene-graph-native.

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

## 6. Where To Extend Outputs So Objects and Relations Are Saved Into `output_scene`

### What is already saved

The current final scene JSON is written here:

- `scene_understanding.py:3012-3014`

and lands at:

- `output_scene/scene_graph/{stem}_scene.json`

Objects already include per-object relation lists under:

- `objects[*].sources.Pix2SG.relations`

### What is not saved cleanly enough

#### 1. No top-level `relations` array

The final file saves:

- `metadata`
- `objects`

but not a graph-global `relations` list.

#### 2. `dm_json` is never written

The depth-mask detail structure is built here:

- `scene_understanding.py:2811`

but not persisted.

#### 3. `_save_sam2_outputs()` is mostly a placeholder

It only returns paths:

- `scene_understanding.py:2393-2414`

It does not actually serialize richer mask/object relation artifacts.

### Best extension points

#### A. Save top-level relations in the main scene JSON

Primary place:

- right before `sam2_scene_output` is created at `scene_understanding.py:3012`

Add a graph-global array such as:

- `relations`
- `relation_summary`

generated from the per-object `sources` blocks.

#### B. Save raw relation triplets separately

Primary place:

- immediately after `pix2sg_out` and `pix2sg_stats` are available: `scene_understanding.py:2916-2939`

Suggested path:

- `output_scene/scene_graph/relation_raw/{stem}_pix2sg_triplets.json`

This preserves source triplets before attachment and dedup.

#### C. Save the dropped depth-mask JSONs

Primary place:

- inside the `for mode in self.depth_mask_modes` loop at `scene_understanding.py:2754-2826`

Suggested path:

- `output_scene/scene_graph/depth_mask/{stem}_depth_mask_A.json`
- `output_scene/scene_graph/depth_mask/{stem}_depth_mask_B.json`

#### D. Save a hierarchy-aware object graph artifact

Best place:

- after `objects_3d` is fully assembled and before `_sam2_mask_array` is stripped: `scene_understanding.py:2872-3011`

Suggested path:

- `output_scene/scene_graph/{stem}_objects_relations.json`

This should contain:

- objects
- top-level relations
- containment / parent-child edges
- source evidence per object and relation

## 7. Important Gaps and Risks Found in the Current Code

### 1. Documentation drift

The README still describes a different labeling chain than the runtime code.

- README says `GDINO -> Florence-2 -> GRiT -> YOLOv8`
- runtime code is `GDINO -> Florence-2 -> RAM++`

See:

- `README.md`
- `scene_understanding.py:2053`
- `config.py:215-232`

### 2. The refactored package is not the real entrypoint yet

`scene_understanding/__init__.py` says there is a package-level `pipeline.py`, but the actual orchestration still runs from the top-level script.

That means future changes must be made carefully in the right place first.

### 3. All masks are kept, including noisy ones

This helps recall but hurts graph quality because background and part masks enter the same object pool.

See `config.py:63-68`.

### 4. AMG-derived objects get a misleading graph id suffix

Every detection gets:

- `det["graph_id"] = f"obj_{i}_GroundedSAM2"`

at `scene_understanding.py:2724`

even when the actual `segmentor` is `SAM2_AMG`.

That will make provenance and later debugging confusing.

### 5. Relation output is asymmetric and object-local

Relations are attached to source objects only, under source-specific blocks.

That is useful for provenance, but not ideal as the canonical scene graph serialization.

## 8. Recommended Implementation Order

If the goal is "better and more accurate scene graphs" with minimal wasted effort, I would do the work in this order:

1. Fix output serialization first.
   - Save top-level `relations`
   - Save raw triplets
   - Save the currently dropped `dm_json`

2. Add object hierarchy and layered-mask bookkeeping.
   - parent/child
   - contains/inside
   - occlusion order

3. Upgrade naming from hard priority to evidence fusion.
   - noun phrase extraction
   - generic-label override
   - alias storage

4. Expand relation candidate generation beyond overlapping masks.
   - near-contact
   - containment
   - support
   - depth-aware adjacency

5. Integrate a panoptic scene-graph backend.
   - OpenPSG-style / PSGTR-style relation grounding
   - keep Pix2SG as fallback

## External References

- SAM 2 paper: https://arxiv.org/abs/2408.00714
- Grounding DINO paper: https://arxiv.org/abs/2303.05499
- Florence-2 paper: https://arxiv.org/abs/2311.06242
- RAM paper: https://arxiv.org/abs/2306.03514
- OpenPSG repo: https://github.com/Jingkang50/OpenPSG
- Pix2SG paper: https://arxiv.org/abs/2303.10944
