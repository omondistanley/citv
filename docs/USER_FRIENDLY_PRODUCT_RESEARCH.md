# CITV User-Friendly Product Research

This note translates the current CITV pipeline into a product shape a user can operate. It is grounded in the existing code, generated artifacts, and the design direction in `docs/path_updates.md`.

## What The Project Does Today

CITV turns one or more RGB images into a structured, depth-aware scene graph. For each image, it can produce:

- metric depth in metres;
- camera intrinsics and optional undistortion;
- object and mask segmentation from GroundedSAM2 plus SAM2 AMG;
- per-mask depth geometry and 3D coordinates;
- open-vocabulary object labels from GroundingDINO, Florence-2, and RAM++;
- object relations from Pix2SG-style geometry plus Florence captions;
- depth regions, region adjacency, layers, and mask hierarchy;
- path hypotheses and animation-related artifacts.

The practical output is a folder of JSON and PNG files under paths like:

- `scene_graph/<track>/{stem}_scene.json`
- `scene_graph/<track>/{stem}_paths/path_hypotheses.json`
- `scene_graph/<track>/{stem}_paths/animation_plan.json`
- `scene_graph/<track>/{stem}_paths/trajectory_hypotheses.json`
- `scene_graph/<track>/{stem}_paths/insertion_path_ensembles.json`
- `scene_graph/<track>/{stem}_paths/path_context_top5.png`
- `scene_graph/<track>/{stem}_paths/motion_contracts_overlay.png`

The repo is strongest as a local scene-analysis and motion-planning engine. It is not yet a user-facing creation tool.

## Current Product Gap

The core outputs are powerful but developer-shaped. A non-technical user currently has to understand file names, config flags, path IDs, object IDs, JSON structures, and generated overlays before they can make anything.

The user wants a creation surface:

1. Provide a scene image.
2. Provide or choose an actor/effect image.
3. Tap where the actor starts.
4. Tap where it should go or what it should interact with.
5. Choose or describe an action.
6. Preview the animation.
7. Export a usable motion/action contract.

That flow should sit on top of the existing pipeline. The model stack already knows enough about masks, regions, depth, relations, and candidate paths to power the interface.

## Key Insight From `path_updates.md`

The most important design point is that animation is not the same as shortest-path search.

A user might ask for:

- walk from person to door;
- fly above a crowd;
- disappear into a shadow;
- orbit around an object;
- shimmer on glass;
- ripple across water;
- hold, push, grab, or lean on something;
- emerge from behind an occluder.

Only some of those are `polyline_2d` routes. The product should expose actions, not just paths.

The internal contract should allow these manifold types:

- `centerline_path`
- `ribbon_path`
- `blob_path`
- `volume_path`
- `contour_path`
- `interior_path`
- `portal_path`
- `occlusion_pulse`
- `contact_patch`
- `effect_field`

This is the bridge between "tap two things" and "make anything imaginable."

## Recommended User Model

### Primary creation loop

Use a three-part loop:

1. Select actor or effect.
2. Select target, path, object, region, or point.
3. Define action.

The app then resolves that into:

- selected source entity or point;
- selected target entity or point;
- chosen action;
- inferred manifold type;
- best existing path hypothesis, if one matches;
- fallback generated path if no existing path matches;
- animation preview;
- exportable action contract.

### Interaction modes

The user should not need to see raw IDs first. They should see selectable boxes, masks, and path overlays.

Recommended modes:

- `Source`: tap the actor/object/start point.
- `Target`: tap the target object/region/end point.
- `Waypoints`: optional path shaping.
- `Action`: choose or type what happens.
- `Preview`: inspect motion before export.
- `Export`: produce JSON for rendering or downstream generation.

### Progressive disclosure

Default view:

- image canvas;
- detected object boxes;
- two selection states;
- action selector;
- preview button;
- export button.

Advanced view:

- path confidence;
- source path ID;
- semantic reasons;
- traversability field;
- depth scale and occlusion flags;
- raw scene/path JSON references.

## MVP Architecture

### Layer 1: Thin local UI

Build a static browser UI that can load:

- a scene image;
- optional actor/effect image;
- `{stem}_scene.json`;
- `path_hypotheses.json`;
- optional `animation_plan.json`.

The UI should allow tapping objects or free points, previewing a selected route, and exporting a contract.

This is low-risk because it does not change the heavy Python model pipeline.

### Layer 2: Action contract

The output should be stable even while internals evolve:

```json
{
  "schema": "citv_user_action_contract_v0",
  "source": {"kind": "object", "id": "obj_0", "xy": [120, 400]},
  "target": {"kind": "object", "id": "obj_3", "xy": [420, 260]},
  "action": {
    "preset": "walk",
    "prompt": "walk behind the chair, then look back",
    "manifold_type": "ribbon_path"
  },
  "trajectory": {
    "polyline_2d": [[120, 400], [190, 370], [420, 260]],
    "duration_s": 3.2,
    "fps": 24
  },
  "evidence": {
    "path_id": "opath_example",
    "confidence": 0.76,
    "source_file": "path_hypotheses.json"
  }
}
```

### Layer 3: Pipeline-backed authoring

Once the UI proves the interaction model, wire it to a backend command:

- upload or choose an image;
- run `scene_understanding.py`;
- stream progress by stage;
- load the generated scene/path files;
- let the user author and export.

This can be done with a small Python web server later. The first prototype can stay static and read existing artifacts.

## What To Improve In The Core Pipeline

### 1. Promote action manifolds to first-class JSON

`animation_plan.json` currently turns ranked paths into simple idle/walk/run sequences. Add an action-level file:

- `action_contracts.json`
- `action_hypotheses.json`
- `manifold_candidates.json`

Each record should include:

- `action_prompt`;
- `action_family`;
- `manifold_type`;
- source/target anchors;
- path or area geometry;
- depth trace;
- visibility trace;
- support/occlusion evidence;
- render hints.

### 2. Add object and region anchors

Centroids are often wrong for animation. Add:

- approach points;
- contact points;
- support patches;
- portal thresholds;
- occlusion boundary points;
- contour handles.

These are what the UI should snap to when a user taps near an object.

### 3. Use captions as structured evidence

The repo already produces captions, labels, RAM++ tags, relations, and region descriptions. Feed them into affordance scoring as structured inputs, not just prose.

For example:

- "glass door" should score for `portal`, `reflective_transparent`, and `interaction_target`.
- "road" should score for `support` and `drive/walk`.
- "water" should score for `liquid`, `ripple`, `swim`, and `float`.
- "shadow" or "behind" should score for `occlusion_pulse` or `portal_path`.

### 4. Preserve fallback freedom

"Anything imaginable" means the system should not reject creative actions too early. It should classify the action into a best-fit manifold and mark uncertainty.

Example:

- prompt: "make the light leak out of the window and curl around the person"
- action family: `effect`
- manifold: `effect_field` plus `contour_path`
- geometry source: window mask, person boundary, generated bezier route
- evidence: low physical plausibility, high effect plausibility

### 5. Separate product modes

Recommended modes:

- `Quick`: tap actor, tap target, choose action, preview.
- `Studio`: edit waypoints, path type, depth scale, occlusion.
- `Developer`: inspect JSON, scores, masks, traces.

## Immediate MVP Built In This Repo

A dependency-free browser prototype lives in `ux_demo/`.

It loads the bundled sample or user-selected files, draws object boxes and path previews, supports source/target tapping, animates an uploaded actor image or default marker, and exports a CITV action contract.

This is not the final product. It is the correct first surface for making the existing work usable.

## Next Implementation Steps

1. Move path/action logic out of the root monolith into staged modules.
2. Add first-class `action_contracts.json`.
3. Add object anchor extraction.
4. Add action prompt to manifold classifier.
5. Add a local backend that runs the pipeline from the UI.
6. Add GIF/MP4 export after preview.
7. Add render-time depth ordering from masks and depth traces.
