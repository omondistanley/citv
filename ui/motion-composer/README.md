# CITV Motion Composer UI

A browser UI plus local Python grounding API for user-authored, scene-adaptive
motion contracts.

The UI is intentionally additive. It does not replace the Python scene pipeline;
it lets users author motion intent, then calls a local backend endpoint that
returns a grounded contract for renderers and exporters.

## Product rule

User intent is the source of truth. Scene evidence is additive.

The user can set a start point, set an end point, draw the route between them,
write what should happen, upload an actor asset, upload their own animation,
select scene objects, mark objects as required/avoid/behind constraints, and
export a grounded contract. The UI does not silently replace the user path with
generated paths.

## Run with backend grounding

From the repo root:

```bash
python ui/motion-composer/server.py --host 127.0.0.1 --port 8088
```

Then open:

```text
http://127.0.0.1:8088/ui/motion-composer/
```

The server does two things:

- serves the static UI;
- exposes `POST /api/motion/ground`, which calls
  `scene_understanding.action_contracts.adapt_motion_contract_to_scene`.

## Browser-only fallback

The UI still works if the Python backend is not running. Uncheck:

```text
Use Python backend grounding when available
```

or just click `Generate Preview`; if the backend is unavailable, the UI falls back
to browser-only approximate grounding and records a warning in the exported JSON.

## Workflow

1. Upload an image.
2. Upload a CITV `{stem}_scene.json` file if available.
3. Upload an actor asset if the user has one.
4. Upload an animation if the user has one. This can be video, GIF, image sequence
   placeholder, Lottie/JSON, or any future animation reference format.
5. Choose a tool:
   - `Point`: one-point actions such as peek, shimmer, appear, sit, hold.
   - `Start / End`: first click sets start, second click sets end.
   - `Draw Path`: draw the desired route. If start/end are also set, the exported
     geometry fuses `start + drawn path + end`.
   - `Region`: area actions such as ripple, drift, swim, smoke, logo field.
   - `Select`: select nearby objects as interaction constraints.
6. Enter actor text and action/animation direction.
7. Mark scene objects as:
   - `required`: must interact with this object.
   - `avoid`: should avoid this object.
   - `behind`: should render behind this object when crossed.
8. Generate preview.
9. Export `grounded_motion_contract.json`.

## API contract

`POST /api/motion/ground`

Request shape:

```json
{
  "motion_contract": { "...": "MotionContract JSON" },
  "scene_graph": { "...": "optional CITV scene JSON" },
  "metric_depth_m": [[1.2, 1.3]],
  "region_label_map": [[1, 1]],
  "object_masks": {
    "object_id": [[false, true]]
  },
  "sample_count": 72
}
```

All grounding arrays are optional. If they are missing, the backend still returns
a contract, but it reports warnings such as missing metric depth or missing masks.
The endpoint also supports local repo-relative paths:

```json
{
  "metric_depth_path": "output_dir/depth_metric.npy",
  "region_label_map_path": "output_dir/regions.npy",
  "object_mask_paths": {
    "object_id": "output_dir/masks/object_id.npy"
  }
}
```

## Geometry model

The UI stores geometry as separate authored parts:

- `start_point`
- `end_point`
- `drawn_path_2d`
- `region_polygon_2d`

At preview/export time, it creates a fused path while still preserving the raw
parts:

```text
start + drawn_path_2d + end
```

This allows the user to specify intent at multiple levels:

- exact start/end anchors;
- exact drawn route;
- semantic action text;
- uploaded animation timing/style;
- scene object constraints.

## Uploaded animation model

Uploaded animation references are exported as:

```json
{
  "uploaded_animation_ref": "browser_uploaded_animation",
  "uploaded_animation": {
    "name": "...",
    "type": "...",
    "size_bytes": 12345,
    "retargeting_policy": "preserve_timing_and_style_then_ground_to_scene"
  }
}
```

The UI and backend preserve the animation reference and retargeting policy. Full
frame-level animation retargeting is still a renderer task: extract frames/masks,
preserve timing/style, then adapt position, scale, occlusion, and contact to the
scene-grounded path.

## What is grounded today

Backend grounding can use:

- scene JSON objects/regions;
- optional metric depth arrays;
- optional region label maps;
- optional object masks.

The browser fallback grounds approximately against:

- `objects[].bbox`
- `objects[].mask_centroid_2d`
- `objects[].canonical_name` / `objects[].label`
- `regions.regions[].centroid_2d_px`
- `regions.regions[].semantic_label`

Both paths export:

- exact raw start/end/drawn-path/region geometry;
- fused user polyline;
- resampled preview path;
- depth trace;
- support trace;
- occlusion hints;
- render layers: `in_front`, `partially_occluded`, `behind_object`;
- uploaded animation reference and retargeting policy;
- asset policy with `no_hard_coded_actor_fallback: true`;
- report showing preserved, adapted, warnings, and scores.

## Still remaining for full production realism

The current branch wires the UI to a grounding backend. The full product still
needs renderer/compositor work:

- real mask-level occlusion from CITV mask artifacts;
- actor/animation frame extraction and alpha mattes;
- animation retargeting along the grounded path;
- path optimization inside the user corridor using traversability maps;
- contact shadows and contact patches;
- color, blur, grain, and lighting match;
- multi-take timeline editor;
- final video/GIF export.

## Industry-standard design principles captured

- AR-style hit tests and anchors: taps become scene-grounded points, not just x/y.
- Semantic scene understanding: objects and regions become constraints.
- Depth/occlusion awareness: foreground objects can partially hide virtual actors.
- Non-destructive layer model: raw user geometry is preserved while adapted traces
  are stored separately.
- VFX-style holdout/compositing mindset: render layers and occluder IDs are
  first-class outputs, not hidden rendering side effects.
- Animation-editor mindset: user-authored key anchors, paths, timing, uploaded
  motion, and textual direction all coexist instead of competing.
- Open-vocabulary actor/action input: the UI does not hard-code a finite actor
  catalog.
