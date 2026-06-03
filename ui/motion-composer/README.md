# CITV Motion Composer UI

A self-contained browser UI for user-authored, scene-adaptive motion contracts.

This UI is intentionally additive and build-free. Open `index.html` in a browser
or serve this folder with any static file server. It does not replace the Python
pipeline; it creates and previews the same kind of contract that backend
renderers can later consume.

## Product rule

User intent is the source of truth. Scene evidence is additive.

The user can set a start point, set an end point, draw the route between them,
write what should happen, upload an actor asset, upload their own animation,
select scene objects, mark objects as required/avoid/behind constraints, and
export a grounded contract. The UI does not silently replace the user path with
generated paths.

## Run

From the repo root:

```bash
python -m http.server 8088
```

Then open:

```text
http://localhost:8088/ui/motion-composer/
```

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

## Geometry model

The UI now stores geometry as separate authored parts:

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

The browser preview does not yet retarget the uploaded animation frames. It
preserves the reference so the backend renderer can retarget the uploaded timing,
style, and motion onto the scene-grounded path.

## UI layout

The UI follows a canvas-first professional creative-tool layout:

- top bar for ingest/export;
- left tool rail for direct manipulation;
- center canvas for image, start/end anchors, paths, regions, objects, actor
  preview, and occlusion;
- bottom timeline for preview timing;
- right inspector for actor/action/constraints/contract JSON.

This matches how AR/VFX tools are usually operated: the user manipulates the
scene first, then the system exposes structured constraints and diagnostics.

## What is grounded today

The browser prototype can ground against scene JSON fields that are already
common in CITV outputs:

- `objects[].bbox`
- `objects[].mask_centroid_2d`
- `objects[].canonical_name` / `objects[].label`
- `regions.regions[].centroid_2d_px`
- `regions.regions[].semantic_label`

It exports:

- exact raw start/end/drawn-path/region geometry;
- fused user polyline;
- resampled preview path;
- approximate depth trace;
- support trace from nearest region labels;
- object-box occlusion hints;
- render layers: `in_front`, `partially_occluded`, `behind_object`;
- uploaded animation reference and retargeting policy;
- asset policy with `no_hard_coded_actor_fallback: true`;
- report showing preserved, adapted, warnings, and scores.

## What the backend should do next

The UI is designed to hand off to `scene_understanding.action_contracts` and the
Python scene adapter added in the same branch. The next backend integration
should replace the browser approximations with:

- metric depth sampling from `depth_metric.npy` or cached depth map;
- real mask-level occluder sampling instead of bbox-only checks;
- region label map sampling;
- nearest object/contact/approach/occlusion anchors;
- path bending inside the user corridor using traversability maps;
- product-placement compositor hooks for shadows, blur, color match, grain, and
  holdout masks;
- uploaded animation retargeting that preserves timing/style while adapting
  position, scale, occlusion, and contact to the scene.

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
