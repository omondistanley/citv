# CITV Motion Composer UI

A self-contained browser UI for user-authored, scene-adaptive motion contracts.

This UI is intentionally additive and build-free. Open `index.html` in a browser
or serve this folder with any static file server. It does not replace the Python
pipeline; it creates and previews the same kind of contract that backend
renderers can later consume.

## Product rule

User intent is the source of truth. Scene evidence is additive.

The user can tap, draw, upload an actor asset, select scene objects, mark objects
as required/avoid/behind constraints, and export a grounded contract. The UI does
not silently replace the user path with generated paths.

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
4. Choose a tool:
   - `Point`: one-point actions such as peek, shimmer, appear, sit, hold.
   - `Draw Path`: path actions such as walk, run, roll, slither, drive.
   - `Region`: area actions such as ripple, drift, swim, smoke, logo field.
   - `Select`: select nearby objects as interaction constraints.
5. Enter actor text and action text.
6. Mark scene objects as:
   - `required`: must interact with this object.
   - `avoid`: should avoid this object.
   - `behind`: should render behind this object when crossed.
7. Generate preview.
8. Export `grounded_motion_contract.json`.

## UI layout

The UI follows a canvas-first professional creative-tool layout:

- top bar for ingest/export;
- left tool rail for direct manipulation;
- center canvas for image, paths, regions, objects, actor preview, and occlusion;
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

- exact raw user geometry;
- resampled preview path;
- approximate depth trace;
- support trace from nearest region labels;
- object-box occlusion hints;
- render layers: `in_front`, `partially_occluded`, `behind_object`;
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
- optional uploaded animation retargeting.

## Industry-standard design principles captured

- AR-style hit tests and anchors: taps become scene-grounded points, not just x/y.
- Semantic scene understanding: objects and regions become constraints.
- Depth/occlusion awareness: foreground objects can partially hide virtual actors.
- Non-destructive layer model: raw user geometry is preserved while adapted traces
  are stored separately.
- VFX-style holdout/compositing mindset: render layers and occluder IDs are
  first-class outputs, not hidden rendering side effects.
- Open-vocabulary actor/action input: the UI does not hard-code a finite actor
  catalog.
