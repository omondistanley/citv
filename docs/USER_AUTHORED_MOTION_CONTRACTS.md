# User-Authored Scene-Adaptive Motion Contracts

This document defines the implementation contract for user-directed animation,
product placement, uploaded animations, and open-vocabulary actor/action editing.

## Product rule

User intent is the source of truth. Scene evidence is additive.

A user-authored path, tap point, region, uploaded animation, actor description,
or interaction requirement must never be silently replaced by a generated path or
hard-coded actor. The system may adapt previews to the scene, but it must report
what it preserved, what it adapted, and what remains uncertain.

## New package

`scene_understanding/action_contracts/`

- `contracts.py` contains the dataclasses used by UI, backend, and renderers.
- `json_extraction.py` replaces brittle greedy JSON regex parsing.
- `scene_adapter.py` enriches user-authored contracts with scene evidence.

## Data flow

```text
user taps / draws / uploads animation
        ↓
MotionContract
        ↓
adapt_motion_contract_to_scene(...)
        ↓
GroundedMotionContract
        ↓
renderer / exporter / UI take manager
```

## MotionContract

A `MotionContract` stores the creative instruction before grounding:

```json
{
  "contract_id": "take_user_001",
  "actor": {
    "actor_text": "photorealistic cobra",
    "actor_source": "generated_asset",
    "asset_ref": "assets/cobra.png",
    "visual_style": "photorealistic"
  },
  "action_text": "cobra slithers along the drawn path then rears up",
  "user_geometry": {
    "mode": "polyline",
    "points": [[590, 150], [610, 220], [597, 750]],
    "source": "user_drawn",
    "corridor_radius_px": 28
  },
  "duration_s": 4.0,
  "source": "user_authored"
}
```

Important fields:

- `actor.actor_text` is open vocabulary. Do not restrict to snake/bird/bunny/person.
- `actor.asset_ref` can point to an uploaded image, generated/retrieved asset, or cutout.
- `user_geometry.points` is the raw user input and must remain unchanged.
- `uploaded_animation_ref` can point to a user-supplied animation that should be retargeted, not replaced.
- `policy` distinguishes hard creative constraints from soft scene adaptation constraints.

## GroundedMotionContract

`adapt_motion_contract_to_scene` returns a `GroundedMotionContract` with:

- `grounded_geometry.user_polyline_2d`: exact user path, preserved.
- `grounded_geometry.adapted_polyline_2d`: resampled/adaptation-ready preview path.
- `traces.depth_trace_m`: sampled metric depth along the path.
- `traces.support_trace`: region/support labels along the path.
- `traces.visibility_profile`: approximate visibility values for occlusion-aware renderers.
- `traces.occluder_ids`: masks hit along the path.
- `rendering.asset_policy.no_hard_coded_actor_fallback`: explicit guardrail.
- `report`: preserved/adapted/warnings/scores.

The adapter is intentionally conservative. It does not pretend to solve full
physics or final compositing. Its job is to produce a transparent scene-grounded
contract that renderers can consume.

## Robust JSON parsing

Do not parse model output with greedy regexes such as:

```js
/\[[\s\S]*\]/
```

That fails when the model returns an object with array fields and can produce
invalid fragments like:

```text
[[590, 150], [610, 220]], "depth_trace_m": [2.1, 2.3]
```

Use `extract_first_json_object` instead. It scans for a balanced JSON object while
respecting strings and escapes. It rejects naked arrays for grounding contracts,
because grounding output should be named-field JSON, not anonymous coordinate
lists.

## Scene adaptation policy

The adapter uses the following hierarchy:

### Hard creative truth

- actor text or uploaded asset
- action text
- raw user geometry
- requested interaction object ids
- requested avoid object/region ids
- requested behind-object behavior
- uploaded animation timing/reference

### Additive scene truth

- depth samples
- support/region trace
- occluder masks
- render layers
- nearest scene entities
- confidence/warnings

### Renderer responsibilities

A renderer can use the grounded contract to:

- preserve the user's path while bending inside the corridor when allowed;
- scale the actor using depth traces;
- mask the actor behind occluders;
- generate contact shadows and product-placement compositing;
- retarget uploaded/drawn animation onto the grounded path;
- show warnings instead of silently correcting user intent.

## Tests

`tests/test_action_contracts.py` covers:

- balanced JSON extraction from prose;
- rejection of naked coordinate arrays;
- open-vocabulary actor contract construction;
- preservation of raw user geometry;
- depth/support/visibility trace output;
- hard guardrail against hard-coded actor fallbacks.
