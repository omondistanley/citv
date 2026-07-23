# Tier 3: AI-generative video — design contract (implementation deferred)

Status: **architecture only, no code**. Per the plan, actual model
integration is gated on having a GPU dev/test environment to evaluate
candidate models against, which isn't available where Tiers 1-2 were built.
This document is the contract Tier 3 must satisfy once that environment
exists, so a future implementation pass has a concrete target rather than
starting from a blank page.

## Problem framing: this is video inpainting, not video generation

Every other tier in this plan (Tier 1, Tier 2) treats the uploaded photo as
an immutable background plate and only ever draws the actor + its shadow on
top of it. Tier 3 must satisfy the same rule, but naive text-to-video or
image-to-video generation does not: those models regenerate the whole frame,
which means the "background" is a hallucinated re-synthesis of the photo,
not the photo itself — pixel drift, lighting drift, and object drift are all
expected side effects of that class of model, and would violate this
project's core photorealism requirement (see the plan's cross-cutting
requirement #1).

The correct framing is **background-preserving video inpainting**:

1. The real uploaded photo is frame 0, unmodified (LTX-Video-style
   first-frame conditioning — the model is given the actual pixels of frame
   0 and told to extend motion from it, not asked to imagine a frame 0).
2. A **mask** — the actor's region at each timestep, derived from this
   project's own path/pose data (see "Conditioning signals" below) — is the
   *only* region the model is allowed to regenerate per frame. Everything
   outside that mask must either be copied verbatim from the source photo
   (hard constraint, enforced in post-processing regardless of what the
   model outputs) or left to the model only within a thin feathered border
   for blend quality.
3. Reference architectures that already do exactly this: **InVi** (ControlNet-based
   inpainting diffusion, anchors each frame's inpainting on the *previous*
   generated frame for temporal consistency — prevents flicker) and
   **ReplaceAnyone** (image-conditioned video inpainting with pose guidance,
   a hybrid inpainting encoder specifically built to preserve background
   detail, and diverse mask shapes to prevent the actor's silhouette from
   leaking scene information it shouldn't have). **AniCrafter** (insert +
   animate a character into a background video via a pretrained image-to-video
   diffusion transformer conditioned on "reference image + target video") is
   a plausible alternative architecture family if InVi/ReplaceAnyone's public
   weights turn out to be unavailable/unlicensed for this use.

## Request contract (reuses existing schema fields, invents nothing new)

```json
{
  "schema": "citv_tier3_generative_request_v1",
  "source_photo": "<base64 or object storage ref — the real, unmodified upload>",
  "actor": {
    "actor_text": "open-vocabulary description, e.g. 'a photorealistic red fox'",
    "asset_ref": "optional — an uploaded reference image of the actor, if provided (Phase 3 contract)",
    "visual_style": "photorealistic"
  },
  "path_hypothesis": {
    "polyline_2d": "... from path_hypotheses.json v3, unchanged",
    "depth_trace_m": "... unchanged — per-frame depth conditions perspective/scale",
    "visibility_profile": "... unchanged — render_layer + occluder_ids define per-frame inpainting mask geometry",
    "kinematic_signatures": "... unchanged — segment motion labels (walk/run/jump/climb/descend/crawl) condition the pose/motion guidance signal"
  },
  "fps": 24,
  "duration_s": 4.0
}
```

Nothing here is a new field invented for Tier 3 — every conditioning signal
already exists in the `path_hypotheses.json` v3 schema `paths_export.py`
produces (Phase 1) and that Tiers 1-2 already consume. This is deliberate:
Tier 3 should be a third *consumer* of the same contract, not a fork of it.

### Deriving the per-frame inpainting mask

The mask a ControlNet-inpainting model is allowed to touch, per frame, is
built the same way Tier 2's compositor places its sprite: the actor's real
segmentation mask (if animating a detected scene object) or a generated
actor-region estimate (if the actor is purely text-described, in which case
the mask is a soft ellipse/silhouette placeholder at the path's
depth-scaled width — see `paths_export.py`'s `width_profile_px` — sized to
roughly the actor's expected on-screen footprint) positioned at
`polyline_2d[i]`, dilated by a small margin for the inpainting model's own
blend region. `visibility_profile[i].render_layer`/`occluder_ids` still
apply: where `render_layer == "behind_object"`, the occluder's real pixels
must be composited back on top in post-processing exactly like Tiers 1-2 do
— **never trust the diffusion model's own occlusion handling** for the hard
constraint; treat any occlusion it produces as a bonus, and enforce
correctness with the same post-processing restore step used elsewhere.

## Response contract

```json
{
  "schema": "citv_tier3_generative_response_v1",
  "video_path": "mp4, background-pixel-identical to source_photo outside the actor mask region",
  "per_frame_mask_used": "for QA — lets a reviewer verify the model didn't touch background pixels",
  "model_id": "whichever backend model actually ran",
  "generation_time_s": 0.0
}
```

## Deployment: separate, stateless, scale-to-zero — never on the always-on CPU VM

Per Phase 4's cost constraint (avoid idle GPU billing on the always-on box),
Tier 3 must be its own service:

- A stateless HTTP endpoint (`POST /tier3/generate`, matching the request/response
  contract above) deployed as a Cloud Run service with a GPU, or a GKE
  Autopilot node pool with a GPU node pool that scales to zero when idle.
- The main app (Tiers 1-2, CPU-only, always-on per Phase 4) calls this
  endpoint only when the user explicitly selects Tier 3 (or Tier 1/2 aren't
  applicable — e.g. a fully open-vocabulary actor with no rigged archetype
  match, see Phase 2's Tier 1 archetype-library gap) — never as a default,
  since every invocation costs real GPU-minutes.
- Cold-start latency (a scale-to-zero GPU node taking 30s-2min to spin up)
  should be surfaced to the user as an explicit wait state in the UI, not
  hidden — this is a real, user-visible cost of choosing Tier 3 over the
  always-warm Tiers 1-2.

## Open decisions for the next implementation pass

1. **Model selection** — InVi/ReplaceAnyone/AniCrafter are the closest
   published architectures to this exact problem, but none has been run or
   benchmarked here; picking one (or finding their public weights don't
   exist/aren't licensed for this use, and falling back to a from-scratch
   ControlNet-inpainting pipeline built on a general video diffusion base
   like LTX-Video or Stable Video Diffusion) requires a GPU environment to
   actually load and test candidate models — not assessable from
   architecture papers alone.
2. **Mask quality for open-vocabulary actors** — when there's no real scene
   object to derive a mask from (a purely text-described actor), the
   ellipse/silhouette placeholder above is a reasonable v1, but a
   text-to-segmentation step (e.g. asking an open-vocabulary segmentation
   model to propose actor silhouette given the actor description + a rough
   placement) would likely produce better inpainting masks — worth
   evaluating once a GPU environment exists to test both approaches.
3. **Temporal consistency budget** — InVi's previous-frame-anchoring and
   ReplaceAnyone's hybrid inpainting encoder both target flicker reduction,
   but neither is free; actual flicker/quality needs to be measured against
   real output once a candidate model is running, not assumed from the papers.
