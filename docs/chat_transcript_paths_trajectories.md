# Pathing and Trajectory Chat Transcript (Hybrid)

This document captures a hybrid transcript of the planning discussion: structured decisions plus selected verbatim requirement excerpts.

## 1) Objective Summary

The requested upgrade is a CPU-first pathing and trajectory pipeline that is:

- strongly semantic-aware,
- scene/context-aware,
- depth and 3D sensitive,
- relation-aware,
- faithful to meandering path and trajectory geometry,
- capable of producing richer overlay outputs for interpretation,
- and performant under a strict `<200s` per-image CPU-only target.

## 2) Key Verbatim Requirement Excerpts

Selected user excerpts (preserved verbatim):

- \"we should not be using any hard coded values, in fact all across hard coded values are our enemy\"
- \"reject plain pixel shortest path, use geodesic/optimal control on a multi-layer semantic-physical cost field with constraints\"
- \"we want all trajectories, bot per mask, mask-object, object-object, region-mask, Region-to-region, region-to-object, object-to-regions\"
- \"the polyline shouldnt be curvature constrained, or maniforld constrained, it should always be able to map out the exact path/trajectory\"
- \"we also need to ensure we are processing the images as fast as possible possibly in less than 200s, and we are not using a GPU\"
- \"infact on the plan, the first step should be checking the specs of the machine being used\"

## 3) Agreed Technical Direction

### Planning and Scoring

- Replace hardcoded semantic/hybrid decision thresholds with calibrated, data-derived policies.
- Use semantic-physical energy minimization with hard constraints.
- Keep FMM/geodesic search as a solver substrate, but optimize rich scene-aware energy terms rather than naive pixel distance.
- Feed semantics, depth, 3D geometry, and relations directly into constraints and ranking.

### Geometry Fidelity

- Rendering must preserve exact planner/trajectory geometry.
- No render-time smoothing, curvature constraints, or manifold constraints that alter the original polyline/state sequence.
- Meanders are meaningful and must remain visible.

### Output Expectations

- Keep line-only ranked atlas panels.
- Expand atlas to 5 panels (50 paths at 10 per panel by default).
- Add ranked-panel overlays on original input image for interpretability.
- Maintain context overlays (`path_context_top5`, trajectory context variants).
- Add `descriptions.md` per image with path/trajectory formation rationale.

### Coverage Expectations

Target directional families include:

- mask->mask, mask->object, mask->region
- object->object, object->mask, object->region
- region->region, region->mask, region->object

## 4) Contract and Schema Expectations

- Produce machine-readable input/output contract docs.
- Introduce canonical `SceneContext` representation that explicitly includes scene graph context, while preserving in-memory high-fidelity fields.
- Extend path and trajectory schemas with calibrated evidence, constraints, and provenance fields.

## 5) Performance and Runtime Guardrails

- CPU-only benchmark mode required for acceptance testing.
- Stage-level timing checkpoints and wall-time pass/fail gate.
- Hard pass criterion: `<200s` wall time per image for benchmark mode.
- Optimization loop to target major regressors (segmentation, path export, geometry exports, captions).

## 6) Delivery Notes

- This transcript is linked from `thoughts.md` under \"Planning Tabs\".
- It is intended to remain as design context for implementation and regression reviews.
