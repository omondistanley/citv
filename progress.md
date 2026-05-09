# CITV Progress Report

This document summarizes implementation progress against `docs/path_updates.md`, including the latest updates from this chat.

---

## 1) Objective and Methodology

### Objective

Move CITV from a centroid-to-centroid pathing system toward a grounded, open-vocabulary, action-manifold pipeline that is:

- scene-aware (scene/object/mask evidence),
- depth/geometry-aware,
- manifold-aware (not only centerlines),
- animation-contract aware (renderability, occlusion, visibility, motion style),
- and reviewer-auditable (JSON/MD/atlas/video artifacts).

### Methodology used

Implementation followed a staged, additive migration strategy:

1. Keep existing outputs working.
2. Add new grounding and manifold artifacts before changing acceptance logic.
3. Propagate new contracts through path -> action -> trajectory -> animation -> QA.
4. Replace brittle binary decisions with evidence-weighted, manifold-specific scoring.
5. Add compliance and visual QA so quality regressions are detectable.
6. Add final consolidated export (`paths.json`) for downstream consumers.

---

## 2) Progress by Subsystem

## 2.1 Grounded Scene Index + Open Vocabulary

### What was implemented

- New module: `scene_understanding/pathing/scene_grounding_index.py`
  - `build_scene_grounding_index(...)`
  - Fuses object/region/relation + affordances + captions into unified entities.
- New module: `scene_understanding/pathing/open_vocab_grounding.py`
  - `concept_score(...)`
  - `dynamic_concepts_from_evidence(...)`
  - `build_open_vocab_grounding(...)`

### Methodology

- Build canonical entity records with:
  - roles/actions/path modes,
  - geometry anchors and mask-derived geometry,
  - evidence text across labels/captions/tags.
- Score concepts using seed + dynamic concept terms from evidence tokens.
- Explicitly encode uncertainty/contradictions (e.g., weak labels, support-vs-obstacle conflicts).

### Results and outcome

- Path generation now has a reusable grounding layer (`scene_grounding_index.json`) instead of ad-hoc local checks.
- Open-vocabulary signals become first-class priors (`open_vocab_grounding.json`), reducing dependence on hardcoded labels.

### Why this matters

- This directly addresses `docs/path_updates.md` sections on open-vocab grounding and scene/object/mask evidence fusion.
- It enables scalable action interpretation for unseen labels/synonyms.

### Next steps

- Add richer phrase-level normalization and cross-modal contradiction scoring.
- Add direct user action-text grounding path (currently mostly evidence-driven, not full intent-first text input).

---

## 2.2 Affordance Rasterization (Soft Evidence Fields)

### What was implemented

- New module: `scene_understanding/pathing/affordance_rasters.py`
  - `build_affordance_rasters(...)`
  - `save_affordance_rasters(...)`
- Raster channels include support, blocker, contact target, occlusion edge, portal, open-air, interior, uncertainty, depth discontinuity.

### Methodology

- Paint soft per-pixel evidence from grounding entities and masks.
- Blend existing support/obstacle masks with role/action-derived scores.
- Add depth-discontinuity-derived occlusion edge field.

### Results and outcome

- Pathing now has continuous evidence fields instead of binary route assumptions.
- Manifest (`affordance_rasters_manifest.json`) captures channel shape/range/nonzero coverage.

### Why this matters

- Supports manifold-specific plausibility and local grounding checks.
- Reduces false hard-rejections for non-ground manifolds (fly/swim/occlusion/effect cases).

### Next steps

- Increase coupling of these channels into full FMM cost synthesis (beyond current soft obstacle and support usage).

---

## 2.3 Manifold Candidate Generation

### What was implemented

- New module: `scene_understanding/pathing/manifold_candidate_generation.py`
  - `build_manifold_candidates(...)`
- Generates grounded candidates for `contact_patch`, `occlusion_pulse`, `volume_path`, `ribbon_path`, `contour_path`.

### Methodology

- Candidate manifold type chosen from entity role/action evidence.
- Candidate confidence from local grounding evidence.
- Includes grounding evidence block + routing provenance per candidate.

### Results and outcome

- Pipeline no longer relies solely on object-pair geodesics as starting point.
- Introduces local-action-manifold proposals early in generation.

### Why this matters

- Aligns with `path_updates.md` “action manifold types” and “not every action is a polyline”.

### Next steps

- Expand candidate generators for interior/portal/effect from richer mask geometry and region topology.

---

## 2.4 Path Export Core Integration

### What was implemented

In `scene_understanding/stages/paths_export.py`:

- Added grounded artifact orchestration:
  - scene grounding index
  - open-vocab grounding
  - affordance rasters + manifest
  - manifold candidates
- Enhanced support trace + semantic trace with channel-level evidence.
- Added `ground_object_classification` and `trajectory_contract`.
- Added many additive fields into exported hypotheses.

### Methodology

- Generate grounding artifacts before path ranking.
- Sample support channels along path geometry.
- Inject scene/object/mask evidence into path-level traces and scores.

### Results and outcome

- Richer `path_hypotheses.json` contracts, more complete `per_path` records.
- Better local evidence representation for acceptance and animation stages.

### Why this matters

- Closes parity gap between design goals and staged path exporter behavior.

### Next steps

- Further improve candidate diversity when local evidence is weak but non-contradictory.

---

## 2.5 Manifold-Fit Scoring + Acceptance Model

### What was implemented

- New module: `scene_understanding/pathing/manifold_fit_scoring.py`
  - channel means
  - manifold support fit
  - contradiction score
  - geometry contract score
  - renderability and uncertainty scores
- In `paths_export.py`, integrated manifold-fit into `_update_path_scores(...)`.
- Added manifold-specific thresholds in `_manifold_acceptance_thresholds(...)`.
- Reworked `_path_contract_status(...)` with statuses:
  - `accepted`
  - `plausible_uncertain`
  - `low_confidence`
  - `rejected`

### Methodology

- Score route/manifold consistency by manifold type, not one-size-fits-all.
- Separate uncertainty from contradiction.
- Promote uncertain-but-viable paths to `plausible_uncertain` instead of immediate rejection.

### Results and outcome

- Better status semantics and improved explainability.
- Reduced over-pruning of candidates that are geometrically plausible but evidence-limited.

### Why this matters

- Critical for increasing useful path pool and avoiding brittle binary filtering.

### Next steps

- Calibrate thresholds per scene domain and actor profile.
- Add confidence calibration reports over fixtures.

---

## 2.6 Action Export Propagation

### What was implemented

In `scene_understanding/stages/action_export.py`:

- Action status now aligns with path status model (includes `plausible_uncertain`).
- Added grounding metadata propagation:
  - `source_evidence`
  - `uncertainty_reasons`
  - `contradiction_reasons`
- Updated summary counts to include plausible status.

### Methodology

- Use path contract as authoritative source whenever available.
- Preserve action-level fallback logic where needed.

### Results and outcome

- `action_hypotheses.json` now preserves path evidence chain.
- Easier downstream debugging of why an action was accepted/rejected.

### Why this matters

- Action stage now reflects evidence quality instead of flattening it.

### Next steps

- Wire direct action-intent text compilation fully (currently partial per roadmap).

---

## 2.7 Animation Plan + Trajectory Contract Improvements

### What was implemented

In `scene_understanding/stages/animation_export.py`:

- Default QA modes include 24 and 120 fps (`path_animation_qa_modes` robust parsing).
- Motion inference refactored away from hardcoded motion lists:
  - `_ground_object_classification_from_path(...)`
  - `_primary_motion_from_path(...)`
  - `_motion_candidates_from_path(...)`
- Animation plan includes dynamic `motion_mode_candidates`.
- Propagates uncertainty/contradiction/grounding context.

### Methodology

- Derive motion from ontology + support priors + semantic hints + path contracts.
- Keep fallback behavior but make defaults ontology-driven.

### Results and outcome

- Better motion selection for walk/swim/fly-like contexts.
- More scalable behavior as vocabulary grows.

### Why this matters

- Directly addresses scalability and “remove hardcoded semantic motion behavior”.

### Next steps

- Add actor-profile conditioning (size/type capability constraints) into speed/motion policy.

---

## 2.8 Visual QA + Per-Path Exports

### What was implemented

In `scene_understanding/pathing/path_visual_qa_export.py`:

- `path_visual_qa.json` now includes:
  - full status counts including plausible
  - `ground_object_classification_summary`
  - expanded per-path summaries
- `per_path/*.json` and `per_path/*.md` include grounding/uncertainty/contradiction and contracts.

### Methodology

- Build QA payload from ranked path records + trajectory/action linkage maps.
- Keep reviewer-first readability in markdown while preserving full JSON detail.

### Results and outcome

- Stronger reviewer tooling and traceability.

### Why this matters

- Enables fast validation loops and concrete artifact-level audits.

### Next steps

- Add diff-friendly QA snapshots for regression comparisons between runs.

---

## 2.9 Compliance Tracking

### What was implemented

In `scene_understanding/pathing/path_contract_compliance.py`:

- Added plausible-status accounting.
- Added checks for grounding and raster artifacts availability.
- Updated phase checks for:
  - grounding availability
  - geometry contract quality
  - local evidence sufficiency
  - render-contract to trajectory linkage

### Methodology

- Evaluate artifacts from `*_paths` and sibling staged outputs.
- Convert checks into phase-level met/partial/missing status.

### Results and outcome

- Compliance report now reflects the new architecture and status model.

### Why this matters

- Makes roadmap progress measurable and repeatable.

### Next steps

- Add trend tracking across multiple runs/images.

---

## 2.10 Path Atlas + Animation QA Renderer Fidelity

### What was implemented

In `scene_understanding/pathing/path_atlas_canvas.py` and `scene_understanding/visualization/animation_qa_renderer.py`:

- Added lane separation and clearer tapered drawing for overlap-heavy scenes.
- Added directional manifold markers (contact, blob/interior, occlusion pulse, portal, volume, effect).
- Added motion-style overlays (`fluid`, `aerial`, `ground`).
- Renderer motion vocabulary now derived from timeline/candidates/contracts, not fixed keyword lists.
- Manifold-specific visual primitives are rendered during QA video playback.

### Methodology

- Render manifold semantics directly, not just actor center movement.
- Use support classification to choose motion style and micro-kinematics.

### Results and outcome

- Improved visual interpretability of atlas panels and QA video clips.
- Better alignment between contract semantics and what reviewers see.

### Why this matters

- Reduces false visual negatives during QA and better communicates intent fidelity.

### Next steps

- Add richer portal/effect transitions and style-specific occlusion compositing options.

---

## 2.11 Final Consolidated Export (`paths.json`) — This Chat

### What was implemented

- New module: `scene_understanding/pathing/final_paths_export.py`
- New output directory per image:
  - `scene_graph/staged/<stem>_final_paths/`
  - `paths.json`
  - `final_paths_manifest.json`
  - `visuals/` containing copied atlas/trajectory PNG artifacts when present
- Wired from `animation_export.run(...)` via `export_final_paths_bundle(...)`.

### Methodology

- Build consolidated payload from staged artifacts:
  - scene summary
  - object affordances + mask depth profile + object geometry
  - ranked path records + per-path payloads
  - trajectory/action linkage
  - grounding and raster manifests
- Infer requested scene/object fields:
  - `environment`, `has_water`, `people_count`
  - `is_walkable_surface`, `is_climbable_surface`, `is_water`, `is_obstacle`
  - depth stats (`mean`, `median`, `min`, `max`, `p10`, `p90` where available)

### Results and outcome

- Single authoritative enriched bundle for downstream animation/action consumers.
- Keeps full `per_path` detail via `full_per_path_record` while exposing normalized summaries.

### Why this matters

- Solves fragmentation across many artifacts and provides a stable integration contract for later systems.

### Next steps

- Add strict schema validation pass before writing `paths.json`.
- Add configurable output root strategy for cross-run aggregation.

---

## 2.12 Batch PNG Crash Fix — This Chat

### What was implemented

- Fixed `list index out of range` in `path_atlas_canvas.py`:
  - function `_path_midpoint_heading(...)`
  - added explicit 2-point polyline branch.

### Root cause

- For 2-point polylines, midpoint logic used `pts[mid + 1]` where `mid` could be last index.
- Directional marker draw path then crashed in trajectory batch render loop.

### Results and outcome

- Trajectory batch writer no longer fails on 2-point geometry.
- Smoke verification successfully produced batch outputs without index errors.

### Why this matters

- Removes a critical runtime failure that blocked path trajectory batch image generation.

### Next steps

- Add a small unit test for 0/1/2-point path marker behavior.

---

## 3) Observed Current Outcome

From the staged pipeline logs and current artifact behavior:

- paths generated at scale (example run produced hundreds of hypotheses),
- action and animation artifacts generated with richer contracts,
- final consolidated `paths.json` now emitted in dedicated final directory,
- trajectory batch PNG crash path has been fixed.

Net effect: the system now better reflects the action-manifold architecture in `docs/path_updates.md` while preserving additive compatibility.

---

## 4) Remaining Gaps vs `docs/path_updates.md`

1. Full intent-first arbitrary user action text compilation is still partial.
2. Affordance rasters are integrated, but cost-field fusion can be deepened.
3. Threshold calibration needs systematic tuning across fixture families.
4. Some fixture/contract checks can be extended for stronger regression guarantees.
5. Final `paths.json` schema enforcement should be formalized in-code.

---

## 5) Recommended Next Steps (Priority Order)

1. Add `final_paths` schema validator + failure report.
2. Add tests for marker safety and trajectory batch render edge cases.
3. Expand manifold candidate generation for interior/portal/effect with richer geometry cues.
4. Improve local grounding confidence calibration and adaptive thresholds.
5. Complete direct action-text intent compiler path and wire into staged CLI inputs.
6. Add run-to-run QA diff tooling (`path_visual_qa` and final bundle deltas).

---

## 6) Summary

The codebase has moved materially toward the roadmap target:

- grounded, open-vocabulary evidence ingestion,
- manifold-aware candidate generation and scoring,
- richer action/trajectory/render contracts,
- stronger QA and compliance visibility,
- and now a consolidated final `paths.json` export for downstream use.

This is a substantial architecture-level step from “line path planning” toward “scene-grounded action manifold planning.”

---

## 7) Detailed Roadmap Traceability (`docs/path_updates.md`)

This section maps roadmap intent to concrete implementation artifacts and practical effect.

## 7.1 Phase 1 — Contracts and Documentation

### Roadmap intent

Define additive contracts first, then migrate implementation against those contracts.

### Implemented

- Contracted payloads are now concretely produced and linked in staged outputs:
  - `scene_graph/staged/<stem>_paths/path_hypotheses.json`
  - `scene_graph/staged/<stem>_paths/action_hypotheses.json`
  - `scene_graph/staged/<stem>_paths/trajectory_hypotheses.json`
  - `scene_graph/staged/<stem>_paths/animation_components.json`
  - `scene_graph/staged/<stem>_paths/path_visual_qa.json`
  - `scene_graph/staged/<stem>_paths/per_path/*.json`
- New consolidated contract:
  - `scene_graph/staged/<stem>_final_paths/paths.json`

### Why this is meaningful

Consumers can now read a stable layered contract:

1. generation candidates and traces,
2. action hypotheses and status semantics,
3. trajectory and render contract,
4. final consolidated downstream bundle.

The migration is additive: legacy consumers can ignore new keys.

## 7.2 Phase 2 — Staged Path Parity

### Roadmap intent

Bring staged path export closer to monolith behavior while preserving maintainability.

### Implemented in practice

- `paths_export` now performs pre-path grounding artifact generation.
- Path records include richer geometry and semantics:
  - `polyline_3d`, `depth_trace_m`, `support_trace`, `semantic_trace`, `caption_trace`,
  - `visibility_profile`, `width_profile_px`, `path_shape_contract`, `trajectory_contract`,
  - `grounding_evidence`, `ground_object_classification`, uncertainty/contradiction reasons.

### Why this is meaningful

Path hypotheses are no longer “just lines + confidence”; they are animation-ready evidence contracts.

## 7.3 Phase 3 — Affordance Layer and Open Vocabulary

### Roadmap intent

Use scene/object/mask affordance understanding and open vocabulary evidence as first-class generation inputs.

### Implemented in practice

- New grounding modules:
  - `scene_understanding/pathing/scene_grounding_index.py`
  - `scene_understanding/pathing/open_vocab_grounding.py`
  - `scene_understanding/pathing/affordance_rasters.py`
- New candidate module:
  - `scene_understanding/pathing/manifold_candidate_generation.py`
- Integrated into staged path generation before final ranking/gating.

### Why this is meaningful

This changes the architecture from:

- route-first, evidence-second

to:

- evidence-first candidate generation and manifold selection.

## 7.4 Phase 4 — Action Hypotheses

### Roadmap intent

Represent path-level and non-line actions with manifold-aware status and evidence.

### Implemented in practice

- Action statuses now align with path statuses:
  - `accepted`, `plausible_uncertain`, `low_confidence`, `rejected`.
- Action records include propagated evidence and reason fields.

### Why this is meaningful

Action output now preserves uncertainty and contradiction rather than flattening everything to accepted/rejected.

## 7.5 Phase 5 — Occlusion-Aware Trajectories

### Roadmap intent

Depth-aware scale/width/alpha/visibility and render-layer aware trajectory behavior.

### Implemented in practice

- Trajectory and animation contracts now include:
  - render layer semantics,
  - motion mode candidates,
  - contract propagation from path/action context.
- Renderer uses manifold-aware primitives and motion-style overlays.

### Why this is meaningful

Video QA reflects semantic manifold intent, not only geometric interpolation.

## 7.6 Phase 6 — QA and Tests

### Roadmap intent

Make generated artifacts auditable through overlays, videos, and compliance checks.

### Implemented in practice

- `path_visual_qa` exports with expanded summaries and per-path payloads.
- `path_updates_compliance` expanded to include plausible statuses and grounding/raster availability.
- Visual atlas and trajectory batches are now robust against previously observed 2-point marker crashes.

### Why this is meaningful

Regression detection moved from manual visual guesswork toward contract-aware diagnostics.

---

## 8) End-to-End Dataflow: What Happens Now

The practical flow for one image in staged mode:

1. segmentation + depth + labeling + relations + captions
2. affordances stage emits:
   - scene/object/mask affordance JSON artifacts
3. paths stage:
   - builds grounding index + open-vocab summary
   - rasterizes affordance channels
   - builds manifold candidates
   - generates and enriches path hypotheses
   - computes manifold-fit and contract scores
   - assigns acceptance status
4. action stage:
   - emits action hypotheses linked to paths and propagated evidence
5. animation stage:
   - builds trajectory hypotheses and animation components
   - writes path trajectory overview and batch PNGs
   - writes path visual QA JSON/MD and per-path QA JSON/MD
   - writes atlas and animation QA videos
   - writes compliance reports
   - writes final consolidated bundle (`<stem>_final_paths/paths.json`)

This sequence is now materially aligned with the conceptual model in `docs/path_updates.md`.

---

## 9) File-Level Change Log With Method and Effect

## 9.1 `scene_understanding/stages/paths_export.py`

### Method

- Added a pre-routing grounding artifact pipeline.
- Introduced multi-channel support trace sampling and manifold-fit scoring integration.
- Added manifold-specific acceptance thresholds and status model.

### Effect

- More nuanced path viability decisions.
- Better compatibility with manifold-specific motion semantics.
- Richer contracts for downstream stages.

## 9.2 `scene_understanding/stages/action_export.py`

### Method

- Aligned status semantics with path contracts.
- Propagated grounding and uncertainty diagnostics.

### Effect

- Action outputs now explain why actions are uncertain or contradicted.

## 9.3 `scene_understanding/stages/animation_export.py`

### Method

- Made motion mode derivation ontology/evidence-driven.
- Strengthened trajectory context propagation.
- Added final bundle export wiring.

### Effect

- Animation planning scales better with new motion vocabulary.
- Added direct output for downstream consumers via final `paths.json`.

## 9.4 `scene_understanding/pathing/path_atlas_canvas.py`

### Method

- Added visual lane separation and directional manifold markers.
- Added explicit 2-point midpoint/heading handling.

### Effect

- Better readability in dense path atlases.
- Fixed runtime crash in trajectory batch rendering.

## 9.5 `scene_understanding/visualization/animation_qa_renderer.py`

### Method

- Dynamic motion vocabulary from timeline/candidates/contracts.
- Manifold-specific primitive rendering and motion-style overlays.

### Effect

- QA videos now visually express manifold class and movement style intent.

## 9.6 New modules introduced

- `scene_understanding/pathing/scene_grounding_index.py`
- `scene_understanding/pathing/open_vocab_grounding.py`
- `scene_understanding/pathing/affordance_rasters.py`
- `scene_understanding/pathing/manifold_candidate_generation.py`
- `scene_understanding/pathing/manifold_fit_scoring.py`
- `scene_understanding/pathing/final_paths_export.py`

These modules represent the core architectural shift toward evidence-first manifold generation and export.

---

## 10) Scoring and Gating: Practical Behavior

The new path confidence and status behavior now depends on:

- local grounding evidence (`local_grounding_score`),
- manifold support fit (`manifold_fit_score`),
- geometry contract quality (`geometry_contract_score`),
- renderability (`renderability_score`),
- contradiction score,
- uncertainty score,
- support channel means.

### Why this matters in real scenes

- A route can now be retained as `plausible_uncertain` instead of being prematurely dropped.
- Contradictions (hard blockers, severe geometry mismatch) are separated from uncertainty (weak local evidence), enabling better downstream triage.

---

## 11) Geometry Quality and Visual Defect Mitigation

Observed defect classes from prior discussions:

- zigzags,
- sharp turns,
- depth jumps,
- support snap displacement.

Implemented contract fields expose these directly:

- `path_geometry_quality.zigzag_score`
- `path_geometry_quality.turn_angle_p95`
- `path_geometry_quality.depth_jump_count`
- `path_geometry_quality.support_snap_displacement_px`

### Outcome

These are now auditable in:

- `path_hypotheses.json`
- `path_visual_qa.json`
- `per_path/*.json`

and influence acceptance/status decisions instead of being post-hoc visual comments.

---

## 12) Final `paths.json` Consolidation Design: Deep Notes

The final bundle in `scene_graph/staged/<stem>_final_paths/paths.json` is designed for:

- animation engines,
- action planners,
- external evaluators and analytics.

### Key design choices

1. Keep normalized summaries for fast consumption.
2. Preserve full path detail via `full_per_path_record`.
3. Include scene context fields requested in this project:
   - `environment`, `has_water`, `people_count`.
4. Include object affordance flags and depth summaries:
   - `is_walkable_surface`, `is_climbable_surface`, `is_water`, `is_obstacle`,
   - `mean_depth`, `median_depth`, `min_depth`, `max_depth`.
5. Keep links to canonical staged artifacts for traceability.

### Why this matters

Downstream consumers no longer need to stitch 10+ artifacts manually to reconstruct intent and contracts.

---

## 13) Error Analysis: `path_trajectories batch PNGs failed`

### Symptom

`[AnimationExport] path_trajectories batch PNGs failed: list index out of range`

### Root cause

`_path_midpoint_heading` in `path_atlas_canvas` assumed access to `pts[mid + 1]` after midpoint clamping logic that is unsafe for `len(pts) == 2`.

### Fix

Added explicit 2-point branch:

- midpoint computed as average of endpoints,
- heading computed from endpoint delta,
- avoids out-of-range access.

### Verified outcome

Trajectory batch writer smoke test succeeds after patch.

---

## 14) Practical Results So Far

At current state, the pipeline now delivers:

- grounded path generation artifacts,
- manifold-specific scoring and statuses,
- expanded action and animation contracts,
- richer visual QA and compliance reporting,
- final consolidated `paths.json` in dedicated final directory,
- resolved crash on path trajectory batch rendering.

This is a substantive milestone toward the architecture described in `docs/path_updates.md`.

---

## 15) Gaps and Risk Register

## 15.1 Intent-first action text (remaining)

- Risk: still partial for arbitrary action prompts directly from user text.
- Impact: open-vocab generation remains mostly evidence-led from scene artifacts.

## 15.2 Threshold calibration and drift

- Risk: manifold thresholds may under/over-accept in outlier scenes.
- Impact: unstable acceptance distribution across datasets.

## 15.3 Cost-field fusion depth

- Risk: soft raster channels not yet fully fused into all routing strategies.
- Impact: some candidate diversity not realized under difficult geometry.

## 15.4 Validation breadth

- Risk: not all edge classes have dedicated unit/fixture coverage yet.
- Impact: regressions can reappear in under-tested manifold/geometry corner cases.

---

## 16) Recommended Execution Plan (Detailed)

## 16.1 Immediate hardening (short horizon)

1. Add unit tests for `_path_midpoint_heading` for `len=0/1/2/N`.
2. Add tests for trajectory batch rendering with minimal 2-point and degenerate paths.
3. Add final bundle schema validation (`citv_final_paths_bundle_v1`) before write.
4. Emit explicit warning if no visuals copied into final `visuals/` directory.

## 16.2 Quality calibration (medium horizon)

1. Build calibration script over fixtures:
   - acceptance status distributions by manifold.
2. Tune `_manifold_acceptance_thresholds` using fixture-driven metrics.
3. Add confidence reliability curves for `accepted` vs `plausible_uncertain`.

## 16.3 Capability completion (medium/long horizon)

1. Fully wire direct action-text intent compilation path.
2. Deepen raster-to-cost fusion in routing.
3. Expand manifold generators for interior/portal/effect edge cases.

---

## 17) Suggested Reviewer Checklist for Next Runs

For each new image run:

1. Confirm `scene_grounding_index.json` and `open_vocab_grounding.json` exist.
2. Confirm `affordance_rasters.npz` + manifest exists and non-empty channels.
3. Confirm path statuses include plausible/low/rejected buckets (not all collapsed).
4. Inspect `path_visual_qa.json` for geometry quality regressions.
5. Verify trajectory batch PNG generation does not fail.
6. Verify final bundle exists:
   - `scene_graph/staged/<stem>_final_paths/paths.json`
   - `scene_graph/staged/<stem>_final_paths/final_paths_manifest.json`
7. Verify final bundle contains:
   - scene context fields,
   - object affordance flags and depth stats,
   - full per-path contract references.

---

## 18) Pair Programming Focus Areas

If we continue collaboratively, highest leverage pair sessions are:

1. Threshold calibration workshop with fixture metrics.
2. Action-intent compiler integration session.
3. Final schema validator and compatibility policy definition.
4. Visual QA regression suite setup.

---

## 19) Closing Statement

The project is no longer in a “path as line only” state. It now has the foundation of a grounded action-manifold system:

- fused evidence,
- manifold-aware candidate generation,
- status and confidence semantics that distinguish contradiction vs uncertainty,
- richer animation and QA contracts,
- and a final consolidated `paths.json` export that is directly useful for downstream action and animation systems.

The next critical push is not architecture invention, but hardening and calibration of the architecture now in place.

