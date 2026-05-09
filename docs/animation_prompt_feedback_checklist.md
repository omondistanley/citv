# Animation Prompt Feedback Checklist

This checklist maps animation-team requirements to exported path/trajectory components.

## Required Motion Types

- [ ] idle
- [ ] walk
- [ ] run
- [ ] jump
- [ ] crawl (optional)

## Non-Negotiable Prompt/Component Fields

- [ ] `motion_mode_candidates` with confidence values
- [ ] `timeline_records` with `t0_s` and `t1_s`
- [ ] `trajectory_points` / `states_t` for spatial travel
- [ ] `path_id` <-> `trajectory_id` linkage
- [ ] relation guardrails (`relation_ids`, relation support count)
- [ ] confidence rationale (semantic, geometric, relation, depth)

## Timeline + Trajectory Coupling

- [ ] Every timeline segment references a valid trajectory interval.
- [ ] Motion segment transitions align with path geometry progression.
- [ ] Path/trajectory ranking carries through to animation selection order.

## Rejection Conditions

- [ ] Reject candidates with hard relation violations.
- [ ] Reject candidates with severe semantic non-traversability.
- [ ] Reject candidates with geometry invalidity beyond threshold.

## Validation Hooks

- [ ] Validation pass reports contract/path rollout checks as passing.
- [ ] Exports include ranked panel trio:
  - line-only
  - context
  - paths+trajectories overlay
