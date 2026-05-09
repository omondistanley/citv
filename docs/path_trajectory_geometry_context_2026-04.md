# Path geometry, trajectories, and semantic context (Apr 2026)

This document captures the **problems, metrics, pipeline anchors, and planned fixes** from the path/trajectory design discussion (single-image exports, walkable fusion, ranking, and 3D lift).

## Problems observed

- **Hanging paths:** Endpoints anchored in sky/background or off support surfaces; chords through implausible regions.
- **Straight vs curved:** Many hypotheses are short **2-point** `polyline_2d`; validation reports **max turn 0°** when no triple of vertices exists. Curves come from FMM refinement + smoothing + Bézier when those stages run.
- **Visualization parity:** `path_map_all.png` draws **all** paths’ `polyline_2d`; **top-N** and **atlas** use **confidence ranking** and may **prefer `polyline_geodesic_2d`** — different artifacts can look inconsistent.
- **Intra vs inter (product intent):** **Intra** should mean motion **inside** an instance volume; **`mask_contour`** is **boundary-only** diagnostics. **Inter** spans entity pairs (object–object, region–region, cross-family, etc.).
- **Bad pairs:** Object–object proposals from **lexical similarity + relation mention counts** and **static–static fallback** can yield spurious routes.
- **Region graph:** Weak **adjacency** edges (thin borders, large depth jump) yield bad **region sequences**.

## Metrics (path_descriptions / validity)

- **Invalid-pixel crossing ratio (`invalid_ratio`):** Fraction of **sampled points along polyline segments** that fall outside **`feasible`** (e.g. `path_walkable`) or inside **`obstacles`**. Not a smoothness metric.
- **Max turn (`max_turn_deg`):** Maximum turning angle over **consecutive edges**; with **only two vertices**, the triple loop never runs → **0.0** means **“not applicable”**, not “perfectly straight policy.”
- **`energy_total`:** Weighted sum of **penalty** terms \(1 - \text{score}\) (geometry, semantic, relation, motion) plus depth/uncertainty — **lower** is less penalty. It is **not** a “higher is better” confidence.
- **Hybrid ranking:** `path_score_weight_geometric`, `path_semantic`, `path_relation`, `path_action_fit` (normalized in code) drive **`hybrid_overall`** and **`overall_confidence`**; increase **geometric** weight to favor traversability evidence.

## Pipeline anchors (code)

| Concern | Location |
|--------|----------|
| Pair proposals | `scene_understanding.py` (`pair_proposals`, relation mentions, lexical fallback) |
| Region adjacency | `_build_region_adjacency_graph` (`scene_understanding.py`); edges: `shared_border_px`, `depth_delta_m` |
| Cost field | `scene_understanding/pathing/cost_map.py` |
| FMM / geodesic | Object-pair block, `_fmm_backtrace_from_T`, `_fmm_k_diverse_from_T` |
| Walkable + semantic fusion | `scene_understanding/pathing/walkable_mask.py`; early `semantic_layer` in path export |
| Mask paths | `mask_contour`, `mask_axis`; **interior geodesic** (`mask_interior_geodesic`) |
| Hybrid scores | `scene_understanding/pathing/hybrid_scores.py` |
| Trajectories | `scene_path_motion_contracts.py` (`z_m`, motion modes, timelines) |

## Implemented checklist (this rollout)

- [x] **Planning tab** in `thoughts.md` → this file.
- [x] **Phase A:** Pair gates (distance, optional relation-required), lexical fallback flag.
- [x] **Phase B:** Adjacency edge filter by `depth_delta_m`; optional corridor quality rejection (mean cost / walkable off-ratio).
- [x] **Phase C:** `mask_interior_geodesic` + per-export toggles for contour/axis/interior.
- [x] **Phase D:** Optional endpoint walkable snap/gate; description text for short polylines; optional densified `grounded_inputs`.
- [x] **Phase E:** `path_map_all_draw_geodesic`, `path_map_all_use_recommended_only`.
- [x] **Phase F:** Optional `polyline_3d` from `metric_depth_m` + trajectory use when linking paths.
- [x] **Phase G:** Tests + `verify_path_rollout` notes/flags as needed.

## 3D direction (research themes)

- **Lift 2D→3D:** Sample metric depth at polyline vertices; optional local ground clamp — does not replace a full volumetric planner.
- **Richer scene contact:** Literature on metric human–scene interaction from single images (e.g. joint depth + contact optimization).
- **Egocentric / terrain:** Depth-driven traversability and obstacle height for plausible locomotion.

## Config quick reference

See `config.py` for: `path_pair_*`, `region_adjacency_max_depth_delta_m`, `path_sequence_*`, `path_mask_export_*`, `path_endpoint_*`, `path_map_all_*`, `path_export_polyline_3d`, `path_score_weight_*`.
