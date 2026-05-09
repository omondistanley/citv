# Trajectory atlas (trajs-upt): ranked line-only QA and roadmap

This document describes **Phase 1** exports (`path_top30_atlas.md`, `path_atlas_manifest.json`, `path_atlas_ranked_panel_*.png`, optional `traj_atlas_line_only.png`) and records a **roadmap** for richer trajectory handling discussed alongside this feature.

---

## Phase 1 — What ships today

### Purpose

- **Visual QA** without drawing the photograph, region contours, or object boxes: only **tapered polylines** on a flat background at **full image resolution** so pixel coordinates stay aligned with `path_hypotheses.json`.
- **Top N paths** (default **30**) taken from a configurable rank pool, split into **panels** of **10** paths each (default **three** PNGs when enough paths exist).
- **Stable colors** per `path_id` (SHA-256–based hue) so CI and cross-machine runs match; optional **per-panel hue separation** to reduce collisions.
- A single **Markdown atlas** (`path_top30_atlas.md`) with color tables, links to other JSON/PNG artifacts, duplicate **route signatures**, and per-path detail.

### Enabling

In [`config.py`](../config.py) (`SceneUnderstandingConfig`), set:

- `path_atlas_enabled: bool = True`
- Optional tuning: `path_atlas_total`, `path_atlas_panel_size`, `path_atlas_rank_source` (`"paths_sorted"` | `"paths_recommended"`), `path_atlas_background_bgr`, `path_atlas_prefer_geodesic`, `path_atlas_min_hue_separation`, `path_atlas_endpoint_dots`, `path_atlas_export_trajectory_line_only`, trajectory stroke widths.

Defaults keep the atlas **off** so existing pipelines are unchanged. Mirror literals in [`scene_understanding/core/reviewer_config_defaults.py`](../scene_understanding/core/reviewer_config_defaults.py) when cited in reviewer docs.

### Output files (under `{stem}_paths/`)

| File | Role |
|------|------|
| `path_atlas_ranked_panel_01.png` … | Line-only BGR panels; panel *i* draws up to 10 paths. |
| `path_atlas_manifest.json` | Schema `citv_path_atlas_manifest_v1`: ranking policy, panel list, per-path `path_num`, `path_id`, `color_bgr`, `color_hex`, `suppressed`, levels, `route_signature`. |
| `path_top30_atlas.md` | Human-readable atlas with tables and per-path sections. |
| `traj_atlas_line_only.png` | Optional: short `trajectory_hypotheses` segments with a **distinct** palette (`traj:` salt in SHA input). |

### Ranking policy

- **`paths_sorted`**: same ordering used elsewhere after hybrid scoring—**recommended** paths sorted by `overall_confidence`, or **all valid** paths if the recommended list is empty.
- **`paths_recommended`**: only non-suppressed paths, sorted by confidence.
- **`path_num`**: global rank by confidence across **all** paths (assigned before atlas); atlas rows reference this stable id.

### Code map

| Area | Location |
|------|-----------|
| Stable colors | [`scene_understanding/pathing/path_colors.py`](../scene_understanding/pathing/path_colors.py) |
| Line-only drawing | [`scene_understanding/pathing/path_atlas_canvas.py`](../scene_understanding/pathing/path_atlas_canvas.py) |
| Manifest + MD | [`scene_understanding/pathing/path_atlas_export.py`](../scene_understanding/pathing/path_atlas_export.py) |
| Export hook | [`scene_understanding.py`](../scene_understanding.py) `_export_path_hypotheses_for_track` (after `path_visual_index.json`) |
| Package re-exports | [`scene_understanding/pathing/__init__.py`](../scene_understanding/pathing/__init__.py) |

### Relation to `path_context_top5.png`

- [`path_context_top5.png`](path_context_top5_reviewer.md) overlays top‑K paths on the **real image** plus region edges and boxes (see [path_context_top5_reviewer.md](path_context_top5_reviewer.md)).
- The **atlas panels** intentionally omit that context for **geometry-only** review; use both together in review.

`path_reasoning.md` includes a one-line pointer to `path_top30_atlas.md` when atlas export is enabled.

---

## TDD / regression tests

- [`test_path_atlas_export.py`](../test_path_atlas_export.py): synthetic scene with `path_atlas_enabled=True`; asserts manifest schema, first panel PNG dimensions, `traj_atlas_line_only.png` when trajectory export flag is on, and hex/`color_bgr` consistency.
- [`test_path_per_image_export.py`](../test_path_per_image_export.py): unchanged defaults (atlas off) keep the full export contract stable.

Future hardening: optional bounds on non-background ink fraction per panel; golden PNG hashes only if OpenCV/AA versions are pinned.

---

## Roadmap (later phases — not all implemented in code yet)

These items capture design from broader trajectory discussions; track implementation separately.

1. **Structured motion / timeline components** — JSON motion segments (type, `t0_s`, `t1_s`) from prompts or classifiers, merged with `polyline_2d` via arc-length resampling; extend `animation_plan.json` beyond confidence-heuristic walk/run.
2. **`motion_support` / path modes** — Explicit taxonomy: `freespace_nav`, `on_surface`, `boundary_orbit`, `intra_region`, `intra_layer`, proximity corridors; separate ranking from mask diagnostics.
3. **Intra-entity path families** — Graphs on mask pixels, region interiors, or depth-layer masks; k-shortest routes between multiple anchors.
4. **Region polylines vs obstacles** — Optionally validate region centroid–portal routes against `all_obs` union or soft penalties.
5. **Planner diversity** — Rely more on region-graph and traversability geodesics; optional **`path_use_astar`** bypass when A* is undesirable for a given product mode.
6. **Insertion contracts** — Populate `corridor` and geometry-derived `clearance_m` / `footprint_m` in [`scene_path_motion_contracts.py`](../scene_path_motion_contracts.py).
7. **Prompt ingestion** — Constrain pair proposals or semantic weights from external structured JSON (not free-text full motion generation).

---

## Changelog (Phase 1)

- Introduced trajs-upt atlas exports and this document.
- Cross-link from path reasoning MD and from the path context reviewer guide.
