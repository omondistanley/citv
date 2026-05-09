# CPU Benchmark Baseline (No GPU)

This document captures the machine baseline and runtime budget policy for the CPU-first pathing/trajectory upgrade.

## Machine Specs

- Host OS: `Darwin 25.4.0` (`arm64`)
- CPU: `Apple M2`
- Cores: `8 physical`, `8 logical`
- RAM: `8 GiB`
- Python: `3.14.3`
- Workspace filesystem free space: `~10 GiB` available on `/System/Volumes/Data`

## Runtime Goal

- Target end-to-end wall time per image: `< 200s` on CPU-only mode.
- CPU-only means:
  - no CUDA execution path,
  - no MPS acceleration path,
  - deterministic worker and export behavior tuned for this host.

## Stage Budget Checkpoints (Per Image)

These checkpoints are used to track budget drift and enforce optimization priorities:

- `pre_depth`: <= 5s
- `depth`: <= 20s
- `segmentation`: <= 35s
- `regions`: <= 8s
- `enrich_regions`: <= 20s
- `path_hypotheses_export`: <= 25s
- `relations_layers_maps`: <= 8s
- `track_geometry_exports`: <= 60s
- `caption_exports`: <= 15s
- `release_heavy_models`: <= 4s
- `wall_total`: <= 200s

## Current Observations

Historical timing logs in `logs/timing_log.md` include runs that are both below and above target.
The optimization loop must treat `<200s` as a hard acceptance gate for CPU benchmark mode.

## Verification Policy

- Every benchmark run records:
  - machine spec snapshot,
  - stage times,
  - wall time pass/fail status.
- A run is accepted only when `wall_total_ms <= 200000`.
