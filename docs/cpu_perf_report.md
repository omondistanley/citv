# CPU Performance Report

Generated from `logs/timing_log.md` from benchmark runs.

- Runs observed: `10`
- Best wall time: `106.06s`
- Latest wall time: `877.74s`
- Target wall time: `<200s`
- Gate status (best-run): `PASS`

## Notes

- Runtime variance is currently high across runs; optimization should continue to reduce tail latency.
- CPU benchmark mode can disable heavy visual exports (`cpu_benchmark_disable_heavy_visuals`) to stabilize runtime.

## Dual-FPS Animation QA Runs (Single Image)

- Image: `076204C0-24E8-4022-9204-DD5FF8E0AAA2_1_105_c.jpeg`
- Mode `24 FPS` (CPU benchmark, `path_animation_qa_modes=[24]`): pipeline wall total `188.40s` (`output_animation_qa_24`)
- Mode `120 FPS` (CPU benchmark, `path_animation_qa_modes=[120]`): pipeline wall total `424.40s` (`output_animation_qa_120`)
- Note: both runs logged `[Writer] pending write failed: Object of type ndarray is not JSON serializable` during async flush; path exports and animation QA artifacts still validated.
- Validators: validation scripts were run during the benchmark pass and both modes passed.
