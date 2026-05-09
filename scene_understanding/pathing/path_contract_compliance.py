"""Artifact-level compliance checks for path/trajectory contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def evaluate_paths_root(paths_root: Path) -> Dict[str, Any]:
    path_payload = _load_json(paths_root / "path_hypotheses.json")
    traj_payload = _load_json(paths_root / "trajectory_hypotheses.json")
    atlas_payload = _load_json(paths_root / "path_atlas_manifest.json")
    batch_payload = _load_json(paths_root / "path_trajectories_batches.json")
    qa_payload = _load_json(paths_root / "path_visual_qa.json")
    action_payload = _load_json(paths_root / "action_hypotheses.json")
    anim_manifest = _load_json(paths_root / "animation_qa_24" / "animation_qa_manifest.json")
    anim_scores = _load_json(paths_root / "animation_qa_24" / "animation_qa_scores.json")
    grounding_payload = _load_json(paths_root / "scene_grounding_index.json")
    open_vocab_payload = _load_json(paths_root / "open_vocab_grounding.json")
    rasters_manifest = _load_json(paths_root / "affordance_rasters_manifest.json")
    staged_dir = paths_root.parent
    stem = paths_root.name[:-len("_paths")] if paths_root.name.endswith("_paths") else paths_root.name
    caption_payload = _load_json(staged_dir / f"{stem}_caption_evidence.json")
    object_aff_payload = _load_json(staged_dir / f"{stem}_object_affordances.json")
    mask_aff_payload = _load_json(staged_dir / f"{stem}_mask_affordances.json")
    scene_aff_payload = _load_json(staged_dir / f"{stem}_scene_affordances.json")

    paths = list(path_payload.get("hypotheses") or path_payload.get("paths") or [])
    traj = list(traj_payload.get("hypotheses") or traj_payload.get("trajectories") or [])
    atlas_paths = list(atlas_payload.get("paths") or [])
    batches = list(batch_payload.get("batches") or [])
    qa_paths = list(qa_payload.get("paths") or [])
    actions = list(action_payload.get("hypotheses") or action_payload.get("actions") or [])
    segments = sum(len((p.get("segments") or [])) for p in list(anim_manifest.get("panel_videos") or []))
    metrics = sum(len((p.get("metrics") or [])) for p in list(anim_scores.get("panels") or []))

    batch_sizes = [len(b.get("path_ids") or []) for b in batches if isinstance(b, dict)]
    linked_path_ids = set()
    for h in traj:
        if not isinstance(h, dict):
            continue
        pid = str((h.get("action_context") or {}).get("path_id") or h.get("continues_from_path_id") or "")
        if pid:
            linked_path_ids.add(pid)

    all_path_ids = {str(p.get("path_id", "")) for p in paths if isinstance(p, dict) and str(p.get("path_id", "")).strip()}
    per_path_dir = paths_root / "per_path"
    per_path_json = len(list(per_path_dir.glob("*.json"))) if per_path_dir.exists() else 0
    per_path_md = len(list(per_path_dir.glob("*.md"))) if per_path_dir.exists() else 0
    shape_contracts = sum(1 for p in paths if isinstance(p, dict) and isinstance(p.get("path_shape_contract"), dict) and p.get("path_shape_contract"))
    render_contracts = sum(1 for p in paths if isinstance(p, dict) and isinstance(p.get("animation_render_contract"), dict) and p.get("animation_render_contract"))
    geometry_contracts = sum(1 for p in paths if isinstance(p, dict) and isinstance(p.get("path_geometry_quality"), dict) and p.get("path_geometry_quality"))
    display_geometry = sum(1 for p in paths if isinstance(p, dict) and isinstance(p.get("display_polyline_2d"), list) and len(p.get("display_polyline_2d") or []) >= 2)
    accepted_paths = [p for p in paths if isinstance(p, dict) and str(p.get("acceptance_status", "")) == "accepted"]
    plausible_paths = [p for p in paths if isinstance(p, dict) and str(p.get("acceptance_status", "")) == "plausible_uncertain"]
    low_conf_paths = [p for p in paths if isinstance(p, dict) and str(p.get("acceptance_status", "")) == "low_confidence"]
    rejected_paths = [p for p in paths if isinstance(p, dict) and str(p.get("acceptance_status", "")) == "rejected"]
    accepted_actions = [a for a in actions if isinstance(a, dict) and str(a.get("action_status", "")) == "accepted"]
    plausible_actions = [a for a in actions if isinstance(a, dict) and str(a.get("action_status", "")) == "plausible_uncertain"]
    low_conf_actions = [a for a in actions if isinstance(a, dict) and str(a.get("action_status", "")) == "low_confidence"]
    rejected_actions = [a for a in actions if isinstance(a, dict) and str(a.get("action_status", "")) == "rejected"]
    high_conf_unknown_only = []
    high_conf_blocking = []
    high_conf_straight_unjustified = []
    high_conf_bad_geometry = []
    missing_display_geometry = []
    missing_local_evidence = []
    for p in paths:
        if not isinstance(p, dict):
            continue
        conf = _float((p.get("scores") or {}).get("overall_confidence"), 0.0)
        sem = p.get("semantic_trace") if isinstance(p.get("semantic_trace"), dict) else {}
        support = dict(sem.get("support_kind_counts") or {})
        total = max(1.0, sum(_float(v) for v in support.values()))
        unknown = _float(support.get("unknown"), 0.0) / total
        blocking = _float(support.get("blocking"), 0.0) / total
        shape = p.get("path_shape_contract") if isinstance(p.get("path_shape_contract"), dict) else {}
        quality = p.get("path_geometry_quality") if isinstance(p.get("path_geometry_quality"), dict) else {}
        reasons = set(str(r) for r in list(p.get("rejection_reasons") or []) + list(shape.get("rejection_reasons") or []))
        geometry_reasons = [str(r) for r in list(quality.get("geometry_rejection_reasons") or []) if str(r)]
        accepted_or_plausible = str(p.get("acceptance_status", "")) in {"accepted", "plausible_uncertain"}
        accepted_only = str(p.get("acceptance_status", "")) == "accepted"
        if accepted_only and conf >= 0.48 and unknown >= 0.80:
            high_conf_unknown_only.append(str(p.get("path_id", "")))
        if accepted_or_plausible and conf >= 0.48 and blocking >= 0.50:
            high_conf_blocking.append(str(p.get("path_id", "")))
        if accepted_or_plausible and conf >= 0.48 and bool(shape.get("straight_line_like")) and reasons.intersection({"straight_line_shape_unjustified", "two_point_route_shape_underexplained", "straight_line_crosses_region_transition_without_shape_context"}):
            high_conf_straight_unjustified.append(str(p.get("path_id", "")))
        if accepted_only and conf >= 0.48 and geometry_reasons:
            high_conf_bad_geometry.append(str(p.get("path_id", "")))
        if not (isinstance(p.get("display_polyline_2d"), list) and len(p.get("display_polyline_2d") or []) >= 2):
            missing_display_geometry.append(str(p.get("path_id", "")))
        grounding = p.get("grounding_evidence") if isinstance(p.get("grounding_evidence"), dict) else {}
        local_conf = max(
            _float((p.get("scores") or {}).get("local_action_evidence_confidence"), 0.0),
            _float(grounding.get("local_evidence_confidence"), 0.0),
        )
        if accepted_or_plausible and conf >= 0.48 and local_conf <= 0.01:
            missing_local_evidence.append(str(p.get("path_id", "")))
    grounding_available = bool(grounding_payload.get("entities"))
    raster_artifacts_available = bool(rasters_manifest.get("rasters")) and (paths_root / "affordance_rasters.npz").exists()
    open_vocab_available = bool(open_vocab_payload.get("dynamic_concepts")) or bool(caption_payload.get("open_vocab_concepts")) or bool((scene_aff_payload.get("open_vocab") or {}).get("dynamic_concepts"))
    scene_evidence_available = bool(scene_aff_payload.get("summary"))
    object_evidence_available = bool(object_aff_payload.get("objects"))
    mask_evidence_available = bool(mask_aff_payload.get("masks"))
    actions_have_intent = bool(actions) and all(isinstance(a, dict) and isinstance(a.get("intent_compiler"), dict) for a in actions[: min(20, len(actions))])
    render_contract_traj_links = sum(
        1
        for h in traj
        if isinstance(h, dict)
        and isinstance((h.get("action_context") or {}).get("animation_render_contract"), dict)
        and (h.get("action_context") or {}).get("animation_render_contract")
    )
    scene_context_loaded = bool((anim_manifest.get("scene_context") or {}).get("loaded"))

    failures: List[str] = []
    if len(paths) == 0:
        failures.append("no_paths_emitted")
    if len(traj) == 0:
        failures.append("no_trajectories_emitted")
    if len(atlas_paths) != len(paths):
        failures.append("atlas_path_count_mismatch")
    if len(linked_path_ids) != len(all_path_ids):
        failures.append("missing_trajectory_links")
    if batch_sizes and any((n > 10) for n in batch_sizes):
        failures.append("batch_size_exceeds_10")
    if qa_paths and len(qa_paths) != len(paths):
        failures.append("path_visual_qa_count_mismatch")
    if all_path_ids and (per_path_json < len(all_path_ids) or per_path_md < len(all_path_ids)):
        failures.append("per_path_json_md_incomplete")
    if anim_manifest and segments == 0:
        failures.append("animation_segments_empty")
    if anim_scores and metrics == 0:
        failures.append("animation_metrics_empty")
    strict_contract_schema = str(path_payload.get("schema", "")).startswith("citv_path_hypotheses_v3")
    if paths and strict_contract_schema and shape_contracts < len(paths):
        failures.append("path_shape_contracts_incomplete")
    if paths and strict_contract_schema and render_contracts < len(paths):
        failures.append("animation_render_contracts_incomplete")
    if paths and strict_contract_schema and geometry_contracts < len(paths):
        failures.append("path_geometry_quality_incomplete")
    if paths and strict_contract_schema and display_geometry < len(paths):
        failures.append("display_geometry_incomplete")
    if high_conf_unknown_only:
        failures.append("high_confidence_unknown_only_support")
    if high_conf_blocking:
        failures.append("high_confidence_blocking_support")
    if high_conf_straight_unjustified:
        failures.append("high_confidence_unjustified_straight_paths")
    if high_conf_bad_geometry:
        failures.append("high_confidence_bad_path_geometry")
    if missing_local_evidence:
        failures.append("high_confidence_missing_local_action_evidence")

    phase_compliance = _phase_compliance(
        paths=len(paths),
        trajectories=len(traj),
        actions=len(actions),
        shape_contracts=shape_contracts,
        render_contracts=render_contracts,
        geometry_contracts=geometry_contracts,
        display_geometry=display_geometry,
        per_path_json=per_path_json,
        per_path_md=per_path_md,
        open_vocab_available=open_vocab_available,
        scene_evidence_available=scene_evidence_available,
        object_evidence_available=object_evidence_available,
        mask_evidence_available=mask_evidence_available,
        grounding_available=grounding_available,
        raster_artifacts_available=raster_artifacts_available,
        high_conf_unknown_only=len(high_conf_unknown_only),
        high_conf_blocking=len(high_conf_blocking),
        high_conf_straight_unjustified=len(high_conf_straight_unjustified),
        high_conf_bad_geometry=len(high_conf_bad_geometry),
        missing_local_evidence=len(missing_local_evidence),
        actions_have_intent=actions_have_intent,
        accepted_actions=len(accepted_actions),
        plausible_actions=len(plausible_actions),
        render_contract_traj_links=render_contract_traj_links,
        batch_sizes=batch_sizes,
        segments=segments,
        metrics=metrics,
        scene_context_loaded=scene_context_loaded,
    )

    return {
        "paths": len(paths),
        "trajectories": len(traj),
        "actions": len(actions),
        "atlas_paths": len(atlas_paths),
        "batch_sizes": batch_sizes,
        "path_visual_qa_paths": len(qa_paths),
        "linked_paths": len(linked_path_ids),
        "per_path_json": per_path_json,
        "per_path_md": per_path_md,
        "path_shape_contracts": shape_contracts,
        "animation_render_contracts": render_contracts,
        "path_geometry_quality_contracts": geometry_contracts,
        "display_geometry_paths": display_geometry,
        "accepted_paths": len(accepted_paths),
        "plausible_uncertain_paths": len(plausible_paths),
        "low_confidence_paths": len(low_conf_paths),
        "rejected_paths": len(rejected_paths),
        "accepted_actions": len(accepted_actions),
        "plausible_uncertain_actions": len(plausible_actions),
        "low_confidence_actions": len(low_conf_actions),
        "rejected_actions": len(rejected_actions),
        "open_vocab_available": open_vocab_available,
        "scene_grounding_index_available": grounding_available,
        "affordance_rasters_available": raster_artifacts_available,
        "scene_evidence_available": scene_evidence_available,
        "object_evidence_available": object_evidence_available,
        "mask_evidence_available": mask_evidence_available,
        "actions_have_intent_compiler": actions_have_intent,
        "trajectory_render_contract_links": render_contract_traj_links,
        "scene_context_loaded_for_animation_qa": scene_context_loaded,
        "quality_regressions": {
            "high_confidence_unknown_only_support": high_conf_unknown_only[:40],
            "high_confidence_blocking_support": high_conf_blocking[:40],
            "high_confidence_unjustified_straight_paths": high_conf_straight_unjustified[:40],
            "high_confidence_bad_path_geometry": high_conf_bad_geometry[:40],
            "missing_display_geometry": missing_display_geometry[:40],
            "high_confidence_missing_local_action_evidence": missing_local_evidence[:40],
        },
        "phase_compliance": phase_compliance,
        "animation_segments": segments,
        "animation_metrics": metrics,
        "failures": failures,
        "ok": not failures,
    }


def _float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _status(ok: bool, partial: bool) -> str:
    if ok:
        return "met"
    if partial:
        return "partial"
    return "missing"


def _phase_compliance(**m: Any) -> List[Dict[str, Any]]:
    phases: List[Dict[str, Any]] = []
    paths = int(m.get("paths", 0))
    actions = int(m.get("actions", 0))
    traj = int(m.get("trajectories", 0))
    shape_ok = paths > 0 and int(m.get("shape_contracts", 0)) >= paths
    render_ok = paths > 0 and int(m.get("render_contracts", 0)) >= paths
    geometry_ok = paths > 0 and int(m.get("geometry_contracts", 0)) >= paths and int(m.get("display_geometry", 0)) >= paths
    per_path_ok = paths > 0 and int(m.get("per_path_json", 0)) >= paths and int(m.get("per_path_md", 0)) >= paths
    phase1_ok = shape_ok and render_ok and geometry_ok and per_path_ok
    phases.append({
        "phase": 1,
        "name": "Contracts And Compliance",
        "status": _status(phase1_ok, shape_ok or render_ok or geometry_ok or per_path_ok),
        "evidence": {
            "path_shape_contracts": int(m.get("shape_contracts", 0)),
            "animation_render_contracts": int(m.get("render_contracts", 0)),
            "path_geometry_quality_contracts": int(m.get("geometry_contracts", 0)),
            "display_geometry_paths": int(m.get("display_geometry", 0)),
            "per_path_json": int(m.get("per_path_json", 0)),
            "per_path_md": int(m.get("per_path_md", 0)),
        },
    })
    bad_support = int(m.get("high_conf_unknown_only", 0)) + int(m.get("high_conf_blocking", 0))
    bad_geometry = int(m.get("high_conf_bad_geometry", 0))
    grounded_ok = bool(m.get("grounding_available")) and bool(m.get("raster_artifacts_available"))
    phase2_ok = paths > 0 and shape_ok and geometry_ok and grounded_ok and bad_support == 0 and bad_geometry == 0 and int(m.get("missing_local_evidence", 0)) == 0
    phases.append({
        "phase": 2,
        "name": "Scene-Grounded Path Generation",
        "status": _status(phase2_ok, paths > 0 and shape_ok),
        "evidence": {
            "paths": paths,
            "scene_evidence_available": bool(m.get("scene_evidence_available")),
            "object_evidence_available": bool(m.get("object_evidence_available")),
            "mask_evidence_available": bool(m.get("mask_evidence_available")),
            "scene_grounding_index_available": bool(m.get("grounding_available")),
            "affordance_rasters_available": bool(m.get("raster_artifacts_available")),
            "high_confidence_unknown_or_blocking": bad_support,
            "high_confidence_bad_path_geometry": bad_geometry,
            "high_confidence_missing_local_action_evidence": int(m.get("missing_local_evidence", 0)),
        },
    })
    phase3_ok = bool(m.get("open_vocab_available")) and bool(m.get("grounding_available")) and bool(m.get("scene_evidence_available")) and bool(m.get("object_evidence_available"))
    phases.append({
        "phase": 3,
        "name": "Open-Vocabulary Affordance Grounding",
        "status": _status(phase3_ok, bool(m.get("open_vocab_available"))),
        "evidence": {
            "open_vocab_available": bool(m.get("open_vocab_available")),
            "scene_grounding_index_available": bool(m.get("grounding_available")),
            "scene_evidence_available": bool(m.get("scene_evidence_available")),
            "object_evidence_available": bool(m.get("object_evidence_available")),
            "mask_evidence_available": bool(m.get("mask_evidence_available")),
        },
    })
    phase4_ok = actions > 0 and bool(m.get("actions_have_intent")) and (int(m.get("accepted_actions", 0)) + int(m.get("plausible_actions", 0))) > 0
    phases.append({
        "phase": 4,
        "name": "Action Hypotheses And Manifold Selection",
        "status": _status(phase4_ok, actions > 0),
        "evidence": {
            "actions": actions,
            "accepted_actions": int(m.get("accepted_actions", 0)),
            "plausible_uncertain_actions": int(m.get("plausible_actions", 0)),
            "actions_have_intent_compiler": bool(m.get("actions_have_intent")),
        },
    })
    phase5_ok = traj > 0 and render_ok and geometry_ok and int(m.get("render_contract_traj_links", 0)) >= max(1, min(traj, paths))
    phases.append({
        "phase": 5,
        "name": "Trajectory Direction And Rendering",
        "status": _status(phase5_ok, traj > 0 and render_ok),
        "evidence": {
            "trajectories": traj,
            "trajectory_render_contract_links": int(m.get("render_contract_traj_links", 0)),
            "high_confidence_unjustified_straight_paths": int(m.get("high_conf_straight_unjustified", 0)),
            "display_geometry_paths": int(m.get("display_geometry", 0)),
            "high_confidence_bad_path_geometry": int(m.get("high_conf_bad_geometry", 0)),
        },
    })
    batch_sizes = list(m.get("batch_sizes") or [])
    phase6_ok = bool(batch_sizes) and all(int(n) <= 10 for n in batch_sizes) and int(m.get("segments", 0)) > 0 and int(m.get("metrics", 0)) > 0 and bool(m.get("scene_context_loaded"))
    phases.append({
        "phase": 6,
        "name": "QA Videos And Reviewer Artifacts",
        "status": _status(phase6_ok, bool(batch_sizes) and all(int(n) <= 10 for n in batch_sizes)),
        "evidence": {
            "batch_sizes": batch_sizes,
            "animation_segments": int(m.get("segments", 0)),
            "animation_metrics": int(m.get("metrics", 0)),
            "scene_context_loaded_for_animation_qa": bool(m.get("scene_context_loaded")),
        },
    })
    return phases


def write_path_updates_compliance(paths_root: Path) -> Dict[str, str]:
    report = evaluate_paths_root(paths_root)
    json_path = paths_root / "path_updates_compliance.json"
    md_path = paths_root / "path_updates_compliance.md"
    payload = {
        "schema": "citv_path_updates_compliance_v1",
        "paths_root": str(paths_root),
        "report": report,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Path Updates Compliance",
        "",
        f"- paths: {report.get('paths', 0)}",
        f"- trajectories: {report.get('trajectories', 0)}",
        f"- actions: {report.get('actions', 0)}",
        f"- accepted/plausible/low/rejected paths: {report.get('accepted_paths', 0)}/{report.get('plausible_uncertain_paths', 0)}/{report.get('low_confidence_paths', 0)}/{report.get('rejected_paths', 0)}",
        f"- accepted/plausible/low/rejected actions: {report.get('accepted_actions', 0)}/{report.get('plausible_uncertain_actions', 0)}/{report.get('low_confidence_actions', 0)}/{report.get('rejected_actions', 0)}",
        f"- scene grounding index: {report.get('scene_grounding_index_available', False)}",
        f"- affordance rasters: {report.get('affordance_rasters_available', False)}",
        f"- display geometry paths: {report.get('display_geometry_paths', 0)}",
        f"- path geometry quality contracts: {report.get('path_geometry_quality_contracts', 0)}",
        "",
        "## Phase Status",
        "",
    ]
    for phase in report.get("phase_compliance") or []:
        lines.append(f"- Phase {phase.get('phase')} — {phase.get('name')}: **{phase.get('status')}**")
    lines.extend(["", "## Failures", ""])
    if report.get("failures"):
        for failure in report.get("failures") or []:
            lines.append(f"- {failure}")
    else:
        lines.append("- none")
    regressions = report.get("quality_regressions") if isinstance(report.get("quality_regressions"), dict) else {}
    lines.extend(["", "## Quality Regressions", ""])
    for key, rows in regressions.items():
        lines.append(f"- {key}: {len(rows or [])} sample(s)")
        for pid in list(rows or [])[:8]:
            lines.append(f"  - `{pid}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


__all__ = ["evaluate_paths_root", "write_path_updates_compliance"]
