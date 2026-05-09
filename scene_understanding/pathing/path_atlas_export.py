"""Export path_top30_atlas.md, path_atlas_manifest.json, and line-only panel PNGs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scene_understanding.pathing.path_atlas_canvas import (
    write_path_rank_panels_line_only,
    write_path_rank_panels_overlay,
    write_path_rank_panels_paths_trajectories_overlay,
    write_trajectory_atlas_line_only,
)
from scene_understanding.pathing.path_colors import (
    bgr_from_stable_id,
    bgr_from_stable_id_trajectory,
    bgr_list_with_min_hue_separation,
    bgr_to_hex_rgb,
)


def _route_signature(p: Dict[str, Any]) -> str:
    src = (p.get("source_entity") or {}) if isinstance(p.get("source_entity"), dict) else {}
    tgt = (p.get("target_entity") or {}) if isinstance(p.get("target_entity"), dict) else {}
    sid = str(src.get("id", ""))
    tid = str(tgt.get("id", ""))
    regs = tuple(p.get("regions_traversed") or [])
    return f"{sid}|{tid}|{regs}"


def _rank_pool(
    paths_sorted: List[Dict[str, Any]],
    paths_recommended: List[Dict[str, Any]],
    rank_source: str,
) -> List[Dict[str, Any]]:
    rs = (rank_source or "paths_sorted").strip().lower()
    if rs == "paths_recommended":
        return sorted(
            list(paths_recommended),
            key=lambda x: float((x.get("scores") or {}).get("overall_confidence", 0.0)),
            reverse=True,
        )
    return list(paths_sorted)


def export_trajs_upt_atlas(
    *,
    paths_root_dir: Path,
    path_stem: str,
    track_dir_name: str,
    paths_sorted: List[Dict[str, Any]],
    paths_recommended: List[Dict[str, Any]],
    descriptions_by_id: Dict[str, Any],
    path_num_by_id: Dict[str, int],
    min_conf: float,
    traj_bundle: Optional[Dict[str, Any]],
    cfg: Any,
    width: int,
    height: int,
    export_desc: bool,
    animation_rel: str,
    insertion_rel: str,
    trajectory_rel: str,
    hyp_rel: str,
    desc_rel: str,
    diag_rel: str,
    semantic_rel: str,
    pair_rel: str,
    descriptions_md_rel: str,
    cost_png_rel: str,
    cost_npy_rel: str,
    trav_png_rel: str,
    trav_npy_rel: str,
    img_bgr: Any = None,
    label_map: Any = None,
    objects: Optional[List[Dict[str, Any]]] = None,
    metric_depth_m: Any = None,
) -> Dict[str, str]:
    """
    Write path_atlas_ranked_panel_*.png, optional traj_atlas_line_only.png,
    path_atlas_manifest.json, path_top30_atlas.md.
    Returns relative paths for keys path_atlas_manifest_json, path_top30_atlas_md,
    path_atlas_panel_*_image, path_atlas_trajectory_line_only_image (optional).
    """
    if not bool(getattr(cfg, "path_atlas_enabled", False)):
        return {}

    rank_source = str(getattr(cfg, "path_atlas_rank_source", "paths_sorted") or "paths_sorted")
    pool = _rank_pool(paths_sorted, paths_recommended, rank_source)
    perf = bool(getattr(cfg, "path_qa_perf_mode", False)) if cfg else False
    compliance = bool(getattr(cfg, "path_contract_compliance_mode", False)) if cfg else False
    if perf or compliance:
        total = len(pool)
    else:
        total = max(0, int(getattr(cfg, "path_atlas_total", 50)) if cfg else 50)
    panel_sz = max(1, int(getattr(cfg, "path_atlas_panel_size", 10)) if cfg else 10)
    panel_count = max(1, int(getattr(cfg, "path_atlas_ranked_panels", 5)) if cfg else 5)
    ranked = pool[: total]

    auto_panel = bool(getattr(cfg, "path_atlas_auto_panel_count", False)) if cfg else False
    if auto_panel or perf or compliance:
        panel_count = max(1, int(math.ceil(len(ranked) / float(panel_sz)))) if ranked else 1

    panels: List[List[Dict[str, Any]]] = []
    for i in range(panel_count):
        lo = i * panel_sz
        hi = lo + panel_sz
        panels.append(ranked[lo:hi] if lo < len(ranked) else [])

    rel_base = f"scene_graph/{track_dir_name}/{path_stem}_paths"
    out: Dict[str, str] = {}

    if panels:
        abs_written = write_path_rank_panels_line_only(
            paths_root_dir=paths_root_dir,
            height=height,
            width=width,
            panels=panels,
            cfg=cfg,
        )
        for i, ap in enumerate(abs_written, start=1):
            out[f"path_atlas_panel_{i:02d}_image"] = f"{rel_base}/path_atlas_ranked_panel_{i:02d}.png"
        if img_bgr is not None and bool(getattr(cfg, "path_atlas_overlay_enabled", True)):
            abs_overlay = write_path_rank_panels_overlay(
                paths_root_dir=paths_root_dir,
                base_img_bgr=img_bgr,
                label_map=label_map,
                objects=objects or [],
                metric_depth_m=metric_depth_m,
                panels=panels,
                cfg=cfg,
            )
            for i, _ap in enumerate(abs_overlay, start=1):
                out[f"path_atlas_panel_{i:02d}_context_image"] = f"{rel_base}/path_atlas_ranked_panel_{i:02d}_context.png"
            if bool(getattr(cfg, "path_atlas_export_paths_trajectories_overlay", True)):
                abs_pt = write_path_rank_panels_paths_trajectories_overlay(
                    paths_root_dir=paths_root_dir,
                    base_img_bgr=img_bgr,
                    metric_depth_m=metric_depth_m,
                    panels=panels,
                    traj_bundle=traj_bundle,
                    cfg=cfg,
                )
                for i, _ap in enumerate(abs_pt, start=1):
                    out[f"path_atlas_panel_{i:02d}_paths_trajectories_image"] = (
                        f"{rel_base}/path_atlas_ranked_panel_{i:02d}_paths_trajectories.png"
                    )

    traj_line_rel = ""
    if traj_bundle and bool(getattr(cfg, "path_atlas_export_trajectory_line_only", False)):
        tp = write_trajectory_atlas_line_only(
            paths_root_dir=paths_root_dir,
            height=height,
            width=width,
            traj_bundle=traj_bundle,
            cfg=cfg,
        )
        if tp:
            traj_line_rel = f"{rel_base}/traj_atlas_line_only.png"
            out["path_atlas_trajectory_line_only_image"] = traj_line_rel

    entries: List[Dict[str, Any]] = []
    sig_map: Dict[str, List[str]] = {}
    for panel_idx, plist in enumerate(panels, start=1):
        ids = [str(p.get("path_id", "")) for p in plist if str(p.get("path_id", "")).strip()]
        min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
        cols = bgr_list_with_min_hue_separation(ids, bgr_from_stable_id, min_sep) if ids else []
        id_to_col = dict(zip(ids, cols))
        for p in plist:
            pid = str(p.get("path_id", "")).strip()
            if not pid:
                continue
            bgr = id_to_col.get(pid) or bgr_from_stable_id(pid)
            sig = _route_signature(p)
            sig_map.setdefault(sig, []).append(pid)
            pnum = int(path_num_by_id.get(pid, p.get("path_num", 0)) or 0)
            entries.append(
                {
                    "path_num": pnum,
                    "path_id": pid,
                    "panel_index": panel_idx,
                    "color_bgr": [int(bgr[0]), int(bgr[1]), int(bgr[2])],
                    "color_hex": bgr_to_hex_rgb(bgr),
                    "suppressed": bool(p.get("suppressed", False)),
                    "path_level": str(p.get("path_level", "")),
                    "path_type": str(p.get("path_type", "")),
                    "route_signature": sig,
                    "overall_confidence": float((p.get("scores") or {}).get("overall_confidence", 0.0)),
                }
            )

    traj_color_rows: List[Dict[str, Any]] = []
    if traj_bundle and traj_line_rel:
        hyps = traj_bundle.get("hypotheses") or []
        tids = [str(h.get("trajectory_id", "")) for h in hyps if str(h.get("trajectory_id", "")).strip()]
        min_sep = float(getattr(cfg, "path_atlas_min_hue_separation", 0.07)) if cfg else 0.07
        tcols = bgr_list_with_min_hue_separation(tids, bgr_from_stable_id_trajectory, min_sep) if tids else []
        for tid, bgr in zip(tids, tcols):
            traj_color_rows.append(
                {
                    "trajectory_id": tid,
                    "color_bgr": [int(bgr[0]), int(bgr[1]), int(bgr[2])],
                    "color_hex": bgr_to_hex_rgb(bgr),
                }
            )

    manifest = {
        "schema": "citv_path_atlas_manifest_v1",
        "version": "1.0",
        "image_stem": path_stem,
        "track": track_dir_name,
        "ranking_policy": {
            "rank_source": rank_source,
            "path_atlas_total": total,
            "path_atlas_panel_size": panel_sz,
            "path_atlas_ranked_panels": panel_count,
            "min_confidence_gate": float(min_conf),
        },
        "panels": [
            {
                "index": i + 1,
                "image": f"{rel_base}/path_atlas_ranked_panel_{i + 1:02d}.png",
                "context_image": f"{rel_base}/path_atlas_ranked_panel_{i + 1:02d}_context.png",
                "paths_trajectories_image": f"{rel_base}/path_atlas_ranked_panel_{i + 1:02d}_paths_trajectories.png",
            }
            for i in range(len(panels))
        ],
        "paths": entries,
        "trajectory_colors": traj_color_rows,
        "trajectory_line_only_image": traj_line_rel or None,
    }
    man_path = paths_root_dir / "path_atlas_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out["path_atlas_manifest_json"] = f"{rel_base}/path_atlas_manifest.json"

    # --- Markdown atlas ---
    dup_notes = [f"- Signature `{sig}`: path_ids {', '.join(pids)}" for sig, pids in sig_map.items() if len(pids) > 1]

    md: List[str] = [
        f"# Path trajectory atlas (trajs-upt): `{path_stem}`",
        "",
        "Visual QA uses **line-only** full-resolution panels (no scene photo, region contours, or boxes). "
        "Each `path_id` has a **stable color** (SHA-256 hue); see tables below.",
        "",
        "## Ranking policy",
        "",
        f"- **Rank pool**: `{rank_source}` (`paths_sorted` = recommended-first global rank when available; "
        "`paths_recommended` = non-suppressed only).",
        f"- **Atlas cap**: first **{total}** paths in that pool, split into panels of **{panel_sz}**.",
        f"- **Minimum confidence** used elsewhere in export: `{min_conf:.3f}` (paths may still appear here for QA).",
        "",
        "## Output files",
        "",
    ]
    for k in sorted(out.keys()):
        if k.endswith("_image") or k.endswith("_json") or k.endswith("_md"):
            md.append(f"- `{out[k]}`")
    md.append("")
    md.append("### Related JSON / images")
    md.append("")
    md.append(f"- `{hyp_rel}` — full path hypotheses")
    if desc_rel:
        md.append(f"- `{desc_rel}` — per-path descriptions")
    if descriptions_md_rel:
        md.append(f"- `{descriptions_md_rel}` — explainability report")
    if diag_rel:
        md.append(f"- `{diag_rel}` — diagnostics")
    if semantic_rel:
        md.append(f"- `{semantic_rel}` — semantic layer")
    if pair_rel:
        md.append(f"- `{pair_rel}` — pair proposals")
    if animation_rel:
        md.append(f"- `{animation_rel}` — animation plan")
    if insertion_rel:
        md.append(f"- `{insertion_rel}` — insertion path ensembles")
    if trajectory_rel:
        md.append(f"- `{trajectory_rel}` — trajectory hypotheses")
    if cost_png_rel:
        md.append(f"- `{cost_png_rel}` / `{cost_npy_rel}` — cost field")
    if trav_png_rel:
        md.append(f"- `{trav_png_rel}` / `{trav_npy_rel}` — traversability speed")
    md.append("")
    md.append("## Color legend by panel (paths)")
    md.append("")
    md.append("| path_num | path_id | panel | Hex | suppressed | level | type | confidence |")
    md.append("|----------|---------|-------|-----|------------|-------|------|------------|")
    for e in entries:
        md.append(
            f"| {e['path_num']} | `{e['path_id']}` | {e['panel_index']} | **{e['color_hex']}** | "
            f"{e['suppressed']} | {e['path_level']} | {e['path_type']} | {e['overall_confidence']:.3f} |"
        )
    md.append("")
    if traj_color_rows:
        md.append("## Trajectory colors (`traj_atlas_line_only.png`)")
        md.append("")
        md.append("| trajectory_id | Hex |")
        md.append("|---------------|-----|")
        for row in traj_color_rows:
            md.append(f"| `{row['trajectory_id']}` | **{row['color_hex']}** |")
        md.append("")
    md.append("## Duplicate / related route signatures")
    md.append("")
    if dup_notes:
        md.extend(dup_notes)
    else:
        md.append("- No duplicate signatures among atlas paths.")
    md.append("")
    md.append("## Per-path detail (atlas subset)")
    md.append("")
    for e in entries:
        pid = e["path_id"]
        p = next((x for x in ranked if str(x.get("path_id", "")) == pid), None)
        if p is None:
            continue
        md.append(f"### Path {e['path_num']}: `{pid}`")
        md.append("")
        md.append(f"- **Color**: {e['color_hex']} (BGR `{e['color_bgr']}`)")
        md.append(f"- **Panel**: {e['panel_index']}, **suppressed**: {e['suppressed']}")
        md.append(f"- **Route signature**: `{e['route_signature']}`")
        regs = ", ".join(str(x) for x in (p.get("regions_traversed") or []))
        md.append(f"- **Regions traversed**: {regs or '(none)'}")
        sc = p.get("scores") or {}
        md.append(
            f"- **Scores**: overall {float(sc.get('overall_confidence', 0.0)):.3f}; "
            f"geometric {float(sc.get('geometric_feasibility', 0.0)):.3f}"
        )
        if export_desc and pid in descriptions_by_id:
            d = descriptions_by_id[pid]
            summ = str(d.get("summary", "") or d.get("why_valid", "")).strip()
            if summ:
                md.append(f"- **Description**: {summ}")
        else:
            src = (p.get("source_entity") or {}).get("display_label", "source")
            tgt = (p.get("target_entity") or {}).get("display_label", "target")
            md.append(f"- **Description** (fallback): {src} → {tgt} ({p.get('path_level', '')} / {p.get('path_type', '')}).")
        md.append("")
    md_path = paths_root_dir / "path_top30_atlas.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    out["path_top30_atlas_md"] = f"{rel_base}/path_top30_atlas.md"

    return out
