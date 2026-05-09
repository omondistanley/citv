"""Stage: path hypotheses + traversability export for the staged pipeline.

Reads from ``ctx`` (objects, regions, depth, relations) and calls the
package-internal path-hypothesis machinery.  The exports are written under
``{output_dir}/scene_graph/staged/{stem}_paths/`` and the relative paths
are stored in ``ctx.path_exports`` so ``scene_write`` can embed them in the
scene JSON metadata block.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..action_ontology import list_section, load_action_ontology, number, string
from ..pipeline_context import PipelineContext

_MAX_LABELLED_PATHS = 15


def _build_obstacle_mask(
    objects: List[Dict[str, Any]],
    h: int,
    w: int,
) -> np.ndarray:
    """Union of all object masks → hard-obstacle raster."""
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]

    obs = np.zeros((h, w), dtype=bool)
    for o in objects:
        m = o.get("_sam2_mask_array")
        if m is None:
            continue
        mm = np.asarray(m, dtype=bool)
        if mm.shape[:2] != (h, w):
            if cv2 is not None:
                mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                continue
        obs |= mm
    return obs


def _build_affordance_graded_obstacle_mask(
    objects: List[Dict[str, Any]],
    affordance_rows: List[Dict[str, Any]],
    h: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grade obstacle mask using affordance roles from affordances_export.

    Returns (hard_obs, soft_cost) where hard_obs is the refined binary
    obstacle raster and soft_cost is a [0..1] float array of additional
    speed penalties for soft obstacles.  Objects with high support scores
    are removed from the obstacle mask.  Unknown/low-confidence objects fall
    back to hard obstacle (conservative).
    """
    try:
        import cv2 as _cv2
    except ImportError:
        _cv2 = None  # type: ignore[assignment]

    aff_by_id: Dict[str, Dict[str, float]] = {}
    for row in affordance_rows:
        oid = str(row.get("object_id", "") or "")
        if not oid:
            continue
        roles: Dict[str, float] = {}
        for r in list(row.get("roles") or []):
            if isinstance(r, dict):
                roles[str(r.get("name", ""))] = _float(r.get("score"), 0.0)
        aff_by_id[oid] = roles

    hard_obs = np.zeros((h, w), dtype=bool)
    soft_cost = np.zeros((h, w), dtype=np.float32)

    for obj in objects:
        oid = str(obj.get("id", "") or "")
        m = obj.get("_sam2_mask_array")
        if m is None:
            continue
        mm = np.asarray(m, dtype=bool)
        if mm.shape[:2] != (h, w):
            if _cv2 is not None:
                mm = _cv2.resize(mm.astype(np.uint8), (w, h), interpolation=_cv2.INTER_NEAREST) > 0
            else:
                continue
        roles = aff_by_id.get(oid, {})
        hard = _float(roles.get("hard_obstacle"), 0.5)
        soft = _float(roles.get("soft_obstacle"), 0.0)
        support = _float(roles.get("support"), 0.0)
        if support >= 0.60:
            # Confidently a support surface: remove from obstacle mask.
            pass
        elif hard >= 0.55:
            hard_obs |= mm
        elif soft >= 0.45 or hard >= 0.35:
            # Passable but costly; blend a speed penalty rather than block.
            penalty = np.float32(max(soft, hard * 0.6) * 0.75)
            soft_cost = np.where(mm, np.maximum(soft_cost, penalty), soft_cost)
        else:
            # Unknown / low-confidence: conservative hard block.
            hard_obs |= mm

    return hard_obs, soft_cost


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _build_grounded_generation_artifacts(
    ctx: PipelineContext,
    *,
    objects: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    obs_mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    speed_map: Optional[np.ndarray],
    paths_root: Path,
) -> Optional[np.ndarray]:
    """Promote staged metadata into path-generation artifacts and soft rasters."""
    try:
        from ..pathing.affordance_rasters import build_affordance_rasters, save_affordance_rasters
        from ..pathing.manifold_candidate_generation import build_manifold_candidates
        from ..pathing.scene_grounding_index import build_scene_grounding_index
    except Exception as exc:
        print(f"  [PathsExport] grounded generation imports failed: {exc}")
        return speed_map

    try:
        caption_bundle = ctx.caption_evidence or ctx.extra.get("caption_evidence") or {}
        grounding = build_scene_grounding_index(
            stem=ctx.stem,
            objects=objects,
            regions=regions,
            relations=list(ctx.relations or []),
            scene_affordances=ctx.scene_affordances or {},
            object_affordances=ctx.object_affordances or {},
            mask_affordances=ctx.mask_affordances or {},
            caption_bundle=caption_bundle if isinstance(caption_bundle, dict) else {},
        )
        grounding_path = paths_root / "scene_grounding_index.json"
        _write_json(grounding, grounding_path)
        ctx.extra["scene_grounding_index"] = grounding
        ctx.path_exports["scene_grounding_index_json"] = (
            f"scene_graph/staged/{ctx.stem}_paths/scene_grounding_index.json"
        )

        open_vocab = dict(grounding.get("open_vocab_summary") or {})
        _write_json(open_vocab, paths_root / "open_vocab_grounding.json")
        ctx.extra["open_vocab_grounding"] = open_vocab
        ctx.path_exports["open_vocab_grounding_json"] = (
            f"scene_graph/staged/{ctx.stem}_paths/open_vocab_grounding.json"
        )

        obj_by_id = {str(o.get("id", "")): o for o in objects if str(o.get("id", ""))}
        rasters = build_affordance_rasters(
            grounding_index=grounding,
            objects_by_id=obj_by_id,
            support_mask=support_mask,
            obstacle_mask=obs_mask,
            metric_depth=ctx.metric_depth,
            shape=(int(ctx.height), int(ctx.width)),
        )
        rasters_path = paths_root / "affordance_rasters.npz"
        rasters_manifest = save_affordance_rasters(rasters, rasters_path)
        _write_json(rasters_manifest, paths_root / "affordance_rasters_manifest.json")
        ctx.extra["affordance_rasters"] = rasters
        ctx.extra["affordance_rasters_manifest"] = rasters_manifest
        ctx.path_exports["affordance_rasters_npz"] = (
            f"scene_graph/staged/{ctx.stem}_paths/affordance_rasters.npz"
        )
        ctx.path_exports["affordance_rasters_manifest_json"] = (
            f"scene_graph/staged/{ctx.stem}_paths/affordance_rasters_manifest.json"
        )

        candidate_bundle = build_manifold_candidates(
            grounding_index=grounding,
            rasters_manifest=rasters_manifest,
            max_candidates=96,
        )
        _write_json(candidate_bundle, paths_root / "manifold_candidates.json")
        ctx.extra["grounded_manifold_candidates"] = candidate_bundle
        ctx.path_exports["manifold_candidates_json"] = (
            f"scene_graph/staged/{ctx.stem}_paths/manifold_candidates.json"
        )

        # Feed the raster evidence back into generation. Support evidence can
        # reopen surfaces that the binary obstacle/support mask could not prove;
        # blocker evidence remains a soft speed penalty rather than the only gate.
        if speed_map is not None:
            sm = np.asarray(speed_map, dtype=np.float32).copy()
            support_score = np.asarray(rasters.get("support_surface_score"), dtype=np.float32)
            blocker_score = np.asarray(rasters.get("blocker_score"), dtype=np.float32)
            if support_score.shape == sm.shape:
                sm = np.maximum(sm, support_score * np.float32(0.72))
            if blocker_score.shape == sm.shape:
                sm = np.clip(sm * (1.0 - 0.45 * blocker_score), 0.0, 1.0)
            ctx.extra["path_traversability_speed_map_grounded"] = sm
            return sm
    except Exception as exc:
        print(f"  [PathsExport] grounded generation artifact build failed: {exc}")
    return speed_map


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Export path hypotheses and traversability speed map for the staged context."""
    cfg = getattr(pipeline, "config", None)
    enabled = bool(getattr(cfg, "export_path_hypotheses", True)) if cfg else True
    if not enabled:
        return ctx

    objects: List[Dict[str, Any]] = list(ctx.extra.get("objects", []))
    h, w = ctx.height, ctx.width
    lm = ctx.region_label_map
    regions_block = ctx.regions_block
    region_partition_meta = list(ctx.region_partition_meta)

    staged_dir = ctx.output_dir / "scene_graph" / "staged"

    # Region-level artifacts run regardless of whether objects were detected.
    if regions_block:
        _write_region_relations_json(ctx, regions_block,
                                     staged_dir / f"{ctx.stem}_paths", staged_dir)
        _write_region_relations_map(ctx, regions_block, staged_dir)
    if ctx.mask_hierarchy:
        _write_mask_hierarchy_detailed(ctx, staged_dir / f"{ctx.stem}_paths", staged_dir)
        _write_mask_hierarchy_levels(ctx, regions_block, staged_dir)

    # Object- and traversability-dependent outputs require at least one object.
    if not objects:
        print(f"  [PathsExport] {len(ctx.path_exports)} export keys for {ctx.stem} (no objects)")
        return ctx

    # Need at least a region partition to route paths.
    if lm is None or not region_partition_meta:
        return ctx

    paths_root = staged_dir / f"{ctx.stem}_paths"
    paths_root.mkdir(parents=True, exist_ok=True)

    try:
        from ..pathing.export_workspace import prepare_path_hypotheses_workspace
        adjacency = dict((regions_block or {}).get("adjacency") or {}) if regions_block else {}
        prepare_path_hypotheses_workspace(cfg, ctx.img_bgr, ctx.stem, staged_dir, lm, regions_block, adjacency)
    except Exception as exc:
        print(f"  [PathsExport] workspace preparation failed: {exc}")

    obs_mask = _build_obstacle_mask(objects, h, w)

    # Phase 2 affordance-graded obstacle refinement: when affordance data is
    # available from affordances_export, replace the binary obstacle raster
    # with a role-scored version so support surfaces are freed and soft
    # obstacles become cost penalties rather than hard blocks.
    soft_obs_cost: Optional[np.ndarray] = None
    aff_data = ctx.extra.get("affordances") or {}
    obj_affs = aff_data.get("object") or ctx.object_affordances or {}
    if isinstance(obj_affs, dict) and obj_affs.get("objects"):
        try:
            obs_mask, soft_obs_cost = _build_affordance_graded_obstacle_mask(
                objects, list(obj_affs["objects"]), h, w
            )
            ctx.extra["affordance_graded_obs_mask"] = True
            print(
                f"  [PathsExport] affordance-graded obstacle mask: "
                f"hard={int(obs_mask.sum())} px, "
                f"soft_penalty_nonzero={int((soft_obs_cost > 0).sum())} px"
            )
        except Exception as _aff_exc:
            print(f"  [PathsExport] affordance grading skipped: {_aff_exc}")
            soft_obs_cost = None

    # Phase 2.1: reuse the support mask built by affordances_export when
    # available; otherwise build here so paths still anchor correctly when
    # affordance export was disabled.
    support_mask: Optional[np.ndarray] = ctx.extra.get("support_mask")
    if support_mask is None:
        try:
            from ..pathing.ground_plane import build_support_mask
            support_mask, support_info = build_support_mask(
                ctx.metric_depth,
                ctx.intrinsics,
                lm,
                list(region_partition_meta),
                object_mask=obs_mask,
            )
            if support_mask is not None and support_mask.any():
                ctx.extra["support_mask"] = support_mask
                ctx.extra["support_mask_info"] = support_info
                print(
                    f"  [PathsExport] support mask: union={int(support_mask.sum())} px "
                    f"(plane={support_info.get('plane_pixel_count', 0)}, "
                    f"semantic={support_info.get('semantic_pixel_count', 0)})"
                )
            else:
                print("  [PathsExport] support mask empty; falling back to walkable")
        except Exception as _se:
            print(f"  [PathsExport] support mask build failed: {_se}")

    # Traversability speed map(s): base + optional fused semantic cost + multi-channel.
    speed_map: Optional[np.ndarray] = None
    trav_meta: Dict[str, Any] = {}
    try:
        from ..pathing.traversability import build_traversability_speed_map
        K = dict(ctx.intrinsics) if ctx.intrinsics else None
        speed_base, trav_meta = build_traversability_speed_map(
            ctx.metric_depth, lm, obs_mask, ctx.img_bgr, cfg, K=K
        )
        speed_map = speed_base

        # Apply affordance-derived soft obstacle penalty to the base speed map.
        # This is the Phase 2 generative wiring: objects graded as soft obstacles
        # (rather than hard blocks) reduce traversal speed without full exclusion.
        if soft_obs_cost is not None and soft_obs_cost.shape == speed_map.shape:
            s_floor = _float(getattr(cfg, "trav_speed_floor", 0.06), 0.06) if cfg else 0.06
            speed_map = np.clip(
                speed_map * (1.0 - 0.65 * soft_obs_cost.astype(np.float32)),
                s_floor, 1.0,
            )
            ctx.extra["affordance_soft_obstacle_applied"] = True

        fused_meta: Dict[str, Any] = {}
        if bool(getattr(cfg, "path_use_fused_semantic_cost", False)) if cfg else False:
            try:
                from ..pathing.semantic_provenance import save_semantic_cost_artifacts
                from ..pathing.staged_semantic_cost import (
                    blend_traversability_with_fused_cost,
                    build_staged_semantic_cost_map,
                )

                scm = build_staged_semantic_cost_map(
                    ctx.img_bgr,
                    ctx.metric_depth,
                    obs_mask,
                    objects,
                    cfg,
                    K,
                )
                blend_w = _float(getattr(cfg, "path_fused_semantic_cost_blend", 0.72), 0.72) if cfg else 0.72
                s_floor = _float(getattr(cfg, "trav_speed_floor", 0.06), 0.06) if cfg else 0.06
                speed_map = blend_traversability_with_fused_cost(
                    speed_base, scm.cost, blend=blend_w, speed_floor=s_floor
                )
                fused_meta = dict(scm.meta or {})
                ctx.extra["staged_semantic_cost_meta"] = fused_meta
                emit_cost = bool(getattr(cfg, "path_cost_emit_layers_for_qa", False)) if cfg else False
                if emit_cost:
                    save_semantic_cost_artifacts(paths_root, scm)
                    ctx.path_exports["cost_map_npz"] = f"scene_graph/staged/{ctx.stem}_paths/cost_map.npz"
                    ctx.path_exports["cost_meta_json"] = f"scene_graph/staged/{ctx.stem}_paths/cost_meta.json"
                    ctx.path_exports["cost_provenance_png"] = (
                        f"scene_graph/staged/{ctx.stem}_paths/cost_provenance.png"
                    )
            except Exception as _fuse_exc:
                print(f"  [PathsExport] fused semantic cost skipped: {_fuse_exc}")
                speed_map = speed_base

        multi_ch = bool(getattr(cfg, "path_multi_channel_traversability", False)) if cfg else False
        if multi_ch and speed_map is not None:
            try:
                from ..pathing.staged_semantic_cost import (
                    blend_traversability_with_fused_cost,
                    build_staged_semantic_cost_map,
                )

                s_floor = _float(getattr(cfg, "trav_speed_floor", 0.06), 0.06) if cfg else 0.06
                blend_w = _float(getattr(cfg, "path_fused_semantic_cost_blend", 0.72), 0.72) if cfg else 0.72
                sp_fluid, _ = build_traversability_speed_map(
                    ctx.metric_depth, lm, obs_mask, ctx.img_bgr, cfg, K=K, actor_type="fluid"
                )
                sp_air, _ = build_traversability_speed_map(
                    ctx.metric_depth, lm, obs_mask, ctx.img_bgr, cfg, K=K, actor_type="aerial"
                )
                try:
                    scm_mc = build_staged_semantic_cost_map(
                        ctx.img_bgr, ctx.metric_depth, obs_mask, objects, cfg, K
                    )
                    sp_fluid = blend_traversability_with_fused_cost(
                        sp_fluid, scm_mc.cost * 0.85, blend=blend_w * 0.9, speed_floor=s_floor
                    )
                    sp_air = blend_traversability_with_fused_cost(
                        sp_air, scm_mc.cost * 0.35, blend=blend_w * 0.45, speed_floor=s_floor
                    )
                except Exception:
                    pass
                np.save(str(paths_root / "path_traversability_speed_fluid.npy"), sp_fluid)
                np.save(str(paths_root / "path_traversability_speed_aerial.npy"), sp_air)
                ctx.path_exports["traversability_speed_fluid_npy"] = (
                    f"scene_graph/staged/{ctx.stem}_paths/path_traversability_speed_fluid.npy"
                )
                ctx.path_exports["traversability_speed_aerial_npy"] = (
                    f"scene_graph/staged/{ctx.stem}_paths/path_traversability_speed_aerial.npy"
                )
                ctx.extra["speed_map_fluid"] = sp_fluid
                ctx.extra["speed_map_aerial"] = sp_air
            except Exception as _mc_exc:
                print(f"  [PathsExport] multi-channel traversability skipped: {_mc_exc}")

        speed_map = _build_grounded_generation_artifacts(
            ctx,
            objects=objects,
            regions=list(region_partition_meta),
            obs_mask=obs_mask,
            support_mask=support_mask,
            speed_map=speed_map,
            paths_root=paths_root,
        )

        ts_npy = paths_root / "path_traversability_speed.npy"
        np.save(str(ts_npy), speed_map)
        ctx.extra["path_traversability_speed_map"] = speed_map
        try:
            import cv2
            ts_u8 = np.clip(speed_map * 255.0, 0, 255).astype(np.uint8)
            ts_color = cv2.applyColorMap(ts_u8, cv2.COLORMAP_VIRIDIS)
            ts_png = paths_root / "path_traversability_speed.png"
            cv2.imwrite(str(ts_png), ts_color)
            ctx.path_exports["traversability_speed_npy"] = (
                f"scene_graph/staged/{ctx.stem}_paths/path_traversability_speed.npy"
            )
            ctx.path_exports["traversability_speed_png"] = (
                f"scene_graph/staged/{ctx.stem}_paths/path_traversability_speed.png"
            )
        except Exception:
            pass
        ctx.path_exports["traversability_meta"] = trav_meta
    except Exception as exc:
        print(f"  [PathsExport] traversability skipped: {exc}")
        speed_map = None

    # Walkable mask snapshot.
    try:
        from ..pathing.walkable_mask import build_path_walkable_mask
        if speed_map is not None:
            walkable, walk_meta = build_path_walkable_mask(
                lm, obs_mask, speed_map, cfg,
                regions_meta=region_partition_meta,
                support_mask=support_mask,
            )
            ctx.path_exports["walkable_ratio"] = walk_meta.get("walkable_ratio", 0.0)
    except Exception:
        pass

    # Navigation zone map (idle/walk/run/careful) — built once per image.
    nav_zones: Optional[np.ndarray] = None
    if speed_map is not None:
        try:
            import cv2 as _cv2
            from ..pathing.navigation_zones import (
                build_navigation_zones,
                navigation_zones_to_rgba,
            )
            _feasible_nz = (lm > 0) & (~obs_mask)
            _cost_nz = 1.0 - np.clip(speed_map, 0.0, 1.0)
            _sa_layer = dict(ctx.scene_affordances or {})
            nav_zones, _nz_meta = build_navigation_zones(
                _cost_nz,
                _feasible_nz,
                lm,
                speed_map=speed_map,
                metric_depth_m=ctx.metric_depth,
                semantic_layer=_sa_layer,
                regions_meta=list(region_partition_meta),
            )
            _nz_npy = paths_root / "navigation_zones.npy"
            np.save(str(_nz_npy), nav_zones)
            _nz_bgr = navigation_zones_to_rgba(nav_zones)
            _nz_png = paths_root / "navigation_zones.png"
            _cv2.imwrite(str(_nz_png), _nz_bgr)
            ctx.extra["navigation_zones"] = nav_zones
            ctx.path_exports["navigation_zones_npy"] = (
                f"scene_graph/staged/{ctx.stem}_paths/navigation_zones.npy"
            )
            ctx.path_exports["navigation_zones_png"] = (
                f"scene_graph/staged/{ctx.stem}_paths/navigation_zones.png"
            )
        except Exception as _exc:
            print(f"  [PathsExport] navigation zones failed: {_exc}")

    # FMM path hypotheses (requires scikit-fmm + region label map).
    if speed_map is not None:
        _write_fmm_path_hypotheses(ctx, pipeline, objects, lm, obs_mask, speed_map, paths_root)

    # Semantic cost map QA when layered export requested but fused block did not write artifacts.
    if speed_map is not None and bool(getattr(cfg, "path_cost_emit_layers_for_qa", False) if cfg else False):
        if "cost_map_npz" not in ctx.path_exports:
            try:
                from ..pathing.semantic_cost import CostLayer, precision_weighted_fuse
                from ..pathing.semantic_provenance import save_semantic_cost_artifacts
                cost_arr = (1.0 - np.clip(speed_map, 0.0, 1.0)).astype(np.float32)
                prec_arr = np.where(speed_map > 1e-6, 1.0, 0.0).astype(np.float32)
                layer = CostLayer(
                    name="traversability", cost=cost_arr, precision=prec_arr,
                    provenance="speed_map_inversion", diagnostics={},
                )
                scm = precision_weighted_fuse([layer])
                save_semantic_cost_artifacts(paths_root, scm)
                ctx.path_exports["cost_map_npz"] = f"scene_graph/staged/{ctx.stem}_paths/cost_map.npz"
                ctx.path_exports["cost_meta_json"] = f"scene_graph/staged/{ctx.stem}_paths/cost_meta.json"
                ctx.path_exports["cost_provenance_png"] = f"scene_graph/staged/{ctx.stem}_paths/cost_provenance.png"
            except Exception as exc:
                print(f"  [PathsExport] semantic cost artifacts failed: {exc}")

    print(
        f"  [PathsExport] done; {len(ctx.path_exports)} export keys for {ctx.stem}"
    )
    return ctx


def _write_region_relations_json(
    ctx: PipelineContext,
    regions_block: Dict[str, Any],
    paths_root: Path,
    staged_dir: Path,
) -> None:
    """Write ``{stem}_region_relations.json`` under staged dir."""
    try:
        regions_list = regions_block.get("regions", [])
        adjacency = regions_block.get("adjacency") or {}
        rr_payload: Dict[str, Any] = {
            "image": str(ctx.image_path),
            "stem": ctx.stem,
            "timestamp": ctx.timestamp,
            "region_count": len(regions_list),
            "regions": regions_list,
            "adjacency": adjacency,
            "relations": [],
        }
        # Build region-level relations from adjacency graph.
        rel_list = []
        for rid, neighbours in (adjacency.items() if isinstance(adjacency, dict) else []):
            for nid in (neighbours if isinstance(neighbours, list) else []):
                rel_list.append({
                    "subject": str(rid),
                    "predicate": "adjacent_to",
                    "object": str(nid),
                    "source_layer": "depth_partition",
                })
        rr_payload["relations"] = rel_list
        out_path = staged_dir / f"{ctx.stem}_region_relations.json"
        _write_json(rr_payload, out_path)
        ctx.path_exports["region_relations_json"] = (
            f"scene_graph/staged/{ctx.stem}_region_relations.json"
        )
    except Exception as exc:
        print(f"  [PathsExport] region_relations write failed: {exc}")


def _write_mask_hierarchy_detailed(
    ctx: PipelineContext,
    paths_root: Path,
    staged_dir: Path,
) -> None:
    """Write ``{stem}_mask_hierarchy_detailed.json`` with containment depth per node."""
    try:
        hier = ctx.mask_hierarchy or {}

        def _annotate_depth(node: Any, depth: int) -> Any:
            if isinstance(node, dict):
                node = dict(node)
                node["containment_depth"] = depth
                children = node.get("children")
                if isinstance(children, list):
                    node["children"] = [_annotate_depth(c, depth + 1) for c in children]
            elif isinstance(node, list):
                node = [_annotate_depth(c, depth) for c in node]
            return node

        detailed = _annotate_depth(hier, 0)
        out_path = staged_dir / f"{ctx.stem}_mask_hierarchy_detailed.json"
        _write_json({"stem": ctx.stem, "hierarchy": detailed}, out_path)
        ctx.path_exports["mask_hierarchy_detailed_json"] = (
            f"scene_graph/staged/{ctx.stem}_mask_hierarchy_detailed.json"
        )
    except Exception as exc:
        print(f"  [PathsExport] mask_hierarchy_detailed write failed: {exc}")


def _write_mask_hierarchy_levels(
    ctx: PipelineContext,
    regions_block: Optional[Dict[str, Any]],
    staged_dir: Path,
) -> None:
    """Write ``{stem}_mask_hierarchy_levels.json`` grouped by depth layer."""
    try:
        regions = list((regions_block or {}).get("regions", []) or [])
        levels = []
        for lname in ("foreground", "midground", "background", "far_background"):
            rows = []
            for r in regions:
                if str(r.get("layer_type", r.get("type", ""))).lower() != lname:
                    continue
                rows.append({
                    "region_id": str(r.get("id", "")),
                    "region_index": int(r.get("region_index", 0) or 0),
                    "label": str(r.get("semantic_label", r.get("type", ""))),
                    "depth_z": float((r.get("depth_stats") or {}).get("mean", 0.0) or 0.0),
                    "object_ids": list(r.get("object_ids", [])),
                })
            if rows:
                levels.append({"layer": lname, "region_count": len(rows), "regions": rows})
        if not levels and regions:
            levels.append({
                "layer": "unclassified",
                "region_count": len(regions),
                "regions": [
                    {
                        "region_id": str(r.get("id", "")),
                        "region_index": int(r.get("region_index", 0) or 0),
                        "label": str(r.get("semantic_label", r.get("type", ""))),
                        "depth_z": float((r.get("depth_stats") or {}).get("mean", 0.0) or 0.0),
                        "object_ids": list(r.get("object_ids", [])),
                    }
                    for r in regions
                ],
            })
        out_path = staged_dir / f"{ctx.stem}_mask_hierarchy_levels.json"
        _write_json(
            {
                "stem": ctx.stem,
                "timestamp": ctx.timestamp,
                "schema": "citv_mask_hierarchy_levels_v1",
                "level_count": len(levels),
                "levels": levels,
            },
            out_path,
        )
        ctx.path_exports["mask_hierarchy_levels_json"] = (
            f"scene_graph/staged/{ctx.stem}_mask_hierarchy_levels.json"
        )
    except Exception as exc:
        print(f"  [PathsExport] mask_hierarchy_levels write failed: {exc}")


def _write_region_relations_map(
    ctx: PipelineContext,
    regions_block: Dict[str, Any],
    staged_dir: Path,
) -> None:
    """Write a region-region relation map with centroid edges."""
    try:
        import cv2
        canvas = ctx.img_bgr.copy()
        regions = list(regions_block.get("regions", []) or [])
        adjacency = regions_block.get("adjacency") or {}
        centroids = {}
        for r in regions:
            rid = str(r.get("id", ""))
            c = r.get("centroid_2d_px") or [ctx.width // 2, ctx.height // 2]
            cx = int(min(max(0, float(c[0])), ctx.width - 1))
            cy = int(min(max(0, float(c[1])), ctx.height - 1))
            centroids[rid] = (cx, cy)
        for rid, neighbours in (adjacency.items() if isinstance(adjacency, dict) else []):
            a = centroids.get(str(rid))
            if not a:
                continue
            for nid in neighbours if isinstance(neighbours, list) else []:
                b = centroids.get(str(nid))
                if b:
                    cv2.arrowedLine(canvas, a, b, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.15)
                    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
                    cv2.putText(canvas, "adjacent_to", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 80), 1, cv2.LINE_AA)
        for rid, cxy in centroids.items():
            cv2.circle(canvas, cxy, 6, (80, 255, 80), -1, cv2.LINE_AA)
            cv2.putText(canvas, rid, (cxy[0] + 8, cxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
        out_path = staged_dir / f"{ctx.stem}_region_relations_map.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        ctx.path_exports["region_relations_map_image"] = (
            f"scene_graph/staged/{ctx.stem}_region_relations_map.png"
        )
    except Exception as exc:
        print(f"  [PathsExport] region_relations_map write failed: {exc}")


def _write_fmm_path_hypotheses(
    ctx: PipelineContext,
    pipeline: Any,
    objects: List[Dict[str, Any]],
    lm: np.ndarray,
    obs_mask: np.ndarray,
    speed_map: np.ndarray,
    paths_root: Path,
) -> None:
    """Compute and write FMM-based path hypotheses for object pairs."""
    try:
        from ..pathing.semantic_fmm import k_diverse_from_T, time_of_arrival_from_speed
        from ..pathing.walkable_mask import snap_uv_to_walkable
    except ImportError as exc:
        print(f"  [PathsExport] FMM import failed: {exc}")
        return

    cfg = getattr(pipeline, "config", None)
    h, w = ctx.height, ctx.width
    support_mask: Optional[np.ndarray] = ctx.extra.get("support_mask")
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)

    speed_floor = _float(getattr(cfg, "trav_speed_floor", 0.06), 0.06) if cfg else 0.06
    try:
        from ..pathing.routing_feasible import build_feasible_base, build_feasible_routing

        feasible_base = build_feasible_base(lm, obs_mask, speed_map, speed_floor=speed_floor)
        feasible, feas_variant = build_feasible_routing(
            feasible_base, lm, obs_mask, speed_map, support_mask, cfg, speed_floor=speed_floor
        )
    except Exception:
        physical_gate = speed_map > (speed_floor * 1.5)
        feasible_base = (lm > 0) & (~obs_mask) & physical_gate
        feasible = feasible_base
        feas_variant = "base"

    ctx.extra["path_feasible_variant"] = feas_variant
    if bool(getattr(cfg, "path_export_feasible_routing_png", False)) if cfg else False:
        try:
            import cv2 as _cv2

            vis = (feasible.astype(np.float32) * 255.0).astype(np.uint8)
            vis_bgr = _cv2.cvtColor(vis, _cv2.COLOR_GRAY2BGR)
            _cv2.imwrite(str(paths_root / "path_feasible_routing.png"), vis_bgr)
            ctx.path_exports["path_feasible_routing_png"] = (
                f"scene_graph/staged/{ctx.stem}_paths/path_feasible_routing.png"
            )
        except Exception:
            pass

    k = int(getattr(cfg, "path_geodesic_k_alt", 2)) if cfg else 2
    pen = _float(getattr(cfg, "path_geodesic_edge_penalty", 0.35), 0.35) if cfg else 0.35
    top_k = int(getattr(cfg, "path_top_k_per_pair", 3)) if cfg else 3

    # ENHANCEMENT (plan §2.3): prefer foot/support-contact anchors over the
    # object's mass centroid. Centroids place locomotion endpoints at the
    # actor's chest/hip; foot/support anchors place them at the floor pixel
    # directly below the actor so paths visually "land" instead of hanging.
    obj_aff_by_id = {
        str(o.get("object_id", "")): o
        for o in list((ctx.object_affordances or {}).get("objects") or [])
        if isinstance(o, dict)
    }

    # Collect goals first; ground-intercept repair for vertical targets; then cache T(goal).
    goal_T: Dict[Any, Optional[np.ndarray]] = {}
    obj_goals: Dict[str, Tuple[int, int]] = {}
    anchor_provenance_log: Dict[str, str] = {}
    for o in objects:
        oid = str(o.get("id", ""))
        aff = obj_aff_by_id.get(oid, {})
        anchors = aff.get("anchors", {})

        priority_uvs: List[Tuple[List[Any], str]] = []
        if isinstance(anchors.get("support_contact_uv"), (list, tuple)):
            priority_uvs.append((anchors["support_contact_uv"], "support_contact_uv"))
        if isinstance(anchors.get("foot_uv"), (list, tuple)):
            priority_uvs.append((anchors["foot_uv"], "foot_uv"))
        for ap in anchors.get("approach_points", []) or []:
            priority_uvs.append((ap, "approach_point"))
        for cp in anchors.get("contact_points", []) or []:
            priority_uvs.append((cp, "contact_point"))

        fallback_uv = (
            o.get("mask_centroid_2d")
            or o.get("mask_centroid_2d_no_erosion")
            or _bbox_center_uv(o)
            or [0, 0]
        )
        try:
            best_gp = snap_uv_to_walkable(int(fallback_uv[0]), int(fallback_uv[1]), feasible, w, h)
        except (TypeError, IndexError):
            best_gp = snap_uv_to_walkable(0, 0, feasible, w, h)
        chosen_source = "mask_centroid_fallback"
        for cand, source in priority_uvs:
            if len(cand) >= 2:
                try:
                    cx_i = int(round(float(cand[0])))
                    cy_i = int(round(float(cand[1])))
                except (TypeError, ValueError):
                    continue
                if 0 <= cy_i < h and 0 <= cx_i < w and feasible[cy_i, cx_i]:
                    best_gp = (cx_i, cy_i)
                    chosen_source = source
                    break
                snapped = snap_uv_to_walkable(cx_i, cy_i, feasible, w, h)
                if snapped != best_gp:
                    best_gp = snapped
                    chosen_source = f"{source}_snapped"
                    break

        obj_goals[oid] = best_gp
        anchor_provenance_log[oid] = chosen_source

    if bool(getattr(cfg, "path_goal_ground_intercept_vertical", False)) if cfg else False:
        try:
            from ..pathing.goal_anchors import ground_intercept_goal_uv, vertical_structure_heuristic

            vthresh = _float(getattr(cfg, "path_goal_vertical_aspect_thresh", 1.35), 1.35) if cfg else 1.35
            for o in objects:
                oid = str(o.get("id", ""))
                if not oid or not vertical_structure_heuristic(o, h, w, aspect_thresh=vthresh):
                    continue
                gi = ground_intercept_goal_uv(
                    o, support_mask, feasible, h, w, snap_walkable=snap_uv_to_walkable
                )
                if gi is not None:
                    obj_goals[oid] = gi
                    anchor_provenance_log[oid] = (
                        anchor_provenance_log.get(oid, "") + "|ground_intercept_uv"
                    ).strip("|")
        except Exception:
            pass

    for gp in set(obj_goals.values()):
        if gp not in goal_T:
            try:
                sm = np.where(feasible, speed_map, speed_map * 0.02)
                goal_T[gp] = time_of_arrival_from_speed(sm, gp)
            except Exception:
                goal_T[gp] = None
    ctx.extra["path_anchor_provenance"] = anchor_provenance_log

    # Plan §2.4: per-actor CC. Pairs whose foot anchors fall in different
    # connected components of the walkable mask cannot route normally; we
    # emit a portal_path candidate instead so the cross-region intent is
    # explicit rather than silently snapped to the dominant CC.
    actor_cc: Dict[str, int] = {}
    cc_labels: Optional[np.ndarray] = None
    try:
        from ..pathing.walkable_mask import per_actor_connected_components
        ordered_ids = [str(o.get("id", "")) for o in objects]
        ordered_uvs = [obj_goals.get(oid, (0, 0)) for oid in ordered_ids]
        cc_labels, per_actor_cc = per_actor_connected_components(feasible, ordered_uvs)
        actor_cc = {oid: cc for oid, cc in zip(ordered_ids, per_actor_cc)}
        ctx.extra["per_actor_cc"] = actor_cc
    except Exception as _cce:
        print(f"  [PathsExport] per-actor CC skipped: {_cce}")

    hypotheses: List[Dict[str, Any]] = []
    done_pairs: set = set()
    for src in objects:
        src_id = str(src.get("id", ""))
        sp = obj_goals.get(src_id)
        if sp is None:
            continue
        for tgt in objects:
            tgt_id = str(tgt.get("id", ""))
            if not tgt_id or tgt_id == src_id:
                continue
            pair_key = tuple(sorted([src_id, tgt_id]))
            if pair_key in done_pairs:
                continue
            done_pairs.add(pair_key)
            gp = obj_goals.get(tgt_id)
            if gp is None:
                continue
            if sp == gp:
                continue

            try:
                from ..pathing.goal_anchors import pair_passes_locomotion_gates

                rel_list = list(getattr(ctx, "relations", []) or [])
                ok_pair, _rej = pair_passes_locomotion_gates(
                    src, tgt, rel_list, cfg, h=h, w=w
                )
                if not ok_pair:
                    continue
            except Exception:
                pass

            # Cross-CC pair → emit portal_path instead of forcing FMM through
            # an unreachable corridor (§2.4 plan).
            sc = actor_cc.get(src_id, 0)
            tc = actor_cc.get(tgt_id, 0)
            if sc and tc and sc != tc:
                bridged_ok = False
                if bool(getattr(cfg, "path_cc_bridge_portals_enabled", True)) if cfg else True:
                    try:
                        from ..pathing.routing_feasible import (
                            build_feasible_bridge,
                            cc_label_at,
                            connected_labels,
                        )

                        fb = build_feasible_bridge(
                            feasible, lm, obs_mask, speed_map, support_mask, cfg, speed_floor=speed_floor
                        )
                        lab, _nlab = connected_labels(fb)
                        la = cc_label_at(lab, sp, h, w)
                        lb = cc_label_at(lab, gp, h, w)
                        if la > 0 and la == lb:
                            sm_fb = np.where(fb, speed_map, speed_map * 0.02)
                            tkey = ("bridge", gp)
                            if tkey not in goal_T:
                                goal_T[tkey] = time_of_arrival_from_speed(sm_fb, gp)
                            Tb = goal_T.get(tkey)
                            if Tb is not None:
                                gpaths_b = k_diverse_from_T(Tb, sp, k=min(k, top_k), edge_penalty=pen)
                                for kidx, gpath in enumerate(gpaths_b, start=1):
                                    if len(gpath) < 2:
                                        continue
                                    raw_poly = [list(p) for p in gpath]
                                    smooth_poly = _bezier_smooth_polyline(raw_poly, w, h, fb)
                                    smooth_poly = _snap_polyline_to_feasible(
                                        smooth_poly, fb, w, h, cfg, snap_uv_to_walkable
                                    )
                                    hypotheses.append({
                                        "path_id": f"staged_fmm_bridge_{src_id}_to_{tgt_id}_k{kidx:02d}",
                                        "path_level": "object",
                                        "path_type": "fmm_geodesic",
                                        "manifold_type": "ribbon_path",
                                        "action_family": "locomotion",
                                        "source_entity": {"type": "object", "id": src_id, "start_uv": list(sp)},
                                        "target_entity": {"type": "object", "id": tgt_id, "goal_uv": list(gp)},
                                        "polyline_2d_raw": raw_poly,
                                        "polyline_2d": smooth_poly,
                                        "scores": {"overall_confidence": 0.85 / float(kidx)},
                                        "routing_meta": {
                                            "feasible_variant": feas_variant,
                                            "cross_cc_bridge": True,
                                            "src_cc": int(sc),
                                            "tgt_cc": int(tc),
                                            "motion_channel": "ground",
                                            "path_granularity": "object_pair",
                                        },
                                    })
                                bridged_ok = True
                    except Exception:
                        bridged_ok = False
                if not bridged_ok:
                    portal_poly = _curved_transition_polyline(sp, gp, w, h)
                    hypotheses.append({
                        "path_id": f"staged_portal_{src_id}_to_{tgt_id}",
                        "path_level": "object",
                        "path_type": "portal",
                        "manifold_type": "portal_path",
                        "action_family": "transition",
                        "source_entity": {"type": "object", "id": src_id, "start_uv": list(sp)},
                        "target_entity": {"type": "object", "id": tgt_id, "goal_uv": list(gp)},
                        "polyline_2d": portal_poly,
                        "scores": {"overall_confidence": 0.4},
                        "portal_reason": f"src_cc={sc}_tgt_cc={tc}_disjoint_walkable",
                        "routing_meta": {
                            "feasible_variant": feas_variant,
                            "shape_fallback": "curved_portal_transition",
                            "straight_line_forced": False,
                        },
                    })
                continue

            T = goal_T.get(gp)
            if T is None:
                continue
            try:
                use_manifold = bool(getattr(cfg, "path_fmm_on_ground_manifold", False)) if cfg else False
                coarse_path: List[Tuple[int, int]] = []
                if use_manifold and support_mask is not None:
                    try:
                        from ..pathing.ground_route_grid import plan_coarse_support_path

                        step_g = int(getattr(cfg, "path_ground_manifold_grid_step_px", 14)) if cfg else 14
                        coarse_path = plan_coarse_support_path(
                            speed_map, support_mask, feasible, sp, gp, step=step_g
                        )
                    except Exception:
                        coarse_path = []
                gpaths = k_diverse_from_T(T, sp, k=min(k, top_k), edge_penalty=pen)
                used_coarse_insert = False
                if coarse_path and len(coarse_path) >= 2:
                    span_c = sum(
                        math.hypot(
                            float(coarse_path[i][0]) - float(coarse_path[i - 1][0]),
                            float(coarse_path[i][1]) - float(coarse_path[i - 1][1]),
                        )
                        for i in range(1, len(coarse_path))
                    )
                    if span_c >= 12.0:
                        cap = min(k, top_k)
                        gpaths = [coarse_path] + [p for p in gpaths if p != coarse_path][: max(0, cap - 1)]
                        used_coarse_insert = True
                for kidx, gpath in enumerate(gpaths, start=1):
                    if len(gpath) < 2:
                        continue
                    raw_poly = [list(p) for p in gpath]
                    smooth_poly = _bezier_smooth_polyline(raw_poly, w, h, feasible)
                    smooth_poly = _snap_polyline_to_feasible(
                        smooth_poly, feasible, w, h, cfg, snap_uv_to_walkable
                    )
                    rm = {
                        "feasible_variant": feas_variant,
                        "motion_channel": "ground",
                        "path_granularity": "object_pair",
                    }
                    if used_coarse_insert and kidx == 1:
                        rm["ground_manifold_coarse"] = True
                    hypotheses.append({
                        "path_id": f"staged_fmm_{src_id}_to_{tgt_id}_k{kidx:02d}",
                        "path_level": "object",
                        "path_type": "fmm_geodesic",
                        "manifold_type": "ribbon_path",
                        "action_family": "locomotion",
                        "source_entity": {"type": "object", "id": src_id, "start_uv": list(sp)},
                        "target_entity": {"type": "object", "id": tgt_id, "goal_uv": list(gp)},
                        "polyline_2d_raw": raw_poly,
                        "polyline_2d": smooth_poly,
                        "scores": {"overall_confidence": 1.0 / kidx},
                        "routing_meta": rm,
                    })
            except Exception:
                continue

    fmm_ribbon_count = sum(1 for h in hypotheses if h.get("manifold_type") == "ribbon_path")
    policy = str(getattr(cfg, "path_region_fmm_policy", "when_sparse")) if cfg else "when_sparse"
    min_rib = int(getattr(cfg, "path_region_fmm_min_ribbons", 0)) if cfg else 0
    max_extra = int(getattr(cfg, "path_region_fmm_max_extra", 24)) if cfg else 24
    want_region_extra = False
    if policy == "always_extra":
        want_region_extra = lm is not None and max_extra > 0
    elif policy == "when_sparse":
        want_region_extra = lm is not None and (fmm_ribbon_count < max(1, min_rib) or fmm_ribbon_count == 0)
    # legacy: only when zero ribbons
    elif policy == "legacy":
        want_region_extra = fmm_ribbon_count == 0 and lm is not None

    if want_region_extra and bool(getattr(cfg, "path_enable_region", True)) if cfg else True:
        try:
            max_labels = int(getattr(cfg, "path_region_fmm_max_labels_sampled", 12)) if cfg else 12
            region_hyps = _region_level_fmm_paths(
                ctx, lm, feasible, speed_map, k, pen, top_k, paths_root, w, h,
                max_hypotheses=max_extra if policy != "legacy" else 20,
                max_labels_sampled=max_labels,
            )
            hypotheses.extend(region_hyps)
        except Exception as _re:
            print(f"  [PathsExport] region-level FMM failed: {_re}")

    if bool(getattr(cfg, "path_object_region_enabled", True)) if cfg else True:
        try:
            _append_object_to_region_paths(
                ctx,
                cfg,
                objects,
                obj_goals,
                lm,
                feasible,
                speed_map,
                k,
                pen,
                top_k,
                hypotheses,
                w,
                h,
                feas_variant,
            )
        except Exception as _ore:
            print(f"  [PathsExport] object→region paths skipped: {_ore}")

    # Grounded manifold candidates are generated from the fused scene/object/mask
    # index before validation.  They are first-class path candidates, not QA-only
    # annotations, and preserve plausible local actions even when support is not
    # fully proven by the binary support mask.
    grounded_bundle = ctx.extra.get("grounded_manifold_candidates")
    if isinstance(grounded_bundle, dict):
        grounded_hyps = [
            h for h in list(grounded_bundle.get("candidates") or [])
            if isinstance(h, dict) and h.get("polyline_2d")
        ]
        hypotheses.extend(grounded_hyps)

    # Non-locomotion manifold hypotheses (blob, volume, occlusion_pulse, portal, effect, contact).
    manifold_hyps = _generate_manifold_hypotheses(ctx, objects, feasible, speed_map, load_action_ontology(cfg))
    hypotheses.extend(manifold_hyps)
    try:
        hypotheses.extend(_emit_aerial_approach_hypotheses(ctx, objects, cfg))
        hypotheses.extend(_emit_contour_hypotheses(ctx, objects, cfg))
    except Exception as _ex_man:
        print(f"  [PathsExport] extra manifold hypotheses skipped: {_ex_man}")

    if hypotheses:
        generated_hypotheses = list(hypotheses)
        max_cand = int(getattr(cfg, "path_max_candidates", 500)) if cfg else 500
        if len(hypotheses) > max_cand:
            hypotheses.sort(
                key=lambda r: float((r.get("scores") or {}).get("overall_confidence", 0.0)),
                reverse=True,
            )
            del hypotheses[max_cand:]

        # v3: deduplicate paths (single key `hypotheses`).  Raw, validated,
        # and display polylines are additive contract fields used by QA and
        # animation; legacy consumers may continue reading `polyline_2d`.

        if bool(getattr(cfg, "export_path_hypotheses_candidates", True)) if cfg else True:
            import copy

            cand_payload: Dict[str, Any] = {
                "schema": "citv_path_hypotheses_candidates_v1",
                "version": "1.0",
                "parent_schema": "citv_path_hypotheses_v3",
                "pre_cap": True,
                "pre_dedupe": True,
                "stem": ctx.stem,
                "hypothesis_count_pre_cap": len(generated_hypotheses),
                "hypothesis_count_pre_dedupe": len(generated_hypotheses),
                "accepted_candidate_cap": max_cand,
                "hypotheses": copy.deepcopy(generated_hypotheses),
            }
            if bool(getattr(cfg, "path_candidates_include_enrichment", False)) if cfg else False:
                try:
                    cand_list = list(cand_payload["hypotheses"])
                    _enrich_path_hypotheses(ctx, pipeline, cand_list, speed_map, objects, lm)
                    cand_payload["hypotheses"] = cand_list
                    cand_payload["candidates_enriched"] = True
                except Exception as _ce:
                    cand_payload["candidates_enriched"] = False
                    cand_payload["candidates_enrich_error"] = str(_ce)
            try:
                cand_path = paths_root / "path_hypotheses_candidates.json"
                _write_json(cand_payload, cand_path)
                ctx.path_exports["path_hypotheses_candidates_json"] = (
                    f"scene_graph/staged/{ctx.stem}_paths/path_hypotheses_candidates.json"
                )
            except Exception as _wc:
                print(f"  [PathsExport] path_hypotheses_candidates write failed: {_wc}")

        # Top-K + Hausdorff dedupe BEFORE expensive enrichment to avoid wasted
        # per-vertex trace computation on near-duplicate routes.
        try:
            from ..pathing.path_hypotheses_paths import dedupe_paths
            _max_per_pair = int(getattr(cfg, "path_dedupe_max_per_pair", 5)) if cfg else 5
            _frechet_thresh = _float(getattr(cfg, "path_dedupe_frechet_thresh_px", 18.0), 18.0) if cfg else 18.0
            if _max_per_pair <= 0:
                print(f"  [PathsExport] dedupe disabled (path_dedupe_max_per_pair<=0); keeping {len(hypotheses)} paths")
                dropped = []
            else:
                kept, dropped = dedupe_paths(
                    hypotheses,
                    max_per_pair=_max_per_pair,
                    frechet_thresh_px=_frechet_thresh,
                    samples=16,
                )
                hypotheses = kept
                if dropped:
                    ctx.extra.setdefault("path_dropped", []).extend(dropped)
                    print(f"  [PathsExport] dedupe: kept {len(kept)} / dropped {len(dropped)}")
        except Exception as _de:
            print(f"  [PathsExport] dedupe skipped: {_de}")

        _enrich_path_hypotheses(ctx, pipeline, hypotheses, speed_map, objects, lm)
        hyp_path = paths_root / "path_hypotheses.json"
        payload = {
            "schema": "citv_path_hypotheses_v3",
            "version": "3.0",
            "stem": ctx.stem,
            "hypotheses": hypotheses,
            "additive_fields": [
                "polyline_3d",
                "depth_trace_m",
                "width_profile_px",
                "support_trace",
                "semantic_trace",
                "caption_trace",
                "visibility_profile",
                "render_layers",
                "region_boundary_trace",
                "movement_scope",
                "boundary_interaction",
                "cost_trace",
                "motion_hints",
                "action_hints",
                "path_shape_contract",
                "animation_render_contract",
                "contract_status",
                "acceptance_status",
                "rejection_reasons",
                "routing_meta",
                "goal_generation",
                "image_dimensions",
                "natural_direction_2d_deg",
                "polyline_2d_raw",
                "polyline_2d_validated",
                "display_polyline_2d",
                "display_polyline_3d",
                "display_depth_trace_m",
                "geometry_smoothing_provenance",
                "path_geometry_quality",
                "grounding_evidence",
                "ground_object_classification",
                "uncertainty_reasons",
                "contradiction_reasons",
            ],
            "additive_fields_dropped": ["paths"],
            "compat_note": (
                "v3 removes the duplicate `paths` key. Downstream code should read "
                "`hypotheses`; legacy centerlines remain in `polyline_2d`, while "
                "`display_polyline_2d` is the validated QA/animation geometry."
            ),
        }
        _write_json(payload, hyp_path)
        ctx.extra["path_hypotheses"] = payload
        ctx.path_exports["path_hypotheses_json"] = (
            f"scene_graph/staged/{ctx.stem}_paths/path_hypotheses.json"
        )
        # path_hypotheses_full.json is intentionally not written in v3.
        ctx.path_exports.pop("path_hypotheses_full_json", None)
        _write_path_overlay_image(ctx, hypotheses, paths_root)
        try:
            from ..pathing.path_canvas import write_path_context_top5_png
            write_path_context_top5_png(
                paths_root_dir=paths_root,
                img_bgr=ctx.img_bgr,
                lm=lm,
                objs=objects,
                paths=hypotheses,
                cfg=cfg,
                metric_depth_m=ctx.metric_depth,
            )
            ctx.path_exports["path_context_top5_image"] = (
                f"scene_graph/staged/{ctx.stem}_paths/path_context_top5.png"
            )
        except Exception as exc:
            print(f"  [PathsExport] path_context_top5 failed: {exc}")

        # Plan §2.9: motion contract overlay (additive). The trajectory bundle
        # is added later by animation_export; we render with an empty bundle
        # here and animation_export overwrites with the trajectory arrows.
        try:
            from ..visualization.motion_contract_overlay import write_motion_contract_overlay
            ranked_for_overlay = sorted(
                hypotheses,
                key=lambda r: float((r.get("scores") or {}).get("overall_confidence", 0.0)),
                reverse=True,
            )
            write_motion_contract_overlay(
                ctx.img_bgr,
                ranked_for_overlay,
                {"hypotheses": []},
                paths_root / "motion_contracts_overlay.png",
                cfg=cfg,
                support_mask=ctx.extra.get("support_mask"),
                object_affordances=ctx.object_affordances,
                metric_depth_m=ctx.metric_depth,
            )
            ctx.path_exports["motion_contracts_overlay_image"] = (
                f"scene_graph/staged/{ctx.stem}_paths/motion_contracts_overlay.png"
            )
        except Exception as exc:
            print(f"  [PathsExport] motion_contracts_overlay failed: {exc}")

        _write_path_qa_overlays(ctx, hypotheses, paths_root)
        print(f"  [PathsExport] {len(hypotheses)} FMM path hypotheses written")


def _write_path_overlay_image(
    ctx: PipelineContext,
    hypotheses: List[Dict[str, Any]],
    paths_root: Path,
) -> None:
    """Draw top ≤15 FMM paths labelled on the scene image and save as PNG."""
    try:
        import cv2
    except ImportError:
        return
    if ctx.img_bgr is None or not hypotheses:
        return

    ranked = sorted(
        hypotheses,
        key=lambda p: float((p.get("scores") or {}).get("overall_confidence", 0.0)),
        reverse=True,
    )[:_MAX_LABELLED_PATHS]

    canvas = ctx.img_bgr.copy()
    h, w = canvas.shape[:2]
    dm = np.asarray(ctx.metric_depth, dtype=np.float32) if ctx.metric_depth is not None else None

    try:
        from ..pathing.path_colors import bgr_list_with_min_hue_separation, bgr_from_stable_id
        pid_list = [str(p.get("path_id", f"p{i}")) for i, p in enumerate(ranked)]
        colors = bgr_list_with_min_hue_separation(pid_list, bgr_from_stable_id)
    except Exception:
        colors = []
        for i in range(len(ranked)):
            hue = int(180 * i / max(1, len(ranked)))
            hsv = np.uint8([[[hue, 220, 210]]])
            bgr_arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            colors.append((int(bgr_arr[0]), int(bgr_arr[1]), int(bgr_arr[2])))

    legend_entries: List[Tuple[int, Any, float, str, str]] = []
    for rank_i, (path, color) in enumerate(zip(ranked, colors), start=1):
        # Plan §2.7: prefer the perspective-correct reprojection.
        pts_raw = path.get("polyline_2d_reprojected") or path.get("polyline_2d") or []
        pts = [
            (max(0, min(w - 1, int(round(_float(xy[0]))))),
             max(0, min(h - 1, int(round(_float(xy[1]))))))
            for xy in pts_raw
            if isinstance(xy, (list, tuple)) and len(xy) >= 2
        ]
        if len(pts) < 2:
            continue

        conf = float((path.get("scores") or {}).get("overall_confidence", 0.0))
        src_id = str((path.get("source_entity") or {}).get("id", "?"))[:8]
        tgt_id = str((path.get("target_entity") or {}).get("id", "?"))[:8]

        try:
            from ..pathing.path_canvas import tapered_polyline_draw, draw_direction_heads
            depth_vals: Optional[List[float]] = None
            if dm is not None:
                depth_vals = [
                    float(dm[y, x])
                    if 0 <= y < dm.shape[0] and 0 <= x < dm.shape[1]
                    else float("nan")
                    for x, y in pts
                ]
            tapered_polyline_draw(
                canvas, pts, color,
                start_w=5, end_w=2,
                alpha_start=0.88, alpha_end=0.58,
                depth_values=depth_vals,
                width_profile_px=path.get("width_profile_px"),
                visibility_profile=path.get("visibility_profile"),
                metric_depth_m=dm,
                occlusion_compositing=True,
            )
        except Exception:
            ov = canvas.copy()
            cv2.polylines(ov, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.80, canvas, 0.20, 0.0, dst=canvas)

        # Direction arrow on the last "in_front" segment when visibility info exists.
        try:
            draw_direction_heads(
                canvas, pts, color,
                thickness=2, tip_len=0.30,
                visibility_profile=path.get("visibility_profile"),
            )
        except Exception:
            if len(pts) >= 2:
                cv2.arrowedLine(canvas, pts[-2], pts[-1], color, 2, cv2.LINE_AA, tipLength=0.30)

        mid_idx = max(0, len(pts) // 2 - 1)
        mx, my = pts[mid_idx]
        ptype = str(path.get("path_type", "") or "fmm")[:10]
        label = f"P{rank_i}:{ptype}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(canvas, (mx - 2, my - th - 2), (mx + tw + 4, my + 3), (0, 0, 0), -1)
        cv2.putText(canvas, label, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

        legend_entries.append((rank_i, color, conf, src_id, tgt_id))

    # Semi-transparent legend panel.
    leg_x, leg_y, leg_w, leg_line = 8, 8, 295, 20
    leg_h = 22 + len(legend_entries) * leg_line
    ov = canvas.copy()
    cv2.rectangle(ov, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.68, canvas, 0.32, 0.0, dst=canvas)
    cv2.putText(
        canvas, f"Top {len(legend_entries)} paths (max {_MAX_LABELLED_PATHS}, by confidence)",
        (leg_x + 4, leg_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
    )
    for rank_i, color, conf, src_id, tgt_id in legend_entries:
        ey = leg_y + 22 + (rank_i - 1) * leg_line
        cv2.line(canvas, (leg_x + 4, ey + 6), (leg_x + 22, ey + 6), color, 3, cv2.LINE_AA)
        cv2.putText(
            canvas, f"P{rank_i} ({conf:.2f}) {src_id}→{tgt_id}"[:50],
            (leg_x + 26, ey + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA,
        )

    out_path = paths_root / "path_overlay.png"
    try:
        cv2.imwrite(str(out_path), canvas)
        ctx.path_exports["path_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/path_overlay.png"
        )
    except Exception as exc:
        print(f"  [PathsExport] path_overlay write failed: {exc}")


def _write_path_qa_overlays(
    ctx: PipelineContext,
    hypotheses: List[Dict[str, Any]],
    paths_root: Path,
) -> None:
    """Write 6 diagnostic QA overlay PNGs for path_updates.md Phase 6.

    All overlays are non-fatal: a failure writes nothing and continues.
    Overlays written:
      support_trace_overlay.png      — paths coloured by dominant support kind
      semantic_trace_overlay.png     — paths coloured by semantic confidence
      width_profile_overlay.png      — paths drawn as ribbons (left/right bounds)
      visibility_profile_overlay.png — paths coloured by mean visible fraction
      occlusion_state_overlay.png    — paths coloured by render-layer state
      acceptance_reasons_overlay.png — accepted/low_confidence/rejected with labels
    """
    try:
        import cv2
    except ImportError:
        return
    if ctx.img_bgr is None or not hypotheses:
        return

    h, w = ctx.img_bgr.shape[:2]

    def _pts(path: Dict[str, Any]) -> List[Tuple[int, int]]:
        poly = (
            path.get("display_polyline_2d")
            or path.get("polyline_2d_validated")
            or path.get("polyline_2d")
            or []
        )
        out: List[Tuple[int, int]] = []
        for xy in poly:
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                x = int(round(max(0.0, min(float(w - 1), _float(xy[0])))))
                y = int(round(max(0.0, min(float(h - 1), _float(xy[1])))))
                out.append((x, y))
        return out

    def _draw_polyline(canvas: np.ndarray, pts: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int = 2) -> None:
        if len(pts) >= 2:
            arr = np.asarray(pts, dtype=np.int32)
            cv2.polylines(canvas, [arr], False, color, thickness, cv2.LINE_AA)

    def _label(canvas: np.ndarray, pts: List[Tuple[int, int]], text: str, color: Tuple[int, int, int]) -> None:
        if pts:
            cv2.putText(canvas, text[:40], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, text[:40], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

    # ── 1. Support trace overlay ─────────────────────────────────────────────
    _SUPPORT_COLORS: Dict[str, Tuple[int, int, int]] = {
        "floor": (60, 200, 60),
        "support": (80, 210, 80),
        "walkable": (100, 220, 100),
        "blocking": (40, 40, 230),
        "hard_obstacle": (20, 20, 210),
        "unknown": (160, 160, 160),
        "aerial": (220, 180, 40),
        "liquid": (200, 120, 20),
    }
    try:
        canvas = ctx.img_bgr.copy()
        for path in hypotheses:
            pts = _pts(path)
            sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
            counts = dict(sem.get("support_kind_counts") or {})
            dominant = max(counts, key=lambda k: counts[k]) if counts else "unknown"
            color = _SUPPORT_COLORS.get(dominant, (160, 160, 160))
            _draw_polyline(canvas, pts, color)
        # Legend
        ly = 14
        for kind, col in list(_SUPPORT_COLORS.items())[:6]:
            cv2.rectangle(canvas, (4, ly - 8), (16, ly + 2), col, -1)
            cv2.putText(canvas, kind, (19, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.27, col, 1, cv2.LINE_AA)
            ly += 13
        cv2.imwrite(str(paths_root / "support_trace_overlay.png"), canvas)
        ctx.path_exports["support_trace_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/support_trace_overlay.png"
        )
    except Exception:
        pass

    # ── 2. Semantic trace overlay ────────────────────────────────────────────
    try:
        canvas = ctx.img_bgr.copy()
        for path in hypotheses:
            pts = _pts(path)
            sem_conf = _float((path.get("scores") or {}).get("semantic_confidence"), 0.5)
            # Hue: 0=red (low sem), 60=yellow, 120=green (high sem)
            hue = int(sem_conf * 120)
            bgr_arr = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
            color = (int(bgr_arr[0]), int(bgr_arr[1]), int(bgr_arr[2]))
            _draw_polyline(canvas, pts, color)
            if pts:
                _label(canvas, pts, f"{sem_conf:.2f}", color)
        cv2.imwrite(str(paths_root / "semantic_trace_overlay.png"), canvas)
        ctx.path_exports["semantic_trace_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/semantic_trace_overlay.png"
        )
    except Exception:
        pass

    # ── 3. Width profile overlay ─────────────────────────────────────────────
    try:
        canvas = ctx.img_bgr.copy()
        for path in hypotheses:
            left_b = path.get("left_boundary_2d") or []
            right_b = path.get("right_boundary_2d") or []
            center = _pts(path)
            if center:
                _draw_polyline(canvas, center, (200, 200, 200), 1)
            left_pts = []
            for xy in left_b:
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    left_pts.append((int(round(max(0.0, min(float(w - 1), _float(xy[0]))))),
                                     int(round(max(0.0, min(float(h - 1), _float(xy[1])))))))
            right_pts = []
            for xy in right_b:
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    right_pts.append((int(round(max(0.0, min(float(w - 1), _float(xy[0]))))),
                                      int(round(max(0.0, min(float(h - 1), _float(xy[1])))))))
            if len(left_pts) >= 2:
                _draw_polyline(canvas, left_pts, (60, 120, 220), 1)
            if len(right_pts) >= 2:
                _draw_polyline(canvas, right_pts, (220, 120, 60), 1)
            # Fill ribbon polygon
            if len(left_pts) >= 2 and len(right_pts) >= 2:
                poly_pts = left_pts + list(reversed(right_pts))
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [np.asarray(poly_pts, dtype=np.int32)], (180, 180, 60))
                cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.imwrite(str(paths_root / "width_profile_overlay.png"), canvas)
        ctx.path_exports["width_profile_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/width_profile_overlay.png"
        )
    except Exception:
        pass

    # ── 4. Visibility profile overlay ────────────────────────────────────────
    try:
        canvas = ctx.img_bgr.copy()
        for path in hypotheses:
            vis_profile = list(path.get("visibility_profile") or [])
            pts = _pts(path)
            if not pts or not vis_profile:
                _draw_polyline(canvas, pts, (160, 160, 160), 1)
                continue
            # Draw segment by segment coloured by visible_fraction
            for i in range(1, min(len(pts), len(vis_profile))):
                vf = _float(vis_profile[i - 1].get("visible_fraction") if isinstance(vis_profile[i - 1], dict) else vis_profile[i - 1], 1.0)
                hue = int(vf * 120)  # 0=red, 120=green
                bgr_a = cv2.cvtColor(np.uint8([[[hue, 220, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
                seg_color = (int(bgr_a[0]), int(bgr_a[1]), int(bgr_a[2]))
                cv2.line(canvas, pts[i - 1], pts[i], seg_color, 2, cv2.LINE_AA)
        cv2.imwrite(str(paths_root / "visibility_profile_overlay.png"), canvas)
        ctx.path_exports["visibility_profile_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/visibility_profile_overlay.png"
        )
    except Exception:
        pass

    # ── 5. Occlusion state overlay ───────────────────────────────────────────
    _LAYER_COLORS: Dict[str, Tuple[int, int, int]] = {
        "in_front": (50, 200, 50),
        "partially_occluded": (40, 200, 220),
        "behind_object": (40, 40, 220),
        "inside_mask": (180, 80, 220),
        "reflected": (220, 200, 40),
        "fading_disappearing": (160, 160, 160),
    }
    try:
        canvas = ctx.img_bgr.copy()
        for path in hypotheses:
            pts = _pts(path)
            render_layers = list(path.get("render_layers") or ["in_front"])
            dominant_layer = render_layers[0] if render_layers else "in_front"
            color = _LAYER_COLORS.get(dominant_layer, (160, 160, 160))
            _draw_polyline(canvas, pts, color)
            occ = path.get("occlusion_trace") if isinstance(path.get("occlusion_trace"), dict) else {}
            occ_ids = list(occ.get("occluder_ids") or [])
            if occ_ids and pts:
                _label(canvas, pts, f"occ:{','.join(occ_ids[:2])}", color)
        ly = 14
        for layer_name, col in list(_LAYER_COLORS.items())[:5]:
            cv2.rectangle(canvas, (4, ly - 8), (16, ly + 2), col, -1)
            cv2.putText(canvas, layer_name, (19, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.27, col, 1, cv2.LINE_AA)
            ly += 13
        cv2.imwrite(str(paths_root / "occlusion_state_overlay.png"), canvas)
        ctx.path_exports["occlusion_state_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/occlusion_state_overlay.png"
        )
    except Exception:
        pass

    # ── 6. Acceptance reasons overlay ────────────────────────────────────────
    _STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "accepted": (40, 200, 40),
        "plausible_uncertain": (40, 170, 230),
        "low_confidence": (40, 200, 220),
        "rejected": (40, 40, 220),
    }
    try:
        canvas = ctx.img_bgr.copy()
        for path in hypotheses:
            pts = _pts(path)
            status = str(path.get("acceptance_status", "low_confidence"))
            color = _STATUS_COLORS.get(status, (160, 160, 160))
            thickness = 3 if status == "accepted" else 2 if status in {"plausible_uncertain", "low_confidence"} else 1
            _draw_polyline(canvas, pts, color, thickness)
            contract = path.get("path_shape_contract") if isinstance(path.get("path_shape_contract"), dict) else {}
            rejection_reasons = list(contract.get("rejection_reasons") or [])
            quality = path.get("path_geometry_quality") if isinstance(path.get("path_geometry_quality"), dict) else {}
            rejection_reasons += list(quality.get("geometry_rejection_reasons") or [])
            if rejection_reasons and pts:
                reason_text = rejection_reasons[0][:32]
                _label(canvas, pts, f"[{status[:3]}] {reason_text}", color)
        # Status legend
        ly = 14
        for st, col in _STATUS_COLORS.items():
            cv2.rectangle(canvas, (4, ly - 8), (16, ly + 2), col, -1)
            cv2.putText(canvas, st, (19, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.27, col, 1, cv2.LINE_AA)
            ly += 13
        cv2.imwrite(str(paths_root / "acceptance_reasons_overlay.png"), canvas)
        ctx.path_exports["acceptance_reasons_overlay_image"] = (
            f"scene_graph/staged/{ctx.stem}_paths/acceptance_reasons_overlay.png"
        )
    except Exception:
        pass


def _bearing_first_segment_deg(poly2: List[List[float]]) -> Optional[float]:
    if len(poly2) < 2:
        return None
    try:
        x0, y0 = float(poly2[0][0]), float(poly2[0][1])
        x1, y1 = float(poly2[1][0]), float(poly2[1][1])
        return round(float(math.degrees(math.atan2(y1 - y0, x1 - x0))), 3)
    except (TypeError, ValueError, IndexError):
        return None


def _attach_path_qa_fields(path: Dict[str, Any], ctx: PipelineContext, poly2: List[List[float]]) -> None:
    """Additive validation fields (image size, bearing, feasible variant)."""
    path["image_dimensions"] = [int(ctx.width), int(ctx.height)]
    rep = path.get("polyline_2d_reprojected") or poly2
    if isinstance(rep, list) and len(rep) >= 2:
        path["natural_direction_2d_deg"] = _bearing_first_segment_deg(
            [[_float(xy[0]), _float(xy[1])] for xy in rep if isinstance(xy, (list, tuple)) and len(xy) >= 2]
        )
    feas_v = str((ctx.extra or {}).get("path_feasible_variant", "base"))
    rm = path.get("routing_meta")
    if isinstance(rm, dict):
        rm.setdefault("feasible_variant", feas_v)
    else:
        path["routing_meta"] = {"feasible_variant": feas_v}


def _enrich_path_hypotheses(
    ctx: PipelineContext,
    pipeline: Any,
    hypotheses: List[Dict[str, Any]],
    speed_map: np.ndarray,
    objects: List[Dict[str, Any]],
    lm: np.ndarray,
) -> None:
    """Attach semantic, depth, caption, width, and occlusion traces additively."""
    cfg = getattr(pipeline, "config", None)
    ontology = load_action_ontology(cfg)
    obj_by_id = {str(o.get("id", "")): o for o in objects}
    obj_aff_by_id = {
        str(o.get("object_id", "")): o
        for o in list((ctx.object_affordances or {}).get("objects") or [])
        if isinstance(o, dict)
    }
    mask_aff_by_id = {
        str(m.get("object_id", "")): m
        for m in list((ctx.mask_affordances or {}).get("masks") or [])
        if isinstance(m, dict)
    }
    regions_by_index = _regions_by_index(ctx)
    regions_by_id = {
        str(r.get("region_id", "")): r
        for r in list((ctx.scene_affordances or {}).get("regions") or [])
        if isinstance(r, dict)
    }
    caption_lookup = dict(ctx.extra.get("caption_lookup") or {})
    occluders = _prepare_occluders(objects, ctx.width, ctx.height)
    object_masks_by_id = {
        str(occ.get("id", "")): np.asarray(occ.get("mask"), dtype=bool)
        for occ in occluders
        if str(occ.get("id", "")) and occ.get("mask") is not None
    }
    affordance_rasters = ctx.extra.get("affordance_rasters")
    affordance_rasters = affordance_rasters if isinstance(affordance_rasters, dict) else {}
    cost_map = 1.0 - np.asarray(speed_map, dtype=np.float32)

    for path in hypotheses:
        poly2 = _polyline_2d(path)
        if len(poly2) < 2:
            path["image_dimensions"] = [int(ctx.width), int(ctx.height)]
            continue
        path["manifold_type"] = path.get("manifold_type", "ribbon_path")
        path["action_family"] = path.get("action_family", "locomotion")
        support_bound_manifold = str(path.get("manifold_type", "ribbon_path")) in {"ribbon_path", "contour_path", "interior_path", "blob_path"}
        poly3d = _lift_polyline(ctx, poly2, cfg, snap_to_support=support_bound_manifold)
        if poly3d:
            path["polyline_3d"] = poly3d
            # Plan §2.7: smooth in 3D and reproject through intrinsics so
            # rendered polylines follow perspective. This is additive — the
            # original polyline_2d still drives existing FMM/cost consumers.
            try:
                from ..pathing.polyline_3d import smooth_polyline_in_3d

                smooth = smooth_polyline_in_3d(
                    poly3d,
                    dict(ctx.intrinsics) if ctx.intrinsics else None,
                    smoothing_window=int(getattr(cfg, "polyline_smoothing_window", 5)) if cfg else 5,
                )
                if smooth.get("polyline_2d_reprojected"):
                    path["polyline_2d_reprojected"] = smooth["polyline_2d_reprojected"]
                if smooth.get("polyline_3d_smoothed"):
                    path["polyline_3d_smoothed"] = smooth["polyline_3d_smoothed"]
            except Exception:
                pass
        depth_trace = _depth_trace(poly3d)
        if depth_trace:
            path["depth_trace_m"] = depth_trace
        width_profile = _width_profile(ctx, poly3d, poly2, cfg)
        path["width_profile_px"] = width_profile

        support_trace = _support_trace(
            poly2,
            poly3d,
            lm,
            regions_by_index,
            ctx.width,
            ctx.height,
            ontology,
            affordance_rasters=affordance_rasters,
            object_masks_by_id=object_masks_by_id,
        )
        path["support_trace"] = support_trace
        path["region_boundary_trace"] = _region_boundary_trace(
            poly2,
            lm,
            regions_by_index,
            ctx.width,
            ctx.height,
            ontology,
        )
        path["movement_scope"] = str(path["region_boundary_trace"].get("movement_scope", ""))
        path["boundary_interaction"] = str(path["region_boundary_trace"].get("boundary_interaction", ""))
        path["semantic_trace"] = _semantic_trace(
            path,
            support_trace,
            obj_by_id,
            obj_aff_by_id,
            mask_aff_by_id,
            regions_by_id,
        )
        path["caption_trace"] = _caption_trace(path, support_trace, caption_lookup)
        path["visibility_profile"] = _visibility_profile(
            path,
            poly2,
            poly3d,
            width_profile,
            occluders,
            obj_by_id,
            ctx.width,
            ctx.height,
            ontology,
        )
        path["render_layers"] = _render_layers(path["visibility_profile"])
        path["occlusion_trace"] = _occlusion_summary(path["visibility_profile"])
        path["alpha_profile"] = _alpha_profile_from_visibility(path["visibility_profile"])
        path["cost_trace"] = _cost_trace(poly2, cost_map, speed_map, ctx.width, ctx.height)
        path["motion_hints"] = _motion_hints(path, ontology)
        path["action_hints"] = _action_hints(path, obj_aff_by_id, mask_aff_by_id, ctx.scene_affordances or {})
        path["ground_object_classification"] = _ground_object_classification(path, ontology)
        path.setdefault("grounding_evidence", _grounding_evidence_from_path(path))

        _attach_display_geometry_and_quality(ctx, cfg, path, poly2, poly3d, speed_map)
        display_poly2 = _polyline_2d({"polyline_2d": path.get("display_polyline_2d") or poly2})
        display_poly3d = _lift_polyline(ctx, display_poly2, cfg, snap_to_support=support_bound_manifold) if display_poly2 else []
        if display_poly3d:
            path["display_polyline_3d"] = display_poly3d
            display_depth_trace = _depth_trace(display_poly3d)
            if display_depth_trace:
                path["display_depth_trace_m"] = display_depth_trace

        # ── NEW: navigation zone trace, ribbon boundaries, kinematic sigs, trajectory contract ──
        nav_zones = ctx.extra.get("navigation_zones")
        path["navigation_zone_trace"] = _nav_zone_trace(display_poly2, nav_zones, ctx.width, ctx.height)
        left_b, right_b = _ribbon_boundaries(display_poly2, path.get("width_profile_px") or [])
        path["left_boundary_2d"] = left_b
        path["right_boundary_2d"] = right_b
        path["kinematic_signatures"] = _kinematic_signatures_from_3d(display_poly3d or poly3d)
        _update_path_scores(path, ontology)
        path["path_shape_contract"] = _path_shape_contract(path, display_poly2, display_poly3d or poly3d, left_b, right_b)
        path["contract_status"] = _path_contract_status(path, ontology)
        _apply_contract_score_gate(path, ontology)
        path["animation_render_contract"] = _animation_render_contract(path)
        path["trajectory_contract"] = _trajectory_contract(path)
        _attach_path_qa_fields(path, ctx, poly2)
        path["contract_field_availability"] = {
            "polyline_3d": bool(path.get("polyline_3d")),
            "depth_trace_m": bool(path.get("depth_trace_m")),
            "support_trace": bool(path.get("support_trace")),
            "semantic_trace": bool(path.get("semantic_trace")),
            "caption_trace": bool(path.get("caption_trace")),
            "visibility_profile": bool(path.get("visibility_profile")),
            "alpha_profile": bool(path.get("alpha_profile")),
            "region_boundary_trace": bool(path.get("region_boundary_trace")),
            "trajectory_contract": bool(path.get("trajectory_contract")),
            "path_shape_contract": bool(path.get("path_shape_contract")),
            "animation_render_contract": bool(path.get("animation_render_contract")),
            "unavailable_reasons": _contract_unavailable_reasons(path),
        }


def _polyline_2d(path: Dict[str, Any]) -> List[List[float]]:
    pts: List[List[float]] = []
    for xy in path.get("polyline_2d") or []:
        if isinstance(xy, Sequence) and len(xy) >= 2:
            pts.append([_float(xy[0]), _float(xy[1])])
    return pts


def _contract_unavailable_reasons(path: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not path.get("polyline_3d"):
        reasons.append("polyline_3d_unavailable_from_depth")
    if not path.get("depth_trace_m"):
        reasons.append("depth_trace_unavailable")
    if not path.get("visibility_profile"):
        reasons.append("visibility_profile_unavailable")
    if not path.get("trajectory_contract"):
        reasons.append("trajectory_contract_unavailable")
    if not path.get("path_shape_contract"):
        reasons.append("path_shape_contract_unavailable")
    if not path.get("animation_render_contract"):
        reasons.append("animation_render_contract_unavailable")
    return reasons


def _attach_display_geometry_and_quality(
    ctx: PipelineContext,
    cfg: Any,
    path: Dict[str, Any],
    poly2: List[List[float]],
    poly3d: List[List[float]],
    speed_map: np.ndarray,
) -> None:
    """Attach raw/validated/display geometry and measurable shape quality."""
    try:
        from ..pathing.path_geometry_quality import build_display_geometry, evaluate_geometry_quality

        speed = np.asarray(speed_map, dtype=np.float32)
        min_speed = _float(getattr(cfg, "path_display_min_speed", 0.03), 0.03) if cfg else 0.03
        feasible_mask = speed > float(min_speed) if speed.ndim == 2 and speed.size else None
        support_mask = ctx.extra.get("support_mask")
        manifold = str(path.get("manifold_type", "ribbon_path"))
        support_bound = manifold in {"ribbon_path", "contour_path", "interior_path", "blob_path"}
        support_for_geometry = support_mask if support_bound else None
        existing_raw = _polyline_2d({"polyline_2d": path.get("polyline_2d_raw") or []})
        raw = existing_raw if len(existing_raw) >= 2 else poly2
        geom = build_display_geometry(
            raw,
            width=int(ctx.width),
            height=int(ctx.height),
            feasible_mask=feasible_mask,
            support_mask=support_for_geometry if support_for_geometry is not None else None,
            cfg=cfg,
        )
        path["polyline_2d_raw"] = geom["polyline_2d_raw"]
        path["polyline_2d_validated"] = geom["polyline_2d_validated"]
        path["display_polyline_2d"] = geom["display_polyline_2d"]
        path["geometry_smoothing_provenance"] = geom["geometry_smoothing_provenance"]
        quality = evaluate_geometry_quality(
            raw_polyline=path["polyline_2d_raw"],
            display_polyline=path["display_polyline_2d"],
            polyline_3d=poly3d,
            feasible_mask=feasible_mask,
            support_mask=support_for_geometry if support_for_geometry is not None else None,
            boundary_trace=path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {},
            kinematic_signatures=path.get("kinematic_signatures") if isinstance(path.get("kinematic_signatures"), list) else [],
            cfg=cfg,
        )
        path["path_geometry_quality"] = quality
    except Exception as exc:
        path["display_polyline_2d"] = [[float(p[0]), float(p[1])] for p in poly2 if len(p) >= 2]
        path["polyline_2d_validated"] = list(path["display_polyline_2d"])
        path.setdefault("polyline_2d_raw", list(path["display_polyline_2d"]))
        path["geometry_smoothing_provenance"] = {
            "source": "raw_fallback_after_error",
            "smoothability_status": "error",
            "error": str(exc),
        }
        path["path_geometry_quality"] = {
            "schema": "citv_path_geometry_quality_v1",
            "smoothability_status": "error",
            "geometry_rejection_reasons": ["geometry_quality_unavailable"],
        }


def _lift_polyline(
    ctx: PipelineContext,
    polyline_2d: List[List[float]],
    cfg: Any,
    *,
    snap_to_support: bool = True,
) -> List[List[float]]:
    if ctx.metric_depth is None:
        return []
    try:
        from ..pathing.polyline_3d import lift_polyline_2d_to_3d

        inv = _float(getattr(cfg, "path_polyline_3d_invalid_depth_value", -1.0), -1.0) if cfg else -1.0
        default_snap = max(4, min(48, int(round(float(max(1, ctx.height)) * 0.08))))
        snap_px = int(getattr(cfg, "polyline_support_snap_max_px", default_snap)) if cfg else default_snap
        return lift_polyline_2d_to_3d(
            polyline_2d,
            ctx.metric_depth,
            invalid_z=inv,
            support_mask=ctx.extra.get("support_mask") if snap_to_support else None,
            snap_search_px=snap_px,
        )
    except Exception:
        return []


def _depth_trace(poly3d: List[List[float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = max(1, len(poly3d) - 1)
    for idx, xyz in enumerate(_decimate(poly3d, max_items=64)):
        if len(xyz) < 3:
            continue
        rows.append({
            "s": round(float(idx) / max(1, min(len(poly3d), 64) - 1), 4),
            "u": round(_float(xyz[0]), 3),
            "v": round(_float(xyz[1]), 3),
            "z_m": round(_float(xyz[2], -1.0), 4),
        })
    if poly3d:
        valid = [_float(p[2]) for p in poly3d if len(p) >= 3 and _float(p[2]) > 0.0]
        if valid:
            rows.insert(0, {
                "summary": True,
                "valid_fraction": round(len(valid) / max(1, len(poly3d)), 4),
                "min_m": round(min(valid), 4),
                "max_m": round(max(valid), 4),
                "mean_m": round(float(np.mean(valid)), 4),
                "delta_m": round(valid[-1] - valid[0], 4),
            })
    return rows


def _width_profile(
    ctx: PipelineContext,
    poly3d: List[List[float]],
    poly2d: List[List[float]],
    cfg: Any,
) -> List[Dict[str, Any]]:
    fx = _float((ctx.intrinsics or {}).get("fx"), max(1.0, float(ctx.width)))
    actor_width_m = _float(getattr(cfg, "path_actor_width_m", 0.55), 0.55) if cfg else 0.55
    min_px = _float(getattr(cfg, "path_min_width_px", 3.0), 3.0) if cfg else 3.0
    max_px = _float(getattr(cfg, "path_max_width_px", max(12.0, min(ctx.width, ctx.height) * 0.20)), max(12.0, min(ctx.width, ctx.height) * 0.20)) if cfg else max(12.0, min(ctx.width, ctx.height) * 0.20)
    rows: List[Dict[str, Any]] = []
    base = poly3d if poly3d else [[p[0], p[1], -1.0] for p in poly2d]
    sample = _decimate(base, max_items=64)
    for idx, xyz in enumerate(sample):
        z = _float(xyz[2], -1.0) if len(xyz) >= 3 else -1.0
        if z > 1e-6:
            width = fx * actor_width_m / max(1e-6, z)
        else:
            width = min_px
        width = max(min_px, min(max_px, width))
        rows.append({
            "s": round(float(idx) / max(1, len(sample) - 1), 4),
            "u": round(_float(xyz[0]), 3),
            "v": round(_float(xyz[1]), 3),
            "z_m": round(z, 4),
            "width_px": round(float(width), 3),
        })
    return rows


def _support_trace(
    poly2d: List[List[float]],
    poly3d: List[List[float]],
    lm: np.ndarray,
    regions_by_index: Dict[int, Dict[str, Any]],
    width: int,
    height: int,
    ontology: Dict[str, Any],
    *,
    affordance_rasters: Optional[Dict[str, Any]] = None,
    object_masks_by_id: Optional[Dict[str, np.ndarray]] = None,
    max_items: int = 48,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    samples = _sample_polyline(poly2d, max_items=max_items)
    z_samples = _sample_z_along_polyline(poly2d, poly3d, max_items=len(samples))
    channel_keys = (
        "support_surface_score",
        "route_corridor_score",
        "contact_target_score",
        "occlusion_edge_score",
        "contour_boundary_score",
        "open_air_score",
        "portal_score",
        "interior_motion_score",
        "effect_host_score",
        "confirmed_blocker_score",
        "uncertainty_score",
    )
    rasters = affordance_rasters if isinstance(affordance_rasters, dict) else {}
    obj_masks = object_masks_by_id if isinstance(object_masks_by_id, dict) else {}
    for sample_idx, (s, u, v) in enumerate(samples):
        xi = int(round(max(0.0, min(float(width - 1), u)))) if width > 0 else int(round(u))
        yi = int(round(max(0.0, min(float(height - 1), v)))) if height > 0 else int(round(v))
        label_idx = 0
        if lm is not None and lm.ndim == 2 and 0 <= yi < lm.shape[0] and 0 <= xi < lm.shape[1]:
            label_idx = int(lm[yi, xi])
        region = regions_by_index.get(label_idx, {})
        roles = list(region.get("roles") or [])
        actions = list(region.get("actions") or [])
        support_kind = _support_kind(roles, actions, ontology)
        channel_scores = _support_channel_scores(rasters, xi, yi, support_kind=support_kind, channel_keys=channel_keys)
        rows.append({
            "s": round(s, 4),
            "u": round(u, 3),
            "v": round(v, 3),
            "z_m": round(z_samples[sample_idx] if sample_idx < len(z_samples) else -1.0, 4),
            "region_label": label_idx,
            "region_id": str(region.get("region_id", "")),
            "region_type": str(region.get("region_type", "")),
            "semantic_label": str(region.get("semantic_label", "")),
            "support_kind": support_kind,
            "top_roles": _names_scores(roles, 3),
            "top_actions": _names_scores(actions, 3),
            "nearby_object_ids": _nearby_object_ids(obj_masks, xi, yi, radius_px=4, limit=6),
            **channel_scores,
        })
    return rows


def _support_channel_scores(
    rasters: Dict[str, Any],
    xi: int,
    yi: int,
    *,
    support_kind: str,
    channel_keys: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    observed = 0.0
    for key in channel_keys:
        arr = rasters.get(key)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size:
            val = _array_sample(arr, xi, yi, 0.0)
            out[key] = round(float(max(0.0, min(1.0, val))), 4)
            observed += out[key]
        else:
            out[key] = 0.0
    if observed <= 1e-6:
        if support_kind == "support_surface":
            out["support_surface_score"] = 0.8
            out["route_corridor_score"] = 0.55
        elif support_kind == "open_air":
            out["open_air_score"] = 0.85
        elif support_kind == "portal":
            out["portal_score"] = 0.72
            out["route_corridor_score"] = 0.45
        elif support_kind == "liquid":
            out["interior_motion_score"] = 0.70
        elif support_kind == "blocking":
            out["confirmed_blocker_score"] = 0.85
        else:
            out["uncertainty_score"] = 0.65
    return out


def _nearby_object_ids(
    object_masks_by_id: Dict[str, np.ndarray],
    xi: int,
    yi: int,
    *,
    radius_px: int = 4,
    limit: int = 6,
) -> List[str]:
    if not object_masks_by_id:
        return []
    out: List[str] = []
    rr = max(1, int(radius_px))
    for oid, mask in object_masks_by_id.items():
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
            continue
        h, w = mask.shape[:2]
        x0, x1 = max(0, xi - rr), min(w, xi + rr + 1)
        y0, y1 = max(0, yi - rr), min(h, yi + rr + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        if bool(np.any(mask[y0:y1, x0:x1])):
            out.append(str(oid))
            if len(out) >= limit:
                break
    return out


def _region_boundary_mask(lm: np.ndarray) -> np.ndarray:
    """Return the same 4-neighbor region-seam mask used by yellow QA contours."""
    if lm is None:
        return np.zeros((0, 0), dtype=bool)
    arr = np.asarray(lm, dtype=np.int32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 0), dtype=bool)
    b = np.zeros(arr.shape[:2], dtype=bool)
    b[1:, :] |= arr[1:, :] != arr[:-1, :]
    b[:, 1:] |= arr[:, 1:] != arr[:, :-1]
    b &= arr > 0
    return b


def _region_depth_mean(region: Dict[str, Any]) -> Optional[float]:
    ds = region.get("depth_stats") if isinstance(region.get("depth_stats"), dict) else {}
    for key in ("mean", "mean_m", "z_mean", "z_val", "median", "median_m"):
        val = _float(ds.get(key), -1.0)
        if val > 0.0:
            return val
    return None


def _region_ref(
    label_idx: int,
    regions_by_index: Dict[int, Dict[str, Any]],
    ontology: Dict[str, Any],
) -> Dict[str, Any]:
    region = regions_by_index.get(int(label_idx), {}) if label_idx > 0 else {}
    return {
        "region_label": int(label_idx),
        "region_id": str(region.get("region_id", "")),
        "region_type": str(region.get("region_type", "")),
        "semantic_label": str(region.get("semantic_label", "")),
        "support_kind": _support_kind(
            list(region.get("roles") or []),
            list(region.get("actions") or []),
            ontology,
        ) if region else "",
        "depth_mean_m": _region_depth_mean(region),
    }


def _boundary_motion_implications(
    *,
    transition_count: int,
    boundary_sample_fraction: float,
    max_depth_delta_m: float,
) -> List[str]:
    out: List[str] = []
    if transition_count > 0:
        out.append("inter_region_transition")
        out.append("recheck_support_and_motion_label_at_region_seams")
    if boundary_sample_fraction >= 0.30:
        out.append("boundary_following_or_portal_context")
    if max_depth_delta_m >= 0.35:
        out.append("depth_step_or_occlusion_boundary")
    if not out:
        out.append("intra_region_continuity")
    return out


def _region_boundary_trace(
    poly2d: List[List[float]],
    lm: np.ndarray,
    regions_by_index: Dict[int, Dict[str, Any]],
    width: int,
    height: int,
    ontology: Dict[str, Any],
    *,
    max_items: int = 48,
    neighborhood_px: int = 2,
) -> Dict[str, Any]:
    """Sample region-contour evidence along a path without treating contours as walls.

    The yellow contours in scene QA images are the rendered form of this same
    signal: neighboring ``region_label_map`` pixels with different integer
    labels.  Here the signal is converted into path evidence: intra/inter
    movement scope, region transitions, boundary-following likelihood, and
    depth discontinuities that should influence motion/render contracts.
    """
    if lm is None or len(poly2d) < 2:
        return {
            "schema": "citv_region_boundary_trace_v1",
            "available": False,
            "reason": "missing_label_map_or_polyline",
        }
    arr = np.asarray(lm, dtype=np.int32)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "schema": "citv_region_boundary_trace_v1",
            "available": False,
            "reason": "empty_label_map",
        }
    h, w = arr.shape[:2]
    bmask = _region_boundary_mask(arr)
    samples = _sample_polyline(poly2d, max_items=max_items)
    rows: List[Dict[str, Any]] = []
    label_counts: Dict[int, int] = {}
    near_boundary_count = 0

    for s, u, v in samples:
        xi = int(round(max(0.0, min(float(min(width, w) - 1), u)))) if min(width, w) > 0 else int(round(u))
        yi = int(round(max(0.0, min(float(min(height, h) - 1), v)))) if min(height, h) > 0 else int(round(v))
        label_idx = int(arr[yi, xi]) if 0 <= yi < h and 0 <= xi < w else 0
        if label_idx > 0:
            label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
        r = max(1, int(neighborhood_px))
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        patch_labels = sorted(
            int(x)
            for x in np.unique(arr[y0:y1, x0:x1]).tolist()
            if int(x) > 0 and int(x) != label_idx
        )
        near_boundary = bool(np.any(bmask[y0:y1, x0:x1])) or bool(patch_labels)
        if near_boundary:
            near_boundary_count += 1
        rows.append({
            "s": round(float(s), 4),
            "u": round(float(u), 3),
            "v": round(float(v), 3),
            "region": _region_ref(label_idx, regions_by_index, ontology),
            "near_region_boundary": near_boundary,
            "adjacent_region_labels": patch_labels[:8],
            "adjacent_regions": [_region_ref(lbl, regions_by_index, ontology) for lbl in patch_labels[:4]],
        })

    transitions: List[Dict[str, Any]] = []
    prev_label = int(rows[0]["region"]["region_label"]) if rows else 0
    for row in rows[1:]:
        cur_label = int((row.get("region") or {}).get("region_label", 0))
        if cur_label <= 0 or prev_label <= 0 or cur_label == prev_label:
            if cur_label > 0:
                prev_label = cur_label
            continue
        a = _region_ref(prev_label, regions_by_index, ontology)
        b = _region_ref(cur_label, regions_by_index, ontology)
        za = a.get("depth_mean_m")
        zb = b.get("depth_mean_m")
        depth_delta = abs(float(za) - float(zb)) if za is not None and zb is not None else None
        transitions.append({
            "s": row.get("s", 0.0),
            "from_region": a,
            "to_region": b,
            "depth_delta_m": round(float(depth_delta), 4) if depth_delta is not None else None,
            "transition_kind": "depth_step_or_occlusion_edge"
            if depth_delta is not None and depth_delta >= 0.35
            else "support_or_semantic_region_transition",
        })
        prev_label = cur_label

    sample_count = max(1, len(rows))
    unique_labels = [label for label, _ in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    sequence = [
        {
            **_region_ref(label, regions_by_index, ontology),
            "sample_fraction": round(count / sample_count, 4),
        }
        for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    boundary_fraction = near_boundary_count / sample_count
    max_depth_delta = max(
        [_float(t.get("depth_delta_m"), 0.0) for t in transitions],
        default=0.0,
    )
    if not unique_labels:
        movement_scope = "unknown"
    elif len(unique_labels) == 1:
        movement_scope = "intra_region"
    else:
        movement_scope = "inter_region"
    if transitions and boundary_fraction >= 0.30:
        interaction = "crosses_and_tracks_region_boundaries"
    elif transitions:
        interaction = "crosses_region_boundary"
    elif boundary_fraction >= 0.30:
        interaction = "tracks_region_boundary"
    elif movement_scope == "intra_region":
        interaction = "stays_inside_region"
    else:
        interaction = "unclassified"
    return {
        "schema": "citv_region_boundary_trace_v1",
        "available": True,
        "source": "region_label_map_4_neighbor_boundaries",
        "same_signal_as_yellow_contours": True,
        "boundary_semantics": (
            "Region seams are context for support, portal, occlusion, and motion changes; "
            "they are not hard obstacles by default."
        ),
        "movement_scope": movement_scope,
        "boundary_interaction": interaction,
        "sample_count": len(rows),
        "boundary_sample_fraction": round(float(boundary_fraction), 4),
        "transition_count": len(transitions),
        "max_transition_depth_delta_m": round(float(max_depth_delta), 4),
        "regions_sequence": sequence,
        "transitions": transitions[:24],
        "motion_implications": _boundary_motion_implications(
            transition_count=len(transitions),
            boundary_sample_fraction=boundary_fraction,
            max_depth_delta_m=max_depth_delta,
        ),
        "samples": rows,
    }


def _semantic_trace(
    path: Dict[str, Any],
    support_trace: List[Dict[str, Any]],
    obj_by_id: Dict[str, Dict[str, Any]],
    obj_aff_by_id: Dict[str, Dict[str, Any]],
    mask_aff_by_id: Dict[str, Dict[str, Any]],
    regions_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sid = str((path.get("source_entity") or {}).get("id", ""))
    tid = str((path.get("target_entity") or {}).get("id", ""))
    region_ids = sorted(set(str(r.get("region_id", "")) for r in support_trace if str(r.get("region_id", ""))))
    support_counts: Dict[str, int] = {}
    support_channel_means: Dict[str, float] = {}
    nearby_obj_ids: List[str] = []
    for row in support_trace:
        kind = str(row.get("support_kind", "unknown"))
        support_counts[kind] = support_counts.get(kind, 0) + 1
        for key in (
            "support_surface_score",
            "route_corridor_score",
            "contact_target_score",
            "occlusion_edge_score",
            "contour_boundary_score",
            "open_air_score",
            "portal_score",
            "interior_motion_score",
            "effect_host_score",
            "confirmed_blocker_score",
            "uncertainty_score",
        ):
            support_channel_means[key] = support_channel_means.get(key, 0.0) + _float(row.get(key), 0.0)
        nearby_obj_ids.extend(str(x) for x in list(row.get("nearby_object_ids") or []) if str(x))
    denom = float(max(1, len(support_trace)))
    support_channel_means = {
        k: round(float(max(0.0, min(1.0, v / denom))), 4)
        for k, v in support_channel_means.items()
    }
    return {
        "source": _entity_semantics(sid, obj_by_id, obj_aff_by_id, mask_aff_by_id),
        "target": _entity_semantics(tid, obj_by_id, obj_aff_by_id, mask_aff_by_id),
        "regions_traversed": [
            {
                "region_id": rid,
                "region_type": str((regions_by_id.get(rid, {}) or {}).get("region_type", "")),
                "semantic_label": str((regions_by_id.get(rid, {}) or {}).get("semantic_label", "")),
                "roles": _names_scores((regions_by_id.get(rid, {}) or {}).get("roles") or [], 4),
                "actions": _names_scores((regions_by_id.get(rid, {}) or {}).get("actions") or [], 4),
            }
            for rid in region_ids
        ],
        "support_kind_counts": support_counts,
        "support_channel_means": support_channel_means,
        "nearby_object_ids": sorted(set(nearby_obj_ids))[:24],
    }


def _entity_semantics(
    entity_id: str,
    obj_by_id: Dict[str, Dict[str, Any]],
    obj_aff_by_id: Dict[str, Dict[str, Any]],
    mask_aff_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    obj = obj_by_id.get(entity_id, {})
    aff = obj_aff_by_id.get(entity_id, {})
    mask = mask_aff_by_id.get(entity_id, {})
    return {
        "id": entity_id,
        "label": str(obj.get("label", aff.get("label", ""))),
        "roles": _names_scores(aff.get("roles") or [], 5),
        "actions": _names_scores(aff.get("actions") or [], 5),
        "path_modes": _names_scores(mask.get("path_modes") or [], 5, name_key="mode"),
    }


def _caption_trace(
    path: Dict[str, Any],
    support_trace: List[Dict[str, Any]],
    caption_lookup: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    sid = str((path.get("source_entity") or {}).get("id", ""))
    tid = str((path.get("target_entity") or {}).get("id", ""))
    object_lookup = caption_lookup.get("object", {})
    region_lookup = caption_lookup.get("region", {})
    samples: List[Dict[str, Any]] = []
    for row in _decimate(support_trace, max_items=24):
        rid = str(row.get("region_id", ""))
        rec = region_lookup.get(rid, {})
        samples.append({
            "s": row.get("s", 0.0),
            "region_id": rid,
            "caption": str(rec.get("text", ""))[:360],
            "precision": _float(rec.get("precision"), 0.0),
            "confidence": _float(rec.get("confidence"), 0.0),
        })
    source_rec = object_lookup.get(sid, {})
    target_rec = object_lookup.get(tid, {})
    cap_scores = [
        _float(source_rec.get("confidence"), 0.0),
        _float(target_rec.get("confidence"), 0.0),
        *[_float(s.get("confidence"), 0.0) for s in samples],
    ]
    cap_scores = [x for x in cap_scores if x > 0.0]
    return {
        "source_caption": str(source_rec.get("text", ""))[:480],
        "target_caption": str(target_rec.get("text", ""))[:480],
        "region_samples": samples,
        "mean_caption_confidence": round(float(np.mean(cap_scores)), 4) if cap_scores else 0.0,
    }


def _visibility_profile(
    path: Dict[str, Any],
    poly2d: List[List[float]],
    poly3d: List[List[float]],
    width_profile: List[Dict[str, Any]],
    occluders: List[Dict[str, Any]],
    obj_by_id: Dict[str, Dict[str, Any]],
    width: int,
    height: int,
    ontology: Dict[str, Any],
    *,
    max_items: int = 48,
) -> List[Dict[str, Any]]:
    src = str((path.get("source_entity") or {}).get("id", ""))
    tgt = str((path.get("target_entity") or {}).get("id", ""))
    skip_ids = {src, tgt}
    samples = _sample_polyline(poly2d, max_items=max_items)
    z_samples = _sample_z_along_polyline(poly2d, poly3d, max_items=len(samples))
    widths = _sample_widths(width_profile, max_items=len(samples))
    rows: List[Dict[str, Any]] = []
    for idx, (s, u, v) in enumerate(samples):
        z_path = z_samples[idx] if idx < len(z_samples) else -1.0
        radius = max(
            number(ontology, "visibility", "sample_radius_min_px", 1.0),
            min(
                number(ontology, "visibility", "sample_radius_max_px", 12.0),
                (widths[idx] if idx < len(widths) else 4.0)
                * number(ontology, "visibility", "sample_radius_width_fraction", 0.35),
            ),
        )
        footprint = [(0.0, 0.0), (radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)]
        visible = 0
        occluder_ids: List[str] = []
        for dx, dy in footprint:
            px = int(round(max(0.0, min(float(width - 1), u + dx)))) if width > 0 else int(round(u + dx))
            py = int(round(max(0.0, min(float(height - 1), v + dy)))) if height > 0 else int(round(v + dy))
            occluded = False
            for occ in occluders:
                oid = str(occ.get("id", ""))
                if oid in skip_ids:
                    continue
                mask = occ.get("mask")
                if mask is None or not (0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]):
                    continue
                if not bool(mask[py, px]):
                    continue
                z_occ = _float(occ.get("z_m"), -1.0)
                if (
                    z_path <= 0.0
                    or z_occ <= 0.0
                    or z_occ <= z_path - number(ontology, "visibility", "depth_margin_m", 0.03)
                ):
                    occluded = True
                    occluder_ids.append(oid)
                    break
            if not occluded:
                visible += 1
        visible_fraction = visible / max(1, len(footprint))
        rows.append({
            "s": round(s, 4),
            "u": round(u, 3),
            "v": round(v, 3),
            "z_m": round(z_path, 4),
            "visible_fraction": round(float(visible_fraction), 4),
            "render_layer": _render_layer(visible_fraction, ontology),
            "occluder_ids": sorted(set(occluder_ids)),
        })
    return rows


def _render_layers(visibility_profile: List[Dict[str, Any]]) -> List[str]:
    order = ["in_front", "partially_occluded", "behind_object"]
    present = {str(r.get("render_layer", "")) for r in visibility_profile}
    return [x for x in order if x in present] or ["in_front"]


def _occlusion_summary(visibility_profile: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not visibility_profile:
        return {"mean_visible_fraction": 1.0, "occluder_ids": []}
    vals = [_float(r.get("visible_fraction"), 1.0) for r in visibility_profile]
    ids = sorted(set(str(oid) for r in visibility_profile for oid in (r.get("occluder_ids") or []) if str(oid)))
    return {
        "mean_visible_fraction": round(float(np.mean(vals)), 4),
        "min_visible_fraction": round(float(np.min(vals)), 4),
        "occluded_sample_fraction": round(sum(1 for v in vals if v < 0.95) / max(1, len(vals)), 4),
        "occluder_ids": ids,
    }


def _alpha_profile_from_visibility(
    visibility_profile: List[Dict[str, Any]],
) -> List[float]:
    """Derive a dense per-sample alpha curve from the visibility profile.

    Completes Phase 5 by producing the ``alpha_profile`` field as a plain
    list of floats rather than a policy-only render contract, so animation
    consumers can drive per-frame alpha directly without re-deriving it.
    """
    if not visibility_profile:
        return [1.0]
    return [
        round(max(0.0, min(1.0, _float(s.get("visible_fraction"), 1.0))), 3)
        for s in visibility_profile
    ]


def _cost_trace(
    poly2d: List[List[float]],
    cost_map: np.ndarray,
    speed_map: np.ndarray,
    width: int,
    height: int,
    *,
    max_items: int = 48,
) -> Dict[str, Any]:
    samples = _sample_polyline(poly2d, max_items=max_items)
    rows: List[Dict[str, Any]] = []
    costs: List[float] = []
    speeds: List[float] = []
    for s, u, v in samples:
        xi = int(round(max(0.0, min(float(width - 1), u)))) if width > 0 else int(round(u))
        yi = int(round(max(0.0, min(float(height - 1), v)))) if height > 0 else int(round(v))
        c = _array_sample(cost_map, xi, yi, 1.0)
        spd = _array_sample(speed_map, xi, yi, 0.0)
        costs.append(c)
        speeds.append(spd)
        rows.append({
            "s": round(s, 4),
            "u": round(u, 3),
            "v": round(v, 3),
            "cost": round(c, 4),
            "speed": round(spd, 4),
        })
    return {
        "sample_count": len(rows),
        "mean_cost": round(float(np.mean(costs)), 4) if costs else None,
        "p90_cost": round(float(np.percentile(costs, 90)), 4) if costs else None,
        "mean_speed": round(float(np.mean(speeds)), 4) if speeds else None,
        "samples": rows,
    }


def _motion_hints(path: Dict[str, Any], ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    support_counts = dict((path.get("semantic_trace") or {}).get("support_kind_counts") or {})
    total_support = max(1, sum(int(v) for v in support_counts.values()))
    support_thr = number(ontology, "motion_hint_policy", "support_fraction_threshold", 0.25)
    if support_counts.get("open_air", 0) / total_support > support_thr:
        hints.append({
            "motion": string(ontology, "motion_hint_policy", "open_air_motion", "fly_or_hover"),
            "score": number(ontology, "motion_hint_policy", "support_motion_score", 0.65),
            "reason": "open_air_support_trace",
        })
    if support_counts.get("liquid", 0) / total_support > support_thr:
        hints.append({
            "motion": string(ontology, "motion_hint_policy", "liquid_motion", "swim_or_float"),
            "score": number(ontology, "motion_hint_policy", "support_motion_score", 0.65),
            "reason": "liquid_support_trace",
        })
    depth_rows = [r for r in path.get("depth_trace_m") or [] if not r.get("summary") and _float(r.get("z_m"), -1.0) > 0.0]
    if len(depth_rows) >= 2:
        zs = [_float(r.get("z_m")) for r in depth_rows]
        dz = zs[-1] - zs[0]
        depth_thr = number(ontology, "motion_hint_policy", "depth_delta_threshold_m", 0.35)
        if dz > depth_thr:
            hints.append({
                "motion": "climb_or_approach_farther_depth",
                "score": number(ontology, "motion_hint_policy", "depth_motion_score", 0.55),
                "reason": "positive_depth_delta",
            })
        elif dz < -depth_thr:
            hints.append({
                "motion": "descend_or_approach_camera",
                "score": number(ontology, "motion_hint_policy", "depth_motion_score", 0.55),
                "reason": "negative_depth_delta",
            })
    occ = path.get("occlusion_trace") or {}
    if _float(occ.get("occluded_sample_fraction"), 0.0) > number(ontology, "motion_hint_policy", "occluded_sample_fraction_threshold", 0.15):
        hints.append({
            "motion": "occlusion_aware_traverse",
            "score": number(ontology, "motion_hint_policy", "occlusion_motion_score", 0.60),
            "reason": "visibility_profile",
        })
    cost = path.get("cost_trace") or {}
    if _float(cost.get("p90_cost"), 0.0) > number(ontology, "motion_hint_policy", "high_path_cost_threshold", 0.65):
        hints.append({
            "motion": "careful",
            "score": number(ontology, "motion_hint_policy", "careful_motion_score", 0.52),
            "reason": "high_path_cost",
        })
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    if str(boundary.get("movement_scope", "")) == "inter_region":
        hints.append({
            "motion": "region_transition_traverse",
            "score": 0.58,
            "reason": "region_boundary_trace_inter_region",
        })
    interaction = str(boundary.get("boundary_interaction", ""))
    if interaction in {"tracks_region_boundary", "crosses_and_tracks_region_boundaries"}:
        hints.append({
            "motion": "edge_follow_or_orbit",
            "score": 0.54,
            "reason": "region_boundary_trace_boundary_following",
        })
    if _float(boundary.get("max_transition_depth_delta_m"), 0.0) >= 0.35:
        hints.append({
            "motion": "step_or_occlusion_boundary_transition",
            "score": 0.56,
            "reason": "region_boundary_trace_depth_delta",
        })
    if not hints:
        hints.append({
            "motion": string(ontology, "motion_hint_policy", "default_motion", "walk"),
            "score": number(ontology, "motion_hint_policy", "default_motion_score", 0.50),
            "reason": "default_locomotion_path",
        })
    return hints


def _action_hints(
    path: Dict[str, Any],
    obj_aff_by_id: Dict[str, Dict[str, Any]],
    mask_aff_by_id: Dict[str, Dict[str, Any]],
    scene_affordances: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sid = str((path.get("source_entity") or {}).get("id", ""))
    tid = str((path.get("target_entity") or {}).get("id", ""))
    rows: List[Dict[str, Any]] = []
    for role, oid in (("source", sid), ("target", tid)):
        aff = obj_aff_by_id.get(oid, {})
        mask = mask_aff_by_id.get(oid, {})
        for a in list(aff.get("actions") or [])[:5]:
            rows.append({
                "entity_role": role,
                "object_id": oid,
                "action": str(a.get("name", "")),
                "score": _float(a.get("score"), 0.0),
                "evidence_terms": list(a.get("evidence_terms") or []),
            })
        for m in list(mask.get("path_modes") or [])[:4]:
            rows.append({
                "entity_role": role,
                "object_id": oid,
                "path_mode": str(m.get("mode", "")),
                "score": _float(m.get("score"), 0.0),
            })
    for a in list((scene_affordances.get("summary") or {}).get("dominant_actions") or [])[:5]:
        rows.append({
            "entity_role": "scene",
            "action": str(a.get("name", "")),
            "score": _float(a.get("score"), 0.0),
            "evidence_terms": list(a.get("evidence_terms") or []),
        })
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    interaction = str(boundary.get("boundary_interaction", ""))
    scope = str(boundary.get("movement_scope", ""))
    if scope == "inter_region":
        rows.append({
            "entity_role": "region_boundary",
            "action": "cross_region_transition",
            "path_mode": "inter_region",
            "score": 0.60,
            "evidence_terms": ["region_label_map", "yellow_contour_boundary", interaction],
        })
    elif scope == "intra_region":
        rows.append({
            "entity_role": "region_boundary",
            "action": "continue_within_region",
            "path_mode": "intra_region",
            "score": 0.44,
            "evidence_terms": ["region_label_map", interaction],
        })
    if interaction in {"tracks_region_boundary", "crosses_and_tracks_region_boundaries"}:
        rows.append({
            "entity_role": "region_boundary",
            "action": "edge_follow_or_portal_approach",
            "path_mode": "boundary_context",
            "score": 0.56,
            "evidence_terms": ["region_label_map", "region_boundary_trace"],
        })
    rows.sort(key=lambda r: _float(r.get("score"), 0.0), reverse=True)
    return rows[:16]


def _update_path_scores(path: Dict[str, Any], ontology: Dict[str, Any]) -> None:
    scores = dict(path.get("scores") or {})
    base = _float(scores.get("overall_confidence"), 0.5)
    cost = path.get("cost_trace") or {}
    occ = path.get("occlusion_trace") or {}
    cap = path.get("caption_trace") or {}
    mean_cost = _float(cost.get("mean_cost"), 0.5)
    mean_visible = _float(occ.get("mean_visible_fraction"), 1.0)
    caption_conf = _float(cap.get("mean_caption_confidence"), 0.0)
    support_counts = dict((path.get("semantic_trace") or {}).get("support_kind_counts") or {})
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    boundary_conf = 0.0
    if boundary.get("available"):
        boundary_conf = max(0.0, min(1.0, 0.35
            + 0.20 * min(1.0, _float(boundary.get("boundary_sample_fraction"), 0.0))
            + 0.08 * min(4.0, _float(boundary.get("transition_count"), 0.0))
        ))
    support_total = max(1.0, sum(_float(v) for v in support_counts.values()))
    blocked = _float(support_counts.get("blocking", 0.0), 0.0) / support_total
    unknown = _float(support_counts.get("unknown", 0.0), 0.0) / support_total
    support_grounding = _support_grounding_confidence(path, support_counts)
    anchor_conf = _entity_anchor_confidence(path, ontology)
    local_action_conf = _local_action_evidence_confidence(path)
    local_grounding_conf = max(0.0, min(1.0, 0.65 * local_action_conf + 0.35 * anchor_conf))
    geometric_conf = max(0.0, min(1.0,
        number(ontology, "path_score_weights", "geometric_cost_weight", 0.55) * (1.0 - mean_cost)
        + number(ontology, "path_score_weights", "geometric_visibility_weight", 0.45) * mean_visible
    ))
    manifold_fit = support_grounding
    contradiction_score = blocked
    uncertainty_score = unknown
    geometry_contract_score = _float((path.get("path_shape_contract") or {}).get("confidence"), geometric_conf)
    renderability_score = mean_visible
    support_channel_means = dict((path.get("semantic_trace") or {}).get("support_channel_means") or {})
    try:
        from ..pathing.manifold_fit_scoring import compute_manifold_fit_scores

        fit_scores = compute_manifold_fit_scores(
            manifold_type=str(path.get("manifold_type", "ribbon_path")),
            support_trace=list(path.get("support_trace") or []),
            visibility_profile=list(path.get("visibility_profile") or []),
            geometry_quality=path.get("path_geometry_quality") if isinstance(path.get("path_geometry_quality"), dict) else {},
            local_grounding_score=local_grounding_conf,
        )
        manifold_fit = _float(fit_scores.get("manifold_fit_score"), manifold_fit)
        local_grounding_conf = _float(fit_scores.get("local_grounding_score"), local_grounding_conf)
        geometry_contract_score = _float(fit_scores.get("geometry_contract_score"), geometry_contract_score)
        renderability_score = _float(fit_scores.get("renderability_score"), renderability_score)
        contradiction_score = _float(fit_scores.get("contradiction_score"), contradiction_score)
        uncertainty_score = _float(fit_scores.get("uncertainty_score"), uncertainty_score)
        support_channel_means = dict(fit_scores.get("support_channel_means") or support_channel_means)
    except Exception:
        pass
    semantic_conf = max(0.0, min(1.0,
        number(ontology, "path_score_weights", "semantic_base", 0.72)
        - number(ontology, "path_score_weights", "semantic_blocking_penalty", 0.45) * blocked
        - number(ontology, "path_score_weights", "semantic_unknown_penalty", 0.30) * unknown
        + number(ontology, "path_score_weights", "semantic_caption_bonus", 0.18) * caption_conf
        + number(ontology, "path_score_weights", "semantic_support_bonus", 0.18) * support_grounding
        + number(ontology, "path_score_weights", "semantic_anchor_bonus", 0.10) * anchor_conf
        + number(ontology, "path_score_weights", "semantic_local_action_bonus", 0.10) * local_action_conf
        + 0.18 * manifold_fit
        - 0.20 * contradiction_score
    ))
    kinematic_conf = max(0.0, min(1.0,
        number(ontology, "path_score_weights", "kinematic_base", 0.45)
        + number(ontology, "path_score_weights", "kinematic_visibility_bonus", 0.35) * mean_visible
        + number(ontology, "path_score_weights", "kinematic_cost_bonus", 0.20) * (1.0 - mean_cost)
    ))
    scores.update({
        "geometric_confidence": round(float(geometric_conf), 4),
        "semantic_confidence": round(float(semantic_conf), 4),
        "caption_confidence": round(float(caption_conf), 4),
        "boundary_context_confidence": round(float(boundary_conf), 4),
        "kinematic_confidence": round(float(kinematic_conf), 4),
        "support_grounding_confidence": round(float(support_grounding), 4),
        "entity_anchor_confidence": round(float(anchor_conf), 4),
        "local_action_evidence_confidence": round(float(local_action_conf), 4),
        "local_grounding_score": round(float(local_grounding_conf), 4),
        "manifold_fit_score": round(float(manifold_fit), 4),
        "geometry_contract_score": round(float(geometry_contract_score), 4),
        "renderability_score": round(float(renderability_score), 4),
        "contradiction_score": round(float(contradiction_score), 4),
        "uncertainty_score": round(float(uncertainty_score), 4),
        "support_channel_means": {
            str(k): round(float(max(0.0, min(1.0, _float(v, 0.0)))), 4)
            for k, v in support_channel_means.items()
            if str(k)
        },
        "unknown_support_fraction": round(float(unknown), 4),
        "blocking_support_fraction": round(float(blocked), 4),
        "overall_confidence": round(float(max(0.0, min(1.0,
            number(ontology, "path_score_weights", "overall_base_weight", 0.45) * base
            + number(ontology, "path_score_weights", "overall_geometric_weight", 0.25) * geometric_conf
            + number(ontology, "path_score_weights", "overall_semantic_weight", 0.20) * semantic_conf
            + number(ontology, "path_score_weights", "overall_caption_weight", 0.10) * caption_conf
            + 0.10 * manifold_fit
            - 0.08 * contradiction_score
        ))), 4),
    })
    path["scores"] = scores


def _support_grounding_confidence(path: Dict[str, Any], support_counts: Dict[str, Any]) -> float:
    total = max(1.0, sum(_float(v) for v in support_counts.values()))
    manifold = str(path.get("manifold_type", "ribbon_path"))
    ch = dict((path.get("semantic_trace") or {}).get("support_channel_means") or {})
    if manifold == "volume_path":
        good = 0.7 * _float(ch.get("open_air_score"), 0.0) * total + 0.3 * _float(support_counts.get("open_air"), 0.0)
    elif manifold in {"blob_path", "interior_path"}:
        good = (
            0.55 * _float(ch.get("interior_motion_score"), 0.0) * total
            + 0.25 * _float(ch.get("contact_target_score"), 0.0) * total
            + 0.20 * (_float(support_counts.get("liquid"), 0.0) + _float(support_counts.get("support_surface"), 0.0))
        )
    elif manifold == "portal_path":
        good = (
            0.60 * _float(ch.get("portal_score"), 0.0) * total
            + 0.20 * _float(ch.get("route_corridor_score"), 0.0) * total
            + 0.20 * (_float(support_counts.get("portal"), 0.0) + _float(support_counts.get("support_surface"), 0.0))
        )
    elif manifold in {"contact_patch", "occlusion_pulse", "effect_field"}:
        good = (
            0.50 * _float(ch.get("contact_target_score"), 0.0) * total
            + 0.30 * _float(ch.get("occlusion_edge_score"), 0.0) * total
            + 0.20 * (total - _float(support_counts.get("blocking"), 0.0))
        )
    else:
        good = (
            0.45 * _float(ch.get("support_surface_score"), 0.0) * total
            + 0.25 * _float(ch.get("route_corridor_score"), 0.0) * total
            + 0.10 * _float(ch.get("portal_score"), 0.0) * total
            + 0.20 * (_float(support_counts.get("support_surface"), 0.0) + _float(support_counts.get("portal"), 0.0))
        )
    unknown = _float(support_counts.get("unknown"), 0.0)
    blocking = max(_float(support_counts.get("blocking"), 0.0), _float(ch.get("confirmed_blocker_score"), 0.0) * total)
    conf = good / total
    conf -= 0.35 * (unknown / total)
    conf -= (0.55 if manifold in {"ribbon_path", "contour_path", "portal_path"} else 0.30) * (blocking / total)
    if manifold in {"contact_patch", "occlusion_pulse", "effect_field"} and path.get("target_entity"):
        conf = max(conf, 0.45)
    return max(0.0, min(1.0, conf))


def _label_quality_terms_from_ontology(ontology: Dict[str, Any]) -> Tuple[set, set, set]:
    lq = ontology.get("label_quality") if isinstance(ontology.get("label_quality"), dict) else {}
    generic = {str(x).strip().lower() for x in list(lq.get("generic_labels") or []) if str(x).strip()}
    meta_terms = {str(x).strip().lower() for x in list(lq.get("meta_visual_terms") or []) if str(x).strip()}
    meta_phrases = {str(x).strip().lower() for x in list(lq.get("meta_visual_phrases") or []) if str(x).strip()}
    return generic, meta_terms, meta_phrases


def _semantic_entity_quality(ent: Dict[str, Any], ontology: Dict[str, Any]) -> float:
    if not ent:
        return 0.35
    generic, meta_terms, meta_phrases = _label_quality_terms_from_ontology(ontology)
    label = str(ent.get("label", "")).strip().lower()
    roles = list(ent.get("roles") or [])
    actions = list(ent.get("actions") or [])
    modes = list(ent.get("path_modes") or [])
    role_score = max([_float(r.get("score"), 0.0) for r in roles if isinstance(r, dict)] or [0.0])
    action_score = max([_float(a.get("score"), 0.0) for a in actions if isinstance(a, dict)] or [0.0])
    mode_score = max([_float(m.get("score"), 0.0) for m in modes if isinstance(m, dict)] or [0.0])
    q = 0.30 + 0.35 * max(role_score, action_score, mode_score) + 0.20 * min(1.0, len(label.split()) / 3.0)
    toks = set(label.split())
    if label in generic or label in meta_phrases:
        q -= 0.30
    if toks and len(toks.intersection(meta_terms)) >= max(1, len(toks) // 2):
        q -= 0.25
    if label in {"", "unknown", "thin", "set"}:
        q -= 0.20
    return max(0.0, min(1.0, q))


def _entity_anchor_confidence(path: Dict[str, Any], ontology: Dict[str, Any]) -> float:
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    src = sem.get("source") if isinstance(sem.get("source"), dict) else {}
    tgt = sem.get("target") if isinstance(sem.get("target"), dict) else {}
    src_type = str((path.get("source_entity") or {}).get("type", ""))
    tgt_type = str((path.get("target_entity") or {}).get("type", ""))
    pseudo_types = {"open_air", "sky_region", "candidate_actor", "mask_interior", "occluder", "effect_surface"}
    src_q = 0.65 if src_type in pseudo_types else _semantic_entity_quality(src, ontology)
    tgt_q = 0.65 if tgt_type in pseudo_types else _semantic_entity_quality(tgt, ontology)
    if str(path.get("path_level", "")).lower() == "region":
        return max(src_q, tgt_q, 0.45)
    return max(0.0, min(1.0, (src_q + tgt_q) * 0.5))


def _local_action_evidence_confidence(path: Dict[str, Any]) -> float:
    rows = list(path.get("action_hints") or [])
    local_scores: List[float] = []
    scene_scores: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("entity_role", ""))
        score = _float(row.get("score"), 0.0)
        if role == "scene":
            scene_scores.append(score)
        elif role:
            local_scores.append(score)
    if local_scores:
        return max(0.0, min(1.0, max(local_scores)))
    if scene_scores:
        # Scene/global affordances are priors. They should not validate a
        # specific path unless object, mask, region, boundary, or support
        # evidence also grounds that action locally.
        return 0.0
    return 0.0


def _grounding_evidence_from_path(path: Dict[str, Any]) -> Dict[str, Any]:
    rows = [r for r in list(path.get("action_hints") or []) if isinstance(r, dict)]
    local_scores: List[float] = []
    scene_scores: List[float] = []
    levels: List[str] = []
    for row in rows:
        role = str(row.get("entity_role", "") or "")
        score = _float(row.get("score"), 0.0)
        if role == "scene":
            scene_scores.append(score)
        elif role:
            local_scores.append(score)
            levels.append(role)
    local_conf = max(local_scores) if local_scores else 0.0
    scene_conf = max(scene_scores) if scene_scores else 0.0
    return {
        "schema": "citv_path_grounding_evidence_v1",
        "candidate_source": str((path.get("routing_meta") or {}).get("generation_order", path.get("path_type", ""))),
        "local_evidence_confidence": round(float(local_conf), 4),
        "scene_prior_confidence": round(float(scene_conf), 4),
        "global_only": bool(scene_conf > 0.0 and local_conf <= 0.0),
        "local_evidence_levels": sorted(set(levels))[:8],
        "manifold_type": str(path.get("manifold_type", "")),
    }


def _motion_tokens(value: Any) -> List[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    text = text.replace("_or_", " ").replace("/", " ").replace("|", " ").replace(",", " ")
    out: List[str] = []
    for tok in text.split():
        t = tok.strip()
        if t and t not in out:
            out.append(t)
    return out


def _ground_object_classification(path: Dict[str, Any], ontology: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    onto = ontology if isinstance(ontology, dict) and ontology else load_action_ontology(None)
    sem = path.get("semantic_trace") if isinstance(path.get("semantic_trace"), dict) else {}
    counts = dict(sem.get("support_kind_counts") or {})
    channels = dict(sem.get("support_channel_means") or {})
    total = float(max(1.0, sum(_float(v, 0.0) for v in counts.values())))
    walkable = (
        _float(counts.get("support_surface"), 0.0)
        + _float(counts.get("floor"), 0.0)
        + _float(counts.get("walkable"), 0.0)
        + _float(channels.get("route_corridor_score"), 0.0) * total * 0.40
        + _float(channels.get("support_surface_score"), 0.0) * total * 0.30
    ) / total
    swimmable = (
        _float(counts.get("liquid"), 0.0)
        + _float(channels.get("interior_motion_score"), 0.0) * total * 0.40
    ) / total
    open_air = (
        _float(counts.get("open_air"), 0.0)
        + _float(channels.get("open_air_score"), 0.0) * total * 0.50
    ) / total
    blocked = (
        _float(counts.get("blocking"), 0.0)
        + _float(counts.get("hard_obstacle"), 0.0)
        + _float(channels.get("confirmed_blocker_score"), 0.0) * total
    ) / total
    dominant = max(counts, key=counts.get) if counts else "unknown"
    action_priors: Dict[str, float] = {}
    for rule in list_section(onto, "support_kind_policy"):
        kind = str(rule.get("kind", "")).strip()
        if not kind:
            continue
        weight = max(0.0, min(1.0, _float(counts.get(kind), 0.0) / total))
        if weight <= 0.0:
            continue
        for action in list(rule.get("actions") or []):
            for tok in _motion_tokens(action):
                action_priors[tok] = max(action_priors.get(tok, 0.0), weight)
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict):
            motion_txt = hint.get("motion", "")
            score = max(0.0, min(1.0, _float(hint.get("score"), 0.0)))
        else:
            motion_txt = hint
            score = 0.4
        for tok in _motion_tokens(motion_txt):
            action_priors[tok] = max(action_priors.get(tok, 0.0), score)
    default_motion = string(onto, "motion_hint_policy", "default_motion", "walk")
    default_tokens = _motion_tokens(default_motion)
    if action_priors:
        rec = max(action_priors.items(), key=lambda kv: kv[1])[0]
    else:
        rec = default_tokens[0] if default_tokens else "walk"
    labels: List[str] = []
    if walkable >= 0.22:
        labels.append("walkable")
    if swimmable >= 0.22:
        labels.append("swimmable")
    if open_air >= 0.22:
        labels.append("open_air")
    if blocked >= 0.30:
        labels.append("obstacle_heavy")
    if not labels:
        labels.append("uncertain")
    return {
        "schema": "citv_ground_object_classification_v1",
        "dominant_support_kind": str(dominant),
        "walkable_fraction": round(float(max(0.0, min(1.0, walkable))), 4),
        "swimmable_fraction": round(float(max(0.0, min(1.0, swimmable))), 4),
        "open_air_fraction": round(float(max(0.0, min(1.0, open_air))), 4),
        "blocking_fraction": round(float(max(0.0, min(1.0, blocked))), 4),
        "terrain_labels": labels,
        "recommended_motion": str(rec),
        "support_action_priors": {str(k): round(float(v), 4) for k, v in sorted(action_priors.items(), key=lambda kv: (-kv[1], kv[0]))[:12]},
    }


def _regions_by_index(ctx: PipelineContext) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    scene_regions = {
        str(r.get("region_id", "")): r
        for r in list((ctx.scene_affordances or {}).get("regions") or [])
        if isinstance(r, dict)
    }
    for region in ctx.region_partition_meta or []:
        idx = int(region.get("region_index", 0) or 0)
        rid = str(region.get("id", "") or "")
        row = {
            "region_id": rid,
            "region_index": idx,
            "region_type": str(region.get("type", "")),
            "semantic_label": str(region.get("semantic_label", "")),
            "depth_stats": dict(region.get("depth_stats") or {}),
        }
        row.update(scene_regions.get(rid, {}))
        if idx > 0:
            out[idx] = row
    return out


def _support_kind(
    roles: Sequence[Dict[str, Any]],
    actions: Sequence[Dict[str, Any]],
    ontology: Dict[str, Any],
) -> str:
    role_scores = {str(r.get("name", "")): _float(r.get("score"), 0.0) for r in roles if isinstance(r, dict)}
    action_scores = {str(a.get("name", "")): _float(a.get("score"), 0.0) for a in actions if isinstance(a, dict)}
    for rule in list_section(ontology, "support_kind_policy"):
        threshold = _float(rule.get("threshold"), 0.2)
        vals = [
            *(role_scores.get(str(role), 0.0) for role in rule.get("roles", []) or []),
            *(action_scores.get(str(action), 0.0) for action in rule.get("actions", []) or []),
        ]
        if vals and max(vals) > threshold:
            return str(rule.get("kind", "unknown"))
    return "unknown"


def _prepare_occluders(objects: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in objects:
        m = obj.get("_sam2_mask_array")
        if m is None:
            continue
        try:
            mask = np.asarray(m, dtype=bool)
            if height > 0 and width > 0 and mask.shape[:2] != (height, width):
                import cv2

                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            out.append({
                "id": str(obj.get("id", "")),
                "label": str(obj.get("label", "")),
                "z_m": _float((obj.get("depth_stats") or {}).get("z_val"), _float((obj.get("coordinates_3d") or {}).get("z"), -1.0)),
                "mask": mask,
            })
        except Exception:
            continue
    return out


def _sample_polyline(poly2d: List[List[float]], *, max_items: int = 48) -> List[Tuple[float, float, float]]:
    if not poly2d:
        return []
    if len(poly2d) == 1:
        return [(0.0, _float(poly2d[0][0]), _float(poly2d[0][1]))]
    xs = np.asarray([p[0] for p in poly2d], dtype=np.float64)
    ys = np.asarray([p[1] for p in poly2d], dtype=np.float64)
    d = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    n = max(2, int(max_items))
    if total <= 1e-9:
        return [(0.0, float(xs[0]), float(ys[0])) for _ in range(n)]
    rows: List[Tuple[float, float, float]] = []
    for i in range(n):
        s = i / max(1, n - 1)
        target = s * total
        idx = int(np.searchsorted(cum, target, side="right") - 1)
        idx = max(0, min(len(poly2d) - 2, idx))
        seg = float(cum[idx + 1] - cum[idx]) or 1e-6
        t = (target - float(cum[idx])) / seg
        x = float(xs[idx] + t * (xs[idx + 1] - xs[idx]))
        y = float(ys[idx] + t * (ys[idx + 1] - ys[idx]))
        rows.append((float(s), x, y))
    return rows


def _sample_z_along_polyline(poly2d: List[List[float]], poly3d: List[List[float]], *, max_items: int) -> List[float]:
    if not poly3d:
        return [-1.0 for _ in range(max_items)]
    z_by_sample = []
    samples = _sample_polyline([[p[0], p[1]] for p in poly3d if len(p) >= 2], max_items=max_items)
    for idx, _ in enumerate(samples):
        src_idx = int(round((idx / max(1, len(samples) - 1)) * (len(poly3d) - 1)))
        z_by_sample.append(_float(poly3d[src_idx][2], -1.0) if len(poly3d[src_idx]) >= 3 else -1.0)
    return z_by_sample


def _sample_widths(width_profile: List[Dict[str, Any]], *, max_items: int) -> List[float]:
    if not width_profile:
        return [4.0 for _ in range(max_items)]
    rows = _decimate(width_profile, max_items=max_items)
    vals = [_float(r.get("width_px"), 4.0) for r in rows if isinstance(r, dict)]
    if len(vals) < max_items:
        vals.extend([vals[-1] if vals else 4.0] * (max_items - len(vals)))
    return vals[:max_items]


def _decimate(rows: Sequence[Any], *, max_items: int) -> List[Any]:
    if len(rows) <= max_items:
        return list(rows)
    if max_items <= 1:
        return [rows[0]]
    idxs = np.linspace(0, len(rows) - 1, max_items)
    return [rows[int(round(i))] for i in idxs]


def _array_sample(arr: np.ndarray, x: int, y: int, default: float) -> float:
    try:
        if arr is None or arr.ndim != 2 or not (0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]):
            return default
        val = float(arr[y, x])
        return val if math.isfinite(val) else default
    except Exception:
        return default


def _render_layer(visible_fraction: float, ontology: Dict[str, Any]) -> str:
    if visible_fraction >= number(ontology, "visibility", "in_front_threshold", 0.95):
        return "in_front"
    if visible_fraction > number(ontology, "visibility", "partially_visible_threshold", 0.05):
        return "partially_occluded"
    return "behind_object"


def _names_scores(rows: Sequence[Dict[str, Any]], limit: int, *, name_key: str = "name") -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get(name_key, ""))
        if not name:
            continue
        out.append({"name": name, "score": round(_float(row.get("score"), 0.0), 4)})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:limit]


def _float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Bézier smoothing helper
# ─────────────────────────────────────────────────────────────────────────────

def _snap_polyline_to_feasible(
    poly: List[List[float]],
    feasible: np.ndarray,
    w: int,
    h: int,
    cfg: Any,
    snap_uv_to_walkable: Any,
) -> List[List[float]]:
    """Project off-feasible Bézier vertices back onto the walkable mask."""
    if not poly or not bool(getattr(cfg, "path_bezier_snap_feasible_enabled", True)) if cfg else True:
        return poly
    max_px = _float(getattr(cfg, "path_bezier_snap_feasible_max_px", 8.0), 8.0) if cfg else 8.0
    out: List[List[float]] = []
    fe = np.asarray(feasible, dtype=bool)
    for p in poly:
        if len(p) < 2:
            continue
        try:
            xi, yi = int(round(float(p[0]))), int(round(float(p[1])))
        except (TypeError, ValueError):
            out.append(p)
            continue
        if 0 <= yi < h and 0 <= xi < w and fe[yi, xi]:
            out.append([float(xi), float(yi)])
            continue
        su, sv = snap_uv_to_walkable(xi, yi, fe, w, h)
        dist = math.hypot(float(su - xi), float(sv - yi))
        if dist <= max_px:
            out.append([float(su), float(sv)])
        else:
            out.append([float(xi), float(yi)])
    return out if len(out) >= 2 else poly


def _bbox_center_uv(obj: Dict[str, Any]) -> Optional[List[float]]:
    """Return [cx, cy] from obj's bbox fields, or None."""
    bbox = obj.get("bbox") or obj.get("bbox_2d")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
    geom = obj.get("geometry") or {}
    bp = geom.get("bbox_px")
    if isinstance(bp, (list, tuple)) and len(bp) >= 4:
        return [(bp[0] + bp[2]) / 2.0, (bp[1] + bp[3]) / 2.0]
    return None


def _bezier_smooth_polyline(
    raw_poly: List[List[float]],
    w: int,
    h: int,
    walkable: Optional[np.ndarray] = None,
) -> List[List[float]]:
    """Return Bézier-rounded version of *raw_poly*; fall back to raw on error."""
    if len(raw_poly) < 3:
        return raw_poly
    try:
        from ..pathing.polyline_bezier import bezier_round_polyline
        pts_in = [(int(round(p[0])), int(round(p[1]))) for p in raw_poly]
        smoothed = bezier_round_polyline(pts_in, w=w, h=h, walkable=walkable)
        return [[float(p[0]), float(p[1])] for p in smoothed]
    except Exception:
        return raw_poly


def _curved_transition_polyline(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    w: int,
    h: int,
    *,
    samples: int = 17,
) -> List[List[float]]:
    """Create a gentle quadratic curve for portal/fallback transitions."""
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(goal[0]), float(goal[1])
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return [[x0, y0], [x1, y1]]
    nx, ny = -dy / length, dx / length
    cx_img, cy_img = float(w) * 0.5, float(h) * 0.5
    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    # Bend away from image centre so the transition is visible but stable.
    sign = 1.0 if ((mx - cx_img) * nx + (my - cy_img) * ny) >= 0.0 else -1.0
    bend = min(max(length * 0.18, 18.0), max(24.0, min(float(w), float(h)) * 0.18))
    cx, cy = mx + sign * nx * bend, my + sign * ny * bend
    n = max(3, int(samples))
    out: List[List[float]] = []
    for i in range(n):
        t = i / max(1, n - 1)
        xa = (1.0 - t) * x0 + t * cx
        ya = (1.0 - t) * y0 + t * cy
        xb = (1.0 - t) * cx + t * x1
        yb = (1.0 - t) * cy + t * y1
        x = (1.0 - t) * xa + t * xb
        y = (1.0 - t) * ya + t * yb
        out.append([
            round(max(0.0, min(float(max(0, w - 1)), x)), 3),
            round(max(0.0, min(float(max(0, h - 1)), y)), 3),
        ])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Navigation zone trace per path
# ─────────────────────────────────────────────────────────────────────────────

def _nav_zone_trace(
    poly2d: List[List[float]],
    nav_zones: Optional[np.ndarray],
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Sample (H,W,4) nav-zone map along *poly2d* and return mean channel scores."""
    if nav_zones is None or not poly2d:
        return {}
    try:
        from ..pathing.navigation_zones import trace_navigation_along_polyline
        return trace_navigation_along_polyline(poly2d, nav_zones, width, height)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Ribbon boundary (left + right edges of the path corridor)
# ─────────────────────────────────────────────────────────────────────────────

def _ribbon_boundaries(
    poly2d: List[List[float]],
    width_profile: List[Dict[str, Any]],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute left and right boundary polylines from centerline + width profile.

    At each sample the half-width is projected perpendicular to the forward
    tangent.  Boundaries share the same length as the (decimated) poly2d.
    """
    if len(poly2d) < 2:
        return [], []
    widths = _sample_widths(width_profile, max_items=len(poly2d))
    left: List[List[float]] = []
    right: List[List[float]] = []
    n = len(poly2d)
    for i in range(n):
        x0, y0 = _float(poly2d[i][0]), _float(poly2d[i][1])
        # tangent: forward difference, backward at tail
        if i < n - 1:
            dx = _float(poly2d[i + 1][0]) - x0
            dy = _float(poly2d[i + 1][1]) - y0
        else:
            dx = x0 - _float(poly2d[i - 1][0])
            dy = y0 - _float(poly2d[i - 1][1])
        mag = math.hypot(dx, dy) + 1e-9
        # perpendicular (rotate 90°): (-dy, dx)
        nx, ny = -dy / mag, dx / mag
        hw = (widths[i] if i < len(widths) else 4.0) * 0.5
        left.append([round(x0 + nx * hw, 3), round(y0 + ny * hw, 3)])
        right.append([round(x0 - nx * hw, 3), round(y0 - ny * hw, 3)])
    return left, right


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Kinematic signatures from 3D polyline
# ─────────────────────────────────────────────────────────────────────────────

def _kinematic_signatures_from_3d(
    poly3d: List[List[float]],
    *,
    jump_z: float = 0.25,
    climb_z: float = 0.12,
    idle_frac: float = 0.08,
) -> List[Dict[str, Any]]:
    """Segment poly3d into labelled motion runs (walk/climb/jump/descend/crawl)."""
    if len(poly3d) < 2:
        return []
    zs = [_float(pt[2], 0.0) if len(pt) > 2 else 0.0 for pt in poly3d]
    # 3-sample median filter to suppress monocular depth spikes
    if len(zs) >= 3:
        med = [zs[0]]
        for i in range(1, len(zs) - 1):
            med.append(sorted([zs[i - 1], zs[i], zs[i + 1]])[1])
        med.append(zs[-1])
        zs = med
    n = len(zs)
    sigs: List[Dict[str, Any]] = []
    i = 0
    while i < n - 1:
        dz = zs[i + 1] - zs[i]
        if abs(dz) < idle_frac:
            motion = "walk"
        elif dz > jump_z:
            motion = "jump"
        elif dz > climb_z:
            motion = "climb"
        elif dz < -jump_z:
            motion = "descend"
        elif dz < -climb_z:
            motion = "descend"
        else:
            motion = "walk"
        j = i + 1
        while j < n - 1:
            dz_n = zs[j + 1] - zs[j]
            if abs(dz_n) < idle_frac:
                nm = "walk"
            elif dz_n > jump_z:
                nm = "jump"
            elif dz_n > climb_z:
                nm = "climb"
            elif dz_n < -jump_z:
                nm = "descend"
            elif dz_n < -climb_z:
                nm = "descend"
            else:
                nm = "walk"
            if nm != motion:
                break
            j += 1
        if motion == "walk" and (j - i) >= 6:
            z_var = max(zs[i:j + 1]) - min(zs[i:j + 1])
            if z_var < idle_frac / 4.0:
                motion = "crawl"
        sigs.append({
            "start_idx": i,
            "end_idx": j,
            "motion": motion,
            "dz_m": round(float(zs[j] - zs[i]), 4),
        })
        i = j
    return sigs


def _polyline_length_px(poly2d: List[List[float]]) -> float:
    if len(poly2d) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(_float(poly2d[i][0]) - _float(poly2d[i - 1][0]), _float(poly2d[i][1]) - _float(poly2d[i - 1][1]))
            for i in range(1, len(poly2d))
        )
    )


def _straightness_ratio(poly2d: List[List[float]]) -> float:
    if len(poly2d) < 2:
        return 1.0
    length = _polyline_length_px(poly2d)
    chord = math.hypot(_float(poly2d[-1][0]) - _float(poly2d[0][0]), _float(poly2d[-1][1]) - _float(poly2d[0][1]))
    if length <= 1e-6:
        return 1.0
    return max(0.0, min(1.0, chord / length))


def _manifold_shape_type(manifold: str) -> str:
    mapping = {
        "ribbon_path": "support_aware_ribbon",
        "contour_path": "mask_or_region_boundary_contour",
        "interior_path": "mask_interior_area",
        "blob_path": "mask_or_liquid_blob_area",
        "volume_path": "open_space_volume",
        "portal_path": "entry_exit_curve_with_fade",
        "occlusion_pulse": "occlusion_edge_pulse",
        "contact_patch": "local_contact_patch",
        "effect_field": "local_effect_field",
    }
    return mapping.get(str(manifold), "polyline_path")


def _path_direction_profile(poly2d: List[List[float]], poly3d: List[List[float]]) -> Dict[str, Any]:
    if len(poly2d) < 2:
        return {"available": False, "reason": "polyline_too_short"}
    x0, y0 = _float(poly2d[0][0]), _float(poly2d[0][1])
    x1, y1 = _float(poly2d[-1][0]), _float(poly2d[-1][1])
    dx, dy = x1 - x0, y1 - y0
    heading_start = _bearing_first_segment_deg(poly2d)
    heading_end = _bearing_first_segment_deg(poly2d[-2:]) if len(poly2d) >= 2 else heading_start
    horiz = "right" if dx > 8 else "left" if dx < -8 else "center"
    vert = "down" if dy > 8 else "up" if dy < -8 else "level"
    if horiz == "center":
        image_direction = vert
    elif vert == "level":
        image_direction = horiz
    else:
        image_direction = f"{vert}_{horiz}"
    valid_z = [_float(p[2], -1.0) for p in poly3d if isinstance(p, (list, tuple)) and len(p) >= 3 and _float(p[2], -1.0) > 0.0]
    z_delta = valid_z[-1] - valid_z[0] if len(valid_z) >= 2 else None
    if z_delta is None:
        camera_trend = "unknown"
        scale_trend = "unknown"
    elif z_delta > 0.20:
        camera_trend = "away_from_camera"
        scale_trend = "smaller_over_time"
    elif z_delta < -0.20:
        camera_trend = "toward_camera"
        scale_trend = "larger_over_time"
    else:
        camera_trend = "roughly_constant_depth"
        scale_trend = "roughly_constant_scale"
    signed_area = 0.0
    if len(poly2d) >= 3:
        pts = poly2d
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            signed_area += _float(pts[i][0]) * _float(pts[j][1]) - _float(pts[j][0]) * _float(pts[i][1])
    turn_direction = "counterclockwise" if signed_area > 20 else "clockwise" if signed_area < -20 else "not_orbital"
    return {
        "available": True,
        "heading_start_deg": heading_start,
        "heading_end_deg": heading_end,
        "image_direction": image_direction,
        "camera_depth_trend": camera_trend,
        "depth_delta_m": round(float(z_delta), 4) if z_delta is not None else None,
        "scale_trend": scale_trend,
        "approach_retreat": "approach" if camera_trend == "toward_camera" else "retreat" if camera_trend == "away_from_camera" else "neutral",
        "orbit_direction": turn_direction,
    }


def _shape_justification_reasons(path: Dict[str, Any], poly2d: List[List[float]], straight_ratio: float) -> List[str]:
    manifold = str(path.get("manifold_type", "ribbon_path"))
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    reasons: List[str] = []
    if manifold in {"contact_patch", "effect_field"}:
        reasons.append("local_anchor_not_route")
    if manifold == "occlusion_pulse":
        reasons.append("edge_pulse_anchor_set")
    if manifold == "portal_path":
        reasons.append("entry_exit_transition")
    if len(poly2d) > 3 and straight_ratio < 0.96:
        reasons.append("multi_vertex_scene_shape")
    if boundary.get("available"):
        reasons.append("region_boundary_context_sampled")
    if path.get("support_trace"):
        reasons.append("support_trace_sampled")
    return sorted(set(reasons))


def _path_shape_contract(
    path: Dict[str, Any],
    poly2d: List[List[float]],
    poly3d: List[List[float]],
    left_boundary: List[List[float]],
    right_boundary: List[List[float]],
) -> Dict[str, Any]:
    manifold = str(path.get("manifold_type", "ribbon_path"))
    length = _polyline_length_px(poly2d)
    chord = math.hypot(_float(poly2d[-1][0]) - _float(poly2d[0][0]), _float(poly2d[-1][1]) - _float(poly2d[0][1])) if len(poly2d) >= 2 else 0.0
    straight = _straightness_ratio(poly2d)
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    support_counts = dict((path.get("semantic_trace") or {}).get("support_kind_counts") or {})
    geometry_quality = path.get("path_geometry_quality") if isinstance(path.get("path_geometry_quality"), dict) else {}
    reasons = _shape_justification_reasons(path, poly2d, straight)
    rejection: List[str] = []
    route_manifold = manifold in {"ribbon_path", "contour_path", "portal_path", "volume_path", "blob_path", "interior_path"}
    if route_manifold and len(poly2d) <= 2 and manifold not in {"contact_patch", "effect_field", "occlusion_pulse"}:
        rejection.append("two_point_route_shape_underexplained")
    if route_manifold and straight >= 0.985 and len(poly2d) <= 3 and _float(boundary.get("transition_count"), 0.0) > 0:
        rejection.append("straight_line_crosses_region_transition_without_shape_context")
    if manifold == "ribbon_path" and _support_grounding_confidence(path, support_counts) < 0.25:
        rejection.append("ribbon_path_lacks_local_support_evidence")
    rejection.extend(str(r) for r in list(geometry_quality.get("geometry_rejection_reasons") or []) if str(r))
    shape_conf = 1.0
    if rejection:
        shape_conf -= 0.35
    if geometry_quality.get("smoothability_status") == "rejected":
        shape_conf -= 0.18
    if _float(geometry_quality.get("zigzag_score"), 0.0) > 0.0:
        shape_conf -= min(0.18, _float(geometry_quality.get("zigzag_score"), 0.0) * 0.12)
    if route_manifold and straight >= 0.985 and len(poly2d) <= 3:
        shape_conf -= 0.20
    if len(poly2d) >= 4:
        shape_conf += 0.08
    return {
        "schema": "citv_path_shape_contract_v1",
        "shape_type": _manifold_shape_type(manifold),
        "manifold_type": manifold,
        "geometry_refs": {
            "raw_centerline": "polyline_2d_raw" if path.get("polyline_2d_raw") else "polyline_2d",
            "validated_centerline": "polyline_2d_validated" if path.get("polyline_2d_validated") else "polyline_2d",
            "display_centerline": "display_polyline_2d" if path.get("display_polyline_2d") else "polyline_2d",
            "centerline": "display_polyline_2d" if path.get("display_polyline_2d") else "polyline_2d_reprojected" if path.get("polyline_2d_reprojected") else "polyline_2d",
            "polyline_3d": "polyline_3d",
            "display_polyline_3d": "display_polyline_3d" if path.get("display_polyline_3d") else "",
            "left_boundary": "left_boundary_2d" if left_boundary else "",
            "right_boundary": "right_boundary_2d" if right_boundary else "",
            "mask_or_effect_refs": [
                key for key in ("blob_mask_id", "occluder_id", "portal_entry_uv", "contact_points", "effect_center_uv")
                if path.get(key) is not None
            ],
        },
        "vertex_count": len(poly2d),
        "length_px": round(float(length), 3),
        "direct_distance_px": round(float(chord), 3),
        "straightness_ratio": round(float(straight), 4),
        "straight_line_like": bool(straight >= 0.985 and len(poly2d) <= 3),
        "direction_profile": _path_direction_profile(poly2d, poly3d),
        "support_evidence": {
            "support_kind_counts": support_counts,
            "support_grounding_confidence": round(float(_support_grounding_confidence(path, support_counts)), 4),
            "movement_scope": str(boundary.get("movement_scope", "")),
            "boundary_interaction": str(boundary.get("boundary_interaction", "")),
            "region_transition_count": int(_float(boundary.get("transition_count"), 0.0)),
        },
        "confidence": round(max(0.0, min(1.0, shape_conf)), 4),
        "shape_justification": reasons,
        "geometry_quality": {
            "zigzag_score": geometry_quality.get("zigzag_score"),
            "turn_angle_p95": geometry_quality.get("turn_angle_p95"),
            "curvature_energy": geometry_quality.get("curvature_energy"),
            "vertical_shoot_score": geometry_quality.get("vertical_shoot_score"),
            "depth_jump_count": geometry_quality.get("depth_jump_count"),
            "support_snap_displacement_px": geometry_quality.get("support_snap_displacement_px"),
            "smoothability_status": geometry_quality.get("smoothability_status"),
        },
        "rejection_reasons": rejection,
    }


def _alpha_policy_for_manifold(manifold: str) -> str:
    if manifold == "occlusion_pulse":
        return "peek_hide_pulse_from_visibility_curve"
    if manifold == "portal_path":
        return "fade_taper_enter_exit"
    if manifold == "effect_field":
        return "effect_opacity_wave"
    if manifold == "volume_path":
        return "depth_visibility_volume"
    return "visibility_profile"


def _render_primitive_for_manifold(manifold: str) -> str:
    return {
        "ribbon_path": "depth_tapered_corridor_plus_actor",
        "contour_path": "boundary_following_actor",
        "interior_path": "area_constrained_blob_motion",
        "blob_path": "area_constrained_blob_motion",
        "volume_path": "open_volume_actor_or_field",
        "portal_path": "fade_taper_entry_exit_actor",
        "occlusion_pulse": "edge_anchored_hide_peek_pulse",
        "contact_patch": "anchored_reach_hold_contact",
        "effect_field": "local_wave_or_reflection_field",
    }.get(manifold, "polyline_actor")


def _animation_render_contract(path: Dict[str, Any]) -> Dict[str, Any]:
    manifold = str(path.get("manifold_type", "ribbon_path"))
    wp = list(path.get("width_profile_px") or [])
    vp = list(path.get("visibility_profile") or [])
    motion_labels = []
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict) and str(hint.get("motion", "")).strip():
            motion_labels.append(str(hint.get("motion", "")).strip())
        elif isinstance(hint, str) and hint.strip():
            motion_labels.append(hint.strip())
    action_labels = []
    for hint in list(path.get("action_hints") or []):
        if isinstance(hint, dict):
            action_labels.extend(str(hint.get(k, "")).strip() for k in ("action", "path_mode") if str(hint.get(k, "")).strip())
    occlusion = path.get("occlusion_trace") if isinstance(path.get("occlusion_trace"), dict) else {}
    render_layers = list(path.get("render_layers") or [])
    if not render_layers:
        render_layers = ["in_front"]
    return {
        "schema": "citv_animation_render_contract_v1",
        "render_primitive": _render_primitive_for_manifold(manifold),
        "manifold_type": manifold,
        "motion_labels": sorted(set(motion_labels))[:8],
        "action_labels": sorted(set(action_labels))[:10],
        "alpha_policy": _alpha_policy_for_manifold(manifold),
        "width_policy": "depth_width_profile" if wp else "fixed_minimum_width",
        "depth_scale_policy": "metric_depth_trace" if path.get("depth_trace_m") else "image_space_fallback",
        "render_layers": render_layers,
        "occluders": list(occlusion.get("occluder_ids") or []),
        "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
        "host_mask_contact_anchors": {
            "contact_points": list(path.get("contact_points") or [])[:8],
            "approach_points": list(path.get("approach_points") or [])[:4],
            "anchor_points": list(path.get("anchor_points") or [])[:8],
            "portal_entry_uv": path.get("portal_entry_uv"),
            "effect_center_uv": path.get("effect_center_uv"),
            "blob_mask_id": path.get("blob_mask_id"),
        },
        "effect_parameters": {
            "pulse_period_s": path.get("pulse_period_s"),
            "frequency_hz": path.get("frequency_hz"),
            "alpha_range": path.get("alpha_range"),
            "oscillation_radius_px": path.get("oscillation_radius_px"),
        },
        "sample_state_fields": [
            "s", "position_px", "depth_m", "heading_deg", "speed_hint", "scale_hint",
            "width_px", "alpha", "visible_fraction", "render_layer", "motion_label", "occluder_ids",
        ],
        "sample_state_preview": _render_contract_sample_preview(path, wp, vp),
    }


def _render_contract_sample_preview(
    path: Dict[str, Any],
    width_profile: List[Dict[str, Any]],
    visibility_profile: List[Dict[str, Any]],
    *,
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    pts = _polyline_2d({"polyline_2d": path.get("display_polyline_2d") or path.get("polyline_2d_reprojected") or path.get("polyline_2d") or []})
    if not pts:
        return []
    sampled = _sample_polyline(pts, max_items=min(max_items, max(2, len(pts))))
    widths = _sample_widths(width_profile, max_items=len(sampled))
    vis = _decimate(visibility_profile, max_items=len(sampled)) if visibility_profile else []
    depth_rows = [r for r in list(path.get("depth_trace_m") or []) if isinstance(r, dict) and not r.get("summary")]
    depths = _decimate(depth_rows, max_items=len(sampled)) if depth_rows else []
    motion = "traverse"
    for hint in list(path.get("motion_hints") or []):
        if isinstance(hint, dict) and str(hint.get("motion", "")).strip():
            motion = str(hint.get("motion", "")).strip()
            break
    rows: List[Dict[str, Any]] = []
    for i, (s, u, v) in enumerate(sampled):
        vi = vis[i] if i < len(vis) and isinstance(vis[i], dict) else {}
        di = depths[i] if i < len(depths) and isinstance(depths[i], dict) else {}
        visible = _float(vi.get("visible_fraction"), 1.0)
        width_px = widths[i] if i < len(widths) else 4.0
        if i < len(sampled) - 1:
            _, nx, ny = sampled[i + 1]
            heading = math.degrees(math.atan2(ny - v, nx - u))
        elif rows:
            heading = _float(rows[-1].get("heading_deg"), 0.0)
        else:
            heading = 0.0
        rows.append({
            "s": round(float(s), 4),
            "position_px": [round(float(u), 3), round(float(v), 3)],
            "depth_m": _float(di.get("z_m"), _float(vi.get("z_m"), -1.0)),
            "heading_deg": round(float(heading), 3),
            "speed_hint": _float((path.get("cost_trace") or {}).get("mean_speed"), 0.0),
            "scale_hint": round(float(max(0.25, min(2.5, width_px / 12.0))), 4),
            "width_px": round(float(width_px), 3),
            "alpha": round(float(max(0.15, min(1.0, visible))), 4),
            "visible_fraction": round(float(visible), 4),
            "render_layer": str(vi.get("render_layer", "in_front")),
            "motion_label": motion,
            "occluder_ids": list(vi.get("occluder_ids") or []),
        })
    return rows


def _manifold_acceptance_thresholds(manifold: str) -> Dict[str, float]:
    base = {
        "accepted_confidence": 0.48,
        "accepted_manifold_fit": 0.54,
        "accepted_local_grounding": 0.28,
        "accepted_geometry_contract": 0.45,
        "accepted_renderability": 0.42,
        "accepted_contradiction_score": 0.28,
        "plausible_confidence": 0.32,
        "plausible_manifold_fit": 0.34,
        "plausible_geometry_contract": 0.30,
        "plausible_renderability": 0.28,
        "plausible_contradiction_score": 0.45,
    }
    m = str(manifold or "ribbon_path")
    if m == "ribbon_path":
        base.update({"accepted_manifold_fit": 0.58, "accepted_contradiction_score": 0.24, "plausible_manifold_fit": 0.40})
    elif m in {"portal_path", "contour_path"}:
        base.update({"accepted_manifold_fit": 0.53, "plausible_manifold_fit": 0.36})
    elif m in {"contact_patch", "occlusion_pulse", "effect_field"}:
        base.update({
            "accepted_confidence": 0.42,
            "accepted_manifold_fit": 0.46,
            "accepted_local_grounding": 0.34,
            "accepted_contradiction_score": 0.34,
            "plausible_confidence": 0.28,
            "plausible_manifold_fit": 0.30,
            "plausible_contradiction_score": 0.52,
        })
    elif m == "volume_path":
        base.update({
            "accepted_confidence": 0.44,
            "accepted_manifold_fit": 0.45,
            "accepted_geometry_contract": 0.35,
            "plausible_confidence": 0.28,
            "plausible_manifold_fit": 0.30,
            "plausible_geometry_contract": 0.24,
        })
    return base


def _path_contract_status(path: Dict[str, Any], ontology: Dict[str, Any]) -> Dict[str, Any]:
    scores = path.get("scores") if isinstance(path.get("scores"), dict) else {}
    conf = _float(scores.get("overall_confidence"), 0.0)
    manifold = str(path.get("manifold_type", "ribbon_path"))
    route_manifold = manifold in {"ribbon_path", "contour_path", "portal_path", "blob_path", "interior_path"}
    support_counts = dict((path.get("semantic_trace") or {}).get("support_kind_counts") or {})
    total = max(1.0, sum(_float(v) for v in support_counts.values()))
    unknown_frac = _float(support_counts.get("unknown"), 0.0) / total
    blocking_frac = _float(support_counts.get("blocking"), 0.0) / total
    manifold_fit = _float(scores.get("manifold_fit_score"), _support_grounding_confidence(path, support_counts))
    local_grounding = _float(scores.get("local_grounding_score"), _local_action_evidence_confidence(path))
    geometry_contract = _float(scores.get("geometry_contract_score"), 0.5)
    renderability = _float(scores.get("renderability_score"), _float((path.get("occlusion_trace") or {}).get("mean_visible_fraction"), 1.0))
    contradiction_score = _float(scores.get("contradiction_score"), 0.0)
    uncertainty_score = _float(scores.get("uncertainty_score"), 0.0)
    shape = path.get("path_shape_contract") if isinstance(path.get("path_shape_contract"), dict) else {}
    raw_reasons: List[str] = list(shape.get("rejection_reasons") or [])
    quality = path.get("path_geometry_quality") if isinstance(path.get("path_geometry_quality"), dict) else {}
    raw_reasons.extend(str(r) for r in list(quality.get("geometry_rejection_reasons") or []) if str(r))
    notes: List[str] = []
    if not path.get("polyline_2d"):
        raw_reasons.append("missing_2d_geometry")
    if route_manifold and unknown_frac >= 0.80 and _support_grounding_confidence(path, support_counts) < 0.20:
        raw_reasons.append("unknown_only_or_unsupported_route")
    if route_manifold and (blocking_frac >= 0.50 or contradiction_score >= 0.55):
        raw_reasons.append("blocking_support_dominates_route")
    if (not route_manifold) and blocking_frac >= 0.50:
        notes.append("blocking_context_present_for_non_route_manifold")
    if _entity_anchor_confidence(path, ontology) < 0.32 and str(path.get("path_level", "")) == "object":
        raw_reasons.append("weak_source_or_target_anchor")
    if _local_action_evidence_confidence(path) <= 0.01:
        raw_reasons.append("no_local_action_evidence")
    elif _local_action_evidence_confidence(path) < 0.22:
        notes.append("local_action_evidence_weak")
    if contradiction_score >= 0.58:
        raw_reasons.append("impossible_manifold_evidence_mismatch")
    if uncertainty_score >= 0.62 and local_grounding < 0.35:
        raw_reasons.append("insufficient_local_grounding_evidence")
    if bool(shape.get("straight_line_like")) and str(path.get("manifold_type", "")) in {"ribbon_path", "portal_path"}:
        if not shape.get("shape_justification"):
            raw_reasons.append("straight_line_shape_unjustified")
        else:
            notes.append("straight_line_shape_requires_review")
    grounding = path.get("grounding_evidence") if isinstance(path.get("grounding_evidence"), dict) else {}
    if bool(grounding.get("global_only")):
        raw_reasons.append("global_only_action_without_local_grounding")
    hard = {
        "missing_2d_geometry",
        "blocking_support_dominates_route",
        "geometry_missing_display_polyline",
        "geometry_support_snap_displacement_too_large",
        "geometry_low_feasible_fraction",
        "global_only_action_without_local_grounding",
        "impossible_manifold_evidence_mismatch",
        "invalid_occlusion_render_contract",
    }
    uncertainty_only = {
        "unknown_only_or_unsupported_route",
        "weak_source_or_target_anchor",
        "ribbon_path_lacks_local_support_evidence",
        "geometry_low_support_fraction",
        "insufficient_local_grounding_evidence",
        "geometry_high_zigzag",
        "geometry_sharp_turns_unexplained",
        "geometry_vertical_shoot_unexplained",
        "geometry_depth_jump_unexplained",
        "straight_line_shape_unjustified",
        "straight_line_crosses_region_transition_without_shape_context",
        "two_point_route_shape_underexplained",
        "no_local_action_evidence",
    }
    raw_reasons = sorted(set(r for r in raw_reasons if r))
    contradictions = sorted(set(raw_reasons).intersection(hard))
    uncertainty = sorted(set(raw_reasons).intersection(uncertainty_only))
    unclassified = sorted(set(raw_reasons) - set(contradictions) - set(uncertainty))
    uncertainty.extend(unclassified)
    has_local = local_grounding > 0.01 or _float(grounding.get("local_evidence_confidence"), 0.0) > 0.01
    thresholds = _manifold_acceptance_thresholds(manifold)
    critical_uncertainty = {"unknown_only_or_unsupported_route", "geometry_low_support_fraction", "ribbon_path_lacks_local_support_evidence"}
    if (
        conf >= thresholds["accepted_confidence"]
        and manifold_fit >= thresholds["accepted_manifold_fit"]
        and local_grounding >= thresholds["accepted_local_grounding"]
        and geometry_contract >= thresholds["accepted_geometry_contract"]
        and renderability >= thresholds["accepted_renderability"]
        and contradiction_score < thresholds["accepted_contradiction_score"]
        and not contradictions
        and has_local
        and not set(uncertainty).intersection(critical_uncertainty)
    ):
        status = "accepted"
    elif (
        conf >= thresholds["plausible_confidence"]
        and manifold_fit >= thresholds["plausible_manifold_fit"]
        and geometry_contract >= thresholds["plausible_geometry_contract"]
        and renderability >= thresholds["plausible_renderability"]
        and contradiction_score < thresholds["plausible_contradiction_score"]
        and not contradictions
        and path.get("polyline_2d")
    ):
        status = "plausible_uncertain"
    elif conf >= 0.20 and "missing_2d_geometry" not in raw_reasons:
        status = "low_confidence"
    else:
        status = "rejected"
    if not notes and status == "accepted":
        notes.append("local_scene_evidence_sufficient_for_current_cpu_contract")
    if status == "plausible_uncertain" and not notes:
        notes.append("plausible_but_under_evidenced")
    return {
        "schema": "citv_path_contract_status_v1",
        "status": status,
        "confidence": round(float(conf), 4),
        "rejection_reasons": contradictions if status == "rejected" else [],
        "uncertainty_reasons": sorted(set(uncertainty))[:16],
        "contradiction_reasons": sorted(set(contradictions))[:16],
        "all_contract_reasons": raw_reasons[:24],
        "validation_notes": sorted(set(notes))[:12],
        "unknown_support_fraction": round(float(unknown_frac), 4),
        "blocking_support_fraction": round(float(blocking_frac), 4),
        "manifold_fit_score": round(float(manifold_fit), 4),
        "local_grounding_score": round(float(local_grounding), 4),
        "geometry_contract_score": round(float(geometry_contract), 4),
        "renderability_score": round(float(renderability), 4),
        "contradiction_score": round(float(contradiction_score), 4),
        "uncertainty_score": round(float(uncertainty_score), 4),
        "local_action_evidence_confidence": round(float(_local_action_evidence_confidence(path)), 4),
        "entity_anchor_confidence": round(float(_entity_anchor_confidence(path, ontology)), 4),
    }


def _apply_contract_score_gate(path: Dict[str, Any], ontology: Dict[str, Any]) -> None:
    status = path.get("contract_status") if isinstance(path.get("contract_status"), dict) else {}
    scores = dict(path.get("scores") or {})
    old = _float(scores.get("overall_confidence"), 0.0)
    manifold = str(path.get("manifold_type", "ribbon_path"))
    route_manifold = manifold in {"ribbon_path", "contour_path", "portal_path", "blob_path", "interior_path"}
    thresholds = _manifold_acceptance_thresholds(manifold)
    reasons = set(str(r) for r in status.get("rejection_reasons") or [])
    uncertainty = set(str(r) for r in status.get("uncertainty_reasons") or [])
    all_reasons = reasons.union(uncertainty)
    multiplier = 1.0
    if "unknown_only_or_unsupported_route" in all_reasons:
        multiplier *= 0.84 if route_manifold else 0.94
    if "blocking_support_dominates_route" in reasons:
        multiplier *= 0.52 if route_manifold else 0.88
    if "weak_source_or_target_anchor" in all_reasons:
        multiplier *= 0.86
    if "no_local_action_evidence" in all_reasons:
        multiplier *= 0.84
    if "insufficient_local_grounding_evidence" in all_reasons:
        multiplier *= 0.88
    if "two_point_route_shape_underexplained" in all_reasons or "straight_line_shape_unjustified" in all_reasons:
        multiplier *= 0.88
    if "geometry_high_zigzag" in all_reasons:
        multiplier *= 0.82
    if "geometry_sharp_turns_unexplained" in all_reasons:
        multiplier *= 0.84
    if "geometry_vertical_shoot_unexplained" in all_reasons:
        multiplier *= 0.80
    if "geometry_depth_jump_unexplained" in all_reasons:
        multiplier *= 0.84
    if "geometry_support_snap_displacement_too_large" in reasons:
        multiplier *= 0.55
    if "geometry_low_feasible_fraction" in reasons:
        multiplier *= 0.58
    if "geometry_low_support_fraction" in all_reasons:
        multiplier *= 0.88
    contradiction_score = _float(scores.get("contradiction_score"), 0.0)
    uncertainty_score = _float(scores.get("uncertainty_score"), 0.0)
    if contradiction_score >= 0.55:
        multiplier *= 0.58
    elif contradiction_score >= 0.40:
        multiplier *= 0.78
    if uncertainty_score >= 0.65:
        multiplier *= 0.84
    shape_conf = _float((path.get("path_shape_contract") or {}).get("confidence"), 0.5)
    contract_conf = max(0.0, min(1.0, _float(status.get("confidence"), old) * multiplier))
    scores["shape_confidence"] = round(float(shape_conf), 4)
    scores["contract_confidence"] = round(float(contract_conf), 4)
    scores["overall_confidence_before_contract_gate"] = round(float(old), 4)
    scores["overall_confidence"] = round(float(min(old, contract_conf)), 4)
    path["scores"] = scores
    status["confidence"] = scores["overall_confidence"]
    # Recompute status after the confidence gate without dropping the reasons.
    if status.get("status") == "accepted" and scores["overall_confidence"] < thresholds["accepted_confidence"]:
        status["status"] = "plausible_uncertain"
    if status.get("status") == "plausible_uncertain" and scores["overall_confidence"] < thresholds["plausible_confidence"]:
        status["status"] = "low_confidence"
    if status.get("status") == "low_confidence" and scores["overall_confidence"] < 0.18:
        status["status"] = "rejected"
    path["contract_status"] = status
    path["acceptance_status"] = str(status.get("status", "low_confidence"))
    path["rejection_reasons"] = list(status.get("rejection_reasons") or [])
    path["uncertainty_reasons"] = list(status.get("uncertainty_reasons") or [])
    path["contradiction_reasons"] = list(status.get("contradiction_reasons") or [])
    path["validation_notes"] = list(status.get("validation_notes") or [])


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Trajectory contract — animation-ready summary per path
# ─────────────────────────────────────────────────────────────────────────────

def _trajectory_contract(path: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble one concise contract dict that the animation stage can consume directly."""
    scores = dict(path.get("scores") or {})
    occ = path.get("occlusion_trace") or {}
    cost = path.get("cost_trace") or {}
    depth_rows = [r for r in (path.get("display_depth_trace_m") or path.get("depth_trace_m") or []) if not r.get("summary") and _float(r.get("z_m"), -1.0) > 0.0]
    valid_z = [_float(r["z_m"]) for r in depth_rows]
    motion_hints = path.get("motion_hints") or []
    dominant_motion = "walk"
    for h in motion_hints:
        m = str(h.get("motion", "")) if isinstance(h, dict) else str(h)
        if m.strip():
            dominant_motion = m.strip()
            break
    nzt = path.get("navigation_zone_trace") or {}
    quality = path.get("path_geometry_quality") if isinstance(path.get("path_geometry_quality"), dict) else {}
    nav_dominant = str(nzt.get("dominant_motion_hint") or "")
    boundary = path.get("region_boundary_trace") if isinstance(path.get("region_boundary_trace"), dict) else {}
    kin_sigs = path.get("kinematic_signatures") or []
    support_counts = dict((path.get("semantic_trace") or {}).get("support_kind_counts") or {})
    dominant_support = max(support_counts, key=support_counts.get) if support_counts else "unknown"
    ground_cls = (
        path.get("ground_object_classification")
        if isinstance(path.get("ground_object_classification"), dict)
        else _ground_object_classification(path, None)
    )
    wp = path.get("width_profile_px") or []
    widths = [_float(r.get("width_px"), 4.0) for r in wp if isinstance(r, dict)]
    has_3d = bool(path.get("polyline_3d"))
    has_viz = bool(path.get("visibility_profile"))
    has_nz = bool(nav_dominant)
    has_kin = bool(kin_sigs)
    reasons = []
    if has_3d:
        reasons.append("polyline_3d")
    if has_viz:
        reasons.append("visibility")
    if has_nz:
        reasons.append("nav_zones")
    if has_kin:
        reasons.append("kinematic_sigs")
    animation_ready = has_3d and has_viz
    return {
        "schema": "citv_trajectory_contract_v1",
        "manifold_type": str(path.get("manifold_type", "ribbon_path")),
        "action_family": str(path.get("action_family", "locomotion")),
        "path_shape_contract_ref": "path_shape_contract",
        "animation_render_contract_ref": "animation_render_contract",
        "display_geometry_ref": "display_polyline_2d" if path.get("display_polyline_2d") else "polyline_2d",
        "shape_type": str((path.get("path_shape_contract") or {}).get("shape_type", "")),
        "direction_profile": dict((path.get("path_shape_contract") or {}).get("direction_profile") or {}),
        "acceptance_status": str(path.get("acceptance_status", "")),
        "rejection_reasons": list(path.get("rejection_reasons") or []),
        "dominant_motion": dominant_motion,
        "nav_zone_dominant": nav_dominant,
        "nav_zone_channels": dict(nzt.get("mean_channels") or {}),
        "movement_scope": str(boundary.get("movement_scope", "")),
        "boundary_interaction": str(boundary.get("boundary_interaction", "")),
        "region_transition_count": int(_float(boundary.get("transition_count"), 0.0)),
        "boundary_sample_fraction": _float(boundary.get("boundary_sample_fraction"), 0.0),
        "region_sequence": list(boundary.get("regions_sequence") or [])[:8],
        "boundary_motion_implications": list(boundary.get("motion_implications") or [])[:8],
        "kinematic_signatures": kin_sigs,
        "depth_min_m": round(min(valid_z), 4) if valid_z else None,
        "depth_max_m": round(max(valid_z), 4) if valid_z else None,
        "depth_mean_m": round(float(np.mean(valid_z)), 4) if valid_z else None,
        "depth_delta_m": round(valid_z[-1] - valid_z[0], 4) if len(valid_z) >= 2 else None,
        "mean_visible_fraction": _float(occ.get("mean_visible_fraction"), 1.0),
        "min_visible_fraction": _float(occ.get("min_visible_fraction"), 1.0),
        "has_occlusion": _float(occ.get("occluded_sample_fraction"), 0.0) > 0.05,
        "occluder_ids": list(occ.get("occluder_ids") or []),
        "mean_cost": _float(cost.get("mean_cost")),
        "p90_cost": _float(cost.get("p90_cost")),
        "mean_speed": _float(cost.get("mean_speed")),
        "support_dominant": dominant_support,
        "ground_object_classification": dict(ground_cls),
        "recommended_motion": str(ground_cls.get("recommended_motion", dominant_motion)),
        "geometry_quality": {
            "zigzag_score": quality.get("zigzag_score"),
            "turn_angle_p95": quality.get("turn_angle_p95"),
            "vertical_shoot_score": quality.get("vertical_shoot_score"),
            "depth_jump_count": quality.get("depth_jump_count"),
            "smoothability_status": quality.get("smoothability_status"),
            "geometry_rejection_reasons": list(quality.get("geometry_rejection_reasons") or []),
        },
        "width_min_px": round(min(widths), 3) if widths else None,
        "width_max_px": round(max(widths), 3) if widths else None,
        "width_mean_px": round(float(np.mean(widths)), 3) if widths else None,
        "overall_confidence": _float(scores.get("overall_confidence")),
        "geometric_confidence": _float(scores.get("geometric_confidence")),
        "semantic_confidence": _float(scores.get("semantic_confidence")),
        "has_3d": has_3d,
        "has_visibility": has_viz,
        "has_nav_zones": has_nz,
        "has_kinematic_sigs": has_kin,
        "animation_ready": animation_ready,
        "animation_readiness_reason": "+".join(reasons) if reasons else "none",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Non-locomotion manifold hypothesis generation
# ─────────────────────────────────────────────────────────────────────────────

_MANIFOLD_MIN_SCORE = 0.18


def _emit_aerial_approach_hypotheses(
    ctx: PipelineContext,
    objects: List[Dict[str, Any]],
    cfg: Any,
) -> List[Dict[str, Any]]:
    """Extra ``volume_path`` samples: open-air toward actor-like objects (scalable, ontology-free NN)."""
    if not bool(getattr(cfg, "path_emit_aerial_approach_hypotheses", True)) if cfg else True:
        return []
    h, w = ctx.height, ctx.width
    obj_aff_by_id = {
        str(o.get("object_id", "")): o
        for o in list((ctx.object_affordances or {}).get("objects") or [])
        if isinstance(o, dict)
    }
    out: List[Dict[str, Any]] = []
    top_y = max(2, int(h * 0.06))
    cx_img = w // 2
    for obj in objects:
        oid = str(obj.get("id", ""))
        if not oid:
            continue
        aff = obj_aff_by_id.get(oid, {})
        act_scores = {str(a.get("name", "")): _float(a.get("score")) for a in (aff.get("actions") or [])}
        role_scores = {str(r.get("name", "")): _float(r.get("score")) for r in (aff.get("roles") or [])}
        fly = max(act_scores.get("fly", 0.0), act_scores.get("hover", 0.0))
        actor = float(role_scores.get("actor", 0.0))
        sky_role = float(role_scores.get("sky_open_air", 0.0))
        if fly < 0.15 and actor < 0.35 and sky_role < 0.2:
            continue
        uv = obj.get("mask_centroid_2d") or _bbox_center_uv(obj) or [cx_img, h // 2]
        tx, ty = int(round(float(uv[0]))), int(round(float(uv[1])))
        tx = int(np.clip(tx, 0, w - 1))
        ty = int(np.clip(ty, 0, h - 1))
        poly = [[float(cx_img), float(top_y)], [float((cx_img + tx) / 2), float((top_y + ty) / 2)], [float(tx), float(ty)]]
        out.append({
            "path_id": f"manifold_aerial_approach_{oid}",
            "path_level": "scene",
            "path_type": "volume_path",
            "manifold_type": "volume_path",
            "action_family": "locomotion",
            "action_name": "fly",
            "source_entity": {"type": "open_air", "id": "sky_band"},
            "target_entity": {"type": "object", "id": oid},
            "polyline_2d": poly,
            "volume_samples_2d": poly,
            "scores": {"overall_confidence": round(min(0.85, 0.35 + fly + 0.25 * actor), 4)},
            "routing_meta": {"motion_channel": "aerial", "path_granularity": "scene"},
            "constraint_refs": [],
        })
    return out[: max(1, int(getattr(cfg, "path_max_aerial_hypotheses", 12)))]


def _emit_contour_hypotheses(
    ctx: PipelineContext,
    objects: List[Dict[str, Any]],
    cfg: Any,
) -> List[Dict[str, Any]]:
    """``contour_path`` along obstacle-like instances (sampled mask boundary)."""
    if not bool(getattr(cfg, "path_emit_contour_hypotheses", True)) if cfg else True:
        return []
    try:
        import cv2
    except Exception:
        return []
    h, w = ctx.height, ctx.width
    obj_aff_by_id = {
        str(o.get("object_id", "")): o
        for o in list((ctx.object_affordances or {}).get("objects") or [])
        if isinstance(o, dict)
    }
    max_h = int(getattr(cfg, "path_max_contour_hypotheses", 8)) if cfg else 8
    out: List[Dict[str, Any]] = []
    for obj in objects:
        if len(out) >= max_h:
            break
        oid = str(obj.get("id", ""))
        aff = obj_aff_by_id.get(oid, {})
        role_scores = {str(r.get("name", "")): _float(r.get("score")) for r in (aff.get("roles") or [])}
        obst = max(
            role_scores.get("hard_obstacle", 0.0),
            role_scores.get("soft_obstacle", 0.0),
            role_scores.get("occluder", 0.0),
        )
        if obst < 0.22:
            continue
        mm = _mask_array(obj, h, w)
        if mm is None or not mm.any():
            continue
        m8 = (mm.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        longest = max(contours, key=lambda c: cv2.arcLength(c, closed=False))
        if longest.shape[0] < 3:
            continue
        step = max(1, len(longest) // 28)
        pts = longest[::step].reshape(-1, 2)
        poly = [[float(x), float(y)] for x, y in pts]
        if len(poly) < 3:
            continue
        out.append({
            "path_id": f"manifold_contour_{oid}",
            "path_level": "object",
            "path_type": "contour_path",
            "manifold_type": "contour_path",
            "action_family": "locomotion",
            "action_name": "inspect",
            "source_entity": {"type": "object", "id": oid},
            "target_entity": {"type": "object", "id": oid},
            "polyline_2d": poly,
            "scores": {"overall_confidence": round(min(0.78, 0.28 + obst), 4)},
            "routing_meta": {"motion_channel": "ground", "path_granularity": "object_part"},
            "constraint_refs": [],
        })
    return out


def _generate_manifold_hypotheses(
    ctx: "PipelineContext",
    objects: List[Dict[str, Any]],
    feasible: np.ndarray,
    speed_map: np.ndarray,
    ontology: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Scan object affordances and produce non-locomotion manifold hypotheses.

    Generates blob_path, volume_path, occlusion_pulse, portal_path,
    effect_field, and contact_patch hypotheses where affordance evidence
    exceeds *_MANIFOLD_MIN_SCORE*.  Each hypothesis is self-contained so the
    action_export and animation_export stages can consume it without FMM.
    """
    obj_aff_by_id: Dict[str, Dict[str, Any]] = {
        str(o.get("object_id", "")): o
        for o in list((ctx.object_affordances or {}).get("objects") or [])
        if isinstance(o, dict)
    }
    mask_aff_by_id: Dict[str, Dict[str, Any]] = {
        str(m.get("object_id", "")): m
        for m in list((ctx.mask_affordances or {}).get("masks") or [])
        if isinstance(m, dict)
    }
    h, w = ctx.height, ctx.width
    results: List[Dict[str, Any]] = []
    ontology = ontology or load_action_ontology()
    policy = list_section(ontology, "manifold_policy")
    for obj in objects:
        oid = str(obj.get("id", ""))
        aff = obj_aff_by_id.get(oid, {})
        mask_aff = mask_aff_by_id.get(oid, {})
        act_scores = {str(a.get("name", "")): _float(a.get("score")) for a in (aff.get("actions") or [])}
        role_scores = {str(r.get("name", "")): _float(r.get("score")) for r in (aff.get("roles") or [])}

        for rule in policy:
            mtype = str(rule.get("manifold_type", ""))
            score, action_name = _manifold_policy_score(rule, act_scores, role_scores)
            if score < _MANIFOLD_MIN_SCORE:
                continue
            if mtype == "blob_path":
                m = _try_blob_path(obj, mask_aff, oid, h, w, score, action_name)
            elif mtype == "volume_path":
                m = _try_volume_path(obj, mask_aff, oid, h, w, ctx.metric_depth, score, action_name)
            elif mtype == "occlusion_pulse":
                m = _try_occlusion_pulse(obj, mask_aff, oid, h, w, score, action_name)
            elif mtype == "portal_path":
                m = _try_portal_path(obj, mask_aff, oid, feasible, speed_map, h, w, ctx.metric_depth, score, action_name)
            elif mtype == "effect_field":
                m = _try_effect_field(obj, mask_aff, oid, h, w, score, action_name)
            elif mtype == "contact_patch":
                m = _try_contact_patch(obj, mask_aff, aff, oid, h, w, ctx.metric_depth, score, action_name)
            else:
                continue
            if m:
                results.append(m)

    return results


def _manifold_policy_score(
    rule: Dict[str, Any],
    act_scores: Dict[str, float],
    role_scores: Dict[str, float],
) -> Tuple[float, str]:
    best_score = 0.0
    best_action = ""
    for name in [str(x) for x in list(rule.get("actions") or [])]:
        score = float(act_scores.get(name, 0.0))
        if score > best_score:
            best_score = score
            best_action = name
    for name in [str(x) for x in list(rule.get("roles") or [])]:
        score = float(role_scores.get(name, 0.0))
        if score > best_score:
            best_score = score
            best_action = str((rule.get("actions") or [name])[0])
    return best_score, best_action


def _mask_array(obj: Dict[str, Any], h: int, w: int) -> Optional[np.ndarray]:
    """Return boolean (H,W) mask, resized if needed, or None."""
    raw = obj.get("_sam2_mask_array")
    if raw is None:
        return None
    try:
        import cv2 as _cv2
        mm = np.asarray(raw, dtype=bool)
        if mm.shape[:2] != (h, w):
            mm = _cv2.resize(mm.astype(np.uint8), (w, h), interpolation=_cv2.INTER_NEAREST) > 0
        return mm
    except Exception:
        return None


def _mask_centroid(obj: Dict[str, Any], h: int, w: int) -> List[float]:
    uv = obj.get("mask_centroid_2d") or []
    if isinstance(uv, (list, tuple)) and len(uv) >= 2:
        return [_float(uv[0]), _float(uv[1])]
    bbox = list(obj.get("bbox") or [0, 0, 1, 1])[:4]
    return [_float(bbox[0]) + _float(bbox[2]) * 0.5, _float(bbox[1]) + _float(bbox[3]) * 0.5]


def _obj_depth_m(obj: Dict[str, Any]) -> float:
    v = (obj.get("depth_stats") or {}).get("median") or (obj.get("coordinates_3d") or {}).get("z") or 0.0
    return _float(v)


def _try_blob_path(
    obj: Dict[str, Any],
    mask_aff: Dict[str, Any],
    oid: str,
    h: int,
    w: int,
    score: float,
    action_name: str,
) -> Optional[Dict[str, Any]]:
    try:
        from ..pathing.mask_interior_path import mask_interior_geodesic_points
        mm = _mask_array(obj, h, w)
        if mm is None:
            return None
        pts = mask_interior_geodesic_points(mm, w, h)
        if len(pts) < 2:
            return None
        centroid = _mask_centroid(obj, h, w)
        geom = dict(mask_aff.get("geometry") or {})
        return {
            "path_id": f"manifold_blob_{oid}",
            "path_level": "object",
            "path_type": "blob_path",
            "manifold_type": "blob_path",
            "action_family": "locomotion",
            "action_name": action_name,
            "source_entity": {"type": "mask_interior", "id": oid},
            "target_entity": {"type": "mask_interior", "id": oid},
            "polyline_2d": [[float(p[0]), float(p[1])] for p in pts],
            "blob_mask_id": oid,
            "interior_seed_uv": centroid,
            "depth_m": _obj_depth_m(obj),
            "contour_sample_px": list(geom.get("contour_sample_px") or [])[:32],
            "label": str(obj.get("label", "")),
            "scores": {
                "overall_confidence": round(float(score), 4),
                "geometric_feasibility": 0.72,
                "semantic_confidence": round(float(score), 4),
            },
        }
    except Exception:
        return None


def _try_volume_path(
    obj: Dict[str, Any],
    mask_aff: Dict[str, Any],
    oid: str,
    h: int,
    w: int,
    metric_depth: Optional[np.ndarray],
    score: float,
    action_name: str,
) -> Optional[Dict[str, Any]]:
    """Sample a grid of (u,v) points inside the sky/open-air mask as a volume."""
    try:
        mm = _mask_array(obj, h, w)
        centroid = _mask_centroid(obj, h, w)
        if mm is not None and mm.any():
            ys, xs = np.where(mm)
            stride = max(1, int(math.sqrt(ys.size / 24.0)))
            sampled = list(zip(xs[::stride].tolist(), ys[::stride].tolist()))[:24]
            volume_pts = [[float(x), float(y)] for x, y in sampled]
        else:
            # Fallback: horizontal stripe across top third of image
            volume_pts = [[float(x), float(h * 0.15)] for x in range(0, w, max(1, w // 8))][:8]
        depth_range = [8.0, 40.0]
        if metric_depth is not None and mm is not None and mm.any():
            zs = metric_depth[mm]
            valid_z = zs[zs > 0.1]
            if valid_z.size > 0:
                depth_range = [float(np.percentile(valid_z, 10)), float(np.percentile(valid_z, 90))]
        return {
            "path_id": f"manifold_volume_{oid}",
            "path_level": "object",
            "path_type": "volume_path",
            "manifold_type": "volume_path",
            "action_family": "locomotion",
            "action_name": action_name,
            "source_entity": {"type": "sky_region", "id": oid},
            "target_entity": {"type": "sky_region", "id": oid},
            "polyline_2d": volume_pts,
            "volume_samples_2d": volume_pts,
            "center_uv": centroid,
            "depth_range_m": depth_range,
            "label": str(obj.get("label", "")),
            "scores": {
                "overall_confidence": round(float(score), 4),
                "geometric_feasibility": 0.65,
                "semantic_confidence": round(float(score), 4),
            },
        }
    except Exception:
        return None


def _try_occlusion_pulse(
    obj: Dict[str, Any],
    mask_aff: Dict[str, Any],
    oid: str,
    h: int,
    w: int,
    score: float,
    action_name: str,
) -> Optional[Dict[str, Any]]:
    """Build an occlusion-pulse manifold anchored to the object's boundary."""
    try:
        import cv2 as _cv2
        mm = _mask_array(obj, h, w)
        centroid = _mask_centroid(obj, h, w)
        anchor_pts: List[List[float]] = []
        if mm is not None and mm.any():
            boundary = mm.astype(np.uint8) - _cv2.erode(mm.astype(np.uint8), np.ones((5, 5), np.uint8))
            ys, xs = np.where(boundary > 0)
            if xs.size > 0:
                stride = max(1, xs.size // 6)
                anchor_pts = [[float(xs[i]), float(ys[i])] for i in range(0, xs.size, stride)][:6]
        if not anchor_pts:
            anchor_pts = [centroid]
        vis_curve = [
            {"t": 0.0, "visible_fraction": 0.10},
            {"t": 0.4, "visible_fraction": 0.55},
            {"t": 0.8, "visible_fraction": 0.15},
            {"t": 1.0, "visible_fraction": 0.08},
        ]
        geom = dict(mask_aff.get("geometry") or {})
        return {
            "path_id": f"manifold_occ_pulse_{oid}",
            "path_level": "object",
            "path_type": "occlusion_pulse",
            "manifold_type": "occlusion_pulse",
            "action_family": "occlusion_interaction",
            "action_name": action_name,
            "source_entity": {"type": "occluder", "id": oid},
            "target_entity": {"type": "occluder", "id": oid},
            "polyline_2d": anchor_pts,
            "occluder_id": oid,
            "anchor_points": anchor_pts,
            "visibility_curve": vis_curve,
            "pulse_period_s": 1.2,
            "depth_m": _obj_depth_m(obj),
            "occlusion_boundary_score": _float(geom.get("occlusion_boundary_score"), 0.5),
            "label": str(obj.get("label", "")),
            "scores": {
                "overall_confidence": round(float(score), 4),
                "geometric_feasibility": 0.70,
                "semantic_confidence": round(float(score), 4),
            },
        }
    except Exception:
        return None


def _try_portal_path(
    obj: Dict[str, Any],
    mask_aff: Dict[str, Any],
    oid: str,
    feasible: np.ndarray,
    speed_map: np.ndarray,
    h: int,
    w: int,
    metric_depth: Optional[np.ndarray],
    score: float,
    action_name: str,
) -> Optional[Dict[str, Any]]:
    """FMM path from image centre toward the portal object, with fade visibility."""
    try:
        from ..pathing.semantic_fmm import time_of_arrival_from_speed, k_diverse_from_T
        from ..pathing.walkable_mask import snap_uv_to_walkable
        centroid = _mask_centroid(obj, h, w)
        gp = snap_uv_to_walkable(int(centroid[0]), int(centroid[1]), feasible, w, h)
        sp = snap_uv_to_walkable(w // 2, h // 2, feasible, w, h)
        sm = np.where(feasible, speed_map, speed_map * 0.02)
        T = time_of_arrival_from_speed(sm, gp)
        if T is None:
            return None
        gpaths = k_diverse_from_T(T, sp, k=1, edge_penalty=0.35)
        if not gpaths or len(gpaths[0]) < 2:
            return None
        pts = [[float(p[0]), float(p[1])] for p in gpaths[0]]
        n_pts = len(pts)
        vis_profile = [{"t": round(i / max(1, n_pts - 1), 3), "visible_fraction": round(1.0 - i / max(1, n_pts - 1), 4)} for i in range(n_pts)]
        depth_m = _obj_depth_m(obj)
        if metric_depth is not None:
            xi = max(0, min(w - 1, int(centroid[0])))
            yi = max(0, min(h - 1, int(centroid[1])))
            dv = float(metric_depth[yi, xi])
            if dv > 0.0:
                depth_m = dv
        return {
            "path_id": f"manifold_portal_{oid}",
            "path_level": "object",
            "path_type": "portal_path",
            "manifold_type": "portal_path",
            "action_family": "locomotion",
            "action_name": action_name,
            "source_entity": {"type": "candidate_actor", "id": "actor"},
            "target_entity": {"type": "portal", "id": oid},
            "polyline_2d": pts,
            "visibility_profile": vis_profile,
            "portal_entry_uv": list(gp),
            "depth_m": depth_m,
            "label": str(obj.get("label", "")),
            "scores": {
                "overall_confidence": round(float(score), 4),
                "geometric_feasibility": 0.68,
                "semantic_confidence": round(float(score), 4),
            },
        }
    except Exception:
        return None


def _try_effect_field(
    obj: Dict[str, Any],
    mask_aff: Dict[str, Any],
    oid: str,
    h: int,
    w: int,
    score: float,
    action_name: str,
) -> Optional[Dict[str, Any]]:
    """Oscillating effect field on a reflective/transparent mask."""
    try:
        import cv2 as _cv2
        mm = _mask_array(obj, h, w)
        centroid = _mask_centroid(obj, h, w)
        geom = dict(mask_aff.get("geometry") or obj.get("geometry") or {})
        bbox = list(geom.get("bbox_px") or obj.get("bbox") or [0, 0, w, h])[:4]
        if mm is not None and mm.any():
            contours, _ = _cv2.findContours(mm.astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            contour_pts: List[List[float]] = []
            if contours:
                cnt = max(contours, key=_cv2.contourArea)
                epsilon = 0.02 * _cv2.arcLength(cnt, True)
                approx = _cv2.approxPolyDP(cnt, epsilon, True)
                contour_pts = [[float(p[0][0]), float(p[0][1])] for p in approx][:24]
        else:
            contour_pts = []
        return {
            "path_id": f"manifold_effect_{oid}",
            "path_level": "object",
            "path_type": "effect_field",
            "manifold_type": "effect_field",
            "action_family": "visual_effect",
            "action_name": action_name,
            "source_entity": {"type": "effect_surface", "id": oid},
            "target_entity": {"type": "effect_surface", "id": oid},
            "polyline_2d": contour_pts if contour_pts else [centroid],
            "effect_center_uv": centroid,
            "effect_bbox_px": bbox,
            "contour_sample_px": contour_pts,
            "oscillation_radius_px": max(2.0, min(12.0, (bbox[2] if len(bbox) > 2 else 20) * 0.04)),
            "frequency_hz": 1.4,
            "alpha_range": [0.2, 0.8],
            "depth_m": _obj_depth_m(obj),
            "label": str(obj.get("label", "")),
            "scores": {
                "overall_confidence": round(float(score), 4),
                "geometric_feasibility": 0.75,
                "semantic_confidence": round(float(score), 4),
            },
        }
    except Exception:
        return None


def _try_contact_patch(
    obj: Dict[str, Any],
    mask_aff: Dict[str, Any],
    obj_aff: Dict[str, Any],
    oid: str,
    h: int,
    w: int,
    metric_depth: Optional[np.ndarray],
    score: float,
    action_name: str,
) -> Optional[Dict[str, Any]]:
    """Contact-patch manifold: actor approach + touch/hold/interact with object."""
    try:
        import cv2 as _cv2
        centroid = _mask_centroid(obj, h, w)
        anchors = dict(obj_aff.get("anchors") or {})
        contact_pts = list(anchors.get("contact_points") or [])
        approach_pts = list(anchors.get("approach_points") or [])
        if not contact_pts:
            contact_pts = [centroid]
        if not approach_pts:
            # derive approach point by stepping off centroid toward image centre
            cx = float(centroid[0])
            cy = float(centroid[1])
            dx = (w * 0.5 - cx)
            dy = (h * 0.5 - cy)
            mag = math.hypot(dx, dy) + 1e-9
            off = min(30.0, max(12.0, float(h + w) * 0.04))
            approach_pts = [[round(cx + dx / mag * off, 3), round(cy + dy / mag * off, 3)]]
        depth_m = _obj_depth_m(obj)
        if metric_depth is not None:
            xi = max(0, min(w - 1, int(centroid[0])))
            yi = max(0, min(h - 1, int(centroid[1])))
            dv = float(metric_depth[yi, xi])
            if dv > 0.0:
                depth_m = dv
        return {
            "path_id": f"manifold_contact_{oid}",
            "path_level": "object",
            "path_type": "contact_patch",
            "manifold_type": "contact_patch",
            "action_family": "contact_interaction",
            "action_name": action_name,
            "source_entity": {"type": "candidate_actor", "id": "actor"},
            "target_entity": {"type": "interaction_target", "id": oid},
            "polyline_2d": approach_pts[:1] + contact_pts[:1],
            "contact_points": contact_pts[:4],
            "approach_points": approach_pts[:2],
            "depth_m": depth_m,
            "label": str(obj.get("label", "")),
            "scores": {
                "overall_confidence": round(float(score), 4),
                "geometric_feasibility": 0.68,
                "semantic_confidence": round(float(score), 4),
            },
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Region-level FMM fallback — fires when no object-pair ribbon paths succeeded
# ─────────────────────────────────────────────────────────────────────────────

def _region_level_fmm_paths(
    ctx: "PipelineContext",
    lm: np.ndarray,
    feasible: np.ndarray,
    speed_map: np.ndarray,
    k: int,
    pen: float,
    top_k: int,
    paths_root: "Path",
    w: int,
    h: int,
    *,
    max_hypotheses: int = 20,
    max_labels_sampled: int = 8,
) -> List[Dict[str, Any]]:
    """Ribbon paths between region centres (budgeted FMM on ``feasible``)."""
    from ..pathing.semantic_fmm import k_diverse_from_T, time_of_arrival_from_speed
    from ..pathing.walkable_mask import snap_uv_to_walkable

    hypotheses: List[Dict[str, Any]] = []
    unique_labels = [int(v) for v in np.unique(lm) if v > 0]
    if len(unique_labels) < 2:
        return hypotheses

    region_centres: List[Tuple[int, int, int]] = []  # (label, cx, cy)
    for label in unique_labels:
        ys, xs = np.where(lm == label)
        if xs.size == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        gp = snap_uv_to_walkable(cx, cy, feasible, w, h)
        region_centres.append((label, gp[0], gp[1]))

    cap = max(2, int(max_labels_sampled))
    step = max(1, len(region_centres) // cap)
    sampled = region_centres[::step][:cap]
    if len(sampled) < 2:
        return hypotheses

    sm = np.where(feasible, speed_map, speed_map * 0.02)
    goal_T: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    for _, cx, cy in sampled:
        gp = (cx, cy)
        if gp not in goal_T:
            goal_T[gp] = time_of_arrival_from_speed(sm, gp)

    done_pairs: set = set()
    for i, (lbl_src, sx, sy) in enumerate(sampled):
        for lbl_tgt, gx, gy in sampled[i + 1:]:
            sp, gp = (sx, sy), (gx, gy)
            if sp == gp:
                continue
            pair_key = tuple(sorted([sp, gp]))
            if pair_key in done_pairs:
                continue
            done_pairs.add(pair_key)
            T = goal_T.get(gp)
            if T is None:
                continue
            try:
                gpaths = k_diverse_from_T(T, sp, k=min(k, top_k), edge_penalty=pen)
                for kidx, gpath in enumerate(gpaths, start=1):
                    if len(gpath) < 2:
                        continue
                    raw_poly = [list(p) for p in gpath]
                    smooth_poly = _bezier_smooth_polyline(raw_poly, w, h, feasible)
                    hypotheses.append({
                        "path_id": f"region_fmm_{lbl_src}_to_{lbl_tgt}_k{kidx:02d}",
                        "path_level": "region",
                        "path_type": "region_fmm",
                        "manifold_type": "ribbon_path",
                        "action_family": "locomotion",
                        "source_entity": {"type": "region", "id": f"region_{lbl_src}", "start_uv": list(sp)},
                        "target_entity": {"type": "region", "id": f"region_{lbl_tgt}", "goal_uv": list(gp)},
                        "polyline_2d_raw": raw_poly,
                        "polyline_2d": smooth_poly,
                        "scores": {"overall_confidence": 1.0 / kidx},
                    })
                    if len(hypotheses) >= int(max_hypotheses):
                        return hypotheses
            except Exception:
                continue
    return hypotheses


def _region_bottom_centroid(lm: np.ndarray, label_idx: int, w: int, h: int) -> Tuple[int, int]:
    ys, xs = np.where(np.asarray(lm, dtype=np.int32) == int(label_idx))
    if xs.size == 0:
        return w // 2, h // 2
    row_max = int(ys.max())
    row_mask = ys == row_max
    bx = xs[row_mask]
    return int(np.mean(bx)), row_max


def _relation_boost_region_ids(obj_id: str, relations: List[Dict[str, Any]], region_ids: set) -> set:
    out: set = set()
    hints = {"on", "near", "beside", "inside", "above", "below", "touching", "adjacent_to"}
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        pred = str(rel.get("pred") or rel.get("predicate") or "").lower()
        sid = str(rel.get("sub_id") or rel.get("subject_id") or "")
        oid = str(rel.get("obj_id") or rel.get("object_id") or "")
        if pred not in hints:
            continue
        if sid == obj_id and oid in region_ids:
            out.add(oid)
        if oid == obj_id and sid in region_ids:
            out.add(sid)
    return out


def _append_object_to_region_paths(
    ctx: PipelineContext,
    cfg: Any,
    objects: List[Dict[str, Any]],
    obj_goals: Dict[str, Tuple[int, int]],
    lm: np.ndarray,
    feasible: np.ndarray,
    speed_map: np.ndarray,
    k: int,
    pen: float,
    top_k: int,
    hypotheses: List[Dict[str, Any]],
    w: int,
    h: int,
    feas_variant: str,
) -> None:
    from ..pathing.semantic_fmm import k_diverse_from_T, time_of_arrival_from_speed
    from ..pathing.walkable_mask import snap_uv_to_walkable

    top_k_goals = int(getattr(cfg, "path_object_region_goal_top_k", 6)) if cfg else 6
    max_cand = int(getattr(cfg, "path_max_candidates", 500)) if cfg else 500
    max_obj_paths = int(getattr(cfg, "path_max_object_paths", 220)) if cfg else 220
    obj_path_count = sum(1 for h in hypotheses if str(h.get("path_level")) == "object")
    if obj_path_count >= max_obj_paths:
        return

    regions_meta = list(ctx.region_partition_meta or [])
    region_ids = {str(r.get("id", "")) for r in regions_meta if str(r.get("id", "")).strip()}
    relations = list(ctx.relations or [])

    for o in objects:
        oid = str(o.get("id", ""))
        if not oid:
            continue
        sp = obj_goals.get(oid)
        if sp is None:
            continue
        rel_boost = _relation_boost_region_ids(oid, relations, region_ids)
        scored: List[Tuple[float, str, Tuple[int, int], str]] = []
        for r in regions_meta:
            rid = str(r.get("id", ""))
            if not rid:
                continue
            ridx = int(r.get("region_index", 0) or 0)
            if ridx <= 0:
                continue
            cx, cy = _region_bottom_centroid(lm, ridx, w, h)
            gp = snap_uv_to_walkable(cx, cy, feasible, w, h)
            sem = str(r.get("semantic_label", "") or r.get("type", "")).lower()
            score = 0.1
            if rid in rel_boost:
                score += 1.0
            for tok in ("floor", "ground", "path", "road", "stair", "step", "deck", "wood", "platform"):
                if tok in sem:
                    score += 0.25
            # Prefer goals below the actor foot (encourage climb-from-below narratives).
            if sp[1] > gp[1] + 8:
                score += 0.15
            scored.append((score, rid, gp, sem))

        scored.sort(key=lambda t: t[0], reverse=True)
        goal_T_local: Dict[Tuple[int, int], Any] = {}
        sm_arr = np.where(feasible, speed_map, speed_map * 0.02)
        for score, rid, gp, _sem in scored[: max(1, top_k_goals)]:
            if len(hypotheses) >= max_cand:
                return
            if gp not in goal_T_local:
                goal_T_local[gp] = time_of_arrival_from_speed(sm_arr, gp)
            T = goal_T_local.get(gp)
            if T is None:
                continue
            try:
                gpaths = k_diverse_from_T(T, sp, k=min(k, top_k), edge_penalty=pen)
                for kidx, gpath in enumerate(gpaths, start=1):
                    if len(gpath) < 2:
                        continue
                    raw_poly = [list(p) for p in gpath]
                    smooth_poly = _bezier_smooth_polyline(raw_poly, w, h, feasible)
                    hypotheses.append({
                        "path_id": f"staged_objreg_{oid}_to_{rid}_k{kidx:02d}",
                        "path_level": "object_region",
                        "path_type": "object_region_fmm",
                        "manifold_type": "ribbon_path",
                        "action_family": "locomotion",
                        "source_entity": {"type": "object", "id": oid, "start_uv": list(sp)},
                        "target_entity": {"type": "region", "id": rid, "goal_uv": list(gp)},
                        "polyline_2d_raw": raw_poly,
                        "polyline_2d": smooth_poly,
                        "scores": {"overall_confidence": min(0.95, 0.35 + 0.12 * score) / float(kidx)},
                        "goal_generation": {
                            "sources": ["object_anchor", "region_bottom_centroid"],
                            "relation_boost_regions": sorted(rel_boost),
                        },
                        "routing_meta": {"feasible_variant": feas_variant},
                    })
            except Exception:
                continue
