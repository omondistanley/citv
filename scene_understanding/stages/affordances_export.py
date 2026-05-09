"""Stage: caption-aware scene/object/mask affordance exports.

This stage implements the CPU-first affordance layer from
``docs/path_updates.md``. It does not run new vision models. Instead, it
grounds open-vocabulary action prompts against already available evidence:
object labels, Florence/RAM++ captions and tags, relation text, region
summaries, mask geometry, metric depth, and scene graph metadata.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..action_ontology import load_action_ontology, number, prompt_bank
from ..evidence import normalize_relation
from ..pipeline_context import PipelineContext


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "its", "near", "of", "on", "or", "that", "the",
    "this", "to", "with", "within",
}


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run(pipeline: Any, ctx: PipelineContext) -> PipelineContext:
    """Export caption evidence and scene/object/mask affordance contracts."""
    cfg = getattr(pipeline, "config", None)
    enabled = bool(getattr(cfg, "export_affordance_hypotheses", True)) if cfg else True
    if not enabled:
        return ctx

    objects: List[Dict[str, Any]] = list(ctx.extra.get("objects", []))
    staged_dir = ctx.output_dir / "scene_graph" / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    stem = ctx.stem

    # Phase 2.1/2.2: build support mask BEFORE per-object anchor synthesis so
    # foot/ground-contact anchors can snap onto a real surface. The mask is
    # stashed on ctx.extra so paths_export reuses it without recomputing.
    _ensure_support_mask(ctx, objects)

    ontology = load_action_ontology(cfg)
    action_prompts = _prompt_config(cfg, "path_action_prompts", prompt_bank(ontology, "action_prompts"))
    role_prompts = _prompt_config(cfg, "path_affordance_role_prompts", prompt_bank(ontology, "role_prompts"))

    caption_evidence = _build_caption_evidence(ctx, objects, ontology)
    object_affordances = _build_object_affordances(
        ctx,
        objects,
        caption_evidence,
        action_prompts,
        role_prompts,
        ontology,
    )
    mask_affordances = _build_mask_affordances(ctx, objects, object_affordances, ontology)
    scene_affordances = _build_scene_affordances(
        ctx,
        caption_evidence,
        object_affordances,
        mask_affordances,
        action_prompts,
        role_prompts,
        ontology,
    )

    _write_json(caption_evidence, staged_dir / f"{stem}_caption_evidence.json")
    _write_json(scene_affordances, staged_dir / f"{stem}_scene_affordances.json")
    _write_json(object_affordances, staged_dir / f"{stem}_object_affordances.json")
    _write_json(mask_affordances, staged_dir / f"{stem}_mask_affordances.json")

    ctx.caption_evidence = caption_evidence
    ctx.scene_affordances = scene_affordances
    ctx.object_affordances = object_affordances
    ctx.mask_affordances = mask_affordances
    ctx.extra["caption_evidence"] = caption_evidence
    ctx.extra["caption_lookup"] = _caption_lookup(caption_evidence)
    ctx.extra["affordances"] = {
        "scene": scene_affordances,
        "object": object_affordances,
        "mask": mask_affordances,
    }
    ctx.path_exports["caption_evidence_json"] = f"scene_graph/staged/{stem}_caption_evidence.json"
    ctx.path_exports["scene_affordances_json"] = f"scene_graph/staged/{stem}_scene_affordances.json"
    ctx.path_exports["object_affordances_json"] = f"scene_graph/staged/{stem}_object_affordances.json"
    ctx.path_exports["mask_affordances_json"] = f"scene_graph/staged/{stem}_mask_affordances.json"

    print(
        f"  [AffordancesExport] {stem}: "
        f"{len(object_affordances.get('objects', []))} object, "
        f"{len(mask_affordances.get('masks', []))} mask, "
        f"{len(scene_affordances.get('affordances', []))} scene affordances"
    )
    return ctx


def _prompt_config(cfg: Any, attr: str, default: Dict[str, List[str]]) -> Dict[str, List[str]]:
    raw = getattr(cfg, attr, None) if cfg is not None else None
    if isinstance(raw, dict):
        out: Dict[str, List[str]] = {}
        for k, v in raw.items():
            if isinstance(v, str):
                out[str(k)] = [v]
            elif isinstance(v, Iterable):
                out[str(k)] = [str(x) for x in v if str(x).strip()]
        if out:
            return out
    return {k: list(v) for k, v in default.items()}


def _tokenize(*parts: Any) -> List[str]:
    text = " ".join(str(p) for p in parts if p is not None).lower()
    return _TOKEN_RE.findall(text)


def _text_blob(*parts: Any) -> str:
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


def _score_prompt_bank(
    text: str,
    prompt_bank: Dict[str, List[str]],
    *,
    hit_lambda: float = 0.55,
) -> Dict[str, Dict[str, Any]]:
    tokens = set(_tokenize(text))
    raw = f" {text.lower()} "
    scores: Dict[str, Dict[str, Any]] = {}
    for name, prompts in prompt_bank.items():
        hits: List[str] = []
        soft_hits: List[str] = []
        soft_score = 0.0
        for phrase in prompts:
            p = str(phrase).lower().strip()
            if not p:
                continue
            if " " in p:
                if f" {p} " in raw or p in raw:
                    hits.append(p)
                else:
                    p_tokens = set(_tokenize(p))
                    if p_tokens:
                        overlap = len(tokens.intersection(p_tokens)) / max(1, len(p_tokens))
                        if overlap >= 0.60:
                            soft_hits.append(p)
                            soft_score += 0.35 * overlap
            elif p in tokens:
                hits.append(p)
            elif p:
                # CPU-first open-vocab soft grounding: seed prompts remain
                # priors, but partial lexical overlap can still contribute
                # evidence instead of forcing a closed exact-match vocabulary.
                p_tokens = set(_tokenize(p))
                if p_tokens:
                    overlap = len(tokens.intersection(p_tokens)) / max(1, len(p_tokens))
                    if overlap >= 0.60:
                        soft_hits.append(p)
                        soft_score += 0.25 * overlap
        unique_hits = sorted(set(hits))
        unique_soft = sorted(set(soft_hits) - set(unique_hits))
        score = 1.0 - math.exp(-float(hit_lambda) * len(unique_hits)) if unique_hits else 0.0
        if unique_soft:
            score = max(score, min(0.72, soft_score))
        scores[name] = {
            "score": round(float(min(1.0, score)), 4),
            "evidence_terms": unique_hits[:12],
            "soft_evidence_terms": unique_soft[:12],
            "grounding_method": "exact_phrase_plus_token_overlap",
        }
    return scores


def _top_scores(scores: Dict[str, Dict[str, Any]], limit: int = 8, min_score: float = 0.01) -> List[Dict[str, Any]]:
    rows = [
        {"name": name, **dict(data)}
        for name, data in scores.items()
        if float(data.get("score", 0.0)) >= min_score
    ]
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return rows[:limit]


def _float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _int_bbox(obj: Dict[str, Any]) -> List[int]:
    vals = list(obj.get("bbox") or [0, 0, 0, 0])[:4]
    vals = vals + [0] * (4 - len(vals))
    x1, y1, a, b = [int(round(_float(v))) for v in vals]
    # Staged objects historically used xyxy while path/affordance helpers expect
    # xywh. Accept both to avoid corrupting anchors and areas.
    if a > x1 and b > y1:
        return [x1, y1, max(1, a - x1), max(1, b - y1)]
    return [x1, y1, max(1, a), max(1, b)]


def _centroid(obj: Dict[str, Any], bbox: Optional[List[int]] = None) -> List[float]:
    uv = obj.get("mask_centroid_2d") or obj.get("centroid_2d") or []
    if isinstance(uv, Sequence) and len(uv) >= 2:
        return [_float(uv[0]), _float(uv[1])]
    x, y, bw, bh = bbox or _int_bbox(obj)
    return [float(x + bw * 0.5), float(y + bh * 0.5)]


def _clamp_uv(uv: Sequence[float], width: int, height: int) -> List[float]:
    if width <= 0 or height <= 0:
        return [_float(uv[0]), _float(uv[1])]
    return [
        round(max(0.0, min(float(width - 1), _float(uv[0]))), 3),
        round(max(0.0, min(float(height - 1), _float(uv[1]))), 3),
    ]


def _object_text(obj: Dict[str, Any], caption_row: Optional[Dict[str, Any]]) -> str:
    src = obj.get("sources") or {}
    f2 = src.get("Florence2") or {}
    ram = src.get("RAM++") or {}
    gdino = src.get("GroundedSAM2") or {}
    parts = [
        obj.get("label"),
        obj.get("canonical_label"),
        obj.get("caption"),
        obj.get("label_warning"),
        " ".join(str(c.get("label", "")) for c in obj.get("label_candidates", []) or [] if isinstance(c, dict)),
        " ".join(str(x) for x in obj.get("visual_quality_attributes", []) or []),
        gdino.get("label"),
        gdino.get("caption"),
        f2.get("label"),
        f2.get("caption"),
        ram.get("label"),
        ram.get("caption"),
        " ".join(str(t) for t in ram.get("tags", []) or []),
    ]
    if caption_row:
        parts.extend([
            caption_row.get("label"),
            caption_row.get("canonical_label"),
            caption_row.get("gdino_label"),
            caption_row.get("florence_label"),
            caption_row.get("florence_caption"),
            caption_row.get("rampp_label"),
            " ".join(str(t) for t in caption_row.get("rampp_tags", []) or []),
            caption_row.get("label_warning"),
        ])
    return _text_blob(*parts)


def _build_caption_evidence(
    ctx: PipelineContext,
    objects: List[Dict[str, Any]],
    ontology: Dict[str, Any],
) -> Dict[str, Any]:
    tiers = dict(ctx.extra.get("caption_tiers") or {})
    bundle = dict(ctx.extra.get("caption_bundle") or {})
    records: List[Dict[str, Any]] = []

    tier1 = dict(tiers.get("global_scene") or {})
    global_text = str(tier1.get("caption") or bundle.get("global_caption") or "").strip()
    if global_text:
        records.append({
            "entity_type": "scene",
            "entity_id": ctx.stem,
            "source": "caption_global",
            "text": global_text,
            "labels": [],
            "tags": [],
            "confidence": (
                number(ontology, "scoring", "global_generated_confidence", 0.72)
                if "generated" in str(tier1.get("status", ""))
                else number(ontology, "scoring", "global_fallback_confidence", 0.45)
            ),
            "precision": number(ontology, "scoring", "global_precision", 0.55),
            "uncertainty": [],
        })

    object_rows = list((tiers.get("per_object") or {}).get("objects") or [])
    obj_row_by_id = {str(r.get("object_id", "")): r for r in object_rows if isinstance(r, dict)}
    for obj in objects:
        oid = str(obj.get("id", "") or "")
        row = obj_row_by_id.get(oid, {})
        text = _object_text(obj, row)
        labels = [
            str(x).strip()
            for x in [
                obj.get("label"),
                obj.get("canonical_label"),
                row.get("label"),
                row.get("canonical_label"),
                row.get("gdino_label"),
                row.get("florence_label"),
                row.get("rampp_label"),
            ]
            if str(x or "").strip()
        ]
        tags = [str(t) for t in row.get("rampp_tags", []) or [] if str(t).strip()]
        conf = max(
            _float(obj.get("conf"), 0.0),
            _float(row.get("gdino_conf"), 0.0),
            number(ontology, "scoring", "object_text_floor_confidence", 0.45) if text else 0.0,
        )
        uncertainty = []
        if str(row.get("label_warning") or obj.get("label_warning") or "").strip():
            uncertainty.append("label_warning")
        ds = obj.get("depth_stats") or {}
        if bool(ds.get("possibly_transparent")):
            uncertainty.append("possibly_transparent_depth")
        records.append({
            "entity_type": "object",
            "entity_id": oid,
            "source": "caption_object_and_labels",
            "text": text,
            "labels": labels,
            "tags": tags,
            "confidence": round(float(min(1.0, conf)), 4),
            "precision": round(_caption_precision(text, uncertainty, ontology), 4),
            "uncertainty": uncertainty,
            "region_id": str(obj.get("region_id", "")),
        })

    for row in list((tiers.get("per_region") or {}).get("regions") or []):
        if not isinstance(row, dict):
            continue
        rid = str(row.get("region_id", "") or "")
        narrative = str(row.get("narrative", "") or "")
        labels = [str(x) for x in row.get("object_labels", []) or [] if str(x).strip()]
        text = _text_blob(row.get("region_type"), " ".join(labels), narrative)
        records.append({
            "entity_type": "region",
            "entity_id": rid,
            "source": "caption_region_summary",
            "text": text,
            "labels": labels,
            "tags": [],
            "confidence": number(ontology, "scoring", "region_confidence", 0.58) if text else 0.0,
            "precision": number(ontology, "scoring", "region_precision", 0.48),
            "uncertainty": [],
            "depth_mean_m": _float(row.get("depth_mean_m"), 0.0),
        })

    for idx, row in enumerate(list((tiers.get("cross_region") or {}).get("interactions") or [])):
        if not isinstance(row, dict):
            continue
        text = _text_blob(
            row.get("subject_label"),
            row.get("predicate"),
            row.get("object_label"),
            row.get("subject_region"),
            row.get("object_region"),
        )
        records.append({
            "entity_type": "relation",
            "entity_id": f"caption_relation_{idx:04d}",
            "source": "caption_cross_region",
            "text": text,
            "labels": [str(row.get("subject_label", "")), str(row.get("object_label", ""))],
            "tags": [str(row.get("predicate", ""))],
            "confidence": _float(row.get("score"), number(ontology, "scoring", "relation_default_confidence", 0.5)),
            "precision": number(ontology, "scoring", "relation_precision", 0.50),
            "uncertainty": [],
            "subject_id": str(row.get("subject_id", "")),
            "object_id": str(row.get("object_id", "")),
        })

    for row in list((tiers.get("uncertainty") or {}).get("notes") or []):
        if not isinstance(row, dict):
            continue
        oid = str(row.get("object_id", "") or "")
        records.append({
            "entity_type": "uncertainty",
            "entity_id": oid,
            "source": "caption_uncertainty",
            "text": _text_blob(row.get("label"), " ".join(str(x) for x in row.get("issues", []) or [])),
            "labels": [str(row.get("label", ""))],
            "tags": [str(x) for x in row.get("issues", []) or []],
            "confidence": number(ontology, "scoring", "uncertainty_confidence", 0.65),
            "precision": number(ontology, "scoring", "uncertainty_precision", 0.35),
            "uncertainty": [str(x) for x in row.get("issues", []) or []],
        })

    open_vocab_concepts = _extract_open_vocab_concepts(records, ontology)
    return {
        "schema": "citv_caption_evidence_v1",
        "version": "1.0",
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
        "image": str(ctx.image_path),
        "source_artifacts": dict((bundle.get("tiers") or {})),
        "dense_captioning_note": (
            "Caption tiers are first-class semantic evidence for affordances, paths, "
            "actions, trajectories, and animation contracts."
        ),
        "records": records,
        "open_vocab_concepts": open_vocab_concepts,
        "open_vocab_contract": {
            "seed_prompts_are_priors": True,
            "closed_vocabulary": False,
            "concept_source": "captions_labels_tags_relations_regions_uncertainty",
            "cpu_first": True,
        },
        "summary": {
            "record_count": len(records),
            "scene_caption_available": bool(global_text),
            "object_caption_count": sum(1 for r in records if r["entity_type"] == "object"),
            "region_caption_count": sum(1 for r in records if r["entity_type"] == "region"),
            "relation_caption_count": sum(1 for r in records if r["entity_type"] == "relation"),
            "open_vocab_concept_count": len(open_vocab_concepts),
        },
    }


def _label_quality_terms(ontology: Dict[str, Any]) -> Tuple[set, set, set]:
    lq = ontology.get("label_quality") if isinstance(ontology.get("label_quality"), dict) else {}
    generic = {str(x).strip().lower() for x in list(lq.get("generic_labels") or []) if str(x).strip()}
    meta_terms = {str(x).strip().lower() for x in list(lq.get("meta_visual_terms") or []) if str(x).strip()}
    meta_phrases = {str(x).strip().lower() for x in list(lq.get("meta_visual_phrases") or []) if str(x).strip()}
    return generic, meta_terms, meta_phrases


def _extract_open_vocab_concepts(
    records: Sequence[Dict[str, Any]],
    ontology: Dict[str, Any],
    *,
    max_terms: int = 96,
) -> List[Dict[str, Any]]:
    """Extract dynamic scene concepts from available evidence without a fixed label list."""
    generic, meta_terms, meta_phrases = _label_quality_terms(ontology)
    blocked = generic | meta_terms | set(_STOPWORDS)
    weights: Dict[str, float] = {}
    etypes: Dict[str, set] = {}
    examples: Dict[str, List[str]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        text = _text_blob(rec.get("text"), " ".join(str(x) for x in rec.get("labels") or []), " ".join(str(x) for x in rec.get("tags") or []))
        toks = [t for t in _tokenize(text) if t not in blocked and len(t) >= 3]
        if not toks:
            continue
        conf = _float(rec.get("confidence"), 0.35)
        precision = _float(rec.get("precision"), 0.35)
        rec_w = max(0.05, min(1.0, conf * precision))
        phrases: List[str] = []
        phrases.extend(toks)
        for n in (2, 3):
            for i in range(0, max(0, len(toks) - n + 1)):
                phr = " ".join(toks[i:i + n])
                if phr in meta_phrases:
                    continue
                phrases.append(phr)
        for term in phrases:
            if term in blocked or term in meta_phrases:
                continue
            weights[term] = weights.get(term, 0.0) + rec_w
            etypes.setdefault(term, set()).add(str(rec.get("entity_type", "")))
            if len(examples.setdefault(term, [])) < 3:
                examples[term].append(str(rec.get("entity_id", "")))
    rows = [
        {
            "term": term,
            "score": round(float(min(1.0, score)), 4),
            "entity_types": sorted(x for x in etypes.get(term, set()) if x),
            "example_entity_ids": [x for x in examples.get(term, []) if x],
        }
        for term, score in weights.items()
        if score >= 0.08
    ]
    rows.sort(key=lambda r: (float(r.get("score", 0.0)), len(str(r.get("term", "")))), reverse=True)
    return rows[:max_terms]


def _caption_precision(text: str, uncertainty: Sequence[str], ontology: Dict[str, Any]) -> float:
    n_tokens = len(_tokenize(text))
    precision = number(ontology, "scoring", "caption_precision_base", 0.28) + min(
        number(ontology, "scoring", "caption_precision_token_bonus_cap", 0.42),
        number(ontology, "scoring", "caption_precision_token_bonus", 0.025) * n_tokens,
    )
    precision -= number(ontology, "scoring", "caption_precision_uncertainty_penalty", 0.12) * len([u for u in uncertainty if u])
    return max(
        number(ontology, "scoring", "caption_precision_min", 0.05),
        min(number(ontology, "scoring", "caption_precision_max", 0.95), precision),
    )


def _caption_lookup(caption_evidence: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rec in caption_evidence.get("records") or []:
        if not isinstance(rec, dict):
            continue
        et = str(rec.get("entity_type", ""))
        eid = str(rec.get("entity_id", ""))
        if not et or not eid:
            continue
        lookup.setdefault(et, {})[eid] = rec
    return lookup


def _build_object_affordances(
    ctx: PipelineContext,
    objects: List[Dict[str, Any]],
    caption_evidence: Dict[str, Any],
    action_prompts: Dict[str, List[str]],
    role_prompts: Dict[str, List[str]],
    ontology: Dict[str, Any],
) -> Dict[str, Any]:
    lookup = _caption_lookup(caption_evidence).get("object", {})
    relation_text_by_object = _relation_text_by_object(ctx.relations, objects)
    hit_lambda = number(ontology, "scoring", "prompt_hit_lambda", 0.55)
    rows: List[Dict[str, Any]] = []
    for obj in objects:
        oid = str(obj.get("id", "") or "")
        caption_rec = lookup.get(oid, {})
        full_text = _text_blob(caption_rec.get("text"), relation_text_by_object.get(oid, ""))
        action_scores = _score_prompt_bank(full_text, action_prompts, hit_lambda=hit_lambda)
        role_scores = _score_prompt_bank(full_text, role_prompts, hit_lambda=hit_lambda)
        bbox = _int_bbox(obj)
        anchors = _anchors_for_object(
            obj,
            bbox,
            ctx.width,
            ctx.height,
            ontology,
            support_mask=ctx.extra.get("support_mask"),
        )
        geom = _geometry_for_object(obj, bbox)
        depth_m = _float((obj.get("depth_stats") or {}).get("z_val"), _float((obj.get("coordinates_3d") or {}).get("z"), 0.0))
        precision = _float(caption_rec.get("precision"), 0.35)
        label_quality, label_quality_reasons = _label_quality_factor(
            obj,
            caption_rec,
            ontology,
            return_reasons=True,
        )
        uncertainty_terms = list(caption_rec.get("uncertainty") or [])
        if label_quality < 0.75:
            uncertainty_terms.append("label_noise_risk")
        for rec in action_scores.values():
            rec["score"] = round(
                float(max(0.0, min(1.0, _float(rec.get("score"), 0.0) * label_quality))),
                4,
            )
        for rec in role_scores.values():
            rec["score"] = round(
                float(max(0.0, min(1.0, _float(rec.get("score"), 0.0) * label_quality))),
                4,
            )
        rows.append({
            "object_id": oid,
            "label": str(obj.get("label", "object")),
            "canonical_label": str(obj.get("canonical_label", obj.get("label", "object"))),
            "label_candidates": list(obj.get("label_candidates", [])),
            "rejected_labels": list(obj.get("rejected_labels", [])),
            "visual_quality_attributes": list(obj.get("visual_quality_attributes", [])),
            "caption": str(caption_rec.get("text", obj.get("caption", ""))),
            "region_id": str(obj.get("region_id", "")),
            "region_index": int(obj.get("region_index", 0) or 0),
            "depth_m": round(depth_m, 4),
            "geometry": geom,
            "anchors": anchors,
            "roles": _top_scores(role_scores, limit=10),
            "actions": _top_scores(action_scores, limit=12),
            "score_sources": {
                "caption_precision": round(precision, 4),
                "caption_confidence": _float(caption_rec.get("confidence"), 0.0),
                "relation_text_available": bool(relation_text_by_object.get(oid)),
                "mask_available": bool(obj.get("_sam2_mask_array") is not None),
                "depth_available": depth_m > 0.0,
                "label_quality_factor": round(label_quality, 4),
                "label_quality_reasons": label_quality_reasons,
                "uncertainty_terms": uncertainty_terms,
            },
            "open_vocab_grounding": {
                "closed_vocabulary": False,
                "evidence_text_tokens": _tokenize(full_text)[:48],
                "top_dynamic_terms": _top_dynamic_terms_for_text(full_text, caption_evidence),
                "seed_prompts_are_priors": True,
            },
        })
    return {
        "schema": "citv_object_affordances_v1",
        "version": "1.0",
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
        "object_count": len(rows),
        "objects": rows,
        "scoring": {
            "method": "caption_label_relation_prompt_fusion_cpu",
            "prompt_configurable": True,
            "hard_label_behavior": False,
        },
    }


def _relation_text_by_object(relations: Sequence[Dict[str, Any]], objects: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    labels = {str(o.get("id", "")): str(o.get("canonical_label") or o.get("label", "")) for o in objects}
    out: Dict[str, List[str]] = {}
    for rel in relations:
        nrel = normalize_relation(rel)
        sid = str(nrel.get("subject_id", ""))
        oid = str(nrel.get("object_id", ""))
        pred = str(nrel.get("predicate", ""))
        txt = _text_blob(labels.get(sid, sid), pred, labels.get(oid, oid))
        if sid:
            out.setdefault(sid, []).append(txt)
        if oid:
            out.setdefault(oid, []).append(txt)
    return {k: " ".join(v[:16]) for k, v in out.items()}


def _top_dynamic_terms_for_text(text: str, caption_evidence: Dict[str, Any], *, limit: int = 12) -> List[Dict[str, Any]]:
    toks = set(_tokenize(text))
    rows: List[Dict[str, Any]] = []
    for concept in list(caption_evidence.get("open_vocab_concepts") or []):
        term = str(concept.get("term", ""))
        ctoks = set(_tokenize(term))
        if not ctoks:
            continue
        overlap = len(toks.intersection(ctoks)) / max(1, len(ctoks))
        if overlap <= 0.0:
            continue
        rows.append({
            "term": term,
            "score": round(float(_float(concept.get("score"), 0.0) * overlap), 4),
            "overlap": round(float(overlap), 4),
        })
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return rows[:limit]


def _label_quality_factor(
    obj: Dict[str, Any],
    caption_rec: Dict[str, Any],
    ontology: Optional[Dict[str, Any]] = None,
    *,
    return_reasons: bool = False,
) -> Any:
    ontology = ontology or load_action_ontology()
    labels: List[str] = []
    for raw in (
        obj.get("label"),
        obj.get("canonical_label"),
        *(caption_rec.get("labels") or []),
    ):
        tok = str(raw or "").strip().lower()
        if tok:
            labels.append(tok)
    if not labels:
        score_reasons = (0.70, ["missing_label_text"])
        return score_reasons if return_reasons else score_reasons[0]
    generic, meta_terms, meta_phrases = _label_quality_terms(ontology)
    reasons: List[str] = []
    penalty = 0.0
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        toks = set(_tokenize(lbl))
        if lbl in generic or toks and toks.issubset(generic):
            penalty += 0.18
            reasons.append(f"generic_label:{lbl}")
        if lbl in meta_phrases or (toks and len(toks.intersection(meta_terms)) >= max(1, len(toks) // 2)):
            penalty += 0.20
            reasons.append(f"visual_meta_label:{lbl}")
        if len(toks) == 1 and next(iter(toks), "") in {"thin", "close", "set", "view"}:
            penalty += 0.10
            reasons.append(f"low_specificity_label:{lbl}")
    # Agreement is evidence, but disagreement among multiple sources should not
    # hard-fail. It just lowers confidence so path ranking cannot over-trust it.
    normalized = [" ".join(_tokenize(lbl)) for lbl in labels if _tokenize(lbl)]
    if len(set(normalized)) > 2:
        penalty += 0.08
        reasons.append("multi_source_label_disagreement")
    warning = str(caption_rec.get("label_warning") or obj.get("label_warning") or "").strip()
    if warning:
        penalty += 0.10
        reasons.append("label_warning")
    score = max(0.35, min(1.0, 1.0 - min(0.65, penalty)))
    if not reasons:
        reasons.append("specific_or_supported_label")
    score_reasons = (score, sorted(set(reasons))[:8])
    return score_reasons if return_reasons else score_reasons[0]


def _anchors_for_object(
    obj: Dict[str, Any],
    bbox: List[int],
    width: int,
    height: int,
    ontology: Dict[str, Any],
    *,
    support_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Anchor synthesis (Phase 2.2).

    Replaces the old purely-bbox derivation with mask-aware foot / support
    anchors. The schema is extended additively: existing keys remain so old
    consumers work, and new keys (``foot_uv``, ``support_contact_uv``,
    ``mask_top_uv``, ``mask_contour_sample_uv``, ``anchor_provenance``) are
    consumed by ``paths_export`` per plan §2.3.
    """
    x, y, bw, bh = bbox
    cx, cy = _centroid(obj, bbox)
    pad = max(
        number(ontology, "scoring", "object_anchor_pad_min_px", 4.0),
        min(
            max(float(bw), float(bh)) * number(ontology, "scoring", "object_anchor_pad_fraction", 0.25),
            number(ontology, "scoring", "object_anchor_pad_max_px", 32.0),
        ),
    )
    left = _clamp_uv([x - pad, cy], width, height)
    right = _clamp_uv([x + bw + pad, cy], width, height)
    top = _clamp_uv([cx, y - pad], width, height)
    bottom = _clamp_uv([cx, y + bh + pad], width, height)

    # ---- New: mask-derived foot / contact / contour points ----
    mask_arr = obj.get("_sam2_mask_array")
    foot_uv = list(bottom)  # bbox bottom default
    mask_top_uv = list(top)
    contour_samples: List[List[float]] = []
    provenance = {"foot_source": "bbox_bottom", "support_source": "bbox_bottom"}
    if mask_arr is not None:
        try:
            mm = np.asarray(mask_arr, dtype=bool)
            if mm.shape[:2] == (height, width) and mm.any():
                ys_m, xs_m = np.where(mm)
                # Foot = mask bottom-row centroid (lowest 5% of mask rows).
                row_max = int(ys_m.max())
                row_floor = int(max(ys_m.min(), row_max - max(1, int(round(0.05 * (row_max - ys_m.min() + 1))))))
                bottom_rows = mm[row_floor:row_max + 1, :]
                if bottom_rows.any():
                    by, bx = np.where(bottom_rows)
                    foot_x = float(np.mean(bx))
                    foot_y = float(row_floor + np.mean(by))
                    foot_uv = _clamp_uv([foot_x, foot_y], width, height)
                    provenance["foot_source"] = "mask_bottom_row_centroid"
                # Mask top centroid (lowest v).
                row_min = int(ys_m.min())
                top_band_rows = mm[row_min:max(row_min + 1, row_min + max(1, int(round(0.05 * (row_max - row_min + 1))))), :]
                if top_band_rows.any():
                    ty, tx = np.where(top_band_rows)
                    mask_top_uv = _clamp_uv([float(np.mean(tx)), float(row_min + np.mean(ty))], width, height)
                # Lightweight 8-point contour sample.
                try:
                    import cv2 as _cv2
                    contours, _ = _cv2.findContours(mm.astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_NONE)
                    if contours:
                        ct = max(contours, key=lambda c: c.shape[0])
                        n = ct.shape[0]
                        if n >= 8:
                            stride = max(1, n // 8)
                            contour_samples = [
                                _clamp_uv([float(ct[i, 0, 0]), float(ct[i, 0, 1])], width, height)
                                for i in range(0, n, stride)
                            ][:8]
                except Exception:
                    contour_samples = []
        except Exception:
            pass

    # Support contact = walk the foot down to the support_mask.
    support_contact_uv = list(foot_uv)
    if support_mask is not None:
        try:
            from ..pathing.ground_plane import snap_uv_down_to_support
            sx, sy = snap_uv_down_to_support(
                (int(round(foot_uv[0])), int(round(foot_uv[1]))), support_mask
            )
            support_contact_uv = _clamp_uv([float(sx), float(sy)], width, height)
            provenance["support_source"] = (
                "support_mask_snap_down" if support_contact_uv != foot_uv else "foot_already_on_support"
            )
        except Exception:
            pass
    elif provenance["foot_source"] == "mask_bottom_row_centroid":
        # No support mask → at least carry the foot_uv as the best guess.
        provenance["support_source"] = "mask_foot_fallback"

    # Approach points step left/right from the support contact along the
    # ground; contact_points keep mask-top (placement) + foot_uv (stand-on).
    actor_radius = float(max(8.0, min(40.0, 0.35 * max(bw, bh))))
    approach_points = [
        _clamp_uv([support_contact_uv[0] - actor_radius, support_contact_uv[1]], width, height),
        _clamp_uv([support_contact_uv[0] + actor_radius, support_contact_uv[1]], width, height),
        _clamp_uv([support_contact_uv[0], support_contact_uv[1] - actor_radius], width, height),
        _clamp_uv([support_contact_uv[0], support_contact_uv[1] + actor_radius], width, height),
    ]
    contact_points = [mask_top_uv, list(foot_uv), list(support_contact_uv)] + (
        contour_samples[:2] if contour_samples else []
    )
    occlusion_boundary_points = contour_samples or [left, right, top, bottom]

    support_patch_h = max(
        number(ontology, "scoring", "support_patch_min_px", 2.0),
        bh * number(ontology, "scoring", "support_patch_height_fraction", 0.18),
    )
    return {
        "center_uv": _clamp_uv([cx, cy], width, height),
        "approach_points": approach_points,
        "contact_points": contact_points,
        "occlusion_boundary_points": occlusion_boundary_points,
        "support_contact_patch": {
            "bbox_px": [int(x), int(y + max(0, bh - support_patch_h)), int(bw), int(support_patch_h)],
            "center_uv": list(support_contact_uv),
        },
        "entry_exit_points": [top, bottom, _clamp_uv([cx, cy], width, height)],
        # New foot / support / contour keys for plan §2.3.
        "foot_uv": list(foot_uv),
        "support_contact_uv": list(support_contact_uv),
        "mask_top_uv": list(mask_top_uv),
        "mask_contour_sample_uv": contour_samples,
        "anchor_provenance": provenance,
    }


def _geometry_for_object(obj: Dict[str, Any], bbox: List[int]) -> Dict[str, Any]:
    x, y, bw, bh = bbox
    area = int(max(0, bw) * max(0, bh))
    m = obj.get("_sam2_mask_array")
    mask_area = 0
    if m is not None:
        try:
            mask_area = int(np.asarray(m, dtype=bool).sum())
        except Exception:
            mask_area = 0
    return {
        "bbox_px": [int(x), int(y), int(bw), int(bh)],
        "bbox_area_px": area,
        "mask_area_px": mask_area,
        "aspect_ratio": round(float(bw) / max(1.0, float(bh)), 4),
    }


def _build_mask_affordances(
    ctx: PipelineContext,
    objects: List[Dict[str, Any]],
    object_affordances: Dict[str, Any],
    ontology: Dict[str, Any],
) -> Dict[str, Any]:
    obj_aff_by_id = {str(o.get("object_id", "")): o for o in object_affordances.get("objects") or []}
    rows: List[Dict[str, Any]] = []
    for obj in objects:
        oid = str(obj.get("id", "") or "")
        aff = obj_aff_by_id.get(oid, {})
        role_scores = {str(r.get("name", "")): _float(r.get("score"), 0.0) for r in aff.get("roles") or []}
        action_scores = {str(a.get("name", "")): _float(a.get("score"), 0.0) for a in aff.get("actions") or []}
        bbox = _int_bbox(obj)
        mask_geom = _mask_geometry(obj.get("_sam2_mask_array"), bbox, ctx.width, ctx.height)
        modes = _mask_path_modes(role_scores, action_scores, ontology)
        rows.append({
            "mask_id": oid,
            "object_id": oid,
            "label": str(obj.get("label", "object")),
            "path_modes": modes,
            "geometry": mask_geom,
            "depth_profile": _mask_depth_profile(obj.get("_sam2_mask_array"), ctx.metric_depth),
            "role_scores": role_scores,
            "action_scores": action_scores,
            "anchors": aff.get("anchors", {}),
        })
    return {
        "schema": "citv_mask_affordances_v1",
        "version": "1.0",
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
        "mask_count": len(rows),
        "masks": rows,
        "supported_path_modes": sorted((ontology.get("mask_path_mode_policy") or {}).keys()),
    }


def _mask_path_modes(
    role_scores: Dict[str, float],
    action_scores: Dict[str, float],
    ontology: Dict[str, Any],
) -> List[Dict[str, Any]]:
    policy = ontology.get("mask_path_mode_policy") or {}
    candidates: Dict[str, float] = {}
    if isinstance(policy, dict):
        for mode, rule in policy.items():
            if not isinstance(rule, dict):
                continue
            vals = [
                *(role_scores.get(str(role), 0.0) for role in rule.get("roles", []) or []),
                *(action_scores.get(str(action), 0.0) for action in rule.get("actions", []) or []),
            ]
            candidates[str(mode)] = max(vals) if vals else 0.0
    min_score = number(ontology, "scoring", "top_score_min_score", 0.01)
    rows = [{"mode": k, "score": round(float(v), 4)} for k, v in candidates.items() if v > min_score]
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows


def _mask_geometry(mask: Any, bbox: List[int], width: int, height: int) -> Dict[str, Any]:
    if mask is None:
        x, y, bw, bh = bbox
        cx, cy = x + bw * 0.5, y + bh * 0.5
        return {
            "bbox_px": [int(x), int(y), int(bw), int(bh)],
            "area_px": int(max(0, bw) * max(0, bh)),
            "centroid_uv": _clamp_uv([cx, cy], width, height),
            "contour_sample_px": [],
            "interior_seed_uv": _clamp_uv([cx, cy], width, height),
            "holes_estimated": 0,
        }
    mm = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mm)
    if xs.size == 0:
        return _mask_geometry(None, bbox, width, height)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    contour: List[List[float]] = []
    try:
        import cv2

        contours, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea).reshape(-1, 2)
            step = max(1, len(c) // 24)
            contour = [[float(p[0]), float(p[1])] for p in c[::step][:24]]
    except Exception:
        contour = []
    return {
        "bbox_px": [int(np.min(xs)), int(np.min(ys)), int(np.max(xs) - np.min(xs) + 1), int(np.max(ys) - np.min(ys) + 1)],
        "area_px": int(xs.size),
        "centroid_uv": _clamp_uv([cx, cy], width, height),
        "contour_sample_px": contour,
        "interior_seed_uv": _clamp_uv([cx, cy], width, height),
        "holes_estimated": 0,
    }


def _mask_depth_profile(mask: Any, depth: Optional[np.ndarray]) -> Dict[str, Any]:
    if mask is None or depth is None:
        return {"available": False}
    try:
        mm = np.asarray(mask, dtype=bool)
        dep = np.asarray(depth, dtype=np.float32)
        if mm.shape[:2] != dep.shape[:2]:
            import cv2

            mm = cv2.resize(mm.astype(np.uint8), (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        vals = dep[mm]
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size == 0:
            return {"available": False}
        return {
            "available": True,
            "mean_m": round(float(np.mean(vals)), 4),
            "min_m": round(float(np.min(vals)), 4),
            "max_m": round(float(np.max(vals)), 4),
            "p10_m": round(float(np.percentile(vals, 10)), 4),
            "p90_m": round(float(np.percentile(vals, 90)), 4),
        }
    except Exception:
        return {"available": False}


def _build_scene_affordances(
    ctx: PipelineContext,
    caption_evidence: Dict[str, Any],
    object_affordances: Dict[str, Any],
    mask_affordances: Dict[str, Any],
    action_prompts: Dict[str, List[str]],
    role_prompts: Dict[str, List[str]],
    ontology: Dict[str, Any],
) -> Dict[str, Any]:
    scene_text = " ".join(
        str(r.get("text", ""))
        for r in caption_evidence.get("records") or []
        if str(r.get("entity_type", "")) in {"scene", "region", "relation"}
    )
    hit_lambda = number(ontology, "scoring", "prompt_hit_lambda", 0.55)
    action_scene = _score_prompt_bank(scene_text, action_prompts, hit_lambda=hit_lambda)
    role_scene = _score_prompt_bank(scene_text, role_prompts, hit_lambda=hit_lambda)

    action_aggr = _aggregate_scores(
        [a for o in object_affordances.get("objects") or [] for a in o.get("actions") or []],
        action_scene,
        ontology,
    )
    role_aggr = _aggregate_scores(
        [r for o in object_affordances.get("objects") or [] for r in o.get("roles") or []],
        role_scene,
        ontology,
    )

    region_rows = _scene_region_rows(ctx, caption_evidence, role_prompts, action_prompts, ontology)
    return {
        "schema": "citv_scene_affordances_v1",
        "version": "1.0",
        "stem": ctx.stem,
        "timestamp": ctx.timestamp,
        "image_size": {"width": int(ctx.width), "height": int(ctx.height)},
        "summary": {
            "cpu_first": True,
            "caption_aware": True,
            "object_count": len(object_affordances.get("objects") or []),
            "mask_count": len(mask_affordances.get("masks") or []),
            "dominant_actions": _top_scores(action_aggr, limit=10),
            "dominant_roles": _top_scores(role_aggr, limit=10),
        },
        "affordances": _top_scores(action_aggr, limit=16),
        "roles": _top_scores(role_aggr, limit=16),
        "regions": region_rows,
        "open_vocab": {
            "closed_vocabulary": False,
            "seed_prompt_count": len(action_prompts) + len(role_prompts),
            "dynamic_concepts": list(caption_evidence.get("open_vocab_concepts") or [])[:64],
            "grounding_rule": (
                "Scene-level actions are suggestions. A path/action is accepted only "
                "when local object, mask, region, depth, or visibility evidence supports it."
            ),
        },
        "evidence_contract": {
            "uses_captions": True,
            "uses_labels": True,
            "uses_relations": True,
            "uses_masks": True,
            "uses_depth": ctx.metric_depth is not None,
            "no_new_gpu_model": True,
        },
    }


def _aggregate_scores(
    rows: Sequence[Dict[str, Any]],
    scene_scores: Dict[str, Dict[str, Any]],
    ontology: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, List[float]] = {}
    terms: Dict[str, List[str]] = {}
    for name, data in scene_scores.items():
        values.setdefault(str(name), []).append(_float(data.get("score"), 0.0))
        terms.setdefault(str(name), []).extend(str(t) for t in data.get("evidence_terms", []) or [])
    for row in rows:
        name = str(row.get("name", ""))
        if not name:
            continue
        values.setdefault(name, []).append(_float(row.get("score"), 0.0))
        terms.setdefault(name, []).extend(str(t) for t in row.get("evidence_terms", []) or [])
    out: Dict[str, Dict[str, Any]] = {}
    for name, vals in values.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        fused = float(max(
            np.max(arr),
            np.mean(arr) + number(ontology, "scoring", "aggregate_std_bonus", 0.25) * np.std(arr),
        ))
        out[name] = {
            "score": round(max(0.0, min(1.0, fused)), 4),
            "evidence_terms": sorted(set(terms.get(name, [])))[:12],
        }
    return out


def _scene_region_rows(
    ctx: PipelineContext,
    caption_evidence: Dict[str, Any],
    role_prompts: Dict[str, List[str]],
    action_prompts: Dict[str, List[str]],
    ontology: Dict[str, Any],
) -> List[Dict[str, Any]]:
    captions = _caption_lookup(caption_evidence).get("region", {})
    hit_lambda = number(ontology, "scoring", "prompt_hit_lambda", 0.55)
    rows: List[Dict[str, Any]] = []
    for region in ctx.region_partition_meta or []:
        rid = str(region.get("id", "") or "")
        text = _text_blob(
            region.get("type"),
            region.get("semantic_label"),
            (captions.get(rid, {}) or {}).get("text"),
        )
        rows.append({
            "region_id": rid,
            "region_index": int(region.get("region_index", 0) or 0),
            "region_type": str(region.get("type", "")),
            "semantic_label": str(region.get("semantic_label", "")),
            "depth_stats": dict(region.get("depth_stats") or {}),
            "caption": str((captions.get(rid, {}) or {}).get("text", "")),
            "roles": _top_scores(_score_prompt_bank(text, role_prompts, hit_lambda=hit_lambda), limit=6),
            "actions": _top_scores(_score_prompt_bank(text, action_prompts, hit_lambda=hit_lambda), limit=6),
        })
    return rows


def _ensure_support_mask(ctx: PipelineContext, objects: List[Dict[str, Any]]) -> None:
    """Compute and stash the support mask used by anchor synthesis.

    Idempotent: if ``ctx.extra["support_mask"]`` is already populated (e.g.
    by ``paths_export``), this is a no-op. Otherwise we build the union of
    the RANSAC ground plane and semantic support regions; failures are
    silent so the rest of affordance export still runs.
    """
    if "support_mask" in ctx.extra and isinstance(ctx.extra.get("support_mask"), np.ndarray):
        return
    if ctx.region_label_map is None or ctx.metric_depth is None:
        return
    try:
        from ..pathing.ground_plane import build_support_mask

        # Build a quick object-pixel mask so the plane fit avoids actor pixels.
        h, w = ctx.height, ctx.width
        obj_mask = np.zeros((h, w), dtype=bool)
        try:
            import cv2 as _cv2
            for obj in objects:
                m = obj.get("_sam2_mask_array")
                if m is None:
                    continue
                mm = np.asarray(m, dtype=bool)
                if mm.shape[:2] != (h, w):
                    mm = _cv2.resize(mm.astype(np.uint8), (w, h), interpolation=_cv2.INTER_NEAREST) > 0
                obj_mask |= mm
        except Exception:
            obj_mask = np.zeros((h, w), dtype=bool)

        support_mask, info = build_support_mask(
            ctx.metric_depth,
            ctx.intrinsics,
            ctx.region_label_map,
            list(ctx.region_partition_meta),
            object_mask=obj_mask if obj_mask.any() else None,
        )
        if support_mask is not None and support_mask.any():
            ctx.extra["support_mask"] = support_mask
            ctx.extra["support_mask_info"] = info
    except Exception:
        return


__all__ = ["run"]
