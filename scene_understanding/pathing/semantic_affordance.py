"""Latent-space affordance mapping: RAM++ tags → action-type scores.

Phase 3.1 implementation. Maps per-object RAM++ tags onto configurable
open-vocabulary action prompt banks via simple token-overlap similarity. No
neural embedding model is required at inference; the vocabulary comes from
``path_action_ontology.json`` or a config override.

Outputs:
  ``compute_tag_affordances(objects)`` → list of per-object affordance dicts
  ``build_region_affordance_map(objects, region_partition_meta)`` → per-region
      aggregated affordance scores (max-pooled over member objects)
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..action_ontology import load_action_ontology, prompt_bank
from ..evidence import normalize_label


def _action_keywords(ontology: Optional[Mapping[str, Any]] = None) -> Dict[str, List[str]]:
    bank = prompt_bank(ontology or load_action_ontology(), "action_prompts")
    return {name: values for name, values in bank.items() if values}


def _tag_to_vector(tags: List[str], ontology: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """Count keyword matches between tag list and each action prototype."""
    action_keywords = _action_keywords(ontology)
    if not action_keywords:
        return {}
    tag_set = {normalize_label(t) for t in tags if normalize_label(t)}
    scores: Dict[str, float] = {}
    for action, keywords in action_keywords.items():
        kws = [normalize_label(kw) for kw in keywords if normalize_label(kw)]
        hits = sum(1 for kw in kws if kw in tag_set or any(kw in t for t in tag_set))
        scores[action] = float(hits)
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def compute_tag_affordances(
    objects: List[Dict[str, Any]],
    tags_key: str = "tags",
    *,
    ontology: Optional[Mapping[str, Any]] = None,
    cfg: Any = None,
) -> List[Dict[str, Any]]:
    """Compute per-object action affordance scores from RAM++ tags.

    Args:
        objects: List of object dicts. Each should have a ``sources.RAM++.tags``
                 list or a top-level ``tags`` list.
        tags_key: Key under ``sources.RAM++`` that holds the tag list.

    Returns:
        List of dicts ``{object_id, label, tags, affordances: {action: score}}``.
    """
    onto = ontology or load_action_ontology(cfg)
    results = []
    for obj in objects:
        obj_id = str(obj.get("id", ""))
        label = str(obj.get("label", "object"))
        src = obj.get("sources") or {}
        rampp = src.get("RAM++") or {}
        tags: List[str] = list(rampp.get(tags_key, []) or obj.get(tags_key, []) or [])
        if label and label != "object":
            tags = [label] + tags
        affordances = _tag_to_vector(tags, onto)
        results.append({
            "object_id": obj_id,
            "label": label,
            "tags": tags,
            "affordances": affordances,
            "dominant_action": max(affordances, key=affordances.get) if affordances else "",
        })
    return results


def build_region_affordance_map(
    objects: List[Dict[str, Any]],
    region_partition_meta: Optional[List[Dict[str, Any]]] = None,
    *,
    ontology: Optional[Mapping[str, Any]] = None,
    cfg: Any = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-object affordances to per-region scores via max-pooling.

    Returns a dict mapping ``region_id → {action: score}``.
    """
    onto = ontology or load_action_ontology(cfg)
    per_obj = compute_tag_affordances(objects, ontology=onto)
    obj_aff_by_id = {r["object_id"]: r["affordances"] for r in per_obj}

    by_region: Dict[str, List[Dict[str, float]]] = {}
    for obj in objects:
        rid = str(obj.get("region_id") or f"region_{obj.get('region_index', 0)}")
        oid = str(obj.get("id", ""))
        aff = obj_aff_by_id.get(oid, {})
        by_region.setdefault(rid, []).append(aff)

    region_aff: Dict[str, Dict[str, float]] = {}
    actions = list(_action_keywords(onto).keys())
    for rid, aff_list in by_region.items():
        merged: Dict[str, float] = {}
        for a in actions:
            merged[a] = max((d.get(a, 0.0) for d in aff_list), default=0.0)
        total = sum(merged.values()) or 1.0
        region_aff[rid] = {k: v / total for k, v in merged.items()}

    return region_aff
