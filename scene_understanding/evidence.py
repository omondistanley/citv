"""Shared evidence normalization for staged scene understanding.

The staged pipeline receives semantics from multiple providers that do not use
the same field names. This module keeps that normalization in one place so
caption, affordance, path, action, and bundle exports do not each invent their
own schema assumptions.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*")


def text_blob(*parts: Any) -> str:
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


def tokens(*parts: Any) -> List[str]:
    return _TOKEN_RE.findall(text_blob(*parts).lower())


def normalize_label(value: Any) -> str:
    return " ".join(tokens(value)).strip()


def _string_set(values: Any) -> Set[str]:
    if isinstance(values, str):
        return {normalize_label(values)}
    if isinstance(values, Iterable):
        return {normalize_label(v) for v in values if normalize_label(v)}
    return set()


def label_quality_policy(ontology: Mapping[str, Any]) -> Dict[str, Set[str]]:
    raw = ontology.get("label_quality") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "generic_labels": _string_set(raw.get("generic_labels") or ["object", "thing", "item"]),
        "meta_visual_terms": _string_set(raw.get("meta_visual_terms") or []),
        "meta_visual_phrases": _string_set(raw.get("meta_visual_phrases") or []),
    }


def is_generic_label(label: Any, ontology: Mapping[str, Any]) -> bool:
    label_n = normalize_label(label)
    if not label_n:
        return True
    return label_n in label_quality_policy(ontology)["generic_labels"]


def is_meta_visual_label(label: Any, ontology: Mapping[str, Any]) -> bool:
    label_n = normalize_label(label)
    if not label_n:
        return False
    policy = label_quality_policy(ontology)
    if label_n in policy["meta_visual_phrases"]:
        return True
    label_tokens = set(tokens(label_n))
    return bool(label_tokens and label_tokens <= policy["meta_visual_terms"])


def visual_quality_attributes(label: Any, caption: Any, ontology: Mapping[str, Any]) -> List[str]:
    policy = label_quality_policy(ontology)
    found = sorted({t for t in tokens(label, caption) if t in policy["meta_visual_terms"]})
    return found[:12]


def normalize_relation(rel: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a relation dict with canonical and source-compatible fields."""
    subject_id = str(rel.get("subject") or rel.get("sub_id") or rel.get("source_id") or "").strip()
    object_id = str(rel.get("object") or rel.get("obj_id") or rel.get("target_id") or "").strip()
    predicate = str(rel.get("predicate") or rel.get("pred") or rel.get("relation") or "").strip()
    subject_label = str(rel.get("subject_label") or rel.get("sub") or rel.get("subject_name") or "").strip()
    object_label = str(rel.get("object_label") or rel.get("obj") or rel.get("object_name") or "").strip()
    score = _float(
        rel.get("score", rel.get("confidence", rel.get("conf", rel.get("relation_confidence")))),
        0.5,
    )
    return {
        "subject": subject_id,
        "object": object_id,
        "predicate": predicate,
        "subject_id": subject_id,
        "object_id": object_id,
        "subject_label": subject_label,
        "object_label": object_label,
        "score": score,
        "relation_tier": str(rel.get("relation_tier") or rel.get("source_layer") or "").strip(),
        "source_layer": str(rel.get("source_layer") or rel.get("relation_tier") or "").strip(),
        "raw": dict(rel),
    }


def object_id(obj: Mapping[str, Any]) -> str:
    return str(obj.get("id") or obj.get("object_id") or obj.get("graph_id") or "").strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def build_label_candidates(
    *,
    gdino_label: Any,
    gdino_conf: Any,
    florence_label: Any,
    florence_caption: Any,
    rampp_label: Any = "",
    rampp_tags: Optional[Sequence[Any]] = None,
    current_label: Any = "",
    ontology: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fuse open-vocabulary label evidence without hard-coding entity behavior."""
    rampp_tags = list(rampp_tags or [])
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    def add(label: Any, source: str, score: float, caption: Any = "", tags_in: Sequence[Any] = ()) -> None:
        label_n = normalize_label(label)
        if not label_n:
            return
        rec = {
            "label": label_n,
            "source": source,
            "score": round(max(0.0, min(1.0, float(score))), 4),
            "caption": str(caption or ""),
            "tags": [str(t) for t in tags_in if str(t).strip()],
        }
        if is_generic_label(label_n, ontology):
            rec["rejected_reason"] = "generic_label"
            rejected.append(rec)
            return
        if is_meta_visual_label(label_n, ontology):
            rec["rejected_reason"] = "meta_visual_phrase"
            rejected.append(rec)
            return
        candidates.append(rec)

    add(gdino_label, "GroundingDINO", _float(gdino_conf, 0.0), gdino_label)
    add(florence_label, "Florence2", 0.72, florence_caption)
    add(rampp_label, "RAM++", 0.62 if rampp_tags else 0.45, "", rampp_tags)
    add(current_label, "current", 0.35, current_label)

    for tag in rampp_tags:
        add(tag, "RAM++_tag", 0.50, "", [tag])

    # If Florence says "young boy" but the extracted label is weak, keep useful
    # noun-like words as lower-confidence candidates. These are still open-vocab
    # evidence, not fixed class rules.
    policy = label_quality_policy(ontology)
    for word in tokens(florence_caption):
        if word in policy["generic_labels"] or word in policy["meta_visual_terms"]:
            continue
        if len(word) <= 2:
            continue
        if any(c["label"] == word for c in candidates):
            continue
        add(word, "Florence2_caption_token", 0.28, florence_caption)

    candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    canonical = candidates[0]["label"] if candidates else normalize_label(gdino_label or current_label or florence_label or "object")
    source = candidates[0]["source"] if candidates else "fallback"
    return {
        "canonical_label": canonical or "object",
        "label_source": source,
        "label_candidates": candidates[:12],
        "rejected_labels": rejected[:12],
        "visual_quality_attributes": visual_quality_attributes(
            text_blob(current_label, florence_label), florence_caption, ontology
        ),
        "source_agreement": _source_agreement(candidates),
    }


def _source_agreement(candidates: Sequence[Mapping[str, Any]]) -> float:
    if not candidates:
        return 0.0
    top = str(candidates[0].get("label", ""))
    if not top:
        return 0.0
    hits = sum(1 for c in candidates if str(c.get("label", "")) == top)
    return round(float(hits) / float(max(1, len(candidates))), 4)
