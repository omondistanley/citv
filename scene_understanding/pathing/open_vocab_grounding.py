"""Open-vocabulary evidence fusion for path/action grounding.

The path stage already receives captions, affordances, regions, masks, and
relations.  This module keeps that evidence open-ended: concepts are scored from
the text and scores already produced upstream, rather than from closed label
branches inside the planner.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if val == val else default
    except (TypeError, ValueError):
        return default


def tokens(text: Any) -> List[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(str(text or "").lower())]


def evidence_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "id",
        "label",
        "name",
        "caption",
        "description",
        "semantic_label",
        "region_type",
        "predicate",
        "subject",
        "object",
    ):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    for key in ("tags", "evidence_terms", "soft_evidence_terms"):
        vals = row.get(key)
        if isinstance(vals, Sequence) and not isinstance(vals, (str, bytes)):
            parts.extend(str(v) for v in vals if str(v).strip())
    return " ".join(parts)


def concept_score(text: str, concept_terms: Iterable[str]) -> Dict[str, Any]:
    """Score arbitrary text against an arbitrary set of concept terms."""
    toks = set(tokens(text))
    term_tokens: List[str] = []
    for term in concept_terms:
        term_tokens.extend(tokens(term))
    term_set = set(term_tokens)
    if not toks or not term_set:
        return {"score": 0.0, "evidence_terms": []}
    matched = sorted(toks.intersection(term_set))
    phrase_bonus = 0.0
    lower = text.lower()
    for term in concept_terms:
        t = str(term).strip().lower()
        if t and " " in t and t in lower:
            phrase_bonus = max(phrase_bonus, 0.18)
    score = min(1.0, len(matched) / max(1.0, len(term_set) ** 0.5) + phrase_bonus)
    return {"score": round(float(score), 4), "evidence_terms": matched[:16]}


def dynamic_concepts_from_evidence(rows: Iterable[Dict[str, Any]], *, limit: int = 96) -> List[str]:
    counts: Dict[str, int] = {}
    stop = {
        "the", "and", "with", "from", "into", "onto", "this", "that", "image",
        "photo", "picture", "scene", "object", "region", "mask", "unknown",
    }
    for row in rows:
        for tok in tokens(evidence_text(row)):
            if len(tok) < 3 or tok in stop:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    return [k for k, _v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def build_open_vocab_grounding(
    *,
    evidence_rows: List[Dict[str, Any]],
    seed_concepts: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    dynamic = dynamic_concepts_from_evidence(evidence_rows)
    concepts: Dict[str, Dict[str, Any]] = {}
    all_text = "\n".join(evidence_text(r) for r in evidence_rows)
    for name, terms in seed_concepts.items():
        merged_terms = list(terms or []) + dynamic
        scored = concept_score(all_text, merged_terms)
        concepts[str(name)] = {
            "score": scored["score"],
            "evidence_terms": scored["evidence_terms"],
            "scope": "open_vocab_seed_plus_dynamic",
        }
    return {
        "schema": "citv_open_vocab_grounding_v1",
        "evidence_record_count": len(evidence_rows),
        "dynamic_concepts": dynamic,
        "concepts": concepts,
        "local_validation_policy": {
            "scene_level_is_prior_only": True,
            "local_evidence_required_for_acceptance": True,
        },
    }


__all__ = [
    "build_open_vocab_grounding",
    "concept_score",
    "dynamic_concepts_from_evidence",
    "evidence_text",
    "tokens",
]
