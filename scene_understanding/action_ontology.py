"""Configurable action and affordance ontology/policy loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


_DEFAULT_PATH = Path(__file__).resolve().parent / "resources" / "path_action_ontology.json"


def load_action_ontology(cfg: Any = None) -> Dict[str, Any]:
    """Load the action ontology from config override or package resource."""
    inline = getattr(cfg, "path_action_ontology", None) if cfg is not None else None
    if isinstance(inline, Mapping):
        return _merge_dicts(_read_default(), dict(inline))

    path = getattr(cfg, "path_action_ontology_json", None) if cfg is not None else None
    if path:
        try:
            return _merge_dicts(_read_default(), json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception:
            return _read_default()
    return _read_default()


def prompt_bank(ontology: Mapping[str, Any], key: str) -> Dict[str, List[str]]:
    raw = ontology.get(key) or {}
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, List[str]] = {}
    for name, values in raw.items():
        if isinstance(values, str):
            out[str(name)] = [values]
        elif isinstance(values, Iterable):
            out[str(name)] = [str(v) for v in values if str(v).strip()]
    return out


def number(ontology: Mapping[str, Any], section: str, key: str, default: float) -> float:
    raw = ontology.get(section) or {}
    if not isinstance(raw, Mapping):
        return default
    try:
        val = float(raw.get(key, default))
        return val if val == val else default
    except (TypeError, ValueError):
        return default


def integer(ontology: Mapping[str, Any], section: str, key: str, default: int) -> int:
    return int(round(number(ontology, section, key, float(default))))


def string(ontology: Mapping[str, Any], section: str, key: str, default: str) -> str:
    raw = ontology.get(section) or {}
    if not isinstance(raw, Mapping):
        return default
    val = str(raw.get(key, default))
    return val if val else default


def list_section(ontology: Mapping[str, Any], key: str) -> List[Dict[str, Any]]:
    raw = ontology.get(key) or []
    if not isinstance(raw, list):
        return []
    return [dict(x) for x in raw if isinstance(x, Mapping)]


def dict_section(ontology: Mapping[str, Any], key: str) -> Dict[str, Any]:
    raw = ontology.get(key) or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _read_default() -> Dict[str, Any]:
    try:
        return json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"schema": "citv_path_action_ontology_v1", "version": "fallback"}


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_dicts(dict(out[key]), dict(val))
        else:
            out[key] = val
    return out


__all__ = [
    "dict_section",
    "integer",
    "list_section",
    "load_action_ontology",
    "number",
    "prompt_bank",
    "string",
]
