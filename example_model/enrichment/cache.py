"""Raw API cache helpers for enrichment pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foundation.manifest import PROJECT_ROOT

ENRICHED_DIR = PROJECT_ROOT / "data" / "enriched"
_CACHE_ROOT = ENRICHED_DIR / "cache"
_VALID_APIS = {"tmdb", "omdb", "wikidata"}


def ensure_cache_dirs() -> None:
    for api in _VALID_APIS:
        (_CACHE_ROOT / api).mkdir(parents=True, exist_ok=True)


def cache_path(api: str, entity_id: str) -> Path:
    normalized_api = api.strip().lower()
    if normalized_api not in _VALID_APIS:
        raise ValueError(f"Unsupported cache api {api!r}; expected one of {sorted(_VALID_APIS)}")
    safe_id = str(entity_id).strip()
    if not safe_id:
        raise ValueError("entity_id must be non-empty")
    return _CACHE_ROOT / normalized_api / f"{safe_id}.json"


def is_cached(api: str, entity_id: str) -> bool:
    return cache_path(api, entity_id).exists()


def read_cache(api: str, entity_id: str) -> dict[str, Any] | None:
    path = cache_path(api, entity_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache(api: str, entity_id: str, data: dict[str, Any]) -> Path:
    path = cache_path(api, entity_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path
