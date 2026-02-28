"""Person enrichment orchestration from TMDB person API + cache."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rdflib import Graph

from enrichment.cache import ENRICHED_DIR, ensure_cache_dirs, is_cached
from enrichment.person_ttl_writer import generate_person_metadata_ttl
from enrichment.tmdb_person_client import fetch_tmdb_person
from foundation.loader import load_foundation_graph

logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file


@dataclass(frozen=True)
class PersonEnrichmentSummary:
    people_seen: int
    fetched: int
    cache_hits: int
    errors: int
    enriched: int
    output_path: str


def _candidate_tmdb_people(cache_root: Path) -> list[tuple[str, str]]:
    people: dict[str, str] = {}
    for movie_cache in (cache_root / "tmdb").glob("*.json"):
        if movie_cache.name.startswith("person_"):
            continue
        try:
            import json

            payload = json.loads(movie_cache.read_text(encoding="utf-8"))
        except Exception:
            continue
        credits = payload.get("credits") or {}
        for section in ("crew", "cast"):
            for entry in credits.get(section, []) or []:
                pid = str(entry.get("id") or "").strip()
                name = str(entry.get("name") or "").strip()
                if pid and name:
                    people[pid] = name
    return sorted(people.items())


def run_person_enrichment_pipeline(
    *,
    force: bool = False,
    cache_only: bool = False,
    dry_run: bool = False,
    foundation_graph: Graph | None = None,
) -> PersonEnrichmentSummary:
    ensure_cache_dirs()
    graph = foundation_graph if foundation_graph is not None else load_foundation_graph()
    cache_root = ENRICHED_DIR / "cache"
    candidates = _candidate_tmdb_people(cache_root)

    tmdb_key_available = bool(os.getenv("TMDB_API_KEY"))
    if not tmdb_key_available and not cache_only:
        logger.warning("TMDB_API_KEY not set; person API fetches will be skipped.")

    fetched = 0
    cache_hits = 0
    errors = 0
    for idx, (pid, _name) in enumerate(candidates, start=1):
        if dry_run:
            continue
        if cache_only:
            continue

        cache_id = f"person_{pid}"
        if not force and is_cached("tmdb", cache_id):
            cache_hits += 1
        elif tmdb_key_available:
            try:
                fetch_tmdb_person(pid, force=force)
                fetched += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.error("Failed person enrichment for TMDB person %s: %s", pid, exc)

        if idx % 100 == 0 or idx == len(candidates):
            logger.info(
                "Person enrichment progress %d/%d (cache_hits=%d fetched=%d errors=%d)",
                idx,
                len(candidates),
                cache_hits,
                fetched,
                errors,
            )

    output = ENRICHED_DIR / "person_metadata.ttl"
    enriched = 0
    if not dry_run:
        enriched = generate_person_metadata_ttl(graph, output, cache_root)

    return PersonEnrichmentSummary(
        people_seen=len(candidates),
        fetched=fetched,
        cache_hits=cache_hits,
        errors=errors,
        enriched=enriched,
        output_path=str(output),
    )
