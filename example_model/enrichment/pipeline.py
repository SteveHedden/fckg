"""Orchestrator for film enrichment with raw-cache + derived-TTL layers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rdflib import Graph

from enrichment.cache import ENRICHED_DIR, ensure_cache_dirs, is_cached
from enrichment.identifier_parsing import parse_external_id_from_iri
from enrichment.omdb_client import fetch_omdb_movie
from enrichment.person_identifier_writer import generate_person_identifiers_ttl
from enrichment.tmdb_client import fetch_tmdb_movie
from enrichment.wikidata_client import fetch_wikidata_film
from enrichment.ttl_writer import generate_film_credits_ttl, generate_film_metadata_ttl
from foundation.loader import load_foundation_graph

logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file


@dataclass(frozen=True)
class FilmTarget:
    film_uri: str
    release_year: int | None
    tmdb_id: str | None
    imdb_id: str | None
    wikidata_id: str | None


@dataclass(frozen=True)
class EnrichmentSummary:
    films_seen: int
    films_processed: int
    api_fetched: int
    cache_hits: int
    errors: int
    metadata_films_enriched: int
    credits_films_enriched: int
    people_enriched: int
    person_tmdb_identifiers_enriched: int
    person_wikidata_identifiers_enriched: int
    metadata_ttl_path: str
    credits_ttl_path: str
    person_ttl_path: str
    person_identifiers_ttl_path: str


def _safe_year(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def _load_film_targets(foundation_graph: Graph, *, year: int | None = None) -> list[FilmTarget]:
    query = """
    SELECT ?film ?releaseYear ?identifier WHERE {
      ?film a <http://example.org/ontologies/MovieSHACL3#Film> .
      OPTIONAL { ?film <http://example.org/ontologies/MovieSHACL3#releaseYear> ?releaseYear . }
      OPTIONAL { ?film <http://example.org/ontologies/MovieSHACL3#hasIdentifier> ?identifier . }
    }
    """
    targets_by_film: dict[str, dict[str, Any]] = {}
    for row in foundation_graph.query(query):
        film_uri = str(row.film)
        release_year = _safe_year(str(row.releaseYear) if row.releaseYear else None)
        entry = targets_by_film.setdefault(
            film_uri,
            {
                "film_uri": film_uri,
                "release_year": release_year,
                "tmdb_id": None,
                "imdb_id": None,
                "wikidata_id": None,
            },
        )
        if entry["release_year"] is None and release_year is not None:
            entry["release_year"] = release_year
        identifier_iri = str(row.identifier) if row.identifier else None
        if identifier_iri:
            parsed = parse_external_id_from_iri(identifier_iri)
            if parsed is None:
                continue
            provider, external_id = parsed
            if provider == "tmdb" and entry["tmdb_id"] is None:
                entry["tmdb_id"] = external_id
            elif provider == "imdb" and entry["imdb_id"] is None:
                entry["imdb_id"] = external_id
            elif provider == "wikidata" and entry["wikidata_id"] is None:
                entry["wikidata_id"] = external_id

    targets: list[FilmTarget] = []
    for film_uri in sorted(targets_by_film):
        entry = targets_by_film[film_uri]
        release_year = entry["release_year"]
        if year is not None and release_year != year:
            continue
        targets.append(
            FilmTarget(
                film_uri=film_uri,
                release_year=release_year,
                tmdb_id=entry["tmdb_id"],
                imdb_id=entry["imdb_id"],
                wikidata_id=entry["wikidata_id"],
            )
        )
    return targets


def run_enrichment_pipeline(
    *,
    year: int | None = None,
    force: bool = False,
    cache_only: bool = False,
    dry_run: bool = False,
    foundation_graph: Graph | None = None,
) -> EnrichmentSummary:
    ensure_cache_dirs()
    graph = foundation_graph if foundation_graph is not None else load_foundation_graph()
    targets = _load_film_targets(graph, year=year)

    tmdb_key_available = bool(os.getenv("TMDB_API_KEY"))
    omdb_key_available = bool(os.getenv("OMDB_API_KEY"))
    if not tmdb_key_available and not cache_only:
        logger.warning("TMDB_API_KEY not set; TMDB fetches will be skipped.")
    if not omdb_key_available and not cache_only:
        logger.warning("OMDB_API_KEY not set; OMDB fetches will be skipped.")

    api_fetched = 0
    cache_hits = 0
    errors = 0
    films_processed = 0

    for idx, film in enumerate(targets, start=1):
        films_processed += 1
        if dry_run:
            continue
        if not cache_only:
            if film.tmdb_id:
                if not force and is_cached("tmdb", film.tmdb_id):
                    cache_hits += 1
                elif tmdb_key_available:
                    try:
                        fetch_tmdb_movie(film.tmdb_id, force=force)
                        api_fetched += 1
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        logger.error("TMDB enrichment failed for %s: %s", film.film_uri, exc)

            if film.imdb_id:
                if not force and is_cached("omdb", film.imdb_id):
                    cache_hits += 1
                elif omdb_key_available:
                    try:
                        fetch_omdb_movie(film.imdb_id, force=force)
                        api_fetched += 1
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        logger.error("OMDB enrichment failed for %s: %s", film.film_uri, exc)

            wikidata_cache_key = film.wikidata_id or film.imdb_id
            if wikidata_cache_key:
                if not force and is_cached("wikidata", wikidata_cache_key):
                    cache_hits += 1
                else:
                    try:
                        fetch_wikidata_film(
                            wikidata_id=film.wikidata_id,
                            imdb_id=film.imdb_id,
                            force=force,
                        )
                        api_fetched += 1
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        logger.error("Wikidata enrichment failed for %s: %s", film.film_uri, exc)

        if idx % 50 == 0 or idx == len(targets):
            logger.info(
                "Enrichment progress %d/%d films (cache_hits=%d, fetched=%d, errors=%d)",
                idx,
                len(targets),
                cache_hits,
                api_fetched,
                errors,
            )

    metadata_path = ENRICHED_DIR / "film_metadata.ttl"
    credits_path = ENRICHED_DIR / "film_credits.ttl"
    person_path = ENRICHED_DIR / "person_metadata.ttl"
    person_identifiers_path = ENRICHED_DIR / "person_identifiers.ttl"
    metadata_count = 0
    credits_count = 0
    people_count = 0
    person_tmdb_identifiers_count = 0
    person_wikidata_identifiers_count = 0
    if not dry_run:
        metadata_count = generate_film_metadata_ttl(graph, metadata_path, ENRICHED_DIR / "cache")
        credits_count, people_count = generate_film_credits_ttl(
            graph,
            credits_path,
            person_path,
            ENRICHED_DIR / "cache",
        )
        person_tmdb_identifiers_count, person_wikidata_identifiers_count = generate_person_identifiers_ttl(
            graph,
            person_metadata_path=person_path,
            output_path=person_identifiers_path,
            cache_only=cache_only,
        )

    return EnrichmentSummary(
        films_seen=len(targets),
        films_processed=films_processed,
        api_fetched=api_fetched,
        cache_hits=cache_hits,
        errors=errors,
        metadata_films_enriched=metadata_count,
        credits_films_enriched=credits_count,
        people_enriched=people_count,
        person_tmdb_identifiers_enriched=person_tmdb_identifiers_count,
        person_wikidata_identifiers_enriched=person_wikidata_identifiers_count,
        metadata_ttl_path=str(metadata_path),
        credits_ttl_path=str(credits_path),
        person_ttl_path=str(person_path),
        person_identifiers_ttl_path=str(person_identifiers_path),
    )
