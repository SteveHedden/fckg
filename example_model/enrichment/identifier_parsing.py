"""Helpers for extracting external IDs from identifier IRIs."""

from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urlparse

TMDB_MOVIE_PATH_RE = re.compile(r"^/movie/(\d+)$")
IMDB_TITLE_PATH_RE = re.compile(r"^/title/(tt\d+)$", re.IGNORECASE)
WIKIDATA_PATH_RE = re.compile(r"^/(?:wiki|entity)/(Q\d+)$", re.IGNORECASE)


def _normalize_path(path: str) -> str:
    normalized = path.rstrip("/")
    return normalized or path


def parse_external_id_from_iri(identifier_iri: str) -> tuple[str, str] | None:
    """Return a canonical ``(provider, id)`` pair parsed from an identifier IRI."""
    text = str(identifier_iri or "").strip()
    if not text:
        return None

    parsed = urlparse(text)
    host = parsed.netloc.lower()
    path = _normalize_path(parsed.path)

    tmdb_match = TMDB_MOVIE_PATH_RE.match(path)
    if tmdb_match and "themoviedb.org" in host:
        return "tmdb", tmdb_match.group(1)

    imdb_match = IMDB_TITLE_PATH_RE.match(path)
    if imdb_match and "imdb.com" in host:
        return "imdb", imdb_match.group(1)

    wikidata_match = WIKIDATA_PATH_RE.match(path)
    if wikidata_match and "wikidata.org" in host:
        return "wikidata", wikidata_match.group(1).upper()

    return None


def parse_external_ids(identifier_iris: Iterable[str]) -> dict[str, str | None]:
    """Extract known provider IDs from one or more identifier IRIs."""
    ids: dict[str, str | None] = {"tmdb": None, "imdb": None, "wikidata": None}
    for identifier in identifier_iris:
        parsed = parse_external_id_from_iri(identifier)
        if parsed is None:
            continue
        provider, value = parsed
        if provider in ids and ids[provider] is None:
            ids[provider] = value
    return ids

