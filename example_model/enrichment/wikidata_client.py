"""Wikidata film enrichment client — caches full entity JSON via the Entity Data API."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

import requests

from enrichment.cache import read_cache, write_cache

logger = logging.getLogger(__name__)
_WIKIDATA_THROTTLE_SECONDS = 1.0  # <= 1 req/sec
_last_call = 0.0
_ENTITY_API = "https://www.wikidata.org/wiki/Special:EntityData"
_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def _throttle() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _WIKIDATA_THROTTLE_SECONDS:
        time.sleep(_WIKIDATA_THROTTLE_SECONDS - elapsed)
    _last_call = time.time()


def _normalize_qid(value: str | None) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if re.fullmatch(r"Q\d+", text):
        return text
    return None


def _normalize_imdb(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if re.fullmatch(r"tt\d+", text):
        return text
    return None


def _resolve_qid_from_imdb(
    imdb_id: str,
    *,
    request_fn: Callable[..., requests.Response] | None = None,
) -> str | None:
    """Look up the Wikidata Q-ID for an IMDB ID via SPARQL."""
    query = f'SELECT ?film WHERE {{ ?film wdt:P345 "{imdb_id}" . }} LIMIT 1'
    requester = request_fn or requests.get
    _throttle()
    try:
        response = requester(
            _SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "fckg-enrichment/1.0",
            },
            timeout=45,
        )
    except requests.RequestException as exc:
        logger.warning("SPARQL lookup failed for %s: %s", imdb_id, exc)
        return None

    if response.status_code != 200:
        logger.warning("SPARQL lookup returned %d for %s", response.status_code, imdb_id)
        return None

    bindings = response.json().get("results", {}).get("bindings", [])
    if not bindings:
        return None
    film_iri = str(bindings[0].get("film", {}).get("value", "")).strip()
    qid = film_iri.rstrip("/").split("/")[-1].upper()
    return qid if re.fullmatch(r"Q\d+", qid) else None


def resolve_qid_labels(
    qids: list[str],
    *,
    request_fn: Callable[..., requests.Response] | None = None,
    label_cache: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve Wikidata Q-IDs to English labels via wbgetentities API.

    Batches up to 50 IDs per request.  Returns {qid: label} mapping.
    Unknown Q-IDs are mapped to the Q-ID itself.
    """
    if label_cache is None:
        label_cache = {}
    result: dict[str, str] = {}
    to_resolve: list[str] = []
    for qid in qids:
        if qid in label_cache:
            result[qid] = label_cache[qid]
        else:
            to_resolve.append(qid)

    requester = request_fn or requests.get
    # Batch in groups of 50
    for i in range(0, len(to_resolve), 50):
        batch = to_resolve[i : i + 50]
        _throttle()
        try:
            response = requester(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": "fckg-enrichment/1.0"},
                timeout=45,
            )
        except requests.RequestException as exc:
            logger.warning("Label resolution failed: %s", exc)
            for qid in batch:
                result[qid] = qid
            continue

        if response.status_code != 200:
            logger.warning("Label resolution returned %d", response.status_code)
            for qid in batch:
                result[qid] = qid
            continue

        entities = response.json().get("entities", {})
        for qid in batch:
            entity = entities.get(qid, {})
            label = entity.get("labels", {}).get("en", {}).get("value")
            resolved = label if label else qid
            result[qid] = resolved
            label_cache[qid] = resolved

    return result


def fetch_wikidata_film(
    *,
    wikidata_id: str | None = None,
    imdb_id: str | None = None,
    force: bool = False,
    max_retries: int = 3,
    request_fn: Callable[..., requests.Response] | None = None,
) -> dict[str, Any]:
    """Fetch and cache the full Wikidata entity JSON for a film.

    Prefers lookup by Q-ID (from hasIdentifier).  Falls back to resolving
    Q-ID from IMDB ID via a lightweight SPARQL query, then fetching the
    entity JSON.  The full entity payload is cached so downstream code can
    extract any property without re-fetching.
    """
    qid = _normalize_qid(wikidata_id)
    imdb = _normalize_imdb(imdb_id)
    if not qid and not imdb:
        raise ValueError("Provide wikidata_id (Q...) or imdb_id (tt...).")

    cache_key = qid or imdb
    assert cache_key is not None

    if not force:
        cached = read_cache("wikidata", cache_key)
        if cached is not None:
            return cached

    # Resolve Q-ID if we only have IMDB ID
    if not qid and imdb:
        qid = _resolve_qid_from_imdb(imdb, request_fn=request_fn)
        if not qid:
            sentinel: dict[str, Any] = {"error": "not_found", "source": "wikidata", "id": cache_key}
            write_cache("wikidata", cache_key, sentinel)
            return sentinel

    assert qid is not None
    requester = request_fn or requests.get
    url = f"{_ENTITY_API}/{qid}.json"

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            response = requester(
                url,
                headers={"User-Agent": "fckg-enrichment/1.0"},
                timeout=45,
            )
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "Wikidata entity request failed for %s (attempt %d/%d): %s",
                qid, attempt, max_retries, exc,
            )
            if attempt < max_retries:
                time.sleep(1.0 * attempt)
                continue
            break

        if response.status_code == 200:
            raw = response.json()
            entities = raw.get("entities", {})
            entity = entities.get(qid)
            if entity is None:
                sentinel = {"error": "not_found", "source": "wikidata", "id": qid}
                write_cache("wikidata", cache_key, sentinel)
                return sentinel
            # Store the full entity with a source marker
            entity["_source"] = "wikidata"
            entity["_cache_key"] = cache_key
            write_cache("wikidata", cache_key, entity)
            return entity

        if response.status_code == 429 and attempt < max_retries:
            retry_after = float(response.headers.get("Retry-After", "1") or 1)
            logger.warning("Wikidata rate-limited for %s, retrying in %.2fs", qid, retry_after)
            time.sleep(max(1.0, retry_after))
            continue

        if response.status_code >= 500 and attempt < max_retries:
            time.sleep(1.0 * attempt)
            continue

        raise RuntimeError(f"Wikidata entity request failed for {qid}: {response.status_code}")

    raise RuntimeError(f"Wikidata entity request failed for {qid} after {max_retries} attempts: {last_error}")
