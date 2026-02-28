"""TMDB person client with cache-first behavior."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import requests

from enrichment.cache import read_cache, write_cache

logger = logging.getLogger(__name__)
_TMDB_THROTTLE_SECONDS = 0.25
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _TMDB_THROTTLE_SECONDS:
        time.sleep(_TMDB_THROTTLE_SECONDS - elapsed)
    _last_call = time.time()


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("TMDB_API_KEY")
    if not key:
        raise RuntimeError(
            "TMDB_API_KEY environment variable is not set. "
            "Get a free key at https://www.themoviedb.org/settings/api"
        )
    return key


def fetch_tmdb_person(
    person_id: str | int,
    *,
    force: bool = False,
    api_key: str | None = None,
    max_retries: int = 3,
    request_fn: Callable[..., requests.Response] | None = None,
) -> dict[str, Any]:
    pid = str(person_id).strip()
    if not pid:
        raise ValueError("person_id must be non-empty")

    cache_id = f"person_{pid}"
    if not force:
        cached = read_cache("tmdb", cache_id)
        if cached is not None:
            return cached

    key = _resolve_api_key(api_key)
    requester = request_fn or requests.get
    url = f"https://api.themoviedb.org/3/person/{pid}"
    headers = {"Accept": "application/json"}

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            response = requester(url, headers=headers, params={"api_key": key}, timeout=30)
        except requests.RequestException as exc:  # pragma: no cover
            last_error = exc
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
            break

        if response.status_code == 200:
            payload = response.json()
            write_cache("tmdb", cache_id, payload)
            return payload
        if response.status_code == 404:
            sentinel = {"error": "not_found", "source": "tmdb_person", "id": pid}
            write_cache("tmdb", cache_id, sentinel)
            return sentinel
        if response.status_code == 429 and attempt < max_retries:
            retry_after = float(response.headers.get("Retry-After", "1") or 1)
            time.sleep(retry_after)
            continue
        if response.status_code >= 500 and attempt < max_retries:
            time.sleep(0.5 * attempt)
            continue
        raise RuntimeError(f"TMDB person request failed for {pid}: {response.status_code} {response.text[:200]}")

    raise RuntimeError(f"TMDB person request failed for {pid} after {max_retries} attempts: {last_error}")
