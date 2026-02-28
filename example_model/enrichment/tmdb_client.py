"""TMDB client with local raw-response caching and throttling."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import requests

from enrichment.cache import read_cache, write_cache

logger = logging.getLogger(__name__)
_TMDB_THROTTLE_SECONDS = 0.25  # <= 4 req/sec to stay below 40/10s
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


def fetch_tmdb_movie(
    tmdb_id: str,
    *,
    force: bool = False,
    api_key: str | None = None,
    max_retries: int = 3,
    request_fn: Callable[..., requests.Response] | None = None,
) -> dict[str, Any]:
    entity_id = str(tmdb_id).strip()
    if entity_id.endswith(".0"):
        entity_id = entity_id[:-2]
    if not entity_id:
        raise ValueError("tmdb_id must be non-empty")

    if not force:
        cached = read_cache("tmdb", entity_id)
        if cached is not None:
            return cached

    key = _resolve_api_key(api_key)
    requester = request_fn or requests.get
    url = f"https://api.themoviedb.org/3/movie/{entity_id}"
    headers = {"Accept": "application/json"}

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            response = requester(
                url,
                headers=headers,
                params={"api_key": key, "append_to_response": "credits"},
                timeout=30,
            )
        except requests.RequestException as exc:  # pragma: no cover - exercised via mock side-effects
            last_error = exc
            logger.warning("TMDB request failed for %s (attempt %d/%d): %s", entity_id, attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
            break

        if response.status_code == 200:
            payload = response.json()
            write_cache("tmdb", entity_id, payload)
            return payload

        if response.status_code == 404:
            sentinel = {"error": "not_found", "source": "tmdb", "id": entity_id}
            write_cache("tmdb", entity_id, sentinel)
            return sentinel

        if response.status_code == 429 and attempt < max_retries:
            retry_after = float(response.headers.get("Retry-After", "1") or 1)
            logger.warning("TMDB rate-limited for %s, retrying in %.2fs", entity_id, retry_after)
            time.sleep(retry_after)
            continue

        if response.status_code >= 500 and attempt < max_retries:
            time.sleep(0.5 * attempt)
            continue

        raise RuntimeError(f"TMDB request failed for {entity_id}: {response.status_code} {response.text[:200]}")

    raise RuntimeError(f"TMDB request failed for {entity_id} after {max_retries} attempts: {last_error}")
