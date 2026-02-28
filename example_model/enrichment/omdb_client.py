"""OMDB client with local raw-response caching and throttling."""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Any, Callable

import requests

from enrichment.cache import read_cache, write_cache

logger = logging.getLogger(__name__)
_OMDB_THROTTLE_SECONDS = 0.5  # <= 2 req/sec
_last_call = 0.0
_daily_count = 0
_daily_date = date.today()
MAX_DAILY = 950


def _throttle() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _OMDB_THROTTLE_SECONDS:
        time.sleep(_OMDB_THROTTLE_SECONDS - elapsed)
    _last_call = time.time()


def _check_daily_limit() -> None:
    global _daily_count, _daily_date
    today = date.today()
    if today != _daily_date:
        _daily_date = today
        _daily_count = 0
    _daily_count += 1
    if _daily_count > MAX_DAILY:
        raise RuntimeError(f"OMDB daily limit ({MAX_DAILY}) reached. Re-run tomorrow.")


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("OMDB_API_KEY")
    if not key:
        raise RuntimeError(
            "OMDB_API_KEY environment variable is not set. "
            "Get a key at https://www.omdbapi.com/apikey.aspx"
        )
    return key


def fetch_omdb_movie(
    imdb_id: str,
    *,
    force: bool = False,
    api_key: str | None = None,
    max_retries: int = 3,
    request_fn: Callable[..., requests.Response] | None = None,
) -> dict[str, Any]:
    entity_id = str(imdb_id).strip()
    if not entity_id:
        raise ValueError("imdb_id must be non-empty")

    if not force:
        cached = read_cache("omdb", entity_id)
        if cached is not None:
            return cached

    key = _resolve_api_key(api_key)
    requester = request_fn or requests.get
    url = "https://www.omdbapi.com/"

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        _check_daily_limit()
        _throttle()
        try:
            response = requester(url, params={"i": entity_id, "apikey": key}, timeout=30)
        except requests.RequestException as exc:  # pragma: no cover - exercised via mock side-effects
            last_error = exc
            logger.warning("OMDB request failed for %s (attempt %d/%d): %s", entity_id, attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
            break

        if response.status_code == 200:
            payload = response.json()
            if str(payload.get("Response", "True")).lower() == "false":
                error_msg = str(payload.get("Error", "")).lower()
                if "not found" in error_msg:
                    sentinel = {"error": "not_found", "source": "omdb", "id": entity_id}
                    write_cache("omdb", entity_id, sentinel)
                    return sentinel
            write_cache("omdb", entity_id, payload)
            return payload

        if response.status_code == 404:
            sentinel = {"error": "not_found", "source": "omdb", "id": entity_id}
            write_cache("omdb", entity_id, sentinel)
            return sentinel

        if response.status_code == 429 and attempt < max_retries:
            time.sleep(1.0)
            continue

        if response.status_code >= 500 and attempt < max_retries:
            time.sleep(0.5 * attempt)
            continue

        raise RuntimeError(f"OMDB request failed for {entity_id}: {response.status_code} {response.text[:200]}")

    raise RuntimeError(f"OMDB request failed for {entity_id} after {max_retries} attempts: {last_error}")
