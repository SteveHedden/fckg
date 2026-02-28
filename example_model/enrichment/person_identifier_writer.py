"""Writer for person external identifier enrichment overlays."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

from enrichment.cache import read_cache, write_cache

MSH = Namespace("http://example.org/ontologies/MovieSHACL3#")
logger = logging.getLogger(__name__)
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
TMDB_PERSON_BASE = "https://www.themoviedb.org/person/"
WIKIDATA_ENTITY_BASE = "https://www.wikidata.org/wiki/"
IMDB_PERSON_PATH_RE = re.compile(r"^/name/(nm\d+)$", re.IGNORECASE)
WIKIDATA_ENTITY_RE = re.compile(r"/(Q\d+)$", re.IGNORECASE)
TMDB_PERSON_URI_RE = re.compile(r"Person_tmdb_(\d+)$", re.IGNORECASE)


def _extract_imdb_person_id(identifier_iri: str) -> str | None:
    text = str(identifier_iri or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    if "imdb.com" not in parsed.netloc.lower():
        return None
    path = parsed.path.rstrip("/") or parsed.path
    match = IMDB_PERSON_PATH_RE.match(path)
    if match is None:
        return None
    return match.group(1).lower()


def _extract_wikidata_qid(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"Q\d+", text, flags=re.IGNORECASE):
        return text.upper()
    match = WIKIDATA_ENTITY_RE.search(text.rstrip("/"))
    if match is None:
        return None
    return match.group(1).upper()


def _extract_tmdb_person_id_from_person_uri(person_uri: URIRef) -> str | None:
    match = TMDB_PERSON_URI_RE.search(str(person_uri).strip())
    if match is None:
        return None
    return match.group(1)


def _add_identifier(
    graph: Graph,
    *,
    person_uri: URIRef,
    scheme_uri: URIRef,
    scheme_label: str,
    value: str,
    iri_base: str,
) -> None:
    identifier_uri = URIRef(f"{iri_base}{value}")
    graph.add((scheme_uri, RDF.type, MSH.IdentifierScheme))
    graph.add((scheme_uri, RDFS.label, Literal(scheme_label)))
    graph.add((identifier_uri, RDF.type, MSH.Identifier))
    graph.add((identifier_uri, MSH.scheme, scheme_uri))
    graph.add((identifier_uri, MSH.value, Literal(value, datatype=XSD.string)))
    graph.add((identifier_uri, RDFS.label, Literal(f"{scheme_label}:{value}")))
    graph.add((person_uri, MSH.hasIdentifier, identifier_uri))


class WikidataLookupClient:
    """IMDb->Wikidata resolver with cache + throttling."""

    def __init__(
        self,
        *,
        request_fn: Callable[..., requests.Response] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
        min_interval_seconds: float = 1.0,
        cache_only: bool = False,
    ) -> None:
        self._request_fn = request_fn or requests.get
        self._sleep_fn = sleep_fn or time.sleep
        self._time_fn = time_fn or time.monotonic
        self._min_interval = max(0.0, float(min_interval_seconds))
        self._cache_only = cache_only
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._time_fn() - self._last_request_at
        if elapsed < self._min_interval:
            self._sleep_fn(self._min_interval - elapsed)

    def lookup_by_imdb(self, imdb_person_id: str) -> str | None:
        imdb_id = str(imdb_person_id or "").strip().lower()
        if not imdb_id:
            return None

        cached = read_cache("wikidata", imdb_id)
        if cached is not None:
            qid = _extract_wikidata_qid(cached.get("wikidata_id") or cached.get("qid") or "")
            return qid
        if self._cache_only:
            return None

        query = f"""
        SELECT ?person WHERE {{
          ?person wdt:P345 "{imdb_id}" .
        }}
        LIMIT 1
        """
        self._throttle()
        try:
            response = self._request_fn(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json", "User-Agent": "fckg-enrichment/1.0"},
                timeout=30,
            )
        except requests.RequestException as exc:
            logger.warning("Wikidata lookup failed for %s: %s", imdb_id, exc)
            write_cache("wikidata", imdb_id, {"error": "request_failed", "imdb_id": imdb_id})
            return None
        finally:
            self._last_request_at = self._time_fn()

        if response.status_code != 200:
            logger.warning("Wikidata lookup non-200 for %s: %s", imdb_id, response.status_code)
            write_cache("wikidata", imdb_id, {"error": "http_error", "status_code": response.status_code, "imdb_id": imdb_id})
            return None

        payload = response.json()
        bindings = payload.get("results", {}).get("bindings", [])
        if not bindings:
            write_cache("wikidata", imdb_id, {"error": "not_found", "imdb_id": imdb_id})
            return None

        qid = _extract_wikidata_qid(bindings[0].get("person", {}).get("value", ""))
        if qid is None:
            write_cache("wikidata", imdb_id, {"error": "invalid_response", "imdb_id": imdb_id})
            return None
        write_cache("wikidata", imdb_id, {"imdb_id": imdb_id, "wikidata_id": qid})
        return qid


def generate_person_identifiers_ttl(
    foundation_graph: Graph,
    *,
    person_metadata_path: Path,
    output_path: Path,
    wikidata_client: WikidataLookupClient | None = None,
    cache_only: bool = False,
) -> tuple[int, int]:
    """Write person identifier overlay and return (tmdb_count, wikidata_count)."""
    graph = Graph()
    graph.bind("msh", MSH)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)

    metadata_graph = Graph()
    if person_metadata_path.exists():
        metadata_graph.parse(person_metadata_path, format="turtle")

    tmdb_count = 0
    wikidata_count = 0
    tmdb_scheme_uri = MSH.tmdb
    wikidata_scheme_uri = MSH.wikidata

    tmdb_by_person: dict[URIRef, str] = {}
    for person_uri, tmdb_value in metadata_graph.subject_objects(MSH.tmdbPersonId):
        tmdb_id = str(tmdb_value).strip()
        if not tmdb_id:
            continue
        tmdb_by_person[URIRef(person_uri)] = tmdb_id

    # Backfill support: older person overlays used Person_tmdb_<id> URIs without
    # emitting explicit msh:tmdbPersonId literals.
    for person_uri in set(metadata_graph.subjects(RDF.type, MSH.Person)):
        person_ref = URIRef(person_uri)
        if person_ref in tmdb_by_person:
            continue
        tmdb_id = _extract_tmdb_person_id_from_person_uri(person_ref)
        if tmdb_id is None:
            continue
        tmdb_by_person[person_ref] = tmdb_id

    for person_uri, tmdb_id in sorted(tmdb_by_person.items(), key=lambda item: str(item[0])):
        _add_identifier(
            graph,
            person_uri=person_uri,
            scheme_uri=tmdb_scheme_uri,
            scheme_label="tmdb",
            value=tmdb_id,
            iri_base=TMDB_PERSON_BASE,
        )
        tmdb_count += 1

    imdb_by_person: dict[URIRef, str] = {}
    for person_uri in set(foundation_graph.subjects(RDF.type, MSH.Person)):
        for identifier in foundation_graph.objects(person_uri, MSH.hasIdentifier):
            imdb_id = _extract_imdb_person_id(str(identifier))
            if imdb_id is None:
                continue
            imdb_by_person[URIRef(person_uri)] = imdb_id
            break

    client = wikidata_client or WikidataLookupClient(cache_only=cache_only)
    for person_uri, imdb_id in sorted(imdb_by_person.items(), key=lambda item: str(item[0])):
        qid = client.lookup_by_imdb(imdb_id)
        if qid is None:
            continue
        _add_identifier(
            graph,
            person_uri=person_uri,
            scheme_uri=wikidata_scheme_uri,
            scheme_label="wikidata",
            value=qid,
            iri_base=WIKIDATA_ENTITY_BASE,
        )
        wikidata_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=output_path, format="turtle")
    return tmdb_count, wikidata_count
