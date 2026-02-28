"""Serialize cached enrichment payloads into overlay TTL files."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

from enrichment.identifier_parsing import parse_external_id_from_iri
from enrichment.wikidata_client import resolve_qid_labels

MSH = Namespace("http://example.org/ontologies/MovieSHACL3#")
logger = logging.getLogger(__name__)
_PERSON_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
_PERSON_MATCH_THRESHOLD = 0.90


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unknown"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_sentinel(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("error") == "not_found")


def _cache_date(path: Path) -> str | None:
    """Return ISO date (YYYY-MM-DD) of a cache file's last modification."""
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _wd_entity_labels(entity: dict[str, Any], prop_id: str) -> list[str]:
    """Extract English labels for entity-reference claims from a raw Wikidata entity."""
    labels: list[str] = []
    for claim in entity.get("claims", {}).get(prop_id, []):
        ms = claim.get("mainsnak", {})
        dv = ms.get("datavalue", {})
        if dv.get("type") != "wikibase-entityid":
            continue
        qid = dv.get("value", {}).get("id")
        if not qid:
            continue
        # Wikidata entity JSON doesn't include labels for referenced entities,
        # but some claim formats include them.  Fall back to the Q-ID.
        labels.append(qid)
    return labels


def _wd_string_values(entity: dict[str, Any], prop_id: str) -> list[str]:
    """Extract string/monolingualtext values from a raw Wikidata entity."""
    values: list[str] = []
    for claim in entity.get("claims", {}).get(prop_id, []):
        ms = claim.get("mainsnak", {})
        dv = ms.get("datavalue", {})
        if dv.get("type") == "string":
            val = str(dv.get("value", "")).strip()
            if val:
                values.append(val)
        elif dv.get("type") == "monolingualtext":
            val = str(dv.get("value", {}).get("text", "")).strip()
            if val:
                values.append(val)
    return values


def _parse_wikidata_entity(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Parse a Wikidata entity payload into enrichment fields.

    Supports both the new full-entity format (from Entity Data API) and the
    legacy parsed SPARQL format (for backward compatibility with old caches).
    """
    # Legacy SPARQL format — already has resolved labels
    if payload.get("source") == "wikidata" and "claims" not in payload:
        return {
            "festival_awards": payload.get("festival_awards") or [],
            "country_of_origin": payload.get("country_of_origin") or [],
            "distributors": payload.get("distributors") or [],
        }

    # New entity format — extract Q-IDs (label resolution happens separately)
    if "claims" not in payload:
        return {"festival_awards": [], "country_of_origin": [], "distributors": []}

    # P166 = award received, P1411 = nominated for
    award_qids = _wd_entity_labels(payload, "P166") + _wd_entity_labels(payload, "P1411")
    country_qids = _wd_entity_labels(payload, "P495")
    distributor_qids = _wd_entity_labels(payload, "P750")

    return {
        "festival_awards": sorted(set(award_qids)),
        "country_of_origin": sorted(set(country_qids)),
        "distributors": sorted(set(distributor_qids)),
        # Raw Q-IDs — task 102 should add batch label resolution
        "_needs_label_resolution": True,
    }


def _emit_wikidata_triples(
    graph: Graph, film_uri: URIRef, wd: dict[str, list[str]]
) -> None:
    """Add Wikidata-sourced triples for a film to the graph."""
    for award_name in wd.get("festival_awards", []):
        if award_name:
            graph.add((film_uri, MSH.festivalAward, Literal(award_name, datatype=XSD.string)))
    if not list(graph.objects(film_uri, MSH.country)):
        for country_name in wd.get("country_of_origin", []):
            if country_name:
                graph.add((film_uri, MSH.country, Literal(country_name, datatype=XSD.string)))
    for distributor_name in wd.get("distributors", []):
        if distributor_name:
            graph.add((film_uri, MSH.distributor, Literal(distributor_name, datatype=XSD.string)))


def _normalize_person_name_for_match(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2018\u2019\u201c\u201d\"']", "", text)
    text = re.sub(r"\((?:\"[^\"]+\"|[^)]+)\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(token for token in text.split() if token)


def _remove_person_suffix(normalized_name: str) -> str:
    tokens = normalized_name.split()
    while tokens and tokens[-1] in _PERSON_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _person_name_index(foundation_graph: Graph) -> list[dict[str, str | int]]:
    nomination_counts: dict[str, int] = {}
    for _, _, person_uri in foundation_graph.triples((None, MSH.hasNominee, None)):
        person_key = str(person_uri)
        nomination_counts[person_key] = nomination_counts.get(person_key, 0) + 1

    rows: list[dict[str, str | int]] = []
    for person_uri in set(foundation_graph.subjects(RDF.type, MSH.Person)):
        raw_name_obj = next(foundation_graph.objects(person_uri, MSH.fullName), None)
        raw_name = str(raw_name_obj).strip() if raw_name_obj is not None else ""
        normalized_name = _normalize_person_name_for_match(raw_name)
        if not normalized_name:
            continue
        rows.append(
            {
                "uri": str(person_uri),
                "name": raw_name,
                "nom_count": nomination_counts.get(str(person_uri), 0),
                "normalized": normalized_name,
                "normalized_no_suffix": _remove_person_suffix(normalized_name),
            }
        )
    return rows


def _resolve_person_uri(
    *,
    person_name: str,
    person_id: str,
    canonical_people: list[dict[str, str | int]],
    exact_index: dict[str, list[dict[str, str | int]]],
    exact_no_suffix_index: dict[str, list[dict[str, str | int]]],
    first_token_index: dict[str, list[dict[str, str | int]]],
    last_token_index: dict[str, list[dict[str, str | int]]],
    resolution_cache: dict[tuple[str, str], tuple[URIRef, bool]],
) -> tuple[URIRef, bool]:
    normalized = _normalize_person_name_for_match(person_name)
    normalized_no_suffix = _remove_person_suffix(normalized)
    cache_key = (normalized, person_id)
    cached = resolution_cache.get(cache_key)
    if cached is not None:
        return cached

    def _pick_best(candidates: list[dict[str, str | int]]) -> dict[str, str | int]:
        return sorted(candidates, key=lambda item: (-int(item["nom_count"]), str(item["uri"])))[0]

    for key, index in ((normalized, exact_index), (normalized_no_suffix, exact_no_suffix_index)):
        if not key:
            continue
        matched = index.get(key)
        if matched:
            best = _pick_best(matched)
            resolved = URIRef(str(best["uri"])), True
            resolution_cache[cache_key] = resolved
            return resolved

    first_token = normalized.split()[0] if normalized else ""
    last_token = normalized.split()[-1] if normalized else ""
    candidates_by_first = first_token_index.get(first_token) if first_token else None
    candidates_by_last = last_token_index.get(last_token) if last_token else None
    if candidates_by_first and candidates_by_last:
        by_first_uri = {str(candidate["uri"]): candidate for candidate in candidates_by_first}
        intersection = [candidate for candidate in candidates_by_last if str(candidate["uri"]) in by_first_uri]
        candidates = intersection if intersection else (
            candidates_by_last if len(candidates_by_last) < len(candidates_by_first) else candidates_by_first
        )
    elif candidates_by_first:
        candidates = candidates_by_first
    elif candidates_by_last:
        candidates = candidates_by_last
    else:
        # No shared first or last name token — extremely unlikely to match at
        # 0.90 threshold.  Skip expensive fuzzy scan over all 8k+ canonical
        # entries and go straight to fallback.
        candidates = []

    best_candidate: dict[str, str | int] | None = None
    best_score = 0.0
    if normalized:
        for candidate in candidates:
            candidate_norm = str(candidate["normalized"])
            candidate_norm_no_suffix = str(candidate["normalized_no_suffix"])
            score = SequenceMatcher(a=normalized, b=candidate_norm).ratio()
            if normalized_no_suffix and candidate_norm_no_suffix:
                score = max(score, SequenceMatcher(a=normalized_no_suffix, b=candidate_norm_no_suffix).ratio())
            if score > best_score:
                best_score = score
                best_candidate = candidate
            elif score == best_score and best_candidate is not None:
                if int(candidate["nom_count"]) > int(best_candidate["nom_count"]):
                    best_candidate = candidate

    if best_candidate is not None and best_score >= _PERSON_MATCH_THRESHOLD:
        resolved = URIRef(str(best_candidate["uri"])), True
        resolution_cache[cache_key] = resolved
        return resolved

    fallback_uri = URIRef(f"{MSH}Person_tmdb_{person_id}")
    resolved = fallback_uri, False
    resolution_cache[cache_key] = resolved
    return resolved


def _film_index(foundation_graph: Graph) -> list[dict[str, str | None]]:
    query = """
    SELECT ?film ?identifier WHERE {
      ?film a <http://example.org/ontologies/MovieSHACL3#Film> .
      OPTIONAL { ?film <http://example.org/ontologies/MovieSHACL3#hasIdentifier> ?identifier . }
    }
    """
    by_film: dict[str, dict[str, str | None]] = {}
    for row in foundation_graph.query(query):
        film_uri = str(row.film)
        entry = by_film.setdefault(
            film_uri,
            {
                "film": film_uri,
                "tmdbId": None,
                "imdbId": None,
                "wikidataId": None,
            },
        )
        identifier_iri = str(row.identifier) if row.identifier else None
        if not identifier_iri:
            continue
        parsed = parse_external_id_from_iri(identifier_iri)
        if parsed is None:
            continue
        provider, external_id = parsed
        if provider == "tmdb" and entry["tmdbId"] is None:
            entry["tmdbId"] = external_id
        elif provider == "imdb" and entry["imdbId"] is None:
            entry["imdbId"] = external_id
        elif provider == "wikidata" and entry["wikidataId"] is None:
            entry["wikidataId"] = external_id
    return [by_film[key] for key in sorted(by_film)]


def generate_film_metadata_ttl(
    foundation_graph: Graph,
    output_path: Path,
    cache_dir: Path,
) -> int:
    graph = Graph()
    graph.bind("msh", MSH)
    graph.bind("xsd", XSD)
    graph.bind("rdfs", RDFS)

    enriched_films: set[str] = set()
    pending_wikidata: list[tuple[URIRef, dict[str, Any]]] = []
    for entry in _film_index(foundation_graph):
        film_uri = URIRef(entry["film"])
        tmdb_payload = _read_json(cache_dir / "tmdb" / f"{entry['tmdbId']}.json") if entry["tmdbId"] else None
        omdb_payload = _read_json(cache_dir / "omdb" / f"{entry['imdbId']}.json") if entry["imdbId"] else None

        wikidata_payload = None
        wikidata_cache_key = None
        if entry["wikidataId"]:
            wikidata_cache_key = entry["wikidataId"]
            wikidata_payload = _read_json(cache_dir / "wikidata" / f"{entry['wikidataId']}.json")
        elif entry["imdbId"]:
            wikidata_cache_key = entry["imdbId"]
            wikidata_payload = _read_json(cache_dir / "wikidata" / f"{entry['imdbId']}.json")

        if _is_sentinel(tmdb_payload):
            tmdb_payload = None
        if _is_sentinel(omdb_payload):
            omdb_payload = None
        if _is_sentinel(wikidata_payload):
            wikidata_payload = None
        if not tmdb_payload and not omdb_payload and not wikidata_payload:
            continue

        # Determine enrichment date from cache file modification times
        dates: list[str] = []
        if tmdb_payload and entry["tmdbId"]:
            d = _cache_date(cache_dir / "tmdb" / f"{entry['tmdbId']}.json")
            if d:
                dates.append(d)
        if omdb_payload and entry["imdbId"]:
            d = _cache_date(cache_dir / "omdb" / f"{entry['imdbId']}.json")
            if d:
                dates.append(d)
        if wikidata_payload and wikidata_cache_key:
            d = _cache_date(cache_dir / "wikidata" / f"{wikidata_cache_key}.json")
            if d:
                dates.append(d)
        if dates:
            graph.add((film_uri, MSH.enrichedAt, Literal(max(dates), datatype=XSD.date)))

        if tmdb_payload:
            budget = _safe_int(tmdb_payload.get("budget"))
            if budget is not None:
                graph.add((film_uri, MSH.budget, Literal(budget, datatype=XSD.integer)))
            revenue = _safe_int(tmdb_payload.get("revenue"))
            if revenue is not None:
                graph.add((film_uri, MSH.revenue, Literal(revenue, datatype=XSD.integer)))
            runtime = _safe_int(tmdb_payload.get("runtime"))
            if runtime is not None:
                graph.add((film_uri, MSH.runtime, Literal(runtime, datatype=XSD.integer)))
            rating = _safe_float(tmdb_payload.get("vote_average"))
            if rating is not None:
                graph.add((film_uri, MSH.tmdbRating, Literal(rating, datatype=XSD.float)))
            popularity = _safe_float(tmdb_payload.get("popularity"))
            if popularity is not None:
                graph.add((film_uri, MSH.tmdbPopularity, Literal(popularity, datatype=XSD.float)))
            vote_count = _safe_int(tmdb_payload.get("vote_count"))
            if vote_count is not None:
                graph.add((film_uri, MSH.voteCount, Literal(vote_count, datatype=XSD.integer)))
            original_language = tmdb_payload.get("original_language")
            if original_language:
                graph.add((film_uri, MSH.originalLanguage, Literal(str(original_language), datatype=XSD.string)))
            release_date = str(tmdb_payload.get("release_date") or "").strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", release_date):
                graph.add((film_uri, MSH.releaseDate, Literal(release_date, datatype=XSD.date)))
            tagline = str(tmdb_payload.get("tagline") or "").strip()
            if tagline:
                graph.add((film_uri, MSH.tagline, Literal(tagline, datatype=XSD.string)))
            production_companies = tmdb_payload.get("production_companies") or []
            for company in production_companies:
                if isinstance(company, dict):
                    name = str((company or {}).get("name") or "").strip()
                else:
                    name = str(company).strip()
                if not name:
                    continue
                graph.add((film_uri, MSH.productionCompany, Literal(name, datatype=XSD.string)))
            for genre in tmdb_payload.get("genres", []) or []:
                name = str(genre.get("name") or "").strip()
                if not name:
                    continue
                genre_uri = URIRef(f"{MSH}Genre_{_slugify(name)}")
                graph.add((genre_uri, RDF.type, MSH.Genre))
                graph.add((genre_uri, RDFS.label, Literal(name)))
                graph.add((film_uri, MSH.hasGenre, genre_uri))

        if omdb_payload:
            imdb_rating = _safe_float(omdb_payload.get("imdbRating"))
            if imdb_rating is not None:
                graph.add((film_uri, MSH.imdbRating, Literal(imdb_rating, datatype=XSD.float)))
            rated = str(omdb_payload.get("Rated") or "").strip()
            if rated and rated != "N/A":
                graph.add((film_uri, MSH.mpaaRating, Literal(rated, datatype=XSD.string)))
            for rating_item in omdb_payload.get("Ratings", []) or []:
                source = str(rating_item.get("Source") or "").strip().lower()
                value = str(rating_item.get("Value") or "").strip()
                if source == "rotten tomatoes" and value.endswith("%"):
                    parsed = _safe_int(value.rstrip("%"))
                    if parsed is not None:
                        graph.add((film_uri, MSH.rottenTomatoesScore, Literal(parsed, datatype=XSD.integer)))
                if source == "metacritic" and "/" in value:
                    parsed = _safe_int(value.split("/", 1)[0])
                    if parsed is not None:
                        graph.add((film_uri, MSH.metacriticScore, Literal(parsed, datatype=XSD.integer)))

        if wikidata_payload:
            wd_parsed = _parse_wikidata_entity(wikidata_payload)
            if wd_parsed.get("_needs_label_resolution"):
                # Collect Q-IDs for batch resolution after the loop
                pending_wikidata.append((film_uri, wd_parsed))
            else:
                # Legacy format — already has labels
                _emit_wikidata_triples(graph, film_uri, wd_parsed)

        enriched_films.add(entry["film"])

    # Batch-resolve Wikidata Q-IDs to labels, then emit triples
    if pending_wikidata:
        all_qids: set[str] = set()
        for _, wd_parsed in pending_wikidata:
            for key in ("festival_awards", "country_of_origin", "distributors"):
                for val in wd_parsed.get(key, []):
                    if re.fullmatch(r"Q\d+", val):
                        all_qids.add(val)
        label_cache: dict[str, str] = {}
        if all_qids:
            logger.info("Resolving %d Wikidata Q-IDs to labels...", len(all_qids))
            label_cache = resolve_qid_labels(sorted(all_qids), label_cache=label_cache)
        for film_uri, wd_parsed in pending_wikidata:
            resolved = {
                key: [label_cache.get(v, v) for v in wd_parsed.get(key, [])]
                for key in ("festival_awards", "country_of_origin", "distributors")
            }
            _emit_wikidata_triples(graph, film_uri, resolved)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=output_path, format="turtle")
    return len(enriched_films)


def generate_film_credits_ttl(
    foundation_graph: Graph,
    output_path: Path,
    person_output_path: Path,
    cache_dir: Path,
) -> tuple[int, int]:
    credits_graph = Graph()
    people_graph = Graph()
    for graph in (credits_graph, people_graph):
        graph.bind("msh", MSH)
        graph.bind("xsd", XSD)
        graph.bind("rdfs", RDFS)

    film_count = 0
    person_count = 0
    seen_films: set[str] = set()
    seen_people: set[str] = set()
    unresolved_people: set[tuple[str, str]] = set()
    canonical_people = _person_name_index(foundation_graph)
    exact_index: dict[str, list[dict[str, str | int]]] = {}
    exact_no_suffix_index: dict[str, list[dict[str, str | int]]] = {}
    first_token_index: dict[str, list[dict[str, str | int]]] = {}
    last_token_index: dict[str, list[dict[str, str | int]]] = {}
    resolution_cache: dict[tuple[str, str], tuple[URIRef, bool]] = {}
    for candidate in canonical_people:
        normalized = str(candidate["normalized"])
        normalized_no_suffix = str(candidate["normalized_no_suffix"])
        exact_index.setdefault(normalized, []).append(candidate)
        if normalized_no_suffix:
            exact_no_suffix_index.setdefault(normalized_no_suffix, []).append(candidate)
        if normalized:
            tokens = normalized.split()
            first_token_index.setdefault(tokens[0], []).append(candidate)
            last_token_index.setdefault(tokens[-1], []).append(candidate)

    for entry in _film_index(foundation_graph):
        if not entry["tmdbId"]:
            continue
        tmdb_payload = _read_json(cache_dir / "tmdb" / f"{entry['tmdbId']}.json")
        if not tmdb_payload or _is_sentinel(tmdb_payload):
            continue
        credits = tmdb_payload.get("credits") or {}
        crew = credits.get("crew") or []
        cast = credits.get("cast") or []
        film_uri = URIRef(entry["film"])

        has_credit = False
        for crew_entry in crew:
            person_id = crew_entry.get("id")
            if person_id is None:
                continue
            person_id_str = str(person_id).strip()
            if not person_id_str:
                continue
            name = str(crew_entry.get("name") or "").strip()
            job = str(crew_entry.get("job") or "").strip().lower()
            person_uri, matched = _resolve_person_uri(
                person_name=name,
                person_id=person_id_str,
                canonical_people=canonical_people,
                exact_index=exact_index,
                exact_no_suffix_index=exact_no_suffix_index,
                first_token_index=first_token_index,
                last_token_index=last_token_index,
                resolution_cache=resolution_cache,
            )
            predicate = None
            if job == "director":
                predicate = MSH.hasDirector
            elif job in {"writer", "screenplay", "story"}:
                predicate = MSH.hasWriter
            elif job in {"producer", "executive producer", "co-producer"}:
                predicate = MSH.hasProducer
            elif job in {"director of photography", "cinematography"}:
                predicate = MSH.hasCinematographer
            elif job in {"original music composer", "composer"}:
                predicate = MSH.hasComposer
            if predicate is None:
                continue
            credits_graph.add((film_uri, predicate, person_uri))
            has_credit = True
            if str(person_uri) not in seen_people:
                seen_people.add(str(person_uri))
                people_graph.add((person_uri, RDF.type, MSH.Person))
                if not matched and name:
                    people_graph.add((person_uri, MSH.fullName, Literal(name, datatype=XSD.string)))
                    people_graph.add((person_uri, RDFS.label, Literal(name)))
                person_count += 1
            people_graph.add((person_uri, MSH.tmdbPersonId, Literal(person_id_str, datatype=XSD.string)))
            if not matched:
                unresolved_people.add((person_id_str, name))

        top_cast = sorted(cast, key=lambda c: c.get("order", 9999))[:10]
        for cast_entry in top_cast:
            person_id = cast_entry.get("id")
            if person_id is None:
                continue
            person_id_str = str(person_id).strip()
            if not person_id_str:
                continue
            name = str(cast_entry.get("name") or "").strip()
            person_uri, matched = _resolve_person_uri(
                person_name=name,
                person_id=person_id_str,
                canonical_people=canonical_people,
                exact_index=exact_index,
                exact_no_suffix_index=exact_no_suffix_index,
                first_token_index=first_token_index,
                last_token_index=last_token_index,
                resolution_cache=resolution_cache,
            )
            credits_graph.add((film_uri, MSH.hasActor, person_uri))
            has_credit = True
            if str(person_uri) not in seen_people:
                seen_people.add(str(person_uri))
                people_graph.add((person_uri, RDF.type, MSH.Person))
                if not matched and name:
                    people_graph.add((person_uri, MSH.fullName, Literal(name, datatype=XSD.string)))
                    people_graph.add((person_uri, RDFS.label, Literal(name)))
                person_count += 1
            people_graph.add((person_uri, MSH.tmdbPersonId, Literal(person_id_str, datatype=XSD.string)))
            if not matched:
                unresolved_people.add((person_id_str, name))

        if has_credit:
            seen_films.add(entry["film"])
            film_count = len(seen_films)

    resolved_count = person_count - len(unresolved_people)
    logger.info(
        "Credits reconciliation: resolved=%d unresolved=%d total=%d (%.1f%% match rate)",
        resolved_count,
        len(unresolved_people),
        person_count,
        (resolved_count / person_count * 100) if person_count else 0,
    )
    if unresolved_people:
        sample = ", ".join(f"{name}({pid})" for pid, name in sorted(unresolved_people)[:10])
        logger.info(
            "Credits reconciliation unresolved sample=[%s]",
            sample,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    credits_graph.serialize(destination=output_path, format="turtle")
    people_graph.serialize(destination=person_output_path, format="turtle")
    return film_count, person_count
