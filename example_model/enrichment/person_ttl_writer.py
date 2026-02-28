"""Serializer for person enrichment TTL from TMDB person cache."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

MSH = Namespace("http://example.org/ontologies/MovieSHACL3#")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unknown"


def _person_name_index(foundation_graph: Graph) -> dict[str, list[tuple[str, int]]]:
    query = """
    SELECT ?person ?name (COUNT(?nom) AS ?nomCount) WHERE {
      ?person a <http://example.org/ontologies/MovieSHACL3#Person> ;
              <http://example.org/ontologies/MovieSHACL3#fullName> ?name .
      OPTIONAL {
        ?nom a <http://example.org/ontologies/MovieSHACL3#Nomination> ;
             <http://example.org/ontologies/MovieSHACL3#hasNominee> ?person .
      }
    }
    GROUP BY ?person ?name
    """
    index: dict[str, list[tuple[str, int]]] = {}
    for row in foundation_graph.query(query):
        key = str(row.name).strip().lower()
        if not key:
            continue
        index.setdefault(key, []).append((str(row.person), int(row.nomCount)))
    for key in index:
        index[key].sort(key=lambda item: item[1], reverse=True)
    return index


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_person_metadata_ttl(
    foundation_graph: Graph,
    output_path: Path,
    cache_dir: Path,
) -> int:
    graph = Graph()
    graph.bind("msh", MSH)
    graph.bind("xsd", XSD)
    graph.bind("rdfs", RDFS)

    name_index = _person_name_index(foundation_graph)
    count = 0
    tmdb_dir = cache_dir / "tmdb"
    for path in sorted(tmdb_dir.glob("person_*.json")):
        payload = _read_json(path)
        if not payload or payload.get("error") == "not_found":
            continue

        person_id = str(payload.get("id") or "").strip()
        name = str(payload.get("name") or "").strip()
        if not person_id or not name:
            continue

        matches = name_index.get(name.lower(), [])
        if matches:
            person_uri = URIRef(matches[0][0])
        else:
            person_uri = URIRef(f"{MSH}Person_tmdb_{_slugify(person_id)}")

        graph.add((person_uri, RDF.type, MSH.Person))
        graph.add((person_uri, MSH.tmdbPersonId, Literal(person_id, datatype=XSD.string)))
        graph.add((person_uri, MSH.fullName, Literal(name, datatype=XSD.string)))
        graph.add((person_uri, RDFS.label, Literal(name)))

        birthday = str(payload.get("birthday") or "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", birthday):
            graph.add((person_uri, MSH.birthYear, Literal(int(birthday[:4]), datatype=XSD.integer)))

        birth_place = str(payload.get("place_of_birth") or "").strip()
        if birth_place:
            graph.add((person_uri, MSH.birthPlace, Literal(birth_place, datatype=XSD.string)))

        biography = str(payload.get("biography") or "").strip()
        if biography:
            graph.add((person_uri, MSH.biography, Literal(biography[:500], datatype=XSD.string)))

        count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=output_path, format="turtle")
    return count
