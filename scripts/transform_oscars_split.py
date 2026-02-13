"""Transform OSCARS clean CSV into ontology-aligned split TTL instance files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.transformation import infer_nominee_type

MSH = Namespace("http://example.org/ontologies/MovieSHACL3#")
SCHEMA = Namespace("https://schema.org/")
SYSTEM_CODE = "OSCARS"
SYSTEM_CODE_LOWER = SYSTEM_CODE.lower()
SYSTEM_NAME = "Academy Awards"
SYSTEM_SHORT_NAME = "Oscars"
SYSTEM_URI = URIRef(f"{MSH}AwardSystem_oscars")

WIKIDATA_BASE = "https://www.wikidata.org/wiki/"
IMDB_TITLE_BASE = "https://www.imdb.com/title/"
IMDB_NAME_BASE = "https://www.imdb.com/name/"
TMDB_MOVIE_BASE = "https://www.themoviedb.org/movie/"
LOCAL_FILM_ID_BASE = "urn:film:"
ACTING_CATEGORY_NAMES = {
    "actor",
    "actress",
    "actor in a leading role",
    "actress in a leading role",
    "actor in a supporting role",
    "actress in a supporting role",
}


@dataclass
class FilmEntity:
    uri: URIRef
    title: str
    release_year: int
    tmdb_ids: set[str] = field(default_factory=set)
    imdb_ids: set[str] = field(default_factory=set)
    wikidata_ids: set[str] = field(default_factory=set)


@dataclass
class PersonEntity:
    uri: URIRef
    full_name: str
    imdb_person_ids: set[str] = field(default_factory=set)


@dataclass
class RowNomination:
    ceremony_year: int
    ceremony_uri: URIRef
    category_key: str
    category_uri: URIRef
    category_name: str
    year_film: int
    film_key: str
    film_uri: URIRef | None = None
    canon_category_uri: URIRef | None = None
    winner: bool = False
    nominee_uri: URIRef | None = None
    nominee_id_token: str | None = None
    group_key: tuple[int, str, str, int] = (0, "", "", 0)


def slugify(value: str) -> str:
    value = str(value)
    value = value.strip().replace("'", "")
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def as_str(x: Any) -> str | None:
    if x is None:
        return None
    text = str(x).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    return text


def as_int(x: Any) -> int | None:
    text = as_str(x)
    if text is None:
        return None
    try:
        return int(text)
    except Exception:
        try:
            fx = float(text)
            if fx.is_integer():
                return int(fx)
        except Exception:
            return None
    return None


def normalize_id(x: Any) -> str | None:
    text = as_str(x)
    if text is None:
        return None
    try:
        fx = float(text)
        if fx.is_integer():
            return str(int(fx))
    except Exception:
        pass
    return text


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    text = as_str(x)
    if text is None:
        return False
    return text.lower() in {"true", "1", "yes", "y"}


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        delimiter = ","
        first_line = sample.splitlines()[0] if sample else ""
        if first_line.count("\t") > first_line.count(","):
            delimiter = "\t"
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                delimiter = dialect.delimiter
            except Exception:
                if "\t" in first_line and "," not in first_line:
                    delimiter = "\t"
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def _nomination_hash(
    *,
    system: str,
    ceremony_year: int,
    category: str,
    film: str,
    nominee_ids: list[str],
    winner: bool | None = None,
) -> str:
    key_parts = [
        system.strip().lower(),
        str(ceremony_year),
        category.strip().lower(),
        film.strip().lower(),
        "|".join(sorted(nominee_ids)),
        "winner:" + (str(winner).lower() if winner is not None else "none"),
    ]
    digest = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:16]
    return digest


def _is_acting_category(category_name: str) -> bool:
    return category_name.strip().casefold() in ACTING_CATEGORY_NAMES


def add_identifier(
    g: Graph,
    *,
    subject: URIRef,
    scheme: str,
    value: str | None,
    base_url: str,
    metadata_graph: Graph | None = None,
) -> None:
    if not value:
        return
    if metadata_graph is None:
        metadata_graph = g
    scheme_uri = MSH[scheme.lower()]
    metadata_graph.add((scheme_uri, RDF.type, MSH.IdentifierScheme))
    metadata_graph.add((scheme_uri, RDFS.label, Literal(scheme.lower())))
    ident_uri = URIRef(f"{base_url}{value}")
    metadata_graph.add((ident_uri, RDF.type, MSH.Identifier))
    metadata_graph.add((ident_uri, MSH.scheme, scheme_uri))
    metadata_graph.add((ident_uri, MSH.value, Literal(value)))
    metadata_graph.add((ident_uri, RDFS.label, Literal(f"{scheme.lower()}:{value}")))
    g.add((subject, MSH.hasIdentifier, ident_uri))
    g.add((subject, SCHEMA.identifier, ident_uri))


def _build_graphs(rows: list[dict[str, str]]) -> tuple[Graph, Graph, Graph, dict[str, int]]:
    nominations_graph = Graph()
    films_graph = Graph()
    people_graph = Graph()
    for graph in (nominations_graph, films_graph, people_graph):
        graph.bind("msh", MSH)
        graph.bind("rdfs", RDFS)
        graph.bind("xsd", XSD)

    films: dict[str, FilmEntity] = {}
    people: dict[str, PersonEntity] = {}
    ceremonies: dict[int, URIRef] = {}
    categories: dict[str, URIRef] = {}
    category_labels: dict[str, str] = {}
    row_nominations: list[RowNomination] = []
    ceremony_numbers: dict[int, int] = {}

    for row in rows:
        year_film = as_int(row.get("year_film") or row.get("year"))
        year_ceremony = as_int(row.get("year_ceremony") or row.get("year"))
        if year_film is None:
            year_film = year_ceremony
        if year_film is None or year_ceremony is None:
            continue

        ceremony_number = as_int(row.get("ceremony"))
        category = as_str(row.get("category"))
        canon_category = as_str(row.get("canon_category"))
        film_name = as_str(row.get("film"))
        person_name = as_str(row.get("name") or row.get("nominee"))
        if category is None:
            continue

        film_uri: URIRef | None = None
        if film_name:
            film_uri = URIRef(f"{MSH}Film_{slugify(film_name)}_{year_film}")
            film_key = str(film_uri)
            film = films.get(film_key)
            if film is None:
                film = FilmEntity(uri=film_uri, title=film_name, release_year=year_film)
                films[film_key] = film
            tmdb_id = normalize_id(row.get("tmdb_id"))
            imdb_id = normalize_id(row.get("imdb_id"))
            wikidata_id = normalize_id(row.get("wikidata_id"))
            if tmdb_id:
                film.tmdb_ids.add(tmdb_id)
            if imdb_id:
                film.imdb_ids.add(imdb_id)
            if wikidata_id:
                film.wikidata_ids.add(wikidata_id)

        person_uri: URIRef | None = None
        if person_name:
            person_uri = URIRef(f"{MSH}Person_{slugify(person_name)}")
            person_key = str(person_uri)
            person = people.get(person_key)
            if person is None:
                person = PersonEntity(uri=person_uri, full_name=person_name)
                people[person_key] = person
            nominee_imdb_id = normalize_id(row.get("nominee_imdb_id"))
            if nominee_imdb_id:
                person.imdb_person_ids.add(nominee_imdb_id)

        ceremony_uri = ceremonies.get(year_ceremony)
        if ceremony_uri is None:
            ceremony_uri = URIRef(f"{MSH}Ceremony_oscars_{year_ceremony}")
            ceremonies[year_ceremony] = ceremony_uri

        category_key = category.casefold()
        category_uri = URIRef(f"{MSH}Category_oscars_{slugify(category)}")
        category_uri_key = str(category_uri)
        categories[category_uri_key] = category_uri
        if category_uri_key not in category_labels:
            category_labels[category_uri_key] = category

        canon_uri: URIRef | None = None
        if canon_category:
            canon_uri = URIRef(f"{MSH}Category_oscars_{slugify(canon_category)}")
            canon_uri_key = str(canon_uri)
            categories[canon_uri_key] = canon_uri
            if canon_uri_key not in category_labels:
                category_labels[canon_uri_key] = canon_category

        film_key = (film_name or "").casefold()
        group_key = (year_ceremony, category_key, film_key, year_film)
        nominee_token = str(person_uri).split("#")[-1] if person_uri is not None else None
        row_nominations.append(
            RowNomination(
                ceremony_year=year_ceremony,
                ceremony_uri=ceremony_uri,
                category_key=category_key,
                category_uri=category_uri,
                category_name=category,
                year_film=year_film,
                film_key=film_key,
                film_uri=film_uri,
                canon_category_uri=canon_uri,
                winner=as_bool(row.get("winner")),
                nominee_uri=person_uri,
                nominee_id_token=nominee_token,
                group_key=group_key,
            )
        )

        if ceremony_number is not None:
            nominations_graph.add((ceremony_uri, MSH.ceremonyNumber, Literal(ceremony_number, datatype=XSD.integer)))
            ceremony_numbers.setdefault(year_ceremony, ceremony_number)

    nominations_graph.add((SYSTEM_URI, RDF.type, MSH.AwardSystem))
    nominations_graph.add((SYSTEM_URI, MSH.systemName, Literal(SYSTEM_NAME)))
    nominations_graph.add((SYSTEM_URI, MSH.shortName, Literal(SYSTEM_SHORT_NAME)))
    nominations_graph.add((SYSTEM_URI, RDFS.label, Literal(SYSTEM_CODE)))

    for ceremony_year, ceremony_uri in sorted(ceremonies.items()):
        nominations_graph.add((ceremony_uri, RDF.type, MSH.AwardCeremony))
        nominations_graph.add((ceremony_uri, MSH.yearCeremony, Literal(str(ceremony_year), datatype=XSD.gYear)))
        ceremony_number = ceremony_numbers.get(ceremony_year)
        if ceremony_number is not None:
            ceremony_name = f"{_ordinal(ceremony_number)} Academy Awards"
        else:
            ceremony_name = f"Academy Awards {ceremony_year}"
        nominations_graph.add((ceremony_uri, MSH.ceremonyName, Literal(ceremony_name)))
        nominations_graph.add((ceremony_uri, RDFS.label, Literal(ceremony_name)))
        nominations_graph.add((ceremony_uri, MSH.hasAwardSystem, SYSTEM_URI))

    for category_uri_key, category_uri in sorted(categories.items()):
        category_name = category_labels.get(category_uri_key, category_uri_key)
        nominations_graph.add((category_uri, RDF.type, MSH.AwardCategory))
        nominations_graph.add((category_uri, MSH.categoryName, Literal(category_name)))
        nominations_graph.add((category_uri, RDFS.label, Literal(category_name)))
        nominations_graph.add((category_uri, MSH.hasAwardSystem, SYSTEM_URI))

    nomination_uris: set[URIRef] = set()
    grouped_nominations: dict[tuple[Any, ...], list[RowNomination]] = {}
    for row_nom in row_nominations:
        key = (
            row_nom.ceremony_year,
            row_nom.category_key,
            row_nom.film_key,
            row_nom.nominee_id_token or "",
        )
        grouped_nominations.setdefault(key, []).append(row_nom)

    for key in sorted(grouped_nominations):
        group_rows = grouped_nominations[key]
        winner_groups: dict[bool, list[RowNomination]] = {}
        for row in group_rows:
            winner_groups.setdefault(row.winner, []).append(row)

        for winner_value in sorted(winner_groups):
            resolved_rows = winner_groups[winner_value]
            base = resolved_rows[0]
            nominee_ids_for_hash = sorted({n.nominee_id_token for n in resolved_rows if n.nominee_id_token})
            digest = _nomination_hash(
                system=SYSTEM_CODE,
                ceremony_year=base.ceremony_year,
                category=base.category_key,
                film=base.film_key,
                nominee_ids=nominee_ids_for_hash,
                winner=winner_value,
            )
            film_token = slugify(base.film_key or "none")
            category_token = slugify(base.category_key or "unknown")
            nomination_uri = URIRef(
                f"{MSH}Nomination_oscars_{base.ceremony_year}_{category_token}_{film_token}_{digest}"
            )
            nomination_uris.add(nomination_uri)

            nomination_nominee_uris = sorted({n.nominee_uri for n in resolved_rows if n.nominee_uri is not None}, key=str)
            nominee_count = len(nomination_nominee_uris)
            nominee_type = infer_nominee_type(base.category_name, nominee_count)
            film_uri = next((n.film_uri for n in resolved_rows if n.film_uri is not None), None)
            canon_category_uri = next((n.canon_category_uri for n in resolved_rows if n.canon_category_uri is not None), None)
            nominations_graph.add((nomination_uri, RDF.type, MSH.Nomination))
            nominations_graph.set((nomination_uri, MSH.hasCeremony, base.ceremony_uri))
            nominations_graph.set((nomination_uri, MSH.hasCategory, base.category_uri))
            nominations_graph.set((nomination_uri, MSH.yearFilm, Literal(str(base.year_film), datatype=XSD.gYear)))
            nominations_graph.set((nomination_uri, MSH.winner, Literal(winner_value, datatype=XSD.boolean)))
            nominations_graph.set((nomination_uri, MSH.nomineeType, Literal(nominee_type)))
            if film_uri is not None:
                nominations_graph.set((nomination_uri, MSH.hasFilm, film_uri))
            if canon_category_uri is not None:
                nominations_graph.set((nomination_uri, MSH.hasCanonCategory, canon_category_uri))
            for nominee_uri in nomination_nominee_uris:
                nominations_graph.add((nomination_uri, MSH.hasNominee, nominee_uri))

    for film in sorted(films.values(), key=lambda f: str(f.uri)):
        films_graph.add((film.uri, RDF.type, MSH.Film))
        films_graph.add((film.uri, MSH.title, Literal(film.title)))
        films_graph.add((film.uri, MSH.releaseYear, Literal(str(film.release_year), datatype=XSD.gYear)))

        tmdb_id = sorted(film.tmdb_ids)[0] if film.tmdb_ids else None
        imdb_id = sorted(film.imdb_ids)[0] if film.imdb_ids else None
        wikidata_id = sorted(film.wikidata_ids)[0] if film.wikidata_ids else None

        add_identifier(films_graph, subject=film.uri, scheme="tmdb", value=tmdb_id, base_url=TMDB_MOVIE_BASE)
        add_identifier(films_graph, subject=film.uri, scheme="imdb", value=imdb_id, base_url=IMDB_TITLE_BASE)
        add_identifier(films_graph, subject=film.uri, scheme="wikidata", value=wikidata_id, base_url=WIKIDATA_BASE)
        if not any((tmdb_id, imdb_id, wikidata_id)):
            film_local_id = str(film.uri).split("#")[-1]
            add_identifier(
                films_graph,
                subject=film.uri,
                scheme="msh-local-film",
                value=film_local_id,
                base_url=LOCAL_FILM_ID_BASE,
            )

    for person in sorted(people.values(), key=lambda p: str(p.uri)):
        people_graph.add((person.uri, RDF.type, MSH.Person))
        people_graph.add((person.uri, MSH.fullName, Literal(person.full_name)))
        people_graph.add((person.uri, RDFS.label, Literal(person.full_name)))
        for imdb_person_id in sorted(person.imdb_person_ids):
            add_identifier(
                people_graph,
                subject=person.uri,
                scheme="imdb",
                value=imdb_person_id,
                base_url=IMDB_NAME_BASE,
            )

    counts = {
        "rows": len(rows),
        "nominations": len(nomination_uris),
        "films": len(films),
        "persons": len(people),
    }
    return nominations_graph, films_graph, people_graph, counts


def run_transform_oscars_split(
    *,
    csv_path: Path,
    nominations_output: Path,
    films_output: Path,
    people_output: Path,
    limit: int,
    dry_run: bool,
    force: bool,
) -> dict[str, int]:
    rows = _read_csv_rows(csv_path)
    if limit > 0:
        rows = rows[:limit]

    nominations_graph, films_graph, people_graph, counts = _build_graphs(rows)

    if force and not dry_run:
        for path in (nominations_output, films_output, people_output):
            if path.exists():
                path.unlink()

    if not dry_run:
        nominations_output.parent.mkdir(parents=True, exist_ok=True)
        films_output.parent.mkdir(parents=True, exist_ok=True)
        people_output.parent.mkdir(parents=True, exist_ok=True)
        nominations_graph.serialize(destination=nominations_output, format="turtle")
        films_graph.serialize(destination=films_output, format="turtle")
        people_graph.serialize(destination=people_output, format="turtle")

    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transform OSCARS clean CSV into split TTL files.")
    parser.add_argument(
        "--csv",
        default="data/external/oscar_data/oscars_clean.csv",
        help="Input OSCARS clean CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Row limit (0 means all rows)",
    )
    parser.add_argument(
        "--nominations-output",
        default="data/instances/oscar_nominations.ttl",
        help="Output TTL path for nominations/ceremonies/categories",
    )
    parser.add_argument(
        "--films-output",
        default="data/instances/films.ttl",
        help="Output TTL path for film instances",
    )
    parser.add_argument(
        "--people-output",
        default="data/instances/people.ttl",
        help="Output TTL path for person instances",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output TTL files before writing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run transform/validation without writing output files",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    counts = run_transform_oscars_split(
        csv_path=Path(args.csv),
        nominations_output=Path(args.nominations_output),
        films_output=Path(args.films_output),
        people_output=Path(args.people_output),
        limit=args.limit,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(
        "Transformed OSCARS clean CSV rows={rows} nominations={nominations} films={films} persons={persons}".format(
            **counts
        )
    )
    print(
        f"Wrote split TTL: nominations={args.nominations_output}, "
        f"films={args.films_output}, people={args.people_output}"
    )


if __name__ == "__main__":
    main()
