from __future__ import annotations

import csv
import re
from pathlib import Path

from pyshacl import validate
from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef, XSD
from rdflib.collection import Collection

from scripts.transform_oscars_split import build_arg_parser, run_transform_oscars_split

MSH = Namespace("http://example.org/ontologies/MovieSHACL3#")
SH = Namespace("http://www.w3.org/ns/shacl#")
DASH = Namespace("http://datashapes.org/dash#")
SCHEMA = Namespace("https://schema.org/")


def _write_clean_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "year_film",
        "year_ceremony",
        "ceremony",
        "category",
        "canon_category",
        "name",
        "film",
        "winner",
        "tmdb_id",
        "imdb_id",
        "wikidata_id",
        "previous_nominations",
        "previous_wins",
        "nominee_imdb_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_shapes(ontology_path: Path) -> Graph:
    g = Graph()
    g.parse(str(ontology_path), format="turtle")
    shape_string = BNode()
    shape_lang = BNode()
    g.add((shape_string, SH.datatype, XSD.string))
    g.add((shape_lang, SH.datatype, RDF.langString))
    Collection(g, DASH.StringOrLangString, [shape_string, shape_lang])
    return g


def test_transform_oscars_split_builds_aligned_split_graphs(tmp_path: Path) -> None:
    csv_path = tmp_path / "oscars_clean.csv"
    nominations_out = tmp_path / "oscar_nominations.ttl"
    films_out = tmp_path / "films.ttl"
    people_out = tmp_path / "people.ttl"
    _write_clean_csv(
        csv_path,
        [
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Picture",
                "canon_category": "Best Picture",
                "name": "",
                "film": "Anora",
                "winner": "true",
                "tmdb_id": "1",
                "imdb_id": "tt123",
                "wikidata_id": "Q123",
                "previous_nominations": "0",
                "previous_wins": "0",
                "nominee_imdb_id": "",
            },
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Director",
                "canon_category": "Best Director",
                "name": "Sean Baker",
                "film": "Anora",
                "winner": "true",
                "tmdb_id": "1",
                "imdb_id": "tt123",
                "wikidata_id": "Q123",
                "previous_nominations": "3",
                "previous_wins": "1",
                "nominee_imdb_id": "nm0004778",
            },
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Original Song",
                "canon_category": "Best Original Song",
                "name": "Alice Example",
                "film": "Melody",
                "winner": "false",
                "tmdb_id": "2",
                "imdb_id": "tt234",
                "wikidata_id": "Q234",
                "previous_nominations": "1",
                "previous_wins": "0",
                "nominee_imdb_id": "nm1111111",
            },
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Original Song",
                "canon_category": "Best Original Song",
                "name": "Bob Example",
                "film": "Melody",
                "winner": "false",
                "tmdb_id": "2",
                "imdb_id": "tt234",
                "wikidata_id": "Q234",
                "previous_nominations": "4",
                "previous_wins": "2",
                "nominee_imdb_id": "nm2222222",
            },
            {
                "year_film": "1935",
                "year_ceremony": "1936",
                "ceremony": "8",
                "category": "Actor",
                "canon_category": "Actor",
                "name": "Clark Gable",
                "film": "Mutiny on the Bounty",
                "winner": "false",
                "tmdb_id": "3",
                "imdb_id": "tt0026752",
                "wikidata_id": "Q300",
                "previous_nominations": "2",
                "previous_wins": "1",
                "nominee_imdb_id": "nm0000022",
            },
            {
                "year_film": "1935",
                "year_ceremony": "1936",
                "ceremony": "8",
                "category": "Actor",
                "canon_category": "Actor",
                "name": "Charles Laughton",
                "film": "Mutiny on the Bounty",
                "winner": "false",
                "tmdb_id": "3",
                "imdb_id": "tt0026752",
                "wikidata_id": "Q300",
                "previous_nominations": "1",
                "previous_wins": "0",
                "nominee_imdb_id": "nm0491665",
            },
        ],
    )

    counts = run_transform_oscars_split(
        csv_path=csv_path,
        nominations_output=nominations_out,
        films_output=films_out,
        people_output=people_out,
        limit=0,
        dry_run=False,
        force=True,
    )

    assert counts["rows"] == 6
    assert counts["nominations"] == 6
    assert counts["films"] == 3
    assert counts["persons"] == 5

    nominations_graph = Graph()
    nominations_graph.parse(nominations_out, format="turtle")
    films_graph = Graph()
    films_graph.parse(films_out, format="turtle")
    people_graph = Graph()
    people_graph.parse(people_out, format="turtle")

    film_types = {obj for obj in films_graph.objects(None, RDF.type)}
    people_types = {obj for obj in people_graph.objects(None, RDF.type)}
    assert MSH.Film in film_types
    assert MSH.Person in people_types
    assert MSH.Identifier in film_types
    assert MSH.Identifier in people_types
    assert MSH.IdentifierScheme in film_types
    assert MSH.IdentifierScheme in people_types

    assert (MSH.AwardSystem_oscars, RDF.type, MSH.AwardSystem) in nominations_graph
    assert (MSH.AwardSystem_oscars, MSH.systemName, Literal("Academy Awards")) in nominations_graph
    assert (MSH.AwardSystem_oscars, MSH.shortName, Literal("Oscars")) in nominations_graph
    assert list(nominations_graph.triples((None, MSH.hasAwardSystem, MSH.AwardSystem_oscars)))
    assert (MSH.Ceremony_oscars_2025, MSH.ceremonyName, Literal("97th Academy Awards")) in nominations_graph
    assert list(people_graph.triples((None, MSH.hasIdentifier, None)))
    assert list(films_graph.triples((None, MSH.hasIdentifier, None)))
    assert list(films_graph.triples((None, RDF.type, MSH.Identifier)))
    assert list(people_graph.triples((None, RDF.type, MSH.Identifier)))
    assert not list(nominations_graph.triples((None, RDF.type, MSH.Identifier)))
    assert not list(nominations_graph.triples((None, RDF.type, MSH.IdentifierScheme)))
    assert not list(nominations_graph.triples((None, MSH.hasIdentifier, None)))
    assert not list(nominations_graph.triples((None, SCHEMA.identifier, None)))
    assert not list(films_graph.triples((None, MSH.imdbId, None)))
    assert not list(films_graph.triples((None, MSH.tmdbId, None)))
    assert not list(films_graph.triples((None, MSH.wikidataId, None)))

    nominee_types = {str(obj) for obj in nominations_graph.objects(None, MSH.nomineeType)}
    assert nominee_types == {"FILM", "PERSON", "COLLABORATION"}

    film_uris = {s for s in films_graph.subjects(RDF.type, MSH.Film)}
    nominee_uris = {s for s in people_graph.subjects(RDF.type, MSH.Person)}
    for ceremony in nominations_graph.subjects(RDF.type, MSH.AwardCeremony):
        assert list(nominations_graph.objects(ceremony, MSH.ceremonyName))
        assert list(nominations_graph.objects(ceremony, MSH.hasAwardSystem))

    for category in nominations_graph.subjects(RDF.type, MSH.AwardCategory):
        assert list(nominations_graph.objects(category, MSH.categoryName))
        assert list(nominations_graph.objects(category, MSH.hasAwardSystem))

    for person in people_graph.subjects(RDF.type, MSH.Person):
        assert list(people_graph.objects(person, MSH.fullName))

    for nom in nominations_graph.subjects(RDF.type, MSH.Nomination):
        for film_ref in nominations_graph.objects(nom, MSH.hasFilm):
            assert film_ref in film_uris
        for person_ref in nominations_graph.objects(nom, MSH.hasNominee):
            assert person_ref in nominee_uris
        assert str(nom).startswith(str(MSH) + "Nomination_oscars_")

    melody_song_nominees = []
    mutiny_actor_nominees = []
    for nom in nominations_graph.subjects(RDF.type, MSH.Nomination):
        film_refs = list(nominations_graph.objects(nom, MSH.hasFilm))
        if not film_refs:
            continue
        film_ref = film_refs[0]
        if film_ref == MSH.Film_Melody_2024:
            melody_song_nominees.append(len(list(nominations_graph.objects(nom, MSH.hasNominee))))
        if film_ref == MSH.Film_Mutiny_on_the_Bounty_1935:
            mutiny_actor_nominees.append(len(list(nominations_graph.objects(nom, MSH.hasNominee))))
    assert sorted(melody_song_nominees) == [1, 1]
    assert sorted(mutiny_actor_nominees) == [1, 1]
    for g in (nominations_graph, films_graph, people_graph):
        for s, p, o in g:
            assert "filmclub.org" not in str(s)
            assert "filmclub.org" not in str(p)
            if hasattr(o, "startswith"):
                assert "filmclub.org" not in str(o)

    combined = Graph()
    combined += nominations_graph
    combined += films_graph
    combined += people_graph
    ontology_path = Path(__file__).resolve().parent.parent / "movieontology.ttl"
    shapes_graph = _load_shapes(ontology_path)
    conforms, _, report = validate(
        combined,
        shacl_graph=shapes_graph,
        ont_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
    )
    assert conforms, f"Combined split graph should pass SHACL:\n{report}"


def test_transform_oscars_split_parser_defaults_to_clean_csv() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.csv == "data/external/oscar_data/oscars_clean.csv"


def test_oscar_split_instances_are_within_expected_bounds_and_shacl_conform() -> None:
    root = Path(__file__).resolve().parent.parent
    nominations_path = root / "data" / "instances" / "oscar_nominations.ttl"
    films_path = root / "data" / "instances" / "films.ttl"
    people_path = root / "data" / "instances" / "people.ttl"
    for path in (nominations_path, films_path, people_path):
        assert path.exists(), f"Missing required OSCARS split file: {path}"

    nominations_graph = Graph()
    nominations_graph.parse(nominations_path, format="turtle")
    films_graph = Graph()
    films_graph.parse(films_path, format="turtle")
    people_graph = Graph()
    people_graph.parse(people_path, format="turtle")

    film_types = {obj for obj in films_graph.objects(None, RDF.type)}
    people_types = {obj for obj in people_graph.objects(None, RDF.type)}
    assert MSH.Film in film_types
    assert MSH.Person in people_types
    assert MSH.Identifier in film_types
    assert MSH.Identifier in people_types
    assert MSH.IdentifierScheme in film_types
    assert MSH.IdentifierScheme in people_types

    nomination_count = len(set(nominations_graph.subjects(RDF.type, MSH.Nomination)))
    film_count = len(set(films_graph.subjects(RDF.type, MSH.Film)))
    person_count = len(set(people_graph.subjects(RDF.type, MSH.Person)))
    print(f"OSCARS split counts: nominations={nomination_count}, films={film_count}, persons={person_count}")
    reference_path = root / "data" / "instances" / "oscar_nominations copy.ttl"
    assert reference_path.exists(), f"Missing parity reference file: {reference_path}"
    reference_graph = Graph()
    reference_graph.parse(reference_path, format="turtle")
    reference_nomination_count = len(set(reference_graph.subjects(RDF.type, MSH.Nomination)))
    reference_film_count = len(set(reference_graph.subjects(RDF.type, MSH.Film)))
    reference_person_count = len(set(reference_graph.subjects(RDF.type, MSH.Person)))
    assert reference_nomination_count == 15_980
    assert reference_film_count == 4_957
    assert reference_person_count == 7_959

    nomination_delta = abs(nomination_count - reference_nomination_count)
    film_delta = abs(film_count - reference_film_count)
    person_delta = abs(person_count - reference_person_count)
    print(
        "OSCARS parity deltas vs reference: "
        f"nominations={nomination_delta}, films={film_delta}, persons={person_delta}"
    )
    assert nomination_delta <= 500
    assert film_delta <= 50
    assert person_delta <= 400

    film_uris = set(films_graph.subjects(RDF.type, MSH.Film))
    person_uris = set(people_graph.subjects(RDF.type, MSH.Person))
    for film in film_uris:
        assert list(films_graph.objects(film, MSH.hasIdentifier))
    assert list(films_graph.triples((None, RDF.type, MSH.Identifier)))
    assert list(people_graph.triples((None, RDF.type, MSH.Identifier)))
    assert not list(nominations_graph.triples((None, RDF.type, MSH.Identifier)))
    assert not list(nominations_graph.triples((None, RDF.type, MSH.IdentifierScheme)))
    assert not list(nominations_graph.triples((None, MSH.hasIdentifier, None)))
    assert not list(nominations_graph.triples((None, SCHEMA.identifier, None)))
    assert not list(films_graph.triples((None, MSH.imdbId, None)))
    assert not list(films_graph.triples((None, MSH.tmdbId, None)))
    assert not list(films_graph.triples((None, MSH.wikidataId, None)))
    for nom in nominations_graph.subjects(RDF.type, MSH.Nomination):
        for film_ref in nominations_graph.objects(nom, MSH.hasFilm):
            assert film_ref in film_uris
        for person_ref in nominations_graph.objects(nom, MSH.hasNominee):
            assert person_ref in person_uris

    acting_categories = {
        "actor",
        "actress",
        "actor in a leading role",
        "actress in a leading role",
        "actor in a supporting role",
        "actress in a supporting role",
    }
    saw_collaboration = False
    for nom in nominations_graph.subjects(RDF.type, MSH.Nomination):
        nominee_count = len(set(nominations_graph.objects(nom, MSH.hasNominee)))
        category_uri = next(nominations_graph.objects(nom, MSH.hasCategory), None)
        category_name = None
        if category_uri is not None:
            category_name_obj = next(nominations_graph.objects(category_uri, MSH.categoryName), None)
            if category_name_obj is not None:
                category_name = str(category_name_obj).strip().casefold()
        if category_name in acting_categories:
            assert nominee_count == 1
        nominee_type_obj = next(nominations_graph.objects(nom, MSH.nomineeType), None)
        if nominee_type_obj is not None and str(nominee_type_obj) == "COLLABORATION":
            assert nominee_count >= 1
            saw_collaboration = True
    assert saw_collaboration

    mutiny_nominations = [
        nom
        for nom in nominations_graph.subjects(RDF.type, MSH.Nomination)
        if (nom, MSH.hasCeremony, MSH.Ceremony_oscars_1936) in nominations_graph
        and (nom, MSH.hasFilm, MSH.Film_Mutiny_on_the_Bounty_1935) in nominations_graph
        and any(
            str(name).strip().casefold() == "actor"
            for category_uri in nominations_graph.objects(nom, MSH.hasCategory)
            for name in nominations_graph.objects(category_uri, MSH.categoryName)
        )
    ]
    assert len(mutiny_nominations) == 3

    sunrise_cinematography_nominations = [
        nom
        for nom in nominations_graph.subjects(RDF.type, MSH.Nomination)
        if (nom, MSH.hasCeremony, MSH.Ceremony_oscars_1928) in nominations_graph
        and (nom, MSH.hasFilm, MSH.Film_Sunrise_1927) in nominations_graph
        and any(
            str(name).strip().casefold() == "cinematography"
            for category_uri in nominations_graph.objects(nom, MSH.hasCategory)
            for name in nominations_graph.objects(category_uri, MSH.categoryName)
        )
    ]
    assert sunrise_cinematography_nominations
    assert any(
        str(next(nominations_graph.objects(nom, MSH.nomineeType), "")) == "COLLABORATION"
        for nom in sunrise_cinematography_nominations
    )

    def _as_str(x: str | None) -> str | None:
        if x is None:
            return None
        text = str(x).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return None
        return text

    def _normalize_id(x: str | None) -> str | None:
        text = _as_str(x)
        if text is None:
            return None
        try:
            fx = float(text)
            if fx.is_integer():
                return str(int(fx))
        except Exception:
            pass
        return text

    def _slugify(value: str) -> str:
        value = str(value).strip().replace("'", "")
        value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
        value = re.sub(r"_+", "_", value)
        return value.strip("_")

    clean_csv_path = root / "data" / "external" / "oscar_data" / "oscars_clean.csv"
    with clean_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = _as_str(row.get("name") or row.get("nominee"))
            nominee_imdb_id = _normalize_id(row.get("nominee_imdb_id"))
            if not name or not nominee_imdb_id:
                continue
            person_uri = MSH[f"Person_{_slugify(name)}"]
            assert (person_uri, MSH.hasIdentifier, None) in people_graph
            assert (person_uri, MSH.hasIdentifier, URIRef(f"https://www.imdb.com/name/{nominee_imdb_id}")) in people_graph

    combined = Graph()
    combined += nominations_graph
    combined += films_graph
    combined += people_graph
    for s, p, o in combined:
        assert "filmclub.org" not in str(s)
        assert "filmclub.org" not in str(p)
        assert "filmclub.org" not in str(o)

    ontology_path = root / "movieontology.ttl"
    shapes_graph = _load_shapes(ontology_path)
    conforms, _, report = validate(
        combined,
        shacl_graph=shapes_graph,
        ont_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
    )
    assert conforms, f"Combined OSCARS split files must pass SHACL:\n{report}"


def test_transform_oscars_split_is_byte_identical_on_repeat_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "oscars_clean.csv"
    nominations_out = tmp_path / "oscar_nominations.ttl"
    films_out = tmp_path / "films.ttl"
    people_out = tmp_path / "people.ttl"
    _write_clean_csv(
        csv_path,
        [
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Director",
                "canon_category": "Best Director",
                "name": "Sean Baker",
                "film": "Anora",
                "winner": "true",
                "tmdb_id": "1",
                "imdb_id": "tt123",
                "wikidata_id": "Q123",
                "previous_nominations": "3",
                "previous_wins": "1",
                "nominee_imdb_id": "nm0004778",
            },
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Picture",
                "canon_category": "Best Picture",
                "name": "",
                "film": "Anora",
                "winner": "true",
                "tmdb_id": "1",
                "imdb_id": "tt123",
                "wikidata_id": "Q123",
                "previous_nominations": "0",
                "previous_wins": "0",
                "nominee_imdb_id": "",
            },
        ],
    )

    run_transform_oscars_split(
        csv_path=csv_path,
        nominations_output=nominations_out,
        films_output=films_out,
        people_output=people_out,
        limit=0,
        dry_run=False,
        force=True,
    )
    first_nominations = nominations_out.read_bytes()
    first_films = films_out.read_bytes()
    first_people = people_out.read_bytes()

    run_transform_oscars_split(
        csv_path=csv_path,
        nominations_output=nominations_out,
        films_output=films_out,
        people_output=people_out,
        limit=0,
        dry_run=False,
        force=True,
    )
    second_nominations = nominations_out.read_bytes()
    second_films = films_out.read_bytes()
    second_people = people_out.read_bytes()

    assert first_nominations == second_nominations
    assert first_films == second_films
    assert first_people == second_people


def test_transform_oscars_split_splits_conflicting_winner_values_without_coercion(tmp_path: Path) -> None:
    csv_path = tmp_path / "oscars_clean.csv"
    nominations_out = tmp_path / "oscar_nominations.ttl"
    _write_clean_csv(
        csv_path,
        [
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Director",
                "canon_category": "Best Director",
                "name": "Sean Baker",
                "film": "Anora",
                "winner": "true",
                "tmdb_id": "1",
                "imdb_id": "tt123",
                "wikidata_id": "Q123",
                "previous_nominations": "3",
                "previous_wins": "1",
                "nominee_imdb_id": "nm0004778",
            },
            {
                "year_film": "2024",
                "year_ceremony": "2025",
                "ceremony": "97",
                "category": "Best Director",
                "canon_category": "Best Director",
                "name": "Sean Baker",
                "film": "Anora",
                "winner": "false",
                "tmdb_id": "1",
                "imdb_id": "tt123",
                "wikidata_id": "Q123",
                "previous_nominations": "3",
                "previous_wins": "1",
                "nominee_imdb_id": "nm0004778",
            },
        ],
    )

    run_transform_oscars_split(
        csv_path=csv_path,
        nominations_output=nominations_out,
        films_output=tmp_path / "films.ttl",
        people_output=tmp_path / "people.ttl",
        limit=0,
        dry_run=False,
        force=True,
    )

    g = Graph()
    g.parse(nominations_out, format="turtle")
    winner_values = [obj.toPython() for obj in g.objects(None, MSH.winner)]
    assert True in winner_values
    assert False in winner_values
