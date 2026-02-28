# Film Club Knowledge Graph (FCKG)

An RDF knowledge graph of award nominations across six major film award systems, with a SHACL-based ontology for validation and inference. Designed for predictive modeling, longitudinal analysis, and enrichment via external APIs.

**[Interactive Ontology Diagram →](https://stevehedden.github.io/fckg/)**

## Data Foundation

The publishable data foundation consists of the ontology plus instance data. Everything a consumer needs to load, query, and enrich the graph.

### Ontology

`movieontology.ttl` defines the schema using SHACL node shapes and property shapes:

| Class | Description |
|-------|-------------|
| **Film** | A motion picture with title, release year, and external identifiers |
| **Person** | An individual involved in film production (director, actor, writer, etc.) |
| **Nomination** | A reified time-bound event linking a film and nominee(s) to an award category and ceremony |
| **AwardSystem** | An award-granting body (Oscars, BAFTA, SAG, DGA, PGA, Golden Globes) |
| **AwardCeremony** | A specific year's ceremony within an award system |
| **AwardCategory** | A category within an award system (e.g., Best Picture, Best Director) |
| **Identifier** | A reified external identifier (IMDB, TMDB, Wikidata) |

#### Key Modeling Patterns

- **Namespace**: `http://example.org/ontologies/MovieSHACL3#` (prefix `msh:`)
- **Nominations are the core entity**: A Nomination links a Film, one or more Person nominees, a Category, and a Ceremony. Persons and Films are connected indirectly through Nominations.
- **nomineeType enum**: `PERSON` (individual nomination, e.g., Best Actor), `COLLABORATION` (multi-person nomination, e.g., Sound), or `FILM` (film-level nomination with no individual nominee)
- **Collaborative nominations**: Non-acting categories group all collaborators into a single Nomination with multiple `hasNominee` links
- **hasCategory vs hasCanonCategory**: `hasCategory` is the historical category name; `hasCanonCategory` maps to a normalized modern equivalent (Oscar data only)
- **External IDs**: Films and Persons carry identifiers via `hasIdentifier` pointing to full URL IRIs (e.g., `<https://www.imdb.com/title/tt15398776>`)

### Instance Data

All instance files live under `data/instances/`:

| File | Contents |
|------|----------|
| `oscar_nominations.ttl` | ~10,600 Oscar nominations (1927-2025), ceremonies, categories |
| `films.ttl` | ~6,300 films with titles, release years, and external identifiers |
| `people.ttl` | ~10,700 persons with names and IMDB identifiers |
| `bafta_nominations.ttl` | ~4,800 BAFTA nominations (1948-2025) |
| `golden_globes_nominations.ttl` | ~3,700 Golden Globe nominations (1944-2025) |
| `sag_nominations.ttl` | ~780 SAG nominations (1994-2025) |
| `dga_nominations.ttl` | ~500 DGA nominations (1948-2025) |
| `pga_nominations.ttl` | ~270 PGA nominations (1989-2025) |

`films.ttl` and `people.ttl` are shared across all award systems. Each nomination file references Film and Person URIs defined in those shared files.

### Loading the Graph

```python
from rdflib import Graph

g = Graph()
g.parse("movieontology.ttl", format="turtle")
for f in Path("data/instances").glob("*.ttl"):
    g.parse(f, format="turtle")
```

### External Identifiers

Films carry up to three external identifiers via `hasIdentifier`:

```turtle
msh:Film_Oppenheimer_2023 msh:hasIdentifier
    <https://www.imdb.com/title/tt15398776>,
    <https://www.themoviedb.org/movie/872585>,
    <https://www.wikidata.org/wiki/Q105584780> .
```

Persons carry IMDB identifiers:

```turtle
msh:Person_Christopher_Nolan msh:hasIdentifier
    <https://www.imdb.com/name/nm0634240> .
```

These identifiers enable enrichment from any external source. To extract the numeric ID from a URL, parse the last path segment (e.g., `tt15398776` from the IMDB URL, `872585` from the TMDB URL, `Q105584780` from the Wikidata URL).

### Enrichment

The data foundation is designed to be enriched by consumers using whatever sources and features they find valuable. The external identifiers give direct access to:

- **TMDB** (`api.themoviedb.org`): budget, revenue, runtime, cast/crew credits, ratings, genres, production companies
- **OMDB** (`omdbapi.com`): IMDB ratings, Rotten Tomatoes scores, Metacritic scores, MPAA ratings, box office
- **Wikidata** (`query.wikidata.org`): festival awards (Cannes, Venice, Berlin), country of origin, original language, biographical data

Enriched data should be stored as additive RDF overlays that reference the same Film/Person URIs, not by modifying the foundation files.

## Demo Notebook

Run `fckg_demo_simply.ipynb` to:

1. Load `movieontology.ttl` and `data/instances/*.ttl`
2. Execute example SPARQL queries
3. See single-title API examples for TMDB, OMDb, and Wikidata

## Acknowledgments

Oscar nomination data is derived from [DLu/oscar_data](https://github.com/DLu/oscar_data) (BSD-2-Clause). Copyright (c) David Lu. See that repository for the original dataset and license terms.

## License

MIT. See `LICENSE`.
