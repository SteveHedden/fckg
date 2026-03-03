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
| **Genre** | A thematic/stylistic classification for films (Drama, Comedy, Science Fiction, etc.) |
| **Nomination** | A reified time-bound event linking a film and nominee(s) to an award category and ceremony |
| **AwardSystem** | An award-granting body (Oscars, BAFTA, SAG, DGA, PGA, Golden Globes) |
| **AwardCeremony** | A specific year's ceremony within an award system |
| **AwardCategory** | A specific category within an award system (e.g., Best Picture, Best Director) |
| **AwardConcept** | An abstract cross-system achievement concept grouping equivalent categories across award bodies (e.g., "Best Male Lead Actor" groups Oscar Best Actor, SAG Male Lead, BAFTA Best Actor) |
| **Award** | A recognition or honor bestowed on a film or person (used for festival awards such as Palme d'Or, Golden Lion) |
| **Identifier** | A reified external identifier (IMDB, TMDB, Wikidata) |
| **IdentifierScheme** | A controlled vocabulary entry for identifier types (imdb, tmdb, wikidata) |
| **ForecastSet** | A set of probabilistic predictions from one model run for a specific category and ceremony year |
| **Forecast** | A single binary probabilistic prediction about whether a specific nomination will resolve as a win |

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
| `people.ttl` | ~8,300 persons with names and IMDB identifiers |
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

## Example Model

`example_model/` is a complete award prediction pipeline built on top of the FCKG data foundation. It goes from raw RDF graph data to ranked win-probability predictions with SHAP explanations for any supported award category.

### Setup

```bash
pip install rdflib pyshacl scikit-learn pandas numpy python-dotenv requests matplotlib shap
cp .env.example .env   # fill in TMDB_API_KEY and OMDB_API_KEY
```

### Pipeline

**Step 1 — Build foundation graph** (optional; combines ontology + instances into a single validated TTL):
```bash
python example_model/foundation_builder.py
```

**Step 2 — Enrich** (fetches budget, revenue, ratings, genres, festival awards from TMDB / OMDb / Wikidata):
```bash
python example_model/enrichment_runner.py           # all films (~2–3 hrs first run)
python example_model/enrichment_runner.py --year 2024  # only films from one release year
python example_model/enrichment_runner.py --cache-only # rebuild TTLs from cached responses
```

**Step 3 — Feature selection** (identifies predictive features for a category):
```bash
python example_model/feature_selection.py --category best_picture
python example_model/feature_selection.py --category best_actress
python example_model/feature_selection.py --category best_actor --award-system sag
```

Features are handled in three tiers:

- **Tier 1** — Category-specific precursor features (e.g., `bafta_actress_winner` for best_actress): always forced into the model, bypass RFECV.
- **Tier 2** — Generic precursor winners (e.g., `pga_winner`, `dga_winner`): go through RFECV so the model decides if they add value. Generic `*_nominee` flags are excluded (noisy signal).
- **Tier 3** — Everything else (ratings, financials, genres, etc.): goes through RFECV.

By default, conceptual feature grouping runs before RFECV: within each group (financial scale, critical reception, audience ratings, etc.) only the single most predictive feature is kept. Disable with `--no-conceptual`.

An EPV-based cap (events-per-variable ≈ 5) limits the number of non-precursor features to prevent overfitting in small-sample categories.

Outputs `data/reports/<award_system>/<category>/selected_features.json`.

**Step 4 — Model selection** (backtests model types and picks the best by AUC):
```bash
python example_model/model_selection.py --category best_picture
python example_model/model_selection.py --category best_actress --award-system sag
python example_model/model_selection.py --category best_picture --model-types LR RF GB
python example_model/model_selection.py --category best_picture --start-year 2010 --end-year 2024
```

The default model type is `ConstrainedLR` — a logistic regression with non-negative coefficient constraints on all precursor features. This ensures that winning a precursor award can never hurt a nominee's predicted odds. Other available types: `LR`, `Ridge`, `RF`, `GB`, `LR+RF`.

Outputs `grid_results.csv`, `backcast_results.csv`, `backcast_summary.txt`, `feature_importance.csv/png` to `data/reports/<award_system>/<category>/`.

**Step 5 — Predict** (trains on all historical data, generates ranked win probabilities with SHAP explanations):
```bash
python example_model/predict.py --category best_picture --year 2025
python example_model/predict.py --category best_actress --award-system sag --year 2025
python example_model/predict.py --category best_picture --year 2025 --model LR \
    --features pga_winner dga_winner sag_winner
```

Win percentages are computed via softmax over logits — this converts the model's independent per-nominee probability estimates into a proper categorical distribution that sums to 100%.

Outputs `predictions_<year>.csv`, `shap_values_<year>.csv`, `shap_summary_<year>.png`, and per-nominee waterfall plots to `data/reports/<award_system>/<category>/`.

### Supported categories

| `--award-system` | `--category` options |
|---|---|
| `oscars` (default) | `best_picture`, `best_director`, `best_actor`, `best_actress`, `best_supporting_actor`, `best_supporting_actress`, `animated_film`, `international_film`, `documentary`, `best_adapted_screenplay`, `best_original_screenplay`, `cinematography`, `film_editing`, `original_score`, `original_song`, `production_design`, `costume_design`, `sound`, `visual_effects`, `makeup` |
| `sag` | `best_ensemble_cast`, `best_actor`, `best_actress`, `best_supporting_actor`, `best_supporting_actress` |
| `bafta` | `best_film`, `best_director`, `best_actor`, `best_actress`, `best_supporting_actor`, `best_supporting_actress`, `best_adapted_screenplay`, `best_original_screenplay`, `cinematography`, `film_editing` |

## Demo Notebook

Run `fckg_demo_simply.ipynb` to:

1. Load `movieontology.ttl` and `data/instances/*.ttl`
2. Execute example SPARQL queries
3. See single-title API examples for TMDB, OMDb, and Wikidata

## Acknowledgments

Oscar nomination data is derived from [DLu/oscar_data](https://github.com/DLu/oscar_data) (BSD-2-Clause). Copyright (c) David Lu. See that repository for the original dataset and license terms.

## License

MIT. See `LICENSE`.
