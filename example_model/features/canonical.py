"""Extract features from the canonical RDF nomination graph.

All features in this module are derived purely from the canonical instance
data (nominations, ceremonies, categories, films, persons). No external API
data is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph, Namespace

MSH = Namespace("http://example.org/ontologies/MovieSHACL3#")

# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

# Maps upper-cased category prefixes/names to a feature group.
# Only categories relevant to predictive modelling are classified;
# others (shorts, docs, honorary, BP itself) are excluded.
_ACTING_NAMES = frozenset({
    "ACTOR", "ACTRESS",
    "ACTOR IN A LEADING ROLE", "ACTOR IN A SUPPORTING ROLE",
    "ACTRESS IN A LEADING ROLE", "ACTRESS IN A SUPPORTING ROLE",
})


def _classify_category(cat: str) -> str | None:
    """Classify an Oscar or BAFTA category name into a feature group."""
    upper = cat.upper().strip()
    # Oscar acting (exact set)
    if upper in _ACTING_NAMES:
        return "acting"
    # Oscar prefix patterns
    if upper.startswith("DIRECTING"):
        return "directing"
    if upper.startswith("WRITING"):
        return "writing"
    if upper.startswith("CINEMATOGRAPHY"):
        return "cinematography"
    if upper == "FILM EDITING":
        return "editing"
    if upper.startswith("MUSIC"):
        return "music"
    if any(upper.startswith(t) for t in (
        "SOUND", "VISUAL EFFECTS", "SPECIAL VISUAL EFFECTS", "SPECIAL EFFECTS",
        "COSTUME DESIGN", "MAKEUP", "ART DIRECTION", "PRODUCTION DESIGN",
        "SPECIAL ACHIEVEMENT AWARD",
    )):
        return "technical"
    # BAFTA-style names (prefixed with BEST or containing role descriptions)
    if any(phrase in upper for phrase in (
        "ACTOR IN A LEADING ROLE", "ACTRESS IN A LEADING ROLE",
        "ACTOR IN A SUPPORTING ROLE", "ACTRESS IN A SUPPORTING ROLE",
    )):
        return "acting"
    if upper in {"BEST DIRECTOR", "BEST DIRECTION"}:
        return "directing"
    if "SCREENPLAY" in upper:
        return "writing"
    if "CINEMATOGRAPHY" in upper:
        return "cinematography"
    if upper == "BEST EDITING":
        return "editing"
    return None


# Best-Picture category names used historically
_BP_NAMES = frozenset({
    "BEST PICTURE",
    "BEST MOTION PICTURE",
    "OUTSTANDING MOTION PICTURE",
    "OUTSTANDING PICTURE",
    "OUTSTANDING PRODUCTION",
    "UNIQUE AND ARTISTIC PICTURE",
})

# ---------------------------------------------------------------------------
# SPARQL queries
# ---------------------------------------------------------------------------

_QUERY_ALL_NOMINATIONS = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?nom ?cat ?film ?filmTitle ?yearFilm ?catName ?winner ?nominee ?nomineeType
       ?tmdbId ?imdbId
WHERE {
  ?nom a msh:Nomination ;
       msh:hasCategory ?cat ;
       msh:hasFilm ?film ;
       msh:yearFilm ?yearFilm ;
       msh:winner ?winner .
  ?cat msh:categoryName ?catName .
  ?film msh:title ?filmTitle .
  OPTIONAL { ?nom msh:hasNominee ?nominee }
  OPTIONAL { ?nom msh:nomineeType ?nomineeType }
  OPTIONAL { ?film msh:tmdbId ?tmdbId }
  OPTIONAL { ?film msh:imdbId ?imdbId }
}
"""


def load_graph(*paths: str | Path) -> Graph:
    """Internal utility: parse one or more Turtle files into a single graph."""
    g = Graph()
    for p in paths:
        g.parse(str(p), format="turtle")
    return g


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _query_nominations(graph: Graph) -> pd.DataFrame:
    """Run the nomination SPARQL and return a raw DataFrame."""
    rows: list[dict[str, Any]] = []
    for row in graph.query(_QUERY_ALL_NOMINATIONS):
        rows.append({
            "nom": str(row.nom),
            "cat_uri": str(row.cat),
            "film_uri": str(row.film),
            "film": str(row.filmTitle),
            "year_film": int(str(row.yearFilm)[:4]),
            "category": str(row.catName),
            "winner": str(row.winner).lower() == "true",
            "nominee": str(row.nominee) if row.nominee else None,
            "nominee_type": str(row.nomineeType) if row.nomineeType else None,
            "tmdb_id": str(row.tmdbId) if row.tmdbId else None,
            "imdb_id": str(row.imdbId) if row.imdbId else None,
        })
    return pd.DataFrame(rows)


def extract_category_features(all_noms: pd.DataFrame) -> pd.DataFrame:
    """Compute per-film category group nomination counts.

    Returns a DataFrame indexed by (film, year_film) with columns:
      acting_noms, directing_nom, writing_noms, cinematography_nom,
      editing_nom, technical_noms, music_noms, above_line_noms, other_noms
    """
    all_noms = all_noms.copy()
    all_noms["category_group"] = all_noms["category"].apply(_classify_category)

    # Total distinct categories per film/year
    other_noms = (
        all_noms
        .groupby(["film", "year_film"])["category"]
        .nunique()
        .reset_index()
        .rename(columns={"category": "other_noms"})
    )

    # Pivot classified categories
    classified = all_noms[all_noms["category_group"].notna()].copy()
    cat_counts = (
        classified
        .groupby(["film", "year_film", "category_group"])["category"]
        .nunique()
        .reset_index()
        .rename(columns={"category": "count"})
    )
    cat_pivot = cat_counts.pivot_table(
        index=["film", "year_film"],
        columns="category_group",
        values="count",
        fill_value=0,
    ).reset_index()

    rename_map = {
        "acting": "acting_noms",
        "directing": "directing_nom",
        "writing": "writing_noms",
        "cinematography": "cinematography_nom",
        "editing": "editing_nom",
        "technical": "technical_noms",
        "music": "music_noms",
    }
    cat_pivot = cat_pivot.rename(columns=rename_map)
    for col in rename_map.values():
        if col not in cat_pivot.columns:
            cat_pivot[col] = 0

    cat_pivot["above_line_noms"] = (
        cat_pivot["acting_noms"]
        + cat_pivot["directing_nom"]
        + cat_pivot["writing_noms"]
    )

    result = cat_pivot.merge(other_noms, on=["film", "year_film"], how="left")
    return result


def extract_bp_features(all_noms: pd.DataFrame) -> pd.DataFrame:
    """Extract Best Picture nominee features from all nominations.

    Returns a film-level DataFrame with producer history and external IDs.
    """
    bp_mask = all_noms["category"].str.upper().str.strip().isin(_BP_NAMES)
    bp = all_noms[bp_mask].copy()

    if bp.empty:
        return pd.DataFrame()

    # Compute prior nomination/win counts per person before each ceremony year.
    # For each BP nomination row, count how many prior nominations/wins
    # that nominee had across ALL categories.
    person_noms = all_noms[all_noms["nominee"].notna()].copy()
    person_history: dict[tuple[str, int], dict[str, int]] = {}
    for nominee, grp in person_noms.groupby("nominee"):
        sorted_years = sorted(grp["year_film"].unique())
        cum_noms = 0
        cum_wins = 0
        for yr in sorted_years:
            person_history[(str(nominee), int(yr))] = {
                "prev_noms": cum_noms,
                "prev_wins": cum_wins,
            }
            yr_data = grp[grp["year_film"] == yr]
            cum_noms += len(yr_data)
            cum_wins += int(yr_data["winner"].sum())

    bp["previous_nominations"] = bp.apply(
        lambda r: person_history.get(
            (str(r["nominee"]), int(r["year_film"])), {}
        ).get("prev_noms", 0)
        if r["nominee"] else 0,
        axis=1,
    )
    bp["previous_wins"] = bp.apply(
        lambda r: person_history.get(
            (str(r["nominee"]), int(r["year_film"])), {}
        ).get("prev_wins", 0)
        if r["nominee"] else 0,
        axis=1,
    )

    # Aggregate to film level
    bp_agg = bp.groupby(["film", "year_film"]).agg(
        film_uri=("film_uri", "first"),
        previous_nominations_sum=("previous_nominations", "sum"),
        previous_nominations_max=("previous_nominations", "max"),
        previous_nominations_avg=("previous_nominations", "mean"),
        previous_wins_sum=("previous_wins", "sum"),
        previous_wins_max=("previous_wins", "max"),
        previous_wins_any=("previous_wins", lambda x: int((x > 0).any())),
        num_producers=("nominee", "size"),
        winner=("winner", "max"),
        tmdb_id=("tmdb_id", "first"),
        imdb_id=("imdb_id", "first"),
    ).reset_index()

    return bp_agg


# ---------------------------------------------------------------------------
# Precursor award extraction from canonical TTL files
# ---------------------------------------------------------------------------

# Maps award system short names to their feature column name.
# Each precursor TTL is a separate canonical vocabulary.
_PRECURSOR_SYSTEMS = {
    "dga": "dga_winner",
    "pga": "pga_winner",
    "bafta": "bafta_winner",
    "sag": "sag_winner",
    "golden_globes": "globe_winner",
}

_QUERY_PRECURSOR_WINNERS = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>

SELECT ?film ?ceremonyYear ?systemShort
WHERE {
  ?nom a msh:Nomination ;
       msh:hasFilm ?film ;
       msh:hasCeremony ?cer ;
       msh:winner true .
  ?film msh:title ?filmTitle .
  ?cer msh:yearCeremony ?ceremonyYear .
  ?cer msh:hasAwardSystem ?sys .
  ?sys msh:shortName ?systemShort .
}
"""

def extract_precursor_features(
    graph: Graph,
    bp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract precursor award winner indicators using shared Film URIs."""
    result = bp_df.copy()

    # Initialize precursor columns
    for col in _PRECURSOR_SYSTEMS.values():
        result[col] = 0
    if result.empty:
        return result

    winner_rows: list[dict[str, Any]] = []
    for row in graph.query(_QUERY_PRECURSOR_WINNERS):
        winner_rows.append({
            "film_uri": str(row.film),
            "ceremony_year": int(str(row.ceremonyYear)[:4]),
            "system": str(row.systemShort).strip().lower(),
        })
    if not winner_rows:
        return result

    winner_df = pd.DataFrame(winner_rows)

    for idx, row in result.iterrows():
        film_uri = row.get("film_uri")
        if not film_uri:
            continue
        year = int(row["year_film"])
        matches = winner_df[
            (winner_df["film_uri"] == film_uri)
            & (winner_df["ceremony_year"] >= year - 2)
            & (winner_df["ceremony_year"] <= year + 2)
        ]
        if matches.empty:
            continue
        for system in matches["system"].unique():
            for sys_prefix, col in _PRECURSOR_SYSTEMS.items():
                if system == sys_prefix or sys_prefix in system:
                    result.at[idx, col] = 1
                    break

    return result


def extract_canonical_features(graph: Graph) -> pd.DataFrame:
    """Full canonical feature extraction pipeline.

    Queries the graph, computes category features and BP nominee features,
    and merges precursor award indicators from the same graph.
    Returns a merged film-level DataFrame for Best Picture nominees.
    """
    all_noms = _query_nominations(graph)
    if all_noms.empty:
        return pd.DataFrame()

    cat_features = extract_category_features(all_noms)
    bp_features = extract_bp_features(all_noms)

    if bp_features.empty:
        return pd.DataFrame()

    # Merge category features into BP features
    cat_cols = [
        "film", "year_film",
        "acting_noms", "directing_nom", "writing_noms",
        "cinematography_nom", "editing_nom", "technical_noms",
        "music_noms", "above_line_noms", "other_noms",
    ]
    result = bp_features.merge(
        cat_features[cat_cols], on=["film", "year_film"], how="left"
    )
    # Fill NaN category counts with 0
    fill_cols = [c for c in cat_cols if c not in ("film", "year_film")]
    result[fill_cols] = result[fill_cols].fillna(0).astype(int)

    # Merge precursor award indicators
    result = extract_precursor_features(graph, result)

    return result
