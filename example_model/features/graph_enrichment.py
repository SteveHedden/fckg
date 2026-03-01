"""Graph-native enrichment feature extraction from unified RDF graph."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from rdflib import Graph


_QUERY_FILM_METADATA = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>

SELECT ?film ?budget ?revenue ?runtime ?tmdbRating ?tmdbPopularity ?voteCount
       ?imdbRating ?rtScore ?metacritic ?mpaaRating ?origLang ?releaseDate ?tagline
WHERE {
  ?film a msh:Film .
  OPTIONAL { ?film msh:budget ?budget }
  OPTIONAL { ?film msh:revenue ?revenue }
  OPTIONAL { ?film msh:runtime ?runtime }
  OPTIONAL { ?film msh:tmdbRating ?tmdbRating }
  OPTIONAL { ?film msh:tmdbPopularity ?tmdbPopularity }
  OPTIONAL { ?film msh:voteCount ?voteCount }
  OPTIONAL { ?film msh:imdbRating ?imdbRating }
  OPTIONAL { ?film msh:rottenTomatoesScore ?rtScore }
  OPTIONAL { ?film msh:metacriticScore ?metacritic }
  OPTIONAL { ?film msh:mpaaRating ?mpaaRating }
  OPTIONAL { ?film msh:originalLanguage ?origLang }
  OPTIONAL { ?film msh:releaseDate ?releaseDate }
  OPTIONAL { ?film msh:tagline ?tagline }
}
"""

_QUERY_FESTIVAL_AWARDS = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>

SELECT ?film ?award
WHERE {
  ?film a msh:Film .
  ?film msh:festivalAward ?award .
}
"""

_QUERY_FILM_COUNTRY = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>

SELECT ?film ?country
WHERE {
  ?film a msh:Film .
  ?film msh:country ?country .
}
"""

_QUERY_FILM_GENRES = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?film ?genre ?genreLabel
WHERE {
  ?film a msh:Film .
  ?film msh:hasGenre ?genre .
  OPTIONAL { ?genre rdfs:label ?genreLabel }
}
"""

_QUERY_FILM_COMPANIES = """\
PREFIX msh: <http://example.org/ontologies/MovieSHACL3#>

SELECT ?film ?company
WHERE {
  ?film a msh:Film .
  ?film msh:productionCompany ?company .
}
"""


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _normalize_genre_name(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if "#" in text and "Genre_" in text:
        text = text.split("#", 1)[1].replace("Genre_", "")
    text = text.replace("_", " ")
    return " ".join(w.capitalize() for w in text.split())


def _parse_release_month(value: Any) -> float:
    s = str(value or "").strip()
    m = re.match(r"^\d{4}-(\d{2})-\d{2}$", s)
    if not m:
        return np.nan
    return float(int(m.group(1)))


def extract_graph_enrichment(graph: Graph, canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Merge enrichment features from RDF into canonical film-level DataFrame."""
    if canonical_df.empty or "film_uri" not in canonical_df.columns:
        return canonical_df

    rows: list[dict[str, Any]] = []
    for r in graph.query(_QUERY_FILM_METADATA):
        rows.append(
            {
                "film_uri": str(r.film),
                "budget": _to_float(r.budget),
                "revenue": _to_float(r.revenue),
                "runtime": _to_float(r.runtime),
                "vote_average": _to_float(r.tmdbRating),
                "vote_count": _to_float(r.voteCount),
                "popularity": _to_float(r.tmdbPopularity),
                "imdb_rating": _to_float(r.imdbRating),
                "rotten_tomatoes_pct": _to_float(r.rtScore),
                "metacritic_score": _to_float(r.metacritic),
                "mpaa_rating": str(r.mpaaRating) if r.mpaaRating else None,
                "original_language": str(r.origLang) if r.origLang else None,
                "release_date": str(r.releaseDate) if r.releaseDate else None,
                "tagline": str(r.tagline) if r.tagline else None,
            }
        )
    meta_df = pd.DataFrame(rows)
    if meta_df.empty:
        result = canonical_df.copy()
        return result

    meta_df = meta_df.groupby("film_uri", as_index=False).first()

    genre_rows: list[dict[str, str]] = []
    for r in graph.query(_QUERY_FILM_GENRES):
        raw = str(r.genreLabel) if r.genreLabel else str(r.genre)
        genre = _normalize_genre_name(raw)
        if genre:
            genre_rows.append({"film_uri": str(r.film), "genre": genre})
    genre_df = pd.DataFrame(genre_rows)
    genre_map: dict[str, list[str]] = {}
    if not genre_df.empty:
        for film_uri, grp in genre_df.groupby("film_uri"):
            genre_map[film_uri] = sorted(set(grp["genre"]))

    company_rows: list[dict[str, str]] = []
    for r in graph.query(_QUERY_FILM_COMPANIES):
        company = str(r.company).strip()
        if company:
            company_rows.append({"film_uri": str(r.film), "company": company})
    company_df = pd.DataFrame(company_rows)
    company_map: dict[str, list[str]] = {}
    if not company_df.empty:
        for film_uri, grp in company_df.groupby("film_uri"):
            company_map[film_uri] = sorted(set(grp["company"]))

    result = canonical_df.copy()
    result = result.merge(meta_df, on="film_uri", how="left")

    # Derived features aligned with legacy CSV enrichment outputs
    result["log_budget"] = np.where(result["budget"].notna(), np.log1p(result["budget"]), np.nan)
    result["log_revenue"] = np.where(result["revenue"].notna(), np.log1p(result["revenue"]), np.nan)
    result["budget_revenue_ratio"] = np.where(
        (result["budget"].notna()) & (result["budget"] > 0),
        result["revenue"] / result["budget"],
        np.nan,
    )
    result["english_language"] = (result["original_language"] == "en").astype(int)
    result["release_month"] = result["release_date"].apply(_parse_release_month)
    result["has_tagline"] = result["tagline"].apply(lambda x: 1 if isinstance(x, str) and x.strip() else 0)

    for mpaa, col in {"R": "rated_R", "PG-13": "rated_PG13", "PG": "rated_PG", "G": "rated_G"}.items():
        result[col] = (result["mpaa_rating"] == mpaa).astype(int)

    result["_genres"] = result["film_uri"].map(lambda u: genre_map.get(u, []))
    result["num_genres"] = result["_genres"].apply(len)
    all_genres = sorted({g for genres in result["_genres"] for g in genres})
    for genre in all_genres:
        col = f"genre_{genre.lower().replace(' ', '_').replace('-', '_')}"
        result[col] = result["_genres"].apply(lambda gl, g=genre: int(g in gl))

    result["_companies"] = result["film_uri"].map(lambda u: company_map.get(u, []))
    result["num_production_companies"] = result["_companies"].apply(len)
    winner_companies = result[result["winner"] == True]["_companies"].explode()
    prestige = set()
    if winner_companies is not None and not winner_companies.empty:
        counts = winner_companies.value_counts()
        prestige = set(counts[counts >= 3].index)
    result["has_prestige_company"] = result["_companies"].apply(
        lambda cl: int(any(c in prestige for c in cl)) if isinstance(cl, list) else 0
    )

    result = result.drop(columns=["_genres", "_companies"])

    # --- Festival awards (from Wikidata via msh:festivalAward) ---
    festival_rows: list[dict[str, str]] = []
    for r in graph.query(_QUERY_FESTIVAL_AWARDS):
        festival_rows.append({"film_uri": str(r.film), "award": str(r.award)})
    festival_df = pd.DataFrame(festival_rows)
    festival_map: dict[str, list[str]] = {}
    if not festival_df.empty:
        for film_uri, grp in festival_df.groupby("film_uri"):
            festival_map[str(film_uri)] = list(grp["award"])

    def _has_award(awards: list[str], *keywords: str) -> int:
        return int(any(all(k.lower() in a.lower() for k in keywords) for a in awards))

    result["_awards"] = result["film_uri"].map(lambda u: festival_map.get(u, []))
    result["num_wikidata_awards"]   = result["_awards"].apply(len)
    result["palme_dor"]             = result["_awards"].apply(lambda a: _has_award(a, "palme d'or"))
    result["cannes_grand_prix"]     = result["_awards"].apply(lambda a: _has_award(a, "cannes", "grand prix"))
    result["golden_lion"]           = result["_awards"].apply(lambda a: _has_award(a, "golden lion"))
    result["venice_jury_prize"]     = result["_awards"].apply(lambda a: _has_award(a, "venice", "jury"))
    result["golden_bear"]           = result["_awards"].apply(lambda a: _has_award(a, "golden bear"))
    result["big_three_festival"]    = (
        (result["palme_dor"] | result["golden_lion"] | result["golden_bear"]).astype(int)
    )
    result["sundance_grand_jury"]   = result["_awards"].apply(
        lambda a: _has_award(a, "sundance", "grand jury prize") or _has_award(a, "sundance", "grand jury prize")
    )
    result["spirit_best_film"]      = result["_awards"].apply(
        lambda a: _has_award(a, "independent spirit award for best film")
                  or _has_award(a, "independent spirit award for best feature")
    )
    result["nsfc_best_film"]        = result["_awards"].apply(lambda a: _has_award(a, "national society of film critics", "best film"))
    result["nyfc_best_film"]        = result["_awards"].apply(lambda a: _has_award(a, "new york film critics circle", "best film"))
    result["lafca_best_film"]       = result["_awards"].apply(lambda a: _has_award(a, "los angeles film critics", "best film"))
    result["razzie_worst_picture"]  = result["_awards"].apply(
        lambda a: _has_award(a, "worst picture") or _has_award(a, "worst film")
    )
    result = result.drop(columns=["_awards"])

    # --- Country of origin (from Wikidata via msh:country) ---
    country_rows: list[dict[str, str]] = []
    for r in graph.query(_QUERY_FILM_COUNTRY):
        country_rows.append({"film_uri": str(r.film), "country": str(r.country)})
    country_df = pd.DataFrame(country_rows)
    country_map: dict[str, list[str]] = {}
    if not country_df.empty:
        for film_uri, grp in country_df.groupby("film_uri"):
            country_map[str(film_uri)] = list(grp["country"])

    result["_countries"] = result["film_uri"].map(lambda u: country_map.get(u, []))
    _english_countries = {"United States", "United Kingdom", "Australia", "Canada", "Ireland", "New Zealand"}
    result["country_us"]            = result["_countries"].apply(lambda c: int("United States" in c))
    result["country_uk"]            = result["_countries"].apply(lambda c: int("United Kingdom" in c))
    result["country_france"]        = result["_countries"].apply(lambda c: int("France" in c))
    result["country_non_english"]   = result["_countries"].apply(
        lambda c: int(bool(c) and not any(x in _english_countries for x in c))
    )
    result = result.drop(columns=["_countries"])

    return result
