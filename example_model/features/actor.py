"""Extract features for Best Actor prediction from Oscar nomination graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph, URIRef

from foundation.loader import load_foundation_graph
from .canonical import MSH, _query_nominations, extract_category_features, load_graph
from .graph_enrichment import extract_graph_enrichment

_BEST_ACTOR_NAMES = frozenset({
    "ACTOR",
    "ACTOR IN A LEADING ROLE",
})


def _local_id(uri: str | None) -> str | None:
    if not uri:
        return None
    text = str(uri)
    if "#" in text:
        return text.split("#", 1)[1]
    return text


def _person_name(graph: Graph, nominee_uri: str | None) -> str | None:
    if not nominee_uri:
        return None
    full_name = graph.value(URIRef(nominee_uri), MSH.fullName)
    return str(full_name) if full_name is not None else _local_id(nominee_uri)


def _norm_name(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _prior_history(
    rows: pd.DataFrame,
    nominee_col: str,
    year_col: str,
    winner_col: str,
) -> dict[tuple[str, int], dict[str, int]]:
    """Build nominee-year lookup of prior nomination/win counts."""
    history: dict[tuple[str, int], dict[str, int]] = {}
    for nominee, grp in rows.groupby(nominee_col):
        years = sorted(grp[year_col].dropna().astype(int).unique())
        cum_noms = 0
        cum_wins = 0
        for yr in years:
            history[(str(nominee), int(yr))] = {
                "prev_noms": cum_noms,
                "prev_wins": cum_wins,
            }
            yr_rows = grp[grp[year_col] == yr]
            cum_noms += len(yr_rows)
            cum_wins += int(yr_rows[winner_col].sum())
    return history


def _extract_sag_actor_flags(actor_df: pd.DataFrame, sag_graph: Graph) -> pd.DataFrame:
    """Add SAG Best Actor nominee/winner flags matched by year + nominee name."""
    rows: list[dict[str, Any]] = []
    for nom in sag_graph.subjects(MSH.hasCeremony, None):
        ceremony = sag_graph.value(nom, MSH.hasCeremony)
        category = sag_graph.value(nom, MSH.hasCategory)
        nominee = sag_graph.value(nom, MSH.hasNominee)
        winner_obj = sag_graph.value(nom, MSH.winner)

        if not isinstance(ceremony, URIRef) or not isinstance(category, URIRef) or nominee is None:
            continue

        year_obj = sag_graph.value(ceremony, MSH.yearCeremony)
        cat_obj = sag_graph.value(category, MSH.categoryName)
        if year_obj is None or cat_obj is None:
            continue

        year_text = str(year_obj)
        if len(year_text) < 4 or not year_text[:4].isdigit():
            continue
        ceremony_year = int(year_text[:4])
        category_name = str(cat_obj)
        winner = str(winner_obj).strip().lower() == "true"

        rows.append(
            {
                "ceremony_year": ceremony_year,
                "category": category_name,
                "nominee": str(nominee),
                "winner": winner,
            }
        )

    sag_noms = pd.DataFrame(rows)
    if sag_noms.empty:
        out = actor_df.copy()
        out["sag_actor_nominee"] = 0
        out["sag_actor_winner"] = 0
        return out

    # Keep SAG male-actor-leading category rows only.
    cat = sag_noms["category"].str.upper().fillna("")
    mask = (
        cat.str.contains("ACTOR IN A LEADING ROLE")
        | (cat.str.contains("MALE ACTOR") & cat.str.contains("LEADING ROLE"))
    )
    sag_actor = sag_noms[mask & sag_noms["nominee"].notna()].copy()
    if sag_actor.empty:
        out = actor_df.copy()
        out["sag_actor_nominee"] = 0
        out["sag_actor_winner"] = 0
        return out

    sag_actor["candidate_name"] = sag_actor["nominee"].apply(lambda n: _person_name(sag_graph, n))
    sag_actor["candidate_key"] = sag_actor["candidate_name"].apply(_norm_name)
    sag_actor["year_key"] = sag_actor["ceremony_year"].astype(int) - 1  # SAG ceremony year -> Oscar film year

    nominee_keys = set(zip(sag_actor["year_key"], sag_actor["candidate_key"]))
    winner_rows = sag_actor[sag_actor["winner"] == True]
    winner_keys = set(zip(winner_rows["year_key"], winner_rows["candidate_key"]))

    out = actor_df.copy()
    out_key = out["candidate_name"].apply(_norm_name)
    out_year = out["year_film"].astype(int)
    out["sag_actor_nominee"] = [
        1 if (int(y), k) in nominee_keys else 0 for y, k in zip(out_year, out_key)
    ]
    out["sag_actor_winner"] = [
        1 if (int(y), k) in winner_keys else 0 for y, k in zip(out_year, out_key)
    ]
    return out


def extract_best_actor_features(
    graph: Graph,
    *,
    enrichment_ttl_paths: str | Path | list[str | Path] | None = None,
    sag_ttl_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build nominee-level feature table for Best Actor prediction."""
    all_noms = _query_nominations(graph)
    if all_noms.empty:
        return pd.DataFrame()

    actor_mask = all_noms["category"].str.upper().str.strip().isin(_BEST_ACTOR_NAMES)
    actor = all_noms[actor_mask & all_noms["nominee"].notna()].copy()
    if actor.empty:
        return pd.DataFrame()

    # Prior history across all categories and within best-actor categories.
    person_all = all_noms[all_noms["nominee"].notna()].copy()
    person_actor = actor.copy()

    all_hist = _prior_history(person_all, "nominee", "year_film", "winner")
    actor_hist = _prior_history(person_actor, "nominee", "year_film", "winner")

    actor["previous_all_nominations"] = actor.apply(
        lambda r: all_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_noms", 0),
        axis=1,
    )
    actor["previous_all_wins"] = actor.apply(
        lambda r: all_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_wins", 0),
        axis=1,
    )
    actor["previous_actor_nominations"] = actor.apply(
        lambda r: actor_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_noms", 0),
        axis=1,
    )
    actor["previous_actor_wins"] = actor.apply(
        lambda r: actor_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_wins", 0),
        axis=1,
    )

    # Merge film-level category context (nominations of associated film).
    film_cat = extract_category_features(all_noms)
    actor = actor.merge(film_cat, on=["film", "year_film"], how="left")

    actor["candidate_id"] = actor["nominee"].apply(_local_id)
    actor["candidate_name"] = actor["nominee"].apply(lambda n: _person_name(graph, n))

    out = actor[
        [
            "candidate_id",
            "candidate_name",
            "film_uri",
            "film",
            "year_film",
            "winner",
            "tmdb_id",
            "imdb_id",
            "nom",
            "previous_all_nominations",
            "previous_all_wins",
            "previous_actor_nominations",
            "previous_actor_wins",
            "acting_noms",
            "directing_nom",
            "writing_noms",
            "cinematography_nom",
            "editing_nom",
            "technical_noms",
            "music_noms",
            "above_line_noms",
            "other_noms",
        ]
    ].copy()

    # Optional SAG Best Actor precursor features.
    if sag_ttl_path:
        sag_path = Path(sag_ttl_path)
        if sag_path.exists():
            sag_graph = load_graph(sag_path)
            out = _extract_sag_actor_flags(out, sag_graph)

    parsed_overlay = False
    if enrichment_ttl_paths is not None:
        ttl_paths = [enrichment_ttl_paths] if isinstance(enrichment_ttl_paths, (str, Path)) else enrichment_ttl_paths
        for ttl in ttl_paths:
            ttl_path = Path(ttl)
            if ttl_path.exists():
                graph.parse(str(ttl_path), format="turtle")
                parsed_overlay = True

    if parsed_overlay:
        out = extract_graph_enrichment(graph, out)

    return out


class BestActorFeatureExtractor:
    """Feature extractor wrapper matching the existing FeatureExtractor style."""

    def __init__(
        self,
        *instance_paths: str | Path,
        enrichment_ttl_paths: str | Path | list[str | Path] | None = None,
        sag_ttl_path: str | Path | None = None,
    ):
        self._instance_paths = instance_paths
        self._enrichment_ttl_paths = enrichment_ttl_paths
        self._sag_ttl_path = sag_ttl_path
        self._result: pd.DataFrame | None = None

    def extract(self) -> pd.DataFrame:
        if self._instance_paths:
            graph = load_graph(*self._instance_paths)
        else:
            graph = load_foundation_graph(include_ontology=True)
        self._result = extract_best_actor_features(
            graph,
            enrichment_ttl_paths=self._enrichment_ttl_paths,
            sag_ttl_path=self._sag_ttl_path,
        )
        return self._result

    @property
    def features(self) -> pd.DataFrame:
        if self._result is None:
            return self.extract()
        return self._result

    def feature_columns(self) -> list[str]:
        df = self.features
        exclude = {"candidate_id", "candidate_name", "film_uri", "film", "year_film", "tmdb_id", "imdb_id", "nom"}
        return [c for c in df.columns if c not in exclude]
