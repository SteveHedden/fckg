"""Extract features for Best Actress prediction from Oscar nomination graphs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph, URIRef

from .utils import normalize_person_name

from .canonical import MSH, _query_nominations, extract_category_features, load_graph
from .graph_enrichment import extract_graph_enrichment

_BEST_ACTRESS_NAMES = frozenset({
    "ACTRESS",
    "BEST ACTRESS",
    "ACTRESS IN A LEADING ROLE",
    "PERFORMANCE BY AN ACTRESS IN A LEADING ROLE",
})


def _is_best_actress_category(category: str | None) -> bool:
    if not category:
        return False
    upper = str(category).upper().strip()
    if not upper:
        return False
    if "SUPPORTING" in upper:
        return False
    if upper in _BEST_ACTRESS_NAMES:
        return True
    if "ACTRESS" in upper and ("LEADING ROLE" in upper or "LEAD" in upper):
        return True
    if "FEMALE ACTOR" in upper and "LEADING ROLE" in upper:
        return True
    return False


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
    normalized = normalize_person_name(value or "")
    return "".join(ch.lower() for ch in str(normalized) if ch.isalnum())


def _prior_history(
    rows: pd.DataFrame,
    nominee_col: str,
    year_col: str,
    winner_col: str,
) -> dict[tuple[str, int], dict[str, int]]:
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


def _extract_sag_actress_flags(actress_df: pd.DataFrame, sag_graph: Graph) -> pd.DataFrame:
    """Add SAG Best Actress nominee/winner flags matched by year + nominee name."""
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

        rows.append(
            {
                "ceremony_year": int(year_text[:4]),
                "category": str(cat_obj),
                "nominee": str(nominee),
                "winner": str(winner_obj).strip().lower() == "true",
            }
        )

    sag_noms = pd.DataFrame(rows)
    if sag_noms.empty:
        out = actress_df.copy()
        out["sag_actress_nominee"] = 0
        out["sag_actress_winner"] = 0
        return out

    cat = sag_noms["category"].str.upper().fillna("").str.strip()
    mask = (
        (cat == "ACTRESS")
        | cat.str.contains("ACTRESS IN A LEADING ROLE")
        | (cat.str.contains("FEMALE ACTOR") & cat.str.contains("LEADING ROLE"))
    ) & ~cat.str.contains("SUPPORTING")

    sag_actress = sag_noms[mask & sag_noms["nominee"].notna()].copy()
    if sag_actress.empty:
        out = actress_df.copy()
        out["sag_actress_nominee"] = 0
        out["sag_actress_winner"] = 0
        return out

    sag_actress["candidate_name"] = sag_actress["nominee"].apply(lambda n: _person_name(sag_graph, n))
    sag_actress["candidate_key"] = sag_actress["candidate_name"].apply(_norm_name)
    sag_actress["year_key"] = sag_actress["ceremony_year"].astype(int) - 1

    oscar_keys_by_year: dict[int, set[str]] = {}
    for yr, grp in actress_df.groupby(actress_df["year_film"].astype(int)):
        oscar_keys_by_year[int(yr)] = {k for k in grp["candidate_name"].apply(_norm_name) if k}

    for _, row in sag_actress.iterrows():
        year_key = int(row["year_key"])
        cand_key = str(row["candidate_key"])
        if not cand_key:
            warnings.warn(
                f"Unable to normalize SAG nominee name for year {year_key + 1}; continuing",
                stacklevel=2,
            )
            continue
        expected = oscar_keys_by_year.get(year_key, set())
        if expected and cand_key not in expected:
            warnings.warn(
                f"SAG/Oscar actress name mismatch for film year {year_key}: {row['candidate_name']}",
                stacklevel=2,
            )

    nominee_keys = set(zip(sag_actress["year_key"], sag_actress["candidate_key"]))
    winner_rows = sag_actress[sag_actress["winner"] == True]
    winner_keys = set(zip(winner_rows["year_key"], winner_rows["candidate_key"]))

    out = actress_df.copy()
    out_key = out["candidate_name"].apply(_norm_name)
    out_year = out["year_film"].astype(int)
    out["sag_actress_nominee"] = [
        1 if (int(y), k) in nominee_keys else 0 for y, k in zip(out_year, out_key)
    ]
    out["sag_actress_winner"] = [
        1 if (int(y), k) in winner_keys else 0 for y, k in zip(out_year, out_key)
    ]
    return out


def extract_best_actress_features(
    graph: Graph,
    *,
    enrichment_ttl_paths: str | Path | list[str | Path] | None = None,
    sag_ttl_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build nominee-level feature table for Best Actress prediction."""
    all_noms = _query_nominations(graph)
    if all_noms.empty:
        return pd.DataFrame()

    actress_mask = all_noms["category"].apply(_is_best_actress_category)
    actress = all_noms[actress_mask & all_noms["nominee"].notna()].copy()
    if actress.empty:
        return pd.DataFrame()

    person_all = all_noms[all_noms["nominee"].notna()].copy()
    person_actress = actress.copy()

    all_hist = _prior_history(person_all, "nominee", "year_film", "winner")
    actress_hist = _prior_history(person_actress, "nominee", "year_film", "winner")

    actress["previous_all_nominations"] = actress.apply(
        lambda r: all_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_noms", 0),
        axis=1,
    )
    actress["previous_all_wins"] = actress.apply(
        lambda r: all_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_wins", 0),
        axis=1,
    )
    actress["previous_actress_nominations"] = actress.apply(
        lambda r: actress_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_noms", 0),
        axis=1,
    )
    actress["previous_actress_wins"] = actress.apply(
        lambda r: actress_hist.get((str(r["nominee"]), int(r["year_film"])), {}).get("prev_wins", 0),
        axis=1,
    )

    film_cat = extract_category_features(all_noms)
    actress = actress.merge(film_cat, on=["film", "year_film"], how="left")

    actress["candidate_id"] = actress["nominee"].apply(_local_id)
    actress["candidate_name"] = actress["nominee"].apply(lambda n: _person_name(graph, n))

    out = actress[
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
            "previous_actress_nominations",
            "previous_actress_wins",
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

    if sag_ttl_path:
        sag_path = Path(sag_ttl_path)
        if sag_path.exists():
            sag_graph = load_graph(sag_path)
            out = _extract_sag_actress_flags(out, sag_graph)

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


class BestActressFeatureExtractor:
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
        graph = load_graph(*self._instance_paths)
        self._result = extract_best_actress_features(
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

